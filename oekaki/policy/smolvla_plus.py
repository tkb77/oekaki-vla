"""SmolVLA extensions for smoother action chunks (used by scripts/train_smolvla.py and oekaki/eval/policy.py).

* Training-time RTC (Black et al. 2025, "Training-Time Action Conditioning for Efficient Real-Time
  Chunking", arXiv:2512.05964). The flow time becomes per action token. During training a random
  delay d is sampled, the first d tokens of the chunk are given as clean ground-truth actions (flow time
  0 = clean in LeRobot's convention), and the loss is computed only on the remaining tokens. At inference
  the actions of the previous chunk that are already committed are passed as that prefix. This needs no
  guidance and no extra backward pass.
* Hermite-VLA_Reg (Lv et al. 2026, "Hermite Curves as Trajectory Priors for VLA Models",
  arXiv:2608.01265). An auxiliary head predicts K cubic Hermite segments (endpoint positions and
  velocities) from the mean action-expert feature. It is supervised in trajectory space, as
  lambda * ||H theta - a||^2. The flow-matching head and sampler are unchanged, and the auxiliary head
  is discarded at inference.
* Progress input (ours). Elapsed episode time t (s), clipped at max_s and divided by scale, is written
  into the first zero-padded slot of SmolVLA's 32-D state vector. Without it the policy cannot tell
  "about to start the outline" from "outline just closed": the joint state and the camera views are
  almost the same at both moments. Closed-loop rollouts then keep tracing the shape past its end
  (about twice the expert's ink).

Extensions are described by a plain dict (``extras``) that is stored as ``oekaki_extras.json`` next to
the checkpoint:
    {"tt_rtc": {"max_delay": 6, "decay": 0.75}, "hermite": {"K": 2, "dims": 6, "lam": 10.0},
     "progress": {"fps": 30, "scale": 10.0, "max_s": 16.5}}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding, make_att_2d_masks
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

EXTRAS_FILE = "oekaki_extras.json"
PROGRESS_KEY = "oekaki.elapsed_s"  # (B,) elapsed episode time in seconds, set by the caller


# ----------------------------------------------------------------------------- Hermite basis
def hermite_basis(T: int, K: int) -> Tensor:
    """(T, 4K) matrix H with traj = H @ theta, theta = [p_s, v_s, p_e, v_e] per segment.

    Velocities are expressed per unit of the normalised segment parameter u, i.e. scaled by the segment
    span, as in Hermite-VLA's target encoding.
    """
    bounds = np.linspace(0, T - 1, K + 1)
    H = np.zeros((T, 4 * K))
    for tau in range(T):
        k = min(int(np.searchsorted(bounds, tau, side="right") - 1), K - 1)
        u = (tau - bounds[k]) / (bounds[k + 1] - bounds[k])
        H[tau, 4 * k:4 * k + 4] = [2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2]
    return torch.tensor(H, dtype=torch.float32)


class HermiteHead(nn.Module):
    def __init__(self, hidden: int, T: int, K: int, dims: int):
        super().__init__()
        self.K, self.dims = K, dims
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4 * K * dims))
        self.register_buffer("H", hermite_basis(T, K), persistent=False)

    def forward(self, feats: Tensor) -> Tensor:
        """feats (B, T, hidden) -> reconstructed trajectory (B, T, dims)."""
        theta = self.mlp(feats.float().mean(1)).float().view(-1, 4 * self.K, self.dims)
        return torch.einsum("tk,bkd->btd", self.H, theta)


# ----------------------------------------------------------------------------- model with per-token time
class VLAFlowMatchingPlus(VLAFlowMatching):
    """VLAFlowMatching whose flow time may be given per action token: (B,) or (B, T)."""

    def embed_suffix(self, noisy_actions, timestep):
        if timestep.ndim == 1:
            return super().embed_suffix(noisy_actions, timestep)
        action_emb = self.action_in_proj(noisy_actions)
        bsize, T = timestep.shape
        time_emb = create_sinusoidal_pos_embedding(
            timestep.reshape(-1), self.vlm_with_expert.expert_hidden_size, self.config.min_period,
            self.config.max_period, device=action_emb.device).view(bsize, T, -1).type(action_emb.dtype)
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2))))
        pad_masks = torch.ones(bsize, T, dtype=torch.bool, device=x.device)
        att_masks = torch.ones(bsize, T, dtype=x.dtype, device=x.device)
        return x, pad_masks, att_masks

    def forward_plus(self, images, img_masks, lang_tokens, lang_masks, state, actions, delay: Tensor | None):
        """Training forward. Returns per-element flow losses (B, T, A) and action-expert features (B, T, H)."""
        bsize, T = actions.shape[:2]
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(bsize, actions.device)
        if delay is not None and bool((delay > 0).any()):
            t_tok = time[:, None].expand(bsize, T).clone()
            t_tok[torch.arange(T, device=actions.device)[None] < delay[:, None]] = 0.0  # clean prefix
            te = t_tok[..., None]
        else:
            t_tok, te = time, time[:, None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, t_tok)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)
        suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none"), suffix_out

    @torch.no_grad()
    def sample_actions_prefix(self, images, img_masks, lang_tokens, lang_masks, state, prefix: Tensor | None,
                              noise: Tensor | None = None) -> Tensor:
        """Euler sampling where the first d = prefix.shape[1] tokens are fixed to `prefix` (flow time 0)."""
        bsize = state.shape[0]
        shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        x = self.sample_noise(shape, state.device) if noise is None else noise
        d = 0 if prefix is None else prefix.shape[1]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
            position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=self.config.use_cache)
        dt = -1.0 / self.config.num_steps
        for step in range(self.config.num_steps):
            t_tok = torch.full(shape[:2], 1.0 + step * dt, dtype=torch.float32, device=state.device)
            if d:
                x[:, :d] = prefix
                t_tok[:, :d] = 0.0
            v = self.denoise_step(x_t=x, prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                                  timestep=t_tok if d else t_tok[:, 0])
            x = x + dt * v
        if d:
            x[:, :d] = prefix
        return x


# ----------------------------------------------------------------------------- policy-level helpers
def upgrade_policy(policy, extras: dict | None):
    """Swap in the per-token-time model and (for training) attach the Hermite head."""
    policy.model.__class__ = VLAFlowMatchingPlus
    extras = extras or {}
    if "hermite" in extras and not hasattr(policy.model, "hermite_head"):
        h = extras["hermite"]
        head = HermiteHead(policy.model.vlm_with_expert.expert_hidden_size, policy.config.chunk_size,
                           h.get("K", 2), h.get("dims", policy.config.action_feature.shape[0]))
        policy.model.hermite_head = head.to(next(policy.parameters()).device)
    if "progress" in extras and not getattr(policy, "_oekaki_progress", False):
        _attach_progress(policy, extras["progress"])
    policy.oekaki_extras = extras
    return policy


def _attach_progress(policy, cfg: dict):
    slot = policy.config.input_features["observation.state"].shape[0]
    assert slot < policy.config.max_state_dim
    scale, max_s = cfg.get("scale", 10.0), cfg.get("max_s", 16.5)
    orig = policy.prepare_state

    def prepare_state(batch):
        state = orig(batch)
        if PROGRESS_KEY in batch:
            state = state.clone()
            state[:, slot] = batch[PROGRESS_KEY].to(state).reshape(-1).clamp(0, max_s) / scale
        return state

    policy.prepare_state = prepare_state
    policy._oekaki_progress = True


def save_extras(d: Path, extras: dict):
    json.dump(extras, open(Path(d) / EXTRAS_FILE, "w"), indent=2)


def load_extras(d: Path) -> dict:
    p = Path(d) / EXTRAS_FILE
    return json.load(open(p)) if p.exists() else {}


def sample_delays(bsize: int, max_delay: int, decay: float, device) -> Tensor:
    """d in {0..max_delay} with exponentially decreasing weights (as in training-time RTC's simulated setup)."""
    w = torch.tensor([decay ** i for i in range(max_delay + 1)])
    return torch.multinomial(w / w.sum(), bsize, replacement=True).to(device)


def plus_forward(policy, batch: dict, extras: dict) -> tuple[Tensor, dict]:
    """Training loss with the enabled extensions (flow loss on the postfix + lambda * Hermite loss)."""
    m = policy.model
    if "progress" in extras:
        batch[PROGRESS_KEY] = batch["frame_index"].float() / extras["progress"].get("fps", 30)
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    bsize, T = actions.shape[:2]
    valid = ~batch["action_is_pad"] if "action_is_pad" in batch else torch.ones(bsize, T, dtype=torch.bool,
                                                                                device=actions.device)
    delay = None
    if "tt_rtc" in extras:
        r = extras["tt_rtc"]
        delay = sample_delays(bsize, r.get("max_delay", 6), r.get("decay", 0.75), actions.device)
    losses, feats = m.forward_plus(images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                                   state, actions, delay)
    adim = policy.config.action_feature.shape[0]
    w = valid.clone()
    if delay is not None:
        w &= torch.arange(T, device=actions.device)[None] >= delay[:, None]
    w = w[..., None].float()
    flow = (losses[:, :, :adim] * w).sum() / (w.sum() * adim).clamp_min(1)
    loss, info = flow, {"flow": flow.item()}
    if "hermite" in extras:
        h = m.hermite_head
        traj = h(feats)
        vm = valid[..., None].float()
        herm = (((traj - actions[:, :, :h.dims]) ** 2) * vm).sum() / (vm.sum() * h.dims).clamp_min(1)
        loss = loss + extras["hermite"].get("lam", 10.0) * herm
        info["hermite"] = herm.item()
    return loss, info


@torch.no_grad()
def predict_chunk_prefix(policy, batch: dict, prefix: Tensor | None, noise: Tensor | None = None) -> Tensor:
    """Normalised (B, T, action_dim) chunk whose first prefix.shape[1] actions equal `prefix` (normalised)."""
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    if prefix is not None:
        if prefix.shape[1] == 0:
            prefix = None
        else:
            prefix = F.pad(prefix, (0, policy.config.max_action_dim - prefix.shape[-1]))
    x = policy.model.sample_actions_prefix(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                                           batch[OBS_LANGUAGE_ATTENTION_MASK], state, prefix, noise=noise)
    return x[:, :, :policy.config.action_feature.shape[0]]
