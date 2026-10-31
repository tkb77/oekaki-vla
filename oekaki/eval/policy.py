"""Policy wrappers with a common `reset(task, obs)` / `act(obs)` interface for rollouts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from oekaki.data.rollout import state_vector


class SmolVLAAgent:
    def __init__(self, ckpt: str | Path, device: str = "cuda", n_action_steps: int | None = None,
                 num_steps: int | None = None, ensemble_every: int = 0, ensemble_m: float = 0.1,
                 mode: str | None = None, replan: int = 5, delay: int = 0, rtc_horizon: int = 20,
                 rtc_beta: float = 5.0, noise_scale: float = 1.0):
        """Inference schemes
        * ensemble_every > 0 (mode "ensemble"): predict a full chunk every `ensemble_every` steps and execute the
          exponentially weighted average of all overlapping chunk predictions (ACT temporal ensembling).
        * mode "chunk" / "rtc" / "ttrtc": a new chunk every `replan` steps. The first `delay` actions after a
          replan are the ones the previous chunk had already committed (they would run while the model
          is inferring on a real robot, i.e. a latency of `delay` control steps). The new chunk is executed after them:
            chunk: plain (naive asynchronous) chunking;
            rtc:   inference-time RTC (guided inpainting toward the previous chunk, soft mask over
                   `rtc_horizon` steps, guidance clipped at `rtc_beta`);
            ttrtc: training-time RTC (the committed actions are fed as a clean prefix).
        * mode None with ensemble_every == 0: LeRobot select_action (n_action_steps open-loop chunking).
        noise_scale: std of the initial flow-matching noise (sampling temperature; 1 = as trained). Lower values
        remove most of the high-frequency wobble inside a sampled chunk."""
        self.noise_scale = noise_scale
        self.ensemble_every = ensemble_every
        self.ensemble_m = ensemble_m
        self.mode = "ensemble" if ensemble_every > 0 else mode
        self.replan, self.delay, self.rtc_horizon = replan, delay, rtc_horizon
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(str(ckpt))
        if self.mode == "rtc":
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.configs import RTCAttentionSchedule
            self.policy.config.rtc_config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.EXP,
                                                      max_guidance_weight=rtc_beta, execution_horizon=rtc_horizon)
            self.policy.init_rtc_processor()
        from oekaki.policy.smolvla_plus import PROGRESS_KEY, load_extras, upgrade_policy
        extras = load_extras(ckpt)
        self.progress = extras.get("progress")
        self.progress_key = PROGRESS_KEY
        if self.mode == "ttrtc" or self.progress:
            upgrade_policy(self.policy, {"progress": self.progress} if self.progress else None)
        if n_action_steps is not None:
            self.policy.config.n_action_steps = n_action_steps
        if num_steps is not None:
            self.policy.config.num_steps = num_steps
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.device = device
        self.state_dim = self.policy.config.input_features["observation.state"].shape[0]
        self.with_contact = self.state_dim == 8
        self.image_keys = list(self.policy.config.image_features)  # Model G also has ...images.goal
        self.task = ""
        self.goal = None

    def reset(self, task: str, obs: dict | None = None, goal: np.ndarray | None = None):
        """goal: (H, W, 3) uint8 goal image; ignored by models without the goal camera."""
        self.task = task
        self.goal = goal
        self.policy.reset()
        self.t = 0
        self.chunks = []  # (start_step, (T, D) unnormalised actions)
        self.cur = None   # chunk mode: (start_step, normalised (T, D) tensor, unnormalised (T, D) array)
        self.seams = []   # steps at which a newly predicted chunk starts to be executed

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        def img(x):
            return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        batch = {
            "observation.state": torch.from_numpy(state_vector(obs, self.with_contact)).unsqueeze(0),
            "task": [self.task],
        }
        for k in self.image_keys:
            cam = k.rsplit(".", 1)[-1]
            batch[k] = img(self.goal if cam == "goal" else obs["images"][cam])
        if self.mode == "ensemble":
            return self._act_ensemble(batch)
        if self.mode in ("chunk", "rtc", "ttrtc"):
            return self._act_chunk(batch)
        batch = self._pre(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = self.policy.select_action(batch)
        action = self.post(action)
        self.t += 1
        return action.float().cpu().numpy().reshape(-1)[:6]

    def _pre(self, batch):
        b = self.pre(batch)
        if self.progress:
            b[self.progress_key] = torch.tensor([self.t / self.progress.get("fps", 30)], device=self.device)
        return b

    def _noise(self):
        c = self.policy.config
        return torch.randn(1, c.chunk_size, c.max_action_dim, device=self.device) * self.noise_scale

    def _act_chunk(self, batch) -> np.ndarray:
        if self.cur is None or self.t - self.cur[0] >= self.replan:
            b = self._pre(batch)
            prev = None
            d = 0
            if self.cur is not None:
                k = self.t - self.cur[0]
                prev = self.cur[1][:, k:]  # (1, T-k, D) not yet executed, normalised
                d = min(self.delay, prev.shape[1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                noise = self._noise()
                if self.mode == "ttrtc":
                    from oekaki.policy.smolvla_plus import predict_chunk_prefix
                    new = predict_chunk_prefix(self.policy, b, prev[:, :d] if prev is not None else None, noise=noise)
                elif self.mode == "rtc" and prev is not None:
                    new = self.policy.predict_action_chunk(b, noise=noise, inference_delay=d, prev_chunk_left_over=prev,
                                                           execution_horizon=self.rtc_horizon)
                else:
                    new = self.policy.predict_action_chunk(b, noise=noise)
            new = new.float()
            if d:  # actions committed before the new chunk is available
                new = torch.cat([prev[:, :d], new[:, d:]], 1)
            if prev is not None:
                self.seams.append(self.t + d)
            self.cur = (self.t, new, self.post(new.clone()).float().cpu().numpy()[0, :, :6])
        a = self.cur[2][self.t - self.cur[0]]
        self.t += 1
        return a

    def _act_ensemble(self, batch) -> np.ndarray:
        if self.t % self.ensemble_every == 0:
            b = self._pre(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = self.policy.predict_action_chunk(b, noise=self._noise())  # (1, T, D) normalised
            chunk = self.post(chunk).float().cpu().numpy()[0, :, :6]
            if self.chunks:
                self.seams.append(self.t)
            self.chunks.append((self.t, chunk))
        preds, ages = [], []
        for start, ch in self.chunks:
            k = self.t - start
            if k < len(ch):
                preds.append(ch[k])
                ages.append(start)
        self.chunks = [(st, ch) for st, ch in self.chunks if self.t - st < len(ch)]
        w = np.exp(-self.ensemble_m * np.arange(len(preds)))  # oldest prediction gets the largest weight (ACT)
        a = (np.stack(preds) * w[:, None]).sum(0) / w.sum()
        self.t += 1
        return a


class ExpertAgent:
    """The scripted expert (needs a sampled shape) behind the same interface (reference upper bound)."""

    def __init__(self, env, dcfg: dict):
        from oekaki.expert.expert import DrawingExpert
        self.env = env
        self.expert = DrawingExpert(env, dcfg["expert"])
        self.dcfg = dcfg

    def reset(self, task: str, obs: dict, strokes_world=None):
        self.expert.reset(strokes_world, obs)

    def act(self, obs: dict) -> np.ndarray:
        return self.expert.act(obs)
