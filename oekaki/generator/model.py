"""Class-conditional DDPM for 64 x 64 line drawings.

The U-Net, the cosine schedule and the ancestral sampler are adapted from `train_quickdraw_ddpm.py` of
https://huggingface.co/lcccluck/sketch-model (Apache-2.0; a copy is in third_party/sketch-model). Changes:
training images come from our common renderer, EMA weights, DDIM sampling, and the sampler can return the
intermediate images (for the demo animation).

Image convention inside the model: ink = +1, background = -1 (the renderer's white-background image is
converted by `to_model` / `from_model`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_model(img_u8: np.ndarray | torch.Tensor) -> torch.Tensor:
    """(N,H,W) uint8 white-background images -> (N,1,H,W) float in [-1, 1], ink = +1."""
    x = torch.as_tensor(img_u8).float()
    return (1.0 - x / 127.5).unsqueeze(1)


def from_model(x: torch.Tensor) -> np.ndarray:
    """(N,1,H,W) model images -> (N,H,W) uint8 white-background images."""
    return ((1.0 - x.clamp(-1, 1)) * 127.5).round().byte().squeeze(1).cpu().numpy()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallConditionalUNet(nn.Module):
    def __init__(self, num_classes: int, base_channels: int = 64, emb_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.null_label = num_classes
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbedding(emb_dim), nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)
        c = base_channels
        self.in_conv = nn.Conv2d(1, c, 3, padding=1)
        self.down1 = ResBlock(c, c, emb_dim)
        self.downsample1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
        self.down2 = ResBlock(c * 2, c * 2, emb_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)
        self.mid1 = ResBlock(c * 4, c * 4, emb_dim)
        self.mid2 = ResBlock(c * 4, c * 4, emb_dim)
        self.upsample2 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.up2 = ResBlock(c * 4, c * 2, emb_dim)
        self.upsample1 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.up1 = ResBlock(c * 2, c, emb_dim)
        self.out_norm = nn.GroupNorm(min(8, c), c)
        self.out_conv = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x, t, y):
        emb = self.time_mlp(t) + self.class_emb(y)
        x0 = self.in_conv(x)
        x1 = self.down1(x0, emb)
        x2 = self.down2(self.downsample1(x1), emb)
        x3 = self.mid2(self.mid1(self.downsample2(x2), emb), emb)
        x = self.up2(torch.cat([self.upsample2(x3), x2], dim=1), emb)
        x = self.up1(torch.cat([self.upsample1(x), x1], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(x)))


@dataclass
class Schedule:
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor

    def to(self, device):
        return Schedule(*(v.to(device) for v in (self.betas, self.alphas_cumprod, self.alphas_cumprod_prev)))


def make_schedule(timesteps: int) -> Schedule:
    """Cosine schedule (Nichol & Dhariwal), as in the reference script."""
    x = torch.linspace(0, timesteps, timesteps + 1)
    ac = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = (1.0 - ac[1:] / ac[:-1]).clamp(1e-4, 0.999)
    ac = torch.cumprod(1.0 - betas, 0)
    return Schedule(betas, ac, F.pad(ac[:-1], (1, 0), value=1.0))


def q_sample(x0, t, noise, sch: Schedule):
    a = sch.alphas_cumprod[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def _eps(model, x, t, labels, guidance: float):
    if guidance == 1.0:
        return model(x, t, labels)
    null = torch.full_like(labels, model.null_label)
    e = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([labels, null]))
    ec, eu = e.chunk(2)
    return eu + guidance * (ec - eu)


@torch.no_grad()
def sample_ddim(model, labels: torch.Tensor, size: int, sch: Schedule, steps: int = 50, guidance: float = 3.0,
                eta: float = 0.0, generator: torch.Generator | None = None, return_frames: bool = False):
    """DDIM sampling with classifier-free guidance. Returns x0 in [-1, 1] (N,1,H,W), and if return_frames the
    list of (x_t, predicted x0) after every step (for the generation animation)."""
    device = labels.device
    T = len(sch.betas)
    ts = torch.linspace(T - 1, 0, steps, device=device).round().long()
    x = torch.randn(len(labels), 1, size, size, device=device, generator=generator)
    frames = []
    for i, t in enumerate(ts):
        tt = t.repeat(len(labels))
        eps = _eps(model, x, tt, labels, guidance)
        a = sch.alphas_cumprod[t]
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()  # re-derive after clamping
        a_prev = sch.alphas_cumprod[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
        sigma = eta * ((1 - a_prev) / (1 - a) * (1 - a / a_prev)).clamp_min(0).sqrt()
        noise = torch.randn(x.shape, device=device, generator=generator) if eta > 0 else 0.0
        x = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp_min(0).sqrt() * eps + sigma * noise
        if return_frames:
            frames.append((x.detach().cpu(), x0.detach().cpu()))
    return (x.clamp(-1, 1), frames) if return_frames else x.clamp(-1, 1)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        import copy
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for e, p in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                e.copy_(p)


def load_generator(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = SmallConditionalUNet(len(ck["classes"]), ck["base_channels"]).to(device)
    m.load_state_dict(ck["ema"] if ck.get("ema") is not None else ck["model"])
    m.eval()
    return m, make_schedule(ck["timesteps"]).to(device), ck
