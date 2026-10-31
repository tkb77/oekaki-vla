"""Instruction -> generated goal drawing."""
from __future__ import annotations

import torch

from oekaki.generator.model import from_model, load_generator, sample_ddim
from oekaki.generator.postprocess import postprocess, to_unit


def parse_category(text: str, classes: list[str]) -> str | None:
    """Category named in the instruction (simple string match, longest name first); None if none is named."""
    t = text.lower()
    for c in sorted(classes, key=len, reverse=True):
        if c in t:
            return c
    return None


class Imaginer:
    def __init__(self, ckpt, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.sch, self.ck = load_generator(ckpt, self.device)
        self.classes = list(self.ck["classes"])
        self.size = int(self.ck["image_size"])
        self.scfg, self.pcfg = self.ck["sample"], self.ck["postprocess"]

    def generate(self, category: str, seed: int, frames: bool = False, guidance: float | None = None) -> dict:
        """Returns image (64x64 uint8, white background), unit-coordinate skeleton segments, and optionally the
        DDIM trajectory (x_t and predicted x0 per step, uint8) for the demo animation."""
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        lab = torch.tensor([self.classes.index(category)], device=self.device)
        out = sample_ddim(self.model, lab, self.size, self.sch, self.scfg["ddim_steps"],
                          guidance if guidance is not None else self.scfg["guidance"], self.scfg["eta"],
                          generator=g, return_frames=frames)
        x, traj = out if frames else (out, None)
        img = from_model(x)[0]
        segs = postprocess(img, self.pcfg["thresh"], self.pcfg["min_area"])
        res = dict(image=img, segments=segs, unit=to_unit(segs) if segs else [], category=category, seed=int(seed))
        if frames:
            res["trajectory"] = [(from_model(xt)[0], from_model(x0)[0]) for xt, x0 in traj]
        return res
