"""Train the judge CNN. Training images: the generator's rendered
Quick, Draw! train split with random scale / shift / rotation / mirror (re-rendered from strokes, so the line
width stays the standard one). Test: the held-out Quick, Draw! test split, both centred and randomly laid out.

Usage: python scripts/train_classifier.py [--config configs/classifier.yaml] [--epochs N] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

from oekaki.classifier.model import JudgeCNN, to_input
from oekaki.config import load_config, resolve
from oekaki.layout import affine
from oekaki.quickdraw import load_split
from oekaki.render import render_strokes


class Augmented(torch.utils.data.Dataset):
    """Renders each drawing with a random similarity transform every time it is read."""

    def __init__(self, sets: dict, cats: list, cfg: dict, seed: int, n_per_class: int | None = None):
        self.items = [(li, s, i) for li, c in enumerate(cats) for s in [sets[c]]
                      for i in range(min(len(s), n_per_class or len(s)))]
        self.cfg, self.seed = cfg, seed
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        li, s, i = self.items[k]
        rng = np.random.default_rng((self.seed, self.epoch, k))
        return render_random(s[i], rng, self.cfg), li


def render_random(strokes, rng, cfg) -> np.ndarray:
    a, size = cfg["augment"], cfg["image_size"]
    scale = 0.8 * rng.uniform(*a["scale"])
    T = affine(scale, rng.uniform(-a["rot_deg"], a["rot_deg"]), bool(rng.random() < a["flip_prob"]), (0.5, 0.5))
    p = np.concatenate(strokes) @ T[:, :2].T + T[:, 2]
    lo, hi = p.min(0), p.max(0)
    room_lo, room_hi = 0.02 - lo, 0.98 - hi  # keep a small border
    shift = np.array([rng.uniform(min(room_lo[j], 0), max(room_hi[j], 0)) for j in range(2)]) * a["max_shift"]
    T[:, 2] += shift
    return render_strokes(strokes, size, transform=T)


def collate(batch):
    imgs = np.stack([b[0] for b in batch])
    return to_input(imgs), torch.tensor([b[1] for b in batch])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = c = 0
    for x, y in loader:
        p = model(x.to(device)).argmax(-1).cpu()
        c += int((p == y).sum()); n += len(y)
    model.train()
    return c / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classifier.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-per-class", type=int, default=None, help="smoke tests: use fewer drawings")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["train"]
    if args.epochs:
        tc["epochs"] = args.epochs
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = resolve(args.output_dir or cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    q = load_config(load_config(cfg["generator_config"])["quickdraw_config"])
    cats = q["categories"]
    tr = load_split(resolve(q["out_dir"]), cats, "train")
    te = load_split(resolve(q["out_dir"]), cats, "test")
    lim = args.limit_per_class
    train_ds = Augmented(tr, cats, cfg, seed, lim)
    test_rand = Augmented(te, cats, cfg, seed + 1, lim or 2000)
    cfg_centred = dict(cfg, augment=dict(cfg["augment"], scale=[1.0, 1.0], rot_deg=0.0, flip_prob=0.0, max_shift=0.0))
    test_cent = Augmented(te, cats, cfg_centred, seed + 2, lim or 2000)
    kw = dict(batch_size=tc["batch_size"], num_workers=tc["num_workers"], collate_fn=collate)
    g = torch.Generator(); g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(train_ds, shuffle=True, drop_last=True, generator=g, **kw)
    model = JudgeCNN(len(cats)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=max(total, 1))
    hist = []
    for ep in range(tc["epochs"]):
        train_ds.epoch = ep
        losses = []
        for x, y in dl:
            loss = F.cross_entropy(model(x.to(device)), y.to(device), label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        acc_r = evaluate(model, torch.utils.data.DataLoader(test_rand, **kw), device)
        acc_c = evaluate(model, torch.utils.data.DataLoader(test_cent, **kw), device)
        hist.append(dict(epoch=ep + 1, loss=float(np.mean(losses)), test_acc_random_layout=acc_r,
                         test_acc_centred=acc_c))
        print(hist[-1], flush=True)
        torch.save(dict(model=model.state_dict(), classes=cats, image_size=cfg["image_size"], history=hist),
                   out / "judge.pt")
    json.dump(hist, open(out / "history.json", "w"), indent=1)
    print("saved", out / "judge.pt")


if __name__ == "__main__":
    main()
