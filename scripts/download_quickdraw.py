"""Download and preprocess Quick, Draw! (CC BY 4.0) for the drawing categories, and save a contact sheet of
drawings rendered by the common renderer.

Usage: python scripts/download_quickdraw.py [--config configs/quickdraw.yaml]
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from oekaki.config import load_config, resolve
from oekaki.quickdraw import download, load_split, preprocess
from oekaki.render import render_strokes


def contact_sheet(sets: dict, n: int, size: int, fill: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for w, s in sets.items():
        tiles = []
        for i in rng.choice(len(s), n, replace=False):
            t = np.array([[fill, 0, 0.5 - fill / 2], [0, fill, 0.5 - fill / 2]])
            im = render_strokes(s[int(i)], size, transform=t)
            tiles.append(cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=180))
        lab = np.full((18, (size + 2) * n), 255, np.uint8)
        cv2.putText(lab, w, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1, cv2.LINE_AA)
        rows += [lab, np.hstack(tiles)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/quickdraw.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw_dir, out_dir = resolve(cfg["raw_dir"]), resolve(cfg["out_dir"])
    stats = []
    for w in cfg["categories"]:
        raw = download(w, cfg["url"], raw_dir)
        st = preprocess(w, cfg, raw, out_dir)
        print(st, flush=True)
        stats.append(st)
    res = resolve("results/goal/quickdraw"); res.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(res / "quickdraw_stats.json", "w"), indent=1)
    for split in ("train", "test"):
        sets = load_split(out_dir, cfg["categories"], split)
        cv2.imwrite(str(res / f"quickdraw_{split}_64.png"), contact_sheet(sets, 16, 64, 0.8, cfg["seed"]))
        cv2.imwrite(str(res / f"quickdraw_{split}_128.png"), contact_sheet(sets, 10, 128, 0.8, cfg["seed"] + 1))
    print("wrote", res)


if __name__ == "__main__":
    main()
