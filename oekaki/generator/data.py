"""Generator / classifier training images: Quick, Draw! drawings rendered by the common renderer (64 x 64,
centred, longer side = fill), cached as uint8 arrays."""
from __future__ import annotations


import numpy as np

from oekaki.config import load_config, resolve
from oekaki.layout import centred
from oekaki.quickdraw import load_split
from oekaki.render import render_strokes


def rendered_split(gcfg: dict, split: str, log=print) -> tuple[np.ndarray, np.ndarray, list[str]]:
    q = load_config(gcfg["quickdraw_config"])
    cats = q["categories"]
    cache = resolve(gcfg["cache_dir"]) / f"{split}_{gcfg['image_size']}_{gcfg['fill']}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["images"], z["labels"], cats
    sets = load_split(resolve(q["out_dir"]), cats, split)
    T = centred(gcfg["fill"])
    imgs, labels = [], []
    for li, c in enumerate(cats):
        s = sets[c]
        log(f"rendering {split}/{c}: {len(s)}")
        imgs.append(np.stack([render_strokes(s[i], gcfg["image_size"], transform=T) for i in range(len(s))]))
        labels.append(np.full(len(s), li, np.int64))
    imgs, labels = np.concatenate(imgs), np.concatenate(labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, images=imgs, labels=labels)
    return imgs, labels, cats
