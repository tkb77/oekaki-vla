"""Quick, Draw! simplified drawings: download, filter, split and store as normalised strokes.

A drawing is stored in *unit drawing coordinates*: normalised image coordinates (u right, v down) in which the
bounding box is centred at (0.5, 0.5) and its longer side is 1. `oekaki.layout` places such a drawing on the
paper; the generator renders it centred with a margin.

Storage (one .npz per category and split): points (P,2) float32, stroke_ptr (S+1,), drawing_ptr (D+1,) into
stroke_ptr, key_id (D,) int64.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def file_name(word: str) -> str:
    return word.replace(" ", "_")


def download(word: str, url_tpl: str, raw_dir: Path) -> Path:
    import urllib.parse
    import urllib.request
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{file_name(word)}.ndjson"
    if not out.exists():
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(url_tpl.format(word=urllib.parse.quote(word)), tmp)
        tmp.rename(out)
    return out


def normalise(drawing: list) -> list[np.ndarray]:
    """Quick, Draw! [[xs], [ys]] strokes (0..255, y down) -> unit drawing coordinates."""
    strokes = [np.stack([np.asarray(s[0], np.float64), np.asarray(s[1], np.float64)], 1) for s in drawing]
    allp = np.concatenate(strokes)
    lo, hi = allp.min(0), allp.max(0)
    ext = max(float((hi - lo).max()), 1e-6)
    c = (lo + hi) / 2
    return [(s - c) / ext + 0.5 for s in strokes]


def keep(item: dict, cfg: dict) -> bool:
    d = item["drawing"]
    if cfg["recognized_only"] and not item.get("recognized", False):
        return False
    if len(d) > cfg["max_strokes"] or sum(len(s[0]) for s in d) > cfg["max_points"]:
        return False
    xs = np.concatenate([s[0] for s in d]); ys = np.concatenate([s[1] for s in d])
    return max(np.ptp(xs), np.ptp(ys)) >= cfg["min_extent"]


def pack(drawings: list[list[np.ndarray]], key_ids: list[int]) -> dict:
    pts, sp, dp = [], [0], [0]
    for d in drawings:
        for s in d:
            pts.append(s.astype(np.float32)); sp.append(sp[-1] + len(s))
        dp.append(len(sp) - 1)
    return dict(points=np.concatenate(pts) if pts else np.zeros((0, 2), np.float32),
                stroke_ptr=np.asarray(sp, np.int64), drawing_ptr=np.asarray(dp, np.int64),
                key_id=np.asarray(key_ids, np.int64))


class SketchSet:
    """Read-only access to one packed category/split."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.points, self.sp, self.dp, self.key_id = z["points"], z["stroke_ptr"], z["drawing_ptr"], z["key_id"]

    def __len__(self):
        return len(self.dp) - 1

    def __getitem__(self, i: int) -> list[np.ndarray]:
        return [self.points[self.sp[k]:self.sp[k + 1]].astype(np.float64) for k in range(self.dp[i], self.dp[i + 1])]


def processed_path(out_dir: Path, word: str, split: str) -> Path:
    return out_dir / f"{file_name(word)}_{split}.npz"


def load_split(out_dir: Path, categories: list[str], split: str) -> dict[str, SketchSet]:
    return {w: SketchSet(processed_path(out_dir, w, split)) for w in categories}


def preprocess(word: str, cfg: dict, raw: Path, out_dir: Path) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    buf = {"train": ([], []), "test": ([], [])}
    n_in = 0
    with open(raw) as f:
        for line in f:
            n_in += 1
            item = json.loads(line)
            if not keep(item, cfg):
                continue
            kid = int(item["key_id"])
            split = "test" if kid % cfg["test_mod"] == 0 else "train"
            buf[split][0].append(normalise(item["drawing"])); buf[split][1].append(kid)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = dict(word=word, read=n_in)
    for split, cap in (("train", cfg["max_train_per_class"]), ("test", cfg["max_test_per_class"])):
        d, k = buf[split]
        sel = np.sort(rng.permutation(len(d))[:cap])
        np.savez_compressed(processed_path(out_dir, word, split), **pack([d[i] for i in sel], [k[i] for i in sel]))
        stats[f"{split}_kept"] = len(d); stats[split] = int(len(sel))
    return stats
