"""the expert draws laid-out Quick, Draw! drawings (all categories, dots included) under random
paper heights (and disturbances). Saves goal-vs-canvas sheets and per-episode metrics to results/goal/expert.

Usage: python scripts/validate_goal_expert.py --n 20 [--disturb-prob 0.25]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json

import cv2
import numpy as np

from oekaki.config import load_config, resolve
from oekaki.data.rollout import run_expert_episode
from oekaki.data.goal import build_episode, load_sets, load_goal_config, pick_sketches
from oekaki.env.drawing_env import DrawingEnv
from oekaki.eval.goal_metrics import goal_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="episodes per category")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="results/goal/expert")
    args = ap.parse_args()
    cfg, lcfg, env_cfg = load_goal_config(), load_config("configs/layout.yaml"), load_config()
    S = env_cfg["canvas"]["size"]
    sets = load_sets(args.split)
    jobs = pick_sketches(sets, cfg, lcfg, S, args.n, args.seed)
    env = DrawingEnv(env_cfg, render=False)
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], {c: [] for c in cfg["categories"]}
    for j in jobs:
        ep = build_episode(j, sets, cfg, lcfg, S)
        r = run_expert_episode(env, cfg, j["category"], np.random.default_rng(ep["rng_seed"]), record=False,
                               shape=ep["shape"], task=ep["task"])
        m = r["meta"]
        gm = goal_metrics(ep["goal"], r["canvas"])
        rows.append(dict(category=j["category"], key_id=j["key_id"], success=m["success"], n_steps=m["n_steps"],
                         coverage=m["coverage"], precision=m["precision"], air_ratio=m["air_ratio"],
                         height=m["paper_height_offset"], disturb=m["disturbance"] is not None,
                         n_strokes=len(ep["strokes_local"]), size_frac=ep["layout"]["size_frac"], **gm))
        g = cv2.resize(ep["goal"], (128, 128), interpolation=cv2.INTER_AREA)
        c = cv2.resize(r["canvas"], (128, 128), interpolation=cv2.INTER_AREA)
        tiles[j["category"]].append(np.hstack([g, np.full((128, 2, 3), 160, np.uint8), c[..., ::-1]]))
        print(f"{j['category']:12s} ok={m['success']} steps={m['n_steps']} cov={m['coverage']:.2f} "
              f"iou={gm['dilated_iou']:.2f} chamfer={gm['chamfer_px']:.2f}", flush=True)
    env.close()
    json.dump(rows, open(out / "expert.json", "w"), indent=1)
    sheet = []
    for cat, ts in tiles.items():
        ts = ts[:8]
        row = np.hstack([cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in ts])
        lab = np.full((20, row.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, f"{cat}: goal | expert canvas", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        sheet += [lab, row]
    cv2.imwrite(str(out / "expert_sheet.png"), np.vstack(sheet))
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby("category")[["success", "coverage", "precision", "dilated_iou", "chamfer_px", "air_ratio",
                                   "n_steps"]].mean()
    print(summ.round(3).to_string())
    summ.round(3).to_markdown(open(out / "expert_summary.md", "w"))


if __name__ == "__main__":
    main()
