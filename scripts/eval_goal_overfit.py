"""Pre-check: does the overfitted model G draw what its goal image shows, where it shows it?

Replays the training episodes of the overfit run (same paper height, instruction, goal) and also runs each
episode with the goal of *another* training episode of the same category ("swapped"). If G reads the goal,
the drawing follows the swapped goal (high IoU with it, low IoU with the episode's own goal); if it memorised
the episodes from the other inputs it keeps drawing the original. Writes results/goal/precheck/.

Usage: python scripts/eval_goal_overfit.py --policy outputs/train/goal_overfit_G/checkpoints/last [--per-category 6]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json

import cv2
import numpy as np

from oekaki.config import load_config, resolve
from oekaki.data.goal import episode_index, first_episodes_per_category, goal_image, load_goal_config
from oekaki.env.drawing_env import DrawingEnv, EpisodeParams
from oekaki.eval.goal_metrics import goal_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="outputs/train/goal_overfit_G/checkpoints/last")
    ap.add_argument("--dataset", default="data/goal_v1/G")
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--max-seconds", type=float, default=40.0)
    ap.add_argument("--out", default="results/goal/precheck")
    args = ap.parse_args()
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    env_cfg, gcfg, icfg = load_config(), load_goal_config(), load_config("configs/eval_goal.yaml")
    metas = episode_index(resolve(args.dataset))
    eps = first_episodes_per_category(resolve(args.dataset), args.per_category)
    from oekaki.eval.policy import SmolVLAAgent
    inf = icfg["inference"]
    agent = SmolVLAAgent(resolve(args.policy), mode=inf["mode"], replan=inf["replan"], delay=inf["delay"],
                         noise_scale=inf["noise_scale"])
    env = DrawingEnv(env_cfg, render=True)
    rows, tiles = [], []
    for k, e in enumerate(eps):
        m = metas[e]
        same = [x for x in eps if metas[x]["category"] == m["category"] and x != e]
        swap = same[k % len(same)]
        own_goal = goal_image(m["strokes_norm"], gcfg["goal"]["size"])
        swap_goal = goal_image(metas[swap]["strokes_norm"], gcfg["goal"]["size"])
        for kind, goal in (("own", own_goal), ("swapped", swap_goal)):
            obs = env.reset(EpisodeParams(height_offset=m["paper_height_offset"], xy_offset=tuple(m["paper_xy_offset"])))
            agent.reset(m["task"], obs, goal=goal)
            for t in range(int(args.max_seconds * env.fps)):
                obs = env.step(agent.act(obs))
            canvas = env.canvas.img.copy()
            g_own, g_sw = goal_metrics(own_goal, canvas), goal_metrics(swap_goal, canvas)
            rows.append(dict(episode=e, category=m["category"], kind=kind, swap_with=swap,
                             iou_shown=(g_own if kind == "own" else g_sw)["dilated_iou"],
                             iou_other=(g_sw if kind == "own" else g_own)["dilated_iou"],
                             chamfer_shown_mm=(g_own if kind == "own" else g_sw)["chamfer_mm"],
                             offset_shown_mm=(g_own if kind == "own" else g_sw)["centroid_offset_mm"],
                             size_ratio_shown=(g_own if kind == "own" else g_sw)["size_ratio"]))
            print(rows[-1], flush=True)
            sm = lambda x: cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA)  # noqa: E731
            if kind == "own":
                row_imgs = [sm(own_goal), sm(canvas)]
            else:
                row_imgs += [sm(swap_goal), sm(canvas)]
                tiles.append(np.hstack([np.pad(x, ((2, 2), (2, 2), (0, 0)), constant_values=255) for x in row_imgs]))
    env.close()
    json.dump(rows, open(out / "overfit_eval.json", "w"), indent=1)
    head = np.full((22, tiles[0].shape[1], 3), 255, np.uint8)
    cv2.putText(head, "own goal | drawn            swapped goal | drawn", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1)
    cv2.imwrite(str(out / "overfit_sheet.png"), np.vstack([head] + tiles)[..., ::-1])
    import pandas as pd
    df = pd.DataFrame(rows)
    s = df.groupby("kind")[["iou_shown", "iou_other", "chamfer_shown_mm", "offset_shown_mm", "size_ratio_shown"]].mean()
    print(s.round(3).to_string())
    s.round(3).to_markdown(open(out / "overfit_summary.md", "w"))


if __name__ == "__main__":
    main()
