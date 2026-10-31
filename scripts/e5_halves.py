"""E5 (two drawings) per-half check: split goal and canvas into left / right halves, classify each half with the
judge CNN, and count episodes where the drawn half has the goal half's class. The goal halves' classes are the
reference (the judge reads clean goals almost perfectly). Writes results/goal/eval/e5_halves.md.

Usage: python scripts/e5_halves.py [--agents G G_2x G_pair] [--eval outputs/eval_goal]
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from oekaki.classifier.model import Judge
from oekaki.config import resolve
from oekaki.render import ink


def halves(img: np.ndarray, canvas: bool):
    """Left / right halves; a half without ink is None."""
    h, w = img.shape[:2]
    out = []
    for part in (img[:, : w // 2], img[:, w // 2:]):
        m = ink(part, 200 if canvas else 128)
        out.append(part if m.sum() > 20 else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["G", "G_2x", "G_pair"])
    ap.add_argument("--eval", default="outputs/eval_goal")
    ap.add_argument("--judge", default="outputs/classifier/judge.pt")
    args = ap.parse_args()
    ev = resolve(args.eval) / "E5"
    judge = Judge(resolve(args.judge))
    rows = []
    for f in sorted((ev / "goal").glob("*.png")):
        goal = cv2.imread(str(f))[..., ::-1]
        gt = [None if h is None else judge.classes[int(judge.classify_drawing(h).argmax())] for h in halves(goal, False)]
        for a in args.agents:
            p = ev / a / f.name
            if not p.exists():
                continue
            cv = cv2.imread(str(p))[..., ::-1]
            pred = [None if h is None else judge.classes[int(judge.classify_drawing(h, canvas=True).argmax())]
                    for h in halves(cv, True)]
            ok = [g is not None and g == q for g, q in zip(gt, pred)]
            rows.append(dict(agent=a, episode=f.stem, both_drawn=all(q is not None for q in pred),
                             halves_correct=sum(ok), both_correct=all(ok)))
    df = pd.DataFrame(rows)
    s = df.groupby("agent").agg(n=("episode", "size"), both_drawn=("both_drawn", "mean"),
                                halves_correct=("halves_correct", lambda x: x.mean() / 2),
                                both_correct=("both_correct", "mean")).round(3)
    s = s.loc[[a for a in args.agents if a in s.index]]
    out = resolve("results/goal/eval/e5_halves.md")
    with open(out, "w") as fh:
        fh.write("# E5 per-half check (judge CNN on the left / right halves)\n\n"
                 "both_drawn: ink in both halves; halves_correct: fraction of halves whose class matches the goal "
                 "half; both_correct: both halves match.\n\n")
        fh.write(s.to_markdown() + "\n")
    print(s.to_string())


if __name__ == "__main__":
    main()
