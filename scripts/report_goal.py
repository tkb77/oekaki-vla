"""Goal-image report: tables with bootstrap 95 % confidence intervals, bar charts, goal-vs-drawing
grids and a failure breakdown from outputs/eval_goal/<setting>/results_<agent>.jsonl -> results/goal/eval/.

Failure classes (first match): nothing drawn; position off (centroid > 12 mm from the goal's); size off (drawn
extent < 0.6x or > 1.6x the goal's); dots missing (smiley face: fewer separate marks than the goal); shape broken
(dilated IoU < 0.5); otherwise OK.

Usage: python scripts/report_goal.py [--eval outputs/eval_goal] [--out results/goal/eval]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oekaki.config import resolve
from oekaki.render import ink

METRICS = [("dilated_iou", "goal IoU (dilated) ↑"), ("chamfer_mm", "Chamfer to goal (mm) ↓"),
           ("coverage_goal", "goal covered (1 - break rate) ↑"), ("judge_correct", "judge-CNN class accuracy ↑"),
           ("centroid_offset_mm", "position error (mm) ↓"), ("path_air_ratio", "air ratio of the pen path ↓")]
COLORS = {"N": "#7d7d78", "G": "#2a78d6", "expert": "#1baf7a"}
SETTING_TXT = {"E1": "E1 true goal", "E2": "E2 generated goal", "E3": "E3 fixed layouts",
               "E4": "E4 no category name", "E5": "E5 two drawings"}
TICK_TXT = {"E1": "E1\ntrue goal", "E2": "E2\ngenerated", "E3": "E3\nfixed layout", "E4": "E4\nunnamed",
            "E5": "E5\ntwo drawings"}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def n_marks(path: Path, canvas: bool) -> int:
    im = cv2.imread(str(path))
    if im is None:
        return 0
    m = ink(im, 200 if canvas else 128).astype(np.uint8)
    m = cv2.resize(m, (128, 128), interpolation=cv2.INTER_AREA) > 0
    return cv2.connectedComponents(m.astype(np.uint8), 8)[0] - 1


def failure(r, ev: Path) -> str:
    if not r.get("drawn", True) or r.get("ink_px", 1) < 50:
        return "nothing drawn"
    if r["centroid_offset_mm"] > 12:
        return "position off"
    if r["size_ratio"] < 0.6 or r["size_ratio"] > 1.6:
        return "size off"
    if r["category"] == "smiley face":
        key = f"{r['category'].replace(' ', '_')}_{r['i']:03d}.png"
        if n_marks(ev / r["setting"] / r["agent"] / key, True) < n_marks(ev / r["setting"] / "goal" / key, False):
            return "dots missing"
    if r["dilated_iou"] < 0.5:
        return "shape broken"
    return "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="outputs/eval_goal")
    ap.add_argument("--out", default="results/goal/eval")
    args = ap.parse_args()
    ev, out = resolve(args.eval), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for f in sorted(ev.glob("E*/results_*.jsonl")) for l in open(f)]
    df = pd.DataFrame(rows)
    if "gen_failed" in df:
        n_gen_fail = df.groupby("setting")["gen_failed"].apply(lambda s: int(s.fillna(False).sum())).to_dict()
        df = df[df.gen_failed.fillna(False) == False]  # noqa: E712
    else:
        n_gen_fail = {}
    if "judge_correct" in df:
        df["judge_correct"] = df["judge_correct"].astype(float)
    df["failure"] = [failure(r, ev) for r in df.to_dict("records")]
    df.to_csv(out / "episodes.csv", index=False)

    # ---- summary with bootstrap CIs ----
    summ = []
    for (s, a), g in df.groupby(["setting", "agent"]):
        row = dict(setting=s, agent=a, n=len(g))
        for m, _ in METRICS:
            if m in g:
                mu, lo, hi = boot_ci(g[m])
                row[m] = mu; row[f"{m}_lo"] = lo; row[f"{m}_hi"] = hi
        row["ok_rate"] = float((g.failure == "OK").mean())
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    with open(out / "summary.md", "w") as f:
        f.write("| setting | agent | n | " + " | ".join(t for m, t in METRICS if m in summ) + " | OK rate |\n")
        f.write("|---|---|---|" + "---|" * (sum(m in summ for m, _ in METRICS) + 1) + "\n")
        for r in summ.to_dict("records"):
            cells = [f"{r[m]:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]" for m, _ in METRICS if m in summ]
            f.write(f"| {r['setting']} | {r['agent']} | {r['n']} | " + " | ".join(cells) + f" | {r['ok_rate']:.2f} |\n")
        if n_gen_fail:
            f.write(f"\nGenerator produced an empty image (episode skipped): {n_gen_fail}\n")
    per_cat = df.groupby(["setting", "agent", "category"])[[m for m, _ in METRICS if m in df]].mean().round(3)
    per_cat.to_markdown(open(out / "per_category.md", "w"))
    fails = pd.crosstab([df.setting, df.agent], df.failure)
    fails.to_markdown(open(out / "failures.md", "w"))

    # ---- bar charts: settings x agents, mean with 95 % CI ----
    settings = sorted(df.setting.unique())
    agents = [a for a in ["N", "G", "expert"] if a in set(df.agent)] + sorted(set(df.agent) - {"N", "G", "expert"})
    mets = [(m, t) for m, t in METRICS if m in summ]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
    w = 0.8 / len(agents)
    for ax, (m, t) in zip(axs.flat, mets):
        for j, a in enumerate(agents):
            xs, ys, err = [], [], [[], []]
            for i, s in enumerate(settings):
                r = summ[(summ.setting == s) & (summ.agent == a)]
                if len(r) and not np.isnan(r[m].iloc[0]):
                    xs.append(i + (j - (len(agents) - 1) / 2) * w); ys.append(r[m].iloc[0])
                    err[0].append(r[m].iloc[0] - r[f"{m}_lo"].iloc[0]); err[1].append(r[f"{m}_hi"].iloc[0] - r[m].iloc[0])
            ax.bar(xs, ys, w * 0.9, yerr=err, color=COLORS.get(a, "#8c5bd6"), label=a, capsize=2,
                   error_kw=dict(lw=0.8))
        ax.set_xticks(range(len(settings)), [TICK_TXT.get(s, s) for s in settings], fontsize=8)
        ax.set_title(t, fontsize=10, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.yaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
    axs.flat[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=150)
    plt.close(fig)

    # ---- goal | drawing grids: best and worst G episodes per setting ----
    for s in settings:
        for a in agents:
            g = df[(df.setting == s) & (df.agent == a)].sort_values("dilated_iou")
            if not len(g):
                continue
            picks = [("worst", g.head(6)), ("best", g.tail(6)[::-1])]
            blocks = []
            for lab, sel in picks:
                tiles = []
                for r in sel.to_dict("records"):
                    key = f"{r['category'].replace(' ', '_')}_{r['i']:03d}.png"
                    gi = cv2.imread(str(ev / s / "goal" / key)); ci = cv2.imread(str(ev / s / a / key))
                    if gi is None or ci is None:
                        continue
                    gi = cv2.resize(gi, (128, 128), interpolation=cv2.INTER_AREA)
                    ci = cv2.resize(ci, (128, 128), interpolation=cv2.INTER_AREA)
                    t = np.hstack([gi, np.full((128, 2, 3), 160, np.uint8), ci])
                    t = cv2.copyMakeBorder(t, 18, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    cv2.putText(t, f"{r['category'][:8]} IoU {r['dilated_iou']:.2f} {r['failure']}", (4, 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
                    tiles.append(t)
                if tiles:
                    row = np.hstack(tiles)
                    head = np.full((22, row.shape[1], 3), 255, np.uint8)
                    cv2.putText(head, f"{SETTING_TXT.get(s, s)} / {a}: {lab} (goal | drawn)", (4, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    blocks += [head, row]
            if blocks:
                wmax = max(b.shape[1] for b in blocks)
                blocks = [cv2.copyMakeBorder(b, 0, 0, 0, wmax - b.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                          for b in blocks]
                cv2.imwrite(str(out / f"grid_{s}_{a}.png"), np.vstack(blocks))
    print(open(out / "summary.md").read())
    print(fails.to_string())


if __name__ == "__main__":
    main()
