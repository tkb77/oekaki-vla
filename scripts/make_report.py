"""Learning curves, A/B comparison table + charts, and canvas galleries for the report.

Usage: python scripts/make_report.py --eval artifacts/eval/results_A_B_expert.json --out artifacts/report
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

from oekaki.config import load_config, resolve

COLORS = {"A": "#2a78d6", "B": "#eb6834", "expert": "#1baf7a"}
LABELS = {"A": "A: joints + contact", "B": "B: joints only", "expert": "scripted expert"}
TEXT2 = "#52514e"
GRID = "#e4e3df"
SHAPES = ["circle", "square", "triangle", "star", "line"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def learning_curves(out: Path, runs: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, d in runs.items():
        f = resolve(d) / "train_log.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        ax.plot(df["step"], df["loss"].rolling(5, min_periods=1).mean(), color=COLORS[cond], lw=2,
                label=f"{LABELS[cond]} (train)")
        v = df.dropna(subset=["val_loss"])
        v = v[v["val_loss"].astype(str) != ""]
        if len(v):
            ax.plot(v["step"], v["val_loss"].astype(float), color=COLORS[cond], lw=2, ls="--", marker="o", ms=4,
                    label=f"{LABELS[cond]} (val)")
    ax.set_yscale("log"); ax.set_xlabel("optimizer step", color=TEXT2); ax.set_ylabel("flow-matching loss", color=TEXT2)
    ax.set_title("SmolVLA fine-tuning loss", loc="left", fontsize=11)
    style(ax); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=150); plt.close(fig)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clip_correct"] = df["clip_pred"] == df["shape"] if "clip_pred" in df else np.nan
    df["geo_correct"] = df["geo_pred"] == df["shape"]
    df["drew"] = df["ink_px"] > 200
    df["success"] = df["clip_correct"] & (df["path_air_ratio"] < 0.10) & df["drew"]
    # The floor plane sits 2.4 mm below z=0: when the paper is lowered more than ~22 mm (out_of_range or a
    # large downward disturbance) its surface sinks below the floor and disappears from the camera images.
    cfg = load_config()
    hide_at = cfg["visual_floor_z"] - cfg["canvas"]["height"] if "visual_floor_z" in cfg else -0.0024 - cfg["canvas"]["height"]
    low = [h + min(0.0, d["delta"]) if isinstance(d, dict) else h for h, d in zip(df["height"], df["disturb"])]
    df["paper_hidden"] = np.array(low) < hide_at
    agg = df.groupby(["cond", "agent"]).agg(
        n=("shape", "size"),
        success=("success", "mean"),
        clip_acc=("clip_correct", "mean"),
        geo_acc=("geo_correct", "mean"),
        air_ratio=("air_ratio", "mean"),
        path_air_ratio=("path_air_ratio", "mean"),
        breaks=("n_breaks", "mean"),
        fragments=("n_fragments", "mean"),
        over_press=("over_press_ratio", "mean"),
        max_comp_mm=("max_compression_mm", "mean"),
        jerk=("jerk", "mean"),
    ).reset_index()
    return df, agg


def bars(out: Path, agg: pd.DataFrame, agents: list[str], conds: list[str]):
    metrics = [("success", "overall success ↑"), ("clip_acc", "recognised (CLIP) ↑"),
               ("path_air_ratio", "path drawn in the air ↓"), ("over_press", "over-press ratio ↓")]
    fig, axs = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    w = 0.8 / len(agents)
    for ax, (m, title) in zip(axs, metrics):
        for k, a in enumerate(agents):
            vals = [agg[(agg.cond == c) & (agg.agent == a)][m].mean() for c in conds]
            x = np.arange(len(conds)) + (k - (len(agents) - 1) / 2) * w
            bs = ax.bar(x, vals, w * 0.92, color=COLORS.get(a, "#555"), label=LABELS.get(a, a))
            for b, v in zip(bs, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom",
                            fontsize=7, color=TEXT2)
        ax.set_xticks(np.arange(len(conds))); ax.set_xticklabels(conds)
        ax.set_title(title, loc="left", fontsize=10); style(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=len(agents))
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out / "ab_comparison.png", dpi=150); plt.close(fig)


def per_shape(out: Path, df: pd.DataFrame, agents, cond):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    w = 0.8 / len(agents)
    for ax, (m, title) in zip(axs, [("success", f"success by shape ({cond})"),
                                    ("path_air_ratio", f"path in the air by shape ({cond})")]):
        for k, a in enumerate(agents):
            sub = df[(df.agent == a) & (df.cond == cond)]
            vals = [sub[sub["shape"] == s][m].mean() for s in SHAPES]
            x = np.arange(len(SHAPES)) + (k - (len(agents) - 1) / 2) * w
            ax.bar(x, vals, w * 0.92, color=COLORS.get(a, "#555"), label=LABELS.get(a, a))
        ax.set_xticks(range(len(SHAPES))); ax.set_xticklabels(SHAPES)
        ax.set_title(title, loc="left", fontsize=10); style(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=len(agents))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out / f"per_shape_{cond}.png", dpi=150); plt.close(fig)


def gallery(out: Path, eval_dir: Path, agents, cond, n=3):
    rows = []
    for a in agents:
        tiles = []
        for s in SHAPES:
            for i in range(n):
                p = eval_dir / a / "canvas" / f"{cond}_{s}_{i:02d}.png"
                im = cv2.imread(str(p)) if p.exists() else np.full((512, 512, 3), 200, np.uint8)
                im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
                cv2.rectangle(im, (0, 0), (127, 127), (180, 180, 180), 1)
                tiles.append(im)
        row = np.hstack(tiles)
        label = np.full((22, row.shape[1], 3), 255, np.uint8)
        cv2.putText(label, f"{LABELS.get(a, a)} - {cond}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1,
                    cv2.LINE_AA)
        rows += [label, row]
    cv2.imwrite(str(out / f"gallery_{cond}.png"), np.vstack(rows))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_stats(df: pd.DataFrame, a: str = "A", b: str = "B", n_perm: int = 20000, seed: int = 0) -> pd.DataFrame:
    """A-vs-B differences on paired episodes (same seed -> same paper height, offset, disturbance, task).
    success: exact McNemar; continuous metrics: sign-flip permutation test + bootstrap 95% CI of the mean diff."""
    rng = np.random.default_rng(seed)
    key = ["cond", "shape", "ep"]
    m = df[df.agent == a].merge(df[df.agent == b], on=key, suffixes=("_a", "_b"))
    rows = []
    for cond, g in m.groupby("cond", sort=False):
        sa, sb = g["success_a"].astype(bool).values, g["success_b"].astype(bool).values
        n10, n01 = int((sa & ~sb).sum()), int((~sa & sb).sum())
        rows.append(dict(cond=cond, metric="success", n=len(g), mean_a=sa.mean(), mean_b=sb.mean(),
                         diff=sa.mean() - sb.mean(), ci_lo=np.nan, ci_hi=np.nan,
                         p=mcnemar_exact(n10, n01), note=f"A-only {n10} / B-only {n01}"))
        for met in ["path_air_ratio", "air_ratio", "n_breaks", "n_fragments", "max_compression_mm", "jerk"]:
            d = (g[f"{met}_a"] - g[f"{met}_b"]).values.astype(float)
            obs = d.mean()
            flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
            p = float(((np.abs((flips * d).mean(1)) >= abs(obs) - 1e-12).sum() + 1) / (n_perm + 1))
            boots = d[rng.integers(0, len(d), size=(n_perm, len(d)))].mean(1)
            rows.append(dict(cond=cond, metric=met, n=len(d), mean_a=g[f"{met}_a"].mean(), mean_b=g[f"{met}_b"].mean(),
                             diff=obs, ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5), p=p, note=""))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True, help="results_*.json files from evaluate.py")
    ap.add_argument("--out", default="artifacts/report")
    ap.add_argument("--runs", nargs="*", default=["A=outputs/train/A", "B=outputs/train/B"])
    args = ap.parse_args()
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    learning_curves(out, dict(r.split("=", 1) for r in args.runs))
    rows = []
    for f in args.eval:
        rows += json.load(open(resolve(f)))
    df = pd.DataFrame(rows)
    df, agg = summarize(df)
    order = [a for a in ["A", "B", "expert"] if a in set(df.agent)] + \
        sorted(set(df.agent) - {"A", "B", "expert"})
    conds = [c for c in ["standard", "disturb", "out_of_range"] if c in set(df.cond)]
    agg.to_csv(out / "comparison.csv", index=False)
    with open(out / "comparison.md", "w") as f:
        f.write(agg.round(3).to_markdown(index=False))
    print(agg.round(3).to_string(index=False))
    bars(out, agg, order, conds)
    eval_dir = resolve(args.eval[0]).parent
    for c in conds:
        per_shape(out, df, order, c)
        gallery(out, eval_dir, order, c)
    df.to_csv(out / "episodes.csv", index=False)
    vis = df[~df["paper_hidden"]]
    if df["paper_hidden"].any():
        _, agg_v = summarize(vis.drop(columns=["success", "clip_correct", "geo_correct", "drew", "paper_hidden"]))
        n_hidden = df.groupby(["cond", "agent"])["paper_hidden"].sum().rename("n_hidden").reset_index()
        agg_v = agg_v.merge(n_hidden, on=["cond", "agent"])
        agg_v.to_csv(out / "comparison_paper_visible.csv", index=False)
        with open(out / "comparison_paper_visible.md", "w") as f:
            f.write("Episodes whose paper sank below the floor plane (invisible to the cameras) excluded.\n\n")
            f.write(agg_v.round(3).to_markdown(index=False))
    if {"A", "B"} <= set(df.agent):
        st = paired_stats(df)
        if df["paper_hidden"].any():
            sv = paired_stats(vis)
            sv["cond"] = sv["cond"] + " (paper visible)"
            st = pd.concat([st, sv[sv["cond"].str.startswith(tuple(df[df.paper_hidden].cond.unique()))]])
        st.to_csv(out / "paired_stats.csv", index=False)
        with open(out / "paired_stats.md", "w") as f:
            f.write(st.round(4).to_markdown(index=False))
        print(st.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
