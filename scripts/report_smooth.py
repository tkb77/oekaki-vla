"""Smooth-drawing report: drawing quality of the training variants (A and B) x inference schemes.

Reads artifacts/eval_smooth/results_{model}_{scheme}.json (+ results_expert_expert.json for the reference ink
amount) and writes artifacts/report_smooth/: summary.{csv,md}, paired.{csv,md} (each run vs the baseline with
temporal ensembling and vs the baseline with inference-time RTC at latency 3), quality.png, galleries.

quality success = success (CLIP correct, < 10 % of the path in the air, ink drawn)
                  and 0.6 <= ink / expert ink <= 1.4 (no unfinished or over-drawn shape)
                  and Chamfer distance to the target shape <= 2 mm (pose-invariant, see shape_fidelity).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from make_report import SHAPES, TEXT2, mcnemar_exact, style, summarize  # noqa: E402

from oekaki.config import resolve  # noqa: E402

MODELS = {"B": ("B: joints only", "#7d7d78"), "base": ("A: baseline", "#2a78d6"), "hermite": ("+ Hermite aux.", "#eb6834"),
          "ttrtc": ("+ training-time RTC", "#1baf7a"), "both": ("+ TT-RTC + Hermite", "#8c5bd6"),
          "time": ("+ elapsed-time input", "#d6a01b"), "hermite1": ("+ Hermite aux. (λ=1)", "#c4475b"),
          "tttime": ("+ TT-RTC + elapsed time", "#1f8f8f")}
# smooth schemes: initial flow noise 0, new chunk every 30 of 50 steps (picked by the sweep)
SCHEMES = {"naive": "chunking, noise 1, replan 5 (before)", "ensemble": "temporal ensembling, noise 1 (before)",
           "chunk_d0": "chunking, no latency", "chunk_d3": "chunking, latency 3",
           "rtc_d3": "inference-time RTC, latency 3", "ttrtc_d3": "training-time RTC, latency 3"}
REF = ("base", "ensemble")
# report storyline: plain SmolVLA -> smooth inference -> final model (A + TT-RTC + elapsed time)
STORY = [("B", "naive", "1 plain\nB"), ("base", "naive", "1 plain\nA"),
         ("B", "rtc_d3", "2 smooth\nB"), ("base", "rtc_d3", "2 smooth\nA"),
         ("tttime", "ttrtc_d3", "3 final\nA+TT-RTC\n+time")]


def load(eval_dir: Path, expert_json: Path | None = None) -> pd.DataFrame:
    rows = []
    files = sorted(eval_dir.glob("results_*.json"))
    if expert_json is not None:
        files.append(expert_json)
    for f in files:
        model, scheme = f.stem[len("results_"):].split("_", 1)
        for r in json.load(open(f)):
            rows.append(dict(r, model=model, scheme=scheme, agent=f"{model}|{scheme}"))
    df, _ = summarize(pd.DataFrame(rows))
    exp = df[df.model == "expert"][["cond", "shape", "ep", "ink_px"]].rename(columns={"ink_px": "ink_expert"})
    df = df[df.model != "expert"].merge(exp, on=["cond", "shape", "ep"], how="left")
    df["ink_ratio"] = df["ink_px"] / df["ink_expert"]
    df["quality"] = df["success"] & df["ink_ratio"].between(0.6, 1.4) & (df["chamfer_mm"] <= 2.0)
    df["seam_ratio"] = df["seam_jump"] / df["step_jump"]
    return df


METRICS = ["success", "quality", "chamfer_mm", "shape_prec", "shape_cov", "ink_ratio", "path_air_ratio", "jerk",
           "seam_jump", "step_jump"]


def summary(df):
    g = df.groupby(["model", "scheme", "cond"])
    agg = g[METRICS].mean()
    agg["n"] = g.size()
    agg = agg.reset_index()
    agg["model"] = pd.Categorical(agg["model"], list(MODELS))
    agg["scheme"] = pd.Categorical(agg["scheme"], list(SCHEMES))
    return agg.sort_values(["cond", "scheme", "model"])


def paired(df, ref, n_perm=20000, seed=0):
    rng = np.random.default_rng(seed)
    key = ["cond", "shape", "ep"]
    r = df[df.agent == f"{ref[0]}|{ref[1]}"]
    rows = []
    for run, g0 in df.groupby("agent"):
        if run == f"{ref[0]}|{ref[1]}":
            continue
        m = g0.merge(r, on=key, suffixes=("_a", "_b"))
        for cond, g in m.groupby("cond"):
            for met in ["success", "quality"]:
                sa, sb = g[f"{met}_a"].astype(bool).values, g[f"{met}_b"].astype(bool).values
                rows.append(dict(run=run, ref="|".join(ref), cond=cond, metric=met, n=len(g), mean_run=sa.mean(),
                                 mean_ref=sb.mean(), diff=sa.mean() - sb.mean(),
                                 p=mcnemar_exact(int((sa & ~sb).sum()), int((~sa & sb).sum()))))
            for met in ["chamfer_mm", "ink_ratio", "jerk", "seam_jump"]:
                d = (g[f"{met}_a"] - g[f"{met}_b"]).values.astype(float)
                d = d[~np.isnan(d)]
                if len(d) < 3:
                    continue
                flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
                p = float(((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (n_perm + 1))
                rows.append(dict(run=run, ref="|".join(ref), cond=cond, metric=met, n=len(d),
                                 mean_run=g[f"{met}_a"].mean(), mean_ref=g[f"{met}_b"].mean(), diff=d.mean(), p=p))
    return pd.DataFrame(rows)


def figure(out, agg):
    metrics = [("quality", "quality success ↑"), ("success", "success (CLIP + contact) ↑"),
               ("chamfer_mm", "Chamfer to target shape (mm) ↓"), ("jerk", "action jerk (rad²/s⁶) ↓"),
               ("seam_jump", "joint step at chunk seams (rad) ↓"), ("ink_ratio", "ink / expert ink (1 = expert)")]
    schemes = [s for s in SCHEMES if s in set(agg.scheme.astype(str))]
    for cond in ["standard", "disturb"]:
        a = agg[agg.cond == cond]
        fig, axs = plt.subplots(2, 3, figsize=(15, 7.2))
        w = 0.8 / len(MODELS)
        for ax, (m, title) in zip(axs.flat, metrics):
            for j, (mk, (lab, col)) in enumerate(MODELS.items()):
                xs, ys = [], []
                for i, s in enumerate(schemes):
                    v = a[(a.model == mk) & (a.scheme == s)][m]
                    if len(v):
                        xs.append(i + (j - (len(MODELS) - 1) / 2) * w)
                        ys.append(float(v.iloc[0]))
                ax.bar(xs, ys, w * 0.92, color=col, label=lab)
            if m == "ink_ratio":
                ax.axhline(1.0, color=TEXT2, lw=1, ls="--")
            if m == "jerk":
                ax.set_yscale("log")
            ax.set_xticks(range(len(schemes)), [SCHEMES[s].replace(", ", "\n") for s in schemes], fontsize=8)
            ax.set_title(title, fontsize=10, color="#1a1a19", loc="left")
            style(ax)
        h, l = axs.flat[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=len(MODELS), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / f"quality_{cond}.png", dpi=150)
        plt.close(fig)


def gallery(out, eval_dir, runs, cond, shapes=SHAPES, n=2):
    rows = []
    for model, scheme in runs:
        tiles = []
        for s in shapes:
            for i in range(n):
                p = eval_dir / scheme / model / "canvas" / f"{cond}_{s}_{i:02d}.png"
                if model == "expert":
                    p = eval_dir / "expert" / "expert" / "canvas" / f"{cond}_{s}_{i:02d}.png"
                im = cv2.imread(str(p)) if p.exists() else np.full((512, 512, 3), 200, np.uint8)
                im = cv2.resize(im, (120, 120), interpolation=cv2.INTER_AREA)
                cv2.rectangle(im, (0, 0), (119, 119), (180, 180, 180), 1)
                tiles.append(im)
        row = np.hstack(tiles)
        label = np.full((22, row.shape[1], 3), 255, np.uint8)
        name = "expert" if model == "expert" else f"{MODELS[model][0]} / {SCHEMES[scheme]}"
        cv2.putText(label, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        rows += [label, row]
    cv2.imwrite(str(out / f"gallery_{cond}.png"), np.vstack(rows))


def story(out, eval_dir, df):
    """Headline figure (quality and jerk along the storyline) and a gallery of the same (unpicked) episodes."""
    runs = [r for r in STORY if f"{r[0]}|{r[1]}" in set(df.agent)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    cols = {"standard": "#2a78d6", "disturb": "#eb6834"}
    for ax, (m, title) in zip(axs, [("quality", "quality success ↑"), ("chamfer_mm", "Chamfer to target (mm) ↓"),
                                    ("jerk", "action jerk (rad²/s⁶, log) ↓")]):
        for k, cond in enumerate(["standard", "disturb"]):
            ys = [df[(df.agent == f"{mo}|{sc}") & (df.cond == cond)][m].mean() for mo, sc, _ in runs]
            ax.bar(np.arange(len(runs)) + (k - 0.5) * 0.38, ys, 0.36, color=cols[cond], label=cond)
        ex = {"quality": 1.0, "chamfer_mm": None, "jerk": None}[m]
        if ex:
            ax.axhline(ex, color=TEXT2, lw=1, ls="--")
        if m == "jerk":
            ax.set_yscale("log")
        ax.set_xticks(range(len(runs)), [r[2] for r in runs], fontsize=8.5)
        ax.set_title(title, fontsize=10, color="#1a1a19", loc="left")
        style(ax)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.text(0.01, 0.01, "1 plain: sampled chunks (σ=1), replan every 5 steps.   2 smooth: σ=0, replan every 30 of 50 steps, "
             "inference-time RTC, latency 3.   3 final: training-time RTC + elapsed-time input, latency 3.",
             fontsize=8, color=TEXT2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "story.png", dpi=160)
    plt.close(fig)
    rows = [("B", "naive"), ("B", "rtc_d3"), ("tttime", "ttrtc_d3"), ("expert", "expert")]
    names = ["1. plain SmolVLA (B, sampled chunks, replan 5)", "2. + smooth inference (B, sigma=0, replan 30, RTC)",
             "3. final (A + TT-RTC + elapsed time)", "scripted expert"]
    for cond in ["standard", "disturb"]:
        blocks = []
        for (model, scheme), name in zip(rows, names):
            tiles = []
            for sh in SHAPES:
                for i in range(2):
                    sub = "expert/expert" if model == "expert" else f"{scheme}/{model}"
                    p = eval_dir / sub / "canvas" / f"{cond}_{sh}_{i:02d}.png"
                    im = cv2.imread(str(p)) if p.exists() else np.full((512, 512, 3), 200, np.uint8)
                    im = cv2.resize(im, (150, 150), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(im, (0, 0), (149, 149), (180, 180, 180), 1)
                    tiles.append(im)
            row = np.hstack(tiles)
            label = np.full((26, row.shape[1], 3), 255, np.uint8)
            cv2.putText(label, name, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
            blocks += [label, row]
        cv2.imwrite(str(out / f"story_gallery_{cond}.png"), np.vstack(blocks))


def main():
    eval_dir, out = resolve("artifacts/eval_smooth"), resolve("artifacts/report_smooth")
    out.mkdir(parents=True, exist_ok=True)
    df = load(eval_dir)
    df.drop(columns=["clip_probs"], errors="ignore").to_csv(out / "episodes.csv", index=False)
    agg = summary(df)
    agg.to_csv(out / "summary.csv", index=False)
    with open(out / "summary.md", "w") as f:
        f.write(agg.to_markdown(index=False, floatfmt=".3f"))
    refs = [REF, ("base", "rtc_d3"), ("B", "naive"), ("B", "rtc_d3")]  # before the smoothing work / best baseline scheme at latency 3
    ps = pd.concat([paired(df, r) for r in refs if f"{r[0]}|{r[1]}" in set(df.agent)])
    ps.to_csv(out / "paired.csv", index=False)
    with open(out / "paired.md", "w") as f:
        f.write(ps.to_markdown(index=False, floatfmt=".4g"))
    figure(out, agg)
    runs = [("expert", "expert")] + sorted({(m, s) for m, s in zip(df.model, df.scheme)},
                                           key=lambda x: (list(SCHEMES).index(x[1]), list(MODELS).index(x[0])))
    for cond in ["standard", "disturb"]:
        gallery(out, eval_dir, runs, cond)
    story(out, eval_dir, df)
    print(agg.to_string())


if __name__ == "__main__":
    main()
