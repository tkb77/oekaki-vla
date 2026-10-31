"""Execution-horizon x sampling-temperature sweep (artifacts/eval_smooth_sweep) -> artifacts/report_smooth/sweep.{csv,md,png}.

Run tags look like chunk_d0_r10_n0.5 / rtc_d3_r5_n1 / ttrtc_d3_r20_n0 / ensemble_n1.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from make_report import TEXT2, style  # noqa: E402
from report_smooth import load  # noqa: E402

from oekaki.config import resolve  # noqa: E402

NOISE_COLORS = {1.0: "#2a78d6", 0.5: "#eb6834", 0.0: "#1baf7a"}


def parse(tag: str) -> dict:
    m = re.match(r"(?P<mode>[a-z]+)(?:_d(?P<d>\d+))?(?:_r(?P<r>\d+))?_n(?P<n>[\d.]+)", tag)
    return dict(mode=m["mode"], delay=int(m["d"] or 0), replan=int(m["r"] or 5), noise=float(m["n"]))


def main():
    out = resolve("artifacts/report_smooth")
    out.mkdir(parents=True, exist_ok=True)
    df = load(resolve("artifacts/eval_smooth_sweep"), resolve("artifacts/eval_smooth/results_expert_expert.json"))
    df = pd.concat([df, pd.DataFrame([parse(t) for t in df.scheme], index=df.index)], axis=1)
    cols = ["success", "quality", "chamfer_mm", "shape_prec", "ink_ratio", "path_air_ratio", "jerk", "seam_jump",
            "step_jump"]
    agg = df.groupby(["model", "mode", "delay", "replan", "noise"])[cols].mean()
    agg["n"] = df.groupby(["model", "mode", "delay", "replan", "noise"]).size()
    agg = agg.reset_index().sort_values(["model", "mode", "noise", "replan"])
    agg.to_csv(out / "sweep.csv", index=False)
    with open(out / "sweep.md", "w") as f:
        f.write(agg.to_markdown(index=False, floatfmt=".3f"))
    print(agg.to_string())

    groups = [(m, mo) for m, mo in agg[["model", "mode"]].drop_duplicates().itertuples(index=False) if mo != "ensemble"]
    metrics = [("jerk", "action jerk ↓ (log)"), ("seam_jump", "joint step at seams (rad) ↓"),
               ("quality", "quality success ↑"), ("chamfer_mm", "Chamfer to target (mm) ↓"),
               ("ink_ratio", "ink / expert ink")]
    fig, axs = plt.subplots(len(groups), len(metrics), figsize=(3.2 * len(metrics), 2.8 * len(groups)), squeeze=False)
    for gi, (model, mode) in enumerate(groups):
        a = agg[(agg.model == model) & (agg["mode"] == mode)]
        ens = agg[(agg.model == model) & (agg["mode"] == "ensemble") & (agg.noise == 1.0)]
        for ax, (m, title) in zip(axs[gi], metrics):
            for n, g in a.groupby("noise"):
                ax.plot(g.replan, g[m], marker="o", ms=5, lw=2, color=NOISE_COLORS.get(n, "#888"), label=f"noise ×{n:g}")
            if len(ens):
                ax.axhline(float(ens[m].iloc[0]), color=TEXT2, ls="--", lw=1, label="temporal ensembling")
            if m == "jerk":
                ax.set_yscale("log")
            if m == "ink_ratio":
                ax.axhline(1.0, color="#bbb", lw=1)
            ax.set_xticks(sorted(a.replan.unique()))
            ax.set_xlabel("actions executed per chunk", fontsize=8, color=TEXT2)
            ax.set_title(f"{model} / {mode}: {title}", fontsize=9, loc="left")
            style(ax)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "sweep.png", dpi=140)


if __name__ == "__main__":
    main()
