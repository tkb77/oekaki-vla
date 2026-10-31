"""Inference-scheme ablation figure: temporal ensembling vs plain chunking (replan every 5 steps),
for conditions A and B on the same paired episodes.

Usage: python scripts/plot_ablation.py   (reads artifacts/eval/results_{A,B}_main.json and
       artifacts/eval_chunk5/results_{A,B}_chunk5.json, writes artifacts/report_chunk5/ablation_inference.png)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from make_report import COLORS, TEXT2, style, summarize  # noqa: E402

from oekaki.config import resolve  # noqa: E402

SCHEMES = [("ens", "temporal ensembling", "artifacts/eval/results_{c}_main.json"),
           ("chunk", "replan every 5 steps", "artifacts/eval_chunk5/results_{c}_chunk5.json")]


def main():
    rows = []
    for c in "AB":
        for key, _, path in SCHEMES:
            for r in json.load(open(resolve(path.format(c=c)))):
                rows.append(dict(r, agent=c, scheme=key))
    df, _ = summarize(pd.DataFrame(rows))
    agg = df.groupby(["cond", "agent", "scheme"])[["success", "path_air_ratio", "jerk"]].mean()
    conds = ["standard", "disturb"]
    metrics = [("success", "success rate ↑"), ("path_air_ratio", "path drawn in the air ↓"),
               ("jerk", "action jerk (rad/s³) ↓")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    w = 0.19
    for ax, (m, title) in zip(axs, metrics):
        k = 0
        for c in "AB":
            for key, label, _ in SCHEMES:
                vals = [agg.loc[(cd, c, key), m] for cd in conds]
                x = np.arange(len(conds)) + (k - 1.5) * w
                hatch = "////" if key == "ens" else None
                bs = ax.bar(x, vals, w * 0.9, color=COLORS[c] if key == "chunk" else "white",
                            edgecolor=COLORS[c], hatch=hatch, lw=1.2, label=f"{c}: {label}")
                for b, v in zip(bs, vals):
                    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}" if m != "jerk" else f"{v / 1e3:.0f}k",
                            ha="center", va="bottom", fontsize=7, color=TEXT2)
                k += 1
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
        ax.set_title(title, loc="left", fontsize=10); style(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = resolve("artifacts/report_chunk5"); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "ablation_inference.png", dpi=150); plt.close(fig)
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
