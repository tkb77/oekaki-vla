"""Join the per-(setting, category) shards of scripts/cloud_eval_goal.sh into one evaluation directory, laid out as a
single evaluate_goal.py run: <out>/<setting>/{<agent>,goal}/*.png|mp4 and <out>/<setting>/results_<agent>.jsonl
(rows in category order, then episode order).

Usage: python scripts/merge_eval_shards.py [--shards outputs/eval_goal_shards] [--out outputs/eval_goal]
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict

from oekaki.config import resolve
from oekaki.data.goal import load_goal_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="outputs/eval_goal_shards")
    ap.add_argument("--out", default="outputs/eval_goal")
    args = ap.parse_args()
    src, out = resolve(args.shards), resolve(args.out)
    order = {c: k for k, c in enumerate(load_goal_config()["categories"])}
    rows = defaultdict(list)
    n_files = 0
    for shard in sorted(p for p in src.iterdir() if p.is_dir()):
        for sdir in sorted(p for p in shard.iterdir() if p.is_dir()):
            for f in sdir.glob("results_*.jsonl"):
                rows[(sdir.name, f.stem[len("results_"):])] += [json.loads(l) for l in open(f) if l.strip()]
            for sub in (p for p in sdir.iterdir() if p.is_dir()):
                (out / sdir.name / sub.name).mkdir(parents=True, exist_ok=True)
                for f in sub.iterdir():
                    shutil.copy2(f, out / sdir.name / sub.name / f.name)
                    n_files += 1
    for (setting, agent), rs in sorted(rows.items()):
        rs.sort(key=lambda r: (order.get(r["category"], 99), r["i"]))
        with open(out / setting / f"results_{agent}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
        print(f"{setting}/{agent}: {len(rs)} episodes")
    print(f"{n_files} files -> {out}")


if __name__ == "__main__":
    main()
