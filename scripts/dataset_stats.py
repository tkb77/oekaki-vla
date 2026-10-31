"""Dataset statistics + integrity checks (episode/frame counts, NaNs, per-shape lengths, randomisation
coverage) and a visual summary. Also verifies both datasets load through LeRobotDataset.

Usage: python scripts/dataset_stats.py --name touch  -> artifacts/touch/dataset_stats_<name>.{json,png}
"""
import argparse
import json
from collections import Counter

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oekaki.config import ROOT, load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="touch")
    args = ap.parse_args()
    dcfg = load_config("configs/datagen.yaml")
    root = resolve(dcfg["dataset"]["root"]) / args.name
    out = ROOT / "artifacts" / "touch"
    out.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    stats = {}
    for cond in ("A", "B"):
        ds = LeRobotDataset(f"local/{dcfg['dataset']['repo_prefix']}_{args.name}_{cond}", root=root / cond)
        df = pd.concat([pd.read_parquet(p) for p in sorted((root / cond / "data").rglob("*.parquet"))])
        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())
        item = ds[len(ds) // 2]
        stats[cond] = dict(
            episodes=ds.num_episodes, frames=ds.num_frames, fps=ds.fps,
            state_dim=int(state.shape[1]), action_dim=int(action.shape[1]),
            nan_state=int(np.isnan(state).sum()), nan_action=int(np.isnan(action).sum()),
            tasks=int(len(ds.meta.tasks)),
            image_shape=list(item["observation.images.top"].shape),
            state_mean=state.mean(0).round(4).tolist(), state_std=state.std(0).round(4).tolist(),
            action_mean=action.mean(0).round(4).tolist(), action_std=action.std(0).round(4).tolist(),
        )
        if cond == "A":
            stateA, dfA, dsA = state, df, ds
    assert stats["A"]["frames"] == stats["B"]["frames"], "A/B frame mismatch"

    mapping = json.load(open(root / "episode_to_job.json"))
    metas = {m["job_id"]: m for m in (json.load(open(p)) for p in (root / "meta").glob("job_*.json"))}
    failed = [m for m in metas.values() if not m["success"]]
    used = [metas[j] for j in mapping]
    lens = dfA.groupby("episode_index").size()
    per_shape = {}
    for s in dcfg["shapes_to_use"]:
        ep_idx = [i for i, m in enumerate(used) if m["shape"] == s]
        L = lens.loc[ep_idx].to_numpy()
        per_shape[s] = dict(episodes=len(ep_idx), steps_mean=float(L.mean()), steps_min=int(L.min()),
                            steps_max=int(L.max()))
    summary = dict(
        datasets=stats, jobs=len(metas), failed_jobs=len(failed), per_shape=per_shape,
        noise_fraction=float(np.mean([m["noise"]["enabled"] for m in used])),
        disturbance_fraction=float(np.mean([m["disturbance"] is not None for m in used])),
        perturbed_fraction=float(np.mean([m["noise"]["enabled"] or m["disturbance"] is not None for m in used])),
        expert_coverage_mean=float(np.mean([m["coverage"] for m in used])),
        expert_air_ratio_mean=float(np.mean([m["air_ratio"] for m in used])),
        contact_fraction_frames=float(stateA[:, 6].mean()),
        tasks=Counter(m["task"] for m in used).most_common(),
        total_duration_min=float(stats["A"]["frames"] / stats["A"]["fps"] / 60),
    )
    json.dump(summary, open(out / f"dataset_stats_{args.name}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("datasets", "tasks")}, indent=1))
    print({c: {k: stats[c][k] for k in ("episodes", "frames", "state_dim", "nan_state", "nan_action", "tasks")}
           for c in stats})

    # ---------- figure ----------
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    ax = axs[0, 0]
    ax.bar(per_shape.keys(), [v["steps_mean"] / 30 for v in per_shape.values()], color="#4C72B0")
    ax.set_ylabel("mean episode length [s]"); ax.set_title(f"{stats['A']['episodes']} episodes / {stats['A']['frames']} frames")
    ax = axs[0, 1]
    ax.hist([m["paper_height_offset"] * 1e3 for m in used], bins=20, color="#55A868")
    ax.set_xlabel("paper height offset [mm]"); ax.set_title("paper height randomisation")
    ax = axs[0, 2]
    cats = ["clean", "noise only", "disturb only", "noise+disturb"]
    cnt = Counter(
        cats[int(m["noise"]["enabled"]) + 2 * int(m["disturbance"] is not None)] for m in used)
    ax.bar(cats, [cnt[c] for c in cats], color="#C44E52"); ax.set_title("episode variants")
    ax = axs[1, 0]
    ax.hist(stateA[:, 7], bins=50, color="#8172B2", log=True)
    ax.set_xlabel("compression [mm]"); ax.set_title("compression distribution (all frames)")
    ax = axs[1, 1]
    ep0 = dfA[dfA["episode_index"] == 0]
    st0 = np.stack(ep0["observation.state"].to_numpy())
    ax.plot(st0[:, 7], label="compression [mm]"); ax.plot(st0[:, 6] * 3, label="contact x3")
    ax.set_xlabel("frame"); ax.set_title(f"episode 0: {used[0]['task']}"); ax.legend()
    ax = axs[1, 2]
    tiles = []
    for s in dcfg["shapes_to_use"]:
        j = next(i for i, m in enumerate(used) if m["shape"] == s)
        tiles.append(cv2.imread(str(root / "meta" / f"job_{mapping[j]:05d}.png"))[..., ::-1])
    ax.imshow(np.hstack([cv2.resize(t, (128, 128)) for t in tiles])); ax.axis("off")
    ax.set_title("final canvas_img examples")
    fig.tight_layout(); fig.savefig(out / f"dataset_stats_{args.name}.png", dpi=110)

    # sample camera frames
    rows = []
    for e in [0, 1, 2]:
        idx = int(dsA.meta.episodes["dataset_from_index"][e]) + int(lens.iloc[e]) // 2
        it = dsA[idx]
        rows.append(np.hstack([(it[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                               for c in ("top", "wrist")]))
    cv2.imwrite(str(out / f"dataset_frames_{args.name}.png"), np.vstack(rows)[..., ::-1])


if __name__ == "__main__":
    main()
