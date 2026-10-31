"""Generate dataset goal_v1: the expert draws laid-out Quick, Draw! drawings; the goal image is a
third camera stream `observation.images.goal` (constant within an episode). State = joints + contact +
compression (condition A of the touch study). One dataset serves both models: N ignores the goal stream.

Per-episode metadata (category, Quick, Draw! key_id, layout transform, strokes, instruction, paper height,
disturbance, noise) goes to <root>/<name>/meta/; goal and final canvas are saved as PNG next to it.

Usage: python scripts/generate_goal_dataset.py [--config configs/datagen_goal_v1.yaml] [--episodes-per-category 200]
       [--workers 6] [--name goal_v1]
A config with a `pairs` block (configs/datagen_goal_v2.yaml) adds episodes with two drawings side by side.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from oekaki.config import load_config, resolve

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GOAL_KEY = "observation.images.goal"


def features(img: int) -> dict:
    names = list(JOINT_NAMES) + ["contact", "compression_mm"]
    vid = {"dtype": "video", "shape": (img, img, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.top": vid,
        "observation.images.wrist": dict(vid),
        GOAL_KEY: dict(vid),
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
    }


def worker(wid: int, jobs: list[dict], out: str, img: int, split: str, q, cfg_path: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from oekaki.data.rollout import run_expert_episode, state_vector
    from oekaki.data.goal import build_episode, load_sets, load_goal_config
    from oekaki.env.drawing_env import DrawingEnv

    env_cfg = load_config()
    env_cfg["cameras"]["width"] = env_cfg["cameras"]["height"] = img
    cfg, lcfg = load_goal_config(cfg_path), load_config("configs/layout.yaml")
    S = env_cfg["canvas"]["size"]
    sets = load_sets(split)
    out = Path(out)
    env = DrawingEnv(env_cfg, render=True)
    ds = LeRobotDataset.create(
        repo_id=f"local/shard_{wid:02d}", fps=env_cfg["sim"]["fps"], features=features(img),
        root=out / "shards" / f"{wid:02d}", robot_type="so101_pen", use_videos=True,
        streaming_encoding=True, encoder_threads=1, encoder_queue_maxsize=4096)
    local = 0
    for job in jobs:
        ep = build_episode(job, sets, cfg, lcfg, S)
        goal = ep["goal"] if ep["goal"].shape[0] == img else cv2.resize(ep["goal"], (img, img), interpolation=cv2.INTER_AREA)
        r = run_expert_episode(env, cfg, job["category"], np.random.default_rng(ep["rng_seed"]), shape=ep["shape"],
                               task=ep["task"])
        meta = r["meta"]
        meta.update(job_id=job["id"], worker=wid, category=job["category"], key_id=job["key_id"],
                    sketch_idx=job["sketch_idx"], sketch_split=split, layout=ep["layout"], named=ep["named"],
                    strokes_norm=[s.tolist() for s in ep["strokes_norm"]], pair=bool(job.get("pair")),
                    category2=job.get("category2"), key_id2=job.get("key_id2"), sketch_idx2=job.get("sketch_idx2"))
        if not meta["success"]:
            meta["shard_episode"] = None
        else:
            for fr in r["frames"]:
                obs = dict(joint_pos=fr["joint_pos"], contact=fr["contact"], compression=fr["compression"])
                ds.add_frame({
                    "observation.images.top": fr["images"]["top"],
                    "observation.images.wrist": fr["images"]["wrist"],
                    GOAL_KEY: goal,
                    "observation.state": state_vector(obs, True),
                    "action": fr["action"],
                    "task": meta["task"],
                })
            ds.save_episode()
            meta["shard_episode"] = local
            local += 1
        with open(out / "meta" / f"job_{job['id']:05d}.json", "w") as f:
            json.dump(meta, f)
        cv2.imwrite(str(out / "meta" / f"job_{job['id']:05d}_canvas.png"), r["canvas"][..., ::-1])
        cv2.imwrite(str(out / "meta" / f"job_{job['id']:05d}_goal.png"), ep["goal"][..., 0])
        q.put((wid, job["id"], meta["success"], meta["n_steps"]))
    ds.finalize()
    env.close()
    q.put((wid, None, None, None))


def check_video_integrity(root: Path):
    import av
    import pandas as pd
    n_data = sum(len(pd.read_parquet(p)) for p in (root / "data").rglob("*.parquet"))
    for cam_dir in sorted((root / "videos").iterdir()):
        n_vid = 0
        for f in sorted(cam_dir.rglob("*.mp4")):
            with av.open(str(f)) as c:
                n_vid += sum(1 for _ in c.decode(video=0))
        print(f"[integrity] {cam_dir.name}: video {n_vid} frames, data {n_data} rows -> "
              f"{'OK' if n_vid == n_data else 'MISMATCH'}")
        if n_vid != n_data:
            raise RuntimeError(f"video/data frame mismatch in {cam_dir.name}")


def main():
    from oekaki.data.goal import load_sets, load_goal_config, pick_pairs, pick_sketches
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datagen_goal_v1.yaml")
    ap.add_argument("--pairs", type=int, default=None, help="override pairs.n of the config")
    ap.add_argument("--episodes-per-category", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg, lcfg, env_cfg = load_goal_config(args.config), load_config("configs/layout.yaml"), load_config()
    dc = cfg["dataset"]
    n_per = args.episodes_per_category or cfg["episodes_per_category"]
    n_workers = args.workers or dc["workers"]
    name = args.name or dc["name"]
    img = dc["image_size"]
    out = resolve(dc["root"]) / name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    from oekaki.utils.gpu import gpu_status, wait_for_ram
    wait_for_ram(4000 * n_workers // 2)
    st = gpu_status()
    max_by_mem = int(max(st["free"] - 2000, 0) // 300)
    if max_by_mem < n_workers:
        print(f"[gpu-guard] free {st['free']:.0f} MiB -> limiting workers {n_workers} -> {max(max_by_mem, 1)}")
        n_workers = max(max_by_mem, 1)

    sets = load_sets(cfg["sketch_split"])
    jobs = pick_sketches(sets, cfg, lcfg, env_cfg["canvas"]["size"], n_per, cfg["seed"])
    if cfg.get("pairs"):
        n_pairs = cfg["pairs"]["n"] if args.pairs is None else args.pairs
        jobs += pick_pairs(sets, cfg, lcfg, env_cfg["canvas"]["size"], n_pairs, cfg["seed"] + 1)
    order = np.random.default_rng(cfg["seed"]).permutation(len(jobs))
    jobs = [dict(jobs[i], id=k) for k, i in enumerate(order)]
    json.dump(jobs, open(out / "jobs.json", "w"))
    chunks = [jobs[w::n_workers] for w in range(n_workers)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(w, chunks[w], str(out), img, cfg["sketch_split"], q, args.config))
             for w in range(n_workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    done_workers, n_done, n_fail, frames = 0, 0, 0, 0
    while done_workers < n_workers:
        wid, jid, ok, n = q.get()
        if jid is None:
            done_workers += 1
            continue
        n_done += 1; n_fail += int(not ok); frames += n if ok else 0
        if n_done % 25 == 0 or n_done == len(jobs):
            el = time.time() - t0
            print(f"{n_done}/{len(jobs)} episodes, {frames} frames, fail {n_fail}, {el:.0f}s "
                  f"(eta {el / n_done * (len(jobs) - n_done):.0f}s)", flush=True)
    for p in procs:
        p.join()
    print(f"generation done in {time.time() - t0:.0f}s")

    from lerobot.datasets.aggregate import aggregate_datasets
    shard_dirs = sorted((out / "shards").glob("*"))
    aggregate_datasets(repo_ids=[f"local/shard_{d.name}" for d in shard_dirs], aggr_repo_id=dc["repo_id"],
                       roots=shard_dirs, aggr_root=out / "G")
    metas = [json.load(open(p)) for p in sorted((out / "meta").glob("job_*.json"))]
    mapping = []
    for d in shard_dirs:
        wid = int(d.name)
        mine = sorted([m for m in metas if m["worker"] == wid and m["shard_episode"] is not None],
                      key=lambda m: m["shard_episode"])
        mapping += [m["job_id"] for m in mine]
    json.dump(mapping, open(out / "episode_to_job.json", "w"))
    check_video_integrity(out / "G")
    shutil.rmtree(out / "shards")
    print("dataset written:", out / "G")


if __name__ == "__main__":
    main()
