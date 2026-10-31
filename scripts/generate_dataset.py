"""Generate expert demonstrations as LeRobotDatasets for condition A (state = joints + contact +
compression) and condition B (state = joints only) from the *same* simulated episodes.

Each worker writes its own shard (A and B); shards are merged with lerobot's aggregate_datasets.
Per-episode metadata (paper height, disturbance, noise, ground-truth strokes, final canvas) is
saved under <root>/meta/.

Usage: python scripts/generate_dataset.py [--episodes-per-shape 100] [--workers 10] [--name touch]
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


def features(img: int, with_contact: bool) -> dict:
    names = list(JOINT_NAMES) + (["contact", "compression_mm"] if with_contact else [])
    vid = {"dtype": "video", "shape": (img, img, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.images.top": vid,
        "observation.images.wrist": dict(vid),
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINT_NAMES)},
    }


def worker(wid: int, jobs: list[dict], out: str, img: int, q):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from oekaki.data.rollout import run_expert_episode, state_vector
    from oekaki.env.drawing_env import DrawingEnv

    cfg = load_config()
    cfg["cameras"]["width"] = cfg["cameras"]["height"] = img
    dcfg = load_config("configs/datagen.yaml")
    out = Path(out)
    env = DrawingEnv(cfg, render=True)
    ds = {}
    for cond, wc in (("A", True), ("B", False)):
        ds[cond] = LeRobotDataset.create(
            repo_id=f"local/shard_{cond}_{wid:02d}", fps=cfg["sim"]["fps"], features=features(img, wc),
            root=out / "shards" / f"{cond}_{wid:02d}", robot_type="so101_pen", use_videos=True,
            streaming_encoding=True, encoder_threads=1,
            # offline generation: the queue must hold a whole episode, otherwise lerobot's real-time
            # streaming encoder silently drops frames when the CPU is busy
            encoder_queue_maxsize=4096)
    local = 0
    for job in jobs:
        rng = np.random.default_rng(job["seed"])
        ep = run_expert_episode(env, dcfg, job["shape"], rng)
        meta = ep["meta"]
        meta.update(job_id=job["id"], worker=wid)
        if not meta["success"]:
            meta["shard_episode"] = None
        else:
            for cond, wc in (("A", True), ("B", False)):
                for fr in ep["frames"]:
                    obs = dict(joint_pos=fr["joint_pos"], contact=fr["contact"], compression=fr["compression"])
                    ds[cond].add_frame({
                        "observation.images.top": fr["images"]["top"],
                        "observation.images.wrist": fr["images"]["wrist"],
                        "observation.state": state_vector(obs, wc),
                        "action": fr["action"],
                        "task": meta["task"],
                    })
                ds[cond].save_episode()
            meta["shard_episode"] = local
            local += 1
        with open(out / "meta" / f"job_{job['id']:05d}.json", "w") as f:
            json.dump(meta, f)
        cv2.imwrite(str(out / "meta" / f"job_{job['id']:05d}.png"), ep["canvas"][..., ::-1])
        q.put((wid, job["id"], meta["success"], meta["n_steps"]))
    for d in ds.values():
        d.finalize()
    env.close()
    q.put((wid, None, None, None))


def check_video_integrity(out: Path):
    """Every camera's concatenated videos must contain exactly as many frames as the parquet data."""
    import av
    import pandas as pd
    for cond in ("A", "B"):
        n_data = sum(len(pd.read_parquet(p)) for p in (out / cond / "data").rglob("*.parquet"))
        for cam_dir in sorted((out / cond / "videos").iterdir()):
            n_vid = 0
            for f in sorted(cam_dir.rglob("*.mp4")):
                with av.open(str(f)) as c:
                    n_vid += sum(1 for _ in c.decode(video=0))
            status = "OK" if n_vid == n_data else "MISMATCH"
            print(f"[integrity] {cond} {cam_dir.name}: video {n_vid} frames, data {n_data} rows -> {status}")
            if n_vid != n_data:
                raise RuntimeError(f"video/data frame mismatch in {cond}/{cam_dir.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-shape", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--name", default="touch")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dcfg = load_config("configs/datagen.yaml")
    n_per = args.episodes_per_shape or dcfg["episodes_per_shape"]
    n_workers = args.workers or dcfg["dataset"]["workers"]
    img = dcfg["dataset"]["image_size"]
    out = resolve(dcfg["dataset"]["root"]) / args.name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    # GPU guard: EGL rendering needs only ~150 MiB per worker; shrink the pool if memory is scarce.
    from oekaki.utils.gpu import gpu_status
    st = gpu_status()
    max_by_mem = int(max(st["free"] - 2000, 0) // 300)
    if max_by_mem < n_workers:
        print(f"[gpu-guard] free {st['free']:.0f} MiB -> limiting workers {n_workers} -> {max(max_by_mem, 1)}")
        n_workers = max(max_by_mem, 1)
    print(f"[gpu-guard] GPU free {st['free']:.0f} MiB, util {st['util']}%, procs {[(p['pid'], p['mem']) for p in st['procs']]}")

    rng = np.random.default_rng(dcfg["seed"])
    jobs = []
    for shape in dcfg["shapes_to_use"]:
        for _ in range(n_per):
            jobs.append(dict(shape=shape, seed=int(rng.integers(2**31))))
    order = rng.permutation(len(jobs))
    jobs = [dict(jobs[i], id=k) for k, i in enumerate(order)]
    chunks = [jobs[w::n_workers] for w in range(n_workers)]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(w, chunks[w], str(out), img, q)) for w in range(n_workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    done_workers, n_done, n_fail, frames = 0, 0, 0, 0
    while done_workers < n_workers:
        wid, jid, ok, n = q.get()
        if jid is None:
            done_workers += 1
            continue
        n_done += 1
        n_fail += int(not ok)
        frames += n if ok else 0
        if n_done % 25 == 0 or n_done == len(jobs):
            el = time.time() - t0
            print(f"{n_done}/{len(jobs)} episodes, {frames} frames, fail {n_fail}, {el:.0f}s "
                  f"(eta {el / n_done * (len(jobs) - n_done):.0f}s)", flush=True)
    for p in procs:
        p.join()
    print(f"generation done in {time.time() - t0:.0f}s")

    # ---- merge shards -------------------------------------------------------------------------
    from lerobot.datasets.aggregate import aggregate_datasets
    for cond in ("A", "B"):
        shard_dirs = sorted((out / "shards").glob(f"{cond}_*"))
        aggregate_datasets(repo_ids=[f"local/shard_{d.name.split('_')[0]}_{d.name.split('_')[1]}" for d in shard_dirs],
                           aggr_repo_id=f"local/{dcfg['dataset']['repo_prefix']}_{args.name}_{cond}",
                           roots=shard_dirs, aggr_root=out / cond)
    # episode index -> job mapping (aggregation keeps shard order, then episode order)
    metas = [json.load(open(p)) for p in sorted((out / "meta").glob("job_*.json"))]
    mapping = []
    for d in sorted((out / "shards").glob("A_*")):
        wid = int(d.name.split("_")[1])
        mine = sorted([m for m in metas if m["worker"] == wid and m["shard_episode"] is not None],
                      key=lambda m: m["shard_episode"])
        mapping += [m["job_id"] for m in mine]
    json.dump(mapping, open(out / "episode_to_job.json", "w"))
    check_video_integrity(out)
    shutil.rmtree(out / "shards")
    print("datasets written:", out / "A", out / "B")


if __name__ == "__main__":
    main()
