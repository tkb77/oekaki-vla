"""Fine-tune SmolVLA (lerobot/smolvla_base) on the oekaki dataset for condition A or B.

The policy's input features are taken from *our* dataset (state = 8-D for A, 6-D for B; cameras
top + wrist) while the weights come from smolvla_base. State/action are zero-padded to
max_state_dim/max_action_dim inside SmolVLA, so all pretrained weights load unchanged.

GPU safety: checks nvidia-smi first, waits if memory is taken by other users, and picks the
micro-batch so that gradient accumulation keeps the configured effective batch size. bf16 autocast.

Goal-image options (config): `exclude_features` (inputs the policy does not get, e.g. the goal camera for model N),
`train_episodes` / `val_episodes` (explicit episode lists, e.g. the overfitting test). `--dedicated` turns the
shared-machine guards off (own GPU / cloud instance).

Usage: python scripts/train_smolvla.py --config configs/train_A.yaml [--steps N] [--output-dir DIR] [--dedicated]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch

from oekaki.config import load_config, resolve
from oekaki.utils.gpu import gpu_status, ram_available_mib, wait_for_gpu, wait_for_ram

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("train")


def pick_micro_batch(free_mib: float, max_micro: int) -> int:
    # measured on L40S: 7.3 GiB peak at micro-batch 32 (bf16 autocast, frozen vision tower) -> ~0.14 GiB/sample + ~3 GiB fixed
    per_sample, fixed, margin = 160.0, 3500.0, 3000.0
    mb = int((free_mib - fixed - margin) // per_sample)
    mb = max(1, min(max_micro, mb))
    p = 1
    while p * 2 <= mb:
        p *= 2
    return p


def build_policy(cfg: dict, ds_meta, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    pcfg = PreTrainedConfig.from_pretrained(cfg["base_model"])
    feats = dataset_to_policy_features(ds_meta.features)
    pcfg.output_features = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    excl = set(cfg.get("exclude_features") or [])
    pcfg.input_features = {k: f for k, f in feats.items() if f.type is not FeatureType.ACTION and k not in excl}
    pcfg.chunk_size = cfg["chunk_size"]
    pcfg.n_action_steps = cfg["n_action_steps"]
    pcfg.device = device
    pcfg.pretrained_path = cfg["base_model"]
    pcfg.optimizer_lr = cfg["lr"]
    pcfg.scheduler_warmup_steps = cfg["warmup_steps"]
    pcfg.scheduler_decay_steps = cfg["decay_steps"]
    pcfg.scheduler_decay_lr = cfg["decay_lr"]
    policy = SmolVLAPolicy.from_pretrained(cfg["base_model"], config=pcfg)
    policy.to(device)
    if cfg.get("extras"):
        from oekaki.policy.smolvla_plus import upgrade_policy
        upgrade_policy(policy, cfg["extras"])
    pre, post = make_pre_post_processors(pcfg, dataset_stats=ds_meta.stats)
    return policy, pre, post, pcfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-micro-batch", type=int, default=None)
    ap.add_argument("--dedicated", action="store_true",
                    help="GPU and host are not shared: no waiting for memory, no foreign-load micro-batch cap")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dedicated:
        cfg.update(gpu_margin_mib=0, ram_start_mib=0, ram_pause_mib=0, dedicated=True)
    if args.steps:
        cfg["steps"] = args.steps
        cfg["decay_steps"] = args.steps
    if args.max_micro_batch:
        cfg["max_micro_batch"] = args.max_micro_batch
    out = resolve(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ---------------- GPU guard ----------------
    # start only when our estimated footprint plus a margin for one more process of another user is free;
    # our own allocations are capped so that we never grow into that margin
    margin = cfg.get("gpu_margin_mib", 0)
    max_mb = cfg["max_micro_batch"]
    if margin:
        max_mb = min(max_mb, cfg.get("shared_gpu_micro_batch", 32))
    need = 3500 + 160 * max_mb + 1500
    wait_for_ram(cfg.get("ram_start_mib", 24000), log=log.info)  # host RAM is shared too
    st = wait_for_gpu(required_mib=0 if cfg.get("dedicated") else max(12000 if not margin else 0, need + margin),
                      log=log.info)
    foreign = [p for p in st["procs"] if p["mem"] >= 2000]
    if foreign and st["util"] > 50 and not cfg.get("dedicated"):
        # another project is computing on the GPU: use a smaller micro-batch (less memory pressure);
        # gradient accumulation keeps the effective batch identical
        max_mb = min(max_mb, cfg.get("shared_gpu_micro_batch", 32))
        log.info(f"[gpu-guard] foreign GPU load detected (util {st['util']}%, {[(p['pid'], p['mem']) for p in foreign]}) "
                 f"-> micro-batch capped at {max_mb}")
    mb = pick_micro_batch(st["free"] - margin, max_mb)
    if margin:
        cap = 3500 + 160 * mb + 1500
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / st["total"]))
        log.info(f"[gpu-guard] memory cap {cap:.0f} MiB, margin kept free for others {margin} MiB")
    eff = cfg["effective_batch_size"]
    accum = max(1, math.ceil(eff / mb))
    log.info(f"GPU free {st['free']:.0f} MiB (others: {[(p['pid'], p['name'], p['mem']) for p in st['procs']]}) "
             f"-> micro-batch {mb} x accum {accum} = {mb * accum}")

    # ---------------- data ----------------
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    root = resolve(cfg["dataset_root"])
    meta = LeRobotDatasetMetadata(cfg["repo_id"], root=root)
    n_ep = meta.total_episodes
    if cfg.get("train_episodes_per_category"):
        from oekaki.data.goal import first_episodes_per_category
        cfg["train_episodes"] = first_episodes_per_category(root, cfg["train_episodes_per_category"])
    if cfg.get("episode_select"):  # subset of a goal_v2 dataset: {singles_per_category: N or null, pairs: bool}
        from oekaki.data.goal import select_episodes
        sel = select_episodes(root, **cfg["episode_select"])
        cfg["val_episodes"] = sel[::cfg["val_every_n_episode"]]
        cfg["train_episodes"] = [e for e in sel if e not in set(cfg["val_episodes"])]
    if cfg.get("train_episodes") is not None:
        train_eps = list(cfg["train_episodes"])
        val_eps = list(cfg.get("val_episodes") or train_eps)
    else:
        val_eps = list(range(0, n_ep, cfg["val_every_n_episode"]))
        train_eps = [e for e in range(n_ep) if e not in set(val_eps)]
    device = "cuda"
    policy, pre, post, pcfg = build_policy(cfg, meta, device)
    delta = resolve_delta_timestamps(pcfg, meta)
    vb = os.environ.get("OEKAKI_VIDEO_BACKEND") or None  # pyav where torchcodec has no system FFmpeg (remote nodes)
    train_ds = LeRobotDataset(cfg["repo_id"], root=root, episodes=train_eps, delta_timestamps=delta, video_backend=vb)
    val_ds = LeRobotDataset(cfg["repo_id"], root=root, episodes=val_eps, delta_timestamps=delta, video_backend=vb)
    log.info(f"episodes: train {len(train_eps)} ({train_ds.num_frames} frames), val {len(val_eps)} "
             f"({val_ds.num_frames} frames); state dim {meta.features['observation.state']['shape']}")
    g = torch.Generator(); g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=mb, shuffle=True, num_workers=cfg["num_workers"],
                                     pin_memory=True, drop_last=True, persistent_workers=True,
                                     prefetch_factor=cfg.get("prefetch_factor", 4), generator=g)
    vdl = torch.utils.data.DataLoader(val_ds, batch_size=mb, shuffle=True, num_workers=2, drop_last=True,
                                      generator=torch.Generator().manual_seed(seed + 1))

    # ---------------- optim ----------------
    params = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in policy.parameters())
    log.info(f"trainable params {n_train / 1e6:.1f}M / {n_all / 1e6:.1f}M")
    optim = pcfg.get_optimizer_preset().build(params)
    sched = pcfg.get_scheduler_preset().build(optim, num_training_steps=cfg["steps"])
    amp = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16

    def to_device(batch):
        fi = batch.get("frame_index")
        out = pre(batch)
        if fi is not None and "frame_index" not in out:  # the processor drops it; the progress input needs it
            out["frame_index"] = fi.to(device)
        return out

    extras = cfg.get("extras") or {}
    if extras:
        from oekaki.policy.smolvla_plus import plus_forward, save_extras

    def compute_loss(batch):
        if extras:
            return plus_forward(policy, batch, extras)
        return policy.forward(batch)

    def val_loss():
        policy.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            for i, b in enumerate(vdl):
                if i >= cfg["val_batches"]:
                    break
                try:
                    loss, _ = compute_loss(to_device(b))
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue
                losses.append(loss.item())
        policy.train()
        return float(np.mean(losses))

    json.dump(dict(cfg=cfg, micro_batch=mb, accum=accum, gpu=torch.cuda.get_device_name(0),
                   train_episodes=len(train_eps), val_episodes=len(val_eps)),
              open(out / "run_info.json", "w"), indent=2)
    csv_f = open(out / "train_log.csv", "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["step", "loss", "val_loss", "lr", "grad_norm", "elapsed_s", "peak_mem_gib"])

    def save(tag: str):
        d = out / "checkpoints" / tag
        d.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(d)
        if extras:
            save_extras(d, extras)
        pre.save_pretrained(d)
        post.save_pretrained(d)
        log.info(f"saved {d}")

    policy.train()
    it = iter(dl)
    t0 = time.time()
    run_loss = []
    aux = {}
    torch.cuda.reset_peak_memory_stats()
    def train_step():
        nonlocal it
        optim.zero_grad(set_to_none=True)
        tot, infos = 0.0, []
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = to_device(batch)
            with torch.autocast("cuda", dtype=amp):
                loss, info = compute_loss(batch)
            (loss / accum).backward()
            tot += loss.item() / accum
            infos.append(info)
        return tot, infos

    for step in range(1, cfg["steps"] + 1):
        if margin and step % 50 == 0 and gpu_status()["free"] < 1500:
            # another user is short of memory: hand back our cached blocks and wait
            torch.cuda.empty_cache()
            log.info("[gpu-guard] GPU memory nearly exhausted by other processes -> released cache, waiting")
            wait_for_gpu(required_mib=3000, poll_s=30, log=log.info)
        if cfg.get("ram_pause_mib", 8000) and step % 50 == 0 and ram_available_mib() < cfg.get("ram_pause_mib", 8000):
            # host RAM nearly exhausted (other users or us): pause instead of growing into the OOM killer
            log.info("[ram-guard] host RAM low -> pausing")
            wait_for_ram(cfg.get("ram_pause_mib", 8000) * 2, poll_s=30, log=log.info)
        for attempt in range(20):
            try:
                tot, infos = train_step()
                break
            except torch.OutOfMemoryError:
                optim.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log.info(f"[gpu-guard] OOM at step {step} (attempt {attempt}) -> released memory, waiting")
                wait_for_gpu(required_mib=need + margin, poll_s=60, log=log.info)
        else:
            raise RuntimeError("repeated OOM")
        for info in infos:
            for k, v in info.items():
                if k not in ("loss", "losses_after_forward", "losses_after_in_ep_bound", "losses_after_rm_padding"):
                    aux.setdefault(k, []).append(v)
        gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        optim.step()
        sched.step()
        run_loss.append(tot)
        vl = ""
        if step % cfg["val_interval"] == 0 or step == cfg["steps"]:
            vl = val_loss()
        if step % cfg["log_interval"] == 0 or vl != "":
            el = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            writer.writerow([step, float(np.mean(run_loss)), vl, sched.get_last_lr()[0], float(gn), round(el, 1),
                             round(peak, 2)])
            csv_f.flush()
            log.info(f"step {step}/{cfg['steps']} loss {np.mean(run_loss):.4f} "
                     f"{'val ' + format(vl, '.4f') if vl != '' else ''} lr {sched.get_last_lr()[0]:.2e} "
                     f"gn {float(gn):.2f} {el / step:.2f}s/step peak {peak:.1f}GiB "
                     f"eta {(cfg['steps'] - step) * el / step / 60:.0f}min "
                     + " ".join(f"{k} {np.mean(v):.4f}" for k, v in aux.items()))
            run_loss = []
            aux = {}
        if step % cfg["save_interval"] == 0 and step != cfg["steps"]:
            save(f"{step:06d}")
    save("last")
    csv_f.close()
    info = json.load(open(out / "run_info.json"))
    info["train_time_s"] = time.time() - t0
    info["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(info, open(out / "run_info.json", "w"), indent=2)
    log.info("done")


if __name__ == "__main__":
    main()
