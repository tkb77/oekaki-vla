"""Check a training environment (e.g. the docker image) without training: load dataset goal_v1 with the policy's delta
timestamps (decodes all three cameras), build SmolVLA from the Hugging Face cache and run one loss forward on the CPU.

Usage (inside the image, repository mounted at /repo):
  HF_HUB_OFFLINE=1 PYTHONPATH=/repo /opt/dvla/.venv/bin/python /repo/scripts/dev/check_image.py [--config configs/train_goal_G.yaml]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import torch

from oekaki.config import load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_goal_G.yaml")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("train_smolvla", resolve("scripts/train_smolvla.py"))
    ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    cfg = load_config(args.config)
    root = resolve(cfg["dataset_root"])
    meta = LeRobotDatasetMetadata(cfg["repo_id"], root=root)
    t0 = time.time()
    policy, pre, _, pcfg = ts.build_policy(cfg, meta, "cpu")
    print(f"policy built in {time.time() - t0:.0f} s; inputs {sorted(pcfg.input_features)}")
    ds = LeRobotDataset(cfg["repo_id"], root=root, episodes=[0, 1], delta_timestamps=resolve_delta_timestamps(pcfg, meta),
                        video_backend=os.environ.get("OEKAKI_VIDEO_BACKEND") or None)
    item = ds[10]
    print("decoded:", {k: tuple(v.shape) for k, v in item.items() if k.startswith("observation.images")},
          "backend", ds._video_backend)
    batch = torch.utils.data.default_collate([item])
    batch = pre(batch)
    policy.train()
    out = policy.forward(batch)
    loss = out[0] if isinstance(out, tuple) else out
    print("loss", float(loss), "torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
