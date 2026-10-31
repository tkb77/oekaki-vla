"""Train the class-conditional DDPM on Quick, Draw! drawings rendered by the common
renderer. Runs on any single CUDA GPU (or CPU for a smoke test); saves checkpoints (model + EMA), sample grids
and a loss log to <output_dir>.

Usage: python scripts/train_generator.py [--config configs/generator.yaml] [--steps N] [--output-dir DIR] [--resume CKPT]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from oekaki.config import load_config, resolve
from oekaki.generator.data import rendered_split
from oekaki.generator.model import EMA, SmallConditionalUNet, from_model, make_schedule, q_sample, sample_ddim, to_model


def grid(imgs: np.ndarray, nrow: int) -> np.ndarray:
    n, h, w = imgs.shape
    rows = [np.hstack([np.pad(im, 1, constant_values=160) for im in imgs[r * nrow:(r + 1) * nrow]])
            for r in range(int(np.ceil(n / nrow)))]
    return np.vstack(rows)


def save_samples(model, sch, cats, gcfg, path, device, n_per=8, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    labels = torch.arange(len(cats), device=device).repeat_interleave(n_per)
    s = gcfg["sample"]
    x = sample_ddim(model, labels, gcfg["image_size"], sch, s["ddim_steps"], s["guidance"], s["eta"], generator=g)
    cv2.imwrite(str(path), grid(from_model(x), n_per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generator.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    gcfg = load_config(args.config)
    tc = gcfg["train"]
    if args.steps:
        tc["steps"] = args.steps
        tc["sample_every"] = min(tc["sample_every"], args.steps)
        tc["save_every"] = min(tc["save_every"], args.steps)
    if args.batch_size:
        tc["batch_size"] = args.batch_size
    out = resolve(args.output_dir or gcfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    seed = gcfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = tc.get("amp") == "bf16" and device == "cuda" and torch.cuda.is_bf16_supported()

    imgs, labels, cats = rendered_split(gcfg, "train")
    print(f"train images {imgs.shape}, classes {cats}, device {device}, bf16 {use_bf16}", flush=True)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(imgs), torch.from_numpy(labels))
    g = torch.Generator(); g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True, generator=g,
                                     num_workers=tc["num_workers"], pin_memory=device == "cuda",
                                     persistent_workers=tc["num_workers"] > 0)
    mc = gcfg["model"]
    model = SmallConditionalUNet(len(cats), mc["base_channels"]).to(device)
    ema = EMA(model, tc["ema_decay"])
    sch = make_schedule(mc["timesteps"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tc["warmup_steps"]))
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"]); step = ck["step"]
        for _ in range(step):
            sched.step()
    n_params = sum(p.numel() for p in model.parameters())
    json.dump(dict(gcfg, n_params=n_params, device=device), open(out / "config.json", "w"), indent=1)
    print(f"params {n_params / 1e6:.2f}M", flush=True)
    logf = open(out / "train_log.csv", "a", newline="")
    logw = csv.writer(logf)
    if step == 0:
        logw.writerow(["step", "loss", "lr", "sec_per_step"])
    it = iter(dl)
    t0, run_loss = time.time(), []
    model.train()
    while step < tc["steps"]:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x0 = to_model(x).to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y = y.masked_fill(torch.rand(y.shape, device=device) < tc["cfg_drop_prob"], model.null_label)
        t = torch.randint(0, mc["timesteps"], (len(y),), device=device)
        noise = torch.randn_like(x0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            pred = model(q_sample(x0, t, noise, sch), t, y)
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step(); sched.step(); ema.update(model)
        step += 1
        run_loss.append(loss.item())
        if step % tc["log_every"] == 0 or step == tc["steps"]:
            sps = (time.time() - t0) / len(run_loss)
            logw.writerow([step, np.mean(run_loss), sched.get_last_lr()[0], sps]); logf.flush()
            print(f"step {step}/{tc['steps']} loss {np.mean(run_loss):.4f} {sps:.3f}s/step "
                  f"eta {(tc['steps'] - step) * sps / 60:.0f} min", flush=True)
            t0, run_loss = time.time(), []
        if step % tc["sample_every"] == 0 or step == tc["steps"]:
            save_samples(ema.model, sch, cats, gcfg, out / f"samples_{step:07d}.png", device)
        if step % tc["save_every"] == 0 or step == tc["steps"]:
            ck = dict(model=model.state_dict(), ema=ema.model.state_dict(), optimizer=opt.state_dict(), step=step,
                      classes=cats, image_size=gcfg["image_size"], fill=gcfg["fill"], timesteps=mc["timesteps"],
                      base_channels=mc["base_channels"], sample=gcfg["sample"], postprocess=gcfg["postprocess"])
            torch.save(ck, out / "last.pt")
            print(f"saved {out / 'last.pt'} (step {step})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
