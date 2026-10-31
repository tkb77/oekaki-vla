"""sample the trained generator, post-process, score with the judge CNN, and save
  results/goal/generator/samples_raw.png      raw 64x64 samples (rows = categories)
  results/goal/generator/samples_post.png     after post-processing, redrawn by the common renderer
  results/goal/generator/judge.json           judge accuracy per category (raw and post-processed), empty outputs
  results/goal/generator/trajectory_<cat>.png the DDIM trajectory (x_t and predicted x0) of one sample per category

Usage: python scripts/sample_generator.py --ckpt outputs/generator/last.pt --judge outputs/classifier/judge.pt [--n 200]
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch

from oekaki.config import resolve, resolve_ckpt
from oekaki.generator.imagine import Imaginer
from oekaki.generator.model import from_model, sample_ddim
from oekaki.generator.postprocess import postprocess, render_generated, to_unit
from oekaki.layout import centred


def tile(imgs, nrow, pad=1):
    rows = [np.hstack([np.pad(im, pad, constant_values=160) for im in imgs[r * nrow:(r + 1) * nrow]])
            for r in range(int(np.ceil(len(imgs) / nrow)))]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/generator/last.pt", help="last.pt or Hub repo id")
    ap.add_argument("--judge", default="outputs/classifier/judge.pt")
    ap.add_argument("--n", type=int, default=200, help="samples per category for the judge accuracy")
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/goal/generator")
    args = ap.parse_args()
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    im = Imaginer(resolve_ckpt(args.ckpt, "last.pt"))
    s = im.scfg
    guidance = args.guidance if args.guidance is not None else s["guidance"]
    judge = None
    if resolve(args.judge).exists():
        from oekaki.classifier.model import Judge
        judge = Judge(resolve(args.judge), im.device)
    raws, posts, stats = [], [], {}
    fill = im.ck.get("fill", 0.8)
    for ci, cat in enumerate(im.classes):
        g = torch.Generator(device=im.device).manual_seed(args.seed + ci)
        lab = torch.full((args.n,), ci, device=im.device, dtype=torch.long)
        imgs = []
        for k in range(0, args.n, 100):
            x = sample_ddim(im.model, lab[k:k + 100], im.size, im.sch, s["ddim_steps"], guidance, s["eta"], generator=g)
            imgs.append(from_model(x))
        imgs = np.concatenate(imgs)
        post = []
        for a in imgs:
            segs = postprocess(a, im.pcfg["thresh"], im.pcfg["min_area"])
            post.append(render_generated(to_unit(segs), im.size, centred(fill)) if segs else np.full_like(a, 255))
        post = np.stack(post)
        raws.append(imgs[:12]); posts.append(post[:12])
        st = dict(empty=int((post.min(axis=(1, 2)) > 128).sum()))
        if judge is not None:
            pr, pp = judge.probs(imgs), judge.probs(post)
            st.update(judge_acc_raw=float((pr.argmax(1) == judge.classes.index(cat)).mean()),
                      judge_acc_post=float((pp.argmax(1) == judge.classes.index(cat)).mean()))
        stats[cat] = st
        print(cat, st, flush=True)
        # one trajectory for the demo
        res = im.generate(cat, seed=args.seed + 1000 + ci, frames=True, guidance=guidance)
        traj = res["trajectory"]
        idx = np.linspace(0, len(traj) - 1, 10).round().astype(int)
        cv2.imwrite(str(out / f"trajectory_{cat.replace(' ', '_')}.png"),
                    np.vstack([tile([traj[k][0] for k in idx], 10), tile([traj[k][1] for k in idx], 10)]))
    cv2.imwrite(str(out / "samples_raw.png"), np.vstack([tile(r, 12) for r in raws]))
    cv2.imwrite(str(out / "samples_post.png"), np.vstack([tile(p, 12) for p in posts]))
    if judge is not None:
        stats["mean"] = dict(judge_acc_raw=float(np.mean([v["judge_acc_raw"] for v in stats.values()])),
                             judge_acc_post=float(np.mean([v["judge_acc_post"] for k, v in stats.items() if k != "mean"])))
    json.dump(dict(stats, guidance=guidance, ddim_steps=s["ddim_steps"], n=args.n), open(out / "judge.json", "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
