"""Closed-loop rollout evaluation of trained policies (and of the expert as a reference).

For each test condition x shape x episode, the paper height / offset / disturbance are sampled
from a seed that is identical for every agent, so conditions A and B are compared on paired
episodes.

Usage:
  python scripts/evaluate.py --agent A=outputs/train/A/checkpoints/last --agent B=outputs/train/B/checkpoints/last \
      --episodes-per-shape 10 --conditions standard disturb --out artifacts/eval
  python scripts/evaluate.py --agent expert=expert ...
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time

import cv2
import numpy as np

from oekaki.config import load_config, resolve
from oekaki.env.drawing_env import Disturbance, DrawingEnv
from oekaki.eval.metrics import contact_metrics, seam_metrics, shape_fidelity, smoothness
from oekaki.eval.recognize import geometric_classify
from oekaki.expert.shapes import instruction, make_shape

TEST_CONDITIONS = {
    # name: (height range scale, disturbance)
    "standard": dict(height_scale=1.0, disturb=False),
    "disturb": dict(height_scale=1.0, disturb=True),
    "out_of_range": dict(height_scale=1.4, disturb=False),
}


def episode_spec(cond: str, shape: str, i: int, cfg: dict, seed: int) -> dict:
    rng = np.random.default_rng([seed, list(TEST_CONDITIONS).index(cond), ["circle", "square", "triangle", "star", "line"].index(shape), i])
    r = cfg["randomization"]
    tc = TEST_CONDITIONS[cond]
    hr = r["height_range"]
    if tc["height_scale"] > 1.0:  # outside the training band, on either side
        h = rng.choice([-1, 1]) * rng.uniform(hr, hr * tc["height_scale"])
    else:
        h = rng.uniform(-hr, hr)
    spec = dict(height=float(h), xy=rng.uniform(-r["xy_range"], r["xy_range"], 2).tolist(),
                task=instruction(shape, rng), disturb=None, shape_seed=int(rng.integers(2**31)))
    if tc["disturb"]:
        d = r["disturbance"]
        mag = rng.uniform(d["delta_min"], d["delta_max"])
        sign = rng.choice([-1.0, 1.0])
        if abs(h + sign * mag) > hr + 0.005:
            sign = -sign
        spec["disturb"] = dict(delta=float(sign * mag), rate=d["rate"], delay=float(rng.uniform(1.0, 3.0)))
    return spec


def path_air_ratio(tips_xy, phys, contacts) -> float:
    """Distance-weighted share of the pen path (between first and last contact) travelled without touching."""
    idx = np.nonzero(contacts)[0]
    if len(idx) == 0:
        return 1.0
    a, b = idx[0], idx[-1]
    p = np.asarray(tips_xy[a:b + 1])
    if len(p) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    touching = np.asarray(phys[a + 1:b + 1], bool)
    tot = seg.sum()
    return float(seg[~touching].sum() / tot) if tot > 1e-6 else 0.0


def run_episode(env: DrawingEnv, agent, spec: dict, shape: str, max_steps: int, dcfg: dict,
                video_cam: str | None = None, video_size: int = 384) -> dict:
    from oekaki.env.drawing_env import EpisodeParams
    params = EpisodeParams(height_offset=spec["height"], xy_offset=tuple(spec["xy"]))
    if spec["disturb"]:
        params.disturbance = Disturbance(t_start=-1.0, delta=spec["disturb"]["delta"], rate=spec["disturb"]["rate"])
    obs = env.reset(params)
    shp = make_shape(shape, np.random.default_rng(spec["shape_seed"]), {**dcfg, "canvas": env.cfg["canvas"]})
    if hasattr(agent, "expert"):
        strokes_world = [s + env.data.mocap_pos[env.paper_mocap, :2] for s in shp["strokes"]]
        agent.reset(spec["task"], obs, strokes_world)
    else:
        agent.reset(spec["task"], obs)
    contacts, comps, phys, tips, actions, frames, paper_z = [], [], [], [], [], [], []
    first_contact_t = None
    t_inf = 0.0
    for t in range(max_steps):
        t0 = time.time()
        a = agent.act(obs)
        t_inf += time.time() - t0
        actions.append(a)
        if obs["contact"] and first_contact_t is None:
            first_contact_t = env.t
            if params.disturbance is not None:
                params.disturbance.t_start = env.t + spec["disturb"]["delay"]
        contacts.append(obs["contact"]); comps.append(obs["compression"]); phys.append(obs["physical_contact"])
        tips.append(obs["tip_pos"][:2].copy()); paper_z.append(obs["paper_z"])
        if video_cam is not None:
            frames.append(_video_frame(env, obs, video_size))
        obs = env.step(a)
        if hasattr(agent, "expert") and agent.expert.done:
            break
    contacts, comps = np.array(contacts), np.array(comps)
    m = contact_metrics(contacts, comps, env.travel)
    m["path_air_ratio"] = path_air_ratio(tips, phys, contacts)
    m.update(smoothness(np.array(actions), env.fps))
    m.update(seam_metrics(np.array(actions), getattr(agent, "seams", [])))
    m.update(shape_fidelity(env.canvas.img, shp["strokes"], env.cfg["canvas"]["size"]))
    g = geometric_classify(env.canvas.img)
    m.update(geo_pred=g["pred"], n_fragments=g["n_fragments"], ink_px=int(env.canvas.ink_mask().sum()),
             max_compression_mm=float(comps.max() * 1e3), inference_s_per_step=t_inf / max(len(actions), 1),
             first_contact_t=first_contact_t, steps=len(actions))
    return dict(metrics=m, canvas=env.canvas.img.copy(), frames=frames,
                trace=dict(contact=contacts.tolist(), compression=comps.tolist(), paper_z=paper_z))


def _video_frame(env, obs, size):
    top = cv2.resize(obs["images"]["top"], (size, size), interpolation=cv2.INTER_LINEAR)
    wrist = cv2.resize(obs["images"]["wrist"], (size // 2, size // 2))
    canvas = cv2.resize(env.canvas.img, (size // 2, size // 2), interpolation=cv2.INTER_AREA)
    right = np.vstack([wrist, canvas])
    return np.hstack([top, right])


def make_agent(name: str, path: str, env, dcfg, n_action_steps, ensemble_every: int = 0, **infer):
    if path == "expert":
        from oekaki.eval.policy import ExpertAgent
        return ExpertAgent(env, dcfg)
    from oekaki.eval.policy import SmolVLAAgent
    return SmolVLAAgent(resolve(path), n_action_steps=n_action_steps, ensemble_every=ensemble_every, **infer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="append", required=True, help="name=checkpoint_dir | name=expert")
    ap.add_argument("--episodes-per-shape", type=int, default=10)
    ap.add_argument("--shapes", nargs="+", default=["circle", "square", "triangle", "star", "line"])
    ap.add_argument("--conditions", nargs="+", default=["standard", "disturb"])
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--ensemble-every", type=int, default=5, help=">0: temporal ensembling (replan period); 0: plain chunking")
    ap.add_argument("--mode", choices=["ensemble", "chunk", "rtc", "ttrtc"], default=None,
                    help="inference scheme (overrides --ensemble-every unless 'ensemble'); see SmolVLAAgent")
    ap.add_argument("--replan", type=int, default=5, help="chunk modes: steps between two chunk predictions")
    ap.add_argument("--delay", type=int, default=0, help="chunk modes: simulated inference latency (control steps)")
    ap.add_argument("--rtc-horizon", type=int, default=20)
    ap.add_argument("--rtc-beta", type=float, default=5.0)
    ap.add_argument("--noise-scale", type=float, default=1.0, help="std of the initial flow noise (temperature)")
    ap.add_argument("--required-gpu-mib", type=float, default=6000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--video-per-cell", type=int, default=1, help="episodes per (cond, shape) saved as video")
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the results file")
    args = ap.parse_args()

    from oekaki.utils.gpu import wait_for_gpu, wait_for_ram
    wait_for_ram(12000)  # host RAM is shared with other projects
    st = wait_for_gpu(required_mib=args.required_gpu_mib)
    print(f"[gpu-guard] free {st['free']:.0f} MiB; procs {[(p['pid'], p['mem']) for p in st['procs']]}")

    cfg, dcfg = load_config(), load_config("configs/datagen.yaml")
    out = resolve(args.out)
    env = DrawingEnv(cfg, render=True)
    max_steps = int(args.max_seconds * env.fps)
    results = []
    for spec_s in args.agent:
        name, path = spec_s.split("=", 1)
        infer = {}
        ens = args.ensemble_every
        if path != "expert":
            infer = dict(noise_scale=args.noise_scale)
        if args.mode and args.mode != "ensemble":
            ens = 0
            infer.update(mode=args.mode, replan=args.replan, delay=args.delay, rtc_horizon=args.rtc_horizon,
                         rtc_beta=args.rtc_beta)
        agent = make_agent(name, path, env, dcfg, args.n_action_steps, ens, **infer)
        infer_desc = "expert" if path == "expert" else (
            f"ensemble/{ens}" if ens > 0 else f"{args.mode}/r{args.replan}/d{args.delay}" if args.mode else "select")
        if path != "expert":
            infer_desc += f"/n{args.noise_scale:g}"
        adir = out / name
        (adir / "canvas").mkdir(parents=True, exist_ok=True)
        (adir / "video").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for cond in args.conditions:
            for shape in args.shapes:
                for i in range(args.episodes_per_shape):
                    spec = episode_spec(cond, shape, i, cfg, args.seed)
                    vid = i < args.video_per_cell
                    ep = run_episode(env, agent, spec, shape, max_steps, dcfg, video_cam="top" if vid else None)
                    key = f"{cond}_{shape}_{i:02d}"
                    cv2.imwrite(str(adir / "canvas" / f"{key}.png"), ep["canvas"][..., ::-1])
                    if vid:
                        import imageio.v2 as imageio
                        imageio.mimwrite(adir / "video" / f"{key}.mp4", ep["frames"], fps=env.fps, quality=7,
                                         macro_block_size=1)
                        json.dump(ep["trace"], open(adir / "video" / f"{key}_trace.json", "w"))
                    row = dict(agent=name, cond=cond, shape=shape, ep=i, task=spec["task"], height=spec["height"],
                               disturb=spec["disturb"], infer=infer_desc, **ep["metrics"])
                    results.append(row)
                    print(f"[{name}] {key} air {row['air_ratio']:.3f} path_air {row['path_air_ratio']:.3f} "
                          f"frag {row['n_fragments']} geo {row['geo_pred']} ink {row['ink_px']} "
                          f"chamfer {row['chamfer_mm']:.2f}mm jerk {row['jerk']:.0f} seam {row['seam_jump']:.4f}/{row['step_jump']:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        del agent
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

    if not args.no_clip:
        from oekaki.eval.recognize import ClipRecognizer
        clip = ClipRecognizer()
        canv = [cv2.imread(str(out / r["agent"] / "canvas" / f"{r['cond']}_{r['shape']}_{r['ep']:02d}.png"))[..., ::-1]
                for r in results]
        for r, p in zip(results, clip(canv)):
            r["clip_pred"] = p["pred"]
            r["clip_probs"] = p["probs"]
    tag = f"_{args.tag}" if args.tag else ""
    json.dump(results, open(out / f"results_{'_'.join(a.split('=')[0] for a in args.agent)}{tag}.json", "w"), indent=1)
    env.close()
    print("saved", out)


if __name__ == "__main__":
    main()
