"""Showcase videos: replay identical test episodes with several agents and render them side by side
in high resolution with overlays (instruction, contact switch, compression, paper height).

Usage:
  python scripts/render_video.py --agent A=outputs/train/A/checkpoints/last --agent B=outputs/train/B/checkpoints/last \
      --cond disturb --shapes star circle --ep 0 --out artifacts/videos
An agent spec may carry an inference scheme and a display label:
  --agent "naive=outputs/train/smooth_B/checkpoints/last|mode=chunk,replan=5,noise_scale=1|B: plain SmolVLA"
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import episode_spec, make_agent  # noqa: E402

from oekaki.config import load_config, resolve  # noqa: E402
from oekaki.env.drawing_env import Disturbance, DrawingEnv, EpisodeParams  # noqa: E402
from oekaki.expert.shapes import make_shape  # noqa: E402

W, H = 640, 480
LABELS = {"A": "A: joints + contact sensor", "B": "B: joints only", "expert": "scripted expert"}
COL = {"A": (214, 120, 42), "B": (52, 104, 235), "expert": (122, 175, 27)}  # BGR


def showcase_camera(cfg):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [cfg["canvas"]["center"][0] - 0.01, 0.0, cfg["canvas"]["height"] + 0.02]
    cam.distance, cam.azimuth, cam.elevation = 0.46, 105.0, -42.0
    return cam


def put(img, text, org, scale=0.6, color=(30, 30, 30), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def parse_agent(s):
    """name=path[|key=val,...[|label]] -> name, path, inference kwargs, label"""
    parts = s.split("|")
    name, path = parts[0].split("=", 1)
    infer = {}
    if len(parts) > 1 and parts[1]:
        for kv in parts[1].split(","):
            k, v = kv.split("=")
            infer[k] = v if k == "mode" else (float(v) if "." in v or k == "noise_scale" else int(v))
    return name, path, infer, parts[2] if len(parts) > 2 else LABELS.get(name, name)


def run(env, agent, name, spec, shape, dcfg, max_steps, renderer, cam, z0, label=None):
    params = EpisodeParams(height_offset=spec["height"], xy_offset=tuple(spec["xy"]))
    if spec["disturb"]:
        params.disturbance = Disturbance(-1.0, spec["disturb"]["delta"], spec["disturb"]["rate"])
    obs = env.reset(params)
    if hasattr(agent, "expert"):
        shp = make_shape(shape, np.random.default_rng(spec["shape_seed"]), {**dcfg, "canvas": env.cfg["canvas"]})
        agent.reset(spec["task"], obs, [s + env.data.mocap_pos[env.paper_mocap, :2] for s in shp["strokes"]])
    else:
        agent.reset(spec["task"], obs)
    frames, first = [], None
    for t in range(max_steps):
        a = agent.act(obs)
        if obs["contact"] and first is None:
            first = env.t
            if params.disturbance is not None:
                params.disturbance.t_start = env.t + spec["disturb"]["delay"]
        renderer._gl_context.make_current()
        mujoco.mjr_uploadTexture(env.model, renderer._mjr_context, env.tex_id)
        renderer.update_scene(env.data, cam)
        img = renderer.render()[..., ::-1].copy()
        # overlays
        cv2.rectangle(img, (0, 0), (W, 34), (255, 255, 255), -1)
        put(img, label or LABELS.get(name, name), (10, 24), 0.65, COL.get(name, (0, 0, 0)), 2)
        put(img, f'"{spec["task"]}"', (10, 60), 0.6)
        on = obs["contact"] == 1
        cv2.circle(img, (W - 120, 52), 10, (40, 180, 40) if on else (190, 190, 190), -1)
        put(img, "contact", (W - 102, 58), 0.5)
        comp = max(obs["compression"], 0) * 1e3
        cv2.rectangle(img, (W - 120, 72), (W - 20, 84), (220, 220, 220), -1)
        cv2.rectangle(img, (W - 120, 72), (W - 120 + int(min(comp, 10) * 10), 84), (60, 60, 200), -1)
        put(img, f"{comp:4.1f} mm", (W - 120, 102), 0.45)
        dz = (env.paper_z - z0) * 1e3
        put(img, f"paper height {dz:+5.1f} mm", (10, H - 14), 0.55,
            (0, 0, 200) if params.disturbance is not None and abs(dz - spec["height"] * 1e3) > 0.5 else (30, 30, 30))
        put(img, f"t = {env.t:4.1f} s", (W - 110, H - 14), 0.5)
        canvas = cv2.resize(env.canvas.img[..., ::-1], (150, 150), interpolation=cv2.INTER_AREA)
        img[H - 190:H - 40, 10:160] = canvas
        cv2.rectangle(img, (10, H - 190), (160, H - 40), (120, 120, 120), 1)
        frames.append(img)
        obs = env.step(a)
        if hasattr(agent, "expert") and agent.expert.done:
            break
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="append", required=True)
    ap.add_argument("--cond", default="disturb")
    ap.add_argument("--shapes", nargs="+", default=["star"])
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-seconds", type=float, default=18.0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor (written in the video)")
    ap.add_argument("--out", default="artifacts/videos")
    args = ap.parse_args()
    cfg, dcfg = load_config(), load_config("configs/datagen.yaml")
    env = DrawingEnv(cfg, render=True)
    renderer = mujoco.Renderer(env.model, H, W)
    cam = showcase_camera(cfg)
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    agents = []
    for s in args.agent:
        name, path, infer, label = parse_agent(s)
        ens = 0 if infer.get("mode", "ensemble") != "ensemble" else 5
        infer.pop("mode", None) if ens else None
        agents.append((name, make_agent(name, path, env, dcfg, 10, ensemble_every=ens, **infer), label))
    max_steps = int(args.max_seconds * env.fps)
    for shape in args.shapes:
        spec = episode_spec(args.cond, shape, args.ep, cfg, args.seed)
        z0 = env.paper_nominal[2]
        clips = [run(env, ag, name, spec, shape, dcfg, max_steps, renderer, cam, z0, label)
                 for name, ag, label in agents]
        n = max(len(c) for c in clips)
        clips = [c + [c[-1]] * (n - len(c)) for c in clips]
        frames = [np.hstack(f)[..., ::-1] for f in zip(*clips)]
        if args.speed != 1.0:
            for f in frames:
                put(f, f"x{args.speed:g} speed", (f.shape[1] // 2 - 50, H - 14), 0.55)
        tag = "_vs_".join(n for n, *_ in agents)
        path = out / f"{args.cond}_{shape}_{args.ep:02d}_{tag}.mp4"
        imageio.mimwrite(path, frames, fps=env.fps * args.speed, quality=8, macro_block_size=1)
        print("wrote", path)
    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
