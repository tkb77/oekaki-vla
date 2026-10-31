"""Imagine-then-draw demo: instruction -> generator -> goal image -> SmolVLA (model G)
draws it in MuJoCo. Saves the generated image, the DDIM trajectory, the goal, the final canvas, metrics and a
demo video laid out as follows:
  left: generation (noise -> line drawing, then the goal on the paper frame); right: the robot (overhead camera);
  bottom: the instruction; at the end the goal and the finished drawing side by side.

Usage:
  python scripts/imagine_and_draw.py --prompt "draw a star" --seed 0 --out outputs/demo_star \
      [--policy outputs/train/goal_G/checkpoints/last] [--generator outputs/generator/last.pt] \
      [--size 0.6 --center 0.5 0.5]     # optional fixed layout (fraction of the drawable area, normalised centre)
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json

import cv2
import imageio.v2 as imageio
import numpy as np

from oekaki.config import load_config, resolve, resolve_ckpt
from oekaki.data.goal import goal_image, load_goal_config
from oekaki.env.drawing_env import DrawingEnv, EpisodeParams
from oekaki.eval.goal_metrics import goal_metrics
from oekaki.generator.imagine import Imaginer, parse_category
from oekaki.layout import affine, drawable_box, sample_layout
from oekaki.render import apply_affine

V = 384  # panel size of the video


def caption(img, text, y, scale=0.7, color=(30, 30, 30)):
    cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def panel(gray_or_rgb, title=None):
    im = gray_or_rgb if gray_or_rgb.ndim == 3 else np.repeat(gray_or_rgb[..., None], 3, -1)
    im = cv2.resize(im, (V, V), interpolation=cv2.INTER_NEAREST if im.shape[0] <= 128 else cv2.INTER_AREA)
    if title:
        cv2.rectangle(im, (0, 0), (V, 28), (255, 255, 255), -1)
        cv2.putText(im, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return im


def compose(left, right, text):
    bar = np.full((52, 2 * V, 3), 255, np.uint8)
    caption(bar, f'"{text}"', 35)
    return np.vstack([np.hstack([left, right]), bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--category", default=None, help="override the category parsed from the prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="outputs/train/goal_G/checkpoints/last", help="checkpoint dir or Hub repo id")
    ap.add_argument("--generator", default="outputs/generator/last.pt", help="last.pt or Hub repo id")
    ap.add_argument("--size", type=float, default=None)
    ap.add_argument("--center", type=float, nargs=2, default=None)
    ap.add_argument("--paper-height", type=float, default=None, help="m, default: random from the seed")
    ap.add_argument("--disturb", type=float, default=None, help="m, move the paper by this much while drawing")
    ap.add_argument("--max-seconds", type=float, default=40.0)
    ap.add_argument("--out", default="outputs/demo")
    args = ap.parse_args()
    out = resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    env_cfg, lcfg, gcfg, icfg = load_config(), load_config("configs/layout.yaml"), load_goal_config(), load_config("configs/eval_goal.yaml")
    S = env_cfg["canvas"]["size"]
    rng = np.random.default_rng(args.seed)

    # 1-2: instruction -> category -> generated image (with trajectory)
    im = Imaginer(resolve_ckpt(args.generator, "last.pt"))
    cat = args.category or parse_category(args.prompt, im.classes)
    if cat is None:
        raise SystemExit(f"no known category in {args.prompt!r} (known: {im.classes}); pass --category")
    gen = im.generate(cat, seed=args.seed, frames=True)
    if not gen["unit"]:
        raise SystemExit("the generator produced an empty image; try another seed")
    # 3: layout -> goal image on the paper frame
    if args.size is not None:
        lo, hi = drawable_box(S, lcfg["paper_margin"])
        c = lo + np.asarray(args.center or (0.5, 0.5)) * (hi - lo)
        T, lay = affine(args.size * (hi - lo), 0.0, False, c), dict(size_frac=args.size, center=list(c))
    else:
        T, lay = sample_layout(gen["unit"], rng, lcfg, S)
    goal = goal_image([apply_affine(s, T) for s in gen["unit"]], gcfg["goal"]["size"])
    cv2.imwrite(str(out / "generated.png"), gen["image"])
    cv2.imwrite(str(out / "goal.png"), goal[..., 0])

    # 4: model G draws it
    from oekaki.eval.policy import SmolVLAAgent
    inf = icfg["inference"]
    agent = SmolVLAAgent(resolve_ckpt(args.policy), mode=inf["mode"], replan=inf["replan"], delay=inf["delay"],
                         noise_scale=inf["noise_scale"])
    env = DrawingEnv(env_cfg, render=True)
    r = env_cfg["randomization"]
    h = args.paper_height if args.paper_height is not None else float(rng.uniform(-r["height_range"], r["height_range"]))
    params = EpisodeParams(height_offset=h, xy_offset=(0.0, 0.0))
    if args.disturb:
        from oekaki.env.drawing_env import Disturbance
        params.disturbance = Disturbance(-1.0, args.disturb, r["disturbance"]["rate"])
    obs = env.reset(params)
    agent.reset(args.prompt, obs, goal=goal)
    renderer_frames = []
    first = None
    for t in range(int(args.max_seconds * env.fps)):
        a = agent.act(obs)
        if obs["contact"] and first is None:
            first = env.t
            if params.disturbance is not None:
                params.disturbance.t_start = env.t + 2.0
        renderer_frames.append(obs["images"]["top"].copy())
        obs = env.step(a)
    canvas = env.canvas.img.copy()
    env.close()
    cv2.imwrite(str(out / "canvas.png"), canvas[..., ::-1])
    gm = goal_metrics(goal, canvas)
    json.dump(dict(prompt=args.prompt, category=cat, seed=args.seed, layout=lay, paper_height=h, **gm),
              open(out / "result.json", "w"), indent=1)

    # 5: demo video
    frames = []
    idle = panel(renderer_frames[0], "robot (overhead camera)")
    for xt, x0 in gen["trajectory"]:
        frames.append(compose(panel(xt, "imagining ... (diffusion)"), idle, args.prompt))
    for _ in range(20):
        frames.append(compose(panel(gen["image"], "generated drawing"), idle, args.prompt))
    for _ in range(20):
        frames.append(compose(panel(goal, "goal (placed on the paper)"), idle, args.prompt))
    gp = panel(goal, "goal")
    for f in renderer_frames:
        frames.append(compose(gp, panel(f, "robot (overhead camera)"), args.prompt))
    end = compose(panel(goal, "goal"), panel(canvas, f"drawn (IoU {gm['dilated_iou']:.2f})"), args.prompt)
    frames += [end] * 60
    imageio.mimwrite(out / "demo.mp4", frames, fps=30, quality=8, macro_block_size=1)  # frames are RGB
    print(json.dumps(dict(category=cat, **gm)), "->", out)


if __name__ == "__main__":
    main()
