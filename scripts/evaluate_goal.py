"""Goal-image closed-loop evaluation E1-E5. Every agent sees the same episodes: drawing (held-out
Quick, Draw! test split), layout, instruction and paper height come from one seed per (setting, category, i).

  E1 true goal, named instruction                  N, G (+ expert as reference)
  E2 generated goal (generator output), named       G
  E3 true goal at fixed positions / sizes, named    N, G
  E4 true goal, instruction without category        N, G
  E5 two drawings side by side, no category         G

Per episode: goal-vs-canvas metrics (oekaki.eval.goal_metrics), judge-CNN class of the canvas, contact / air
ratios, jerk; the canvas and goal are saved as PNG. Results: <out>/<setting>/results_<agent>.jsonl.

Usage:
  python scripts/evaluate_goal.py --agent G=outputs/train/goal_G/checkpoints/last --agent N=outputs/train/goal_N/checkpoints/last \
      --agent expert=expert --settings E1 E2 E3 E4 E5 --generator outputs/generator/last.pt --judge outputs/classifier/judge.pt
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from oekaki.config import load_config, resolve, resolve_ckpt
from oekaki.data.goal import goal_image, lay_out, load_sets, load_goal_config
from oekaki.env.drawing_env import DrawingEnv, EpisodeParams
from oekaki.eval.goal_metrics import goal_metrics
from oekaki.eval.metrics import contact_metrics, smoothness
from oekaki.expert.shapes import INSTRUCTION_TEMPLATES
from oekaki.layout import affine, drawable_box, sample_layout
from oekaki.render import apply_affine, norm_to_paper

SETTINGS = ["E1", "E2", "E3", "E4", "E5"]


def plan_episode(setting: str, cat: str, i: int, ctx: dict) -> dict:
    """Deterministic episode content (shared by all agents)."""
    ecfg, cats, S = ctx["ecfg"], ctx["cats"], ctx["S"]
    st = ecfg["settings"][setting]
    rng = np.random.default_rng([ecfg["seed"], SETTINGS.index(setting), cats.index(cat), i])
    # the same test drawing for the same (category, i) in every setting -> E1/E3/E4 are paired
    sketch_idx = int(ctx["perm"][cat][i])
    sketch = ctx["sets"][cat][sketch_idx]
    r = ctx["env_cfg"]["randomization"]
    height = float(rng.uniform(-r["height_range"], r["height_range"]))
    xy = rng.uniform(-r["xy_range"], r["xy_range"], 2).tolist()
    lo, hi = drawable_box(S, ctx["lcfg"]["paper_margin"])
    ep = dict(setting=setting, category=cat, i=i, height=height, xy=xy, sketch_idx=sketch_idx, strokes_local=None)
    if st["goal"] == "true_goal":
        if st.get("layout") == "fixed":
            fl = ecfg["fixed_layouts"][i % len(ecfg["fixed_layouts"])]
            c = lo + np.asarray(fl["center"]) * (hi - lo)
            lay = lay_out(sketch, rng, ctx["lcfg"], S, transform=affine(fl["size"] * (hi - lo), 0.0, False, c))
            lay["layout"].update(size_frac=fl["size"], center=fl["center"])
        else:
            lay = lay_out(sketch, rng, ctx["lcfg"], S)
        ep.update(strokes_norm=lay["strokes_norm"], strokes_local=lay["strokes_local"], layout=lay["layout"])
    elif st["goal"] == "generated":
        gen = ctx["imaginer"].generate(cat, seed=int(rng.integers(2**31)))
        if not gen["unit"]:
            ep.update(strokes_norm=[], layout=None, gen_failed=True)
        else:
            T, params = sample_layout(gen["unit"], rng, ctx["lcfg"], S)
            ep.update(strokes_norm=[apply_affine(s, T) for s in gen["unit"]], layout=params, gen_seed=gen["seed"],
                      gen_image=gen["image"])
    elif st["goal"] == "pair":
        other = [c for c in cats if c != cat][int(rng.integers(len(cats) - 1))]
        sk2 = ctx["sets"][other][int(ctx["perm"][other][-1 - i])]
        norm, local = [], []
        pair = [(sketch, 0.27), (sk2, 0.73)] if rng.random() < 0.5 else [(sketch, 0.73), (sk2, 0.27)]
        for sk, cx in pair:
            c = lo + np.array([cx, 0.5]) * (hi - lo)
            T = affine(ecfg["pair_size"] * (hi - lo), 0.0, False, c)
            n = [apply_affine(s, T) for s in sk]
            norm += n; local += [norm_to_paper(s, S) for s in n]
        ep.update(strokes_norm=norm, strokes_local=local, layout=dict(pair=[cat, other]), category2=other)
    ep["named"] = bool(st["named"])
    ep["task"] = (INSTRUCTION_TEMPLATES[int(rng.integers(len(INSTRUCTION_TEMPLATES)))].format(obj=cat) if st["named"]
                  else "draw the picture shown")
    ep["goal"] = goal_image(ep["strokes_norm"], ctx["gcfg"]["goal"]["size"])
    return ep


def rollout(env, agent, ep: dict, max_steps: int, is_expert: bool, video: bool = False) -> dict:
    params = EpisodeParams(height_offset=ep["height"], xy_offset=tuple(ep["xy"]))
    obs = env.reset(params)
    if is_expert:
        paper = env.data.mocap_pos[env.paper_mocap, :2].copy()
        agent.reset(ep["task"], obs, [s + paper for s in ep["strokes_local"]])
    else:
        agent.reset(ep["task"], obs, goal=ep["goal"])
    contacts, comps, phys, tips, actions, frames = [], [], [], [], [], []
    for t in range(max_steps):
        a = agent.act(obs)
        actions.append(a)
        contacts.append(obs["contact"]); comps.append(obs["compression"]); phys.append(obs["physical_contact"])
        tips.append(obs["tip_pos"][:2].copy())
        if video:
            frames.append(np.hstack([obs["images"]["top"], cv2.resize(ep["goal"], obs["images"]["top"].shape[:2][::-1]),
                                     cv2.resize(env.canvas.img, obs["images"]["top"].shape[:2][::-1],
                                                interpolation=cv2.INTER_AREA)]))
        obs = env.step(a)
        if is_expert and agent.expert.done:
            break
    from evaluate import path_air_ratio
    m = contact_metrics(np.array(contacts), np.array(comps), env.travel)
    m["path_air_ratio"] = path_air_ratio(tips, phys, np.array(contacts))
    m.update(smoothness(np.array(actions), env.fps))
    m.update(steps=len(actions), ink_px=int(env.canvas.ink_mask().sum()))
    return dict(metrics=m, canvas=env.canvas.img.copy(), frames=frames)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="append", required=True, help="name=checkpoint_dir (or Hub repo id) | expert=expert")
    ap.add_argument("--settings", nargs="+", default=["E1", "E2", "E3", "E4"])
    ap.add_argument("--config", default="configs/eval_goal.yaml")
    ap.add_argument("--episodes-per-category", type=int, default=None)
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--generator", default="outputs/generator/last.pt", help="last.pt or Hub repo id")
    ap.add_argument("--judge", default="outputs/classifier/judge.pt")
    ap.add_argument("--video-per-cell", type=int, default=1)
    ap.add_argument("--out", default="outputs/eval_goal")
    ap.add_argument("--required-gpu-mib", type=float, default=6000)
    args = ap.parse_args()
    ecfg = load_config(args.config)
    env_cfg, lcfg, gcfg = load_config(), load_config("configs/layout.yaml"), load_goal_config()
    S = env_cfg["canvas"]["size"]
    cats = args.categories or gcfg["categories"]
    n = args.episodes_per_category or ecfg["episodes_per_category"]
    sets = load_sets(ecfg["sketch_split"], categories=gcfg["categories"])
    perm = {c: np.random.default_rng([ecfg["seed"], k]).permutation(len(sets[c])) for k, c in enumerate(gcfg["categories"])}
    ctx = dict(ecfg=ecfg, cats=gcfg["categories"], S=S, sets=sets, perm=perm, env_cfg=env_cfg, lcfg=lcfg, gcfg=gcfg)
    if "E2" in args.settings:
        from oekaki.generator.imagine import Imaginer
        ctx["imaginer"] = Imaginer(resolve_ckpt(args.generator, "last.pt"))
    judge = None
    if Path(resolve(args.judge)).exists():
        from oekaki.classifier.model import Judge
        judge = Judge(resolve(args.judge))
    else:
        print(f"[warn] judge {args.judge} not found: class accuracy is not computed")
    if any(a.split("=", 1)[1] != "expert" for a in args.agent):
        from oekaki.utils.gpu import wait_for_gpu
        wait_for_gpu(args.required_gpu_mib)
    env = DrawingEnv(env_cfg, render=True)
    max_steps = int(ecfg["max_episode_s"] * env.fps)
    out = resolve(args.out)
    inf = ecfg["inference"]
    for spec in args.agent:
        name, path = spec.split("=", 1)
        is_expert = path == "expert"
        if is_expert:
            from oekaki.eval.policy import ExpertAgent
            agent = ExpertAgent(env, gcfg)
        else:
            from oekaki.eval.policy import SmolVLAAgent
            agent = SmolVLAAgent(resolve_ckpt(path), mode=inf["mode"], replan=inf["replan"], delay=inf["delay"],
                                 noise_scale=inf["noise_scale"])
        for setting in args.settings:
            st = ecfg["settings"][setting]
            if not is_expert and name not in st["models"] and name.split("_")[0] not in st["models"]:
                continue
            if is_expert and st["goal"] == "generated":
                continue
            sdir = out / setting
            (sdir / name).mkdir(parents=True, exist_ok=True)
            (sdir / "goal").mkdir(parents=True, exist_ok=True)
            res_f = open(sdir / f"results_{name}.jsonl", "w")
            t0 = time.time()
            for cat in cats:
                for i in range(n):
                    ep = plan_episode(setting, cat, i, ctx)
                    key = f"{cat.replace(' ', '_')}_{i:03d}"
                    cv2.imwrite(str(sdir / "goal" / f"{key}.png"), ep["goal"][..., 0])
                    if ep.get("gen_failed"):
                        row = dict(setting=setting, agent=name, category=cat, i=i, gen_failed=True)
                        res_f.write(json.dumps(row) + "\n")
                        continue
                    vid = i < args.video_per_cell
                    r = rollout(env, agent, ep, max_steps, is_expert, video=vid)
                    cv2.imwrite(str(sdir / name / f"{key}.png"), r["canvas"][..., ::-1])
                    if vid:
                        import imageio.v2 as imageio
                        imageio.mimwrite(sdir / name / f"{key}.mp4", r["frames"], fps=env.fps, quality=7,
                                         macro_block_size=1)
                    gm = goal_metrics(ep["goal"], r["canvas"], ecfg["metrics"]["size"], ecfg["metrics"]["dilate_px"],
                                      ecfg["metrics"]["cover_tol_px"])
                    row = dict(setting=setting, agent=name, category=cat, i=i, task=ep["task"], named=ep["named"],
                               height=ep["height"], sketch_idx=ep["sketch_idx"], layout=ep.get("layout"),
                               category2=ep.get("category2"), **r["metrics"], **gm)
                    if judge is not None:
                        p = judge.classify_drawing(r["canvas"], canvas=True)
                        row.update(judge_pred=judge.classes[int(p.argmax())], judge_p=float(p.max()),
                                   judge_correct=judge.classes[int(p.argmax())] == cat)
                        if setting == "E2":
                            pg = judge.classify_drawing(ep["goal"])
                            row.update(goal_judge_pred=judge.classes[int(pg.argmax())],
                                       goal_judge_correct=judge.classes[int(pg.argmax())] == cat)
                    res_f.write(json.dumps(row) + "\n"); res_f.flush()
                    print(f"[{setting}/{name}] {key} iou {gm['dilated_iou']:.2f} chamfer {gm['chamfer_mm']:.1f}mm "
                          f"cov {gm['coverage_goal']:.2f} off {gm['centroid_offset_mm']:.1f}mm "
                          f"judge {row.get('judge_pred', '-')} air {row['air_ratio']:.2f} ({time.time() - t0:.0f}s)",
                          flush=True)
            res_f.close()
        del agent
    env.close()
    print("saved", out)


if __name__ == "__main__":
    main()
