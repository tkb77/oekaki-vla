"""Goal-image episodes: a Quick, Draw! drawing laid out on the paper, its goal image and the instruction."""
from __future__ import annotations

import copy

import numpy as np

from oekaki.config import load_config, resolve
from oekaki.expert.shapes import INSTRUCTION_TEMPLATES
from oekaki.layout import affine, drawable_box, sample_layout
from oekaki.render import apply_affine, norm_to_paper, render_strokes


def load_goal_config(path="configs/datagen_goal_v1.yaml") -> dict:
    """A datagen config deep-merged over its `base` config (recursively: goal_v2 -> goal_v1 -> datagen.yaml)."""
    cfg = load_config(path)
    base = load_goal_config(cfg.pop("base")) if "base" in cfg else {}

    def merge(a, b):
        out = copy.deepcopy(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out
    return merge(base, cfg)


def goal_instruction(category: str, rng: np.random.Generator, cfg: dict, named: bool, heldout: bool = False) -> str:
    if named:
        return INSTRUCTION_TEMPLATES[int(rng.integers(len(INSTRUCTION_TEMPLATES)))].format(obj=category)
    t = cfg["heldout_unnamed_templates"] if heldout else cfg["unnamed_templates"]
    return t[int(rng.integers(len(t)))]


def goal_image(strokes_norm, size: int) -> np.ndarray:
    """Goal image (RGB uint8): the whole paper from above, black lines on white, common renderer style."""
    g = render_strokes(strokes_norm, size)
    return np.repeat(g[..., None], 3, axis=-1)


def path_length_m(strokes_norm, paper_size: float) -> float:
    return float(sum(np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in strokes_norm) * paper_size)


def lay_out(sketch, rng: np.random.Generator, lcfg: dict, paper_size: float, transform=None) -> dict:
    """Place a unit-coordinate drawing on the paper (random layout unless `transform` is given)."""
    if transform is None:
        T, params = sample_layout(sketch, rng, lcfg, paper_size)
    else:
        T, params = np.asarray(transform), dict(transform=np.asarray(transform).tolist())
    norm = [apply_affine(s, T) for s in sketch]
    return dict(strokes_norm=norm, strokes_local=[norm_to_paper(s, paper_size) for s in norm], layout=params)


def pick_sketches(sets: dict, cfg: dict, lcfg: dict, paper_size: float, n_per_cat: int, seed: int) -> list[dict]:
    """Episode jobs: distinct drawings per category (no drawing reused), each with its own layout seed. A drawing
    whose laid-out path would be too long for the time limit is skipped."""
    rng = np.random.default_rng(seed)
    jobs = []
    for cat in cfg["categories"]:
        s = sets[cat]
        order = rng.permutation(len(s))
        k = 0
        for idx in order:
            if k == n_per_cat:
                break
            lay_seed = int(rng.integers(2**31))
            ep = lay_out(s[int(idx)], np.random.default_rng(lay_seed), lcfg, paper_size)
            if path_length_m(ep["strokes_norm"], paper_size) > cfg["max_path_m"]:
                continue
            jobs.append(dict(category=cat, sketch_idx=int(idx), key_id=int(s.key_id[idx]), seed=lay_seed))
            k += 1
    return jobs


def lay_out_pair(sk1, sk2, rng: np.random.Generator, lcfg: dict, pcfg: dict, paper_size: float) -> dict:
    """Two drawings side by side (left/right or top/bottom), each with its own size, rotation and mirroring.
    Wider than evaluation E5 (fixed left/right, size 0.42, upright) so that E5 stays a held-out composition."""
    lo, hi = drawable_box(paper_size, lcfg["paper_margin"])
    axis = int(rng.integers(2))
    first = rng.permutation([0.27, 0.73])
    norm, params = [], []
    for sk, a in zip((sk1, sk2), first):
        c = np.empty(2)
        c[axis] = a + rng.uniform(-pcfg["jitter"], pcfg["jitter"])
        c[1 - axis] = 0.5 + rng.uniform(-pcfg["cross_jitter"], pcfg["cross_jitter"])
        size = rng.uniform(*pcfg["size_range"])
        rot = rng.uniform(-lcfg["rot_deg"], lcfg["rot_deg"])
        flip = bool(rng.random() < lcfg["flip_prob"])
        T = affine(size * (hi - lo), rot, flip, lo + c * (hi - lo))
        norm += [apply_affine(s, T) for s in sk]
        params.append(dict(size_frac=float(size), center=c.tolist(), rot_deg=float(rot), flip=flip))
    return dict(strokes_norm=norm, strokes_local=[norm_to_paper(s, paper_size) for s in norm],
                layout=dict(pair=params, axis=axis))


def pick_pairs(sets: dict, cfg: dict, lcfg: dict, paper_size: float, n: int, seed: int) -> list[dict]:
    """Episode jobs with two drawings of two different categories (pairs evenly spread over the categories)."""
    rng = np.random.default_rng(seed)
    cats, pcfg = cfg["categories"], cfg["pairs"]
    jobs = []
    while len(jobs) < n:
        c1, c2 = rng.choice(len(cats), 2, replace=False)
        c1, c2 = cats[int(c1)], cats[int(c2)]
        i1, i2 = int(rng.integers(len(sets[c1]))), int(rng.integers(len(sets[c2])))
        lay_seed = int(rng.integers(2**31))
        ep = lay_out_pair(sets[c1][i1], sets[c2][i2], np.random.default_rng(lay_seed), lcfg, pcfg, paper_size)
        if path_length_m(ep["strokes_norm"], paper_size) > pcfg["max_path_m"]:
            continue
        jobs.append(dict(category=c1, category2=c2, sketch_idx=i1, sketch_idx2=i2, key_id=int(sets[c1].key_id[i1]),
                         key_id2=int(sets[c2].key_id[i2]), seed=lay_seed, pair=True))
    return jobs


def build_episode(job: dict, sets: dict, cfg: dict, lcfg: dict, paper_size: float, heldout_text: bool = False,
                  named: bool | None = None, transform=None) -> dict:
    """Deterministic episode content from a job (layout, instruction, goal image, strokes)."""
    rng = np.random.default_rng(job["seed"])
    if job.get("pair"):
        pcfg = cfg["pairs"]
        ep = lay_out_pair(sets[job["category"]][job["sketch_idx"]], sets[job["category2"]][job["sketch_idx2"]],
                          rng, lcfg, pcfg, paper_size)
        ep["named"] = named = bool(rng.random() < pcfg["named_prob"]) if named is None else named
        if named:
            t = pcfg["named_templates"][int(rng.integers(len(pcfg["named_templates"])))]
            ep["task"] = t.format(a=job["category"], b=job["category2"])
        else:
            ep["task"] = goal_instruction(job["category"], rng, cfg, False, heldout_text)
        ep["goal"] = goal_image(ep["strokes_norm"], cfg["goal"]["size"])
        ep["shape"] = dict(name=job["category"], strokes=ep["strokes_local"], params=ep["layout"])
        ep["rng_seed"] = int(rng.integers(2**31))
        return ep
    sketch = sets[job["category"]][job["sketch_idx"]]
    ep = lay_out(sketch, rng, lcfg, paper_size, transform)
    if named is None:
        named = bool(rng.random() < cfg["named_prob"])
    ep["named"] = named
    ep["task"] = goal_instruction(job["category"], rng, cfg, named, heldout_text)
    ep["goal"] = goal_image(ep["strokes_norm"], cfg["goal"]["size"])
    ep["shape"] = dict(name=job["category"], strokes=ep["strokes_local"], params=ep["layout"])
    ep["rng_seed"] = int(rng.integers(2**31))  # for the simulation (paper height, noise, disturbance)
    return ep


def load_sets(split: str, qcfg_path="configs/quickdraw.yaml", categories=None):
    from oekaki.quickdraw import load_split
    q = load_config(qcfg_path)
    return load_split(resolve(q["out_dir"]), categories or q["categories"], split)


def episode_index(dataset_dir) -> list[dict]:
    """Per LeRobot episode index of a goal dataset (<root>/<name>/G): the job metadata (category, layout, ...)."""
    from pathlib import Path
    import json
    base = Path(dataset_dir).parent
    mapping = json.load(open(base / "episode_to_job.json"))
    return [json.load(open(base / "meta" / f"job_{j:05d}.json")) for j in mapping]


def first_episodes_per_category(dataset_dir, n: int) -> list[int]:
    out, count = [], {}
    for e, m in enumerate(episode_index(dataset_dir)):
        if count.get(m["category"], 0) < n:
            out.append(e)
            count[m["category"]] = count.get(m["category"], 0) + 1
    return out


def select_episodes(dataset_dir, singles_per_category: int | None = None, pairs: bool = True) -> list[int]:
    """LeRobot episode indices: the first `singles_per_category` single-drawing episodes of each category (all if
    None) plus, if `pairs`, every two-drawing episode."""
    out, count = [], {}
    for e, m in enumerate(episode_index(dataset_dir)):
        if m.get("pair"):
            if pairs:
                out.append(e)
        elif singles_per_category is None or count.get(m["category"], 0) < singles_per_category:
            out.append(e)
            count[m["category"]] = count.get(m["category"], 0) + 1
    return out
