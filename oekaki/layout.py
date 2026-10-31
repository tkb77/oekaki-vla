"""Layout randomisation.

A drawing in unit drawing coordinates (see oekaki.quickdraw) is scaled, optionally mirrored, rotated about its
centre and moved inside the drawable area of the paper. The result is one 2x3 affine transform from unit drawing
coordinates to normalised paper coordinates (oekaki.render), applied identically to the expert strokes and to
the goal image.
"""
from __future__ import annotations

import numpy as np

from oekaki.render import apply_affine


def affine(scale: float, rot_deg: float, flip: bool, center_uv) -> np.ndarray:
    """Unit drawing coords -> normalised paper coords. Rotation is counter-clockwise on the paper (v points down,
    so it is applied as a clockwise rotation in (u, v))."""
    th = np.radians(rot_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, s], [-s, c]])
    F = np.diag([-1.0, 1.0]) if flip else np.eye(2)
    A = scale * R @ F
    t = np.asarray(center_uv, np.float64) - A @ np.array([0.5, 0.5])
    return np.hstack([A, t[:, None]])


def drawable_box(paper_size: float, margin: float) -> tuple[float, float]:
    m = margin / paper_size
    return m, 1.0 - m


def sample_layout(strokes, rng: np.random.Generator, cfg: dict, paper_size: float) -> tuple[np.ndarray, dict]:
    lo, hi = drawable_box(paper_size, cfg["paper_margin"])
    side = hi - lo
    scale = rng.uniform(*cfg["size_frac"]) * side
    rot = rng.uniform(-cfg["rot_deg"], cfg["rot_deg"])
    flip = bool(rng.random() < cfg["flip_prob"])
    pts = np.concatenate(strokes)
    # extent of the transformed drawing around its (centred) bounding-box centre
    A0 = affine(scale, rot, flip, (0.0, 0.0))
    p = apply_affine(pts, A0)
    pmin, pmax = p.min(0), p.max(0)
    if np.any(pmax - pmin > side):  # rotation made it too big: shrink to fit
        k = side / (pmax - pmin).max() * 0.999
        scale *= k; A0 = affine(scale, rot, flip, (0.0, 0.0)); p = apply_affine(pts, A0); pmin, pmax = p.min(0), p.max(0)
    c = np.array([rng.uniform(lo - pmin[i], hi - pmax[i]) for i in range(2)])
    T = affine(scale, rot, flip, c)
    params = dict(scale=float(scale), size_frac=float(scale / side), rot_deg=float(rot), flip=flip,
                  center_uv=c.tolist(), transform=T.tolist())
    return T, params


def centred(fill: float) -> np.ndarray:
    """Unit drawing coords -> a centred drawing whose longer side is `fill` of the image (generator frame)."""
    return np.array([[fill, 0.0, 0.5 - fill / 2], [0.0, fill, 0.5 - fill / 2]])


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """outer o inner for 2x3 affines."""
    M = np.vstack([outer, [0, 0, 1]]) @ np.vstack([inner, [0, 0, 1]])
    return M[:2]


def invert(T: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.vstack([T, [0, 0, 1]]))[:2]
