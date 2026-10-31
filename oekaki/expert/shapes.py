"""Shape stroke generation (paper-local metres) and stroke resampling."""
from __future__ import annotations

import numpy as np

SHAPES = ["circle", "square", "triangle", "star", "line"]


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _closed_polygon(verts: np.ndarray, start: int, reverse: bool) -> np.ndarray:
    verts = np.roll(verts, -start, axis=0)
    if reverse:
        verts = np.concatenate([verts[:1], verts[1:][::-1]])
    return np.concatenate([verts, verts[:1]])


def resample(poly: np.ndarray, spacing: float) -> np.ndarray:
    """Uniform arc-length resampling of a polyline (keeps both end points)."""
    poly = np.asarray(poly, dtype=np.float64)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / spacing)), 1)
    t = np.linspace(0, s[-1], n + 1)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], axis=1)


def make_shape(name: str, rng: np.random.Generator, cfg: dict) -> dict:
    """Returns {'name', 'strokes': [ (N,2) dense polylines ], 'params'} in paper-local metres."""
    c = cfg["shapes"]
    center = rng.uniform(-c["center_jitter"], c["center_jitter"], size=2)
    rot = np.radians(rng.uniform(-c["rot_jitter_deg"], c["rot_jitter_deg"]))
    reverse = bool(rng.random() < 0.5)
    params = dict(center=center.tolist(), rot_deg=float(np.degrees(rot)), reverse=reverse)
    if name == "circle":
        r = rng.uniform(*c["circle_radius"])
        a0 = rng.uniform(0, 2 * np.pi)
        a = a0 + np.linspace(0, 2 * np.pi, 361) * (-1 if reverse else 1)
        stroke = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        params.update(radius=float(r), start_angle=float(a0))
    elif name == "square":
        h = rng.uniform(*c["square_half"])
        verts = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        start = int(rng.integers(4))
        stroke = _closed_polygon(verts, start, reverse)
        params.update(half=float(h), start=start)
    elif name == "triangle":
        r = rng.uniform(*c["triangle_radius"])
        ang = np.pi / 2 + np.arange(3) * 2 * np.pi / 3
        verts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        verts[:, 1] -= r * 0.25  # centre the bounding box vertically
        start = int(rng.integers(3))
        stroke = _closed_polygon(verts, start, reverse)
        params.update(radius=float(r), start=start)
    elif name == "star":
        r = rng.uniform(*c["star_radius"])
        ang = np.pi / 2 + np.arange(10) * np.pi / 5
        rad = np.where(np.arange(10) % 2 == 0, r, r * c["star_inner_ratio"])
        verts = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        verts[:, 1] -= r * 0.1
        start = int(rng.integers(5)) * 2  # start at an outer point
        stroke = _closed_polygon(verts, start, reverse)
        params.update(radius=float(r), start=start)
    elif name == "line":
        length = rng.uniform(*c["line_length"])
        theta = rng.uniform(0, np.pi)
        d = np.array([np.cos(theta), np.sin(theta)]) * length / 2
        stroke = np.stack([-d, d]) if not reverse else np.stack([d, -d])
        params.update(length=float(length), angle_deg=float(np.degrees(theta)))
        rot = 0.0
    else:
        raise ValueError(name)
    stroke = stroke @ _rot(rot).T + center
    lim = cfg["canvas"]["size"] / 2 - c["margin"]
    stroke = np.clip(stroke, -lim, lim)
    return dict(name=name, strokes=[stroke], params=params)


INSTRUCTION_TEMPLATES = [
    "draw a {obj}",
    "draw a {obj} on the paper",
    "please sketch a {obj}",
    "make a drawing of a {obj}",
]
HELDOUT_TEMPLATES = [
    "could you draw a {obj}?",
    "sketch a {obj} for me",
]


def instruction(name: str, rng: np.random.Generator, heldout: bool = False) -> str:
    templates = HELDOUT_TEMPLATES if heldout else INSTRUCTION_TEMPLATES
    return templates[int(rng.integers(len(templates)))].format(obj=name)
