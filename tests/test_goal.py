"""Smoke tests of the goal-image parts: common renderer, layout, restyling, goal metrics, category parsing."""
import numpy as np

from oekaki.eval.goal_metrics import goal_metrics
from oekaki.generator.imagine import parse_category
from oekaki.layout import affine, drawable_box, invert, compose, sample_layout
from oekaki.render import apply_affine, ink, line_width_for, render_strokes, restyle


def test_line_width_ratio():
    assert line_width_for(64) == 2 and line_width_for(128) == 4 and line_width_for(256) == 8


def test_dot_is_drawn():
    img = render_strokes([np.array([[0.5, 0.5]])], 64)
    assert ink(img).sum() >= 3


def test_layout_inside_drawable_area():
    rng = np.random.default_rng(0)
    cfg = dict(paper_margin=0.012, size_frac=[0.4, 0.9], rot_deg=15.0, flip_prob=0.5)
    lo, hi = drawable_box(0.15, 0.012)
    sk = [np.array([[0.0, 0.5], [1.0, 0.5]]), np.array([[0.5, 0.0], [0.5, 1.0]])]
    for _ in range(200):
        T, p = sample_layout(sk, rng, cfg, 0.15)
        q = np.concatenate([apply_affine(s, T) for s in sk])
        assert q.min() >= lo - 1e-9 and q.max() <= hi + 1e-9


def test_affine_inverse():
    T = affine(0.5, 10.0, True, (0.3, 0.6))
    assert np.allclose(compose(T, invert(T)), np.array([[1, 0, 0], [0, 1, 0]]), atol=1e-9)


def test_goal_metrics_identity_and_shift():
    sq = [np.array([[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]])]
    g = np.repeat(render_strokes(sq, 256)[..., None], 3, -1)
    m = goal_metrics(g, g)
    assert m["dilated_iou"] > 0.95 and m["chamfer_mm"] < 0.5 and m["coverage_goal"] > 0.95
    moved = np.repeat(render_strokes([s + 0.2 for s in sq], 256)[..., None], 3, -1)
    m2 = goal_metrics(g, moved)
    assert m2["dilated_iou"] < 0.3 and m2["centroid_offset_mm"] > 25


def test_restyle_keeps_dots():
    img = render_strokes([np.array([[0.2, 0.2]]), np.array([[0.4, 0.6], [0.8, 0.6]])], 64)
    out = restyle(img, 128)
    import cv2
    n = cv2.connectedComponents(ink(out).astype(np.uint8))[0] - 1
    assert n == 2


def test_parse_category():
    cats = ["circle", "square", "triangle", "star", "smiley face"]
    assert parse_category("Please sketch a Smiley Face", cats) == "smiley face"
    assert parse_category("draw the picture shown", cats) is None
