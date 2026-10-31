"""Goal-vs-drawing metrics. Both images are brought to the common renderer style at
128 x 128 (whole paper, same coordinates) before comparison.

dilated_iou: IoU of the two line masks after dilating both by `dilate_px` (tolerates small offsets).
chamfer_px / chamfer_mm: symmetric mean nearest-neighbour distance between the two skeletons (centre lines).
coverage: share of the goal's centre line within `cover_tol_px` of the drawn ink (1 - break rate);
precision: share of the drawn centre line within `cover_tol_px` of the goal's ink.
"""
from __future__ import annotations

import cv2
import numpy as np

from oekaki.render import ink, restyle

PAPER_MM = 150.0


def to_style(img: np.ndarray, size: int = 128, canvas: bool = False) -> np.ndarray:
    """Goal image (black on white) or robot canvas (coloured ink on tinted paper) -> standard-style mask."""
    mask = ink(img, 200) if canvas else ink(img, 128)
    return ink(restyle(mask, size))


def _mean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """mean distance from the pixels of a to the nearest pixel of b"""
    dt = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)
    return float(dt[a].mean())


def goal_metrics(goal_img: np.ndarray, canvas_img: np.ndarray, size: int = 128, dilate_px: int = 3,
                 cover_tol_px: int = 2) -> dict:
    g = to_style(goal_img, size)
    c = to_style(canvas_img, size, canvas=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    gd, cd = cv2.dilate(g.astype(np.uint8), k) > 0, cv2.dilate(c.astype(np.uint8), k) > 0
    union = (gd | cd).sum()
    iou = float((gd & cd).sum() / union) if union else 0.0
    cov = prec = 0.0
    if g.any() and c.any():
        from skimage.morphology import skeletonize
        gs, cs = skeletonize(g), skeletonize(c)
        ch = 0.5 * (_mean_dist(gs, cs) + _mean_dist(cs, gs))
        kt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cover_tol_px + 1, 2 * cover_tol_px + 1))
        cov = float((gs & (cv2.dilate(c.astype(np.uint8), kt) > 0)).sum() / max(gs.sum(), 1))
        prec = float((cs & (cv2.dilate(g.astype(np.uint8), kt) > 0)).sum() / max(cs.sum(), 1))
    else:
        ch = float(size)  # nothing drawn: worst case
    ys, xs = np.nonzero(g)
    yc, xc = np.nonzero(c)
    off = float(np.hypot(xs.mean() - xc.mean(), ys.mean() - yc.mean())) if len(xs) and len(xc) else float(size)
    ext = lambda y, x: float(max(np.ptp(x), np.ptp(y))) if len(x) else 0.0  # noqa: E731
    return dict(dilated_iou=iou, chamfer_px=ch, chamfer_mm=ch * PAPER_MM / size, coverage_goal=cov,
                precision_goal=prec, centroid_offset_mm=off * PAPER_MM / size,
                size_ratio=ext(yc, xc) / max(ext(ys, xs), 1e-6), drawn=bool(c.any()))
