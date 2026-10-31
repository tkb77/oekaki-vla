"""Drawing quality metrics computed from canvas_img (ground truth, not rendered images)."""
from __future__ import annotations

import cv2
import numpy as np

from oekaki.env.canvas import render_strokes


def coverage_metrics(canvas_img: np.ndarray, strokes_local: list[np.ndarray], size_m: float,
                     tol_px: int = 8, line_width_px: int = 8) -> dict:
    """coverage: fraction of the GT stroke covered by ink (within tol); precision: fraction of ink near GT."""
    res = canvas_img.shape[0]
    gt = render_strokes(strokes_local, size_m, res, 2)
    ink = canvas_img.min(axis=-1) < 200
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1, 2 * tol_px + 1))
    ink_d = cv2.dilate(ink.astype(np.uint8), k) > 0
    gt_d = cv2.dilate(gt.astype(np.uint8), k) > 0
    n_gt = max(int(gt.sum()), 1)
    coverage = float((gt & ink_d).sum() / n_gt)
    precision = float((ink & gt_d).sum() / max(int(ink.sum()), 1)) if ink.any() else 0.0
    return dict(coverage=coverage, break_rate=1.0 - coverage, precision=precision,
                ink_px=int(ink.sum()))


def contact_metrics(contacts: np.ndarray, compressions: np.ndarray, travel: float,
                    drawing_mask: np.ndarray | None = None) -> dict:
    """air_ratio: share of the drawing period with contact==0; over_press_ratio: compression near the limit.

    If no explicit drawing mask is given, the drawing period is the span between the first and last
    contact-switch activation.
    """
    contacts = np.asarray(contacts).astype(bool)
    comp = np.asarray(compressions)
    if drawing_mask is None:
        idx = np.nonzero(contacts)[0]
        drawing_mask = np.zeros_like(contacts)
        if len(idx):
            drawing_mask[idx[0]:idx[-1] + 1] = True
    n = int(drawing_mask.sum())
    air = float((~contacts & drawing_mask).sum() / n) if n else 1.0
    over = float((comp > 0.9 * travel).sum() / max(int(contacts.sum()), 1)) if contacts.any() else 0.0
    n_lifts = int(((~contacts[1:]) & contacts[:-1] & drawing_mask[1:]).sum())
    return dict(air_ratio=air, over_press_ratio=over, n_breaks=n_lifts, contact_steps=int(contacts.sum()))


def smoothness(actions: np.ndarray, fps: float) -> dict:
    """Mean squared joint jerk of the action sequence (lower = smoother)."""
    a = np.asarray(actions)
    if len(a) < 4:
        return dict(jerk=float("nan"))
    j = np.diff(a, n=3, axis=0) * fps ** 3
    return dict(jerk=float(np.mean(j[:, :5] ** 2)))


def seam_metrics(actions: np.ndarray, seams: list[int]) -> dict:
    """Joint-space step size |a_t - a_{t-1}| (rad, arm joints) where a new chunk takes over vs. elsewhere."""
    a = np.asarray(actions)[:, :5]
    if len(a) < 3:
        return dict(seam_jump=float("nan"), step_jump=float("nan"))
    step = np.linalg.norm(np.diff(a, axis=0), axis=1)  # step[i] = |a_{i+1} - a_i|
    idx = np.array([s - 1 for s in seams if 1 <= s < len(a)], int)
    mask = np.zeros(len(step), bool)
    mask[idx] = True
    return dict(seam_jump=float(step[mask].mean()) if mask.any() else float("nan"),
                step_jump=float(step[~mask].mean()))


def _ink_points(mask: np.ndarray, n: int, rng) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    p = np.stack([xs, ys], 1).astype(float)
    if len(p) > n:
        p = p[rng.choice(len(p), n, replace=False)]
    return p


def shape_fidelity(canvas_img: np.ndarray, strokes_local: list[np.ndarray], size_m: float,
                   tol: float = 0.12) -> dict:
    """Pose-invariant comparison of the drawn ink with the target shape.

    The policy is not told the size/rotation/position the expert drew with, so both point sets are
    centred, scaled to unit RMS radius and aligned over rotations (+/-45 deg). Reports the symmetric
    Chamfer distance in mm (rescaled by the target's RMS radius), `shape_cov` = share of the target within
    `tol` (x RMS radius) of the ink and `shape_prec` = share of the ink within `tol` of the target
    (stray strokes / scribbles lower it).
    """
    res = canvas_img.shape[0]
    ink = canvas_img.min(axis=-1) < 200
    if ink.sum() < 30:
        return dict(chamfer_mm=float("nan"), shape_cov=0.0, shape_prec=0.0)
    rng = np.random.default_rng(0)
    gt = render_strokes(strokes_local, size_m, res, 2)
    P, G = _ink_points(ink, 1500, rng), _ink_points(gt, 600, rng)
    norm = []
    for X in (P, G):
        Xc = X - X.mean(0)
        r = np.sqrt((Xc ** 2).sum(1).mean())
        norm.append((Xc / max(r, 1e-6), r))
    (Pn, _), (Gn, rg) = norm
    best = None
    for th in np.radians(np.arange(-45, 46, 3)):
        c, s = np.cos(th), np.sin(th)
        Pr = Pn @ np.array([[c, -s], [s, c]]).T
        D = np.sqrt(((Pr[:, None] - Gn[None]) ** 2).sum(-1))
        dp, dg = D.min(1), D.min(0)
        ch = 0.5 * (dp.mean() + dg.mean())
        if best is None or ch < best[0]:
            best = (ch, float((dg < tol).mean()), float((dp < tol).mean()))
    px_m = size_m / res
    return dict(chamfer_mm=float(best[0] * rg * px_m * 1e3), shape_cov=best[1], shape_prec=best[2])
