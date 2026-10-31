"""Common line-drawing renderer.

Every line drawing -- generator training images, the goal image of the dataset, and both sides of
the goal-vs-canvas comparison -- is produced by `render_strokes`, so that all of them look the same.

Coordinates are *normalised image coordinates*: u in [0, 1] to the right, v in [0, 1] downwards, covering the
whole image. For the goal image the image is the whole paper seen from above (axis aligned, no perspective):
    u = (x + S/2) / S,  v = (S/2 - y) / S        (x, y paper-local metres, S = paper size)
which is the same pixel layout as `Canvas.img`.

A raster (generator output, robot canvas) is brought into the same style with `raster_to_segments`: binarise,
skeletonise, and turn the skeleton into 2-point segments that `render_strokes` redraws with the standard width.
"""
from __future__ import annotations

import cv2
import numpy as np

LINE_RATIO = 1 / 32  # line width / image size (64 px -> 2 px, 128 px -> 4 px)
_SHIFT = 4           # cv2 fixed-point bits (sub-pixel accuracy)


def line_width_for(size: int, ratio: float = LINE_RATIO) -> int:
    return max(1, int(round(size * ratio)))


def apply_affine(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    """points (N,2) in normalised coords; transform: 2x3 affine in normalised coords."""
    if transform is None:
        return np.asarray(points, np.float64)
    p = np.asarray(points, np.float64)
    return p @ np.asarray(transform)[:, :2].T + np.asarray(transform)[:, 2]


def render_strokes(strokes, size: int, line_width: int | None = None, transform: np.ndarray | None = None,
                   point_eps: float | None = None) -> np.ndarray:
    """Rasterise strokes (list of (N,2) arrays in normalised coords) to a uint8 (size, size) image:
    white (255) background, black (0) anti-aliased lines of `line_width` px (default size * LINE_RATIO).
    A stroke whose extent is below `point_eps` (default: half a line width) is drawn as a dot."""
    lw = line_width or line_width_for(size)
    eps = point_eps if point_eps is not None else 0.5 * lw / size
    img = np.full((size, size), 255, np.uint8)
    f = size * (1 << _SHIFT)
    for s in strokes:
        p = apply_affine(np.asarray(s, np.float64).reshape(-1, 2), transform)
        if len(p) == 0:
            continue
        q = np.round(p * f).astype(np.int32)
        if len(p) == 1 or np.ptp(p, axis=0).max() < eps:
            c = np.round(p.mean(0) * f).astype(np.int32)
            cv2.circle(img, (int(c[0]), int(c[1])), int(round(lw / 2 * (1 << _SHIFT))), 0, -1, cv2.LINE_AA,
                       shift=_SHIFT)
        else:
            cv2.polylines(img, [q.reshape(-1, 1, 2)], False, 0, lw, cv2.LINE_AA, shift=_SHIFT)
    return img


def ink(img: np.ndarray, thresh: int = 128) -> np.ndarray:
    """Binary ink mask of a rendered (white background) image; accepts grey or RGB."""
    g = img.min(axis=-1) if img.ndim == 3 else img
    return g < thresh


# ---------------------------------------------------------------------------------------------------------
def clean_mask(mask: np.ndarray, min_area: int = 4, keep_dots: bool = True) -> np.ndarray:
    """Remove small connected components. Components that are compact blobs (dots, e.g. the eyes of a smiley)
    are kept when keep_dots, unless they are smaller than 2 px."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, bool)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= min_area or (keep_dots and a >= 2):
            out |= lab == i
    return out


def raster_to_segments(mask: np.ndarray) -> list[np.ndarray]:
    """Skeleton of a binary mask as a list of strokes in normalised coords: one 2-point segment per pair of
    8-connected skeleton pixels, and a 1-point stroke per isolated pixel (dot)."""
    from skimage.morphology import skeletonize
    h, w = mask.shape
    sk = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(sk)
    if len(xs) == 0:
        return []
    idx = -np.ones((h + 2, w + 2), np.int64)
    idx[ys + 1, xs + 1] = np.arange(len(xs))
    pts = np.stack([(xs + 0.5) / w, (ys + 0.5) / h], 1)
    segs, has_nb = [], np.zeros(len(xs), bool)
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):  # each neighbour pair once
        j = idx[ys + 1 + dy, xs + 1 + dx]
        ok = j >= 0
        for a, b in zip(np.nonzero(ok)[0], j[ok]):
            segs.append(np.stack([pts[a], pts[b]]))
        has_nb[ok] = True
        has_nb[j[ok]] = True
    segs += [pts[i:i + 1] for i in np.nonzero(~has_nb)[0]]
    return segs


def restyle(img_or_mask: np.ndarray, size: int, transform: np.ndarray | None = None, thresh: int = 128,
            min_area: int = 4, line_width: int | None = None) -> np.ndarray:
    """Any line raster (white background, or a bool mask) -> standard style at `size`, optionally moved by a
    normalised-coordinate affine transform (the line width does not scale with the transform)."""
    mask = img_or_mask if img_or_mask.dtype == bool else ink(img_or_mask, thresh)
    mask = clean_mask(mask, min_area)
    return render_strokes(raster_to_segments(mask), size, line_width, transform, point_eps=0.0)


# ---------------------------------------------------------------------------------------------------------
def paper_to_norm(xy: np.ndarray, paper_size: float) -> np.ndarray:
    xy = np.asarray(xy, np.float64)
    return np.stack([(xy[..., 0] + paper_size / 2) / paper_size, (paper_size / 2 - xy[..., 1]) / paper_size], -1)


def norm_to_paper(uv: np.ndarray, paper_size: float) -> np.ndarray:
    uv = np.asarray(uv, np.float64)
    return np.stack([uv[..., 0] * paper_size - paper_size / 2, paper_size / 2 - uv[..., 1] * paper_size], -1)
