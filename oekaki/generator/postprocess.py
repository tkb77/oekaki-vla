"""Post-processing of generated 64 x 64 drawings: binarise, drop small noise components (dots are
kept), and redraw the skeleton with the common renderer's line width -- optionally moved by a layout transform."""
from __future__ import annotations

import numpy as np

from oekaki.render import clean_mask, ink, raster_to_segments, render_strokes


def postprocess(img_u8: np.ndarray, thresh: int = 128, min_area: int = 4) -> list[np.ndarray]:
    """Generated image (white background) -> skeleton segments in the generator's normalised frame."""
    return raster_to_segments(clean_mask(ink(img_u8, thresh), min_area, keep_dots=True))


def segments_bbox(segs) -> tuple[np.ndarray, np.ndarray]:
    p = np.concatenate(segs)
    return p.min(0), p.max(0)


def to_unit(segs) -> list[np.ndarray]:
    """Segments -> unit drawing coordinates (bbox centred at 0.5, longer side 1), the frame the layout uses."""
    lo, hi = segments_bbox(segs)
    ext = max(float((hi - lo).max()), 1e-6)
    c = (lo + hi) / 2
    return [(s - c) / ext + 0.5 for s in segs]


def render_generated(segs, size: int, transform=None) -> np.ndarray:
    return render_strokes(segs, size, transform=transform, point_eps=0.0)
