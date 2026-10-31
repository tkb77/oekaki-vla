"""Canvas buffer: paper-local coordinates <-> pixels, line drawing.

Paper-local frame: origin at the centre of the paper top surface, x/y parallel to the robot base
frame (x forward, y left). Pixel layout of `img` (verified against the MuJoCo texture mapping):
column grows with +x, row grows with -y, i.e. viewing `img` is a top view with x right / y up.
"""
from __future__ import annotations

import cv2
import numpy as np


class Canvas:
    def __init__(self, size_m: float, res: int, line_width_px: int = 4, color=(20, 20, 120)):
        self.size = size_m
        self.res = res
        self.lw = int(line_width_px)
        self.color = tuple(int(c) for c in color)
        self.img = np.full((res, res, 3), 255, np.uint8)
        self._last: tuple[int, int] | None = None
        self.n_points = 0

    def clear(self, tint=(255, 255, 255)):
        self.img[:] = np.asarray(tint, np.uint8)
        self._last = None
        self.n_points = 0

    def to_px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        half = self.size / 2
        col = (xy[..., 0] + half) / self.size * self.res
        row = (half - xy[..., 1]) / self.size * self.res
        return np.stack([col, row], axis=-1)

    def to_m(self, px) -> np.ndarray:
        px = np.asarray(px, dtype=np.float64)
        half = self.size / 2
        x = px[..., 0] / self.res * self.size - half
        y = half - px[..., 1] / self.res * self.size
        return np.stack([x, y], axis=-1)

    def pen_down(self, xy):
        """Add a contact point (paper-local metres); connects to the previous point if the pen stayed down."""
        c, r = self.to_px(xy)
        p = (int(round(c * 16)), int(round(r * 16)))  # cv2 fixed-point (shift=4) for sub-pixel accuracy
        if self._last is None:
            cv2.circle(self.img, p, self.lw * 8, self.color, -1, cv2.LINE_AA, shift=4)
        else:
            cv2.line(self.img, self._last, p, self.color, self.lw, cv2.LINE_AA, shift=4)
        self._last = p
        self.n_points += 1

    def pen_up(self):
        self._last = None

    def ink_mask(self, thresh: int = 200) -> np.ndarray:
        return self.img.min(axis=-1) < thresh


def render_strokes(strokes, size_m: float, res: int, line_width_px: int) -> np.ndarray:
    """Rasterise ground-truth strokes (list of (N,2) arrays in paper-local metres) to a binary mask."""
    cv = Canvas(size_m, res, line_width_px, (0, 0, 0))
    for s in strokes:
        cv.pen_up()
        for p in s:
            cv.pen_down(p)
    return cv.ink_mask()
