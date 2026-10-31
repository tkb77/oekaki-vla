"""Recognising what was drawn on canvas_img: CLIP zero-shot + a geometric contour classifier."""
from __future__ import annotations

import cv2
import numpy as np

LABELS = ["circle", "square", "triangle", "star", "line"]


class ClipRecognizer:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda",
                 labels=LABELS):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.device = device
        self.labels = list(labels)
        # class names only in the object slot (a "line drawing of ..." template biases towards "line")
        desc = {"circle": "a circle", "square": "a square", "triangle": "a triangle",
                "star": "a five-pointed star", "line": "a single straight stroke"}
        prompts = [f"a black ink sketch of {desc.get(l, 'a ' + l)} on white paper" for l in self.labels]
        with torch.no_grad():
            t = self.proc(text=prompts, return_tensors="pt", padding=True).to(device)
            e = self.model.get_text_features(**t)
            e = e if isinstance(e, torch.Tensor) else e.pooler_output
            self.text = e / e.norm(dim=-1, keepdim=True)

    def __call__(self, canvases: list[np.ndarray], batch: int = 32) -> list[dict]:
        torch = self.torch
        out = []
        for i in range(0, len(canvases), batch):
            ims = [_prep(c) for c in canvases[i:i + batch]]
            with torch.no_grad():
                x = self.proc(images=ims, return_tensors="pt").to(self.device)
                e = self.model.get_image_features(pixel_values=x["pixel_values"].half())
                e = e if isinstance(e, torch.Tensor) else e.pooler_output
                e = e / e.norm(dim=-1, keepdim=True)
                p = (100 * e @ self.text.T).softmax(-1).float().cpu().numpy()
            for row in p:
                out.append(dict(pred=self.labels[int(row.argmax())], probs=row.round(4).tolist()))
        return out


def _prep(canvas: np.ndarray) -> np.ndarray:
    """Black ink on white, cropped around the drawing, to 224 px."""
    ink = canvas.min(-1) < 200
    img = np.full(canvas.shape[:2], 255, np.uint8)
    img[ink] = 0
    img = cv2.dilate(255 - img, np.ones((3, 3), np.uint8))
    img = 255 - img
    ys, xs = np.nonzero(ink)
    if len(xs):
        c = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
        h = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.65 + 20
        x0, y0 = int(max(c[0] - h, 0)), int(max(c[1] - h, 0))
        x1, y1 = int(min(c[0] + h, img.shape[1])), int(min(c[1] + h, img.shape[0]))
        img = img[y0:y1, x0:x1]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return np.stack([img] * 3, -1)


def geometric_classify(canvas: np.ndarray) -> dict:
    """Heuristic classifier from the convex hull of all ink (robust to open / fragmented strokes)."""
    ink = (canvas.min(-1) < 200).astype(np.uint8)
    if ink.sum() < 30:
        return dict(pred="none", n_fragments=0)
    n_frag = cv2.connectedComponents(ink)[0] - 1
    hull = cv2.convexHull(cv2.findNonZero(ink))
    (w, h) = cv2.minAreaRect(hull)[1]
    elong = max(w, h) / max(min(w, h), 1)
    peri = cv2.arcLength(hull, True)
    circ = 4 * np.pi * cv2.contourArea(hull) / max(peri ** 2, 1)  # of the hull: circle ~1, square .79, triangle .60
    filled = np.zeros_like(ink)
    cv2.drawContours(filled, [hull], -1, 1, -1)
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    depth = float(np.median(dist[ink > 0]) / max(dist.max(), 1))  # how far inside the hull the ink lies
    # thresholds from 500 expert canvases: star depth .27-.30 (others <= .13); hull circularity
    # circle >= .98, square .80-.87, triangle .64-.70 (used when the corner count is ambiguous); line elongation >= 5.4 (others <= 1.2)
    if elong > 3.5:
        pred = "line"
    elif depth > 0.2:
        pred = "star"
    else:
        nv = len(cv2.approxPolyDP(hull, 0.03 * peri, True))  # expert: triangle 3, square 4, circle 7-8
        if nv <= 3:
            pred = "triangle"
        elif nv == 4:
            pred = "square"
        else:
            pred = "circle" if circ > 0.88 else ("square" if circ > 0.74 else "triangle")
    return dict(pred=pred, n_fragments=int(n_frag), circularity=float(circ), depth=depth, elongation=float(elong))
