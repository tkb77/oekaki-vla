"""Small CNN judge for the goal images (5 classes, 64 x 64 standard-style drawings, ink = 1)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class JudgeCNN(nn.Module):
    def __init__(self, n_classes: int, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(block(1, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 8 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(8 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def to_input(img_u8: np.ndarray) -> torch.Tensor:
    """(N,H,W) white-background uint8 -> (N,1,H,W) float, ink = 1."""
    return (1.0 - torch.as_tensor(img_u8).float() / 255.0).unsqueeze(1)


class Judge:
    """Loads a trained judge; classifies standard-style 64 x 64 images, goal images or robot canvases."""

    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.classes = ck["classes"]
        self.size = ck["image_size"]
        self.model = JudgeCNN(len(self.classes)).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def probs(self, imgs_u8: np.ndarray) -> np.ndarray:
        x = to_input(imgs_u8).to(self.device)
        return torch.softmax(self.model(x), -1).cpu().numpy()

    def classify_drawing(self, img: np.ndarray, canvas: bool = False) -> np.ndarray:
        """Goal image / canvas (any size, RGB or grey) -> class probabilities (restyled to 64 x 64 first)."""
        from oekaki.render import ink, restyle
        mask = ink(img, 200 if canvas else 128)
        return self.probs(restyle(mask, self.size)[None])[0]
