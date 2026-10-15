"""Synthetic burned-in-text fixtures: a smooth MR-like phantom with known text rendered into it.

No real patient data and no downloads: the phantom is generated, the text is fictitious and drawn
with Pillow's bundled font, and the per-pixel glyph coverage (alpha) is returned as ground truth.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def phantom(size: int = 256, seed: int = 0) -> np.ndarray:
    """Pelvis-like phantom: dark background, textured ellipse 'tissue', brighter inner 'gland'."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    body = ((xx - 0.5) / 0.42) ** 2 + ((yy - 0.55) / 0.33) ** 2 <= 1
    gland = ((xx - 0.5) / 0.12) ** 2 + ((yy - 0.55) / 0.10) ** 2 <= 1
    img = np.zeros((size, size), np.float64)
    img[body] = 900 + 250 * np.sin(xx[body] * 23) * np.cos(yy[body] * 17)
    img[gland] += 600
    img += rng.normal(0, 25, img.shape)
    img = np.clip(img, 0, None)
    return img.astype(np.uint16)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def burn(img: np.ndarray, items, size_px: int, level: float | None = None):
    """Render ``[(text, (x, y)), ...]`` at ``level`` (default: image max). Returns (burned, alpha)."""
    h, w = img.shape
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    f = _font(size_px)
    for text, xy in items:
        d.text(xy, text, fill=255, font=f)
    alpha = np.asarray(layer, np.float64) / 255.0
    lvl = float(img.max()) if level is None else level
    burned = np.rint(img * (1 - alpha) + lvl * alpha).astype(img.dtype)
    return burned, alpha


def burned_slice(layout: str = "corner", size: int = 256):
    """(clean, burned, alpha) for a corner-annotation or text-on-anatomy layout."""
    clean = phantom(size)
    fs = max(11, size // 20)
    if layout == "corner":
        items = [("DOE^JANE 67Y", (4, 4)), ("ID SYN-0012345", (4, 4 + fs + 4))]
        burned, alpha = burn(clean, items, fs, level=float(clean.max()) * 1.05)
    else:
        items = [("Lesion PZ DOE", (size // 2 - 5 * fs // 2, int(size * 0.33)))]
        burned, alpha = burn(clean, items, fs, level=float(clean.max()) * 1.05)
    return clean, burned, alpha
