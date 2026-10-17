"""
Quantitative verification of burned-in text removal.

Two kinds of numbers, kept apart on purpose:

1. **Measured on the real text** (paper eq. 40). For every region the removal touched,

   * ``retained(B) = 1 - |G_B| / |B|`` -- the fraction of the box left bit-identical
     (``G_B``: pixels synthesised or blacked out inside box ``B``);
   * ``c_audit(B) = min(1, max(0, n_OCR(B) / 100))`` -- the normalised OCR confidence recorded in
     the audit trail.

   These need no ground truth and are available for every image.

2. **Synthetic self-test** (paper eq. 41). Error against a clean slice ``T`` needs the pixels that
   the burned-in text overwrote, which no longer exist on a real upload. The self-test therefore
   takes the de-identified output as ``T``, burns *fictitious* corner annotations into it at the
   usual viewer positions (known glyph footprint ``F``), runs the same removal, and reports over ``F``

   ``MAE = 1/|F| sum |I^(p) - T(p)|``,  ``MSE = 1/|F| sum (I^(p) - T(p))^2``,
   ``PSNR = 10 log10(1 / MSE)``

   on intensities scaled to [0, 1] by ``T``'s range, plus how many characters OCR can still read.
   It measures how well the method restores text on *this image's anatomy*; it is not an error on
   the removed patient text.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "region_metrics",
    "summarize_regions",
    "footprint_errors",
    "SelfTestResult",
    "synthetic_self_test",
]

# Fictitious overlay, laid out like a scanner viewer export (four corner blocks). No real data.
_SYNTHETIC_BLOCKS = {
    "top_left": ["DOE^JANE", "SYN-0012345", "01-Jan-1970 F", "MRI PROSTATE"],
    "top_right": ["Example Hospital", "02-Feb-2026 10:20:30"],
    "bottom_left": ["ST: 3.00 mm", "Images: 12/35", "Series: 901"],
    "bottom_right": ["WL: 1000 WW: 2000"],
}


# ---------------------------------------------------------------------------
# 1. Measured on the real text (eq. 40)
# ---------------------------------------------------------------------------

def region_metrics(regions) -> list[dict[str, Any]]:
    """One row per text region: box, action, ``retained(B)`` and ``c_audit(B)``.

    ``regions`` are :class:`prostate_deid.text_restore.RegionResult` objects. Duplicate rows (the
    same box and action reported twice) are dropped.
    """
    rows, seen = [], set()
    for r in regions:
        key = (r.slice_index, tuple(r.box), r.action)
        if key in seen:
            continue
        seen.add(key)
        x0, y0, x1, y1 = r.box
        rows.append({
            "slice": int(r.slice_index),
            "box": f"({x0}, {y0}) - ({x1}, {y1})",
            "action": r.action,
            "box_px": int(r.box_pixels),
            "changed_px": int(r.mask_pixels),
            "retained": float(r.retained_fraction),
            "c_audit": float(min(1.0, max(0.0, r.ocr_conf / 100.0))) if r.ocr_conf >= 0 else float("nan"),
        })
    return rows


def summarize_regions(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate :func:`region_metrics` rows: counts, mean / min retained fraction, mean c_audit.

    Only regions whose pixels were changed count toward the retained statistics (a region left
    untouched because it was not text has retained = 1 by construction).
    """
    changed = [r for r in rows if r["action"] != "skipped_not_text"]
    ret = [r["retained"] for r in changed]
    conf = [r["c_audit"] for r in rows if not math.isnan(r["c_audit"])]
    return {
        "regions": len(rows),
        "restored": sum(r["action"] == "restored" for r in rows),
        "fallback": sum(r["action"] in ("fallback_blackout", "blackout") for r in rows),
        "skipped": sum(r["action"] == "skipped_not_text" for r in rows),
        "retained_mean": float(np.mean(ret)) if ret else float("nan"),
        "retained_min": float(np.min(ret)) if ret else float("nan"),
        "c_audit_mean": float(np.mean(conf)) if conf else float("nan"),
    }


# ---------------------------------------------------------------------------
# 2. Error against a clean slice (eq. 41)
# ---------------------------------------------------------------------------

def _gray(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a[..., :3].mean(axis=-1) if a.ndim == 3 and a.shape[-1] in (3, 4) else a


def footprint_errors(output: np.ndarray, clean: np.ndarray, footprint: np.ndarray) -> dict[str, float]:
    """MAE, MSE and PSNR of ``output`` against ``clean`` over the boolean ``footprint`` (eq. 41).

    Both images are scaled to [0, 1] by the clean image's range, so PSNR uses a peak of 1.
    """
    t = _gray(clean)
    o = _gray(output)
    f = np.asarray(footprint, dtype=bool)
    lo, hi = float(t.min()), float(t.max())
    scale = (hi - lo) or 1.0
    err = ((o - lo) / scale - (t - lo) / scale)[f]
    if err.size == 0:
        return {"mae": float("nan"), "mse": float("nan"), "psnr_db": float("nan"), "footprint_px": 0}
    mae = float(np.abs(err).mean())
    mse = float((err ** 2).mean())
    psnr = float(10.0 * math.log10(1.0 / mse)) if mse > 0 else float("inf")
    return {"mae": mae, "mse": mse, "psnr_db": psnr, "footprint_px": int(f.sum())}


@dataclass
class SelfTestResult:
    mae: float
    mse: float
    psnr_db: float
    footprint_px: int
    chars_total: int
    chars_readable_before: int  # characters OCR reads on the synthetic input (reference)
    chars_readable_after: int  # characters OCR still reads after removal
    glyph_px_changed: float  # fraction of solid glyph pixels that were replaced
    outside_px_changed: int  # pixels changed away from the synthetic text (more than 2 px from F)
    seconds: float
    font_px: int
    burned: np.ndarray = field(repr=False)
    restored: np.ndarray = field(repr=False)
    footprint: np.ndarray = field(repr=False)

    @property
    def residual_fraction(self) -> float:
        """Characters still readable after removal, relative to those readable before it."""
        return self.chars_readable_after / self.chars_readable_before if self.chars_readable_before else 0.0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("burned", "restored", "footprint"):
            d.pop(k)
        d["residual_fraction"] = self.residual_fraction
        return d


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _render_synthetic(shape: tuple[int, int], font_px: int):
    """Alpha mask of the fictitious corner blocks and one box per line: ``(alpha, [(box, text)])``."""
    from PIL import Image, ImageDraw

    h, w = shape
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    f = _font(font_px)
    pitch = font_px + max(2, font_px // 4)
    margin = max(2, font_px // 4)
    lines = []
    for corner, texts in _SYNTHETIC_BLOCKS.items():
        top = corner.startswith("top")
        n = len(texts)
        for k, text in enumerate(texts):
            y = margin + k * pitch if top else h - margin - (n - k) * pitch
            tw = int(d.textlength(text, font=f))
            x = margin if corner.endswith("left") else w - margin - tw
            d.text((x, y), text, fill=255, font=f)
            lines.append(((max(0, x - 1), max(0, y - 1), min(w, x + tw + 1), min(h, y + pitch)), text))
    return np.asarray(layer, np.float64) / 255.0, lines


def synthetic_self_test(
    clean: np.ndarray,
    method: str = "telea",
    font_px: int | None = None,
    **restore_kw,
) -> SelfTestResult:
    """Burn fictitious corner text into ``clean``, remove it with ``method``, score the result.

    ``clean`` is one 2D (grayscale) or ``(H, W, C)`` slice -- typically the de-identified output.
    ``restore_kw`` is passed to :func:`prostate_deid.text_restore.restore_burned_in_text`
    (``upscale_factors``, ``check_inverted``, ``detect_overlay``).
    """
    from scipy import ndimage as ndi

    from .text_restore import read_crop_text, restore_burned_in_text
    from .universal_io import _ocr_available

    arr = np.asarray(clean)
    is_color = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    h, w = arr.shape[:2]
    fs = font_px or max(8, round(min(h, w) / 28))
    alpha, lines = _render_synthetic((h, w), fs)
    level = float(arr.max())
    a = alpha[..., None] if is_color else alpha
    burned = np.rint(arr.astype(np.float64) * (1 - a) + level * a)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        burned = np.clip(burned, info.min, info.max)
    burned = burned.astype(arr.dtype)

    t0 = time.perf_counter()
    res = restore_burned_in_text(burned[None], method=method, **restore_kw)
    seconds = time.perf_counter() - t0
    restored = res.pixels[0]

    footprint = alpha > 0.02
    errs = footprint_errors(restored, arr, footprint)
    glyph = alpha > 0.5
    changed = _gray(restored) != _gray(burned)
    near_text = ndi.binary_dilation(footprint, iterations=2)
    outside = int((changed & ~near_text).sum())

    total = sum(sum(ch.isalnum() for ch in text) for _, text in lines)
    before = after = 0
    if _ocr_available():
        gb, gr = _gray(burned), _gray(restored)
        lo, hi = float(gb.min()), float(gb.max()) or 1.0
        gb01 = ((gb - lo) / ((hi - lo) or 1.0)).astype(np.float32)
        gr01 = np.clip((gr - lo) / ((hi - lo) or 1.0), 0, 1).astype(np.float32)
        for box, text in lines:
            n_true = sum(ch.isalnum() for ch in text)
            before += min(n_true, read_crop_text(gb01, box, "bright", min_conf=40)[0])
            after += min(n_true, read_crop_text(gr01, box, None, min_conf=40)[0])

    return SelfTestResult(
        mae=errs["mae"], mse=errs["mse"], psnr_db=errs["psnr_db"], footprint_px=errs["footprint_px"],
        chars_total=total, chars_readable_before=before, chars_readable_after=after,
        glyph_px_changed=float(changed[glyph].mean()) if glyph.any() else float("nan"),
        outside_px_changed=outside, seconds=seconds, font_px=fs,
        burned=burned, restored=restored, footprint=footprint,
    )
