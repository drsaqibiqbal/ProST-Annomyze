"""
Burned-in text removal that keeps the image (v2).

v1 (:func:`prostate_deid.universal_io.redact_burned_in_text`) blacks out the
whole padded row box around every OCR-detected text line. That is safe, but
every pixel in the box is destroyed -- including the anatomy *between and
around* the glyph strokes when text is overlaid on tissue (fusion-biopsy
labels, measurement call-outs, DWI/ADC screenshot annotations). This module
keeps v1's detector unchanged and replaces only what was actually text:

1. **Detect** -- v1's multi-scale, both-polarity Tesseract detector
   (:func:`~prostate_deid.universal_io.detect_text_boxes`) proposes row
   boxes. Detection recall is therefore identical to v1.
2. **Segment** -- inside each box, glyph strokes are separated from the
   background with a morphological top-hat (thin bright/dark structure
   relative to a local opening/closing), and an exact per-pixel glyph mask is
   grown from high-confidence stroke seeds by bounded region growing, which
   picks up anti-aliased edges but cannot flood into large anatomy. A final
   safety dilation covers JPEG ringing / halo pixels.
3. **Restore** -- only masked pixels are re-synthesised from their
   surroundings. Methods: ``"telea"`` / ``"ns"`` (OpenCV fast-marching /
   Navier-Stokes), ``"biharmonic"`` (scikit-image), ``"vit_mae"`` (a
   pretrained ViT masked autoencoder, facebook/vit-mae-base, reconstructs
   exactly the 16x16 patches that contain text from the visible patches), and
   ``"vit_mae_poisson"`` (the ViT prediction used as a guidance field for a
   Poisson solve, so the fill's structure comes from the ViT and its
   boundary values match the true surrounding pixels exactly).
   Every pixel outside the glyph mask is copied bit-for-bit from the input.
4. **Verify** -- each restored region is checked again with the rendered-text
   pixel signature and re-OCR'd. If text is still found, that region falls
   back to v1's black box and is flagged for review (asymmetric-error policy:
   missed PHI is worse than over-redaction). If segmentation yields an
   implausible mask, the region also falls back.

What this does NOT do: it cannot recover the true pixel values that were
under the text -- the burned-in text overwrote them. Restored pixels are a
plausible estimate from context, not measurements. They are therefore
recorded per slice in an inpaint mask and in the audit trail, and must not be
used for quantitative analysis (e.g. ADC values) or diagnosis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .core import AuditEntry

__all__ = [
    "RESTORE_METHODS",
    "detect_text_boxes_v2",
    "detect_overlay_text_boxes",
    "read_crop_text",
    "ghost_ratio",
    "RegionResult",
    "RestoreResult",
    "segment_text_pixels",
    "inpaint_mask",
    "restore_burned_in_text",
    "vit_mae_available",
]

Method = Literal["blackout", "telea", "ns", "biharmonic", "vit_mae", "vit_mae_poisson"]
RESTORE_METHODS: tuple[str, ...] = ("blackout", "telea", "ns", "biharmonic", "vit_mae", "vit_mae_poisson")

_VIT_MAE_MODEL = "facebook/vit-mae-base"

# --- segmentation constants ([0,1] slice-intensity units; spatial scales come from measured stroke width) ---
_EXTREME = 0.12  # burned-in glyphs are rendered at the display extremes: within this of the slice max/min
_MIN_EXTREME_FRAC = 0.01  # an extreme class must cover at least this much of the box to be a text candidate
_SEED_CONTRAST = 0.35  # seed pixel: top-hat response >= this fraction of the box's text-to-background contrast
_GROW_CONTRAST = 0.10  # growth pixel: top-hat response >= this fraction (picks up anti-aliased glyph edges)
_SAFETY_DILATE = 1  # px of extra dilation after growing (JPEG ringing, sub-pixel edges)
_MIN_ABS_CONTRAST = 0.15  # below this the box has no clear text strokes
_MAX_MASK_FRAC = 0.85  # a glyph mask covering more of the box than this is not plausible text -> fall back


@dataclass
class RegionResult:
    """What happened to one detected text box."""

    slice_index: int
    box: tuple[int, int, int, int]  # x0, y0, x1, y1
    ocr_conf: int
    action: str  # "restored" | "blackout" | "fallback_blackout" | "skipped_not_text"
    reason: str
    polarity: str  # "bright" | "dark" | "n/a"
    mask_pixels: int
    box_pixels: int

    @property
    def retained_fraction(self) -> float:
        """Fraction of the box's pixels that were NOT altered."""
        if self.box_pixels == 0:
            return 1.0
        return 1.0 - self.mask_pixels / self.box_pixels


@dataclass
class RestoreResult:
    pixels: np.ndarray
    inpaint_mask: np.ndarray  # bool, (slices, H, W): True where pixel values were synthesised or blacked out
    regions: list[RegionResult] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    method: str = ""
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# Segmentation: exact glyph pixels inside a detected box
# ---------------------------------------------------------------------------

def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _stroke_width(binary: np.ndarray) -> float:
    """Typical stroke width (px) of a binary glyph layer: twice the 90th-percentile inscribed radius."""
    from scipy import ndimage as ndi

    if not binary.any():
        return 0.0
    dt = ndi.distance_transform_edt(binary)
    return float(2.0 * np.percentile(dt[binary], 90))


def segment_text_pixels(
    gray01: np.ndarray,
    box: tuple[int, int, int, int],
    safety_dilate: int = _SAFETY_DILATE,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Segment glyph pixels inside ``box`` of a ``[0, 1]`` grayscale slice.

    Returns ``(mask, info)`` where ``mask`` is a full-slice bool array (only
    pixels inside the box can be True), or ``None`` when no plausible glyph
    mask could be found -- the caller must then fall back to blacking out the
    whole box. ``info`` holds ``polarity``, ``stroke_px`` and a ``reason``.

    1. Polarity. Burned-in annotation glyphs are drawn at the display
       extremes (white or black). The near-max and near-min pixel classes of
       the box are candidates; when both occur (e.g. white text on a black
       corner) the text is the minority class, since a flat background
       occupies most of a text box. When neither occurs the polarity with the
       stronger thin-structure response is used.
    2. Scale. Stroke width is measured on the candidate class with a distance
       transform, and the top-hat disk is made just wider than one stroke --
       so it removes strokes but not anatomy, independent of image
       resolution or of how many text lines the detector merged into the box.
    3. Region growing. Seeds are extreme-valued pixels with a strong top-hat
       response; the mask grows from them by geodesic dilation through pixels
       with a weaker positive response (anti-aliased glyph edges), for at most
       about half a stroke width, so it cannot flood into bright anatomy.
    """
    from scipy import ndimage as ndi

    x0, y0, x1, y1 = box
    h_img, w_img = gray01.shape
    margin = max(3, (y1 - y0) // 4)
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(w_img, x1 + margin), min(h_img, y1 + margin)
    crop = gray01[cy0:cy1, cx0:cx1].astype(np.float32)
    in_box = np.zeros(crop.shape, dtype=bool)
    in_box[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = True
    n_box = int(in_box.sum())
    if n_box == 0:
        return None, {"polarity": "n/a", "stroke_px": 0.0, "reason": "empty box"}

    bright_c = (crop >= 1.0 - _EXTREME) & in_box
    dark_c = (crop <= _EXTREME) & in_box
    fb, fd = bright_c.sum() / n_box, dark_c.sum() / n_box
    candidates = []
    if fb >= _MIN_EXTREME_FRAC and fd >= _MIN_EXTREME_FRAC:
        # Both extremes occur (white text on a black corner, or white text next to dark anatomy).
        # The text is the class made of thin strokes: most of it disappears under a morphological
        # opening a little wider than its own stroke width, unlike a flat background or a broad
        # anatomical structure. Ties go to the minority class.
        def thin_fraction(cls):
            r = int(np.ceil(_stroke_width(cls) / 2)) + 1
            return float((cls & ~ndi.binary_opening(cls, structure=_disk(r))).sum() / max(1, cls.sum()))

        tb_, td_ = thin_fraction(bright_c), thin_fraction(dark_c)
        if abs(tb_ - td_) > 0.1:
            candidates = ["bright"] if tb_ > td_ else ["dark"]
        else:
            candidates = ["bright"] if fb <= fd else ["dark"]
    elif fb >= _MIN_EXTREME_FRAC:
        candidates = ["bright"]
    elif fd >= _MIN_EXTREME_FRAC:
        candidates = ["dark"]
    else:
        candidates = ["bright", "dark"]  # unsaturated text: decide by thin-structure energy below

    best = None
    for polarity in candidates:
        cls = bright_c if polarity == "bright" else dark_c
        saturated = bool(cls.sum() >= _MIN_EXTREME_FRAC * n_box)
        if not saturated:  # unsaturated text: use the Otsu side of this polarity as the stroke layer
            from skimage.filters import threshold_otsu

            t = threshold_otsu(crop[in_box]) if np.ptp(crop[in_box]) > 0 else 0.5
            cls = ((crop > t) if polarity == "bright" else (crop < t)) & in_box
        sw = max(1.0, _stroke_width(cls))
        footprint = _disk(int(np.ceil(sw)) + 1)
        if polarity == "bright":
            resp = crop - ndi.grey_opening(crop, footprint=footprint)
        else:
            resp = ndi.grey_closing(crop, footprint=footprint) - crop
        resp = np.where(in_box, resp, 0.0)
        contrast = float(np.percentile(resp[in_box], 99.5))
        if contrast < _MIN_ABS_CONTRAST:
            continue
        seeds = (resp >= _SEED_CONTRAST * contrast)
        if saturated:  # seeds must be extreme-valued (or very strongly thin) pixels
            seeds &= cls | (resp >= 0.6 * contrast)
        energy = float(resp[seeds].sum())
        if best is None or energy > best[0]:
            best = (energy, polarity, resp, contrast, seeds, sw)

    if best is None:
        return None, {"polarity": "n/a", "stroke_px": 0.0, "reason": "no thin-stroke contrast inside box"}

    _, polarity, resp, contrast, seeds, sw = best
    grow_ok = resp >= _GROW_CONTRAST * contrast
    # Growth may reach ~half a stroke plus 2 px: enough to take in anti-aliasing and the JPEG ringing
    # halo around glyphs (8x8-block codecs smear 2-4 px), which would otherwise survive as a faint
    # letter outline. grow_ok (a thin-structure response) keeps growth off flat background and anatomy.
    steps = max(2, int(np.ceil(sw / 2)) + 2)
    mask = seeds & grow_ok
    struct8 = np.ones((3, 3), dtype=bool)
    for _ in range(steps):  # bounded region growing (geodesic dilation inside grow_ok)
        grown = ndi.binary_dilation(mask, structure=struct8) & grow_ok
        if np.array_equal(grown, mask):
            break
        mask = grown
    if safety_dilate > 0:
        mask = ndi.binary_dilation(mask, structure=struct8, iterations=safety_dilate)
    mask &= in_box

    n_mask = int(mask.sum())
    info = {"polarity": polarity, "stroke_px": round(sw, 2)}
    if n_mask == 0:
        return None, {**info, "reason": "empty glyph mask"}
    if n_mask > _MAX_MASK_FRAC * n_box:
        return None, {**info, "reason": f"glyph mask covers {n_mask / n_box:.0%} of box (implausible)"}

    full = np.zeros(gray01.shape, dtype=bool)
    full[cy0:cy1, cx0:cx1] = mask
    return full, {**info, "reason": "segmented"}


# ---------------------------------------------------------------------------
# Inpainting backends
# ---------------------------------------------------------------------------

def _inpaint_cv2(img01: np.ndarray, mask: np.ndarray, flag: str) -> np.ndarray:
    import cv2

    radius = 3
    algo = cv2.INPAINT_TELEA if flag == "telea" else cv2.INPAINT_NS
    # 16-bit, not float32: OpenCV's float32 Telea path returns values far outside the input range.
    u16 = np.clip(np.rint(img01 * 65535.0), 0, 65535).astype(np.uint16)
    return cv2.inpaint(u16, mask.astype(np.uint8), radius, algo).astype(np.float32) / 65535.0


def _inpaint_biharmonic(img01: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from skimage.restoration import inpaint_biharmonic

    return inpaint_biharmonic(img01.astype(np.float64), mask).astype(np.float32)


_MAE_CACHE: dict[str, Any] = {}


def vit_mae_available() -> bool:
    """True if torch + transformers import and the ViT-MAE weights can be loaded (downloads once)."""
    try:
        _load_mae()
        return True
    except Exception:
        return False


def _load_mae():
    if "model" in _MAE_CACHE:
        return _MAE_CACHE["model"], _MAE_CACHE["mean"], _MAE_CACHE["std"]
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conda MKL + torch OpenMP clash on Windows
    import torch  # noqa: F401
    from transformers import ViTMAEForPreTraining

    model = ViTMAEForPreTraining.from_pretrained(_VIT_MAE_MODEL)
    model.eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    _MAE_CACHE.update(model=model, mean=mean, std=std)
    return model, mean, std


def _mae_window(win01: np.ndarray, win_mask: np.ndarray) -> np.ndarray:
    """Run ViT-MAE on one 224x224 window, masking exactly the patches that touch ``win_mask``."""
    import torch

    model, mean, std = _load_mae()
    size = model.config.image_size
    p = model.config.patch_size
    g = size // p
    assert win01.shape == (size, size)

    patch_has_text = win_mask.reshape(g, p, g, p).any(axis=(1, 3)).reshape(-1)
    n_masked = int(patch_has_text.sum())
    if n_masked == 0:
        return win01.copy()

    # Hide masked pixels from the encoder input as well (their patches are masked anyway; this only
    # guards against the text leaking through patch-boundary positional details).
    rgb = np.repeat(win01[None, :, :], 3, axis=0)
    x = (rgb - mean[:, None, None]) / std[:, None, None]
    x = torch.from_numpy(x[None].astype(np.float32))

    # Patches are kept in ascending noise order: give text patches noise 1, the rest 0, and set the
    # mask ratio so exactly n_masked patches are masked.
    noise = torch.from_numpy(patch_has_text.astype(np.float32))[None]
    old_ratio = model.config.mask_ratio
    model.config.mask_ratio = (n_masked - 0.5) / (g * g)
    try:
        with torch.no_grad():
            out = model(pixel_values=x, noise=noise)
        rec = model.unpatchify(out.logits)[0].numpy()  # (3, size, size), normalized space
    finally:
        model.config.mask_ratio = old_ratio
    rec = rec * std[:, None, None] + mean[:, None, None]
    return np.clip(rec.mean(axis=0), 0.0, 1.0).astype(np.float32)


def _inpaint_vit_mae(img01: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tile 224x224 windows over the masked area; each window is reconstructed by ViT-MAE.

    Returns ``(pasted, raw)``: ``pasted`` has only masked pixels replaced by the reconstruction;
    ``raw`` holds the reconstruction over every ViT patch that touches the mask (used as the guidance
    field for :func:`_poisson_blend`, whose boundary values then come from the real image).

    Windows are cut at native resolution (no rescale) when the slice is at least 224 px, so the
    16 px ViT patch is about one scanner text line. Smaller slices are upsampled to 224 first.
    """
    from PIL import Image

    model, _, _ = _load_mae()
    size = model.config.image_size
    p = model.config.patch_size
    h, w = img01.shape
    out = img01.copy()
    raw = img01.copy()

    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
        up = np.asarray(Image.fromarray(img01.astype(np.float32)).resize((nw, nh), Image.BICUBIC))
        up_mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)) > 0
        _, raw_up = _inpaint_vit_mae(up, up_mask)
        raw = np.asarray(Image.fromarray(raw_up).resize((w, h), Image.BICUBIC)).astype(np.float32)
        touched = np.asarray(Image.fromarray((np.abs(raw_up - up) > 0).astype(np.uint8) * 255)
                             .resize((w, h), Image.NEAREST)) > 0
        raw = np.where(touched | mask, raw, img01).astype(np.float32)
        out[mask] = raw[mask]
        return out, raw

    ys, xs = np.nonzero(mask)
    y_lo, y_hi, x_lo, x_hi = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    # Tile with a stride of half a window over the mask's bounding box; centre the text in each window
    # so the ViT sees context on all sides where the image allows it.
    todo = mask.copy()
    stride = size // 2
    for cy in range(y_lo, y_hi, stride):
        for cx in range(x_lo, x_hi, stride):
            sub = todo[cy:min(cy + stride, y_hi), cx:min(cx + stride, x_hi)]
            if not sub.any():
                continue
            wy0 = int(np.clip(cy + stride // 2 - size // 2, 0, h - size))
            wx0 = int(np.clip(cx + stride // 2 - size // 2, 0, w - size))
            win = img01[wy0:wy0 + size, wx0:wx0 + size].copy()
            win_mask = mask[wy0:wy0 + size, wx0:wx0 + size]
            win[win_mask] = 0.0
            rec = _mae_window(win, win_mask)
            g = size // p
            patch_px = np.repeat(np.repeat(win_mask.reshape(g, p, g, p).any(axis=(1, 3)), p, axis=0), p, axis=1)
            sel = todo[wy0:wy0 + size, wx0:wx0 + size]
            # the patches this window is responsible for: those containing still-unassigned mask pixels
            own = patch_px & np.repeat(np.repeat(sel.reshape(g, p, g, p).any(axis=(1, 3)), p, axis=0), p, axis=1)
            out[wy0:wy0 + size, wx0:wx0 + size][sel] = rec[sel]
            raw[wy0:wy0 + size, wx0:wx0 + size][own] = rec[own]
            todo[wy0:wy0 + size, wx0:wx0 + size] = False
    return out, raw


def _poisson_blend(img01: np.ndarray, mask: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """Solve lap(f) = lap(guide) inside ``mask`` with f = img01 on the mask boundary (Dirichlet).

    Gradient-domain blending (Perez et al. 2003): the fill takes its *structure* (gradients) from
    ``guide`` and its *level* from the true surrounding pixels, so a guide with a brightness offset
    (as ViT-MAE has on out-of-domain MRI) does not leave a visible glyph-shaped silhouette.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    out = img01.astype(np.float64).copy()
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n == 0:
        return out.astype(np.float32)
    index = -np.ones(mask.shape, dtype=np.int64)
    index[ys, xs] = np.arange(n)
    g = guide.astype(np.float64)
    rows, cols, vals = [], [], []
    diag = np.zeros(n)
    rhs = np.zeros(n)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        k = np.nonzero(ok)[0]
        yy, xx = yy[ok], xx[ok]
        diag[k] += 1
        rhs[k] += g[ys[k], xs[k]] - g[yy, xx]
        inside = mask[yy, xx]
        rows.append(k[inside])
        cols.append(index[yy[inside], xx[inside]])
        vals.append(-np.ones(int(inside.sum())))
        rhs[k[~inside]] += out[yy[~inside], xx[~inside]]
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    a = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    out[ys, xs] = spsolve(a, rhs)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def inpaint_mask(img01: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    """Fill ``mask`` pixels of a ``[0, 1]`` grayscale slice with ``method``; other pixels are untouched."""
    if not mask.any():
        return img01.copy()
    if method in ("telea", "ns"):
        filled = _inpaint_cv2(img01, mask, method)
    elif method == "biharmonic":
        filled = _inpaint_biharmonic(img01, mask)
    elif method == "vit_mae":
        filled = _inpaint_vit_mae(img01, mask)[0]
    elif method == "vit_mae_poisson":
        filled = _poisson_blend(img01, mask, _inpaint_vit_mae(img01, mask)[1])
    elif method == "blackout":
        filled = img01.copy()
        filled[mask] = 0.0
    else:
        raise ValueError(f"unknown restore method {method!r}; choose one of {RESTORE_METHODS}")
    out = img01.copy()
    out[mask] = filled[mask]
    return out


# ---------------------------------------------------------------------------
# Overlay-text detector (complements v1's whole-slice Tesseract pass)
# ---------------------------------------------------------------------------
#
# v1 runs Tesseract with --psm 6 on the whole slice. That finds corner annotations on a flat
# background, but text drawn ON anatomy (lesion labels, measurement call-outs) is missed: the
# anatomy texture breaks Tesseract's page-layout analysis, and v1's 300 px width normalization
# shrinks text on large slices below legibility. The benchmark (benchmarks/) measured most such
# characters still legible after v1. This detector proposes text lines from pixels alone
# (saturated thin strokes grouped into lines), then confirms each by OCR on the crop only, at
# native resolution, where Tesseract reads overlay text reliably.

_CROP_TARGET_H = 48  # crops are upscaled so a text line is ~this tall for Tesseract
_CROP_MIN_CONF = 60  # a word counts toward confirmation only above this OCR confidence
_CROP_MIN_CHARS = 3  # confirmed text line: at least this many alphanumeric chars in confident words
_VERIFY_MIN_CONF = 40  # after restoration, even moderately confident reads count as leaked text
_GHOST_RATIO = 1.5  # mask-boundary step / natural neighbour step above this = visible glyph silhouette
_MAX_CANDIDATES = 24  # OCR-verified candidates per slice per polarity; bounds cost on texture-rich slices
_STROKE_LEVEL = 0.45  # overlay stroke pixel: within this of the extreme (small anti-aliased fonts never saturate)
_STROKE_TOPHAT = 0.25  # ...and at least this much brighter/darker than its local opening/closing


def read_crop_text(
    gray01: np.ndarray,
    box: tuple[int, int, int, int],
    polarity: str | None = None,
    min_conf: float = _CROP_MIN_CONF,
) -> tuple[int, str]:
    """OCR one box at native resolution; return (confident alphanumeric char count, text).

    ``polarity`` "bright" means light text on dark (the crop is inverted so Tesseract sees dark
    on light); ``None`` tries both and keeps the better read.
    """
    import pytesseract
    from PIL import Image

    from .universal_io import _ocr_available

    _ocr_available()
    x0, y0, x1, y1 = box
    h, w = gray01.shape
    pad = max(3, (y1 - y0) // 3)
    crop = gray01[max(0, y0 - pad):min(h, y1 + pad), max(0, x0 - pad):min(w, x1 + pad)]
    if crop.size == 0 or float(crop.max()) - float(crop.min()) < 0.05:
        return 0, ""
    u8 = (np.clip((crop - crop.min()) / (crop.max() - crop.min()), 0, 1) * 255).astype(np.uint8)
    scale = float(np.clip(_CROP_TARGET_H / max(1, y1 - y0), 1.0, 8.0))
    best = (0, "")
    for pol in ([polarity] if polarity else ["bright", "dark"]):
        img = 255 - u8 if pol == "bright" else u8
        pil = Image.fromarray(img)
        pil = pil.resize((max(1, round(pil.width * scale)), max(1, round(pil.height * scale))), Image.LANCZOS)
        d = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config="--psm 7")
        words, n = [], 0
        for t, c in zip(d["text"], d["conf"]):
            try:
                c = float(c)
            except (TypeError, ValueError):
                c = -1
            if t.strip() and c >= min_conf:
                words.append(t)
                n += sum(ch.isalnum() for ch in t)
        if n > best[0]:
            best = (n, " ".join(words))
    return best


def propose_overlay_text_boxes(gray01: np.ndarray) -> list[tuple[int, int, int, int, str, int]]:
    """Candidate text lines from saturated thin strokes: ``(x0, y0, x1, y1, polarity, n_glyphs)``."""
    from scipy import ndimage as ndi

    cands = []
    for polarity in ("bright", "dark"):
        ext = gray01 >= 1.0 - _STROKE_LEVEL if polarity == "bright" else gray01 <= _STROKE_LEVEL
        if not ext.any():
            continue
        strokes = np.zeros(gray01.shape, dtype=bool)
        for r in (2, 4):
            fp = _disk(r)
            if polarity == "bright":
                th = gray01 - ndi.grey_opening(gray01, footprint=fp)
            else:
                th = ndi.grey_closing(gray01, footprint=fp) - gray01
            strokes |= ext & (th >= _STROKE_TOPHAT)
        lab, n = ndi.label(strokes, structure=np.ones((3, 3)))
        if n == 0:
            continue
        objs = ndi.find_objects(lab)
        heights = np.array([sl[0].stop - sl[0].start for sl in objs])
        sizes = np.asarray(ndi.sum(strokes, lab, index=np.arange(1, n + 1)))
        keep = (sizes >= 3) & (heights >= 3) & (heights <= 80)
        if not keep.any():
            continue
        glyphs = np.isin(lab, np.nonzero(keep)[0] + 1)
        hm = int(np.median(heights[keep]))
        # Link glyphs on one line: close horizontal gaps up to ~one glyph height (inter-letter and
        # word spaces), and a little vertically for ascenders/descenders.
        linked = ndi.binary_closing(glyphs, structure=np.ones((1, max(3, round(hm * 2.0)))))
        linked = ndi.binary_dilation(linked, structure=np.ones((max(1, hm // 4), 1)))
        llab, _ = ndi.label(linked)
        for k, sl in enumerate(ndi.find_objects(llab), start=1):
            if sl is None:
                continue
            ys, xs = sl
            hh, ww = ys.stop - ys.start, xs.stop - xs.start
            if not (4 <= hh <= 90) or ww < 1.5 * hh:
                continue
            sub = glyphs[sl] & (llab[sl] == k)
            n_glyphs = int(ndi.label(sub, structure=np.ones((3, 3)))[1])
            fill = sub.sum() / float(hh * ww)
            if n_glyphs < 1 or not (0.05 <= fill <= 0.75):
                continue
            cands.append((xs.start, ys.start, xs.stop, ys.stop, polarity, n_glyphs))
    cands.sort(key=lambda c: -c[5])
    return cands


def detect_overlay_text_boxes(gray01: np.ndarray, existing=(), padding: int = 3) -> list[tuple[int, int, int, int, int]]:
    """OCR-confirmed overlay text boxes not already covered by ``existing`` boxes.

    Confidence is reported on v1's 0-100 scale from the confirmed character count (>= 10 -> 100).
    """
    h, w = gray01.shape
    cands = propose_overlay_text_boxes(gray01)

    def overlap_frac(b, others):
        x0, y0, x1, y1 = b[:4]
        area = max(1, (x1 - x0) * (y1 - y0))
        best = 0.0
        for o in others:
            ix = max(0, min(x1, o[2]) - max(x0, o[0]))
            iy = max(0, min(y1, o[3]) - max(y0, o[1]))
            best = max(best, ix * iy / area)
        return best

    # 1. OCR-confirm the most glyph-rich candidates (>= 3 glyphs) not already covered by v1 boxes.
    confirmed: list[list] = []
    tried = {"bright": 0, "dark": 0}
    for c in cands:
        x0, y0, x1, y1, pol, n_glyphs = c
        if n_glyphs < 3 or tried[pol] >= _MAX_CANDIDATES or overlap_frac(c, existing) > 0.6:
            continue
        tried[pol] += 1
        n_chars, _ = read_crop_text(gray01, (x0, y0, x1, y1), pol)
        # Second, independent condition: real burned-in text yields a plausible glyph mask. OCR alone
        # "reads" a few characters in noisy tissue texture; requiring both keeps such boxes out
        # (they would otherwise be blacked out over anatomy).
        if n_chars >= _CROP_MIN_CHARS and segment_text_pixels(gray01, (x0, y0, x1, y1))[0] is not None:
            confirmed.append([x0, y0, x1, y1, pol, n_chars])

    # 2. Grow each confirmed line along its row: absorb every same-polarity candidate (confirmed or
    #    not -- e.g. a short trailing initial OCR could not confirm alone) that sits on the same text
    #    row within a few line-heights. Asymmetric-error policy: a stray word next to confirmed PHI
    #    is treated as PHI.
    changed = True
    while changed:
        changed = False
        for box in confirmed:
            for c in cands:
                if c[4] != box[4]:
                    continue
                bh = box[3] - box[1]
                vov = min(box[3], c[3]) - max(box[1], c[1])
                hgap = max(c[0] - box[2], box[0] - c[2], 0)
                inside = c[0] >= box[0] and c[2] <= box[2] and c[1] >= box[1] and c[3] <= box[3]
                if inside or vov < 0.5 * min(bh, c[3] - c[1]) or hgap > 3 * bh:
                    continue
                if (c[3] - c[1]) > 2 * bh:  # a much taller blob is anatomy, not a glyph on this row
                    continue
                box[0], box[1] = min(box[0], c[0]), min(box[1], c[1])
                box[2], box[3] = max(box[2], c[2]), max(box[3], c[3])
                changed = True

    out: list[tuple[int, int, int, int, int]] = []
    for x0, y0, x1, y1, _, n_chars in confirmed:
        p = max(padding, (y1 - y0) // 4)
        b = (max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p), min(100, n_chars * 10))
        if overlap_frac(b, out) < 0.9:
            out.append(b)
    return out


def detect_text_boxes_v2(
    frame: np.ndarray,
    upscale_factors: tuple[int, ...] | None = None,
    check_inverted: bool = True,
    padding: int = 4,
    detect_overlay: bool = True,
) -> list[tuple[int, int, int, int, int]]:
    """v1's detector boxes, plus OCR-confirmed overlay-text boxes v1 missed (see notes above)."""
    from .universal_io import detect_text_boxes

    boxes = detect_text_boxes(frame, upscale_factors=upscale_factors, check_inverted=check_inverted, padding=padding)
    if detect_overlay:
        is_color = frame.ndim == 3 and frame.shape[-1] in (3, 4)
        gray = frame[..., :3].mean(axis=-1) if is_color else frame
        lo, hi = float(gray.min()), float(gray.max())
        if hi > lo:
            g01 = ((gray - lo) / (hi - lo)).astype(np.float32)
            boxes = boxes + detect_overlay_text_boxes(g01, existing=boxes)
    return boxes


# ---------------------------------------------------------------------------
# Verification: is the restored region still text?
# ---------------------------------------------------------------------------

def ghost_ratio(restored01: np.ndarray, mask: np.ndarray) -> float:
    """How visible the glyph silhouette is: mean step across the mask boundary divided by the mean
    step between neighbouring pixels just outside it (the image's own texture level).

    ~1 means the fill continues the surroundings; a clear excess means the letter shapes can still be
    seen (e.g. a fill that is uniformly a little too bright), which is a PHI leak even when every
    original text pixel was replaced.
    """
    from scipy import ndimage as ndi

    if not mask.any():
        return 0.0
    ring = ndi.binary_dilation(mask, iterations=3) & ~mask
    h, w = mask.shape
    across, natural = [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = restored01[: h - dy, : w - dx]
        b = restored01[dy:, dx:]
        ma, mb = mask[: h - dy, : w - dx], mask[dy:, dx:]
        ra, rb = ring[: h - dy, : w - dx], ring[dy:, dx:]
        d = np.abs(a - b)
        across.append(d[ma ^ mb])
        natural.append(d[ra & rb])
    across_m = float(np.mean(np.concatenate(across))) if sum(x.size for x in across) else 0.0
    nat = np.concatenate(natural)
    natural_m = float(np.mean(nat)) if nat.size else 0.0
    return across_m / (natural_m + 1e-3)


def _boxes_still_text(
    restored01: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    upscale_factors,
    check_inverted,
) -> list[bool]:
    """Two independent checks per restored box; either one finding text means "still text".

    1. v1's detector on the whole restored slice (run on the full slice, not a crop, because its
       tuned size constants assume the slice's own resolution normalization).
    2. A native-resolution OCR read of the box crop (both polarities); at least _CROP_MIN_CHARS
       characters read with confidence >= _VERIFY_MIN_CONF means legible text survived.
    The pixel-level silhouette check (:func:`ghost_ratio`) is applied by the caller, per box mask.
    """
    from .universal_io import detect_text_boxes

    found = detect_text_boxes(restored01, upscale_factors=upscale_factors, check_inverted=check_inverted, padding=0)
    flags = []
    for x0, y0, x1, y1 in boxes:
        area = max(1, (x1 - x0) * (y1 - y0))
        hit = False
        for fx0, fy0, fx1, fy1, _ in found:
            f_area = max(1, (fx1 - fx0) * (fy1 - fy0))
            inter = max(0, min(x1, fx1) - max(x0, fx0)) * max(0, min(y1, fy1) - max(y0, fy0))
            # Only a detection that lies mostly inside this region and is not much larger than it counts:
            # on noisy screenshots Tesseract's whole-slice layout pass can return slice-sized junk rows
            # after the text is gone, which overlap every region without meaning anything is left.
            if inter >= 0.5 * f_area and f_area <= 2 * area:
                hit = True
                break
        if not hit:
            hit = read_crop_text(restored01, (x0, y0, x1, y1), min_conf=_VERIFY_MIN_CONF)[0] >= _CROP_MIN_CHARS
        flags.append(hit)
    return flags


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def restore_burned_in_text(
    pixels: np.ndarray,
    method: str = "telea",
    padding: int = 4,
    flag_confidence: int = 40,
    upscale_factors: tuple[int, ...] | None = None,
    check_inverted: bool = True,
    verify: bool = True,
    boxes_per_slice: list[list[tuple[int, int, int, int, int]]] | None = None,
    detect_overlay: bool = True,
) -> RestoreResult:
    """Detect burned-in text and remove it while keeping the image.

    Parameters
    ----------
    pixels:
        ``(slices, H, W)`` grayscale or ``(slices, H, W, C)`` colour array, any numeric dtype.
        The returned array has the same dtype; every pixel outside the inpaint mask is
        bit-identical to the input.
    method:
        ``"blackout"`` (v1 behaviour: whole box set to 0), or a restoration method from
        :data:`RESTORE_METHODS`. ``"telea"`` is the recommended default (best balance of privacy,
        retention and fidelity in ``benchmarks/``; see docs/v2_analysis.md). ``"vit_mae_poisson"``
        uses the ViT; plain ``"vit_mae"`` is research-only (it can leave a faint glyph silhouette)
        and every use is flagged. ViT methods need ``torch`` + ``transformers`` (``[vit]`` extra);
        if unavailable they fall back to ``"biharmonic"`` with a flagged audit entry.
    verify:
        Re-check every restored region and fall back to blacking out the whole box if text is
        still detectable. Strongly recommended; disabling it is for benchmarking only.
    detect_overlay:
        Also run the overlay-text detector (:func:`detect_overlay_text_boxes`) that finds text
        drawn on anatomy, which v1's whole-slice OCR pass misses. Adds OCR calls on candidate
        crops only; disable for the lightest performance mode.
    boxes_per_slice:
        Pre-computed detector boxes (skips OCR detection). Used by the benchmark to compare
        methods on identical detections, and by callers that already ran detection.
    """
    from .universal_io import _UPSCALE_FACTORS, _ocr_available

    t_start = time.perf_counter()
    audit: list[AuditEntry] = []
    notes: list[str] = []
    arr = np.asarray(pixels)
    orig_dtype = arr.dtype
    n_slices = arr.shape[0]
    is_color = arr.ndim == 4 and arr.shape[-1] in (3, 4)
    spatial = arr.shape[1:3]
    inpaint_mask_all = np.zeros((n_slices, *spatial), dtype=bool)

    if boxes_per_slice is None and not _ocr_available():
        msg = ("OCR backend unavailable (pytesseract or tesseract binary not found); "
               "pixel data was NOT inspected for burned-in text -- treat this file as unredacted.")
        audit.append(AuditEntry("burned_in_phi", "ocr_unavailable", msg, flagged_for_review=True))
        return RestoreResult(arr.copy(), inpaint_mask_all, [], audit, [msg], method, 0.0)

    if method not in RESTORE_METHODS:
        raise ValueError(f"unknown restore method {method!r}; choose one of {RESTORE_METHODS}")
    if method == "vit_mae":
        audit.append(AuditEntry(
            "burned_in_phi", "research_only_method",
            "vit_mae pastes the ViT reconstruction directly; the benchmark showed it can leave a faint "
            "glyph silhouette that verification does not always catch -- use vit_mae_poisson",
            flagged_for_review=True,
        ))
    if method.startswith("vit_mae") and not vit_mae_available():
        audit.append(AuditEntry(
            "burned_in_phi", "vit_unavailable",
            f"{method} requested but torch/transformers/{_VIT_MAE_MODEL} unavailable; used biharmonic instead",
            flagged_for_review=True,
        ))
        notes.append(f"{method} unavailable -> biharmonic inpainting used instead")
        method = "biharmonic"

    factors = tuple(upscale_factors) if upscale_factors else _UPSCALE_FACTORS
    out = arr.astype(np.float64).copy()
    regions: list[RegionResult] = []

    for i in range(n_slices):
        frame = out[i]
        gray = frame[..., :3].mean(axis=-1) if is_color else frame
        lo, hi = float(gray.min()), float(gray.max())
        if hi <= lo:
            continue
        if boxes_per_slice is not None:
            boxes = boxes_per_slice[i]
        else:
            boxes = detect_text_boxes_v2(arr[i].astype(np.float32), upscale_factors=factors,
                                         check_inverted=check_inverted, padding=padding,
                                         detect_overlay=detect_overlay)
        if not boxes:
            continue

        gray01 = ((gray - lo) / (hi - lo)).astype(np.float32)
        slice_mask = np.zeros(spatial, dtype=bool)  # pixels to synthesise this slice
        black_mask = np.zeros(spatial, dtype=bool)  # pixels to black out this slice
        pending: list[tuple[tuple[int, int, int, int], int, dict, int]] = []

        for x0, y0, x1, y1, conf in boxes:
            box = (x0, y0, x1, y1)
            n_box = (x1 - x0) * (y1 - y0)
            if method == "blackout":
                black_mask[y0:y1, x0:x1] = True
                regions.append(RegionResult(i, box, conf, "blackout", "method=blackout", "n/a", n_box, n_box))
                continue
            m, info = segment_text_pixels(gray01, box)
            if m is None and read_crop_text(gray01, box, min_conf=_VERIFY_MIN_CONF)[0] < _CROP_MIN_CHARS:
                # Two independent checks say this detector box is not text: no plausible glyph strokes,
                # and a native-resolution OCR read finds no legible characters (typical of v1's
                # whole-slice pass "reading" noisy tissue). Blacking it out would destroy anatomy, so
                # leave it untouched and flag it for human review instead.
                regions.append(RegionResult(i, box, conf, "skipped_not_text",
                                            info["reason"] + "; crop OCR found no legible text",
                                            info["polarity"], 0, n_box))
                continue
            if m is None:
                black_mask[y0:y1, x0:x1] = True
                regions.append(RegionResult(i, box, conf, "fallback_blackout", info["reason"],
                                            info["polarity"], n_box, n_box))
                continue
            slice_mask |= m
            pending.append((box, conf, info, n_box))

        restored01 = inpaint_mask(gray01, slice_mask, method) if slice_mask.any() else gray01

        still = (_boxes_still_text(restored01, [p[0] for p in pending], factors, check_inverted)
                 if verify and pending else [False] * len(pending))
        for (box, conf, info, n_box), leaked in zip(pending, still):
            x0, y0, x1, y1 = box
            box_mask = slice_mask[y0:y1, x0:x1]
            ghost = ghost_ratio(restored01[y0:y1, x0:x1], box_mask) if verify else 0.0
            if leaked or ghost > _GHOST_RATIO:
                black_mask[y0:y1, x0:x1] = True
                why = "text still detected after restoration" if leaked else f"glyph silhouette visible (ghost ratio {ghost:.2f})"
                regions.append(RegionResult(i, box, conf, "fallback_blackout", why, info["polarity"], n_box, n_box))
            else:
                regions.append(RegionResult(i, box, conf, "restored", info["reason"], info["polarity"],
                                            int(box_mask.sum()), n_box))

        # Write back in the slice's own intensity units. Colour slices: each channel is restored with
        # the same mask, using the channel's own range.
        synth = slice_mask & ~black_mask
        if synth.any():
            if is_color:
                for c in range(frame.shape[-1]):
                    ch = frame[..., c]
                    clo, chi = float(ch.min()), float(ch.max())
                    if chi <= clo:
                        continue
                    ch01 = ((ch - clo) / (chi - clo)).astype(np.float32)
                    rest = inpaint_mask(ch01, synth, method)
                    ch[synth] = rest[synth] * (chi - clo) + clo
            else:
                frame[synth] = restored01[synth] * (hi - lo) + lo
        if black_mask.any():
            frame[black_mask] = 0
        inpaint_mask_all[i] = synth | black_mask

    # Cast back; pixels outside the mask are restored from the untouched input so they are bit-identical.
    if np.issubdtype(orig_dtype, np.integer):
        info_t = np.iinfo(orig_dtype)
        cast = np.clip(np.rint(out), info_t.min, info_t.max).astype(orig_dtype)
    else:
        cast = out.astype(orig_dtype)
    m_full = inpaint_mask_all[..., None] if is_color else inpaint_mask_all
    final = np.where(m_full, cast, arr)

    n_restored = sum(r.action == "restored" for r in regions)
    n_fallback = sum(r.action == "fallback_blackout" for r in regions)
    n_black = sum(r.action == "blackout" for r in regions)
    n_skip = sum(r.action == "skipped_not_text" for r in regions)
    box_px = sum(r.box_pixels for r in regions)
    changed_px = int(inpaint_mask_all.sum())
    for r in regions:
        flagged = r.action in ("fallback_blackout", "skipped_not_text", "restored") or r.ocr_conf < flag_confidence
        audit.append(AuditEntry(
            step="burned_in_phi",
            action={"restored": "restored_region", "blackout": "redacted_region",
                    "fallback_blackout": "redacted_region_fallback",
                    "skipped_not_text": "skipped_region_not_text"}[r.action],
            detail=(f"slice={r.slice_index} bbox={r.box} method={method if r.action == 'restored' else 'blackout'} "
                    f"polarity={r.polarity} glyph_px={r.mask_pixels}/{r.box_pixels} ({r.reason})"),
            confidence=max(0.0, min(1.0, r.ocr_conf / 100.0)) if r.ocr_conf >= 0 else None,
            flagged_for_review=flagged,
        ))

    if not regions:
        notes.append("OCR ran on all slices; no burned-in text detected")
    else:
        notes.append(
            f"burned-in text: {len(regions)} region(s) across {n_slices} slice(s) -- "
            f"{n_restored} restored with {method}, {n_fallback} fell back to blackout, {n_black} blacked out, "
            f"{n_skip} left untouched as not text (review)"
        )
        if box_px:
            notes.append(
                f"{changed_px} pixel(s) altered vs {box_px} inside detected boxes "
                f"({1 - changed_px / box_px:.1%} of box pixels retained bit-identical)"
            )
        if n_restored:
            notes.append("restored pixels are synthesised estimates (see inpaint mask) -- "
                         "not valid for quantitative analysis or diagnosis; human review recommended")
        if n_fallback:
            notes.append(f"{n_fallback} region(s) blacked out because segmentation or verification failed -- review")
    if factors != _UPSCALE_FACTORS or not check_inverted:
        notes.append(
            f"reduced-thoroughness OCR pass (upscale_factors={factors}, check_inverted={check_inverted}) "
            "-- lower recall than the full default; recommend human review"
        )

    return RestoreResult(final, inpaint_mask_all, regions, audit, notes, method, time.perf_counter() - t_start)
