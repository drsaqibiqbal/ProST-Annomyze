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
    "changed_region_boxes",
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
_TEXT_CROSSING_DENSITY = 0.10  # stroke crossings per px of row length; text lines exceed it, tissue does not
_GLYPH_CORE = 0.06  # a glyph component's core must reach within this of the slice max (bright) / min (dark)

# --- stroke-line detector constants (text that is NOT at the display extreme, e.g. drawn over anatomy) ---
_LINE_TOPHAT = 0.12  # thin-stroke pixel: >= this much brighter/darker than a small local opening/closing
_LINE_MIN_GLYPHS = 3  # a text line needs at least this many full-height glyphs
_LINE_HEIGHT_CV = 0.25  # full-height glyphs of one line vary in height by at most this (coefficient of variation)
_LINE_BASELINE = 0.20  # ...and their bottoms lie within this fraction of the line height of each other
_LINE_MIN_H, _LINE_MAX_H = 4, 48  # plausible overlay glyph heights, px
_LINE_MAX_GAP = 0.8  # median gap between neighbouring full-height glyphs, as a fraction of line height
_BORDER_ZONE = 0.12  # lines starting within this fraction of the slice size from an edge: annotation zone
_STROKE_MIN_SIZE = 512  # smaller slices are upscaled to at least this for stroke-line detection
_STROKE_EXTREME = 0.35  # second stroke map: only stroke pixels within this of the slice max (bright) / min (dark)
_BG_DARK = 0.12  # a glyph whose surroundings stay below this (90th pct) sits on background: fill flat
_GRID_MAX_GAP = 4  # line-grid completion fills at most this many missing lines between two confirmed ones
_ROW_EXTEND = 5  # a confirmed line takes in strokes along its row up to this many line heights past each end
_BLOCK_END_ROWS = 2  # rows past the inner end of a corner block that OCR may confirm as text


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


def _keep_glyph_components(crop: np.ndarray, mask: np.ndarray, polarity: str) -> tuple[np.ndarray, int]:
    """Drop stroke components that are anatomy, not glyphs. Returns (mask, number rejected).

    Thin bright (or dark) anatomy -- the subcutaneous fat rim, fat septa, vessel walls -- passes the
    top-hat test exactly like a letter stroke. Two properties separate burned-in glyphs from it:

    * Saturation: overlay text is rendered at the display extreme, so every glyph has a core at
      (almost) the slice maximum / minimum; tissue rarely reaches it.
    * Shape: glyphs on one text line share a height. A component much taller than the line's
      glyph height, or a long thin line (wide and much flatter than a glyph), is anatomy.

    The glyph height is the median height of the saturated components, so the rule adapts to font
    size and resolution.
    """
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return mask, 0
    objs = ndi.find_objects(lab)
    if polarity == "bright":
        core = np.asarray(ndi.maximum(crop, lab, index=np.arange(1, n + 1)))
        saturated = core >= 1.0 - _GLYPH_CORE
    else:
        core = np.asarray(ndi.minimum(crop, lab, index=np.arange(1, n + 1)))
        saturated = core <= _GLYPH_CORE
    heights = np.array([sl[0].stop - sl[0].start for sl in objs])
    widths = np.array([sl[1].stop - sl[1].start for sl in objs])
    ref = heights[saturated & (heights >= 3)]
    if ref.size == 0:
        return np.zeros_like(mask), n
    hg = float(np.median(ref))
    shape_ok = (heights <= 1.8 * hg) & ~((widths > 3 * hg) & (heights < 0.5 * hg))
    keep = saturated & shape_ok
    # Small anti-aliased glyphs (thin letters, punctuation, JPEG-softened strokes) may not reach the
    # extreme. Accept such a component when it sits on the same text line as an accepted glyph:
    # vertically overlapping it by half its height and within ~3 glyph heights horizontally,
    # chaining along the line. Tissue off the text rows never qualifies.
    y0s = np.array([sl[0].start for sl in objs])
    y1s = np.array([sl[0].stop for sl in objs])
    x0s = np.array([sl[1].start for sl in objs])
    x1s = np.array([sl[1].stop for sl in objs])
    cand = shape_ok & ~keep
    changed = True
    while changed and cand.any():
        changed = False
        for c in np.nonzero(cand)[0]:
            anchors = np.nonzero(keep)[0]
            vov = np.minimum(y1s[anchors], y1s[c]) - np.maximum(y0s[anchors], y0s[c])
            hgap = np.maximum(np.maximum(x0s[anchors] - x1s[c], x0s[c] - x1s[anchors]), 0)
            if np.any((vov >= 0.5 * heights[c]) & (hgap <= 3 * hg)):
                keep[c], cand[c], changed = True, False, True
    keep_ids = np.nonzero(keep)[0] + 1
    return np.isin(lab, keep_ids), int(n - keep.sum())


def _text_evidence(gray01: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray) -> tuple[bool, str]:
    """Positive evidence that a segmented region is a text line, not tissue.

    Measured on synthetic pelvis phantoms with bright fat rims / septa (docs/v2_analysis.md §7):
    text zones have >= 0.10 light/dark stroke crossings per pixel of row length (tissue <= 0.098),
    or OCR reads >= 3 characters at confidence >= 60 (never seen on tissue).
    """
    zb = _glyph_zone_bbox(mask[box[1]:box[3], box[0]:box[2]])
    if zb is None:
        return False, "empty glyph mask"
    x0, y0 = box[0] + zb[0], box[1] + zb[1]
    x1, y1 = box[0] + zb[2], box[1] + zb[3]
    from scipy import ndimage as ndi

    region = gray01[y0:y1, x0:x1]
    sel = mask[y0:y1, x0:x1]
    ring = ndi.binary_dilation(sel, iterations=2) & ~sel
    if not sel.any() or not ring.any():
        return False, "empty glyph mask"
    # Binarize halfway between the glyphs and their immediate surroundings, toward the glyph polarity,
    # so dark text on dark-ish tissue (or bright text on bright tissue) is measured correctly.
    g_in, g_out = float(np.median(region[sel])), float(np.median(region[ring]))
    thr = (g_in + g_out) / 2
    # Bright text keeps the absolute 0.6 floor (bright fat on darker tissue must not look like text).
    z = region >= max(thr, 0.6) if g_in >= g_out else region <= min(thr, 0.4)
    rows = int(z.any(axis=1).sum())
    density = float(np.abs(np.diff(z.astype(np.int8), axis=1)).sum()) / max(1, rows * (x1 - x0))
    if density >= _TEXT_CROSSING_DENSITY:
        return True, f"stroke density {density:.2f}"
    n = read_crop_text(gray01, (x0, y0, x1, y1), min_conf=_CROP_MIN_CONF)[0]
    if n >= _CROP_MIN_CHARS:
        return True, f"OCR read {n} char(s)"
    return False, f"stroke density {density:.2f} and only {n} OCR char(s)"


def _glyph_groups(mask: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Split a glyph mask into clusters of neighbouring components (words / text lines).

    Components are linked across horizontal gaps up to ~1.5 glyph heights (letter and word spacing)
    and small vertical offsets; tissue that is not adjacent to text ends up in its own cluster.
    """
    from scipy import ndimage as ndi

    x0, y0, x1, y1 = box
    local = mask[y0:y1, x0:x1]
    lab, n = ndi.label(local, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return [mask] if n == 1 else []
    hs = np.array([sl[0].stop - sl[0].start for sl in ndi.find_objects(lab)])
    hg = max(3, int(np.median(hs)))
    linked = ndi.binary_dilation(local, structure=np.ones((max(1, hg // 3), max(3, int(1.5 * hg)))))
    glab, gn = ndi.label(linked)
    out = []
    for k in range(1, gn + 1):
        g = np.zeros(mask.shape, dtype=bool)
        g[y0:y1, x0:x1] = local & (glab == k)
        if g.any():
            out.append(g)
    return out


def _resolve_box(gray01: np.ndarray, box: tuple[int, int, int, int]):
    """Decide what to do with one detector box. Yields ``(kind, box, payload)`` with kind in
    {"glyphs", "blackout", "skip"}.

    * Plausible glyph mask with text evidence -> restore those glyphs.
    * Otherwise (mask implausible, or no text evidence -- typically a v1 box that also spans
      anatomy) split it into text-line bands of saturated strokes and resolve each band the same way.
    * A band (or box) with no plausible mask but legible OCR text -> black it out (v1 policy).
    * Nothing text-like -> leave untouched, flagged ("skip").
    """
    def single(b, allow_split):
        m, info = segment_text_pixels(gray01, b)
        if m is not None:
            # Judge each cluster of neighbouring glyphs (a word / text line) on its own, so a bright
            # tissue line running through the same rows neither hides the text nor gets restored.
            accepted = np.zeros_like(m)
            whys, rejected = [], 0
            for gm in _glyph_groups(m, b):
                ok, why = _text_evidence(gray01, b, gm)
                if ok:
                    accepted |= gm
                    whys.append(why)
                else:
                    rejected += 1
            if accepted.any():
                note = f"segmented ({whys[0]}" + (f"; {rejected} non-text cluster(s) left untouched)" if rejected else ")")
                return [("glyphs", b, (accepted, {**info, "reason": note}))]
            reason = f"no text evidence: {why}"
        else:
            reason = info["reason"]
        if allow_split:
            bands = [bb for bb in _text_line_bands(gray01, b) if bb != b]
            if bands:
                out = []
                for bb in bands:
                    out += single(bb, False)
                if any(k != "skip" for k, _, _ in out):
                    return [o for o in out if o[0] != "skip"] + [o for o in out if o[0] == "skip"]
        if m is None and read_crop_text(gray01, b, min_conf=_VERIFY_MIN_CONF)[0] >= _CROP_MIN_CHARS:
            return [("blackout", b, {**info, "reason": reason + "; legible text -> blacked out"})]
        if m is None and "implausible" in reason and _has_saturated_strokes(gray01, b) and not allow_split:
            return [("blackout", b, {**info, "reason": reason + " (text-line band) -> blacked out"})]
        return [("skip", b, {**info, "reason": reason + "; not text -> left untouched"})]

    yield from single(box, True)


def _saturated_strokes(gray01: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Box-local map of saturated thin strokes (either polarity): candidate glyph cores."""
    from scipy import ndimage as ndi

    x0, y0, x1, y1 = box
    crop = gray01[y0:y1, x0:x1].astype(np.float32)
    fp = _disk(2)
    bright = (crop >= 1.0 - _GLYPH_CORE) & ((crop - ndi.grey_opening(crop, footprint=fp)) >= 0.25)
    dark = (crop <= _GLYPH_CORE) & ((ndi.grey_closing(crop, footprint=fp) - crop) >= 0.25)
    return bright | dark


def _has_saturated_strokes(gray01: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    return bool(_saturated_strokes(gray01, box).sum() >= 8)


def _text_line_bands(gray01: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Split a mixed box into bands of rows that contain saturated thin strokes (text lines).

    Each band is cropped horizontally to the columns holding those strokes (plus padding). Bands
    that are taller than ~3x their stroke content would suggest are not split further.
    """
    from scipy import ndimage as ndi

    x0, y0, x1, y1 = box
    strokes = _saturated_strokes(gray01, box)
    if strokes.sum() < 8:
        return []
    rows = ndi.binary_dilation(strokes.any(axis=1), iterations=2)
    lab, n = ndi.label(rows)
    bands = []
    h, w = gray01.shape
    for k in range(1, n + 1):
        ys = np.nonzero(lab == k)[0]
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        cols = np.nonzero(strokes[by0:by1].any(axis=0))[0]
        if cols.size == 0 or strokes[by0:by1].sum() < 8:
            continue
        pad = max(3, (by1 - by0) // 3)
        bands.append((max(0, x0 + int(np.min(cols)) - pad), max(0, y0 + by0 - 1),
                      min(w, x0 + int(np.max(cols)) + 1 + pad), min(h, y0 + by1 + 1)))
    return bands


@dataclass
class StrokeLine:
    """One text line found by :func:`detect_stroke_text_lines`."""

    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (padded)
    polarity: str
    mask: np.ndarray  # full-slice bool glyph mask (safety-dilated)
    n_glyphs: int
    height: float
    inferred: bool = False  # True: filled in from the line grid of its block, not measured on its own


def detect_stroke_text_lines(gray01: np.ndarray) -> list[StrokeLine]:
    """Find rendered text lines from their stroke geometry alone, with glyph masks (see
    :func:`_detect_stroke_text_lines_native`).

    Small slices (exported 224-256 px screenshots) carry ~6 px glyphs with 1-2 px line spacing; JPEG
    breaks their strokes into fragments and merges neighbouring lines. They are upscaled (bicubic) to
    at least ``_STROKE_MIN_SIZE`` px for detection, and each glyph mask is mapped back with an
    any-pixel reduction so every original pixel a stroke touched is included.
    """
    from scipy import ndimage as ndi

    h, w = gray01.shape
    k = int(np.ceil(_STROKE_MIN_SIZE / min(h, w)))
    if k <= 1:
        return _detect_stroke_text_lines_native(gray01)
    from PIL import Image

    big = Image.fromarray(gray01.astype(np.float32)).resize((w * k, h * k), Image.BICUBIC)
    g = gray01.astype(np.float32)
    weak_native = {  # weak-stroke maps at native resolution, for rows filled in from the line grid
        "bright": (g - ndi.grey_opening(g, footprint=_disk(2))) >= 0.5 * _LINE_TOPHAT,
        "dark": (ndi.grey_closing(g, footprint=_disk(2)) - g) >= 0.5 * _LINE_TOPHAT,
    }
    lines = []
    for ln in _detect_stroke_text_lines_native(np.clip(np.asarray(big), 0.0, 1.0)):
        m = ln.mask[: h * k, : w * k].reshape(h, k, w, k).any(axis=(1, 3))
        x0, y0, x1, y1 = ln.box
        box = (x0 // k, y0 // k, min(w, -(-x1 // k)), min(h, -(-y1 // k)))
        if ln.inferred:
            # Faint strokes lose contrast in the bicubic upscale; take the row's stroke pixels from
            # the native slice (the row's position came from the line grid, not from these pixels).
            band = np.zeros_like(m)
            band[box[1]:box[3], box[0]:box[2]] = weak_native[ln.polarity][box[1]:box[3], box[0]:box[2]]
            m |= ndi.binary_dilation(band, structure=np.ones((3, 3), dtype=bool), iterations=_SAFETY_DILATE + 1)
        lines.append(StrokeLine(box, ln.polarity, m, ln.n_glyphs, ln.height / k, ln.inferred))
    return lines


def _detect_stroke_text_lines_native(gray01: np.ndarray) -> list[StrokeLine]:
    """Find rendered text lines from their stroke geometry alone, with glyph masks.

    Complements the OCR detectors and the saturation-based segmentation, which both assume overlay
    text is the brightest (or darkest) thing on the slice. That fails when anatomy is brighter than
    the text (bright fat or bladder on T2/ADC) and when text is drawn over anatomy, where OCR cannot
    read it. Here a stroke pixel only has to stand out from its *immediate* surroundings: a small
    top-hat (disk just wider than a glyph stroke) responds to 1-3 px letter strokes but not to
    anatomy, which is broader.

    Thin anatomy (fat septa, vessel walls) also passes a top-hat, so a group of strokes is accepted
    as a text line only when it has the regularity of typeset text: at least ``_LINE_MIN_GLYPHS``
    full-height glyphs whose heights agree (coefficient of variation <= ``_LINE_HEIGHT_CV``) and whose
    bottoms share a baseline (spread <= ``_LINE_BASELINE`` x line height).
    """
    from scipy import ndimage as ndi

    h, w = gray01.shape
    r = max(2, round(min(h, w) / 256))  # stroke scale grows with matrix size (1-2 px at 512, 3 px at 800)
    # Smallest legible overlay glyph also grows with the matrix (4 px at 512, 8 px at 1024): tissue
    # texture on large matrices forms short runs of tiny, aligned "glyphs".
    min_h = max(_LINE_MIN_H, min(h, w) / 128)
    g = gray01.astype(np.float32)
    struct8 = np.ones((3, 3), dtype=bool)
    lines: list[StrokeLine] = []
    for polarity in ("bright", "dark"):
        measured: list[StrokeLine] = []
        if polarity == "bright":
            th = g - ndi.grey_opening(g, footprint=_disk(r))
        else:
            th = ndi.grey_closing(g, footprint=_disk(r)) - g
        # Two stroke maps: every thin bright/dark structure, and only those at the display extreme.
        # On speckled tissue (ADC) the first merges letters with same-sized tissue specks; the second
        # keeps saturated overlay glyphs apart from tissue that does not reach the extreme.
        extreme = g >= 1.0 - _STROKE_EXTREME if polarity == "bright" else g <= _STROKE_EXTREME
        # JPEG leaves only stroke cores near the extreme: reconnect them by one px of growth through
        # stroke response, so each glyph is one component again.
        cores = ndi.binary_dilation((th >= _LINE_TOPHAT) & extreme, structure=struct8) & (th >= 0.5 * _LINE_TOPHAT)
        for strokes in (th >= _LINE_TOPHAT, cores):
            lab, n = ndi.label(strokes, structure=struct8)
            if n == 0:
                continue
            objs = ndi.find_objects(lab)
            hs = np.array([sl[0].stop - sl[0].start for sl in objs])
            ws = np.array([sl[1].stop - sl[1].start for sl in objs])
            sizes = np.asarray(ndi.sum(strokes, lab, index=np.arange(1, n + 1)))
            glyph_like = (sizes >= 2) & (hs <= _LINE_MAX_H) & (ws <= 3 * np.maximum(hs, 2) + 2)
            if not glyph_like.any():
                continue
            glyphs = np.isin(lab, np.nonzero(glyph_like)[0] + 1)
            tall = glyph_like & (hs >= min_h)
            hm = float(np.median(hs[tall])) if tall.any() else float(min_h)
            # Link letters and words of one line: horizontal gaps up to ~1.5 glyph heights.
            linked = ndi.binary_closing(glyphs, structure=np.ones((1, max(3, round(1.5 * hm)))))
            llab, nl = ndi.label(linked, structure=struct8)
            for k, sl in enumerate(ndi.find_objects(llab), start=1):
                if sl is None:
                    continue
                ids = np.unique(lab[sl][(llab[sl] == k) & glyphs[sl]])
                ids = ids[ids > 0] - 1
                if ids.size < _LINE_MIN_GLYPHS:
                    continue
                ch = hs[ids]
                line_h = float(np.percentile(ch, 75))
                if not (min_h <= line_h <= _LINE_MAX_H):
                    continue
                full = ids[ch >= 0.6 * line_h]
                if full.size < _LINE_MIN_GLYPHS:
                    continue
                fh = hs[full].astype(float)
                bottoms = np.array([objs[i][0].stop for i in full], dtype=float)
                if fh.std() / fh.mean() > _LINE_HEIGHT_CV or bottoms.std() > _LINE_BASELINE * line_h:
                    continue
                # Letter spacing: typeset glyphs sit close together (word spaces are the minority, so the
                # median ignores them); thin anatomy such as sinus or fat septa is spaced more widely.
                # Viewer annotation blocks touch the slice border, where blur and punctuation (": 3.00")
                # widen the measured gaps, so the rule applies only to lines away from the border.
                spans = sorted((objs[i][1].start, objs[i][1].stop) for i in full)
                gaps = [max(0, b0 - a1) for (_, a1), (b0, _) in zip(spans[:-1], spans[1:])]
                ly0, ly1, lx0, lx1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
                near_border = min(lx0, ly0, w - lx1, h - ly1) < _BORDER_ZONE * min(h, w)
                if not near_border and np.median(gaps) > _LINE_MAX_GAP * line_h:
                    continue
                if (sl[1].stop - sl[1].start) < 2.0 * line_h:
                    continue
                # Glyph mask: the line's stroke components, grown 2 px through weaker top-hat response
                # (anti-aliased edges, JPEG halo), then safety-dilated.
                m = np.isin(lab, ids + 1)
                weak = th >= 0.4 * _LINE_TOPHAT
                for _ in range(2):
                    m = ndi.binary_dilation(m, structure=struct8) & weak | m
                m = ndi.binary_dilation(m, structure=struct8, iterations=_SAFETY_DILATE)
                ys, xs = np.nonzero(m)
                pad = max(2, round(line_h / 4))
                box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                       min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))
                if any((m & q.mask).sum() > 0.5 * m.sum() for q in measured):
                    continue  # the same line, found in the other stroke map
                measured.append(StrokeLine(box, polarity, m, int(full.size), line_h))
        lines += measured + _complete_line_grid(measured, th, polarity, g)
    return lines


def _short_word_at_edge(th: np.ndarray, g: np.ndarray, polarity: str, pitch: float, hgt: float,
                        from_left: bool) -> tuple[int, int] | None:
    """Column span of a short overlay word starting at the block edge of a row band, or None.

    For the row just past a confirmed block only. Mixed-case words ("dadc") fail the equal-height
    rule, so heights may vary here (lowercase is ~0.6 of the cap height), but the glyphs must be
    overlay-bright (near the display extreme, which tissue there rarely reaches), sit on one baseline,
    be spaced like letters of one word, and start at the block edge.
    """
    from scipy import ndimage as ndi

    extreme = g >= 1.0 - _STROKE_EXTREME if polarity == "bright" else g <= _STROKE_EXTREME
    struct8 = np.ones((3, 3), dtype=bool)
    cores = ndi.binary_dilation((th >= _LINE_TOPHAT) & extreme, structure=struct8) & (th >= 0.5 * _LINE_TOPHAT)
    if not from_left:
        cores = cores[:, ::-1]
    lab, n = ndi.label(cores, structure=struct8)
    if n < _LINE_MIN_GLYPHS:
        return None
    objs = ndi.find_objects(lab)
    # JPEG splits thin letters ("c") into pieces and merges neighbours ("ad"): pieces down to 0.2 of
    # the pitch count when they sit on the word's baseline, and at least two members must be tall.
    glyphs = sorted((sl[1].start, sl[1].stop, sl[0].stop, sl[0].stop - sl[0].start) for sl in objs
                    if 0.2 * pitch <= sl[0].stop - sl[0].start <= 1.0 * pitch)
    chain: list[tuple[int, int, int, int]] = []
    for x0, x1, bottom, gh in glyphs:
        if not chain:
            if x0 > 2 * hgt:  # the word must start at the block edge
                return None
            chain.append((x0, x1, bottom, gh))
        elif x0 - max(c[1] for c in chain) <= hgt:  # letter / short word spacing
            chain.append((x0, x1, bottom, gh))
        else:
            break
    tall = [c for c in chain if c[3] >= 0.5 * pitch]
    if len(tall) < 2:
        return None
    base = float(np.median([c[2] for c in tall]))
    chain = [c for c in chain if abs(c[2] - base) <= _LINE_BASELINE * pitch]
    if len(chain) < _LINE_MIN_GLYPHS:
        return None
    a, b = chain[0][0], chain[-1][1]
    w = th.shape[1]
    return (a, b) if from_left else (w - b, w - a)


def _complete_line_grid(lines: list[StrokeLine], th: np.ndarray, polarity: str,
                        gray01: np.ndarray | None = None) -> list[StrokeLine]:
    """Fill in text lines that sit between confirmed lines of the same annotation block.

    Viewer overlays print their corner blocks on a fixed line pitch, left- or right-aligned. A line
    drawn over bright anatomy can lose too much contrast to pass the glyph-regularity test on its own
    (``RT: ...`` between ``ST: ...`` and ``Images: ...``). When two confirmed lines of one aligned
    block are a whole number k >= 2 of line pitches apart, the k-1 rows between them are taken as
    text lines if they contain thin strokes starting at the block's edge. Asymmetric-error policy: a
    row inside a confirmed text block is treated as text.
    """
    from scipy import ndimage as ndi

    if len(lines) < 2:
        return []
    h_img, w_img = th.shape
    weak = th >= 0.5 * _LINE_TOPHAT
    struct8 = np.ones((3, 3), dtype=bool)
    out: list[StrokeLine] = []
    done: set[tuple[int, int]] = set()
    # Same-row bridging: two confirmed pieces of one row (e.g. "ST: 3.00" and "18.19 mm" with a
    # low-contrast "mm SL:" between them) take in the thin strokes between them.
    for a in lines:
        for b in lines:
            if a is b or b.box[0] <= a.box[2]:
                continue
            hgt = max(a.height, b.height)
            box_h = max(a.box[3] - a.box[1], b.box[3] - b.box[1])  # robust to blur shrinking measured glyphs
            vov = min(a.box[3], b.box[3]) - max(a.box[1], b.box[1])
            if vov < 0.5 * min(a.box[3] - a.box[1], b.box[3] - b.box[1]) or b.box[0] - a.box[2] > 5 * box_h:
                continue
            y0, y1 = max(a.box[1], b.box[1]), min(a.box[3], b.box[3])
            m = np.zeros(th.shape, dtype=bool)
            m[y0:y1, a.box[2]:b.box[0]] = weak[y0:y1, a.box[2]:b.box[0]]
            if m.sum() < 2 * hgt:
                continue
            m = ndi.binary_dilation(m, structure=struct8, iterations=_SAFETY_DILATE)
            out.append(StrokeLine((a.box[2], y0, b.box[0], y1), polarity, m, 0, hgt, inferred=True))
    # Row extension: a confirmed line whose start or end runs over bright tissue ("27-J" of a date
    # line) loses those glyphs to the regularity test. Strokes continuing along the line's own row,
    # with letter-sized gaps, are taken in, at most _ROW_EXTEND line heights beyond each end.
    for a in lines:
        # Only lines in the annotation zone along the border: extending a stray mid-slice detection
        # would only repaint more tissue.
        if a.inferred or min(a.box[0], a.box[1], w_img - a.box[2], h_img - a.box[3]) >= _BORDER_ZONE * min(h_img, w_img):
            continue
        hgt = a.height
        y0, y1 = a.box[1], a.box[3]
        cols = weak[y0:y1].any(axis=0)
        for side in (-1, 1):
            start = a.box[0] - 1 if side < 0 else a.box[2]
            limit = int(round(_ROW_EXTEND * hgt))
            far, gap, x = None, 0, start
            for _ in range(limit):
                if not (0 <= x < w_img):
                    break
                if cols[x]:
                    far, gap = x, 0
                else:
                    gap += 1
                    if gap > 1.5 * hgt:
                        break
                x += side
            if far is None:
                continue
            ex0, ex1 = (far, a.box[0]) if side < 0 else (a.box[2], far + 1)
            m = np.zeros(th.shape, dtype=bool)
            m[y0:y1, ex0:ex1] = weak[y0:y1, ex0:ex1]
            if m.sum() < hgt:
                continue
            m = ndi.binary_dilation(m, structure=struct8, iterations=_SAFETY_DILATE)
            out.append(StrokeLine((ex0, y0, ex1, y1), polarity, m, 0, hgt, inferred=True))
    for align in ("left", "right"):
        edge = (lambda ln: ln.box[0]) if align == "left" else (lambda ln: ln.box[2])
        remaining = sorted(lines, key=edge)
        while remaining:
            seed = remaining[0]
            tol = max(8.0, 2 * seed.height)
            group = [ln for ln in remaining if abs(edge(ln) - edge(seed)) <= tol]
            remaining = [ln for ln in remaining if ln not in group]
            if len(group) < 2:
                continue
            group.sort(key=lambda ln: (ln.box[1] + ln.box[3]) / 2)
            hgt = float(np.median([ln.height for ln in group]))
            cys = np.array([(ln.box[1] + ln.box[3]) / 2 for ln in group])
            steps = np.diff(cys)
            pitches = steps[(steps >= 1.1 * hgt) & (steps <= 2.6 * hgt)]
            if pitches.size == 0:
                continue
            p = float(np.median(pitches))
            x_edge = int(min(edge(ln) for ln in group)) if align == "left" else int(max(edge(ln) for ln in group))
            for a, b in zip(group[:-1], group[1:]):
                d = (b.box[1] + b.box[3]) / 2 - (a.box[1] + a.box[3]) / 2
                # A filled-in line may not run further from the edge than any confirmed line in this
                # stretch of the block (+4 glyphs): over textured tissue the weak-stroke map never shows a
                # clean end-of-line gap. All lines count here, not only this edge group: a long row is
                # often confirmed in pieces.
                span = [ln for ln in lines if ln.box[3] >= a.box[1] - p and ln.box[1] <= b.box[3] + p]
                far = (max(ln.box[2] for ln in span) - x_edge) if align == "left" else (x_edge - min(ln.box[0] for ln in span))
                reach = far + 4 * hgt
                k = int(round(d / p))
                # At most 4 missing lines: a larger gap separates two blocks (top and bottom corners).
                if k < 2 or k > _GRID_MAX_GAP + 1 or abs(d - k * p) > 0.35 * p:
                    continue
                for j in range(1, k):
                    cy = (a.box[1] + a.box[3]) / 2 + j * d / k
                    # The row spans half a pitch either side of its centre: sized from the pitch, not
                    # the measured glyph height, which blur and JPEG make come out short.
                    y0 = max(0, int(round(cy - 0.5 * d / k)))
                    y1 = min(h_img, int(round(cy + 0.5 * d / k)) + 1)
                    if (y0, x_edge) in done or any(
                        abs((ln.box[1] + ln.box[3]) / 2 - cy) < 0.4 * p and abs(edge(ln) - x_edge) <= tol
                        for ln in lines
                    ):
                        continue
                    band = weak[y0:y1]
                    cols = band.any(axis=0)
                    # Walk away from the aligned edge; the line ends at the first gap wider than ~2.5 glyphs.
                    order = range(max(0, x_edge), w_img) if align == "left" else range(min(w_img, x_edge) - 1, -1, -1)
                    first = last = None
                    gap = 0
                    for x in order:
                        if abs(x - x_edge) > reach:
                            break
                        if cols[x]:
                            first = x if first is None else first
                            last, gap = x, 0
                        elif first is not None:
                            gap += 1
                            if gap > 2.5 * hgt:
                                break
                        elif abs(x - x_edge) > 2 * hgt:  # nothing near the block edge: not a line of this block
                            break
                    if first is None or last is None:
                        continue
                    lx0, lx1 = (first, last + 1) if align == "left" else (last, first + 1)
                    m = np.zeros(th.shape, dtype=bool)
                    m[y0:y1, lx0:lx1] = band[:, lx0:lx1]
                    if m.sum() < 2 * hgt or ndi.label(m, structure=struct8)[1] < 2:
                        continue
                    # One px more than measured lines: a faint line's weakest stroke pixels fall below
                    # the weak-stroke threshold.
                    m = ndi.binary_dilation(m, structure=struct8, iterations=_SAFETY_DILATE + 1)
                    pad = max(2, round(hgt / 4))
                    box = (max(0, lx0 - pad), max(0, y0 - pad), min(w_img, lx1 + pad), min(h_img, y1 + pad))
                    done.add((y0, x_edge))
                    out.append(StrokeLine(box, polarity, m, 0, hgt, inferred=True))
            # Block-end extension: the row just past the inner end of a corner block (below a top
            # block, above a bottom block) can hold a short word the regularity test rejects ("dadc":
            # two full-height glyphs). Pixel evidence alone cannot tell it from tissue at the block's
            # edge, so it is accepted only when OCR reads >= _CROP_MIN_CHARS confident characters.
            if gray01 is None:
                continue
            runs, cur = [], [group[0]]
            for a, b in zip(group[:-1], group[1:]):
                if (b.box[1] + b.box[3]) / 2 - (a.box[1] + a.box[3]) / 2 > (_GRID_MAX_GAP + 1.5) * p:
                    runs.append(cur)
                    cur = []
                cur.append(b)
            runs.append(cur)
            for run in runs:
                top = np.mean([(ln.box[1] + ln.box[3]) / 2 for ln in run]) < h_img / 2
                end = run[-1] if top else run[0]
                reach = max(ln.box[2] - ln.box[0] for ln in run) + 4 * hgt
                cy = (end.box[1] + end.box[3]) / 2
                for _ in range(_BLOCK_END_ROWS):
                    cy += p if top else -p
                    y0, y1 = max(0, int(round(cy - 0.5 * p))), min(h_img, int(round(cy + 0.5 * p)) + 1)
                    if y1 - y0 < 3 or any(abs((ln.box[1] + ln.box[3]) / 2 - cy) < 0.4 * p and abs(edge(ln) - x_edge) <= tol
                                          for ln in lines + out):
                        break
                    if align == "left":
                        lx0, lx1 = max(0, x_edge), min(w_img, int(x_edge + reach))
                    else:
                        lx0, lx1 = max(0, int(x_edge - reach)), min(w_img, x_edge)
                    word = _short_word_at_edge(th[y0:y1, lx0:lx1], gray01[y0:y1, lx0:lx1], polarity, p,
                                               hgt, from_left=(align == "left"))
                    if word is not None:
                        bx0, bx1 = lx0 + word[0], lx0 + word[1]
                    else:
                        band = weak[y0:y1, lx0:lx1]
                        hit = np.nonzero(band.any(axis=0))[0]
                        if hit.size == 0:
                            break
                        bx0, bx1 = lx0 + int(np.min(hit)), lx0 + int(np.max(hit)) + 1
                        from .universal_io import _ocr_available

                        if (not _ocr_available()
                                or read_crop_text(gray01, (bx0, y0, bx1, y1), polarity)[0] < _CROP_MIN_CHARS):
                            break
                    m = np.zeros(th.shape, dtype=bool)
                    m[y0:y1, bx0:bx1] = weak[y0:y1, bx0:bx1]
                    m = ndi.binary_dilation(m, structure=struct8, iterations=_SAFETY_DILATE + 1)
                    out.append(StrokeLine((bx0, y0, bx1, y1), polarity, m, 0, hgt, inferred=True))
    return out


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
        # anatomical structure. Ties go to the minority class. This only sets the ORDER: the other
        # polarity is still tried if the preferred one yields no plausible glyphs.
        def thin_fraction(cls):
            r = int(np.ceil(_stroke_width(cls) / 2)) + 1
            return float((cls & ~ndi.binary_opening(cls, structure=_disk(r))).sum() / max(1, cls.sum()))

        tb_, td_ = thin_fraction(bright_c), thin_fraction(dark_c)
        if abs(tb_ - td_) > 0.1:
            candidates = ["bright", "dark"] if tb_ > td_ else ["dark", "bright"]
        else:
            candidates = ["bright", "dark"] if fb <= fd else ["dark", "bright"]
    elif fb >= _MIN_EXTREME_FRAC:
        candidates = ["bright"]
    elif fd >= _MIN_EXTREME_FRAC:
        candidates = ["dark"]
    else:
        candidates = ["bright", "dark"]  # unsaturated text: ordered by thin-structure energy below

    def stroke_response(polarity):
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
            return None
        seeds = resp >= _SEED_CONTRAST * contrast
        if saturated:  # seeds must be extreme-valued (or very strongly thin) pixels
            seeds &= cls | (resp >= 0.6 * contrast)
        return float(resp[seeds].sum()), resp, contrast, seeds, sw

    responses = [(pol, stroke_response(pol)) for pol in candidates]
    responses = [(pol, r) for pol, r in responses if r is not None]
    if not fb >= _MIN_EXTREME_FRAC and not fd >= _MIN_EXTREME_FRAC:
        responses.sort(key=lambda pr: -pr[1][0])
    if not responses:
        return None, {"polarity": "n/a", "stroke_px": 0.0, "reason": "no thin-stroke contrast inside box"}

    struct8 = np.ones((3, 3), dtype=bool)
    failure = None
    for polarity, (_, resp, contrast, seeds, sw) in responses:
        grow_ok = resp >= _GROW_CONTRAST * contrast
        # Growth may reach ~half a stroke plus 2 px: enough to take in anti-aliasing and the JPEG
        # ringing halo around glyphs (8x8-block codecs smear 2-4 px), which would otherwise survive as a
        # faint letter outline. grow_ok (a thin-structure response) keeps growth off flat background.
        steps = max(2, int(np.ceil(sw / 2)) + 2)
        mask = seeds & grow_ok
        for _ in range(steps):  # bounded region growing (geodesic dilation inside grow_ok)
            grown = ndi.binary_dilation(mask, structure=struct8) & grow_ok
            if np.array_equal(grown, mask):
                break
            mask = grown
        mask, n_rejected = _keep_glyph_components(crop, mask & in_box, polarity)
        if safety_dilate > 0:
            mask = ndi.binary_dilation(mask, structure=struct8, iterations=safety_dilate)
        mask &= in_box

        n_mask = int(mask.sum())
        info = {"polarity": polarity, "stroke_px": round(sw, 2), "rejected_components": n_rejected}
        if n_mask == 0:
            failure = failure or (None, {**info, "reason": "empty glyph mask"})
            continue
        if n_mask > _MAX_MASK_FRAC * n_box:
            failure = (None, {**info, "reason": f"glyph mask covers {n_mask / n_box:.0%} of box (implausible)"})
            continue
        full = np.zeros(gray01.shape, dtype=bool)
        full[cy0:cy1, cx0:cx1] = mask
        return full, {**info, "reason": "segmented"}
    if failure is None:  # unreachable: every polarity either returns or records a failure
        return None, {"polarity": "n/a", "stroke_px": 0.0, "reason": "no plausible glyph mask"}
    return failure


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


def _fill_flat_on_background(img01: np.ndarray, mask: np.ndarray, out01: np.ndarray) -> np.ndarray:
    """Replace the inpainted fill with the local background level where text sits on background.

    Corner annotations are usually drawn on the black area around the body. Diffusion inpainting
    fills a text line from its whole boundary, so a line that runs close to bright tissue pulls grey
    streaks into the black corner. When the pixels around a word are background (90th percentile
    below ``_BG_DARK``), the word's pixels are set to the median of those surroundings instead.
    """
    from scipy import ndimage as ndi

    words = ndi.binary_dilation(mask, structure=np.ones((3, 5), dtype=bool))  # letters of a word together
    lab, n = ndi.label(words)
    if n == 0:
        return out01
    out01 = out01.copy()
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        y0, y1 = max(0, sl[0].start - 3), min(mask.shape[0], sl[0].stop + 3)
        x0, x1 = max(0, sl[1].start - 3), min(mask.shape[1], sl[1].stop + 3)
        word = (lab[y0:y1, x0:x1] == k) & mask[y0:y1, x0:x1]
        ring = ndi.binary_dilation(lab[y0:y1, x0:x1] == k, iterations=3) & ~mask[y0:y1, x0:x1]
        if not ring.any():
            continue
        vals = img01[y0:y1, x0:x1][ring]
        if np.percentile(vals, 90) < _BG_DARK:
            out01[y0:y1, x0:x1][word] = np.median(vals)
    return out01


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
        lo, hi = float(np.min(gray)), float(np.max(gray))
        if hi > lo:
            g01 = ((gray - lo) / (hi - lo)).astype(np.float32)
            boxes = boxes + detect_overlay_text_boxes(g01, existing=boxes)
    # v1 can report the same box from two passes (with different confidences); process it once.
    best: dict[tuple[int, int, int, int], int] = {}
    for x0, y0, x1, y1, conf in boxes:
        best[(x0, y0, x1, y1)] = max(conf, best.get((x0, y0, x1, y1), conf))
    return [(*b, c) for b, c in best.items()]


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


def _glyph_zone_bbox(glyph: np.ndarray, pad: int = 3) -> tuple[int, int, int, int] | None:
    """Tight (x0, y0, x1, y1) around a box-local glyph mask, padded; None if the mask is empty."""
    ys, xs = np.nonzero(glyph)
    if ys.size == 0:
        return None
    h, w = glyph.shape
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(w, int(xs.max()) + 1 + pad), min(h, int(ys.max()) + 1 + pad))


def _boxes_still_text(
    restored01: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    glyph_masks: list[np.ndarray],
    upscale_factors,
    check_inverted,
) -> list[bool]:
    """Two independent checks per restored region; either one finding text means "still text".

    Both look only where pixels were replaced (the glyph mask and a few pixels around it), not the
    whole detector box: a v1 box often also covers anatomy that Tesseract "reads" as text both
    before and after restoration, which says nothing about whether the removed text is gone.

    1. v1's detector on the whole restored slice (run on the full slice, because its tuned size
       constants assume the slice's own resolution normalization); a detection counts if it overlaps
       the glyph zone and is not much larger than the region (slice-sized junk rows are ignored).
    2. A native-resolution OCR read of the glyph zone (both polarities); at least _CROP_MIN_CHARS
       characters read with confidence >= _VERIFY_MIN_CONF means legible text survived.
    The pixel-level silhouette check (:func:`ghost_ratio`) is applied by the caller.
    """
    from scipy import ndimage as ndi

    from .universal_io import detect_text_boxes

    found = detect_text_boxes(restored01, upscale_factors=upscale_factors, check_inverted=check_inverted, padding=0)
    flags = []
    for (x0, y0, x1, y1), glyph in zip(boxes, glyph_masks):
        zb = _glyph_zone_bbox(glyph)
        if zb is None:
            flags.append(False)
            continue
        zone = np.zeros(restored01.shape, dtype=bool)
        zone[y0:y1, x0:x1] = ndi.binary_dilation(glyph, iterations=3)
        area = max(1, (x1 - x0) * (y1 - y0))
        hit = False
        for fx0, fy0, fx1, fy1, _ in found:
            f_area = max(1, (fx1 - fx0) * (fy1 - fy0))
            if f_area > 2 * area:
                continue
            if zone[fy0:fy1, fx0:fx1].sum() >= max(4, 0.05 * f_area):
                hit = True
                break
        if not hit:
            zx0, zy0, zx1, zy1 = zb
            hit = read_crop_text(restored01, (x0 + zx0, y0 + zy0, x0 + zx1, y0 + zy1),
                                 min_conf=_VERIFY_MIN_CONF)[0] >= _CROP_MIN_CHARS
        flags.append(hit)
    return flags


def changed_region_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """``(x0, y0, x1, y1)`` rectangles around changed areas of a 2D inpaint mask, merged per text line.

    For displaying *where* pixels changed. Never draw the mask itself (outline or fill): it is the
    exact glyph shape, so it would re-display the removed PHI.
    """
    from scipy import ndimage as ndi

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    h, w = mask.shape
    gap = max(3, w // 60)  # joins the letters and words of one line
    merged = ndi.binary_dilation(mask, structure=np.ones((max(1, gap // 2), gap), dtype=bool))
    lab, _ = ndi.label(merged)
    return [(sl[1].start, sl[0].start, min(w, sl[1].stop), min(h, sl[0].stop)) for sl in ndi.find_objects(lab)]


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
        lo, hi = float(np.min(gray)), float(np.max(gray))
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
        fill_rects: list[tuple[int, int, int, int]] = []  # fallback rectangles (filled flat, see below)
        pending: list[tuple[tuple[int, int, int, int], int, dict, int]] = []

        for x0, y0, x1, y1, conf in boxes:
            box = (x0, y0, x1, y1)
            n_box = (x1 - x0) * (y1 - y0)
            if method == "blackout":
                black_mask[y0:y1, x0:x1] = True
                regions.append(RegionResult(i, box, conf, "blackout", "method=blackout", "n/a", n_box, n_box))
                continue
            for kind, rbox, payload in _resolve_box(gray01, box):
                bx0, by0, bx1, by1 = rbox
                nb = (bx1 - bx0) * (by1 - by0)
                if kind == "glyphs":
                    mb, ib = payload
                    slice_mask |= mb
                    pending.append((rbox, conf, ib, nb))
                elif kind == "blackout":
                    black_mask[by0:by1, bx0:bx1] = True
                    fill_rects.append(rbox)
                    regions.append(RegionResult(i, rbox, conf, "fallback_blackout", payload["reason"],
                                                payload.get("polarity", "n/a"), nb, nb))
                else:  # "skip"
                    regions.append(RegionResult(i, rbox, conf, "skipped_not_text", payload["reason"],
                                                payload.get("polarity", "n/a"), 0, nb))

        # Text lines the OCR detectors and saturation-based segmentation missed: text dimmer than the
        # brightest anatomy, text drawn over tissue, and lines dropped when a mixed box was split.
        if method != "blackout":
            covered = slice_mask | black_mask
            for ln in detect_stroke_text_lines(gray01):
                if (ln.mask & ~covered).sum() < 0.2 * ln.mask.sum():
                    continue  # already handled by the detector boxes
                slice_mask |= ln.mask
                covered |= ln.mask
                bx0, by0, bx1, by1 = ln.box
                why = ("stroke-line detector: row inside a confirmed text block (line grid)" if ln.inferred
                       else f"stroke-line detector: {ln.n_glyphs} aligned glyphs of equal height")
                pending.append((ln.box, 0, {"polarity": ln.polarity, "reason": why}, (bx1 - bx0) * (by1 - by0)))

        restored01 = (_fill_flat_on_background(gray01, slice_mask, inpaint_mask(gray01, slice_mask, method))
                      if slice_mask.any() else gray01)

        glyphs = [slice_mask[b[1]:b[3], b[0]:b[2]] for b, *_ in pending]
        still = (_boxes_still_text(restored01, [p[0] for p in pending], glyphs, factors, check_inverted)
                 if verify and pending else [False] * len(pending))
        for (box, conf, info, n_box), leaked, box_mask in zip(pending, still, glyphs):
            x0, y0, x1, y1 = box
            ghost = ghost_ratio(restored01[y0:y1, x0:x1], box_mask) if verify else 0.0
            if leaked or ghost > _GHOST_RATIO:
                # Black out the rectangle around the replaced glyphs (the text line itself), not the
                # whole detector box, which may extend over anatomy. A rectangle -- never the glyph
                # shape -- so no letter silhouette is left behind. Exception: an OCR-detector box in
                # the annotation zone along the border is covered whole, because rows of it whose
                # glyphs were not segmented (a date line over bright tissue) would otherwise survive.
                zx0, zy0, zx1, zy1 = _glyph_zone_bbox(box_mask) or (0, 0, x1 - x0, y1 - y0)
                hh, ww = spatial
                if conf > 0 and min(x0, y0, ww - x1, hh - y1) < _BORDER_ZONE * min(hh, ww):
                    zx0, zy0, zx1, zy1 = 0, 0, x1 - x0, y1 - y0
                black_mask[y0 + zy0:y0 + zy1, x0 + zx0:x0 + zx1] = True
                fill_rects.append((x0 + zx0, y0 + zy0, x0 + zx1, y0 + zy1))
                n_black_px = (zx1 - zx0) * (zy1 - zy0)
                why = "text still detected after restoration" if leaked else f"glyph silhouette visible (ghost ratio {ghost:.2f})"
                regions.append(RegionResult(i, box, conf, "fallback_blackout", why + "; blacked out the text line only",
                                            info["polarity"], n_black_px, n_box))
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
                    rest = _fill_flat_on_background(ch01, synth, inpaint_mask(ch01, synth, method))
                    ch[synth] = rest[synth] * (chi - clo) + clo
            else:
                frame[synth] = restored01[synth] * (hi - lo) + lo
        if black_mask.any():
            frame[black_mask] = 0
            # Restoration methods: a fallback rectangle is re-synthesised from the pixels AROUND it
            # (biharmonic interpolation of the whole rectangle: smooth, no fast-marching streaks)
            # instead of set to 0. Nothing inside the rectangle is used, so nothing of the text
            # survives; but a black bar on the grey background of an ADC map or a screenshot -- or a
            # flat bar across tissue -- draws the eye.
            rect = np.zeros(spatial, dtype=bool)
            for rx0, ry0, rx1, ry1 in fill_rects:
                rect[ry0:ry1, rx0:rx1] = True
            if rect.any():
                chans = [frame[..., c] for c in range(frame.shape[-1])] if is_color else [frame]
                for ch in chans:
                    clo, chi = float(ch.min()), float(ch.max())
                    if chi <= clo:
                        continue
                    ch01 = ((ch - clo) / (chi - clo)).astype(np.float32)
                    rest = _inpaint_biharmonic(ch01, rect)
                    ch[rect] = rest[rect] * (chi - clo) + clo
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
