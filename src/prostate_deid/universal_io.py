"""
Universal format bridge: PNG / DICOM / NIfTI in, fully de-identified DICOM
out, with an optional second hop from that DICOM to ``.nii.gz``.

Header/tag scrubbing (below) is the "dumbest possible" path to a safe
DICOM -- delete every non-technical field, full stop. Burned-in pixel text
(scanner corner annotations, fusion-biopsy overlays baked directly into the
image) is a different, probabilistic problem the header scrub cannot touch:
:func:`redact_burned_in_text` runs Tesseract OCR over the pixel data itself
and blacks out every detected text region. The library's own
``BurnedInTextRedact`` transform documents this same need but ships as an
unimplemented stub (see :mod:`prostate_deid.burned_in_text`); this function
is a real, working implementation using ``pytesseract``.

  * PNG / NIfTI input carries no patient header to begin with, so the only
    thing to do is build a brand-new minimal DICOM dataset around the pixel
    data (new random UIDs, only the technical tags required for a valid
    file) rather than ever populating identifying fields in the first
    place.
  * Real DICOM (.dcm) input goes through :func:`full_wipe_dicom`, which
    keeps only pixel data plus the technical tags a DICOM file needs to
    remain valid, and discards every other element -- including private
    tags and anything outside the library's named HIPAA Safe Harbor /
    PS3.15 profiles in :mod:`prostate_deid.dicom_scrub`. This is a stricter
    (not a looser) policy than ``default_pipeline``.
  * Every input path also runs burned-in pixel text through
    :func:`redact_burned_in_text` before the DICOM is built/saved, so the
    corner-annotation case (name/DOB/institution rendered into the image
    itself) is covered regardless of source format.

Every entry point here returns a small, flat list of plain-string audit
notes (not :class:`~prostate_deid.core.AuditEntry`, since there is no
per-tag "before" value worth logging when the policy is simply "delete
everything non-technical").
"""

from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "load_any",
    "full_wipe_dicom",
    "array_to_dicom",
    "dicom_to_nifti_bytes",
    "redact_burned_in_text",
    "SUPPORTED_INPUT_EXTS",
]

SUPPORTED_INPUT_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii", ".nii.gz")

# Technical tags kept by full_wipe_dicom -- everything needed to make the
# pixel data self-describing and the file DICOM-valid, nothing that can
# carry patient/institution identity.
_TECHNICAL_KEEP = {
    "SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID",
    "Modality", "Rows", "Columns", "BitsAllocated", "BitsStored", "HighBit",
    "PixelRepresentation", "SamplesPerPixel", "PhotometricInterpretation",
    "PlanarConfiguration", "NumberOfFrames", "TransferSyntaxUID",
    "RescaleIntercept", "RescaleSlope", "PixelSpacing", "SliceThickness",
    "ImageOrientationPatient", "ImagePositionPatient",
}


def _new_uid() -> str:
    """Deterministic-format, non-identifying UID under the 2.25 (UUID) root."""
    return f"2.25.{uuid.uuid4().int}"[:64]


# ---------------------------------------------------------------------------
# Loading heterogeneous input -> a plain float32 pixel array
# ---------------------------------------------------------------------------

def _ext_of(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower).suffix


def load_any(path_or_bytes: str | os.PathLike | bytes, filename: str | None = None) -> dict[str, Any]:
    """Load a PNG/JPEG/BMP/TIFF, DICOM, or NIfTI file into a common shape.

    Returns a dict with:
      - ``pixels``: float32 ndarray, ``(slices, H, W)`` (single-slice 2D
        images become ``(1, H, W)``).
      - ``kind``: one of ``"image"``, ``"dicom"``, ``"nifti"``.
      - ``source_dataset``: the ``pydicom.Dataset`` when ``kind == "dicom"``,
        else ``None``.
      - ``affine``: the NIfTI affine (4x4) when ``kind == "nifti"``, else
        ``None`` -- kept only for a possible future NIfTI-preserving path;
        the DICOM conversion below does not use it, since a fresh DICOM
        cannot represent it and dropping it is data, not PHI.
    """
    name = filename or (path_or_bytes if isinstance(path_or_bytes, (str, os.PathLike)) else "upload")
    ext = _ext_of(str(name))

    if isinstance(path_or_bytes, (bytes, bytearray)):
        buf: Any = io.BytesIO(path_or_bytes)
    else:
        buf = path_or_bytes

    if ext == ".dcm":
        import pydicom

        ds = pydicom.dcmread(buf)
        pixels = np.asarray(ds.pixel_array).astype(np.float32)
        if pixels.ndim == 2:
            pixels = pixels[np.newaxis, :, :]
        elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            # a color frame (H, W, C) rather than a (N, H, W) stack
            pixels = pixels[np.newaxis, ...]
        return {"pixels": pixels, "kind": "dicom", "source_dataset": ds, "affine": None}

    if ext in (".nii", ".nii.gz"):
        import gzip

        import nibabel as nib

        if isinstance(buf, io.BytesIO):
            raw = buf.getvalue()
            if ext == ".nii.gz" or raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            fh = nib.FileHolder(fileobj=io.BytesIO(raw))
            img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
        else:
            img = nib.load(str(buf))
        data = np.asanyarray(img.dataobj).astype(np.float32)
        # NIfTI convention (X, Y, Z[, ...]) -> slice-stack (Z, Y, X) for display/DICOM
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim >= 3:
            data = np.moveaxis(data, 2, 0)
            if data.ndim > 3:
                data = data.reshape(data.shape[0], data.shape[1], data.shape[2], -1)[..., 0]
        return {"pixels": data, "kind": "nifti", "source_dataset": None, "affine": img.affine}

    # Plain raster image (PNG/JPEG/BMP/TIFF/...)
    from PIL import Image

    im = Image.open(buf)
    im.load()
    arr = np.asarray(im).astype(np.float32)
    if arr.ndim == 2:
        pixels = arr[np.newaxis, :, :]
    else:
        pixels = arr[np.newaxis, ...]  # (1, H, W, C)
    return {"pixels": pixels, "kind": "image", "source_dataset": None, "affine": None}


# ---------------------------------------------------------------------------
# Burned-in pixel text detection + redaction (OCR)
# ---------------------------------------------------------------------------

def _find_tesseract_cmd() -> str | None:
    """Locate the tesseract binary on Windows/PATH, since a fresh pytesseract
    install has no way to find it unless it's already on PATH."""
    import shutil

    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _ocr_available() -> bool:
    try:
        import pytesseract
    except ImportError:
        return False
    cmd = _find_tesseract_cmd()
    if cmd is None:
        return False
    pytesseract.pytesseract.tesseract_cmd = cmd
    return True


_UPSCALE_FACTORS = (3, 4, 5)  # tried in turn; Tesseract's line/word segmentation on tiny scanner
# fonts is sensitive to the exact upscale factor in ways that are not predictable in advance --
# testing found a single fixed factor unreliably missed a real text line that a different factor
# caught on the SAME image, so detection unions results across several factors rather than
# picking one. Cost is roughly linear in the number of factors; kept small since this runs per
# slice, per (normal + inverted) pass.

_REFERENCE_WIDTH = 300  # every pixel-size constant below (_MAX_WORD_HEIGHT, _ROW_EXTEND_TILE, ...) was
# tuned against real scanner screenshots downscaled/cropped to roughly this width. A real upload can be
# any resolution (a 2000px-wide DICOM export has corner-annotation text many times larger in absolute
# pixels than the same text on a 300px screenshot crop), so detection always runs on a COPY of each
# slice normalized to this reference width -- see _normalize_for_ocr / _denormalize_box -- and every
# constant below is expressed in that normalized space, never in the caller's native resolution.


_LINE_PAD_FRAC = 0.35  # extra vertical/horizontal margin as a fraction of line height
_ROW_GAP_FRAC = 1.2  # merge two words into one row-band if their vertical gap < this * min(their heights)
_ROW_HGAP_FRAC = 4.0  # ...and horizontal gap < this * min(their heights) -- caps how far a merge can reach sideways
_MIN_RANGE = 135  # a real overlay-text glyph box spans at least this much 0-255 intensity range (min-to-max)
_ROW_EXTEND_TILE = 6  # column-width (px) stepped outward from a confirmed text row when extending it
_ROW_EXTEND_GAP_TILES = 2  # stop extending after this many consecutive non-text tiles (tolerates inter-word gaps)
_MAX_WORD_HEIGHT = 18  # reject OCR word boxes taller than this (px, native resolution) as not real overlay-text glyphs
_MIN_TRANSITIONS_PER_PX = 0.19  # a real glyph box has repeated light/dark strokes; a single anatomy edge does not


_NORMALIZE_SKIP_BAND = 0.20  # skip the resize entirely when width is already within +/-20% of reference


def _normalize_for_ocr(gray_u8: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize a grayscale slice to :data:`_REFERENCE_WIDTH` so every tuned constant applies unchanged.

    Returns the resized array and the scale FACTOR applied (normalized_px =
    original_px * factor), so callers can map detected box coordinates back
    to the original resolution with :func:`_denormalize_box`. Uses NEAREST
    resampling, not LANCZOS/bilinear: this downscale is a coordinate-space
    normalization for detection, not a quality-sensitive resample, and
    NEAREST avoids inventing new intensity values that could shift a real
    text glyph's min/max range right at the :data:`_MIN_RANGE` threshold.

    Skips the resize (returns the array unchanged, factor 1.0) when the
    width is already within :data:`_NORMALIZE_SKIP_BAND` of the reference:
    testing found that even a near-identity resize (~1-2% factor) can shift
    pixel alignment enough to flip Tesseract's line segmentation on some
    real images -- a real miss on an image that would have detected fine
    left alone. The skip band accepts that the tuned constants are only
    approximately resolution-independent near their tuning point, in
    exchange for not introducing risk where there is no real resolution
    mismatch to correct.
    """
    h, w = gray_u8.shape
    factor = _REFERENCE_WIDTH / w
    if abs(factor - 1.0) <= _NORMALIZE_SKIP_BAND:
        return gray_u8, 1.0

    from PIL import Image

    new_h = max(1, round(h * factor))
    resized = np.array(
        Image.fromarray(gray_u8).resize((_REFERENCE_WIDTH, new_h), Image.NEAREST)
    )
    return resized, factor


def _denormalize_box(
    box: tuple[int, int, int, int, int], factor: float
) -> tuple[int, int, int, int, int]:
    """Map a box detected in normalized-resolution space back to the original slice's coordinates."""
    x0, y0, x1, y1, conf = box
    if factor == 1.0:
        return box
    inv = 1.0 / factor
    return (
        int(x0 * inv), int(y0 * inv), int(x1 * inv) + 1, int(y1 * inv) + 1, conf,
    )


def _looks_like_rendered_text(gray_u8: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> bool:
    """Reject a candidate word box whose pixels look like MRI tissue, not rendered overlay text.

    Two independent pixel-statistics signals must both hold, since either
    alone was found (empirically, against real scanner screenshots) to
    admit anatomy false positives the other catches:

    1. Intensity range: scanner corner annotations are burned in as flat,
       near-black background with near-white (or vice-versa) glyphs -- even
       a small, tightly-cropped word box spans nearly the full 0-255
       intensity range. A single strong tissue-boundary edge can ALSO span
       most of that range (e.g. bone against an air/fluid void), so range
       alone is not sufficient.
    2. Stroke density: real text glyphs are repeated thin strokes, so a
       horizontal scan crosses back and forth across the box's own
       mid-intensity many times per pixel of width. A single anatomy edge
       is one dominant transition -- low crossing density even when its
       intensity range is text-like.

    Neither signal alone reliably separates real burned-in text from MRI
    anatomy on real (not synthetic) scanner screenshots -- this was found
    the hard way across several real test images, each breaking a
    single-signal version of this check in a different place (see git
    history / PR discussion for the iterations). OCR confidence does not
    help here either: Tesseract happily "reads" tissue-boundary texture as
    low-confidence garbage characters with no reliable confidence gap
    versus real text. Padding is added by the caller only after a box
    passes this check, since padding in before measuring would pull in
    enough true background to make anatomy boxes falsely pass too.
    """
    y0c, y1c = max(0, y0), min(gray_u8.shape[0], y1)
    x0c, x1c = max(0, x0), min(gray_u8.shape[1], x1)
    if y1c <= y0c or x1c <= x0c:
        return False
    patch = gray_u8[y0c:y1c, x0c:x1c]
    if patch.size == 0:
        return False

    lo, hi = int(patch.min()), int(patch.max())
    if (hi - lo) < _MIN_RANGE:
        return False

    mid = (lo + hi) / 2
    binary = patch > mid
    transitions = np.abs(np.diff(binary.astype(np.int8), axis=1)).sum()
    return bool((transitions / patch.size) >= _MIN_TRANSITIONS_PER_PX)


def _cluster_rows(words: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    """Merge word boxes into full-width row bands by vertical (y) overlap/proximity.

    Tesseract's own line/block/paragraph grouping (``line_num`` in
    ``image_to_data``) is unreliable on small, sparse scanner-overlay text:
    on real corner annotations it sometimes splits one visual line (e.g.
    ``"RT: 640.6420226025 ET: 10:"``) into several unconnected "lines" that
    only cover the first word, leaving the rest of the line unredacted. This
    reimplements line grouping from scratch, directly from each word's own
    ``(y0, y1)``: any two words whose vertical spans overlap, or are
    separated by less than ``_ROW_GAP_FRAC`` times the shorter word's
    height, AND whose horizontal gap is less than ``_ROW_HGAP_FRAC`` times
    the shorter word's height, are merged into the same row band spanning
    the full horizontal extent of every word placed in it -- independent of
    whatever block/line id Tesseract assigned them. The horizontal-gap cap
    matters as much as the vertical one: without it, scattered single-glyph
    OCR misreads inside anatomy texture (individual punctuation marks
    Tesseract hallucinates from tissue boundaries) chain transitively across
    the whole image width into one giant false row, which is worse than any
    individual false positive because it blacks out the anatomy.
    """
    if not words:
        return []
    words = sorted(words, key=lambda w: w[1])
    rows: list[dict] = []
    for x0, y0, x1, y1, conf in words:
        h = y1 - y0
        placed = False
        for row in rows:
            vgap = max(y0 - row["y1"], row["y0"] - y1, 0)
            voverlap = min(y1, row["y1"]) - max(y0, row["y0"])
            hgap = max(x0 - row["x1"], row["x0"] - x1, 0)
            min_h = min(h, row["y1"] - row["y0"])
            if (voverlap > 0 or vgap < _ROW_GAP_FRAC * min_h) and hgap < _ROW_HGAP_FRAC * min_h:
                row["x0"] = min(row["x0"], x0)
                row["y0"] = min(row["y0"], y0)
                row["x1"] = max(row["x1"], x1)
                row["y1"] = max(row["y1"], y1)
                row["conf"] = min(row["conf"], conf)
                row["n"] += 1
                placed = True
                break
        if not placed:
            rows.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "conf": conf, "n": 1})
    return [(r["x0"], r["y0"], r["x1"], r["y1"], r["conf"]) for r in rows]


def _extend_row_horizontally(
    gray_u8: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> tuple[int, int]:
    """Grow a CONFIRMED text row's box left/right to cover text OCR failed to transcribe as a word.

    Real corner-annotation lines are sometimes only partially readable by
    Tesseract (tiny/degraded fonts, compression artifacts): a value like
    ``"44.87 mm"`` at the end of a line can fail to produce any OCR word at
    all even though the pixels are unambiguously text by the same
    min-to-max range signature :func:`_looks_like_rendered_text` uses. This
    only ever extends a row that OCR has already confirmed as text (never
    starts a new region on its own), by stepping outward one
    ``_ROW_EXTEND_TILE``-wide column at a time and checking that column's
    strip for the text signature, stopping after
    ``_ROW_EXTEND_GAP_TILES`` consecutive columns without it (so a run of
    background between two words on the same line doesn't end the scan
    early, but the gap between one text line and unrelated anatomy does).
    """
    h = y1 - y0
    if h <= 0:
        return x0, x1

    def strip_is_text(cx0: int, cx1: int) -> bool:
        cx0c, cx1c = max(0, cx0), min(gray_u8.shape[1], cx1)
        if cx1c <= cx0c:
            return False
        patch = gray_u8[max(0, y0):min(gray_u8.shape[0], y1), cx0c:cx1c]
        if patch.size == 0:
            return False
        return (int(patch.max()) - int(patch.min())) >= _MIN_RANGE

    new_x1 = x1
    miss_run = 0
    cx = x1
    while cx < gray_u8.shape[1] and miss_run < _ROW_EXTEND_GAP_TILES:
        if strip_is_text(cx, cx + _ROW_EXTEND_TILE):
            new_x1 = cx + _ROW_EXTEND_TILE
            miss_run = 0
        else:
            miss_run += 1
        cx += _ROW_EXTEND_TILE

    new_x0 = x0
    miss_run = 0
    cx = x0
    while cx > 0 and miss_run < _ROW_EXTEND_GAP_TILES:
        if strip_is_text(cx - _ROW_EXTEND_TILE, cx):
            new_x0 = cx - _ROW_EXTEND_TILE
            miss_run = 0
        else:
            miss_run += 1
        cx -= _ROW_EXTEND_TILE

    return max(0, new_x0), min(gray_u8.shape[1], new_x1)


def _ocr_words_at_scale(
    pil_im: "Image.Image", gray_u8: np.ndarray, scale: int
) -> list[tuple[int, int, int, int, int]]:
    """Run Tesseract once at one upscale factor; return surviving word boxes in ORIGINAL pixel space.

    ``--psm 6`` (assume a single uniform block of text) finds scanner
    corner-annotation lines far more reliably than the sparse-text mode
    (``--psm 11``): on real scanner screenshots, PSM 11 fragments each line
    into near-random garbage tokens scattered across the whole frame
    (including inside anatomy), which both misses real text lines and
    redacts the wrong pixels.

    Each candidate word is checked against ``gray_u8`` (the ORIGINAL,
    non-upscaled grayscale slice this OCR pass was run on) by
    :func:`_looks_like_rendered_text`, which rejects anatomy texture that
    Tesseract otherwise "reads" as garbage characters -- this is the
    primary anatomy-vs-text discriminator, not OCR confidence.
    """
    import pytesseract
    from PIL import Image

    big = pil_im.resize((pil_im.width * scale, pil_im.height * scale), Image.LANCZOS)
    data = pytesseract.image_to_data(big, output_type=pytesseract.Output.DICT, config="--psm 6")

    words = []
    for j, text in enumerate(data["text"]):
        if not text.strip():
            continue
        try:
            conf = int(float(data["conf"][j]))
        except (ValueError, TypeError):
            conf = -1
        x, y, w, h = data["left"][j], data["top"][j], data["width"][j], data["height"][j]
        x0, y0, x1, y1 = x // scale, y // scale, (x + w) // scale, (y + h) // scale
        if (y1 - y0) > _MAX_WORD_HEIGHT:
            # Tesseract sometimes fuses a real text glyph with adjacent anatomy into one
            # abnormally tall "word" box (e.g. a circular-FOV edge sitting directly under a
            # corner-annotation line) -- an anomalously tall box corrupts row-clustering's
            # gap tolerance (see _cluster_rows) far more than skipping it costs in coverage,
            # since real annotation glyphs are consistently short.
            continue
        if not _looks_like_rendered_text(gray_u8, x0, y0, x1, y1):
            continue
        words.append((x0, y0, x1, y1, conf))
    return words


def _ocr_boxes(
    pil_im: "Image.Image",
    gray_u8: np.ndarray,
    upscale_factors: tuple[int, ...] = _UPSCALE_FACTORS,
) -> list[tuple[int, int, int, int, int]]:
    """Run Tesseract across several upscale factors; return merged row-band (x0,y0,x1,y1,conf) boxes.

    Word boxes from every factor in ``upscale_factors`` (:data:`_UPSCALE_FACTORS`
    by default) are pooled before clustering (not clustered separately per
    factor and unioned after), since a single scale was found to unreliably
    drop a real text line that another scale on the SAME image caught -- see
    :data:`_UPSCALE_FACTORS`'s comment. Pooled words are merged into
    full-width row bands by :func:`_cluster_rows` (not Tesseract's own line
    grouping, which is unreliable on this kind of sparse small text -- see
    that function's docstring), padded by a fraction of each row's own
    height, so gaps between/around words on the same visual line can't
    leave legible PHI fragments. Confidence is the minimum word confidence
    within the row, so one badly-read word doesn't hide the row's low
    reliability.

    Passing fewer ``upscale_factors`` (e.g. a single factor instead of all
    three) trades detection recall for speed -- each factor is a full
    Tesseract pass, so this is the main lever for running on CPU-constrained
    or mobile hardware where the default 3-factors x 2-polarities-per-slice
    cost (see :func:`redact_burned_in_text`) is too slow.
    """
    words: list[tuple[int, int, int, int, int]] = []
    for scale in upscale_factors:
        words.extend(_ocr_words_at_scale(pil_im, gray_u8, scale))

    boxes = []
    for x0, y0, x1, y1, conf in _cluster_rows(words):
        x0, x1 = _extend_row_horizontally(gray_u8, x0, y0, x1, y1)
        h = y1 - y0
        pad = max(int(h * _LINE_PAD_FRAC), 1)
        x0p, y0p = x0 - pad, y0 - pad
        x1p, y1p = x1 + pad, y1 + pad
        boxes.append((x0p, y0p, x1p, y1p, conf))
    return boxes


def redact_burned_in_text(
    pixels: np.ndarray,
    padding: int = 4,
    flag_confidence: int = 40,
    upscale_factors: tuple[int, ...] | None = None,
    check_inverted: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Black out every OCR-detected text region in a ``(slices, H, W[, C])`` array.

    Any text rendered into the pixel data of a medical image (scanner
    corner annotations -- patient name/DOB/institution/date -- or
    fusion-biopsy overlay labels) is treated as PHI and redacted
    unconditionally, regardless of OCR confidence: this function does not
    attempt to classify *which* detected text is identifying, and does not
    use a confidence threshold to decide whether to redact, matching the
    library's asymmetric-error policy (a missed detection is a privacy
    incident; an over-redaction is only a data-quality loss -- so
    ``flag_confidence`` only controls the audit note, never the redaction
    itself). Detection runs twice per slice by default, on the plain
    grayscale and on its contrast-inverted version, since scanner overlay
    text is sometimes dark-on-light rather than light-on-dark; the image is
    also upscaled before OCR because small corner-annotation glyphs (often
    <10px tall) are below Tesseract's reliable detection size otherwise.
    Every detected box is padded by ``padding`` pixels on each side to
    fully cover anti-aliased glyph edges.

    Parameters
    ----------
    upscale_factors:
        Which upscale factors to run Tesseract at (default
        :data:`_UPSCALE_FACTORS`, three passes). Each factor is a full
        Tesseract pass per polarity per slice, so this is the dominant cost
        for multi-slice volumes -- pass a shorter tuple (e.g. ``(3,)``) to
        run on CPU-constrained or mobile hardware at reduced recall.
    check_inverted:
        When ``False``, skip the contrast-inverted OCR pass, halving the
        Tesseract calls at the cost of missing dark-on-light overlay text.
        ``True`` (default) preserves the original, most-thorough behavior.

    Returns the redacted array (a copy; the input is not mutated) and a
    list of plain-string audit notes. If no OCR backend is available, the
    array is returned unchanged with a note flagging that pixel data was
    NOT inspected -- callers should surface this prominently rather than
    silently proceeding as if redaction happened.
    """
    if not _ocr_available():
        return pixels, [
            "OCR backend unavailable (pytesseract or tesseract binary not found); "
            "pixel data was NOT inspected for burned-in text -- treat this file as unredacted."
        ]

    from PIL import Image, ImageOps

    factors = tuple(upscale_factors) if upscale_factors else _UPSCALE_FACTORS
    out = pixels.copy()
    notes: list[str] = []
    n_slices = out.shape[0]
    total_regions = 0
    low_conf_count = 0

    for i in range(n_slices):
        frame = out[i]
        is_color = frame.ndim == 3 and frame.shape[-1] in (3, 4)
        gray = frame[..., :3].mean(axis=-1) if is_color else frame

        lo, hi = float(gray.min()), float(gray.max())
        if hi <= lo:
            continue
        gray_u8 = ((gray - lo) / (hi - lo) * 255).astype(np.uint8)

        # Detection runs on a copy normalized to _REFERENCE_WIDTH, so every tuned pixel-size
        # constant (_MAX_WORD_HEIGHT, _ROW_EXTEND_TILE, ...) applies correctly regardless of the
        # caller's actual resolution -- see _normalize_for_ocr. Within its skip band (native
        # width already close to reference) this is a no-op and gray_u8 is used as-is; only a
        # genuinely different resolution is actually resized.
        norm_u8, scale_factor = _normalize_for_ocr(gray_u8)
        pil_norm = Image.fromarray(norm_u8)
        norm_boxes = _ocr_boxes(pil_norm, norm_u8, upscale_factors=factors)
        if check_inverted:
            norm_u8_inv = 255 - norm_u8
            pil_norm_inv = ImageOps.invert(pil_norm)
            norm_boxes += _ocr_boxes(pil_norm_inv, norm_u8_inv, upscale_factors=factors)
        boxes = [_denormalize_box(b, scale_factor) for b in norm_boxes]

        pad_native = max(int(round(padding / min(scale_factor, 1.0))), padding)

        regions_this_slice = 0
        for x0, y0, x1, y1, conf in boxes:
            y0p = max(0, y0 - pad_native)
            y1p = min(gray.shape[0], y1 + pad_native)
            x0p = max(0, x0 - pad_native)
            x1p = min(gray.shape[1], x1 + pad_native)
            if y1p <= y0p or x1p <= x0p:
                continue
            out[i, y0p:y1p, x0p:x1p] = 0
            regions_this_slice += 1
            if conf < flag_confidence:
                low_conf_count += 1

        if regions_this_slice:
            total_regions += regions_this_slice
            notes.append(f"slice {i}: redacted {regions_this_slice} burned-in text region(s)")

    if total_regions == 0:
        notes.append("OCR ran on all slices; no burned-in text detected")
    else:
        notes.insert(0, f"burned-in text OCR redaction: {total_regions} region(s) blacked out across {n_slices} slice(s)")
        if low_conf_count:
            notes.append(
                f"{low_conf_count} region(s) redacted despite low OCR confidence "
                "(redacted anyway per asymmetric-error policy: missed PHI > over-redaction) -- recommend human review"
            )

    if factors != _UPSCALE_FACTORS or not check_inverted:
        notes.append(
            f"reduced-thoroughness OCR pass (upscale_factors={factors}, "
            f"check_inverted={check_inverted}) -- lower recall than the full "
            f"{_UPSCALE_FACTORS} x both-polarities default; recommend human review"
        )

    return out, notes


# ---------------------------------------------------------------------------
# PNG / NIfTI -> fresh minimal DICOM
# ---------------------------------------------------------------------------

def array_to_dicom(
    pixels: np.ndarray,
    modality: str = "OT",
    redact_pixels: bool = True,
    ocr_upscale_factors: tuple[int, ...] | None = None,
    ocr_check_inverted: bool = True,
) -> tuple[Any, list[str]]:
    """Build a brand-new minimal DICOM ``Dataset`` around a pixel array.

    No patient/institution identifying field is ever set -- this is not a
    scrub of existing values, there simply are none. Only the technical
    tags required for a valid, viewable DICOM file are written, all UIDs
    are freshly random, and ``PatientIdentityRemoved`` / a
    ``DeidentificationMethod`` note document that intent for any downstream
    consumer.

    Parameters
    ----------
    pixels:
        ``(slices, H, W)`` grayscale, or ``(slices, H, W, C)`` color, any
        numeric dtype -- rescaled to ``uint8`` (color) or ``uint16``
        (grayscale) for storage.
    modality:
        DICOM Modality code; ``"OT"`` (Other) is the safe default for
        pixel data that did not originate from a DICOM modality.
    redact_pixels:
        When ``True`` (default), OCRs the pixel data and blacks out any
        detected text (e.g. a PNG screenshot of a viewer with a
        corner-annotated patient banner) before it is baked into the
        DICOM. See :func:`redact_burned_in_text`.
    ocr_upscale_factors, ocr_check_inverted:
        Passed straight through to :func:`redact_burned_in_text` to trade
        OCR thoroughness for speed on CPU-constrained/mobile hardware. See
        that function's docstring.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    notes: list[str] = []

    pixels = np.asarray(pixels)
    if redact_pixels:
        pixels, ocr_notes = redact_burned_in_text(
            pixels.astype(np.float32),
            upscale_factors=ocr_upscale_factors,
            check_inverted=ocr_check_inverted,
        )
        notes.extend(ocr_notes)

    is_color = pixels.ndim == 4 and pixels.shape[-1] in (3, 4)
    n_slices = pixels.shape[0]

    if is_color:
        arr = pixels[..., :3]
        arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() <= 255 else _rescale_to(arr, np.uint8)
    else:
        arr = _rescale_to(pixels, np.uint16)

    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = modality
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.ConversionType = "WSD"

    ds.Rows, ds.Columns = int(arr.shape[1]), int(arr.shape[2])
    ds.NumberOfFrames = int(n_slices)
    ds.SamplesPerPixel = 3 if is_color else 1
    ds.PhotometricInterpretation = "RGB" if is_color else "MONOCHROME2"
    if is_color:
        ds.PlanarConfiguration = 0
    ds.BitsAllocated = 8 if is_color else 16
    ds.BitsStored = 8 if is_color else 16
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "ProST-Annomyze: synthesized, no ID fields set"

    notes.append(f"synthesized fresh DICOM dataset (modality={modality}, frames={n_slices})")
    notes.append("no PatientName/PatientID/dates ever set (source had no such header)")
    notes.append(f"StudyInstanceUID/SeriesInstanceUID/SOPInstanceUID freshly randomized ({sop_instance_uid[:24]}...)")

    return ds, notes


def _rescale_to(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    arr = arr.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=dtype)
    max_val = np.iinfo(dtype).max
    scaled = (arr - lo) / (hi - lo) * max_val
    return scaled.astype(dtype)


# ---------------------------------------------------------------------------
# Real DICOM -> full-wipe DICOM (strip everything non-technical)
# ---------------------------------------------------------------------------

def full_wipe_dicom(
    ds: Any,
    redact_pixels: bool = True,
    ocr_upscale_factors: tuple[int, ...] | None = None,
    ocr_check_inverted: bool = True,
) -> tuple[Any, list[str]]:
    """Strip every tag from ``ds`` except the minimal technical set.

    Stricter than :func:`prostate_deid.pipeline.default_pipeline`: no named
    profile, no exclusions -- anything not in :data:`_TECHNICAL_KEEP` is
    removed, public or private, regardless of whether it appears on any
    HIPAA Safe Harbor list. All UIDs are replaced with freshly random ones
    (not deterministically pseudonymized), since this path makes no promise
    of cross-file linkability.

    When ``redact_pixels`` is ``True`` (default), also OCRs the pixel data
    itself and blacks out any detected text -- header tag removal alone
    does not touch PHI burned directly into the image (scanner corner
    annotations, fusion-biopsy overlay labels), which is a separate attack
    surface from the header. See :func:`redact_burned_in_text`.

    ``ocr_upscale_factors``/``ocr_check_inverted`` are passed straight
    through to :func:`redact_burned_in_text` to trade OCR thoroughness for
    speed on CPU-constrained/mobile hardware.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    notes: list[str] = []
    removed = 0
    for elem in list(ds):
        if elem.tag == 0x7FE00010:  # PixelData
            continue
        keyword = elem.keyword
        if keyword not in _TECHNICAL_KEEP:
            del ds[elem.tag]
            removed += 1
    notes.append(f"removed {removed} non-technical tag(s) (full wipe, no profile exclusions)")

    if redact_pixels and hasattr(ds, "pixel_array"):
        pixels = np.asarray(ds.pixel_array)
        orig_dtype = pixels.dtype
        if pixels.ndim == 2:
            stack = pixels[np.newaxis, :, :]
        elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            stack = pixels[np.newaxis, ...]
        else:
            stack = pixels
        redacted, ocr_notes = redact_burned_in_text(
            stack.astype(np.float32),
            upscale_factors=ocr_upscale_factors,
            check_inverted=ocr_check_inverted,
        )
        notes.extend(ocr_notes)
        redacted = redacted.astype(orig_dtype).reshape(pixels.shape)
        ds.PixelData = redacted.tobytes()

    new_sop = generate_uid()
    new_study = generate_uid()
    new_series = generate_uid()
    ds.SOPInstanceUID = new_sop
    ds.StudyInstanceUID = new_study
    ds.SeriesInstanceUID = new_series
    notes.append("StudyInstanceUID/SeriesInstanceUID/SOPInstanceUID replaced with fresh random UIDs")

    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPInstanceUID = new_sop
    ds.file_meta.MediaStorageSOPClassUID = getattr(ds, "SOPClassUID", pydicom.uid.SecondaryCaptureImageStorage)
    if not getattr(ds.file_meta, "TransferSyntaxUID", None):
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "ProST-Annomyze: full wipe, technical tags only"
    notes.append("PatientIdentityRemoved=YES, DeidentificationMethod set")

    return ds, notes


# ---------------------------------------------------------------------------
# DICOM -> .nii.gz
# ---------------------------------------------------------------------------

def dicom_to_nifti_bytes(ds: Any) -> bytes:
    """Encode a (already de-identified) DICOM dataset's pixel data as gzip-compressed ``.nii.gz`` bytes.

    Only pixel data and voxel spacing survive -- a NIfTI header has no
    patient-identifying fields to carry over in the first place, so there
    is nothing further to scrub on this hop.
    """
    import gzip

    import nibabel as nib

    pixels = np.asarray(ds.pixel_array)
    if pixels.ndim == 2:
        pixels = pixels[:, :, np.newaxis]
    elif pixels.ndim == 3:
        # (frames, H, W) -> NIfTI (X, Y, Z) = (W, H, frames)
        pixels = np.moveaxis(pixels, 0, -1)

    spacing = getattr(ds, "PixelSpacing", None)
    thickness = getattr(ds, "SliceThickness", None)
    affine = np.eye(4, dtype=np.float64)
    if spacing:
        affine[0, 0] = float(spacing[1])
        affine[1, 1] = float(spacing[0])
    if thickness:
        affine[2, 2] = float(thickness)

    img = nib.Nifti1Image(pixels.astype(np.float32), affine)
    raw_buf = io.BytesIO()
    file_map = img.make_file_map({"image": raw_buf, "header": raw_buf})
    img.to_file_map(file_map)

    gz_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buf, mode="wb", mtime=0) as gz:
        gz.write(raw_buf.getvalue())
    return gz_buf.getvalue()
