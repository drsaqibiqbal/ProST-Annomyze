"""v2 burned-in text removal: segmentation, image retention, verification fallbacks, v1 equivalence."""

from __future__ import annotations

import os

import numpy as np
import pytest
from _text_fixtures import burned_slice

from prostate_deid import text_restore as tr
from prostate_deid import universal_io as uio
from prostate_deid.text_restore import (
    _GHOST_RATIO,
    ghost_ratio,
    inpaint_mask,
    restore_burned_in_text,
    segment_text_pixels,
)

needs_ocr = pytest.mark.skipif(not uio._ocr_available(), reason="tesseract not installed")


def _box_of(alpha, pad=3):
    ys, xs = np.nonzero(alpha > 0.02)
    return (int(xs.min()) - pad, int(ys.min()) - pad, int(xs.max()) + 1 + pad, int(ys.max()) + 1 + pad)


def _norm(a):
    a = a.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min())


@pytest.mark.parametrize("layout", ["corner", "anatomy"])
def test_segmentation_finds_glyph_pixels(layout):
    clean, burned, alpha = burned_slice(layout)
    box = _box_of(alpha)
    mask, info = segment_text_pixels(_norm(burned), box)
    assert mask is not None, info
    glyph = alpha > 0.5
    assert (mask & glyph).sum() / glyph.sum() >= 0.97  # recall of solid glyph pixels
    x0, y0, x1, y1 = box
    assert mask[y0:y1, x0:x1].mean() < 0.6  # most of the box is NOT replaced
    assert info["polarity"] == "bright"


def test_restore_keeps_every_unmasked_pixel_bit_identical():
    clean, burned, alpha = burned_slice("anatomy")
    boxes = [(*_box_of(alpha), 90)]
    res = restore_burned_in_text(burned[None], method="telea", boxes_per_slice=[boxes], verify=False)
    out = res.pixels[0]
    assert out.dtype == burned.dtype
    keep = ~res.inpaint_mask[0]
    assert np.array_equal(out[keep], burned[keep])
    assert res.inpaint_mask[0].sum() < 0.6 * (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])


@pytest.mark.parametrize("method", ["telea", "ns", "biharmonic"])
def test_restoration_is_closer_to_truth_than_blackout(method):
    clean, burned, alpha = burned_slice("anatomy")
    boxes = [(*_box_of(alpha), 90)]
    restored = restore_burned_in_text(burned[None], method=method, boxes_per_slice=[boxes], verify=False).pixels[0]
    black = restore_burned_in_text(burned[None], method="blackout", boxes_per_slice=[boxes], verify=False).pixels[0]
    fp = alpha > 0.02
    err_r = np.abs(restored.astype(float) - clean)[fp].mean()
    err_b = np.abs(black.astype(float) - clean)[fp].mean()
    assert err_r < 0.5 * err_b


def test_blackout_matches_v1_redaction_exactly(monkeypatch):
    clean, burned, alpha = burned_slice("corner")
    boxes = [(*_box_of(alpha), 50), (0, 0, 10, 10, 20)]
    monkeypatch.setattr(uio, "_ocr_available", lambda: True)
    monkeypatch.setattr(uio, "detect_text_boxes", lambda *a, **k: boxes)
    v1, _ = uio.redact_burned_in_text(burned[None].astype(np.float32))
    v2 = restore_burned_in_text(burned[None].astype(np.float32), method="blackout", boxes_per_slice=[boxes]).pixels
    assert np.array_equal(v1, v2)


def test_ghost_ratio_flags_offset_fill():
    clean, burned, alpha = burned_slice("anatomy")
    g = _norm(burned)
    from scipy import ndimage as ndi

    m = ndi.binary_dilation(alpha > 0.02, iterations=1)
    good = inpaint_mask(g, m, "biharmonic")
    bad = good.copy()
    bad[m] = np.clip(bad[m] + 0.15, 0, 1)  # uniformly too bright: letter shapes visible
    assert ghost_ratio(good, m) < _GHOST_RATIO < ghost_ratio(bad, m)


def test_failed_restoration_falls_back_to_blackout(monkeypatch):
    clean, burned, alpha = burned_slice("corner")
    boxes = [(*_box_of(alpha), 90)]
    monkeypatch.setattr(tr, "inpaint_mask", lambda img, mask, method: img.copy())  # "restores" nothing
    monkeypatch.setattr(tr, "_boxes_still_text", lambda restored, bxs, *a: [True] * len(bxs))
    res = restore_burned_in_text(burned[None], method="telea", boxes_per_slice=[boxes], verify=True)
    assert res.regions[0].action == "fallback_blackout"
    x0, y0, x1, y1, _ = boxes[0]
    # Restoration methods re-synthesise the fallback rectangle from the pixels around it (here the
    # black corner), so nothing of the text survives.
    filled = res.pixels[0][max(0, y0):y1, max(0, x0):x1]
    assert filled.max() < 0.1 * burned.max()
    assert any(e.action == "redacted_region_fallback" and e.flagged_for_review for e in res.audit)


def test_implausible_segmentation_falls_back():
    flat = np.zeros((60, 60), np.float32)
    flat[10:30, 10:50] = 1.0  # a solid block is not text
    m, info = segment_text_pixels(flat, (10, 10, 50, 30))
    assert m is None


def test_vit_unavailable_falls_back_to_biharmonic(monkeypatch):
    clean, burned, alpha = burned_slice("anatomy")
    monkeypatch.setattr(tr, "vit_mae_available", lambda: False)
    res = restore_burned_in_text(burned[None], method="vit_mae_poisson", boxes_per_slice=[[(*_box_of(alpha), 90)]],
                                 verify=False)
    assert res.method == "biharmonic"
    assert any(e.action == "vit_unavailable" and e.flagged_for_review for e in res.audit)


def test_no_ocr_returns_input_and_flags(monkeypatch):
    clean, burned, alpha = burned_slice("corner")
    monkeypatch.setattr(uio, "_ocr_available", lambda: False)
    res = restore_burned_in_text(burned[None])
    assert np.array_equal(res.pixels[0], burned)
    assert any(e.action == "ocr_unavailable" and e.flagged_for_review for e in res.audit)


@needs_ocr
def test_overlay_text_on_anatomy_found_where_v1_misses():
    clean, burned, alpha = burned_slice("anatomy")
    res = restore_burned_in_text(burned[None], method="telea")
    glyph = alpha > 0.5
    assert (res.pixels[0][glyph] != burned[glyph]).mean() > 0.95
    assert tr.read_crop_text(_norm(res.pixels[0]), _box_of(alpha), min_conf=60)[0] < 3


@needs_ocr
def test_corner_text_removed_end_to_end():
    clean, burned, alpha = burned_slice("corner")
    res = restore_burned_in_text(burned[None], method="telea")
    glyph = alpha > 0.5
    assert (res.pixels[0][glyph] != burned[glyph]).mean() > 0.95
    assert res.regions and all(r.action in ("restored", "fallback_blackout") for r in res.regions)


@pytest.mark.skipif(os.environ.get("PROSTATE_DEID_TEST_VIT") != "1",
                    reason="set PROSTATE_DEID_TEST_VIT=1 to run (downloads facebook/vit-mae-base, ~450 MB)")
def test_vit_mae_poisson_restores_without_ghost():
    clean, burned, alpha = burned_slice("anatomy")
    from scipy import ndimage as ndi

    m = ndi.binary_dilation(alpha > 0.02, iterations=1)
    g = _norm(burned)
    out = inpaint_mask(g, m, "vit_mae_poisson")
    assert ghost_ratio(out, m) < _GHOST_RATIO
    assert np.array_equal(out[~m], g[~m])


@needs_ocr
def test_jpeg_screenshot_text_restored_not_blacked_out():
    # Scanner-screenshot layout: white corner text on black, JPEG-compressed. v2 must restore the glyphs
    # (not fall back to whole-box blackout because of JPEG halos) and must not touch the anatomy.
    import io

    from _text_fixtures import burn, phantom
    from PIL import Image

    from prostate_deid.convert import _raster_to_array

    clean = (phantom(256).astype(np.float32) / 2200 * 160).clip(0, 160).astype(np.uint8)
    burned, alpha = burn(clean, [("DOE^JANE 67Y", (3, 3)), ("ID SYN-0012345", (3, 17))], 12, level=255)
    buf = io.BytesIO()
    Image.fromarray(burned).save(buf, "JPEG", quality=85)
    arr, _ = _raster_to_array(buf.getvalue())
    res = restore_burned_in_text(arr[None], method="telea")
    assert res.regions and all(r.action == "restored" for r in res.regions if r.action != "skipped_not_text")
    assert not (res.inpaint_mask[0] & (clean > 20)).any()  # anatomy untouched
    assert tr.read_crop_text(_norm(res.pixels[0]), _box_of(alpha), min_conf=40)[0] < 3


@needs_ocr
@pytest.mark.parametrize("rim", [0.8, 0.88])
def test_bright_fat_rim_and_septa_are_not_treated_as_text(rim):
    # Thin bright tissue (subcutaneous fat rim, fat septa) passes a top-hat test like a letter stroke.
    # v2 must remove the burned-in text and leave every tissue pixel away from the text unchanged.
    from _text_fixtures import burn
    from scipy import ndimage as ndi

    rng = np.random.default_rng(3)
    h, w = 300, 360
    yy, xx = np.mgrid[0:h, 0:w]
    r = ((xx - w / 2) / 160) ** 2 + ((yy - h / 2) / 100) ** 2
    img = np.zeros((h, w))
    img[r <= 1] = 90 + 30 * np.sin(xx[r <= 1] / 7) * np.cos(yy[r <= 1] / 11) + rng.normal(0, 12, (r <= 1).sum())
    rim_px = (r > 0.90) & (r <= 0.97)
    img[rim_px] = 255 * rim
    t = np.linspace(0, np.pi, 400)
    for k in range(5):
        px = (w / 2 + 60 * np.cos(k) + 70 * np.cos(t + k)).astype(int)
        py = (h / 2 + 30 * np.sin(2 * k) + 25 * np.sin(3 * t + k)).astype(int)
        img[np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)] = 255 * rim
    clean = np.clip(img, 0, 255).astype(np.uint8)
    items = [("DOE^JANE 67Y", (3, 3)), ("ID SYN-0012345", (3, 18)), ("Images: 9/30", (3, h - 30))]
    burned, alpha = burn(clean, items, 12, level=255)
    res = restore_burned_in_text(burned[None], method="telea")
    text_zone = ndi.binary_dilation(alpha > 0.02, iterations=4)
    changed = res.pixels[0] != burned
    assert not (changed & ~text_zone & (clean > 0)).any()
    glyph = alpha > 0.5
    assert (res.pixels[0][glyph] != burned[glyph]).mean() > 0.95


def _viewer_screenshot(text_level: int, tissue_level: int = 0, rng_seed: int = 5):
    """Viewer-export layout: bright fat at 255 (brighter than the text), a left-aligned corner block at
    the bottom drawn at ``text_level`` over a band of ``tissue_level`` texture. Returns
    (clean, burned, alpha, items)."""
    from _text_fixtures import burn

    rng = np.random.default_rng(rng_seed)
    h, w = 320, 320
    yy, xx = np.mgrid[0:h, 0:w]
    r = ((xx - w / 2) / 140) ** 2 + ((yy - h / 2.4) / 90) ** 2
    img = np.zeros((h, w))
    img[r <= 1] = 70 + 20 * np.sin(xx[r <= 1] / 6) + rng.normal(0, 8, (r <= 1).sum())
    img[(r > 0.85) & (r <= 1)] = 255  # fat rim: the brightest thing on the slice
    if tissue_level:
        band = yy >= h - 110
        img[band] = tissue_level + 25 * np.sin(xx[band] / 9) * np.cos(yy[band] / 7) + rng.normal(0, 8, band.sum())
    clean = np.clip(img, 0, 255).astype(np.uint8)
    items = [(t, (3, h - 100 + 18 * k)) for k, t in
             enumerate(["ST: 3.00 mm SL: -18.19", "RT: 4337.826 ET: 79.2", "FS: 1.5", "Images: 12/35", "Series: 804"])]
    burned, alpha = burn(clean, items, 12, level=text_level)
    return clean, burned, alpha, items


def test_stroke_lines_find_text_dimmer_than_anatomy():
    # Text at 215 with fat at 255: the saturation rule (glyph core within 6% of the slice max) rejects
    # every line; the stroke-line detector measures strokes against their surroundings instead.
    clean, burned, alpha, _ = _viewer_screenshot(text_level=215)
    lines = tr.detect_stroke_text_lines(_norm(burned))
    found = np.zeros(burned.shape, bool)
    for ln in lines:
        found |= ln.mask
    glyph = alpha > 0.5
    assert (found & glyph).sum() / glyph.sum() >= 0.95


def test_stroke_lines_find_text_over_bright_tissue():
    clean, burned, alpha, _ = _viewer_screenshot(text_level=230, tissue_level=150)
    found = np.zeros(burned.shape, bool)
    for ln in tr.detect_stroke_text_lines(_norm(burned)):
        found |= ln.mask
    glyph = alpha > 0.5
    assert (found & glyph).sum() / glyph.sum() >= 0.9


@pytest.mark.parametrize("tissue_level", [0, 150])
def test_stroke_lines_ignore_clean_anatomy(tissue_level):
    # Same slices without text: fat rim, textured tissue and bright band must not form text lines.
    clean, _, _, _ = _viewer_screenshot(text_level=230, tissue_level=tissue_level)
    assert tr.detect_stroke_text_lines(_norm(clean)) == []


@pytest.mark.parametrize("name", ["MR2_UNCI.dcm", "MR-SIEMENS-DICOM-WithOverlays.dcm"])
def test_stroke_lines_ignore_real_mr_without_text(name):
    # Public pydicom test slices (1024 and 484 px) carry no burned-in text: fine tissue texture on a
    # large matrix must not form text lines.
    import pydicom
    from pydicom.data import get_testdata_file

    px = pydicom.dcmread(get_testdata_file(name)).pixel_array
    assert tr.detect_stroke_text_lines(_norm(px)) == []


def test_line_grid_fills_a_line_too_faint_to_pass_alone():
    # Middle line of a block drawn barely above the tissue: it is recovered from the block's line pitch.
    from _text_fixtures import burn

    clean, burned, alpha, items = _viewer_screenshot(text_level=230, tissue_level=150)
    faint, faint_alpha = burn(clean, [items[2]], 12, level=175)
    others = np.where(alpha > 0, burned, clean)
    mixed = np.where(faint_alpha > 0, faint, others).astype(np.uint8)
    lines = tr.detect_stroke_text_lines(_norm(mixed))
    found = np.zeros(mixed.shape, bool)
    for ln in lines:
        found |= ln.mask
    glyph = faint_alpha > 0.5
    assert (found & glyph).sum() / glyph.sum() >= 0.9


@needs_ocr
def test_text_dimmer_than_anatomy_removed_end_to_end():
    # Regression (0.2.1-0.2.3): lines of a viewer block below the slice max were left on the image.
    clean, burned, alpha, _ = _viewer_screenshot(text_level=215)
    res = restore_burned_in_text(burned[None], method="telea")
    glyph = alpha > 0.5
    assert (res.pixels[0][glyph] != burned[glyph]).mean() > 0.95


def test_fallback_on_grey_background_is_filled_grey(monkeypatch):
    # A fallback rectangle on a grey background (ADC map, screenshot) takes the surrounding level, not 0.
    from _text_fixtures import burn

    clean = np.full((128, 128), 90, np.uint8)
    burned, alpha = burn(clean, [("ID SYN-0012345", (3, 3))], 12, level=255)
    boxes = [(*_box_of(alpha), 90)]
    monkeypatch.setattr(tr, "_boxes_still_text", lambda restored, bxs, *a: [True] * len(bxs))
    res = restore_burned_in_text(burned[None], method="telea", boxes_per_slice=[boxes], verify=True)
    x0, y0, x1, y1, _ = boxes[0]
    out = res.pixels[0]
    assert res.regions[0].action == "fallback_blackout"
    assert np.all(np.abs(out[res.inpaint_mask[0]].astype(int) - 90) <= 2)


def test_duplicate_detector_boxes_are_processed_once(monkeypatch):
    box = (2, 2, 40, 14, 55)
    monkeypatch.setattr(uio, "detect_text_boxes", lambda *a, **k: [box, (*box[:4], 70), box])
    out = tr.detect_text_boxes_v2(np.zeros((64, 64), np.float32), detect_overlay=False)
    assert out == [(*box[:4], 70)]


def test_changed_region_boxes_hide_glyph_shapes():
    # The review view must show only rectangles: drawing the inpaint mask itself redraws the letters.
    from prostate_deid.text_restore import changed_region_boxes

    clean, burned, alpha = burned_slice("corner")
    mask = alpha > 0.02
    boxes = changed_region_boxes(mask)
    from scipy import ndimage as ndi

    n_letters = ndi.label(mask)[1]
    assert 1 <= len(boxes) <= n_letters // 4  # per word / line, never per letter
    covered = np.zeros_like(mask)
    for x0, y0, x1, y1 in boxes:
        covered[y0:y1, x0:x1] = True
    assert covered[mask].all()
    assert changed_region_boxes(np.zeros((8, 8), bool)) == []
