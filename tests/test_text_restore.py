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
    assert (res.pixels[0][max(0, y0):y1, max(0, x0):x1] == 0).all()
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

    from PIL import Image
    from _text_fixtures import burn, phantom

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
