"""Quantitative verification: eq. 40 region metrics, eq. 41 footprint errors, synthetic self-test."""

from __future__ import annotations

import math

import numpy as np
import pytest

from prostate_deid import universal_io as uio
from prostate_deid.evaluation import footprint_errors, region_metrics, summarize_regions, synthetic_self_test
from prostate_deid.text_restore import RegionResult

needs_ocr = pytest.mark.skipif(not uio._ocr_available(), reason="tesseract not installed")


def test_region_metrics_match_eq40():
    regions = [
        RegionResult(0, (0, 0, 10, 10), 85, "restored", "segmented", "bright", 25, 100),
        RegionResult(0, (0, 0, 10, 10), 85, "restored", "segmented", "bright", 25, 100),  # duplicate
        RegionResult(0, (20, 0, 30, 5), 140, "fallback_blackout", "leaked", "bright", 50, 50),
        RegionResult(0, (40, 0, 50, 5), -5, "skipped_not_text", "not text", "n/a", 0, 50),
    ]
    rows = region_metrics(regions)
    assert len(rows) == 3
    assert rows[0]["retained"] == pytest.approx(0.75)  # 1 - 25/100
    assert rows[0]["c_audit"] == pytest.approx(0.85)
    assert rows[1]["retained"] == 0.0 and rows[1]["c_audit"] == 1.0  # clipped to 1
    assert math.isnan(rows[2]["c_audit"])  # no OCR confidence recorded
    s = summarize_regions(rows)
    assert (s["regions"], s["restored"], s["fallback"], s["skipped"]) == (3, 1, 1, 1)
    assert s["retained_mean"] == pytest.approx(0.375)  # skipped regions do not count
    assert s["retained_min"] == 0.0


def test_footprint_errors_match_eq41():
    clean = np.zeros((4, 4))
    clean[0, 0] = 10.0  # range 0..10 -> scale 10
    out = clean.copy()
    out[1, 1] = 1.0  # error 0.1 after scaling
    out[1, 2] = 3.0  # error 0.3
    f = np.zeros((4, 4), bool)
    f[1, 1:3] = True
    e = footprint_errors(out, clean, f)
    assert e["mae"] == pytest.approx(0.2)
    assert e["mse"] == pytest.approx((0.01 + 0.09) / 2)
    assert e["psnr_db"] == pytest.approx(10 * math.log10(1 / 0.05))
    assert footprint_errors(clean, clean, f)["psnr_db"] == math.inf


@needs_ocr
def test_synthetic_self_test_removes_text_and_keeps_the_rest():
    from _text_fixtures import phantom

    clean = (phantom(256).astype(np.float32) / 2200 * 200).clip(0, 200).astype(np.uint8)
    res = synthetic_self_test(clean, method="telea")
    assert res.chars_total > 50
    assert res.glyph_px_changed > 0.95
    assert res.chars_readable_after <= 0.1 * max(1, res.chars_readable_before)
    assert res.psnr_db > 15
    # Away from the text, changes are the padded line bands refilled on the black background; tissue
    # pixels (clean > 20) away from the text stay essentially untouched.
    from scipy import ndimage as ndi

    away = ~ndi.binary_dilation(res.footprint, iterations=2)
    changed = res.restored.astype(int) != res.burned.astype(int)
    assert (changed & away & (clean > 20)).sum() < 0.001 * clean.size
    assert res.outside_px_changed < 0.03 * clean.size
    d = res.as_dict()
    assert "burned" not in d and "residual_fraction" in d
