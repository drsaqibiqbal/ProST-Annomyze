"""Tests for Robust Anatomy-Referenced Normalization (RARN).

Includes the scanner-invariance check described in
docs/normalization_math.md Section 3: two synthetic "scanners" related by
an affine gain (a=2.5, b=40) should normalize to (numerically) the same
distribution, since RARN's location/scale estimators transform the same way
under an affine gain and cancel out (Section 3, equations 3-8).
"""

import numpy as np

from prostate_deid.core import Sample
from prostate_deid.preprocess import IntensityNormalize


def _make_sample(image):
    return Sample(image=image.astype(np.float32))


def test_robust_normalize_whole_image_no_mask_falls_back_to_otsu_or_whole_image():
    """No mask -> reference falls back to Otsu foreground (or whole image if
    Otsu can't separate a foreground), per the priority order in
    docs/normalization_math.md Section 2. Build a background+object image so
    Otsu foreground is meaningful, and check the object region is
    approximately centered at 0 after normalization."""
    rng = np.random.default_rng(0)
    image = rng.normal(loc=20, scale=5, size=(1, 16, 16, 16))  # background
    image[:, 4:12, 4:12, 4:12] = rng.normal(loc=500, scale=50, size=(1, 8, 8, 8))  # object

    sample = _make_sample(image)
    out = IntensityNormalize(method="robust", always_apply=True)(sample)

    assert len(out.audit) == 1
    assert out.audit[0].step == "intensity_normalize"
    detail = out.audit[0].detail
    assert "reference=otsu_foreground" in detail or "reference=whole_image" in detail
    object_region = out.image[:, 4:12, 4:12, 4:12]
    assert abs(float(np.median(object_region))) < 1.0


def test_robust_normalize_uses_anatomical_mask_when_present(synthetic_sample):
    out = IntensityNormalize(method="robust", always_apply=True)(synthetic_sample)
    assert "anatomical_mask" in out.audit[0].detail


def test_scanner_invariance_affine_equivariance():
    rng = np.random.default_rng(1)
    base = rng.normal(loc=500, scale=50, size=(1, 20, 20, 20))

    a, b = 2.5, 40.0
    scanner_a = base.copy()
    scanner_b = a * base + b

    norm_a = IntensityNormalize(method="robust", always_apply=True)(_make_sample(scanner_a))
    norm_b = IntensityNormalize(method="robust", always_apply=True)(_make_sample(scanner_b))

    # Unclipped voxels should normalize to (numerically) the same distribution;
    # allow small tolerance for winsorization-boundary voxels.
    diff = np.abs(norm_a.image - norm_b.image)
    assert float(np.percentile(diff, 95)) < 0.15


def test_zero_scale_reference_falls_back_to_std():
    constant = np.full((1, 8, 8, 8), 100.0)
    out = IntensityNormalize(method="robust", always_apply=True)(_make_sample(constant))
    assert np.all(np.isfinite(out.image))


def test_zone_aware_mode_adds_per_zone_audit_entries(synthetic_sample):
    out = IntensityNormalize(method="robust", zone_aware=True, always_apply=True)(synthetic_sample)
    zone_entries = [e for e in out.audit if e.action == "rarn_zone_aware"]
    assert len(zone_entries) >= 1


def test_minmax_and_zscore_modes_run():
    rng = np.random.default_rng(2)
    image = rng.normal(loc=200, scale=30, size=(1, 10, 10, 10))
    for method in ("zscore", "minmax"):
        out = IntensityNormalize(method=method, always_apply=True)(_make_sample(image.copy()))
        assert np.all(np.isfinite(out.image))
