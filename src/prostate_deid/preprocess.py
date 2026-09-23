"""
Robust Anatomy-Referenced Normalization (RARN).

Standardizes MRI intensity using robust statistics of a reference region
rather than the whole image, so the normalizer is (to first order)
equivariant to the unknown per-scan affine gain that makes raw MRI
intensities non-quantitative. See ``docs/normalization_math.md`` (or the
manuscript's Section 2.1.1) for the full derivation; this module is a direct
implementation of equations (3)-(10) there.

Nothing here is claimed as a novel estimator: the median, the median
absolute deviation (MAD), and the 1.4826 consistency constant are standard
robust statistics, and reference-region normalization for MRI is prior art
(e.g. WhiteStripe). The contribution is the specialization to the prostate
gland/zonal labels this library already carries, the optional zone-aware
mode, and recording the exact normalization affine in the same audit trail
as the de-identification steps, which is what makes the mapping invertible
and reproducible (see :meth:`IntensityNormalize.apply`, which records
``(location, scale, q_lo, q_hi, reference_kind)`` per call).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .core import AuditEntry, Sample, Transform
from .organ_mask import WHOLE_GLAND

__all__ = ["IntensityNormalize"]

_MAD_CONSISTENCY_CONSTANT = 1.4826  # 1 / Phi^-1(0.75)

Method = Literal["robust", "zscore", "minmax"]


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's threshold: maximizes between-class variance omega0*omega1*(mu0-mu1)^2."""
    hist, bin_edges = np.histogram(values, bins=n_bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return float(values.min()) if values.size else 0.0

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    weighted_sum = np.cumsum(hist * bin_centers)
    total_sum = weighted_sum[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean1 = weighted_sum / weight1
        mean2 = (total_sum - weighted_sum) / weight2

    between_var = weight1 * weight2 * (mean1 - mean2) ** 2
    between_var = np.nan_to_num(between_var, nan=-1.0)
    idx = int(np.argmax(between_var))
    return float(bin_centers[idx])


def _reference_values(sample: Sample, reference_labels: Sequence[int] | None) -> tuple:
    """Return (values, reference_kind) per the priority order in the math doc."""
    if sample.mask is not None:
        labels = set(reference_labels) if reference_labels is not None else set(WHOLE_GLAND)
        keep = np.isin(sample.mask, list(labels))
        if keep.any():
            has_channel_axis = sample.image.ndim > sample.mask.ndim
            ref = sample.image[..., keep] if has_channel_axis else sample.image[keep]
            return ref, "anatomical_mask"

    flat = sample.image.reshape(-1)
    if flat.size == 0:
        return flat, "whole_image"
    tau = _otsu_threshold(flat)
    fg = flat[flat >= tau]
    if fg.size >= max(64, int(0.01 * flat.size)):
        return fg, "otsu_foreground"

    return flat, "whole_image"


def _robust_location_scale(values: np.ndarray) -> tuple:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = _MAD_CONSISTENCY_CONSTANT * mad
    if scale <= 0:
        std = float(np.std(values))
        scale = std if std > 0 else 1.0
    return med, scale


class IntensityNormalize(Transform):
    """Robust Anatomy-Referenced Normalization (RARN).

    Parameters
    ----------
    method:
        ``"robust"`` (default): location/scale = median / (1.4826 * MAD) of
        the winsorized reference. ``"zscore"``: mean/SD of the winsorized
        reference. ``"minmax"``: location/scale = (q_lo, q_hi - q_lo).
    reference_labels:
        Mask label values defining the reference region when a mask is
        present (default: :data:`prostate_deid.organ_mask.WHOLE_GLAND`).
    percentiles:
        ``(p_lo, p_hi)`` winsorization tail percentiles (default ``(1, 99)``).
    zone_aware:
        If ``True``, after the global map, each present zone label is
        additionally standardized with its own robust statistics
        (equation 10 in the math doc). This changes inter-zone contrast and
        is appropriate only when the downstream task is zone-local (e.g.
        PI-RADS-style zonal assessment); use the global map when absolute
        inter-zone contrast must be preserved. Degrades to the global map
        when no mask is present.
    """

    def __init__(
        self,
        method: Method = "robust",
        reference_labels: Sequence[int] | None = None,
        percentiles: tuple = (1.0, 99.0),
        zone_aware: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.reference_labels = reference_labels
        self.percentiles = percentiles
        self.zone_aware = zone_aware

    def apply(self, sample: Sample) -> Sample:
        ref_values, reference_kind = _reference_values(sample, self.reference_labels)
        p_lo, p_hi = self.percentiles
        q_lo, q_hi = np.percentile(ref_values, [p_lo, p_hi])

        winsorized_ref = np.clip(ref_values, q_lo, q_hi)

        if self.method == "robust":
            location, scale = _robust_location_scale(winsorized_ref)
        elif self.method == "zscore":
            location, scale = float(np.mean(winsorized_ref)), float(np.std(winsorized_ref)) or 1.0
        elif self.method == "minmax":
            location, scale = float(q_lo), float(q_hi - q_lo) or 1.0
        else:
            raise ValueError(f"Unknown method {self.method!r}; expected 'robust', 'zscore', or 'minmax'")

        clipped = np.clip(sample.image, q_lo, q_hi)
        sample.image = (clipped - location) / scale
        sample.image_dirty = True

        sample.log(AuditEntry(
            step="intensity_normalize",
            action=f"rarn_{self.method}",
            detail=(
                f"reference={reference_kind}, location={location:.4f}, scale={scale:.4f}, "
                f"q_lo={q_lo:.4f}, q_hi={q_hi:.4f}, percentiles={self.percentiles}"
            ),
        ))

        if self.zone_aware and sample.mask is not None:
            for zone_label in sorted(WHOLE_GLAND):
                zone_voxels = sample.mask == zone_label
                if not zone_voxels.any():
                    continue
                has_channel_axis = sample.image.ndim > sample.mask.ndim
                zone_values = (
                    sample.image[..., zone_voxels] if has_channel_axis else sample.image[zone_voxels]
                )
                z_loc, z_scale = _robust_location_scale(zone_values) if self.method == "robust" else (
                    float(np.mean(zone_values)), float(np.std(zone_values)) or 1.0
                )
                if sample.image.ndim > sample.mask.ndim:
                    sample.image[..., zone_voxels] = (sample.image[..., zone_voxels] - z_loc) / z_scale
                else:
                    sample.image[zone_voxels] = (sample.image[zone_voxels] - z_loc) / z_scale
                sample.log(AuditEntry(
                    step="intensity_normalize",
                    action="rarn_zone_aware",
                    detail=f"zone_label={zone_label}, location={z_loc:.4f}, scale={z_scale:.4f}",
                ))

        return sample
