"""
Deterministic prostate anatomical masking.

Restricts an image to a segmented region -- whole gland, a single zone, or a
lesion -- given a pre-computed label mask. This module does not segment
anything itself (see README limitations): it assumes a mask has already been
produced by a segmentation model or a human annotator and shares that mask's
convention with mask-consistent augmentation transforms, so the same
``(image, mask)`` pair can flow through this library and then into an
augmentation pipeline (e.g. Bio-Volumentations) without a format conversion
step in between.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .core import AuditEntry, Sample, Transform

__all__ = ["OrganMask", "WHOLE_GLAND", "PERIPHERAL_ZONE", "TRANSITION_ZONE", "LESION"]

# Label convention (uint8). Real segmentations label zones exclusively
# (a voxel is PZ or TZ, not both, and not a redundant separate "gland"
# label), so WHOLE_GLAND is defined as the union of the zonal + lesion
# labels rather than a label of its own.
BACKGROUND = 0
PERIPHERAL_ZONE = 2
TRANSITION_ZONE = 3
LESION = 4
WHOLE_GLAND = frozenset({PERIPHERAL_ZONE, TRANSITION_ZONE, LESION})


class OrganMask(Transform):
    """Zero out image voxels outside the requested anatomical label(s).

    Parameters
    ----------
    keep_labels:
        Iterable of mask label values to keep (default: :data:`WHOLE_GLAND`).
    fill_value:
        Value assigned to voxels outside ``keep_labels``.
    """

    def __init__(self, keep_labels: Iterable[int] | None = None, fill_value: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.keep_labels = frozenset(keep_labels) if keep_labels is not None else WHOLE_GLAND
        self.fill_value = fill_value

    def apply(self, sample: Sample) -> Sample:
        if sample.mask is None:
            return sample

        keep = np.isin(sample.mask, list(self.keep_labels))
        outside = ~keep
        sample.image = np.where(outside, self.fill_value, sample.image)
        sample.image_dirty = True

        sample.log(AuditEntry(
            step="organ_mask", action="masked_outside_labels",
            detail=f"kept_labels={sorted(self.keep_labels)}, voxels_zeroed={int(outside.sum())}",
        ))
        return sample
