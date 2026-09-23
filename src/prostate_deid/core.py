"""
Core primitives shared by every transform in prostate-deid.

Design contract (mirrors the Transform/Compose pattern used by volumetric
augmentation libraries such as Albumentations and Bio-Volumentations):

    - `Sample` is the unit of data that flows through a pipeline.
    - `Transform` subclasses implement `apply(sample) -> sample`.
    - `Compose` chains `Transform`s and threads ONE `Sample` (and therefore
      one shared audit trail) through all of them.

The one deliberate departure from a generic augmentation library: every
`Transform` that changes the sample must append a structured `AuditEntry`
instead of mutating data silently. That is what turns a pipeline run into a
compliance-ready report rather than an opaque one-way transform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["AuditEntry", "Sample", "Transform", "Compose"]


# ---------------------------------------------------------------------------
# Audit trail primitives
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """One record of exactly one change made by one Transform.

    Attributes
    ----------
    step:
        Name of the transform family, e.g. ``"dicom_tag_scrub"``,
        ``"pseudonymize"``, ``"burned_in_phi"``, ``"organ_mask"``,
        ``"intensity_normalize"``.
    action:
        Short verb describing what happened, e.g. ``"removed_tag"``,
        ``"redacted_region"``, ``"masked_outside_gland"``.
    detail:
        Human-readable detail (tag name, bbox, voxel count, ...). Must never
        contain the PHI value itself -- callers are responsible for hashing
        or otherwise redacting free-text detail before it is logged.
    confidence:
        ``None`` for deterministic steps (tag removal, masking). A float in
        ``[0, 1]`` for probabilistic steps (burned-in text detection).
    flagged_for_review:
        ``True`` when a human should look at this action before the result
        is trusted. Sampling policy across the library: **a missed PHI
        detection is a privacy incident; an over-redaction is a
        data-quality loss.** These are not symmetric failures, so
        probabilistic steps bias toward flagging under uncertainty rather
        than resolving ambiguity silently.
    """

    step: str
    action: str
    detail: str
    confidence: float | None = None
    flagged_for_review: bool = False

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        conf = f", confidence={self.confidence:.2f}" if self.confidence is not None else ""
        flag = " [FLAGGED]" if self.flagged_for_review else ""
        return f"[{self.step}] {self.action}: {self.detail}{conf}{flag}"


@dataclass
class Sample:
    """The unit that flows through a pipeline.

    Attributes
    ----------
    image:
        Volumetric pixel array, conventionally ``CZYX`` (channel, depth,
        height, width), float32.
    dicom_meta:
        Header identifiers exposed as ``{keyword: value}``. Private
        (vendor-specific) elements are exposed under their
        ``"(GGGG,EEEE)"`` tag string rather than a keyword, since most have
        none.
    mask:
        Optional label array, same spatial shape as ``image`` minus the
        channel axis. Label convention for prostate zonal masks: ``0`` =
        background, ``2`` = peripheral zone, ``3`` = transition zone,
        ``4`` = lesion.
    audit:
        Ordered list of every change made to this sample so far.
    source_dataset:
        The originating ``pydicom.Dataset`` when the sample was loaded from
        a real ``.dcm`` file, kept so that :func:`save_sample` can
        reconcile transform changes back into a real DICOM object rather
        than reconstructing one from scratch.
    image_dirty:
        Set by any transform that rewrites pixel data, so that
        :func:`save_sample` knows whether the pixel data needs re-encoding.
    """

    image: np.ndarray
    dicom_meta: dict[str, Any] | None = None
    mask: np.ndarray | None = None
    audit: list[AuditEntry] = field(default_factory=list)
    source_dataset: Any | None = None
    image_dirty: bool = False

    def log(self, entry: AuditEntry) -> None:
        self.audit.append(entry)


# ---------------------------------------------------------------------------
# Transform / Compose
# ---------------------------------------------------------------------------

class Transform(ABC):
    """Base class for every de-identification / masking / preprocessing step.

    Parameters
    ----------
    p:
        Probability of applying the transform when called. ``1.0`` for
        every de-identification transform used in practice -- de-id steps
        are not meant to be stochastic -- but kept for symmetry with the
        augmentation-library convention this API mirrors, and because a
        caller composing this library's transforms into an augmentation
        pipeline (e.g. alongside Bio-Volumentations) may want the same
        probabilistic-application idiom throughout.
    always_apply:
        Shorthand for ``p=1.0``.
    """

    def __init__(self, p: float = 1.0, always_apply: bool = False):
        self.p = 1.0 if always_apply else p

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        ...

    def __call__(self, sample: Sample) -> Sample:
        if np.random.rand() <= self.p:
            return self.apply(sample)
        return sample

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(p={self.p})"


class Compose:
    """Runs a list of Transforms in sequence over one shared Sample.

    All audit entries produced by every transform accumulate on the same
    ``sample.audit`` list, in execution order, so the result of a
    ``Compose`` call is both the processed sample and a single flat,
    ordered compliance report.
    """

    def __init__(self, transforms: list[Transform]):
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        inner = ",\n    ".join(repr(t) for t in self.transforms)
        return f"Compose([\n    {inner}\n])"
