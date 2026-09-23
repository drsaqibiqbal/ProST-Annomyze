"""
One-call convenience API on top of the Transform/Compose building blocks.

Most callers don't need to hand-assemble a ``Compose([...])`` pipeline --
they need "de-identify this file" with sane, documented defaults. This
module is that entry point, plus the ordering policy that the lower-level
transforms don't (and shouldn't) enforce themselves.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

from .burned_in_text import BurnedInTextRedact
from .core import AuditEntry, Compose
from .dicom_scrub import DicomTagScrub
from .pseudonymize import MappingStore, Pseudonymize

__all__ = ["default_pipeline", "anonymize_dicom", "summarize_audit"]

# Fields Pseudonymize remaps; DicomTagScrub must not delete these outright,
# or it silently undoes the remap. This was a real bug found during
# development (see tests/test_pipeline.py::test_ordering_regression) and is
# the reason Pseudonymize -> DicomTagScrub ordering is baked into
# default_pipeline rather than left to caller discretion.
_PSEUDONYMIZED_FIELDS = [
    "PatientID",
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
]


def default_pipeline(
    mapping_store: str | os.PathLike | MappingStore,
    profile: str = "hipaa_safe_harbor",
    redact_burned_in_text: bool = True,
    require_ocr: bool = False,
    secret: str | None = None,
) -> Compose:
    """Build the recommended pipeline with correct step ordering.

    Order (must not be changed without also updating ``exclude_tags``):
      1. :class:`~prostate_deid.pseudonymize.Pseudonymize` -- remaps
         PatientID, shifts dates, regenerates UIDs.
      2. :class:`~prostate_deid.dicom_scrub.DicomTagScrub` -- removes the
         rest of the HIPAA-Safe-Harbor identifier set and all private
         tags, explicitly excluding the fields step 1 already handled.
      3. :class:`~prostate_deid.burned_in_text.BurnedInTextRedact` --
         pixel-level PHI + overlay planes, degrading gracefully if no OCR
         backend is installed.

    Note this pipeline does not include
    :class:`~prostate_deid.organ_mask.OrganMask` or
    :class:`~prostate_deid.preprocess.IntensityNormalize`: those are
    task-specific preprocessing steps a caller adds explicitly (see
    README quickstart), not part of the de-identification default, since
    not every de-identification use case wants the image masked or
    renormalized.
    """
    return Compose([
        Pseudonymize(mapping_store=mapping_store, secret=secret, always_apply=True),
        DicomTagScrub(
            profile=profile,
            exclude_tags=_PSEUDONYMIZED_FIELDS,
            strip_private_tags=True,
            always_apply=True,
        ),
        BurnedInTextRedact(
            require_ocr=require_ocr,
            always_apply=redact_burned_in_text,
            p=1.0 if redact_burned_in_text else 0.0,
        ),
    ])


def anonymize_dicom(
    in_path: str | os.PathLike,
    out_path: str | os.PathLike,
    mapping_store: str | os.PathLike | MappingStore,
    profile: str = "hipaa_safe_harbor",
    redact_burned_in_text: bool = True,
    require_ocr: bool = False,
    secret: str | None = None,
) -> list[AuditEntry]:
    """Load, de-identify, and save one DICOM file. Returns its audit trail.

    Thin wrapper around :func:`default_pipeline` + the ``dicom_io`` I/O
    functions, for the common single-file case. For batch/cohort
    orchestration, call ``default_pipeline`` once and reuse it across many
    ``load_sample``/``save_sample`` pairs (cheaper than rebuilding the
    pipeline, and required for the ``MappingStore`` to accumulate a
    consistent per-patient mapping across a whole study).
    """
    from .dicom_io import (  # local import: keep pydicom optional at package import time
        load_sample,
        save_sample,
    )

    sample = load_sample(in_path)
    pipeline = default_pipeline(
        mapping_store=mapping_store,
        profile=profile,
        redact_burned_in_text=redact_burned_in_text,
        require_ocr=require_ocr,
        secret=secret,
    )
    result = pipeline(sample)
    save_sample(result, out_path)
    return result.audit


def summarize_audit(audit: list[AuditEntry]) -> dict[str, Any]:
    """Aggregate a flat audit trail into headline counts for reporting/dashboards.

    Returns counts by ``(step, action)`` plus a total flagged-for-review
    count, which is the number a reviewer dashboard or a compliance report
    leads with.
    """
    by_step_action = Counter((e.step, e.action) for e in audit)
    return {
        "total_entries": len(audit),
        "flagged_for_review": sum(1 for e in audit if e.flagged_for_review),
        "by_step_action": dict(by_step_action),
        "tags_removed": sum(1 for e in audit if e.action == "removed_tag"),
        "private_tags_removed": sum(
            1 for e in audit if e.action == "removed_tag" and "private_tag_policy" in e.detail
        ),
        "ids_pseudonymized": sum(1 for e in audit if e.action == "remapped_patient_id"),
        "dates_shifted": sum(1 for e in audit if e.action == "shifted_date"),
        "uids_regenerated": sum(1 for e in audit if e.action == "regenerated_uid"),
        "pixel_regions_redacted": sum(1 for e in audit if e.action == "redacted_region"),
        "overlay_planes_scrubbed": sum(1 for e in audit if e.action == "scrubbed_overlay_plane"),
    }
