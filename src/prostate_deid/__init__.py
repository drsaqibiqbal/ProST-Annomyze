"""
prostate-deid: de-identification and anatomical masking for prostate MRI,
behind a single Compose-style API with one shared, auditable trail.

Quickstart
----------
>>> from prostate_deid import anonymize_dicom
>>> audit = anonymize_dicom(
...     "study.dcm", "study_deid.dcm",
...     mapping_store="/secure/institution-only/mapping.json",
...     profile="hipaa_safe_harbor",
... )

See the README and ``docs/`` for the full pipeline-composition API,
the Robust Anatomy-Referenced Normalization (RARN) preprocessing stage,
and the compliance disclaimer (pseudonymization != anonymization; running
this library does not by itself constitute HIPAA or GDPR compliance).
"""

from .burned_in_text import BurnedInTextRedact
from .core import AuditEntry, Compose, Sample, Transform
from .dicom_scrub import DicomTagScrub
from .organ_mask import LESION, PERIPHERAL_ZONE, TRANSITION_ZONE, WHOLE_GLAND, OrganMask
from .pipeline import anonymize_dicom, default_pipeline, summarize_audit
from .preprocess import IntensityNormalize
from .pseudonymize import MappingStore, Pseudonymize

try:
    from .dicom_io import load_sample, save_sample
except ImportError:  # pragma: no cover - pydicom should always be present (core dependency)
    pass

try:
    from .universal_io import (
        array_to_dicom,
        detect_text_boxes,
        dicom_to_nifti_bytes,
        full_wipe_dicom,
        load_any,
        redact_burned_in_text,
    )
except ImportError:  # pragma: no cover - pydicom should always be present (core dependency)
    pass

# v2: image-preserving text removal and the MR-aware, geometry-preserving converter
from ._version import __version__
from .convert import (
    StudyContext,
    check_conformance,
    deidentify_dicom,
    nifti_to_dicom_series,
    raster_to_dicom,
    series_to_nifti,
)
from .nifti_io import save_nifti
from .text_restore import RESTORE_METHODS, detect_text_boxes_v2, restore_burned_in_text

__all__ = [
    "AuditEntry", "Sample", "Transform", "Compose",
    "DicomTagScrub",
    "Pseudonymize", "MappingStore",
    "BurnedInTextRedact",
    "OrganMask", "WHOLE_GLAND", "PERIPHERAL_ZONE", "TRANSITION_ZONE", "LESION",
    "IntensityNormalize",
    "default_pipeline", "anonymize_dicom", "summarize_audit",
    "load_sample", "save_sample",
    "load_any", "array_to_dicom", "full_wipe_dicom", "dicom_to_nifti_bytes", "redact_burned_in_text",
    "detect_text_boxes",
    "restore_burned_in_text", "detect_text_boxes_v2", "RESTORE_METHODS",
    "StudyContext", "deidentify_dicom", "nifti_to_dicom_series", "raster_to_dicom", "series_to_nifti",
    "check_conformance", "save_nifti",
    "__version__",
]
