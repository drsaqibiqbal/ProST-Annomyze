# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Quantitative validation against MIDI-B / TCIA Pseudo-PHI-DICOM-Data
  (header de-identification) and PROSTATEx (zonal/lesion masking Dice) —
  see `docs/technical_details.md` Section 9.
- First-party DICOM → NIfTI export helper.
- Batch/cohort orchestration and DICOMweb ingestion.
- Compressed transfer-syntax support in `dicom_io.save_sample` pixel
  write-back (currently assumes uncompressed pixel data).

## [0.1.0] - 2026-09-20

### Added
- Core `Sample` / `AuditEntry` / `Transform` / `Compose` primitives.
- `DicomTagScrub`: `hipaa_safe_harbor` and `dicom_ps3.15_basic` profiles;
  private (odd-group) tag stripping on by default.
- `Pseudonymize`: HMAC-consistent PatientID remap, per-patient date
  shifting (bounded to ±20 years), deterministic UID regeneration under
  the `2.25` root, JSON-backed `MappingStore`.
- `BurnedInTextRedact`: OCR/NER hook with overlay-plane scrubbing and
  graceful degradation (`require_ocr=False` default) when the optional
  `[ocr]` backend is unavailable.
- `OrganMask`: whole-gland / zonal / lesion masking (`WHOLE_GLAND` =
  {peripheral zone, transition zone, lesion}).
- `IntensityNormalize`: Robust Anatomy-Referenced Normalization (RARN) —
  `robust` / `zscore` / `minmax` modes, optional `zone_aware` mode.
- `dicom_io.load_sample` / `save_sample`: real pydicom-backed I/O.
- `pipeline.default_pipeline` / `anonymize_dicom` / `summarize_audit`:
  one-call API with the correct Pseudonymize → DicomTagScrub ordering.
- `export.export_review_bundle` + Streamlit `dashboard/anonymize_app.py`
  and `dashboard/review_app.py`.
- `comparison.py`: single-source capability comparison table.
- Packaging: `pyproject.toml` (hatchling), MIT `LICENSE`, `CITATION.cff`,
  `.zenodo.json`, CI/release GitHub Actions workflows.
