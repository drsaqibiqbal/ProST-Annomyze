# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Quantitative validation against MIDI-B / TCIA Pseudo-PHI-DICOM-Data
  (header de-identification) and PROSTATEx / PI-CAI (utility: b-values, ADC,
  co-registration, segmentation Dice before vs after) — see
  `docs/technical_details.md` Section 9.
- Fine-tuning the ViT-MAE restorer on prostate MRI; a local vision-language
  model as a second pixel check; NER for free-text fields.
- dciodvfy / highdicom validation in CI; FSL bvec export; tests on real
  enhanced multi-frame MR.

## [0.2.0] - 2026-10-11 — ProST-Annomyze v2

### Added
- `text_restore`: removes burned-in text while keeping the image.
  - Segmentation: thin-stroke polarity, stroke-width-scaled top-hat and
    bounded region growing.
  - Restoration methods: `telea` (default), `ns`, `biharmonic`,
    `vit_mae_poisson` (ViT-MAE + gradient-domain blending), and `blackout`
    (v1 behaviour).
  - Overlay-text detector for text drawn on anatomy.
  - Verification by re-OCR and a ghost-ratio silhouette check, with fallback
    to blackout.
  - Per-slice inpaint mask and `AuditEntry` records.
- `convert`:
  - `StudyContext` (keyed study-level pseudonyms, including
    FrameOfReferenceUID; date shift with intervals kept);
  - `deidentify_dicom` (MR-aware allowlist);
  - `nifti_to_dicom_series` (affine → IPP / IOP / PixelSpacing; values
    preserved);
  - `raster_to_dicom`;
  - `series_to_nifti` (full affine, exact scaling, 4D multi-b-value, JSON
    sidecar, orientation self-check);
  - `check_conformance`;
  - PHI-safe `output_filename`.
- `mr_tags`:
  - MR-safe keep list;
  - Siemens / GE / Philips diffusion b-value and gradient rescue into
    (0018,9087) / (0018,9089);
  - free-text scrubbing of SeriesDescription / ProtocolName /
    StudyDescription.
- `nifti_io` (annolib API, `save_nifti` / `dicom_to_nifti`) on the shared
  orientation-aware writer.
- `benchmarks/text_removal_benchmark.py` and `docs/v2_analysis.md`
  (v1 vs v2, 24 cases); `docs/v2_changes.md` (briefing → code → test map).
- Dashboard `convert_app.py` v2:
  - multi-file / ZIP upload and series table;
  - method and header-policy choice;
  - pseudonymization key;
  - original / de-identified / changed-pixel views;
  - audit table;
  - ZIP and NIfTI + sidecar downloads.
- Extras: `[vit]`, `[codecs]`. Dependencies: nibabel, Pillow, scipy,
  scikit-image, opencv-python-headless.
- Tests: `test_text_restore`, `test_convert`, `test_mr_tags`,
  `test_universal_io` (84 passing).

### Fixed (v1 code-review findings, also in the `universal_io` compatibility API)
- NIfTI slices were transposed (loaded as (Z, X, Y)).
- The NIfTI affine was dropped, and in-plane spacing was swapped on export.
- Compressed DICOM crashed on save (native pixels under a JPEG transfer
  syntax).
- Type 2 Patient / Study attributes were missing.
- Volumes used the single-frame SC class.
- Intensities were min–max rescaled.
- Palette PNGs were read as indices, and grey+alpha PNGs had the wrong pixel
  length.
- `BurnedInTextRedact` raised `NotImplementedError`; it now runs real
  detection and needs only Tesseract (no spaCy).
- Strict (technical-only) wipe left an invalid MR Image; it is now relabelled
  as Secondary Capture.

### Changed
- Requires Python ≥ 3.10 (pydicom 3).

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
