# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Dashboard section "Quantitative verification": per-region `retained(B) = 1 - |G_B|/|B|` and
  `c_audit(B) = min(1, max(0, n_OCR/100))` on the real text (eq. 40), and a synthetic self-test that
  burns fictitious corner annotations into the de-identified image, removes them with the same
  settings and reports MAE, MSE and PSNR over the text footprint plus residual readable characters
  (eq. 41); CSV export.
- `prostate_deid.evaluation`: `region_metrics`, `summarize_regions`, `footprint_errors`,
  `synthetic_self_test`.

### Planned
- Quantitative validation against MIDI-B / TCIA Pseudo-PHI-DICOM-Data
  (header de-identification) and PROSTATEx / PI-CAI (utility: b-values, ADC,
  co-registration, segmentation Dice before vs after) — see
  `docs/technical_details.md` Section 9.
- Fine-tuning the ViT-MAE restorer on prostate MRI; a local vision-language
  model as a second pixel check; NER for free-text fields.
- dciodvfy / highdicom validation in CI; FSL bvec export; tests on real
  enhanced multi-frame MR.

## [0.3.0] - 2026-10-15

### Fixed (privacy)
- Burned-in text dimmer than the brightest anatomy (bright fat or bladder on T2 / ADC) and text drawn
  over tissue was left on the image. Since 0.2.1 a glyph had to reach within 6% of the slice maximum,
  so most lines of a viewer's corner block failed; the lines of a detector box that a text-line split
  could not confirm were dropped without an audit entry.
- New pixel-only stroke-line detector (`detect_stroke_text_lines`): thin strokes measured against
  their immediate surroundings, accepted as a text line only with typeset regularity (>= 3 glyphs of
  equal height on one baseline). Lines a block's line pitch places between two confirmed lines, and
  low-contrast gaps between confirmed pieces of one row, are filled in (`inferred`, flagged).
- Small exported slices (224-256 px JPEG screenshots, ~6 px glyphs) are upscaled for stroke-line
  detection; lines such as "Images: 25/56" and "Series: 1003" were left on these images.
- Text on the black background around the body is filled with the local background level instead
  of diffusion inpainting, which pulled grey streaks from nearby tissue into the corners.
- Text over speckled bright tissue (ADC maps): a second stroke map keeps only near-extreme stroke
  cores, so letters no longer merge with same-sized tissue specks ("WL: 2048 WW: 4096").
- Lines in the annotation zone take in strokes continuing along their own row (a date line whose
  start runs over bright tissue).
- The row just past the inner end of a corner block is checked for a short overlay word that the
  equal-height rule rejects (mixed case, e.g. "dadc"): overlay-bright glyphs on one baseline,
  starting at the block edge, or OCR >= 3 confident characters.
- Fallback rectangles of the restoration methods are re-synthesised from the pixels around them
  (biharmonic) instead of set to 0 (a black bar on a grey ADC background); an OCR-detector box in the
  border zone that falls back is covered whole. `method="blackout"` is unchanged (identical to v1).
- Detector boxes reported twice by v1's two passes are processed once.

## [0.2.3] - 2026-10-11

### Changed
- Dashboard: the compliance caveat moved from the hero banner to a short note below it, reworded
  ("built to support HIPAA- and GDPR-aligned workflows; compliance also depends on institutional
  policy and human review"). The substance is unchanged and matches the README.
- Dashboard: the version moved from the hero banner to a small sidebar footer (still written into
  every output's DeidentificationMethod and the NIfTI sidecar).

## [0.2.2] - 2026-10-11

### Fixed (privacy)
- **The "Where pixels changed" panel could redraw the removed text.** Its outline (and the
  optional red fill) traced the exact glyph shapes, so the removed text was readable again in red.
  The panel now draws only rectangles around changed areas, merged per text line
  (`text_restore.changed_region_boxes`), and the fill option is gone. Exported files were never
  affected.

## [0.2.1] - 2026-10-11

Fixes found while testing v2 on real pelvic MRI screenshots. Benchmark details are in
`docs/v2_analysis.md` §7.

### Fixed
- **Thin bright tissue was treated as text.**
  - Cause: fat rims and fat septa inside v1 detector boxes were inpainted, or blacked out by the
    fallback.
  - Fix: stroke components must reach the display extreme (or sit on a confirmed text line) and
    match the line's glyph height.
  - Each glyph cluster is restored only with positive text evidence: stroke-crossing density
    ≥ 0.10, or OCR ≥ 3 characters at confidence 60.
  - Boxes that mix text and anatomy are split into text-line bands.
  - Result on pelvis phantoms: tissue pixels changed fell from about 80k to 1,977 (in total over
    9 variants).
- **JPEG screenshots fell back to whole-box blackout.**
  - Cause: the faint ringing halo around glyphs, plus slice-sized "detections" in the verification
    pass.
  - Fix: region growing now covers the halo, and verification only looks where pixels were
    replaced.
  - When a fallback is needed, only the text line's rectangle is blacked out, not the whole
    detector box.
- **Wrong polarity guesses could leave text behind.** Both polarities are now tried before a box
  is given up.
- **The text-evidence test mis-measured dark text.** It is now polarity-aware.
- **Detector boxes with no glyph strokes and no legible OCR were blacked out.** They are now left
  untouched and flagged (`skipped_region_not_text`).

### Changed
- Dashboard: the third panel shows the de-identified image with a thin outline around changed
  pixels; "Fill changed pixels" is optional. The banner shows the package version.
- Version is single-sourced in `src/prostate_deid/_version.py` (`pyproject.toml` reads it through
  hatch; the DeidentificationMethod tag, the NIfTI sidecar and the dashboard use it).
- `CITATION.cff` and `.zenodo.json` updated to 0.2.1 and to the ProST-Annomyze repository.
  `tests/test_version.py` keeps them in sync.

### Benchmark (24 cases, Telea default)

| Metric | 0.2.0 | 0.2.1 |
|---|---|---|
| Text still readable | 5.0% | 4.3% |
| Box pixels kept untouched | 55.7% | 57.3% |
| SSIM vs clean image | 0.77 | 0.79 |
| Fallbacks to blackout | 5 | 1 |

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
