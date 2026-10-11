# v2 change map: briefing items → code → tests

Applies to version 0.2.1 (see CHANGELOG.md).

This maps each recommendation in the two October 2026 briefings to what v2
does about it.

* Briefing 1: *DICOM, NIfTI and ProST-Annomyze*, §A3 / §B7.
* Briefing 2: *De-identification of Prostate MRI*, §5.3.

Status key:
* ✔ done and tested;
* ◐ partly done (what is missing is stated);
* ✘ not done, with the reason.

## Burned-in text (user requirement 1)

| Item | Status | Where | Test |
|---|---|---|---|
| Remove text but keep the image | ✔ | `text_restore.py`: glyph segmentation + restoration; only glyph pixels change | `test_restore_keeps_every_unmasked_pixel_bit_identical` |
| Segmentation by region growing | ✔ | `segment_text_pixels`: stroke-width-scaled top-hat, seeds, bounded geodesic growth | `test_segmentation_finds_glyph_pixels` |
| ViT | ✔ | `vit_mae_poisson`: ViT-MAE (facebook/vit-mae-base) rebuilds the text patches, blended in the gradient domain | `test_vit_mae_poisson_restores_without_ghost` (opt-in) |
| Analyse first, then apply | ✔ | `benchmarks/text_removal_benchmark.py`, `docs/v2_analysis.md` | 24-case benchmark, v1 vs v2 |
| Do not miss text on anatomy | ✔ | `detect_overlay_text_boxes` (OCR-confirmed stroke lines) on top of v1's detector | `test_overlay_text_on_anatomy_found_where_v1_misses` |
| Never change tissue that only looks like text (fat rim / septa) | ✔ | glyph-component test, per-cluster text evidence, band splitting (0.2.1) | `test_bright_fat_rim_and_septa_are_not_treated_as_text` |
| JPEG screenshots restored, not blacked out | ✔ | halo-aware growth, mask-local verification (0.2.1) | `test_jpeg_screenshot_text_restored_not_blacked_out` |
| Never ship leftover text | ✔ | re-OCR + ghost-ratio check → fallback to blackout, flagged | `test_failed_restoration_falls_back_to_blackout`, `test_ghost_ratio_flags_offset_fill` |
| v1 behaviour available unchanged | ✔ | `method="blackout"`; `redact_burned_in_text` gives identical output | `test_blackout_matches_v1_redaction_exactly`, benchmark `identical_to_v1` |
| Library `BurnedInTextRedact` works (was `NotImplementedError`) | ✔ | `burned_in_text.py` delegates to `text_restore`; spaCy no longer needed | `test_detection_redacts_real_burned_in_text` |
| Local VLM second check (briefing 2 §5.3) | ✘ | not added; the re-OCR + ghost check is the second check. A local VLM is the next candidate | – |

## DICOM conversion (briefing 1 §A3)

| Item | Status | Where | Test |
|---|---|---|---|
| NIfTI axis order (Z, Y, X); no transposed slices | ✔ | `convert.nifti_to_dicom_series`, `universal_io.load_any` | `test_slice_k_matches_nifti_k`, `test_load_any_nifti_axis_order` |
| Geometry from the affine (PixelSpacing, SliceThickness, IPP, IOP; RAS→LPS) | ✔ | `_affine_to_dicom_geometry` | `test_nifti_roundtrip_oblique_nonsquare_is_exact` |
| One file per slice in a shared series / frame of reference | ✔ | `nifti_to_dicom_series` | `test_nifti_output_conformance` |
| Type 2 attributes present; MF Grayscale Word SC class for 16-bit volumes | ✔ | `_fill_common_type2`, `universal_io.array_to_dicom` | `test_array_to_dicom_volume_class_and_values` |
| Validate every output | ◐ | `check_conformance` self-check (Type 1/2, pixel length, transfer syntax) runs on every output in the dashboard; no dciodvfy / highdicom in CI yet | conformance asserted in tests |
| Compressed DICOM decoded; output is Explicit VR LE | ✔ | `convert.ensure_uncompressed` (+ `[codecs]` extra) | `test_compressed_dicom_is_decoded_and_saved_uncompressed`, `test_full_wipe_compressed_saves` |
| Folder / ZIP / series input; consistent pseudonyms | ✔ | dashboard multi-file + ZIP; `StudyContext` (keyed UIDs incl. FrameOfReferenceUID, PatientID, date shift) | `test_series_linkage_and_date_intervals`, `test_pseudonyms_are_keyed` |
| Preserve intensities (no min–max rescale) | ✔ | `_choose_integer_storage`; NIfTI scl_slope → RescaleSlope/Intercept | `test_integer_valued_float_adc_is_bit_identical`, `test_nifti_scl_slope_carried_exactly` |
| Keep safe MR acquisition tags, or relabel as SC | ✔ | `mr_tags.MR_SAFE_KEEP`; strict mode relabels to Secondary Capture | `test_deidentify_removes_phi_keeps_mr_utility`, `test_v1_policy_still_available_and_conformant` |
| Converter emits AuditEntry | ✔ | every step in `convert.py`; dashboard table + `audit_trail.json` in the ZIP | throughout |
| Palette PNG, grey+alpha PNG, EXIF orientation | ✔ | `convert._raster_to_array` | `test_palette_png_becomes_colours`, `test_gray_alpha_png_matches_rows_columns`, `test_jpeg_exif_not_carried_over` |
| nibabel / Pillow declared; README / CHANGELOG; banner text; `.nii.gz`-only gz | ✔ | `pyproject.toml`, `README.md`, `CHANGELOG.md`, `dashboard/convert_app.py` | – |
| Tests for the converter | ✔ | `tests/test_convert.py`, `test_universal_io.py`, `test_mr_tags.py`, `test_text_restore.py` | 89 passed (with `test_version.py`) |

## DICOM → NIfTI (briefing 1 §B7)

| Item | Status | Where | Test |
|---|---|---|---|
| One writer, `.nii.gz`, full orientation-aware affine | ✔ | `convert.series_to_nifti`; `universal_io.dicom_to_nifti_bytes` and `nifti_io.save_nifti` share the same geometry | `test_dicom_to_nifti_bytes_spacing_not_swapped`, `test_nifti_io_save_nifti_matches_converter_affine` |
| In-plane spacing no longer swapped | ✔ | as above | as above |
| Orientation self-check against source geometry | ✔ | `orientation_check_max_error_mm` in the sidecar + audit | round-trip tests (< 0.001 mm) |
| Multi-b-value DWI → 4D + b-values; spacing-uniformity and mixed-orientation checks | ✔ | `series_to_nifti` | `test_multi_bvalue_series_becomes_4d_with_bvals`, `test_mixed_orientation_series_is_rejected` |
| RARN normalization before export | ✔ | `series_to_nifti(normalize=IntensityNormalize())`; dashboard toggle | – |
| Merge annolib `nifti_io` with ProST-Annomyze `universal_io` | ◐ | `nifti_io.py` (annolib API) is now in this package on the shared writer. annolib's other modules (e.g. `profiles.py`) already have equivalents here; the repositories themselves are not merged | – |
| PHI-safe output file names | ✔ | `convert.output_filename` (pseudonym + series/instance) | `test_output_filename_never_uses_upload_name` |
| FSL bvec files | ✘ | b-values go in the JSON sidecar; gradient vectors are not converted to FSL's voxel convention, to avoid a silent sign error. Use dcm2niix for full DTI | – |
| Enhanced multi-frame MR geometry | ◐ | functional-group sequences are kept (recursively allowlisted) and read by `series_to_nifti`; not yet tested on a real enhanced-MR file | – |

## Prostate mpMRI (briefing 2 §5.3)

| Item | Status | Where | Test |
|---|---|---|---|
| Rescue Siemens / GE (incl. +1e9) / Philips b-values into public tags | ✔ | `mr_tags.rescue_diffusion_tags` | `test_siemens_bvalue_rescued_to_public_tags`, `test_ge_bvalue_rescued_with_1e9_offset_removed` |
| Sequence-aware keep list; SeriesDescription / ProtocolName after scrubbing | ✔ | `MR_SAFE_KEEP`, `scrub_free_text` (names from the file's own PN fields, PN patterns, titles, dates, long IDs) | `test_free_text_scrub_keeps_sequence_label`, `test_free_text_scrub_removes_ids_and_dates` |
| NER-based free-text scrubbing | ✘ | rule-based only; every scrubbed field is flagged for review | – |
| Study-level pseudonymization (one study, one frame of reference, intervals kept) | ✔ | `StudyContext` | `test_series_linkage_and_date_intervals` |
| ADC values bit-identical | ✔ | integer path | `test_integer_valued_float_adc_is_bit_identical` |
| Ages > 89 aggregated | ✔ | `deidentify_dicom` | `test_age_over_89_aggregated` |
| MIDI-B / PI-CAI utility validation | ✘ | needs those datasets; the benchmark harness is the template | – |
