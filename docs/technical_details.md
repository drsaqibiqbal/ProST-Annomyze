================================================================================
prostate-deid — TECHNICAL DETAILS FOR THE SoftwareX MANUSCRIPT
Repository: https://github.com/drsaqibiqbal/annolib
Package name (PyPI/import): prostate-deid / prostate_deid
Version at time of writing: 0.1.0
Prepared: 2026-07-10
================================================================================

READ THIS FIRST — SCIENTIFIC-INTEGRITY NOTE
--------------------------------------------------------------------------------
This document contains NO fabricated performance numbers. The comparison with
other tools (Section 8) is a CAPABILITY (feature-support) comparison drawn from
each tool's public documentation, not a measured accuracy/recall benchmark.

Any quantitative claim of the form "prostate-deid achieves X% PHI-detection
recall" or "outperforms tool Y by Z%" MUST NOT be written into the manuscript
until you have actually run all compared tools on a shared, ground-truthed
evaluation set (see Section 9). Inventing such numbers would be research
misconduct and, for a privacy tool, would also be dangerous: it would imply a
safety guarantee that has not been established. The best current published
DICOM de-identification systems do not reach 100% on the MIDI-B benchmark, and
the manuscript must not imply otherwise.


================================================================================
1. WHAT THE SOFTWARE IS
================================================================================
prostate-deid is a Python library and dashboard that de-identifies prostate
MRI DICOM studies and (optionally) applies prostate-specific anatomical
masking, while producing one machine-readable audit trail of every change.

It is scoped deliberately to prostate MRI. Because prostate MRI is a pelvic
acquisition, there is NO facial anatomy to remove, so — unlike general
medical-image de-identification frameworks — the software does not implement
defacing. This narrower scope lets the anatomical-masking component (gland /
peripheral zone / transition zone / lesion) be a first-class, well-specified
feature rather than an afterthought.

Primary use cases:
  (i)  De-identify prostate MRI DICOM studies before sharing/archiving, while
       preserving per-patient longitudinal linkage (pseudonymization).
  (ii) Apply zonal/lesion masking as part of a segmentation or detection
       pipeline, sharing the same data representation as the de-id step.


================================================================================
2. ARCHITECTURE
================================================================================
The API follows the Transform / Compose convention established by volumetric
augmentation libraries (Albumentations, Bio-Volumentations), so users familiar
with that pattern adopt it directly. The key architectural difference: every
Transform appends a structured record to a shared audit trail instead of
mutating data silently.

Core primitives (src/prostate_deid/core.py):
  - Sample:      carries image (NumPy CZYX), dicom_meta (dict), mask (NumPy),
                 audit (list of AuditEntry), source_dataset (the originating
                 pydicom Dataset when loaded from a real .dcm), image_dirty.
  - AuditEntry:  (step, action, detail, confidence, flagged_for_review).
  - Transform:   abstract base; subclasses implement apply(sample)->sample.
  - Compose:     chains Transforms, threading one Sample through all of them.

Real DICOM I/O (src/prostate_deid/dicom_io.py) — the bridge to actual files:
  - load_sample(path): pydicom.dcmread -> Sample. Exposes header identifiers
    as {keyword: value} in dicom_meta, surfaces PRIVATE elements under their
    "(GGGG,EEEE)" tag string, decodes pixel data to a CZYX float array, and
    keeps the full Dataset on sample.source_dataset.
  - save_sample(sample, path): reconciles Transform changes back into the real
    Dataset (deletes removed keywords/private tags, writes back pseudonymized
    values, re-encodes pixels only if image_dirty), keeps file-meta SOP UID in
    sync, sets PatientIdentityRemoved=YES and DeidentificationMethod, saves.

One-call API (src/prostate_deid/pipeline.py):
  - anonymize_dicom(in_path, out_path, ...) -> audit trail.
  - default_pipeline(...) -> the recommended Compose pipeline.
  - summarize_audit(audit) -> headline counts for reporting/dashboards.


================================================================================
3. THE FOUR TRANSFORM FAMILIES
================================================================================
3.1 DicomTagScrub (src/prostate_deid/dicom_scrub.py)
    Deterministic header removal by named profile:
      - "hipaa_safe_harbor": maps the HIPAA identifier categories onto DICOM
        header fields (name, ID, dates, physician, institution, station, etc.).
      - "dicom_ps3.15_basic": a minimal subset of the DICOM PS3.15 Basic
        Confidentiality Profile.
    PRIVATE-TAG POLICY (default ON): every private (odd-group) element is
    stripped, because standard profiles only cover the PUBLIC DICOM dictionary
    and vendors routinely hide PHI in undocumented private tags. Private tags
    are detected authoritatively by odd group number (DICOM PS3.5), not by a
    keyword-dictionary guess.
    exclude_tags: fields already handled by Pseudonymize are excluded so the
    scrub does not delete the pseudonymized values (see ordering note below).

3.2 Pseudonymize (src/prostate_deid/pseudonymize.py)
    THIS IS PSEUDONYMIZATION, NOT ANONYMIZATION (see Section 6).
      - Patient ID  -> consistent pseudo-ID via HMAC-SHA256(secret, id).
      - Date fields -> shifted by a consistent per-patient offset (bounded to
        +/- 20 years) so relative timing between a patient's scans is
        preserved while absolute dates are not.
      - UIDs (Study/Series/SOP Instance UID) -> deterministically regenerated
        under the 2.25 root as 2.25.<128-bit int from HMAC(secret, uid)>,
        capped at 64 chars. Same source UID -> same replacement, so
        intra-study cross-references survive; original institution/device UID
        roots are not leaked.
      - Mapping table persisted to a MappingStore (a JSON file) whose location
        the caller chooses and MUST keep separate from the de-identified data.

3.3 BurnedInTextRedact (src/prostate_deid/burned_in_text.py)
    OCR (pytesseract) detects text in pixel data; candidate spans are
    classified as PHI by (a) exact/cross-reference match against this study's
    own header identifiers — high confidence — or (b) spaCy NER PERSON/ORG/
    GPE/DATE entities — medium confidence. Detected regions are blacked out and
    the image marked dirty for write-back. Also scans DICOM overlay planes,
    which are a separate bitmap layer easily forgotten.
    ROBUSTNESS: OCR/NER are OPTIONAL extras. If the tesseract binary or the
    extras are absent, the step DOES NOT crash the pipeline (require_ocr=False
    default): it records a flagged "ocr_unavailable" audit entry so the gap is
    visible and the operator can review pixels manually. It also honours the
    DICOM BurnedInAnnotation header flag, flagging a YES-but-no-OCR-hit case.

3.4 OrganMask (src/prostate_deid/organ_mask.py)
    Deterministic prostate masking. Mask label convention (uint8):
      0 = background, 2 = peripheral zone, 3 = transition zone, 4 = lesion.
    WHOLE_GLAND is shorthand for the union {2,3,4} (real segmentations label
    zones exclusively, not with a redundant outer "gland" label). Voxels
    outside keep_labels are set to fill_value; one audit entry records the
    kept labels and the voxel count zeroed.


================================================================================
4. RECOMMENDED PIPELINE AND ORDERING (IMPORTANT)
================================================================================
    from prostate_deid import anonymize_dicom
    audit = anonymize_dicom("in.dcm", "out.dcm",
                            mapping_store="/secure/institution-only/map.json",
                            profile="hipaa_safe_harbor",
                            redact_burned_in_text=True)

Equivalent explicit pipeline:
    Compose([
        Pseudonymize(mapping_store=..., always_apply=True),
        DicomTagScrub(profile="hipaa_safe_harbor",
                      exclude_tags=[PatientID, StudyDate, SeriesDate,
                                    AcquisitionDate, ContentDate,
                                    StudyInstanceUID, SeriesInstanceUID,
                                    SOPInstanceUID],
                      strip_private_tags=True, always_apply=True),
        BurnedInTextRedact(require_ocr=False, always_apply=True),
    ])

ORDERING RULE: Pseudonymize MUST run before DicomTagScrub, and DicomTagScrub
must EXCLUDE the fields Pseudonymize remapped. Otherwise the HIPAA profile
deletes PatientID / dates / UIDs outright and silently undoes the remap. This
was a real bug found and fixed during development; it is covered by a
regression test (tests/test_pipeline.py).


================================================================================
5. THE AUDIT TRAIL (THE DISTINGUISHING FEATURE)
================================================================================
After a run, sample.audit is a flat, ordered list of AuditEntry records — one
per change — across ALL steps. summarize_audit() aggregates it to headline
counts (tags removed, private tags removed, IDs pseudonymized, dates shifted,
UIDs regenerated, pixel regions redacted, flagged-for-review).

Asymmetric-error policy: a MISSED PHI detection (false negative) is a privacy
incident; an OVER-redaction (false positive) is a data-quality loss. These are
not equally bad, so probabilistic steps bias toward flagging/redacting under
uncertainty and record flagged_for_review=True rather than resolving ambiguity
silently. The reviewer dashboard consumes these flags (Section 7); a flag with
no reviewer is worse than no flag, because it manufactures false confidence.


================================================================================
6. ANONYMIZATION vs PSEUDONYMIZATION (LEGAL, NOT COSMETIC)
================================================================================
The software performs PSEUDONYMIZATION by default. A reversible mapping
(patient-ID -> pseudo-ID, plus the date-shift offset) is retained ON PURPOSE so
that longitudinal research (PSA follow-up, repeat MRI over a treatment course)
remains possible.

Under GDPR (Art. 4(5), recital 26), retaining ANY re-identifying mapping — even
institution-side, even encrypted — makes the output pseudonymized data, a
legally distinct category from anonymized data, with its own obligations. The
manuscript and README must state this explicitly and must NOT claim the output
is anonymized. Key-management responsibilities (set PROSTATE_DEID_KEY; store
the mapping separately) are documented in the code and README.


================================================================================
7. DASHBOARDS
================================================================================
7.1 Anonymization dashboard (dashboard/anonymize_app.py) — Streamlit
    Upload a .dcm -> runs default_pipeline -> shows: live metric tiles;
    before/after header table (identifier fields and private tags marked, each
    row tagged removed/changed/kept); before/after mid-slice image; full audit
    trail; the capability comparison; and a download button for the
    de-identified .dcm. Runtime is measured and displayed per run. Verified to
    boot headless (HTTP 200) and to execute end-to-end via Streamlit's AppTest
    harness with no uncaught exceptions.

7.2 Reviewer QA dashboard (dashboard/review_app.py) — Streamlit
    Loads an exported audit bundle (prostate_deid.export.export_review_bundle)
    and surfaces every flagged_for_review entry as a cropped thumbnail for
    human triage, with a disposition control per item.

Run:  pip install prostate-deid[dashboard]
      streamlit run dashboard/anonymize_app.py


================================================================================
8. CAPABILITY COMPARISON  (feature support — NOT a performance benchmark)
================================================================================
Generated from src/prostate_deid/comparison.py. Legend: ✓ full, ~ partial,
– not addressed. Re-verify every cell against each tool's current docs before
submission.

| Capability | Presidio Image Redactor | deid (Vanderbilt / pydicom) | CTP DicomAnonymizer | prostate-deid |
|---|---|---|---|---|
| DICOM metadata scrubbing (profile-based)  | –  | ✓ | ✓ | ✓ |
| Private (vendor) tag stripping            | –  | ~ | ✓ | ✓ |
| Consistent ID pseudonymization            | –  | ~ | ✓ | ✓ |
| Consistent date shifting                  | –  | ~ | ✓ | ✓ |
| Consistent UID regeneration               | –  | ~ | ✓ | ✓ |
| Burned-in pixel text redaction (OCR+NER)  | ✓  | – | – | ✓ |
| Overlay-plane scrubbing                   | –  | – | ~ | ✓ |
| Prostate zonal / lesion masking           | –  | – | – | ✓ |
| Per-action confidence scoring             | ~  | – | – | ✓ |
| Unified audit log across all steps        | –  | ~ | ~ | ✓ |
| Human-review dashboard for flagged items  | –  | – | – | ✓ |
| Single Compose-style pipeline API         | –  | – | – | ✓ |

The defensible novelty claim for the paper: prostate-deid is (to the authors'
knowledge) the first tool to unify profile-based header scrubbing, consistent
pseudonymization + UID regeneration, burned-in/overlay pixel PHI redaction,
AND prostate zonal/lesion masking behind ONE Compose-style API with ONE shared
audit trail and a human-review surface. This is a capability/integration claim,
which the table supports; it is NOT an accuracy claim.


================================================================================
9. WHAT IS NEEDED FOR REAL QUANTITATIVE RESULTS (the evidentiary core)
================================================================================
To turn the manuscript's benchmark section from [TODO] into real evidence:

  A. Header de-identification correctness
     - Dataset: TCIA Pseudo-PHI-DICOM-Data (synthetic PHI inserted into real
       images; openly licensed; safe for CI and publication).
     - Metric: fraction of required de-identification ACTIONS executed
       correctly, reported the way MIDI-B reports it. Compare against deid
       and CTP on the SAME files.
     - Do NOT report on your own private held-out set as if it were a standard.

  B. Burned-in text redaction recall/precision
     - Needs images with known burned-in PHI locations (ground truth boxes).
       Report precision AND recall; recall is the safety-critical number.
     - Compare against Presidio Image Redactor on the same images.

  C. Zonal / lesion masking agreement
     - Dataset: PROSTATEx (TCIA; requires a data usage agreement in the
       author's / institution's name — the authors must obtain this; it cannot
       be obtained on their behalf).
     - Metric: Dice vs. a reference segmentation.

  D. Runtime
     - Wall-clock per pipeline stage, mean ± SD over N runs, on a stated
       CPU/GPU configuration (Bio-Volumentations Tables 3–4 style).

Until A–D exist, keep the benchmark subsection explicitly marked as pending and
do not let the abstract imply measured superiority.


================================================================================
10. TESTING, CI, PACKAGING
================================================================================
- Tests (tests/): synthetic fixtures ONLY — no real patient data, ever. Covers
  each transform, the DICOM round-trip on a real synthetic .dcm, UID-regen
  consistency, graceful OCR degradation, and the end-to-end pipeline/ordering.
  30 passing, 1 skipped (the skip is the OCR path when tesseract is absent).
- Lint: ruff clean. CI: GitHub Actions matrix (Python 3.9–3.12).
- Packaging: pyproject.toml (hatchling), MIT license, CITATION.cff,
  .zenodo.json. Optional extras: [ocr], [dashboard], [docs], [dev].
- Docs: MkDocs site (builds with --strict), plus README.


================================================================================
11. LIMITATIONS / FUTURE WORK (for the manuscript)
================================================================================
- No quantitative validation yet (Section 9) — the single most important gap.
- Burned-in redaction quality is bounded by the OCR/NER backends; recall on
  names dissimilar to NER training data is a documented weak point (mitigated,
  not solved, by header cross-referencing).
- Pixel write-back currently assumes uncompressed pixel data reshaped to the
  original geometry; compressed transfer syntaxes need explicit handling.
- Masking assumes an externally provided segmentation mask; the library does
  not itself segment the gland/zones/lesion.
- Single-file oriented; batch/cohort orchestration and DICOMweb ingestion are
  future work.


================================================================================
12. COMPLIANCE DISCLAIMER (must appear in the paper and README)
================================================================================
Running this pipeline does not by itself constitute HIPAA or GDPR compliance.
It is a tool toward compliance, not a certification of it. Actual compliance
requires the institution's own review (e.g. HIPAA Expert Determination, or a
GDPR DPIA). Provided under the MIT License, which disclaims all warranty,
including any implied claim of regulatory compliance.
================================================================================
