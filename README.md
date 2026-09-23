# prostate-deid

[![CI](https://github.com/drsaqibiqbal/annolib/actions/workflows/ci.yml/badge.svg)](https://github.com/drsaqibiqbal/annolib/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/prostate-deid.svg)](https://pypi.org/project/prostate-deid/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/prostate-deid/badge/?version=latest)](https://drsaqibiqbal.github.io/annolib/)

**De-identification and anatomical masking for prostate MRI, behind one
Compose-style pipeline that produces a single, shared, auditable trail of
exactly what was changed.**

DICOM metadata scrubbing, patient ID/date pseudonymization, burned-in pixel
PHI redaction, and prostate zonal/lesion masking are usually four separate
tools with four separate interfaces and no shared record of what happened.
`prostate-deid` unifies them behind the `Transform`/`Compose` API pattern
established by volumetric augmentation libraries (Albumentations,
Bio-Volumentations), so every step — deterministic or probabilistic — logs
into the same audit trail.

## Install

```bash
pip install prostate-deid
# OCR/NER burned-in-text detection is an optional extra:
pip install prostate-deid[ocr]
# Streamlit dashboards:
pip install prostate-deid[dashboard]
```

## Quickstart

```python
from prostate_deid import anonymize_dicom

audit = anonymize_dicom(
    "study.dcm", "study_deid.dcm",
    mapping_store="/secure/institution-only/mapping.json",  # keep this OUT of shared output
    profile="hipaa_safe_harbor",
    redact_burned_in_text=True,
)
for entry in audit:
    print(entry)
```

Composing the pipeline explicitly, and adding gland-only masking plus
Robust Anatomy-Referenced Normalization (RARN) as a downstream preprocessing
step for a segmentation pipeline:

```python
from prostate_deid import (
    Compose, Pseudonymize, DicomTagScrub, BurnedInTextRedact,
    OrganMask, IntensityNormalize, WHOLE_GLAND, load_sample,
)

sample = load_sample("study.dcm")
pipeline = Compose([
    Pseudonymize(mapping_store="/secure/institution-only/mapping.json", always_apply=True),
    DicomTagScrub(
        profile="hipaa_safe_harbor",
        exclude_tags=["PatientID", "StudyDate", "SeriesDate",
                      "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"],
        always_apply=True,
    ),
    BurnedInTextRedact(require_ocr=False, always_apply=True),
    OrganMask(keep_labels=WHOLE_GLAND, always_apply=True),
    IntensityNormalize(method="robust", always_apply=True),  # RARN
])
result = pipeline(sample)
```

> DICOM → NIfTI export (so header PHI does not survive to the exported
> file) is not yet a first-party function in this skeleton — wire
> `result.image`/`result.mask` to your own NIfTI writer (e.g.
> `SimpleITK.WriteImage`) for now; see `CHANGELOG.md` / open issues.

## Why pseudonymization, not anonymization

`prostate-deid` performs **pseudonymization**, not anonymization, by
default: patient IDs, dates, and UIDs are remapped *consistently* (not
deleted or randomized per file) so longitudinal research across a patient's
own scans remains possible. A reversible mapping is kept on purpose. Under
GDPR (Art. 4(5), recital 26), retaining any re-identifying mapping — even
encrypted, even institution-side only — makes the output legally
pseudonymized data, not anonymized data. Keep the mapping store (`secret=`
/ `PROSTATE_DEID_KEY`, and the `MappingStore` JSON file) out of the
de-identified output directory, out of version control, and governed more
strictly than the data itself.

## Compliance disclaimer

Running this pipeline does not by itself constitute HIPAA or GDPR
compliance. It is a tool toward compliance, not a certification of it.
Actual compliance requires your institution's own review (e.g. a HIPAA
Expert Determination, or a GDPR Data Protection Impact Assessment).
Provided under the MIT License, which disclaims all warranty, including any
implied claim of regulatory compliance.

## What's in the box

| Module | What it does |
|---|---|
| `DicomTagScrub` | Deterministic header removal by named profile (`hipaa_safe_harbor`, `dicom_ps3.15_basic`); strips all private (odd-group) tags by default |
| `Pseudonymize` | HMAC-consistent PatientID remap, per-patient date shifting, deterministic UID regeneration under the `2.25` root |
| `BurnedInTextRedact` | OCR + NER pixel-level PHI detection incl. DICOM overlay planes; optional extra, degrades gracefully (never crashes the pipeline) when unavailable |
| `OrganMask` | Deterministic whole-gland / zonal / lesion masking from a segmentation label mask |
| `IntensityNormalize` | Robust Anatomy-Referenced Normalization (RARN) — see `docs/normalization_math.md` |
| `Compose` / `Sample` / `AuditEntry` | Shared pipeline + audit-trail primitives every transform above builds on |

Every `Transform` that changes a `Sample` appends a structured `AuditEntry`
(`step`, `action`, `detail`, `confidence`, `flagged_for_review`) rather than
mutating data silently — a pipeline run's `sample.audit` list is a
compliance-ready report of exactly what changed. `summarize_audit()`
aggregates it to headline counts.

**Asymmetric-error policy.** A missed PHI detection (false negative) is a
privacy incident; an over-redaction (false positive) is a data-quality
loss. These are not equally bad, so probabilistic steps bias toward
flagging/redacting under uncertainty rather than resolving ambiguity
silently — see `flagged_for_review` on every `AuditEntry`.

## Reviewer dashboards

```bash
pip install prostate-deid[dashboard]
streamlit run dashboard/anonymize_app.py   # run + inspect a single study
streamlit run dashboard/review_app.py      # triage every flagged_for_review item
```

A `flagged_for_review` entry with no reviewer is worse than not flagging at
all — it manufactures false confidence. `prostate_deid.export.export_review_bundle`
exports every flagged entry as a cropped thumbnail bundle the review
dashboard consumes.

## Scope

`prostate-deid` is scoped to prostate MRI, a pelvic acquisition with no
facial anatomy — it deliberately does **not** implement defacing (unlike
general-purpose medical-image de-identification frameworks). This keeps the
anatomical-masking component a first-class, well-specified feature instead
of an afterthought. See `docs/technical_details.md` for the full
rationale, limitations, and future work.

## Validation status

No quantitative PHI-detection recall/precision or masking-agreement (Dice)
numbers are published yet — see `docs/technical_details.md` Section 9 for
the evaluation protocol (TCIA Pseudo-PHI-DICOM-Data for header/burned-in
text; PROSTATEx for zonal/lesion masking, under its TCIA data use
agreement) that will produce them. Table 1 (capability comparison) is a
feature-support comparison, not a performance benchmark.

## Citing

See [`CITATION.cff`](CITATION.cff), or use GitHub's "Cite this repository"
button. A DOI-bearing Zenodo archive is minted on the first GitHub Release —
see [`.github/GITHUB_PUBLISHING_GUIDE.md`](.github/GITHUB_PUBLISHING_GUIDE.md).

## License

MIT — see [`LICENSE`](LICENSE).
