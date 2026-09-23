# Contributing to prostate-deid

Thanks for considering a contribution. This project handles PHI-adjacent
code, so a couple of the usual open-source rules are stricter here than you
might expect.

## Dev setup

```bash
git clone https://github.com/drsaqibiqbal/annolib.git
cd annolib
pip install -e ".[dev,ocr,dashboard]"
pytest tests/ -v
ruff check .
mypy src/
```

## Absolute rules

- **Never commit real patient data, or a real DICOM file, to this
  repository — including as a test fixture.** All test fixtures must be
  synthetically generated (see `tests/fixtures/`), following the TCIA
  Pseudo-PHI-DICOM-Data model of inserting synthetic PHI into otherwise
  real, non-identifying pixel data. If you're unsure whether a file you're
  about to add contains real PHI, don't add it — ask first.
- **Never write a hardcoded pseudonymization secret** into source, tests,
  or examples. Use the `PROSTATE_DEID_KEY` environment variable or an
  explicit `secret=` parameter pointed at a value that is itself not
  committed.
- **Don't add a quantitative performance claim** (recall, precision, Dice,
  "outperforms X") to the README, docstrings, or `docs/` without a
  reproducible evaluation script and dataset citation backing it. See
  `docs/technical_details.md` Section 9 for what's required before such a
  claim is meaningful.

## Pull requests

1. One logical change per PR.
2. Add or update a test for anything in `src/prostate_deid/` you touch.
3. `ruff check .` and `mypy src/` clean, `pytest` green, before requesting
   review.
4. If you change transform ordering semantics (see the
   `Pseudonymize` → `DicomTagScrub` note in `pipeline.py`), add a
   regression test — an ordering bug here is a privacy bug, not a
   cosmetic one.

## Reporting a vulnerability or a de-identification gap

See [`SECURITY.md`](SECURITY.md) — please do not open a public issue for a
detection-recall gap or a real PHI leak; report it privately first.
