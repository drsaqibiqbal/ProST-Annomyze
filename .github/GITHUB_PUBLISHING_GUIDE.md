# Publishing a Python Library on GitHub — Full Guide

Practical, current (2026) checklist for taking a library from working code to a
properly published, citable, installable open-source package — written for the
de-identification/masking library discussed in this conversation, but generic
enough for any Python library.

---

## 1. Repository structure

Modern convention uses a `src/` layout — it prevents accidentally importing your
uninstalled development code and forces tests to run against the actually-installed
package, catching packaging bugs that a flat layout hides.

```
your-library/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # lint + test on every push/PR
│   │   └── release.yml         # build + publish on tagged release
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── src/
│   └── your_library/
│       ├── __init__.py
│       ├── py.typed            # marks the package as type-annotated
│       └── ...
├── tests/
│   └── test_*.py
├── docs/                       # Sphinx or mkdocs source
├── examples/                   # runnable example scripts/notebooks
├── pyproject.toml              # single source of truth for metadata + build config
├── README.md
├── LICENSE
├── CITATION.cff                # makes the repo "Cite this repository" work + feeds Zenodo
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
└── .zenodo.json                # optional, richer Zenodo metadata than CITATION.cff alone
```

---

## 2. `pyproject.toml` — the core metadata file

`setup.py`/`setup.cfg` are legacy; `pyproject.toml` (PEP 621) is the single file
all modern tools (pip, build, uv, hatch) read.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "your-library"                     # exactly what `pip install` will use
version = "0.1.0"                          # or use dynamic versioning from git tags
description = "Unified de-identification and masking for volumetric medical images"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "Saqib Iqbal", email = "you@example.com"},
]
keywords = ["medical imaging", "de-identification", "DICOM", "privacy", "masking"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
]
dependencies = [
    "numpy>=1.26",
    "pydicom>=2.4",
    "SimpleITK>=2.3",
    "scikit-image>=0.24",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff", "mypy"]
ocr = ["pytesseract", "spacy"]             # keep heavy/optional deps out of the core install

[project.urls]
Homepage = "https://github.com/<user>/your-library"
Documentation = "https://your-library.readthedocs.io"
Repository = "https://github.com/<user>/your-library"
"Bug Tracker" = "https://github.com/<user>/your-library/issues"
```

Notes:
- Declare optional heavy dependencies (OCR models, `pydeface` subprocess wrapper)
  as **optional extras** (`pip install your-library[ocr]`) rather than hard
  requirements — this is exactly the pattern Bio-Volumentations itself needed to
  avoid forcing a DL framework dependency on every user.
- `dynamic = ["version"]` with `hatch-vcs` lets the version come directly from the
  git tag you push, so you never hand-edit a version number.

---

## 3. Essential root files

| File | Purpose |
|---|---|
| **README.md** | First thing anyone sees — installation, quickstart code example, badges (CI status, PyPI version, license, DOI), link to docs |
| **LICENSE** | MIT recommended (matches the Bio-Volumentations precedent and is the least friction for reuse); GitHub and PyPI both auto-detect and display it |
| **CITATION.cff** | Machine-readable citation metadata — powers GitHub's "Cite this repository" button *and* is read directly by Zenodo's GitHub integration to populate author/title metadata on the archived DOI record |
| **CONTRIBUTING.md** | How to set up a dev environment, run tests, submit a PR — signals the project accepts outside contributions |
| **CODE_OF_CONDUCT.md** | Standard for any project expecting external contributors (Contributor Covenant is the common default) |
| **CHANGELOG.md** | Human-readable release history, ideally following [Keep a Changelog](https://keepachangelog.com) format |
| **.zenodo.json** | Optional, gives you more control over the archived record's metadata than `CITATION.cff` alone (both are read; if both exist, `.zenodo.json` takes precedence) |

Minimal `CITATION.cff` example:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
title: "your-library: Unified de-identification and masking for volumetric medical images"
authors:
  - family-names: Iqbal
    given-names: Saqib
    orcid: "https://orcid.org/0000-0003-0500-3676"
version: 0.1.0
date-released: 2026-07-08
license: MIT
repository-code: "https://github.com/<user>/your-library"
```

---

## 4. Testing

- Use `pytest`. Put tests in `tests/`, mirroring the `src/` package structure.
- Test both **deterministic** paths (DICOM tag scrubbing, organ masking — exact
  expected output) and **probabilistic** paths (OCR/NER detection, defacing QC)
  against a small fixture dataset with known ground truth, so CI catches
  regressions in recall/precision, not just crashes.
- Include at least one **end-to-end pipeline test** (`Compose([...])` run on a
  synthetic sample) so refactors that break the chaining logic itself get caught,
  not just individual transform bugs.
- Aim to publish a coverage badge; it's a cheap trust signal for reviewers and users.

---

## 5. Continuous integration (`.github/workflows/ci.yml`)

Run on every push and PR, across a Python version matrix:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e .[dev]
      - run: ruff check .
      - run: mypy src/
      - run: pytest tests/ -v --cov
```

---

## 6. Publishing to PyPI — use Trusted Publishing, not API tokens

As of 2025–2026 this is the settled best practice: **no long-lived PyPI tokens
stored in GitHub Secrets at all**. Instead, PyPI trusts a specific GitHub
repository + workflow file directly via OIDC, and issues a short-lived token
per run. This eliminates the leaked-token risk entirely.

**Setup (one-time):**
1. On PyPI → your account → *Publishing* → *Add a new pending publisher*.
2. Enter your PyPI project name, GitHub org/user, repo name, and the workflow
   filename (e.g. `release.yml`). This works even before the package exists on
   PyPI — the "pending" publisher activates on first successful publish.
3. Repeat on TestPyPI with a separate environment name if you want a staging step.

**`.github/workflows/release.yml`:**

```yaml
name: Release
on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.x"}
      - run: pip install build
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with: {name: dist, path: dist/}

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/your-library
    permissions:
      id-token: write     # mandatory for Trusted Publishing — no secrets needed
    steps:
      - uses: actions/download-artifact@v4
        with: {name: dist, path: dist/}
      - uses: pypa/gh-action-pypi-publish@release/v1
```

**Release flow, once this is set up:**
```bash
git tag v0.1.0
git push origin v0.1.0
# → CI builds, tests, and publishes to PyPI automatically
```

Test locally before every tagged release:
```bash
python -m build
python -m twine check dist/*
pip install dist/*.whl --force-reinstall   # smoke test the built wheel
```

---

## 7. Archiving on Zenodo for a citable DOI

This is what the SoftwareX "Permanent link to Reproducible Capsule" field
actually points to, and what makes the software citable independent of GitHub's
uptime.

1. Go to zenodo.org → log in with GitHub → authorize.
2. On the [Zenodo GitHub page](https://zenodo.org/account/settings/github/),
   toggle your repository **on**. (Repo must be public; for an org-owned repo,
   the org owner may need to approve access.)
3. Make sure a `LICENSE` file exists in the repo root — Zenodo reads it directly
   and overwrites its default CC-BY-4.0 assumption with your actual license.
4. Cut a **GitHub Release** (not just a tag) — this is the trigger. Zenodo
   downloads the repository at that release as a zip and mints a new DOI
   (or a new version under the same "concept DOI" if one already exists).
5. Copy the resulting DOI badge into your README and into the SoftwareX
   "Code metadata" table's Reproducible Capsule field.

Important caveats:
- You **cannot** pre-reserve or predict the DOI before the first GitHub release —
  if you need a DOI in-hand before release (e.g., to put in a submitted-but-not-published
  manuscript), do one manual archive upload directly on Zenodo's site instead,
  which does support DOI reservation.
- If both `CITATION.cff` and `.zenodo.json` exist in the repo, Zenodo's agent uses
  `.zenodo.json` and **ignores** `CITATION.cff` entirely — don't maintain both
  expecting them to merge.
- Version management should flow through GitHub release tags, not by hand-editing
  the metadata file's version field each time.

---

## 8. Documentation

- Docstrings on every public class/function (NumPy or Google style — pick one,
  stay consistent) so autodoc tools can generate reference pages without extra work.
- Host on **ReadTheDocs** (free for open source) via Sphinx, or MkDocs if you
  prefer Markdown-first docs — either integrates with a `docs/` folder and builds
  automatically from your repo on every push.
- Include a "Quickstart" in the README itself — most users decide whether to try
  a library from the README alone, before ever reaching full docs.

---

## 9. Badges worth adding to the README

```markdown
[![CI](https://github.com/<user>/your-library/actions/workflows/ci.yml/badge.svg)](...)
[![PyPI version](https://img.shields.io/pypi/v/your-library.svg)](https://pypi.org/project/your-library/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/your-library/badge/?version=latest)](...)
```

---

## 10. Security / repo hygiene

- **Branch protection** on `main`: require PR review + passing CI before merge;
  restrict who can push tags, since a tag push is what triggers a PyPI release.
- Never commit secrets — double check `.gitignore` covers `.env`, credentials,
  any real patient data or test fixtures containing real PHI (extremely relevant
  given this library's own subject matter — a leaked "test DICOM" with real PHI
  in your own test fixtures would be a serious irony to avoid).
- Enable 2FA on both your GitHub and PyPI accounts.
- Add a `SECURITY.md` describing how to responsibly report vulnerabilities,
  especially relevant here since this library handles PHI redaction — a
  detection-recall bug is a privacy incident, not just a normal bug.

---

## Order of operations, start to finish

1. Structure the repo (`src/` layout, tests, docs skeleton).
2. Write `pyproject.toml`, `LICENSE` (MIT), `README.md`, `CITATION.cff`.
3. Push to GitHub, set up branch protection.
4. Add CI workflow — get it green on a real PR before anything else.
5. Set up PyPI Trusted Publishing (pending publisher) and the release workflow.
6. Tag `v0.1.0`, push — confirm it lands on PyPI.
7. Enable Zenodo GitHub integration, cut a GitHub Release, get your DOI.
8. Set up ReadTheDocs, point the docs link in `pyproject.toml` and the SoftwareX
   code-metadata table at it.
9. Only now does the SoftwareX manuscript's `[TODO]` fields for GitHub URL,
   Zenodo DOI, and documentation link become fillable with real values.
