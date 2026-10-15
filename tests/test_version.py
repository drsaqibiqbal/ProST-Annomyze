"""All version references must agree with the single source, src/prostate_deid/_version.py."""

import json
import re
from pathlib import Path

import prostate_deid
from prostate_deid._version import __version__
from prostate_deid.convert import SOFTWARE_NAME

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_single_sourced():
    assert prostate_deid.__version__ == __version__
    assert SOFTWARE_NAME.endswith(__version__)
    assert 'dynamic = ["version"]' in (ROOT / "pyproject.toml").read_text(encoding="utf8")


def test_release_metadata_matches():
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf8")
    assert re.search(rf"^version: {re.escape(__version__)}$", cff, re.M)
    assert json.loads((ROOT / ".zenodo.json").read_text(encoding="utf8"))["version"] == __version__
    assert f"## [{__version__}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf8")
