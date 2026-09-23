
import pytest

from prostate_deid.core import Compose
from prostate_deid.dicom_scrub import DicomTagScrub
from prostate_deid.pipeline import default_pipeline, summarize_audit
from prostate_deid.pseudonymize import Pseudonymize


@pytest.fixture(autouse=True)
def _set_test_secret(monkeypatch):
    monkeypatch.setenv("PROSTATE_DEID_KEY", "synthetic-test-secret-not-for-real-use")


def test_default_pipeline_orders_pseudonymize_before_scrub(tmp_path):
    pipeline = default_pipeline(mapping_store=str(tmp_path / "map.json"), redact_burned_in_text=False)
    kinds = [type(t).__name__ for t in pipeline.transforms]
    assert kinds.index("Pseudonymize") < kinds.index("DicomTagScrub")


def test_ordering_regression_scrub_after_pseudonymize_preserves_remap(tmp_path, synthetic_sample):
    """Regression test for the real bug documented in pipeline.py:
    DicomTagScrub must run AFTER Pseudonymize and must exclude the fields
    Pseudonymize remapped, or the HIPAA profile deletes PatientID/dates/UIDs
    outright and silently undoes the remap."""
    mapping_path = str(tmp_path / "map.json")
    pipeline = default_pipeline(mapping_store=mapping_path, redact_burned_in_text=False)
    out = pipeline(synthetic_sample)

    # PatientID and UIDs must SURVIVE as pseudonymized values, not be deleted.
    assert out.dicom_meta.get("PatientID", "").startswith("PSD-")
    assert out.dicom_meta.get("StudyInstanceUID", "").startswith("2.25.")
    # But other HIPAA-profile fields must still be gone.
    assert "PatientName" not in out.dicom_meta
    assert "InstitutionName" not in out.dicom_meta


def test_wrong_order_would_lose_the_remap_demonstrating_the_bug(tmp_path, synthetic_sample):
    """Demonstrates WHY the ordering rule exists: scrub-before-pseudonymize
    with no exclude_tags deletes PatientID outright instead of preserving a
    pseudonymized value."""
    wrong_order = Compose([
        DicomTagScrub(profile="hipaa_safe_harbor", always_apply=True),
        Pseudonymize(mapping_store=str(tmp_path / "map2.json"), always_apply=True),
    ])
    out = wrong_order(synthetic_sample)
    # PatientID was deleted by the scrub before Pseudonymize ever ran on it,
    # so Pseudonymize has nothing left to remap.
    assert "PatientID" not in out.dicom_meta


def test_summarize_audit_headline_counts(tmp_path, synthetic_sample):
    pipeline = default_pipeline(mapping_store=str(tmp_path / "map.json"), redact_burned_in_text=False)
    out = pipeline(synthetic_sample)
    summary = summarize_audit(out.audit)

    assert summary["total_entries"] == len(out.audit)
    assert summary["ids_pseudonymized"] == 1
    assert summary["uids_regenerated"] == 3
    assert summary["private_tags_removed"] >= 1
