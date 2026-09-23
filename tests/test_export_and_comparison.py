import json

from prostate_deid.comparison import CAPABILITIES, MATRIX, TOOLS, as_markdown_table
from prostate_deid.core import AuditEntry
from prostate_deid.export import export_review_bundle


def test_export_review_bundle_writes_manifest_for_flagged_entries(tmp_path, synthetic_sample):
    synthetic_sample.log(AuditEntry(
        step="burned_in_phi", action="redacted_region",
        detail="bbox=(0, 2, 0, 2), text_hash=abc123",
        confidence=0.4, flagged_for_review=True,
    ))
    synthetic_sample.log(AuditEntry(
        step="dicom_tag_scrub", action="removed_tag", detail="PatientName (profile)",
    ))  # not flagged -> excluded from bundle

    out_dir = export_review_bundle(synthetic_sample, tmp_path / "bundle", study_id="STUDY-1")
    manifest = json.loads((out_dir / "manifest.json").read_text())

    assert len(manifest) == 1
    assert manifest[0]["study_id"] == "STUDY-1"
    assert manifest[0]["thumbnail"] == "flag_0000.npy"
    assert (out_dir / "flag_0000.npy").exists()


def test_comparison_matrix_shape_matches_capabilities_and_tools():
    assert len(MATRIX) == len(CAPABILITIES)
    assert all(len(row) == len(TOOLS) for row in MATRIX)


def test_comparison_markdown_table_contains_all_tools():
    table = as_markdown_table()
    for tool in TOOLS:
        assert tool in table
