import numpy as np
import pytest

from prostate_deid.core import Sample
from prostate_deid.dicom_scrub import DicomTagScrub, resolve_profile


def test_resolve_profile_hipaa_safe_harbor_contains_core_identifiers():
    tags = resolve_profile("hipaa_safe_harbor")
    for expected in ["PatientName", "PatientID", "PatientBirthDate", "InstitutionName"]:
        assert expected in tags


def test_resolve_unknown_profile_raises():
    with pytest.raises(KeyError):
        resolve_profile("not_a_real_profile")


def test_scrub_removes_profile_tags_and_private_tags(synthetic_sample):
    scrub = DicomTagScrub(profile="hipaa_safe_harbor", always_apply=True)
    out = scrub(synthetic_sample)

    assert "PatientName" not in out.dicom_meta
    assert "InstitutionName" not in out.dicom_meta
    assert "(0009,1001)" not in out.dicom_meta  # private tag stripped by default
    assert out.dicom_meta.get("Modality") == "MR"  # not in profile, not private -> kept

    actions = {e.detail.split(" ")[0] for e in out.audit}
    assert "PatientName" in actions


def test_exclude_tags_protects_pseudonymized_fields(synthetic_sample):
    scrub = DicomTagScrub(
        profile="hipaa_safe_harbor",
        exclude_tags=["PatientID"],
        always_apply=True,
    )
    out = scrub(synthetic_sample)
    assert out.dicom_meta.get("PatientID") == "SYN-0001"


def test_strip_private_tags_can_be_disabled(synthetic_sample):
    scrub = DicomTagScrub(profile="hipaa_safe_harbor", strip_private_tags=False, always_apply=True)
    out = scrub(synthetic_sample)
    assert "(0009,1001)" in out.dicom_meta


def test_no_dicom_meta_is_a_noop():
    s = Sample(image=np.zeros((1, 2, 2)))
    out = DicomTagScrub(always_apply=True)(s)
    assert out.dicom_meta is None
