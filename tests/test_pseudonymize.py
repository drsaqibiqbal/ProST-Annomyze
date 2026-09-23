
import pytest

from prostate_deid.pseudonymize import MappingStore, Pseudonymize


@pytest.fixture(autouse=True)
def _set_test_secret(monkeypatch):
    monkeypatch.setenv("PROSTATE_DEID_KEY", "synthetic-test-secret-not-for-real-use")


def test_pseudonymize_is_consistent_across_two_samples(tmp_path, synthetic_sample):
    mapping_path = tmp_path / "mapping.json"
    p1 = Pseudonymize(mapping_store=str(mapping_path), always_apply=True)
    out1 = p1(synthetic_sample)
    pseudo_id_1 = out1.dicom_meta["PatientID"]

    # Fresh Pseudonymize instance, same mapping store on disk, same source ID
    import copy
    sample2 = copy.deepcopy(synthetic_sample)
    sample2.dicom_meta["PatientID"] = "SYN-0001"  # same original patient

    p2 = Pseudonymize(mapping_store=str(mapping_path), always_apply=True)
    out2 = p2(sample2)
    assert out2.dicom_meta["PatientID"] == pseudo_id_1


def test_pseudonymize_shifts_dates_but_keeps_them_valid(tmp_path, synthetic_sample):
    mapping_path = tmp_path / "mapping.json"
    p = Pseudonymize(mapping_store=str(mapping_path), always_apply=True)
    out = p(synthetic_sample)
    shifted = out.dicom_meta["StudyDate"]
    assert shifted != "20260101"
    assert len(shifted) == 8 and shifted.isdigit()


def test_pseudonymize_regenerates_uids_under_225_root(tmp_path, synthetic_sample):
    mapping_path = tmp_path / "mapping.json"
    p = Pseudonymize(mapping_store=str(mapping_path), always_apply=True)
    out = p(synthetic_sample)
    for field in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        assert out.dicom_meta[field].startswith("2.25.")
        assert len(out.dicom_meta[field]) <= 64


def test_missing_secret_raises(tmp_path, synthetic_sample, monkeypatch):
    monkeypatch.delenv("PROSTATE_DEID_KEY", raising=False)
    with pytest.raises(RuntimeError):
        Pseudonymize(mapping_store=str(tmp_path / "mapping.json"))


def test_mapping_store_roundtrip(tmp_path):
    path = tmp_path / "mapping.json"
    store = MappingStore(str(path))
    store.put("patient_id", "orig", "pseudo")
    store.save()

    reloaded = MappingStore(str(path))
    assert reloaded.get("patient_id", "orig") == "pseudo"
