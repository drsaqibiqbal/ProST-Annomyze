"""End-to-end DICOM round-trip test using a synthetic, in-memory pydicom
Dataset built from scratch -- never a real patient file (see CONTRIBUTING.md)."""

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from prostate_deid.dicom_io import load_sample, save_sample  # noqa: E402
from prostate_deid.dicom_scrub import DicomTagScrub  # noqa: E402


def _make_synthetic_dicom(path):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientName = "Synthetic^Test"
    ds.PatientID = "SYN-0002"
    ds.PatientBirthDate = "19900101"
    ds.StudyDate = "20260215"
    ds.SeriesDate = "20260215"
    ds.Modality = "MR"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.Rows, ds.Columns = 8, 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    pixels = (np.random.default_rng(0).random((8, 8)) * 500).astype(np.uint16)
    ds.PixelData = pixels.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), write_like_original=False)
    return path


def test_load_sample_reads_pixels_and_meta(tmp_path):
    dcm_path = _make_synthetic_dicom(tmp_path / "synthetic.dcm")
    sample = load_sample(dcm_path)

    assert sample.dicom_meta["PatientID"] == "SYN-0002"
    assert sample.image.shape[-2:] == (8, 8)
    assert sample.source_dataset is not None


def test_save_sample_removes_scrubbed_tags_and_sets_deid_flags(tmp_path):
    in_path = _make_synthetic_dicom(tmp_path / "in.dcm")
    out_path = tmp_path / "out.dcm"

    sample = load_sample(in_path)
    sample = DicomTagScrub(profile="hipaa_safe_harbor", always_apply=True)(sample)
    save_sample(sample, out_path)

    reloaded = load_sample(out_path)
    assert "PatientName" not in reloaded.dicom_meta
    assert reloaded.source_dataset.PatientIdentityRemoved == "YES"
    assert hasattr(reloaded.source_dataset, "DeidentificationMethod")
