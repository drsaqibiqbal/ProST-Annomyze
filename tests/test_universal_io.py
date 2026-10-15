"""v1-compatible universal_io API: regression tests for the bugs found in the v1 code review."""

from __future__ import annotations

import io

import nibabel as nib
import numpy as np
import pydicom
import pytest
from _dicom_fixtures import make_mr, save_load

from prostate_deid.convert import check_conformance, load_nifti
from prostate_deid.universal_io import array_to_dicom, dicom_to_nifti_bytes, full_wipe_dicom, load_any


def _nii(data, affine):
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    img.to_file_map(img.make_file_map({"image": buf, "header": buf}))
    return buf.getvalue()


def test_load_any_nifti_axis_order():
    data = np.zeros((40, 30, 5), np.int16)  # X, Y, Z
    data[3, 7, 2] = 1
    loaded = load_any(_nii(data, np.eye(4)), filename="x.nii")
    assert loaded["pixels"].shape == (5, 30, 40)  # v1 returned (5, 40, 30)
    assert loaded["pixels"][2, 7, 3] == 1


def test_full_wipe_compressed_saves(monkeypatch):
    pytest.importorskip("pylibjpeg")
    from pydicom.data import get_testdata_file

    ds = pydicom.dcmread(get_testdata_file("MR_small_jpeg_ls_lossless.dcm"))
    before = ds.pixel_array.copy()
    out, notes = full_wipe_dicom(ds, redact_pixels=False)
    back = save_load(out)  # v1 crashed here: native pixels under a JPEG-LS transfer syntax
    assert np.array_equal(back.pixel_array, before)
    assert "PatientName" in back and back.PatientName == ""


def test_array_to_dicom_volume_class_and_values():
    vol = (np.arange(3 * 6 * 7) % 1000).reshape(3, 6, 7).astype(np.float32)
    ds, notes = array_to_dicom(vol, redact_pixels=False)
    back = save_load(ds)
    assert back.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7.3"  # Multi-frame Grayscale Word SC
    assert np.array_equal(back.pixel_array, vol.astype(np.uint16))  # no min-max rescale
    assert check_conformance(back) == []


def test_dicom_to_nifti_bytes_spacing_not_swapped():
    ds = save_load(make_mr(spacing=(0.7, 0.5)))  # row spacing 0.7, column spacing 0.5
    img = load_nifti(dicom_to_nifti_bytes(ds))
    zooms = img.header.get_zooms()
    assert np.allclose(zooms[:2], (0.5, 0.7))  # i (columns) = 0.5, j (rows) = 0.7
    assert np.allclose(img.affine[:3, 3], [100.0, 100.0, 0.0])  # LPS (-100,-100,0) -> RAS


def test_nifti_io_save_nifti_matches_converter_affine(tmp_path):
    from prostate_deid.dicom_io import load_sample
    from prostate_deid.nifti_io import save_nifti

    ds = make_mr(spacing=(0.7, 0.5), iop=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), ipp=(-12.0, 30.0, 55.0))
    src = tmp_path / "in.dcm"
    ds.save_as(src, enforce_file_format=True)
    out = save_nifti(load_sample(src), tmp_path / "out.nii.gz")
    a = nib.load(str(out))
    b = load_nifti(dicom_to_nifti_bytes(save_load(ds)))
    assert np.allclose(a.affine[:, :2], b.affine[:, :2]) and np.allclose(a.affine[:3, 3], b.affine[:3, 3])
    assert np.array_equal(np.asarray(a.dataobj), np.asarray(b.dataobj).astype(np.float32))
