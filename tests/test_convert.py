"""v2 converter: geometry, intensities, conformance, series linkage, compressed input, raster quirks."""

from __future__ import annotations

import io

import nibabel as nib
import numpy as np
import pydicom
import pytest
from _dicom_fixtures import make_mr, oblique_affine, save_load
from PIL import Image

from prostate_deid.convert import (
    StudyContext,
    check_conformance,
    deidentify_dicom,
    load_nifti,
    nifti_to_dicom_series,
    output_filename,
    raster_to_dicom,
    series_to_nifti,
)

KEY = "unit-test-key"


def _nifti_bytes(data, affine, slope=None, inter=None):
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    fmap = img.make_file_map({"image": buf, "header": buf})
    img.to_file_map(fmap)
    raw = bytearray(buf.getvalue())
    if slope is not None:  # store raw ints + header scaling (nibabel would rescale an in-memory array)
        import struct

        struct.pack_into("<ff", raw, 112, float(slope), float(inter or 0.0))
    return bytes(raw)


def _roundtrip(nifti_bytes, **kw):
    ctx = StudyContext(secret=KEY)
    res = nifti_to_dicom_series(nifti_bytes, ctx, redact_pixels=False, source_key="t", **kw)
    dss = [save_load(d) for d in res.datasets]
    gz, side, audit = series_to_nifti(dss)
    return res, dss, load_nifti(gz), side, audit


# --- NIfTI <-> DICOM geometry ------------------------------------------------------------------

def test_nifti_roundtrip_oblique_nonsquare_is_exact():
    rng = np.random.default_rng(0)
    data = rng.integers(-500, 3000, size=(40, 30, 5)).astype(np.int16)  # X=40, Y=30, Z=5
    aff = oblique_affine()
    res, dss, back, side, audit = _roundtrip(_nifti_bytes(data, aff))
    assert len(dss) == 5
    assert (dss[0].Rows, dss[0].Columns) == (30, 40)  # v1 bug: slices were transposed (40 x 30)
    assert np.allclose([float(v) for v in dss[0].PixelSpacing], [0.7, 0.5])  # [row (j), column (i)]
    assert np.array_equal(np.asanyarray(back.dataobj), data)
    assert np.allclose(back.affine, aff, atol=1e-4)
    assert side["orientation_check_max_error_mm"] < 1e-3


def test_slice_k_matches_nifti_k():
    data = np.zeros((8, 6, 3), dtype=np.int16)
    data[2, 4, 1] = 99  # i=2 (column), j=4 (row), k=1
    res, dss, *_ = _roundtrip(_nifti_bytes(data, np.diag([1.0, 1.0, 2.0, 1.0])))
    px = dss[1].pixel_array
    assert px[4, 2] == 99 and px.sum() == 99


def test_integer_valued_float_adc_is_bit_identical():
    rng = np.random.default_rng(1)
    adc = rng.integers(0, 3200, size=(16, 12, 4)).astype(np.float32)  # ADC in 1e-6 mm^2/s
    res, dss, back, side, _ = _roundtrip(_nifti_bytes(adc, np.diag([0.8, 0.8, 3.0, 1.0])))
    assert float(dss[0].RescaleSlope) == 1.0 and float(dss[0].RescaleIntercept) == 0.0
    assert np.array_equal(back.get_fdata().astype(np.float32), adc)
    assert any(e.action == "preserved_values" for e in res.audit)


def test_nifti_scl_slope_carried_exactly():
    raw = np.arange(4 * 5 * 2, dtype=np.int16).reshape(4, 5, 2) * 7
    res, dss, back, *_ = _roundtrip(_nifti_bytes(raw, np.eye(4), slope=0.5, inter=-10.0))
    assert float(dss[0].RescaleSlope) == 0.5 and float(dss[0].RescaleIntercept) == -10.0
    assert np.allclose(back.get_fdata(), raw * 0.5 - 10.0)


def test_true_float_is_quantised_with_bounded_logged_error():
    vals = np.linspace(0.0, 1.0, 6 * 5 * 2, dtype=np.float32).reshape(6, 5, 2) + 0.123
    res, dss, back, *_ = _roundtrip(_nifti_bytes(vals, np.eye(4)))
    q = [e for e in res.audit if e.action == "quantised_values"]
    assert q and q[0].flagged_for_review
    assert np.max(np.abs(back.get_fdata() - vals)) <= float(dss[0].RescaleSlope) / 2 + 1e-6


@pytest.mark.parametrize("modality", ["MR", "OT"])
def test_nifti_output_conformance(modality):
    data = np.ones((6, 5, 3), dtype=np.int16)
    ctx = StudyContext(secret=KEY)
    res = nifti_to_dicom_series(_nifti_bytes(data, np.eye(4)), ctx, modality=modality, redact_pixels=False)
    for d in res.datasets:
        assert check_conformance(save_load(d)) == []
    assert len({d.SeriesInstanceUID for d in res.datasets}) == 1
    assert len({d.FrameOfReferenceUID for d in res.datasets}) == 1


def test_multi_bvalue_series_becomes_4d_with_bvals():
    dss = []
    for z in range(3):
        for k, b in enumerate((50, 800, 1400)):
            d = make_mr(pixels=np.full((4, 5), z * 10 + k, np.uint16), ipp=(0.0, 0.0, 3.0 * z),
                        instance=z * 3 + k + 1, sop_uid=f"1.2.3.{z}.{k}")
            d.DiffusionBValue = float(b)
            dss.append(save_load(d))
    gz, side, _ = series_to_nifti(dss)
    img = load_nifti(gz)
    assert img.shape == (5, 4, 3, 3)
    assert side["DiffusionBValues"] == [50.0, 800.0, 1400.0]
    assert int(np.asanyarray(img.dataobj)[0, 0, 2, 1]) == 21


def test_mixed_orientation_series_is_rejected():
    a = save_load(make_mr(sop_uid="1.2.3.1"))
    b = save_load(make_mr(sop_uid="1.2.3.2", iop=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), ipp=(0.0, 0.0, 3.0)))
    with pytest.raises(ValueError, match="ImageOrientationPatient"):
        series_to_nifti([a, b])


# --- real DICOM de-identification ---------------------------------------------------------------

def test_deidentify_removes_phi_keeps_mr_utility():
    ds = make_mr()
    res = deidentify_dicom(ds, StudyContext(secret=KEY), redact_pixels=False)
    out = save_load(res.datasets[0])
    for kw in ("InstitutionName", "ReferringPhysicianName", "OperatorsName", "StationName", "DeviceSerialNumber",
               "PatientBirthDate"):
        assert kw not in out or out[kw].value in ("", None)
    assert out.PatientID.startswith("PSD-") and str(out.PatientName) == out.PatientID
    assert out.AccessionNumber == ""
    assert not any(e.tag.is_private for e in out)
    assert not any(0x6000 <= e.tag.group <= 0x60FF for e in out)
    for kw in ("ImageType", "EchoTime", "RepetitionTime", "MagneticFieldStrength", "FrameOfReferenceUID",
               "ImagePositionPatient", "SeriesDescription"):
        assert kw in out
    assert "DOE" not in out.SeriesDescription and "ep2d_diff_b50_800_1400" in out.SeriesDescription
    assert "Smith" not in out.ProtocolName
    assert out.StudyDate != "20260101" and out.PatientIdentityRemoved == "YES"
    assert check_conformance(out) == []


def test_series_linkage_and_date_intervals():
    ctx = StudyContext(secret=KEY)
    t2 = deidentify_dicom(make_mr(series_uid="1.2.3.10", sop_uid="1.2.3.10.1", series_number=1), ctx,
                          redact_pixels=False).datasets[0]
    adc = deidentify_dicom(make_mr(series_uid="1.2.3.20", sop_uid="1.2.3.20.1", series_number=2,
                                   study_date="20260111"), ctx, redact_pixels=False).datasets[0]
    assert t2.StudyInstanceUID == adc.StudyInstanceUID != "1.2.826.0.1.3680043.99.1"
    assert t2.FrameOfReferenceUID == adc.FrameOfReferenceUID
    assert t2.SeriesInstanceUID != adc.SeriesInstanceUID
    assert t2.PatientID == adc.PatientID
    from datetime import datetime

    d = (datetime.strptime(adc.StudyDate, "%Y%m%d") - datetime.strptime(t2.StudyDate, "%Y%m%d")).days
    assert d == 10


def test_pseudonyms_are_keyed():
    a = deidentify_dicom(make_mr(), StudyContext(secret=KEY), redact_pixels=False).datasets[0]
    b = deidentify_dicom(make_mr(), StudyContext(secret=KEY), redact_pixels=False).datasets[0]
    c = deidentify_dicom(make_mr(), StudyContext(secret="other-key"), redact_pixels=False).datasets[0]
    assert a.PatientID == b.PatientID != c.PatientID
    assert a.SOPInstanceUID == b.SOPInstanceUID != c.SOPInstanceUID


def test_v1_policy_still_available_and_conformant():
    out = deidentify_dicom(make_mr(), StudyContext(secret=KEY), keep_mr_tags=False, redact_pixels=False).datasets[0]
    assert "EchoTime" not in out and "SeriesDescription" not in out
    assert out.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"  # relabelled: no longer a valid MR Image
    assert check_conformance(save_load(out)) == []


def test_compressed_dicom_is_decoded_and_saved_uncompressed():
    from pydicom.data import get_testdata_file

    pytest.importorskip("pylibjpeg")
    ds = pydicom.dcmread(get_testdata_file("MR_small_jpeg_ls_lossless.dcm"))
    before = ds.pixel_array.copy()
    out = save_load(deidentify_dicom(ds, StudyContext(secret=KEY), redact_pixels=False).datasets[0])
    assert out.file_meta.TransferSyntaxUID == pydicom.uid.ExplicitVRLittleEndian
    assert np.array_equal(out.pixel_array, before)
    assert check_conformance(out) == []


def test_output_filename_never_uses_upload_name():
    out = deidentify_dicom(make_mr(), StudyContext(secret=KEY), redact_pixels=False).datasets[0]
    name = output_filename(out)
    assert name.startswith(out.PatientID) and "DOE" not in name and name.endswith(".dcm")


# --- raster input ------------------------------------------------------------------------------

def _png(im, **kw):
    b = io.BytesIO()
    im.save(b, "PNG", **kw)
    return b.getvalue()


def test_palette_png_becomes_colours():
    im = Image.new("P", (20, 10))
    im.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    im.paste(1, (0, 0, 10, 10))
    out = save_load(raster_to_dicom(_png(im), StudyContext(secret=KEY), redact_pixels=False).datasets[0])
    assert out.PhotometricInterpretation == "RGB"
    assert list(out.pixel_array[0, 0]) == [255, 0, 0] and list(out.pixel_array[0, 15]) == [0, 0, 0]
    assert check_conformance(out) == []


def test_gray_alpha_png_matches_rows_columns():
    out = save_load(raster_to_dicom(_png(Image.new("LA", (20, 10), (200, 128))), StudyContext(secret=KEY),
                                    redact_pixels=False).datasets[0])
    assert out.pixel_array.shape == (10, 20) and out.PhotometricInterpretation == "MONOCHROME2"
    assert int(out.pixel_array[0, 0]) == round(200 * 128 / 255)


def test_16bit_png_values_not_rescaled():
    arr = (np.arange(12 * 8, dtype=np.uint16).reshape(8, 12) * 300)
    out = save_load(raster_to_dicom(_png(Image.fromarray(arr)), StudyContext(secret=KEY), redact_pixels=False)
                    .datasets[0])
    assert np.array_equal(out.pixel_array, arr)


def test_jpeg_exif_not_carried_over():
    im = Image.new("RGB", (16, 16), (10, 20, 30))
    exif = Image.Exif()
    exif[0x013B] = "Dr. Secret Person"  # Artist
    b = io.BytesIO()
    im.save(b, "JPEG", exif=exif.tobytes())
    res = raster_to_dicom(b.getvalue(), StudyContext(secret=KEY), redact_pixels=False)
    raw = io.BytesIO()
    res.datasets[0].save_as(raw, enforce_file_format=True)
    assert b"Secret" not in raw.getvalue()
    assert any(e.action == "stripped_metadata" for e in res.audit)
