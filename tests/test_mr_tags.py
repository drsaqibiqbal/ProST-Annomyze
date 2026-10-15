"""mpMRI header policy: vendor b-value rescue, free-text scrubbing, age aggregation."""

from __future__ import annotations

import numpy as np
import pytest
from _dicom_fixtures import add_ge_diffusion, add_siemens_diffusion, make_mr, save_load

from prostate_deid.convert import StudyContext, deidentify_dicom
from prostate_deid.mr_tags import read_bvalue, scrub_free_text

KEY = "unit-test-key"


def test_siemens_bvalue_rescued_to_public_tags():
    ds = add_siemens_diffusion(make_mr(), b=800)
    res = deidentify_dicom(ds, StudyContext(secret=KEY), redact_pixels=False)
    out = save_load(res.datasets[0])
    assert float(out.DiffusionBValue) == 800.0
    assert np.allclose([float(v) for v in out.DiffusionGradientOrientation], [0.1, 0.2, 0.9746794])
    assert out.DiffusionDirectionality == "DIRECTIONAL"
    assert not any(e.tag.is_private for e in out)
    assert any(e.step == "diffusion_rescue" for e in res.audit)


@pytest.mark.parametrize("offset", [True, False])
def test_ge_bvalue_rescued_with_1e9_offset_removed(offset):
    ds = add_ge_diffusion(make_mr(), b=1000, offset=offset)
    out = deidentify_dicom(ds, StudyContext(secret=KEY), redact_pixels=False).datasets[0]
    assert float(out.DiffusionBValue) == 1000.0
    assert np.allclose([float(v) for v in out.DiffusionGradientOrientation], [0.0, 0.6, 0.8])


def test_bvalue_survives_after_roundtrip_through_file():
    ds = save_load(add_siemens_diffusion(make_mr(), b=1400))
    assert read_bvalue(ds)[0] == 1400.0


def test_public_bvalue_left_alone():
    ds = make_mr()
    ds.DiffusionBValue = 50.0
    out = deidentify_dicom(ds, StudyContext(secret=KEY), redact_pixels=False).datasets[0]
    assert float(out.DiffusionBValue) == 50.0


def test_free_text_scrub_keeps_sequence_label():
    ds = make_mr()
    audit = scrub_free_text(ds)
    assert ds.SeriesDescription == "ep2d_diff_b50_800_1400"
    assert ds.ProtocolName == "t2_tse_tra"
    assert len(audit) == 2 and all(e.flagged_for_review for e in audit)
    assert all("DOE" not in e.detail and "Smith" not in e.detail for e in audit)


def test_free_text_scrub_removes_ids_and_dates():
    ds = make_mr()
    ds.StudyDescription = "Prostate MRI 2026-01-01 MRN 12345678"
    scrub_free_text(ds)
    assert "2026" not in ds.StudyDescription and "12345678" not in ds.StudyDescription
    assert "Prostate MRI" in ds.StudyDescription


def test_age_over_89_aggregated():
    ds = make_mr()
    ds.PatientAge = "095Y"
    out = deidentify_dicom(ds, StudyContext(secret=KEY), redact_pixels=False).datasets[0]
    assert out.PatientAge == "090Y"
