"""Shared synthetic fixtures. NO real patient data is ever used here or
anywhere else in this test suite -- see CONTRIBUTING.md."""

from __future__ import annotations

import numpy as np
import pytest

from prostate_deid.core import Sample


@pytest.fixture
def synthetic_meta():
    return {
        "PatientName": "Doe^Jane",
        "PatientID": "SYN-0001",
        "PatientBirthDate": "19800101",
        "StudyDate": "20260101",
        "SeriesDate": "20260101",
        "StudyInstanceUID": "1.2.840.99999.1",
        "SeriesInstanceUID": "1.2.840.99999.1.1",
        "SOPInstanceUID": "1.2.840.99999.1.1.1",
        "ReferringPhysicianName": "Smith^Robert",
        "InstitutionName": "Synthetic General Hospital",
        "Modality": "MR",
        "(0009,1001)": "VENDOR_SECRET_BLOB",  # synthetic private tag
    }


@pytest.fixture
def synthetic_mask():
    """8x8x8 volume: background=0, PZ ring=2, TZ core=3, one lesion voxel=4."""
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 2  # peripheral zone shell
    mask[3:5, 3:5, 3:5] = 3  # transition zone core
    mask[4, 4, 4] = 4        # lesion
    return mask


@pytest.fixture
def synthetic_sample(synthetic_meta, synthetic_mask):
    rng = np.random.default_rng(42)
    image = rng.normal(loc=500, scale=50, size=(1, 8, 8, 8)).astype(np.float32)
    return Sample(image=image, dicom_meta=dict(synthetic_meta), mask=synthetic_mask)
