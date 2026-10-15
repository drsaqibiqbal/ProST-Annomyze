"""Synthetic DICOM / NIfTI builders for the converter tests. All identifiers are fictitious."""

from __future__ import annotations

import io

import numpy as np


def make_mr(
    pixels: np.ndarray | None = None,
    patient_id: str = "SYN-0001",
    study_uid: str = "1.2.826.0.1.3680043.99.1",
    series_uid: str = "1.2.826.0.1.3680043.99.1.1",
    sop_uid: str = "1.2.826.0.1.3680043.99.1.1.1",
    for_uid: str = "1.2.826.0.1.3680043.99.1.9",
    study_date: str = "20260101",
    series_number: int = 1,
    instance: int = 1,
    ipp=(-100.0, -100.0, 0.0),
    iop=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    spacing=(0.7, 0.5),
    manufacturer: str = "SIEMENS",
):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    if pixels is None:
        pixels = (np.arange(24 * 32, dtype=np.uint16).reshape(24, 32) % 977)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.MRImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, for_uid
    ds.PatientName = "DOE^JANE"
    ds.PatientID = patient_id
    ds.PatientBirthDate = "19580314"
    ds.PatientSex = "F"
    ds.PatientAge = "067Y"
    ds.InstitutionName = "Synthetic General Hospital"
    ds.ReferringPhysicianName = "SMITH^ROBERT"
    ds.OperatorsName = "TECH^TOM"
    ds.StationName = "MRI-ROOM-3"
    ds.DeviceSerialNumber = "SN-123456"
    ds.AccessionNumber = "ACC778899"
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = study_date
    ds.StudyTime = "094200"
    ds.Modality = "MR"
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = "Synthetic 3T"
    ds.SeriesNumber, ds.InstanceNumber = series_number, instance
    ds.SeriesDescription = "ep2d_diff_b50_800_1400 DOE^JANE 20260101"
    ds.ProtocolName = "t2_tse_tra Dr. Smith"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    ds.ScanningSequence, ds.SequenceVariant = "SE", "SK"
    ds.ScanOptions, ds.MRAcquisitionType = "", "2D"
    ds.EchoTime, ds.RepetitionTime, ds.EchoTrainLength = 89.0, 4000.0, 1
    ds.MagneticFieldStrength = 3.0
    ds.ImagePositionPatient = list(ipp)
    ds.ImageOrientationPatient = list(iop)
    ds.PixelSpacing = list(spacing)
    ds.SliceThickness = 3.0
    ds.PositionReferenceIndicator = ""
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.PixelData = pixels.astype(np.uint16).tobytes()
    ds.add_new(0x60003000, "OW", b"\x00" * 16)  # overlay plane bitmap
    ds.add_new(0x60000010, "US", 4)
    return ds


def add_siemens_diffusion(ds, b: int = 800, g=(0.1, 0.2, 0.9746794)):
    block = ds.private_block(0x0019, "SIEMENS MR HEADER", create=True)
    block.add_new(0x0C, "IS", str(b))
    block.add_new(0x0E, "FD", list(g))
    return ds


def add_ge_diffusion(ds, b: int = 1000, offset: bool = True, g=(0.0, 0.6, 0.8)):
    ds.Manufacturer = "GE MEDICAL SYSTEMS"
    parm = ds.private_block(0x0043, "GEMS_PARM_01", create=True)
    parm.add_new(0x39, "IS", [str(int(b + (1e9 if offset else 0))), "8", "0", "0"])
    acq = ds.private_block(0x0019, "GEMS_ACQU_01", create=True)
    acq.add_new(0xBB, "DS", str(g[0]))
    acq.add_new(0xBC, "DS", str(g[1]))
    acq.add_new(0xBD, "DS", str(g[2]))
    return ds


def save_load(ds):
    import pydicom

    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return pydicom.dcmread(io.BytesIO(buf.getvalue()))


def oblique_affine(spacing=(0.5, 0.7, 3.0), angle_deg: float = 20.0, origin=(30.0, -40.0, 12.0)):
    """RAS affine with a rotation about z and x (an oblique prostate-style plane) and non-square voxels."""
    a = np.deg2rad(angle_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(a / 2), -np.sin(a / 2)], [0, np.sin(a / 2), np.cos(a / 2)]])
    aff = np.eye(4)
    aff[:3, :3] = rz @ rx @ np.diag(spacing)
    aff[:3, 3] = origin
    return aff
