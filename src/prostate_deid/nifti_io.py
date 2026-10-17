"""NIfTI export for :class:`~prostate_deid.core.Sample` objects (annolib-compatible API).

This merges annolib's ``nifti_io`` (``save_nifti`` / ``dicom_to_nifti``) into
ProST-Annomyze, so one package now has both the universal converter and the
Sample-based NIfTI export. There is a single writer underneath: the affine is
built the same way as :func:`prostate_deid.convert.series_to_nifti` -- full
orientation from ImageOrientationPatient / ImagePositionPatient /
PixelSpacing, LPS -> RAS -- instead of annolib's diagonal approximation, and
the output is ``.nii.gz`` when the path ends in ``.gz``.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any

import numpy as np

from .core import Sample

__all__ = ["save_nifti", "dicom_to_nifti", "affine_from_dataset"]


def affine_from_dataset(ds: Any, n_slices: int = 1) -> np.ndarray:
    """RAS voxel->world affine for a (Z, Y, X) volume from one DICOM dataset's geometry.

    Uses IOP/IPP/PixelSpacing when present (slice direction = row x column cosines, spacing =
    SpacingBetweenSlices or SliceThickness); identity spacing otherwise.
    """
    lps = np.eye(4)
    if ds is not None:
        ps = getattr(ds, "PixelSpacing", None) or [1.0, 1.0]
        iop = getattr(ds, "ImageOrientationPatient", None) or [1, 0, 0, 0, 1, 0]
        ipp = getattr(ds, "ImagePositionPatient", None) or [0, 0, 0]
        row_cos = np.array([float(v) for v in iop[:3]])
        col_cos = np.array([float(v) for v in iop[3:]])
        step = float(getattr(ds, "SpacingBetweenSlices", None) or getattr(ds, "SliceThickness", None) or 1.0)
        lps[:3, 0] = row_cos * float(ps[1])
        lps[:3, 1] = col_cos * float(ps[0])
        lps[:3, 2] = np.cross(row_cos, col_cos) * step
        lps[:3, 3] = [float(v) for v in ipp]
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps


def save_nifti(sample: Sample, path: str | Path, affine: np.ndarray | None = None) -> Path:
    """Write ``sample.image`` (CZYX / ZYX / YX) to ``.nii`` or ``.nii.gz`` (by suffix).

    ``affine``: 4x4 RAS voxel->world matrix; derived from ``sample.source_dataset`` geometry when
    omitted. The first channel / volume is written; values are written as float32.
    """
    import nibabel as nib

    path = Path(path)
    arr = np.squeeze(np.asarray(sample.image, dtype=np.float32))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    while arr.ndim > 3:
        arr = arr[0]
    if affine is None:
        affine = affine_from_dataset(sample.source_dataset, arr.shape[0])
    img = nib.Nifti1Image(np.ascontiguousarray(np.transpose(arr, (2, 1, 0))), np.asarray(affine, float))
    img.set_sform(img.affine, code=1)
    img.set_qform(img.affine, code=1)
    img.header.set_xyzt_units("mm", "sec")
    buf = io.BytesIO()
    img.to_file_map(img.make_file_map({"image": buf, "header": buf}))
    data = buf.getvalue()
    if path.suffix == ".gz":
        gz = io.BytesIO()
        with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0) as fh:
            fh.write(data)
        data = gz.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def dicom_to_nifti(in_path: str | Path, out_path: str | Path, normalize: Any = None) -> Path:
    """Read one DICOM file, optionally normalize (e.g. RARN), write ``.nii`` / ``.nii.gz``."""
    from .dicom_io import load_sample

    sample = load_sample(in_path)
    if normalize is not None:
        sample = normalize(sample)
    return save_nifti(sample, out_path)
