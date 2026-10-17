"""
ProST-Annomyze v2 converter: any input -> de-identified, standards-shaped DICOM
-> optional geometry-correct .nii.gz, with one shared audit trail.

This module implements the fixes listed in the DICOM/NIfTI and prostate-MRI
de-identification briefings (docs/v2_changes.md maps each item to code):

* Geometry. NIfTI input is read in the correct axis order (rows = j,
  columns = i) and its affine is written to PixelSpacing, SliceThickness,
  ImagePositionPatient and ImageOrientationPatient (RAS -> LPS), one file per
  slice in a shared series and frame of reference. DICOM -> NIfTI builds the
  full orientation-aware affine from IOP/IPP (LPS -> RAS), sorts slices along
  the slice normal, checks spacing uniformity, and self-checks every slice
  position against the source geometry.
* Conformance. Type 1/2 attributes of the Patient, General Study, General
  Series, Equipment, Image Pixel, Image Plane, MR Image and SC modules are
  always present (empty where unknown); :func:`check_conformance` verifies
  this for every output.
* Compressed DICOM is decoded first and always written back as Explicit VR
  Little Endian, so redacted pixels and the transfer syntax always agree.
* Series and consistent IDs. :class:`StudyContext` routes PatientID, every
  UID (incl. FrameOfReferenceUID) and dates through the library's keyed
  pseudonymization, so T2W, DWI and ADC from one patient stay in one study,
  share one frame of reference and keep their date intervals.
* Intensities. Stored values are never min-max rescaled: integers are kept
  exactly, NIfTI scl_slope/scl_inter become RescaleSlope/Intercept, and only
  genuinely non-integer float data is quantised (with the maximum error
  logged and flagged).
* MR content. :mod:`prostate_deid.mr_tags` keeps the non-identifying MR
  acquisition / diffusion / series-identity tags, rescues vendor b-values into
  public tags and scrubs free-text descriptions.
* Pixels. Burned-in text is removed with :mod:`prostate_deid.text_restore`
  (segment + restore, or v1 blackout), on every input type.
* Audit. Every step emits :class:`~prostate_deid.core.AuditEntry` objects, so
  the converter is part of the same trail the review dashboard reads.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._version import __version__
from .core import AuditEntry
from .mr_tags import (
    FUNCTIONAL_GROUP_KEEP,
    MR_SAFE_KEEP,
    TECHNICAL_KEEP,
    read_bvalue,
    rescue_diffusion_tags,
    scrub_free_text,
)
from .pseudonymize import MappingStore, _date_offset_days, _pseudo_patient_id, _pseudo_uid, _shift_dicom_date

__all__ = [
    "StudyContext",
    "ConversionResult",
    "deidentify_dicom",
    "nifti_to_dicom_series",
    "raster_to_dicom",
    "series_to_nifti",
    "check_conformance",
    "output_filename",
    "load_nifti",
    "SOFTWARE_NAME",
]

SOFTWARE_NAME = f"ProST-Annomyze {__version__}"

_MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
_SC_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.7"
_MF_GRAY_BYTE_SC = "1.2.840.10008.5.1.4.1.1.7.2"
_MF_GRAY_WORD_SC = "1.2.840.10008.5.1.4.1.1.7.3"
_CLASS_UIDS_NOT_REMAPPED = {"SOPClassUID", "ReferencedSOPClassUID", "MediaStorageSOPClassUID", "TransferSyntaxUID",
                            "CodingSchemeUID", "MappingResourceUID", "ContextUID"}

_LOSSLESS_TS = {
    "1.2.840.10008.1.2.4.57", "1.2.840.10008.1.2.4.70",  # JPEG Lossless
    "1.2.840.10008.1.2.4.80",  # JPEG-LS Lossless
    "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
    "1.2.840.10008.1.2.4.201", "1.2.840.10008.1.2.4.202",  # HTJ2K Lossless (+RPCL)
    "1.2.840.10008.1.2.5",  # RLE Lossless
    "1.2.840.10008.1.2.1.99",  # Deflated Explicit VR Little Endian
}

# DICOM PS3.16 CID 7050 de-identification method codes for what this converter does.
_DEID_CODES = [
    ("113100", "Basic Application Confidentiality Profile"),
    ("113101", "Clean Pixel Data Option"),
    ("113105", "Clean Descriptors Option"),
    ("113107", "Retain Longitudinal Temporal Information Modified Dates Option"),
    ("113108", "Retain Patient Characteristics Option"),
    ("113109", "Retain Device Identity Option"),
]


# ---------------------------------------------------------------------------
# Study-level pseudonymization context
# ---------------------------------------------------------------------------

class StudyContext:
    """Consistent pseudonyms for everything converted in one batch (or across batches with a key).

    Parameters
    ----------
    secret:
        HMAC key. With the same key, the same source PatientID / UID always maps to the same
        pseudonym, across runs. If omitted, ``PROSTATE_DEID_KEY`` is used; if that is unset too, a
        random per-session key is generated -- consistent within this session only, and the output
        is then not re-linkable by anyone (closer to anonymization than pseudonymization).
    mapping_store:
        Optional :class:`~prostate_deid.pseudonymize.MappingStore` (or path) recording
        original -> pseudonym for authorised re-identification. Keep it away from the output.
    subject_label:
        Pseudonymous subject ID to use for inputs with no PatientID (raster / NIfTI). If omitted, one
        is derived from the key and the upload batch.
    shift_dates:
        Shift dates by a consistent per-patient offset (intervals preserved). ``False`` removes them.
    """

    def __init__(
        self,
        secret: str | None = None,
        mapping_store: str | os.PathLike | MappingStore | None = None,
        subject_label: str | None = None,
        shift_dates: bool = True,
        prefix: str = "PSD-",
    ):
        if secret is None:
            secret = os.environ.get("PROSTATE_DEID_KEY")
        self.session_key = secret is None
        self._secret = (secret or os.urandom(32).hex()).encode("utf-8")
        if mapping_store is not None and not isinstance(mapping_store, MappingStore):
            mapping_store = MappingStore(mapping_store)
        self.store = mapping_store
        self.subject_label = subject_label
        self.shift_dates = shift_dates
        self.prefix = prefix

    # --- identities ---------------------------------------------------------
    def patient_id(self, original: str | None, fallback_key: str = "batch") -> str:
        if not original:
            if self.subject_label:
                return self.subject_label
            original = f"__no_patient_id__:{fallback_key}"
        if self.store is not None:
            hit = self.store.get("patient_id", original)
            if hit:
                return hit
        pid = _pseudo_patient_id(self._secret, original, self.prefix)
        if self.store is not None:
            self.store.put("patient_id", original, pid)
        return pid

    def uid(self, original: str) -> str:
        if self.store is not None:
            hit = self.store.get("uid", original)
            if hit:
                return hit
        new = _pseudo_uid(self._secret, original)
        if self.store is not None:
            self.store.put("uid", original, new)
        return new

    def synth_uid(self, *parts: str) -> str:
        """Deterministic UID for objects that have no source UID (raster / NIfTI inputs)."""
        return _pseudo_uid(self._secret, "synth:" + "|".join(parts))

    def date_offset(self, original_pid: str | None) -> int:
        return _date_offset_days(self._secret, original_pid or "__none__")

    def shift_date(self, value: str, original_pid: str | None) -> str:
        if not value:
            return value
        return _shift_dicom_date(str(value)[:8], self.date_offset(original_pid))

    def shift_datetime(self, value: str, original_pid: str | None) -> str:
        v = str(value)
        if len(v) < 8 or not v[:8].isdigit():
            return v
        return self.shift_date(v[:8], original_pid) + v[8:]

    def save(self) -> None:
        if self.store is not None:
            self.store.save()


@dataclass
class ConversionResult:
    datasets: list[Any]
    audit: list[AuditEntry] = field(default_factory=list)
    restore: Any = None  # text_restore.RestoreResult (or None when pixels were not inspected)
    filenames: list[str] = field(default_factory=list)

    @property
    def notes(self) -> list[str]:
        return [str(e) for e in self.audit]


# ---------------------------------------------------------------------------
# Pixel helpers
# ---------------------------------------------------------------------------

def _restore_pixels(stack: np.ndarray, text_method: str, restore_kwargs: dict) -> tuple[np.ndarray, Any, list[AuditEntry]]:
    """Run burned-in text removal on a (frames, H, W[, C]) stack. Returns (stack, result, audit)."""
    from .text_restore import restore_burned_in_text

    res = restore_burned_in_text(stack, method=text_method, **restore_kwargs)
    audit = list(res.audit)
    for n in res.notes:
        audit.append(AuditEntry("burned_in_phi", "summary", n,
                                flagged_for_review="NOT inspected" in n or "fell back" in n or "review" in n))
    return res.pixels, res, audit


def _choose_integer_storage(values: np.ndarray) -> tuple[np.ndarray, float, float, bool, float]:
    """Store ``values`` in 16-bit DICOM pixels without min-max rescaling where at all possible.

    Returns ``(stored, slope, intercept, lossless, max_abs_error)``.
    """
    v = np.asarray(values)
    finite = np.isfinite(v)
    if not finite.all():
        v = np.where(finite, v, 0)
    if np.issubdtype(v.dtype, np.integer) or np.all(np.mod(v, 1) == 0):
        lo, hi = float(v.min()), float(v.max())
        if lo >= 0 and hi <= 65535:
            return v.astype(np.uint16), 1.0, 0.0, True, 0.0
        if lo >= -32768 and hi <= 32767:
            return v.astype(np.int16), 1.0, 0.0, True, 0.0
    lo, hi = float(np.min(v)), float(np.max(v))
    slope = (hi - lo) / 65535.0 if hi > lo else 1.0
    stored = np.clip(np.rint((v - lo) / slope), 0, 65535).astype(np.uint16)
    err = float(np.max(np.abs(stored * slope + lo - v))) if v.size else 0.0
    return stored, float(slope), float(lo), False, err


def _set_pixel_module(ds: Any, frame: np.ndarray) -> None:
    color = frame.ndim == 3 and frame.shape[-1] == 3
    ds.Rows, ds.Columns = int(frame.shape[0]), int(frame.shape[1])
    ds.SamplesPerPixel = 3 if color else 1
    ds.PhotometricInterpretation = "RGB" if color else "MONOCHROME2"
    if color:
        ds.PlanarConfiguration = 0
    bits = 8 if frame.dtype in (np.uint8, np.int8) else 16
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 1 if np.issubdtype(frame.dtype, np.signedinteger) else 0
    ds.PixelData = np.ascontiguousarray(frame).tobytes()


def _new_dataset(sop_class: str, sop_uid: str) -> Any:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_uid
    return ds


def _fill_common_type2(ds: Any, pid: str) -> None:
    """Patient / General Study / Series / Equipment Type 1-2 attributes, empty where unknown."""
    ds.PatientName = pid
    ds.PatientID = pid
    for kw in ("PatientBirthDate", "PatientSex", "ReferringPhysicianName", "StudyID", "AccessionNumber",
               "StudyDate", "StudyTime", "Manufacturer"):
        if kw not in ds:
            setattr(ds, kw, "")
    if "SeriesNumber" not in ds:
        ds.SeriesNumber = 1
    if "InstanceNumber" not in ds:
        ds.InstanceNumber = 1


def _mark_deidentified(ds: Any, method: str) -> None:
    from pydicom.dataset import Dataset

    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = f"{SOFTWARE_NAME}: {method}"[:64]
    seq = []
    for code, meaning in _DEID_CODES:
        item = Dataset()
        item.CodeValue = code
        item.CodingSchemeDesignator = "DCM"
        item.CodeMeaning = meaning
        seq.append(item)
    ds.DeidentificationMethodCodeSequence = seq
    ds.LongitudinalTemporalInformationModified = "MODIFIED"


# ---------------------------------------------------------------------------
# Real DICOM -> de-identified DICOM
# ---------------------------------------------------------------------------

def ensure_uncompressed(ds: Any) -> list[AuditEntry]:
    """Decode a compressed transfer syntax in place so pixels can be redacted and re-saved.

    After this, ``ds`` is Explicit VR Little Endian and PixelData is native; a lossy source keeps
    LossyImageCompression = "01". Raises ``RuntimeError`` with an install hint if no codec can decode it.
    """
    from pydicom.uid import ExplicitVRLittleEndian

    audit: list[AuditEntry] = []
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if ts is not None and ts.is_compressed and "PixelData" in ds:
        try:
            ds.decompress(generate_instance_uid=False)
        except Exception as exc:  # pragma: no cover - depends on installed codecs
            raise RuntimeError(
                f"cannot decode compressed DICOM ({ts.name}): {exc}. Install the codecs extra: "
                "pip install prostate-deid[codecs]"
            ) from exc
        audit.append(AuditEntry("dicom_io", "decompressed", f"{ts.name} -> Explicit VR Little Endian"))
        if ts not in _LOSSLESS_TS and getattr(ds, "LossyImageCompression", None) != "01":
            ds.LossyImageCompression = "01"
    if getattr(ds, "file_meta", None) is not None:
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    return audit


def _is_overlay_or_curve(tag) -> bool:
    return 0x6000 <= tag.group <= 0x60FF or 0x5000 <= tag.group <= 0x50FF


def _scrub_level(ds: Any, keep: set[str], counters: dict) -> None:
    """Allowlist one dataset level; recurse into kept sequences with the functional-group allowlist."""
    for elem in list(ds):
        tag = elem.tag
        if tag == 0x7FE00010:
            continue
        if tag.is_private or _is_overlay_or_curve(tag) or elem.keyword not in keep:
            del ds[tag]
            counters["private" if tag.is_private else "removed"] += 1
            continue
        if elem.VR == "SQ":
            for item in elem.value:
                _scrub_level(item, FUNCTIONAL_GROUP_KEEP | TECHNICAL_KEEP, counters)


def _remap_nested(ds: Any, ctx: StudyContext, pid: str | None, top: bool = True) -> None:
    """Remap every instance UID, and shift every DA / DT value, at all levels."""
    for elem in ds:
        if elem.VR == "SQ":
            for item in elem.value:
                _remap_nested(item, ctx, pid, top=False)
        elif elem.VR == "UI" and elem.keyword not in _CLASS_UIDS_NOT_REMAPPED and elem.value:
            if elem.VM > 1:
                elem.value = [ctx.uid(str(v)) for v in elem.value]
            else:
                elem.value = ctx.uid(str(elem.value))
        elif elem.VR == "DA" and elem.value and ctx.shift_dates:
            elem.value = ctx.shift_date(str(elem.value), pid)
        elif elem.VR == "DA" and elem.value:
            elem.value = ""
        elif elem.VR == "DT" and elem.value:
            elem.value = ctx.shift_datetime(str(elem.value), pid) if ctx.shift_dates else ""


def deidentify_dicom(
    ds: Any,
    ctx: StudyContext | None = None,
    keep_mr_tags: bool = True,
    redact_pixels: bool = True,
    text_method: str = "telea",
    **restore_kwargs: Any,
) -> ConversionResult:
    """De-identify one real DICOM dataset (modified in place and returned).

    Order matters: decode -> scrub free text (needs the names it removes) -> rescue diffusion tags
    (needs the private tags) -> allowlist -> pseudonymize UIDs/dates/IDs -> clean pixels ->
    Type 2 attributes -> de-identification marking.
    """
    ctx = ctx or StudyContext()
    audit: list[AuditEntry] = []
    audit += ensure_uncompressed(ds)

    original_pid = str(getattr(ds, "PatientID", "") or "") or None
    fallback = str(getattr(ds, "StudyInstanceUID", "") or "unknown-study")
    pid = ctx.patient_id(original_pid, fallback_key=fallback)

    if keep_mr_tags:
        audit += scrub_free_text(ds)
        audit += rescue_diffusion_tags(ds)
    age = str(getattr(ds, "PatientAge", "") or "")
    if re.fullmatch(r"\d{3}Y", age) and int(age[:3]) > 89:
        ds.PatientAge = "090Y"
        audit.append(AuditEntry("dicom_tag_scrub", "aggregated_age", "PatientAge > 89Y -> 090Y (HIPAA)"))

    keep = set(TECHNICAL_KEEP) | (MR_SAFE_KEEP if keep_mr_tags else set())
    counters = {"removed": 0, "private": 0}
    _scrub_level(ds, keep, counters)
    audit.append(AuditEntry("dicom_tag_scrub", "allowlist_wipe",
                            f"removed {counters['removed']} public + {counters['private']} private/overlay element(s); "
                            f"policy={'technical + MR-safe' if keep_mr_tags else 'technical only (v1)'}"))

    _remap_nested(ds, ctx, original_pid)
    audit.append(AuditEntry("pseudonymize", "remapped_uids_and_dates",
                            "all instance UIDs (incl. FrameOfReferenceUID) keyed-remapped; dates "
                            + ("shifted by a consistent per-patient offset" if ctx.shift_dates else "removed")))
    audit.append(AuditEntry("pseudonymize", "remapped_patient_id", "PatientID/PatientName -> study pseudonym"
                            + (" (per-session key: not re-linkable)" if ctx.session_key else "")))

    restore = None
    if redact_pixels and "PixelData" in ds:
        pixels = np.asarray(ds.pixel_array)
        n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        color = int(getattr(ds, "SamplesPerPixel", 1)) > 1
        stack = pixels if (n_frames > 1) else pixels[np.newaxis, ...]
        if color and stack.ndim == 3:
            stack = stack[np.newaxis, ...]
        new_stack, restore, r_audit = _restore_pixels(stack, text_method, restore_kwargs)
        audit += r_audit
        ds.PixelData = np.ascontiguousarray(new_stack.astype(pixels.dtype).reshape(pixels.shape)).tobytes()
        ocr_ran = not any(e.action == "ocr_unavailable" for e in restore.audit)
        if ocr_ran:
            ds.BurnedInAnnotation = "NO"

    _fill_common_type2(ds, pid)
    mr_type1 = ("ImageType", "ScanningSequence", "SequenceVariant")
    if ds.SOPClassUID == _MR_IMAGE_STORAGE and not all(k in ds and ds[k].value for k in mr_type1):
        # Strict (technical-only) policy removed required MR Image attributes: the object is no longer a
        # valid MR Image, so relabel it as Secondary Capture rather than emit a non-conformant MR file.
        ds.SOPClassUID = _SC_IMAGE_STORAGE
        ds.ConversionType = "WSD"
        ds.PatientOrientation = ""
        audit.append(AuditEntry("dicom_tag_scrub", "relabelled_secondary_capture",
                                "required MR Image attributes removed by the strict policy -> SOP class set to "
                                "Secondary Capture"))
    if "FrameOfReferenceUID" in ds and "PositionReferenceIndicator" not in ds:
        ds.PositionReferenceIndicator = ""
    if ds.SOPClassUID == _SC_IMAGE_STORAGE and "ConversionType" not in ds:
        ds.ConversionType = "WSD"
    _mark_deidentified(ds, "allowlist + MR-safe tags + pixel clean" if keep_mr_tags else "allowlist + pixel clean")

    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    return ConversionResult([ds], audit, restore, [output_filename(ds)])


# ---------------------------------------------------------------------------
# NIfTI -> DICOM series (geometry + intensities preserved)
# ---------------------------------------------------------------------------

def load_nifti(path_or_bytes: Any) -> Any:
    """Load a .nii / .nii.gz from a path or bytes (gzip detected by magic bytes)."""
    import nibabel as nib

    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = bytes(path_or_bytes)
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        fh = nib.FileHolder(fileobj=io.BytesIO(raw))
        return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    return nib.load(str(path_or_bytes))


def _affine_to_dicom_geometry(affine_ras: np.ndarray) -> dict[str, Any]:
    """RAS voxel->world affine -> DICOM IOP / PixelSpacing / slice step, all in LPS."""
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(affine_ras, dtype=np.float64)
    col_vec, row_vec, slc_vec = lps[:3, 0], lps[:3, 1], lps[:3, 2]  # i -> columns, j -> rows, k -> slices
    dx, dy, dz = (float(np.linalg.norm(v)) or 1.0 for v in (col_vec, row_vec, slc_vec))
    row_cos, col_cos = col_vec / dx, row_vec / dy  # DICOM IOP = [along a row (i), along a column (j)]
    shear = abs(float(np.dot(row_cos, col_cos)))
    return {"lps": lps, "iop": [*row_cos.tolist(), *col_cos.tolist()], "pixel_spacing": [dy, dx],
            "slice_step": dz, "shear": shear}


def nifti_to_dicom_series(
    nifti: Any,
    ctx: StudyContext | None = None,
    modality: str = "MR",
    series_description: str = "",
    series_number: int = 1,
    volume_index: int = 0,
    source_key: str = "nifti",
    redact_pixels: bool = True,
    text_method: str = "telea",
    **restore_kwargs: Any,
) -> ConversionResult:
    """Convert a NIfTI volume to a DICOM series (one file per slice) with geometry and values kept.

    ``nifti``: a nibabel image, a path, or raw bytes. ``modality="MR"`` writes MR Image Storage
    (Type 1 MR attributes set to neutral derived values); anything else writes Secondary Capture
    with Image Plane attributes. Raw stored integers and the NIfTI scl_slope/scl_inter are carried
    over exactly as RescaleSlope/Intercept.
    """
    ctx = ctx or StudyContext()
    img = nifti if hasattr(nifti, "affine") else load_nifti(nifti)
    audit: list[AuditEntry] = []

    raw = np.asanyarray(img.dataobj.get_unscaled()) if hasattr(img.dataobj, "get_unscaled") else np.asanyarray(img.dataobj)
    # nibabel moves scl_slope/scl_inter onto the array proxy at load time (the header then reports
    # (None, None)), so read them from the proxy first.
    slope, inter = getattr(img.dataobj, "slope", None), getattr(img.dataobj, "inter", None)
    if slope is None:
        slope, inter = img.header.get_slope_inter()
    slope = 1.0 if slope is None or not np.isfinite(slope) else float(slope)
    inter = 0.0 if inter is None or not np.isfinite(inter) else float(inter)
    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]
    if raw.ndim > 3:
        n_vol = int(np.prod(raw.shape[3:]))
        raw = raw.reshape(*raw.shape[:3], n_vol)[..., min(volume_index, n_vol - 1)]
        audit.append(AuditEntry("nifti_io", "selected_volume",
                                f"4D input: wrote volume {min(volume_index, n_vol - 1)} of {n_vol}", flagged_for_review=True))

    # NIfTI data[i, j, k] -> DICOM slice k with rows = j, columns = i  (fixes v1's transposed slices)
    vol_zyx = np.ascontiguousarray(np.transpose(raw, (2, 1, 0)))

    if np.issubdtype(vol_zyx.dtype, np.integer) and (slope, inter) != (1.0, 0.0):
        stored_src, r_slope, r_inter = vol_zyx, slope, inter  # raw ints + header scaling: exact
    else:
        stored_src, r_slope, r_inter = (vol_zyx * slope + inter) if (slope, inter) != (1.0, 0.0) else vol_zyx, 1.0, 0.0

    restore = None
    if redact_pixels:
        stored_src, restore, r_audit = _restore_pixels(stored_src, text_method, restore_kwargs)
        audit += r_audit

    stored, q_slope, q_inter, lossless, err = _choose_integer_storage(stored_src)
    r_slope, r_inter = r_slope * q_slope, r_inter + r_slope * q_inter
    audit.append(AuditEntry(
        "intensity", "preserved_values" if lossless else "quantised_values",
        f"stored as {stored.dtype} with RescaleSlope={r_slope:.6g}, RescaleIntercept={r_inter:.6g}"
        + ("" if lossless else f"; max abs quantisation error {err:.3g} (non-integer float input)"),
        flagged_for_review=not lossless,
    ))

    geo = _affine_to_dicom_geometry(img.affine)
    if geo["shear"] > 1e-3:
        audit.append(AuditEntry("geometry", "sheared_affine",
                                f"row/column axes not orthogonal (|cos|={geo['shear']:.3g}); IOP is approximate",
                                flagged_for_review=True))
    pid = ctx.patient_id(None, fallback_key=source_key)
    study_uid = ctx.synth_uid(source_key, "study")
    series_uid = ctx.synth_uid(source_key, "series", str(series_number), str(volume_index))
    for_uid = ctx.synth_uid(source_key, "frame_of_reference")
    sop_class = _MR_IMAGE_STORAGE if modality == "MR" else _SC_IMAGE_STORAGE

    datasets, names = [], []
    lps = geo["lps"]
    for k in range(stored.shape[0]):
        sop_uid = ctx.synth_uid(source_key, "sop", str(series_number), str(volume_index), str(k))
        ds = _new_dataset(sop_class, sop_uid)
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study_uid, series_uid, for_uid
        ds.PositionReferenceIndicator = ""
        ds.Modality = modality if modality != "SC" else "OT"
        ds.SeriesNumber, ds.InstanceNumber = int(series_number), k + 1
        ds.SeriesDescription = series_description or "converted from NIfTI"
        ipp = lps @ np.array([0.0, 0.0, float(k), 1.0])
        ds.ImagePositionPatient = [round(float(v), 6) for v in ipp[:3]]
        ds.ImageOrientationPatient = [round(float(v), 8) for v in geo["iop"]]
        ds.PixelSpacing = [round(float(v), 6) for v in geo["pixel_spacing"]]
        ds.SliceThickness = round(float(geo["slice_step"]), 6)
        ds.SpacingBetweenSlices = round(float(geo["slice_step"]), 6)
        normal = np.cross(np.array(geo["iop"][:3]), np.array(geo["iop"][3:]))
        ds.SliceLocation = round(float(np.dot(normal, ipp[:3])), 6)
        ds.RescaleSlope, ds.RescaleIntercept = f"{r_slope:.10g}", f"{r_inter:.10g}"
        ds.RescaleType = "US"
        ds.ImageType = ["DERIVED", "SECONDARY"]
        if sop_class == _MR_IMAGE_STORAGE:
            ds.ScanningSequence, ds.SequenceVariant = "RM", "NONE"
            for kw in ("ScanOptions", "MRAcquisitionType", "RepetitionTime", "EchoTime", "EchoTrainLength"):
                setattr(ds, kw, "")
        else:
            ds.ConversionType = "WSD"
            ds.PatientOrientation = ""
        _fill_common_type2(ds, pid)
        _set_pixel_module(ds, stored[k])
        ds.BurnedInAnnotation = "NO" if redact_pixels and restore is not None and not any(
            e.action == "ocr_unavailable" for e in restore.audit) else ""
        _mark_deidentified(ds, "synthesized from NIfTI; no identifying fields set")
        datasets.append(ds)
        names.append(output_filename(ds))

    audit.append(AuditEntry("nifti_io", "nifti_to_dicom",
                            f"{len(datasets)} slice(s) -> {'MR' if sop_class == _MR_IMAGE_STORAGE else 'SC'} series; "
                            f"PixelSpacing={geo['pixel_spacing']}, slice step={geo['slice_step']:.4g} mm, "
                            "IPP/IOP from affine (RAS->LPS)"))
    return ConversionResult(datasets, audit, restore, names)


# ---------------------------------------------------------------------------
# Raster -> DICOM Secondary Capture
# ---------------------------------------------------------------------------

def _raster_to_array(data: bytes) -> tuple[np.ndarray, list[AuditEntry]]:
    from PIL import Image, ImageOps

    audit: list[AuditEntry] = []
    im = Image.open(io.BytesIO(data))
    im.load()
    carried = [k for k in ("exif", "xmp", "XML:com.adobe.xmp", "icc_profile", "comment", "dpi") if k in im.info]
    if im.getexif():
        carried.append("EXIF tags")
    if carried:
        audit.append(AuditEntry("raster_io", "stripped_metadata",
                                f"not carried into DICOM: {sorted(set(carried))} (only pixels are used)"))
    orientation = im.getexif().get(0x0112, 1)
    if orientation != 1:
        im = ImageOps.exif_transpose(im)
        audit.append(AuditEntry("raster_io", "applied_exif_orientation",
                                f"EXIF Orientation={orientation} applied to pixels before the tag was dropped"))

    mode = im.mode
    if mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        audit.append(AuditEntry("raster_io", "expanded_palette", "palette indices -> colours"))
        mode = im.mode
    if mode in ("LA", "RGBA", "PA"):
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32)
        alpha = rgba[..., 3:4] / 255.0
        comp = rgba[..., :3] * alpha  # composite on black, the DICOM display background
        im_arr = np.rint(comp).astype(np.uint8)
        if mode == "LA" or np.array_equal(im_arr[..., 0], im_arr[..., 1]) and np.array_equal(im_arr[..., 1], im_arr[..., 2]):
            im_arr = im_arr[..., 0]
        audit.append(AuditEntry("raster_io", "composited_alpha", f"{mode}: alpha composited onto black, alpha dropped"))
        arr = im_arr
    elif mode == "1":
        arr = np.asarray(im.convert("L"))
    elif mode in ("I;16", "I;16B", "I;16L"):
        arr = np.asarray(im).astype(np.uint16)
    elif mode in ("I", "F"):
        arr = np.asarray(im)
    elif mode == "CMYK" or mode == "YCbCr" or mode == "RGB":
        arr = np.asarray(im.convert("RGB"))
    else:
        arr = np.asarray(im.convert("L"))
    if arr.ndim == 3 and arr.shape[-1] == 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
        arr = arr[..., 0]
        audit.append(AuditEntry("raster_io", "gray_rgb_to_mono", "R=G=B image stored as MONOCHROME2"))
    return arr, audit


def raster_to_dicom(
    data: bytes,
    ctx: StudyContext | None = None,
    modality: str = "OT",
    source_key: str = "raster",
    instance_number: int = 1,
    series_number: int = 1,
    redact_pixels: bool = True,
    text_method: str = "telea",
    **restore_kwargs: Any,
) -> ConversionResult:
    """PNG / JPEG / BMP / TIFF bytes -> one Secondary Capture DICOM (values not rescaled)."""
    ctx = ctx or StudyContext()
    arr, audit = _raster_to_array(data)
    restore = None
    if redact_pixels:
        stack, restore, r_audit = _restore_pixels(arr[np.newaxis, ...], text_method, restore_kwargs)
        audit += r_audit
        arr = stack[0]
    if arr.dtype not in (np.uint8, np.uint16, np.int16):
        arr, sl, ic, lossless, err = _choose_integer_storage(arr)
        rescale = (sl, ic)
        audit.append(AuditEntry("intensity", "preserved_values" if lossless else "quantised_values",
                                f"float raster stored as {arr.dtype}" + ("" if lossless else f", max error {err:.3g}"),
                                flagged_for_review=not lossless))
    else:
        rescale = (1.0, 0.0)

    pid = ctx.patient_id(None, fallback_key=source_key)
    sop_uid = ctx.synth_uid(source_key, "sop", str(series_number), str(instance_number))
    ds = _new_dataset(_SC_IMAGE_STORAGE, sop_uid)
    ds.StudyInstanceUID = ctx.synth_uid(source_key, "study")
    ds.SeriesInstanceUID = ctx.synth_uid(source_key, "series", str(series_number))
    ds.Modality = modality
    ds.ConversionType = "WSD"
    ds.SeriesNumber, ds.InstanceNumber = int(series_number), int(instance_number)
    ds.PatientOrientation = ""
    ds.ImageType = ["DERIVED", "SECONDARY"]
    if arr.ndim == 2 and rescale != (1.0, 0.0):
        ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = f"{rescale[0]:.10g}", f"{rescale[1]:.10g}", "US"
    _fill_common_type2(ds, pid)
    _set_pixel_module(ds, arr)
    ds.BurnedInAnnotation = "NO" if redact_pixels and restore is not None and not any(
        e.action == "ocr_unavailable" for e in restore.audit) else ""
    _mark_deidentified(ds, "synthesized from raster; no identifying fields set")
    audit.append(AuditEntry("raster_io", "raster_to_dicom", f"{arr.shape} {arr.dtype} -> Secondary Capture"))
    return ConversionResult([ds], audit, restore, [output_filename(ds)])


# ---------------------------------------------------------------------------
# DICOM series -> .nii.gz (full affine, exact values, sidecar)
# ---------------------------------------------------------------------------

def _frame_geometry(ds: Any) -> list[tuple[np.ndarray | None, np.ndarray | None, list[float] | None, float | None]]:
    """Per-frame (ipp, iop, pixel_spacing, b_value) for classic single-frame or enhanced multi-frame."""
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)
    shared = getattr(ds, "SharedFunctionalGroupsSequence", [None])[0] if "SharedFunctionalGroupsSequence" in ds else None
    per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)

    def find(item, seq_kw, kw):
        if item is not None and seq_kw in item and item[seq_kw].value and kw in item[seq_kw].value[0]:
            return item[seq_kw].value[0][kw].value
        return None

    out = []
    for f in range(n):
        pf = per[f] if per is not None and f < len(per) else None
        ipp = find(pf, "PlanePositionSequence", "ImagePositionPatient") or getattr(ds, "ImagePositionPatient", None)
        iop = (find(pf, "PlaneOrientationSequence", "ImageOrientationPatient")
               or find(shared, "PlaneOrientationSequence", "ImageOrientationPatient")
               or getattr(ds, "ImageOrientationPatient", None))
        ps = (find(pf, "PixelMeasuresSequence", "PixelSpacing") or find(shared, "PixelMeasuresSequence", "PixelSpacing")
              or getattr(ds, "PixelSpacing", None))
        b = find(pf, "MRDiffusionSequence", "DiffusionBValue")
        if b is None:
            b = read_bvalue(ds)[0]
        out.append((
            np.array([float(v) for v in ipp]) if ipp is not None else None,
            np.array([float(v) for v in iop]) if iop is not None else None,
            [float(v) for v in ps] if ps is not None else None,
            None if b is None else float(b),
        ))
    return out


def series_to_nifti(
    datasets: list[Any],
    normalize: Any = None,
) -> tuple[bytes, dict[str, Any], list[AuditEntry]]:
    """Encode one (de-identified) DICOM series as ``.nii.gz`` bytes + a JSON-able sidecar dict.

    * Slices are sorted along the slice normal; the affine is built from IOP / IPP / PixelSpacing
      (LPS -> RAS), so oblique prostate acquisitions keep their orientation in world space.
    * Several images at the same position (e.g. multiple b-values in one DWI series) become a 4D
      NIfTI, with the b-values listed in the sidecar.
    * Stored integers + a shared RescaleSlope/Intercept are written exactly via scl_slope/scl_inter.
    * Every slice position is re-derived from the affine and compared with its source IPP
      (``orientation_check_max_error_mm`` in the sidecar).
    * ``normalize``: optional :class:`~prostate_deid.preprocess.IntensityNormalize` (RARN) applied
      to the float volume before writing; its parameters are logged.
    """
    import nibabel as nib

    audit: list[AuditEntry] = []
    frames = []  # (ipp, iop, ps, b, pixels2d, slope, inter, instance_number)
    for ds in datasets:
        pix = np.asarray(ds.pixel_array)
        n = int(getattr(ds, "NumberOfFrames", 1) or 1)
        stack = pix if n > 1 else pix[np.newaxis, ...]
        if stack.ndim == 4:
            raise ValueError("colour (RGB) series cannot be written as a scalar NIfTI volume")
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
        for f_idx, ((ipp, iop, ps, b), fr) in enumerate(zip(_frame_geometry(ds), stack)):
            frames.append((ipp, iop, ps, b, fr, slope, inter, inst * 10000 + f_idx))
    if not frames:
        raise ValueError("no image frames")

    first = frames[0]
    has_geo = first[0] is not None and first[1] is not None
    if has_geo:
        iop0 = first[1]
        assert iop0 is not None  # has_geo
        mixed = [f for f in frames if f[1] is None or np.max(np.abs(f[1] - iop0)) > 1e-3]
        if mixed:
            raise ValueError(f"{len(mixed)} frame(s) have a different ImageOrientationPatient (e.g. localizers); "
                             "split the series by orientation before converting")
        row_cos, col_cos = iop0[:3], iop0[3:]
        normal = np.cross(row_cos, col_cos)
        dist = np.array([float(np.dot(normal, f[0])) for f in frames])
    else:
        row_cos, col_cos, normal = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
        dist = np.arange(len(frames), dtype=float)
        audit.append(AuditEntry("geometry", "no_patient_geometry",
                                "source has no IPP/IOP (e.g. Secondary Capture): identity orientation used",
                                flagged_for_review=True))

    # Group frames by position; multiple frames per position -> 4D (b-values / repetitions).
    order = np.argsort(dist, kind="stable")
    positions: list[list[int]] = []
    for idx in order:
        if positions and abs(dist[idx] - dist[positions[-1][0]]) < 1e-3:
            positions[-1].append(idx)
        else:
            positions.append([idx])
    # Keep acquisition order when it runs against the slice normal (so k follows InstanceNumber and
    # a NIfTI -> DICOM -> NIfTI round trip returns the identical array; the affine absorbs the sign).
    if len(positions) > 1:
        firsts = [frames[p[0]][7] for p in positions]
        if all(b < a for a, b in zip(firsts, firsts[1:])):
            positions = positions[::-1]
    n_vol = max(len(p) for p in positions)
    if any(len(p) != n_vol for p in positions):
        raise ValueError("unequal number of images per slice position; cannot form a regular 4D volume")
    if n_vol > 1:  # order each position's frames by b-value (then original order) so volume v is consistent
        positions = [sorted(p, key=lambda i: (frames[i][3] if frames[i][3] is not None else 0.0, i)) for p in positions]

    ps = first[2] or [1.0, 1.0]
    row_sp, col_sp = float(ps[0]), float(ps[1])
    zs = np.array([dist[p[0]] for p in positions])
    if len(zs) > 1:
        steps = np.diff(zs)
        step = float(np.median(steps))
        dev = float(np.max(np.abs(steps - step)))
        if step and dev > 0.01 * abs(step):
            audit.append(AuditEntry("geometry", "non_uniform_spacing",
                                    f"slice spacing varies by up to {dev:.3g} mm (median {step:.4g}); NIfTI assumes "
                                    "uniform spacing -- slice positions are approximate", flagged_for_review=True))
    else:
        step = float(getattr(datasets[0], "SliceThickness", 1.0) or 1.0)

    if has_geo:
        ipp0 = frames[positions[0][0]][0]
        ipp_last = frames[positions[-1][0]][0]
        assert ipp0 is not None and ipp_last is not None  # has_geo: every frame has a position
        slice_vec = (ipp_last - ipp0) / (len(positions) - 1) if len(positions) > 1 else normal * step
    else:
        ipp0, slice_vec = np.zeros(3), normal * step
    lps = np.eye(4)
    lps[:3, 0] = row_cos * col_sp  # i: along a row (column index)
    lps[:3, 1] = col_cos * row_sp  # j: down a column (row index)
    lps[:3, 2] = slice_vec
    lps[:3, 3] = ipp0
    ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps

    # Volume (Z, Y, X[, V]) -> NIfTI (X, Y, Z[, V])
    vols = []
    for v in range(n_vol):
        vols.append(np.stack([frames[p[v]][4] for p in positions], axis=0))
    data = np.stack(vols, axis=-1) if n_vol > 1 else vols[0]
    slopes = {(frames[i][5], frames[i][6]) for p in positions for i in p}
    exact = len(slopes) == 1 and np.issubdtype(data.dtype, np.integer)
    if exact:
        slope, inter = slopes.pop()
        raw = data
    else:
        scale = np.zeros(data.shape[:1] + ((n_vol,) if n_vol > 1 else ()), dtype=np.float64)
        raw = data.astype(np.float32)
        for zi, p in enumerate(positions):
            for v, i in enumerate(p):
                sl = (zi, Ellipsis, v) if n_vol > 1 else (zi,)
                raw[sl] = raw[sl] * frames[i][5] + frames[i][6]
        del scale
        slope, inter = 1.0, 0.0
        audit.append(AuditEntry("intensity", "applied_per_slice_rescale",
                                "slices differ in RescaleSlope/Intercept: values rescaled to float32"))

    if normalize is not None and n_vol == 1:
        from .core import Sample

        vol_f = (raw.astype(np.float32) * slope + inter) if exact else raw.astype(np.float32)
        smp = normalize(Sample(image=vol_f[np.newaxis, ...]))  # CZYX
        raw = smp.image[0].astype(np.float32)
        slope, inter, exact = 1.0, 0.0, False
        audit += smp.audit
    elif normalize is not None:
        audit.append(AuditEntry("intensity_normalize", "skipped",
                                "RARN normalization is defined per 3D volume; 4D (multi-b) output left unnormalized",
                                flagged_for_review=True))

    xyz = np.transpose(raw, (2, 1, 0) + ((3,) if raw.ndim == 4 else ()))
    img = nib.Nifti1Image(np.ascontiguousarray(xyz), ras)
    img.header.set_xyzt_units("mm", "sec")
    img.set_sform(ras, code=1)
    img.set_qform(ras, code=1)
    if exact and (slope, inter) != (1.0, 0.0):
        img.header.set_slope_inter(slope, inter)
    buf = io.BytesIO()
    fmap = img.make_file_map({"image": buf, "header": buf})
    img.to_file_map(fmap)
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0) as fh:
        fh.write(buf.getvalue())

    # Orientation self-check: affine applied to (0,0,k) must land on each slice's own IPP.
    max_err = 0.0
    if has_geo:
        for zi, p in enumerate(positions):
            world = ras @ np.array([0.0, 0.0, float(zi), 1.0])
            src = frames[p[0]][0] * np.array([-1.0, -1.0, 1.0])
            max_err = max(max_err, float(np.linalg.norm(world[:3] - src)))
        audit.append(AuditEntry("geometry", "orientation_self_check",
                                f"max slice-position error affine vs source IPP = {max_err:.4f} mm",
                                flagged_for_review=max_err > 0.1))

    ref = datasets[0]
    bvals = [frames[p][3] for p in positions[0]] if n_vol > 1 else [frames[positions[0][0]][3]]
    sidecar = {
        "ConversionSoftware": SOFTWARE_NAME,
        "Modality": str(getattr(ref, "Modality", "")),
        "Manufacturer": str(getattr(ref, "Manufacturer", "")),
        "ManufacturerModelName": str(getattr(ref, "ManufacturerModelName", "")),
        "MagneticFieldStrength": _num(getattr(ref, "MagneticFieldStrength", None)),
        "EchoTime": _sec(getattr(ref, "EchoTime", None)),
        "RepetitionTime": _sec(getattr(ref, "RepetitionTime", None)),
        "FlipAngle": _num(getattr(ref, "FlipAngle", None)),
        "SeriesDescription": str(getattr(ref, "SeriesDescription", "")),
        "ProtocolName": str(getattr(ref, "ProtocolName", "")),
        "ImageType": [str(x) for x in getattr(ref, "ImageType", [])],
        "SeriesNumber": _num(getattr(ref, "SeriesNumber", None)),
        "VoxelSize_mm": [col_sp, row_sp, float(np.linalg.norm(slice_vec))],
        "Shape": list(xyz.shape),
        "DiffusionBValues": bvals if any(b is not None for b in bvals) else None,
        "ScaledExactly": bool(exact),
        "orientation_check_max_error_mm": max_err if has_geo else None,
        "PatientIdentityRemoved": str(getattr(ref, "PatientIdentityRemoved", "")),
        "DeidentificationMethod": str(getattr(ref, "DeidentificationMethod", "")),
    }
    audit.append(AuditEntry("nifti_io", "dicom_to_nifti",
                            f"{len(frames)} frame(s) -> .nii.gz {list(xyz.shape)}, voxel {sidecar['VoxelSize_mm']} mm, "
                            f"{'exact stored values' if exact else 'float32'}"))
    return gz.getvalue(), sidecar, audit


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _sec(v):
    x = _num(v)
    return None if x is None else x / 1000.0


# ---------------------------------------------------------------------------
# Conformance self-check and safe filenames
# ---------------------------------------------------------------------------

_REQ_COMMON = {
    1: ["SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "Modality", "Rows", "Columns",
        "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation", "SamplesPerPixel",
        "PhotometricInterpretation", "PixelData"],
    2: ["PatientName", "PatientID", "PatientBirthDate", "PatientSex", "StudyDate", "StudyTime",
        "ReferringPhysicianName", "StudyID", "AccessionNumber", "SeriesNumber", "InstanceNumber", "Manufacturer"],
}
_REQ_BY_CLASS = {
    _MR_IMAGE_STORAGE: {1: ["ImageType", "ScanningSequence", "SequenceVariant", "FrameOfReferenceUID",
                            "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"],
                        2: ["ScanOptions", "MRAcquisitionType", "EchoTime", "EchoTrainLength", "SliceThickness",
                            "PositionReferenceIndicator"]},
    _SC_IMAGE_STORAGE: {1: ["ConversionType"], 2: ["PatientOrientation"]},
    _MF_GRAY_BYTE_SC: {1: ["NumberOfFrames", "FrameIncrementPointer"], 2: []},
    _MF_GRAY_WORD_SC: {1: ["NumberOfFrames", "FrameIncrementPointer"], 2: []},
}


def check_conformance(ds: Any) -> list[str]:
    """Return a list of problems (empty = passes) for the IOD modules this converter writes.

    A self-check of Type 1 (present, non-empty) and Type 2 (present) attributes plus
    pixel-data length; not a substitute for a full validator such as dciodvfy.
    """
    problems = []
    req = {1: list(_REQ_COMMON[1]), 2: list(_REQ_COMMON[2])}
    extra = _REQ_BY_CLASS.get(str(getattr(ds, "SOPClassUID", "")), {})
    for t in (1, 2):
        req[t] += extra.get(t, [])
    for kw in req[1]:
        if kw not in ds or ds[kw].value in (None, "", b""):
            problems.append(f"Type 1 {kw} missing or empty")
    for kw in req[2]:
        if kw not in ds:
            problems.append(f"Type 2 {kw} missing")
    if "PixelData" in ds and "Rows" in ds and "Columns" in ds and "BitsAllocated" in ds:
        n = int(getattr(ds, "NumberOfFrames", 1) or 1)
        expected = ds.Rows * ds.Columns * int(ds.SamplesPerPixel) * n * (ds.BitsAllocated // 8)
        got = len(ds.PixelData)
        if got not in (expected, expected + 1):
            problems.append(f"PixelData length {got} != Rows*Columns*Samples*Frames*Bytes = {expected}")
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if ts is not None and ts.is_compressed:
        problems.append(f"transfer syntax {ts.name} is compressed but pixel data was rewritten natively")
    return problems


def output_filename(ds: Any) -> str:
    """PHI-safe file name from the pseudonymous ID and series/instance numbers -- never the upload name."""
    pid = re.sub(r"[^A-Za-z0-9\-]", "", str(getattr(ds, "PatientID", "") or "anon")) or "anon"
    sn = int(getattr(ds, "SeriesNumber", 0) or 0)
    inum = int(getattr(ds, "InstanceNumber", 0) or 0)
    tag = hashlib.sha1(str(getattr(ds, "SOPInstanceUID", "")).encode()).hexdigest()[:6]
    return f"{pid}_S{sn:03d}_I{inum:04d}_{tag}.dcm"
