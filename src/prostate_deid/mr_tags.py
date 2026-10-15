"""
mpMRI-aware header policy for prostate MRI (T2W, DWI, ADC).

v1's ``full_wipe_dicom`` keeps ~22 technical tags. That is safe, but it also
deletes everything that tells T2W, DWI and ADC apart (ImageType,
SeriesDescription, ProtocolName), the acquisition parameters (TE, TR,
b-values) and the frame of reference that lets the three series be
co-registered -- and with them most of the research value of the data.

This module defines what a de-identified MR file may keep, as an explicit
allowlist (unknown and private tags are still removed by default), plus two
steps that must run *before* the private tags are deleted:

* :func:`rescue_diffusion_tags` copies vendor-private diffusion b-values and
  gradient directions (Siemens, GE, Philips) into the public DICOM tags
  (0018,9087) DiffusionBValue / (0018,9089) DiffusionGradientOrientation, so
  ADC can still be recomputed after every private tag is gone.
* :func:`scrub_free_text` cleans SeriesDescription / ProtocolName /
  StudyDescription instead of deleting them: the sequence label survives,
  person names, IDs and dates do not.

Vendor tag locations follow the QIN / NA-MIC diffusion tag surveys cited in
docs/ (Siemens (0019,100C)/(0019,100E) in "SIEMENS MR HEADER"; GE
(0043,1039) in "GEMS_PARM_01" with the occasional +1e9 offset and
(0019,10BB-BD) in "GEMS_ACQU_01"; Philips public (0018,9087)/(0018,9089) or
private (2001,1003) in "Philips Imaging DD 001").
"""

from __future__ import annotations

import re
import struct
from typing import Any

from .core import AuditEntry

__all__ = [
    "TECHNICAL_KEEP",
    "MR_SAFE_KEEP",
    "FUNCTIONAL_GROUP_KEEP",
    "FREE_TEXT_SCRUB",
    "rescue_diffusion_tags",
    "scrub_free_text",
    "read_bvalue",
]

# Pixel description + identity of the object itself (same as v1's _TECHNICAL_KEEP).
TECHNICAL_KEEP = {
    "SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID",
    "Modality", "Rows", "Columns", "BitsAllocated", "BitsStored", "HighBit",
    "PixelRepresentation", "SamplesPerPixel", "PhotometricInterpretation",
    "PlanarConfiguration", "NumberOfFrames", "TransferSyntaxUID",
    "RescaleIntercept", "RescaleSlope", "RescaleType", "PixelSpacing", "SliceThickness",
    "ImageOrientationPatient", "ImagePositionPatient", "SpecificCharacterSet",
    "LossyImageCompression", "LossyImageCompressionRatio", "LossyImageCompressionMethod",
    "WindowCenter", "WindowWidth", "VOILUTFunction",
}

# Non-identifying content that makes prostate mpMRI usable: series identity, geometry,
# acquisition and diffusion parameters. Dates are kept only because they are shifted by the
# study pseudonymizer (see convert.StudyContext); UIDs are remapped consistently.
MR_SAFE_KEEP = {
    # series / image identity (T2W vs DWI vs ADC)
    "ImageType", "SeriesNumber", "InstanceNumber", "AcquisitionNumber", "SequenceName",
    "SeriesDescription", "ProtocolName", "StudyDescription",  # kept only after scrub_free_text
    "BodyPartExamined", "PatientPosition",
    # geometry / co-registration
    "FrameOfReferenceUID", "SliceLocation", "SpacingBetweenSlices", "PositionReferenceIndicator",
    # MR acquisition
    "ScanningSequence", "SequenceVariant", "ScanOptions", "MRAcquisitionType", "RepetitionTime",
    "EchoTime", "InversionTime", "NumberOfAverages", "ImagingFrequency", "ImagedNucleus",
    "EchoNumbers", "MagneticFieldStrength", "NumberOfPhaseEncodingSteps", "EchoTrainLength",
    "PercentSampling", "PercentPhaseFieldOfView", "PixelBandwidth", "AcquisitionMatrix",
    "InPlanePhaseEncodingDirection", "FlipAngle", "VariableFlipAngleFlag", "SAR", "dBdt",
    "ParallelAcquisitionTechnique", "ParallelReductionFactorInPlane",
    # diffusion (public tags, also the destination of rescue_diffusion_tags)
    "DiffusionBValue", "DiffusionGradientOrientation", "DiffusionDirectionality",
    "DiffusionBMatrixSequence",
    # equipment class, not the individual device (no serial number / station / institution)
    "Manufacturer", "ManufacturerModelName",
    # dates/times (dates are shifted; see convert.StudyContext)
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate",
    "StudyTime", "SeriesTime", "AcquisitionTime", "ContentTime",
    "PatientSex", "PatientAge",
    # enhanced multi-frame geometry lives in functional-group sequences (scrubbed recursively)
    "SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence",
    "DimensionOrganizationSequence", "DimensionIndexSequence", "DimensionOrganizationType",
    "BurnedInAnnotation",
}

# Keywords allowed INSIDE functional-group sequence items (enhanced MR geometry, timing, diffusion).
FUNCTIONAL_GROUP_KEEP = {
    "PlanePositionSequence", "PlaneOrientationSequence", "PixelMeasuresSequence",
    "FrameContentSequence", "MRDiffusionSequence", "PixelValueTransformationSequence",
    "FrameVOILUTSequence", "MREchoSequence", "MRTimingAndRelatedParametersSequence",
    "MRModifierSequence", "MRImageFrameTypeSequence", "MRAveragesSequence",
    "DiffusionGradientDirectionSequence", "MRFOVGeometrySequence",
    "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing", "SliceThickness",
    "SpacingBetweenSlices", "RescaleIntercept", "RescaleSlope", "RescaleType",
    "WindowCenter", "WindowWidth", "FrameType", "DimensionIndexValues", "StackID",
    "InStackPositionNumber", "TemporalPositionIndex", "FrameAcquisitionNumber",
    "FrameAcquisitionDateTime", "FrameReferenceDateTime", "FrameAcquisitionDuration",
    "DiffusionBValue", "DiffusionGradientOrientation", "DiffusionDirectionality",
    "EffectiveEchoTime", "RepetitionTime", "FlipAngle", "EchoTrainLength",
    "DimensionOrganizationUID", "DimensionIndexPointer", "FunctionalGroupPointer",
    "DimensionDescriptionLabel",
}

FREE_TEXT_SCRUB = ("SeriesDescription", "ProtocolName", "StudyDescription")


# ---------------------------------------------------------------------------
# Diffusion rescue
# ---------------------------------------------------------------------------

def _decode_number_list(value: Any) -> list[float]:
    """Decode IS/DS/FD values that may arrive as bytes (VR UN in implicit-VR files)."""
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        try:
            txt = raw.decode("ascii").strip("\x00 ")
            if txt and re.fullmatch(r"[\s\d\.\-+eE\\]+", txt):
                return [float(v) for v in txt.split("\\") if v.strip()]
        except UnicodeDecodeError:
            pass
        if len(raw) % 8 == 0 and raw:
            return list(struct.unpack(f"<{len(raw) // 8}d", raw))
        return []
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, str):
        out = []
        for v in value:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                pass
        return out
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _private_value(ds: Any, group: int, creator: str, offset: int) -> Any:
    try:
        block = ds.private_block(group, creator)
        return block[offset].value
    except (KeyError, ValueError, AttributeError):
        return None


def read_bvalue(ds: Any) -> tuple[float | None, list[float] | None, str]:
    """Return ``(b_value, gradient_direction, source)`` from public or vendor-private tags."""
    b = getattr(ds, "DiffusionBValue", None)
    g = getattr(ds, "DiffusionGradientOrientation", None)
    if b is not None:
        return float(b), ([float(x) for x in g] if g is not None else None), "public"

    manu = str(getattr(ds, "Manufacturer", "")).upper()
    # Siemens
    sb = _private_value(ds, 0x0019, "SIEMENS MR HEADER", 0x0C)
    if sb is not None or "SIEMENS" in manu:
        vals = _decode_number_list(sb)
        if vals:
            gv = _decode_number_list(_private_value(ds, 0x0019, "SIEMENS MR HEADER", 0x0E))
            return vals[0], (gv[:3] if len(gv) >= 3 else None), "siemens (0019,100C)"
    # GE
    gb = _private_value(ds, 0x0043, "GEMS_PARM_01", 0x39)
    if gb is not None or "GE" in manu.split():
        vals = _decode_number_list(gb)
        if vals:
            bval = vals[0]
            if bval >= 1e9:  # some GE versions store b + 1e9
                bval -= 1e9
            comps = [_decode_number_list(_private_value(ds, 0x0019, "GEMS_ACQU_01", off)) for off in (0xBB, 0xBC, 0xBD)]
            gdir = [c[0] for c in comps] if all(comps) else None
            return bval, gdir, "ge (0043,1039)"
    # Philips private fallback
    pb = _private_value(ds, 0x2001, "Philips Imaging DD 001", 0x03)
    if pb is not None:
        vals = _decode_number_list(pb)
        if vals:
            return vals[0], None, "philips (2001,1003)"
    return None, None, "none"


def rescue_diffusion_tags(ds: Any) -> list[AuditEntry]:
    """Copy vendor-private diffusion parameters to public tags (before private tags are stripped)."""
    audit: list[AuditEntry] = []
    if getattr(ds, "DiffusionBValue", None) is not None:
        return audit
    b, g, source = read_bvalue(ds)
    if b is None:
        return audit
    ds.DiffusionBValue = float(b)
    detail = f"DiffusionBValue={b:g} copied from {source}"
    if g is not None and len(g) == 3:
        ds.DiffusionGradientOrientation = [float(x) for x in g]
        detail += "; DiffusionGradientOrientation copied"
        ds.DiffusionDirectionality = "DIRECTIONAL" if any(abs(x) > 1e-6 for x in g) else "ISOTROPIC"
    elif b == 0:
        ds.DiffusionDirectionality = "NONE"
    audit.append(AuditEntry("diffusion_rescue", "copied_private_to_public", detail))
    return audit


# ---------------------------------------------------------------------------
# Free-text scrubbing
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"(?<!\d)(19|20)\d{2}[-./]?(0[1-9]|1[0-2])[-./]?(0[1-9]|[12]\d|3[01])(?!\d)"
                      r"|(?<!\d)(0[1-9]|[12]\d|3[01])[-./](0[1-9]|1[0-2])[-./](19|20)?\d{2}(?!\d)")
_LONG_DIGITS_RE = re.compile(r"\d{6,}")
_PN_RE = re.compile(r"\b[A-Za-z][A-Za-z'\-]+\^[A-Za-z][A-Za-z'\-^]*")  # DICOM PN-style "Doe^Jane"
_TITLE_NAME_RE = re.compile(r"\b(Dr|Prof|Mr|Mrs|Ms)\.?\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?")


def _name_tokens(ds: Any) -> set[str]:
    toks: set[str] = set()
    for kw in ("PatientName", "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
               "InstitutionName", "PhysiciansOfRecord", "OtherPatientNames"):
        val = getattr(ds, kw, None)
        if val is None:
            continue
        from pydicom.multival import MultiValue

        vals = list(val) if isinstance(val, (list, tuple, MultiValue)) else [val]
        for v in vals:
            for t in re.split(r"[\^\s,.\-]+", str(v)):
                if len(t) >= 3:
                    toks.add(t.lower())
    pid = str(getattr(ds, "PatientID", "") or "")
    if len(pid) >= 3:
        toks.add(pid.lower())
    return toks


def scrub_free_text(ds: Any, keywords: tuple[str, ...] = FREE_TEXT_SCRUB) -> list[AuditEntry]:
    """Remove names (from this file's own PN fields), PN-like tokens, titles+names, dates and long
    digit runs from free-text description fields. Must run before identity tags are deleted, since
    it uses them as the list of names to look for. The original text is never logged."""
    audit: list[AuditEntry] = []
    names = _name_tokens(ds)
    for kw in keywords:
        val = getattr(ds, kw, None)
        if val is None or not str(val).strip():
            continue
        text = str(val)
        new = _PN_RE.sub("", text)
        new = _TITLE_NAME_RE.sub("", new)
        new = _DATE_RE.sub("", new)
        new = _LONG_DIGITS_RE.sub("", new)
        if names:
            pattern = re.compile(r"(?i)(?<![A-Za-z])(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                                 + r")(?![A-Za-z])")
            new = pattern.sub("", new)
        new = re.sub(r"\s{2,}", " ", new).strip(" _-^,;")
        if new != text:
            setattr(ds, kw, new)
            audit.append(AuditEntry("free_text_scrub", "scrubbed_free_text",
                                    f"{kw}: removed {len(text) - len(new)} char(s) of name/ID/date text",
                                    flagged_for_review=True))
    return audit
