"""
Burned-in pixel PHI detection and redaction.

Text can be baked directly into DICOM pixel data (scanner overlays,
fusion-biopsy annotation frames) or into a separate DICOM *overlay plane* --
a distinct bitmap layer historically used for annotations, easy to forget
precisely because it is not "the image" an OCR pass over pixel data would
scan. This module scrubs both.

v2: pixel detection is now real. It delegates to
:func:`prostate_deid.text_restore.restore_burned_in_text` (v1's multi-scale
Tesseract detector + the overlay-text detector, glyph segmentation, and
restoration or blackout with re-OCR verification). Every detected text line is
treated as PHI -- no NER classification is applied, matching the
asymmetric-error policy -- so only Tesseract is required; spaCy is no longer
needed.

This is the one transform family in the library that is inherently
probabilistic, and it is optional by design: OCR (pytesseract) is declared as
the ``[ocr]`` extra, not a hard dependency. When the OCR backend is
unavailable this transform does NOT crash the pipeline (``require_ocr=False``
is the default): it records a flagged ``"ocr_unavailable"`` audit entry so the
gap is visible to whoever consumes the audit trail, rather than silently
producing pixel data that looks scrubbed but was never inspected.
"""

from __future__ import annotations

import numpy as np

from .core import AuditEntry, Sample, Transform

__all__ = ["BurnedInTextRedact"]

BBox = tuple[int, int, int, int]  # (y0, y1, x0, x1)


def _try_import_ocr_stack():
    """True if pytesseract and a working tesseract binary are available, else None."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return None
    from .universal_io import _ocr_available

    try:
        if not _ocr_available():
            return None
        import pytesseract as _pt

        # Fails fast (and cheaply) if the binary is missing or broken even though the python
        # package is installed -- two independent failure modes that must both degrade gracefully.
        _pt.get_tesseract_version()
    except Exception:
        return None
    return True


class BurnedInTextRedact(Transform):
    """Detect and black out PHI text baked into pixel data and overlay planes.

    Parameters
    ----------
    ocr_engine:
        Backend identifier. Only ``"tesseract"`` is implemented; the
        parameter exists so a cloud OCR backend can be added later without
        an API break (see README limitations / future work).
    confidence_threshold:
        Detections below this confidence are still redacted (asymmetric
        failure policy: a missed detection is a privacy incident, an
        over-redaction is a data-quality loss) but are additionally flagged
        for human review.
    require_ocr:
        If ``True``, raise ``RuntimeError`` when the OCR/NER backend is
        unavailable. If ``False`` (default), degrade gracefully: log a
        flagged ``"ocr_unavailable"`` entry and leave pixel data untouched,
        so a missing optional dependency cannot silently take down an
        otherwise-deterministic pipeline.
    scrub_overlay_planes:
        Whether to also enumerate and scrub DICOM overlay-plane bitmaps
        (group ``60xx``, element ``3000``), not only the primary pixel
        array.
    restore_method:
        How detected text is removed: ``"blackout"`` (v1: whole box set to 0) or a
        restoration method from :data:`prostate_deid.text_restore.RESTORE_METHODS`
        (only glyph pixels are replaced; failed verification falls back to blackout).
    detect_overlay:
        Also run the overlay-text detector for text drawn on anatomy.
    """

    def __init__(
        self,
        ocr_engine: str = "tesseract",
        confidence_threshold: float = 0.6,
        require_ocr: bool = False,
        scrub_overlay_planes: bool = True,
        restore_method: str = "telea",
        detect_overlay: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if ocr_engine != "tesseract":
            raise NotImplementedError(
                f"OCR engine {ocr_engine!r} is not implemented; only 'tesseract' is available."
            )
        self.ocr_engine = ocr_engine
        self.confidence_threshold = confidence_threshold
        self.require_ocr = require_ocr
        self.scrub_overlay_planes = scrub_overlay_planes
        self.restore_method = restore_method
        self.detect_overlay = detect_overlay

    def apply(self, sample: Sample) -> Sample:
        backend_ok = _try_import_ocr_stack()

        # Overlay-plane scrubbing is deterministic (no OCR/NER involved) and
        # therefore runs regardless of OCR backend availability -- a missing
        # optional dependency should not also silently skip the one part of
        # this transform that never needed it.
        if self.scrub_overlay_planes:
            self._scrub_overlay_planes(sample)

        if not backend_ok:
            if self.require_ocr:
                raise RuntimeError(
                    "BurnedInTextRedact requires the 'ocr' extra "
                    "(pip install prostate-deid[ocr]) and a working tesseract "
                    "binary; it was not found, and require_ocr=True."
                )
            sample.log(AuditEntry(
                step="burned_in_phi", action="ocr_unavailable",
                detail="OCR backend not installed or tesseract binary missing; "
                       "pixel data was NOT inspected for burned-in text.",
                flagged_for_review=True,
            ))
            self._check_burned_in_annotation_header(sample, ocr_ran=False)
            return sample

        from .text_restore import restore_burned_in_text

        image = np.asarray(sample.image)
        vol = image.reshape((-1,) + image.shape[-3:]) if image.ndim >= 3 else image[np.newaxis, np.newaxis]
        n_regions = 0
        out_channels = []
        for c in range(vol.shape[0]):
            res = restore_burned_in_text(vol[c], method=self.restore_method, detect_overlay=self.detect_overlay)
            out_channels.append(res.pixels)
            for entry in res.audit:
                if entry.confidence is not None and entry.confidence < self.confidence_threshold:
                    entry.flagged_for_review = True
                sample.log(entry)
            n_regions += len(res.regions)
        if n_regions:
            sample.image = np.stack(out_channels).reshape(image.shape).astype(image.dtype)
            sample.image_dirty = True
        self._check_burned_in_annotation_header(sample, ocr_ran=True, regions_found=n_regions)
        return sample

    @staticmethod
    def _black_out(image: np.ndarray, bbox: BBox) -> None:
        y0, y1, x0, x1 = bbox
        image[..., y0:y1, x0:x1] = 0

    def _scrub_overlay_planes(self, sample: Sample) -> None:
        """Zero out DICOM overlay-plane bitmaps present in dicom_meta, if any.

        Overlay planes are stored under repeating-group tags (6000,3000),
        (6002,3000), ... (6006,3000) — enumerated here by the '(GGGG,EEEE)'
        string convention used elsewhere in this library for tags without a
        stable keyword. This only acts on overlay data actually surfaced in
        sample.dicom_meta by dicom_io.load_sample(); it does not read a raw
        file itself.
        """
        if sample.dicom_meta is None:
            return
        overlay_tags = [
            key for key in sample.dicom_meta
            if key.startswith("(60") and key.endswith(",3000)")
        ]
        for tag in overlay_tags:
            del sample.dicom_meta[tag]
            sample.log(AuditEntry(
                step="burned_in_phi", action="scrubbed_overlay_plane",
                detail=f"removed overlay bitmap at {tag}",
            ))

    def _check_burned_in_annotation_header(self, sample: Sample, ocr_ran: bool, regions_found: int = 0) -> None:
        """Honour the DICOM BurnedInAnnotation header flag.

        If the header itself declares BurnedInAnnotation=YES but no OCR hit
        was produced (either because OCR didn't run, or it ran and found
        nothing), that disagreement is exactly the case that should not be
        silently trusted -- flag it.
        """
        if sample.dicom_meta is None:
            return
        flag = sample.dicom_meta.get("BurnedInAnnotation")
        if flag == "YES" and regions_found == 0:
            sample.log(AuditEntry(
                step="burned_in_phi", action="header_flag_yes_no_ocr_hit",
                detail=f"BurnedInAnnotation=YES in header (ocr_ran={ocr_ran}); "
                       "manual pixel review recommended.",
                flagged_for_review=True,
            ))
