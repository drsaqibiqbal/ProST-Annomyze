"""OCR/NER is an optional extra. This suite verifies graceful degradation
without requiring tesseract/spaCy to be installed in CI; if a real OCR
backend IS present, the "require_ocr=True with no backend" test is skipped
since the premise (no backend) no longer holds."""

import pytest

from prostate_deid.burned_in_text import BurnedInTextRedact, _try_import_ocr_stack


@pytest.fixture
def sample_with_overlay(synthetic_sample):
    synthetic_sample.dicom_meta["(6000,3000)"] = b"\x00" * 8
    synthetic_sample.dicom_meta["BurnedInAnnotation"] = "YES"
    return synthetic_sample


def test_degrades_gracefully_when_ocr_unavailable(sample_with_overlay, monkeypatch):
    monkeypatch.setattr("prostate_deid.burned_in_text._try_import_ocr_stack", lambda: None)
    t = BurnedInTextRedact(require_ocr=False, always_apply=True)
    out = t(sample_with_overlay)

    flagged = [e for e in out.audit if e.action == "ocr_unavailable"]
    assert len(flagged) == 1
    assert flagged[0].flagged_for_review is True


def test_require_ocr_true_raises_when_unavailable(sample_with_overlay, monkeypatch):
    monkeypatch.setattr("prostate_deid.burned_in_text._try_import_ocr_stack", lambda: None)
    t = BurnedInTextRedact(require_ocr=True, always_apply=True)
    with pytest.raises(RuntimeError):
        t(sample_with_overlay)


def test_overlay_plane_scrubbed_even_without_ocr(sample_with_overlay, monkeypatch):
    monkeypatch.setattr("prostate_deid.burned_in_text._try_import_ocr_stack", lambda: None)
    t = BurnedInTextRedact(require_ocr=False, always_apply=True)
    out = t(sample_with_overlay)
    assert "(6000,3000)" not in out.dicom_meta


def test_burned_in_annotation_header_flag_is_honoured(sample_with_overlay, monkeypatch):
    monkeypatch.setattr("prostate_deid.burned_in_text._try_import_ocr_stack", lambda: None)
    t = BurnedInTextRedact(require_ocr=False, always_apply=True)
    out = t(sample_with_overlay)
    header_flag_entries = [e for e in out.audit if e.action == "header_flag_yes_no_ocr_hit"]
    assert len(header_flag_entries) == 1


@pytest.mark.skipif(_try_import_ocr_stack() is None, reason="tesseract/spaCy not installed")
def test_detection_raises_notimplemented_without_model_wiring(synthetic_sample):
    # OCR stack IS available in this environment, but no spaCy model / bbox
    # detector has been wired in -- see BurnedInTextRedact._detect_phi_regions
    # docstring. This documents that gap rather than silently no-op'ing.
    t = BurnedInTextRedact(require_ocr=True, always_apply=True)
    with pytest.raises(NotImplementedError):
        t(synthetic_sample)
