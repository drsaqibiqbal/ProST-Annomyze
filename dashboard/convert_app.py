"""
ProST-Annomyze: upload a PNG/JPEG/BMP/TIFF, DICOM, or NIfTI (.nii/.nii.gz)
file, fully de-identify it, and download it as a clean DICOM. Optionally
convert that de-identified DICOM into .nii.gz as a second step.

Run:
    pip install prostate-deid[dashboard]
    streamlit run dashboard/convert_app.py

This is a thin UI over prostate_deid.universal_io; it holds no
de-identification logic of its own so the dashboard and the library can
never silently disagree about what "de-identified" means.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st

from prostate_deid.universal_io import (
    array_to_dicom,
    dicom_to_nifti_bytes,
    full_wipe_dicom,
    load_any,
)

st.set_page_config(page_title="ProST-Annomyze", layout="wide", page_icon="🩺")
st.title("ProST-Annomyze")
st.caption(
    "Upload a PNG, DICOM, or NIfTI image → get back a fully de-identified DICOM, "
    "with an optional second export to .nii.gz. "
    "Running this tool does not by itself constitute HIPAA or GDPR compliance."
)

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "nii", "gz"],
    help="Accepts raster images (PNG/JPEG/BMP/TIFF), DICOM (.dcm), or NIfTI (.nii / .nii.gz).",
)

with st.sidebar:
    st.header("Options")
    modality = st.selectbox(
        "Modality tag for synthesized DICOM (PNG/NIfTI input only)",
        ["OT", "MR", "CT", "US", "SC"],
        help="Ignored for DICOM input, which keeps its own Modality tag.",
    )
    st.divider()
    st.markdown(
        "**Policy**\n\n"
        "- PNG/NIfTI input: a brand-new minimal DICOM is built around the pixel data. "
        "No patient/institution field is ever set.\n"
        "- DICOM input: every non-technical tag is removed (full wipe), not just a "
        "named identifier list — private tags included. All UIDs are freshly randomized.\n"
        "- All inputs: pixel data is OCR-scanned (Tesseract) and every detected text "
        "region is blacked out, regardless of OCR confidence — covers scanner corner "
        "annotations and other burned-in text that header scrubbing cannot reach."
    )

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    filename = uploaded.name

    try:
        loaded = load_any(io.BytesIO(file_bytes), filename=filename)
    except Exception as exc:
        st.error(f"Could not read {filename}: {exc}")
        st.stop()

    st.success(f"Loaded {filename} — detected as **{loaded['kind']}**, shape {loaded['pixels'].shape}")

    notes: list[str] = []
    if loaded["kind"] == "dicom":
        deid_ds, notes = full_wipe_dicom(loaded["source_dataset"])
    else:
        deid_ds, notes = array_to_dicom(loaded["pixels"], modality=modality)

    dcm_buf = io.BytesIO()
    deid_ds.save_as(dcm_buf, enforce_file_format=True)
    dcm_bytes = dcm_buf.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        pixels = loaded["pixels"]
        mid = pixels.shape[0] // 2
        frame = pixels[mid]
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            st.image(np.clip(frame, 0, 255).astype(np.uint8), caption=f"slice {mid + 1}/{pixels.shape[0]}")
        else:
            st.image(frame, caption=f"slice {mid + 1}/{pixels.shape[0]}", clamp=True)

    with col2:
        st.subheader("De-identified DICOM")
        out_pixels = np.asarray(deid_ds.pixel_array)
        out_frame = out_pixels if out_pixels.ndim == 2 else out_pixels[out_pixels.shape[0] // 2] if out_pixels.ndim == 3 and out_pixels.shape[-1] not in (3, 4) else out_pixels
        st.image(out_frame, caption="header stripped + burned-in text redacted", clamp=True)

    ocr_failed = any("OCR backend unavailable" in n for n in notes)
    if ocr_failed:
        st.error(
            "OCR is unavailable in this environment — burned-in pixel text (patient name, "
            "DOB, institution, dates baked into the image) was NOT inspected or redacted. "
            "Do not treat this output as fully de-identified until OCR is fixed."
        )

    st.subheader("What changed")
    for n in notes:
        st.markdown(f"- {n}")

    remaining = [
        elem.keyword or f"({elem.tag.group:04X},{elem.tag.element:04X})"
        for elem in deid_ds
        if elem.tag != 0x7FE00010
    ]
    st.subheader(f"Remaining tags in output ({len(remaining)})")
    st.dataframe({"tag": remaining}, use_container_width=True)

    out_stem = Path(filename).stem.replace(".nii", "")
    st.download_button(
        "Download de-identified .dcm",
        data=dcm_bytes,
        file_name=f"deid_{out_stem}.dcm",
        mime="application/dicom",
    )

    st.divider()
    st.subheader("Optional: convert this DICOM to .nii.gz")
    if st.button("Convert to .nii.gz"):
        nii_bytes = dicom_to_nifti_bytes(deid_ds)
        st.download_button(
            "Download .nii.gz",
            data=nii_bytes,
            file_name=f"deid_{out_stem}.nii.gz",
            mime="application/gzip",
        )
else:
    st.info("Upload a PNG, DICOM (.dcm), or NIfTI (.nii / .nii.gz) file to get started.")
