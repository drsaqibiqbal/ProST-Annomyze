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

# --- Theme: validated categorical/status palette, light + dark aware ---------
st.markdown(
    """
    <style>
    :root {
        --brand-blue:    #2a78d6;
        --brand-aqua:    #1baf7a;
        --brand-violet:  #4a3aa7;
        --brand-orange:  #eb6834;
        --status-good:     #0ca30c;
        --status-warning:  #fab219;
        --status-serious:  #ec835a;
        --status-critical: #d03b3b;
        --surface-1: #fcfcfb;
        --surface-2: #f9f9f7;
        --text-primary: #0b0b0b;
        --text-secondary: #52514e;
        --border-hairline: rgba(11,11,11,0.10);
    }
    @media (prefers-color-scheme: dark) {
        :root:not([data-theme="light"]) {
            --brand-blue:    #3987e5;
            --brand-aqua:    #199e70;
            --brand-violet:  #9085e9;
            --brand-orange:  #d95926;
            --status-good:     #0ca30c;
            --status-warning:  #fab219;
            --status-serious:  #ec835a;
            --status-critical: #d03b3b;
            --surface-1: #1a1a19;
            --surface-2: #0d0d0d;
            --text-primary: #ffffff;
            --text-secondary: #c3c2b7;
            --border-hairline: rgba(255,255,255,0.10);
        }
    }

    /* Hero title band */
    .prost-hero {
        background: linear-gradient(120deg, var(--brand-blue) 0%, var(--brand-violet) 55%, var(--brand-aqua) 100%);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px rgba(42, 120, 214, 0.25);
    }
    .prost-hero h1 {
        color: #ffffff !important;
        margin: 0 0 0.35rem 0;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .prost-hero p {
        color: rgba(255,255,255,0.92) !important;
        margin: 0;
        font-size: 1.02rem;
    }

    /* Sidebar accent */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface-1) 100%);
        border-right: 1px solid var(--border-hairline);
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] .stMarkdown h2 {
        color: var(--brand-blue);
    }

    /* Card wrapper used around each column's content */
    .prost-card {
        background: var(--surface-1);
        border: 1px solid var(--border-hairline);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 2px 10px rgba(11,11,11,0.06);
        height: 100%;
    }
    .prost-card h3 {
        margin-top: 0;
    }
    .prost-card-original h3 { color: var(--brand-orange); }
    .prost-card-deid h3 { color: var(--brand-aqua); }

    /* Section labels as colored pills */
    .prost-pill {
        display: inline-block;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .prost-pill-original { background: rgba(235,104,52,0.15); color: var(--brand-orange); }
    .prost-pill-deid { background: rgba(27,175,122,0.15); color: var(--brand-aqua); }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(120deg, var(--brand-blue), var(--brand-violet));
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1.3rem;
        box-shadow: 0 4px 14px rgba(42, 120, 214, 0.3);
        transition: transform 0.08s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(42, 120, 214, 0.4);
        color: #ffffff;
    }

    /* File uploader accent border */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--brand-blue) !important;
        border-radius: 12px !important;
        background: rgba(42, 120, 214, 0.04) !important;
    }

    /* Divider color */
    hr {
        border-top: 2px solid var(--brand-blue) !important;
        opacity: 0.25;
    }

    /* Mobile / narrow-viewport tweaks */
    @media (max-width: 640px) {
        .prost-hero { padding: 1.1rem 1.25rem; border-radius: 12px; }
        .prost-hero h1 { font-size: 1.5rem; }
        .prost-hero p { font-size: 0.92rem; }
        .prost-card { padding: 0.85rem 0.9rem; }
        .stButton > button, .stDownloadButton > button {
            width: 100%;
            padding: 0.65rem 1rem;
        }
    }
    </style>
    <div class="prost-hero">
        <h1>🩺 ProST-Annomyze</h1>
        <p>Upload a PNG, DICOM, or NIfTI image → get back a fully de-identified DICOM,
        with an optional second export to .nii.gz. Running this tool does not by itself
        constitute HIPAA or GDPR compliance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "📤 Upload an image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "nii", "gz"],
    help="Accepts raster images (PNG/JPEG/BMP/TIFF), DICOM (.dcm), or NIfTI (.nii / .nii.gz).",
)

_PERFORMANCE_MODES = {
    "🖥️ Thorough (desktop/server CPU)": {
        "upscale_factors": None,  # library default: (3, 4, 5)
        "check_inverted": True,
        "help": "Runs Tesseract at 3 upscale factors x both polarities per slice "
        "(the library default) — highest OCR recall, slowest.",
    },
    "⚖️ Balanced (laptop CPU)": {
        "upscale_factors": (3, 4),
        "check_inverted": True,
        "help": "Drops one upscale factor (2 of 3) — noticeably faster with a small "
        "recall cost, still checks both polarities.",
    },
    "📱 Lightweight (mobile / low-power CPU)": {
        "upscale_factors": (3,),
        "check_inverted": False,
        "help": "Single upscale factor, no inverted-polarity pass — up to ~6x fewer "
        "Tesseract calls per slice. Faster on phones and constrained CPUs, but "
        "may miss more burned-in text; every run is still flagged for human review.",
    },
}

with st.sidebar:
    st.header("⚙️ Options")
    modality = st.selectbox(
        "Modality tag for synthesized DICOM (PNG/NIfTI input only)",
        ["OT", "MR", "CT", "US", "SC"],
        help="Ignored for DICOM input, which keeps its own Modality tag.",
    )
    st.divider()
    st.subheader("⚡ Performance mode")
    perf_mode = st.radio(
        "OCR thoroughness vs. speed",
        list(_PERFORMANCE_MODES.keys()),
        index=0,
        help="Controls how many Tesseract passes run per slice during burned-in "
        "text redaction — the dominant CPU cost for multi-slice volumes. Pick a "
        "lighter mode on CPU-only or mobile hardware; it never disables redaction, "
        "only how exhaustively it searches.",
    )
    perf_cfg = _PERFORMANCE_MODES[perf_mode]
    st.caption(perf_cfg["help"])
    st.divider()
    st.markdown(
        "**🔒 Policy**\n\n"
        "- **PNG/NIfTI input:** a brand-new minimal DICOM is built around the pixel data. "
        "No patient/institution field is ever set.\n"
        "- **DICOM input:** every non-technical tag is removed (full wipe), not just a "
        "named identifier list — private tags included. All UIDs are freshly randomized.\n"
        "- **All inputs:** pixel data is OCR-scanned (Tesseract) and every detected text "
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

    st.success(f"✅ Loaded {filename} — detected as **{loaded['kind']}**, shape {loaded['pixels'].shape}")

    notes: list[str] = []
    if loaded["kind"] == "dicom":
        deid_ds, notes = full_wipe_dicom(
            loaded["source_dataset"],
            ocr_upscale_factors=perf_cfg["upscale_factors"],
            ocr_check_inverted=perf_cfg["check_inverted"],
        )
    else:
        deid_ds, notes = array_to_dicom(
            loaded["pixels"],
            modality=modality,
            ocr_upscale_factors=perf_cfg["upscale_factors"],
            ocr_check_inverted=perf_cfg["check_inverted"],
        )

    dcm_buf = io.BytesIO()
    deid_ds.save_as(dcm_buf, enforce_file_format=True)
    dcm_bytes = dcm_buf.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="prost-card prost-card-original">', unsafe_allow_html=True)
        st.markdown('<span class="prost-pill prost-pill-original">Original</span>', unsafe_allow_html=True)
        pixels = loaded["pixels"]
        mid = pixels.shape[0] // 2
        frame = pixels[mid]
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            st.image(np.clip(frame, 0, 255).astype(np.uint8), caption=f"slice {mid + 1}/{pixels.shape[0]}")
        else:
            st.image(frame, caption=f"slice {mid + 1}/{pixels.shape[0]}", clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="prost-card prost-card-deid">', unsafe_allow_html=True)
        st.markdown('<span class="prost-pill prost-pill-deid">🔒 De-identified DICOM</span>', unsafe_allow_html=True)
        out_pixels = np.asarray(deid_ds.pixel_array)
        out_frame = out_pixels if out_pixels.ndim == 2 else out_pixels[out_pixels.shape[0] // 2] if out_pixels.ndim == 3 and out_pixels.shape[-1] not in (3, 4) else out_pixels
        st.image(out_frame, caption="header stripped + burned-in text redacted", clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

    ocr_failed = any("OCR backend unavailable" in n for n in notes)
    if ocr_failed:
        st.error(
            "🚨 OCR is unavailable in this environment — burned-in pixel text (patient name, "
            "DOB, institution, dates baked into the image) was NOT inspected or redacted. "
            "Do not treat this output as fully de-identified until OCR is fixed."
        )

    st.markdown("### 📝 What changed")
    for n in notes:
        st.markdown(f"- {n}")

    remaining = [
        elem.keyword or f"({elem.tag.group:04X},{elem.tag.element:04X})"
        for elem in deid_ds
        if elem.tag != 0x7FE00010
    ]
    with st.expander(f"🏷️ Remaining tags in output ({len(remaining)})"):
        st.dataframe({"tag": remaining}, use_container_width=True)

    out_stem = Path(filename).stem.replace(".nii", "")
    st.download_button(
        "⬇️ Download de-identified .dcm",
        data=dcm_bytes,
        file_name=f"deid_{out_stem}.dcm",
        mime="application/dicom",
    )

    st.divider()
    st.markdown("### 🔄 Convert this DICOM to .nii.gz")
    if st.button("Convert to .nii.gz"):
        nii_bytes = dicom_to_nifti_bytes(deid_ds)
        st.download_button(
            "⬇️ Download .nii.gz",
            data=nii_bytes,
            file_name=f"deid_{out_stem}.nii.gz",
            mime="application/gzip",
        )
else:
    st.info("👋 Upload a PNG, DICOM (.dcm), or NIfTI (.nii / .nii.gz) file to get started.")
