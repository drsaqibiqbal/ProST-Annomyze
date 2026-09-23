"""
Anonymization dashboard: upload a .dcm, run the default pipeline, inspect
before/after header + pixel data + the full audit trail, download the
de-identified file.

Run:
    pip install prostate-deid[dashboard]
    streamlit run dashboard/anonymize_app.py

This is a thin UI over prostate_deid.pipeline.anonymize_dicom /
default_pipeline; it holds no de-identification logic of its own so the
dashboard and the library can never silently disagree about what "de-identified"
means.
"""

from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import streamlit as st

from prostate_deid import default_pipeline, load_sample, save_sample, summarize_audit
from prostate_deid.comparison import as_markdown_table

st.set_page_config(page_title="prostate-deid — Anonymize", layout="wide")
st.title("prostate-deid — anonymize a study")
st.caption(
    "Running this pipeline does not by itself constitute HIPAA or GDPR "
    "compliance. It is a tool toward compliance, not a certification of it."
)

uploaded = st.file_uploader("Upload a DICOM (.dcm) file", type=["dcm"])

with st.sidebar:
    st.header("Options")
    profile = st.selectbox("De-identification profile", ["hipaa_safe_harbor", "dicom_ps3.15_basic"])
    redact_text = st.checkbox("Redact burned-in pixel text (OCR/NER)", value=True)
    require_ocr = st.checkbox("Require OCR backend (fail if unavailable)", value=False)
    secret = st.text_input(
        "Pseudonymization secret",
        type="password",
        help="Falls back to PROSTATE_DEID_KEY env var if left blank. "
             "Never reuse a demo secret for real data.",
    )
    st.divider()
    st.markdown("**Capability comparison**")
    st.markdown(as_markdown_table())

if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.dcm"
        out_path = Path(tmp) / "out.dcm"
        map_path = Path(tmp) / "mapping.json"
        in_path.write_bytes(uploaded.getvalue())

        before = load_sample(in_path)

        t0 = time.perf_counter()
        pipeline = default_pipeline(
            mapping_store=str(map_path),
            profile=profile,
            redact_burned_in_text=redact_text,
            require_ocr=require_ocr,
            secret=secret or None,
        )
        after = pipeline(load_sample(in_path))
        save_sample(after, out_path)
        runtime_s = time.perf_counter() - t0

        summary = summarize_audit(after.audit)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tags removed", summary["tags_removed"])
        col2.metric("Flagged for review", summary["flagged_for_review"])
        col3.metric("UIDs regenerated", summary["uids_regenerated"])
        col4.metric("Runtime (s)", f"{runtime_s:.3f}")

        st.subheader("Header: before / after")
        before_keys = set(before.dicom_meta or {})
        after_keys = set(after.dicom_meta or {})
        rows = []
        for key in sorted(before_keys | after_keys):
            if key in before_keys and key not in after_keys:
                status = "removed"
            elif key in before_keys and after.dicom_meta.get(key) != before.dicom_meta.get(key):
                status = "changed"
            elif key not in before_keys:
                status = "added"
            else:
                status = "kept"
            rows.append({"field": key, "status": status})
        st.dataframe(rows, use_container_width=True)

        st.subheader("Mid-slice: before / after")
        img_col1, img_col2 = st.columns(2)
        mid = before.image.shape[1] // 2 if before.image.ndim > 2 else 0
        img_col1.image(before.image[0, mid] if before.image.ndim > 2 else before.image[0], caption="Before", clamp=True)
        img_col2.image(after.image[0, mid] if after.image.ndim > 2 else after.image[0], caption="After", clamp=True)

        st.subheader("Full audit trail")
        st.dataframe(
            [
                {
                    "step": e.step, "action": e.action, "detail": e.detail,
                    "confidence": e.confidence, "flagged": e.flagged_for_review,
                }
                for e in after.audit
            ],
            use_container_width=True,
        )

        st.download_button(
            "Download de-identified .dcm",
            data=out_path.read_bytes(),
            file_name=f"deid_{uploaded.name}",
            mime="application/dicom",
        )
else:
    st.info("Upload a synthetic or already-consented DICOM file to try the pipeline.")
