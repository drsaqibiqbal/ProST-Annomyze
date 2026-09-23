"""
Reviewer QA dashboard: loads a review bundle exported by
prostate_deid.export.export_review_bundle and surfaces every
flagged_for_review entry as a cropped thumbnail for human triage.

Rationale: flagged_for_review sitting only inside an audit-trail JSON is a
dead-end data point with no consumer. A flag nobody looks at manufactures
false confidence, worse than not flagging at all. This dashboard is the
minimal bridge from "flagged" to "triaged by a human."

Run:
    pip install prostate-deid[dashboard]
    streamlit run dashboard/review_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="prostate-deid — Review flagged items", layout="wide")
st.title("prostate-deid — reviewer QA")

bundle_dir = st.text_input("Review bundle directory", value="")

if bundle_dir:
    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        st.error(f"No manifest.json found at {manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text())
        st.write(f"{len(manifest)} flagged item(s) in this bundle.")

        if "dispositions" not in st.session_state:
            st.session_state.dispositions = {}

        for item in manifest:
            with st.expander(
                f"#{item['index']} — {item['step']} / {item['action']} "
                f"(confidence={item['confidence']})",
                expanded=True,
            ):
                col1, col2 = st.columns([1, 2])
                with col1:
                    if item["thumbnail"]:
                        thumb = np.load(bundle_path / item["thumbnail"])
                        # Reduce to a 2D preview regardless of source ndim.
                        preview = np.squeeze(thumb)
                        while preview.ndim > 2:
                            preview = preview[preview.shape[0] // 2]
                        st.image(preview, clamp=True, caption="Cropped region")
                    else:
                        st.write("No spatial thumbnail available for this entry.")
                with col2:
                    st.text(item["detail"])
                    disposition = st.radio(
                        "Disposition",
                        ["Pending", "Confirmed PHI (kept redacted)", "False positive (restore)"],
                        key=f"disp_{item['index']}",
                        horizontal=True,
                    )
                    st.session_state.dispositions[item["index"]] = disposition

        if st.button("Export dispositions"):
            out = bundle_path / "dispositions.json"
            out.write_text(json.dumps(st.session_state.dispositions, indent=2))
            st.success(f"Wrote {out}")
else:
    st.info(
        "Point this at a directory produced by "
        "prostate_deid.export.export_review_bundle(sample, out_dir)."
    )
