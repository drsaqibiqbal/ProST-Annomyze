"""
ProST-Annomyze v2: upload DICOM (single files, whole series or a ZIP), NIfTI
(.nii / .nii.gz) or raster images (PNG / JPEG / BMP / TIFF); get back
de-identified, conformance-checked DICOM, and optionally a geometry-correct
.nii.gz with a JSON sidecar.

What is new in v2 (see docs/v2_analysis.md and docs/v2_changes.md):
  * burned-in text is removed while keeping the image: only the glyph pixels
    are replaced (Telea / Navier-Stokes / biharmonic / ViT-MAE + Poisson),
    every other pixel is bit-identical, and every restored region is
    re-checked (OCR + silhouette test) and falls back to a black box if text
    is still visible; text drawn on anatomy is now detected too;
  * MR-aware header policy (series labels, TE/TR, b-values rescued from
    vendor-private tags, frame of reference) and study-level pseudonyms so
    T2W, DWI and ADC stay linked;
  * geometry and intensities preserved in both directions; one shared audit
    trail.

Run:
    pip install prostate-deid[dashboard]          # + [vit] for the ViT option
    streamlit run dashboard/convert_app.py

This is a thin UI over prostate_deid.convert / prostate_deid.text_restore; it
holds no de-identification logic of its own.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath

import numpy as np
import streamlit as st

from prostate_deid.convert import (
    StudyContext,
    check_conformance,
    deidentify_dicom,
    load_nifti,
    nifti_to_dicom_series,
    raster_to_dicom,
    series_to_nifti,
)
from prostate_deid._version import __version__
from prostate_deid.mr_tags import read_bvalue
from prostate_deid.text_restore import changed_region_boxes
from prostate_deid.universal_io import _ocr_available

st.set_page_config(page_title="ProST-Annomyze v2", layout="wide", page_icon="🩺")

# --- Theme (same palette as v1) -------------------------------------------------------------
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
    .prost-badge { display:inline-block; padding:0.1rem 0.55rem; border-radius:999px; font-size:0.75rem;
                   font-weight:700; margin-right:0.35rem; }
    .prost-ok { background: rgba(12,163,12,0.15); color: var(--status-good); }
    .prost-warn { background: rgba(250,178,25,0.18); color: #9a6a00; }
    .prost-bad { background: rgba(208,59,59,0.15); color: var(--status-critical); }
    </style>
    <div class="prost-hero">
        <h1>🩺 ProST-Annomyze <span style="font-weight:600;opacity:.85">v{__version__}</span></h1>
        <p>DICOM (files, series or ZIP), NIfTI (.nii / .nii.gz), PNG, JPEG, BMP or TIFF in →
        de-identified, conformance-checked DICOM out, with burned-in text removed <b>while keeping
        the image</b>, and an optional geometry-correct .nii.gz.</p>
    </div>
    """.replace("{__version__}", __version__),
    unsafe_allow_html=True,
)

_TEXT_METHODS = {
    "🩹 Restore image — Telea (recommended)": "telea",
    "🩹 Restore image — Navier–Stokes": "ns",
    "🩹 Restore image — biharmonic": "biharmonic",
    "🧠 Restore image — ViT-MAE + Poisson (slow, needs [vit])": "vit_mae_poisson",
    "⬛ Black box (v1 behaviour)": "blackout",
}

_PERFORMANCE_MODES = {
    "🖥️ Thorough (desktop/server CPU)": {"upscale_factors": None, "check_inverted": True, "detect_overlay": True,
                                        "help": "v1's 3 upscale factors x both polarities, plus the overlay-text "
                                                "detector for text drawn on anatomy. Highest recall."},
    "⚖️ Balanced (laptop CPU)": {"upscale_factors": (3, 4), "check_inverted": True, "detect_overlay": True,
                                "help": "Two upscale factors; overlay-text detector on."},
    "📱 Lightweight (mobile / low-power CPU)": {"upscale_factors": (3,), "check_inverted": False,
                                              "detect_overlay": False,
                                              "help": "One upscale factor, no inverted pass, no overlay-text "
                                                      "detector. Fastest, lowest recall; always review."},
}

_MAX_ZIP_MEMBERS = 5000
_MAX_UNZIPPED_BYTES = 2 * 1024 ** 3

with st.sidebar:
    st.header("⚙️ Options")
    st.subheader("🔤 Burned-in text")
    text_label = st.radio("How to remove detected text", list(_TEXT_METHODS), index=0,
                          help="Restore methods replace only the glyph pixels and keep the rest of the image "
                               "bit-identical; regions that still look like text after restoration are blacked "
                               "out automatically. Restored pixels are estimates, never measurements.")
    text_method = _TEXT_METHODS[text_label]
    perf_label = st.radio("Performance mode", list(_PERFORMANCE_MODES), index=0)
    perf = _PERFORMANCE_MODES[perf_label]
    st.caption(perf["help"])
    redact_pixels = st.checkbox("Inspect pixels for burned-in text", value=True)

    st.divider()
    st.subheader("🏷️ Header policy")
    mr_aware = st.radio(
        "DICOM tags to keep",
        ["MR-aware (recommended for prostate mpMRI)", "Strict: technical tags only (v1)"], index=0,
        help="MR-aware keeps series labels (after name/ID/date scrubbing), TE/TR/flip angle/field strength, "
             "diffusion b-values (rescued from Siemens/GE/Philips private tags) and the frame of reference, "
             "so T2W/DWI/ADC stay usable and linked. Private and unknown tags are always removed.",
    ).startswith("MR")

    st.divider()
    st.subheader("🔑 Pseudonymization")
    key = st.text_input("Pseudonymization key (optional)", type="password",
                        help="Same key → same pseudonyms across sessions (re-linkable by whoever holds the key). "
                             "Leave empty for a random per-session key: consistent within this run only.")
    subject = st.text_input("Subject label for NIfTI / image inputs (optional)", value="",
                            help="Used as PatientID for inputs that carry none. Never enter a real identifier.")
    shift_dates = st.checkbox("Shift dates (keep intervals)", value=True)
    modality = st.selectbox("Modality for NIfTI / image inputs", ["MR", "OT", "CT", "US", "SC"], index=0,
                            help="MR writes MR Image Storage with geometry; others write Secondary Capture.")

    st.divider()
    st.subheader("🧠 NIfTI export")
    rarn = st.checkbox("RARN intensity normalization", value=False,
                       help="Robust Anatomy-Referenced Normalization before writing .nii.gz (parameters are "
                            "logged). Off keeps stored values exactly.")

st.caption(
    "🛡️ Built to support HIPAA- and GDPR-aligned de-identification workflows. Compliance also depends on "
    "your institution's policies and human review of flagged items — see the README's compliance section."
)

uploads = st.file_uploader(
    "📤 Upload files (select several, or a .zip, to convert a whole study)",
    type=["dcm", "nii", "gz", "zip", "png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    help="All files from one upload share one pseudonymization context, so series of one patient stay in one "
         "study. .gz files must be .nii.gz.",
)


# ---------------------------------------------------------------------------
# Input expansion and classification
# ---------------------------------------------------------------------------

def _kind(name: str, data: bytes) -> str | None:
    low = name.lower()
    if low.endswith(".nii") or low.endswith(".nii.gz"):
        return "nifti"
    if low.endswith(".dcm") or (len(data) > 132 and data[128:132] == b"DICM"):
        return "dicom"
    if low.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        return "raster"
    return None


def _expand(files) -> tuple[list[tuple[str, bytes, str]], list[str]]:
    items, skipped = [], []
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                members = [m for m in zf.infolist() if not m.is_dir()]
                if len(members) > _MAX_ZIP_MEMBERS or sum(m.file_size for m in members) > _MAX_UNZIPPED_BYTES:
                    skipped.append(f"{f.name}: too many files or too large when unzipped")
                    continue
                for m in members:
                    name = PurePosixPath(m.filename).name
                    if name.startswith(".") or "__MACOSX" in m.filename:
                        continue
                    blob = zf.read(m)
                    k = _kind(name, blob)
                    (items.append((name, blob, k)) if k else skipped.append(name))
            continue
        if f.name.lower().endswith(".gz") and not f.name.lower().endswith(".nii.gz"):
            skipped.append(f"{f.name}: only .nii.gz is accepted among .gz files")
            continue
        k = _kind(f.name, data)
        (items.append((f.name, data, k)) if k else skipped.append(f.name))
    return items, skipped


# ---------------------------------------------------------------------------
# Processing (cached in session state: OCR is the slow part)
# ---------------------------------------------------------------------------

def _process(items, opts):
    import pydicom

    ctx = StudyContext(secret=opts["key"] or None, subject_label=opts["subject"] or None,
                       shift_dates=opts["shift_dates"])
    restore_kw = dict(upscale_factors=opts["upscale_factors"], check_inverted=opts["check_inverted"],
                      detect_overlay=opts["detect_overlay"])
    outputs = []  # dicts: name, kind, ds (output), original (pixels), mask, audit
    errors = []
    raster_i = 0
    bar = st.progress(0.0, text="De-identifying…")
    for n, (name, data, kind) in enumerate(items):
        bar.progress((n + 0.5) / len(items), text=f"{n + 1}/{len(items)}: {kind}")
        try:
            if kind == "dicom":
                ds = pydicom.dcmread(io.BytesIO(data))
                try:
                    orig = np.asarray(ds.pixel_array).copy()
                except Exception:
                    orig = None
                res = deidentify_dicom(ds, ctx, keep_mr_tags=opts["mr_aware"], redact_pixels=opts["redact"],
                                       text_method=opts["method"], **restore_kw)
                outs = [(res.datasets[0], orig, res.restore.inpaint_mask if res.restore else None)]
            elif kind == "nifti":
                img = load_nifti(data)
                src_key = hashlib.sha256(data).hexdigest()
                res = nifti_to_dicom_series(img, ctx, modality=opts["modality"], source_key=src_key,
                                            series_number=900 + n, redact_pixels=opts["redact"],
                                            text_method=opts["method"], **restore_kw)
                vol = np.transpose(np.asanyarray(img.dataobj).reshape(img.shape[:3] + (-1,))[..., 0], (2, 1, 0))
                m = res.restore.inpaint_mask if res.restore else None
                outs = [(d, vol[k], m[k] if m is not None else None) for k, d in enumerate(res.datasets)]
            else:
                raster_i += 1
                res = raster_to_dicom(data, ctx, modality="OT" if opts["modality"] == "MR" else opts["modality"],
                                      source_key="raster-batch", instance_number=raster_i, series_number=800,
                                      redact_pixels=opts["redact"], text_method=opts["method"], **restore_kw)
                from prostate_deid.convert import _raster_to_array

                orig = _raster_to_array(data)[0]
                m = res.restore.inpaint_mask[0] if res.restore else None
                outs = [(res.datasets[0], orig, m)]
            for k, (d, orig, m) in enumerate(outs):
                outputs.append({"kind": kind, "ds": d, "orig": orig, "mask": m,
                                "audit": res.audit if k == 0 else [], "upload": name})
        except Exception as exc:  # keep going; report per file
            errors.append(f"{name}: {exc}")
    bar.empty()
    return outputs, errors, ctx.session_key


def _frame(arr, idx=None):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] not in (3, 4):
        a = a[idx if idx is not None else a.shape[0] // 2]
    elif a.ndim == 4:
        a = a[idx if idx is not None else a.shape[0] // 2]
    return a


def _to_display(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:
        return np.clip(a, 0, 255).astype(np.uint8)
    lo, hi = np.percentile(a, [0.5, 99.5]) if a.size else (0, 1)
    if hi <= lo:
        hi = lo + 1
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _overlay(display_u8, mask):
    """The de-identified image as-is, with a red rectangle around every changed area."""
    rgb = np.repeat(display_u8[..., None], 3, axis=2) if display_u8.ndim == 2 else display_u8.copy()
    if mask is None or not mask.any():
        return rgb
    h, w = mask.shape
    t = max(1, round(min(h, w) / 256))
    red = np.array([235, 60, 52], np.uint8)
    for x0, y0, x1, y1 in changed_region_boxes(mask):
        x1, y1 = min(w, x1), min(h, y1)
        rgb[y0:min(h, y0 + t), x0:x1] = red
        rgb[max(0, y1 - t):y1, x0:x1] = red
        rgb[y0:y1, x0:min(w, x0 + t)] = red
        rgb[y0:y1, max(0, x1 - t):x1] = red
    return rgb


if uploads:
    items, skipped = _expand(uploads)
    if skipped:
        st.warning("Skipped (unsupported or not recognised): " + ", ".join(skipped[:20])
                   + (" …" if len(skipped) > 20 else ""))
    if not items:
        st.stop()
    counts = defaultdict(int)
    for _, _, k in items:
        counts[k] += 1
    st.success("✅ " + ", ".join(f"{v} {k}" for k, v in counts.items()) + " file(s) ready")

    opts = dict(method=text_method, mr_aware=mr_aware, redact=redact_pixels, key=key, subject=subject,
                shift_dates=shift_dates, modality=modality, **{k: perf[k] for k in
                                                               ("upscale_factors", "check_inverted", "detect_overlay")})
    sig = hashlib.sha256(b"".join(hashlib.sha256(d).digest() for _, d, _ in items)
                         + json.dumps(opts, sort_keys=True, default=str).encode()).hexdigest()

    if redact_pixels and not _ocr_available():
        st.error("🚨 OCR (Tesseract) is unavailable — burned-in pixel text will NOT be inspected or removed. "
                 "Do not treat the output as fully de-identified.")

    if st.session_state.get("sig") != sig:
        if st.button("🚀 Run de-identification", type="primary"):
            outputs, errors, session_key = _process(items, opts)
            st.session_state.update(sig=sig, outputs=outputs, errors=errors, session_key=session_key)
        else:
            st.info("Options or files changed — press **Run de-identification**.")
            st.stop()

    outputs = st.session_state["outputs"]
    for e in st.session_state.get("errors", []):
        st.error(e)
    if not outputs:
        st.stop()
    if st.session_state.get("session_key"):
        st.caption("🔑 No key given: pseudonyms come from a random per-session key and cannot be re-linked later.")

    # --- group by series -------------------------------------------------------------------
    series = defaultdict(list)
    for o in outputs:
        series[str(o["ds"].SeriesInstanceUID)].append(o)
    for s in series.values():
        s.sort(key=lambda o: int(getattr(o["ds"], "InstanceNumber", 0) or 0))

    all_audit = [e for o in outputs for e in o["audit"]]
    n_flag = sum(e.flagged_for_review for e in all_audit)
    n_rest = sum(e.action == "restored_region" for e in all_audit)
    n_fall = sum(e.action == "redacted_region_fallback" for e in all_audit)
    n_black = sum(e.action == "redacted_region" for e in all_audit)
    conf_fail = sum(bool(check_conformance(o["ds"])) for o in outputs)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Images", len(outputs))
    c2.metric("Series", len(series))
    c3.metric("Text regions restored", n_rest)
    c4.metric("Blacked out", n_black + n_fall, help="v1 blackout method, or restoration fallbacks")
    c5.metric("Flagged for review", n_flag)
    if conf_fail:
        st.warning(f"{conf_fail} output file(s) failed the conformance self-check — see the series table.")

    rows = []
    for uid, s in series.items():
        d0 = s[0]["ds"]
        bvals = sorted({b for b in (read_bvalue(o["ds"])[0] for o in s) if b is not None})
        probs = {p for o in s for p in check_conformance(o["ds"])}
        rows.append({"series": int(getattr(d0, "SeriesNumber", 0) or 0),
                     "description": str(getattr(d0, "SeriesDescription", "")),
                     "modality": str(getattr(d0, "Modality", "")), "images": len(s),
                     "image type": "\\".join(str(x) for x in getattr(d0, "ImageType", [])),
                     "b-values": ", ".join(f"{b:g}" for b in bvals),
                     "conformance": "✔" if not probs else "✘ " + "; ".join(sorted(probs))[:120]})
    st.markdown("### 📚 Series")
    st.dataframe(rows, width="stretch", hide_index=True)

    # --- viewer ----------------------------------------------------------------------------
    st.markdown("### 🔍 Before / after")
    labels = {uid: f"S{r['series']:03d} · {r['description'] or r['modality']} ({r['images']} img)"
              for uid, r in zip(series, rows)}
    uid = st.selectbox("Series", list(series), format_func=lambda u: labels[u])
    s = series[uid]
    first_changed = next((k for k, o in enumerate(s) if o["mask"] is not None and np.asarray(o["mask"]).any()), 0)
    idx = st.slider("Image", 1, len(s), first_changed + 1) - 1 if len(s) > 1 else 0
    o = s[idx]
    out_px = np.asarray(o["ds"].pixel_array)
    n_frames = int(getattr(o["ds"], "NumberOfFrames", 1) or 1)
    fidx = None
    if n_frames > 1:
        fidx = st.slider("Frame", 1, n_frames, n_frames // 2 + 1) - 1
    orig = _frame(o["orig"], fidx)
    out = _frame(out_px, fidx)
    mask = _frame(o["mask"], fidx) if o["mask"] is not None else None
    if mask is not None and mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    cols = st.columns(3)
    with cols[0]:
        st.markdown('<span class="prost-pill prost-pill-original">Original</span>', unsafe_allow_html=True)
        if orig is not None:
            st.image(_to_display(orig), width="stretch")
    with cols[1]:
        st.markdown('<span class="prost-pill prost-pill-deid">🔒 De-identified</span>', unsafe_allow_html=True)
        st.image(_to_display(out), width="stretch")
    with cols[2]:
        st.markdown('<span class="prost-pill prost-pill-original">Where pixels changed</span>', unsafe_allow_html=True)
        st.image(_overlay(_to_display(out), mask if mask is not None and mask.shape == out.shape[:2] else None),
                 width="stretch")
        if mask is not None and mask.size:
            st.caption(f"{int(mask.sum())} pixel(s) altered ({mask.mean():.2%} of the image) inside the red "
                       "boxes; every other pixel is bit-identical to the input. Boxes, not outlines, so the "
                       "removed text is never redrawn.")

    with st.expander(f"🏷️ Tags kept in this file ({sum(1 for e in o['ds'] if e.tag != 0x7FE00010)})"):
        st.dataframe([{"tag": e.keyword or str(e.tag), "value": str(e.value)[:80]} for e in o["ds"]
                      if e.tag != 0x7FE00010 and e.VR != "SQ"], width="stretch", hide_index=True)

    st.markdown("### 📝 Audit trail")
    only_flag = st.checkbox("Show only items flagged for review", value=False)
    st.dataframe([{"file": o2["upload"], "step": e.step, "action": e.action, "detail": e.detail,
                   "confidence": e.confidence, "review": "⚑" if e.flagged_for_review else ""}
                  for o2 in outputs for e in o2["audit"] if (e.flagged_for_review or not only_flag)],
                 width="stretch", hide_index=True)

    # --- downloads ---------------------------------------------------------------------------
    st.markdown("### ⬇️ Downloads")
    from prostate_deid.convert import output_filename

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for suid, s2 in series.items():
            d0 = s2[0]["ds"]
            folder = f"{d0.PatientID}/S{int(getattr(d0, 'SeriesNumber', 0) or 0):03d}"
            for o2 in s2:
                b = io.BytesIO()
                o2["ds"].save_as(b, enforce_file_format=True)
                zf.writestr(f"{folder}/{output_filename(o2['ds'])}", b.getvalue())
        zf.writestr("audit_trail.json", json.dumps(
            [{"step": e.step, "action": e.action, "detail": e.detail, "confidence": e.confidence,
              "flagged_for_review": e.flagged_for_review} for e in all_audit], indent=1))
    pid = str(outputs[0]["ds"].PatientID)
    st.download_button("⬇️ De-identified DICOM (.zip, all series + audit)", data=zbuf.getvalue(),
                       file_name=f"{pid}_deid_dicom.zip", mime="application/zip")

    st.markdown("#### 🔄 Series → .nii.gz")
    if st.button("Convert selected series to .nii.gz"):
        try:
            norm = None
            if rarn:
                from prostate_deid.preprocess import IntensityNormalize

                norm = IntensityNormalize(always_apply=True)
            gz, sidecar, naudit = series_to_nifti([o2["ds"] for o2 in s], normalize=norm)
            stem = f"{s[0]['ds'].PatientID}_S{int(getattr(s[0]['ds'], 'SeriesNumber', 0) or 0):03d}"
            for e in naudit:
                (st.warning if e.flagged_for_review else st.caption)(str(e))
            st.download_button("⬇️ .nii.gz", data=gz, file_name=f"{stem}.nii.gz", mime="application/gzip")
            st.download_button("⬇️ JSON sidecar", data=json.dumps(sidecar, indent=1, default=str),
                               file_name=f"{stem}.json", mime="application/json")
        except ValueError as exc:
            st.error(f"Cannot write this series as NIfTI: {exc}")
else:
    st.info("👋 Upload DICOM files (or a .zip of a study), a NIfTI volume, or an image to get started.")
