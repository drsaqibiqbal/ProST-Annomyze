"""
Benchmark: v1 black-box redaction vs v2 segment-and-restore, on real MR slices
with synthetic burned-in PHI whose ground truth is known exactly.

For every case the clean slice, the burned slice and the per-pixel glyph
coverage (alpha) are known, so we can measure:

  * privacy   -- residual legible characters (Tesseract re-read of each text
                 line in the output) and glyph pixels left unmodified;
  * retention -- how many pixels inside the detected boxes were left
                 bit-identical (v1 destroys all of them);
  * fidelity  -- error of the output vs the CLEAN slice inside the text
                 footprint (MAE, PSNR, SSIM), i.e. how close the restored
                 pixels are to what was really there before text was burned in.

Every method runs on the SAME detector boxes per case, so differences come
only from segmentation + restoration. "oracle" rows use the ground-truth glyph
mask to isolate inpainting quality from segmentation quality.

Run:  python benchmarks/text_removal_benchmark.py  [--out benchmarks/results]
No real patient data is used: MR slices are public test images bundled with
pydicom / scikit-image, and all PHI strings are fictitious.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prostate_deid import universal_io as uio  # noqa: E402
from prostate_deid.text_restore import (  # noqa: E402
    detect_text_boxes_v2,
    ghost_ratio,
    inpaint_mask,
    restore_burned_in_text,
    vit_mae_available,
)

FONTS = {
    "lucida_console": "C:/Windows/Fonts/lucon.ttf",
    "arial": "C:/Windows/Fonts/arial.ttf",
    "consolas": "C:/Windows/Fonts/consola.ttf",
}

CORNER_LINES = ["DOE^JANE  F  67Y", "ID: SYN-0012345", "DOB 1958-03-14", "2026-10-11 09:42"]
RIGHT_LINES = ["Synthetic General Hosp", "Dr. R. Smith"]
OVERLAY_LINE = "Lesion PZ 12mm  DOE J"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_slices() -> list[tuple[str, np.ndarray]]:
    import pydicom
    from pydicom.data import get_testdata_file

    out = []
    try:
        import skimage.data

        brain = skimage.data.brain()  # (10, 256, 256) uint16, public MR
        for k in (3, 6):
            out.append((f"skimage_brain_s{k}", brain[k]))
    except Exception as exc:  # pragma: no cover - needs pooch download
        print("skimage brain unavailable:", exc)
    for name in ("MR2_UNCI.dcm", "MR-SIEMENS-DICOM-WithOverlays.dcm"):
        ds = pydicom.dcmread(get_testdata_file(name))
        out.append((name.replace(".dcm", ""), np.asarray(ds.pixel_array)))
    return out


def _render_alpha(shape, items, font_path, size_px) -> tuple[np.ndarray, list[dict]]:
    """Render text lines into an alpha layer; return alpha in [0,1] and per-line metadata."""
    h, w = shape
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    font = ImageFont.truetype(font_path, size_px)
    lines = []
    for text, (x, y) in items:
        draw.text((x, y), text, fill=255, font=font)
        bb = draw.textbbox((x, y), text, font=font)
        lines.append({"text": text, "bbox": [int(v) for v in bb]})
    return np.asarray(layer, dtype=np.float32) / 255.0, lines


def make_cases(slices):
    cases = []
    for name, img in slices:
        img = img.astype(np.float32)
        h, w = img.shape
        lo, hi = float(img.min()), float(img.max())
        size = max(9, round(w / 34))  # scanner overlay text is ~3% of image width
        lh = round(size * 1.25)
        corner = [(t, (max(2, w // 60), max(2, h // 60) + k * lh)) for k, t in enumerate(CORNER_LINES)]
        right = [(t, (w - round(size * 0.62 * len(t)) - w // 60, h // 60 + k * lh)) for k, t in enumerate(RIGHT_LINES)]
        # Overlay on anatomy: put the label over the brightest-textured band near the centre.
        overlay = [(OVERLAY_LINE, (w // 2 - round(size * 0.62 * len(OVERLAY_LINE)) // 2, h // 2))]
        for font_name, font_path in FONTS.items():
            if font_name != "lucida_console" and name not in ("skimage_brain_s3", "MR2_UNCI"):
                continue  # keep the run short: every slice gets the scanner font, two get all fonts
            for layout, items, polarity in (
                ("corner_annotations", corner + right, "bright"),
                ("overlay_on_anatomy", overlay, "bright"),
                ("dark_overlay_on_anatomy", overlay, "dark"),
            ):
                alpha, lines = _render_alpha((h, w), items, font_path, size)
                level = hi if polarity == "bright" else lo
                burned = img * (1 - alpha) + level * alpha
                burned = np.clip(np.rint(burned), lo, hi).astype(np.uint16 if lo >= 0 else np.int16)
                cases.append({
                    "id": f"{name}|{font_name}|{layout}",
                    "slice": name, "font": font_name, "layout": layout,
                    "clean": img, "burned": burned, "alpha": alpha, "lines": lines,
                })
    return cases


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _norm(a, lo, hi):
    return (a.astype(np.float64) - lo) / max(hi - lo, 1e-9)


def residual_chars(out, lines, lo, hi) -> float:
    """Fraction of each original text line's characters Tesseract can still read from the output."""
    import pytesseract

    uio._ocr_available()
    u8 = (np.clip(_norm(out, lo, hi), 0, 1) * 255).astype(np.uint8)
    total, matched = 0, 0
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        pad = max(3, (y1 - y0) // 2)
        crop = u8[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        if crop.size == 0:
            continue
        best = 0
        for inv in (False, True):
            c = 255 - crop if inv else crop
            pil = Image.fromarray(c).resize((c.shape[1] * 4, c.shape[0] * 4), Image.LANCZOS)
            txt = pytesseract.image_to_string(pil, config="--psm 7").strip()
            sm = difflib.SequenceMatcher(None, ln["text"].lower(), txt.lower())
            best = max(best, sum(b.size for b in sm.get_matching_blocks()))
        total += len(ln["text"])
        matched += best
    return matched / max(total, 1)


def ssim_bbox(a, b, mask):
    from skimage.metrics import structural_similarity

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return float("nan")
    m = 3
    y0, y1 = max(0, ys.min() - m), ys.max() + 1 + m
    x0, x1 = max(0, xs.min() - m), xs.max() + 1 + m
    pa, pb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    win = min(7, (min(pa.shape) // 2) * 2 + 1)
    if win < 3:
        return float("nan")
    return float(structural_similarity(pa, pb, data_range=1.0, win_size=win))


def score(case, out, boxes, seconds):
    clean, burned, alpha = case["clean"], case["burned"], case["alpha"]
    lo, hi = float(clean.min()), float(clean.max())
    glyph = alpha >= 0.5
    footprint = alpha > 0.02
    o, c = _norm(out, lo, hi), _norm(clean, lo, hi)
    changed = out.astype(np.int64) != burned.astype(np.int64)
    box_mask = np.zeros(alpha.shape, dtype=bool)
    for x0, y0, x1, y1, *_ in boxes:
        box_mask[y0:y1, x0:x1] = True
    err = np.abs(o - c)[footprint]
    mse = float(np.mean(err ** 2)) if err.size else float("nan")
    return {
        "residual_chars": residual_chars(out, case["lines"], lo, hi),
        "glyph_px_unmodified": float((~changed & glyph).sum() / max(glyph.sum(), 1)),
        "box_px_retained": float((~changed & box_mask).sum() / max(box_mask.sum(), 1)),
        "non_text_px_altered": int((changed & ~footprint).sum()),
        "mae_text_footprint": float(err.mean()) if err.size else float("nan"),
        "psnr_text_footprint": float(10 * np.log10(1.0 / mse)) if mse and mse > 0 else float("inf"),
        "ssim_text_bbox": ssim_bbox(o, c, footprint),
        "ghost_ratio": ghost_ratio(o.astype(np.float32), changed) if changed.any() else 0.0,
        "seconds": seconds,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_v1_module(v1_root: Path):
    """Import the untouched v1 universal_io.py from the v1 checkout under a distinct module name."""
    path = v1_root / "src" / "prostate_deid" / "universal_io.py"
    spec = importlib.util.spec_from_file_location("v1_universal_io", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    ap.add_argument("--v1-root", default=str(ROOT.parent / "ProST-Annomyze"))
    ap.add_argument("--no-vit", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not uio._ocr_available():
        sys.exit("Tesseract is required for this benchmark.")
    v1 = load_v1_module(Path(args.v1_root)) if Path(args.v1_root).exists() else None
    methods = ["blackout", "telea", "ns", "biharmonic"]
    if not args.no_vit and vit_mae_available():
        methods += ["vit_mae", "vit_mae_poisson"]

    cases = make_cases(load_slices())
    print(f"{len(cases)} cases, methods={methods}")
    rows = []
    examples = {}
    for case in cases:
        burned = case["burned"]
        lo, hi = float(case["clean"].min()), float(case["clean"].max())
        boxes_v1 = uio.detect_text_boxes(burned.astype(np.float32))
        t = time.perf_counter()
        boxes = detect_text_boxes_v2(burned.astype(np.float32))
        t_detect = time.perf_counter() - t
        n_lines = len(case["lines"])
        det = {"case": case["id"], "boxes_v1": len(boxes_v1), "boxes": len(boxes), "text_lines": n_lines}

        if v1 is not None:  # v1 reference + refactor equivalence check
            t = time.perf_counter()
            v1_out, _ = v1.redact_burned_in_text(burned[None].astype(np.float32))
            v1_secs = time.perf_counter() - t
            v1_out = v1_out[0].astype(burned.dtype)
            r = {**det, "method": "v1_redact", "mask": "detector", **score(case, v1_out, boxes_v1, v1_secs)}
            rows.append(r)
            eq = restore_burned_in_text(burned[None], method="blackout", boxes_per_slice=[boxes_v1]).pixels[0]
            rows[-1]["identical_to_v1"] = bool(np.array_equal(eq, v1_out))
            examples.setdefault(case["id"], {})["v1_redact"] = v1_out

        for method in methods:
            t = time.perf_counter()
            res = restore_burned_in_text(burned[None], method=method, boxes_per_slice=[boxes], verify=True)
            secs = time.perf_counter() - t + t_detect
            out = res.pixels[0]
            r = {**det, "method": method, "mask": "detector",
                 "fallbacks": sum(x.action == "fallback_blackout" for x in res.regions),
                 **score(case, out, boxes, secs)}
            rows.append(r)
            examples.setdefault(case["id"], {})["v2_" + method] = out

        # Oracle: ground-truth glyph mask (dilated 1px), inpainting only.
        from scipy import ndimage as ndi

        gt = ndi.binary_dilation(case["alpha"] > 0.02, iterations=1)
        g01 = _norm(burned, lo, hi).astype(np.float32)
        for method in methods[1:]:
            t = time.perf_counter()
            rest = inpaint_mask(g01, gt, method)
            secs = time.perf_counter() - t
            out = np.clip(np.rint(rest * (hi - lo) + lo), lo, hi).astype(burned.dtype)
            out = np.where(gt, out, burned)
            rows.append({**det, "method": method, "mask": "oracle", **score(case, out, boxes, secs)})
        print(f"  {case['id']}: v1 {len(boxes_v1)} / v2 {len(boxes)} boxes, {n_lines} lines", flush=True)
        examples[case["id"]]["_burned"] = burned
        examples[case["id"]]["_clean"] = case["clean"].astype(burned.dtype)

    with open(out_dir / "rows.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)
    summary = summarize(rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    save_panels(examples, out_dir)
    print_summary(summary)


def summarize(rows):
    keys = ["residual_chars", "glyph_px_unmodified", "box_px_retained", "non_text_px_altered",
            "mae_text_footprint", "psnr_text_footprint", "ssim_text_bbox", "ghost_ratio", "seconds"]
    groups = {}
    for r in rows:
        groups.setdefault((r["mask"], r["method"]), []).append(r)
    out = []
    for (mask, method), rs in sorted(groups.items()):
        row = {"mask": mask, "method": method, "n": len(rs)}
        for k in keys:
            vals = np.array([x[k] for x in rs], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            row[k] = float(vals.mean()) if vals.size else float("nan")
        row["max_residual_chars"] = float(max(x["residual_chars"] for x in rs))
        if any("fallbacks" in x for x in rs):
            row["fallbacks"] = int(sum(x.get("fallbacks", 0) for x in rs))
        if any("identical_to_v1" in x for x in rs):
            row["identical_to_v1"] = all(x.get("identical_to_v1", False) for x in rs)
        for layout in ("corner_annotations", "overlay_on_anatomy", "dark_overlay_on_anatomy"):
            sub = [x for x in rs if x["case"].split("|")[2] == layout]
            if sub:
                row[f"psnr_{layout}"] = float(np.mean([min(x["psnr_text_footprint"], 99) for x in sub]))
                row[f"residual_{layout}"] = float(np.mean([x["residual_chars"] for x in sub]))
        out.append(row)
    return out


def print_summary(summary):
    cols = ["mask", "method", "n", "residual_chars", "max_residual_chars", "glyph_px_unmodified",
            "box_px_retained", "mae_text_footprint", "psnr_text_footprint", "ssim_text_bbox", "ghost_ratio", "seconds",
            "fallbacks"]
    print("\n" + " | ".join(cols))
    for r in summary:
        print(" | ".join(
            f"{r.get(c, ''):.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols
        ))


def save_panels(examples, out_dir: Path):
    """One PNG per case: clean | burned | each method (crop around the text), for the report."""
    for cid, outs in examples.items():
        if "|lucida_console|" not in cid:
            continue
        clean, burned = outs["_clean"], outs["_burned"]
        changed = np.any([o != burned for k, o in outs.items() if not k.startswith("_")], axis=0)
        ys, xs = np.nonzero(changed) if changed.any() else (np.array([0]), np.array([0]))
        m = 12
        y0, y1 = max(0, ys.min() - m), min(clean.shape[0], ys.max() + m)
        x0, x1 = max(0, xs.min() - m), min(clean.shape[1], xs.max() + m)
        lo, hi = float(clean.min()), float(clean.max())
        tiles = [("clean (truth)", clean), ("burned-in", burned)] + [(k, v) for k, v in outs.items() if not k.startswith("_")]
        crops = [((np.clip(_norm(t[y0:y1, x0:x1], lo, hi), 0, 1)) * 255).astype(np.uint8) for _, t in tiles]
        scale = max(1, 300 // max(1, crops[0].shape[1]))
        ims = [Image.fromarray(c).resize((c.shape[1] * scale, c.shape[0] * scale), Image.NEAREST) for c in crops]
        wt, ht = ims[0].size
        sheet = Image.new("L", (wt, (ht + 16) * len(ims)), 255)
        d = ImageDraw.Draw(sheet)
        for k, (im, (label, _)) in enumerate(zip(ims, tiles)):
            d.text((2, k * (ht + 16) + 2), label, fill=0)
            sheet.paste(im, (0, k * (ht + 16) + 16))
        safe = cid.replace("|", "__")
        sheet.save(out_dir / f"panel_{safe}.png")


if __name__ == "__main__":
    main()
