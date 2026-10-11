# v2 analysis: removing burned-in text while keeping the image

This note records the analysis done before changing the burned-in text step,
and the evidence behind the v2 defaults. Everything here is reproducible with:

```
python benchmarks/text_removal_benchmark.py      # writes benchmarks/results/
```

## 1. What v1 does and why it loses image data

v1 (`universal_io.redact_burned_in_text`) runs Tesseract over each slice at
three upscale factors and both polarities, keeps word boxes that pass a
"looks like rendered text" pixel test, merges them into row bands and **sets
every pixel in each padded row box to 0**. Two problems follow:

1. **Every pixel inside the box is destroyed**, including the anatomy between
   and around the letter strokes. In the benchmark only 10% of the pixels in
   v1's boxes were left unchanged.
2. **Text drawn on anatomy is often not found at all.** v1 runs Tesseract
   with `--psm 6` (one uniform text block) on the whole slice. That works on
   corner annotations over a flat black background, but anatomy texture
   breaks Tesseract's page-layout step. v1 also shrinks every slice to 300 px
   wide before OCR, which makes text on large slices too small to read. On
   fusion-biopsy-style labels drawn over tissue, v1 detected nothing in most
   cases, and **71% of those characters (worst case 95%) were still readable
   after v1**.

## 2. Benchmark design

* **Images:** real public MR slices, with no patient data: two slices of the
  scikit-image brain MRI (256 px), the pydicom `MR2_UNCI` Philips slice
  (1024 px) and the pydicom Siemens slice (484 px).
* **Burned-in PHI:** fictitious names, IDs, dates and an institution, drawn
  with three fonts (Lucida Console, Arial, Consolas) at about 3% of the image
  width, in three layouts:
  * corner annotations on a dark background;
  * a bright label over anatomy;
  * a dark label over anatomy.

  This gives 24 cases. The clean slice and the exact per-pixel glyph coverage
  are kept as ground truth.
* **Fair comparison:** all v2 methods use the same detector boxes for a case.
  "Oracle" rows use the true glyph mask instead, to separate segmentation
  quality from inpainting quality.
* **Metrics:**
  * *privacy*: share of the original characters Tesseract can still read in
    each text line of the output. Because OCR picks up stray characters, this
    has a floor of about 3–5% even when the text is fully removed;
  * *retention*: share of pixels inside the detected boxes left
    bit-identical;
  * *fidelity*: MAE, PSNR and SSIM against the **clean** slice over the text
    footprint;
  * *ghost ratio*: how visible the letter silhouette still is (see §4).

## 3. Results (24 cases)

| Method | Text readable afterwards (mean / worst) | Box pixels kept | MAE vs clean | SSIM vs clean | Fallbacks to blackout |
|---|---|---|---|---|---|
| v1 (black box, v1 detector) | 33.6% / 95.2% | 9.8% | 0.188 | 0.51 | – |
| v2 detector + black box | 6.1% / 28.6% | 11.0% | 0.171 | 0.40 | 0 |
| **v2 Telea (default)** | **5.0% / 28.6%** | **55.7%** | **0.041** | **0.77** | **5** |
| v2 Navier–Stokes | 5.6% / 28.6% | 52.5% | 0.037 | 0.79 | 10 |
| v2 biharmonic | 3.6% / 28.6% | 47.9% | 0.054 | 0.71 | 11 |
| v2 ViT-MAE + Poisson | 4.4% / 28.6% | 39.9% | 0.075 | 0.65 | 15 |
| v2 plain ViT-MAE | 5.5% / 28.6% | 20.8% | 0.104 | 0.53 | 49 |

Text left readable, by layout (mean / worst case):

| Layout | v1 | v2 Telea |
|---|---|---|
| Corner annotations | 3.3% / 5.3% | 4.9% / 6.3% (OCR noise floor) |
| Bright label over anatomy | **71.4% / 95.2%** | 4.8% / 23.8% |
| Dark label over anatomy | 26.2% / 81.0% | 5.4% / 28.6% |

With the oracle (true glyph) mask, i.e. inpainting quality alone, PSNR over the text footprint:

| Method | All cases | Text over anatomy | SSIM |
|---|---|---|---|
| biharmonic | 45.3 dB | 32.6 dB | 0.959 |
| **ViT-MAE + Poisson** | 41.5 dB | **33.1 dB** | 0.958 |
| Navier–Stokes | 42.0 dB | 28.5 dB | 0.952 |
| Telea | 41.2 dB | 28.0 dB | 0.951 |
| plain ViT-MAE | 30.7 dB | 29.2 dB | 0.832 |

Visual examples (crop around the text; truth, burned-in, v1, then each v2
method): `benchmarks/results/panel_*.png`.

## 4. Findings

1. **Detection was the biggest privacy gap, not inpainting.** Adding a
   detector that finds text lines from pixels (near-white or near-black thin
   strokes, found with a top-hat filter, grouped into lines), confirms each
   by OCR on the crop at native resolution, and then absorbs neighbouring
   words on the same row, cut readable overlay text from 71% to about 5%.
   v1's detector is kept unchanged, so corner-annotation recall is unchanged.
2. **Replacing only the letter pixels keeps over half of each box.** v2 keeps
   about 50–56% of box pixels bit-identical, against 10% for v1. Restored
   pixels are much closer to the truth than black (MAE 0.04 vs 0.17–0.19).
   Every pixel outside the glyph mask is bit-identical to the input; this is
   asserted in the tests.
3. **A plain ViT inpainter leaks PHI.** Pasting the ViT-MAE reconstruction
   into the letter-shaped hole leaves a faint silhouette: the predicted grey
   level is slightly off on out-of-domain MRI. It can be read by OCR (49.5%
   of corner characters with the oracle mask). This motivated two safeguards
   that now apply to every method:
   * **Poisson (gradient-domain) blending:** the ViT supplies the structure
     and the real surrounding pixels supply the level. This removes the
     silhouette (ghost ratio 6.9 → 0.63) and makes ViT-MAE + Poisson the
     most accurate restorer on text over anatomy when the mask is exact
     (33.1 dB).
   * **A silhouette ("ghost ratio") check plus a stricter re-OCR.** Any
     region that still looks like text falls back to v1's black box and is
     flagged.

   Plain `vit_mae` is kept for research only and every use is flagged; it is
   not offered in the dashboard.
4. **Why Telea is the default.** With real (detector) masks, Telea has the
   fewest fallbacks (5/24 cases), the highest retention (55.7%) and SSIM 0.77.
   ViT-MAE + Poisson is best with exact masks, but with detector masks its
   fallback rate is higher (15). It is offered as an option (`[vit]` extra);
   the pretrained model is ImageNet-trained, and fine-tuning on prostate MRI
   is the obvious next step.
5. **The polarity rule had to be thinness-based.** "Text is the minority
   extreme class" failed when dark anatomy next to white text was rarer than
   the text. "Text is the extreme class made of thin strokes" (the share
   removed by a morphological opening slightly wider than one stroke) gave
   the correct polarity in every case, with 98–100% glyph-pixel recall.

## 5. Limitations (stated plainly)

* **Low-contrast text over similar tissue is still missed** (for example dark
  text on dark brain tissue, or white text on a bright gland). It is also
  hard for OCR to read, but it is not zero risk. Human review stays
  mandatory (MIDI report).
* **Restored pixels are estimates, not measurements.** They are recorded in
  the per-slice inpaint mask and the audit trail, and must not be used for
  quantitative analysis (e.g. ADC values) or diagnosis.
* **The benchmark is synthetic PHI on public MR slices**, not real scanner
  screenshots or prostate studies. Before any accuracy claim in the paper,
  recall must be measured on the MIDI / TCIA Pseudo-PHI data.
* **The ViT is a generic ImageNet MAE** (facebook/vit-mae-base, 112M
  parameters), not trained on MRI.

## 6. Algorithm summary (as implemented in `text_restore.py`)

1. **Detect:** v1 Tesseract rows plus OCR-confirmed overlay-text lines.
2. **Segment** inside each box:
   * polarity = the extreme class made of thin strokes;
   * stroke width from a distance transform;
   * top-hat filter with a disk just wider than one stroke;
   * seeds = extreme-valued pixels with a strong response;
   * bounded region growing through weaker responses (anti-aliased edges);
   * 1 px safety dilation;
   * a mask that is implausible (empty, or more than 85% of the box) falls
     back to blackout.
3. **Restore** only the masked pixels: Telea, NS, biharmonic or ViT-MAE +
   Poisson.
4. **Verify:** v1 detector on the full restored slice, crop OCR (confidence
   ≥ 40) and the ghost ratio (≤ 1.5). Any failure → black box + review flag.
