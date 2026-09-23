# Robust Anatomy-Referenced Normalization (RARN)
### Mathematics of the `prostate-deid` preprocessing stage
Repository: https://github.com/drsaqibiqbal/annolib · module: `prostate_deid.preprocess`

---

## Integrity note (read first)

This document is precise about what is established and what is the
contribution. The robust estimators used here — the median, the median
absolute deviation (MAD), and the consistency constant `1.4826` — are
**standard robust statistics** and are *not* claimed as novel. Reference-region
intensity normalization for MRI is also prior art (e.g. WhiteStripe, which
normalizes brain MRI to normal-appearing white matter).

**The contribution of RARN is the specialization + integration:** (i) using the
prostate **gland / zonal labels** the library already carries as the reference
region; (ii) an optional **zone-aware** mode that standardizes the peripheral
and transition zones separately; and (iii) recording the exact normalization
affine in the **same audit trail** as the de-identification steps, which makes
the whole preprocessing chain reproducible and **exactly invertible on the
unclipped support**. Points (i)–(iii) together are, to our knowledge, not
offered by existing tools; that is the honest, defensible claim for the paper.

---

## 1. Problem statement

MRI intensities are not quantitative. For an image `I : Ω → ℝ` over voxels
`Ω`, the raw value at a voxel depends on scanner, coil, sequence, and gain, so
the *same tissue* maps to different numbers across acquisitions. Concretely,
two acquisitions of the same anatomy are related (to first order) by an unknown
affine gain

```
    I_b(x) ≈ a · I_a(x) + b ,      a > 0,  b ∈ ℝ                       (1)
```

Any downstream model that assumes intensity comparability must first undo this
nuisance transform. A normalizer `N` should therefore be **affine-equivariant**
in the sense that it maps both `I_a` and `I_b` to (approximately) the same
standardized image:

```
    N(I_b) ≈ N(I_a)    whenever (1) holds.                            (2)
```

The whole-image z-score `(I − mean)/std` satisfies (2) *only if* the mean and
std are computed over corresponding voxels. In pelvic MRI they are not: air,
background, and a variable field of view dominate the histogram and are not
stable across scans, so whole-image statistics drift. RARN fixes the reference
to an anatomy that *is* comparable across scans.

---

## 2. Reference set

Let `R ⊆ Ω` be a reference region and define the reference intensity multiset

```
    𝓡 = { I(x) : x ∈ R } .                                           (3)
```

RARN chooses `R` in priority order:

1. **Anatomical mask** — `R = { x : M(x) ∈ L }`, where `M` is the segmentation
   label field and `L` is `reference_labels` (default `WHOLE_GLAND = {PZ, TZ,
   lesion} = {2,3,4}`). This is the preferred reference: prostate tissue is the
   stable, comparable anatomy of interest.

2. **Otsu foreground** — when no mask is present, `R = { x : I(x) ≥ τ }` with
   `τ` the Otsu threshold that maximizes between-class variance

   ```
       τ = argmax_t  ω₀(t) ω₁(t) [ μ₀(t) − μ₁(t) ]² ,                 (4)
   ```

   where `ω₀,ω₁` are the mass and `μ₀,μ₁` the means of the below/above-`t`
   classes. This removes air/background without needing a segmentation.

3. **Whole image** — fallback.

---

## 3. Robust location and scale

From `𝓡`, RARN estimates a location `μ_R` and scale `σ_R`. In `"robust"` mode:

```
    μ_R  = median(𝓡)                                                 (5)

    MAD  = median_{x∈R} | I(x) − median(𝓡) |                         (6)

    σ_R  = 1.4826 · MAD                                               (7)
```

**Why these estimators.**
- The **median** (5) has a breakdown point of 50%: up to half the reference
  voxels can be arbitrary outliers (biopsy clips, hemorrhage, a stray bright
  vessel) before the estimate is corrupted. The mean has a breakdown point of
  0%.
- The **MAD** (6) is the matching 50%-breakdown scale estimator.
- The constant **1.4826 = 1 / Φ⁻¹(0.75)** rescales MAD so that, for Gaussian
  data, `σ_R` is a *consistent* estimator of the standard deviation. (For
  `X ~ N(μ,σ²)`, `median(|X−μ|) = σ·Φ⁻¹(0.75) ≈ 0.6745 σ`, and
  `1/0.6745 ≈ 1.4826`.) So RARN's scale agrees with the classical SD on clean
  Gaussian tissue but resists outliers.

If the reference is (near-)constant, `σ_R = 0`; RARN then falls back to the
ordinary standard deviation, guarded to be positive, to avoid division by zero.

The `"zscore"` mode replaces (5)–(7) with the mean and SD of the *winsorized*
reference (Section 4), and `"minmax"` uses the winsorization bounds directly as
`(location, scale) = (q_lo, q_hi − q_lo)`.

---

## 4. Winsorization and the normalizer

Let `q_α` denote the `α`-percentile of `𝓡`, and pick tail percentiles
`(p_lo, p_hi)` (default `(1, 99)`). The RARN map is

```
                clip( I(x), q_{p_lo}, q_{p_hi} ) − μ_R
    Ĩ(x)  =  ─────────────────────────────────────────── .           (8)
                              σ_R
```

Clipping (winsorization) caps extreme intensities *before* standardization so
a handful of hyperintense voxels cannot stretch the output scale; the location
and scale in the denominator already come from the robust estimators of
Section 3.

**Affine-equivariance (property (2)).** Under the scanner gain (1), the
reference statistics transform as `μ_R ↦ a μ_R + b` and `σ_R ↦ a σ_R`, and the
percentiles as `q_α ↦ a q_α + b`. Substituting into (8), the `a` and `b`
cancel, so `Ĩ_b(x) = Ĩ_a(x)` for all unclipped `x`. This is exactly the
scanner-invariance the tests verify empirically (two synthetic "scanners" with
`a=2.5, b=40` normalize to the same distribution).

---

## 5. Audit-reversibility

Equation (8) is affine in `I(x)` on the region where no clipping occurs
(`q_{p_lo} ≤ I(x) ≤ q_{p_hi}`). Its inverse there is simply

```
    I(x)  =  Ĩ(x) · σ_R + μ_R .                                      (9)
```

RARN writes `(μ_R, σ_R, q_{p_lo}, q_{p_hi}, reference-kind)` into an
`AuditEntry` in the same audit trail as the de-identification steps. Therefore
the normalization is not an opaque, one-way transform: its exact parameters are
recorded, the mapping is reproducible, and (9) recovers the original
intensities wherever winsorization did not truncate them. This is the direct
tie-in to the library's distinguishing feature — provenance-by-construction —
and is what makes the *preprocessing* auditable in the same way the
*de-identification* already is.

---

## 6. Zone-aware extension

Prostate T2W intensities differ systematically between the peripheral zone
(PZ) and transition zone (TZ). In `"zone_aware"` mode RARN first applies the
global map (8), then, for each present zone label `ℓ ∈ L`, recomputes robust
statistics `(μ_ℓ, σ_ℓ)` from that zone alone and standardizes its voxels with
its own affine:

```
    for x with M(x) = ℓ :   Ĩ(x) = ( I(x) − μ_ℓ ) / σ_ℓ .           (10)
```

Each zone is thereby aligned to `N(0,1)`, removing *inter-scan, per-zone*
intensity drift. **Tradeoff (documented, not hidden):** because PZ and TZ are
mapped by *different* affines, (10) alters the inter-zone contrast ratio; it is
appropriate when the downstream task is defined per zone (e.g. PI-RADS-style
zonal assessment) and inappropriate when absolute inter-zone contrast must be
preserved — in which case the global mode (8) is the right choice. If no mask
is available, `"zone_aware"` degrades to the global robust map.

---

## 7. Where it sits in the API

RARN is an ordinary `Transform`, so it composes in a `Compose` pipeline exactly
like the de-identification steps and shares their audit trail:

```python
from prostate_deid import (Compose, Pseudonymize, DicomTagScrub,
                           IntensityNormalize, load_sample, save_nifti)

sample = load_sample("study.dcm")
pipe = Compose([
    Pseudonymize(mapping_store="/secure/key.json", always_apply=True),
    DicomTagScrub(profile="hipaa_safe_harbor",
                  exclude_tags=["PatientID","StudyDate","StudyInstanceUID",
                                "SeriesInstanceUID","SOPInstanceUID"],
                  always_apply=True),
    IntensityNormalize(method="robust", always_apply=True),   # <-- RARN
])
result = pipe(sample)
save_nifti(result, "study.nii")     # DICOM header PHI does not survive to NIfTI
```

This realizes the requested `DICOM → NIfTI → normalization` preprocessing as a
first-class part of the same audited pipeline: de-identify, normalize with
recorded/invertible parameters, and export to a header-PHI-free NIfTI.

---

## 8. Summary of claims for the manuscript

| Element | Status |
|---|---|
| Median / MAD / 1.4826 constant | established robust statistics (cited, not claimed) |
| Reference-region normalization | prior art (WhiteStripe et al.) |
| Prostate gland/zonal reference | **contribution** (specialization) |
| Zone-aware dual standardization (10) | **contribution** |
| Audit-recorded, invertible affine (9) | **contribution** (integration) |
| Scanner-invariance (2) via (8) | proved above, verified in tests |

The defensible novelty is the **combination**, not any single formula.
Quantitative benefit (e.g. improved cross-scanner segmentation Dice) still
requires the validation described in the technical-details document.
