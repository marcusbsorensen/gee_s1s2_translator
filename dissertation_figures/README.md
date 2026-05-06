# The Shape of Fire — figure set

Five figures built from the `gee_s1s2_translator` operational pipeline
to accompany Sonia's dissertation. Each figure supports a specific
argument; the captions below pair the figure with the dissertation
section it informs.

Generated 2026-05-06.

## Figure 1 — `figure_1_shape_over_time_brentmoor.png`

**Argument:** the dNBR signal of a small heath fire is strongest in the
days immediately after the burn and decays toward the background
within weeks as heath regrowth fills in. The model's hybrid prediction
at 60 days post-fire shows the variance-collapse limit — it cannot
recover the small-fire dNBR signature that has already faded in the
optical record.

Three panels (real / real / hybrid) over Brentmoor Heath, all in the
same RdYlGn_r colour ramp at [-0.5, 0.5] dNBR, cropped to a 1.5 km
square around the SWT field-mapped perimeter (~0.33 ha). Real S2
panels are date-mosaicked across the T30UXB / T30UXC tile boundary
that crosses the AOI.

- **Panel A:** real Sentinel-2 26-Apr-2022 minus real Sentinel-2
  14-Aug-2022 (5 days post-fire). Median dNBR inside SWT perimeter
  **+0.401**, outside +0.003. Strong burn signature.
- **Panel B:** real 26-Apr-2022 minus real 18-Oct-2022 (70 days post-
  fire). Median inside +0.116, outside −0.113. Burn signal has
  decayed and AOI-wide seasonal yellowing dominates.
- **Panel C:** real 26-Apr-2022 minus **predicted** S2 from cloud-
  covered S1 at 08-Oct-2022 (60 days post-fire). Median inside
  −0.220 (negative; the variance-collapsed prediction reads as
  greener than truth inside the perimeter), outside −0.067. The
  model cannot recover the small-fire dNBR signal at this temporal
  interval.

**Pairs with:** dissertation discussion of temporal interval +
operational scope note (Brentmoor-class fires <1 ha at >6 weeks fall
outside the operational window).

## Figure 2 — `figure_2_shape_at_scale.png`

**Argument:** the model is operationally credible for fire-scar
mapping on substantial fires (Poors-Allotment-class, ~6 ha) at
moderate post-fire intervals. It is not credible on small fires
(Brentmoor-class, <1 ha), even though it makes spatially-coherent
predictions everywhere. The fire-size dependence is a property of
the model's variance-retention floor, not of the prediction's
spatial structure.

Two panels at the same temporal interval (60 days post-fire,
predicted S2 from S1 at 08-Oct-2022), both in the same dNBR colour
ramp at [-0.5, 0.5], cropped to 1.5 km around each SWT perimeter:

- **Panel A — Brentmoor (~0.33 ha):** median hybrid dNBR inside
  perimeter −0.220, outside −0.067. No detectable burn signal.
- **Panel B — Poors Allotment (~6.79 ha):** median hybrid dNBR
  inside perimeter **+0.288**, outside +0.068. Clear burn signature
  visible as a red patch matching the perimeter shape.

**Pairs with:** dissertation operational-deployment discussion +
the fire-size operational scope note in the Wednesday delivery
package.

## Figure 3 — `figure_3_shape_across_regions.png`

**Argument:** the per-pixel reconstruction quality of the predicted
RGB tracks landscape spatial homogeneity across three out-of-
distribution sites. Geographic distance from training is not the
principal predictor of transfer quality.

Three rows × two columns, all RGB at fixed [0, 0.3] reflectance
stretch:

- **Row 1 — Hankley Common** (in-region training site, OOD via
  held-out test split, 20-May-2024): truth and prediction closely
  match; B08 MAE 0.067, driver-band variance retention 73 %.
  Italic note explains that the prediction appears slightly darker
  than truth at the shared stretch because the bright-pixel high
  tail (p98) is compressed; band medians match within ±5 %.
- **Row 2 — Cavenham Heath** (Suffolk Brecks, ~100 km OOD,
  26-Jun-2024): sharp field-boundary mosaic in truth becomes
  visibly smoothed in prediction; B08 MAE 0.102, driver var 42 %
  (collapsed).
- **Row 3 — Berwyn SSSI** (north Wales upland heath, ~250 km OOD,
  02-Jun-2024): spatially-homogeneous upland heath; prediction is
  recognisably the same landscape; B08 MAE 0.126, driver var 63 %.

The Hankley → Cavenham → Berwyn pattern shows variance retention
*does not monotonically degrade with distance*. Berwyn is the
furthest OOD site but lands between Hankley and Cavenham on
driver-band retention because upland heath is more spatially uniform
than the Brecks heath/arable mosaic.

**Pairs with:** dissertation transfer-claim section + the spatial-
homogeneity operational note in `OPERATIONAL_DEPLOYMENT.md`.

## Figure 4 — `figure_4_capacity_evolution.png`

**Argument:** the operational pipeline reached its current quality
through three discrete post-processing improvements, not through
model architecture changes. The U-Net checkpoint is the same in all
three panels.

Three panels, all RGB at fixed [0, 0.3] reflectance stretch, all on
Brentmoor 18-Oct-2022:

- **Panel A — v0.5.x:** mosaic-merged outputs without blending;
  visible tile boundaries with sharp edges between 256-pixel patches.
- **Panel B — v0.6.0:** cosine-blended patch mosaic + v1 single-
  scene affine calibration; tile boundaries removed, colour balance
  uneven (calibration scene was Hankley only).
- **Panel C — v0.7.0:** cosine-blended patch mosaic + v3 multi-
  scene Huber affine calibration; uniform colour balance across the
  AOI.

**Pairs with:** dissertation methodology section on the post-
processing pipeline (v0.5 → v0.6 → v0.7) + the operational
deployment workflow.

## Figure 5 — `figure_5_variance_retention_attempts.png`

**Argument:** four interventions (Phase B v2 variance-aware loss,
Phase B v3 band-weighted variance, multi-temporal v1, post-hoc
calibration) all targeted the same variance-retention bound but
fail in **three distinct ways** at the same OOD scene. The figure
makes the failure-mode diversity visible: variance collapse looks
different from variance overshooting, and a model trained with
variance-loss generalises worse than baseline at the most extreme
OOD point.

Five panels, all Cavenham Heath 26-Jun-2024, identical RGB stretch
[0, 0.3], identical band mapping, identical projection extent:

- **Panel A — truth Sentinel-2:** sharp landscape with field
  boundaries clearly visible. Reference reflectance.
- **Panel B — baseline + v3 calibration:** smoothed, low-contrast.
  Driver-band variance retention **41 %** on Cavenham. Variance-
  collapsed but credible reflectance distribution.
- **Panel C — Phase B v2** (variance-aware loss): similar to B with
  isolated variance-overshoot patches in the lower-mid region.
  Driver var retention **50 %**.
- **Panel D — Phase B v3** (band-weighted variance): saturates white
  across most of the AOI. The raw model output on this OOD scene
  lands in the 0.4-0.55 reflectance range, well above the [0, 0.3]
  stretch ceiling. The v3 affine calibration was fit on baseline
  outputs and does not transfer to B v3's distribution. Driver var
  retention **14 %** (heavy collapse around an off-distribution mean).
- **Panel E — Multi-temporal v1:** variance overshooting at
  driver-band retention **81 %** (closest to truth on this metric)
  but with extreme pixel outliers visible as bright magenta/blue
  blobs throughout. The linear-output multi-temporal model produces
  values that overshoot the realistic reflectance range; v3
  calibration cannot fix the overshoot.

**Three failure modes from four interventions:**

1. **Variance collapse** (baseline + v3, Phase B v2): smooth,
   low-contrast, plausible reflectance distribution but driver-band
   variance falls 50-60 percentage points below truth.
2. **Off-distribution mean** (Phase B v3): the band-weighted
   variance loss pushes the model toward a different reflectance
   regime that doesn't match truth's; on the worst OOD site this
   is severe enough to saturate the visualisation.
3. **Variance overshoot with outliers** (Multi-temporal v1): the
   linear-output model produces extreme pixel values that survive
   calibration; std-ratio looks good (81 %) but visual quality is
   noisy.

**Pairs with:** dissertation experimental-sequence discussion of
the four documented negative results, particularly the section
arguing that variance retention is bounded by dataset scale rather
than tractable through architecture or loss-shaping at this scale.

## Provenance + reproducibility

All scripts to regenerate the figures live alongside the figures in
the project tree under `figures_work/fig{1..5}/render_figN.py` plus
the GEE Sentinel-2 fetch script for Figure 1 (`fig1/fetch_s2_truth.py`)
and the local multi-temporal inference script for Figure 5
(`fig5/mt_inference_cavenham.py`).

Real Sentinel-2 imagery for Figures 1 and 2 was harvested from the
COPERNICUS/S2_SR_HARMONIZED collection via Earth Engine on 2026-05-06,
mosaicked across the T30UXB / T30UXC tile boundary where the AOI
crosses it. AOI bounds in `fig1/brentmoor_footprint.json` match the
existing prediction grid byte-for-byte.

The Phase B v3 prediction on Cavenham used in Figure 5 was generated
by Vertex job `603586116477517824` on 2026-05-06 (T4, ~5 min). The
multi-temporal v1 prediction on Cavenham used a small one-AOI MT
harvest (4 patches at `multitemporal_v1_cavenham_inference/` on GCS)
followed by local TF inference reusing the project's `build_unet`
architecture and v3 calibration JSON.

SWT perimeter geometry comes from
`s1s2-translator/inputs/SWT_MappedFires_20220911.kml` (Surrey Wildlife
Trust field-mapped perimeters, 11-Sep-2022 export).

Cost: GEE Sentinel-2 fetches < £0.01; Phase B v3 Vertex inference
~£0.02; MT harvest + local inference < £0.01 (no Vertex compute
billed for MT). Total figure-set cost ~£0.03.
