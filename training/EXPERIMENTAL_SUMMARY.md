# Experimental summary — `gee_s1s2_translator`

A one-page methodological backstory for someone Sonia is showing the
work to. This sits in front of the validation reports and explains
how the operational training pipeline relates to the v1 / v1.5 / v2
PyTorch references and to the four experimental variants tried in
this repo (Phase B v2, Phase B v3, multi-temporal v1,
scene-conditional MLP).

## What we tried

The repo's operational pipeline (`operational_v1`) is a deliberate
GEE port of the v2 PyTorch + Microsoft Planetary Computer
implementation. We kept the U-Net architecture, the L1 + 0.5 × L2
loss, the 50 / 30 / 30 split policy, and the Mullissa-style
Sentinel-1 calibration. We changed only the things that had to
change to run on Google Earth Engine + Vertex AI: the data harvest,
the calibration recipe (Vollrath/Mullissa volumetric γ⁰ in GEE
versus ESA RTC in MPC), and the export plumbing (TFRecord on GCS
versus PyTorch tensors on local disk). Three documented divergences
from v2 are listed in `docs/methodology_divergences.md`.

After landing the v2-equivalent baseline (`v2_equivalent_initial`),
we ran four experimental variants on the same dataset to test
whether the variance-collapse problem identified in validation
could be closed by loss-shaping, architectural change, more
temporal context, or post-hoc calibration:

- **Phase B v2 (variance-aware loss):** L1 + 0.5 × L2 augmented with
  a per-band standard-deviation matching term, weight 0.3, warmup at
  epoch 15, cosine LR decay over 120 epochs, patience 40. Linear
  output activation (clipped to `[0, 1]` at inference).
- **Phase B v3 (band-weighted variance):** same as B v2 but with
  per-band weights — 0.6 on B04 / B11 / B12 (driver), 0.3 on
  B02 / B03 / B08. Hypothesis: heavier penalty on the bands B v2
  did not move (B04, B11, B12).
- **Multi-temporal v1:** 6-channel S1 input from three acquisitions
  per training pair (t_0, t_-1w, t_-3w), otherwise identical to
  Phase B v3 hyperparameters and loss. Tests whether more temporal
  context closes the variance gap at fixed architecture.
- **Scene-conditional MLP calibration (v4 candidate):** small sklearn
  MLP, hidden (64, 32), conditioned on (predicted reflectance + month
  + lat/lon + region one-hot), output corrected reflectance.
  Validated leave-one-scene-out across 6 calibration scenes vs
  the v3 global affine (multi-scene Huber).

## What worked

- **The baseline is operationally usable for batch inference.** Best
  validation RMSE 0.0769 reflectance; Brentmoor + Poors Allotment
  post-fire predictions land in plausible NBR space against the SWT
  field-mapped perimeters. Three OOD-site validation (Ashdown,
  Cavenham, Berwyn) shows the model transfers across UK heath
  regimes for perimeter delineation.
- **Three-OOD-site transfer demonstrated.** Berwyn SSSI (north Wales,
  upland heath, ~250 km OOD) reaches the operational variance
  bracket on B11 (79 %) and B12 (79 %) — the first OOD site to do
  so on either driver band — with overall MAE 0.043 (better than
  Cavenham's 0.065). The transfer pattern suggests per-pixel MAE
  tracks landscape spatial homogeneity rather than geographic
  distance from training. See `WORKED_EXAMPLE_BERWYN.md`.
- **Variance retention can be improved at single-band scale by loss
  shaping.** Phase B v2 pushed B08 driver-band variance retention
  from 60 % to 75.3 % (into the operational pass bracket). Phase B
  v3 extended this to B12 (61.8 %, marginal improvement). Both at
  measurable cost to overall MAE/RMSE.
- **Vertex AI as the operational platform was the right choice.**
  Reproducing v2's training run on T4 takes ~15 minutes for ~£0.30,
  fits within free-tier compute budgets, and pairs cleanly with the
  GEE harvest that produces the training data. End-to-end NGO
  deployment is single-day work for a competent operator (see
  `OPERATIONAL_DEPLOYMENT.md`).

## What did not work — four documented negative results

- **Phase B v2 (variance-aware loss).** Improved driver-band mean
  variance retention from 55.5 % → 61.5 % at the cost of ~12 % MAE
  / ~16 % RMSE. Only B08 made the [75-105] pass bracket. B04 — the
  most NBR-relevant band — stayed at 45 %. Documented negative
  result: variance loss helps B08 only, at MAE cost.
- **Phase B v3 (band-weighted variance).** Driver retention 61.5 % →
  63.7 %; B12 improved from 54.6 % to 61.8 %, B04 unchanged at 43 %
  despite 0.6 weight. Test RMSE essentially unchanged from B v2
  (0.0857 → 0.0855). Per-band weight asymmetry helped B08 + B12
  only; B04 is structurally resistant to this intervention.
  Documented negative result: B04 variance does not move under
  loss-shaping at this dataset scale.
- **Multi-temporal v1.** Best val RMSE 0.0946 vs baseline 0.0769
  (+13 percentage points worse). Best B04 retention of any model
  (65 %) — the first improvement on B04 — but lost ~15 pts on B08
  vs B v2/v3 and cost ~30 % MAE / ~13 % RMSE vs baseline. The B04
  gain does not compensate for the B08 regression or the overall
  fidelity cost. Documented negative result: multi-temporal does
  not converge to a competitive minimum at this dataset scale
  (~700 train patches over 13 AOIs).
- **Scene-conditional MLP calibration.** LOSO CV across 6
  calibration scenes vs v3 global affine: MLP loses on every band's
  mean MAE (B02 +24 %, B03 +18 %, B04 +32 %, B08 +35 %, B11 +48 %,
  B12 +50 % MAE worse than v3). The MLP wins on std ratio (5-6/6
  bands per fold) but overshoots median ratio, often pushing it
  from v3's ~110-130 % up to 150-200 %. The combination wrecks the
  MAE. With only 4 unique AOIs in the calibration set (12 free
  parameters in v3 vs ~3000 in the MLP), the MLP overfits scene-
  specific spectral idiosyncrasies that don't transfer.
  Documented negative result: scene-conditional calibration is
  dataset-scale-bounded; the v3 global affine is the right
  calibrator at the current calibration-set size.

## Operational implication

The operational pipeline ships the **baseline checkpoint
(`v2_equivalent_initial`) plus v3 affine calibration plus
cosine-blended patch mosaic (v0.7.0)** as the production
configuration: lowest RMSE, full field-validation chain, and a
documented recipe for empirically re-calibrating dNBR thresholds
in new regional regimes (Cavenham, Berwyn worked examples). All
four experimental variants are documented negative results
preserved in the model artifacts directory and in
`training/phase_bcm_comparison.md`; none are recommended for
operational use at the current dataset scale.

## Residual limitations and concrete future-work directions

The four-day experimental sequence articulates a complete story:
**variance retention is bounded by dataset scale rather than
tractable through loss shaping, architectural extension, or
post-hoc calibration at this scale.** Three of four interventions
were targeted improvements that produced documented negative
results; the fourth (scene-conditional calibration) hit the same
ceiling from the post-hoc side. Generalisation has been
demonstrated across three OOD regions with band-specific
performance that varies by landscape spatial homogeneity.

Concrete future-work directions:

- **More training data.** 943 patches across 13 southern-England
  training AOIs is the binding constraint on every variance-
  retention intervention tested. Adding 20-40 sites across UK
  upland and lowland regimes (Pennines, Welsh uplands, Scottish
  bog, Norfolk Broads, additional Brecks) is the most direct
  investment for breaking the variance-retention floor.
- **Multi-temporal at larger scale.** The multi-temporal v1
  negative result is dataset-scale-bounded, not architecture-
  bounded. Re-running with 3-5× more training pairs (achievable
  via a multi-day GEE harvest) would test whether the additional
  temporal channels help at sufficient scale.
- **Scene-conditional calibration with more scene diversity.** The
  v4 LOSO CV used 4 unique AOIs in the calibration set — too few
  for the MLP to learn transferable scene-conditional structure.
  10-15 unique calibration scenes spanning regional regimes
  (Brecks, Welsh upland, Scottish bog, southern lowland) would
  give the MLP enough feature-space coverage to potentially beat
  the global affine.
- **Diffusion or transformer architectures.** The U-Net's
  variance-collapse pattern is a known signature of regression-to-
  the-mean under L1/L2 loss. Diffusion models or vision
  transformers with explicit per-pixel uncertainty modelling are
  the architectural directions most likely to break the variance
  floor at any dataset scale.

## Other limitations

- **Single regional biome at training time.** The model was trained
  and validated on UK lowland heath at 50–51 ° N. The Berwyn worked
  example shows transfer to upland heath works for perimeter
  delineation; severity classification still needs local re-training
  or recalibration for any new regime.
- **Variance is a summary statistic.** Where in the scene the model
  smooths matters operationally; the variance retention number
  doesn't capture that. A spatial-residual diagnostic on the
  Cavenham + Berwyn worked examples is the next obvious
  investigation.
- **dNBR thresholds are region-specific.** Three OOD sites all show
  different median-ratio biases on B04 and B08, so dNBR thresholds
  fitted on southern-England training data do not transfer cleanly
  to either Brecks or upland-heath regimes. Operational deployment
  in a new region requires empirically fitting the threshold
  against a known-good fire perimeter from that region.
