# Experimental summary — `gee_s1s2_translator`

A one-page methodological backstory for someone Sonia is showing the
work to. This sits in front of the validation reports and explains
how the operational training pipeline relates to the v1 / v1.5 / v2
PyTorch references and to the Phase B / B v2 / C variants tried in
this repo.

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
we ran two experimental variants on the same dataset to test
whether the variance-collapse problem identified in validation
could be closed by either loss-shaping or architectural change:

- **Phase B (loss shaping):** linear output activation (clipped to
  `[0, 1]` at inference), L1 + 0.5 × L2 augmented with a per-band
  standard-deviation matching term, cosine LR decay over 120 epochs,
  patience 25.
- **Phase B v2 (corrected schedule):** same as B, but with the
  variance term activated at epoch 15 instead of 30, and patience 40
  instead of 25 — to fix the bug where Phase B's best checkpoint
  landed *before* the variance term ever influenced the gradient.
- **Phase C (architecture):** U-Net learns a delta on top of a 1×1
  linear projection of the Sentinel-1 input, with sigmoid applied to
  the sum. Aim: give the network a useful prior so it only has to
  learn the correction.

## What worked

- **The baseline is operationally usable for batch inference.** Best
  validation RMSE 0.0769 reflectance; Brentmoor + Poors Allotment
  post-fire predictions land in plausible NBR space against the SWT
  field-mapped perimeters.
- **The variance hypothesis is partially confirmed.** Phase B v2
  pushed driver-band B08 (NIR) variance retention from ~73 % to
  75.3 % — into the operational pass bracket — at the cost of
  ~3 % worse overall RMSE. The signal is real but small at the
  current dataset scale.
- **Vertex AI as the operational platform was the right choice.**
  Reproducing v2's training run on T4 takes ~15 minutes for ~£0.30,
  fits within free-tier compute budgets, and pairs cleanly with the
  GEE harvest that produces the training data. End-to-end NGO
  deployment is single-day work for a competent operator (see
  `OPERATIONAL_DEPLOYMENT.md`).

## What did not work

- **Phase B (uncorrected) didn't test the hypothesis it set out to.**
  The variance term warmed up at epoch 30 but the best validation
  checkpoint was at epoch 25. Documented as a planning bug; v2
  corrected it.
- **Phase C did not help.** Driver-band variance retention dropped
  to 49.1 % (worst of the four runs); test RMSE 0.0852. The 1×1
  baseline projection appears to give the U-Net no useful prior at
  this dataset scale and may amplify high-frequency noise.
- **Phase B v2 only shifted one driver band into the pass bracket.**
  B04, B11, B12 stayed below 75 %. Either the variance weight needs
  to be band-specific (heavier on B04 / B12) or the dataset scale
  itself is the binding constraint.

## Operational implication

The operational pipeline ships the baseline checkpoint
(`v2_equivalent_initial`) as the production model: lowest RMSE, full
field-validation chain, and a documented recipe for empirically
calibrating dNBR thresholds against real Sentinel-2 to compensate
for the variance-collapse on the smoothed bands. Phase B v2 is the
recommended foundation for a future re-train with a larger or
better-distributed training set; Phase C is parked.

## Residual limitations

- **Dataset scale.** 943 patches across 13 southern-England training
  AOIs is small by deep-learning standards. Per-band variance
  retention below the operational floor on three of four driver bands
  is consistent with regression-to-the-mean from a small training
  set; loss shaping can't close that on its own.
- **Single regional biome.** The model was trained and validated on
  UK lowland heath at 50–51 ° N. Application to Scottish blanket bog,
  Mediterranean garrigue, or boreal heath needs re-training (Workflow
  4 in `OPERATIONAL_DEPLOYMENT.md`).
- **Variance is a summary statistic.** Where in the scene the model
  smooths matters operationally; the variance retention number
  doesn't capture that. A spatial-residual diagnostic on the
  out-of-distribution worked example (`WORKED_EXAMPLE_CAVENHAM.md`)
  is the next obvious investigation.
- **No multi-temporal input.** The model sees a single S1 scene at a
  time. Multi-date stacks (the v3 line of work) would likely close
  more of the variance gap than per-loss tweaks at the same dataset
  size — but require harvest-pipeline changes outside this repo's
  current scope.
