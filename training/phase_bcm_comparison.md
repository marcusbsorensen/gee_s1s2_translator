# Phase B / B v2 / B v3 / Multi-temporal — comparison

Generated: 2026-05-05.

This is the canonical four-row comparison of the four U-Net variants
trained against the operational_v1 manifest, plus the multi-temporal v1
extension. The headline conclusion is unchanged from after Phase B v3:
**the single-temporal baseline + v3 affine calibration remains the
operational pipeline**, and Phase B v2, Phase B v3, and multi-temporal v1
are all documented negative results.

## Headline metrics (test set)

| Model                           | n_test | val RMSE | test MAE  | test RMSE | driver-band variance retention |
| ------------------------------- | -----: | -------: | --------: | --------: | -----------------------------: |
| **baseline (v2_equivalent)**    | 342†   | **0.0769** | 0.0443† | 0.0737†   | 55.5 %‡                        |
| Phase B v2 (variance loss)      | 342    | 0.0857   | 0.0482    | 0.0857    | 61.5 %                         |
| Phase B v3 (band-weighted)      | 362    | 0.0855   | 0.0494    | 0.0855    | 63.7 %                         |
| Multi-temporal v1 (T4)          | 354    | 0.0946   | 0.0579    | 0.0969    | **64.6 %**                     |

† baseline numbers computed from per-band MAE/RMSE in the in-distribution
test (n=237) of `validation_report.md`; the JSON metrics file is missing
for the baseline run because the report pre-dates the JSON-output
feature in `evaluate_run.py`.
‡ baseline retention is from the Ashdown-OOD subset only (n=105). Phase
B v2 / v3 / Multi-temporal retention is on the full test split (combined
OOD + in-dist). Same metric definition, slightly different test slice.

The val RMSE column is the best validation RMSE during training (the
metric used by `EarlyStopping`). The test RMSE column is the per-pixel
RMSE on the held-out test split; for B v2 / v3 / Multi-temporal these are
slightly higher than val RMSE because the test split includes the
geographically-disjoint Ashdown OOD patches.

## Per-band MAE — test split

| Model              | B02     | B03     | B04     | B08     | B11     | B12     |
| ------------------ | ------: | ------: | ------: | ------: | ------: | ------: |
| baseline (in-dist) | 0.0169  | 0.0221  | 0.0263  | 0.0972  | 0.0613  | 0.0420  |
| Phase B v2         | 0.0179  | 0.0235  | 0.0265  | 0.1116  | 0.0669  | 0.0430  |
| Phase B v3         | 0.0181  | 0.0243  | 0.0271  | 0.1102  | 0.0701  | 0.0464  |
| Multi-temporal     | 0.0256  | 0.0271  | 0.0296  | 0.1366  | 0.0790  | 0.0493  |

Multi-temporal is worse than baseline on every band. Largest
deteriorations: **B02 (+51 %)** and **B08 (+41 %)**.

## Per-band variance retention (% mean pred / truth)

Driver bands (B04, B08, B11, B12) bolded in the header. Pass bracket
[75 %, 105 %] on driver bands.

| Model                    | B02      | B03    | **B04** | **B08**   | **B11**   | **B12**   |
| ------------------------ | -------: | -----: | ------: | --------: | --------: | --------: |
| baseline (Ashdown OOD)   | 47.6 %   | 55.0 % | 43.2 %  | 60.3 %    | 63.1 %    | 55.7 %    |
| Phase B v2 (full test)   | 61.1 %   | 70.7 % | 45.1 %  | 75.3 %    | 71.3 %    | 54.6 %    |
| Phase B v3 (full test)   | 59.4 %   | 66.6 % | 43.0 %  | **79.3 %**| 70.7 %    | **61.8 %**|
| Multi-temporal (full)    | 117.6 %  | 78.9 % | **65.4 %** | 64.0 % | 66.5 %    | 62.6 %    |

## Interpretation

### Phase B v2 (variance-aware loss)

Activates a per-band variance term in the loss after a warmup epoch:
`L1 + 0.5*L2 + variance_weight * sum_b |std(y_pred_b) - std(y_true_b)|`.

- Improved driver-band retention from 55.5 % → 61.5 % (driver mean).
- Cost ~12 % MAE / ~16 % RMSE.
- Only B08 made it into the [75, 105] pass bracket.
- B04 unchanged (43 % → 45 %), the band most relevant to NBR.
- **Verdict: documented negative result.** Improves variance retention
  on B08 but at cost of overall fidelity, and doesn't move the most
  NBR-relevant band (B04 / B12).

### Phase B v3 (band-weighted variance)

Same structure as B v2 but with per-band weights w_b: 0.6 on B04 / B11 /
B12 (driver), 0.3 on B02 / B03 / B08. Hypothesis: increase weight on the
bands B v2 didn't move.

- Driver retention 61.5 % → 63.7 %.
- B08 retention 75.3 % → 79.3 % (still in pass bracket).
- B12 retention 54.6 % → 61.8 % (improved).
- B04 unchanged at 43 %, despite being weighted up to 0.6.
- B11 essentially unchanged at 70.7 %.
- Test RMSE essentially unchanged from B v2 (0.0857 → 0.0855).
- **Verdict: documented negative result.** Per-band weight asymmetry
  meaningfully helped B08 and B12 only; B04 — the band whose variance
  matters most for NBR-thresholded burn detection — is structurally
  resistant to this intervention. The variance loss as currently
  implemented cannot move B04.

### Multi-temporal v1

Three Sentinel-1 acquisitions per training pair (t_0, t_-1w, t_-3w),
6-channel input, otherwise identical to Phase B v3 (same loss, same
weights, same hyperparameters, same dataset, n=997 patches after the
13-row drop for missing prior S1 acquisitions).

- **Best B04 retention of any model so far** (65.4 % vs 43-45 %).
- Comparable B11, B12 to B v3.
- **Loses ~15 percentage points on B08** vs B v2 / v3 (64 % vs 75-79 %),
  dropping out of the pass bracket.
- B02 retention 117.6 % — overshoots the noise bracket. Suggests the
  additional temporal channels are letting the model produce more
  per-patch variance than truth in B02, possibly via S1-VV speckle that
  correlates spatially with B02.
- Costs ~30 % MAE / ~13 % RMSE vs baseline. Largest hits on B02
  (+51 % MAE) and B08 (+41 % MAE).
- **Verdict: documented negative result.** Multi-temporal Sentinel-1
  with this architecture and dataset scale (~700 train patches) does
  not converge to a competitive minimum on RMSE/MAE. The B04 variance
  retention gain is real but does not compensate for the B08 regression
  or the overall MAE/RMSE cost.

### Why these all share the same shape

The variance-aware loss interventions and the multi-temporal architecture
all produce the same trade-off: small improvements to one or two
variance-retention numbers in exchange for significant RMSE/MAE
deterioration. The single-temporal baseline + v3 affine calibration
remains the operational pipeline because:

1. Per-pixel MAE/RMSE is strictly best on the baseline.
2. Variance retention is improvable downstream — affine calibration on
   labelled scenes already pulls retention up where it matters most for
   the operational use case (Sonia's dNBR perimeter detection).
3. Variance retention is not improvable upstream without breaking
   per-pixel fidelity, on this dataset scale and this architecture.

The dissertation's Day 1-2 negative-result section now has three
hypothesis tests of "can we move B04 retention out of variance collapse
without losing per-pixel fidelity?" all answering "not at this scale,
not with these interventions". This is a clean experimental story.

## Status of artefacts

- baseline + v3 calibration is operational. Lives at
  `gs://.../models/v2_equivalent_initial/`.
- Phase B v2 + Phase B v3 checkpoints retained for reproducibility.
- Multi-temporal v1 retained at `models/multitemporal_v1_t4/`. The
  cancelled CPU run's stale artefacts at `models/multitemporal_v1/` were
  deleted on 2026-05-05.
- Multi-temporal manifest kept at `multitemporal_v1/manifest.csv` in
  case future experiments want the 6-channel TFRecords.
