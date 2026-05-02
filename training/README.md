# `gee_s1s2_translator/training` — Phase 2 (U-Net training in Colab)

Re-runnable Colab notebook + supporting Python modules for training the
S1→S2 translator on the Phase 1 GCS-harvested dataset. Phase 2 is the
direct successor to Phase 1 (harvesting); Phase 3 wraps the trained
model in a Vertex AI Endpoint.

The notebook is parameterised so Sonia can re-run training on a fresh
manifest without modifying any code: edit only the parameter cell with
project / bucket / run-name, then Run All.

## Opening the notebook in Colab

1. **Open** `notebooks/02_unet_training.ipynb` in Colab.
   - From GitHub: `https://colab.research.google.com/github/<owner>/gee_s1s2_translator/blob/main/training/notebooks/02_unet_training.ipynb` (replace `<owner>` with the GitHub account that hosts this fork).
   - Or upload the local `.ipynb` directly.
2. **Set the runtime**: `Runtime → Change runtime type → Hardware accelerator → GPU`. The free-tier T4 is sufficient for this model and dataset; no Colab Pro needed for the v2-equivalent harvest size.
3. **Configure parameters** (Cell 2). Two equivalent approaches — pick one:
   - **(a) Edit Cell 2 directly.** Change `PROJECT_ID = "your-gcp-project-id-here"` and `GCS_BUCKET = "your-bucket-name-here"` to your real values. `TRAINING_RUN_NAME` is a literal you change per run.
   - **(b) Set environment variables** before running the notebook. Cell 2 reads from `os.environ` with placeholder defaults, so setting `GEE_S1S2_PROJECT_ID` and `GEE_S1S2_BUCKET` (and optionally `GEE_S1S2_PREFIX`) bypasses the need to edit the cell.
     - In Colab: insert a prior cell with `import os; os.environ["GEE_S1S2_BUCKET"] = "..."; os.environ["GEE_S1S2_PROJECT_ID"] = "..."`.
     - Locally: `export GEE_S1S2_BUCKET=...; export GEE_S1S2_PROJECT_ID=...` before `jupyter notebook`.
   - Cell 2 fails fast with a clear `RuntimeError` if the placeholders are still in place when you Run all, so you can't accidentally run with the wrong target.
   - Other Cell 2 parameters (`BATCH_SIZE`, `MAX_EPOCHS`, `LEARNING_RATE`, `EARLY_STOPPING_PATIENCE`, `RANDOM_SEED`, `S1_LEE_ALREADY_APPLIED_AT_HARVEST`) have v2-matching defaults; leave them alone unless you have a reason.
4. **Run all cells** (`Runtime → Run all`). Colab will prompt for Google authentication when Cell 3 runs.
5. **Wait** ~30–60 minutes (T4). The validation report is written to GCS in Cell 9 and a side-by-side vs v2 is printed in Cell 10.

## Expected runtime and cost

| Item | Free Colab tier (T4) |
| --- | --- |
| Linear baseline training | ~3–5 min |
| U-Net training (≤ 80 epochs, early-stop usually fires by epoch 30–50) | ~25–50 min |
| Validation pass on full test set | ~2–5 min |
| **Total** | **30–60 min** |
| Cost | **£0** (Colab free tier) |

If GPU allocation fails repeatedly during testing (rare but possible on
free tier), Colab Pro at £9.99/month is a fallback; the notebook works
unchanged. For a one-off operational training run, the free tier is
fine.

## What the notebook produces

All artifacts are written to `MODEL_OUTPUT_PREFIX` in GCS:

```
gs://<bucket>/gee_s1s2_translator/operational_v1/models/<run-name>/
├── unet.keras                   # trained U-Net weights
├── unet_sidecar.json            # best epoch, val RMSE, hyperparameters, timestamps
├── train_unet.csv               # per-epoch loss/metric curve
├── linear_baseline.keras        # baseline weights
├── linear_baseline_sidecar.json
├── train_linear_baseline.csv
├── validation_report.md         # human-readable validation report
├── validation_report.json       # structured metrics for programmatic comparison
└── parameters.json              # exact Cell 2 contents for reproducibility
```

`s1_stats.json` lives one level up at the operational-v1 prefix so it
can be shared across training runs.

## Architecture (matches v2 PyTorch)

| Component | Detail |
| --- | --- |
| Input | 256×256 patches with 2 S1 channels (VV, VH); z-score normalised |
| Output | 256×256 patches with 6 S2 reflectance bands (B02, B03, B04, B08, B11, B12), sigmoid-clamped to [0, 1] |
| Encoder / decoder | 4 levels each; double conv (Conv → BN → ReLU → Conv → BN → ReLU) |
| Downsampling | 2×2 max pool |
| Upsampling | bilinear (matching v2; not transposed conv) |
| Bottleneck | 512 channels at level 4 (with `base_channels=32`) |
| Loss | combined L1 + 0.5 × L2 |
| Optimiser | Adam @ 1e-4 |
| Batch size | 8 |
| Early stopping | patience 15 on val RMSE |
| Linear baseline | per-pixel 1×1 conv from S1 to S2; sigmoid output |

## Tests

`tests/` contains unit tests that run **without GPU and without GCS**
using small synthetic fixtures:

* `test_data.py` — manifest loading, split routing, TFRecord parsing
  shape, S1 z-score normalisation, NaN handling, wildcard URI cleanup
* `test_model.py` — U-Net forward-pass shape, sigmoid output range,
  skip-connection presence, parameter count in expected order; linear
  baseline shape + parameter count + translation invariance

Run them with `pytest gee_s1s2_translator/training/tests/`. Training
loop tests are intentionally out of scope — training is end-to-end
validation through the notebook.

## What this pipeline does *not* do

* **Re-apply Lee speckle filter to S1.** The Phase 1 harvest applies
  Lee 5×5 server-side via the GEE calibration recipe; applying it again
  in this pipeline would over-smooth. There is an opt-in flag in
  Cell 2 (`S1_LEE_ALREADY_APPLIED_AT_HARVEST`) for the (unusual) case
  where someone re-harvests with `speckle_filter.enabled: false` and
  wants to apply Lee at training time instead.
* **Compute derived indices (NDVI / NBR / NDWI) as model outputs.**
  v2 explicitly dropped these from the model's prediction targets in
  the methodology refinement; we compute them post-hoc from the six
  predicted reflectance bands. The notebook does not currently render
  the post-hoc indices — `validation_report.md` is reflectance-only.

## Methodology divergences vs v2

Three differences are documented in
`docs/methodology_divergences.md`. All are operationally inert; you
can train on the GEE-harvested dataset and compare back to v2 with
appropriate footnotes:

1. **S1 calibration −1.5 dB offset** vs MPC RTC (Vollrath/Mullissa
   volumetric model vs ESA range-Doppler).
2. **Ashdown held-out test count: 105 vs v2's 76** (random sampling +
   slightly different S2 scene availability). Cell 9 computes Ashdown
   metrics on the full 105 patches; Cell 10's comparison footnote
   flags the divergence.
3. **T30UWB partial-coverage edge case** at Beaulieu Heath. Zero
   recoverable patches; defensive None-handling at harvest is correct.

## Where to look next

* **Validation report**: GCS at `MODEL_OUTPUT_PREFIX + 'validation_report.md'`.
* **v2 reference numbers**: `training/v2_reference_results.json`
  (footnotes spell out every divergence inline).
* **Methodology divergences**: `docs/methodology_divergences.md`.
* **Phase 3 (Vertex AI deployment)**: not yet started; Phase 3 prompt
  follows after Phase 2 review-gate clears.
