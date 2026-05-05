"""Vertex AI entrypoint: evaluate a finished training run and write a
validation report alongside the checkpoint.

Reproduces the metrics in the Phase 2 Colab notebook but runs as a
standalone Vertex Custom Job, so each Phase B / Phase C run gets its
own ``validation_report.md`` + ``validation_metrics.json`` without
needing an interactive notebook.

Inputs (env vars):
    GEE_S1S2_PROJECT_ID
    GEE_S1S2_BUCKET
    GEE_S1S2_PREFIX (default ``gee_s1s2_translator/operational_v1``)
    GEE_S1S2_TRAINING_RUN_NAME (the run to evaluate)
    GEE_S1S2_PHASE ("", "baseline", "b", "c") — chooses whether to
        clip predictions and which architecture to rebuild
    GEE_S1S2_BATCH_SIZE (default 8)

Outputs (under ``gs://<bucket>/<prefix>/models/<run>/``):
    validation_report.md          — human-readable per-band table
    validation_metrics.json       — structured copy of the same numbers
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from .data import (
    build_dataset, load_manifest, load_or_compute_s1_stats, split_uris,
)
from .metrics import (
    DRIVER_BANDS, S2_BAND_ORDER, VARIANCE_BRACKET_HIGH, VARIANCE_BRACKET_LOW,
    driver_band_mean_retention_pct, patch_specific_variance_retention,
    per_band_mae_rmse,
)
from .model import build_residual_unet, build_unet

LOG = logging.getLogger("evaluate_run")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _build_for_phase(phase: str) -> tf.keras.Model:
    if phase == "c":
        return build_residual_unet(
            input_shape=(256, 256, 2), out_channels=6, base_channels=32,
        )
    if phase == "b":
        return build_unet(
            input_shape=(256, 256, 2), out_channels=6, base_channels=32,
            output_activation="linear",
        )
    if phase == "mt":
        # Multi-temporal: 6-channel S1 input, linear output (Phase B v2 family).
        return build_unet(
            input_shape=(256, 256, 6), out_channels=6, base_channels=32,
            output_activation="linear",
        )
    return build_unet(input_shape=(256, 256, 2), out_channels=6, base_channels=32)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_id = _env("GEE_S1S2_PROJECT_ID")
    bucket = _env("GEE_S1S2_BUCKET")
    prefix = _env("GEE_S1S2_PREFIX", "gee_s1s2_translator/operational_v1")
    run_name = _env("GEE_S1S2_TRAINING_RUN_NAME")
    phase = (_env("GEE_S1S2_PHASE", "") or "").strip().lower()
    batch_size = int(_env("GEE_S1S2_BATCH_SIZE", "8"))
    if not project_id or not bucket or not run_name:
        raise RuntimeError(
            "GEE_S1S2_PROJECT_ID / GEE_S1S2_BUCKET / GEE_S1S2_TRAINING_RUN_NAME "
            "must all be set."
        )

    base = f"gs://{bucket}/{prefix}"
    # Multi-temporal eval reads patches from a separate prefix (matches
    # what train_unet does for phase=mt). Models are still under the
    # operational_v1/models/<run> tree so eval reports land beside it.
    if phase == "mt":
        mt_prefix = os.environ.get(
            "GEE_S1S2_MULTITEMPORAL_PREFIX",
            "gee_s1s2_translator/multitemporal_v1",
        )
        manifest_uri = f"gs://{bucket}/{mt_prefix}/manifest.csv"
    else:
        manifest_uri = f"{base}/manifest.csv"
    s1_stats_uri = f"{base}/s1_stats.json"
    model_dir = f"{base}/models/{run_name}/"
    keras_uri = model_dir + "unet.keras"
    report_uri = model_dir + "validation_report.md"
    metrics_uri = model_dir + "validation_metrics.json"

    LOG.info("evaluating run=%s phase=%s", run_name, phase or "baseline")
    LOG.info("checkpoint=%s manifest=%s", keras_uri, manifest_uri)

    # --- Build architecture + load weights ---
    model = _build_for_phase(phase)
    LOG.info("Built %s (params=%d). Loading weights from GCS...",
             model.name, model.count_params())
    # Same local-temp + tf.io.gfile.copy pattern as deploy_endpoint, in case
    # the Keras 3 zip-on-GCS read path is broken on this image.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        local_path = tmp.name
    try:
        tf.io.gfile.copy(keras_uri, local_path, overwrite=True)
        model.load_weights(local_path)
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass

    # --- Test split dataset ---
    entries = load_manifest(manifest_uri)
    test_uris = split_uris(entries, "test")
    LOG.info("test shards: %d", len(test_uris))
    train_uris = split_uris(entries, "train")
    stats = load_or_compute_s1_stats(
        train_uris, s1_stats_uri, n_patches=min(200, len(train_uris)),
    )
    LOG.info("S1 stats: VV mean=%+.3f std=%.3f | VH mean=%+.3f std=%.3f",
             stats.mean["VV"], stats.std["VV"],
             stats.mean["VH"], stats.std["VH"])

    test_ds = build_dataset(
        test_uris, stats=stats, batch_size=batch_size,
        shuffle=False, apply_lee=False,
        multitemporal=(phase == "mt"),
    )

    # --- Predict + accumulate (N, H, W, C) ---
    LOG.info("Running predictions over %d test shards...", len(test_uris))
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    for s1_batch, s2_batch in test_ds:
        pred = model.predict(s1_batch, verbose=0)
        y_pred_chunks.append(pred)
        y_true_chunks.append(s2_batch.numpy())
    y_true = np.concatenate(y_true_chunks, axis=0)
    y_pred = np.concatenate(y_pred_chunks, axis=0)
    LOG.info("Stacked predictions: y_true=%s y_pred=%s",
             y_true.shape, y_pred.shape)

    # Phase B has linear output — clip to [0, 1] at evaluation time so
    # metrics are computed in the operational reflectance range. The
    # baseline / Phase C use sigmoid so are already in [0, 1] modulo
    # numeric noise; clip is a no-op for them.
    if phase in ("b", "mt"):
        y_pred = np.clip(y_pred, 0.0, 1.0)
        LOG.info("Phase %s: clipped predictions to [0, 1].", phase.upper())

    # --- Compute metrics ---
    per_band = per_band_mae_rmse(y_true, y_pred)
    var_rows = patch_specific_variance_retention(y_true, y_pred)
    overall_rmse = float(np.sqrt(np.mean(np.square(y_pred - y_true))))
    overall_mae = float(np.mean(np.abs(y_pred - y_true)))
    driver_mean_pct = driver_band_mean_retention_pct(var_rows)

    # --- Write structured JSON ---
    metrics_record = {
        "run_name": run_name,
        "phase": phase or "baseline",
        "n_test_patches": int(y_true.shape[0]),
        "overall_test_mae": overall_mae,
        "overall_test_rmse": overall_rmse,
        "per_band": [{"band": m.band, "mae": m.mae, "rmse": m.rmse}
                     for m in per_band],
        "variance_retention": [
            {"band": v.band, "is_driver": v.is_driver,
             "mean_pred_over_truth_pct": v.mean_pred_over_truth_pct,
             "median_pred_over_truth_pct": v.median_pred_over_truth_pct,
             "pass_75_105_bracket": v.pass_75_105_bracket}
            for v in var_rows
        ],
        "driver_band_mean_retention_pct": driver_mean_pct,
        "variance_bracket_low_pct": VARIANCE_BRACKET_LOW * 100,
        "variance_bracket_high_pct": VARIANCE_BRACKET_HIGH * 100,
        "driver_bands": DRIVER_BANDS,
        "band_order": S2_BAND_ORDER,
    }
    with tf.io.gfile.GFile(metrics_uri, "w") as f:
        json.dump(metrics_record, f, indent=2)
    LOG.info("Wrote metrics: %s", metrics_uri)

    # --- Write Markdown report ---
    lines: list[str] = []
    lines.append(f"# Validation report — {run_name}")
    lines.append("")
    lines.append(f"Phase: **{phase or 'baseline'}**  ")
    lines.append(f"Test patches: {y_true.shape[0]}  ")
    lines.append(f"Overall test MAE: **{overall_mae:.4f}**  ")
    lines.append(f"Overall test RMSE: **{overall_rmse:.4f}**")
    lines.append("")
    lines.append("## Per-band MAE / RMSE — in-distribution test set")
    lines.append("")
    lines.append("| Band | MAE | RMSE |")
    lines.append("| --- | ---: | ---: |")
    for m in per_band:
        lines.append(f"| {m.band} | {m.mae:.4f} | {m.rmse:.4f} |")
    lines.append("")
    lines.append("## Variance-collapse diagnostic — patch-specific truth std")
    lines.append("")
    lines.append(
        f"Pass bracket: [{VARIANCE_BRACKET_LOW * 100:.0f} %, "
        f"{VARIANCE_BRACKET_HIGH * 100:.0f} %] on driver bands "
        f"({', '.join(DRIVER_BANDS)}). Below = collapse, above = noise."
    )
    lines.append("")
    lines.append("| Band | Driver | Mean pred/truth (%) | Median (%) | Pass |")
    lines.append("| --- | :---: | ---: | ---: | :---: |")
    for v in var_rows:
        drv = "✓" if v.is_driver else ""
        passed = ("OK" if v.pass_75_105_bracket else "FAIL") if v.is_driver else "—"
        lines.append(
            f"| {v.band} | {drv} | {v.mean_pred_over_truth_pct:.1f} "
            f"| {v.median_pred_over_truth_pct:.1f} | {passed} |"
        )
    lines.append("")
    lines.append(f"**Driver-band mean retention: {driver_mean_pct:.1f} %**")
    with tf.io.gfile.GFile(report_uri, "w") as f:
        f.write("\n".join(lines) + "\n")
    LOG.info("Wrote report:  %s", report_uri)

    # Print headline numbers to job logs for quick scanning.
    LOG.info(
        "Headline: test_rmse=%.4f, driver-band variance retention=%.1f %%",
        overall_rmse, driver_mean_pct,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
