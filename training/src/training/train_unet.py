"""Vertex AI Custom Training entrypoint for the Phase 2 U-Net.

Mirrors the notebook's Cells 4, 6, 8 in a single script. Reads
configuration from environment variables (same names as the notebook's
Cell 2 supports), so a Vertex Custom Training job can be configured by
setting env vars on the worker pool spec. This is the same code path
the notebook uses; only the orchestration changes.

Smoke-test mode: set ``GEE_S1S2_SMOKE_TEST_N_PATCHES=32`` (or any small
int) to slice the manifest down to the first N train patches. Used to
exercise the data pipeline + one full epoch end-to-end without paying
for a full run.

Run as a module so the relative imports resolve::

    python -m training.train_unet
"""
from __future__ import annotations

import logging
import os
import sys

import tensorflow as tf

from .data import (
    build_dataset, load_manifest, load_or_compute_s1_stats, split_uris,
)
from .model import build_unet
from .trainer import train as train_model

LOG = logging.getLogger("train_unet")


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"true", "1", "yes", "y"}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_id = _env("GEE_S1S2_PROJECT_ID")
    gcs_bucket = _env("GEE_S1S2_BUCKET")
    gcs_prefix = _env("GEE_S1S2_PREFIX", "gee_s1s2_translator/operational_v1")
    training_run_name = _env("GEE_S1S2_TRAINING_RUN_NAME", "v2_equivalent_initial")
    if not project_id or not gcs_bucket:
        raise RuntimeError(
            "GEE_S1S2_PROJECT_ID and GEE_S1S2_BUCKET must both be set."
        )

    batch_size = _env_int("GEE_S1S2_BATCH_SIZE", 8)
    max_epochs = _env_int("GEE_S1S2_MAX_EPOCHS", 80)
    learning_rate = _env_float("GEE_S1S2_LEARNING_RATE", 1e-4)
    early_stopping_patience = _env_int("GEE_S1S2_EARLY_STOPPING_PATIENCE", 15)
    random_seed = _env_int("GEE_S1S2_RANDOM_SEED", 42)
    s1_lee_already_applied = _env_bool(
        "GEE_S1S2_S1_LEE_ALREADY_APPLIED_AT_HARVEST", True,
    )
    smoke_n = os.environ.get("GEE_S1S2_SMOKE_TEST_N_PATCHES")
    smoke_n = int(smoke_n) if smoke_n else None

    base_uri = f"gs://{gcs_bucket}/{gcs_prefix}"
    harvest_manifest_uri = f"{base_uri}/manifest.csv"
    model_output_prefix = f"{base_uri}/models/{training_run_name}/"
    s1_stats_uri = f"{base_uri}/s1_stats.json"

    LOG.info("project=%s bucket=%s prefix=%s run_name=%s",
             project_id, gcs_bucket, gcs_prefix, training_run_name)
    LOG.info("manifest=%s output_prefix=%s s1_stats=%s",
             harvest_manifest_uri, model_output_prefix, s1_stats_uri)
    LOG.info("hyperparams: batch_size=%d max_epochs=%d lr=%g patience=%d seed=%d",
             batch_size, max_epochs, learning_rate, early_stopping_patience,
             random_seed)
    if smoke_n:
        LOG.warning("SMOKE TEST MODE: slicing train manifest to first %d patches",
                    smoke_n)

    # GPU presence is informational only (Vertex always provisions one when
    # asked; leaving the log line so smoke-test output proves the driver is up).
    gpus = tf.config.list_physical_devices("GPU")
    LOG.info("GPUs visible: %d (%s)", len(gpus), [g.name for g in gpus])

    tf.keras.utils.set_random_seed(random_seed)

    entries = load_manifest(harvest_manifest_uri)
    train_uris = split_uris(entries, "train")
    val_uris = split_uris(entries, "val")
    if smoke_n:
        train_uris = train_uris[:smoke_n]
        val_uris = val_uris[: max(8, smoke_n // 4)]
    LOG.info("train shards=%d val shards=%d", len(train_uris), len(val_uris))

    stats = load_or_compute_s1_stats(
        train_uris, s1_stats_uri,
        n_patches=min(200, len(train_uris)),
    )
    LOG.info("S1 stats: VV mean=%+.3f std=%.3f | VH mean=%+.3f std=%.3f",
             stats.mean["VV"], stats.std["VV"],
             stats.mean["VH"], stats.std["VH"])

    apply_lee_in_pipeline = not s1_lee_already_applied
    train_ds = build_dataset(
        train_uris, stats=stats, batch_size=batch_size,
        shuffle=True, apply_lee=apply_lee_in_pipeline, seed=random_seed,
    )
    val_ds = build_dataset(
        val_uris, stats=stats, batch_size=batch_size,
        shuffle=False, apply_lee=apply_lee_in_pipeline,
    )

    unet = build_unet(input_shape=(256, 256, 2), out_channels=6, base_channels=32)
    LOG.info("U-Net parameter count: %d", unet.count_params())

    result = train_model(
        unet,
        train_ds=train_ds, val_ds=val_ds,
        checkpoint_uri=model_output_prefix + "unet.keras",
        sidecar_uri=model_output_prefix + "unet_sidecar.json",
        log_uri=model_output_prefix + "train_unet.csv",
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        extra_metadata={
            "run_name": training_run_name,
            "manifest_uri": harvest_manifest_uri,
            "n_train_shards": len(train_uris),
            "n_val_shards": len(val_uris),
            "base_channels": 32,
            "vertex_ai_run": True,
            "smoke_test_n_patches": smoke_n,
        },
    )
    LOG.info("Training done: best val RMSE=%.4f at epoch %d (%d epochs run).",
             result.best_val_rmse, result.best_epoch, result.epochs_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
