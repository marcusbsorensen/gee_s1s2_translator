"""Vertex AI Custom Training entrypoint for the Phase 2 U-Net.

Mirrors the notebook's Cells 4, 6, 8 in a single script. Reads
configuration from environment variables (same names as the notebook's
Cell 2 supports), so a Vertex Custom Training job can be configured by
setting env vars on the worker pool spec. This is the same code path
the notebook uses; only the orchestration changes.

Phase dispatch — the entrypoint supports three model variants, selected
by ``GEE_S1S2_PHASE``:

* ``""`` or ``"baseline"`` (default) — the v2-equivalent run (sigmoid
  output, L1+0.5*L2 loss, constant LR 1e-4).
* ``"b"`` — Phase B: linear output (clipped at inference), L1+0.5*L2 +
  variance-matching term (warmup at epoch 30, weight 0.3), cosine LR
  decay 1e-4 -> 1e-5 over the full 120-epoch budget, patience 25.
* ``"c"`` — Phase C: residual U-Net (linear S1->reflectance baseline +
  U-Net delta, sigmoid'd at the sum). All other hyperparameters match
  the baseline.

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
from .losses import CombinedL1L2VarianceLoss, VarianceWeightWarmup
from .model import build_unet, build_residual_unet
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

    phase = (_env("GEE_S1S2_PHASE", "") or "").strip().lower()
    if phase not in ("", "baseline", "b", "c", "mt"):
        raise RuntimeError(
            f"Unrecognised GEE_S1S2_PHASE={phase!r}; "
            "must be one of '', 'baseline', 'b', 'c', 'mt'."
        )

    # Per-phase defaults override the legacy defaults; explicit env vars still
    # win, so the operator can dial in any individual value.
    if phase == "b":
        default_max_epochs = 120
        default_patience = 25
    elif phase == "mt":
        # Multi-temporal uses Phase B v2 hyperparameters per Day 2 spec.
        default_max_epochs = 120
        default_patience = 40
    else:
        default_max_epochs = 80
        default_patience = 15

    batch_size = _env_int("GEE_S1S2_BATCH_SIZE", 8)
    max_epochs = _env_int("GEE_S1S2_MAX_EPOCHS", default_max_epochs)
    learning_rate = _env_float("GEE_S1S2_LEARNING_RATE", 1e-4)
    early_stopping_patience = _env_int(
        "GEE_S1S2_EARLY_STOPPING_PATIENCE", default_patience,
    )
    random_seed = _env_int("GEE_S1S2_RANDOM_SEED", 42)
    s1_lee_already_applied = _env_bool(
        "GEE_S1S2_S1_LEE_ALREADY_APPLIED_AT_HARVEST", True,
    )
    smoke_n = os.environ.get("GEE_S1S2_SMOKE_TEST_N_PATCHES")
    smoke_n = int(smoke_n) if smoke_n else None

    # Phase B-specific knobs.
    variance_term_weight = _env_float("GEE_S1S2_PHASE_B_VARIANCE_WEIGHT", 0.3)
    variance_warmup_epoch = _env_int("GEE_S1S2_PHASE_B_VARIANCE_WARMUP_EPOCH", 30)
    cosine_min_lr = _env_float("GEE_S1S2_PHASE_B_COSINE_MIN_LR", 1e-5)
    # Optional per-band variance term weights (Phase B v3 onwards). Comma-
    # separated 6 floats in S2_BANDS order (B02, B03, B04, B08, B11, B12).
    # Unset -> uniform variance_term_weight across bands.
    variance_band_weights_env = os.environ.get("GEE_S1S2_PHASE_B_VARIANCE_BAND_WEIGHTS")
    variance_band_weights: list[float] | None = None
    if variance_band_weights_env:
        variance_band_weights = [float(x) for x in variance_band_weights_env.split(",")]
        if len(variance_band_weights) != 6:
            raise RuntimeError(
                f"GEE_S1S2_PHASE_B_VARIANCE_BAND_WEIGHTS must have 6 floats; "
                f"got {len(variance_band_weights)}"
            )

    # Multi-temporal training reads from a parallel prefix
    # (gs://<bucket>/gee_s1s2_translator/multitemporal_v1/) which has its
    # own manifest and patches; models still land under operational_v1/
    # so the eval pipeline can load them next to the baseline.
    if phase == "mt":
        mt_prefix = os.environ.get(
            "GEE_S1S2_MULTITEMPORAL_PREFIX",
            "gee_s1s2_translator/multitemporal_v1",
        )
        manifest_base = f"gs://{gcs_bucket}/{mt_prefix}"
        harvest_manifest_uri = f"{manifest_base}/manifest.csv"
    else:
        manifest_base = f"gs://{gcs_bucket}/{gcs_prefix}"
        harvest_manifest_uri = f"{manifest_base}/manifest.csv"
    base_uri = f"gs://{gcs_bucket}/{gcs_prefix}"
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
    is_multitemporal = (phase == "mt")
    train_ds = build_dataset(
        train_uris, stats=stats, batch_size=batch_size,
        shuffle=True, apply_lee=apply_lee_in_pipeline, seed=random_seed,
        multitemporal=is_multitemporal,
    )
    val_ds = build_dataset(
        val_uris, stats=stats, batch_size=batch_size,
        shuffle=False, apply_lee=apply_lee_in_pipeline,
        multitemporal=is_multitemporal,
    )

    # Build per-phase model + loss + LR schedule + extra callbacks.
    extra_callbacks: list = []
    loss_fn = None  # falls through to combined_l1_l2_loss in trainer
    loss_label = "L1 + 0.5 * L2"
    lr_for_optimizer = learning_rate

    if phase == "b":
        unet = build_unet(
            input_shape=(256, 256, 2), out_channels=6, base_channels=32,
            output_activation="linear",
        )
        loss_fn = CombinedL1L2VarianceLoss(
            variance_weight=0.0,
            variance_band_weights=variance_band_weights,
        )
        # The warmup callback gates variance_weight (the global multiplier);
        # per-band weights stay constant. When per-band weights are set, the
        # global multiplier should be 1.0 after warmup so per-band weights
        # are taken at face value; when not set, fall back to the scalar
        # variance_term_weight (Phase B v2 behaviour).
        warmup_target = 1.0 if variance_band_weights else variance_term_weight
        extra_callbacks.append(VarianceWeightWarmup(
            loss_fn=loss_fn,
            target_weight=warmup_target,
            warmup_epoch=variance_warmup_epoch,
        ))
        if variance_band_weights:
            loss_label = (
                f"L1 + 0.5 * L2 + sum_b w_b * |std-diff_b| (warmup at epoch "
                f"{variance_warmup_epoch}; band weights "
                f"{','.join(f'{w:.2f}' for w in variance_band_weights)})"
            )
        else:
            loss_label = (
                f"L1 + 0.5 * L2 + {variance_term_weight} * variance_term "
                f"(warmup at epoch {variance_warmup_epoch})"
            )
        # Cosine LR decay 1e-4 -> 1e-5 over the full epoch budget.
        steps_per_epoch = max(1, len(train_uris) // batch_size)
        total_decay_steps = max_epochs * steps_per_epoch
        lr_for_optimizer = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_decay_steps,
            alpha=cosine_min_lr / learning_rate,
        )
        LOG.info(
            "Phase B: linear output, variance term weight=%.3f from epoch %d, "
            "cosine LR %.3g -> %.3g over %d steps (%d epochs * %d steps/epoch).",
            variance_term_weight, variance_warmup_epoch,
            learning_rate, cosine_min_lr, total_decay_steps,
            max_epochs, steps_per_epoch,
        )
    elif phase == "c":
        unet = build_residual_unet(
            input_shape=(256, 256, 2), out_channels=6, base_channels=32,
        )
        LOG.info("Phase C: residual U-Net (linear S1 baseline + delta, sigmoid'd).")
    elif phase == "mt":
        # Multi-temporal: 6-channel S1 input (VV/VH × 3 acquisitions). Reuse
        # Phase B v2 hyperparameters: linear output + variance-aware loss
        # + cosine LR decay + warmup at epoch 15 + patience 40.
        unet = build_unet(
            input_shape=(256, 256, 6), out_channels=6, base_channels=32,
            output_activation="linear",
        )
        loss_fn = CombinedL1L2VarianceLoss(variance_weight=0.0)
        extra_callbacks.append(VarianceWeightWarmup(
            loss_fn=loss_fn, target_weight=variance_term_weight,
            warmup_epoch=variance_warmup_epoch,
        ))
        loss_label = (
            f"L1 + 0.5 * L2 + {variance_term_weight} * variance_term "
            f"(warmup at epoch {variance_warmup_epoch}); 6-channel S1 input"
        )
        steps_per_epoch = max(1, len(train_uris) // batch_size)
        total_decay_steps = max_epochs * steps_per_epoch
        lr_for_optimizer = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_decay_steps,
            alpha=cosine_min_lr / learning_rate,
        )
        LOG.info("Phase MT: 6-channel S1 input (3 acquisitions x 2 polarisations); "
                 "Phase B v2 loss family.")
    else:
        unet = build_unet(input_shape=(256, 256, 2), out_channels=6, base_channels=32)

    LOG.info("Model parameter count: %d (name=%s)", unet.count_params(), unet.name)

    result = train_model(
        unet,
        train_ds=train_ds, val_ds=val_ds,
        checkpoint_uri=model_output_prefix + "unet.keras",
        sidecar_uri=model_output_prefix + "unet_sidecar.json",
        log_uri=model_output_prefix + "train_unet.csv",
        learning_rate=lr_for_optimizer,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        loss_fn=loss_fn,
        extra_callbacks=extra_callbacks,
        loss_label=loss_label,
        extra_metadata={
            "run_name": training_run_name,
            "phase": phase or "baseline",
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
