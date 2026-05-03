"""Training loop with early stopping + GCS-backed CSV logging.

Mirrors v2's trainer behaviour:

* Adam optimiser, learning rate 1e-4 (configurable).
* Combined L1 + 0.5 * L2 loss (:mod:`losses`).
* Early stopping on val RMSE with patience 15.
* Best-checkpoint by val RMSE.
* CSV training log written to GCS after each epoch so an interrupted
  Colab session leaves a usable record.
* Sidecar JSON with run metadata + best metric saved alongside the
  checkpoint.

Resumability: if ``checkpoint_uri`` and ``sidecar_uri`` already exist
when :func:`train` is called, weights and epoch counter are loaded from
GCS and training continues from where it left off. This makes Colab's
12-hour session ceiling a non-issue: a partially-trained run picks up
on the next ``Run all``. Pass ``resume=False`` to force a fresh start.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

from .losses import combined_l1_l2_loss

LOG = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    best_val_rmse: float
    best_epoch: int
    epochs_run: int
    history: list[dict]            # one dict per epoch (loss, val_loss, val_rmse, ...)
    checkpoint_uri: str
    sidecar_uri: str
    log_uri: str


def _val_rmse_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def train(
    model: tf.keras.Model,
    *,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    checkpoint_uri: str,
    sidecar_uri: str,
    log_uri: str,
    learning_rate: float = 1e-4,
    max_epochs: int = 80,
    early_stopping_patience: int = 15,
    extra_metadata: dict | None = None,
    resume: bool = True,
) -> TrainingResult:
    """Run training. Returns :class:`TrainingResult` with best-epoch metrics.

    The model is compiled internally with Adam + combined L1+L2 loss + RMSE
    metric. Pass a fresh (uncompiled) :class:`tf.keras.Model`.

    If ``resume=True`` (default) and prior ``checkpoint_uri`` + ``sidecar_uri``
    exist on GCS, weights are loaded and training continues from the next
    epoch. Set ``resume=False`` to force a fresh run (e.g. after a code
    change to the model architecture).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_l1_l2_loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    # Resume from prior checkpoint if both checkpoint and sidecar exist.
    # The optimizer is freshly compiled above, so Adam moments restart from
    # zero — for our short runs this re-warms within an epoch and is cheaper
    # than the alternative (load_model with optimizer state, which clashes
    # with the freshly-built `model` argument).
    initial_epoch = 0
    history_rows: list[dict] = []
    if resume and tf.io.gfile.exists(checkpoint_uri) and tf.io.gfile.exists(sidecar_uri):
        try:
            model.load_weights(checkpoint_uri)
            with tf.io.gfile.GFile(sidecar_uri, "r") as f:
                prev_sidecar = json.load(f)
            initial_epoch = int(prev_sidecar.get("epochs_run", 0))
            if tf.io.gfile.exists(log_uri):
                with tf.io.gfile.GFile(log_uri, "r") as f:
                    history_rows = [
                        {k: (float(v) if k != "epoch" else int(v)) for k, v in row.items()}
                        for row in csv.DictReader(f)
                    ]
            LOG.info(
                "Resuming from %s at epoch %d (prior best val_rmse=%.4f).",
                checkpoint_uri, initial_epoch,
                prev_sidecar.get("best_val_rmse", float("nan")),
            )
        except Exception as e:
            LOG.warning("Could not resume from %s (%s); starting fresh.",
                        checkpoint_uri, e)
            initial_epoch = 0
            history_rows = []

    callbacks = []

    # Early stopping on val RMSE.
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_rmse", mode="min",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early)

    # CSV log to GCS after each epoch (re-write the whole CSV each time;
    # epoch counts are small enough that this is cheap). When resuming,
    # ``history_rows`` is pre-seeded with the prior run's rows so the new
    # log appends rather than overwrites.

    class GcsCsvLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            row = {"epoch": epoch + 1, **(logs or {})}
            history_rows.append(row)
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(row.keys()))
            writer.writeheader()
            for r in history_rows:
                writer.writerow(r)
            with tf.io.gfile.GFile(log_uri, "w") as f:
                f.write(buf.getvalue())

    callbacks.append(GcsCsvLogger())

    # Snapshot prior history before fit() appends to history_rows.
    prior_val_rmse = [float(r["val_rmse"]) for r in history_rows if "val_rmse" in r]

    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        initial_epoch=initial_epoch,
        epochs=max_epochs, callbacks=callbacks, verbose=2,
    )
    elapsed_s = time.time() - t0

    # Combine current run's val_rmse with prior history so best-epoch
    # bookkeeping survives resumption.
    new_val_rmse = [float(v) for v in history.history.get("val_rmse", [])]
    all_val_rmse = prior_val_rmse + new_val_rmse
    if all_val_rmse:
        best_rmse = float(min(all_val_rmse))
        best_epoch = int(np.argmin(all_val_rmse)) + 1
    else:
        best_rmse = float("nan")
        best_epoch = 0
    epochs_run = len(history_rows)

    # Save model + sidecar.
    model.save(checkpoint_uri)
    sidecar = {
        "best_val_rmse": best_rmse,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "training_seconds": elapsed_s,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "loss": "L1 + 0.5 * L2",
        "model_name": model.name,
        "extra": extra_metadata or {},
    }
    with tf.io.gfile.GFile(sidecar_uri, "w") as f:
        json.dump(sidecar, f, indent=2)
    LOG.info(
        "Training done: best val RMSE=%.4f at epoch %d (%d epochs in %.1fs).",
        best_rmse, best_epoch, epochs_run, elapsed_s,
    )

    return TrainingResult(
        best_val_rmse=best_rmse, best_epoch=best_epoch,
        epochs_run=epochs_run, history=history_rows,
        checkpoint_uri=checkpoint_uri, sidecar_uri=sidecar_uri, log_uri=log_uri,
    )
