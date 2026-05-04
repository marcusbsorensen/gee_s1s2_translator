"""Loss functions matching v2's PyTorch training.

v2 used a combined ``L1 + 0.5 * L2`` loss to keep the training signal
robust to outlier pixels (L1) while still penalising large errors
quadratically (L2). The 1.0 / 0.5 weighting was chosen empirically
in v2 and we mirror it here.

Phase B adds an optional variance-matching term: per-batch absolute
difference between predicted and true per-band standard deviations,
summed across the six output bands. The intent is to push back against
the variance-collapse failure mode where the U-Net regresses toward
per-band means. A scalar ``tf.Variable`` weight is exposed on
:class:`CombinedL1L2VarianceLoss` so a callback can ramp the term in
after a warmup window without recompiling the model.
"""
from __future__ import annotations

import tensorflow as tf

L1_WEIGHT = 1.0
L2_WEIGHT = 0.5


def combined_l1_l2_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """L1 + 0.5 * L2 mean reduction across all dims."""
    abs_err = tf.abs(y_true - y_pred)
    sq_err = tf.square(y_true - y_pred)
    l1 = tf.reduce_mean(abs_err)
    l2 = tf.reduce_mean(sq_err)
    return L1_WEIGHT * l1 + L2_WEIGHT * l2


class CombinedL1L2VarianceLoss(tf.keras.losses.Loss):
    """L1 + 0.5*L2 + variance_weight * sum_b |std(y_pred_b) - std(y_true_b)|.

    The variance term is computed per-batch (std taken across the batch +
    spatial dims for each output channel). ``variance_weight`` is a
    :class:`tf.Variable` so a callback can update it on epoch boundaries
    without recompiling the model. Set it to 0.0 during the warmup epochs
    and to the target weight afterwards.
    """

    def __init__(
        self,
        l1_weight: float = L1_WEIGHT,
        l2_weight: float = L2_WEIGHT,
        variance_weight: float = 0.0,
        name: str = "combined_l1_l2_variance",
    ) -> None:
        super().__init__(name=name)
        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)
        self.variance_weight = tf.Variable(
            float(variance_weight),
            trainable=False,
            dtype=tf.float32,
            name="variance_weight",
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
        l2 = tf.reduce_mean(tf.square(y_true - y_pred))
        # Per-band std across (batch, H, W). Output shape: (C,).
        true_std = tf.math.reduce_std(tf.cast(y_true, tf.float32), axis=[0, 1, 2])
        pred_std = tf.math.reduce_std(tf.cast(y_pred, tf.float32), axis=[0, 1, 2])
        var_term = tf.reduce_sum(tf.abs(true_std - pred_std))
        return (self.l1_weight * l1
                + self.l2_weight * l2
                + self.variance_weight * var_term)


class VarianceWeightWarmup(tf.keras.callbacks.Callback):
    """Step the loss's variance-term weight from 0 to ``target_weight``
    on ``on_epoch_begin`` of epoch ``warmup_epoch``.

    Epoch numbering matches Keras's ``on_epoch_begin`` (0-based), so
    ``warmup_epoch=30`` activates the term at the start of the 31st
    training epoch (1-based).
    """

    def __init__(
        self,
        loss_fn: CombinedL1L2VarianceLoss,
        target_weight: float = 0.3,
        warmup_epoch: int = 30,
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.target_weight = float(target_weight)
        self.warmup_epoch = int(warmup_epoch)

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        new_weight = self.target_weight if epoch >= self.warmup_epoch else 0.0
        current = float(self.loss_fn.variance_weight.numpy())
        if current != new_weight:
            self.loss_fn.variance_weight.assign(new_weight)
