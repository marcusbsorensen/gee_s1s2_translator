"""Loss functions matching v2's PyTorch training.

v2 used a combined ``L1 + 0.5 * L2`` loss to keep the training signal
robust to outlier pixels (L1) while still penalising large errors
quadratically (L2). The 1.0 / 0.5 weighting was chosen empirically
in v2 and we mirror it here.
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
