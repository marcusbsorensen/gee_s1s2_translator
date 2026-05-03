"""Model-architecture tests: forward-pass shape, parameter count, baseline."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from training.linear_baseline import build_linear_baseline
from training.model import build_unet


def test_unet_output_shape() -> None:
    m = build_unet(input_shape=(64, 64, 2), out_channels=6, base_channels=8)
    x = tf.random.normal((2, 64, 64, 2))
    y = m(x, training=False)
    assert y.shape == (2, 64, 64, 6)


def test_unet_output_is_in_unit_interval() -> None:
    """Sigmoid output should be in [0, 1]."""
    m = build_unet(input_shape=(32, 32, 2), out_channels=6, base_channels=8)
    x = tf.random.normal((1, 32, 32, 2))
    y = m(x, training=False).numpy()
    assert y.min() >= 0.0 - 1e-6
    assert y.max() <= 1.0 + 1e-6


def test_unet_has_skip_connections() -> None:
    """Concat layers indicate the U-shape — one per level."""
    m = build_unet(input_shape=(32, 32, 2), out_channels=6, base_channels=8)
    concat_layers = [
        l for l in m.layers if isinstance(l, tf.keras.layers.Concatenate)
    ]
    assert len(concat_layers) == 4   # one per encoder level


def test_unet_param_count_in_expected_order() -> None:
    """Default 32-channel U-Net should have ~7.7M params (matching v2 ballpark)."""
    m = build_unet(input_shape=(256, 256, 2), out_channels=6, base_channels=32)
    n_params = m.count_params()
    assert 5_000_000 < n_params < 12_000_000, (
        f"U-Net parameter count {n_params:,} outside expected 5M-12M range "
        f"for the v2-equivalent architecture"
    )


# -------------------- linear baseline -------------------- #

def test_linear_baseline_output_shape() -> None:
    m = build_linear_baseline(input_shape=(64, 64, 2), out_channels=6)
    y = m(tf.random.normal((2, 64, 64, 2)), training=False)
    assert y.shape == (2, 64, 64, 6)


def test_linear_baseline_param_count() -> None:
    """Per-pixel 1x1 conv has 2*6 + 6 = 18 params total."""
    m = build_linear_baseline(input_shape=(256, 256, 2), out_channels=6)
    assert m.count_params() == 18


def test_linear_baseline_is_translation_invariant() -> None:
    """Per-pixel model: shifting input row should shift output row identically."""
    m = build_linear_baseline(input_shape=(8, 8, 2), out_channels=6)
    x = tf.random.normal((1, 8, 8, 2), seed=tf.random.set_seed(0))
    # Shift by one column.
    y_full = m(x, training=False).numpy()
    y_shift_col = m(tf.roll(x, shift=1, axis=2), training=False).numpy()
    np.testing.assert_allclose(
        y_shift_col[..., :, 1:, :], y_full[..., :, :-1, :], atol=1e-5,
    )
