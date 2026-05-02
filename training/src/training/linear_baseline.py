"""Per-pixel linear baseline matching v2.

Architecture: a single 1x1 convolution mapping the 2-channel S1 input
to 6 reflectance channels. No spatial context — every output pixel is
a fixed linear combination of the same-position S1 pixel. Sigmoid
activation keeps the output in [0, 1] like the U-Net.

This is the reference "no spatial reasoning" baseline. The U-Net's
gain over this baseline is the architectural argument: v2 measured a
**3.3x driver-band variance retention** on Ashdown (U-Net 58.3 % vs
baseline 17.5 %), which is exactly the result we want to reproduce
(within the GEE-port's documented divergences) on the GEE-harvested
test set.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as kl, Model


def build_linear_baseline(
    input_shape: tuple[int, int, int] = (256, 256, 2),
    out_channels: int = 6,
) -> Model:
    """Build the per-pixel 1x1 conv baseline."""
    inputs = kl.Input(shape=input_shape, name="s1_input")
    outputs = kl.Conv2D(
        out_channels, 1, activation="sigmoid", name="output",
    )(inputs)
    return Model(inputs=inputs, outputs=outputs, name="linear_baseline_s1_to_s2")
