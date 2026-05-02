"""4-level U-Net matching v2's PyTorch architecture, ported to TensorFlow/Keras.

Architecture choices (mirrored from v2):

* 4 encoder blocks, 4 decoder blocks, with a bottleneck.
* Each conv block: ``Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU``.
* Downsampling: 2x2 max pool. Upsampling: bilinear (not transposed conv).
* Skip connections concatenate the encoder feature maps with the
  upsampled decoder maps.
* Output: 1x1 conv to 6 channels, sigmoid activation, mapping to the
  [0, 1] reflectance range of the L2A bands the model predicts.

Channel counts at each level: 32 -> 64 -> 128 -> 256 -> 512 (bottleneck),
then mirrored back. v2 used the same depths.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as kl, Model

DEFAULT_BASE_CHANNELS = 32


def _conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Conv-BN-ReLU x 2 (the U-Net standard double-conv)."""
    x = kl.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = kl.BatchNormalization(name=f"{name}_bn1")(x)
    x = kl.ReLU(name=f"{name}_relu1")(x)
    x = kl.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = kl.BatchNormalization(name=f"{name}_bn2")(x)
    x = kl.ReLU(name=f"{name}_relu2")(x)
    return x


def build_unet(
    input_shape: tuple[int, int, int] = (256, 256, 2),
    out_channels: int = 6,
    base_channels: int = DEFAULT_BASE_CHANNELS,
) -> Model:
    """Build a 4-level U-Net.

    Args:
        input_shape: (H, W, C). C must equal the number of S1 polarisations
            fed in (default 2 for VV, VH).
        out_channels: number of S2 reflectance bands predicted (default 6:
            B02, B03, B04, B08, B11, B12).
        base_channels: filter count at level 0 (doubled at each deeper level).

    Returns:
        A compiled-ready Keras :class:`Model`. Compile it externally with
        the loss + optimizer of your choice.
    """
    inputs = kl.Input(shape=input_shape, name="s1_input")

    # Encoder
    e1 = _conv_block(inputs, base_channels, "enc1")
    p1 = kl.MaxPool2D(2, name="pool1")(e1)
    e2 = _conv_block(p1, base_channels * 2, "enc2")
    p2 = kl.MaxPool2D(2, name="pool2")(e2)
    e3 = _conv_block(p2, base_channels * 4, "enc3")
    p3 = kl.MaxPool2D(2, name="pool3")(e3)
    e4 = _conv_block(p3, base_channels * 8, "enc4")
    p4 = kl.MaxPool2D(2, name="pool4")(e4)

    # Bottleneck
    b = _conv_block(p4, base_channels * 16, "bottleneck")

    # Decoder with bilinear upsampling + concat skip + double-conv.
    u4 = kl.UpSampling2D(size=2, interpolation="bilinear", name="up4")(b)
    u4 = kl.Concatenate(name="concat4")([u4, e4])
    d4 = _conv_block(u4, base_channels * 8, "dec4")

    u3 = kl.UpSampling2D(size=2, interpolation="bilinear", name="up3")(d4)
    u3 = kl.Concatenate(name="concat3")([u3, e3])
    d3 = _conv_block(u3, base_channels * 4, "dec3")

    u2 = kl.UpSampling2D(size=2, interpolation="bilinear", name="up2")(d3)
    u2 = kl.Concatenate(name="concat2")([u2, e2])
    d2 = _conv_block(u2, base_channels * 2, "dec2")

    u1 = kl.UpSampling2D(size=2, interpolation="bilinear", name="up1")(d2)
    u1 = kl.Concatenate(name="concat1")([u1, e1])
    d1 = _conv_block(u1, base_channels, "dec1")

    outputs = kl.Conv2D(out_channels, 1, activation="sigmoid", name="output")(d1)

    return Model(inputs=inputs, outputs=outputs, name="unet_s1_to_s2")
