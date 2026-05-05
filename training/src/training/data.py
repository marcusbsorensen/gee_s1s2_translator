"""TFRecord parsing + manifest-driven train/val/test split iteration.

Each Phase 1 TFRecord shard contains exactly one ``tf.train.Example`` —
one 256x256 patch with eight float32 bands::

    VV, VH                       (S1, calibrated gamma-naught dB,
                                  Lee 5x5 already applied at harvest)
    B02, B03, B04, B08, B11, B12 (S2 reflectance, already scaled to [0, 1]
                                  at harvest by ``stack_pair_image``)

Two GEE-port specifics worth knowing about (full discussion in
``docs/methodology_divergences.md``):

* **Lee speckle filter is already applied at harvest** (the default
  ``operational_v1.yaml`` enables ``calibration.speckle_filter``). The v2
  PyTorch project applied Lee client-side during training. So *do not*
  re-apply Lee in this pipeline. The opt-in flag in :func:`build_dataset`
  exists for the (unusual) case where someone re-harvests with
  ``speckle_filter.enabled: false`` and wants to apply Lee at training
  time instead.

* **S2 reflectance is already in [0, 1]** (divided by 10000 at harvest).
  No further scaling is needed.

S1 is normalised here by per-band z-score from ``s1_stats.json``; if
no stats file exists in GCS yet, :func:`load_or_compute_s1_stats`
computes one from the train split and writes it back.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import tensorflow as tf

LOG = logging.getLogger(__name__)

S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
ALL_BANDS = S1_BANDS + S2_BANDS
# Multi-temporal harvest stacks 3 S1 acquisitions (t0 = paired-with-S2,
# t1w = 7-14d prior, t3w = 14-28d prior) per truth pair.
S1_BANDS_MULTITEMPORAL = [
    "VV_t0", "VH_t0", "VV_t1w", "VH_t1w", "VV_t3w", "VH_t3w",
]
ALL_BANDS_MULTITEMPORAL = S1_BANDS_MULTITEMPORAL + S2_BANDS
DEFAULT_PATCH_HW = 256


# --------------------------------------------------------------------------- #
# Manifest loading + split routing
# --------------------------------------------------------------------------- #

@dataclass
class ManifestEntry:
    pair_id: str
    aoi_name: str
    split: str
    tfrecord_uri: str
    origin_index: int


def load_manifest(manifest_uri: str) -> list[ManifestEntry]:
    """Load the manifest CSV from GCS (or local) and return one entry per row.

    ``manifest_uri`` may be ``gs://bucket/path/manifest.csv`` or a local
    filesystem path. ``tf.io.gfile`` handles both transparently.
    """
    with tf.io.gfile.GFile(manifest_uri, "r") as f:
        text = f.read()
    rows = list(csv.DictReader(io.StringIO(text)))
    out: list[ManifestEntry] = []
    for r in rows:
        # Phase 1 wrote the URI with a trailing wildcard during the
        # earlier per-pair export; per-origin export writes a concrete
        # ``.tfrecord.gz`` URI. Tolerate both.
        uri = r["tfrecord_uri"].replace("*.tfrecord.gz", ".tfrecord.gz")
        out.append(ManifestEntry(
            pair_id=r["pair_id"],
            aoi_name=r["aoi_name"],
            split=r["split"],
            tfrecord_uri=uri,
            origin_index=int(r.get("origin_index") or 0),
        ))
    return out


def split_uris(entries: list[ManifestEntry], split: str) -> list[str]:
    return [e.tfrecord_uri for e in entries if e.split == split]


# --------------------------------------------------------------------------- #
# S1 statistics: compute once, persist to GCS, re-use
# --------------------------------------------------------------------------- #

@dataclass
class S1Stats:
    mean: dict[str, float]    # per-band mean (dB)
    std: dict[str, float]     # per-band std (dB)


def _band_feature_spec(hw: int = DEFAULT_PATCH_HW, multitemporal: bool = False) -> dict:
    """Per-band feature spec for the GEE-exported TFRecord format.

    GEE's ``Export.image.toCloudStorage`` with ``patchDimensions=[H, W]``
    writes each band as a flat ``tf.train.FloatList`` of ``H*W`` values
    inside the Example proto. So the right spec is
    ``FixedLenFeature([H*W], tf.float32)`` and we reshape after parsing.

    With ``multitemporal=True``, the spec covers the 6-channel S1 stack
    (VV/VH × {t0, t1w, t3w}) instead of the single 2-channel S1.
    """
    bands = ALL_BANDS_MULTITEMPORAL if multitemporal else ALL_BANDS
    return {b: tf.io.FixedLenFeature([hw * hw], tf.float32) for b in bands}


def _peek_patches(uris: Iterable[str], n_max: int,
                  hw: int = DEFAULT_PATCH_HW) -> list[dict[str, np.ndarray]]:
    """Read ``n_max`` patches without batching, return list of dicts."""
    spec = _band_feature_spec(hw)

    def _decode(rec):
        parsed = tf.io.parse_single_example(rec, spec)
        return {b: tf.reshape(parsed[b], [hw, hw]) for b in ALL_BANDS}

    out: list[dict[str, np.ndarray]] = []
    ds = tf.data.TFRecordDataset(list(uris), compression_type="GZIP").map(_decode)
    for example in ds.take(n_max):
        out.append({b: example[b].numpy() for b in ALL_BANDS})
    return out


def compute_s1_stats_from_train(
    train_uris: list[str], n_patches: int = 100,
) -> S1Stats:
    """Sample ``n_patches`` train shards and compute per-band S1 mean/std.

    Per-band statistics are computed across all sampled pixels, ignoring
    non-finite values.
    """
    peek = _peek_patches(train_uris, n_patches)
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for b in S1_BANDS:
        flat = np.concatenate([p[b].ravel() for p in peek])
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            raise RuntimeError(f"No finite values for S1 band {b!r} in {n_patches} sampled patches")
        means[b] = float(finite.mean())
        stds[b] = float(finite.std())
    return S1Stats(mean=means, std=stds)


def load_or_compute_s1_stats(
    train_uris: list[str], stats_uri: str, n_patches: int = 100,
) -> S1Stats:
    """Load ``stats_uri`` if it exists; otherwise compute and write back."""
    if tf.io.gfile.exists(stats_uri):
        with tf.io.gfile.GFile(stats_uri, "r") as f:
            d = json.load(f)
        return S1Stats(mean=d["mean"], std=d["std"])
    LOG.info("S1 stats file %s not found; computing from %d train patches.",
             stats_uri, n_patches)
    stats = compute_s1_stats_from_train(train_uris, n_patches=n_patches)
    with tf.io.gfile.GFile(stats_uri, "w") as f:
        json.dump({"mean": stats.mean, "std": stats.std,
                   "n_patches_sampled": n_patches,
                   "_about": "Per-band S1 mean/std (dB), used for z-score "
                             "normalisation in training/src/training/data.py"},
                  f, indent=2)
    LOG.info("Wrote S1 stats to %s", stats_uri)
    return stats


# --------------------------------------------------------------------------- #
# tf.data pipeline
# --------------------------------------------------------------------------- #

def _lee_filter_5x5(x: tf.Tensor) -> tf.Tensor:
    """Boxcar-mean Lee filter, 5x5. Operates per channel.

    Only used when callers opt in via ``apply_lee=True`` — the default
    Phase 1 harvest already applies Lee server-side, so applying it
    again here would over-smooth. See module docstring.
    """
    # 5x5 mean filter via depthwise conv with a uniform kernel.
    k = tf.ones((5, 5, x.shape[-1], 1), dtype=tf.float32) / 25.0
    return tf.nn.depthwise_conv2d(x[None, ...], k, strides=[1, 1, 1, 1],
                                  padding="SAME")[0]


def _build_decoder(stats: S1Stats, apply_lee: bool, hw: int = DEFAULT_PATCH_HW,
                   multitemporal: bool = False):
    if multitemporal:
        # Re-use t0 stats (computed on single-temporal corpus) for all 3
        # S1 epochs since they're the same Mullissa-calibrated S1 distribution.
        s1_band_names = S1_BANDS_MULTITEMPORAL
        s1_mean = tf.constant(
            [stats.mean["VV"], stats.mean["VH"]] * 3, dtype=tf.float32,
        )
        s1_std = tf.constant(
            [stats.std["VV"], stats.std["VH"]] * 3, dtype=tf.float32,
        )
    else:
        s1_band_names = S1_BANDS
        s1_mean = tf.constant([stats.mean[b] for b in S1_BANDS], dtype=tf.float32)
        s1_std = tf.constant([stats.std[b] for b in S1_BANDS], dtype=tf.float32)
    spec = _band_feature_spec(hw, multitemporal=multitemporal)
    n_s1 = len(s1_band_names)

    def _decode(rec):
        parsed = tf.io.parse_single_example(rec, spec)
        # Each band feature is a flat float32 list of length H*W from the
        # GEE export; reshape to (H, W) and stack to (H, W, C). S1 first,
        # then S2, matching stack_pair_image's export order.
        s1 = tf.stack(
            [tf.reshape(parsed[b], [hw, hw]) for b in s1_band_names], axis=-1,
        )
        s2 = tf.stack(
            [tf.reshape(parsed[b], [hw, hw]) for b in S2_BANDS], axis=-1,
        )
        # Pin the spatial shape so downstream layers know it.
        s1.set_shape([hw, hw, n_s1])
        s2.set_shape([hw, hw, len(S2_BANDS)])

        # Replace any non-finite pixels with zero before normalisation.
        s1 = tf.where(tf.math.is_finite(s1), s1, tf.zeros_like(s1))
        s2 = tf.where(tf.math.is_finite(s2), s2, tf.zeros_like(s2))

        if apply_lee:
            s1 = _lee_filter_5x5(s1)

        # Z-score normalise S1 (per-channel; for multi-temporal each of the
        # 6 channels is independently normalised against the same VV/VH stats).
        s1 = (s1 - s1_mean) / s1_std
        # S2 already in [0, 1] from harvest; clamp to be safe.
        s2 = tf.clip_by_value(s2, 0.0, 1.0)
        return s1, s2

    return _decode


def build_dataset(
    uris: list[str],
    stats: S1Stats,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    shuffle_buffer: int = 256,
    apply_lee: bool = False,
    repeat: bool = False,
    seed: int = 42,
    patch_hw: int = DEFAULT_PATCH_HW,
    multitemporal: bool = False,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` of (s1, s2) tensors from TFRecord URIs.

    Decoded patches are cached in memory so subsequent epochs do not re-stream
    from GCS — the dominant bottleneck on free-tier Colab. The Phase 1 dataset
    (~600 patches at 256x256x8 float32 ≈ 1.2 GB decoded) fits comfortably in
    T4 RAM. For larger datasets, swap ``cache()`` for ``cache("/tmp/<name>")``
    to spill to the runtime's local SSD.

    With ``multitemporal=True``, the decoder reads 6 S1 channels per patch
    (VV/VH × {t0, t1w, t3w}) instead of 2; the returned ``s1`` tensor has
    shape (H, W, 6).
    """
    decode = _build_decoder(stats, apply_lee=apply_lee, hw=patch_hw,
                            multitemporal=multitemporal)
    ds = tf.data.TFRecordDataset(uris, compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
