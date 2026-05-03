"""Data-pipeline tests using small synthetic TFRecord fixtures.

Runs without GPU and without GCS — the tests write a tiny TFRecord
shard to a tmp_path, build the dataset, and assert on shape /
normalisation / NaN handling.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from training.data import (
    ALL_BANDS, S1_BANDS, S2_BANDS, S1Stats,
    build_dataset, load_manifest, split_uris,
)


def _make_tfrecord(path: Path, n_records: int = 2, hw: int = 32, seed: int = 0) -> None:
    """Write a tiny gzipped TFRecord matching the GEE export format.

    GEE's ``Export.image.toCloudStorage`` with ``patchDimensions`` writes
    each band as a flat ``tf.train.FloatList`` of ``H*W`` values directly
    in the Example proto (NOT as a serialised tensor). We mirror that
    format here so the tests exercise the real production decode path.

    Patch size kept small (32x32 instead of 256x256) to keep tests fast.
    """
    rng = np.random.default_rng(seed)
    with tf.io.TFRecordWriter(str(path), options="GZIP") as w:
        for _ in range(n_records):
            features = {}
            for b in ALL_BANDS:
                if b in S1_BANDS:
                    arr = rng.normal(loc=-12.0, scale=2.0, size=(hw, hw)).astype(np.float32)
                else:
                    arr = rng.uniform(0.0, 1.0, size=(hw, hw)).astype(np.float32)
                features[b] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=arr.ravel().tolist())
                )
            example = tf.train.Example(features=tf.train.Features(feature=features))
            w.write(example.SerializeToString())


@pytest.fixture
def synthetic_shard(tmp_path: Path) -> Path:
    p = tmp_path / "synthetic.tfrecord.gz"
    _make_tfrecord(p, n_records=4, hw=32)
    return p


@pytest.fixture
def synthetic_manifest(tmp_path: Path, synthetic_shard: Path) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "aoi_name", "s1_id", "s1_acquired", "s2_id", "s2_acquired",
            "split", "tfrecord_uri", "origin_index",
        ])
        writer.writeheader()
        for split, idx in [("train", 0), ("train", 1), ("val", 2), ("test", 3)]:
            writer.writerow({
                "pair_id": f"p{idx}", "aoi_name": "Test AOI",
                "s1_id": "S1", "s1_acquired": "2024-01-01T00:00:00Z",
                "s2_id": "S2", "s2_acquired": "2024-01-01T00:00:00Z",
                "split": split, "tfrecord_uri": str(synthetic_shard),
                "origin_index": idx,
            })
    return manifest_path


# -------------------- manifest + split routing -------------------- #

def test_load_manifest_returns_one_entry_per_row(synthetic_manifest: Path) -> None:
    entries = load_manifest(str(synthetic_manifest))
    assert len(entries) == 4
    assert {e.split for e in entries} == {"train", "val", "test"}


def test_split_uris_filters_correctly(synthetic_manifest: Path) -> None:
    entries = load_manifest(str(synthetic_manifest))
    train = split_uris(entries, "train")
    val = split_uris(entries, "val")
    test = split_uris(entries, "test")
    assert len(train) == 2
    assert len(val) == 1
    assert len(test) == 1
    assert all(u.endswith(".tfrecord.gz") for u in train + val + test)


def test_load_manifest_strips_wildcard_uris(tmp_path: Path) -> None:
    """Phase-1 v1 wrote ``...*.tfrecord.gz``; load_manifest tolerates both forms."""
    p = tmp_path / "manifest.csv"
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "aoi_name", "s1_id", "s1_acquired", "s2_id", "s2_acquired",
            "split", "tfrecord_uri", "origin_index",
        ])
        writer.writeheader()
        writer.writerow({
            "pair_id": "p", "aoi_name": "T", "s1_id": "S1",
            "s1_acquired": "2024-01-01T00:00:00Z",
            "s2_id": "S2", "s2_acquired": "2024-01-01T00:00:00Z",
            "split": "train",
            "tfrecord_uri": "gs://bucket/path/example*.tfrecord.gz",
            "origin_index": 0,
        })
    e = load_manifest(str(p))[0]
    assert e.tfrecord_uri == "gs://bucket/path/example.tfrecord.gz"


# -------------------- dataset shape + normalisation -------------------- #

def test_build_dataset_shapes(synthetic_shard: Path) -> None:
    stats = S1Stats(mean={"VV": -10.0, "VH": -16.0},
                    std={"VV": 2.0, "VH": 2.0})
    # The synthetic fixture has hw=32; the production hw=256 default
    # is the build_dataset default. Override via the patch_hw kwarg.
    ds = build_dataset([str(synthetic_shard)], stats=stats,
                       batch_size=2, shuffle=False, patch_hw=32)
    for s1, s2 in ds.take(1):
        assert s1.shape == (2, 32, 32, 2)
        assert s2.shape == (2, 32, 32, 6)
        # Z-scored S1 should be centred near 0 (input mean was -12, stats mean -10/-16
        # so the offsets are non-zero but the std should be order-unity).
        assert tf.reduce_max(tf.abs(s1)).numpy() < 50.0
        # S2 clipped to [0, 1].
        assert float(tf.reduce_min(s2).numpy()) >= 0.0
        assert float(tf.reduce_max(s2).numpy()) <= 1.0


def test_dataset_replaces_nan_with_zero(tmp_path: Path) -> None:
    """A patch with NaN should not produce NaN tensors."""
    p = tmp_path / "withnan.tfrecord.gz"
    rng = np.random.default_rng(0)
    with tf.io.TFRecordWriter(str(p), options="GZIP") as w:
        features = {}
        for b in ALL_BANDS:
            arr = rng.uniform(0.0, 1.0, size=(16, 16)).astype(np.float32)
            arr[0, 0] = np.nan
            features[b] = tf.train.Feature(
                float_list=tf.train.FloatList(value=arr.ravel().tolist())
            )
        example = tf.train.Example(features=tf.train.Features(feature=features))
        w.write(example.SerializeToString())

    stats = S1Stats(mean={"VV": 0.0, "VH": 0.0}, std={"VV": 1.0, "VH": 1.0})
    ds = build_dataset([str(p)], stats=stats, batch_size=1, shuffle=False, patch_hw=16)
    for s1, s2 in ds.take(1):
        assert not bool(tf.reduce_any(tf.math.is_nan(s1)).numpy())
        assert not bool(tf.reduce_any(tf.math.is_nan(s2)).numpy())
