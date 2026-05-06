"""Multi-temporal v1 inference on Cavenham 26-Jun-2024.

Self-contained local inference. Reads the 4 multi-temporal TFRecord
patches that the small MT harvest produced, decodes them, runs the
multi-temporal U-Net forward pass, mosaics with cosine blending, and
writes a 9-band predicted GeoTIFF aligned with the existing
single-temporal predictions for direct visual comparison in Figure 5.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
import tensorflow as tf
from google.cloud import storage

# ---------- constants matching the training config ----------
S1_BANDS_MULTITEMPORAL = ["VV_t0", "VH_t0", "VV_t1w", "VH_t1w", "VV_t3w", "VH_t3w"]
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
ALL_BANDS_MT = S1_BANDS_MULTITEMPORAL + S2_BANDS
HW = 256

# ---------- paths ----------
BUCKET = "marcus-heath-fire-mapping"
PROJECT = "wildfire-495012"
PATCH_PREFIX = "gee_s1s2_translator/multitemporal_v1_cavenham_inference/patches/test/cavenham-heath/"
MODEL_URI = "gs://marcus-heath-fire-mapping/gee_s1s2_translator/operational_v1/models/multitemporal_v1_t4/unet.keras"
S1_STATS_URI = "gs://marcus-heath-fire-mapping/gee_s1s2_translator/operational_v1/s1_stats.json"
V3_CALIB_LOCAL = "/Users/marcus/Documents/Healing Is Here/Client admin/Sonia Lasocki/Wildfire project/gee_s1s2_translator/training/src/training/calibration/postfit_affine_v3.json"

OUT_LOCAL = Path(__file__).parent / "cavenham_mt_predicted_s2.tif"


def _band_feature_spec_mt():
    return {b: tf.io.FixedLenFeature([HW * HW], tf.float32) for b in ALL_BANDS_MT}


def _decode_patch(local_path: str):
    spec = _band_feature_spec_mt()
    raw = next(iter(tf.data.TFRecordDataset([local_path], compression_type="GZIP")))
    parsed = tf.io.parse_single_example(raw, spec)
    s1 = tf.stack([tf.reshape(parsed[b], [HW, HW]) for b in S1_BANDS_MULTITEMPORAL], axis=-1)
    s2 = tf.stack([tf.reshape(parsed[b], [HW, HW]) for b in S2_BANDS], axis=-1)
    return s1.numpy().astype(np.float32), s2.numpy().astype(np.float32)


def _read_sidecar(bucket, gcs_path: str) -> Tuple[str, Affine]:
    """Read GEE export sidecar JSON next to the TFRecord; return (crs, transform)."""
    sidecar = gcs_path.replace(".tfrecord.gz", ".json")
    text = bucket.blob(sidecar).download_as_text()
    j = json.loads(text)
    proj = j["projection"]
    crs = proj["crs"]
    aff = proj["affine"]["doubleMatrix"]
    transform = Affine(*aff[:6])
    return crs, transform


def _normalize_s1(s1: np.ndarray, mean_vv: float, std_vv: float, mean_vh: float, std_vh: float) -> np.ndarray:
    """6-channel normalization: VV+VH stats reused across the 3 acquisitions."""
    means = np.array([mean_vv, mean_vh, mean_vv, mean_vh, mean_vv, mean_vh], dtype=np.float32)
    stds  = np.array([std_vv,  std_vh,  std_vv,  std_vh,  std_vv,  std_vh ], dtype=np.float32)
    return (s1 - means[None, None, :]) / stds[None, None, :]


def _cosine_taper(h: int, w: int, margin: int = 32) -> np.ndarray:
    def _ax(n):
        x = np.ones(n, dtype=np.float32)
        m = min(margin, n // 2)
        for i in range(m):
            x[i] = 0.5 * (1 - np.cos(np.pi * i / m))
            x[n - 1 - i] = 0.5 * (1 - np.cos(np.pi * i / m))
        return x
    return np.outer(_ax(h), _ax(w)).astype(np.float32)


def _mosaic_patches(patches: List[Tuple[np.ndarray, str, Affine]], margin: int = 32):
    """Cosine-blended mosaic. patches: list of (CHW, crs, affine)."""
    crs = patches[0][1]
    res_x = patches[0][2].a
    res_y = -patches[0][2].e
    minx = min(p[2].c for p in patches)
    maxy = max(p[2].f for p in patches)
    maxx = max(p[2].c + p[0].shape[2] * res_x for p in patches)
    miny = min(p[2].f - p[0].shape[1] * res_y for p in patches)
    W = int(round((maxx - minx) / res_x))
    H = int(round((maxy - miny) / res_y))
    transform = Affine(res_x, 0.0, minx, 0.0, -res_y, maxy)
    C = patches[0][0].shape[0]
    ws = np.zeros((C, H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    for arr, pcrs, pt in patches:
        assert pcrs == crs
        c, h, w = arr.shape
        col = int(round((pt.c - minx) / res_x))
        row = int(round((maxy - pt.f) / res_y))
        win = _cosine_taper(h, w, margin=margin)
        finite = np.isfinite(arr).all(axis=0)
        eff = win * finite.astype(np.float32)
        clean = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
        ws[:, row:row+h, col:col+w] += clean * eff[None, :, :]
        wsum[row:row+h, col:col+w] += eff
    out = np.full_like(ws, np.nan)
    valid = wsum > 0
    for i in range(C):
        out[i][valid] = ws[i][valid] / wsum[valid]
    return out, transform, crs


def main():
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)

    # Load s1 stats
    print(f"loading s1 stats from {S1_STATS_URI}")
    stats_text = bucket.blob(S1_STATS_URI.replace(f"gs://{BUCKET}/", "")).download_as_text()
    stats = json.loads(stats_text)
    mean_vv, std_vv = stats["mean"]["VV"], stats["std"]["VV"]
    mean_vh, std_vh = stats["mean"]["VH"], stats["std"]["VH"]
    print(f"  S1 stats: VV mean={mean_vv:+.3f} std={std_vv:.3f} | VH mean={mean_vh:+.3f} std={std_vh:.3f}")

    # Load v3 calibration (lives in the project repo, not on GCS)
    print(f"loading v3 calibration from local: {V3_CALIB_LOCAL}")
    with open(V3_CALIB_LOCAL) as f:
        v3 = json.load(f)
    slopes = np.array([v3["calibration"][b]["slope"]     for b in S2_BANDS], dtype=np.float32)
    intercepts = np.array([v3["calibration"][b]["intercept"] for b in S2_BANDS], dtype=np.float32)

    # Build MT architecture locally (the saved .keras file has TF 2.15
    # internals that don't load cleanly under TF 2.20), then load_weights.
    print(f"building MT architecture and loading weights from {MODEL_URI}")
    import sys as _sys
    _sys.path.insert(0, "/Users/marcus/Documents/Healing Is Here/Client admin/Sonia Lasocki/Wildfire project/gee_s1s2_translator/training/src")
    from training.model import build_unet
    model = build_unet(input_shape=(256, 256, 6), out_channels=6, base_channels=32, output_activation="linear")
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        local_keras = tmp.name
    bucket.blob(MODEL_URI.replace(f"gs://{BUCKET}/", "")).download_to_filename(local_keras)
    model.load_weights(local_keras)
    print(f"  loaded model: {model.name}, params={model.count_params():,}")

    # List MT TFRecord patches
    blobs = [b for b in bucket.list_blobs(prefix=PATCH_PREFIX)
             if b.name.endswith(".tfrecord.gz")]
    print(f"\nfound {len(blobs)} MT patches at {PATCH_PREFIX}")
    for b in blobs:
        print(f"  {b.name}")
    if len(blobs) == 0:
        raise SystemExit("no patches found — wait for GEE export tasks to finish")

    # Decode + infer per patch
    pred_patches: list = []
    for b in blobs:
        local = Path(tempfile.gettempdir()) / Path(b.name).name
        b.download_to_filename(local)
        s1, s2_truth = _decode_patch(str(local))  # s1 (HW, HW, 6)  s2_truth (HW, HW, 6)
        s1_norm = _normalize_s1(s1, mean_vv, std_vv, mean_vh, std_vh)
        # Forward pass (linear output for MT model)
        x = s1_norm[None, ...]  # (1, HW, HW, 6)
        y = model.predict(x, verbose=0)[0]  # (HW, HW, 6)
        y = np.clip(y, 0.0, 1.0)
        # Apply v3 calibration per band
        y_cal = y * slopes[None, None, :] + intercepts[None, None, :]
        y_cal = np.clip(y_cal, 0.0, 1.0)
        # Read sidecar for georef
        crs, transform = _read_sidecar(bucket, b.name)
        # CHW
        chw = np.moveaxis(y_cal, -1, 0).astype(np.float32)
        pred_patches.append((chw, crs, transform))
        print(f"  patch {Path(b.name).name}: median predicted reflectance B04={float(np.median(y_cal[..., 2])):.3f} B08={float(np.median(y_cal[..., 3])):.3f}")

    # Mosaic
    print("\nmosaicking with cosine blend")
    mosaic, transform, crs = _mosaic_patches(pred_patches, margin=32)
    # Add NDVI/NBR/NDWI
    b02, b03, b04, b08, b11, b12 = mosaic[0], mosaic[1], mosaic[2], mosaic[3], mosaic[4], mosaic[5]
    eps = 1e-9
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    nbr  = (b08 - b12) / (b08 + b12 + eps)
    ndwi = (b03 - b08) / (b03 + b08 + eps)
    full = np.stack([b02, b03, b04, b08, b11, b12, ndvi, nbr, ndwi], axis=0).astype(np.float32)
    print(f"mosaic shape: {full.shape}, transform: {transform}, crs: {crs}")

    # Save
    H, W = full.shape[1], full.shape[2]
    profile = dict(driver="GTiff", height=H, width=W, count=full.shape[0],
                   dtype="float32", crs=crs, transform=transform,
                   compress="deflate", predictor=2)
    with rasterio.open(OUT_LOCAL, "w", **profile) as ds:
        for i in range(full.shape[0]):
            ds.write(full[i], i + 1)
        ds.descriptions = tuple(S2_BANDS + ["NDVI", "NBR", "NDWI"])
    print(f"\nwrote {OUT_LOCAL} ({OUT_LOCAL.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
