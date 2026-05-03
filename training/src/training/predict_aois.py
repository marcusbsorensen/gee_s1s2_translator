"""Vertex AI inference entrypoint: run U-Net + linear-baseline over the
three operational target AOIs and write 9-band predicted Sentinel-2
GeoTIFFs (B02, B03, B04, B08, B11, B12, NDVI, NBR, NDWI) to GCS.

For each target the script reads every TFRecord patch for the chosen
S1 acquisition date, runs the model, mosaics the predicted reflectance
into a single AOI-level raster using the per-patch GEE export sidecar
(``<patch>.json``) for georeferencing, computes the three derived
indices, and writes the result as a 9-band GeoTIFF.

The Hankley sanity-check additionally writes the truth Sentinel-2
GeoTIFF (assembled from the ``s2`` half of the same TFRecord patches)
and computes per-band + per-index MAE between predicted and truth.

Configuration via environment variables:

  GEE_S1S2_PROJECT_ID, GEE_S1S2_BUCKET, GEE_S1S2_PREFIX,
  GEE_S1S2_TRAINING_RUN_NAME (drives the input model paths).

Targets are hard-coded (matches the user's spec) — Brentmoor 2023-05-26,
Poors Allotment 2023-05-26, Hankley 2024-05-20. Predicted GeoTIFFs land
under ``models/<run>/predictions/{unet,linear_baseline}/``. Hankley truth
goes alongside the Hankley predictions; the MAE summary is written to
``predictions/hankley_sanity_mae.json``.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import Affine
import tensorflow as tf

from .data import ALL_BANDS, S1_BANDS, S2_BANDS, S1Stats, _band_feature_spec
from .model import build_unet
from .linear_baseline import build_linear_baseline

LOG = logging.getLogger("predict_aois")

# Default targets — the three operational AOIs covered by the pre-existing
# (cloud-free) harvest. Override via the ``GEE_S1S2_PREDICT_TARGETS_JSON``
# env var (a JSON array of the same dict shape) to repurpose this entry
# point for other AOI/date combinations, e.g. the post-fire 2022 cloud-
# covered window once that harvest completes.
DEFAULT_TARGETS = [
    {
        "aoi_slug": "brentmoor-area-training",
        "s1_date": "20230526",
        "out_label": "brentmoor-area-training_20230526",
        "save_truth": False,
    },
    {
        "aoi_slug": "poors-allotment-area-training",
        "s1_date": "20230526",
        "out_label": "poors-allotment-area-training_20230526",
        "save_truth": False,
    },
    {
        "aoi_slug": "hankley-common",
        "s1_date": "20240520",
        "out_label": "hankley-common_20240520",
        "save_truth": True,  # truth + MAE sanity check
    },
]

S2_BAND_INDEX = {b: i for i, b in enumerate(S2_BANDS)}


def _list_patches(bucket, aoi_slug: str, s1_date: str) -> List[str]:
    """List patch TFRecord blobs for a given AOI + S1 acquisition date.

    Scans both ``operational_v1/patches/`` (initial training harvest) and
    ``operational_v1/operational_v1/patches/`` (post-fire harvest, which ran
    with a doubled-prefix layout because GCS_PREFIX already included
    ``operational_v1`` and the harvest run_name appended it again).
    """
    prefixes = [
        os.environ.get("GEE_S1S2_PATCH_PREFIX",
                       "gee_s1s2_translator/operational_v1/patches/"),
        "gee_s1s2_translator/operational_v1/operational_v1/patches/",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for prefix in prefixes:
        for blob in bucket.list_blobs(prefix=prefix):
            name = blob.name
            if name in seen:
                continue
            if aoi_slug not in name or not name.endswith(".tfrecord.gz"):
                continue
            fname = name.split("/")[-1]
            parts = fname.replace(".tfrecord.gz", "").split("__")
            if len(parts) < 4:
                continue
            s1_s2_tile = parts[2]
            first_dash = s1_s2_tile.split("-")[0]
            if first_dash[:8] == s1_date:
                out.append(name)
                seen.add(name)
    out.sort()
    return out


def _decode_patch(tfrecord_uri: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a single-example TFRecord and return (s1, s2) at 256x256."""
    spec = _band_feature_spec(256)
    raw = next(iter(tf.data.TFRecordDataset([tfrecord_uri], compression_type="GZIP")))
    parsed = tf.io.parse_single_example(raw, spec)
    s1 = tf.stack([tf.reshape(parsed[b], [256, 256]) for b in S1_BANDS], axis=-1)
    s2 = tf.stack([tf.reshape(parsed[b], [256, 256]) for b in S2_BANDS], axis=-1)
    s1 = tf.where(tf.math.is_finite(s1), s1, tf.zeros_like(s1))
    s2 = tf.where(tf.math.is_finite(s2), s2, tf.zeros_like(s2))
    return s1.numpy(), s2.numpy()


def _read_sidecar(bucket, tfrecord_uri: str) -> Tuple[str, Affine]:
    """Read the patch-level sidecar JSON, return (crs_str, rasterio Affine)."""
    sidecar_path = tfrecord_uri.replace(".tfrecord.gz", ".json")
    blob = bucket.blob(sidecar_path)
    meta = json.loads(blob.download_as_text())
    crs = meta["projection"]["crs"]
    a, b, c, d, e, f = meta["projection"]["affine"]["doubleMatrix"]
    return crs, Affine(a, b, c, d, e, f)


def _normalize_s1(s1: np.ndarray, stats: S1Stats) -> np.ndarray:
    mean = np.array([stats.mean[b] for b in S1_BANDS], dtype=np.float32)
    std = np.array([stats.std[b] for b in S1_BANDS], dtype=np.float32)
    return (s1 - mean) / std


def _compute_indices(s2_chw: np.ndarray) -> np.ndarray:
    """Compute NDVI, NBR, NDWI from a 6-band (C,H,W) reflectance array.

    Returns a 9-band (C,H,W) array: B02, B03, B04, B08, B11, B12, NDVI, NBR, NDWI.
    """
    eps = 1e-9
    B02 = s2_chw[S2_BAND_INDEX["B02"]]
    B03 = s2_chw[S2_BAND_INDEX["B03"]]
    B04 = s2_chw[S2_BAND_INDEX["B04"]]
    B08 = s2_chw[S2_BAND_INDEX["B08"]]
    B11 = s2_chw[S2_BAND_INDEX["B11"]]
    B12 = s2_chw[S2_BAND_INDEX["B12"]]
    NDVI = (B08 - B04) / (B08 + B04 + eps)
    NBR = (B08 - B12) / (B08 + B12 + eps)
    NDWI = (B03 - B08) / (B03 + B08 + eps)
    return np.stack([B02, B03, B04, B08, B11, B12, NDVI, NBR, NDWI], axis=0).astype(np.float32)


def _mosaic_patches(patch_data: List[Tuple[np.ndarray, str, Affine]]) -> Tuple[np.ndarray, Affine, str]:
    """Mosaic per-patch (C,H,W) arrays into a single CHW raster using rasterio.

    All patches share CRS (single UTM zone per AOI). The final transform is the
    union extent's transform; pixels outside any patch are NaN.
    """
    crs = patch_data[0][1]
    datasets = []
    for arr, patch_crs, transform in patch_data:
        assert patch_crs == crs, f"CRS mismatch: {patch_crs} != {crs}"
        c, h, w = arr.shape
        memfile = MemoryFile()
        with memfile.open(
            driver="GTiff", height=h, width=w, count=c,
            dtype="float32", crs=crs, transform=transform, nodata=np.nan,
        ) as ds:
            ds.write(arr.astype(np.float32))
        datasets.append(memfile.open())

    mosaic, mosaic_transform = merge(datasets, nodata=np.nan)
    for ds in datasets:
        ds.close()
    return mosaic, mosaic_transform, crs


def _write_geotiff(uri: str, data_chw: np.ndarray, transform: Affine, crs: str,
                   band_descriptions: List[str]) -> int:
    """Write a CHW float32 raster as a multi-band GeoTIFF to a gs:// URI."""
    c, h, w = data_chw.shape
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        local_path = tmp.name
    try:
        with rasterio.open(
            local_path, "w",
            driver="GTiff", height=h, width=w, count=c,
            dtype="float32", crs=crs, transform=transform,
            compress="DEFLATE", predictor=3, tiled=True,
            blockxsize=256, blockysize=256, nodata=np.nan,
        ) as ds:
            ds.write(data_chw.astype(np.float32))
            for i, name in enumerate(band_descriptions, start=1):
                ds.set_band_description(i, name)
        size = os.path.getsize(local_path)
        tf.io.gfile.copy(local_path, uri, overwrite=True)
        return size
    finally:
        try:
            os.remove(local_path)
        except OSError:
            pass


def _per_band_mae(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Mean absolute error per band, ignoring NaN cells."""
    mask = np.isfinite(pred) & np.isfinite(truth)
    bands = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NBR", "NDWI"]
    out = {}
    for i, name in enumerate(bands):
        valid = mask[i]
        if not valid.any():
            out[name] = float("nan")
            continue
        diff = np.abs(pred[i][valid] - truth[i][valid])
        out[name] = float(diff.mean())
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project_id = os.environ["GEE_S1S2_PROJECT_ID"]
    gcs_bucket = os.environ["GEE_S1S2_BUCKET"]
    gcs_prefix = os.environ.get("GEE_S1S2_PREFIX", "gee_s1s2_translator/operational_v1")
    run_name = os.environ.get("GEE_S1S2_TRAINING_RUN_NAME", "v2_equivalent_initial")

    base = f"gs://{gcs_bucket}/{gcs_prefix}"
    model_prefix = f"{base}/models/{run_name}"
    s1_stats_uri = f"{base}/s1_stats.json"
    pred_root = f"{model_prefix}/predictions"

    LOG.info("project=%s bucket=%s run=%s", project_id, gcs_bucket, run_name)

    # ---- Load models ----
    LOG.info("Loading S1 stats from %s", s1_stats_uri)
    with tf.io.gfile.GFile(s1_stats_uri, "r") as fh:
        sd = json.load(fh)
    stats = S1Stats(mean=sd["mean"], std=sd["std"])

    def _load_full(uri: str, builder) -> tf.keras.Model:
        # Workaround for tf-gpu.2-15 + .keras format on gs:// (same as trainer.py):
        # download to local temp, build the architecture, load weights.
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            local = tmp.name
        try:
            tf.io.gfile.copy(uri, local, overwrite=True)
            model = builder()
            model.load_weights(local)
            return model
        finally:
            try:
                os.remove(local)
            except OSError:
                pass

    LOG.info("Loading U-Net from %s/unet.keras", model_prefix)
    unet = _load_full(f"{model_prefix}/unet.keras",
                      lambda: build_unet(input_shape=(256, 256, 2),
                                         out_channels=6, base_channels=32))
    LOG.info("Loading linear baseline from %s/linear_baseline.keras", model_prefix)
    lb = _load_full(f"{model_prefix}/linear_baseline.keras",
                    lambda: build_linear_baseline(input_shape=(256, 256, 2),
                                                  out_channels=6))

    from google.cloud import storage
    sclient = storage.Client(project=project_id)
    bucket = sclient.bucket(gcs_bucket)

    out_band_names = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NBR", "NDWI"]
    hankley_summary = None

    targets_json = os.environ.get("GEE_S1S2_PREDICT_TARGETS_JSON")
    targets = json.loads(targets_json) if targets_json else DEFAULT_TARGETS
    LOG.info("Predicting %d targets: %s", len(targets),
             [t["out_label"] for t in targets])

    output_subdir = os.environ.get("GEE_S1S2_PREDICT_OUTPUT_SUBDIR", "").strip("/")
    if output_subdir:
        LOG.info("Output sub-directory: %s/", output_subdir)

    for tgt in targets:
        aoi = tgt["aoi_slug"]
        s1_date = tgt["s1_date"]
        label = tgt["out_label"]
        LOG.info("Target: aoi=%s s1_date=%s", aoi, s1_date)

        patches = _list_patches(bucket, aoi, s1_date)
        LOG.info("  %d patches", len(patches))
        if not patches:
            LOG.warning("  no patches for %s @ %s; skipping", aoi, s1_date)
            continue

        # Pre-load all patches + sidecars + normalize S1.
        per_patch = []  # list of (s1_norm, s2_truth, crs, transform)
        for p in patches:
            uri = f"gs://{gcs_bucket}/{p}"
            s1, s2 = _decode_patch(uri)
            s1_norm = _normalize_s1(s1, stats)
            s2_clip = np.clip(s2, 0.0, 1.0)
            crs, transform = _read_sidecar(bucket, p)
            per_patch.append((s1_norm, s2_clip, crs, transform))

        # Stack into a batch for prediction.
        s1_batch = np.stack([p[0] for p in per_patch], axis=0).astype(np.float32)
        LOG.info("  running U-Net inference on %d patches", len(per_patch))
        unet_preds = unet.predict(s1_batch, batch_size=8, verbose=0)
        LOG.info("  running linear-baseline inference")
        lb_preds = lb.predict(s1_batch, batch_size=8, verbose=0)

        # Mosaic each model's predictions into AOI-level GeoTIFF (9 bands).
        for model_label, preds in [("unet", unet_preds), ("linear_baseline", lb_preds)]:
            patch_data = []
            for i, (_s1, _s2, crs, transform) in enumerate(per_patch):
                pred_chw = np.transpose(preds[i], (2, 0, 1))  # H,W,C -> C,H,W
                full = _compute_indices(pred_chw)
                patch_data.append((full, crs, transform))
            mosaic, mtrans, mcrs = _mosaic_patches(patch_data)
            uri = (f"{pred_root}/{model_label}/{output_subdir}/{label}_predicted_s2.tif"
                   if output_subdir
                   else f"{pred_root}/{model_label}/{label}_predicted_s2.tif")
            size = _write_geotiff(uri, mosaic, mtrans, mcrs, out_band_names)
            LOG.info("  wrote %s (%d B)", uri, size)

        # Hankley: also write truth and compute per-band MAE.
        if tgt["save_truth"]:
            patch_data = []
            for _s1, s2_truth, crs, transform in per_patch:
                truth_chw = np.transpose(s2_truth, (2, 0, 1))  # H,W,C -> C,H,W
                full_truth = _compute_indices(truth_chw)
                patch_data.append((full_truth, crs, transform))
            truth_mosaic, t_trans, t_crs = _mosaic_patches(patch_data)
            truth_uri = (f"{pred_root}/truth/{output_subdir}/{label}_truth_s2.tif"
                         if output_subdir
                         else f"{pred_root}/truth/{label}_truth_s2.tif")
            tsize = _write_geotiff(truth_uri, truth_mosaic, t_trans, t_crs, out_band_names)
            LOG.info("  wrote %s (%d B)", truth_uri, tsize)

            # Compute MAE between U-Net prediction mosaic and truth mosaic.
            unet_patch_data = []
            for i, (_s1, _s2, crs, transform) in enumerate(per_patch):
                pred_chw = np.transpose(unet_preds[i], (2, 0, 1))
                unet_patch_data.append((_compute_indices(pred_chw), crs, transform))
            unet_mosaic, _, _ = _mosaic_patches(unet_patch_data)

            lb_patch_data = []
            for i, (_s1, _s2, crs, transform) in enumerate(per_patch):
                pred_chw = np.transpose(lb_preds[i], (2, 0, 1))
                lb_patch_data.append((_compute_indices(pred_chw), crs, transform))
            lb_mosaic, _, _ = _mosaic_patches(lb_patch_data)

            hankley_summary = {
                "aoi": aoi,
                "s1_date": s1_date,
                "n_patches": len(per_patch),
                "unet_per_band_mae": _per_band_mae(unet_mosaic, truth_mosaic),
                "linear_baseline_per_band_mae": _per_band_mae(lb_mosaic, truth_mosaic),
            }

    if hankley_summary is not None:
        summary_uri = f"{pred_root}/hankley_sanity_mae.json"
        with tf.io.gfile.GFile(summary_uri, "w") as fh:
            json.dump(hankley_summary, fh, indent=2)
        LOG.info("wrote %s", summary_uri)
        LOG.info("Hankley U-Net per-band MAE: %s",
                 json.dumps(hankley_summary["unet_per_band_mae"], indent=2))
        LOG.info("Hankley linear-baseline per-band MAE: %s",
                 json.dumps(hankley_summary["linear_baseline_per_band_mae"], indent=2))

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
