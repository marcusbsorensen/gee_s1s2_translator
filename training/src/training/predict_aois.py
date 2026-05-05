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


def _load_postfit_calibration() -> Tuple[np.ndarray, np.ndarray, str] | None:
    """Load the per-band affine calibration shipped with the package, if
    present. Returns (slope, intercept, source_label) where slope/intercept
    are length-6 arrays in S2_BANDS order, or None if no calibration file
    is bundled. Apply at inference as ``y_corrected = slope * y + intercept``
    on the 6 reflectance bands; recompute indices from the corrected bands.

    Loader preference: v3 (multi-scene Huber fit on May-Aug 2024 paired
    scenes across 4 sites) wins; v1 (single-scene LSQ fit on Cavenham
    only) is left in the package as a historical artefact and only used
    if v3 is missing. v3 has tightly-bounded slopes [0.74, 1.09] and
    near-zero intercepts; v1 collapsed to constant on B11/B12 due to the
    single-scene over-fit and is not recommended.
    """
    cal_dir = os.path.join(os.path.dirname(__file__), "calibration")
    candidates = ["postfit_affine_v3.json", "postfit_affine_v1.json"]
    for name in candidates:
        cal_path = os.path.join(cal_dir, name)
        if not os.path.exists(cal_path):
            continue
        try:
            with open(cal_path) as f:
                doc = json.load(f)
        except Exception as e:
            LOG.warning("Could not read calibration %s: %s", cal_path, e)
            continue
        # Schema: v3 keeps coefficients under "calibration"; v1 used "bands".
        bands_doc = doc.get("calibration") or doc.get("bands") or {}
        if not all(b in bands_doc for b in S2_BANDS):
            LOG.warning("Calibration %s missing one of %s; trying next.",
                        cal_path, S2_BANDS)
            continue
        slopes = np.array([bands_doc[b]["slope"] for b in S2_BANDS], dtype=np.float32)
        intercepts = np.array([bands_doc[b]["intercept"] for b in S2_BANDS], dtype=np.float32)
        source = doc.get("fit_source", name)
        LOG.info("Using calibration %s (%s)", name, source)
        return slopes, intercepts, source
    return None


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


def _cosine_taper_window(h: int, w: int, margin: int = 32) -> np.ndarray:
    """Build a 2D cosine-tapered window of shape (h, w) with a Hann ramp
    of width ``margin`` at each edge and weight 1.0 in the centre.

    Used by :func:`_mosaic_patches` to feather patch predictions at their
    edges, so that overlapping patches are blended (weighted average)
    rather than producing the hard seams that ``rasterio.merge`` (which
    is "first non-NaN wins") leaves at every patch boundary.
    """
    def _axis(n: int) -> np.ndarray:
        x = np.ones(n, dtype=np.float32)
        m = min(margin, n // 2)
        # Left edge: 0 at i=0, 1 at i=m
        for i in range(m):
            x[i] = 0.5 * (1 - np.cos(np.pi * i / m))
        # Right edge: mirror of the left edge
        for i in range(m):
            x[n - 1 - i] = 0.5 * (1 - np.cos(np.pi * i / m))
        return x
    wy = _axis(h)
    wx = _axis(w)
    return np.outer(wy, wx).astype(np.float32)


def _mosaic_patches(
    patch_data: List[Tuple[np.ndarray, str, Affine]],
    taper_margin: int = 32,
) -> Tuple[np.ndarray, Affine, str]:
    """Cosine-blended mosaic of per-patch (C,H,W) arrays into a single
    CHW raster.

    For each patch we accumulate ``prediction * window`` into a
    weighted-sum raster and ``window`` into a sum-of-weights raster at
    the union extent. The final output is ``weighted_sum / weight_sum``,
    NaN where no patch covered the pixel. This replaces the prior
    ``rasterio.merge`` (method="first") implementation that produced
    visible seams at patch boundaries (12-15x background-gradient,
    measured on the Brentmoor + Poors post-fire 2022 outputs).

    All patches must share the same CRS and pixel resolution.
    """
    if not patch_data:
        raise ValueError("patch_data is empty")
    crs = patch_data[0][1]
    res_x = patch_data[0][2].a
    res_y = -patch_data[0][2].e  # rasterio's e is negative for north-up rasters

    # Union extent in CRS coordinates.
    minx = min(t[2].c for t in patch_data)
    maxy = max(t[2].f for t in patch_data)
    maxx = max(t[2].c + t[0].shape[2] * res_x for t in patch_data)
    miny = min(t[2].f - t[0].shape[1] * res_y for t in patch_data)
    union_w = int(round((maxx - minx) / res_x))
    union_h = int(round((maxy - miny) / res_y))
    union_transform = Affine(res_x, 0.0, minx, 0.0, -res_y, maxy)

    n_channels = patch_data[0][0].shape[0]
    weighted_sum = np.zeros((n_channels, union_h, union_w), dtype=np.float32)
    weight_sum = np.zeros((union_h, union_w), dtype=np.float32)

    for arr, patch_crs, transform in patch_data:
        assert patch_crs == crs, f"CRS mismatch: {patch_crs} != {crs}"
        c, h, w = arr.shape
        col_off = int(round((transform.c - minx) / res_x))
        row_off = int(round((maxy - transform.f) / res_y))

        win = _cosine_taper_window(h, w, margin=taper_margin)
        # Zero out the window where any band is NaN so we don't pull
        # nodata into the weighted sum.
        finite_all = np.isfinite(arr).all(axis=0)
        eff_win = win * finite_all.astype(np.float32)

        # Replace NaN with 0 in arr for the multiplication; the eff_win
        # mask already excludes those pixels from weight accumulation.
        arr_clean = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)

        weighted_sum[:, row_off:row_off + h, col_off:col_off + w] += (
            arr_clean * eff_win[None, :, :]
        )
        weight_sum[row_off:row_off + h, col_off:col_off + w] += eff_win

    out = np.full_like(weighted_sum, np.nan)
    valid = weight_sum > 0
    for ci in range(n_channels):
        ch = out[ci]
        ws = weighted_sum[ci]
        ch[valid] = ws[valid] / weight_sum[valid]
    return out, union_transform, crs


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


def _write_preview_png(
    mosaic: np.ndarray,
    transform: Affine,
    crs: str,
    out_uri: str,
    title: str = "",
) -> None:
    """Render a 3-panel preview PNG (RGB / false-colour / NBR) and upload to ``out_uri``.

    Panels:
      1. RGB composite (B04 R, B03 G, B02 B), explicit stretch [0, 0.3].
      2. False-colour (B08 R, B04 G, B03 B), explicit stretch [0, 0.5].
      3. NBR (band 8 of the mosaic), diverging RdYlGn_r, stretch [-0.5, 0.5].

    Layout is side-by-side at 150 dpi. Coordinates use the mosaic's
    transform so axes are CRS-true.
    """
    # Lazy import so a matplotlib failure doesn't abort GeoTIFF writes.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if mosaic.shape[0] < 9:
        raise ValueError(f"preview expects 9-band mosaic, got {mosaic.shape[0]}")
    b02, b03, b04, b08, _b11, _b12 = mosaic[0], mosaic[1], mosaic[2], mosaic[3], mosaic[4], mosaic[5]
    nbr = mosaic[7]

    h, w = b04.shape
    extent = (transform.c, transform.c + w * transform.a,
              transform.f + h * transform.e, transform.f)  # left, right, bottom, top
    valid = np.isfinite(b04)

    def _stretch(arr: np.ndarray, vmax: float, gamma: float = 0.7) -> np.ndarray:
        return np.clip(arr / vmax, 0.0, 1.0) ** gamma

    rgb = np.dstack([_stretch(b04, 0.3), _stretch(b03, 0.3), _stretch(b02, 0.3)])
    fc = np.dstack([_stretch(b08, 0.5), _stretch(b04, 0.5), _stretch(b03, 0.5)])
    rgb[~valid] = 1.0
    fc[~valid] = 1.0
    nbr_for_plot = np.where(valid, nbr, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=150)
    axes[0].imshow(rgb, extent=extent, origin="upper")
    axes[0].set_title("RGB (B04/B03/B02), stretch [0, 0.3]", fontsize=10)
    axes[1].imshow(fc, extent=extent, origin="upper")
    axes[1].set_title("False-colour (B08/B04/B03), stretch [0, 0.5]", fontsize=10)
    im = axes[2].imshow(nbr_for_plot, extent=extent, origin="upper",
                        cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
    axes[2].set_title("NBR (B08-B12)/(B08+B12)", fontsize=10)
    fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.04, shrink=0.85)
    for ax in axes:
        ax.set_xlabel("easting (m)")
        ax.set_ylabel("northing (m)")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_aspect("equal")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    with tempfile.NamedTemporaryFile(suffix="_preview.png", delete=False) as tmp:
        local_path = tmp.name
    try:
        fig.savefig(local_path, bbox_inches="tight")
        plt.close(fig)
        tf.io.gfile.copy(local_path, out_uri, overwrite=True)
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

    # Phase B was trained with output_activation="linear" (sigmoid removed,
    # predictions clipped at inference). The default training run is
    # sigmoid-output. Pick the right architecture before load_weights or
    # the loaded weights will saturate through an unintended sigmoid.
    output_activation = os.environ.get("GEE_S1S2_OUTPUT_ACTIVATION", "sigmoid").strip().lower() or "sigmoid"
    if output_activation not in ("sigmoid", "linear"):
        raise RuntimeError(f"Unsupported GEE_S1S2_OUTPUT_ACTIVATION={output_activation!r}")
    LOG.info("Loading U-Net from %s/unet.keras (output_activation=%s)",
             model_prefix, output_activation)
    unet = _load_full(f"{model_prefix}/unet.keras",
                      lambda: build_unet(input_shape=(256, 256, 2),
                                         out_channels=6, base_channels=32,
                                         output_activation=output_activation))
    # Linear baseline is optional. The Phase B v2 / B v3 runs don't fit a
    # linear baseline, so loading it would fail on those checkpoints. Try
    # to load and downgrade to None if the file isn't there.
    lb_uri = f"{model_prefix}/linear_baseline.keras"
    try:
        if tf.io.gfile.exists(lb_uri):
            LOG.info("Loading linear baseline from %s", lb_uri)
            lb = _load_full(lb_uri,
                            lambda: build_linear_baseline(input_shape=(256, 256, 2),
                                                          out_channels=6))
        else:
            LOG.info("No linear_baseline.keras under %s; skipping linear baseline.",
                     model_prefix)
            lb = None
    except Exception as e:
        LOG.warning("Could not load linear baseline from %s (%s); skipping.", lb_uri, e)
        lb = None

    # Optional per-band affine calibration (only applied to U-Net predictions;
    # the linear baseline has fundamentally different output statistics so the
    # U-Net-fitted calibration doesn't transfer).
    # Bypass via GEE_S1S2_DISABLE_POSTFIT_CALIBRATION=true — used during the
    # calibration-set inference pass so the fit isn't trained on its own output.
    disable_cal = os.environ.get("GEE_S1S2_DISABLE_POSTFIT_CALIBRATION", "").strip().lower() in {"true", "1", "yes"}
    if disable_cal:
        LOG.info("GEE_S1S2_DISABLE_POSTFIT_CALIBRATION=true; running uncalibrated.")
        calibration = None
    else:
        calibration = _load_postfit_calibration()
    if calibration is not None:
        cal_slope, cal_intercept, cal_source = calibration
        LOG.info("Loaded post-fit affine calibration from %s; will apply to U-Net.",
                 cal_source)
        for i, b in enumerate(S2_BANDS):
            LOG.info("  %s: slope=%.4f intercept=%.5f", b, cal_slope[i], cal_intercept[i])
    elif not disable_cal:
        LOG.info("No post-fit affine calibration found; U-Net output used as-is.")

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
        if lb is not None:
            LOG.info("  running linear-baseline inference")
            lb_preds = lb.predict(s1_batch, batch_size=8, verbose=0)
        else:
            lb_preds = None
        # Phase B was trained with linear output; clip to operational [0,1]
        # reflectance range here so downstream calibration + index recompute
        # see in-range data. No-op for sigmoid-output runs.
        if output_activation == "linear":
            unet_preds = np.clip(unet_preds, 0.0, 1.0).astype(np.float32)

        # Mosaic each model's predictions into AOI-level GeoTIFF (9 bands).
        mosaic_pairs = [("unet", unet_preds)]
        if lb_preds is not None:
            mosaic_pairs.append(("linear_baseline", lb_preds))
        for model_label, preds in mosaic_pairs:
            patch_data = []
            for i, (_s1, _s2, crs, transform) in enumerate(per_patch):
                pred_chw = np.transpose(preds[i], (2, 0, 1))  # H,W,C -> C,H,W
                # Apply per-band affine calibration to U-Net reflectance only.
                # The calibration was fit on Cavenham U-Net predicted-vs-truth
                # and corrects systematic dark bias + variance compression on
                # visible bands; applying to the linear baseline (different
                # output statistics) would be incorrect, so we skip it there.
                if model_label == "unet" and calibration is not None:
                    pred_chw = pred_chw * cal_slope[:, None, None] + cal_intercept[:, None, None]
                    pred_chw = np.clip(pred_chw, 0.0, 1.0).astype(np.float32)
                full = _compute_indices(pred_chw)
                patch_data.append((full, crs, transform))
            mosaic, mtrans, mcrs = _mosaic_patches(patch_data)
            uri = (f"{pred_root}/{model_label}/{output_subdir}/{label}_predicted_s2.tif"
                   if output_subdir
                   else f"{pred_root}/{model_label}/{label}_predicted_s2.tif")
            size = _write_geotiff(uri, mosaic, mtrans, mcrs, out_band_names)
            LOG.info("  wrote %s (%d B)", uri, size)
            # Sidecar PNG preview (RGB + false-colour + NBR) for fast visual QA.
            try:
                preview_uri = uri.replace(".tif", "_preview.png")
                _write_preview_png(mosaic, mtrans, mcrs, preview_uri,
                                   title=f"{label} ({model_label})")
                LOG.info("  wrote %s", preview_uri)
            except Exception as e:
                LOG.warning("  preview render failed for %s: %s", uri, e)

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
            try:
                _write_preview_png(truth_mosaic, t_trans, t_crs,
                                   truth_uri.replace(".tif", "_preview.png"),
                                   title=f"{label} (truth)")
            except Exception as e:
                LOG.warning("  truth preview render failed: %s", e)

            # Compute MAE between U-Net prediction mosaic and truth mosaic.
            # Apply the same per-band calibration we shipped on the saved
            # GeoTIFF so the MAE compares the corrected prediction to truth.
            unet_patch_data = []
            for i, (_s1, _s2, crs, transform) in enumerate(per_patch):
                pred_chw = np.transpose(unet_preds[i], (2, 0, 1))
                if calibration is not None:
                    pred_chw = pred_chw * cal_slope[:, None, None] + cal_intercept[:, None, None]
                    pred_chw = np.clip(pred_chw, 0.0, 1.0).astype(np.float32)
                unet_patch_data.append((_compute_indices(pred_chw), crs, transform))
            unet_mosaic, _, _ = _mosaic_patches(unet_patch_data)

            lb_mae = None
            if lb_preds is not None:
                lb_patch_data = []
                for i, (_s1, _s2, crs, transform) in enumerate(per_patch):
                    pred_chw = np.transpose(lb_preds[i], (2, 0, 1))
                    lb_patch_data.append((_compute_indices(pred_chw), crs, transform))
                lb_mosaic, _, _ = _mosaic_patches(lb_patch_data)
                lb_mae = _per_band_mae(lb_mosaic, truth_mosaic)

            hankley_summary = {
                "aoi": aoi,
                "s1_date": s1_date,
                "n_patches": len(per_patch),
                "unet_per_band_mae": _per_band_mae(unet_mosaic, truth_mosaic),
            }
            if lb_mae is not None:
                hankley_summary["linear_baseline_per_band_mae"] = lb_mae

    if hankley_summary is not None:
        summary_uri = f"{pred_root}/hankley_sanity_mae.json"
        with tf.io.gfile.GFile(summary_uri, "w") as fh:
            json.dump(hankley_summary, fh, indent=2)
        LOG.info("wrote %s", summary_uri)
        LOG.info("Hankley U-Net per-band MAE: %s",
                 json.dumps(hankley_summary["unet_per_band_mae"], indent=2))
        if "linear_baseline_per_band_mae" in hankley_summary:
            LOG.info("Hankley linear-baseline per-band MAE: %s",
                     json.dumps(hankley_summary["linear_baseline_per_band_mae"], indent=2))

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
