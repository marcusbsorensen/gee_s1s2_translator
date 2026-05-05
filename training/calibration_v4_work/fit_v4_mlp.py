"""Scene-conditional MLP calibration — leave-one-scene-out cross-validation.

Day 3 morning. Compares a small sklearn MLP, conditioned on (predicted
reflectance + month + lat + lon + region one-hot), against the v3 global
affine on the same 6 calibration scenes. Outputs per-scene per-band
median ratio, std ratio, and MAE for both calibrators.

Run with the project venv that has rasterio + sklearn (1.8+).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rasterio
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

WORK = Path(__file__).parent
SCENES_DIR = WORK / "scenes"
RESULTS_DIR = WORK / "results"
RESULTS_DIR.mkdir(exist_ok=True)

V3_PATH = WORK.parent / "src" / "training" / "calibration" / "postfit_affine_v3.json"

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# Scene metadata: AOI lat/lon (from operational_v1.yaml) and region.
SCENE_META = {
    "cavenham-heath_20240520":               {"lat": 52.291, "lon":  0.587, "region": "Suffolk"},
    "cavenham-heath_20240626":               {"lat": 52.291, "lon":  0.587, "region": "Suffolk"},
    "hankley-common_20240520":               {"lat": 51.180, "lon": -0.730, "region": "Surrey"},
    "hankley-common_20240729":               {"lat": 51.180, "lon": -0.730, "region": "Surrey"},
    "beaulieu-heath_20240719":               {"lat": 50.815, "lon": -1.450, "region": "New Forest"},
    "studland-and-godlingston-heath_20240719": {"lat": 50.655, "lon": -1.961, "region": "Dorset"},
}
REGIONS = ["Surrey", "New Forest", "Dorset", "Suffolk", "Sussex"]

# Feature normalisation constants (fixed so we can apply identically at
# inference). Latitude span 50.0..53.0, longitude span -2.0..1.0.
LAT_LO, LAT_HI = 50.0, 53.0
LON_LO, LON_HI = -2.0, 1.0


def load_scene(slug_date: str):
    pred_path = SCENES_DIR / f"{slug_date}_predicted.tif"
    truth_path = SCENES_DIR / f"{slug_date}_truth.tif"
    with rasterio.open(pred_path) as ds:
        pred = ds.read().astype(np.float32)  # (6, H, W)
    with rasterio.open(truth_path) as ds:
        truth = ds.read().astype(np.float32)
    assert pred.shape == truth.shape, f"shape mismatch: {pred.shape} vs {truth.shape}"
    # mask invalid pixels in either image (NaN, or all-zero)
    mask_pred = np.isfinite(pred).all(axis=0) & (pred.sum(axis=0) > 0)
    mask_truth = np.isfinite(truth).all(axis=0) & (truth.sum(axis=0) > 0)
    mask = mask_pred & mask_truth
    pred_pix = pred[:, mask].T   # (n_pixels, 6)
    truth_pix = truth[:, mask].T
    return pred_pix, truth_pix


def build_features(pred_pix, slug_date):
    """Build (n_pixels, n_features) feature matrix.

    Features: 6 predicted reflectance bands, month_norm, lat_norm,
    lon_norm, 5 region one-hot.
    """
    meta = SCENE_META[slug_date]
    date = slug_date.split("_")[-1]
    month = int(date[4:6])
    month_norm = (month - 1) / 11.0
    lat_norm = (meta["lat"] - LAT_LO) / (LAT_HI - LAT_LO)
    lon_norm = (meta["lon"] - LON_LO) / (LON_HI - LON_LO)
    region_idx = REGIONS.index(meta["region"])
    region_onehot = np.zeros(len(REGIONS), dtype=np.float32)
    region_onehot[region_idx] = 1.0

    n = pred_pix.shape[0]
    extra = np.tile(
        np.concatenate([[month_norm, lat_norm, lon_norm], region_onehot]).astype(np.float32),
        (n, 1),
    )
    return np.hstack([pred_pix, extra])


def apply_v3_affine(pred_pix, v3):
    """Apply per-band slope+intercept to predicted reflectance."""
    out = np.empty_like(pred_pix)
    for j, b in enumerate(S2_BANDS):
        s = v3["calibration"][b]["slope"]
        c = v3["calibration"][b]["intercept"]
        out[:, j] = s * pred_pix[:, j] + c
    return out


def per_band_metrics(corrected, truth):
    """Per-band (median ratio, std ratio, MAE)."""
    metrics = {}
    eps = 1e-9
    for j, b in enumerate(S2_BANDS):
        c = corrected[:, j]
        t = truth[:, j]
        med_t = np.median(t); std_t = np.std(t); mae = float(np.mean(np.abs(c - t)))
        metrics[b] = {
            "med_ratio_pct": float(100.0 * np.median(c) / (med_t + eps)),
            "std_ratio_pct": float(100.0 * np.std(c) / (std_t + eps)),
            "mae": mae,
            "med_truth": float(med_t),
            "std_truth": float(std_t),
        }
    return metrics


def fit_mlp(X_train, Y_train, hidden=(64, 32), random_state=0):
    scaler_X = StandardScaler().fit(X_train)
    Xs = scaler_X.transform(X_train)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu",
        solver="adam", learning_rate_init=1e-3, max_iter=200,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
        random_state=random_state, verbose=False,
    )
    mlp.fit(Xs, Y_train)
    return mlp, scaler_X


def main(hidden=(64, 32), max_pix_per_scene=80_000, seed=0):
    rng = np.random.default_rng(seed)
    with open(V3_PATH) as f:
        v3 = json.load(f)

    scenes = list(SCENE_META.keys())
    cache = {}  # slug_date -> (pred_pix, truth_pix, X_features)
    for sd in scenes:
        pred_pix, truth_pix = load_scene(sd)
        X = build_features(pred_pix, sd)
        # subsample uniformly for fitting
        if pred_pix.shape[0] > max_pix_per_scene:
            idx = rng.choice(pred_pix.shape[0], size=max_pix_per_scene, replace=False)
            pred_pix_sub = pred_pix[idx]; truth_pix_sub = truth_pix[idx]; X_sub = X[idx]
        else:
            pred_pix_sub = pred_pix; truth_pix_sub = truth_pix; X_sub = X
        cache[sd] = {
            "pred_full": pred_pix, "truth_full": truth_pix,
            "X_full": X, "pred_sub": pred_pix_sub, "truth_sub": truth_pix_sub,
            "X_sub": X_sub,
        }
        print(f"  loaded {sd}: {pred_pix.shape[0]} pixels (fit subset {pred_pix_sub.shape[0]})")

    # LOSO
    loso_results = {}
    for held in scenes:
        train_scenes = [s for s in scenes if s != held]
        X_train = np.vstack([cache[s]["X_sub"] for s in train_scenes])
        Y_train = np.vstack([cache[s]["truth_sub"] for s in train_scenes])

        mlp, scaler = fit_mlp(X_train, Y_train, hidden=hidden, random_state=seed)
        # apply on FULL held-out scene
        X_held = cache[held]["X_full"]
        pred_held = cache[held]["pred_full"]
        truth_held = cache[held]["truth_full"]

        Xs_held = scaler.transform(X_held)
        mlp_corr = mlp.predict(Xs_held).astype(np.float32)
        # clip to valid reflectance range
        mlp_corr = np.clip(mlp_corr, 0.0, 1.0)

        v3_corr = apply_v3_affine(pred_held, v3)
        v3_corr = np.clip(v3_corr, 0.0, 1.0)

        loso_results[held] = {
            "mlp": per_band_metrics(mlp_corr, truth_held),
            "v3":  per_band_metrics(v3_corr, truth_held),
            "raw": per_band_metrics(pred_held, truth_held),
            "n_held_pixels": int(pred_held.shape[0]),
            "n_train_pixels": int(X_train.shape[0]),
            "mlp_loss": float(mlp.loss_) if hasattr(mlp, "loss_") else None,
            "mlp_n_iter": int(mlp.n_iter_),
        }
        print(f"  LOSO held={held}: trained on {X_train.shape[0]} pixels, mlp loss={mlp.loss_:.5f} after {mlp.n_iter_} iter")

    out = {
        "config": {
            "hidden_layers": list(hidden),
            "max_pix_per_scene_for_fit": max_pix_per_scene,
            "regions": REGIONS,
            "lat_norm_range": [LAT_LO, LAT_HI],
            "lon_norm_range": [LON_LO, LON_HI],
            "feature_order": [f"pred_{b}" for b in S2_BANDS] + ["month_norm","lat_norm","lon_norm"] + [f"region_{r}" for r in REGIONS],
            "n_features": 6 + 3 + len(REGIONS),
            "seed": seed,
        },
        "scenes": list(scenes),
        "loso": loso_results,
    }
    with open(RESULTS_DIR / "loso_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {RESULTS_DIR/'loso_results.json'}")
    return out


if __name__ == "__main__":
    main()
