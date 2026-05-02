"""Per-band metrics and the variance-collapse diagnostic.

Two metric families:

1. **Per-band MAE / RMSE** — straightforward; computed in numpy from
   the (n, H, W, C) prediction / truth tensors.
2. **Variance-collapse diagnostic** — v2's central quality check,
   computed against **patch-specific truth std** (the v2 refinement
   over the earlier project-wide reference). For each test patch and
   each S2 band, compute ``pred_std / truth_std`` (both per-patch
   spatial std). Aggregate across patches by mean. Pass criterion is
   the [75 %, 105 %] bracket on driver bands B04, B08, B11, B12 (the
   bands that drive NDVI / NBR / NDWI). Below 75 % = the model has
   spatially smeared predictions; above 105 % = noisy. Both modes
   break downstream NBR thresholding.

The driver-band set and the bracket match v2 exactly so the GEE-port
result is comparable line-for-line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

S2_BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12"]
DRIVER_BANDS = ["B04", "B08", "B11", "B12"]
VARIANCE_BRACKET_LOW = 0.75
VARIANCE_BRACKET_HIGH = 1.05


@dataclass
class PerBandMetric:
    band: str
    mae: float
    rmse: float


def per_band_mae_rmse(
    y_true: np.ndarray,             # (N, H, W, C)
    y_pred: np.ndarray,             # (N, H, W, C)
    band_order: list[str] = S2_BAND_ORDER,
) -> list[PerBandMetric]:
    """Per-band mean absolute and root-mean-squared error across (N, H, W)."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: truth={y_true.shape} pred={y_pred.shape}")
    out: list[PerBandMetric] = []
    for i, band in enumerate(band_order):
        diff = y_pred[..., i] - y_true[..., i]
        finite = np.isfinite(diff)
        mae = float(np.abs(diff[finite]).mean()) if finite.any() else float("nan")
        rmse = float(np.sqrt(np.square(diff[finite]).mean())) if finite.any() else float("nan")
        out.append(PerBandMetric(band=band, mae=mae, rmse=rmse))
    return out


@dataclass
class VarianceRetention:
    band: str
    is_driver: bool
    mean_pred_over_truth_pct: float        # mean across patches of pred_std / truth_std, in %
    median_pred_over_truth_pct: float
    pass_75_105_bracket: bool


def patch_specific_variance_retention(
    y_true: np.ndarray,             # (N, H, W, C)
    y_pred: np.ndarray,
    band_order: list[str] = S2_BAND_ORDER,
    driver_bands: list[str] = DRIVER_BANDS,
) -> list[VarianceRetention]:
    """For each band: compute pred_std/truth_std per patch, aggregate.

    Returns one :class:`VarianceRetention` row per band. Pass criterion
    is the mean ratio inside [75 %, 105 %] on driver bands only — non-driver
    bands are reported but not gated.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: truth={y_true.shape} pred={y_pred.shape}")
    n_patches = y_true.shape[0]
    out: list[VarianceRetention] = []
    for i, band in enumerate(band_order):
        truth = y_true[..., i].reshape(n_patches, -1)
        pred = y_pred[..., i].reshape(n_patches, -1)
        truth_std = truth.std(axis=1)
        pred_std = pred.std(axis=1)
        # Ratio undefined on patches where truth has zero variance; drop those.
        valid = truth_std > 1e-9
        if not valid.any():
            out.append(VarianceRetention(
                band=band, is_driver=band in driver_bands,
                mean_pred_over_truth_pct=float("nan"),
                median_pred_over_truth_pct=float("nan"),
                pass_75_105_bracket=False,
            ))
            continue
        ratios = pred_std[valid] / truth_std[valid]
        mean_pct = float(ratios.mean()) * 100
        median_pct = float(np.median(ratios)) * 100
        is_driver = band in driver_bands
        bracket_pass = (
            VARIANCE_BRACKET_LOW * 100 <= mean_pct <= VARIANCE_BRACKET_HIGH * 100
        ) if is_driver else True
        out.append(VarianceRetention(
            band=band, is_driver=is_driver,
            mean_pred_over_truth_pct=mean_pct,
            median_pred_over_truth_pct=median_pct,
            pass_75_105_bracket=bracket_pass,
        ))
    return out


def driver_band_mean_retention_pct(rows: list[VarianceRetention]) -> float:
    """Mean of the driver-band mean-pct values; the headline number."""
    drivers = [r.mean_pred_over_truth_pct for r in rows if r.is_driver]
    if not drivers:
        return float("nan")
    return float(np.mean(drivers))
