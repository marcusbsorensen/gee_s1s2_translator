"""Figure 3 — The shape of the prediction across regions.

Three rows × two columns: truth + predicted RGB at three sites:
  Row 1: Hankley Common 20-May-2024
  Row 2: Cavenham Heath 26-Jun-2024
  Row 3: Berwyn SSSI 02-Jun-2024

All RGB composites with identical stretch [0, 0.3]. Per-row caption
includes B08 MAE and driver-band variance retention computed on the
fly from jointly-valid pixels.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path(__file__).parent
WP = WORK.parent.parent

# (short site label, longer caption, truth path, predicted path, optional brightness note)
ROWS = [
    ("Hankley Common", "20 May 2024  ·  in-region training site (held-out test-split OOD)",
     WP / "gee_s1s2_translator" / "training" / "calibration_v4_work" / "scenes" / "hankley-common_20240520_truth.tif",
     WP / "gee_s1s2_translator" / "training" / "calibration_v4_work" / "scenes" / "hankley-common_20240520_predicted.tif",
     "predicted appears slightly darker than truth at the shared [0, 0.3] stretch because\n"
     "the bright-pixel high tail (p98) is compressed; band medians match within ±5 %"),
    ("Cavenham Heath", "26 Jun 2024  ·  Suffolk Brecks  ·  ~100 km OOD",
     WP / "gee_s1s2_translator" / "training" / "calibration_v4_work" / "scenes" / "cavenham-heath_20240626_truth.tif",
     WP / "gee_s1s2_translator" / "training" / "calibration_v4_work" / "scenes" / "cavenham-heath_20240626_predicted.tif",
     None),
    ("Berwyn SSSI", "02 Jun 2024  ·  north Wales upland heath  ·  ~250 km OOD",
     WP / "gee_s1s2_translator" / "training" / "welsh_ood_work" / "berwyn_truth.tif",
     WP / "gee_s1s2_translator" / "training" / "welsh_ood_work" / "berwyn_predicted.tif",
     None),
]

BAND_IDX = {"B02": 1, "B03": 2, "B04": 3, "B08": 4, "B11": 5, "B12": 6}
DRIVERS = ["B04", "B08", "B11", "B12"]


def stretch_fixed(arr, lo=0.0, hi=0.3):
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def metrics(truth_path, pred_path):
    with rasterio.open(truth_path) as t, rasterio.open(pred_path) as p:
        # take the band-1..6 reflectance regardless of any extra index bands
        truth = np.stack([t.read(BAND_IDX[b]) for b in ["B02","B03","B04","B08","B11","B12"]]).astype(np.float32)
        pred  = np.stack([p.read(BAND_IDX[b]) for b in ["B02","B03","B04","B08","B11","B12"]]).astype(np.float32)
    valid = (np.isfinite(truth).all(axis=0) & np.isfinite(pred).all(axis=0)
             & (truth.sum(axis=0) > 0) & (pred.sum(axis=0) > 0))
    n_valid = int(valid.sum())
    res = {}
    for j, b in enumerate(["B02","B03","B04","B08","B11","B12"]):
        tt = truth[j][valid]; pp = pred[j][valid]
        res[b] = {
            "mae": float(np.mean(np.abs(pp - tt))),
            "std_ratio_pct": float(100.0 * np.std(pp) / (np.std(tt) + 1e-9)),
        }
    driver_mean_var = sum(res[b]["std_ratio_pct"] for b in DRIVERS) / len(DRIVERS)
    return n_valid, res, driver_mean_var


def render_rgb(path):
    with rasterio.open(path) as ds:
        b02 = ds.read(BAND_IDX["B02"]).astype(np.float32)
        b03 = ds.read(BAND_IDX["B03"]).astype(np.float32)
        b04 = ds.read(BAND_IDX["B04"]).astype(np.float32)
    valid = np.isfinite(b02) & np.isfinite(b03) & np.isfinite(b04) & ((b02 + b03 + b04) > 0)
    rgb = np.dstack([stretch_fixed(b04), stretch_fixed(b03), stretch_fixed(b02)])
    rgb[~valid] = 1.0  # white background outside coverage
    return rgb


def main():
    # Wider left margin (0.20) gives enough rotated-text room for
    # "Hankley Common" / "Cavenham Heath" / "Berwyn SSSI" + a second-line
    # description; the bright-tail caption goes BELOW each row instead of
    # to the side.
    fig, axes = plt.subplots(3, 2, figsize=(11, 13.5),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.40,
                                          "left": 0.18, "right": 0.99,
                                          "top": 0.92, "bottom": 0.03})
    for row_idx, (short_label, sub_label, truth_p, pred_p, note) in enumerate(ROWS):
        n, m, dvar = metrics(truth_p, pred_p)
        b08_mae = m["B08"]["mae"]
        rgb_t = render_rgb(truth_p)
        rgb_p = render_rgb(pred_p)
        axes[row_idx, 0].imshow(rgb_t)
        axes[row_idx, 0].set_title("truth Sentinel-2", fontsize=10)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(rgb_p)
        axes[row_idx, 1].set_title(
            f"predicted (baseline + v3 calibration)\n"
            f"B08 MAE {b08_mae:.3f}  ·  driver var retention {dvar:.0f}%",
            fontsize=10,
        )
        axes[row_idx, 1].axis("off")

        # Row label sits in the dedicated label column (x ∈ [0, 0.18])
        bbox = axes[row_idx, 0].get_position()
        cy = (bbox.y0 + bbox.y1) / 2.0
        # Bigger, bolder site name; smaller subtitle below
        fig.text(0.085, cy + 0.025, short_label,
                 rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
        fig.text(0.13, cy, sub_label,
                 rotation=90, va="center", ha="center", fontsize=9)

        # Optional explanatory note BELOW the row (e.g. Hankley brightness)
        if note is not None:
            fig.text((bbox.x0 + axes[row_idx, 1].get_position().x1) / 2.0,
                     bbox.y0 - 0.025, note,
                     ha="center", va="top", fontsize=8, style="italic", color="#444")

        print(f"row {row_idx+1} {short_label} — {sub_label}")
        print(f"  n_valid={n:,}  B08 MAE={b08_mae:.4f}  driver var={dvar:.1f}%")
    fig.suptitle(
        "Figure 3 — The shape of the prediction across regions\n"
        "RGB composites at fixed [0, 0.3] reflectance stretch for honest cross-site comparison",
        fontsize=12,
    )
    out = WORK / "figure_3_shape_across_regions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
