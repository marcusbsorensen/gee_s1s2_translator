"""Figure 4 — The shape of capacity evolution.

Brentmoor 18-Oct 2022 RGB rendered three times across pipeline versions:
  Panel A: v0.5.x — mosaic-merged, no post-fit calibration
  Panel B: v0.6.0 — cosine-blended mosaic, v1 single-scene affine
  Panel C: v0.7.0 — cosine-blended mosaic, v3 multi-scene Huber affine

Identical RGB stretch [0, 0.3], identical band mapping. Same site,
same date, same model checkpoint underneath.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = Path(__file__).parent
WP = WORK.parent.parent

PANELS = [
    ("A — v0.5.x  ·  mosaic-merged, no post-fit calibration",
     WP / "sonia_predictions" / "brentmoor-area-training_20221018_postfire_predicted_s2.tif"),
    ("B — v0.6.0  ·  cosine-blended mosaic + v1 single-scene affine",
     WP / "sonia_predictions_v060" / "brentmoor-area-training_20221018_postfire_predicted_s2.tif"),
    ("C — v0.7.0  ·  cosine-blended mosaic + v3 multi-scene Huber affine",
     WP / "sonia_predictions_v070" / "brentmoor-area-training_20221018_postfire_predicted_s2.tif"),
]

BAND_IDX = {"B02": 1, "B03": 2, "B04": 3}


def stretch_fixed(arr, lo=0.0, hi=0.3):
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def rgb(path):
    with rasterio.open(path) as ds:
        b02 = ds.read(BAND_IDX["B02"]).astype(np.float32)
        b03 = ds.read(BAND_IDX["B03"]).astype(np.float32)
        b04 = ds.read(BAND_IDX["B04"]).astype(np.float32)
    valid = np.isfinite(b02) & np.isfinite(b03) & np.isfinite(b04) & ((b02 + b03 + b04) > 0)
    img = np.dstack([stretch_fixed(b04), stretch_fixed(b03), stretch_fixed(b02)])
    img[~valid] = 1.0
    return img, (b02.shape[1], b02.shape[0])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6),
                             gridspec_kw={"wspace": 0.04, "left": 0.02, "right": 0.99,
                                          "top": 0.85, "bottom": 0.04})
    for ax, (caption, path) in zip(axes, PANELS):
        if not path.exists():
            ax.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off"); continue
        img, _ = rgb(path)
        ax.imshow(img)
        ax.set_title(caption, fontsize=10)
        ax.axis("off")
        print(f"rendered {path.name}: shape={img.shape}")
    fig.suptitle(
        "Figure 4 — The shape of capacity evolution (Brentmoor 18-Oct-2022)\n"
        "Same scene, same checkpoint; only the post-processing pipeline changes",
        fontsize=12, y=0.96,
    )
    out = WORK / "figure_4_capacity_evolution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
