"""Figure 5 — The shape of variance retention across improvement attempts.

Five panels of Cavenham Heath 26-Jun-2024 RGB at fixed [0, 0.3] stretch:
  Panel A: truth Sentinel-2
  Panel B: baseline (v2_equivalent_initial) + v3 calibration
  Panel C: Phase B v2
  Panel D: Phase B v3
  Panel E: multi-temporal v1
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google.cloud import storage

WORK = Path(__file__).parent
WP = WORK.parent.parent
BUCKET = "marcus-heath-fire-mapping"

# Fetch from GCS (or local cache)
PANELS = [
    ("A — truth Sentinel-2",
     None,
     "gee_s1s2_translator/operational_v1/models/v2_equivalent_initial/predictions/truth/worked_example/cavenham-heath_20240626_truth_s2.tif",
     None),
    ("B — baseline + v3 calibration",
     "v2_equivalent_initial",
     "gee_s1s2_translator/operational_v1/models/v2_equivalent_initial/predictions/unet/worked_example/cavenham-heath_20240626_predicted_s2.tif",
     None),
    ("C — Phase B v2 (variance loss)",
     "phase_b_v2_variance_active",
     "gee_s1s2_translator/operational_v1/models/phase_b_v2_variance_active/predictions/unet/phase_b_v2_worked_example/cavenham-heath_20240626_predicted_s2.tif",
     None),
    ("D — Phase B v3 (band-weighted)",
     "phase_b_v3_band_weighted",
     "gee_s1s2_translator/operational_v1/models/phase_b_v3_band_weighted/predictions/unet/phase_b_v3_worked_example/cavenham-heath_20240626_predicted_s2.tif",
     None),
    ("E — Multi-temporal v1",
     "multitemporal_v1_t4",
     None,  # Local, not GCS
     str(WORK / "cavenham_mt_predicted_s2.tif")),
]

BAND_IDX = {"B02": 1, "B03": 2, "B04": 3, "B08": 4, "B11": 5, "B12": 6}
DRIVERS = ["B04", "B08", "B11", "B12"]


def stretch_fixed(arr, lo=0.0, hi=0.3):
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def stretch_adaptive(arr, valid_mask, p_lo: float = 2.0, p_hi: float = 98.0):
    """Per-array p2-p98 stretch over jointly-valid pixels.

    Returns (stretched_array, lo, hi) so the caller can put the actual
    reflectance bracket in the panel caption.
    """
    v = arr[valid_mask]
    if v.size == 0:
        return np.zeros_like(arr), 0.0, 1.0
    lo = float(np.percentile(v, p_lo))
    hi = float(np.percentile(v, p_hi))
    if hi <= lo:
        return np.clip(arr, 0, 1), lo, max(lo + 1e-6, hi)
    return np.clip((arr - lo) / (hi - lo), 0, 1), lo, hi


def load_local(path: str) -> str:
    return path


def load_from_gcs(client, bucket, gcs_path: str, suffix: str = "") -> str:
    """Cache locally with a suffix so files at different GCS paths but the
    same basename do not collide (baseline / B v2 / B v3 / truth all use
    'cavenham-heath_20240626_predicted_s2.tif' as their basename)."""
    name = Path(gcs_path).name
    local = WORK / (suffix + "__" + name if suffix else name)
    if not local.exists():
        bucket.blob(gcs_path).download_to_filename(local)
    return str(local)


def driver_var_retention(truth_path: str, pred_path: str) -> float | None:
    """Compute driver-band mean std-ratio over jointly-valid pixels at the
    intersection of truth and pred grids."""
    if pred_path is None or truth_path is None:
        return None
    with rasterio.open(truth_path) as t, rasterio.open(pred_path) as p:
        if t.bounds == p.bounds and t.width == p.width and t.height == p.height:
            T = np.stack([t.read(BAND_IDX[b]) for b in S2_BANDS]).astype(np.float32)
            P = np.stack([p.read(BAND_IDX[b]) for b in S2_BANDS]).astype(np.float32)
        else:
            # window-read pred at truth bounds
            T = np.stack([t.read(BAND_IDX[b]) for b in S2_BANDS]).astype(np.float32)
            win = from_bounds(*t.bounds, transform=p.transform)
            P = np.stack([p.read(BAND_IDX[b], window=win,
                                 out_shape=(t.height, t.width),
                                 resampling=rasterio.enums.Resampling.bilinear) for b in S2_BANDS]).astype(np.float32)
    valid = (np.isfinite(T).all(axis=0) & np.isfinite(P).all(axis=0)
             & (T.sum(axis=0) > 0) & (P.sum(axis=0) > 0))
    rs = []
    for b in DRIVERS:
        j = S2_BANDS.index(b)
        tt = T[j][valid]; pp = P[j][valid]
        rs.append(100.0 * float(np.std(pp)) / (float(np.std(tt)) + 1e-9))
    return sum(rs) / len(rs)


def render_rgb(path):
    """Identical [0, 0.3] stretch — for the cross-variant comparison row."""
    with rasterio.open(path) as ds:
        b02 = ds.read(BAND_IDX["B02"]).astype(np.float32)
        b03 = ds.read(BAND_IDX["B03"]).astype(np.float32)
        b04 = ds.read(BAND_IDX["B04"]).astype(np.float32)
    valid = np.isfinite(b02) & np.isfinite(b03) & np.isfinite(b04) & ((b02+b03+b04) > 0)
    rgb = np.dstack([stretch_fixed(b04), stretch_fixed(b03), stretch_fixed(b02)])
    rgb[~valid] = 1.0
    return rgb


def render_rgb_adaptive(path):
    """Per-panel p2-p98 stretch — for the variant-fairness row.

    Per-band stretch: each band gets its own [p2, p98] computed over
    valid pixels. This reveals spatial structure within each band's
    actual distribution — important for Phase B v3, whose RGB-band
    means are clustered around 0.4-0.55 but each band still has
    spatial variance around its own mean. A joint stretch would
    collapse that structure into uniform colour because B04's mean
    sits near the joint p98 ceiling.

    Multi-temporal v1's outlier pixels render as bright tints under
    per-band stretching (because outliers in different bands occur
    at different pixel locations); the caption notes this honestly
    and points the viewer to row 1 for the unfiltered view.

    Returns (rgb, lo, hi) where [lo, hi] is the joint min/max of the
    three per-band brackets — a single bracket the caption can show.
    """
    with rasterio.open(path) as ds:
        b02 = ds.read(BAND_IDX["B02"]).astype(np.float32)
        b03 = ds.read(BAND_IDX["B03"]).astype(np.float32)
        b04 = ds.read(BAND_IDX["B04"]).astype(np.float32)
    valid = np.isfinite(b02) & np.isfinite(b03) & np.isfinite(b04) & ((b02+b03+b04) > 0)
    s02, lo02, hi02 = stretch_adaptive(b02, valid)
    s03, lo03, hi03 = stretch_adaptive(b03, valid)
    s04, lo04, hi04 = stretch_adaptive(b04, valid)
    rgb = np.dstack([s04, s03, s02])
    rgb[~valid] = 1.0
    lo = min(lo02, lo03, lo04)
    hi = max(hi02, hi03, hi04)
    return rgb, lo, hi


S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def main():
    client = storage.Client(project="wildfire-495012")
    bucket = client.bucket(BUCKET)
    # Resolve all paths. Suffix downloads with the run name to avoid
    # local-cache filename collisions across variants.
    truth_path = None
    resolved = []
    for cap, run, gcs, local in PANELS:
        if local:
            p = local
            assert os.path.exists(p), f"missing local: {p}"
        else:
            suffix = run if run else "truth"
            p = load_from_gcs(client, bucket, gcs, suffix=suffix)
        resolved.append((cap, run, p))
        if cap.startswith("A —"):
            truth_path = p

    # Per-panel captions for both rows
    row1_captions = []  # cross-variant comparison [0, 0.3]
    row2_captions = []  # variant fairness, p2-p98
    retention_by_run: dict = {}
    for cap, run, p in resolved:
        letter = cap.split(" — ")[0]
        short = cap.split(" — ")[1].split(" (")[0] if " — " in cap else cap
        if cap.startswith("A —"):
            row1_captions.append("A — truth Sentinel-2\n(reference reflectance)")
        else:
            r = driver_var_retention(truth_path, p)
            retention_by_run[run] = r
            row1_captions.append(f"{letter} — {short}\ndriver var retention {r:.0f}%")
            print(f"{cap}: driver var retention = {r:.1f}%")

    # Row 2 stretch brackets — pre-compute and build captions
    row2_brackets = {}
    for cap, run, p in resolved:
        _, lo, hi = render_rgb_adaptive(p)
        row2_brackets[cap] = (lo, hi)
        letter = cap.split(" — ")[0]
        short = cap.split(" — ")[1].split(" (")[0] if " — " in cap else cap
        if cap.startswith("A —"):
            row2_captions.append(f"A — truth Sentinel-2\nstretch [{lo:.2f}, {hi:.2f}]")
        elif run == "multitemporal_v1_t4":
            row2_captions.append(
                f"{letter} — {short}\nstretch [{lo:.2f}, {hi:.2f}] — extreme pixel outliers clipped\n(visible in row 1)")
        else:
            row2_captions.append(f"{letter} — {short}\nstretch [{lo:.2f}, {hi:.2f}]")

    fig, axes = plt.subplots(2, 5, figsize=(22, 12.0),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.30,
                                          "left": 0.01, "right": 0.99,
                                          "top": 0.91, "bottom": 0.13})

    # Row 1: identical [0, 0.3] stretch
    for ax, (cap, run, path), title in zip(axes[0], resolved, row1_captions):
        ax.imshow(render_rgb(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Row 2: per-panel p2-p98 stretch
    for ax, (cap, run, path), title in zip(axes[1], resolved, row2_captions):
        rgb, _, _ = render_rgb_adaptive(path)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Row headers (positioned just above each row's panels)
    fig.text(0.5, 0.93, "Identical [0, 0.3] stretch — cross-variant comparison",
             ha="center", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.485, "Per-panel p2 to p98 stretch — each variant on its own terms",
             ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Figure 5 — The shape of variance retention across improvement attempts (Cavenham 26-Jun-2024)",
        fontsize=13, y=0.99,
    )
    fig.text(0.5, 0.954,
             "Top row: identical [0, 0.3] stretch reveals where each variant's distribution lands relative to truth's (the cross-variant fairness view). "
             "Bottom row: per-panel adaptive stretch reveals what each variant actually produces when rendered on its own terms (the variant-fairness view). "
             "Both views are honest; they answer different questions.",
             ha="center", fontsize=9.5, style="italic", color="#333")

    bottom_caption = (
        "The four interventions fail at variance retention in three distinct ways. "
        "Variance collapse (baseline + v3, Phase B v2) produces smooth low-contrast outputs that respect truth's mean but compress its variance. "
        "Off-distribution mean (Phase B v3) shifts the predicted reflectance regime above the [0, 0.3] stretch ceiling because v3 calibration was fit on baseline outputs and does not transfer;\n"
        "row 2 reveals that Phase B v3 does produce structured spatial output, just at a different mean. "
        "Variance overshoot with outliers (multi-temporal v1) produces extreme pixel values that survive calibration; row 2 reveals that the underlying spatial structure is plausible when outliers are clipped. "
        "None of the four interventions matches truth's joint distribution of mean and variance,\n"
        "supporting the dissertation's central methodological finding that variance retention is bounded by dataset scale at this data volume."
    )
    fig.text(0.5, 0.012, bottom_caption,
             ha="center", va="bottom", fontsize=9, style="italic", color="#333")

    out = WORK / "figure_5_variance_retention_attempts.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")
    print(f"\nrow 2 brackets:")
    for cap, (lo, hi) in row2_brackets.items():
        print(f"  {cap}: [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
