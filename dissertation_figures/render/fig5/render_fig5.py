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
    with rasterio.open(path) as ds:
        b02 = ds.read(BAND_IDX["B02"]).astype(np.float32)
        b03 = ds.read(BAND_IDX["B03"]).astype(np.float32)
        b04 = ds.read(BAND_IDX["B04"]).astype(np.float32)
    valid = np.isfinite(b02) & np.isfinite(b03) & np.isfinite(b04) & ((b02+b03+b04) > 0)
    rgb = np.dstack([stretch_fixed(b04), stretch_fixed(b03), stretch_fixed(b02)])
    rgb[~valid] = 1.0
    return rgb


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

    # Per-panel caption — keep short to fit small panels. Honest about
    # the visual character: baseline + B v2 land in collapsed territory;
    # B v3 saturates white (model means 0.4-0.55 land above the [0, 0.3]
    # ceiling); MT overshoots variance with extreme pixel outliers.
    captions = []
    for cap, run, p in resolved:
        if cap.startswith("A —"):
            captions.append("A — truth Sentinel-2\n(reference reflectance)")
        else:
            r = driver_var_retention(truth_path, p)
            short = cap.split(" — ")[1].split(" (")[0] if " — " in cap else cap
            letter = cap.split(" — ")[0]
            captions.append(f"{letter} — {short}\ndriver var retention {r:.0f}%")
            print(f"{cap}: driver var retention = {r:.1f}%")

    fig, axes = plt.subplots(1, 5, figsize=(22, 5.6),
                             gridspec_kw={"wspace": 0.04, "left": 0.01, "right": 0.99,
                                          "top": 0.84, "bottom": 0.10})
    for ax, (cap, run, path), title in zip(axes, resolved, captions):
        ax.imshow(render_rgb(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        "Figure 5 — The shape of variance retention across improvement attempts (Cavenham 26-Jun-2024)",
        fontsize=12, y=0.97,
    )
    fig.text(0.5, 0.04,
             "Identical RGB stretch [0, 0.3]. Truth (A) is sharp; baseline+v3 (B) and Phase B v2 (C) land in similar variance-collapsed territory; "
             "Phase B v3 (D) saturates white because its raw outputs on this OOD scene land in 0.4-0.55 reflectance (v3 calibration was fit on baseline);\n"
             "Multi-temporal v1 (E) shows variance overshooting with extreme pixel outliers. The four interventions fail at variance retention in three distinct ways.",
             ha="center", fontsize=9.5, style="italic", color="#333")
    out = WORK / "figure_5_variance_retention_attempts.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
