"""Figure 1 — The shape of a fire over time.

Three-panel composite of dNBR over Brentmoor Heath:
  Panel A: real S2 prefire − real S2 +5d post-fire (14 Aug 2022)
  Panel B: real S2 prefire − real S2 +70d post-fire (18 Oct 2022)
  Panel C: real S2 prefire − predicted S2 +60d (8 Oct 2022)

Same colour ramp (RdYlGn_r), same stretch [-0.5, 0.5], same projection,
same extent. SWT field-mapped fire perimeter overlaid in red.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET

WORK = Path(__file__).parent
WP = WORK.parent.parent
KML = WP / "s1s2-translator" / "inputs" / "SWT_MappedFires_20220911.kml"

PREFIRE = WORK / "brentmoor_truth_prefire_20220426.tif"
POST5D  = WORK / "brentmoor_truth_post5d_20220814.tif"
POST70D = WORK / "brentmoor_truth_post70d_20221018.tif"
PREDICTED_60D = WP / "sonia_predictions_v070" / "brentmoor-area-training_20221008_postfire_predicted_s2.tif"


def parse_swt_kml_brentmoor():
    """Return the Brentmoor Heath polygon in EPSG:4326 (lon, lat ring)."""
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(KML)
    root = tree.getroot()
    for pm in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        # find ExtendedData with Site=Brentmoor Heath
        site = None
        for sd in pm.iter("{http://www.opengis.net/kml/2.2}SimpleData"):
            if sd.attrib.get("name") == "Site":
                site = sd.text
        if site != "Brentmoor Heath":
            continue
        coords_el = pm.find(".//k:coordinates", ns)
        coords = coords_el.text.strip().split()
        ring = []
        for c in coords:
            parts = c.split(",")
            lon, lat = float(parts[0]), float(parts[1])
            ring.append((lon, lat))
        return ring
    raise RuntimeError("Brentmoor Heath polygon not found in KML")


def read_band_aligned_to(ref_path, src_path, band_indices):
    """Read selected bands from src, cropped/resampled to match ref's grid."""
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        # If grids differ, use a window read aligned to ref bounds
        if (ref.crs == src.crs and abs(ref.transform.a - src.transform.a) < 1e-6
                and ref.bounds == src.bounds and ref.width == src.width and ref.height == src.height):
            return np.stack([src.read(b) for b in band_indices]).astype(np.float32)
        # Otherwise window-read the overlap and resample if needed
        win = from_bounds(*ref.bounds, transform=src.transform)
        data = src.read(
            indexes=band_indices,
            window=win,
            out_shape=(len(band_indices), ref.height, ref.width),
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype(np.float32)
        return data


def nbr_from(ref_path, src_path, b08_idx, b12_idx):
    """Compute NBR aligned to ref grid, masking pixels where either band <=0 or NaN."""
    arr = read_band_aligned_to(ref_path, src_path, [b08_idx, b12_idx])
    b08, b12 = arr[0], arr[1]
    mask = (b08 > 0) & (b12 > 0) & np.isfinite(b08) & np.isfinite(b12)
    nbr = np.where(mask, (b08 - b12) / (b08 + b12 + 1e-9), np.nan)
    return nbr, mask


def main():
    # The fetched truth tiffs have 6 bands B02/B03/B04/B08/B11/B12 (band 4 = B08, band 6 = B12).
    # The predicted tif has 9 bands (6 reflectance + NDVI/NBR/NDWI); band 4 = B08, band 6 = B12.
    REF = PREFIRE  # use the pre-fire truth as reference grid

    nbr_pre, m_pre  = nbr_from(REF, PREFIRE,  b08_idx=4, b12_idx=6)
    nbr_5d,  m_5d   = nbr_from(REF, POST5D,   b08_idx=4, b12_idx=6)
    nbr_70d, m_70d  = nbr_from(REF, POST70D,  b08_idx=4, b12_idx=6)
    nbr_pred,m_pred = nbr_from(REF, PREDICTED_60D, b08_idx=4, b12_idx=6)

    dnbr_5d  = np.where(m_pre & m_5d,  nbr_pre - nbr_5d,  np.nan)
    dnbr_70d = np.where(m_pre & m_70d, nbr_pre - nbr_70d, np.nan)
    dnbr_60d = np.where(m_pre & m_pred,nbr_pre - nbr_pred,np.nan)

    # SWT perimeter (lon/lat ring) → reproject to ref CRS for overlay
    ring_lonlat = parse_swt_kml_brentmoor()
    import pyproj
    with rasterio.open(REF) as ds:
        ref_crs = ds.crs
        ref_transform = ds.transform
        ref_h, ref_w = ds.height, ds.width
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
    ring_utm = [to_utm.transform(lon, lat) for lon, lat in ring_lonlat]
    # Convert UTM coords to pixel coords on ref grid
    inv = ~ref_transform
    ring_px = np.array([inv * (x, y) for x, y in ring_utm])

    # Crop a 1.5 km × 1.5 km window centred on the SWT perimeter centroid
    # so the small fire perimeter is visible. ref grid is 10 m/px → 150 px.
    cx_px = float(np.mean(ring_px[:, 0]))
    cy_px = float(np.mean(ring_px[:, 1]))
    half = 75  # 75 * 10 m = 750 m → 1.5 km square
    x0 = max(0, int(cx_px - half)); x1 = min(ref_w, int(cx_px + half))
    y0 = max(0, int(cy_px - half)); y1 = min(ref_h, int(cy_px + half))
    print(f"crop: x[{x0}:{x1}] y[{y0}:{y1}] ({x1-x0}x{y1-y0} px)")

    def crop(a):
        return a[y0:y1, x0:x1]
    ring_local = ring_px - np.array([[x0, y0]])

    # Render
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    cm = "RdYlGn_r"
    titles = [
        "A — real dNBR at +5 days\npre-fire 26-Apr − real S2 14-Aug-2022\n(14-Aug acquisition has 31 % scene cloud cover)",
        "B — real dNBR at +70 days\npre-fire 26-Apr − real S2 18-Oct-2022\n(18-Oct cloud-free)",
        "C — hybrid dNBR at +60 days\npre-fire 26-Apr − predicted S2 08-Oct-2022\n(predicted from cloud-covered S1)",
    ]
    for ax, dnbr, title in zip(axes, [dnbr_5d, dnbr_70d, dnbr_60d], titles):
        im = ax.imshow(crop(dnbr), cmap=cm, vmin=-0.5, vmax=0.5)
        ax.plot(ring_local[:, 0], ring_local[:, 1], color="red", linewidth=2.2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        # Add scale bar (200 m at 10 m/px = 20 px)
        ax.plot([10, 30], [y1-y0-15, y1-y0-15], color="black", linewidth=2)
        ax.text(20, y1-y0-22, "200 m", color="black", ha="center", fontsize=8)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="dNBR")
    fig.suptitle("Figure 1 — The shape of a fire over time, Brentmoor Heath 2022\n(SWT field-mapped perimeter in red, ~0.33 ha)",
                 fontsize=11, y=1.04)
    out = WORK.parent.parent / "figures_work" / "fig1" / "figure_1_shape_over_time_brentmoor.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Headline stats per panel within the SWT polygon vs outside
    from shapely.geometry import Polygon, Point
    poly_utm = Polygon(ring_utm)

    def stats(dnbr, label):
        finite = np.isfinite(dnbr)
        med = float(np.nanmedian(dnbr[finite])) if finite.any() else float("nan")
        # mean inside perimeter (rough mask via pixel-grid fill)
        # Use rasterio.features.rasterize for accuracy
        from rasterio.features import rasterize
        mask = rasterize(
            [(poly_utm, 1)], out_shape=(ref_h, ref_w),
            transform=ref_transform, fill=0, dtype="uint8",
        )
        inside = (mask == 1) & finite
        med_in = float(np.nanmedian(dnbr[inside])) if inside.any() else float("nan")
        outside = (mask == 0) & finite
        med_out = float(np.nanmedian(dnbr[outside])) if outside.any() else float("nan")
        print(f"  [{label}]  median (full) {med:+.3f}  inside perimeter {med_in:+.3f}  outside {med_out:+.3f}  (n_in={int(inside.sum())})")

    print("\nPer-panel dNBR stats:")
    stats(dnbr_5d,  "A real +5d")
    stats(dnbr_70d, "B real +70d")
    stats(dnbr_60d, "C hybrid +60d")


if __name__ == "__main__":
    main()
