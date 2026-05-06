"""Figure 2 — The shape of fire at scale.

Side-by-side hybrid dNBR for Brentmoor (~0.33 ha SWT perimeter) and
Poors Allotment (~6.79 ha SWT perimeter), both at +60 days post-fire
(predicted S2 from S1 at 08 October 2022) against their respective
pre-fire S2 baselines.
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
import pyproj
import ee
import requests
from shapely.geometry import Polygon

WORK = Path(__file__).parent
WP = WORK.parent.parent
KML = WP / "s1s2-translator" / "inputs" / "SWT_MappedFires_20220911.kml"

PRED_BRENT = WP / "sonia_predictions_v070" / "brentmoor-area-training_20221008_postfire_predicted_s2.tif"
PRED_POORS = WP / "sonia_predictions_v070" / "poors-allotment-area-training_20221008_postfire_predicted_s2.tif"
PREFIRE_BRENT = WP / "figures_work" / "fig1" / "brentmoor_truth_prefire_20220426.tif"


def parse_swt_kml(site_name):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(KML)
    root = tree.getroot()
    for pm in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        site = None
        for sd in pm.iter("{http://www.opengis.net/kml/2.2}SimpleData"):
            if sd.attrib.get("name") == "Site":
                site = sd.text
        if site != site_name:
            continue
        coords = pm.find(".//k:coordinates", ns).text.strip().split()
        return [tuple(map(float, c.split(",")[:2])) for c in coords]
    raise RuntimeError(f"polygon not found: {site_name}")


def fetch_s2_truth_for(pred_path, target_iso, days, out_path):
    """Fetch real S2 over the same footprint as pred_path (date-proximity)."""
    if out_path.exists():
        print(f"  skip {out_path.name} (exists)")
        return out_path
    with rasterio.open(pred_path) as ds:
        epsg = ds.crs.to_epsg()
        L, B, R, T = ds.bounds
        res_x, _ = ds.res
    to_wgs = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    ll = to_wgs.transform(L, B); ur = to_wgs.transform(R, T)
    aoi = ee.Geometry.Rectangle([ll[0], ll[1], ur[0], ur[1]], proj="EPSG:4326", evenOdd=True)

    target = ee.Date(target_iso); start = target.advance(-days, "day"); end = target.advance(days, "day")
    coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start, end).filterBounds(aoi)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    n = coll.size().getInfo()
    print(f"  {target_iso} ±{days}d -> {n} candidates")
    img = ee.Image(coll.sort("CLOUDY_PIXEL_PERCENTAGE").first())
    info = img.getInfo()
    print(f"  picked {info.get('id')} (CLOUDY={info.get('properties',{}).get('CLOUDY_PIXEL_PERCENTAGE')})")
    SCL_VALID = [4, 5, 6, 7, 11]
    scl = img.select("SCL")
    valid = scl.remap(SCL_VALID, [1] * len(SCL_VALID), 0)
    refl = (img.select(["B2","B3","B4","B8","B11","B12"], ["B02","B03","B04","B08","B11","B12"])
            .divide(10000.0).updateMask(valid))
    url = refl.getDownloadURL({
        "scale": res_x, "crs": f"EPSG:{epsg}", "region": aoi, "format": "GEO_TIFF",
    })
    r = requests.get(url, timeout=180); r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"  wrote {out_path.name} ({out_path.stat().st_size:,} bytes)")
    return out_path


def read_aligned(ref_path, src_path, band_indices):
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        if (ref.crs == src.crs and ref.bounds == src.bounds and ref.width == src.width and ref.height == src.height):
            return np.stack([src.read(b) for b in band_indices]).astype(np.float32)
        win = from_bounds(*ref.bounds, transform=src.transform)
        return src.read(
            indexes=band_indices, window=win,
            out_shape=(len(band_indices), ref.height, ref.width),
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype(np.float32)


def nbr_aligned(ref_path, src_path, b08_idx, b12_idx):
    arr = read_aligned(ref_path, src_path, [b08_idx, b12_idx])
    b08, b12 = arr[0], arr[1]
    mask = (b08 > 0) & (b12 > 0) & np.isfinite(b08) & np.isfinite(b12)
    return np.where(mask, (b08 - b12) / (b08 + b12 + 1e-9), np.nan), mask


def build_panel(ref_path, prefire_path, pred_path, ring_lonlat):
    nbr_pre, m_pre = nbr_aligned(ref_path, prefire_path, b08_idx=4, b12_idx=6)
    nbr_pred, m_pred = nbr_aligned(ref_path, pred_path, b08_idx=4, b12_idx=6)
    dnbr = np.where(m_pre & m_pred, nbr_pre - nbr_pred, np.nan)
    with rasterio.open(ref_path) as ds:
        ref_crs = ds.crs; ref_transform = ds.transform; ref_h, ref_w = ds.height, ds.width
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", ref_crs, always_xy=True)
    ring_utm = [to_utm.transform(lon, lat) for lon, lat in ring_lonlat]
    inv = ~ref_transform
    ring_px = np.array([inv * (x, y) for x, y in ring_utm])
    poly_utm = Polygon(ring_utm)
    mask_inside = rasterize([(poly_utm, 1)], out_shape=(ref_h, ref_w),
                            transform=ref_transform, fill=0, dtype="uint8")
    return dnbr, ring_px, mask_inside, ref_transform, (ref_h, ref_w)


def main():
    ee.Initialize(project="wildfire-495012")

    # Get real pre-fire S2 for Poors (Brentmoor we already have)
    prefire_poors = WORK / "poors_truth_prefire_20220710.tif"
    fetch_s2_truth_for(PRED_POORS, "2022-07-10", days=21, out_path=prefire_poors)

    ring_brent = parse_swt_kml("Brentmoor Heath")
    ring_poors = parse_swt_kml("Poors Allotment")

    # Brentmoor: ref grid = pre-fire truth (Brentmoor)
    dnbr_b, rb_px, m_in_b, t_b, (hb, wb) = build_panel(
        PREFIRE_BRENT, PREFIRE_BRENT, PRED_BRENT, ring_brent
    )
    # Poors: ref grid = poors pre-fire truth
    dnbr_p, rp_px, m_in_p, t_p, (hp, wp) = build_panel(
        prefire_poors, prefire_poors, PRED_POORS, ring_poors
    )

    # Crop each to a 1.5 km square around its perimeter centroid
    def crop(arr, ring_px, h, w, half_m=750, res=10):
        cx = float(np.mean(ring_px[:, 0])); cy = float(np.mean(ring_px[:, 1]))
        half_px = int(half_m / res)
        x0 = max(0, int(cx - half_px)); x1 = min(w, int(cx + half_px))
        y0 = max(0, int(cy - half_px)); y1 = min(h, int(cy + half_px))
        ring_local = ring_px - np.array([[x0, y0]])
        return arr[y0:y1, x0:x1], ring_local, (y1 - y0, x1 - x0)

    crop_b, ring_b_loc, sh_b = crop(dnbr_b, rb_px, hb, wb)
    crop_p, ring_p_loc, sh_p = crop(dnbr_p, rp_px, hp, wp)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    cm = "RdYlGn_r"
    titles = [
        "A — Brentmoor Heath\nhybrid dNBR at +60 days\n(SWT perimeter ~0.33 ha)",
        "B — Poors Allotment\nhybrid dNBR at +60 days\n(SWT perimeter ~6.79 ha)",
    ]
    for ax, dnbr, ring, sh, title in zip(axes, [crop_b, crop_p], [ring_b_loc, ring_p_loc], [sh_b, sh_p], titles):
        im = ax.imshow(dnbr, cmap=cm, vmin=-0.5, vmax=0.5)
        ax.plot(ring[:, 0], ring[:, 1], color="red", linewidth=2.0)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        # Scale bar 200 m = 20 px
        ax.plot([10, 30], [sh[0] - 15, sh[0] - 15], color="black", linewidth=2)
        ax.text(20, sh[0] - 22, "200 m", color="black", ha="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="dNBR")
    fig.suptitle("Figure 2 — The shape of fire at scale (08 Oct 2022 prediction)\n"
                 "Both panels: pre-fire real S2 minus predicted post-fire S2 from cloud-covered S1",
                 fontsize=11, y=1.04)
    out = WORK / "figure_2_shape_at_scale.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # Stats
    for label, dnbr_full, m_in in (
        ("Brentmoor (~0.33 ha)", dnbr_b, m_in_b),
        ("Poors    (~6.79 ha)", dnbr_p, m_in_p),
    ):
        finite = np.isfinite(dnbr_full)
        inside = (m_in == 1) & finite
        outside = (m_in == 0) & finite
        med_in = float(np.nanmedian(dnbr_full[inside])) if inside.any() else float("nan")
        med_out = float(np.nanmedian(dnbr_full[outside])) if outside.any() else float("nan")
        print(f"  {label}: median dNBR inside perimeter {med_in:+.3f}  outside {med_out:+.3f}  n_in={int(inside.sum())}")


if __name__ == "__main__":
    main()
