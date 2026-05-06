"""Fetch real Sentinel-2 surface-reflectance GeoTIFFs from GEE for the
Brentmoor area-training AOI at three dates (pre-fire 2022, +5d post,
+70d post), in the same CRS / bounds / resolution as the existing
Brentmoor predictions, so dNBR computations are on the same grid.

Output: 6-band float32 GeoTIFFs (B02, B03, B04, B08, B11, B12) in
[0, 1] reflectance, masked at SCL cloud/shadow classes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import ee
import requests
import rasterio
from rasterio.transform import from_bounds

WORK = Path(__file__).parent
FOOTPRINT = json.loads((WORK / "brentmoor_footprint.json").read_text())

EPSG = FOOTPRINT["crs_epsg"]
LEFT, BOTTOM, RIGHT, TOP = FOOTPRINT["bounds"]
WIDTH, HEIGHT = FOOTPRINT["width"], FOOTPRINT["height"]
RES_X, RES_Y = FOOTPRINT["res"]

# Target dates and labels. Tuple is (iso, label, days_window, sort_strategy).
# sort_strategy "cloud" sorts by ascending CLOUDY_PIXEL_PERCENTAGE (least
# cloudy first); "exact" uses date proximity then cloud cover; "mosaic"
# mosaics ALL images on the same target date to fill cross-tile gaps
# (the Brentmoor AOI crosses the T30UXB/T30UXC seam).
TARGETS = [
    ("2022-04-26", "prefire", 21, "cloud"),
    ("2022-08-14", "post5d", 1, "mosaic"),
    ("2022-10-18", "post70d", 1, "mosaic"),
]

S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
OUT_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12"]

# SCL valid classes (vegetation, bare soil, water, snow, unclassified — exclude clouds/shadows/cirrus)
SCL_VALID = [4, 5, 6, 7, 11]  # 4 veg, 5 bare, 6 water, 7 unclassified, 11 snow


def main():
    ee.Initialize(project="wildfire-495012")

    # Convert UTM bbox to WGS84 corners for the AOI ee.Geometry, then ask
    # for the download in EPSG:32630 to keep the grid aligned with the
    # existing predictions.
    import pyproj
    to_wgs = pyproj.Transformer.from_crs(
        f"EPSG:{EPSG}", "EPSG:4326", always_xy=True,
    )
    ll = to_wgs.transform(LEFT, BOTTOM)
    ur = to_wgs.transform(RIGHT, TOP)
    aoi = ee.Geometry.Rectangle(
        [ll[0], ll[1], ur[0], ur[1]],
        proj="EPSG:4326", evenOdd=True,
    )

    for target_iso, label, days, strategy in TARGETS:
        out = WORK / f"brentmoor_truth_{label}_{target_iso.replace('-', '')}.tif"
        if out.exists():
            print(f"  skip {out.name} (exists)")
            continue
        target = ee.Date(target_iso)
        start = target.advance(-days, "day")
        end   = target.advance(+days, "day")

        coll = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(start, end)
                .filterBounds(aoi)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80)))
        n = coll.size().getInfo()
        print(f"\n[{label}] {target_iso} ±{days}d ({strategy}) -> {n} S2 candidates")
        if n == 0:
            print(f"  WARNING no candidates; skip {out.name}")
            continue

        if strategy == "mosaic":
            # Build a single image from all images on the target date
            # (mosaicked across tiles). SCL-mask each before mosaicking
            # so cloud-shadow artefacts are excluded from each tile.
            SCL_VALID_LIST = [4, 5, 6, 7, 11]
            def _scl_mask(img):
                scl = img.select("SCL")
                v = scl.remap(SCL_VALID_LIST, [1] * len(SCL_VALID_LIST), 0)
                return img.updateMask(v)
            coll_masked = coll.map(_scl_mask)
            img = coll_masked.mosaic()
            print(f"  mosaic of {n} images")
            ids = coll.aggregate_array("system:index").getInfo()
            print(f"  image ids: {ids}")
        elif strategy == "exact":
            target_ms = target.millis()
            def _stamp(img):
                t = ee.Number(img.get("system:time_start"))
                d = t.subtract(target_ms).abs()
                return img.set("days_from_target_ms", d)
            coll = coll.map(_stamp).sort("days_from_target_ms")
            img = ee.Image(coll.first())
        else:
            coll = coll.sort("CLOUDY_PIXEL_PERCENTAGE")
            img = ee.Image(coll.first())
        if strategy != "mosaic":
            info = img.getInfo()
            sys_id = info.get("id", "?")
            cloud_pct = info.get("properties", {}).get("CLOUDY_PIXEL_PERCENTAGE", "?")
            print(f"  picked {sys_id} (CLOUDY_PIXEL_PERCENTAGE={cloud_pct})")
            scl = img.select("SCL")
            valid = scl.remap(SCL_VALID, [1] * len(SCL_VALID), 0).rename("valid")
            refl = img.select(S2_BANDS, OUT_NAMES).divide(10000.0).updateMask(valid)
        else:
            # Mosaic was already SCL-masked tile-by-tile above.
            refl = img.select(S2_BANDS, OUT_NAMES).divide(10000.0)

        # Download URL: ask for the exact CRS/region/scale matching the prediction footprint
        url = refl.getDownloadURL({
            "scale": RES_X,
            "crs": f"EPSG:{EPSG}",
            "region": aoi,
            "format": "GEO_TIFF",
        })
        print(f"  url len={len(url)}")
        # Stream the TIF
        for attempt in range(3):
            try:
                r = requests.get(url, stream=True, timeout=120)
                if r.status_code == 200:
                    with open(out, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            f.write(chunk)
                    print(f"  wrote {out.name} ({out.stat().st_size:,} bytes)")
                    break
                else:
                    print(f"  HTTP {r.status_code}; retry {attempt}")
                    time.sleep(2)
            except Exception as e:
                print(f"  exception {e}; retry {attempt}")
                time.sleep(2)
        else:
            print(f"  FAILED to download {out.name}")
            continue

        # Sanity check shape
        with rasterio.open(out) as ds:
            print(f"  shape: ({ds.count}, {ds.height}, {ds.width}) crs={ds.crs} res={ds.res}")


if __name__ == "__main__":
    main()
