"""Multi-temporal Sentinel-1 harvest: re-export every patch in the
existing operational_v1 manifest with a 3-acquisition S1 stack instead
of a single S1 acquisition.

The S1 stack contains, for each manifest row:

    t_0:   the S1 acquisition the manifest already pairs with the S2 truth
    t_1w:  the most recent S1 over the same AOI in [t_0 - 14d, t_0 - 7d]
    t_3w:  the most recent S1 over the same AOI in [t_0 - 28d, t_0 - 14d]

Each TFRecord patch retains the S2 truth from the original manifest row,
plus 6 S1 channels (2 polarisations × 3 acquisitions). Origin pixel
coords match the original patch exactly so the train/val/test split
assignments transfer one-for-one.

A new manifest is written under ``multitemporal_v1/manifest.csv`` with
the additional ``s1_t1w_id``/``s1_t3w_id`` columns; rows for which one
of the priors couldn't be found are skipped (we don't pad with zeros —
that would silently degrade training).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

LOG = logging.getLogger(__name__)


@dataclass
class _PriorS1:
    image: Any           # ee.Image (calibrated)
    iso: str
    image_id: str


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def find_prior_s1(
    s1_calibrated_collection: Any,
    aoi_geom: Any,
    t0_iso: str,
    days_min: int,
    days_max: int,
    orbit_pass: str | None = None,
) -> _PriorS1 | None:
    """Find the most recent calibrated S1 image over the AOI in
    ``[t0 - days_max, t0 - days_min]``. Same orbit pass if specified.

    Returns ``None`` if no acquisition in the window meets the criteria
    (eg the AOI's S1 history doesn't go back that far).
    """
    import ee  # noqa: PLC0415
    t0 = _parse_iso(t0_iso)
    end = t0 - timedelta(days=days_min)
    start = t0 - timedelta(days=days_max)
    coll = s1_calibrated_collection.filterDate(str(start.date()), str(end.date()))
    if orbit_pass:
        coll = coll.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    coll = coll.filterBounds(aoi_geom).sort("system:time_start", False)

    info = coll.limit(1).getInfo()
    feats = info.get("features", [])
    if not feats:
        return None
    f = feats[0]
    img_id = f.get("id", "").split("/")[-1]
    ts = int(f["properties"]["system:time_start"])
    iso = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    img = coll.first()
    return _PriorS1(image=img, iso=iso, image_id=img_id)


def stack_3s1_with_s2(
    s1_t0: Any,
    s1_t1w: Any,
    s1_t3w: Any,
    s2_image: Any,
    s2_bands: list[str],
    polarisations: list[str],
) -> tuple[Any, list[str]]:
    """Build a multi-temporal stacked image and return (image, ordered_band_names).

    Output band order:
      VV_t0, VH_t0, VV_t1w, VH_t1w, VV_t3w, VH_t3w, B02..B12, [SCL]
    """
    import ee  # noqa: PLC0415

    def _rename(img: Any, suffix: str) -> Any:
        new = [f"{p}_{suffix}" for p in polarisations]
        return img.select(polarisations).rename(new)

    s1_stack = (
        _rename(s1_t0, "t0")
        .addBands(_rename(s1_t1w, "t1w"))
        .addBands(_rename(s1_t3w, "t3w"))
    )
    s2 = s2_image.select(s2_bands).toFloat().divide(10000.0)
    out = s1_stack.addBands(s2)
    out_names = (
        [f"{p}_t0" for p in polarisations]
        + [f"{p}_t1w" for p in polarisations]
        + [f"{p}_t3w" for p in polarisations]
        + s2_bands
    )
    return out, out_names


def patch_geom_from_origin(
    aoi_geom: Any,
    origin_lon: float,
    origin_lat: float,
    origin_utm_x_m: float,
    origin_utm_y_m: float,
    origin_utm_epsg: int,
    patch_size_pixels: int,
    resolution_metres: float,
) -> Any:
    """Re-construct the patch's UTM rectangle from the manifest row's
    stored origin metadata.

    The harvest writes (origin_utm_x_m, origin_utm_y_m, origin_utm_epsg)
    into each manifest row at top-left corner of the patch, plus
    (origin_lon, origin_lat) for cross-checking. We rebuild the
    ee.Geometry.Rectangle in the same UTM CRS so the new 3-S1-stack
    export lands at exactly the same pixel coordinates as the original
    patch. (This is the key alignment guarantee for train/val/test
    label transfer.)
    """
    import ee  # noqa: PLC0415
    import pyproj  # noqa: PLC0415
    half_m = (patch_size_pixels * resolution_metres) / 2.0
    # Manifest stores the cell-centre coords (cx, cy) per
    # patchspec_to_origin_meta in sampling.py. The original harvest builds
    # the patch bbox as [cx-half, cy-half, cx+half, cy+half] in UTM, then
    # reprojects to WGS84 for the Rectangle. We replicate that exactly so
    # the multi-temporal patch lands at the same pixel coords.
    x_min = origin_utm_x_m - half_m
    x_max = origin_utm_x_m + half_m
    y_min = origin_utm_y_m - half_m
    y_max = origin_utm_y_m + half_m
    to_wgs = pyproj.Transformer.from_crs(
        f"EPSG:{int(origin_utm_epsg)}", "EPSG:4326", always_xy=True,
    )
    ll = to_wgs.transform(x_min, y_min)
    ur = to_wgs.transform(x_max, y_max)
    return ee.Geometry.Rectangle(
        [ll[0], ll[1], ur[0], ur[1]], proj="EPSG:4326", evenOdd=True,
    )
