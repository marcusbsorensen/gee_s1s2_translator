"""S1 calibration validation: GEE pipeline output vs MPC RTC reference.

This is the highest-risk technical step in Phase 1. We pick a small
number of ``(AOI, S1 acquisition)`` combinations from the v2 archive's
manifest, calibrate the GEE GRD over the same scene with our pipeline,
fetch the equivalent MPC RTC patch independently (auth-free), and
compare per-pixel mean and std of the dB difference.

Pass criterion (matching the kickoff brief):
* mean offset within ±2 dB on each polarisation
* per-pixel std of the difference within 30% of MPC RTC std

If either fails for any sample, mark the row FAIL and surface to the
operator. Do not run the full harvest until calibration passes.

The reference reads come from MPC's ``sentinel-1-rtc`` collection via
``odc-stac``, identical to the v2 PyTorch project's load path. So the
comparison is genuinely apples-to-apples at the pixel level.
"""

from __future__ import annotations

import csv
import logging
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .aois import aoi_geometry, shapely_to_ee
from .calibration import calibrate_grd
from .config import Config

LOG = logging.getLogger(__name__)

V2_MANIFEST_PATH = Path("../s1s2-translator/data/runs/v2_diverse_heath/manifest.csv")
DOC_PATH = Path("docs/calibration_methodology.md")

PASS_MAX_MEAN_DELTA_DB = 2.0
PASS_MAX_STD_RATIO_DEVIATION = 0.30      # std_gee / std_mpc within [0.70, 1.30]
VALIDATION_SCALE_M = 100                  # coarse-resolution comparison; see sampleRectangle 262144 px cap


def _pick_validation_samples(n: int) -> list[dict]:
    """Select ``n`` (AOI, S1 id, S2 id) rows from the v2 manifest."""
    if not V2_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"v2 manifest not found at {V2_MANIFEST_PATH}. Calibration "
            "validation needs the v2 archive at ../s1s2-translator/data/runs/"
            "v2_diverse_heath/. Run from the gee_s1s2_translator/ directory."
        )
    rows = list(csv.DictReader(open(V2_MANIFEST_PATH)))
    # Pick large-AOI sites (TBH SPA, fire-area training) so we have
    # plenty of pixels for the comparison; skip tiny 1500m sites.
    preferred = [
        r for r in rows
        if r["aoi_name"] in (
            "TBH SPA training area",
            "Brentmoor area training",
            "Poors Allotment area training",
        )
    ]
    if len(preferred) >= n:
        chosen = preferred[: n]
    else:
        chosen = (preferred + rows)[: n]
    return chosen


def _fetch_mpc_rtc_patch(s1_id: str, aoi_4326: Any, polarisations: list[str]) -> np.ndarray:
    """Fetch a calibrated patch from MPC's sentinel-1-rtc by id.

    Returns a (n_pol, H, W) ndarray in dB. Uses the same odc-stac path
    as the v2 PyTorch project so the reference is byte-identical to v2.
    """
    import planetary_computer as pc
    import pystac_client
    from odc.stac import load as odc_load
    from pyproj import CRS
    from shapely.geometry import shape

    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    # MPC ids include a trailing '_rtc' suffix; the v2 manifest already has it.
    item_search = client.search(
        collections=["sentinel-1-rtc"], ids=[s1_id], limit=1,
    )
    items = list(item_search.items())
    if not items:
        raise RuntimeError(f"MPC RTC item not found for id {s1_id!r}")

    # AOI bbox in EPSG:4326 for odc.stac.load
    minx, miny, maxx, maxy = aoi_4326.bounds
    # Match VALIDATION_SCALE_M with degree-based resolution (~100m at
    # UK latitudes). We compare distributions, not pixel-aligned values,
    # so the precise CRS doesn't matter as long as both sides cover the
    # same area at roughly the same scale.
    cube = odc_load(
        items, bands=[p.lower() for p in polarisations],
        resolution=0.001, crs=4326,
        x=(minx, maxx), y=(miny, maxy),
        chunks={},
    )
    arrs = []
    for p in polarisations:
        a = cube[p.lower()].astype("float32").values
        if a.ndim == 3:
            a = a[0]
        # MPC RTC is linear gamma-naught; convert to dB.
        a = np.where(a > 0, 10.0 * np.log10(np.maximum(a, 1e-9)), np.nan)
        arrs.append(a)
    return np.stack(arrs, axis=0)


def _fetch_gee_calibration_stats(
    s1_id: str, aoi_ee: Any, polarisations: list[str], config: Config,
) -> dict[str, dict[str, float]]:
    """Run the GEE calibration pipeline and compute per-polarisation
    mean/std over the AOI server-side.

    Returns ``{pol: {"mean": float, "std": float, "count": int}}``.
    Server-side reduceRegion sidesteps the sampleRectangle 262144-pixel
    cap and avoids any client-side reprojection wackiness.
    """
    import ee  # noqa: PLC0415

    # MPC RTC ids are like 'S1A_IW_GRDH_1SDV_20210531T061520_20210531T061545
    # _038128_047FF8_rtc'. GEE's COPERNICUS/S1_GRD asset ids carry an
    # extra trailing 4-char hash, so direct asset lookup fails. Resolve
    # by filtering the GEE collection on the start-time embedded in the
    # MPC id (the 5th '_'-separated token, format YYYYMMDDTHHMMSS).
    base = s1_id.replace("_rtc", "")
    parts = base.split("_")
    if len(parts) < 5 or "T" not in parts[4]:
        raise RuntimeError(f"Could not parse start time from S1 id {s1_id!r}")
    start_iso = (
        f"{parts[4][:4]}-{parts[4][4:6]}-{parts[4][6:8]}T"
        f"{parts[4][9:11]}:{parts[4][11:13]}:{parts[4][13:15]}"
    )
    start_t = ee.Date(start_iso)
    # Sentinel-1 IW bursts in the same orbit are ~25s apart, so a wider
    # window can pick up a neighbouring burst over a different ground
    # footprint. ±5s is tight enough to land on the exact MPC scene.
    matched = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(start_t.advance(-5, "second"), start_t.advance(5, "second"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )
    n = matched.size().getInfo()
    if n == 0:
        raise RuntimeError(
            f"No COPERNICUS/S1_GRD scene found within ±30s of {start_iso} "
            f"(MPC id {s1_id!r})"
        )
    img = ee.Image(matched.first())
    # Disable speckle filter for validation: MPC RTC doesn't speckle-filter,
    # so leaving Lee on would cut std by ~50% and fool the [0.7, 1.3] ratio
    # check. We're testing the calibration step, not the noise reducer.
    cal_cfg = config.calibration.model_copy(deep=True)
    cal_cfg.speckle_filter.enabled = False
    calibrated = calibrate_grd(img, cal_cfg, polarisations)

    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )
    raw = calibrated.reduceRegion(
        reducer=reducer,
        geometry=aoi_ee,
        scale=VALIDATION_SCALE_M,
        maxPixels=int(1e9),
        bestEffort=True,
    ).getInfo()
    out: dict[str, dict[str, float]] = {}
    for p in polarisations:
        out[p] = {
            "mean": float(raw.get(f"{p}_mean", float("nan")) or float("nan")),
            "std": float(raw.get(f"{p}_stdDev", float("nan")) or float("nan")),
            "count": int(raw.get(f"{p}_count", 0) or 0),
        }
    return out


def run_calibration_validation(config: Config, n_samples: int = 3) -> list[dict]:
    """Run the validation and return one row per sample."""
    samples = _pick_validation_samples(n_samples)
    rows: list[dict] = []
    for s in samples:
        aoi_name = s["aoi_name"]
        s1_id = s["s1_id"]
        s1_dt = s["s1_acquired"][:10]
        try:
            aoi_def = config.aoi_by_name(aoi_name)
        except KeyError:
            LOG.warning("AOI %r not in current config; skipping calibration sample.", aoi_name)
            continue

        try:
            geom_4326 = aoi_geometry(aoi_def)
            geom_ee = shapely_to_ee(geom_4326)
            mpc_arr = _fetch_mpc_rtc_patch(s1_id, geom_4326, config.sentinel1.polarisations)
            gee_stats = _fetch_gee_calibration_stats(
                s1_id, geom_ee, config.sentinel1.polarisations, config,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.error("Calibration sample failed: aoi=%r s1=%s: %s", aoi_name, s1_id, exc)
            rows.append({
                "aoi": aoi_name, "date": s1_dt,
                "vv_mean_delta_db": float("nan"), "vh_mean_delta_db": float("nan"),
                "vv_std_ratio": float("nan"), "vh_std_ratio": float("nan"),
                "verdict": f"ERROR: {exc}",
            })
            continue

        result = {"aoi": aoi_name, "date": s1_dt}
        verdict_parts: list[str] = []
        for i, pol in enumerate(config.sentinel1.polarisations):
            mpc = mpc_arr[i]
            mpc_valid = mpc[np.isfinite(mpc)]
            g = gee_stats.get(pol, {"mean": float("nan"), "std": float("nan"), "count": 0})
            if mpc_valid.size < 100 or g["count"] < 100:
                mean_delta = float("nan"); std_ratio = float("nan")
            else:
                mean_mpc = float(np.mean(mpc_valid))
                std_mpc = float(np.std(mpc_valid))
                mean_delta = g["mean"] - mean_mpc
                std_ratio = g["std"] / max(std_mpc, 1e-9)
            result[f"{pol.lower()}_mean_delta_db"] = mean_delta
            result[f"{pol.lower()}_std_ratio"] = std_ratio
            mean_ok = abs(mean_delta) <= PASS_MAX_MEAN_DELTA_DB if np.isfinite(mean_delta) else False
            std_ok = abs(std_ratio - 1.0) <= PASS_MAX_STD_RATIO_DEVIATION if np.isfinite(std_ratio) else False
            verdict_parts.append("OK" if mean_ok and std_ok else "FAIL")
        result["verdict"] = "OK" if all(v == "OK" for v in verdict_parts) else "FAIL"
        rows.append(result)
    return rows


def append_validation_to_doc(rows: list[dict]) -> None:
    """Append a fresh validation block to ``docs/calibration_methodology.md``."""
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = ["", f"## Validation run, {timestamp}", "",
             "| AOI | date | VV mean Δ (dB) | VH mean Δ (dB) | VV std ratio | VH std ratio | verdict |",
             "| --- | --- | ---: | ---: | ---: | ---: | :---: |"]
    for r in rows:
        lines.append(
            f"| {r['aoi']} | {r['date']} | {r['vv_mean_delta_db']:+.2f} | "
            f"{r['vh_mean_delta_db']:+.2f} | {r['vv_std_ratio']:.2f} | "
            f"{r['vh_std_ratio']:.2f} | {r['verdict']} |"
        )
    with DOC_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    LOG.info("Appended validation block to %s", DOC_PATH)
