"""Earth Engine collection wrappers for Sentinel-1 GRD and Sentinel-2 SR.

Two minimal builders that return ``ee.ImageCollection`` filtered to an AOI,
date window, and required polarisations / cloud constraints. They do not
themselves run any GEE computation; they just compose the collection
expression. Materialisation happens later when the harvest asks for items
or exports.

Reference docs (consult these directly before changing anything):

* COPERNICUS/S1_GRD: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD
* COPERNICUS/S2_SR_HARMONIZED: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from .config import Sentinel1Section, Sentinel2Section

LOG = logging.getLogger(__name__)


def s1_grd_collection(
    cfg: Sentinel1Section,
    aoi: Any,                # ee.Geometry
    start: date,
    end: date,
) -> Any:                    # ee.ImageCollection
    """Build a filtered Sentinel-1 GRD ``ee.ImageCollection``.

    Filters applied:

    * geographic intersection with ``aoi``
    * acquisition between ``start`` and ``end``
    * IW mode (Interferometric Wide swath) — the only mode for the UK heath
    * polarisations include all entries in ``cfg.polarisations``
    * orbit pass (ascending / descending / any), per ``cfg.orbit``

    The returned collection contains *uncalibrated* GRD; pass it through
    :func:`gee_s1s2.calibration.calibrate_grd_collection` to get
    gamma-naught dB comparable to the v2 MPC RTC product.
    """
    import ee  # noqa: PLC0415

    coll = (
        ee.ImageCollection(cfg.collection_id)
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
    )
    # Polarisation filter: every requested polarisation must be present.
    for pol in cfg.polarisations:
        coll = coll.filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))

    if cfg.orbit != "any":
        # GEE uses ASCENDING / DESCENDING in S1 metadata.
        coll = coll.filter(ee.Filter.eq("orbitProperties_pass", cfg.orbit.upper()))

    LOG.debug(
        "S1 collection: %s, AOI=%s, date=%s..%s, pols=%s, orbit=%s",
        cfg.collection_id, aoi.getInfo() if hasattr(aoi, "getInfo") else "<ee.Geometry>",
        start, end, cfg.polarisations, cfg.orbit,
    )
    return coll


def s2_sr_collection(
    cfg: Sentinel2Section,
    aoi: Any,                # ee.Geometry
    start: date,
    end: date,
    cloud_cover_override_pct: float | None = None,
) -> Any:                    # ee.ImageCollection
    """Build a filtered Sentinel-2 SR_HARMONIZED ``ee.ImageCollection``.

    Filters applied at this stage are deliberately loose at the cloud
    side: we use the tile-level ``CLOUDY_PIXEL_PERCENTAGE`` only as a
    coarse pre-filter (``< 1.5x`` the configured ``max_aoi_cloud_cover_percent``,
    matching v2's behaviour). Final per-AOI cloud filtering happens in
    :mod:`gee_s1s2.filtering` against the SCL band.

    ``cloud_cover_override_pct`` lets a per-window override (e.g. the
    inference ``post-fire 2022`` window) widen the tile-level prefilter
    to admit cloudy scenes for the cloud-penetration use case.
    """
    import ee  # noqa: PLC0415

    cloud_basis = float(
        cloud_cover_override_pct
        if cloud_cover_override_pct is not None
        else cfg.max_aoi_cloud_cover_percent
    )
    pre_filter_pct = cloud_basis * 1.5
    coll = (
        ee.ImageCollection(cfg.collection_id)
        .filterBounds(aoi)
        .filterDate(str(start), str(end))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", pre_filter_pct))
    )
    # GEE's COPERNICUS/S2_SR_HARMONIZED labels reflectance bands as
    # 'B1', 'B2', ... while MPC and the v2 PyTorch config use the
    # zero-padded 'B01', 'B02', ... convention. Rename here so the
    # rest of the pipeline (config, manifest, export) sees the names
    # the v2 schema already specifies.
    gee_band_names = [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
        "B8A", "B9", "B11", "B12", "SCL",
    ]
    mpc_band_names = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
        "B8A", "B09", "B11", "B12", "SCL",
    ]
    coll = coll.map(lambda img: img.select(gee_band_names, mpc_band_names))
    LOG.debug(
        "S2 collection: %s, AOI=%s, date=%s..%s, max_aoi_cloud=%s%% "
        "(tile pre-filter <%.1f%%)",
        cfg.collection_id, "<ee.Geometry>", start, end,
        cfg.max_aoi_cloud_cover_percent, pre_filter_pct,
    )
    return coll


def list_collection_ids(coll: Any, max_items: int = 500) -> list[str]:
    """Eagerly list the system:index ids for a small collection.

    Useful for dry-run reporting. ``coll.aggregate_array('system:index')``
    is a single getInfo() call, so this is cheap for collections under
    a few hundred items. For larger collections, use ``coll.size()`` and
    paginate via ``coll.toList()``.
    """
    return list(coll.limit(max_items).aggregate_array("system:index").getInfo())
