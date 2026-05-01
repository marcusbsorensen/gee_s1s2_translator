"""AOI loading: KML / GeoJSON / point_buffer to Earth Engine geometries.

Mirrors the v2 PyTorch project's AOI handling (KML SimpleData filtering and
point_buffer) so the GEE port produces semantically identical AOIs. Output
is always an ``ee.Geometry`` ready to feed into ``filterBounds``.

For source ``kml`` and ``geojson``, the file is parsed in shapely and
converted to GeoJSON, then handed to ``ee.Geometry``. For ``point_buffer``,
the buffer is computed in a local UTM projection and reprojected into
EPSG:4326, then handed to ``ee.Geometry``.

Force-split routing and exclude-windows logic live on the ``AOIDef`` model
itself (see ``config.py``); this module only handles geometry construction.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyproj
import shapely.ops
from shapely.geometry import (
    LinearRing,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import AOIDef, AOISource

KML_NS = "{http://www.opengis.net/kml/2.2}"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedAOI:
    name: str
    geometry: BaseGeometry  # in EPSG:4326
    properties: dict


def slugify(name: str) -> str:
    """Filesystem and bucket-key safe slug. Same algorithm as the v2 project."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return cleaned or "aoi"


# --------------------------------------------------------------------------- #
# KML / GeoJSON parsing
# --------------------------------------------------------------------------- #

def _parse_coordinate_block(text: str) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for tok in text.replace("\n", " ").split():
        parts = tok.split(",")
        if len(parts) < 2:
            continue
        try:
            coords.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return coords


def _polygon_from_kml(polygon_el: ET.Element) -> Polygon | None:
    outer_el = polygon_el.find(f"{KML_NS}outerBoundaryIs/{KML_NS}LinearRing/{KML_NS}coordinates")
    if outer_el is None or outer_el.text is None:
        return None
    outer = _parse_coordinate_block(outer_el.text)
    if len(outer) < 4:
        return None
    inner_rings: list[LinearRing] = []
    for inner_el in polygon_el.findall(
        f"{KML_NS}innerBoundaryIs/{KML_NS}LinearRing/{KML_NS}coordinates"
    ):
        if inner_el.text is None:
            continue
        inner = _parse_coordinate_block(inner_el.text)
        if len(inner) >= 4:
            inner_rings.append(LinearRing(inner))
    return Polygon(outer, holes=inner_rings)


def _placemark_geometry(placemark: ET.Element) -> BaseGeometry | None:
    polys: list[Polygon] = []
    for poly_el in placemark.iter(f"{KML_NS}Polygon"):
        poly = _polygon_from_kml(poly_el)
        if poly is not None and not poly.is_empty:
            polys.append(poly)
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


def _placemark_properties(placemark: ET.Element) -> dict:
    props: dict = {}
    name_el = placemark.find(f"{KML_NS}name")
    if name_el is not None and name_el.text:
        props["Name"] = name_el.text.strip()
    for sd in placemark.iter(f"{KML_NS}SimpleData"):
        key = sd.get("name")
        if key is not None:
            props[key] = sd.text.strip() if sd.text else ""
    return props


def load_kml(path: Path) -> list[LoadedAOI]:
    if not path.exists():
        raise FileNotFoundError(f"KML not found: {path}")
    tree = ET.parse(path)
    root = tree.getroot()
    out: list[LoadedAOI] = []
    for placemark in root.iter(f"{KML_NS}Placemark"):
        geom = _placemark_geometry(placemark)
        if geom is None:
            continue
        props = _placemark_properties(placemark)
        out.append(LoadedAOI(
            name=props.get("Name") or props.get("Site") or path.stem,
            geometry=geom,
            properties=props,
        ))
    LOG.info("Loaded %d AOI(s) from %s", len(out), path)
    return out


def load_geojson(path: Path) -> list[LoadedAOI]:
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    out: list[LoadedAOI] = []
    for idx, row in gdf.iterrows():
        props = {k: v for k, v in row.items() if k != gdf.geometry.name}
        out.append(LoadedAOI(
            name=str(props.get("name") or props.get("Name") or f"{path.stem}-{idx}"),
            geometry=row.geometry,
            properties=props,
        ))
    return out


# --------------------------------------------------------------------------- #
# point_buffer source
# --------------------------------------------------------------------------- #

def _utm_epsg(latitude: float, longitude: float) -> int:
    zone = int((longitude + 180) // 6) + 1
    return 32600 + zone if latitude >= 0 else 32700 + zone


def load_point_buffer(latitude: float, longitude: float, buffer_metres: float) -> BaseGeometry:
    """Build a circular AOI by buffering a centre point in metric UTM."""
    utm_epsg = _utm_epsg(latitude, longitude)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    to_wgs = pyproj.Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(longitude, latitude)
    circle_utm = Point(cx, cy).buffer(buffer_metres)
    return shapely.ops.transform(to_wgs.transform, circle_utm)


# --------------------------------------------------------------------------- #
# Public entry: AOISource → shapely → ee.Geometry
# --------------------------------------------------------------------------- #

def resolve_source(source: AOISource) -> BaseGeometry:
    """Resolve any AOISource to a shapely geometry in EPSG:4326."""
    if source.type == "kml":
        records = load_kml(source.path)  # type: ignore[arg-type]
        if source.filter is not None:
            records = [
                r for r in records
                if str(r.properties.get(source.filter.field, "")) == str(source.filter.equals)
            ]
            if not records:
                raise ValueError(
                    f"AOI filter {source.filter.field}={source.filter.equals!r} "
                    f"matched no features in {source.path}"
                )
        if not records:
            raise ValueError(f"No usable features in {source.path}")
        return unary_union([r.geometry for r in records])
    if source.type == "geojson":
        records = load_geojson(source.path)  # type: ignore[arg-type]
        if not records:
            raise ValueError(f"No features in {source.path}")
        return unary_union([r.geometry for r in records])
    if source.type == "point_buffer":
        assert source.latitude is not None and source.longitude is not None
        assert source.buffer_metres is not None
        return load_point_buffer(source.latitude, source.longitude, source.buffer_metres)
    raise ValueError(f"Unknown AOI source type: {source.type!r}")


def aoi_geometry(aoi: AOIDef) -> BaseGeometry:
    """Resolve a full AOI definition (source + buffer_metres) to shapely."""
    geom = resolve_source(aoi.source)
    if aoi.buffer_metres > 0:
        # Buffer in UTM for metric accuracy.
        cx, cy = float(geom.centroid.x), float(geom.centroid.y)
        utm_epsg = _utm_epsg(cy, cx)
        to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        to_wgs = pyproj.Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
        utm_geom = shapely.ops.transform(to_utm.transform, geom)
        buffered = utm_geom.buffer(aoi.buffer_metres)
        geom = shapely.ops.transform(to_wgs.transform, buffered)
    return geom


def shapely_to_ee(geom: BaseGeometry) -> Any:
    """Convert a shapely geometry in EPSG:4326 to an ``ee.Geometry``.

    Earth Engine ``ee.Geometry`` accepts GeoJSON dicts directly. Importing
    ``ee`` lazily so this module remains importable without authenticating.
    """
    import ee  # noqa: PLC0415
    return ee.Geometry(mapping(geom), proj="EPSG:4326", evenOdd=True)
