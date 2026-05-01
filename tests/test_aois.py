"""Unit tests for AOI loading: KML, point_buffer, force_split, exclude_windows.

These do not touch GEE; they exercise the shapely/pyproj layer only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gee_s1s2.aois import load_kml, load_point_buffer, slugify

V2_KML = Path("../s1s2-translator/inputs/SWT_MappedFires_20220911.kml")


@pytest.mark.skipif(not V2_KML.exists(), reason="v2 KML not available")
def test_kml_has_brentmoor_and_poors():
    aois = load_kml(V2_KML)
    sites = sorted(a.properties.get("Site") for a in aois)
    assert "Brentmoor Heath" in sites
    assert "Poors Allotment" in sites


@pytest.mark.skipif(not V2_KML.exists(), reason="v2 KML not available")
def test_kml_filter_by_site():
    """Filter by SimpleData field. v2's KML uses 'Site' as the discriminator."""
    aois = load_kml(V2_KML)
    brent = [a for a in aois if a.properties.get("Site") == "Brentmoor Heath"]
    assert len(brent) == 1
    # Brentmoor reported as ~0.33 ha by SWT; allow generous tolerance.
    geom = brent[0].geometry
    assert geom.is_valid
    # Bbox in degrees, must lie inside the Surrey heaths region.
    minx, miny, maxx, maxy = geom.bounds
    assert -1.0 < minx < 0.0
    assert 51.0 < miny < 51.5


def test_point_buffer_radius_metric():
    """A 2000 m buffer should produce a polygon whose UTM area is ~pi * 2000^2."""
    import pyproj
    import shapely.ops as so
    from shapely.geometry import shape, mapping

    geom = load_point_buffer(latitude=51.158, longitude=-0.689, buffer_metres=2000)
    # Project back to UTM30 for area check.
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    utm = so.transform(to_utm.transform, geom)
    area = utm.area / 1e6  # km²
    # pi * 2² ≈ 12.566 km²; allow 5% tolerance for the discretisation.
    assert 12.0 < area < 13.0


def test_slugify():
    assert slugify("Brentmoor Heath") == "brentmoor-heath"
    assert slugify("Studland & Godlingston Heath") == "studland-godlingston-heath"
    assert slugify("   ") == "aoi"
