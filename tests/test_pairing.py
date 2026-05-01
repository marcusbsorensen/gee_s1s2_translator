"""Unit tests for the pair-finding logic. No GEE needed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gee_s1s2.config import PairingSection
from gee_s1s2.pairing import Pair, S1Item, S2Item, find_pairs


def _s1(id_: str, dt: datetime, orbit_pass: str = "ASCENDING",
        relative_orbit: int = 132) -> S1Item:
    return S1Item(
        id=id_, datetime_iso=dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        orbit_pass=orbit_pass, relative_orbit=relative_orbit,
        polarisations=["VV", "VH"], image_obj=None,
    )


def _s2(id_: str, dt: datetime, relative_orbit: int = 132) -> S2Item:
    return S2Item(
        id=id_, datetime_iso=dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        relative_orbit=relative_orbit, image_obj=None,
    )


def test_pair_within_window():
    base = datetime(2022, 8, 1, 10, 0, tzinfo=timezone.utc)
    s1 = [_s1("a", base)]
    s2 = [_s2("b", base + timedelta(hours=4))]
    out = find_pairs("aoi", s1, s2, PairingSection(max_temporal_separation_days=1))
    assert len(out) == 1
    assert out[0].separation_days < 1


def test_pair_outside_window_rejected():
    base = datetime(2022, 8, 1, 10, 0, tzinfo=timezone.utc)
    s1 = [_s1("a", base)]
    s2 = [_s2("b", base + timedelta(days=10))]
    out = find_pairs("aoi", s1, s2, PairingSection(max_temporal_separation_days=3))
    assert out == []


def test_prefer_same_orbit():
    base = datetime(2022, 8, 1, 10, 0, tzinfo=timezone.utc)
    s1 = [
        _s1("near-other-orbit", base + timedelta(hours=1), relative_orbit=88),
        _s1("far-same-orbit",  base + timedelta(hours=20), relative_orbit=132),
    ]
    s2 = [_s2("target", base, relative_orbit=132)]
    out = find_pairs("aoi", s1, s2, PairingSection(prefer_same_orbit=True))
    assert out[0].s1.id == "far-same-orbit"


def test_require_same_relative_orbit():
    base = datetime(2022, 8, 1, 10, 0, tzinfo=timezone.utc)
    s1 = [_s1("a", base, relative_orbit=88)]
    s2 = [_s2("b", base + timedelta(hours=4), relative_orbit=132)]
    out = find_pairs("aoi", s1, s2,
                     PairingSection(require_same_relative_orbit=True))
    assert out == []
