"""S1/S2 pair finding for the GEE port.

Same logic as the v2 PyTorch project (``s1s2/pairs.py``) but adapted for
GEE feature collections rather than STAC items. We materialise the two
collections to lightweight Python records via a single ``getInfo()`` per
collection, then run the pairing logic locally. The pairs are small
(under a few thousand items per harvest), so client-side pairing is
appropriate; pushing this into GEE server-side would be more complex
and gain nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .config import PairingSection

LOG = logging.getLogger(__name__)


@dataclass
class S1Item:
    """Lightweight S1 record extracted from an ``ee.Image`` properties dict."""
    id: str
    datetime_iso: str
    orbit_pass: str           # "ASCENDING" / "DESCENDING"
    relative_orbit: int | None
    polarisations: list[str]
    image_obj: Any            # ee.Image (calibrated)


@dataclass
class S2Item:
    id: str
    datetime_iso: str
    relative_orbit: int | None
    image_obj: Any            # ee.Image


@dataclass
class Pair:
    s1: S1Item
    s2: S2Item
    aoi_name: str
    separation_seconds: float

    @property
    def separation_days(self) -> float:
        return self.separation_seconds / 86400.0

    @property
    def pair_id(self) -> str:
        return f"{self.aoi_name}::{self.s1.id}::{self.s2.id}"


def _ts_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def materialise_s1(coll: Any) -> list[S1Item]:
    """Pull S1 items from an ``ee.ImageCollection`` into local records."""
    info = coll.getInfo()
    items: list[S1Item] = []
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        items.append(S1Item(
            id=feat.get("id", "").split("/")[-1],
            datetime_iso=_ts_to_iso(int(props["system:time_start"])),
            orbit_pass=props.get("orbitProperties_pass", ""),
            relative_orbit=props.get("relativeOrbitNumber_start"),
            polarisations=list(props.get("transmitterReceiverPolarisation", [])),
            image_obj=None,  # filled in by caller via .filter(eq id)
        ))
    return items


def materialise_s2(coll: Any) -> list[S2Item]:
    info = coll.getInfo()
    items: list[S2Item] = []
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        items.append(S2Item(
            id=feat.get("id", "").split("/")[-1],
            datetime_iso=_ts_to_iso(int(props["system:time_start"])),
            relative_orbit=props.get("SENSING_ORBIT_NUMBER"),
            image_obj=None,
        ))
    return items


def find_pairs(
    aoi_name: str,
    s1_items: list[S1Item],
    s2_items: list[S2Item],
    cfg: PairingSection,
) -> list[Pair]:
    """Return accepted pairs sorted by ascending temporal separation."""
    max_sep = cfg.max_temporal_separation_days * 86400.0
    pairs: list[Pair] = []
    accepted = 0
    rejected_temporal = 0
    rejected_orbit = 0

    for s2 in s2_items:
        s2_t = _parse_iso(s2.datetime_iso)
        candidates: list[tuple[float, S1Item]] = []
        for s1 in s1_items:
            sep = abs((s2_t - _parse_iso(s1.datetime_iso)).total_seconds())
            if sep > max_sep:
                rejected_temporal += 1
                continue
            if cfg.require_same_relative_orbit:
                if s1.relative_orbit is None or s2.relative_orbit is None or \
                   s1.relative_orbit != s2.relative_orbit:
                    rejected_orbit += 1
                    continue
            candidates.append((sep, s1))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        if cfg.prefer_same_orbit and s2.relative_orbit is not None:
            preferred = [
                (sep, s1) for sep, s1 in candidates
                if s1.relative_orbit == s2.relative_orbit
            ]
            if preferred:
                candidates = preferred + [c for c in candidates if c not in preferred]

        sep, best = candidates[0]
        pairs.append(Pair(
            s1=best, s2=s2, aoi_name=aoi_name, separation_seconds=sep,
        ))
        accepted += 1
        LOG.info(
            "Accepted pair: aoi=%r s1=%s s2=%s separation_days=%.2f",
            aoi_name, best.id, s2.id, sep / 86400.0,
        )

    LOG.info(
        "Pair summary for %r: accepted=%d rejected_temporal=%d rejected_orbit=%d",
        aoi_name, accepted, rejected_temporal, rejected_orbit,
    )
    pairs.sort(key=lambda p: p.separation_seconds)
    return pairs
