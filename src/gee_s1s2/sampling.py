"""Random patch sampling per scene with n_patches budget per (AOI, window).

Mirrors the v2 PyTorch project's ``random_patches`` strategy with the
same min-spacing semantics: minimum spacing between accepted origins is
``patch_size_pixels - patch_overlap_pixels`` in either axis (OR rule).
The total budget for an ``(AOI, window)`` bucket is distributed across
the accepted scenes by :func:`distribute_budget`, exactly as v2.

The sampling itself runs client-side. To keep the AOI mask tractable
for large polygons (TBH SPA training area is ~95 km², which would be
~10M cells at 10 m), the polygon is rasterised at a *coarse* grid
step (default 200 m). The sampler runs at that coarse grid; accepted
origins are upsampled back to 10 m for the export geometry. The
patch size in coarse units is therefore ``ceil(patch_size_pixels *
pixel_size_m / coarse_step_m)``.

Per-pixel S1/S2 validity masks are not used in the GEE port: we
already filter at scene level via the AOI cloud check (≤8 % cloud in
``filtering.aoi_cloud_pct``). Per-pixel validity could be wired in
later by sampling SCL and S1 mask bands at the same coarse step.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform

LOG = logging.getLogger(__name__)


@dataclass
class PatchSpec:
    """A single sampled patch: pixel-space origin and the geometry to export."""
    pair_id: str
    index: int
    y0: int
    x0: int
    geometry: Any            # ee.Geometry.Rectangle in EPSG:4326


def distribute_budget(total: int, n_buckets: int) -> list[int]:
    """Split ``total`` across ``n_buckets`` as evenly as possible.

    Identical to ``patches.distribute_budget`` in the v2 project.
    """
    if n_buckets <= 0 or total <= 0:
        return [0] * max(0, n_buckets)
    base, rem = divmod(total, n_buckets)
    return [base + (1 if i < rem else 0) for i in range(n_buckets)]


def per_scene_seed(global_seed: int, pair_id: str) -> int:
    """Deterministic per-scene RNG seed; same algorithm as v2."""
    raw = f"{global_seed}:{pair_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0xFFFFFFFF


def sample_origins(
    H: int,
    W: int,
    patch_size: int,
    overlap: int,
    s2_valid_mask: np.ndarray,
    s1_valid_mask: np.ndarray,
    aoi_mask: np.ndarray,
    n_patches: int,
    rng: np.random.Generator,
    max_invalid_fraction: float = 0.30,
    s1_max_invalid_fraction: float = 0.05,
    max_attempts_factor: int = 200,
) -> list[tuple[int, int]]:
    """Sample up to ``n_patches`` non-overlapping origins.

    Same min-spacing rule as the v2 PyTorch project: two patches don't
    overlap by more than ``overlap`` iff ``|dy| > P - overlap`` OR
    ``|dx| > P - overlap``. The first random sample stochastically
    blocks roughly half of the theoretically valid grid positions for
    small AOIs; this is documented in the v2 README and is expected.
    """
    if n_patches <= 0:
        return []
    range_y = H - patch_size
    range_x = W - patch_size
    if range_y < 0 or range_x < 0:
        LOG.warning(
            "Patch size %d exceeds data window %dx%d; cannot sample.",
            patch_size, H, W,
        )
        return []

    min_spacing = max(1, patch_size - overlap)
    accepted: list[tuple[int, int]] = []
    max_attempts = max(max_attempts_factor * n_patches, 200)
    attempts = 0
    while len(accepted) < n_patches and attempts < max_attempts:
        attempts += 1
        y = int(rng.integers(0, range_y + 1))
        x = int(rng.integers(0, range_x + 1))
        cy, cx = y + patch_size // 2, x + patch_size // 2
        if cy >= H or cx >= W or not aoi_mask[cy, cx]:
            continue
        if any(
            (abs(y - y0) < min_spacing) and (abs(x - x0) < min_spacing)
            for y0, x0 in accepted
        ):
            continue
        sub_s2 = s2_valid_mask[y : y + patch_size, x : x + patch_size]
        if (1.0 - float(sub_s2.mean())) > max_invalid_fraction:
            continue
        sub_s1 = s1_valid_mask[y : y + patch_size, x : x + patch_size]
        if (1.0 - float(sub_s1.mean())) > s1_max_invalid_fraction:
            continue
        accepted.append((y, x))
    if len(accepted) < n_patches:
        LOG.warning(
            "random_patches: only %d/%d sampled after %d attempts.",
            len(accepted), n_patches, attempts,
        )
    return accepted


@dataclass
class CoarseAoiGrid:
    """Coarse-grid rasterised AOI in UTM, ready for the sampler."""
    mask: np.ndarray              # (H, W) bool, True = inside polygon
    H: int                        # rows in coarse grid
    W: int                        # cols in coarse grid
    coarse_step_m: float          # metres per coarse cell
    utm_origin_x: float           # UTM x (metres) of mask[0, 0] top-left corner
    utm_origin_y: float           # UTM y (metres) of mask[0, 0] top-left corner
    utm_epsg: int


def utm_epsg_for(lon: float, lat: float) -> int:
    """Northern-hemisphere UTM zone EPSG for a (lon, lat) seed."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def coarse_aoi_mask(
    geom_4326: Any,                 # shapely Polygon/MultiPolygon in EPSG:4326
    coarse_step_m: float,
) -> CoarseAoiGrid:
    """Rasterise the AOI polygon to a coarse boolean mask in UTM.

    The output is suitable for :func:`sample_origins` together with a
    matching coarse patch size (in cells, not pixels).
    """
    import pyproj  # noqa: PLC0415

    centroid = geom_4326.centroid
    utm_epsg = utm_epsg_for(centroid.x, centroid.y)
    to_utm = pyproj.Transformer.from_crs(
        "EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True,
    )
    geom_utm = shapely_transform(lambda x, y: to_utm.transform(x, y), geom_4326)
    minx, miny, maxx, maxy = geom_utm.bounds

    H = int(math.ceil((maxy - miny) / coarse_step_m))
    W = int(math.ceil((maxx - minx) / coarse_step_m))
    if H <= 0 or W <= 0:
        raise ValueError(
            f"Invalid coarse AOI size H={H} W={W} (bounds={geom_utm.bounds!r})"
        )

    # Rasterise with a simple per-cell centroid containment test. For
    # AOIs of typical heath size (1–100 km²) at coarse_step_m=200 the
    # mask is ~30–250 cells per side, so this is fast.
    mask = np.zeros((H, W), dtype=bool)
    for j in range(H):
        cy = maxy - (j + 0.5) * coarse_step_m
        for i in range(W):
            cx = minx + (i + 0.5) * coarse_step_m
            from shapely.geometry import Point  # noqa: PLC0415
            if geom_utm.contains(Point(cx, cy)):
                mask[j, i] = True

    return CoarseAoiGrid(
        mask=mask, H=H, W=W,
        coarse_step_m=coarse_step_m,
        utm_origin_x=minx, utm_origin_y=maxy,
        utm_epsg=utm_epsg,
    )


def origins_to_patch_geoms(
    origins: list[tuple[int, int]],
    grid: CoarseAoiGrid,
    pixel_size_m: float,
    patch_size_pixels: int,
    pair_id: str,
) -> list[PatchSpec]:
    """Convert coarse-grid origins to ``ee.Geometry.Rectangle`` patches.

    ``origins`` are ``(y_cell, x_cell)`` indices into ``grid.mask`` (the
    coarse rasterised AOI). For each origin we produce a UTM rectangle
    of ``patch_size_pixels × patch_size_pixels`` at ``pixel_size_m``,
    centred on the cell, then reproject the corners to EPSG:4326 for the
    ``ee.Geometry.Rectangle`` used by ``Export.image.toCloudStorage``.
    """
    import ee  # noqa: PLC0415
    import pyproj

    to_wgs = pyproj.Transformer.from_crs(
        f"EPSG:{grid.utm_epsg}", "EPSG:4326", always_xy=True,
    )
    patch_size_m = patch_size_pixels * pixel_size_m
    half = patch_size_m / 2.0

    patches: list[PatchSpec] = []
    for idx, (y_cell, x_cell) in enumerate(origins):
        # Coarse-cell centre in UTM metres.
        cx = grid.utm_origin_x + (x_cell + 0.5) * grid.coarse_step_m
        cy = grid.utm_origin_y - (y_cell + 0.5) * grid.coarse_step_m
        # Patch UTM bbox centred on the cell.
        x_min, x_max = cx - half, cx + half
        y_min, y_max = cy - half, cy + half
        ll = to_wgs.transform(x_min, y_min)
        ur = to_wgs.transform(x_max, y_max)
        rect = ee.Geometry.Rectangle(
            [ll[0], ll[1], ur[0], ur[1]], proj="EPSG:4326", evenOdd=True,
        )
        patches.append(PatchSpec(
            pair_id=pair_id, index=idx, y0=y_cell, x0=x_cell, geometry=rect,
        ))
    return patches


def patchspec_to_origin_meta(p: PatchSpec, grid: CoarseAoiGrid) -> dict[str, float]:
    """Per-patch metadata for the manifest: cell index + UTM + lon/lat."""
    import pyproj  # noqa: PLC0415
    to_wgs = pyproj.Transformer.from_crs(
        f"EPSG:{grid.utm_epsg}", "EPSG:4326", always_xy=True,
    )
    cx = grid.utm_origin_x + (p.x0 + 0.5) * grid.coarse_step_m
    cy = grid.utm_origin_y - (p.y0 + 0.5) * grid.coarse_step_m
    lon, lat = to_wgs.transform(cx, cy)
    return {
        "origin_index": p.index,
        "origin_y_cell": p.y0,
        "origin_x_cell": p.x0,
        "origin_utm_epsg": grid.utm_epsg,
        "origin_utm_x_m": cx,
        "origin_utm_y_m": cy,
        "origin_lon": lon,
        "origin_lat": lat,
    }
