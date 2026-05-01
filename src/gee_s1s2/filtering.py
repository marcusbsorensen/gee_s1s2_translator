"""Per-AOI cloud filtering and per-pixel validity masking.

Two checks live here:

1. **Per-pair AOI cloud cover**, computed from the Sentinel-2 SCL band
   over the AOI mask. Mirrors the v2 PyTorch project's
   ``quick_aoi_cloud_pct``.

2. **Per-pixel validity**, combining the SCL invalid classes for S2 and
   the S1 swath footprint (any-band-non-zero) for S1.

Each function returns either a scalar percentage (for the cloud check)
or an ``ee.Image`` boolean mask (for per-pixel validity).
"""

from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger(__name__)

# SCL classes considered invalid for training. Same as v2.
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)
# 0 no_data, 1 saturated_or_defective, 3 cloud_shadows,
# 8 cloud_medium_probability, 9 cloud_high_probability,
# 10 thin_cirrus, 11 snow_or_ice


def s2_validity_mask(image: Any) -> Any:
    """Boolean mask: True where SCL is *not* in the invalid set."""
    import ee  # noqa: PLC0415
    scl = image.select("SCL")
    invalid_imgs = [scl.eq(c) for c in SCL_INVALID]
    invalid = ee.Image(invalid_imgs[0])
    for im in invalid_imgs[1:]:
        invalid = invalid.Or(im)
    return invalid.Not().rename("s2_valid")


def s1_validity_mask(image: Any, polarisations: list[str]) -> Any:
    """Boolean mask: True where any S1 polarisation has non-zero data.

    GEE's S1 GRD masks off-swath pixels at ingest, so an explicit zero
    check on linear-power values is robust whether the image is in
    linear or dB.
    """
    import ee  # noqa: PLC0415
    bands = [image.select(p).neq(0) for p in polarisations]
    out = bands[0]
    for b in bands[1:]:
        out = out.Or(b)
    return out.rename("s1_valid")


def aoi_cloud_pct(s2_image: Any, aoi: Any, scale_m: float = 60.0) -> float:
    """Compute the AOI cloud-or-shadow percentage from SCL.

    Reads SCL at coarse resolution (default 60 m) inside the AOI, counts
    pixels in the SCL_INVALID set, returns percentage. Server-side
    reduction; one ``getInfo()`` per call.
    """
    import ee  # noqa: PLC0415
    scl = s2_image.select("SCL")
    invalid_imgs = [scl.eq(c) for c in SCL_INVALID]
    invalid = ee.Image(invalid_imgs[0])
    for im in invalid_imgs[1:]:
        invalid = invalid.Or(im)
    stats = invalid.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale_m,
        maxPixels=1e9,
        bestEffort=True,
    )
    val = stats.get("SCL")
    return float(ee.Number(val).getInfo()) * 100.0
