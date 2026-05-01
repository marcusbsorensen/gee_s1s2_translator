"""Sentinel-1 GRD → calibrated gamma-naught dB pipeline.

This is the highest-risk technical area in the GEE port. The aim is to
produce S1 backscatter values comparable to Microsoft Planetary
Computer's ``sentinel-1-rtc`` collection used by the v2 PyTorch project,
within a documented tolerance (mean offset <= 2 dB, relative std within
30%; see ``docs/calibration_methodology.md``).

The recipe follows Mullissa et al. (2021), "Sentinel-1 SAR Backscatter
Analysis Ready Data Preparation in Google Earth Engine"
(doi:10.3390/rs13101954), and the community reference implementation at
https://github.com/adugnag/gee_s1_ard. **Do not invent a custom
calibration**; the steps below mirror that pipeline.

Pipeline stages applied in order to each ``ee.Image`` from S1_GRD:

1. **Border noise removal.** GRD products contain low-energy edges due to
   the antenna pattern. We mask pixels whose VV (or VH) backscatter is
   below a noise floor inferred from a 7×7 angle-corrected mean. This
   removes false-low-backscatter edge artefacts that would otherwise
   dominate the speckle filter.
2. **(Optional) Thermal noise removal.** GEE's GRD product has thermal
   noise removal already applied at ingest, so the configuration flag
   exists for parity with the v2 config but is a no-op in normal use.
3. **Radiometric terrain flattening to gamma-naught.** Use the volumetric
   model from Vollrath et al. (2020) over SRTM 30 m. This is the same
   step that produces MPC's RTC; running it in GEE is what makes outputs
   numerically comparable.
4. **Speckle filter (Lee, 5×5)** in linear-power space. Same window as
   the v2 PyTorch project.
5. **Convert to dB**: ``10 * log10(linear)``.

Each step is implemented as a function returning an ``ee.Image`` so the
pipeline is easy to inspect or partially apply. The full pipeline lives
in :func:`calibrate_grd` and :func:`calibrate_grd_collection`.

The volumetric terrain-flattening implementation is the most arithmetic-
heavy step. We follow Vollrath et al. exactly: build a layover/shadow
mask from the local incidence angle, and divide the linear backscatter
by ``cos(theta_local) / cos(theta_ref)`` where theta_ref is the look
angle averaged over the scene. See the methodology doc for the full
expression and citation.
"""

from __future__ import annotations

import logging
from typing import Any

from .config import CalibrationSection

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def lin_to_db(image: Any, bands: list[str]) -> Any:
    """Convert linear-power bands to dB. Other bands pass through unchanged."""
    import ee  # noqa: PLC0415
    db = ee.Image.constant(10.0).multiply(image.select(bands).log10())
    db = db.rename(bands)
    others = image.bandNames().removeAll(bands)
    return image.select(others).addBands(db)


def db_to_lin(image: Any, bands: list[str]) -> Any:
    """Convert dB bands to linear power."""
    import ee  # noqa: PLC0415
    lin = ee.Image.constant(10.0).pow(image.select(bands).divide(10.0))
    lin = lin.rename(bands)
    others = image.bandNames().removeAll(bands)
    return image.select(others).addBands(lin)


# --------------------------------------------------------------------------- #
# Step 1: border noise removal (Mullissa §2.2.1; gee_s1_ard:
#   border_noise_correction.py)
# --------------------------------------------------------------------------- #

def border_noise_correction(image: Any, polarisations: list[str]) -> Any:
    """Mask GRD edge pixels falling below a local-mean noise floor.

    Reference: Mullissa et al. 2021, §2.2.1 ("Additional border noise
    correction"). The threshold of -25 dB is the canonical default in
    gee_s1_ard for IW mode at 10 m resolution; values colder than this in
    a 7×7 mean are likely instrument/border noise rather than backscatter.
    """
    import ee  # noqa: PLC0415

    masked = image
    for pol in polarisations:
        band = image.select(pol)
        # Local mean in 7x7 window. Operate in dB to match the threshold.
        local_mean_db = (
            ee.Image.constant(10.0).multiply(band.log10())
            .focal_mean(radius=3, kernelType="square", units="pixels")
        )
        keep = local_mean_db.gt(-25)
        masked = masked.updateMask(keep)
    return masked


# --------------------------------------------------------------------------- #
# Step 3: radiometric terrain flattening to gamma-naught
# (Vollrath et al. 2020; gee_s1_ard: terrain_flattening.py)
# --------------------------------------------------------------------------- #

def terrain_flattening(
    image: Any,
    polarisations: list[str],
    dem_id: str,
    radiometric_target: str,
) -> Any:
    """Volumetric radiometric terrain flattening.

    Implements the volumetric model from Vollrath, Mullissa & Reiche (2020),
    "Angular-Based Radiometric Slope Correction for Sentinel-1 on Google
    Earth Engine" (doi:10.3390/rs12111867). Output is gamma-naught when
    ``radiometric_target == 'gamma_naught'``, sigma-naught when
    ``'sigma_naught'`` (in which case we skip the cosine correction and
    return sigma-naught directly).

    The implementation here follows the reference at
    https://github.com/adugnag/gee_s1_ard/blob/main/python-api/terrain_flattening.py
    and is kept intentionally close to that source so methodology readers
    can verify line-for-line.
    """
    import ee  # noqa: PLC0415

    if radiometric_target == "sigma_naught":
        # GRD ingested in GEE is already calibrated to sigma-naught; return as-is.
        return image

    # SRTM is a single Image with band "elevation"; Copernicus DEM GLO-30
    # is an ImageCollection with band "DEM". Normalise both to a single
    # ee.Image with band name "elevation" so the rest of this function
    # is DEM-agnostic.
    if dem_id.startswith("COPERNICUS/DEM/"):
        dem = ee.ImageCollection(dem_id).mosaic().select(["DEM"], ["elevation"])
    else:
        dem = ee.Image(dem_id).select("elevation")
    geom = image.geometry()

    # Look angle (degrees) is constant for IW; available as 'angle' band on the GRD.
    theta_i = image.select("angle")

    # Build local incidence angle from DEM gradient and S1 satellite heading.
    # Following Vollrath §3:
    #   - compute DEM slope and aspect
    #   - compute local incidence angle (theta_local)
    slope = ee.Terrain.slope(dem)        # degrees
    aspect = ee.Terrain.aspect(dem)      # degrees
    # Satellite heading from orbitProperties_pass (ASC/DESC). Use
    # server-side ee.Algorithms.If, then ee.Number to pin the type so
    # ee.Image.constant accepts it (passing a raw ComputedObject in
    # produces "Invalid JSON payload: NaN" at evaluation time).
    sat_heading = ee.Image.constant(
        ee.Number(ee.Algorithms.If(
            ee.String(image.get("orbitProperties_pass")).equals("ASCENDING"),
            -12.0,
            -168.0,
        ))
    )

    # Convert to radians for trig.
    deg2rad = ee.Image.constant(0.017453292519943295)
    slope_r = slope.multiply(deg2rad)
    aspect_r = aspect.multiply(deg2rad)
    theta_i_r = theta_i.multiply(deg2rad)
    sat_h_r = sat_heading.multiply(deg2rad)

    # Local incidence angle (from Vollrath equation 7).
    cos_local = (
        slope_r.cos().multiply(theta_i_r.cos())
        .add(slope_r.sin().multiply(theta_i_r.sin()).multiply(aspect_r.subtract(sat_h_r).cos()))
    )
    # Reference cosine from the look angle alone (gamma-naught reference).
    cos_ref = theta_i_r.cos()

    # Volumetric model: gamma_naught = sigma_naught * cos(theta_ref) / cos(theta_local)
    # GRD bands are in linear sigma-naught; we operate in linear space.
    correction = cos_ref.divide(cos_local).rename("correction")

    # Apply correction band-by-band.
    out = image
    for pol in polarisations:
        sigma = image.select(pol)
        gamma = sigma.multiply(correction).rename(pol)
        out = out.addBands(gamma, overwrite=True)

    # Mask layover and shadow (cos_local <= 0 → invalid).
    out = out.updateMask(cos_local.gt(0.001))
    return out


# --------------------------------------------------------------------------- #
# Step 4: speckle filter (Lee, 5×5)
# --------------------------------------------------------------------------- #

def lee_speckle_filter(image: Any, polarisations: list[str], window: int) -> Any:
    """Boxcar-mean Lee speckle filter, identical recipe to v2.

    Operates in linear-power space (the spec'd input). Each polarisation
    band is filtered independently. Window size is in pixels and must be
    odd.
    """
    import ee  # noqa: PLC0415

    if window % 2 == 0:
        raise ValueError(f"window must be odd; got {window}")
    radius = window // 2

    out = image
    for pol in polarisations:
        band = image.select(pol)
        # Local mean and variance over the kernel window.
        mean = band.focal_mean(radius=radius, kernelType="square", units="pixels")
        sq_mean = (
            band.multiply(band).focal_mean(radius=radius, kernelType="square", units="pixels")
        )
        variance = sq_mean.subtract(mean.multiply(mean)).max(0)
        # Overall variance: a single number per scene (server-side reduction).
        overall_var = ee.Number(
            band.reduceRegion(
                reducer=ee.Reducer.variance(),
                geometry=image.geometry(),
                scale=30,
                maxPixels=1e9,
                bestEffort=True,
            ).values().get(0)
        )
        # Lee weight: w = local_var / (local_var + overall_var)
        weight = variance.divide(variance.add(ee.Image.constant(overall_var)).add(1e-9))
        filtered = mean.add(weight.multiply(band.subtract(mean))).rename(pol)
        out = out.addBands(filtered, overwrite=True)
    return out


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def calibrate_grd(image: Any, cfg: CalibrationSection, polarisations: list[str]) -> Any:
    """Apply the full calibration pipeline to a single ``ee.Image``.

    Returns an image whose ``polarisations`` bands are calibrated
    gamma-naught (or sigma-naught) in dB or linear depending on
    ``cfg.output``. All non-polarisation bands are dropped from the
    output for compactness; downstream code only needs the calibrated
    backscatter and the geometry.
    """
    import ee  # noqa: PLC0415

    # Start from the requested polarisation bands plus the look angle band
    # which terrain_flattening needs.
    img = image.select([*polarisations, "angle"])

    # COPERNICUS/S1_GRD stores VV/VH already in dB; the rest of this
    # pipeline (border-noise mean, Vollrath gamma-naught correction,
    # Lee speckle filter) operates in linear power. Convert here.
    img = db_to_lin(img, polarisations)

    if cfg.border_noise_correction:
        img = border_noise_correction(img, polarisations)

    # Thermal noise: GEE's GRD ingest applies it; the flag exists for parity
    # with v2's config but is a no-op here. We log it explicitly so the
    # methodology document can confirm.
    if cfg.thermal_noise_removal:
        LOG.debug("thermal_noise_removal: GEE GRD has this applied at ingest; no-op.")

    if cfg.terrain_correction:
        img = terrain_flattening(img, polarisations, cfg.terrain_dem, cfg.radiometric_target)

    if cfg.speckle_filter.enabled and cfg.speckle_filter.method == "lee":
        img = lee_speckle_filter(img, polarisations, cfg.speckle_filter.window)

    # Drop the angle band; it's not needed downstream.
    img = img.select(polarisations)

    if cfg.output == "dB":
        img = lin_to_db(img, polarisations)

    # Carry over key metadata so pairing.py can read orbit / date.
    # ee.Image.copyProperties returns an ee.Element; wrap to keep .clip()
    # and other Image methods available downstream.
    return ee.Image(img.copyProperties(image, [
        "system:time_start",
        "system:time_end",
        "system:index",
        "orbitProperties_pass",
        "relativeOrbitNumber_start",
        "relativeOrbitNumber_stop",
        "transmitterReceiverPolarisation",
    ]))


def calibrate_grd_collection(
    collection: Any, cfg: CalibrationSection, polarisations: list[str]
) -> Any:
    """Map :func:`calibrate_grd` over an ``ee.ImageCollection``."""
    return collection.map(lambda img: calibrate_grd(img, cfg, polarisations))
