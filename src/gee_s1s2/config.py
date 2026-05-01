"""Pydantic v2 schemas for the GEE port's YAML configuration.

This module mirrors the v2 PyTorch project's config schema (AOIs, date
windows, S2 bands, pairing rules) and adds three GEE-specific blocks:

* ``sentinel1.source: gee_grd_calibrated`` and a new ``calibration`` block
  defining the S1 GRD-to-gamma-naught preprocessing recipe.
* ``sentinel2.source: gee_sr_harmonized`` referencing the
  ``COPERNICUS/S2_SR_HARMONIZED`` collection.
* ``export`` controlling TFRecord layout and concurrency on GCS.

Validation is strict: typos and ambiguous combinations are rejected at load
time with messages telling the user exactly what to fix. Reuses the v2
patterns deliberately so Sonia, who already saw the v2 config, finds this
familiar.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# These mirror the v2 project's allowed values exactly. Kept in this file
# (rather than imported across projects) because the GEE port is meant to be
# standalone and reproducible by Sonia from this repo alone.
ALLOWED_POLARISATIONS = {"VV", "VH", "HH", "HV"}
ALLOWED_S2_BANDS = {
    "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B11", "B12",
}
ALLOWED_INDICES = {"NDVI", "NBR", "NDWI", "NDMI", "SAVI"}


class _StrictBase(BaseModel):
    """Forbid unknown keys so config typos surface as errors immediately."""
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Project, AOI, date window
# --------------------------------------------------------------------------- #

class ProjectSection(_StrictBase):
    name: str
    workspace: Path = Field(default=Path("./local_cache"))
    random_seed: int = 42


class AOISourceFilter(_StrictBase):
    field: str
    equals: str


class AOISource(_StrictBase):
    """Source for an AOI geometry. ``kml`` and ``geojson`` are file-based;
    ``point_buffer`` builds a circular buffer around a lat/lon centre."""

    type: Literal["kml", "geojson", "point_buffer"]
    # File-based sources
    path: Path | None = None
    filter: AOISourceFilter | None = None
    # point_buffer source
    latitude: float | None = None
    longitude: float | None = None
    buffer_metres: float | None = None

    @model_validator(mode="after")
    def _check_required(self) -> "AOISource":
        if self.type in {"kml", "geojson"}:
            if self.path is None:
                raise ValueError(f"source.type={self.type!r} requires source.path")
            if not self.path.exists():
                raise ValueError(f"AOI source path does not exist: {self.path}")
        elif self.type == "point_buffer":
            if self.latitude is None or self.longitude is None or self.buffer_metres is None:
                raise ValueError(
                    "source.type='point_buffer' requires latitude, longitude, buffer_metres"
                )
            if not -90 <= self.latitude <= 90:
                raise ValueError(f"latitude out of range: {self.latitude}")
            if not -180 <= self.longitude <= 180:
                raise ValueError(f"longitude out of range: {self.longitude}")
            if self.buffer_metres <= 0:
                raise ValueError(f"buffer_metres must be positive: {self.buffer_metres}")
        return self


class AOISample(_StrictBase):
    strategy: Literal["random_patches", "grid"] = "random_patches"
    n_patches: int = Field(default=50, ge=1)
    patch_size_pixels: int = Field(default=256, ge=16)
    resolution_metres: float = Field(default=10.0, gt=0.0)

    @field_validator("patch_size_pixels")
    @classmethod
    def _multiple_of_16(cls, v: int) -> int:
        if v % 16 != 0:
            raise ValueError(
                f"patch_size_pixels must be a multiple of 16 (U-Net depth constraint); got {v}."
            )
        return v


class AOIDef(_StrictBase):
    name: str
    role: Literal["target", "training", "both"]
    source: AOISource
    buffer_metres: float = 0.0
    sample: AOISample | None = None
    force_split: Literal["train", "val", "test"] | None = None
    exclude_windows: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_role_sample(self) -> "AOIDef":
        if self.role == "target" and self.sample is not None:
            raise ValueError(
                f"AOI {self.name!r}: role 'target' must not specify a sample block."
            )
        if self.role in {"training", "both"} and self.sample is None:
            self.sample = AOISample()
        return self


class DateWindow(_StrictBase):
    start: date
    end: date
    label: str
    role: Literal["training", "inference"] = "training"

    @model_validator(mode="after")
    def _start_before_end(self) -> "DateWindow":
        if self.start >= self.end:
            raise ValueError(
                f"DateWindow {self.label!r}: start ({self.start}) must be before end ({self.end})."
            )
        return self


# --------------------------------------------------------------------------- #
# Sentinel-1 / Sentinel-2 sections
# --------------------------------------------------------------------------- #

class Sentinel2Section(_StrictBase):
    source: Literal["gee_sr_harmonized"] = "gee_sr_harmonized"
    product: Literal["L2A"] = "L2A"
    collection_id: str = "COPERNICUS/S2_SR_HARMONIZED"
    max_aoi_cloud_cover_percent: float = Field(default=8.0, ge=0.0, le=100.0)
    bands: list[str] = Field(default_factory=lambda: ["B02", "B03", "B04", "B08", "B11", "B12"])
    derived_indices: list[str] = Field(default_factory=lambda: ["NDVI", "NBR", "NDWI"])
    resolution_metres: float = Field(default=10.0, gt=0.0)

    @field_validator("bands")
    @classmethod
    def _check_bands(cls, v: list[str]) -> list[str]:
        bad = [b for b in v if b not in ALLOWED_S2_BANDS]
        if bad:
            raise ValueError(f"Unknown S2 bands: {bad}; allowed are {sorted(ALLOWED_S2_BANDS)}")
        return v

    @field_validator("derived_indices")
    @classmethod
    def _check_indices(cls, v: list[str]) -> list[str]:
        bad = [i for i in v if i not in ALLOWED_INDICES]
        if bad:
            raise ValueError(
                f"Unknown derived indices: {bad}; allowed are {sorted(ALLOWED_INDICES)}"
            )
        return v


class Sentinel1Section(_StrictBase):
    source: Literal["gee_grd_calibrated"] = "gee_grd_calibrated"
    collection_id: str = "COPERNICUS/S1_GRD"
    polarisations: list[str] = Field(default_factory=lambda: ["VV", "VH"])
    orbit: Literal["any", "ascending", "descending"] = "any"
    resolution_metres: float = Field(default=10.0, gt=0.0)

    @field_validator("polarisations")
    @classmethod
    def _check_pols(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("polarisations must not be empty.")
        bad = [p for p in v if p not in ALLOWED_POLARISATIONS]
        if bad:
            raise ValueError(
                f"Unknown polarisations: {bad}; allowed are {sorted(ALLOWED_POLARISATIONS)}"
            )
        return v


# --------------------------------------------------------------------------- #
# GEE-specific blocks
# --------------------------------------------------------------------------- #

class SpeckleFilter(_StrictBase):
    enabled: bool = True
    method: Literal["lee", "refined_lee", "none"] = "lee"
    window: int = Field(default=5, ge=3)

    @field_validator("window")
    @classmethod
    def _odd_window(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError(f"speckle_filter.window must be odd; got {v}.")
        return v


class CalibrationSection(_StrictBase):
    """S1 GRD calibration recipe.

    Defaults follow the canonical recipe from Mullissa et al. 2021 and the
    community ``adugnag/gee_s1_ard`` reference implementation, which is what
    Microsoft Planetary Computer's RTC product is benchmarked against.
    """

    thermal_noise_removal: bool = True
    border_noise_correction: bool = True
    radiometric_target: Literal["gamma_naught", "sigma_naught"] = "gamma_naught"
    terrain_correction: bool = True
    terrain_dem: str = "USGS/SRTMGL1_003"
    speckle_filter: SpeckleFilter = Field(default_factory=SpeckleFilter)
    output: Literal["dB", "linear"] = "dB"


class PairingSection(_StrictBase):
    max_temporal_separation_days: int = Field(default=3, ge=0)
    prefer_same_orbit: bool = True
    require_same_relative_orbit: bool = False


class StorageSection(_StrictBase):
    format: Literal["tfrecord", "geotiff"] = "tfrecord"
    patch_size_pixels: int = Field(default=256, ge=16)
    patch_overlap_pixels: int = Field(default=128, ge=0)
    manifest_path: Path = Field(default=Path("manifest.csv"))

    @field_validator("patch_size_pixels")
    @classmethod
    def _multiple_of_16(cls, v: int) -> int:
        if v % 16 != 0:
            raise ValueError(
                f"patch_size_pixels must be a multiple of 16; got {v}."
            )
        return v


class ExportSection(_StrictBase):
    patches_subprefix: str = "patches"
    shards_per_aoi_window: int | Literal["auto"] = "auto"
    max_concurrent_tasks: int = Field(default=20, ge=1)
    poll_interval_seconds: int = Field(default=30, ge=5)
    # Per-origin export submits one task per (pair, origin). For an
    # operational harvest with hundreds of pairs that's a few thousand
    # tasks — they are submitted in chunks with a brief pause between
    # to keep GEE's queue from saturating.
    task_submit_chunk_size: int = Field(default=50, ge=1)
    task_submit_pause_seconds: float = Field(default=5.0, ge=0.0)


class TrainingSplitSection(_StrictBase):
    validation_split: float = Field(default=0.2, ge=0.0, lt=1.0)
    test_split: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _splits_sum(self) -> "TrainingSplitSection":
        if self.validation_split + self.test_split >= 1.0:
            raise ValueError(
                "validation_split + test_split must leave room for training; got "
                f"{self.validation_split} + {self.test_split}."
            )
        return self


class InferenceTarget(_StrictBase):
    aoi: str
    date_window: str


class InferenceSection(_StrictBase):
    output_subprefix: str = "predictions"
    derive_indices_from_predicted_reflectance: bool = True
    apply_to: list[InferenceTarget] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Top-level Config
# --------------------------------------------------------------------------- #

class Config(_StrictBase):
    project: ProjectSection
    aois: Annotated[list[AOIDef], Field(min_length=1)]
    date_windows: Annotated[list[DateWindow], Field(min_length=1)]
    sentinel2: Sentinel2Section = Field(default_factory=Sentinel2Section)
    sentinel1: Sentinel1Section = Field(default_factory=Sentinel1Section)
    calibration: CalibrationSection = Field(default_factory=CalibrationSection)
    pairing: PairingSection = Field(default_factory=PairingSection)
    storage: StorageSection = Field(default_factory=StorageSection)
    export: ExportSection = Field(default_factory=ExportSection)
    training_split: TrainingSplitSection = Field(default_factory=TrainingSplitSection)
    inference: InferenceSection = Field(default_factory=InferenceSection)

    @model_validator(mode="after")
    def _cross_check(self) -> "Config":
        aoi_names = {a.name for a in self.aois}
        window_labels = {w.label for w in self.date_windows}
        for target in self.inference.apply_to:
            if target.aoi not in aoi_names:
                raise ValueError(
                    f"inference.apply_to references unknown AOI {target.aoi!r}; "
                    f"known AOIs are {sorted(aoi_names)}."
                )
            if target.date_window not in window_labels:
                raise ValueError(
                    f"inference.apply_to references unknown date_window "
                    f"{target.date_window!r}; known windows are {sorted(window_labels)}."
                )
        # exclude_windows references must resolve
        for aoi in self.aois:
            for w in aoi.exclude_windows:
                if w not in window_labels:
                    raise ValueError(
                        f"AOI {aoi.name!r}: exclude_windows references unknown "
                        f"date_window {w!r}; known windows are {sorted(window_labels)}."
                    )
        return self

    def aoi_by_name(self, name: str) -> AOIDef:
        for a in self.aois:
            if a.name == name:
                return a
        raise KeyError(name)

    def window_by_label(self, label: str) -> DateWindow:
        for w in self.date_windows:
            if w.label == label:
                return w
        raise KeyError(label)


def load_config(path: Path) -> Config:
    """Load and validate a YAML config from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping; got {type(raw).__name__}.")
    return Config.model_validate(raw)
