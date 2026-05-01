"""GCS-backed manifest with the same schema as the v2 PyTorch project.

Stored at ``gs://<bucket>/<prefix>/operational_v1/manifest.csv``. Idempotent
across runs: re-running the harvest never duplicates rows. Schema mirrors
``s1s2.manifest`` in the v2 project byte-for-byte so Phase 2 (training in
Colab) can read either manifest with the same loader.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

LOG = logging.getLogger(__name__)


MANIFEST_HEADER = [
    "pair_id",
    "aoi_name",
    "date_window",
    "s1_id",
    "s1_acquired",
    "s1_orbit",
    "s1_pol",
    "s2_id",
    "s2_acquired",
    "s2_aoi_cloud_pct",
    "separation_days",
    "patch_count",
    "split",
    "harvested_at",
    "status",
    "random_seed",
    "n_patches_budget",
    "sample_strategy",
    # GEE-specific extras (kept at the end so v2 readers can ignore them).
    "tfrecord_uri",
    "calibration_recipe",
    # Per-origin extras for the per-origin export. Phase 1 writes one
    # row per (pair, origin); each row's tfrecord_uri points to that
    # origin's TFRecord.
    "origin_index",
    "origin_y_cell",
    "origin_x_cell",
    "origin_utm_epsg",
    "origin_utm_x_m",
    "origin_utm_y_m",
    "origin_lon",
    "origin_lat",
    "task_id",
]


@dataclass
class ManifestRow:
    pair_id: str
    aoi_name: str
    date_window: str
    s1_id: str
    s1_acquired: str
    s1_orbit: str
    s1_pol: str
    s2_id: str
    s2_acquired: str
    s2_aoi_cloud_pct: float
    separation_days: float
    patch_count: int
    split: str
    harvested_at: str
    status: str
    random_seed: int = 0
    n_patches_budget: int = 0
    sample_strategy: str = ""
    tfrecord_uri: str = ""
    calibration_recipe: str = "mullissa2021/gee_s1_ard:gamma_naught_volumetric_lee5"
    origin_index: int = 0
    origin_y_cell: int = 0
    origin_x_cell: int = 0
    origin_utm_epsg: int = 0
    origin_utm_x_m: float = 0.0
    origin_utm_y_m: float = 0.0
    origin_lon: float = 0.0
    origin_lat: float = 0.0
    task_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {k: ("" if v is None else str(v)) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestRow":
        return cls(
            pair_id=d["pair_id"],
            aoi_name=d["aoi_name"],
            date_window=d.get("date_window", ""),
            s1_id=d["s1_id"],
            s1_acquired=d["s1_acquired"],
            s1_orbit=d.get("s1_orbit", ""),
            s1_pol=d.get("s1_pol", ""),
            s2_id=d["s2_id"],
            s2_acquired=d["s2_acquired"],
            s2_aoi_cloud_pct=float(d.get("s2_aoi_cloud_pct") or 0),
            separation_days=float(d.get("separation_days") or 0),
            patch_count=int(d.get("patch_count") or 0),
            split=d.get("split", ""),
            harvested_at=d.get("harvested_at", ""),
            status=d.get("status", ""),
            random_seed=int(d.get("random_seed") or 0),
            n_patches_budget=int(d.get("n_patches_budget") or 0),
            sample_strategy=d.get("sample_strategy", ""),
            tfrecord_uri=d.get("tfrecord_uri", ""),
            calibration_recipe=d.get("calibration_recipe", ""),
            origin_index=int(d.get("origin_index") or 0),
            origin_y_cell=int(d.get("origin_y_cell") or 0),
            origin_x_cell=int(d.get("origin_x_cell") or 0),
            origin_utm_epsg=int(d.get("origin_utm_epsg") or 0),
            origin_utm_x_m=float(d.get("origin_utm_x_m") or 0),
            origin_utm_y_m=float(d.get("origin_utm_y_m") or 0),
            origin_lon=float(d.get("origin_lon") or 0),
            origin_lat=float(d.get("origin_lat") or 0),
            task_id=d.get("task_id", ""),
        )


class GcsManifest:
    """File-backed manifest stored as a single CSV blob in GCS."""

    def __init__(self, gcs_client: Any, bucket: str, blob_path: str) -> None:
        self._client = gcs_client
        self._bucket_name = bucket
        self._blob_path = blob_path
        self._rows: list[ManifestRow] = []
        self._load()

    def _blob(self) -> Any:
        return self._client.bucket(self._bucket_name).blob(self._blob_path)

    def _load(self) -> None:
        blob = self._blob()
        if not blob.exists():
            LOG.info(
                "Manifest does not exist yet at gs://%s/%s; will be created on flush.",
                self._bucket_name, self._blob_path,
            )
            return
        text = blob.download_as_text()
        reader = csv.DictReader(io.StringIO(text))
        self._rows = [ManifestRow.from_dict(d) for d in reader]
        LOG.info("Loaded %d manifest rows from gs://%s/%s",
                 len(self._rows), self._bucket_name, self._blob_path)

    def _flush(self) -> None:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        for row in self._rows:
            writer.writerow(row.to_dict())
        self._blob().upload_from_string(buf.getvalue(), content_type="text/csv")
        LOG.debug("Flushed %d rows to gs://%s/%s",
                  len(self._rows), self._bucket_name, self._blob_path)

    @property
    def rows(self) -> list[ManifestRow]:
        return list(self._rows)

    def existing_keys(self) -> set[tuple[str, str, str, int]]:
        """Dedup key now includes origin_index — one row per (pair, origin)."""
        return {(r.s1_id, r.s2_id, r.aoi_name, r.origin_index) for r in self._rows}

    def add(self, row: ManifestRow) -> bool:
        key = (row.s1_id, row.s2_id, row.aoi_name, row.origin_index)
        if key in self.existing_keys():
            LOG.info("Manifest dedup: skipping existing key %s", key)
            return False
        self._rows.append(row)
        self._flush()
        return True

    def summary(self) -> dict:
        out: dict = {
            "total": len(self._rows),
            "by_aoi": {},
            "by_window": {},
            "by_split": {},
        }
        for r in self._rows:
            out["by_aoi"][r.aoi_name] = out["by_aoi"].get(r.aoi_name, 0) + 1
            out["by_window"][r.date_window] = out["by_window"].get(r.date_window, 0) + 1
            out["by_split"][r.split] = out["by_split"].get(r.split, 0) + 1
        return out


def now_isoformat() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
