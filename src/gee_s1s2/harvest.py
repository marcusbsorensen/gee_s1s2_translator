"""Top-level harvest orchestration.

Wires the catalogue, calibration, pairing, filtering, sampling, export,
and manifest modules together. The CLI's ``harvest`` command calls
:func:`run_harvest` directly. Two-pass per (AOI, window) bucket:

1. Search S1 + S2, find pairs, AOI cloud check, collect accepted pairs.
2. Distribute the bucket's ``n_patches`` budget across accepted pairs;
   for each pair, sample n random origins (min-spacing rule from
   ``sampling.py``) and submit one TFRecord export task per origin
   with rate-limiting + retry/backoff. State is persisted via
   :mod:`gee_s1s2.state` so an interrupted run can be resumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .aois import aoi_geometry, shapely_to_ee, slugify
from .auth import get_gcs_bucket, get_gcs_client, get_gcs_prefix
from .calibration import calibrate_grd_collection
from .catalogue import s1_grd_collection, s2_sr_collection
from .config import Config
from .filtering import aoi_cloud_pct
from .manifest import GcsManifest, ManifestRow, now_isoformat
from .pairing import find_pairs, materialise_s1, materialise_s2
from .sampling import (
    CoarseAoiGrid,
    coarse_aoi_mask,
    distribute_budget,
    origins_to_patch_geoms,
    patchspec_to_origin_meta,
    per_scene_seed,
    sample_origins,
)
from . import state as state_mod

LOG = logging.getLogger(__name__)


@dataclass
class HarvestSummary:
    candidates: int = 0
    new_pairs: int = 0
    accepted_after_cloud: int = 0
    written_to_manifest: int = 0
    patches_written: int = 0
    tasks_submitted: int = 0
    tasks_skipped_resume: int = 0
    per_bucket: dict[tuple[str, str], int] = field(default_factory=dict)
    skipped_excluded: list[tuple[str, str]] = field(default_factory=list)


def _stratified_split(
    aoi_name: str,
    window_label: str,
    aoi_window_labels: list[str],
    val_frac: float,
    test_frac: float,
) -> str:
    """Same per-AOI stratified split as the v2 project."""
    import math
    sorted_windows = sorted(aoi_window_labels)
    n = len(sorted_windows)
    if n == 0 or window_label not in sorted_windows:
        return "train"
    n_test = max(1, math.ceil(test_frac * n)) if n >= 2 else 0
    n_val = max(1, math.ceil(val_frac * n)) if n >= 3 else 0
    n_train = max(0, n - n_test - n_val)
    idx = sorted_windows.index(window_label)
    if idx < n_train:
        return "train"
    if idx < n_train + n_val:
        return "val"
    return "test"


def run_harvest(
    config: Config,
    dry_run: bool = False,
    only_aoi: str | None = None,
    only_window: str | None = None,
    include_inference_windows: bool = False,
) -> HarvestSummary:
    """Run the harvest. Returns a :class:`HarvestSummary`.

    ``include_inference_windows`` defaults to False so the standard training
    harvest skips windows declared with ``role: inference`` (the original
    behaviour). The post-fire 2022 cloud-penetration extension flips this to
    True so it can harvest the inference-only window into the same patch
    layout as everything else.
    """
    summary = HarvestSummary()

    if not dry_run:
        gcs_client = get_gcs_client()
        bucket = get_gcs_bucket()
        prefix = get_gcs_prefix()
        manifest_blob = f"{prefix}/operational_v1/{config.storage.manifest_path}"
        manifest = GcsManifest(gcs_client, bucket, manifest_blob)
    else:
        manifest = None
        bucket = ""
        prefix = ""

    # Window labels available for training (non-inference).
    training_windows = [w.label for w in config.date_windows if w.role != "inference"]

    # Per-AOI list of windows for stratified split (after exclude).
    for aoi in config.aois:
        if aoi.role == "target":
            continue
        if only_aoi and aoi.name != only_aoi:
            continue

        try:
            geom_shapely = aoi_geometry(aoi)
            geom_ee = shapely_to_ee(geom_shapely)
        except Exception as exc:  # noqa: BLE001
            LOG.error("AOI %r: failed to load geometry: %s", aoi.name, exc)
            continue

        aoi_training_windows = [
            w for w in training_windows if w not in (aoi.exclude_windows or [])
        ]

        for window in config.date_windows:
            if window.role == "inference" and not include_inference_windows:
                continue
            if only_window and window.label != only_window:
                continue
            if window.label in (aoi.exclude_windows or []):
                summary.skipped_excluded.append((aoi.name, window.label))
                LOG.info(
                    "Skipping (AOI, window) per exclude_windows: %r x %r",
                    aoi.name, window.label,
                )
                continue

            LOG.info("Harvest AOI=%r window=%r", aoi.name, window.label)
            try:
                s2_coll = s2_sr_collection(
                    config.sentinel2, geom_ee, window.start, window.end,
                    cloud_cover_override_pct=window.cloud_cover_override_pct,
                )
                s1_raw = s1_grd_collection(config.sentinel1, geom_ee, window.start, window.end)
                s1_coll = calibrate_grd_collection(
                    s1_raw, config.calibration, config.sentinel1.polarisations,
                )
                # Pull metadata from the RAW S1 collection. Materialising
                # the calibrated collection forces GEE to evaluate the full
                # Mullissa pipeline per scene just to read system:time_start,
                # which blows past the noncommercial concurrent-aggregation
                # quota and 429s every search.
                s1_items = materialise_s1(s1_raw)
                s2_items = materialise_s2(s2_coll)
            except Exception as exc:  # noqa: BLE001
                LOG.error(
                    "STAC search failed for AOI %r window %r: %s. Skipping.",
                    aoi.name, window.label, exc,
                )
                continue

            pairs = find_pairs(aoi.name, s1_items, s2_items, config.pairing)
            summary.per_bucket[(aoi.name, window.label)] = len(pairs)
            summary.candidates += len(pairs)

            # Dedup against manifest (skip in dry-run; nothing to dedup against).
            if manifest is not None:
                existing = manifest.existing_keys()
                pairs = [p for p in pairs if (p.s1.id, p.s2.id, aoi.name) not in existing]
            summary.new_pairs += len(pairs)

            # Pass 1: AOI cloud check.
            accepted: list[tuple] = []
            for pair in pairs:
                # NOTE: re-fetching s2 image by id; pairing.py left image_obj=None.
                # Use the original collection: filter by system:index.
                try:
                    import ee  # noqa: PLC0415
                    s2_img = (
                        s2_coll.filter(ee.Filter.eq("system:index", pair.s2.id)).first()
                    )
                    cloud_pct = aoi_cloud_pct(s2_img, geom_ee)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(
                        "Cloud check failed for %s: %s; treating as 100%% cloud.",
                        pair.s2.id, exc,
                    )
                    cloud_pct = 100.0
                cloud_limit = (
                    window.cloud_cover_override_pct
                    if window.cloud_cover_override_pct is not None
                    else config.sentinel2.max_aoi_cloud_cover_percent
                )
                ok = cloud_pct <= cloud_limit
                LOG.info(
                    "AOI cloud check: %s %.1f%% (limit %.1f%%%s) %s",
                    pair.s2.id, cloud_pct, cloud_limit,
                    " override" if window.cloud_cover_override_pct is not None else "",
                    "ACCEPT" if ok else "REJECT",
                )
                if ok:
                    accepted.append((pair, cloud_pct))
            summary.accepted_after_cloud += len(accepted)

            if dry_run or not accepted:
                continue

            # Pass 2: distribute n_patches across accepted pairs, sample
            # random origins per pair (min-spacing rule), submit one
            # export task per origin with rate-limiting + state.
            from . import export as export_mod  # noqa: PLC0415

            n_patches_total = aoi.sample.n_patches if aoi.sample else 0
            per_pair_budgets = distribute_budget(n_patches_total, len(accepted))
            split = (
                aoi.force_split if aoi.force_split is not None
                else _stratified_split(
                    aoi.name, window.label, aoi_training_windows,
                    config.training_split.validation_split,
                    config.training_split.test_split,
                )
            )

            # Build the coarse AOI mask once per (AOI, window). The mask
            # is the same regardless of pair, since the AOI polygon
            # doesn't change.
            patch_size_pixels = config.storage.patch_size_pixels
            pixel_size_m = config.sentinel2.resolution_metres
            # coarse_step ≈ patch / 8 so we have ~8 cells per patch dimension —
            # enough resolution to land patches on different parts of a
            # large AOI but coarse enough to keep mask memory tiny even for
            # 100 km² AOIs.
            coarse_step_m = max(50.0, patch_size_pixels * pixel_size_m / 8.0)
            grid = coarse_aoi_mask(geom_shapely, coarse_step_m=coarse_step_m)
            patch_size_cells = max(1, int(np.ceil(
                patch_size_pixels * pixel_size_m / coarse_step_m
            )))
            # Use the configured patch_overlap_pixels: same overlap budget
            # as the v2 PyTorch project (default 128 px = 50 % overlap), so
            # the min-spacing rule lets the sampler fit multiple origins
            # in a moderate-sized AOI.
            overlap_pixels = config.storage.patch_overlap_pixels
            overlap_cells = max(0, int(np.floor(
                overlap_pixels * pixel_size_m / coarse_step_m
            )))
            LOG.info(
                "AOI %r grid: H=%d W=%d cells at step %.1f m; "
                "patch_size_cells=%d (10 m patch=%d px = %.0f m)",
                aoi.name, grid.H, grid.W, grid.coarse_step_m,
                patch_size_cells, patch_size_pixels,
                patch_size_pixels * pixel_size_m,
            )

            # Resume state — keyed on (pair_id, origin_index).
            state = state_mod.load_state()

            for (pair, cloud_pct), budget in zip(accepted, per_pair_budgets, strict=True):
                if budget <= 0:
                    continue
                seed = per_scene_seed(config.project.random_seed, pair.pair_id)
                rng = np.random.default_rng(seed)
                # Trivial validity masks: per-pixel S1/S2 validity isn't
                # checked client-side in the GEE port (scene-level cloud
                # check happens in filtering.aoi_cloud_pct already). Pass
                # all-ones so sample_origins falls through to the
                # min-spacing + AOI-mask checks only.
                ones = np.ones_like(grid.mask, dtype=bool)
                origins = sample_origins(
                    H=grid.H, W=grid.W,
                    patch_size=patch_size_cells, overlap=overlap_cells,
                    s2_valid_mask=ones, s1_valid_mask=ones,
                    aoi_mask=grid.mask,
                    n_patches=budget, rng=rng,
                )
                if not origins:
                    LOG.warning(
                        "Sampler returned 0 origins for pair %s (budget=%d); skipping.",
                        pair.pair_id, budget,
                    )
                    continue
                patches = origins_to_patch_geoms(
                    origins=origins, grid=grid,
                    pixel_size_m=pixel_size_m,
                    patch_size_pixels=patch_size_pixels,
                    pair_id=pair.pair_id,
                )

                import ee  # noqa: PLC0415
                s1_img = s1_coll.filter(ee.Filter.eq("system:index", pair.s1.id)).first()
                s2_img = s2_coll.filter(ee.Filter.eq("system:index", pair.s2.id)).first()
                stacked = export_mod.stack_pair_image(
                    s1_img, s2_img,
                    config.sentinel2.bands,
                    config.sentinel1.polarisations,
                )

                # Batch the per-origin submissions for this pair so we
                # respect the GEE concurrent-task ceiling. ``submit_in_chunks``
                # also pauses between chunks to give the queue room.
                def _make_callable(ps, _stacked=stacked, _pair=pair, _cloud=cloud_pct,
                                   _seed=seed, _budget=budget):
                    def _do():
                        task = export_mod.export_one_patch(
                            image=_stacked,
                            patch_geom=ps.geometry,
                            aoi_name=_pair.aoi_name,
                            window_label=window.label,
                            pair_id=_pair.pair_id,
                            origin_index=ps.index,
                            origin_y_cell=ps.y0,
                            origin_x_cell=ps.x0,
                            config=config,
                            bucket=bucket,
                            prefix=prefix,
                            split=split,
                        )
                        state_mod.record_submitted(
                            state, _pair.pair_id, ps.index,
                            task_id=task.task_id, file_prefix=task.file_prefix,
                            split=split,
                        )
                        meta = patchspec_to_origin_meta(ps, grid)
                        row = ManifestRow(
                            pair_id=_pair.pair_id,
                            aoi_name=_pair.aoi_name,
                            date_window=window.label,
                            s1_id=_pair.s1.id,
                            s1_acquired=_pair.s1.datetime_iso,
                            s1_orbit=_pair.s1.orbit_pass.lower() if _pair.s1.orbit_pass else "",
                            s1_pol=",".join(config.sentinel1.polarisations),
                            s2_id=_pair.s2.id,
                            s2_acquired=_pair.s2.datetime_iso,
                            s2_aoi_cloud_pct=round(float(_cloud), 3),
                            separation_days=round(_pair.separation_days, 3),
                            patch_count=1,
                            split=split,
                            harvested_at=now_isoformat(),
                            status="exporting",
                            random_seed=_seed,
                            n_patches_budget=_budget,
                            sample_strategy=(aoi.sample.strategy if aoi.sample else ""),
                            tfrecord_uri=f"gs://{bucket}/{task.file_prefix}.tfrecord.gz",
                            origin_index=meta["origin_index"],
                            origin_y_cell=meta["origin_y_cell"],
                            origin_x_cell=meta["origin_x_cell"],
                            origin_utm_epsg=meta["origin_utm_epsg"],
                            origin_utm_x_m=round(meta["origin_utm_x_m"], 1),
                            origin_utm_y_m=round(meta["origin_utm_y_m"], 1),
                            origin_lon=round(meta["origin_lon"], 6),
                            origin_lat=round(meta["origin_lat"], 6),
                            task_id=task.task_id,
                        )
                        if manifest.add(row):
                            summary.written_to_manifest += 1
                            summary.patches_written += 1
                        return task
                    return _do

                # Skip patches already submitted in a prior run.
                callables_for_pair = []
                for ps in patches:
                    if state_mod.has_submitted(state, pair.pair_id, ps.index):
                        summary.tasks_skipped_resume += 1
                        LOG.debug(
                            "Resume: skipping already-submitted (%s, origin=%d)",
                            pair.pair_id, ps.index,
                        )
                        continue
                    callables_for_pair.append(_make_callable(ps))

                if callables_for_pair:
                    submitted = export_mod.submit_in_chunks(
                        callables_for_pair,
                        chunk_size=config.export.task_submit_chunk_size,
                        pause_between_chunks_s=config.export.task_submit_pause_seconds,
                        max_active_tasks=config.export.max_concurrent_tasks,
                    )
                    summary.tasks_submitted += len(submitted)

                # Persist state after each pair so an interrupt mid-AOI
                # still leaves a usable resume checkpoint.
                state_mod.save_state(state)

    return summary
