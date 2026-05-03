"""Vertex AI Custom Job entry point: harvest the post-fire 2022 cloud-covered
window for the Brentmoor and Poors Allotment area-training AOIs.

This is the cloud-penetration use case — Sentinel-2 is largely unusable in
this window (autumn 2022, persistent cloud cover over southern UK), but
Sentinel-1 is weather-independent. The harvest accepts any S2 scene as the
pairing target (cloud_cover_override_pct=100 on this window in the YAML),
exports the S1+S2 patches to GCS, and writes manifest rows. Downstream
inference uses only the S1 channels; the S2 half is captured for
completeness but not used.

Auth: the Vertex worker runs as the dedicated ``gee-harvester`` service
account, which has ``roles/earthengine.writer`` and ``roles/storage.objectAdmin``
on the project. Earth Engine and GCS both authenticate via ADC from the
worker's metadata server.

Inputs (env vars):
    GEE_PROJECT_ID, GCS_BUCKET, GCS_PREFIX (read by ``gee_s1s2.auth``)
    Plus we resolve the AOI KMLs from a GCS staging area (uploaded once;
    the operational config references local paths that don't exist on the
    Vertex worker).
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

LOG = logging.getLogger("post_fire_harvest")

# AOIs and window we're restricting to.
TARGET_AOIS = ["Brentmoor area training", "Poors Allotment area training"]
TARGET_WINDOW = "post-fire 2022"

# KML files we need on the worker. They live on GCS at this prefix and
# the config will be patched to reference the local download paths.
KML_GCS_PREFIX = "gee_s1s2_translator/operational_v1/inputs/"
KML_FILES = ["SWT_MappedFires_20220911.kml", "TBH_SPA_Surrey_400m_Buffer_Dissolved.kml"]


def _stage_kmls(bucket_name: str, local_dir: Path) -> dict[str, Path]:
    """Download AOI KMLs from GCS to a local directory; return original->local map."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for name in KML_FILES:
        src_blob = bucket.blob(KML_GCS_PREFIX + name)
        local = local_dir / name
        src_blob.download_to_filename(str(local))
        mapping[name] = local
        LOG.info("Staged KML: gs://%s/%s%s -> %s", bucket_name, KML_GCS_PREFIX, name, local)
    return mapping


def _patch_config_kml_paths(config, kml_local_paths: dict[str, Path]) -> None:
    """Mutate config in place: rewrite each AOI's KML source path to the
    locally-staged copy (matched by basename)."""
    for aoi in config.aois:
        if aoi.source.type != "kml":
            continue
        original = Path(aoi.source.path)
        local = kml_local_paths.get(original.name)
        if local is None:
            LOG.warning("AOI %r: no local KML staged for %s; leaving original",
                        aoi.name, original)
            continue
        aoi.source.path = local
        LOG.info("AOI %r: KML path patched -> %s", aoi.name, local)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_id = os.environ.get("GEE_PROJECT_ID") or os.environ["GEE_S1S2_PROJECT_ID"]
    bucket_name = os.environ.get("GCS_BUCKET") or os.environ["GEE_S1S2_BUCKET"]
    gcs_prefix = os.environ.get("GCS_PREFIX") or os.environ.get(
        "GEE_S1S2_PREFIX", "gee_s1s2_translator")
    # gee_s1s2.auth reads these names; mirror in case the caller used the alt set.
    os.environ["GEE_PROJECT_ID"] = project_id
    os.environ["GCS_BUCKET"] = bucket_name
    os.environ["GCS_PREFIX"] = gcs_prefix

    LOG.info("project=%s bucket=%s prefix=%s", project_id, bucket_name, gcs_prefix)

    # --- Earth Engine init via ADC ---
    # The Vertex worker runs as gee-harvester@... which has roles/earthengine.writer.
    import ee
    import google.auth
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/earthengine",
                "https://www.googleapis.com/auth/cloud-platform"]
    )
    ee.Initialize(credentials=creds, project=project_id)
    LOG.info("Earth Engine initialised via ADC.")

    # --- Stage the AOI KML files locally ---
    workdir = Path(tempfile.mkdtemp(prefix="post_fire_harvest_"))
    kml_dir = workdir / "kmls"
    kml_local_paths = _stage_kmls(bucket_name, kml_dir)

    # --- Locate the operational config that ships with the package ---
    # We bundle config/operational_v1.yaml in the sdist; resolve it via the
    # gee_s1s2 package directory.
    import gee_s1s2
    pkg_root = Path(gee_s1s2.__file__).resolve().parent
    config_candidates = [
        pkg_root / "operational_v1.yaml",                 # package_data path
        pkg_root.parent.parent / "config" / "operational_v1.yaml",  # repo layout
        Path("/app/config/operational_v1.yaml"),          # alt container layout
    ]
    config_path = next((p for p in config_candidates if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(
            f"Could not find operational_v1.yaml. Tried: {config_candidates}"
        )
    LOG.info("Loading config from %s", config_path)

    # Pydantic validates KML paths at load time (AOISource has an `exists()`
    # check), so we patch the raw YAML text before validation: rewrite the
    # operational config's relative ``../s1s2-translator/inputs/<NAME>.kml``
    # paths to the absolute locally-staged KMLs.
    raw_yaml = config_path.read_text(encoding="utf-8")
    for fname, local in kml_local_paths.items():
        raw_yaml = raw_yaml.replace(
            f"../s1s2-translator/inputs/{fname}", str(local)
        )
    patched_path = workdir / "operational_v1_patched.yaml"
    patched_path.write_text(raw_yaml, encoding="utf-8")
    LOG.info("Wrote patched config: %s", patched_path)

    from .config import load_config
    config = load_config(patched_path)

    # --- Run the filtered harvest ---
    from .harvest import run_harvest
    LOG.info("Running harvest restricted to AOIs=%s window=%r ...",
             TARGET_AOIS, TARGET_WINDOW)

    summary = None
    for aoi_name in TARGET_AOIS:
        LOG.info("=== Harvest pass: AOI=%r window=%r ===", aoi_name, TARGET_WINDOW)
        s = run_harvest(
            config,
            dry_run=False,
            only_aoi=aoi_name,
            only_window=TARGET_WINDOW,
            include_inference_windows=True,
        )
        LOG.info("AOI %r summary: candidates=%d new_pairs=%d accepted_after_cloud=%d "
                 "tasks_submitted=%d patches_written=%d skipped_resume=%d",
                 aoi_name, s.candidates, s.new_pairs, s.accepted_after_cloud,
                 s.tasks_submitted, s.patches_written, s.tasks_skipped_resume)

    # All export tasks have been submitted to GEE asynchronously; the worker
    # must not exit until they complete (otherwise the inference job that
    # follows runs on missing data). Poll GEE's task list until no more
    # post-fire-2022 tasks are RUNNING / READY.
    LOG.info("All harvest tasks submitted. Polling GEE task queue for completion...")
    import time
    poll_interval_s = 60
    timeout_s = 4 * 3600  # 4 h hard cap
    deadline = time.time() + timeout_s
    while True:
        # Filter task list to ones we recognise from this run (description
        # prefix "gee_s1s2 " plus our window slug "post-fire-2022").
        all_tasks = ee.data.getTaskList()
        ours = [
            t for t in all_tasks
            if t.get("description", "").startswith("gee_s1s2 ")
            and "post-fire-2022" in t.get("description", "")
        ]
        active = [t for t in ours if t.get("state") in ("READY", "RUNNING")]
        completed = [t for t in ours if t.get("state") == "COMPLETED"]
        failed = [t for t in ours if t.get("state") in ("FAILED", "CANCELLED", "CANCEL_REQUESTED")]
        LOG.info(
            "Task queue: %d total, %d active, %d completed, %d failed/cancelled.",
            len(ours), len(active), len(completed), len(failed),
        )
        if not active:
            LOG.info("All harvest tasks reached a terminal state.")
            break
        if time.time() > deadline:
            LOG.warning("Wait timeout (%ds) exceeded with %d still-active tasks; "
                        "exiting anyway.", timeout_s, len(active))
            break
        time.sleep(poll_interval_s)

    LOG.info("Harvest complete. Worker terminating.")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
