"""TFRecord export of paired S1/S2 patches to Google Cloud Storage.

Phase 1 ships **per-origin export**: for each accepted scene-pair, the
sampler chooses ``n_patches`` random origins (with the v2 min-spacing
rule), and one ``Export.image.toCloudStorage`` task is submitted per
origin. This replaces an earlier patchDimensions-based grid-tiling
approach that dominated training data with whatever site had the
largest polygon — a problem we already fixed in v2 by switching to
random sampling, and that needed re-fixing for the GEE port so Sonia's
operational pipeline behaves the same way as the validated v2 design.

Three operational concerns are handled here:

* **Rate limiting.** GEE's noncommercial concurrent-task quota is
  ~10–30. We submit in small chunks and pause between chunks
  (:func:`submit_in_chunks`) so the queue never goes over a configured
  ceiling.
* **Retry with exponential backoff.** Transient HTTP 429/5xx from the
  Earth Engine compute endpoint are retried with the same scheme as the
  v2 STAC client (:func:`submit_one_with_retry`).
* **Resumable state.** Each submission writes its task ID and file
  prefix to ``state.py``'s persistence layer, keyed on
  ``(pair_id, origin_index)``, so re-running ``gee_s1s2 harvest`` after
  an interrupt skips work already in flight.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from .aois import slugify
from .config import Config

LOG = logging.getLogger(__name__)


@dataclass
class ExportTask:
    """Lightweight wrapper around an ``ee.batch.Task``."""
    task: Any
    description: str
    bucket: str
    file_prefix: str
    pair_id: str
    origin_index: int

    @property
    def state(self) -> str:
        return self.task.status().get("state", "UNKNOWN")

    @property
    def task_id(self) -> str:
        return self.task.status().get("id", "")

    def wait(self, poll_interval_seconds: int) -> str:
        """Block until the task reaches a terminal state. Return final state."""
        while True:
            state = self.state
            if state in {"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"}:
                return state
            time.sleep(poll_interval_seconds)


def stack_pair_image(
    s1_image: Any,
    s2_image: Any,
    s2_bands: list[str],
    s1_polarisations: list[str],
) -> Any:
    """Stack S1 (calibrated) and S2 reflectance into one image.

    Output band order is S1 polarisations followed by S2 reflectance bands,
    matching the v2 PyTorch project's training-tensor convention so that
    downstream code in Phase 2 can consume the TFRecord without remapping.
    """
    import ee  # noqa: PLC0415

    s1 = s1_image.select(s1_polarisations)
    # S2 reflectance is uint16 with a 10000 scaling factor; divide to [0, 1].
    s2 = s2_image.select(s2_bands).toFloat().divide(10000.0)
    return s1.addBands(s2)


def _origin_filename_suffix(y_cell: int, x_cell: int) -> str:
    """Filename component encoding the origin cell index."""
    return f"y{y_cell:05d}_x{x_cell:05d}"


def submit_one_with_retry(
    submit_fn: Callable[[], Any],
    *,
    max_attempts: int = 5,
    base_delay_s: float = 1.5,
    max_delay_s: float = 30.0,
) -> Any:
    """Submit a GEE task with exponential-backoff retry on transient errors.

    ``submit_fn`` must be a no-arg callable that returns the GEE task
    object (or raises). 4xx errors that are NOT 429 are non-retryable.
    Mirrors the v2 STAC retry-with-backoff pattern.
    """
    delay = base_delay_s
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return submit_fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            transient = (
                "429" in msg
                or "Too many" in msg
                or "concurrent" in msg
                or "503" in msg or "502" in msg or "500" in msg
                or "Internal error" in msg
            )
            if not transient or attempt == max_attempts:
                raise
            jitter = random.uniform(0, delay * 0.25)
            wait = min(delay + jitter, max_delay_s)
            LOG.warning(
                "Submit failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt, max_attempts, msg[:120], wait,
            )
            time.sleep(wait)
            delay = min(delay * 2, max_delay_s)
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("submit_one_with_retry exhausted retries with no exception")


def export_one_patch(
    image: Any,
    patch_geom: Any,                # ee.Geometry.Rectangle for the 256m × 256m sub-region
    aoi_name: str,
    window_label: str,
    pair_id: str,
    origin_index: int,
    origin_y_cell: int,
    origin_x_cell: int,
    config: Config,
    bucket: str,
    prefix: str,
    split: str,
) -> ExportTask:
    """Submit one TFRecord export task for one random patch origin."""
    import ee  # noqa: PLC0415

    aoi_slug = slugify(aoi_name)
    window_slug = slugify(window_label)
    pair_suffix = slugify(pair_id.rsplit("::", 1)[-1])
    origin_suffix = _origin_filename_suffix(origin_y_cell, origin_x_cell)
    file_prefix = (
        f"{prefix}/operational_v1/patches/{split}/{aoi_slug}/"
        f"{aoi_slug}__{window_slug}__{pair_suffix}__{origin_suffix}"
    )
    description = (f"gee_s1s2 {aoi_slug}__{window_slug} {pair_suffix} {origin_suffix}")[:100]

    def _build_and_start() -> Any:
        # Each task is a single 256×256 patch — no patchDimensions tiling.
        # GEE writes the result as one TFRecord with one example.
        task = ee.batch.Export.image.toCloudStorage(
            image=image.clip(patch_geom),
            description=description,
            bucket=bucket,
            fileNamePrefix=file_prefix,
            scale=config.sentinel2.resolution_metres,
            region=patch_geom,
            fileFormat="TFRecord",
            formatOptions={
                "patchDimensions": [
                    config.storage.patch_size_pixels,
                    config.storage.patch_size_pixels,
                ],
                "kernelSize": [1, 1],
                "compressed": True,
            },
            maxPixels=int(1e9),
        )
        task.start()
        return task

    task = submit_one_with_retry(_build_and_start)
    return ExportTask(
        task=task,
        description=description,
        bucket=bucket,
        file_prefix=file_prefix,
        pair_id=pair_id,
        origin_index=origin_index,
    )


def submit_in_chunks(
    submit_callables: list[Callable[[], ExportTask]],
    *,
    chunk_size: int = 50,
    pause_between_chunks_s: float = 5.0,
    max_active_tasks: int = 50,
) -> list[ExportTask]:
    """Submit a list of export tasks in chunks, throttling against the
    GEE concurrent-task ceiling.

    ``submit_callables`` is a list of zero-arg functions, each of which
    submits a single task and returns the resulting :class:`ExportTask`.
    We process them in batches of ``chunk_size``, waiting for the active
    task count to drop below ``max_active_tasks`` between chunks.

    The chunked-submit pattern keeps GEE's noncommercial concurrent-task
    quota (~10–30 active at any time on a freshly registered project)
    from being overrun, while still exploiting whatever headroom the
    user's quota allows.
    """
    submitted: list[ExportTask] = []
    n = len(submit_callables)
    for chunk_start in range(0, n, chunk_size):
        chunk = submit_callables[chunk_start: chunk_start + chunk_size]
        for fn in chunk:
            submitted.append(fn())
        if chunk_start + chunk_size < n:
            _wait_for_queue_room(max_active_tasks, pause_between_chunks_s)
    return submitted


def _wait_for_queue_room(max_active_tasks: int, base_pause_s: float) -> None:
    """Block until the GEE active-task count is below ``max_active_tasks``."""
    import ee  # noqa: PLC0415

    pause = base_pause_s
    while True:
        try:
            tasks = ee.batch.Task.list()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Could not list tasks (%s); pausing %.1fs", exc, pause)
            time.sleep(pause)
            pause = min(pause * 1.5, 30.0)
            continue
        active = sum(
            1 for t in tasks
            if t.status().get("state") in {"READY", "RUNNING"}
        )
        if active < max_active_tasks:
            return
        LOG.info(
            "GEE has %d active tasks (ceiling %d); waiting %.1fs.",
            active, max_active_tasks, pause,
        )
        time.sleep(pause)
        pause = min(pause * 1.2, 30.0)


def open_tfrecord_local(path: str, bands: list[str]) -> Any:
    """Open a downloaded TFRecord shard and yield decoded patch tensors.

    Used by :mod:`tests` and the notebook walkthrough to verify that
    exported patches have the expected shape. Imports ``tensorflow``
    lazily so installations without TF still get a useful error message.
    """
    try:
        import tensorflow as tf  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Reading TFRecord requires tensorflow. Install with "
            "'pip install tensorflow' or 'pip install gee-s1s2-translator[notebook]'."
        ) from exc

    feature_spec = {
        b: tf.io.FixedLenFeature([], tf.string) for b in bands
    }

    def _decode(rec: "tf.train.Example") -> dict[str, "tf.Tensor"]:
        parsed = tf.io.parse_single_example(rec, feature_spec)
        return {
            b: tf.io.parse_tensor(parsed[b], out_type=tf.float32) for b in bands
        }

    ds = tf.data.TFRecordDataset(path, compression_type="GZIP")
    return ds.map(_decode)
