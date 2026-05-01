"""Persistent state for resumable harvest.

The harvest may submit thousands of per-origin export tasks. Each task
takes minutes to clear GEE's queue, so a full operational harvest is
hours of wall-clock. We persist task IDs + their (pair, origin) keys
to a JSON file so that an interrupted run can be resumed without
re-submitting work that's already done or in flight.

State key schema: ``"{pair_id}::{origin_index}"`` → entry::

    {
        "task_id":      "<GEE task id>",
        "submitted_at": "<ISO timestamp UTC>",
        "file_prefix":  "<gs:// path prefix written by the task>",
        "split":        "train" | "val" | "test",
    }

The file lives in the working directory by default
(``./.gee_s1s2_harvest_state.json``). Move/delete it to force a
re-submit; the AOI cloud check and pair manifest still dedup separately.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path(".gee_s1s2_harvest_state.json")


def _key(pair_id: str, origin_index: int) -> str:
    return f"{pair_id}::{origin_index}"


def load_state(path: Path = DEFAULT_STATE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOG.warning(
            "State file %s is corrupt (%s); ignoring and starting fresh.",
            path, exc,
        )
        return {}


def save_state(state: dict[str, dict], path: Path = DEFAULT_STATE_PATH) -> None:
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def has_submitted(state: dict[str, dict], pair_id: str, origin_index: int) -> bool:
    return _key(pair_id, origin_index) in state


def record_submitted(
    state: dict[str, dict],
    pair_id: str,
    origin_index: int,
    task_id: str,
    file_prefix: str,
    split: str,
) -> None:
    state[_key(pair_id, origin_index)] = {
        "task_id": task_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_prefix": file_prefix,
        "split": split,
    }
