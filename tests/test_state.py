"""Resumable-harvest state tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gee_s1s2 import state as state_mod


def test_load_state_missing_file_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    assert state_mod.load_state(p) == {}


def test_load_state_corrupt_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    p.write_text("{not json}")
    assert state_mod.load_state(p) == {}


def test_record_then_save_then_reload(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    state: dict[str, dict] = {}
    state_mod.record_submitted(
        state, "pair-A", 0,
        task_id="ABC123", file_prefix="prefix/A", split="train",
    )
    state_mod.save_state(state, p)

    reloaded = state_mod.load_state(p)
    assert "pair-A::0" in reloaded
    entry = reloaded["pair-A::0"]
    assert entry["task_id"] == "ABC123"
    assert entry["file_prefix"] == "prefix/A"
    assert entry["split"] == "train"
    assert "submitted_at" in entry


def test_has_submitted_distinguishes_pair_and_origin() -> None:
    state: dict[str, dict] = {}
    state_mod.record_submitted(state, "pair-A", 0, "T1", "prefix/A0", "train")
    assert state_mod.has_submitted(state, "pair-A", 0)
    assert not state_mod.has_submitted(state, "pair-A", 1)
    assert not state_mod.has_submitted(state, "pair-B", 0)


def test_save_state_is_overwrite_not_append(tmp_path: Path) -> None:
    """A second save_state should fully overwrite — no stale entries hang around."""
    p = tmp_path / "state.json"
    s1 = {"pair-A::0": {"task_id": "T1", "submitted_at": "2026-01-01",
                         "file_prefix": "p", "split": "train"}}
    state_mod.save_state(s1, p)
    s2 = {"pair-B::3": {"task_id": "T2", "submitted_at": "2026-02-02",
                         "file_prefix": "q", "split": "val"}}
    state_mod.save_state(s2, p)
    reloaded = state_mod.load_state(p)
    assert "pair-A::0" not in reloaded
    assert "pair-B::3" in reloaded
