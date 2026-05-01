"""Export-layer tests: retry/backoff and chunked submission."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from gee_s1s2 import export as export_mod


# ---------------------------------------------------------------------------
# submit_one_with_retry: transient vs permanent error handling
# ---------------------------------------------------------------------------

class _Boom(Exception):
    """Signals a transient or permanent error in tests."""


def _make_submit(error_count: int, exc_type: type[Exception], exc_msg: str):
    """Build a closure that raises ``exc_type`` ``error_count`` times then succeeds."""
    state = {"calls": 0}

    def submit() -> str:
        state["calls"] += 1
        if state["calls"] <= error_count:
            raise exc_type(exc_msg)
        return "TASK_OK"

    return submit, state


def test_retry_on_429_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_mod.time, "sleep", lambda _: None)
    submit, state = _make_submit(2, _Boom, "HTTP 429 Too Many concurrent aggregations")
    result = export_mod.submit_one_with_retry(submit, max_attempts=4, base_delay_s=0.01)
    assert result == "TASK_OK"
    assert state["calls"] == 3


def test_retry_on_503_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_mod.time, "sleep", lambda _: None)
    submit, state = _make_submit(1, _Boom, "503 Service Unavailable Internal error")
    result = export_mod.submit_one_with_retry(submit, max_attempts=4, base_delay_s=0.01)
    assert result == "TASK_OK"
    assert state["calls"] == 2


def test_retry_does_not_retry_permanent_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_mod.time, "sleep", lambda _: None)
    submit, state = _make_submit(99, _Boom, "Image.select: invalid band name")
    with pytest.raises(_Boom, match="invalid band"):
        export_mod.submit_one_with_retry(submit, max_attempts=4, base_delay_s=0.01)
    assert state["calls"] == 1


def test_retry_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_mod.time, "sleep", lambda _: None)
    submit, state = _make_submit(99, _Boom, "429 Too many")
    with pytest.raises(_Boom, match="429"):
        export_mod.submit_one_with_retry(submit, max_attempts=3, base_delay_s=0.01)
    assert state["calls"] == 3


# ---------------------------------------------------------------------------
# submit_in_chunks: ordering, chunking, and queue-room waiting
# ---------------------------------------------------------------------------

class _FakeTask:
    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:
        return f"_FakeTask({self.label!r})"


def test_submit_in_chunks_calls_each_callable_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """All callables run exactly once and results come back in order."""
    waits: list[float] = []
    monkeypatch.setattr(export_mod.time, "sleep", lambda s: waits.append(s))
    monkeypatch.setattr(export_mod, "_wait_for_queue_room", lambda *a, **k: None)

    callables = [(lambda i=i: _FakeTask(f"task-{i:03d}")) for i in range(5)]
    submitted = export_mod.submit_in_chunks(
        callables, chunk_size=2, pause_between_chunks_s=0.0,
    )
    assert [t.label for t in submitted] == [
        "task-000", "task-001", "task-002", "task-003", "task-004"
    ]


def test_submit_in_chunks_waits_for_queue_room_between_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Between chunks (but not after the final one), queue-room is checked."""
    monkeypatch.setattr(export_mod.time, "sleep", lambda _: None)
    queue_calls: list[tuple] = []

    def _fake_wait(max_active_tasks: int, base_pause_s: float) -> None:
        queue_calls.append((max_active_tasks, base_pause_s))

    monkeypatch.setattr(export_mod, "_wait_for_queue_room", _fake_wait)
    callables = [(lambda i=i: _FakeTask(str(i))) for i in range(7)]
    export_mod.submit_in_chunks(
        callables, chunk_size=3,
        pause_between_chunks_s=2.0, max_active_tasks=10,
    )
    # 7 callables / chunk_size=3 → chunks of [3, 3, 1] → 2 inter-chunk waits.
    assert len(queue_calls) == 2
    assert queue_calls[0] == (10, 2.0)


def test_submit_in_chunks_no_wait_after_last_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(export_mod.time, "sleep", lambda _: None)
    queue_calls: list[tuple] = []
    monkeypatch.setattr(
        export_mod, "_wait_for_queue_room",
        lambda max_active_tasks, base_pause_s: queue_calls.append(
            (max_active_tasks, base_pause_s)
        ),
    )
    callables = [(lambda i=i: _FakeTask(str(i))) for i in range(3)]
    # Single chunk: no inter-chunk wait.
    export_mod.submit_in_chunks(
        callables, chunk_size=10, max_active_tasks=10, pause_between_chunks_s=1.0,
    )
    assert queue_calls == []


def test_submit_in_chunks_propagates_callable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a callable raises, submit_in_chunks doesn't swallow it."""
    monkeypatch.setattr(export_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(export_mod, "_wait_for_queue_room", lambda *a, **k: None)

    def _ok() -> _FakeTask:
        return _FakeTask("ok")

    def _boom() -> _FakeTask:
        raise RuntimeError("simulated submit error after retry exhausted")

    with pytest.raises(RuntimeError, match="simulated"):
        export_mod.submit_in_chunks(
            [_ok, _boom, _ok], chunk_size=2,
            pause_between_chunks_s=0.0, max_active_tasks=10,
        )
