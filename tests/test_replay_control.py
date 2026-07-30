from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from stocker_prospective.replay_control import ReplayController, ReplayStartRequest


@dataclass(frozen=True)
class _ReplayResult:
    records_replayed: int = 3
    raw_events_replayed: int = 2
    digest: str = "digest"
    stage_counts: dict[str, int] = field(
        default_factory=lambda: {"raw_market_event": 2, "m1c_prediction": 1}
    )
    maximum_floating_difference: float = 0.0
    ibkr_connections_attempted: int = 0
    broker_state_mutated: bool = False


def _wait_for_state(
    controller: ReplayController,
    expected: str,
    *,
    timeout_seconds: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if controller.status().state == expected:
            return
        time.sleep(0.001)
    raise AssertionError(
        f"replay did not reach {expected!r}; current={controller.status().model_dump()!r}"
    )


def test_start_while_stopped_assigns_execution_identity() -> None:
    started = threading.Event()
    release = threading.Event()

    def runner(_request: ReplayStartRequest, _stop_event: threading.Event) -> _ReplayResult:
        started.set()
        release.wait(timeout=1.0)
        return _ReplayResult()

    controller = ReplayController(runner=runner)
    state = controller.start(ReplayStartRequest())

    assert started.wait(timeout=1.0)
    assert state.state == "running"
    assert state.execution_id
    assert state.generation == 1
    assert state.worker_alive is True
    assert state.ibkr_connections_attempted == 0
    assert state.broker_state_mutated is False
    release.set()
    _wait_for_state(controller, "completed")


def test_duplicate_identical_start_returns_existing_execution() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def runner(_request: ReplayStartRequest, _stop_event: threading.Event) -> _ReplayResult:
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=1.0)
        return _ReplayResult()

    controller = ReplayController(runner=runner)
    request = ReplayStartRequest(mode="accelerated", speed=25.0)
    first = controller.start(request)
    assert started.wait(timeout=1.0)

    duplicate = controller.start(request)

    assert duplicate.execution_id == first.execution_id
    assert duplicate.generation == first.generation
    assert calls == 1
    release.set()
    _wait_for_state(controller, "completed")


def test_conflicting_start_while_running_is_rejected() -> None:
    started = threading.Event()
    release = threading.Event()

    def runner(_request: ReplayStartRequest, _stop_event: threading.Event) -> _ReplayResult:
        started.set()
        release.wait(timeout=1.0)
        return _ReplayResult()

    controller = ReplayController(runner=runner)
    controller.start(ReplayStartRequest(speed=10.0))
    assert started.wait(timeout=1.0)

    with pytest.raises(ValueError, match="different replay is already running"):
        controller.start(ReplayStartRequest(speed=20.0))

    release.set()
    _wait_for_state(controller, "completed")


def test_stop_while_running_cancels_joins_and_reaches_stopped() -> None:
    started = threading.Event()

    def runner(_request: ReplayStartRequest, stop_event: threading.Event) -> _ReplayResult:
        started.set()
        assert stop_event.wait(timeout=1.0)
        return _ReplayResult(records_replayed=0, raw_events_replayed=0)

    controller = ReplayController(runner=runner, stop_timeout_seconds=0.5)
    running = controller.start(ReplayStartRequest())
    assert started.wait(timeout=1.0)

    stopped = controller.stop()

    assert stopped.state == "stopped"
    assert stopped.execution_id == running.execution_id
    assert stopped.worker_alive is False
    assert stopped.ibkr_connections_attempted == 0
    assert stopped.broker_state_mutated is False


def test_immediate_restart_while_stopping_is_rejected() -> None:
    started = threading.Event()
    release = threading.Event()

    def runner(_request: ReplayStartRequest, _stop_event: threading.Event) -> _ReplayResult:
        started.set()
        release.wait(timeout=1.0)
        return _ReplayResult()

    controller = ReplayController(runner=runner, stop_timeout_seconds=0.5)
    controller.start(ReplayStartRequest())
    assert started.wait(timeout=1.0)
    stop_thread = threading.Thread(target=controller.stop)
    stop_thread.start()
    _wait_for_state(controller, "stopping")

    with pytest.raises(ValueError, match="replay is stopping"):
        controller.start(ReplayStartRequest())

    release.set()
    stop_thread.join(timeout=1.0)
    assert not stop_thread.is_alive()
    assert controller.status().state == "stopped"


def test_worker_exception_is_reported_without_weakening_safety_fields() -> None:
    def runner(_request: ReplayStartRequest, _stop_event: threading.Event) -> _ReplayResult:
        raise RuntimeError("synthetic replay failure")

    controller = ReplayController(runner=runner)
    controller.start(ReplayStartRequest())
    _wait_for_state(controller, "failed")
    failed = controller.status()

    assert failed.error == "synthetic replay failure"
    assert failed.worker_alive is False
    assert failed.ibkr_connections_attempted == 0
    assert failed.broker_state_mutated is False


def test_slow_worker_that_ignores_cancellation_fails_closed() -> None:
    started = threading.Event()
    release = threading.Event()

    def runner(_request: ReplayStartRequest, _stop_event: threading.Event) -> _ReplayResult:
        started.set()
        release.wait(timeout=1.0)
        return _ReplayResult()

    controller = ReplayController(runner=runner, stop_timeout_seconds=0.01)
    running = controller.start(ReplayStartRequest())
    assert started.wait(timeout=1.0)

    failed = controller.stop()

    assert failed.state == "failed"
    assert failed.execution_id == running.execution_id
    assert failed.worker_alive is True
    assert failed.error == "blocked_replay_stop_timeout_worker_alive"
    with pytest.raises(ValueError, match="previous replay worker is still alive"):
        controller.start(ReplayStartRequest())

    release.set()
    deadline = time.monotonic() + 1.0
    while controller.status().worker_alive and time.monotonic() < deadline:
        time.sleep(0.001)
    assert controller.status().worker_alive is False
    assert controller.status().state == "failed"


def test_old_worker_completion_cannot_overwrite_newer_generation() -> None:
    first_release = threading.Event()
    second_release = threading.Event()
    second_started = threading.Event()
    calls = 0

    def runner(_request: ReplayStartRequest, _stop_event: threading.Event) -> _ReplayResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_release.wait(timeout=1.0)
        else:
            second_started.set()
            second_release.wait(timeout=1.0)
        return _ReplayResult()

    controller = ReplayController(runner=runner)
    first = controller.start(ReplayStartRequest(speed=10.0))
    first_release.set()
    _wait_for_state(controller, "completed")
    second = controller.start(ReplayStartRequest(speed=20.0))
    assert second_started.wait(timeout=1.0)

    controller._record_success(  # noqa: SLF001 - exercises stale worker ownership boundary
        request=ReplayStartRequest(speed=10.0),
        generation=first.generation,
        execution_id=first.execution_id or "",
        result=_ReplayResult(records_replayed=999),
    )

    current = controller.status()
    assert current.state == "running"
    assert current.execution_id == second.execution_id
    assert current.generation == second.generation
    assert current.records_replayed == 0
    second_release.set()
    controller.stop()


def test_shutdown_stops_running_worker_cleanly() -> None:
    started = threading.Event()

    def runner(_request: ReplayStartRequest, stop_event: threading.Event) -> _ReplayResult:
        started.set()
        stop_event.wait(timeout=1.0)
        return _ReplayResult(records_replayed=0, raw_events_replayed=0)

    controller = ReplayController(runner=runner, stop_timeout_seconds=0.5)
    controller.start(ReplayStartRequest())
    assert started.wait(timeout=1.0)

    shutdown = controller.shutdown()

    assert shutdown.state == "stopped"
    assert shutdown.worker_alive is False
