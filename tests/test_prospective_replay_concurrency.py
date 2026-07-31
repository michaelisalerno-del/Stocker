from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from stocker_prospective.replay_control import ReplayController, ReplayStartRequest


def result(digest: str = "digest") -> SimpleNamespace:
    return SimpleNamespace(
        records_replayed=10,
        raw_events_replayed=8,
        digest=digest,
        stage_counts={"raw": 8},
        maximum_floating_difference=0.0,
        ibkr_connections_attempted=0,
        broker_state_mutated=False,
    )


def join_worker(controller: ReplayController) -> None:
    worker = controller._worker
    assert worker is not None
    worker.join(1)


def test_stop_followed_immediately_by_start_uses_new_generation() -> None:
    started = threading.Event()

    def runner(
        request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> SimpleNamespace:
        started.set()
        stop_event.wait(1)
        return result(request.episode_id or "all")

    controller = ReplayController(runner=runner, stop_join_timeout_seconds=1)
    first = controller.start(ReplayStartRequest(mode="episode_only", episode_id="first"))
    assert started.wait(1)
    stopped = controller.stop()
    second = controller.start(ReplayStartRequest(mode="episode_only", episode_id="second"))

    assert stopped.state == "stopped"
    assert stopped.stop_clean is True
    assert second.state == "running"
    assert second.generation == first.generation + 1
    controller.stop()


@pytest.mark.parametrize("stale_throws", [False, True])
def test_old_worker_cannot_overwrite_new_generation(
    stale_throws: bool,
) -> None:
    new_release = threading.Event()
    stale_release = threading.Event()

    def runner(
        request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> SimpleNamespace:
        if request.episode_id == "stale":
            stale_release.wait(1)
            if stale_throws:
                raise RuntimeError("stale failure")
            return result("stale")
        new_release.wait(1)
        return result("new")

    controller = ReplayController(runner=runner, stop_join_timeout_seconds=1)
    active = controller.start(ReplayStartRequest(mode="episode_only", episode_id="new"))
    stale_thread = threading.Thread(
        target=controller._run,
        args=(
            active.generation - 1,
            "stale-execution",
            ReplayStartRequest(mode="episode_only", episode_id="stale"),
            threading.Event(),
        ),
    )
    stale_thread.start()
    stale_release.set()
    stale_thread.join(1)

    status = controller.status()
    assert status.generation == active.generation
    assert status.state == "running"
    assert status.result_digest is None
    assert status.error is None
    new_release.set()
    join_worker(controller)
    assert controller.status().result_digest == "new"


def test_worker_that_ignores_stop_exposes_failure_and_blocks_restart() -> None:
    release = threading.Event()
    started = threading.Event()

    def runner(
        _request: ReplayStartRequest,
        _stop_event: threading.Event,
    ) -> SimpleNamespace:
        started.set()
        release.wait(1)
        return result()

    controller = ReplayController(runner=runner, stop_join_timeout_seconds=0.01)
    controller.start(ReplayStartRequest())
    assert started.wait(1)

    stopped = controller.stop()

    assert stopped.state == "stop_failed"
    assert stopped.stop_clean is False
    assert stopped.stopped_at_utc is None
    assert stopped.termination_reason == "stop_timeout"
    with pytest.raises(ValueError, match="failed to stop"):
        controller.start(ReplayStartRequest())
    release.set()
    join_worker(controller)
    assert controller.status().state == "stop_failed"


def test_repeated_identical_start_is_idempotent_only_while_generation_active() -> None:
    release = threading.Event()

    def runner(
        _request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> SimpleNamespace:
        release.wait(1)
        if stop_event.is_set():
            return result("stopped")
        return result("completed")

    controller = ReplayController(runner=runner, stop_join_timeout_seconds=1)
    request = ReplayStartRequest(mode="accelerated", speed=20)
    first = controller.start(request)
    duplicate = controller.start(request)
    with pytest.raises(ValueError, match="different replay"):
        controller.start(request.model_copy(update={"speed": 21}))

    assert duplicate.generation == first.generation
    release.set()
    join_worker(controller)
    restarted = controller.start(request)
    assert restarted.generation == first.generation + 1
    release.clear()
    release.set()
    join_worker(controller)


def test_repeated_stop_is_idempotent() -> None:
    def runner(
        _request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> SimpleNamespace:
        stop_event.wait(1)
        return result()

    controller = ReplayController(runner=runner, stop_join_timeout_seconds=1)
    controller.start(ReplayStartRequest())
    first = controller.stop()
    second = controller.stop()

    assert first == second
    assert second.state == "stopped"
    assert second.termination_reason == "stop_requested"


def test_immediate_restart_is_rejected_while_stop_is_joining_worker() -> None:
    started = threading.Event()
    cancellation_seen = threading.Event()
    release = threading.Event()

    def runner(
        _request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> SimpleNamespace:
        started.set()
        assert stop_event.wait(1)
        cancellation_seen.set()
        release.wait(1)
        return result("stopped")

    controller = ReplayController(runner=runner, stop_join_timeout_seconds=1)
    controller.start(ReplayStartRequest())
    assert started.wait(1)
    stopping = threading.Thread(target=controller.stop)
    stopping.start()
    assert cancellation_seen.wait(1)

    with pytest.raises(ValueError, match="still stopping"):
        controller.start(ReplayStartRequest())

    release.set()
    stopping.join(1)
    assert not stopping.is_alive()
    assert controller.status().state == "stopped"


def test_worker_exception_is_failed_and_preserves_broker_isolation() -> None:
    def runner(
        _request: ReplayStartRequest,
        _stop_event: threading.Event,
    ) -> SimpleNamespace:
        raise RuntimeError("synthetic replay failure")

    controller = ReplayController(runner=runner)
    started = controller.start(ReplayStartRequest())
    join_worker(controller)
    failed = controller.status()

    assert failed.state == "failed"
    assert failed.execution_id == started.execution_id
    assert failed.error == "REPLAY_WORKER_EXCEPTION:RuntimeError"
    assert failed.ibkr_connections_attempted == 0
    assert failed.broker_state_mutated is False


def test_broker_isolation_violation_fails_closed() -> None:
    def runner(
        _request: ReplayStartRequest,
        _stop_event: threading.Event,
    ) -> SimpleNamespace:
        unsafe = result()
        unsafe.ibkr_connections_attempted = 1
        return unsafe

    controller = ReplayController(runner=runner)
    controller.start(ReplayStartRequest())
    join_worker(controller)

    assert controller.status().state == "failed"
    assert controller.status().ibkr_connections_attempted == 0
    assert controller.status().broker_state_mutated is False


def test_shutdown_cancels_and_joins_running_worker() -> None:
    started = threading.Event()

    def runner(
        _request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> SimpleNamespace:
        started.set()
        assert stop_event.wait(1)
        return result("shutdown")

    controller = ReplayController(runner=runner, stop_join_timeout_seconds=1)
    controller.start(ReplayStartRequest())
    assert started.wait(1)

    stopped = controller.shutdown()

    assert stopped.state == "stopped"
    assert stopped.stop_clean is True
    assert stopped.ibkr_connections_attempted == 0
    assert stopped.broker_state_mutated is False
