"""Generation-fenced deterministic replay controls; never imports a broker."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReplayStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["real_time", "accelerated", "step", "episode_only"] = "accelerated"
    speed: float = Field(default=10.0, gt=0.0, le=10_000.0)
    episode_id: str | None = None


class ReplayControlState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: Literal[
        "stopped",
        "running",
        "stopping",
        "completed",
        "failed",
        "stop_failed",
    ]
    execution_id: str | None = None
    generation: int = Field(ge=0)
    mode: str | None
    speed: float | None
    episode_id: str | None
    changed_at_utc: datetime
    started_at_utc: datetime | None = None
    stop_requested_at_utc: datetime | None = None
    stopped_at_utc: datetime | None = None
    records_replayed: int = 0
    raw_events_replayed: int = 0
    result_digest: str | None = None
    stage_counts: dict[str, int] = Field(default_factory=dict)
    maximum_floating_difference: float = 0.0
    error: str | None = None
    termination_reason: str | None = None
    stop_clean: bool | None = None
    ibkr_connections_attempted: Literal[0] = 0
    broker_state_mutated: Literal[False] = False


ReplayRunner = Callable[[ReplayStartRequest, threading.Event], Any]


class ReplayController:
    """Own exactly one worker generation and reject every stale state write."""

    def __init__(
        self,
        *,
        runner: ReplayRunner,
        stop_join_timeout_seconds: float = 5.0,
    ) -> None:
        if stop_join_timeout_seconds <= 0:
            raise ValueError("replay stop join timeout must be positive")
        self._lock = threading.RLock()
        self._runner = runner
        self._stop_join_timeout_seconds = stop_join_timeout_seconds
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._generation = 0
        self._active_request: ReplayStartRequest | None = None
        now = datetime.now(UTC)
        self._state = ReplayControlState(
            state="stopped",
            generation=0,
            mode=None,
            speed=None,
            episode_id=None,
            changed_at_utc=now,
            stopped_at_utc=now,
            termination_reason="not_started",
            stop_clean=True,
        )

    def start(self, request: ReplayStartRequest) -> ReplayControlState:
        if request.mode == "episode_only" and not request.episode_id:
            raise ValueError("episode-only replay requires episode_id")
        with self._lock:
            worker_alive = self._worker is not None and self._worker.is_alive()
            if worker_alive:
                if self._state.state == "running" and self._active_request == request:
                    return self._state
                if self._state.state == "stopping":
                    raise ValueError("replay worker is still stopping")
                if self._state.state == "stop_failed":
                    raise ValueError("replay worker failed to stop")
                raise ValueError("a different replay worker is still active")
            self._worker = None
            self._generation += 1
            generation = self._generation
            execution_id = str(uuid.uuid4())
            self._active_request = request
            self._stop_event = threading.Event()
            started_at = datetime.now(UTC)
            self._state = ReplayControlState(
                state="running",
                execution_id=execution_id,
                generation=generation,
                mode=request.mode,
                speed=request.speed,
                episode_id=request.episode_id,
                changed_at_utc=started_at,
                started_at_utc=started_at,
                termination_reason=None,
                stop_clean=None,
            )
            started = self._state
            self._worker = threading.Thread(
                target=self._run,
                args=(generation, execution_id, request, self._stop_event),
                name=f"stocker-evidence-replay-{generation}",
                daemon=True,
            )
            self._worker.start()
            return started

    @staticmethod
    def _result_fields(result: Any) -> dict[str, Any]:
        return {
            "records_replayed": int(getattr(result, "records_replayed", 0)),
            "raw_events_replayed": int(getattr(result, "raw_events_replayed", 0)),
            "result_digest": (
                None if getattr(result, "digest", None) is None else str(result.digest)
            ),
            "stage_counts": dict(getattr(result, "stage_counts", {})),
            "maximum_floating_difference": float(
                getattr(result, "maximum_floating_difference", 0.0)
            ),
            "ibkr_connections_attempted": int(getattr(result, "ibkr_connections_attempted", 0)),
            "broker_state_mutated": bool(getattr(result, "broker_state_mutated", False)),
        }

    def _run(
        self,
        generation: int,
        execution_id: str,
        request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> None:
        try:
            result = self._runner(request, stop_event)
            if int(getattr(result, "ibkr_connections_attempted", 0)) != 0 or bool(
                getattr(result, "broker_state_mutated", False)
            ):
                raise RuntimeError("blocked_replay_broker_isolation_violation")
        except Exception as exc:
            with self._lock:
                if generation != self._generation or execution_id != self._state.execution_id:
                    return
                if self._state.state not in {"running", "stopping"}:
                    return
                failed_at = datetime.now(UTC)
                self._state = ReplayControlState(
                    state="failed",
                    execution_id=execution_id,
                    generation=generation,
                    mode=request.mode,
                    speed=request.speed,
                    episode_id=request.episode_id,
                    changed_at_utc=failed_at,
                    started_at_utc=self._state.started_at_utc,
                    stop_requested_at_utc=self._state.stop_requested_at_utc,
                    stopped_at_utc=failed_at,
                    error=f"REPLAY_WORKER_EXCEPTION:{type(exc).__name__}",
                    termination_reason="worker_exception",
                    stop_clean=False,
                )
            return

        with self._lock:
            if generation != self._generation or execution_id != self._state.execution_id:
                return
            if self._state.state not in {"running", "stopping"}:
                return
            completed_at = datetime.now(UTC)
            stopped = self._state.state == "stopping" or stop_event.is_set()
            self._state = ReplayControlState(
                state="stopped" if stopped else "completed",
                execution_id=execution_id,
                generation=generation,
                mode=request.mode,
                speed=request.speed,
                episode_id=request.episode_id,
                changed_at_utc=completed_at,
                started_at_utc=self._state.started_at_utc,
                stop_requested_at_utc=self._state.stop_requested_at_utc,
                stopped_at_utc=completed_at,
                termination_reason=("stop_requested" if stopped else "completed"),
                stop_clean=True,
                **self._result_fields(result),
            )

    def stop(self) -> ReplayControlState:
        with self._lock:
            worker = self._worker
            generation = self._generation
            execution_id = self._state.execution_id
            if worker is None or not worker.is_alive():
                if self._state.state == "stopping":
                    stopped_at = datetime.now(UTC)
                    self._state = self._state.model_copy(
                        update={
                            "state": "stopped",
                            "changed_at_utc": stopped_at,
                            "stopped_at_utc": stopped_at,
                            "termination_reason": "stop_requested",
                            "stop_clean": True,
                        }
                    )
                return self._state
            if self._state.state == "stop_failed":
                return self._state
            if self._state.state == "running":
                requested_at = datetime.now(UTC)
                self._state = self._state.model_copy(
                    update={
                        "state": "stopping",
                        "changed_at_utc": requested_at,
                        "stop_requested_at_utc": requested_at,
                        "termination_reason": "stop_requested",
                        "stop_clean": None,
                    }
                )
            self._stop_event.set()

        worker.join(timeout=self._stop_join_timeout_seconds)

        with self._lock:
            if generation != self._generation or execution_id != self._state.execution_id:
                return self._state
            if worker.is_alive():
                failed_at = datetime.now(UTC)
                self._state = self._state.model_copy(
                    update={
                        "state": "stop_failed",
                        "changed_at_utc": failed_at,
                        "stopped_at_utc": None,
                        "error": "REPLAY_WORKER_STOP_TIMEOUT",
                        "termination_reason": "stop_timeout",
                        "stop_clean": False,
                    }
                )
                return self._state
            if self._state.state == "stopping":
                stopped_at = datetime.now(UTC)
                self._state = self._state.model_copy(
                    update={
                        "state": "stopped",
                        "changed_at_utc": stopped_at,
                        "stopped_at_utc": stopped_at,
                        "termination_reason": "stop_requested",
                        "stop_clean": True,
                    }
                )
            return self._state

    def status(self) -> ReplayControlState:
        with self._lock:
            return self._state

    def shutdown(self) -> ReplayControlState:
        """Cancel and bounded-join an active replay during application shutdown."""

        return self.stop()


__all__ = ["ReplayController", "ReplayControlState", "ReplayStartRequest"]
