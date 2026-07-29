"""In-process controls for deterministic replay; never imports an IBKR adapter."""

from __future__ import annotations

import threading
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

    state: Literal["stopped", "running", "completed", "failed"]
    mode: str | None
    speed: float | None
    episode_id: str | None
    changed_at_utc: datetime
    records_replayed: int = 0
    raw_events_replayed: int = 0
    result_digest: str | None = None
    stage_counts: dict[str, int] = Field(default_factory=dict)
    maximum_floating_difference: float = 0.0
    error: str | None = None
    ibkr_connections_attempted: int = 0
    broker_state_mutated: bool = False


ReplayRunner = Callable[[ReplayStartRequest, threading.Event], Any]


class ReplayController:
    """Thread-safe broker-isolated controller that runs persisted replay evidence."""

    def __init__(self, *, runner: ReplayRunner) -> None:
        self._lock = threading.RLock()
        self._runner = runner
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._state = ReplayControlState(
            state="stopped",
            mode=None,
            speed=None,
            episode_id=None,
            changed_at_utc=datetime.now(UTC),
        )

    def start(self, request: ReplayStartRequest) -> ReplayControlState:
        with self._lock:
            if self._state.state == "running":
                if (
                    self._state.mode != request.mode
                    or self._state.speed != request.speed
                    or self._state.episode_id != request.episode_id
                ):
                    raise ValueError("a different replay is already running")
                return self._state
            if request.mode == "episode_only" and not request.episode_id:
                raise ValueError("episode-only replay requires episode_id")
            self._state = ReplayControlState(
                state="running",
                mode=request.mode,
                speed=request.speed,
                episode_id=request.episode_id,
                changed_at_utc=datetime.now(UTC),
            )
            started = self._state
            self._stop_event = threading.Event()
            self._worker = threading.Thread(
                target=self._run,
                args=(request, self._stop_event),
                name="stocker-evidence-replay",
                daemon=True,
            )
            self._worker.start()
            return started

    def _run(
        self,
        request: ReplayStartRequest,
        stop_event: threading.Event,
    ) -> None:
        try:
            result = self._runner(request, stop_event)
        except Exception as exc:
            with self._lock:
                if self._state.state == "running":
                    self._state = ReplayControlState(
                        state="failed",
                        mode=request.mode,
                        speed=request.speed,
                        episode_id=request.episode_id,
                        changed_at_utc=datetime.now(UTC),
                        error=str(exc),
                    )
            return
        with self._lock:
            if self._state.state != "running":
                return
            self._state = ReplayControlState(
                state="completed",
                mode=request.mode,
                speed=request.speed,
                episode_id=request.episode_id,
                changed_at_utc=datetime.now(UTC),
                records_replayed=int(result.records_replayed),
                raw_events_replayed=int(result.raw_events_replayed),
                result_digest=str(result.digest),
                stage_counts=dict(result.stage_counts),
                maximum_floating_difference=float(result.maximum_floating_difference),
                ibkr_connections_attempted=int(result.ibkr_connections_attempted),
                broker_state_mutated=bool(result.broker_state_mutated),
            )

    def stop(self) -> ReplayControlState:
        self._stop_event.set()
        with self._lock:
            self._state = ReplayControlState(
                state="stopped",
                mode=None,
                speed=None,
                episode_id=None,
                changed_at_utc=datetime.now(UTC),
            )
            return self._state

    def status(self) -> ReplayControlState:
        with self._lock:
            return self._state


__all__ = ["ReplayController", "ReplayStartRequest"]
