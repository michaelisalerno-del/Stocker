"""In-process controls for deterministic replay; never imports an IBKR adapter."""

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

    state: Literal["stopped", "running", "stopping", "completed", "failed"]
    execution_id: str | None = None
    generation: int = 0
    worker_alive: bool = False
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
    ibkr_connections_attempted: Literal[0] = 0
    broker_state_mutated: Literal[False] = False


ReplayRunner = Callable[[ReplayStartRequest, threading.Event], Any]


class ReplayController:
    """Thread-safe broker-isolated controller that owns one worker generation."""

    def __init__(
        self,
        *,
        runner: ReplayRunner,
        stop_timeout_seconds: float = 2.0,
    ) -> None:
        if stop_timeout_seconds <= 0.0:
            raise ValueError("stop_timeout_seconds must be positive")
        self._lock = threading.RLock()
        self._runner = runner
        self._stop_timeout_seconds = stop_timeout_seconds
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._generation = 0
        self._state = ReplayControlState(
            state="stopped",
            mode=None,
            speed=None,
            episode_id=None,
            changed_at_utc=datetime.now(UTC),
        )

    def start(self, request: ReplayStartRequest) -> ReplayControlState:
        with self._lock:
            self._refresh_worker_liveness_locked()
            if self._worker is not None and self._worker.is_alive():
                if self._state.state == "stopping":
                    raise ValueError("replay is stopping")
                if self._state.state != "running":
                    raise ValueError("previous replay worker is still alive")
                if (
                    self._state.mode != request.mode
                    or self._state.speed != request.speed
                    or self._state.episode_id != request.episode_id
                ):
                    raise ValueError("a different replay is already running")
                return self._state
            if request.mode == "episode_only" and not request.episode_id:
                raise ValueError("episode-only replay requires episode_id")

            self._generation += 1
            generation = self._generation
            execution_id = str(uuid.uuid4())
            self._state = ReplayControlState(
                state="running",
                execution_id=execution_id,
                generation=generation,
                worker_alive=True,
                mode=request.mode,
                speed=request.speed,
                episode_id=request.episode_id,
                changed_at_utc=datetime.now(UTC),
            )
            started = self._state
            self._stop_event = threading.Event()
            self._worker = threading.Thread(
                target=self._run,
                args=(request, self._stop_event, generation, execution_id),
                name=f"stocker-evidence-replay-{generation}",
                daemon=True,
            )
            try:
                self._worker.start()
            except Exception as exc:
                self._worker = None
                self._state = self._state.model_copy(
                    update={
                        "state": "failed",
                        "worker_alive": False,
                        "changed_at_utc": datetime.now(UTC),
                        "error": str(exc),
                    }
                )
                raise
            return started

    def _run(
        self,
        request: ReplayStartRequest,
        stop_event: threading.Event,
        generation: int,
        execution_id: str,
    ) -> None:
        try:
            result = self._runner(request, stop_event)
        except Exception as exc:
            self._record_failure(
                request=request,
                generation=generation,
                execution_id=execution_id,
                error=str(exc),
            )
        else:
            self._record_success(
                request=request,
                generation=generation,
                execution_id=execution_id,
                result=result,
            )
        finally:
            self._record_worker_exit(
                generation=generation,
                execution_id=execution_id,
                worker=threading.current_thread(),
            )

    def _owns_state(self, *, generation: int, execution_id: str) -> bool:
        return self._state.generation == generation and self._state.execution_id == execution_id

    def _record_failure(
        self,
        *,
        request: ReplayStartRequest,
        generation: int,
        execution_id: str,
        error: str,
    ) -> None:
        with self._lock:
            if not self._owns_state(
                generation=generation,
                execution_id=execution_id,
            ):
                return
            reported_error: str | None = error
            if self._state.state == "stopping":
                state: Literal["stopped", "failed"] = "stopped"
                reported_error = None
            elif self._state.state == "running":
                state = "failed"
            else:
                return
            self._state = ReplayControlState(
                state=state,
                execution_id=execution_id,
                generation=generation,
                worker_alive=True,
                mode=request.mode,
                speed=request.speed,
                episode_id=request.episode_id,
                changed_at_utc=datetime.now(UTC),
                error=reported_error,
            )

    def _record_success(
        self,
        *,
        request: ReplayStartRequest,
        generation: int,
        execution_id: str,
        result: Any,
    ) -> None:
        with self._lock:
            if not self._owns_state(
                generation=generation,
                execution_id=execution_id,
            ):
                return
            if self._state.state == "stopping":
                self._state = ReplayControlState(
                    state="stopped",
                    execution_id=execution_id,
                    generation=generation,
                    worker_alive=True,
                    mode=request.mode,
                    speed=request.speed,
                    episode_id=request.episode_id,
                    changed_at_utc=datetime.now(UTC),
                )
                return
            if self._state.state != "running":
                return
            if int(result.ibkr_connections_attempted) != 0 or bool(result.broker_state_mutated):
                self._state = ReplayControlState(
                    state="failed",
                    execution_id=execution_id,
                    generation=generation,
                    worker_alive=True,
                    mode=request.mode,
                    speed=request.speed,
                    episode_id=request.episode_id,
                    changed_at_utc=datetime.now(UTC),
                    error="blocked_replay_broker_isolation_violation",
                )
                return
            self._state = ReplayControlState(
                state="completed",
                execution_id=execution_id,
                generation=generation,
                worker_alive=True,
                mode=request.mode,
                speed=request.speed,
                episode_id=request.episode_id,
                changed_at_utc=datetime.now(UTC),
                records_replayed=int(result.records_replayed),
                raw_events_replayed=int(result.raw_events_replayed),
                result_digest=str(result.digest),
                stage_counts=dict(result.stage_counts),
                maximum_floating_difference=float(result.maximum_floating_difference),
            )

    def _record_worker_exit(
        self,
        *,
        generation: int,
        execution_id: str,
        worker: threading.Thread,
    ) -> None:
        with self._lock:
            if not self._owns_state(
                generation=generation,
                execution_id=execution_id,
            ):
                return
            if self._worker is worker:
                self._worker = None
            updates: dict[str, Any] = {
                "worker_alive": False,
                "changed_at_utc": datetime.now(UTC),
            }
            if self._state.state == "stopping":
                updates["state"] = "stopped"
            elif self._state.state == "running":
                updates.update(
                    {
                        "state": "failed",
                        "error": "replay_worker_exited_without_terminal_state",
                    }
                )
            self._state = self._state.model_copy(update=updates)

    def _refresh_worker_liveness_locked(self) -> None:
        worker = self._worker
        if worker is None or worker.is_alive():
            return
        self._worker = None
        updates: dict[str, Any] = {"worker_alive": False}
        if self._state.state == "stopping":
            updates.update(
                {
                    "state": "stopped",
                    "changed_at_utc": datetime.now(UTC),
                }
            )
        elif self._state.state == "running":
            updates.update(
                {
                    "state": "failed",
                    "changed_at_utc": datetime.now(UTC),
                    "error": "replay_worker_exited_without_terminal_state",
                }
            )
        self._state = self._state.model_copy(update=updates)

    def stop(self) -> ReplayControlState:
        with self._lock:
            self._refresh_worker_liveness_locked()
            worker = self._worker
            if worker is None:
                self._state = self._state.model_copy(
                    update={
                        "state": "stopped",
                        "worker_alive": False,
                        "changed_at_utc": datetime.now(UTC),
                        "error": None,
                    }
                )
                return self._state
            generation = self._state.generation
            execution_id = self._state.execution_id
            self._stop_event.set()
            self._state = self._state.model_copy(
                update={
                    "state": "stopping",
                    "worker_alive": True,
                    "changed_at_utc": datetime.now(UTC),
                    "error": None,
                }
            )

        worker.join(timeout=self._stop_timeout_seconds)

        with self._lock:
            self._refresh_worker_liveness_locked()
            if not self._owns_state(
                generation=generation,
                execution_id=execution_id or "",
            ):
                return self._state
            if self._worker is worker and worker.is_alive():
                self._state = self._state.model_copy(
                    update={
                        "state": "failed",
                        "worker_alive": True,
                        "changed_at_utc": datetime.now(UTC),
                        "error": "blocked_replay_stop_timeout_worker_alive",
                    }
                )
            elif self._state.state == "stopping":
                self._state = self._state.model_copy(
                    update={
                        "state": "stopped",
                        "worker_alive": False,
                        "changed_at_utc": datetime.now(UTC),
                    }
                )
            return self._state

    def status(self) -> ReplayControlState:
        with self._lock:
            self._refresh_worker_liveness_locked()
            return self._state

    def shutdown(self) -> ReplayControlState:
        """Cancel and join the active replay during application shutdown."""

        return self.stop()


__all__ = ["ReplayController", "ReplayControlState", "ReplayStartRequest"]
