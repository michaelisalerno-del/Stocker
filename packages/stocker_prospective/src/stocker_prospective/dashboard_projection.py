"""Small operational projections shared by the prospective web routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

RecorderOperationalStateName = Literal[
    "inactive",
    "waiting_for_prospective_start",
    "recording",
    "stale",
    "blocked",
    "unknown",
]

_TERMINAL_RUN_STATUSES = frozenset({"inactive", "stopped", "closed", "completed"})
_TERMINAL_SESSION_STATUSES = frozenset(
    {
        "inactive",
        "stopped",
        "closed",
        "completed",
        "replay_complete",
    }
)


class RecorderOperationalState(BaseModel):
    """Authoritative, evidence-derived recorder liveness contract."""

    model_config = ConfigDict(frozen=True)

    state: RecorderOperationalStateName
    reason: str
    healthy: bool
    evaluated_at_utc: str
    prospective_start_utc: str
    stale_after_seconds: int
    lease_present: bool
    heartbeat_at_utc: str | None
    heartbeat_age_seconds: float | None
    last_completed_bar_timestamp_utc: str | None
    last_raw_or_capture_timestamp_utc: str | None
    runtime_blockers: list[str]


def _timestamp_from_projection(
    projection: object,
    *field_names: str,
) -> str | None:
    if not isinstance(projection, dict):
        return None
    for field_name in field_names:
        value = projection.get(field_name)
        if value is not None:
            return str(value)
    return None


def _blocker_codes(runtime: dict[str, Any]) -> list[str]:
    result: list[str] = []
    blockers = runtime.get("blockers")
    if not isinstance(blockers, list):
        return result
    for blocker in blockers:
        value = blocker.get("blocker_code") if isinstance(blocker, dict) else blocker
        if value is not None and str(value) not in result:
            result.append(str(value))
    return result


def project_recorder_operational_state(
    *,
    runtime: dict[str, Any],
    prospective_start_utc: datetime,
    stale_after_seconds: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Project recorder liveness without inferring activity from historical rows."""

    evaluated_at = datetime.now(UTC) if now is None else now
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("recorder operational-state clock must be timezone-aware")
    evaluated_at = evaluated_at.astimezone(UTC)
    prospective_start = prospective_start_utc.astimezone(UTC)
    run = runtime.get("run")
    lease = runtime.get("recorder_lease")
    session = runtime.get("session")
    blockers = _blocker_codes(runtime)
    last_completed_bar = _timestamp_from_projection(
        runtime.get("last_completed_bar"),
        "bar_end_utc",
    )
    last_capture = _timestamp_from_projection(
        runtime.get("latest_capture"),
        "target_timestamp_utc",
        "recorded_at_utc",
    )

    def result(
        state: RecorderOperationalStateName,
        reason: str,
        *,
        heartbeat: datetime | None = None,
    ) -> dict[str, Any]:
        heartbeat_utc = None if heartbeat is None else heartbeat.astimezone(UTC)
        age = None if heartbeat_utc is None else (evaluated_at - heartbeat_utc).total_seconds()
        return RecorderOperationalState(
            state=state,
            reason=reason,
            healthy=state == "recording",
            evaluated_at_utc=evaluated_at.isoformat(),
            prospective_start_utc=prospective_start.isoformat(),
            stale_after_seconds=stale_after_seconds,
            lease_present=isinstance(lease, dict),
            heartbeat_at_utc=(None if heartbeat_utc is None else heartbeat_utc.isoformat()),
            heartbeat_age_seconds=age,
            last_completed_bar_timestamp_utc=last_completed_bar,
            last_raw_or_capture_timestamp_utc=last_capture,
            runtime_blockers=blockers,
        ).model_dump(mode="json")

    if run is None and lease is None:
        return result("inactive", "run_absent")
    run_status = None if not isinstance(run, dict) else run.get("status")
    session_status = None if not isinstance(session, dict) else session.get("status")
    session_closed = isinstance(session, dict) and session.get("closed_at_utc") is not None
    if (
        str(run_status).lower() in _TERMINAL_RUN_STATUSES
        or str(session_status).lower() in _TERMINAL_SESSION_STATUSES
        or session_closed
    ):
        return result("inactive", "runtime_session_stopped")
    if blockers:
        return result("blocked", "runtime_blocker_active")
    if lease is None:
        return result("inactive", "lease_absent")
    if not isinstance(lease, dict):
        return result("unknown", "lease_malformed")

    if isinstance(run, dict):
        run_id = run.get("run_id")
        lease_run_id = lease.get("run_id")
        if run_id is None or lease_run_id is None or str(run_id) != str(lease_run_id):
            return result("unknown", "lease_run_mismatch")

    raw_heartbeat = lease.get("heartbeat_at_utc")
    try:
        heartbeat = datetime.fromisoformat(str(raw_heartbeat).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return result("unknown", "lease_heartbeat_malformed")
    if heartbeat.tzinfo is None or heartbeat.utcoffset() is None:
        return result("unknown", "lease_heartbeat_timezone_missing")
    heartbeat = heartbeat.astimezone(UTC)
    heartbeat_age = (evaluated_at - heartbeat).total_seconds()
    if heartbeat_age > stale_after_seconds:
        return result("stale", "lease_heartbeat_stale", heartbeat=heartbeat)

    if evaluated_at < prospective_start:
        return result(
            "waiting_for_prospective_start",
            "prospective_start_in_future",
            heartbeat=heartbeat,
        )
    if run is None:
        return result("unknown", "lease_without_run", heartbeat=heartbeat)
    return result("recording", "lease_fresh", heartbeat=heartbeat)


__all__ = [
    "RecorderOperationalState",
    "RecorderOperationalStateName",
    "project_recorder_operational_state",
]
