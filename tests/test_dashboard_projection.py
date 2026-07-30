from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stocker_prospective.dashboard_projection import project_recorder_operational_state

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
STARTED = NOW - timedelta(days=1)


def runtime_evidence(
    *,
    run: bool = True,
    heartbeat: object | None = NOW,
    blockers: tuple[str, ...] = (),
    session_status: str | None = None,
    session_closed: bool = False,
    historical_rows: bool = False,
) -> dict[str, object]:
    lease = (
        None
        if heartbeat is None
        else {
            "lease_key": "prospective_recorder",
            "run_id": "run-1",
            "owner_id": "recorder-1",
            "heartbeat_at_utc": (
                heartbeat.isoformat() if isinstance(heartbeat, datetime) else heartbeat
            ),
        }
    )
    session = (
        None
        if session_status is None
        else {
            "status": session_status,
            "closed_at_utc": NOW.isoformat() if session_closed else None,
        }
    )
    return {
        "run": {"run_id": "run-1"} if run else None,
        "recorder_lease": lease,
        "session": session,
        "blockers": [
            {
                "blocker_code": blocker,
                "component": "synthetic",
                "message": blocker,
                "severity": "error",
            }
            for blocker in blockers
        ],
        "last_completed_bar": (
            {"bar_end_utc": (NOW - timedelta(minutes=5)).isoformat()}
            if historical_rows
            else None
        ),
        "latest_capture": (
            {"target_timestamp_utc": (NOW - timedelta(minutes=4)).isoformat()}
            if historical_rows
            else None
        ),
    }


@pytest.mark.parametrize(
    ("evidence", "prospective_start", "expected_state", "expected_reason"),
    [
        (
            runtime_evidence(run=False, heartbeat=None),
            STARTED,
            "inactive",
            "run_absent",
        ),
        (
            runtime_evidence(heartbeat=None),
            STARTED,
            "inactive",
            "lease_absent",
        ),
        (
            runtime_evidence(heartbeat=NOW - timedelta(seconds=10)),
            STARTED,
            "recording",
            "lease_fresh",
        ),
        (
            runtime_evidence(heartbeat=NOW - timedelta(seconds=61)),
            STARTED,
            "stale",
            "lease_heartbeat_stale",
        ),
        (
            runtime_evidence(heartbeat="2026-07-30T11:59:50"),
            STARTED,
            "unknown",
            "lease_heartbeat_timezone_missing",
        ),
        (
            runtime_evidence(heartbeat="not-a-timestamp"),
            STARTED,
            "unknown",
            "lease_heartbeat_malformed",
        ),
        (
            runtime_evidence(heartbeat=NOW),
            NOW + timedelta(hours=1),
            "waiting_for_prospective_start",
            "prospective_start_in_future",
        ),
        (
            runtime_evidence(heartbeat=NOW, blockers=("blocked_synthetic_runtime",)),
            STARTED,
            "blocked",
            "runtime_blocker_active",
        ),
        (
            runtime_evidence(
                heartbeat=NOW,
                session_status="replay_complete",
                session_closed=True,
                historical_rows=True,
            ),
            STARTED,
            "inactive",
            "runtime_session_stopped",
        ),
    ],
)
def test_recorder_operational_state_is_derived_from_current_runtime_evidence(
    evidence: dict[str, object],
    prospective_start: datetime,
    expected_state: str,
    expected_reason: str,
) -> None:
    projected = project_recorder_operational_state(
        runtime=evidence,
        prospective_start_utc=prospective_start,
        stale_after_seconds=60,
        now=NOW,
    )

    assert projected["state"] == expected_state
    assert projected["reason"] == expected_reason
    assert projected["healthy"] is (expected_state == "recording")
    assert projected["evaluated_at_utc"] == NOW.isoformat()
    assert projected["stale_after_seconds"] == 60


def test_fresh_lease_for_wrong_run_is_never_reported_as_recording() -> None:
    evidence = runtime_evidence()
    assert isinstance(evidence["recorder_lease"], dict)
    evidence["recorder_lease"]["run_id"] = "different-run"

    projected = project_recorder_operational_state(
        runtime=evidence,
        prospective_start_utc=STARTED,
        stale_after_seconds=60,
        now=NOW,
    )

    assert projected["state"] == "unknown"
    assert projected["reason"] == "lease_run_mismatch"
    assert projected["healthy"] is False
