"""Pure selection and per-feed readiness calculations for the read-only web process."""

from __future__ import annotations

import sqlite3
from typing import Any

RECORDER_HEARTBEAT_FRESH_US = 5_000_000
READINESS_INBOX_BACKLOG_LIMIT = 5_000
READINESS_LATEST_CALLBACK_SEEK_SQL = (
    "SELECT callback.received_at_us FROM callback_inbox AS callback "
    "INDEXED BY callback_inbox_readiness_latest_idx "
    "WHERE callback.run_id=? AND callback.recorder_generation=? "
    "AND callback.connection_generation=? AND callback.request_id=? "
    "AND callback.lifecycle='acknowledged' "
    "ORDER BY callback.received_at_us DESC LIMIT 1"
)

READINESS_FEEDS_SQL = (
    "SELECT subscription.subscription_id, subscription.instrument_id, "
    "subscription.feed_kind, subscription.request_id, subscription.lifecycle, "
    "subscription.optional, subscription.snapshot, subscription.stale_after_us, "
    "subscription.opened_at_us, subscription.retry_count, "
    "subscription.next_retry_at_us, subscription.last_attempt_at_us, "
    "subscription.last_error_code, subscription.permanent_failure, "
    "(SELECT callback.received_at_us FROM callback_inbox AS callback "
    "INDEXED BY callback_inbox_readiness_latest_idx "
    "WHERE callback.run_id=subscription.run_id "
    "AND callback.recorder_generation=subscription.recorder_generation "
    "AND callback.connection_generation=subscription.connection_generation "
    "AND callback.request_id=subscription.request_id "
    "AND callback.lifecycle='acknowledged' "
    "ORDER BY callback.received_at_us DESC LIMIT 1) AS latest_callback_at_us, "
    "(SELECT incident.code FROM incidents incident "
    "WHERE incident.run_id=subscription.run_id "
    "AND incident.subscription_id=subscription.subscription_id "
    "AND incident.resolved_at_us IS NULL "
    "ORDER BY incident.opened_at_us DESC, incident.incident_id DESC LIMIT 1) "
    "AS incident_code FROM subscriptions subscription WHERE subscription.run_id=? "
    "AND subscription.recorder_generation=? AND subscription.connection_generation=? "
    "AND subscription.lifecycle!='closed' "
    "ORDER BY subscription.optional, subscription.instrument_id, "
    "subscription.feed_kind, subscription.request_id"
)


def _heartbeat_is_fresh(heartbeat: int | None, *, now_us: int) -> bool:
    if heartbeat is None:
        return False
    age_us = now_us - int(heartbeat)
    return 0 <= age_us <= RECORDER_HEARTBEAT_FRESH_US


def select_operational_run(
    connection: sqlite3.Connection,
    *,
    pinned_run_id: str | None,
    now_us: int,
) -> tuple[dict[str, Any] | None, str]:
    """Select a pinned run exactly or the strongest current recorder evidence."""

    projection = (
        "SELECT run.run_id, run.mode, run.status, run.started_at_us, "
        "state.recorder_generation, state.lifecycle, state.reason, "
        "state.process_heartbeat_at_us, state.connection_state, "
        "state.connection_generation, state.inbox_nonterminal_count, "
        "state.inbox_bytes, state.database_bytes, state.wal_bytes, "
        "generation.ended_at_us FROM runs run "
        "LEFT JOIN runtime_state state ON state.run_id=run.run_id "
        "LEFT JOIN recorder_generations generation ON generation.run_id=state.run_id "
        "AND generation.generation=state.recorder_generation "
    )
    if pinned_run_id is not None:
        row = connection.execute(projection + "WHERE run.run_id=?", (pinned_run_id,)).fetchone()
        if row is None:
            return None, "pinned_run_unavailable"
        return dict(row), "pinned_run"
    row = connection.execute(
        projection + "WHERE run.status='running' AND generation.ended_at_us IS NULL "
        "AND state.lifecycle NOT IN ('stopped','fatal') "
        "ORDER BY state.process_heartbeat_at_us DESC, run.started_at_us DESC, run.run_id DESC "
        "LIMIT 1"
    ).fetchone()
    if row is None:
        return None, "no_active_operational_run"
    selected = dict(row)
    heartbeat = selected["process_heartbeat_at_us"]
    fresh = _heartbeat_is_fresh(heartbeat, now_us=now_us)
    return selected, "fresh_recorder_generation" if fresh else "stale_recorder_generation"


def calculate_readiness(
    connection: sqlite3.Connection,
    *,
    pinned_run_id: str | None,
    now_us: int,
    expected_since_us: int | None,
) -> dict[str, Any]:
    """Calculate bounded operational readiness from one query-only snapshot."""

    selected, selection_reason = select_operational_run(
        connection,
        pinned_run_id=pinned_run_id,
        now_us=now_us,
    )
    reasons: list[str] = []
    session = {
        "calendar": "XNYS",
        "state": "regular_session" if expected_since_us is not None else "outside_regular_session",
        "expected_since_us": expected_since_us,
    }
    if selected is None:
        reasons.append(selection_reason.upper())
        return {
            "ready": False,
            "selection_reason": selection_reason,
            "newer_nonoperational_run": None,
            "selected_run": pinned_run_id,
            "recorder_generation": None,
            "session": session,
            "reasons": reasons,
            "socket": None,
            "inbox": None,
            "feeds": [],
            "database": {"readable": True, "writer_admission_healthy": False},
        }

    run_id = str(selected["run_id"])
    newer_nonoperational = None
    if pinned_run_id is None:
        newer_nonoperational = connection.execute(
            "SELECT run_id, status, started_at_us FROM runs WHERE run_id!=? "
            "AND started_at_us>? ORDER BY started_at_us DESC, run_id DESC LIMIT 1",
            (run_id, selected["started_at_us"]),
        ).fetchone()
        if newer_nonoperational is not None:
            reasons.append("NEWER_NONOPERATIONAL_RUN_PRESENT")
    generation = selected["recorder_generation"]
    connection_generation = selected["connection_generation"]
    heartbeat = selected["process_heartbeat_at_us"]
    heartbeat_fresh = _heartbeat_is_fresh(heartbeat, now_us=now_us)
    if selected["status"] != "running":
        reasons.append("RUN_NOT_OPERATIONAL")
    if selected["ended_at_us"] is not None or selected["lifecycle"] in {"stopped", "fatal"}:
        reasons.append("RECORDER_LIFECYCLE_INACTIVE")
    elif selected["lifecycle"] != "running":
        reasons.append("RECORDER_LIFECYCLE_NOT_RUNNING")
        if selected["reason"] is not None:
            reasons.append(f"RECORDER_DEGRADED:{selected['reason']}")
    if not heartbeat_fresh:
        reason = "RECORDER_HEARTBEAT_STALE"
        if heartbeat is not None and int(heartbeat) > now_us:
            reason = "RECORDER_HEARTBEAT_IN_FUTURE"
        reasons.append(reason)
    if selected["connection_state"] != "connected":
        reasons.append("IBKR_SOCKET_NOT_CONNECTED")
    backlog = int(selected["inbox_nonterminal_count"] or 0)
    if backlog > READINESS_INBOX_BACKLOG_LIMIT:
        reasons.append("DURABLE_INBOX_BACKLOG_HIGH")

    rows = connection.execute(
        READINESS_FEEDS_SQL,
        (run_id, generation, connection_generation),
    ).fetchall()
    feeds: list[dict[str, Any]] = []
    required_count = 0
    for row in rows:
        optional = bool(row["optional"])
        required_count += int(not optional)
        lifecycle = str(row["lifecycle"])
        latest_callback = row["latest_callback_at_us"]
        reference = int(row["opened_at_us"])
        if latest_callback is not None:
            reference = int(latest_callback)
        if expected_since_us is not None:
            reference = max(reference, expected_since_us)
        stale = expected_since_us is not None and now_us - reference > int(row["stale_after_us"])
        permanently_rejected = bool(row["permanent_failure"])
        retrying = (
            not permanently_rejected
            and row["last_error_code"] is not None
            and lifecycle in {"connecting", "disconnected", "degraded"}
        )
        active = lifecycle == "active"
        identity = f"{row['instrument_id']}:{row['feed_kind']}:{row['request_id']}"
        if not optional:
            if not active:
                reasons.append(f"REQUIRED_SUBSCRIPTION_INACTIVE:{identity}")
            if stale:
                reasons.append(f"REQUIRED_FEED_STALE:{identity}")
            if row["incident_code"] is not None:
                reasons.append(f"REQUIRED_FEED_INCIDENT:{identity}")
        feeds.append(
            {
                "identity": identity,
                "instrument_id": row["instrument_id"],
                "feed_kind": row["feed_kind"],
                "request_id": row["request_id"],
                "expected": True,
                "required": not optional,
                "optional": optional,
                "active": active,
                "stale": stale,
                "retrying": retrying,
                "permanently_rejected": permanently_rejected,
                "latest_callback_at_us": latest_callback,
                "incident": row["incident_code"],
                "retry": {
                    "count": row["retry_count"],
                    "next_at_us": row["next_retry_at_us"],
                    "last_attempt_at_us": row["last_attempt_at_us"],
                    "last_error_code": row["last_error_code"],
                },
            }
        )
    if required_count == 0:
        reasons.append("REQUIRED_SUBSCRIPTION_SET_EMPTY")
    reasons = list(dict.fromkeys(reasons))
    return {
        "ready": not reasons,
        "selection_reason": selection_reason,
        "newer_nonoperational_run": (
            None if newer_nonoperational is None else dict(newer_nonoperational)
        ),
        "selected_run": run_id,
        "recorder_generation": generation,
        "recorder": {
            "lifecycle": selected["lifecycle"],
            "reason": selected["reason"],
            "process_heartbeat_at_us": heartbeat,
        },
        "session": session,
        "reasons": reasons,
        "socket": {
            "state": selected["connection_state"],
            "connection_generation": connection_generation,
        },
        "inbox": {
            "nonterminal": backlog,
            "threshold": READINESS_INBOX_BACKLOG_LIMIT,
            "bytes": int(selected["inbox_bytes"] or 0),
        },
        "feeds": feeds,
        "database": {
            "readable": True,
            "writer_admission_healthy": (
                selected["status"] == "running"
                and selected["ended_at_us"] is None
                and selected["lifecycle"] in {"running", "degraded"}
                and heartbeat_fresh
            ),
            "database_bytes": selected["database_bytes"],
            "wal_bytes": selected["wal_bytes"],
        },
    }
