"""Shared durable terminal transitions for shadow positions."""

from __future__ import annotations

import sqlite3

from stocker_runtime.storage.repository import canonical_json_text

ENTRY_EVIDENCE_EXPIRED = "entry_evidence_expired"
MAX_PENDING_TERMINALIZATIONS_PER_PASS = 32
MAX_SHADOW_OUTCOME_PAYLOAD_BYTES = 16_384


def terminalize_expired_pending_positions(
    connection: sqlite3.Connection,
    *,
    now_us: int,
    limit: int,
    position_id: str | None = None,
) -> tuple[str, ...]:
    """Atomically record bounded incomplete outcomes for expired pending entries."""

    if limit <= 0:
        return ()
    limit = min(limit, MAX_PENDING_TERMINALIZATIONS_PER_PASS)
    if position_id is not None:
        rows = tuple(
            connection.execute(
                "SELECT position.position_id, position.proposed_trade_output_id, "
                "position.policy_hash FROM shadow_positions position "
                "JOIN shadow_progress progress ON progress.position_id=position.position_id "
                "WHERE position.position_id=? AND position.lifecycle='pending' "
                "AND progress.pending_evidence_drained=1 "
                "AND progress.pending_retention_deadline_us<? LIMIT ?",
                (position_id, now_us, limit),
            )
        )
    else:
        rows = tuple(
            connection.execute(
                "SELECT position.position_id, position.proposed_trade_output_id, "
                "position.policy_hash FROM shadow_progress progress "
                "INDEXED BY shadow_progress_pending_expiry_idx "
                "JOIN shadow_positions position ON position.position_id=progress.position_id "
                "WHERE progress.pending_evidence_drained=1 "
                "AND progress.pending_retention_deadline_us<? "
                "AND position.lifecycle='pending' "
                "ORDER BY progress.pending_retention_deadline_us, progress.position_id LIMIT ?",
                (now_us, limit),
            )
        )

    terminalized: list[str] = []
    for row in rows:
        durable_position_id = str(row["position_id"])
        updated = connection.execute(
            "UPDATE shadow_positions SET lifecycle='invalid', closed_at_us=?, invalid_reason=? "
            "WHERE position_id=? AND lifecycle='pending'",
            (now_us, ENTRY_EVIDENCE_EXPIRED, durable_position_id),
        )
        if updated.rowcount != 1:
            continue
        payload_json = canonical_json_text(
            {
                "policy_hash": str(row["policy_hash"]),
                "proposed_trade_output_id": str(row["proposed_trade_output_id"]),
            },
            max_bytes=MAX_SHADOW_OUTCOME_PAYLOAD_BYTES,
        )
        connection.execute(
            "INSERT INTO shadow_outcomes(position_id, outcome_at_us, reason, gross_pnl, "
            "net_pnl, return_value, mfe, mae, completeness, payload_json) "
            "VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, 'incomplete', ?)",
            (durable_position_id, now_us, ENTRY_EVIDENCE_EXPIRED, payload_json),
        )
        terminalized.append(durable_position_id)
    return tuple(terminalized)
