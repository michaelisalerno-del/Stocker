"""Bounded retention and visible storage-cap state for Stocker V2."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from stocker_runtime.storage.connection import connect_v2

DAY_US = 86_400_000_000
MAX_MAINTENANCE_BATCH_ROWS = 10_000


class RetentionInvariantError(RuntimeError):
    """Compaction cannot prove the evidence chain required for deletion."""


class StorageCapState(StrEnum):
    NORMAL = "normal"
    SOFT = "soft_cap"
    DEGRADED = "degraded"
    FATAL = "fatal"


@dataclass(frozen=True)
class RetentionPolicy:
    """Frozen operational retention defaults; smaller values support deterministic tests."""

    callback_payload_us: int = DAY_US
    receipt_us: int = 90 * DAY_US
    max_receipts_per_run: int = 2_048
    tombstone_us: int = 7 * DAY_US
    raw_market_event_us: int = 30 * DAY_US
    completed_bar_us: int = 400 * DAY_US
    closed_subscription_us: int = 90 * DAY_US
    resolved_diagnostic_us: int = 400 * DAY_US
    idea_shadow_us: int = 7 * 365 * DAY_US
    database_cap_bytes: int = 8 * 1024**3
    wal_cap_bytes: int = 64 * 1024**2
    maintenance_batch_rows: int = MAX_MAINTENANCE_BATCH_ROWS

    def __post_init__(self) -> None:
        integer_values = (
            self.callback_payload_us,
            self.receipt_us,
            self.max_receipts_per_run,
            self.tombstone_us,
            self.raw_market_event_us,
            self.completed_bar_us,
            self.closed_subscription_us,
            self.resolved_diagnostic_us,
            self.idea_shadow_us,
            self.database_cap_bytes,
            self.wal_cap_bytes,
            self.maintenance_batch_rows,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("retention values must be positive")
        if self.maintenance_batch_rows > MAX_MAINTENANCE_BATCH_ROWS:
            raise ValueError("maintenance batches may not exceed 10,000 rows")


@dataclass(frozen=True)
class RetentionResult:
    cap_state: StorageCapState
    database_bytes: int
    wal_bytes: int
    payloads_compacted: int
    receipts_rolled: int
    expired_rows_deleted: int
    admission_allowed: bool
    optional_feeds_allowed: bool
    required_action: str | None
    checkpoint_attempted: bool
    incremental_vacuum_attempted: bool


def _cap_state(database_bytes: int, cap_bytes: int) -> StorageCapState:
    ratio = database_bytes / cap_bytes
    if ratio >= 1:
        return StorageCapState.FATAL
    if ratio >= 0.95:
        return StorageCapState.DEGRADED
    if ratio >= 0.85:
        return StorageCapState.SOFT
    return StorageCapState.NORMAL


class RetentionManager:
    """Run small transactional expiry passes without deleting unexpired protected evidence."""

    def __init__(self, database_path: str | Path, policy: RetentionPolicy | None = None) -> None:
        self.database_path = Path(database_path)
        self.policy = policy or RetentionPolicy()

    def _measured_sizes(self, connection: sqlite3.Connection) -> tuple[int, int]:
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        wal_path = Path(f"{self.database_path}-wal")
        return page_count * page_size, wal_path.stat().st_size if wal_path.exists() else 0

    def _compact_payloads(self, connection: sqlite3.Connection, cutoff_us: int, limit: int) -> int:
        cursor = connection.execute(
            """
            UPDATE callback_inbox
            SET payload_json = NULL
            WHERE source_sequence IN (
                SELECT source_sequence
                FROM callback_inbox
                WHERE lifecycle = 'acknowledged'
                  AND normalized_event_id IS NOT NULL
                  AND acknowledged_at_us IS NOT NULL
                  AND acknowledged_at_us <= ?
                  AND receipt_batch_id IS NOT NULL
                  AND payload_json IS NOT NULL
                  AND (EXISTS (
                      SELECT 1 FROM callback_receipts
                      WHERE callback_receipts.batch_id = callback_inbox.receipt_batch_id
                        AND callback_receipts.run_id = callback_inbox.run_id
                        AND callback_inbox.source_sequence BETWEEN
                            callback_receipts.first_source_sequence
                            AND callback_receipts.last_source_sequence
                  ) OR EXISTS (
                      SELECT 1 FROM callback_compaction_watermarks watermark
                      WHERE watermark.run_id = callback_inbox.run_id
                        AND watermark.compacted_through_sequence >= callback_inbox.source_sequence
                  ))
                ORDER BY source_sequence
                LIMIT ?
            )
            """,
            (cutoff_us, limit),
        )
        return cursor.rowcount

    def _receipt_candidates(
        self, connection: sqlite3.Connection, run_id: str, cutoff_us: int, remaining: int
    ) -> tuple[sqlite3.Row, ...]:
        rows = tuple(
            connection.execute(
                "SELECT * FROM callback_receipts WHERE run_id = ? "
                "ORDER BY first_source_sequence, batch_id",
                (run_id,),
            )
        )
        excess = max(0, len(rows) - self.policy.max_receipts_per_run)
        candidate_count = 0
        for index, row in enumerate(rows):
            if int(row["created_at_us"]) <= cutoff_us or index < excess:
                candidate_count += 1
            else:
                break
        return rows[: min(candidate_count, remaining)]

    def _roll_receipts(self, connection: sqlite3.Connection, cutoff_us: int, limit: int) -> int:
        rolled = 0
        run_ids = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT run_id FROM callback_receipts ORDER BY run_id"
            )
        )
        for run_id in run_ids:
            if rolled >= limit:
                break
            watermark = connection.execute(
                "SELECT * FROM callback_compaction_watermarks WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            candidates = self._receipt_candidates(
                connection,
                run_id,
                cutoff_us,
                limit - rolled,
            )
            if not candidates:
                continue
            expected_first = (
                1 if watermark is None else int(watermark["compacted_through_sequence"]) + 1
            )
            for receipt in candidates:
                if int(receipt["first_source_sequence"]) != expected_first:
                    raise RetentionInvariantError(
                        f"receipt chain for run {run_id} is not contiguous at {expected_first}"
                    )
                expected_first = int(receipt["last_source_sequence"]) + 1
            previous_count = 0 if watermark is None else int(watermark["cumulative_callback_count"])
            previous_first = None if watermark is None else watermark["first_received_at_us"]
            callback_count = previous_count + sum(int(row["callback_count"]) for row in candidates)
            first_received = (
                int(candidates[0]["first_received_at_us"])
                if previous_first is None
                else int(previous_first)
            )
            last = candidates[-1]
            connection.execute(
                """
                INSERT INTO callback_compaction_watermarks(
                    run_id, compacted_through_sequence, cumulative_callback_count,
                    first_received_at_us, last_received_at_us, rolled_receipt_chain_hash,
                    updated_at_us
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    compacted_through_sequence = excluded.compacted_through_sequence,
                    cumulative_callback_count = excluded.cumulative_callback_count,
                    first_received_at_us = excluded.first_received_at_us,
                    last_received_at_us = excluded.last_received_at_us,
                    rolled_receipt_chain_hash = excluded.rolled_receipt_chain_hash,
                    updated_at_us = excluded.updated_at_us
                """,
                (
                    run_id,
                    int(last["last_source_sequence"]),
                    callback_count,
                    first_received,
                    int(last["last_received_at_us"]),
                    str(last["chained_payload_hash"]),
                    cutoff_us + self.policy.receipt_us,
                ),
            )
            connection.executemany(
                "DELETE FROM callback_receipts WHERE batch_id = ?",
                ((str(row["batch_id"]),) for row in candidates),
            )
            rolled += len(candidates)
        return rolled

    def _delete_limited(
        self,
        connection: sqlite3.Connection,
        table: str,
        identity_column: str,
        predicate: str,
        parameters: tuple[object, ...],
        remaining: int,
    ) -> int:
        if remaining <= 0:
            return 0
        cursor = connection.execute(
            f"DELETE FROM {table} WHERE {identity_column} IN ("  # noqa: S608
            f"SELECT {identity_column} FROM {table} WHERE {predicate} "
            f"ORDER BY {identity_column} LIMIT ?)",
            (*parameters, remaining),
        )
        return cursor.rowcount

    def _prune_expired(self, connection: sqlite3.Connection, now_us: int, limit: int) -> int:
        deleted = 0

        def prune(table: str, identity: str, predicate: str, *parameters: object) -> None:
            nonlocal deleted
            deleted += self._delete_limited(
                connection, table, identity, predicate, parameters, limit - deleted
            )

        protected_cutoff = now_us - self.policy.idea_shadow_us
        prune(
            "shadow_marks",
            "rowid",
            "marked_at_us <= ? AND EXISTS (SELECT 1 FROM shadow_positions p "
            "WHERE p.position_id = shadow_marks.position_id "
            "AND p.closed_at_us IS NOT NULL AND p.closed_at_us <= ?)",
            protected_cutoff,
            protected_cutoff,
        )
        prune(
            "shadow_outcomes",
            "position_id",
            "outcome_at_us <= ? AND EXISTS (SELECT 1 FROM shadow_positions p "
            "WHERE p.position_id = shadow_outcomes.position_id "
            "AND p.closed_at_us IS NOT NULL AND p.closed_at_us <= ?)",
            protected_cutoff,
            protected_cutoff,
        )
        prune(
            "shadow_legs",
            "rowid",
            "EXISTS (SELECT 1 FROM shadow_positions p "
            "WHERE p.position_id = shadow_legs.position_id "
            "AND p.closed_at_us IS NOT NULL AND p.closed_at_us <= ?)",
            protected_cutoff,
        )
        prune(
            "shadow_positions",
            "position_id",
            "closed_at_us IS NOT NULL AND closed_at_us <= ? "
            "AND NOT EXISTS (SELECT 1 FROM shadow_marks m "
            "WHERE m.position_id = shadow_positions.position_id) "
            "AND NOT EXISTS (SELECT 1 FROM shadow_outcomes o "
            "WHERE o.position_id = shadow_positions.position_id) "
            "AND NOT EXISTS (SELECT 1 FROM shadow_legs l "
            "WHERE l.position_id = shadow_positions.position_id)",
            protected_cutoff,
        )
        prune(
            "idea_output_legs",
            "rowid",
            "EXISTS (SELECT 1 FROM idea_outputs o "
            "WHERE o.output_id = idea_output_legs.output_id "
            "AND o.emitted_at_us <= ? "
            "AND NOT EXISTS (SELECT 1 FROM shadow_positions p "
            "WHERE p.proposed_trade_output_id = o.output_id))",
            protected_cutoff,
        )
        prune(
            "idea_outputs",
            "output_id",
            "emitted_at_us <= ? AND NOT EXISTS (SELECT 1 FROM shadow_positions p "
            "WHERE p.proposed_trade_output_id = idea_outputs.output_id) "
            "AND NOT EXISTS (SELECT 1 FROM idea_output_legs l "
            "WHERE l.output_id = idea_outputs.output_id)",
            protected_cutoff,
        )
        prune(
            "market_events",
            "event_id",
            "event_kind NOT IN ('bar', 'historical_bar') AND event_at_us <= ? "
            "AND NOT EXISTS (SELECT 1 FROM market_latest l "
            "WHERE l.event_id = market_events.event_id) "
            "AND NOT EXISTS (SELECT 1 FROM idea_outputs o "
            "WHERE o.first_input_event_id = market_events.event_id "
            "OR o.last_input_event_id = market_events.event_id) "
            "AND NOT EXISTS (SELECT 1 FROM shadow_legs s "
            "WHERE s.entry_market_event_id = market_events.event_id "
            "OR s.exit_market_event_id = market_events.event_id)",
            now_us - self.policy.raw_market_event_us,
        )
        prune(
            "market_events",
            "event_id",
            "event_kind IN ('bar', 'historical_bar') AND event_at_us <= ? "
            "AND NOT EXISTS (SELECT 1 FROM market_latest l "
            "WHERE l.event_id = market_events.event_id) "
            "AND NOT EXISTS (SELECT 1 FROM idea_outputs o "
            "WHERE o.first_input_event_id = market_events.event_id "
            "OR o.last_input_event_id = market_events.event_id) "
            "AND NOT EXISTS (SELECT 1 FROM shadow_legs s "
            "WHERE s.entry_market_event_id = market_events.event_id "
            "OR s.exit_market_event_id = market_events.event_id)",
            now_us - self.policy.completed_bar_us,
        )
        prune(
            "callback_inbox",
            "source_sequence",
            "lifecycle = 'acknowledged' AND payload_json IS NULL "
            "AND acknowledged_at_us <= ? "
            "AND NOT EXISTS (SELECT 1 FROM market_events e "
            "WHERE e.source_sequence = callback_inbox.source_sequence) "
            "AND EXISTS (SELECT 1 FROM callback_compaction_watermarks w "
            "WHERE w.run_id = callback_inbox.run_id "
            "AND w.compacted_through_sequence >= callback_inbox.source_sequence)",
            now_us - self.policy.tombstone_us,
        )
        prune(
            "subscriptions",
            "subscription_id",
            "closed_at_us IS NOT NULL AND closed_at_us <= ?",
            now_us - self.policy.closed_subscription_us,
        )
        prune(
            "incidents",
            "incident_id",
            "resolved_at_us IS NOT NULL AND resolved_at_us <= ?",
            now_us - self.policy.resolved_diagnostic_us,
        )
        prune(
            "gaps",
            "gap_id",
            "resolved_at_us IS NOT NULL AND resolved_at_us <= ?",
            now_us - self.policy.resolved_diagnostic_us,
        )
        return deleted

    def run(
        self,
        *,
        now_us: int,
        measured_database_bytes: int | None = None,
        measured_wal_bytes: int | None = None,
    ) -> RetentionResult:
        """Run one bounded pass and return machine-readable cap/admission state."""

        connection = connect_v2(self.database_path)
        payloads_compacted = receipts_rolled = expired_rows_deleted = 0
        try:
            measured = self._measured_sizes(connection)
            database_bytes = (
                measured[0] if measured_database_bytes is None else measured_database_bytes
            )
            wal_bytes = measured[1] if measured_wal_bytes is None else measured_wal_bytes
            if database_bytes < 0 or wal_bytes < 0:
                raise ValueError("measured storage sizes cannot be negative")
            connection.execute("BEGIN IMMEDIATE")
            remaining = self.policy.maintenance_batch_rows
            payloads_compacted = self._compact_payloads(
                connection, now_us - self.policy.callback_payload_us, remaining
            )
            remaining -= payloads_compacted
            receipts_rolled = self._roll_receipts(
                connection, now_us - self.policy.receipt_us, remaining
            )
            remaining -= receipts_rolled
            expired_rows_deleted = self._prune_expired(connection, now_us, remaining)
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            connection.execute("PRAGMA incremental_vacuum(64)")
            post_maintenance_sizes = self._measured_sizes(connection)
            if measured_database_bytes is None:
                database_bytes = post_maintenance_sizes[0]
            if measured_wal_bytes is None:
                wal_bytes = post_maintenance_sizes[1]
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

        state = _cap_state(database_bytes, self.policy.database_cap_bytes)
        wal_over_cap = wal_bytes >= self.policy.wal_cap_bytes
        if wal_over_cap and state in {StorageCapState.NORMAL, StorageCapState.SOFT}:
            state = StorageCapState.DEGRADED
        return RetentionResult(
            cap_state=state,
            database_bytes=database_bytes,
            wal_bytes=wal_bytes,
            payloads_compacted=payloads_compacted,
            receipts_rolled=receipts_rolled,
            expired_rows_deleted=expired_rows_deleted,
            admission_allowed=state is not StorageCapState.FATAL,
            optional_feeds_allowed=state not in {StorageCapState.DEGRADED, StorageCapState.FATAL},
            required_action="STORAGE_CAP_FATAL" if state is StorageCapState.FATAL else None,
            checkpoint_attempted=True,
            incremental_vacuum_attempted=True,
        )
