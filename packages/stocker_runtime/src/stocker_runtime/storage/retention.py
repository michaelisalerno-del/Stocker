"""Bounded retention and visible storage-cap state for Stocker V2."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.storage.connection import connect_v2
from stocker_runtime.storage.repository import (
    CallbackReceiptRecord,
    callback_rows_hash,
    canonical_json_text,
    receipt_chain_hash,
)

DAY_US = 86_400_000_000
MAX_MAINTENANCE_BATCH_ROWS = 10_000

_RECEIPT_PROOF_SQL = """
AND receipt_batch_id IS NOT NULL
AND (EXISTS (
    SELECT 1 FROM callback_receipts receipt
    WHERE receipt.batch_id = callback_inbox.receipt_batch_id
      AND receipt.run_id = callback_inbox.run_id
      AND callback_inbox.source_sequence BETWEEN
          receipt.first_source_sequence AND receipt.last_source_sequence
) OR EXISTS (
    SELECT 1 FROM callback_compaction_watermarks watermark
    WHERE watermark.run_id = callback_inbox.run_id
      AND watermark.compacted_through_sequence >= callback_inbox.source_sequence
))
"""
ACK_PAYLOAD_CANDIDATES_SQL = """
SELECT source_sequence, run_id, receipt_batch_id
FROM callback_inbox INDEXED BY callback_inbox_terminal_idx
WHERE lifecycle = 'acknowledged' AND payload_json IS NOT NULL
  AND normalized_event_id IS NOT NULL AND acknowledged_at_us IS NOT NULL
  AND acknowledged_at_us <= ?
""" + _RECEIPT_PROOF_SQL + " ORDER BY acknowledged_at_us, source_sequence LIMIT ?"
FAILED_PAYLOAD_CANDIDATES_SQL = """
SELECT source_sequence, run_id, receipt_batch_id
FROM callback_inbox INDEXED BY callback_inbox_failed_payload_idx
WHERE lifecycle = 'failed' AND payload_json IS NOT NULL
  AND failure_code IS NOT NULL AND received_at_us <= ?
  AND EXISTS (SELECT 1 FROM runs terminal_run
      WHERE terminal_run.run_id = callback_inbox.run_id
        AND terminal_run.status IN ('stopped', 'fatal'))
""" + _RECEIPT_PROOF_SQL + " ORDER BY received_at_us, source_sequence LIMIT ?"
ACK_TOMBSTONE_CANDIDATES_SQL = """
SELECT source_sequence, run_id, receipt_batch_id
FROM callback_inbox INDEXED BY callback_inbox_ack_tombstone_idx
WHERE lifecycle = 'acknowledged' AND payload_json IS NULL
  AND acknowledged_at_us IS NOT NULL AND acknowledged_at_us <= ?
""" + _RECEIPT_PROOF_SQL + " ORDER BY acknowledged_at_us, source_sequence LIMIT ?"
FAILED_TOMBSTONE_CANDIDATES_SQL = """
SELECT source_sequence, run_id, receipt_batch_id
FROM callback_inbox INDEXED BY callback_inbox_failed_tombstone_idx
WHERE lifecycle = 'failed' AND payload_json IS NULL
  AND received_at_us <= ?
  AND EXISTS (SELECT 1 FROM runs terminal_run
      WHERE terminal_run.run_id = callback_inbox.run_id
        AND terminal_run.status IN ('stopped', 'fatal'))
""" + _RECEIPT_PROOF_SQL + " ORDER BY received_at_us, source_sequence LIMIT ?"


class RetentionInvariantError(RuntimeError):
    """Compaction cannot prove the evidence chain required for deletion."""


class MaintenanceDeadlineExceeded(RuntimeError):
    """A retention transaction exceeded its 100 ms hard deadline and was rolled back."""


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
    maintenance_transaction_ms: int = 100

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
            self.maintenance_transaction_ms,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("retention values must be positive")
        if self.maintenance_batch_rows > MAX_MAINTENANCE_BATCH_ROWS:
            raise ValueError("maintenance batches may not exceed 10,000 rows")
        if self.maintenance_transaction_ms > 100:
            raise ValueError("maintenance transactions may not exceed 100 ms")


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

    def __init__(
        self,
        database_path: str | Path,
        policy: RetentionPolicy | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.database_path = Path(database_path)
        self.policy = policy or RetentionPolicy()
        self._monotonic = monotonic

    def _measured_sizes(self, connection: sqlite3.Connection) -> tuple[int, int]:
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        wal_path = Path(f"{self.database_path}-wal")
        return page_count * page_size, wal_path.stat().st_size if wal_path.exists() else 0

    def _compact_payloads(self, connection: sqlite3.Connection, cutoff_us: int, limit: int) -> int:
        acknowledged = tuple(
            connection.execute(ACK_PAYLOAD_CANDIDATES_SQL, (cutoff_us, limit))
        )
        failed = tuple(
            connection.execute(
                FAILED_PAYLOAD_CANDIDATES_SQL,
                (cutoff_us, max(0, limit - len(acknowledged))),
            )
        )
        candidates = acknowledged + failed
        verified_batches: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (str(candidate["run_id"]), str(candidate["receipt_batch_id"]))
            receipt = connection.execute(
                "SELECT * FROM callback_receipts WHERE run_id = ? AND batch_id = ?",
                key,
            ).fetchone()
            if receipt is not None and key not in verified_batches:
                self._verify_granular_receipt(connection, receipt)
                verified_batches.add(key)
        connection.executemany(
            "UPDATE callback_inbox SET payload_json = NULL WHERE source_sequence = ?",
            ((int(row["source_sequence"]),) for row in candidates),
        )
        return len(candidates)

    def _verify_granular_receipt(
        self, connection: sqlite3.Connection, receipt: sqlite3.Row
    ) -> None:
        run_id = str(receipt["run_id"])
        target_first = int(receipt["first_source_sequence"])
        watermark = connection.execute(
            "SELECT compacted_through_sequence, last_receipt_chain_hash "
            "FROM callback_compaction_watermarks WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        after_sequence = -1
        expected_prior = "0" * 64
        if watermark is not None:
            after_sequence = int(watermark["compacted_through_sequence"])
            expected_prior = str(watermark["last_receipt_chain_hash"])
        chain = tuple(
            connection.execute(
                "SELECT * FROM callback_receipts WHERE run_id = ? "
                "AND first_source_sequence > ? AND first_source_sequence <= ? "
                "ORDER BY first_source_sequence, batch_id LIMIT ?",
                (run_id, after_sequence, target_first, MAX_MAINTENANCE_BATCH_ROWS + 1),
            )
        )
        if not chain or str(chain[-1]["batch_id"]) != str(receipt["batch_id"]):
            raise RetentionInvariantError("receipt predecessor chain is missing")
        if len(chain) > MAX_MAINTENANCE_BATCH_ROWS:
            raise RetentionInvariantError("receipt predecessor chain exceeds verification bound")
        expected_first = (
            int(chain[0]["first_source_sequence"]) if watermark is None else after_sequence + 1
        )
        for linked_receipt in chain:
            record = self._verified_receipt(
                connection,
                linked_receipt,
                expected_first=expected_first,
                expected_prior_hash=expected_prior,
            )
            expected_first = record.last_source_sequence + 1
            expected_prior = str(linked_receipt["chained_payload_hash"])

    def _receipt_candidates(
        self, connection: sqlite3.Connection, run_id: str, cutoff_us: int, remaining: int
    ) -> tuple[sqlite3.Row, ...]:
        total = int(
            connection.execute(
                "SELECT count(*) FROM callback_receipts WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        )
        excess = max(0, total - self.policy.max_receipts_per_run)
        rows = tuple(
            connection.execute(
                "SELECT * FROM callback_receipts WHERE run_id = ? "
                "ORDER BY first_source_sequence, batch_id LIMIT ?",
                (run_id, remaining),
            )
        )
        candidate_count = 0
        for index, row in enumerate(rows):
            if int(row["created_at_us"]) <= cutoff_us or index < excess:
                candidate_count += 1
            else:
                break
        return rows[: min(candidate_count, remaining)]

    def _verified_receipt(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        expected_first: int,
        expected_prior_hash: str,
    ) -> CallbackReceiptRecord:
        first = int(row["first_source_sequence"])
        last = int(row["last_source_sequence"])
        count = int(row["callback_count"])
        if first != expected_first or count != last - first + 1:
            raise RetentionInvariantError("receipt source sequence/count invariant failed")
        if count > MAX_MAINTENANCE_BATCH_ROWS:
            raise RetentionInvariantError("receipt callback count exceeds verification bound")
        if str(row["prior_chain_hash"]) != expected_prior_hash:
            raise RetentionInvariantError("receipt predecessor hash invariant failed")

        def counts(field: str) -> JsonValue:
            raw = str(row[field])
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise RetentionInvariantError(f"{field} is invalid JSON") from error
            if (
                not isinstance(value, dict)
                or any(
                    not isinstance(key, str)
                    or not isinstance(item, int)
                    or isinstance(item, bool)
                    or item < 0
                    for key, item in value.items()
                )
                or sum(value.values()) != count
                or canonical_json_text(cast(JsonValue, value), max_bytes=16_384) != raw
            ):
                raise RetentionInvariantError(f"{field} is not canonical valid counts")
            return cast(JsonValue, value)

        record = CallbackReceiptRecord(
            batch_id=str(row["batch_id"]),
            run_id=str(row["run_id"]),
            first_source_sequence=first,
            last_source_sequence=last,
            callback_count=count,
            first_received_at_us=int(row["first_received_at_us"]),
            last_received_at_us=int(row["last_received_at_us"]),
            kind_counts=counts("kind_counts_json"),
            status_counts=counts("status_counts_json"),
            callback_rows_hash=str(row["callback_rows_hash"]),
            first_normalized_event_id=(
                None
                if row["first_normalized_event_id"] is None
                else str(row["first_normalized_event_id"])
            ),
            last_normalized_event_id=(
                None
                if row["last_normalized_event_id"] is None
                else str(row["last_normalized_event_id"])
            ),
            created_at_us=int(row["created_at_us"]),
            prior_chain_hash=str(row["prior_chain_hash"]),
        )
        if record.last_received_at_us < record.first_received_at_us:
            raise RetentionInvariantError("receipt time range is reversed")
        if receipt_chain_hash(record) != str(row["chained_payload_hash"]):
            raise RetentionInvariantError("receipt content hash invariant failed")
        callback_rows = tuple(
            connection.execute(
                "SELECT event_uid, payload_sha256, run_id, source_sequence, callback_kind, "
                "lifecycle, received_at_us, provider_at_us, normalized_event_id, "
                "acknowledged_at_us, failure_code FROM callback_inbox "
                "WHERE run_id = ? AND source_sequence BETWEEN ? AND ? "
                "ORDER BY source_sequence LIMIT ?",
                (record.run_id, first, last, count + 1),
            )
        )
        if callback_rows:
            if len(callback_rows) != count:
                raise RetentionInvariantError("receipt callback row count invariant failed")
            derived_kind_counts: dict[str, int] = {}
            derived_status_counts: dict[str, int] = {}
            for callback in callback_rows:
                kind = str(callback["callback_kind"])
                status = str(callback["lifecycle"])
                derived_kind_counts[kind] = derived_kind_counts.get(kind, 0) + 1
                derived_status_counts[status] = derived_status_counts.get(status, 0) + 1
            first_callback = callback_rows[0]
            last_callback = callback_rows[-1]
            if (
                int(first_callback["source_sequence"]) != first
                or int(last_callback["source_sequence"]) != last
                or record.kind_counts != derived_kind_counts
                or record.status_counts != derived_status_counts
                or record.first_received_at_us != int(first_callback["received_at_us"])
                or record.last_received_at_us != int(last_callback["received_at_us"])
                or record.first_normalized_event_id
                != first_callback["normalized_event_id"]
                or record.last_normalized_event_id
                != last_callback["normalized_event_id"]
                or record.callback_rows_hash
                != callback_rows_hash(tuple(dict(item) for item in callback_rows))
            ):
                raise RetentionInvariantError("receipt does not match authoritative callback rows")
        return record

    def _roll_receipts(
        self,
        connection: sqlite3.Connection,
        cutoff_us: int,
        updated_at_us: int,
        limit: int,
    ) -> tuple[int, int]:
        rolled = 0
        changed = 0
        for run_row in connection.execute(
            "SELECT DISTINCT run_id FROM callback_receipts ORDER BY run_id"
        ):
            if changed + 1 >= limit:
                break
            run_id = str(run_row[0])
            watermark = connection.execute(
                "SELECT * FROM callback_compaction_watermarks WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            candidates = self._receipt_candidates(
                connection,
                run_id,
                cutoff_us,
                limit - changed - 1,
            )
            if not candidates:
                continue
            expected_first = int(candidates[0]["first_source_sequence"])
            expected_prior_hash = "0" * 64
            if watermark is not None:
                expected_first = int(watermark["compacted_through_sequence"]) + 1
                expected_prior_hash = str(watermark["last_receipt_chain_hash"])
            verified: list[CallbackReceiptRecord] = []
            chain_hashes: list[str] = []
            for receipt in candidates:
                record = self._verified_receipt(
                    connection,
                    receipt,
                    expected_first=expected_first,
                    expected_prior_hash=expected_prior_hash,
                )
                verified.append(record)
                expected_first = record.last_source_sequence + 1
                expected_prior_hash = str(receipt["chained_payload_hash"])
                chain_hashes.append(expected_prior_hash)
            previous_count = 0 if watermark is None else int(watermark["cumulative_callback_count"])
            previous_first = None if watermark is None else watermark["first_received_at_us"]
            previous_rollup_hash = (
                "0" * 64 if watermark is None else str(watermark["rolled_receipt_chain_hash"])
            )
            callback_count = previous_count + sum(item.callback_count for item in verified)
            first_received = (
                verified[0].first_received_at_us if previous_first is None else int(previous_first)
            )
            last = verified[-1]
            rollup_records = [
                {
                    "record": json.loads(
                        canonical_json_text(
                            cast(
                                JsonValue,
                                {
                                    "batch_id": item.batch_id,
                                    "run_id": item.run_id,
                                    "first_source_sequence": item.first_source_sequence,
                                    "last_source_sequence": item.last_source_sequence,
                                    "callback_count": item.callback_count,
                                    "first_received_at_us": item.first_received_at_us,
                                    "last_received_at_us": item.last_received_at_us,
                                    "kind_counts": item.kind_counts,
                                    "status_counts": item.status_counts,
                                    "callback_rows_hash": item.callback_rows_hash,
                                    "first_normalized_event_id": item.first_normalized_event_id,
                                    "last_normalized_event_id": item.last_normalized_event_id,
                                    "created_at_us": item.created_at_us,
                                    "prior_chain_hash": item.prior_chain_hash,
                                },
                            ),
                            max_bytes=65_536,
                        )
                    ),
                    "chain_hash": chain_hash,
                }
                for item, chain_hash in zip(verified, chain_hashes, strict=True)
            ]
            rollup_material = cast(
                JsonValue,
                {
                    "prior_rollup_hash": previous_rollup_hash,
                    "receipts": rollup_records,
                },
            )
            rollup_hash = hashlib.sha256(canonical_json_bytes(rollup_material)).hexdigest()
            connection.execute(
                """
                INSERT INTO callback_compaction_watermarks(
                    run_id, compacted_through_sequence, cumulative_callback_count,
                    first_received_at_us, last_received_at_us, rolled_receipt_chain_hash,
                    last_receipt_chain_hash, updated_at_us
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    compacted_through_sequence = excluded.compacted_through_sequence,
                    cumulative_callback_count = excluded.cumulative_callback_count,
                    first_received_at_us = excluded.first_received_at_us,
                    last_received_at_us = excluded.last_received_at_us,
                    rolled_receipt_chain_hash = excluded.rolled_receipt_chain_hash,
                    last_receipt_chain_hash = excluded.last_receipt_chain_hash,
                    updated_at_us = excluded.updated_at_us
                """,
                (
                    run_id,
                    last.last_source_sequence,
                    callback_count,
                    first_received,
                    last.last_received_at_us,
                    rollup_hash,
                    chain_hashes[-1],
                    updated_at_us,
                ),
            )
            connection.executemany(
                "DELETE FROM callback_receipts WHERE batch_id = ?",
                ((str(row["batch_id"]),) for row in candidates),
            )
            rolled += len(candidates)
            changed += len(candidates) + 1
        return rolled, changed

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
        order_by = {
            "shadow_marks": "marked_at_us, position_id",
            "shadow_outcomes": "outcome_at_us, position_id",
            "shadow_positions": "closed_at_us, position_id",
            "idea_outputs": "emitted_at_us, output_id",
            "market_events": "event_at_us, event_kind, event_id",
            "subscriptions": "closed_at_us, subscription_id",
            "incidents": "resolved_at_us, incident_id",
            "gaps": "resolved_at_us, gap_id",
        }.get(table, identity_column)
        cursor = connection.execute(
            f"DELETE FROM {table} WHERE {identity_column} IN ("  # noqa: S608
            f"SELECT {identity_column} FROM {table} WHERE {predicate} "
            f"ORDER BY {order_by} LIMIT ?)",
            (*parameters, remaining),
        )
        return cursor.rowcount

    def _prune_callback_tombstones(
        self, connection: sqlite3.Connection, cutoff_us: int, limit: int
    ) -> int:
        if limit <= 0:
            return 0
        acknowledged = tuple(
            connection.execute(ACK_TOMBSTONE_CANDIDATES_SQL, (cutoff_us, limit))
        )
        failed = tuple(
            connection.execute(
                FAILED_TOMBSTONE_CANDIDATES_SQL,
                (cutoff_us, max(0, limit - len(acknowledged))),
            )
        )
        candidates = acknowledged + failed
        grouped_sequences: dict[tuple[str, str], set[int]] = {}
        for candidate in candidates:
            key = (str(candidate["run_id"]), str(candidate["receipt_batch_id"]))
            grouped_sequences.setdefault(key, set()).add(int(candidate["source_sequence"]))
        complete_candidates: list[sqlite3.Row] = []
        for key, sequences in grouped_sequences.items():
            receipt = connection.execute(
                "SELECT * FROM callback_receipts WHERE run_id = ? AND batch_id = ?", key
            ).fetchone()
            if receipt is None:
                complete_candidates.extend(
                    row
                    for row in candidates
                    if (str(row["run_id"]), str(row["receipt_batch_id"])) == key
                )
                continue
            authoritative = {
                int(row[0])
                for row in connection.execute(
                    "SELECT source_sequence FROM callback_inbox WHERE run_id = ? "
                    "AND source_sequence BETWEEN ? AND ?",
                    (
                        key[0],
                        int(receipt["first_source_sequence"]),
                        int(receipt["last_source_sequence"]),
                    ),
                )
            }
            if sequences == authoritative:
                complete_candidates.extend(
                    row
                    for row in candidates
                    if (str(row["run_id"]), str(row["receipt_batch_id"])) == key
                )
        candidates = tuple(complete_candidates)
        verified_batches: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (str(candidate["run_id"]), str(candidate["receipt_batch_id"]))
            receipt = connection.execute(
                "SELECT * FROM callback_receipts WHERE run_id = ? AND batch_id = ?",
                key,
            ).fetchone()
            if receipt is not None and key not in verified_batches:
                self._verify_granular_receipt(connection, receipt)
                verified_batches.add(key)
        connection.executemany(
            "DELETE FROM callback_inbox WHERE source_sequence = ?",
            ((int(row["source_sequence"]),) for row in candidates),
        )
        return len(candidates)

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
            "WHERE l.event_id = market_events.event_id)",
            now_us - self.policy.raw_market_event_us,
        )
        prune(
            "market_events",
            "event_id",
            "event_kind IN ('bar', 'historical_bar') AND event_at_us <= ? "
            "AND NOT EXISTS (SELECT 1 FROM market_latest l "
            "WHERE l.event_id = market_events.event_id)",
            now_us - self.policy.completed_bar_us,
        )
        deleted += self._prune_callback_tombstones(
            connection,
            now_us - self.policy.tombstone_us,
            limit - deleted,
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
        deadline_hit = False
        try:
            measured = self._measured_sizes(connection)
            database_bytes = (
                measured[0] if measured_database_bytes is None else measured_database_bytes
            )
            wal_bytes = measured[1] if measured_wal_bytes is None else measured_wal_bytes
            if database_bytes < 0 or wal_bytes < 0:
                raise ValueError("measured storage sizes cannot be negative")
            deadline = self._monotonic() + self.policy.maintenance_transaction_ms / 1_000

            def progress() -> int:
                nonlocal deadline_hit
                deadline_hit = self._monotonic() >= deadline
                return 1 if deadline_hit else 0

            def check_deadline() -> None:
                nonlocal deadline_hit
                deadline_hit = self._monotonic() >= deadline
                if deadline_hit:
                    raise MaintenanceDeadlineExceeded(
                        "retention transaction exceeded its configured deadline"
                    )

            connection.set_progress_handler(progress, 1_000)
            connection.execute("BEGIN IMMEDIATE")
            check_deadline()
            remaining = self.policy.maintenance_batch_rows
            payloads_compacted = self._compact_payloads(
                connection, now_us - self.policy.callback_payload_us, remaining
            )
            check_deadline()
            remaining -= payloads_compacted
            receipts_rolled, receipt_rows_changed = self._roll_receipts(
                connection, now_us - self.policy.receipt_us, now_us, remaining
            )
            check_deadline()
            remaining -= receipt_rows_changed
            expired_rows_deleted = self._prune_expired(connection, now_us, remaining)
            check_deadline()
            connection.commit()
            connection.set_progress_handler(None, 0)
            connection.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            connection.execute("PRAGMA incremental_vacuum(64)")
            post_maintenance_sizes = self._measured_sizes(connection)
            if measured_database_bytes is None:
                database_bytes = post_maintenance_sizes[0]
            if measured_wal_bytes is None:
                wal_bytes = post_maintenance_sizes[1]
        except Exception as error:
            if connection.in_transaction:
                connection.rollback()
            if deadline_hit and isinstance(error, sqlite3.OperationalError):
                raise MaintenanceDeadlineExceeded(
                    "retention transaction exceeded its configured deadline"
                ) from error
            raise
        finally:
            connection.set_progress_handler(None, 0)
            connection.close()

        state = _cap_state(database_bytes, self.policy.database_cap_bytes)
        wal_over_cap = wal_bytes >= self.policy.wal_cap_bytes
        database_fatal = state is StorageCapState.FATAL
        if wal_over_cap:
            state = StorageCapState.FATAL
        required_action = None
        if database_fatal:
            required_action = "STORAGE_CAP_FATAL"
        elif wal_over_cap:
            required_action = "WAL_CAP_FATAL"
        return RetentionResult(
            cap_state=state,
            database_bytes=database_bytes,
            wal_bytes=wal_bytes,
            payloads_compacted=payloads_compacted,
            receipts_rolled=receipts_rolled,
            expired_rows_deleted=expired_rows_deleted,
            admission_allowed=state is not StorageCapState.FATAL,
            optional_feeds_allowed=state not in {StorageCapState.DEGRADED, StorageCapState.FATAL},
            required_action=required_action,
            checkpoint_attempted=True,
            incremental_vacuum_attempted=True,
        )
