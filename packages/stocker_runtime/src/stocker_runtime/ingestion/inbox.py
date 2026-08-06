"""Durable admission and projection boundary for IBKR market-data callbacks."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.storage import (
    MAX_CALLBACK_PAYLOAD_BYTES,
    MAX_NONTERMINAL_CALLBACK_ROWS,
    CallbackReceiptRecord,
    JsonAdmissionError,
    callback_rows_hash,
    canonical_json_text,
    connect_v2,
    receipt_chain_hash,
)


class InboxAdmissionError(RuntimeError):
    """A callback could not be durably admitted without violating an invariant."""


class InboxFullError(InboxAdmissionError):
    """The hard nonterminal callback bound has closed admission."""


class CallbackIdentityCollision(InboxAdmissionError):
    """A deterministic callback identity names different durable content."""


class NormalizationError(ValueError):
    """A callback is durable but cannot safely become a typed market event."""


@dataclass(frozen=True)
class CallbackFence:
    """Durable recorder, socket, and request ownership presented by one callback."""

    run_id: str
    recorder_generation: int
    connection_generation: int
    request_id: int | None
    subscription_id: str | None


@dataclass(frozen=True)
class MarketDataCallback:
    """One bounded callback captured at the external API boundary."""

    callback_kind: str
    received_at_us: int
    provider_at_us: int | None
    payload: JsonValue


@dataclass(frozen=True)
class AdmissionResult:
    source_sequence: int
    event_uid: str
    inserted: bool


@dataclass(frozen=True)
class LeasedCallback:
    source_sequence: int
    event_uid: str
    run_id: str
    recorder_generation: int
    connection_generation: int
    request_id: int | None
    callback_kind: str
    received_at_us: int
    provider_at_us: int | None
    payload: JsonValue
    payload_sha256: str
    lease_owner: str


@dataclass(frozen=True)
class ProjectionResult:
    event_id: str
    inserted: bool


def _event_uid(fence: CallbackFence, callback: MarketDataCallback, payload_hash: str) -> str:
    material: JsonValue = {
        "run_id": fence.run_id,
        "recorder_generation": fence.recorder_generation,
        "connection_generation": fence.connection_generation,
        "request_id": fence.request_id,
        "subscription_id": fence.subscription_id,
        "callback_kind": callback.callback_kind,
        "received_at_us": callback.received_at_us,
        "provider_at_us": callback.provider_at_us,
        "payload_sha256": payload_hash,
    }
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


class CallbackInbox:
    """Admit before return, then lease and project callbacks in source order."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        max_nonterminal_rows: int = MAX_NONTERMINAL_CALLBACK_ROWS,
    ) -> None:
        if not 1 <= max_nonterminal_rows <= MAX_NONTERMINAL_CALLBACK_ROWS:
            raise ValueError("nonterminal callback bound must be between 1 and 50,000")
        self.database_path = Path(database_path)
        self.max_nonterminal_rows = max_nonterminal_rows
        with connect_v2(self.database_path):
            pass

    def _connect(self) -> sqlite3.Connection:
        """Open after constructor-time schema verification under the sole-writer lease."""

        return connect_v2(self.database_path, verify_schema=False)

    def nonterminal_count(self) -> int:
        """Return the bounded recovery backlog for lifecycle orchestration."""

        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending','leased')"
                ).fetchone()[0]
            )

    def admit(self, fence: CallbackFence, callback: MarketDataCallback) -> AdmissionResult:
        """Commit a canonical callback before returning to the external API thread."""

        if not callback.callback_kind or callback.received_at_us < 0:
            raise InboxAdmissionError("callback identity and receive time are required")
        try:
            payload_json = canonical_json_text(
                callback.payload, max_bytes=MAX_CALLBACK_PAYLOAD_BYTES
            )
        except JsonAdmissionError as error:
            raise InboxAdmissionError(str(error)) from error
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        event_uid = _event_uid(fence, callback, payload_hash)
        try:
            connection = self._connect()
        except (OSError, sqlite3.Error) as error:
            raise InboxAdmissionError(f"callback durable admission failed: {error}") from error
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM callback_inbox WHERE event_uid = ?", (event_uid,)
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["run_id"]) != fence.run_id
                    or int(existing["recorder_generation"]) != fence.recorder_generation
                    or int(existing["connection_generation"]) != fence.connection_generation
                    or existing["request_id"] != fence.request_id
                    or str(existing["callback_kind"]) != callback.callback_kind
                    or int(existing["received_at_us"]) != callback.received_at_us
                    or existing["provider_at_us"] != callback.provider_at_us
                    or str(existing["payload_sha256"]) != payload_hash
                    or existing["payload_json"] != payload_json
                ):
                    raise CallbackIdentityCollision(
                        f"callback identity {event_uid} names different content"
                    )
                connection.commit()
                return AdmissionResult(int(existing["source_sequence"]), event_uid, False)
            nonterminal = int(
                connection.execute(
                    "SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending', 'leased')"
                ).fetchone()[0]
            )
            if nonterminal >= self.max_nonterminal_rows:
                self._record_fatal(connection, fence.run_id, callback.received_at_us, "INBOX_FULL")
                connection.commit()
                raise InboxFullError("callback inbox hard limit of 50,000 is reached")
            prior = connection.execute(
                "SELECT received_at_us FROM callback_inbox WHERE run_id = ? "
                "ORDER BY source_sequence DESC LIMIT 1",
                (fence.run_id,),
            ).fetchone()
            if prior is not None and callback.received_at_us < int(prior[0]):
                self._record_fatal(
                    connection,
                    fence.run_id,
                    callback.received_at_us,
                    "CALLBACK_ORDERING_LOSS",
                )
                connection.commit()
                raise InboxAdmissionError("callback receive ordering moved backwards")
            fence_failure = self._fence_failure(connection, fence)
            lifecycle = "failed" if fence_failure is not None else "pending"
            cursor = connection.execute(
                """
                INSERT INTO callback_inbox(
                    event_uid, run_id, recorder_generation, connection_generation,
                    request_id, callback_kind, received_at_us, provider_at_us,
                    payload_json, payload_sha256, lifecycle, failure_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_uid,
                    fence.run_id,
                    fence.recorder_generation,
                    fence.connection_generation,
                    fence.request_id,
                    callback.callback_kind,
                    callback.received_at_us,
                    callback.provider_at_us,
                    payload_json,
                    payload_hash,
                    lifecycle,
                    fence_failure,
                ),
            )
            if cursor.lastrowid is None:
                raise InboxAdmissionError("callback admission did not allocate a source sequence")
            sequence = cursor.lastrowid
            if fence_failure is not None:
                self._record_scoped_gap(
                    connection,
                    fence,
                    callback.received_at_us,
                    fence_failure,
                    sequence,
                )
            connection.execute(
                "UPDATE runtime_state SET callback_heartbeat_at_us = ?, "
                "admission_heartbeat_at_us = ?, inbox_nonterminal_count = ? "
                "WHERE run_id = ? AND recorder_generation = ?",
                (
                    callback.received_at_us,
                    callback.received_at_us,
                    nonterminal + int(fence_failure is None),
                    fence.run_id,
                    fence.recorder_generation,
                ),
            )
            connection.commit()
            return AdmissionResult(sequence, event_uid, True)
        except InboxFullError:
            raise
        except sqlite3.Error as error:
            if connection.in_transaction:
                connection.rollback()
            raise InboxAdmissionError(f"callback durable admission failed: {error}") from error
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _fence_failure(connection: sqlite3.Connection, fence: CallbackFence) -> str | None:
        generation = connection.execute(
            "SELECT ended_at_us FROM recorder_generations WHERE run_id = ? AND generation = ?",
            (fence.run_id, fence.recorder_generation),
        ).fetchone()
        if generation is None:
            raise InboxAdmissionError("callback recorder generation is unknown")
        if generation["ended_at_us"] is not None:
            return "STALE_RECORDER_GENERATION"
        if fence.subscription_id is not None:
            subscription = connection.execute(
                "SELECT 1 FROM subscriptions WHERE subscription_id = ? AND run_id = ? "
                "AND recorder_generation = ? AND connection_generation = ? AND request_id = ? "
                "AND lifecycle = 'active'",
                (
                    fence.subscription_id,
                    fence.run_id,
                    fence.recorder_generation,
                    fence.connection_generation,
                    fence.request_id,
                ),
            ).fetchone()
            if subscription is None:
                return "STALE_REQUEST_GENERATION"
        return None

    @staticmethod
    def _record_scoped_gap(
        connection: sqlite3.Connection,
        fence: CallbackFence,
        opened_at_us: int,
        code: str,
        sequence: int,
    ) -> None:
        gap_id = hashlib.sha256(
            f"{fence.run_id}|{fence.subscription_id}|{code}|{sequence}".encode()
        ).hexdigest()
        connection.execute(
            "INSERT INTO gaps(gap_id, run_id, subscription_id, started_at_us, reason, "
            "data_loss_possible, continuity_required) VALUES (?, ?, ?, ?, ?, 1, 1)",
            (gap_id, fence.run_id, fence.subscription_id, opened_at_us, code),
        )
        incident_id = hashlib.sha256(
            f"{fence.run_id}|{fence.subscription_id}|{code}|{sequence}|incident".encode()
        ).hexdigest()
        connection.execute(
            "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
            "subscription_id, opened_at_us, details_json) "
            "VALUES (?, ?, 'callback', 'degraded', ?, ?, ?, '{}')",
            (
                incident_id,
                fence.run_id,
                code,
                fence.subscription_id,
                opened_at_us,
            ),
        )

    @staticmethod
    def _record_fatal(
        connection: sqlite3.Connection, run_id: str, opened_at_us: int, code: str
    ) -> None:
        incident_id = hashlib.sha256(f"{run_id}|{code}".encode()).hexdigest()
        connection.execute("UPDATE runs SET status = 'fatal' WHERE run_id = ?", (run_id,))
        connection.execute(
            "UPDATE runtime_state SET lifecycle = 'fatal', reason = ? WHERE run_id = ?",
            (code, run_id),
        )
        connection.execute(
            "INSERT OR IGNORE INTO incidents(incident_id, run_id, scope, severity, code, "
            "opened_at_us, details_json) VALUES (?, ?, 'recorder', 'fatal', ?, ?, '{}')",
            (incident_id, run_id, code, opened_at_us),
        )

    def reclaim_expired_leases(self, *, now_us: int) -> int:
        """Return expired leases to pending without changing source order."""

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE callback_inbox SET lifecycle = 'pending', lease_owner = NULL, "
                "lease_expires_at_us = NULL WHERE lifecycle = 'leased' "
                "AND lease_expires_at_us <= ?",
                (now_us,),
            )
            return cursor.rowcount

    def lease_pending(
        self,
        owner: str,
        *,
        now_us: int,
        lease_us: int,
        limit: int,
    ) -> tuple[LeasedCallback, ...]:
        """Lease a bounded pending prefix; never jump over an earlier active lease."""

        if not owner or lease_us <= 0 or not 1 <= limit <= 10_000:
            raise ValueError("lease owner, duration, and bounded limit are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = tuple(
                connection.execute(
                    "SELECT * FROM callback_inbox WHERE lifecycle IN ('pending', 'leased') "
                    "ORDER BY source_sequence LIMIT ?",
                    (limit,),
                )
            )
            selected: list[sqlite3.Row] = []
            for row in rows:
                if str(row["lifecycle"]) != "pending":
                    break
                selected.append(row)
            if selected:
                connection.executemany(
                    "UPDATE callback_inbox SET lifecycle = 'leased', lease_owner = ?, "
                    "lease_expires_at_us = ?, attempts = attempts + 1 "
                    "WHERE source_sequence = ? AND lifecycle = 'pending'",
                    ((owner, now_us + lease_us, int(row["source_sequence"])) for row in selected),
                )
            connection.commit()
            return tuple(self._leased(row, owner) for row in selected)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _leased(row: sqlite3.Row, owner: str) -> LeasedCallback:
        parsed = json.loads(str(row["payload_json"]))
        return LeasedCallback(
            source_sequence=int(row["source_sequence"]),
            event_uid=str(row["event_uid"]),
            run_id=str(row["run_id"]),
            recorder_generation=int(row["recorder_generation"]),
            connection_generation=int(row["connection_generation"]),
            request_id=None if row["request_id"] is None else int(row["request_id"]),
            callback_kind=str(row["callback_kind"]),
            received_at_us=int(row["received_at_us"]),
            provider_at_us=(None if row["provider_at_us"] is None else int(row["provider_at_us"])),
            payload=cast(JsonValue, parsed),
            payload_sha256=str(row["payload_sha256"]),
            lease_owner=owner,
        )

    @staticmethod
    def _number(payload: Mapping[str, object], name: str) -> float | None:
        value = payload.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise NormalizationError(f"{name} must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise NormalizationError(f"{name} must be a finite number")
        return result

    def project(self, leased: LeasedCallback) -> ProjectionResult:
        """Idempotently write one typed event and its latest projection in one transaction."""

        if not isinstance(leased.payload, Mapping):
            raise NormalizationError("callback payload must be an object")
        payload = cast(Mapping[str, object], leased.payload)
        event_at = payload.get("event_at_us")
        if isinstance(event_at, bool) or not isinstance(event_at, int) or event_at < 0:
            raise NormalizationError("event_at_us must be a nonnegative integer")
        if leased.callback_kind not in {"quote", "trade", "bar"}:
            raise NormalizationError("callback kind is not a normalized market-data surface")
        values = {
            field: self._number(payload, field)
            for field in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "bid",
                "ask",
                "last",
                "size",
            )
        }
        payload_json = canonical_json_text(
            cast(JsonValue, leased.payload), max_bytes=MAX_CALLBACK_PAYLOAD_BYTES
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            inbox_row = connection.execute(
                "SELECT lifecycle, lease_owner, event_uid FROM callback_inbox "
                "WHERE source_sequence = ?",
                (leased.source_sequence,),
            ).fetchone()
            if (
                inbox_row is None
                or str(inbox_row["lifecycle"]) != "leased"
                or str(inbox_row["lease_owner"]) != leased.lease_owner
                or str(inbox_row["event_uid"]) != leased.event_uid
            ):
                raise InboxAdmissionError("callback lease is no longer owned")
            subscription = connection.execute(
                "SELECT instrument_id, feed_kind FROM subscriptions WHERE run_id = ? "
                "AND recorder_generation = ? AND connection_generation = ? AND request_id = ?",
                (
                    leased.run_id,
                    leased.recorder_generation,
                    leased.connection_generation,
                    leased.request_id,
                ),
            ).fetchone()
            if subscription is None:
                raise NormalizationError("callback subscription provenance is absent")
            event_id = leased.event_uid
            content = (
                leased.run_id,
                leased.source_sequence,
                str(subscription["instrument_id"]),
                str(subscription["feed_kind"]),
                leased.callback_kind,
                event_at,
                leased.received_at_us,
                leased.connection_generation,
                values["open"],
                values["high"],
                values["low"],
                values["close"],
                values["volume"],
                values["bid"],
                values["ask"],
                values["last"],
                values["size"],
                payload_json,
                leased.payload_sha256,
            )
            existing = connection.execute(
                "SELECT run_id, source_sequence, instrument_id, feed_kind, event_kind, "
                "event_at_us, received_at_us, connection_generation, open_value, high_value, "
                "low_value, close_value, volume_value, bid_value, ask_value, last_value, "
                "size_value, payload_json, payload_sha256 FROM market_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            inserted = existing is None
            if existing is None:
                connection.execute(
                    "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                    "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                    "open_value, high_value, low_value, close_value, volume_value, bid_value, "
                    "ask_value, last_value, size_value, payload_json, payload_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (event_id, *content),
                )
            elif tuple(existing) != content:
                raise CallbackIdentityCollision(
                    f"normalized event identity {event_id} names different content"
                )
            latest = connection.execute(
                "SELECT event.source_sequence FROM market_latest latest "
                "JOIN market_events event ON event.event_id = latest.event_id "
                "WHERE latest.instrument_id = ? AND latest.feed_kind = ?",
                (str(subscription["instrument_id"]), str(subscription["feed_kind"])),
            ).fetchone()
            if latest is None or int(latest[0]) <= leased.source_sequence:
                connection.execute(
                    "INSERT INTO market_latest(run_id, instrument_id, feed_kind, event_id, "
                    "event_at_us, received_at_us, event_kind, quality_bits, bid_value, ask_value, "
                    "last_value, close_value) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?) "
                    "ON CONFLICT(instrument_id, feed_kind) DO UPDATE SET run_id=excluded.run_id, "
                    "event_id=excluded.event_id, event_at_us=excluded.event_at_us, "
                    "received_at_us=excluded.received_at_us, event_kind=excluded.event_kind, "
                    "quality_bits=excluded.quality_bits, bid_value=excluded.bid_value, "
                    "ask_value=excluded.ask_value, last_value=excluded.last_value, "
                    "close_value=excluded.close_value",
                    (
                        leased.run_id,
                        str(subscription["instrument_id"]),
                        str(subscription["feed_kind"]),
                        event_id,
                        event_at,
                        leased.received_at_us,
                        leased.callback_kind,
                        values["bid"],
                        values["ask"],
                        values["last"],
                        values["close"],
                    ),
                )
            connection.execute(
                "UPDATE subscriptions SET latest_event_id = ? WHERE run_id = ? "
                "AND connection_generation = ? AND request_id = ?",
                (event_id, leased.run_id, leased.connection_generation, leased.request_id),
            )
            connection.execute(
                "UPDATE runtime_state SET projection_heartbeat_at_us = ? WHERE run_id = ?",
                (leased.received_at_us, leased.run_id),
            )
            connection.commit()
            return ProjectionResult(event_id, inserted)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def acknowledge(
        self, leased: LeasedCallback, event_id: str, *, acknowledged_at_us: int
    ) -> None:
        """Mark terminal only after the exact durable projection is present."""

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE callback_inbox SET lifecycle = 'acknowledged', lease_owner = NULL, "
                "lease_expires_at_us = NULL, normalized_event_id = ?, acknowledged_at_us = ? "
                "WHERE source_sequence = ? AND lifecycle = 'leased' AND lease_owner = ?",
                (event_id, acknowledged_at_us, leased.source_sequence, leased.lease_owner),
            )
            if cursor.rowcount != 1:
                raise InboxAdmissionError("callback acknowledgement lost its lease")
            connection.execute(
                "UPDATE runtime_state SET inbox_nonterminal_count = "
                "(SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending','leased')) "
                "WHERE run_id = ?",
                (leased.run_id,),
            )
            subscription = connection.execute(
                "SELECT subscription_id, instrument_id, feed_kind FROM subscriptions "
                "WHERE run_id=? "
                "AND connection_generation=? AND request_id=?",
                (
                    leased.run_id,
                    leased.connection_generation,
                    leased.request_id,
                ),
            ).fetchone()
            if subscription is not None:
                connection.execute(
                    "UPDATE gaps SET ended_at_us=?, resolved_at_us=? WHERE run_id=? "
                    "AND subscription_id IN (SELECT subscription_id FROM subscriptions "
                    "WHERE run_id=? AND instrument_id=? AND feed_kind=?) "
                    "AND resolved_at_us IS NULL",
                    (
                        acknowledged_at_us,
                        acknowledged_at_us,
                        leased.run_id,
                        leased.run_id,
                        str(subscription["instrument_id"]),
                        str(subscription["feed_kind"]),
                    ),
                )

    def fail(self, leased: LeasedCallback, code: str, *, failed_at_us: int) -> None:
        """Quarantine one poison callback while leaving later callbacks serviceable."""

        if not code:
            raise ValueError("failure code is required")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE callback_inbox SET lifecycle = 'failed', lease_owner = NULL, "
                "lease_expires_at_us = NULL, failure_code = ? WHERE source_sequence = ? "
                "AND lifecycle = 'leased' AND lease_owner = ?",
                (code, leased.source_sequence, leased.lease_owner),
            )
            if cursor.rowcount != 1:
                raise InboxAdmissionError("failed callback lost its lease")
            incident_id = hashlib.sha256(
                f"{leased.run_id}|{leased.source_sequence}|{code}".encode()
            ).hexdigest()
            connection.execute(
                "INSERT INTO incidents(incident_id, run_id, scope, severity, code, "
                "opened_at_us, details_json) VALUES (?, ?, 'callback', 'degraded', ?, ?, '{}')",
                (incident_id, leased.run_id, code, failed_at_us),
            )
            connection.execute(
                "UPDATE runtime_state SET inbox_nonterminal_count = "
                "(SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending','leased')) "
                "WHERE run_id = ?",
                (leased.run_id,),
            )

    def create_receipt(
        self, run_id: str, *, created_at_us: int, limit: int = 10_000
    ) -> CallbackReceiptRecord | None:
        """Receipt one contiguous terminal prefix using the Phase 2 evidence contract."""

        if not 1 <= limit <= 10_000:
            raise ValueError("receipt limit must be between 1 and 10,000")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                "SELECT last_source_sequence, chained_payload_hash FROM callback_receipts "
                "WHERE run_id = ? ORDER BY last_source_sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            watermark = connection.execute(
                "SELECT compacted_through_sequence, last_receipt_chain_hash "
                "FROM callback_compaction_watermarks WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            after = -1
            prior_hash = "0" * 64
            if watermark is not None:
                after = int(watermark["compacted_through_sequence"])
                prior_hash = str(watermark["last_receipt_chain_hash"])
            if previous is not None:
                after = int(previous["last_source_sequence"])
                prior_hash = str(previous["chained_payload_hash"])
            rows = tuple(
                connection.execute(
                    "SELECT event_uid, payload_sha256, run_id, source_sequence, callback_kind, "
                    "lifecycle, received_at_us, provider_at_us, normalized_event_id, "
                    "acknowledged_at_us, failure_code FROM callback_inbox WHERE run_id = ? "
                    "AND source_sequence > ? ORDER BY source_sequence LIMIT ?",
                    (run_id, after, limit),
                )
            )
            terminal: list[sqlite3.Row] = []
            for row in rows:
                if str(row["lifecycle"]) not in {"acknowledged", "failed"}:
                    break
                terminal.append(row)
            if not terminal:
                connection.commit()
                return None
            first = terminal[0]
            last = terminal[-1]
            if int(last["source_sequence"]) - int(first["source_sequence"]) + 1 != len(terminal):
                raise InboxAdmissionError("receipt callback source sequence is not contiguous")
            if int(last["received_at_us"]) < int(first["received_at_us"]):
                raise InboxAdmissionError("receipt callback receive order is reversed")
            row_hash = callback_rows_hash(tuple(dict(row) for row in terminal))
            batch_id = hashlib.sha256(
                f"{run_id}|{first['source_sequence']}|{last['source_sequence']}|{row_hash}".encode()
            ).hexdigest()
            kind_counts = dict(Counter(str(row["callback_kind"]) for row in terminal))
            status_counts = dict(Counter(str(row["lifecycle"]) for row in terminal))
            record = CallbackReceiptRecord(
                batch_id=batch_id,
                run_id=run_id,
                first_source_sequence=int(first["source_sequence"]),
                last_source_sequence=int(last["source_sequence"]),
                callback_count=len(terminal),
                first_received_at_us=int(first["received_at_us"]),
                last_received_at_us=int(last["received_at_us"]),
                kind_counts=cast(JsonValue, kind_counts),
                status_counts=cast(JsonValue, status_counts),
                callback_rows_hash=row_hash,
                first_normalized_event_id=(
                    None
                    if first["normalized_event_id"] is None
                    else str(first["normalized_event_id"])
                ),
                last_normalized_event_id=(
                    None
                    if last["normalized_event_id"] is None
                    else str(last["normalized_event_id"])
                ),
                created_at_us=created_at_us,
                prior_chain_hash=prior_hash,
            )
            connection.execute(
                "INSERT INTO callback_receipts(batch_id, run_id, first_source_sequence, "
                "last_source_sequence, callback_count, first_received_at_us, last_received_at_us, "
                "kind_counts_json, status_counts_json, callback_rows_hash, prior_chain_hash, "
                "chained_payload_hash, first_normalized_event_id, last_normalized_event_id, "
                "created_at_us) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.batch_id,
                    record.run_id,
                    record.first_source_sequence,
                    record.last_source_sequence,
                    record.callback_count,
                    record.first_received_at_us,
                    record.last_received_at_us,
                    canonical_json_text(record.kind_counts, max_bytes=16_384),
                    canonical_json_text(record.status_counts, max_bytes=16_384),
                    record.callback_rows_hash,
                    record.prior_chain_hash,
                    receipt_chain_hash(record),
                    record.first_normalized_event_id,
                    record.last_normalized_event_id,
                    record.created_at_us,
                ),
            )
            connection.execute(
                "UPDATE callback_inbox SET receipt_batch_id = ? WHERE run_id = ? "
                "AND source_sequence BETWEEN ? AND ?",
                (
                    batch_id,
                    run_id,
                    record.first_source_sequence,
                    record.last_source_sequence,
                ),
            )
            connection.commit()
            return record
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def create_pending_receipts(
        self, *, created_at_us: int, limit: int = 10_000
    ) -> tuple[CallbackReceiptRecord, ...]:
        """Receipt terminal prefixes across current and late prior-run callbacks."""

        if not 1 <= limit <= 10_000:
            raise ValueError("receipt limit must be between 1 and 10,000")
        with self._connect() as connection:
            run_ids = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT run_id FROM callback_inbox WHERE receipt_batch_id IS NULL "
                    "AND lifecycle IN ('acknowledged','failed') GROUP BY run_id "
                    "ORDER BY MIN(source_sequence) LIMIT ?",
                    (limit,),
                )
            )
        receipts: list[CallbackReceiptRecord] = []
        remaining = limit
        for run_id in run_ids:
            receipt = self.create_receipt(
                run_id,
                created_at_us=created_at_us,
                limit=remaining,
            )
            if receipt is None:
                continue
            receipts.append(receipt)
            remaining -= receipt.callback_count
            if remaining == 0:
                break
        return tuple(receipts)
