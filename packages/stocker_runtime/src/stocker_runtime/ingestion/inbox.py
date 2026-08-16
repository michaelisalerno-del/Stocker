"""Durable admission and projection boundary for IBKR market-data callbacks."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
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


class InboxAuthorityLost(InboxAdmissionError):
    """The caller no longer matches the sole active writer."""


class InboxFullError(InboxAdmissionError):
    """The hard nonterminal callback bound has closed admission."""


class CallbackIdentityCollision(InboxAdmissionError):
    """A deterministic callback identity names different durable content."""


class CallbackTimestampOrderingLoss(InboxAdmissionError):
    """Durable callback evidence is later than its attempted acknowledgement."""


class NormalizationError(ValueError):
    """A callback is durable but cannot safely become a typed market event."""


class _AdmissionConnectionOwner:
    """Close a cached SQLite handle deterministically when its callback thread exits."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection: sqlite3.Connection | None = connection

    def close(self) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            connection.close()

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()


CALLBACK_RECOVERABLE_GAP_REASONS = frozenset(
    {
        "IBKR_CONNECT_FAILED",
        "IBKR_DISCONNECT",
        "IBKR_SUBSCRIBE_FAILED",
        "RECONNECT_UNCERTAINTY",
        "STREAM_STALE",
    }
)
TRANSPORT_INCIDENT_CODES = ("IBKR_CONNECT_FAILED", "IBKR_SUBSCRIBE_FAILED")
_PROJECTION_TRANSACTION_LIMIT = 32
CALLBACK_GAP_RECOVERY_SQL = (
    "UPDATE gaps SET ended_at_us=?, resolved_at_us=? WHERE gap_id IN ("
    "SELECT gap.gap_id FROM gaps gap JOIN subscriptions prior "
    "ON prior.subscription_id=gap.subscription_id WHERE gap.run_id=? "
    "AND gap.resolved_at_us IS NULL AND prior.run_id=? "
    "AND prior.instrument_id=? AND prior.feed_kind=? AND prior.snapshot=? "
    "AND prior.recorder_generation=? AND prior.connection_generation<=? "
    "AND gap.reason IN (?, ?, ?, ?, ?) AND gap.started_at_us<=? AND ?<=?)"
)
CALLBACK_INCIDENT_RECOVERY_SQL = (
    "UPDATE incidents SET resolved_at_us=? WHERE incident_id IN (?, ?, ?, ?) "
    "AND opened_at_us<=? AND resolved_at_us IS NULL"
)


def transport_incident_id(
    run_id: str,
    code: str,
    *,
    instrument_id: str | None = None,
    feed_kind: str | None = None,
    snapshot: bool | None = None,
) -> str:
    """Identify one bounded connection or semantic-subscription incident."""

    subscription_scope = (instrument_id, feed_kind, snapshot)
    if any(value is not None for value in subscription_scope) and any(
        value is None for value in subscription_scope
    ):
        raise ValueError("transport incident subscription scope is incomplete")
    semantic_scope = (
        "connection"
        if instrument_id is None
        else (f"subscription|{instrument_id}|{feed_kind}|{'snapshot' if snapshot else 'stream'}")
    )
    return hashlib.sha256(f"{run_id}|transport|{semantic_scope}|{code}".encode()).hexdigest()


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


@dataclass(frozen=True)
class ProjectionBatchResult:
    processed: int
    causal_now_us: int


@dataclass(frozen=True)
class WriterAuthority:
    run_id: str
    recorder_generation: int
    owner_id: str


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
        self._admission_local = threading.local()
        self._writer_condition = threading.Condition()
        self._writer_active = False
        self._next_writer_ticket = 0
        self._serving_writer_ticket = 0
        with connect_v2(self.database_path):
            pass

    def _connect(self) -> sqlite3.Connection:
        """Open after constructor-time schema verification under the sole-writer lease."""

        return connect_v2(self.database_path, verify_schema=False)

    def _admission_connection(self) -> sqlite3.Connection:
        """Reuse one callback-thread connection while preserving per-callback commits."""

        owner = getattr(self._admission_local, "owner", None)
        if owner is None:
            owner = _AdmissionConnectionOwner(self._connect())
            self._admission_local.owner = owner
        connection = cast(_AdmissionConnectionOwner, owner).connection
        if connection is None:
            raise InboxAdmissionError("callback admission connection is closed")
        return connection

    def _discard_admission_connection(self, connection: sqlite3.Connection) -> None:
        owner = getattr(self._admission_local, "owner", None)
        if isinstance(owner, _AdmissionConnectionOwner) and owner.connection is connection:
            owner.close()
            del self._admission_local.owner

    def _acquire_admission_writer(self) -> None:
        """Enter the writer queue without starving projection work."""

        self._acquire_writer_ticket()

    def _acquire_projection_writer(self) -> None:
        """Enter one bounded projection transaction in FIFO writer order."""

        self._acquire_writer_ticket()

    def _acquire_writer_ticket(self) -> None:
        """Serialize admissions and projection chunks in bounded FIFO order."""

        with self._writer_condition:
            ticket = self._next_writer_ticket
            self._next_writer_ticket += 1
            while self._writer_active or ticket != self._serving_writer_ticket:
                self._writer_condition.wait()
            self._writer_active = True

    def _release_writer(self) -> None:
        with self._writer_condition:
            if not self._writer_active:
                raise RuntimeError("callback writer arbitration is unbalanced")
            self._writer_active = False
            self._serving_writer_ticket += 1
            self._writer_condition.notify_all()

    @contextmanager
    def _connection_scope(
        self, connection: sqlite3.Connection | None
    ) -> Iterator[sqlite3.Connection]:
        if connection is not None:
            if connection.in_transaction:
                yield connection
            else:
                with connection:
                    yield connection
            return
        owned = self._connect()
        try:
            with owned:
                yield owned
        finally:
            owned.close()

    @staticmethod
    def verify_writer(connection: sqlite3.Connection, authority: WriterAuthority) -> None:
        row = connection.execute(
            "SELECT state.lifecycle, run.status, generation.owner_id, generation.ended_at_us "
            "FROM runtime_state state JOIN runs run ON run.run_id=state.run_id "
            "JOIN recorder_generations generation ON generation.run_id=state.run_id "
            "AND generation.generation=state.recorder_generation "
            "WHERE state.run_id=? AND state.recorder_generation=?",
            (authority.run_id, authority.recorder_generation),
        ).fetchone()
        if (
            row is None
            or str(row["status"]) != "running"
            or str(row["owner_id"]) != authority.owner_id
            or row["ended_at_us"] is not None
            or str(row["lifecycle"]) not in {"recovering", "connecting", "running", "degraded"}
        ):
            raise InboxAdmissionError("authoritative writer lease is no longer owned")

    def nonterminal_count(self) -> int:
        """Return the bounded recovery backlog for lifecycle orchestration."""

        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending','leased')"
                ).fetchone()[0]
            )

    @staticmethod
    def _refresh_nonterminal_count(connection: sqlite3.Connection, run_ids: set[str]) -> None:
        nonterminal = int(
            connection.execute(
                "SELECT count(*) FROM callback_inbox WHERE lifecycle IN ('pending', 'leased')"
            ).fetchone()[0]
        )
        connection.executemany(
            "UPDATE runtime_state SET inbox_nonterminal_count=? WHERE run_id=?",
            ((nonterminal, run_id) for run_id in sorted(run_ids)),
        )

    def admit(
        self,
        fence: CallbackFence,
        callback: MarketDataCallback,
        *,
        authority: WriterAuthority | None = None,
    ) -> AdmissionResult:
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
        self._acquire_admission_writer()
        try:
            connection = self._admission_connection()
        except (OSError, sqlite3.Error) as error:
            self._release_writer()
            raise InboxAdmissionError(f"callback durable admission failed: {error}") from error
        except BaseException:
            self._release_writer()
            raise
        try:
            connection.execute("BEGIN IMMEDIATE")
            authoritative = self._authoritative_admission(connection)
            if authority is not None and (
                authority.run_id != str(authoritative["run_id"])
                or authority.recorder_generation != int(authoritative["recorder_generation"])
                or authority.owner_id != str(authoritative["owner_id"])
            ):
                raise InboxAuthorityLost("authoritative writer lease is no longer owned")
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
                    or (
                        existing["payload_json"] is not None
                        and existing["payload_json"] != payload_json
                    )
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
                self._record_fatal(
                    connection,
                    str(authoritative["run_id"]),
                    callback.received_at_us,
                    "INBOX_FULL",
                    authority=WriterAuthority(
                        str(authoritative["run_id"]),
                        int(authoritative["recorder_generation"]),
                        str(authoritative["owner_id"]),
                    ),
                )
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
                    str(authoritative["run_id"]),
                    callback.received_at_us,
                    "CALLBACK_ORDERING_LOSS",
                    authority=WriterAuthority(
                        str(authoritative["run_id"]),
                        int(authoritative["recorder_generation"]),
                        str(authoritative["owner_id"]),
                    ),
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
                    str(authoritative["run_id"]),
                    int(authoritative["recorder_generation"]),
                ),
            )
            connection.commit()
            return AdmissionResult(sequence, event_uid, True)
        except InboxFullError:
            raise
        except sqlite3.Error as error:
            try:
                if connection.in_transaction:
                    connection.rollback()
            finally:
                self._discard_admission_connection(connection)
            raise InboxAdmissionError(f"callback durable admission failed: {error}") from error
        except BaseException:
            try:
                if connection.in_transaction:
                    connection.rollback()
            finally:
                self._discard_admission_connection(connection)
            raise
        finally:
            self._release_writer()

    @staticmethod
    def _authoritative_admission(connection: sqlite3.Connection) -> sqlite3.Row:
        rows = tuple(
            connection.execute(
                "SELECT state.run_id, state.recorder_generation, generation.owner_id "
                "FROM runtime_state state JOIN runs run ON run.run_id=state.run_id "
                "JOIN recorder_generations generation ON generation.run_id=state.run_id "
                "AND generation.generation=state.recorder_generation "
                "WHERE run.status='running' AND generation.ended_at_us IS NULL "
                "AND state.lifecycle IN ('recovering','connecting','running','degraded')"
            )
        )
        if len(rows) != 1:
            raise InboxAuthorityLost("authoritative recorder admission state is absent or split")
        return cast(sqlite3.Row, rows[0])

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
                "AND lifecycle IN ('connecting', 'active')",
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
        subscription = connection.execute(
            "SELECT continuity_required FROM subscriptions WHERE subscription_id=? AND run_id=?",
            (fence.subscription_id, fence.run_id),
        ).fetchone()
        continuity_required = 1 if subscription is None else int(subscription[0])
        gap_id = hashlib.sha256(
            f"{fence.run_id}|{fence.subscription_id}|{code}|{sequence}".encode()
        ).hexdigest()
        connection.execute(
            "INSERT INTO gaps(gap_id, run_id, subscription_id, started_at_us, reason, "
            "data_loss_possible, continuity_required) VALUES (?, ?, ?, ?, ?, 1, ?)",
            (
                gap_id,
                fence.run_id,
                fence.subscription_id,
                opened_at_us,
                code,
                continuity_required,
            ),
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
        connection: sqlite3.Connection,
        run_id: str,
        opened_at_us: int,
        code: str,
        *,
        authority: WriterAuthority,
    ) -> None:
        current = connection.execute(
            "SELECT state.recorder_generation, state.lifecycle, state.reason, generation.owner_id "
            "FROM runtime_state state JOIN recorder_generations generation "
            "ON generation.run_id=state.run_id "
            "AND generation.generation=state.recorder_generation WHERE state.run_id=?",
            (run_id,),
        ).fetchone()
        if current is None or (
            authority.run_id != run_id
            or authority.recorder_generation != int(current["recorder_generation"])
            or authority.owner_id != str(current["owner_id"])
        ):
            raise InboxAdmissionError("fatal transition no longer owns writer authority")
        generation = int(current["recorder_generation"])
        terminal_code = (
            str(current["reason"])
            if str(current["lifecycle"]) == "fatal" and current["reason"] is not None
            else code
        )
        incident_id = hashlib.sha256(f"{run_id}|{generation}|{terminal_code}".encode()).hexdigest()
        connection.execute("UPDATE runs SET status = 'fatal' WHERE run_id = ?", (run_id,))
        connection.execute(
            "UPDATE subscriptions SET lifecycle='disconnected' WHERE run_id=? "
            "AND recorder_generation=? AND lifecycle!='closed'",
            (run_id, generation),
        )
        connection.execute(
            "UPDATE recorder_generations SET ended_at_us=COALESCE(ended_at_us, ?), "
            "clean_stop=0, termination_code=COALESCE(termination_code, ?) "
            "WHERE run_id=? AND generation=?",
            (opened_at_us, terminal_code, run_id, generation),
        )
        connection.execute(
            "UPDATE runtime_state SET lifecycle = 'fatal', reason = ?, "
            "connection_state = 'disconnected' WHERE run_id = ? AND recorder_generation=?",
            (terminal_code, run_id, generation),
        )
        connection.execute(
            "INSERT OR IGNORE INTO incidents(incident_id, run_id, scope, severity, code, "
            "opened_at_us, details_json) VALUES (?, ?, 'recorder', 'fatal', ?, ?, '{}')",
            (incident_id, run_id, terminal_code, opened_at_us),
        )

    def reclaim_expired_leases(self, *, now_us: int, authority: WriterAuthority) -> int:
        """Return expired leases to pending without changing source order."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self.verify_writer(connection, authority)
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
        authority: WriterAuthority,
    ) -> tuple[LeasedCallback, ...]:
        """Lease a bounded pending prefix; never jump over an earlier active lease."""

        if not owner or lease_us <= 0 or not 1 <= limit <= 10_000:
            raise ValueError("lease owner, duration, and bounded limit are required")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self.verify_writer(connection, authority)
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
                cursor = connection.executemany(
                    "UPDATE callback_inbox SET lifecycle = 'leased', lease_owner = ?, "
                    "lease_expires_at_us = ?, attempts = attempts + 1 "
                    "WHERE source_sequence = ? AND lifecycle = 'pending'",
                    ((owner, now_us + lease_us, int(row["source_sequence"])) for row in selected),
                )
                if cursor.rowcount != len(selected):
                    raise InboxAdmissionError("callback lease acquisition was not atomic")
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

    def project_batch(
        self,
        leased_callbacks: tuple[LeasedCallback, ...],
        *,
        now_us: int,
        authority: WriterAuthority,
    ) -> ProjectionBatchResult:
        """Durably project, then terminalize a leased prefix in bounded transactions."""

        causal_now_us = now_us
        processed = 0
        connection = self._connect()
        try:
            for offset in range(0, len(leased_callbacks), _PROJECTION_TRANSACTION_LIMIT):
                chunk = leased_callbacks[offset : offset + _PROJECTION_TRANSACTION_LIMIT]
                terminal_actions: list[tuple[LeasedCallback, str | None, int]] = []
                self._acquire_projection_writer()
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    self.verify_writer(connection, authority)
                    for leased in chunk:
                        causal_now_us = max(causal_now_us, leased.received_at_us)
                        connection.execute("SAVEPOINT callback_projection")
                        try:
                            result = self.project(
                                leased,
                                authority=authority,
                                connection=connection,
                            )
                        except NormalizationError:
                            connection.execute("ROLLBACK TO callback_projection")
                            connection.execute("RELEASE callback_projection")
                            terminal_actions.append((leased, None, causal_now_us))
                            continue
                        connection.execute("RELEASE callback_projection")
                        terminal_actions.append((leased, result.event_id, causal_now_us))
                    connection.commit()
                except BaseException:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
                finally:
                    self._release_writer()

                self._acquire_projection_writer()
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    self.verify_writer(connection, authority)
                    affected_run_ids: set[str] = set()
                    successful = 0
                    for leased, event_id, terminal_at_us in terminal_actions:
                        if event_id is None:
                            self.fail(
                                leased,
                                "MALFORMED_CALLBACK",
                                failed_at_us=terminal_at_us,
                                authority=authority,
                                connection=connection,
                                _refresh_nonterminal_count=False,
                            )
                        else:
                            self.acknowledge(
                                leased,
                                event_id,
                                acknowledged_at_us=terminal_at_us,
                                authority=authority,
                                connection=connection,
                                _refresh_nonterminal_count=False,
                            )
                            successful += 1
                        affected_run_ids.add(leased.run_id)
                    self._refresh_nonterminal_count(connection, affected_run_ids)
                    connection.commit()
                    processed += successful
                except BaseException:
                    if connection.in_transaction:
                        connection.rollback()
                    raise
                finally:
                    self._release_writer()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        return ProjectionBatchResult(processed=processed, causal_now_us=causal_now_us)

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

    def project(
        self,
        leased: LeasedCallback,
        *,
        authority: WriterAuthority,
        connection: sqlite3.Connection | None = None,
    ) -> ProjectionResult:
        """Idempotently write one typed event and its latest projection in one transaction."""
        owns_connection = connection is None
        if connection is None:
            connection = self._connect()
        started_transaction = not connection.in_transaction
        try:
            if started_transaction:
                connection.execute("BEGIN IMMEDIATE")
            self.verify_writer(connection, authority)
            inbox_row = connection.execute(
                "SELECT * FROM callback_inbox WHERE source_sequence = ?",
                (leased.source_sequence,),
            ).fetchone()
            if (
                inbox_row is None
                or str(inbox_row["lifecycle"]) != "leased"
                or str(inbox_row["lease_owner"]) != leased.lease_owner
                or str(inbox_row["event_uid"]) != leased.event_uid
                or str(inbox_row["run_id"]) != leased.run_id
                or int(inbox_row["recorder_generation"]) != leased.recorder_generation
                or int(inbox_row["connection_generation"]) != leased.connection_generation
                or inbox_row["request_id"] != leased.request_id
                or str(inbox_row["callback_kind"]) != leased.callback_kind
                or int(inbox_row["received_at_us"]) != leased.received_at_us
                or inbox_row["provider_at_us"] != leased.provider_at_us
                or str(inbox_row["payload_sha256"]) != leased.payload_sha256
            ):
                raise InboxAdmissionError("callback lease token no longer matches durable evidence")
            payload_json = inbox_row["payload_json"]
            if payload_json is None:
                raise InboxAdmissionError("leased callback payload is absent")
            payload_text = str(payload_json)
            durable_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
            if durable_hash != str(inbox_row["payload_sha256"]):
                raise CallbackIdentityCollision("durable callback payload hash mismatch")
            try:
                parsed = json.loads(payload_text)
            except json.JSONDecodeError as error:
                raise CallbackIdentityCollision("durable callback payload is invalid") from error
            if not isinstance(parsed, Mapping):
                raise NormalizationError("callback payload must be an object")
            payload = cast(Mapping[str, object], parsed)
            canonical = canonical_json_text(
                cast(JsonValue, parsed), max_bytes=MAX_CALLBACK_PAYLOAD_BYTES
            )
            if canonical != payload_text:
                raise CallbackIdentityCollision("durable callback payload is not canonical")
            event_at = payload.get("event_at_us")
            if isinstance(event_at, bool) or not isinstance(event_at, int) or event_at < 0:
                raise NormalizationError("event_at_us must be a nonnegative integer")
            callback_kind = str(inbox_row["callback_kind"])
            if callback_kind not in {
                "quote",
                "trade",
                "bar",
                "option_computation",
                "option_snapshot_end",
            }:
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
                    "bid_size",
                    "ask_size",
                    "last",
                    "size",
                )
            }
            for field in (
                "call_open_interest",
                "put_open_interest",
                "call_option_volume",
                "put_option_volume",
                "implied_volatility",
                "delta",
                "option_price",
                "present_value_dividend",
                "gamma",
                "vega",
                "theta",
                "underlying_price",
            ):
                self._number(payload, field)
            run_id = str(inbox_row["run_id"])
            recorder_generation = int(inbox_row["recorder_generation"])
            connection_generation = int(inbox_row["connection_generation"])
            request_id = inbox_row["request_id"]
            received_at_us = int(inbox_row["received_at_us"])
            subscription = connection.execute(
                "SELECT instrument_id, feed_kind FROM subscriptions WHERE run_id = ? "
                "AND recorder_generation = ? AND connection_generation = ? AND request_id = ?",
                (run_id, recorder_generation, connection_generation, request_id),
            ).fetchone()
            if subscription is None:
                raise NormalizationError("callback subscription provenance is absent")
            event_id = str(inbox_row["event_uid"])
            content = (
                run_id,
                int(inbox_row["source_sequence"]),
                str(subscription["instrument_id"]),
                str(subscription["feed_kind"]),
                callback_kind,
                event_at,
                received_at_us,
                connection_generation,
                values["open"],
                values["high"],
                values["low"],
                values["close"],
                values["volume"],
                values["bid"],
                values["ask"],
                values["bid_size"],
                values["ask_size"],
                values["last"],
                values["size"],
                payload_text,
                durable_hash,
            )
            existing = connection.execute(
                "SELECT run_id, source_sequence, instrument_id, feed_kind, event_kind, "
                "event_at_us, received_at_us, connection_generation, open_value, high_value, "
                "low_value, close_value, volume_value, bid_value, ask_value, bid_size_value, "
                "ask_size_value, last_value, size_value, payload_json, payload_sha256 "
                "FROM market_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            inserted = existing is None
            if existing is None:
                connection.execute(
                    "INSERT INTO market_events(event_id, run_id, source_sequence, instrument_id, "
                    "feed_kind, event_kind, event_at_us, received_at_us, connection_generation, "
                    "open_value, high_value, low_value, close_value, volume_value, bid_value, "
                    "ask_value, bid_size_value, ask_size_value, last_value, size_value, "
                    "payload_json, payload_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            if latest is None or int(latest[0]) <= int(inbox_row["source_sequence"]):
                connection.execute(
                    "INSERT INTO market_latest(run_id, instrument_id, feed_kind, event_id, "
                    "event_at_us, received_at_us, event_kind, quality_bits, "
                    "bid_value, bid_source_event_id, ask_value, ask_source_event_id, "
                    "bid_size_value, bid_size_source_event_id, "
                    "ask_size_value, ask_size_source_event_id, "
                    "last_value, last_source_event_id, size_value, size_source_event_id, "
                    "close_value, close_source_event_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(instrument_id, feed_kind) DO UPDATE SET run_id=excluded.run_id, "
                    "event_id=excluded.event_id, event_at_us=excluded.event_at_us, "
                    "received_at_us=excluded.received_at_us, event_kind=excluded.event_kind, "
                    "quality_bits=excluded.quality_bits, "
                    "bid_value=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.bid_value, market_latest.bid_value) "
                    "ELSE excluded.bid_value END, "
                    "bid_source_event_id=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.bid_source_event_id, market_latest.bid_source_event_id) "
                    "ELSE excluded.bid_source_event_id END, "
                    "ask_value=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.ask_value, market_latest.ask_value) "
                    "ELSE excluded.ask_value END, "
                    "ask_source_event_id=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.ask_source_event_id, market_latest.ask_source_event_id) "
                    "ELSE excluded.ask_source_event_id END, "
                    "bid_size_value=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.bid_size_value, market_latest.bid_size_value) "
                    "ELSE excluded.bid_size_value END, "
                    "bid_size_source_event_id=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.bid_size_source_event_id, "
                    "market_latest.bid_size_source_event_id) "
                    "ELSE excluded.bid_size_source_event_id END, "
                    "ask_size_value=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.ask_size_value, market_latest.ask_size_value) "
                    "ELSE excluded.ask_size_value END, "
                    "ask_size_source_event_id=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.ask_size_source_event_id, "
                    "market_latest.ask_size_source_event_id) "
                    "ELSE excluded.ask_size_source_event_id END, "
                    "last_value=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.last_value, market_latest.last_value) "
                    "ELSE excluded.last_value END, "
                    "last_source_event_id=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.last_source_event_id, market_latest.last_source_event_id) "
                    "ELSE excluded.last_source_event_id END, "
                    "size_value=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.size_value, market_latest.size_value) "
                    "ELSE excluded.size_value END, "
                    "size_source_event_id=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.size_source_event_id, market_latest.size_source_event_id) "
                    "ELSE excluded.size_source_event_id END, "
                    "close_value=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.close_value, market_latest.close_value) "
                    "ELSE excluded.close_value END, "
                    "close_source_event_id=CASE WHEN market_latest.run_id=excluded.run_id THEN "
                    "COALESCE(excluded.close_source_event_id, market_latest.close_source_event_id) "
                    "ELSE excluded.close_source_event_id END",
                    (
                        run_id,
                        str(subscription["instrument_id"]),
                        str(subscription["feed_kind"]),
                        event_id,
                        event_at,
                        received_at_us,
                        callback_kind,
                        values["bid"],
                        event_id if values["bid"] is not None else None,
                        values["ask"],
                        event_id if values["ask"] is not None else None,
                        values["bid_size"],
                        event_id if values["bid_size"] is not None else None,
                        values["ask_size"],
                        event_id if values["ask_size"] is not None else None,
                        values["last"],
                        event_id if values["last"] is not None else None,
                        values["size"],
                        event_id if values["size"] is not None else None,
                        values["close"],
                        event_id if values["close"] is not None else None,
                    ),
                )
            connection.execute(
                "UPDATE subscriptions SET latest_event_id = ? WHERE run_id = ? "
                "AND connection_generation = ? AND request_id = ?",
                (event_id, run_id, connection_generation, request_id),
            )
            connection.execute(
                "UPDATE runtime_state SET projection_heartbeat_at_us = ? WHERE run_id = ?",
                (received_at_us, run_id),
            )
            if started_transaction:
                connection.commit()
            return ProjectionResult(event_id, inserted)
        except Exception:
            if started_transaction and connection.in_transaction:
                connection.rollback()
            raise
        finally:
            if owns_connection:
                connection.close()

    def acknowledge(
        self,
        leased: LeasedCallback,
        event_id: str,
        *,
        acknowledged_at_us: int,
        authority: WriterAuthority,
        connection: sqlite3.Connection | None = None,
        _refresh_nonterminal_count: bool = True,
    ) -> None:
        """Mark terminal only after the exact durable projection is present."""

        with self._connection_scope(connection) as connection:
            if not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            self.verify_writer(connection, authority)
            evidence = connection.execute(
                "SELECT callback.received_at_us AS callback_received_at_us, "
                "event.received_at_us AS event_received_at_us, "
                "event.instrument_id, event.feed_kind FROM callback_inbox callback "
                "JOIN market_events event ON event.event_id=? "
                "AND event.source_sequence=callback.source_sequence "
                "AND event.run_id=callback.run_id "
                "AND event.connection_generation=callback.connection_generation "
                "WHERE callback.source_sequence=? AND callback.lifecycle='leased' "
                "AND callback.lease_owner=? AND callback.run_id=? "
                "AND callback.recorder_generation=? AND callback.connection_generation=? "
                "AND callback.request_id IS ?",
                (
                    event_id,
                    leased.source_sequence,
                    leased.lease_owner,
                    leased.run_id,
                    leased.recorder_generation,
                    leased.connection_generation,
                    leased.request_id,
                ),
            ).fetchone()
            if evidence is None:
                raise InboxAdmissionError(
                    "callback acknowledgement lacks exact durable event evidence"
                )
            callback_received_at_us = int(evidence["callback_received_at_us"])
            event_received_at_us = int(evidence["event_received_at_us"])
            if (
                callback_received_at_us != leased.received_at_us
                or callback_received_at_us != event_received_at_us
                or event_received_at_us > acknowledged_at_us
            ):
                raise CallbackTimestampOrderingLoss(
                    "callback acknowledgement timestamp ordering moved backwards"
                )
            cursor = connection.execute(
                "UPDATE callback_inbox SET lifecycle = 'acknowledged', lease_owner = NULL, "
                "lease_expires_at_us = NULL, normalized_event_id = ?, acknowledged_at_us = ? "
                "WHERE source_sequence = ? AND lifecycle = 'leased' AND lease_owner = ?",
                (event_id, acknowledged_at_us, leased.source_sequence, leased.lease_owner),
            )
            if cursor.rowcount != 1:
                raise InboxAdmissionError("callback acknowledgement lost its lease")
            if _refresh_nonterminal_count:
                self._refresh_nonterminal_count(connection, {leased.run_id})
            subscription = connection.execute(
                "SELECT subscription_id, instrument_id, feed_kind, snapshot FROM subscriptions "
                "WHERE run_id=? AND recorder_generation=? "
                "AND connection_generation=? AND request_id IS ? AND lifecycle='active'",
                (
                    leased.run_id,
                    leased.recorder_generation,
                    leased.connection_generation,
                    leased.request_id,
                ),
            ).fetchone()
            if (
                subscription is not None
                and str(evidence["instrument_id"]) == str(subscription["instrument_id"])
                and str(evidence["feed_kind"]) == str(subscription["feed_kind"])
            ):
                # Gap starts and callback receipt times share the recorder's local
                # clock. Provider/event times may lag and cannot prove recovery.
                evidence_at_us = event_received_at_us
                connection.execute(
                    CALLBACK_GAP_RECOVERY_SQL,
                    (
                        evidence_at_us,
                        acknowledged_at_us,
                        leased.run_id,
                        leased.run_id,
                        str(subscription["instrument_id"]),
                        str(subscription["feed_kind"]),
                        int(subscription["snapshot"]),
                        leased.recorder_generation,
                        leased.connection_generation,
                        *sorted(CALLBACK_RECOVERABLE_GAP_REASONS),
                        evidence_at_us,
                        evidence_at_us,
                        acknowledged_at_us,
                    ),
                )
                scoped_incident_ids = tuple(
                    transport_incident_id(
                        leased.run_id,
                        code,
                        instrument_id=str(subscription["instrument_id"]),
                        feed_kind=str(subscription["feed_kind"]),
                        snapshot=bool(subscription["snapshot"]),
                    )
                    for code in TRANSPORT_INCIDENT_CODES
                )
                global_incident_ids = tuple(
                    transport_incident_id(leased.run_id, code) for code in TRANSPORT_INCIDENT_CODES
                )
                connection.execute(
                    CALLBACK_INCIDENT_RECOVERY_SQL,
                    (
                        acknowledged_at_us,
                        *global_incident_ids,
                        *scoped_incident_ids,
                        evidence_at_us,
                    ),
                )

    def fail(
        self,
        leased: LeasedCallback,
        code: str,
        *,
        failed_at_us: int,
        authority: WriterAuthority,
        connection: sqlite3.Connection | None = None,
        _refresh_nonterminal_count: bool = True,
    ) -> None:
        """Quarantine one poison callback while leaving later callbacks serviceable."""

        if not code:
            raise ValueError("failure code is required")
        with self._connection_scope(connection) as connection:
            if not connection.in_transaction:
                connection.execute("BEGIN IMMEDIATE")
            self.verify_writer(connection, authority)
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
            if _refresh_nonterminal_count:
                self._refresh_nonterminal_count(connection, {leased.run_id})

    def create_receipt(
        self,
        run_id: str,
        *,
        created_at_us: int,
        limit: int = 256,
        authority: WriterAuthority,
    ) -> CallbackReceiptRecord | None:
        """Receipt one contiguous terminal prefix using the Phase 2 evidence contract."""

        if not 1 <= limit <= 256:
            raise ValueError("receipt limit must be between 1 and 256")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self.verify_writer(connection, authority)
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
            if int(last["received_at_us"]) < int(first["received_at_us"]):
                raise InboxAdmissionError("receipt callback receive order is reversed")
            receipt_created_at_us = max(created_at_us, int(last["received_at_us"]))
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
                created_at_us=receipt_created_at_us,
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
        self,
        *,
        created_at_us: int,
        limit: int = 10_000,
        authority: WriterAuthority,
    ) -> tuple[CallbackReceiptRecord, ...]:
        """Receipt terminal prefixes across current and late prior-run callbacks."""

        if not 1 <= limit <= 10_000:
            raise ValueError("receipt limit must be between 1 and 10,000")
        with self._connect() as connection:
            self.verify_writer(connection, authority)
            run_ids = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT run_id FROM callback_inbox "
                    "INDEXED BY callback_inbox_unreceipted_terminal_idx "
                    "WHERE receipt_batch_id IS NULL "
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
                limit=min(remaining, 256),
                authority=authority,
            )
            if receipt is None:
                continue
            receipts.append(receipt)
            remaining -= receipt.callback_count
            if remaining == 0:
                break
        return tuple(receipts)
