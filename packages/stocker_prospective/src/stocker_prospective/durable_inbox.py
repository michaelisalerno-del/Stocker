"""Durable callback admission, leasing, acknowledgement, and fatal incidents."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.sqlite_coordination import CoordinatedSQLiteConnection

_TRANSIENT_WRITER_RETRY_DELAYS_SECONDS = (0.005, 0.01, 0.025, 0.05)


class CallbackInboxError(RuntimeError):
    """Base class for stable durable-inbox failures."""


class CallbackInboxOverflow(CallbackInboxError):
    """The configured unacknowledged callback bound has been reached."""


class CallbackIdentityCollision(CallbackInboxError):
    """A supposedly stable provider identity was reused for different evidence."""


class CallbackLeaseLost(CallbackInboxError):
    """A lease owner or generation attempted a stale state transition."""


class CallbackClassification(StrEnum):
    ACCEPTED_ACTIVE = "accepted_active_callback"
    EXPECTED_LATE = "expected_late_callback_after_cancellation"
    PREVIOUS_CONNECTION = "callback_from_previous_connection_generation"
    DUPLICATE = "duplicate_callback"
    UNKNOWN = "unknown_callback"
    AFTER_DATA_LOSS_LATCH = "callback_after_data_loss_latch"
    CONTROL = "control_callback"


class InboxStatus(StrEnum):
    PROVIDER_PENDING = "provider_pending"
    PENDING = "pending"
    LEASED = "leased"
    ACKNOWLEDGED = "acknowledged"
    QUARANTINED = "quarantined"
    DIAGNOSTIC = "diagnostic"


class CallbackInboxEvent(BaseModel):
    """One original callback envelope leased without destructive removal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inbox_event_id: str
    source_sequence: int = Field(gt=0)
    callback_kind: str
    request_id: int
    received_utc: datetime
    received_monotonic_ns: int | None
    provider_timestamp_utc: datetime | None
    original_payload: dict[str, Any]
    admission_run_id: str | None
    admission_recorder_generation: int | None
    connection_generation: int = Field(ge=0)
    subscription_owner: str | None
    symbol: str | None
    stream_owner: dict[str, Any] | None
    callback_classification: CallbackClassification
    provider_envelope_event_id: str | None
    lease_owner: str | None
    lease_generation: int = Field(ge=0)
    lease_batch_id: str | None
    lease_timestamp_utc: datetime | None
    attempt_count: int = Field(ge=0)
    status: InboxStatus
    acknowledgement_timestamp_utc: datetime | None
    failure_classification: str | None
    associated_raw_partition_hashes: tuple[str, ...]

    @field_validator(
        "received_utc",
        "provider_timestamp_utc",
        "lease_timestamp_utc",
        "acknowledgement_timestamp_utc",
    )
    @classmethod
    def _timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("callback inbox timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def normalizer_payload(self) -> dict[str, Any]:
        """Recreate the exact normalizer envelope without losing null fields."""

        return {
            **self.original_payload,
            "kind": self.callback_kind,
            "request_id": self.request_id,
            "received_timestamp_utc": self.received_utc.isoformat(),
            "received_monotonic_ns": self.received_monotonic_ns,
            "source_sequence": self.source_sequence,
            "persisted_stream_owner": self.stream_owner,
            "inbox_event_id": self.inbox_event_id,
            "callback_classification": self.callback_classification.value,
            "connection_generation": self.connection_generation,
            "subscription_owner": self.subscription_owner,
            "subscription_symbol": self.symbol,
            "admission_run_id": self.admission_run_id,
            "admission_recorder_generation": self.admission_recorder_generation,
        }


class InboxAdmissionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: CallbackInboxEvent
    duplicate: bool


class RawMaterialization(BaseModel):
    """The immutable raw identity already assigned to one callback batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    partition_hashes: tuple[str, ...]
    raw_event_ids: tuple[str, ...]


class InboxAccounting(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted: int = Field(ge=0)
    pending: int = Field(ge=0)
    leased: int = Field(ge=0)
    acknowledged: int = Field(ge=0)
    quarantined: int = Field(ge=0)
    diagnostic: int = Field(ge=0)
    highest_source_sequence: int = Field(ge=0)
    highest_acknowledged_sequence: int = Field(ge=0)
    oldest_unacknowledged_at_utc: datetime | None


class RequestTombstone(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: int
    connection_generation: int = Field(ge=0)
    subscription_owner: str | None
    symbol: str | None
    cancellation_reason: str
    cancelled_at_utc: datetime
    expires_at_utc: datetime


def _utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return _utc(value, label="callback payload timestamp").isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_value(item) for item in value), key=str)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        # JSON has no lossless representation for non-finite provider values.
        # Preserve the original category so admission succeeds and the
        # normalizer can quarantine it instead of losing the callback.
        return {"__non_finite_float__": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"callback payload value is not serialisable: {type(value).__name__}")


def _encoded(value: object) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parsed_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        candidate = value
    elif isinstance(value, str):
        try:
            candidate = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if candidate.tzinfo is None or candidate.utcoffset() is None:
        return None
    return candidate.astimezone(UTC)


def _event_id(
    *,
    callback_kind: str,
    request_id: int,
    received_utc: datetime,
    received_monotonic_ns: int | None,
    admission_run_id: str | None,
    connection_generation: int,
    payload_json: str,
) -> str:
    identity = "|".join(
        (
            callback_kind,
            str(request_id),
            received_utc.isoformat(),
            str(received_monotonic_ns),
            str(admission_run_id),
            str(connection_generation),
            payload_json,
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()


class DurableCallbackInbox:
    """SQLite WAL spool whose rows survive callback-loop and recorder crashes."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        max_unacknowledged: int = 65_536,
        max_tombstones: int = 4_096,
        busy_timeout_ms: int = 5_000,
        run_id: str | None = None,
        recorder_generation: int | None = None,
        owner_id: str | None = None,
    ) -> None:
        if max_unacknowledged <= 0:
            raise ValueError("max_unacknowledged must be positive")
        if max_tombstones <= 0:
            raise ValueError("max_tombstones must be positive")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be nonnegative")
        self.database_path = Path(database_path)
        self.max_unacknowledged = max_unacknowledged
        self.max_tombstones = max_tombstones
        self.busy_timeout_ms = busy_timeout_ms
        self.run_id = run_id
        self.recorder_generation = recorder_generation
        self.owner_id = owner_id
        self._connection_local = threading.local()
        self._active_count_run_id: str | None = None
        self._active_count: int | None = None
        self._oldest_active_received_at: datetime | None = None

    def configure_recorder(
        self,
        *,
        run_id: str,
        recorder_generation: int,
        owner_id: str,
    ) -> None:
        if not run_id or recorder_generation <= 0 or not owner_id:
            raise ValueError("durable inbox recorder identity is invalid")
        self.run_id = run_id
        self.recorder_generation = recorder_generation
        self.owner_id = owner_id
        self._active_count_run_id = None
        self._active_count = None
        self._oldest_active_received_at = None

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = getattr(
            self._connection_local,
            "connection",
            None,
        )
        if connection is not None:
            return connection

        connection = sqlite3.connect(
            self.database_path,
            timeout=self.busy_timeout_ms / 1_000,
            factory=CoordinatedSQLiteConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        self._connection_local.connection = connection
        return connection

    @staticmethod
    def _is_transient_writer_contention(error: sqlite3.OperationalError) -> bool:
        error_code = getattr(error, "sqlite_errorcode", None)
        if isinstance(error_code, int) and error_code & 0xFF in {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
        }:
            return True
        message = str(error).lower()
        return "database is locked" in message or "database is busy" in message

    def _begin_immediate(self, connection: sqlite3.Connection) -> None:
        """Wait through bounded transient writer contention without losing ingress."""

        for delay_seconds in (*_TRANSIENT_WRITER_RETRY_DELAYS_SECONDS, None):
            try:
                connection.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as error:
                if delay_seconds is None or not self._is_transient_writer_contention(error):
                    raise
                time.sleep(delay_seconds)

    @contextmanager
    def callback_batch(self) -> Iterator[None]:
        """Commit several callback state transitions in one durable transaction."""

        connection = self._connect()
        if connection.in_transaction:
            raise RuntimeError("callback batch transaction is already active")
        self._begin_immediate(connection)
        try:
            yield
        except Exception:
            connection.rollback()
            self._active_count_run_id = None
            self._active_count = None
            self._oldest_active_received_at = None
            raise
        else:
            connection.commit()

    def _ensure_active_accounting(self, connection: sqlite3.Connection) -> None:
        if self._active_count is not None and self._active_count_run_id == self.run_id:
            return
        row = connection.execute(
            """
            SELECT COUNT(*) AS backlog, MIN(received_utc) AS oldest
            FROM callback_inbox_v1
            WHERE admission_run_id IS ?
              AND status IN (
                  'provider_pending', 'pending', 'leased', 'quarantined'
              )
            """,
            (self.run_id,),
        ).fetchone()
        assert row is not None
        self._active_count_run_id = self.run_id
        self._active_count = int(row["backlog"])
        self._oldest_active_received_at = (
            None if row["oldest"] is None else datetime.fromisoformat(str(row["oldest"]))
        )

    def _increment_active_accounting(self, *, received_at: datetime) -> None:
        assert self._active_count is not None
        self._active_count += 1
        if self._oldest_active_received_at is None or received_at < self._oldest_active_received_at:
            self._oldest_active_received_at = received_at

    def _decrement_active_accounting(self) -> None:
        assert self._active_count is not None and self._active_count > 0
        self._active_count -= 1
        if self._active_count == 0:
            self._oldest_active_received_at = None

    @staticmethod
    def _event(row: sqlite3.Row) -> CallbackInboxEvent:
        return CallbackInboxEvent(
            inbox_event_id=str(row["inbox_event_id"]),
            source_sequence=int(row["source_sequence"]),
            callback_kind=str(row["callback_kind"]),
            request_id=int(row["request_id"]),
            received_utc=datetime.fromisoformat(str(row["received_utc"])),
            received_monotonic_ns=(
                None if row["received_monotonic_ns"] is None else int(row["received_monotonic_ns"])
            ),
            provider_timestamp_utc=(
                None
                if row["provider_timestamp_utc"] is None
                else datetime.fromisoformat(str(row["provider_timestamp_utc"]))
            ),
            original_payload=json.loads(str(row["original_payload_json"])),
            admission_run_id=(
                None if row["admission_run_id"] is None else str(row["admission_run_id"])
            ),
            admission_recorder_generation=(
                None
                if row["admission_recorder_generation"] is None
                else int(row["admission_recorder_generation"])
            ),
            connection_generation=int(row["connection_generation"]),
            subscription_owner=(
                None if row["subscription_owner"] is None else str(row["subscription_owner"])
            ),
            symbol=None if row["symbol"] is None else str(row["symbol"]),
            stream_owner=(
                None
                if row["stream_owner_json"] is None
                else json.loads(str(row["stream_owner_json"]))
            ),
            callback_classification=CallbackClassification(str(row["callback_classification"])),
            provider_envelope_event_id=(
                None
                if row["provider_envelope_event_id"] is None
                else str(row["provider_envelope_event_id"])
            ),
            lease_owner=None if row["lease_owner"] is None else str(row["lease_owner"]),
            lease_generation=int(row["lease_generation"]),
            lease_batch_id=(None if row["lease_batch_id"] is None else str(row["lease_batch_id"])),
            lease_timestamp_utc=(
                None
                if row["lease_timestamp_utc"] is None
                else datetime.fromisoformat(str(row["lease_timestamp_utc"]))
            ),
            attempt_count=int(row["attempt_count"]),
            status=InboxStatus(str(row["status"])),
            acknowledgement_timestamp_utc=(
                None
                if row["acknowledgement_timestamp_utc"] is None
                else datetime.fromisoformat(str(row["acknowledgement_timestamp_utc"]))
            ),
            failure_classification=(
                None
                if row["failure_classification"] is None
                else str(row["failure_classification"])
            ),
            associated_raw_partition_hashes=tuple(
                str(item) for item in json.loads(str(row["associated_raw_partition_hashes_json"]))
            ),
        )

    def admit(
        self,
        *,
        callback_kind: str,
        request_id: int,
        payload: Mapping[str, object],
        connection_generation: int,
        classification: CallbackClassification,
        received_utc: datetime,
        received_monotonic_ns: int | None,
        inbox_event_id: str | None = None,
        subscription_owner: str | None = None,
        symbol: str | None = None,
        stream_owner: Mapping[str, object] | None = None,
        diagnostic: bool = False,
        provider_envelope: bool = False,
        provider_envelope_event_id: str | None = None,
        allow_after_data_loss_provider_processing: bool = False,
    ) -> InboxAdmissionResult:
        """Commit the original serialisable callback before returning it."""

        kind = callback_kind.strip()
        if not kind:
            raise ValueError("callback kind is required")
        if connection_generation < 0:
            raise ValueError("connection generation must be nonnegative")
        received = _utc(received_utc, label="callback received timestamp")
        if received_monotonic_ns is not None and received_monotonic_ns < 0:
            raise ValueError("callback monotonic timestamp must be nonnegative")
        original_payload = dict(payload)
        payload_json = _encoded(original_payload)
        stream_owner_json = None if stream_owner is None else _encoded(dict(stream_owner))
        event_id = inbox_event_id or _event_id(
            callback_kind=kind,
            request_id=request_id,
            received_utc=received,
            received_monotonic_ns=received_monotonic_ns,
            admission_run_id=self.run_id,
            connection_generation=connection_generation,
            payload_json=payload_json,
        )
        if not event_id:
            raise ValueError("inbox event ID is required")
        provider = next(
            (
                parsed
                for key in (
                    "provider_timestamp_utc",
                    "source_timestamp_utc",
                    "bar_timestamp_utc",
                    "timestamp_utc",
                )
                if (parsed := _parsed_timestamp(original_payload.get(key))) is not None
            ),
            None,
        )
        received_encoded = received.isoformat()
        if provider_envelope and provider_envelope_event_id is not None:
            raise ValueError("a provider envelope cannot reference another provider envelope")
        if allow_after_data_loss_provider_processing and (
            not provider_envelope
            or classification is not CallbackClassification.AFTER_DATA_LOSS_LATCH
        ):
            raise ValueError(
                "after-data-loss processing is restricted to original provider envelopes"
            )
        initial_status = (
            InboxStatus.PROVIDER_PENDING
            if allow_after_data_loss_provider_processing
            else InboxStatus.DIAGNOSTIC
            if diagnostic
            else InboxStatus.DIAGNOSTIC
            if classification
            in {
                CallbackClassification.EXPECTED_LATE,
                CallbackClassification.PREVIOUS_CONNECTION,
            }
            else InboxStatus.QUARANTINED
            if classification
            in {
                CallbackClassification.UNKNOWN,
                CallbackClassification.AFTER_DATA_LOSS_LATCH,
            }
            else InboxStatus.PROVIDER_PENDING
            if provider_envelope
            else InboxStatus.PENDING
        )
        failure = classification.value if initial_status is InboxStatus.QUARANTINED else None
        acknowledgement = received_encoded if initial_status is InboxStatus.DIAGNOSTIC else None
        connection = self._connect()
        owns_transaction = not connection.in_transaction
        if owns_transaction:
            self._begin_immediate(connection)
        try:
            self._ensure_active_accounting(connection)
            existing = connection.execute(
                "SELECT * FROM callback_inbox_v1 WHERE inbox_event_id = ?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                stored_payload_json = str(existing["original_payload_json"])
                stored_payload_matches = stored_payload_json == payload_json
                if not stored_payload_matches:
                    try:
                        compacted = json.loads(stored_payload_json)
                    except json.JSONDecodeError:
                        compacted = None
                    stored_payload_matches = (
                        isinstance(compacted, dict)
                        and compacted.get("__retention_compacted_sha256__")
                        == hashlib.sha256(payload_json.encode()).hexdigest()
                    )
                if (
                    str(existing["callback_kind"]) != kind
                    or int(existing["request_id"]) != request_id
                    or existing["admission_run_id"] != self.run_id
                    or int(existing["connection_generation"]) != connection_generation
                    or existing["stream_owner_json"] != stream_owner_json
                    or not stored_payload_matches
                ):
                    if owns_transaction:
                        connection.rollback()
                    raise CallbackIdentityCollision("CALLBACK_IDENTITY_COLLISION")
                if provider_envelope_event_id is not None:
                    self._complete_provider_envelope_in_transaction(
                        connection,
                        provider_envelope_event_id=provider_envelope_event_id,
                        canonical_event_id=event_id,
                        completed_at=received,
                    )
                if owns_transaction:
                    connection.commit()
                return InboxAdmissionResult(event=self._event(existing), duplicate=True)
            referenced_provider_active = (
                provider_envelope_event_id is not None
                and connection.execute(
                    """
                    SELECT 1
                    FROM callback_inbox_v1
                    WHERE inbox_event_id = ? AND admission_run_id IS ?
                      AND status = 'provider_pending'
                    """,
                    (provider_envelope_event_id, self.run_id),
                ).fetchone()
                is not None
            )
            assert self._active_count is not None
            unacknowledged = self._active_count - int(referenced_provider_active)
            if unacknowledged >= self.max_unacknowledged:
                if owns_transaction:
                    connection.rollback()
                raise CallbackInboxOverflow("CALLBACK_OVERFLOW")
            admitted = datetime.now(UTC)
            admitted_encoded = admitted.isoformat()
            cursor = connection.execute(
                """
                INSERT INTO callback_inbox_v1(
                    inbox_event_id, callback_kind, request_id, received_utc,
                    received_monotonic_ns, provider_timestamp_utc,
                    original_payload_json, admission_run_id,
                    admission_recorder_generation, connection_generation,
                    subscription_owner, symbol, stream_owner_json,
                    callback_classification,
                    provider_envelope_event_id, status, acknowledgement_timestamp_utc,
                    failure_classification, admitted_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    kind,
                    request_id,
                    received_encoded,
                    received_monotonic_ns,
                    None if provider is None else provider.isoformat(),
                    payload_json,
                    self.run_id,
                    self.recorder_generation,
                    connection_generation,
                    subscription_owner,
                    symbol,
                    stream_owner_json,
                    classification.value,
                    provider_envelope_event_id,
                    initial_status.value,
                    acknowledgement,
                    failure,
                    admitted_encoded,
                    admitted_encoded,
                ),
            )
            assert cursor.lastrowid is not None
            source_sequence = int(cursor.lastrowid)
            if provider_envelope_event_id is not None:
                self._complete_provider_envelope_in_transaction(
                    connection,
                    provider_envelope_event_id=provider_envelope_event_id,
                    canonical_event_id=event_id,
                    completed_at=received,
                )
            if initial_status in {
                InboxStatus.PROVIDER_PENDING,
                InboxStatus.PENDING,
                InboxStatus.LEASED,
                InboxStatus.QUARANTINED,
            }:
                self._increment_active_accounting(received_at=received)
            self._update_callback_heartbeats(
                connection,
                received_at=received,
                admitted_at=admitted,
            )
            if owns_transaction:
                connection.commit()
            row = connection.execute(
                "SELECT * FROM callback_inbox_v1 WHERE source_sequence = ?",
                (source_sequence,),
            ).fetchone()
        except Exception:
            if owns_transaction and connection.in_transaction:
                connection.rollback()
            raise
        assert row is not None
        return InboxAdmissionResult(event=self._event(row), duplicate=False)

    def attach_stream_owner(
        self,
        *,
        request_id: int,
        connection_generation: int,
        stream_owner: Mapping[str, object],
        attached_at: datetime,
    ) -> int:
        """Backfill ownership for callbacks delivered during request startup.

        IBKR may invoke a callback before the request method returns.  The
        callback is already durable at that point; registration then fills the
        typed owner only on still-unprocessed rows and never rewrites an
        existing ownership receipt.
        """

        observed = _utc(attached_at, label="stream owner attachment timestamp")
        encoded_owner = _encoded(dict(stream_owner))
        with self._connect() as connection:
            self._begin_immediate(connection)
            conflicting = connection.execute(
                """
                SELECT inbox_event_id
                FROM callback_inbox_v1
                WHERE admission_run_id IS ?
                  AND admission_recorder_generation IS ?
                  AND request_id = ?
                  AND connection_generation = ?
                  AND status IN ('provider_pending', 'pending')
                  AND stream_owner_json IS NOT NULL
                  AND stream_owner_json <> ?
                LIMIT 1
                """,
                (
                    self.run_id,
                    self.recorder_generation,
                    request_id,
                    connection_generation,
                    encoded_owner,
                ),
            ).fetchone()
            if conflicting is not None:
                connection.rollback()
                raise CallbackIdentityCollision("CALLBACK_STREAM_OWNER_COLLISION")
            cursor = connection.execute(
                """
                UPDATE callback_inbox_v1
                SET stream_owner_json = ?, updated_at_utc = ?
                WHERE admission_run_id IS ?
                  AND admission_recorder_generation IS ?
                  AND request_id = ?
                  AND connection_generation = ?
                  AND status IN ('provider_pending', 'pending')
                  AND stream_owner_json IS NULL
                """,
                (
                    encoded_owner,
                    observed.isoformat(),
                    self.run_id,
                    self.recorder_generation,
                    request_id,
                    connection_generation,
                ),
            )
            connection.commit()
            return cursor.rowcount

    def materialize_provider_envelope(
        self,
        *,
        provider_envelope_event_id: str,
        callback_kind: str,
        request_id: int,
        payload: Mapping[str, object],
        materialized_at: datetime,
    ) -> CallbackInboxEvent:
        """Transition one pre-admitted provider delivery into its canonical row."""

        kind = callback_kind.strip()
        if not kind:
            raise ValueError("callback kind is required")
        observed = _utc(materialized_at, label="provider envelope materialization timestamp")
        payload_json = _encoded(dict(payload))
        provider = next(
            (
                parsed
                for key in (
                    "provider_timestamp_utc",
                    "source_timestamp_utc",
                    "bar_timestamp_utc",
                    "timestamp_utc",
                )
                if (parsed := _parsed_timestamp(payload.get(key))) is not None
            ),
            None,
        )
        connection = self._connect()
        owns_transaction = not connection.in_transaction
        if owns_transaction:
            self._begin_immediate(connection)
        try:
            row = connection.execute(
                "SELECT * FROM callback_inbox_v1 WHERE inbox_event_id = ?",
                (provider_envelope_event_id,),
            ).fetchone()
            if row is None:
                if owns_transaction:
                    connection.rollback()
                raise CallbackInboxError("CALLBACK_PROVIDER_ENVELOPE_MISSING")
            if (
                not str(row["callback_kind"]).startswith("official_provider_")
                or int(row["request_id"]) != request_id
                or row["admission_run_id"] != self.run_id
                or InboxStatus(str(row["status"])) is not InboxStatus.PROVIDER_PENDING
            ):
                if owns_transaction:
                    connection.rollback()
                raise CallbackInboxError("CALLBACK_PROVIDER_ENVELOPE_IDENTITY_INVALID")
            cursor = connection.execute(
                """
                UPDATE callback_inbox_v1
                SET callback_kind = ?, provider_timestamp_utc = ?,
                    original_payload_json = ?, status = 'pending',
                    updated_at_utc = ?
                WHERE inbox_event_id = ? AND status = 'provider_pending'
                  AND admission_run_id IS ?
                """,
                (
                    kind,
                    None if provider is None else provider.isoformat(),
                    payload_json,
                    observed.isoformat(),
                    provider_envelope_event_id,
                    self.run_id,
                ),
            )
            if cursor.rowcount != 1:
                if owns_transaction:
                    connection.rollback()
                raise CallbackInboxError("CALLBACK_PROVIDER_ENVELOPE_TRANSITION_CHANGED")
            if owns_transaction:
                connection.commit()
            materialized = connection.execute(
                "SELECT * FROM callback_inbox_v1 WHERE inbox_event_id = ?",
                (provider_envelope_event_id,),
            ).fetchone()
        except Exception:
            if owns_transaction and connection.in_transaction:
                connection.rollback()
            raise
        assert materialized is not None
        return self._event(materialized)

    def _complete_provider_envelope_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        provider_envelope_event_id: str,
        canonical_event_id: str | None,
        completed_at: datetime,
    ) -> None:
        """Atomically bind the original provider delivery to its canonical row."""

        self._ensure_active_accounting(connection)
        provider = connection.execute(
            """
            SELECT callback_kind, admission_run_id, status
            FROM callback_inbox_v1
            WHERE inbox_event_id = ?
            """,
            (provider_envelope_event_id,),
        ).fetchone()
        if provider is None:
            raise CallbackInboxError("CALLBACK_PROVIDER_ENVELOPE_MISSING")
        if (
            not str(provider["callback_kind"]).startswith("official_provider_")
            or provider["admission_run_id"] != self.run_id
        ):
            raise CallbackInboxError("CALLBACK_PROVIDER_ENVELOPE_IDENTITY_INVALID")
        status = InboxStatus(str(provider["status"]))
        if status is InboxStatus.DIAGNOSTIC:
            return
        if status is not InboxStatus.PROVIDER_PENDING:
            raise CallbackInboxError("CALLBACK_PROVIDER_ENVELOPE_TRANSITION_CHANGED")
        failure = (
            "PROVIDER_ENVELOPE_CONTROL_COMPLETED"
            if canonical_event_id is None
            else f"PROVIDER_ENVELOPE_MATERIALIZED:{canonical_event_id}"
        )
        cursor = connection.execute(
            """
            UPDATE callback_inbox_v1
            SET status = 'diagnostic', acknowledgement_timestamp_utc = ?,
                failure_classification = ?, updated_at_utc = ?
            WHERE inbox_event_id = ? AND status = 'provider_pending'
              AND admission_run_id IS ?
            """,
            (
                completed_at.isoformat(),
                failure,
                completed_at.isoformat(),
                provider_envelope_event_id,
                self.run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise CallbackInboxError("CALLBACK_PROVIDER_ENVELOPE_TRANSITION_CHANGED")
        self._decrement_active_accounting()

    def complete_provider_envelope(
        self,
        *,
        provider_envelope_event_id: str,
        completed_at: datetime,
    ) -> None:
        """Complete a provider callback that legitimately emitted no canonical row."""

        observed = _utc(completed_at, label="provider envelope completion timestamp")
        with self._connect() as connection:
            self._begin_immediate(connection)
            self._complete_provider_envelope_in_transaction(
                connection,
                provider_envelope_event_id=provider_envelope_event_id,
                canonical_event_id=None,
                completed_at=observed,
            )
            connection.commit()

    def quarantine_provider_envelope(
        self,
        *,
        provider_envelope_event_id: str,
        failure_classification: str,
        quarantined_at: datetime,
    ) -> None:
        """Best-effort terminal transition after canonical materialisation failed."""

        reason = failure_classification.strip()
        if not reason:
            raise ValueError("provider envelope quarantine reason is required")
        observed = _utc(quarantined_at, label="provider envelope quarantine timestamp")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE callback_inbox_v1
                SET status = 'quarantined', failure_classification = ?,
                    updated_at_utc = ?
                WHERE inbox_event_id = ? AND status = 'provider_pending'
                  AND admission_run_id IS ?
                """,
                (
                    reason,
                    observed.isoformat(),
                    provider_envelope_event_id,
                    self.run_id,
                ),
            )
            if cursor.rowcount not in {0, 1}:
                raise CallbackInboxError("CALLBACK_PROVIDER_QUARANTINE_CHANGED")

    def quarantine_interrupted_provider_envelopes(
        self,
        *,
        current_recorder_generation: int,
        observed_at: datetime,
    ) -> tuple[CallbackInboxEvent, ...]:
        """Quarantine provider deliveries left half-transitioned by an old process."""

        if current_recorder_generation <= 0:
            raise ValueError("recorder generation must be positive")
        observed = _utc(observed_at, label="provider envelope recovery timestamp")
        with self._connect() as connection:
            self._begin_immediate(connection)
            rows = connection.execute(
                """
                SELECT *
                FROM callback_inbox_v1
                WHERE admission_run_id IS ?
                  AND status = 'provider_pending'
                  AND (
                      admission_recorder_generation IS NULL
                      OR admission_recorder_generation <> ?
                  )
                ORDER BY source_sequence
                """,
                (self.run_id, current_recorder_generation),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE callback_inbox_v1
                    SET status = 'quarantined',
                        failure_classification =
                            'CALLBACK_PROVIDER_MATERIALIZATION_INTERRUPTED',
                        updated_at_utc = ?
                    WHERE inbox_event_id = ? AND status = 'provider_pending'
                    """,
                    (observed.isoformat(), str(row["inbox_event_id"])),
                )
            connection.commit()
        return tuple(self._event(row) for row in rows)

    def _update_callback_heartbeats(
        self,
        connection: sqlite3.Connection,
        *,
        received_at: datetime,
        admitted_at: datetime | None,
    ) -> None:
        if self.run_id is None or self.recorder_generation is None:
            return
        self._ensure_active_accounting(connection)
        assert self._active_count is not None
        connection.execute(
            """
            UPDATE recorder_operational_state_v1
            SET latest_callback_received_at_utc = ?,
                latest_callback_durably_admitted_at_utc =
                    COALESCE(?, latest_callback_durably_admitted_at_utc),
                inbox_backlog = ?,
                oldest_unacknowledged_at_utc = ?,
                updated_at_utc = ?
            WHERE run_id = ? AND recorder_generation = ?
            """,
            (
                received_at.isoformat(),
                None if admitted_at is None else admitted_at.isoformat(),
                self._active_count,
                (
                    None
                    if self._oldest_active_received_at is None
                    else self._oldest_active_received_at.isoformat()
                ),
                received_at.isoformat(),
                self.run_id,
                self.recorder_generation,
            ),
        )

    def lease(
        self,
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
        lease_timeout: timedelta,
        limit: int,
    ) -> tuple[CallbackInboxEvent, ...]:
        if not lease_owner or lease_generation <= 0:
            raise ValueError("callback lease identity is invalid")
        if lease_timeout <= timedelta(0) or limit <= 0:
            raise ValueError("callback lease bounds must be positive")
        observed = _utc(now, label="callback lease timestamp")
        expired_before = (observed - lease_timeout).isoformat()
        leased: list[sqlite3.Row] = []

        def owner_binding_pending(row: sqlite3.Row) -> bool:
            return (
                int(row["request_id"]) >= 0
                and row["stream_owner_json"] is None
                and row["admission_recorder_generation"] == self.recorder_generation
            )

        with self._connect() as connection:
            self._begin_immediate(connection)
            # An explicitly abandoned run retains its poison/pending evidence
            # for audit, but it is never part of a different run's backlog.
            # Run identity is the isolation boundary for leasing and health.
            connection.execute(
                """
                UPDATE callback_inbox_v1
                SET status = 'pending', lease_owner = NULL,
                    lease_timestamp_utc = NULL, updated_at_utc = ?
                WHERE status = 'leased' AND lease_timestamp_utc <= ?
                  AND admission_run_id IS ?
                """,
                (observed.isoformat(), expired_before, self.run_id),
            )
            oldest = connection.execute(
                """
                SELECT inbox_event_id, status, lease_batch_id, request_id,
                       stream_owner_json, admission_recorder_generation
                FROM callback_inbox_v1
                WHERE status IN ('pending', 'leased')
                  AND admission_run_id IS ?
                ORDER BY source_sequence, received_monotonic_ns, inbox_event_id
                LIMIT 1
                """,
                (self.run_id,),
            ).fetchone()
            if oldest is None or str(oldest["status"]) == "leased" or owner_binding_pending(oldest):
                event_ids: tuple[str, ...] = ()
                batch_id = None
            elif oldest["lease_batch_id"] is not None:
                batch_id = str(oldest["lease_batch_id"])
                rows = connection.execute(
                    """
                    SELECT inbox_event_id, request_id, stream_owner_json,
                           admission_recorder_generation
                    FROM callback_inbox_v1
                    WHERE status = 'pending'
                      AND admission_run_id IS ?
                      AND lease_batch_id = ?
                    ORDER BY source_sequence, received_monotonic_ns, inbox_event_id
                    """,
                    (self.run_id, batch_id),
                ).fetchall()
                ready_rows = []
                for row in rows:
                    if owner_binding_pending(row):
                        break
                    ready_rows.append(row)
                event_ids = tuple(str(row["inbox_event_id"]) for row in ready_rows)
                if not event_ids:
                    batch_id = None
            else:
                rows = connection.execute(
                    """
                    SELECT inbox_event_id, request_id, stream_owner_json,
                           admission_recorder_generation
                    FROM callback_inbox_v1
                    WHERE status = 'pending'
                      AND admission_run_id IS ?
                      AND lease_batch_id IS NULL
                    ORDER BY source_sequence, received_monotonic_ns, inbox_event_id
                    LIMIT ?
                    """,
                    (self.run_id, limit),
                ).fetchall()
                ready_rows = []
                for row in rows:
                    if owner_binding_pending(row):
                        break
                    ready_rows.append(row)
                event_ids = tuple(str(row["inbox_event_id"]) for row in ready_rows)
                batch_id = (
                    None
                    if not event_ids
                    else hashlib.sha256(
                        "|".join(
                            (
                                str(self.run_id),
                                str(lease_generation),
                                *event_ids,
                            )
                        ).encode()
                    ).hexdigest()
                )
            for event_id in event_ids:
                cursor = connection.execute(
                    """
                    UPDATE callback_inbox_v1
                    SET status = 'leased', lease_owner = ?,
                        lease_generation = ?, lease_batch_id = ?,
                        lease_timestamp_utc = ?,
                        attempt_count = attempt_count + 1, updated_at_utc = ?
                    WHERE inbox_event_id = ? AND status = 'pending'
                      AND admission_run_id IS ?
                    """,
                    (
                        lease_owner,
                        lease_generation,
                        batch_id,
                        observed.isoformat(),
                        observed.isoformat(),
                        event_id,
                        self.run_id,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise CallbackLeaseLost("CALLBACK_LEASE_CHANGED")
            leased = (
                []
                if not event_ids
                else connection.execute(
                    f"""
                    SELECT *
                    FROM callback_inbox_v1
                    WHERE inbox_event_id IN ({",".join("?" for _ in event_ids)})
                    ORDER BY source_sequence, received_monotonic_ns, inbox_event_id
                    """,
                    event_ids,
                ).fetchall()
            )
            connection.commit()
        return tuple(self._event(row) for row in leased)

    def has_unacknowledged_stream_owner(
        self,
        stream_owner: Mapping[str, object],
    ) -> bool:
        """Return whether this run can still lease evidence for one owner."""

        encoded_owner = _encoded(dict(stream_owner))
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM callback_inbox_v1
                WHERE admission_run_id IS ?
                  AND stream_owner_json = ?
                  AND status IN ('provider_pending', 'pending', 'leased')
                LIMIT 1
                """,
                (self.run_id, encoded_owner),
            ).fetchone()
        return row is not None

    def commit_raw_materialization(
        self,
        events: Iterable[CallbackInboxEvent],
        *,
        run_id: str,
        recorder_generation: int,
        raw_partition_hashes: tuple[str, ...],
        raw_event_ids: tuple[str, ...],
        materialized_at: datetime,
    ) -> None:
        """Bind a retry-stable callback batch to immutable raw partitions."""

        batch = tuple(events)
        if not batch:
            return
        batch_ids = {event.lease_batch_id for event in batch}
        if None in batch_ids or len(batch_ids) != 1:
            raise CallbackInboxError("CALLBACK_RAW_BATCH_ID_INVALID")
        batch_id = next(iter(batch_ids))
        assert batch_id is not None
        observed = _utc(materialized_at, label="callback raw materialization timestamp")
        encoded_hashes = _encoded(tuple(sorted(set(raw_partition_hashes))))
        encoded_event_ids = _encoded(tuple(sorted(raw_event_ids)))
        canonical_event_id = min(
            batch,
            key=lambda event: (event.source_sequence, event.inbox_event_id),
        ).inbox_event_id
        with self._connect() as connection:
            self._begin_immediate(connection)
            for event in batch:
                if event.admission_run_id != run_id:
                    connection.rollback()
                    raise CallbackInboxError("CALLBACK_RAW_MATERIALIZATION_RUN_DIFFERS")
                existing = connection.execute(
                    """
                    SELECT run_id, lease_batch_id, raw_partition_hashes_json,
                           raw_event_ids_json
                    FROM callback_raw_materialization_v1
                    WHERE inbox_event_id = ?
                    """,
                    (event.inbox_event_id,),
                ).fetchone()
                expected_hashes = (
                    encoded_hashes if event.inbox_event_id == canonical_event_id else "[]"
                )
                expected_event_ids = (
                    encoded_event_ids if event.inbox_event_id == canonical_event_id else "[]"
                )
                if existing is not None:
                    existing_event_ids = str(existing["raw_event_ids_json"])
                    if (
                        str(existing["run_id"]) != run_id
                        or str(existing["lease_batch_id"]) != batch_id
                        or str(existing["raw_partition_hashes_json"])
                        not in {encoded_hashes, expected_hashes}
                        or existing_event_ids
                        not in {encoded_event_ids, expected_event_ids}
                    ):
                        connection.rollback()
                        raise CallbackInboxError("CALLBACK_RAW_MATERIALIZATION_DIFFERS")
                    continue
                connection.execute(
                    """
                    INSERT INTO callback_raw_materialization_v1(
                        inbox_event_id, source_sequence, run_id,
                        recorder_generation, lease_batch_id,
                        raw_partition_hashes_json, raw_event_ids_json,
                        materialized_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.inbox_event_id,
                        event.source_sequence,
                        run_id,
                        recorder_generation,
                        batch_id,
                        expected_hashes,
                        expected_event_ids,
                        observed.isoformat(),
                    ),
                )
            connection.commit()

    def raw_materialization(
        self,
        events: Iterable[CallbackInboxEvent],
    ) -> RawMaterialization | None:
        """Return the one materialization shared by a complete lease batch."""

        batch = tuple(events)
        if not batch:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT inbox_event_id, source_sequence, lease_batch_id,
                       raw_partition_hashes_json, raw_event_ids_json
                FROM callback_raw_materialization_v1
                WHERE inbox_event_id IN ({",".join("?" for _ in batch)})
                """,
                tuple(event.inbox_event_id for event in batch),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != len(batch):
            raise CallbackInboxError("CALLBACK_RAW_MATERIALIZATION_PARTIAL")
        batch_ids = {str(row["lease_batch_id"]) for row in rows}
        hash_encodings = [str(row["raw_partition_hashes_json"]) for row in rows]
        populated_hashes = {value for value in hash_encodings if value != "[]"}
        event_id_encodings = [str(row["raw_event_ids_json"]) for row in rows]
        populated_event_ids = {value for value in event_id_encodings if value != "[]"}
        expected_batch_ids = {event.lease_batch_id for event in batch}
        if (
            len(batch_ids) != 1
            or len(populated_hashes) > 1
            or len(populated_event_ids) > 1
            or None in expected_batch_ids
            or batch_ids != expected_batch_ids
        ):
            raise CallbackInboxError("CALLBACK_RAW_MATERIALIZATION_INCONSISTENT")
        canonical = min(
            rows,
            key=lambda row: (int(row["source_sequence"]), str(row["inbox_event_id"])),
        )
        if (
            populated_hashes
            and "[]" in hash_encodings
            and str(canonical["raw_partition_hashes_json"]) == "[]"
        ) or (
            populated_event_ids
            and "[]" in event_id_encodings
            and str(canonical["raw_event_ids_json"]) == "[]"
        ):
                raise CallbackInboxError("CALLBACK_RAW_MATERIALIZATION_INCONSISTENT")
        return RawMaterialization(
            partition_hashes=tuple(
                str(item)
                for item in json.loads(
                    "[]" if not populated_hashes else next(iter(populated_hashes))
                )
            ),
            raw_event_ids=tuple(
                str(item)
                for item in json.loads(
                    "[]" if not populated_event_ids else next(iter(populated_event_ids))
                )
            ),
        )

    def release(
        self,
        events: Iterable[CallbackInboxEvent],
        *,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> int:
        observed = _utc(now, label="callback release timestamp")
        released = 0
        with self._connect() as connection:
            self._begin_immediate(connection)
            for event in events:
                cursor = connection.execute(
                    """
                    UPDATE callback_inbox_v1
                    SET status = 'pending', lease_owner = NULL,
                        lease_timestamp_utc = NULL, updated_at_utc = ?
                    WHERE inbox_event_id = ? AND status = 'leased'
                      AND lease_owner = ? AND lease_generation = ?
                    """,
                    (
                        observed.isoformat(),
                        event.inbox_event_id,
                        lease_owner,
                        lease_generation,
                    ),
                )
                released += cursor.rowcount
            connection.commit()
        return released

    def quarantine(
        self,
        event: CallbackInboxEvent,
        *,
        failure_classification: str,
        lease_owner: str,
        lease_generation: int,
        now: datetime,
    ) -> None:
        reason = failure_classification.strip()
        if not reason:
            raise ValueError("quarantine reason is required")
        observed = _utc(now, label="callback quarantine timestamp")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE callback_inbox_v1
                SET status = 'quarantined', failure_classification = ?,
                    lease_owner = NULL, lease_timestamp_utc = NULL,
                    updated_at_utc = ?
                WHERE inbox_event_id = ? AND status = 'leased'
                  AND lease_owner = ? AND lease_generation = ?
                """,
                (
                    reason,
                    observed.isoformat(),
                    event.inbox_event_id,
                    lease_owner,
                    lease_generation,
                ),
            )
            if cursor.rowcount != 1:
                raise CallbackLeaseLost("CALLBACK_QUARANTINE_LEASE_CHANGED")

    def release_quarantined_for_retry(
        self,
        *,
        inbox_event_id: str,
        resolution_evidence: str,
        resolved_at: datetime,
    ) -> None:
        """Explicitly return one audited poison event to its stable batch."""

        evidence = resolution_evidence.strip()
        if not evidence:
            raise ValueError("quarantine resolution evidence is required")
        observed = _utc(resolved_at, label="quarantine resolution timestamp")
        with self._connect() as connection:
            self._begin_immediate(connection)
            cursor = connection.execute(
                """
                UPDATE callback_inbox_v1
                SET status = 'pending', lease_owner = NULL,
                    lease_timestamp_utc = NULL, updated_at_utc = ?
                WHERE inbox_event_id = ? AND status = 'quarantined'
                  AND admission_run_id IS ?
                """,
                (observed.isoformat(), inbox_event_id, self.run_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise CallbackInboxError("CALLBACK_QUARANTINE_RESOLUTION_CHANGED")
            connection.commit()
        self.record_incident(
            stable_error_code="CALLBACK_QUARANTINE_RELEASED_FOR_RETRY",
            component="durable_callback_inbox",
            severity="diagnostic",
            occurred_at=observed,
            error_class="OperatorResolution",
            evidence_loss_possible=False,
            details={
                "inbox_event_id": inbox_event_id,
                "resolution_evidence": evidence,
            },
        )

    def resolve_quarantined_bootstrap_provider_envelope(
        self,
        *,
        inbox_event_id: str,
        expected_failure_classification: str,
        resolution_evidence: str,
        resolved_at: datetime,
    ) -> None:
        """Retire a non-scientific bootstrap envelope with atomic audit evidence."""

        evidence = resolution_evidence.strip()
        expected_failure = expected_failure_classification.strip()
        if not evidence:
            raise ValueError("provider-envelope resolution evidence is required")
        if not expected_failure:
            raise ValueError("expected provider-envelope failure is required")
        observed = _utc(resolved_at, label="provider-envelope resolution timestamp")
        allowed_kinds = {
            "official_provider_contract_details",
            "official_provider_contract_details_end",
        }
        with self._connect() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                """
                SELECT *
                FROM callback_inbox_v1
                WHERE inbox_event_id = ? AND admission_run_id IS ?
                """,
                (inbox_event_id, self.run_id),
            ).fetchone()
            if row is None or str(row["status"]) != InboxStatus.QUARANTINED.value:
                connection.rollback()
                raise CallbackInboxError("CALLBACK_BOOTSTRAP_QUARANTINE_RESOLUTION_CHANGED")
            callback_kind = str(row["callback_kind"])
            if callback_kind not in allowed_kinds or row["provider_envelope_event_id"] is not None:
                connection.rollback()
                raise CallbackInboxError("CALLBACK_BOOTSTRAP_QUARANTINE_NOT_RESOLVABLE")
            if str(row["failure_classification"]) != expected_failure:
                connection.rollback()
                raise CallbackInboxError("CALLBACK_BOOTSTRAP_QUARANTINE_FAILURE_CHANGED")
            cursor = connection.execute(
                """
                UPDATE callback_inbox_v1
                SET status = 'diagnostic',
                    acknowledgement_timestamp_utc = ?,
                    lease_owner = NULL,
                    lease_timestamp_utc = NULL,
                    updated_at_utc = ?
                WHERE inbox_event_id = ? AND admission_run_id IS ?
                  AND status = 'quarantined'
                  AND failure_classification = ?
                """,
                (
                    observed.isoformat(),
                    observed.isoformat(),
                    inbox_event_id,
                    self.run_id,
                    expected_failure,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise CallbackInboxError("CALLBACK_BOOTSTRAP_QUARANTINE_RESOLUTION_CHANGED")
            details_json = _encoded(
                {
                    "inbox_event_id": inbox_event_id,
                    "original_failure_classification": expected_failure,
                    "resolution_evidence": evidence,
                    "replacement_generation_must_reissue_request": True,
                    "scientific_projection_permitted": False,
                }
            )
            incident_id = hashlib.sha256(
                "|".join(
                    (
                        "CALLBACK_BOOTSTRAP_PROVIDER_QUARANTINE_RESOLVED",
                        str(self.run_id),
                        inbox_event_id,
                        observed.isoformat(),
                        details_json,
                    )
                ).encode()
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO operational_incident_v1(
                    incident_id, run_id, recorder_generation,
                    connection_generation, occurred_at_utc, component,
                    severity, stable_error_code, callback_kind, request_id,
                    source_sequence, subscription_owner, symbol, error_class,
                    evidence_loss_possible, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident_id,
                    self.run_id,
                    int(row["admission_recorder_generation"]),
                    int(row["connection_generation"]),
                    observed.isoformat(),
                    "durable_callback_inbox",
                    "diagnostic",
                    "CALLBACK_BOOTSTRAP_PROVIDER_QUARANTINE_RESOLVED",
                    callback_kind,
                    int(row["request_id"]),
                    int(row["source_sequence"]),
                    row["subscription_owner"],
                    row["symbol"],
                    "OperatorBootstrapResolution",
                    0,
                    details_json,
                ),
            )
            connection.commit()
            self._active_count = None
            self._oldest_active_received_at = None

    def resolve_fatal_latch(
        self,
        *,
        latch_kind: Literal["ingestion", "storage"],
        expected_stable_error_code: str,
        resolution_evidence: str,
        resolved_at: datetime,
    ) -> None:
        """Resolve only an exact audited latch; never infer recovery."""

        evidence = resolution_evidence.strip()
        if not evidence:
            raise ValueError("fatal-latch resolution evidence is required")
        observed = _utc(resolved_at, label="fatal-latch resolution timestamp")
        with self._connect() as connection:
            self._begin_immediate(connection)
            if latch_kind == "ingestion":
                quarantined = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM callback_inbox_v1
                        WHERE admission_run_id IS ? AND status = 'quarantined'
                        """,
                        (self.run_id,),
                    ).fetchone()[0]
                )
                if quarantined:
                    connection.rollback()
                    raise CallbackInboxError("CALLBACK_FATAL_RESOLUTION_HAS_QUARANTINED_EVENTS")
            cursor = connection.execute(
                """
                UPDATE recorder_fatal_latch_v1
                SET resolved_at_utc = ?, resolution_evidence = ?
                WHERE run_id IS ? AND latch_kind = ?
                  AND stable_error_code = ? AND resolved_at_utc IS NULL
                """,
                (
                    observed.isoformat(),
                    evidence,
                    self.run_id,
                    latch_kind,
                    expected_stable_error_code,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise CallbackInboxError("CALLBACK_FATAL_RESOLUTION_CHANGED")
            fatal_column = (
                "fatal_ingestion_code" if latch_kind == "ingestion" else "fatal_storage_code"
            )
            connection.execute(
                f"""
                UPDATE recorder_operational_state_v1
                SET {fatal_column} = NULL,
                    state = 'SCIENTIFICALLY_BLOCKED',
                    state_reason_code = 'FATAL_LATCH_RESOLVED_RESTART_REQUIRED',
                    scientific_recording_valid = 0,
                    updated_at_utc = ?
                WHERE run_id IS ?
                """,
                (observed.isoformat(), self.run_id),
            )
            connection.commit()

    def commit_processing(
        self,
        events: Iterable[CallbackInboxEvent],
        *,
        run_id: str,
        recorder_generation: int,
        raw_partition_hashes: tuple[str, ...],
        processing_disposition: Literal[
            "normal_scientific_projection",
            "scientifically_blocked_raw_only",
        ] = "normal_scientific_projection",
        scientific_projection_complete: bool = True,
        committed_at: datetime,
    ) -> None:
        if (
            processing_disposition == "normal_scientific_projection"
            and not scientific_projection_complete
        ):
            raise ValueError("normal callback processing must complete scientific projection")
        if (
            processing_disposition == "scientifically_blocked_raw_only"
            and scientific_projection_complete
        ):
            raise ValueError("blocked raw-only processing cannot claim scientific projection")
        observed = _utc(committed_at, label="callback processing commit timestamp")
        encoded_hashes = _encoded(tuple(sorted(set(raw_partition_hashes))))
        batch = tuple(events)
        canonical_event_id = (
            None
            if not batch
            else min(
                batch,
                key=lambda event: (event.source_sequence, event.inbox_event_id),
            ).inbox_event_id
        )
        with self._connect() as connection:
            self._begin_immediate(connection)
            for event in batch:
                if event.admission_run_id != run_id:
                    connection.rollback()
                    raise CallbackInboxError("CALLBACK_PROCESSING_RUN_DIFFERS")
                existing = connection.execute(
                    """
                    SELECT run_id, recorder_generation, raw_partition_hashes_json,
                           processing_disposition, scientific_projection_complete
                    FROM callback_processing_commit_v1
                    WHERE inbox_event_id = ?
                    """,
                    (event.inbox_event_id,),
                ).fetchone()
                expected_hashes = (
                    encoded_hashes if event.inbox_event_id == canonical_event_id else "[]"
                )
                if existing is not None:
                    if (
                        str(existing["run_id"]) != run_id
                        or int(existing["recorder_generation"]) != recorder_generation
                        or str(existing["raw_partition_hashes_json"])
                        not in {encoded_hashes, expected_hashes}
                        or str(existing["processing_disposition"]) != processing_disposition
                        or bool(existing["scientific_projection_complete"])
                        is not scientific_projection_complete
                    ):
                        connection.rollback()
                        raise CallbackInboxError("CALLBACK_PROCESSING_COMMIT_DIFFERS")
                    continue
                connection.execute(
                    """
                    INSERT INTO callback_processing_commit_v1(
                        inbox_event_id, source_sequence, run_id,
                        recorder_generation, raw_partition_hashes_json,
                        processing_disposition, scientific_projection_complete,
                        committed_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.inbox_event_id,
                        event.source_sequence,
                        run_id,
                        recorder_generation,
                        expected_hashes,
                        processing_disposition,
                        int(scientific_projection_complete),
                        observed.isoformat(),
                    ),
                )
            connection.commit()

    def processing_commit(
        self,
        inbox_event_id: str,
    ) -> tuple[str, ...] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT raw_partition_hashes_json
                FROM callback_processing_commit_v1
                WHERE inbox_event_id = ?
                """,
                (inbox_event_id,),
            ).fetchone()
        if row is None:
            return None
        return tuple(str(item) for item in json.loads(str(row["raw_partition_hashes_json"])))

    def acknowledge(
        self,
        events: Iterable[CallbackInboxEvent],
        *,
        lease_owner: str,
        lease_generation: int,
        raw_partition_hashes: tuple[str, ...],
        acknowledged_at: datetime,
    ) -> int:
        """Ack only the exact current lease generation after processing commit."""

        observed = _utc(acknowledged_at, label="callback acknowledgement timestamp")
        encoded_hashes = _encoded(tuple(sorted(set(raw_partition_hashes))))
        batch = tuple(events)
        canonical_event_id = (
            None
            if not batch
            else min(
                batch,
                key=lambda event: (event.source_sequence, event.inbox_event_id),
            ).inbox_event_id
        )
        acknowledged = 0
        with self._connect() as connection:
            self._begin_immediate(connection)
            for event in batch:
                commit = connection.execute(
                    """
                    SELECT raw_partition_hashes_json
                    FROM callback_processing_commit_v1
                    WHERE inbox_event_id = ?
                    """,
                    (event.inbox_event_id,),
                ).fetchone()
                if commit is None:
                    connection.rollback()
                    raise CallbackInboxError("CALLBACK_ACK_BEFORE_PROCESSING_COMMIT")
                expected_hashes = (
                    encoded_hashes if event.inbox_event_id == canonical_event_id else "[]"
                )
                committed_hashes = str(commit["raw_partition_hashes_json"])
                if committed_hashes not in {encoded_hashes, expected_hashes}:
                    connection.rollback()
                    raise CallbackInboxError("CALLBACK_ACK_PARTITION_HASHES_DIFFER")
                cursor = connection.execute(
                    """
                    UPDATE callback_inbox_v1
                    SET status = 'acknowledged',
                        acknowledgement_timestamp_utc = ?,
                        associated_raw_partition_hashes_json = ?,
                        lease_owner = NULL, lease_timestamp_utc = NULL,
                        updated_at_utc = ?
                    WHERE inbox_event_id = ? AND status = 'leased'
                      AND lease_owner = ? AND lease_generation = ?
                    """,
                    (
                        observed.isoformat(),
                        committed_hashes,
                        observed.isoformat(),
                        event.inbox_event_id,
                        lease_owner,
                        lease_generation,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise CallbackLeaseLost("CALLBACK_ACK_LEASE_CHANGED")
                acknowledged += 1
            accounting = connection.execute(
                """
                SELECT COUNT(*) AS backlog, MIN(received_utc) AS oldest
                FROM callback_inbox_v1
                WHERE admission_run_id IS ?
                  AND status IN (
                      'provider_pending', 'pending', 'leased', 'quarantined'
                  )
                """,
                (self.run_id,),
            ).fetchone()
            assert accounting is not None
            self._active_count_run_id = self.run_id
            self._active_count = int(accounting["backlog"])
            self._oldest_active_received_at = (
                None
                if accounting["oldest"] is None
                else datetime.fromisoformat(str(accounting["oldest"]))
            )
            if self.run_id is not None and self.recorder_generation is not None:
                connection.execute(
                    """
                    UPDATE recorder_operational_state_v1
                    SET latest_inbox_acknowledgement_at_utc = ?,
                        inbox_backlog = ?,
                        oldest_unacknowledged_at_utc = ?,
                        updated_at_utc = ?
                    WHERE run_id = ? AND recorder_generation = ?
                    """,
                    (
                        observed.isoformat(),
                        int(accounting["backlog"]),
                        accounting["oldest"],
                        observed.isoformat(),
                        self.run_id,
                        self.recorder_generation,
                    ),
                )
            connection.commit()
        return acknowledged

    def accounting(self) -> InboxAccounting:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM callback_inbox_v1
                WHERE admission_run_id IS ?
                GROUP BY status
                """,
                (self.run_id,),
            ).fetchall()
            sequence = connection.execute(
                """
                SELECT
                    COALESCE(MAX(source_sequence), 0) AS highest,
                    COALESCE(
                        MIN(
                            CASE WHEN status NOT IN ('acknowledged', 'diagnostic')
                                 THEN source_sequence END
                        ) - 1,
                        MAX(source_sequence),
                        0
                    ) AS highest_ack,
                    MIN(
                        CASE WHEN status IN (
                            'provider_pending', 'pending', 'leased', 'quarantined'
                        )
                             THEN received_utc END
                    ) AS oldest
                FROM callback_inbox_v1
                WHERE admission_run_id IS ?
                """,
                (self.run_id,),
            ).fetchone()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        oldest = sequence["oldest"]
        return InboxAccounting(
            admitted=sum(counts.values()),
            pending=(
                counts.get(InboxStatus.PROVIDER_PENDING.value, 0)
                + counts.get(InboxStatus.PENDING.value, 0)
            ),
            leased=counts.get(InboxStatus.LEASED.value, 0),
            acknowledged=counts.get(InboxStatus.ACKNOWLEDGED.value, 0),
            quarantined=counts.get(InboxStatus.QUARANTINED.value, 0),
            diagnostic=counts.get(InboxStatus.DIAGNOSTIC.value, 0),
            highest_source_sequence=int(sequence["highest"]),
            highest_acknowledged_sequence=int(sequence["highest_ack"]),
            oldest_unacknowledged_at_utc=(
                None if oldest is None else datetime.fromisoformat(str(oldest))
            ),
        )

    def compact_acknowledged(
        self,
        *,
        before: datetime,
        retention_policy_enabled: bool,
        limit: int = 4_096,
    ) -> int:
        """Compact terminal payloads while retaining identity and classification."""

        if not retention_policy_enabled:
            raise ValueError("callback inbox compaction requires an explicit retention policy")
        if limit <= 0:
            raise ValueError("callback inbox compaction limit must be positive")
        cutoff = _utc(before, label="callback retention cutoff")
        with self._connect() as connection:
            self._begin_immediate(connection)
            rows = connection.execute(
                """
                SELECT inbox.inbox_event_id, inbox.original_payload_json
                FROM callback_inbox_v1 AS inbox
                JOIN callback_raw_materialization_v1 AS materialization
                  ON materialization.inbox_event_id = inbox.inbox_event_id
                JOIN callback_processing_commit_v1 AS processing
                  ON processing.inbox_event_id = inbox.inbox_event_id
                 AND processing.raw_partition_hashes_json =
                     materialization.raw_partition_hashes_json
                WHERE inbox.admission_run_id IS ?
                  AND inbox.status = 'acknowledged'
                  AND inbox.acknowledgement_timestamp_utc < ?
                ORDER BY inbox.source_sequence
                LIMIT ?
                """,
                (self.run_id, cutoff.isoformat(), limit),
            ).fetchall()
            compacted_at = datetime.now(UTC).isoformat()
            compacted_count = 0
            for row in rows:
                try:
                    decoded = json.loads(str(row["original_payload_json"]))
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, dict) and "__retention_compacted_sha256__" in decoded:
                    continue
                payload_hash = hashlib.sha256(
                    str(row["original_payload_json"]).encode()
                ).hexdigest()
                connection.execute(
                    """
                    UPDATE callback_inbox_v1
                    SET original_payload_json = ?,
                        updated_at_utc = ?
                    WHERE inbox_event_id = ?
                      AND status = 'acknowledged'
                    """,
                    (
                        _encoded({"__retention_compacted_sha256__": payload_hash}),
                        compacted_at,
                        str(row["inbox_event_id"]),
                    ),
                )
                compacted_count += 1
            connection.commit()
        return compacted_count

    def latest_source_sequence(self) -> int | None:
        """Return the latest sequence admitted for this run, if any."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT MAX(source_sequence) AS latest
                FROM callback_inbox_v1
                WHERE admission_run_id IS ?
                """,
                (self.run_id,),
            ).fetchone()
        assert row is not None
        return None if row["latest"] is None else int(row["latest"])

    def record_tombstone(
        self,
        *,
        request_id: int,
        connection_generation: int,
        subscription_owner: str | None,
        symbol: str | None,
        cancellation_reason: str,
        cancelled_at: datetime,
        ttl: timedelta,
    ) -> RequestTombstone:
        if ttl <= timedelta(0):
            raise ValueError("request tombstone TTL must be positive")
        observed = _utc(cancelled_at, label="request cancellation timestamp")
        expires = observed + ttl
        reason = cancellation_reason.strip()
        if not reason:
            raise ValueError("request cancellation reason is required")
        with self._connect() as connection:
            self._begin_immediate(connection)
            connection.execute(
                "DELETE FROM callback_request_tombstone_v1 WHERE expires_at_utc <= ?",
                (observed.isoformat(),),
            )
            connection.execute(
                """
                INSERT INTO callback_request_tombstone_v1(
                    request_id, connection_generation, subscription_owner,
                    symbol, cancellation_reason, cancelled_at_utc,
                    expires_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id, connection_generation) DO UPDATE SET
                    subscription_owner = excluded.subscription_owner,
                    symbol = excluded.symbol,
                    cancellation_reason = excluded.cancellation_reason,
                    cancelled_at_utc = excluded.cancelled_at_utc,
                    expires_at_utc = excluded.expires_at_utc
                """,
                (
                    request_id,
                    connection_generation,
                    subscription_owner,
                    symbol,
                    reason,
                    observed.isoformat(),
                    expires.isoformat(),
                ),
            )
            connection.execute(
                """
                DELETE FROM callback_request_tombstone_v1
                WHERE (request_id, connection_generation) IN (
                    SELECT request_id, connection_generation
                    FROM callback_request_tombstone_v1
                    ORDER BY cancelled_at_utc DESC, request_id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_tombstones,),
            )
            connection.commit()
        return RequestTombstone(
            request_id=request_id,
            connection_generation=connection_generation,
            subscription_owner=subscription_owner,
            symbol=symbol,
            cancellation_reason=reason,
            cancelled_at_utc=observed,
            expires_at_utc=expires,
        )

    def tombstone(
        self,
        *,
        request_id: int,
        now: datetime,
    ) -> RequestTombstone | None:
        observed = _utc(now, label="request tombstone lookup timestamp")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM callback_request_tombstone_v1
                WHERE request_id = ? AND expires_at_utc > ?
                ORDER BY connection_generation DESC
                LIMIT 1
                """,
                (request_id, observed.isoformat()),
            ).fetchone()
        if row is None:
            return None
        return RequestTombstone(
            request_id=int(row["request_id"]),
            connection_generation=int(row["connection_generation"]),
            subscription_owner=(
                None if row["subscription_owner"] is None else str(row["subscription_owner"])
            ),
            symbol=None if row["symbol"] is None else str(row["symbol"]),
            cancellation_reason=str(row["cancellation_reason"]),
            cancelled_at_utc=datetime.fromisoformat(str(row["cancelled_at_utc"])),
            expires_at_utc=datetime.fromisoformat(str(row["expires_at_utc"])),
        )

    def record_incident(
        self,
        *,
        stable_error_code: str,
        component: str,
        severity: str,
        occurred_at: datetime,
        error_class: str,
        evidence_loss_possible: bool,
        callback_kind: str | None = None,
        request_id: int | None = None,
        source_sequence: int | None = None,
        connection_generation: int | None = None,
        subscription_owner: str | None = None,
        symbol: str | None = None,
        details: Mapping[str, object] | None = None,
        incident_id: str | None = None,
    ) -> str:
        observed = _utc(occurred_at, label="operational incident timestamp")
        details_json = _encoded(dict(details or {}))
        identity = (
            incident_id
            or hashlib.sha256(
                "|".join(
                    (
                        stable_error_code,
                        component,
                        observed.isoformat(),
                        str(request_id),
                        str(source_sequence),
                        details_json,
                    )
                ).encode()
            ).hexdigest()
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO operational_incident_v1(
                    incident_id, run_id, recorder_generation,
                    connection_generation, occurred_at_utc, component,
                    severity, stable_error_code, callback_kind, request_id,
                    source_sequence, subscription_owner, symbol, error_class,
                    evidence_loss_possible, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity,
                    self.run_id,
                    self.recorder_generation,
                    connection_generation,
                    observed.isoformat(),
                    component,
                    severity,
                    stable_error_code,
                    callback_kind,
                    request_id,
                    source_sequence,
                    subscription_owner,
                    symbol,
                    error_class,
                    int(evidence_loss_possible),
                    details_json,
                ),
            )
        return identity

    def latch_fatal(
        self,
        *,
        latch_kind: str,
        stable_error_code: str,
        occurred_at: datetime,
        error_class: str,
        evidence_loss_possible: bool,
        first_possibly_lost_source_sequence: int | None = None,
        callback_kind: str | None = None,
        request_id: int | None = None,
        connection_generation: int | None = None,
    ) -> str:
        if latch_kind not in {"ingestion", "storage"}:
            raise ValueError("fatal latch kind is invalid")
        observed = _utc(occurred_at, label="fatal latch timestamp")
        latch_id = hashlib.sha256(
            "|".join(
                (
                    str(self.run_id),
                    latch_kind,
                    stable_error_code,
                    observed.isoformat(),
                    str(first_possibly_lost_source_sequence),
                )
            ).encode()
        ).hexdigest()
        with self._connect() as connection:
            self._begin_immediate(connection)
            active = connection.execute(
                """
                SELECT latch_id
                FROM recorder_fatal_latch_v1
                WHERE run_id IS ? AND latch_kind = ? AND resolved_at_utc IS NULL
                """,
                (self.run_id, latch_kind),
            ).fetchone()
            if active is None:
                connection.execute(
                    """
                    INSERT INTO recorder_fatal_latch_v1(
                        latch_id, run_id, recorder_generation,
                        connection_generation, latch_kind, stable_error_code,
                        first_possibly_lost_source_sequence, callback_kind,
                        request_id, error_class, evidence_loss_possible,
                        latched_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        latch_id,
                        self.run_id,
                        self.recorder_generation,
                        connection_generation,
                        latch_kind,
                        stable_error_code,
                        first_possibly_lost_source_sequence,
                        callback_kind,
                        request_id,
                        error_class,
                        int(evidence_loss_possible),
                        observed.isoformat(),
                    ),
                )
            else:
                latch_id = str(active["latch_id"])
            if self.run_id is not None and self.recorder_generation is not None:
                state = "INGESTION_FATAL" if latch_kind == "ingestion" else "STORAGE_FATAL"
                column = (
                    "fatal_ingestion_code" if latch_kind == "ingestion" else "fatal_storage_code"
                )
                connection.execute(
                    f"""
                    UPDATE recorder_operational_state_v1
                    SET state = ?, state_reason_code = ?, {column} = ?,
                        scientific_recording_valid = 0, updated_at_utc = ?
                    WHERE run_id = ? AND recorder_generation = ?
                    """,
                    (
                        state,
                        stable_error_code,
                        stable_error_code,
                        observed.isoformat(),
                        self.run_id,
                        self.recorder_generation,
                    ),
                )
            connection.commit()
        return latch_id

    def has_active_fatal(self, latch_kind: str | None = None) -> bool:
        query = "SELECT 1 FROM recorder_fatal_latch_v1 WHERE resolved_at_utc IS NULL"
        arguments: tuple[object, ...] = ()
        if self.run_id is not None:
            query += " AND run_id = ?"
            arguments = (self.run_id,)
        if latch_kind is not None:
            query += " AND latch_kind = ?"
            arguments = (*arguments, latch_kind)
        query += " LIMIT 1"
        with self._connect() as connection:
            return connection.execute(query, arguments).fetchone() is not None

    def active_fatal(
        self,
        latch_kind: str,
    ) -> tuple[str, int | None] | None:
        if latch_kind not in {"ingestion", "storage"}:
            raise ValueError("fatal latch kind is invalid")
        query = """
            SELECT stable_error_code, first_possibly_lost_source_sequence
            FROM recorder_fatal_latch_v1
            WHERE resolved_at_utc IS NULL AND latch_kind = ?
        """
        arguments: tuple[object, ...] = (latch_kind,)
        if self.run_id is not None:
            query += " AND run_id = ?"
            arguments = (*arguments, self.run_id)
        query += " ORDER BY latched_at_utc, latch_id LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, arguments).fetchone()
        if row is None:
            return None
        return (
            str(row["stable_error_code"]),
            (
                None
                if row["first_possibly_lost_source_sequence"] is None
                else int(row["first_possibly_lost_source_sequence"])
            ),
        )


__all__ = [
    "CallbackClassification",
    "CallbackIdentityCollision",
    "CallbackInboxError",
    "CallbackInboxEvent",
    "CallbackInboxOverflow",
    "CallbackLeaseLost",
    "DurableCallbackInbox",
    "InboxAccounting",
    "InboxAdmissionResult",
    "InboxStatus",
    "RawMaterialization",
    "RequestTombstone",
]
