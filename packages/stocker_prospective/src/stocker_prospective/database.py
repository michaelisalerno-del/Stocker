"""SQLite WAL persistence and append-oriented prospective repositories."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.bars import CompletedBar
from stocker_prospective.bundle import SCIENTIFIC_CLASSIFICATION
from stocker_prospective.market_data import MarketDataBudgetSnapshot
from stocker_prospective.migration_order import migration_plan
from stocker_prospective.sqlite_coordination import CoordinatedSQLiteConnection

RECORDER_SQLITE_BUSY_TIMEOUT_MS = 30_000


class RecorderLeaseHeld(RuntimeError):
    """Another recorder currently owns the database lease."""


class SchemaVersionTooNew(RuntimeError):
    """The database contains a migration this application does not understand."""


class EvidenceMetadata(BaseModel):
    """Metadata flattened into every exported runtime evidence record."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    prospective_start_utc: datetime
    app_version: str
    git_commit: str
    model_artifact_id: str
    universe_id: str
    cohort: str
    source_timestamps: list[str]
    recorded_at_utc: datetime

    @field_validator("prospective_start_utc", "recorded_at_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evidence timestamps must be timezone-aware")
        return value.astimezone(UTC)


class ScoreInput(BaseModel):
    """Completed-bar score at the public eventisation seam."""

    model_config = ConfigDict(extra="forbid")

    metadata: EvidenceMetadata
    symbol: str
    bar_end_utc: datetime
    session_date: date
    feature_as_of_utc: datetime
    m0_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    m1_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    frozen_threshold: float = Field(ge=0.0, le=1.0)
    feature_schema_hash: str
    eligibility: bool
    rejection_reason: str | None
    score_label: str


class StoredScore(BaseModel):
    id: int
    input: ScoreInput


class LeaseRecord(BaseModel):
    run_id: str
    owner_id: str
    acquired_at_utc: datetime
    heartbeat_at_utc: datetime
    generation: int
    recovered_stale_owner: bool
    previous_run_id: str | None = None
    previous_owner_id: str | None = None
    previous_heartbeat_at_utc: datetime | None = None


class UnderlyingContractInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: EvidenceMetadata
    symbol: str
    con_id: int | None
    exchange: str | None
    currency: str | None
    local_symbol: str | None
    qualification_status: str
    rejection_reason: str | None


class UnderlyingQuoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: EvidenceMetadata
    symbol: str
    con_id: int
    target_timestamp_utc: datetime
    actual_quote_timestamp_utc: datetime | None
    capture_lag_seconds: float | None
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    last: float | None
    last_size: float | None
    midpoint: float | None
    spread: float | None
    provider_timestamp_utc: datetime | None
    receive_timestamp_utc: datetime | None
    market_data_type: str | None
    freshness: str
    completeness: str
    capture_status: str
    missing_quote_reason: str | None


class SourceBarObservationInput(BaseModel):
    """One external bar retained only as prospective source-parity evidence."""

    model_config = ConfigDict(extra="forbid")

    metadata: EvidenceMetadata
    provider: str
    provider_record_id: str
    symbol: str
    session_date: date
    bar_start_utc: datetime
    bar_end_utc: datetime
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    activity_value: float | None
    activity_semantic_label: str
    source_timestamp_utc: datetime
    receive_timestamp_utc: datetime
    completeness: Literal["complete", "partial"]
    eligibility: Literal[False] = False
    rejection_reason: Literal["parallel_validation_only"] = "parallel_validation_only"


class ProspectiveRepository:
    """Single-writer SQLite repository with explicit migrations."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        busy_timeout_ms: int = RECORDER_SQLITE_BUSY_TIMEOUT_MS,
    ) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("SQLite busy timeout must be positive")
        self.database_path = Path(database_path)
        self.busy_timeout_ms = busy_timeout_ms
        self._anchor: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
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
        return connection

    def open_anchor(self) -> None:
        """Hold WAL coordination files for the recorder process lifetime."""

        if self._anchor is not None:
            return
        connection = self._connect()
        try:
            connection.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        except Exception:
            connection.close()
            raise
        self._anchor = connection

    def close_anchor(self) -> None:
        """Release the recorder's process-lifetime WAL anchor."""

        if self._anchor is None:
            return
        self._anchor.close()
        self._anchor = None

    def migrate(self) -> None:
        """Apply package migrations exactly once."""

        migration_root = Path(__file__).with_name("migrations")
        migration_paths = tuple(item.path for item in migration_plan(migration_root))
        supported = {path.name for path in migration_paths}
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at_utc TEXT NOT NULL
                )
                """
            )
            applied = {
                str(row["version"])
                for row in connection.execute("SELECT version FROM schema_migrations")
            }
            unsupported = tuple(sorted(applied - supported))
            if unsupported:
                raise SchemaVersionTooNew(
                    "blocked_schema_newer_than_supported: " + ",".join(unsupported)
                )
            for path in migration_paths:
                if path.name in applied:
                    continue
                version = path.name.replace("'", "''")
                applied_at = datetime.now(UTC).isoformat().replace("'", "''")
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    f"{path.read_text(encoding='utf-8')}\n"
                    "INSERT INTO schema_migrations(version, applied_at_utc) "
                    f"VALUES ('{version}', '{applied_at}');\n"
                    "COMMIT;\n"
                )

    @staticmethod
    def _validate_metadata(metadata: EvidenceMetadata) -> None:
        if metadata.recorded_at_utc < metadata.prospective_start_utc:
            raise ValueError("recorded_at_utc precedes configured prospective_start_utc")

    def create_run(
        self,
        metadata: EvidenceMetadata,
        *,
        mode: Literal["record_only", "shadow"] = "record_only",
    ) -> None:
        self._validate_metadata(metadata)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO prospective_run(
                    run_id, prospective_start_utc, app_version, git_commit,
                    model_artifact_id, universe_id, cohort, created_at_utc,
                    mode, scientific_classification
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.run_id,
                    metadata.prospective_start_utc.isoformat(),
                    metadata.app_version,
                    metadata.git_commit,
                    metadata.model_artifact_id,
                    metadata.universe_id,
                    metadata.cohort,
                    metadata.recorded_at_utc.isoformat(),
                    mode,
                    SCIENTIFIC_CLASSIFICATION,
                ),
            )
            existing = connection.execute(
                "SELECT * FROM prospective_run WHERE run_id = ?",
                (metadata.run_id,),
            ).fetchone()
            assert existing is not None
            expected = {
                "prospective_start_utc": metadata.prospective_start_utc.isoformat(),
                "app_version": metadata.app_version,
                "git_commit": metadata.git_commit,
                "model_artifact_id": metadata.model_artifact_id,
                "universe_id": metadata.universe_id,
                "cohort": metadata.cohort,
                "mode": mode,
                "scientific_classification": SCIENTIFIC_CLASSIFICATION,
            }
            if any(str(existing[name]) != value for name, value in expected.items()):
                raise ValueError(
                    "blocked_unsafe_runtime_configuration: prospective run identity mismatch"
                )

    def prospective_run_app_version(self, *, run_id: str) -> str | None:
        """Return the immutable first-activation application version, if present."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT app_version FROM prospective_run WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else str(row["app_version"])

    def prospective_run_ids(self) -> tuple[str, ...]:
        """Return persisted run identities eligible for activation-hash reconstruction."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id
                FROM prospective_run
                ORDER BY created_at_utc, run_id
                """
            ).fetchall()
        return tuple(str(row["run_id"]) for row in rows)

    def register_universe_membership(
        self,
        metadata: EvidenceMetadata,
        *,
        symbols: tuple[str, ...],
        operational_status: str | None = None,
        operational_status_by_symbol: Mapping[str, tuple[str, str | None]] | None = None,
    ) -> None:
        """Register every frozen member exactly once; never silently drop one."""

        self._validate_metadata(metadata)
        if len(symbols) != 20 or len(set(symbols)) != 20:
            raise ValueError("blocked_frozen_universe_mismatch")
        if operational_status_by_symbol is not None:
            if set(operational_status_by_symbol) != set(symbols):
                raise ValueError("blocked_frozen_universe_mismatch")
        elif operational_status is None:
            raise ValueError("an explicit operational status is required for every symbol")
        with self._connect() as connection:
            for symbol in symbols:
                status, rejection_reason = (
                    operational_status_by_symbol[symbol]
                    if operational_status_by_symbol is not None
                    else (operational_status, None)
                )
                assert status is not None
                connection.execute(
                    """
                    INSERT OR IGNORE INTO universe_membership(
                        run_id, universe_id, cohort, symbol, operational_status,
                        rejection_reason, recorded_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metadata.run_id,
                        metadata.universe_id,
                        metadata.cohort,
                        symbol,
                        status,
                        rejection_reason,
                        metadata.recorded_at_utc.isoformat(),
                    ),
                )
            rows = connection.execute(
                """
                SELECT symbol FROM universe_membership
                WHERE run_id = ? AND cohort = ?
                """,
                (metadata.run_id, metadata.cohort),
            ).fetchall()
            if {str(row["symbol"]) for row in rows} != set(symbols):
                raise ValueError("blocked_frozen_universe_mismatch")

    def record_underlying_contract(self, item: UnderlyingContractInput) -> int:
        self._validate_metadata(item.metadata)
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM underlying_contract
                WHERE run_id = ? AND symbol = ?
                  AND ((con_id IS NULL AND ? IS NULL) OR con_id = ?)
                ORDER BY id LIMIT 1
                """,
                (
                    item.metadata.run_id,
                    item.symbol,
                    item.con_id,
                    item.con_id,
                ),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            envelope_id = self._insert_envelope(connection, item.metadata)
            cursor = connection.execute(
                """
                INSERT INTO underlying_contract(
                    envelope_id, run_id, symbol, con_id, exchange, currency,
                    local_symbol, qualification_status, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    item.metadata.run_id,
                    item.symbol,
                    item.con_id,
                    item.exchange,
                    item.currency,
                    item.local_symbol,
                    item.qualification_status,
                    item.rejection_reason,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_underlying_bar(
        self,
        metadata: EvidenceMetadata,
        bar: CompletedBar,
        *,
        eligibility: bool,
        rejection_reason: str,
    ) -> int:
        self._validate_metadata(metadata)
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM underlying_bar
                WHERE run_id = ? AND symbol = ? AND bar_end_utc = ?
                """,
                (metadata.run_id, bar.symbol, bar.bar_end_utc.isoformat()),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            envelope_id = self._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO underlying_bar(
                    envelope_id, run_id, symbol, con_id, bar_start_utc, bar_end_utc,
                    session_date, open, high, low, close, activity_value,
                    activity_semantic_label, bar_source, source_timestamp_utc,
                    receive_timestamp_utc, completeness, feature_as_of_utc,
                    m0_probability, m1_probability, frozen_threshold, model_bundle_id,
                    feature_schema_hash, eligibility, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL,
                          NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    bar.symbol,
                    bar.permanent_contract_id,
                    bar.bar_start_utc.isoformat(),
                    bar.bar_end_utc.isoformat(),
                    bar.session_date.isoformat(),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.activity_value,
                    bar.activity_semantic_label,
                    bar.bar_source,
                    bar.source_timestamp_utc.isoformat(),
                    bar.receive_timestamp_utc.isoformat(),
                    "complete" if bar.complete else "partial",
                    bar.feature_as_of_utc.isoformat(),
                    metadata.model_artifact_id,
                    None,
                    int(eligibility),
                    rejection_reason,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_underlying_quote(self, item: UnderlyingQuoteInput) -> int:
        self._validate_metadata(item.metadata)
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM underlying_quote
                WHERE run_id = ? AND signal_episode_id IS NULL
                  AND symbol = ? AND target_timestamp_utc = ?
                """,
                (
                    item.metadata.run_id,
                    item.symbol,
                    item.target_timestamp_utc.isoformat(),
                ),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            envelope_id = self._insert_envelope(connection, item.metadata)
            cursor = connection.execute(
                """
                INSERT INTO underlying_quote(
                    envelope_id, run_id, signal_episode_id, symbol, con_id,
                    target_timestamp_utc,
                    actual_quote_timestamp_utc, capture_lag_seconds, bid, ask,
                    bid_size, ask_size, last, last_size, midpoint, spread,
                    provider_timestamp_utc, receive_timestamp_utc, market_data_type,
                    freshness, completeness, capture_status, missing_quote_reason
                ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    item.metadata.run_id,
                    item.symbol,
                    item.con_id,
                    item.target_timestamp_utc.isoformat(),
                    (
                        None
                        if item.actual_quote_timestamp_utc is None
                        else item.actual_quote_timestamp_utc.isoformat()
                    ),
                    item.capture_lag_seconds,
                    item.bid,
                    item.ask,
                    item.bid_size,
                    item.ask_size,
                    item.last,
                    item.last_size,
                    item.midpoint,
                    item.spread,
                    (
                        None
                        if item.provider_timestamp_utc is None
                        else item.provider_timestamp_utc.isoformat()
                    ),
                    (
                        None
                        if item.receive_timestamp_utc is None
                        else item.receive_timestamp_utc.isoformat()
                    ),
                    item.market_data_type,
                    item.freshness,
                    item.completeness,
                    item.capture_status,
                    item.missing_quote_reason,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_source_bar_observation(self, item: SourceBarObservationInput) -> int:
        """Append one never-score external source observation idempotently."""

        self._validate_metadata(item.metadata)
        payload = {
            key: value
            for key, value in item.model_dump(mode="json").items()
            if key not in {"metadata", "receive_timestamp_utc"}
        }
        record_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id, record_hash FROM source_bar_observation
                WHERE run_id = ? AND provider = ? AND provider_record_id = ?
                """,
                (
                    item.metadata.run_id,
                    item.provider,
                    item.provider_record_id,
                ),
            ).fetchone()
            if existing is not None:
                if str(existing["record_hash"]) != record_hash:
                    raise ValueError("parallel source provider identity collision")
                return int(existing["id"])
            envelope_id = self._insert_envelope(connection, item.metadata)
            cursor = connection.execute(
                """
                INSERT INTO source_bar_observation(
                    envelope_id, run_id, provider, provider_record_id, symbol,
                    session_date, bar_start_utc, bar_end_utc, open, high, low, close,
                    activity_value, activity_semantic_label, source_timestamp_utc,
                    receive_timestamp_utc, completeness, eligibility,
                    rejection_reason, record_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    envelope_id,
                    item.metadata.run_id,
                    item.provider,
                    item.provider_record_id,
                    item.symbol,
                    item.session_date.isoformat(),
                    item.bar_start_utc.astimezone(UTC).isoformat(),
                    item.bar_end_utc.astimezone(UTC).isoformat(),
                    item.open,
                    item.high,
                    item.low,
                    item.close,
                    item.activity_value,
                    item.activity_semantic_label,
                    item.source_timestamp_utc.astimezone(UTC).isoformat(),
                    item.receive_timestamp_utc.astimezone(UTC).isoformat(),
                    item.completeness,
                    item.rejection_reason,
                    record_hash,
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def source_capture_completed(
        self,
        *,
        run_id: str,
        provider: str,
        session_date: date,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM source_capture_completion
                WHERE run_id = ? AND provider = ? AND session_date = ?
                """,
                (run_id, provider, session_date.isoformat()),
            ).fetchone()
        return row is not None

    def record_source_capture_completion(
        self,
        metadata: EvidenceMetadata,
        *,
        provider: str,
        session_date: date,
        status: Literal["complete", "partial"],
        requested_symbol_count: int,
        captured_symbol_count: int,
        bar_count: int,
        missing_symbols: tuple[str, ...],
    ) -> int:
        """Close one session once; a partial capture remains partial."""

        self._validate_metadata(metadata)
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM source_capture_completion
                WHERE run_id = ? AND provider = ? AND session_date = ?
                """,
                (metadata.run_id, provider, session_date.isoformat()),
            ).fetchone()
            if existing is not None:
                return int(existing["id"])
            envelope_id = self._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO source_capture_completion(
                    envelope_id, run_id, provider, session_date, status,
                    requested_symbol_count, captured_symbol_count, bar_count,
                    missing_symbols_json, completed_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    provider,
                    session_date.isoformat(),
                    status,
                    requested_symbol_count,
                    captured_symbol_count,
                    bar_count,
                    json.dumps(missing_symbols, separators=(",", ":")),
                    metadata.recorded_at_utc.isoformat(),
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_data_health_event(
        self,
        metadata: EvidenceMetadata,
        *,
        severity: str,
        blocker_code: str | None,
        component: str,
        message: str,
        details: dict[str, object],
    ) -> int:
        self._validate_metadata(metadata)
        with self._connect() as connection:
            envelope_id = self._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO data_health_event(
                    envelope_id, run_id, severity, blocker_code, component,
                    message, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    severity,
                    blocker_code,
                    component,
                    message,
                    json.dumps(details, sort_keys=True, separators=(",", ":")),
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_ibkr_connection_event(
        self,
        metadata: EvidenceMetadata,
        *,
        state: str,
        error_code: int | None,
        message: str,
        data_maintained: bool | None,
        reconnect_attempt: int | None,
        details: dict[str, object],
    ) -> int:
        self._validate_metadata(metadata)
        with self._connect() as connection:
            envelope_id = self._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO ibkr_connection_event(
                    envelope_id, run_id, state, error_code, message,
                    data_maintained, reconnect_attempt, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    state,
                    error_code,
                    message,
                    None if data_maintained is None else int(data_maintained),
                    reconnect_attempt,
                    json.dumps(details, sort_keys=True, separators=(",", ":")),
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_market_data_budget_event(
        self,
        metadata: EvidenceMetadata,
        snapshot: MarketDataBudgetSnapshot,
    ) -> int:
        self._validate_metadata(metadata)
        with self._connect() as connection:
            envelope_id = self._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO market_data_budget_event(
                    envelope_id, run_id, line_limit, reserved_headroom,
                    usable_lines, active_lines, pending_requests,
                    awaiting_cancellation, current_request_rate,
                    waiting_signals, rejected_signals, recorded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    snapshot.line_limit,
                    snapshot.reserved_headroom,
                    snapshot.usable_lines,
                    snapshot.active_lines,
                    snapshot.pending_requests,
                    snapshot.awaiting_cancellation,
                    snapshot.current_request_rate,
                    snapshot.waiting_signals,
                    snapshot.rejected_signals,
                    metadata.recorded_at_utc.isoformat(),
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def record_audit_event(
        self,
        metadata: EvidenceMetadata,
        *,
        event_type: str,
        actor: str,
        message: str,
        payload: dict[str, object],
    ) -> int:
        self._validate_metadata(metadata)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM audit_event WHERE run_id = ?",
                (metadata.run_id,),
            ).fetchone()
            assert row is not None
            envelope_id = self._insert_envelope(connection, metadata)
            cursor = connection.execute(
                """
                INSERT INTO audit_event(
                    envelope_id, run_id, sequence, event_type, actor, message, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    int(row["next"]),
                    event_type,
                    actor,
                    message,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def _insert_envelope(
        self,
        connection: sqlite3.Connection,
        metadata: EvidenceMetadata,
    ) -> int:
        self._validate_metadata(metadata)
        cursor = connection.execute(
            """
            INSERT INTO evidence_envelope(
                run_id, prospective_start_utc, app_version, git_commit,
                model_artifact_id, universe_id, cohort, source_timestamps_json,
                recorded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.run_id,
                metadata.prospective_start_utc.isoformat(),
                metadata.app_version,
                metadata.git_commit,
                metadata.model_artifact_id,
                metadata.universe_id,
                metadata.cohort,
                json.dumps(metadata.source_timestamps, separators=(",", ":")),
                metadata.recorded_at_utc.isoformat(),
            ),
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    def acquire_recorder_lease(
        self,
        *,
        run_id: str,
        owner_id: str,
        now: datetime,
        stale_after: timedelta,
    ) -> LeaseRecord:
        """Acquire the singleton recorder lease or reject a concurrent owner."""

        now = now.astimezone(UTC)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM recorder_lease WHERE lease_key = 'prospective_recorder'"
            ).fetchone()
            recovered = False
            previous_run_id: str | None = None
            previous_owner_id: str | None = None
            previous_heartbeat_at_utc: datetime | None = None
            if row is not None:
                heartbeat = datetime.fromisoformat(str(row["heartbeat_at_utc"]))
                same_owner = row["owner_id"] == owner_id and row["run_id"] == run_id
                stale = heartbeat <= now - stale_after
                if not same_owner and not stale:
                    raise RecorderLeaseHeld(
                        f"blocked_recorder_lease_held: owner={row['owner_id']} run={row['run_id']}"
                    )
                recovered = not same_owner and stale
                acquired = now.isoformat() if recovered else str(row["acquired_at_utc"])
                if recovered:
                    previous_run_id = str(row["run_id"])
                    previous_owner_id = str(row["owner_id"])
                    previous_heartbeat_at_utc = heartbeat
                    historical_generation = connection.execute(
                        """
                        SELECT COALESCE(MAX(recorder_generation), 0)
                        FROM recorder_generation_v1
                        WHERE run_id = ?
                        """,
                        (run_id,),
                    ).fetchone()[0]
                    generation = (
                        max(
                            int(row["generation"]),
                            int(historical_generation),
                        )
                        + 1
                    )
                else:
                    generation = int(row["generation"])
                connection.execute(
                    """
                    UPDATE recorder_lease
                    SET run_id = ?, owner_id = ?, acquired_at_utc = ?,
                        heartbeat_at_utc = ?, generation = ?, recovered_stale_owner = ?
                    WHERE lease_key = 'prospective_recorder'
                    """,
                    (
                        run_id,
                        owner_id,
                        acquired,
                        now.isoformat(),
                        generation,
                        int(recovered),
                    ),
                )
            else:
                acquired = now.isoformat()
                historical_generation = connection.execute(
                    """
                    SELECT COALESCE(MAX(recorder_generation), 0)
                    FROM recorder_generation_v1
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()[0]
                generation = int(historical_generation) + 1
                connection.execute(
                    """
                    INSERT INTO recorder_lease(
                        lease_key, run_id, owner_id, acquired_at_utc,
                        heartbeat_at_utc, generation, recovered_stale_owner
                    ) VALUES ('prospective_recorder', ?, ?, ?, ?, ?, 0)
                    """,
                    (run_id, owner_id, acquired, now.isoformat(), generation),
                )
            connection.commit()
            return LeaseRecord(
                run_id=run_id,
                owner_id=owner_id,
                acquired_at_utc=datetime.fromisoformat(acquired),
                heartbeat_at_utc=now,
                generation=generation,
                recovered_stale_owner=recovered,
                previous_run_id=previous_run_id,
                previous_owner_id=previous_owner_id,
                previous_heartbeat_at_utc=previous_heartbeat_at_utc,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat_recorder_lease(
        self,
        *,
        run_id: str,
        owner_id: str,
        now: datetime,
    ) -> LeaseRecord:
        now = now.astimezone(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE recorder_lease SET heartbeat_at_utc = ?
                WHERE lease_key = 'prospective_recorder' AND run_id = ? AND owner_id = ?
                """,
                (now.isoformat(), run_id, owner_id),
            )
            if cursor.rowcount != 1:
                raise RecorderLeaseHeld("blocked_recorder_lease_held: lease ownership changed")
            row = connection.execute(
                "SELECT * FROM recorder_lease WHERE lease_key = 'prospective_recorder'"
            ).fetchone()
        assert row is not None
        return LeaseRecord(
            run_id=str(row["run_id"]),
            owner_id=str(row["owner_id"]),
            acquired_at_utc=datetime.fromisoformat(str(row["acquired_at_utc"])),
            heartbeat_at_utc=now,
            generation=int(row["generation"]),
            recovered_stale_owner=bool(row["recovered_stale_owner"]),
        )

    def release_recorder_lease(self, *, run_id: str, owner_id: str) -> bool:
        """Release only the exact process-owned lease during graceful shutdown."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM recorder_lease
                WHERE lease_key = 'prospective_recorder' AND run_id = ? AND owner_id = ?
                """,
                (run_id, owner_id),
            )
        return cursor.rowcount == 1

    def record_score(self, score: ScoreInput) -> StoredScore:
        """Idempotently append one model score."""

        self._validate_metadata(score.metadata)
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id FROM model_score
                WHERE run_id = ? AND cohort = ? AND symbol = ?
                  AND model_bundle_id = ? AND bar_end_utc = ?
                """,
                (
                    score.metadata.run_id,
                    score.metadata.cohort,
                    score.symbol,
                    score.metadata.model_artifact_id,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                ),
            ).fetchone()
            if existing is not None:
                return StoredScore(id=int(existing["id"]), input=score)
            envelope_id = self._insert_envelope(connection, score.metadata)
            cursor = connection.execute(
                """
                INSERT INTO model_score(
                    envelope_id, run_id, cohort, symbol, bar_end_utc, session_date,
                    feature_as_of_utc, m0_probability, m1_probability, frozen_threshold,
                    model_bundle_id, feature_schema_hash, eligibility, rejection_reason,
                    score_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    score.metadata.run_id,
                    score.metadata.cohort,
                    score.symbol,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                    score.session_date.isoformat(),
                    score.feature_as_of_utc.astimezone(UTC).isoformat(),
                    score.m0_probability,
                    score.m1_probability,
                    score.frozen_threshold,
                    score.metadata.model_artifact_id,
                    score.feature_schema_hash,
                    int(score.eligibility),
                    score.rejection_reason,
                    score.score_label,
                ),
            )
            assert cursor.lastrowid is not None
            return StoredScore(id=int(cursor.lastrowid), input=score)

    def previous_eligible_score(self, score: ScoreInput) -> sqlite3.Row | None:
        with self._connect() as connection:
            row: sqlite3.Row | None = connection.execute(
                """
                SELECT * FROM model_score
                WHERE run_id = ? AND cohort = ? AND symbol = ? AND model_bundle_id = ?
                  AND eligibility = 1 AND bar_end_utc < ?
                ORDER BY bar_end_utc DESC LIMIT 1
                """,
                (
                    score.metadata.run_id,
                    score.metadata.cohort,
                    score.symbol,
                    score.metadata.model_artifact_id,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                ),
            ).fetchone()
        return row

    def latest_signal_episode(self, score: ScoreInput) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM signal_episode
                WHERE run_id = ? AND cohort = ? AND symbol = ? AND model_bundle_id = ?
                  AND crossing_timestamp_utc <= ?
                ORDER BY crossing_timestamp_utc DESC LIMIT 1
                """,
                (
                    score.metadata.run_id,
                    score.metadata.cohort,
                    score.symbol,
                    score.metadata.model_artifact_id,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                ),
            ).fetchone()
        return None if row is None else str(row["id"])

    def create_signal_episode(self, stored: StoredScore) -> str:
        score = stored.input
        crossing = score.bar_end_utc.astimezone(UTC).isoformat()
        raw_key = "|".join(
            (
                score.metadata.run_id,
                score.metadata.cohort,
                score.symbol,
                score.metadata.model_artifact_id,
                crossing,
            )
        )
        idempotency_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        episode_id = f"sig-{idempotency_key[:24]}"
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT id FROM signal_episode WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is None:
                envelope_id = self._insert_envelope(connection, score.metadata)
                connection.execute(
                    """
                    INSERT INTO signal_episode(
                        id, envelope_id, run_id, cohort, symbol, model_bundle_id,
                        crossing_timestamp_utc, idempotency_key, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    """,
                    (
                        episode_id,
                        envelope_id,
                        score.metadata.run_id,
                        score.metadata.cohort,
                        score.symbol,
                        score.metadata.model_artifact_id,
                        crossing,
                        idempotency_key,
                    ),
                )
            else:
                episode_id = str(existing["id"])
        self.add_signal_checkpoint(episode_id, stored)
        return episode_id

    def add_signal_checkpoint(self, episode_id: str, stored: StoredScore) -> None:
        score = stored.input
        assert score.m1_probability is not None
        with self._connect() as connection:
            exists = connection.execute(
                """
                SELECT 1 FROM signal_checkpoint
                WHERE signal_episode_id = ? AND model_score_id = ?
                """,
                (episode_id, stored.id),
            ).fetchone()
            if exists is not None:
                return
            envelope_id = self._insert_envelope(connection, score.metadata)
            connection.execute(
                """
                INSERT INTO signal_checkpoint(
                    envelope_id, run_id, signal_episode_id, model_score_id,
                    checkpoint_timestamp_utc, m1_probability, frozen_threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    score.metadata.run_id,
                    episode_id,
                    stored.id,
                    score.bar_end_utc.astimezone(UTC).isoformat(),
                    score.m1_probability,
                    score.frozen_threshold,
                ),
            )

    def record_eventization(
        self,
        stored: StoredScore,
        *,
        status: str,
        episode_id: str | None,
    ) -> None:
        """Append the eventisation decision, including startup-above state."""

        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM signal_eventization WHERE model_score_id = ?",
                (stored.id,),
            ).fetchone()
            if exists is not None:
                return
            envelope_id = self._insert_envelope(connection, stored.input.metadata)
            connection.execute(
                """
                INSERT INTO signal_eventization(
                    envelope_id, run_id, model_score_id, eventization_status,
                    signal_episode_id, recorded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    stored.input.metadata.run_id,
                    stored.id,
                    status,
                    episode_id,
                    stored.input.metadata.recorded_at_utc.isoformat(),
                ),
            )

    def count(self, table: str) -> int:
        allowed = {
            "signal_episode",
            "signal_checkpoint",
            "model_score",
            "evidence_envelope",
            "signal_eventization",
            "source_bar_observation",
        }
        if table not in allowed:
            raise ValueError("unsupported count table")
        with self._connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        assert row is not None
        return int(row["count"])
