"""Authoritative recorder-state projection and append-only operational evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecorderOperationalState(StrEnum):
    INACTIVE = "INACTIVE"
    STARTING = "STARTING"
    WAITING_FOR_PROSPECTIVE_START = "WAITING_FOR_PROSPECTIVE_START"
    MARKET_CLOSED = "MARKET_CLOSED"
    RECORDING_HEALTHY = "RECORDING_HEALTHY"
    RECORDING_DEGRADED = "RECORDING_DEGRADED"
    RECONNECTING = "RECONNECTING"
    STALE_HEARTBEAT = "STALE_HEARTBEAT"
    INGESTION_FATAL = "INGESTION_FATAL"
    STORAGE_FATAL = "STORAGE_FATAL"
    SCIENTIFICALLY_BLOCKED = "SCIENTIFICALLY_BLOCKED"
    STOPPING = "STOPPING"
    STOPPED_CLEANLY = "STOPPED_CLEANLY"


class OperationalThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lease_stale_after: timedelta = timedelta(seconds=30)
    process_heartbeat_stale_after: timedelta = timedelta(seconds=30)
    callback_heartbeat_stale_after: timedelta = timedelta(seconds=30)
    raw_storage_heartbeat_stale_after: timedelta = timedelta(seconds=60)
    acknowledgement_stale_after: timedelta = timedelta(seconds=30)
    maximum_inbox_backlog: int = Field(default=5_000, ge=0)
    maximum_oldest_unacknowledged_age: timedelta = timedelta(seconds=60)


class RecorderStateSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    recorder_generation: int = Field(gt=0)
    owner_id: str
    stored_state: RecorderOperationalState
    process_heartbeat_at_utc: datetime | None
    latest_callback_received_at_utc: datetime | None
    latest_callback_durably_admitted_at_utc: datetime | None
    latest_raw_partition_committed_at_utc: datetime | None
    latest_inbox_acknowledgement_at_utc: datetime | None
    latest_completed_five_minute_bar_at_utc: datetime | None
    latest_successful_checkpoint_at_utc: datetime | None
    inbox_backlog: int = Field(ge=0)
    oldest_unacknowledged_at_utc: datetime | None
    market_session_open: bool
    callbacks_expected: bool
    ibkr_connection_state: str | None
    required_market_data_mode: str | None
    observed_market_data_mode: str | None
    scientific_prerequisites_valid: bool
    expected_artifact_count: int = Field(ge=0)
    frozen_artifacts_verified: bool
    unresolved_required_gap_count: int = Field(ge=0)
    fatal_ingestion_code: str | None
    fatal_storage_code: str | None
    broker_state_mutation_count: int = Field(ge=0)
    lease_owner_id: str | None
    lease_run_id: str | None
    lease_generation: int | None
    lease_heartbeat_at_utc: datetime | None

    @field_validator(
        "process_heartbeat_at_utc",
        "latest_callback_received_at_utc",
        "latest_callback_durably_admitted_at_utc",
        "latest_raw_partition_committed_at_utc",
        "latest_inbox_acknowledgement_at_utc",
        "latest_completed_five_minute_bar_at_utc",
        "latest_successful_checkpoint_at_utc",
        "oldest_unacknowledged_at_utc",
        "lease_heartbeat_at_utc",
    )
    @classmethod
    def _timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("operational timestamps must be timezone-aware")
        return value.astimezone(UTC)


class OperationalStateProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: RecorderOperationalState
    reason_code: str
    healthy: bool
    scientific_recording_valid: bool
    evaluated_at_utc: datetime
    run_id: str | None
    recorder_generation: int | None
    owner_id: str | None
    timestamps: dict[str, str | None]
    inbox: dict[str, int | float | str | None]
    conditions: dict[str, bool | int | str | None]


class GapIncident(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gap_id: str
    run_id: str
    recorder_generation: int = Field(gt=0)
    symbol: str
    stream_kind: str
    request_id: int | None
    connection_generation: int = Field(ge=0)
    start_timestamp_utc: datetime
    end_timestamp_utc: datetime | None = None
    detection_timestamp_utc: datetime
    cause_code: str
    severity: Literal["optional", "degraded", "scientific"]
    recoverability: Literal["recoverable", "unrecoverable", "unknown"]
    backfill_attempted: bool = False
    backfill_result: str | None = None
    affected_first_source_sequence: int | None = None
    affected_last_source_sequence: int | None = None
    affected_episode_ids: tuple[str, ...] = ()
    resolution_timestamp_utc: datetime | None = None
    resolution_evidence: str | None = None


class RuntimeArtifactVerification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verification_id: str
    run_id: str
    recorder_generation: int = Field(gt=0)
    artifact_bundle_id: str
    artifact_name: str
    expected_hash: str
    observed_hash: str | None
    feature_contract_version: str
    activation_receipt_identity: str
    found: bool
    loaded: bool
    schema_validated: bool
    hash_verified: bool
    contract_compatible: bool
    used_by_active_generation: bool
    load_timestamp_utc: datetime
    verification_result: Literal["verified", "blocked"]
    blocker: str | None
    details: dict[str, Any] = Field(default_factory=dict)


def _utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _fresh(
    value: datetime | None,
    *,
    now: datetime,
    maximum_age: timedelta,
) -> bool:
    return value is not None and timedelta(0) <= now - value <= maximum_age


def evaluate_operational_state(
    signals: RecorderStateSignals,
    *,
    now: datetime,
    prospective_start_utc: datetime,
    thresholds: OperationalThresholds,
) -> OperationalStateProjection:
    """Derive every API/dashboard health label from one explicit rule set."""

    observed = _utc(now, label="operational evaluation timestamp")
    prospective_start = _utc(prospective_start_utc, label="prospective start")
    lease_is_current = (
        signals.lease_owner_id == signals.owner_id
        and signals.lease_run_id == signals.run_id
        and signals.lease_generation == signals.recorder_generation
    )
    lease_is_fresh = lease_is_current and _fresh(
        signals.lease_heartbeat_at_utc,
        now=observed,
        maximum_age=thresholds.lease_stale_after,
    )
    process_is_fresh = _fresh(
        signals.process_heartbeat_at_utc,
        now=observed,
        maximum_age=thresholds.process_heartbeat_stale_after,
    )
    callback_received_is_fresh = not signals.callbacks_expected or _fresh(
        signals.latest_callback_received_at_utc,
        now=observed,
        maximum_age=thresholds.callback_heartbeat_stale_after,
    )
    callback_admitted_is_fresh = not signals.callbacks_expected or _fresh(
        signals.latest_callback_durably_admitted_at_utc,
        now=observed,
        maximum_age=thresholds.callback_heartbeat_stale_after,
    )
    callback_is_fresh = callback_received_is_fresh and callback_admitted_is_fresh
    raw_is_fresh = not signals.callbacks_expected or _fresh(
        signals.latest_raw_partition_committed_at_utc,
        now=observed,
        maximum_age=thresholds.raw_storage_heartbeat_stale_after,
    )
    acknowledgement_is_fresh = signals.inbox_backlog == 0 or _fresh(
        signals.latest_inbox_acknowledgement_at_utc,
        now=observed,
        maximum_age=thresholds.acknowledgement_stale_after,
    )
    oldest_age = (
        None
        if signals.oldest_unacknowledged_at_utc is None
        else max(
            0.0,
            (observed - signals.oldest_unacknowledged_at_utc).total_seconds(),
        )
    )
    backlog_within_limit = signals.inbox_backlog <= thresholds.maximum_inbox_backlog and (
        oldest_age is None
        or oldest_age <= thresholds.maximum_oldest_unacknowledged_age.total_seconds()
    )
    expected_market_data_observed = (
        signals.required_market_data_mode is None
        or signals.observed_market_data_mode == signals.required_market_data_mode
    )
    expected_connection = signals.ibkr_connection_state in {
        "connected",
        "CONNECTED",
        "healthy",
    }

    if signals.fatal_storage_code is not None:
        state = RecorderOperationalState.STORAGE_FATAL
        reason = signals.fatal_storage_code
    elif signals.fatal_ingestion_code is not None:
        state = RecorderOperationalState.INGESTION_FATAL
        reason = signals.fatal_ingestion_code
    elif signals.broker_state_mutation_count != 0:
        state = RecorderOperationalState.SCIENTIFICALLY_BLOCKED
        reason = "BROKER_STATE_MUTATION_DETECTED"
    elif signals.stored_state is RecorderOperationalState.STOPPING:
        state = RecorderOperationalState.STOPPING
        reason = "RECORDER_STOPPING"
    elif signals.stored_state is RecorderOperationalState.STOPPED_CLEANLY:
        state = RecorderOperationalState.STOPPED_CLEANLY
        reason = "RECORDER_STOPPED_CLEANLY"
    elif not lease_is_fresh or not process_is_fresh:
        state = RecorderOperationalState.STALE_HEARTBEAT
        reason = "RECORDER_LEASE_OR_PROCESS_HEARTBEAT_STALE"
    elif observed < prospective_start:
        state = RecorderOperationalState.WAITING_FOR_PROSPECTIVE_START
        reason = "PROSPECTIVE_START_NOT_REACHED"
    elif not signals.scientific_prerequisites_valid:
        state = RecorderOperationalState.SCIENTIFICALLY_BLOCKED
        reason = "SCIENTIFIC_PREREQUISITES_INVALID"
    elif not signals.frozen_artifacts_verified:
        state = RecorderOperationalState.SCIENTIFICALLY_BLOCKED
        reason = "RUNTIME_ARTIFACTS_NOT_VERIFIED"
    elif signals.unresolved_required_gap_count:
        state = RecorderOperationalState.SCIENTIFICALLY_BLOCKED
        reason = "UNRESOLVED_REQUIRED_STREAM_GAP"
    elif not expected_market_data_observed:
        state = RecorderOperationalState.SCIENTIFICALLY_BLOCKED
        reason = "REQUIRED_MARKET_DATA_MODE_NOT_OBSERVED"
    elif not signals.market_session_open:
        state = RecorderOperationalState.MARKET_CLOSED
        reason = "MARKET_SESSION_CLOSED"
    elif signals.ibkr_connection_state in {
        "connecting",
        "CONNECTING",
        "disconnected",
        "DISCONNECTED",
        "reconnecting",
        "RECONNECTING",
    }:
        state = RecorderOperationalState.RECONNECTING
        reason = "IBKR_RECONNECT_IN_PROGRESS"
    elif not expected_connection:
        state = RecorderOperationalState.RECORDING_DEGRADED
        reason = "IBKR_CONNECTION_STATE_UNEXPECTED"
    elif not callback_is_fresh:
        state = RecorderOperationalState.RECORDING_DEGRADED
        reason = "CALLBACK_HEARTBEAT_STALE"
    elif not raw_is_fresh:
        state = RecorderOperationalState.RECORDING_DEGRADED
        reason = "RAW_STORAGE_HEARTBEAT_STALE"
    elif not acknowledgement_is_fresh:
        state = RecorderOperationalState.RECORDING_DEGRADED
        reason = "INBOX_ACKNOWLEDGEMENT_STALE"
    elif not backlog_within_limit:
        state = RecorderOperationalState.RECORDING_DEGRADED
        reason = "INBOX_BACKLOG_LIMIT_EXCEEDED"
    else:
        state = RecorderOperationalState.RECORDING_HEALTHY
        reason = "ALL_RECORDING_HEALTH_CONDITIONS_MET"

    healthy = state is RecorderOperationalState.RECORDING_HEALTHY
    scientific_valid = state in {
        RecorderOperationalState.RECORDING_HEALTHY,
        RecorderOperationalState.MARKET_CLOSED,
        RecorderOperationalState.WAITING_FOR_PROSPECTIVE_START,
    }
    timestamps = {
        "process_heartbeat_at_utc": _iso(signals.process_heartbeat_at_utc),
        "latest_callback_received_at_utc": _iso(signals.latest_callback_received_at_utc),
        "latest_callback_durably_admitted_at_utc": _iso(
            signals.latest_callback_durably_admitted_at_utc
        ),
        "latest_raw_partition_committed_at_utc": _iso(
            signals.latest_raw_partition_committed_at_utc
        ),
        "latest_inbox_acknowledgement_at_utc": _iso(signals.latest_inbox_acknowledgement_at_utc),
        "latest_completed_five_minute_bar_at_utc": _iso(
            signals.latest_completed_five_minute_bar_at_utc
        ),
        "latest_successful_checkpoint_at_utc": _iso(signals.latest_successful_checkpoint_at_utc),
    }
    return OperationalStateProjection(
        state=state,
        reason_code=reason,
        healthy=healthy,
        scientific_recording_valid=scientific_valid,
        evaluated_at_utc=observed,
        run_id=signals.run_id,
        recorder_generation=signals.recorder_generation,
        owner_id=signals.owner_id,
        timestamps=timestamps,
        inbox={
            "backlog": signals.inbox_backlog,
            "oldest_unacknowledged_at_utc": _iso(signals.oldest_unacknowledged_at_utc),
            "oldest_unacknowledged_age_seconds": oldest_age,
            "within_limits": backlog_within_limit,
        },
        conditions={
            "fresh_recorder_lease": lease_is_fresh,
            "current_generation_owns_lease": lease_is_current,
            "process_heartbeat_fresh": process_is_fresh,
            "callback_heartbeat_fresh": callback_is_fresh,
            "callback_received_heartbeat_fresh": callback_received_is_fresh,
            "callback_durable_admission_heartbeat_fresh": (callback_admitted_is_fresh),
            "raw_storage_heartbeat_fresh": raw_is_fresh,
            "inbox_acknowledgement_fresh": acknowledgement_is_fresh,
            "expected_ibkr_connection_state": expected_connection,
            "required_market_data_mode_observed": expected_market_data_observed,
            "scientific_prerequisites_valid": signals.scientific_prerequisites_valid,
            "expected_artifact_count": signals.expected_artifact_count,
            "frozen_artifacts_verified": signals.frozen_artifacts_verified,
            "unresolved_required_gap_count": signals.unresolved_required_gap_count,
            "fatal_ingestion_latched": signals.fatal_ingestion_code is not None,
            "fatal_storage_latched": signals.fatal_storage_code is not None,
            "market_session_open": signals.market_session_open,
            "callbacks_expected": signals.callbacks_expected,
            "broker_state_mutation_count_zero": (signals.broker_state_mutation_count == 0),
        },
    )


def inactive_operational_projection(*, now: datetime) -> OperationalStateProjection:
    observed = _utc(now, label="inactive projection timestamp")
    return OperationalStateProjection(
        state=RecorderOperationalState.INACTIVE,
        reason_code="NO_ACTIVE_RECORDER_GENERATION",
        healthy=False,
        scientific_recording_valid=False,
        evaluated_at_utc=observed,
        run_id=None,
        recorder_generation=None,
        owner_id=None,
        timestamps={
            "process_heartbeat_at_utc": None,
            "latest_callback_received_at_utc": None,
            "latest_callback_durably_admitted_at_utc": None,
            "latest_raw_partition_committed_at_utc": None,
            "latest_inbox_acknowledgement_at_utc": None,
            "latest_completed_five_minute_bar_at_utc": None,
            "latest_successful_checkpoint_at_utc": None,
        },
        inbox={
            "backlog": 0,
            "oldest_unacknowledged_at_utc": None,
            "oldest_unacknowledged_age_seconds": None,
            "within_limits": True,
        },
        conditions={
            "fresh_recorder_lease": False,
            "current_generation_owns_lease": False,
            "process_heartbeat_fresh": False,
            "callback_heartbeat_fresh": False,
            "callback_received_heartbeat_fresh": False,
            "callback_durable_admission_heartbeat_fresh": False,
            "raw_storage_heartbeat_fresh": False,
            "inbox_acknowledgement_fresh": False,
            "expected_ibkr_connection_state": False,
            "required_market_data_mode_observed": False,
            "scientific_prerequisites_valid": False,
            "expected_artifact_count": 0,
            "frozen_artifacts_verified": False,
            "unresolved_required_gap_count": 0,
            "fatal_ingestion_latched": False,
            "fatal_storage_latched": False,
            "market_session_open": False,
            "callbacks_expected": False,
            "broker_state_mutation_count_zero": True,
        },
    )


def project_operational_state_from_database(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    now: datetime,
    prospective_start_utc: datetime,
    thresholds: OperationalThresholds,
) -> OperationalStateProjection:
    """Load persisted signals and apply the sole operational-state rule set."""

    state = connection.execute(
        """
        SELECT *
        FROM recorder_operational_state_v1
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    latch_rows = connection.execute(
        """
        SELECT latch_kind, stable_error_code, recorder_generation
        FROM recorder_fatal_latch_v1
        WHERE run_id = ? AND resolved_at_utc IS NULL
        """,
        (run_id,),
    ).fetchall()
    latches = {str(row["latch_kind"]): str(row["stable_error_code"]) for row in latch_rows}
    if state is None:
        inactive = inactive_operational_projection(now=now)
        fatal_kind = "storage" if "storage" in latches else "ingestion"
        fatal_code = latches.get(fatal_kind)
        if fatal_code is None:
            return inactive
        generations = [
            int(row["recorder_generation"])
            for row in latch_rows
            if row["recorder_generation"] is not None
        ]
        return inactive.model_copy(
            update={
                "state": (
                    RecorderOperationalState.STORAGE_FATAL
                    if fatal_kind == "storage"
                    else RecorderOperationalState.INGESTION_FATAL
                ),
                "reason_code": fatal_code,
                "run_id": run_id,
                "recorder_generation": (None if not generations else max(generations)),
                "conditions": {
                    **inactive.conditions,
                    "fatal_ingestion_latched": "ingestion" in latches,
                    "fatal_storage_latched": "storage" in latches,
                },
            }
        )
    lease = connection.execute(
        """
        SELECT *
        FROM recorder_lease
        WHERE lease_key = 'prospective_recorder' AND run_id = ?
        """,
        (run_id,),
    ).fetchone()

    def timestamp(name: str) -> datetime | None:
        value = state[name]
        return None if value is None else datetime.fromisoformat(str(value))

    signals = RecorderStateSignals(
        run_id=run_id,
        recorder_generation=int(state["recorder_generation"]),
        owner_id=str(state["owner_id"]),
        stored_state=RecorderOperationalState(str(state["state"])),
        process_heartbeat_at_utc=timestamp("process_heartbeat_at_utc"),
        latest_callback_received_at_utc=timestamp("latest_callback_received_at_utc"),
        latest_callback_durably_admitted_at_utc=timestamp(
            "latest_callback_durably_admitted_at_utc"
        ),
        latest_raw_partition_committed_at_utc=timestamp("latest_raw_partition_committed_at_utc"),
        latest_inbox_acknowledgement_at_utc=timestamp("latest_inbox_acknowledgement_at_utc"),
        latest_completed_five_minute_bar_at_utc=timestamp(
            "latest_completed_five_minute_bar_at_utc"
        ),
        latest_successful_checkpoint_at_utc=timestamp("latest_successful_checkpoint_at_utc"),
        inbox_backlog=int(state["inbox_backlog"]),
        oldest_unacknowledged_at_utc=timestamp("oldest_unacknowledged_at_utc"),
        market_session_open=bool(state["market_session_open"]),
        callbacks_expected=bool(state["callbacks_expected"]),
        ibkr_connection_state=(
            None if state["ibkr_connection_state"] is None else str(state["ibkr_connection_state"])
        ),
        required_market_data_mode=(
            None
            if state["required_market_data_mode"] is None
            else str(state["required_market_data_mode"])
        ),
        observed_market_data_mode=(
            None
            if state["observed_market_data_mode"] is None
            else str(state["observed_market_data_mode"])
        ),
        scientific_prerequisites_valid=bool(state["scientific_prerequisites_valid"]),
        expected_artifact_count=int(state["expected_artifact_count"]),
        frozen_artifacts_verified=bool(state["frozen_artifacts_verified"]),
        unresolved_required_gap_count=int(state["unresolved_required_gap_count"]),
        fatal_ingestion_code=latches.get("ingestion")
        or (None if state["fatal_ingestion_code"] is None else str(state["fatal_ingestion_code"])),
        fatal_storage_code=latches.get("storage")
        or (None if state["fatal_storage_code"] is None else str(state["fatal_storage_code"])),
        broker_state_mutation_count=int(state["broker_state_mutation_count"]),
        lease_owner_id=None if lease is None else str(lease["owner_id"]),
        lease_run_id=None if lease is None else str(lease["run_id"]),
        lease_generation=None if lease is None else int(lease["generation"]),
        lease_heartbeat_at_utc=(
            None if lease is None else datetime.fromisoformat(str(lease["heartbeat_at_utc"]))
        ),
    )
    return evaluate_operational_state(
        signals,
        now=now,
        prospective_start_utc=prospective_start_utc,
        thresholds=thresholds,
    )


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


class RecorderOperationalRepository:
    """Generation-fenced writes for recorder health, gaps, and artifacts."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def start_generation(
        self,
        *,
        run_id: str,
        recorder_generation: int,
        owner_id: str,
        started_at: datetime,
        required_market_data_mode: str | None,
        expected_artifact_count: int,
    ) -> None:
        if expected_artifact_count <= 0:
            raise ValueError("expected artifact count must be positive")
        observed = _utc(started_at, label="recorder generation start")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                """
                SELECT MAX(recorder_generation)
                FROM recorder_generation_v1
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()[0]
            if latest is not None and recorder_generation < int(latest):
                connection.rollback()
                raise RuntimeError("RECORDER_GENERATION_STALE")
            connection.execute(
                """
                INSERT OR IGNORE INTO recorder_generation_v1(
                    run_id, recorder_generation, owner_id, started_at_utc
                ) VALUES (?, ?, ?, ?)
                """,
                (run_id, recorder_generation, owner_id, observed.isoformat()),
            )
            existing = connection.execute(
                """
                SELECT recorder_generation, owner_id
                FROM recorder_operational_state_v1
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if existing is not None and (
                recorder_generation < int(existing["recorder_generation"])
                or (
                    recorder_generation == int(existing["recorder_generation"])
                    and owner_id != str(existing["owner_id"])
                )
            ):
                connection.rollback()
                raise RuntimeError("RECORDER_GENERATION_STALE")
            connection.execute(
                """
                INSERT INTO recorder_operational_state_v1(
                    run_id, recorder_generation, owner_id, state,
                    state_reason_code, process_heartbeat_at_utc,
                    required_market_data_mode, expected_artifact_count,
                    updated_at_utc
                ) VALUES (?, ?, ?, 'STARTING', 'RECORDER_GENERATION_STARTED',
                          ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    recorder_generation = excluded.recorder_generation,
                    owner_id = excluded.owner_id,
                    state = excluded.state,
                    state_reason_code = excluded.state_reason_code,
                    process_heartbeat_at_utc =
                        excluded.process_heartbeat_at_utc,
                    latest_callback_received_at_utc = NULL,
                    latest_callback_durably_admitted_at_utc = NULL,
                    latest_raw_partition_committed_at_utc = NULL,
                    latest_inbox_acknowledgement_at_utc = NULL,
                    latest_completed_five_minute_bar_at_utc = NULL,
                    latest_successful_checkpoint_at_utc = NULL,
                    inbox_backlog = 0,
                    oldest_unacknowledged_at_utc = NULL,
                    required_market_data_mode =
                        excluded.required_market_data_mode,
                    observed_market_data_mode = NULL,
                    scientific_prerequisites_valid = 0,
                    expected_artifact_count =
                        excluded.expected_artifact_count,
                    frozen_artifacts_verified = 0,
                    unresolved_required_gap_count = 0,
                    fatal_ingestion_code = NULL,
                    fatal_storage_code = NULL,
                    scientific_recording_valid = 0,
                    broker_state_mutation_count = 0,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (
                    run_id,
                    recorder_generation,
                    owner_id,
                    observed.isoformat(),
                    required_market_data_mode,
                    expected_artifact_count,
                    observed.isoformat(),
                ),
            )
            active_latches = {
                str(row["latch_kind"]): str(row["stable_error_code"])
                for row in connection.execute(
                    """
                    SELECT latch_kind, stable_error_code
                    FROM recorder_fatal_latch_v1
                    WHERE run_id = ? AND resolved_at_utc IS NULL
                    """,
                    (run_id,),
                ).fetchall()
            }
            if active_latches:
                storage_code = active_latches.get("storage")
                ingestion_code = active_latches.get("ingestion")
                state = "STORAGE_FATAL" if storage_code is not None else "INGESTION_FATAL"
                reason = storage_code or ingestion_code
                connection.execute(
                    """
                    UPDATE recorder_operational_state_v1
                    SET state = ?, state_reason_code = ?,
                        fatal_ingestion_code = ?,
                        fatal_storage_code = ?,
                        scientific_recording_valid = 0,
                        updated_at_utc = ?
                    WHERE run_id = ? AND recorder_generation = ?
                    """,
                    (
                        state,
                        reason,
                        ingestion_code,
                        storage_code,
                        observed.isoformat(),
                        run_id,
                        recorder_generation,
                    ),
                )
            self._refresh_gap_count(connection, run_id)
            inbox = connection.execute(
                """
                SELECT COUNT(*) AS backlog, MIN(received_utc) AS oldest
                FROM callback_inbox_v1
                WHERE admission_run_id = ?
                  AND status IN (
                      'provider_pending', 'pending', 'leased', 'quarantined'
                  )
                """,
                (run_id,),
            ).fetchone()
            connection.execute(
                """
                UPDATE recorder_operational_state_v1
                SET inbox_backlog = ?, oldest_unacknowledged_at_utc = ?
                WHERE run_id = ? AND recorder_generation = ?
                """,
                (
                    int(inbox["backlog"]),
                    inbox["oldest"],
                    run_id,
                    recorder_generation,
                ),
            )
            connection.commit()

    def touch(
        self,
        *,
        run_id: str,
        recorder_generation: int,
        owner_id: str,
        now: datetime,
        market_session_open: bool | None = None,
        callbacks_expected: bool | None = None,
        ibkr_connection_state: str | None = None,
        observed_market_data_mode: str | None = None,
        scientific_prerequisites_valid: bool | None = None,
        frozen_artifacts_verified: bool | None = None,
        latest_raw_partition_committed_at_utc: datetime | None = None,
        latest_completed_five_minute_bar_at_utc: datetime | None = None,
        latest_successful_checkpoint_at_utc: datetime | None = None,
        broker_state_mutation_count: int | None = None,
    ) -> None:
        observed = _utc(now, label="recorder process heartbeat")
        updates: dict[str, object] = {
            "process_heartbeat_at_utc": observed.isoformat(),
            "updated_at_utc": observed.isoformat(),
        }
        optional: dict[str, object | None] = {
            "market_session_open": (
                None if market_session_open is None else int(market_session_open)
            ),
            "callbacks_expected": (None if callbacks_expected is None else int(callbacks_expected)),
            "ibkr_connection_state": ibkr_connection_state,
            "observed_market_data_mode": observed_market_data_mode,
            "scientific_prerequisites_valid": (
                None
                if scientific_prerequisites_valid is None
                else int(scientific_prerequisites_valid)
            ),
            "frozen_artifacts_verified": (
                None if frozen_artifacts_verified is None else int(frozen_artifacts_verified)
            ),
            "latest_raw_partition_committed_at_utc": _iso(latest_raw_partition_committed_at_utc),
            "latest_completed_five_minute_bar_at_utc": _iso(
                latest_completed_five_minute_bar_at_utc
            ),
            "latest_successful_checkpoint_at_utc": _iso(latest_successful_checkpoint_at_utc),
            "broker_state_mutation_count": broker_state_mutation_count,
        }
        updates.update({key: value for key, value in optional.items() if value is not None})
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE recorder_operational_state_v1
                SET {assignments}
                WHERE run_id = ? AND recorder_generation = ? AND owner_id = ?
                """,
                (*updates.values(), run_id, recorder_generation, owner_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("RECORDER_GENERATION_STALE")

    def refresh_projection(
        self,
        *,
        run_id: str,
        recorder_generation: int,
        owner_id: str,
        now: datetime,
        prospective_start_utc: datetime,
        thresholds: OperationalThresholds,
    ) -> OperationalStateProjection:
        """Persist the generation-fenced result of the authoritative evaluator."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            projection = project_operational_state_from_database(
                connection,
                run_id=run_id,
                now=now,
                prospective_start_utc=prospective_start_utc,
                thresholds=thresholds,
            )
            if (
                projection.recorder_generation != recorder_generation
                or projection.owner_id != owner_id
            ):
                connection.rollback()
                raise RuntimeError("RECORDER_GENERATION_STALE")
            cursor = connection.execute(
                """
                UPDATE recorder_operational_state_v1
                SET state = ?, state_reason_code = ?,
                    scientific_recording_valid = ?, updated_at_utc = ?
                WHERE run_id = ? AND recorder_generation = ? AND owner_id = ?
                """,
                (
                    projection.state.value,
                    projection.reason_code,
                    int(projection.scientific_recording_valid),
                    projection.evaluated_at_utc.isoformat(),
                    run_id,
                    recorder_generation,
                    owner_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("RECORDER_GENERATION_STALE")
            connection.commit()
        return projection

    def set_stopping(
        self,
        *,
        run_id: str,
        recorder_generation: int,
        owner_id: str,
        now: datetime,
    ) -> None:
        observed = _utc(now, label="recorder stopping timestamp")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE recorder_operational_state_v1
                SET state = 'STOPPING', state_reason_code = 'RECORDER_STOPPING',
                    scientific_recording_valid = 0, updated_at_utc = ?
                WHERE run_id = ? AND recorder_generation = ? AND owner_id = ?
                """,
                (observed.isoformat(), run_id, recorder_generation, owner_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("RECORDER_GENERATION_STALE")
            connection.execute(
                """
                UPDATE recorder_generation_v1
                SET stopping_at_utc = ?
                WHERE run_id = ? AND recorder_generation = ? AND owner_id = ?
                """,
                (observed.isoformat(), run_id, recorder_generation, owner_id),
            )
            connection.commit()

    def set_stopped_cleanly(
        self,
        *,
        run_id: str,
        recorder_generation: int,
        owner_id: str,
        now: datetime,
        termination_reason: str,
    ) -> None:
        observed = _utc(now, label="recorder stop timestamp")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE recorder_operational_state_v1
                SET state = 'STOPPED_CLEANLY',
                    state_reason_code = 'RECORDER_STOPPED_CLEANLY',
                    scientific_recording_valid = 0, updated_at_utc = ?
                WHERE run_id = ? AND recorder_generation = ? AND owner_id = ?
                """,
                (observed.isoformat(), run_id, recorder_generation, owner_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise RuntimeError("RECORDER_GENERATION_STALE")
            connection.execute(
                """
                UPDATE recorder_generation_v1
                SET stopped_at_utc = ?, termination_reason = ?,
                    stopped_cleanly = 1
                WHERE run_id = ? AND recorder_generation = ? AND owner_id = ?
                """,
                (
                    observed.isoformat(),
                    termination_reason,
                    run_id,
                    recorder_generation,
                    owner_id,
                ),
            )
            connection.commit()

    def record_gap(self, gap: GapIncident) -> str:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM gap_incident_v1 WHERE gap_id = ?",
                (gap.gap_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO gap_incident_v1(
                        gap_id, run_id, recorder_generation, symbol,
                        stream_kind, request_id, connection_generation,
                        start_timestamp_utc, end_timestamp_utc,
                        detection_timestamp_utc, cause_code, severity,
                        recoverability, backfill_attempted, backfill_result,
                        affected_first_source_sequence,
                        affected_last_source_sequence,
                        affected_episode_ids_json, resolution_timestamp_utc,
                        resolution_evidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?)
                    """,
                    (
                        gap.gap_id,
                        gap.run_id,
                        gap.recorder_generation,
                        gap.symbol,
                        gap.stream_kind,
                        gap.request_id,
                        gap.connection_generation,
                        gap.start_timestamp_utc.isoformat(),
                        _iso(gap.end_timestamp_utc),
                        gap.detection_timestamp_utc.isoformat(),
                        gap.cause_code,
                        gap.severity,
                        gap.recoverability,
                        int(gap.backfill_attempted),
                        gap.backfill_result,
                        gap.affected_first_source_sequence,
                        gap.affected_last_source_sequence,
                        json.dumps(gap.affected_episode_ids, separators=(",", ":")),
                        _iso(gap.resolution_timestamp_utc),
                        gap.resolution_evidence,
                    ),
                )
            else:
                # Resolution fields are intentionally excluded: resolving an incident
                # is an append-only lifecycle transition, and a duplicate detector
                # observation must remain idempotent after that transition.
                stored_identity = (
                    str(existing["run_id"]),
                    int(existing["recorder_generation"]),
                    str(existing["symbol"]),
                    str(existing["stream_kind"]),
                    existing["request_id"],
                    int(existing["connection_generation"]),
                    str(existing["start_timestamp_utc"]),
                    existing["end_timestamp_utc"],
                    str(existing["detection_timestamp_utc"]),
                    str(existing["cause_code"]),
                    str(existing["severity"]),
                    str(existing["recoverability"]),
                    existing["affected_first_source_sequence"],
                    existing["affected_last_source_sequence"],
                    tuple(json.loads(str(existing["affected_episode_ids_json"]))),
                )
                expected_identity = (
                    gap.run_id,
                    gap.recorder_generation,
                    gap.symbol,
                    gap.stream_kind,
                    gap.request_id,
                    gap.connection_generation,
                    _iso(gap.start_timestamp_utc),
                    _iso(gap.end_timestamp_utc),
                    _iso(gap.detection_timestamp_utc),
                    gap.cause_code,
                    gap.severity,
                    gap.recoverability,
                    gap.affected_first_source_sequence,
                    gap.affected_last_source_sequence,
                    gap.affected_episode_ids,
                )
                if stored_identity != expected_identity:
                    connection.rollback()
                    raise ValueError("immutable gap incident differs")
            self._refresh_gap_count(connection, gap.run_id)
            connection.commit()
        return gap.gap_id

    def active_gaps(self, *, run_id: str) -> tuple[GapIncident, ...]:
        """Restore unresolved incidents before a replacement generation scores."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM gap_incident_v1
                WHERE run_id = ? AND resolution_timestamp_utc IS NULL
                ORDER BY start_timestamp_utc, detection_timestamp_utc, gap_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(
            GapIncident(
                gap_id=str(row["gap_id"]),
                run_id=str(row["run_id"]),
                recorder_generation=int(row["recorder_generation"]),
                symbol=str(row["symbol"]),
                stream_kind=str(row["stream_kind"]),
                request_id=None if row["request_id"] is None else int(row["request_id"]),
                connection_generation=int(row["connection_generation"]),
                start_timestamp_utc=datetime.fromisoformat(str(row["start_timestamp_utc"])),
                end_timestamp_utc=(
                    None
                    if row["end_timestamp_utc"] is None
                    else datetime.fromisoformat(str(row["end_timestamp_utc"]))
                ),
                detection_timestamp_utc=datetime.fromisoformat(str(row["detection_timestamp_utc"])),
                cause_code=str(row["cause_code"]),
                severity=cast(
                    Literal["optional", "degraded", "scientific"],
                    str(row["severity"]),
                ),
                recoverability=cast(
                    Literal["recoverable", "unrecoverable", "unknown"],
                    str(row["recoverability"]),
                ),
                backfill_attempted=bool(row["backfill_attempted"]),
                backfill_result=(
                    None if row["backfill_result"] is None else str(row["backfill_result"])
                ),
                affected_first_source_sequence=(
                    None
                    if row["affected_first_source_sequence"] is None
                    else int(row["affected_first_source_sequence"])
                ),
                affected_last_source_sequence=(
                    None
                    if row["affected_last_source_sequence"] is None
                    else int(row["affected_last_source_sequence"])
                ),
                affected_episode_ids=tuple(
                    str(value) for value in json.loads(str(row["affected_episode_ids_json"]))
                ),
            )
            for row in rows
        )

    def resolve_gap(
        self,
        *,
        gap_id: str,
        run_id: str,
        recorder_generation: int,
        resolved_at: datetime,
        resolution_evidence: str,
        end_timestamp_utc: datetime | None = None,
    ) -> None:
        observed = _utc(resolved_at, label="gap resolution timestamp")
        evidence = resolution_evidence.strip()
        if not evidence:
            raise ValueError("gap resolution evidence is required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE gap_incident_v1
                SET end_timestamp_utc = COALESCE(end_timestamp_utc, ?),
                    resolution_timestamp_utc = ?,
                    resolution_evidence = ?
                WHERE gap_id = ? AND run_id = ?
                  AND recorder_generation = ?
                  AND resolution_timestamp_utc IS NULL
                """,
                (
                    _iso(end_timestamp_utc or observed),
                    observed.isoformat(),
                    evidence,
                    gap_id,
                    run_id,
                    recorder_generation,
                ),
            )
            if cursor.rowcount not in {0, 1}:
                connection.rollback()
                raise RuntimeError("GAP_RESOLUTION_FAILED")
            self._refresh_gap_count(connection, run_id)
            connection.commit()

    @staticmethod
    def _refresh_gap_count(
        connection: sqlite3.Connection,
        run_id: str,
    ) -> None:
        count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM gap_incident_v1
                WHERE run_id = ? AND resolution_timestamp_utc IS NULL
                  AND severity = 'scientific'
                """,
                (run_id,),
            ).fetchone()[0]
        )
        connection.execute(
            """
            UPDATE recorder_operational_state_v1
            SET unresolved_required_gap_count = ?
            WHERE run_id = ?
            """,
            (count, run_id),
        )

    def record_artifact_verification(
        self,
        verification: RuntimeArtifactVerification,
    ) -> str:
        encoded_details = json.dumps(
            verification.details,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT *
                FROM runtime_artifact_verification_v1
                WHERE run_id = ? AND recorder_generation = ?
                  AND artifact_name = ?
                """,
                (
                    verification.run_id,
                    verification.recorder_generation,
                    verification.artifact_name,
                ),
            ).fetchone()
            values = (
                verification.verification_id,
                verification.run_id,
                verification.recorder_generation,
                verification.artifact_bundle_id,
                verification.artifact_name,
                verification.expected_hash,
                verification.observed_hash,
                verification.feature_contract_version,
                verification.activation_receipt_identity,
                int(verification.found),
                int(verification.loaded),
                int(verification.schema_validated),
                int(verification.hash_verified),
                int(verification.contract_compatible),
                int(verification.used_by_active_generation),
                verification.load_timestamp_utc.isoformat(),
                verification.verification_result,
                verification.blocker,
                encoded_details,
            )
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO runtime_artifact_verification_v1(
                        verification_id, run_id, recorder_generation,
                        artifact_bundle_id, artifact_name, expected_hash,
                        observed_hash, feature_contract_version,
                        activation_receipt_identity, found, loaded,
                        schema_validated, hash_verified, contract_compatible,
                        used_by_active_generation, load_timestamp_utc,
                        verification_result, blocker, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?)
                    """,
                    values,
                )
            else:
                stored = tuple(
                    existing[name]
                    for name in (
                        "verification_id",
                        "run_id",
                        "recorder_generation",
                        "artifact_bundle_id",
                        "artifact_name",
                        "expected_hash",
                        "observed_hash",
                        "feature_contract_version",
                        "activation_receipt_identity",
                        "found",
                        "loaded",
                        "schema_validated",
                        "hash_verified",
                        "contract_compatible",
                        "used_by_active_generation",
                        "load_timestamp_utc",
                        "verification_result",
                        "blocker",
                        "details_json",
                    )
                )
                if stored != values:
                    connection.rollback()
                    raise ValueError("runtime artifact verification is immutable")
            verified_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM runtime_artifact_verification_v1
                    WHERE run_id = ? AND recorder_generation = ?
                      AND verification_result = 'verified'
                      AND used_by_active_generation = 1
                    """,
                    (verification.run_id, verification.recorder_generation),
                ).fetchone()[0]
            )
            blocked_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM runtime_artifact_verification_v1
                    WHERE run_id = ? AND recorder_generation = ?
                      AND verification_result <> 'verified'
                    """,
                    (verification.run_id, verification.recorder_generation),
                ).fetchone()[0]
            )
            expected_count_row = connection.execute(
                """
                SELECT expected_artifact_count
                FROM recorder_operational_state_v1
                WHERE run_id = ? AND recorder_generation = ?
                """,
                (verification.run_id, verification.recorder_generation),
            ).fetchone()
            expected_count = (
                0
                if expected_count_row is None
                else int(expected_count_row["expected_artifact_count"])
            )
            total_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM runtime_artifact_verification_v1
                    WHERE run_id = ? AND recorder_generation = ?
                    """,
                    (verification.run_id, verification.recorder_generation),
                ).fetchone()[0]
            )
            connection.execute(
                """
                UPDATE recorder_operational_state_v1
                SET frozen_artifacts_verified = ?
                WHERE run_id = ? AND recorder_generation = ?
                """,
                (
                    int(
                        expected_count > 0
                        and total_count == expected_count
                        and verified_count == expected_count
                        and blocked_count == 0
                    ),
                    verification.run_id,
                    verification.recorder_generation,
                ),
            )
            connection.commit()
        return verification.verification_id


def stable_gap_id(
    *,
    run_id: str,
    recorder_generation: int,
    symbol: str,
    stream_kind: str,
    request_id: int | None,
    connection_generation: int,
    start_timestamp_utc: datetime,
    cause_code: str,
) -> str:
    identity = "|".join(
        (
            run_id,
            str(recorder_generation),
            symbol,
            stream_kind,
            str(request_id),
            str(connection_generation),
            _utc(start_timestamp_utc, label="gap start").isoformat(),
            cause_code,
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def stable_artifact_verification_id(
    *,
    run_id: str,
    recorder_generation: int,
    artifact_name: str,
    expected_hash: str,
) -> str:
    return hashlib.sha256(
        f"{run_id}|{recorder_generation}|{artifact_name}|{expected_hash}".encode()
    ).hexdigest()


__all__ = [
    "GapIncident",
    "OperationalStateProjection",
    "OperationalThresholds",
    "RecorderOperationalRepository",
    "RecorderOperationalState",
    "RecorderStateSignals",
    "RuntimeArtifactVerification",
    "evaluate_operational_state",
    "inactive_operational_projection",
    "project_operational_state_from_database",
    "stable_artifact_verification_id",
    "stable_gap_id",
]
