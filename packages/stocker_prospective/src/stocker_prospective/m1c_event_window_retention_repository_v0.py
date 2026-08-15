"""Durable receipts for verified all-M1C event-window retention."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.database import ProspectiveRepository

DATASET_VERSION: Literal["m1c_event_window_retention_v0"] = (
    "m1c_event_window_retention_v0"
)
RETENTION_CONTROLLED_EVENT_TYPES = (
    "raw_callback_envelope_event",
    "underlying_bbo_update",
    "underlying_level1_quote_event",
    "underlying_tick_bidask_event",
    "underlying_tick_trade_event",
    "underlying_trade_update",
    "underlying_depth_event",
    "underlying_depth_snapshot",
)

RETENTION_CONTROLLED_CALLBACK_KINDS = (
    "level1_quote_update",
    "official_provider_tick_by_tick_bidask",
    "official_provider_tick_by_tick_trade",
    "official_provider_tick_price",
    "official_provider_tick_size",
    "official_provider_depth",
    "official_provider_depth_reset",
    "tick_by_tick_bidask",
    "tick_by_tick_trade",
    "tick_price",
    "tick_size",
    "depth",
    "depth_reset",
)
NEW_YORK = ZoneInfo("America/New_York")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retention receipt timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _session_timestamp_bounds(session: date) -> tuple[str, str]:
    """Match event ingestion's America/New_York calendar-date session rule."""

    start = datetime.combine(session, datetime.min.time(), tzinfo=NEW_YORK)
    end = start + timedelta(days=1)
    return start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()


class SourcePartitionV0(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: int
    data_source: str
    session: date
    symbol: str
    event_type: str
    file_path: Path
    row_count: int = Field(ge=0)
    minimum_timestamp_utc: datetime
    maximum_timestamp_utc: datetime
    schema_version: str
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    complete: bool
    gap_count: int = Field(ge=0)
    recorder_version: str
    contract_version: str
    claims_json: str


class PreparedRetainedPartitionV0(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: SourcePartitionV0
    retained_content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    retained_file_path: Path | None = None
    retained_row_count: int = Field(ge=0)
    summarised_row_count: int = Field(ge=0)
    retained_minimum_timestamp_utc: datetime | None = None
    retained_maximum_timestamp_utc: datetime | None = None


class RetainedPartitionReceiptV0(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_content_hash: str
    source_file_path: Path
    source_row_count: int
    retained_content_hash: str | None
    retained_file_path: Path | None
    retained_row_count: int
    summarised_row_count: int
    source_deleted: bool


class SessionRetentionReceiptV0(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: Literal["m1c_event_window_retention_v0"] = DATASET_VERSION
    run_id: str
    session: date
    status: Literal["PREPARED", "COMPLETE"]
    episode_ids: tuple[str, ...]
    source_row_count: int
    retained_row_count: int
    summarised_row_count: int
    source_partitions_deleted: int
    summary_file_path: Path
    summary_sha256: str
    prepared_at_utc: datetime
    completed_at_utc: datetime | None
    partitions: tuple[RetainedPartitionReceiptV0, ...]

    @field_validator("prepared_at_utc", "completed_at_utc")
    @classmethod
    def _timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)


class EventWindowRetentionRepositoryV0:
    """SQLite coordination for a session retention transaction."""

    def __init__(self, database: str | Path, *, run_id: str) -> None:
        self.repository = ProspectiveRepository(database)
        self.run_id = run_id

    def episode_references(self, session: date) -> tuple[dict[str, object], ...]:
        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT episode_id, symbol, session_date, trigger_bar_end_utc
                FROM m1c_episode_v0
                WHERE run_id = ? AND session_date = ?
                ORDER BY symbol, trigger_bar_end_utc, episode_id
                """,
                (self.run_id, session.isoformat()),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def source_partitions(
        self,
        session: date,
        *,
        event_types: tuple[str, ...],
    ) -> tuple[SourcePartitionV0, ...]:
        if not event_types:
            return ()
        with self.repository._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, data_source, session_date, symbol, event_type,
                       file_path, row_count, minimum_timestamp_utc,
                       maximum_timestamp_utc, schema_version, content_hash,
                       complete, gap_count, recorder_version, contract_version,
                       claims_json
                FROM raw_partition_manifest_v0
                WHERE run_id = ? AND session_date = ?
                  AND retention_state IN ('ACTIVE', 'RETIREMENT_PREPARED')
                  AND event_type IN ({','.join('?' for _ in event_types)})
                ORDER BY symbol, event_type, minimum_timestamp_utc, content_hash
                """,
                (self.run_id, session.isoformat(), *event_types),
            ).fetchall()
        return tuple(
            SourcePartitionV0(
                manifest_id=int(row["id"]),
                data_source=str(row["data_source"]),
                session=date.fromisoformat(str(row["session_date"])),
                symbol=str(row["symbol"]),
                event_type=str(row["event_type"]),
                file_path=Path(str(row["file_path"])),
                row_count=int(row["row_count"]),
                minimum_timestamp_utc=datetime.fromisoformat(
                    str(row["minimum_timestamp_utc"])
                ),
                maximum_timestamp_utc=datetime.fromisoformat(
                    str(row["maximum_timestamp_utc"])
                ),
                schema_version=str(row["schema_version"]),
                content_hash=str(row["content_hash"]),
                complete=bool(row["complete"]),
                gap_count=int(row["gap_count"]),
                recorder_version=str(row["recorder_version"]),
                contract_version=str(row["contract_version"]),
                claims_json=str(row["claims_json"]),
            )
            for row in rows
        )

    def assert_sources_are_terminally_materialized(
        self,
        sources: tuple[SourcePartitionV0, ...],
    ) -> None:
        """Block deletion while any callback referencing a source is nonterminal."""

        with self.repository._connect() as connection:
            for source in sources:
                unsafe = int(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM callback_raw_materialization_v1 AS materialization
                        JOIN json_each(materialization.raw_partition_hashes_json) AS hash
                        JOIN callback_inbox_v1 AS inbox
                          ON inbox.inbox_event_id = materialization.inbox_event_id
                        LEFT JOIN callback_processing_commit_v1 AS processing
                          ON processing.inbox_event_id = materialization.inbox_event_id
                        WHERE materialization.run_id = ?
                          AND hash.value = ?
                          AND (
                            inbox.status <> 'acknowledged'
                            OR processing.inbox_event_id IS NULL
                            OR processing.raw_partition_hashes_json <>
                               materialization.raw_partition_hashes_json
                          )
                        """,
                        (self.run_id, source.content_hash),
                    ).fetchone()[0]
                )
                if unsafe:
                    raise RuntimeError("RETENTION_SOURCE_CALLBACK_NOT_TERMINAL")

    def assert_session_callbacks_are_terminal(self, session: date) -> None:
        """Require every admitted underlying callback for the UTC session date to settle."""

        placeholders = ",".join("?" for _ in RETENTION_CONTROLLED_CALLBACK_KINDS)
        session_start, session_end = _session_timestamp_bounds(session)
        with self.repository._connect() as connection:
            unsafe = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM callback_inbox_v1 AS inbox
                    LEFT JOIN callback_raw_materialization_v1 AS materialization
                      ON materialization.inbox_event_id = inbox.inbox_event_id
                    LEFT JOIN callback_processing_commit_v1 AS processing
                      ON processing.inbox_event_id = inbox.inbox_event_id
                    WHERE inbox.admission_run_id = ?
                      AND COALESCE(inbox.provider_timestamp_utc, inbox.received_utc) >= ?
                      AND COALESCE(inbox.provider_timestamp_utc, inbox.received_utc) < ?
                      AND inbox.callback_kind IN ({placeholders})
                      AND (
                        inbox.status <> 'acknowledged'
                        OR materialization.inbox_event_id IS NULL
                        OR processing.inbox_event_id IS NULL
                        OR processing.raw_partition_hashes_json <>
                           materialization.raw_partition_hashes_json
                      )
                    """,
                    (
                        self.run_id,
                        session_start,
                        session_end,
                        *RETENTION_CONTROLLED_CALLBACK_KINDS,
                    ),
                ).fetchone()[0]
            )
        if unsafe:
            raise RuntimeError("RETENTION_SESSION_CALLBACK_NOT_TERMINAL")

    def prepare_session(
        self,
        *,
        plan_json: str,
        session: date,
        episode_ids: tuple[str, ...],
        partitions: tuple[PreparedRetainedPartitionV0, ...],
        source_event_types: tuple[str, ...],
        summary_file_path: Path,
        summary_sha256: str,
        prepared_at: datetime,
    ) -> None:
        observed = _utc(prepared_at)
        source_rows = sum(item.source.row_count for item in partitions)
        retained_rows = sum(item.retained_row_count for item in partitions)
        summarised_rows = sum(item.summarised_row_count for item in partitions)
        with self.repository._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            event_types = tuple(sorted(set(source_event_types)))
            if event_types:
                active_rows = connection.execute(
                    f"""
                    SELECT content_hash
                    FROM raw_partition_manifest_v0
                    WHERE run_id = ? AND session_date = ?
                      AND retention_state IN ('ACTIVE', 'RETIREMENT_PREPARED')
                      AND event_type IN ({','.join('?' for _ in event_types)})
                    ORDER BY content_hash
                    """,
                    (self.run_id, session.isoformat(), *event_types),
                ).fetchall()
                active_hashes = tuple(str(row["content_hash"]) for row in active_rows)
                expected_hashes = tuple(
                    sorted(item.source.content_hash for item in partitions)
                )
                if active_hashes != expected_hashes:
                    connection.rollback()
                    raise RuntimeError("RETENTION_SOURCE_SET_CHANGED_DURING_PREPARE")
            connection.execute(
                """
                INSERT INTO m1c_event_window_retention_session_v0(
                    run_id, session_date, dataset_version, status, plan_json,
                    episode_count, source_partition_count, source_row_count,
                    retained_row_count, summarised_row_count, summary_file_path,
                    summary_sha256, prepared_at_utc, completed_at_utc
                ) VALUES (?, ?, ?, 'PREPARED', ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(run_id, session_date, dataset_version) DO NOTHING
                """,
                (
                    self.run_id,
                    session.isoformat(),
                    DATASET_VERSION,
                    plan_json,
                    len(episode_ids),
                    len(partitions),
                    source_rows,
                    retained_rows,
                    summarised_rows,
                    str(summary_file_path),
                    summary_sha256,
                    observed.isoformat(),
                ),
            )
            stored = connection.execute(
                """
                SELECT plan_json, summary_sha256
                FROM m1c_event_window_retention_session_v0
                WHERE run_id = ? AND session_date = ? AND dataset_version = ?
                """,
                (self.run_id, session.isoformat(), DATASET_VERSION),
            ).fetchone()
            if stored is None or str(stored["plan_json"]) != plan_json:
                connection.rollback()
                raise RuntimeError("RETENTION_PLAN_IDENTITY_MISMATCH")
            if str(stored["summary_sha256"]) != summary_sha256:
                connection.rollback()
                raise RuntimeError("RETENTION_SUMMARY_IDENTITY_MISMATCH")

            for item in partitions:
                retained_manifest_id: int | None = None
                if item.retained_row_count > 0:
                    assert item.retained_content_hash is not None
                    assert item.retained_file_path is not None
                    assert item.retained_minimum_timestamp_utc is not None
                    assert item.retained_maximum_timestamp_utc is not None
                    connection.execute(
                        """
                        INSERT INTO raw_partition_manifest_v0(
                            run_id, data_source, session_date, symbol, event_type,
                            file_path, row_count, minimum_timestamp_utc,
                            maximum_timestamp_utc, schema_version, content_hash,
                            complete, gap_count, recorder_version, contract_version,
                            recorded_at_utc, claims_json, retention_state
                        ) VALUES (?, 'retained_event_windows_v0', ?, ?, ?, ?, ?, ?, ?,
                                  ?, ?, 1, ?, ?, ?, ?, ?, 'RETAINED')
                        ON CONFLICT(run_id, content_hash) DO NOTHING
                        """,
                        (
                            self.run_id,
                            session.isoformat(),
                            item.source.symbol,
                            item.source.event_type,
                            str(item.retained_file_path),
                            item.retained_row_count,
                            item.retained_minimum_timestamp_utc.isoformat(),
                            item.retained_maximum_timestamp_utc.isoformat(),
                            item.source.schema_version,
                            item.retained_content_hash,
                            item.source.gap_count,
                            item.source.recorder_version,
                            item.source.contract_version,
                            observed.isoformat(),
                            item.source.claims_json,
                        ),
                    )
                    retained_row = connection.execute(
                        """
                        SELECT id FROM raw_partition_manifest_v0
                        WHERE run_id = ? AND content_hash = ?
                          AND retention_state = 'RETAINED'
                        """,
                        (self.run_id, item.retained_content_hash),
                    ).fetchone()
                    if retained_row is None:
                        connection.rollback()
                        raise RuntimeError("RETAINED_MANIFEST_MISSING")
                    retained_manifest_id = int(retained_row["id"])
                connection.execute(
                    """
                    INSERT INTO m1c_event_window_retention_partition_v0(
                        run_id, session_date, dataset_version, source_manifest_id,
                        source_content_hash, source_file_path, source_row_count,
                        retained_manifest_id, retained_content_hash,
                        retained_file_path, retained_row_count,
                        summarised_row_count, source_verified, retained_verified,
                        source_deleted, deleted_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, 0, NULL)
                    ON CONFLICT(run_id, source_content_hash) DO NOTHING
                    """,
                    (
                        self.run_id,
                        session.isoformat(),
                        DATASET_VERSION,
                        item.source.manifest_id,
                        item.source.content_hash,
                        str(item.source.file_path),
                        item.source.row_count,
                        retained_manifest_id,
                        item.retained_content_hash,
                        None
                        if item.retained_file_path is None
                        else str(item.retained_file_path),
                        item.retained_row_count,
                        item.summarised_row_count,
                    ),
                )
                updated = connection.execute(
                    """
                    UPDATE raw_partition_manifest_v0
                    SET retention_state = 'RETIREMENT_PREPARED'
                    WHERE id = ? AND run_id = ?
                      AND retention_state IN ('ACTIVE', 'RETIREMENT_PREPARED')
                    """,
                    (item.source.manifest_id, self.run_id),
                )
                if updated.rowcount != 1:
                    connection.rollback()
                    raise RuntimeError("SOURCE_MANIFEST_NOT_ACTIVE")
            connection.commit()

    def mark_source_deleted(self, content_hash: str, *, deleted_at: datetime) -> None:
        observed = _utc(deleted_at)
        with self.repository._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT source_manifest_id
                FROM m1c_event_window_retention_partition_v0
                WHERE run_id = ? AND source_content_hash = ?
                """,
                (self.run_id, content_hash),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise RuntimeError("RETENTION_SOURCE_RECEIPT_MISSING")
            connection.execute(
                """
                UPDATE raw_partition_manifest_v0 SET retention_state = 'RETIRED'
                WHERE id = ? AND retention_state IN ('RETIREMENT_PREPARED', 'RETIRED')
                """,
                (int(row["source_manifest_id"]),),
            )
            connection.execute(
                """
                UPDATE m1c_event_window_retention_partition_v0
                SET source_deleted = 1, deleted_at_utc = ?
                WHERE run_id = ? AND source_content_hash = ?
                """,
                (observed.isoformat(), self.run_id, content_hash),
            )
            connection.commit()

    def complete_session(self, session: date, *, completed_at: datetime) -> None:
        observed = _utc(completed_at)
        with self.repository._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pending = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM m1c_event_window_retention_partition_v0
                    WHERE run_id = ? AND session_date = ? AND source_deleted = 0
                    """,
                    (self.run_id, session.isoformat()),
                ).fetchone()[0]
            )
            if pending:
                connection.rollback()
                raise RuntimeError("RETENTION_SOURCE_DELETION_INCOMPLETE")
            updated = connection.execute(
                """
                UPDATE m1c_event_window_retention_session_v0
                SET status = 'COMPLETE', completed_at_utc = ?
                WHERE run_id = ? AND session_date = ? AND dataset_version = ?
                """,
                (observed.isoformat(), self.run_id, session.isoformat(), DATASET_VERSION),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise RuntimeError("RETENTION_SESSION_RECEIPT_MISSING")
            connection.commit()

    def purge_terminal_session_callbacks(
        self,
        session: date,
        *,
        purged_at: datetime,
    ) -> int:
        """Bound the high-volume inbox after durable raw retention is complete."""

        observed = _utc(purged_at)
        placeholders = ",".join("?" for _ in RETENTION_CONTROLLED_CALLBACK_KINDS)
        session_start, session_end = _session_timestamp_bounds(session)
        with self.repository._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            complete = connection.execute(
                """
                SELECT 1 FROM m1c_event_window_retention_session_v0
                WHERE run_id = ? AND session_date = ? AND dataset_version = ?
                  AND status = 'COMPLETE'
                """,
                (self.run_id, session.isoformat(), DATASET_VERSION),
            ).fetchone()
            if complete is None:
                connection.rollback()
                raise RuntimeError("RETENTION_CALLBACK_PURGE_BEFORE_COMPLETE")
            existing = connection.execute(
                """
                SELECT purged_callback_count
                FROM m1c_event_window_callback_purge_v0
                WHERE run_id = ? AND session_date = ? AND dataset_version = ?
                """,
                (self.run_id, session.isoformat(), DATASET_VERSION),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                return int(existing["purged_callback_count"])
            rows = connection.execute(
                f"""
                SELECT inbox.inbox_event_id
                FROM callback_inbox_v1 AS inbox
                JOIN callback_raw_materialization_v1 AS materialization
                  ON materialization.inbox_event_id = inbox.inbox_event_id
                JOIN callback_processing_commit_v1 AS processing
                  ON processing.inbox_event_id = inbox.inbox_event_id
                 AND processing.raw_partition_hashes_json =
                     materialization.raw_partition_hashes_json
                WHERE inbox.admission_run_id = ?
                  AND COALESCE(inbox.provider_timestamp_utc, inbox.received_utc) >= ?
                  AND COALESCE(inbox.provider_timestamp_utc, inbox.received_utc) < ?
                  AND inbox.callback_kind IN ({placeholders})
                  AND inbox.status = 'acknowledged'
                ORDER BY inbox.inbox_event_id
                """,
                (
                    self.run_id,
                    session_start,
                    session_end,
                    *RETENTION_CONTROLLED_CALLBACK_KINDS,
                ),
            ).fetchall()
            identities = tuple(str(row["inbox_event_id"]) for row in rows)
            identity_hash = hashlib.sha256("\n".join(identities).encode()).hexdigest()
            connection.execute(
                """
                INSERT INTO m1c_event_window_callback_purge_v0(
                    run_id, session_date, dataset_version, purged_callback_count,
                    callback_identity_set_sha256, purged_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    session.isoformat(),
                    DATASET_VERSION,
                    len(identities),
                    identity_hash,
                    observed.isoformat(),
                ),
            )
            if identities:
                connection.executemany(
                    "DELETE FROM callback_inbox_v1 WHERE inbox_event_id = ?",
                    ((identity,) for identity in identities),
                )
            connection.commit()
        return len(identities)

    def session_receipt(self, session: date) -> SessionRetentionReceiptV0 | None:
        with self.repository._connect() as connection:
            session_row = connection.execute(
                """
                SELECT * FROM m1c_event_window_retention_session_v0
                WHERE run_id = ? AND session_date = ? AND dataset_version = ?
                """,
                (self.run_id, session.isoformat(), DATASET_VERSION),
            ).fetchone()
            if session_row is None:
                return None
            partition_rows = connection.execute(
                """
                SELECT * FROM m1c_event_window_retention_partition_v0
                WHERE run_id = ? AND session_date = ?
                ORDER BY source_content_hash
                """,
                (self.run_id, session.isoformat()),
            ).fetchall()
        plan = json.loads(str(session_row["plan_json"]))
        partitions = tuple(
            RetainedPartitionReceiptV0(
                source_content_hash=str(row["source_content_hash"]),
                source_file_path=Path(str(row["source_file_path"])),
                source_row_count=int(row["source_row_count"]),
                retained_content_hash=(
                    None
                    if row["retained_content_hash"] is None
                    else str(row["retained_content_hash"])
                ),
                retained_file_path=(
                    None
                    if row["retained_file_path"] is None
                    else Path(str(row["retained_file_path"]))
                ),
                retained_row_count=int(row["retained_row_count"]),
                summarised_row_count=int(row["summarised_row_count"]),
                source_deleted=bool(row["source_deleted"]),
            )
            for row in partition_rows
        )
        return SessionRetentionReceiptV0(
            run_id=self.run_id,
            session=session,
            status=cast(Literal["PREPARED", "COMPLETE"], str(session_row["status"])),
            episode_ids=tuple(str(value) for value in plan["episode_ids"]),
            source_row_count=int(session_row["source_row_count"]),
            retained_row_count=int(session_row["retained_row_count"]),
            summarised_row_count=int(session_row["summarised_row_count"]),
            source_partitions_deleted=sum(item.source_deleted for item in partitions),
            summary_file_path=Path(str(session_row["summary_file_path"])),
            summary_sha256=str(session_row["summary_sha256"]),
            prepared_at_utc=datetime.fromisoformat(str(session_row["prepared_at_utc"])),
            completed_at_utc=(
                None
                if session_row["completed_at_utc"] is None
                else datetime.fromisoformat(str(session_row["completed_at_utc"]))
            ),
            partitions=partitions,
        )
