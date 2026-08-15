"""Frozen all-M1C event-window retention contract.

The retention selector deliberately has no family field: every immutable
``m1c_episode_v0`` row receives the same causal raw-data window.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.live_bars import xnys_session_bounds
from stocker_prospective.m1c_event_window_retention_repository_v0 import (
    RETENTION_CONTROLLED_EVENT_TYPES,
    EventWindowRetentionRepositoryV0,
    PreparedRetainedPartitionV0,
    RetainedPartitionReceiptV0,
    SessionRetentionReceiptV0,
    SourcePartitionV0,
)
from stocker_prospective.partition_store import atomic_replace, fsync_directory, sha256_path
from stocker_prospective.raw_storage_fence import RawStorageFence

PRE_EVENT_RETENTION = timedelta(minutes=5)
POST_EVENT_RETENTION = timedelta(minutes=20)

RETENTION_CONTROLLED_CALLBACK_KINDS = frozenset(
    {
        "level1_quote_update",
        "official_provider_tick_by_tick_bidask",
        "official_provider_tick_by_tick_trade",
        "official_provider_tick_price",
        "official_provider_tick_size",
        "tick_by_tick_bidask",
        "tick_by_tick_trade",
        "tick_price",
        "tick_size",
        "depth",
        "depth_reset",
        "official_provider_depth",
        "official_provider_depth_reset",
    }
)
RETENTION_CONTROLLED_STREAM_KINDS = frozenset(
    {
        "underlying_level1",
        "underlying_tick_bidask",
        "underlying_tick_last",
        "underlying_depth",
    }
)


def eligible_retention_sessions(*, first_session: date, count: int = 20) -> tuple[date, ...]:
    """Return the frozen consecutive XNYS-session schedule."""

    if count != 20:
        raise ValueError("retention V0 requires exactly 20 eligible sessions")
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise RuntimeError("blocked_market_calendar_unavailable") from exc
    schedule = mcal.get_calendar("XNYS").schedule(
        start_date=first_session.isoformat(),
        end_date=(first_session + timedelta(days=45)).isoformat(),
    )
    sessions = tuple(date.fromisoformat(str(index.date())) for index in schedule.index)
    if not sessions or sessions[0] != first_session or len(sessions) < count:
        raise ValueError("first eligible retention session must be an XNYS session")
    return sessions[:count]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retention timestamps must be timezone-aware")
    return value.astimezone(UTC)


class M1CEventReferenceV0(BaseModel):
    """Minimum immutable M1C identity needed to plan raw retention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    session: date
    t0: datetime

    @field_validator("t0")
    @classmethod
    def _t0_is_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class RetainedSymbolWindowV0(BaseModel):
    """One merged inclusive raw-data interval for one symbol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    start_utc: datetime
    end_utc: datetime
    episode_ids: tuple[str, ...]

    @field_validator("start_utc", "end_utc")
    @classmethod
    def _timestamps_are_utc(cls, value: datetime) -> datetime:
        return _utc(value)

    def contains(self, timestamp: datetime) -> bool:
        observed = _utc(timestamp)
        return self.start_utc <= observed <= self.end_utc


class SessionEventWindowPlanV0(BaseModel):
    """Deterministic union of every M1C event window in one session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str = "m1c_event_window_retention_v0"
    run_id: str = Field(min_length=1)
    session: date
    activation_timestamp_utc: datetime | None = None
    first_eligible_session: date | None = None
    planned_sessions: int = 20
    session_ordinal: int | None = None
    episode_ids: tuple[str, ...]
    windows: tuple[RetainedSymbolWindowV0, ...]

    @classmethod
    def from_events(
        cls,
        *,
        run_id: str,
        session: date,
        events: Iterable[M1CEventReferenceV0],
    ) -> SessionEventWindowPlanV0:
        observed = tuple(events)
        if any(event.session != session for event in observed):
            raise ValueError("retention plan cannot cross sessions")
        grouped: dict[str, list[tuple[datetime, datetime, str]]] = defaultdict(list)
        for event in observed:
            grouped[event.symbol].append(
                (
                    event.t0 - PRE_EVENT_RETENTION,
                    event.t0 + POST_EVENT_RETENTION,
                    event.episode_id,
                )
            )

        windows: list[RetainedSymbolWindowV0] = []
        for symbol in sorted(grouped):
            merged: list[tuple[datetime, datetime, set[str]]] = []
            for start, end, episode_id in sorted(grouped[symbol]):
                if not merged or start > merged[-1][1]:
                    merged.append((start, end, {episode_id}))
                    continue
                previous_start, previous_end, episode_ids = merged[-1]
                episode_ids.add(episode_id)
                merged[-1] = (previous_start, max(previous_end, end), episode_ids)
            windows.extend(
                RetainedSymbolWindowV0(
                    symbol=symbol,
                    start_utc=start,
                    end_utc=end,
                    episode_ids=tuple(sorted(episode_ids)),
                )
                for start, end, episode_ids in merged
            )
        return cls(
            run_id=run_id,
            session=session,
            episode_ids=tuple(sorted(event.episode_id for event in observed)),
            windows=tuple(windows),
        )


def _parsed_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, str):
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise ValueError("unsupported retained event timestamp")


def _ordering_timestamp(row: dict[str, Any]) -> datetime:
    provider = _parsed_timestamp(row.get("provider_timestamp_utc"))
    received = _parsed_timestamp(row.get("received_timestamp_utc"))
    if provider is not None:
        return provider
    if received is None:
        raise ValueError("retained event has no causal timestamp")
    return received


def _is_controlled_row(event_type: str, row: dict[str, Any]) -> bool:
    if event_type != "raw_callback_envelope_event":
        return True
    owner = row.get("stream_owner")
    if isinstance(owner, str):
        try:
            owner = json.loads(owner)
        except json.JSONDecodeError:
            return False
    if not isinstance(owner, dict):
        return False
    return (
        str(row.get("callback_kind")) in RETENTION_CONTROLLED_CALLBACK_KINDS
        and str(owner.get("kind")) in RETENTION_CONTROLLED_STREAM_KINDS
    )


class SessionEventWindowFinalizerV0:
    """Verify, retain and then retire transient market-data partitions."""

    def __init__(
        self,
        *,
        database: str | Path,
        raw_root: str | Path,
        retained_root: str | Path,
        run_id: str,
        activation_timestamp_utc: datetime,
        first_eligible_session: date,
        planned_sessions: int = 20,
        failure_injector: Callable[[str, Path], None] | None = None,
    ) -> None:
        self.raw_root = Path(raw_root).resolve()
        self.retained_root = Path(retained_root).resolve()
        try:
            self.retained_root.relative_to(self.raw_root)
        except ValueError as exc:
            raise ValueError("retained root must be inside the verified raw root") from exc
        if self.raw_root == self.retained_root:
            raise ValueError("retained root must be a child of the transient raw root")
        self.run_id = run_id
        self.activation_timestamp_utc = _utc(activation_timestamp_utc)
        self.eligible_sessions = eligible_retention_sessions(
            first_session=first_eligible_session,
            count=planned_sessions,
        )
        self.failure_injector = failure_injector
        self.repository = EventWindowRetentionRepositoryV0(database, run_id=run_id)
        self.storage_fence = RawStorageFence(self.raw_root)

    def _checkpoint(self, phase: str, path: Path) -> None:
        if self.failure_injector is not None:
            self.failure_injector(phase, path)

    def _plan(self, session: date) -> SessionEventWindowPlanV0:
        events = tuple(
            M1CEventReferenceV0(
                episode_id=str(row["episode_id"]),
                symbol=str(row["symbol"]),
                session=date.fromisoformat(str(row["session_date"])),
                t0=datetime.fromisoformat(str(row["trigger_bar_end_utc"])),
            )
            for row in self.repository.episode_references(session)
        )
        plan = SessionEventWindowPlanV0.from_events(
            run_id=self.run_id,
            session=session,
            events=events,
        )
        return plan.model_copy(
            update={
                "activation_timestamp_utc": self.activation_timestamp_utc,
                "first_eligible_session": self.eligible_sessions[0],
                "planned_sessions": len(self.eligible_sessions),
                "session_ordinal": self.eligible_sessions.index(session) + 1,
            }
        )

    def _verify_source(self, source: SourcePartitionV0) -> Path:
        if not source.complete:
            raise RuntimeError("RETENTION_SOURCE_PARTITION_INCOMPLETE")
        if source.file_path.is_symlink():
            raise RuntimeError("RETENTION_SOURCE_SYMLINK")
        resolved = source.file_path.resolve(strict=True)
        try:
            resolved.relative_to(self.raw_root)
        except ValueError as exc:
            raise RuntimeError("RETENTION_SOURCE_OUTSIDE_RAW_ROOT") from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise RuntimeError("RETENTION_SOURCE_NOT_REGULAR")
        if sha256_path(resolved) != source.content_hash:
            raise RuntimeError("RETENTION_SOURCE_HASH_MISMATCH")
        metadata_path = resolved.with_name(f"part-{source.content_hash}.metadata.json")
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise RuntimeError("RETENTION_SOURCE_METADATA_MISSING")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("RETENTION_SOURCE_METADATA_INVALID") from exc
        if metadata.get("content_hash") != source.content_hash:
            raise RuntimeError("RETENTION_SOURCE_METADATA_HASH_MISMATCH")
        return resolved

    def _retire_prepared_source(
        self,
        partition: RetainedPartitionReceiptV0,
        *,
        observed_at: datetime,
    ) -> None:
        if partition.source_deleted:
            return
        if partition.retained_row_count:
            if (
                partition.retained_file_path is None
                or partition.retained_content_hash is None
            ):
                raise RuntimeError("RETENTION_PREPARED_OUTPUT_IDENTITY_MISSING")
            if partition.retained_file_path.is_symlink():
                raise RuntimeError("RETENTION_OUTPUT_SYMLINK")
            retained_path = partition.retained_file_path.resolve(strict=True)
            try:
                retained_path.relative_to(self.retained_root)
            except ValueError as exc:
                raise RuntimeError("RETENTION_OUTPUT_OUTSIDE_RETAINED_ROOT") from exc
            if retained_path.is_symlink() or sha256_path(retained_path) != (
                partition.retained_content_hash
            ):
                raise RuntimeError("RETENTION_PREPARED_OUTPUT_HASH_MISMATCH")

        source_path = partition.source_file_path
        if source_path.is_symlink():
            raise RuntimeError("RETENTION_SOURCE_SYMLINK")
        resolved_source = source_path.resolve(strict=False)
        try:
            resolved_source.relative_to(self.raw_root)
        except ValueError as exc:
            raise RuntimeError("RETENTION_SOURCE_OUTSIDE_RAW_ROOT") from exc
        metadata_path = resolved_source.with_name(
            f"part-{partition.source_content_hash}.metadata.json"
        )
        if resolved_source.exists():
            if not resolved_source.is_file() or sha256_path(resolved_source) != (
                partition.source_content_hash
            ):
                raise RuntimeError("RETENTION_SOURCE_HASH_MISMATCH")
            resolved_source.unlink()
            fsync_directory(resolved_source.parent)
            self._checkpoint("after_source_data_unlink", resolved_source)
        if metadata_path.exists():
            if metadata_path.is_symlink() or not metadata_path.is_file():
                raise RuntimeError("RETENTION_SOURCE_METADATA_INVALID")
            metadata_path.unlink()
            fsync_directory(metadata_path.parent)
            self._checkpoint("after_source_metadata_unlink", metadata_path)
        self.repository.mark_source_deleted(
            partition.source_content_hash,
            deleted_at=observed_at,
        )

    def _resume_prepared(
        self,
        receipt: SessionRetentionReceiptV0,
        *,
        observed_at: datetime,
    ) -> SessionRetentionReceiptV0:
        if receipt.summary_file_path.is_symlink() or not receipt.summary_file_path.is_file():
            raise RuntimeError("RETENTION_PREPARED_SUMMARY_MISSING")
        try:
            receipt.summary_file_path.resolve(strict=True).relative_to(self.retained_root)
        except ValueError as exc:
            raise RuntimeError("RETENTION_SUMMARY_OUTSIDE_RETAINED_ROOT") from exc
        if sha256_path(receipt.summary_file_path) != receipt.summary_sha256:
            raise RuntimeError("RETENTION_PREPARED_SUMMARY_HASH_MISMATCH")
        for partition in receipt.partitions:
            self._retire_prepared_source(partition, observed_at=observed_at)
        self.repository.complete_session(receipt.session, completed_at=observed_at)
        self.repository.purge_terminal_session_callbacks(
            receipt.session,
            purged_at=observed_at,
        )
        completed = self.repository.session_receipt(receipt.session)
        if completed is None or completed.status != "COMPLETE":
            raise RuntimeError("RETENTION_COMPLETION_RECEIPT_MISSING")
        return completed

    @staticmethod
    def _inside_window(
        *,
        row: dict[str, Any],
        windows: tuple[RetainedSymbolWindowV0, ...],
    ) -> bool:
        timestamp = _ordering_timestamp(row)
        return any(window.contains(timestamp) for window in windows)

    def _retained_partition(
        self,
        *,
        source: SourcePartitionV0,
        windows: tuple[RetainedSymbolWindowV0, ...],
    ) -> tuple[PreparedRetainedPartitionV0, list[dict[str, Any]]]:
        source_path = self._verify_source(source)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("blocked_raw_event_storage_unavailable: pyarrow") from exc
        table = pq.ParquetFile(source_path).read()  # type: ignore[no-untyped-call]
        if table.num_rows != source.row_count:
            raise RuntimeError("RETENTION_SOURCE_ROW_COUNT_MISMATCH")
        rows = table.to_pylist()
        retained_indices: list[int] = []
        summarised_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            controlled = _is_controlled_row(source.event_type, row)
            if not controlled or self._inside_window(row=row, windows=windows):
                retained_indices.append(index)
            elif controlled:
                summarised_rows.append(row)
        if not retained_indices:
            return (
                PreparedRetainedPartitionV0(
                    source=source,
                    retained_row_count=0,
                    summarised_row_count=len(summarised_rows),
                ),
                summarised_rows,
            )

        retained_table = table.take(pa.array(retained_indices, type=pa.int64()))
        # A full-row retention rewrite would otherwise be byte-identical to its
        # source and collide with the run-scoped manifest hash identity. Add
        # immutable lineage metadata without changing any raw evidence column.
        retained_metadata = dict(retained_table.schema.metadata or {})
        retained_metadata.update(
            {
                b"retention_dataset_version": b"m1c_event_window_retention_v0",
                b"retention_source_content_hash": source.content_hash.encode(),
            }
        )
        retained_table = retained_table.replace_schema_metadata(retained_metadata)
        target_directory = (
            self.retained_root
            / "dataset=m1c_event_window_retention_v0"
            / f"session_date={source.session.isoformat()}"
            / f"symbol={source.symbol}"
            / f"event_type={source.event_type}"
        )
        target_directory.mkdir(parents=True, exist_ok=True)
        temporary = target_directory / f".retained-{uuid.uuid4().hex}.tmp.parquet"
        pq.write_table(  # type: ignore[no-untyped-call]
            retained_table, temporary, compression="zstd"
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        retained_hash = sha256_path(temporary)
        retained_path = target_directory / f"part-{retained_hash}.complete.parquet"
        if retained_path.is_file():
            temporary.unlink(missing_ok=True)
            if sha256_path(retained_path) != retained_hash:
                raise RuntimeError("RETENTION_EXISTING_OUTPUT_HASH_MISMATCH")
        else:
            atomic_replace(temporary, retained_path)
        retained_rows = retained_table.to_pylist()
        timestamps = tuple(_ordering_timestamp(row) for row in retained_rows)
        metadata_path = retained_path.with_name(f"part-{retained_hash}.metadata.json")
        metadata_payload = {
            "data_path": str(retained_path),
            "row_count": retained_table.num_rows,
            "minimum_timestamp_utc": min(timestamps).isoformat(),
            "maximum_timestamp_utc": max(timestamps).isoformat(),
            "schema_version": source.schema_version,
            "content_hash": retained_hash,
            "complete": True,
            "gap_count": source.gap_count,
            "recorder_version": source.recorder_version,
            "contract_version": source.contract_version,
            "run_id": self.run_id,
            "source_content_hash": source.content_hash,
            "retention_dataset_version": "m1c_event_window_retention_v0",
        }
        encoded_metadata = json.dumps(
            metadata_payload, sort_keys=True, separators=(",", ":")
        ) + "\n"
        if metadata_path.is_file():
            if metadata_path.read_text(encoding="utf-8") != encoded_metadata:
                raise RuntimeError("RETENTION_EXISTING_METADATA_MISMATCH")
        else:
            metadata_temporary = metadata_path.with_name(
                f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
            )
            metadata_temporary.write_text(encoded_metadata, encoding="utf-8")
            with metadata_temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            atomic_replace(metadata_temporary, metadata_path)
        if sha256_path(retained_path) != retained_hash:
            raise RuntimeError("RETENTION_OUTPUT_HASH_MISMATCH")
        return (
            PreparedRetainedPartitionV0(
                source=source,
                retained_content_hash=retained_hash,
                retained_file_path=retained_path,
                retained_row_count=retained_table.num_rows,
                summarised_row_count=len(summarised_rows),
                retained_minimum_timestamp_utc=min(timestamps),
                retained_maximum_timestamp_utc=max(timestamps),
            ),
            summarised_rows,
        )

    def _write_summary(
        self,
        *,
        session: date,
        partitions: tuple[PreparedRetainedPartitionV0, ...],
        summarised_rows_by_source: dict[str, list[dict[str, Any]]],
    ) -> tuple[Path, str]:
        summary_path = (
            self.retained_root
            / "dataset=m1c_event_window_retention_v0"
            / f"session_date={session.isoformat()}"
            / "collection_quality_summary.csv"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = summary_path.with_name(f".{summary_path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "session_date",
                    "symbol",
                    "event_types",
                    "source_partition_count",
                    "source_hash_set_sha256",
                    "source_rows",
                    "retained_event_rows",
                    "summarised_between_event_rows",
                    "provider_timestamp_available_rows",
                    "first_summarised_event_utc",
                    "last_summarised_event_utc",
                    "longest_summarised_event_gap_seconds",
                    "locked_quote_count",
                    "crossed_quote_count",
                    "invalid_quote_count",
                    "market_data_types",
                ),
                lineterminator="\n",
            )
            writer.writeheader()
            by_symbol: dict[str, list[PreparedRetainedPartitionV0]] = defaultdict(list)
            for item in partitions:
                by_symbol[item.source.symbol].append(item)
            for symbol in sorted(by_symbol):
                items = tuple(by_symbol[symbol])
                rows = [
                    row
                    for item in items
                    for row in summarised_rows_by_source.get(item.source.content_hash, [])
                ]
                timestamps = sorted(_ordering_timestamp(row) for row in rows)
                longest_gap = max(
                    (
                        (current - previous).total_seconds()
                        for previous, current in zip(timestamps, timestamps[1:], strict=False)
                    ),
                    default=0.0,
                )
                quote_source_hashes = {
                    item.source.content_hash
                    for item in items
                    if item.source.event_type
                    in {
                        "underlying_bbo_update",
                        "underlying_level1_quote_event",
                        "underlying_tick_bidask_event",
                    }
                }
                quote_rows = [
                    row
                    for content_hash in quote_source_hashes
                    for row in summarised_rows_by_source.get(content_hash, [])
                ]
                locked = 0
                crossed = 0
                invalid = 0
                for row in quote_rows:
                    bid = row.get("bid")
                    ask = row.get("ask")
                    if (
                        not isinstance(bid, (int, float))
                        or not isinstance(ask, (int, float))
                        or bid <= 0
                        or ask <= 0
                        or row.get("quote_valid") is False
                    ):
                        invalid += 1
                    elif bid == ask:
                        locked += 1
                    elif bid > ask:
                        crossed += 1
                writer.writerow(
                    {
                        "session_date": session.isoformat(),
                        "symbol": symbol,
                        "event_types": "|".join(
                            sorted({item.source.event_type for item in items})
                        ),
                        "source_partition_count": len(items),
                        "source_hash_set_sha256": hashlib.sha256(
                            "|".join(
                                sorted(item.source.content_hash for item in items)
                            ).encode()
                        ).hexdigest(),
                        "source_rows": sum(item.source.row_count for item in items),
                        "retained_event_rows": sum(
                            item.retained_row_count for item in items
                        ),
                        "summarised_between_event_rows": sum(
                            item.summarised_row_count for item in items
                        ),
                        "provider_timestamp_available_rows": sum(
                            row.get("provider_timestamp_utc") is not None for row in rows
                        ),
                        "first_summarised_event_utc": (
                            "" if not timestamps else timestamps[0].isoformat()
                        ),
                        "last_summarised_event_utc": (
                            "" if not timestamps else timestamps[-1].isoformat()
                        ),
                        "longest_summarised_event_gap_seconds": f"{longest_gap:.6f}",
                        "locked_quote_count": locked,
                        "crossed_quote_count": crossed,
                        "invalid_quote_count": invalid,
                        "market_data_types": "|".join(
                            sorted(
                                {
                                    str(row["market_data_type"])
                                    for row in rows
                                    if row.get("market_data_type") is not None
                                }
                            )
                        ),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(temporary, summary_path)
        return summary_path, sha256_path(summary_path)

    def _finalize_session_locked(
        self,
        *,
        session: date,
        observed_at: datetime,
    ) -> SessionRetentionReceiptV0:
        observed = _utc(observed_at)
        existing = self.repository.session_receipt(session)
        if existing is not None and existing.status == "COMPLETE":
            self.repository.purge_terminal_session_callbacks(
                session,
                purged_at=observed,
            )
            return existing
        if existing is not None and existing.status == "PREPARED":
            return self._resume_prepared(existing, observed_at=observed)
        if session not in self.eligible_sessions:
            raise RuntimeError("RETENTION_SESSION_OUTSIDE_FROZEN_SCHEDULE")
        market_open, market_close = xnys_session_bounds(session)
        if market_open < self.activation_timestamp_utc:
            raise RuntimeError("RETENTION_SESSION_PRECEDES_ACTIVATION")
        if observed < market_close + POST_EVENT_RETENTION:
            raise RuntimeError("RETENTION_SESSION_HORIZON_OPEN")
        plan = self._plan(session)
        windows_by_symbol = {
            symbol: tuple(window for window in plan.windows if window.symbol == symbol)
            for symbol in {window.symbol for window in plan.windows}
        }
        source_partitions = self.repository.source_partitions(
            session,
            event_types=RETENTION_CONTROLLED_EVENT_TYPES,
        )
        self.repository.assert_session_callbacks_are_terminal(session)
        self.repository.assert_sources_are_terminally_materialized(source_partitions)
        prepared: list[PreparedRetainedPartitionV0] = []
        summarised_rows_by_source: dict[str, list[dict[str, Any]]] = {}
        for source in source_partitions:
            item, summarised_rows = self._retained_partition(
                source=source,
                windows=windows_by_symbol.get(source.symbol, ()),
            )
            prepared.append(item)
            summarised_rows_by_source[source.content_hash] = summarised_rows
        prepared_tuple = tuple(prepared)
        summary_path, summary_hash = self._write_summary(
            session=session,
            partitions=prepared_tuple,
            summarised_rows_by_source=summarised_rows_by_source,
        )
        self.repository.prepare_session(
            plan_json=plan.model_dump_json(),
            session=session,
            episode_ids=plan.episode_ids,
            partitions=prepared_tuple,
            source_event_types=RETENTION_CONTROLLED_EVENT_TYPES,
            summary_file_path=summary_path,
            summary_sha256=summary_hash,
            prepared_at=observed,
        )
        prepared_receipt = self.repository.session_receipt(session)
        if prepared_receipt is None or prepared_receipt.status != "PREPARED":
            raise RuntimeError("RETENTION_PREPARED_RECEIPT_MISSING")
        return self._resume_prepared(prepared_receipt, observed_at=observed)

    def finalize_session(
        self,
        *,
        session: date,
        observed_at: datetime,
    ) -> SessionRetentionReceiptV0:
        """Finalize under the same fence used for file+manifest recorder commits."""

        with self.storage_fence.exclusive():
            return self._finalize_session_locked(session=session, observed_at=observed_at)
