"""Atomic append-only Parquet storage for high-volume raw market events."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from stocker_prospective.contract import claims_boundary, claims_hash
from stocker_prospective.events import RawEvent

SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


class PartitionWriteResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    data_path: Path
    metadata_path: Path
    row_count: int
    minimum_timestamp_utc: datetime
    maximum_timestamp_utc: datetime
    schema_version: str
    content_hash: str
    complete: bool
    gap_count: int
    recorder_version: str
    contract_version: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: str) -> str:
    if not SAFE_PART.fullmatch(value):
        raise ValueError(f"unsafe partition identity: {value!r}")
    return value


def _event_type(event: RawEvent) -> str:
    names = {
        "UnderlyingLevel1QuoteEvent": "underlying_level1_quote_event",
        "UnderlyingTickBidAskEvent": "underlying_tick_bidask_event",
        "UnderlyingTickTradeEvent": "underlying_tick_trade_event",
        "UnderlyingDepthEvent": "underlying_depth_event",
        "UnderlyingDepthSnapshotEvent": "underlying_depth_snapshot",
        "OptionQuoteEvent": "option_quote_event",
        "FiveMinuteBarEvent": "five_minute_bar_event",
    }
    try:
        return names[type(event).__name__]
    except KeyError as exc:
        raise ValueError(f"unsupported raw event type: {type(event).__name__}") from exc


def _normalise_row(event: RawEvent) -> dict[str, Any]:
    row = event.model_dump(mode="json")
    for key, value in tuple(row.items()):
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return row


class PartitionedEventStore:
    """Write immutable event batches under source/date/symbol/type/hour."""

    def __init__(
        self,
        *,
        root: str | Path,
        prospective_collection_start: datetime,
        recorder_version: str,
        contract_version: str,
        schema_version: str = "raw-market-events-v0",
    ) -> None:
        if (
            prospective_collection_start.tzinfo is None
            or prospective_collection_start.utcoffset() is None
        ):
            raise ValueError("prospective_collection_start must be timezone-aware")
        self.root = Path(root)
        self.prospective_collection_start = prospective_collection_start.astimezone(UTC)
        self.recorder_version = recorder_version
        self.contract_version = contract_version
        self.schema_version = schema_version

    def _partition(
        self,
        *,
        data_source: str,
        event: RawEvent,
    ) -> tuple[Path, tuple[str, str, str, str]]:
        timestamp = event.received_timestamp_utc.astimezone(UTC)
        identity = (
            event.session.isoformat(),
            _safe(event.symbol),
            _event_type(event),
            f"{timestamp.hour:02d}",
        )
        path = (
            self.root
            / f"data_source={_safe(data_source)}"
            / f"session_date={identity[0]}"
            / f"symbol={identity[1]}"
            / f"event_type={identity[2]}"
            / f"hour={identity[3]}"
        )
        return path, identity

    def write_events(
        self,
        *,
        data_source: str,
        events: tuple[RawEvent, ...],
        complete: bool,
        gap_count: int = 0,
    ) -> PartitionWriteResult:
        if not events:
            raise ValueError("at least one raw event is required")
        if gap_count < 0:
            raise ValueError("gap_count must be nonnegative")
        for event in events:
            if event.received_timestamp_utc < self.prospective_collection_start:
                raise ValueError("recorded_at precedes prospective_collection_start")
        ordered = tuple(
            sorted(
                events,
                key=lambda item: (
                    item.ordering_timestamp,
                    item.received_monotonic_ns,
                    item.source_sequence,
                    item.event_id,
                ),
            )
        )
        partition, identity = self._partition(data_source=data_source, event=ordered[0])
        if any(
            self._partition(data_source=data_source, event=event)[1] != identity
            for event in ordered[1:]
        ):
            raise ValueError("one atomic write may contain only one event partition")
        partition.mkdir(parents=True, exist_ok=True)
        rows = [_normalise_row(event) for event in ordered]
        minimum = min(item.received_timestamp_utc for item in ordered)
        maximum = max(item.received_timestamp_utc for item in ordered)
        temporary = partition / f".events-{uuid.uuid4().hex}.tmp.parquet"
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("blocked_raw_event_storage_unavailable: pyarrow") from exc
        table = pa.Table.from_pylist(rows)
        metadata = dict(table.schema.metadata or {})
        metadata.update(
            {
                b"schema_version": self.schema_version.encode(),
                b"contract_version": self.contract_version.encode(),
                b"claims_hash": claims_hash().encode(),
                b"claims_boundary": json.dumps(
                    claims_boundary(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            }
        )
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, temporary, compression="zstd")  # type: ignore[no-untyped-call]
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        content_hash = _sha256(temporary)
        state = "complete" if complete else "incomplete"
        data_path = partition / f"part-{content_hash}.{state}.parquet"
        metadata_path = partition / f"part-{content_hash}.metadata.json"
        staging_path = partition / f".part-{content_hash}.{state}.staged.parquet"
        if data_path.is_file() and metadata_path.is_file():
            temporary.unlink(missing_ok=True)
            result = self._load_result(metadata_path)
            if not self.verify(result):
                raise RuntimeError("existing raw partition hash mismatch")
            return result
        if staging_path.is_file():
            temporary.unlink(missing_ok=True)
            if _sha256(staging_path) != content_hash:
                raise RuntimeError("staged raw partition hash mismatch")
        elif not data_path.is_file():
            os.replace(temporary, staging_path)
        else:
            temporary.unlink(missing_ok=True)
        payload = {
            "data_path": str(data_path),
            "row_count": len(rows),
            "minimum_timestamp_utc": minimum.isoformat(),
            "maximum_timestamp_utc": maximum.isoformat(),
            "schema_version": self.schema_version,
            "content_hash": content_hash,
            "complete": complete,
            "gap_count": gap_count,
            "recorder_version": self.recorder_version,
            "contract_version": self.contract_version,
            "claims_boundary": claims_boundary(),
        }
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError("append-only partition metadata differs")
        else:
            metadata_temporary = metadata_path.with_name(
                f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
            )
            metadata_temporary.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with metadata_temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(metadata_temporary, metadata_path)
        if not data_path.is_file():
            if not staging_path.is_file():
                raise RuntimeError("raw partition staging evidence is absent")
            os.replace(staging_path, data_path)
        else:
            staging_path.unlink(missing_ok=True)
        result = self._load_result(metadata_path)
        if not self.verify(result):
            raise RuntimeError("final raw partition hash mismatch")
        return result

    def write_grouped(
        self,
        *,
        data_source: str,
        events: tuple[RawEvent, ...],
        complete: bool,
        gap_count: int = 0,
    ) -> tuple[PartitionWriteResult, ...]:
        grouped: dict[tuple[str, str, str, str], list[RawEvent]] = {}
        for event in events:
            _, identity = self._partition(data_source=data_source, event=event)
            grouped.setdefault(identity, []).append(event)
        return tuple(
            self.write_events(
                data_source=data_source,
                events=tuple(grouped[key]),
                complete=complete,
                gap_count=gap_count,
            )
            for key in sorted(grouped)
        )

    @staticmethod
    def _load_result(path: Path) -> PartitionWriteResult:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PartitionWriteResult(
            data_path=Path(payload["data_path"]),
            metadata_path=path,
            row_count=int(payload["row_count"]),
            minimum_timestamp_utc=datetime.fromisoformat(payload["minimum_timestamp_utc"]),
            maximum_timestamp_utc=datetime.fromisoformat(payload["maximum_timestamp_utc"]),
            schema_version=str(payload["schema_version"]),
            content_hash=str(payload["content_hash"]),
            complete=bool(payload["complete"]),
            gap_count=int(payload["gap_count"]),
            recorder_version=str(payload["recorder_version"]),
            contract_version=str(payload["contract_version"]),
        )

    @staticmethod
    def verify(result: PartitionWriteResult) -> bool:
        return (
            result.data_path.is_file()
            and result.metadata_path.is_file()
            and _sha256(result.data_path) == result.content_hash
        )
