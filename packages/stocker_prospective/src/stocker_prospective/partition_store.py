"""Atomic append-only Parquet storage for high-volume raw market events."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from stocker_prospective.contract import claims_boundary, claims_hash
from stocker_prospective.events import RawEvent

SAFE_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
STAGED_PART = re.compile(
    r"^\.part-(?P<hash>[0-9a-f]{64})\.(?P<state>complete|incomplete)\.staged\.parquet$"
)
FINAL_PART = re.compile(r"^part-(?P<hash>[0-9a-f]{64})\.(?P<state>complete|incomplete)\.parquet$")
METADATA_PART = re.compile(r"^part-(?P<hash>[0-9a-f]{64})\.metadata\.json$")


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
    run_id: str | None = None


class PartitionRecoveryIssue(BaseModel):
    """Stable machine-readable result from a raw-store recovery scan."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    code: str
    path: Path
    fatal: bool
    detail: str


class PartitionRecoveryReport(BaseModel):
    """The complete deterministic result of one startup recovery scan."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    valid_partitions: tuple[PartitionWriteResult, ...] = ()
    completed_staged_paths: tuple[Path, ...] = ()
    quarantined_paths: tuple[Path, ...] = ()
    issues: tuple[PartitionRecoveryIssue, ...] = ()

    @property
    def fatal_issues(self) -> tuple[PartitionRecoveryIssue, ...]:
        return tuple(issue for issue in self.issues if issue.fatal)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the host supports directory fsync."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems do not permit directory fsync. The file fsync and
        # atomic rename still provide the strongest locally available guarantee.
        pass
    finally:
        os.close(descriptor)


def atomic_replace(source: Path, destination: Path) -> None:
    os.replace(source, destination)
    fsync_directory(destination.parent)


def _safe(value: str) -> str:
    if not SAFE_PART.fullmatch(value):
        raise ValueError(f"unsafe partition identity: {value!r}")
    return value


def _event_type(event: RawEvent) -> str:
    names = {
        "RawCallbackEnvelopeEvent": "raw_callback_envelope_event",
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
        run_id: str | None = None,
        failure_injector: Callable[[str, Path], None] | None = None,
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
        self.run_id = run_id
        self.failure_injector = failure_injector

    def _checkpoint(self, phase: str, path: Path) -> None:
        if self.failure_injector is not None:
            self.failure_injector(phase, path)

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
        self._checkpoint("after_temporary_write", temporary)
        content_hash = sha256_path(temporary)
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
            if sha256_path(staging_path) != content_hash:
                raise RuntimeError("staged raw partition hash mismatch")
        elif not data_path.is_file():
            atomic_replace(temporary, staging_path)
            self._checkpoint("after_staging_rename", staging_path)
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
        if self.run_id is not None:
            payload["run_id"] = self.run_id
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
            self._checkpoint("after_metadata_temporary_write", metadata_temporary)
            atomic_replace(metadata_temporary, metadata_path)
            self._checkpoint("after_metadata_rename", metadata_path)
        if not data_path.is_file():
            if not staging_path.is_file():
                raise RuntimeError("raw partition staging evidence is absent")
            atomic_replace(staging_path, data_path)
            self._checkpoint("after_data_file_rename", data_path)
        else:
            staging_path.unlink(missing_ok=True)
            fsync_directory(staging_path.parent)
        result = self._load_result(metadata_path)
        if not self.verify(result):
            raise RuntimeError("final raw partition hash mismatch")
        self._checkpoint("after_partition_complete", data_path)
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
            run_id=None if payload.get("run_id") is None else str(payload["run_id"]),
        )

    @staticmethod
    def verify(result: PartitionWriteResult) -> bool:
        return (
            result.data_path.is_file()
            and result.metadata_path.is_file()
            and sha256_path(result.data_path) == result.content_hash
        )

    def _metadata_identity_issue(
        self,
        result: PartitionWriteResult,
        metadata_path: Path,
    ) -> PartitionRecoveryIssue | None:
        match = METADATA_PART.fullmatch(metadata_path.name)
        if match is None or match.group("hash") != result.content_hash:
            return PartitionRecoveryIssue(
                code="PARTITION_METADATA_HASH_IDENTITY_MISMATCH",
                path=metadata_path,
                fatal=True,
                detail=result.content_hash,
            )
        expected_state = "complete" if result.complete else "incomplete"
        expected_data = metadata_path.with_name(
            f"part-{result.content_hash}.{expected_state}.parquet"
        )
        if result.data_path != expected_data:
            return PartitionRecoveryIssue(
                code="PARTITION_METADATA_PATH_MISMATCH",
                path=metadata_path,
                fatal=True,
                detail=f"metadata={result.data_path};expected={expected_data}",
            )
        try:
            result.data_path.resolve(strict=False).relative_to(self.root.resolve(strict=False))
        except ValueError:
            return PartitionRecoveryIssue(
                code="PARTITION_PATH_OUTSIDE_RAW_ROOT",
                path=metadata_path,
                fatal=True,
                detail=str(result.data_path),
            )
        if (
            result.row_count <= 0
            or result.minimum_timestamp_utc.tzinfo is None
            or result.maximum_timestamp_utc.tzinfo is None
            or result.minimum_timestamp_utc > result.maximum_timestamp_utc
        ):
            return PartitionRecoveryIssue(
                code="PARTITION_METADATA_SEMANTICS_INVALID",
                path=metadata_path,
                fatal=True,
                detail="row count or timestamp bounds are invalid",
            )
        return None

    @staticmethod
    def _parquet_contract_issue(
        result: PartitionWriteResult,
    ) -> PartitionRecoveryIssue | None:
        try:
            import pyarrow.parquet as pq

            parquet_metadata = pq.read_metadata(result.data_path)  # type: ignore[no-untyped-call]
        except Exception as exc:
            return PartitionRecoveryIssue(
                code="COMPLETED_PARTITION_PARQUET_INVALID",
                path=result.data_path,
                fatal=True,
                detail=type(exc).__name__,
            )
        schema_metadata = parquet_metadata.metadata or {}
        observed_schema = schema_metadata.get(b"schema_version", b"").decode(errors="replace")
        observed_contract = schema_metadata.get(b"contract_version", b"").decode(errors="replace")
        if parquet_metadata.num_rows != result.row_count:
            return PartitionRecoveryIssue(
                code="PARTITION_ROW_COUNT_MISMATCH",
                path=result.data_path,
                fatal=True,
                detail=(f"metadata={result.row_count};parquet={parquet_metadata.num_rows}"),
            )
        if observed_schema != result.schema_version or observed_contract != result.contract_version:
            return PartitionRecoveryIssue(
                code="PARTITION_SCHEMA_CONTRACT_MISMATCH",
                path=result.data_path,
                fatal=True,
                detail=(f"schema={observed_schema};contract={observed_contract}"),
            )
        return None

    def recover(self) -> PartitionRecoveryReport:
        """Recover interrupted writes without inventing evidence.

        Temporary files with no committed metadata are quarantined so the
        durable callback inbox can replay them. A metadata/staged pair is safe
        to complete because the metadata fixes the final path and content hash.
        Corruption of a completed immutable partition is always fatal.
        """

        if not self.root.exists():
            return PartitionRecoveryReport()
        valid: list[PartitionWriteResult] = []
        completed: list[Path] = []
        quarantined: list[Path] = []
        issues: list[PartitionRecoveryIssue] = []
        metadata_by_hash: dict[str, Path] = {}

        for metadata_path in sorted(self.root.rglob("part-*.metadata.json")):
            match = METADATA_PART.fullmatch(metadata_path.name)
            if match is not None:
                metadata_by_hash[match.group("hash")] = metadata_path

        # First finish the only crash transition that has enough durable
        # information to prove its intended final identity.
        for staged_path in sorted(self.root.rglob(".part-*.staged.parquet")):
            match = STAGED_PART.fullmatch(staged_path.name)
            if match is None:
                quarantined.append(self._quarantine(staged_path, "UNRECOGNISED_STAGED_FILE"))
                issues.append(
                    PartitionRecoveryIssue(
                        code="UNRECOGNISED_STAGED_FILE",
                        path=staged_path,
                        fatal=False,
                        detail="staged filename does not encode an immutable hash",
                    )
                )
                continue
            expected_hash = match.group("hash")
            try:
                observed_hash = sha256_path(staged_path)
            except OSError as exc:
                issues.append(
                    PartitionRecoveryIssue(
                        code="STAGED_PARTITION_UNREADABLE",
                        path=staged_path,
                        fatal=False,
                        detail=type(exc).__name__,
                    )
                )
                continue
            staged_metadata_path = metadata_by_hash.get(expected_hash)
            if observed_hash != expected_hash:
                quarantined.append(self._quarantine(staged_path, "CORRUPT_STAGED_PARTITION"))
                issues.append(
                    PartitionRecoveryIssue(
                        code="CORRUPT_STAGED_PARTITION",
                        path=staged_path,
                        fatal=False,
                        detail=f"expected={expected_hash};observed={observed_hash}",
                    )
                )
                continue
            if staged_metadata_path is None:
                quarantined.append(self._quarantine(staged_path, "ORPHAN_STAGED_PARTITION"))
                issues.append(
                    PartitionRecoveryIssue(
                        code="ORPHAN_STAGED_PARTITION",
                        path=staged_path,
                        fatal=False,
                        detail="no committed metadata; durable inbox replay required",
                    )
                )
                continue
            try:
                result = self._load_result(staged_metadata_path)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                issues.append(
                    PartitionRecoveryIssue(
                        code="INVALID_PARTITION_METADATA",
                        path=staged_metadata_path,
                        fatal=True,
                        detail=type(exc).__name__,
                    )
                )
                continue
            identity_issue = self._metadata_identity_issue(
                result,
                staged_metadata_path,
            )
            if identity_issue is not None:
                issues.append(identity_issue)
                continue
            expected_final = staged_path.with_name(
                f"part-{expected_hash}.{match.group('state')}.parquet"
            )
            if result.data_path != expected_final:
                issues.append(
                    PartitionRecoveryIssue(
                        code="PARTITION_METADATA_PATH_MISMATCH",
                        path=staged_metadata_path,
                        fatal=True,
                        detail=f"metadata={result.data_path};staged={expected_final}",
                    )
                )
                continue
            if expected_final.is_file():
                if sha256_path(expected_final) != expected_hash:
                    issues.append(
                        PartitionRecoveryIssue(
                            code="CORRUPT_COMPLETED_PARTITION",
                            path=expected_final,
                            fatal=True,
                            detail=f"expected={expected_hash}",
                        )
                    )
                    continue
                staged_path.unlink(missing_ok=True)
                fsync_directory(staged_path.parent)
            else:
                atomic_replace(staged_path, expected_final)
                completed.append(expected_final)

        # Orphan writer temporaries never became committed evidence. Preserve
        # them for audit in quarantine and let the unacknowledged inbox replay.
        temporary_patterns = (".events-*.tmp.parquet", ".part-*.metadata.json.*.tmp")
        for pattern in temporary_patterns:
            for temporary_path in sorted(self.root.rglob(pattern)):
                quarantined_path = self._quarantine(
                    temporary_path,
                    "INTERRUPTED_PARTITION_TEMPORARY",
                )
                quarantined.append(quarantined_path)
                issues.append(
                    PartitionRecoveryIssue(
                        code="INTERRUPTED_PARTITION_TEMPORARY",
                        path=temporary_path,
                        fatal=False,
                        detail=f"quarantined={quarantined_path}",
                    )
                )

        for metadata_path in sorted(metadata_by_hash.values()):
            try:
                result = self._load_result(metadata_path)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                issues.append(
                    PartitionRecoveryIssue(
                        code="INVALID_PARTITION_METADATA",
                        path=metadata_path,
                        fatal=True,
                        detail=type(exc).__name__,
                    )
                )
                continue
            identity_issue = self._metadata_identity_issue(result, metadata_path)
            if identity_issue is not None:
                issues.append(identity_issue)
                continue
            if not result.data_path.is_file():
                issues.append(
                    PartitionRecoveryIssue(
                        code="PARTITION_DATA_MISSING",
                        path=result.data_path,
                        fatal=True,
                        detail=f"metadata={metadata_path}",
                    )
                )
                continue
            try:
                observed_hash = sha256_path(result.data_path)
            except OSError as exc:
                issues.append(
                    PartitionRecoveryIssue(
                        code="COMPLETED_PARTITION_UNREADABLE",
                        path=result.data_path,
                        fatal=True,
                        detail=type(exc).__name__,
                    )
                )
                continue
            if observed_hash != result.content_hash:
                issues.append(
                    PartitionRecoveryIssue(
                        code="CORRUPT_COMPLETED_PARTITION",
                        path=result.data_path,
                        fatal=True,
                        detail=f"expected={result.content_hash};observed={observed_hash}",
                    )
                )
                continue
            contract_issue = self._parquet_contract_issue(result)
            if contract_issue is not None:
                issues.append(contract_issue)
                continue
            valid.append(result)

        known_data_paths = {partition.data_path for partition in valid}
        for data_path in sorted(self.root.rglob("part-*.parquet")):
            match = FINAL_PART.fullmatch(data_path.name)
            if match is None or data_path in known_data_paths:
                continue
            issues.append(
                PartitionRecoveryIssue(
                    code="PARTITION_METADATA_MISSING",
                    path=data_path,
                    fatal=True,
                    detail="completed raw data lacks immutable metadata",
                )
            )

        return PartitionRecoveryReport(
            valid_partitions=tuple(
                sorted(valid, key=lambda item: (item.content_hash, str(item.data_path)))
            ),
            completed_staged_paths=tuple(completed),
            quarantined_paths=tuple(quarantined),
            issues=tuple(issues),
        )

    def _quarantine(self, path: Path, reason: str) -> Path:
        relative = path.relative_to(self.root)
        identity = hashlib.sha256(f"{relative}|{reason}".encode()).hexdigest()[:16]
        destination = self.root / "_quarantine" / reason / f"{identity}-{path.name.lstrip('.')}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if path.exists() and sha256_path(path) == sha256_path(destination):
                path.unlink(missing_ok=True)
                fsync_directory(path.parent)
                return destination
            raise RuntimeError(f"quarantine identity collision: {destination}")
        atomic_replace(path, destination)
        return destination


__all__ = [
    "PartitionRecoveryIssue",
    "PartitionRecoveryReport",
    "PartitionWriteResult",
    "PartitionedEventStore",
    "atomic_replace",
    "fsync_directory",
    "sha256_path",
]
