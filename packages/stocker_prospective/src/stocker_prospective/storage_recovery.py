"""Startup reconciliation between immutable raw files and SQLite manifests."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.durable_inbox import DurableCallbackInbox
from stocker_prospective.partition_store import (
    PartitionedEventStore,
    PartitionRecoveryIssue,
    PartitionRecoveryReport,
    sha256_path,
)
from stocker_prospective.recorder_repository import FrozenRecorderRepository


class CrossStoreRecoveryReport(BaseModel):
    """Evidence-based startup result; any fatal issue blocks scoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    raw_store: PartitionRecoveryReport
    manifests_registered: tuple[str, ...] = ()
    already_registered: tuple[str, ...] = ()
    fatal_issues: tuple[PartitionRecoveryIssue, ...] = ()

    @property
    def safe_to_score(self) -> bool:
        return not self.fatal_issues


def _partition_identity(path: Path) -> tuple[str, date, str, str]:
    values: dict[str, str] = {}
    for part in path.parts:
        if "=" in part:
            key, value = part.split("=", 1)
            values[key] = value
    required = ("data_source", "session_date", "symbol", "event_type")
    missing = tuple(key for key in required if not values.get(key))
    if missing:
        raise ValueError(f"partition path identity missing: {','.join(missing)}")
    return (
        values["data_source"],
        date.fromisoformat(values["session_date"]),
        values["symbol"],
        values["event_type"],
    )


class CrossStoreReconciler:
    """Repair interrupted registration and fail closed on broken manifests."""

    def __init__(
        self,
        *,
        repository: ProspectiveRepository,
        recorder_repository: FrozenRecorderRepository,
        raw_store: PartitionedEventStore,
        inbox: DurableCallbackInbox,
        run_id: str,
        recorder_generation: int,
    ) -> None:
        self.repository = repository
        self.recorder_repository = recorder_repository
        self.raw_store = raw_store
        self.inbox = inbox
        self.run_id = run_id
        self.recorder_generation = recorder_generation

    def reconcile(
        self,
        metadata: EvidenceMetadata,
        *,
        observed_at: datetime,
    ) -> CrossStoreRecoveryReport:
        observed = observed_at.astimezone(UTC)
        raw_report = self.raw_store.recover()
        fatal = list(raw_report.fatal_issues)
        registered: list[str] = []
        existing: list[str] = []
        manifests = self._manifests()
        manifest_by_hash = {
            str(row["content_hash"]): row for row in manifests if str(row["run_id"]) == self.run_id
        }

        for partition in raw_report.valid_partitions:
            if partition.run_id not in {None, self.run_id}:
                continue
            manifest = manifest_by_hash.get(partition.content_hash)
            if manifest is not None:
                existing.append(partition.content_hash)
                continue
            if partition.run_id is None:
                # Pre-hardening metadata does not prove which run owns an
                # unregistered file, so it is never silently attached.
                continue
            try:
                data_source, session_date, symbol, event_type = _partition_identity(
                    partition.data_path
                )
                self.recorder_repository.record_partition(
                    metadata,
                    data_source=data_source,
                    session_date=session_date,
                    symbol=symbol,
                    event_type=event_type,
                    partition=partition,
                )
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                fatal.append(
                    PartitionRecoveryIssue(
                        code="RAW_MANIFEST_REGISTRATION_FAILED",
                        path=partition.data_path,
                        fatal=True,
                        detail=type(exc).__name__,
                    )
                )
            else:
                registered.append(partition.content_hash)

        valid_by_hash = {
            partition.content_hash: partition for partition in raw_report.valid_partitions
        }
        for manifest in manifests:
            content_hash = str(manifest["content_hash"])
            path = Path(str(manifest["file_path"]))
            try:
                path.resolve(strict=False).relative_to(self.raw_store.root.resolve(strict=False))
            except ValueError:
                fatal.append(
                    PartitionRecoveryIssue(
                        code="MANIFEST_PARTITION_OUTSIDE_RAW_ROOT",
                        path=path,
                        fatal=True,
                        detail=f"content_hash={content_hash}",
                    )
                )
                continue
            if not path.is_file():
                fatal.append(
                    PartitionRecoveryIssue(
                        code="MANIFEST_PARTITION_MISSING",
                        path=path,
                        fatal=True,
                        detail=f"content_hash={content_hash}",
                    )
                )
                continue
            try:
                observed_hash = sha256_path(path)
            except OSError as exc:
                fatal.append(
                    PartitionRecoveryIssue(
                        code="MANIFEST_PARTITION_UNREADABLE",
                        path=path,
                        fatal=True,
                        detail=type(exc).__name__,
                    )
                )
                continue
            if observed_hash != content_hash:
                fatal.append(
                    PartitionRecoveryIssue(
                        code="MANIFEST_PARTITION_HASH_INVALID",
                        path=path,
                        fatal=True,
                        detail=f"expected={content_hash};observed={observed_hash}",
                    )
                )
                continue
            scanned = valid_by_hash.get(content_hash)
            if scanned is not None and scanned.data_path != path:
                fatal.append(
                    PartitionRecoveryIssue(
                        code="MANIFEST_PARTITION_PATH_MISMATCH",
                        path=path,
                        fatal=True,
                        detail=f"metadata={scanned.data_path}",
                    )
                )

        deduplicated_fatal = tuple(
            {(issue.code, str(issue.path), issue.detail): issue for issue in fatal}.values()
        )
        for issue in deduplicated_fatal:
            self.inbox.record_incident(
                stable_error_code=issue.code,
                component="raw_storage_reconciliation",
                severity="fatal",
                occurred_at=observed,
                error_class="StorageIntegrityError",
                evidence_loss_possible=True,
                details={"path": str(issue.path), "detail": issue.detail},
            )
        if deduplicated_fatal:
            first = deduplicated_fatal[0]
            self.inbox.latch_fatal(
                latch_kind="storage",
                stable_error_code=first.code,
                occurred_at=observed,
                error_class="StorageIntegrityError",
                evidence_loss_possible=True,
            )

        return CrossStoreRecoveryReport(
            raw_store=raw_report,
            manifests_registered=tuple(sorted(registered)),
            already_registered=tuple(sorted(existing)),
            fatal_issues=deduplicated_fatal,
        )

    def _manifests(self) -> tuple[sqlite3.Row, ...]:
        with self.repository._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, file_path, content_hash
                FROM raw_partition_manifest_v0
                ORDER BY run_id, content_hash, file_path
                """
            ).fetchall()
        return tuple(rows)


__all__ = ["CrossStoreReconciler", "CrossStoreRecoveryReport"]
