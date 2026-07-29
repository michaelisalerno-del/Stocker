from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.durable_inbox import DurableCallbackInbox
from stocker_prospective.events import UnderlyingLevel1QuoteEvent
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.storage_recovery import CrossStoreReconciler

NOW = datetime(2026, 7, 29, 14, 0, tzinfo=UTC)
RUN_ID = "run-storage-recovery"


class InjectedCrash(RuntimeError):
    pass


def metadata(run_id: str = RUN_ID) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=NOW - timedelta(days=1),
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="frozen-m1c",
        universe_id="anchor-frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[NOW.isoformat()],
        recorded_at_utc=NOW,
    )


def raw_event(sequence: int = 1) -> UnderlyingLevel1QuoteEvent:
    return UnderlyingLevel1QuoteEvent(
        event_id=f"quote-{sequence}",
        received_timestamp_utc=NOW + timedelta(seconds=sequence),
        received_monotonic_ns=1_000 + sequence,
        provider_timestamp_utc=NOW + timedelta(seconds=sequence),
        source_sequence=sequence,
        session=date(2026, 7, 29),
        symbol="AAL",
        con_id=1,
        request_id=10,
        bid=10.0,
        bid_size=100.0,
        ask=10.1,
        ask_size=100.0,
        last=None,
        last_size=None,
        volume=None,
        market_data_type=MarketDataType.LIVE,
        source="fake_ibkr",
        quote_valid=True,
        staleness_ms=0.0,
        tick_type="state_change",
        exchange="SMART",
    )


def migrated_repository(tmp_path: Path) -> ProspectiveRepository:
    repository = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    repository.migrate()
    repository.create_run(metadata())
    return repository


def event_store(
    tmp_path: Path,
    *,
    failure_phase: str | None = None,
    run_id: str = RUN_ID,
) -> PartitionedEventStore:
    def inject(phase: str, _: Path) -> None:
        if phase == failure_phase:
            raise InjectedCrash(phase)

    return PartitionedEventStore(
        root=tmp_path / "raw",
        prospective_collection_start=NOW - timedelta(days=1),
        recorder_version="test",
        contract_version="test",
        run_id=run_id,
        failure_injector=inject if failure_phase is not None else None,
    )


def write(store: PartitionedEventStore) -> None:
    store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(),),
        complete=True,
    )


def test_crash_after_temporary_write_quarantines_and_replays_without_duplicate(
    tmp_path: Path,
) -> None:
    with pytest.raises(InjectedCrash, match="after_temporary_write"):
        write(event_store(tmp_path, failure_phase="after_temporary_write"))

    restarted = event_store(tmp_path)
    recovery = restarted.recover()
    assert len(recovery.quarantined_paths) == 1
    assert recovery.valid_partitions == ()
    assert {issue.code for issue in recovery.issues} == {"INTERRUPTED_PARTITION_TEMPORARY"}

    first = restarted.write_events(
        data_source="fake_ibkr",
        events=(raw_event(),),
        complete=True,
    )
    retry = restarted.write_events(
        data_source="fake_ibkr",
        events=(raw_event(),),
        complete=True,
    )
    assert first.content_hash == retry.content_hash
    assert len(tuple((tmp_path / "raw").rglob("part-*.parquet"))) == 1


def test_pyarrow_write_exception_leaves_no_committed_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic pyarrow write failure")

    monkeypatch.setattr(pq, "write_table", fail_write)

    with pytest.raises(OSError, match="synthetic pyarrow write failure"):
        write(event_store(tmp_path))

    assert not tuple((tmp_path / "raw").rglob("part-*.parquet"))
    assert not tuple((tmp_path / "raw").rglob("part-*.metadata.json"))


def test_crash_after_metadata_temporary_write_is_quarantined_for_replay(
    tmp_path: Path,
) -> None:
    with pytest.raises(InjectedCrash, match="after_metadata_temporary_write"):
        write(event_store(tmp_path, failure_phase="after_metadata_temporary_write"))

    recovery = event_store(tmp_path).recover()

    assert recovery.valid_partitions == ()
    assert len(recovery.quarantined_paths) == 2
    assert {issue.code for issue in recovery.issues} == {
        "INTERRUPTED_PARTITION_TEMPORARY",
        "ORPHAN_STAGED_PARTITION",
    }


def test_crash_after_metadata_rename_completes_staged_partition(
    tmp_path: Path,
) -> None:
    with pytest.raises(InjectedCrash, match="after_metadata_rename"):
        write(event_store(tmp_path, failure_phase="after_metadata_rename"))

    recovery = event_store(tmp_path).recover()
    assert recovery.fatal_issues == ()
    assert len(recovery.completed_staged_paths) == 1
    assert len(recovery.valid_partitions) == 1


def test_crash_after_data_rename_is_already_valid(tmp_path: Path) -> None:
    with pytest.raises(InjectedCrash, match="after_data_file_rename"):
        write(event_store(tmp_path, failure_phase="after_data_file_rename"))

    recovery = event_store(tmp_path).recover()
    assert recovery.fatal_issues == ()
    assert len(recovery.valid_partitions) == 1


def test_corrupt_staged_partition_is_quarantined_and_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(InjectedCrash, match="after_metadata_rename"):
        write(event_store(tmp_path, failure_phase="after_metadata_rename"))
    staged = next((tmp_path / "raw").rglob(".part-*.staged.parquet"))
    staged.write_bytes(b"corrupt staged evidence")

    recovery = event_store(tmp_path).recover()

    assert len(recovery.quarantined_paths) == 1
    assert "CORRUPT_STAGED_PARTITION" in {issue.code for issue in recovery.issues}
    assert "PARTITION_DATA_MISSING" in {issue.code for issue in recovery.fatal_issues}


def test_valid_partition_without_sqlite_manifest_is_registered_on_restart(
    tmp_path: Path,
) -> None:
    repository = migrated_repository(tmp_path)
    store = event_store(tmp_path)
    result = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(),),
        complete=True,
    )
    inbox = DurableCallbackInbox(
        repository.database_path,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
    )
    report = CrossStoreReconciler(
        repository=repository,
        recorder_repository=FrozenRecorderRepository(repository),
        raw_store=store,
        inbox=inbox,
        run_id=RUN_ID,
        recorder_generation=1,
    ).reconcile(metadata(), observed_at=NOW)

    assert report.safe_to_score
    assert report.manifests_registered == (result.content_hash,)
    with repository._connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM raw_partition_manifest_v0 WHERE run_id = ?",
            (RUN_ID,),
        ).fetchone()[0]
    assert count == 1


def test_corrupt_completed_partition_latches_storage_fatal(tmp_path: Path) -> None:
    repository = migrated_repository(tmp_path)
    store = event_store(tmp_path)
    result = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(),),
        complete=True,
    )
    FrozenRecorderRepository(repository).record_partition(
        metadata(),
        data_source="fake_ibkr",
        session_date=date(2026, 7, 29),
        symbol="AAL",
        event_type="underlying_level1_quote_event",
        partition=result,
    )
    result.data_path.write_bytes(b"corrupt")
    inbox = DurableCallbackInbox(
        repository.database_path,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
    )

    report = CrossStoreReconciler(
        repository=repository,
        recorder_repository=FrozenRecorderRepository(repository),
        raw_store=store,
        inbox=inbox,
        run_id=RUN_ID,
        recorder_generation=1,
    ).reconcile(metadata(), observed_at=NOW)

    assert not report.safe_to_score
    assert "CORRUPT_COMPLETED_PARTITION" in {issue.code for issue in report.fatal_issues}
    assert inbox.has_active_fatal("storage")


def test_manifest_pointing_to_missing_partition_latches_storage_fatal(
    tmp_path: Path,
) -> None:
    repository = migrated_repository(tmp_path)
    store = event_store(tmp_path)
    result = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(),),
        complete=True,
    )
    FrozenRecorderRepository(repository).record_partition(
        metadata(),
        data_source="fake_ibkr",
        session_date=date(2026, 7, 29),
        symbol="AAL",
        event_type="underlying_level1_quote_event",
        partition=result,
    )
    result.data_path.unlink()
    inbox = DurableCallbackInbox(
        repository.database_path,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
    )

    report = CrossStoreReconciler(
        repository=repository,
        recorder_repository=FrozenRecorderRepository(repository),
        raw_store=store,
        inbox=inbox,
        run_id=RUN_ID,
        recorder_generation=1,
    ).reconcile(metadata(), observed_at=NOW)

    assert not report.safe_to_score
    assert "MANIFEST_PARTITION_MISSING" in {issue.code for issue in report.fatal_issues}
    assert inbox.has_active_fatal("storage")


def test_reconciliation_validates_manifests_from_prior_runs(
    tmp_path: Path,
) -> None:
    repository = migrated_repository(tmp_path)
    prior_run_id = "prior-invalid-run"
    prior_metadata = metadata(prior_run_id)
    repository.create_run(prior_metadata)
    prior_store = event_store(tmp_path, run_id=prior_run_id)
    prior_partition = prior_store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(2),),
        complete=True,
    )
    FrozenRecorderRepository(repository).record_partition(
        prior_metadata,
        data_source="fake_ibkr",
        session_date=date(2026, 7, 29),
        symbol="AAL",
        event_type="underlying_level1_quote_event",
        partition=prior_partition,
    )
    prior_partition.data_path.unlink()
    prior_partition.metadata_path.unlink()
    inbox = DurableCallbackInbox(
        repository.database_path,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="recorder",
    )

    report = CrossStoreReconciler(
        repository=repository,
        recorder_repository=FrozenRecorderRepository(repository),
        raw_store=event_store(tmp_path),
        inbox=inbox,
        run_id=RUN_ID,
        recorder_generation=1,
    ).reconcile(metadata(), observed_at=NOW)

    assert not report.safe_to_score
    assert "MANIFEST_PARTITION_MISSING" in {issue.code for issue in report.fatal_issues}
    assert inbox.has_active_fatal("storage")


def test_sidecar_cannot_redirect_a_hash_valid_partition_path(
    tmp_path: Path,
) -> None:
    store = event_store(tmp_path)
    result = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(),),
        complete=True,
    )
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    payload["data_path"] = str(tmp_path / "redirected.parquet")
    result.metadata_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    recovery = store.recover()

    assert "PARTITION_METADATA_PATH_MISMATCH" in {issue.code for issue in recovery.fatal_issues}
    assert recovery.valid_partitions == ()


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("row_count", 2, "PARTITION_ROW_COUNT_MISMATCH"),
        (
            "contract_version",
            "tampered-contract",
            "PARTITION_SCHEMA_CONTRACT_MISMATCH",
        ),
    ],
)
def test_sidecar_row_and_contract_claims_are_checked_against_parquet(
    tmp_path: Path,
    field: str,
    value: object,
    expected_code: str,
) -> None:
    store = event_store(tmp_path)
    result = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(),),
        complete=True,
    )
    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    payload[field] = value
    result.metadata_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    recovery = store.recover()

    assert expected_code in {issue.code for issue in recovery.fatal_issues}
    assert recovery.valid_partitions == ()
