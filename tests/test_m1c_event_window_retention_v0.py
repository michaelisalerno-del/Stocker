from __future__ import annotations

import csv
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from stocker_prospective.config import EventWindowRetentionV0Config
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.events import RawCallbackEnvelopeEvent, UnderlyingLevel1QuoteEvent
from stocker_prospective.m1c_event_window_retention_v0 import (
    M1CEventReferenceV0,
    SessionEventWindowFinalizerV0,
    SessionEventWindowPlanV0,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository

SESSION = date(2026, 8, 17)


def _event(episode_id: str, symbol: str, minute: int) -> M1CEventReferenceV0:
    return M1CEventReferenceV0(
        episode_id=episode_id,
        symbol=symbol,
        session=SESSION,
        t0=datetime(2026, 8, 17, 14, minute, tzinfo=UTC),
    )


def test_all_m1c_events_create_merged_symbol_windows() -> None:
    plan = SessionEventWindowPlanV0.from_events(
        run_id="run-retention",
        session=SESSION,
        events=(
            _event("r01", "AAL", 0),
            _event("p01", "AAL", 10),
            _event("hard", "SOFI", 30),
        ),
    )

    assert plan.episode_ids == ("hard", "p01", "r01")
    assert [window.model_dump(mode="json") for window in plan.windows] == [
        {
            "symbol": "AAL",
            "start_utc": "2026-08-17T13:55:00Z",
            "end_utc": "2026-08-17T14:30:00Z",
            "episode_ids": ["p01", "r01"],
        },
        {
            "symbol": "SOFI",
            "start_utc": "2026-08-17T14:25:00Z",
            "end_utc": "2026-08-17T14:50:00Z",
            "episode_ids": ["hard"],
        },
    ]


def test_retention_is_disabled_by_default_and_frozen_when_enabled() -> None:
    assert EventWindowRetentionV0Config().enabled is False
    with pytest.raises(ValidationError, match="requires activation and frozen contract"):
        EventWindowRetentionV0Config(enabled=True)
    with pytest.raises(ValidationError, match="new dataset version"):
        EventWindowRetentionV0Config(before_event_minutes=4)


def _metadata() -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id="run-retention",
        prospective_start_utc=datetime(2026, 8, 1, tzinfo=UTC),
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="frozen-m1c",
        universe_id="anchor-frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=["2026-08-17T14:00:00+00:00"],
        recorded_at_utc=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
    )


def _quote(event_id: str, observed_at: datetime, sequence: int) -> UnderlyingLevel1QuoteEvent:
    return UnderlyingLevel1QuoteEvent(
        event_id=event_id,
        received_timestamp_utc=observed_at,
        received_monotonic_ns=sequence,
        provider_timestamp_utc=observed_at,
        source_sequence=sequence,
        session=SESSION,
        symbol="AAL",
        con_id=1,
        request_id=10,
        bid=10.0,
        bid_size=100.0,
        ask=10.1,
        ask_size=100.0,
        last=None,
        last_size=None,
        market_data_type=MarketDataType.LIVE,
        source="fake_ibkr",
        quote_valid=True,
        tick_type="state_change",
        exchange="SMART",
    )


def _raw_callback(event_id: str, *, stream_kind: str) -> RawCallbackEnvelopeEvent:
    observed = datetime(2026, 8, 17, 14, 21, tzinfo=UTC)
    return RawCallbackEnvelopeEvent(
        event_id=event_id,
        inbox_event_id=f"inbox-{event_id}",
        received_timestamp_utc=observed,
        received_monotonic_ns=100,
        original_received_monotonic_ns=100,
        provider_timestamp_utc=observed,
        source_sequence=100 if stream_kind == "underlying_level1" else 101,
        session=SESSION,
        symbol="AAL",
        subscription_symbol="AAL",
        con_id=1,
        request_id=10,
        callback_kind="official_provider_tick_price",
        connection_generation=1,
        callback_classification="accepted_active_callback",
        subscription_owner="test:AAL",
        stream_owner={
            "request_id": 10,
            "kind": stream_kind,
            "symbol": "AAL",
            "con_id": 1,
            "exchange": "SMART",
            "option_contract": None,
            "episode_id": None,
        },
        original_payload={"field": "bid", "value": 10.0},
        admission_run_id="run-retention",
        admission_recorder_generation=1,
        recovery_disposition="original_provider_callback",
    )


def _database_with_episode_and_partition(tmp_path: Path) -> tuple[Path, Path, Path]:
    database = tmp_path / "prospective.sqlite3"
    repository = ProspectiveRepository(database)
    repository.migrate()
    repository.create_run(_metadata())
    raw_root = tmp_path / "raw"
    store = PartitionedEventStore(
        root=raw_root,
        prospective_collection_start=datetime(2026, 8, 1, tzinfo=UTC),
        recorder_version="test",
        contract_version="test",
        run_id="run-retention",
    )
    partition = store.write_events(
        data_source="fake_ibkr",
        events=(
            _quote("outside", datetime(2026, 8, 17, 14, 21, tzinfo=UTC), 1),
            _quote("inside", datetime(2026, 8, 17, 14, 1, tzinfo=UTC), 2),
        ),
        complete=True,
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO m1c_episode_v0(
                episode_id, envelope_id, checkpoint_id, run_id, symbol,
                session_date, trigger_checkpoint, trigger_bar_end_utc,
                prospective_entry_timestamp_utc, m1c_probability,
                previous_m1c_probability, episode_number,
                minutes_since_previous_episode, scientific_recording_valid,
                rejection_reasons_json, phase, completion_status,
                completed_at_utc, claims_json
            ) VALUES (?, 1, 1, ?, 'AAL', ?, 6, ?, ?, 0.8, NULL, 1,
                      0.0, 1, '[]', 'option_development', 'complete', ?, '{}')
            """,
            (
                "all-family-event",
                "run-retention",
                SESSION.isoformat(),
                datetime(2026, 8, 17, 14, 0, tzinfo=UTC).isoformat(),
                datetime(2026, 8, 17, 14, 0, tzinfo=UTC).isoformat(),
                datetime(2026, 8, 17, 14, 15, tzinfo=UTC).isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO raw_partition_manifest_v0(
                run_id, data_source, session_date, symbol, event_type,
                file_path, row_count, minimum_timestamp_utc,
                maximum_timestamp_utc, schema_version, content_hash,
                complete, gap_count, recorder_version, contract_version,
                recorded_at_utc, claims_json
            ) VALUES (?, 'fake_ibkr', ?, 'AAL', 'underlying_level1_quote_event',
                      ?, ?, ?, ?, ?, ?, 1, 0, 'test', 'test', ?, '{}')
            """,
            (
                "run-retention",
                SESSION.isoformat(),
                str(partition.data_path),
                partition.row_count,
                partition.minimum_timestamp_utc.isoformat(),
                partition.maximum_timestamp_utc.isoformat(),
                partition.schema_version,
                partition.content_hash,
                _metadata().recorded_at_utc.isoformat(),
            ),
        )
    return database, raw_root, partition.data_path


def test_finalizer_verifies_retained_rows_before_deleting_source(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=tmp_path / "retained",
        run_id="run-retention",
    )

    receipt = finalizer.finalize_session(
        session=SESSION,
        observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
    )

    assert receipt.status == "COMPLETE"
    assert receipt.episode_ids == ("all-family-event",)
    assert receipt.source_row_count == 2
    assert receipt.retained_row_count == 1
    assert receipt.summarised_row_count == 1
    assert receipt.source_partitions_deleted == 1
    assert not source_path.exists()
    assert not source_path.with_name(
        f"part-{receipt.partitions[0].source_content_hash}.metadata.json"
    ).exists()
    retained = pq.ParquetFile(receipt.partitions[0].retained_file_path).read().to_pylist()
    assert [row["event_id"] for row in retained] == ["inside"]
    with receipt.summary_file_path.open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert summary[0]["provider_timestamp_available_rows"] == "1"
    assert summary[0]["first_summarised_event_utc"] == "2026-08-17T14:21:00+00:00"
    assert summary[0]["locked_quote_count"] == "0"
    assert summary[0]["crossed_quote_count"] == "0"
    assert summary[0]["invalid_quote_count"] == "0"
    assert finalizer.repository.session_receipt(SESSION) == receipt


def test_finalizer_keeps_source_when_hash_verification_fails(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)
    source_path.write_bytes(source_path.read_bytes() + b"corrupt")
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=tmp_path / "retained",
        run_id="run-retention",
    )

    with pytest.raises(RuntimeError, match="RETENTION_SOURCE_HASH_MISMATCH"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
        )

    assert source_path.exists()
    assert finalizer.repository.session_receipt(SESSION) is None


def test_finalizer_refuses_to_run_before_every_event_horizon_is_closed(
    tmp_path: Path,
) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=tmp_path / "retained",
        run_id="run-retention",
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_HORIZON_OPEN"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 20, 10, tzinfo=UTC),
        )

    assert source_path.exists()


def test_finalizer_never_deletes_partition_referenced_by_pending_callback(
    tmp_path: Path,
) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)
    with sqlite3.connect(database) as connection:
        content_hash = str(
            connection.execute(
                "SELECT content_hash FROM raw_partition_manifest_v0"
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO callback_inbox_v1(
                inbox_event_id, callback_kind, request_id, received_utc,
                received_monotonic_ns, provider_timestamp_utc,
                original_payload_json, admission_run_id,
                admission_recorder_generation, connection_generation,
                subscription_owner, symbol, callback_classification,
                status, admitted_at_utc, updated_at_utc
            ) VALUES (
                'pending-callback', 'level1_quote_update', 10, ?, 1, ?, '{}',
                'run-retention', 1, 1, 'universe:AAL', 'AAL',
                'accepted_active_callback', 'pending', ?, ?
            )
            """,
            (
                datetime(2026, 8, 17, 14, 1, tzinfo=UTC).isoformat(),
                datetime(2026, 8, 17, 14, 1, tzinfo=UTC).isoformat(),
                datetime(2026, 8, 17, 14, 1, tzinfo=UTC).isoformat(),
                datetime(2026, 8, 17, 14, 1, tzinfo=UTC).isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO callback_raw_materialization_v1(
                inbox_event_id, source_sequence, run_id, recorder_generation,
                lease_batch_id, raw_partition_hashes_json, raw_event_ids_json,
                materialized_at_utc
            ) VALUES ('pending-callback', 1, 'run-retention', 1, 'batch-1', ?,
                      '["inside"]', ?)
            """,
            (
                json.dumps([content_hash]),
                datetime(2026, 8, 17, 14, 2, tzinfo=UTC).isoformat(),
            ),
        )
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=tmp_path / "retained",
        run_id="run-retention",
    )

    with pytest.raises(RuntimeError, match="RETENTION_SOURCE_CALLBACK_NOT_TERMINAL"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
        )

    assert source_path.exists()


def test_restart_completes_prepared_retention_after_source_unlink(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)

    def crash(phase: str, _path: Path) -> None:
        if phase == "after_source_data_unlink":
            raise RuntimeError("injected-retention-crash")

    interrupted = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=tmp_path / "retained",
        run_id="run-retention",
        failure_injector=crash,
    )
    with pytest.raises(RuntimeError, match="injected-retention-crash"):
        interrupted.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
        )
    assert not source_path.exists()
    prepared = interrupted.repository.session_receipt(SESSION)
    assert prepared is not None
    assert prepared.status == "PREPARED"

    restarted = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=tmp_path / "retained",
        run_id="run-retention",
    )
    completed = restarted.finalize_session(
        session=SESSION,
        observed_at=datetime(2026, 8, 17, 22, 1, tzinfo=UTC),
    )

    assert completed.status == "COMPLETE"
    assert completed.source_partitions_deleted == 1


def test_completed_session_rejects_late_market_partition(tmp_path: Path) -> None:
    database, raw_root, _ = _database_with_episode_and_partition(tmp_path)
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=tmp_path / "retained",
        run_id="run-retention",
    )
    finalizer.finalize_session(
        session=SESSION,
        observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
    )
    late = PartitionedEventStore(
        root=raw_root,
        prospective_collection_start=datetime(2026, 8, 1, tzinfo=UTC),
        recorder_version="test",
        contract_version="test",
        run_id="run-retention",
    ).write_events(
        data_source="fake_ibkr",
        events=(_quote("late", datetime(2026, 8, 17, 14, 22, tzinfo=UTC), 3),),
        complete=True,
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_ALREADY_SEALED"):
        FrozenRecorderRepository(ProspectiveRepository(database)).record_partition(
            _metadata(),
            data_source="fake_ibkr",
            session_date=SESSION,
            symbol="AAL",
            event_type="underlying_level1_quote_event",
            partition=late,
        )


def test_raw_callback_compaction_preserves_non_underlying_evidence(tmp_path: Path) -> None:
    database, raw_root, _ = _database_with_episode_and_partition(tmp_path)
    callback_partition = PartitionedEventStore(
        root=raw_root,
        prospective_collection_start=datetime(2026, 8, 1, tzinfo=UTC),
        recorder_version="test",
        contract_version="test",
        run_id="run-retention",
    ).write_events(
        data_source="fake_ibkr",
        events=(
            _raw_callback("underlying-outside", stream_kind="underlying_level1"),
            _raw_callback("option-outside", stream_kind="option_quote"),
        ),
        complete=True,
    )
    FrozenRecorderRepository(ProspectiveRepository(database)).record_partition(
        _metadata(),
        data_source="fake_ibkr",
        session_date=SESSION,
        symbol="AAL",
        event_type="raw_callback_envelope_event",
        partition=callback_partition,
    )
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=tmp_path / "retained",
        run_id="run-retention",
    )

    receipt = finalizer.finalize_session(
        session=SESSION,
        observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
    )

    event_ids = {
        str(row["event_id"])
        for partition in receipt.partitions
        if partition.retained_file_path is not None
        for row in pq.ParquetFile(partition.retained_file_path).read().to_pylist()
    }
    assert "option-outside" in event_ids
    assert "underlying-outside" not in event_ids
