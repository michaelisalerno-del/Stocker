from __future__ import annotations

import csv
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from stocker_prospective.cli import _oldest_pending_completed_retention_session
from stocker_prospective.config import EventWindowRetentionV0Config
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.durable_inbox import (
    CallbackClassification,
    CallbackInboxError,
    DurableCallbackInbox,
)
from stocker_prospective.event_window_retention_scope_v0 import (
    RETENTION_CONTROLLED_EVENT_TYPES,
)
from stocker_prospective.events import RawCallbackEnvelopeEvent, UnderlyingLevel1QuoteEvent
from stocker_prospective.m1c_event_window_retention_v0 import (
    M1CEventReferenceV0,
    SessionEventWindowFinalizerV0,
    SessionEventWindowPlanV0,
    eligible_retention_sessions,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository

SESSION = date(2026, 8, 17)
ACTIVATION = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


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
    with pytest.raises(ValidationError, match="requires schedule, frozen contract"):
        EventWindowRetentionV0Config(enabled=True)
    with pytest.raises(ValidationError, match="new dataset version"):
        EventWindowRetentionV0Config(before_event_minutes=4)


def test_retention_schedule_is_exactly_twenty_xnys_sessions() -> None:
    sessions = eligible_retention_sessions(first_session=SESSION, count=20)

    assert len(sessions) == 20
    assert sessions[0] == SESSION
    assert sessions[-1] > sessions[0]
    assert len(set(sessions)) == 20


def test_depth_is_in_high_volume_retention_scope() -> None:
    assert "underlying_depth_event" in RETENTION_CONTROLLED_EVENT_TYPES
    assert "underlying_depth_snapshot" in RETENTION_CONTROLLED_EVENT_TYPES


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


def _database_with_episode_and_partition(
    tmp_path: Path,
    *,
    all_inside: bool = False,
) -> tuple[Path, Path, Path]:
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
            _quote(
                "outside" if not all_inside else "inside-two",
                datetime(2026, 8, 17, 14, 2 if all_inside else 21, tzinfo=UTC),
                1,
            ),
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
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
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
    recovery = PartitionedEventStore(
        root=raw_root,
        prospective_collection_start=datetime(2026, 8, 1, tzinfo=UTC),
        recorder_version="test",
        contract_version="test",
        run_id="run-retention",
    ).recover()
    assert recovery.fatal_issues == ()
    assert receipt.partitions[0].retained_content_hash in {
        partition.content_hash for partition in recovery.valid_partitions
    }
    assert finalizer.repository.session_receipt(SESSION) == receipt


def test_finalizer_keeps_source_when_hash_verification_fails(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)
    source_path.write_bytes(source_path.read_bytes() + b"corrupt")
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
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
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_HORIZON_OPEN"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 20, 10, tzinfo=UTC),
        )

    assert source_path.exists()


def test_finalizer_never_applies_retention_before_activation(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=datetime(2026, 8, 18, tzinfo=UTC),
        first_eligible_session=SESSION,
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_PRECEDES_ACTIVATION"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
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
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_CALLBACK_NOT_TERMINAL"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
        )

    assert source_path.exists()


def test_finalizer_blocks_unmaterialized_session_callback(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)
    with sqlite3.connect(database) as connection:
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
                'unmaterialized-callback', 'level1_quote_update', 10, ?, 1, ?, '{}',
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
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_CALLBACK_NOT_TERMINAL"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
        )

    assert source_path.exists()


def test_after_hours_callback_after_utc_midnight_still_blocks_prior_ny_session(
    tmp_path: Path,
) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)
    # 00:30 UTC on August 18 is still August 17 in America/New_York.
    received = datetime(2026, 8, 18, 0, 30, tzinfo=UTC).isoformat()
    with sqlite3.connect(database) as connection:
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
                'after-midnight-callback', 'level1_quote_update', 10, ?, 1, NULL,
                '{}', 'run-retention', 1, 1, 'universe:AAL', 'AAL',
                'accepted_active_callback', 'pending', ?, ?
            )
            """,
            (received, received, received),
        )
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_CALLBACK_NOT_TERMINAL"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 18, 2, 0, tzinfo=UTC),
        )

    assert source_path.exists()


def test_completed_session_purges_terminal_high_volume_inbox_rows(tmp_path: Path) -> None:
    database, raw_root, _ = _database_with_episode_and_partition(tmp_path)
    with sqlite3.connect(database) as connection:
        content_hash = str(
            connection.execute(
                "SELECT content_hash FROM raw_partition_manifest_v0"
            ).fetchone()[0]
        )
        observed = datetime(2026, 8, 17, 14, 1, tzinfo=UTC).isoformat()
        hashes = json.dumps([content_hash])
        connection.execute(
            """
            INSERT INTO callback_inbox_v1(
                inbox_event_id, callback_kind, request_id, received_utc,
                received_monotonic_ns, provider_timestamp_utc,
                original_payload_json, admission_run_id,
                admission_recorder_generation, connection_generation,
                subscription_owner, symbol, callback_classification,
                status, acknowledgement_timestamp_utc, admitted_at_utc, updated_at_utc
            ) VALUES (
                'terminal-callback', 'level1_quote_update', 10, ?, 1, ?, '{}',
                'run-retention', 1, 1, 'universe:AAL', 'AAL',
                'accepted_active_callback', 'acknowledged', ?, ?, ?
            )
            """,
            (observed, observed, observed, observed, observed),
        )
        connection.execute(
            """
            INSERT INTO callback_raw_materialization_v1(
                inbox_event_id, source_sequence, run_id, recorder_generation,
                lease_batch_id, raw_partition_hashes_json, raw_event_ids_json,
                materialized_at_utc
            ) VALUES ('terminal-callback', 1, 'run-retention', 1, 'batch-1', ?,
                      '["inside"]', ?)
            """,
            (hashes, observed),
        )
        connection.execute(
            """
            INSERT INTO callback_processing_commit_v1(
                inbox_event_id, source_sequence, run_id, recorder_generation,
                raw_partition_hashes_json, committed_at_utc
            ) VALUES ('terminal-callback', 1, 'run-retention', 1, ?, ?)
            """,
            (hashes, observed),
        )
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
    )

    finalizer.finalize_session(
        session=SESSION,
        observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
    )

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM callback_inbox_v1 WHERE inbox_event_id = 'terminal-callback'"
        ).fetchone()[0] == 0
        purge = connection.execute(
            """
            SELECT purged_callback_count
            FROM m1c_event_window_callback_purge_v0
            WHERE run_id = 'run-retention' AND session_date = ?
            """,
            (SESSION.isoformat(),),
        ).fetchone()
    assert purge == (1,)

    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM m1c_event_window_callback_purge_v0 WHERE run_id = 'run-retention'"
        )
    assert _oldest_pending_completed_retention_session(
        finalizer,
        datetime(2026, 8, 18, 2, 0, tzinfo=UTC),
    ) == SESSION


def test_diagnostic_provider_envelope_is_terminal_and_purged_with_canonical_row(
    tmp_path: Path,
) -> None:
    database, raw_root, _ = _database_with_episode_and_partition(tmp_path)
    with sqlite3.connect(database) as connection:
        content_hash = str(
            connection.execute(
                "SELECT content_hash FROM raw_partition_manifest_v0"
            ).fetchone()[0]
        )
        observed = datetime(2026, 8, 17, 14, 1, tzinfo=UTC).isoformat()
        hashes = json.dumps([content_hash])
        connection.execute(
            """
            INSERT INTO callback_inbox_v1(
                inbox_event_id, callback_kind, request_id, received_utc,
                received_monotonic_ns, provider_timestamp_utc,
                original_payload_json, admission_run_id,
                admission_recorder_generation, connection_generation,
                subscription_owner, symbol, callback_classification,
                status, acknowledgement_timestamp_utc, failure_classification,
                admitted_at_utc, updated_at_utc
            ) VALUES (
                'provider-envelope', 'official_provider_tick_by_tick_bidask', 10,
                ?, 1, ?, '{}', 'run-retention', 1, 1, 'universe:AAL', 'AAL',
                'accepted_active_callback', 'diagnostic', ?,
                'PROVIDER_ENVELOPE_MATERIALIZED:canonical-callback', ?, ?
            )
            """,
            (observed, observed, observed, observed, observed),
        )
        connection.execute(
            """
            INSERT INTO callback_inbox_v1(
                inbox_event_id, callback_kind, request_id, received_utc,
                received_monotonic_ns, provider_timestamp_utc,
                original_payload_json, admission_run_id,
                admission_recorder_generation, connection_generation,
                subscription_owner, symbol, callback_classification,
                provider_envelope_event_id, status,
                acknowledgement_timestamp_utc, admitted_at_utc, updated_at_utc
            ) VALUES (
                'canonical-callback', 'tick_by_tick_bidask', 10, ?, 2, ?, '{}',
                'run-retention', 1, 1, 'universe:AAL', 'AAL',
                'accepted_active_callback', 'provider-envelope', 'acknowledged',
                ?, ?, ?
            )
            """,
            (observed, observed, observed, observed, observed),
        )
        connection.execute(
            """
            INSERT INTO callback_raw_materialization_v1(
                inbox_event_id, source_sequence, run_id, recorder_generation,
                lease_batch_id, raw_partition_hashes_json, raw_event_ids_json,
                materialized_at_utc
            ) VALUES ('canonical-callback', 2, 'run-retention', 1, 'batch-1', ?,
                      '["inside"]', ?)
            """,
            (hashes, observed),
        )
        connection.execute(
            """
            INSERT INTO callback_processing_commit_v1(
                inbox_event_id, source_sequence, run_id, recorder_generation,
                raw_partition_hashes_json, committed_at_utc
            ) VALUES ('canonical-callback', 2, 'run-retention', 1, ?, ?)
            """,
            (hashes, observed),
        )
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
    )

    finalizer.finalize_session(
        session=SESSION,
        observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
    )

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM callback_inbox_v1
            WHERE inbox_event_id IN ('provider-envelope', 'canonical-callback')
            """
        ).fetchone()[0] == 0
        assert connection.execute(
            """
            SELECT purged_callback_count
            FROM m1c_event_window_callback_purge_v0
            WHERE run_id = 'run-retention' AND session_date = ?
            """,
            (SESSION.isoformat(),),
        ).fetchone()[0] == 2


def test_callback_admitted_during_finalization_blocks_atomic_prepare(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)

    def admit_before_prepare(phase: str, _path: Path) -> None:
        if phase != "before_prepare_session":
            return
        observed = datetime(2026, 8, 17, 14, 3, tzinfo=UTC).isoformat()
        with sqlite3.connect(database) as connection:
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
                    'racing-callback', 'level1_quote_update', 10, ?, 3, ?, '{}',
                    'run-retention', 1, 1, 'universe:AAL', 'AAL',
                    'accepted_active_callback', 'pending', ?, ?
                )
                """,
                (observed, observed, observed, observed),
            )

    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
        failure_injector=admit_before_prepare,
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_CALLBACK_NOT_TERMINAL"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
        )

    assert source_path.exists()
    assert finalizer.repository.session_receipt(SESSION) is None


def test_restart_completes_prepared_retention_after_source_unlink(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(tmp_path)

    def crash(phase: str, _path: Path) -> None:
        if phase == "after_source_data_unlink":
            raise RuntimeError("injected-retention-crash")

    interrupted = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
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
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
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
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
    )
    finalizer.finalize_session(
        session=SESSION,
        observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
    )
    recorder_repository = FrozenRecorderRepository(ProspectiveRepository(database))
    existing_files = tuple(sorted(raw_root.rglob("*.parquet")))

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_ALREADY_SEALED"):
        recorder_repository.assert_raw_partition_sessions_open(
            run_id="run-retention",
            identities=((SESSION, "underlying_level1_quote_event"),),
        )
    assert tuple(sorted(raw_root.rglob("*.parquet"))) == existing_files

    inbox = DurableCallbackInbox(
        database,
        run_id="run-retention",
        recorder_generation=1,
        owner_id="test-recorder",
    )
    with pytest.raises(CallbackInboxError, match="CALLBACK_AFTER_RETENTION_SESSION_SEALED"):
        inbox.admit(
            callback_kind="level1_quote_update",
            request_id=10,
            payload={"provider_timestamp_utc": "2026-08-17T14:22:00+00:00"},
            connection_generation=1,
            classification=CallbackClassification.ACCEPTED_ACTIVE,
            received_utc=datetime(2026, 8, 17, 14, 22, tzinfo=UTC),
            received_monotonic_ns=4,
            symbol="AAL",
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
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
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


def test_all_rows_in_window_get_distinct_retained_manifest_identity(tmp_path: Path) -> None:
    database, raw_root, source_path = _database_with_episode_and_partition(
        tmp_path,
        all_inside=True,
    )
    source_hash = source_path.name.split("-")[1].split(".")[0]
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=SESSION,
    )

    receipt = finalizer.finalize_session(
        session=SESSION,
        observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
    )

    retained_hash = receipt.partitions[0].retained_content_hash
    assert receipt.retained_row_count == receipt.source_row_count == 2
    assert retained_hash is not None
    assert retained_hash != source_hash


def test_finalizer_rejects_session_outside_frozen_twenty_session_schedule(
    tmp_path: Path,
) -> None:
    database, raw_root, _ = _database_with_episode_and_partition(tmp_path)
    finalizer = SessionEventWindowFinalizerV0(
        database=database,
        raw_root=raw_root,
        retained_root=raw_root / "retained",
        run_id="run-retention",
        activation_timestamp_utc=ACTIVATION,
        first_eligible_session=date(2026, 7, 1),
    )

    with pytest.raises(RuntimeError, match="RETENTION_SESSION_OUTSIDE_FROZEN_SCHEDULE"):
        finalizer.finalize_session(
            session=SESSION,
            observed_at=datetime(2026, 8, 17, 22, 0, tzinfo=UTC),
        )
