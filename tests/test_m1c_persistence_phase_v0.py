from __future__ import annotations

import json
import os
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.activation import ProspectiveActivationLedger
from stocker_prospective.contract import CLAIMS_BOUNDARY
from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.events import (
    FiveMinuteBarEvent,
    OptionQuoteEvent,
    UnderlyingLevel1QuoteEvent,
)
from stocker_prospective.evidence_replay import replay_persisted_evidence
from stocker_prospective.frozen_m1c import EpisodeDecision, FrozenM1CScore
from stocker_prospective.group_o import build_group_o_context
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_ledger import OptionContract
from stocker_prospective.options import DteBucket
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.phase import EpisodeCompletion, ProspectivePhaseLedger
from stocker_prospective.quality_report import build_session_quality_report
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.replay_v0 import ReplayMode, deterministic_replay
from stocker_prospective.safety import (
    EpisodeSafetyDecision,
    EpisodeSafetyInputs,
    evaluate_episode_safety,
)

START = datetime(2026, 7, 24, 13, 30, tzinfo=UTC)


def raw_event(sequence: int) -> UnderlyingLevel1QuoteEvent:
    timestamp = START + timedelta(seconds=sequence)
    return UnderlyingLevel1QuoteEvent(
        event_id=f"q-{sequence}",
        received_timestamp_utc=timestamp,
        received_monotonic_ns=sequence + 1_000,
        provider_timestamp_utc=timestamp,
        source_sequence=sequence + 1_000,
        session=date(2026, 7, 24),
        symbol="AAL",
        con_id=1,
        request_id=10,
        bid=10.0 + sequence / 100,
        bid_size=100.0,
        ask=10.1 + sequence / 100,
        ask_size=100.0,
        last=10.05,
        last_size=1.0,
        market_data_type=MarketDataType.LIVE,
        source="fake_ibkr",
        quote_valid=True,
        staleness_ms=0.0,
        tick_type="state_change",
        exchange="SMART",
    )


def completed_bar(sequence: int) -> FiveMinuteBarEvent:
    bar_end = START + timedelta(minutes=5 * sequence)
    return FiveMinuteBarEvent(
        event_id=f"bar-{sequence}",
        received_timestamp_utc=bar_end,
        received_monotonic_ns=sequence + 2_000,
        provider_timestamp_utc=bar_end,
        source_sequence=sequence + 2_000,
        session=date(2026, 7, 24),
        symbol="AAL",
        con_id=1,
        request_id=10,
        bar_start_utc=bar_end - timedelta(minutes=5),
        bar_end_utc=bar_end,
        checkpoint=sequence,
        open=10.0,
        high=10.2,
        low=9.9,
        close=10.1,
        volume_or_activity_field=1_000.0,
        wap_where_available=10.05,
        trade_count_where_available=100,
        source="fake_ibkr",
        source_completeness="complete",
        finalised=True,
    )


def test_stale_completed_bar_replay_does_not_move_projection_backwards(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="run-bars",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    recorder = FrozenRecorderRepository(database)

    recorder.update_completed_bar_projection(metadata, completed_bar(2))
    recorder.update_completed_bar_projection(metadata, completed_bar(1))

    with database._connect() as connection:
        projected = connection.execute(
            """
            SELECT checkpoint, bar_end_utc
            FROM completed_bar_state_v0
            WHERE run_id = ? AND symbol = ?
            """,
            (metadata.run_id, "AAL"),
        ).fetchone()
    assert projected is not None
    assert int(projected["checkpoint"]) == 2
    assert str(projected["bar_end_utc"]) == completed_bar(2).bar_end_utc.isoformat()


def test_option_schedule_is_durable_before_checkpoint_completion(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="run-option-schedule",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    recorder = FrozenRecorderRepository(database)
    checkpoint_id = recorder.record_checkpoint(
        metadata,
        symbol="AAL",
        session=START.date(),
        checkpoint=6,
        bar_start_utc=START - timedelta(minutes=5),
        bar_end_utc=START,
        score=FrozenM1CScore(
            model_hash="b" * 64,
            probability=0.12,
            threshold=0.488333710794033,
            threshold_passed=False,
            feature_order=("feature",),
            feature_values=(0.0,),
            transformed_values=(0.0,),
            feature_hash="c" * 64,
            missing_feature_count=0,
        ),
        session_context_hash="d" * 64,
        feature_values={"feature": 0.0},
        eligible=True,
        feature_freshness="exact_previous_session",
        rejection_reasons=(),
    )
    recorder.record_option_episode_schedule(
        metadata,
        episode_id="quiet-option-1",
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=START.date(),
        entry_timestamp=START + timedelta(minutes=5),
        episode_kind="quiet",
        probability=0.12,
        quiet_state=True,
        directional_actions={},
        recording_duration=timedelta(minutes=60),
        strike_steps=4,
        maximum_contracts=8,
    )

    restored = recorder.restorable_option_episode_schedules(
        run_id=metadata.run_id,
    )
    assert [row["episode_id"] for row in restored] == ["quiet-option-1"]
    assert recorder.recorded_checkpoint_identities(run_id=metadata.run_id) == set()

    recorder.mark_checkpoint_complete(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=START.date(),
        checkpoint=6,
    )
    recorder.update_option_episode_schedule_status(
        metadata,
        episode_id="quiet-option-1",
        status="complete",
    )
    assert recorder.restorable_option_episode_schedules(run_id=metadata.run_id) == []


def test_activation_is_first_write_immutable_and_carries_required_identity(tmp_path: Path) -> None:
    ledger = ProspectiveActivationLedger(tmp_path / "activation.json")
    first = ledger.activate(
        activation_timestamp_utc=START,
        git_sha="a" * 40,
        model_artifact_hashes={"m1c": "b" * 64},
        configuration_hash="c" * 64,
        ibkr_api_version="10.37.01",
        tws_or_gateway_version="IB Gateway 10.37",
    )
    second = ledger.activate(
        activation_timestamp_utc=START,
        git_sha="a" * 40,
        model_artifact_hashes={"m1c": "b" * 64},
        configuration_hash="c" * 64,
        ibkr_api_version="10.37.01",
        tws_or_gateway_version="IB Gateway 10.37",
    )

    assert first == second
    assert first.prospective_collection_start_utc == START
    assert first.prospective_collection_start_new_york.tzinfo is not None
    with pytest.raises(ValueError, match="immutable"):
        ledger.activate(
            activation_timestamp_utc=START + timedelta(seconds=1),
            git_sha="a" * 40,
            model_artifact_hashes={"m1c": "b" * 64},
            configuration_hash="c" * 64,
            ibkr_api_version="10.37.01",
            tws_or_gateway_version="IB Gateway 10.37",
        )


def test_partition_store_is_append_only_atomic_hashed_and_prospective(tmp_path: Path) -> None:
    store = PartitionedEventStore(
        root=tmp_path / "raw",
        prospective_collection_start=START,
        recorder_version="test",
        contract_version="frozen-m1c-microstructure-recorder-v0",
    )
    result = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(2), raw_event(1)),
        complete=True,
    )

    assert result.data_path.suffix == ".parquet"
    assert result.data_path.is_file()
    assert result.metadata_path.is_file()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["content_hash"] == result.content_hash
    assert metadata["complete"] is True
    assert metadata["claims_boundary"] == CLAIMS_BOUNDARY
    assert store.verify(result) is True
    assert not list(tmp_path.rglob("*.tmp"))

    with pytest.raises(ValueError, match="prospective_collection_start"):
        store.write_events(
            data_source="fake_ibkr",
            events=(raw_event(-1),),
            complete=False,
        )


def test_partition_finalisation_recovers_without_publishing_complete_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PartitionedEventStore(
        root=tmp_path / "raw",
        prospective_collection_start=START,
        recorder_version="test",
        contract_version="frozen-m1c-microstructure-recorder-v0",
    )
    original_replace = os.replace
    interrupted = False

    def interrupt_final_data_publish(source: str | Path, destination: str | Path) -> None:
        nonlocal interrupted
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not interrupted
            and source_path.name.endswith(".staged.parquet")
            and destination_path.name.endswith(".complete.parquet")
        ):
            interrupted = True
            raise OSError("synthetic final publish interruption")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_final_data_publish)
    with pytest.raises(OSError, match="synthetic final publish interruption"):
        store.write_events(
            data_source="fake_ibkr",
            events=(raw_event(1),),
            complete=True,
        )

    assert not list((tmp_path / "raw").rglob("*.complete.parquet"))
    assert list((tmp_path / "raw").rglob("*.staged.parquet"))
    monkeypatch.setattr(os, "replace", original_replace)

    recovered = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(1),),
        complete=True,
    )
    assert recovered.complete is True
    assert store.verify(recovered)
    assert not list((tmp_path / "raw").rglob("*.staged.parquet"))


def test_replay_preserves_provider_then_monotonic_order_and_is_deterministic() -> None:
    unordered = (raw_event(2), raw_event(0), raw_event(1))
    first = deterministic_replay(unordered, mode=ReplayMode.ACCELERATED)
    second = deterministic_replay(unordered, mode=ReplayMode.ACCELERATED)

    assert first.event_ids == ("q-0", "q-1", "q-2")
    assert first.digest == second.digest
    assert first.maximum_floating_difference == 0.0
    assert first.ibkr_connections_attempted == 0


def test_persisted_evidence_replay_reads_hashed_partitions_without_ibkr(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "replay.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="evidence-replay",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    store = PartitionedEventStore(
        root=tmp_path / "raw",
        prospective_collection_start=START,
        recorder_version="test",
        contract_version="frozen-m1c-microstructure-recorder-v0",
    )
    partition = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(2), raw_event(1)),
        complete=True,
    )
    FrozenRecorderRepository(database).record_partition(
        metadata,
        data_source="fake_ibkr",
        session_date=START.date(),
        symbol="AAL",
        event_type="underlying_level1_quote_event",
        partition=partition,
    )
    arguments = {
        "database_path": database.database_path,
        "run_id": metadata.run_id,
        "mode": "accelerated",
        "speed": 10.0,
        "episode_id": None,
        "m1c_feature_manifest_path": None,
        "m1c_threshold_path": None,
        "stop_event": threading.Event(),
    }

    first = replay_persisted_evidence(**arguments)
    second = replay_persisted_evidence(**arguments)

    assert first.raw_events_replayed == 2
    assert first.records_replayed == 2
    assert first.digest == second.digest
    assert first.stage_counts == {"raw_market_event": 2}
    assert first.ibkr_connections_attempted == 0


def test_persisted_evidence_replay_rejects_manifest_over_memory_bound_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = ProspectiveRepository(tmp_path / "bounded-replay.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="bounded-replay",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    store = PartitionedEventStore(
        root=tmp_path / "raw",
        prospective_collection_start=START,
        recorder_version="test",
        contract_version="frozen-m1c-microstructure-recorder-v0",
    )
    partition = store.write_events(
        data_source="fake_ibkr",
        events=(raw_event(1), raw_event(2)),
        complete=True,
    )
    FrozenRecorderRepository(database).record_partition(
        metadata,
        data_source="fake_ibkr",
        session_date=START.date(),
        symbol="AAL",
        event_type="underlying_level1_quote_event",
        partition=partition,
    )
    with database._connect() as connection:
        connection.execute(
            "UPDATE raw_partition_manifest_v0 SET row_count = 1000 WHERE run_id = ?",
            (metadata.run_id,),
        )

    parquet_opened = False

    def fail_if_opened(_path: object) -> object:
        nonlocal parquet_opened
        parquet_opened = True
        raise AssertionError("Parquet must not open after the manifest bound fails")

    monkeypatch.setattr("pyarrow.parquet.ParquetFile", fail_if_opened)

    with pytest.raises(
        RuntimeError,
        match="blocked_replay_record_limit_exceeded: estimated=1000 limit=100",
    ):
        replay_persisted_evidence(
            database_path=database.database_path,
            run_id=metadata.run_id,
            mode="accelerated",
            speed=10.0,
            episode_id=None,
            m1c_feature_manifest_path=None,
            m1c_threshold_path=None,
            stop_event=threading.Event(),
            maximum_records=100,
        )

    assert parquet_opened is False


def test_persisted_evidence_replay_streams_parquet_batches_instead_of_full_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    database = ProspectiveRepository(tmp_path / "batch-replay.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="batch-replay",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    store = PartitionedEventStore(
        root=tmp_path / "raw",
        prospective_collection_start=START,
        recorder_version="test",
        contract_version="frozen-m1c-microstructure-recorder-v0",
    )
    partition = store.write_events(
        data_source="fake_ibkr",
        events=tuple(raw_event(index) for index in range(12)),
        complete=True,
    )
    FrozenRecorderRepository(database).record_partition(
        metadata,
        data_source="fake_ibkr",
        session_date=START.date(),
        symbol="AAL",
        event_type="underlying_level1_quote_event",
        partition=partition,
    )
    original_parquet_file = pq.ParquetFile
    batch_sizes: list[int] = []

    class BatchOnlyParquetFile:
        def __init__(self, path: Path) -> None:
            self._delegate = original_parquet_file(path)

        def read(self) -> object:
            raise AssertionError("full Parquet table reads are forbidden during replay")

        def iter_batches(self, *, batch_size: int) -> object:
            batch_sizes.append(batch_size)
            return self._delegate.iter_batches(batch_size=batch_size)

    monkeypatch.setattr(pq, "ParquetFile", BatchOnlyParquetFile)

    result = replay_persisted_evidence(
        database_path=database.database_path,
        run_id=metadata.run_id,
        mode="accelerated",
        speed=10.0,
        episode_id=None,
        m1c_feature_manifest_path=None,
        m1c_threshold_path=None,
        stop_event=threading.Event(),
        maximum_records=100,
    )

    assert result.raw_events_replayed == 12
    assert batch_sizes == [100]
    assert result.ibkr_connections_attempted == 0
    assert result.broker_state_mutated is False


def test_phase_ledger_is_chronological_immutable_and_keeps_confirmation_closed(
    tmp_path: Path,
) -> None:
    ledger = ProspectivePhaseLedger(tmp_path / "phases.jsonl")
    assignments = [
        ledger.record(
            episode_id=f"episode-{index:03d}",
            occurred_at=START + timedelta(minutes=index),
            completion=EpisodeCompletion.all_valid(),
        )
        for index in range(132)
    ]

    assert assignments[0].phase == "engineering_shakedown"
    assert assignments[29].phase == "engineering_shakedown"
    assert assignments[30].phase == "microstructure_development"
    assert assignments[129].phase == "microstructure_development"
    assert assignments[130].phase == "microstructure_confirmation"
    assert assignments[130].target_dependent_selection_opened is False
    assert assignments[30].scientific_evidence_claim_allowed is False
    assert assignments[130].scientific_evidence_claim_allowed is False
    assert assignments[0].claims_boundary == CLAIMS_BOUNDARY
    assert (
        ledger.record(
            episode_id="episode-131",
            occurred_at=START + timedelta(minutes=131),
            completion=EpisodeCompletion.all_valid(),
        )
        == assignments[131]
    )


def test_end_of_session_quality_report_fails_closed_without_raw_partitions(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "quality.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="quality-v0",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    report = build_session_quality_report(
        database_path=database.database_path,
        run_id=metadata.run_id,
        session_date=START.date(),
        generated_at=START + timedelta(hours=8),
    )

    assert report.expected_universe_minutes == 7800
    assert report.level1_coverage == 0.0
    assert report.complete is False
    assert report.claims == CLAIMS_BOUNDARY


def test_episode_scientific_gate_fails_closed_with_stored_reasons() -> None:
    decision = evaluate_episode_safety(
        EpisodeSafetyInputs(
            capability_preflight_passed=True,
            m1c_parity_passed=True,
            direction_parity_passed=True,
            market_data_type=MarketDataType.DELAYED,
            previous_close_group_o_valid=True,
            trigger_bar_complete=True,
            clock_drift_within_tolerance=True,
            underlying_quote_fresh=True,
            unresolved_bar_gap=False,
            deterministic_episode_identity=True,
            raw_event_storage_writable=True,
        )
    )

    assert decision.scientific_recording_valid is False
    assert decision.rejection_reasons == ("market_data_not_live",)


def test_group_o_context_rejects_same_day_and_database_rows_carry_claims(
    tmp_path: Path,
) -> None:
    invalid = build_group_o_context(
        symbol="AAL",
        signal_session=date(2026, 7, 24),
        actual_option_observation_session=date(2026, 7, 24),
        front_expiry=date(2026, 7, 24),
        dte=0,
        atm_strike=12.0,
        features={"options_missing": 0.0},
        missing_indicators={"options_missing": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )
    assert invalid.eligible is False
    assert "same_day_group_o_rejected" in invalid.rejection_reasons

    context = build_group_o_context(
        symbol="AAL",
        signal_session=date(2026, 7, 24),
        actual_option_observation_session=date(2026, 7, 23),
        front_expiry=date(2026, 7, 24),
        dte=1,
        atm_strike=12.0,
        features={"options_missing": 0.0},
        missing_indicators={"options_missing": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="run-v0",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    recorder = FrozenRecorderRepository(database)
    recorder.record_group_o_context(metadata, context)
    recorder.record_checkpoint(
        metadata,
        symbol="AAL",
        session=date(2026, 7, 24),
        checkpoint=6,
        bar_start_utc=START,
        bar_end_utc=START + timedelta(minutes=5),
        score=FrozenM1CScore(
            model_hash="b" * 64,
            probability=0.6,
            threshold=0.488333710794033,
            threshold_passed=True,
            feature_order=("x",),
            feature_values=(1.0,),
            transformed_values=(1.0,),
            feature_hash="c" * 64,
            missing_feature_count=0,
        ),
        session_context_hash=context.context_hash,
        feature_values={"x": 1.0},
        eligible=True,
        feature_freshness="fresh",
        rejection_reasons=(),
    )

    with database._connect() as connection:
        claims = json.loads(
            str(
                connection.execute("SELECT claims_json FROM m1c_checkpoint_v0").fetchone()[
                    "claims_json"
                ]
            )
        )
    assert claims == CLAIMS_BOUNDARY


def test_option_web_projection_updates_without_replacing_raw_evidence(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="run-options",
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )
    database.create_run(metadata)
    recorder = FrozenRecorderRepository(database)
    checkpoint_id = recorder.record_checkpoint(
        metadata,
        symbol="AAL",
        session=date(2026, 7, 24),
        checkpoint=6,
        bar_start_utc=START,
        bar_end_utc=START + timedelta(minutes=5),
        score=FrozenM1CScore(
            model_hash="b" * 64,
            probability=0.6,
            threshold=0.488333710794033,
            threshold_passed=True,
            feature_order=("x",),
            feature_values=(1.0,),
            transformed_values=(1.0,),
            feature_hash="c" * 64,
            missing_feature_count=0,
        ),
        session_context_hash="d" * 64,
        feature_values={"x": 1.0},
        eligible=True,
        feature_freshness="fresh",
        rejection_reasons=(),
    )
    episode_id = recorder.record_episode(
        metadata,
        checkpoint_id=checkpoint_id,
        decision=EpisodeDecision(
            symbol="AAL",
            session=date(2026, 7, 24),
            checkpoint=6,
            probability=0.6,
            threshold=0.488333710794033,
            raw_above_threshold=True,
            previous_probability=0.4,
            fresh_episode=True,
            episode_id="m1c-test-option",
            episode_number=1,
            minutes_since_previous_episode=None,
            trigger_bar_end=START + timedelta(minutes=5),
            prospective_entry_timestamp=START + timedelta(minutes=5),
            rejection_reason=None,
        ),
        safety=EpisodeSafetyDecision(
            scientific_recording_valid=True,
            rejection_reasons=(),
        ),
    )
    contract = OptionContract(
        underlying_con_id=1,
        con_id=99,
        expiry=date(2026, 7, 24),
        dte=0,
        dte_bucket=DteBucket.ZERO_DTE,
        strike=12.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
    )
    contract_id = recorder.record_option_contract(
        metadata,
        episode_id=episode_id,
        contract=contract,
        selection_rank=1,
        resolution_status="recording",
        rejection_reason=None,
        recording_started_at_utc=START,
        recording_ends_at_utc=START + timedelta(minutes=35),
    )
    event = OptionQuoteEvent(
        event_id="option-1",
        received_timestamp_utc=START + timedelta(minutes=5),
        received_monotonic_ns=1,
        provider_timestamp_utc=START + timedelta(minutes=5),
        source_sequence=1,
        session=date(2026, 7, 24),
        symbol="AAL",
        con_id=99,
        request_id=19,
        episode_id=episode_id,
        expiry=date(2026, 7, 24),
        dte=0,
        dte_bucket=DteBucket.ZERO_DTE,
        strike=12.0,
        right="C",
        multiplier=100,
        exchange="SMART",
        trading_class="AAL",
        bid=1.0,
        bid_size=10.0,
        ask=1.1,
        ask_size=12.0,
        last=1.05,
        last_size=1.0,
        market_data_type=MarketDataType.LIVE,
    )
    recorder.update_option_quote_projection(
        option_contract_id=contract_id,
        event=event,
        recording_status="recording",
        quote_quality_flags=(),
    )

    with database._connect() as connection:
        row = connection.execute("SELECT * FROM option_quote_state_v0").fetchone()
    assert row["bid"] == 1.0
    assert row["ask"] == 1.1
    assert json.loads(row["claims_json"]) == CLAIMS_BOUNDARY
