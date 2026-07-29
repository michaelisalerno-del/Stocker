from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.durable_inbox import (
    CallbackClassification,
    DurableCallbackInbox,
)
from stocker_prospective.event_ingest import (
    IBKRCallbackNormalizer,
    StreamKind,
    StreamOwner,
)
from stocker_prospective.ibkr import IBKRConnectionConfig, IBKRMarketDataAdapter
from stocker_prospective.live_recorder import (
    CallbackNormalizationFatal,
    FrozenM1CLiveRecorder,
    ScientificReadiness,
)
from stocker_prospective.m1c_features import HistoricalActivityBaseline
from stocker_prospective.market_data import MarketDataBudget, MarketDataType
from stocker_prospective.operational_state import RecorderOperationalRepository
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import FrozenM1CRecorderEngine
from stocker_prospective.storage_recovery import CrossStoreReconciler

NOW = datetime(2026, 7, 29, 15, 30, tzinfo=UTC)
START = NOW - timedelta(days=1)
RUN_ID = "run-live-durable"
COHORT = (
    "AAL",
    "AAPL",
    "AMD",
    "AMZN",
    "BAC",
    "F",
    "INTC",
    "META",
    "MSFT",
    "MU",
    "NIO",
    "NVDA",
    "PFE",
    "PLTR",
    "SOFI",
    "T",
    "TSLA",
    "UBER",
    "WBD",
    "XOM",
)


class SimulatedCrash(BaseException):
    pass


def metadata_factory(
    observed_at: datetime,
    source_timestamps: tuple[datetime, ...],
) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=RUN_ID,
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[item.isoformat() for item in source_timestamps],
        recorded_at_utc=max(observed_at, START),
    )


def setup_database(tmp_path: Path) -> ProspectiveRepository:
    database = ProspectiveRepository(tmp_path / "prospective.sqlite3")
    database.migrate()
    database.create_run(metadata_factory(NOW, (NOW,)))
    return database


def build_recorder(
    tmp_path: Path,
    database: ProspectiveRepository,
    *,
    generation: int,
    owner: str,
    inbox: DurableCallbackInbox,
    adapter: IBKRMarketDataAdapter | None = None,
    failure_phase: str | None = None,
    raw_failure_phase: str | None = None,
    raw_crash_phase: str | None = None,
    register_default_stream: bool = True,
) -> tuple[FrozenM1CLiveRecorder, IBKRMarketDataAdapter]:
    operational = RecorderOperationalRepository(database.database_path)
    operational.start_generation(
        run_id=RUN_ID,
        recorder_generation=generation,
        owner_id=owner,
        started_at=NOW,
        required_market_data_mode="live",
        expected_artifact_count=1,
    )
    inbox.configure_recorder(
        run_id=RUN_ID,
        recorder_generation=generation,
        owner_id=owner,
    )
    if adapter is None:
        adapter = IBKRMarketDataAdapter(
            config=IBKRConnectionConfig(
                host="127.0.0.1",
                port=4002,
                client_id=91,
                expected_environment="read_only",
                connect_timeout_seconds=1,
                request_timeout_seconds=1,
                quote_capture_timeout_seconds=1,
                allowed_market_data_types=(MarketDataType.LIVE,),
            ),
            budget=MarketDataBudget(
                line_limit=50,
                reserved_headroom=1,
                request_rate_limit=100,
            ),
            durable_inbox=inbox,
        )
        adapter._connection_generation = 1
        adapter._track_request(7, "AAL:level1")
        adapter._subscription_kinds[7] = "market_data"
        adapter.stream_quotes.register(7)

    def recorder_failure(phase: str) -> None:
        if phase == failure_phase:
            raise SimulatedCrash(phase)

    def raw_failure(phase: str, _: Path) -> None:
        if phase == raw_crash_phase:
            raise SimulatedCrash(phase)
        if phase == raw_failure_phase:
            raise OSError("simulated disk write failure")

    normalizer = IBKRCallbackNormalizer(prospective_collection_start=START)
    recorder = FrozenM1CLiveRecorder(
        adapter=adapter,
        normalizer=normalizer,
        raw_store=PartitionedEventStore(
            root=tmp_path / "raw",
            prospective_collection_start=START,
            recorder_version="test",
            contract_version="frozen-m1c-microstructure-recorder-v0",
            run_id=RUN_ID,
            failure_injector=(
                raw_failure
                if raw_failure_phase is not None or raw_crash_phase is not None
                else None
            ),
        ),
        repository=FrozenRecorderRepository(database),
        engine=cast(FrozenM1CRecorderEngine, object()),
        activity_baseline=HistoricalActivityBaseline(minimum_sessions=1),
        group_o_provider=lambda _symbol, _session: (_ for _ in ()).throw(
            AssertionError("no checkpoint should be scored")
        ),
        metadata_factory=metadata_factory,
        run_id=RUN_ID,
        universe_symbols=COHORT,
        market_proxy_symbol="VTI",
        readiness=ScientificReadiness(
            m1c_parity_passed=True,
            direction_parity_passed=True,
            bar_compatibility_passed=True,
            clock_drift_within_tolerance=True,
        ),
        maximum_quote_age=timedelta(seconds=2),
        durable_inbox=inbox,
        recorder_generation=generation,
        lease_owner=owner,
        inbox_lease_timeout=timedelta(seconds=5),
        operational_repository=operational,
        failure_injector=recorder_failure if failure_phase is not None else None,
    )
    if register_default_stream:
        recorder.register_stream(
            StreamOwner(
                request_id=7,
                kind=StreamKind.UNDERLYING_LEVEL1,
                symbol="AAL",
                con_id=123,
                exchange="SMART",
            )
        )
    return recorder, adapter


def emit(adapter: IBKRMarketDataAdapter, *, field: str = "bid") -> None:
    adapter.on_quote_update(
        7,
        {
            "field": field,
            "value": 10.0 if field == "bid" else 10.02,
            "market_data_type": "live",
            "provider_timestamp_utc": NOW.isoformat(),
        },
    )


def manifest_count(database: ProspectiveRepository) -> int:
    with database._connect() as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM raw_partition_manifest_v0 WHERE run_id = ?",
                (RUN_ID,),
            ).fetchone()[0]
        )


def test_durable_poll_acknowledges_only_after_raw_manifest_commit(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    recorder, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
    )
    emit(adapter)

    result = recorder.poll(now=NOW + timedelta(seconds=1))
    recorder.finalize_durable_poll(
        result,
        acknowledged_at=NOW + timedelta(seconds=1),
    )

    assert result.raw_event_count == 1
    assert len(result.partition_hashes) == 1
    assert manifest_count(database) == 1
    accounting = inbox.accounting()
    assert accounting.acknowledged == 1
    assert accounting.pending == 0
    assert accounting.highest_source_sequence == accounting.highest_acknowledged_sequence


def test_crash_after_lease_is_reclaimed_without_loss(tmp_path: Path) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    first, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
        failure_phase="after_callback_lease",
    )
    emit(adapter)

    with pytest.raises(SimulatedCrash, match="after_callback_lease"):
        first.poll(now=NOW)
    assert inbox.accounting().leased == 1
    assert manifest_count(database) == 0

    restarted, _ = build_recorder(
        tmp_path,
        database,
        generation=2,
        owner="two",
        inbox=inbox,
        adapter=adapter,
    )
    result = restarted.poll(now=NOW + timedelta(seconds=6))
    restarted.finalize_durable_poll(
        result,
        acknowledged_at=NOW + timedelta(seconds=6),
    )
    assert result.raw_event_count == 1
    assert inbox.accounting().acknowledged == 1
    assert manifest_count(database) == 1


def test_replacement_generation_recovers_raw_but_cannot_reenable_scoring(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    first, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="crashed-owner",
        inbox=inbox,
        register_default_stream=False,
    )
    emit(adapter)
    with database._connect() as connection:
        assert (
            connection.execute(
                """
                SELECT stream_owner_json
                FROM callback_inbox_v1
                WHERE status = 'pending'
                """
            ).fetchone()[0]
            is None
        )
    first.register_stream(
        StreamOwner(
            request_id=7,
            kind=StreamKind.UNDERLYING_LEVEL1,
            symbol="AAL",
            con_id=123,
            exchange="SMART",
        )
    )
    assert inbox.accounting().pending == 1
    inbox.latch_fatal(
        latch_kind="ingestion",
        stable_error_code="RECORDER_UNCLEAN_RESTART_STATE_UNCERTAIN",
        occurred_at=NOW,
        error_class="UncleanRecorderRestart",
        evidence_loss_possible=True,
    )

    restarted, adapter = build_recorder(
        tmp_path,
        database,
        generation=2,
        owner="replacement",
        inbox=inbox,
        adapter=adapter,
    )
    restarted.set_session_context_ready(passed=True)
    restarted.set_capability_preflight(passed=True)
    assert restarted.scientific_block_latched
    assert not restarted.scientific_scoring_enabled

    result = restarted.poll(now=NOW + timedelta(seconds=1))
    restarted.finalize_durable_poll(
        result,
        acknowledged_at=NOW + timedelta(seconds=1),
    )

    assert result.raw_event_count == 1
    assert result.checkpoint_count == 0
    assert inbox.accounting().acknowledged == 1
    assert manifest_count(database) == 1


def test_blocked_recovery_persists_original_callback_without_current_owner(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    _, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="crashed-owner",
        inbox=inbox,
    )
    emit(adapter)
    inbox.latch_fatal(
        latch_kind="ingestion",
        stable_error_code="RECORDER_UNCLEAN_RESTART_STATE_UNCERTAIN",
        occurred_at=NOW,
        error_class="UncleanRecorderRestart",
        evidence_loss_possible=True,
    )

    replacement, _ = build_recorder(
        tmp_path,
        database,
        generation=2,
        owner="replacement",
        inbox=inbox,
        adapter=adapter,
    )
    replacement.normalizer.unregister(7)

    result = replacement.poll(now=NOW + timedelta(seconds=1))
    replacement.finalize_durable_poll(
        result,
        acknowledged_at=NOW + timedelta(seconds=1),
    )

    assert result.processing_disposition == "scientifically_blocked_raw_only"
    assert not result.scientific_projection_complete
    assert result.raw_event_count == 1
    assert inbox.accounting().acknowledged == 1
    with database._connect() as connection:
        manifest = connection.execute(
            """
            SELECT event_type, file_path
            FROM raw_partition_manifest_v0
            WHERE run_id = ?
            """,
            (RUN_ID,),
        ).fetchone()
        processing = connection.execute(
            """
            SELECT processing_disposition, scientific_projection_complete
            FROM callback_processing_commit_v1
            WHERE inbox_event_id = ?
            """,
            (result.durable_inbox_event_ids[0],),
        ).fetchone()
        owner_json = connection.execute(
            """
            SELECT stream_owner_json
            FROM callback_inbox_v1
            WHERE inbox_event_id = ?
            """,
            (result.durable_inbox_event_ids[0],),
        ).fetchone()[0]
    assert tuple(processing) == ("scientifically_blocked_raw_only", 0)
    assert str(manifest["event_type"]) == "raw_callback_envelope_event"
    assert json.loads(str(owner_json)) == {
        "con_id": 123,
        "episode_id": None,
        "exchange": "SMART",
        "kind": "underlying_level1",
        "option_contract": None,
        "request_id": 7,
        "symbol": "AAL",
    }
    import pyarrow.parquet as pq

    row = pq.ParquetFile(str(manifest["file_path"])).read().to_pylist()[0]
    assert row["callback_kind"] == "level1_quote_update"
    assert json.loads(str(row["original_payload"]))["value"] == 10.0
    assert row["recovery_disposition"] == "scientifically_blocked_raw_only"


def test_blocked_recovery_reuses_pre_materialized_identity_without_normalizer(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    first, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="crashed-owner",
        inbox=inbox,
    )
    emit(adapter, field="bid")
    initial = first.poll(now=NOW)
    first.finalize_durable_poll(initial, acknowledged_at=NOW)
    emit(adapter, field="ask")
    interrupted = first.poll(now=NOW + timedelta(seconds=1))
    manifest_total = manifest_count(database)
    assert interrupted.raw_materialization_reused is False
    assert inbox.accounting().leased == 1
    inbox.latch_fatal(
        latch_kind="ingestion",
        stable_error_code="RECORDER_UNCLEAN_RESTART_STATE_UNCERTAIN",
        occurred_at=NOW + timedelta(seconds=1),
        error_class="UncleanRecorderRestart",
        evidence_loss_possible=True,
    )

    replacement, _ = build_recorder(
        tmp_path,
        database,
        generation=2,
        owner="replacement",
        inbox=inbox,
        adapter=adapter,
    )
    replacement.normalizer.unregister(7)
    recovered = replacement.poll(now=NOW + timedelta(seconds=7))
    replacement.finalize_durable_poll(
        recovered,
        acknowledged_at=NOW + timedelta(seconds=7),
    )

    assert recovered.raw_materialization_reused
    assert recovered.partition_hashes == interrupted.partition_hashes
    assert recovered.raw_event_ids == interrupted.raw_event_ids
    assert recovered.processing_disposition == "scientifically_blocked_raw_only"
    assert not recovered.scientific_projection_complete
    assert manifest_count(database) == manifest_total
    assert inbox.accounting().acknowledged == 2
    with database._connect() as connection:
        row = connection.execute(
            """
            SELECT processing_disposition, scientific_projection_complete
            FROM callback_processing_commit_v1
            WHERE inbox_event_id = ?
            """,
            (recovered.durable_inbox_event_ids[0],),
        ).fetchone()
    assert tuple(row) == ("scientifically_blocked_raw_only", 0)


@pytest.mark.parametrize(
    "failure_phase",
    [
        "before_callback_processing_commit",
        "after_callback_processing_commit",
    ],
)
def test_restart_after_manifest_or_processing_commit_is_idempotent(
    tmp_path: Path,
    failure_phase: str,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    first, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
        failure_phase=failure_phase,
    )
    emit(adapter)

    first_result = first.poll(now=NOW)
    with pytest.raises(SimulatedCrash, match=failure_phase):
        first.finalize_durable_poll(first_result, acknowledged_at=NOW)
    assert manifest_count(database) == 1
    assert inbox.accounting().acknowledged == 0

    restarted, _ = build_recorder(
        tmp_path,
        database,
        generation=2,
        owner="two",
        inbox=inbox,
        adapter=adapter,
    )
    restarted_result = restarted.poll(now=NOW + timedelta(seconds=6))
    restarted.finalize_durable_poll(
        restarted_result,
        acknowledged_at=NOW + timedelta(seconds=6),
    )
    assert inbox.accounting().acknowledged == 1
    assert manifest_count(database) == 1
    assert len(tuple((tmp_path / "raw").rglob("part-*.parquet"))) == 1


def test_crash_after_parquet_completion_before_manifest_reconciles_on_restart(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    first, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
        raw_crash_phase="after_partition_complete",
    )
    emit(adapter)

    with pytest.raises(SimulatedCrash, match="after_partition_complete"):
        first.poll(now=NOW)
    assert manifest_count(database) == 0
    assert inbox.accounting().leased == 1
    assert len(tuple((tmp_path / "raw").rglob("part-*.parquet"))) == 1

    recovery = CrossStoreReconciler(
        repository=database,
        recorder_repository=FrozenRecorderRepository(database),
        raw_store=first.raw_store,
        inbox=inbox,
        run_id=RUN_ID,
        recorder_generation=2,
    ).reconcile(
        metadata_factory(NOW + timedelta(seconds=6), (NOW,)),
        observed_at=NOW + timedelta(seconds=6),
    )
    assert recovery.safe_to_score
    assert len(recovery.manifests_registered) == 1
    assert manifest_count(database) == 1

    restarted, _ = build_recorder(
        tmp_path,
        database,
        generation=2,
        owner="two",
        inbox=inbox,
        adapter=adapter,
    )
    restarted_result = restarted.poll(now=NOW + timedelta(seconds=6))
    restarted.finalize_durable_poll(
        restarted_result,
        acknowledged_at=NOW + timedelta(seconds=6),
    )
    accounting = inbox.accounting()
    assert accounting.acknowledged == 1
    assert accounting.highest_source_sequence == (accounting.highest_acknowledged_sequence)
    assert manifest_count(database) == 1


def test_outer_application_failure_keeps_raw_evidence_and_blocks_validity(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    recorder, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
    )
    emit(adapter)

    result = recorder.poll(now=NOW)
    recorder.fail_inflight_durable_poll(
        RuntimeError("synthetic failure before outer checkpoint completion"),
        occurred_at=NOW,
    )

    accounting = inbox.accounting()
    assert accounting.pending == 1
    assert accounting.acknowledged == 0
    assert manifest_count(database) == 1
    assert inbox.has_active_fatal("ingestion")
    with database._connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM callback_raw_materialization_v1").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM callback_processing_commit_v1").fetchone()[0]
            == 0
        )
    assert result.partition_hashes
    assert not recorder.scientific_scoring_enabled


def test_ack_failure_after_processing_commit_is_degraded_not_fatal(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    recorder, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
        failure_phase="after_callback_processing_commit",
    )
    emit(adapter)
    result = recorder.poll(now=NOW)

    with pytest.raises(
        SimulatedCrash,
        match="after_callback_processing_commit",
    ) as failure:
        recorder.finalize_durable_poll(result, acknowledged_at=NOW)
    recorder.fail_inflight_durable_poll(
        failure.value,
        occurred_at=NOW,
    )

    assert not inbox.has_active_fatal()
    assert inbox.accounting().leased == 1
    with database._connect() as connection:
        incident = connection.execute(
            """
            SELECT severity, evidence_loss_possible
            FROM operational_incident_v1
            WHERE stable_error_code = 'CALLBACK_ACK_DEFERRED'
            """
        ).fetchone()
    assert tuple(incident) == ("degraded", 0)


def test_application_failure_before_callback_lease_is_persisted_and_fatal(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    recorder, _ = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
    )

    recorder.fail_inflight_durable_poll(
        RuntimeError("synthetic session preflight failure"),
        occurred_at=NOW,
    )

    assert inbox.has_active_fatal("ingestion")
    assert not recorder.scientific_scoring_enabled
    with database._connect() as connection:
        row = connection.execute(
            """
            SELECT stable_error_code, evidence_loss_possible, details_json
            FROM operational_incident_v1
            WHERE component = 'frozen_prospective_application'
            """
        ).fetchone()
    assert row is not None
    assert str(row["stable_error_code"]) == "RECORDER_APPLICATION_COMMIT_FAILED"
    assert bool(row["evidence_loss_possible"])
    assert '"failure_before_callback_lease":true' in str(row["details_json"])


def test_replacement_generation_latches_interrupted_provider_callback(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(
        database.database_path,
        run_id=RUN_ID,
        recorder_generation=1,
        owner_id="old",
    )
    inbox.admit(
        callback_kind="official_provider_tick_price",
        request_id=7,
        payload={"provider_arguments": [7, 1, 10.0]},
        connection_generation=1,
        classification=CallbackClassification.ACCEPTED_ACTIVE,
        received_utc=NOW,
        received_monotonic_ns=100,
        inbox_event_id="interrupted-provider",
        provider_envelope=True,
    )
    replacement, _ = build_recorder(
        tmp_path,
        database,
        generation=2,
        owner="replacement",
        inbox=inbox,
    )

    replacement.poll(now=NOW + timedelta(seconds=10))

    assert inbox.has_active_fatal("ingestion")
    assert not replacement.scientific_scoring_enabled
    with database._connect() as connection:
        row = connection.execute(
            """
            SELECT status, failure_classification
            FROM callback_inbox_v1
            WHERE inbox_event_id = 'interrupted-provider'
            """
        ).fetchone()
    assert tuple(row) == (
        "quarantined",
        "CALLBACK_PROVIDER_MATERIALIZATION_INTERRUPTED",
    )


def test_poison_callback_is_quarantined_and_blocks_scoring(tmp_path: Path) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    recorder, _ = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
    )
    inbox.admit(
        callback_kind="level1_quote_update",
        request_id=7,
        payload={"field": "bid", "value": float("nan")},
        connection_generation=1,
        classification=CallbackClassification.ACCEPTED_ACTIVE,
        received_utc=NOW,
        received_monotonic_ns=1,
        inbox_event_id="poison-event",
        subscription_owner="AAL:level1",
        symbol="AAL",
    )

    with pytest.raises(
        CallbackNormalizationFatal,
        match="CALLBACK_NORMALIZATION_FAILED",
    ):
        recorder.poll(now=NOW)

    assert inbox.accounting().quarantined == 1
    assert inbox.accounting().acknowledged == 0
    assert inbox.has_active_fatal("ingestion")
    assert not recorder._scientific_scoring_enabled


def test_poison_later_in_lease_cannot_partially_persist_earlier_callback(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    recorder, _ = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
    )
    for event_id, monotonic, value in (
        ("valid-first", 1, 10.0),
        ("poison-second", 2, float("nan")),
    ):
        inbox.admit(
            callback_kind="level1_quote_update",
            request_id=7,
            payload={"field": "bid", "value": value},
            connection_generation=1,
            classification=CallbackClassification.ACCEPTED_ACTIVE,
            received_utc=NOW,
            received_monotonic_ns=monotonic,
            inbox_event_id=event_id,
            subscription_owner="AAL:level1",
            symbol="AAL",
        )

    with pytest.raises(CallbackNormalizationFatal):
        recorder.poll(now=NOW)

    accounting = inbox.accounting()
    assert accounting.pending == 1
    assert accounting.quarantined == 1
    assert accounting.acknowledged == 0
    assert manifest_count(database) == 0
    assert not tuple((tmp_path / "raw").rglob("part-*.parquet"))


def test_raw_write_failure_releases_lease_and_latches_storage_fatal(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    recorder, adapter = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="one",
        inbox=inbox,
        raw_failure_phase="after_temporary_write",
    )
    emit(adapter)

    with pytest.raises(OSError, match="simulated disk write failure"):
        recorder.poll(now=NOW)

    assert inbox.accounting().pending == 1
    assert inbox.accounting().acknowledged == 0
    assert inbox.has_active_fatal("storage")
    assert not recorder._scientific_scoring_enabled


def test_replacement_generation_restores_and_resolves_unfinished_scientific_gap(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    inbox = DurableCallbackInbox(database.database_path)
    first, _ = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="old",
        inbox=inbox,
    )
    gap_start = NOW - timedelta(minutes=5)
    first.mark_gap(
        "AAL",
        started_at=gap_start,
        cause_code="RESTART_REQUIRED_GAP",
        request_id=7,
        stream_kind=StreamKind.UNDERLYING_BAR.value,
        recoverability="recoverable",
    )

    replacement, _ = build_recorder(
        tmp_path,
        database,
        generation=2,
        owner="new",
        inbox=inbox,
    )

    assert replacement.gap_overlaps(
        "AAL",
        window_start=gap_start,
        window_end=NOW,
    )
    assert "AAL" in replacement._gap_symbols
    with database._connect() as connection:
        assert (
            connection.execute(
                """
            SELECT unresolved_required_gap_count
            FROM recorder_operational_state_v1
            WHERE run_id = ?
            """,
                (RUN_ID,),
            ).fetchone()[0]
            == 1
        )

    replacement.clear_gap_after_complete_bar("AAL", completed_at=NOW)

    assert "AAL" not in replacement._gap_symbols
    with database._connect() as connection:
        assert (
            connection.execute(
                """
            SELECT COUNT(*)
            FROM gap_incident_v1
            WHERE run_id = ? AND resolution_timestamp_utc IS NULL
            """,
                (RUN_ID,),
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                """
            SELECT unresolved_required_gap_count
            FROM recorder_operational_state_v1
            WHERE run_id = ?
            """,
                (RUN_ID,),
            ).fetchone()[0]
            == 0
        )


def test_complete_bar_does_not_resolve_level1_or_option_gap(
    tmp_path: Path,
) -> None:
    database = setup_database(tmp_path)
    recorder, _ = build_recorder(
        tmp_path,
        database,
        generation=1,
        owner="recorder",
        inbox=DurableCallbackInbox(database.database_path),
    )
    recorder.mark_gap(
        "AAL",
        started_at=NOW - timedelta(minutes=5),
        cause_code="REQUIRED_LEVEL1_INTERRUPTION",
        request_id=7,
        stream_kind=StreamKind.UNDERLYING_LEVEL1.value,
        recoverability="recoverable",
    )
    recorder.mark_gap(
        "AAL",
        started_at=NOW - timedelta(minutes=4),
        cause_code="REQUIRED_OPTION_INTERRUPTION",
        request_id=8,
        stream_kind=StreamKind.OPTION_LEVEL1.value,
        recoverability="recoverable",
    )

    recorder.clear_gap_after_complete_bar("AAL", completed_at=NOW)

    assert "AAL" in recorder._gap_symbols
    with database._connect() as connection:
        rows = connection.execute(
            """
            SELECT stream_kind
            FROM gap_incident_v1
            WHERE run_id = ? AND resolution_timestamp_utc IS NULL
            ORDER BY stream_kind
            """,
            (RUN_ID,),
        ).fetchall()
    assert tuple(str(row[0]) for row in rows) == (
        StreamKind.OPTION_LEVEL1.value,
        StreamKind.UNDERLYING_LEVEL1.value,
    )
