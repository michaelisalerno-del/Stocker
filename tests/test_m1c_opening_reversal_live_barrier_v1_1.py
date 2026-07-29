from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.event_ingest import (
    IBKRCallbackNormalizer,
    StreamKind,
    StreamOwner,
)
from stocker_prospective.live_recorder import (
    FrozenM1CLiveRecorder,
    ScientificReadiness,
)
from stocker_prospective.m1c_features import HistoricalActivityBaseline
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    build_activation_receipt_v1,
    build_frozen_experiment_config_v1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    build_activation_receipt_v1_1,
    build_frozen_timing_addendum_config_v1_1,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.partition_store import PartitionedEventStore
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import FrozenM1CRecorderEngine

BASE_ACTIVATION = datetime(2026, 7, 29, 6, 39, tzinfo=UTC)
ADDENDUM_ACTIVATION = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
SESSION = date(2026, 7, 30)
ENTRY = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)
RECEIPT_CREATED = ENTRY + timedelta(milliseconds=250)
COHORT = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)


def _activation_pair():
    frozen = build_frozen_experiment_config_v1()
    base = build_activation_receipt_v1(
        activation_timestamp_utc=BASE_ACTIVATION,
        new_york_trading_date_at_activation=BASE_ACTIVATION.date(),
        branch="codex/m1c-prospective-opening-reversal-v1",
        commit="a" * 40,
        dirty_working_tree_status="clean",
        configuration_hash=frozen.configuration_hash,
        m1c_version="frozen-m1c-v0",
        tail_phase_version="m1c-tail-phase-v1",
        a1_version="frozen-a1-v0",
    )
    addendum = build_frozen_timing_addendum_config_v1_1(
        superseded_activation_receipt_hash_v1=base.activation_receipt_hash,
        frozen_rule_hash_v1=base.frozen_rule_hash,
        frozen_configuration_hash_v1=base.configuration_hash,
    )
    activation = build_activation_receipt_v1_1(
        activation_timestamp_utc=ADDENDUM_ACTIVATION,
        new_york_trading_date_at_activation=ADDENDUM_ACTIVATION.date(),
        branch="codex/m1c-prospective-opening-reversal-v1",
        commit="b" * 40,
        dirty_working_tree_status="clean",
        timing_addendum_config=addendum,
        superseded_activation_receipt=base,
        m1c_version=base.m1c_version,
        tail_phase_version=base.tail_phase_version,
        a1_version=base.a1_version,
    )
    return base, activation


def _metadata(
    run_id: str,
    observed_at: datetime,
    source_timestamps: tuple[datetime, ...],
) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=BASE_ACTIVATION,
        app_version="test",
        git_commit="c" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[value.isoformat() for value in source_timestamps],
        recorded_at_utc=observed_at,
    )


class _CallbackAdapter:
    def __init__(self, callbacks: tuple[dict[str, object], ...]) -> None:
        self._callbacks = callbacks
        self.connection = SimpleNamespace(
            health=lambda: SimpleNamespace(market_data_type=MarketDataType.LIVE)
        )

    def drain_stream_events(self) -> tuple[dict[str, object], ...]:
        callbacks = self._callbacks
        self._callbacks = ()
        return callbacks


class _OrderingRepository(FrozenRecorderRepository):
    barrier_persisted = False

    def record_opening_reversal_causal_barrier_audit_v1_1(
        self,
        metadata: EvidenceMetadata,
        audit: Any,
    ) -> int:
        row_id = super().record_opening_reversal_causal_barrier_audit_v1_1(
            metadata,
            audit,
        )
        self.barrier_persisted = True
        return row_id

    def update_underlying_live_projection(
        self,
        metadata: EvidenceMetadata,
        event: Any,
        **kwargs: object,
    ) -> None:
        if event.ordering_timestamp >= ENTRY:
            assert self.barrier_persisted
        super().update_underlying_live_projection(
            metadata,
            event,
            **kwargs,
        )


def test_live_v1_1_archives_then_persists_all_receipts_and_barrier_before_projection(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "live-v1-1.sqlite3")
    database.migrate()
    run_id = "live-v1-1"
    database.create_run(_metadata(run_id, ADDENDUM_ACTIVATION, (ADDENDUM_ACTIVATION,)))
    repository = _OrderingRepository(database)
    base, activation = _activation_pair()
    activation_metadata = _metadata(
        run_id,
        ADDENDUM_ACTIVATION,
        (ADDENDUM_ACTIVATION,),
    )
    repository.record_opening_reversal_activation_v1(
        activation_metadata,
        base,
    )
    repository.record_opening_reversal_activation_v1_1(
        activation_metadata,
        activation,
    )
    engine = FrozenM1CRecorderEngine(
        m1c_runtime=cast(Any, object()),
        m1c_features=cast(Any, object()),
        direction_runtime=cast(Any, object()),
        direction_features=cast(Any, object()),
        repository=repository,
        opening_reversal_activation_v1=base,
        opening_reversal_activation_v1_1=activation,
    )
    callback = {
        "kind": "level1_quote_update",
        "request_id": 7,
        "field": "bid",
        "value": 10.0,
        "market_data_type": "live",
        "received_timestamp_utc": (ENTRY + timedelta(milliseconds=1)).isoformat(),
        "provider_timestamp_utc": ENTRY.isoformat(),
        "received_monotonic_ns": 1,
        "source_sequence": 1,
    }
    normalizer = IBKRCallbackNormalizer(prospective_collection_start=BASE_ACTIVATION)
    recorder = FrozenM1CLiveRecorder(
        adapter=cast(Any, _CallbackAdapter((callback,))),
        normalizer=normalizer,
        raw_store=PartitionedEventStore(
            root=tmp_path / "raw",
            prospective_collection_start=BASE_ACTIVATION,
            recorder_version="test",
            contract_version="frozen-m1c-microstructure-recorder-v0",
        ),
        repository=repository,
        engine=engine,
        activity_baseline=HistoricalActivityBaseline(minimum_sessions=1),
        group_o_provider=lambda _symbol, _session: (_ for _ in ()).throw(
            KeyError("engineering context unavailable")
        ),
        metadata_factory=lambda observed, sources: _metadata(
            run_id,
            observed,
            sources,
        ),
        universe_symbols=COHORT,
        market_proxy_symbol="VTI",
        readiness=ScientificReadiness(
            m1c_parity_passed=True,
            direction_parity_passed=True,
            bar_compatibility_passed=True,
            clock_drift_within_tolerance=True,
            capability_preflight_passed=True,
        ),
        maximum_quote_age=timedelta(seconds=2),
    )
    recorder.register_stream(
        StreamOwner(
            request_id=7,
            kind=StreamKind.UNDERLYING_LEVEL1,
            symbol="AAL",
            con_id=123,
            exchange="SMART",
        )
    )

    result = recorder.poll(now=RECEIPT_CREATED)

    assert len(result.opening_reversal_prediction_receipts) == 20
    assert {
        receipt.experiment_version for receipt in result.opening_reversal_prediction_receipts
    } == {"1.1"}
    assert len(result.opening_reversal_causal_barrier_audits_v1_1) == 1
    audit = result.opening_reversal_causal_barrier_audits_v1_1[0]
    assert audit.barrier_status == "passed"
    assert audit.prediction_receipt_count == 20
    assert audit.deferred_event_count == 1
    assert repository.barrier_persisted
    assert list((tmp_path / "raw").rglob("*.parquet"))
    with database._connect() as connection:
        assert (
            connection.execute(
                """
            SELECT COUNT(*) AS n
            FROM opening_reversal_causal_barrier_audit_v1_1
            WHERE barrier_status = 'passed'
            """
            ).fetchone()["n"]
            == 1
        )
        assert (
            connection.execute(
                """
            SELECT COUNT(*) AS n
            FROM underlying_live_state_v0
            WHERE symbol = 'AAL'
            """
            ).fetchone()["n"]
            == 1
        )
