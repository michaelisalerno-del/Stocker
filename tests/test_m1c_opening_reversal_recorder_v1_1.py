from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.frozen_m1c import FrozenM1CScore
from stocker_prospective.group_o import build_group_o_context
from stocker_prospective.m1c_features import LiveFeatureBar
from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    OpeningReversalPredictionInputV1,
    OpeningReversalPredictionTimingEvidenceV1_1,
    build_activation_receipt_v1,
    build_frozen_experiment_config_v1,
    build_prediction_receipt_v1,
    select_promoted_prediction_v1,
)
from stocker_prospective.m1c_prospective_opening_reversal_v1_1 import (
    build_activation_receipt_v1_1,
    build_causal_barrier_audit_v1_1,
    build_frozen_timing_addendum_config_v1_1,
)
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.opening_market_transition_v1 import (
    OpeningTransitionThresholdsV1,
)
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import (
    FrozenM1CRecorderEngine,
    RecorderCheckpointInput,
)
from stocker_prospective.signed_market_shock_v1 import MarketShockBarV1

BASE_ACTIVATION = datetime(2026, 7, 29, 6, 39, tzinfo=UTC)
ADDENDUM_ACTIVATION = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
SESSION = date(2026, 7, 30)
ENTRY = datetime(2026, 7, 30, 14, 0, tzinfo=UTC)
RECEIPT_CREATED = ENTRY + timedelta(milliseconds=250)


def _metadata(run_id: str, observed: datetime) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=BASE_ACTIVATION,
        app_version="test",
        git_commit="c" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[observed.isoformat()],
        recorded_at_utc=observed,
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


def _prediction(addendum_activation_hash: str, *, stock: str = "AAL"):
    timing = OpeningReversalPredictionTimingEvidenceV1_1(
        timing_addendum_activation_receipt_hash_v1_1=(addendum_activation_hash),
        rule_committed_at_utc=ADDENDUM_ACTIVATION,
        causal_barrier_armed_at_utc=ADDENDUM_ACTIVATION,
        predictor_window_completed_at_utc=ENTRY,
        first_entry_or_post_entry_event_buffered_at_utc=(ENTRY + timedelta(milliseconds=1)),
        entry_or_post_entry_data_admitted_before_receipt=False,
        raw_event_archive_write_before_receipt=True,
        decision_surface_release_requires_durable_receipt=True,
        nominal_entry_actionable=False,
        receipt_latency_after_nominal_entry_seconds=0.25,
    )
    return build_prediction_receipt_v1(
        OpeningReversalPredictionInputV1(
            experiment_version="1.1",
            activation_timestamp_utc=ADDENDUM_ACTIVATION,
            cohort_phase="engineering_transfer",
            transfer_status="engineering_transfer_pending",
            session=SESSION,
            stock=stock,
            checkpoint=6,
            signal_timestamp_utc=ENTRY,
            entry_timestamp_utc=ENTRY,
            receipt_created_at_utc=RECEIPT_CREATED,
            m1c_probability=0.70,
            m1c_probability_valid=True,
            high_tail_membership=True,
            fresh_episode_id=f"fresh-{stock.lower()}",
            canonical_fresh_episode=True,
            tail_phase_v1="FIRST_ENTRY",
            market_opening_return_v1=-0.004,
            market_opening_range_v1=0.006,
            opening_market_transition_state_v1=("NEGATIVE_SEVERE_OPENING_TRANSITION"),
            opening_transition_sign_v1=-1,
            opening_transition_event_id_v1="opening-event-1",
            vti_opening_transition_complete=True,
            stock_causal_data_complete=True,
            previous_close_atm_iv_scale_15m=0.01,
            previous_close_atm_iv_scale_valid=True,
            data_source="ibkr",
            capacity_snapshot_id="capacity-1",
            timing_evidence_v1_1=timing,
        )
    )


def test_v1_1_activation_and_prediction_are_durably_bound(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    base, activation = _activation_pair()

    repository.record_opening_reversal_activation_v1(metadata, base)
    repository.record_opening_reversal_activation_v1_1(
        metadata,
        activation,
    )
    receipt = _prediction(activation.activation_receipt_hash_v1_1)
    repository.record_opening_reversal_prediction_v1(metadata, receipt)

    with database._connect() as connection:
        stored_activation = connection.execute(
            """
            SELECT experiment_version, activation_receipt_hash
            FROM opening_reversal_activation_v1
            WHERE run_id = ? AND experiment_version = '1.1'
            """,
            (metadata.run_id,),
        ).fetchone()
        stored_prediction = connection.execute(
            """
            SELECT experiment_version, receipt_hash_v1, receipt_json
            FROM opening_reversal_prediction_v1
            WHERE run_id = ?
            """,
            (metadata.run_id,),
        ).fetchone()
        migration = connection.execute(
            """
            SELECT version FROM schema_migrations
            WHERE version =
                '0015_m1c_prospective_opening_reversal_v1_1.sql'
            """
        ).fetchone()

    assert migration is not None
    assert stored_activation is not None
    assert stored_activation["experiment_version"] == "1.1"
    assert stored_activation["activation_receipt_hash"] == activation.activation_receipt_hash_v1_1
    assert stored_prediction is not None
    assert stored_prediction["experiment_version"] == "1.1"
    assert stored_prediction["receipt_hash_v1"] == receipt.receipt_hash_v1
    assert "timing_evidence_v1_1" in stored_prediction["receipt_json"]


def test_v1_1_promotion_requires_persisted_passing_barrier_audit(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-reversal-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("opening-reversal-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    base, activation = _activation_pair()
    repository.record_opening_reversal_activation_v1(metadata, base)
    repository.record_opening_reversal_activation_v1_1(metadata, activation)
    stocks = ("AAL", *(f"S{index:02d}" for index in range(1, 20)))
    receipts = tuple(
        _prediction(
            activation.activation_receipt_hash_v1_1,
            stock=stock,
        )
        for stock in stocks
    )
    for receipt in receipts:
        repository.record_opening_reversal_prediction_v1(metadata, receipt)
    selection = select_promoted_prediction_v1(receipts)

    with pytest.raises(ValueError, match="causal barrier"):
        repository.record_opening_reversal_promotion_v1(metadata, selection)

    audit = build_causal_barrier_audit_v1_1(
        activation_receipt_hash_v1_1=(activation.activation_receipt_hash_v1_1),
        session=SESSION,
        nominal_entry_timestamp_utc=ENTRY,
        prediction_receipts=receipts,
        deferred_event_received_timestamps=(ENTRY + timedelta(milliseconds=1),),
        entry_or_post_entry_data_admitted_before_receipts=False,
        release_authorized_at_utc=RECEIPT_CREATED,
    )
    repository.record_opening_reversal_causal_barrier_audit_v1_1(
        metadata,
        audit,
    )
    operational = repository.opening_reversal_engineering_operational_evidence_v1(
        run_id=metadata.run_id,
        session=SESSION,
    )

    assert (
        repository.record_opening_reversal_promotion_v1(
            metadata,
            selection,
        )
        > 0
    )
    assert operational.prediction_receipt_count == 20
    assert operational.prediction_receipt_timing_pass


def test_v1_1_activation_requires_a_fresh_engineering_run(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "not-fresh.sqlite3")
    database.migrate()
    metadata = _metadata("not-fresh", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    base, activation = _activation_pair()
    repository.record_opening_reversal_activation_v1(metadata, base)
    # Rebuild through the public V1 builder so its immutable hash matches the
    # strict pre-entry receipt.
    source = OpeningReversalPredictionInputV1(
        experiment_version="1",
        activation_timestamp_utc=BASE_ACTIVATION,
        cohort_phase="engineering_transfer",
        transfer_status="engineering_transfer_pending",
        session=SESSION,
        stock="AAL",
        checkpoint=6,
        signal_timestamp_utc=ENTRY,
        entry_timestamp_utc=ENTRY,
        receipt_created_at_utc=ENTRY - timedelta(microseconds=1),
        m1c_probability=0.70,
        m1c_probability_valid=True,
        high_tail_membership=True,
        fresh_episode_id="fresh-aal",
        canonical_fresh_episode=True,
        tail_phase_v1="FIRST_ENTRY",
        market_opening_return_v1=-0.004,
        market_opening_range_v1=0.006,
        opening_market_transition_state_v1=("NEGATIVE_SEVERE_OPENING_TRANSITION"),
        opening_transition_sign_v1=-1,
        opening_transition_event_id_v1="opening-event-1",
        vti_opening_transition_complete=True,
        stock_causal_data_complete=True,
        previous_close_atm_iv_scale_15m=0.01,
        previous_close_atm_iv_scale_valid=True,
        data_source="ibkr",
        capacity_snapshot_id="capacity-1",
    )
    v1_receipt = build_prediction_receipt_v1(source)
    repository.record_opening_reversal_prediction_v1(
        metadata.model_copy(update={"recorded_at_utc": ENTRY - timedelta(microseconds=1)}),
        v1_receipt,
    )

    with pytest.raises(ValueError, match="fresh run"):
        repository.record_opening_reversal_activation_v1_1(
            metadata,
            activation,
        )


def test_checkpoint_engine_emits_v1_1_receipt_behind_causal_barrier(
    tmp_path: Path,
) -> None:
    class FakeFeatureBuilder:
        def build(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                scaled_features={"x": 1.0},
                feature_hash="e" * 64,
                scaling_artifact_hash="f" * 64,
            )

    class FakeRuntime:
        def missing_group_o_features(self, _: object) -> tuple[str, ...]:
            return ()

        def score(self, **_: object) -> FrozenM1CScore:
            return FrozenM1CScore(
                model_hash="b" * 64,
                probability=0.70,
                threshold=0.488333710794033,
                threshold_passed=True,
                feature_order=("x",),
                feature_values=(1.0,),
                transformed_values=(1.0,),
                feature_hash="c" * 64,
                missing_feature_count=0,
            )

    database = ProspectiveRepository(tmp_path / "engine-v1-1.sqlite3")
    database.migrate()
    metadata = _metadata("engine-v1-1", RECEIPT_CREATED)
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    base, activation = _activation_pair()
    repository.record_opening_reversal_activation_v1(metadata, base)
    repository.record_opening_reversal_activation_v1_1(metadata, activation)
    context = build_group_o_context(
        symbol="AAL",
        signal_session=SESSION,
        actual_option_observation_session=date(2026, 7, 29),
        front_expiry=date(2026, 7, 31),
        dte=1,
        atm_strike=100.0,
        previous_close_implied_movement_15m=0.01,
        features={"x": 1.0},
        missing_indicators={"x": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )
    repository.record_group_o_context(metadata, context)
    session_open = ENTRY - timedelta(minutes=30)
    stock_bars = tuple(
        LiveFeatureBar(
            symbol="AAL",
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=session_open + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=(session_open + timedelta(minutes=5 * (ordinal + 1))),
            open=100.0,
            high=101.0,
            low=99.0,
            close=99.5,
            volume=1_000.0,
            historical_relative_activity=1.0,
            finalised=True,
            source="fixture",
        )
        for ordinal in range(6)
    )
    market_bars = tuple(
        MarketShockBarV1(
            symbol="VTI",
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=session_open + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=(session_open + timedelta(minutes=5 * (ordinal + 1))),
            open=100.0,
            high=101.0,
            low=99.0,
            close=99.5,
            finalised=True,
        )
        for ordinal in range(6)
    )
    thresholds = OpeningTransitionThresholdsV1(
        market_opening_return_q10_v1=-0.00288963733897,
        market_opening_return_q90_v1=0.00225522676046,
        market_opening_range_q75_v1=0.00384818171835,
        market_overnight_gap_q10_v1=-0.00382056890751,
        market_overnight_gap_q90_v1=0.0063796856309,
        market_total_transition_q10_v1=-0.00536060944383,
        market_total_transition_q90_v1=0.00643755517767,
        market_opening_return_support_v1=247,
        market_opening_range_support_v1=247,
        market_overnight_gap_support_v1=247,
        market_total_transition_support_v1=247,
        calibration_complete_v1=True,
        calibration_missing_reason_v1=None,
    )
    engine = FrozenM1CRecorderEngine(
        m1c_runtime=cast(Any, FakeRuntime()),
        m1c_features=cast(Any, FakeFeatureBuilder()),
        direction_runtime=cast(Any, object()),
        direction_features=cast(Any, object()),
        repository=repository,
        opening_transition_thresholds_v1=thresholds,
        opening_transition_activation_status_v1="available",
        opening_reversal_activation_v1=base,
        opening_reversal_activation_v1_1=activation,
    )
    engine.set_opening_reversal_capacity_snapshot_provider_v1(lambda _metadata: "capacity-1")

    result = engine.process_checkpoint(
        RecorderCheckpointInput(
            metadata=metadata,
            symbol="AAL",
            session=SESSION,
            completed_m1c_bars=stock_bars,
            completed_direction_bars=(),
            group_o_context=context,
            market_data_type=MarketDataType.LIVE,
            capability_preflight_passed=True,
            m1c_parity_passed=True,
            direction_parity_passed=False,
            clock_drift_within_tolerance=True,
            underlying_quote_fresh=True,
            unresolved_bar_gap=False,
            raw_event_storage_writable=True,
            completed_market_shock_bars_v1=market_bars,
            market_previous_session_v1=date(2026, 7, 29),
            market_prior_regular_session_close_v1=100.0,
            opening_reversal_receipt_created_at_utc_v1_1=RECEIPT_CREATED,
            opening_reversal_first_buffered_event_received_at_utc_v1_1=(
                ENTRY + timedelta(milliseconds=1)
            ),
            opening_reversal_entry_data_admitted_before_receipt_v1_1=False,
        )
    )

    receipt = result.opening_reversal_prediction_v1
    assert receipt is not None
    assert receipt.experiment_version == "1.1"
    assert receipt.eligibility_v1
    assert receipt.receipt_created_at_utc == RECEIPT_CREATED
    assert receipt.timing_evidence_v1_1 is not None
    assert receipt.timing_evidence_v1_1.entry_or_post_entry_data_admitted_before_receipt is False
