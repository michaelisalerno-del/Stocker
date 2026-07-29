from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.frozen_m1c import FrozenM1CScore
from stocker_prospective.group_o import build_group_o_context
from stocker_prospective.m1c_features import LiveFeatureBar
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.opening_market_transition_v1 import (
    OpeningTransitionThresholdsV1,
    calculate_opening_preentry_window_v1,
    calculate_stock_opening_response_v1,
    classify_opening_market_transition_v1,
)
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import (
    FrozenM1CRecorderEngine,
    RecorderCheckpointInput,
)
from stocker_prospective.signed_market_shock_v1 import MarketShockBarV1

SESSION = date(2025, 1, 2)
PREVIOUS_SESSION = date(2024, 12, 31)
START = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)


def _bars(symbol: str) -> tuple[MarketShockBarV1, ...]:
    return tuple(
        MarketShockBarV1(
            symbol=symbol,
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=START + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=START + timedelta(minutes=5 * (ordinal + 1)),
            open=100.0 + ordinal,
            high=101.0 + ordinal,
            low=99.0 + ordinal,
            close=100.5 + ordinal,
            finalised=True,
        )
        for ordinal in range(6)
    )


def _thresholds() -> OpeningTransitionThresholdsV1:
    return OpeningTransitionThresholdsV1(
        market_opening_return_q10_v1=-0.05,
        market_opening_return_q90_v1=0.01,
        market_opening_range_q75_v1=0.01,
        market_overnight_gap_q10_v1=-0.01,
        market_overnight_gap_q90_v1=0.01,
        market_total_transition_q10_v1=-0.05,
        market_total_transition_q90_v1=0.05,
        market_opening_return_support_v1=247,
        market_opening_range_support_v1=247,
        market_overnight_gap_support_v1=247,
        market_total_transition_support_v1=247,
        calibration_complete_v1=True,
        calibration_missing_reason_v1=None,
    )


def _metadata(run_id: str) -> EvidenceMetadata:
    return EvidenceMetadata(
        run_id=run_id,
        prospective_start_utc=START,
        app_version="test",
        git_commit="a" * 40,
        model_artifact_id="M1C",
        universe_id="frozen-20",
        cohort="anchor_frozen_20",
        source_timestamps=[START.isoformat()],
        recorded_at_utc=START,
    )


def _score() -> FrozenM1CScore:
    return FrozenM1CScore(
        model_hash="b" * 64,
        probability=0.60,
        threshold=0.488333710794033,
        threshold_passed=True,
        feature_order=("x",),
        feature_values=(1.0,),
        transformed_values=(1.0,),
        feature_hash="c" * 64,
        missing_feature_count=0,
    )


def test_recorder_persists_opening_transition_fields_as_logging_only(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "opening-transition.sqlite3")
    database.migrate()
    metadata = _metadata("opening-transition-v1")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(
        database,
        configuration_hash="frozen-recorder-test",
    )
    market_bars = _bars("VTI")
    stock_bars = _bars("AAL")
    signal = START + timedelta(minutes=30)
    window = calculate_opening_preentry_window_v1(
        market_proxy="VTI",
        session=SESSION,
        previous_session=PREVIOUS_SESSION,
        session_open_timestamp=START,
        signal_timestamp=signal,
        entry_timestamp=signal,
        completed_bars=market_bars,
        prior_regular_session_close=99.0,
    )
    state = classify_opening_market_transition_v1(
        window=window,
        thresholds=_thresholds(),
    )
    response = calculate_stock_opening_response_v1(
        symbol="AAL",
        session=SESSION,
        session_open_timestamp=START,
        signal_timestamp=signal,
        completed_stock_bars=stock_bars,
        market_opening_return_v1=window.market_opening_return_v1,
        opening_transition_state_v1=state,
        threshold_15m=0.01,
    )
    score = _score()
    checkpoint_id = repository.record_checkpoint(
        metadata,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        bar_start_utc=stock_bars[-1].bar_start_timestamp,
        bar_end_utc=signal,
        score=score,
        session_context_hash="d" * 64,
        feature_values={"x": 1.0},
        eligible=True,
        feature_freshness="fresh",
        rejection_reasons=(),
    )

    repository.record_opening_market_transition_checkpoint_v1(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=SESSION,
        checkpoint=6,
        opening_window_v1=window,
        opening_transition_state_v1=state,
        stock_opening_response_v1=response,
        opening_thresholds_v1=_thresholds(),
        activation_status_v1="available",
    )

    with database._connect() as connection:
        row = connection.execute(
            """
            SELECT probability, threshold, threshold_passed, eligible,
                   opening_market_proxy_v1, vti_session_open_v1,
                   vti_prior_regular_session_close_v1,
                   opening_expected_bar_count_v1,
                   opening_observed_bar_count_v1,
                   market_opening_return_v1, market_opening_range_v1,
                   market_overnight_gap_v1, market_total_transition_v1,
                   market_gap_open_alignment_v1,
                   opening_market_transition_state_v1,
                   opening_transition_event_id_v1,
                   opening_transition_sign_v1,
                   stock_opening_return_v1,
                   stock_relative_opening_response_v1,
                   stock_opening_response_class_v1,
                   opening_market_complete_v1,
                   stock_opening_response_complete_v1,
                   opening_market_transition_source_v1_json
            FROM m1c_checkpoint_v0
            """
        ).fetchone()

    assert row is not None
    assert row["probability"] == score.probability
    assert row["threshold"] == score.threshold
    assert bool(row["threshold_passed"]) is score.threshold_passed
    assert bool(row["eligible"]) is True
    assert row["opening_market_proxy_v1"] == "VTI"
    assert row["vti_session_open_v1"] == window.market_session_open_v1
    assert row["vti_prior_regular_session_close_v1"] == window.market_prior_regular_session_close_v1
    assert row["opening_expected_bar_count_v1"] == 6
    assert row["opening_observed_bar_count_v1"] == 6
    assert row["market_opening_return_v1"] == window.market_opening_return_v1
    assert row["opening_market_transition_state_v1"] == ("POSITIVE_SEVERE_OPENING_TRANSITION")
    assert row["opening_transition_sign_v1"] == 1
    assert row["stock_opening_return_v1"] == response.stock_opening_return_v1
    assert row["stock_relative_opening_response_v1"] == (
        response.stock_relative_opening_response_v1
    )
    assert bool(row["opening_market_complete_v1"])
    assert bool(row["stock_opening_response_complete_v1"])
    source = json.loads(str(row["opening_market_transition_source_v1_json"]))
    assert source["logging_only"] is True
    assert source["m1c_scoring_changed"] is False
    assert source["episode_promotion_changed"] is False
    assert source["subscription_allocation_changed"] is False
    assert source["direction_decision_changed"] is False
    assert source["order_routing_changed"] is False


@pytest.mark.parametrize("failure_stage", ["calculation", "persistence"])
def test_opening_logging_failure_cannot_suppress_fresh_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
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
            return _score()

    database = ProspectiveRepository(tmp_path / f"{failure_stage}.sqlite3")
    database.migrate()
    metadata = _metadata(f"opening-{failure_stage}")
    database.create_run(metadata)
    repository = FrozenRecorderRepository(database)
    context = build_group_o_context(
        symbol="AAL",
        signal_session=SESSION,
        actual_option_observation_session=PREVIOUS_SESSION,
        front_expiry=date(2025, 1, 3),
        dte=1,
        atm_strike=12.0,
        previous_close_implied_movement_15m=0.01,
        features={"x": 1.0},
        missing_indicators={"x": False},
        quality_status="valid",
        source_receipt_hashes=("a" * 64,),
    )
    repository.record_group_o_context(metadata, context)
    live_bars = tuple(
        LiveFeatureBar(
            symbol="AAL",
            session=SESSION,
            bar_ordinal=ordinal,
            bar_start_timestamp=START + timedelta(minutes=5 * ordinal),
            bar_complete_timestamp=START + timedelta(minutes=5 * (ordinal + 1)),
            open=100.0 + ordinal,
            high=101.0 + ordinal,
            low=99.0 + ordinal,
            close=100.5 + ordinal,
            volume=1_000.0,
            historical_relative_activity=1.0,
            finalised=True,
            source="fixture",
        )
        for ordinal in range(6)
    )
    engine = FrozenM1CRecorderEngine(
        m1c_runtime=cast(Any, FakeRuntime()),
        m1c_features=cast(Any, FakeFeatureBuilder()),
        direction_runtime=cast(Any, object()),
        direction_features=cast(Any, object()),
        repository=repository,
        opening_transition_thresholds_v1=_thresholds(),
        opening_transition_activation_status_v1="available",
    )

    def fail(*_: object, **__: object) -> None:
        raise ValueError(f"injected {failure_stage} failure")

    if failure_stage == "calculation":
        monkeypatch.setattr(
            "stocker_prospective.recorder_v0.calculate_opening_preentry_window_v1",
            fail,
        )
    else:
        monkeypatch.setattr(
            repository,
            "record_opening_market_transition_checkpoint_v1",
            fail,
        )

    result = engine.process_checkpoint(
        RecorderCheckpointInput(
            metadata=metadata,
            symbol="AAL",
            session=SESSION,
            completed_m1c_bars=live_bars,
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
            scientific_recording_authorized=True,
            completed_market_shock_bars_v1=_bars("VTI"),
            market_previous_session_v1=PREVIOUS_SESSION,
            market_prior_regular_session_close_v1=99.0,
        )
    )

    assert result.episode_decision.fresh_episode
    assert result.opening_transition_state_v1.opening_market_transition_state_v1 == (
        "UNKNOWN_INCOMPLETE"
    )
    with database._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM m1c_episode_v0").fetchone()[0] == 1
