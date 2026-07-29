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
from stocker_prospective.recorder_repository import FrozenRecorderRepository
from stocker_prospective.recorder_v0 import (
    FrozenM1CRecorderEngine,
    RecorderCheckpointInput,
)
from stocker_prospective.signed_market_shock_v1 import (
    CheckpointShockThresholdsV1,
    MarketShockBarV1,
    calculate_preentry_windows_v1,
    calculate_stock_shock_response_v1,
    classify_market_shock_state_v1,
)

SESSION = date(2025, 1, 2)
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
        for ordinal in range(8)
    )


def test_recorder_persists_signed_shock_fields_as_logging_only(
    tmp_path: Path,
) -> None:
    database = ProspectiveRepository(tmp_path / "signed-market-shock.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id="signed-market-shock-v1",
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
    repository = FrozenRecorderRepository(
        database,
        configuration_hash="frozen-recorder-test",
    )
    market_bars = _bars("VTI")
    stock_bars = _bars("AAL")
    signal = market_bars[-1].bar_complete_timestamp
    windows = calculate_preentry_windows_v1(
        market_proxy="VTI",
        session=SESSION,
        checkpoint=8,
        signal_timestamp=signal,
        completed_bars=market_bars,
    )
    thresholds = CheckpointShockThresholdsV1(
        checkpoint=8,
        market_return_w0_q10_v1=-0.05,
        market_return_w0_q90_v1=0.01,
        market_range_w0_q75_v1=0.01,
        market_return_w1_q10_v1=-0.05,
        market_return_w1_q90_v1=0.05,
        market_range_w1_q75_v1=0.01,
        market_return_w0_support_v1=252,
        market_range_w0_support_v1=252,
        market_return_w1_support_v1=252,
        market_range_w1_support_v1=252,
        calibration_complete_v1=True,
        calibration_missing_reason_v1=None,
    )
    state = classify_market_shock_state_v1(
        windows=windows,
        thresholds=thresholds,
    )
    response = calculate_stock_shock_response_v1(
        symbol="AAL",
        session=SESSION,
        checkpoint=8,
        signal_timestamp=signal,
        completed_stock_bars=stock_bars,
        market_return_w0_v1=windows.market_return_w0_v1,
        market_shock_state_v1=state,
        threshold_15m=0.01,
    )
    score = FrozenM1CScore(
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

    checkpoint_id = repository.record_checkpoint(
        metadata,
        symbol="AAL",
        session=SESSION,
        checkpoint=8,
        bar_start_utc=stock_bars[-1].bar_start_timestamp,
        bar_end_utc=signal,
        score=score,
        session_context_hash="d" * 64,
        feature_values={"x": 1.0},
        eligible=True,
        feature_freshness="fresh",
        rejection_reasons=(),
    )
    repository.record_signed_market_shock_checkpoint_v1(
        metadata,
        checkpoint_id=checkpoint_id,
        symbol="AAL",
        session=SESSION,
        checkpoint=8,
        market_windows_v1=windows,
        market_shock_state_v1=state,
        stock_shock_response_v1=response,
        market_shock_thresholds_v1=thresholds,
        activation_status_v1="available",
    )

    with database._connect() as connection:
        row = connection.execute(
            """
            SELECT probability, threshold, threshold_passed, eligible,
                   canonical_market_proxy_v1, market_return_w0_v1,
                   market_shock_state_v1, market_shock_event_id_v1,
                   shock_sign_v1, stock_return_w0_v1,
                   shock_relative_response_v1, shock_response_class_v1,
                   market_shock_complete_v1, shock_response_complete_v1,
                   signed_market_shock_source_v1_json
            FROM m1c_checkpoint_v0
            """
        ).fetchone()

    assert row is not None
    assert row["probability"] == score.probability
    assert row["threshold"] == score.threshold
    assert bool(row["threshold_passed"]) is score.threshold_passed
    assert bool(row["eligible"]) is True
    assert row["canonical_market_proxy_v1"] == "VTI"
    assert row["market_return_w0_v1"] == windows.market_return_w0_v1
    assert row["market_shock_state_v1"] == "POSITIVE_SHOCK_ONSET"
    assert row["market_shock_event_id_v1"] == state.market_shock_event_id_v1
    assert row["shock_sign_v1"] == 1
    assert row["stock_return_w0_v1"] == response.stock_return_w0_v1
    assert row["shock_relative_response_v1"] == response.shock_relative_response_v1
    assert row["shock_response_class_v1"] == response.shock_response_class_v1
    assert bool(row["market_shock_complete_v1"])
    assert bool(row["shock_response_complete_v1"])
    source = json.loads(str(row["signed_market_shock_source_v1_json"]))
    assert source["logging_only"] is True
    assert source["m1c_scoring_changed"] is False
    assert source["episode_promotion_changed"] is False
    assert source["subscription_allocation_changed"] is False
    assert source["direction_decision_changed"] is False
    assert source["order_routing_changed"] is False


@pytest.mark.parametrize("failure_stage", ["calculation", "persistence"])
def test_signed_shock_logging_failure_cannot_suppress_fresh_episode(
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

    database = ProspectiveRepository(tmp_path / f"{failure_stage}.sqlite3")
    database.migrate()
    metadata = EvidenceMetadata(
        run_id=f"signed-shock-{failure_stage}",
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
    repository = FrozenRecorderRepository(database)
    context = build_group_o_context(
        symbol="AAL",
        signal_session=SESSION,
        actual_option_observation_session=date(2024, 12, 31),
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
    bars = tuple(
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
    )

    def fail(*_: object, **__: object) -> None:
        raise ValueError(f"injected {failure_stage} failure")

    if failure_stage == "calculation":
        monkeypatch.setattr(
            "stocker_prospective.recorder_v0.calculate_preentry_windows_v1",
            fail,
        )
    else:
        monkeypatch.setattr(
            repository,
            "record_signed_market_shock_checkpoint_v1",
            fail,
        )

    result = engine.process_checkpoint(
        RecorderCheckpointInput(
            metadata=metadata,
            symbol="AAL",
            session=SESSION,
            completed_m1c_bars=bars,
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
        )
    )

    assert result.episode_decision.fresh_episode
    assert result.market_shock_state_v1.market_shock_state_v1 == "UNKNOWN_INCOMPLETE"
    with database._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM m1c_episode_v0"
        ).fetchone()[0] == 1
