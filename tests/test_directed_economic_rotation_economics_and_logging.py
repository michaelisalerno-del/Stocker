from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stocker_research.directed_economic_rotation import (
    ProspectiveRotationLedger,
    apply_cost_stress,
    translate_predictions_to_opportunities,
)

FAMILY_A = "two_transition_return_cycle__state_1"
FAMILY_B = "two_transition_return_cycle__state_2"


def _forecast() -> dict[str, object]:
    return {
        "run_id": "rotation-holdout",
        "git_sha": "abc",
        "contract_hash": "def",
        "data_snapshot_hash": "new-data",
        "model_version": "directed_economic_loop_regime_rotation_v1.0.0",
        "forecast_id": "forecast-1",
        "forecast_session": "2099-01-05",
        "forecast_timestamp": "2099-01-05T14:30:00Z",
        "target_window_sessions": 3,
        "destination_family": FAMILY_B,
        "destination_pair": None,
        "destination_current_economic_state": "retired",
        "destination_own_history_features": {"p_on_next": 0.2},
        "source_family_state_vector": {FAMILY_A: ["newly_decaying"]},
        "system_state_features": {"active_family_fraction": 0.2},
        "predicted_activation_probability": 0.4,
        "activation_base_rate": 0.1,
        "predicted_lift_over_base": 4.0,
        "probability_interval_lower": 0.2,
        "probability_interval_upper": 0.6,
        "probability_no_activation": 0.5,
        "probability_multiple_activation": 0.1,
        "prediction_state": "nominated",
        "reason_codes": [],
        "feature_availability_timestamp": "2099-01-04T20:00:00Z",
        "training_cutoff": "2099-01-04T20:00:00Z",
        "forecast_freeze_timestamp": "2099-01-05T14:30:00Z",
    }


def test_prospective_rotation_forecast_is_create_only_and_outcome_is_separate(
    tmp_path: Path,
) -> None:
    ledger = ProspectiveRotationLedger(tmp_path, opened_periods={2023, 2024, 2025, 2026})
    forecast_path = ledger.append_forecast(_forecast(), holdout=True)
    before = forecast_path.read_bytes()

    outcome = ledger.append_outcome(
        {
            "outcome_id": "outcome-1",
            "forecast_id": "forecast-1",
            "target_start_session": "2099-01-06",
            "target_end_session": "2099-01-08",
            "settlement_timestamp": "2099-01-10T20:00:00Z",
            "destination_activation_observed": True,
            "multiple_activation_flag": False,
        }
    )

    assert forecast_path.read_bytes() == before
    assert outcome.exists()
    with pytest.raises(FileExistsError):
        ledger.append_forecast(_forecast(), holdout=True)
    with pytest.raises(FileExistsError):
        ledger.append_outcome(
            {
                "outcome_id": "outcome-1",
                "forecast_id": "forecast-1",
                "target_start_session": "2099-01-06",
                "target_end_session": "2099-01-08",
                "settlement_timestamp": "2099-01-10T20:00:00Z",
                "destination_activation_observed": False,
                "multiple_activation_flag": False,
            }
        )


def test_prospective_holdout_rejects_opened_data_future_features_and_targets(
    tmp_path: Path,
) -> None:
    ledger = ProspectiveRotationLedger(tmp_path, opened_periods={2023, 2024, 2025, 2026})
    opened = _forecast()
    opened["forecast_id"] = "opened"
    opened["forecast_session"] = "2025-01-05"
    with pytest.raises(ValueError, match="opened period"):
        ledger.append_forecast(opened, holdout=True)

    future = _forecast()
    future["forecast_id"] = "future"
    future["feature_availability_timestamp"] = "2099-01-06T20:00:00Z"
    with pytest.raises(ValueError, match="feature availability"):
        ledger.append_forecast(future, holdout=True)

    leaked = _forecast()
    leaked["forecast_id"] = "leaked"
    leaked["activation_target"] = True
    with pytest.raises(ValueError, match="future target"):
        ledger.append_forecast(leaked, holdout=True)


def test_trade_translation_uses_only_later_matching_family_and_never_replaces() -> None:
    calendar = pd.DataFrame(
        {
            "period": 2025,
            "score_session": ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"],
        }
    )
    predictions = pd.DataFrame(
        [
            {
                "forecast_id": "forecast-b",
                "period": 2025,
                "forecast_session": "2025-01-03",
                "destination_family": FAMILY_B,
                "target_window_sessions": 3,
                "model_name": "M3_directed_family_rotation",
                "prediction_state": "nominated",
                "predicted_activation_probability": 0.4,
            },
            {
                "forecast_id": "forecast-a",
                "period": 2025,
                "forecast_session": "2025-01-03",
                "destination_family": FAMILY_A,
                "target_window_sessions": 3,
                "model_name": "M3_directed_family_rotation",
                "prediction_state": "nominated",
                "predicted_activation_probability": 0.4,
            },
        ]
    )
    opportunities = pd.DataFrame(
        [
            {
                "opportunity_id": "b-later",
                "period": 2025,
                "score_session": "2025-01-06",
                "destination_family": FAMILY_B,
                "status": "filled",
                "stock_id": "AAA",
                "entry_timestamp": pd.Timestamp("2025-01-06 15:00Z"),
                "exit_timestamp": pd.Timestamp("2025-01-06 17:00Z"),
                "gross_payoff_bps": 25.0,
                "primary_total_cost_bps": 10.0,
                "primary_net_payoff_bps": 15.0,
            },
            {
                "opportunity_id": "wrong-family",
                "period": 2025,
                "score_session": "2025-01-07",
                "destination_family": "two_transition_return_cycle__state_7",
                "status": "filled",
                "stock_id": "BBB",
                "entry_timestamp": pd.Timestamp("2025-01-07 15:00Z"),
                "exit_timestamp": pd.Timestamp("2025-01-07 17:00Z"),
                "gross_payoff_bps": 100.0,
                "primary_total_cost_bps": 10.0,
                "primary_net_payoff_bps": 90.0,
            },
        ]
    )

    translated = translate_predictions_to_opportunities(predictions, opportunities, calendar)

    matched = translated.loc[translated["economic_translation_status"].eq("eligible_opportunity")]
    missing = translated.loc[
        translated["economic_translation_status"].eq("no_tradeable_destination_opportunity")
    ]
    assert matched["opportunity_id"].tolist() == ["b-later"]
    assert matched["primary_net_payoff_bps"].item() == 15.0
    assert missing["forecast_id"].tolist() == ["forecast-a"]
    assert "wrong-family" not in translated["opportunity_id"].dropna().tolist()


def test_recent_forecast_receives_opportunity_once_without_capacity_refill() -> None:
    calendar = pd.DataFrame(
        {
            "period": 2025,
            "score_session": ["2025-01-03", "2025-01-06", "2025-01-07"],
        }
    )
    predictions = pd.DataFrame(
        [
            {
                "forecast_id": identifier,
                "period": 2025,
                "forecast_session": session,
                "destination_family": FAMILY_B,
                "target_window_sessions": 3,
                "model_name": "M3_directed_family_rotation",
                "prediction_state": "nominated",
                "predicted_activation_probability": 0.5,
            }
            for identifier, session in (("older", "2025-01-03"), ("newer", "2025-01-06"))
        ]
    )
    opportunities = pd.DataFrame(
        {
            "opportunity_id": ["one-trade"],
            "period": [2025],
            "score_session": ["2025-01-07"],
            "destination_family": [FAMILY_B],
            "status": ["filled"],
            "stock_id": ["AAA"],
            "entry_timestamp": pd.to_datetime(["2025-01-07 15:00Z"], utc=True),
            "exit_timestamp": pd.to_datetime(["2025-01-07 17:00Z"], utc=True),
            "gross_payoff_bps": [20.0],
            "primary_total_cost_bps": [10.0],
            "primary_net_payoff_bps": [10.0],
        }
    )

    translated = translate_predictions_to_opportunities(predictions, opportunities, calendar)
    matched = translated.loc[translated["economic_translation_status"].eq("eligible_opportunity")]

    assert len(matched) == 1
    assert matched["forecast_id"].item() == "newer"
    assert matched["opportunity_id"].is_unique


def test_twice_cost_stress_charges_entry_and_exit_again_without_changing_exit() -> None:
    frame = pd.DataFrame(
        {
            "opportunity_id": ["trade-1"],
            "gross_payoff_bps": [25.0],
            "primary_total_cost_bps": [10.0],
            "primary_net_payoff_bps": [15.0],
            "entry_timestamp": pd.to_datetime(["2025-01-06 15:00Z"], utc=True),
            "exit_timestamp": pd.to_datetime(["2025-01-06 17:00Z"], utc=True),
        }
    )
    before_exit = frame["exit_timestamp"].copy()

    stressed = apply_cost_stress(frame, multiplier=2.0)

    assert stressed["stressed_total_cost_bps"].item() == 20.0
    assert stressed["stressed_net_payoff_bps"].item() == 5.0
    pd.testing.assert_series_equal(stressed["exit_timestamp"], before_exit)


def test_serialized_forecast_carries_research_only_safety_flags(tmp_path: Path) -> None:
    ledger = ProspectiveRotationLedger(tmp_path, opened_periods=set())
    path = ledger.append_forecast(_forecast(), holdout=True)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["research_only"] is True
    assert record["execution_enabled"] is False
    assert record["broker_connection_enabled"] is False
    assert record["order_placement_enabled"] is False
    assert record["position_management_enabled"] is False
