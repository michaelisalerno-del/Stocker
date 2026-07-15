from __future__ import annotations

import math

import pandas as pd
import pytest

from stocker_research.dynamic_loop_edge_state_lead_lag.lead_targets import (
    LeadRegistration,
    build_frozen_forecast_ledger,
    build_lead_target_joins,
    build_settled_outcome_ledger,
)


def _forecast_rows() -> pd.DataFrame:
    sessions = ["2025-01-03", "2025-01-06", "2025-01-08"]
    freeze = pd.to_datetime([f"{session}T14:30:00Z" for session in sessions])
    return pd.DataFrame(
        {
            "period": 2025,
            "score_session": sessions,
            "loop_id": "cycle_01",
            "orientation": "state_2",
            "horizon": 24,
            "model_name": "hierarchical_change_point",
            "prediction_frozen_at": freeze,
            "decision_timestamp": freeze,
            "feature_max_availability_timestamp": freeze - pd.Timedelta(hours=16),
            "training_latest_availability_timestamp": freeze - pd.Timedelta(hours=15),
            "p_next_payoff_positive": [0.4, 0.7, 0.6],
            "p_edge_positive": [0.45, 0.55, 0.58],
            "p_edge_active": [0.2, 0.8, 0.7],
            "p_change_now": [0.05, 0.1, 0.08],
            "p_on_next": [0.02, 0.04, 0.03],
            "p_off_next": [0.04, 0.02, 0.03],
            "p_survive_horizon": [0.98, 0.99, 0.98],
            "posterior_mean_net_bps": [-2.0, 5.0, 3.0],
            "posterior_lower_bound_net_bps": [-8.0, 1.0, 0.5],
            "posterior_run_length_mean": [4.0, 5.0, 6.0],
            "edge_state": ["unknown", "active", "active"],
            "reason_codes": [
                "edge_probability_too_low",
                "admitted_active_edge",
                "admitted_active_edge",
            ],
            "effective_sessions": [5.0, 6.0, 7.0],
            "independent_stocks": [4, 5, 6],
            "effective_sample_size": [10.0, 12.0, 14.0],
            "z__structural_breadth": [-0.5, 1.0, 0.6],
            "run_id": "v2-run",
            "model_version": "v2",
            "configuration_hash": "v2-config",
            "feature_schema_version": "features-v2",
        }
    )


def _metadata() -> dict[str, object]:
    return {
        "run_id": "lead-lag-run",
        "git_sha": "abc123",
        "contract_hash": "contract123",
        "data_snapshot_hash": "data123",
        "experiment_version": "lead-lag-v1",
    }


def test_forecast_ids_and_values_do_not_change_when_future_rows_are_appended() -> None:
    original = _forecast_rows().iloc[:2].copy()
    with_future = _forecast_rows()

    before = build_frozen_forecast_ledger(original, _metadata())
    after = build_frozen_forecast_ledger(with_future, _metadata()).iloc[:2]

    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_forecast_rejects_features_from_the_future_session() -> None:
    rows = _forecast_rows().iloc[:1].copy()
    rows["feature_max_availability_timestamp"] = rows["prediction_frozen_at"] + pd.Timedelta(
        seconds=1
    )

    with pytest.raises(ValueError, match="feature availability"):
        build_frozen_forecast_ledger(rows, _metadata())


def test_explicit_calendar_joins_next_trading_session_not_next_source_row() -> None:
    forecasts = build_frozen_forecast_ledger(_forecast_rows().iloc[:2], _metadata())
    outcomes = build_settled_outcome_ledger(
        pd.DataFrame(
            {
                "period": [2025],
                "session": ["2025-01-06"],
                "loop_id": ["cycle_01"],
                "orientation": ["state_2"],
                "horizon": [24],
                "robust_net_payoff_bps": [12.0],
                "robust_gross_payoff_bps": [22.0],
                "cost_contribution_bps": [10.0],
                "independent_stock_count": [5],
                "independent_stock_ids": ['["AAA","BBB","CCC","DDD","EEE"]'],
                "effective_sample_size": [5.0],
                "data_availability_timestamp": [pd.Timestamp("2025-01-06T20:00:00Z")],
                "source_data_id": ["source-1"],
            }
        ),
        _metadata(),
    )
    cell_calendar = _forecast_rows().loc[
        :, ["period", "score_session", "loop_id", "orientation", "horizon"]
    ]
    opportunities = pd.DataFrame(
        {
            "period": [2025],
            "score_session": ["2025-01-06"],
            "loop_id": ["cycle_01"],
            "orientation": ["state_2"],
            "horizon": [24],
            "status": ["filled"],
            "settlement_timestamp": [pd.Timestamp("2025-01-06T20:00:00Z")],
        }
    )

    joined = build_lead_target_joins(
        forecasts,
        outcomes,
        cell_calendar,
        opportunities,
        LeadRegistration(),
    )
    friday_lead_1 = joined.loc[
        joined["score_session"].eq("2025-01-03") & joined["target_lead_sessions"].eq(1)
    ].iloc[0]

    assert friday_lead_1["target_session"] == "2025-01-06"
    assert friday_lead_1["target_status"] == "payoff_settled"
    assert friday_lead_1["target_robust_net_bps"] == 12.0


def test_missing_payoff_is_missing_and_period_boundary_never_crosses() -> None:
    forecasts = build_frozen_forecast_ledger(_forecast_rows(), _metadata())
    outcomes = build_settled_outcome_ledger(
        pd.DataFrame(
            columns=[
                "period",
                "session",
                "loop_id",
                "orientation",
                "horizon",
                "robust_net_payoff_bps",
                "robust_gross_payoff_bps",
                "cost_contribution_bps",
                "independent_stock_count",
                "independent_stock_ids",
                "effective_sample_size",
                "data_availability_timestamp",
                "source_data_id",
            ]
        ),
        _metadata(),
    )
    calendar = _forecast_rows().loc[
        :, ["period", "score_session", "loop_id", "orientation", "horizon"]
    ]

    joined = build_lead_target_joins(
        forecasts,
        outcomes,
        calendar,
        pd.DataFrame(
            columns=[
                "period",
                "score_session",
                "loop_id",
                "orientation",
                "horizon",
                "status",
                "settlement_timestamp",
            ]
        ),
        LeadRegistration(),
    )
    first_same_session = joined.loc[
        joined["score_session"].eq("2025-01-03") & joined["target_lead_sessions"].eq(0)
    ].iloc[0]
    last_lead_1 = joined.loc[
        joined["score_session"].eq("2025-01-08") & joined["target_lead_sessions"].eq(1)
    ].iloc[0]

    assert first_same_session["target_status"] == "no_opportunity"
    assert math.isnan(first_same_session["target_robust_net_bps"])
    assert last_lead_1["target_status"] == "period_boundary"
    assert pd.isna(last_lead_1["target_session"])


def test_registered_leads_are_deterministic_and_primary_cannot_be_relabelled() -> None:
    registration = LeadRegistration()

    assert registration.leads == (0, 1, 2, 3, 5)
    assert registration.primary_lead == 1
    with pytest.raises(ValueError, match="primary lead"):
        LeadRegistration(primary_lead=2)


def test_predictive_and_latent_probabilities_remain_separate() -> None:
    ledger = build_frozen_forecast_ledger(_forecast_rows().iloc[:1], _metadata())

    assert ledger["p_next_payoff_positive"].iloc[0] == 0.4
    assert ledger["p_edge_positive"].iloc[0] == 0.45
    assert ledger["forecast_id"].is_unique
