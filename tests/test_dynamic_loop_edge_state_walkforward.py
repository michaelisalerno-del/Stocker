from __future__ import annotations

import json

import pandas as pd
from pandas.testing import assert_frame_equal

from stocker_research.dynamic_loop_edge_state.decision import DecisionThresholds
from stocker_research.dynamic_loop_edge_state.online_state import (
    BOCPDSettings,
    HierarchicalSettings,
)
from stocker_research.dynamic_loop_edge_state.walkforward import (
    WalkForwardSettings,
    apply_frozen_admission,
    run_causal_walk_forward,
)

CELL = ("cycle_04", "state_4", 24)
FEATURES = (
    "structural_breadth",
    "breadth_change",
    "top_second_margin",
)


def _calendar(days: int) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=days, freq="B")
    return pd.DataFrame(
        {
            "score_session": dates.strftime("%Y-%m-%d"),
            "decision_timestamp": dates.tz_localize("UTC") + pd.Timedelta(hours=14, minutes=30),
        }
    )


def _panel(days: int, *, late_first: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=days, freq="B")
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        availability = date.tz_localize("UTC") + pd.Timedelta(hours=17)
        if late_first and index == 0:
            next_day = dates[1].tz_localize("UTC")
            availability = next_day + pd.Timedelta(hours=15)
        rows.append(
            {
                "session": date.strftime("%Y-%m-%d"),
                "loop_id": CELL[0],
                "orientation": CELL[1],
                "horizon": CELL[2],
                "robust_net_payoff_bps": 20.0 + index,
                "effective_sample_size": 4.0,
                "independent_stock_count": 4,
                "independent_stock_ids": json.dumps(["A", "B", "C", "D"]),
                "raw_fill_count": 4,
                "data_availability_timestamp": availability,
            }
        )
    return pd.DataFrame(rows)


def _features(
    days: int, *, future_at: int | None = None, extreme_last: bool = False
) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=days, freq="B")
    rows: list[dict[str, object]] = []
    for index, date in enumerate(dates):
        decision = date.tz_localize("UTC") + pd.Timedelta(hours=14, minutes=30)
        availability = decision - pd.Timedelta(minutes=1)
        if future_at == index:
            availability = decision + pd.Timedelta(minutes=1)
        value = 10_000.0 if extreme_last and index == days - 1 else 0.1 * index
        rows.append(
            {
                "score_session": date.strftime("%Y-%m-%d"),
                "loop_id": CELL[0],
                "orientation": CELL[1],
                "horizon": CELL[2],
                "feature_availability_timestamp": availability,
                "structural_breadth": 0.3 + value,
                "breadth_change": value,
                "top_second_margin": 0.2 + value,
            }
        )
    return pd.DataFrame(rows)


def _run(
    days: int,
    *,
    panel: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return run_causal_walk_forward(
        session_calendar=_calendar(days),
        payoff_panel=_panel(days) if panel is None else panel,
        feature_panel=_features(days) if features is None else features,
        cell_keys=[CELL],
        bocpd_settings=BOCPDSettings(),
        hierarchy_settings=HierarchicalSettings(
            pooling_strength_sessions=4.0,
            feature_logit_weights={
                "structural_breadth": 0.25,
                "breadth_change": 0.5,
                "top_second_margin": 0.25,
            },
        ),
        decision_thresholds=DecisionThresholds(
            minimum_independent_sessions=2,
            minimum_independent_stocks=4,
            minimum_effective_sample_size=4.0,
            maximum_posterior_std_net_bps=200.0,
            active_probability=0.8,
            survival_probability=0.8,
        ),
        settings=WalkForwardSettings(
            run_id="test-run",
            model_name="hierarchical_change_point",
            model_version="test-v2",
            configuration_hash="abc123",
            feature_schema_version="features-v2",
            cost_model_version="costs-v1",
            horizon_bars=24,
            session_bars=78,
            required_features=FEATURES,
        ),
    )


def test_current_session_outcome_cannot_train_the_gate_applied_to_that_session() -> None:
    forecasts = _run(3)

    first = forecasts.loc[forecasts["score_session"].eq("2025-01-02")].iloc[0]
    second = forecasts.loc[forecasts["score_session"].eq("2025-01-03")].iloc[0]
    assert first["settled_observation_count"] == 0
    assert first["effective_sessions"] == 0.0
    assert second["settled_observation_count"] == 1
    assert second["training_latest_source_session"] == "2025-01-02"


def test_unsettled_prior_trade_is_excluded_until_a_later_decision() -> None:
    forecasts = _run(4, panel=_panel(4, late_first=True))

    jan3 = forecasts.loc[forecasts["score_session"].eq("2025-01-03")].iloc[0]
    jan6 = forecasts.loc[forecasts["score_session"].eq("2025-01-06")].iloc[0]
    assert jan3["settled_observation_count"] == 0
    assert "unresolved_outcomes" in jan3["reason_codes"]
    assert jan6["settled_observation_count"] >= 2


def test_appending_future_data_does_not_change_frozen_historical_predictions() -> None:
    short = _run(4)
    long = _run(7)

    columns = [
        "score_session",
        "p_edge_active",
        "posterior_mean_net_bps",
        "posterior_std_net_bps",
        "edge_state",
        "reason_codes",
    ]
    assert_frame_equal(
        short.loc[:, columns].reset_index(drop=True),
        long.loc[long["score_session"].isin(short["score_session"]), columns].reset_index(
            drop=True
        ),
        check_exact=True,
    )


def test_future_feature_timestamp_is_not_used_for_scoring() -> None:
    forecasts = _run(4, features=_features(4, future_at=2))

    row = forecasts.loc[forecasts["score_session"].eq("2025-01-06")].iloc[0]
    assert row["required_features_available"] is False or not bool(
        row["required_features_available"]
    )
    assert "missing_features" in row["reason_codes"]
    assert pd.isna(row["feature_max_availability_timestamp"])


def test_future_feature_outlier_cannot_leak_through_expanding_scaling() -> None:
    ordinary = _run(5)
    future_outlier = _run(6, features=_features(6, extreme_last=True))

    columns = ["score_session", "out_of_distribution_score", "p_edge_active"]
    assert_frame_equal(
        ordinary.loc[:, columns].reset_index(drop=True),
        future_outlier.loc[
            future_outlier["score_session"].isin(ordinary["score_session"]), columns
        ].reset_index(drop=True),
        check_exact=True,
    )


def test_every_frozen_prediction_has_traceable_run_metadata_and_causal_cutoffs() -> None:
    forecasts = _run(5)

    assert forecasts["run_id"].eq("test-run").all()
    assert forecasts["configuration_hash"].eq("abc123").all()
    assert forecasts["model_version"].eq("test-v2").all()
    assert (
        forecasts["run_metadata_json"]
        .map(lambda value: json.loads(value)["run_id"])
        .eq("test-run")
        .all()
    )
    settled = forecasts.dropna(subset=["training_latest_availability_timestamp"])
    assert (
        pd.to_datetime(settled["training_latest_availability_timestamp"], utc=True)
        < pd.to_datetime(settled["decision_timestamp"], utc=True)
    ).all()
    features = forecasts.dropna(subset=["feature_max_availability_timestamp"])
    assert (
        pd.to_datetime(features["feature_max_availability_timestamp"], utc=True)
        <= pd.to_datetime(features["decision_timestamp"], utc=True)
    ).all()


def test_frozen_admission_is_applied_without_changing_existing_exit_rule() -> None:
    forecasts = _run(5)
    opportunities = pd.DataFrame(
        {
            "opportunity_id": ["o1"],
            "score_session": ["2025-01-08"],
            "loop_id": [CELL[0]],
            "orientation": [CELL[1]],
            "horizon": [CELL[2]],
            "opportunity_decision_timestamp": [pd.Timestamp("2025-01-08T15:00:00Z")],
            "existing_exit_timestamp": [pd.Timestamp("2025-01-08T17:00:00Z")],
        }
    )

    decisions = apply_frozen_admission(opportunities, forecasts)

    assert decisions["existing_exit_timestamp"].iloc[0] == pd.Timestamp("2025-01-08T17:00:00Z")
    assert decisions["existing_position_action"].iloc[0] == "unchanged_existing_exit_rule"
