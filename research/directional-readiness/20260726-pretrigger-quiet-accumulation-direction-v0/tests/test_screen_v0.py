from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stocker_research.movement_qualified_direction_v0 import construct_fresh_episodes
from stocker_research.pretrigger_quiet_accumulation_v0 import (
    EPSILON,
    M1_THRESHOLD,
    PRIMARY_WINDOW_BARS,
    QUIET_SIGNED_COMPONENTS,
    RobustLocationScale,
    activity_without_displacement,
    aligned_return,
    apply_quiet_score_parameters,
    attach_pretrigger_direction_targets,
    bar_break_failure_asymmetry,
    bar_clv,
    bar_normalised_range,
    bar_relative_return,
    bar_signed_return,
    bar_wick_asymmetry,
    build_pretrigger_feature_rows,
    decide_pretrigger_candidate,
    fit_quiet_score_parameters,
    freeze_confidence_boundary,
    grouped_feature_permutation,
    label_null_within_slates,
    pressure_persistence,
    pressure_slope,
    remaining_fraction,
    score_sign_persistence,
    selective_actions,
    signed_absorption_divergence,
    temporal_placebo_bundle,
    validate_authorized_sessions,
)


def _bars() -> pd.DataFrame:
    start = pd.Timestamp("2024-06-03 13:30:00", tz="UTC")
    rows: list[dict[str, object]] = []
    previous_close = 100.0
    for ordinal in range(24):
        open_price = previous_close
        close = open_price * math.exp(0.0002 * (1 if ordinal % 3 else -1))
        high = max(open_price, close) + 0.05
        low = min(open_price, close) - 0.04
        rows.append(
            {
                "stock": "AAA",
                "session": "2024-06-03",
                "bar_ordinal": ordinal,
                "bar_start_timestamp": start + pd.Timedelta(minutes=5 * ordinal),
                "bar_complete_timestamp": start + pd.Timedelta(minutes=5 * (ordinal + 1)),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "vti__bar_log_return": 0.00005 * (1 if ordinal % 2 else -1),
                "historical_relative_activity": 1.0 + ordinal / 20.0,
                "signed_pressure": (ordinal - 4.0) / 10.0,
            }
        )
        previous_close = close
    return pd.DataFrame(rows)


def _episode(checkpoint: int = 7) -> pd.DataFrame:
    bars = _bars()
    trigger = bars.loc[bars["bar_ordinal"].eq(checkpoint - 1)].iloc[0]
    entry = bars.loc[bars["bar_ordinal"].eq(checkpoint)].iloc[0]
    return pd.DataFrame(
        {
            "stock": ["AAA"],
            "session": ["2024-06-03"],
            "checkpoint": [checkpoint],
            "signal_timestamp": [trigger["bar_complete_timestamp"]],
            "prospective_entry_timestamp": [entry["bar_start_timestamp"]],
            "m1_probability": [0.6],
            "partition": ["development"],
        }
    )


def _raw_score_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(12):
        direction = -1.0 if index < 6 else 1.0
        magnitude = 0.2 + 0.1 * (index % 6)
        row = {
            "partition": "development",
            "net_return_25": direction * magnitude / 100.0,
            "path_length_25": 0.01 + index / 1000.0,
            "range_sum_25": 0.02 + index / 1000.0,
            "pressure_sum_25": direction * magnitude,
            "pressure_persistence_25": direction * 0.6,
            "pressure_slope_25": direction * 0.1,
            "activity_without_displacement_25": direction * 0.7,
            "relative_resilience_25": direction * 0.002,
            "mean_clv_25": direction * 0.4,
            "mean_wick_asymmetry_25": direction * 0.3,
            "break_failure_asymmetry_25": direction * 0.2,
            "mean_vwap_distance_25": direction * 0.5,
            "vwap_side_balance_25": direction * 0.6,
            "vwap_reclaim_balance_25": direction * 0.25,
        }
        for position in range(PRIMARY_WINDOW_BARS):
            row[f"_pressure_sum_3bar_position_{position}"] = direction * (
                magnitude + position / 20.0
            )
            row[f"_net_return_3bar_position_{position}"] = direction * (
                magnitude / 100.0 + position / 10000.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_frozen_m1_threshold_and_fresh_episode_reconstruction() -> None:
    assert M1_THRESHOLD == 0.49588519865576763
    rows = pd.DataFrame(
        {
            "stock": ["AAA"] * 5,
            "session": ["2024-06-03"] * 5,
            "checkpoint": [6, 8, 12, 14, 20],
            "signal_timestamp": pd.to_datetime(
                [
                    "2024-06-03 14:00Z",
                    "2024-06-03 14:10Z",
                    "2024-06-03 14:30Z",
                    "2024-06-03 14:40Z",
                    "2024-06-03 15:10Z",
                ]
            ),
            "prospective_entry_timestamp": pd.to_datetime(
                [
                    "2024-06-03 14:00Z",
                    "2024-06-03 14:10Z",
                    "2024-06-03 14:30Z",
                    "2024-06-03 14:40Z",
                    "2024-06-03 15:10Z",
                ]
            ),
            "m1_probability": [0.50, 0.60, 0.40, 0.70, 0.80],
            "partition": ["development"] * 5,
        }
    )
    episodes = construct_fresh_episodes(rows)
    assert episodes["checkpoint"].tolist() == [6, 14]
    assert episodes.loc[1, "minutes_since_previous_episode"] == 40.0


def test_t_minus_one_marker_trigger_exclusion_and_five_bar_window() -> None:
    features = build_pretrigger_feature_rows(_episode(), _bars())
    bars = _bars()
    assert len(features) == 1
    assert features.loc[0, "marker_bar_ordinal"] == 5
    assert features.loc[0, "trigger_bar_ordinal"] == 6
    assert features.loc[0, "primary_window_bar_ordinals"] == "1,2,3,4,5"
    assert (
        features.loc[0, "pretrigger_marker_timestamp"]
        == bars.loc[bars["bar_ordinal"].eq(5), "bar_complete_timestamp"].iloc[0]
    )
    assert (
        features.loc[0, "maximum_direction_feature_timestamp"]
        < features.loc[0, "trigger_timestamp"]
    )
    expected = np.log(
        bars.loc[bars["bar_ordinal"].eq(5), "close"].iloc[0]
        / bars.loc[bars["bar_ordinal"].eq(0), "close"].iloc[0]
    )
    assert features.loc[0, "net_return_25"] == pytest.approx(expected)


def test_signed_and_relative_returns_and_normalised_range() -> None:
    result = bar_signed_return(101.0, 100.0)
    assert result == pytest.approx(math.log(1.01))
    assert bar_relative_return(result, 0.001) == pytest.approx(result - 0.001)
    assert bar_normalised_range(102.0, 99.0, 100.0) == pytest.approx(0.03)


def test_close_location_and_wick_asymmetry_are_mirrored() -> None:
    assert bar_clv(101.0, 99.0, 100.75) == pytest.approx(0.75, abs=1e-10)
    bullish = bar_wick_asymmetry(100.0, 100.5, 98.0, 100.25)
    bearish = bar_wick_asymmetry(100.0, 102.0, 99.5, 99.75)
    assert bullish > 0.0
    assert bearish < 0.0
    assert bullish == pytest.approx(-bearish)


def test_break_failure_asymmetry_reclaim_and_rejection() -> None:
    prior_highs = np.asarray([100.0, 100.1, 100.2, 100.1, 100.0, 100.2])
    prior_lows = np.asarray([99.0, 99.1, 99.2, 99.1, 99.0, 99.2])
    reclaim = bar_break_failure_asymmetry(
        high=100.0,
        low=98.5,
        close=99.8,
        prior_highs=prior_highs,
        prior_lows=prior_lows,
    )
    rejection = bar_break_failure_asymmetry(
        high=100.7,
        low=99.2,
        close=99.4,
        prior_highs=prior_highs,
        prior_lows=prior_lows,
    )
    assert reclaim > 0.0
    assert rejection < 0.0
    with pytest.raises(ValueError, match="six"):
        bar_break_failure_asymmetry(
            high=100.0,
            low=99.0,
            close=99.5,
            prior_highs=prior_highs[:5],
            prior_lows=prior_lows[:5],
        )


def test_pressure_persistence_and_ols_slope() -> None:
    pressure = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert pressure_persistence(pressure) == 0.0
    cumulative = np.cumsum(pressure)
    expected = np.polyfit(np.arange(5, dtype=float), cumulative, 1)[0]
    assert pressure_slope(pressure) == pytest.approx(expected)


def test_signed_absorption_divergence_is_one_mirrored_formula() -> None:
    bullish = signed_absorption_divergence(pressure_sum=2.0, pressure_z=2.5, price_z=0.5)
    bearish = signed_absorption_divergence(pressure_sum=-2.0, pressure_z=-2.5, price_z=-0.5)
    assert bullish == pytest.approx(2.0)
    assert bearish == pytest.approx(-2.0)


def test_activity_without_displacement_is_signed_and_bounded_by_activity() -> None:
    positive = activity_without_displacement(
        pressure_sum=1.0,
        activity=np.asarray([1.0, 2.0, -1.0, 1.0, 0.0]),
        net_return=0.001,
        path_length=0.01,
    )
    negative = activity_without_displacement(
        pressure_sum=-1.0,
        activity=np.asarray([1.0, 2.0, -1.0, 1.0, 0.0]),
        net_return=-0.001,
        path_length=0.01,
    )
    assert positive > 0.0
    assert positive == pytest.approx(-negative)
    assert positive <= np.mean([1.0, 2.0, 0.0, 1.0, 0.0])


def test_vwap_defence_and_score_persistence_are_causal() -> None:
    features = build_pretrigger_feature_rows(_episode(), _bars())
    bars = _bars()
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    weights = bars["historical_relative_activity"]
    vwap = (typical * weights).cumsum() / weights.cumsum()
    window = bars["bar_ordinal"].between(1, 5)
    above = bars.loc[window, "close"].to_numpy() > vwap.loc[window].to_numpy()
    below = bars.loc[window, "close"].to_numpy() < vwap.loc[window].to_numpy()
    assert features.loc[0, "vwap_side_balance_25"] == pytest.approx(
        (np.count_nonzero(above) - np.count_nonzero(below)) / 5.0
    )
    later = build_pretrigger_feature_rows(_episode(21), bars)
    assert math.isfinite(later.loc[0, "mean_vwap_distance_25"])
    assert score_sign_persistence(np.asarray([2.0, 1.0, 0.0, -1.0, 3.0])) == 0.4


def test_development_only_standardisation_and_equal_weight_composite() -> None:
    development = _raw_score_rows()
    parameters = fit_quiet_score_parameters(development)
    assert parameters.fit_partition == "development"
    assert set(parameters.component_parameters) == set(QUIET_SIGNED_COMPONENTS)
    assert all(
        isinstance(value, RobustLocationScale) for value in parameters.component_parameters.values()
    )
    transformed = apply_quiet_score_parameters(development, parameters)
    expected_core = transformed.loc[
        :, [f"{column}__clipped_z" for column in QUIET_SIGNED_COMPONENTS]
    ].mean(axis=1)
    assert np.allclose(transformed["signed_accumulation_core_25"], expected_core)
    assert np.allclose(
        transformed["quiet_absorption_score_25"],
        transformed["quietness_25"] * expected_core,
    )
    assert transformed.loc[0, "quiet_absorption_score_25"] < 0.0
    assert transformed.loc[11, "quiet_absorption_score_25"] > 0.0
    with pytest.raises(ValueError, match="explicit partition"):
        fit_quiet_score_parameters(development.drop(columns="partition"))
    contaminated = development.copy()
    contaminated.loc[0, "partition"] = "assessment"
    with pytest.raises(ValueError, match="development rows only"):
        fit_quiet_score_parameters(contaminated)


def test_target_entry_horizons_and_pre_entry_remaining_fraction() -> None:
    features = build_pretrigger_feature_rows(_episode(), _bars())
    targets = attach_pretrigger_direction_targets(features, _bars())
    bars = _bars().set_index("bar_ordinal")
    entry = float(bars.loc[7, "open"])
    marker = float(bars.loc[5, "close"])
    expected_10m = math.log(float(bars.loc[8, "close"]) / entry)
    assert targets.loc[0, "entry_price"] == entry
    assert targets.loc[0, "pre_entry_signed_return"] == pytest.approx(math.log(entry / marker))
    assert targets.loc[0, "signed_log_return_10m"] == pytest.approx(expected_10m)
    assert targets.loc[0, "remaining_fraction_10m"] == pytest.approx(
        remaining_fraction(math.log(entry / marker), expected_10m)
    )


def test_oof_threshold_freezing_and_call_put_abstain() -> None:
    probabilities = np.asarray([0.1, 0.2, 0.45, 0.55, 0.8, 0.9])
    boundary = freeze_confidence_boundary(
        probabilities,
        target_coverage=0.5,
        minimum_actions=3,
    )
    actions = selective_actions(probabilities, boundary)
    assert int(np.sum(actions != "ABSTAIN")) >= 3
    assert actions[0] == "PUT"
    assert actions[-1] == "CALL"
    assert actions[2] == "ABSTAIN"
    assert actions[3] == "ABSTAIN"


def test_aligned_return_and_remaining_movement_fraction() -> None:
    assert aligned_return("CALL", 0.01) == 0.01
    assert aligned_return("PUT", -0.01) == 0.01
    assert remaining_fraction(0.01, 0.02) == pytest.approx(0.02 / (0.01 + 0.02 + EPSILON))


def test_grouped_permutation_preserves_group_multiset() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 3 + ["2024-01-03"] * 3,
            "checkpoint": [6] * 6,
            "p1": [1, 2, 3, 4, 5, 6],
            "p2": [11, 12, 13, 14, 15, 16],
            "control": list("abcdef"),
        }
    )
    permuted = grouped_feature_permutation(
        frame,
        feature_columns=["p1", "p2"],
        group_columns=["session", "checkpoint"],
        seed=19,
    )
    assert permuted["control"].tolist() == frame["control"].tolist()
    for _, indices in frame.groupby(["session", "checkpoint"]).groups.items():
        assert sorted(map(tuple, permuted.loc[indices, ["p1", "p2"]].to_numpy())) == sorted(
            map(tuple, frame.loc[indices, ["p1", "p2"]].to_numpy())
        )


def test_label_null_only_permutes_labels_among_stocks_in_slates() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 4,
            "checkpoint": [6, 6, 8, 8],
            "stock": ["AAA", "BBB", "AAA", "BBB"],
            "direction_up_10m": [0, 1, 1, 0],
            "feature": [10, 20, 30, 40],
        }
    )
    null = label_null_within_slates(
        frame,
        target_column="direction_up_10m",
        seed=7,
    )
    assert null["feature"].tolist() == frame["feature"].tolist()
    for _, indices in frame.groupby(["session", "checkpoint"]).groups.items():
        assert sorted(null.loc[indices, "direction_up_10m"]) == sorted(
            frame.loc[indices, "direction_up_10m"]
        )


def test_temporal_placebo_moves_complete_bundle_to_next_episode() -> None:
    frame = pd.DataFrame(
        {
            "stock": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "session": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-02",
                "2024-01-03",
            ],
            "pretrigger_marker_timestamp": pd.to_datetime(
                [
                    "2024-01-02 14:00Z",
                    "2024-01-03 14:00Z",
                    "2024-01-04 14:00Z",
                    "2024-01-02 14:00Z",
                    "2024-01-03 14:00Z",
                ]
            ),
            "p1": [1.0, 2.0, 3.0, 10.0, 20.0],
            "p2": [4.0, 5.0, 6.0, 40.0, 50.0],
            "control": [100, 200, 300, 400, 500],
        }
    )
    placebo = temporal_placebo_bundle(frame, ["p1", "p2"])
    assert math.isnan(placebo.loc[0, "p1"])
    assert placebo.loc[1, "p1"] == 1.0
    assert placebo.loc[2, "p2"] == 5.0
    assert placebo.loc[4, "p1"] == 10.0
    assert placebo["control"].tolist() == frame["control"].tolist()


def test_protected_boundary_enforcement() -> None:
    validate_authorized_sessions(pd.Series(["2024-01-01", "2025-08-22"]))
    with pytest.raises(ValueError, match="forbidden"):
        validate_authorized_sessions(pd.Series(["2025-09-01"]))
    with pytest.raises(ValueError, match="protected"):
        validate_authorized_sessions(pd.Series(["2026-01-02"]))


def test_frozen_decision_logic_and_late_direction_category() -> None:
    evidence: dict[str, object] = {
        "development_support_passed": True,
        "assessment_support_passed": True,
        "selective_support_passed": True,
        "concentration_gates_passed": True,
        "q1_log_loss_improves": True,
        "q1_brier_improves": True,
        "q1_auc": 0.58,
        "q1_balanced_accuracy": 0.54,
        "action_coverage": 0.35,
        "selective_accuracy": 0.59,
        "beats_required_baselines": True,
        "mean_aligned_return_10m": 0.001,
        "median_aligned_return_10m": 0.0005,
        "bootstrap_80_accuracy_lower": 0.53,
        "bootstrap_80_mean_return_lower": 0.0001,
        "positive_month_groups": 7,
        "null_gate_passed": True,
        "temporal_placebo_gate_passed": True,
        "score_monotonic_direction_correct": True,
        "late_direction_problem": False,
        "persistent_pressure_supported": True,
        "absorption_response_supported": True,
    }
    assert (
        decide_pretrigger_candidate(evidence)
        == "pretrigger_quiet_accumulation_direction_candidate_supported"
    )
    evidence["late_direction_problem"] = True
    assert decide_pretrigger_candidate(evidence) == "pretrigger_direction_present_but_too_late"
