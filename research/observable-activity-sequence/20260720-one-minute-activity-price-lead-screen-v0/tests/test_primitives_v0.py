from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.one_minute_activity_price_lead_v0 import (
    IncrementEvidence,
    activity_absorption_interactions,
    activity_acceleration,
    activity_continuation_interactions,
    activity_lead_price_response,
    activity_peak_lead,
    activity_persistence,
    activity_range_response,
    activity_slope,
    assert_allowed_feature_names,
    assert_unprotected_timestamps,
    bar_sign_weighted_activity_proxy,
    causal_ten_minute_window,
    classify_onset,
    cohort_relative_cumulative_returns,
    decide_activity_screen,
    development_onset_barriers,
    directional_efficiency,
    extract_outcome_window,
    historical_activity_normalisation,
    manual_logistic_probability,
    permute_activity_bundle_within_slates,
    progress_per_activity,
    prove_timestamp_convention,
    session_block_bootstrap_draws,
    slate_row_weights,
)


def _one_minute_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 14:30", periods=10, freq="1min", tz="UTC")
    opens = np.arange(100.0, 110.0)
    closes = opens + 0.5
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": opens + 1.0,
            "low": opens - 1.0,
            "close": closes,
            "volume": np.arange(1.0, 11.0) * 100.0,
        }
    )


def _five_minute_aggregation(one_minute: pd.DataFrame) -> pd.DataFrame:
    grouped = one_minute.assign(bucket=one_minute["timestamp"].dt.floor("5min")).groupby(
        "bucket", sort=True
    )
    return grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index(names="timestamp")


def test_timestamp_convention_is_proved_by_exact_five_minute_aggregation() -> None:
    starts = _one_minute_bars()
    five_minute = _five_minute_aggregation(starts)

    assert prove_timestamp_convention(starts, five_minute) == "bar_start"
    ends = starts.assign(timestamp=starts["timestamp"] + pd.Timedelta(minutes=1))
    assert prove_timestamp_convention(ends, five_minute) == "bar_end"


def test_causal_window_contains_only_ten_fully_completed_predecision_bars() -> None:
    timestamps = pd.date_range("2024-01-02 14:40", periods=21, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
    )
    decision = pd.Timestamp("2024-01-02 15:00:00Z")

    window = causal_ten_minute_window(bars, decision, convention="bar_start")

    assert window["relative_minute"].tolist() == list(range(-10, 0))
    assert window["bar_start_timestamp"].iloc[0] == pd.Timestamp("2024-01-02 14:50:00Z")
    assert window["bar_start_timestamp"].iloc[-1] == pd.Timestamp("2024-01-02 14:59:00Z")
    assert window["bar_complete_timestamp"].le(decision).all()
    assert window["bar_start_timestamp"].lt(decision).all()


def test_causal_window_rejects_an_unproved_timestamp_convention() -> None:
    with pytest.raises(ValueError, match="timestamp convention"):
        causal_ten_minute_window(
            _one_minute_bars(),
            pd.Timestamp("2024-01-02 14:40:00Z"),
            convention="unknown",  # type: ignore[arg-type]
        )


def test_historical_activity_normalisation_is_same_minute_and_development_only() -> None:
    development_sessions = pd.date_range("2024-01-02", periods=21, freq="B")
    frame = pd.DataFrame(
        {
            "symbol": "AAL",
            "session": [*development_sessions.strftime("%Y-%m-%d"), "2025-01-02", "2025-01-03"],
            "minute_of_session_ordinal": 29,
            "volume": [*range(1, 22), 44, 440],
        }
    )

    normalised = historical_activity_normalisation(frame)

    assert normalised.loc[19, "relative_activity"] != normalised.loc[19, "relative_activity"]
    assert normalised.loc[20, "historical_median_volume"] == pytest.approx(10.5)
    assert normalised.loc[20, "relative_activity"] == pytest.approx(2.0)
    assert normalised.loc[21, "historical_median_volume"] == pytest.approx(11.0)
    assert normalised.loc[22, "historical_median_volume"] == pytest.approx(11.0)
    assert normalised.loc[21, "log_relative_activity"] == pytest.approx(np.log1p(4.0))


def test_activity_acceleration_and_slope_use_the_fixed_latest_five_minutes() -> None:
    log_activity = np.array([0.0, 0.0, 0.0, 2.0, 2.0])

    assert activity_acceleration(log_activity) == pytest.approx(2.0)
    assert activity_slope(np.arange(5.0)) == pytest.approx(1.0)


def test_elevated_activity_persistence_counts_and_longest_run() -> None:
    result = activity_persistence(
        [2.0, 2.1, 0.5, 1.5, 0.8],
        same_clock_p90=[1.8, 1.8, 1.8, 1.8, 1.8],
    )

    assert result == {
        "above_one_count": 3,
        "above_same_clock_p90_count": 2,
        "longest_elevated_run": 2,
    }


def test_peak_activity_lead_timing_uses_fixed_indices_and_clip() -> None:
    relative_activity = [1.0, 1.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    returns = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, -0.9, 0.0, 0.0]

    result = activity_peak_lead(relative_activity, returns)

    assert result["activity_peak_index"] == -8
    assert result["price_peak_index"] == -3
    assert result["price_peak_index_minus_activity_peak_index"] == 5


def test_signed_activity_proxy_is_bar_sign_weighted_not_trade_flow() -> None:
    assert bar_sign_weighted_activity_proxy([0.01, -0.02, 0.0], [2.0, 3.0, 4.0]) == pytest.approx(
        -1.0
    )


def test_directional_efficiency_uses_signed_and_absolute_progress() -> None:
    result = directional_efficiency([1.0, -0.5, 1.5])

    assert result["signed_efficiency"] == pytest.approx(2.0 / 3.0)
    assert result["absolute_efficiency"] == pytest.approx(2.0 / 3.0)
    assert directional_efficiency([0.0, 0.0, 0.0]) == {
        "signed_efficiency": 0.0,
        "absolute_efficiency": 0.0,
    }


def test_continuation_interactions_are_fixed_activity_times_efficiency() -> None:
    assert activity_continuation_interactions(
        mean_relative_activity_3=2.0,
        signed_efficiency_3=0.5,
        mean_relative_activity_5=3.0,
        signed_efficiency_5=-0.25,
    ) == pytest.approx(
        {
            "activity_continuation_3": 1.0,
            "activity_continuation_5": -0.75,
        }
    )


def test_absorption_interactions_use_inefficiency_and_absolute_wick() -> None:
    assert activity_absorption_interactions(
        mean_relative_activity_3=2.5,
        absolute_efficiency_3=0.2,
        absolute_wick_imbalance_3=0.4,
    ) == pytest.approx(
        {
            "activity_absorption_3": 2.0,
            "activity_absorption_wick": 1.0,
        }
    )


def test_progress_per_activity_uses_safe_total_activity_denominator() -> None:
    assert progress_per_activity(30.0, [2.0, 3.0, 1.0]) == pytest.approx(
        {
            "signed_progress_per_activity_3": 5.0,
            "absolute_progress_per_activity_3": 5.0,
        }
    )


def test_fixed_activity_lead_and_range_response_interactions() -> None:
    assert activity_lead_price_response(
        early_activity=2.0,
        cumulative_return_last_2=-0.03,
        cumulative_return_minutes_minus_5_through_minus_3=0.01,
    ) == pytest.approx(0.04)
    assert activity_range_response(activity_acceleration_value=0.5, range_acceleration=1.4) == (
        pytest.approx(0.7)
    )


def test_onset_barrier_uses_only_2024_maximum_absolute_paths() -> None:
    rows: list[dict[str, object]] = []
    for checkpoint, maxima in ((6, [10.0, 20.0, 30.0, 40.0]), (12, [4.0, 8.0, 12.0, 16.0])):
        for decision_id, maximum in enumerate(maxima):
            rows.extend(
                [
                    {
                        "decision_id": f"{checkpoint}-{decision_id}",
                        "year": 2024,
                        "decision_ordinal": checkpoint,
                        "relative_minute": 2,
                        "cumulative_residual_return_bps": maximum / 2.0,
                    },
                    {
                        "decision_id": f"{checkpoint}-{decision_id}",
                        "year": 2024,
                        "decision_ordinal": checkpoint,
                        "relative_minute": 6,
                        "cumulative_residual_return_bps": -maximum,
                    },
                ]
            )
    rows.append(
        {
            "decision_id": "assessment-outlier",
            "year": 2025,
            "decision_ordinal": 6,
            "relative_minute": 6,
            "cumulative_residual_return_bps": 10_000.0,
        }
    )

    barriers = development_onset_barriers(pd.DataFrame(rows))

    assert barriers == pytest.approx({6: 32.5, 12: 13.0})


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ([10.0, 55.0, -80.0], "UP_ONSET"),
        ([-10.0, -55.0, 80.0], "DOWN_ONSET"),
        ([10.0, -20.0, 49.9], "NO_ONSET"),
    ],
)
def test_onset_label_uses_first_completed_close_crossing(path: list[float], expected: str) -> None:
    assert classify_onset(path, barrier_bps=50.0) == expected


def test_delayed_entry_and_fixed_fifteen_and_thirty_minute_terminals() -> None:
    timestamps = pd.date_range("2024-01-02 14:30", periods=100, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "minute_of_session_ordinal": np.arange(100),
            "bar_start_timestamp": timestamps,
            "open": 100.0 + np.arange(100),
            "close": 100.5 + np.arange(100),
        }
    )

    window = extract_outcome_window(bars, pd.Timestamp("2024-01-02 15:00:00Z"))

    assert window.entry_minute_ordinal == 31
    assert window.entry_open == pytest.approx(131.0)
    assert window.onset_minute_ordinals == (31, 32, 33, 34, 35)
    assert window.fifteen_minute_terminal_ordinal == 45
    assert window.fifteen_minute_terminal_close == pytest.approx(145.5)
    assert window.thirty_minute_terminal_ordinal == 60
    assert window.thirty_minute_terminal_close == pytest.approx(160.5)


def test_cohort_relative_return_uses_leave_one_stock_out_median() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": "2025-01-02|6",
            "symbol": ["A", "B", "C"],
            "relative_minute": 2,
            "cumulative_return_bps": [10.0, 20.0, 30.0],
        }
    )

    result = cohort_relative_cumulative_returns(frame)

    assert result["cohort_median_cumulative_return_bps"].tolist() == [25.0, 20.0, 15.0]
    assert result["cumulative_residual_return_bps"].tolist() == [-15.0, 0.0, 15.0]


def test_each_admitted_slate_receives_equal_total_weight() -> None:
    frame = pd.DataFrame({"parent_slate_id": ["s1", "s1", "s2"], "symbol": ["A", "B", "C"]})

    weights = slate_row_weights(frame)

    assert weights.tolist() == [0.5, 0.5, 1.0]
    assert weights.groupby(frame["parent_slate_id"]).sum().to_dict() == pytest.approx(
        {"s1": 1.0, "s2": 1.0}
    )


def test_manual_logistic_probability_reconstructs_standardised_linear_score() -> None:
    features = pd.DataFrame({"x1": [3.0], "x2": [5.0]})

    probability = manual_logistic_probability(
        features,
        feature_names=["x1", "x2"],
        means=[1.0, 1.0],
        scales=[2.0, 4.0],
        coefficients=[0.5, -0.25],
        intercept=0.1,
    )

    assert probability.tolist() == pytest.approx([1.0 / (1.0 + np.exp(-0.35))])


def test_session_block_bootstrap_is_fixed_seed_and_samples_whole_sessions() -> None:
    sessions = ["s1", "s1", "s2", "s3", "s3"]

    first = session_block_bootstrap_draws(sessions, draws=4, seed=20260720)
    second = session_block_bootstrap_draws(sessions, draws=4, seed=20260720)

    assert first == second
    assert len(first) == 4
    assert all(len(draw) == 3 for draw in first)
    assert all(set(draw).issubset({"s1", "s2", "s3"}) for draw in first)


def test_activity_bundle_null_preserves_pairs_and_nonactivity_fields_within_slate() -> None:
    frame = pd.DataFrame(
        {
            "parent_slate_id": ["s1", "s1", "s1", "s2"],
            "symbol": ["A", "B", "C", "D"],
            "price_feature": [1.0, 2.0, 3.0, 4.0],
            "activity_a": [10.0, 20.0, 30.0, 40.0],
            "activity_b": [100.0, 200.0, 300.0, 400.0],
        }
    )

    permuted = permute_activity_bundle_within_slates(
        frame,
        bundle_columns=["activity_a", "activity_b"],
        seed=20260721,
    )

    assert permuted[["symbol", "price_feature"]].equals(frame[["symbol", "price_feature"]])
    assert set(map(tuple, permuted.loc[:2, ["activity_a", "activity_b"]].to_numpy())) == {
        (10.0, 100.0),
        (20.0, 200.0),
        (30.0, 300.0),
    }
    assert permuted.loc[3, ["activity_a", "activity_b"]].tolist() == [40.0, 400.0]


def test_protected_date_rejection_occurs_before_feature_materialisation() -> None:
    assert_unprotected_timestamps([pd.Timestamp("2025-08-22 20:00:00Z")])
    with pytest.raises(ValueError, match="protected"):
        assert_unprotected_timestamps([pd.Timestamp("2025-08-23 00:00:00Z")])


def test_forbidden_structural_predictor_names_fail_closed() -> None:
    assert_allowed_feature_names(["relative_activity_minus_1", "cohort_relative_return_3"])
    with pytest.raises(ValueError, match="forbidden"):
        assert_allowed_feature_names(["future_volume"])


def test_increment_gate_requires_probability_stability_and_null_when_applicable() -> None:
    passing = IncrementEvidence(
        brier_improvement=0.001,
        log_loss_improvement=0.002,
        bootstrap_90_lower_brier=0.0,
        bootstrap_90_lower_log_loss=0.0,
        auc_not_reduced=True,
        positive_months=5,
        neither_checkpoint_materially_adverse=True,
        exceeds_null_90th_percentile=True,
        concentration_passes=True,
    )
    failing_null = IncrementEvidence(**{**passing.__dict__, "exceeds_null_90th_percentile": False})

    assert passing.passes(requires_null=True, requires_concentration=True) is True
    assert failing_null.passes(requires_null=True, requires_concentration=True) is False


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (
            {"raw_activity_onset": True, "raw_activity_direction": True},
            "one_minute_activity_leads_onset_and_direction",
        ),
        ({"raw_activity_onset": True}, "one_minute_activity_leads_onset_only"),
        ({"raw_activity_direction": True}, "one_minute_activity_adds_direction_only"),
        ({"interaction_onset": True}, "activity_price_response_interaction_only"),
        ({"price_onset": True}, "one_minute_price_sequence_only"),
        ({}, "no_one_minute_activity_increment"),
    ],
)
def test_decision_logic_uses_exact_category_precedence(
    flags: dict[str, bool], expected: str
) -> None:
    assert decide_activity_screen(**flags) == expected
