from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.movement_qualified_direction_v0 import (
    aligned_returns,
    permute_labels_within_slates,
)
from stocker_research.stock_local_directional_archetypes_v0 import (
    ABSORPTION_FEATURES,
    CONTINUATION_FEATURES,
    activity_price_impact,
    add_relative_strength_features,
    apply_selective_policy,
    apply_stock_local_normalisation,
    archetype_decision,
    assign_stock_local_session_weights,
    beta_adjusted_residual,
    build_movement_dependency_audit,
    build_raw_archetype_features,
    checkpoint_group,
    construct_fresh_episodes,
    directional_efficiency,
    fit_stock_local_normalisation,
    fit_stock_market_betas,
    mirrored_boundary_failure,
    mirrored_wick_rejection,
    reject_protected_sessions,
    remaining_fraction,
    residual_persistence,
    shift_features_to_next_episode,
    transitive_descendants,
    weighted_quantile,
)


def test_stock_local_weights_are_invariant_to_other_stocks_and_future_targets() -> None:
    base = pd.DataFrame(
        {
            "stock": ["AAA", "AAA"],
            "session": ["2024-01-02", "2024-01-02"],
            "checkpoint": [6, 8],
            "future_target_available": [True, False],
        }
    )
    expanded = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "stock": ["BBB"],
                    "session": ["2024-01-02"],
                    "checkpoint": [6],
                    "future_target_available": [True],
                }
            ),
        ],
        ignore_index=True,
    )

    base_weighted = assign_stock_local_session_weights(base)
    expanded_weighted = assign_stock_local_session_weights(expanded)

    assert base_weighted["row_weight"].tolist() == [0.5, 0.5]
    assert expanded_weighted.loc[expanded_weighted["stock"].eq("AAA"), "row_weight"].tolist() == [
        0.5,
        0.5,
    ]
    assert expanded_weighted.groupby(["stock", "session"])["row_weight"].sum().eq(1.0).all()


def test_contaminated_dependency_detection_and_transitive_feature_removal() -> None:
    graph = {
        "future_filtered_peer_slate": ("peer_median",),
        "peer_median": ("signed_progress", "absolute_progress"),
        "signed_progress": ("signed_pressure",),
        "absolute_progress": ("tension",),
        "signed_pressure": ("M1_probability",),
        "tension": ("M1_probability",),
        "M1_probability": ("threshold_membership",),
        "threshold_membership": ("fresh_episode_identity",),
    }

    descendants = transitive_descendants(graph, ("future_filtered_peer_slate",))
    audit = build_movement_dependency_audit(
        graph=graph,
        contaminated_roots=("future_filtered_peer_slate",),
        group_i_features=("arousal", "tension", "signed_pressure", "local_range"),
        peer_normalised_features=(),
    )

    assert descendants == (
        "M1_probability",
        "absolute_progress",
        "fresh_episode_identity",
        "peer_median",
        "signed_pressure",
        "signed_progress",
        "tension",
        "threshold_membership",
    )
    assert audit["archived_m1_numerically_affected"] is True
    assert audit["contaminated_group_i_features"] == ["signed_pressure", "tension"]
    assert audit["causal_group_i_features"] == ["arousal", "local_range"]


def test_weighted_causal_movement_threshold_uses_midpoint_cdf() -> None:
    probabilities = np.array([0.10, 0.20, 0.30, 0.90])
    weights = np.ones(4)

    assert weighted_quantile(probabilities, weights, 0.50) == pytest.approx(0.25)
    assert weighted_quantile(probabilities, weights, 0.95) == pytest.approx(0.90)


def test_fresh_episode_construction_uses_immediate_eligible_checkpoint_and_spacing() -> None:
    rows = pd.DataFrame(
        {
            "stock": ["AAA"] * 7 + ["BBB"] * 2,
            "session": ["2025-01-02"] * 9,
            "checkpoint": [6, 8, 10, 12, 14, 16, 18, 6, 8],
            "signal_timestamp": pd.to_datetime(
                [
                    "2025-01-02 15:00Z",
                    "2025-01-02 15:10Z",
                    "2025-01-02 15:20Z",
                    "2025-01-02 15:30Z",
                    "2025-01-02 15:40Z",
                    "2025-01-02 15:50Z",
                    "2025-01-02 16:00Z",
                    "2025-01-02 15:00Z",
                    "2025-01-02 15:10Z",
                ]
            ),
            "prospective_entry_timestamp": pd.to_datetime(
                [
                    "2025-01-02 15:00Z",
                    "2025-01-02 15:10Z",
                    "2025-01-02 15:20Z",
                    "2025-01-02 15:30Z",
                    "2025-01-02 15:40Z",
                    "2025-01-02 15:50Z",
                    "2025-01-02 16:00Z",
                    "2025-01-02 15:00Z",
                    "2025-01-02 15:10Z",
                ]
            ),
            "movement_probability": [0.60, 0.70, 0.40, 0.65, 0.30, 0.66, 0.70, 0.55, 0.60],
            "partition": ["assessment"] * 9,
        }
    )

    episodes = construct_fresh_episodes(rows, threshold=0.50)

    assert episodes[["stock", "checkpoint"]].to_records(index=False).tolist() == [
        ("AAA", 6),
        ("AAA", 12),
        ("BBB", 6),
    ]
    assert episodes.loc[episodes["stock"].eq("AAA"), "episode_number"].tolist() == [1, 2]
    assert episodes["minutes_since_previous_episode"].dropna().min() >= 30.0


def test_t_minus_one_marker_and_trigger_exclusion_are_enforced_by_episode_timestamps() -> None:
    rows = pd.DataFrame(
        {
            "stock": ["AAA"],
            "session": ["2025-01-02"],
            "checkpoint": [6],
            "signal_timestamp": pd.to_datetime(["2025-01-02 15:00Z"]),
            "prospective_entry_timestamp": pd.to_datetime(["2025-01-02 15:00Z"]),
            "movement_probability": [0.8],
            "partition": ["assessment"],
        }
    )
    episodes = construct_fresh_episodes(rows, threshold=0.5)

    assert episodes.loc[0, "trigger_bar_ordinal"] == 5
    assert episodes.loc[0, "marker_bar_ordinal"] == 4
    assert episodes.loc[0, "trigger_bar_excluded_from_direction_features"]


def test_protected_boundary_enforcement_rejects_2026_without_materialising_it() -> None:
    reject_protected_sessions(pd.Series(["2024-01-02", "2025-12-31"]))

    with pytest.raises(ValueError, match="protected"):
        reject_protected_sessions(pd.Series(["2025-12-31", "2026-01-02"]))


def test_stock_local_median_iqr_and_time_of_day_normalisation() -> None:
    frame = pd.DataFrame(
        {
            "stock": ["AAA"] * 40,
            "session": [f"2024-01-{day:02d}" for day in range(1, 21)] * 2,
            "checkpoint": [6] * 20 + [8] * 20,
            "feature": list(range(20)) + list(range(100, 120)),
        }
    )
    parameters = fit_stock_local_normalisation(
        frame,
        feature_columns=("feature",),
        minimum_support=20,
    )
    transformed, fallbacks = apply_stock_local_normalisation(
        frame.iloc[[9, 29]].copy(),
        parameters,
        feature_columns=("feature",),
    )

    assert transformed["feature"].tolist() == pytest.approx([-0.5 / 9.5, -0.5 / 9.5])
    assert set(fallbacks["fallback_level"]) == {"stock_checkpoint"}
    exact = parameters.loc[parameters["stock"].eq("AAA") & parameters["checkpoint"].eq(6)].iloc[0]
    assert exact["median"] == pytest.approx(9.5)
    assert exact["iqr"] == pytest.approx(9.5)


def test_stock_local_normalisation_fallback_order() -> None:
    rows: list[dict[str, object]] = []
    for checkpoint, count, base in ((6, 5, 0.0), (16, 20, 10.0), (26, 5, 20.0)):
        for index in range(count):
            rows.append(
                {
                    "stock": "AAA",
                    "session": f"2024-02-{index + 1:02d}",
                    "checkpoint": checkpoint,
                    "feature": base + index,
                }
            )
    frame = pd.DataFrame(rows)
    parameters = fit_stock_local_normalisation(
        frame,
        feature_columns=("feature",),
        minimum_support=20,
    )
    target = pd.DataFrame(
        {
            "stock": ["AAA", "BBB"],
            "session": ["2025-01-02", "2025-01-02"],
            "checkpoint": [6, 6],
            "feature": [11.0, 11.0],
        }
    )
    _, fallbacks = apply_stock_local_normalisation(
        target,
        parameters,
        feature_columns=("feature",),
    )

    assert checkpoint_group(6) == "early"
    assert checkpoint_group(16) == "middle"
    assert checkpoint_group(26) == "late"
    assert fallbacks["fallback_level"].tolist() == [
        "stock_adjacent_checkpoint_group",
        "development_pooled",
    ]


def test_beta_fitting_uses_development_only_and_can_exclude_oof_sessions() -> None:
    frame = pd.DataFrame(
        {
            "stock": ["AAA"] * 8,
            "session": ["2024-01-02"] * 4 + ["2024-01-03"] * 4,
            "checkpoint_group": ["early"] * 8,
            "stock_return": [0.01, 0.02, 0.03, 0.04, 1.0, 1.0, 1.0, 1.0],
            "market_return": [0.005, 0.01, 0.015, 0.02, 0.005, 0.01, 0.015, 0.02],
        }
    )
    full = fit_stock_market_betas(frame, minimum_support=4)
    oof = fit_stock_market_betas(
        frame,
        minimum_support=4,
        excluded_sessions=("2024-01-03",),
    )

    assert full.loc[0, "development_end"] == "2024-01-03"
    assert oof.loc[0, "development_end"] == "2024-01-02"
    assert oof.loc[0, "beta"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="2024"):
        fit_stock_market_betas(
            frame.assign(session="2025-01-02"),
            minimum_support=4,
        )


def test_continuation_efficiency_and_mirrored_reversal_primitives() -> None:
    assert directional_efficiency([0.01, 0.02, -0.01, 0.03]) == pytest.approx(0.05 / 0.07)
    assert mirrored_wick_rejection(
        attempt_sign=-1,
        open_price=10.0,
        high=10.2,
        low=9.0,
        close=10.1,
    ) == pytest.approx(0.75)
    assert mirrored_wick_rejection(
        attempt_sign=1,
        open_price=10.0,
        high=11.0,
        low=9.8,
        close=9.9,
    ) == pytest.approx(-0.75)
    assert (
        mirrored_boundary_failure(
            attempt_sign=-1,
            attempted_extreme=9.0,
            boundary=9.5,
            response_close=9.8,
        )
        > 0.0
    )
    assert (
        mirrored_boundary_failure(
            attempt_sign=1,
            attempted_extreme=11.0,
            boundary=10.5,
            response_close=10.2,
        )
        < 0.0
    )


def test_relative_residual_policy_aligned_return_and_remaining_fraction() -> None:
    assert beta_adjusted_residual(
        stock_return=0.012,
        market_return=0.005,
        alpha=0.001,
        beta=1.4,
        bars=1,
    ) == pytest.approx(0.004)
    assert apply_selective_policy([0.8, 0.2, 0.55], 0.1).tolist() == [
        "CALL",
        "PUT",
        "ABSTAIN",
    ]
    assert remaining_fraction(0.01, 0.02) == pytest.approx(2.0 / 3.0)


def test_temporal_placebo_shifts_within_stock_only() -> None:
    frame = pd.DataFrame(
        {
            "stock": ["AAA", "AAA", "BBB"],
            "session": ["2024-01-02", "2024-01-03", "2024-01-02"],
            "checkpoint": [6, 8, 6],
            "feature": [1.0, 2.0, 7.0],
        }
    )
    shifted = shift_features_to_next_episode(frame, ("feature",))

    assert np.isnan(shifted.loc[0, "feature"])
    assert shifted.loc[1, "feature"] == 1.0
    assert np.isnan(shifted.loc[2, "feature"])


def test_individual_archetype_decision_gate_is_not_relaxed() -> None:
    passing = {
        "log_loss_improves": True,
        "brier_improves": True,
        "auc": 0.56,
        "balanced_accuracy": 0.53,
        "action_coverage": 0.35,
        "selective_accuracy": 0.58,
        "beats_all_selective_baselines": True,
        "mean_aligned_return": 0.001,
        "median_aligned_return": 0.0001,
        "bootstrap_80_accuracy_lower": 0.51,
        "bootstrap_80_mean_return_lower": 0.0,
        "positive_months": 6,
        "null_predictive_wins": 9,
        "null_return_wins": 9,
        "beats_temporal_placebo": True,
        "selective_support_passed": True,
        "concentration_passed": True,
        "late_direction_problem": False,
    }
    assert archetype_decision(passing) == "supported"
    assert archetype_decision({**passing, "selective_accuracy": 0.5699}) == "not_supported"


def _bars(
    *,
    stock_returns: list[float],
    closes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    activity: float = 2.0,
) -> pd.DataFrame:
    count = len(stock_returns)
    close_values = closes or [10.0 + 0.01 * index for index in range(count)]
    high_values = highs or [value + 0.05 for value in close_values]
    low_values = lows or [value - 0.05 for value in close_values]
    starts = pd.date_range("2024-01-02 14:30Z", periods=count, freq="5min")
    return pd.DataFrame(
        {
            "stock": ["AAA"] * count,
            "session": ["2024-01-02"] * count,
            "bar_ordinal": range(count),
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": starts + pd.Timedelta(minutes=5),
            "open": close_values,
            "high": high_values,
            "low": low_values,
            "close": close_values,
            "volume": [100.0] * count,
            "historical_relative_activity": [activity] * count,
            "vti__bar_log_return": [0.0] * count,
            "bar_log_return": stock_returns,
        }
    )


def test_continuation_momentum_uses_completed_marker_bar() -> None:
    bars = _bars(stock_returns=[0.001] * 13)
    checkpoint = pd.DataFrame({"stock": ["AAA"], "session": ["2024-01-02"], "checkpoint": [12]})
    features = build_raw_archetype_features(checkpoint, bars)

    assert features.loc[0, "c_z_return_5m"] == pytest.approx(0.001)
    assert features.loc[0, "c_z_return_10m"] == pytest.approx(0.002)
    assert features.loc[0, "c_z_return_20m"] == pytest.approx(0.004)
    assert features.loc[0, "c_z_return_30m"] == pytest.approx(0.006)


def test_trigger_bar_exclusion_is_numerically_invariant() -> None:
    bars = _bars(stock_returns=[0.001] * 13)
    mutated = bars.copy()
    mutated.loc[mutated["bar_ordinal"].eq(11), "bar_log_return"] = 9.0
    mutated.loc[mutated["bar_ordinal"].eq(11), "close"] = 99.0
    checkpoint = pd.DataFrame({"stock": ["AAA"], "session": ["2024-01-02"], "checkpoint": [12]})
    first = build_raw_archetype_features(checkpoint, bars)
    second = build_raw_archetype_features(checkpoint, mutated)

    assert first[list(CONTINUATION_FEATURES)].to_numpy() == pytest.approx(
        second[list(CONTINUATION_FEATURES)].to_numpy()
    )
    assert first[list(ABSORPTION_FEATURES)].to_numpy() == pytest.approx(
        second[list(ABSORPTION_FEATURES)].to_numpy()
    )


def test_boundary_breakout_and_acceptance_use_prior_completed_six_bars() -> None:
    closes = [10.0] * 7 + [10.5, 10.45, 10.4, 10.35, 10.3, 10.3]
    highs = [10.1] * 7 + [10.6, 10.55, 10.5, 10.45, 10.4, 10.4]
    lows = [9.9] * 13
    bars = _bars(
        stock_returns=[0.0] * 13,
        closes=closes,
        highs=highs,
        lows=lows,
    )
    checkpoint = pd.DataFrame({"stock": ["AAA"], "session": ["2024-01-02"], "checkpoint": [12]})
    features = build_raw_archetype_features(checkpoint, bars)

    assert features.loc[0, "c_break_above_prior_six_high"] == 1.0
    assert features.loc[0, "c_break_below_prior_six_low"] == 0.0
    assert features.loc[0, "c_signed_boundary_acceptance_count"] == 4.0
    assert features.loc[0, "c_boundary_rejection"] == 0.0


def test_failed_attempt_construction_uses_t_minus_five_through_t_minus_three() -> None:
    returns = [-0.01, -0.01, -0.01, 0.005, 0.005, 0.0, 0.0]
    bars = _bars(stock_returns=returns, activity=2.0)
    checkpoint = pd.DataFrame({"stock": ["AAA"], "session": ["2024-01-02"], "checkpoint": [6]})
    features = build_raw_archetype_features(checkpoint, bars)

    assert features.loc[0, "a_attempt_return_abs"] == pytest.approx(0.03)
    assert features.loc[0, "a_attempt_path_length"] == pytest.approx(0.03)
    assert features.loc[0, "a_response_followthrough"] == pytest.approx(0.01)


def test_activity_impact_uses_historical_relative_activity_proxy() -> None:
    assert activity_price_impact(0.03, 2.0) == pytest.approx(0.015)


def test_relative_residual_persistence_preserves_direction() -> None:
    assert residual_persistence([0.01, 0.02, -0.01, 0.0]) == pytest.approx(0.25)


def test_relative_strength_features_use_frozen_stock_beta() -> None:
    raw = pd.DataFrame(
        {
            "stock": ["AAA"],
            "session": ["2025-01-02"],
            "checkpoint": [12],
            "checkpoint_group": ["early"],
            "_stock_return_lag_0": [0.01],
            "_stock_return_lag_1": [0.01],
            "_stock_return_lag_2": [0.01],
            "_stock_return_lag_3": [0.01],
            "_market_return_lag_0": [0.005],
            "_market_return_lag_1": [0.005],
            "_market_return_lag_2": [0.005],
            "_market_return_lag_3": [0.005],
        }
    )
    beta = pd.DataFrame(
        {
            "stock": ["AAA"],
            "checkpoint_group": ["early"],
            "alpha": [0.0],
            "beta": [1.0],
            "residual_scale": [0.005],
            "residual_range_low": [-0.005],
            "residual_range_high": [0.005],
            "stock_abs_return_median": [0.01],
        }
    )
    result = add_relative_strength_features(raw, beta)

    assert result.loc[0, "r_residual_return_5m"] == pytest.approx(0.005)
    assert result.loc[0, "r_residual_return_10m"] == pytest.approx(0.01)
    assert result.loc[0, "r_residual_persistence"] == 1.0


def test_call_decision_uses_upper_symmetric_boundary() -> None:
    assert apply_selective_policy([0.61], 0.10).tolist() == ["CALL"]


def test_put_decision_uses_lower_symmetric_boundary() -> None:
    assert apply_selective_policy([0.39], 0.10).tolist() == ["PUT"]


def test_abstain_decision_uses_same_symmetric_boundary() -> None:
    assert apply_selective_policy([0.50], 0.10).tolist() == ["ABSTAIN"]


def test_aligned_return_is_underlying_return_times_predicted_side() -> None:
    assert aligned_returns(["CALL", "PUT"], [0.01, -0.02]).tolist() == pytest.approx([0.01, 0.02])


def test_remaining_fraction_uses_pre_and_post_entry_absolute_displacement() -> None:
    assert remaining_fraction(-0.01, 0.03) == pytest.approx(0.75)


def test_label_null_preserves_each_session_checkpoint_slate() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 4,
            "checkpoint_group": ["early", "early", "middle", "middle"],
            "stock": ["AAA", "BBB", "AAA", "BBB"],
            "label": [0, 1, 1, 0],
        }
    )
    permuted = permute_labels_within_slates(
        frame,
        label_column="label",
        strata=("session", "checkpoint_group"),
        seed=2026072601,
    )

    for _, indices in frame.groupby(["session", "checkpoint_group"]).groups.items():
        assert sorted(permuted.loc[list(indices)].tolist()) == sorted(
            frame.loc[list(indices), "label"].tolist()
        )
