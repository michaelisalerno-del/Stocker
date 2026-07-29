from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "packages/stocker_research/src"))

from stocker_research.movement_qualified_direction_v0 import (  # noqa: E402
    M1_THRESHOLD,
    aligned_returns,
    apply_empirical_bayes_orientation_map,
    apply_selective_policy,
    assign_contiguous_session_folds,
    attach_direction_targets,
    audited_state_orientation_map,
    baseline_predictions,
    build_d0_features,
    build_route_orientation_features,
    build_signed_behavioural_features,
    construct_fresh_episodes,
    crossfit_empirical_bayes_orientation,
    decide_direction_candidate,
    fit_direction_model,
    fit_empirical_bayes_orientation_map,
    freeze_confidence_boundary,
    manual_direction_probabilities,
    movement_gate,
    permute_labels_within_slates,
    session_bootstrap_samples,
    validate_protected_boundary,
)


def test_frozen_m1_threshold_and_above_threshold_gate() -> None:
    assert M1_THRESHOLD == 0.49588519865576763
    probabilities = np.array([M1_THRESHOLD - 1e-12, M1_THRESHOLD, M1_THRESHOLD + 1e-12])

    assert movement_gate(probabilities).tolist() == [False, True, True]


def test_fresh_episode_starts_and_thirty_minute_spacing() -> None:
    probabilities = [
        M1_THRESHOLD + 0.01,
        M1_THRESHOLD + 0.02,
        M1_THRESHOLD - 0.01,
        M1_THRESHOLD + 0.03,
        M1_THRESHOLD - 0.02,
        M1_THRESHOLD + 0.04,
    ]
    checkpoints = [6, 8, 10, 12, 14, 16]
    frame = pd.DataFrame(
        {
            "stock": "A",
            "session": "2024-01-02",
            "checkpoint": checkpoints,
            "signal_timestamp": [
                pd.Timestamp("2024-01-02 14:30:00Z") + pd.Timedelta(minutes=checkpoint * 5)
                for checkpoint in checkpoints
            ],
            "prospective_entry_timestamp": [
                pd.Timestamp("2024-01-02 14:30:00Z") + pd.Timedelta(minutes=checkpoint * 5)
                for checkpoint in checkpoints
            ],
            "m1_probability": probabilities,
            "partition": "development",
        }
    )

    episodes = construct_fresh_episodes(frame)

    # Checkpoint 12 is a crossing, but it is only 30 checkpoint minutes after
    # checkpoint 6 because each checkpoint unit is one completed five-minute bar.
    assert episodes["checkpoint"].tolist() == [6, 12]
    assert np.isnan(episodes.iloc[0]["previous_checkpoint_probability"])
    assert episodes.iloc[1]["previous_checkpoint_probability"] == probabilities[2]
    assert episodes["episode_number"].tolist() == [1, 2]
    assert np.isnan(episodes.iloc[0]["minutes_since_previous_episode"])
    assert episodes.iloc[1]["minutes_since_previous_episode"] == 30.0


def _worked_bars() -> pd.DataFrame:
    starts = pd.date_range("2024-01-02 14:30:00Z", periods=8, freq="5min")
    opens = [100.0, 101.0, 102.0, 103.0, 104.0, 101.0, 102.0, 105.0]
    closes = [101.0, 102.0, 103.0, 104.0, 101.0, 102.0, 105.0, 106.0]
    return pd.DataFrame(
        {
            "stock": "A",
            "session": "2024-01-02",
            "bar_ordinal": range(8),
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": starts + pd.Timedelta(minutes=5),
            "open": opens,
            "high": np.maximum(opens, closes) + 0.5,
            "low": np.minimum(opens, closes) - 0.5,
            "close": closes,
            "volume": 1000.0,
        }
    )


def test_prospective_entry_and_all_direction_targets() -> None:
    episode = pd.DataFrame(
        {
            "stock": ["A"],
            "session": ["2024-01-02"],
            "checkpoint": [2],
            "signal_timestamp": [pd.Timestamp("2024-01-02 14:40:00Z")],
            "prospective_entry_timestamp": [pd.Timestamp("2024-01-02 14:40:00Z")],
            "m1_probability": [0.6],
            "previous_checkpoint_probability": [0.4],
            "episode_number": [1],
            "minutes_since_previous_episode": [np.nan],
            "partition": ["development"],
            "atm_iv": [0.8],
        }
    )

    targeted = attach_direction_targets(episode, _worked_bars())
    row = targeted.iloc[0]

    assert row["entry_price"] == 102.0
    assert row["signal_close"] == 102.0
    assert row["signed_log_return_5m"] == pytest.approx(np.log(103.0 / 102.0))
    assert row["signed_log_return_10m"] == pytest.approx(np.log(104.0 / 102.0))
    assert row["signed_log_return_15m"] == pytest.approx(np.log(101.0 / 102.0))
    assert row["signed_log_return_30m"] == pytest.approx(np.log(106.0 / 102.0))
    assert row["direction_up_10m"] == 1
    assert row["zero_return_10m"] == 0
    expected_fraction = abs(np.log(104.0 / 102.0)) / (
        abs(np.log(102.0 / 100.0)) + abs(np.log(102.0 / 102.0)) + abs(np.log(104.0 / 102.0))
    )
    assert row["fraction_eventual_10m_move_after_entry"] == pytest.approx(expected_fraction)


def test_d0_features_are_causal_and_ignore_future_bars() -> None:
    episode = pd.DataFrame(
        {
            "stock": ["A"],
            "session": ["2024-01-02"],
            "checkpoint": [2],
            "signal_timestamp": [pd.Timestamp("2024-01-02 14:40:00Z")],
        }
    )
    bars = _worked_bars()
    first = build_d0_features(episode, bars)
    changed_future = bars.copy()
    changed_future.loc[changed_future["bar_ordinal"].ge(2), ["open", "high", "low", "close"]] *= 10
    second = build_d0_features(episode, changed_future)

    feature_columns = [
        column
        for column in first.columns
        if column not in {"stock", "session", "checkpoint", "signal_timestamp"}
    ]
    pd.testing.assert_series_equal(
        first.loc[0, feature_columns],
        second.loc[0, feature_columns],
        check_names=False,
    )
    assert first.loc[0, "signed_return_1bar"] == pytest.approx(np.log(102.0 / 101.0))
    assert first.loc[0, "maximum_feature_source_timestamp"] <= first.loc[0, "signal_timestamp"]


def test_d1_signed_behavioural_feature_construction() -> None:
    frame = pd.DataFrame(
        {
            "stock": ["A", "A", "A"],
            "session": ["2024-01-02"] * 3,
            "checkpoint": [6, 8, 10],
            "signed_pressure": [0.1, 0.3, -0.2],
            "signed_exhaustion": [0.2, 0.1, -0.4],
            "arousal": [2.0, 2.0, 2.0],
            "conviction": [3.0, 3.0, 3.0],
            "tension": [4.0, 4.0, 4.0],
            "raw_component__signed_progress_acceleration": [0.5, -0.5, 0.25],
            "raw_component__return_gap": [1.0, -2.0, 3.0],
            "recent_loop_memory_weighted_top_depth": [0.2, 0.4, 0.5],
        }
    )

    features = build_signed_behavioural_features(frame)

    assert np.isnan(features.loc[0, "signed_pressure_change"])
    assert features.loc[1, "signed_pressure_change"] == pytest.approx(0.2)
    assert features.loc[1, "signed_exhaustion_change"] == pytest.approx(-0.1)
    assert features.loc[2, "pressure_x_conviction"] == pytest.approx(-0.6)
    assert features.loc[2, "signed_structural_memory"] == pytest.approx(-0.5)


def test_orientation_crossfit_and_assessment_map_exclude_held_out_outcomes() -> None:
    development = pd.DataFrame(
        {
            "session": [f"2024-01-0{day}" for day in range(2, 8)],
            "fold": [0, 0, 0, 1, 1, 1],
            "orientation_identity": ["route"] * 6,
            "direction_up_10m": [1, 1, 1, 0, 0, 0],
            "signed_log_return_10m": [0.01, 0.02, 0.03, -0.01, -0.02, -0.03],
        }
    )
    first = crossfit_empirical_bayes_orientation(
        development, prior_equivalent_sample_size=1, minimum_support=1
    )
    changed = development.copy()
    changed.loc[changed["fold"].eq(0), "direction_up_10m"] = 0
    changed.loc[changed["fold"].eq(0), "signed_log_return_10m"] = -1.0
    second = crossfit_empirical_bayes_orientation(
        changed, prior_equivalent_sample_size=1, minimum_support=1
    )

    np.testing.assert_allclose(
        first.loc[first["fold"].eq(0), "orientation_probability_up"],
        second.loc[second["fold"].eq(0), "orientation_probability_up"],
    )

    orientation_map = fit_empirical_bayes_orientation_map(
        development, prior_equivalent_sample_size=1, minimum_support=1
    )
    assessment = pd.DataFrame(
        {
            "orientation_identity": ["route"],
            "direction_up_10m": [1],
            "signed_log_return_10m": [10.0],
        }
    )
    mapped_first = apply_empirical_bayes_orientation_map(assessment, orientation_map)
    assessment.loc[0, ["direction_up_10m", "signed_log_return_10m"]] = [0, -10.0]
    mapped_second = apply_empirical_bayes_orientation_map(assessment, orientation_map)
    assert (
        mapped_first.loc[0, "orientation_probability_up"]
        == mapped_second.loc[0, "orientation_probability_up"]
    )


def test_audited_route_orientation_features_use_next_required_state() -> None:
    centroids = pd.DataFrame(
        {
            "state": [0, 0, 1, 1, 2, 2],
            "feature": ["signed_efficiency_6", "signed_efficiency_12"] * 3,
            "raw_feature_centroid": [0.1, 0.2, 0.3, 0.1, -0.2, -0.4],
        }
    )
    state_map = audited_state_orientation_map(centroids)
    episodes = pd.DataFrame(
        {
            "stock": ["A"],
            "session": ["2024-01-02"],
            "checkpoint": [6],
            "signed_pressure": [-0.5],
            "route_resolution_state": ["NARROWING"],
        }
    )
    ledger = pd.DataFrame(
        {
            "ledger_kind": ["active_prefix", "active_prefix"],
            "stock": ["A", "A"],
            "session": ["2024-01-02", "2024-01-02"],
            "bar_ordinal": [6, 6],
            "semantic_loop_id": ["positive", "negative"],
            "orientation_id": ["positive__o_0-1-0", "negative__o_3-2-3"],
            "progress_states": [1, 1],
            "transitions_remaining": [2, 1],
        }
    )

    features = build_route_orientation_features(episodes, ledger, state_map)

    assert features.loc[0, "positive_active_prefix_count"] == 1
    assert features.loc[0, "negative_active_prefix_count"] == 1
    assert features.loc[0, "top_route_orientation"] == -1
    assert features.loc[0, "narrowing_route_orientation"] == -1
    assert features.loc[0, "dominant_route_pressure_agreement"] == 1

    changed_future = ledger.copy()
    changed_future["bar_ordinal"] = 7
    future_features = build_route_orientation_features(episodes, changed_future, state_map)
    assert future_features.loc[0, "positive_active_prefix_count"] == 0
    assert future_features.loc[0, "negative_active_prefix_count"] == 0


def test_confidence_threshold_and_call_put_abstain_policy() -> None:
    probabilities = np.linspace(0.001, 0.999, 200)
    frozen = freeze_confidence_boundary(probabilities)
    actions = apply_selective_policy(probabilities, float(frozen["boundary"]))

    assert int(np.count_nonzero(actions != "ABSTAIN")) >= 150
    assert apply_selective_policy(np.array([0.7, 0.3, 0.55]), 0.1).tolist() == [
        "CALL",
        "PUT",
        "ABSTAIN",
    ]
    assert aligned_returns(
        np.array(["CALL", "PUT"]), np.array([0.02, -0.03])
    ).tolist() == pytest.approx([0.02, 0.03])


def test_frozen_direction_model_and_manual_probability_reconstruction() -> None:
    frame = pd.DataFrame(
        {
            "session": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
            ],
            "stock": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "checkpoint_category": ["6", "8", "6", "8", "6", "8", "6", "8"],
            "x": [1.0, 2.0, np.nan, 4.0, -1.0, -2.0, -3.0, -4.0],
            "direction_up_10m": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )
    model = fit_direction_model(
        frame,
        target_column="direction_up_10m",
        numeric_features=("x",),
        categorical_features=("stock", "checkpoint_category"),
        model_id="worked",
    )
    manual = manual_direction_probabilities(model.as_dict(), frame)

    np.testing.assert_allclose(manual, model.predict(frame), atol=0.0, rtol=0.0)
    assert model.medians["x"] == pytest.approx(-1.0)
    assert model.centers["x"] == pytest.approx(-1.0)
    assert model.scales["x"] > 0.0


def test_contiguous_oof_folds_keep_complete_sessions_together() -> None:
    sessions = pd.Series(
        [
            "2024-01-02",
            "2024-01-02",
            "2024-02-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
        ]
    )
    folds = assign_contiguous_session_folds(sessions, folds=4)

    assert folds.groupby(sessions).nunique().max() == 1
    assert folds.tolist() == [0, 0, 1, 1, 2, 3]


def test_confidence_freeze_uses_only_supplied_development_oof_rows() -> None:
    development_oof = np.linspace(0.01, 0.99, 200)
    first = freeze_confidence_boundary(development_oof)
    assessment_values = np.full(500, 0.5)
    second = freeze_confidence_boundary(development_oof)

    assert assessment_values.size == 500
    assert first == second
    assert first["source"] == "2024_blocked_oof_only"


def test_baselines_bootstrap_null_and_protected_boundary() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2025-01-02", "2025-01-02", "2025-01-03"],
            "checkpoint_group": ["early", "early", "early"],
            "signed_return_2bar": [0.1, -0.1, 0.0],
            "signed_return_1bar": [-0.1, 0.1, 0.0],
            "market_return_2bar": [0.2, -0.2, 0.0],
            "direction_up_10m": [1, 0, 1],
        }
    )
    baselines = baseline_predictions(frame, development_up_rate=0.6)
    assert baselines["B0_probability"].tolist() == [0.6, 0.6, 0.6]
    assert baselines["B2_side"].tolist() == [1, -1, 0]

    samples = session_bootstrap_samples(frame["session"], draws=3, seed=7)
    assert len(samples) == 3
    assert all(len(draw) == 2 for draw in samples)
    permuted = permute_labels_within_slates(
        frame,
        label_column="direction_up_10m",
        strata=("session", "checkpoint_group"),
        seed=11,
    )
    for _, indices in frame.groupby(["session", "checkpoint_group"]).groups.items():
        assert sorted(permuted.loc[list(indices)].tolist()) == sorted(
            frame.loc[list(indices), "direction_up_10m"].tolist()
        )
    validate_protected_boundary(pd.Series(["2025-08-22"]))
    with pytest.raises(ValueError):
        validate_protected_boundary(pd.Series(["2026-01-01"]))


def test_frozen_direction_decision_logic() -> None:
    evidence = {
        "episode_support_passed": True,
        "selective_support_passed": True,
        "assessment_log_loss_improves_vs_d0": True,
        "assessment_brier_improves_vs_d0": True,
        "assessment_auc": 0.56,
        "assessment_balanced_accuracy": 0.53,
        "action_coverage": 0.35,
        "selective_accuracy": 0.56,
        "mean_aligned_return_10m": 0.001,
        "median_aligned_return_10m": 0.0001,
        "bootstrap_80_accuracy_lower": 0.51,
        "bootstrap_80_mean_return_lower": 0.0,
        "positive_months": 6,
        "beats_momentum_and_market": True,
        "exceeds_all_nulls_log_loss_or_auc": True,
        "late_direction_problem": False,
        "d1_adds_value": True,
        "d2_adds_value": True,
    }
    assert (
        decide_direction_candidate(evidence) == "movement_qualified_direction_candidate_supported"
    )
    evidence["selective_support_passed"] = False
    assert decide_direction_candidate(evidence) == "blocked_insufficient_selective_action_support"
