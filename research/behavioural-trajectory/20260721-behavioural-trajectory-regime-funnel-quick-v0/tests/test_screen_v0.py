from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stocker_research.behavioural_state_dimensions_v0 import opening_raw_components
from stocker_research.behavioural_trajectory_regime_funnel_v0 import (
    BlockedScreen,
    anchor_formula_availability,
    build_trajectory_interactions,
    causal_anchor_prefix,
    decide_trajectory_screen,
    manual_multinomial_probabilities,
    multiclass_brier,
    permute_trajectory_bundle_within_slates,
    prediction_entropy,
    reject_protected_dates,
    session_block_bootstrap_draws,
    trajectory_anchors,
    trajectory_feature_values,
)


def test_ordinal_6_uses_preregistered_completed_bar_anchors() -> None:
    assert trajectory_anchors(6) == (2, 4, 6)


def test_ordinal_12_uses_preregistered_completed_bar_anchors() -> None:
    assert trajectory_anchors(12) == (6, 9, 12)


def test_missing_anchor_handling_preserves_frozen_even_window_contract() -> None:
    assert anchor_formula_availability(9) == (
        False,
        "frozen_opening_raw_components_requires_even_completed_bar_count",
    )


def test_ordinal_12_middle_anchor_is_rejected_by_the_frozen_formula() -> None:
    bars = pd.DataFrame(
        {
            "open": [100.0] * 9,
            "high": [101.0] * 9,
            "low": [99.0] * 9,
            "close": [100.5] * 9,
            "historical_relative_activity": [1.0] * 9,
            "return_bps": [1.0] * 9,
            "true_range_bps": [2.0] * 9,
            "close_location": [0.75] * 9,
            "upper_wick_fraction": [0.25] * 9,
            "lower_wick_fraction": [0.25] * 9,
        }
    )

    with pytest.raises(ValueError, match="even number of completed bars"):
        opening_raw_components(
            bars,
            trailing_opening_range_median_bps=10.0,
            signed_progress_bps=0.0,
            signed_progress_acceleration_bps=0.0,
            return_gap_bps=0.0,
        )


def test_earlier_anchor_never_contains_a_future_bar() -> None:
    starts = pd.date_range("2025-01-02 14:30:00+00:00", periods=12, freq="5min")
    bars = pd.DataFrame(
        {
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": starts + pd.Timedelta(minutes=5),
            "value": range(12),
        }
    )

    prefix = causal_anchor_prefix(bars, 4)

    assert prefix["value"].tolist() == [0, 1, 2, 3]
    assert prefix["bar_complete_timestamp"].max() == pd.Timestamp("2025-01-02 14:50:00+00:00")


def test_emotion_change_uses_final_minus_earliest() -> None:
    assert trajectory_feature_values(1.0, 3.0, 2.0)["change"] == 1.0


def test_emotion_recent_change_uses_final_minus_middle() -> None:
    assert trajectory_feature_values(1.0, 3.0, 2.0)["recent_change"] == -1.0


def test_emotion_acceleration_is_difference_of_changes() -> None:
    assert trajectory_feature_values(1.0, 3.0, 2.0)["acceleration"] == -3.0


def test_emotion_persistence_reports_strict_monotonic_direction() -> None:
    assert trajectory_feature_values(1.0, 2.0, 3.0)["persistence"] == 1
    assert trajectory_feature_values(3.0, 2.0, 1.0)["persistence"] == -1
    assert trajectory_feature_values(1.0, 3.0, 2.0)["persistence"] == 0


def test_emotion_reversal_requires_two_nonzero_changes_with_different_signs() -> None:
    assert trajectory_feature_values(1.0, 3.0, 2.0)["reversal"] == 1
    assert trajectory_feature_values(1.0, 1.0, 2.0)["reversal"] == 0
    assert trajectory_feature_values(1.0, 2.0, 3.0)["reversal"] == 0


def test_peak_displacement_is_zero_at_peak_and_negative_after_retreat() -> None:
    assert trajectory_feature_values(1.0, 2.0, 3.0)["peak_displacement"] == 0.0
    assert trajectory_feature_values(1.0, 3.0, 2.0)["peak_displacement"] == -1.0


def test_each_preregistered_trajectory_regime_interaction_is_exact() -> None:
    frame = pd.DataFrame(
        {
            "transition_probability": [0.2],
            "posterior_entropy": [0.4],
            "top_second_margin": [0.5],
            "top_state_probability": [0.6],
            "arousal_change": [2.0],
            "frustration_change": [-2.0],
            "conviction_change": [3.0],
            "signed_pressure_acceleration": [-3.0],
            "tension_acceleration": [4.0],
            "signed_exhaustion_change": [-4.0],
        }
    )

    interactions, bounds = build_trajectory_interactions(frame)

    assert bounds == {}
    assert interactions.iloc[0].to_dict() == pytest.approx(
        {
            "transition_probability_x_arousal_change": 0.4,
            "posterior_entropy_x_frustration_change": -0.8,
            "top_second_margin_x_conviction_change": 1.5,
            "transition_probability_x_signed_pressure_acceleration": -0.6,
            "posterior_entropy_x_tension_acceleration": 1.6,
            "top_state_probability_x_signed_exhaustion_change": -2.4,
        }
    )


def test_multiclass_brier_is_mean_rowwise_one_hot_squared_error() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.25, 0.75]])
    assert multiclass_brier(np.asarray([0, 1]), probabilities) == pytest.approx(0.1025)


def test_prediction_entropy_uses_natural_log_and_exact_zero_handling() -> None:
    probabilities = np.asarray([[1.0, 0.0], [0.5, 0.5]])
    assert prediction_entropy(probabilities).tolist() == pytest.approx([0.0, np.log(2.0)])


def test_session_block_bootstrap_retains_complete_session_slates() -> None:
    frame = pd.DataFrame(
        {
            "session": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "decision_ordinal": [6, 6, 12, 12] * 2,
            "symbol": ["X", "Y", "X", "Y"] * 2,
        }
    )

    draws = session_block_bootstrap_draws(frame, draws=3, seed=7)
    repeated = session_block_bootstrap_draws(frame, draws=3, seed=7)

    assert len(draws) == 3
    for draw, repeated_draw in zip(draws, repeated, strict=True):
        assert draw.sampled_sessions == repeated_draw.sampled_sessions
        assert np.array_equal(draw.row_indices, repeated_draw.row_indices)
        assert len(draw.sampled_sessions) == 2
        assert len(draw.row_indices) == 8
        for offset, session in enumerate(draw.sampled_sessions):
            selected = frame.iloc[draw.row_indices[offset * 4 : (offset + 1) * 4]]
            assert selected["session"].eq(session).all()
            assert set(zip(selected["decision_ordinal"], selected["symbol"], strict=True)) == {
                (6, "X"),
                (6, "Y"),
                (12, "X"),
                (12, "Y"),
            }


def test_within_slate_permutation_keeps_each_trajectory_bundle_intact() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["A", "A", "A", "B", "B", "B"],
            "symbol": ["X", "Y", "Z", "X", "Y", "Z"],
            "target_class": ["N", "U", "R", "N", "U", "R"],
            "level": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "trajectory_a": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "trajectory_b": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
        }
    )

    permuted = permute_trajectory_bundle_within_slates(
        frame,
        features=("trajectory_a", "trajectory_b"),
        seed=17,
    )

    assert permuted[["slate_id", "symbol", "target_class", "level"]].equals(
        frame[["slate_id", "symbol", "target_class", "level"]]
    )
    for slate_id in ("A", "B"):
        before = frame.loc[frame["slate_id"].eq(slate_id), ["trajectory_a", "trajectory_b"]]
        after = permuted.loc[permuted["slate_id"].eq(slate_id), ["trajectory_a", "trajectory_b"]]
        assert sorted(map(tuple, before.to_numpy())) == sorted(map(tuple, after.to_numpy()))


def test_protected_date_rejection_fails_closed_before_transformation() -> None:
    safe = pd.DataFrame({"session": ["2025-08-22"]})
    reject_protected_dates(safe)

    with pytest.raises(BlockedScreen) as raised:
        reject_protected_dates(pd.DataFrame({"session": ["2025-08-23"]}))

    assert raised.value.code == "blocked_protected_boundary_failure"


def test_decision_logic_uses_preregistered_precedence_and_blockers() -> None:
    assert (
        decide_trajectory_screen(t1_pass=True, t2_pass=True, descriptive_structure=True)
        == "regime_mix_filters_behavioural_trajectories"
    )
    assert (
        decide_trajectory_screen(t1_pass=True, t2_pass=False, descriptive_structure=True)
        == "trajectory_main_effects_only"
    )
    assert (
        decide_trajectory_screen(t1_pass=False, t2_pass=False, descriptive_structure=True)
        == "descriptive_trajectory_structure_only"
    )
    assert (
        decide_trajectory_screen(t1_pass=False, t2_pass=False, descriptive_structure=False)
        == "no_behavioural_trajectory_increment"
    )
    assert (
        decide_trajectory_screen(
            t1_pass=False,
            t2_pass=False,
            descriptive_structure=False,
            blocker="blocked_insufficient_trajectory_support",
        )
        == "blocked_insufficient_trajectory_support"
    )


def test_frozen_m2_probabilities_reconstruct_from_serialized_parameters() -> None:
    repository = Path(__file__).resolve().parents[4]
    predecessor = (
        repository
        / "research"
        / "loop-funnel"
        / "20260721-emotion-regime-coarse-loop-family-v0"
        / "artifacts"
        / "primary"
    )
    panel = pd.read_parquet(predecessor / "decision_panel.parquet")
    assessment = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2025)].reset_index(
        drop=True
    )
    archived = pd.read_parquet(predecessor / "assessment_predictions.parquet")
    payload = json.loads((predecessor / "model_coefficients.json").read_text())["models"]["M2"]

    actual = manual_multinomial_probabilities(assessment, payload)
    expected = archived[
        [f"probability__M2__{target}" for target in payload["class_order"]]
    ].to_numpy(dtype=float)

    assert assessment[["symbol", "session", "decision_ordinal"]].equals(
        archived[["symbol", "session", "decision_ordinal"]]
    )
    assert float(np.max(np.abs(actual - expected))) <= 1e-12
