from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.opening_regime_path_screen_v0 import (
    assert_allowed_feature_names,
    cohort_relative_returns_bps,
    current_regime_features,
    decide_screen,
    decision_bar_start_ordinal,
    decision_time_local,
    delayed_entry_and_terminal,
    development_movement_thresholds,
    equal_slate_weights,
    fit_fixed_logistic,
    interaction_features,
    manual_logistic_prediction,
    opening_path_features,
    permute_structural_bundle_within_slates,
    reject_invalid_decision_history,
    session_block_bootstrap_draws,
)


def _session_bars() -> pd.DataFrame:
    starts = pd.date_range(
        "2025-02-03 09:30",
        periods=78,
        freq="5min",
        tz="America/New_York",
    )
    return pd.DataFrame(
        {
            "bar_ordinal": np.arange(78),
            "bar_start_timestamp": starts.tz_convert("UTC"),
            "bar_complete_timestamp": (starts + pd.Timedelta(minutes=5)).tz_convert("UTC"),
            "open": 100.0 + np.arange(78),
            "close": 100.5 + np.arange(78),
            "segment_id": "AAL|2025-02-03|0",
            "session": "2025-02-03",
            "session_source_complete": True,
            "source_data_error_in_session": False,
            "expected_session_bars": 78,
        }
    )


def test_ordinal_6_extracts_bar_completed_at_1000() -> None:
    assert decision_bar_start_ordinal(6) == 5
    assert decision_time_local(6).isoformat() == "10:00:00"


def test_ordinal_12_extracts_bar_completed_at_1030() -> None:
    assert decision_bar_start_ordinal(12) == 11
    assert decision_time_local(12).isoformat() == "10:30:00"


def test_decision_requires_completed_bar() -> None:
    bars = _session_bars()
    valid = reject_invalid_decision_history(bars, decision_ordinal=6)
    assert valid.bar_start_ordinal == 5
    assert (
        valid.feature_available_timestamp.tz_convert("America/New_York").strftime("%H:%M")
        == "10:00"
    )

    incomplete = bars.copy()
    incomplete.loc[5, "bar_complete_timestamp"] = incomplete.loc[5, "bar_start_timestamp"]
    with pytest.raises(ValueError, match="completed"):
        reject_invalid_decision_history(incomplete, decision_ordinal=6)


def test_delayed_entry_uses_t_plus_2_and_terminal_is_session_close() -> None:
    bars = _session_bars()
    anchor = delayed_entry_and_terminal(bars, decision_ordinal=6)
    assert anchor.entry_bar_ordinal == 7
    assert anchor.delayed_entry_open == 107.0
    assert anchor.terminal_bar_ordinal == 77
    assert anchor.terminal_close == 177.5


def test_cohort_relative_outcome_uses_leave_one_stock_out_median() -> None:
    raw = np.array([10.0, 20.0, 100.0])
    residual, medians = cohort_relative_returns_bps(raw)
    np.testing.assert_allclose(medians, [60.0, 55.0, 15.0])
    np.testing.assert_allclose(residual, [-50.0, -35.0, 85.0])


def test_movement_threshold_is_fit_on_2024_only_by_checkpoint() -> None:
    panel = pd.DataFrame(
        {
            "year": [2024] * 8 + [2025] * 4,
            "decision_ordinal": [6] * 4 + [12] * 4 + [6, 6, 12, 12],
            "residual_remaining_return_bps": [
                -10.0,
                20.0,
                -30.0,
                100.0,
                -40.0,
                50.0,
                -60.0,
                200.0,
                9999.0,
                -9999.0,
                9999.0,
                -9999.0,
            ],
        }
    )
    thresholds = development_movement_thresholds(panel)
    assert thresholds == {6: 47.5, 12: 95.0}


def test_source_gap_is_rejected() -> None:
    bars = _session_bars()
    bars.loc[3, "segment_id"] = "AAL|2025-02-03|1"
    with pytest.raises(ValueError, match="source gap"):
        reject_invalid_decision_history(bars, decision_ordinal=6)


def test_session_boundary_is_rejected() -> None:
    bars = _session_bars()
    bars.loc[3, "session"] = "2025-02-04"
    with pytest.raises(ValueError, match="session boundary"):
        reject_invalid_decision_history(bars, decision_ordinal=6)


def test_protected_date_is_rejected() -> None:
    bars = _session_bars()
    bars["bar_start_timestamp"] = bars["bar_start_timestamp"] + pd.Timedelta(days=300)
    bars["bar_complete_timestamp"] = bars["bar_complete_timestamp"] + pd.Timedelta(days=300)
    with pytest.raises(ValueError, match="protected"):
        reject_invalid_decision_history(bars, decision_ordinal=6)


def test_transition_and_unique_state_counting() -> None:
    features = opening_path_features([1, 1, 2, 2, 3, 3])
    assert features["opening_transition_count"] == 2.0
    assert features["opening_unique_state_count"] == 3.0


def test_two_state_opening_closure() -> None:
    features = opening_path_features([1, 1, 2, 2, 1])
    assert features["opening_two_state_closure_count"] == 1.0
    assert features["opening_three_state_closure_count"] == 0.0
    assert features["opening_any_short_closure"] == 1.0
    assert features["opening_return_to_origin_count"] == 1.0


def test_three_state_opening_closure() -> None:
    features = opening_path_features([1, 2, 3, 1])
    assert features["opening_two_state_closure_count"] == 0.0
    assert features["opening_three_state_closure_count"] == 1.0
    assert features["opening_any_short_closure"] == 1.0


def test_non_closure_path() -> None:
    features = opening_path_features([1, 1, 2, 3, 3])
    assert features["opening_any_short_closure"] == 0.0
    assert features["opening_most_recent_path_was_closure"] == 0.0


def test_opening_state_revisit_count() -> None:
    features = opening_path_features([1, 2, 3, 2])
    assert features["opening_state_revisit_count"] == 1.0
    assert features["opening_return_to_origin_count"] == 0.0


def test_state_occupancy_entropy() -> None:
    features = opening_path_features([1, 1, 1, 2, 2])
    expected = -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))
    assert features["opening_state_occupancy_entropy"] == pytest.approx(expected)
    assert features["opening_largest_state_occupancy_fraction"] == pytest.approx(0.6)


def test_current_state_and_posterior_reconstruction() -> None:
    posterior = np.array([0.05, 0.1, 0.7, 0.05, 0.025, 0.025, 0.025, 0.025])
    features = current_regime_features([1, 1, 2, 2], posterior)
    assert features["current_state_2"] == 1.0
    assert features["previous_completed_state_1"] == 1.0
    assert features["posterior_state_2"] == pytest.approx(0.7)
    assert features["current_state_age"] == 2.0
    assert features["opening_state_equals_current"] == 0.0


def test_loop_by_current_state_interactions_are_frozen() -> None:
    topology = opening_path_features([1, 2, 3, 2, 2])
    interactions = interaction_features(2, topology, state_count=8)
    assert len(interactions) == 32
    assert interactions["current_state_2_x_any_short_closure"] == 1.0
    assert interactions["current_state_2_x_opening_return_to_origin_count"] == 0.0
    assert interactions["current_state_2_x_transition_rate"] == pytest.approx(3.0 / 5.0)
    assert interactions["current_state_2_x_current_state_age"] == 2.0
    assert interactions["current_state_1_x_transition_rate"] == 0.0


@pytest.mark.parametrize(
    "field",
    [
        "future_state",
        "future_run_duration",
        "profitable_loop_label",
        "payoff_history",
        "exact_loop_id",
    ],
)
def test_forbidden_future_state_and_economic_history_fields(field: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_allowed_feature_names(["opening_transition_count", field])


def test_equal_slate_weighting() -> None:
    weights = equal_slate_weights(pd.Series(["a", "a", "b", "b", "b"]))
    np.testing.assert_allclose(weights, [0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])
    assert weights[:2].sum() == pytest.approx(1.0)
    assert weights[2:].sum() == pytest.approx(1.0)


def test_manual_logistic_reconstruction() -> None:
    frame = pd.DataFrame(
        {
            "x": [-2.0, -1.0, 1.0, 2.0],
            "checkpoint_60m": [0.0, 1.0, 0.0, 1.0],
            "slate_id": ["a", "a", "b", "b"],
        }
    )
    model = fit_fixed_logistic(
        frame,
        pd.Series([0, 0, 1, 1]),
        features=["x", "checkpoint_60m"],
        slate_column="slate_id",
        model_id="M0_TEST",
    )
    np.testing.assert_allclose(
        model.predict(frame), manual_logistic_prediction(model.as_dict(), frame)
    )
    assert model.converged
    assert model.iterations < 250


def test_session_block_bootstrap_keeps_complete_sessions() -> None:
    draws = session_block_bootstrap_draws(
        ["2025-01-02", "2025-01-03", "2025-01-06"], draws=300, seed=20260720
    )
    assert len(draws) == 300
    assert all(len(draw.sampled_sessions) == 3 for draw in draws)
    assert set(draws[0].sampled_sessions).issubset({"2025-01-02", "2025-01-03", "2025-01-06"})


def test_within_slate_structural_permutation_keeps_bundle_together() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["a", "a", "a", "b", "b", "b"],
            "observable": [1, 2, 3, 4, 5, 6],
            "outcome": [0, 1, 0, 1, 0, 1],
            "structural_a": [10, 20, 30, 40, 50, 60],
            "structural_b": [11, 21, 31, 41, 51, 61],
        }
    )
    shuffled = permute_structural_bundle_within_slates(
        frame,
        structural_columns=["structural_a", "structural_b"],
        seed=20260721,
        draw=0,
    )
    assert shuffled[["observable", "outcome"]].equals(frame[["observable", "outcome"]])
    for slate in ("a", "b"):
        original_pairs = set(
            map(
                tuple,
                frame.loc[frame["slate_id"].eq(slate), ["structural_a", "structural_b"]].to_numpy(),
            )
        )
        shuffled_pairs = set(
            map(
                tuple,
                shuffled.loc[
                    shuffled["slate_id"].eq(slate), ["structural_a", "structural_b"]
                ].to_numpy(),
            )
        )
        assert shuffled_pairs == original_pairs
    assert not shuffled[["structural_a", "structural_b"]].equals(
        frame[["structural_a", "structural_b"]]
    )


@pytest.mark.parametrize(
    ("movement", "direction", "interaction", "expected"),
    [
        (True, True, False, "opening_regime_path_adds_movement_and_direction"),
        (True, False, False, "opening_regime_path_adds_movement_only"),
        (False, True, False, "opening_regime_path_adds_direction_only"),
        (False, False, True, "opening_loop_regime_interaction_only"),
        (False, False, False, "opening_structure_no_increment_over_price"),
    ],
)
def test_decision_logic(movement: bool, direction: bool, interaction: bool, expected: str) -> None:
    assert (
        decide_screen(
            {
                "movement_increment_passes": movement,
                "direction_increment_passes": direction,
                "interaction_increment_passes": interaction,
                "integrity_blocker": None,
            }
        )
        == expected
    )


def test_decision_logic_prioritizes_integrity_blocker() -> None:
    assert (
        decide_screen(
            {
                "movement_increment_passes": True,
                "direction_increment_passes": True,
                "interaction_increment_passes": True,
                "integrity_blocker": "blocked_chronology_or_leakage_failure",
            }
        )
        == "blocked_chronology_or_leakage_failure"
    )
