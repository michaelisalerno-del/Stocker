from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from stocker_research.pressure_onset_screen_v0 import (
    activity_acceleration,
    assert_allowed_feature_names,
    assert_safe_timestamps,
    classify_onset,
    close_location_pressure,
    cohort_relative_cumulative_paths_bps,
    confirmation_deltas,
    decide_pressure_screen,
    decision_bar_start_ordinal,
    decision_time_local,
    development_onset_barriers,
    directional_efficiency,
    equal_slate_weights,
    expanding_monthly_oof_probabilities,
    extract_decision_window,
    manual_logistic_prediction,
    movement_admission_thresholds,
    new_extreme_counts,
    opening_range_acceptance,
    permute_feature_bundle_within_slates,
    progress_per_activity,
    range_acceleration,
    relative_strength_acceleration,
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
            "high": 101.0 + np.arange(78),
            "low": 99.0 + np.arange(78),
            "close": 100.5 + np.arange(78),
            "volume": 1_000.0 + np.arange(78),
            "session": "2025-02-03",
            "source_quality_passed": True,
            "corporate_action_passed": True,
        }
    )


def test_ordinal_6_and_12_use_completed_1000_and_1030_bars() -> None:
    assert decision_bar_start_ordinal(6) == 5
    assert decision_time_local(6).isoformat() == "10:00:00"
    assert decision_bar_start_ordinal(12) == 11
    assert decision_time_local(12).isoformat() == "10:30:00"


def test_confirmation_t_plus_1_entry_t_plus_2_and_three_bar_onset_path() -> None:
    window = extract_decision_window(_session_bars(), decision_ordinal=6)
    assert window.decision_bar_ordinal == 5
    assert window.confirmation_bar_ordinal == 6
    assert (
        window.confirmation_available_timestamp.tz_convert("America/New_York").strftime("%H:%M")
        == "10:05"
    )
    assert window.entry_bar_ordinal == 7
    assert window.delayed_entry_open == 107.0
    assert window.onset_bar_ordinals == (7, 8, 9)
    np.testing.assert_allclose(window.onset_closes, [107.5, 108.5, 109.5])
    assert window.continuation_exit_bar_ordinal == 13
    assert window.continuation_exit_close == 113.5
    assert window.terminal_bar_ordinal == 77


def test_cohort_relative_cumulative_return_uses_leave_one_out_median() -> None:
    raw_bps = np.array(
        [
            [10.0, 30.0, 50.0],
            [20.0, 10.0, -10.0],
            [30.0, 40.0, 20.0],
        ]
    )
    residuals, medians = cohort_relative_cumulative_paths_bps(raw_bps)
    np.testing.assert_allclose(medians, [[25.0, 25.0, 5.0], [20.0, 35.0, 35.0], [15.0, 20.0, 20.0]])
    np.testing.assert_allclose(
        residuals,
        [[-15.0, 5.0, 45.0], [0.0, -25.0, -45.0], [15.0, 20.0, 0.0]],
    )


def test_onset_barrier_is_checkpoint_specific_and_2024_only() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024] * 8 + [2025] * 2,
            "decision_ordinal": [6] * 4 + [12] * 4 + [6, 12],
            "residual_t_plus_2_bps": [10, 20, 30, 100, 40, 50, 60, 200, 9999, 9999],
            "residual_t_plus_3_bps": [0] * 10,
            "residual_t_plus_4_bps": [0] * 10,
        }
    )
    assert development_onset_barriers(frame) == {6: 47.5, 12: 95.0}


def test_upward_onset_uses_first_completed_close_crossing() -> None:
    assert classify_onset([5.0, 12.0, -15.0], barrier_bps=10.0) == "UP_ONSET"


def test_downward_onset_uses_first_completed_close_crossing() -> None:
    assert classify_onset([-3.0, -11.0, 20.0], barrier_bps=10.0) == "DOWN_ONSET"


def test_no_onset_when_neither_barrier_is_crossed() -> None:
    assert classify_onset([-3.0, 9.0, -9.0], barrier_bps=10.0) == "NO_ONSET"


def test_relative_strength_acceleration_is_latest_minus_previous() -> None:
    assert relative_strength_acceleration(14.0, -6.0) == 20.0


def test_activity_acceleration_uses_log1p_window_means() -> None:
    actual = activity_acceleration([3.0, 5.0], [1.0, 1.0, 1.0, 1.0])
    assert actual == pytest.approx(np.log1p(4.0) - np.log1p(1.0))


def test_range_acceleration_handles_effectively_zero_denominator() -> None:
    assert range_acceleration([4.0, 6.0], [2.0, 2.0, 2.0, 2.0]) == 2.5
    assert np.isnan(range_acceleration([4.0, 6.0], [0.0, 0.0, 0.0, 0.0]))


def test_signed_and_absolute_efficiency_use_completed_returns() -> None:
    signed, absolute = directional_efficiency([0.01, -0.005, 0.015])
    assert signed == pytest.approx(2.0 / 3.0)
    assert absolute == pytest.approx(2.0 / 3.0)
    unavailable = directional_efficiency([0.0, 0.0, 0.0])
    assert np.isnan(unavailable[0]) and np.isnan(unavailable[1])


def test_progress_per_activity_uses_safe_positive_denominator() -> None:
    assert progress_per_activity(30.0, 2.0) == 15.0
    assert progress_per_activity(30.0, 0.0) == pytest.approx(30.0 / 1e-12)


def test_close_location_pressure_uses_latest_three_completed_bars() -> None:
    pressure = close_location_pressure(
        highs=[10.0, 12.0, 14.0],
        lows=[0.0, 2.0, 4.0],
        closes=[9.0, 7.0, 5.0],
    )
    assert pressure["current_close_location"] == 0.1
    assert pressure["mean_close_location_last_3"] == pytest.approx(0.5)
    assert pressure["upper_quartile_close_fraction_last_3"] == pytest.approx(1.0 / 3.0)
    assert pressure["lower_quartile_close_fraction_last_3"] == pytest.approx(1.0 / 3.0)


def test_new_extreme_progression_counts_latest_three_completed_bars() -> None:
    highs = [10.0, 11.0, 10.5, 12.0, 11.5]
    lows = [8.0, 8.5, 7.5, 7.0, 7.2]
    assert new_extreme_counts(highs, lows, latest=3) == (1, 2)


def test_opening_range_acceptance_detects_return_inside() -> None:
    acceptance = opening_range_acceptance(
        closes=[100.0, 101.0, 99.0, 102.0, 100.5],
        initial_high=101.0,
        initial_low=99.0,
    )
    assert acceptance == {
        "close_above_initial_3_high": 0.0,
        "close_below_initial_3_low": 0.0,
        "completed_closes_outside_initial_range": 1.0,
        "latest_close_returned_inside_initial_range": 1.0,
    }


def test_confirmation_deltas_use_only_t_and_completed_t_plus_1() -> None:
    at_t = {
        "cohort_relative_return_bps": 10.0,
        "relative_strength_acceleration": -2.0,
        "activity_shock": 0.4,
        "range_acceleration": 1.2,
        "signed_efficiency_3": 0.1,
        "current_close_location": 0.3,
    }
    at_t_plus_1 = {
        "cohort_relative_return_bps": 25.0,
        "relative_strength_acceleration": 3.0,
        "activity_shock": 0.7,
        "range_acceleration": 1.5,
        "signed_efficiency_3": 0.4,
        "current_close_location": 0.8,
    }
    result = confirmation_deltas(
        at_t,
        at_t_plus_1,
        new_high=True,
        new_low=False,
        favourable_retracement_bps=7.0,
        opening_range_acceptance_persisted=True,
        predicted_direction_remained_same=False,
    )
    assert result == pytest.approx(
        {
            "change_cohort_relative_return_bps": 15.0,
            "change_relative_strength_acceleration": 5.0,
            "change_activity_shock": 0.3,
            "change_range_acceleration": 0.3,
            "change_signed_efficiency_3": 0.3,
            "change_close_location": 0.5,
            "new_high_at_t_plus_1": 1.0,
            "new_low_at_t_plus_1": 0.0,
            "favourable_retracement_bps": 7.0,
            "opening_range_acceptance_persisted": 1.0,
            "predicted_direction_remained_same": 0.0,
        }
    )


def test_expanding_monthly_movement_probabilities_are_chronology_safe() -> None:
    rows: list[dict[str, object]] = []
    for month in range(1, 9):
        for index, (feature, target) in enumerate(
            zip([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1], strict=True)
        ):
            rows.append(
                {
                    "year_month": f"2024-{month:02d}",
                    "slate_id": f"2024-{month:02d}|{index}",
                    "x": feature,
                    "target": target,
                }
            )
    frame = pd.DataFrame(rows)
    probabilities, manifest = expanding_monthly_oof_probabilities(
        frame,
        target_column="target",
        features=["x"],
        slate_column="slate_id",
        model_id="movement_oof_test",
    )
    assert probabilities.loc[frame["year_month"].le("2024-06")].isna().all()
    assert probabilities.loc[frame["year_month"].ge("2024-07")].between(0.0, 1.0).all()
    assert [fold["score_month"] for fold in manifest] == ["2024-07", "2024-08"]
    assert all(fold["training_end_month"] < fold["score_month"] for fold in manifest)


def test_movement_admission_threshold_uses_only_oof_development_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024] * 8 + [2025, 2025],
            "decision_ordinal": [6] * 4 + [12] * 4 + [6, 12],
            "p_large_remaining_move": [0.1, 0.2, 0.3, 0.4, 0.2, 0.4, 0.6, 0.8, 0.999, 0.999],
        }
    )
    assert movement_admission_thresholds(frame) == {6: 0.325, 12: 0.65}


def test_equal_slate_weighting_gives_each_slate_total_weight_one() -> None:
    slates = pd.Series(["a", "a", "b", "b", "b"])
    np.testing.assert_allclose(equal_slate_weights(slates), [0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])


def test_manual_logistic_reconstruction_matches_worked_probability() -> None:
    model = {
        "feature_names": ["x", "y"],
        "means": [1.0, 2.0],
        "scales": [2.0, 4.0],
        "coefficients": [0.5, -1.0],
        "intercept": 0.25,
    }
    frame = pd.DataFrame({"x": [3.0], "y": [6.0]})
    expected = 1.0 / (1.0 + np.exp(0.25))
    assert manual_logistic_prediction(model, frame)[0] == pytest.approx(expected)


def test_predecessor_movement_model_reconstructs_original_2025_predictions() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    artifacts = (
        repo_root
        / "research/opening-regime-path/20260720-opening-regime-path-direction-screen-v0"
        / "artifacts/primary"
    )
    panel = pd.read_parquet(artifacts / "opening_decision_panel.parquet")
    assessment = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    coefficients = json.loads((artifacts / "model_coefficients.json").read_text())
    model = coefficients["models"]["large_remaining_move"]["M1"]
    scoring = panel.loc[panel["year"].eq(2025)].sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    expected = assessment.sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    assert (
        scoring[["symbol", "session", "decision_ordinal"]]
        .reset_index(drop=True)
        .equals(expected[["symbol", "session", "decision_ordinal"]].reset_index(drop=True))
    )
    reconstructed = manual_logistic_prediction(model, scoring)
    archived = expected["p__large_remaining_move__M1"].to_numpy(dtype=float)
    assert float(np.max(np.abs(reconstructed - archived))) <= 1e-12
    labels = expected["large_remaining_move"].to_numpy(dtype=int)
    metrics = pd.read_csv(artifacts / "movement_metrics.csv")
    archived_metrics = metrics.loc[metrics["scope"].eq("pooled") & metrics["model"].eq("M1")].iloc[
        0
    ]
    assert abs(brier_score_loss(labels, reconstructed) - archived_metrics["brier_score"]) <= 1e-12
    assert abs(log_loss(labels, reconstructed) - archived_metrics["log_loss"]) <= 1e-12
    assert abs(roc_auc_score(labels, reconstructed) - archived_metrics["auc"]) <= 1e-12


def test_session_block_bootstrap_preserves_complete_sessions() -> None:
    draws = session_block_bootstrap_draws(["s1", "s2", "s3"], draws=4, seed=9)
    assert len(draws) == 4
    assert all(len(draw.sampled_sessions) == 3 for draw in draws)
    assert all(set(draw.sampled_sessions).issubset({"s1", "s2", "s3"}) for draw in draws)


def test_within_slate_bundle_permutation_keeps_pressure_features_together() -> None:
    frame = pd.DataFrame(
        {
            "slate_id": ["a", "a", "a", "b", "b", "b"],
            "readiness": [1, 2, 3, 4, 5, 6],
            "outcome": [0, 1, 0, 1, 0, 1],
            "pressure_x": [10, 20, 30, 40, 50, 60],
            "pressure_y": [11, 21, 31, 41, 51, 61],
        }
    )
    permuted = permute_feature_bundle_within_slates(frame, ["pressure_x", "pressure_y"], seed=14)
    assert permuted[["readiness", "outcome", "slate_id"]].equals(
        frame[["readiness", "outcome", "slate_id"]]
    )
    assert (permuted["pressure_y"] - permuted["pressure_x"]).eq(1).all()
    for slate_id in ("a", "b"):
        before = frame.loc[frame["slate_id"].eq(slate_id), "pressure_x"].sort_values().tolist()
        after = permuted.loc[permuted["slate_id"].eq(slate_id), "pressure_x"].sort_values().tolist()
        assert before == after


def test_protected_date_rejection() -> None:
    with pytest.raises(ValueError, match="protected"):
        assert_safe_timestamps(pd.Series([pd.Timestamp("2025-08-23T00:00:00Z")]))


@pytest.mark.parametrize(
    "field",
    [
        "current_regime",
        "state_age",
        "loop_score",
        "closure_count",
        "excursion_bps",
        "transition_count",
        "posterior_probability",
        "structural_path",
    ],
)
def test_forbidden_loop_regime_and_state_fields(field: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_allowed_feature_names([field])


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (
            {"occurrence_passes": True, "direction_passes": True},
            "pressure_onset_and_direction_increment_observed",
        ),
        (
            {"occurrence_passes": True, "direction_passes": False},
            "pressure_onset_occurrence_only",
        ),
        (
            {"occurrence_passes": False, "direction_passes": True},
            "directional_pressure_only",
        ),
        (
            {
                "occurrence_passes": False,
                "direction_passes": False,
                "confirmation_occurrence_passes": True,
            },
            "one_bar_confirmation_required",
        ),
        (
            {
                "occurrence_passes": False,
                "direction_passes": False,
                "readiness_useful": True,
            },
            "movement_readiness_but_direction_unresolved",
        ),
        ({"occurrence_passes": False, "direction_passes": False}, "no_pressure_onset_increment"),
    ],
)
def test_decision_logic(evidence: dict[str, bool], expected: str) -> None:
    assert decide_pressure_screen(evidence) == expected


def test_decision_logic_fails_closed_on_integrity_blocker() -> None:
    evidence = {
        "occurrence_passes": True,
        "direction_passes": True,
        "integrity_blocker": "blocked_reproducibility_or_audit_failure",
    }
    assert decide_pressure_screen(evidence) == "blocked_reproducibility_or_audit_failure"
