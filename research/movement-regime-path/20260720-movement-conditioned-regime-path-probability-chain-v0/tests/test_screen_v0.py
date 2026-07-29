from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.movement_regime_path_screen_v0 import (
    assert_allowed_feature_names,
    assert_protected_boundary,
    assert_stacking_chronology,
    bounded_monthly_smoke_population,
    circular_shift_session_blocks,
    classify_state_path,
    decide_screen,
    equal_slate_weights,
    expanding_month_folds,
    fit_fixed_logistic,
    fixed_ordinal_rows,
    leave_one_out_median,
    movement_thresholds,
    probability_chain,
    sampled_sessions,
)


def test_fixed_ordinal_extraction_is_exact_and_stable() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02"] * 5,
            "symbol": ["B", "A", "A", "A", "B"],
            "bar_ordinal": [36, 11, 12, 36, 12],
        }
    )
    result = fixed_ordinal_rows(frame)
    assert result["bar_ordinal"].tolist() == [12, 12, 36, 36]
    assert result["symbol"].tolist() == ["A", "B", "A", "B"]
    with pytest.raises(ValueError, match="exactly"):
        fixed_ordinal_rows(frame, ordinals=(12, 24))


def test_leave_one_out_cohort_median() -> None:
    result = leave_one_out_median([1.0, 2.0, 100.0])
    np.testing.assert_allclose(result, [51.0, 50.5, 1.5])


def test_movement_threshold_uses_2024_training_only_and_each_clock() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024] * 8 + [2025] * 2,
            "decision_ordinal": [12] * 4 + [36] * 4 + [12, 36],
            "absolute_movement_bps": [1, 2, 3, 4, 10, 20, 30, 40, 1_000_000, 2_000_000],
        }
    )
    result = movement_thresholds(frame)
    assert result == {12: 3.25, 36: 32.5}


@pytest.mark.parametrize(
    ("path", "transitions", "closure", "unique"),
    [
        ([1, 0] + [0] * 22, 2, True, 2),
        ([1, 2, 0] + [0] * 21, 3, True, 3),
        ([1] * 24, 1, False, None),
    ],
)
def test_transition_and_closure_topology(
    path: list[int], transitions: int, closure: bool, unique: int | None
) -> None:
    result = classify_state_path(0, path)
    assert result.transition_count == transitions
    assert result.transition_burst is (transitions >= 2)
    assert result.short_closure is closure
    assert result.first_closure_unique_states == unique


def test_source_gap_rejects_structural_target() -> None:
    result = classify_state_path(0, [1, 0] + [0] * 22, source_gap=True)
    assert result.transition_count == 0
    assert not result.transition_burst
    assert not result.short_closure


def test_session_boundary_rejects_structural_target() -> None:
    result = classify_state_path(0, [1, 0] + [0] * 22, crosses_session=True)
    assert result.transition_count == 0
    assert not result.transition_burst
    assert not result.short_closure


def test_expanding_oof_starts_with_january_through_june() -> None:
    sessions = pd.date_range("2024-01-01", "2024-12-01", freq="MS") + pd.Timedelta(days=14)
    frame = pd.DataFrame({"session": sessions})
    folds = expanding_month_folds(frame)
    assert [fold.score_month for fold in folds] == [
        "2024-07",
        "2024-08",
        "2024-09",
        "2024-10",
        "2024-11",
        "2024-12",
    ]
    assert folds[0].training_months == (
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
        "2024-06",
    )


def test_oof_upstream_provenance_is_strictly_earlier() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-07-02", "2024-08-02"],
            "p_move": [0.2, 0.3],
            "p_move__trained_through": ["2024-06-28", "2024-07-31"],
        }
    )
    assert_stacking_chronology(frame, prediction_columns=("p_move",))


def test_in_sample_stacked_feature_is_rejected() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-07-02"],
            "p_move": [0.2],
            "p_move__trained_through": ["2024-07-02"],
        }
    )
    with pytest.raises(AssertionError, match="in-sample"):
        assert_stacking_chronology(frame, prediction_columns=("p_move",))


def test_probability_multiplication_and_signed_score() -> None:
    result = probability_chain([0.8], [0.75], [100.0])
    assert result["p_long"][0] == pytest.approx(0.6)
    assert result["p_short"][0] == pytest.approx(0.2)
    assert result["p_neutral"][0] == pytest.approx(0.2)
    assert result["score"][0] == pytest.approx(40.0)
    assert sum(result[name][0] for name in ("p_long", "p_short", "p_neutral")) == pytest.approx(1.0)


def test_model_weights_give_each_slate_total_one() -> None:
    slates = pd.Series(["a", "a", "b", "b", "b", "b"])
    weights = equal_slate_weights(slates)
    assert weights[:2].sum() == pytest.approx(1.0)
    assert weights[2:].sum() == pytest.approx(1.0)


def test_smoke_population_keeps_complete_slates_across_every_month() -> None:
    frame = pd.DataFrame(
        {
            "session": [
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-07-02",
                "2024-07-02",
                "2024-07-03",
                "2024-07-03",
                "2025-01-02",
                "2025-01-02",
                "2025-01-03",
                "2025-01-03",
            ],
            "slate_id": ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 2 + ["e"] * 2 + ["f"] * 2,
            "symbol": ["A", "B"] * 6,
            "decision_ordinal": [12] * 12,
        }
    )
    result = bounded_monthly_smoke_population(frame, 6)
    assert result["slate_id"].unique().tolist() == ["a", "c", "e"]
    assert result.groupby("slate_id").size().eq(2).all()
    assert pd.to_datetime(result["session"]).dt.strftime("%Y-%m").nunique() == 3


def test_smoke_population_rejects_budget_that_cannot_cover_every_month() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-02", "2025-01-02", "2025-01-02"],
            "slate_id": ["a", "a", "b", "b"],
        }
    )
    with pytest.raises(ValueError, match="one complete slate per observed month"):
        bounded_monthly_smoke_population(frame, 3)


def test_manual_logistic_prediction_reconstruction() -> None:
    rows = 40
    frame = pd.DataFrame(
        {
            "x1": np.linspace(-2.0, 2.0, rows),
            "x2": np.tile([-1.0, 1.0], rows // 2),
            "slate_id": [f"s{index // 4}" for index in range(rows)],
        }
    )
    target = (frame["x1"] + 0.25 * frame["x2"] > 0.0).astype(int)
    model = fit_fixed_logistic(
        frame,
        target,
        features=("x1", "x2"),
        slate_column="slate_id",
        model_id="manual_test",
    )
    values = frame[["x1", "x2"]].to_numpy(float)
    linear = model.intercept + ((values - model.means) / model.scales) @ model.coefficients
    manual = 1.0 / (1.0 + np.exp(-linear))
    np.testing.assert_allclose(model.predict(frame), manual, atol=1e-14)


def test_protected_date_rejection() -> None:
    assert_protected_boundary(["2025-08-22T20:00:00Z"])
    with pytest.raises(ValueError, match="protected"):
        assert_protected_boundary(["2025-08-23T00:00:00Z"])
    with pytest.raises(ValueError, match="protected"):
        assert_protected_boundary(["2026-01-02T14:30:00Z"])


def test_null_shift_preserves_whole_session_cross_section() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2025-01-02"] * 2 + ["2025-01-03"] * 2 + ["2025-01-06"] * 2,
            "decision_ordinal": [12] * 6,
            "symbol": ["A", "B"] * 3,
            "p": [1.0, 2.0, 11.0, 12.0, 21.0, 22.0],
        }
    )
    shifted, manifest = circular_shift_session_blocks(frame, value_columns=("p",), draw=0, seed=123)
    original_blocks = {
        tuple(group.sort_values("symbol")["p"]) for _, group in frame.groupby("session", sort=True)
    }
    shifted_blocks = {
        tuple(group.sort_values("symbol")["p"])
        for _, group in shifted.groupby("session", sort=True)
    }
    assert shifted_blocks == original_blocks
    assert all(row["membership_size"] == 2 for row in manifest)


def test_bootstrap_draws_are_whole_session_identifiers() -> None:
    draws = sampled_sessions(["a", "b", "c"], draws=5, seed=7)
    assert len(draws) == 5
    assert all(len(draw) == 3 for draw in draws)
    assert all(set(draw).issubset({"a", "b", "c"}) for draw in draws)


def _passing_evidence() -> dict[str, object]:
    return {
        "p1_minus_p0_brier_improvement": 0.01,
        "p1_minus_p0_log_loss_improvement": 0.01,
        "b1_minus_b0_brier_improvement": 0.01,
        "b1_minus_b0_log_loss_improvement": 0.01,
        "d1_minus_d0_brier_improvement": 0.01,
        "d1_minus_d0_log_loss_improvement": 0.01,
        "b1_positive_months": 5,
        "d1_positive_months": 5,
        "b1_bootstrap_90_lower": 0.0,
        "d1_bootstrap_90_lower": 0.0,
        "b1_null_percentile": 0.90,
        "d1_null_percentile": 0.90,
        "observable_spearman": 0.01,
        "path_spearman": 0.02,
        "observable_top_one_minus_median": 1.0,
        "path_top_one_minus_median": 2.0,
        "concentration_passed": True,
        "exact_rerun_passed": True,
        "independent_audit_passed": True,
    }


def test_decision_gate_requires_every_promising_condition() -> None:
    assert decide_screen(_passing_evidence()) == "promising_probability_chain_for_intensive_v1"
    evidence = _passing_evidence()
    evidence["d1_minus_d0_brier_improvement"] = -0.01
    assert decide_screen(evidence) == "structural_increment_without_directional_value"
    evidence = _passing_evidence()
    evidence["b1_minus_b0_brier_improvement"] = -0.01
    assert decide_screen(evidence) == "movement_predictable_but_no_structural_increment"


@pytest.mark.parametrize(
    "feature",
    [
        "exact_loop_id",
        "selected_loop_membership",
        "prior_profitable_loop_label",
        "payoff_history",
        "future_state_path",
        "future_movement",
        "outcome_label",
    ],
)
def test_forbidden_exact_loop_future_and_payoff_fields(feature: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_allowed_feature_names((feature,))


def test_outcome_free_causal_surface_accepts_declared_primitives() -> None:
    assert_allowed_feature_names(
        (
            "absolute_return_12",
            "posterior_state_0",
            "posterior_entropy",
            "current_hard_state_age",
            "p_move",
        )
    )
