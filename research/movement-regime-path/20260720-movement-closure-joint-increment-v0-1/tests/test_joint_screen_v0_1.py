from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocker_research.movement_closure_joint_screen_v0_1 import (
    add_joint_probability_features,
    assert_compact_panel_has_no_forbidden_fields,
    assert_protected_date_boundary,
    assert_upstream_chronology,
    classify_joint_decision,
    evaluate_support,
    exact_active_pair_join,
    fit_fixed_logistic,
    logit_probability,
    session_block_bootstrap_improvements,
    split_development_assessment,
    whole_session_shift,
    whole_session_shift_feasibility,
    with_equal_slate_weights,
)


def _movement(
    *,
    movement_row_id: str = "movement-1",
    clock: str = "2025-01-02T15:35:00Z",
    ordinal: int = 12,
    representation_id: str = "repr-1",
    state: int = 3,
    available: bool = True,
    source_gap: bool = False,
) -> pd.DataFrame:
    fixed_clock = pd.Timestamp(clock)
    return pd.DataFrame(
        {
            "movement_row_id": [movement_row_id],
            "representation_id": [representation_id],
            "source_lineage_id": ["lineage-1"],
            "stock": ["AAA"],
            "session": [str(fixed_clock.date())],
            "decision_ordinal": [ordinal],
            "fixed_clock_timestamp": [fixed_clock],
            "movement_horizon_terminal_timestamp": [fixed_clock + pd.Timedelta(minutes=120)],
            "origin_segment_id": [f"AAA::{fixed_clock.date()}::segment_00"],
            "current_state_b": [state],
            "scheduled_bars_remaining": [65 if ordinal == 12 else 41],
            "p_move": [0.4],
            "predicted_absolute_movement_bps": [150.0],
            "large_move": [1],
            "movement_available": [available],
            "source_gap": [source_gap],
        }
    )


def _closure(
    *,
    pair_forecast_id: str = "pair-1",
    forecast: str = "2025-01-02T15:20:00Z",
    resolution: str | None = "2025-01-02T16:00:00Z",
    representation_id: str = "repr-1",
    state: int = 3,
    available: bool = True,
    source_gap: bool = False,
) -> pd.DataFrame:
    pair_forecast = pd.Timestamp(forecast)
    resolution_timestamp = pd.NaT if resolution is None else pd.Timestamp(resolution)
    return pd.DataFrame(
        {
            "pair_forecast_id": [pair_forecast_id],
            "representation_id": [representation_id],
            "source_lineage_id": ["lineage-1"],
            "stock": ["AAA"],
            "session": [str(pair_forecast.date())],
            "pair_forecast_timestamp": [pair_forecast],
            "closure_resolution_timestamp": [resolution_timestamp],
            "segment_id": [f"AAA::{pair_forecast.date()}::segment_00"],
            "current_state_b": [state],
            "pair_orientation": ["2->3"],
            "p_close_m2": [0.3],
            "p_close_m5": [0.5],
            "immediate_pair_closure": [1],
            "closure_available": [available],
            "source_gap": [source_gap],
        }
    )


def test_exact_active_pair_join_retains_only_causal_pair() -> None:
    result = exact_active_pair_join(_movement(), _closure())
    assert len(result.frame) == 1
    row = result.frame.iloc[0]
    assert row["pair_forecast_id"] == "pair-1"
    assert row["movement_row_id"] == "movement-1"
    assert row["pair_age_bars"] == 3
    assert row["closure_resolution_timestamp"] > row["fixed_clock_timestamp"]
    assert result.accounting["exact_joined_rows"] == 1


def test_pair_resolved_before_fixed_clock_is_excluded() -> None:
    closure = _closure(resolution="2025-01-02T15:30:00Z")
    result = exact_active_pair_join(_movement(), closure)
    assert result.frame.empty
    assert result.accounting["excluded_resolved_before_clock"] == 1


def test_pair_created_after_fixed_clock_is_excluded() -> None:
    closure = _closure(forecast="2025-01-02T15:40:00Z", resolution="2025-01-02T16:00:00Z")
    result = exact_active_pair_join(_movement(), closure)
    assert result.frame.empty
    assert result.accounting["excluded_no_active_pair"] == 1


def test_representation_mismatch_is_excluded() -> None:
    result = exact_active_pair_join(
        _movement(representation_id="repr-a"), _closure(representation_id="repr-b")
    )
    assert result.frame.empty
    assert result.accounting["excluded_representation_mismatch"] == 1


def test_source_gap_is_excluded() -> None:
    result = exact_active_pair_join(_movement(), _closure(source_gap=True))
    assert result.frame.empty
    assert result.accounting["excluded_source_gap"] == 1


def test_earliest_fixed_clock_deduplicates_one_pair_forecast() -> None:
    early = _movement(movement_row_id="early", clock="2025-01-02T15:35:00Z", ordinal=12)
    late = _movement(movement_row_id="late", clock="2025-01-02T17:35:00Z", ordinal=36)
    movement = pd.concat([early, late], ignore_index=True)
    closure = _closure(forecast="2025-01-02T15:20:00Z", resolution="2025-01-02T18:00:00Z")
    result = exact_active_pair_join(movement, closure)
    assert result.frame["movement_row_id"].tolist() == ["early"]
    assert result.accounting["excluded_duplicate_later_clock"] == 1


def test_pair_age_is_completed_five_minute_bar_count() -> None:
    result = exact_active_pair_join(
        _movement(clock="2025-01-02T15:35:00Z"),
        _closure(forecast="2025-01-02T15:05:00Z"),
    )
    assert result.frame.loc[0, "pair_age_bars"] == 6


def test_right_censored_pair_remains_unavailable_not_false() -> None:
    result = exact_active_pair_join(_movement(), _closure(resolution=None, available=False))
    assert result.frame.empty
    assert result.accounting["excluded_closure_unavailable"] == 1


def test_logit_clips_only_for_numerical_conversion() -> None:
    values = np.asarray([0.0, 0.25, 1.0])
    logits = logit_probability(values)
    assert logits[0] == pytest.approx(np.log(1e-6 / (1.0 - 1e-6)))
    assert logits[1] == pytest.approx(np.log(1.0 / 3.0))
    assert logits[2] == pytest.approx(np.log((1.0 - 1e-6) / 1e-6))
    assert values.tolist() == [0.0, 0.25, 1.0]


def test_closure_history_increment_and_joint_target() -> None:
    frame = pd.DataFrame(
        {
            "p_move": [0.25, 0.75],
            "p_close_m2": [0.2, 0.5],
            "p_close_m5": [0.4, 0.5],
            "predicted_absolute_movement_bps": [100.0, 200.0],
            "large_move": [1, 0],
            "immediate_pair_closure": [1, 1],
        }
    )
    result = add_joint_probability_features(frame)
    assert result.loc[0, "closure_history_increment"] == pytest.approx(
        np.log(0.4 / 0.6) - np.log(0.2 / 0.8)
    )
    assert result["joint_large_move_and_closure"].tolist() == [1, 0]
    assert result["p_move"].tolist() == frame["p_move"].tolist()


def test_2024_upstream_probabilities_must_be_out_of_fold() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2025],
            "session": ["2024-07-02", "2025-01-02"],
            "movement_oof": [True, False],
            "closure_m2_oof": [True, False],
            "closure_m5_oof": [True, False],
            "movement_trained_through": ["2024-06-28", "2024-12-31"],
            "movement_size_trained_through": ["2024-06-28", "2024-12-31"],
            "closure_trained_through": ["2024-06-30", "2024-12-31"],
            "movement_frozen_before_outcome": [False, True],
            "closure_frozen_before_outcome": [False, True],
            "movement_chronology_evidence_id": ["movement-proof", "movement-proof"],
            "closure_chronology_evidence_id": ["closure-proof", "closure-proof"],
        }
    )
    assert_upstream_chronology(frame)
    frame.loc[0, "movement_trained_through"] = "2024-07-02"
    with pytest.raises(AssertionError, match="in-sample"):
        assert_upstream_chronology(frame)


def test_no_2025_row_enters_downstream_fitting() -> None:
    frame = pd.DataFrame({"year": [2024, 2024, 2025], "value": [1, 2, 999]})
    development, assessment = split_development_assessment(frame)
    assert development["value"].tolist() == [1, 2]
    assert assessment["value"].tolist() == [999]


def test_equal_slate_weighting_totals_one_per_clock_slate() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2025-01-02"] * 6,
            "decision_ordinal": [12, 12, 36, 36, 36, 36],
        }
    )
    result = with_equal_slate_weights(frame)
    totals = result.groupby(["session", "decision_ordinal"])["row_weight"].sum()
    np.testing.assert_allclose(totals.to_numpy(), [1.0, 1.0])


def test_manual_logistic_prediction_reconstruction() -> None:
    rows = 60
    frame = pd.DataFrame(
        {
            "x1": np.linspace(-2.0, 2.0, rows),
            "x2": np.tile([-1.0, 1.0], rows // 2),
            "slate_id": [f"s-{index // 3}" for index in range(rows)],
        }
    )
    target = (frame["x1"] + 0.2 * frame["x2"] > 0.0).astype(int)
    model = fit_fixed_logistic(
        frame,
        target,
        features=("x1", "x2"),
        slate_column="slate_id",
        model_id="manual",
    )
    stored = model.as_dict()
    values = frame[["x1", "x2"]].to_numpy(float)
    standardized = (values - np.asarray(stored["means"])) / np.asarray(stored["scales"])
    linear = float(stored["intercept"]) + standardized @ np.asarray(stored["coefficients"])
    manual = 1.0 / (1.0 + np.exp(-linear))
    np.testing.assert_allclose(model.predict(frame), manual, atol=1e-14)


def test_session_block_bootstrap_keeps_paired_predictions() -> None:
    frame = pd.DataFrame(
        {
            "session": ["a", "a", "b", "b"],
            "slate_id": ["a-12", "a-12", "b-12", "b-12"],
            "target": [0, 1, 0, 1],
            "baseline": [0.4, 0.6, 0.4, 0.6],
            "candidate": [0.2, 0.8, 0.2, 0.8],
        }
    )
    result = session_block_bootstrap_improvements(
        frame,
        target="target",
        baseline="baseline",
        candidate="candidate",
        draws=5,
        seed=7,
    )
    assert len(result) == 10
    assert set(result["metric"]) == {"brier", "log_loss"}
    assert result["improvement"].gt(0.0).all()


def test_whole_session_shift_preserves_cross_section_within_ordinal() -> None:
    frame = pd.DataFrame(
        {
            "session": ["a"] * 2 + ["b"] * 2 + ["c"] * 2,
            "decision_ordinal": [12] * 6,
            "stock": ["A", "B"] * 3,
            "p1": [1.0, 2.0, 11.0, 12.0, 21.0, 22.0],
            "p2": [3.0, 4.0, 13.0, 14.0, 23.0, 24.0],
        }
    )
    shifted, manifest = whole_session_shift(frame, value_columns=("p1", "p2"), draw=0, seed=123)
    original_blocks = {
        tuple(map(tuple, group.sort_values("stock")[["p1", "p2"]].to_numpy()))
        for _, group in frame.groupby("session")
    }
    shifted_blocks = {
        tuple(map(tuple, group.sort_values("stock")[["p1", "p2"]].to_numpy()))
        for _, group in shifted.groupby("session")
    }
    assert shifted_blocks == original_blocks
    assert all(row["membership_size"] == 2 for row in manifest)
    assert all(row["source_session"] != row["destination_session"] for row in manifest)


def test_whole_session_shift_fails_closed_for_singleton_membership() -> None:
    frame = pd.DataFrame(
        {
            "session": ["a", "a", "b"],
            "decision_ordinal": [12, 12, 12],
            "stock": ["A", "B", "A"],
            "p1": [1.0, 2.0, 3.0],
        }
    )
    feasibility = whole_session_shift_feasibility(frame)
    assert int(feasibility["unshiftable_blocks"].sum()) == 2
    with pytest.raises(RuntimeError, match="blocked_join_semantics_failure"):
        whole_session_shift(frame, value_columns=("p1",), draw=0, seed=123)


def test_support_gates_accept_declared_minimums() -> None:
    sessions = [f"2025-{month:02d}-{day:02d}" for month in range(1, 9) for day in range(1, 11)]
    stocks = [f"S{index:02d}" for index in range(20)]
    rows = [(session, stock) for session in sessions for stock in stocks]
    frame = pd.DataFrame(rows, columns=["session", "stock"])
    frame["large_move"] = (np.arange(len(frame)) % 4 == 0).astype(int)
    frame["immediate_pair_closure"] = (np.arange(len(frame)) % 4 == 0).astype(int)
    frame["joint_large_move_and_closure"] = frame["large_move"] & frame["immediate_pair_closure"]
    frame["pair_orientation"] = [f"o-{index % 8}" for index in range(len(frame))]
    support = evaluate_support(frame)
    assert support["primary_support_passed"] is True
    assert support["joint_support_status"] == "sufficient"


def test_support_gate_blocks_primary_shortfall_and_only_marks_joint_shortfall() -> None:
    frame = pd.DataFrame(
        {
            "session": [f"2025-01-{(index % 28) + 1:02d}" for index in range(1500)],
            "stock": [f"S{index % 15:02d}" for index in range(1500)],
            "large_move": [1] * 300 + [0] * 1200,
            "immediate_pair_closure": [1] * 300 + [0] * 1200,
            "joint_large_move_and_closure": [1] * 99 + [0] * 1401,
            "pair_orientation": [f"o-{index % 6}" for index in range(1500)],
        }
    )
    support = evaluate_support(frame)
    assert support["primary_support_passed"] is False
    assert support["blocker"] == "blocked_insufficient_joint_increment_support"
    assert support["joint_support_status"] == "secondary_insufficient_joint_support"


@pytest.mark.parametrize(
    ("arm_a", "arm_b", "arm_c", "expected"),
    [
        (True, True, False, "mutually_informative_movement_closure_process"),
        (True, False, False, "movement_adds_to_closure_only"),
        (False, True, False, "closure_history_adds_to_movement_only"),
        (False, False, True, "joint_interaction_only"),
        (False, False, False, "separate_predictable_processes_no_increment"),
    ],
)
def test_decision_classification(arm_a: bool, arm_b: bool, arm_c: bool, expected: str) -> None:
    assert classify_joint_decision(arm_a_pass=arm_a, arm_b_pass=arm_b, arm_c_pass=arm_c) == expected


def test_blocker_decision_takes_precedence() -> None:
    assert (
        classify_joint_decision(
            arm_a_pass=True,
            arm_b_pass=True,
            arm_c_pass=True,
            blocker="blocked_chronology_or_leakage_failure",
        )
        == "blocked_chronology_or_leakage_failure"
    )


def test_protected_date_rejection() -> None:
    assert_protected_date_boundary(["2025-08-22T20:00:00Z"])
    with pytest.raises(ValueError, match="protected"):
        assert_protected_date_boundary(["2025-08-23T00:00:00Z"])
    with pytest.raises(ValueError, match="protected"):
        assert_protected_date_boundary(["2026-01-02T14:30:00Z"])


@pytest.mark.parametrize(
    "field",
    [
        "future_signed_return",
        "long_probability",
        "short_probability",
        "strategy_return",
        "pnl",
        "mfe",
        "mae",
        "profitable_loop_label",
        "payoff_history",
        "exact_five_state_history",
    ],
)
def test_forbidden_direction_and_economic_fields(field: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_compact_panel_has_no_forbidden_fields(pd.DataFrame({field: [0]}))
