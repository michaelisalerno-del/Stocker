from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stocker_research.m1c_low_movement_v0 import (
    assert_unprotected_sessions,
    assign_frozen_bins,
    calculate_checkpoint_outcomes,
    choose_overall_decision,
    construct_fresh_quiet_episodes,
    evaluate_low_movement_veto_gate,
    evaluate_short_premium_readiness_gate,
    evaluate_support_gate,
    freeze_weighted_boundaries,
    iv_expected_absolute,
    iv_sigma,
    matched_random_selection,
    permute_probabilities_within_sessions,
    reconstruct_frozen_probabilities,
    tail_memberships,
    tail_overlap,
    validate_causal_features,
    weighted_quantile,
    whole_session_bootstrap_plan,
)


def test_weighted_low_tail_quantiles_use_the_frozen_midpoint_cdf() -> None:
    values = np.array([0.10, 0.20, 0.30, 0.90])
    weights = np.ones(4)

    assert weighted_quantile(values, weights, 0.05) == pytest.approx(0.10)
    assert weighted_quantile(values, weights, 0.50) == pytest.approx(0.25)
    assert weighted_quantile(values, weights, 0.95) == pytest.approx(0.90)


def test_frozen_decile_boundaries_are_fit_once_and_applied_inclusively() -> None:
    development = np.arange(0.05, 1.0, 0.1)
    boundaries = freeze_weighted_boundaries(
        development,
        np.ones(10),
        quantiles=tuple(index / 10 for index in range(1, 10)),
    )

    assert boundaries == pytest.approx(tuple(index / 10 for index in range(1, 10)))
    assert assign_frozen_bins(
        np.array([0.05, 0.10, 0.1001, 0.90, 0.95]),
        boundaries,
    ).tolist() == [1, 1, 2, 9, 10]


def test_bottom_five_ten_and_twenty_percent_memberships_are_inclusive() -> None:
    probabilities = pd.Series([0.049, 0.050, 0.051, 0.10, 0.20, 0.201])
    memberships = tail_memberships(
        probabilities,
        {
            "bottom_5_percent": 0.05,
            "bottom_10_percent": 0.10,
            "bottom_20_percent": 0.20,
        },
    )

    assert memberships["bottom_5_percent"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert memberships["bottom_10_percent"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert memberships["bottom_20_percent"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]


def test_causal_feature_validation_rejects_contaminated_descendants() -> None:
    permitted = validate_causal_features(
        ("atm_iv", "arousal", "prior_6_mean_range"),
        forbidden=(
            "signed_pressure",
            "tension",
            "peer_normalised_pressure",
        ),
    )
    assert permitted == ("atm_iv", "arousal", "prior_6_mean_range")

    with pytest.raises(ValueError, match="contaminated"):
        validate_causal_features(
            ("atm_iv", "signed_pressure"),
            forbidden=("signed_pressure", "tension"),
        )


def test_causal_m1c_probability_reconstruction_uses_frozen_design_order() -> None:
    frame = pd.DataFrame(
        {
            "stock": ["AAA", "BBB"],
            "feature": [0.5, 0.5],
        }
    )
    specification = {
        "kind": "logistic",
        "numeric_features": ["feature"],
        "numeric_medians": [0.0],
        "numeric_means": [0.0],
        "numeric_scales": [1.0],
        "category_controls": ["stock"],
        "category_levels": {"stock": ["AAA", "BBB"]},
        "design_columns": ["feature", "control_stock__BBB"],
        "coefficients": [2.0, -1.0],
        "intercept": 0.0,
    }

    probabilities = reconstruct_frozen_probabilities(frame, specification)

    assert probabilities.tolist() == pytest.approx([0.7310585786300049, 0.5])


def _worked_five_minute_bars() -> pd.DataFrame:
    closes = [
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        101.0,
        99.0,
        102.0,
        100.0,
        98.0,
        103.0,
        101.0,
        100.0,
        102.0,
        99.0,
        101.0,
        104.0,
    ]
    highs = [100.5] * 6 + [
        103.0,
        102.0,
        105.0,
        103.0,
        101.0,
        104.0,
        103.0,
        102.0,
        104.0,
        101.0,
        103.0,
        106.0,
    ]
    lows = [99.5] * 6 + [
        99.0,
        96.0,
        98.0,
        97.0,
        95.0,
        97.0,
        96.0,
        97.0,
        98.0,
        94.0,
        97.0,
        98.0,
    ]
    starts = pd.date_range("2025-01-02 14:30Z", periods=18, freq="5min")
    return pd.DataFrame(
        {
            "stock": "AAA",
            "session": "2025-01-02",
            "bar_ordinal": range(18),
            "bar_start_timestamp": starts,
            "bar_complete_timestamp": starts + pd.Timedelta(minutes=5),
            "open": [100.0] * 18,
            "high": highs,
            "low": lows,
            "close": closes,
            "vti__bar_log_return": [0.0] * 18,
        }
    )


def test_entry_and_terminal_returns_use_the_first_future_bar_open() -> None:
    checkpoints = pd.DataFrame(
        {
            "row_id": ["AAA|2025-01-02|6"],
            "stock": ["AAA"],
            "session": ["2025-01-02"],
            "checkpoint": [6],
            "feature_available_timestamp_utc": pd.to_datetime(["2025-01-02 15:00Z"]),
            "atm_iv": [0.50],
        }
    )

    movement, _ = calculate_checkpoint_outcomes(checkpoints, _worked_five_minute_bars())
    row = movement.iloc[0]

    assert row["entry_timestamp"] == pd.Timestamp("2025-01-02 15:00Z")
    assert row["entry_price"] == 100.0
    assert row["signed_return_5m"] == pytest.approx(0.009950330853168092)
    assert row["signed_return_10m"] == pytest.approx(-0.01005033585350145)
    assert row["signed_return_15m"] == pytest.approx(0.01980262729617973)
    assert row["signed_return_30m"] == pytest.approx(0.02955880224154443)
    assert row["signed_return_60m"] == pytest.approx(0.03922071315328133)
    assert row["absolute_return_15m"] == pytest.approx(0.01980262729617973)
    assert row["terminal_return_sign_15m"] == 1


def test_path_excursions_and_range_include_every_future_bar_through_horizon() -> None:
    checkpoints = pd.DataFrame(
        {
            "row_id": ["AAA|2025-01-02|6"],
            "stock": ["AAA"],
            "session": ["2025-01-02"],
            "checkpoint": [6],
            "feature_available_timestamp_utc": pd.to_datetime(["2025-01-02 15:00Z"]),
            "atm_iv": [0.50],
        }
    )

    _, paths = calculate_checkpoint_outcomes(checkpoints, _worked_five_minute_bars())
    row = paths.iloc[0]

    assert row["maximum_up_excursion_15m"] == pytest.approx(0.04879016416943205)
    assert row["maximum_down_excursion_15m"] == pytest.approx(-0.040821994520255166)
    assert row["maximum_absolute_excursion_15m"] == pytest.approx(0.04879016416943205)
    assert row["realised_path_range_15m"] == pytest.approx(0.08961215868968714)
    assert row["time_to_maximum_up_excursion_15m"] == 15
    assert row["time_to_maximum_down_excursion_15m"] == 10
    assert row["time_to_maximum_absolute_excursion_15m"] == 15
    assert bool(row["crossed_above_and_below_entry_15m"])
    assert row["time_to_1_5sigma_breach_15m"] == 5
    assert row["breach_direction_1_5sigma_15m"] == "both"
    assert not bool(row["breach_mean_reverted_1_5sigma_15m"])


def test_iv_sigma_expected_absolute_residual_and_excursion_ratio_scaling() -> None:
    sigma = iv_sigma(0.50, 15)
    expected = iv_expected_absolute(0.50, 15)

    assert sigma == pytest.approx(0.00617707763884251)
    assert expected == pytest.approx(0.004928594878913057)
    assert 0.01980262729617973 - expected == pytest.approx(0.014874032417266673)
    assert 0.04879016416943205 / sigma == pytest.approx(7.89858360572178)
    assert expected == pytest.approx(sigma * math.sqrt(2.0 / math.pi))


def test_fresh_quiet_episodes_require_a_low_crossing_and_thirty_minute_spacing() -> None:
    rows = pd.DataFrame(
        {
            "row_id": [f"AAA|2025-01-02|{checkpoint}" for checkpoint in range(6, 20, 2)],
            "stock": ["AAA"] * 7,
            "session": ["2025-01-02"] * 7,
            "checkpoint": list(range(6, 20, 2)),
            "m1c_probability": [0.09, 0.08, 0.12, 0.07, 0.06, 0.13, 0.05],
            "entry_timestamp": pd.date_range("2025-01-02 15:00Z", periods=7, freq="10min"),
            "available_horizons": ["5,10,15,30,60"] * 7,
        }
    )

    episodes = construct_fresh_quiet_episodes(
        rows,
        threshold=0.10,
        probability_column="m1c_probability",
    )

    assert episodes["checkpoint"].tolist() == [6, 12, 18]
    assert episodes["episode_number"].tolist() == [1, 2, 3]
    assert episodes["minutes_since_previous_quiet_episode"].iloc[1:].tolist() == [30.0, 30.0]
    assert math.isnan(float(episodes.iloc[0]["previous_m1c_probability"]))
    assert episodes.iloc[1]["previous_m1c_probability"] == pytest.approx(0.12)
    assert episodes["current_m1c_probability"].tolist() == pytest.approx([0.09, 0.07, 0.05])


def test_protected_boundary_rejects_2026_before_outcome_construction() -> None:
    assert_unprotected_sessions(pd.Series(["2024-01-02", "2025-12-31"]))

    with pytest.raises(ValueError, match="protected"):
        assert_unprotected_sessions(pd.Series(["2025-12-31", "2026-01-01"]))


def test_m0_m1c_tail_overlap_uses_exact_row_identities() -> None:
    overlap = tail_overlap(
        ("a", "b", "c"),
        ("b", "c", "d", "e"),
    )

    assert overlap == {
        "intersection_rows": 2,
        "union_rows": 5,
        "jaccard_overlap": 0.4,
        "m1c_only_rows": 1,
        "m0_only_rows": 2,
    }


def test_matched_random_selection_preserves_size_and_audits_nearest_fallback() -> None:
    population = pd.DataFrame(
        {
            "row_id": ["tail-a", "tail-b", "exact-a", "fallback-b", "other"],
            "period": ["assessment"] * 5,
            "stock": ["AAA", "BBB", "AAA", "BBB", "CCC"],
            "month": ["2025-01", "2025-01", "2025-01", "2025-01", "2025-01"],
            "checkpoint_group": ["early", "late", "early", "late", "middle"],
            "atm_iv_quartile": [1, 4, 1, 3, 2],
        }
    )
    real_tail = population.iloc[:2].copy()

    selected, audit = matched_random_selection(population, real_tail, seed=2026072701)

    assert len(selected) == len(real_tail) == selected["row_id"].nunique()
    assert set(selected["row_id"]) == {"exact-a", "fallback-b"}
    assert audit["exact_match_rows"] == 1
    assert audit["nearest_cell_fallback_rows"] == 1


def test_probability_permutation_stays_within_session_checkpoint_group() -> None:
    frame = pd.DataFrame(
        {
            "session": ["2025-01-02"] * 4 + ["2025-01-03"] * 2,
            "checkpoint_group": ["early", "early", "late", "late", "early", "early"],
            "m1c_probability": [0.01, 0.02, 0.80, 0.90, 0.30, 0.40],
        }
    )

    permuted = permute_probabilities_within_sessions(
        frame,
        probability_column="m1c_probability",
        seed=2026072711,
    )

    for indices in frame.groupby(["session", "checkpoint_group"]).groups.values():
        positions = list(indices)
        assert sorted(permuted.iloc[positions].tolist()) == sorted(
            frame.iloc[positions]["m1c_probability"].tolist()
        )


def test_whole_session_bootstrap_plan_preserves_complete_session_draw_units() -> None:
    plan = whole_session_bootstrap_plan(
        pd.Series(["s1", "s1", "s2", "s3", "s3"]),
        draws=100,
        seed=2026072721,
    )

    assert len(plan) == 100
    assert all(len(draw) == 3 for draw in plan)
    assert all(set(draw).issubset({"s1", "s2", "s3"}) for draw in plan)
    assert plan == whole_session_bootstrap_plan(
        pd.Series(["s1", "s1", "s2", "s3", "s3"]),
        draws=100,
        seed=2026072721,
    )


def _support_frame(months: tuple[str, ...], sessions_per_month: int = 10) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    stocks = tuple(f"S{index:02d}" for index in range(16))
    for month in months:
        for session_index in range(sessions_per_month):
            session = f"{month}-{session_index + 1:02d}"
            for stock in stocks:
                records.append(
                    {
                        "stock": stock,
                        "session": session,
                        "checkpoint": 6 + 2 * (session_index % 15),
                    }
                )
    return pd.DataFrame(records)


def test_checkpoint_and_fresh_episode_support_gates_do_not_relax() -> None:
    assessment = _support_frame(
        tuple(f"2025-{month:02d}" for month in range(1, 9)),
    )
    stress = _support_frame(
        tuple(f"2025-{month:02d}" for month in range(9, 13)),
        sessions_per_month=12,
    )

    assert evaluate_support_gate(
        assessment,
        period="assessment",
        population="checkpoint",
    )["passed"]
    assert evaluate_support_gate(
        stress,
        period="stress",
        population="checkpoint",
    )["passed"]
    assert evaluate_support_gate(
        assessment,
        period="assessment",
        population="fresh_episode",
    )["passed"]

    concentrated = assessment.loc[assessment["stock"].eq("S00")].copy()
    assert not evaluate_support_gate(
        concentrated,
        period="assessment",
        population="checkpoint",
    )["passed"]


def _passing_veto_evidence() -> dict[str, object]:
    period = {
        "remains_below_iv_rate": 0.87,
        "npv_lift": 0.09,
        "mean_iv_residual": -0.001,
        "median_iv_residual": -0.001,
        "bootstrap_80_npv_lift_lower": 0.001,
        "bootstrap_80_mean_residual_upper": -0.0001,
        "m1c_beats_m0_remains_below": True,
        "m1c_beats_m0_mean_residual": True,
        "m1c_beats_m0_1_5_sigma_breach": True,
        "matched_npv_lift_wins": 18,
        "matched_mean_residual_wins": 18,
        "permutation_npv_lift_wins": 9,
        "permutation_mean_residual_wins": 9,
        "negative_residual_months": 6,
        "required_negative_residual_months": 6,
        "support_passed": True,
        "not_dependent_on_one_stock": True,
    }
    stress = dict(period)
    stress["negative_residual_months"] = 3
    stress["required_negative_residual_months"] = 3
    return {
        "assessment": period,
        "stress": stress,
        "score_decile_direction_correct": True,
        "protected_boundary_passed": True,
        "chronology_audit_passed": True,
    }


def test_binding_low_movement_veto_gate_uses_every_frozen_requirement() -> None:
    evidence = _passing_veto_evidence()
    result = evaluate_low_movement_veto_gate(evidence)

    assert result["passed"]
    assert all(result["checks"].values())

    failing = _passing_veto_evidence()
    failing["stress"] = dict(failing["stress"])  # type: ignore[arg-type]
    failing["stress"]["npv_lift"] = 0.079  # type: ignore[index]
    assert not evaluate_low_movement_veto_gate(failing)["passed"]


def test_short_premium_readiness_gate_is_containment_only_and_requires_veto() -> None:
    period = {
        "fresh_1_5_sigma_lower_than_full": True,
        "fresh_1_5_sigma_lower_than_m0": True,
        "fresh_2_sigma_lower_than_full": True,
        "bootstrap_80_1_5_sigma_difference_upper": -0.001,
        "two_sigma_containment_rate": 0.80,
        "support_passed": True,
    }
    evidence: dict[str, object] = {
        "veto_gate_passed": True,
        "assessment": period,
        "stress": dict(period),
        "surprise_movers_not_concentrated": True,
        "thirty_minute_containment_favourable": True,
    }

    assert evaluate_short_premium_readiness_gate(evidence)["passed"]
    evidence["veto_gate_passed"] = False
    assert not evaluate_short_premium_readiness_gate(evidence)["passed"]


def test_overall_decision_prefers_blockers_then_binding_gates() -> None:
    assert (
        choose_overall_decision(
            blocker="blocked_previous_close_iv_failure",
            veto_supported=True,
            readiness_supported=True,
            descriptive_signal=True,
        )
        == "blocked_previous_close_iv_failure"
    )
    assert (
        choose_overall_decision(
            blocker=None,
            veto_supported=True,
            readiness_supported=True,
            descriptive_signal=True,
        )
        == "m1c_low_movement_veto_supported_and_short_premium_recording_prioritised"
    )
    assert (
        choose_overall_decision(
            blocker=None,
            veto_supported=False,
            readiness_supported=False,
            descriptive_signal=True,
        )
        == "m1c_bottom_tail_below_iv_descriptive_only"
    )
    assert (
        choose_overall_decision(
            blocker=None,
            veto_supported=False,
            readiness_supported=False,
            descriptive_signal=False,
        )
        == "m1c_low_movement_veto_not_supported"
    )
