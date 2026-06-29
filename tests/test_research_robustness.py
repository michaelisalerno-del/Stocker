from __future__ import annotations

import math

import pytest

from stocker_research.classification import classify_research_result
from stocker_research.robustness import (
    RobustnessGatePolicy,
    build_intraday_robustness_diagnostics,
    build_partial_pass_row,
    build_robustness_flags,
    summarize_cost_stress,
    summarize_split_returns,
    summarize_trade_returns,
)


def test_summarize_trade_returns_reports_concentration_and_shape() -> None:
    summary = summarize_trade_returns(
        [0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, -0.04, -0.02, -0.01]
    )

    assert summary["number_of_trades"] == 10
    assert summary["average_trade"] == pytest.approx(0.0185)
    assert summary["median_trade"] == pytest.approx(0.015)
    assert summary["win_rate"] == pytest.approx(0.7)
    assert summary["profit_factor"] == pytest.approx(0.255 / 0.07)
    assert summary["top5_winners_share_of_positive_profit"] == pytest.approx(0.24 / 0.255)
    assert summary["top10_winners_share_of_positive_profit"] == pytest.approx(1.0)
    assert summary["top5_losers_share_of_total_loss"] == pytest.approx(1.0)
    assert summary["longest_losing_streak"] == 3


def test_summarize_trade_returns_handles_all_winners() -> None:
    summary = summarize_trade_returns([0.01, 0.02, 0.03])

    assert summary["profit_factor"] == math.inf
    assert summary["win_rate"] == pytest.approx(1.0)
    assert summary["longest_losing_streak"] == 0


def test_summarize_split_returns_flags_one_split_dominance() -> None:
    summary = summarize_split_returns(
        [
            {"split_id": "split_001", "test_net_return": -0.02},
            {"split_id": "split_002", "test_net_return": 0.01},
            {"split_id": "split_003", "test_net_return": 0.04},
            {"split_id": "split_004", "test_net_return": 0.005},
        ]
    )

    assert summary["split_count"] == 4
    assert summary["profitable_split_pct"] == pytest.approx(0.75)
    assert summary["best_split_id"] == "split_003"
    assert summary["worst_split_id"] == "split_001"
    assert summary["top_positive_split_share"] == pytest.approx(0.04 / 0.055)


def test_summarize_cost_stress_finds_first_failure_multiplier() -> None:
    summary = summarize_cost_stress(
        [
            {
                "cost_multiplier": 1.0,
                "net_return": 0.01,
                "classification_under_existing_gates": "candidate_intraday_test",
            },
            {
                "cost_multiplier": 1.5,
                "net_return": -0.001,
                "classification_under_existing_gates": "rejected_costs_kill_edge",
            },
            {
                "cost_multiplier": 2.0,
                "net_return": -0.01,
                "classification_under_existing_gates": "rejected_costs_kill_edge",
            },
        ]
    )

    assert summary["first_nonpositive_net_multiplier"] == pytest.approx(1.5)
    assert summary["first_costs_kill_multiplier"] == pytest.approx(1.5)
    assert summary["survives_1_5x_costs"] is False


def test_partial_pass_row_and_flags_capture_fragility() -> None:
    row = build_partial_pass_row(
        symbol="CRM",
        benchmark_pass=True,
        null_pass=True,
        net_return=0.004,
        cost_stress_survives_1_5x=False,
        median_trade=-0.0007,
        top_positive_split_share=0.91,
        top_winner_share=0.51,
        stability_score=0.98,
        train_selection_succeeded=True,
        session_flat_compliant=True,
    )

    assert row["positive_net_return"] is True
    assert row["median_trade_positive"] is False
    assert row["top_split_concentration_ok"] is False
    assert row["top_trade_concentration_ok"] is False
    assert row["stability_score_at_least_threshold"] is True

    flags = build_robustness_flags(row, trade_count=68)

    assert "fragile_costs" in flags
    assert "split_concentrated" in flags
    assert "trade_concentrated" in flags
    assert "negative_median_trade" in flags
    assert "stable_but_low_return" in flags
    assert "robust_partial_pass" not in flags


def test_robust_partial_pass_requires_all_core_checks() -> None:
    row = build_partial_pass_row(
        symbol="TEST",
        benchmark_pass=True,
        null_pass=True,
        net_return=0.03,
        cost_stress_survives_1_5x=True,
        median_trade=0.001,
        top_positive_split_share=0.4,
        top_winner_share=0.35,
        stability_score=0.7,
        train_selection_succeeded=True,
        session_flat_compliant=True,
    )

    assert build_robustness_flags(row, trade_count=50) == ["robust_partial_pass"]


def _passing_cost_stress_rows() -> list[dict[str, float]]:
    return [
        {"cost_multiplier": 1.0, "net_return": 0.04},
        {"cost_multiplier": 1.5, "net_return": 0.03},
        {"cost_multiplier": 2.0, "net_return": 0.02},
        {"cost_multiplier": 3.0, "net_return": 0.01},
    ]


def _passing_split_rows() -> list[dict[str, float | str]]:
    return [
        {"split_id": "split_001", "test_net_return": 0.015},
        {"split_id": "split_002", "test_net_return": 0.014},
        {"split_id": "split_003", "test_net_return": 0.013},
    ]


def _passing_trade_returns() -> list[float]:
    return [0.010] * 10 + [-0.004, -0.004]


def _robust_candidate_result(
    diagnostics: dict[str, object],
    *,
    holding_policy_classification: str = "candidate_intraday_test",
) -> str:
    result = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.10,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification=holding_policy_classification,
        holding_policy_reasons=["session_flat_compliant"],
        intraday_robustness_diagnostics=diagnostics,
    )
    return result.classification


def test_intraday_robustness_diagnostics_use_required_report_shape() -> None:
    diagnostics = build_intraday_robustness_diagnostics(
        cost_stress_rows=_passing_cost_stress_rows(),
        trade_returns=_passing_trade_returns(),
        split_rows=_passing_split_rows(),
        policy=RobustnessGatePolicy(),
    )

    assert diagnostics["cost_stress"] == {
        "base_net_return": pytest.approx(0.04),
        "net_return_1_5x_costs": pytest.approx(0.03),
        "net_return_2x_costs": pytest.approx(0.02),
        "net_return_3x_costs": pytest.approx(0.01),
        "survives_1_5x_costs": True,
        "first_non_positive_cost_multiplier": None,
    }
    assert diagnostics["trade_concentration"]["reconstructed_round_trips"] == len(
        _passing_trade_returns()
    )
    assert diagnostics["trade_concentration"]["median_trade_return"] > 0
    assert diagnostics["trade_concentration"]["profit_factor"] >= 1.10
    assert diagnostics["trade_concentration"]["top_5_winners_profit_share"] <= 0.50
    assert diagnostics["split_concentration"]["top_positive_split_share"] <= 0.50
    assert diagnostics["robustness_flags"] == []


def test_clean_intraday_candidate_that_passes_robustness_remains_candidate() -> None:
    diagnostics = build_intraday_robustness_diagnostics(
        cost_stress_rows=_passing_cost_stress_rows(),
        trade_returns=_passing_trade_returns(),
        split_rows=_passing_split_rows(),
        policy=RobustnessGatePolicy(),
    )

    assert _robust_candidate_result(diagnostics) == "candidate_intraday_test"


def test_crm_like_candidate_failing_cost_stress_is_downgraded_not_rejected() -> None:
    diagnostics = build_intraday_robustness_diagnostics(
        cost_stress_rows=[
            {"cost_multiplier": 1.0, "net_return": 0.004},
            {"cost_multiplier": 1.5, "net_return": -0.002},
            {"cost_multiplier": 2.0, "net_return": -0.010},
            {"cost_multiplier": 3.0, "net_return": -0.025},
        ],
        trade_returns=_passing_trade_returns(),
        split_rows=_passing_split_rows(),
        policy=RobustnessGatePolicy(),
    )

    result = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.10,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="candidate_intraday_test",
        holding_policy_reasons=["session_flat_compliant"],
        intraday_robustness_diagnostics=diagnostics,
    )

    assert result.classification == "interesting_intraday_needs_more_tests"
    assert "failed_cost_stress" in result.reasons
    assert "session_flat_compliant" in result.reasons


@pytest.mark.parametrize(
    ("trade_returns", "expected_reason"),
    [
        ([0.010, 0.008, -0.001, -0.002, -0.003, -0.004, -0.005], "negative_median_trade"),
        ([0.010] * 10 + [-0.100], "weak_profit_factor"),
        ([0.040, 0.030, 0.020, 0.010, 0.005, 0.001, -0.010], "trade_concentrated"),
    ],
)
def test_trade_robustness_failures_prevent_intraday_candidate(
    trade_returns: list[float],
    expected_reason: str,
) -> None:
    diagnostics = build_intraday_robustness_diagnostics(
        cost_stress_rows=_passing_cost_stress_rows(),
        trade_returns=trade_returns,
        split_rows=_passing_split_rows(),
        policy=RobustnessGatePolicy(),
    )
    result = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.10,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="candidate_intraday_test",
        holding_policy_reasons=["session_flat_compliant"],
        intraday_robustness_diagnostics=diagnostics,
    )

    assert result.classification == "interesting_intraday_needs_more_tests"
    assert expected_reason in result.reasons


def test_split_concentration_prevents_intraday_candidate() -> None:
    diagnostics = build_intraday_robustness_diagnostics(
        cost_stress_rows=_passing_cost_stress_rows(),
        trade_returns=_passing_trade_returns(),
        split_rows=[
            {"split_id": "split_001", "test_net_return": 0.050},
            {"split_id": "split_002", "test_net_return": 0.010},
            {"split_id": "split_003", "test_net_return": 0.005},
        ],
        policy=RobustnessGatePolicy(),
    )

    result = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.10,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="candidate_intraday_test",
        holding_policy_reasons=["session_flat_compliant"],
        intraday_robustness_diagnostics=diagnostics,
    )

    assert result.classification == "interesting_intraday_needs_more_tests"
    assert "split_concentrated" in result.reasons


def test_existing_rejections_are_not_promoted_by_robustness() -> None:
    diagnostics = build_intraday_robustness_diagnostics(
        cost_stress_rows=_passing_cost_stress_rows(),
        trade_returns=_passing_trade_returns(),
        split_rows=_passing_split_rows(),
        policy=RobustnessGatePolicy(),
    )

    no_edge = classify_research_result(
        net_test_return=-0.01,
        gross_test_return=0.02,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.03,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="candidate_intraday_test",
        holding_policy_reasons=["session_flat_compliant"],
        intraday_robustness_diagnostics=diagnostics,
    )
    too_few = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.10,
        trade_count=5,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="candidate_intraday_test",
        holding_policy_reasons=["session_flat_compliant"],
        intraday_robustness_diagnostics=diagnostics,
    )

    assert no_edge.classification == "rejected_costs_kill_edge"
    assert too_few.classification == "rejected_too_few_trades"


def test_daily_and_swing_candidate_paths_do_not_require_intraday_robustness() -> None:
    paper = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.10,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
    )
    swing = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.10,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="candidate_swing_exceptional",
        holding_policy_reasons=["swing_exceptional_evidence"],
    )

    assert paper.classification == "candidate_paper_test"
    assert swing.classification == "candidate_swing_exceptional"


def test_missing_trade_reconstruction_data_prevents_intraday_candidate() -> None:
    diagnostics = build_intraday_robustness_diagnostics(
        cost_stress_rows=_passing_cost_stress_rows(),
        trade_returns=None,
        split_rows=_passing_split_rows(),
        policy=RobustnessGatePolicy(),
    )
    result = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.10,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.10,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="candidate_intraday_test",
        holding_policy_reasons=["session_flat_compliant"],
        intraday_robustness_diagnostics=diagnostics,
    )

    assert result.classification == "interesting_intraday_needs_more_tests"
    assert "missing_trade_reconstruction" in result.reasons
