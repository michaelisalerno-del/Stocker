from __future__ import annotations

import math

import pytest

from stocker_research.robustness import (
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
