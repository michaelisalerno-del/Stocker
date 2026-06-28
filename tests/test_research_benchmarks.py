import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_research.benchmarks import compare_with_benchmarks
from stocker_research.classification import classify_research_result
from stocker_research.walkforward import WalkForwardSplit


def test_positive_strategy_that_lags_buy_and_hold_fails_benchmark_gate() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC"),
            "open": [100.0, 102.0, 104.0, 108.0, 112.0, 116.0, 120.0, 125.0],
            "high": [101.0, 103.0, 105.0, 109.0, 113.0, 117.0, 121.0, 126.0],
            "low": [99.0, 101.0, 103.0, 107.0, 111.0, 115.0, 119.0, 124.0],
            "close": [100.0, 102.0, 104.0, 108.0, 112.0, 116.0, 120.0, 125.0],
            "volume": [1000] * 8,
        }
    )
    split = WalkForwardSplit(
        split_id="split_001",
        train_start=0,
        train_end=3,
        test_start=3,
        test_end=8,
    )
    selected_result = {
        "parameter_set_id": "ps_0001",
        "test_net_return": 0.01,
        "test_max_drawdown": -0.01,
    }

    comparison = compare_with_benchmarks(
        frame,
        splits=[split],
        selected_result=selected_result,
        cost_model=CostModel(spread_bps=1.0, commission_bps=0.5, slippage_bps=0.5),
        direction="long_only",
    )

    assert comparison["benchmark_results"]["cash"]["net_return"] == 0.0
    assert comparison["benchmark_results"]["buy_and_hold"]["net_return"] > 0.01
    assert comparison["selected_excess_vs_buy_and_hold"] < 0.0
    assert comparison["benchmark_pass"] is False

    classification = classify_research_result(
        net_test_return=0.01,
        gross_test_return=0.02,
        trade_count=50,
        stability_score=0.8,
        profitable_split_pct=1.0,
        max_drawdown=-0.05,
        cost_drag=0.01,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=comparison["benchmark_pass"],
        null_pass=True,
    )

    assert classification.classification == "rejected_no_edge"
    assert "failed_benchmark" in classification.reasons


def test_buy_and_hold_benchmark_remains_long_market_baseline_for_short_direction() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC"),
            "open": [100.0, 95.0, 90.0, 85.0, 80.0, 75.0],
            "high": [101.0, 96.0, 91.0, 86.0, 81.0, 76.0],
            "low": [99.0, 94.0, 89.0, 84.0, 79.0, 74.0],
            "close": [100.0, 95.0, 90.0, 85.0, 80.0, 75.0],
            "volume": [1000] * 6,
        }
    )
    split = WalkForwardSplit(
        split_id="split_001",
        train_start=0,
        train_end=2,
        test_start=2,
        test_end=6,
    )

    comparison = compare_with_benchmarks(
        frame,
        splits=[split],
        selected_result={
            "parameter_set_id": "ps_0001",
            "test_net_return": 0.05,
            "test_max_drawdown": -0.01,
        },
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        direction="short_only",
    )

    assert comparison["strategy_direction"] == "short_only"
    assert comparison["benchmark_policy"] == "long_buy_and_hold_market_baseline"
    assert comparison["benchmark_results"]["buy_and_hold"]["net_return"] < 0.0
