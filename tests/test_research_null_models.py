from statistics import median
from typing import Any

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import evaluate_positions
from stocker_research.null_models import run_null_timing_test, run_null_timing_test_for_splits
from stocker_research.templates.base import StrategyTemplate
from stocker_research.walkforward import WalkForwardSplit


class FramePositionTemplate(StrategyTemplate):
    name = "frame_position"

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        return pd.to_numeric(frame["position"], errors="coerce").fillna(0.0)


def _shift(values: list[float], offset: int) -> list[float]:
    if not values:
        return []
    safe_offset = offset % len(values)
    if safe_offset == 0:
        return values
    return values[-safe_offset:] + values[:-safe_offset]


def _manual_split_null_return(
    frame: pd.DataFrame,
    positions: pd.Series,
    splits: list[WalkForwardSplit],
    offset: int,
) -> float:
    returns: list[float] = []
    for split in splits:
        test_frame = frame.iloc[split.test_start : split.test_end].reset_index(drop=True)
        test_positions = positions.iloc[split.test_start : split.test_end].reset_index(drop=True)
        shifted = pd.Series(_shift([float(value) for value in test_positions], offset))
        result = evaluate_positions(
            test_frame,
            shifted,
            cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
            direction="long_only",
        )
        returns.append(result.net_return)
    return float(sum(returns) / len(returns))


def test_null_timing_results_are_deterministic_and_gate_against_p75() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
            "open": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
            "close": [100 + i for i in range(20)],
            "volume": [1000] * 20,
        }
    )
    positions = pd.Series([1.0 if index % 2 == 0 else 0.0 for index in range(len(frame))])

    first = run_null_timing_test(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        hypothesis_id="hypothesis_a",
        symbol="AAPL.US",
        timeframe="1d",
        parameter_set_id="ps_0001",
        selected_net_return=-0.01,
        null_count=7,
    )
    second = run_null_timing_test(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        hypothesis_id="hypothesis_a",
        symbol="AAPL.US",
        timeframe="1d",
        parameter_set_id="ps_0001",
        selected_net_return=-0.01,
        null_count=7,
    )

    assert first == second
    assert first["count"] == 7
    assert "p75_null_net_return" in first
    assert "p90_null_net_return" in first
    assert first["selected_excess_vs_p75_null"] < 0.0
    assert first["null_pass"] is False


def test_split_null_timing_uses_test_windows_not_full_sample() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC"),
            "open": [100, 300, 90, 100, 80, 120, 250, 500, 125, 100, 130, 95],
            "high": [101, 301, 91, 101, 81, 121, 251, 501, 126, 101, 131, 96],
            "low": [99, 299, 89, 99, 79, 119, 249, 499, 124, 99, 129, 94],
            "close": [100, 300, 90, 100, 80, 120, 250, 500, 125, 100, 130, 95],
            "volume": [1000] * 12,
        }
    )
    positions = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    frame["position"] = positions
    splits = [
        WalkForwardSplit(
            split_id="split_001",
            train_start=0,
            train_end=3,
            test_start=3,
            test_end=6,
        ),
        WalkForwardSplit(
            split_id="split_002",
            train_start=6,
            train_end=9,
            test_start=9,
            test_end=12,
        ),
    ]
    template = FramePositionTemplate()

    split_aligned = run_null_timing_test_for_splits(
        frame,
        splits=splits,
        template=template,
        selected_params={},
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        hypothesis_id="hypothesis_a",
        symbol="AAPL.US",
        timeframe="1d",
        parameter_set_id="ps_0001",
        selected_net_return=0.01,
        null_count=3,
        direction="long_only",
    )
    full_sample = run_null_timing_test(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        hypothesis_id="hypothesis_a",
        symbol="AAPL.US",
        timeframe="1d",
        parameter_set_id="ps_0001",
        selected_net_return=0.01,
        null_count=3,
        direction="long_only",
    )
    expected_null_returns = [
        _manual_split_null_return(frame, positions, splits, int(offset))
        for offset in split_aligned["offsets"]
    ]

    assert split_aligned["count"] == 3
    assert split_aligned["window_policy"] == "walk_forward_test_windows_with_indicator_context"
    assert split_aligned["median_null_net_return"] == median(expected_null_returns)
    assert split_aligned["median_null_net_return"] != full_sample["median_null_net_return"]
    assert {
        "count",
        "median_null_net_return",
        "p75_null_net_return",
        "p90_null_net_return",
        "selected_excess_vs_median_null",
        "selected_excess_vs_p75_null",
        "null_pass",
        "offsets",
    }.issubset(split_aligned)
