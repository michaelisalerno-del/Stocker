"""Simple deterministic benchmark comparisons for research results."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import DirectionMode, evaluate_positions
from stocker_research.walkforward import WalkForwardSplit


def _empty_result() -> dict[str, float | int]:
    return {
        "gross_return": 0.0,
        "net_return": 0.0,
        "max_drawdown": 0.0,
        "trade_count": 0,
    }


def _aggregate_results(results: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not results:
        return _empty_result()
    return {
        "gross_return": float(
            sum(float(result["gross_return"]) for result in results) / len(results)
        ),
        "net_return": float(sum(float(result["net_return"]) for result in results) / len(results)),
        "max_drawdown": float(min(float(result["max_drawdown"]) for result in results)),
        "trade_count": int(sum(int(result["trade_count"]) for result in results)),
    }


def _test_windows(frame: pd.DataFrame, splits: list[WalkForwardSplit]) -> list[pd.DataFrame]:
    if not splits:
        return [frame.reset_index(drop=True)]
    return [
        frame.iloc[split.test_start : split.test_end].reset_index(drop=True)
        for split in splits
        if split.test_end > split.test_start
    ]


def _selected_float(selected_result: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = selected_result.get(key)
        if value is not None:
            return float(value)
    return 0.0


def compare_with_benchmarks(
    frame: pd.DataFrame,
    *,
    splits: list[WalkForwardSplit],
    selected_result: dict[str, Any],
    cost_model: CostModel,
    direction: DirectionMode,
) -> dict[str, Any]:
    """Compare a selected result with cash and same-window long buy-and-hold."""

    cash_results: list[dict[str, float | int]] = []
    buy_hold_results: list[dict[str, float | int]] = []
    for window in _test_windows(frame, splits):
        if window.empty:
            continue
        cash = evaluate_positions(
            window,
            pd.Series([0.0] * len(window)),
            cost_model=cost_model,
            direction=direction,
        )
        buy_hold = evaluate_positions(
            window,
            pd.Series([1.0] * len(window)),
            cost_model=cost_model,
            direction="long_only",
        )
        cash_results.append(
            {
                "gross_return": cash.gross_return,
                "net_return": cash.net_return,
                "max_drawdown": cash.max_drawdown,
                "trade_count": cash.number_of_trades,
            }
        )
        buy_hold_results.append(
            {
                "gross_return": buy_hold.gross_return,
                "net_return": buy_hold.net_return,
                "max_drawdown": buy_hold.max_drawdown,
                "trade_count": buy_hold.number_of_trades,
            }
        )

    cash_summary = _aggregate_results(cash_results)
    buy_hold_summary = _aggregate_results(buy_hold_results)
    selected_net = _selected_float(selected_result, "test_net_return", "net_return")
    selected_drawdown = _selected_float(
        selected_result,
        "test_max_drawdown",
        "max_drawdown",
    )
    buy_hold_net = float(buy_hold_summary["net_return"])
    buy_hold_drawdown = float(buy_hold_summary["max_drawdown"])
    selected_excess = selected_net - buy_hold_net
    selected_excess_drawdown = selected_drawdown - buy_hold_drawdown
    return {
        "benchmark_results": {
            "cash": cash_summary,
            "buy_and_hold": buy_hold_summary,
        },
        "selected_excess_vs_buy_and_hold": float(selected_excess),
        "selected_excess_drawdown_vs_buy_and_hold": float(selected_excess_drawdown),
        "benchmark_pass": bool(selected_net > buy_hold_net),
        "strategy_direction": direction,
        "benchmark_policy": "long_buy_and_hold_market_baseline",
    }
