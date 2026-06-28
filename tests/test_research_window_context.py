from statistics import median
from typing import Any

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import evaluate_positions
from stocker_research.experiments import _run_grid
from stocker_research.hypothesis import Hypothesis
from stocker_research.null_models import run_null_timing_test_for_splits
from stocker_research.parameters import ParameterSet
from stocker_research.templates import MovingAverageMomentumTemplate
from stocker_research.templates.base import StrategyTemplate
from stocker_research.walkforward import WalkForwardSplit
from stocker_research.windows import (
    NULL_WINDOW_POLICY_WITH_INDICATOR_CONTEXT,
    build_evaluation_window,
    evaluate_window_with_context,
)


def _warmup_frame(rows: int = 230) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B", tz="UTC")
    close = pd.Series([100.0 + index * 0.5 for index in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000 + index for index in range(rows)],
        }
    )


def _hypothesis() -> Hypothesis:
    return Hypothesis.model_validate(
        {
            "id": "window_context_test_hypothesis",
            "name": "Window Context Test Hypothesis",
            "description": "Synthetic hypothesis for walk-forward context tests.",
            "hypothesis_version": 1,
            "market_universe": "unit_test",
            "instrument_type": "stock",
            "timeframe": "1d",
            "data_source": "manual",
            "template": "moving_average_momentum",
            "direction": "long_only",
            "entry_logic": "Fast moving average above slow moving average.",
            "exit_logic": "Flat when fast moving average is no longer above slow average.",
            "holding_period": "Signal persistence only.",
            "expected_edge_reason": "Unit test fixture.",
            "invalidation_rules": ["Unit test invalidation."],
            "minimum_evidence": {
                "min_trades": 0,
                "min_profitable_split_pct": 0.0,
                "min_stability_score": 0.0,
            },
            "parameter_space": {
                "fast_window": [5],
                "holding_period": [1],
                "slow_window": [150],
            },
            "maximum_parameter_sets": 1,
            "costs": {"spread_bps": 0.0, "commission_bps": 0.0, "slippage_bps": 0.0},
            "risk": {"max_drawdown": 0.25},
            "walkforward": {
                "mode": "rolling",
                "train_bars": 140,
                "test_bars": 25,
                "embargo_bars": 0,
                "step_bars": 25,
                "minimum_rows": 165,
            },
            "created_at": "2026-06-28T00:00:00Z",
        }
    )


def _shift(values: list[float], offset: int) -> list[float]:
    safe_offset = offset % len(values) if values else 0
    if not values or safe_offset == 0:
        return values
    return values[-safe_offset:] + values[:-safe_offset]


def test_walk_forward_grid_uses_indicator_context_for_rolling_warmup() -> None:
    frame = _warmup_frame()
    template = MovingAverageMomentumTemplate()
    params = {"fast_window": 5, "slow_window": 150}
    split = WalkForwardSplit(
        split_id="split_001",
        train_start=20,
        train_end=160,
        test_start=170,
        test_end=195,
    )

    isolated_positions = template.generate_positions(
        frame.iloc[split.test_start : split.test_end].reset_index(drop=True),
        params,
    )
    rows = _run_grid(
        frame,
        [split],
        [ParameterSet(parameter_set_id="ps_0001", params=params)],
        _hypothesis(),
    )

    assert float(isolated_positions.sum()) == 0.0
    assert rows[0]["required_lookback_bars"] == 151
    assert rows[0]["test_context_rows_used"] == 151
    assert rows[0]["context_policy"] == "walk_forward_windows_with_indicator_context"
    assert rows[0]["test_trade_count"] >= 1
    assert rows[0]["test_net_return"] > 0.0


def test_evaluation_window_uses_no_future_rows() -> None:
    frame = _warmup_frame()
    template = MovingAverageMomentumTemplate()
    params = {"fast_window": 5, "slow_window": 150}

    original_window = build_evaluation_window(
        frame,
        template,
        params,
        eval_start=170,
        eval_end=195,
    )
    original_result = evaluate_window_with_context(
        frame,
        template,
        params,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        direction="long_only",
        eval_start=170,
        eval_end=195,
    ).result

    mutated = frame.copy()
    mutated.loc[195:, "close"] = mutated.loc[195:, "close"] * 1000.0
    mutated.loc[195:, "high"] = mutated.loc[195:, "high"] * 1000.0
    mutated.loc[195:, "low"] = mutated.loc[195:, "low"] * 1000.0
    mutated_window = build_evaluation_window(
        mutated,
        template,
        params,
        eval_start=170,
        eval_end=195,
    )
    mutated_result = evaluate_window_with_context(
        mutated,
        template,
        params,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        direction="long_only",
        eval_start=170,
        eval_end=195,
    ).result

    assert original_window.eval_positions.equals(mutated_window.eval_positions)
    assert original_result.to_dict() == mutated_result.to_dict()


class FirstContextBarTemplate(StrategyTemplate):
    name = "first_context_bar"

    def required_lookback_bars(self, params: dict[str, Any]) -> int:
        return int(params["lookback"])

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        positions = pd.Series([0.0] * len(frame))
        lookback = int(params["lookback"])
        if len(positions) > lookback:
            positions.iloc[lookback] = 1.0
        return positions


def test_null_timing_uses_same_indicator_context_policy() -> None:
    frame = _warmup_frame(rows=40)
    template = FirstContextBarTemplate()
    params = {"lookback": 5}
    splits = [
        WalkForwardSplit(
            split_id="split_001",
            train_start=0,
            train_end=10,
            test_start=10,
            test_end=14,
        )
    ]

    isolated_positions = template.generate_positions(
        frame.iloc[splits[0].test_start : splits[0].test_end].reset_index(drop=True),
        params,
    )
    context_window = build_evaluation_window(
        frame,
        template,
        params,
        eval_start=splits[0].test_start,
        eval_end=splits[0].test_end,
    )
    result = run_null_timing_test_for_splits(
        frame,
        splits=splits,
        template=template,
        selected_params=params,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        hypothesis_id="window_context_test_hypothesis",
        symbol="AAPL.US",
        timeframe="1d",
        parameter_set_id="ps_0001",
        selected_net_return=0.0,
        null_count=3,
        direction="long_only",
    )
    expected_returns = []
    for offset in result["offsets"]:
        shifted = pd.Series(
            _shift([float(value) for value in context_window.eval_positions], int(offset))
        )
        expected_returns.append(
            evaluate_positions(
                context_window.eval_frame,
                shifted,
                cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
                direction="long_only",
            ).net_return
        )

    assert float(isolated_positions.sum()) == 0.0
    assert float(context_window.eval_positions.sum()) == 1.0
    assert result["window_policy"] == NULL_WINDOW_POLICY_WITH_INDICATOR_CONTEXT
    assert result["median_null_net_return"] == median(expected_returns)
