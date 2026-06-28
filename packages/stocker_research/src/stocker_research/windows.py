"""Context-aware walk-forward evaluation windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import DirectionMode, VectorizedBacktestResult, evaluate_positions
from stocker_research.templates import StrategyTemplate

EVALUATION_POLICY_WITH_INDICATOR_CONTEXT = "walk_forward_with_indicator_context"
GRID_CONTEXT_POLICY_WITH_INDICATOR_CONTEXT = "walk_forward_windows_with_indicator_context"
INDICATOR_CONTEXT_POLICY = "historical_indicator_context_before_window_not_scored"
NULL_WINDOW_POLICY_WITH_INDICATOR_CONTEXT = "walk_forward_test_windows_with_indicator_context"


@dataclass(frozen=True)
class EvaluationWindow:
    """One scoring window with historical indicator context attached."""

    eval_frame: pd.DataFrame
    eval_positions: pd.Series
    context_start: int
    context_rows_used: int
    required_lookback_bars: int


@dataclass(frozen=True)
class WindowEvaluationResult:
    """Backtest result plus the context metadata used to produce positions."""

    result: VectorizedBacktestResult
    window: EvaluationWindow


def build_evaluation_window(
    frame: pd.DataFrame,
    template: StrategyTemplate,
    params: dict[str, Any],
    *,
    eval_start: int,
    eval_end: int,
) -> EvaluationWindow:
    """Build positions with historical context, then score only eval rows."""

    if eval_start < 0:
        raise ValueError("eval_start must be non-negative")
    if eval_end < eval_start:
        raise ValueError("eval_end must be greater than or equal to eval_start")
    if eval_end > len(frame):
        raise ValueError("eval_end must not exceed frame length")

    required_lookback_bars = max(0, int(template.required_lookback_bars(params)))
    context_start = max(0, eval_start - required_lookback_bars)
    context_frame = frame.iloc[context_start:eval_end].reset_index(drop=True)
    raw_positions = template.generate_positions(context_frame, params).reset_index(drop=True)
    positions = (
        raw_positions.astype(float)
        .reindex(range(len(context_frame)))
        .fillna(0.0)
        .reset_index(drop=True)
    )
    eval_frame = frame.iloc[eval_start:eval_end].reset_index(drop=True)
    eval_offset = eval_start - context_start
    eval_positions = (
        positions.iloc[eval_offset : eval_offset + len(eval_frame)]
        .reset_index(drop=True)
        .reindex(eval_frame.index)
        .fillna(0.0)
    )
    return EvaluationWindow(
        eval_frame=eval_frame,
        eval_positions=eval_positions,
        context_start=context_start,
        context_rows_used=eval_start - context_start,
        required_lookback_bars=required_lookback_bars,
    )


def evaluate_window_with_context(
    frame: pd.DataFrame,
    template: StrategyTemplate,
    params: dict[str, Any],
    *,
    cost_model: CostModel,
    direction: DirectionMode,
    eval_start: int,
    eval_end: int,
) -> WindowEvaluationResult:
    """Evaluate one scoring window with pre-window indicator context."""

    window = build_evaluation_window(
        frame,
        template,
        params,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    result = evaluate_positions(
        window.eval_frame,
        window.eval_positions,
        cost_model=cost_model,
        direction=direction,
    )
    return WindowEvaluationResult(result=result, window=window)
