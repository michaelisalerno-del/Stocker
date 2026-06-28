"""Small deterministic null timing tests for research positions."""

from __future__ import annotations

import hashlib
import math
from statistics import median
from typing import Any

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import DirectionMode, evaluate_positions


def _context_seed(
    *,
    hypothesis_id: str,
    symbol: str,
    timeframe: str,
    parameter_set_id: str,
) -> int:
    context = "|".join([hypothesis_id, symbol.upper(), timeframe, parameter_set_id])
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _deterministic_offsets(length: int, count: int, seed: int) -> list[int]:
    if length <= 1 or count <= 0:
        return []
    max_count = min(count, length - 1)
    offsets: list[int] = []
    step = seed % (length - 1) + 1
    candidate = seed % (length - 1) + 1
    while len(offsets) < max_count:
        offset = ((candidate - 1) % (length - 1)) + 1
        if offset not in offsets:
            offsets.append(offset)
        candidate += step
        if len(offsets) < max_count and len(set(offsets)) == length - 1:
            break
    candidate = 1
    while len(offsets) < max_count:
        if candidate not in offsets:
            offsets.append(candidate)
        candidate += 1
    return offsets


def _circular_shift(values: list[float], offset: int) -> list[float]:
    if not values:
        return []
    shift = offset % len(values)
    if shift == 0:
        return values
    return values[-shift:] + values[:-shift]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def run_null_timing_test(
    frame: pd.DataFrame,
    positions: pd.Series,
    *,
    cost_model: CostModel,
    hypothesis_id: str,
    symbol: str,
    timeframe: str,
    parameter_set_id: str,
    selected_net_return: float,
    null_count: int = 7,
    direction: DirectionMode = "long_only",
) -> dict[str, Any]:
    """Evaluate deterministic circular timing shifts of the selected positions."""

    aligned_positions = (
        positions.reset_index(drop=True).astype(float).reindex(frame.index).fillna(0.0)
    )
    values = [float(value) for value in aligned_positions]
    seed = _context_seed(
        hypothesis_id=hypothesis_id,
        symbol=symbol,
        timeframe=timeframe,
        parameter_set_id=parameter_set_id,
    )
    null_returns: list[float] = []
    for offset in _deterministic_offsets(len(values), null_count, seed):
        shifted = pd.Series(_circular_shift(values, offset))
        result = evaluate_positions(
            frame.reset_index(drop=True),
            shifted,
            cost_model=cost_model,
            direction=direction,
        )
        null_returns.append(result.net_return)

    if not null_returns:
        return {
            "count": 0,
            "median_null_net_return": 0.0,
            "p75_null_net_return": 0.0,
            "p90_null_net_return": 0.0,
            "selected_excess_vs_median_null": float(selected_net_return),
            "selected_excess_vs_p75_null": float(selected_net_return),
            "null_pass": False,
            "offsets": [],
        }

    median_null = float(median(null_returns))
    p75_null = _percentile(null_returns, 0.75)
    p90_null = _percentile(null_returns, 0.90)
    return {
        "count": len(null_returns),
        "median_null_net_return": median_null,
        "p75_null_net_return": p75_null,
        "p90_null_net_return": p90_null,
        "selected_excess_vs_median_null": float(selected_net_return - median_null),
        "selected_excess_vs_p75_null": float(selected_net_return - p75_null),
        "null_pass": bool(selected_net_return > p75_null),
        "offsets": _deterministic_offsets(len(values), null_count, seed),
    }
