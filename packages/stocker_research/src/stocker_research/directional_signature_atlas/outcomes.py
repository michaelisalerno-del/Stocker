"""Exact fixed-terminal economic outcomes for the signature atlas."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def classify_terminal_move(
    gross_long_return_bps: float,
    round_trip_cost_bps: float,
    dead_band_cost_multiple: float = 2.0,
) -> str:
    """Classify one symmetric fixed-terminal move as LONG, SHORT, or NEUTRAL."""

    if not math.isfinite(gross_long_return_bps):
        return "UNAVAILABLE"
    threshold = dead_band_cost_multiple * round_trip_cost_bps
    long_net = gross_long_return_bps - round_trip_cost_bps
    short_net = -gross_long_return_bps - round_trip_cost_bps
    is_long = long_net > 0.0 and gross_long_return_bps > threshold
    is_short = short_net > 0.0 and gross_long_return_bps < -threshold
    if is_long and is_short:
        raise AssertionError("terminal target cannot be both LONG and SHORT")
    if is_long:
        return "LONG"
    if is_short:
        return "SHORT"
    return "NEUTRAL"


def movement_permission(
    predicted_future_range_bps: float,
    round_trip_cost_bps: float,
    multiplier: float = 3.0,
) -> bool:
    """Direction-neutral movement gate frozen independently of payoff."""

    return bool(
        math.isfinite(predicted_future_range_bps)
        and predicted_future_range_bps > multiplier * round_trip_cost_bps
    )


def _unavailable(
    decision_ordinal: int, reason: str, *, first_touch_barrier_bps: float
) -> dict[str, Any]:
    return {
        "decision_ordinal": decision_ordinal,
        "score_status": reason,
        "target": "UNAVAILABLE",
        "first_touch_target": "UNAVAILABLE",
        "first_touch_step": None,
        "first_touch_barrier_bps": first_touch_barrier_bps,
    }


def build_economic_outcome(
    session: pd.DataFrame,
    decision_ordinal: int,
    round_trip_cost_bps: float,
    *,
    horizon_bars: int = 24,
    dead_band_cost_multiple: float = 2.0,
    entry_delay_bars: int = 1,
    first_touch_barrier_bps: float | None = None,
) -> dict[str, Any]:
    """Reconstruct exact next-open entry and same-session fixed terminal."""

    by_ordinal = session.set_index("bar_ordinal", drop=False)
    if entry_delay_bars < 1 or entry_delay_bars >= horizon_bars:
        raise ValueError("entry delay must preserve a positive fixed-terminal horizon")
    threshold = dead_band_cost_multiple * round_trip_cost_bps
    touch_barrier = threshold if first_touch_barrier_bps is None else first_touch_barrier_bps
    if not math.isfinite(touch_barrier) or touch_barrier <= 0.0:
        raise ValueError("first-touch barrier must be finite and positive")
    path_ordinals = list(
        range(decision_ordinal + entry_delay_bars, decision_ordinal + horizon_bars + 1)
    )
    if any(ordinal not in by_ordinal.index for ordinal in path_ordinals):
        return _unavailable(
            decision_ordinal,
            "missing_exact_24_bar_path",
            first_touch_barrier_bps=touch_barrier,
        )
    path = by_ordinal.loc[path_ordinals]
    entry_open = float(path.iloc[0]["open"])
    terminal_close = float(path.iloc[-1]["close"])
    required = path[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if required.isna().any().any() or entry_open <= 0.0 or terminal_close <= 0.0:
        return _unavailable(
            decision_ordinal,
            "invalid_outcome_path",
            first_touch_barrier_bps=touch_barrier,
        )
    gross_long = 10_000.0 * (terminal_close / entry_open - 1.0)
    gross_short = -gross_long
    long_net = gross_long - round_trip_cost_bps
    short_net = gross_short - round_trip_cost_bps
    target = classify_terminal_move(gross_long, round_trip_cost_bps, dead_band_cost_multiple)
    highs = required["high"].to_numpy(float)
    lows = required["low"].to_numpy(float)
    opens = required["open"].to_numpy(float)
    upper = entry_open * (1.0 + touch_barrier / 10_000.0)
    lower = entry_open * (1.0 - touch_barrier / 10_000.0)
    first_touch = "NEITHER"
    first_touch_step: int | None = None
    for step, (bar_open, high, low) in enumerate(zip(opens, highs, lows, strict=True), start=1):
        if bar_open >= upper:
            first_touch = "UPPER_FIRST"
            first_touch_step = step
            break
        if bar_open <= lower:
            first_touch = "LOWER_FIRST"
            first_touch_step = step
            break
        upper_touch = high >= upper
        lower_touch = low <= lower
        if upper_touch and lower_touch:
            first_touch = "SAME_BAR_DUAL_TOUCH"
            first_touch_step = step
            break
        if upper_touch:
            first_touch = "UPPER_FIRST"
            first_touch_step = step
            break
        if lower_touch:
            first_touch = "LOWER_FIRST"
            first_touch_step = step
            break
    return {
        "decision_ordinal": decision_ordinal,
        "score_status": "scored",
        "entry_timestamp": pd.Timestamp(path.iloc[0]["timestamp"]),
        "entry_open": entry_open,
        "terminal_timestamp": pd.Timestamp(path.iloc[-1]["timestamp"]),
        "terminal_close": terminal_close,
        "gross_long_return_bps": gross_long,
        "net_long_return_bps": long_net,
        "gross_short_return_bps": gross_short,
        "net_short_return_bps": short_net,
        "absolute_terminal_move_bps": abs(gross_long),
        "future_high_low_range_bps": 10_000.0
        * (float(np.max(highs)) - float(np.min(lows)))
        / entry_open,
        "mfe_long_bps": 10_000.0 * (float(np.max(highs)) / entry_open - 1.0),
        "mae_long_bps": 10_000.0 * (float(np.min(lows)) / entry_open - 1.0),
        "mfe_short_bps": 10_000.0 * (1.0 - float(np.min(lows)) / entry_open),
        "mae_short_bps": 10_000.0 * (1.0 - float(np.max(highs)) / entry_open),
        "directional_move_threshold_bps": threshold,
        "target": target,
        "first_touch_target": first_touch,
        "first_touch_step": first_touch_step,
        "first_touch_barrier_bps": touch_barrier,
    }
