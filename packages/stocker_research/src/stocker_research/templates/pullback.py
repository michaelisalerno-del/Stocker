"""Pullback-in-uptrend research test template."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stocker_research.templates.base import StrategyTemplate


class PullbackInUptrendTemplate(StrategyTemplate):
    """Long after a historical pullback while price remains in an uptrend."""

    name = "pullback_in_uptrend"

    def required_lookback_bars(self, params: dict[str, Any]) -> int:
        trend_window = int(params["trend_window"])
        holding_period = int(params["holding_period"])
        return trend_window + holding_period - 1

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        trend_window = int(params["trend_window"])
        threshold = float(params["pullback_threshold"])
        holding_period = int(params["holding_period"])
        if trend_window <= 1 or holding_period <= 0:
            raise ValueError("trend_window must be > 1 and holding_period must be positive")
        if threshold >= 0:
            raise ValueError("pullback_threshold must be negative")

        close = pd.to_numeric(frame["close"], errors="coerce")
        historical_close = close.shift(1)
        trend = (
            historical_close
            > historical_close.rolling(trend_window, min_periods=trend_window).mean()
        )
        pullback = historical_close.pct_change() <= threshold
        entries = (trend & pullback).fillna(False)
        positions = pd.Series(0.0, index=frame.index)
        for index, should_enter in enumerate(entries):
            if not should_enter:
                continue
            end = min(index + holding_period, len(positions))
            positions.iloc[index:end] = 1.0
        return positions.reset_index(drop=True)
