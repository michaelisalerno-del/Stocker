"""Volatility breakout test template."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stocker_research.templates.base import StrategyTemplate


class VolatilityBreakoutTemplate(StrategyTemplate):
    """Long when historical close breaks above a prior range-adjusted high."""

    name = "volatility_breakout"

    def required_lookback_bars(self, params: dict[str, Any]) -> int:
        return int(params["lookback"]) + 1

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        lookback = int(params["lookback"])
        multiplier = float(params["range_multiplier"])
        if lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        prior_high = high.shift(1).rolling(lookback, min_periods=lookback).max()
        prior_range = (high - low).shift(1).rolling(lookback, min_periods=lookback).mean()
        historical_close = close.shift(1)
        return (
            (historical_close > (prior_high + multiplier * prior_range)).astype(float).fillna(0.0)
        )
