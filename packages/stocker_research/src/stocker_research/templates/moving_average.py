"""Moving-average momentum test template."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stocker_research.templates.base import StrategyTemplate


class MovingAverageMomentumTemplate(StrategyTemplate):
    """Long when fast historical average is above slow historical average."""

    name = "moving_average_momentum"

    def required_lookback_bars(self, params: dict[str, Any]) -> int:
        fast = int(params["fast_window"])
        slow = int(params["slow_window"])
        return max(fast, slow) + 1

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        fast = int(params["fast_window"])
        slow = int(params["slow_window"])
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("Require 0 < fast_window < slow_window")
        close = pd.to_numeric(frame["close"], errors="coerce")
        historical_close = close.shift(1)
        fast_ma = historical_close.rolling(fast, min_periods=fast).mean()
        slow_ma = historical_close.rolling(slow, min_periods=slow).mean()
        return (fast_ma > slow_ma).astype(float).fillna(0.0)
