"""Mean-reversion-after-large-move test template."""

from __future__ import annotations

from typing import Any

import pandas as pd

from stocker_research.templates.base import StrategyTemplate


class MeanReversionTemplate(StrategyTemplate):
    """Long after a large historical down bar, then hold briefly."""

    name = "mean_reversion_after_large_down_day"

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        threshold = float(params["down_threshold"])
        hold_bars = int(params["hold_bars"])
        if hold_bars <= 0:
            raise ValueError("hold_bars must be positive")
        close = pd.to_numeric(frame["close"], errors="coerce")
        prior_return = close.pct_change().shift(1)
        entries = prior_return <= threshold
        position = pd.Series(0.0, index=frame.index)
        for index, is_entry in enumerate(entries):
            if bool(is_entry):
                end = min(index + hold_bars, len(position))
                position.iloc[index:end] = 1.0
        return position
