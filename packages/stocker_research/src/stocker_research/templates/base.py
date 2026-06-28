"""Minimal deterministic strategy template interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class StrategyTemplate(ABC):
    """Strategy template that emits target positions, not orders."""

    name: str

    @abstractmethod
    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Return deterministic target positions aligned to the input rows."""

    def generate_signals(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Return a standard signal DataFrame for reports and leakage checks."""

        positions = self.generate_positions(frame, params).reset_index(drop=True).astype(float)
        signal = positions.diff().fillna(positions).clip(lower=-1.0, upper=1.0)
        output = pd.DataFrame(
            {
                "timestamp": frame["timestamp"].reset_index(drop=True)
                if "timestamp" in frame
                else pd.Series(range(len(frame))),
                "signal": signal,
                "target_position": positions,
                "entry": signal > 0,
                "exit": signal < 0,
                "template_name": self.name,
                "parameter_set_id": str(params.get("parameter_set_id", "unknown")),
            }
        )
        return output
