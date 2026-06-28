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
