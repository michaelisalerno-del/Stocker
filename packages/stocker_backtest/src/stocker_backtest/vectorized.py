"""Placeholder vectorized backtest interface."""

from dataclasses import dataclass
from typing import Any

from stocker_backtest.costs import CostModel


@dataclass(frozen=True)
class VectorizedBacktestRequest:
    """Inputs expected by a future cost-adjusted vectorized backtest."""

    prices: Any
    signals: Any
    cost_model: CostModel


def run_vectorized_backtest(request: VectorizedBacktestRequest) -> dict[str, Any]:
    """Run a future vectorized backtest.

    This is intentionally unimplemented until a research hypothesis defines the exact
    signal, portfolio construction, and accounting rules.
    """

    raise NotImplementedError("Vectorized backtesting is not implemented yet")
