"""Research metrics for disproving ideas quickly."""

import math
from collections.abc import Iterable

import pandas as pd


def annualized_sharpe(returns: Iterable[float], periods_per_year: int = 252) -> float:
    """Compute a simple annualized Sharpe ratio with zero risk-free rate."""

    series = pd.Series(list(returns), dtype="float64").dropna()
    if series.empty:
        return 0.0
    volatility = float(series.std(ddof=1))
    if volatility == 0.0 or math.isnan(volatility):
        return 0.0
    return float(series.mean() / volatility * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: Iterable[float]) -> float:
    """Return max drawdown as a negative fraction."""

    series = pd.Series(list(equity_curve), dtype="float64").dropna()
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    return float(drawdown.min())
