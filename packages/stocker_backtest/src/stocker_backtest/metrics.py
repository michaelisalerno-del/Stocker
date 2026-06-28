"""Small reusable backtest metric helpers."""

from __future__ import annotations

from math import prod

import pandas as pd


def compound_return(returns: pd.Series) -> float:
    """Return compounded return for a simple return series."""

    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float(prod(1.0 + float(value) for value in clean) - 1.0)


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """Return max drawdown from an equity curve."""

    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    drawdown = clean / clean.cummax() - 1.0
    return float(drawdown.min())


def sharpe_like(returns: pd.Series, *, periods_per_year: int = 252) -> float:
    """Return a simple annualized Sharpe-like statistic."""

    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if len(clean) < 2:
        return 0.0
    volatility = clean.std(ddof=1)
    if volatility <= 0:
        return 0.0
    return float(clean.mean() / volatility * periods_per_year**0.5)
