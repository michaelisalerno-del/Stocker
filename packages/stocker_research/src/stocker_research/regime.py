"""Simple historical regime labelling and performance summaries."""

from __future__ import annotations

from math import prod
from typing import Any

import pandas as pd


def label_regimes(frame: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """Label regimes using rolling historical data only."""

    if window <= 1:
        raise ValueError("window must be greater than 1")
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    prior_close = close.shift(1)
    returns = prior_close.pct_change(fill_method=None)

    volatility = returns.rolling(window, min_periods=window).std()
    volatility_median = volatility.expanding(min_periods=window).median()
    short_ma = prior_close.rolling(window, min_periods=window).mean()
    long_ma = prior_close.rolling(window * 2, min_periods=window).mean()
    true_range = (high - low).shift(1).rolling(window, min_periods=window).mean()
    range_median = true_range.expanding(min_periods=window).median()
    drawdown = prior_close / prior_close.cummax() - 1.0

    labels = pd.DataFrame(index=frame.index)
    labels["volatility_regime"] = "unknown"
    labels.loc[volatility > volatility_median, "volatility_regime"] = "high_volatility"
    labels.loc[volatility <= volatility_median, "volatility_regime"] = "low_volatility"
    labels["trend_regime"] = "unknown"
    labels.loc[short_ma > long_ma, "trend_regime"] = "uptrend"
    labels.loc[short_ma <= long_ma, "trend_regime"] = "downtrend"
    labels["range_regime"] = "unknown"
    labels.loc[true_range > range_median, "range_regime"] = "high_range"
    labels.loc[true_range <= range_median, "range_regime"] = "low_range"
    labels["drawdown_regime"] = "unknown"
    labels.loc[drawdown <= -0.1, "drawdown_regime"] = "drawdown"
    labels.loc[drawdown > -0.1, "drawdown_regime"] = "normal"
    return labels


def performance_by_regime(returns: pd.Series, regimes: pd.Series) -> dict[str, dict[str, Any]]:
    """Summarize returns grouped by one regime label series."""

    aligned_returns = returns.reset_index(drop=True)
    aligned_regimes = regimes.reset_index(drop=True)
    output: dict[str, dict[str, Any]] = {}
    for regime, group in aligned_returns.groupby(aligned_regimes):
        clean = pd.Series(pd.to_numeric(group, errors="coerce")).dropna()
        values = [float(value) for value in clean.to_numpy()]
        output[str(regime)] = {
            "rows": len(values),
            "net_return": prod(1.0 + value for value in values) - 1.0 if values else 0.0,
            "mean_return": sum(values) / len(values) if values else 0.0,
            "win_rate": sum(value > 0 for value in values) / len(values) if values else 0.0,
        }
    return output
