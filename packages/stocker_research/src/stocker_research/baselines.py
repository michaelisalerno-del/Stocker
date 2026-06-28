"""Minimal baseline research summaries."""

from typing import Any, cast

import pandas as pd

from stocker_research.metrics import annualized_sharpe, max_drawdown


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return cast(pd.DataFrame, frame.to_pandas())
    return pd.DataFrame(frame)


def ohlc_baseline_summary(frame: Any) -> dict[str, float | int]:
    """Return simple OHLC summary metrics without strategy assumptions."""

    data = _to_pandas(frame)
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required OHLC columns: {', '.join(missing)}")
    if data.empty:
        return {
            "rows": 0,
            "close_start": 0.0,
            "close_end": 0.0,
            "total_return": 0.0,
            "mean_return": 0.0,
            "volatility": 0.0,
            "hit_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    close = pd.to_numeric(data["close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError("Close column contains no numeric values")
    returns = close.pct_change().dropna()
    equity = (1.0 + returns).cumprod()
    close_start = float(close.iloc[0])
    close_end = float(close.iloc[-1])
    total_return = close_end / close_start - 1.0 if close_start != 0 else 0.0

    return {
        "rows": int(len(data)),
        "close_start": close_start,
        "close_end": close_end,
        "total_return": float(total_return),
        "mean_return": float(returns.mean()) if not returns.empty else 0.0,
        "volatility": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "hit_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
        "sharpe": annualized_sharpe(returns),
        "max_drawdown": max_drawdown(equity),
    }
