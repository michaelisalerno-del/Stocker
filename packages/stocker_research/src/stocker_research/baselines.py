"""Minimal baseline research summaries and reports."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_data.storage import DataLayer, DatasetKey, load_dataset
from stocker_research.metrics import annualized_sharpe, max_drawdown


@dataclass(frozen=True)
class BaselineReportResult:
    """Paths produced by a baseline report."""

    markdown_path: Path
    json_path: Path


@dataclass(frozen=True)
class BaselineMetrics:
    """Result metrics for one deliberately simple baseline."""

    name: str
    gross_total_return: float
    net_total_return: float
    annualized_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    number_of_trades: int
    exposure: float
    estimated_costs: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return JSON-serializable metrics."""

        return {
            "name": self.name,
            "gross_total_return": self.gross_total_return,
            "net_total_return": self.net_total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "number_of_trades": self.number_of_trades,
            "exposure": self.exposure,
            "estimated_costs": self.estimated_costs,
        }


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


def _periods_per_year(timeframe: str) -> int | None:
    normalized = timeframe.lower()
    if normalized in {"1d", "d", "day", "daily"}:
        return 252
    if normalized == "1h":
        return 252 * 6
    if normalized == "1m":
        return 252 * 390
    return None


def _annualized_return(total_return: float, periods: int, timeframe: str) -> float:
    periods_per_year = _periods_per_year(timeframe)
    if periods_per_year is None or periods <= 0:
        return 0.0
    return float((1.0 + total_return) ** (periods_per_year / periods) - 1.0)


def _strategy_metrics(
    *,
    name: str,
    close: pd.Series,
    position: pd.Series,
    timeframe: str,
    cost_model: CostModel,
) -> BaselineMetrics:
    returns = close.pct_change().fillna(0.0)
    aligned_position = position.astype(float).reindex(close.index).fillna(0.0)
    gross_returns = aligned_position.shift(1).fillna(0.0) * returns
    position_changes = aligned_position.diff().abs().fillna(aligned_position.abs())
    cost_returns = position_changes * (cost_model.one_way_bps() / 10_000)
    net_returns = gross_returns - cost_returns

    gross_equity = (1.0 + gross_returns).cumprod()
    net_equity = (1.0 + net_returns).cumprod()
    gross_total = float(gross_equity.iloc[-1] - 1.0) if not gross_equity.empty else 0.0
    net_total = float(net_equity.iloc[-1] - 1.0) if not net_equity.empty else 0.0
    trade_returns = net_returns[position_changes > 0]
    return BaselineMetrics(
        name=name,
        gross_total_return=gross_total,
        net_total_return=net_total,
        annualized_return=_annualized_return(net_total, len(net_returns), timeframe),
        volatility=float(net_returns.std(ddof=1)) if len(net_returns) > 1 else 0.0,
        sharpe=annualized_sharpe(net_returns, _periods_per_year(timeframe) or 252),
        max_drawdown=max_drawdown(net_equity),
        win_rate=float((trade_returns > 0).mean()) if not trade_returns.empty else 0.0,
        number_of_trades=int(position_changes.sum()),
        exposure=float(aligned_position.abs().mean()) if not aligned_position.empty else 0.0,
        estimated_costs=float(cost_returns.sum()),
    )


def run_baselines(
    frame: Any,
    *,
    timeframe: str,
    cost_model: CostModel,
    random_seed: int = 42,
) -> list[BaselineMetrics]:
    """Run simple non-optimized baseline policies."""

    data = _to_pandas(frame).sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(data["close"], errors="coerce").dropna().reset_index(drop=True)
    if close.empty:
        raise ValueError("Close column contains no numeric values")

    rng = random.Random(random_seed)
    random_position = pd.Series([float(rng.choice([0, 1])) for _ in range(len(close))])
    sma = close.rolling(3, min_periods=1).mean()
    baselines = {
        "buy_and_hold": pd.Series([1.0] * len(close)),
        "always_flat": pd.Series([0.0] * len(close)),
        "random_entry_exit": random_position,
        "sma_momentum": (close > sma).astype(float),
        "mean_reversion": (close < sma).astype(float),
    }
    return [
        _strategy_metrics(
            name=name,
            close=close,
            position=position,
            timeframe=timeframe,
            cost_model=cost_model,
        )
        for name, position in baselines.items()
    ]


def _baseline_markdown(payload: dict[str, Any]) -> str:
    rows = "\n".join(
        (
            "| {name} | {net_total_return:.6f} | {gross_total_return:.6f} | "
            "{number_of_trades} | {exposure:.3f} |"
        ).format(**result)
        for result in payload["results"]
    )
    return f"""# Baseline Report: {payload["symbol"]} {payload["timeframe"]}

This report contains deliberately simple baselines only. It is not a strategy
optimization report.

| Baseline | Net Return | Gross Return | Trades | Exposure |
| --- | ---: | ---: | ---: | ---: |
{rows}

## Cost Model

```json
{json.dumps(payload["cost_model"], indent=2)}
```
"""


def create_baseline_report(
    *,
    data_dir: str | Path = "data",
    symbol: str,
    timeframe: str,
    source: str = "manual",
    instrument_type: str = "stock",
    layer: DataLayer = "processed",
    spread_bps: float = 0.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    random_seed: int = 42,
) -> BaselineReportResult:
    """Create Markdown and JSON baseline reports for one dataset."""

    key = DatasetKey(
        source=source,
        instrument_type=instrument_type,
        symbol=symbol.upper(),
        timeframe=timeframe,
    )
    frame = load_dataset(key, data_dir=data_dir, layer=layer)
    cost_model = CostModel(
        spread_bps=spread_bps,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    results = run_baselines(
        frame,
        timeframe=timeframe,
        cost_model=cost_model,
        random_seed=random_seed,
    )
    payload: dict[str, Any] = {
        "symbol": key.symbol,
        "timeframe": key.timeframe,
        "source": key.source,
        "instrument_type": key.instrument_type,
        "cost_model": {
            "spread_bps": spread_bps,
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
            "round_trip_bps": cost_model.round_trip_bps(),
        },
        "results": [result.to_dict() for result in results],
    }

    output_dir = Path(data_dir).expanduser() / "reports" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{key.symbol}_{key.timeframe}_baseline"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_baseline_markdown(payload), encoding="utf-8")
    return BaselineReportResult(markdown_path=markdown_path, json_path=json_path)
