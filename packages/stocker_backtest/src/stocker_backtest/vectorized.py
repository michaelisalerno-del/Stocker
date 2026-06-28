"""Transparent cost-adjusted vectorized evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

from stocker_backtest.costs import CostModel

PositionSizingMode = Literal["unit_notional"]
DirectionMode = Literal["long_only", "short_only", "long_short"]


@dataclass(frozen=True)
class Trade:
    """One simplified position-change event."""

    index: int
    timestamp: str | None
    position: float
    notional_change: float
    cost: float
    bar_return: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable trade data."""

        return asdict(self)


@dataclass(frozen=True)
class VectorizedBacktestResult:
    """Structured result from vectorized evaluation."""

    gross_return: float
    net_return: float
    total_costs: float
    number_of_trades: int
    exposure: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    volatility: float
    sharpe: float
    average_trade: float
    best_trade: float
    worst_trade: float
    equity_curve: list[float]
    drawdown_curve: list[float]
    trades: list[Trade]
    net_returns: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable result data."""

        return {
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "total_costs": self.total_costs,
            "number_of_trades": self.number_of_trades,
            "exposure": self.exposure,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "average_trade": self.average_trade,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "equity_curve": self.equity_curve,
            "drawdown_curve": self.drawdown_curve,
            "trades": [trade.to_dict() for trade in self.trades],
            "net_returns": self.net_returns,
        }


@dataclass(frozen=True)
class VectorizedBacktestRequest:
    """Inputs expected by a cost-adjusted vectorized backtest."""

    prices: Any
    signals: Any
    cost_model: CostModel
    initial_capital: float = 100_000.0


def _timestamp_at(frame: pd.DataFrame, index: int) -> str | None:
    if "timestamp" not in frame:
        return None
    return str(frame.iloc[index]["timestamp"])


def _max_drawdown(equity: pd.Series) -> tuple[float, list[float]]:
    if equity.empty:
        return 0.0, []
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min()), [float(value) for value in drawdown]


def evaluate_positions(
    frame: pd.DataFrame,
    positions: pd.Series,
    *,
    cost_model: CostModel,
    initial_capital: float = 100_000.0,
    position_sizing: PositionSizingMode = "unit_notional",
    direction: DirectionMode = "long_only",
) -> VectorizedBacktestResult:
    """Evaluate target positions with simple close-to-close accounting."""

    if position_sizing != "unit_notional":
        raise NotImplementedError("Only unit_notional sizing is implemented")
    close = pd.to_numeric(frame["close"], errors="coerce").reset_index(drop=True)
    raw_position = positions.reset_index(drop=True).astype(float).reindex(close.index).fillna(0.0)
    if direction == "long_only":
        position = raw_position.clip(lower=0.0, upper=1.0)
    elif direction == "short_only":
        position = raw_position.clip(lower=-1.0, upper=0.0)
    else:
        position = raw_position.clip(lower=-1.0, upper=1.0)

    returns = close.pct_change().fillna(0.0)
    gross_returns = position.shift(1).fillna(0.0) * returns
    position_change = position.diff().abs().fillna(position.abs())
    cost_return = position_change * cost_model.one_way_bps() / 10_000
    net_returns = gross_returns - cost_return
    gross_equity = (1.0 + gross_returns).cumprod() * initial_capital
    net_equity = (1.0 + net_returns).cumprod() * initial_capital
    max_dd, drawdown_curve = _max_drawdown(net_equity)

    trades: list[Trade] = []
    trade_returns = net_returns[position_change > 0]
    flat_position_change = position_change.reset_index(drop=True)
    flat_position = position.reset_index(drop=True)
    flat_net_returns = net_returns.reset_index(drop=True)
    flat_frame = frame.reset_index(drop=True)
    for index, change in enumerate(flat_position_change):
        if change <= 0:
            continue
        cost = float(initial_capital * change * cost_model.one_way_bps() / 10_000)
        trades.append(
            Trade(
                index=index,
                timestamp=_timestamp_at(flat_frame, index),
                position=float(flat_position.iloc[index]),
                notional_change=float(change),
                cost=cost,
                bar_return=float(flat_net_returns.iloc[index]),
            )
        )

    wins = gross_returns[gross_returns > 0].sum()
    losses = gross_returns[gross_returns < 0].sum()
    profit_factor = float(wins / abs(losses)) if losses < 0 else float("inf") if wins > 0 else 0.0
    volatility = float(net_returns.std(ddof=1)) if len(net_returns) > 1 else 0.0
    sharpe = float(net_returns.mean() / volatility * (252**0.5)) if volatility > 0 else 0.0
    return VectorizedBacktestResult(
        gross_return=float(gross_equity.iloc[-1] / initial_capital - 1.0),
        net_return=float(net_equity.iloc[-1] / initial_capital - 1.0),
        total_costs=float(initial_capital * cost_return.sum()),
        number_of_trades=int((position_change > 0).sum()),
        exposure=float(position.abs().mean()) if not position.empty else 0.0,
        win_rate=float((trade_returns > 0).mean()) if not trade_returns.empty else 0.0,
        profit_factor=profit_factor,
        max_drawdown=max_dd,
        volatility=volatility,
        sharpe=sharpe,
        average_trade=float(trade_returns.mean()) if not trade_returns.empty else 0.0,
        best_trade=float(trade_returns.max()) if not trade_returns.empty else 0.0,
        worst_trade=float(trade_returns.min()) if not trade_returns.empty else 0.0,
        equity_curve=[float(value) for value in net_equity],
        drawdown_curve=drawdown_curve,
        trades=trades,
        net_returns=[float(value) for value in net_returns],
    )


def run_vectorized_backtest(request: VectorizedBacktestRequest) -> dict[str, Any]:
    """Run a vectorized backtest from the legacy request shape."""

    result = evaluate_positions(
        pd.DataFrame(request.prices),
        pd.Series(request.signals),
        cost_model=request.cost_model,
        initial_capital=request.initial_capital,
    )
    return result.to_dict()
