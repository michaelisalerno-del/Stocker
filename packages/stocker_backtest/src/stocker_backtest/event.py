"""Placeholder event-driven backtest interface."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventBacktestRequest:
    """Inputs expected by a future event-driven backtest."""

    events: Any
    initial_cash: float


def run_event_backtest(request: EventBacktestRequest) -> dict[str, Any]:
    """Run a future event-driven backtest."""

    raise NotImplementedError("Event-driven backtesting is not implemented yet")
