"""Causal next-open remaining-payoff calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RemainingPayoff:
    status: str
    entry_timestamp: pd.Timestamp | None = None
    entry_price: float | None = None
    constant_terminal_exit_timestamp: pd.Timestamp | None = None
    constant_terminal_gross_bps: float | None = None
    constant_terminal_net_bps: float | None = None
    restarted_exit_timestamp: pd.Timestamp | None = None
    restarted_gross_bps: float | None = None
    restarted_net_bps: float | None = None
    remaining_mfe_bps: float | None = None
    remaining_mae_bps: float | None = None


def _gross_bps(direction: int, entry: float, exit_price: float) -> float:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    return 10_000.0 * direction * (exit_price / entry - 1.0)


def remaining_payoff(
    bars: pd.DataFrame,
    *,
    direction: int,
    checkpoint_timestamp: pd.Timestamp,
    terminal_timestamp: pd.Timestamp,
    cost_bps_per_side: float = 5.0,
    restarted_horizon_bars: int = 24,
    execution_delay_bars: int = 0,
) -> RemainingPayoff:
    """Score only payoff available after a frozen checkpoint."""

    checkpoint = pd.Timestamp(checkpoint_timestamp)
    terminal = pd.Timestamp(terminal_timestamp)
    if checkpoint.tzinfo is None:
        checkpoint = checkpoint.tz_localize("UTC")
    else:
        checkpoint = checkpoint.tz_convert("UTC")
    if terminal.tzinfo is None:
        terminal = terminal.tz_localize("UTC")
    else:
        terminal = terminal.tz_convert("UTC")
    if checkpoint >= terminal:
        return RemainingPayoff(status="too_late")
    if bars.empty:
        return RemainingPayoff(status="missing_source_data")
    required = {"timestamp", "open", "high", "low", "close"}
    if required - set(bars.columns):
        return RemainingPayoff(status="missing_source_data")
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    entry_candidates = frame.index[frame["timestamp"].ge(checkpoint)]
    if len(entry_candidates) <= execution_delay_bars:
        return RemainingPayoff(status="missing_source_data")
    entry_index = int(entry_candidates[execution_delay_bars])
    entry_timestamp = pd.Timestamp(frame.iloc[entry_index]["timestamp"])
    if entry_timestamp >= terminal:
        return RemainingPayoff(status="too_late")
    terminal_rows = frame.index[(frame["timestamp"] + pd.Timedelta(minutes=5)).eq(terminal)]
    if len(terminal_rows) != 1:
        return RemainingPayoff(status="missing_source_data")
    terminal_index = int(terminal_rows[0])
    if terminal_index < entry_index:
        return RemainingPayoff(status="too_late")
    entry_price = float(frame.iloc[entry_index]["open"])
    terminal_price = float(frame.iloc[terminal_index]["close"])
    costs = 2.0 * float(cost_bps_per_side)
    constant_gross = _gross_bps(direction, entry_price, terminal_price)

    path = frame.iloc[entry_index : terminal_index + 1]
    high = pd.to_numeric(path["high"], errors="raise").to_numpy(float)
    low = pd.to_numeric(path["low"], errors="raise").to_numpy(float)
    if direction == 1:
        favourable = 10_000.0 * (high / entry_price - 1.0)
        adverse = 10_000.0 * (low / entry_price - 1.0)
    else:
        favourable = 10_000.0 * (1.0 - low / entry_price)
        adverse = 10_000.0 * (1.0 - high / entry_price)

    restarted_index = entry_index + int(restarted_horizon_bars) - 1
    restarted_timestamp: pd.Timestamp | None = None
    restarted_gross: float | None = None
    restarted_net: float | None = None
    if restarted_horizon_bars > 0 and restarted_index < len(frame):
        restarted_row = frame.iloc[restarted_index]
        restarted_timestamp = pd.Timestamp(restarted_row["timestamp"]) + pd.Timedelta(minutes=5)
        entry_session = entry_timestamp.tz_convert("America/New_York").date()
        exit_session = restarted_timestamp.tz_convert("America/New_York").date()
        if entry_session == exit_session:
            restarted_gross = _gross_bps(direction, entry_price, float(restarted_row["close"]))
            restarted_net = restarted_gross - costs
        else:
            restarted_timestamp = None

    return RemainingPayoff(
        status="available",
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        constant_terminal_exit_timestamp=terminal,
        constant_terminal_gross_bps=constant_gross,
        constant_terminal_net_bps=constant_gross - costs,
        restarted_exit_timestamp=restarted_timestamp,
        restarted_gross_bps=restarted_gross,
        restarted_net_bps=restarted_net,
        remaining_mfe_bps=float(np.max(favourable)),
        remaining_mae_bps=float(np.min(adverse)),
    )
