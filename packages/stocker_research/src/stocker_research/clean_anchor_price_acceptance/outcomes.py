"""Exact-clock remaining-payoff calculations for the registered experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BAR_DURATION = pd.Timedelta(minutes=5)


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(str(value))
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class RemainingPayoff:
    """Payoff available strictly after a completed one-bar checkpoint."""

    status: str
    entry_timestamp: pd.Timestamp | None = None
    entry_price: float | None = None
    exit_timestamp: pd.Timestamp | None = None
    exit_price: float | None = None
    gross_payoff_bps: float | None = None
    entry_cost_bps: float | None = None
    exit_cost_bps: float | None = None
    total_cost_bps: float | None = None
    net_payoff_bps: float | None = None
    remaining_mfe_bps: float | None = None
    remaining_mae_bps: float | None = None
    restarted_exit_timestamp: pd.Timestamp | None = None
    restarted_gross_payoff_bps: float | None = None
    restarted_net_payoff_bps: float | None = None


def _one_row_at(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    rows = frame.loc[frame["timestamp"].eq(timestamp)]
    if len(rows) != 1:
        return None
    return rows.iloc[0]


def _gross_bps(direction: int, entry: float, exit_price: float) -> float:
    return 10_000.0 * float(direction) * (exit_price / entry - 1.0)


def calculate_remaining_payoff(
    bars: pd.DataFrame,
    *,
    anchor_timestamp: pd.Timestamp,
    original_terminal_timestamp: pd.Timestamp,
    direction: int,
    cost_bps_per_side: float = 5.0,
    additional_delay_bars: int = 0,
    restarted_horizon_bars: int = 24,
) -> RemainingPayoff:
    """Enter at the exact registered next open and keep the original terminal."""

    if direction not in (-1, 1):
        return RemainingPayoff(status="ambiguous_direction")
    if additional_delay_bars < 0:
        raise ValueError("additional_delay_bars must be non-negative")
    required = {"timestamp", "open", "high", "low", "close"}
    if bars.empty or required - set(bars.columns):
        return RemainingPayoff(status="missing_source_data")
    frame = bars.loc[:, sorted(required)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[frame["timestamp"].notna()].sort_values("timestamp", kind="stable")
    anchor = _utc(anchor_timestamp)
    terminal = _utc(original_terminal_timestamp)
    entry_timestamp = anchor + (2 + additional_delay_bars) * BAR_DURATION
    if entry_timestamp >= terminal:
        return RemainingPayoff(status="too_late")
    entry_row = _one_row_at(frame, entry_timestamp)
    if entry_row is None:
        return RemainingPayoff(status="missing_exact_entry_bar")
    terminal_row_start = terminal - BAR_DURATION
    terminal_row = _one_row_at(frame, terminal_row_start)
    if terminal_row is None:
        return RemainingPayoff(status="missing_exact_terminal_bar")
    entry_price = float(entry_row["open"])
    exit_price = float(terminal_row["close"])
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0.0:
        return RemainingPayoff(status="invalid_entry_or_exit_price")
    expected_path = pd.date_range(
        entry_timestamp,
        terminal_row_start,
        freq=BAR_DURATION,
        tz="UTC",
    )
    path = frame.loc[frame["timestamp"].isin(expected_path)]
    if len(path) != len(expected_path) or path["timestamp"].duplicated().any():
        return RemainingPayoff(status="incomplete_constant_terminal_path")
    high = pd.to_numeric(path["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(path["low"], errors="coerce").to_numpy(float)
    if not np.isfinite(high).all() or not np.isfinite(low).all():
        return RemainingPayoff(status="invalid_path_ohlc")
    if direction == 1:
        favourable = 10_000.0 * (high / entry_price - 1.0)
        adverse = 10_000.0 * (low / entry_price - 1.0)
    else:
        favourable = 10_000.0 * (1.0 - low / entry_price)
        adverse = 10_000.0 * (1.0 - high / entry_price)
    entry_cost = float(cost_bps_per_side)
    exit_cost = float(cost_bps_per_side)
    gross = _gross_bps(direction, entry_price, exit_price)

    restarted_exit: pd.Timestamp | None = None
    restarted_gross: float | None = None
    restarted_net: float | None = None
    if restarted_horizon_bars > 0:
        restarted_starts = pd.date_range(
            entry_timestamp,
            periods=restarted_horizon_bars,
            freq=BAR_DURATION,
            tz="UTC",
        )
        restarted_rows = frame.loc[frame["timestamp"].isin(restarted_starts)]
        if (
            len(restarted_rows) == restarted_horizon_bars
            and not restarted_rows["timestamp"].duplicated().any()
        ):
            candidate_exit = restarted_starts[-1] + BAR_DURATION
            if (
                candidate_exit.tz_convert("America/New_York").date()
                == entry_timestamp.tz_convert("America/New_York").date()
            ):
                restarted_exit = candidate_exit
                restarted_price = float(
                    restarted_rows.loc[
                        restarted_rows["timestamp"].eq(restarted_starts[-1]), "close"
                    ].iloc[0]
                )
                restarted_gross = _gross_bps(direction, entry_price, restarted_price)
                restarted_net = restarted_gross - entry_cost - exit_cost
    return RemainingPayoff(
        status="available",
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        exit_timestamp=terminal,
        exit_price=exit_price,
        gross_payoff_bps=gross,
        entry_cost_bps=entry_cost,
        exit_cost_bps=exit_cost,
        total_cost_bps=entry_cost + exit_cost,
        net_payoff_bps=gross - entry_cost - exit_cost,
        remaining_mfe_bps=float(np.max(favourable)),
        remaining_mae_bps=float(np.min(adverse)),
        restarted_exit_timestamp=restarted_exit,
        restarted_gross_payoff_bps=restarted_gross,
        restarted_net_payoff_bps=restarted_net,
    )
