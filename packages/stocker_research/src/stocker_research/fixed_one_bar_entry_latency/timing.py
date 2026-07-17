"""Exact provider-clock scoring for the frozen T0 versus T1 experiment."""

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


def _gross_bps(direction: int, entry_price: float, exit_price: float) -> float:
    return 10_000.0 * float(direction) * (exit_price / entry_price - 1.0)


def direction_adjusted_entry_move_bps(
    *, direction: int, t0_entry_price: float, t1_entry_price: float
) -> float:
    """Return the registered direction-adjusted price movement between entries."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if t0_entry_price <= 0.0 or t1_entry_price <= 0.0:
        raise ValueError("entry prices must be positive")
    return 10_000.0 * float(direction) * (t1_entry_price / t0_entry_price - 1.0)


@dataclass(frozen=True)
class FixedLatencyResult:
    """One immutable T0 opportunity evaluated at an unconditional later open."""

    status: str
    t0_entry_timestamp: pd.Timestamp | None = None
    t0_entry_price: float | None = None
    t1_expected_timestamp: pd.Timestamp | None = None
    t1_entry_timestamp: pd.Timestamp | None = None
    t1_entry_price: float | None = None
    original_terminal_timestamp: pd.Timestamp | None = None
    terminal_price: float | None = None
    t0_gross_return_bps: float | None = None
    t0_total_cost_bps: float | None = None
    t0_net_return_bps: float | None = None
    t1_gross_return_bps: float | None = None
    t1_total_cost_bps: float | None = None
    t1_net_return_bps: float | None = None
    paired_difference_bps: float | None = None
    direction_adjusted_entry_move_bps: float | None = None
    exact_entry_price_effect_bps: float | None = None
    reconciliation_error_bps: float | None = None
    exposure_bars_remaining: int | None = None
    intervening_bar_open: float | None = None
    intervening_bar_high: float | None = None
    intervening_bar_low: float | None = None
    intervening_bar_close: float | None = None
    first_bar_signed_close_bps: float | None = None
    first_bar_favourable_excursion_bps: float | None = None
    first_bar_adverse_excursion_bps: float | None = None
    restarted_exit_timestamp: pd.Timestamp | None = None
    restarted_terminal_price: float | None = None
    restarted_gross_return_bps: float | None = None
    restarted_net_return_bps: float | None = None


def _one_row(frame: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    rows = frame.loc[frame["timestamp"].eq(timestamp)]
    return rows.iloc[0] if len(rows) == 1 else None


def _unavailable(
    status: str,
    *,
    t0: pd.Timestamp,
    t0_price: float,
    expected: pd.Timestamp,
    terminal: pd.Timestamp,
) -> FixedLatencyResult:
    return FixedLatencyResult(
        status=status,
        t0_entry_timestamp=t0,
        t0_entry_price=t0_price,
        t1_expected_timestamp=expected,
        original_terminal_timestamp=terminal,
    )


def score_fixed_latency(
    bars: pd.DataFrame,
    *,
    anchor_timestamp: pd.Timestamp,
    entry_step: int,
    t0_entry_timestamp: pd.Timestamp,
    t0_entry_price: float,
    original_terminal_timestamp: pd.Timestamp,
    original_terminal_price: float,
    direction: int,
    source_t0_gross_return_bps: float,
    source_t0_net_return_bps: float,
    latency_bars: int = 1,
    cost_bps_per_side: float = 5.0,
    restarted_horizon_bars: int = 24,
    tolerance: float = 1e-8,
) -> FixedLatencyResult:
    """Score a fixed post-fill delay without selecting on intervening prices.

    T0 remains the exact stored OCO breakout fill.  T1 is the provider open
    immediately after ``latency_bars`` completed bars from that stored fill-bar
    timestamp.  Missing exact bars fail closed; later rows never substitute.
    """

    anchor = _utc(anchor_timestamp)
    t0 = _utc(t0_entry_timestamp)
    terminal = _utc(original_terminal_timestamp)
    expected = t0 + int(latency_bars) * BAR_DURATION
    t0_price = float(t0_entry_price)
    terminal_price = float(original_terminal_price)
    if direction not in (-1, 1):
        return _unavailable(
            "ambiguous_direction",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    if entry_step <= 0 or latency_bars <= 0:
        raise ValueError("entry_step and latency_bars must be positive")
    if t0 != anchor + int(entry_step) * BAR_DURATION:
        return _unavailable(
            "t0_timestamp_mismatch",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    if terminal != anchor + 25 * BAR_DURATION:
        return _unavailable(
            "original_terminal_timestamp_mismatch",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    if expected >= terminal:
        return _unavailable(
            "latency_entry_too_late",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    required = {"timestamp", "open", "high", "low", "close"}
    if bars.empty or required - set(bars):
        return _unavailable(
            "missing_provider_data",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    frame = bars.loc[:, sorted(required)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.loc[frame["timestamp"].notna()].sort_values("timestamp", kind="stable")
    t0_row = _one_row(frame, t0)
    if t0_row is None:
        return _unavailable(
            "missing_exact_t0_provider_bar",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    t1_row = _one_row(frame, expected)
    if t1_row is None:
        return _unavailable(
            "missing_exact_t1_open",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    terminal_row_start = terminal - BAR_DURATION
    terminal_row = _one_row(frame, terminal_row_start)
    if terminal_row is None:
        return _unavailable(
            "missing_exact_terminal_bar",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    values = np.array(
        [
            t0_price,
            terminal_price,
            float(t1_row["open"]),
            float(terminal_row["close"]),
            float(t0_row["open"]),
            float(t0_row["high"]),
            float(t0_row["low"]),
            float(t0_row["close"]),
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or (values[:4] <= 0.0).any():
        return _unavailable(
            "invalid_provider_or_source_price",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    if not np.isclose(float(terminal_row["close"]), terminal_price, rtol=0.0, atol=tolerance):
        return _unavailable(
            "terminal_price_mismatch",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    if not (float(t0_row["low"]) - tolerance <= t0_price <= float(t0_row["high"]) + tolerance):
        return _unavailable(
            "t0_fill_outside_provider_bar",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    t0_gross = _gross_bps(direction, t0_price, terminal_price)
    total_cost = 2.0 * float(cost_bps_per_side)
    t0_net = t0_gross - total_cost
    if not np.isclose(t0_gross, source_t0_gross_return_bps, rtol=0.0, atol=tolerance):
        return _unavailable(
            "source_t0_gross_reconciliation_failed",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    if not np.isclose(t0_net, source_t0_net_return_bps, rtol=0.0, atol=tolerance):
        return _unavailable(
            "source_t0_net_reconciliation_failed",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    expected_path = pd.date_range(expected, terminal_row_start, freq=BAR_DURATION, tz="UTC")
    path = frame.loc[frame["timestamp"].isin(expected_path)]
    if len(path) != len(expected_path) or path["timestamp"].duplicated().any():
        return _unavailable(
            "incomplete_constant_terminal_path",
            t0=t0,
            t0_price=t0_price,
            expected=expected,
            terminal=terminal,
        )
    t1_price = float(t1_row["open"])
    t1_gross = _gross_bps(direction, t1_price, terminal_price)
    t1_net = t1_gross - total_cost
    delta = t1_net - t0_net
    entry_move = direction_adjusted_entry_move_bps(
        direction=direction,
        t0_entry_price=t0_price,
        t1_entry_price=t1_price,
    )
    exact_effect = 10_000.0 * float(direction) * terminal_price * (1.0 / t1_price - 1.0 / t0_price)
    close = float(t0_row["close"])
    high = float(t0_row["high"])
    low = float(t0_row["low"])
    signed_close = _gross_bps(direction, t0_price, close)
    if direction == 1:
        favourable = 10_000.0 * (high / t0_price - 1.0)
        adverse = 10_000.0 * (low / t0_price - 1.0)
    else:
        favourable = 10_000.0 * (1.0 - low / t0_price)
        adverse = 10_000.0 * (1.0 - high / t0_price)

    restarted_exit: pd.Timestamp | None = None
    restarted_price: float | None = None
    restarted_gross: float | None = None
    restarted_net: float | None = None
    if restarted_horizon_bars > 0:
        starts = pd.date_range(
            expected, periods=int(restarted_horizon_bars), freq=BAR_DURATION, tz="UTC"
        )
        rows = frame.loc[frame["timestamp"].isin(starts)]
        if len(rows) == len(starts) and not rows["timestamp"].duplicated().any():
            candidate_exit = starts[-1] + BAR_DURATION
            if (
                candidate_exit.tz_convert("America/New_York").date()
                == expected.tz_convert("America/New_York").date()
            ):
                restarted_exit = candidate_exit
                restarted_row = rows.loc[rows["timestamp"].eq(starts[-1])].iloc[0]
                restarted_price = float(restarted_row["close"])
                restarted_gross = _gross_bps(direction, t1_price, restarted_price)
                restarted_net = restarted_gross - total_cost

    return FixedLatencyResult(
        status="available",
        t0_entry_timestamp=t0,
        t0_entry_price=t0_price,
        t1_expected_timestamp=expected,
        t1_entry_timestamp=expected,
        t1_entry_price=t1_price,
        original_terminal_timestamp=terminal,
        terminal_price=terminal_price,
        t0_gross_return_bps=t0_gross,
        t0_total_cost_bps=total_cost,
        t0_net_return_bps=t0_net,
        t1_gross_return_bps=t1_gross,
        t1_total_cost_bps=total_cost,
        t1_net_return_bps=t1_net,
        paired_difference_bps=delta,
        direction_adjusted_entry_move_bps=entry_move,
        exact_entry_price_effect_bps=exact_effect,
        reconciliation_error_bps=delta - exact_effect,
        exposure_bars_remaining=len(expected_path),
        intervening_bar_open=float(t0_row["open"]),
        intervening_bar_high=high,
        intervening_bar_low=low,
        intervening_bar_close=close,
        first_bar_signed_close_bps=signed_close,
        first_bar_favourable_excursion_bps=favourable,
        first_bar_adverse_excursion_bps=adverse,
        restarted_exit_timestamp=restarted_exit,
        restarted_terminal_price=restarted_price,
        restarted_gross_return_bps=restarted_gross,
        restarted_net_return_bps=restarted_net,
    )
