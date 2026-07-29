"""Frozen OCO reconstruction and adverse-entry execution arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, cast

import numpy as np
import pandas as pd

BAR_DURATION: Final[pd.Timedelta] = pd.Timedelta(minutes=5)
FILL_STRESSES_BPS: Final[dict[str, float]] = {
    "F0": 0.0,
    "F5": 5.0,
    "F10": 10.0,
    "F15": 15.0,
    "F20": 20.0,
}
PRIMARY_FILL_MODEL: Final[str] = "F10"


class FillEvidence(StrEnum):
    """Predeclared evidence classes for the frozen reference fill."""

    EXACTLY_OBSERVABLE = "EXACTLY_OBSERVABLE"
    BOUNDED_BUT_NOT_EXACT = "BOUNDED_BUT_NOT_EXACT"
    GAP_FILL_OBSERVABLE = "GAP_FILL_OBSERVABLE"
    AMBIGUOUS_WITHIN_BAR_ORDER = "AMBIGUOUS_WITHIN_BAR_ORDER"
    MISSING_MARKET_DATA = "MISSING_MARKET_DATA"
    SIGNAL_OR_FILL_TIME_AMBIGUOUS = "SIGNAL_OR_FILL_TIME_AMBIGUOUS"
    UNAVAILABLE = "UNAVAILABLE"


class TriggerType(StrEnum):
    """Frozen OCO trigger kinds."""

    INTRABAR_THRESHOLD_CROSS = "intrabar_threshold_cross"
    OPENING_GAP_THROUGH_THRESHOLD = "opening_gap_through_threshold"
    AMBIGUOUS_DUAL_OCO_CROSS = "ambiguous_dual_oco_cross"
    NO_TRIGGER = "no_trigger"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TriggerReconstruction:
    """Causal reconstruction of the first frozen OCO event."""

    status: str
    direction: int | None
    entry_step: int | None
    trigger_type: TriggerType
    trigger_bar_timestamp: pd.Timestamp | None
    trigger_bar_open: float | None
    trigger_bar_high: float | None
    trigger_bar_low: float | None
    trigger_bar_close: float | None
    reference_entry_timestamp: pd.Timestamp | None
    reference_entry_price: float | None
    fill_evidence: FillEvidence
    threshold_known_timestamp: pd.Timestamp
    signal_known_timestamp: pd.Timestamp | None
    market_data_availability_timestamp: pd.Timestamp | None
    signal_fill_time_status: str
    long_threshold: float
    short_threshold: float


@dataclass(frozen=True)
class FillPayoff:
    """One non-cumulative fill-stress outcome at the unchanged terminal."""

    opportunity_id: str
    fill_model: str
    adverse_entry_slippage_bps: float
    direction: int
    reference_entry_price: float
    stressed_entry_price: float
    terminal_timestamp: pd.Timestamp
    terminal_price: float
    gross_payoff_bps: float
    cost_bps: float
    net_payoff_bps: float


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(cast(Any, value))
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _unavailable_trigger(
    *,
    status: str,
    evidence: FillEvidence,
    trigger_type: TriggerType,
    threshold_known: pd.Timestamp,
    long_threshold: float,
    short_threshold: float,
    step: int | None = None,
    row: pd.Series | None = None,
) -> TriggerReconstruction:
    timestamp = _utc(row["timestamp"]) if row is not None else None
    return TriggerReconstruction(
        status=status,
        direction=None,
        entry_step=step,
        trigger_type=trigger_type,
        trigger_bar_timestamp=timestamp,
        trigger_bar_open=float(row["open"]) if row is not None else None,
        trigger_bar_high=float(row["high"]) if row is not None else None,
        trigger_bar_low=float(row["low"]) if row is not None else None,
        trigger_bar_close=float(row["close"]) if row is not None else None,
        reference_entry_timestamp=None,
        reference_entry_price=None,
        fill_evidence=evidence,
        threshold_known_timestamp=threshold_known,
        signal_known_timestamp=None,
        market_data_availability_timestamp=(timestamp + BAR_DURATION if timestamp else None),
        signal_fill_time_status=status,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    )


def reconstruct_frozen_oco_trigger(
    bars: pd.DataFrame,
    *,
    anchor_timestamp: pd.Timestamp,
    long_threshold: float,
    short_threshold: float,
    horizon_bars: int = 24,
) -> TriggerReconstruction:
    """Reconstruct the original first-event OCO convention from exact 5m OHLC.

    Provider timestamps label bar starts. Opening gaps are directly observed.
    A one-sided intrabar high/low cross reproduces the historical threshold
    reference, but five-minute data cannot prove its exact time or execution.
    """

    anchor = _utc(anchor_timestamp)
    upper = float(long_threshold)
    lower = float(short_threshold)
    if not np.isfinite([upper, lower]).all() or lower <= 0.0 or upper <= lower:
        raise ValueError("thresholds must be finite, positive, and ordered")
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    required = {"timestamp", "open", "high", "low", "close"}
    if missing := sorted(required - set(bars.columns)):
        raise ValueError(f"missing OHLC columns: {missing}")
    frame = bars.loc[:, ["timestamp", "open", "high", "low", "close"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values("timestamp", kind="stable")
    threshold_known = anchor + BAR_DURATION
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        return _unavailable_trigger(
            status="invalid_or_duplicate_timestamp",
            evidence=FillEvidence.UNAVAILABLE,
            trigger_type=TriggerType.UNAVAILABLE,
            threshold_known=threshold_known,
            long_threshold=upper,
            short_threshold=lower,
        )
    for step in range(1, horizon_bars + 1):
        expected = anchor + step * BAR_DURATION
        matching = frame.loc[frame["timestamp"].eq(expected)]
        if len(matching) != 1:
            return _unavailable_trigger(
                status="missing_exact_market_bar",
                evidence=FillEvidence.MISSING_MARKET_DATA,
                trigger_type=TriggerType.UNAVAILABLE,
                threshold_known=threshold_known,
                long_threshold=upper,
                short_threshold=lower,
                step=step,
            )
        row = matching.iloc[0]
        values = np.asarray([row["open"], row["high"], row["low"], row["close"]], dtype=float)
        if (
            not np.isfinite(values).all()
            or (values <= 0.0).any()
            or values[2] > min(values[0], values[3])
            or values[1] < max(values[0], values[3])
        ):
            return _unavailable_trigger(
                status="invalid_ohlc",
                evidence=FillEvidence.UNAVAILABLE,
                trigger_type=TriggerType.UNAVAILABLE,
                threshold_known=threshold_known,
                long_threshold=upper,
                short_threshold=lower,
                step=step,
                row=row,
            )
        open_price, high, low, close = values
        open_upper = open_price >= upper
        open_lower = open_price <= lower
        upper_hit = high >= upper
        lower_hit = low <= lower
        if open_upper and open_lower:
            return _unavailable_trigger(
                status="ambiguous_opening_oco",
                evidence=FillEvidence.AMBIGUOUS_WITHIN_BAR_ORDER,
                trigger_type=TriggerType.AMBIGUOUS_DUAL_OCO_CROSS,
                threshold_known=threshold_known,
                long_threshold=upper,
                short_threshold=lower,
                step=step,
                row=row,
            )
        if open_upper or open_lower:
            direction = 1 if open_upper else -1
            entry_price = max(upper, open_price) if direction == 1 else min(lower, open_price)
            return TriggerReconstruction(
                status="triggered",
                direction=direction,
                entry_step=step,
                trigger_type=TriggerType.OPENING_GAP_THROUGH_THRESHOLD,
                trigger_bar_timestamp=expected,
                trigger_bar_open=open_price,
                trigger_bar_high=high,
                trigger_bar_low=low,
                trigger_bar_close=close,
                reference_entry_timestamp=expected,
                reference_entry_price=entry_price,
                fill_evidence=FillEvidence.GAP_FILL_OBSERVABLE,
                threshold_known_timestamp=threshold_known,
                signal_known_timestamp=expected,
                market_data_availability_timestamp=expected,
                signal_fill_time_status="CAUSALLY_ORDERED",
                long_threshold=upper,
                short_threshold=lower,
            )
        if upper_hit and lower_hit:
            return _unavailable_trigger(
                status="ambiguous_dual_oco_cross",
                evidence=FillEvidence.AMBIGUOUS_WITHIN_BAR_ORDER,
                trigger_type=TriggerType.AMBIGUOUS_DUAL_OCO_CROSS,
                threshold_known=threshold_known,
                long_threshold=upper,
                short_threshold=lower,
                step=step,
                row=row,
            )
        if upper_hit or lower_hit:
            direction = 1 if upper_hit else -1
            return TriggerReconstruction(
                status="triggered",
                direction=direction,
                entry_step=step,
                trigger_type=TriggerType.INTRABAR_THRESHOLD_CROSS,
                trigger_bar_timestamp=expected,
                trigger_bar_open=open_price,
                trigger_bar_high=high,
                trigger_bar_low=low,
                trigger_bar_close=close,
                reference_entry_timestamp=expected,
                reference_entry_price=upper if direction == 1 else lower,
                fill_evidence=FillEvidence.BOUNDED_BUT_NOT_EXACT,
                threshold_known_timestamp=threshold_known,
                signal_known_timestamp=None,
                market_data_availability_timestamp=expected + BAR_DURATION,
                signal_fill_time_status="SIGNAL_OR_FILL_TIME_AMBIGUOUS",
                long_threshold=upper,
                short_threshold=lower,
            )
    return _unavailable_trigger(
        status="no_trigger",
        evidence=FillEvidence.UNAVAILABLE,
        trigger_type=TriggerType.NO_TRIGGER,
        threshold_known=threshold_known,
        long_threshold=upper,
        short_threshold=lower,
    )


def apply_adverse_entry_slippage(
    reference_entry_price: float, direction: int, adverse_bps: float
) -> float:
    """Apply one adverse entry stress directly to the frozen reference price."""

    entry = float(reference_entry_price)
    stress = float(adverse_bps)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if not np.isfinite([entry, stress]).all() or entry <= 0.0 or stress < 0.0:
        raise ValueError("entry must be positive and stress non-negative")
    stressed = entry * (1.0 + stress / 10_000.0 if direction == 1 else 1.0 - stress / 10_000.0)
    if stressed <= 0.0:
        raise ValueError("adverse stress produced a non-positive entry")
    return stressed


def gross_payoff_bps(direction: int, entry_price: float, terminal_price: float) -> float:
    """Apply the repository's frozen simple-return convention."""

    entry = float(entry_price)
    terminal = float(terminal_price)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if not np.isfinite([entry, terminal]).all() or entry <= 0.0 or terminal <= 0.0:
        raise ValueError("prices must be finite and positive")
    return 10_000.0 * float(direction) * (terminal / entry - 1.0)


def score_fill_envelope(
    *,
    opportunity_id: str,
    direction: int,
    reference_entry_price: float,
    terminal_timestamp: pd.Timestamp,
    terminal_price: float,
    cost_bps: float,
) -> tuple[FillPayoff, ...]:
    """Score F0/F5/F10/F15/F20 from the same immutable entry and terminal."""

    terminal = _utc(terminal_timestamp)
    costs = float(cost_bps)
    if not opportunity_id:
        raise ValueError("opportunity_id is required")
    if not np.isfinite(costs) or costs < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    rows: list[FillPayoff] = []
    for fill_model, stress in FILL_STRESSES_BPS.items():
        stressed = apply_adverse_entry_slippage(reference_entry_price, direction, stress)
        gross = gross_payoff_bps(direction, stressed, terminal_price)
        rows.append(
            FillPayoff(
                opportunity_id=opportunity_id,
                fill_model=fill_model,
                adverse_entry_slippage_bps=stress,
                direction=direction,
                reference_entry_price=float(reference_entry_price),
                stressed_entry_price=stressed,
                terminal_timestamp=terminal,
                terminal_price=float(terminal_price),
                gross_payoff_bps=gross,
                cost_bps=costs,
                net_payoff_bps=gross - costs,
            )
        )
    return tuple(rows)
