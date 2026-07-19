"""Delayed structural reference timing and target primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetReferenceTimes:
    """Frozen timestamps for one scheduled decision."""

    decision_time: datetime
    immediate_next_bar_open: datetime
    delayed_entry_reference: datetime
    exit_15m: datetime
    exit_30m: datetime
    exit_60m: datetime


def target_reference_times(decision_time: datetime) -> TargetReferenceTimes:
    """Return t+1, delayed t+2 entry, and exact outcome reference times."""

    if decision_time.tzinfo is None:
        raise ValueError("decision timestamp must be timezone-aware")
    delayed_entry = decision_time + timedelta(minutes=5)
    return TargetReferenceTimes(
        decision_time=decision_time,
        immediate_next_bar_open=decision_time,
        delayed_entry_reference=delayed_entry,
        exit_15m=delayed_entry + timedelta(minutes=15),
        exit_30m=delayed_entry + timedelta(minutes=30),
        exit_60m=delayed_entry + timedelta(minutes=60),
    )


def percentile_rank(values: pd.Series) -> pd.Series:
    """Map ascending average ranks to the documented continuous [0, 1] range."""

    valid_count = int(values.notna().sum())
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    if valid_count == 0:
        return result
    if valid_count == 1:
        result.loc[values.notna()] = 0.5
        return result
    ranks = values.loc[values.notna()].rank(method="average", ascending=True)
    result.loc[ranks.index] = (ranks - 1.0) / (valid_count - 1.0)
    return result


def _lookup_bar(
    bars: pd.DataFrame,
    *,
    symbol: str,
    timestamp_column: str,
    timestamp: pd.Timestamp,
) -> pd.Series | None:
    matches = bars.loc[bars["symbol"].eq(symbol) & bars[timestamp_column].eq(timestamp)]
    if len(matches) != 1:
        return None
    row = matches.iloc[0]
    if (
        not bool(row.get("fully_completed", True))
        or row.get("gap_status", "complete") != "complete"
    ):
        return None
    return row


def build_target_ledger(events: pd.DataFrame, settlement_bars: pd.DataFrame) -> pd.DataFrame:
    """Settle frozen slate members without changing their decision-time membership."""

    required_events = {
        "event_id",
        "slate_id",
        "symbol",
        "assigned_decision_time",
        "session_close",
    }
    required_bars = {"symbol", "bar_start", "bar_end", "open", "close"}
    missing_events = sorted(required_events.difference(events.columns))
    missing_bars = sorted(required_bars.difference(settlement_bars.columns))
    if missing_events or missing_bars:
        raise ValueError(f"missing target inputs: events={missing_events}, bars={missing_bars}")
    bars = settlement_bars.copy()
    bars["bar_start"] = pd.to_datetime(bars["bar_start"], utc=True)
    bars["bar_end"] = pd.to_datetime(bars["bar_end"], utc=True)
    ledger = events.copy()
    ledger["assigned_decision_time"] = pd.to_datetime(ledger["assigned_decision_time"], utc=True)
    ledger["session_close"] = pd.to_datetime(ledger["session_close"], utc=True)
    settled: list[dict[str, object]] = []
    for _, event in ledger.iterrows():
        references = target_reference_times(event["assigned_decision_time"].to_pydatetime())
        entry_time = pd.Timestamp(references.delayed_entry_reference)
        reference_times = {
            "15m": pd.Timestamp(references.exit_15m),
            "30m": pd.Timestamp(references.exit_30m),
            "60m": pd.Timestamp(references.exit_60m),
            "session_close": event["session_close"],
        }
        entry = _lookup_bar(
            bars,
            symbol=str(event["symbol"]),
            timestamp_column="bar_start",
            timestamp=entry_time,
        )
        exits = {
            horizon: _lookup_bar(
                bars,
                symbol=str(event["symbol"]),
                timestamp_column="bar_end",
                timestamp=reference_time,
            )
            for horizon, reference_time in reference_times.items()
        }
        row: dict[str, object] = {str(column): value for column, value in event.to_dict().items()}
        row["immediate_next_bar_open_time"] = pd.Timestamp(references.immediate_next_bar_open)
        row["delayed_entry_reference_time"] = entry_time
        row["exit_reference_15m_time"] = reference_times["15m"]
        row["exit_reference_30m_time"] = reference_times["30m"]
        row["exit_reference_60m_time"] = reference_times["60m"]
        row["session_close_reference_time"] = reference_times["session_close"]
        row["entry_reference_open"] = np.nan if entry is None else float(entry["open"])
        unavailable: list[str] = []
        if entry is None:
            unavailable.append("missing_delayed_entry_reference_bar")
        for horizon, exit_row in exits.items():
            output = f"exit_reference_{horizon}_close"
            row[output] = np.nan if exit_row is None else float(exit_row["close"])
            if exit_row is None:
                unavailable.append(f"missing_{horizon}_exit_reference_bar")
        entry_open = float(cast(Any, row["entry_reference_open"]))
        for horizon in ("15m", "30m", "60m", "session_close"):
            exit_close = float(cast(Any, row[f"exit_reference_{horizon}_close"]))
            row[f"future_return_{horizon}"] = (
                exit_close / entry_open - 1.0
                if np.isfinite(entry_open) and np.isfinite(exit_close) and entry_open > 0.0
                else np.nan
            )
        row["future_absolute_movement"] = abs(float(cast(Any, row["future_return_60m"])))
        row["target_unavailable_reason"] = "|".join(unavailable)
        settled.append(row)
    result = pd.DataFrame(settled)
    result["slate_original_size"] = result.groupby("slate_id", sort=True)["event_id"].transform(
        "size"
    )
    result["slate_valid_primary_targets"] = result.groupby("slate_id", sort=True)[
        "future_return_60m"
    ].transform("count")
    result["slate_unavailable_fraction"] = 1.0 - (
        result["slate_valid_primary_targets"] / result["slate_original_size"]
    )
    result["slate_evaluable"] = result["slate_valid_primary_targets"].ge(8) & result[
        "slate_unavailable_fraction"
    ].le(0.10 + 1e-12)
    for output, source in (
        ("target_rank_15m", "future_return_15m"),
        ("target_rank_30m", "future_return_30m"),
        ("target_rank_60m", "future_return_60m"),
        ("target_rank_session_close", "future_return_session_close"),
        ("target_rank_absolute_movement", "future_absolute_movement"),
    ):
        result[output] = np.nan
        for _, group in result.loc[result["slate_evaluable"]].groupby("slate_id", sort=True):
            result.loc[group.index, output] = percentile_rank(group[source])
    return result.sort_values(
        ["assigned_decision_time", "slate_id", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
