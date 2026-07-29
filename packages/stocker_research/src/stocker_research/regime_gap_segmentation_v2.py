"""Deterministic causal source segmentation for repaired regime research V2.

This module is research-only. It contains no broker, order, position, payoff,
or production-runtime surface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
ECONOMIC_OUTCOMES_USED = False
PAYOFF_SELECTION_USED = False
PRODUCTION_RUNTIME_MODIFIED = False
STRATEGY_PROMOTION = False
PART_B_INTERACTION_SCORING_ENABLED = False
SEMANTIC_DICTIONARY_PROMOTION_ENABLED = False

_REQUIRED_COLUMNS = {
    "symbol",
    "session",
    "bar_ordinal",
    "bar_start_timestamp",
}


def _validate_expected_bars(value: int) -> int:
    expected = int(value)
    if expected <= 0 or expected > 78:
        raise ValueError("expected session bars must be in [1, 78]")
    return expected


def annotate_causal_segments(
    frame: pd.DataFrame,
    *,
    expected_bars: Mapping[tuple[str, str], int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort bars deterministically and split every session at a source gap.

    expected_bars is normally derived from the exchange calendar. Tests and
    bounded callers may supply a compact deterministic schedule. A session is
    source-complete only when every scheduled ordinal is present with an exact
    five-minute timestamp increment.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"source frame lacks segmentation columns: {missing}")
    if frame.empty:
        raise ValueError("source frame cannot be empty")

    output = frame.copy()
    output["symbol"] = output["symbol"].astype(str)
    output["session"] = output["session"].astype(str)
    output["bar_ordinal"] = pd.to_numeric(output["bar_ordinal"], errors="raise").astype(np.int16)
    output["bar_start_timestamp"] = pd.to_datetime(
        output["bar_start_timestamp"], utc=True, errors="raise"
    )
    natural_key = ["symbol", "session", "bar_start_timestamp", "bar_ordinal"]
    output = output.sort_values(natural_key, kind="mergesort").reset_index(drop=True)

    if output[["symbol", "session", "bar_ordinal"]].duplicated().any():
        raise ValueError("duplicate symbol/session/bar ordinal natural key")
    if output[["symbol", "session", "bar_start_timestamp"]].duplicated().any():
        raise ValueError("duplicate symbol/session/timestamp natural key")
    if output["bar_ordinal"].lt(0).any() or output["bar_ordinal"].gt(77).any():
        raise ValueError("bar ordinal lies outside regular-session support [0, 77]")

    output["segment_index"] = np.int16(-1)
    output["segment_id"] = ""
    output["segment_bar_ordinal"] = np.int16(-1)
    output["segment_start_reason"] = "continued"
    output["segment_end_reason"] = "continued"
    output["session_source_complete"] = False
    output["expected_session_bars"] = np.int16(0)
    output["source_gap_before"] = False
    output["source_gap_after"] = False
    gap_rows: list[dict[str, Any]] = []

    for (symbol, session), group in output.groupby(["symbol", "session"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        ordinals = group["bar_ordinal"].to_numpy(dtype=int)
        timestamps = pd.DatetimeIndex(group["bar_start_timestamp"])
        if np.any(np.diff(ordinals) <= 0):
            raise ValueError(f"non-increasing bar ordinals for {symbol} {session}")
        ordinal_break = np.diff(ordinals) != 1
        timestamp_break = np.diff(timestamps.view("i8")) != pd.Timedelta(minutes=5).value
        breaks = np.flatnonzero(ordinal_break | timestamp_break) + 1

        if expected_bars is None:
            expected_count = 78
        else:
            key = (str(symbol), str(session))
            if key not in expected_bars:
                raise ValueError(f"missing expected-session schedule for {key}")
            expected_count = _validate_expected_bars(expected_bars[key])
        exact_ordinals = np.array_equal(ordinals, np.arange(expected_count, dtype=int))
        exact_timestamps = not bool(timestamp_break.any())
        source_complete = bool(exact_ordinals and exact_timestamps)

        output.loc[positions, "expected_session_bars"] = expected_count
        output.loc[positions, "session_source_complete"] = source_complete
        split_positions = np.split(positions, breaks)
        for segment_index, segment_positions in enumerate(split_positions):
            if len(segment_positions) == 0:
                continue
            segment_id = f"{symbol}::{session}::segment_{segment_index:02d}"
            output.loc[segment_positions, "segment_index"] = segment_index
            output.loc[segment_positions, "segment_id"] = segment_id
            output.loc[segment_positions, "segment_bar_ordinal"] = np.arange(
                len(segment_positions), dtype=np.int16
            )
            first = int(segment_positions[0])
            last = int(segment_positions[-1])
            if segment_index == 0 and int(cast(Any, output.at[first, "bar_ordinal"])) == 0:
                start_reason = "session_open"
            elif segment_index == 0:
                start_reason = "incomplete_session_start"
            else:
                start_reason = "source_gap"
                output.at[first, "source_gap_before"] = True
            output.at[first, "segment_start_reason"] = start_reason

            if segment_index < len(split_positions) - 1:
                end_reason = "source_gap"
                output.at[last, "source_gap_after"] = True
            elif source_complete:
                end_reason = "scheduled_session_end"
            elif int(cast(Any, output.at[last, "bar_ordinal"])) < expected_count - 1:
                end_reason = "incomplete_session_end"
            else:
                end_reason = "unavailable_or_incomplete_session"
            output.at[last, "segment_end_reason"] = end_reason

        for local_break in breaks:
            previous_position = int(positions[local_break - 1])
            next_position = int(positions[local_break])
            previous_ordinal = int(cast(Any, output.at[previous_position, "bar_ordinal"]))
            next_ordinal = int(cast(Any, output.at[next_position, "bar_ordinal"]))
            gap_rows.append(
                {
                    "symbol": str(symbol),
                    "session": str(session),
                    "previous_position": previous_position,
                    "next_position": next_position,
                    "previous_bar_ordinal": previous_ordinal,
                    "next_bar_ordinal": next_ordinal,
                    "previous_timestamp": output.at[previous_position, "bar_start_timestamp"],
                    "next_timestamp": output.at[next_position, "bar_start_timestamp"],
                    "missing_bar_count": max(0, next_ordinal - previous_ordinal - 1),
                    "gap_reason": (
                        "missing_bar_ordinal"
                        if next_ordinal - previous_ordinal != 1
                        else "timestamp_discontinuity"
                    ),
                }
            )

    if output["segment_id"].eq("").any():
        raise AssertionError("segmentation left a row unassigned")
    output["segment_index"] = output["segment_index"].astype(np.int16)
    output["segment_bar_ordinal"] = output["segment_bar_ordinal"].astype(np.int16)
    output["expected_session_bars"] = output["expected_session_bars"].astype(np.int16)

    gap_columns = [
        "symbol",
        "session",
        "previous_position",
        "next_position",
        "previous_bar_ordinal",
        "next_bar_ordinal",
        "previous_timestamp",
        "next_timestamp",
        "missing_bar_count",
        "gap_reason",
    ]
    gap_ledger = pd.DataFrame(gap_rows, columns=gap_columns)
    return output, gap_ledger


def causal_segment_groups(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Return strictly increasing row positions for each causal segment."""

    if "segment_id" not in frame:
        raise ValueError("frame lacks segment_id")
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("segment frame must use a zero-based contiguous index")
    groups: list[np.ndarray] = []
    for _, group in frame.groupby("segment_id", sort=False):
        positions = group.index.to_numpy(dtype=int)
        if len(positions) == 0 or np.any(np.diff(positions) <= 0):
            raise ValueError("segment positions must be nonempty and increasing")
        groups.append(positions)
    assigned = np.concatenate(groups) if groups else np.asarray([], dtype=int)
    if len(assigned) != len(frame) or len(np.unique(assigned)) != len(frame):
        raise AssertionError("causal segment groups do not partition the frame")
    return tuple(groups)


def reset_stateful_array_by_segment(
    values: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    initial_value: int,
    update: Callable[[int, int], int],
) -> np.ndarray:
    """Apply a scalar state update while resetting at every causal segment."""

    source = np.asarray(values)
    if source.ndim != 1:
        raise ValueError("values must be one-dimensional")
    output = np.empty(len(source), dtype=np.int64)
    assigned = np.zeros(len(source), dtype=bool)
    for raw_positions in groups:
        positions = np.asarray(raw_positions, dtype=int)
        if len(positions) == 0:
            continue
        for offset, position in enumerate(positions):
            if position < 0 or position >= len(source) or assigned[position]:
                raise ValueError("groups overlap or reference an invalid row")
            if offset == 0:
                output[position] = initial_value
            else:
                output[position] = update(
                    int(output[int(positions[offset - 1])]), int(source[position])
                )
            assigned[position] = True
    if not assigned.all():
        raise AssertionError("groups left a stateful row unassigned")
    return output


__all__ = [
    "annotate_causal_segments",
    "causal_segment_groups",
    "reset_stateful_array_by_segment",
]
