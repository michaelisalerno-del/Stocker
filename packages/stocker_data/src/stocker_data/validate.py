"""Data validation placeholders for OHLCV datasets."""

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return cast(pd.DataFrame, frame.to_pandas())
    return pd.DataFrame(frame)


def find_missing_timestamps(
    frame: Any, *, timestamp_col: str = "timestamp", freq: str | None = None
) -> list[pd.Timestamp]:
    """Find expected timestamps that are absent from a time-indexed dataset.

    Future versions should compare the dataset against an exchange calendar and session
    schedule. The bootstrap implementation can only infer gaps when a fixed pandas
    frequency is supplied.
    """

    data = _to_pandas(frame)
    if timestamp_col not in data or freq is None or data.empty:
        return []
    timestamps = pd.to_datetime(data[timestamp_col]).sort_values()
    expected = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq=freq)
    missing = expected.difference(pd.DatetimeIndex(timestamps))
    return list(missing)


def find_duplicate_timestamps(
    frame: Any, *, timestamp_col: str = "timestamp"
) -> list[pd.Timestamp]:
    """Find duplicate timestamps that would corrupt ordered OHLCV bars."""

    data = _to_pandas(frame)
    if timestamp_col not in data:
        return []
    timestamps = pd.to_datetime(data[timestamp_col])
    return list(timestamps[timestamps.duplicated()].unique())


def timestamps_are_timezone_aware(frame: Any, *, timestamp_col: str = "timestamp") -> bool:
    """Check whether timestamps carry timezone information.

    Future live and backtest code should reject ambiguous timestamps before signals are
    evaluated.
    """

    data = _to_pandas(frame)
    if timestamp_col not in data or data.empty:
        return False
    timestamps = pd.to_datetime(data[timestamp_col])
    return timestamps.dt.tz is not None


def validate_ohlc_sanity(
    frame: Any, *, columns: Sequence[str] = ("open", "high", "low", "close")
) -> list[str]:
    """Check basic OHLC invariants.

    This should eventually catch impossible bars, bad splits, stale repeated quotes, and
    vendor adjustment errors. The bootstrap checks high/low containment only.
    """

    data = _to_pandas(frame)
    missing = [column for column in columns if column not in data]
    if missing:
        return [f"missing OHLC columns: {', '.join(missing)}"]
    open_col, high_col, low_col, close_col = columns
    invalid = data[
        (data[high_col] < data[[open_col, close_col]].max(axis=1))
        | (data[low_col] > data[[open_col, close_col]].min(axis=1))
        | (data[high_col] < data[low_col])
    ]
    if invalid.empty:
        return []
    return [f"{len(invalid)} rows violate OHLC high/low containment"]


def validate_volume_sanity(frame: Any, *, volume_col: str = "volume") -> list[str]:
    """Check basic volume invariants.

    Future checks should account for instrument type, session calendars, and whether the
    source provides real volume, tick volume, or quote count. The bootstrap rejects
    negative values only.
    """

    data = _to_pandas(frame)
    if volume_col not in data:
        return []
    invalid = data[data[volume_col] < 0]
    if invalid.empty:
        return []
    return [f"{len(invalid)} rows have negative volume"]
