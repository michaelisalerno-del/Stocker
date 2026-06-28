"""Structured OHLCV validation for local market-data research."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import pandas as pd

from stocker_data.schema import PRICE_COLUMNS

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class ValidationIssue:
    """One structured validation finding."""

    severity: Severity
    code: str
    message: str
    count: int = 0
    first_seen: str | None = None
    last_seen: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable issue data."""

        return asdict(self)


def _to_pandas(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return cast(pd.DataFrame, frame.to_pandas())
    return pd.DataFrame(frame)


def _timestamp_to_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _parse_timestamps(frame: pd.DataFrame, timestamp_col: str) -> pd.Series:
    if timestamp_col not in frame:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(frame[timestamp_col], errors="coerce")


def _parse_utc_timestamps(frame: pd.DataFrame, timestamp_col: str) -> pd.Series:
    if timestamp_col not in frame:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)


def _has_timezone(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return False
    return timestamp.tzinfo is not None


def _freq_for_timeframe(timeframe: str) -> str | None:
    normalized = timeframe.strip().lower()
    if normalized in {"1d", "d", "day", "daily"}:
        return "D"
    if normalized.endswith("m") and normalized[:-1].isdigit():
        return f"{normalized[:-1]}min"
    if normalized.endswith("h") and normalized[:-1].isdigit():
        return f"{normalized[:-1]}h"
    if normalized.endswith("min") and normalized[:-3].isdigit():
        return normalized
    return None


def _is_daily_timeframe(timeframe: str) -> bool:
    return timeframe.strip().lower() in {"1d", "d", "day", "daily"}


def _is_intraday_timeframe(timeframe: str) -> bool:
    normalized = timeframe.strip().lower()
    return (
        (normalized.endswith("m") and normalized[:-1].isdigit())
        or (normalized.endswith("h") and normalized[:-1].isdigit())
        or (normalized.endswith("min") and normalized[:-3].isdigit())
    )


def _issue_for_timestamps(
    severity: Severity, code: str, message: str, timestamps: pd.Series
) -> ValidationIssue:
    clean = timestamps.dropna().sort_values()
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        count=int(len(timestamps)),
        first_seen=_timestamp_to_string(clean.iloc[0]) if not clean.empty else None,
        last_seen=_timestamp_to_string(clean.iloc[-1]) if not clean.empty else None,
    )


def find_missing_timestamps(
    frame: Any, *, timestamp_col: str = "timestamp", freq: str | None = None
) -> list[pd.Timestamp]:
    """Find expected timestamps that are absent from a fixed-frequency dataset."""

    data = _to_pandas(frame)
    timestamps = _parse_timestamps(data, timestamp_col).dropna().sort_values()
    if freq is None or timestamps.empty:
        return []
    expected = pd.date_range(timestamps.iloc[0], timestamps.iloc[-1], freq=freq)
    missing = expected.difference(pd.DatetimeIndex(timestamps))
    return list(missing)


def find_duplicate_timestamps(
    frame: Any, *, timestamp_col: str = "timestamp"
) -> list[pd.Timestamp]:
    """Find duplicate timestamps that would corrupt ordered OHLCV bars."""

    data = _to_pandas(frame)
    timestamps = _parse_timestamps(data, timestamp_col)
    if timestamps.empty:
        return []
    return list(timestamps[timestamps.duplicated()].dropna().unique())


def timestamps_are_timezone_aware(frame: Any, *, timestamp_col: str = "timestamp") -> bool:
    """Check whether all present timestamps carry timezone information."""

    data = _to_pandas(frame)
    if timestamp_col not in data or data.empty:
        return False
    return bool(data[timestamp_col].map(_has_timezone).all())


def validate_ohlc_sanity(frame: Any, *, columns: Sequence[str] = PRICE_COLUMNS) -> list[str]:
    """Backwards-compatible string OHLC sanity helper."""

    issues = _validate_ohlc(_to_pandas(frame), columns=columns)
    return [issue.message for issue in issues]


def validate_volume_sanity(frame: Any, *, volume_col: str = "volume") -> list[str]:
    """Backwards-compatible string volume sanity helper."""

    issues = _validate_volume(_to_pandas(frame), volume_col=volume_col)
    return [issue.message for issue in issues]


def _validate_timestamp_presence(data: pd.DataFrame, timestamp_col: str) -> list[ValidationIssue]:
    if timestamp_col not in data:
        return [
            ValidationIssue(
                severity="error",
                code="missing_timestamp_column",
                message=f"Missing timestamp column: {timestamp_col}",
            )
        ]
    parsed = _parse_timestamps(data, timestamp_col)
    invalid = parsed.isna()
    if not invalid.any():
        return []
    return [
        ValidationIssue(
            severity="error",
            code="unparseable_timestamp",
            message="One or more timestamps could not be parsed",
            count=int(invalid.sum()),
        )
    ]


def _validate_timezone(
    data: pd.DataFrame, *, timestamp_col: str, require_timezone: bool
) -> list[ValidationIssue]:
    if not require_timezone or timestamp_col not in data:
        return []
    aware = data[timestamp_col].map(_has_timezone)
    if bool(aware.all()):
        return []
    return [
        ValidationIssue(
            severity="error",
            code="timezone_naive",
            message="One or more timestamps are timezone-naive",
            count=int((~aware).sum()),
        )
    ]


def _validate_duplicates(data: pd.DataFrame, timestamp_col: str) -> list[ValidationIssue]:
    timestamps = _parse_timestamps(data, timestamp_col)
    duplicate_mask = timestamps.duplicated(keep=False)
    if not duplicate_mask.any():
        return []
    return [
        _issue_for_timestamps(
            "error",
            "duplicate_timestamp",
            "Duplicate timestamps detected",
            timestamps[duplicate_mask],
        )
    ]


def _validate_order(data: pd.DataFrame, timestamp_col: str) -> list[ValidationIssue]:
    timestamps = _parse_timestamps(data, timestamp_col)
    if timestamps.empty or timestamps.is_monotonic_increasing:
        return []
    return [
        ValidationIssue(
            severity="error",
            code="non_monotonic_timestamp",
            message="Timestamps are not sorted in increasing order",
            count=int(len(timestamps)),
            first_seen=_timestamp_to_string(timestamps.iloc[0]),
            last_seen=_timestamp_to_string(timestamps.iloc[-1]),
        )
    ]


def _validate_ohlc(
    data: pd.DataFrame, *, columns: Sequence[str] = PRICE_COLUMNS
) -> list[ValidationIssue]:
    missing = [column for column in columns if column not in data]
    if missing:
        return [
            ValidationIssue(
                severity="error",
                code="missing_ohlc_column",
                message=f"Missing OHLC columns: {', '.join(missing)}",
                count=len(missing),
            )
        ]

    open_col, high_col, low_col, close_col = columns
    prices = data[list(columns)].apply(pd.to_numeric, errors="coerce")
    inconsistent = (
        (prices[high_col] < prices[[open_col, close_col, low_col]].max(axis=1))
        | (prices[low_col] > prices[[open_col, close_col, high_col]].min(axis=1))
        | (prices[high_col] < prices[low_col])
    )
    issues: list[ValidationIssue] = []
    if inconsistent.any():
        issues.append(
            ValidationIssue(
                severity="error",
                code="ohlc_inconsistent",
                message="OHLC high/low containment failed",
                count=int(inconsistent.sum()),
            )
        )

    non_positive = prices <= 0
    count = int(non_positive.sum().sum())
    if count:
        issues.append(
            ValidationIssue(
                severity="error",
                code="non_positive_price",
                message="Prices must be positive",
                count=count,
            )
        )
    return issues


def _validate_volume(data: pd.DataFrame, *, volume_col: str = "volume") -> list[ValidationIssue]:
    if volume_col not in data:
        return [
            ValidationIssue(
                severity="warning",
                code="missing_volume",
                message="Volume column is missing; continuing because some instruments lack volume",
            )
        ]
    volume = pd.to_numeric(data[volume_col], errors="coerce")
    issues: list[ValidationIssue] = []
    negative = volume < 0
    if negative.any():
        issues.append(
            ValidationIssue(
                severity="error",
                code="negative_volume",
                message="Volume must be non-negative when present",
                count=int(negative.sum()),
            )
        )

    zero_runs = (volume.fillna(-1) == 0).astype(int)
    run_id = (zero_runs != zero_runs.shift(fill_value=0)).cumsum()
    longest_zero_run = int(zero_runs.groupby(run_id).sum().max()) if not zero_runs.empty else 0
    if longest_zero_run >= 3:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="zero_volume_run",
                message="Suspicious run of zero-volume bars",
                count=longest_zero_run,
            )
        )
    return issues


def _load_market_calendar(market_calendar: str) -> Any:
    try:
        import pandas_market_calendars as mcal
    except ModuleNotFoundError:
        return None
    return mcal.get_calendar(market_calendar)


def _calendar_gap_skipped_issue(timeframe: str) -> ValidationIssue:
    return ValidationIssue(
        severity="info",
        code="calendar_gap_check_skipped",
        message=(
            "Calendar-aware gap checking was skipped because no market calendar was supplied "
            f"for timeframe {timeframe}"
        ),
    )


def _validate_daily_calendar_gaps(
    data: pd.DataFrame, *, timestamp_col: str, timeframe: str, market_calendar: str
) -> list[ValidationIssue]:
    timestamps = _parse_utc_timestamps(data, timestamp_col).dropna().sort_values()
    if timestamps.empty:
        return []
    calendar = _load_market_calendar(market_calendar)
    if calendar is None:
        return [
            ValidationIssue(
                severity="info",
                code="calendar_unavailable",
                message="pandas-market-calendars is not installed; calendar gap check skipped",
            )
        ]
    schedule = calendar.schedule(
        start_date=timestamps.min().date(),
        end_date=timestamps.max().date(),
    )
    expected = pd.DatetimeIndex(pd.to_datetime(schedule.index.date))
    actual = pd.DatetimeIndex(pd.to_datetime(timestamps.dt.date.unique()))
    missing = expected.difference(actual)
    if missing.empty:
        return []
    return [
        ValidationIssue(
            severity="warning",
            code="timestamp_gap",
            message=(
                f"Detected {len(missing)} missing {market_calendar} sessions "
                f"for expected {timeframe} cadence"
            ),
            count=int(len(missing)),
            first_seen=str(missing[0].date()),
            last_seen=str(missing[-1].date()),
        )
    ]


def _validate_intraday_calendar_gaps(
    data: pd.DataFrame,
    *,
    timestamp_col: str,
    timeframe: str,
    market_calendar: str,
) -> list[ValidationIssue]:
    freq = _freq_for_timeframe(timeframe)
    if freq is None:
        return []
    timestamps = _parse_utc_timestamps(data, timestamp_col).dropna().sort_values()
    if timestamps.empty:
        return []
    calendar = _load_market_calendar(market_calendar)
    if calendar is None:
        return [
            ValidationIssue(
                severity="info",
                code="calendar_unavailable",
                message="pandas-market-calendars is not installed; calendar gap check skipped",
            )
        ]
    schedule = calendar.schedule(
        start_date=timestamps.min().date(),
        end_date=timestamps.max().date(),
    )
    expected_chunks: list[pd.DatetimeIndex] = []
    for _, session in schedule.iterrows():
        session_expected = pd.date_range(
            session["market_open"],
            session["market_close"],
            freq=freq,
            inclusive="left",
        )
        expected_chunks.append(session_expected)
    if not expected_chunks:
        return []
    expected = expected_chunks[0]
    for chunk in expected_chunks[1:]:
        expected = pd.DatetimeIndex(expected.append(chunk))
    expected = pd.DatetimeIndex(
        expected[(expected >= timestamps.min()) & (expected <= timestamps.max())]
    )
    actual = pd.DatetimeIndex(timestamps.drop_duplicates())
    missing = expected.difference(actual)
    if missing.empty:
        return []
    return [
        ValidationIssue(
            severity="warning",
            code="timestamp_gap",
            message=(
                f"Detected {len(missing)} missing in-session bars for expected "
                f"{market_calendar} {timeframe} cadence"
            ),
            count=int(len(missing)),
            first_seen=str(missing[0]),
            last_seen=str(missing[-1]),
        )
    ]


def _validate_gaps(
    data: pd.DataFrame,
    *,
    timestamp_col: str,
    timeframe: str,
    market_calendar: str | None,
    strict_fixed_frequency_gaps: bool,
) -> list[ValidationIssue]:
    freq = _freq_for_timeframe(timeframe)
    if freq is None:
        return [
            ValidationIssue(
                severity="info",
                code="unsupported_gap_timeframe",
                message=f"No fixed-frequency gap check for timeframe {timeframe}",
            )
        ]
    if market_calendar is not None:
        if _is_daily_timeframe(timeframe):
            return _validate_daily_calendar_gaps(
                data,
                timestamp_col=timestamp_col,
                timeframe=timeframe,
                market_calendar=market_calendar,
            )
        if _is_intraday_timeframe(timeframe):
            return _validate_intraday_calendar_gaps(
                data,
                timestamp_col=timestamp_col,
                timeframe=timeframe,
                market_calendar=market_calendar,
            )

    if not strict_fixed_frequency_gaps:
        return [_calendar_gap_skipped_issue(timeframe)]

    missing = find_missing_timestamps(data, timestamp_col=timestamp_col, freq=freq)
    if not missing:
        return []
    return [
        ValidationIssue(
            severity="warning",
            code="timestamp_gap",
            message=f"Detected {len(missing)} missing timestamps for expected {timeframe} cadence",
            count=len(missing),
            first_seen=str(missing[0]),
            last_seen=str(missing[-1]),
        )
    ]


def _validate_large_jumps(data: pd.DataFrame, *, threshold: float = 0.2) -> list[ValidationIssue]:
    if "close" not in data:
        return []
    close = pd.to_numeric(data["close"], errors="coerce")
    jumps = close.pct_change().abs() > threshold
    if not jumps.any():
        return []
    return [
        ValidationIssue(
            severity="warning",
            code="large_price_jump",
            message=f"Close-to-close move exceeds {threshold:.0%}",
            count=int(jumps.sum()),
        )
    ]


def _validate_missing_sessions(
    data: pd.DataFrame, *, timestamp_col: str, market_calendar: str | None
) -> list[ValidationIssue]:
    if market_calendar is None or data.empty or timestamp_col not in data:
        return []
    try:
        import pandas_market_calendars as mcal
    except ModuleNotFoundError:
        return [
            ValidationIssue(
                severity="info",
                code="calendar_unavailable",
                message="pandas-market-calendars is not installed; session check skipped",
            )
        ]
    timestamps = _parse_timestamps(data, timestamp_col).dropna()
    if timestamps.empty:
        return []
    calendar = mcal.get_calendar(market_calendar)
    schedule = calendar.schedule(
        start_date=timestamps.min().date(),
        end_date=timestamps.max().date(),
    )
    expected_sessions = pd.DatetimeIndex(schedule.index.date)
    actual_sessions = pd.DatetimeIndex(timestamps.dt.date.unique())
    missing = expected_sessions.difference(actual_sessions)
    if missing.empty:
        return []
    return [
        ValidationIssue(
            severity="warning",
            code="missing_market_session",
            message=f"Dataset is missing {len(missing)} expected {market_calendar} sessions",
            count=int(len(missing)),
            first_seen=str(missing[0].date()),
            last_seen=str(missing[-1].date()),
        )
    ]


def validate_ohlcv(
    frame: Any,
    *,
    timeframe: str,
    timezone: str,
    timestamp_col: str = "timestamp",
    require_timezone: bool = True,
    market_calendar: str | None = None,
    strict_fixed_frequency_gaps: bool = False,
    large_jump_threshold: float = 0.2,
) -> list[ValidationIssue]:
    """Run structured validation checks over an OHLCV dataset."""

    data = _to_pandas(frame)
    issues: list[ValidationIssue] = []
    issues.extend(_validate_timestamp_presence(data, timestamp_col))
    issues.extend(
        _validate_timezone(
            data,
            timestamp_col=timestamp_col,
            require_timezone=require_timezone,
        )
    )
    if timestamp_col in data:
        issues.extend(_validate_duplicates(data, timestamp_col))
        issues.extend(_validate_order(data, timestamp_col))
        issues.extend(
            _validate_gaps(
                data,
                timestamp_col=timestamp_col,
                timeframe=timeframe,
                market_calendar=market_calendar,
                strict_fixed_frequency_gaps=strict_fixed_frequency_gaps,
            )
        )
        issues.extend(
            _validate_missing_sessions(
                data,
                timestamp_col=timestamp_col,
                market_calendar=market_calendar,
            )
        )
    issues.extend(_validate_ohlc(data))
    issues.extend(_validate_volume(data))
    issues.extend(_validate_large_jumps(data, threshold=large_jump_threshold))
    return issues
