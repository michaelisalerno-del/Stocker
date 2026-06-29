"""Research-only intraday feature helpers.

Feature values are known at the current bar close unless a column explicitly marks
itself unavailable. Future templates must still respect the backtest convention
that target positions are applied with a one-bar lag.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

DEFAULT_STAGE4_1_OUTPUT_DIR = Path(
    "data/reports/research/stage4_1_intraday_feature_audit"
)
DEFAULT_STAGE4_1_UNIVERSE = Path(
    "data/universes/research_ready/us_liquid_25_5m_intraday.json"
)
KEY_FEATURE_COLUMNS = [
    "opening_range_high",
    "opening_range_low",
    "opening_range_mid",
    "opening_range_width",
    "session_vwap",
    "distance_from_vwap",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
    "previous_session_high",
    "previous_session_low",
    "previous_session_close",
    "gap_vs_previous_close",
    "rolling_intraday_range",
    "rolling_intraday_range_pct",
    "compression_zscore",
    "recent_high",
    "recent_low",
]
TIME_WINDOW_FLAG_COLUMNS = [
    "after_open_buffer",
    "before_close_cutoff",
    "in_opening_range_window",
    "in_midday_window",
    "in_late_day_window",
    "no_entry_window",
    "can_enter_after_open_buffer",
    "can_enter_before_close_cutoff",
    "can_open_new_position",
    "must_flatten_now",
]


@dataclass(frozen=True)
class IntradayFeatureConfig:
    """Configuration for deterministic intraday feature generation."""

    timeframe: str = "5m"
    market_calendar: str | None = "XNYS"
    opening_minutes: int = 30
    open_buffer_minutes: int = 15
    entry_cutoff_before_close_minutes: int = 30
    flatten_before_close_minutes: int = 10
    midday_start_minutes: int = 120
    midday_end_minutes: int = 300
    late_day_start_minutes: int = 300
    relative_volume_lookback_sessions: int = 20
    range_lookback_bars: int = 12
    compression_lookback_bars: int = 20
    vwap_price: str = "typical"


@dataclass(frozen=True)
class IntradayFeatureReport:
    """Feature-generation quality summary for one frame."""

    row_count: int
    session_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    incomplete_session_count: int
    timestamp_grid_anomaly_count: int
    warning_reasons: dict[str, int] = field(default_factory=dict)
    null_rates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntradayFeatureAuditResult:
    """Paths and headline counts from the Stage 4.1 feature audit."""

    summary_json_path: Path
    summary_markdown_path: Path
    feature_availability_csv_path: Path
    session_feature_quality_csv_path: Path
    feature_null_rates_csv_path: Path
    symbol_count: int
    feature_availability_summary: dict[str, Any]
    null_rate_summary: dict[str, Any]
    session_warning_summary: dict[str, Any]
    stage_passed: bool


def _timeframe_minutes(timeframe: str) -> int:
    normalized = timeframe.strip().lower()
    if normalized.endswith("m") and normalized[:-1].isdigit():
        return int(normalized[:-1])
    if normalized.endswith("min") and normalized[:-3].isdigit():
        return int(normalized[:-3])
    if normalized.endswith("h") and normalized[:-1].isdigit():
        return int(normalized[:-1]) * 60
    raise ValueError(f"Unsupported intraday timeframe: {timeframe}")


def _is_intraday_timeframe(timeframe: str) -> bool:
    normalized = timeframe.strip().lower()
    return (
        (normalized.endswith("m") and normalized[:-1].isdigit())
        or (normalized.endswith("min") and normalized[:-3].isdigit())
        or (normalized.endswith("h") and normalized[:-1].isdigit())
    )


def _require_intraday_timeframe(timeframe: str) -> None:
    if not _is_intraday_timeframe(timeframe):
        raise ValueError("Intraday features require an intraday timeframe.")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame:
        raise ValueError("Intraday features require a timestamp column.")
    data = frame.copy().reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    return data


def _load_calendar_schedule(
    market_calendar: str | None,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, str | None]:
    if not market_calendar:
        return pd.DataFrame(), "market_calendar_not_provided"
    try:
        import pandas_market_calendars as mcal
    except ModuleNotFoundError:
        return pd.DataFrame(), "pandas_market_calendars_unavailable"
    try:
        calendar = mcal.get_calendar(market_calendar)
        schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    except Exception:
        return pd.DataFrame(), "market_calendar_load_failed"
    if schedule.empty:
        return pd.DataFrame(), "market_calendar_schedule_empty"
    return schedule, None


def _expected_timestamps(
    market_open: pd.Timestamp,
    market_close: pd.Timestamp,
    *,
    timeframe: str,
) -> set[pd.Timestamp]:
    minutes = _timeframe_minutes(timeframe)
    return set(pd.date_range(market_open, market_close, freq=f"{minutes}min"))


def _session_id_from_label(session_label: str) -> int:
    return pd.Timestamp(session_label).date().toordinal()


def _session_labels(timestamps: pd.Series) -> pd.Series:
    return timestamps.dt.date.astype(str)


def add_session_clock_features(
    frame: pd.DataFrame,
    market_calendar: str | None = None,
    *,
    timeframe: str = "5m",
) -> pd.DataFrame:
    """Add session identity and clock columns.

    With a market calendar, session opens/closes come from the exchange schedule.
    Without one, observed first/last timestamps are used as a conservative proxy
    and the limitation is marked in `session_warning_reason`.
    """

    _require_intraday_timeframe(timeframe)
    data = _prepare_frame(frame)
    if data.empty:
        return data

    timestamps = data["timestamp"]
    schedule, calendar_error = _load_calendar_schedule(
        market_calendar,
        timestamps.min().date(),
        timestamps.max().date(),
    )
    schedule_by_date: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    if not schedule.empty:
        for session_date, row in schedule.iterrows():
            label = str(pd.Timestamp(session_date).date())
            schedule_by_date[label] = (
                pd.Timestamp(row["market_open"]).tz_convert("UTC"),
                pd.Timestamp(row["market_close"]).tz_convert("UTC"),
            )

    data["session_date"] = _session_labels(data["timestamp"])
    data["session_id"] = data["session_date"].map(_session_id_from_label).astype(int)
    data["bar_index_in_session"] = data.groupby("session_date").cumcount().astype(int)
    session_sizes = data.groupby("session_date")["timestamp"].transform("size").astype(int)
    data["bars_remaining_in_session"] = (
        session_sizes - data["bar_index_in_session"] - 1
    ).astype(int)
    data["is_first_bar_of_session"] = data["bar_index_in_session"].eq(0)
    data["is_last_bar_of_session"] = data["bars_remaining_in_session"].eq(0)

    data["calendar_session_open"] = pd.Series(
        pd.NaT,
        index=data.index,
        dtype="datetime64[ns, UTC]",
    )
    data["calendar_session_close"] = pd.Series(
        pd.NaT,
        index=data.index,
        dtype="datetime64[ns, UTC]",
    )
    data["session_bar_count"] = session_sizes.astype(int)
    data["session_expected_bar_count"] = 0
    data["session_missing_bar_count"] = 0
    data["session_extra_bar_count"] = 0
    data["session_complete_warning"] = False
    data["session_warning_reason"] = ""
    data["is_regular_session_bar"] = False

    for session_label, indices in data.groupby("session_date").groups.items():
        group_index = list(indices)
        session_timestamps = data.loc[group_index, "timestamp"]
        observed_open = pd.Timestamp(session_timestamps.min())
        observed_close = pd.Timestamp(session_timestamps.max())
        schedule_entry = schedule_by_date.get(str(session_label))
        if schedule_entry is None:
            session_open = observed_open
            session_close = observed_close
            warning_reason = calendar_error or "missing_calendar_session"
        else:
            session_open, session_close = schedule_entry
            warning_reason = ""

        expected = _expected_timestamps(session_open, session_close, timeframe=timeframe)
        actual = {pd.Timestamp(timestamp) for timestamp in session_timestamps}
        missing = expected - actual
        extra = actual - expected
        incomplete = bool(missing or extra or warning_reason)
        if not warning_reason and extra:
            warning_reason = "timestamp_grid_anomaly"
        elif not warning_reason and missing:
            warning_reason = "incomplete_session"

        regular_flags = [
            timestamp in expected
            and timestamp >= session_open
            and timestamp <= session_close
            for timestamp in session_timestamps
        ]
        data.loc[group_index, "calendar_session_open"] = session_open
        data.loc[group_index, "calendar_session_close"] = session_close
        data.loc[group_index, "session_expected_bar_count"] = len(expected)
        data.loc[group_index, "session_missing_bar_count"] = len(missing)
        data.loc[group_index, "session_extra_bar_count"] = len(extra)
        data.loc[group_index, "session_complete_warning"] = incomplete
        data.loc[group_index, "session_warning_reason"] = warning_reason
        data.loc[group_index, "is_regular_session_bar"] = regular_flags

    data["minutes_from_session_open"] = (
        data["timestamp"] - pd.to_datetime(data["calendar_session_open"], utc=True)
    ).dt.total_seconds() / 60.0
    data["minutes_to_session_close"] = (
        pd.to_datetime(data["calendar_session_close"], utc=True) - data["timestamp"]
    ).dt.total_seconds() / 60.0
    return data


def add_time_window_flags(
    frame: pd.DataFrame,
    config: IntradayFeatureConfig | None = None,
) -> pd.DataFrame:
    """Add reusable time-of-day and entry-window flags."""

    cfg = config or IntradayFeatureConfig()
    data = frame.copy()
    if "minutes_from_session_open" not in data or "minutes_to_session_close" not in data:
        data = add_session_clock_features(
            data,
            market_calendar=cfg.market_calendar,
            timeframe=cfg.timeframe,
        )
    minutes_from_open = pd.to_numeric(data["minutes_from_session_open"], errors="coerce")
    minutes_to_close = pd.to_numeric(data["minutes_to_session_close"], errors="coerce")
    regular = data.get("is_regular_session_bar", pd.Series(True, index=data.index)).astype(bool)
    data["after_open_buffer"] = minutes_from_open >= cfg.open_buffer_minutes
    data["before_close_cutoff"] = minutes_to_close > cfg.entry_cutoff_before_close_minutes
    data["in_opening_range_window"] = (
        (minutes_from_open >= 0) & (minutes_from_open < cfg.opening_minutes)
    )
    data["in_midday_window"] = (
        (minutes_from_open >= cfg.midday_start_minutes)
        & (minutes_from_open < cfg.midday_end_minutes)
    )
    data["in_late_day_window"] = minutes_from_open >= cfg.late_day_start_minutes
    data["no_entry_window"] = minutes_to_close <= cfg.entry_cutoff_before_close_minutes
    data["can_enter_after_open_buffer"] = data["after_open_buffer"] & regular
    data["can_enter_before_close_cutoff"] = data["before_close_cutoff"] & regular
    data["must_flatten_now"] = minutes_to_close <= cfg.flatten_before_close_minutes
    data["can_open_new_position"] = (
        data["can_enter_after_open_buffer"]
        & data["can_enter_before_close_cutoff"]
        & ~data["must_flatten_now"]
    )
    return data


def add_opening_range_features(
    frame: pd.DataFrame,
    opening_minutes: int = 30,
    *,
    timeframe: str = "5m",
) -> pd.DataFrame:
    """Add opening-range high/low columns known after the range window closes."""

    data = frame.copy()
    if "session_date" not in data:
        data = add_session_clock_features(data, timeframe=timeframe)
    minutes = _timeframe_minutes(timeframe)
    expected_opening_bars = max(1, math.ceil(opening_minutes / minutes))
    data["opening_range_high"] = math.nan
    data["opening_range_low"] = math.nan
    data["opening_range_mid"] = math.nan
    data["opening_range_width"] = math.nan
    data["opening_range_complete"] = False

    for _, indices in data.groupby("session_date").groups.items():
        group_index = list(indices)
        group = data.loc[group_index]
        opening_mask = (
            (pd.to_numeric(group["minutes_from_session_open"], errors="coerce") >= 0)
            & (pd.to_numeric(group["minutes_from_session_open"], errors="coerce") < opening_minutes)
            & group["is_regular_session_bar"].astype(bool)
        )
        opening_rows = group.loc[opening_mask]
        range_available = len(opening_rows) >= expected_opening_bars
        after_window = pd.to_numeric(group["minutes_from_session_open"], errors="coerce") >= (
            expected_opening_bars * minutes
        )
        complete_indices = list(group.loc[after_window & range_available].index)
        if not complete_indices:
            continue
        high = float(opening_rows["high"].max())
        low = float(opening_rows["low"].min())
        data.loc[complete_indices, "opening_range_high"] = high
        data.loc[complete_indices, "opening_range_low"] = low
        data.loc[complete_indices, "opening_range_mid"] = (high + low) / 2.0
        data.loc[complete_indices, "opening_range_width"] = high - low
        data.loc[complete_indices, "opening_range_complete"] = True
    return data


def add_session_vwap(frame: pd.DataFrame, *, price: str = "typical") -> pd.DataFrame:
    """Add session VWAP columns, resetting the cumulative calculation each session."""

    data = frame.copy()
    if "session_date" not in data:
        data = add_session_clock_features(data)
    volume = pd.to_numeric(data.get("volume", 0.0), errors="coerce").clip(lower=0)
    volume_for_sum = volume.fillna(0.0)
    if price == "close":
        vwap_price = pd.to_numeric(data["close"], errors="coerce")
    elif price == "typical":
        vwap_price = (
            pd.to_numeric(data["high"], errors="coerce")
            + pd.to_numeric(data["low"], errors="coerce")
            + pd.to_numeric(data["close"], errors="coerce")
        ) / 3.0
    else:
        raise ValueError("vwap price must be 'typical' or 'close'.")

    data["volume"] = volume
    data["cumulative_session_volume"] = volume_for_sum.groupby(data["session_date"]).cumsum()
    cumulative_price_volume = (vwap_price * volume_for_sum).groupby(data["session_date"]).cumsum()
    data["session_vwap"] = cumulative_price_volume / data["cumulative_session_volume"]
    data.loc[data["cumulative_session_volume"] <= 0.0, "session_vwap"] = math.nan
    data["distance_from_vwap"] = (
        pd.to_numeric(data["close"], errors="coerce") / data["session_vwap"] - 1.0
    )
    data["above_vwap"] = pd.to_numeric(data["close"], errors="coerce") > data["session_vwap"]
    data["below_vwap"] = pd.to_numeric(data["close"], errors="coerce") < data["session_vwap"]
    data.loc[data["session_vwap"].isna(), ["above_vwap", "below_vwap"]] = False
    return data


def add_relative_volume_features(
    frame: pd.DataFrame,
    lookback_sessions: int = 20,
) -> pd.DataFrame:
    """Compare current bar volume with prior sessions at the same bar index."""

    data = frame.copy()
    if "cumulative_session_volume" not in data:
        data = add_session_vwap(data)
    volume = pd.to_numeric(data.get("volume", 0.0), errors="coerce").clip(lower=0)
    data["volume"] = volume
    data["cumulative_session_volume"] = volume.fillna(0.0).groupby(data["session_date"]).cumsum()

    def prior_mean(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(
            lookback_sessions,
            min_periods=lookback_sessions,
        ).mean()

    baseline_bar_volume = data.groupby("bar_index_in_session")["volume"].transform(prior_mean)
    baseline_cumulative_volume = data.groupby("bar_index_in_session")[
        "cumulative_session_volume"
    ].transform(prior_mean)
    data["relative_volume_at_bar_index"] = data["volume"] / baseline_bar_volume
    data["relative_cumulative_volume"] = (
        data["cumulative_session_volume"] / baseline_cumulative_volume
    )
    data.loc[baseline_bar_volume <= 0.0, "relative_volume_at_bar_index"] = math.nan
    data.loc[baseline_cumulative_volume <= 0.0, "relative_cumulative_volume"] = math.nan
    data["relative_volume_available"] = data["relative_volume_at_bar_index"].notna()
    data["relative_cumulative_volume_available"] = data["relative_cumulative_volume"].notna()
    return data


def add_previous_session_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add previous-session high/low/close and gap columns using prior sessions only."""

    data = frame.copy()
    if "session_date" not in data:
        data = add_session_clock_features(data)
    session_order = (
        data.groupby("session_date", sort=True)["timestamp"]
        .min()
        .sort_values()
        .index.tolist()
    )
    high_by_session = data.groupby("session_date")["high"].max().reindex(session_order)
    low_by_session = data.groupby("session_date")["low"].min().reindex(session_order)
    close_by_session = data.groupby("session_date")["close"].last().reindex(session_order)
    open_by_session = data.groupby("session_date")["open"].first().reindex(session_order)
    prev_high = high_by_session.shift(1)
    prev_low = low_by_session.shift(1)
    prev_close = close_by_session.shift(1)

    data["previous_session_high"] = data["session_date"].map(prev_high.to_dict())
    data["previous_session_low"] = data["session_date"].map(prev_low.to_dict())
    data["previous_session_close"] = data["session_date"].map(prev_close.to_dict())
    session_open = data["session_date"].map(open_by_session.to_dict())
    data["gap_vs_previous_close"] = (
        session_open - data["previous_session_close"]
    ) / data["previous_session_close"]
    data["open_above_previous_high"] = session_open > data["previous_session_high"]
    data["open_below_previous_low"] = session_open < data["previous_session_low"]
    data.loc[
        data["previous_session_close"].isna(),
        ["open_above_previous_high", "open_below_previous_low"],
    ] = False
    return data


def _rolling_by_session(
    data: pd.DataFrame,
    column: str,
    window: int,
    method: str,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    periods = window if min_periods is None else min_periods
    grouped = data.groupby("session_date", group_keys=False)[column]
    if method == "max":
        return grouped.apply(lambda series: series.rolling(window, min_periods=periods).max())
    if method == "min":
        return grouped.apply(lambda series: series.rolling(window, min_periods=periods).min())
    if method == "mean":
        return grouped.apply(lambda series: series.rolling(window, min_periods=periods).mean())
    if method == "std":
        return grouped.apply(lambda series: series.rolling(window, min_periods=periods).std())
    raise ValueError(f"Unsupported rolling method: {method}")


def add_intraday_range_compression_features(
    frame: pd.DataFrame,
    *,
    range_lookback_bars: int = 12,
    compression_lookback_bars: int = 20,
) -> pd.DataFrame:
    """Add rolling intraday range and compression z-score helpers."""

    data = frame.copy()
    if "session_date" not in data:
        data = add_session_clock_features(data)
    data["recent_high"] = _rolling_by_session(
        data,
        "high",
        range_lookback_bars,
        "max",
    )
    data["recent_low"] = _rolling_by_session(
        data,
        "low",
        range_lookback_bars,
        "min",
    )
    data["rolling_intraday_range"] = data["recent_high"] - data["recent_low"]
    data["rolling_intraday_range_pct"] = (
        data["rolling_intraday_range"] / pd.to_numeric(data["close"], errors="coerce")
    )
    rolling_mean = _rolling_by_session(
        data,
        "rolling_intraday_range",
        compression_lookback_bars,
        "mean",
        min_periods=2,
    )
    rolling_std = _rolling_by_session(
        data,
        "rolling_intraday_range",
        compression_lookback_bars,
        "std",
        min_periods=2,
    )
    data["compression_zscore"] = (data["rolling_intraday_range"] - rolling_mean) / rolling_std
    data.loc[rolling_std <= 0.0, "compression_zscore"] = math.nan
    return data


def add_minimum_bars_after_open_flags(
    frame: pd.DataFrame,
    min_bars_after_open: int,
) -> pd.DataFrame:
    """Add a reusable boolean for templates that need open warmup bars."""

    data = frame.copy()
    if "bar_index_in_session" not in data:
        data = add_session_clock_features(data)
    data["can_enter_after_minimum_bars"] = (
        pd.to_numeric(data["bar_index_in_session"], errors="coerce") >= min_bars_after_open
    )
    return data


def add_time_stop_features(
    frame: pd.DataFrame,
    *,
    entries: pd.Series,
    max_bars_held: int,
) -> pd.DataFrame:
    """Add generic bars-since-entry and time-stop columns for future templates."""

    data = frame.copy().reset_index(drop=True)
    entry_flags = entries.reset_index(drop=True).astype(bool).reindex(data.index).fillna(False)
    bars_since_entry: list[float] = []
    current_count: int | None = None
    for is_entry in entry_flags:
        if is_entry:
            current_count = 0
        elif current_count is not None:
            current_count += 1
        bars_since_entry.append(math.nan if current_count is None else float(current_count))
    data["bars_since_entry"] = bars_since_entry
    data["time_stop_reached"] = pd.Series(bars_since_entry).ge(float(max_bars_held)).fillna(False)
    return data


def build_intraday_feature_frame(
    frame: pd.DataFrame,
    config: IntradayFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build the standard reusable intraday feature frame."""

    cfg = config or IntradayFeatureConfig()
    _require_intraday_timeframe(cfg.timeframe)
    data = add_session_clock_features(
        frame,
        market_calendar=cfg.market_calendar,
        timeframe=cfg.timeframe,
    )
    data = add_time_window_flags(data, cfg)
    data = add_opening_range_features(
        data,
        opening_minutes=cfg.opening_minutes,
        timeframe=cfg.timeframe,
    )
    data = add_session_vwap(data, price=cfg.vwap_price)
    data = add_relative_volume_features(
        data,
        lookback_sessions=cfg.relative_volume_lookback_sessions,
    )
    data = add_previous_session_levels(data)
    data = add_intraday_range_compression_features(
        data,
        range_lookback_bars=cfg.range_lookback_bars,
        compression_lookback_bars=cfg.compression_lookback_bars,
    )
    return data


def summarize_intraday_feature_frame(features: pd.DataFrame) -> IntradayFeatureReport:
    """Summarize availability and warning rates for a feature frame."""

    if features.empty:
        return IntradayFeatureReport(
            row_count=0,
            session_count=0,
            first_timestamp=None,
            last_timestamp=None,
            incomplete_session_count=0,
            timestamp_grid_anomaly_count=0,
            warning_reasons={},
            null_rates={},
        )
    session_warnings = (
        features.groupby("session_date")["session_complete_warning"].max()
        if "session_complete_warning" in features
        else pd.Series(dtype=bool)
    )
    warning_reasons: Counter[str] = Counter()
    if "session_warning_reason" in features:
        for _, group in features.groupby("session_date"):
            reasons = sorted({str(reason) for reason in group["session_warning_reason"] if reason})
            for reason in reasons:
                warning_reasons[reason] += 1
    anomaly_sessions = (
        features.groupby("session_date")["session_extra_bar_count"].max()
        if "session_extra_bar_count" in features
        else pd.Series(dtype=int)
    )
    null_rates = {
        column: float(features[column].isna().mean())
        for column in KEY_FEATURE_COLUMNS
        if column in features
    }
    timestamps = pd.to_datetime(features["timestamp"], utc=True)
    return IntradayFeatureReport(
        row_count=int(len(features)),
        session_count=int(features["session_date"].nunique()),
        first_timestamp=str(timestamps.min()),
        last_timestamp=str(timestamps.max()),
        incomplete_session_count=int(session_warnings.sum()),
        timestamp_grid_anomaly_count=int((anomaly_sessions > 0).sum()),
        warning_reasons=dict(warning_reasons),
        null_rates=null_rates,
    )


def _load_qualified_symbols(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = payload.get("qualified_symbols", [])
    output: list[str] = []
    for item in symbols:
        if isinstance(item, dict) and item.get("symbol"):
            output.append(str(item["symbol"]))
        elif isinstance(item, str):
            output.append(item)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _availability_pct(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _mean_dict(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return float(mean(values)) if values else None


def _markdown(summary: dict[str, Any]) -> str:
    availability = summary["feature_availability_summary"]
    null_rates = summary["null_rate_summary"]
    session_warnings = summary["session_warning_summary"]
    actual_range_from = summary["actual_available_range"]["from"]
    actual_range_to = summary["actual_available_range"]["to"]
    previous_level_availability = availability[
        "previous_session_levels_availability_mean"
    ]
    null_lines = "\n".join(
        f"- `{column}`: {rate:.4f}"
        for column, rate in null_rates["mean_null_rates"].items()
    )
    if not null_lines:
        null_lines = "- None"
    warning_lines = "\n".join(
        f"- `{reason}`: {count}"
        for reason, count in session_warnings["warning_reasons"].items()
    )
    if not warning_lines:
        warning_lines = "- None"
    return f"""# Stage 4.1 Intraday Feature Audit

This audit generated research-only intraday features over the existing 25-symbol
5m Stage 3.9 universe. It did not fetch data, run strategies, create candidates,
or change any research gates.

- Symbols inspected: {summary["symbol_count"]}
- Feature-generation errors: {summary["feature_error_count"]}
- Timeframe: `{summary["timeframe"]}`
- Market calendar: `{summary["market_calendar"]}`
- Actual range: {actual_range_from} to {actual_range_to}

## Availability

- Opening range complete mean: {availability["opening_range_complete_mean"]:.4f}
- Session VWAP availability mean: {availability["session_vwap_availability_mean"]:.4f}
- Relative volume availability mean: {availability["relative_volume_availability_mean"]:.4f}
- Previous session levels availability mean: {previous_level_availability:.4f}
- Time-window flag availability mean: {availability["time_window_flag_availability_mean"]:.4f}

## Null Rates

{null_lines}

## Session Warnings

- Incomplete/nonstandard sessions: {session_warnings["incomplete_session_count"]}
- Timestamp-grid anomaly sessions: {session_warnings["timestamp_grid_anomaly_count"]}

{warning_lines}

## Interpretation

The feature layer is deterministic and session-aware. Opening range values are
unavailable until the opening window has completed, VWAP resets each session,
relative volume uses prior sessions only, and previous-session levels are mapped
from completed prior sessions only.

Stage 4.1 passes if tests pass and these audit files are present. The next step
is Stage 4.2: implement the first intraday template family using this feature
layer, with opening range breakout continuation/failure still the most practical
first candidate.
"""


def build_intraday_feature_audit(
    *,
    data_dir: str | Path = "data",
    universe_path: str | Path = DEFAULT_STAGE4_1_UNIVERSE,
    output_dir: str | Path = DEFAULT_STAGE4_1_OUTPUT_DIR,
    source: str = "eodhd",
    instrument_type: str = "stock",
    timeframe: str = "5m",
    market_calendar: str | None = "XNYS",
) -> IntradayFeatureAuditResult:
    """Build Stage 4.1 feature availability audit from existing local data."""

    from stocker_data.storage import DatasetKey, load_dataset

    config = IntradayFeatureConfig(timeframe=timeframe, market_calendar=market_calendar)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    symbols = _load_qualified_symbols(Path(universe_path))
    availability_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    first_timestamps: list[pd.Timestamp] = []
    last_timestamps: list[pd.Timestamp] = []
    warning_counter: Counter[str] = Counter()
    incomplete_session_total = 0
    anomaly_session_total = 0

    for symbol in symbols:
        try:
            frame = load_dataset(
                DatasetKey(
                    source=source,
                    instrument_type=instrument_type,
                    symbol=symbol,
                    timeframe=timeframe,
                ),
                data_dir=data_dir,
            )
            features = build_intraday_feature_frame(frame, config)
            report = summarize_intraday_feature_frame(features)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})
            continue

        timestamps = pd.to_datetime(features["timestamp"], utc=True)
        first_timestamps.append(timestamps.min())
        last_timestamps.append(timestamps.max())
        warning_counter.update(report.warning_reasons)
        incomplete_session_total += report.incomplete_session_count
        anomaly_session_total += report.timestamp_grid_anomaly_count

        availability_rows.append(
            {
                "symbol": symbol,
                "row_count": report.row_count,
                "session_count": report.session_count,
                "first_timestamp": report.first_timestamp,
                "last_timestamp": report.last_timestamp,
                "opening_range_complete_pct": float(features["opening_range_complete"].mean()),
                "session_vwap_availability_pct": float(features["session_vwap"].notna().mean()),
                "relative_volume_availability_pct": float(
                    features["relative_volume_at_bar_index"].notna().mean()
                ),
                "previous_session_levels_availability_pct": float(
                    features["previous_session_close"].notna().mean()
                ),
                "time_window_flag_availability_pct": float(
                    features[TIME_WINDOW_FLAG_COLUMNS].notna().all(axis=1).mean()
                ),
                "incomplete_nonstandard_session_count": report.incomplete_session_count,
                "timestamp_grid_anomaly_count": report.timestamp_grid_anomaly_count,
                "feature_generation_error": "",
            }
        )
        for column in KEY_FEATURE_COLUMNS:
            if column in features:
                null_rows.append(
                    {
                        "symbol": symbol,
                        "feature": column,
                        "null_rate": float(features[column].isna().mean()),
                    }
                )
        for session_date, group in features.groupby("session_date"):
            session_warning_reasons = sorted(
                {
                    str(reason)
                    for reason in group["session_warning_reason"]
                    if reason
                }
            )
            session_rows.append(
                {
                    "symbol": symbol,
                    "session_date": session_date,
                    "row_count": int(len(group)),
                    "expected_bar_count": int(group["session_expected_bar_count"].max()),
                    "opening_range_complete_rows": int(group["opening_range_complete"].sum()),
                    "session_vwap_null_count": int(group["session_vwap"].isna().sum()),
                    "relative_volume_null_count": int(
                        group["relative_volume_at_bar_index"].isna().sum()
                    ),
                    "session_complete_warning": bool(group["session_complete_warning"].max()),
                    "session_warning_reason": "|".join(session_warning_reasons),
                    "timestamp_grid_anomaly_count": int(group["session_extra_bar_count"].max()),
                }
            )

    for error in errors:
        availability_rows.append(
            {
                "symbol": error["symbol"],
                "row_count": 0,
                "session_count": 0,
                "first_timestamp": "",
                "last_timestamp": "",
                "opening_range_complete_pct": 0.0,
                "session_vwap_availability_pct": 0.0,
                "relative_volume_availability_pct": 0.0,
                "previous_session_levels_availability_pct": 0.0,
                "time_window_flag_availability_pct": 0.0,
                "incomplete_nonstandard_session_count": 0,
                "timestamp_grid_anomaly_count": 0,
                "feature_generation_error": error["error"],
            }
        )

    mean_null_rates: dict[str, float] = {}
    for column in KEY_FEATURE_COLUMNS:
        values = [
            float(row["null_rate"])
            for row in null_rows
            if row["feature"] == column
        ]
        if values:
            mean_null_rates[column] = float(mean(values))

    availability_summary = {
        "opening_range_complete_mean": _mean_dict(
            availability_rows,
            "opening_range_complete_pct",
        )
        or 0.0,
        "session_vwap_availability_mean": _mean_dict(
            availability_rows,
            "session_vwap_availability_pct",
        )
        or 0.0,
        "relative_volume_availability_mean": _mean_dict(
            availability_rows,
            "relative_volume_availability_pct",
        )
        or 0.0,
        "previous_session_levels_availability_mean": _mean_dict(
            availability_rows,
            "previous_session_levels_availability_pct",
        )
        or 0.0,
        "time_window_flag_availability_mean": _mean_dict(
            availability_rows,
            "time_window_flag_availability_pct",
        )
        or 0.0,
    }
    null_rate_summary = {
        "mean_null_rates": mean_null_rates,
        "highest_mean_null_rates": dict(
            sorted(mean_null_rates.items(), key=lambda item: item[1], reverse=True)[:10]
        ),
    }
    session_warning_summary = {
        "incomplete_session_count": incomplete_session_total,
        "timestamp_grid_anomaly_count": anomaly_session_total,
        "warning_reasons": dict(warning_counter),
    }
    summary: dict[str, Any] = {
        "stage": "4.1_intraday_feature_audit",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "data_fetched": False,
        "timeframe": timeframe,
        "market_calendar": market_calendar,
        "universe_path": str(universe_path),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "symbols_audited": [
            row["symbol"]
            for row in availability_rows
            if not row["feature_generation_error"]
        ],
        "feature_error_count": len(errors),
        "feature_generation_errors": errors,
        "actual_available_range": {
            "from": str(min(first_timestamps)) if first_timestamps else None,
            "to": str(max(last_timestamps)) if last_timestamps else None,
        },
        "feature_availability_summary": availability_summary,
        "null_rate_summary": null_rate_summary,
        "session_warning_summary": session_warning_summary,
        "feature_functions": [
            "add_session_clock_features",
            "add_time_window_flags",
            "add_opening_range_features",
            "add_session_vwap",
            "add_relative_volume_features",
            "add_previous_session_levels",
            "add_intraday_range_compression_features",
            "add_minimum_bars_after_open_flags",
            "add_time_stop_features",
            "build_intraday_feature_frame",
            "summarize_intraday_feature_frame",
        ],
        "stage_passed": len(errors) == 0,
        "notes": [
            "This audit did not run strategies or create candidates.",
            "Opening range values are unavailable until the configured window is complete.",
            "Relative volume uses prior sessions at the same bar index only.",
            "Session warnings are surfaced explicitly rather than ignored.",
        ],
    }

    summary_json_path = output_path / "summary.json"
    summary_markdown_path = output_path / "summary.md"
    availability_csv_path = output_path / "feature_availability_by_symbol.csv"
    session_quality_csv_path = output_path / "session_feature_quality.csv"
    null_rates_csv_path = output_path / "feature_null_rates.csv"
    _write_csv(
        availability_csv_path,
        availability_rows,
        [
            "symbol",
            "row_count",
            "session_count",
            "first_timestamp",
            "last_timestamp",
            "opening_range_complete_pct",
            "session_vwap_availability_pct",
            "relative_volume_availability_pct",
            "previous_session_levels_availability_pct",
            "time_window_flag_availability_pct",
            "incomplete_nonstandard_session_count",
            "timestamp_grid_anomaly_count",
            "feature_generation_error",
        ],
    )
    _write_csv(
        session_quality_csv_path,
        session_rows,
        [
            "symbol",
            "session_date",
            "row_count",
            "expected_bar_count",
            "opening_range_complete_rows",
            "session_vwap_null_count",
            "relative_volume_null_count",
            "session_complete_warning",
            "session_warning_reason",
            "timestamp_grid_anomaly_count",
        ],
    )
    _write_csv(null_rates_csv_path, null_rows, ["symbol", "feature", "null_rate"])
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    summary_markdown_path.write_text(_markdown(summary), encoding="utf-8")
    return IntradayFeatureAuditResult(
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        feature_availability_csv_path=availability_csv_path,
        session_feature_quality_csv_path=session_quality_csv_path,
        feature_null_rates_csv_path=null_rates_csv_path,
        symbol_count=len(symbols),
        feature_availability_summary=availability_summary,
        null_rate_summary=null_rate_summary,
        session_warning_summary=session_warning_summary,
        stage_passed=len(errors) == 0,
    )
