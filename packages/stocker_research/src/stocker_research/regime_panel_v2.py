"""Archived deterministic panel builder for repaired causal regime research V2.

The builder reconstructs the fourteen combined stock/market emission features
from bounded UTC OHLCV sources. It makes completed-bar availability, exchange
sessions, source gaps, peer availability, ordering, and source identity
explicit. No economic outcome or execution surface is available here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from stocker_research.regime_gap_segmentation_v2 import annotate_causal_segments

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

BAR_DURATION = pd.Timedelta(minutes=5)
REGULAR_SESSION_MAX_BARS = 78
NATURAL_KEY = ("symbol", "session", "bar_start_timestamp", "bar_ordinal")
STOCK_EMISSION_FEATURES = (
    "regime_log_activity_3",
    "regime_log_activity_12",
    "regime_activity_acceleration",
    "signed_efficiency_6",
    "signed_efficiency_12",
    "regime_log_bar_range",
    "close_location_value",
    "regime_wick_balance",
    "log_relative_historical_volume",
    "log_relative_cumulative_historical_volume",
)
MARKET_EMISSION_FEATURES = (
    "regime_log_market_dispersion",
    "vti__signed_efficiency_12",
    "regime_market_breadth_centered",
)
STOCK_RELATIVE_EMISSION_FEATURES = ("regime_stock_minus_market_scaled",)
EMISSION_FEATURES = (
    *STOCK_EMISSION_FEATURES,
    "regime_log_market_dispersion",
    "regime_stock_minus_market_scaled",
    "vti__signed_efficiency_12",
    "regime_market_breadth_centered",
)


@dataclass(frozen=True, slots=True)
class RegimePanelConfig:
    """Immutable bounded source and universe configuration."""

    provider_root: Path
    symbols: tuple[str, ...]
    benchmark_symbol: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("panel requires at least one stock symbol")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("panel symbols must be unique")
        if self.benchmark_symbol in self.symbols:
            raise ValueError("benchmark must be separate from stock symbols")
        start = pd.Timestamp(self.start)
        end = pd.Timestamp(self.end)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("panel bounds must be timezone-aware")
        if start >= end:
            raise ValueError("panel start must precede panel end")


@dataclass(frozen=True, slots=True)
class PanelBuildResult:
    """Complete deterministic panel and its identity surfaces."""

    frame: pd.DataFrame
    gap_ledger: pd.DataFrame
    source_hashes: dict[str, str]
    source_row_counts: dict[str, int]
    data_snapshot_hash: str
    row_key_hash: str
    feature_table_hash: str
    session_expected_bars: dict[str, int]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def provider_path(root: Path, symbol: str) -> Path:
    """Resolve the committed provider convention without implicit discovery."""

    stored = "VTI.US" if symbol == "VTI" else symbol
    return root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def _bounded_source_frame(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("timestamp", ">=", pd.Timestamp(start).to_pydatetime()),
            ("timestamp", "<=", pd.Timestamp(end).to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].lt(start).any() or frame["timestamp"].gt(end).any():
        raise AssertionError("bounded source reader admitted an out-of-period row")
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _arrow_hash(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return _sha256_bytes(sink.getvalue().to_pybytes())


def bounded_source_hash(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, int]:
    """Hash the exact bounded provider rows in deterministic Arrow form."""

    frame = _bounded_source_frame(path, start=start, end=end)
    return _arrow_hash(frame), len(frame)


def verify_source_hashes(
    sources: Mapping[str, Path],
    expected_hashes: Mapping[str, str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, str]:
    """Fail closed unless every declared bounded source matches its hash."""

    if set(sources) != set(expected_hashes):
        raise ValueError("source and expected-hash symbol sets differ")
    actual: dict[str, str] = {}
    for symbol in sorted(sources):
        digest, _ = bounded_source_hash(sources[symbol], start=start, end=end)
        if digest != expected_hashes[symbol]:
            raise ValueError(f"source hash mismatch for {symbol}")
        actual[symbol] = digest
    return actual


def exchange_session_bar_counts(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, int]:
    """Return explicit NYSE scheduled regular-session bar counts."""

    import pandas_market_calendars as market_calendars

    calendar = market_calendars.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=pd.Timestamp(start).date(),
        end_date=pd.Timestamp(end).date(),
    )
    if schedule.empty:
        raise ValueError("exchange schedule is empty")
    counts: dict[str, int] = {}
    for session, row in schedule.iterrows():
        market_open = pd.Timestamp(row["market_open"])
        market_close = pd.Timestamp(row["market_close"])
        minutes = int((market_close - market_open) / pd.Timedelta(minutes=1))
        if minutes <= 0 or minutes % 5 != 0:
            raise ValueError(f"invalid scheduled session length for {session}")
        bars = minutes // 5
        if bars > REGULAR_SESSION_MAX_BARS:
            raise ValueError("scheduled session exceeds regular-session support")
        counts[pd.Timestamp(session).strftime("%Y-%m-%d")] = bars
    return counts


def _rolling_feature(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
    *,
    min_periods: int = 1,
) -> pd.Series:
    grouped = frame.groupby("segment_id", sort=False)[column]
    if operation == "sum":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=min_periods).sum()
        )
    if operation == "mean":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=min_periods).mean()
        )
    if operation == "std":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=min_periods).std()
        )
    raise ValueError(f"unknown rolling operation: {operation}")


def _regular_session_rows(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"raw source lacks OHLCV columns: {missing}")
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    numeric = ["open", "high", "low", "close", "volume"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    regular = minute.ge(570) & minute.lt(960)
    on_grid = ((minute - 570) % 5).eq(0) & local.dt.second.eq(0) & local.dt.microsecond.eq(0)
    frame = frame.loc[regular & on_grid].copy()
    local = frame["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    frame["symbol"] = symbol
    frame["session"] = local.dt.strftime("%Y-%m-%d")
    frame["bar_ordinal"] = ((minute - 570) // 5).astype(np.int16)
    frame = frame.rename(columns={"timestamp": "bar_start_timestamp"})
    frame = frame.sort_values(
        ["symbol", "session", "bar_start_timestamp", "bar_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"bounded regular-session tape is empty for {symbol}")
    return cast(pd.DataFrame, frame)


def build_symbol_features(
    raw: pd.DataFrame,
    *,
    symbol: str,
    expected_bars: Mapping[tuple[str, str], int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct stock features with every in-session rolling value gap-local."""

    source_timestamps = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    source_local = source_timestamps.dt.tz_convert("America/New_York")
    source_minute = source_local.dt.hour * 60 + source_local.dt.minute
    source_regular = source_minute.ge(570) & source_minute.lt(960)
    source_on_grid = (
        ((source_minute - 570) % 5).eq(0)
        & source_local.dt.second.eq(0)
        & source_local.dt.microsecond.eq(0)
    )
    invalid_regular = raw.loc[source_regular & ~source_on_grid].copy()
    invalid_regular["timestamp"] = source_timestamps.loc[invalid_regular.index]
    invalid_sessions = set(source_local.loc[invalid_regular.index].dt.strftime("%Y-%m-%d"))
    regular = _regular_session_rows(raw, symbol=symbol)
    frame, gap_ledger = annotate_causal_segments(
        regular,
        expected_bars=expected_bars,
    )
    frame["source_data_error_in_session"] = frame["session"].isin(invalid_sessions)
    if invalid_sessions:
        frame.loc[frame["source_data_error_in_session"], "session_source_complete"] = False
        invalid_rows = pd.DataFrame(
            {
                "symbol": symbol,
                "session": source_local.loc[invalid_regular.index]
                .dt.strftime("%Y-%m-%d")
                .to_numpy(),
                "previous_position": -1,
                "next_position": -1,
                "previous_bar_ordinal": -1,
                "next_bar_ordinal": -1,
                "previous_timestamp": invalid_regular["timestamp"].to_numpy(),
                "next_timestamp": invalid_regular["timestamp"].to_numpy(),
                "missing_bar_count": 0,
                "gap_reason": "invalid_non_five_minute_source_timestamp",
                "data_error": True,
            }
        )
        gap_ledger = (
            invalid_rows
            if gap_ledger.empty
            else pd.concat([gap_ledger, invalid_rows], ignore_index=True, sort=False)
        )
    segment_group = frame.groupby("segment_id", sort=False)
    previous_close = segment_group["close"].shift(1)
    segment_first = frame["segment_bar_ordinal"].eq(0)
    frame["bar_log_return"] = np.log(
        frame["close"] / previous_close.where(~segment_first, frame["open"])
    )
    frame["current_bar_log_return"] = frame["bar_log_return"]
    frame["abs_bar_log_return"] = frame["bar_log_return"].abs()
    denominator = frame["high"] - frame["low"]
    frame["bar_range_pct"] = denominator / frame["open"]
    frame["close_location_value"] = (
        2.0 * frame["close"] - frame["high"] - frame["low"]
    ) / denominator.replace(0.0, np.nan)
    frame["upper_wick_pct_of_range"] = (
        frame["high"] - frame[["open", "close"]].max(axis=1)
    ) / denominator.replace(0.0, np.nan)
    frame["lower_wick_pct_of_range"] = (
        frame[["open", "close"]].min(axis=1) - frame["low"]
    ) / denominator.replace(0.0, np.nan)
    for window in (3, 6, 12):
        return_sum = _rolling_feature(frame, "bar_log_return", window, "sum")
        absolute_sum = _rolling_feature(frame, "abs_bar_log_return", window, "sum")
        frame[f"return_sum_{window}"] = return_sum
        frame[f"mean_abs_return_{window}"] = _rolling_feature(
            frame, "abs_bar_log_return", window, "mean"
        )
        frame[f"return_std_{window}"] = _rolling_feature(
            frame,
            "bar_log_return",
            window,
            "std",
            min_periods=2,
        )
        frame[f"signed_efficiency_{window}"] = return_sum / absolute_sum.replace(0.0, np.nan)

    segment_open = segment_group["open"].transform("first")
    frame["session_return"] = np.log(frame["close"] / segment_open)
    frame["cumulative_historical_volume"] = segment_group["volume"].cumsum()

    session_group = frame.groupby("session", sort=False)
    daily = session_group.agg(
        session_open=("open", "first"),
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_close=("close", "last"),
        session_historical_volume=("volume", "sum"),
    ).reset_index()
    daily["session_return_daily"] = np.log(daily["session_close"] / daily["session_open"])
    daily["session_range_daily"] = (daily["session_high"] - daily["session_low"]) / daily[
        "session_open"
    ]
    daily["prior_session_close"] = daily["session_close"].shift(1)
    daily["prior_session_return"] = daily["session_return_daily"].shift(1)
    daily["prior_session_range"] = daily["session_range_daily"].shift(1)
    daily["prior_session_historical_volume"] = daily["session_historical_volume"].shift(1)
    daily["prior_volume_baseline"] = (
        daily["session_historical_volume"].expanding(min_periods=10).mean().shift(1)
    )
    daily["prior_session_log_relative_volume"] = np.log1p(
        daily["prior_session_historical_volume"]
        / daily["prior_volume_baseline"].replace(0.0, np.nan)
    )
    frame = frame.merge(
        daily[
            [
                "session",
                "prior_session_close",
                "prior_session_return",
                "prior_session_range",
                "prior_session_log_relative_volume",
            ]
        ],
        on="session",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    frame = frame.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    segment_open = frame.groupby("segment_id", sort=False)["open"].transform("first")
    frame["gap_log_return"] = np.log(
        segment_open / pd.to_numeric(frame["prior_session_close"], errors="coerce")
    )
    frame["historical_volume_baseline_at_bar"] = frame.groupby("bar_ordinal", sort=False)[
        "volume"
    ].transform(lambda values: values.expanding(min_periods=10).mean().shift(1))
    frame["historical_cumulative_volume_baseline_at_bar"] = frame.groupby(
        "bar_ordinal", sort=False
    )["cumulative_historical_volume"].transform(
        lambda values: values.expanding(min_periods=10).mean().shift(1)
    )
    frame["log_relative_historical_volume"] = np.log1p(
        frame["volume"] / frame["historical_volume_baseline_at_bar"].replace(0.0, np.nan)
    )
    frame["log_relative_cumulative_historical_volume"] = np.log1p(
        frame["cumulative_historical_volume"]
        / frame["historical_cumulative_volume_baseline_at_bar"].replace(0.0, np.nan)
    )
    frame["bar_complete_timestamp"] = frame["bar_start_timestamp"] + BAR_DURATION
    frame["bar_is_complete"] = True
    frame["feature_available_timestamp_max"] = frame["bar_complete_timestamp"]
    frame["source_symbol"] = symbol
    frame["source_row_ordinal"] = np.arange(len(frame), dtype=np.int64)
    return frame, gap_ledger


def add_cross_sectional_market_features(
    panel: pd.DataFrame, benchmark: pd.DataFrame
) -> pd.DataFrame:
    """Add peer and benchmark fields using rows available at each timestamp."""

    frame = panel.copy().sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    grouped = frame.groupby("bar_start_timestamp", sort=False)
    frame["market_peer_count"] = grouped["symbol"].transform("nunique")
    for source in (
        "bar_log_return",
        "return_sum_6",
        "return_sum_12",
        "bar_range_pct",
        "log_relative_historical_volume",
    ):
        frame[f"market_median__{source}"] = grouped[source].transform("median")
    frame["market_breadth_bar_positive"] = grouped["bar_log_return"].transform(
        lambda values: float((values > 0.0).mean())
    )
    frame["market_breadth_return_6_positive"] = grouped["return_sum_6"].transform(
        lambda values: float((values > 0.0).mean())
    )
    frame["market_dispersion_return_6"] = grouped["return_sum_6"].transform("std")
    frame["stock_minus_market_return_6"] = (
        frame["return_sum_6"] - frame["market_median__return_sum_6"]
    )

    keep = [
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "bar_log_return",
        "return_sum_6",
        "return_sum_12",
        "signed_efficiency_12",
        "bar_range_pct",
        "log_relative_historical_volume",
    ]
    benchmark_fields = benchmark[keep].copy()
    if benchmark_fields["bar_start_timestamp"].duplicated().any():
        raise ValueError("benchmark has duplicate timestamps")
    benchmark_fields = benchmark_fields.rename(
        columns={
            column: f"vti__{column}"
            for column in keep
            if column not in {"bar_start_timestamp", "bar_complete_timestamp"}
        }
    )
    benchmark_fields = benchmark_fields.rename(
        columns={"bar_complete_timestamp": "benchmark_available_timestamp"}
    )
    before = len(frame)
    frame = frame.merge(
        benchmark_fields,
        on="bar_start_timestamp",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(frame) != before:
        raise AssertionError("benchmark merge changed panel row count")
    frame["stock_minus_vti_return_6"] = frame["return_sum_6"] - frame["vti__return_sum_6"]
    frame["cross_sectional_source_timestamp"] = frame["bar_start_timestamp"]
    frame["cross_sectional_available_timestamp"] = frame["bar_complete_timestamp"]
    benchmark_available = pd.to_datetime(frame["benchmark_available_timestamp"], utc=True)
    frame["feature_available_timestamp_max"] = pd.concat(
        [
            pd.to_datetime(frame["feature_available_timestamp_max"], utc=True),
            benchmark_available,
            pd.to_datetime(frame["cross_sectional_available_timestamp"], utc=True),
        ],
        axis=1,
    ).max(axis=1)
    if frame["feature_available_timestamp_max"].gt(frame["bar_complete_timestamp"]).any():
        raise AssertionError("market feature uses a future bar")
    return frame.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)


def add_emission_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Construct the frozen fourteen combined emission formulas."""

    frame = panel.copy()
    activity_3 = pd.to_numeric(frame["mean_abs_return_3"], errors="coerce").clip(lower=0.0)
    activity_12 = pd.to_numeric(frame["mean_abs_return_12"], errors="coerce").clip(lower=0.0)
    frame["regime_log_activity_3"] = np.log1p(10000.0 * activity_3)
    frame["regime_log_activity_12"] = np.log1p(10000.0 * activity_12)
    frame["regime_activity_acceleration"] = (
        frame["regime_log_activity_3"] - frame["regime_log_activity_12"]
    )
    frame["regime_log_bar_range"] = np.log1p(
        10000.0 * pd.to_numeric(frame["bar_range_pct"], errors="coerce").clip(lower=0.0)
    )
    frame["regime_wick_balance"] = pd.to_numeric(
        frame["upper_wick_pct_of_range"], errors="coerce"
    ) - pd.to_numeric(frame["lower_wick_pct_of_range"], errors="coerce")
    frame["regime_log_market_dispersion"] = np.log1p(
        10000.0
        * pd.to_numeric(frame["market_dispersion_return_6"], errors="coerce").abs().clip(lower=0.0)
    )
    denominator = (6.0 * activity_12.replace(0.0, np.nan)).clip(lower=1e-8)
    frame["regime_stock_minus_market_scaled"] = np.tanh(
        pd.to_numeric(frame["stock_minus_market_return_6"], errors="coerce") / denominator
    )
    frame["regime_market_breadth_centered"] = (
        pd.to_numeric(frame["market_breadth_return_6_positive"], errors="coerce") - 0.5
    )
    missing = [feature for feature in EMISSION_FEATURES if feature not in frame]
    if missing:
        raise AssertionError(f"missing emission features: {missing}")
    if any("future" in feature.lower() for feature in EMISSION_FEATURES):
        raise AssertionError("emission manifest contains a future field")
    return frame


def causal_rank_score(series: pd.Series, *, min_periods: int = 5) -> pd.Series:
    """Causal expanding percentile score including only current and prior rows."""

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    history: list[float] = []
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        history.append(value)
        if len(history) < min_periods:
            continue
        sample = np.asarray(history, dtype=float)
        percentile = (float((sample < value).sum()) + 0.5 * float((sample == value).sum())) / len(
            sample
        )
        output[index] = (percentile - 0.5) * 2.0
    return pd.Series(output, index=series.index)


def confirm_market_states(
    raw_states: Sequence[str],
    *,
    confirm_sessions: int = 5,
    minimum_hold_sessions: int = 15,
) -> list[str]:
    """Apply the frozen causal broad-market state confirmation rule."""

    current = "unknown"
    held = 0
    pending: str | None = None
    pending_count = 0
    output: list[str] = []
    for raw_value in raw_states:
        raw = raw_value or "unknown"
        if current == "unknown" and raw != "unknown":
            current, held, pending, pending_count = raw, 1, None, 0
        elif raw == current or raw == "unknown":
            held += 1
            pending, pending_count = None, 0
        elif held < minimum_hold_sessions:
            held += 1
        else:
            if pending == raw:
                pending_count += 1
            else:
                pending, pending_count = raw, 1
            if pending_count >= confirm_sessions:
                current, held, pending, pending_count = raw, 1, None, 0
            else:
                held += 1
        output.append(current)
    return output


def build_causal_market_context(symbol_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Build prior-session broad-market context from archived panel rows."""

    daily_parts: list[pd.DataFrame] = []
    for frame in symbol_frames:
        daily = (
            frame.groupby("session", sort=True)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                bar_count=("close", "size"),
            )
            .reset_index()
        )
        daily["symbol"] = str(frame["symbol"].iloc[0])
        daily_parts.append(daily)
    panel = pd.concat(daily_parts, ignore_index=True).sort_values(
        ["symbol", "session"], kind="mergesort"
    )
    grouped = panel.groupby("symbol", sort=False)
    panel["daily_return"] = grouped["close"].pct_change(fill_method=None)
    panel["ret_20d"] = grouped["close"].pct_change(20, fill_method=None)
    panel["ma_20d"] = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    panel["above_20d_ma"] = np.where(
        panel["ma_20d"].notna(),
        panel["close"].gt(panel["ma_20d"]).astype(float),
        np.nan,
    )
    panel["rolling_20d_high"] = grouped["close"].transform(
        lambda values: values.rolling(20, min_periods=20).max()
    )
    panel["drawdown_20d"] = panel["close"] / panel["rolling_20d_high"] - 1.0
    panel["realized_vol_20d"] = grouped["daily_return"].transform(
        lambda values: values.rolling(20, min_periods=20).std()
    )
    market = (
        panel.groupby("session", sort=True)
        .agg(
            broad_symbol_count=("symbol", "nunique"),
            broad_median_ret_20d=("ret_20d", "median"),
            broad_breadth_20d_up=(
                "ret_20d",
                lambda values: (
                    float((values.dropna() > 0.0).mean()) if values.notna().any() else math.nan
                ),
            ),
            broad_breadth_above_20d_ma=("above_20d_ma", "mean"),
            broad_median_drawdown_20d=("drawdown_20d", "median"),
            broad_median_realized_vol_20d=("realized_vol_20d", "median"),
        )
        .reset_index()
    )
    broad_columns = (
        "broad_median_ret_20d",
        "broad_breadth_20d_up",
        "broad_breadth_above_20d_ma",
        "broad_median_drawdown_20d",
        "broad_median_realized_vol_20d",
    )
    for column in broad_columns:
        market[f"{column}_prior"] = pd.to_numeric(market[column], errors="coerce").shift(1)
    direction_inputs = (
        "broad_median_ret_20d_prior",
        "broad_breadth_20d_up_prior",
        "broad_breadth_above_20d_ma_prior",
        "broad_median_drawdown_20d_prior",
    )
    score_columns: list[str] = []
    for column in direction_inputs:
        output = f"score__{column}"
        market[output] = causal_rank_score(market[column])
        score_columns.append(output)
    market["b0_direction_score_raw"] = market[score_columns].mean(axis=1)
    market["b0_stress_score_raw"] = causal_rank_score(market["broad_median_realized_vol_20d_prior"])
    market["b0_direction_score"] = (
        market["b0_direction_score_raw"].rolling(15, min_periods=7).mean()
    )
    market["b0_stress_score"] = market["b0_stress_score_raw"].rolling(15, min_periods=7).mean()
    market["b0_raw_state"] = np.select(
        [
            market["b0_direction_score"].le(-0.12),
            market["b0_direction_score"].ge(0.12),
        ],
        ["weak_broad_tape", "strong_broad_tape"],
        default="neutral_broad_tape",
    )
    market.loc[market["b0_direction_score"].isna(), "b0_raw_state"] = "unknown"
    market["causal_slow_b0"] = confirm_market_states(market["b0_raw_state"].astype(str).tolist())
    market["b0_stress_box"] = np.where(
        market["b0_stress_score"].ge(0.0),
        "high_stress",
        "normal_stress",
    )
    market.loc[market["b0_stress_score"].isna(), "b0_stress_box"] = "unknown"
    return market


def canonical_frame_hash(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> str:
    """Hash full-precision values after canonical natural-key ordering."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"canonical hash columns are missing: {missing}")
    sort_columns = [column for column in NATURAL_KEY if column in columns]
    selected = frame.loc[:, list(columns)].copy()
    if sort_columns:
        selected = selected.sort_values(sort_columns, kind="mergesort")
    selected = selected.reset_index(drop=True)
    return _arrow_hash(selected)


def build_regime_panel(
    config: RegimePanelConfig,
    *,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> PanelBuildResult:
    """Build and identity-bind the complete deterministic repaired panel."""

    session_counts = exchange_session_bar_counts(config.start, config.end)
    all_symbols = (*config.symbols, config.benchmark_symbol)
    sources = {symbol: provider_path(config.provider_root, symbol) for symbol in all_symbols}
    source_hashes: dict[str, str] = {}
    source_row_counts: dict[str, int] = {}
    for symbol in sorted(sources):
        digest, row_count = bounded_source_hash(
            sources[symbol],
            start=config.start,
            end=config.end,
        )
        source_hashes[symbol] = digest
        source_row_counts[symbol] = row_count
    if (
        expected_source_hashes is not None
        and dict(sorted(expected_source_hashes.items())) != source_hashes
    ):
        raise ValueError("bounded source snapshot differs from expected hashes")
    snapshot_hash = _sha256_bytes(_canonical_json_bytes(source_hashes))

    symbol_frames: list[pd.DataFrame] = []
    gap_frames: list[pd.DataFrame] = []
    for symbol in config.symbols:
        raw = _bounded_source_frame(
            sources[symbol],
            start=config.start,
            end=config.end,
        )
        observed_sessions = (
            pd.to_datetime(raw["timestamp"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d")
        )
        expected = {
            (symbol, session): session_counts[session]
            for session in sorted(set(observed_sessions))
            if session in session_counts
        }
        if len(expected) != len(set(observed_sessions)):
            missing_schedule = sorted(set(observed_sessions).difference(session_counts))
            raise ValueError(f"source sessions absent from exchange schedule: {missing_schedule}")
        frame, gaps = build_symbol_features(
            raw,
            symbol=symbol,
            expected_bars=expected,
        )
        frame["source_artifact"] = str(sources[symbol])
        frame["source_hash"] = source_hashes[symbol]
        symbol_frames.append(frame)
        if not gaps.empty:
            gaps["source_artifact"] = str(sources[symbol])
            gaps["source_hash"] = source_hashes[symbol]
            gap_frames.append(gaps)

    benchmark_raw = _bounded_source_frame(
        sources[config.benchmark_symbol],
        start=config.start,
        end=config.end,
    )
    benchmark_sessions = (
        pd.to_datetime(benchmark_raw["timestamp"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    benchmark_expected = {
        (config.benchmark_symbol, session): session_counts[session]
        for session in sorted(set(benchmark_sessions))
        if session in session_counts
    }
    benchmark, benchmark_gaps = build_symbol_features(
        benchmark_raw,
        symbol=config.benchmark_symbol,
        expected_bars=benchmark_expected,
    )
    if not benchmark_gaps.empty:
        benchmark_gaps["source_artifact"] = str(sources[config.benchmark_symbol])
        benchmark_gaps["source_hash"] = source_hashes[config.benchmark_symbol]
        gap_frames.append(benchmark_gaps)

    panel = pd.concat(symbol_frames, ignore_index=True)
    panel = add_cross_sectional_market_features(panel, benchmark)
    panel = add_emission_features(panel)
    market_context = build_causal_market_context(symbol_frames)
    panel = panel.merge(
        market_context[
            [
                "session",
                "causal_slow_b0",
                "b0_direction_score",
                "b0_stress_score",
                "b0_stress_box",
            ]
        ],
        on="session",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    panel["b0_state_numeric"] = panel["causal_slow_b0"].map(
        {
            "weak_broad_tape": -1.0,
            "neutral_broad_tape": 0.0,
            "strong_broad_tape": 1.0,
        }
    )
    panel["b0_high_stress"] = panel["b0_stress_box"].map({"normal_stress": 0.0, "high_stress": 1.0})
    panel["clock_phase"] = pd.cut(
        panel["bar_ordinal"],
        bins=[-1, 5, 23, 53, 71, 77],
        labels=["open", "morning", "midday", "afternoon", "close"],
    ).astype(str)
    phase = 2.0 * np.pi * panel["bar_ordinal"].to_numpy(dtype=float) * 5.0 / 390.0
    panel["clock_sin"] = np.sin(phase)
    panel["clock_cos"] = np.cos(phase)
    panel["data_snapshot_hash"] = snapshot_hash
    panel = panel.sort_values(list(NATURAL_KEY), kind="mergesort").reset_index(drop=True)
    if panel[list(NATURAL_KEY)].duplicated().any():
        raise AssertionError("panel natural keys are not unique")
    if panel["bar_start_timestamp"].gt(config.end).any():
        raise AssertionError("panel opened a protected future row")

    row_key_hash = canonical_frame_hash(panel, columns=NATURAL_KEY)
    feature_table_hash = canonical_frame_hash(
        panel,
        columns=(*NATURAL_KEY, *EMISSION_FEATURES),
    )
    gap_ledger = (
        pd.concat(gap_frames, ignore_index=True)
        if gap_frames
        else pd.DataFrame(
            columns=[
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
                "source_artifact",
                "source_hash",
            ]
        )
    )
    return PanelBuildResult(
        frame=panel,
        gap_ledger=gap_ledger,
        source_hashes=source_hashes,
        source_row_counts=source_row_counts,
        data_snapshot_hash=snapshot_hash,
        row_key_hash=row_key_hash,
        feature_table_hash=feature_table_hash,
        session_expected_bars=session_counts,
    )


__all__ = [
    "BAR_DURATION",
    "EMISSION_FEATURES",
    "MARKET_EMISSION_FEATURES",
    "NATURAL_KEY",
    "PanelBuildResult",
    "REGULAR_SESSION_MAX_BARS",
    "RegimePanelConfig",
    "STOCK_EMISSION_FEATURES",
    "STOCK_RELATIVE_EMISSION_FEATURES",
    "add_cross_sectional_market_features",
    "add_emission_features",
    "bounded_source_hash",
    "build_regime_panel",
    "build_symbol_features",
    "canonical_frame_hash",
    "exchange_session_bar_counts",
    "provider_path",
    "verify_source_hashes",
]
