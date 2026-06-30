"""Research-only behavioral state similarity diagnostics.

This module reads existing local OHLCV data, labels temporary intraday behavior
states from current/prior bars only, and compares subsequent response columns as
targets. It does not fetch live data, place orders, change template gates, or
promote candidates.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_research.intraday_features import IntradayFeatureConfig, build_intraday_feature_frame

STATE_LABELS = (
    "liquidation_failed_low_recovery",
    "extension_exhaustion",
    "initiative_buying_continuation",
    "dead_chop",
)
STATE_COLUMNS = tuple(f"state_{label}" for label in STATE_LABELS) + ("state_unclassified",)
DEFAULT_OUTPUT_DIR = Path("data/reports/research/behavioral_state_similarity")
MAX_RANDOM_BASELINE_STATE_ROWS = 1_000
MAX_NEAREST_NEIGHBOR_EVENTS = 3_000
DEFAULT_SIMILARITY_FEATURE_COLUMNS = [
    "bar_return",
    "bar_range_pct",
    "body_pct_of_range",
    "close_location_value",
    "upper_wick_pct_of_range",
    "lower_wick_pct_of_range",
    "prior_3_bar_return",
    "prior_6_bar_return",
    "prior_12_bar_return",
    "directional_efficiency_3",
    "directional_efficiency_6",
    "directional_efficiency_12",
    "distance_from_session_open_pct",
    "distance_from_session_high_pct",
    "distance_from_session_low_pct",
    "distance_from_recent_high_pct",
    "distance_from_recent_low_pct",
    "distance_from_opening_range_mid_pct",
    "distance_from_opening_range_high_pct",
    "distance_from_opening_range_low_pct",
    "opening_range_width_pct",
    "distance_from_vwap_pct",
    "vwap_cross_count_12",
    "range_cross_count_12",
    "compression_zscore",
    "rolling_intraday_range_pct",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
]


@dataclass(frozen=True)
class BehavioralStateConfig:
    """Configuration for the research-only behavioral state lab."""

    timeframe: str = "5m"
    market_calendar: str | None = "XNYS"
    horizons: tuple[int, ...] = (6, 9, 12, 24)
    min_bars_after_open: int = 6
    entry_cutoff_before_close_minutes: int = 30
    relative_volume_lookback_sessions: int = 20
    direction_windows: tuple[int, ...] = (3, 6, 12)
    min_state_occurrences: int = 20
    min_symbols_per_state: int = 3
    nearest_neighbors: int = 10
    random_seed: int = 1337


@dataclass(frozen=True)
class BehavioralStateLabResult:
    """Paths and headline counts from one behavioral state similarity run."""

    run_id: str
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    event_csv_path: Path
    state_summary_csv_path: Path
    match_summary_csv_path: Path
    symbols_requested: list[str]
    symbols_completed: list[str]
    symbols_failed: list[dict[str, str]]
    state_counts: dict[str, int]
    stage_passed: bool
    research_passed: bool


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    result = pd.to_numeric(numerator, errors="coerce") / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _safe_pct_distance(value: pd.Series, reference: pd.Series) -> pd.Series:
    return _safe_divide(value, reference) - 1.0


def _rolling_sum_by_session(data: pd.DataFrame, values: pd.Series, window: int) -> pd.Series:
    return values.groupby(data["session_date"]).transform(
        lambda series: series.rolling(window, min_periods=window).sum()
    )


def _rolling_sum_available_by_session(
    data: pd.DataFrame,
    values: pd.Series,
    window: int,
) -> pd.Series:
    return values.groupby(data["session_date"]).transform(
        lambda series: series.rolling(window, min_periods=1).sum()
    )


def _directional_efficiency(data: pd.DataFrame, close: pd.Series, window: int) -> pd.Series:
    net_move = close.groupby(data["session_date"]).diff(window).abs()
    absolute_move = close.groupby(data["session_date"]).diff().abs()
    path_length = _rolling_sum_by_session(data, absolute_move, window)
    return _safe_divide(net_move, path_length).clip(lower=0.0, upper=1.0)


def _session_shift(data: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return data.groupby("session_date")[column].shift(periods)


def _numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data:
        return pd.Series(np.nan, index=data.index, dtype="float")
    return pd.to_numeric(data[column], errors="coerce")


def _cross_count(
    data: pd.DataFrame,
    value: pd.Series,
    reference: pd.Series,
    *,
    window: int,
) -> pd.Series:
    delta = pd.to_numeric(value, errors="coerce") - pd.to_numeric(reference, errors="coerce")
    sign = pd.Series(np.sign(delta), index=data.index).where(delta.notna(), 0.0)
    previous_sign = sign.groupby(data["session_date"]).shift(1)
    crosses = sign.ne(previous_sign) & sign.ne(0.0) & previous_sign.ne(0.0)
    return _rolling_sum_available_by_session(data, crosses.astype(float), window)


def _timeframe_minutes(timeframe: str) -> int:
    normalized = timeframe.strip().lower()
    if normalized.endswith("m") and normalized[:-1].isdigit():
        return int(normalized[:-1])
    if normalized.endswith("min") and normalized[:-3].isdigit():
        return int(normalized[:-3])
    if normalized.endswith("h") and normalized[:-1].isdigit():
        return int(normalized[:-1]) * 60
    return 5


def build_behavioral_state_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    config: BehavioralStateConfig | None = None,
) -> pd.DataFrame:
    """Build leakage-safe fingerprint features for temporary behavioral states."""

    cfg = config or BehavioralStateConfig()
    intraday_config = IntradayFeatureConfig(
        timeframe=cfg.timeframe,
        market_calendar=cfg.market_calendar,
        entry_cutoff_before_close_minutes=cfg.entry_cutoff_before_close_minutes,
        relative_volume_lookback_sessions=cfg.relative_volume_lookback_sessions,
        range_lookback_bars=max(cfg.direction_windows),
    )
    data = build_intraday_feature_frame(frame, intraday_config).copy()
    data["symbol"] = symbol.upper()

    open_ = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    bar_range = (high - low).clip(lower=0.0)

    data["bar_return"] = _safe_pct_distance(close, open_)
    data["bar_range_pct"] = _safe_divide(bar_range, close)
    data["body_pct_of_range"] = _safe_divide((close - open_).abs(), bar_range).clip(
        lower=0.0,
        upper=1.0,
    )
    data["close_location_value"] = _safe_divide(close - low, bar_range).clip(
        lower=0.0,
        upper=1.0,
    )
    bar_top = pd.concat([open_, close], axis=1).max(axis=1)
    bar_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    data["upper_wick_pct_of_range"] = _safe_divide(high - bar_top, bar_range).clip(
        lower=0.0,
        upper=1.0,
    )
    data["lower_wick_pct_of_range"] = _safe_divide(bar_bottom - low, bar_range).clip(
        lower=0.0,
        upper=1.0,
    )

    for window in cfg.direction_windows:
        shifted_close = _session_shift(data, "close", window)
        data[f"prior_{window}_bar_return"] = _safe_pct_distance(close, shifted_close)
        data[f"directional_efficiency_{window}"] = _directional_efficiency(data, close, window)

    session_open = data.groupby("session_date")["open"].transform("first")
    session_high_to_date = high.groupby(data["session_date"]).cummax()
    session_low_to_date = low.groupby(data["session_date"]).cummin()
    data["session_high_to_date"] = session_high_to_date
    data["session_low_to_date"] = session_low_to_date
    data["distance_from_session_open_pct"] = _safe_pct_distance(close, session_open)
    data["distance_from_session_high_pct"] = _safe_pct_distance(close, session_high_to_date)
    data["distance_from_session_low_pct"] = _safe_pct_distance(close, session_low_to_date)
    data["distance_from_recent_high_pct"] = _safe_pct_distance(close, data["recent_high"])
    data["distance_from_recent_low_pct"] = _safe_pct_distance(close, data["recent_low"])
    data["distance_from_opening_range_mid_pct"] = _safe_pct_distance(
        close,
        data["opening_range_mid"],
    )
    data["distance_from_opening_range_high_pct"] = _safe_pct_distance(
        close,
        data["opening_range_high"],
    )
    data["distance_from_opening_range_low_pct"] = _safe_pct_distance(
        close,
        data["opening_range_low"],
    )
    data["opening_range_width_pct"] = _safe_divide(data["opening_range_width"], close)
    data["distance_from_vwap_pct"] = _safe_pct_distance(close, data["session_vwap"])
    data["vwap_cross_count_12"] = _cross_count(data, close, data["session_vwap"], window=12)
    data["range_cross_count_12"] = _cross_count(
        data,
        close,
        data["opening_range_mid"],
        window=12,
    )

    max_direction_window = max(cfg.direction_windows)
    regular = data.get("is_regular_session_bar", pd.Series(True, index=data.index)).astype(bool)
    minutes_to_close = pd.to_numeric(data["minutes_to_session_close"], errors="coerce")
    data["can_evaluate_state"] = (
        regular
        & data[["open", "high", "low", "close"]].notna().all(axis=1)
        & (pd.to_numeric(data["bar_index_in_session"], errors="coerce") >= cfg.min_bars_after_open)
        & (minutes_to_close > cfg.entry_cutoff_before_close_minutes)
        & data[f"prior_{max_direction_window}_bar_return"].notna()
        & data[f"directional_efficiency_{max_direction_window}"].notna()
    )
    return data


def _future_window_stat(series: pd.Series, horizon: int, method: str) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(-1)
    reversed_shifted = shifted.iloc[::-1]
    rolling = reversed_shifted.rolling(horizon, min_periods=horizon)
    if method == "max":
        return rolling.max().iloc[::-1]
    if method == "min":
        return rolling.min().iloc[::-1]
    raise ValueError(f"Unsupported future statistic: {method}")


def add_forward_response_columns(frame: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Add forward target/response columns within each session only."""

    data = frame.copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    for horizon in horizons:
        future_close = data.groupby("session_date")["close"].shift(-horizon)
        future_high = pd.Series(np.nan, index=data.index, dtype="float")
        future_low = pd.Series(np.nan, index=data.index, dtype="float")
        for indices in data.groupby("session_date").groups.values():
            group_index = list(indices)
            future_high.loc[group_index] = _future_window_stat(
                data.loc[group_index, "high"],
                horizon,
                "max",
            )
            future_low.loc[group_index] = _future_window_stat(
                data.loc[group_index, "low"],
                horizon,
                "min",
            )
        return_column = f"forward_{horizon}_bar_return"
        data[return_column] = _safe_pct_distance(future_close, close)
        data[f"forward_{horizon}_bar_mfe"] = _safe_pct_distance(future_high, close)
        data[f"forward_{horizon}_bar_mae"] = _safe_pct_distance(future_low, close)
        data[f"forward_{horizon}_bar_abs_return"] = data[return_column].abs()
    return data


def _available_or(condition: pd.Series, *values: pd.Series) -> pd.Series:
    unavailable = pd.Series(True, index=condition.index)
    for value in values:
        unavailable &= pd.to_numeric(value, errors="coerce").isna()
    return condition.fillna(False) | unavailable


def label_behavioral_states(
    frame: pd.DataFrame,
    config: BehavioralStateConfig,
) -> pd.DataFrame:
    """Add deterministic, leakage-safe v0 behavioral state labels."""

    data = frame.copy()
    can = data["can_evaluate_state"].astype(bool)
    prior_3 = _numeric_column(data, "prior_3_bar_return")
    prior_6 = _numeric_column(data, "prior_6_bar_return")
    prior_12 = _numeric_column(data, "prior_12_bar_return")
    eff_6 = _numeric_column(data, "directional_efficiency_6")
    eff_12 = _numeric_column(data, "directional_efficiency_12")
    close_location = pd.to_numeric(data["close_location_value"], errors="coerce")
    upper_wick = pd.to_numeric(data["upper_wick_pct_of_range"], errors="coerce")
    bar_return = pd.to_numeric(data["bar_return"], errors="coerce")
    dist_recent_high = pd.to_numeric(data["distance_from_recent_high_pct"], errors="coerce")
    dist_session_high = pd.to_numeric(data["distance_from_session_high_pct"], errors="coerce")
    dist_recent_low = pd.to_numeric(data["distance_from_recent_low_pct"], errors="coerce")
    dist_session_low = pd.to_numeric(data["distance_from_session_low_pct"], errors="coerce")
    dist_vwap = pd.to_numeric(data["distance_from_vwap_pct"], errors="coerce")
    dist_or_mid = pd.to_numeric(data["distance_from_opening_range_mid_pct"], errors="coerce")
    rolling_range = pd.to_numeric(data["rolling_intraday_range_pct"], errors="coerce")
    compression = pd.to_numeric(data["compression_zscore"], errors="coerce")
    vwap_crosses = pd.to_numeric(data["vwap_cross_count_12"], errors="coerce")
    range_crosses = pd.to_numeric(data["range_cross_count_12"], errors="coerce")

    extension_context = _available_or(
        (dist_vwap > 0.0025) | (dist_or_mid > 0.0025),
        dist_vwap,
        dist_or_mid,
    )
    data["state_extension_exhaustion"] = (
        can
        & ((prior_12 > 0.010) | (prior_6 > 0.006))
        & ((dist_recent_high > -0.008) | (dist_session_high > -0.008))
        & ((eff_6 > 0.45) | (eff_12 > 0.45))
        & extension_context
        & (
            (upper_wick >= 0.30)
            | (prior_3 < prior_6 * 0.60)
            | (close_location < 0.75)
            | (bar_return <= 0.001)
        )
    )

    data["state_liquidation_failed_low_recovery"] = (
        can
        & ((prior_12 < -0.010) | (prior_6 < -0.006))
        & ((dist_recent_low < 0.012) | (dist_session_low < 0.012))
        & ((dist_recent_low > 0.002) | (close_location > 0.50))
        & ((bar_return >= 0.0) | (prior_3 > prior_6 * 0.70) | (prior_3 > prior_12 * 0.40))
    )

    data["state_dead_chop"] = (
        can
        & (prior_12.abs() <= 0.004)
        & (eff_12 <= 0.35)
        & ((rolling_range <= 0.006) | (compression <= 0.0) | compression.isna())
        & ((vwap_crosses >= 2.0) | (range_crosses >= 2.0) | (prior_6.abs() <= 0.0025))
    )

    initiative_context = _available_or(
        (dist_vwap > 0.0) | (dist_or_mid > 0.0),
        dist_vwap,
        dist_or_mid,
    )
    data["state_initiative_buying_continuation"] = (
        can
        & ((prior_6 > 0.004) | (prior_12 > 0.008))
        & (eff_6 > 0.50)
        & initiative_context
        & (dist_recent_high > -0.012)
        & (close_location >= 0.55)
        & (upper_wick <= 0.55)
    )

    data["primary_state_label"] = "unclassified"
    priority = [
        ("state_liquidation_failed_low_recovery", "liquidation_failed_low_recovery"),
        ("state_extension_exhaustion", "extension_exhaustion"),
        ("state_initiative_buying_continuation", "initiative_buying_continuation"),
        ("state_dead_chop", "dead_chop"),
    ]
    for column, label in reversed(priority):
        data.loc[data[column].astype(bool), "primary_state_label"] = label
    data["state_unclassified"] = data["primary_state_label"].eq("unclassified")
    return data


def _empty_state_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state",
            "horizon",
            "occurrence_count",
            "symbol_count",
            "session_count",
            "mean_return",
            "median_return",
            "win_rate",
            "median_abs_return",
            "p25_return",
            "p75_return",
            "mean_mfe",
            "mean_mae",
            "median_mfe",
            "median_mae",
            "response_direction",
        ]
    )


def _response_direction(
    *,
    state: str,
    median_return: float,
    win_rate: float,
    median_abs_return: float,
    random_abs_baseline: float | None = None,
) -> str:
    if random_abs_baseline is not None and median_abs_return < random_abs_baseline:
        return "low_movement"
    if state == "dead_chop" and median_abs_return <= 0.0025:
        return "low_movement"
    if median_return > 0.0 and win_rate > 0.52:
        return "positive"
    if median_return < 0.0 and win_rate < 0.48:
        return "negative"
    return "mixed"


def summarize_state_responses(
    events: pd.DataFrame,
    config: BehavioralStateConfig,
) -> pd.DataFrame:
    """Summarize forward response targets by state and horizon."""

    if events.empty:
        return _empty_state_summary()

    rows: list[dict[str, Any]] = []
    for state, state_frame in events.groupby("primary_state_label", sort=True):
        if state == "unclassified":
            continue
        for horizon in config.horizons:
            return_col = f"forward_{horizon}_bar_return"
            if return_col not in state_frame:
                continue
            valid = state_frame[state_frame[return_col].notna()]
            if valid.empty:
                continue
            returns = pd.to_numeric(valid[return_col], errors="coerce").dropna()
            abs_returns = pd.to_numeric(
                valid[f"forward_{horizon}_bar_abs_return"],
                errors="coerce",
            ).dropna()
            median_return = float(returns.median())
            win_rate = float((returns > 0.0).mean())
            median_abs_return = float(abs_returns.median()) if not abs_returns.empty else math.nan
            rows.append(
                {
                    "state": str(state),
                    "horizon": int(horizon),
                    "occurrence_count": int(len(valid)),
                    "symbol_count": int(valid["symbol"].nunique()) if "symbol" in valid else 0,
                    "session_count": int(valid["session_date"].nunique())
                    if "session_date" in valid
                    else 0,
                    "mean_return": float(returns.mean()),
                    "median_return": median_return,
                    "win_rate": win_rate,
                    "median_abs_return": median_abs_return,
                    "p25_return": float(returns.quantile(0.25)),
                    "p75_return": float(returns.quantile(0.75)),
                    "mean_mfe": float(
                        pd.to_numeric(
                            valid[f"forward_{horizon}_bar_mfe"],
                            errors="coerce",
                        ).mean()
                    ),
                    "mean_mae": float(
                        pd.to_numeric(
                            valid[f"forward_{horizon}_bar_mae"],
                            errors="coerce",
                        ).mean()
                    ),
                    "median_mfe": float(
                        pd.to_numeric(
                            valid[f"forward_{horizon}_bar_mfe"],
                            errors="coerce",
                        ).median()
                    ),
                    "median_mae": float(
                        pd.to_numeric(
                            valid[f"forward_{horizon}_bar_mae"],
                            errors="coerce",
                        ).median()
                    ),
                    "response_direction": _response_direction(
                        state=str(state),
                        median_return=median_return,
                        win_rate=win_rate,
                        median_abs_return=median_abs_return,
                    ),
                }
            )
    return pd.DataFrame(rows, columns=_empty_state_summary().columns)


def _sample_matched_random_values(
    eligible: pd.DataFrame,
    state_rows: pd.DataFrame,
    *,
    response_column: str,
    rng: np.random.Generator,
) -> list[float]:
    values: list[float] = []
    for row_index, row in state_rows.iterrows():
        symbol_pool = eligible[
            (eligible["symbol"] == row["symbol"])
            & eligible[response_column].notna()
            & eligible["can_evaluate_state"].astype(bool)
        ].drop(index=row_index, errors="ignore")
        if symbol_pool.empty:
            continue

        exact = symbol_pool[
            pd.to_numeric(symbol_pool["bar_index_in_session"], errors="coerce")
            == float(row["bar_index_in_session"])
        ]
        if exact.empty:
            row_bucket = int(float(row["bar_index_in_session"]) // 6)
            bucket = symbol_pool[
                (
                    pd.to_numeric(symbol_pool["bar_index_in_session"], errors="coerce") // 6
                )
                == row_bucket
            ]
            candidates = bucket if not bucket.empty else symbol_pool
        else:
            candidates = exact
        selected_position = int(rng.integers(0, len(candidates)))
        selected = candidates.iloc[selected_position]
        values.append(float(selected[response_column]))
    return values


def build_random_baseline(
    events: pd.DataFrame,
    *,
    config: BehavioralStateConfig,
) -> pd.DataFrame:
    """Build symbol/time-of-day matched random response baselines."""

    columns = [
        "state",
        "horizon",
        "random_mean_return",
        "random_median_return",
        "random_win_rate",
        "random_median_abs_return",
        "state_excess_vs_random_median",
        "state_abs_movement_vs_random",
        "random_sample_count",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(config.random_seed)
    eligible = events.copy().reset_index(drop=True)
    eligible["_source_row"] = np.arange(len(eligible))
    eligible["_bar_index_int"] = (
        pd.to_numeric(eligible["bar_index_in_session"], errors="coerce").fillna(-1).astype(int)
    )
    eligible["_bar_bucket"] = eligible["_bar_index_int"] // 6
    rows: list[dict[str, Any]] = []
    states = [
        state
        for state in sorted(eligible["primary_state_label"].dropna().unique().tolist())
        if state != "unclassified"
    ]
    for state in states:
        for horizon in config.horizons:
            response_column = f"forward_{horizon}_bar_return"
            abs_column = f"forward_{horizon}_bar_abs_return"
            if response_column not in eligible:
                continue
            horizon_eligible = eligible[
                eligible[response_column].notna() & eligible["can_evaluate_state"].astype(bool)
            ].reset_index(drop=True)
            if horizon_eligible.empty:
                continue
            exact_groups: dict[tuple[str, int], np.ndarray] = {}
            bucket_groups: dict[tuple[str, int], np.ndarray] = {}
            symbol_groups: dict[str, np.ndarray] = {}
            for _, group in horizon_eligible.groupby(
                ["symbol", "_bar_index_int"],
                sort=False,
            ):
                first = group.iloc[0]
                exact_groups[(str(first["symbol"]), int(first["_bar_index_int"]))] = (
                    group.index.to_numpy()
                )
            for _, group in horizon_eligible.groupby(["symbol", "_bar_bucket"], sort=False):
                first = group.iloc[0]
                bucket_groups[(str(first["symbol"]), int(first["_bar_bucket"]))] = (
                    group.index.to_numpy()
                )
            for symbol_value, group in horizon_eligible.groupby("symbol", sort=False):
                symbol_groups[str(symbol_value)] = group.index.to_numpy()

            valid_state = horizon_eligible[horizon_eligible["primary_state_label"] == state]
            if valid_state.empty:
                continue
            random_state_sample = valid_state
            if len(valid_state) > MAX_RANDOM_BASELINE_STATE_ROWS:
                random_state_sample = valid_state.sample(
                    n=MAX_RANDOM_BASELINE_STATE_ROWS,
                    random_state=config.random_seed + int(horizon),
                )
            random_values: list[float] = []
            for _, sample_row in random_state_sample.iterrows():
                symbol_value = str(sample_row["symbol"])
                source_row = int(sample_row["_source_row"])
                bar_index = int(sample_row["_bar_index_int"])
                bar_bucket = int(sample_row["_bar_bucket"])
                candidate_positions = exact_groups.get((symbol_value, bar_index))
                if candidate_positions is None or len(candidate_positions) == 0:
                    candidate_positions = bucket_groups.get((symbol_value, bar_bucket))
                if candidate_positions is None or len(candidate_positions) == 0:
                    candidate_positions = symbol_groups.get(symbol_value)
                if candidate_positions is None or len(candidate_positions) == 0:
                    continue
                source_rows = horizon_eligible.loc[candidate_positions, "_source_row"].to_numpy()
                filtered_positions = candidate_positions[source_rows != source_row]
                if len(filtered_positions) > 0:
                    candidate_positions = filtered_positions
                selected_position = int(rng.choice(candidate_positions))
                selected_value = horizon_eligible.at[selected_position, response_column]
                if not pd.isna(selected_value):
                    random_values.append(float(str(selected_value)))
            state_returns = pd.to_numeric(valid_state[response_column], errors="coerce").dropna()
            state_abs = pd.to_numeric(valid_state[abs_column], errors="coerce").dropna()
            if random_values:
                random_series = pd.Series(random_values, dtype="float")
                random_median = float(random_series.median())
                random_abs = float(random_series.abs().median())
                random_mean = float(random_series.mean())
                random_win_rate = float((random_series > 0.0).mean())
            else:
                random_median = math.nan
                random_abs = math.nan
                random_mean = math.nan
                random_win_rate = math.nan
            rows.append(
                {
                    "state": str(state),
                    "horizon": int(horizon),
                    "random_mean_return": random_mean,
                    "random_median_return": random_median,
                    "random_win_rate": random_win_rate,
                    "random_median_abs_return": random_abs,
                    "state_excess_vs_random_median": float(state_returns.median())
                    - random_median
                    if not math.isnan(random_median)
                    else math.nan,
                    "state_abs_movement_vs_random": float(state_abs.median()) - random_abs
                    if not math.isnan(random_abs)
                    else math.nan,
                    "random_sample_count": int(len(random_values)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _validate_similarity_feature_columns(feature_columns: list[str]) -> None:
    bad_columns = [
        column
        for column in feature_columns
        if column.startswith("forward_")
        or "mfe" in column.lower()
        or "mae" in column.lower()
    ]
    if bad_columns:
        joined = ", ".join(sorted(bad_columns))
        raise ValueError(
            f"Nearest-neighbor feature_columns must exclude response columns: {joined}"
        )


def _empty_match_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state",
            "horizon",
            "match_count",
            "cross_symbol_matches",
            "same_state_match_sign_agreement",
            "cross_symbol_match_sign_agreement",
            "average_abs_response_diff",
            "median_self_forward_return",
            "median_neighbor_forward_return",
            "matched_symbols",
            "matched_state_labels",
            "verdict",
        ]
    )


def _sign_agreement(values: list[float], reference: float) -> float:
    if not values or pd.isna(reference):
        return math.nan
    reference_sign = np.sign(reference)
    if reference_sign == 0:
        return float(np.mean([np.sign(value) == 0 for value in values]))
    return float(np.mean([np.sign(value) == reference_sign for value in values]))


def run_nearest_neighbor_similarity(
    events: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: BehavioralStateConfig,
) -> pd.DataFrame:
    """Compare forward responses among nearest behavioral fingerprint neighbors."""

    _validate_similarity_feature_columns(feature_columns)
    if events.empty:
        return _empty_match_summary()

    available_features = [column for column in feature_columns if column in events.columns]
    if not available_features:
        return _empty_match_summary()

    working = events[events["primary_state_label"].ne("unclassified")].copy()
    if len(working) < 2:
        return _empty_match_summary()

    feature_frame = working[available_features].apply(pd.to_numeric, errors="coerce")
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.dropna(axis=1, how="all")
    if feature_frame.empty:
        return _empty_match_summary()
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    valid_mask = feature_frame.notna().all(axis=1)
    working = working.loc[valid_mask].reset_index(drop=True)
    feature_frame = feature_frame.loc[valid_mask].reset_index(drop=True)
    if len(working) < 2:
        return _empty_match_summary()

    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError:
        return _empty_match_summary()

    scaled = StandardScaler().fit_transform(feature_frame.to_numpy(dtype=float))
    neighbor_count = min(len(working), max(2, config.nearest_neighbors + 1))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count)
    neighbors.fit(scaled)
    _, indices = neighbors.kneighbors(scaled)

    detail_rows: list[dict[str, Any]] = []
    for source_position, neighbor_positions in enumerate(indices):
        source = working.iloc[source_position]
        candidate_positions = [
            int(position) for position in neighbor_positions if int(position) != source_position
        ]
        cross_symbol_positions = [
            position
            for position in candidate_positions
            if working.iloc[position]["symbol"] != source["symbol"]
        ]
        non_same_session_positions = [
            position
            for position in candidate_positions
            if not (
                working.iloc[position]["symbol"] == source["symbol"]
                and working.iloc[position].get("session_date") == source.get("session_date")
            )
        ]
        selected_positions = (
            cross_symbol_positions
            or non_same_session_positions
            or candidate_positions
        )[: config.nearest_neighbors]
        if not selected_positions:
            continue
        selected = working.iloc[selected_positions]

        for horizon in config.horizons:
            response_column = f"forward_{horizon}_bar_return"
            if response_column not in working or pd.isna(source.get(response_column)):
                continue
            neighbor_values = [
                float(value)
                for value in pd.to_numeric(selected[response_column], errors="coerce").dropna()
            ]
            if not neighbor_values:
                continue
            same_state = selected[selected["primary_state_label"] == source["primary_state_label"]]
            same_state_values = [
                float(value)
                for value in pd.to_numeric(same_state[response_column], errors="coerce").dropna()
            ]
            cross_symbol = selected[selected["symbol"] != source["symbol"]]
            cross_symbol_values = [
                float(value)
                for value in pd.to_numeric(cross_symbol[response_column], errors="coerce").dropna()
            ]
            self_response = float(source[response_column])
            neighbor_median = float(pd.Series(neighbor_values).median())
            detail_rows.append(
                {
                    "state": str(source["primary_state_label"]),
                    "horizon": int(horizon),
                    "self_forward_return": self_response,
                    "neighbor_median_forward_return": neighbor_median,
                    "same_state_sign_agreement": _sign_agreement(
                        same_state_values,
                        self_response,
                    ),
                    "cross_symbol_sign_agreement": _sign_agreement(
                        cross_symbol_values,
                        self_response,
                    ),
                    "absolute_response_difference": abs(neighbor_median - self_response),
                    "cross_symbol_neighbor_count": int(len(cross_symbol_values)),
                    "matched_symbols": ",".join(sorted(set(map(str, selected["symbol"].tolist())))),
                    "matched_timestamps": ",".join(
                        sorted(set(map(str, selected["timestamp"].tolist())))
                    ),
                    "matched_state_labels": ",".join(
                        sorted(set(map(str, selected["primary_state_label"].tolist())))
                    ),
                }
            )

    if not detail_rows:
        return _empty_match_summary()

    details = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, Any]] = []
    for key, group in details.groupby(["state", "horizon"], sort=True):
        state_value, horizon_value = key
        state = str(state_value)
        horizon = int(str(horizon_value))
        cross_agreement = pd.to_numeric(
            group["cross_symbol_sign_agreement"],
            errors="coerce",
        ).dropna()
        same_agreement = pd.to_numeric(
            group["same_state_sign_agreement"],
            errors="coerce",
        ).dropna()
        avg_diff = float(group["absolute_response_difference"].mean())
        cross_mean = float(cross_agreement.mean()) if not cross_agreement.empty else math.nan
        if group["cross_symbol_neighbor_count"].sum() <= 0:
            verdict = "not_enough_evidence"
        elif not math.isnan(cross_mean) and cross_mean >= 0.58:
            verdict = "cross_symbol_similarity"
        else:
            verdict = "mixed_response"
        summary_rows.append(
            {
                "state": str(state),
                "horizon": int(horizon),
                "match_count": int(len(group)),
                "cross_symbol_matches": int((group["cross_symbol_neighbor_count"] > 0).sum()),
                "same_state_match_sign_agreement": float(same_agreement.mean())
                if not same_agreement.empty
                else math.nan,
                "cross_symbol_match_sign_agreement": cross_mean,
                "average_abs_response_diff": avg_diff,
                "median_self_forward_return": float(group["self_forward_return"].median()),
                "median_neighbor_forward_return": float(
                    group["neighbor_median_forward_return"].median()
                ),
                "matched_symbols": ",".join(
                    sorted(set(",".join(group["matched_symbols"].tolist()).split(",")))
                ),
                "matched_state_labels": ",".join(
                    sorted(set(",".join(group["matched_state_labels"].tolist()).split(",")))
                ),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(summary_rows, columns=_empty_match_summary().columns)


def _load_qualified_symbols(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = payload.get("qualified_symbols", [])
    output: list[str] = []
    for item in symbols:
        if isinstance(item, dict) and item.get("symbol"):
            output.append(str(item["symbol"]).upper())
        elif isinstance(item, str):
            output.append(item.upper())
    return output


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    text = frame.replace([np.inf, -np.inf], np.nan).to_json(orient="records")
    return list(json.loads(text))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _state_counts(events: pd.DataFrame) -> dict[str, int]:
    if events.empty:
        return {}
    counts = Counter(map(str, events["primary_state_label"].tolist()))
    return dict(sorted(counts.items()))


def _collect_warnings(
    *,
    all_rows: pd.DataFrame,
    events: pd.DataFrame,
    symbols_failed: list[dict[str, str]],
    config: BehavioralStateConfig,
) -> list[str]:
    warnings: list[str] = []
    if symbols_failed:
        warnings.append(f"{len(symbols_failed)} symbols failed to load or process")
    if events.empty:
        warnings.append("no classified events with complete forward responses")
        return warnings

    for state, group in events.groupby("primary_state_label"):
        if len(group) < config.min_state_occurrences:
            warnings.append(f"{state}: low sample ({len(group)} events)")
        symbol_share = group["symbol"].value_counts(normalize=True)
        if not symbol_share.empty and float(symbol_share.iloc[0]) >= 0.80:
            warnings.append(f"{state}: single-symbol dominated by {symbol_share.index[0]}")
        session_share = group["session_date"].value_counts(normalize=True)
        if not session_share.empty and float(session_share.iloc[0]) >= 0.80:
            warnings.append(f"{state}: one-session dominated by {session_share.index[0]}")
    if "relative_volume_at_bar_index" in all_rows:
        relative_volume_missing = float(all_rows["relative_volume_at_bar_index"].isna().mean())
        if relative_volume_missing > 0.50:
            warnings.append(f"missing relative volume on {relative_volume_missing:.1%} of rows")
    if "session_complete_warning" in all_rows and bool(all_rows["session_complete_warning"].any()):
        warnings.append("one or more symbols have incomplete/nonstandard session warnings")
    return sorted(set(warnings))


def _row_for_horizon(frame: pd.DataFrame, state: str, horizon: int) -> dict[str, Any] | None:
    if frame.empty:
        return None
    matched = frame[(frame["state"] == state) & (frame["horizon"] == horizon)]
    if matched.empty:
        return None
    return dict(matched.iloc[0])


def _candidate_verdicts(
    *,
    state_summary: pd.DataFrame,
    random_baseline: pd.DataFrame,
    match_summary: pd.DataFrame,
    config: BehavioralStateConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    if state_summary.empty:
        return candidate_rows, block_rows

    verdict_horizon = 12 if 12 in config.horizons else config.horizons[0]
    for state in sorted(state_summary["state"].unique().tolist()):
        row = _row_for_horizon(state_summary, state, verdict_horizon)
        if row is None:
            continue
        baseline = _row_for_horizon(random_baseline, state, verdict_horizon) or {}
        match = _row_for_horizon(match_summary, state, verdict_horizon) or {}
        occurrence_count = int(row["occurrence_count"])
        symbol_count = int(row["symbol_count"])
        median_return = float(row["median_return"])
        random_median = baseline.get("random_median_return")
        random_median_float = (
            float(random_median)
            if random_median is not None and not pd.isna(random_median)
            else math.nan
        )
        if row["response_direction"] == "low_movement" or state == "dead_chop":
            block_rows.append(
                {
                    "state": state,
                    "horizon": verdict_horizon,
                    "verdict": "block_state",
                    "reason": "low movement / dead chop behavioral state",
                }
            )
            continue
        if occurrence_count < config.min_state_occurrences:
            verdict = "not_enough_evidence"
            reason = f"only {occurrence_count} events"
        else:
            beats_random = math.isnan(random_median_float) or (
                abs(median_return) > abs(random_median_float)
            )
            cross_agreement = match.get("cross_symbol_match_sign_agreement")
            cross_agreement_float = (
                float(cross_agreement)
                if cross_agreement is not None and not pd.isna(cross_agreement)
                else math.nan
            )
            has_cross_symbol = symbol_count >= config.min_symbols_per_state
            meaningful_match = (
                not math.isnan(cross_agreement_float) and cross_agreement_float >= 0.55
            )
            if has_cross_symbol and beats_random and meaningful_match:
                verdict = "portable_candidate"
                reason = "cross-symbol response beats matched random baseline"
            elif beats_random and not has_cross_symbol:
                verdict = "symbol_specific_candidate"
                reason = f"response is mostly from {symbol_count} symbols"
            elif row["response_direction"] == "mixed":
                verdict = "mixed_response"
                reason = "forward signs are inconsistent"
            else:
                verdict = "not_enough_evidence"
                reason = "random baseline or nearest-neighbor evidence is weak"
        candidate_rows.append(
            {
                "state": state,
                "horizon": verdict_horizon,
                "verdict": verdict,
                "reason": reason,
                "events": occurrence_count,
                "symbols": symbol_count,
                "median_return": median_return,
                "random_median_return": random_median,
                "cross_symbol_sign_agreement": match.get("cross_symbol_match_sign_agreement"),
            }
        )
    return candidate_rows, block_rows


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    if not rows:
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(output) + "\n"


def _markdown(summary: dict[str, Any]) -> str:
    state_rows: list[dict[str, Any]] = []
    state_headers = [
        "State",
        "Events",
        "Symbols",
        "Fwd 6 median",
        "Fwd 12 median",
        "Fwd 24 median",
        "Win rate 12",
        "Random median 12",
        "Verdict",
    ]
    match_headers = [
        "State",
        "Cross-symbol matches",
        "Sign agreement 6",
        "Sign agreement 12",
        "Avg abs response diff 12",
        "Verdict",
    ]
    state_summary: list[dict[str, Any]] = summary["state_response_summary"]
    random_summary: list[dict[str, Any]] = summary["random_baseline_summary"]
    verdicts = {
        row["state"]: row["verdict"] for row in summary["candidate_portable_states"]
    }
    verdicts.update({row["state"]: row["verdict"] for row in summary["block_states"]})
    for state in sorted({row["state"] for row in state_summary}):
        h6: dict[str, Any] = next(
            (row for row in state_summary if row["state"] == state and row["horizon"] == 6),
            {},
        )
        h12: dict[str, Any] = next(
            (row for row in state_summary if row["state"] == state and row["horizon"] == 12),
            {},
        )
        h24: dict[str, Any] = next(
            (row for row in state_summary if row["state"] == state and row["horizon"] == 24),
            {},
        )
        random12: dict[str, Any] = next(
            (row for row in random_summary if row["state"] == state and row["horizon"] == 12),
            {},
        )
        state_rows.append(
            {
                "State": state,
                "Events": h12.get("occurrence_count", h6.get("occurrence_count", "")),
                "Symbols": h12.get("symbol_count", h6.get("symbol_count", "")),
                "Fwd 6 median": _format_pct(h6.get("median_return")),
                "Fwd 12 median": _format_pct(h12.get("median_return")),
                "Fwd 24 median": _format_pct(h24.get("median_return")),
                "Win rate 12": _format_pct(h12.get("win_rate")),
                "Random median 12": _format_pct(random12.get("random_median_return")),
                "Verdict": verdicts.get(state, "mixed_response"),
            }
        )

    match_rows = []
    for row in summary["nearest_neighbor_summary"]:
        if row["horizon"] not in {6, 12}:
            continue
        match_rows.append(
            {
                "State": row["state"],
                "Cross-symbol matches": row["cross_symbol_matches"],
                f"Sign agreement {row['horizon']}": _format_pct(
                    row["cross_symbol_match_sign_agreement"]
                ),
                "Avg abs response diff 12": _format_pct(
                    row["average_abs_response_diff"] if row["horizon"] == 12 else None
                ),
                "Verdict": row["verdict"],
            }
        )

    warnings = "\n".join(f"- {warning}" for warning in summary["warnings"]) or "- None"
    return f"""# Behavioral State Similarity Lab

This is a research-only diagnostic report. States were detected using only
current and prior bars available at each bar close. Forward return, MFE, MAE,
and absolute-return columns are response targets only. This run did not fetch
live data, place orders, loosen gates, alter templates, or promote candidates.

- Run id: `{summary["run_id"]}`
- Symbols completed: {len(summary["symbols_completed"])}
- Symbols failed: {len(summary["symbols_failed"])}
- Total classified events: {summary["total_events"]}
- Timeframe: `{summary["config"]["timeframe"]}`

## State Response Summary

{_markdown_table(state_rows, state_headers)}

## Nearest Match Summary

{_markdown_table(match_rows, match_headers)}

## Candidate Portable States

{_markdown_table(summary["candidate_portable_states"], ["state", "horizon", "verdict", "reason"])}

## Block / No-Trade States

{_markdown_table(summary["block_states"], ["state", "horizon", "verdict", "reason"])}

## Warnings

{warnings}
"""


def run_behavioral_state_similarity_lab(
    *,
    data_dir: Path,
    symbols: list[str],
    source: str = "eodhd",
    instrument_type: str = "stock",
    timeframe: str = "5m",
    market_calendar: str | None = "XNYS",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: BehavioralStateConfig | None = None,
) -> BehavioralStateLabResult:
    """Run the research-only behavioral state similarity lab on local datasets."""

    from stocker_data.storage import DatasetKey, load_dataset

    cfg = config or BehavioralStateConfig(timeframe=timeframe, market_calendar=market_calendar)
    run_id = "behavioral_state_similarity_" + datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    symbols_requested = [symbol.upper() for symbol in symbols]
    symbols_completed: list[str] = []
    symbols_failed: list[dict[str, str]] = []
    frames: list[pd.DataFrame] = []

    for symbol in symbols_requested:
        try:
            raw = load_dataset(
                DatasetKey(
                    source=source,
                    instrument_type=instrument_type,
                    symbol=symbol,
                    timeframe=cfg.timeframe,
                ),
                data_dir=data_dir,
            ).sort_values("timestamp")
            state_frame = build_behavioral_state_frame(raw, symbol=symbol, config=cfg)
            state_frame = add_forward_response_columns(state_frame, cfg.horizons)
            state_frame = label_behavioral_states(state_frame, cfg)
        except Exception as exc:
            symbols_failed.append({"symbol": symbol, "error": str(exc)})
            continue
        frames.append(state_frame)
        symbols_completed.append(symbol)

    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    required_forward_columns = [f"forward_{horizon}_bar_return" for horizon in cfg.horizons]
    if all_rows.empty:
        events = pd.DataFrame()
        baseline_input = pd.DataFrame()
    else:
        complete_forward = all_rows[required_forward_columns].notna().all(axis=1)
        baseline_input = all_rows[
            all_rows["can_evaluate_state"].astype(bool) & complete_forward
        ].copy()
        events = baseline_input[baseline_input["primary_state_label"].ne("unclassified")].copy()

    state_summary = summarize_state_responses(events, cfg)
    random_baseline = build_random_baseline(baseline_input, config=cfg)
    similarity_features = [
        column for column in DEFAULT_SIMILARITY_FEATURE_COLUMNS if column in events.columns
    ]
    runtime_warnings: list[str] = []
    if not events.empty:
        largest_state_sample = int(events["primary_state_label"].value_counts().max())
        if largest_state_sample > MAX_RANDOM_BASELINE_STATE_ROWS:
            runtime_warnings.append(
                "random baseline sampled at most "
                f"{MAX_RANDOM_BASELINE_STATE_ROWS} rows per state/horizon"
            )
    neighbor_events = events
    if len(events) > MAX_NEAREST_NEIGHBOR_EVENTS:
        neighbor_events = events.sample(
            n=MAX_NEAREST_NEIGHBOR_EVENTS,
            random_state=cfg.random_seed,
        )
        runtime_warnings.append(
            "nearest-neighbor similarity sampled "
            f"{MAX_NEAREST_NEIGHBOR_EVENTS} of {len(events)} classified events"
        )
    match_summary = run_nearest_neighbor_similarity(
        neighbor_events,
        feature_columns=similarity_features,
        config=cfg,
    )
    candidate_states, block_states = _candidate_verdicts(
        state_summary=state_summary,
        random_baseline=random_baseline,
        match_summary=match_summary,
        config=cfg,
    )
    warnings = _collect_warnings(
        all_rows=all_rows,
        events=events,
        symbols_failed=symbols_failed,
        config=cfg,
    )
    warnings = sorted(set([*warnings, *runtime_warnings]))
    state_counts = _state_counts(events)

    event_csv_path = run_dir / "events.csv"
    state_summary_csv_path = run_dir / "state_summary.csv"
    match_summary_csv_path = run_dir / "match_summary.csv"
    random_baseline_csv_path = run_dir / "random_baseline.csv"
    summary_json_path = run_dir / "summary.json"
    summary_markdown_path = run_dir / "summary.md"
    _write_csv(event_csv_path, events)
    _write_csv(state_summary_csv_path, state_summary)
    _write_csv(match_summary_csv_path, match_summary)
    _write_csv(random_baseline_csv_path, random_baseline)

    research_passed = bool(symbols_completed) and not symbols_failed
    summary = {
        "run_id": run_id,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "research_only": True,
        "data_fetched": False,
        "orders_placed": False,
        "candidate_promotion": False,
        "config": asdict(cfg),
        "symbols_requested": symbols_requested,
        "symbols_completed": symbols_completed,
        "symbols_failed": symbols_failed,
        "total_events": int(len(events)),
        "state_counts": state_counts,
        "state_response_summary": _records(state_summary),
        "random_baseline_summary": _records(random_baseline),
        "nearest_neighbor_summary": _records(match_summary),
        "candidate_portable_states": candidate_states,
        "block_states": block_states,
        "warnings": warnings,
        "files": {
            "events_csv": str(event_csv_path),
            "state_summary_csv": str(state_summary_csv_path),
            "match_summary_csv": str(match_summary_csv_path),
            "random_baseline_csv": str(random_baseline_csv_path),
            "summary_markdown": str(summary_markdown_path),
        },
        "stage_passed": research_passed,
        "research_passed": research_passed,
    }
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    summary_markdown_path.write_text(_markdown(summary), encoding="utf-8")

    return BehavioralStateLabResult(
        run_id=run_id,
        output_dir=run_dir,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        event_csv_path=event_csv_path,
        state_summary_csv_path=state_summary_csv_path,
        match_summary_csv_path=match_summary_csv_path,
        symbols_requested=symbols_requested,
        symbols_completed=symbols_completed,
        symbols_failed=symbols_failed,
        state_counts=state_counts,
        stage_passed=research_passed,
        research_passed=research_passed,
    )


__all__ = [
    "BehavioralStateConfig",
    "BehavioralStateLabResult",
    "add_forward_response_columns",
    "build_behavioral_state_frame",
    "build_random_baseline",
    "label_behavioral_states",
    "run_behavioral_state_similarity_lab",
    "run_nearest_neighbor_similarity",
    "summarize_state_responses",
]
