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

STATE_DEFINITIONS: dict[str, dict[str, str]] = {
    "liquidation_failed_low_recovery": {
        "state_family": "downside_extension",
        "state_subtype": "failed_low_recovery",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "session_low_area",
        "stimulus_label": "failed_new_low_or_reclaim",
    },
    "extension_exhaustion": {
        "state_family": "upside_extension",
        "state_subtype": "extension_exhaustion",
        "state_direction": "up",
        "state_energy": "high",
        "state_location": "session_high_area",
        "stimulus_label": "stall_or_upper_wick",
    },
    "initiative_buying_continuation": {
        "state_family": "initiative_buying",
        "state_subtype": "clean_continuation",
        "state_direction": "up",
        "state_energy": "high",
        "state_location": "above_vwap_or_opening_range",
        "stimulus_label": "continuation_pressure",
    },
    "dead_chop": {
        "state_family": "chop_compression",
        "state_subtype": "dead_chop",
        "state_direction": "neutral",
        "state_energy": "low",
        "state_location": "range_mid",
        "stimulus_label": "none",
    },
    "initiative_buying_controlled_pullback": {
        "state_family": "initiative_buying",
        "state_subtype": "controlled_pullback",
        "state_direction": "up",
        "state_energy": "medium",
        "state_location": "above_vwap_or_opening_range",
        "stimulus_label": "controlled_pullback",
    },
    "momentum_memory": {
        "state_family": "initiative_buying",
        "state_subtype": "momentum_memory_score",
        "state_direction": "up",
        "state_energy": "medium",
        "state_location": "above_session_open",
        "stimulus_label": "continuation_pressure",
    },
    "bullish_shock_failure": {
        "state_family": "shock_failure",
        "state_subtype": "bullish_push_failed",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "session_high_area",
        "stimulus_label": "failed_new_high",
    },
    "elastic_recoil": {
        "state_family": "recoil",
        "state_subtype": "elastic_recoil",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "extension_area",
        "stimulus_label": "sharp_reversal",
    },
    "exhaustion_fade": {
        "state_family": "upside_extension",
        "state_subtype": "exhaustion_fade",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "session_high_area",
        "stimulus_label": "stall_or_upper_wick",
    },
    "active_liquidation": {
        "state_family": "downside_extension",
        "state_subtype": "active_liquidation",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "session_low_area",
        "stimulus_label": "new_low_liquidation",
    },
    "failed_bounce_active_liquidation": {
        "state_family": "downside_extension",
        "state_subtype": "failed_bounce_active_liquidation",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "session_low_area",
        "stimulus_label": "failed_bounce",
    },
    "liquidation_failed_low_squeeze": {
        "state_family": "downside_extension",
        "state_subtype": "failed_low_squeeze",
        "state_direction": "up",
        "state_energy": "high",
        "state_location": "session_low_area",
        "stimulus_label": "failed_new_low_or_reclaim",
    },
    "slow_snapback": {
        "state_family": "recovery",
        "state_subtype": "slow_snapback",
        "state_direction": "up",
        "state_energy": "medium",
        "state_location": "below_or_near_vwap",
        "stimulus_label": "controlled_recovery",
    },
    "opening_drive_up": {
        "state_family": "opening_drive",
        "state_subtype": "opening_drive_up",
        "state_direction": "up",
        "state_energy": "high",
        "state_location": "opening_session",
        "stimulus_label": "opening_drive",
    },
    "opening_drive_down": {
        "state_family": "opening_drive",
        "state_subtype": "opening_drive_down",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "opening_session",
        "stimulus_label": "opening_drive",
    },
    "failed_open_long_block": {
        "state_family": "opening_failure",
        "state_subtype": "failed_open_long_block",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "opening_range_high",
        "stimulus_label": "failed_open",
    },
    "failed_open_down_continuation": {
        "state_family": "opening_failure",
        "state_subtype": "failed_open_down_continuation",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "opening_range_low",
        "stimulus_label": "failed_open",
    },
    "failed_open_high": {
        "state_family": "opening_range_failure",
        "state_subtype": "failed_open_high",
        "state_direction": "down",
        "state_energy": "medium",
        "state_location": "opening_range_high",
        "stimulus_label": "failed_open_high",
    },
    "failed_open_low": {
        "state_family": "opening_range_failure",
        "state_subtype": "failed_open_low",
        "state_direction": "up",
        "state_energy": "medium",
        "state_location": "opening_range_low",
        "stimulus_label": "failed_open_low",
    },
    "compression_pre_break": {
        "state_family": "compression",
        "state_subtype": "pre_break",
        "state_direction": "neutral",
        "state_energy": "low",
        "state_location": "range_mid",
        "stimulus_label": "compression",
    },
    "compression_breakout": {
        "state_family": "compression_break",
        "state_subtype": "breakout_up",
        "state_direction": "up",
        "state_energy": "high",
        "state_location": "recent_high",
        "stimulus_label": "compression_breakout",
    },
    "compression_breakout_up": {
        "state_family": "compression_break",
        "state_subtype": "breakout_up",
        "state_direction": "up",
        "state_energy": "high",
        "state_location": "recent_high",
        "stimulus_label": "compression_breakout",
    },
    "compression_breakdown": {
        "state_family": "compression_break",
        "state_subtype": "breakdown",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "recent_low",
        "stimulus_label": "compression_breakdown",
    },
    "trend_day_up": {
        "state_family": "trend_day",
        "state_subtype": "trend_day_up",
        "state_direction": "up",
        "state_energy": "medium",
        "state_location": "above_session_open",
        "stimulus_label": "trend_persistence",
    },
    "trend_day_down": {
        "state_family": "trend_day",
        "state_subtype": "trend_day_down",
        "state_direction": "down",
        "state_energy": "medium",
        "state_location": "below_session_open",
        "stimulus_label": "trend_persistence",
    },
    "gap_and_go_up": {
        "state_family": "gap_continuation",
        "state_subtype": "gap_and_go_up",
        "state_direction": "up",
        "state_energy": "high",
        "state_location": "above_prior_high",
        "stimulus_label": "gap_and_go",
    },
    "gap_fail_down": {
        "state_family": "gap_failure",
        "state_subtype": "gap_fail_down",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "prior_high_area",
        "stimulus_label": "gap_failure",
    },
    "downside_liquidation": {
        "state_family": "downside_extension",
        "state_subtype": "active_liquidation",
        "state_direction": "down",
        "state_energy": "high",
        "state_location": "session_low_area",
        "stimulus_label": "new_low_liquidation",
    },
    "upside_extension": {
        "state_family": "upside_extension",
        "state_subtype": "upside_extension",
        "state_direction": "up",
        "state_energy": "high",
        "state_location": "session_high_area",
        "stimulus_label": "new_high_extension",
    },
}
STATE_LABELS = tuple(STATE_DEFINITIONS)
STATE_COLUMNS = tuple(f"state_{label}" for label in STATE_LABELS) + ("state_unclassified",)
DEFAULT_OUTPUT_DIR = Path("data/reports/research/behavioral_state_similarity_v2")
MAX_RANDOM_BASELINE_STATE_ROWS = 1_000
MAX_NEAREST_NEIGHBOR_EVENTS = 3_000
MAX_RESPONSE_SHAPE_EVENTS = 2_500
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
STATE_FINGERPRINT_NUMERIC_COLUMNS = [
    "prior_3_bar_return",
    "prior_6_bar_return",
    "prior_12_bar_return",
    "directional_efficiency_6",
    "directional_efficiency_12",
    "close_location_value",
    "body_pct_of_range",
    "upper_wick_pct_of_range",
    "lower_wick_pct_of_range",
    "distance_from_vwap_pct",
    "distance_from_opening_range_mid_pct",
    "distance_from_opening_range_high_pct",
    "distance_from_opening_range_low_pct",
    "opening_range_width_pct",
    "rolling_intraday_range_pct",
    "compression_zscore",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
    "state_age_bars",
]
STATE_FINGERPRINT_CATEGORICAL_COLUMNS = [
    "time_of_day_bucket",
    "bar_index_bucket",
]
STATE_FINGERPRINT_FEATURE_COLUMNS = tuple(
    [*STATE_FINGERPRINT_NUMERIC_COLUMNS, *STATE_FINGERPRINT_CATEGORICAL_COLUMNS]
)
EVENT_MODES = {
    "all_rows",
    "state_entry_only",
    "state_change_only",
    "stimulus_event_only",
    "non_overlapping_by_horizon",
    "state_entry_non_overlapping",
}
DECISION_GATE_DEFAULTS = {
    "min_independent_events_per_state_horizon": 30,
    "min_train_symbols_per_state": 3,
    "min_test_symbols_per_state": 2,
    "max_single_symbol_share": 0.50,
    "max_single_session_share": 0.20,
    "max_single_month_share": 0.50,
    "min_similarity_excess_vs_random": 0.05,
    "min_oos_directional_accuracy_excess_vs_generic": 0.05,
    "min_oos_median_return_excess_vs_generic_bps": 2.0,
    "min_template_net_return_excess_vs_generic_bps": 2.0,
    "permutation_p_value_max": 0.10,
    "required_positive_folds_share": 0.60,
}


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
    event_mode: str = "state_entry_non_overlapping"
    permutation_count: int = 100
    min_independent_events_per_state_horizon: int = 30
    min_train_symbols_per_state: int = 3
    min_test_symbols_per_state: int = 2
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_single_month_share: float = 0.50
    min_similarity_excess_vs_random: float = 0.05
    min_oos_directional_accuracy_excess_vs_generic: float = 0.05
    min_oos_median_return_excess_vs_generic_bps: float = 2.0
    min_template_net_return_excess_vs_generic_bps: float = 2.0
    permutation_p_value_max: float = 0.10
    required_positive_folds_share: float = 0.60
    run_template_overlay: bool = False
    template: str = "opening_range_breakout"


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
    pipeline_passed: bool
    label_similarity_supported: bool
    fingerprint_similarity_supported: bool
    state_similarity_supported: bool
    oos_similarity_supported: bool
    template_overlay_supported: bool
    decision: str
    decision_reasons: list[str]


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
    prior_session_high_to_date = session_high_to_date.groupby(data["session_date"]).shift(1)
    prior_session_low_to_date = session_low_to_date.groupby(data["session_date"]).shift(1)
    data["new_session_high"] = high.gt(prior_session_high_to_date).fillna(False)
    data["new_session_low"] = low.lt(prior_session_low_to_date).fillna(False)
    data["failed_new_high"] = (
        data["new_session_high"] & close.lt(prior_session_high_to_date)
    ).fillna(False)
    data["failed_new_low"] = (
        data["new_session_low"] & close.gt(prior_session_low_to_date)
    ).fillna(False)
    data["prior_high_break"] = high.gt(
        pd.to_numeric(data["previous_session_high"], errors="coerce")
    ).fillna(False)
    data["prior_low_break"] = low.lt(
        pd.to_numeric(data["previous_session_low"], errors="coerce")
    ).fillna(False)
    data["prior_recent_high"] = pd.to_numeric(data["recent_high"], errors="coerce").groupby(
        data["session_date"]
    ).shift(1)
    data["prior_recent_low"] = pd.to_numeric(data["recent_low"], errors="coerce").groupby(
        data["session_date"]
    ).shift(1)
    bar_index = pd.to_numeric(data["bar_index_in_session"], errors="coerce")
    minutes_from_open = pd.to_numeric(data["minutes_from_session_open"], errors="coerce")
    data["bar_index_bucket"] = pd.cut(
        bar_index,
        bins=[-1, 5, 11, 23, 47, math.inf],
        labels=["opening_range", "post_open", "morning", "midday", "late_day"],
    ).astype("string")
    data["time_of_day_bucket"] = pd.cut(
        minutes_from_open,
        bins=[-math.inf, 30, 60, 120, 300, math.inf],
        labels=["opening_range", "post_open", "morning", "midday", "late_day"],
    ).astype("string")
    pct_change = close.groupby(data["session_date"]).pct_change(fill_method=None)
    data["rolling_volatility_12"] = pct_change.groupby(data["session_date"]).transform(
        lambda series: series.rolling(12, min_periods=2).std()
    )
    data["rolling_volatility_24"] = pct_change.groupby(data["session_date"]).transform(
        lambda series: series.rolling(24, min_periods=2).std()
    )
    range_mean = bar_range.groupby(data["session_date"]).transform(
        lambda series: series.rolling(24, min_periods=2).mean()
    )
    range_std = bar_range.groupby(data["session_date"]).transform(
        lambda series: series.rolling(24, min_periods=2).std()
    )
    return_mean = pct_change.groupby(data["session_date"]).transform(
        lambda series: series.rolling(24, min_periods=2).mean()
    )
    return_std = pct_change.groupby(data["session_date"]).transform(
        lambda series: series.rolling(24, min_periods=2).std()
    )
    data["range_zscore"] = _safe_divide(bar_range - range_mean, range_std)
    data["return_zscore"] = _safe_divide(pct_change - return_mean, return_std)

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
    open_ = pd.to_numeric(data["open"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
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
    minutes_from_open = pd.to_numeric(data["minutes_from_session_open"], errors="coerce")
    dist_session_open = pd.to_numeric(data["distance_from_session_open_pct"], errors="coerce")
    dist_or_high = pd.to_numeric(data["distance_from_opening_range_high_pct"], errors="coerce")
    dist_or_low = pd.to_numeric(data["distance_from_opening_range_low_pct"], errors="coerce")
    range_zscore = pd.to_numeric(data["range_zscore"], errors="coerce")
    return_zscore = pd.to_numeric(data["return_zscore"], errors="coerce")
    opening_complete = data.get("opening_range_complete", pd.Series(False, index=data.index)).astype(
        bool
    )
    above_vwap = data.get("above_vwap", pd.Series(False, index=data.index)).astype(bool)
    below_vwap = data.get("below_vwap", pd.Series(False, index=data.index)).astype(bool)
    open_above_previous_high = data.get(
        "open_above_previous_high",
        pd.Series(False, index=data.index),
    ).astype(bool)
    open_below_previous_low = data.get(
        "open_below_previous_low",
        pd.Series(False, index=data.index),
    ).astype(bool)
    previous_high = pd.to_numeric(data["previous_session_high"], errors="coerce")
    previous_low = pd.to_numeric(data["previous_session_low"], errors="coerce")
    prior_recent_high = pd.to_numeric(data["prior_recent_high"], errors="coerce")
    prior_recent_low = pd.to_numeric(data["prior_recent_low"], errors="coerce")

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

    data["state_opening_drive_up"] = (
        can
        & minutes_from_open.between(30, 75, inclusive="both")
        & (prior_6 > 0.004)
        & (eff_6 > 0.45)
        & (dist_session_open > 0.002)
        & (close_location >= 0.60)
    )
    data["state_opening_drive_down"] = (
        can
        & minutes_from_open.between(30, 75, inclusive="both")
        & (prior_6 < -0.004)
        & (eff_6 > 0.45)
        & (dist_session_open < -0.002)
        & (close_location <= 0.40)
    )
    data["state_failed_open_high"] = (
        can
        & opening_complete
        & (high > pd.to_numeric(data["opening_range_high"], errors="coerce"))
        & (close < pd.to_numeric(data["opening_range_high"], errors="coerce"))
    )
    data["state_failed_open_low"] = (
        can
        & opening_complete
        & (low < pd.to_numeric(data["opening_range_low"], errors="coerce"))
        & (close > pd.to_numeric(data["opening_range_low"], errors="coerce"))
    )
    data["state_compression_pre_break"] = (
        can
        & (prior_12.abs() <= 0.004)
        & ((compression <= -0.35) | (range_zscore <= -0.35))
        & (eff_12 <= 0.45)
    )
    data["state_compression_breakout_up"] = (
        can
        & (close > prior_recent_high)
        & (prior_6.abs() <= 0.006)
        & ((compression <= 0.25) | (range_zscore >= 0.75) | (return_zscore >= 0.75))
        & (bar_return > 0.0)
    )
    data["state_compression_breakdown"] = (
        can
        & (close < prior_recent_low)
        & (prior_6.abs() <= 0.006)
        & ((compression <= 0.25) | (range_zscore >= 0.75) | (return_zscore <= -0.75))
        & (bar_return < 0.0)
    )
    data["state_trend_day_up"] = (
        can
        & (minutes_from_open >= 120)
        & (dist_session_open > 0.006)
        & above_vwap
        & (prior_12 > 0.006)
        & (data["new_session_high"].astype(bool) | (dist_session_high > -0.004))
    )
    data["state_trend_day_down"] = (
        can
        & (minutes_from_open >= 120)
        & (dist_session_open < -0.006)
        & below_vwap
        & (prior_12 < -0.006)
        & (data["new_session_low"].astype(bool) | (dist_session_low < 0.004))
    )
    data["state_gap_and_go_up"] = (
        can
        & open_above_previous_high
        & (close > previous_high)
        & (dist_session_open > 0.001)
        & (prior_6 > 0.002)
    )
    data["state_gap_fail_down"] = (
        can
        & (open_above_previous_high | open_below_previous_low)
        & ((close < previous_high) | (close < previous_low) | (dist_session_open < -0.002))
        & (bar_return < 0.0)
    )
    data["state_downside_liquidation"] = (
        can
        & ((prior_12 < -0.010) | (prior_6 < -0.006))
        & data["new_session_low"].astype(bool)
        & (close_location <= 0.45)
    )
    data["state_upside_extension"] = (
        can
        & ((prior_12 > 0.010) | (prior_6 > 0.006))
        & data["new_session_high"].astype(bool)
        & (close_location >= 0.55)
    )
    data["state_initiative_buying_controlled_pullback"] = (
        can
        & (prior_12 > 0.008)
        & (prior_6 > 0.002)
        & (prior_3 < 0.001)
        & (prior_3 > -0.006)
        & above_vwap
        & (close_location >= 0.45)
    )
    data["state_momentum_memory"] = (
        can
        & (prior_12 > 0.012)
        & (prior_6 > 0.003)
        & (eff_12 >= 0.50)
        & above_vwap
        & (dist_session_open > 0.004)
    )
    data["state_bullish_shock_failure"] = (
        can
        & ((prior_6 > 0.006) | data["new_session_high"].astype(bool))
        & ((upper_wick >= 0.35) | data["failed_new_high"].astype(bool))
        & ((bar_return < 0.0) | (close_location < 0.45))
    )
    data["state_elastic_recoil"] = (
        can
        & (prior_12 > 0.010)
        & (bar_return < -0.002)
        & (return_zscore < -0.75)
        & (close_location <= 0.40)
    )
    data["state_exhaustion_fade"] = (
        can
        & ((dist_session_high > -0.006) | data["new_session_high"].astype(bool))
        & ((upper_wick >= 0.35) | (close_location <= 0.45))
        & (prior_6 > 0.003)
        & (bar_return <= 0.0)
    )
    data["state_active_liquidation"] = (
        can
        & ((prior_12 < -0.010) | (prior_6 < -0.006))
        & (data["new_session_low"].astype(bool) | (dist_session_low < 0.003))
        & (close_location <= 0.45)
        & below_vwap
    )
    data["state_failed_bounce_active_liquidation"] = (
        can
        & (prior_12 < -0.008)
        & (prior_3 > prior_12 * 0.30)
        & (bar_return <= 0.0)
        & below_vwap
        & (close_location <= 0.50)
    )
    data["state_liquidation_failed_low_squeeze"] = (
        can
        & ((prior_12 < -0.010) | (prior_6 < -0.006))
        & (data["failed_new_low"].astype(bool) | (dist_session_low < 0.006))
        & (bar_return > 0.001)
        & (close_location >= 0.60)
    )
    data["state_slow_snapback"] = (
        can
        & (prior_12 < -0.006)
        & (prior_3 > 0.0)
        & (eff_6 <= 0.50)
        & (close_location >= 0.45)
        & (bar_return >= -0.001)
    )
    data["state_failed_open_long_block"] = (
        can
        & opening_complete
        & (minutes_from_open <= 120)
        & data["state_failed_open_high"].astype(bool)
        & (bar_return <= 0.0)
    )
    data["state_failed_open_down_continuation"] = (
        can
        & opening_complete
        & (minutes_from_open <= 120)
        & ((close < pd.to_numeric(data["opening_range_low"], errors="coerce")) | (prior_6 < -0.004))
        & below_vwap
    )
    data["state_compression_breakout"] = data["state_compression_breakout_up"]

    data["primary_state_label"] = "unclassified"
    priority = [
        ("state_liquidation_failed_low_recovery", "liquidation_failed_low_recovery"),
        ("state_extension_exhaustion", "extension_exhaustion"),
        ("state_initiative_buying_continuation", "initiative_buying_continuation"),
        ("state_dead_chop", "dead_chop"),
        ("state_failed_open_down_continuation", "failed_open_down_continuation"),
        ("state_failed_open_long_block", "failed_open_long_block"),
        ("state_gap_fail_down", "gap_fail_down"),
        ("state_gap_and_go_up", "gap_and_go_up"),
        ("state_bullish_shock_failure", "bullish_shock_failure"),
        ("state_elastic_recoil", "elastic_recoil"),
        ("state_exhaustion_fade", "exhaustion_fade"),
        ("state_failed_open_high", "failed_open_high"),
        ("state_failed_open_low", "failed_open_low"),
        ("state_liquidation_failed_low_squeeze", "liquidation_failed_low_squeeze"),
        ("state_failed_bounce_active_liquidation", "failed_bounce_active_liquidation"),
        ("state_active_liquidation", "active_liquidation"),
        ("state_downside_liquidation", "downside_liquidation"),
        ("state_upside_extension", "upside_extension"),
        ("state_compression_breakout", "compression_breakout"),
        ("state_compression_breakout_up", "compression_breakout_up"),
        ("state_compression_breakdown", "compression_breakdown"),
        ("state_opening_drive_up", "opening_drive_up"),
        ("state_opening_drive_down", "opening_drive_down"),
        ("state_trend_day_up", "trend_day_up"),
        ("state_trend_day_down", "trend_day_down"),
        ("state_initiative_buying_controlled_pullback", "initiative_buying_controlled_pullback"),
        ("state_momentum_memory", "momentum_memory"),
        ("state_slow_snapback", "slow_snapback"),
        ("state_compression_pre_break", "compression_pre_break"),
    ]
    for column, label in reversed(priority):
        data.loc[data[column].astype(bool), "primary_state_label"] = label
    data["state_unclassified"] = data["primary_state_label"].eq("unclassified")
    state_columns = [f"state_{label}" for label in STATE_LABELS if f"state_{label}" in data]
    overlap_counts = data[state_columns].astype(bool).sum(axis=1) if state_columns else 0
    data["state_overlap_count"] = overlap_counts
    data["state_overlap_labels"] = ""
    for label in STATE_LABELS:
        column = f"state_{label}"
        if column not in data:
            continue
        mask = data[column].astype(bool)
        data.loc[mask, "state_overlap_labels"] = (
            data.loc[mask, "state_overlap_labels"].astype(str)
            + np.where(data.loc[mask, "state_overlap_labels"].astype(str).eq(""), "", "|")
            + label
        )
    data["state_priority_conflict"] = pd.to_numeric(
        data["state_overlap_count"],
        errors="coerce",
    ).gt(1)
    for field in (
        "state_family",
        "state_subtype",
        "state_direction",
        "state_energy",
        "state_location",
        "stimulus_label",
    ):
        data[field] = data["primary_state_label"].map(
            {label: values[field] for label, values in STATE_DEFINITIONS.items()}
        )
    data.loc[data["primary_state_label"].eq("unclassified"), [
        "state_family",
        "state_subtype",
        "state_direction",
        "state_energy",
        "state_location",
        "stimulus_label",
    ]] = ["unclassified", "unclassified", "neutral", "unknown", "unknown", "none"]
    group_keys = ["symbol", "session_date"] if "symbol" in data else ["session_date"]
    data["previous_state_label"] = (
        data.groupby(group_keys)["primary_state_label"].shift(1).fillna("session_start")
    )
    data["state_transition_from"] = data["previous_state_label"]
    data["state_transition_to"] = data["primary_state_label"]
    data["state_transition"] = data["state_transition_from"] + "->" + data["state_transition_to"]
    data["state_entry"] = data["primary_state_label"].ne(data["state_transition_from"]) & data[
        "primary_state_label"
    ].ne("unclassified")
    data["stimulus_event"] = data["stimulus_label"].ne("none") & data["state_entry"].astype(bool)
    segment_id = data.groupby(group_keys)["primary_state_label"].transform(
        lambda series: series.ne(series.shift()).cumsum()
    )
    data["_state_segment_id"] = segment_id
    data["state_age_bars"] = data.groupby(group_keys + ["_state_segment_id"]).cumcount()
    data["state_duration_bars"] = data["state_age_bars"] + 1
    data["state_duration_bars_so_far"] = data["state_duration_bars"]
    next_state = data.groupby(group_keys)["primary_state_label"].shift(-1).fillna("session_end")
    data["state_exit"] = data["primary_state_label"].ne(next_state) & data[
        "primary_state_label"
    ].ne("unclassified")
    data = data.drop(columns=["_state_segment_id"])
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


def build_horizon_events(frame: pd.DataFrame, config: BehavioralStateConfig) -> pd.DataFrame:
    """Convert wide forward responses into horizon-specific event rows."""

    max_horizon = max(config.horizons) if config.horizons else 0
    path_columns = [f"path_return_{step}" for step in range(1, max_horizon + 1)]
    normalized_path_columns = [
        f"normalized_path_return_{step}" for step in range(1, max_horizon + 1)
    ]
    columns = [
        "symbol",
        "timestamp",
        "session_date",
        "bar_index_in_session",
        "bar_index_bucket",
        "time_of_day_bucket",
        "primary_state_label",
        "state_family",
        "state_subtype",
        "state_direction",
        "state_energy",
        "state_location",
        "stimulus_label",
        "previous_state_label",
        "state_transition",
        "state_transition_from",
        "state_transition_to",
        "state_entry",
        "state_exit",
        "stimulus_event",
        "state_age_bars",
        "state_duration_bars",
        "state_duration_bars_so_far",
        "state_overlap_count",
        "state_overlap_labels",
        "state_priority_conflict",
        "response_horizon",
        "response_return",
        "response_mfe",
        "response_mae",
        "final_return",
        "mfe",
        "mae",
        "time_to_mfe",
        "time_to_mae",
        "recoil_ratio",
        "continuation_score",
        "failure_score",
        "path_scale",
        *path_columns,
        *normalized_path_columns,
        "raw_row_index",
        *DEFAULT_SIMILARITY_FEATURE_COLUMNS,
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    data = frame.copy()
    if "state_entry" not in data:
        group_keys = ["symbol", "session_date"] if "symbol" in data else ["session_date"]
        data["state_transition_from"] = (
            data.groupby(group_keys)["primary_state_label"].shift(1).fillna("session_start")
        )
        data["state_transition_to"] = data["primary_state_label"]
        data["state_entry"] = data["primary_state_label"].ne(data["state_transition_from"])
    for column, fallback in {
        "bar_index_bucket": "",
        "time_of_day_bucket": "",
        "state_family": "unknown",
        "state_subtype": "unknown",
        "state_direction": "neutral",
        "state_energy": "unknown",
        "state_location": "unknown",
        "stimulus_label": "none",
        "previous_state_label": "",
        "state_transition": "",
        "state_transition_from": "",
        "state_transition_to": "",
        "state_exit": False,
        "stimulus_event": False,
        "state_age_bars": 0,
        "state_duration_bars": 0,
        "state_duration_bars_so_far": 0,
        "state_overlap_count": 0,
        "state_overlap_labels": "",
        "state_priority_conflict": False,
    }.items():
        if column not in data:
            data[column] = fallback
    if "previous_state_label" in data and not data["previous_state_label"].astype(str).any():
        data["previous_state_label"] = data["state_transition_from"]
    if "state_transition" in data and not data["state_transition"].astype(str).any():
        data["state_transition"] = (
            data["state_transition_from"].astype(str) + "->" + data["state_transition_to"].astype(str)
        )

    close = pd.to_numeric(data["close"], errors="coerce") if "close" in data else pd.Series(
        np.nan,
        index=data.index,
    )
    path_scale = pd.to_numeric(
        data.get("rolling_intraday_range_pct", pd.Series(np.nan, index=data.index)),
        errors="coerce",
    )
    path_scale = path_scale.fillna(
        pd.to_numeric(
            data.get("rolling_volatility_12", pd.Series(np.nan, index=data.index)),
            errors="coerce",
        )
    )
    path_scale = path_scale.fillna(
        pd.to_numeric(data.get("bar_range_pct", pd.Series(np.nan, index=data.index)), errors="coerce")
    )
    path_scale = path_scale.replace(0.0, np.nan).abs()
    data["path_scale"] = path_scale
    path_group_keys = ["symbol", "session_date"] if "symbol" in data else ["session_date"]
    for step in range(1, max_horizon + 1):
        future_close = data.groupby(path_group_keys)["close"].shift(-step) if "close" in data else close
        path_return = _safe_pct_distance(pd.to_numeric(future_close, errors="coerce"), close)
        data[f"path_return_{step}"] = path_return
        data[f"normalized_path_return_{step}"] = _safe_divide(path_return, path_scale)

    rows: list[pd.DataFrame] = []
    base_mask = data["can_evaluate_state"].astype(bool) & data["primary_state_label"].ne(
        "unclassified"
    )
    for horizon in config.horizons:
        return_column = f"forward_{horizon}_bar_return"
        mfe_column = f"forward_{horizon}_bar_mfe"
        mae_column = f"forward_{horizon}_bar_mae"
        if return_column not in data:
            continue
        valid = data[base_mask & data[return_column].notna()].copy()
        if valid.empty:
            continue
        valid["response_horizon"] = int(horizon)
        valid["response_return"] = pd.to_numeric(valid[return_column], errors="coerce")
        valid["response_mfe"] = pd.to_numeric(
            valid.get(mfe_column, pd.Series(np.nan, index=valid.index)),
            errors="coerce",
        )
        valid["response_mae"] = pd.to_numeric(
            valid.get(mae_column, pd.Series(np.nan, index=valid.index)),
            errors="coerce",
        )
        horizon_path_columns = [f"path_return_{step}" for step in range(1, horizon + 1)]
        path_frame = valid[horizon_path_columns].apply(pd.to_numeric, errors="coerce")
        valid["final_return"] = path_frame.iloc[:, -1]
        valid["mfe"] = path_frame.max(axis=1)
        valid["mae"] = path_frame.min(axis=1)
        valid["time_to_mfe"] = path_frame.apply(
            lambda row: int(np.nanargmax(row.to_numpy(dtype=float)) + 1)
            if row.notna().any()
            else math.nan,
            axis=1,
        )
        valid["time_to_mae"] = path_frame.apply(
            lambda row: int(np.nanargmin(row.to_numpy(dtype=float)) + 1)
            if row.notna().any()
            else math.nan,
            axis=1,
        )
        valid["recoil_ratio"] = _safe_divide(valid["mfe"] - valid["final_return"], valid["mfe"].abs())
        valid["continuation_score"] = _safe_divide(valid["final_return"], path_frame.abs().max(axis=1))
        valid["failure_score"] = _safe_divide(-valid["final_return"], path_frame.abs().max(axis=1))
        valid["raw_row_index"] = valid.index.astype(int)
        rows.append(valid)
    if not rows:
        return pd.DataFrame(columns=columns)
    events = pd.concat(rows, ignore_index=True)
    return events.reindex(columns=columns)


def _event_sort_columns(events: pd.DataFrame) -> list[str]:
    columns = [column for column in ("symbol", "session_date", "response_horizon") if column in events]
    if "timestamp" in events:
        columns.append("timestamp")
    elif "bar_index_in_session" in events:
        columns.append("bar_index_in_session")
    return columns


def _add_event_count_columns(
    selected: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    if selected.empty:
        output = selected.copy()
        output["raw_row_count"] = pd.Series(dtype="int")
        output["independent_event_count"] = pd.Series(dtype="int")
        return output
    group_columns = ["symbol", "primary_state_label", "response_horizon"]
    if "session_date" in source:
        group_columns.append("session_date")
    raw_counts = source.groupby(group_columns, dropna=False).size().rename("raw_row_count")
    independent_counts = selected.groupby(group_columns, dropna=False).size().rename(
        "independent_event_count"
    )
    output = selected.copy()
    key_frame = output[group_columns]
    output["raw_row_count"] = [
        int(raw_counts.loc[tuple(row)]) if tuple(row) in raw_counts.index else 0
        for row in key_frame.itertuples(index=False, name=None)
    ]
    output["independent_event_count"] = [
        int(independent_counts.loc[tuple(row)]) if tuple(row) in independent_counts.index else 0
        for row in key_frame.itertuples(index=False, name=None)
    ]
    return output


def _enforce_horizon_embargo(events: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    group_columns = ["symbol", "primary_state_label", "response_horizon"]
    if "session_date" in events:
        group_columns.append("session_date")
    sorted_events = events.sort_values(_event_sort_columns(events)).reset_index(drop=True)
    for _, group in sorted_events.groupby(group_columns, sort=False, dropna=False):
        last_bar: float | None = None
        for _, row in group.iterrows():
            horizon = int(row["response_horizon"])
            bar_index = float(row.get("bar_index_in_session", len(selected_rows)))
            if last_bar is not None and bar_index < last_bar + horizon:
                continue
            selected_rows.append(row)
            last_bar = bar_index
    if not selected_rows:
        return events.iloc[0:0].copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def extract_independent_events(events: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    """Extract independent evidence events from horizon-specific rows."""

    if mode not in EVENT_MODES:
        raise ValueError(f"event mode must be one of {sorted(EVENT_MODES)}")
    if events.empty:
        return _add_event_count_columns(events.copy(), events)

    source = events.sort_values(_event_sort_columns(events)).reset_index(drop=True)
    selected = source.copy()
    if mode in {"state_entry_only", "state_entry_non_overlapping"}:
        selected = selected[selected["state_entry"].astype(bool)].copy()
    elif mode == "state_change_only":
        selected = selected[
            selected["primary_state_label"].ne(selected["state_transition_from"])
        ].copy()
    elif mode == "stimulus_event_only":
        selected = selected[selected["stimulus_event"].astype(bool)].copy()

    if mode in {"non_overlapping_by_horizon", "state_entry_non_overlapping"}:
        selected = _enforce_horizon_embargo(selected)
    return _add_event_count_columns(selected.reset_index(drop=True), source)


def _empty_horizon_state_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "state",
            "horizon",
            "event_count",
            "symbol_count",
            "session_count",
            "median_forward_return",
            "win_rate",
            "median_mfe",
            "median_mae",
        ]
    )


def summarize_horizon_state_responses(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize horizon-specific event rows by state and horizon."""

    if events.empty:
        return _empty_horizon_state_summary()
    rows: list[dict[str, Any]] = []
    for (state, horizon), group in events.groupby(
        ["primary_state_label", "response_horizon"],
        sort=True,
    ):
        returns = pd.to_numeric(group["response_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        rows.append(
            {
                "state": str(state),
                "horizon": int(horizon),
                "event_count": int(len(returns)),
                "symbol_count": int(group["symbol"].nunique()) if "symbol" in group else 0,
                "session_count": int(group["session_date"].nunique())
                if "session_date" in group
                else 0,
                "median_forward_return": float(returns.median()),
                "win_rate": float((returns > 0.0).mean()),
                "median_mfe": float(pd.to_numeric(group["response_mfe"], errors="coerce").median()),
                "median_mae": float(pd.to_numeric(group["response_mae"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows, columns=_empty_horizon_state_summary().columns)


def _walk_forward_mask(events: pd.DataFrame, train_fraction: float = 0.60) -> pd.Series:
    if events.empty:
        return pd.Series(dtype=bool)
    timestamps = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
    sorted_positions = timestamps.sort_values(kind="mergesort").index.tolist()
    train_count = max(1, min(len(sorted_positions) - 1, int(len(sorted_positions) * train_fraction)))
    train_indices = set(sorted_positions[:train_count])
    return pd.Series([index in train_indices for index in events.index], index=events.index)


def _expected_direction(returns: pd.Series) -> int:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0
    median = float(clean.median())
    if median > 0.0:
        return 1
    if median < 0.0:
        return -1
    win_rate = float((clean > 0.0).mean())
    if win_rate > 0.50:
        return 1
    if win_rate < 0.50:
        return -1
    return 0


def _directional_accuracy(returns: pd.Series, direction: int) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty or direction == 0:
        return math.nan
    return float((np.sign(clean) == direction).mean())


def _run_oos_for_split(
    events: pd.DataFrame,
    *,
    config: BehavioralStateConfig,
    train_mask: pd.Series,
    split_mode: str,
    fold: str,
) -> pd.DataFrame:
    train = events[train_mask].copy()
    test = events[~train_mask].copy()
    columns = [
        "split_mode",
        "fold",
        "state",
        "horizon",
        "train_event_count",
        "test_event_count",
        "train_symbol_count",
        "test_symbol_count",
        "train_median_return",
        "test_median_return",
        "expected_direction",
        "directional_accuracy",
        "generic_directional_accuracy",
        "directional_accuracy_excess_vs_generic",
        "generic_train_median_return",
        "generic_test_median_return",
        "oos_median_return_excess_vs_generic_bps",
        "gate_passed",
        "verdict",
    ]
    if train.empty or test.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for (state, horizon), state_train in train.groupby(
        ["primary_state_label", "response_horizon"],
        sort=True,
    ):
        state_test = test[
            (test["primary_state_label"] == state) & (test["response_horizon"] == horizon)
        ]
        if state_test.empty:
            continue
        train_returns = pd.to_numeric(state_train["response_return"], errors="coerce").dropna()
        test_returns = pd.to_numeric(state_test["response_return"], errors="coerce").dropna()
        generic_train = train[train["response_horizon"] == horizon]
        generic_test = test[test["response_horizon"] == horizon]
        generic_train_returns = pd.to_numeric(
            generic_train["response_return"],
            errors="coerce",
        ).dropna()
        generic_test_returns = pd.to_numeric(
            generic_test["response_return"],
            errors="coerce",
        ).dropna()
        if train_returns.empty or test_returns.empty or generic_train_returns.empty:
            continue
        direction = _expected_direction(train_returns)
        generic_direction = _expected_direction(generic_train_returns)
        accuracy = _directional_accuracy(test_returns, direction)
        generic_accuracy = _directional_accuracy(generic_test_returns, generic_direction)
        test_median = float(test_returns.median())
        generic_test_median = (
            float(generic_test_returns.median()) if not generic_test_returns.empty else math.nan
        )
        aligned_state = direction * test_median if direction else 0.0
        aligned_generic = generic_direction * generic_test_median if generic_direction else 0.0
        excess_bps = (aligned_state - aligned_generic) * 10_000
        train_symbol_count = int(state_train["symbol"].nunique()) if "symbol" in state_train else 0
        test_symbol_count = int(state_test["symbol"].nunique()) if "symbol" in state_test else 0
        gate_passed = bool(
            len(state_test) >= config.min_independent_events_per_state_horizon
            and train_symbol_count >= config.min_train_symbols_per_state
            and test_symbol_count >= config.min_test_symbols_per_state
            and not math.isnan(accuracy)
            and not math.isnan(generic_accuracy)
            and accuracy - generic_accuracy
            >= config.min_oos_directional_accuracy_excess_vs_generic
            and excess_bps >= config.min_oos_median_return_excess_vs_generic_bps
        )
        rows.append(
            {
                "split_mode": split_mode,
                "fold": fold,
                "state": str(state),
                "horizon": int(horizon),
                "train_event_count": int(len(train_returns)),
                "test_event_count": int(len(test_returns)),
                "train_symbol_count": train_symbol_count,
                "test_symbol_count": test_symbol_count,
                "train_median_return": float(train_returns.median()),
                "test_median_return": test_median,
                "expected_direction": int(direction),
                "directional_accuracy": accuracy,
                "generic_directional_accuracy": generic_accuracy,
                "directional_accuracy_excess_vs_generic": accuracy - generic_accuracy
                if not math.isnan(accuracy) and not math.isnan(generic_accuracy)
                else math.nan,
                "generic_train_median_return": float(generic_train_returns.median()),
                "generic_test_median_return": generic_test_median,
                "oos_median_return_excess_vs_generic_bps": float(excess_bps),
                "gate_passed": gate_passed,
                "verdict": "continue_research" if gate_passed else "mixed_response",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_oos_state_response_test(
    events: pd.DataFrame,
    *,
    config: BehavioralStateConfig,
    split_mode: str = "walk_forward",
) -> pd.DataFrame:
    """Estimate state response on train rows only, then evaluate held-out rows."""

    if events.empty:
        return _run_oos_for_split(
            events,
            config=config,
            train_mask=pd.Series(dtype=bool),
            split_mode=split_mode,
            fold="empty",
        )
    data = events.sort_values("timestamp").reset_index(drop=True)
    if split_mode == "walk_forward":
        return _run_oos_for_split(
            data,
            config=config,
            train_mask=_walk_forward_mask(data),
            split_mode=split_mode,
            fold="time_split_60_40",
        )
    if split_mode == "leave_one_symbol_out":
        rows: list[pd.DataFrame] = []
        for symbol in sorted(map(str, data["symbol"].dropna().unique().tolist())):
            train_mask = data["symbol"].ne(symbol)
            rows.append(
                _run_oos_for_split(
                    data,
                    config=config,
                    train_mask=train_mask,
                    split_mode=split_mode,
                    fold=symbol,
                )
            )
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    raise ValueError("split_mode must be 'walk_forward' or 'leave_one_symbol_out'")


def build_permutation_baseline(
    events: pd.DataFrame,
    *,
    config: BehavioralStateConfig,
    permutation_count: int | None = None,
) -> pd.DataFrame:
    """Shuffle state labels inside local symbol/time buckets and score null labels."""

    columns = [
        "baseline",
        "state",
        "horizon",
        "observed_event_count",
        "permuted_event_count",
        "observed_median_return",
        "permutation_median_return_mean",
        "permutation_median_return_p05",
        "permutation_median_return_p95",
        "permutation_percentile",
        "permutation_p_value",
        "permutation_count",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    count = int(config.permutation_count if permutation_count is None else permutation_count)
    count = max(1, count)
    data = events.copy().reset_index(drop=True)
    if "bar_index_bucket" not in data:
        data["bar_index_bucket"] = ""
    if "time_of_day_bucket" not in data:
        data["time_of_day_bucket"] = ""
    rng = np.random.default_rng(config.random_seed)
    states = sorted(map(str, data["primary_state_label"].dropna().unique().tolist()))
    rows: list[dict[str, Any]] = []
    baselines = {
        "shuffle_within_symbol_bar_bucket": ["symbol", "bar_index_bucket"],
        "shuffle_within_symbol_time_bucket": ["symbol", "time_of_day_bucket"],
    }
    observed_stats = (
        data.groupby(["primary_state_label", "response_horizon"], sort=False)["response_return"]
        .agg(["count", "median"])
        .rename(columns={"count": "observed_event_count", "median": "observed_median_return"})
    )
    keys = [(state, int(horizon)) for state in states for horizon in config.horizons]
    for baseline_name, group_columns in baselines.items():
        bucket_indices = [
            group.index.to_numpy()
            for _, group in data.groupby(group_columns, sort=False, dropna=False)
        ]
        permuted_medians: dict[tuple[str, int], list[float]] = {key: [] for key in keys}
        permuted_counts: dict[tuple[str, int], list[int]] = {key: [] for key in keys}
        original_labels = data["primary_state_label"].astype(str).to_numpy(copy=True)
        for _ in range(count):
            shuffled_labels = original_labels.copy()
            for indices in bucket_indices:
                labels = shuffled_labels[indices].copy()
                rng.shuffle(labels)
                shuffled_labels[indices] = labels
            permuted = pd.DataFrame(
                {
                    "state": shuffled_labels,
                    "horizon": data["response_horizon"].to_numpy(),
                    "response_return": data["response_return"].to_numpy(),
                }
            )
            permuted_stats = permuted.groupby(["state", "horizon"], sort=False)[
                "response_return"
            ].agg(["count", "median"])
            for key in keys:
                if key in permuted_stats.index:
                    stat = permuted_stats.loc[key]
                    permuted_counts[key].append(int(stat["count"]))
                    permuted_medians[key].append(float(stat["median"]))
                else:
                    permuted_counts[key].append(0)
                    permuted_medians[key].append(math.nan)
        for key in keys:
            state, horizon = key
            if key not in observed_stats.index:
                continue
            observed = observed_stats.loc[key]
            if int(observed["observed_event_count"]) <= 0:
                continue
            observed_median = float(observed["observed_median_return"])
            perm_series = pd.Series(permuted_medians[key], dtype="float").dropna()
            if perm_series.empty:
                p_value = math.nan
                percentile = math.nan
                perm_mean = math.nan
                perm_p05 = math.nan
                perm_p95 = math.nan
            else:
                p_value = float(
                    ((perm_series.abs() >= abs(observed_median)).sum() + 1)
                    / (len(perm_series) + 1)
                )
                percentile = float((perm_series <= observed_median).mean())
                perm_mean = float(perm_series.mean())
                perm_p05 = float(perm_series.quantile(0.05))
                perm_p95 = float(perm_series.quantile(0.95))
            rows.append(
                {
                    "baseline": baseline_name,
                    "state": state,
                    "horizon": int(horizon),
                    "observed_event_count": int(observed["observed_event_count"]),
                    "permuted_event_count": int(round(float(np.nanmedian(permuted_counts[key]))))
                    if permuted_counts[key]
                    else 0,
                    "observed_median_return": observed_median,
                    "permutation_median_return_mean": perm_mean,
                    "permutation_median_return_p05": perm_p05,
                    "permutation_median_return_p95": perm_p95,
                    "permutation_percentile": percentile,
                    "permutation_p_value": p_value,
                    "permutation_count": count,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _validate_similarity_feature_columns(feature_columns: list[str]) -> None:
    bad_columns = [
        column
        for column in feature_columns
        if column.startswith("forward_")
        or column.startswith("response_")
        or column.startswith("path_return_")
        or column.startswith("normalized_path_return_")
        or column
        in {
            "final_return",
            "time_to_mfe",
            "time_to_mae",
            "recoil_ratio",
            "continuation_score",
            "failure_score",
        }
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


def run_nearest_neighbor_oos_similarity(
    events: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: BehavioralStateConfig,
) -> pd.DataFrame:
    """Fit scaler and nearest-neighbor index on train rows only, query test rows only."""

    columns = [
        "state",
        "horizon",
        "fit_scope",
        "train_row_count",
        "test_row_count",
        "match_count",
        "cross_symbol_matches",
        "cross_symbol_match_sign_agreement",
        "median_self_forward_return",
        "median_neighbor_forward_return",
        "average_abs_response_diff",
        "verdict",
    ]
    _validate_similarity_feature_columns(feature_columns)
    if events.empty:
        return pd.DataFrame(columns=columns)
    available_features = [column for column in feature_columns if column in events.columns]
    if not available_features:
        return pd.DataFrame(columns=columns)

    data = events.copy().reset_index(drop=True)
    train_mask = _walk_forward_mask(data)
    train = data[train_mask].reset_index(drop=True)
    test = data[~train_mask].reset_index(drop=True)
    if len(train) < 2 or test.empty:
        return pd.DataFrame(columns=columns)

    train_features = train[available_features].apply(pd.to_numeric, errors="coerce")
    test_features = test[available_features].apply(pd.to_numeric, errors="coerce")
    medians = train_features.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    train_features = train_features.replace([np.inf, -np.inf], np.nan).fillna(medians)
    test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(medians)
    valid_train = train_features.notna().all(axis=1)
    valid_test = test_features.notna().all(axis=1)
    train = train.loc[valid_train].reset_index(drop=True)
    test = test.loc[valid_test].reset_index(drop=True)
    train_features = train_features.loc[valid_train].reset_index(drop=True)
    test_features = test_features.loc[valid_test].reset_index(drop=True)
    if len(train) < 2 or test.empty:
        return pd.DataFrame(columns=columns)

    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError:
        return pd.DataFrame(columns=columns)

    scaler = StandardScaler().fit(train_features.to_numpy(dtype=float))
    train_scaled = scaler.transform(train_features.to_numpy(dtype=float))
    test_scaled = scaler.transform(test_features.to_numpy(dtype=float))
    neighbor_count = min(len(train), max(1, config.nearest_neighbors))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count)
    neighbors.fit(train_scaled)
    _, indices = neighbors.kneighbors(test_scaled)

    details: list[dict[str, Any]] = []
    for test_position, neighbor_positions in enumerate(indices):
        source = test.iloc[test_position]
        selected = train.iloc[[int(position) for position in neighbor_positions]]
        for horizon in config.horizons:
            if int(source["response_horizon"]) != int(horizon):
                continue
            same_horizon = selected[selected["response_horizon"].eq(horizon)]
            if same_horizon.empty:
                continue
            neighbor_values = pd.to_numeric(
                same_horizon["response_return"],
                errors="coerce",
            ).dropna()
            if neighbor_values.empty:
                continue
            self_response = float(source["response_return"])
            cross_symbol = same_horizon[same_horizon["symbol"].ne(source["symbol"])]
            cross_values = [
                float(value)
                for value in pd.to_numeric(cross_symbol["response_return"], errors="coerce").dropna()
            ]
            neighbor_median = float(neighbor_values.median())
            details.append(
                {
                    "state": str(source["primary_state_label"]),
                    "horizon": int(horizon),
                    "self_forward_return": self_response,
                    "neighbor_median_forward_return": neighbor_median,
                    "cross_symbol_sign_agreement": _sign_agreement(cross_values, self_response),
                    "absolute_response_difference": abs(neighbor_median - self_response),
                    "cross_symbol_neighbor_count": int(len(cross_values)),
                }
            )
    if not details:
        return pd.DataFrame(columns=columns)
    detail_frame = pd.DataFrame(details)
    rows: list[dict[str, Any]] = []
    for (state, horizon), group in detail_frame.groupby(["state", "horizon"], sort=True):
        cross_agreement = pd.to_numeric(
            group["cross_symbol_sign_agreement"],
            errors="coerce",
        ).dropna()
        cross_mean = float(cross_agreement.mean()) if not cross_agreement.empty else math.nan
        verdict = (
            "cross_symbol_similarity"
            if not math.isnan(cross_mean) and cross_mean >= 0.58
            else "mixed_response"
        )
        rows.append(
            {
                "state": str(state),
                "horizon": int(horizon),
                "fit_scope": "train_only",
                "train_row_count": int(len(train)),
                "test_row_count": int(len(test)),
                "match_count": int(len(group)),
                "cross_symbol_matches": int((group["cross_symbol_neighbor_count"] > 0).sum()),
                "cross_symbol_match_sign_agreement": cross_mean,
                "median_self_forward_return": float(group["self_forward_return"].median()),
                "median_neighbor_forward_return": float(
                    group["neighbor_median_forward_return"].median()
                ),
                "average_abs_response_diff": float(group["absolute_response_difference"].mean()),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _path_columns_for_horizon(events: pd.DataFrame, horizon: int) -> list[str]:
    normalized = [f"normalized_path_return_{step}" for step in range(1, horizon + 1)]
    if all(column in events for column in normalized):
        return normalized
    raw = [f"path_return_{step}" for step in range(1, horizon + 1)]
    return [column for column in raw if column in events]


def _path_vector(row: pd.Series, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.array([], dtype=float)
    values = pd.to_numeric(row.reindex(columns), errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).all():
        return np.array([], dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _shape_metrics(source_vector: np.ndarray, match_vector: np.ndarray) -> dict[str, float]:
    if source_vector.size == 0 or match_vector.size == 0 or source_vector.size != match_vector.size:
        return {
            "path_correlation": math.nan,
            "cosine_similarity": math.nan,
            "euclidean_distance": math.nan,
        }
    source_std = float(np.std(source_vector))
    match_std = float(np.std(match_vector))
    if source_std > 0.0 and match_std > 0.0:
        correlation = float(np.corrcoef(source_vector, match_vector)[0, 1])
    else:
        correlation = math.nan
    source_norm = float(np.linalg.norm(source_vector))
    match_norm = float(np.linalg.norm(match_vector))
    cosine = (
        float(np.dot(source_vector, match_vector) / (source_norm * match_norm))
        if source_norm > 0.0 and match_norm > 0.0
        else math.nan
    )
    return {
        "path_correlation": correlation,
        "cosine_similarity": cosine,
        "euclidean_distance": float(np.linalg.norm(source_vector - match_vector)),
    }


def _feature_matrix(events: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    available = [column for column in feature_columns if column in events.columns]
    if not available:
        return pd.DataFrame(index=events.index)
    features = events[available].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    medians = features.median(numeric_only=True)
    features = features.fillna(medians)
    std = features.std(ddof=0).replace(0.0, np.nan)
    return ((features - features.mean(numeric_only=True)) / std).fillna(0.0)


def _fit_state_fingerprint(
    events: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    _validate_similarity_feature_columns(feature_columns)
    numeric_columns = [
        column
        for column in feature_columns
        if column in events.columns and column not in STATE_FINGERPRINT_CATEGORICAL_COLUMNS
    ]
    categorical_columns = [
        column
        for column in feature_columns
        if column in events.columns and column in STATE_FINGERPRINT_CATEGORICAL_COLUMNS
    ]
    numeric = events[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame(index=events.index)
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = numeric.median(numeric_only=True) if not numeric.empty else pd.Series(dtype=float)
    filled = numeric.fillna(medians) if not numeric.empty else numeric
    means = filled.mean(numeric_only=True) if not filled.empty else pd.Series(dtype=float)
    stds = filled.std(ddof=0).replace(0.0, np.nan) if not filled.empty else pd.Series(dtype=float)
    levels = {
        column: sorted(
            events[column].astype("string").fillna("missing").astype(str).unique().tolist()
        )
        for column in categorical_columns
    }
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "medians": medians,
        "means": means,
        "stds": stds,
        "levels": levels,
    }


def _transform_state_fingerprint(events: pd.DataFrame, fit: dict[str, Any]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    numeric_columns = list(fit["numeric_columns"])
    if numeric_columns:
        numeric = events.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(fit["medians"])
        scaled = ((numeric - fit["means"]) / fit["stds"]).replace([np.inf, -np.inf], np.nan)
        pieces.append(scaled.fillna(0.0))
    for column in fit["categorical_columns"]:
        values = events[column].astype("string").fillna("missing").astype(str)
        for level in fit["levels"][column]:
            pieces.append(
                pd.DataFrame(
                    {f"{column}={level}": values.eq(level).astype(float)},
                    index=events.index,
                )
            )
    if not pieces:
        return pd.DataFrame(index=events.index)
    return pd.concat(pieces, axis=1).fillna(0.0)


def _state_fingerprint_matrix(
    events: pd.DataFrame,
    feature_columns: list[str],
    *,
    fit: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fitted = _fit_state_fingerprint(events, feature_columns) if fit is None else fit
    return _transform_state_fingerprint(events, fitted), fitted


def _feature_distance(
    feature_frame: pd.DataFrame,
    source_index: int,
    candidate_indices: pd.Index,
) -> pd.Series:
    if feature_frame.empty:
        return pd.Series(0.0, index=candidate_indices)
    source = feature_frame.loc[source_index].to_numpy(dtype=float)
    candidates = feature_frame.loc[candidate_indices].to_numpy(dtype=float)
    distances = np.linalg.norm(candidates - source, axis=1)
    return pd.Series(distances, index=candidate_indices)


def _match_row(
    *,
    source: pd.Series,
    match: pd.Series,
    source_vector: np.ndarray,
    match_vector: np.ndarray,
    feature_distance: float,
    baseline: str,
) -> dict[str, Any]:
    shape = _shape_metrics(source_vector, match_vector)
    return {
        "baseline": baseline,
        "source_symbol": str(source["symbol"]),
        "source_timestamp": str(source["timestamp"]),
        "source_state": str(source["primary_state_label"]),
        "source_stimulus": str(source.get("stimulus_label", "")),
        "source_horizon": int(source["response_horizon"]),
        "match_symbol": str(match["symbol"]),
        "match_timestamp": str(match["timestamp"]),
        "match_state": str(match["primary_state_label"]),
        "match_stimulus": str(match.get("stimulus_label", "")),
        "feature_distance": float(feature_distance),
        "path_correlation": shape["path_correlation"],
        "cosine_similarity": shape["cosine_similarity"],
        "euclidean_distance": shape["euclidean_distance"],
        "source_response_return": float(source["response_return"]),
        "match_response_return": float(match["response_return"]),
        "source_mfe": float(source.get("response_mfe", math.nan)),
        "match_mfe": float(match.get("response_mfe", math.nan)),
        "source_mae": float(source.get("response_mae", math.nan)),
        "match_mae": float(match.get("response_mae", math.nan)),
    }


def _empty_same_state_match_rows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "baseline",
            "source_symbol",
            "source_timestamp",
            "source_state",
            "source_stimulus",
            "source_horizon",
            "match_symbol",
            "match_timestamp",
            "match_state",
            "match_stimulus",
            "feature_distance",
            "path_correlation",
            "cosine_similarity",
            "euclidean_distance",
            "source_response_return",
            "match_response_return",
            "source_mfe",
            "match_mfe",
            "source_mae",
            "match_mae",
        ]
    )


def _summarize_shape_rows(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "baseline",
        "state",
        "horizon",
        "match_count",
        "median_path_correlation",
        "median_cosine_similarity",
        "median_euclidean_distance",
        "median_abs_return_diff",
        "median_source_return",
        "median_match_return",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    output_rows: list[dict[str, Any]] = []
    for (baseline, state, horizon), group in rows.groupby(
        ["baseline", "source_state", "source_horizon"],
        sort=True,
    ):
        source_returns = pd.to_numeric(group["source_response_return"], errors="coerce")
        match_returns = pd.to_numeric(group["match_response_return"], errors="coerce")
        output_rows.append(
            {
                "baseline": str(baseline),
                "state": str(state),
                "horizon": int(horizon),
                "match_count": int(len(group)),
                "median_path_correlation": float(
                    pd.to_numeric(group["path_correlation"], errors="coerce").median()
                ),
                "median_cosine_similarity": float(
                    pd.to_numeric(group["cosine_similarity"], errors="coerce").median()
                ),
                "median_euclidean_distance": float(
                    pd.to_numeric(group["euclidean_distance"], errors="coerce").median()
                ),
                "median_abs_return_diff": float((source_returns - match_returns).abs().median()),
                "median_source_return": float(source_returns.median()),
                "median_match_return": float(match_returns.median()),
            }
        )
    return pd.DataFrame(output_rows, columns=columns)


def run_same_state_cross_symbol_similarity(
    events: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: BehavioralStateConfig,
    top_k: int = 5,
    require_same_stimulus: bool = False,
    match_time_bucket: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict manual-method diagnostic: same state, different symbol, response-shape match."""

    _validate_similarity_feature_columns(feature_columns)
    if events.empty:
        empty = _empty_same_state_match_rows()
        return empty, _summarize_shape_rows(empty), _summarize_shape_rows(empty)

    data = events[events["primary_state_label"].ne("unclassified")].copy().reset_index(drop=True)
    if data.empty:
        empty = _empty_same_state_match_rows()
        return empty, _summarize_shape_rows(empty), _summarize_shape_rows(empty)
    feature_frame = _feature_matrix(data, feature_columns)
    rng = np.random.default_rng(config.random_seed)
    match_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for source_index, source in data.iterrows():
        horizon = int(source["response_horizon"])
        path_columns = _path_columns_for_horizon(data, horizon)
        source_vector = _path_vector(source, path_columns)
        if source_vector.size == 0:
            continue
        same_horizon = data["response_horizon"].eq(horizon)
        different_symbol = data["symbol"].ne(source["symbol"])
        same_state = data["primary_state_label"].eq(source["primary_state_label"])
        candidate_mask = same_horizon & different_symbol & same_state
        if require_same_stimulus and "stimulus_label" in data:
            candidate_mask &= data["stimulus_label"].eq(source.get("stimulus_label"))
        if match_time_bucket and "time_of_day_bucket" in data:
            candidate_mask &= data["time_of_day_bucket"].eq(source.get("time_of_day_bucket"))
        candidates = data[candidate_mask].copy()
        if candidates.empty and match_time_bucket and "time_of_day_bucket" in data:
            candidate_mask = same_horizon & different_symbol & same_state
            candidates = data[candidate_mask].copy()
        if candidates.empty:
            continue
        distances = _feature_distance(feature_frame, source_index, candidates.index)
        candidates["_feature_distance"] = distances
        candidates["_same_session"] = candidates["session_date"].eq(source.get("session_date"))
        candidates = candidates.sort_values(["_same_session", "_feature_distance"]).head(top_k)
        for _, match in candidates.iterrows():
            match_vector = _path_vector(match, path_columns)
            match_rows.append(
                _match_row(
                    source=source,
                    match=match,
                    source_vector=source_vector,
                    match_vector=match_vector,
                    feature_distance=float(match["_feature_distance"]),
                    baseline="same_state_cross_symbol",
                )
            )

        baseline_specs = {
            "random_cross_symbol": same_horizon & different_symbol,
            "different_state_cross_symbol": same_horizon
            & different_symbol
            & data["primary_state_label"].ne(source["primary_state_label"]),
            "same_symbol_random": same_horizon & data["symbol"].eq(source["symbol"]) & (
                data.index != source_index
            ),
        }
        for baseline_name, mask in baseline_specs.items():
            if "time_of_day_bucket" in data and baseline_name != "same_symbol_random":
                bucket_mask = mask & data["time_of_day_bucket"].eq(source.get("time_of_day_bucket"))
                pool = data[bucket_mask]
                if pool.empty:
                    pool = data[mask]
            else:
                pool = data[mask]
            if pool.empty:
                continue
            selected = pool.iloc[int(rng.integers(0, len(pool)))]
            selected_vector = _path_vector(selected, path_columns)
            baseline_rows.append(
                _match_row(
                    source=source,
                    match=selected,
                    source_vector=source_vector,
                    match_vector=selected_vector,
                    feature_distance=float(
                        _feature_distance(feature_frame, source_index, pd.Index([selected.name]))
                        .iloc[0]
                    ),
                    baseline=baseline_name,
                )
            )

    matches = pd.DataFrame(match_rows, columns=_empty_same_state_match_rows().columns)
    baselines_raw = pd.DataFrame(baseline_rows, columns=_empty_same_state_match_rows().columns)
    summary = _summarize_shape_rows(matches)
    baseline_summary = pd.concat(
        [_summarize_shape_rows(matches), _summarize_shape_rows(baselines_raw)],
        ignore_index=True,
    )
    return matches, summary, baseline_summary


def run_fingerprint_cross_symbol_similarity(
    events: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: BehavioralStateConfig,
    top_k: int = 5,
    match_time_bucket: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match different-symbol events by observable current/prior state fingerprint."""

    _validate_similarity_feature_columns(feature_columns)
    if events.empty:
        empty = _empty_same_state_match_rows()
        return empty, _summarize_shape_rows(empty), _summarize_shape_rows(empty)

    data = events[events["primary_state_label"].ne("unclassified")].copy().reset_index(drop=True)
    if data.empty:
        empty = _empty_same_state_match_rows()
        return empty, _summarize_shape_rows(empty), _summarize_shape_rows(empty)
    feature_frame, _ = _state_fingerprint_matrix(data, feature_columns)
    rng = np.random.default_rng(config.random_seed)
    match_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for source_index, source in data.iterrows():
        horizon = int(source["response_horizon"])
        path_columns = _path_columns_for_horizon(data, horizon)
        source_vector = _path_vector(source, path_columns)
        if source_vector.size == 0:
            continue
        same_horizon = data["response_horizon"].eq(horizon)
        different_symbol = data["symbol"].ne(source["symbol"])
        candidate_mask = same_horizon & different_symbol
        if match_time_bucket and "time_of_day_bucket" in data:
            candidate_mask &= data["time_of_day_bucket"].eq(source.get("time_of_day_bucket"))
        candidates = data[candidate_mask].copy()
        if candidates.empty and match_time_bucket and "time_of_day_bucket" in data:
            candidates = data[same_horizon & different_symbol].copy()
        if candidates.empty:
            continue
        distances = _feature_distance(feature_frame, source_index, candidates.index)
        candidates["_feature_distance"] = distances
        candidates["_same_session"] = candidates["session_date"].eq(source.get("session_date"))
        candidates = candidates.sort_values(["_same_session", "_feature_distance"]).head(top_k)
        for _, match in candidates.iterrows():
            match_rows.append(
                _match_row(
                    source=source,
                    match=match,
                    source_vector=source_vector,
                    match_vector=_path_vector(match, path_columns),
                    feature_distance=float(match["_feature_distance"]),
                    baseline="fingerprint_cross_symbol",
                )
            )

        baseline_specs = {
            "random_cross_symbol": same_horizon & different_symbol,
            "different_state_cross_symbol": same_horizon
            & different_symbol
            & data["primary_state_label"].ne(source["primary_state_label"]),
            "same_symbol_random": same_horizon & data["symbol"].eq(source["symbol"]) & (
                data.index != source_index
            ),
        }
        for baseline_name, mask in baseline_specs.items():
            pool = data[mask]
            if "time_of_day_bucket" in data and baseline_name != "same_symbol_random":
                bucket_pool = data[mask & data["time_of_day_bucket"].eq(source.get("time_of_day_bucket"))]
                if not bucket_pool.empty:
                    pool = bucket_pool
            if pool.empty:
                continue
            selected = pool.iloc[int(rng.integers(0, len(pool)))]
            baseline_rows.append(
                _match_row(
                    source=source,
                    match=selected,
                    source_vector=source_vector,
                    match_vector=_path_vector(selected, path_columns),
                    feature_distance=float(
                        _feature_distance(feature_frame, source_index, pd.Index([selected.name]))
                        .iloc[0]
                    ),
                    baseline=baseline_name,
                )
            )

    matches = pd.DataFrame(match_rows, columns=_empty_same_state_match_rows().columns)
    baselines_raw = pd.DataFrame(baseline_rows, columns=_empty_same_state_match_rows().columns)
    summary = _summarize_shape_rows(matches)
    baseline_summary = pd.concat(
        [summary, _summarize_shape_rows(baselines_raw)],
        ignore_index=True,
    )
    return matches, summary, baseline_summary


def _empty_oos_response_shape_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "similarity_mode",
            "split_mode",
            "fold",
            "fit_scope",
            "baseline",
            "state",
            "horizon",
            "test_event_count",
            "match_count",
            "median_path_correlation",
            "median_cosine_similarity",
            "median_euclidean_distance",
            "median_abs_return_diff",
            "median_source_return",
            "median_match_return",
            "verdict",
        ]
    )


def _summarize_oos_shape_rows(
    rows: pd.DataFrame,
    *,
    similarity_mode: str,
    split_mode: str,
    fold: str,
) -> pd.DataFrame:
    columns = _empty_oos_response_shape_summary().columns
    if rows.empty:
        return pd.DataFrame(columns=columns)
    output_rows: list[dict[str, Any]] = []
    for (baseline, state, horizon), group in rows.groupby(
        ["baseline", "source_state", "source_horizon"],
        sort=True,
    ):
        source_returns = pd.to_numeric(group["source_response_return"], errors="coerce")
        match_returns = pd.to_numeric(group["match_response_return"], errors="coerce")
        source_event_count = int(
            group[["source_symbol", "source_timestamp", "source_horizon"]].drop_duplicates().shape[0]
        )
        output_rows.append(
            {
                "similarity_mode": similarity_mode,
                "split_mode": split_mode,
                "fold": fold,
                "fit_scope": "train_only",
                "baseline": str(baseline),
                "state": str(state),
                "horizon": int(horizon),
                "test_event_count": source_event_count,
                "match_count": int(len(group)),
                "median_path_correlation": float(
                    pd.to_numeric(group["path_correlation"], errors="coerce").median()
                ),
                "median_cosine_similarity": float(
                    pd.to_numeric(group["cosine_similarity"], errors="coerce").median()
                ),
                "median_euclidean_distance": float(
                    pd.to_numeric(group["euclidean_distance"], errors="coerce").median()
                ),
                "median_abs_return_diff": float((source_returns - match_returns).abs().median()),
                "median_source_return": float(source_returns.median()),
                "median_match_return": float(match_returns.median()),
                "verdict": "diagnostic",
            }
        )
    return pd.DataFrame(output_rows, columns=columns)


def run_oos_response_shape_similarity(
    events: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: BehavioralStateConfig,
    similarity_mode: str = "fingerprint",
    split_mode: str = "walk_forward",
) -> pd.DataFrame:
    """Compare held-out response paths using train-only label or fingerprint matching."""

    if similarity_mode not in {"label", "fingerprint"}:
        raise ValueError("similarity_mode must be 'label' or 'fingerprint'")
    _validate_similarity_feature_columns(feature_columns)
    if events.empty:
        return _empty_oos_response_shape_summary()

    data = events[events["primary_state_label"].ne("unclassified")].copy()
    if data.empty:
        return _empty_oos_response_shape_summary()
    data = data.sort_values("timestamp").reset_index(drop=True)
    if split_mode != "walk_forward":
        raise ValueError("split_mode must be 'walk_forward'")
    train_mask = _walk_forward_mask(data)
    train = data[train_mask].copy().reset_index(drop=True)
    test = data[~train_mask].copy().reset_index(drop=True)
    if train.empty or test.empty:
        return _empty_oos_response_shape_summary()

    train_features, fit = _state_fingerprint_matrix(train, feature_columns)
    test_features, _ = _state_fingerprint_matrix(test, feature_columns, fit=fit)
    rng = np.random.default_rng(config.random_seed)
    rows: list[dict[str, Any]] = []

    for source_index, source in test.iterrows():
        horizon = int(source["response_horizon"])
        path_columns = _path_columns_for_horizon(test, horizon)
        if not path_columns:
            path_columns = _path_columns_for_horizon(train, horizon)
        source_vector = _path_vector(source, path_columns)
        if source_vector.size == 0:
            continue
        same_horizon = train["response_horizon"].eq(horizon)
        different_symbol = train["symbol"].ne(source["symbol"])
        candidate_mask = same_horizon & different_symbol
        if similarity_mode == "label":
            candidate_mask &= train["primary_state_label"].eq(source["primary_state_label"])
        candidates = train[candidate_mask].copy()
        if not candidates.empty:
            source_feature = test_features.loc[source_index].to_numpy(dtype=float)
            candidate_matrix = train_features.loc[candidates.index].to_numpy(dtype=float)
            distances = np.linalg.norm(candidate_matrix - source_feature, axis=1)
            candidates["_feature_distance"] = distances
            candidates = candidates.sort_values("_feature_distance").head(
                max(1, config.nearest_neighbors)
            )
            baseline_name = (
                "label_cross_symbol"
                if similarity_mode == "label"
                else "fingerprint_cross_symbol"
            )
            for _, match in candidates.iterrows():
                rows.append(
                    _match_row(
                        source=source,
                        match=match,
                        source_vector=source_vector,
                        match_vector=_path_vector(match, path_columns),
                        feature_distance=float(match["_feature_distance"]),
                        baseline=baseline_name,
                    )
                )

        baseline_specs = {
            "random_cross_symbol": same_horizon & different_symbol,
            "different_state_cross_symbol": same_horizon
            & different_symbol
            & train["primary_state_label"].ne(source["primary_state_label"]),
            "same_symbol_random": same_horizon & train["symbol"].eq(source["symbol"]),
        }
        for baseline_name, mask in baseline_specs.items():
            pool = train[mask]
            if "time_of_day_bucket" in train and baseline_name != "same_symbol_random":
                bucket_pool = train[
                    mask & train["time_of_day_bucket"].eq(source.get("time_of_day_bucket"))
                ]
                if not bucket_pool.empty:
                    pool = bucket_pool
            if pool.empty:
                continue
            selected = pool.iloc[int(rng.integers(0, len(pool)))]
            rows.append(
                _match_row(
                    source=source,
                    match=selected,
                    source_vector=source_vector,
                    match_vector=_path_vector(selected, path_columns),
                    feature_distance=math.nan,
                    baseline=baseline_name,
                )
            )

    raw = pd.DataFrame(rows, columns=_empty_same_state_match_rows().columns)
    return _summarize_oos_shape_rows(
        raw,
        similarity_mode=similarity_mode,
        split_mode=split_mode,
        fold="time_split_60_40",
    )


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


def build_manual_audit_examples() -> pd.DataFrame:
    """Reference rows from the manual audit that the v2 lab should make inspectable."""

    rows = [
        ("HOOD", "2026-04-15", "initiative buying / controlled pullback", "initiative_buying_controlled_pullback"),
        ("HOOD", "2025-07-02", "strong squeeze / initiative buying after early pullback", "momentum_memory"),
        ("GLW", "2026-06-24", "liquidation/whipsaw into squeeze/initiative continuation", "liquidation_failed_low_squeeze"),
        ("GLW", "2026-06-24", "huge extension near high into exhaustion/recoil", "exhaustion_fade"),
        ("HOOD", "2025-07-01", "bullish shock failure / recoil", "bullish_shock_failure"),
        ("CRM", "2026-06-24", "early pop, rejection, fade", "failed_open_long_block"),
        ("HOOD", "2026-06-24", "opening liquidation / failed-open selling", "failed_open_down_continuation"),
        ("GLW", "2026-05-12", "active liquidation then failed early bounce", "failed_bounce_active_liquidation"),
        ("GLW", "2026-02-17", "flush/reclaim/controlled recovery", "liquidation_failed_low_squeeze"),
        ("FCX", "2026-02-17", "flush/reclaim but muted", "slow_snapback"),
        ("CRM", "2025-10-10", "failed-open / trend-down", "trend_day_down"),
        ("CRM", "2026-02-17", "clean opening-drive/downtrend state", "opening_drive_down"),
        ("CRM", "2026-05-12", "failed-open / drift-down", "failed_open_down_continuation"),
        ("FCX", "2025-10-10", "clean selloff/liquidation", "active_liquidation"),
        ("FCX", "2026-05-12", "slow snapback / controlled recovery", "slow_snapback"),
        ("GLW", "2025-07-01", "dead chop / low-opportunity", "dead_chop"),
        ("GLW", "2025-08-15", "dead chop / no-trade", "dead_chop"),
        ("FCX", "2026-06-24", "range/chop with mild downside", "dead_chop"),
    ]
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "session_date": session_date,
                "manual_state_note": note,
                "expected_lab_state": expected_state,
                "report_found_in_checkout": False,
            }
            for symbol, session_date, note, expected_state in rows
        ]
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _state_counts(events: pd.DataFrame) -> dict[str, int]:
    if events.empty:
        return {}
    counts = Counter(map(str, events["primary_state_label"].tolist()))
    return dict(sorted(counts.items()))


def build_state_overlap_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Count overlapping boolean state labels."""

    state_columns = [f"state_{label}" for label in STATE_LABELS if f"state_{label}" in rows]
    if rows.empty or not state_columns:
        return pd.DataFrame(columns=["state_a", "state_b", "overlap_count"])
    bools = rows[state_columns].astype(bool)
    output_rows: list[dict[str, Any]] = []
    for left in state_columns:
        for right in state_columns:
            output_rows.append(
                {
                    "state_a": left.removeprefix("state_"),
                    "state_b": right.removeprefix("state_"),
                    "overlap_count": int((bools[left] & bools[right]).sum()),
                }
            )
    return pd.DataFrame(output_rows)


def build_state_priority_conflicts(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows where multiple state booleans were true before priority assignment."""

    columns = [
        "symbol",
        "timestamp",
        "session_date",
        "bar_index_in_session",
        "primary_state_label",
        "state_overlap_count",
        "state_overlap_labels",
    ]
    if rows.empty or "state_priority_conflict" not in rows:
        return pd.DataFrame(columns=columns)
    conflicts = rows[rows["state_priority_conflict"].astype(bool)].copy()
    return conflicts.reindex(columns=columns)


def build_primary_state_distribution(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["primary_state_label", "row_count", "row_share"])
    counts = rows["primary_state_label"].value_counts(dropna=False).rename_axis(
        "primary_state_label"
    )
    output = counts.reset_index(name="row_count")
    output["row_share"] = output["row_count"] / max(1, int(len(rows)))
    return output


def build_per_symbol_state_counts(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "state", "horizon", "event_count"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        events.groupby(["symbol", "primary_state_label", "response_horizon"], dropna=False)
        .size()
        .reset_index(name="event_count")
    )
    return grouped.rename(
        columns={"primary_state_label": "state", "response_horizon": "horizon"}
    ).reindex(columns=columns)


def build_per_symbol_state_response(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "state",
        "horizon",
        "event_count",
        "median_forward_return",
        "win_rate",
        "median_mfe",
        "median_mae",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (symbol, state, horizon), group in events.groupby(
        ["symbol", "primary_state_label", "response_horizon"],
        sort=True,
    ):
        returns = pd.to_numeric(group["response_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        rows.append(
            {
                "symbol": str(symbol),
                "state": str(state),
                "horizon": int(horizon),
                "event_count": int(len(returns)),
                "median_forward_return": float(returns.median()),
                "win_rate": float((returns > 0.0).mean()),
                "median_mfe": float(pd.to_numeric(group["response_mfe"], errors="coerce").median()),
                "median_mae": float(pd.to_numeric(group["response_mae"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_state_transition_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["state_transition_from", "state_transition_to", "transition_count"]
    if rows.empty or "state_transition_from" not in rows:
        return pd.DataFrame(columns=columns)
    return (
        rows.groupby(["state_transition_from", "state_transition_to"], dropna=False)
        .size()
        .reset_index(name="transition_count")
        .reindex(columns=columns)
    )


def build_state_duration_summary(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "state",
        "segment_count",
        "median_duration_bars",
        "p75_duration_bars",
        "max_duration_bars",
    ]
    if rows.empty or "state_duration_bars" not in rows:
        return pd.DataFrame(columns=columns)
    segments = rows[
        rows["state_entry"].astype(bool) & rows["primary_state_label"].ne("unclassified")
    ].copy()
    if segments.empty:
        return pd.DataFrame(columns=columns)
    duration = pd.to_numeric(segments["state_duration_bars"], errors="coerce")
    segments["duration"] = duration
    output_rows: list[dict[str, Any]] = []
    for state, group in segments.groupby("primary_state_label", sort=True):
        values = pd.to_numeric(group["duration"], errors="coerce").dropna()
        if values.empty:
            continue
        output_rows.append(
            {
                "state": str(state),
                "segment_count": int(len(values)),
                "median_duration_bars": float(values.median()),
                "p75_duration_bars": float(values.quantile(0.75)),
                "max_duration_bars": float(values.max()),
            }
        )
    return pd.DataFrame(output_rows, columns=columns)


def build_time_of_day_state_summary(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["time_of_day_bucket", "state", "horizon", "event_count", "median_forward_return"]
    if events.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (bucket, state, horizon), group in events.groupby(
        ["time_of_day_bucket", "primary_state_label", "response_horizon"],
        sort=True,
        dropna=False,
    ):
        returns = pd.to_numeric(group["response_return"], errors="coerce").dropna()
        rows.append(
            {
                "time_of_day_bucket": str(bucket),
                "state": str(state),
                "horizon": int(horizon),
                "event_count": int(len(returns)),
                "median_forward_return": float(returns.median()) if not returns.empty else math.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_concentration_reports(
    events: pd.DataFrame,
    *,
    config: BehavioralStateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbol_columns = ["state", "horizon", "top_symbol", "top_symbol_share", "event_count"]
    session_columns = ["state", "horizon", "top_session", "top_session_share", "event_count"]
    warning_columns = ["scope", "state", "horizon", "dominant_value", "share", "threshold", "warning"]
    if events.empty:
        return (
            pd.DataFrame(columns=symbol_columns),
            pd.DataFrame(columns=session_columns),
            pd.DataFrame(columns=warning_columns),
        )
    symbol_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for (state, horizon), group in events.groupby(
        ["primary_state_label", "response_horizon"],
        sort=True,
    ):
        event_count = int(len(group))
        symbol_share = group["symbol"].value_counts(normalize=True)
        session_share = group["session_date"].value_counts(normalize=True)
        top_symbol = str(symbol_share.index[0]) if not symbol_share.empty else ""
        top_symbol_share = float(symbol_share.iloc[0]) if not symbol_share.empty else math.nan
        top_session = str(session_share.index[0]) if not session_share.empty else ""
        top_session_share = float(session_share.iloc[0]) if not session_share.empty else math.nan
        symbol_rows.append(
            {
                "state": str(state),
                "horizon": int(horizon),
                "top_symbol": top_symbol,
                "top_symbol_share": top_symbol_share,
                "event_count": event_count,
            }
        )
        session_rows.append(
            {
                "state": str(state),
                "horizon": int(horizon),
                "top_session": top_session,
                "top_session_share": top_session_share,
                "event_count": event_count,
            }
        )
        if top_symbol_share > config.max_single_symbol_share:
            warnings.append(
                {
                    "scope": "symbol",
                    "state": str(state),
                    "horizon": int(horizon),
                    "dominant_value": top_symbol,
                    "share": top_symbol_share,
                    "threshold": config.max_single_symbol_share,
                    "warning": "single_symbol_concentration",
                }
            )
        if top_session_share > config.max_single_session_share:
            warnings.append(
                {
                    "scope": "session",
                    "state": str(state),
                    "horizon": int(horizon),
                    "dominant_value": top_session,
                    "share": top_session_share,
                    "threshold": config.max_single_session_share,
                    "warning": "single_session_concentration",
                }
            )
        if "timestamp" in group:
            months = pd.to_datetime(group["timestamp"], utc=True, errors="coerce").dt.strftime(
                "%Y-%m"
            )
            month_share = months.astype(str).value_counts(normalize=True)
            if not month_share.empty and float(month_share.iloc[0]) > config.max_single_month_share:
                warnings.append(
                    {
                        "scope": "month",
                        "state": str(state),
                        "horizon": int(horizon),
                        "dominant_value": str(month_share.index[0]),
                        "share": float(month_share.iloc[0]),
                        "threshold": config.max_single_month_share,
                        "warning": "single_month_concentration",
                    }
                )
        if "time_of_day_bucket" in group:
            time_share = group["time_of_day_bucket"].value_counts(normalize=True)
            if not time_share.empty and float(time_share.iloc[0]) > 0.80:
                warnings.append(
                    {
                        "scope": "time_of_day",
                        "state": str(state),
                        "horizon": int(horizon),
                        "dominant_value": str(time_share.index[0]),
                        "share": float(time_share.iloc[0]),
                        "threshold": 0.80,
                        "warning": "time_of_day_concentration",
                    }
                )
    return (
        pd.DataFrame(symbol_rows, columns=symbol_columns),
        pd.DataFrame(session_rows, columns=session_columns),
        pd.DataFrame(warnings, columns=warning_columns),
    )


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


def build_stimulus_response_matrix(
    events: pd.DataFrame,
    *,
    permutation_baseline: pd.DataFrame,
    config: BehavioralStateConfig,
) -> pd.DataFrame:
    """Compare state-conditioned stimulus responses with generic stimulus baselines."""

    columns = [
        "stimulus_label",
        "state_label",
        "horizon",
        "event_count",
        "median_forward_return",
        "win_rate",
        "median_mfe",
        "median_mae",
        "generic_stimulus_median_return",
        "state_excess_vs_generic",
        "random_label_excess",
        "permutation_percentile",
        "verdict",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)
    generic = (
        events.groupby(["stimulus_label", "response_horizon"], dropna=False)["response_return"]
        .median()
        .rename("generic_stimulus_median_return")
    )
    permutation_lookup: dict[tuple[str, int], dict[str, float]] = {}
    if not permutation_baseline.empty:
        for _, row in permutation_baseline.iterrows():
            key = (str(row["state"]), int(row["horizon"]))
            current = permutation_lookup.get(key)
            p_value = float(row["permutation_p_value"])
            if current is None or p_value < current["permutation_p_value"]:
                permutation_lookup[key] = {
                    "permutation_p_value": p_value,
                    "permutation_percentile": float(row["permutation_percentile"]),
                    "permutation_median_return_mean": float(
                        row["permutation_median_return_mean"]
                    ),
                }
    rows: list[dict[str, Any]] = []
    for (stimulus, state, horizon), group in events.groupby(
        ["stimulus_label", "primary_state_label", "response_horizon"],
        sort=True,
        dropna=False,
    ):
        returns = pd.to_numeric(group["response_return"], errors="coerce").dropna()
        if returns.empty:
            continue
        key = (str(stimulus), int(horizon))
        generic_median = float(generic.loc[key]) if key in generic.index else math.nan
        median_return = float(returns.median())
        state_excess = median_return - generic_median if not math.isnan(generic_median) else math.nan
        permutation = permutation_lookup.get((str(state), int(horizon)), {})
        random_mean = permutation.get("permutation_median_return_mean", math.nan)
        random_label_excess = median_return - random_mean if not math.isnan(random_mean) else math.nan
        p_value = permutation.get("permutation_p_value", math.nan)
        verdict = "continue_research" if (
            len(returns) >= config.min_independent_events_per_state_horizon
            and not math.isnan(state_excess)
            and state_excess * 10_000 >= config.min_oos_median_return_excess_vs_generic_bps
            and not math.isnan(p_value)
            and p_value <= config.permutation_p_value_max
        ) else "mixed_response"
        rows.append(
            {
                "stimulus_label": str(stimulus),
                "state_label": str(state),
                "horizon": int(horizon),
                "event_count": int(len(returns)),
                "median_forward_return": median_return,
                "win_rate": float((returns > 0.0).mean()),
                "median_mfe": float(pd.to_numeric(group["response_mfe"], errors="coerce").median()),
                "median_mae": float(pd.to_numeric(group["response_mae"], errors="coerce").median()),
                "generic_stimulus_median_return": generic_median,
                "state_excess_vs_generic": state_excess,
                "random_label_excess": random_label_excess,
                "permutation_percentile": permutation.get("permutation_percentile", math.nan),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def apply_state_gate_to_positions(positions: pd.Series, allowed_entries: pd.Series) -> pd.Series:
    """Suppress new entries when state gate is false while preserving existing exits."""

    raw = positions.astype(float).reset_index(drop=True).fillna(0.0)
    allowed = allowed_entries.astype(bool).reset_index(drop=True).reindex(raw.index).fillna(False)
    gated = pd.Series(0.0, index=raw.index)
    previous_raw = 0.0
    current_position = 0.0
    for index, target in enumerate(raw):
        target = float(target)
        is_new_entry = previous_raw <= 0.0 and target > 0.0
        is_exit = previous_raw > 0.0 and target <= 0.0
        if is_exit:
            current_position = 0.0
        elif is_new_entry:
            current_position = target if bool(allowed.iloc[index]) else 0.0
        elif target > 0.0 and current_position <= 0.0 and bool(allowed.iloc[index]):
            current_position = target
        elif target > 0.0 and current_position > 0.0:
            current_position = target
        elif target <= 0.0:
            current_position = 0.0
        gated.iloc[index] = current_position
        previous_raw = target
    return gated


def run_template_overlay_diagnostics(
    frames: dict[str, pd.DataFrame],
    *,
    config: BehavioralStateConfig,
) -> pd.DataFrame:
    """Optionally compare generic and state-gated template positions."""

    columns = [
        "template",
        "variant",
        "symbol",
        "net_return",
        "gross_return",
        "number_of_trades",
        "exposure",
        "state_gate_trade_count",
        "net_return_excess_vs_generic_bps",
        "net_return_excess_vs_random_gate_bps",
        "verdict",
    ]
    if not config.run_template_overlay:
        return pd.DataFrame(
            [
                {
                    "template": config.template,
                    "variant": "not_run",
                    "symbol": "ALL",
                    "net_return": math.nan,
                    "gross_return": math.nan,
                    "number_of_trades": 0,
                    "exposure": math.nan,
                    "state_gate_trade_count": 0,
                    "net_return_excess_vs_generic_bps": math.nan,
                    "net_return_excess_vs_random_gate_bps": math.nan,
                    "verdict": "not_run",
                }
            ],
            columns=columns,
        )
    try:
        from stocker_backtest.costs import CostModel
        from stocker_backtest.vectorized import evaluate_positions
        from stocker_research.templates import get_template
    except Exception:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(config.random_seed)
    template = get_template(config.template)
    params = {
        "timeframe": config.timeframe,
        "market_calendar": config.market_calendar,
        "relative_volume_lookback_sessions": config.relative_volume_lookback_sessions,
        "parameter_set_id": "behavioral_state_similarity_overlay",
    }
    cost_model = CostModel()
    rows: list[dict[str, Any]] = []
    for symbol, frame in frames.items():
        reset = frame.reset_index(drop=True)
        generic_positions = template.generate_positions(reset, params).reset_index(drop=True)
        allowed = reset["primary_state_label"].ne("unclassified")
        state_positions = apply_state_gate_to_positions(generic_positions, allowed)
        same_count_allowed = pd.Series(False, index=reset.index)
        allowed_count = int(allowed.sum())
        if allowed_count > 0:
            chosen = rng.choice(reset.index.to_numpy(), size=allowed_count, replace=False)
            same_count_allowed.loc[chosen] = True
        random_positions = apply_state_gate_to_positions(generic_positions, same_count_allowed)
        results = {
            "generic": evaluate_positions(reset, generic_positions, cost_model=cost_model),
            "state_gated": evaluate_positions(reset, state_positions, cost_model=cost_model),
            "same_trade_count_random_gate": evaluate_positions(
                reset,
                random_positions,
                cost_model=cost_model,
            ),
        }
        generic_net = results["generic"].net_return
        random_net = results["same_trade_count_random_gate"].net_return
        for variant, result in results.items():
            excess = (result.net_return - generic_net) * 10_000
            random_excess = (result.net_return - random_net) * 10_000
            verdict = "continue_research" if (
                variant == "state_gated"
                and excess >= config.min_template_net_return_excess_vs_generic_bps
                and random_excess >= config.min_template_net_return_excess_vs_generic_bps
                and result.number_of_trades > 0
            ) else "reject_no_template_lift" if variant == "state_gated" else "diagnostic"
            rows.append(
                {
                    "template": config.template,
                    "variant": variant,
                    "symbol": symbol,
                    "net_return": float(result.net_return),
                    "gross_return": float(result.gross_return),
                    "number_of_trades": int(result.number_of_trades),
                    "exposure": float(result.exposure),
                    "state_gate_trade_count": int(results["state_gated"].number_of_trades),
                    "net_return_excess_vs_generic_bps": float(excess),
                    "net_return_excess_vs_random_gate_bps": float(random_excess),
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _shape_similarity_supported(
    shape_baselines: pd.DataFrame,
    *,
    main_baseline: str,
    config: BehavioralStateConfig,
) -> bool:
    if shape_baselines.empty:
        return False
    for (_state, _horizon), group in shape_baselines.groupby(["state", "horizon"], sort=True):
        main = group[group["baseline"].eq(main_baseline)]
        random = group[group["baseline"].eq("random_cross_symbol")]
        different = group[group["baseline"].eq("different_state_cross_symbol")]
        if main.empty or random.empty or different.empty:
            continue
        main_row = main.iloc[0]
        random_row = random.iloc[0]
        different_row = different.iloc[0]
        main_similarity = float(main_row.get("median_cosine_similarity", math.nan))
        random_similarity = float(random_row.get("median_cosine_similarity", math.nan))
        different_similarity = float(different_row.get("median_cosine_similarity", math.nan))
        main_distance = float(main_row.get("median_euclidean_distance", math.nan))
        random_distance = float(random_row.get("median_euclidean_distance", math.nan))
        different_distance = float(different_row.get("median_euclidean_distance", math.nan))
        match_count = int(main_row.get("match_count", 0))
        if (
            match_count >= config.min_independent_events_per_state_horizon
            and not math.isnan(main_similarity)
            and not math.isnan(random_similarity)
            and not math.isnan(different_similarity)
            and main_similarity - random_similarity >= config.min_similarity_excess_vs_random
            and main_similarity - different_similarity >= config.min_similarity_excess_vs_random
            and not math.isnan(main_distance)
            and not math.isnan(random_distance)
            and not math.isnan(different_distance)
            and main_distance < random_distance
            and main_distance < different_distance
        ):
            return True
    return False


def build_decision_summary(
    *,
    oos_state_response: pd.DataFrame,
    permutation_baseline: pd.DataFrame,
    concentration_warnings: pd.DataFrame,
    template_overlay_summary: pd.DataFrame,
    pipeline_passed: bool,
    config: BehavioralStateConfig,
    response_shape_baselines: pd.DataFrame | None = None,
    response_shape_similarity_summary: pd.DataFrame | None = None,
    fingerprint_shape_baselines: pd.DataFrame | None = None,
    fingerprint_similarity_summary: pd.DataFrame | None = None,
    oos_response_shape_similarity: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Apply conservative evidence gates; default to rejection."""

    reasons: list[str] = []
    if not pipeline_passed:
        reasons.append("pipeline did not complete for all requested symbols")

    shape_baselines = response_shape_baselines if response_shape_baselines is not None else pd.DataFrame()
    fingerprint_baselines = (
        fingerprint_shape_baselines if fingerprint_shape_baselines is not None else pd.DataFrame()
    )
    shape_summary = (
        response_shape_similarity_summary
        if response_shape_similarity_summary is not None
        else pd.DataFrame()
    )
    fingerprint_summary = (
        fingerprint_similarity_summary
        if fingerprint_similarity_summary is not None
        else pd.DataFrame()
    )
    oos_shape = (
        oos_response_shape_similarity
        if oos_response_shape_similarity is not None
        else pd.DataFrame()
    )
    label_shape_diagnostic_supported = False
    if shape_baselines.empty or shape_summary.empty:
        reasons.append("same-state cross-symbol response paths were not available")
    else:
        same_state = shape_baselines[shape_baselines["baseline"].eq("same_state_cross_symbol")]
        label_shape_diagnostic_supported = _shape_similarity_supported(
            shape_baselines,
            main_baseline="same_state_cross_symbol",
            config=config,
        )
        if not label_shape_diagnostic_supported:
            reasons.append(
                "same-state cross-symbol response paths did not beat random and different-state baselines"
            )
        if same_state.empty:
            reasons.append("no strict same-state cross-symbol matches were found")
    fingerprint_shape_diagnostic_supported = False
    if fingerprint_baselines.empty or fingerprint_summary.empty:
        reasons.append("fingerprint cross-symbol response paths were not available")
    else:
        fingerprint_shape_diagnostic_supported = _shape_similarity_supported(
            fingerprint_baselines,
            main_baseline="fingerprint_cross_symbol",
            config=config,
        )
        if not fingerprint_shape_diagnostic_supported:
            reasons.append(
                "fingerprint cross-symbol response paths did not beat random and different-state baselines"
            )

    if oos_state_response.empty:
        reasons.append("no out-of-sample state response rows were available")
    positive_oos = (
        oos_state_response[oos_state_response["gate_passed"].astype(bool)]
        if not oos_state_response.empty and "gate_passed" in oos_state_response
        else pd.DataFrame()
    )
    positive_oos_groups: set[tuple[str, int]] = set()
    if not oos_state_response.empty and "gate_passed" in oos_state_response:
        for (state, horizon), group in oos_state_response.groupby(["state", "horizon"], sort=True):
            pass_share = float(group["gate_passed"].astype(bool).mean())
            if pass_share >= config.required_positive_folds_share:
                positive_oos_groups.add((str(state), int(horizon)))
    permutation_ok = False
    if positive_oos_groups and not permutation_baseline.empty:
        for state, horizon in positive_oos_groups:
            matches = permutation_baseline[
                (permutation_baseline["state"] == state)
                & (permutation_baseline["horizon"] == horizon)
                & (pd.to_numeric(permutation_baseline["permutation_p_value"], errors="coerce")
                <= config.permutation_p_value_max)
            ]
            if not matches.empty:
                permutation_ok = True
                break
    label_oos_supported = _shape_similarity_supported(
        oos_shape[oos_shape["similarity_mode"].eq("label")] if not oos_shape.empty else pd.DataFrame(),
        main_baseline="label_cross_symbol",
        config=config,
    )
    fingerprint_oos_supported = _shape_similarity_supported(
        oos_shape[oos_shape["similarity_mode"].eq("fingerprint")]
        if not oos_shape.empty
        else pd.DataFrame(),
        main_baseline="fingerprint_cross_symbol",
        config=config,
    )
    if not label_oos_supported:
        reasons.append("label response-shape similarity did not pass out of sample")
    if not fingerprint_oos_supported:
        reasons.append("fingerprint response-shape similarity did not pass out of sample")
    label_similarity_supported = bool(
        label_shape_diagnostic_supported
        and label_oos_supported
        and positive_oos_groups
        and permutation_ok
    )
    fingerprint_similarity_supported = bool(
        fingerprint_shape_diagnostic_supported
        and fingerprint_oos_supported
        and positive_oos_groups
        and permutation_ok
    )
    oos_similarity_supported = bool(label_similarity_supported or fingerprint_similarity_supported)
    if positive_oos.empty or not positive_oos_groups:
        reasons.append("state response did not beat generic baseline out of sample")
    if not permutation_ok:
        reasons.append("state response did not beat shuffled-label baselines")
    concentrated = not concentration_warnings.empty
    if concentrated:
        reasons.append("one or more state/horizon groups are concentration dominated")

    template_overlay_supported = False
    if config.run_template_overlay:
        state_gated = template_overlay_summary[
            template_overlay_summary.get("variant", pd.Series(dtype=str)).eq("state_gated")
        ]
        if not state_gated.empty:
            generic_excess = pd.to_numeric(
                state_gated["net_return_excess_vs_generic_bps"],
                errors="coerce",
            )
            random_excess = pd.to_numeric(
                state_gated.get(
                    "net_return_excess_vs_random_gate_bps",
                    pd.Series(np.nan, index=state_gated.index),
                ),
                errors="coerce",
            )
            template_overlay_supported = bool(
                float(generic_excess.median()) >= config.min_template_net_return_excess_vs_generic_bps
                and float(random_excess.median())
                >= config.min_template_net_return_excess_vs_generic_bps
            )
        if not template_overlay_supported:
            reasons.append("state-gated template did not beat generic and random-gated templates")
    else:
        reasons.append("template overlay was not run")

    insufficient_events = (
        oos_state_response.empty
        or (
            "test_event_count" in oos_state_response
            and pd.to_numeric(oos_state_response["test_event_count"], errors="coerce").max()
            < config.min_independent_events_per_state_horizon
        )
    )
    if not pipeline_passed:
        decision = "reject_insufficient_independent_events"
    elif concentrated:
        decision = "reject_concentrated"
    elif insufficient_events:
        decision = "reject_insufficient_independent_events"
    elif not oos_similarity_supported:
        decision = "reject_no_portability"
    elif not config.run_template_overlay or not template_overlay_supported:
        decision = "reject_no_template_lift"
    else:
        decision = "continue_research_only"

    return {
        "pipeline_passed": bool(pipeline_passed),
        "label_similarity_supported": bool(label_similarity_supported),
        "fingerprint_similarity_supported": bool(fingerprint_similarity_supported),
        "state_similarity_supported": bool(label_similarity_supported),
        "label_similarity_diagnostic_supported": bool(label_shape_diagnostic_supported),
        "fingerprint_similarity_diagnostic_supported": bool(fingerprint_shape_diagnostic_supported),
        "oos_similarity_supported": bool(oos_similarity_supported),
        "template_overlay_supported": bool(template_overlay_supported),
        "decision": decision,
        "decision_reasons": sorted(set(reasons)),
        "evidence_gates": {
            "min_independent_events_per_state_horizon": (
                config.min_independent_events_per_state_horizon
            ),
            "min_train_symbols_per_state": config.min_train_symbols_per_state,
            "min_test_symbols_per_state": config.min_test_symbols_per_state,
            "max_single_symbol_share": config.max_single_symbol_share,
            "max_single_session_share": config.max_single_session_share,
            "max_single_month_share": config.max_single_month_share,
            "min_similarity_excess_vs_random": config.min_similarity_excess_vs_random,
            "min_oos_directional_accuracy_excess_vs_generic": (
                config.min_oos_directional_accuracy_excess_vs_generic
            ),
            "min_oos_median_return_excess_vs_generic_bps": (
                config.min_oos_median_return_excess_vs_generic_bps
            ),
            "min_template_net_return_excess_vs_generic_bps": (
                config.min_template_net_return_excess_vs_generic_bps
            ),
            "permutation_p_value_max": config.permutation_p_value_max,
            "required_positive_folds_share": config.required_positive_folds_share,
        },
    }


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
    decision_reasons = "\n".join(
        f"- {reason}" for reason in summary["decision_reasons"]
    ) or "- None"
    warnings = "\n".join(f"- {warning}" for warning in summary["warnings"]) or "- None"
    top_rejected = summary.get("top_rejected_states", [])
    continuing = summary.get("states_worth_continuing_research", [])
    best_same_state = summary.get("best_same_state_cross_stock_states", [])
    best_fingerprint = summary.get("best_fingerprint_cross_stock_states", [])
    subtype_split = summary.get("states_requiring_subtype_split", [])
    oos_rows = summary.get("oos_state_response_summary", [])[:12]
    loso_rows = summary.get("leave_one_symbol_out_summary", [])[:12]
    shape_rows = summary.get("response_shape_baselines", [])[:18]
    fingerprint_rows = summary.get("fingerprint_response_shape_baselines", [])[:18]
    oos_shape_rows = summary.get("oos_response_shape_similarity", [])[:18]
    stimulus_rows = summary.get("stimulus_response_matrix", [])[:12]
    template_rows = summary.get("template_overlay_summary", [])[:12]
    dead_chop_rows = summary.get("dead_chop_blocking_quality", [])[:12]
    audit_rows = summary.get("manual_audit_examples", [])[:12]
    return f"""# Behavioral State Similarity Lab

## Decision

Decision: {summary["decision"]}
pipeline_passed: {summary["pipeline_passed"]}
label_similarity_supported: {summary["label_similarity_supported"]}
fingerprint_similarity_supported: {summary["fingerprint_similarity_supported"]}
state_similarity_supported: {summary["state_similarity_supported"]}
oos_similarity_supported: {summary["oos_similarity_supported"]}
template_overlay_supported: {summary["template_overlay_supported"]}
decision_reasons:
{decision_reasons}

Raw classified horizon rows: {summary["total_horizon_events"]}
Independent evidence events: {summary["total_independent_events"]}

This is a research-only diagnostic report. States were detected using only
current and prior bars available at each bar close. Forward return, MFE, MAE,
and absolute-return columns are response targets only. This run did not fetch
live data, place orders, loosen gates, alter live templates, or promote candidates.

- Run id: `{summary["run_id"]}`
- Symbols completed: {len(summary["symbols_completed"])}
- Symbols failed: {len(summary["symbols_failed"])}
- Event mode: `{summary["config"]["event_mode"]}`
- Timeframe: `{summary["config"]["timeframe"]}`

## Same-State Cross-Symbol Similarity

{_markdown_table(shape_rows, ["baseline", "state", "horizon", "match_count", "median_cosine_similarity", "median_euclidean_distance", "median_abs_return_diff"])}

## Fingerprint Cross-Symbol Similarity

{_markdown_table(fingerprint_rows, ["baseline", "state", "horizon", "match_count", "median_cosine_similarity", "median_euclidean_distance", "median_abs_return_diff"])}

## OOS Response-Shape Similarity

{_markdown_table(oos_shape_rows, ["similarity_mode", "baseline", "state", "horizon", "test_event_count", "median_cosine_similarity", "median_euclidean_distance", "median_abs_return_diff", "fit_scope"])}

## Out-of-Sample State Response

{_markdown_table(oos_rows, ["split_mode", "fold", "state", "horizon", "test_event_count", "directional_accuracy_excess_vs_generic", "oos_median_return_excess_vs_generic_bps", "verdict"])}

## Leave-One-Symbol-Out Summary

{_markdown_table(loso_rows, ["split_mode", "fold", "state", "horizon", "test_event_count", "directional_accuracy_excess_vs_generic", "oos_median_return_excess_vs_generic_bps", "verdict"])}

## Stimulus Response Matrix

{_markdown_table(stimulus_rows, ["stimulus_label", "state_label", "horizon", "event_count", "state_excess_vs_generic", "random_label_excess", "permutation_percentile", "verdict"])}

## Best Same-State Cross-Stock States

{_markdown_table(best_same_state, ["state", "horizon", "match_count", "median_cosine_similarity", "random_cosine_similarity", "different_state_cosine_similarity"])}

## Best Fingerprint Cross-Stock States

{_markdown_table(best_fingerprint, ["state", "horizon", "match_count", "median_cosine_similarity", "random_cosine_similarity", "different_state_cosine_similarity"])}

## Top Rejected States

{_markdown_table(top_rejected, ["state", "horizon", "verdict", "reason"])}

## States Worth Continuing Research

{_markdown_table(continuing, ["state", "horizon", "verdict", "reason"])}

## States Needing Subtype Split

{_markdown_table(subtype_split, ["state", "horizon", "match_count", "median_cosine_similarity", "random_cosine_similarity", "different_state_cosine_similarity", "reason"])}

## Dead Chop Blocking Quality

{_markdown_table(dead_chop_rows, ["stimulus_label", "state_label", "horizon", "event_count", "median_forward_return", "win_rate", "state_excess_vs_generic", "verdict"])}

## Template Overlay

{_markdown_table(template_rows, ["template", "variant", "symbol", "net_return", "number_of_trades", "net_return_excess_vs_generic_bps", "net_return_excess_vs_random_gate_bps", "verdict"])}

## Manual Audit Examples

{_markdown_table(audit_rows, ["symbol", "session_date", "manual_state_note", "expected_lab_state", "report_found_in_checkout"])}

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
    if cfg.event_mode not in EVENT_MODES:
        raise ValueError(f"event mode must be one of {sorted(EVENT_MODES)}")
    run_id = "behavioral_state_similarity_" + datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    symbols_requested = [symbol.upper() for symbol in symbols]
    symbols_completed: list[str] = []
    symbols_failed: list[dict[str, str]] = []
    frames: list[pd.DataFrame] = []
    frames_by_symbol: dict[str, pd.DataFrame] = {}

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
        frames_by_symbol[symbol] = state_frame
        symbols_completed.append(symbol)

    all_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if all_rows.empty:
        wide_events = pd.DataFrame()
        baseline_input = pd.DataFrame()
    else:
        baseline_input = all_rows[all_rows["can_evaluate_state"].astype(bool)].copy()
        wide_events = baseline_input[
            baseline_input["primary_state_label"].ne("unclassified")
        ].copy()

    horizon_events = build_horizon_events(all_rows, cfg)
    independent_events = extract_independent_events(horizon_events, mode=cfg.event_mode)
    state_summary = summarize_state_responses(wide_events, cfg)
    horizon_state_summary = summarize_horizon_state_responses(independent_events)
    random_baseline = build_random_baseline(baseline_input, config=cfg)
    permutation_baseline = build_permutation_baseline(
        independent_events,
        config=cfg,
        permutation_count=cfg.permutation_count,
    )
    walk_forward_oos = run_oos_state_response_test(
        independent_events,
        config=cfg,
        split_mode="walk_forward",
    )
    leave_one_symbol_out = run_oos_state_response_test(
        independent_events,
        config=cfg,
        split_mode="leave_one_symbol_out",
    )
    oos_state_response = pd.concat(
        [walk_forward_oos, leave_one_symbol_out],
        ignore_index=True,
    )
    similarity_features = [
        column for column in DEFAULT_SIMILARITY_FEATURE_COLUMNS if column in wide_events.columns
    ]
    runtime_warnings: list[str] = []
    if not wide_events.empty:
        largest_state_sample = int(wide_events["primary_state_label"].value_counts().max())
        if largest_state_sample > MAX_RANDOM_BASELINE_STATE_ROWS:
            runtime_warnings.append(
                "random baseline sampled at most "
                f"{MAX_RANDOM_BASELINE_STATE_ROWS} rows per state/horizon"
            )
    neighbor_events = wide_events
    if len(wide_events) > MAX_NEAREST_NEIGHBOR_EVENTS:
        neighbor_events = wide_events.sample(
            n=MAX_NEAREST_NEIGHBOR_EVENTS,
            random_state=cfg.random_seed,
        )
        runtime_warnings.append(
            "nearest-neighbor similarity sampled "
            f"{MAX_NEAREST_NEIGHBOR_EVENTS} of {len(wide_events)} classified events"
        )
    match_summary = run_nearest_neighbor_similarity(
        neighbor_events,
        feature_columns=similarity_features,
        config=cfg,
    )
    oos_similarity_features = [
        column for column in DEFAULT_SIMILARITY_FEATURE_COLUMNS if column in independent_events.columns
    ]
    oos_neighbor_events = independent_events
    if len(oos_neighbor_events) > MAX_NEAREST_NEIGHBOR_EVENTS:
        oos_neighbor_events = oos_neighbor_events.sample(
            n=MAX_NEAREST_NEIGHBOR_EVENTS,
            random_state=cfg.random_seed,
        )
        runtime_warnings.append(
            "OOS nearest-neighbor similarity sampled "
            f"{MAX_NEAREST_NEIGHBOR_EVENTS} of {len(independent_events)} independent events"
        )
    nearest_neighbor_oos = run_nearest_neighbor_oos_similarity(
        oos_neighbor_events,
        feature_columns=oos_similarity_features,
        config=cfg,
    )
    response_shape_events = independent_events
    if len(response_shape_events) > MAX_RESPONSE_SHAPE_EVENTS:
        response_shape_events = response_shape_events.sample(
            n=MAX_RESPONSE_SHAPE_EVENTS,
            random_state=cfg.random_seed,
        )
        runtime_warnings.append(
            "same-state response-shape matching sampled "
            f"{MAX_RESPONSE_SHAPE_EVENTS} of {len(independent_events)} independent events"
        )
    (
        same_state_cross_symbol_matches,
        response_shape_similarity_summary,
        response_shape_baselines,
    ) = run_same_state_cross_symbol_similarity(
        response_shape_events,
        feature_columns=oos_similarity_features,
        config=cfg,
    )
    fingerprint_features = [
        column for column in STATE_FINGERPRINT_FEATURE_COLUMNS if column in independent_events.columns
    ]
    (
        fingerprint_cross_symbol_matches,
        fingerprint_similarity_summary,
        fingerprint_response_shape_baselines,
    ) = run_fingerprint_cross_symbol_similarity(
        response_shape_events,
        feature_columns=fingerprint_features,
        config=cfg,
    )
    oos_response_shape_similarity = pd.concat(
        [
            run_oos_response_shape_similarity(
                response_shape_events,
                feature_columns=fingerprint_features,
                config=cfg,
                similarity_mode="label",
            ),
            run_oos_response_shape_similarity(
                response_shape_events,
                feature_columns=fingerprint_features,
                config=cfg,
                similarity_mode="fingerprint",
            ),
        ],
        ignore_index=True,
    )
    candidate_states, block_states = _candidate_verdicts(
        state_summary=state_summary,
        random_baseline=random_baseline,
        match_summary=match_summary,
        config=cfg,
    )
    state_overlap_matrix = build_state_overlap_matrix(all_rows)
    state_priority_conflicts = build_state_priority_conflicts(all_rows)
    primary_state_distribution = build_primary_state_distribution(all_rows)
    per_symbol_state_counts = build_per_symbol_state_counts(independent_events)
    per_symbol_state_response = build_per_symbol_state_response(independent_events)
    state_transition_matrix = build_state_transition_matrix(all_rows)
    state_duration_summary = build_state_duration_summary(all_rows)
    time_of_day_state_summary = build_time_of_day_state_summary(independent_events)
    (
        symbol_concentration_report,
        session_concentration_report,
        concentration_warnings,
    ) = build_concentration_reports(independent_events, config=cfg)
    stimulus_response_matrix = build_stimulus_response_matrix(
        independent_events,
        permutation_baseline=permutation_baseline,
        config=cfg,
    )
    template_overlay_summary = run_template_overlay_diagnostics(frames_by_symbol, config=cfg)
    pipeline_passed = bool(symbols_completed) and not symbols_failed
    decision = build_decision_summary(
        oos_state_response=oos_state_response,
        permutation_baseline=permutation_baseline,
        concentration_warnings=concentration_warnings,
        template_overlay_summary=template_overlay_summary,
        pipeline_passed=pipeline_passed,
        config=cfg,
        response_shape_baselines=response_shape_baselines,
        response_shape_similarity_summary=response_shape_similarity_summary,
        fingerprint_shape_baselines=fingerprint_response_shape_baselines,
        fingerprint_similarity_summary=fingerprint_similarity_summary,
        oos_response_shape_similarity=oos_response_shape_similarity,
    )
    warnings = _collect_warnings(
        all_rows=all_rows,
        events=independent_events,
        symbols_failed=symbols_failed,
        config=cfg,
    )
    concentration_warning_text = [
        f"{row['state']} h{row['horizon']}: {row['warning']} {row['dominant_value']} "
        f"share={float(row['share']):.2f}"
        for _, row in concentration_warnings.iterrows()
    ]
    warnings = sorted(set([*warnings, *runtime_warnings, *concentration_warning_text]))
    state_counts = _state_counts(independent_events)

    top_rejected_states: list[dict[str, Any]] = []
    states_worth_continuing: list[dict[str, Any]] = []
    if not oos_state_response.empty:
        for _, row in oos_state_response.sort_values(
            ["test_event_count", "state"],
            ascending=[False, True],
        ).iterrows():
            output_row = {
                "state": row["state"],
                "horizon": int(row["horizon"]),
                "verdict": row["verdict"],
                "reason": (
                    "passed OOS gates"
                    if bool(row.get("gate_passed", False))
                    else "did not beat generic OOS baseline"
                ),
            }
            if bool(row.get("gate_passed", False)):
                states_worth_continuing.append(output_row)
            elif len(top_rejected_states) < 12:
                top_rejected_states.append(output_row)
    if not states_worth_continuing:
        states_worth_continuing = []

    best_same_state_states: list[dict[str, Any]] = []
    best_fingerprint_states: list[dict[str, Any]] = []
    states_requiring_subtype_split: list[dict[str, Any]] = []
    if not response_shape_baselines.empty:
        for (state, horizon), group in response_shape_baselines.groupby(
            ["state", "horizon"],
            sort=True,
        ):
            same = group[group["baseline"].eq("same_state_cross_symbol")]
            random = group[group["baseline"].eq("random_cross_symbol")]
            different = group[group["baseline"].eq("different_state_cross_symbol")]
            if same.empty:
                continue
            same_row = same.iloc[0]
            random_similarity = (
                float(random.iloc[0]["median_cosine_similarity"]) if not random.empty else math.nan
            )
            different_similarity = (
                float(different.iloc[0]["median_cosine_similarity"])
                if not different.empty
                else math.nan
            )
            same_similarity = float(same_row["median_cosine_similarity"])
            output_row = {
                "state": str(state),
                "horizon": int(horizon),
                "match_count": int(same_row["match_count"]),
                "median_cosine_similarity": same_similarity,
                "random_cosine_similarity": random_similarity,
                "different_state_cosine_similarity": different_similarity,
            }
            beats_random = (
                not math.isnan(same_similarity)
                and not math.isnan(random_similarity)
                and same_similarity - random_similarity >= cfg.min_similarity_excess_vs_random
            )
            beats_different = (
                not math.isnan(same_similarity)
                and not math.isnan(different_similarity)
                and same_similarity - different_similarity >= cfg.min_similarity_excess_vs_random
            )
            if beats_random and beats_different and len(best_same_state_states) < 12:
                best_same_state_states.append(output_row)
            elif len(states_requiring_subtype_split) < 12:
                states_requiring_subtype_split.append(
                    {
                        **output_row,
                        "reason": "same label did not separate response shape from controls",
                    }
                )
    if not fingerprint_response_shape_baselines.empty:
        for (state, horizon), group in fingerprint_response_shape_baselines.groupby(
            ["state", "horizon"],
            sort=True,
        ):
            fingerprint = group[group["baseline"].eq("fingerprint_cross_symbol")]
            random = group[group["baseline"].eq("random_cross_symbol")]
            different = group[group["baseline"].eq("different_state_cross_symbol")]
            if fingerprint.empty:
                continue
            fingerprint_row = fingerprint.iloc[0]
            fingerprint_similarity = float(fingerprint_row["median_cosine_similarity"])
            random_similarity = (
                float(random.iloc[0]["median_cosine_similarity"]) if not random.empty else math.nan
            )
            different_similarity = (
                float(different.iloc[0]["median_cosine_similarity"])
                if not different.empty
                else math.nan
            )
            beats_random = (
                not math.isnan(fingerprint_similarity)
                and not math.isnan(random_similarity)
                and fingerprint_similarity - random_similarity >= cfg.min_similarity_excess_vs_random
            )
            beats_different = (
                not math.isnan(fingerprint_similarity)
                and not math.isnan(different_similarity)
                and fingerprint_similarity - different_similarity
                >= cfg.min_similarity_excess_vs_random
            )
            if beats_random and beats_different and len(best_fingerprint_states) < 12:
                best_fingerprint_states.append(
                    {
                        "state": str(state),
                        "horizon": int(horizon),
                        "match_count": int(fingerprint_row["match_count"]),
                        "median_cosine_similarity": fingerprint_similarity,
                        "random_cosine_similarity": random_similarity,
                        "different_state_cosine_similarity": different_similarity,
                    }
                )

    manual_audit_examples = build_manual_audit_examples()
    if same_state_cross_symbol_matches.empty:
        state_match_examples = same_state_cross_symbol_matches.copy()
    else:
        state_match_examples = same_state_cross_symbol_matches.sort_values(
            ["cosine_similarity", "path_correlation"],
            ascending=[False, False],
        ).head(50)

    dead_chop_blocking_quality = [
        row
        for row in _records(stimulus_response_matrix)
        if row.get("state_label") == "dead_chop"
    ][:12]

    event_csv_path = run_dir / "events.csv"
    horizon_events_csv_path = run_dir / "horizon_events.csv"
    independent_events_csv_path = run_dir / "independent_events.csv"
    state_summary_csv_path = run_dir / "state_summary.csv"
    horizon_state_summary_csv_path = run_dir / "horizon_state_summary.csv"
    match_summary_csv_path = run_dir / "match_summary.csv"
    same_state_cross_symbol_matches_csv_path = run_dir / "same_state_cross_symbol_matches.csv"
    response_shape_similarity_summary_csv_path = run_dir / "response_shape_similarity_summary.csv"
    response_shape_baselines_csv_path = run_dir / "response_shape_baselines.csv"
    fingerprint_cross_symbol_matches_csv_path = run_dir / "fingerprint_cross_symbol_matches.csv"
    fingerprint_similarity_summary_csv_path = run_dir / "fingerprint_similarity_summary.csv"
    fingerprint_response_shape_baselines_csv_path = (
        run_dir / "fingerprint_response_shape_baselines.csv"
    )
    oos_response_shape_similarity_csv_path = run_dir / "oos_response_shape_similarity.csv"
    random_baseline_csv_path = run_dir / "random_baseline.csv"
    permutation_baseline_csv_path = run_dir / "permutation_baseline.csv"
    oos_state_response_csv_path = run_dir / "oos_state_response.csv"
    leave_one_symbol_out_csv_path = run_dir / "leave_one_symbol_out_summary.csv"
    nearest_neighbor_oos_csv_path = run_dir / "nearest_neighbor_oos_summary.csv"
    stimulus_response_matrix_csv_path = run_dir / "stimulus_response_matrix.csv"
    template_overlay_summary_csv_path = run_dir / "template_overlay_summary.csv"
    per_symbol_state_counts_csv_path = run_dir / "per_symbol_state_counts.csv"
    per_symbol_state_response_csv_path = run_dir / "per_symbol_state_response.csv"
    per_symbol_oos_portability_csv_path = run_dir / "per_symbol_oos_portability.csv"
    state_overlap_matrix_csv_path = run_dir / "state_overlap_matrix.csv"
    state_priority_conflicts_csv_path = run_dir / "state_priority_conflicts.csv"
    primary_state_distribution_csv_path = run_dir / "primary_state_distribution.csv"
    state_transition_matrix_csv_path = run_dir / "state_transition_matrix.csv"
    state_duration_summary_csv_path = run_dir / "state_duration_summary.csv"
    time_of_day_state_summary_csv_path = run_dir / "time_of_day_state_summary.csv"
    symbol_concentration_report_csv_path = run_dir / "symbol_concentration_report.csv"
    session_concentration_report_csv_path = run_dir / "session_concentration_report.csv"
    concentration_warnings_csv_path = run_dir / "concentration_warnings.csv"
    state_definitions_json_path = run_dir / "state_definitions.json"
    manual_audit_examples_csv_path = run_dir / "manual_audit_examples.csv"
    state_match_examples_csv_path = run_dir / "state_match_examples.csv"
    decision_json_path = run_dir / "decision.json"
    summary_json_path = run_dir / "summary.json"
    summary_markdown_path = run_dir / "summary.md"
    _write_csv(event_csv_path, horizon_events)
    _write_csv(horizon_events_csv_path, horizon_events)
    _write_csv(independent_events_csv_path, independent_events)
    _write_csv(state_summary_csv_path, state_summary)
    _write_csv(horizon_state_summary_csv_path, horizon_state_summary)
    _write_csv(match_summary_csv_path, match_summary)
    _write_csv(same_state_cross_symbol_matches_csv_path, same_state_cross_symbol_matches)
    _write_csv(response_shape_similarity_summary_csv_path, response_shape_similarity_summary)
    _write_csv(response_shape_baselines_csv_path, response_shape_baselines)
    _write_csv(fingerprint_cross_symbol_matches_csv_path, fingerprint_cross_symbol_matches)
    _write_csv(fingerprint_similarity_summary_csv_path, fingerprint_similarity_summary)
    _write_csv(fingerprint_response_shape_baselines_csv_path, fingerprint_response_shape_baselines)
    _write_csv(oos_response_shape_similarity_csv_path, oos_response_shape_similarity)
    _write_csv(random_baseline_csv_path, random_baseline)
    _write_csv(permutation_baseline_csv_path, permutation_baseline)
    _write_csv(oos_state_response_csv_path, oos_state_response)
    _write_csv(leave_one_symbol_out_csv_path, leave_one_symbol_out)
    _write_csv(nearest_neighbor_oos_csv_path, nearest_neighbor_oos)
    _write_csv(stimulus_response_matrix_csv_path, stimulus_response_matrix)
    _write_csv(template_overlay_summary_csv_path, template_overlay_summary)
    _write_csv(per_symbol_state_counts_csv_path, per_symbol_state_counts)
    _write_csv(per_symbol_state_response_csv_path, per_symbol_state_response)
    _write_csv(per_symbol_oos_portability_csv_path, leave_one_symbol_out)
    _write_csv(state_overlap_matrix_csv_path, state_overlap_matrix)
    _write_csv(state_priority_conflicts_csv_path, state_priority_conflicts)
    _write_csv(primary_state_distribution_csv_path, primary_state_distribution)
    _write_csv(state_transition_matrix_csv_path, state_transition_matrix)
    _write_csv(state_duration_summary_csv_path, state_duration_summary)
    _write_csv(time_of_day_state_summary_csv_path, time_of_day_state_summary)
    _write_csv(symbol_concentration_report_csv_path, symbol_concentration_report)
    _write_csv(session_concentration_report_csv_path, session_concentration_report)
    _write_csv(concentration_warnings_csv_path, concentration_warnings)
    _write_csv(manual_audit_examples_csv_path, manual_audit_examples)
    _write_csv(state_match_examples_csv_path, state_match_examples)
    state_definitions_json_path.write_text(
        json.dumps(STATE_DEFINITIONS, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    decision_json_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

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
        "total_raw_rows": int(len(all_rows)),
        "total_horizon_events": int(len(horizon_events)),
        "total_independent_events": int(len(independent_events)),
        "state_counts": state_counts,
        "raw_state_counts": _state_counts(wide_events),
        "state_response_summary": _records(state_summary),
        "horizon_state_response_summary": _records(horizon_state_summary),
        "random_baseline_summary": _records(random_baseline),
        "permutation_baseline_summary": _records(permutation_baseline),
        "oos_state_response_summary": _records(oos_state_response),
        "leave_one_symbol_out_summary": _records(leave_one_symbol_out),
        "nearest_neighbor_summary": _records(match_summary),
        "nearest_neighbor_oos_summary": _records(nearest_neighbor_oos),
        "response_shape_similarity_summary": _records(response_shape_similarity_summary),
        "response_shape_baselines": _records(response_shape_baselines),
        "fingerprint_similarity_summary": _records(fingerprint_similarity_summary),
        "fingerprint_response_shape_baselines": _records(fingerprint_response_shape_baselines),
        "oos_response_shape_similarity": _records(oos_response_shape_similarity),
        "stimulus_response_matrix": _records(stimulus_response_matrix),
        "template_overlay_summary": _records(template_overlay_summary),
        "candidate_portable_states": candidate_states,
        "block_states": block_states,
        "best_same_state_cross_stock_states": best_same_state_states,
        "best_fingerprint_cross_stock_states": best_fingerprint_states,
        "top_rejected_states": top_rejected_states,
        "states_worth_continuing_research": states_worth_continuing,
        "states_requiring_subtype_split": states_requiring_subtype_split,
        "dead_chop_blocking_quality": dead_chop_blocking_quality,
        "manual_audit_examples": _records(manual_audit_examples),
        "state_match_examples": _records(state_match_examples),
        "warnings": warnings,
        "files": {
            "events_csv": str(event_csv_path),
            "horizon_events_csv": str(horizon_events_csv_path),
            "independent_events_csv": str(independent_events_csv_path),
            "state_summary_csv": str(state_summary_csv_path),
            "horizon_state_summary_csv": str(horizon_state_summary_csv_path),
            "match_summary_csv": str(match_summary_csv_path),
            "same_state_cross_symbol_matches_csv": str(same_state_cross_symbol_matches_csv_path),
            "response_shape_similarity_summary_csv": str(
                response_shape_similarity_summary_csv_path
            ),
            "response_shape_baselines_csv": str(response_shape_baselines_csv_path),
            "fingerprint_cross_symbol_matches_csv": str(fingerprint_cross_symbol_matches_csv_path),
            "fingerprint_similarity_summary_csv": str(fingerprint_similarity_summary_csv_path),
            "fingerprint_response_shape_baselines_csv": str(
                fingerprint_response_shape_baselines_csv_path
            ),
            "oos_response_shape_similarity_csv": str(oos_response_shape_similarity_csv_path),
            "random_baseline_csv": str(random_baseline_csv_path),
            "permutation_baseline_csv": str(permutation_baseline_csv_path),
            "oos_state_response_csv": str(oos_state_response_csv_path),
            "leave_one_symbol_out_csv": str(leave_one_symbol_out_csv_path),
            "nearest_neighbor_oos_csv": str(nearest_neighbor_oos_csv_path),
            "stimulus_response_matrix_csv": str(stimulus_response_matrix_csv_path),
            "template_overlay_summary_csv": str(template_overlay_summary_csv_path),
            "per_symbol_state_counts_csv": str(per_symbol_state_counts_csv_path),
            "per_symbol_state_response_csv": str(per_symbol_state_response_csv_path),
            "per_symbol_oos_portability_csv": str(per_symbol_oos_portability_csv_path),
            "state_overlap_matrix_csv": str(state_overlap_matrix_csv_path),
            "state_priority_conflicts_csv": str(state_priority_conflicts_csv_path),
            "primary_state_distribution_csv": str(primary_state_distribution_csv_path),
            "state_transition_matrix_csv": str(state_transition_matrix_csv_path),
            "state_duration_summary_csv": str(state_duration_summary_csv_path),
            "time_of_day_state_summary_csv": str(time_of_day_state_summary_csv_path),
            "symbol_concentration_report_csv": str(symbol_concentration_report_csv_path),
            "session_concentration_report_csv": str(session_concentration_report_csv_path),
            "concentration_warnings_csv": str(concentration_warnings_csv_path),
            "state_definitions_json": str(state_definitions_json_path),
            "manual_audit_examples_csv": str(manual_audit_examples_csv_path),
            "state_match_examples_csv": str(state_match_examples_csv_path),
            "decision_json": str(decision_json_path),
            "summary_json": str(summary_json_path),
            "summary_markdown": str(summary_markdown_path),
        },
        **decision,
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
        pipeline_passed=decision["pipeline_passed"],
        label_similarity_supported=decision["label_similarity_supported"],
        fingerprint_similarity_supported=decision["fingerprint_similarity_supported"],
        state_similarity_supported=decision["state_similarity_supported"],
        oos_similarity_supported=decision["oos_similarity_supported"],
        template_overlay_supported=decision["template_overlay_supported"],
        decision=decision["decision"],
        decision_reasons=decision["decision_reasons"],
    )


__all__ = [
    "BehavioralStateConfig",
    "BehavioralStateLabResult",
    "STATE_FINGERPRINT_FEATURE_COLUMNS",
    "add_forward_response_columns",
    "apply_state_gate_to_positions",
    "build_behavioral_state_frame",
    "build_decision_summary",
    "build_horizon_events",
    "build_manual_audit_examples",
    "build_permutation_baseline",
    "build_random_baseline",
    "extract_independent_events",
    "label_behavioral_states",
    "run_fingerprint_cross_symbol_similarity",
    "run_behavioral_state_similarity_lab",
    "run_nearest_neighbor_oos_similarity",
    "run_nearest_neighbor_similarity",
    "run_oos_state_response_test",
    "run_oos_response_shape_similarity",
    "run_same_state_cross_symbol_similarity",
    "summarize_state_responses",
]
