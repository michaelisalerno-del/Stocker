"""Research-only event detector for manual behavioral state examples.

This module intentionally emits sparse event rows instead of assigning a broad
state label to every intraday bar. Detection features are current/prior-bar only;
forward response columns are added after event detection as targets.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocker_data.storage import DatasetKey, dataset_path, read_parquet
from stocker_research.behavioral_state_similarity import (
    BehavioralStateConfig,
    add_forward_response_columns,
    build_behavioral_state_frame,
)

DEFAULT_OUTPUT_DIR = Path("data/reports/research/state_event_detector_v0")
EVENT_STATES = (
    "controlled_pullback_after_bullish_impulse",
    "failed_bullish_impulse_recoil",
    "liquidation_failed_low_reclaim",
    "failed_bounce_active_liquidation",
    "failed_open_down_continuation",
    "slow_snapback_after_dip",
    "dead_chop_blocker",
)
EVENT_DEFINITIONS: dict[str, dict[str, str]] = {
    "controlled_pullback_after_bullish_impulse": {
        "event_family": "bullish_impulse_pullback",
        "event_direction": "up",
    },
    "failed_bullish_impulse_recoil": {
        "event_family": "bullish_impulse_failure",
        "event_direction": "down",
    },
    "liquidation_failed_low_reclaim": {
        "event_family": "liquidation_reclaim",
        "event_direction": "up",
    },
    "failed_bounce_active_liquidation": {
        "event_family": "active_liquidation_blocker",
        "event_direction": "down",
    },
    "failed_open_down_continuation": {
        "event_family": "opening_structure_failure",
        "event_direction": "down",
    },
    "slow_snapback_after_dip": {
        "event_family": "slow_recovery_after_dip",
        "event_direction": "up",
    },
    "dead_chop_blocker": {
        "event_family": "dead_chop_blocker",
        "event_direction": "neutral",
    },
}
DETECTION_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
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
    "distance_from_vwap_pct",
    "distance_from_opening_range_mid_pct",
    "distance_from_opening_range_high_pct",
    "distance_from_opening_range_low_pct",
    "distance_from_session_open_pct",
    "distance_from_session_high_pct",
    "distance_from_session_low_pct",
    "distance_from_recent_high_pct",
    "distance_from_recent_low_pct",
    "opening_range_high",
    "opening_range_low",
    "opening_range_mid",
    "opening_range_width",
    "session_vwap",
    "session_high_to_date",
    "session_low_to_date",
    "recent_high",
    "recent_low",
    "vwap_cross_count_12",
    "range_cross_count_12",
    "rolling_intraday_range_pct",
    "compression_zscore",
    "range_zscore",
    "return_zscore",
    "relative_volume_at_bar_index",
    "relative_cumulative_volume",
    "impulse_midpoint",
    "impulse_return_12",
    "impulse_range_zscore_max_12",
    "impulse_relative_volume_max_12",
    "impulse_volume_ratio",
    "pullback_depth_from_recent_high",
    "reclaim_from_recent_low",
]


@dataclass(frozen=True)
class StateEventDetectorConfig:
    """Configuration for the focused event-detector lab."""

    timeframe: str = "5m"
    market_calendar: str | None = "XNYS"
    horizons: tuple[int, ...] = (6, 9, 12, 24)
    min_bars_after_open: int = 6
    entry_cutoff_before_close_minutes: int = 30
    relative_volume_lookback_sessions: int = 20
    direction_windows: tuple[int, ...] = (3, 6, 12)
    event_mode: str = "state_entry_non_overlapping"
    random_seed: int = 1337
    min_events_for_similarity: int = 30
    min_symbols_for_key_state: int = 3
    max_single_symbol_share: float = 0.50
    max_single_session_share: float = 0.20
    max_similarity_events: int = 4_000

    def behavioral_config(self) -> BehavioralStateConfig:
        return BehavioralStateConfig(
            timeframe=self.timeframe,
            market_calendar=self.market_calendar,
            horizons=self.horizons,
            min_bars_after_open=self.min_bars_after_open,
            entry_cutoff_before_close_minutes=self.entry_cutoff_before_close_minutes,
            relative_volume_lookback_sessions=self.relative_volume_lookback_sessions,
            direction_windows=self.direction_windows,
            random_seed=self.random_seed,
            event_mode=self.event_mode,
            min_independent_events_per_state_horizon=self.min_events_for_similarity,
        )


@dataclass(frozen=True)
class StateEventDetectorResult:
    """Paths and headline counts from a state-event-detector run."""

    run_id: str
    output_dir: Path
    summary_json_path: Path
    summary_markdown_path: Path
    event_rows_csv_path: Path
    manual_state_audit_csv_path: Path
    event_state_summary_csv_path: Path
    same_event_cross_symbol_similarity_csv_path: Path
    random_baseline_csv_path: Path
    oos_event_response_csv_path: Path
    concentration_warnings_csv_path: Path
    decision_json_path: Path
    symbols_requested: list[str]
    symbols_completed: list[str]
    symbols_failed: dict[str, str]
    total_event_rows: int
    manual_audit_passed: bool
    decision: str


DEFAULT_MANUAL_AUDIT_EXAMPLES: list[dict[str, Any]] = [
    {
        "symbol": "HOOD",
        "session_date": "2026-04-15",
        "expected_event_states": ("controlled_pullback_after_bullish_impulse",),
        "manual_note": "HOOD controlled pullback after impulse",
    },
    {
        "symbol": "HOOD",
        "session_date": "2025-07-02",
        "expected_event_states": ("controlled_pullback_after_bullish_impulse",),
        "manual_note": "HOOD controlled pullback after impulse",
    },
    {
        "symbol": "HOOD",
        "session_date": "2025-07-01",
        "expected_event_states": ("failed_bullish_impulse_recoil",),
        "manual_note": "HOOD bullish impulse failure",
    },
    {
        "symbol": "HOOD",
        "session_date": "2026-06-24",
        "expected_event_states": (
            "failed_open_down_continuation",
            "liquidation_failed_low_reclaim",
        ),
        "manual_note": "HOOD open failure and/or low reclaim",
    },
    {
        "symbol": "GLW",
        "session_date": "2026-06-24",
        "expected_event_states": (
            "liquidation_failed_low_reclaim",
            "controlled_pullback_after_bullish_impulse",
        ),
        "manual_note": "GLW early reclaim or controlled pullback",
    },
    {
        "symbol": "GLW",
        "session_date": "2026-06-24",
        "expected_event_states": ("failed_bullish_impulse_recoil",),
        "manual_note": "GLW later extension failure",
    },
    {
        "symbol": "GLW",
        "session_date": "2026-05-12",
        "expected_event_states": ("failed_bounce_active_liquidation",),
        "manual_note": "GLW early failed bounce in active liquidation",
    },
    {
        "symbol": "GLW",
        "session_date": "2026-05-12",
        "expected_event_states": ("liquidation_failed_low_reclaim",),
        "manual_note": "GLW later failed low reclaim",
    },
    {
        "symbol": "GLW",
        "session_date": "2026-02-17",
        "expected_event_states": ("liquidation_failed_low_reclaim",),
        "manual_note": "GLW failed low reclaim",
    },
    {
        "symbol": "GLW",
        "session_date": "2025-07-01",
        "expected_event_states": ("dead_chop_blocker",),
        "manual_note": "GLW dead chop blocker",
    },
    {
        "symbol": "GLW",
        "session_date": "2025-08-15",
        "expected_event_states": ("dead_chop_blocker",),
        "manual_note": "GLW dead chop blocker",
    },
    {
        "symbol": "FCX",
        "session_date": "2025-10-10",
        "expected_event_states": ("failed_open_down_continuation",),
        "manual_note": "FCX failed open down continuation or active liquidation",
    },
    {
        "symbol": "FCX",
        "session_date": "2026-02-17",
        "expected_event_states": ("slow_snapback_after_dip",),
        "manual_note": "FCX slow snapback after dip",
    },
    {
        "symbol": "FCX",
        "session_date": "2026-05-12",
        "expected_event_states": ("slow_snapback_after_dip",),
        "manual_note": "FCX slow snapback after dip",
    },
    {
        "symbol": "FCX",
        "session_date": "2026-06-24",
        "expected_event_states": ("dead_chop_blocker",),
        "manual_note": "FCX dead chop blocker",
    },
    {
        "symbol": "CRM",
        "session_date": "2025-10-10",
        "expected_event_states": ("failed_open_down_continuation",),
        "manual_note": "CRM failed open down continuation",
    },
    {
        "symbol": "CRM",
        "session_date": "2026-02-17",
        "expected_event_states": ("failed_open_down_continuation",),
        "manual_note": "CRM failed open down continuation",
    },
    {
        "symbol": "CRM",
        "session_date": "2026-05-12",
        "expected_event_states": ("failed_open_down_continuation",),
        "manual_note": "CRM failed open down continuation",
    },
    {
        "symbol": "CRM",
        "session_date": "2026-06-24",
        "expected_event_states": ("failed_bullish_impulse_recoil",),
        "manual_note": "CRM early pop rejection",
    },
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    output = numerator / denominator
    output = output.replace([np.inf, -np.inf], np.nan)
    return output


def _safe_pct_distance(value: pd.Series, reference: pd.Series) -> pd.Series:
    return _safe_divide(value - reference, reference)


def _numeric(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data:
        return pd.Series(np.nan, index=data.index, dtype="float")
    return pd.to_numeric(data[column], errors="coerce")


def _bool(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data:
        return pd.Series(False, index=data.index, dtype="bool")
    return data[column].astype(bool)


def _session_shift(data: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    return series.groupby(data["session_date"]).shift(periods)


def _rolling_by_session(
    data: pd.DataFrame,
    series: pd.Series,
    window: int,
    method: str,
    *,
    min_periods: int = 1,
) -> pd.Series:
    grouped = series.groupby(data["session_date"], group_keys=False)
    if method == "max":
        return grouped.apply(lambda item: item.rolling(window, min_periods=min_periods).max())
    if method == "min":
        return grouped.apply(lambda item: item.rolling(window, min_periods=min_periods).min())
    if method == "mean":
        return grouped.apply(lambda item: item.rolling(window, min_periods=min_periods).mean())
    if method == "sum":
        return grouped.apply(lambda item: item.rolling(window, min_periods=min_periods).sum())
    raise ValueError(f"Unsupported rolling method: {method}")


def _with_event_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = _numeric(data, "close")
    high = _numeric(data, "high")
    low = _numeric(data, "low")
    volume = _numeric(data, "volume")

    prior_12_close = _session_shift(data, close, 12)
    recent_high_12 = _rolling_by_session(data, high, 12, "max", min_periods=3)
    recent_low_12 = _rolling_by_session(data, low, 12, "min", min_periods=3)
    fallback_midpoint = (recent_high_12 + recent_low_12) / 2.0
    impulse_midpoint = prior_12_close + ((recent_high_12 - prior_12_close) * 0.5)
    data["impulse_midpoint"] = impulse_midpoint.where(impulse_midpoint.notna(), fallback_midpoint)
    data["impulse_return_12"] = _safe_pct_distance(recent_high_12, prior_12_close)
    data["impulse_range_zscore_max_12"] = _rolling_by_session(
        data,
        _numeric(data, "range_zscore"),
        12,
        "max",
        min_periods=2,
    )
    data["impulse_relative_volume_max_12"] = _rolling_by_session(
        data,
        _numeric(data, "relative_volume_at_bar_index"),
        12,
        "max",
        min_periods=2,
    )

    pullback_volume = _rolling_by_session(data, volume, 3, "mean", min_periods=2)
    prior_volume = _rolling_by_session(
        data,
        volume.groupby(data["session_date"]).shift(3),
        9,
        "mean",
        min_periods=3,
    )
    data["impulse_volume_ratio"] = _safe_divide(pullback_volume, prior_volume)
    data["pullback_depth_from_recent_high"] = _safe_pct_distance(close, recent_high_12)
    data["reclaim_from_recent_low"] = _safe_pct_distance(close, recent_low_12)

    max_horizon = 0
    for column in data.columns:
        if column.startswith("forward_") and column.endswith("_bar_return"):
            try:
                max_horizon = max(max_horizon, int(column.split("_")[1]))
            except (IndexError, ValueError):
                continue
    for step in range(1, max_horizon + 1):
        future_close = close.groupby(data["session_date"]).shift(-step)
        data[f"path_return_{step}"] = _safe_pct_distance(future_close, close)
    return data


def _confidence(*components: pd.Series) -> pd.Series:
    if not components:
        return pd.Series(dtype="float")
    frame = pd.concat([item.astype(float) for item in components], axis=1)
    return frame.mean(axis=1).fillna(0.0).clip(lower=0.0, upper=1.0)


def _available_or_true(condition: pd.Series, availability: pd.Series) -> pd.Series:
    return condition.fillna(False) | availability.isna()


def _candidate_conditions(data: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series, str]]:
    can = _bool(data, "can_evaluate_state")
    opening_complete = _bool(data, "opening_range_complete")
    above_vwap = _bool(data, "above_vwap")
    below_vwap = _bool(data, "below_vwap")
    new_session_high = _bool(data, "new_session_high")
    failed_new_high = _bool(data, "failed_new_high")
    new_session_low = _bool(data, "new_session_low")
    failed_new_low = _bool(data, "failed_new_low")

    close = _numeric(data, "close")
    high = _numeric(data, "high")
    low = _numeric(data, "low")
    bar_return = _numeric(data, "bar_return")
    prior_3 = _numeric(data, "prior_3_bar_return")
    prior_6 = _numeric(data, "prior_6_bar_return")
    prior_12 = _numeric(data, "prior_12_bar_return")
    eff_6 = _numeric(data, "directional_efficiency_6")
    eff_12 = _numeric(data, "directional_efficiency_12")
    close_location = _numeric(data, "close_location_value")
    upper_wick = _numeric(data, "upper_wick_pct_of_range")
    lower_wick = _numeric(data, "lower_wick_pct_of_range")
    dist_vwap = _numeric(data, "distance_from_vwap_pct")
    dist_or_mid = _numeric(data, "distance_from_opening_range_mid_pct")
    dist_or_low = _numeric(data, "distance_from_opening_range_low_pct")
    dist_session_low = _numeric(data, "distance_from_session_low_pct")
    dist_session_open = _numeric(data, "distance_from_session_open_pct")
    rolling_range = _numeric(data, "rolling_intraday_range_pct")
    compression = _numeric(data, "compression_zscore")
    range_zscore = _numeric(data, "range_zscore")
    impulse_range_zscore = _numeric(data, "impulse_range_zscore_max_12")
    impulse_relative_volume = _numeric(data, "impulse_relative_volume_max_12")
    return_zscore = _numeric(data, "return_zscore")
    vwap_crosses = _numeric(data, "vwap_cross_count_12")
    range_crosses = _numeric(data, "range_cross_count_12")
    minutes_from_open = _numeric(data, "minutes_from_session_open")
    opening_low = _numeric(data, "opening_range_low")
    opening_mid = _numeric(data, "opening_range_mid")
    prior_recent_low = _numeric(data, "prior_recent_low")
    prior_recent_high = _numeric(data, "prior_recent_high")
    impulse_midpoint = _numeric(data, "impulse_midpoint")
    impulse_return = _numeric(data, "impulse_return_12")
    impulse_volume_ratio = _numeric(data, "impulse_volume_ratio")
    pullback_depth = _numeric(data, "pullback_depth_from_recent_high")
    reclaim_from_low = _numeric(data, "reclaim_from_recent_low")

    above_reference = (
        above_vwap
        | (dist_vwap > -0.0015)
        | (dist_or_mid > -0.0015)
        | dist_or_mid.isna()
    )
    bullish_impulse = (
        (prior_12 > 0.007)
        | (prior_6 > 0.0045)
        | (impulse_return > 0.008)
    )
    abnormal_impulse = _available_or_true(
        (impulse_range_zscore > 0.35)
        | (impulse_relative_volume > 1.10)
        | (impulse_return > 0.008),
        impulse_range_zscore.combine_first(impulse_relative_volume),
    )
    shallow_pullback = (
        (prior_3 <= 0.0025)
        & (prior_3 > -0.0075)
        & (pullback_depth > -0.018)
    )
    structure_held = close >= (impulse_midpoint * 0.995)
    volume_contracts = _available_or_true(impulse_volume_ratio <= 1.10, impulse_volume_ratio)
    stabilizes = (close_location >= 0.45) & (
        (bar_return >= -0.0015) | (close > _session_shift(data, close, 1))
    )
    controlled_pullback = (
        can
        & bullish_impulse
        & abnormal_impulse
        & above_reference
        & shallow_pullback
        & structure_held
        & volume_contracts
        & stabilizes
    )

    failed_impulse = (
        can
        & ((prior_6 > 0.0045) | (prior_12 > 0.008) | new_session_high)
        & (
            (upper_wick >= 0.32)
            | failed_new_high
            | ((high > prior_recent_high) & (close < prior_recent_high))
        )
        & (
            (close < impulse_midpoint)
            | (close_location <= 0.45)
            | (bar_return < 0.0)
            | (return_zscore < -0.5)
        )
    )

    downside_impulse = (prior_12 < -0.0075) | (prior_6 < -0.0045)
    near_low = new_session_low | (dist_session_low < 0.0075) | (low <= prior_recent_low)
    selling_slows = (prior_3 > prior_6 * 0.65) | (close_location >= 0.50)
    reclaim_low = (
        failed_new_low
        | (close > prior_recent_low)
        | (lower_wick >= 0.30)
        | (reclaim_from_low > 0.003)
    )
    low_reclaim = (
        can
        & downside_impulse
        & near_low
        & selling_slows
        & reclaim_low
        & (close_location >= 0.48)
        & (bar_return >= -0.002)
    )

    failed_bounce = (
        can
        & (prior_12 < -0.006)
        & ((prior_3 > -0.001) | (high > _session_shift(data, high, 1)))
        & ((below_vwap | (dist_vwap < 0.001)) & ((dist_or_mid < 0.001) | dist_or_mid.isna()))
        & ((bar_return <= 0.0005) | (close < _session_shift(data, close, 1)))
        & (close_location <= 0.50)
        & ((close < opening_mid) | opening_mid.isna())
    )

    open_down = (
        can
        & opening_complete
        & minutes_from_open.between(30, 150, inclusive="both")
        & (below_vwap | (dist_vwap < -0.001))
        & ((close < opening_mid) | (close < opening_low) | (dist_or_low < 0.0025))
        & ((prior_6 < -0.0035) | (prior_12 < -0.0045) | (dist_session_open < -0.003))
    )

    slow_snapback = (
        can
        & minutes_from_open.between(45, 300, inclusive="both")
        & ((prior_12 < -0.004) | (dist_session_low < 0.012))
        & (reclaim_from_low > 0.004)
        & (prior_3 > -0.001)
        & (bar_return < 0.004)
        & (eff_6 <= 0.55)
        & (close_location >= 0.42)
        & ((range_zscore < 1.25) | range_zscore.isna())
    )

    dead_chop = (
        can
        & (prior_12.abs() <= 0.0055)
        & (eff_12 <= 0.38)
        & ((rolling_range <= 0.008) | (compression <= 0.20) | compression.isna())
        & ((vwap_crosses >= 2.0) | (range_crosses >= 2.0) | (prior_6.abs() <= 0.0028))
    )

    return {
        "controlled_pullback_after_bullish_impulse": (
            controlled_pullback,
            _confidence(
                bullish_impulse,
                above_reference,
                shallow_pullback,
                structure_held,
                volume_contracts,
                stabilizes,
            ),
            "bullish impulse, shallow low-volume pullback, structure held, stabilization/reclaim",
        ),
        "failed_bullish_impulse_recoil": (
            failed_impulse,
            _confidence(
                (prior_6 > 0.0045) | (prior_12 > 0.008) | new_session_high,
                (upper_wick >= 0.32) | failed_new_high,
                (close_location <= 0.45) | (bar_return < 0.0),
            ),
            "bullish impulse/new high rejected by upper wick or weak close",
        ),
        "liquidation_failed_low_reclaim": (
            low_reclaim,
            _confidence(
                downside_impulse,
                near_low,
                selling_slows,
                reclaim_low,
                close_location >= 0.48,
            ),
            "downside impulse reaches low area, selling slows, low is reclaimed",
        ),
        "failed_bounce_active_liquidation": (
            failed_bounce,
            _confidence(
                prior_12 < -0.006,
                below_vwap | (dist_vwap < 0.001),
                close_location <= 0.50,
            ),
            "bounce attempt fails below VWAP/opening midpoint during active liquidation",
        ),
        "failed_open_down_continuation": (
            open_down,
            _confidence(
                opening_complete,
                below_vwap | (dist_vwap < -0.001),
                (close < opening_mid) | (close < opening_low),
                (prior_6 < -0.0035) | (prior_12 < -0.0045),
            ),
            "opening range structure failed below VWAP/opening midpoint",
        ),
        "slow_snapback_after_dip": (
            slow_snapback,
            _confidence(
                (prior_12 < -0.004) | (dist_session_low < 0.012),
                reclaim_from_low > 0.004,
                eff_6 <= 0.55,
                close_location >= 0.42,
            ),
            "dip stabilizes with gradual reclaim and moderate/low directional efficiency",
        ),
        "dead_chop_blocker": (
            dead_chop,
            _confidence(prior_12.abs() <= 0.0055, eff_12 <= 0.38, vwap_crosses >= 2.0),
            "low efficiency, low/normal range, repeated VWAP/opening-mid crosses",
        ),
    }


def _state_entry_mask(data: pd.DataFrame, condition: pd.Series) -> pd.Series:
    current = condition.fillna(False).astype(bool)
    grouped_previous = current.groupby([data["symbol"], data["session_date"]]).shift(1)
    previous = grouped_previous.astype("boolean").fillna(False).astype(bool)
    return current & ~previous


def _candidate_labels_at_index(
    candidate_conditions: dict[str, tuple[pd.Series, pd.Series, str]],
    index: int,
) -> str:
    labels = [
        state
        for state, (condition, _, _) in candidate_conditions.items()
        if bool(condition.loc[index])
    ]
    return "|".join(labels)


def _select_non_overlapping(events: pd.DataFrame, *, embargo_bars: int) -> pd.DataFrame:
    if events.empty or embargo_bars <= 0:
        return events.reset_index(drop=True)
    selected_rows: list[pd.Series] = []
    data = events.sort_values(["symbol", "session_date", "bar_index_in_session", "event_state"])
    for _, group in data.groupby(["symbol", "session_date", "event_state"], sort=False):
        next_allowed = -math.inf
        for _, row in group.iterrows():
            bar_index = int(row["bar_index_in_session"])
            if bar_index < next_allowed:
                continue
            selected_rows.append(row)
            next_allowed = bar_index + embargo_bars
    if not selected_rows:
        return events.iloc[0:0].copy().reset_index(drop=True)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def detect_state_events(
    frame: pd.DataFrame,
    *,
    symbol: str,
    config: StateEventDetectorConfig | None = None,
) -> pd.DataFrame:
    """Return sparse event rows for one symbol using current/prior bars only."""

    cfg = config or StateEventDetectorConfig()
    features = build_behavioral_state_frame(
        frame,
        symbol=symbol,
        config=cfg.behavioral_config(),
    )
    targets = add_forward_response_columns(features, cfg.horizons)
    targets = _with_event_features(targets)
    candidates = _candidate_conditions(targets)
    rows: list[dict[str, Any]] = []
    feature_columns = [column for column in DETECTION_FEATURE_COLUMNS if column in targets]
    forward_columns: list[str] = []
    for horizon in cfg.horizons:
        forward_columns.extend(
            [
                f"forward_{horizon}_bar_return",
                f"forward_{horizon}_bar_mfe",
                f"forward_{horizon}_bar_mae",
                f"forward_{horizon}_bar_abs_return",
            ]
        )
    path_columns = [
        f"path_return_{step}"
        for step in range(1, max(cfg.horizons, default=0) + 1)
        if f"path_return_{step}" in targets
    ]

    for event_state in EVENT_STATES:
        condition, confidence, reason = candidates[event_state]
        entry = _state_entry_mask(targets, condition)
        for index in targets.index[entry]:
            row = targets.loc[index]
            event_definition = EVENT_DEFINITIONS[event_state]
            output: dict[str, Any] = {
                "symbol": str(row["symbol"]).upper(),
                "timestamp": row["timestamp"],
                "session_date": str(row["session_date"]),
                "bar_index_in_session": int(row["bar_index_in_session"]),
                "bar_index_bucket": str(row.get("bar_index_bucket", "")),
                "time_of_day_bucket": str(row.get("time_of_day_bucket", "")),
                "event_state": event_state,
                "event_family": event_definition["event_family"],
                "event_direction": event_definition["event_direction"],
                "event_confidence_score": float(confidence.loc[index]),
                "trigger_reason": reason,
                "overlap_candidates": _candidate_labels_at_index(candidates, index),
                "manual_audit_match": False,
                "state_entry": True,
                "raw_row_index": int(index),
            }
            for column in feature_columns + forward_columns + path_columns:
                output[column] = row.get(column, np.nan)
            rows.append(output)

    if not rows:
        base_columns = [
            "symbol",
            "timestamp",
            "session_date",
            "bar_index_in_session",
            "bar_index_bucket",
            "time_of_day_bucket",
            "event_state",
            "event_family",
            "event_direction",
            "event_confidence_score",
            "trigger_reason",
            "overlap_candidates",
            "manual_audit_match",
            "state_entry",
            "raw_row_index",
            *feature_columns,
            *forward_columns,
            *path_columns,
        ]
        return pd.DataFrame(columns=base_columns)

    events = pd.DataFrame(rows)
    if cfg.event_mode == "state_entry_non_overlapping":
        events = _select_non_overlapping(events, embargo_bars=max(cfg.horizons, default=0))
    elif cfg.event_mode not in {"all_rows", "state_entry_only"}:
        raise ValueError(
            "event_mode must be one of all_rows, state_entry_only, "
            "state_entry_non_overlapping"
        )
    return events.sort_values(["symbol", "timestamp", "event_state"]).reset_index(drop=True)


def _normalize_examples(examples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for example in examples:
        states = example.get("expected_event_states", ())
        if isinstance(states, str):
            expected_states = tuple(item.strip() for item in states.split("|") if item.strip())
        else:
            expected_states = tuple(str(item) for item in states)
        normalized.append(
            {
                "symbol": str(example["symbol"]).upper(),
                "session_date": str(example["session_date"]),
                "expected_event_states": expected_states,
                "manual_note": str(example.get("manual_note", "")),
            }
        )
    return normalized


def audit_manual_state_examples(
    event_rows: pd.DataFrame,
    *,
    examples: Sequence[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Check whether expected manual examples appear at least once per session."""

    manual_examples = _normalize_examples(
        DEFAULT_MANUAL_AUDIT_EXAMPLES if examples is None else examples
    )
    columns = [
        "symbol",
        "session_date",
        "expected_event_state",
        "manual_note",
        "detected_expected_event",
        "detected_event_timestamps",
        "detected_event_bar_indices",
        "nearest_detected_alternative_event",
        "nearest_alternative_timestamp",
        "nearest_alternative_bar_index",
        "pass_fail",
        "failure_notes",
        "status",
        "detected_session_event_states",
    ]
    rows: list[dict[str, Any]] = []
    data = event_rows.copy()
    if not data.empty:
        data["symbol"] = data["symbol"].astype(str).str.upper()
        data["session_date"] = data["session_date"].astype(str)
    for example in manual_examples:
        symbol = example["symbol"]
        session_date = example["session_date"]
        expected_states = tuple(example["expected_event_states"])
        session_events = data[
            data.get("symbol", pd.Series(dtype=str)).eq(symbol)
            & data.get("session_date", pd.Series(dtype=str)).eq(session_date)
        ] if not data.empty else pd.DataFrame()
        matched = session_events[
            session_events.get("event_state", pd.Series(dtype=str)).isin(expected_states)
        ]
        passed = not matched.empty
        alternative = session_events.sort_values("bar_index_in_session").head(1)
        rows.append(
            {
                "symbol": symbol,
                "session_date": session_date,
                "expected_event_state": "|".join(expected_states),
                "manual_note": example["manual_note"],
                "detected_expected_event": bool(passed),
                "detected_event_timestamps": "|".join(
                    pd.to_datetime(matched["timestamp"], utc=True).astype(str).tolist()
                )
                if passed
                else "",
                "detected_event_bar_indices": "|".join(
                    str(int(value)) for value in matched["bar_index_in_session"].tolist()
                )
                if passed
                else "",
                "nearest_detected_alternative_event": ""
                if alternative.empty
                else str(alternative.iloc[0]["event_state"]),
                "nearest_alternative_timestamp": ""
                if alternative.empty
                else str(pd.Timestamp(alternative.iloc[0]["timestamp"])),
                "nearest_alternative_bar_index": ""
                if alternative.empty
                else int(alternative.iloc[0]["bar_index_in_session"]),
                "pass_fail": "pass" if passed else "fail",
                "failure_notes": ""
                if passed
                else "Expected event state was not detected during this session.",
                "status": "manual_reproduced" if passed else "manual_reproduction_failed",
                "detected_session_event_states": "|".join(
                    session_events["event_state"].astype(str).drop_duplicates().tolist()
                )
                if not session_events.empty
                else "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _mark_manual_audit_matches(
    event_rows: pd.DataFrame,
    examples: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    if event_rows.empty or not examples:
        return event_rows
    data = event_rows.copy()
    data["manual_audit_match"] = False
    for example in _normalize_examples(examples):
        mask = (
            data["symbol"].astype(str).str.upper().eq(example["symbol"])
            & data["session_date"].astype(str).eq(example["session_date"])
            & data["event_state"].isin(example["expected_event_states"])
        )
        data.loc[mask, "manual_audit_match"] = True
    return data


def summarize_event_state_responses(
    event_rows: pd.DataFrame,
    *,
    config: StateEventDetectorConfig,
) -> pd.DataFrame:
    columns = [
        "event_state",
        "horizon",
        "event_count",
        "symbol_count",
        "session_count",
        "median_forward_return",
        "mean_forward_return",
        "win_rate",
        "median_mfe",
        "median_mae",
        "median_abs_return",
        "p25_return",
        "p75_return",
        "single_symbol_share",
        "single_session_share",
        "concentration_warning",
    ]
    if event_rows.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for event_state, state_rows in event_rows.groupby("event_state", sort=True):
        for horizon in config.horizons:
            return_column = f"forward_{horizon}_bar_return"
            mfe_column = f"forward_{horizon}_bar_mfe"
            mae_column = f"forward_{horizon}_bar_mae"
            if return_column not in state_rows:
                continue
            valid = state_rows[state_rows[return_column].notna()]
            if valid.empty:
                continue
            returns = pd.to_numeric(valid[return_column], errors="coerce").dropna()
            if returns.empty:
                continue
            symbol_counts = valid["symbol"].value_counts(normalize=True)
            session_counts = (
                valid["symbol"].astype(str) + "|" + valid["session_date"].astype(str)
            ).value_counts(normalize=True)
            single_symbol_share = float(symbol_counts.iloc[0])
            single_session_share = float(session_counts.iloc[0])
            concentration_warning = (
                single_symbol_share > config.max_single_symbol_share
                or single_session_share > config.max_single_session_share
            )
            rows.append(
                {
                    "event_state": str(event_state),
                    "horizon": int(horizon),
                    "event_count": int(len(returns)),
                    "symbol_count": int(valid["symbol"].nunique()),
                    "session_count": int(
                        (
                            valid["symbol"].astype(str)
                            + "|"
                            + valid["session_date"].astype(str)
                        ).nunique()
                    ),
                    "median_forward_return": float(returns.median()),
                    "mean_forward_return": float(returns.mean()),
                    "win_rate": float((returns > 0.0).mean()),
                    "median_mfe": float(pd.to_numeric(valid[mfe_column], errors="coerce").median()),
                    "median_mae": float(pd.to_numeric(valid[mae_column], errors="coerce").median()),
                    "median_abs_return": float(returns.abs().median()),
                    "p25_return": float(returns.quantile(0.25)),
                    "p75_return": float(returns.quantile(0.75)),
                    "single_symbol_share": single_symbol_share,
                    "single_session_share": single_session_share,
                    "concentration_warning": concentration_warning,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _horizon_event_rows(
    event_rows: pd.DataFrame,
    *,
    config: StateEventDetectorConfig,
) -> pd.DataFrame:
    max_horizon = max(config.horizons, default=0)
    path_columns = [f"path_return_{step}" for step in range(1, max_horizon + 1)]
    columns = [
        "symbol",
        "timestamp",
        "session_date",
        "bar_index_in_session",
        "bar_index_bucket",
        "time_of_day_bucket",
        "primary_state_label",
        "event_state",
        "response_horizon",
        "response_return",
        "response_mfe",
        "response_mae",
        *path_columns,
    ]
    if event_rows.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, event in event_rows.iterrows():
        for horizon in config.horizons:
            return_column = f"forward_{horizon}_bar_return"
            if return_column not in event_rows or pd.isna(event.get(return_column)):
                continue
            output = {
                "symbol": event["symbol"],
                "timestamp": event["timestamp"],
                "session_date": event["session_date"],
                "bar_index_in_session": int(event["bar_index_in_session"]),
                "bar_index_bucket": event.get("bar_index_bucket", ""),
                "time_of_day_bucket": event.get("time_of_day_bucket", ""),
                "primary_state_label": event["event_state"],
                "event_state": event["event_state"],
                "response_horizon": int(horizon),
                "response_return": event.get(return_column),
                "response_mfe": event.get(f"forward_{horizon}_bar_mfe"),
                "response_mae": event.get(f"forward_{horizon}_bar_mae"),
            }
            for column in path_columns:
                output[column] = event.get(column, np.nan)
            rows.append(output)
    return pd.DataFrame(rows, columns=columns)


def _path_vector(row: pd.Series, horizon: int) -> np.ndarray:
    values = [
        row.get(f"path_return_{step}", np.nan)
        for step in range(1, horizon + 1)
    ]
    vector = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return vector


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return math.nan
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return math.nan
    return float(np.dot(left, right) / denom)


def _path_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2 or left.size != right.size:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _match_metrics(source: pd.Series, match: pd.Series, *, baseline: str) -> dict[str, Any]:
    horizon = int(source["response_horizon"])
    source_path = _path_vector(source, horizon)
    match_path = _path_vector(match, horizon)
    source_return = float(source["response_return"])
    match_return = float(match["response_return"])
    return {
        "baseline": baseline,
        "source_event_state": str(source["event_state"]),
        "horizon": horizon,
        "source_symbol": str(source["symbol"]),
        "match_symbol": str(match["symbol"]),
        "source_timestamp": source["timestamp"],
        "match_timestamp": match["timestamp"],
        "match_event_state": str(match["event_state"]),
        "source_response_return": source_return,
        "match_response_return": match_return,
        "response_sign_agreement": bool(np.sign(source_return) == np.sign(match_return)),
        "abs_return_difference": abs(source_return - match_return),
        "cosine_similarity": _cosine_similarity(source_path, match_path),
        "path_correlation": _path_correlation(source_path, match_path),
    }


def _select_pool_row(pool: pd.DataFrame, rng: np.random.Generator) -> pd.Series | None:
    if pool.empty:
        return None
    return pool.iloc[int(rng.integers(0, len(pool)))]


def run_same_event_cross_symbol_similarity(
    horizon_events: pd.DataFrame,
    *,
    config: StateEventDetectorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "baseline",
        "event_state",
        "horizon",
        "source_event_count",
        "match_count",
        "response_sign_agreement",
        "median_abs_return_difference",
        "median_cosine_similarity",
        "median_path_correlation",
        "median_source_return",
        "median_match_return",
    ]
    if horizon_events.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty
    data = horizon_events.dropna(subset=["response_return"]).reset_index(drop=True)
    if len(data) > config.max_similarity_events:
        data = (
            data.sample(n=config.max_similarity_events, random_state=config.random_seed)
            .sort_values(["timestamp", "symbol", "event_state", "response_horizon"])
            .reset_index(drop=True)
        )
    rng = np.random.default_rng(config.random_seed)
    raw_rows: list[dict[str, Any]] = []
    for source_index, source in data.iterrows():
        same_horizon = data["response_horizon"].eq(source["response_horizon"])
        different_symbol = data["symbol"].ne(source["symbol"])
        same_bucket = data["time_of_day_bucket"].eq(source.get("time_of_day_bucket", ""))
        same_state = data["event_state"].eq(source["event_state"])

        same_state_pool = data[same_horizon & different_symbol & same_state & same_bucket]
        if same_state_pool.empty:
            same_state_pool = data[same_horizon & different_symbol & same_state]
        if not same_state_pool.empty:
            chosen = same_state_pool.assign(
                _bar_distance=(
                    pd.to_numeric(same_state_pool["bar_index_in_session"], errors="coerce")
                    - float(source["bar_index_in_session"])
                ).abs()
            ).sort_values(["_bar_distance", "timestamp"]).iloc[0]
            raw_rows.append(_match_metrics(source, chosen, baseline="same_event_cross_symbol"))

        baseline_pools = {
            "random_cross_symbol_same_time_bucket": data[
                same_horizon & different_symbol & same_bucket
            ],
            "different_event_cross_symbol_same_time_bucket": data[
                same_horizon & different_symbol & same_bucket & ~same_state
            ],
            "same_symbol_random_event": data[
                same_horizon & data["symbol"].eq(source["symbol"]) & (data.index != source_index)
            ],
        }
        for baseline, pool in baseline_pools.items():
            chosen = _select_pool_row(pool, rng)
            if chosen is not None:
                raw_rows.append(_match_metrics(source, chosen, baseline=baseline))

    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty
    summary_rows: list[dict[str, Any]] = []
    for (baseline, event_state, horizon), group in raw.groupby(
        ["baseline", "source_event_state", "horizon"],
        sort=True,
    ):
        source_events = group[["source_symbol", "source_timestamp", "horizon"]].drop_duplicates()
        summary_rows.append(
            {
                "baseline": str(baseline),
                "event_state": str(event_state),
                "horizon": int(horizon),
                "source_event_count": int(len(source_events)),
                "match_count": int(len(group)),
                "response_sign_agreement": float(group["response_sign_agreement"].mean()),
                "median_abs_return_difference": float(
                    pd.to_numeric(group["abs_return_difference"], errors="coerce").median()
                ),
                "median_cosine_similarity": float(
                    pd.to_numeric(group["cosine_similarity"], errors="coerce").median()
                ),
                "median_path_correlation": float(
                    pd.to_numeric(group["path_correlation"], errors="coerce").median()
                ),
                "median_source_return": float(
                    pd.to_numeric(group["source_response_return"], errors="coerce").median()
                ),
                "median_match_return": float(
                    pd.to_numeric(group["match_response_return"], errors="coerce").median()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows, columns=columns)
    baselines = summary[summary["baseline"].ne("same_event_cross_symbol")].reset_index(drop=True)
    return summary, baselines


def run_oos_event_response_test(
    horizon_events: pd.DataFrame,
    *,
    config: StateEventDetectorConfig,
) -> pd.DataFrame:
    columns = [
        "split_mode",
        "fold",
        "event_state",
        "horizon",
        "train_event_count",
        "test_event_count",
        "train_symbol_count",
        "test_symbol_count",
        "train_median_return",
        "test_median_return",
        "expected_direction",
        "test_directional_accuracy",
        "generic_directional_accuracy",
        "directional_accuracy_excess_vs_generic",
        "generic_test_median_return",
        "median_return_excess_vs_generic_bps",
        "gate_passed",
        "verdict",
    ]
    if horizon_events.empty:
        return pd.DataFrame(columns=columns)

    def expected_direction(returns: pd.Series) -> int:
        clean = pd.to_numeric(returns, errors="coerce").dropna()
        if clean.empty:
            return 0
        median = float(clean.median())
        if median > 0.0:
            return 1
        if median < 0.0:
            return -1
        return 0

    def directional_accuracy(returns: pd.Series, direction: int) -> float:
        clean = pd.to_numeric(returns, errors="coerce").dropna()
        if clean.empty or direction == 0:
            return math.nan
        return float((np.sign(clean) == direction).mean())

    def run_split(train_mask: pd.Series, *, split_mode: str, fold: str) -> list[dict[str, Any]]:
        data = horizon_events.sort_values("timestamp").reset_index(drop=True)
        train = data[train_mask].copy()
        test = data[~train_mask].copy()
        rows: list[dict[str, Any]] = []
        if train.empty or test.empty:
            return rows
        for (event_state, horizon), state_train in train.groupby(
            ["event_state", "response_horizon"],
            sort=True,
        ):
            state_test = test[
                test["event_state"].eq(event_state) & test["response_horizon"].eq(horizon)
            ]
            if state_test.empty:
                continue
            train_returns = pd.to_numeric(state_train["response_return"], errors="coerce").dropna()
            test_returns = pd.to_numeric(state_test["response_return"], errors="coerce").dropna()
            generic_train = train[train["response_horizon"].eq(horizon)]
            generic_test = test[test["response_horizon"].eq(horizon)]
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
            direction = expected_direction(train_returns)
            generic_direction = expected_direction(generic_train_returns)
            accuracy = directional_accuracy(test_returns, direction)
            generic_accuracy = directional_accuracy(generic_test_returns, generic_direction)
            test_median = float(test_returns.median())
            generic_test_median = (
                float(generic_test_returns.median()) if not generic_test_returns.empty else math.nan
            )
            aligned_state = direction * test_median if direction else 0.0
            aligned_generic = generic_direction * generic_test_median if generic_direction else 0.0
            excess_bps = (aligned_state - aligned_generic) * 10_000
            train_symbol_count = int(state_train["symbol"].nunique())
            test_symbol_count = int(state_test["symbol"].nunique())
            gate_passed = bool(
                len(test_returns) >= config.min_events_for_similarity
                and train_symbol_count >= config.min_symbols_for_key_state
                and test_symbol_count >= 2
                and not math.isnan(accuracy)
                and not math.isnan(generic_accuracy)
                and accuracy > generic_accuracy
                and excess_bps > 0.0
            )
            rows.append(
                {
                    "split_mode": split_mode,
                    "fold": fold,
                    "event_state": str(event_state),
                    "horizon": int(horizon),
                    "train_event_count": int(len(train_returns)),
                    "test_event_count": int(len(test_returns)),
                    "train_symbol_count": train_symbol_count,
                    "test_symbol_count": test_symbol_count,
                    "train_median_return": float(train_returns.median()),
                    "test_median_return": test_median,
                    "expected_direction": int(direction),
                    "test_directional_accuracy": accuracy,
                    "generic_directional_accuracy": generic_accuracy,
                    "directional_accuracy_excess_vs_generic": accuracy - generic_accuracy
                    if not math.isnan(accuracy) and not math.isnan(generic_accuracy)
                    else math.nan,
                    "generic_test_median_return": generic_test_median,
                    "median_return_excess_vs_generic_bps": float(excess_bps),
                    "gate_passed": gate_passed,
                    "verdict": "continue_research" if gate_passed else "mixed_response",
                }
            )
        return rows

    data = horizon_events.sort_values("timestamp").reset_index(drop=True)
    train_count = max(1, min(len(data) - 1, int(len(data) * 0.60)))
    all_rows = run_split(
        pd.Series([index < train_count for index in data.index], index=data.index),
        split_mode="walk_forward",
        fold="time_split_60_40",
    )
    for symbol in sorted(data["symbol"].astype(str).unique().tolist()):
        all_rows.extend(
            run_split(
                data["symbol"].ne(symbol),
                split_mode="leave_one_symbol_out",
                fold=f"holdout_{symbol}",
            )
        )
    return pd.DataFrame(all_rows, columns=columns)


def build_concentration_warnings(
    event_state_summary: pd.DataFrame,
    *,
    config: StateEventDetectorConfig,
) -> pd.DataFrame:
    columns = ["event_state", "horizon", "warning", "severity", "value", "threshold"]
    rows: list[dict[str, Any]] = []
    if event_state_summary.empty:
        rows.append(
            {
                "event_state": "",
                "horizon": "",
                "warning": "no event states detected",
                "severity": "reject",
                "value": 0,
                "threshold": config.min_events_for_similarity,
            }
        )
        return pd.DataFrame(rows, columns=columns)
    for _, row in event_state_summary.iterrows():
        state = str(row["event_state"])
        horizon = int(row["horizon"])
        if int(row["event_count"]) < config.min_events_for_similarity:
            rows.append(
                {
                    "event_state": state,
                    "horizon": horizon,
                    "warning": "low event count",
                    "severity": "warn",
                    "value": int(row["event_count"]),
                    "threshold": config.min_events_for_similarity,
                }
            )
        if int(row["symbol_count"]) < config.min_symbols_for_key_state:
            rows.append(
                {
                    "event_state": state,
                    "horizon": horizon,
                    "warning": "fewer than 3 contributing symbols",
                    "severity": "reject",
                    "value": int(row["symbol_count"]),
                    "threshold": config.min_symbols_for_key_state,
                }
            )
        if float(row["single_symbol_share"]) > config.max_single_symbol_share:
            rows.append(
                {
                    "event_state": state,
                    "horizon": horizon,
                    "warning": "single symbol concentration too high",
                    "severity": "reject",
                    "value": float(row["single_symbol_share"]),
                    "threshold": config.max_single_symbol_share,
                }
            )
        if float(row["single_session_share"]) > config.max_single_session_share:
            rows.append(
                {
                    "event_state": state,
                    "horizon": horizon,
                    "warning": "single session concentration too high",
                    "severity": "reject",
                    "value": float(row["single_session_share"]),
                    "threshold": config.max_single_session_share,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _same_event_similarity_supported_pairs(similarity: pd.DataFrame) -> set[tuple[str, int]]:
    if similarity.empty:
        return set()
    same = similarity[similarity["baseline"].eq("same_event_cross_symbol")]
    random_baseline = similarity[similarity["baseline"].eq("random_cross_symbol_same_time_bucket")]
    different = similarity[
        similarity["baseline"].eq("different_event_cross_symbol_same_time_bucket")
    ]
    supported: set[tuple[str, int]] = set()
    for _, row in same.iterrows():
        state = row["event_state"]
        horizon = row["horizon"]
        random_row = random_baseline[
            random_baseline["event_state"].eq(state) & random_baseline["horizon"].eq(horizon)
        ]
        different_row = different[
            different["event_state"].eq(state) & different["horizon"].eq(horizon)
        ]
        if random_row.empty or different_row.empty:
            continue
        random_first = random_row.iloc[0]
        different_first = different_row.iloc[0]
        if (
            float(row["response_sign_agreement"]) > float(random_first["response_sign_agreement"])
            and float(row["response_sign_agreement"])
            > float(different_first["response_sign_agreement"])
            and float(row["median_abs_return_difference"]) < float(
                random_first["median_abs_return_difference"]
            )
        ):
            supported.add((str(state), int(horizon)))
    return supported


def _same_event_similarity_supported(similarity: pd.DataFrame) -> bool:
    return bool(_same_event_similarity_supported_pairs(similarity))


def build_decision(
    *,
    manual_audit: pd.DataFrame,
    event_state_summary: pd.DataFrame,
    similarity: pd.DataFrame,
    oos_response: pd.DataFrame,
    concentration_warnings: pd.DataFrame,
    config: StateEventDetectorConfig,
) -> dict[str, Any]:
    manual_failed = not manual_audit.empty and not bool(
        manual_audit["detected_expected_event"].all()
    )
    enough_sample_frame = (
        event_state_summary[
            (event_state_summary["event_count"] >= config.min_events_for_similarity)
            & (event_state_summary["symbol_count"] >= config.min_symbols_for_key_state)
        ]
        if not event_state_summary.empty
        else pd.DataFrame()
    )
    enough_pairs = {
        (str(row["event_state"]), int(row["horizon"]))
        for _, row in enough_sample_frame.iterrows()
    }
    same_event_pairs = _same_event_similarity_supported_pairs(similarity)
    oos_pairs = {
        (str(row["event_state"]), int(row["horizon"]))
        for _, row in oos_response.iterrows()
        if bool(row.get("gate_passed", False))
    } if not oos_response.empty else set()
    supported_pairs = enough_pairs & same_event_pairs & oos_pairs
    enough_samples = bool(enough_pairs)
    same_event_supported = bool(same_event_pairs)
    oos_supported = bool(oos_pairs)
    concentrated = bool(
        not concentration_warnings.empty
        and concentration_warnings["severity"].astype(str).eq("reject").any()
    )
    if manual_failed:
        decision = "reject_manual_reproduction_failed"
        reasons = ["manual_reproduction_failed"]
    elif not same_event_supported:
        decision = "reject_no_state_similarity"
        reasons = ["same-event cross-symbol response did not beat random/different-event baselines"]
    elif not oos_supported or not supported_pairs:
        decision = "reject_no_oos_edge"
        reasons = ["held-out response did not survive on the same state/horizon as transfer"]
    elif concentrated:
        decision = "reject_concentrated"
        reasons = ["single symbol/session concentration exceeded thresholds"]
    elif not enough_samples:
        decision = "reject_insufficient_sample"
        reasons = ["no event state has enough events from at least 3 symbols"]
    else:
        decision = "continue_research"
        reasons = ["manual audit mostly passed and transfer/OOS gates survived"]
    return {
        "decision": decision,
        "decision_reasons": reasons,
        "manual_audit_passed": not manual_failed,
        "manual_reproduction_failed": manual_failed,
        "event_states_with_enough_sample": event_state_summary[
            (event_state_summary["event_count"] >= config.min_events_for_similarity)
            & (event_state_summary["symbol_count"] >= config.min_symbols_for_key_state)
        ]["event_state"].drop_duplicates().astype(str).tolist()
        if not event_state_summary.empty
        else [],
        "supported_state_horizons": [
            {"event_state": state, "horizon": horizon}
            for state, horizon in sorted(supported_pairs)
        ],
        "same_event_cross_symbol_supported_pairs": [
            {"event_state": state, "horizon": horizon}
            for state, horizon in sorted(same_event_pairs)
        ],
        "oos_supported_pairs": [
            {"event_state": state, "horizon": horizon}
            for state, horizon in sorted(oos_pairs)
        ],
        "same_event_cross_symbol_supported": same_event_supported,
        "oos_supported": oos_supported,
        "concentrated": concentrated,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, limit: int = 12) -> str:
    if frame.empty:
        return "(empty)"
    display = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _summary_markdown(
    *,
    summary: dict[str, Any],
    manual_audit: pd.DataFrame,
    event_state_summary: pd.DataFrame,
    similarity: pd.DataFrame,
    oos_response: pd.DataFrame,
    concentration_warnings: pd.DataFrame,
) -> str:
    decision = summary["decision"]["decision"]
    manual_status = (
        "manual_reproduced"
        if summary["decision"]["manual_audit_passed"]
        else "manual_reproduction_failed"
    )
    failed_manual = manual_audit[manual_audit["detected_expected_event"].eq(False)]
    enough = event_state_summary[
        event_state_summary["event_state"].isin(summary["decision"]["event_states_with_enough_sample"])
    ] if not event_state_summary.empty else pd.DataFrame()
    same_event = similarity[similarity["baseline"].eq("same_event_cross_symbol")]
    failed_states = event_state_summary[
        ~event_state_summary["event_state"].isin(summary["decision"]["event_states_with_enough_sample"])
    ] if not event_state_summary.empty else pd.DataFrame()
    supported_horizons = pd.DataFrame(summary["decision"].get("supported_state_horizons", []))
    supported_event_states = (
        set(supported_horizons["event_state"].astype(str).tolist())
        if not supported_horizons.empty and "event_state" in supported_horizons
        else set()
    )
    transfer_failures = (
        event_state_summary[
            ~event_state_summary["event_state"].astype(str).isin(supported_event_states)
        ]
        if not event_state_summary.empty
        else pd.DataFrame()
    )
    continuation_text = (
        "Continue research only on the supported event-state/horizon pairs."
        if decision == "continue_research"
        else "Kill or revise this detector path before drawing any broader conclusion."
    )
    failure_interpretation = (
        "Manual reproduction failed, so the detector is not yet testing the manual examples."
        if summary["decision"]["manual_reproduction_failed"]
        else (
            "Manual reproduction passed; any rejection is from transfer, OOS, "
            "or concentration gates."
        )
    )
    failed_manual_columns = [
        "symbol",
        "session_date",
        "expected_event_state",
        "nearest_detected_alternative_event",
        "pass_fail",
        "failure_notes",
    ]
    enough_columns = [
        "event_state",
        "horizon",
        "event_count",
        "symbol_count",
        "session_count",
        "median_forward_return",
        "win_rate",
    ]
    similarity_columns = [
        "event_state",
        "horizon",
        "source_event_count",
        "match_count",
        "response_sign_agreement",
        "median_abs_return_difference",
        "median_cosine_similarity",
    ]
    oos_columns = [
        "split_mode",
        "fold",
        "event_state",
        "horizon",
        "test_event_count",
        "directional_accuracy_excess_vs_generic",
        "median_return_excess_vs_generic_bps",
        "verdict",
    ]
    failed_state_columns = [
        "event_state",
        "horizon",
        "event_count",
        "symbol_count",
        "single_symbol_share",
        "single_session_share",
    ]
    concentration_columns = [
        "event_state",
        "horizon",
        "warning",
        "severity",
        "value",
        "threshold",
    ]
    return f"""# State Event Detector V0

Research-only run. Rejection remains expected. No edge is claimed.

decision: {decision}
manual_audit_status: {manual_status}
total_event_rows: {summary["total_event_rows"]}
symbols_completed: {", ".join(summary["symbols_completed"])}

## Manual Audit

Did the code reproduce the manual examples? {manual_status}

Failed examples:

{_markdown_table(failed_manual, failed_manual_columns)}

## Event States With Enough Sample

{_markdown_table(enough, enough_columns)}

## Same-Event Cross-Stock Transfer

{_markdown_table(same_event, similarity_columns)}

Supported state/horizon pairs:

{_markdown_table(supported_horizons, ["event_state", "horizon"])}

## OOS Response

{_markdown_table(oos_response, oos_columns)}

## Failing Or Thin States

{_markdown_table(failed_states, failed_state_columns)}

## States Without Full Gate Support

{_markdown_table(transfer_failures, failed_state_columns)}

## Continue Or Kill

{continuation_text}

Failure interpretation: {failure_interpretation}

## Concentration Warnings

{_markdown_table(concentration_warnings, concentration_columns)}
"""


def _load_symbol_frame(
    *,
    data_dir: Path,
    source: str,
    instrument_type: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    key = DatasetKey(
        source=source,
        instrument_type=instrument_type,
        symbol=symbol.upper(),
        timeframe=timeframe,
    )
    path = dataset_path(key, data_dir=data_dir)
    if path.exists():
        return read_parquet(path)
    if symbol.upper().endswith(".US"):
        fallback_symbol = symbol.upper().removesuffix(".US")
    else:
        fallback_symbol = f"{symbol.upper()}.US"
    fallback = dataset_path(
        DatasetKey(
            source=source,
            instrument_type=instrument_type,
            symbol=fallback_symbol,
            timeframe=timeframe,
        ),
        data_dir=data_dir,
    )
    if fallback.exists():
        return read_parquet(fallback)
    raise FileNotFoundError(f"Missing local parquet for {symbol}: {path}")


def run_state_event_detector_lab(
    *,
    data_dir: Path,
    symbols: Sequence[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source: str = "eodhd",
    instrument_type: str = "stock",
    timeframe: str = "5m",
    market_calendar: str | None = "XNYS",
    config: StateEventDetectorConfig | None = None,
    manual_examples: Sequence[dict[str, Any]] | None = None,
) -> StateEventDetectorResult:
    cfg = config or StateEventDetectorConfig(timeframe=timeframe, market_calendar=market_calendar)
    run_id = "state_event_detector_v0_" + datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_symbols = [str(symbol).upper() for symbol in symbols]
    event_frames: list[pd.DataFrame] = []
    completed: list[str] = []
    failed: dict[str, str] = {}
    for symbol in requested_symbols:
        try:
            raw = _load_symbol_frame(
                data_dir=data_dir,
                source=source,
                instrument_type=instrument_type,
                symbol=symbol,
                timeframe=cfg.timeframe,
            )
            event_frames.append(detect_state_events(raw, symbol=symbol, config=cfg))
            completed.append(symbol)
        except Exception as exc:  # pragma: no cover - surfaced in report output
            failed[symbol] = str(exc)

    event_rows = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(columns=["symbol", "timestamp", "session_date", "event_state"])
    )
    examples = DEFAULT_MANUAL_AUDIT_EXAMPLES if manual_examples is None else list(manual_examples)
    event_rows = _mark_manual_audit_matches(event_rows, examples)
    manual_audit = audit_manual_state_examples(event_rows, examples=examples)
    event_state_summary = summarize_event_state_responses(event_rows, config=cfg)
    horizon_events = _horizon_event_rows(event_rows, config=cfg)
    similarity, random_baseline = run_same_event_cross_symbol_similarity(
        horizon_events,
        config=cfg,
    )
    oos_response = run_oos_event_response_test(horizon_events, config=cfg)
    concentration_warnings = build_concentration_warnings(event_state_summary, config=cfg)
    decision = build_decision(
        manual_audit=manual_audit,
        event_state_summary=event_state_summary,
        similarity=similarity,
        oos_response=oos_response,
        concentration_warnings=concentration_warnings,
        config=cfg,
    )

    paths = {
        "summary_json": run_dir / "summary.json",
        "summary_md": run_dir / "summary.md",
        "event_rows": run_dir / "event_rows.csv",
        "manual_state_audit": run_dir / "manual_state_audit.csv",
        "event_state_summary": run_dir / "event_state_summary.csv",
        "same_event_cross_symbol_similarity": run_dir / "same_event_cross_symbol_similarity.csv",
        "random_baseline": run_dir / "random_baseline.csv",
        "oos_event_response": run_dir / "oos_event_response.csv",
        "concentration_warnings": run_dir / "concentration_warnings.csv",
        "decision": run_dir / "decision.json",
    }
    summary = {
        "run_id": run_id,
        "output_dir": str(run_dir),
        "config": asdict(cfg),
        "symbols_requested": requested_symbols,
        "symbols_completed": completed,
        "symbols_failed": failed,
        "total_event_rows": int(len(event_rows)),
        "event_counts": event_rows["event_state"].value_counts().to_dict()
        if not event_rows.empty and "event_state" in event_rows
        else {},
        "manual_audit_status": "manual_reproduced"
        if decision["manual_audit_passed"]
        else "manual_reproduction_failed",
        "decision": decision,
        "files": {key: str(value) for key, value in paths.items()},
    }

    _write_csv(paths["event_rows"], event_rows)
    _write_csv(paths["manual_state_audit"], manual_audit)
    _write_csv(paths["event_state_summary"], event_state_summary)
    _write_csv(paths["same_event_cross_symbol_similarity"], similarity)
    _write_csv(paths["random_baseline"], random_baseline)
    _write_csv(paths["oos_event_response"], oos_response)
    _write_csv(paths["concentration_warnings"], concentration_warnings)
    _write_json(paths["summary_json"], summary)
    _write_json(paths["decision"], decision)
    paths["summary_md"].write_text(
        _summary_markdown(
            summary=summary,
            manual_audit=manual_audit,
            event_state_summary=event_state_summary,
            similarity=similarity,
            oos_response=oos_response,
            concentration_warnings=concentration_warnings,
        ),
        encoding="utf-8",
    )

    return StateEventDetectorResult(
        run_id=run_id,
        output_dir=run_dir,
        summary_json_path=paths["summary_json"],
        summary_markdown_path=paths["summary_md"],
        event_rows_csv_path=paths["event_rows"],
        manual_state_audit_csv_path=paths["manual_state_audit"],
        event_state_summary_csv_path=paths["event_state_summary"],
        same_event_cross_symbol_similarity_csv_path=paths["same_event_cross_symbol_similarity"],
        random_baseline_csv_path=paths["random_baseline"],
        oos_event_response_csv_path=paths["oos_event_response"],
        concentration_warnings_csv_path=paths["concentration_warnings"],
        decision_json_path=paths["decision"],
        symbols_requested=requested_symbols,
        symbols_completed=completed,
        symbols_failed=failed,
        total_event_rows=int(len(event_rows)),
        manual_audit_passed=bool(decision["manual_audit_passed"]),
        decision=str(decision["decision"]),
    )


__all__ = [
    "DEFAULT_MANUAL_AUDIT_EXAMPLES",
    "StateEventDetectorConfig",
    "StateEventDetectorResult",
    "audit_manual_state_examples",
    "detect_state_events",
    "run_state_event_detector_lab",
    "run_same_event_cross_symbol_similarity",
    "run_oos_event_response_test",
    "summarize_event_state_responses",
]
