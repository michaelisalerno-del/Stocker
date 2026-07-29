"""Opening-range intraday continuation research template."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stocker_research.intraday_features import (
    IntradayFeatureConfig,
    _timeframe_minutes,
    add_minimum_bars_after_open_flags,
    build_intraday_feature_frame,
)
from stocker_research.templates.base import StrategyTemplate

VALID_EXIT_MODES = {"time_stop", "range_mid_reclaim_fail"}


@dataclass(frozen=True)
class OpeningRangeBreakoutParams:
    """Validated knobs for the opening-range breakout template."""

    opening_minutes: int
    breakout_buffer_bps: float
    min_bars_after_open: int
    max_hold_bars: int
    min_relative_volume: float
    max_opening_range_width_pct: float
    exit_mode: str
    timeframe: str
    market_calendar: str | None
    relative_volume_lookback_sessions: int
    entry_cutoff_before_close_minutes: int
    flatten_before_close_minutes: int
    bars_per_session_context: int


@dataclass(frozen=True)
class OpeningRangeComputation:
    """Internal deterministic signal state."""

    features: pd.DataFrame
    positions: pd.Series
    raw_entries: pd.Series
    time_stop_exits: pd.Series
    range_reclaim_exits: pd.Series
    breakout_threshold: pd.Series


def _optional_market_calendar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _validated_params(params: dict[str, Any]) -> OpeningRangeBreakoutParams:
    opening_minutes = int(params.get("opening_minutes", 30))
    breakout_buffer_bps = float(params.get("breakout_buffer_bps", 0.0))
    min_bars_after_open = int(params.get("min_bars_after_open", 1))
    max_hold_bars = int(params.get("max_hold_bars", 12))
    min_relative_volume = float(params.get("min_relative_volume", 0.0))
    max_width_pct = float(params.get("max_opening_range_width_pct", 0.04))
    exit_mode = str(params.get("exit_mode", "time_stop"))
    timeframe = str(params.get("timeframe", "5m"))
    market_calendar = _optional_market_calendar(params.get("market_calendar", "XNYS"))
    relative_volume_lookback_sessions = int(params.get("relative_volume_lookback_sessions", 20))
    entry_cutoff = int(params.get("entry_cutoff_before_close_minutes", 30))
    flatten_before_close = int(params.get("flatten_before_close_minutes", 10))
    bars_per_session_context = int(params.get("bars_per_session_context", 80))

    if opening_minutes <= 0:
        raise ValueError("opening_minutes must be positive")
    if breakout_buffer_bps < 0:
        raise ValueError("breakout_buffer_bps must be non-negative")
    if min_bars_after_open < 0:
        raise ValueError("min_bars_after_open must be non-negative")
    if max_hold_bars <= 0:
        raise ValueError("max_hold_bars must be positive")
    if min_relative_volume < 0:
        raise ValueError("min_relative_volume must be non-negative")
    if max_width_pct <= 0:
        raise ValueError("max_opening_range_width_pct must be positive")
    if exit_mode not in VALID_EXIT_MODES:
        raise ValueError(f"exit_mode must be one of {sorted(VALID_EXIT_MODES)}")
    if relative_volume_lookback_sessions < 0:
        raise ValueError("relative_volume_lookback_sessions must be non-negative")
    if entry_cutoff < 0:
        raise ValueError("entry_cutoff_before_close_minutes must be non-negative")
    if flatten_before_close < 0:
        raise ValueError("flatten_before_close_minutes must be non-negative")
    if bars_per_session_context <= 0:
        raise ValueError("bars_per_session_context must be positive")

    return OpeningRangeBreakoutParams(
        opening_minutes=opening_minutes,
        breakout_buffer_bps=breakout_buffer_bps,
        min_bars_after_open=min_bars_after_open,
        max_hold_bars=max_hold_bars,
        min_relative_volume=min_relative_volume,
        max_opening_range_width_pct=max_width_pct,
        exit_mode=exit_mode,
        timeframe=timeframe,
        market_calendar=market_calendar,
        relative_volume_lookback_sessions=relative_volume_lookback_sessions,
        entry_cutoff_before_close_minutes=entry_cutoff,
        flatten_before_close_minutes=flatten_before_close,
        bars_per_session_context=bars_per_session_context,
    )


class OpeningRangeBreakoutTemplate(StrategyTemplate):
    """Long-only continuation after an early completed opening-range breakout."""

    name = "opening_range_breakout"

    def __init__(self) -> None:
        self._feature_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    def required_lookback_bars(self, params: dict[str, Any]) -> int:
        cfg = _validated_params(params)
        opening_bars = max(1, -(-cfg.opening_minutes // _timeframe_minutes(cfg.timeframe)))
        relative_volume_context = (
            cfg.relative_volume_lookback_sessions * cfg.bars_per_session_context
        )
        return relative_volume_context + opening_bars + cfg.max_hold_bars + 2

    def _feature_cache_key(
        self,
        frame: pd.DataFrame,
        cfg: OpeningRangeBreakoutParams,
    ) -> tuple[Any, ...]:
        reset = frame.reset_index(drop=True)
        timestamps = (
            pd.to_datetime(reset["timestamp"], utc=True, errors="coerce")
            if "timestamp" in reset
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        numeric_sums = []
        for column in ("open", "high", "low", "close", "volume"):
            if column in reset:
                numeric_sums.append(
                    float(pd.to_numeric(reset[column], errors="coerce").fillna(0.0).sum())
                )
            else:
                numeric_sums.append(0.0)
        first_timestamp = None if timestamps.empty else str(timestamps.iloc[0])
        last_timestamp = None if timestamps.empty else str(timestamps.iloc[-1])
        return (
            len(reset),
            first_timestamp,
            last_timestamp,
            *numeric_sums,
            cfg.opening_minutes,
            cfg.timeframe,
            cfg.market_calendar,
            cfg.relative_volume_lookback_sessions,
            cfg.entry_cutoff_before_close_minutes,
            cfg.flatten_before_close_minutes,
            cfg.min_bars_after_open,
        )

    def _feature_frame(
        self,
        frame: pd.DataFrame,
        cfg: OpeningRangeBreakoutParams,
    ) -> pd.DataFrame:
        cache_key = self._feature_cache_key(frame, cfg)
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        features = build_intraday_feature_frame(
            frame,
            IntradayFeatureConfig(
                timeframe=cfg.timeframe,
                market_calendar=cfg.market_calendar,
                opening_minutes=cfg.opening_minutes,
                open_buffer_minutes=0,
                entry_cutoff_before_close_minutes=cfg.entry_cutoff_before_close_minutes,
                flatten_before_close_minutes=cfg.flatten_before_close_minutes,
                relative_volume_lookback_sessions=cfg.relative_volume_lookback_sessions,
            ),
        )
        features = add_minimum_bars_after_open_flags(features, cfg.min_bars_after_open)
        self._feature_cache[cache_key] = features.copy()
        return features.copy()

    def _compute(
        self,
        frame: pd.DataFrame,
        params: dict[str, Any],
    ) -> OpeningRangeComputation:
        cfg = _validated_params(params)
        features = self._feature_frame(frame, cfg).reset_index(drop=True)
        close = pd.to_numeric(features["close"], errors="coerce")
        previous_close = close.groupby(features["session_date"]).shift(1)
        breakout_threshold = pd.to_numeric(
            features["opening_range_high"],
            errors="coerce",
        ) * (1.0 + cfg.breakout_buffer_bps / 10_000.0)
        opening_mid = pd.to_numeric(features["opening_range_mid"], errors="coerce")
        opening_width = pd.to_numeric(features["opening_range_width"], errors="coerce")
        opening_width_pct = opening_width / opening_mid

        breakout_cross = (close > breakout_threshold) & (previous_close <= breakout_threshold)
        range_width_ok = opening_width_pct.le(cfg.max_opening_range_width_pct).fillna(False)
        relative_volume = pd.to_numeric(
            features.get("relative_volume_at_bar_index", pd.Series(index=features.index)),
            errors="coerce",
        )
        if cfg.min_relative_volume > 0:
            relative_volume_ok = relative_volume.ge(cfg.min_relative_volume).fillna(False)
        else:
            relative_volume_ok = pd.Series(True, index=features.index)

        can_open = (
            features["opening_range_complete"].astype(bool)
            & features["can_open_new_position"].astype(bool)
            & features["can_enter_after_minimum_bars"].astype(bool)
            & breakout_cross.fillna(False)
            & range_width_ok
            & relative_volume_ok
        )

        positions = pd.Series(0.0, index=features.index)
        raw_entries = pd.Series(False, index=features.index)
        time_stop_exits = pd.Series(False, index=features.index)
        range_reclaim_exits = pd.Series(False, index=features.index)

        for _, group_index in features.groupby("session_date", sort=False).groups.items():
            indices = [int(index) for index in group_index]
            in_position = False
            bars_held = 0
            for index in indices:
                if in_position:
                    if bars_held >= cfg.max_hold_bars:
                        time_stop_exits.iloc[index] = True
                        in_position = False
                        bars_held = 0
                        continue
                    reclaim_failed = (
                        cfg.exit_mode == "range_mid_reclaim_fail"
                        and close.iloc[index] < opening_mid.iloc[index]
                    )
                    if bool(reclaim_failed):
                        range_reclaim_exits.iloc[index] = True
                        in_position = False
                        bars_held = 0
                        continue
                    positions.iloc[index] = 1.0
                    bars_held += 1
                    continue

                if bool(can_open.iloc[index]):
                    raw_entries.iloc[index] = True
                    positions.iloc[index] = 1.0
                    in_position = True
                    bars_held = 1

        return OpeningRangeComputation(
            features=features,
            positions=positions.reset_index(drop=True),
            raw_entries=raw_entries.reset_index(drop=True),
            time_stop_exits=time_stop_exits.reset_index(drop=True),
            range_reclaim_exits=range_reclaim_exits.reset_index(drop=True),
            breakout_threshold=breakout_threshold.reset_index(drop=True),
        )

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        return self._compute(frame, params).positions

    def generate_signals(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        computed = self._compute(frame, params)
        positions = computed.positions.astype(float).reset_index(drop=True)
        signal = positions.diff().fillna(positions).clip(lower=-1.0, upper=1.0)
        features = computed.features.reset_index(drop=True)
        return pd.DataFrame(
            {
                "timestamp": features["timestamp"],
                "signal": signal,
                "target_position": positions,
                "entry": signal > 0,
                "exit": signal < 0,
                "template_name": self.name,
                "parameter_set_id": str(params.get("parameter_set_id", "unknown")),
                "opening_range_complete": features["opening_range_complete"].astype(bool),
                "opening_range_high": features["opening_range_high"],
                "opening_range_low": features["opening_range_low"],
                "opening_range_mid": features["opening_range_mid"],
                "breakout_threshold": computed.breakout_threshold,
                "raw_entry": computed.raw_entries,
                "time_stop_exit": computed.time_stop_exits,
                "range_reclaim_exit": computed.range_reclaim_exits,
            }
        )
