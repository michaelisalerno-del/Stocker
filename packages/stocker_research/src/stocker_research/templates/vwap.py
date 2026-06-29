"""VWAP reclaim and rejection intraday research template."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stocker_research.intraday_features import (
    IntradayFeatureConfig,
    add_minimum_bars_after_open_flags,
    build_intraday_feature_frame,
)
from stocker_research.templates.base import StrategyTemplate

VALID_ENTRY_MODES = {"reclaim", "rejection", "both"}
VALID_EXIT_MODES = {"time_stop", "vwap_lost"}


@dataclass(frozen=True)
class VWAPReclaimRejectionParams:
    """Validated knobs for the VWAP reclaim/rejection template."""

    entry_mode: str
    entry_buffer_bps: float
    min_reclaim_distance_bps: float
    reclaim_lookback_bars: int
    rejection_lookback_bars: int
    max_reclaim_distance_from_vwap_pct: float
    max_rejection_distance_from_vwap_pct: float
    min_bounce_distance_bps: float
    min_bars_after_open: int
    max_hold_bars: int
    min_relative_volume: float
    exit_mode: str
    timeframe: str
    market_calendar: str | None
    relative_volume_lookback_sessions: int
    open_buffer_minutes: int
    entry_cutoff_before_close_minutes: int
    flatten_before_close_minutes: int
    bars_per_session_context: int


@dataclass(frozen=True)
class VWAPReclaimRejectionComputation:
    """Internal deterministic signal state."""

    features: pd.DataFrame
    positions: pd.Series
    entry_modes: pd.Series
    raw_entries: pd.Series
    time_stop_exits: pd.Series
    vwap_lost_exits: pd.Series
    reclaim_recent_below_vwap: pd.Series
    reclaim_cross_above_vwap: pd.Series
    reclaim_distance_ok: pd.Series
    rejection_valid_test: pd.Series
    rejection_bounce: pd.Series
    rejection_distance_ok: pd.Series
    relative_volume_ok: pd.Series
    distance_filter_ok: pd.Series


def _optional_market_calendar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _validated_params(params: dict[str, Any]) -> VWAPReclaimRejectionParams:
    entry_mode = str(params.get("entry_mode", "reclaim"))
    entry_buffer_bps = float(params.get("entry_buffer_bps", 0.0))
    min_reclaim_distance_bps = float(params.get("min_reclaim_distance_bps", 0.0))
    reclaim_lookback_bars = int(params.get("reclaim_lookback_bars", 3))
    rejection_lookback_bars = int(params.get("rejection_lookback_bars", 3))
    max_reclaim_distance = float(params.get("max_reclaim_distance_from_vwap_pct", 0.10))
    max_rejection_distance = float(params.get("max_rejection_distance_from_vwap_pct", 0.02))
    min_bounce_distance_bps = float(params.get("min_bounce_distance_bps", 0.0))
    min_bars_after_open = int(params.get("min_bars_after_open", 1))
    max_hold_bars = int(params.get("max_hold_bars", 12))
    min_relative_volume = float(params.get("min_relative_volume", 0.0))
    exit_mode = str(params.get("exit_mode", "time_stop"))
    timeframe = str(params.get("timeframe", "5m"))
    market_calendar = _optional_market_calendar(params.get("market_calendar", "XNYS"))
    relative_volume_lookback_sessions = int(params.get("relative_volume_lookback_sessions", 20))
    open_buffer_minutes = int(params.get("open_buffer_minutes", 0))
    entry_cutoff = int(params.get("entry_cutoff_before_close_minutes", 30))
    flatten_before_close = int(params.get("flatten_before_close_minutes", 10))
    bars_per_session_context = int(params.get("bars_per_session_context", 80))

    if entry_mode not in VALID_ENTRY_MODES:
        raise ValueError(f"entry_mode must be one of {sorted(VALID_ENTRY_MODES)}")
    if entry_buffer_bps < 0:
        raise ValueError("entry_buffer_bps must be non-negative")
    if min_reclaim_distance_bps < 0:
        raise ValueError("min_reclaim_distance_bps must be non-negative")
    if reclaim_lookback_bars <= 0:
        raise ValueError("reclaim_lookback_bars must be positive")
    if rejection_lookback_bars <= 0:
        raise ValueError("rejection_lookback_bars must be positive")
    if max_reclaim_distance < 0:
        raise ValueError("max_reclaim_distance_from_vwap_pct must be non-negative")
    if max_rejection_distance < 0:
        raise ValueError("max_rejection_distance_from_vwap_pct must be non-negative")
    if min_bounce_distance_bps < 0:
        raise ValueError("min_bounce_distance_bps must be non-negative")
    if min_bars_after_open < 0:
        raise ValueError("min_bars_after_open must be non-negative")
    if max_hold_bars <= 0:
        raise ValueError("max_hold_bars must be positive")
    if min_relative_volume < 0:
        raise ValueError("min_relative_volume must be non-negative")
    if exit_mode not in VALID_EXIT_MODES:
        raise ValueError(f"exit_mode must be one of {sorted(VALID_EXIT_MODES)}")
    if relative_volume_lookback_sessions < 0:
        raise ValueError("relative_volume_lookback_sessions must be non-negative")
    if open_buffer_minutes < 0:
        raise ValueError("open_buffer_minutes must be non-negative")
    if entry_cutoff < 0:
        raise ValueError("entry_cutoff_before_close_minutes must be non-negative")
    if flatten_before_close < 0:
        raise ValueError("flatten_before_close_minutes must be non-negative")
    if bars_per_session_context <= 0:
        raise ValueError("bars_per_session_context must be positive")

    return VWAPReclaimRejectionParams(
        entry_mode=entry_mode,
        entry_buffer_bps=entry_buffer_bps,
        min_reclaim_distance_bps=min_reclaim_distance_bps,
        reclaim_lookback_bars=reclaim_lookback_bars,
        rejection_lookback_bars=rejection_lookback_bars,
        max_reclaim_distance_from_vwap_pct=max_reclaim_distance,
        max_rejection_distance_from_vwap_pct=max_rejection_distance,
        min_bounce_distance_bps=min_bounce_distance_bps,
        min_bars_after_open=min_bars_after_open,
        max_hold_bars=max_hold_bars,
        min_relative_volume=min_relative_volume,
        exit_mode=exit_mode,
        timeframe=timeframe,
        market_calendar=market_calendar,
        relative_volume_lookback_sessions=relative_volume_lookback_sessions,
        open_buffer_minutes=open_buffer_minutes,
        entry_cutoff_before_close_minutes=entry_cutoff,
        flatten_before_close_minutes=flatten_before_close,
        bars_per_session_context=bars_per_session_context,
    )


class VWAPReclaimRejectionTemplate(StrategyTemplate):
    """Long-only VWAP reclaim and VWAP rejection continuation template."""

    name = "vwap_reclaim_rejection"

    def __init__(self) -> None:
        self._feature_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    def required_lookback_bars(self, params: dict[str, Any]) -> int:
        cfg = _validated_params(params)
        relative_volume_context = (
            cfg.relative_volume_lookback_sessions * cfg.bars_per_session_context
        )
        signal_context = max(cfg.reclaim_lookback_bars, cfg.rejection_lookback_bars)
        return relative_volume_context + signal_context + cfg.max_hold_bars + 2

    def _feature_cache_key(
        self,
        frame: pd.DataFrame,
        cfg: VWAPReclaimRejectionParams,
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
            cfg.timeframe,
            cfg.market_calendar,
            cfg.relative_volume_lookback_sessions,
            cfg.open_buffer_minutes,
            cfg.entry_cutoff_before_close_minutes,
            cfg.flatten_before_close_minutes,
            cfg.min_bars_after_open,
        )

    def _feature_frame(
        self,
        frame: pd.DataFrame,
        cfg: VWAPReclaimRejectionParams,
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
                open_buffer_minutes=cfg.open_buffer_minutes,
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
    ) -> VWAPReclaimRejectionComputation:
        cfg = _validated_params(params)
        features = self._feature_frame(frame, cfg).reset_index(drop=True)
        close = pd.to_numeric(features["close"], errors="coerce")
        low = pd.to_numeric(features["low"], errors="coerce")
        vwap = pd.to_numeric(features["session_vwap"], errors="coerce")
        distance = pd.to_numeric(features["distance_from_vwap"], errors="coerce")
        previous_close = close.groupby(features["session_date"]).shift(1)
        previous_vwap = vwap.groupby(features["session_date"]).shift(1)
        previous_distance = distance.groupby(features["session_date"]).shift(1)

        entry_threshold = vwap * (1.0 + cfg.entry_buffer_bps / 10_000.0)
        previous_entry_threshold = previous_vwap * (1.0 + cfg.entry_buffer_bps / 10_000.0)
        minimum_reclaim_distance = cfg.min_reclaim_distance_bps / 10_000.0
        minimum_bounce_distance = cfg.min_bounce_distance_bps / 10_000.0

        below_vwap_previous = previous_distance < -minimum_reclaim_distance
        reclaim_recent_below = (
            below_vwap_previous.groupby(features["session_date"])
            .rolling(cfg.reclaim_lookback_bars, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .astype(bool)
        )
        recent_abs_distance = (
            previous_distance.abs()
            .groupby(features["session_date"])
            .rolling(cfg.reclaim_lookback_bars, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
        reclaim_distance_ok = recent_abs_distance.le(
            cfg.max_reclaim_distance_from_vwap_pct
        ).fillna(False)
        reclaim_cross = (
            (close > entry_threshold)
            & (previous_close <= previous_entry_threshold)
            & reclaim_recent_below
            & reclaim_distance_ok
        ).fillna(False)

        previous_low = low.groupby(features["session_date"]).shift(1)
        previous_test_distance = (previous_low / previous_vwap - 1.0).abs()
        previous_close_distance = previous_close / previous_vwap - 1.0
        previous_valid_test = (
            (previous_close >= previous_vwap)
            & (previous_close_distance > 0.0)
            & (previous_low <= previous_vwap * (1.0 + cfg.max_rejection_distance_from_vwap_pct))
            & (previous_test_distance <= cfg.max_rejection_distance_from_vwap_pct)
        ).fillna(False)
        rejection_valid_test = (
            previous_valid_test.groupby(features["session_date"])
            .rolling(cfg.rejection_lookback_bars, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .astype(bool)
        )
        recent_vwap_lost = (
            (previous_distance < 0.0)
            .groupby(features["session_date"])
            .rolling(cfg.rejection_lookback_bars, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
            .astype(bool)
        )
        rejection_valid_test = rejection_valid_test & ~recent_vwap_lost
        rejection_bounce = (
            (close > previous_close)
            & (close >= vwap * (1.0 + minimum_bounce_distance))
            & (close > entry_threshold)
        ).fillna(False)
        rejection_distance_ok = distance.abs().le(
            cfg.max_rejection_distance_from_vwap_pct
        ).fillna(False)
        rejection_entry = (rejection_valid_test & rejection_bounce & rejection_distance_ok).fillna(
            False
        )

        relative_volume = pd.to_numeric(
            features.get("relative_volume_at_bar_index", pd.Series(index=features.index)),
            errors="coerce",
        )
        if cfg.min_relative_volume > 0:
            relative_volume_ok = relative_volume.ge(cfg.min_relative_volume).fillna(False)
        else:
            relative_volume_ok = pd.Series(True, index=features.index)

        allow_reclaim = cfg.entry_mode in {"reclaim", "both"}
        allow_rejection = cfg.entry_mode in {"rejection", "both"}
        raw_reclaim = reclaim_cross if allow_reclaim else pd.Series(False, index=features.index)
        raw_rejection = (
            rejection_entry if allow_rejection else pd.Series(False, index=features.index)
        )
        can_open = (
            features["can_open_new_position"].astype(bool)
            & features["can_enter_after_minimum_bars"].astype(bool)
            & relative_volume_ok
            & (raw_reclaim | raw_rejection)
        )
        distance_filter_ok = (
            (raw_reclaim & reclaim_distance_ok) | (raw_rejection & rejection_distance_ok)
        ).fillna(False)

        positions = pd.Series(0.0, index=features.index)
        entry_modes = pd.Series("", index=features.index, dtype=object)
        raw_entries = pd.Series(False, index=features.index)
        time_stop_exits = pd.Series(False, index=features.index)
        vwap_lost_exits = pd.Series(False, index=features.index)

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
                    vwap_lost = (
                        cfg.exit_mode == "vwap_lost" and close.iloc[index] < vwap.iloc[index]
                    )
                    if bool(vwap_lost):
                        vwap_lost_exits.iloc[index] = True
                        in_position = False
                        bars_held = 0
                        continue
                    positions.iloc[index] = 1.0
                    bars_held += 1
                    continue

                if bool(can_open.iloc[index]):
                    raw_entries.iloc[index] = True
                    positions.iloc[index] = 1.0
                    entry_modes.iloc[index] = (
                        "reclaim" if bool(raw_reclaim.iloc[index]) else "rejection"
                    )
                    in_position = True
                    bars_held = 1

        return VWAPReclaimRejectionComputation(
            features=features,
            positions=positions.reset_index(drop=True),
            entry_modes=entry_modes.reset_index(drop=True),
            raw_entries=raw_entries.reset_index(drop=True),
            time_stop_exits=time_stop_exits.reset_index(drop=True),
            vwap_lost_exits=vwap_lost_exits.reset_index(drop=True),
            reclaim_recent_below_vwap=reclaim_recent_below.reset_index(drop=True),
            reclaim_cross_above_vwap=reclaim_cross.reset_index(drop=True),
            reclaim_distance_ok=reclaim_distance_ok.reset_index(drop=True),
            rejection_valid_test=rejection_valid_test.reset_index(drop=True),
            rejection_bounce=rejection_bounce.reset_index(drop=True),
            rejection_distance_ok=rejection_distance_ok.reset_index(drop=True),
            relative_volume_ok=relative_volume_ok.reset_index(drop=True),
            distance_filter_ok=distance_filter_ok.reset_index(drop=True),
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
                "session_vwap": features["session_vwap"],
                "distance_from_vwap": features["distance_from_vwap"],
                "entry_mode": computed.entry_modes,
                "raw_entry": computed.raw_entries,
                "time_stop_exit": computed.time_stop_exits,
                "vwap_lost_exit": computed.vwap_lost_exits,
                "reclaim_recent_below_vwap": computed.reclaim_recent_below_vwap,
                "reclaim_cross_above_vwap": computed.reclaim_cross_above_vwap,
                "reclaim_distance_ok": computed.reclaim_distance_ok,
                "rejection_valid_test": computed.rejection_valid_test,
                "rejection_bounce": computed.rejection_bounce,
                "rejection_distance_ok": computed.rejection_distance_ok,
                "relative_volume_ok": computed.relative_volume_ok,
                "distance_filter_ok": computed.distance_filter_ok,
            }
        )
