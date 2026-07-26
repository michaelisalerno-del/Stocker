"""Causal primitives for Movement-Qualified Directional Readiness V0.

The frozen M1 movement score is only an eligibility gate.  Every direction
feature and model in this module is a separate second-stage construction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

M1_THRESHOLD: Final[float] = 0.49588519865576763
HORIZONS: Final[tuple[int, ...]] = (5, 10, 15, 30)
ANNUAL_TRADING_MINUTES: Final[int] = 252 * 390
D0_FEATURES: Final[tuple[str, ...]] = (
    "signed_return_1bar",
    "signed_return_2bar",
    "signed_return_3bar",
    "signed_return_6bar",
    "short_return_slope",
    "return_acceleration",
    "distance_from_session_open",
    "distance_from_vwap",
    "distance_from_opening_range_midpoint",
    "distance_from_opening_range_high",
    "distance_from_opening_range_low",
    "distance_from_recent_high",
    "distance_from_recent_low",
    "recent_breakout_direction",
    "recent_rejection_direction",
    "current_candle_body_direction",
    "wick_imbalance",
    "market_return_1bar",
    "market_return_2bar",
    "market_return_3bar",
    "stock_minus_market_return_3bar",
    "market_breadth_direction",
)
D1_FEATURES: Final[tuple[str, ...]] = (
    "signed_pressure",
    "signed_pressure_change",
    "signed_exhaustion",
    "signed_exhaustion_change",
    "compression_release_direction",
    "pressure_x_conviction",
    "pressure_x_tension",
    "pressure_x_arousal",
    "signed_activity_imbalance",
    "signed_structural_memory",
)
D2_FEATURES: Final[tuple[str, ...]] = (
    "positive_active_prefix_count",
    "negative_active_prefix_count",
    "neutral_active_prefix_count",
    "depth_weighted_positive_orientation",
    "depth_weighted_negative_orientation",
    "top_route_orientation",
    "second_route_orientation",
    "orientation_margin",
    "orientation_agreement",
    "orientation_disagreement",
    "narrowing_route_orientation",
    "recent_completed_loop_orientation",
    "recent_same_orientation_loop_memory_score",
    "dominant_route_pressure_agreement",
)


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(cast(Any, value))


def movement_gate(
    probabilities: Sequence[float] | FloatArray,
    *,
    threshold: float = M1_THRESHOLD,
) -> BoolArray:
    """Apply the exact frozen M1 threshold without rank forcing."""

    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("M1 probabilities must be finite")
    if threshold != M1_THRESHOLD:
        raise ValueError("the frozen M1 threshold cannot be changed")
    return np.asarray(values >= threshold, dtype=np.bool_)


def construct_fresh_episodes(
    checkpoint_rows: pd.DataFrame,
    *,
    minimum_spacing_minutes: int = 30,
) -> pd.DataFrame:
    """Convert frozen-gate crossings into one decision per fresh episode."""

    required = {
        "stock",
        "session",
        "checkpoint",
        "signal_timestamp",
        "prospective_entry_timestamp",
        "m1_probability",
        "partition",
    }
    missing = sorted(required.difference(checkpoint_rows.columns))
    if missing:
        raise ValueError(f"episode inputs missing: {missing}")
    if minimum_spacing_minutes != 30:
        raise ValueError("episode spacing is frozen at thirty minutes")

    ordered = checkpoint_rows.copy()
    ordered["signal_timestamp"] = pd.to_datetime(
        ordered["signal_timestamp"], utc=True, errors="raise"
    )
    ordered["prospective_entry_timestamp"] = pd.to_datetime(
        ordered["prospective_entry_timestamp"], utc=True, errors="raise"
    )
    ordered = ordered.sort_values(["stock", "session", "checkpoint"], kind="mergesort").reset_index(
        drop=True
    )
    if ordered.duplicated(["stock", "session", "checkpoint"]).any():
        raise ValueError("checkpoint identity must be unique")

    probabilities = pd.to_numeric(ordered["m1_probability"], errors="raise").to_numpy(float)
    ordered["above_frozen_threshold"] = movement_gate(probabilities)
    ordered["previous_checkpoint_probability"] = ordered.groupby(["stock", "session"], sort=False)[
        "m1_probability"
    ].shift()
    ordered["fresh_crossing"] = ordered["above_frozen_threshold"] & (
        ordered["previous_checkpoint_probability"].isna()
        | ordered["previous_checkpoint_probability"].lt(M1_THRESHOLD)
    )

    selected_indices: list[int] = []
    episode_numbers: dict[int, int] = {}
    minutes_since: dict[int, float] = {}
    crossings = ordered.loc[ordered["fresh_crossing"]]
    for _, group in crossings.groupby(["stock", "session"], sort=True):
        previous_start: pd.Timestamp | None = None
        number = 0
        for index, row in group.iterrows():
            current_start = _as_timestamp(row["signal_timestamp"])
            elapsed = (
                None
                if previous_start is None
                else (current_start - previous_start).total_seconds() / 60.0
            )
            if elapsed is not None and elapsed < minimum_spacing_minutes:
                continue
            number += 1
            index_value = _as_int(index)
            selected_indices.append(index_value)
            episode_numbers[index_value] = number
            minutes_since[index_value] = np.nan if elapsed is None else float(elapsed)
            previous_start = current_start

    episodes = ordered.loc[selected_indices].copy()
    episodes["episode_number"] = [episode_numbers[int(index)] for index in selected_indices]
    episodes["minutes_since_previous_episode"] = [
        minutes_since[int(index)] for index in selected_indices
    ]
    columns = [
        "stock",
        "session",
        "checkpoint",
        "signal_timestamp",
        "prospective_entry_timestamp",
        "m1_probability",
        "previous_checkpoint_probability",
        "episode_number",
        "minutes_since_previous_episode",
        "partition",
    ]
    return episodes.loc[:, columns].reset_index(drop=True)


def attach_direction_targets(
    episodes: pd.DataFrame,
    completed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Attach next-bar entry, signed horizons, excursions, and remaining move."""

    episode_required = {
        "stock",
        "session",
        "checkpoint",
        "signal_timestamp",
        "prospective_entry_timestamp",
    }
    bar_required = {
        "stock",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
    }
    missing_episode = sorted(episode_required.difference(episodes.columns))
    missing_bar = sorted(bar_required.difference(completed_bars.columns))
    if missing_episode or missing_bar:
        raise ValueError(f"target inputs missing: episodes={missing_episode}, bars={missing_bar}")

    bars = completed_bars.copy()
    bars["bar_start_timestamp"] = pd.to_datetime(
        bars["bar_start_timestamp"], utc=True, errors="raise"
    )
    bars["bar_complete_timestamp"] = pd.to_datetime(
        bars["bar_complete_timestamp"], utc=True, errors="raise"
    )
    bars = bars.sort_values(["stock", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )
    if bars.duplicated(["stock", "session", "bar_ordinal"]).any():
        raise ValueError("completed-bar identity must be unique")
    bar_index = bars.set_index(["stock", "session", "bar_ordinal"])

    output = episodes.copy()
    output["signal_timestamp"] = pd.to_datetime(
        output["signal_timestamp"], utc=True, errors="raise"
    )
    output["prospective_entry_timestamp"] = pd.to_datetime(
        output["prospective_entry_timestamp"], utc=True, errors="raise"
    )
    attached_rows: list[dict[str, object]] = []
    for episode in output.itertuples(index=False):
        stock = str(episode.stock)
        session = str(episode.session)
        checkpoint = _as_int(episode.checkpoint)

        def bar_at(
            ordinal: int,
            *,
            stock_key: str = stock,
            session_key: str = session,
        ) -> pd.Series:
            try:
                indexer = cast(Any, bar_index.loc)
                row = indexer[(stock_key, session_key, ordinal)]
            except KeyError as error:
                raise ValueError(
                    f"missing completed bar for {stock_key}|{session_key}|{ordinal}"
                ) from error
            if isinstance(row, pd.DataFrame):
                raise ValueError("completed-bar identity must be unique")
            return cast(pd.Series, row)

        signal_bar = bar_at(checkpoint - 1)
        entry_bar = bar_at(checkpoint)
        session_open_bar = bar_at(0)
        signal_timestamp = _as_timestamp(episode.signal_timestamp)
        entry_timestamp = _as_timestamp(episode.prospective_entry_timestamp)
        if signal_timestamp != pd.Timestamp(signal_bar["bar_complete_timestamp"]):
            raise ValueError("signal timestamp is not the completed checkpoint-bar close")
        if entry_timestamp != pd.Timestamp(entry_bar["bar_start_timestamp"]):
            raise ValueError("prospective entry is not the next completed-bar open")
        if entry_timestamp < signal_timestamp:
            raise ValueError("prospective entry precedes the signal")

        entry_price = float(entry_bar["open"])
        signal_close = float(signal_bar["close"])
        session_open = float(session_open_bar["open"])
        if min(entry_price, signal_close, session_open) <= 0.0:
            raise ValueError("direction target prices must be positive")
        values: dict[str, object] = {
            "session_open_price": session_open,
            "signal_open": float(signal_bar["open"]),
            "signal_high": float(signal_bar["high"]),
            "signal_low": float(signal_bar["low"]),
            "signal_close": signal_close,
            "entry_price": entry_price,
            "return_realised_before_signal": math.log(signal_close / session_open),
            "return_signal_to_entry": math.log(entry_price / signal_close),
        }
        for horizon in HORIZONS:
            bars_forward = horizon // 5
            horizon_bar = bar_at(checkpoint + bars_forward - 1)
            horizon_close = float(horizon_bar["close"])
            if horizon_close <= 0.0:
                raise ValueError("direction horizon close must be positive")
            signed = math.log(horizon_close / entry_price)
            values[f"close_{horizon}m"] = horizon_close
            values[f"signed_log_return_{horizon}m"] = signed
            values[f"absolute_log_return_{horizon}m"] = abs(signed)
            values[f"return_after_entry_{horizon}m"] = signed
            if "atm_iv" in output.columns:
                atm_iv = _as_float(episode.atm_iv)
                expectation = (
                    atm_iv * math.sqrt(horizon / ANNUAL_TRADING_MINUTES) * math.sqrt(2.0 / math.pi)
                    if math.isfinite(atm_iv) and atm_iv > 0.0
                    else math.nan
                )
            else:
                expectation = math.nan
            values[f"iv_expected_absolute_{horizon}m"] = expectation
            values[f"realised_iv_excess_{horizon}m"] = (
                bool(abs(signed) > expectation) if math.isfinite(expectation) else False
            )

            path = [bar_at(checkpoint + offset) for offset in range(bars_forward)]
            highs = np.asarray([float(row["high"]) for row in path], dtype=float)
            lows = np.asarray([float(row["low"]) for row in path], dtype=float)
            up_moves = np.log(highs / entry_price)
            down_moves = np.log(entry_price / lows)
            values[f"upside_mfe_{horizon}m"] = float(max(0.0, np.max(up_moves)))
            values[f"upside_mae_{horizon}m"] = float(max(0.0, np.max(down_moves)))
            values[f"downside_mfe_{horizon}m"] = float(max(0.0, np.max(down_moves)))
            values[f"downside_mae_{horizon}m"] = float(max(0.0, np.max(up_moves)))
            values[f"time_of_upside_mfe_{horizon}m"] = (int(np.argmax(up_moves)) + 1) * 5
            values[f"time_of_upside_mae_{horizon}m"] = (int(np.argmax(down_moves)) + 1) * 5

        primary = _as_float(values["signed_log_return_10m"])
        values["zero_return_10m"] = int(primary == 0.0)
        values["direction_up_10m"] = np.nan if primary == 0.0 else int(primary > 0.0)
        before = abs(_as_float(values["return_realised_before_signal"]))
        gap = abs(_as_float(values["return_signal_to_entry"]))
        for horizon in (10, 30):
            after = abs(_as_float(values[f"return_after_entry_{horizon}m"]))
            denominator = before + gap + after
            values[f"fraction_eventual_{horizon}m_move_after_entry"] = (
                after / denominator if denominator > 0.0 else np.nan
            )
        attached_rows.append(values)

    attached = pd.DataFrame(attached_rows, index=output.index)
    return pd.concat([output, attached], axis=1)


def build_d0_features(
    episodes: pd.DataFrame,
    completed_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Build the fixed causal price/market baseline from completed bars only."""

    episode_required = {"stock", "session", "checkpoint", "signal_timestamp"}
    bar_required = {
        "stock",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "open",
        "high",
        "low",
        "close",
    }
    missing_episode = sorted(episode_required.difference(episodes.columns))
    missing_bar = sorted(bar_required.difference(completed_bars.columns))
    if missing_episode or missing_bar:
        raise ValueError(f"D0 inputs missing: episodes={missing_episode}, bars={missing_bar}")
    bars = completed_bars.copy()
    bars["bar_complete_timestamp"] = pd.to_datetime(
        bars["bar_complete_timestamp"], utc=True, errors="raise"
    )
    bars = bars.sort_values(["stock", "session", "bar_ordinal"], kind="mergesort").reset_index(
        drop=True
    )
    output = episodes.copy().reset_index(drop=True)
    output["signal_timestamp"] = pd.to_datetime(
        output["signal_timestamp"], utc=True, errors="raise"
    )
    feature_rows: list[dict[str, object]] = []

    for episode in output.itertuples(index=False):
        stock = str(episode.stock)
        session = str(episode.session)
        checkpoint = _as_int(episode.checkpoint)
        available = bars.loc[
            bars["stock"].astype(str).eq(stock)
            & bars["session"].astype(str).eq(session)
            & bars["bar_ordinal"].astype(int).lt(checkpoint)
        ].sort_values("bar_ordinal", kind="mergesort")
        if len(available) != checkpoint:
            raise ValueError(f"incomplete causal bar prefix for {stock}|{session}|{checkpoint}")
        maximum_timestamp = pd.Timestamp(available["bar_complete_timestamp"].max())
        signal_timestamp = _as_timestamp(episode.signal_timestamp)
        if maximum_timestamp != signal_timestamp:
            raise ValueError("D0 feature prefix does not end at the signal close")

        opens = available["open"].to_numpy(float)
        highs = available["high"].to_numpy(float)
        lows = available["low"].to_numpy(float)
        closes = available["close"].to_numpy(float)
        if (
            not np.isfinite(np.concatenate([opens, highs, lows, closes])).all()
            or np.min(np.concatenate([opens, highs, lows, closes])) <= 0.0
        ):
            raise ValueError("D0 prices must be finite and positive")
        denominators = np.concatenate(([opens[0]], closes[:-1]))
        returns = np.log(closes / denominators)

        def trailing_sum(length: int, values: FloatArray = returns) -> float:
            return float(np.sum(values[-min(length, len(values)) :]))

        recent_three = returns[-min(3, len(returns)) :]
        earlier_three = returns[max(0, len(returns) - 6) : max(0, len(returns) - 3)]
        slope = float(
            np.mean(recent_three) - np.mean(earlier_three)
            if len(earlier_three)
            else np.mean(recent_three)
        )
        acceleration = float(returns[-1] - returns[-2] if len(returns) >= 2 else returns[-1])
        current_open = float(opens[-1])
        current_high = float(highs[-1])
        current_low = float(lows[-1])
        current_close = float(closes[-1])
        current_range = current_high - current_low
        upper_wick = (
            (current_high - max(current_open, current_close)) / current_range
            if current_range > 0.0
            else 0.0
        )
        lower_wick = (
            (min(current_open, current_close) - current_low) / current_range
            if current_range > 0.0
            else 0.0
        )
        wick_imbalance = float(lower_wick - upper_wick)
        recent_start = max(0, len(available) - 6)
        recent_high = float(np.max(highs[recent_start:]))
        recent_low = float(np.min(lows[recent_start:]))
        prior_high = float(np.max(highs[:-1])) if len(highs) > 1 else current_high
        prior_low = float(np.min(lows[:-1])) if len(lows) > 1 else current_low
        breakout = int(current_close > prior_high) - int(current_close < prior_low)
        opening_count = min(6, len(available))
        opening_high = float(np.max(highs[:opening_count]))
        opening_low = float(np.min(lows[:opening_count]))
        opening_mid = 0.5 * (opening_high + opening_low)

        if "volume" in available.columns:
            volume = pd.to_numeric(available["volume"], errors="coerce").to_numpy(float)
        else:
            volume = np.ones(len(available), dtype=float)
        typical = (highs + lows + closes) / 3.0
        usable_volume = np.where(np.isfinite(volume) & (volume > 0.0), volume, 0.0)
        vwap = float(
            np.average(typical, weights=usable_volume)
            if usable_volume.sum() > 0.0
            else np.mean(typical)
        )

        if "vti__bar_log_return" in available.columns:
            market_returns = pd.to_numeric(
                available["vti__bar_log_return"], errors="coerce"
            ).to_numpy(float)
        else:
            market_returns = np.full(len(available), np.nan, dtype=float)

        def market_sum(length: int, source: FloatArray = market_returns) -> float:
            values = source[-min(length, len(source)) :]
            return float(np.sum(values)) if np.isfinite(values).all() else math.nan

        if "market_breadth_bar_positive" in available.columns:
            breadth = float(available.iloc[-1]["market_breadth_bar_positive"])
        else:
            breadth = math.nan
        rejection_window = available.iloc[-min(3, len(available)) :]
        if {
            "upper_wick_pct_of_range",
            "lower_wick_pct_of_range",
        }.issubset(rejection_window.columns):
            recent_rejection = float(
                (
                    pd.to_numeric(rejection_window["lower_wick_pct_of_range"], errors="coerce")
                    - pd.to_numeric(rejection_window["upper_wick_pct_of_range"], errors="coerce")
                ).mean()
            )
        else:
            recent_rejection = wick_imbalance
        market_3 = market_sum(3)
        row = {
            "signed_return_1bar": trailing_sum(1),
            "signed_return_2bar": trailing_sum(2),
            "signed_return_3bar": trailing_sum(3),
            "signed_return_6bar": trailing_sum(6),
            "short_return_slope": slope,
            "return_acceleration": acceleration,
            "distance_from_session_open": math.log(current_close / opens[0]),
            "distance_from_vwap": math.log(current_close / vwap),
            "distance_from_opening_range_midpoint": math.log(current_close / opening_mid),
            "distance_from_opening_range_high": math.log(current_close / opening_high),
            "distance_from_opening_range_low": math.log(current_close / opening_low),
            "distance_from_recent_high": math.log(current_close / recent_high),
            "distance_from_recent_low": math.log(current_close / recent_low),
            "recent_breakout_direction": float(breakout),
            "recent_rejection_direction": recent_rejection,
            "current_candle_body_direction": math.log(current_close / current_open),
            "wick_imbalance": wick_imbalance,
            "market_return_1bar": market_sum(1),
            "market_return_2bar": market_sum(2),
            "market_return_3bar": market_3,
            "stock_minus_market_return_3bar": (
                trailing_sum(3) - market_3 if math.isfinite(market_3) else math.nan
            ),
            "market_breadth_direction": breadth,
            "maximum_feature_source_timestamp": maximum_timestamp,
        }
        feature_rows.append(row)

    return pd.concat([output, pd.DataFrame(feature_rows)], axis=1)


def build_signed_behavioural_features(
    checkpoint_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build the fixed D1 signed layer from already-audited causal fields."""

    required = {"stock", "session", "checkpoint", "signed_pressure"}
    missing = sorted(required.difference(checkpoint_rows.columns))
    if missing:
        raise ValueError(f"D1 inputs missing: {missing}")
    ordered = checkpoint_rows.copy()
    ordered["_original_order"] = np.arange(len(ordered), dtype=int)
    ordered = ordered.sort_values(["stock", "session", "checkpoint"], kind="mergesort").reset_index(
        drop=True
    )
    pressure = pd.to_numeric(ordered["signed_pressure"], errors="coerce")
    ordered["signed_pressure"] = pressure
    ordered["signed_pressure_change"] = ordered.groupby(["stock", "session"], sort=False)[
        "signed_pressure"
    ].diff()

    exhaustion_source = next(
        (
            column
            for column in ("signed_exhaustion", "audited_signed_exhaustion")
            if column in ordered.columns
        ),
        None,
    )
    ordered["signed_exhaustion"] = (
        pd.to_numeric(ordered[exhaustion_source], errors="coerce")
        if exhaustion_source is not None
        else np.nan
    )
    ordered["signed_exhaustion_change"] = ordered.groupby(["stock", "session"], sort=False)[
        "signed_exhaustion"
    ].diff()

    acceleration_source = next(
        (
            column
            for column in (
                "raw_component__signed_progress_acceleration",
                "raw_signed_progress_acceleration",
            )
            if column in ordered.columns
        ),
        None,
    )
    ordered["compression_release_direction"] = (
        pd.to_numeric(ordered[acceleration_source], errors="coerce")
        if acceleration_source is not None
        else np.nan
    )
    for dimension, output_name in (
        ("conviction", "pressure_x_conviction"),
        ("tension", "pressure_x_tension"),
        ("arousal", "pressure_x_arousal"),
    ):
        ordered[output_name] = (
            pressure * pd.to_numeric(ordered[dimension], errors="coerce")
            if dimension in ordered.columns
            else np.nan
        )
    activity_source = next(
        (
            column
            for column in (
                "raw_component__return_gap",
                "stock_minus_market_return_3bar",
            )
            if column in ordered.columns
        ),
        None,
    )
    ordered["signed_activity_imbalance"] = (
        pd.to_numeric(ordered[activity_source], errors="coerce")
        if activity_source is not None
        else np.nan
    )
    memory = (
        pd.to_numeric(ordered["recent_loop_memory_weighted_top_depth"], errors="coerce")
        if "recent_loop_memory_weighted_top_depth" in ordered.columns
        else pd.Series(np.nan, index=ordered.index)
    )
    ordered["signed_structural_memory"] = np.sign(pressure) * memory
    ordered = ordered.sort_values("_original_order", kind="mergesort").reset_index(drop=True)
    return ordered.loc[:, ["stock", "session", "checkpoint", *D1_FEATURES]]


def fit_empirical_bayes_orientation_map(
    development: pd.DataFrame,
    *,
    identity_column: str = "orientation_identity",
    label_column: str = "direction_up_10m",
    return_column: str = "signed_log_return_10m",
    prior_equivalent_sample_size: int = 50,
    minimum_support: int = 20,
) -> pd.DataFrame:
    """Fit the specified development-only empirical-Bayes orientation map."""

    required = {identity_column, label_column, return_column}
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError(f"orientation-map inputs missing: {missing}")
    if prior_equivalent_sample_size <= 0 or minimum_support <= 0:
        raise ValueError("orientation prior and support must be positive")
    frame = development.loc[
        development[label_column].notna() & development[return_column].notna()
    ].copy()
    if frame.empty:
        raise ValueError("orientation map needs development outcomes")
    frame[identity_column] = frame[identity_column].astype(str)
    labels = pd.to_numeric(frame[label_column], errors="raise").to_numpy(float)
    returns = pd.to_numeric(frame[return_column], errors="raise").to_numpy(float)
    if not np.isin(labels, [0.0, 1.0]).all() or not np.isfinite(returns).all():
        raise ValueError("orientation development outcomes are invalid")
    global_up = float(np.mean(labels))
    global_return = float(np.mean(returns))
    return_scale = float(np.std(returns))
    if not math.isfinite(return_scale) or return_scale <= 1e-12:
        return_scale = 1.0

    rows: list[dict[str, object]] = []
    for identity, group in frame.groupby(identity_column, sort=True):
        support = int(len(group))
        up_sum = float(pd.to_numeric(group[label_column], errors="raise").sum())
        return_sum = float(pd.to_numeric(group[return_column], errors="raise").sum())
        shrunk_up = (up_sum + prior_equivalent_sample_size * global_up) / (
            support + prior_equivalent_sample_size
        )
        shrunk_return = (return_sum + prior_equivalent_sample_size * global_return) / (
            support + prior_equivalent_sample_size
        )
        if support < minimum_support:
            orientation_class = "neutral"
        elif shrunk_up > global_up and shrunk_return > global_return:
            orientation_class = "positive"
        elif shrunk_up < global_up and shrunk_return < global_return:
            orientation_class = "negative"
        else:
            orientation_class = "neutral"
        score = (shrunk_up - global_up) + 0.5 * ((shrunk_return - global_return) / return_scale)
        rows.append(
            {
                "orientation_identity": str(identity),
                "raw_support": support,
                "raw_probability_up": up_sum / support,
                "raw_mean_return": return_sum / support,
                "orientation_probability_up": shrunk_up,
                "orientation_mean_return": shrunk_return,
                "orientation_score": score,
                "orientation_class": orientation_class,
                "global_probability_up": global_up,
                "global_mean_return": global_return,
                "prior_equivalent_sample_size": prior_equivalent_sample_size,
                "minimum_support": minimum_support,
            }
        )
    rows.append(
        {
            "orientation_identity": "__GLOBAL__",
            "raw_support": int(len(frame)),
            "raw_probability_up": global_up,
            "raw_mean_return": global_return,
            "orientation_probability_up": global_up,
            "orientation_mean_return": global_return,
            "orientation_score": 0.0,
            "orientation_class": "neutral",
            "global_probability_up": global_up,
            "global_mean_return": global_return,
            "prior_equivalent_sample_size": prior_equivalent_sample_size,
            "minimum_support": minimum_support,
        }
    )
    return (
        pd.DataFrame(rows)
        .sort_values("orientation_identity", kind="mergesort")
        .reset_index(drop=True)
    )


def apply_empirical_bayes_orientation_map(
    frame: pd.DataFrame,
    orientation_map: pd.DataFrame,
    *,
    identity_column: str = "orientation_identity",
) -> pd.DataFrame:
    """Apply a frozen orientation map without reading any row outcome."""

    if identity_column not in frame.columns:
        raise ValueError(f"orientation identity missing: {identity_column}")
    required_map = {
        "orientation_identity",
        "raw_support",
        "orientation_probability_up",
        "orientation_mean_return",
        "orientation_score",
        "orientation_class",
    }
    missing = sorted(required_map.difference(orientation_map.columns))
    if missing:
        raise ValueError(f"orientation map fields missing: {missing}")
    global_rows = orientation_map.loc[
        orientation_map["orientation_identity"].astype(str).eq("__GLOBAL__")
    ]
    if len(global_rows) != 1:
        raise ValueError("orientation map must contain one global fallback")
    global_row = global_rows.iloc[0]
    usable_map = orientation_map.loc[
        ~orientation_map["orientation_identity"].astype(str).eq("__GLOBAL__"),
        list(required_map),
    ].copy()
    usable_map = usable_map.rename(columns={"orientation_identity": identity_column})
    output = frame.copy()
    output[identity_column] = output[identity_column].astype(str)
    output = output.merge(
        usable_map,
        on=identity_column,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    fallback_values = {
        "raw_support": 0,
        "orientation_probability_up": float(global_row["orientation_probability_up"]),
        "orientation_mean_return": float(global_row["orientation_mean_return"]),
        "orientation_score": 0.0,
        "orientation_class": "neutral",
    }
    return output.fillna(value=fallback_values)


def crossfit_empirical_bayes_orientation(
    development: pd.DataFrame,
    *,
    fold_column: str = "fold",
    session_column: str = "session",
    prior_equivalent_sample_size: int = 50,
    minimum_support: int = 20,
) -> pd.DataFrame:
    """Give every development row an orientation map excluding its fold."""

    required = {
        fold_column,
        session_column,
        "orientation_identity",
        "direction_up_10m",
        "signed_log_return_10m",
    }
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError(f"orientation crossfit inputs missing: {missing}")
    session_folds = development.groupby(session_column, sort=False)[fold_column].nunique()
    if bool(session_folds.gt(1).any()):
        raise ValueError("complete sessions must belong to one orientation fold")
    source = development.copy()
    source["_crossfit_row"] = np.arange(len(source), dtype=int)
    pieces: list[pd.DataFrame] = []
    for fold in sorted(source[fold_column].drop_duplicates().tolist()):
        training = source.loc[source[fold_column].ne(fold)]
        held_out = source.loc[source[fold_column].eq(fold)]
        orientation_map = fit_empirical_bayes_orientation_map(
            training,
            prior_equivalent_sample_size=prior_equivalent_sample_size,
            minimum_support=minimum_support,
        )
        pieces.append(apply_empirical_bayes_orientation_map(held_out, orientation_map))
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values("_crossfit_row", kind="mergesort")
        .drop(columns="_crossfit_row")
        .reset_index(drop=True)
    )


def audited_state_orientation_map(centroids: pd.DataFrame) -> pd.DataFrame:
    """Reuse the audited sign of mean raw 6/12-bar signed-efficiency centroids."""

    required = {"state", "feature", "raw_feature_centroid"}
    missing = sorted(required.difference(centroids.columns))
    if missing:
        raise ValueError(f"audited orientation centroid fields missing: {missing}")
    selected = centroids.loc[
        centroids["feature"].astype(str).isin(["signed_efficiency_6", "signed_efficiency_12"])
    ].copy()
    grouped = selected.groupby("state", sort=True)["raw_feature_centroid"].mean()
    if grouped.empty:
        raise ValueError("audited orientation centroids are empty")
    result = grouped.rename("mean_raw_signed_efficiency").reset_index()
    result["state"] = result["state"].astype(int)
    result["orientation_sign"] = np.where(
        result["mean_raw_signed_efficiency"].to_numpy(float) >= 0.0, 1, -1
    ).astype(int)
    result["source_rule"] = (
        "sign(mean(raw signed_efficiency_6, raw signed_efficiency_12)); zero maps to +1"
    )
    return result


def _orientation_path(value: object) -> tuple[int, ...]:
    text = str(value)
    if "__o_" not in text:
        return ()
    payload = text.split("__o_", maxsplit=1)[1]
    try:
        return tuple(int(token) for token in payload.split("-"))
    except ValueError:
        return ()


def build_route_orientation_features(
    episodes: pd.DataFrame,
    structural_ledger: pd.DataFrame,
    state_orientation_map: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate audited active-prefix signs at each completed signal bar."""

    episode_required = {
        "stock",
        "session",
        "checkpoint",
        "signed_pressure",
        "route_resolution_state",
    }
    ledger_required = {
        "ledger_kind",
        "stock",
        "session",
        "bar_ordinal",
        "orientation_id",
    }
    missing_episode = sorted(episode_required.difference(episodes.columns))
    missing_ledger = sorted(ledger_required.difference(structural_ledger.columns))
    if missing_episode or missing_ledger:
        raise ValueError(f"D2 inputs missing: episodes={missing_episode}, ledger={missing_ledger}")
    if not {"state", "orientation_sign"}.issubset(state_orientation_map.columns):
        raise ValueError("audited state orientation map is incomplete")
    state_sign = {
        _as_int(row.state): _as_int(row.orientation_sign)
        for row in state_orientation_map.itertuples(index=False)
    }
    ledger = structural_ledger.copy()
    ledger["bar_ordinal"] = pd.to_numeric(ledger["bar_ordinal"], errors="raise").astype(int)
    rows: list[dict[str, float]] = []
    for episode in episodes.itertuples(index=False):
        stock = str(episode.stock)
        session = str(episode.session)
        checkpoint = _as_int(episode.checkpoint)
        active = ledger.loc[
            ledger["ledger_kind"].astype(str).eq("active_prefix")
            & ledger["stock"].astype(str).eq(stock)
            & ledger["session"].astype(str).eq(session)
            & ledger["bar_ordinal"].eq(checkpoint)
        ].copy()
        records: list[tuple[int, float, str]] = []
        for prefix in active.itertuples(index=False):
            path = _orientation_path(prefix.orientation_id)
            progress = int(
                float(getattr(prefix, "progress_states", 0))
                if pd.notna(getattr(prefix, "progress_states", np.nan))
                else 0
            )
            remaining = int(
                float(getattr(prefix, "transitions_remaining", 0))
                if pd.notna(getattr(prefix, "transitions_remaining", np.nan))
                else 0
            )
            next_state = path[progress] if progress < len(path) else None
            sign = state_sign.get(next_state, 0) if next_state is not None else 0
            denominator = progress + remaining - 1
            depth = (
                float(np.clip((progress - 1) / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
            )
            identity = str(getattr(prefix, "semantic_loop_id", prefix.orientation_id))
            records.append((sign, depth, identity))
        records.sort(key=lambda item: (-item[1], item[2]))
        signs = np.asarray([record[0] for record in records], dtype=int)
        depths = np.asarray([record[1] for record in records], dtype=float)
        positive = int(np.sum(signs > 0))
        negative = int(np.sum(signs < 0))
        neutral = int(np.sum(signs == 0))
        weighted_positive = float(np.sum(depths[signs > 0]))
        weighted_negative = float(np.sum(depths[signs < 0]))
        top = int(signs[0]) if len(signs) else 0
        second = int(signs[1]) if len(signs) > 1 else 0
        nonneutral = positive + negative
        weight_total = weighted_positive + weighted_negative
        margin = (
            (weighted_positive - weighted_negative) / weight_total
            if weight_total > 0.0
            else ((positive - negative) / nonneutral if nonneutral > 0 else 0.0)
        )
        agreement = max(positive, negative) / len(signs) if len(signs) else 0.0

        completions = ledger.loc[
            ledger["ledger_kind"].astype(str).eq("registered_completion")
            & ledger["stock"].astype(str).eq(stock)
            & ledger["session"].astype(str).eq(session)
            & ledger["bar_ordinal"].le(checkpoint)
        ].copy()
        completion_records: list[tuple[int, int]] = []
        for completion in completions.itertuples(index=False):
            path = _orientation_path(completion.orientation_id)
            sign = int(state_sign.get(path[-1], 0)) if path else 0
            completion_records.append((_as_int(completion.bar_ordinal), sign))
        completion_records.sort()
        recent_completion = completion_records[-1][1] if completion_records else 0
        same_memory = float(
            sum(
                math.exp(-(checkpoint - ordinal) / 6.0)
                for ordinal, sign in completion_records
                if top != 0 and sign == top
            )
        )
        pressure_sign = int(np.sign(_as_float(episode.signed_pressure)))
        rows.append(
            {
                "positive_active_prefix_count": float(positive),
                "negative_active_prefix_count": float(negative),
                "neutral_active_prefix_count": float(neutral),
                "depth_weighted_positive_orientation": weighted_positive,
                "depth_weighted_negative_orientation": weighted_negative,
                "top_route_orientation": float(top),
                "second_route_orientation": float(second),
                "orientation_margin": float(margin),
                "orientation_agreement": float(agreement),
                "orientation_disagreement": float(1.0 - agreement),
                "narrowing_route_orientation": float(
                    top if str(episode.route_resolution_state) == "NARROWING" else 0
                ),
                "recent_completed_loop_orientation": float(recent_completion),
                "recent_same_orientation_loop_memory_score": same_memory,
                "dominant_route_pressure_agreement": float(top * pressure_sign),
            }
        )
    return pd.concat(
        [
            episodes.loc[:, ["stock", "session", "checkpoint"]].reset_index(drop=True),
            pd.DataFrame(rows),
        ],
        axis=1,
    )


def freeze_confidence_boundary(
    development_oof_probabilities: Sequence[float] | FloatArray,
    *,
    weights: Sequence[float] | FloatArray | None = None,
    target_coverage: float = 0.35,
    minimum_actions: int = 150,
) -> dict[str, float | int | bool | str]:
    """Freeze one symmetric boundary from development OOF confidence only."""

    probabilities = np.asarray(development_oof_probabilities, dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or bool((probabilities < 0.0).any())
        or bool((probabilities > 1.0).any())
    ):
        raise ValueError("development OOF probabilities must lie in [0, 1]")
    if not 0.0 < target_coverage <= 1.0 or minimum_actions <= 0:
        raise ValueError("confidence coverage contract is invalid")
    sample_weights = (
        np.ones(len(probabilities), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if (
        len(sample_weights) != len(probabilities)
        or not np.isfinite(sample_weights).all()
        or bool((sample_weights <= 0.0).any())
    ):
        raise ValueError("confidence weights must be finite and positive")
    if not len(probabilities):
        raise ValueError("confidence boundary needs development OOF rows")

    required_coverage = max(target_coverage, min(1.0, minimum_actions / len(probabilities)))
    confidence = np.abs(probabilities - 0.5)
    order = np.lexsort((np.arange(len(confidence)), -confidence))
    ordered_confidence = confidence[order]
    ordered_weights = sample_weights[order]
    target_weight = required_coverage * float(sample_weights.sum())
    position = int(np.searchsorted(np.cumsum(ordered_weights), target_weight, side="left"))
    position = min(position, len(ordered_confidence) - 1)
    boundary = float(ordered_confidence[position])
    actioned = confidence >= boundary
    return {
        "boundary": boundary,
        "target_coverage": target_coverage,
        "effective_target_coverage": required_coverage,
        "minimum_actions": minimum_actions,
        "development_rows": int(len(probabilities)),
        "development_actions": int(actioned.sum()),
        "development_action_coverage": float(
            np.average(actioned.astype(float), weights=sample_weights)
        ),
        "same_call_put_boundary": True,
        "source": "2024_blocked_oof_only",
        "method": "deterministic_descending_weighted_quantile",
    }


def apply_selective_policy(
    probabilities: Sequence[float] | FloatArray,
    boundary: float,
) -> NDArray[np.str_]:
    """Apply the frozen symmetric CALL/PUT/ABSTAIN rule."""

    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or bool((values < 0.0).any()) or bool((values > 1.0).any()):
        raise ValueError("direction probabilities must lie in [0, 1]")
    if not 0.0 <= boundary <= 0.5:
        raise ValueError("direction confidence boundary must lie in [0, 0.5]")
    actions = np.full(len(values), "ABSTAIN", dtype="<U7")
    actions[values >= 0.5 + boundary] = "CALL"
    actions[values <= 0.5 - boundary] = "PUT"
    return actions


def aligned_returns(
    actions_or_sides: Sequence[object] | NDArray[np.object_],
    signed_returns: Sequence[float] | FloatArray,
) -> FloatArray:
    """Align underlying signed returns to CALL (+1) or PUT (-1)."""

    raw_sides = np.asarray(actions_or_sides)
    returns = np.asarray(signed_returns, dtype=float)
    if len(raw_sides) != len(returns):
        raise ValueError("actions and returns must have equal length")
    sides = np.full(len(raw_sides), np.nan, dtype=float)
    text = raw_sides.astype(str)
    sides[text == "CALL"] = 1.0
    sides[text == "PUT"] = -1.0
    numeric = np.asarray(
        [
            float(value) if isinstance(value, (int, float, np.number)) else np.nan
            for value in raw_sides
        ],
        dtype=float,
    )
    sides[np.isfinite(numeric) & (numeric == 1.0)] = 1.0
    sides[np.isfinite(numeric) & (numeric == -1.0)] = -1.0
    return np.asarray(sides * returns, dtype=np.float64)


def baseline_predictions(
    frame: pd.DataFrame,
    *,
    development_up_rate: float,
) -> pd.DataFrame:
    """Construct the five frozen pre-entry direction baselines."""

    required = {
        "signed_return_2bar",
        "signed_return_1bar",
        "market_return_2bar",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"baseline inputs missing: {missing}")
    if not 0.0 <= development_up_rate <= 1.0:
        raise ValueError("development prior must lie in [0, 1]")
    output = pd.DataFrame(index=frame.index)
    output["B0_probability"] = development_up_rate
    output["B0_side"] = 1 if development_up_rate >= 0.5 else -1
    output["B1_side"] = 1
    output["B2_side"] = np.sign(pd.to_numeric(frame["signed_return_2bar"], errors="coerce")).astype(
        int
    )
    output["B3_side"] = np.sign(pd.to_numeric(frame["signed_return_1bar"], errors="coerce")).astype(
        int
    )
    output["B4_side"] = np.sign(
        pd.to_numeric(frame["market_return_2bar"], errors="coerce").fillna(0.0)
    ).astype(int)
    return output


def session_bootstrap_samples(
    sessions: Sequence[object] | pd.Series,
    *,
    draws: int = 100,
    seed: int = 20260726,
) -> tuple[tuple[str, ...], ...]:
    """Draw complete session identifiers with replacement."""

    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    unique = np.asarray(sorted(pd.Series(sessions).astype(str).unique()), dtype=object)
    if not len(unique):
        raise ValueError("bootstrap needs at least one session")
    rng = np.random.default_rng(seed)
    return tuple(
        tuple(str(value) for value in rng.choice(unique, size=len(unique), replace=True))
        for _ in range(draws)
    )


def permute_labels_within_slates(
    frame: pd.DataFrame,
    *,
    label_column: str,
    strata: Sequence[str],
    seed: int,
) -> pd.Series:
    """Permute binary labels among stocks inside every frozen causal slate."""

    required = {label_column, *strata}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"label-null inputs missing: {missing}")
    output = frame[label_column].copy()
    rng = np.random.default_rng(seed)
    for indices in frame.groupby(list(strata), sort=True, dropna=False).groups.values():
        positions = list(indices)
        values = frame.loc[positions, label_column].to_numpy(copy=True)
        output.loc[positions] = rng.permutation(values)
    return output


def validate_protected_boundary(
    sessions: Sequence[object] | pd.Series,
    *,
    development_start: str = "2024-01-01",
    assessment_end: str = "2025-08-22",
) -> None:
    """Reject opened-holdout and protected rows before materialisation."""

    dates = pd.to_datetime(pd.Series(sessions), errors="raise").dt.normalize()
    if bool(dates.lt(pd.Timestamp(development_start)).any()):
        raise ValueError("a row predates the authorized development period")
    if bool(dates.gt(pd.Timestamp(assessment_end)).any()):
        raise ValueError("opened holdout or protected rows are forbidden")


def decide_direction_candidate(evidence: Mapping[str, object]) -> str:
    """Apply the frozen overall direction decision logic without relaxation."""

    blocker = evidence.get("blocker")
    if blocker:
        allowed_blockers = {
            "blocked_missing_auditable_orientation",
            "blocked_movement_model_reconstruction_failure",
            "blocked_chronology_or_leakage_failure",
            "blocked_model_convergence_failure",
            "blocked_reproducibility_or_audit_failure",
        }
        if str(blocker) not in allowed_blockers:
            raise ValueError(f"unknown direction blocker: {blocker}")
        return str(blocker)
    if not bool(evidence.get("episode_support_passed", False)):
        return "blocked_insufficient_direction_episode_support"
    if not bool(evidence.get("selective_support_passed", False)):
        return "blocked_insufficient_selective_action_support"

    full_gate = all(
        (
            bool(evidence.get("assessment_log_loss_improves_vs_d0", False)),
            bool(evidence.get("assessment_brier_improves_vs_d0", False)),
            _as_float(evidence.get("assessment_auc", -math.inf)) >= 0.55,
            _as_float(evidence.get("assessment_balanced_accuracy", -math.inf)) > 0.52,
            0.20 <= _as_float(evidence.get("action_coverage", math.nan)) <= 0.50,
            _as_float(evidence.get("selective_accuracy", -math.inf)) >= 0.55,
            _as_float(evidence.get("mean_aligned_return_10m", -math.inf)) > 0.0,
            _as_float(evidence.get("median_aligned_return_10m", -math.inf)) > 0.0,
            _as_float(evidence.get("bootstrap_80_accuracy_lower", -math.inf)) > 0.50,
            _as_float(evidence.get("bootstrap_80_mean_return_lower", -math.inf)) >= 0.0,
            _as_int(evidence.get("positive_months", 0)) >= 6,
            bool(evidence.get("beats_momentum_and_market", False)),
            bool(evidence.get("exceeds_all_nulls_log_loss_or_auc", False)),
            not bool(evidence.get("late_direction_problem", True)),
        )
    )
    if full_gate:
        return "movement_qualified_direction_candidate_supported"
    d1_adds = bool(evidence.get("d1_adds_value", False))
    d2_adds = bool(evidence.get("d2_adds_value", False))
    if d2_adds:
        return "route_orientation_adds_directional_value_but_full_gate_not_met"
    if d1_adds and bool(evidence.get("signed_behaviour_supported", False)):
        return "signed_behaviour_direction_supported_route_orientation_not_supported"
    if bool(evidence.get("directional_information_present", d1_adds)):
        if bool(evidence.get("stability_failed", False)):
            return "directional_candidate_unstable"
        return "directional_information_present_but_not_trade_ready"
    return "no_incremental_directional_signal"


def assign_contiguous_session_folds(
    sessions: Sequence[object] | pd.Series,
    *,
    folds: int = 4,
) -> pd.Series:
    """Assign complete sessions to deterministic contiguous calendar blocks."""

    if folds < 2:
        raise ValueError("OOF fitting needs at least two folds")
    values = pd.Series(sessions).astype(str)
    unique = np.asarray(sorted(values.unique()), dtype=object)
    if len(unique) < folds:
        raise ValueError("OOF fitting has fewer sessions than folds")
    mapping: dict[str, int] = {}
    for fold, block in enumerate(np.array_split(unique, folds)):
        for session in block:
            mapping[str(session)] = fold
    return values.map(mapping).astype(int)


def _frozen_categorical_levels(
    frame: pd.DataFrame, categorical_features: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    levels: dict[str, tuple[str, ...]] = {}
    for column in categorical_features:
        values = frame[column].fillna("__MISSING__").astype(str)
        observed = sorted(set(values).difference({"__MISSING__", "__UNKNOWN__"}))
        levels[column] = tuple([*observed, "__MISSING__", "__UNKNOWN__"])
    return levels


def _direction_design(
    frame: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    medians: Mapping[str, float],
    centers: Mapping[str, float],
    scales: Mapping[str, float],
    categorical_levels: Mapping[str, Sequence[str]],
) -> tuple[FloatArray, tuple[str, ...]]:
    pieces: list[FloatArray] = []
    names: list[str] = []
    for column in numeric_features:
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        missing = ~np.isfinite(raw)
        imputed = np.where(missing, float(medians[column]), raw)
        standardized = (imputed - float(centers[column])) / float(scales[column])
        pieces.extend(
            [
                np.asarray(standardized[:, None], dtype=np.float64),
                np.asarray(missing.astype(float)[:, None], dtype=np.float64),
            ]
        )
        names.extend([column, f"{column}__missing"])
    for column in categorical_features:
        values = frame[column].fillna("__MISSING__").astype(str)
        levels = tuple(str(value) for value in categorical_levels[column])
        known = set(levels)
        values = values.where(values.isin(known), "__UNKNOWN__")
        for level in levels:
            pieces.append(np.asarray(values.eq(level).to_numpy(float)[:, None], dtype=np.float64))
            names.append(f"{column}=={level}")
    design = (
        np.concatenate(pieces, axis=1) if pieces else np.empty((len(frame), 0), dtype=np.float64)
    )
    if not np.isfinite(design).all():
        raise ValueError("direction design contains non-finite values")
    return np.asarray(design, dtype=np.float64), tuple(names)


@dataclass(frozen=True)
class FrozenDirectionModel:
    """Serialized development-fitted deterministic second-stage logistic model."""

    model_id: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    medians: dict[str, float]
    centers: dict[str, float]
    scales: dict[str, float]
    categorical_levels: dict[str, tuple[str, ...]]
    design_feature_names: tuple[str, ...]
    coefficients: FloatArray
    intercept: float
    iterations: int

    def design(self, frame: pd.DataFrame) -> FloatArray:
        design, names = _direction_design(
            frame,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            medians=self.medians,
            centers=self.centers,
            scales=self.scales,
            categorical_levels=self.categorical_levels,
        )
        if names != self.design_feature_names:
            raise ValueError("direction design feature order drifted")
        return design

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        linear = self.design(frame) @ self.coefficients + self.intercept
        probabilities = np.empty(len(linear), dtype=float)
        positive = linear >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
        exponential = np.exp(linear[~positive])
        probabilities[~positive] = exponential / (1.0 + exponential)
        return np.asarray(probabilities, dtype=np.float64)

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "medians": self.medians,
            "robust_centers": self.centers,
            "robust_scales": self.scales,
            "categorical_levels": {
                key: list(value) for key, value in self.categorical_levels.items()
            },
            "design_feature_names": list(self.design_feature_names),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "iterations": self.iterations,
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "random_state": 20260726,
            "episode_weight": "equal",
        }

    @classmethod
    def from_dict(cls, specification: Mapping[str, object]) -> FrozenDirectionModel:
        return cls(
            model_id=str(specification["model_id"]),
            numeric_features=tuple(
                str(value) for value in cast(Sequence[object], specification["numeric_features"])
            ),
            categorical_features=tuple(
                str(value)
                for value in cast(Sequence[object], specification["categorical_features"])
            ),
            medians={
                str(key): _as_float(value)
                for key, value in cast(Mapping[str, object], specification["medians"]).items()
            },
            centers={
                str(key): _as_float(value)
                for key, value in cast(
                    Mapping[str, object], specification["robust_centers"]
                ).items()
            },
            scales={
                str(key): _as_float(value)
                for key, value in cast(Mapping[str, object], specification["robust_scales"]).items()
            },
            categorical_levels={
                str(key): tuple(str(item) for item in cast(Sequence[object], value))
                for key, value in cast(
                    Mapping[str, object], specification["categorical_levels"]
                ).items()
            },
            design_feature_names=tuple(
                str(value)
                for value in cast(Sequence[object], specification["design_feature_names"])
            ),
            coefficients=np.asarray(
                cast(Sequence[float], specification["coefficients"]), dtype=float
            ),
            intercept=_as_float(specification["intercept"]),
            iterations=_as_int(specification["iterations"]),
        )


def fit_direction_model(
    development: pd.DataFrame,
    *,
    target_column: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    model_id: str,
) -> FrozenDirectionModel:
    """Fit one frozen equal-episode-weight L2/liblinear direction model."""

    required = {target_column, *numeric_features, *categorical_features}
    missing = sorted(required.difference(development.columns))
    if missing:
        raise ValueError(f"direction model inputs missing: {missing}")
    training = development.loc[development[target_column].notna()].copy()
    target = pd.to_numeric(training[target_column], errors="raise").to_numpy(int)
    if not np.isin(target, [0, 1]).all() or len(np.unique(target)) != 2:
        raise ValueError("direction fitting needs both UP and DOWN outcomes")
    numeric = tuple(str(value) for value in numeric_features)
    categorical = tuple(str(value) for value in categorical_features)
    medians: dict[str, float] = {}
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in numeric:
        raw = pd.to_numeric(training[column], errors="coerce").to_numpy(float)
        finite = raw[np.isfinite(raw)]
        median = float(np.median(finite)) if len(finite) else 0.0
        imputed = np.where(np.isfinite(raw), raw, median)
        center = float(np.median(imputed))
        q25, q75 = np.quantile(imputed, [0.25, 0.75])
        scale = float(q75 - q25)
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        medians[column] = median
        centers[column] = center
        scales[column] = scale
    levels = _frozen_categorical_levels(training, categorical)
    design, names = _direction_design(
        training,
        numeric_features=numeric,
        categorical_features=categorical,
        medians=medians,
        centers=centers,
        scales=scales,
        categorical_levels=levels,
    )
    estimator = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="liblinear",
        max_iter=300,
        class_weight=None,
        random_state=20260726,
    )
    estimator.fit(design, target)
    iterations = int(estimator.n_iter_[0])
    if iterations >= 300:
        raise RuntimeError(f"{model_id} reached max_iter")
    return FrozenDirectionModel(
        model_id=model_id,
        numeric_features=numeric,
        categorical_features=categorical,
        medians=medians,
        centers=centers,
        scales=scales,
        categorical_levels=levels,
        design_feature_names=names,
        coefficients=np.asarray(estimator.coef_[0], dtype=np.float64),
        intercept=float(estimator.intercept_[0]),
        iterations=iterations,
    )


def manual_direction_probabilities(
    specification: Mapping[str, object],
    frame: pd.DataFrame,
) -> FloatArray:
    """Reconstruct serialized model probabilities without an sklearn estimator."""

    model = FrozenDirectionModel.from_dict(specification)
    return model.predict(frame)


def binary_direction_metrics(
    labels: Sequence[float] | FloatArray,
    probabilities: Sequence[float] | FloatArray,
) -> dict[str, float | int]:
    """Calculate the fixed unweighted binary direction metric surface."""

    target = np.asarray(labels, dtype=float)
    predicted = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(target) & np.isfinite(predicted)
    target = target[valid].astype(int)
    predicted = np.clip(predicted[valid], 1e-12, 1.0 - 1e-12)
    if not len(target) or not np.isin(target, [0, 1]).all():
        raise ValueError("direction metrics need valid binary outcomes")
    hard = (predicted >= 0.5).astype(int)
    auc = float(roc_auc_score(target, predicted)) if len(np.unique(target)) == 2 else math.nan
    average_precision = (
        float(average_precision_score(target, predicted))
        if len(np.unique(target)) == 2
        else math.nan
    )
    logits = np.log(predicted / (1.0 - predicted)).reshape(-1, 1)
    if len(np.unique(target)) == 2:
        calibration = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=300,
            random_state=20260726,
        ).fit(logits, target)
        calibration_intercept = float(calibration.intercept_[0])
        calibration_slope = float(calibration.coef_[0, 0])
    else:
        calibration_intercept = math.nan
        calibration_slope = math.nan
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (predicted >= lower) & (predicted <= upper if upper >= 1.0 else predicted < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(target[mask].mean()) - float(predicted[mask].mean())
            )
    return {
        "log_loss": float(log_loss(target, predicted, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, predicted)),
        "auc": auc,
        "average_precision": average_precision,
        "accuracy": float(accuracy_score(target, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(target, hard)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(target, hard)),
        "up_base_rate": float(target.mean()),
        "predicted_up_rate": float(hard.mean()),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "expected_calibration_error": ece,
        "episodes": int(len(target)),
    }


def selective_policy_metrics(
    frame: pd.DataFrame,
    *,
    action_column: str,
    horizon_minutes: int,
) -> dict[str, float | int]:
    """Calculate frozen underlying-return metrics on non-abstained episodes."""

    return_column = f"signed_log_return_{horizon_minutes}m"
    required = {
        action_column,
        return_column,
        f"upside_mfe_{horizon_minutes}m",
        f"upside_mae_{horizon_minutes}m",
        f"downside_mfe_{horizon_minutes}m",
        f"downside_mae_{horizon_minutes}m",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"selective metric inputs missing: {missing}")
    actions = frame[action_column].astype(str)
    actioned = frame.loc[actions.ne("ABSTAIN")].copy()
    action_values = actioned[action_column].astype(str).to_numpy()
    returns = pd.to_numeric(actioned[return_column], errors="coerce").to_numpy(float)
    aligned = aligned_returns(action_values, returns)
    sides = np.where(action_values == "CALL", 1, -1)
    valid_direction = np.isfinite(returns) & (returns != 0.0)
    true_side = np.sign(returns[valid_direction]).astype(int)
    predicted_side = sides[valid_direction]
    if valid_direction.any():
        accuracy = float(np.mean(predicted_side == true_side))
        balanced = float(
            balanced_accuracy_score((true_side > 0).astype(int), (predicted_side > 0).astype(int))
        )
    else:
        accuracy = math.nan
        balanced = math.nan
    favourable = np.where(
        sides > 0,
        pd.to_numeric(actioned[f"upside_mfe_{horizon_minutes}m"], errors="coerce").to_numpy(float),
        pd.to_numeric(actioned[f"downside_mfe_{horizon_minutes}m"], errors="coerce").to_numpy(
            float
        ),
    )
    adverse = np.where(
        sides > 0,
        pd.to_numeric(actioned[f"upside_mae_{horizon_minutes}m"], errors="coerce").to_numpy(float),
        pd.to_numeric(actioned[f"downside_mae_{horizon_minutes}m"], errors="coerce").to_numpy(
            float
        ),
    )
    finite_aligned = aligned[np.isfinite(aligned)]
    sorted_aligned = np.sort(finite_aligned)
    trim = int(math.floor(0.10 * len(sorted_aligned)))
    trimmed = (
        sorted_aligned[trim : len(sorted_aligned) - trim]
        if trim and len(sorted_aligned) > 2 * trim
        else sorted_aligned
    )
    calls = int(np.sum(action_values == "CALL"))
    puts = int(np.sum(action_values == "PUT"))
    absolute_total = float(np.nansum(np.abs(returns)))
    captured = float(np.nansum(np.maximum(aligned, 0.0)))
    mean_adverse = float(np.nanmean(adverse)) if len(adverse) else math.nan
    return {
        "horizon_minutes": horizon_minutes,
        "episodes": int(len(frame)),
        "actions": int(len(actioned)),
        "action_coverage": float(len(actioned) / len(frame)) if len(frame) else math.nan,
        "abstention_rate": float(1.0 - len(actioned) / len(frame)) if len(frame) else math.nan,
        "call_count": calls,
        "put_count": puts,
        "call_put_balance": float(calls / puts) if puts else math.inf,
        "directional_accuracy": accuracy,
        "balanced_accuracy": balanced,
        "mean_aligned_return": float(np.nanmean(aligned)) if len(aligned) else math.nan,
        "median_aligned_return": float(np.nanmedian(aligned)) if len(aligned) else math.nan,
        "positive_aligned_return_rate": float(np.nanmean(aligned > 0.0))
        if len(aligned)
        else math.nan,
        "trimmed_mean_aligned_return": float(np.mean(trimmed)) if len(trimmed) else math.nan,
        "mean_favourable_excursion": float(np.nanmean(favourable)) if len(favourable) else math.nan,
        "mean_adverse_excursion": mean_adverse,
        "favourable_adverse_excursion_ratio": (
            float(np.nanmean(favourable)) / mean_adverse
            if math.isfinite(mean_adverse) and mean_adverse > 0.0
            else math.nan
        ),
        "percent_total_absolute_movement_captured": (
            captured / absolute_total if absolute_total > 0.0 else math.nan
        ),
    }
