from __future__ import annotations

import math

import pandas as pd
import pytest

from stocker_research.intraday_features import (
    IntradayFeatureConfig,
    add_session_vwap,
    add_time_stop_features,
    build_intraday_feature_frame,
)


def _session_timestamps(session_date: str) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{session_date} 13:30",
        f"{session_date} 20:00",
        freq="5min",
        tz="UTC",
    )


def _intraday_frame(
    session_dates: list[str],
    *,
    volume_start: int = 1_000,
    missing_volume: bool = False,
) -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(
        [
            timestamp
            for session_date in session_dates
            for timestamp in _session_timestamps(session_date)
        ]
    )
    row_count = len(timestamps)
    close = pd.Series([100.0 + index * 0.1 for index in range(row_count)])
    high = close + 0.2
    low = close - 0.2
    volume = pd.Series([volume_start + index for index in range(row_count)], dtype="float")
    if missing_volume:
        volume.iloc[0] = math.nan
        volume.iloc[1] = 0.0
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _config(**overrides: object) -> IntradayFeatureConfig:
    values = {
        "timeframe": "5m",
        "market_calendar": "XNYS",
        "opening_minutes": 30,
        "open_buffer_minutes": 15,
        "entry_cutoff_before_close_minutes": 30,
        "flatten_before_close_minutes": 10,
        "relative_volume_lookback_sessions": 2,
        "range_lookback_bars": 3,
        "compression_lookback_bars": 4,
    }
    values.update(overrides)
    return IntradayFeatureConfig(**values)


def test_session_clock_assigns_sessions_and_resets_bar_index() -> None:
    frame = _intraday_frame(["2026-06-23", "2026-06-24"])

    features = build_intraday_feature_frame(frame, _config())

    assert features["session_id"].iloc[0] == features["session_id"].iloc[78]
    assert features["session_id"].iloc[79] != features["session_id"].iloc[0]
    assert features["bar_index_in_session"].iloc[0] == 0
    assert features["bar_index_in_session"].iloc[78] == 78
    assert features["bar_index_in_session"].iloc[79] == 0
    assert features["bars_remaining_in_session"].iloc[0] == 78
    assert features["bars_remaining_in_session"].iloc[78] == 0
    assert features["minutes_from_session_open"].iloc[0] == 0.0
    assert features["minutes_from_session_open"].iloc[6] == 30.0
    assert features["minutes_to_session_close"].iloc[0] == 390.0
    assert features["minutes_to_session_close"].iloc[78] == 0.0
    assert features["is_first_bar_of_session"].iloc[0]
    assert features["is_last_bar_of_session"].iloc[78]
    assert not features["session_complete_warning"].any()


def test_time_window_flags_no_entry_and_flatten_windows() -> None:
    frame = _intraday_frame(["2026-06-23"])

    features = build_intraday_feature_frame(frame, _config())

    assert not features["after_open_buffer"].iloc[0]
    assert features["after_open_buffer"].iloc[3]
    assert features["in_opening_range_window"].iloc[5]
    assert not features["in_opening_range_window"].iloc[6]
    assert features["can_open_new_position"].iloc[10]
    assert features["no_entry_window"].iloc[72]
    assert not features["can_enter_before_close_cutoff"].iloc[72]
    assert features["must_flatten_now"].iloc[76]
    assert not features["can_open_new_position"].iloc[76]


def test_opening_range_values_only_available_after_window_completes() -> None:
    frame = _intraday_frame(["2026-06-23"])
    frame.loc[:5, "high"] = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    frame.loc[:5, "low"] = [99.0, 98.0, 97.0, 96.0, 95.0, 94.0]

    features = build_intraday_feature_frame(frame, _config())

    assert not features["opening_range_complete"].iloc[5]
    assert pd.isna(features["opening_range_high"].iloc[5])
    assert features["opening_range_complete"].iloc[6]
    assert features["opening_range_high"].iloc[6] == 106.0
    assert features["opening_range_low"].iloc[6] == 94.0
    assert features["opening_range_mid"].iloc[6] == 100.0
    assert features["opening_range_width"].iloc[6] == 12.0


def test_session_vwap_resets_each_session_and_handles_missing_volume() -> None:
    frame = _intraday_frame(["2026-06-23", "2026-06-24"], missing_volume=True)

    features = add_session_vwap(build_intraday_feature_frame(frame, _config()))

    assert pd.isna(features["session_vwap"].iloc[0])
    assert pd.isna(features["session_vwap"].iloc[1])
    second_session_first = 79
    typical_price = (
        features["high"].iloc[second_session_first]
        + features["low"].iloc[second_session_first]
        + features["close"].iloc[second_session_first]
    ) / 3.0
    assert features["session_vwap"].iloc[second_session_first] == pytest.approx(typical_price)


def test_relative_volume_uses_prior_sessions_only() -> None:
    frame = _intraday_frame(["2026-06-23", "2026-06-24", "2026-06-25"])
    frame.loc[0, "volume"] = 100.0
    frame.loc[79, "volume"] = 200.0
    frame.loc[158, "volume"] = 900.0

    features = build_intraday_feature_frame(frame, _config(relative_volume_lookback_sessions=2))

    assert pd.isna(features["relative_volume_at_bar_index"].iloc[0])
    assert pd.isna(features["relative_volume_at_bar_index"].iloc[79])
    assert features["relative_volume_at_bar_index"].iloc[158] == pytest.approx(900.0 / 150.0)


def test_previous_session_levels_use_prior_session_only() -> None:
    frame = _intraday_frame(["2026-06-23", "2026-06-24", "2026-06-25"])
    frame.loc[:78, "high"] = 111.0
    frame.loc[:78, "low"] = 91.0
    frame.loc[78, "close"] = 105.0
    frame.loc[79, "open"] = 112.0
    frame.loc[158, "open"] = 90.0

    features = build_intraday_feature_frame(frame, _config())

    assert pd.isna(features["previous_session_high"].iloc[0])
    assert features["previous_session_high"].iloc[79] == 111.0
    assert features["previous_session_low"].iloc[79] == 91.0
    assert features["previous_session_close"].iloc[79] == 105.0
    assert features["gap_vs_previous_close"].iloc[79] == pytest.approx(7.0 / 105.0)
    assert features["open_above_previous_high"].iloc[79]
    assert features["open_below_previous_low"].iloc[158]


def test_range_compression_uses_only_current_and_prior_bars() -> None:
    frame = _intraday_frame(["2026-06-23"])

    features = build_intraday_feature_frame(frame, _config(range_lookback_bars=3))

    assert pd.isna(features["recent_high"].iloc[1])
    assert features["recent_high"].iloc[2] == pytest.approx(frame["high"].iloc[:3].max())
    assert features["recent_low"].iloc[2] == pytest.approx(frame["low"].iloc[:3].min())
    assert features["rolling_intraday_range"].iloc[2] == pytest.approx(
        features["recent_high"].iloc[2] - features["recent_low"].iloc[2]
    )


def test_incomplete_calendar_session_produces_warning_flag() -> None:
    frame = _intraday_frame(["2026-06-23"]).iloc[:10].reset_index(drop=True)

    features = build_intraday_feature_frame(frame, _config())

    assert features["session_complete_warning"].all()
    assert "incomplete_session" in set(features["session_warning_reason"])


def test_future_rows_after_eval_window_do_not_change_feature_values() -> None:
    frame = _intraday_frame(["2026-06-23", "2026-06-24", "2026-06-25"])
    eval_end = 79 + 20
    full_features = build_intraday_feature_frame(frame, _config())
    mutated = frame.copy()
    mutated.loc[eval_end + 1 :, ["high", "low", "close", "volume"]] = 10_000.0

    mutated_features = build_intraday_feature_frame(mutated, _config())

    columns = [
        "opening_range_high",
        "session_vwap",
        "relative_volume_at_bar_index",
        "previous_session_high",
        "recent_high",
        "rolling_intraday_range",
    ]
    pd.testing.assert_frame_equal(
        full_features.loc[:eval_end, columns],
        mutated_features.loc[:eval_end, columns],
        check_dtype=False,
    )


def test_context_window_features_match_full_frame_for_eval_rows() -> None:
    frame = _intraday_frame(["2026-06-23", "2026-06-24", "2026-06-25"])
    full_features = build_intraday_feature_frame(frame, _config())
    context_frame = frame.iloc[79:180].reset_index(drop=True)
    context_features = build_intraday_feature_frame(context_frame, _config())
    eval_timestamps = context_frame["timestamp"].iloc[79:].reset_index(drop=True)
    full_eval = full_features[full_features["timestamp"].isin(eval_timestamps)].reset_index(
        drop=True
    )
    context_eval = context_features.iloc[79:].reset_index(drop=True)

    columns = [
        "session_id",
        "bar_index_in_session",
        "opening_range_high",
        "session_vwap",
        "recent_high",
        "can_open_new_position",
    ]
    pd.testing.assert_frame_equal(
        full_eval[columns],
        context_eval[columns],
        check_dtype=False,
    )


def test_daily_timeframe_does_not_pretend_to_have_intraday_features() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-23", "2026-06-24"], utc=True),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1_000, 1_100],
        }
    )

    with pytest.raises(ValueError, match="intraday timeframe"):
        build_intraday_feature_frame(frame, _config(timeframe="1d"))


def test_time_stop_helper_counts_bars_since_entry_without_strategy_logic() -> None:
    frame = _intraday_frame(["2026-06-23"]).iloc[:6].reset_index(drop=True)
    entries = pd.Series([False, True, False, False, True, False])

    features = add_time_stop_features(frame, entries=entries, max_bars_held=2)

    assert pd.isna(features["bars_since_entry"].iloc[0])
    assert features["bars_since_entry"].iloc[1:].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]
    assert features["time_stop_reached"].tolist() == [False, False, False, True, False, False]
