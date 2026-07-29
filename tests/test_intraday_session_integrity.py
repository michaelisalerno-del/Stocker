from __future__ import annotations

import pandas as pd

from stocker_research.intraday_session_integrity import (
    analyze_symbol_sessions,
    expected_session_timestamps,
)


def _ohlcv_frame(timestamps: list[pd.Timestamp], symbol: str = "AAPL.US") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "symbol": symbol,
            "source": "eodhd",
            "instrument_type": "stock",
            "timeframe": "5m",
        }
    )


def test_expected_session_timestamps_include_vendor_boundary_labels() -> None:
    timestamps = expected_session_timestamps(
        pd.Timestamp("2026-01-05 14:30:00+00:00"),
        pd.Timestamp("2026-01-05 21:00:00+00:00"),
        timeframe="5m",
    )

    assert len(timestamps) == 79
    assert timestamps[0] == pd.Timestamp("2026-01-05 14:30:00+00:00")
    assert timestamps[-1] == pd.Timestamp("2026-01-05 21:00:00+00:00")


def test_session_integrity_marks_regular_and_dst_shifted_sessions_complete() -> None:
    winter = expected_session_timestamps(
        pd.Timestamp("2026-01-05 14:30:00+00:00"),
        pd.Timestamp("2026-01-05 21:00:00+00:00"),
        timeframe="5m",
    )
    summer = expected_session_timestamps(
        pd.Timestamp("2026-06-25 13:30:00+00:00"),
        pd.Timestamp("2026-06-25 20:00:00+00:00"),
        timeframe="5m",
    )

    rows = analyze_symbol_sessions(
        _ohlcv_frame([*winter, *summer]),
        symbol="AAPL.US",
        timeframe="5m",
        market_calendar="XNYS",
    )

    assert [row["bucket"] for row in rows] == ["complete", "complete"]
    assert all(row["appears_complete"] for row in rows)
    assert all(row["dst_handled_correctly"] for row in rows)
    assert {row["timestamp_convention"] for row in rows} == {"boundary_inclusive"}


def test_session_integrity_buckets_early_close_as_expected_info() -> None:
    early_close = expected_session_timestamps(
        pd.Timestamp("2025-07-03 13:30:00+00:00"),
        pd.Timestamp("2025-07-03 17:00:00+00:00"),
        timeframe="5m",
    )

    rows = analyze_symbol_sessions(
        _ohlcv_frame(early_close),
        symbol="AAPL.US",
        timeframe="5m",
        market_calendar="XNYS",
    )

    assert len(rows) == 1
    assert rows[0]["is_early_close"] is True
    assert rows[0]["bucket"] == "expected_market_early_close"
    assert rows[0]["severity"] == "info"
    assert rows[0]["appears_complete"] is True


def test_session_integrity_buckets_missing_boundary_bars() -> None:
    full = expected_session_timestamps(
        pd.Timestamp("2026-01-05 14:30:00+00:00"),
        pd.Timestamp("2026-01-05 21:00:00+00:00"),
        timeframe="5m",
    )

    missing_close = analyze_symbol_sessions(
        _ohlcv_frame(full[:-1]),
        symbol="AAPL.US",
        timeframe="5m",
        market_calendar="XNYS",
    )
    missing_open = analyze_symbol_sessions(
        _ohlcv_frame(full[1:]),
        symbol="AAPL.US",
        timeframe="5m",
        market_calendar="XNYS",
    )

    assert missing_close[0]["bucket"] == "possible_missing_close_bar"
    assert missing_close[0]["missing_bars_at_close"] is True
    assert missing_open[0]["bucket"] == "possible_missing_open_bar"
    assert missing_open[0]["missing_bars_at_open"] is True


def test_session_integrity_buckets_mid_session_gap() -> None:
    full = expected_session_timestamps(
        pd.Timestamp("2026-01-05 14:30:00+00:00"),
        pd.Timestamp("2026-01-05 21:00:00+00:00"),
        timeframe="5m",
    )
    frame = _ohlcv_frame([*full[:10], *full[11:]])

    rows = analyze_symbol_sessions(
        frame,
        symbol="AAPL.US",
        timeframe="5m",
        market_calendar="XNYS",
    )

    assert rows[0]["bucket"] == "possible_mid_session_gap"
    assert rows[0]["missing_bars_in_middle"] is True
