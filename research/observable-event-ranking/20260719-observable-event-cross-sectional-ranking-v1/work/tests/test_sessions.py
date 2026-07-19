from __future__ import annotations

import pandas as pd

from stocker_research.observable_event_ranking_v1.sessions import (
    official_session_schedule,
    validate_regular_session_bars,
)


def test_official_schedule_handles_dst_holiday_and_early_close() -> None:
    dst = official_session_schedule("2025-03-07", "2025-03-10")
    july = official_session_schedule("2025-07-03", "2025-07-07")
    early = official_session_schedule("2025-11-28", "2025-11-28")

    assert list(dst["market_open_utc"]) == [
        pd.Timestamp("2025-03-07T14:30:00Z"),
        pd.Timestamp("2025-03-10T13:30:00Z"),
    ]
    assert "2025-07-04" not in set(july["session"].dt.strftime("%Y-%m-%d"))
    assert early.iloc[0]["market_close_utc"] == pd.Timestamp("2025-11-28T18:00:00Z")


def test_session_bar_validation_fails_closed_on_every_structural_problem() -> None:
    rows = [
        {
            "session": pd.Timestamp("2025-01-02", tz="UTC"),
            "bar_start": pd.Timestamp("2025-01-02T14:30:00Z"),
            "bar_end": pd.Timestamp("2025-01-02T14:35:00Z"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "fully_completed": True,
        },
        {
            "session": pd.Timestamp("2025-01-02", tz="UTC"),
            "bar_start": pd.Timestamp("2025-01-02T14:40:00Z"),
            "bar_end": pd.Timestamp("2025-01-02T14:45:00Z"),
            "open": 100.0,
            "high": 99.0,
            "low": 101.0,
            "close": 100.5,
            "fully_completed": False,
        },
        {
            "session": pd.Timestamp("2025-01-02", tz="UTC"),
            "bar_start": pd.Timestamp("2025-01-02T14:40:00Z"),
            "bar_end": pd.Timestamp("2025-01-02T14:45:00Z"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "fully_completed": True,
        },
        {
            "session": pd.Timestamp("2025-01-06", tz="UTC"),
            "bar_start": pd.Timestamp("2025-01-06T21:00:00Z"),
            "bar_end": pd.Timestamp("2025-01-06T21:05:00Z"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "fully_completed": True,
        },
    ]

    issues = validate_regular_session_bars(pd.DataFrame(rows), source_convention_proven=False)
    codes = {issue.code for issue in issues}

    assert {
        "unproven_source_timestamp_convention",
        "source_gap",
        "duplicate_bar",
        "impossible_ohlc",
        "incomplete_bar",
        "bar_outside_regular_session",
        "missing_session",
    }.issubset(codes)
