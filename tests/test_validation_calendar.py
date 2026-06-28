import pandas as pd

from stocker_data.validate import validate_ohlcv


def _frame(timestamps: pd.DatetimeIndex | list[pd.Timestamp]) -> pd.DataFrame:
    count = len(timestamps)
    base = pd.Series(range(count), dtype="float64") * 0.01 + 100.0
    return pd.DataFrame(
        {
            "timestamp": list(timestamps),
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.25,
            "volume": 1_000,
        }
    )


def _issue_codes(frame: pd.DataFrame, *, timeframe: str, market_calendar: str | None) -> set[str]:
    issues = validate_ohlcv(
        frame,
        timeframe=timeframe,
        timezone="UTC",
        require_timezone=True,
        market_calendar=market_calendar,
    )
    return {issue.code for issue in issues}


def test_daily_stock_weekend_is_not_flagged_with_xnys_calendar() -> None:
    frame = _frame(
        [
            pd.Timestamp("2024-01-05T00:00:00Z"),
            pd.Timestamp("2024-01-08T00:00:00Z"),
        ]
    )

    codes = _issue_codes(frame, timeframe="1d", market_calendar="XNYS")

    assert "timestamp_gap" not in codes
    assert "missing_market_session" not in codes


def test_daily_stock_missing_true_session_is_flagged_with_xnys_calendar() -> None:
    frame = _frame(
        [
            pd.Timestamp("2024-01-05T00:00:00Z"),
            pd.Timestamp("2024-01-09T00:00:00Z"),
        ]
    )

    codes = _issue_codes(frame, timeframe="1d", market_calendar="XNYS")

    assert "timestamp_gap" in codes
    assert "missing_market_session" in codes


def test_intraday_stock_overnight_gap_is_not_flagged_with_xnys_calendar() -> None:
    day_one = pd.date_range(
        "2024-01-02 14:30",
        "2024-01-02 21:00",
        freq="1min",
        inclusive="left",
        tz="UTC",
    )
    day_two = pd.date_range(
        "2024-01-03 14:30",
        "2024-01-03 21:00",
        freq="1min",
        inclusive="left",
        tz="UTC",
    )
    frame = _frame(day_one.append(day_two))

    codes = _issue_codes(frame, timeframe="1m", market_calendar="XNYS")

    assert "timestamp_gap" not in codes


def test_intraday_stock_missing_in_session_bar_is_flagged_with_xnys_calendar() -> None:
    session = pd.date_range(
        "2024-01-02 14:30",
        "2024-01-02 21:00",
        freq="1min",
        inclusive="left",
        tz="UTC",
    )
    frame = _frame(session.delete(10))

    codes = _issue_codes(frame, timeframe="1m", market_calendar="XNYS")

    assert "timestamp_gap" in codes


def test_no_market_calendar_skips_noisy_gap_check() -> None:
    frame = _frame(
        [
            pd.Timestamp("2024-01-05T00:00:00Z"),
            pd.Timestamp("2024-01-08T00:00:00Z"),
        ]
    )

    issues = validate_ohlcv(
        frame,
        timeframe="1d",
        timezone="UTC",
        require_timezone=True,
        market_calendar=None,
    )

    assert any(issue.code == "calendar_gap_check_skipped" for issue in issues)
    assert not any(issue.code == "timestamp_gap" for issue in issues)
