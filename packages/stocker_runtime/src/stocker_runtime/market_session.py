"""Shared XNYS regular-session expectation without widening market scope."""

from __future__ import annotations

from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Any, cast
from zoneinfo import ZoneInfo

_NEW_YORK = ZoneInfo("America/New_York")


@lru_cache(maxsize=32)
def xnys_session_window_us(session_date: date) -> tuple[int, int] | None:
    """Return the official NYSE regular-session window for one local date."""

    import pandas_market_calendars as market_calendars

    schedule = market_calendars.get_calendar("NYSE").schedule(
        start_date=session_date.isoformat(),
        end_date=session_date.isoformat(),
    )
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    market_open = cast(Any, row["market_open"])
    market_close = cast(Any, row["market_close"])
    return (
        int(market_open.timestamp() * 1_000_000),
        int(market_close.timestamp() * 1_000_000),
    )


def market_data_expected_since_us(now_us: int) -> int | None:
    """Return the current XNYS regular-session open, or None outside that session."""

    session_date = datetime.fromtimestamp(now_us / 1_000_000, UTC).astimezone(_NEW_YORK).date()
    window = xnys_session_window_us(session_date)
    if window is None:
        return None
    opened_at_us, closed_at_us = window
    return opened_at_us if opened_at_us <= now_us < closed_at_us else None
