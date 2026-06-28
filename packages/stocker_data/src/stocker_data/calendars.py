"""Exchange calendar helpers."""

from typing import Any


def get_market_calendar(name: str = "NYSE") -> Any:
    """Return a pandas-market-calendars calendar by name."""

    import pandas_market_calendars as mcal

    return mcal.get_calendar(name)
