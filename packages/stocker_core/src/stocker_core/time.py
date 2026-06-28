"""Time helpers for market-data and execution-safe code."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(tz=UTC)


def ensure_timezone(value: datetime, timezone: str = "UTC") -> datetime:
    """Attach or convert a datetime to the requested timezone."""

    zone = ZoneInfo(timezone)
    if value.tzinfo is None:
        return value.replace(tzinfo=zone)
    return value.astimezone(zone)
