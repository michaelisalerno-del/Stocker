"""Shared immutable scope and session rules for event-window retention V0."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

RETENTION_CONTROLLED_EVENT_TYPES = (
    "raw_callback_envelope_event",
    "underlying_bbo_update",
    "underlying_level1_quote_event",
    "underlying_tick_bidask_event",
    "underlying_tick_trade_event",
    "underlying_trade_update",
    "underlying_depth_event",
    "underlying_depth_snapshot",
)

RETENTION_CONTROLLED_CALLBACK_KINDS = (
    "level1_quote_update",
    "official_provider_tick_by_tick_bidask",
    "official_provider_tick_by_tick_trade",
    "official_provider_tick_price",
    "official_provider_tick_size",
    "official_provider_depth",
    "official_provider_depth_reset",
    "tick_by_tick_bidask",
    "tick_by_tick_trade",
    "tick_price",
    "tick_size",
    "depth",
    "depth_reset",
)

RETENTION_CONTROLLED_STREAM_KINDS = frozenset(
    {
        "underlying_level1",
        "underlying_tick_bidask",
        "underlying_tick_last",
        "underlying_depth",
    }
)

NEW_YORK = ZoneInfo("America/New_York")


def retention_session(timestamp: datetime) -> date:
    """Match event ingestion's America/New_York calendar-date session rule."""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("retention session timestamp must be timezone-aware")
    return timestamp.astimezone(NEW_YORK).date()


def retention_session_timestamp_bounds(session: date) -> tuple[str, str]:
    """Return UTC text bounds compatible with persisted ISO timestamps."""

    start = datetime.combine(session, datetime.min.time(), tzinfo=NEW_YORK)
    end = start + timedelta(days=1)
    return start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()
