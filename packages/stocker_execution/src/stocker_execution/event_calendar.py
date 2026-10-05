"""Hand-maintained scheduled-release calendar. Observation and display context only.

Weekly rules use America/New_York local time. Holiday shifts are not inferred: edit the
file with the publisher's announced dates when a release moves.
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import AwareDatetime, Field, model_validator

from stocker_execution.config import Market, Strict
from stocker_execution.rules import NY

WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI")


class WeeklyRule(Strict):
    weekday: Literal["MON", "TUE", "WED", "THU", "FRI"]
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")


class ScheduledEvent(Strict):
    name: str = Field(min_length=3, max_length=120)
    markets: list[Market] = Field(min_length=1)
    weekly: WeeklyRule | None = None
    at: AwareDatetime | None = None

    @model_validator(mode="after")
    def one_schedule(self) -> "ScheduledEvent":
        if (self.weekly is None) == (self.at is None):
            raise ValueError("EVENT_NEEDS_EXACTLY_ONE_OF_WEEKLY_OR_AT")
        return self


class EventCalendar(Strict):
    events: list[ScheduledEvent] = Field(default_factory=list, max_length=500)

    def occurrences(self, market: str, day: date) -> list[tuple[datetime, str]]:
        found = []
        for event in self.events:
            if market not in event.markets:
                continue
            if event.at is not None:
                if event.at.astimezone(NY).date() == day:
                    found.append((event.at, event.name))
            elif event.weekly and WEEKDAYS.index(event.weekly.weekday) == day.weekday():
                hour, minute = map(int, event.weekly.time.split(":"))
                found.append((datetime.combine(day, time(hour, minute), NY), event.name))
        return sorted(found)

    def near(self, market: str, clock: datetime) -> list[dict[str, str]]:
        """Releases in the hour before a clock or inside its 60-minute holding window."""
        result = []
        # The hour before the 00:00 clock lies on the previous New York date.
        days = {clock.astimezone(NY).date(), (clock - timedelta(minutes=60)).astimezone(NY).date()}
        for at, name in sorted(o for day in sorted(days) for o in self.occurrences(market, day)):
            if clock - timedelta(minutes=60) <= at < clock:
                relation = "HOUR_BEFORE_CLOCK"
            elif clock <= at < clock + timedelta(minutes=60):
                relation = "DURING_HOLDING_WINDOW"
            else:
                continue
            result.append({"name": name, "at": at.isoformat(), "relation": relation})
        return result


def load_calendar(path: Path) -> EventCalendar:
    if path.stat().st_size > 256 * 1024:
        raise ValueError("EVENT_CALENDAR_TOO_LARGE")
    return EventCalendar.model_validate(yaml.safe_load(path.read_text()) or {})
