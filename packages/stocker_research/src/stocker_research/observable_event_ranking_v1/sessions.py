"""Official exchange-session schedules and bounded five-minute bar validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from stocker_data.calendars import get_market_calendar


@dataclass(frozen=True)
class SessionIssue:
    """One fail-closed chronology or bar-integrity issue."""

    code: str
    count: int
    detail: str


def official_session_schedule(start_date: str, end_date: str) -> pd.DataFrame:
    """Return the repository-supported XNYS schedule with canonical UTC timestamps."""

    calendar = get_market_calendar("XNYS")
    raw = cast(
        pd.DataFrame,
        calendar.schedule(start_date=start_date, end_date=end_date),
    )
    schedule = raw.reset_index().rename(
        columns={
            raw.index.name or "index": "session",
            "market_open": "market_open_utc",
            "market_close": "market_close_utc",
        }
    )
    for column in ("session", "market_open_utc", "market_close_utc"):
        schedule[column] = pd.to_datetime(schedule[column], utc=True)
    return schedule.loc[:, ["session", "market_open_utc", "market_close_utc"]].sort_values(
        "session", kind="mergesort", ignore_index=True
    )


def validate_regular_session_bars(
    bars: pd.DataFrame,
    *,
    source_convention_proven: bool,
) -> list[SessionIssue]:
    """Validate one projected symbol panel without creating full-panel copies."""

    required = {
        "session",
        "bar_start",
        "bar_end",
        "open",
        "high",
        "low",
        "close",
        "fully_completed",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        return [SessionIssue("missing_required_columns", len(missing), ",".join(missing))]
    if bars.empty:
        return [SessionIssue("empty_bar_panel", 1, "no regular-session bars supplied")]

    frame = bars.loc[:, sorted(required)].copy()
    for column in ("session", "bar_start", "bar_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["session", "bar_start", "bar_end"], kind="mergesort")
    issues: list[SessionIssue] = []
    if not source_convention_proven:
        issues.append(
            SessionIssue(
                "unproven_source_timestamp_convention",
                1,
                "source bars cannot enter a scientific run until label semantics are proven",
            )
        )

    schedule = official_session_schedule(
        frame["session"].min().strftime("%Y-%m-%d"),
        frame["session"].max().strftime("%Y-%m-%d"),
    )
    schedule_by_date: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for index in schedule.index:
        schedule_session = pd.Timestamp(cast(Any, schedule.at[index, "session"]))
        schedule_by_date[schedule_session.strftime("%Y-%m-%d")] = (
            pd.Timestamp(cast(Any, schedule.at[index, "market_open_utc"])),
            pd.Timestamp(cast(Any, schedule.at[index, "market_close_utc"])),
        )
    observed_dates = set(frame["session"].dt.strftime("%Y-%m-%d"))
    missing_sessions = sorted(set(schedule_by_date).difference(observed_dates))
    if missing_sessions:
        issues.append(
            SessionIssue("missing_session", len(missing_sessions), "|".join(missing_sessions))
        )

    duplicate = frame.duplicated(["session", "bar_start", "bar_end"], keep=False)
    if duplicate.any():
        issues.append(SessionIssue("duplicate_bar", int(duplicate.sum()), "duplicate timestamps"))

    duration_invalid = frame["bar_end"].sub(frame["bar_start"]).ne(pd.Timedelta(minutes=5))
    if duration_invalid.any():
        issues.append(
            SessionIssue(
                "invalid_bar_duration",
                int(duration_invalid.sum()),
                "five-minute bars must span exactly five minutes",
            )
        )

    finite_positive = frame[["open", "high", "low", "close"]].gt(0.0).all(axis=1)
    impossible = (
        ~finite_positive
        | frame["high"].lt(frame[["open", "close", "low"]].max(axis=1))
        | frame["low"].gt(frame[["open", "close", "high"]].min(axis=1))
    )
    if impossible.any():
        issues.append(SessionIssue("impossible_ohlc", int(impossible.sum()), "OHLC invariant"))

    incomplete = ~frame["fully_completed"].astype(bool)
    if incomplete.any():
        issues.append(
            SessionIssue("incomplete_bar", int(incomplete.sum()), "bar not fully completed")
        )

    outside_count = 0
    gap_count = 0
    unknown_session_count = 0
    for session, group in frame.groupby("session", sort=True, observed=True):
        session_key = pd.Timestamp(cast(Any, session)).strftime("%Y-%m-%d")
        bounds = schedule_by_date.get(session_key)
        if bounds is None:
            unknown_session_count += len(group)
            continue
        market_open, market_close = bounds
        outside_count += int(
            (group["bar_start"].lt(market_open) | group["bar_end"].gt(market_close)).sum()
        )
        unique = group.drop_duplicates(["bar_start", "bar_end"]).sort_values(
            "bar_start", kind="mergesort"
        )
        previous_end = unique["bar_end"].shift(1)
        gap_count += int(unique["bar_start"].ne(previous_end).iloc[1:].sum())
    if unknown_session_count:
        issues.append(
            SessionIssue(
                "session_not_in_calendar",
                unknown_session_count,
                "bar session absent from XNYS calendar",
            )
        )
    if outside_count:
        issues.append(
            SessionIssue(
                "bar_outside_regular_session",
                outside_count,
                "bar crosses official regular-session bounds",
            )
        )
    if gap_count:
        issues.append(SessionIssue("source_gap", gap_count, "non-contiguous five-minute bars"))
    return sorted(issues, key=lambda issue: issue.code)
