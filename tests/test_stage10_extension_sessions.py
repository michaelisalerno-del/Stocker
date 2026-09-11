from __future__ import annotations

from datetime import UTC, datetime

from stocker_core.markets import MarketId, get_market
from stocker_core.runs import Environment, RunConfig, RunWindow
from stocker_execution.runtime import ExchangeSessionResolver


def test_schedule_is_reused_but_clock_window_and_exchange_day_are_current(monkeypatch):
    from datetime import time

    import stocker_execution.runtime as module

    original = module.get_market_calendar
    calls = []

    def counted(name):
        calls.append(name)
        return original(name)

    monkeypatch.setattr(module, "get_market_calendar", counted)
    resolver = ExchangeSessionResolver()
    run = run_for(MarketId.US_NASDAQ)
    states = [
        resolver.resolve(run, datetime(2026, 11, 27, hour, tzinfo=UTC)) for hour in [13, 15, 19]
    ]
    assert [r.state.value for r in states] == ["BEFORE_SESSION", "ACTIVE_SESSION", "AFTER_SESSION"]
    assert all(r.closes_at == datetime(2026, 11, 27, 18, tzinfo=UTC) for r in states)
    later = run.model_copy(update={"session": run.session.model_copy(update={"start": time(10)})})
    assert resolver.resolve(later, datetime(2026, 11, 27, 15, tzinfo=UTC)).opens_at == datetime(
        2026, 11, 27, 15, tzinfo=UTC
    )
    assert len(calls) == 1, "The same exchange day must not rebuild its calendar on every cycle"
    holiday = resolver.resolve(run, datetime(2026, 11, 26, 15, tzinfo=UTC))
    assert holiday.state.value == "CLOSED_DAY"
    summer = resolver.resolve(run, datetime(2026, 10, 30, 15, tzinfo=UTC))
    winter = resolver.resolve(run, datetime(2026, 11, 2, 15, tzinfo=UTC))
    assert summer.opens_at.hour == 13 and winter.opens_at.hour == 14
    assert len(calls) == 4


def run_for(market_id: MarketId) -> RunConfig:
    market = get_market(market_id)
    return RunConfig(
        run_id=market_id.value,
        universe="TEST",
        strategy="SESSION_HARD_HV",
        environment=Environment.PAPER,
        session=RunWindow(
            start=market.regular_sessions[0].opens_at,
            end=market.regular_sessions[-1].closes_at,
            timezone=market.timezone,
            calendar=market.calendar,
        ),
    )


def test_us_checkpoint_timestamps_remain_the_canonical_sequence() -> None:
    session = ExchangeSessionResolver().resolve(
        run_for(MarketId.US_NASDAQ), datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
    )
    assert [(count, timestamp.isoformat()) for count, timestamp in session.checkpoint_times()] == [
        (6, "2026-09-02T14:00:00+00:00"),
        (8, "2026-09-02T14:10:00+00:00"),
        (10, "2026-09-02T14:20:00+00:00"),
        (12, "2026-09-02T14:30:00+00:00"),
        (14, "2026-09-02T14:40:00+00:00"),
        (16, "2026-09-02T14:50:00+00:00"),
        (18, "2026-09-02T15:00:00+00:00"),
        (20, "2026-09-02T15:10:00+00:00"),
        (22, "2026-09-02T15:20:00+00:00"),
        (24, "2026-09-02T15:30:00+00:00"),
        (26, "2026-09-02T15:40:00+00:00"),
        (28, "2026-09-02T15:50:00+00:00"),
        (30, "2026-09-02T16:00:00+00:00"),
        (32, "2026-09-02T16:10:00+00:00"),
        (34, "2026-09-02T16:20:00+00:00"),
    ]


def test_hong_kong_lunch_break_has_no_slots_and_bar_count_resumes_after_lunch() -> None:
    session = ExchangeSessionResolver().resolve(
        run_for(MarketId.HONG_KONG_HKEX), datetime(2026, 9, 2, 4, 0, tzinfo=UTC)
    )
    checkpoints = dict(session.checkpoint_times())
    assert checkpoints[28].isoformat() == "2026-09-02T03:50:00+00:00"
    assert checkpoints[30].isoformat() == "2026-09-02T05:00:00+00:00"
    assert checkpoints[34].isoformat() == "2026-09-02T05:20:00+00:00"
    assert not any(
        datetime(2026, 9, 2, 4, 0, tzinfo=UTC) <= timestamp < datetime(2026, 9, 2, 5, 0, tzinfo=UTC)
        for timestamp in session.active_bar_starts
    )
