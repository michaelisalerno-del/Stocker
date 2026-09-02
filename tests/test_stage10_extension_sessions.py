from __future__ import annotations

from datetime import UTC, date, datetime
from math import isclose

from stocker_core.markets import MarketId, get_market
from stocker_core.runs import Environment, RunConfig, RunWindow
from stocker_execution.pre_context import (
    calculate_volatility,
    pre_context_bar_starts,
    pre_context_session_contract,
)
from stocker_execution.runtime import ExchangeSessionResolver


def run_for(market_id: MarketId) -> RunConfig:
    market = get_market(market_id)
    return RunConfig(
        run_id=market_id.value,
        universe="TEST",
        strategy="SESSION_HARD",
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


def test_market_local_previous_session_contract_is_timezone_independent() -> None:
    observation, previous_open, previous_close, target_open = pre_context_session_contract(
        date(2026, 9, 2), calendar="XLON"
    )
    assert observation == date(2026, 9, 1)
    assert previous_open.tzinfo is UTC
    assert previous_close.tzinfo is UTC
    assert target_open.tzinfo is UTC
    assert previous_open.hour == 7
    assert previous_close.hour == 15 and previous_close.minute == 30


def test_prior_session_history_excludes_exchange_lunch_break() -> None:
    starts = pre_context_bar_starts(date(2026, 9, 2), calendar="XHKG")

    assert len(starts) == 66
    assert starts[0].isoformat() == "2026-09-01T01:30:00+00:00"
    assert starts[29].isoformat() == "2026-09-01T03:55:00+00:00"
    assert starts[30].isoformat() == "2026-09-01T05:00:00+00:00"
    assert starts[-1].isoformat() == "2026-09-01T07:55:00+00:00"


def test_us_volatility_clock_is_numerically_unchanged_and_non_us_uses_active_minutes() -> None:
    us = calculate_volatility(call_model_iv=0.25, put_model_iv=0.27)
    explicit_us = calculate_volatility(
        call_model_iv=0.25,
        put_model_iv=0.27,
        annual_regular_trading_minutes=252 * 390,
    )
    london = calculate_volatility(
        call_model_iv=0.25,
        put_model_iv=0.27,
        annual_regular_trading_minutes=252 * get_market(MarketId.UK_LSE).active_regular_minutes,
    )
    assert us == explicit_us
    assert isclose(us.expected_absolute_return_15m, 0.0025628693370347896)
    assert london.expected_absolute_return_15m != us.expected_absolute_return_15m
