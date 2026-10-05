import math
from datetime import UTC, datetime, timedelta

import pytest

from stocker_execution import forecast
from stocker_execution.rules import NY, Bar


def at_ny(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=NY).astimezone(UTC)


def bars_moving(clock: datetime, scale: float, market: str = "NQ") -> list[Bar]:
    """Every minute moves `scale` times its normal, so each look14 ingredient reads `scale`."""
    start = forecast.session_open(clock)
    profile, close, out = forecast.PROFILES[market], 100.0, []
    for k in range(int((clock - start).total_seconds() // 60)):
        if k:
            close *= math.exp((-1) ** k * scale * math.sqrt(profile[k]))
        out.append(Bar(start + timedelta(minutes=k), close, close, close, close, 1))
    return out


def test_level_is_the_frozen_blend_of_movement_against_normal():
    clock = at_ny(1, 10)
    seen = forecast.level(bars_moving(clock, 1.5), clock, "NQ")
    assert seen["status"] == "OBSERVED"
    for name in ("today", "last_hour", "last_15_minutes"):
        assert seen[name] == pytest.approx(1.5)
    assert seen["level"] == pytest.approx(0.97 * 1.5 ** (0.19 + 0.36 + 0.13))


def test_a_minute_without_trades_is_flat_and_the_next_bar_spans_it():
    clock = at_ny(1, 10)
    bars = bars_moving(clock, 1.0)
    gap = [b for b in bars if b.at != clock - timedelta(minutes=10)]
    seen = forecast.level(gap, clock, "NQ")
    # Two alternating moves merge into one smaller one: the hour reads below normal.
    assert seen["status"] == "OBSERVED" and seen["last_hour"] < 1.0


def test_level_needs_the_last_minute_and_an_hour_of_bars():
    clock = at_ny(1, 10)
    bars = bars_moving(clock, 1.0)
    assert forecast.level(bars[:-1], clock, "NQ")["reason"] == "LAST_MINUTE_MISSING"
    assert forecast.level(bars[-50:], clock, "NQ")["reason"] == "LESS_THAN_AN_HOUR_OF_BARS"


def test_normal_variance_counts_weekday_sessions_to_a_later_expiry():
    p = forecast.PROFILES["NG"]
    k = lambda hour, minute=0: (hour - 18) % 24 * 60 + minute  # noqa: E731
    thu, fri = at_ny(1, 15), at_ny(2, 14, 30)
    assert forecast.normal_variance("NG", thu, at_ny(1, 16)) == pytest.approx(sum(p[k(15) : k(16)]))
    assert forecast.normal_variance("NG", thu, fri) == pytest.approx(
        sum(p[k(15) :]) + sum(p[: k(14, 30)])
    )
    # Friday afternoon to Monday: the weekend adds nothing; to Tuesday adds Monday's session.
    assert forecast.normal_variance("NG", fri, at_ny(5, 14, 30)) == pytest.approx(
        sum(p[k(14, 30) :]) + sum(p[: k(14, 30)])
    )
    assert forecast.normal_variance("NG", fri, at_ny(6, 14, 30)) == pytest.approx(
        sum(p[k(14, 30) :]) + sum(p) + sum(p[: k(14, 30)])
    )


def test_an_option_priced_at_the_forecast_reads_one():
    clock = at_ny(1, 13)
    bars = bars_moving(clock, 1.2)
    level = forecast.level(bars, clock, "NQ")["level"]
    expiry = at_ny(1, 16)
    move = level * math.sqrt(forecast.normal_variance("NQ", clock, expiry))
    fair = forecast.black(30800.0, 30600.0, move, call=False)
    option = {
        "identity": {
            "uic": 7,
            "right": "Put",
            "strike": 30600.0,
            "expiry_instant": expiry.isoformat(),
        },
        "quote": {"Bid": fair - 0.25, "Ask": fair + 0.25},
        "quote_status": "OBSERVED",
    }
    seen = forecast.observe(bars, clock, "NQ", option, {"Bid": 30799.75, "Ask": 30800.25})
    assert seen["option"]["implied_over_forecast"] == pytest.approx(1.0, abs=0.01)
    assert seen["option"]["ask_over_forecast_fair"] == pytest.approx((fair + 0.25) / fair)
    stale = {**option, "quote_status": "STALE_OR_MISSING"}
    assert forecast.observe(bars, clock, "NQ", stale, {})["option"]["reason"] == "QUOTE_NOT_USABLE"


def test_break_even_is_the_move_that_gets_the_bid_back_to_the_ask():
    """Display only (2026-10-05): the move a purchase at the ask needs (the frozen veto's)."""
    forward, strike, hours = 100.0, 102.0, 6.0
    mid = forecast.black(forward, strike, 0.02, True)
    bid, ask = mid * 0.9, mid * 1.1
    move = forecast.break_even_move(bid, ask, forward, strike, True, hours)
    assert move is not None and move > 0
    # At that move the model bid after an hour equals the ask paid.
    after = forecast.implied_move(mid, forward, strike, True) * math.sqrt((hours - 1) / hours)
    assert forecast.black(forward * math.exp(move), strike, after, True) - (
        mid - bid
    ) == pytest.approx(ask, rel=1e-6)
    # A wider spread or a fall in implied volatility needs a bigger move; a put needs it downwards.
    assert forecast.break_even_move(mid * 0.8, mid * 1.2, forward, strike, True, hours) > move
    assert forecast.break_even_move(bid, ask, forward, strike, True, hours, iv_scale=0.9) > move
    put_mid = forecast.black(forward, 98.0, 0.02, False)
    assert forecast.break_even_move(put_mid * 0.9, put_mid * 1.1, forward, 98.0, False, hours) > 0
    # An option expiring inside the hour is worth its intrinsic value; no implied move, no answer.
    assert forecast.break_even_move(bid, ask, forward, strike, True, 0.5) > math.log(
        strike / forward
    )
    assert forecast.break_even_move(0.0, 0.0, forward, strike, True, hours) is None


def test_break_even_view_reads_in_normal_hours_and_names_what_is_missing():
    at = datetime(2026, 10, 6, 14, tzinfo=UTC)
    identity = {
        "uic": 1,
        "right": "Put",
        "strike": 98.0,
        "expiry_instant": (at + timedelta(hours=6)).isoformat(),
    }
    mid = forecast.black(100.0, 98.0, 0.02, False)
    option = {
        "identity": identity,
        "quote_status": "OBSERVED",
        "quote": {"Bid": mid * 0.9, "Ask": mid * 1.1},
    }
    futures = {"Bid": 99.99, "Ask": 100.01}
    view = forecast.break_even_view(option, futures, 0.004, at)
    assert view["ratio"] > 0 and 0 < view["chance"] < 0.5 and view["ratio_iv_down"] > view["ratio"]
    assert forecast.break_even_view(option, futures, None, at) == {"reason": "NORMAL_HOUR_UNKNOWN"}
    assert (
        forecast.break_even_view(
            {**option, "quote_status": "STALE_OR_MISSING"}, futures, 0.004, at
        )["reason"]
        == "QUOTE_NOT_USABLE"
    )
