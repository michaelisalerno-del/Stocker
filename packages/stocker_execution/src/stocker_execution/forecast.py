"""look14: how far the future should move, from its normal for the time of day. Observation only.

Frozen from the bar research of 2026-09-30 (look14_indicator.py: weights fitted on days 1-40 of the
cached bars, judged on days 41-80). The level is 0.97 x (today so far)^0.19 x (last hour)^0.36 x
(last 15 minutes)^0.13, each the movement against the normal for the same minutes, and the forecast
movement over a later span is the level times the normal for that span. Set against the movement an
option's price implies, it says whether the option is dear or cheap against the forecast. Never an
entry rule.
"""

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from importlib import resources
from statistics import NormalDist
from typing import Any

from stocker_execution.rules import NY, Bar

VERSION = "LOOK14_FORECAST_V1"
SESSION_MINUTES = 1380  # 18:00 -> 16:59 New York, as in the profile
SCALE, TODAY, HOUR, QUARTER = 0.97, 0.19, 0.36, 0.13
_RAW = resources.files("stocker_execution").joinpath("look14_profile.json").read_bytes()
PROFILE_SHA256 = hashlib.sha256(_RAW).hexdigest()
# Normal squared one-minute log return by minute of the session, per market.
PROFILES: dict[str, list[float]] = json.loads(_RAW)["markets"]
N = NormalDist().cdf


def session_open(at: datetime) -> datetime:
    """The 18:00 New York open of the session that contains `at`, in UTC."""
    local = at.astimezone(NY)
    day = local.date() if local.hour >= 18 else local.date() - timedelta(days=1)
    return datetime(day.year, day.month, day.day, 18, tzinfo=NY).astimezone(UTC)


def level(bars: list[Bar], at: datetime, market: str) -> dict[str, Any]:
    profile = PROFILES[market]
    start = session_open(at)
    now_k = int((at - start).total_seconds() // 60)
    closes = {b.at: b.close for b in bars if start <= b.at < at and b.valid()}
    if not 60 < now_k <= SESSION_MINUTES or not closes:
        return {"status": "UNAVAILABLE", "reason": "OUTSIDE_SESSION_OR_NO_BARS"}
    if at - timedelta(minutes=1) not in closes:
        return {"status": "UNAVAILABLE", "reason": "LAST_MINUTE_MISSING"}
    first = int((min(closes) - start).total_seconds() // 60)
    if first > now_k - 61:
        return {"status": "UNAVAILABLE", "reason": "LESS_THAN_AN_HOUR_OF_BARS"}
    # Saxo omits minutes without trades: such a minute is flat and the next bar's return spans
    # it, as in the research grid, which carried the last close forward.
    squares: dict[int, float] = {}
    last = closes[min(closes)]
    for k in range(first + 1, now_k):
        close = closes.get(start + timedelta(minutes=k))
        if close is not None:
            squares[k] = math.log(close / last) ** 2
            last = close

    def against_normal(lo: int) -> float:
        return math.sqrt(
            sum(squares.get(k, 0.0) for k in range(lo, now_k)) / sum(profile[lo:now_k])
        )

    today, hour, quarter = (
        against_normal(first + 1),
        against_normal(now_k - 60),
        against_normal(now_k - 15),
    )
    if not min(today, hour, quarter) > 0:
        return {"status": "UNAVAILABLE", "reason": "NO_MOVEMENT"}
    return {
        "status": "OBSERVED",
        "level": SCALE * today**TODAY * hour**HOUR * quarter**QUARTER,
        "today": today,
        "last_hour": hour,
        "last_15_minutes": quarter,
        # The 1,200 Saxo bars may start after the session's 18:00 open; today counts from there.
        "today_minutes_counted": now_k - first - 1,
    }


def normal_variance(market: str, start: datetime, end: datetime) -> float:
    """The profile's variance from start to end, with a full session for each weekday between.

    Exchange holidays are not known here and count as normal sessions.
    """
    profile = PROFILES[market]
    first, last = session_open(start), session_open(end)
    k0 = int((start - first).total_seconds() // 60)
    k1 = min(int((end - last).total_seconds() // 60), SESSION_MINUTES)
    if first == last:
        return sum(profile[k0:k1])
    # A session is named by the New York date it closes on.
    day, final = ((s.astimezone(NY) + timedelta(hours=6)).date() for s in (first, last))
    between = 0
    while (day := day + timedelta(days=1)) < final:
        between += day.weekday() < 5
    return sum(profile[k0:]) + between * sum(profile) + sum(profile[:k1])


def black(forward: float, strike: float, move: float, call: bool) -> float:
    """Undiscounted Black price for a total standard deviation `move` of the log price."""
    if move <= 0:
        return max(0.0, forward - strike if call else strike - forward)
    d1 = math.log(forward / strike) / move + move / 2
    if call:
        return forward * N(d1) - strike * N(d1 - move)
    return strike * N(move - d1) - forward * N(-d1)


def implied_move(price: float, forward: float, strike: float, call: bool) -> float | None:
    if not price > black(forward, strike, 0.0, call) or black(forward, strike, 2.0, call) < price:
        return None
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if black(forward, strike, mid, call) < price else (lo, mid)
    return (lo + hi) / 2


def option_check(
    option: dict[str, Any], value: float, market: str, at: datetime, futures: dict[str, Any]
) -> dict[str, Any]:
    """The movement to expiry the option's mid implies against the forecast's (both totals)."""
    identity, quote = option["identity"], option.get("quote") or {}
    raw = (quote.get("Bid"), quote.get("Ask"), futures.get("Bid"), futures.get("Ask"))
    prices = [float(x) for x in raw if isinstance(x, (int, float)) and x > 0]
    result: dict[str, Any] = {
        "uic": identity["uic"],
        "right": identity["right"],
        "strike": identity["strike"],
    }
    if option.get("quote_status") != "OBSERVED" or len(prices) < 4:
        return {**result, "reason": "QUOTE_NOT_USABLE"}
    bid, ask, f_bid, f_ask = prices
    if not identity.get("expiry_instant"):
        return {**result, "reason": "EXPIRY_INSTANT_UNKNOWN"}
    expiry = datetime.fromisoformat(identity["expiry_instant"]).astimezone(UTC)
    forward, strike, call = (
        (f_bid + f_ask) / 2,
        float(identity["strike"]),
        identity["right"] == "Call",
    )
    forecast = value * math.sqrt(normal_variance(market, at, expiry))
    implied = implied_move((bid + ask) / 2, forward, strike, call)
    fair = black(forward, strike, forecast, call)
    return {
        **result,
        "futures_mid": forward,
        "bid": bid,
        "ask": ask,
        "expiry_instant": identity["expiry_instant"],
        "forecast_move_to_expiry": forecast,
        "implied_move_to_expiry": implied,
        # Above 1: the option is priced for more movement than the forecast expects.
        "implied_over_forecast": implied / forecast if implied and forecast > 0 else None,
        "forecast_fair_price": fair,
        "ask_over_forecast_fair": ask / fair if fair > 0 else None,
    }


def observe(
    bars: list[Bar],
    at: datetime,
    market: str,
    option: dict[str, Any] | None,
    futures: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "version": VERSION,
        "profile_sha256": PROFILE_SHA256,
        "at": at.isoformat(),
    }
    result.update(level(bars, at, market))
    if result["status"] == "OBSERVED":
        result["next_hour_move"] = result["level"] * math.sqrt(
            normal_variance(market, at, at + timedelta(minutes=60))
        )
        if option:
            result["option"] = option_check(option, result["level"], market, at, futures)
    return result
