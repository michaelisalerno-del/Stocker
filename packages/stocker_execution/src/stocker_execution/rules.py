"""Frozen clocks and completed-bar mathematics. No broker or order dependencies."""

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from statistics import NormalDist, median
from zoneinfo import ZoneInfo

from stocker_execution.config import RULE_VERSION
from stocker_execution.frozen_sources import FROZEN, SOURCE_HASHES

NY = ZoneInfo("America/New_York")
RIGHTS = {"CL": "C", "ES": "P", "GC": "P", "NQ": "P"}


@dataclass(frozen=True)
class Bar:
    at: datetime  # UTC interval start, complete only at at + one minute
    open: float
    high: float
    low: float
    close: float
    volume: float
    average: float | None = None  # not supplied by Saxo charts; never fabricate VWAP
    # Kept for the record only (Saxo's per-minute open interest and trading state; the price
    # stream carries no futures open interest). Nothing frozen reads them.
    interest: float | None = None
    state: str | None = None

    def valid(self) -> bool:
        return (
            self.at.utcoffset() == timedelta(0)
            and self.at.second == self.at.microsecond == 0
            and all(
                math.isfinite(x) and x > 0 for x in (self.open, self.high, self.low, self.close)
            )
            and (self.average is None or math.isfinite(self.average) and self.average > 0)
            and (self.interest is None or math.isfinite(self.interest) and self.interest >= 0)
            and math.isfinite(self.volume)
            and self.volume >= 0
            and self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high
        )


def next_weekday(day: date) -> date:
    day += timedelta(days=1)
    while day.weekday() > 4:
        day += timedelta(days=1)
    return day


def session_day(now: datetime) -> date:
    """The CME session date: after the 18:00 New York open, and at weekends, the next weekday's."""
    local = now.astimezone(NY)
    return next_weekday(local.date()) if local.hour >= 18 or local.weekday() > 4 else local.date()


def clocks(day: date) -> list[datetime]:
    """Every hour of the CME session at which a trade is possible (the user's decision, 2026-10-04,
    amended the same day): the session opens at 18:00 New York Sunday to Thursday and runs to
    17:00 the next weekday. The 18:00 clock has no completed minutes before it (17:00-18:00 is the
    maintenance break) and a 16:00 clock would exit at the 17:00 close, outside every option's
    trading session, so neither could ever enter and both are left out: the clocks are 19:00-23:00
    on those evenings and 00:00-15:00 on weekdays. The original 09:00-15:00 clocks are among them,
    unchanged (docs/CLOCK-SET-AMENDMENT-20261004.md)."""
    weekday = day.weekday()
    hours = list(range(0, 16)) if weekday <= 4 else []
    if weekday == 6 or weekday <= 3:
        hours += range(19, 24)
    return [datetime.combine(day, time(h), NY).astimezone(UTC) for h in hours]


def us_clock(at: datetime) -> bool:
    """The original clocks (09:00-16:00 New York) keep the inherited US-session definitions."""
    return 9 <= at.astimezone(NY).hour <= 16


def session_start(at: datetime) -> datetime:
    """The 18:00 New York open of the CME session containing `at`."""
    local = at.astimezone(NY)
    day = local.date() if local.hour >= 18 else local.date() - timedelta(days=1)
    return datetime.combine(day, time(18), NY).astimezone(UTC)


def next_clock(now: datetime) -> datetime:
    day = now.astimezone(NY).date()
    for offset in range(8):
        for stamp in clocks(day + timedelta(days=offset)):
            if stamp > now:
                return stamp
    raise ValueError("NO_NEXT_CLOCK")


def session_clock(at: datetime) -> bool:
    """The first clock of the CME session (19:00 New York) carries the session trade
    (docs/SESSION-POLICY-RUNTIME-20261009.md): held to 09:25 New York on the next weekday, or
    sold earlier on a wider armed trail (arm 1.50x entry, keep 0.60x the high)."""
    return at.astimezone(NY).hour == 19


def session_exit(at: datetime) -> datetime:
    """09:25 New York on the first weekday after the session clock's evening."""
    day = next_weekday(at.astimezone(NY).date())
    return datetime.combine(day, time(9, 25), NY).astimezone(UTC)


def day_clock(at: datetime) -> bool:
    """The 10:00 New York clock carries the DAY trade (the user's decision, 2026-10-09): the end
    of the open's first pullback, on the NEXT trading day's expiry (never the same day's), held
    to 15:00 New York or sold earlier on the session trail."""
    return at.astimezone(NY).hour == 10


def day_exit(at: datetime) -> datetime:
    local = at.astimezone(NY)
    return datetime.combine(local.date(), time(15), NY).astimezone(UTC)


def day_expiry_floor(at: datetime) -> datetime:
    """Expiries at or before the 17:00 New York close of the clock's day are not the DAY trade's."""
    local = at.astimezone(NY)
    return datetime.combine(local.date(), time(17), NY).astimezone(UTC)


SESSION_TRAIL = (1.50, 0.60)
HOURLY_TRAIL = (1.20, 0.75)
DAY_VETO_MARKETS = ("ES", "NQ")
DAY_VETO_BP = 10.0


def day_against_veto(bars: list[Bar], at: datetime, market: str) -> str:
    """Skip the DAY trade when the open went against it (the user's decision, 2026-10-09; the
    look in STUDY_LOOK.md: ES/NQ lose 23-26 bp from 10:00 to 15:00 on such days, GC/CL show
    nothing). Over the 30 completed one-minute bars from 09:30 New York, every close sat at
    least DAY_VETO_BP basis points against the market's direction relative to the 09:30 open.
    Any missing bar means no veto."""
    if market not in DAY_VETO_MARKETS or not day_clock(at):
        return ""
    local = at.astimezone(NY)
    start = datetime.combine(local.date(), time(9, 30), NY).astimezone(UTC)
    by_time = {b.at: b for b in bars}
    window = [by_time.get(start + timedelta(minutes=i)) for i in range(30)]
    if any(b is None for b in window):
        return ""
    sign = 1 if RIGHTS[market] == "C" else -1
    open_ = window[0].open
    best = max((b.close - open_) * sign / open_ * 1e4 for b in window)
    return "DAY_OPEN_AGAINST" if best <= -DAY_VETO_BP else ""


def opportunity(market: str, con_id: int, at: datetime) -> dict[str, object]:
    local = at.astimezone(NY)
    if at not in clocks(local.date()):
        raise ValueError("OUTSIDE_ENTRY_WINDOW")
    session, day = session_clock(at), day_clock(at)
    if session:
        exit_at, policy = session_exit(at), "SESSION"
    elif day:
        exit_at, policy = day_exit(at), "DAY"
    else:
        exit_at, policy = at + timedelta(minutes=60), "HOURLY"
    return {
        "id": f"{RULE_VERSION}|{market}|{con_id}|{at.isoformat()}",
        "market": market,
        "rule_version": RULE_VERSION,
        "signal_con_id": con_id,
        "signal_at": at.isoformat(),
        "purchase_at": at.isoformat(),
        "exit_at": exit_at.isoformat(),
        "policy": policy,
        "trail": list(SESSION_TRAIL if policy != "HOURLY" else HOURLY_TRAIL),
        **({"expiry_after": day_expiry_floor(at).isoformat()} if day else {}),
        "right": RIGHTS[market],
        "target_delta": 0.1,
        "veto": "",
        "frozen_definition": FROZEN[market],
        "source_hashes": SOURCE_HASHES,
    }


def prior_rv(
    bars: list[Bar], at: datetime, count: int = 15, index: dict[datetime, Bar] | None = None
) -> float:
    # Every looked-up bar starts at least one minute before `at`, so it is completed.
    by_time = index if index is not None else {b.at: b for b in bars}
    needed = [by_time.get(at - timedelta(minutes=i)) for i in range(count + 1, 0, -1)]
    if any(b is None or not b.valid() for b in needed):
        raise ValueError("INCOMPLETE_COMPLETED_HISTORY")
    closes = [b.close for b in needed if b is not None]
    return math.sqrt(sum(math.log(b / a) ** 2 for a, b in zip(closes, closes[1:], strict=False)))


def model_delta(
    futures: float, strike: float, rv15: float, at: datetime, expiry: datetime, right: str
) -> float:
    v = rv15 * math.sqrt((expiry - at).total_seconds() / (60 * 15))
    if min(v, futures, strike) <= 0:
        raise ValueError("INVALID_FROZEN_PRICING_INPUT")
    return NormalDist().cdf((1 if right == "C" else -1) * (math.log(futures / strike) / v + v / 2))


def eligibility(
    bars: list[Bar], at: datetime, last_trade: date, reference: list[dict[int, dict[str, float]]]
) -> dict[str, float]:
    """The inherited finite-feature gate, not an added predictive filter.

    Five previous selected-contract source sessions supply hourly medians.
    Source session VWAP starts at 08:00 NY; gaps are never filled.
    The derived finite checks reduce algebraically to these positive denominators.
    """
    if next_weekday(at.astimezone(NY).date()) >= last_trade:
        raise ValueError("INHERITED_FUTURES_EXPIRY_GUARD")
    if len(reference) < 5:
        raise ValueError("WARMING_UP_FIVE_REFERENCE_SESSIONS")
    prefix = sorted((b for b in bars if b.at < at), key=lambda b: b.at)
    prior_rv(prefix, at, 30)  # requires 31 contiguous completed closes
    recent = prefix[-31:]
    if any(not b.valid() for b in recent):
        raise ValueError("INVALID_OHLCV")
    rv15 = prior_rv(prefix, at)
    prior15 = prior_rv(prefix, at - timedelta(minutes=5))
    hour = at.astimezone(NY).hour
    norms = {}
    for field in ("rv15", "range15", "volume15"):
        values = [r.get(hour, {}).get(field, math.nan) for r in reference[-5:]]
        # pandas median ignores missing observations, but an all-missing median is invalid.
        values = [v for v in values if math.isfinite(v)]
        norms[field] = median(values) if values else math.nan

    def span(items: list[Bar]) -> float:
        return max(b.high for b in items) - min(b.low for b in items)

    if us_clock(at):
        session = [
            b
            for b in prefix
            if b.at.astimezone(NY).date() == at.astimezone(NY).date()
            and b.at.astimezone(NY).hour >= 8
        ]
        opening = [
            b for b in session if b.at.astimezone(NY).hour == 8 and b.at.astimezone(NY).minute < 30
        ]
    else:  # the other clocks: the CME session since its 18:00 open, opening = its first 30 minutes
        start = session_start(at)
        session = [b for b in prefix if b.at >= start]
        opening = [b for b in session if b.at < start + timedelta(minutes=30)]
    denominators = [
        rv15,
        prior_rv(prefix, at, 5),
        prior15,
        span(recent[-15:]),
        span(recent[-30:]),
        span(recent[-20:-5]),
        sum(b.volume for b in recent[-15:]),
        sum(b.volume for b in recent[-20:-5]),
        sum(b.volume for b in session),
        *norms.values(),
    ]
    if (
        not opening
        or not session
        or span(session) <= 0
        or any(not math.isfinite(x) or x <= 0 for x in denominators)
    ):
        raise ValueError("INHERITED_FEATURE_AVAILABILITY_GATE")
    return {
        "rv15": rv15,
        "futures_price": recent[-1].close,
        "volume15": sum(b.volume for b in recent[-15:]),
        "range3": span(recent[-3:]) / recent[-1].close,
        "volume_observation_only": 1.0,
    }


def observation(
    bars: list[Bar], at: datetime, references: list[dict[int, dict[str, float]]]
) -> dict[str, float | int | str | None]:
    """Observation only, never a gate: the ingredients for judging each clock afterwards.

    rv60 and the session's travel use completed one-minute bars only; a gap leaves rv60 unset
    and is never bridged. The hour's reference rv15 is the median of the reference sessions.
    """
    by_time = {b.at: b for b in bars}
    try:
        rv60: float | None = prior_rv(bars, at, 60, by_time)
    except ValueError:
        rv60 = None
    day = at.astimezone(NY).date()
    start = session_start(at)
    session = sorted(
        (
            b
            for b in bars
            if b.at < at
            and b.valid()
            and (
                ((s := b.at.astimezone(NY)).date() == day and s.hour >= 8)
                if us_clock(at)
                else b.at >= start
            )
        ),
        key=lambda b: b.at,
    )
    pairs = [
        (a, b)
        for a, b in zip(session, session[1:], strict=False)
        if b.at - a.at == timedelta(minutes=1)
    ]
    medians = [
        v
        for r in references[-5:]
        if math.isfinite(v := r.get(at.astimezone(NY).hour, {}).get("rv15", math.nan))
    ]
    return {
        "rv60": rv60,
        "session_travel_since_0800": math.sqrt(
            sum(math.log(b.close / a.close) ** 2 for a, b in pairs)
        )
        if pairs
        else None,
        "session_minutes_counted": len(pairs),
        # Travel counts from 08:00 at the original clocks, from the 18:00 session open otherwise.
        "session_travel_from": "08:00" if us_clock(at) else "18:00",
        "hour_reference_rv15_median": median(medians) if medians else None,
    }


def reference_summary(bars: list[Bar]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    by_time = {b.at: b for b in bars}
    for hour in [h for h in range(24) if h != 17]:  # every session hour (17:00-18:00 is the break)
        samples: dict[str, list[float]] = {k: [] for k in ("rv15", "range15", "volume15")}
        for bar in bars:
            if bar.at.astimezone(NY).hour != hour:
                continue
            try:
                rv = prior_rv(bars, bar.at, index=by_time)
                prior = [by_time[bar.at - timedelta(minutes=i)] for i in range(1, 16)]
            except (KeyError, ValueError):
                continue
            samples["rv15"].append(rv)
            samples["range15"].append(
                (max(b.high for b in prior) - min(b.low for b in prior))
                / by_time[bar.at - timedelta(minutes=1)].close
            )
            samples["volume15"].append(sum(b.volume for b in prior))
        if 8 <= hour <= 16 or any(samples.values()):  # US hours always, as before; others if seen
            result[hour] = {k: median(v) for k, v in samples.items() if v}
    return result
