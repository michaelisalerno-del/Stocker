"""Frozen clocks and completed-bar mathematics. No broker or order dependencies."""

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from statistics import NormalDist, median
from zoneinfo import ZoneInfo

from stocker_execution.config import RULE_VERSION
from stocker_execution.frozen_sources import FROZEN, SOURCE_HASHES

NY = ZoneInfo("America/New_York")
RIGHTS = {"CL": "C", "GC": "P", "NG": "P", "NQ": "P", "SI": "P"}


@dataclass(frozen=True)
class Bar:
    at: datetime  # UTC interval start, complete only at at + one minute
    open: float
    high: float
    low: float
    close: float
    volume: float
    average: float | None = None  # not supplied by Saxo charts; never fabricate VWAP

    def valid(self) -> bool:
        return (
            self.at.utcoffset() == timedelta(0)
            and self.at.second == self.at.microsecond == 0
            and all(
                math.isfinite(x) and x > 0 for x in (self.open, self.high, self.low, self.close)
            )
            and (self.average is None or math.isfinite(self.average) and self.average > 0)
            and math.isfinite(self.volume)
            and self.volume >= 0
            and self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high
        )


def next_weekday(day: date) -> date:
    day += timedelta(days=1)
    while day.weekday() > 4:
        day += timedelta(days=1)
    return day


def clocks(day: date) -> list[datetime]:
    if day.weekday() > 4:
        return []
    return [datetime.combine(day, time(h), NY).astimezone(UTC) for h in range(9, 17)]


def next_clock(now: datetime) -> datetime:
    day = now.astimezone(NY).date()
    for offset in range(8):
        for stamp in clocks(day + timedelta(days=offset)):
            if stamp > now:
                return stamp
    raise ValueError("NO_NEXT_CLOCK")


def opportunity(market: str, con_id: int, at: datetime) -> dict[str, object]:
    local = at.astimezone(NY)
    if at not in clocks(local.date()):
        raise ValueError("OUTSIDE_ENTRY_WINDOW")
    return {
        "id": f"{RULE_VERSION}|{market}|{con_id}|{at.isoformat()}",
        "market": market,
        "rule_version": RULE_VERSION,
        "signal_con_id": con_id,
        "signal_at": at.isoformat(),
        "purchase_at": at.isoformat(),
        "exit_at": (at + timedelta(minutes=60)).isoformat(),
        "right": RIGHTS[market],
        "target_delta": 0.2 if market == "SI" else 0.1,
        "veto": "NG_CLOCK_13" if market == "NG" and local.hour == 13 else "",
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


def frozen_strike(
    futures: float, rv15: float, at: datetime, expiry: datetime, right: str, delta: float
) -> float:
    years = (expiry - at).total_seconds() / (365 * 86400)
    if not (years > 0 and rv15 > 0 and futures > 0 and right in {"C", "P"}):
        raise ValueError("INVALID_FROZEN_PRICING_INPUT")
    sigma = rv15 * math.sqrt(525600 / 15)
    v = sigma * math.sqrt(years)
    sign = 1 if right == "C" else -1
    return futures * math.exp(0.5 * v * v - sign * NormalDist().inv_cdf(delta) * v)


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

    session = [
        b
        for b in prefix
        if b.at.astimezone(NY).date() == at.astimezone(NY).date() and b.at.astimezone(NY).hour >= 8
    ]
    opening = [
        b for b in session if b.at.astimezone(NY).hour == 8 and b.at.astimezone(NY).minute < 30
    ]
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


def reference_summary(bars: list[Bar]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    by_time = {b.at: b for b in bars}
    for hour in range(8, 17):
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
        result[hour] = {k: median(v) for k, v in samples.items() if v}
    return result
