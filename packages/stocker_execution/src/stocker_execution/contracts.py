"""Listed identity, exchange calendars, quote quality and cash debit checks."""

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from stocker_execution.config import ProductMapping

EXCHANGES = {"BTC": "CME", "CL": "NYMEX", "GC": "COMEX", "NG": "NYMEX", "NQ": "CME", "SI": "COMEX"}


def utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("NAIVE_BROKER_TIMESTAMP")
    return value.astimezone(UTC)


def zoned(stamp: str, timezone: str) -> datetime:
    zone = ZoneInfo(timezone)
    naive = datetime.strptime(stamp, "%Y%m%d:%H%M")
    first = naive.replace(tzinfo=zone, fold=0)
    second = naive.replace(tzinfo=zone, fold=1)
    if first.utcoffset() != second.utcoffset():
        raise ValueError("AMBIGUOUS_OR_NONEXISTENT_EXCHANGE_TIME")
    return first.astimezone(UTC)


@dataclass(frozen=True)
class Calendar:
    sessions: tuple[tuple[datetime, datetime, str], ...]
    covered_dates: tuple[str, ...]
    timezone: str
    observed_at: datetime

    @classmethod
    def from_details(cls, details: Any, now: datetime) -> "Calendar":
        sessions, dates = [], []
        for row in details.tradingHours.split(";"):
            if not row:
                continue
            day, times = row.split(":", 1)
            dates.append(day)
            if times == "CLOSED":
                continue
            for part in times.split(","):
                start, end = part.split("-")
                start = start if ":" in start else day + ":" + start
                end = end if ":" in end else day + ":" + end
                a, b = zoned(start, details.timeZoneId), zoned(end, details.timeZoneId)
                if b <= a:
                    raise ValueError("INVALID_EXCHANGE_SESSION")
                # tradingHours dates label intervals, not authoritative clearing trade dates.
                sessions.append((a, b, ""))
        if not dates:
            raise ValueError("CONTRACT_CALENDAR_UNAVAILABLE")
        return cls(tuple(sorted(sessions)), tuple(dates), details.timeZoneId, now)

    def state(self, now: datetime) -> tuple[str, datetime | None, str | None]:
        if not 0 <= (now - self.observed_at).total_seconds() <= 86400:
            return "BLOCKED", None, None
        for start, end, trade_date in self.sessions:
            if start <= now < end:
                return "OPEN", end, trade_date or None
        upcoming = next((a for a, _, _ in self.sessions if a > now), None)
        day = now.astimezone(ZoneInfo(self.timezone)).strftime("%Y%m%d")
        if day not in self.covered_dates:
            return "BLOCKED", upcoming, None
        before = any(b <= now for _, b, _ in self.sessions)
        if before and upcoming and upcoming - now <= timedelta(hours=4):
            return "MAINTENANCE", upcoming, None
        return "CLOSED", upcoming, None

    def supports(self, start: datetime, end: datetime) -> bool:
        return any(a <= start < end < b for a, b, _ in self.sessions)


def nearby_futures(details: list[Any], day: date) -> list[Any]:
    """Use the same bounded candidate rule independently for each source date."""
    return sorted(
        (
            d
            for d in details
            if d.contract.secType == "FUT"
            and d.contract.conId > 0
            and len(d.contract.lastTradeDateOrContractMonth) >= 8
            and d.contract.lastTradeDateOrContractMonth[:8] > day.strftime("%Y%m%d")
        ),
        key=lambda d: (d.contract.lastTradeDateOrContractMonth, d.contract.conId),
    )[:6]


def select_future(
    candidates: list[tuple[Any, date, float]], previous_session: date, today: date
) -> Any:
    """Prior completed session volume; same-day observations never rank contracts."""
    valid = []
    for details, volume_date, volume in candidates:
        c = details.contract
        expiry = date.fromisoformat(
            datetime.strptime(c.lastTradeDateOrContractMonth[:8], "%Y%m%d").date().isoformat()
        )
        if c.secType != "FUT" or c.conId <= 0 or expiry <= today:
            continue
        if volume_date != previous_session or not math.isfinite(volume) or volume < 0:
            raise ValueError("ROLLOVER_PRIOR_SESSION_VOLUME_UNAVAILABLE")
        valid.append((details, volume))
    if not valid:
        raise ValueError("ROLLOVER_PRIOR_SESSION_VOLUME_UNAVAILABLE")
    return min(valid, key=lambda x: (-x[1], x[0].contract.conId))[0]


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_at: datetime
    ask_at: datetime
    data_type: int

    def validate(self, now: datetime) -> None:
        if self.data_type != 1:
            raise ValueError("OPTION_QUOTE_NOT_REALTIME")
        if any(not math.isfinite(x) or x <= 0 for x in (self.bid, self.ask)):
            raise ValueError("OPTION_QUOTE_MISSING")
        if self.bid > self.ask:
            raise ValueError("OPTION_QUOTE_CROSSED")
        if any(not 0 <= (now - utc(t)).total_seconds() <= 5 for t in (self.bid_at, self.ask_at)):
            raise ValueError("OPTION_QUOTE_STALE")


def tick_price(price: float, rules: list[tuple[float, float]], buy: bool) -> float:
    if not math.isfinite(price) or price <= 0 or not rules:
        raise ValueError("PRICE_INCREMENT_UNVERIFIED")
    eligible = [(edge, tick) for edge, tick in rules if edge <= price and tick > 0]
    if not eligible:
        raise ValueError("PRICE_INCREMENT_UNVERIFIED")
    tick = Decimal(str(max(eligible)[1]))
    # Entries never raise the ask/debit to the next tick; exits round down as well.
    return float((Decimal(str(price)) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick)


def budget(
    price: float,
    multiplier: float,
    factor: float,
    gbp_per_unit: float,
    fx_at: datetime,
    now: datetime,
    fee_gbp: float,
) -> dict[str, float | int | str]:
    if any(
        not math.isfinite(x) or x <= 0 for x in (price, multiplier, factor, gbp_per_unit, fee_gbp)
    ):
        raise ValueError("BUDGET_INPUT_UNVERIFIED")
    if not 0 <= (now - utc(fx_at)).total_seconds() <= 30:
        raise ValueError("FX_STALE_OR_UNAVAILABLE")
    native = Decimal(str(price)) * Decimal(str(multiplier)) * Decimal(str(factor))
    premium = native * Decimal(str(gbp_per_unit))
    total = premium + Decimal(str(fee_gbp))
    pennies = int((total * 100).to_integral_value(rounding=ROUND_CEILING))
    if pennies > 1000:
        raise ValueError("SKIP_BUDGET_TOO_SMALL")
    return {
        "quantity": 1,
        "premium_native": float(native),
        "premium_gbp": float(premium),
        "fee_reserve_gbp": fee_gbp,
        "total_entry_cash_gbp": float(total),
        "cash_pennies": pennies,
        "allocation_pennies": 1000,
        "fx_rate": gbp_per_unit,
        "fx_at": fx_at.isoformat(),
    }


def verify_option(
    details: Any,
    future_id: int,
    market: str,
    mapping: ProductMapping,
    now: datetime,
    exit_at: datetime,
    right: str,
) -> tuple[datetime, Calendar]:
    c = details.contract
    if (
        c.secType != "FOP"
        or c.conId <= 0
        or details.underConId != future_id
        or c.symbol != mapping.symbol
        or c.exchange != mapping.exchange
        or c.tradingClass != mapping.trading_class
        or c.currency != mapping.currency
        or float(c.multiplier) != mapping.multiplier
        or details.priceMagnifier != mapping.price_magnifier
        or c.right != right
    ):
        raise ValueError("OPTION_IDENTITY_OR_UNDERLYING_MISMATCH")
    if not details.realExpirationDate or not details.lastTradeTime or not details.timeZoneId:
        raise ValueError("OPTION_TERMINATION_UNVERIFIED")
    stamp = datetime.strptime(
        details.realExpirationDate + " " + details.lastTradeTime, "%Y%m%d %H:%M:%S"
    )
    expiry = stamp.replace(tzinfo=ZoneInfo(details.timeZoneId)).astimezone(UTC)
    local = expiry.astimezone(ZoneInfo(mapping.expiry_timezone))
    if local.strftime("%H:%M:%S") != mapping.termination_time:
        raise ValueError("OPTION_TERMINATION_MAPPING_MISMATCH")
    if local.date() != now.astimezone(ZoneInfo(mapping.expiry_timezone)).date() or expiry <= now:
        raise ValueError("NO_REAL_0DTE_MATCH")
    # BTC products retain independent clocks; the source never authorises substitution.
    if market == "BTC":
        expected = {
            "BTC": ("Europe/London", "16:00:00"),
            "MBT": ("Europe/London", "16:00:00"),
            "BFF": ("America/New_York", "16:00:00"),
        }.get(mapping.product)
        if expected != (mapping.expiry_timezone, mapping.termination_time):
            raise ValueError("BTC_PRODUCT_CUTOFF_UNVERIFIED")
    calendar = Calendar.from_details(details, now)
    if expiry <= exit_at + timedelta(seconds=120) or not calendar.supports(
        now, exit_at + timedelta(seconds=120)
    ):
        raise ValueError("UNSUPPORTED_EXIT_BEFORE_CONTRACT_CUTOFF")
    return expiry, calendar
