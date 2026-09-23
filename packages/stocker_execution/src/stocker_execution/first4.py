"""Frozen September 23 FIRST4 decisions. No broker or synthetic pricing dependencies."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

Q5 = 4.459368321659181
METHOD = "FIRST4_PRIOR15_Q5_98P_102C_20260923"


@dataclass(frozen=True)
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float


def prior15(bars: list[Bar], session_open: datetime, information_at: datetime) -> float | None:
    """Exact historical.excursion and expanded.coverage window/reference semantics."""
    if information_at.tzinfo is None or session_open.tzinfo is None:
        raise ValueError("Timezone-aware clocks required")
    elapsed = (information_at - session_open).total_seconds()
    if elapsed % 60 or elapsed < 900:
        return None
    j = int(elapsed / 60) - 1
    by_time = {b.time: b for b in bars}
    if len(by_time) != len(bars):
        return None
    start = information_at - timedelta(minutes=15)
    window = [by_time.get(start + timedelta(minutes=i)) for i in range(15)]
    if any(b is None for b in window):
        return None
    valid = [b for b in window if b is not None]
    if any(
        not all(isfinite(x) and x > 0 for x in (b.open, b.high, b.low, b.close))
        or b.high < max(b.open, b.close)
        or b.low > min(b.open, b.close)
        for b in valid
    ):
        return None
    previous = by_time.get(start - timedelta(minutes=1))
    ref = valid[0].open if j == 14 else previous.close if previous else float("nan")
    if not isfinite(ref) or ref <= 0:
        return None
    # Preserve the original operation order, including bps then percent.
    up = max(0, (max(b.high for b in valid) / ref - 1) * 10000)
    down = max(0, (1 - min(b.low for b in valid) / ref) * 10000)
    return (up + down) / 100


def eligible(price: float, change_pct: float, prior: float | None) -> str:
    if not isfinite(price) or not 0 < price < 20:
        return "PRICE_OUTSIDE_STRICT_BOUNDARY"
    if not isfinite(change_pct) or change_pct < 5.5:
        return "CHANGE_BELOW_5_5"
    if prior is None or not isfinite(prior):
        return "PRIOR15_UNAVAILABLE"
    return "Q5" if prior > Q5 else "NOT_Q5"


def times(information_at: datetime, session_close: datetime) -> tuple[datetime, datetime, datetime]:
    entry = information_at + timedelta(minutes=1)
    return entry, entry + timedelta(minutes=2880), session_close
