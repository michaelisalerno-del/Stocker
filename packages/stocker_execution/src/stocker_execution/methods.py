"""The frozen paper methods (docs/PAPER-RECORD-PROTOCOL.md). Pure calculations, no broker access.

1. First candle: the 09:30-09:35 New York 5-minute candle of an index CFD against EMA(12) of the
   CFD's 5-minute closes (overnight included) -> long above, short below; a paper CFD trade sized
   so its stop costs $200, held to the 16:00 close.
2. The same Nasdaq signal as a same-day QQQ option (|delta| nearest 0.40): ask at 09:35, bid at
   13:00.
3. Connors on QQQ: close above its 200-day average and RSI2(today) + RSI2(yesterday) < 35 at 15:50
   -> the next expiry's call with delta nearest 0.50, ask at 15:50, bid at 15:55 on its expiry day.
4. VWAP crossings (amendment 2): on a day the Nasdaq first candle is long, each 10:00-15:00 close at
   which QQQ's EMA(10) crosses up through QQQ's session VWAP -> a paper long Nasdaq CFD, out at the
   first later close back below VWAP or at 16:00; one at a time.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
RISK_USD = 200.0


@dataclass(frozen=True)
class Cfd:
    market: str
    uic: int
    name: str
    stop_bp: float


CFDS = (Cfd("NAS100", 4912, "US Tech 100", 100.0), Cfd("US500", 4913, "US 500", 50.0))
SIGNAL_MARKET = "NAS100"  # the QQQ option follows the Nasdaq CFD's signal
QQQ_UIC = 4328771
QQQ_OPTION_ROOT = 306
OPTION_MULTIPLIER = 100

EMA_ALPHA = 2 / 13
PRIOR_WINDOW = timedelta(hours=3)
PRIOR_MIN_BARS = 30  # of the 36 five-minute candles in the three hours before the open
FIRST_CANDLE_DELTA = 0.40
FIRST_CANDLE_DELTA_RANGE = (0.25, 0.55)
CONNORS_DELTA = 0.50
CONNORS_DELTA_RANGE = (0.42, 0.58)
CONNORS_DAYS = 200
BOARD_HALF_WIDTH = 15  # strikes either side of the price saved with each option decision

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
OPTION_EXIT = time(13, 0)


@dataclass(frozen=True)
class Bar:
    start: datetime  # UTC; Saxo stamps a chart sample with its start
    close: float


@dataclass(frozen=True)
class FirstCandle:
    side: int  # +1 long, -1 short, 0 no trade
    close: float | None
    ema: float | None
    prior_bars: int
    reason: str  # empty when the signal is usable


def ema(values: Sequence[float], alpha: float = EMA_ALPHA) -> float:
    """pandas ewm(alpha, adjust=False): seeded with the first value."""
    value = values[0]
    for x in values[1:]:
        value += alpha * (x - value)
    return value


def first_candle(bars: Sequence[Bar], open_at: datetime) -> FirstCandle:
    """`bars` as Saxo returns them; the newest sample is still forming, so the 09:30 candle is
    final only once a later one exists."""
    by_start = {b.start: b for b in bars if b.start <= open_at}
    ordered = [by_start[k] for k in sorted(by_start)]
    if not ordered or ordered[-1].start != open_at:
        return FirstCandle(0, None, None, 0, "NO_FIRST_CANDLE")
    if not any(b.start > open_at for b in bars):
        return FirstCandle(0, None, None, 0, "FIRST_CANDLE_NOT_FINAL")
    prior = sum(1 for b in ordered if open_at - PRIOR_WINDOW <= b.start < open_at)
    if prior < PRIOR_MIN_BARS:
        return FirstCandle(0, None, None, prior, "TOO_FEW_PRIOR_CANDLES")
    closes = [b.close for b in ordered]
    close, average = closes[-1], ema(closes)
    side = 1 if close > average else -1 if close < average else 0
    return FirstCandle(side, close, average, prior, "" if side else "CLOSE_EQUALS_EMA")


def cfd_amount(entry: float, stop_bp: float, decimals: int, factor: float) -> float:
    """Contracts whose stop costs RISK_USD (`factor` dollars a point per contract; 1 for Saxo's
    index CFDs), rounded down to Saxo's amount decimals."""
    step = 10.0**decimals
    return math.floor(RISK_USD / (entry * stop_bp / 1e4 * factor) * step) / step


def stop_price(entry: float, side: int, stop_bp: float) -> float:
    return entry * (1 - side * stop_bp / 1e4)


def stop_fill(side: int, stop: float, bid: float, ask: float) -> float | None:
    """The price a stop closes at once crossed: the bid for a long, the ask for a short."""
    if side > 0 and bid <= stop:
        return bid
    if side < 0 and ask >= stop:
        return ask
    return None


def cfd_pnl(side: int, entry: float, exit_: float, amount: float, factor: float) -> float:
    return side * (exit_ - entry) * amount * factor


@dataclass(frozen=True)
class OptionQuote:
    uic: int
    strike: float
    put_call: str  # "Call" or "Put"
    bid: float | None
    ask: float | None
    delta: float | None
    mid_vol: float | None
    cost_buy: float | None
    cost_sell: float | None
    delayed: int | None
    market_state: str | None


def pick_option(
    board: Iterable[OptionQuote], put_call: str, target: float, band: tuple[float, float]
) -> OptionQuote | None:
    """Two-sided quote with |delta| inside the band, nearest the target; ties: tighter spread."""
    usable = [
        o
        for o in board
        if o.put_call == put_call
        and o.bid
        and o.ask
        and 0 < o.bid <= o.ask
        and o.delta is not None
        and band[0] <= abs(o.delta) <= band[1]
    ]
    return min(
        usable,
        # Rounded so equally distant deltas (0.42 and 0.38 for 0.40) tie on float noise too.
        key=lambda o: (
            round(abs(abs(o.delta or 0) - target), 6),
            (o.ask or 0) - (o.bid or 0),
            o.strike,
        ),
        default=None,
    )


def option_pnl(
    entry_ask: float, exit_bid: float, cost_buy: float | None, cost_sell: float | None
) -> float | None:
    """One contract, net of Saxo's commission; None when a commission is unknown."""
    if cost_buy is None or cost_sell is None:
        return None
    return (exit_bid - entry_ask) * OPTION_MULTIPLIER - cost_buy - cost_sell


def rsi2(closes: Sequence[float]) -> list[float]:
    """Wilder RSI(2) as the research computed it: ewm(alpha=0.5, adjust=False) of gains/losses."""
    out = [math.nan]
    up = down = None
    for previous, current in zip(closes, closes[1:], strict=False):
        gain, loss = max(current - previous, 0.0), max(previous - current, 0.0)
        up = gain if up is None else up + 0.5 * (gain - up)
        down = loss if down is None else down + 0.5 * (loss - down)
        out.append(100.0 if down == 0 else 100 - 100 / (1 + up / down))
    return out


@dataclass(frozen=True)
class Connors:
    signal: bool
    close: float | None
    sma200: float | None
    rsi: float | None
    rsi_prev: float | None
    reason: str


def connors(closes: Sequence[float]) -> Connors:
    """Daily closes oldest first; the last one is today's 15:50 price."""
    if len(closes) < CONNORS_DAYS:
        return Connors(False, None, None, None, None, "TOO_FEW_DAILY_CLOSES")
    average = sum(closes[-CONNORS_DAYS:]) / CONNORS_DAYS
    r = rsi2(closes)
    signal = closes[-1] > average and r[-1] + r[-2] < 35
    return Connors(signal, closes[-1], average, r[-1], r[-2], "")


CROSS_ALPHA = 2 / 11  # EMA(10)
CANDLE = timedelta(minutes=5)
CROSS_FIRST = time(10, 0)  # entry candles closing 10:00-15:00 New York
CROSS_LAST = time(15, 0)
CROSS_EXIT_LAST = time(15, 55)  # a cross back at the 16:00 close is the 16:00 exit itself


@dataclass(frozen=True)
class Candle:
    start: datetime  # UTC
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Cross:
    entry: datetime  # start of the candle whose close crossed up; the trade enters at its close
    entry_close: float
    entry_vwap: float
    entry_ema: float
    exit: datetime | None  # start of the candle whose close fell back below VWAP; None: not yet
    exit_close: float | None


def vwap_crosses(candles: Sequence[Candle], day: date) -> list[Cross]:
    """Method 4 replayed on QQQ's 5-minute candles as Saxo returns them (the newest still forming):
    EMA(10) of every final regular-session close, sessions joined, seeded with the oldest read;
    the day's VWAP of typical price x volume from 09:30. Long when EMA - VWAP turns from <= 0 to
    > 0 at a close; out at the first later close below 0; the next entry only after that exit."""
    if not candles:
        return []
    newest = max(c.start for c in candles)
    by_start = {c.start: c for c in candles if c.start < newest}
    regular = [
        by_start[k]
        for k in sorted(by_start)
        if REGULAR_OPEN <= k.astimezone(NY).time() < REGULAR_CLOSE
    ]
    today = [c for c in regular if c.start.astimezone(NY).date() == day]
    if not today or today[0].start.astimezone(NY).time() != REGULAR_OPEN:
        return []
    value, ema_at = regular[0].close, {}
    for c in regular:
        value += CROSS_ALPHA * (c.close - value)
        ema_at[c.start] = value
    vwaps: list[float | None] = []
    pv = vol = 0.0
    for c in today:
        pv += (c.high + c.low + c.close) / 3 * c.volume
        vol += c.volume
        vwaps.append(pv / vol if vol > 0 else None)
    gap = [None if w is None else ema_at[c.start] - w for c, w in zip(today, vwaps, strict=True)]

    def closes(k: int) -> time:
        return (today[k].start + CANDLE).astimezone(NY).time()

    out: list[Cross] = []
    i = 1
    while i < len(today) and closes(i) <= CROSS_LAST:
        g0, g1 = gap[i - 1], gap[i]
        if closes(i) < CROSS_FIRST or g0 is None or g1 is None or not g0 <= 0 < g1:
            i += 1
            continue
        back = next((k for k in range(i + 1, len(today)) if (gap[k] or 0) < 0), None)
        if back is not None and closes(back) > CROSS_EXIT_LAST:
            back = None
        c, w = today[i], vwaps[i]
        assert w is not None
        out.append(
            Cross(
                c.start,
                c.close,
                w,
                ema_at[c.start],
                None if back is None else today[back].start,
                None if back is None else today[back].close,
            )
        )
        if back is None:
            break  # held to 16:00 (or still open): no later entry today
        i = back + 1
    return out


@dataclass(frozen=True)
class Session:
    open: datetime  # UTC
    close: datetime  # UTC

    @property
    def day(self) -> date:
        return self.open.astimezone(NY).date()

    @property
    def regular(self) -> bool:
        """A full 09:30-16:00 New York day; early closes are skipped by the protocol."""
        return (
            self.open.astimezone(NY).time() == REGULAR_OPEN
            and self.close.astimezone(NY).time() == REGULAR_CLOSE
        )


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def trading_sessions(exchange: dict[str, Any]) -> list[Session]:
    """Continuous-trading sessions from GET /ref/v1/exchanges/{id}, earliest first."""
    return sorted(
        (
            Session(utc(s["StartTime"]), utc(s["EndTime"]))
            for s in exchange.get("ExchangeSessions") or []
            if s.get("State") == "AutomatedTrading"
        ),
        key=lambda s: s.open,
    )


def session_on(sessions: Iterable[Session], day: date) -> Session | None:
    return next((s for s in sessions if s.day == day), None)


def ny_time(day: date, at: time) -> datetime:
    return datetime.combine(day, at, NY)
