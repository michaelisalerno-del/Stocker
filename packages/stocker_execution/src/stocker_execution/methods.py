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
5. Dip rules (amendment 3): at 15:50, on QQQ's daily closes with the 15:50 price as today's, a
   paper long Nasdaq CFD held over days: Connors (method 3's signal; out at the first later 15:50
   with RSI2 above 65) and Turnaround Tuesday (a Monday at least 1% below the last close; out at
   the next session's 15:50). Amendment 4 adds the fear rule: three down closes with QQQ's 30-day
   at-the-money implied volatility at 30% or more; out at the fifth session's 15:50.
6. Context and books (amendment 4): each first-candle signal is saved with its reference ETF's
   closes through the prior session (50- and 100-day averages, the prior day's change, the run of
   down closes, RSI2); the Record page sums the frozen subsets in BOOKS. The Today page's playbook
   shows the plain trade, skipped when the Nasdaq and S&P signals disagree (22 October 2026,
   PREREG139). Nothing traded changes.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
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


CONNORS_EXIT_RSI = 65.0
TURNAROUND_DROP = -0.01
FEAR_DOWN_CLOSES = 3  # the fear rule (amendment 4): three down closes, today's 15:50 price last
FEAR_VOL = 0.30  # ... with QQQ's 30-day at-the-money implied volatility at 30% or more
FEAR_DAYS = 30  # the expiry nearest this many calendar days out gives the reading
FEAR_HOLD = 5  # sessions held: out at the fifth session's 15:50 after the entry day


def down_run(closes: Sequence[float]) -> int:
    """Consecutive down closes ending at the last one."""
    run = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] >= closes[i - 1]:
            break
        run += 1
    return run


@dataclass(frozen=True)
class Dip:
    connors: Connors
    change: float | None  # today's price against the last daily close
    connors_exit: bool  # RSI2(today) above 65: an open Connors trade closes
    turnaround: bool  # a Monday at least 1% down: Turnaround Tuesday buys
    reason: str
    fear: bool = False  # three down closes with the 30-day implied volatility at 30% or more
    down_closes: int = 0


def dip_rules(closes: Sequence[float], day: date, atm_vol: float | None = None) -> Dip:
    """Method 5 on daily closes oldest first, the last one today's 15:50 price (as `connors`);
    `atm_vol` is QQQ's 30-day at-the-money implied volatility, None when it could not be read."""
    c = connors(closes)
    if c.reason:
        return Dip(c, None, False, False, c.reason)
    assert c.rsi is not None
    change = closes[-1] / closes[-2] - 1
    run = down_run(closes)
    return Dip(
        c,
        change,
        c.rsi > CONNORS_EXIT_RSI,
        day.weekday() == 0 and change <= TURNAROUND_DROP,
        "",
        run >= FEAR_DOWN_CLOSES and atm_vol is not None and atm_vol >= FEAR_VOL,
        run,
    )


def atm_vol(board: Iterable[OptionQuote], centre: float) -> float | None:
    """Saxo's MidVol averaged over the call and put at the strike nearest `centre`."""
    quotes = [o for o in board if o.mid_vol]
    if not quotes:
        return None
    strike = min((o.strike for o in quotes), key=lambda k: abs(k - centre))
    vols = [o.mid_vol for o in quotes if o.strike == strike and o.mid_vol]
    return sum(vols) / len(vols) if vols else None


# context and books (amendment 4) ------------------------------------------------------------
SPY_UIC = 36590  # SPY:arcx; the research judged the S&P trade against SPY's own closes
TREND_UIC = {"NAS100": QQQ_UIC, "US500": SPY_UIC}  # each CFD's reference ETF
TREND_DAYS, TREND_DAYS_LONG = 50, 100
OVERSOLD_RSI2 = 10.0
BIG_DAY = 0.01
BOOKS = (
    "PULLBACK",  # long signal with the prior close below the 50-day average (any weekday)
    "WEEKDAYS",  # Tuesday to Friday
    "AGREE",  # the Nasdaq and S&P signals point the same way
    "NO OVERSOLD SHORTS",  # longs, and shorts unless 3+ down closes or RSI2 below 10
    "AFTER A +1% DAY",  # the prior session closed at least 1% up
    "PLAYBOOK",  # what the leads together say: see `playbook`
)
CROSS_BOOKS = ("CROSS PULLBACK", "CROSS BEFORE 14:00")
CROSS_LATE_HOUR = 14


@dataclass(frozen=True)
class Context:
    """The reference ETF's closes through the prior session, as the research used them."""

    prior_close: float
    sma50: float
    sma100: float
    prior_change: float  # the prior close against the close before it
    down_closes: int  # consecutive down closes ending at the prior close
    rsi2: float  # Wilder RSI(2) at the prior close

    def detail(self) -> dict[str, Any]:
        return {
            "prior_close": self.prior_close,
            "sma50": round(self.sma50, 4),
            "sma100": round(self.sma100, 4),
            "below_50d": self.prior_close < self.sma50,
            "below_100d": self.prior_close < self.sma100,
            "prior_change": round(self.prior_change, 6),
            "down_closes": self.down_closes,
            "rsi2_prior": round(self.rsi2, 4),
        }


def trend_context(closes: Sequence[float]) -> Context | None:
    """Daily closes oldest first, the last one the prior session's; None with too few."""
    if len(closes) < TREND_DAYS_LONG + 1:
        return None
    return Context(
        closes[-1],
        sum(closes[-TREND_DAYS:]) / TREND_DAYS,
        sum(closes[-TREND_DAYS_LONG:]) / TREND_DAYS_LONG,
        closes[-1] / closes[-2] - 1,
        down_run(closes),
        rsi2(closes)[-1],
    )


@dataclass(frozen=True)
class Day:
    """A first-candle day as the books see it: the signal and its saved context."""

    side: int
    weekday: int  # Monday 0
    agree: bool | None  # the other index's signal points the same way; None when unknown
    context: dict[str, Any]  # the signal detail; context keys absent before amendment 4

    @property
    def known(self) -> bool:
        return "below_50d" in self.context

    @property
    def pullback(self) -> bool:
        return self.side == 1 and bool(self.context.get("below_50d"))

    @property
    def oversold(self) -> bool:
        return bool(
            self.context.get("down_closes", 0) >= FEAR_DOWN_CLOSES
            or self.context.get("rsi2_prior", 100.0) < OVERSOLD_RSI2
        )


def in_book(book: str, d: Day, instrument: str) -> bool:
    """Membership of one first-candle trade (instrument 'NAS100', 'US500' or 'QQQ option')."""
    if book == "WEEKDAYS":
        return d.weekday != 0
    if book == "AGREE":
        return d.agree is True
    if not d.known:
        return False
    if book == "PULLBACK":
        return d.pullback
    if book == "NO OVERSOLD SHORTS":
        return d.side == 1 or not d.oversold
    if book == "AFTER A +1% DAY":
        return float(d.context.get("prior_change", 0.0)) >= BIG_DAY
    if book == "PLAYBOOK":
        if instrument == "QQQ option" or instrument == "US500":
            return d.pullback  # the pullback cell: the call, and the S&P only there
        return (d.side == 1 and not d.pullback) or (
            d.side == -1 and d.weekday != 0 and not d.oversold
        )
    raise ValueError(book)


SIZED_BOOKS = ("SIZED V2", "SIZED V3")
WHAT_IF_STOP_BP = (
    75.0  # the Nasdaq CFD trade re-run with this stop at the same $200 risk (report only)
)


def size_v2(d: Day, instrument: str) -> int:
    """The playbook as units of size: the pullback cell 2; other longs 1; shorts 1 unless it is
    a Monday or the close was oversold; the QQQ option and the S&P only in the cell."""
    if not d.known or d.side == 0:
        return 0
    if instrument in ("QQQ option", "US500"):
        return 2 if d.pullback else 0
    if d.pullback:
        return 2
    if d.side == 1:
        return 1
    return 0 if d.weekday == 0 or d.oversold else 1


def size_v3(d: Day) -> int:
    """Score the leads: +1 pullback cell, +1 after a +1% day, +1 the other index agrees,
    -1 Monday, -1 a short after an oversold close; size = 1 + score, kept within 0..3."""
    if not d.known or d.side == 0:
        return 0
    score = int(d.pullback) + int(float(d.context.get("prior_change", 0.0)) >= BIG_DAY)
    score += int(d.agree is True) - int(d.weekday == 0) - int(d.side == -1 and d.oversold)
    return max(0, min(3, 1 + score))


def what_if_stop(
    side: int, entry: float, worst: float | None, pnl: float | None, stop_bp: float, live_bp: float
) -> float | None:
    """A trade's result had its stop been `stop_bp` wide at the same dollar risk: stopped at
    RISK_USD lost when the worst adverse price crossed that stop, else the recorded result
    scaled by the size the narrower stop allows. None when the worst price was not recorded."""
    if worst is None or pnl is None:
        return None
    adverse_bp = side * (entry - worst) / entry * 1e4
    if adverse_bp >= stop_bp:
        return -RISK_USD
    return pnl * live_bp / stop_bp


def playbook(d: Day, market: str) -> tuple[str, list[str]]:
    """Today's action for a first-candle signal: the plain CFD trade, skipped when the Nasdaq and
    S&P signals disagree (PREREG28; the one lead that held on unseen dates in PREREG139). The
    other leads stay as books."""
    tags = []
    if d.agree is not None:
        tags.append("agrees with the other index" if d.agree else "disagrees with the other index")
    if d.side == 0:
        return "no trade", tags
    if d.agree is False:
        return "skip", tags
    name = "Nasdaq CFD" if market == "NAS100" else "S&P CFD"
    return f"{name} {'long' if d.side == 1 else 'short'}", tags


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


# stock dips (amendment 5) --------------------------------------------------------------------
# The Connors Research (2012) "ConnorsRSI Pullback" worked example, as PREREG131 tested it on
# 2001-2026: a liquid stock that sold off hard and closed near its low is bought the next session
# on a limit 8% below the close and sold at the first close with ConnorsRSI above 70.
STOCK_MIN_PRICE, STOCK_MAX_PRICE = 5.0, 5000.0
STOCK_MIN_VOLUME = 250_000  # 21-session average shares a day
STOCK_UNIVERSE, STOCK_TOP = 1500, 500  # ranks by 21-session dollar volume; the top 500 is tier A
STOCK_DOLLAR_CAP = 10.0  # a 21-session dollar volume above ten times SPY's is bad data
STOCK_MIN_CORRELATION = 0.05  # with SPY's daily returns over the trailing 250 sessions
STOCK_HISTORY = 260  # sessions of bars a stock needs before it can set up
STOCK_SELL_OFF, STOCK_CLOSING_RANGE, STOCK_ADX, STOCK_CRSI_IN = 0.04, 0.25, 30.0, 10.0
STOCK_LIMIT, STOCK_CRSI_OUT = 0.08, 70.0
STOCK_NOTIONAL_USD = 2000.0  # a trade's size; the rule has no stop, so this is the money at risk
STOCK_COST = {"A": 0.002, "B": 0.004}  # round trip, as the research charged each tier
STOCK_NO_BAR_SESSIONS = 10  # an open trade with no bar for this long closes at its last close


@dataclass(frozen=True)
class StockBar:
    day: str  # ISO date
    open: float
    high: float
    low: float
    close: float
    adj: float  # the close adjusted for later splits and dividends (EODHD adjusted_close)
    volume: float


def wilder(values: Sequence[float | None], n: int) -> list[float | None]:
    """pandas ewm(alpha=1/n, adjust=False) seeded with the first finite value."""
    out: list[float | None] = []
    value: float | None = None
    for x in values:
        if x is not None and math.isfinite(x):
            value = x if value is None else value + (x - value) / n
        out.append(value)
    return out


def adx10(bars: Sequence[StockBar]) -> float | None:
    """Wilder's ADX(10) on split-adjusted prices, as the research computed it."""
    if len(bars) < 2:
        return None
    f = [b.adj / b.close for b in bars]
    highs, lows, closes = (
        [b.high * k for b, k in zip(bars, f, strict=True)],
        [b.low * k for b, k in zip(bars, f, strict=True)],
        [b.adj for b in bars],
    )
    tr: list[float | None] = [highs[0] - lows[0]]
    plus: list[float | None] = [0.0]
    minus: list[float | None] = [0.0]
    for i in range(1, len(bars)):
        tr.append(
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        )
        up, down = highs[i] - highs[i - 1], lows[i - 1] - lows[i]
        plus.append(up if up > down and up > 0 else 0.0)
        minus.append(down if down > up and down > 0 else 0.0)
    s_tr, s_plus, s_minus = wilder(tr, 10), wilder(plus, 10), wilder(minus, 10)
    dx: list[float | None] = []
    for t, p, m in zip(s_tr, s_plus, s_minus, strict=True):
        if not t or p is None or m is None or p + m == 0:
            dx.append(None)
            continue
        pdi, mdi = 100 * p / t, 100 * m / t
        dx.append(100 * abs(pdi - mdi) / (pdi + mdi))
    return wilder(dx, 10)[-1]


def rsi_n(values: Sequence[float], n: int) -> float | None:
    """Wilder RSI(n) of a series (the research's ewm(alpha=1/n, adjust=False) of gains/losses)."""
    up = down = None
    for previous, current in zip(values, values[1:], strict=False):
        gain, loss = max(current - previous, 0.0), max(previous - current, 0.0)
        up = gain if up is None else up + (gain - up) / n
        down = loss if down is None else down + (loss - down) / n
    if up is None or down is None:
        return None
    if down == 0:
        return 100.0 if up > 0 else None
    return 100 - 100 / (1 + up / down)


def streaks(closes: Sequence[float]) -> list[float]:
    """+k after k up closes in a row, -k after k down closes, 0 on an unchanged close."""
    out = [0.0]
    for previous, current in zip(closes, closes[1:], strict=False):
        sign = 1.0 if current > previous else -1.0 if current < previous else 0.0
        same_way = sign != 0 and out[-1] != 0 and (out[-1] > 0) == (sign > 0)
        out.append(sign * (abs(out[-1]) + 1) if same_way else sign)
    return out


def connors_rsi(closes: Sequence[float]) -> float | None:
    """ConnorsRSI(3, 2, 100) at the last close: the mean of RSI(3) of the closes, RSI(2) of the
    up/down streak, and the percent rank of today's return among the previous 100."""
    if len(closes) < 102:
        return None
    returns = [b / a - 1 for a, b in zip(closes, closes[1:], strict=False)]
    rank = sum(1 for r in returns[-101:-1] if r < returns[-1])
    r3, r2 = rsi_n(closes, 3), rsi_n(streaks(closes), 2)
    if r3 is None or r2 is None:
        return None
    return (r3 + r2 + rank) / 3


@dataclass(frozen=True)
class StockSetup:
    code: str
    close: float  # the signal day's close, in dollars as traded
    limit: float  # the next session's buy limit, 8% below that close
    sell_off: float  # the low's fall from the prior close
    closing_range: float  # (close - low) / (high - low)
    adx: float
    crsi: float


def stock_setup(code: str, bars: Sequence[StockBar]) -> StockSetup | None:
    """Rules 3-6 of the worked example at the last bar (rules 1-2 and the universe are the
    runtime's: price, volume, rank). None when a rule fails or the history is too short."""
    if len(bars) < STOCK_HISTORY:
        return None
    last, prior = bars[-1], bars[-2]
    f_last, f_prior = last.adj / last.close, prior.adj / prior.close
    sell_off = 1 - (last.low * f_last) / (prior.close * f_prior)
    day_range = last.high - last.low
    closing_range = (last.close - last.low) / day_range if day_range > 0 else None
    adx, crsi = adx10(bars), connors_rsi([b.adj for b in bars])
    if (
        closing_range is None
        or adx is None
        or crsi is None
        or sell_off < STOCK_SELL_OFF
        or closing_range > STOCK_CLOSING_RANGE
        or adx <= STOCK_ADX
        or crsi >= STOCK_CRSI_IN
    ):
        return None
    return StockSetup(
        code,
        last.close,
        round(last.close * (1 - STOCK_LIMIT), 4),
        round(sell_off, 6),
        round(closing_range, 6),
        round(adx, 4),
        round(crsi, 4),
    )


def stock_fill(limit: float, bar: StockBar) -> float | None:
    """A resting buy limit fills when the session's low trades below it, at the open if the
    stock opened under the limit."""
    return min(bar.open, limit) if bar.low < limit else None


def stock_pnl(entry: float, exit_: float, tier: str) -> float:
    """The trade's result on STOCK_NOTIONAL_USD after the tier's round-trip cost."""
    return STOCK_NOTIONAL_USD * ((exit_ / entry - 1) - STOCK_COST[tier])


def correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None


def stock_universe(
    bars: Mapping[str, Sequence[StockBar]], spy: Sequence[StockBar]
) -> dict[str, tuple[int, str]]:
    """Rules 1-2 and the research's cohorts at the last bar: {code: (rank, tier)} for the
    1,500 largest by 21-session dollar volume among stocks priced $5-$5,000 with 250k shares a
    day, a dollar volume at most ten times SPY's, 260 sessions of bars and a daily-return
    correlation with SPY of at least 0.05 over the trailing 250 sessions (120 needed)."""
    spy_dollar = sum(b.close * b.volume for b in spy[-21:]) / 21
    spy_returns = {b.day: b.adj / a.adj - 1 for a, b in zip(spy[-251:], spy[-250:], strict=False)}
    ranked = []
    for code, series in bars.items():
        if len(series) < STOCK_HISTORY or code == "SPY":
            continue
        last = series[-1]
        volume = sum(b.volume for b in series[-21:]) / 21
        dollars = sum(b.close * b.volume for b in series[-21:]) / 21
        if not (
            STOCK_MIN_PRICE < last.close <= STOCK_MAX_PRICE
            and volume >= STOCK_MIN_VOLUME
            and dollars <= STOCK_DOLLAR_CAP * spy_dollar
        ):
            continue
        own, market = [], []
        for a, b in zip(series[-251:], series[-250:], strict=False):
            if b.day in spy_returns:
                own.append(b.adj / a.adj - 1)
                market.append(spy_returns[b.day])
        rho = correlation(own, market) if len(own) >= 120 else None
        if rho is None or rho < STOCK_MIN_CORRELATION:
            continue
        ranked.append((dollars, code))
    ranked.sort(reverse=True)
    return {
        code: (rank, "A" if rank <= STOCK_TOP else "B")
        for rank, (_, code) in enumerate(ranked[:STOCK_UNIVERSE], start=1)
    }
