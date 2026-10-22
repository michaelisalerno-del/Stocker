"""The paper record's clock: each New York trading day's events, CFD stop checks and the login.

Nothing here can place an order. Every decision reads Saxo prices and writes the paper record.
QQQ prices reach this session delayed (Saxo's real-time slot belongs to the user's SaxoTraderGO),
so a QQQ event due at T is read at T + the delay Saxo reports, and recorded as of T.
"""

import asyncio
import fcntl
import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from stocker_execution import market, methods
from stocker_execution.config import PaperConfig
from stocker_execution.methods import CFDS, NY, QQQ_OPTION_ROOT, QQQ_UIC, Session
from stocker_execution.saxo_auth import OAuth, SaxoError
from stocker_execution.saxo_client import SaxoClient
from stocker_execution.store import Store, stamp

log = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
UK = ZoneInfo("Europe/London")
ENTRY_GRACE = timedelta(seconds=60)  # an entry later than this is missed, never replayed
# An exit read more than this after its time (a restart, a lost read, a delay that changed
# under it) has no price: it is recorded missed, never filled at a much later price.
EXIT_GRACE = timedelta(minutes=30)
RETRY_EVERY = 5.0  # seconds between attempts at a failed event
STOP_POLL_SECONDS = 2.0
MARK_FRESH_SECONDS = 10.0  # the dashboard marks open CFD positions only at a quote this fresh
LOGIN_CHECK_SECONDS = 30.0
FC, FC_QQQ, CONNORS_QQQ = "FIRST_CANDLE", "FIRST_CANDLE_QQQ", "CONNORS_QQQ"
CROSS = "VWAP_CROSS"
DIP = "DIP"
# The dip rules and their entry readings, in the order the one-at-a-time book takes them.
DIP_RULES = (("CONNORS", "connors_in"), ("TURNAROUND", "turnaround"), ("FEAR", "fear"))
DIP_EXITS = {"CONNORS": "RSI2_ABOVE_65", "TURNAROUND": "NEXT_SESSION", "FEAR": "FIFTH_SESSION"}
# The stock dips (amendment 5) run once a session after the close, on Saxo's daily bars: due
# three hours after the close, retried every quarter hour until the next morning.
STOCK = "STOCK_DIPS"
STOCK_DELAY = timedelta(hours=3)
STOCK_GRACE = timedelta(hours=14)
STOCK_RETRY_SECONDS = 900.0
STOCK_STORE_DOLLARS = 5e6  # a stock's bars are read once it trades this much in a day
STOCK_CHART_COUNT = 300  # daily bars read for a stock new to the record or whose history changed
STOCK_REFRESH_COUNT = 5  # daily bars read on other nights
STOCK_REWRITE_TOLERANCE = 0.003  # a stored close that no longer matches Saxo's: history re-read
STOCK_KEEP_DAYS = 600  # calendar days of bars kept
STOCK_POLL_SECONDS = 60.0  # display only: the Today page's stock prices in the session
CROSS_POLL_SECONDS = 5.0  # while a QQQ candle is due
CROSS_DELAYED_POLL_SECONDS = 60.0  # while QQQ's chart is delayed: entries are missed anyway
# The dashboard's chart: each market's latest 5-minute candles, refreshed every minute from three
# hours before the open to ten minutes after the close, every fifteen minutes otherwise.
CHARTS = tuple((c.market, "CfdOnIndex", c.uic) for c in CFDS if c.market == "US500") + (
    ("QQQ", "Etf", QQQ_UIC),
)
CHART_REFRESH_SECONDS = 60.0
CHART_IDLE_SECONDS = 900.0


class Problem(Exception):
    """A coded reason this attempt could not complete; the event retries until its deadline."""


class Done(Exception):
    """The event completed without a trade, for the coded reason given."""


def failure_code(exc: BaseException) -> str:
    """Coded errors are the application's own reasons; other text may hold secrets."""
    if isinstance(exc, ValueError | Problem | Done) and re.fullmatch(
        r"[A-Za-z0-9_]{1,120}", str(exc)
    ):
        return str(exc)
    return type(exc).__name__


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


class Runtime:
    def __init__(
        self,
        config: PaperConfig,
        store: Store,
        client: SaxoClient | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.config, self.store = config, store
        store.bind(config.data_environment)
        self.root = Path(store.db.execute("PRAGMA database_list").fetchone()[2]).parent
        self.client = client or SaxoClient(OAuth(config.data_environment, config.saxo, self.root))
        self.now = clock or (lambda: datetime.now(UTC))
        self.worker_health = self.web_health = "STARTING"
        self.stopping = False
        self.owner: Any = None
        self.started_at = self.now()
        self.sessions: list[Session] = []
        self.sessions_day: date | None = None
        self.sessions_retry_at = 0.0
        self.cfd_info: dict[int, tuple[int, float]] = {}
        self.cfd_quotes: dict[int, market.Quote] = {}
        self.cfd_polled = 0.0
        self.login_checked = 0.0
        self.capabilities: dict[str, Any] = {}
        self.capabilities_read = 0.0
        self.attempted: dict[tuple[str, str], float] = {}
        self.problems: dict[str, str] = {}
        self.qqq_delay: int | None = None
        self.cross_done: datetime | None = None  # the latest QQQ candle close acted on
        self.cross_next = 0.0
        self.chart_bars: dict[str, dict[str, Any]] = {}  # display only: the dashboard's candles
        self.charts_next = 0.0
        self.stock_uics: dict[str, int] = {}  # Saxo's US stock list, as the night's scan read it
        self.stock_marks: dict[str, dict[str, Any]] = {}  # display only: the latest price read
        self.stocks_next = 0.0
        directory = self.root / config.data_environment
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        handler = RotatingFileHandler(directory / "paper.log", maxBytes=2 * 1024**2, backupCount=3)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        self.log_handler = handler
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)

    # the loop ---------------------------------------------------------------------------------
    async def run(self) -> None:
        # One OAuth owner: Saxo rotates the refresh token, so a second refresher would break both.
        lock = self.root / self.config.data_environment / "data.owner.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        self.owner = lock.open("a")
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.worker_health = "RUNNING"
        try:
            while not self.stopping:
                try:
                    await self.step()
                except Exception as exc:  # one bad pass is logged; the record keeps running
                    self.note("loop", exc)
                await asyncio.sleep(1)
        finally:
            await self.stop()

    async def step(self) -> None:
        await self.keep_login()
        now = self.now()
        day = now.astimezone(NY).date()
        if self.sessions_day != day and time.monotonic() >= self.sessions_retry_at:
            try:
                self.sessions = await market.sessions(self.client)
                self.sessions_day = day
                self.problems.pop("calendar", None)
            except SaxoError as exc:
                self.sessions_retry_at = time.monotonic() + 60
                self.note("calendar", exc)
        session = methods.session_on(self.sessions, day) if self.sessions_day == day else None
        if session is not None:
            for name, due, delayed, handler in self.plan(session):
                await self.event(day.isoformat(), name, due, delayed, handler, now)
        for row in self.store.pending_events(STOCK):  # a night's scan that ran past midnight
            if row["day"] < day.isoformat():
                await self.event(
                    row["day"], STOCK, at(row["due"]), False, self.stock_night(row["day"]), now
                )
        await self.watch_stops(self.now(), session)
        await self.watch_crosses(self.now(), session)
        await self.watch_charts(self.now(), session)
        await self.watch_stocks(self.now(), session)

    async def watch_charts(self, now: datetime, session: Session | None) -> None:
        """Display only: keep each market's latest candles for the dashboard's chart. A failed
        read keeps the last candles and notes the reason; nothing recorded depends on it."""
        if time.monotonic() < self.charts_next:
            return
        busy = session is not None and (
            session.open - timedelta(hours=3) <= now <= session.close + timedelta(minutes=10)
        )
        self.charts_next = time.monotonic() + (
            CHART_REFRESH_SECONDS if busy else CHART_IDLE_SECONDS
        )
        for name, asset_type, uic in CHARTS:
            try:
                rows, delay = await market.chart_rows(self.client, asset_type, uic)
            except SaxoError as exc:
                self.note("charts", exc)
                return
            self.chart_bars[name] = {"read": stamp(self.now()), "delay": delay, "bars": rows}
        self.problems.pop("charts", None)

    def charts(self) -> dict[str, Any]:
        return {"markets": self.chart_bars}

    async def watch_stocks(self, now: datetime, s: Session | None) -> None:
        """Display only: in the session, a price read a minute for the stocks with an order
        resting today or an open position, and ConnorsRSI as if a held stock closed at that
        price. Fills and exits stay the night scan's, on the daily bars."""
        if s is None or not s.open <= now < s.close:
            return
        if time.monotonic() < self.stocks_next:
            return
        self.stocks_next = time.monotonic() + STOCK_POLL_SECONDS
        day = s.day.isoformat()
        held = {t["code"] for t in self.store.open_stock_trades()}
        codes = held | {
            o["code"] for o in self.store.pending_stock_orders() if o["signal_day"] < day
        }
        if not codes:
            return
        try:
            if not self.stock_uics:  # after a restart, until the night's scan reads it again
                self.stock_uics = await market.us_stocks(self.client)
            uics = {self.stock_uics[c]: c for c in codes if c in self.stock_uics}
            prices = await market.stock_prices(self.client, sorted(uics))
        except SaxoError as exc:
            self.note("stock prices", exc)
            return
        read, marks = stamp(self.now()), {}
        for uic, p in prices.items():
            code = uics[uic]
            if not p.last:
                continue
            crsi = None
            if code in held:
                closes = [b.adj for b in self.store.stock_bars(code) if b.day < day]
                crsi = methods.connors_rsi([*closes, p.last])
            marks[code] = {
                "now_price": p.last,
                "now_open": p.open,
                "now_low": p.low,
                "now_delay": p.delayed,
                "now_at": read,
                "crsi_now": None if crsi is None else round(crsi, 1),
            }
        self.stock_marks = marks
        self.problems.pop("stock prices", None)

    def plan(
        self, s: Session
    ) -> list[tuple[str, datetime, bool, Callable[[datetime, int], Awaitable[None]]]]:
        """(event, due, read QQQ delayed?, handler) in the order they run."""
        nine35 = s.open + timedelta(minutes=5)
        return sorted(
            [
                (FC, nine35, False, lambda t, d: self.first_candle(s, t)),
                (FC_QQQ + "_ENTRY", nine35, True, lambda t, d: self.option_entry(s, t, d)),
                (
                    FC_QQQ + "_EXIT",
                    methods.ny_time(s.day, methods.OPTION_EXIT),
                    True,
                    lambda t, d: self.option_exit(s, t, d),
                ),
                (
                    CONNORS_QQQ,
                    s.close - timedelta(minutes=10),
                    True,
                    lambda t, d: self.connors(s, t, d),
                ),
                (
                    DIP + "_EXIT",
                    s.close - timedelta(minutes=10),
                    False,
                    lambda t, d: self.dip_exit(s, t),
                ),
                (
                    DIP + "_ENTRY",
                    s.close - timedelta(minutes=10),
                    False,
                    lambda t, d: self.dip_entry(s, t),
                ),
                (
                    FC_QQQ + "_LATE",
                    s.close - timedelta(minutes=5),
                    True,
                    lambda t, d: self.option_late(s, t, d),
                ),
                (
                    CONNORS_QQQ + "_EXIT",
                    s.close - timedelta(minutes=5),
                    True,
                    lambda t, d: self.connors_exit(s, t, d),
                ),
                ("CFD_EXIT", s.close, False, lambda t, d: self.cfd_exit(s, t)),
                (CROSS + "_EXIT", s.close, False, lambda t, d: self.cross_exit(s, t)),
                (STOCK, s.close + STOCK_DELAY, False, lambda t, d: self.stock_dips(s.day, t)),
            ],
            key=lambda e: e[1],
        )

    async def event(
        self,
        day: str,
        name: str,
        due: datetime,
        delayed: bool,
        handler: Callable[[datetime, int], Awaitable[None]],
        now: datetime,
    ) -> None:
        row = self.store.event(day, name)
        if now < due or (row is not None and row["status"] in {"DONE", "FAILED", "MISSED"}):
            return
        entry = name in {FC, FC_QQQ + "_ENTRY", CONNORS_QQQ, DIP + "_ENTRY"}
        if row is None and entry and now > due + ENTRY_GRACE:
            # An entry first tried late is never replayed at a later price.
            note = "APP_NOT_RUNNING_AT_ENTRY" if self.started_at > due else "NOT_TRIED_IN_TIME"
            self.store.set_event(day, name, due, "MISSED", note=note)
            return
        retry_every = STOCK_RETRY_SECONDS if name == STOCK else RETRY_EVERY
        if time.monotonic() - self.attempted.get((day, name), -1e9) < retry_every:
            return
        self.attempted[(day, name)] = time.monotonic()
        grace = STOCK_GRACE if name == STOCK else ENTRY_GRACE if entry else EXIT_GRACE
        delay = 0
        if delayed:
            if row is None or row["effective"] is None:
                try:
                    delay = await self.measure_qqq_delay()
                except (SaxoError, Problem) as exc:
                    self.give_up_or_retry(day, name, due, due, grace, entry, now, exc)
                    return
                self.store.set_event(
                    day, name, due, "WAITING", effective=due + timedelta(minutes=delay)
                )
                row = self.store.event(day, name)
            assert row is not None
            delay = round((at(row["effective"]) - due).total_seconds() / 60)
        effective = due + timedelta(minutes=delay)
        if now < effective:
            return
        if entry and now > effective + ENTRY_GRACE:
            # An entry only ever runs within a minute of its time: never at a later price.
            failed = row is not None and row["status"] == "RETRYING"
            late = (
                "APP_NOT_RUNNING_AT_ENTRY" if self.started_at > effective else "NOT_TRIED_IN_TIME"
            )
            self.store.set_event(
                day,
                name,
                due,
                "FAILED" if failed else "MISSED",
                effective=effective,
                note=row["note"] if failed and row else late,
            )
            self.problems.pop(name, None)
            return
        try:
            await handler(due, delay)
        except Done as reason:
            self.store.set_event(day, name, due, "DONE", effective=effective, note=str(reason))
            self.problems.pop(name, None)
            return
        except (SaxoError, Problem, KeyError, TypeError, ValueError) as exc:
            self.give_up_or_retry(day, name, due, effective, grace, entry, now, exc)
            return
        self.store.set_event(day, name, due, "DONE", effective=effective)
        self.problems.pop(name, None)

    def give_up_or_retry(
        self,
        day: str,
        name: str,
        due: datetime,
        effective: datetime,
        grace: timedelta,
        entry: bool,
        now: datetime,
        exc: BaseException,
    ) -> None:
        code = failure_code(exc)
        self.note(name, exc)
        final = now > effective + grace
        if final:
            self.problems.pop(name, None)  # the event's status now carries the reason
        # The delay follows who holds Saxo's real-time slot (a SaxoTraderGO login takes it). An
        # entry cannot recover its as-of price; an exit measures again and reads at the new time.
        remeasure = code == "QQQ_DELAY_CHANGED" and not entry
        self.store.set_event(
            day,
            name,
            due,
            "FAILED" if final else "RETRYING",
            effective=None if remeasure or effective == due else effective,
            note=code,
        )

    def note(self, area: str, exc: BaseException) -> None:
        code = failure_code(exc)
        if self.problems.get(area) != code:
            log.warning("%s: %s", area, code)
        self.problems[area] = code

    async def keep_login(self) -> None:
        """Refresh the access token before it lapses, even on days with nothing to record."""
        if time.monotonic() - self.login_checked < LOGIN_CHECK_SECONDS:
            return
        self.login_checked = time.monotonic()
        try:
            await self.client.oauth.access_token()
            self.problems.pop("login", None)
            if time.monotonic() - self.capabilities_read >= 600:
                self.capabilities_read = time.monotonic()
                self.capabilities = await self.client.get("/root/v1/sessions/capabilities")
        except SaxoError as exc:
            self.note("login", exc)

    async def measure_qqq_delay(self) -> int:
        q = (await market.quotes(self.client, [QQQ_UIC], "Etf")).get(QQQ_UIC)
        if q is None or q.delayed is None or not 0 <= q.delayed <= 30:
            raise Problem("QQQ_DELAY_UNKNOWN")
        self.qqq_delay = q.delayed
        return q.delayed

    # first candle on the CFDs -----------------------------------------------------------------
    async def first_candle(self, s: Session, due: datetime) -> None:
        day = s.day.isoformat()
        if not s.regular:
            raise Done("EARLY_CLOSE_DAY_SKIPPED")
        pending = ""
        for cfd in CFDS:
            signal = self.store.signal(day, FC, cfd.market)
            if self.store.cfd_trade(day, cfd.market) or (signal and signal["side"] == 0):
                continue
            try:
                await self.first_candle_market(s, cfd, due)
            except (SaxoError, Problem) as exc:
                pending = failure_code(exc)
        if pending:  # retried until the entry deadline, then the event is FAILED
            raise Problem(pending)

    async def first_candle_market(self, s: Session, cfd: methods.Cfd, due: datetime) -> None:
        day = s.day.isoformat()
        saved = self.store.signal(day, FC, cfd.market)
        if saved is None:
            bars, delay = await market.cfd_bars(self.client, cfd.uic)
            if delay != 0:
                raise Problem("CFD_CHART_DELAYED")
            signal = methods.first_candle(bars, s.open)
            if signal.reason in {"NO_FIRST_CANDLE", "FIRST_CANDLE_NOT_FINAL"}:
                raise Problem(signal.reason)
            context = await self.day_context(cfd.market, s.day)
            # Saved at once: the QQQ option follows the signal even if the CFD trade cannot open.
            self.store.add_signal(
                day,
                FC,
                cfd.market,
                signal.side,
                signal.reason,
                {
                    "close": signal.close,
                    "ema": signal.ema,
                    "prior_candles": signal.prior_bars,
                    "bars_read": len(bars),
                    "last_bars": [[stamp(b.start), b.close] for b in bars[-40:]],
                    **context,
                },
            )
            side = signal.side
        else:
            side = saved["side"]
        if side == 0:
            return
        if cfd.uic not in self.cfd_info:
            self.cfd_info[cfd.uic] = await market.cfd_details(self.client, cfd.uic)
        decimals, factor = self.cfd_info[cfd.uic]
        q = (await market.quotes(self.client, [cfd.uic], "CfdOnIndex")).get(cfd.uic)
        read = self.now()
        if read > due + ENTRY_GRACE:
            raise Problem("ENTRY_DEADLINE_PASSED")
        if q is None or not q.live:
            raise Problem("CFD_QUOTE_NOT_LIVE")
        assert q.bid is not None and q.ask is not None
        entry = q.ask if side > 0 else q.bid
        amount = methods.cfd_amount(entry, cfd.stop_bp, decimals, factor)
        if amount <= 0:
            raise Problem("CFD_SIZE_ROUNDS_TO_ZERO")
        self.store.insert(
            "cfd_trades",
            {
                "day": day,
                "market": cfd.market,
                "uic": cfd.uic,
                "side": side,
                "amount": amount,
                "factor": factor,
                "stop": round(methods.stop_price(entry, side, cfd.stop_bp), 2),
                "entry_at": stamp(read),
                "entry_price": entry,
                "entry_bid": q.bid,
                "entry_ask": q.ask,
            },
        )
        log.info("%s %s %s %.2f at %.2f", day, cfd.market, side, amount, entry)

    async def day_context(self, cfd_market: str, day: date) -> dict[str, Any]:
        """Amendment 4: the reference ETF's closes through the prior session, saved with the
        signal for the books and the playbook. Reported only: a read that fails leaves its
        reason and never holds up the trade."""
        try:
            dated = await market.daily_closes(self.client, methods.TREND_UIC[cfd_market], day)
        except SaxoError as exc:
            return {"context_reason": failure_code(exc)}
        if not dated or (day - dated[-1][0]).days > 4:
            return {"context_reason": "DAILY_CLOSES_OUT_OF_DATE"}
        context = methods.trend_context([close for _, close in dated])
        return context.detail() if context else {"context_reason": "TOO_FEW_DAILY_CLOSES"}

    async def watch_stops(self, now: datetime, session: Session | None) -> None:
        """Stops are watched during their own session only. A trade still open on a later day
        missed its 16:00 exit: it is recorded without a price, never closed at a later one."""
        trades = self.store.open_cfd()
        today = session.day.isoformat() if session else None
        for t in trades:
            if today is None or t["day"] < today:
                self.store.update(
                    "cfd_trades", t["id"], {"exit_at": stamp(now), "exit_reason": "MISSED_EXIT"}
                )
        watched = [t for t in trades if t["day"] == today and session and now < session.close]
        if not watched:
            self.problems.pop("stops", None)
            return
        if time.monotonic() - self.cfd_polled < STOP_POLL_SECONDS:
            return
        self.cfd_polled = time.monotonic()
        try:
            self.cfd_quotes = await market.quotes(
                self.client, sorted({t["uic"] for t in watched}), "CfdOnIndex"
            )
        except SaxoError as exc:
            self.note("stops", exc)
            return
        stale = False
        for t in watched:
            q = self.cfd_quotes.get(t["uic"])
            if q is None or not q.live:
                stale = True
                continue
            assert q.bid is not None and q.ask is not None
            adverse = q.bid if t["side"] > 0 else q.ask  # the price the trade would close at
            if t["worst"] is None or t["side"] * (adverse - t["worst"]) < 0:
                self.store.update("cfd_trades", t["id"], {"worst": adverse})
            fill = methods.stop_fill(t["side"], t["stop"], q.bid, q.ask)
            if fill is not None:
                self.close_cfd(t, fill, q, "STOP", now)
        if stale:
            self.problems["stops"] = "CFD_QUOTE_NOT_LIVE"
        else:
            self.problems.pop("stops", None)

    def close_cfd(
        self, t: sqlite3.Row, price: float, q: market.Quote, reason: str, now: datetime
    ) -> None:
        pnl = methods.cfd_pnl(t["side"], t["entry_price"], price, t["amount"], t["factor"])
        self.store.update(
            "cfd_trades",
            t["id"],
            {
                "exit_at": stamp(now),
                "exit_price": price,
                "exit_bid": q.bid,
                "exit_ask": q.ask,
                "exit_reason": reason,
                "pnl_usd": round(pnl, 2),
            },
        )
        log.info("%s %s closed %s %.2f", t["day"], t["market"], reason, pnl)

    async def cfd_exit(self, s: Session, due: datetime) -> None:
        trades = [t for t in self.store.open_cfd() if t["day"] == s.day.isoformat()]
        if not trades:
            raise Done("NO_OPEN_CFD_TRADE")
        quotes = await market.quotes(self.client, sorted({t["uic"] for t in trades}), "CfdOnIndex")
        read = self.now()
        for t in trades:
            if read > due + EXIT_GRACE:  # too far from 16:00 to be its price
                self.store.update(
                    "cfd_trades", t["id"], {"exit_at": stamp(read), "exit_reason": "MISSED_EXIT"}
                )
                continue
            q = quotes.get(t["uic"])
            if q is None or not q.live:
                raise Problem("CFD_QUOTE_NOT_LIVE")
            assert q.bid is not None and q.ask is not None
            late = read > due + ENTRY_GRACE
            self.close_cfd(
                t, q.bid if t["side"] > 0 else q.ask, q, "CLOSE_LATE" if late else "CLOSE", read
            )

    # VWAP crossings (amendment 2) ---------------------------------------------------------------
    async def watch_crosses(self, now: datetime, s: Session | None) -> None:
        """On a long Nasdaq first-candle day, after each 5-minute close from 10:00, read QQQ's
        candles, replay the rule and bring the record into line: open at the crossing's close,
        close at the close back below VWAP. A crossing or exit not acted on within its grace is
        recorded missed, never filled at a later price."""
        today = s.day.isoformat() if s else None
        for t in self.store.open_crosses():
            if today is None or t["day"] < today:
                self.store.update(
                    "cross_trades", t["id"], {"exit_at": stamp(now), "exit_reason": "MISSED_EXIT"}
                )
        if s is None or not s.regular or not s.open + timedelta(minutes=30) <= now < s.close:
            return
        signal = self.store.signal(today or "", FC, methods.SIGNAL_MARKET)
        if signal is None or signal["side"] != 1:
            return
        due = s.open + methods.CANDLE * int((now - s.open) / methods.CANDLE)  # latest close
        if self.cross_done == due or time.monotonic() < self.cross_next:
            return
        self.cross_next = time.monotonic() + CROSS_POLL_SECONDS
        try:
            candles, delay = await market.etf_candles(self.client, QQQ_UIC)
        except SaxoError as exc:
            self.note(CROSS, exc)
            return
        if delay != 0:
            self.cross_next = time.monotonic() + CROSS_DELAYED_POLL_SECONDS
        crosses = methods.vwap_crosses(candles, s.day)
        settled = await self.reconcile_crosses(today or "", crosses, now, delay)
        if settled and candles and max(c.start for c in candles) >= due:
            self.cross_done = due  # the candle closing at `due` is final and acted on
            if delay == 0:
                self.problems.pop(CROSS, None)
        if delay != 0:
            self.problems[CROSS] = "QQQ_CHART_DELAYED"

    async def reconcile_crosses(
        self, day: str, crosses: list[methods.Cross], now: datetime, delay: int | None
    ) -> bool:
        settled = True
        for c in crosses:
            key, opens_at = stamp(c.entry), c.entry + methods.CANDLE
            row = self.store.cross_trade(day, key)
            if row is None and now > opens_at + ENTRY_GRACE:
                why = (
                    "APP_NOT_RUNNING_AT_ENTRY"
                    if self.started_at > opens_at
                    else "QQQ_CHART_DELAYED"
                    if delay
                    else self.problems.get(CROSS, "NOT_TRIED_IN_TIME")
                )
                self.store.insert(
                    "cross_trades",
                    self.cross_row(day, c)
                    | {"exit_at": stamp(now), "exit_reason": "MISSED_ENTRY", "note": why},
                )
                continue
            if row is None:
                if self.store.open_crosses():
                    continue  # one at a time, whatever the replay says
                try:
                    await self.open_cross(day, c, opens_at)
                except (SaxoError, Problem) as exc:
                    self.note(CROSS, exc)
                    settled = False
                continue
            if c.exit is None or row["exit_at"] is not None:
                continue
            closes_at = c.exit + methods.CANDLE
            if now > closes_at + EXIT_GRACE:
                self.store.update(
                    "cross_trades", row["id"], {"exit_at": stamp(now), "exit_reason": "MISSED_EXIT"}
                )
                continue
            try:
                q = await self.live_cfd()
            except (SaxoError, Problem) as exc:
                self.note(CROSS, exc)
                settled = False
                continue
            assert q.bid is not None
            late = self.now() > closes_at + ENTRY_GRACE
            self.close_cross(row, q, "BACK_BELOW_VWAP_LATE" if late else "BACK_BELOW_VWAP", c)
        return settled

    def cross_row(self, day: str, c: methods.Cross) -> dict[str, Any]:
        nas = next(x for x in CFDS if x.market == methods.SIGNAL_MARKET)
        return {
            "day": day,
            "signal_candle": stamp(c.entry),
            "market": nas.market,
            "uic": nas.uic,
            "qqq_close": c.entry_close,
            "qqq_vwap": round(c.entry_vwap, 4),
            "qqq_ema": round(c.entry_ema, 4),
        }

    async def live_cfd(self) -> market.Quote:
        uic = next(x.uic for x in CFDS if x.market == methods.SIGNAL_MARKET)
        q = (await market.quotes(self.client, [uic], "CfdOnIndex")).get(uic)
        if q is None or not q.live:
            raise Problem("CFD_QUOTE_NOT_LIVE")
        return q

    async def open_cross(self, day: str, c: methods.Cross, opens_at: datetime) -> None:
        nas = next(x for x in CFDS if x.market == methods.SIGNAL_MARKET)
        if nas.uic not in self.cfd_info:
            self.cfd_info[nas.uic] = await market.cfd_details(self.client, nas.uic)
        decimals, factor = self.cfd_info[nas.uic]
        q = await self.live_cfd()
        read = self.now()
        if read > opens_at + ENTRY_GRACE:
            raise Problem("ENTRY_DEADLINE_PASSED")
        assert q.ask is not None
        # Sized as the first-candle Nasdaq trade ($200 over a 100 bp move); the rule has no stop.
        amount = methods.cfd_amount(q.ask, nas.stop_bp, decimals, factor)
        if amount <= 0:
            raise Problem("CFD_SIZE_ROUNDS_TO_ZERO")
        self.store.insert(
            "cross_trades",
            self.cross_row(day, c)
            | {
                "amount": amount,
                "factor": factor,
                "entry_at": stamp(read),
                "entry_price": q.ask,
                "entry_bid": q.bid,
                "entry_ask": q.ask,
            },
        )
        log.info("%s %s cross long %.2f at %.2f", day, nas.market, amount, q.ask)

    def close_cross(
        self, t: sqlite3.Row, q: market.Quote, reason: str, c: methods.Cross | None = None
    ) -> None:
        assert q.bid is not None
        pnl = methods.cfd_pnl(1, t["entry_price"], q.bid, t["amount"], t["factor"])
        self.store.update(
            "cross_trades",
            t["id"],
            {
                "exit_candle": stamp(c.exit) if c and c.exit else None,
                "qqq_exit_close": c.exit_close if c else None,
                "exit_at": stamp(self.now()),
                "exit_price": q.bid,
                "exit_bid": q.bid,
                "exit_ask": q.ask,
                "exit_reason": reason,
                "pnl_usd": round(pnl, 2),
            },
        )
        log.info("%s cross closed %s %.2f", t["day"], reason, pnl)

    async def cross_exit(self, s: Session, due: datetime) -> None:
        trades = [t for t in self.store.open_crosses() if t["day"] == s.day.isoformat()]
        if not trades:
            raise Done("NO_OPEN_CROSS_TRADE")
        q = await self.live_cfd()
        read = self.now()
        for t in trades:
            if read > due + EXIT_GRACE:
                self.store.update(
                    "cross_trades", t["id"], {"exit_at": stamp(read), "exit_reason": "MISSED_EXIT"}
                )
                continue
            self.close_cross(t, q, "CLOSE_LATE" if read > due + ENTRY_GRACE else "CLOSE")

    # dip rules (amendment 3) ------------------------------------------------------------------
    async def dip_signal(self, s: Session) -> sqlite3.Row:
        """Today's Connors and Turnaround Tuesday readings, saved once and shared by the exit and
        entry events. The CFD trades at its live price, so QQQ's price must be live too."""
        day = s.day.isoformat()
        saved = self.store.signal(day, DIP, "QQQ")
        if saved is not None:
            return saved
        if self.now() >= s.close:
            raise Problem("SESSION_CLOSED")
        dated = await market.daily_closes(self.client, QQQ_UIC, s.day)
        if not dated or (s.day - dated[-1][0]).days > 4:
            raise Problem("DAILY_CLOSES_OUT_OF_DATE")
        q = (await market.quotes(self.client, [QQQ_UIC], "Etf")).get(QQQ_UIC)
        if q is None or q.mid is None:
            raise Problem("QQQ_QUOTE_UNAVAILABLE")
        if q.delayed != 0:
            raise Problem("QQQ_PRICE_DELAYED")
        closes = [close for _, close in dated]
        vol = await self.atm_vol_30d(s.day, q.mid)
        d = methods.dip_rules([*closes, q.mid], s.day, vol["atm_vol_30d"])
        self.store.add_signal(
            day,
            DIP,
            "QQQ",
            int(d.connors.signal or d.turnaround or d.fear),
            d.reason,
            {
                "price_15_50": q.mid,
                "previous_close": closes[-1],
                "change": d.change,
                "sma200": d.connors.sma200,
                "rsi2": d.connors.rsi,
                "rsi2_prev": d.connors.rsi_prev,
                "connors_in": d.connors.signal,
                "connors_exit": d.connors_exit,
                "turnaround": d.turnaround,
                "fear": d.fear,
                "down_closes": d.down_closes,
                "daily_closes": len(closes),
                "newest_close_day": dated[-1][0].isoformat(),
                **vol,
            },
        )
        saved = self.store.signal(day, DIP, "QQQ")
        assert saved is not None
        return saved

    async def atm_vol_30d(self, day: date, centre: float) -> dict[str, Any]:
        """Amendment 4: Saxo's MidVol of the QQQ call and put at the strike nearest the price on
        the expiry nearest 30 days out — the fear rule's reading. A read that fails is recorded
        with its reason; the other dip rules go ahead without it."""
        try:
            later = [e for e in await market.expiries(self.client, QQQ_OPTION_ROOT) if e > day]
            if not later:
                return {"atm_vol_30d": None, "atm_vol_reason": "NO_LATER_EXPIRY"}
            expiry = min(later, key=lambda e: abs((e - day).days - methods.FEAR_DAYS))
            listed = await market.series(self.client, QQQ_OPTION_ROOT, expiry)
            board, _ = await market.board(self.client, listed, centre, 1)
        except SaxoError as exc:
            return {"atm_vol_30d": None, "atm_vol_reason": failure_code(exc)}
        vol = methods.atm_vol(board, centre)
        return {
            "atm_vol_30d": vol,
            "atm_vol_expiry": expiry.isoformat(),
            "atm_vol_delay": max((o.delayed or 0 for o in board), default=None),
            "atm_vol_reason": "" if vol is not None else "NO_ATM_VOL",
        }

    async def dip_exit(self, s: Session, due: datetime) -> None:
        """Turnaround Tuesday closes at the next session; Connors at the first later 15:50 with
        RSI2 above 65. An exit day the app did not record leaves the trade without a price."""
        today = s.day.isoformat()
        trades = [t for t in self.store.open_dips() if t["day"] < today]
        if not trades:
            raise Done("NO_OPEN_DIP_TRADE")
        dated = await market.daily_closes(self.client, QQQ_UIC, s.day)
        if not dated or (s.day - dated[-1][0]).days > 4:
            raise Problem("DAILY_CLOSES_OUT_OF_DATE")
        d: dict[str, Any] | None = None  # today's readings: only Connors needs QQQ's live price
        pending = ""  # a Connors check that could not run yet; the other exits go ahead
        for t in trades:
            passed = [x.isoformat() for x, _ in dated if x.isoformat() > t["day"]]
            if t["rule"] == "TURNAROUND":  # sessions since its entry: its exit day has gone
                missed, due_now = bool(passed), True
            elif t["rule"] == "FEAR":  # out at the fifth session's 15:50; any later, it has gone
                missed = len(passed) >= methods.FEAR_HOLD
                due_now = len(passed) == methods.FEAR_HOLD - 1
            else:  # a session since its entry whose exit check the app did not complete
                missed = any(
                    (e := self.store.event(x, DIP + "_EXIT")) is None or e["status"] != "DONE"
                    for x in passed
                )
                if not missed and d is None:
                    try:
                        signal = await self.dip_signal(s)
                    except (SaxoError, Problem) as exc:
                        pending = failure_code(exc)
                        continue
                    if signal["reason"]:
                        pending = signal["reason"]
                        continue
                    d = json.loads(signal["detail"])
                due_now = not missed and d is not None and d["connors_exit"]
            if not (missed or due_now):
                continue
            q = None if missed else await self.live_cfd()
            read = self.now()
            if q is None or read >= s.close:  # never closed at a price from after the close
                self.store.update(
                    "dip_trades",
                    t["id"],
                    {"exit_day": today, "exit_at": stamp(read), "exit_reason": "MISSED_EXIT"},
                )
                continue
            assert q.bid is not None
            reason = DIP_EXITS[t["rule"]]
            pnl = methods.cfd_pnl(1, t["entry_price"], q.bid, t["amount"], t["factor"])
            self.store.update(
                "dip_trades",
                t["id"],
                {
                    "exit_day": today,
                    "exit_at": stamp(read),
                    "exit_price": q.bid,
                    "exit_bid": q.bid,
                    "exit_ask": q.ask,
                    "exit_reason": reason + ("_LATE" if read > due + ENTRY_GRACE else ""),
                    "qqq_exit_price": d["price_15_50"] if d else None,
                    "pnl_usd": round(pnl, 2),
                },
            )
            log.info("%s dip %s closed %s %.2f", t["day"], t["rule"], reason, pnl)
        # Retried until the exit's grace, then FAILED: the next session records the trade missed.
        if pending:
            raise Problem(pending)

    async def dip_entry(self, s: Session, due: datetime) -> None:
        """Each rule holds one trade at a time; the book flag marks the trades a one-at-a-time
        book would hold (the first dip trade while none of its own is open)."""
        day = s.day.isoformat()
        if not s.regular:
            raise Done("EARLY_CLOSE_DAY_SKIPPED")
        signal = await self.dip_signal(s)
        if signal["reason"]:
            raise Done(signal["reason"])
        d = json.loads(signal["detail"])
        held = self.store.open_dips()
        wanted = [
            rule
            for rule, key in DIP_RULES
            if d[key] and not self.store.dip(day, rule) and all(t["rule"] != rule for t in held)
        ]
        if not wanted:
            raise Done("ALREADY_IN_A_TRADE" if signal["side"] else "NO_DIP_SIGNAL")
        nas = next(x for x in CFDS if x.market == methods.SIGNAL_MARKET)
        if nas.uic not in self.cfd_info:
            self.cfd_info[nas.uic] = await market.cfd_details(self.client, nas.uic)
        decimals, factor = self.cfd_info[nas.uic]
        q = await self.live_cfd()
        read = self.now()
        if read > due + ENTRY_GRACE:
            raise Problem("ENTRY_DEADLINE_PASSED")
        assert q.ask is not None and q.bid is not None
        # Sized as the first-candle Nasdaq trade ($200 over a 100 bp move); the rules have no stop.
        amount = methods.cfd_amount(q.ask, nas.stop_bp, decimals, factor)
        if amount <= 0:
            raise Problem("CFD_SIZE_ROUNDS_TO_ZERO")
        book_busy = any(t["book"] for t in held)
        for rule in wanted:
            self.store.insert(
                "dip_trades",
                {
                    "day": day,
                    "rule": rule,
                    "market": nas.market,
                    "uic": nas.uic,
                    "book": int(not book_busy),
                    "qqq_price": d["price_15_50"],
                    "amount": amount,
                    "factor": factor,
                    "entry_at": stamp(read),
                    "entry_price": q.ask,
                    "entry_bid": q.bid,
                    "entry_ask": q.ask,
                },
            )
            book_busy = True
            log.info("%s dip %s long %.2f at %.2f", day, rule, amount, q.ask)

    # stock dips (amendment 5) -----------------------------------------------------------------
    def stock_night(self, day: str) -> Callable[[datetime, int], Awaitable[None]]:
        night = date.fromisoformat(day)

        async def run(due: datetime, delay: int) -> None:
            await self.stock_dips(night, due)

        return run

    async def stock_dips(self, day: date, due: datetime) -> None:
        """After the close, on Saxo's daily bars: last night's orders filled or not at the
        session's low, open positions closed on their rule at the close, and tonight's orders for
        the next session. Three stages: Saxo's US stock list, one price read per 50 stocks (the
        day's last price and volume), then daily bars for the stocks that pass."""
        key = day.isoformat()
        listed = self.stock_uics = await market.us_stocks(self.client)
        held = {t["code"] for t in self.store.open_stock_trades()}
        held |= {o["code"] for o in self.store.pending_stock_orders()}
        snaps = await market.stock_snapshots(self.client, sorted(set(listed.values())))
        keep = {c for c in held if c in listed}
        for code, uic in listed.items():
            q = snaps.get(uic)
            if (
                q is not None
                and q.last
                and q.volume
                and q.last >= methods.STOCK_MIN_PRICE
                and q.last * q.volume >= STOCK_STORE_DOLLARS
            ):
                keep.add(code)
        # SPY first: its bar for the day says whether the day's bars are out at all.
        await self.stock_history("SPY", methods.SPY_UIC, "Etf", key)
        read = failed = 0
        for code in sorted(keep):
            try:
                await self.stock_history(code, listed[code], "Stock", key)
                read += 1
            except (SaxoError, Problem) as exc:
                failed += 1  # this stock sits out tonight; its stored bars stay as they were
                log.info("%s stock dips: %s skipped, %s", key, code, failure_code(exc))
        self.store.prune_stock_bars((day - timedelta(days=STOCK_KEEP_DAYS)).isoformat())
        bars = {c: self.store.latest_stock_bar(c) for c in held}
        filled = closed = 0
        for o in self.store.pending_stock_orders():
            if o["signal_day"] >= key:
                continue
            bar = bars.get(o["code"])
            if bar is None or bar.day != key:
                self.store.update(
                    "stock_orders",
                    o["id"],
                    {"status": "CANCELLED", "fill_day": key, "note": "NO_BAR"},
                )
                continue
            price = methods.stock_fill(o["limit_price"], bar)
            if price is None:
                self.store.update(
                    "stock_orders", o["id"], {"status": "NOT_FILLED", "fill_day": key}
                )
                continue
            self.store.update(
                "stock_orders", o["id"], {"status": "FILLED", "fill_day": key, "fill_price": price}
            )
            self.store.insert(
                "stock_trades",
                {
                    "order_id": o["id"],
                    "code": o["code"],
                    "tier": o["tier"],
                    "signal_day": o["signal_day"],
                    "entry_day": key,
                    "entry_price": price,
                    "notional_usd": methods.STOCK_NOTIONAL_USD,
                },
            )
            filled += 1
            log.info("%s stock dip %s bought at %.4f", key, o["code"], price)
        for t in self.store.open_stock_trades():
            series = self.store.stock_bars(t["code"])
            if not series:
                continue
            last = series[-1]
            if last.day != key:
                if self.store.sessions_after(last.day) >= methods.STOCK_NO_BAR_SESSIONS:
                    self.close_stock(t, last, "NO_BAR_10_SESSIONS", None, key)
                    closed += 1
                continue
            crsi = methods.connors_rsi([b.adj for b in series])
            if crsi is not None and crsi > methods.STOCK_CRSI_OUT:
                self.close_stock(t, last, "CRSI_ABOVE_70", crsi, key)
                closed += 1
        series_by_code = self.store.all_stock_bars()
        universe = methods.stock_universe(series_by_code, series_by_code["SPY"])
        busy = {t["code"] for t in self.store.open_stock_trades()}
        busy |= {o["code"] for o in self.store.pending_stock_orders()}
        orders = 0
        for code, (rank, tier) in universe.items():
            if code in busy or series_by_code[code][-1].day != key:
                continue
            setup = methods.stock_setup(code, series_by_code[code])
            if setup is None:
                continue
            self.store.insert(
                "stock_orders",
                {
                    "signal_day": key,
                    "code": code,
                    "close": setup.close,
                    "limit_price": setup.limit,
                    "sell_off": setup.sell_off,
                    "closing_range": setup.closing_range,
                    "adx": setup.adx,
                    "crsi": setup.crsi,
                    "rank": rank,
                    "tier": tier,
                    "status": "PENDING",
                },
            )
            orders += 1
        self.store.add_signal(
            key,
            "STOCK",
            "US",
            int(orders > 0),
            "",
            {
                "listed": len(listed),
                "priced": len(snaps),
                "stocks_kept": len(keep),
                "histories_read": read,
                "histories_failed": failed,
                "universe": len(universe),
                "orders": orders,
                "filled": filled,
                "closed": closed,
                "saxo_calls": self.client.calls,
            },
        )
        log.info("%s stock dips: %d orders, %d filled, %d closed", key, orders, filled, closed)

    async def stock_history(self, code: str, uic: int, asset_type: str, key: str) -> None:
        """Bring a stock's stored daily bars up to the day `key`: a few newest bars on an ordinary
        night; the whole history when the stock is new to the record, has a gap, or Saxo now
        shows a different close for a stored day (an adjustment). The day's own bar must be there,
        else the day's bars are not out yet."""
        stored = self.store.stock_bars(code)
        fresh = await market.stock_daily(self.client, uic, asset_type, STOCK_REFRESH_COUNT)
        by_day = {b.day: b for b in stored}
        rewritten = any(
            abs(b.close / by_day[b.day].close - 1) > STOCK_REWRITE_TOLERANCE
            for b in fresh
            if b.day in by_day
        )
        gap = bool(stored) and bool(fresh) and fresh[0].day > stored[-1].day
        if len(stored) < methods.STOCK_HISTORY or rewritten or gap:
            fresh = await market.stock_daily(self.client, uic, asset_type, STOCK_CHART_COUNT)
            if not fresh:
                raise Problem("NO_DAILY_BARS")
            self.store.replace_stock_bars(code, fresh)
        else:
            for b in fresh:
                self.store.put_stock_bar(code, b)
        if not fresh or fresh[-1].day != key:
            raise Problem("DAILY_BARS_NOT_PUBLISHED")

    def close_stock(
        self, t: sqlite3.Row, bar: methods.StockBar, reason: str, crsi: float | None, day: str
    ) -> None:
        pnl = methods.stock_pnl(t["entry_price"], bar.close, t["tier"])
        self.store.update(
            "stock_trades",
            t["id"],
            {
                "exit_day": day,
                "exit_price": bar.close,
                "exit_reason": reason,
                "crsi_exit": None if crsi is None else round(crsi, 4),
                "pnl_usd": round(pnl, 2),
            },
        )
        log.info("%s stock dip %s closed %s %.2f", day, t["code"], reason, pnl)

    # QQQ options ------------------------------------------------------------------------------
    async def qqq_price(self, delay: int) -> float:
        q = (await market.quotes(self.client, [QQQ_UIC], "Etf")).get(QQQ_UIC)
        if q is None or q.mid is None:
            raise Problem("QQQ_QUOTE_UNAVAILABLE")
        if q.delayed != delay:
            raise Problem("QQQ_DELAY_CHANGED")
        return q.mid

    async def choose(
        self,
        due: datetime,
        day: str,
        method: str,
        side: int,
        expiry: date,
        put_call: str,
        target: float,
        band: tuple[float, float],
        delay: int,
    ) -> None:
        centre = await self.qqq_price(delay)
        listed = await market.series(self.client, QQQ_OPTION_ROOT, expiry)
        board, raw = await market.board(self.client, listed, centre, methods.BOARD_HALF_WIDTH)
        read = self.now()  # the board's prices apply to read - delay
        if any(o.delayed != delay for o in board):
            raise Problem("QQQ_DELAY_CHANGED")
        if read - timedelta(minutes=delay) > due + ENTRY_GRACE:
            raise Problem("ENTRY_DEADLINE_PASSED")
        self.store.snapshot(
            day,
            method + "_ENTRY",
            {
                "read": stamp(read),
                "delay": delay,
                "underlying": centre,
                "expiry": expiry.isoformat(),
                "rows": [vars(o) for o in board],
            },
        )
        pick = methods.pick_option(board, put_call, target, band)
        if pick is None:
            raise Done("NO_QUALIFYING_STRIKE")
        assert pick.bid is not None and pick.ask is not None
        self.store.insert(
            "option_trades",
            {
                "day": day,
                "method": method,
                "side": side,
                "uic": pick.uic,
                "expiry": expiry.isoformat(),
                "put_call": pick.put_call,
                "strike": pick.strike,
                "exit_day": expiry.isoformat() if method == CONNORS_QQQ else day,
                "entry_for": stamp(read - timedelta(minutes=delay)),
                "entry_read": stamp(read),
                "entry_delay": delay,
                "entry_bid": pick.bid,
                "entry_ask": pick.ask,
                "entry_delta": pick.delta,
                "entry_vol": pick.mid_vol,
                "entry_underlying": centre,
                "cost_buy": pick.cost_buy,
            },
        )
        log.info("%s %s %s %s at %.2f", day, method, pick.put_call, pick.strike, pick.ask)

    async def option_entry(self, s: Session, due: datetime, delay: int) -> None:
        day = s.day.isoformat()
        if not s.regular:
            raise Done("EARLY_CLOSE_DAY_SKIPPED")
        if self.store.option(day, FC_QQQ):
            return
        signal = self.store.signal(day, FC, methods.SIGNAL_MARKET)
        first = self.store.event(day, FC)
        if signal is None and (first is None or first["status"] not in {"FAILED", "MISSED"}):
            raise Problem("WAITING_FOR_NASDAQ_SIGNAL")
        if signal is None or signal["side"] == 0:
            raise Done("NO_NASDAQ_SIGNAL")
        if s.day not in await market.expiries(self.client, QQQ_OPTION_ROOT):
            raise Done("NO_SAME_DAY_EXPIRY")
        await self.choose(
            due,
            day,
            FC_QQQ,
            signal["side"],
            s.day,
            "Call" if signal["side"] > 0 else "Put",
            methods.FIRST_CANDLE_DELTA,
            methods.FIRST_CANDLE_DELTA_RANGE,
            delay,
        )

    async def mark(self, t: sqlite3.Row, delay: int) -> tuple[datetime, methods.OptionQuote, float]:
        underlying = await self.qqq_price(delay)
        q = await market.option_quote(
            self.client, {"Uic": t["uic"], "StrikePrice": t["strike"], "PutCall": t["put_call"]}
        )
        read = self.now()  # the quotes apply to read - delay
        if q is None:
            raise Problem("OPTION_QUOTE_UNAVAILABLE")
        if q.delayed != delay:
            raise Problem("QQQ_DELAY_CHANGED")
        return read, q, underlying

    def missed_exit(self, t: sqlite3.Row, as_of: datetime, delay: int) -> None:
        """No price inside the exit's window (the app was down, or the market had closed)."""
        self.store.update(
            "option_trades",
            t["id"],
            {
                "exit_read": stamp(self.now()),
                "exit_for": stamp(as_of),
                "exit_delay": delay,
                "exit_reason": "MISSED_EXIT",
            },
        )

    async def close_option(
        self, t: sqlite3.Row, delay: int, reason: str, due: datetime, close: datetime
    ) -> None:
        if self.now() - timedelta(minutes=delay) >= close:
            self.missed_exit(t, self.now() - timedelta(minutes=delay), delay)
            return
        read, q, underlying = await self.mark(t, delay)
        as_of = read - timedelta(minutes=delay)
        if as_of >= close or as_of > due + EXIT_GRACE:
            self.missed_exit(t, as_of, delay)
            return
        bid = q.bid or 0.0  # no bid: the option cannot be sold, so it is worth nothing
        reason += "_LATE" if as_of > due + ENTRY_GRACE else ""
        self.store.update(
            "option_trades",
            t["id"],
            {
                "exit_for": stamp(as_of),
                "exit_read": stamp(read),
                "exit_delay": delay,
                "exit_bid": bid,
                "exit_ask": q.ask,
                "exit_delta": q.delta,
                "exit_underlying": underlying,
                "cost_sell": q.cost_sell,
                "exit_reason": reason if q.bid else reason + "_NO_BID",
                "pnl_usd": methods.option_pnl(t["entry_ask"], bid, t["cost_buy"], q.cost_sell),
            },
        )

    async def option_exit(self, s: Session, due: datetime, delay: int) -> None:
        today = s.day.isoformat()
        for old in self.store.open_options():  # a same-day option left open on an earlier day
            if old["method"] == FC_QQQ and old["day"] < today:
                self.missed_exit(old, due, delay)
        t = self.store.option(today, FC_QQQ)
        if t is None or t["exit_read"] is not None:
            raise Done("NO_OPEN_OPTION")
        await self.close_option(t, delay, "AT_13_00", due, s.close)

    async def option_late(self, s: Session, due: datetime, delay: int) -> None:
        """Reported only: the same option's quote five minutes before the close."""
        t = self.store.option(s.day.isoformat(), FC_QQQ)
        if t is None:
            raise Done("NO_OPTION_TODAY")
        read, q, underlying = await self.mark(t, delay)
        as_of = read - timedelta(minutes=delay)
        if as_of >= s.close or as_of > due + ENTRY_GRACE:
            raise Done("LATE_MARK_MISSED")
        self.store.update(
            "option_trades",
            t["id"],
            {
                "late_for": stamp(as_of),
                "late_bid": q.bid,
                "late_ask": q.ask,
                "late_underlying": underlying,
            },
        )

    async def connors(self, s: Session, due: datetime, delay: int) -> None:
        day = s.day.isoformat()
        if not s.regular:
            raise Done("EARLY_CLOSE_DAY_SKIPPED")
        if self.store.option(day, CONNORS_QQQ):
            return
        if self.store.signal(day, "CONNORS", "QQQ") is None:
            dated = await market.daily_closes(self.client, QQQ_UIC, s.day)
            # The newest close must be the last session's: at most a long weekend old.
            if not dated or (s.day - dated[-1][0]).days > 4:
                self.store.add_signal(
                    day,
                    "CONNORS",
                    "QQQ",
                    0,
                    "DAILY_CLOSES_OUT_OF_DATE",
                    {"newest_close_day": dated[-1][0].isoformat() if dated else None},
                )
                raise Done("DAILY_CLOSES_OUT_OF_DATE")
            closes = [close for _, close in dated]
            price = await self.qqq_price(delay)
            c = methods.connors([*closes, price])
            self.store.add_signal(
                day,
                "CONNORS",
                "QQQ",
                int(c.signal),
                c.reason,
                {
                    "price_15_50": c.close,
                    "sma200": c.sma200,
                    "rsi2": c.rsi,
                    "rsi2_prev": c.rsi_prev,
                    "daily_closes": len(closes),
                    "newest_close_day": dated[-1][0].isoformat(),
                },
            )
        signal = self.store.signal(day, "CONNORS", "QQQ")
        assert signal is not None
        if signal["side"] != 1:
            raise Done("NO_CONNORS_SIGNAL" if not signal["reason"] else signal["reason"])
        later = [e for e in await market.expiries(self.client, QQQ_OPTION_ROOT) if e > s.day]
        if not later:
            raise Done("NO_LATER_EXPIRY")
        await self.choose(
            due,
            day,
            CONNORS_QQQ,
            1,
            later[0],
            "Call",
            methods.CONNORS_DELTA,
            methods.CONNORS_DELTA_RANGE,
            delay,
        )

    async def connors_exit(self, s: Session, due: datetime, delay: int) -> None:
        trades = [t for t in self.store.open_options() if t["method"] == CONNORS_QQQ]
        today = s.day.isoformat()
        for t in trades:
            if t["exit_day"] < today:  # the app was not running on its expiry day
                self.missed_exit(t, due, delay)
        due_today = [t for t in trades if t["exit_day"] == today]
        if not due_today:
            raise Done("NO_CONNORS_EXIT_TODAY")
        for t in due_today:
            await self.close_option(t, delay, "BEFORE_CLOSE", due, s.close)

    # dashboard -------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        oauth = self.client.oauth
        refresh_left = float(oauth.tokens.get("refresh_expires_at", 0)) - time.time()
        now = self.now()
        day = now.astimezone(NY).date()
        session = methods.session_on(self.sessions, day) if self.sessions_day == day else None
        return {
            "server_time": stamp(now),
            "build": os.environ.get("STOCKER_BUILD_REVISION", "development"),
            "data_environment": self.config.data_environment,
            "worker_health": self.worker_health,
            "web_health": self.web_health,
            "oauth": oauth.status,
            "oauth_failure": oauth.failure_reason,
            "login_minutes_left": round(refresh_left / 60) if oauth.tokens else None,
            "trade_level": self.capabilities.get("TradeLevel"),
            "data_level": self.capabilities.get("DataLevel"),
            "qqq_delay_minutes": self.qqq_delay,
            "session": None
            if session is None
            else {
                "open": stamp(session.open),
                "close": stamp(session.close),
                "regular": session.regular,
            },
            "calendar_loaded": self.sessions_day.isoformat() if self.sessions_day else None,
            "problems": dict(sorted(self.problems.items())),
            "rate_limits": dict(sorted(self.client.rate_headers.items())),
            "saxo_calls": self.client.calls,
        }

    def today(self) -> dict[str, Any]:
        now = self.now()
        day = now.astimezone(NY).date()
        view = self.store.day_view(day.isoformat())
        # Open CFD positions (first candle, crossings, dips) marked at the stop watcher's latest
        # quote while it is fresh; otherwise they are shown open without a figure, never at a
        # stale price. Display only.
        fresh = time.monotonic() - self.cfd_polled < MARK_FRESH_SECONDS
        for t in (*view["cfd"], *view["crosses"], *view["dips"]):
            q = self.cfd_quotes.get(t["uic"])
            if (
                fresh
                and t["exit_at"] is None
                and t["entry_price"] is not None
                and q is not None
                and q.bid
                and q.ask
            ):
                side = t.get("side", 1)  # crossings and dips are always long
                mark = q.bid if side > 0 else q.ask
                t["open_pnl_usd"] = round(
                    methods.cfd_pnl(side, t["entry_price"], mark, t["amount"], t["factor"]), 2
                )
        # Stock dips: today's latest price read beside the orders resting today and the open
        # positions, with each position's result if sold at it. Display only.
        for row in (*view["stock_orders"], *view["stock_trades"]):
            m = self.stock_marks.get(row["code"])
            if m is None or at(m["now_at"]).astimezone(NY).date() != day:
                continue
            if row.get("status") == "PENDING" and row["signal_day"] < day.isoformat():
                row.update({k: v for k, v in m.items() if k != "crsi_now"})
            elif "entry_price" in row and row["exit_day"] is None:
                row.update(m)
                pnl = methods.stock_pnl(row["entry_price"], m["now_price"], row["tier"])
                row["open_pnl_usd"] = round(pnl, 2)
        session = methods.session_on(self.sessions, day) if self.sessions_day == day else None
        plan = []
        if session is not None:
            for name, due, delayed, _ in self.plan(session):
                plan.append(
                    {
                        "event": name,
                        "due": stamp(due),
                        "qqq_delayed": delayed,
                        "new_york": due.astimezone(NY).strftime("%H:%M"),
                        "uk": due.astimezone(UK).strftime("%H:%M"),
                    }
                )
        return {
            "day": day.isoformat(),
            "plan": plan,
            **view,
            "playbook": self.playbook(view["signals"], day),
            "stock_scan": self.store.latest_signal("STOCK", "US"),
            "system": self.status(),
        }

    @staticmethod
    def playbook(signals: list[dict[str, Any]], day: date) -> list[dict[str, Any]]:
        """Today's action per CFD: the plain trade, skipped when the two signals disagree."""
        fc = {s["market"]: s for s in signals if s["method"] == FC}
        out = []
        for cfd in CFDS:
            s = fc.get(cfd.market)
            if s is None:
                continue
            other = fc.get(next(x.market for x in CFDS if x.market != cfd.market))
            agree = (
                None
                if other is None or not other["side"] or not s["side"]
                else other["side"] == s["side"]
            )
            d = methods.Day(s["side"], day.weekday(), agree, s["detail"])
            action, tags = methods.playbook(d, cfd.market)
            out.append({"market": cfd.market, "action": action, "tags": tags})
        return out

    async def stop(self) -> None:
        if self.stopping and self.owner is None:
            return
        self.stopping = True
        await self.client.close()
        log.removeHandler(self.log_handler)
        self.log_handler.close()
        if self.owner:
            self.owner.close()
            self.owner = None

    def report_failure(self, worker: str, exc: BaseException) -> None:
        log.error("%s failed: %s", worker, failure_code(exc))
