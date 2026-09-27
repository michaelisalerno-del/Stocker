"""Continuous six-market monitoring, with independent position management."""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ib_async import Future
from ib_async.objects import BarData

from stocker_execution.broker import PaperBroker, now
from stocker_execution.config import MARKETS, RULE_VERSION, FuturesConfig
from stocker_execution.contracts import EXCHANGES, Calendar, nearby_futures, select_future, utc
from stocker_execution.depth import DepthObserver
from stocker_execution.pacing import CORE, priority
from stocker_execution.rules import (
    NY,
    Bar,
    clocks,
    eligibility,
    next_clock,
    opportunity,
    prior_rv,
    reference_summary,
)
from stocker_execution.store import Store, encode

log = logging.getLogger(__name__)


@dataclass
class MarketState:
    market: str
    detail: Any = None
    calendar: Calendar | None = None
    stream: Any = None
    ticker: Any = None
    bars: list[Bar] = field(default_factory=list)
    references: list[dict[int, dict[str, float]]] = field(default_factory=list)
    problem: str = "NOT_CONNECTED"
    last_update: datetime | None = None
    live_since: datetime | None = None
    last_clock: datetime | None = None
    selected_day: date | None = None
    last_attempt: datetime | None = None
    features: dict[str, float] = field(default_factory=dict)
    gap_repair_at: datetime | None = None


class Runtime:
    def __init__(self, config: FuturesConfig, store: Store, broker: PaperBroker | None = None):
        self.config, self.store = config, store
        self.broker = broker or PaperBroker(config, store)
        self.markets = {m: MarketState(m) for m in MARKETS}
        self.worker_health = self.manager_health = self.web_health = "STARTING"
        self.stopping = False
        self.tasks: set[asyncio.Task[Any]] = set()
        self.owner: Any = None
        db_path = Path(store.db.execute("PRAGMA database_list").fetchone()[2])
        self.depth = DepthObserver(self.broker.data, store, config.l2, db_path.with_suffix(".l2"))
        self.core_restored = False

    @property
    def pause(self) -> bool:
        return bool(self.store.get_meta("paused", False))

    @pause.setter
    def pause(self, value: bool) -> None:
        self.store.set_meta("paused", value)

    def report_failure(self, worker: str, exc: BaseException) -> None:
        log.error("%s failed: %s", worker, exc)

    async def cancel_tasks(self, tasks: set[asyncio.Task[Any]]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def status(self) -> dict[str, Any]:
        return {
            "application": "SLRNO",
            "environment": "PAPER",
            "account": self.config.expected_account,
            "connected": self.broker.ib.isConnected(),
            "reconciled": self.broker.reconciled,
            "armed": self.config.armed,
            "paused": self.pause,
            "entry_block_reason": self.broker.entry_reason(),
            "problem": self.broker.problem,
            "worker_health": self.worker_health,
            "manager_health": self.manager_health,
            "web_health": self.web_health,
            "rule_version": RULE_VERSION,
            "reconciliation": self.store.get_meta("reconciliation"),
            "management_problems": self.broker.management_problems,
            "live_available": False,
            "market_data": self.broker.data.snapshot(),
            "l2_recording": self.depth.status(),
            **self.store.capacity(),
        }

    def overview(self) -> dict[str, Any]:
        active = self.store.active()
        cards = []
        for market, state in self.markets.items():
            market_status, next_market, trade_date = (
                state.calendar.state(now()) if state.calendar else ("BLOCKED", None, None)
            )
            mapping_reason = (
                ""
                if market in self.config.mappings
                else "LISTED_PRODUCT_AND_DELTA_TOLERANCE_UNAPPROVED"
            )
            reason = state.problem or mapping_reason or self.broker.entry_reason()
            local = now().astimezone(NY)
            outside_window = local.weekday() > 4 or local.hour < 9 or local.hour > 16
            positions = [r for r in active if r["market"] == market]
            qty = sum(self.store.exposure(r["id"]) for r in positions)
            strategy = (
                "POSITION_OPEN"
                if qty
                else "ORDER_PENDING"
                if positions
                else "BLOCKED"
                if reason
                else "MONITORING"
            )
            if not positions and market_status in {"CLOSED", "MAINTENANCE"}:
                strategy = market_status
            elif not positions and "WARMING" in reason:
                strategy = "WARMING_UP"
            elif not positions and outside_window:
                strategy = "OUTSIDE_ENTRY_WINDOW"
            contract = state.detail.contract if state.detail else None
            l1 = next(
                (
                    s
                    for s in self.broker.data.items.values()
                    if f"core:{market}" in s.consumers and s.feed == "QUOTE"
                ),
                None,
            )
            values = []
            for row in positions:
                plan = json.loads(row["plan"])
                quote = self.broker.option_quotes.get(plan["option"]["conId"])
                valuation: dict[str, Any] = {
                    "value_gbp": None,
                    "method": "BID_MINUS_AVERAGE_PREMIUM_AT_CURRENT_FX_BEFORE_COMMISSIONS",
                    "quote_at": quote.bid_at.isoformat() if quote else None,
                    "fresh": False,
                    "fx_rate": self.broker.fx[0] if self.broker.fx else None,
                    "fx_at": self.broker.fx[1].isoformat() if self.broker.fx else None,
                }
                if quote:
                    try:
                        quote.validate(now())
                        fills = self.store.fills(row["id"])
                        fx = self.broker.fx
                        if fx and 0 <= (now() - fx[1]).total_seconds() <= 30 and fills:
                            native_entry = sum(
                                f["quantity"] * f["price"] for f in fills if f["side"] == "BOT"
                            )
                            bought = sum(f["quantity"] for f in fills if f["side"] == "BOT")
                            remaining = self.store.exposure(row["id"])
                            valuation.update(
                                value_gbp=(quote.bid - native_entry / bought)
                                * remaining
                                * plan["multiplier"]
                                * plan["price_unit_factor"]
                                * fx[0],
                                fresh=True,
                            )
                    except ValueError:
                        pass
                values.append(
                    {
                        "id": row["id"],
                        "state": row["state"],
                        "quantity": self.store.exposure(row["id"]),
                        "entry_at": next(
                            (f["at"] for f in self.store.fills(row["id"]) if f["side"] == "BOT"),
                            None,
                        ),
                        "exit_at": row["exit_at"],
                        "option": plan["option"],
                        "valuation": valuation,
                    }
                )
            cards.append(
                {
                    "market": market,
                    "contract": contract.localSymbol if contract else None,
                    "market_status": market_status,
                    "data_status": "CURRENT"
                    if state.last_update and (now() - state.last_update).total_seconds() < 90
                    else "STALE_OR_MISSING",
                    "l1": {
                        "status": l1.state if l1 else "DISCONNECTED",
                        "data_type": state.ticker.marketDataType if state.ticker else None,
                        "last_receipt": l1.last_receipt.isoformat()
                        if l1 and l1.last_receipt
                        else None,
                        "last_change": l1.last_change.isoformat()
                        if l1 and l1.last_change
                        else None,
                    },
                    "updated_at": state.last_update.isoformat() if state.last_update else None,
                    "entry_enabled": not reason and market_status == "OPEN" and not outside_window,
                    "strategy_state": strategy,
                    "direction": "CALL / LONG"
                    if market in {"BTC", "CL"}
                    else "PUT / SHORT DIRECTION",
                    "block_reason": reason,
                    "conditions": state.features,
                    "chart": [{"at": b.at.isoformat(), "close": b.close} for b in state.bars[-90:]],
                    "signals": [
                        dict(r)
                        for r in self.store.db.execute(
                            "SELECT id,signal_at,decision,reason FROM signals "
                            "WHERE market=? AND signal_at>=? ORDER BY signal_at DESC LIMIT 50",
                            (
                                market,
                                state.bars[-90].at.isoformat()
                                if len(state.bars) >= 90
                                else state.bars[0].at.isoformat()
                                if state.bars
                                else now().isoformat(),
                            ),
                        )
                    ],
                    "trades": values,
                    "l2": self.depth.view(market),
                    "next_time": next_clock(now()).isoformat(),
                    "next_market_time": next_market.isoformat() if next_market else None,
                    "exchange_trade_date": trade_date,
                    "rule_version": RULE_VERSION,
                    "diagnostic": "Experimental management disabled" if market == "GC" else "",
                    "details": {
                        "signal_contract": contract.dict() if contract else None,
                        "mapping": self.config.mappings[market].model_dump()
                        if market in self.config.mappings
                        else None,
                        "reference_sessions": len(state.references),
                        "entry_timezone": "America/New_York",
                        "entry_clocks": "09:00–16:00 weekdays; NG 13:00 veto",
                        "exit_anchor": "Original opportunity + 60 minutes",
                        "volume": "Observation only",
                    },
                }
            )
        return {"system": self.status(), "markets": cards, "pnl": self.store.economics()}

    async def manager(self) -> None:
        self.manager_health = "RUNNING"
        while not self.stopping:
            try:
                if self.broker.ib.isConnected() and self.broker.synchronized:
                    await self.broker.manage()
            except Exception as exc:
                self.broker.reconciled = False
                self.broker.problem = str(exc)
                self.report_failure("management", exc)
            await asyncio.sleep(2)

    async def optional_data(self) -> None:
        try:
            await self.depth.start()
        except Exception as exc:
            self.depth.paused = f"OPTIONAL_RECORDING_START_FAILED:{exc}"
            return
        while not self.stopping:
            try:
                candidates = {
                    m: s.detail
                    for m, s in self.markets.items()
                    if s.detail and s.calendar and s.calendar.state(now())[0] == "OPEN"
                }
                await self.depth.tick(
                    candidates,
                    {m: next_clock(now()) for m in MARKETS},
                    self.broker.synchronized and self.core_restored,
                    {
                        m
                        for m, s in self.markets.items()
                        if s.calendar and s.calendar.state(now())[0] in {"CLOSED", "MAINTENANCE"}
                    },
                )
            except Exception as exc:
                self.depth.paused = f"OPTIONAL_L2_FAILURE:{type(exc).__name__}:{exc}"
                log.warning("Optional depth paused: %s", exc)
            await asyncio.sleep(1)

    async def run(self) -> None:
        # Lock the namespace for its lifetime; SQLite admission alone does not own the socket.
        import fcntl

        db_path = Path(self.store.db.execute("PRAGMA database_list").fetchone()[2])
        self.owner = db_path.with_suffix(".owner.lock").open("a")
        fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.worker_health = "RUNNING"
        manager = asyncio.create_task(self.manager())
        self.tasks.add(manager)
        self.tasks.add(asyncio.create_task(self.optional_data()))
        try:
            while not self.stopping:
                try:
                    if not self.broker.ib.isConnected() or not self.broker.synchronized:
                        self.core_restored = False
                        for state in self.markets.values():
                            state.live_since = None
                            state.problem = "BROKER_RECONNECTING"
                            state.stream = state.ticker = None
                            state.last_attempt = None
                        await self.broker.connect()
                    # Daily rollover/bootstrap is isolated from the independent exit manager.
                    for state in self.markets.values():
                        feeds = [
                            s
                            for s in self.broker.data.items.values()
                            if f"core:{state.market}" in s.consumers
                        ]
                        if state.live_since and len(feeds) < 2:
                            state.problem = "CORE_DATA_PREEMPTED_OR_DISCONNECTED"
                            state.live_since = None
                        failed = next(
                            (
                                s
                                for s in self.broker.data.items.values()
                                if f"core:{state.market}" in s.consumers and s.state == "FAILED"
                            ),
                            None,
                        )
                        if failed:
                            state.problem = failed.error
                            state.live_since = None
                        if (
                            state.live_since is None
                            or state.selected_day != now().astimezone(NY).date()
                        ) and (
                            not state.last_attempt
                            or now() - state.last_attempt >= timedelta(minutes=5)
                        ):
                            state.last_attempt = now()
                            try:
                                await self.monitor(state)
                            except Exception as exc:
                                state.problem = str(exc)
                        elif state.stream is not None:
                            await self.repair_gap(state)
                    self.core_restored = True
                    await self.decisions()
                except Exception as exc:
                    self.broker.reconciled = False
                    self.broker.problem = str(exc)
                    self.report_failure("monitor", exc)
                    await asyncio.sleep(10)
                await asyncio.sleep(0.5)
        finally:
            await self.stop()

    async def stop(self) -> None:
        self.stopping = True
        await self.cancel_tasks(self.tasks)
        await self.depth.close()
        for state in self.markets.values():
            await self.unsubscribe(state)
        await self.broker.data.close()
        if self.broker.ib.isConnected():
            self.broker.ib.disconnect()
        if self.owner:
            self.owner.close()
            self.owner = None

    async def unsubscribe(self, state: MarketState) -> None:
        await self.broker.data.release_owner(f"core:{state.market}")
        state.stream = state.ticker = None

    async def history(
        self, contract: Any, duration: str, size: str = "1 min", end: Any = "", stream: bool = False
    ) -> Any:
        if stream:
            raise ValueError("STREAMS_MUST_USE_SHARED_SUBSCRIPTION_OWNER")
        bars = await self.broker.data.historical(
            (contract.conId, duration, size, str(end)),
            lambda: self.broker.ib.reqHistoricalDataAsync(
                contract,
                end,
                duration,
                size,
                "TRADES",
                useRTH=False,
                formatDate=2,
                keepUpToDate=False,
                timeout=15,
            ),
        )
        if not bars:
            raise ValueError("IBKR_HISTORY_UNAVAILABLE")
        return bars

    async def monitor(self, state: MarketState) -> None:
        cached_bars = list(state.bars)
        cached_contract = state.detail.contract.conId if state.detail else None
        await self.unsubscribe(state)
        state.problem = "WARMING_UP"
        state.bars = []
        state.references = []
        state.live_since = None
        state.last_update = None
        at = now()
        today = at.astimezone(NY).date()
        listed = await self.broker.ib.reqContractDetailsAsync(
            Future(
                symbol=state.market,
                exchange=EXCHANGES[state.market],
                currency="USD",
                includeExpired=True,
            )
        )
        details = nearby_futures(listed, today)
        if not details:
            raise ValueError("NO_LISTED_SIGNAL_FUTURE")
        async with asyncio.timeout(15):
            schedule = await self.broker.data.historical(
                (details[0].contract.conId, "schedule", today.isoformat()),
                lambda: self.broker.ib.reqHistoricalScheduleAsync(
                    details[0].contract, 14, useRTH=False
                ),
            )
        zone = ZoneInfo(schedule.timeZone)
        previous = [
            datetime.strptime(s.refDate, "%Y%m%d").date()
            for s in schedule.sessions
            if datetime.strptime(s.endDateTime, "%Y%m%d-%H:%M:%S")
            .replace(tzinfo=zone)
            .astimezone(UTC)
            < at
        ]
        if not previous:
            raise ValueError("PREVIOUS_COMPLETED_EXCHANGE_SESSION_UNKNOWN")
        previous_day = max(previous)
        reference_days = sorted(d for d in set(previous) if d < today and d.weekday() < 5)[-5:]
        candidates = {day: nearby_futures(listed, day) for day in [*reference_days, today]}
        history_contracts = {d.contract.conId: d for group in candidates.values() for d in group}
        daily: dict[int, dict[date, float]] = {}
        for detail in history_contracts.values():
            rows = await self.history(detail.contract, "14 D", "1 day")
            daily[detail.contract.conId] = {
                b.date: float(b.volume) for b in rows if isinstance(b.date, date)
            }
        selected = select_future(
            [
                (d, previous_day, daily[d.contract.conId].get(previous_day, float("nan")))
                for d in details
            ],
            previous_day,
            today,
        )
        state.detail = selected
        state.calendar = Calendar.from_details(selected, at)
        if selected.contract.conId != details[0].contract.conId:
            async with asyncio.timeout(15):
                schedule = await self.broker.data.historical(
                    (selected.contract.conId, "schedule", today.isoformat()),
                    lambda: self.broker.ib.reqHistoricalScheduleAsync(
                        selected.contract, 14, useRTH=False
                    ),
                )
            zone = ZoneInfo(schedule.timeZone)
        # HistoricalSchedule.refDate is the broker's explicit exchange trade date.
        dated_sessions = []
        for start, end, _ in state.calendar.sessions:
            ref = next(
                (
                    s.refDate
                    for s in schedule.sessions
                    if datetime.strptime(s.startDateTime, "%Y%m%d-%H:%M:%S")
                    .replace(tzinfo=zone)
                    .astimezone(UTC)
                    == start
                    and datetime.strptime(s.endDateTime, "%Y%m%d-%H:%M:%S")
                    .replace(tzinfo=zone)
                    .astimezone(UTC)
                    == end
                ),
                "",
            )
            dated_sessions.append((start, end, ref))
        state.calendar = Calendar(
            tuple(dated_sessions),
            state.calendar.covered_dates,
            state.calendar.timezone,
            state.calendar.observed_at,
        )
        # Five source reference dates, using their own preceding daily volume and contract identity.
        cached: dict[int, list[Bar]] = {}
        reference_key = f"futures_reference:{RULE_VERSION}:{state.market}:{today.isoformat()}"
        saved = self.store.get_meta(reference_key)
        if saved:
            state.references = [{int(h): v for h, v in ref.items()} for ref in saved]
        for day in [] if saved else reference_days:
            prior = max((d for d in set(previous) if d < day), default=None)
            if prior is None:
                continue
            historical = select_future(
                [
                    (d, prior, daily[d.contract.conId].get(prior, float("nan")))
                    for d in candidates[day]
                ],
                prior,
                day,
            )
            cid = historical.contract.conId
            if cid not in cached:
                raw = await self.history(historical.contract, "10 D")
                cached[cid] = self.convert(raw, at)
            bars = [
                b
                for b in cached[cid]
                if b.at.astimezone(NY).date() == day and 8 <= b.at.astimezone(NY).hour < 17
            ]
            if bars:
                state.references.append(reference_summary(bars))
        if len(state.references) == 5:
            self.store.set_meta(reference_key, state.references)
        reusable = (
            cached_contract == selected.contract.conId
            and bool(cached_bars)
            and (at - cached_bars[-1].at).total_seconds() < 1800
        )
        owner = f"core:{state.market}"
        bars_feed = await self.broker.data.acquire(
            selected.contract,
            "BARS",
            owner,
            "core_bars",
            {"duration": "1800 S" if reusable else "2 D"},
        )
        state.stream = bars_feed.value
        if reusable and state.stream:
            first = utc(state.stream[0].date)
            state.stream[:0] = [
                BarData(
                    date=b.at,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                    average=b.average,
                )
                for b in cached_bars
                if b.at < first
            ]
        feed = await self.broker.data.acquire(selected.contract, "QUOTE", owner, "core_quote")
        state.ticker = feed.value
        state.live_since = now()
        state.selected_day = today
        state.last_clock = now().replace(second=0, microsecond=0)
        self.update_bars(state)

    async def repair_gap(self, state: MarketState) -> None:
        # At most one bounded IB-originated repair per market / five minutes. No fabricated bars.
        at = now()
        if (
            not state.calendar
            or state.calendar.state(at)[0] != "OPEN"
            or (state.gap_repair_at and at - state.gap_repair_at < timedelta(minutes=5))
        ):
            return
        recent = self.convert(state.stream[-32:-1], at)
        if len(recent) < 2 or all(
            b.at - a.at == timedelta(minutes=1) for a, b in zip(recent, recent[1:], strict=False)
        ):
            return
        state.gap_repair_at = at
        try:
            raw = await self.history(state.detail.contract, "1800 S")
            # Only bars preceding the live mutable tail qualify. Retain broker timestamps/volume.
            tail = utc(state.stream[-1].date)
            merged = {utc(b.date): b for b in state.stream}
            merged.update({utc(b.date): b for b in raw if utc(b.date) < tail})
            state.stream[:] = [merged[t] for t in sorted(merged)][-1442:]
            state.last_update = None
            # A repaired older clock must never be replayed as a fresh order.
            state.last_clock = max(state.last_clock or at, at.replace(second=0, microsecond=0))
        except Exception as exc:
            log.warning("Bounded bar repair %s failed: %s", state.market, exc)

    @staticmethod
    def convert(raw: Any, at: datetime) -> list[Bar]:
        result = []
        for b in raw:
            if not isinstance(b.date, datetime):
                continue
            start = utc(b.date)
            if start + timedelta(minutes=1) <= at:
                bar = Bar(
                    start,
                    float(b.open),
                    float(b.high),
                    float(b.low),
                    float(b.close),
                    float(b.volume),
                    float(b.average),
                )
                if bar.valid():
                    result.append(bar)
        return result

    def update_bars(self, state: MarketState) -> None:
        if state.stream is None:
            return
        if (
            len(state.stream) >= 2
            and isinstance(state.stream[-2].date, datetime)
            and utc(state.stream[-2].date) + timedelta(minutes=1) == state.last_update
        ):
            return
        # The current stream tail is mutable. Only a subsequent IB bar proves rollover;
        # the wall clock alone must not promote the still-updating tail to a decision input.
        completed = self.convert(state.stream[:-1], now())
        if not completed:
            state.problem = "COMPLETED_BARS_UNAVAILABLE"
            return
        latest = completed[-1].at + timedelta(minutes=1)
        if state.last_update == latest:
            return
        new = [
            b
            for b in completed
            if not state.last_update or b.at + timedelta(minutes=1) > state.last_update
        ]
        with self.store.db:
            for bar in new:
                data = asdict(bar)
                data["at"] = bar.at.isoformat()
                self.store.db.execute(
                    "INSERT OR IGNORE INTO bars VALUES(?,?,?,?)",
                    (state.detail.contract.conId, bar.at.isoformat(), state.market, encode(data)),
                )
        state.bars = completed[-1440:]
        state.last_update = latest
        # Bound the in-memory streaming series while retaining completed OHLCV in the ledger.
        if len(state.stream) > 1442:
            del state.stream[:-1442]
        try:
            state.features = {
                "rv15": prior_rv(state.bars, latest),
                "completed_closes": min(len(state.bars), 31),
            }
            state.problem = (
                "" if len(state.references) >= 5 else "WARMING_UP_FIVE_REFERENCE_SESSIONS"
            )
        except ValueError as exc:
            state.problem = str(exc)

    async def decisions(self) -> None:
        # Admission order is opportunity UTC then fixed market order; no performance ranking.
        pending = []
        at = now()
        for state in self.markets.values():
            self.update_bars(state)
            if state.live_since is None or state.detail is None:
                continue
            for clock in clocks(at.astimezone(NY).date()):
                if state.live_since <= clock <= at and (
                    state.last_clock is None or clock > state.last_clock
                ):
                    pending.append((clock, MARKETS.index(state.market), state))
        for clock, _, state in sorted(pending, key=lambda item: (item[0], item[1])):
            if (state.last_update is None or state.last_update < clock) and (
                now() - clock
            ).total_seconds() < self.config.entry_deadline_seconds:
                continue
            event = opportunity(state.market, state.detail.contract.conId, clock)
            reason = str(event["veto"])
            inputs: dict[str, float] = {}
            try:
                if (
                    not reason
                    and (now() - clock).total_seconds() >= self.config.entry_deadline_seconds
                ):
                    reason = "STALE_SIGNAL_NO_REPLAY"
                if not reason:
                    if not state.calendar or state.calendar.state(now())[0] != "OPEN":
                        raise ValueError("SIGNAL_MARKET_NOT_OPEN")
                    if not state.ticker or state.ticker.marketDataType != 1:
                        raise ValueError("SIGNAL_DATA_NOT_REALTIME")
                    inputs = eligibility(
                        state.bars,
                        clock,
                        datetime.strptime(
                            state.detail.contract.lastTradeDateOrContractMonth[:8], "%Y%m%d"
                        ).date(),
                        state.references,
                    )
            except ValueError as exc:
                reason = str(exc)
            state.last_clock = clock
            observed = self.store.observe(event, reason, inputs)
            if observed:
                try:
                    self.depth.signal(event)
                except Exception as exc:
                    self.depth.paused = f"CAPTURE_METADATA_FAILED:{type(exc).__name__}"
                    log.warning("Optional capture failed: %s", exc)
            if not observed or reason:
                continue
            try:
                deadline = asyncio.get_running_loop().time() + max(
                    0, self.config.entry_deadline_seconds - (now() - clock).total_seconds()
                )
                with priority(CORE, deadline):
                    async with asyncio.timeout_at(deadline):
                        plan = await self.broker.prepare(event, state.detail.contract, inputs)
                reason = await self.broker.enter(event, plan)
                if reason:
                    self.store.decision(str(event["id"]), "SKIPPED", reason)
            except Exception as exc:
                reserved = self.store.db.execute(
                    "SELECT 1 FROM reservations WHERE id=?", (event["id"],)
                ).fetchone()
                self.store.decision(
                    str(event["id"]),
                    "ORDER_STATUS_UNCERTAIN" if reserved else "SKIPPED",
                    str(exc)
                    or (
                        "ENTRY_PREPARATION_DEADLINE"
                        if isinstance(exc, TimeoutError)
                        else type(exc).__name__
                    ),
                )
