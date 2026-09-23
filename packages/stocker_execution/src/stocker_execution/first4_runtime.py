"""One US scanner -> frozen allocation -> PAPER options pipeline."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from ib_async import ScannerSubscription, TagValue

from stocker_data.calendars import get_market_calendar
from stocker_execution.first4 import METHOD, Bar, prior15
from stocker_execution.first4_broker import PaperBroker, now
from stocker_execution.first4_config import First4Config
from stocker_execution.first4_store import Store

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, config: First4Config, store: Store):
        self.config, self.store = config, store
        self.broker = PaperBroker(config, store)
        self.running = True
        self.session: str | None = None
        self.schedule: list[tuple[str, datetime, datetime]] = []
        self.history_limit = asyncio.Semaphore(4)
        self.tasks: set[asyncio.Task[None]] = set()
        self.pause = bool(store.get_meta("paused", False))
        self.problem = "STARTING"

    def status(self) -> dict[str, Any]:
        return {
            "method": METHOD,
            "market": "US",
            "environment": "PAPER",
            "account": self.config.expected_account,
            "connected": self.broker.ib.isConnected(),
            "reconciled": self.broker.reconciled,
            "armed": self.config.armed
            and not self.pause
            and not self.config.missing()
            and self.broker.reconciled
            and not self.problem
            and not self.broker.entry_blocker,
            "configured_armed": self.config.armed,
            "missing_settings": self.config.missing(),
            "problem": self.problem or self.broker.problem or self.broker.entry_blocker,
            "session": self.session,
            "live_enabled": False,
            "settings": self.config.model_dump(),
        }

    async def history(self, contract: Any, end: datetime, duration: str) -> list[Any]:
        async with self.history_limit:
            result = await self.broker.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end,
                durationStr=duration,
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                keepUpToDate=False,
                timeout=15,
            )
            return list(result)

    async def candidate(
        self, row: Any, opened: datetime, clock: datetime, previous_close: datetime
    ) -> dict[str, Any]:
        contract = row.contractDetails.contract
        result = {
            "symbol": contract.symbol,
            "con_id": contract.conId,
            "rank": row.rank + 1,
            "price": float("nan"),
            "change_pct": float("nan"),
            "prior15": None,
        }
        try:
            bars, previous = await asyncio.gather(
                self.history(contract, clock, "960 S"),
                self.history(contract, previous_close, "60 S"),
            )
            values = [
                Bar(b.date, b.open, b.high, b.low, b.close)
                for b in bars
                if opened <= b.date < clock
            ]
            completed = next((b for b in values if b.time == clock - timedelta(minutes=1)), None)
            prev = next(
                (b for b in previous if b.date == previous_close - timedelta(minutes=1)), None
            )
            if completed:
                result["price"] = completed.close
            if completed and prev and prev.close > 0:
                result["change_pct"] = 100 * (completed.close / prev.close - 1)
            result["prior15"] = prior15(values, opened, clock)
            result["detail"] = {
                "source": "IBKR_RTH_TRADES",
                "previous_close_at": previous_close.isoformat(),
            }
        except Exception as exc:
            result["detail"] = {"error": str(exc)}
        return result

    async def execute(self, event: dict[str, Any], underlying: Any) -> None:
        if not self.config.armed or self.config.missing() or self.pause:
            self.store.outcome(event, "UNARMED", {"missing_settings": self.config.missing()})
            return
        ticker = None
        callback = None
        try:
            self.broker.guard()
            baseline = datetime.fromisoformat(event["entry_at"])
            if now() >= baseline:
                raise ValueError("BASELINE_ANCHOR_MISSED")
            # Subscribe before the boundary. The first eligible trade supplies open(j+2).
            anchor = asyncio.get_running_loop().create_future()
            ticker = self.broker.ib.reqTickByTickData(underlying, "Last", 0, False)

            def on_tick(t: Any) -> None:
                for tick in t.tickByTicks:
                    if tick.time >= baseline and tick.price > 0 and not anchor.done():
                        anchor.set_result((tick.price, tick.time))

            callback = on_tick
            ticker.updateEvent += callback
            await self.broker.chain(underlying)
            deadline = baseline + timedelta(seconds=self.config.number("entry_deadline_seconds"))
            price, stamp = await asyncio.wait_for(
                anchor, max(0, (deadline - now()).total_seconds())
            )
            self.store.outcome(
                event, "ANCHOR_OBSERVED", {"price": price, "broker_trade_time": stamp.isoformat()}
            )
            if self.pause:
                raise ValueError("ENTRIES_PAUSED")
            async with asyncio.timeout(max(0, (deadline - now()).total_seconds())):
                await self.broker.enter(event, underlying, price)
        except Exception as exc:
            self.store.outcome(event, "EXECUTION_FAILED", {"error": str(exc) or type(exc).__name__})
            log.warning("FIRST4 execution failed %s: %s", event["symbol"], exc)
        finally:
            if ticker is not None:
                ticker.updateEvent -= callback
                self.broker.ib.cancelTickByTickData(underlying, "Last")

    async def scan(
        self,
        session: str,
        opened: datetime,
        closed: datetime,
        previous_close: datetime,
        clock: datetime,
    ) -> None:
        sub = ScannerSubscription(
            instrument="STK", locationCode="STK.US", scanCode="MOST_ACTIVE", numberOfRows=25
        )
        rows = await self.broker.ib.reqScannerDataAsync(
            sub, [], [TagValue("changePercAbove", "5.5"), TagValue("priceBelow", "20")]
        )
        if len(rows) > 25:
            raise ValueError("SCANNER_ROW_LIMIT_VIOLATION")
        seen = {
            r[0]
            for r in self.store.db.execute(
                "SELECT symbol FROM first4_events WHERE session=?", (session,)
            )
        }
        fresh = [r for r in rows if r.contractDetails.contract.symbol not in seen]
        candidates = await asyncio.gather(
            *(self.candidate(r, opened, clock, previous_close) for r in fresh)
        )
        # Inputs complete as one batch; asynchronous response order cannot allocate slots.
        selected = self.store.observe(session, clock, closed, candidates)
        contracts = {r.contractDetails.contract.conId: r.contractDetails.contract for r in fresh}
        for event in selected:
            task = asyncio.create_task(self.execute(event, contracts[event["con_id"]]))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def run(self) -> None:
        manager = asyncio.create_task(self.maintain_broker())
        try:
            await self.scan_sessions()
        finally:
            manager.cancel()
            await asyncio.gather(manager, return_exceptions=True)

    async def maintain_broker(self) -> None:
        """Exit obligations never wait for scanner history or option qualification."""
        while self.running:
            try:
                if not self.broker.ib.isConnected():
                    async with asyncio.timeout(30):
                        await self.broker.connect()
                    if self.session:
                        self.store.block(self.session, "SCANNER_CONTINUITY_LOST_AFTER_DISCONNECT")
                        self.problem = "SCANNER_CONTINUITY_LOST_AFTER_DISCONNECT"
                elif not self.broker.reconciled:
                    async with asyncio.timeout(15):
                        await self.broker.reconcile()
                await self.broker.close_due()
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.broker.problem = str(exc) or type(exc).__name__
                self.store.set_meta(
                    "broker_manager_error",
                    {
                        "time": now().isoformat(),
                        "error": self.broker.problem,
                    },
                )
                await asyncio.sleep(5)

    async def scan_sessions(self) -> None:
        calendar = get_market_calendar("NYSE")
        next_clock: datetime = now()
        while self.running:
            try:
                if not self.broker.reconciled:
                    await asyncio.sleep(0.1)
                    continue
                today = now().astimezone(ZoneInfo("America/New_York")).date()
                if not self.schedule or self.schedule[-1][0] < today.isoformat():
                    # Calendar construction is off the broker event loop, once per day.
                    frame = await asyncio.to_thread(
                        calendar.schedule,
                        start_date=today - timedelta(days=10),
                        end_date=today + timedelta(days=1),
                    )
                    self.schedule = [
                        (
                            str(d.date()),
                            r.market_open.to_pydatetime(),
                            r.market_close.to_pydatetime(),
                        )
                        for d, r in frame.iterrows()
                    ]
                current = next((r for r in self.schedule if r[0] == today.isoformat()), None)
                if current:
                    day, opened, closed = current
                    if self.session != day:
                        self.session = day
                        self.problem = ""
                        self.broker.chains.clear()
                        if now() >= opened + timedelta(minutes=1):
                            self.store.block(day, "SESSION_START_OR_SCANNER_HISTORY_MISSED")
                            self.problem = "SESSION_START_OR_SCANNER_HISTORY_MISSED"
                        next_clock = opened + timedelta(minutes=1)
                    if opened < now() < closed and not self.problem and now() >= next_clock:
                        if now() >= next_clock + timedelta(seconds=2):
                            raise ValueError("SCANNER_MINUTE_MISSED")
                        prior_close = [r[2] for r in self.schedule if r[0] < day][-1]
                        async with asyncio.timeout(45):
                            await self.scan(day, opened, closed, prior_close, next_clock)
                        next_clock += timedelta(minutes=1)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.problem = str(exc) or type(exc).__name__
                self.store.set_meta(
                    "runtime_error", {"time": now().isoformat(), "error": self.problem}
                )
                if self.session:
                    self.store.block(self.session, self.problem)
                log.exception("FIRST4 runtime blocked; exit obligations retained")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self.running = False
        self.pause = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.broker.ib.disconnect()
