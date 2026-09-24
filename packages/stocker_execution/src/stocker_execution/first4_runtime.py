"""One US scanner -> frozen allocation -> PAPER options pipeline."""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from ib_async import ScannerSubscription, TagValue, util

from stocker_data.calendars import get_market_calendar
from stocker_execution.first4 import METHOD, Bar, prior15
from stocker_execution.first4_broker import PaperBroker, now
from stocker_execution.first4_config import First4Config
from stocker_execution.first4_readiness import option_access
from stocker_execution.first4_store import Store

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, config: First4Config, store: Store):
        self.config, self.store = config, store
        self.broker = PaperBroker(config, store)
        self.running = True
        self.session: str | None = None
        self.schedule: list[tuple[str, datetime, datetime]] = []
        self.calendar_covered_through: date | None = None
        self.history_limit = asyncio.Semaphore(4)
        self.tasks: set[asyncio.Task[None]] = set()
        self.pause = bool(store.get_meta("paused", False))
        self.problem = "STARTING"

    def status(self) -> dict[str, Any]:
        state = self.store.db.execute(
            "SELECT blocked FROM first4_sessions WHERE session=?", (self.session,)
        ).fetchone()
        scanner_problem = state["blocked"] if state else ""
        return {
            "method": METHOD,
            "market": "US",
            "environment": "PAPER",
            "account": self.config.expected_account,
            "connected": self.broker.ib.isConnected(),
            "upstream_lost": self.broker.upstream_lost,
            "market_data_block": self.broker.market_data_block,
            "entry_blocker": self.broker.entry_blocker,
            "trading_ready": self.broker.ib.isConnected()
            and not self.broker.upstream_lost
            and self.broker.reconciled
            and not self.problem
            and not scanner_problem
            and not self.broker.entry_blocker
            and not self.broker.market_data_block,
            "reconciled": self.broker.reconciled,
            "armed": self.broker.entries_armed()
            and not self.pause
            and not self.config.missing()
            and self.broker.reconciled
            and not self.problem
            and not self.broker.entry_blocker,
            "configured_armed": self.config.armed,
            "opening_check": self.store.get_meta(
                "opening_check:" + str(self.config.arm_after_quote_check_on), {}
            ),
            "missing_settings": self.config.missing(),
            "problem": self.problem
            or self.broker.problem
            or scanner_problem
            or self.broker.entry_blocker,
            "session": self.session,
            "live_enabled": False,
            "settings": self.config.model_dump(mode="json"),
        }

    async def history(self, contract: Any, end: datetime, duration: str) -> list[Any]:
        async with self.history_limit:
            ib = self.broker.ib
            request_id = ib.client.getReqId()
            bars: list[Any] = []
            future = ib.wrapper.startReq(request_id, contract, container=bars)
            failure = None
            completed = False

            def failed(req_id: int, code: int, message: str, *args: Any) -> None:
                nonlocal failure
                if req_id == request_id:
                    failure = ValueError(f"HISTORY_REQUEST_FAILED:{code}:{message}")
                    if not future.done():
                        future.set_result(bars)

            ib.errorEvent += failed
            try:
                ib.client.reqHistoricalData(
                    request_id,
                    contract,
                    util.formatIBDatetime(end),
                    duration,
                    "1 min",
                    "TRADES",
                    True,
                    2,
                    False,
                    [],
                )
                await asyncio.wait_for(future, 15)
                if failure:
                    raise failure
                completed = True
                return list(bars)
            finally:
                ib.errorEvent -= failed
                try:
                    if not completed:
                        ib.client.cancelHistoricalData(request_id)
                finally:
                    future.cancel()
                    ib.wrapper._futures.pop(request_id, None)
                    ib.wrapper._results.pop(request_id, None)
                    ib.wrapper._reqId2Contract.pop(request_id, None)

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
            async with asyncio.TaskGroup() as group:
                recent = group.create_task(self.history(contract, clock, "960 S"))
                prior = group.create_task(self.history(contract, previous_close, "60 S"))
            bars, previous = recent.result(), prior.result()
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
        if not self.broker.entries_armed() or self.config.missing() or self.pause:
            self.store.outcome(event, "UNARMED", {"missing_settings": self.config.missing()})
            return
        data_generation = self.broker.market_data_generation
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
                    if (
                        data_generation == self.broker.market_data_generation
                        and not self.broker.market_data_block
                        and baseline <= tick.time < baseline + timedelta(minutes=1)
                        and tick.price > 0
                        and not anchor.done()
                    ):
                        anchor.set_result((tick.price, tick.time))

            callback = on_tick
            ticker.updateEvent += callback
            deadline = baseline + timedelta(seconds=self.config.number("entry_deadline_seconds"))
            await asyncio.wait_for(
                self.broker.chain(underlying), max(0, (deadline - now()).total_seconds())
            )
            self.broker.check_data_generation(data_generation)
            price, stamp = await asyncio.wait_for(
                anchor,
                max(0, (min(deadline, baseline + timedelta(minutes=1)) - now()).total_seconds()),
            )
            self.broker.check_data_generation(data_generation)
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
        try:
            await self._scan(session, opened, closed, previous_close, clock)
        except BaseException:
            self.problem = "SCANNER_OBSERVATION_UNKNOWN_CONTINUITY_LOST"
            self.store.block(session, self.problem)
            raise

    async def scanner_rows(self, sub: Any) -> list[Any]:
        # ib_async 2.1's convenience helper treats API errors as partial success
        # and lacks finally cleanup. Only scannerDataEnd emits this update event.
        done: asyncio.Future[list[Any]] = asyncio.get_running_loop().create_future()
        rows = self.broker.ib.reqScannerSubscription(
            sub, [], [TagValue("changePercAbove", "5.5"), TagValue("priceBelow", "20")]
        )

        def complete(data: Any) -> None:
            if not done.done():
                done.set_result(list(data))

        def failed(request_id: int, code: int, message: str, *args: Any) -> None:
            if (
                request_id == rows.reqId or code in {1100, 1101, 1102, 2110, 10197}
            ) and not done.done():
                done.set_exception(ValueError(f"SCANNER_REQUEST_FAILED:{code}:{message}"))

        rows.updateEvent += complete
        self.broker.ib.errorEvent += failed
        try:
            return await done
        finally:
            rows.updateEvent -= complete
            self.broker.ib.errorEvent -= failed
            try:
                self.broker.ib.client.cancelScannerSubscription(rows.reqId)
            finally:
                self.broker.ib.wrapper.endSubscription(rows)

    async def _scan(
        self,
        session: str,
        opened: datetime,
        closed: datetime,
        previous_close: datetime,
        clock: datetime,
    ) -> None:
        generation = self.broker.connection_generation
        data_generation = self.broker.market_data_generation
        sub = ScannerSubscription(
            instrument="STK", locationCode="STK.US", scanCode="MOST_ACTIVE", numberOfRows=25
        )
        rows = await self.scanner_rows(sub)
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
        self.broker.check_data_generation(data_generation)
        if self.broker.market_data_block:
            raise ValueError("MARKET_DATA_COMPETING_SESSION_10197")
        if generation != self.broker.connection_generation or self.broker.upstream_lost:
            raise ValueError("SCANNER_CONNECTION_CHANGED")
        # Inputs complete as one batch; asynchronous response order cannot allocate slots.
        selected = self.store.observe(session, clock, closed, candidates)
        contracts = {r.contractDetails.contract.conId: r.contractDetails.contract for r in fresh}
        for event in selected:
            task = asyncio.create_task(self.execute(event, contracts[event["con_id"]]))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def run(self) -> None:
        manager = asyncio.create_task(self.maintain_broker())
        opening = asyncio.create_task(self.arm_at_open())
        try:
            await self.scan_sessions()
        finally:
            manager.cancel()
            opening.cancel()
            await asyncio.gather(manager, opening, return_exceptions=True)

    async def arm_at_open(self) -> None:
        target = self.config.arm_after_quote_check_on
        if target is None:
            return
        key = "opening_check:" + target.isoformat()
        # Durable audit is not an authorization token after a process restart.
        if self.store.get_meta(key, {}).get("status") in {"CHECKING", "ARMED", "FAILED"}:
            return
        self.store.set_meta(key, {"status": "WAITING_FOR_OPEN", "session": target.isoformat()})
        while self.running:
            current = now()
            if current.astimezone(ZoneInfo("America/New_York")).date() > target:
                self.store.set_meta(key, {"status": "EXPIRED", "session": target.isoformat()})
                return
            session = next((r for r in self.schedule if r[0] == target.isoformat()), None)
            if session and current >= session[1]:
                await self.check_opening(session[0], session[1])
                return
            await asyncio.sleep(0.5)

    def opening_guard(self, day: str, opened: datetime) -> None:
        self.broker.guard()
        self.config.require_settings()
        if self.config.armed or str(self.config.arm_after_quote_check_on) != day:
            raise ValueError("OPENING_ARM_DATE_NOT_AUTHORIZED")
        # Finish before the first possible Q5 admission at open + 15 minutes.
        if not opened <= now() < opened + timedelta(minutes=14):
            raise ValueError("OPENING_VERIFICATION_WINDOW_EXPIRED")
        if self.pause or self.store.get_meta("paused", False):
            raise ValueError("ENTRIES_PAUSED")
        if self.session != day or self.problem or self.broker.entry_blocker:
            raise ValueError(self.problem or self.broker.entry_blocker or "SESSION_NOT_READY")
        state = self.store.db.execute(
            "SELECT blocked FROM first4_sessions WHERE session=?", (day,)
        ).fetchone()
        if state and state["blocked"]:
            raise ValueError(state["blocked"])
        if any(p.position for p in self.broker.ib.positions(self.config.expected_account)):
            raise ValueError("OPENING_CHECK_REQUIRES_RECONCILED_FLAT_ACCOUNT")

    async def check_opening(self, day: str, opened: datetime) -> None:
        generation = self.broker.connection_generation
        key = "opening_check:" + day
        report: dict[str, Any] = {
            "session": day,
            "status": "CHECKING",
            "started_at": now().isoformat(),
            "deadline": (opened + timedelta(minutes=14)).isoformat(),
            "purpose": "READ_ONLY_DATA_CHECK_NOT_FIRST4_SIGNAL",
            "transmitted_orders": 0,
        }
        self.store.set_meta(key, report)
        try:
            self.opening_guard(day, opened)
            async with asyncio.timeout(15):
                await self.broker.reconcile(require_flat=True)
            self.opening_guard(day, opened)
            deadline = opened + timedelta(minutes=14)
            async with asyncio.timeout(max(0, (deadline - now()).total_seconds())):
                report.update(await option_access(self.broker, deadline))
            self.opening_guard(day, opened)
            if report.get("blockers") or not all(
                report.get("checks", {}).get(name) is True
                for name in (
                    "qualified_usd_standard_multiplier",
                    "fresh_realtime_option_quotes",
                    "combo_price_increment",
                )
            ):
                raise ValueError("OPENING_CHECK_INCOMPLETE")
            if generation != self.broker.connection_generation:
                raise ValueError("CONNECTION_CHANGED_DURING_OPENING_CHECK")
            report["status"] = "ARMED"
            report["armed_at"] = now().isoformat()
            self.store.set_meta(key, report)
            self.broker.opening_verified_session = day
            log.info("FIRST4 PAPER opening verification passed for %s; entries enabled", day)
        except asyncio.CancelledError:
            self.broker.opening_verified_session = None
            report.update(status="FAILED", error="OPENING_CHECK_INTERRUPTED")
            self.store.set_meta(key, report)
            raise
        except Exception as exc:
            self.broker.opening_verified_session = None
            report.update(status="FAILED", error=str(exc) or type(exc).__name__)
            self.store.set_meta(key, report)
            log.warning("FIRST4 opening verification failed: %s", report["error"])

    async def maintain_broker(self) -> None:
        """Exit obligations never wait for scanner history or option qualification."""
        while self.running:
            try:
                if not self.broker.ib.isConnected():
                    async with asyncio.timeout(30):
                        await self.broker.connect()
                    if self.broker.block_scanner_gap():
                        self.problem = "SCANNER_CONTINUITY_LOST_AFTER_DISCONNECT"
                elif self.broker.upstream_lost:
                    await asyncio.sleep(0.1)
                    continue
                elif not self.broker.reconciled and not self.broker.reconciliation_lock.locked():
                    async with asyncio.timeout(15):
                        await self.broker.reconcile()
                try:
                    async with asyncio.timeout(10):
                        await self.broker.cancel_due_entries()
                except Exception as exc:
                    self.broker.problem = str(exc) or type(exc).__name__
                    self.store.set_meta(
                        "entry_cancellation_error",
                        {"time": now().isoformat(), "error": self.broker.problem},
                    )
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

    async def refresh_calendar(self, calendar: Any, today: date) -> None:
        if self.calendar_covered_through is not None and today <= self.calendar_covered_through:
            return
        frame = await asyncio.to_thread(
            calendar.schedule,
            start_date=today - timedelta(days=10),
            end_date=today + timedelta(days=1),
        )
        self.schedule = [
            (str(d.date()), r.market_open.to_pydatetime(), r.market_close.to_pydatetime())
            for d, r in frame.iterrows()
        ]
        self.calendar_covered_through = today + timedelta(days=1)

    async def scan_sessions(self) -> None:
        calendar = get_market_calendar("NYSE")
        next_clock: datetime = now()
        while self.running:
            try:
                if not self.broker.reconciled:
                    await asyncio.sleep(0.1)
                    continue
                today = now().astimezone(ZoneInfo("America/New_York")).date()
                await self.refresh_calendar(calendar, today)
                current = next((r for r in self.schedule if r[0] == today.isoformat()), None)
                if current:
                    day, opened, closed = current
                    if self.session != day:
                        self.session = day
                        self.broker.active_session = day
                        self.broker.active_session_open = opened
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
