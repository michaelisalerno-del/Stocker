"""One US scanner -> frozen allocation -> PAPER options pipeline."""

import asyncio
import logging
import math
import re
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from ib_async import RequestError, ScannerSubscription, TagValue

from stocker_data.calendars import get_market_calendar
from stocker_execution.first4 import METHOD, Bar, prior15
from stocker_execution.first4_broker import OptionChainError, PaperBroker, now
from stocker_execution.first4_config import First4Config
from stocker_execution.first4_flow_observer import FlowObserver
from stocker_execution.first4_readiness import option_access
from stocker_execution.first4_store import Store

log = logging.getLogger(__name__)
Health = Literal["STARTING", "RUNNING", "DEGRADED", "FAILED", "STOPPED"]
OPENING_ATTEMPT_SECONDS = 60
OPENING_RETRY_SECONDS = 5


async def opening_pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


def exception_info(exc: BaseException) -> dict[str, Any]:
    message = re.sub(r"\b(?:D[UF]|[UF])\d+\b", "[account]", str(exc))
    return {"type": type(exc).__name__, "message": " ".join(message.split())[:300]}


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
        self.worker_health: Health = "STARTING"
        self.manager_health: Health = "STARTING"
        self.web_health: Health = "STARTING"
        self.calendar_date: date | None = None
        self.next_clock: datetime | None = None
        self.last_observation: str | None = None
        self.scan_metrics: dict[str, Any] = {}
        self.critical_tasks: set[asyncio.Task[None]] = set()
        self.stopping = False
        self.opening_active = False
        self.order_flow = FlowObserver(self)

    def status(self) -> dict[str, Any]:
        available = True
        opening: dict[str, Any] = {}
        last_clock = self.last_observation
        obligations = None
        try:
            opening = self.store.get_meta(
                "opening_check:" + str(self.config.arm_after_quote_check_on), {}
            )
            reason = self.broker.entry_reason(self.session)
            state = self.store.db.execute(
                "SELECT last_clock FROM first4_sessions WHERE session=?", (self.session,)
            ).fetchone()
            if state:
                last_clock = state[0]
            obligations = self.store.db.execute(
                "SELECT count(*) FROM first4_orders WHERE role='ENTRY' AND obligation_done=0"
            ).fetchone()[0]
        except sqlite3.Error as exc:
            available = False
            reason = "LEDGER_UNAVAILABLE: " + str(exc)
            self.broker.fatal_error = reason
            self.broker.persistence_failed = True
        reason = (
            self.broker.fatal_error
            or self.broker.management_block
            or self.problem
            or ("ENTRIES_PAUSED" if self.pause else "")
            or reason
        )
        return {
            "method": METHOD,
            "market": "US",
            "environment": "PAPER",
            "account": self.config.expected_account,
            "connected": self.broker.ib.isConnected(),
            "reconciled": self.broker.reconciled,
            "armed": not reason
            and not self.config.missing()
            and self.worker_health == "RUNNING"
            and self.manager_health == "RUNNING",
            "configured_armed": self.config.armed,
            "opening_check": opening,
            "opening_check_active": self.opening_active,
            "opening_remaining_seconds": max(
                0, (datetime.fromisoformat(opening["deadline"]) - now()).total_seconds()
            )
            if self.opening_active and opening.get("deadline")
            else None,
            "worker_health": self.worker_health,
            "manager_health": self.manager_health,
            "web_health": self.web_health,
            "ledger_available": available and not self.broker.persistence_failed,
            "upstream_available": self.broker.upstream_available,
            "upstream_status": self.broker.upstream_status,
            "last_option_quote_check": (
                self.broker.last_quote_at.isoformat() if self.broker.last_quote_at else None
            ),
            "option_quote_state": (
                "UNOBSERVED"
                if self.broker.last_quote_at is None
                else "FRESH"
                if 0 <= (now() - self.broker.last_quote_at).total_seconds() <= 5
                else "STALE"
            ),
            "data_problem": self.broker.data_problem,
            "market_data_block": self.broker.market_data_block,
            "last_scanner_observation": last_clock,
            "entry_block_reason": reason,
            "outstanding_obligations": obligations,
            "operator_exceptions": self.broker.operator_exceptions,
            "scan_metrics": self.scan_metrics,
            "missing_settings": self.config.missing(),
            "problem": reason or self.broker.problem,
            "session": self.session,
            "live_enabled": False,
            "settings": self.config.model_dump(mode="json"),
        }

    async def history(self, contract: Any, end: datetime, duration: str) -> list[Any]:
        queued = asyncio.get_running_loop().time()
        async with self.history_limit:
            started = asyncio.get_running_loop().time()
            self.scan_metrics["queue_seconds"] = max(
                self.scan_metrics.get("queue_seconds", 0), started - queued
            )
            try:
                return list(await self.broker.ib.history(contract, end, duration))
            finally:
                self.scan_metrics["request_seconds"] = max(
                    self.scan_metrics.get("request_seconds", 0),
                    asyncio.get_running_loop().time() - started,
                )

    async def candidate(
        self,
        row: Any,
        opened: datetime,
        clock: datetime,
        previous_close: datetime,
        deadline: float | None = None,
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
            # The two end/duration identities differ; neither is redundant.
            # TaskGroup drains the sibling request if either one fails.
            async with asyncio.timeout_at(deadline):
                async with asyncio.TaskGroup() as group:
                    recent_task = group.create_task(self.history(contract, clock, "960 S"))
                    prior_task = group.create_task(self.history(contract, previous_close, "60 S"))
                bars, previous = recent_task.result(), prior_task.result()
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
            result["detail"] = {
                "error": repr(exc),
                "request_status": "FAILED_OR_TIMED_OUT",
            }
        return result

    async def execute(self, event: dict[str, Any], underlying: Any) -> None:
        if not self.broker.entries_armed() or self.config.missing() or self.pause:
            self.store.outcome(event, "UNARMED", {"missing_settings": self.config.missing()})
            return
        ticker = None
        callback = None
        anchor: asyncio.Future[Any] | None = None
        request_id: int | None = None
        stage = "ENTRY_GUARD"
        ticks_received = 0
        last_trade_at: str | None = None
        chain_task: asyncio.Task[Any] | None = None
        anchor_received: tuple[float, str] | None = None
        try:
            self.broker.guard()
            generation = self.broker.data_generation
            baseline = datetime.fromisoformat(event["entry_at"])
            if now() >= baseline:
                raise ValueError("BASELINE_ANCHOR_MISSED")
            # Subscribe before the boundary. The first eligible trade supplies open(j+2).
            stage = "BASELINE_SUBSCRIPTION"
            ticker = self.broker.ib.reqTickByTickData(underlying, "Last", 0, False)
            request_id = self.broker.ib.wrapper.ticker2ReqId["Last"][ticker]
            # Streaming subscriptions have no error future in ib_async. Register
            # this wait so its existing RaiseRequestErrors handling reaches us.
            anchor = self.broker.ib.wrapper.startReq(request_id, underlying)

            def on_tick(t: Any) -> None:
                nonlocal ticks_received, last_trade_at, anchor_received
                for tick in t.tickByTicks:
                    ticks_received += 1
                    last_trade_at = tick.time.isoformat()
                    if (
                        generation == self.broker.data_generation
                        and not self.broker.market_data_block
                        and baseline <= tick.time < baseline + timedelta(minutes=1)
                        and tick.price > 0
                        and math.isfinite(tick.price)
                        and anchor is not None
                        and not anchor.done()
                    ):
                        anchor_received = (tick.price, tick.time.isoformat())
                        anchor.set_result((tick.price, tick.time))

            callback = on_tick
            ticker.updateEvent += callback
            deadline = baseline + timedelta(seconds=self.config.number("entry_deadline_seconds"))
            stage = "OPTION_CHAIN"
            chain_task = asyncio.create_task(self.broker.standard_chain(underlying))
            done, _ = await asyncio.wait(
                {chain_task, anchor},
                timeout=max(0, (deadline - now()).total_seconds()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if chain_task in done:
                chain_task.result()
            elif anchor in done:
                anchor.result()  # Subscription failure interrupts pending metadata.
                await asyncio.wait_for(chain_task, max(0, (deadline - now()).total_seconds()))
            else:
                raise TimeoutError()
            stage = "BASELINE_ANCHOR"
            price, stamp = await asyncio.wait_for(
                anchor,
                max(0, (min(deadline, baseline + timedelta(minutes=1)) - now()).total_seconds()),
            )
            self.store.outcome(
                event, "ANCHOR_OBSERVED", {"price": price, "broker_trade_time": stamp.isoformat()}
            )
            if self.pause:
                raise ValueError("ENTRIES_PAUSED")
            if generation != self.broker.data_generation:
                raise ValueError("BASELINE_DATA_INTERRUPTED")
            stage = "ENTRY_PREPARATION"
            async with asyncio.timeout(max(0, (deadline - now()).total_seconds())):
                await self.broker.enter(event, underlying, price)
        except Exception as exc:
            failures = list(exc.exceptions) if isinstance(exc, ExceptionGroup) else [exc]
            exc = next((e for e in failures if not isinstance(e, TimeoutError)), failures[0])
            stage = getattr(exc, "entry_stage", getattr(exc.__cause__, "entry_stage", stage))
            reason = str(exc) or type(exc).__name__
            detail: dict[str, Any] = {}
            if isinstance(exc, OptionChainError):
                detail.update(exc.detail)
            if anchor is not None and anchor.done() and not anchor.cancelled():
                tick_error = anchor.exception()
                if isinstance(tick_error, RequestError) and tick_error is not exc:
                    detail["tick_subscription_error"] = {
                        "request_id": tick_error.reqId,
                        "code": tick_error.code,
                        "message": tick_error.message,
                    }
            if isinstance(exc, RequestError):
                detail["broker_error"] = {
                    "request_id": exc.reqId,
                    "code": exc.code,
                    "message": exc.message,
                }
                if exc.reqId == request_id:
                    stage = "BASELINE_SUBSCRIPTION"
                    reason = "BASELINE_SUBSCRIPTION_FAILED"
            if isinstance(exc, TimeoutError):
                reason = (
                    "BASELINE_TRADE_NOT_RECEIVED"
                    if stage == "BASELINE_ANCHOR"
                    else stage + "_TIMEOUT"
                )
            self.store.outcome(
                event,
                "EXECUTION_FAILED",
                {
                    "error": reason,
                    "exception": repr(exc),
                    "stage": stage,
                    "ticks_received": ticks_received,
                    "last_trade_at": last_trade_at,
                    "tick_request_id": request_id,
                    "anchor_received": anchor_received is not None,
                    "anchor_evidence": anchor_received,
                    **self.store.execution_evidence(event),
                    "stage_errors": [
                        {"stage": getattr(e, "entry_stage", stage), **exception_info(e)}
                        for e in failures
                    ],
                    **detail,
                },
            )
            log.warning("FIRST4 execution failed %s at %s: %s", event["symbol"], stage, reason)
        finally:
            if chain_task is not None:
                if not chain_task.done():
                    chain_task.cancel()
                await asyncio.gather(chain_task, return_exceptions=True)
            if ticker is not None:
                ticker.updateEvent -= callback
                try:
                    self.broker.ib.cancelTickByTickData(underlying, "Last")
                finally:
                    if anchor is not None:
                        anchor.cancel()
                        if not anchor.cancelled():
                            anchor.exception()  # Retrieve errors even if chain lookup failed first.
                    if request_id is not None:
                        self.broker.ib.finish_request(request_id)
                        if not self.broker.ib.flow_last_retained(request_id):
                            self.broker.ib.wrapper.reqId2Ticker.pop(request_id, None)

    async def scan(
        self,
        session: str,
        opened: datetime,
        closed: datetime,
        previous_close: datetime,
        clock: datetime,
    ) -> None:
        sub = ScannerSubscription(
            instrument="STK",
            locationCode="STK.US",
            scanCode="MOST_ACTIVE",
            numberOfRows=25,
            averageOptionVolumeAbove=1,
        )
        started = asyncio.get_running_loop().time()
        self.scan_metrics = {}
        generation = self.broker.data_generation
        rows = await self.broker.ib.scanner(
            sub, [TagValue("changePercAbove", "5.5"), TagValue("priceBelow", "20")]
        )
        if generation != self.broker.data_generation:
            raise ValueError("SCANNER_DATA_INTERRUPTED")
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
            *(self.candidate(r, opened, clock, previous_close, started + 43) for r in fresh)
        )
        # The native observation above is already known. A later history failure
        # rejects that frozen appearance with its actual request error; it does
        # not erase other successfully observed candidates or permit readmission.
        # Inputs complete as one batch; asynchronous response order cannot allocate slots.
        selected = self.store.observe(session, clock, closed, candidates)
        self.last_observation = clock.isoformat()
        self.scan_metrics.update(
            batch_seconds=asyncio.get_running_loop().time() - started,
            candidates=len(fresh),
            requests=2 * len(fresh),
        )
        contracts = {r.contractDetails.contract.conId: r.contractDetails.contract for r in fresh}
        for event in selected:
            task = asyncio.create_task(self.execute(event, contracts[event["con_id"]]))
            self.tasks.add(task)
            task.add_done_callback(self.execution_done)
        if self.config.order_flow.enabled:
            try:
                for event in selected:
                    self.order_flow.allocate(event, contracts[event["con_id"]])
            except Exception as exc:
                self.order_flow.problem = "ALLOCATION_CAPTURE_FAILED: " + str(exc)
                log.exception("Optional observation allocation failed")

    def execution_done(self, task: asyncio.Task[None]) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            self.broker.fatal_error = "EXECUTION_TASK_FAILED: " + str(error)
            self.report_failure("execution_task", error)

    def report_failure(self, name: str, error: BaseException) -> None:
        self.broker.report_error(
            name, {"time": now().isoformat(), "error": str(error) or type(error).__name__}
        )

    async def cancel_tasks(self, tasks: set[asyncio.Task[None]]) -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=5)
            for task in done:
                if not task.cancelled():
                    task.exception()
            if pending:
                self.broker.fatal_error = "SHUTDOWN_TASK_TIMEOUT"
                log.critical("FIRST4 cleanup exceeded five seconds: %s", pending)
                for task in pending:
                    task.add_done_callback(self.execution_done)

    async def critical(self, name: str, work: Callable[[], Awaitable[None]]) -> None:
        try:
            await work()
            if name != "opening" and not self.stopping:
                raise RuntimeError(name + " terminated unexpectedly")
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError) and self.stopping:
                raise
            if name == "manager":
                self.manager_health = "FAILED"
            else:
                self.worker_health = "FAILED"
            # Inhibit entries in the failing task before another ready coroutine
            # can resume preparation or reach the submission boundary.
            self.broker.management_block = "CRITICAL_TASK_FAILED: " + name
            self.broker.fatal_error = self.broker.management_block
            self.report_failure(name, exc)
            if isinstance(exc, asyncio.CancelledError):
                raise RuntimeError(name + " was cancelled unexpectedly") from exc
            raise

    async def run(self) -> None:
        self.stopping = False
        self.worker_health = self.manager_health = "RUNNING"
        manager = asyncio.create_task(self.critical("manager", self.maintain_broker))
        opening = asyncio.create_task(self.critical("opening", self.arm_at_open))
        scanner = asyncio.create_task(self.critical("worker", self.scan_sessions))
        self.critical_tasks = {manager, opening, scanner}
        observer = asyncio.create_task(self.order_flow.run())
        try:
            watched = set(self.critical_tasks)
            while watched:
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    watched.remove(task)
                    task.result()  # A normal dated-check completion is expected.
        finally:
            self.stopping = True
            self.broker.management_block = self.broker.management_block or "WORKER_STOPPED"
            await self.cancel_tasks(self.critical_tasks | self.tasks)
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            self.mark_stopped()

    def mark_stopped(self) -> None:
        if self.worker_health != "FAILED":
            self.worker_health = "STOPPED"
        if self.manager_health != "FAILED":
            self.manager_health = "STOPPED"

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

    def opening_guard(self, day: str, opened: datetime, *, reconciling: bool = False) -> None:
        self.broker.guard(require_reconciled=not reconciling)
        self.config.require_settings()
        if self.config.armed or str(self.config.arm_after_quote_check_on) != day:
            raise ValueError("OPENING_ARM_DATE_NOT_AUTHORIZED")
        # Finish before the first possible Q5 admission at open + 15 minutes.
        if not opened <= now() < opened + timedelta(minutes=14):
            raise ValueError("OPENING_VERIFICATION_WINDOW_EXPIRED")
        if self.pause or self.store.get_meta("paused", False):
            raise ValueError("ENTRIES_PAUSED")
        if self.broker.market_data_block or self.broker.data_problem:
            raise ValueError(self.broker.data_problem or "MARKET_DATA_COMPETING_SESSION_10197")
        if self.broker.management_block:
            raise ValueError(self.broker.management_block)
        if self.session != day or self.problem or self.broker.entry_blocker:
            raise ValueError(self.problem or self.broker.entry_blocker or "SESSION_NOT_READY")
        state = self.store.db.execute(
            "SELECT blocked FROM first4_sessions WHERE session=?", (day,)
        ).fetchone()
        if state and state["blocked"]:
            raise ValueError(state["blocked"])
        if any(p.position for p in self.broker.ib.positions(self.config.expected_account)):
            raise ValueError("OPENING_CHECK_REQUIRES_RECONCILED_FLAT_ACCOUNT")

    async def opening_wait(
        self,
        work: Awaitable[Any],
        day: str,
        opened: datetime,
        generation: int,
        deadline: datetime,
        *,
        reconciling: bool = False,
    ) -> Any:
        task = asyncio.ensure_future(work)
        stop = asyncio.get_running_loop().time() + max(0, (deadline - now()).total_seconds())
        try:
            while True:
                # Even a simultaneous timeout must not mask a completed fatal error.
                if task.done():
                    result = task.result()
                    if now() >= deadline or asyncio.get_running_loop().time() >= stop:
                        raise TimeoutError()
                    return result
                if generation != self.broker.data_generation:
                    raise ValueError("OPENING_CHECK_INTERRUPTED")
                # Reconciliation temporarily clears its own completion flag.
                self.opening_guard(day, opened, reconciling=reconciling)
                remaining = min(
                    (deadline - now()).total_seconds(), stop - asyncio.get_running_loop().time()
                )
                if remaining <= 0:
                    raise TimeoutError()
                await asyncio.wait({task}, timeout=min(0.05, remaining))
        finally:
            if not task.done():
                task.cancel()
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result

    async def check_opening(self, day: str, opened: datetime) -> None:
        if self.opening_active:
            return
        key = "opening_check:" + day
        report: dict[str, Any] = {
            "session": day,
            "status": "CHECKING",
            "started_at": now().isoformat(),
            "deadline": (opened + timedelta(minutes=14)).isoformat(),
            "purpose": "READ_ONLY_DATA_CHECK_NOT_FIRST4_SIGNAL",
            "transmitted_orders": 0,
            "attempts": [],
            "policy": "DEADLINE_RETRIES_60S_ATTEMPT_5S_PAUSE",
        }
        self.store.set_meta(key, report)
        self.opening_active = True
        try:
            generation = self.broker.data_generation
            deadline = opened + timedelta(minutes=14)
            attempt = 0
            while True:
                if generation != self.broker.data_generation:
                    raise ValueError("OPENING_CHECK_INTERRUPTED")
                self.opening_guard(day, opened)
                attempt += 1
                report.update(attempt=attempt)
                report.pop("next_attempt_at", None)
                self.store.set_meta(key, report)
                attempt_deadline = min(deadline, now() + timedelta(seconds=OPENING_ATTEMPT_SECONDS))
                audit: dict[str, Any] = {
                    "attempt": attempt,
                    "stage": "RECONCILIATION",
                    "started_at": now().isoformat(),
                    "deadline": attempt_deadline.isoformat(),
                    "start_margin_seconds": (deadline - now()).total_seconds(),
                    "stock_quote": None,
                }
                self.broker.opening_diagnostic = audit
                report["attempts"].append(audit)
                self.store.set_meta(key, report)
                try:
                    await self.opening_wait(
                        self.broker.reconcile(require_flat=True),
                        day,
                        opened,
                        generation,
                        min(attempt_deadline, now() + timedelta(seconds=15)),
                        reconciling=True,
                    )
                    self.opening_guard(day, opened)
                    audit["stage"] = "OPTION_ACCESS"
                    evidence = await self.opening_wait(
                        option_access(self.broker, attempt_deadline),
                        day,
                        opened,
                        generation,
                        attempt_deadline,
                    )
                    if generation != self.broker.data_generation:
                        raise ValueError("OPENING_CHECK_INTERRUPTED")
                    self.opening_guard(day, opened)
                    if evidence.get("blockers") or not all(
                        evidence.get("checks", {}).get(name) is True
                        for name in (
                            "qualified_usd_standard_multiplier",
                            "fresh_realtime_option_quotes",
                            "combo_price_increment",
                        )
                    ):
                        raise ValueError("OPENING_CHECK_INCOMPLETE")
                    report.update(evidence)
                    audit["outcome"] = "PASSED"
                    report.pop("last_attempt_error", None)
                    break
                except BaseException as exc:
                    audit.update(
                        outcome="CANCELLED"
                        if isinstance(exc, asyncio.CancelledError)
                        else "FAILED",
                        exception=exception_info(exc),
                    )
                    audit.update(
                        ended_at=now().isoformat(),
                        end_margin_seconds=(deadline - now()).total_seconds(),
                    )
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    if generation != self.broker.data_generation:
                        raise ValueError("OPENING_CHECK_INTERRUPTED") from exc
                    self.opening_guard(day, opened)
                    # Only transient read-only failures retry. Identity, ownership,
                    # metadata ambiguity, missing observations and revocation do not.
                    retryable = (
                        isinstance(exc, TimeoutError)
                        and audit["stage"]
                        in {
                            "OPTION_ACCESS",
                            "STOCK_QUALIFICATION",
                            "OPTION_CHAIN",
                            "STOCK_QUOTE",
                            "CONTRACT_QUALIFICATION",
                            "COMBO_METADATA",
                            "OPTION_QUOTES",
                        }
                    ) or (
                        isinstance(exc, ValueError)
                        and str(exc)
                        in {
                            "PROBE_STOCK_QUOTE_UNAVAILABLE",
                            "OPTION_QUOTES_INVALID_STALE_OR_UNAVAILABLE",
                            "COMBO_PRICE_INCREMENT_UNAVAILABLE",
                        }
                    )
                    if not retryable:
                        raise
                    if now() + timedelta(seconds=OPENING_RETRY_SECONDS) >= deadline:
                        raise ValueError("OPENING_VERIFICATION_WINDOW_EXPIRED") from exc
                    report.update(
                        last_attempt_error=exception_info(exc)["message"] or type(exc).__name__,
                        next_attempt_at=(
                            now() + timedelta(seconds=OPENING_RETRY_SECONDS)
                        ).isoformat(),
                    )
                    # Keep CHECKING durable throughout retries: restart must never
                    # replay a completed or interrupted dated opening check.
                    self.store.set_meta(key, report)
                    log.warning(
                        "FIRST4 opening attempt %s failed; retrying: %s",
                        attempt,
                        report["last_attempt_error"],
                    )
                    await self.opening_wait(
                        opening_pause(OPENING_RETRY_SECONDS),
                        day,
                        opened,
                        generation,
                        deadline,
                    )
                finally:
                    if "ended_at" not in audit:
                        audit.update(
                            ended_at=now().isoformat(),
                            end_margin_seconds=(deadline - now()).total_seconds(),
                        )
                    self.store.set_meta(key, report)
            self.opening_guard(day, opened)
            report["status"] = "ARMED"
            report["armed_at"] = now().isoformat()
            self.store.set_meta(key, report)
            self.broker.opening_verified_session = day
            log.info("FIRST4 PAPER opening verification passed for %s; entries enabled", day)
        except asyncio.CancelledError:
            self.broker.opening_verified_session = None
            report.update(status="FAILED", error="OPENING_CHECK_INTERRUPTED")
            self.broker.report_error(key, report)
            raise
        except Exception as exc:
            self.broker.opening_verified_session = None
            report.update(
                status="FAILED", error=exception_info(exc)["message"] or type(exc).__name__
            )
            self.broker.report_error(key, report)
            log.warning("FIRST4 opening verification failed: %s", report["error"])
        finally:
            self.opening_active = False

    async def maintain_broker(self) -> None:
        """Exit obligations never wait for scanner history or option qualification."""
        while self.running:
            try:
                if self.broker.fatal_error:
                    raise sqlite3.OperationalError(self.broker.fatal_error)
                if not self.broker.ib.isConnected():
                    async with asyncio.timeout(30):
                        await self.broker.connect()
                elif not self.broker.reconciled and not self.broker.reconciliation_lock.locked():
                    async with asyncio.timeout(15):
                        await self.broker.reconcile()
                try:
                    async with asyncio.timeout(10):
                        await self.broker.cancel_due_entries()
                except Exception as exc:
                    self.broker.problem = str(exc) or type(exc).__name__
                    self.broker.report_error(
                        "entry_cancellation_error",
                        {"time": now().isoformat(), "error": self.broker.problem},
                    )
                await self.broker.close_due()
                if self.broker.fatal_error:
                    raise sqlite3.OperationalError(self.broker.fatal_error)
                self.manager_health = "RUNNING"
                self.broker.management_block = ""
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise
            except sqlite3.Error:
                raise
            except Exception as exc:
                self.manager_health = "DEGRADED"
                self.broker.management_block = "BROKER_MANAGER_UNAVAILABLE"
                self.broker.problem = str(exc) or type(exc).__name__
                self.broker.report_error(
                    "broker_manager_error",
                    {
                        "time": now().isoformat(),
                        "error": self.broker.problem,
                    },
                )
                await asyncio.sleep(5)

    async def scan_step(self, calendar: Any) -> None:
        current_time = now()
        today = current_time.astimezone(ZoneInfo("America/New_York")).date()
        if self.calendar_date != today:
            frame = await asyncio.to_thread(
                calendar.schedule,
                start_date=today - timedelta(days=10),
                end_date=today + timedelta(days=1),
            )
            self.schedule = [
                (str(d.date()), r.market_open.to_pydatetime(), r.market_close.to_pydatetime())
                for d, r in frame.iterrows()
            ]
            self.calendar_date = today
        current = next((r for r in self.schedule if r[0] == today.isoformat()), None)
        if self.broker.opening_verified_session != today.isoformat():
            self.broker.opening_verified_session = None
        if not current:
            self.problem = "EXCHANGE_CLOSED"
            return
        day, opened, closed = current
        if self.session != day:
            self.session = day
            self.broker.chains.clear()
            state = self.store.db.execute(
                "SELECT * FROM first4_sessions WHERE session=?", (day,)
            ).fetchone()
            self.problem = state["blocked"] if state and state["blocked"] else ""
            self.next_clock = opened + timedelta(minutes=1)
            if current_time >= self.next_clock and not self.problem:
                self.problem = "SESSION_START_OR_SCANNER_HISTORY_MISSED"
                self.store.block(day, self.problem)
        if self.problem or self.next_clock is None:
            return
        # Advance only after Store.observe commits the required minute. A
        # reconnect before any due observation is not a scanner-history gap.
        if self.next_clock < closed and current_time >= self.next_clock + timedelta(seconds=2):
            self.problem = "SCANNER_MINUTE_MISSED"
            self.store.block(day, self.problem)
            return
        if not opened < current_time < closed or current_time < self.next_clock:
            return
        if (
            not self.broker.ib.isConnected()
            or not self.broker.reconciled
            or not self.broker.upstream_available
            or self.broker.market_data_block
        ):
            return
        prior_close = [r[2] for r in self.schedule if r[0] < day][-1]
        async with asyncio.timeout(45):
            await self.scan(day, opened, closed, prior_close, self.next_clock)
        self.next_clock += timedelta(minutes=1)

    async def scan_sessions(self) -> None:
        calendar = get_market_calendar("NYSE")
        while self.running:
            try:
                await self.scan_step(calendar)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise
            except sqlite3.Error:
                raise
            except Exception as exc:
                self.problem = str(exc) or type(exc).__name__
                self.report_failure("runtime_error", exc)
                if self.session:
                    self.store.block(self.session, self.problem)
                log.exception("FIRST4 runtime blocked; exit obligations retained")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self.stopping = True
        self.running = False
        self.pause = True
        self.broker.management_block = "STOPPING"
        await self.cancel_tasks(set(self.tasks))
        self.broker.ib.disconnect()
