"""One owner of qualified feeds, shared consumers, quotas and cancellation debt."""

import asyncio
import json
import math
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ib_async import Ticker
from ib_async.objects import BarDataList

from stocker_execution.config import MarketDataConfig
from stocker_execution.pacing import CORE, EXPOSURE, OPTIONAL, drain, priority


@dataclass
class Subscription:
    key: tuple[Any, ...]
    request_id: int
    generation: int
    contract: Any
    feed: str
    params: dict[str, Any]
    consumers: dict[str, str] = field(default_factory=dict)
    state: str = "REQUESTED"
    value: Any = None
    task: asyncio.Task[Any] | None = None
    cancellation: asyncio.Task[Any] | None = None
    error: str = ""
    last_receipt: datetime | None = None
    last_change: datetime | None = None
    bid_at: datetime | None = None
    ask_at: datetime | None = None
    fingerprint: tuple[Any, ...] = ()
    sink: Callable[[dict[str, Any]], None] | None = None


class Subscriptions:
    def __init__(self, ib: Any, config: MarketDataConfig):
        self.ib, self.config = ib, config
        self.items: dict[tuple[Any, ...], Subscription] = {}
        self.by_request: dict[int, Subscription] = {}
        self.recent: deque[dict[str, Any]] = deque(maxlen=64)
        self.errors: deque[dict[str, Any]] = deque(maxlen=32)
        self.failed_until: dict[tuple[Any, ...], float] = {}
        self.generation = 0
        self.lock = asyncio.Lock()
        self.history_lock = asyncio.Lock()
        self.history_slots = asyncio.Semaphore(2)
        self.history_next = 0.0
        self.history_active = 0
        self.history_cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self.history_pending: dict[tuple[Any, ...], asyncio.Task[Any]] = {}
        self.capacity_cooldown = 0.0
        self.effective_cap = config.line_budget
        self.tasks: set[asyncio.Task[Any]] = set()
        self.quote_sink: Callable[[Subscription], None] | None = None
        self.ib.errorEvent += self.error
        self.ib.disconnectedEvent += self.disconnected

    def launch(self, coroutine: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        return task

    @staticmethod
    def identity(contract: Any, feed: str, params: dict[str, Any]) -> tuple[Any, ...]:
        if contract.conId <= 0 or contract.secType not in {"FUT", "FOP", "CASH"}:
            raise ValueError("QUALIFIED_DATA_CONTRACT_REQUIRED")
        return (
            contract.conId,
            contract.secType,
            contract.exchange,
            contract.currency,
            feed,
            json.dumps(params, sort_keys=True),
        )

    def count(self, feed: str | None = None) -> int:
        return sum(
            s.state != "DISCONNECTED" and (feed is None or s.feed == feed)
            for s in self.items.values()
        )

    def temporary_count(self) -> int:
        return sum("selection" in s.consumers.values() for s in self.items.values())

    def check_capacity(self, feed: str, purpose: str, level: int) -> None:
        if feed == "TICK_BY_TICK":
            raise ValueError("TICK_BY_TICK_NOT_ENABLED")
        if asyncio.get_running_loop().time() < self.capacity_cooldown and level >= CORE:
            raise ValueError("BROKER_CAPACITY_BACKOFF")
        # Keep eight lines for four owned options and four distinct retained underlyings.
        reserve = 8 if purpose in {"selection", "depth"} else 0
        if self.count() >= max(0, self.effective_cap - reserve):
            raise ValueError("APP_MARKET_DATA_BUDGET_EXHAUSTED")
        if feed == "DEPTH" and self.count("DEPTH") >= self.config.depth_slots:
            raise ValueError("L2_NOT_CAPTURED_CAPACITY")
        if purpose == "selection" and self.temporary_count() >= self.config.temporary_option_quotes:
            raise ValueError("OPTION_QUOTE_POOL_EXHAUSTED")

    async def acquire(
        self,
        contract: Any,
        feed: str,
        owner: str,
        purpose: str,
        params: dict[str, Any] | None = None,
        level: int = CORE,
        deadline: float | None = None,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> Subscription:
        params = params or {}
        key = self.identity(contract, feed, params)
        deadline = deadline or asyncio.get_running_loop().time() + 15
        if key not in self.items and level <= EXPOSURE and self.count() >= self.effective_cap:
            await self.shed_optional()
            # A reduced broker allowance may exhaust the reserved headroom. Sacrifice only
            # unshared entry-monitoring feeds; never another position's data or FX.
            for victim in reversed(list(self.items.values())):
                if self.count() < self.effective_cap:
                    break
                if set(victim.consumers.values()) <= {"core_quote", "core_bars"}:
                    victim.error = "CORE_DATA_PREEMPTED_FOR_EXPOSURE"
                    for consumer in list(victim.consumers):
                        await self.release(victim, consumer)
        async with self.lock:
            existing = self.items.get(key)
            if existing:
                if existing.state in {"CANCELLING", "FAILED", "DISCONNECTED"}:
                    raise ValueError(existing.error or "SUBSCRIPTION_CANCELLATION_PENDING")
                if (
                    purpose == "selection"
                    and "selection" not in existing.consumers.values()
                    and self.temporary_count() >= self.config.temporary_option_quotes
                ):
                    raise ValueError("OPTION_QUOTE_POOL_EXHAUSTED")
                existing.consumers[owner] = purpose
                sub = existing
            else:
                if self.failed_until.get(key, 0) > asyncio.get_running_loop().time():
                    raise ValueError("DATA_REQUEST_BACKOFF")
                self.check_capacity(feed, purpose, level)
                self.generation += 1
                sub = Subscription(
                    key,
                    self.ib.client.getReqId(),
                    self.generation,
                    contract,
                    feed,
                    params,
                    {owner: purpose},
                    sink=sink,
                )
                self.items[key] = sub
                self.by_request[sub.request_id] = sub
                sub.task = self.launch(self._start(sub, level, deadline))
        try:
            if sub.task:
                async with asyncio.timeout_at(deadline):
                    await asyncio.shield(sub.task)
            if sub.error:
                raise ValueError(sub.error)
            return sub
        except BaseException as exc:
            await self.release(sub, owner)
            task = asyncio.current_task()
            if isinstance(exc, asyncio.CancelledError) and not (task and task.cancelling()):
                raise ValueError("SUBSCRIPTION_DISCONNECTED") from None
            raise

    async def _start(self, sub: Subscription, level: int, deadline: float) -> None:
        try:
            with priority(level, deadline) as group:
                if sub.feed == "QUOTE":
                    ticker = Ticker(contract=sub.contract, defaults=self.ib.wrapper.defaults)
                    ticker.marketDataType = 0
                    sub.value = ticker
                    self.ib.wrapper.reqId2Ticker[sub.request_id] = ticker
                    ticker.updateEvent += lambda t: self.quote_update(sub, t)
                    self.ib.client.reqMktData(sub.request_id, sub.contract, "", False, False, [])
                elif sub.feed == "DEPTH":
                    if sub.contract.secType != "FUT" or sub.contract.exchange == "SMART":
                        raise ValueError("DIRECT_FUTURES_DEPTH_ONLY")
                    self.ib.wrapper.depth_handlers[sub.request_id] = lambda e: self.depth_update(
                        sub, e
                    )
                    self.ib.client.reqMktDepth(
                        sub.request_id, sub.contract, sub.params["levels"], False, []
                    )
                elif sub.feed == "BARS":
                    bars = BarDataList()  # type: ignore[no-untyped-call]
                    bars.reqId, bars.contract = sub.request_id, sub.contract
                    sub.value = bars
                    future = self.ib.wrapper.startReq(sub.request_id, sub.contract, container=bars)
                    self.ib.wrapper.startSubscription(sub.request_id, bars, sub.contract)
                    bars.updateEvent += lambda *args: self.bar_update(sub)
                    await self.history_turn()
                    self.ib.client.reqHistoricalData(
                        sub.request_id,
                        sub.contract,
                        "",
                        sub.params["duration"],
                        "1 min",
                        "TRADES",
                        False,
                        2,
                        True,
                        [],
                    )
                    await drain(group)
                    async with asyncio.timeout_at(deadline):
                        await future
                    sub.state = "ACTIVE"
                    sub.last_receipt = datetime.now(UTC)
                else:
                    raise ValueError("UNSUPPORTED_FEED")
                await drain(group)
        except BaseException as exc:
            sub.error = str(exc) or type(exc).__name__
            sub.state = "FAILED"
            raise

    def quote_update(self, sub: Subscription, ticker: Any) -> None:
        if self.by_request.get(sub.request_id) is not sub or sub.state in {"FAILED", "CANCELLING"}:
            return
        sub.state = "ACTIVE"
        sub.last_receipt = datetime.now(UTC)
        for tick in ticker.ticks:
            if tick.tickType == 1:
                sub.bid_at = tick.time
            elif tick.tickType == 2:
                sub.ask_at = tick.time
        values = tuple(
            v if math.isfinite(v) else None for v in (ticker.bid, ticker.ask, ticker.last)
        )
        if values != sub.fingerprint:
            sub.fingerprint = values
            sub.last_change = sub.last_receipt
        if self.quote_sink:
            self.quote_sink(sub)

    def bar_update(self, sub: Subscription) -> None:
        if self.by_request.get(sub.request_id) is sub and sub.state not in {"CANCELLING", "FAILED"}:
            sub.last_receipt = datetime.now(UTC)
            sub.state = "ACTIVE"

    def depth_update(self, sub: Subscription, event: dict[str, Any]) -> None:
        if self.by_request.get(sub.request_id) is not sub or sub.state in {"CANCELLING", "FAILED"}:
            return
        sub.last_receipt = datetime.now(UTC)
        sub.state = "ACTIVE"
        if sub.sink:
            try:
                sub.sink({**event, "generation": sub.generation, "request_id": sub.request_id})
            except Exception as exc:
                sub.error = f"L2_CALLBACK_FAILED:{type(exc).__name__}"
                sub.state = "FAILED"

    async def release(
        self, sub: Subscription, owner: str, reason: str = "SUBSCRIPTION_RELEASED"
    ) -> None:
        async with self.lock:
            if self.items.get(sub.key) is not sub:
                return
            purpose = sub.consumers.pop(owner, None)
            if sub.consumers:
                return
            if not sub.cancellation:
                sub.state = "CANCELLING"
                self.depth_update_error(sub, reason)
                level = (
                    EXPOSURE
                    if purpose in {"exposure", "fx"}
                    else CORE
                    if purpose in {"core_quote", "core_bars"}
                    else OPTIONAL
                )
                sub.cancellation = self.launch(self._cancel(sub, level))
        try:
            await asyncio.shield(sub.cancellation)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if (task and task.cancelling()) or self.items.get(sub.key) is sub:
                raise

    async def _cancel(self, sub: Subscription, level: int) -> None:
        # Cancel before awaiting a still-pending start: the start must not outlive its owner.
        if sub.task and not sub.task.done() and sub.task is not asyncio.current_task():
            sub.task.cancel()
            await asyncio.gather(sub.task, return_exceptions=True)
        sub.state = "CANCELLING"
        try:
            if self.ib.isConnected():
                with priority(level, asyncio.get_running_loop().time() + 15) as group:
                    if sub.feed == "QUOTE":
                        self.ib.client.cancelMktData(sub.request_id)
                    elif sub.feed == "DEPTH":
                        self.ib.client.cancelMktDepth(sub.request_id, False)
                    elif sub.feed == "BARS":
                        self.ib.client.cancelHistoricalData(sub.request_id)
                    await drain(group)
                # Local wire completion + drain, NOT a broker cancellation acknowledgement.
                await asyncio.sleep(self.config.cancel_drain_seconds)
            self._remove(sub)
        except BaseException:
            # Retain capacity debt until disconnected; no speculative replacement.
            if self.items.get(sub.key) is sub:
                sub.error = "CANCEL_WIRE_UNCERTAIN"
            raise

    def _remove(self, sub: Subscription) -> None:
        if self.items.get(sub.key) is sub:
            self.items.pop(sub.key)
        if self.by_request.get(sub.request_id) is sub:
            self.by_request.pop(sub.request_id, None)
            self.ib.wrapper.reqId2Ticker.pop(sub.request_id, None)
            self.ib.wrapper.depth_handlers.pop(sub.request_id, None)
            if sub.feed == "BARS" and sub.value is not None:
                self.ib.wrapper.endSubscription(sub.value)
                self.ib.finish_request(sub.request_id)
        self.recent.append(
            {
                "con_id": sub.contract.conId,
                "feed": sub.feed,
                "generation": sub.generation,
                "state": sub.state,
                "error": sub.error,
            }
        )

    def error(self, request_id: int, code: int, message: str, contract: Any = None) -> None:
        sub = self.by_request.get(request_id)
        if code == 317:
            return  # Raw wrapper delivered a reset before this error event.
        if code not in {100, 101, 162, 200, 309, 354, 420, 10089, 10090, 10167, 10168, 10225}:
            return
        self.errors.append(
            {
                "at": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "code": code,
                "message": message,
            }
        )
        if sub:
            sub.state, sub.error = "FAILED", f"IBKR_{code}:{message}"
            if len(self.failed_until) >= 128:
                self.failed_until.pop(next(iter(self.failed_until)))
            self.failed_until[sub.key] = (
                asyncio.get_running_loop().time() + self.config.rejection_backoff_seconds
            )
            if sub.sink:
                self.depth_update_error(sub, sub.error)
        if code in {100, 101}:
            self.capacity_cooldown = (
                asyncio.get_running_loop().time() + self.config.rejection_backoff_seconds
            )
            if code == 101:
                self.effective_cap = min(self.effective_cap, max(0, self.count() - 1))
            self.launch(self.shed_optional())

    def depth_update_error(self, sub: Subscription, reason: str) -> None:
        if sub.sink:
            sub.sink(
                {
                    "kind": "GAP",
                    "reason": reason,
                    "generation": sub.generation,
                    "request_id": sub.request_id,
                }
            )

    async def shed_optional(self) -> None:
        releases = []
        for sub in list(self.items.values()):
            if sub.feed == "DEPTH" or set(sub.consumers.values()) <= {"selection"}:
                for owner in list(sub.consumers):
                    releases.append(self.release(sub, owner, "OPTIONAL_CAPACITY_SHED"))
        await asyncio.gather(*releases, return_exceptions=True)

    def disconnected(self, *args: Any) -> None:
        for sub in list(self.items.values()):
            sub.state = "DISCONNECTED"
            self.depth_update_error(sub, "DISCONNECTED")
            if sub.task and not sub.task.done():
                sub.task.cancel()
            if sub.cancellation and not sub.cancellation.done():
                sub.cancellation.cancel()
            self._remove(sub)
        for task in self.history_pending.values():
            task.cancel()
        # Broker rejects remain backed off across reconnects. No retry storm.

    async def history_turn(self) -> None:
        async with self.history_lock:
            delay = self.history_next - asyncio.get_running_loop().time()
            if delay > 0:
                await asyncio.sleep(delay)
            self.history_next = asyncio.get_running_loop().time() + 0.35

    async def historical(
        self, key: tuple[Any, ...], request: Callable[[], Any], cache_seconds: int = 300
    ) -> Any:
        cached = self.history_cache.get(key)
        if cached and cached[0] > asyncio.get_running_loop().time():
            return cached[1]
        if key not in self.history_pending:
            if len(self.history_pending) >= 16:
                raise ValueError("HISTORICAL_QUEUE_FULL")
            task = self.launch(self._historical(key, request, cache_seconds))
            self.history_pending[key] = task
            task.add_done_callback(lambda _: self.history_pending.pop(key, None))
        try:
            return await asyncio.shield(self.history_pending[key])
        except asyncio.CancelledError:
            caller = asyncio.current_task()
            if caller and caller.cancelling():
                raise
            raise ValueError("HISTORICAL_DATA_DISCONNECTED") from None

    async def _historical(
        self, key: tuple[Any, ...], request: Callable[[], Any], cache_seconds: int
    ) -> Any:
        async with asyncio.timeout(20), self.history_slots:
            self.history_active += 1
            try:
                await self.history_turn()
                with priority(CORE, asyncio.get_running_loop().time() + 20):
                    value = await request()
                if value and cache_seconds:
                    if len(self.history_cache) >= 64:
                        self.history_cache.pop(next(iter(self.history_cache)))
                    self.history_cache[key] = (
                        asyncio.get_running_loop().time() + cache_seconds,
                        value,
                    )
                return value
            finally:
                self.history_active -= 1

    @asynccontextmanager
    async def option_batch(
        self, contracts: list[Any], owner: str, deadline: float
    ) -> AsyncIterator[list[Subscription]]:
        """One small batch; caller preserves its exact selection rule across sequential batches."""
        if len(contracts) > self.config.option_batch_size:
            raise ValueError("OPTION_BATCH_TOO_LARGE")
        leases = []
        try:
            for contract in contracts:
                if contract.secType != "FOP":
                    raise ValueError("OPTION_QUOTES_REQUIRE_FOP")
                leases.append(
                    await self.acquire(contract, "QUOTE", owner, "selection", deadline=deadline)
                )
            yield leases
        finally:
            await asyncio.gather(*(self.release(sub, owner) for sub in leases))

    async def release_owner(self, owner: str) -> None:
        for sub in list(self.items.values()):
            if owner in sub.consumers:
                await self.release(sub, owner)

    def snapshot(self) -> dict[str, Any]:
        return {
            "owned_lines": self.count(),
            "app_budget": self.effective_cap,
            "configured_app_cap": self.config.app_line_cap,
            "total_account_allowance": self.config.total_lines,
            "allowance_status": self.config.allowance_status,
            "allowance_source": self.config.allowance_source,
            "verified_at": self.config.verified_at,
            "external_usage": self.config.known_external_lines,
            "external_headroom": self.config.external_headroom,
            "depth_used": self.count("DEPTH"),
            "depth_limit": self.config.depth_slots,
            "temporary_quotes": self.temporary_count(),
            "temporary_quote_limit": self.config.temporary_option_quotes,
            "tick_by_tick": 0,
            "history_inflight": self.history_active,
            "errors": list(self.errors),
            "subscriptions": [
                {
                    "con_id": s.contract.conId,
                    "feed": s.feed,
                    "state": s.state,
                    "generation": s.generation,
                    "consumers": s.consumers,
                    "last_receipt": s.last_receipt.isoformat() if s.last_receipt else None,
                    "last_change": s.last_change.isoformat() if s.last_change else None,
                    "error": s.error,
                }
                for s in self.items.values()
            ],
            "pacing": self.ib.client.snapshot() if hasattr(self.ib.client, "snapshot") else {},
        }

    async def close(self) -> None:
        for sub in list(self.items.values()):
            for owner in list(sub.consumers):
                await self.release(sub, owner)
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
