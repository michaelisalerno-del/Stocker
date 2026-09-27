"""Replace ib_async's single FIFO throttle with one bounded priority wire scheduler."""

import asyncio
import heapq
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from ib_async.client import Client

from stocker_execution.config import MarketDataConfig

URGENT, EXPOSURE, CORE, OPTIONAL = range(4)


@dataclass
class Traffic:
    priority: int = CORE
    deadline: float = float("inf")
    immediate: bool = False
    sent: list[asyncio.Future[None]] = field(default_factory=list)


traffic: ContextVar[Traffic | None] = ContextVar("slrno_traffic", default=None)


@contextmanager
def priority(
    level: int, deadline: float = float("inf"), immediate: bool = False
) -> Iterator[Traffic]:
    group = Traffic(level, deadline, immediate)
    token = traffic.set(group)
    try:
        yield group
    finally:
        # Abandoned/expired callers cannot leave unsent work in the wire queue.
        for future in group.sent:
            if not future.done():
                future.cancel()
        traffic.reset(token)


class PacedClient(Client):
    def __init__(self, wrapper: Any, config: MarketDataConfig):
        self.limits = config
        self.pending: list[tuple[int, int, str, Traffic, asyncio.Future[None]]] = []
        self.sent_times: deque[tuple[float, int]] = deque()
        self.serial = 0
        self.timer: asyncio.TimerHandle | None = None
        self.rejected = self.expired = self.high_water = 0
        self.connection_epoch = 0
        super().__init__(wrapper)  # type: ignore[no-untyped-call]

    def reset(self) -> None:
        self.connection_epoch += 1
        if self.timer:
            self.timer.cancel()
            self.timer = None
        for _, _, _, _, future in self.pending:
            if not future.done():
                future.cancel()
        self.pending.clear()
        self.sent_times.clear()
        super().reset()  # type: ignore[no-untyped-call]

    def available(self, level: int) -> bool:
        now = asyncio.get_event_loop().time()
        while self.sent_times and now - self.sent_times[0][0] >= 1:
            self.sent_times.popleft()
        ceiling = self.limits.request_budget
        if level > URGENT:
            ceiling -= self.limits.urgent_reserve
        if level == OPTIONAL and sum(p == OPTIONAL for _, p in self.sent_times) >= 5:
            return False
        return len(self.sent_times) < ceiling

    async def ready(self, level: int, deadline: float) -> None:
        # No queue of optional coroutines sits ahead of urgent work.
        while not self.available(level):
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("API_PACING_DEADLINE")
            await asyncio.sleep(0.02)

    def sendMsg(self, msg: str | None) -> None:
        if msg:
            group = traffic.get() or Traffic()
            if group.immediate:
                if (
                    not self.available(group.priority)
                    or asyncio.get_event_loop().time() >= group.deadline
                ):
                    raise TimeoutError("API_PACING_DEADLINE")
                self._write(msg, group.priority)
                return
            limit = self.limits.queue_size * (8, 7, 6, 4)[group.priority] // 8
            if len(self.pending) >= limit:
                self.rejected += 1
                raise ValueError("API_QUEUE_FULL")
            future = asyncio.get_event_loop().create_future()
            # Retrieve failure even for native library calls without an explicit drain waiter.
            future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
            group.sent.append(future)
            self.serial += 1
            heapq.heappush(self.pending, (group.priority, self.serial, msg, group, future))
            self.high_water = max(self.high_water, len(self.pending))
        self._drain()

    def _write(self, msg: str, level: int) -> None:
        self.conn.sendMsg(self._prefix(msg.encode()))  # type: ignore[no-untyped-call]
        self.sent_times.append((asyncio.get_event_loop().time(), level))

    def _drain(self) -> None:
        if self.timer:
            self.timer.cancel()
            self.timer = None
        while self.pending:
            level, _, msg, group, future = self.pending[0]
            if future.cancelled() or asyncio.get_event_loop().time() >= group.deadline:
                heapq.heappop(self.pending)
                if not future.done():
                    future.set_exception(TimeoutError("API_PACING_DEADLINE"))
                    self.expired += 1
                continue
            if not self.available(level):
                break
            heapq.heappop(self.pending)
            try:
                self._write(msg, level)
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)
        if self.pending:
            self.timer = asyncio.get_event_loop().call_later(0.02, self._drain)

    def snapshot(self) -> dict[str, Any]:
        self.available(URGENT)
        return {
            "outbound_last_second": len(self.sent_times),
            "outbound_cap": self.limits.request_budget,
            "urgent_reserve": self.limits.urgent_reserve,
            "queued": len(self.pending),
            "queue_high_water": self.high_water,
            "expired": self.expired,
            "rejected": self.rejected,
        }


async def drain(group: Traffic) -> None:
    try:
        if group.sent:
            await asyncio.gather(*group.sent)
    finally:
        for future in group.sent:
            if not future.done():
                future.cancel()
