"""Market-data-only primitives shared by replay and the optional IBKR adapter."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class MarketDataType(StrEnum):
    """IBKR market-data modes, kept explicit in every quote record."""

    LIVE = "live"
    FROZEN = "frozen"
    DELAYED = "delayed"
    DELAYED_FROZEN = "delayed_frozen"

    @property
    def primary_eligible(self) -> bool:
        return self is MarketDataType.LIVE


class RequestIdAllocator:
    """Thread-safe request identifier allocator compatible with IBKR ``nextValidId``."""

    def __init__(self, *, start: int = 1) -> None:
        if start < 0:
            raise ValueError("request IDs must be nonnegative")
        self._next = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            request_id = self._next
            self._next += 1
            return request_id

    def synchronise(self, server_next_id: int) -> None:
        if server_next_id < 0:
            raise ValueError("server request ID must be nonnegative")
        with self._lock:
            self._next = max(self._next, server_next_id)


class CallbackRequestError(RuntimeError):
    """A bounded request failed, timed out, or was interrupted."""


@dataclass(frozen=True)
class CallbackResult:
    request_id: int
    kind: str
    items: tuple[Any, ...]
    complete: bool
    error: str | None


@dataclass
class _PendingCallback:
    kind: str
    event: threading.Event
    items: list[Any]
    complete: bool = False
    error: str | None = None


class BoundedCallbackRegistry:
    """Correlate asynchronous callbacks without an unbounded pending queue."""

    def __init__(
        self,
        *,
        max_pending_requests: int,
        max_items_per_request: int,
        max_finished_requests: int | None = None,
    ) -> None:
        if max_pending_requests <= 0 or max_items_per_request <= 0:
            raise ValueError("callback bounds must be positive")
        finished_bound = (
            max_pending_requests if max_finished_requests is None else max_finished_requests
        )
        if finished_bound <= 0:
            raise ValueError("finished callback bound must be positive")
        self._max_pending = max_pending_requests
        self._max_items = max_items_per_request
        self._max_finished = finished_bound
        self._pending: dict[int, _PendingCallback] = {}
        self._finished: OrderedDict[int, CallbackResult] = OrderedDict()
        self._lock = threading.RLock()

    def begin(self, request_id: int, *, kind: str) -> None:
        with self._lock:
            if request_id in self._pending or request_id in self._finished:
                raise CallbackRequestError("duplicate_callback_request_id")
            if len(self._pending) >= self._max_pending:
                raise CallbackRequestError("bounded_pending_request_queue_exhausted")
            self._pending[request_id] = _PendingCallback(
                kind=kind,
                event=threading.Event(),
                items=[],
            )

    def is_pending(self, request_id: int) -> bool:
        with self._lock:
            return request_id in self._pending

    def add(self, request_id: int, item: Any) -> None:
        with self._lock:
            pending = self._require_pending(request_id)
            if len(pending.items) >= self._max_items:
                pending.error = "bounded_callback_queue_exhausted"
                pending.event.set()
                raise CallbackRequestError(pending.error)
            pending.items.append(item)

    def complete(self, request_id: int) -> None:
        with self._lock:
            pending = self._require_pending(request_id)
            pending.complete = True
            pending.event.set()

    def fail(self, request_id: int, reason: str) -> None:
        with self._lock:
            pending = self._require_pending(request_id)
            pending.error = reason
            pending.event.set()

    def abort(self, request_id: int, reason: str) -> None:
        """Remove a request whose upstream call failed before callbacks could run."""

        with self._lock:
            pending = self._require_pending(request_id)
            result = CallbackResult(
                request_id=request_id,
                kind=pending.kind,
                items=tuple(pending.items),
                complete=False,
                error=reason,
            )
            self._pending.pop(request_id)
            self._store_finished(result)
            pending.event.set()

    def wait(self, request_id: int, *, timeout_seconds: float) -> CallbackResult:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be nonnegative")
        with self._lock:
            finished = self._finished.get(request_id)
            if finished is not None:
                if finished.error is not None:
                    raise CallbackRequestError(finished.error)
                return finished
            pending = self._require_pending(request_id)
            event = pending.event
        signalled = event.wait(timeout_seconds)
        with self._lock:
            current = self._pending.get(request_id)
            if current is None:
                finished = self._finished[request_id]
            else:
                if not signalled and current.error is None:
                    current.error = "incomplete_callback_timeout"
                finished = CallbackResult(
                    request_id=request_id,
                    kind=current.kind,
                    items=tuple(current.items),
                    complete=current.complete and current.error is None,
                    error=current.error,
                )
                self._store_finished(finished)
                self._pending.pop(request_id, None)
        if finished.error is not None:
            raise CallbackRequestError(finished.error)
        return finished

    def shutdown(self) -> tuple[int, ...]:
        with self._lock:
            request_ids = tuple(self._pending)
            for request_id, pending in tuple(self._pending.items()):
                result = CallbackResult(
                    request_id=request_id,
                    kind=pending.kind,
                    items=tuple(pending.items),
                    complete=False,
                    error="shutdown_during_pending_request",
                )
                self._store_finished(result)
                pending.event.set()
            self._pending.clear()
            return request_ids

    def _store_finished(self, result: CallbackResult) -> None:
        self._finished[result.request_id] = result
        self._finished.move_to_end(result.request_id)
        while len(self._finished) > self._max_finished:
            self._finished.popitem(last=False)

    def _require_pending(self, request_id: int) -> _PendingCallback:
        pending = self._pending.get(request_id)
        if pending is None:
            raise CallbackRequestError("unknown_or_finished_callback_request")
        return pending


class BoundedStreamQuoteCache:
    """Keep only the latest fields for a bounded set of streaming requests."""

    def __init__(self, *, max_subscriptions: int, max_fields_per_subscription: int) -> None:
        if max_subscriptions <= 0 or max_fields_per_subscription <= 0:
            raise ValueError("stream cache bounds must be positive")
        self._max_subscriptions = max_subscriptions
        self._max_fields = max_fields_per_subscription
        self._quotes: dict[int, OrderedDict[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def register(self, request_id: int) -> None:
        with self._lock:
            if request_id in self._quotes:
                return
            if len(self._quotes) >= self._max_subscriptions:
                raise CallbackRequestError("bounded_stream_subscription_cache_exhausted")
            self._quotes[request_id] = OrderedDict()

    def add(self, request_id: int, payload: dict[str, Any]) -> None:
        with self._lock:
            fields = self._quotes.get(request_id)
            if fields is None:
                raise CallbackRequestError("unknown_stream_subscription")
            field = str(payload.get("field", "unknown"))
            source = payload.get("computation_source")
            key = field if source is None else f"{field}:{source}"
            if key not in fields and len(fields) >= self._max_fields:
                raise CallbackRequestError("bounded_stream_field_cache_exhausted")
            fields[key] = dict(payload)
            fields.move_to_end(key)

    def snapshot(self, request_id: int) -> tuple[dict[str, Any], ...]:
        with self._lock:
            fields = self._quotes.get(request_id)
            if fields is None:
                raise CallbackRequestError("unknown_stream_subscription")
            return tuple(dict(payload) for payload in fields.values())

    def remove(self, request_id: int) -> bool:
        with self._lock:
            return self._quotes.pop(request_id, None) is not None

    def clear(self) -> tuple[int, ...]:
        with self._lock:
            request_ids = tuple(self._quotes)
            self._quotes.clear()
            return request_ids


@dataclass(frozen=True)
class RealtimeBarUpdate:
    request_id: int
    source_timestamp_utc: datetime
    receive_timestamp_utc: datetime
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    wap: float | None
    trade_count: int | None


class BoundedRealtimeBarQueue:
    """Bound realtime-bar callbacks and make overflow explicit."""

    def __init__(self, *, max_items: int) -> None:
        if max_items <= 0:
            raise ValueError("realtime bar queue bound must be positive")
        self._max_items = max_items
        self._items: deque[RealtimeBarUpdate] = deque()
        self._lock = threading.RLock()

    def add(self, update: RealtimeBarUpdate) -> None:
        with self._lock:
            if len(self._items) >= self._max_items:
                raise CallbackRequestError("bounded_realtime_bar_queue_exhausted")
            self._items.append(update)

    def drain(self, *, limit: int | None = None) -> tuple[RealtimeBarUpdate, ...]:
        if limit is not None and limit <= 0:
            raise ValueError("drain limit must be positive")
        with self._lock:
            count = len(self._items) if limit is None else min(limit, len(self._items))
            return tuple(self._items.popleft() for _ in range(count))

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)


class SubscriptionRegistry:
    """Track active subscriptions and deterministic reconnect rebuilds."""

    def __init__(self) -> None:
        self._active: dict[str, int] = {}
        self._lock = threading.RLock()

    def register(self, key: str, request_id: int) -> bool:
        with self._lock:
            if key in self._active:
                return False
            self._active[key] = request_id
            return True

    def remove(self, key: str) -> int | None:
        with self._lock:
            return self._active.pop(key, None)

    def after_reconnect(self, *, data_maintained: bool) -> tuple[str, ...]:
        with self._lock:
            if data_maintained:
                return ()
            keys = tuple(self._active)
            self._active.clear()
            return keys

    def active_items(self) -> tuple[tuple[str, int], ...]:
        with self._lock:
            return tuple(self._active.items())

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)


def classify_ibkr_error(code: int) -> str:
    """Map official market-data failures into stable, auditable blockers."""

    return {
        100: "blocked_ibkr_market_data_subscription:pacing_error",
        101: "blocked_market_data_budget_exhausted",
        326: "blocked_ibkr_connection:client_id_in_use",
        354: "blocked_ibkr_market_data_subscription:missing_subscription",
        502: "blocked_ibkr_connection:socket_or_port",
        10089: "blocked_ibkr_market_data_subscription:partial_subscription",
        10090: "blocked_ibkr_market_data_subscription:partial_subscription",
        10186: "blocked_ibkr_market_data_subscription:missing_subscription",
        10197: "blocked_ibkr_market_data_subscription:competing_session",
    }.get(code, f"blocked_ibkr_market_data_subscription:ibkr_error_{code}")


class MarketDataBudgetError(RuntimeError):
    """Raised when a request would violate a configured market-data bound."""


@dataclass(frozen=True)
class MarketDataBudgetSnapshot:
    line_limit: int
    reserved_headroom: int
    usable_lines: int
    active_lines: int
    pending_requests: int
    awaiting_cancellation: int
    current_request_rate: int
    waiting_signals: int
    rejected_signals: int


@dataclass
class _Reservation:
    lines: int
    status: str


class MarketDataBudget:
    """Strict line and request-rate budget for all subscriptions.

    A duplicate key is idempotent. Capacity reserved as headroom can never be
    consumed by normal requests.
    """

    def __init__(
        self,
        *,
        line_limit: int,
        reserved_headroom: int,
        request_rate_limit: int,
        request_rate_window_seconds: float = 1.0,
        max_waiting_signals: int = 0,
    ) -> None:
        if line_limit <= 0:
            raise ValueError("line_limit must be positive")
        if reserved_headroom < 0 or reserved_headroom >= line_limit:
            raise ValueError("reserved_headroom must be in [0, line_limit)")
        if request_rate_limit <= 0 or request_rate_window_seconds <= 0:
            raise ValueError("request rate bounds must be positive")
        if max_waiting_signals < 0:
            raise ValueError("max_waiting_signals must be nonnegative")
        self._line_limit = line_limit
        self._reserved_headroom = reserved_headroom
        self._request_rate_limit = request_rate_limit
        self._request_rate_window_seconds = request_rate_window_seconds
        self._max_waiting_signals = max_waiting_signals
        self._reservations: dict[str, _Reservation] = {}
        self._request_times: deque[float] = deque()
        self._waiting_signals = 0
        self._rejected_signals = 0
        self._lock = threading.RLock()

    def reserve(self, key: str, *, lines: int = 1, now: float | None = None) -> None:
        if not key:
            raise ValueError("reservation key is required")
        if lines < 0:
            raise ValueError("lines must be nonnegative")
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            existing = self._reservations.get(key)
            if existing is not None:
                if existing.lines != lines:
                    raise MarketDataBudgetError("duplicate reservation has a different line count")
                return
            self._trim_rate_window(timestamp)
            if len(self._request_times) >= self._request_rate_limit:
                self._rejected_signals += 1
                raise MarketDataBudgetError("blocked_market_data_budget_exhausted: request_rate")
            usable = self._line_limit - self._reserved_headroom
            if self._active_lines() + lines > usable:
                self._rejected_signals += 1
                raise MarketDataBudgetError(
                    "blocked_market_data_budget_exhausted: market_data_lines"
                )
            self._request_times.append(timestamp)
            self._reservations[key] = _Reservation(lines=lines, status="pending")

    def mark_active(self, key: str) -> None:
        with self._lock:
            reservation = self._reservations.get(key)
            if reservation is None:
                raise KeyError(key)
            reservation.status = "active"

    def request_cancellation(self, key: str) -> bool:
        with self._lock:
            reservation = self._reservations.get(key)
            if reservation is None:
                return False
            reservation.status = "awaiting_cancellation"
            return True

    def confirm_cancellation(self, key: str) -> bool:
        with self._lock:
            return self._reservations.pop(key, None) is not None

    def note_waiting_signal(self) -> bool:
        with self._lock:
            if self._waiting_signals >= self._max_waiting_signals:
                self._rejected_signals += 1
                return False
            self._waiting_signals += 1
            return True

    def resolve_waiting_signal(self) -> None:
        with self._lock:
            self._waiting_signals = max(0, self._waiting_signals - 1)

    def shutdown(self) -> tuple[str, ...]:
        """Cancel every local reservation and return the keys to cancel upstream."""

        with self._lock:
            keys = tuple(self._reservations)
            self._reservations.clear()
            self._waiting_signals = 0
            return keys

    def snapshot(self, *, now: float | None = None) -> MarketDataBudgetSnapshot:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            self._trim_rate_window(timestamp)
            return MarketDataBudgetSnapshot(
                line_limit=self._line_limit,
                reserved_headroom=self._reserved_headroom,
                usable_lines=self._line_limit - self._reserved_headroom,
                active_lines=self._active_lines(),
                pending_requests=sum(
                    reservation.status == "pending" for reservation in self._reservations.values()
                ),
                awaiting_cancellation=sum(
                    reservation.status == "awaiting_cancellation"
                    for reservation in self._reservations.values()
                ),
                current_request_rate=len(self._request_times),
                waiting_signals=self._waiting_signals,
                rejected_signals=self._rejected_signals,
            )

    def _active_lines(self) -> int:
        return sum(reservation.lines for reservation in self._reservations.values())

    def _trim_rate_window(self, now: float) -> None:
        cutoff = now - self._request_rate_window_seconds
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()


class ConnectionState(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    PORT_RESET = "socket_port_mismatch_or_reset"
    SHUTTING_DOWN = "shutting_down"


@dataclass(frozen=True)
class ConnectionEvent:
    recorded_at: datetime
    state: ConnectionState
    code: int | None
    message: str
    data_maintained: bool | None


@dataclass(frozen=True)
class ConnectionHealth:
    state: ConnectionState
    market_data_type: MarketDataType | None
    subscriptions_require_rebuild: bool
    last_error_code: int | None
    last_message: str


class ConnectionTracker:
    """Deterministic interpretation of the official IBKR connectivity events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = ConnectionState.DISCONNECTED
        self._market_data_type: MarketDataType | None = None
        self._requires_rebuild = False
        self._last_error_code: int | None = None
        self._last_message = "not_connected"
        self.events: list[ConnectionEvent] = []

    def connecting(self) -> None:
        with self._lock:
            self._record(ConnectionState.CONNECTING, None, "connecting", None)

    def connected(self, market_data_type: MarketDataType | None) -> None:
        with self._lock:
            self._market_data_type = market_data_type
            self._requires_rebuild = False
            self._record(ConnectionState.CONNECTED, None, "connected", None)

    def subscriptions_rebuilt(self) -> None:
        with self._lock:
            self._requires_rebuild = False

    def connection_lost(self, *, code: int, message: str) -> None:
        with self._lock:
            self._record(ConnectionState.DISCONNECTED, code, message, False)

    def connection_restored(self, *, data_maintained: bool, code: int) -> None:
        with self._lock:
            self._requires_rebuild = not data_maintained
            message = (
                "connectivity_restored_data_maintained"
                if data_maintained
                else "connectivity_restored_data_lost"
            )
            self._record(ConnectionState.CONNECTED, code, message, data_maintained)

    def socket_port_reset(self, port: int) -> None:
        with self._lock:
            self._requires_rebuild = True
            self._record(
                ConnectionState.PORT_RESET,
                1300,
                f"socket_port_mismatch_or_reset:{port}",
                False,
            )

    def degraded(self, *, code: int, message: str) -> None:
        with self._lock:
            self._record(ConnectionState.DEGRADED, code, message, None)

    def shutting_down(self) -> None:
        with self._lock:
            self._record(ConnectionState.SHUTTING_DOWN, None, "shutting_down", None)

    def health(self) -> ConnectionHealth:
        with self._lock:
            return ConnectionHealth(
                state=self._state,
                market_data_type=self._market_data_type,
                subscriptions_require_rebuild=self._requires_rebuild,
                last_error_code=self._last_error_code,
                last_message=self._last_message,
            )

    def drain_events(self) -> tuple[ConnectionEvent, ...]:
        """Return and clear events after the recorder persists them."""

        with self._lock:
            events = tuple(self.events)
            self.events.clear()
            return events

    def _record(
        self,
        state: ConnectionState,
        code: int | None,
        message: str,
        data_maintained: bool | None,
    ) -> None:
        self._state = state
        self._last_error_code = code
        self._last_message = message
        self.events.append(
            ConnectionEvent(
                recorded_at=datetime.now(UTC),
                state=state,
                code=code,
                message=message,
                data_maintained=data_maintained,
            )
        )
