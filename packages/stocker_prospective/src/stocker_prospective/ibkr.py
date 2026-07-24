"""Optional official IBKR market-data adapter.

This module deliberately exposes no trading or account methods. The official
``ibapi`` client is imported only when an IBKR recorder is configured, keeping
CI and replay independent from the optional dependency.
"""

from __future__ import annotations

import importlib.util
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from stocker_prospective.market_data import (
    BoundedCallbackRegistry,
    BoundedRealtimeBarQueue,
    BoundedStreamQuoteCache,
    CallbackResult,
    ConnectionTracker,
    MarketDataBudget,
    MarketDataType,
    RealtimeBarUpdate,
    RequestIdAllocator,
    SubscriptionRegistry,
    classify_ibkr_error,
)

IBKR_DEPENDENCY_BLOCKER = "blocked_official_ibkr_api_not_installed"


class OfficialIBKRDependencyError(RuntimeError):
    """The operator has not installed the official IBKR Python client."""


@dataclass(frozen=True)
class IBKRConnectionConfig:
    host: str
    port: int
    client_id: int
    expected_environment: str
    connect_timeout_seconds: float
    request_timeout_seconds: float
    quote_capture_timeout_seconds: float
    allowed_market_data_types: tuple[MarketDataType, ...]

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("IBKR host is required")
        if not 1 <= self.port <= 65535:
            raise ValueError("IBKR port must be explicitly configured")
        if self.client_id < 0:
            raise ValueError("IBKR client_id must be nonnegative")
        if self.expected_environment not in {"paper", "live_read_only"}:
            raise ValueError("expected_environment must be paper or live_read_only")
        if not self.allowed_market_data_types:
            raise ValueError("at least one market-data type must be allowed")


def official_ibkr_api_available() -> bool:
    return importlib.util.find_spec("ibapi") is not None


def require_official_ibkr_api() -> ModuleType:
    if not official_ibkr_api_available():
        raise OfficialIBKRDependencyError(
            f"{IBKR_DEPENDENCY_BLOCKER}: install the official TWS API Python client "
            "from the IBKR Latest Mac/Unix distribution"
        )
    module = __import__("ibapi")
    if not isinstance(module, ModuleType):
        raise OfficialIBKRDependencyError(f"{IBKR_DEPENDENCY_BLOCKER}: invalid ibapi module")
    return module


class IBKRMarketDataAdapter:
    """Narrow event-loop owner for exact market-data requests only."""

    def __init__(
        self,
        *,
        config: IBKRConnectionConfig,
        budget: MarketDataBudget,
    ) -> None:
        self.config = config
        self.budget = budget
        self.request_ids = RequestIdAllocator()
        self.connection = ConnectionTracker()
        self.callbacks = BoundedCallbackRegistry(
            max_pending_requests=256,
            max_items_per_request=256,
        )
        self.stream_quotes = BoundedStreamQuoteCache(
            max_subscriptions=256,
            max_fields_per_subscription=64,
        )
        self.realtime_bars = BoundedRealtimeBarQueue(max_items=4096)
        self.subscriptions = SubscriptionRegistry()
        self._subscription_kinds: dict[int, str] = {}
        self._loop_thread: threading.Thread | None = None
        self._client: Any | None = None
        self._stopping = threading.Event()
        self._connected = threading.Event()

    @property
    def dependency_blocker(self) -> str | None:
        return None if official_ibkr_api_available() else IBKR_DEPENDENCY_BLOCKER

    def attach_official_client(self, client: Any) -> None:
        """Attach a configured official EClient wrapper before ``start``.

        The wrapper is injected because the official API uses callback
        inheritance. Stocker never adds order callbacks or account routing.
        """

        require_official_ibkr_api()
        required = ("connect", "disconnect", "run", "reqMktData", "cancelMktData")
        missing = [name for name in required if not callable(getattr(client, name, None))]
        if missing:
            raise TypeError(f"official client missing market-data methods: {missing}")
        forbidden = ("placeOrder", "cancelOrder", "reqOpenOrders", "reqGlobalCancel")
        if any(callable(getattr(client, name, None)) for name in forbidden):
            raise TypeError("order-capable wrapper cannot be attached to the recorder")
        self._client = client

    def start(self) -> None:
        require_official_ibkr_api()
        if self._client is None:
            raise RuntimeError("official IBKR callback client has not been attached")
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self.connection.connecting()
        self._connected.clear()
        connection_result = self._client.connect(
            self.config.host,
            self.config.port,
            self.config.client_id,
        )
        if connection_result is False:
            self.connection.degraded(
                code=-1,
                message="blocked_ibkr_connection",
            )
            raise RuntimeError("blocked_ibkr_connection")
        self._stopping.clear()
        self._loop_thread = threading.Thread(
            target=self._client.run,
            name="stocker-ibkr-callback-loop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._connected.wait(self.config.connect_timeout_seconds):
            self.connection.degraded(
                code=-1,
                message="blocked_ibkr_connection: handshake_timeout",
            )
            self.stop()
            raise RuntimeError("blocked_ibkr_connection: handshake_timeout")

    def stop(self) -> None:
        self.connection.shutting_down()
        self._stopping.set()
        self.callbacks.shutdown()
        self.stream_quotes.clear()
        for key in self.budget.shutdown():
            if key.isdigit():
                self._cancel_upstream(int(key))
        self._subscription_kinds.clear()
        if self._client is not None:
            self._client.disconnect()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=self.config.connect_timeout_seconds)
        self._loop_thread = None
        self._connected.clear()

    def reconnect(self) -> None:
        """Reconnect the socket after confirmed loss; subscriptions rebuild above."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        self._connected.clear()
        try:
            self._client.disconnect()
        finally:
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=self.config.connect_timeout_seconds)
            self._loop_thread = None
        self.connection.connecting()
        connection_result = self._client.connect(
            self.config.host,
            self.config.port,
            self.config.client_id,
        )
        if connection_result is False:
            self.connection.degraded(code=-1, message="blocked_ibkr_connection")
            raise RuntimeError("blocked_ibkr_connection")
        self._loop_thread = threading.Thread(
            target=self._client.run,
            name="stocker-ibkr-callback-loop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._connected.wait(self.config.connect_timeout_seconds):
            self.connection.degraded(
                code=-1,
                message="blocked_ibkr_connection: reconnect_handshake_timeout",
            )
            raise RuntimeError("blocked_ibkr_connection: reconnect_handshake_timeout")

    def request_market_data(
        self,
        contract: Any,
        *,
        subscription_key: str,
        generic_ticks: str = "",
    ) -> int:
        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        request_id = self.request_ids.next()
        key = str(request_id)
        self.budget.reserve(key)
        self.stream_quotes.register(request_id)
        if not self.subscriptions.register(subscription_key, request_id):
            self.stream_quotes.remove(request_id)
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is not None:
                self.subscriptions.register(subscription_key, existing)
                return existing
        try:
            self._subscription_kinds[request_id] = "market_data"
            self._client.reqMktData(
                request_id,
                contract,
                generic_ticks,
                False,
                False,
                [],
            )
            self.budget.mark_active(key)
        except Exception:
            self.budget.confirm_cancellation(key)
            self.stream_quotes.remove(request_id)
            self.subscriptions.remove(subscription_key)
            self._subscription_kinds.pop(request_id, None)
            raise
        return request_id

    def cancel_market_data(
        self,
        request_id: int,
        *,
        subscription_key: str | None = None,
    ) -> None:
        key = str(request_id)
        if not self.budget.request_cancellation(key):
            return
        self._cancel_upstream(request_id)
        self.budget.confirm_cancellation(key)
        self.stream_quotes.remove(request_id)
        self._subscription_kinds.pop(request_id, None)
        if subscription_key is not None:
            self.subscriptions.remove(subscription_key)

    def request_realtime_bars(
        self,
        contract: Any,
        *,
        subscription_key: str,
    ) -> int:
        """Request bounded 5-second RTH trade bars for diagnostic aggregation."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqRealTimeBars", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id = self.request_ids.next()
        key = str(request_id)
        self.budget.reserve(key)
        if not self.subscriptions.register(subscription_key, request_id):
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is not None:
                self.subscriptions.register(subscription_key, existing)
                return existing
        try:
            self._subscription_kinds[request_id] = "realtime_bars"
            method(
                request_id,
                contract,
                5,
                "TRADES",
                True,
                [],
            )
            self.budget.mark_active(key)
        except Exception:
            self.budget.confirm_cancellation(key)
            self.subscriptions.remove(subscription_key)
            self._subscription_kinds.pop(request_id, None)
            raise
        return request_id

    def cancel_realtime_bars(
        self,
        request_id: int,
        *,
        subscription_key: str | None = None,
    ) -> None:
        key = str(request_id)
        if not self.budget.request_cancellation(key):
            return
        self._cancel_upstream(request_id)
        self.budget.confirm_cancellation(key)
        self._subscription_kinds.pop(request_id, None)
        if subscription_key is not None:
            self.subscriptions.remove(subscription_key)

    def _cancel_upstream(self, request_id: int) -> None:
        if self._client is None:
            return
        kind = self._subscription_kinds.get(request_id)
        if kind == "realtime_bars":
            method = getattr(self._client, "cancelRealTimeBars", None)
            if callable(method):
                method(request_id)
            return
        self._client.cancelMktData(request_id)

    def capture_temporary_quote(
        self,
        *,
        contract: Any,
        timeout_seconds: float | None = None,
        generic_ticks: str = "",
    ) -> CallbackResult:
        """Capture one official snapshot and always cancel local capacity.

        IBKR snapshots complete through ``tickSnapshotEnd``. Generic ticks are
        deliberately unavailable in snapshot mode, so volume/open-interest
        remain missing unless present in the standard callbacks.
        """

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        if generic_ticks:
            raise ValueError("generic ticks are unavailable for bounded snapshot captures")
        request_id = self.request_ids.next()
        key = str(request_id)
        self.budget.reserve(key)
        self.callbacks.begin(request_id, kind="temporary_quote")
        try:
            try:
                self._client.reqMktData(
                    request_id,
                    contract,
                    generic_ticks,
                    True,
                    False,
                    [],
                )
            except Exception:
                self.callbacks.abort(request_id, "upstream_quote_request_failed")
                raise
            self.budget.mark_active(key)
            return self.callbacks.wait(
                request_id,
                timeout_seconds=(
                    self.config.quote_capture_timeout_seconds
                    if timeout_seconds is None
                    else timeout_seconds
                ),
            )
        finally:
            self.budget.request_cancellation(key)
            self._client.cancelMktData(request_id)
            self.budget.confirm_cancellation(key)

    def request_option_chain_metadata(
        self,
        *,
        underlying_symbol: str,
        exchange: str,
        underlying_security_type: str,
        underlying_contract_id: int,
    ) -> CallbackResult:
        """Call official ``reqSecDefOptParams``; never stream every contract."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqSecDefOptParams", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id = self.request_ids.next()
        budget_key = f"metadata:{request_id}"
        self.budget.reserve(budget_key, lines=0)
        try:
            self.callbacks.begin(request_id, kind="option_chain_metadata")
            try:
                method(
                    request_id,
                    underlying_symbol,
                    exchange,
                    underlying_security_type,
                    underlying_contract_id,
                )
            except Exception:
                self.callbacks.abort(request_id, "upstream_metadata_request_failed")
                raise
            return self.callbacks.wait(
                request_id,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        finally:
            self.budget.confirm_cancellation(budget_key)

    def qualify_exact_contract(self, contract: Any) -> CallbackResult:
        """Qualify one exact bounded contract through ``reqContractDetails``."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqContractDetails", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id = self.request_ids.next()
        budget_key = f"qualification:{request_id}"
        self.budget.reserve(budget_key, lines=0)
        try:
            self.callbacks.begin(request_id, kind="exact_contract_qualification")
            try:
                method(request_id, contract)
            except Exception:
                self.callbacks.abort(request_id, "upstream_contract_request_failed")
                raise
            return self.callbacks.wait(
                request_id,
                timeout_seconds=self.config.request_timeout_seconds,
            )
        finally:
            self.budget.confirm_cancellation(budget_key)

    def on_option_parameter(self, request_id: int, payload: Any) -> None:
        self.callbacks.add(request_id, payload)

    def on_option_parameter_end(self, request_id: int) -> None:
        self.callbacks.complete(request_id)

    def on_contract_details(self, request_id: int, payload: Any) -> None:
        self.callbacks.add(request_id, payload)

    def on_contract_details_end(self, request_id: int) -> None:
        self.callbacks.complete(request_id)

    def on_quote_update(
        self,
        request_id: int,
        payload: dict[str, Any],
        *,
        complete: bool = False,
    ) -> None:
        """Receive a quote update without converting missing fields to zero."""

        if self.callbacks.is_pending(request_id):
            self.callbacks.add(request_id, payload)
            if complete:
                self.callbacks.complete(request_id)
            return
        self.stream_quotes.add(request_id, payload)

    def on_connected(self, market_data_type: MarketDataType | None) -> None:
        self.connection.connected(market_data_type)
        self._connected.set()

    def on_realtime_bar(self, update: RealtimeBarUpdate) -> None:
        self.realtime_bars.add(update)

    def _clear_lost_subscriptions(self) -> None:
        self.callbacks.shutdown()
        for _, request_id in self.subscriptions.active_items():
            self.budget.confirm_cancellation(str(request_id))
            self.stream_quotes.remove(request_id)
            self._subscription_kinds.pop(request_id, None)
        self.subscriptions.after_reconnect(data_maintained=False)

    def on_connection_closed(self) -> None:
        self._connected.clear()
        self.connection.connection_lost(
            code=1100,
            message="official_socket_connection_closed",
        )
        self._clear_lost_subscriptions()

    def on_error(self, request_id: int, code: int, message: str) -> None:
        if code == 1100:
            self._connected.clear()
            self.connection.connection_lost(code=code, message=message)
            return
        if code == 1101:
            self.connection.connection_restored(data_maintained=False, code=code)
            self._clear_lost_subscriptions()
            self._connected.set()
            return
        if code == 1102:
            self.connection.connection_restored(data_maintained=True, code=code)
            self._connected.set()
            return
        if code == 1300:
            self._connected.clear()
            self.connection.socket_port_reset(self.config.port)
            self._clear_lost_subscriptions()
            return
        reason = classify_ibkr_error(code)
        self.connection.degraded(code=code, message=f"{reason}:{message}")
        try:
            self.callbacks.fail(request_id, reason)
        except Exception:
            return
