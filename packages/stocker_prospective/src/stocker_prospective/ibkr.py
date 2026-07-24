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
    CallbackResult,
    ConnectionTracker,
    MarketDataBudget,
    MarketDataType,
    RequestIdAllocator,
    SubscriptionRegistry,
    classify_ibkr_error,
)

IBKR_DEPENDENCY_BLOCKER = "blocked_official_ibkr_api_not_installed"


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
        raise RuntimeError(
            f"{IBKR_DEPENDENCY_BLOCKER}: install the official TWS API Python client "
            "from the IBKR Latest Mac/Unix distribution"
        )
    module = __import__("ibapi")
    if not isinstance(module, ModuleType):
        raise RuntimeError(f"{IBKR_DEPENDENCY_BLOCKER}: invalid ibapi module")
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
        self.subscriptions = SubscriptionRegistry()
        self._loop_thread: threading.Thread | None = None
        self._client: Any | None = None
        self._stopping = threading.Event()

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
        forbidden = ("placeOrder", "cancelOrder", "reqOpenOrders")
        if any(name in client.__class__.__dict__ for name in forbidden):
            raise TypeError("order-capable wrapper cannot be attached to the recorder")
        self._client = client

    def start(self) -> None:
        require_official_ibkr_api()
        if self._client is None:
            raise RuntimeError("official IBKR callback client has not been attached")
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self.connection.connecting()
        connected = bool(
            self._client.connect(
                self.config.host,
                self.config.port,
                self.config.client_id,
            )
        )
        if not connected:
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

    def stop(self) -> None:
        self.connection.shutting_down()
        self._stopping.set()
        self.callbacks.shutdown()
        for key in self.budget.shutdown():
            if self._client is not None and key.isdigit():
                self._client.cancelMktData(int(key))
        if self._client is not None:
            self._client.disconnect()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=self.config.connect_timeout_seconds)
        self._loop_thread = None

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
        if not self.subscriptions.register(subscription_key, request_id):
            self.budget.confirm_cancellation(key)
            existing = self.subscriptions.remove(subscription_key)
            if existing is not None:
                self.subscriptions.register(subscription_key, existing)
                return existing
        try:
            self._client.reqMktData(
                request_id,
                contract,
                generic_ticks,
                False,
                False,
                [],
            )
        except Exception:
            self.budget.confirm_cancellation(key)
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
        if self._client is not None:
            self._client.cancelMktData(request_id)
        self.budget.confirm_cancellation(key)
        if subscription_key is not None:
            self.subscriptions.remove(subscription_key)

    def capture_temporary_quote(
        self,
        *,
        contract: Any,
        timeout_seconds: float | None = None,
        generic_ticks: str = "100,101,104,106",
    ) -> CallbackResult:
        """Capture a bounded temporary stream and always cancel it."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        request_id = self.request_ids.next()
        key = str(request_id)
        self.budget.reserve(key)
        self.callbacks.begin(request_id, kind="temporary_quote")
        try:
            self._client.reqMktData(
                request_id,
                contract,
                generic_ticks,
                False,
                False,
                [],
            )
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
        self.callbacks.begin(request_id, kind="option_chain_metadata")
        method(
            request_id,
            underlying_symbol,
            exchange,
            underlying_security_type,
            underlying_contract_id,
        )
        return self.callbacks.wait(
            request_id,
            timeout_seconds=self.config.request_timeout_seconds,
        )

    def qualify_exact_contract(self, contract: Any) -> CallbackResult:
        """Qualify one exact bounded contract through ``reqContractDetails``."""

        if self._client is None:
            raise RuntimeError("blocked_ibkr_connection")
        method = getattr(self._client, "reqContractDetails", None)
        if not callable(method):
            raise RuntimeError("blocked_ibkr_market_data_subscription")
        request_id = self.request_ids.next()
        self.callbacks.begin(request_id, kind="exact_contract_qualification")
        method(request_id, contract)
        return self.callbacks.wait(
            request_id,
            timeout_seconds=self.config.request_timeout_seconds,
        )

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

        self.callbacks.add(request_id, payload)
        if complete:
            self.callbacks.complete(request_id)

    def on_error(self, request_id: int, code: int, message: str) -> None:
        if code == 1100:
            self.connection.connection_lost(code=code, message=message)
            return
        if code == 1101:
            self.connection.connection_restored(data_maintained=False, code=code)
            self.subscriptions.after_reconnect(data_maintained=False)
            return
        if code == 1102:
            self.connection.connection_restored(data_maintained=True, code=code)
            return
        if code == 1300:
            self.connection.socket_port_reset(self.config.port)
            self.subscriptions.after_reconnect(data_maintained=False)
            return
        reason = classify_ibkr_error(code)
        self.connection.degraded(code=code, message=f"{reason}:{message}")
        try:
            self.callbacks.fail(request_id, reason)
        except Exception:
            return
