"""Private lazy boundary around the inseparable official IBKR client surface."""

from __future__ import annotations

import ipaddress
import threading
import time
from collections.abc import Callable
from typing import Any, cast

from stocker_runtime.ingestion.ibkr_market_data import (
    IBKRSubscription,
    MarketDataAdapter,
    MarketDataStatus,
)
from stocker_runtime.ingestion.inbox import (
    AdmissionResult,
    CallbackFence,
    MarketDataCallback,
)


class OfficialBridgeUnavailable(RuntimeError):
    """The externally installed official API cannot be verified or loaded."""


def _price_tick_projection(feed_kind: str, tick_type: int) -> tuple[str, str] | None:
    if feed_kind == "quotes" and tick_type in {1, 2}:
        return "quote", {1: "bid", 2: "ask"}[tick_type]
    if feed_kind == "trades" and tick_type == 4:
        return "trade", "last"
    return None


def _size_tick_projection(feed_kind: str, tick_type: int) -> tuple[str, str] | None:
    if feed_kind == "quotes" and tick_type in {0, 3}:
        return "quote", {0: "bid_size", 3: "ask_size"}[tick_type]
    if feed_kind == "trades" and tick_type == 5:
        return "trade", "size"
    return None


def create_official_bridge(
    *,
    host: str,
    port: int,
    client_id: int,
    read_only: bool,
    external_read_only_verified: bool,
    subscriptions: tuple[IBKRSubscription, ...] = (),
) -> MarketDataAdapter:
    """Construct a private official client holder; imports remain lazy and server-only."""

    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise OfficialBridgeUnavailable("IBKR host must be a literal loopback address") from error
    if not address.is_loopback or not 1 <= port <= 65_535 or client_id < 0:
        raise OfficialBridgeUnavailable("unsafe IBKR socket configuration")
    if not read_only or not external_read_only_verified:
        raise OfficialBridgeUnavailable("external IBKR Read-Only API verification is required")
    if len({item.request_id for item in subscriptions}) != len(subscriptions):
        raise OfficialBridgeUnavailable("IBKR request identifiers must be unique")
    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except ImportError as error:
        raise OfficialBridgeUnavailable("official IBKR API is not installed") from error

    configured = {item.request_id: item for item in subscriptions}
    owner: _PrivateOfficialBridge | None = None

    class _Callbacks(EWrapper):  # type: ignore[misc]
        def connectionClosed(self) -> None:  # noqa: N802
            if owner is not None:
                owner.connection_closed()

        def tickPrice(self, reqId: int, tickType: int, price: float, _attrib: Any) -> None:  # noqa: N802
            if owner is not None:
                owner.tick_price(reqId, tickType, float(price))

        def tickSize(self, reqId: int, tickType: int, size: Any) -> None:  # noqa: N802
            if owner is not None:
                owner.tick_size(reqId, tickType, float(size))

        def error(self, reqId: int, errorCode: int, errorString: str, *args: Any) -> None:  # noqa: N802
            del args
            if owner is not None:
                owner.official_status(reqId, errorCode, errorString)

        def realtimeBar(  # noqa: N802
            self,
            reqId: int,
            bar_time: int,
            open_: float,
            high: float,
            low: float,
            close: float,
            volume: Any,
            _wap: Any,
            _count: int,
        ) -> None:
            if owner is not None:
                owner.emit(
                    reqId,
                    "bar",
                    {
                        "event_at_us": int(bar_time) * 1_000_000,
                        "open": float(open_),
                        "high": float(high),
                        "low": float(low),
                        "close": float(close),
                        "volume": float(volume),
                    },
                )

    wrapper = _Callbacks()
    client = EClient(wrapper)

    def make_contract(item: IBKRSubscription) -> Any:
        contract = Contract()
        contract.conId = item.con_id
        contract.symbol = item.symbol
        contract.secType = item.security_type
        contract.exchange = item.exchange
        contract.currency = item.currency
        return contract

    contracts = {request_id: make_contract(item) for request_id, item in configured.items()}
    owner = _PrivateOfficialBridge(
        client=client,
        host=host,
        port=port,
        client_id=client_id,
        configured=configured,
        contracts=contracts,
        contract_factory=make_contract,
    )
    return cast(MarketDataAdapter, owner)


class _PrivateOfficialBridge:
    """The sole Phase 3 holder of the official client instance."""

    def __init__(
        self,
        *,
        client: Any,
        host: str,
        port: int,
        client_id: int,
        configured: dict[int, IBKRSubscription],
        contracts: dict[int, Any],
        contract_factory: Callable[[IBKRSubscription], Any],
    ) -> None:
        self.__client = client
        self._host = host
        self._port = port
        self._client_id = client_id
        self._configured = configured
        self._contracts = contracts
        self._contract_factory = contract_factory
        self._fences: dict[int, CallbackFence] = {}
        self._callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult] | None = None
        self._disconnect_callback: Callable[[int], None] | None = None
        self._status_callback: Callable[[MarketDataStatus], None] | None = None
        self._thread: threading.Thread | None = None

    def set_callback(
        self, callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult]
    ) -> None:
        self._callback = callback

    def set_disconnect_callback(self, callback: Callable[[int], None]) -> None:
        self._disconnect_callback = callback

    def set_status_callback(self, callback: Callable[[MarketDataStatus], None]) -> None:
        self._status_callback = callback

    def connect(self) -> None:
        result = self.__client.connect(self._host, self._port, self._client_id)
        if result is False:
            raise OfficialBridgeUnavailable("official IBKR socket connection failed")
        self._thread = threading.Thread(
            target=self.__client.run,
            name="stocker-v2-ibkr-market-data",
            daemon=True,
        )
        self._thread.start()

    def disconnect(self) -> None:
        self.__client.disconnect()

    def configure_subscriptions(self, subscriptions: tuple[IBKRSubscription, ...]) -> None:
        """Install the core-owned exact read-only request set before connecting."""

        if self._thread is not None:
            raise OfficialBridgeUnavailable("subscriptions cannot change after connect")
        if len({item.request_id for item in subscriptions}) != len(subscriptions):
            raise OfficialBridgeUnavailable("IBKR request identifiers must be unique")
        configured = {item.request_id: item for item in subscriptions}
        if self._configured and self._configured != configured:
            raise OfficialBridgeUnavailable("configured subscription identity changed")
        self._configured = configured
        self._contracts = {
            request_id: self._contract_factory(item) for request_id, item in configured.items()
        }

    def subscribe(self, fence: CallbackFence) -> None:
        if fence.request_id is None or fence.request_id not in self._configured:
            raise OfficialBridgeUnavailable("subscription request is not configured")
        request_id = fence.request_id
        item = self._configured[request_id]
        self._fences[request_id] = fence
        if item.feed_kind == "bars":
            self.__client.reqRealTimeBars(
                request_id,
                self._contracts[request_id],
                5,
                "TRADES",
                False,
                [],
            )
        else:
            self.__client.reqMktData(
                request_id,
                self._contracts[request_id],
                "",
                False,
                False,
                [],
            )

    def cancel(self, request_id: int) -> None:
        configured = self._configured.get(request_id)
        if configured is None:
            return
        if configured.feed_kind == "bars":
            self.__client.cancelRealTimeBars(request_id)
        else:
            self.__client.cancelMktData(request_id)
        self._fences.pop(request_id, None)

    def emit(self, request_id: int, kind: str, values: dict[str, object]) -> None:
        fence = self._fences.get(request_id)
        callback = self._callback
        if fence is None or callback is None:
            return
        received_at_us = time.time_ns() // 1_000
        payload = {"event_at_us": received_at_us, **values}
        callback(
            fence,
            MarketDataCallback(
                callback_kind=kind,
                received_at_us=received_at_us,
                provider_at_us=None,
                payload=cast(Any, payload),
            ),
        )

    def tick_price(self, request_id: int, tick_type: int, price: float) -> None:
        configured = self._configured.get(request_id)
        if configured is None:
            return
        projection = _price_tick_projection(configured.feed_kind, tick_type)
        if projection is not None:
            kind, name = projection
            self.emit(request_id, kind, {name: price})

    def tick_size(self, request_id: int, tick_type: int, size: float) -> None:
        configured = self._configured.get(request_id)
        if configured is None:
            return
        projection = _size_tick_projection(configured.feed_kind, tick_type)
        if projection is not None:
            kind, name = projection
            self.emit(request_id, kind, {name: size})

    def official_status(self, request_id: int, code: int, message: str) -> None:
        callback = self._status_callback
        if callback is None:
            return
        temporary = {1100, 1300, 2110}
        recovered = {1101, 1102}
        farm_degraded = {2103: ("quotes", "trades"), 2105: ("bars",)}
        farm_recovered = {2104: ("quotes", "trades"), 2106: ("bars",), 2158: ()}
        pacing = {100, 101, 420}
        rejected = {162, 200, 354, 10167, 10168}
        if code in temporary:
            kind = "temporary_disconnect"
        elif code in recovered:
            kind = "recovered"
        elif code in farm_degraded:
            kind = "farm_degraded"
        elif code in farm_recovered:
            kind = "farm_recovered"
        elif code in pacing:
            kind = "pacing"
        elif code in rejected:
            kind = "request_rejected"
        else:
            return
        callback(
            MarketDataStatus(
                kind=cast(Any, kind),
                code=code,
                request_id=None if request_id < 0 else request_id,
                message=message,
                received_at_us=time.time_ns() // 1_000,
                affected_feed_kinds=cast(
                    Any,
                    farm_degraded.get(code, farm_recovered.get(code, ())),
                ),
            )
        )

    def connection_closed(self) -> None:
        self._fences.clear()
        callback = self._disconnect_callback
        if callback is not None:
            callback(time.time_ns() // 1_000)
