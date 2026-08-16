"""Private lazy boundary around the inseparable official IBKR client surface."""

from __future__ import annotations

import ipaddress
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, cast

from stocker_runtime.ingestion.dynamic_market_data import (
    MAX_EXACT_CONTRACT_CANDIDATES,
    MAX_OPTION_PARAMETER_SETS,
    ContractCandidate,
    OptionParameterSet,
)
from stocker_runtime.ingestion.ibkr_api import (
    OfficialIBKRDependencyError,
    require_official_ibkr_api,
)
from stocker_runtime.ingestion.ibkr_market_data import (
    MAX_MARKET_DATA_REQUEST_ID,
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


@dataclass
class _MetadataWaiter:
    kind: str
    completed: threading.Event = field(default_factory=threading.Event)
    parameter_sets: list[OptionParameterSet] = field(default_factory=list)
    contracts: list[ContractCandidate] = field(default_factory=list)
    error: str | None = None


def _price_tick_projection(feed_kind: str, tick_type: int) -> tuple[str, str] | None:
    if feed_kind == "quotes" and tick_type in {1, 2, 9}:
        return "quote", {1: "bid", 2: "ask", 9: "close"}[tick_type]
    if feed_kind == "trades" and tick_type == 4:
        return "trade", "last"
    return None


def _size_tick_projection(feed_kind: str, tick_type: int) -> tuple[str, str] | None:
    if feed_kind == "quotes" and tick_type in {0, 3, 27, 28, 29, 30}:
        return "quote", {
            0: "bid_size",
            3: "ask_size",
            27: "call_open_interest",
            28: "put_open_interest",
            29: "call_option_volume",
            30: "put_option_volume",
        }[tick_type]
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
        require_official_ibkr_api()
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except (ImportError, OfficialIBKRDependencyError) as error:
        raise OfficialBridgeUnavailable("official IBKR API is not verified") from error

    configured = {item.request_id: item for item in subscriptions}
    owner: _PrivateOfficialBridge | None = None

    class _Callbacks(EWrapper):  # type: ignore[misc]
        def connectionClosed(self) -> None:  # noqa: N802
            if owner is not None:
                owner.connection_closed()

        def nextValidId(self, _order_id: int) -> None:  # noqa: N802
            if owner is not None:
                owner._session_ready_callback()

        def tickPrice(self, reqId: int, tickType: int, price: float, _attrib: Any) -> None:  # noqa: N802
            if owner is not None:
                owner.tick_price(reqId, tickType, float(price))

        def tickSize(self, reqId: int, tickType: int, size: Any) -> None:  # noqa: N802
            if owner is not None:
                owner.tick_size(reqId, tickType, float(size))

        def tickSnapshotEnd(self, reqId: int) -> None:  # noqa: N802
            if owner is not None:
                owner.snapshot_end(reqId)

        def tickOptionComputation(  # noqa: N802
            self,
            reqId: int,
            tickType: int,
            _tickAttrib: Any,
            impliedVol: float,
            delta: float,
            optPrice: float,
            pvDividend: float,
            gamma: float,
            vega: float,
            theta: float,
            undPrice: float,
        ) -> None:
            if owner is not None:
                owner.tick_option_computation(
                    reqId,
                    tickType,
                    implied_volatility=float(impliedVol),
                    delta=float(delta),
                    option_price=float(optPrice),
                    present_value_dividend=float(pvDividend),
                    gamma=float(gamma),
                    vega=float(vega),
                    theta=float(theta),
                    underlying_price=float(undPrice),
                )

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

        def securityDefinitionOptionParameter(  # noqa: N802
            self,
            reqId: int,
            exchange: str,
            underlyingConId: int,
            tradingClass: str,
            multiplier: str,
            expirations: set[str],
            strikes: set[float],
        ) -> None:
            del underlyingConId
            if owner is not None:
                owner.option_parameter(
                    reqId,
                    exchange,
                    tradingClass,
                    multiplier,
                    expirations,
                    strikes,
                )

        def securityDefinitionOptionParameterEnd(self, reqId: int) -> None:  # noqa: N802
            if owner is not None:
                owner.metadata_end(reqId, "parameters")

        def contractDetails(self, reqId: int, contractDetails: Any) -> None:  # noqa: N802
            if owner is not None:
                owner.contract_detail(reqId, contractDetails)

        def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
            if owner is not None:
                owner.metadata_end(reqId, "contracts")

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

    def make_metadata_contract() -> Any:
        return Contract()

    contracts = {request_id: make_contract(item) for request_id, item in configured.items()}
    owner = _PrivateOfficialBridge(
        client=client,
        host=host,
        port=port,
        client_id=client_id,
        configured=configured,
        contracts=contracts,
        contract_factory=make_contract,
        metadata_contract_factory=make_metadata_contract,
    )
    return cast(MarketDataAdapter, owner)


class _PrivateOfficialBridge:
    """The sole Phase 3 holder of the official client instance."""

    _THREAD_STOP_TIMEOUT_SECONDS = 5.0
    _SESSION_READY_TIMEOUT_SECONDS = 5.0

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
        metadata_contract_factory: Callable[[], Any],
    ) -> None:
        self.__client = client
        self._host = host
        self._port = port
        self._client_id = client_id
        self._configured = configured
        self._contracts = contracts
        self._contract_factory = contract_factory
        self._metadata_contract_factory = metadata_contract_factory
        self._fences: dict[int, CallbackFence] = {}
        self._logical_transport_ids: dict[int, int] = {}
        self._transport_logical_ids: dict[int, int] = {}
        self._retired_transport_ids: dict[int, int] = {}
        self._next_retry_transport_id = MAX_MARKET_DATA_REQUEST_ID
        self._callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult] | None = None
        self._disconnect_callback: Callable[[int], None] | None = None
        self._status_callback: Callable[[MarketDataStatus], None] | None = None
        self._thread: threading.Thread | None = None
        self._connection_state_lock = threading.Lock()
        self._callback_context = threading.local()
        self._connection_epoch = 0
        self._active_connection_epoch: int | None = None
        self._ready_connection_epoch: int | None = None
        self._session_ready_event = threading.Event()
        self._metadata_operation_lock = threading.Lock()
        self._metadata_state_lock = threading.Lock()
        self._metadata_waiters: dict[int, _MetadataWaiter] = {}
        self._next_metadata_request_id = 1_500_000_000

    def set_callback(
        self, callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult]
    ) -> None:
        self._callback = callback

    def set_disconnect_callback(self, callback: Callable[[int], None]) -> None:
        self._disconnect_callback = callback

    def set_status_callback(self, callback: Callable[[MarketDataStatus], None]) -> None:
        self._status_callback = callback

    def connect(self) -> None:
        with self._connection_state_lock:
            if self._active_connection_epoch is not None:
                raise OfficialBridgeUnavailable("official IBKR socket is already connected")
            if self._thread is not None and self._thread.is_alive():
                raise OfficialBridgeUnavailable("prior official IBKR socket thread is still active")
            self._connection_epoch += 1
            connection_epoch = self._connection_epoch
            self._active_connection_epoch = connection_epoch
            self._ready_connection_epoch = None
            self._session_ready_event.clear()
        try:
            result = self.__client.connect(self._host, self._port, self._client_id)
        except Exception:
            with self._connection_state_lock:
                if self._active_connection_epoch == connection_epoch:
                    self._active_connection_epoch = None
            raise
        if result is False:
            with self._connection_state_lock:
                if self._active_connection_epoch == connection_epoch:
                    self._active_connection_epoch = None
            raise OfficialBridgeUnavailable("official IBKR socket connection failed")
        thread = threading.Thread(
            target=self._run_connection,
            args=(connection_epoch,),
            name="stocker-v2-ibkr-market-data",
            daemon=True,
        )
        with self._connection_state_lock:
            if self._active_connection_epoch != connection_epoch:
                raise OfficialBridgeUnavailable("official IBKR socket closed while connecting")
            self._thread = thread
        try:
            thread.start()
        except Exception:
            with self._connection_state_lock:
                if self._active_connection_epoch == connection_epoch:
                    self._active_connection_epoch = None
                if self._thread is thread:
                    self._thread = None
            with suppress(Exception):
                self.__client.disconnect()
            self._fences.clear()
            self._logical_transport_ids.clear()
            self._transport_logical_ids.clear()
            self._retired_transport_ids.clear()
            self._configured.clear()
            self._contracts.clear()
            raise
        if not self._session_ready_event.wait(timeout=self._SESSION_READY_TIMEOUT_SECONDS):
            self.disconnect()
            raise OfficialBridgeUnavailable("official IBKR session readiness timed out")
        with self._connection_state_lock:
            ready = (
                self._active_connection_epoch == connection_epoch
                and self._ready_connection_epoch == connection_epoch
            )
        if not ready:
            self.disconnect()
            raise OfficialBridgeUnavailable("official IBKR session closed before readiness")

    def _run_connection(self, connection_epoch: int) -> None:
        self._callback_context.connection_epoch = connection_epoch
        try:
            self.__client.run()
        finally:
            try:
                self.connection_closed()
            finally:
                del self._callback_context.connection_epoch
                with self._connection_state_lock:
                    if self._thread is threading.current_thread():
                        self._thread = None

    def _session_ready_callback(self) -> None:
        callback_epoch = getattr(self._callback_context, "connection_epoch", None)
        with self._connection_state_lock:
            if callback_epoch is None or callback_epoch != self._active_connection_epoch:
                return
            self._ready_connection_epoch = callback_epoch
            self._session_ready_event.set()

    def _callback_is_current_connection(self) -> bool:
        callback_epoch = getattr(self._callback_context, "connection_epoch", None)
        with self._connection_state_lock:
            active_epoch = self._active_connection_epoch
        return active_epoch is not None and callback_epoch == active_epoch

    def disconnect(self) -> None:
        with self._connection_state_lock:
            self._active_connection_epoch = None
            self._ready_connection_epoch = None
            self._session_ready_event.set()
            thread = self._thread
        try:
            self.__client.disconnect()
        finally:
            self._fences.clear()
            self._logical_transport_ids.clear()
            self._transport_logical_ids.clear()
            self._retired_transport_ids.clear()
            self._configured.clear()
            self._contracts.clear()
            if thread is not None and thread is not threading.current_thread():
                if thread.ident is not None:
                    thread.join(timeout=self._THREAD_STOP_TIMEOUT_SECONDS)
                    if thread.is_alive():
                        raise OfficialBridgeUnavailable(
                            "official IBKR socket thread did not stop after disconnect"
                        )
                with self._connection_state_lock:
                    if self._thread is thread:
                        self._thread = None

    def configure_subscriptions(self, subscriptions: tuple[IBKRSubscription, ...]) -> None:
        """Add core-owned exact identities without reusing a request id."""

        if len({item.request_id for item in subscriptions}) != len(subscriptions):
            raise OfficialBridgeUnavailable("IBKR request identifiers must be unique")
        configured = {item.request_id: item for item in subscriptions}
        for request_id, item in configured.items():
            existing = self._configured.get(request_id)
            if existing is not None and existing != item:
                raise OfficialBridgeUnavailable("configured subscription identity changed")
        additions = {
            request_id: item
            for request_id, item in configured.items()
            if request_id not in self._configured
        }
        self._configured.update(additions)
        self._contracts.update(
            {request_id: self._contract_factory(item) for request_id, item in additions.items()}
        )

    def _new_metadata_waiter(self, kind: str) -> tuple[int, _MetadataWaiter]:
        with self._metadata_state_lock:
            request_id = self._next_metadata_request_id
            self._next_metadata_request_id += 1
            if self._next_metadata_request_id >= 2_000_000_000:
                raise OfficialBridgeUnavailable("IBKR metadata request id range exhausted")
            waiter = _MetadataWaiter(kind)
            self._metadata_waiters[request_id] = waiter
            return request_id, waiter

    def _finish_metadata_waiter(self, request_id: int, waiter: _MetadataWaiter) -> _MetadataWaiter:
        if not waiter.completed.wait(timeout=5.0):
            with self._metadata_state_lock:
                self._metadata_waiters.pop(request_id, None)
            raise OfficialBridgeUnavailable("IBKR option metadata request timed out")
        with self._metadata_state_lock:
            self._metadata_waiters.pop(request_id, None)
        if waiter.error is not None:
            raise OfficialBridgeUnavailable(waiter.error)
        return waiter

    def option_parameters(
        self, *, underlying_con_id: int, symbol: str
    ) -> tuple[OptionParameterSet, ...]:
        """Request bounded option metadata; this never creates a live subscription."""

        if underlying_con_id <= 0 or not symbol:
            raise OfficialBridgeUnavailable("underlying option identity is invalid")
        with self._metadata_operation_lock:
            request_id, waiter = self._new_metadata_waiter("parameters")
            try:
                self.__client.reqSecDefOptParams(
                    request_id,
                    symbol,
                    "",
                    "STK",
                    underlying_con_id,
                )
            except Exception:
                with self._metadata_state_lock:
                    self._metadata_waiters.pop(request_id, None)
                raise
            return tuple(self._finish_metadata_waiter(request_id, waiter).parameter_sets)

    def option_contracts(
        self,
        *,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        multiplier: str,
        trading_class: str,
    ) -> tuple[ContractCandidate, ...]:
        """Resolve one exact selected contract identity without streaming a chain."""

        if (
            not symbol
            or len(expiry) != 8
            or not expiry.isdigit()
            or strike <= 0
            or right not in {"C", "P"}
            or not multiplier
            or not trading_class
        ):
            raise OfficialBridgeUnavailable("exact option contract query is invalid")
        with self._metadata_operation_lock:
            request_id, waiter = self._new_metadata_waiter("contracts")
            contract = self._metadata_contract_factory()
            contract.symbol = symbol
            contract.secType = "OPT"
            contract.exchange = "SMART"
            contract.currency = "USD"
            contract.lastTradeDateOrContractMonth = expiry
            contract.strike = strike
            contract.right = right
            contract.multiplier = multiplier
            contract.tradingClass = trading_class
            try:
                self.__client.reqContractDetails(request_id, contract)
            except Exception:
                with self._metadata_state_lock:
                    self._metadata_waiters.pop(request_id, None)
                raise
            return tuple(self._finish_metadata_waiter(request_id, waiter).contracts)

    def option_parameter(
        self,
        request_id: int,
        exchange: str,
        trading_class: str,
        multiplier: str,
        expirations: set[str],
        strikes: set[float],
    ) -> None:
        if not self._callback_is_current_connection():
            return
        with self._metadata_state_lock:
            waiter = self._metadata_waiters.get(request_id)
            if waiter is None or waiter.kind != "parameters" or waiter.completed.is_set():
                return
            try:
                if len(waiter.parameter_sets) >= MAX_OPTION_PARAMETER_SETS:
                    raise ValueError("option parameter set bound exceeded")
                waiter.parameter_sets.append(
                    OptionParameterSet(
                        exchange=str(exchange),
                        trading_class=str(trading_class),
                        multiplier=str(multiplier),
                        expirations=tuple(sorted(str(value) for value in expirations)),
                        strikes=tuple(sorted(float(value) for value in strikes)),
                    )
                )
            except Exception:
                waiter.error = "IBKR option parameter metadata is invalid or unbounded"
                waiter.completed.set()

    def contract_detail(self, request_id: int, details: Any) -> None:
        if not self._callback_is_current_connection():
            return
        with self._metadata_state_lock:
            waiter = self._metadata_waiters.get(request_id)
            if waiter is None or waiter.kind != "contracts" or waiter.completed.is_set():
                return
            try:
                if len(waiter.contracts) >= MAX_EXACT_CONTRACT_CANDIDATES:
                    raise ValueError("exact option contract candidate bound exceeded")
                contract = details.contract
                waiter.contracts.append(
                    ContractCandidate(
                        con_id=int(contract.conId),
                        symbol=str(contract.symbol),
                        expiry=str(contract.lastTradeDateOrContractMonth),
                        strike=float(contract.strike),
                        right=str(contract.right),
                        multiplier=str(contract.multiplier),
                        exchange=str(contract.exchange),
                        currency=str(contract.currency),
                        trading_class=str(contract.tradingClass),
                    )
                )
            except Exception:
                waiter.error = "IBKR exact option contract metadata is invalid or unbounded"
                waiter.completed.set()

    def metadata_end(self, request_id: int, kind: str) -> None:
        if not self._callback_is_current_connection():
            return
        with self._metadata_state_lock:
            waiter = self._metadata_waiters.get(request_id)
            if waiter is not None and waiter.kind == kind:
                waiter.completed.set()

    def _metadata_error(self, request_id: int, code: int) -> None:
        with self._metadata_state_lock:
            waiter = self._metadata_waiters.get(request_id)
            if waiter is not None:
                waiter.error = f"IBKR option metadata request rejected ({code})"
                waiter.completed.set()

    def subscribe(self, fence: CallbackFence) -> None:
        if fence.request_id is None or fence.request_id not in self._configured:
            raise OfficialBridgeUnavailable("subscription request is not configured")
        logical_request_id = fence.request_id
        if logical_request_id in self._logical_transport_ids:
            raise OfficialBridgeUnavailable("subscription request is already active")
        self._start_transport_request(logical_request_id, logical_request_id, fence)

    def _start_transport_request(
        self,
        logical_request_id: int,
        transport_request_id: int,
        fence: CallbackFence,
    ) -> None:
        item = self._configured[logical_request_id]
        self._logical_transport_ids[logical_request_id] = transport_request_id
        self._transport_logical_ids[transport_request_id] = logical_request_id
        self._fences[transport_request_id] = fence
        if item.feed_kind == "bars":
            self.__client.reqRealTimeBars(
                transport_request_id,
                self._contracts[logical_request_id],
                5,
                "TRADES",
                False,
                [],
            )
        else:
            self.__client.reqMktData(
                transport_request_id,
                self._contracts[logical_request_id],
                "100,101" if item.security_type == "OPT" else "",
                item.snapshot,
                False,
                [],
            )

    def retry_subscription(self, fence: CallbackFence) -> None:
        """Cancel and replace one configured request without disturbing healthy requests."""

        if fence.request_id is None or fence.request_id not in self._configured:
            raise OfficialBridgeUnavailable("subscription retry request is not configured")
        request_id = fence.request_id
        configured = self._configured[request_id]
        transport_request_id = self._logical_transport_ids.pop(
            request_id,
            self._retired_transport_ids.get(request_id),
        )
        if transport_request_id is None:
            raise OfficialBridgeUnavailable("subscription retry request is not active")
        self._fences.pop(transport_request_id, None)
        self._transport_logical_ids.pop(transport_request_id, None)
        self._retired_transport_ids[request_id] = transport_request_id
        if configured.feed_kind == "bars":
            self.__client.cancelRealTimeBars(transport_request_id)
        else:
            self.__client.cancelMktData(transport_request_id)
        self._retired_transport_ids.pop(request_id, None)
        while (
            self._next_retry_transport_id in self._configured
            or self._next_retry_transport_id in self._transport_logical_ids
            or self._next_retry_transport_id in self._retired_transport_ids.values()
        ):
            self._next_retry_transport_id -= 1
        if self._next_retry_transport_id < 0:
            raise OfficialBridgeUnavailable("IBKR retry request id range exhausted")
        replacement_request_id = self._next_retry_transport_id
        self._next_retry_transport_id -= 1
        self._start_transport_request(request_id, replacement_request_id, fence)

    def cancel(self, request_id: int) -> None:
        configured = self._configured.get(request_id)
        if configured is None:
            return
        transport_request_id = self._logical_transport_ids.get(request_id)
        if transport_request_id is None:
            return
        if configured.feed_kind == "bars":
            self.__client.cancelRealTimeBars(transport_request_id)
        else:
            self.__client.cancelMktData(transport_request_id)
        self._fences.pop(transport_request_id, None)
        self._transport_logical_ids.pop(transport_request_id, None)
        self._logical_transport_ids.pop(request_id, None)
        self._configured.pop(request_id, None)
        self._contracts.pop(request_id, None)

    def _configured_for_transport(self, request_id: int) -> IBKRSubscription | None:
        logical_request_id = self._transport_logical_ids.get(request_id)
        return None if logical_request_id is None else self._configured.get(logical_request_id)

    def emit(self, request_id: int, kind: str, values: dict[str, object]) -> None:
        if not self._callback_is_current_connection():
            return
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
        configured = self._configured_for_transport(request_id)
        if configured is None:
            return
        projection = _price_tick_projection(configured.feed_kind, tick_type)
        if projection is not None:
            kind, name = projection
            self.emit(request_id, kind, {name: price})

    def tick_size(self, request_id: int, tick_type: int, size: float) -> None:
        configured = self._configured_for_transport(request_id)
        if configured is None:
            return
        if tick_type in {27, 28, 29, 30} and configured.security_type != "OPT":
            return
        projection = _size_tick_projection(configured.feed_kind, tick_type)
        if projection is not None:
            kind, name = projection
            self.emit(request_id, kind, {name: size})

    def tick_option_computation(
        self,
        request_id: int,
        tick_type: int,
        *,
        implied_volatility: float,
        delta: float,
        option_price: float,
        present_value_dividend: float,
        gamma: float,
        vega: float,
        theta: float,
        underlying_price: float,
    ) -> None:
        configured = self._configured_for_transport(request_id)
        if (
            configured is None
            or configured.feed_kind != "quotes"
            or configured.security_type != "OPT"
        ):
            return
        self.emit(
            request_id,
            "option_computation",
            {
                "tick_type": int(tick_type),
                "implied_volatility": implied_volatility,
                "delta": delta,
                "option_price": option_price,
                "present_value_dividend": present_value_dividend,
                "gamma": gamma,
                "vega": vega,
                "theta": theta,
                "underlying_price": underlying_price,
            },
        )

    def snapshot_end(self, request_id: int) -> None:
        if not self._callback_is_current_connection():
            return
        configured = self._configured_for_transport(request_id)
        if configured is None or not configured.snapshot:
            return
        logical_request_id = self._transport_logical_ids[request_id]
        self.emit(request_id, "option_snapshot_end", {"complete": True})
        self._fences.pop(request_id, None)
        self._transport_logical_ids.pop(request_id, None)
        self._logical_transport_ids.pop(logical_request_id, None)
        self._configured.pop(logical_request_id, None)
        self._contracts.pop(logical_request_id, None)
        callback = self._status_callback
        if callback is not None:
            callback(
                MarketDataStatus(
                    kind="snapshot_end",
                    code=0,
                    request_id=logical_request_id,
                    message="option snapshot completed",
                    received_at_us=time.time_ns() // 1_000,
                )
            )

    def official_status(self, request_id: int, code: int, message: str) -> None:
        if not self._callback_is_current_connection():
            return
        if request_id >= 0:
            self._metadata_error(request_id, code)
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
        logical_request_id = None if request_id < 0 else self._transport_logical_ids.get(request_id)
        if request_id >= 0 and logical_request_id is None:
            return
        callback(
            MarketDataStatus(
                kind=cast(Any, kind),
                code=code,
                request_id=logical_request_id,
                message=message,
                received_at_us=time.time_ns() // 1_000,
                affected_feed_kinds=cast(
                    Any,
                    farm_degraded.get(code, farm_recovered.get(code, ())),
                ),
            )
        )

    def connection_closed(self) -> None:
        callback_epoch = getattr(self._callback_context, "connection_epoch", None)
        with self._connection_state_lock:
            active_epoch = self._active_connection_epoch
            if active_epoch is None or callback_epoch != active_epoch:
                return
            self._active_connection_epoch = None
            self._ready_connection_epoch = None
            self._session_ready_event.set()
            callback = self._disconnect_callback
        self._fences.clear()
        self._logical_transport_ids.clear()
        self._transport_logical_ids.clear()
        self._retired_transport_ids.clear()
        if callback is not None:
            callback(time.time_ns() // 1_000)
