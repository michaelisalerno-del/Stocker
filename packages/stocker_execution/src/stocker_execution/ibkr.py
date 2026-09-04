"""Minimal IBKR connection, market-data, and explicit execution boundary."""

import asyncio
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import pairwise
from math import isfinite
from typing import Any, Literal, Protocol, cast

from stocker_core.config import IbkrConfig
from stocker_core.markets import CAP_BUCKETS_V1, ActivityScanner, CapBucket, MarketDefinition
from stocker_core.runs import Environment
from stocker_execution.activity_shortlist import ScannerCandidate, ScannerCapabilities
from stocker_execution.execution_models import (
    BrokerAccountState,
    BrokerFill,
    BrokerOpenOrder,
    BrokerOrderIds,
    BrokerOrderStatus,
    BrokerPosition,
    OrderAction,
    OrderLifecycle,
    OrderPlan,
    OrderRole,
)


class IbkrError(RuntimeError):
    """A clear failure at the IBKR connection or data boundary."""


MAX_ACTIVE_SCANNERS = 10
HISTORICAL_REQUEST_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class IbkrApiError:
    """One sanitized TWS/Gateway API error observed during a data request."""

    request_id: int
    code: int
    message: str
    con_id: int | None
    symbol: str | None
    exchange: str | None


_IBKR_ACCOUNT_IDENTIFIER = re.compile(
    r"\b((?:DUP|DU|U|F|FA|D|DF|I|IB|M|S))(\d{4,})\b", re.IGNORECASE
)


def sanitize_ibkr_message(message: object) -> str:
    """Mask recognizable IBKR account identifiers in diagnostic text."""

    return _IBKR_ACCOUNT_IDENTIFIER.sub(
        lambda match: f"{match.group(1)}***{match.group(2)[-3:]}",
        str(message),
    )


def mask_ibkr_account(account_id: str | None) -> str | None:
    """Mask an account identifier for logs and human-readable status output."""

    if account_id is None:
        return None
    if len(account_id) <= 5:
        return "***"
    return f"{account_id[:2]}***{account_id[-3:]}"


class _IbClient(Protocol):
    client: Any

    async def connectAsync(
        self,
        host: str,
        port: int,
        *,
        clientId: int,
        timeout: float,
        readonly: bool,
        account: str,
        raiseSyncErrors: bool,
        fetchFields: Any,
    ) -> object: ...

    def disconnect(self) -> object: ...

    def isConnected(self) -> bool: ...

    def managedAccounts(self) -> list[str]: ...

    async def qualifyContractsAsync(
        self, *contracts: object, returnAll: bool = False
    ) -> list[object]: ...

    async def reqHistoricalDataAsync(self, contract: object, **kwargs: object) -> list[object]: ...

    async def reqTickersAsync(
        self, *contracts: object, regulatorySnapshot: bool = False
    ) -> list[object]: ...

    def reqMktData(
        self,
        contract: object,
        genericTickList: str = "",
        snapshot: bool = False,
        regulatorySnapshot: bool = False,
    ) -> object: ...

    def cancelMktData(self, contract: object) -> bool: ...

    def reqMarketDataType(self, marketDataType: int) -> object: ...

    async def reqSecDefOptParamsAsync(
        self,
        underlyingSymbol: str,
        futFopExchange: str,
        underlyingSecType: str,
        underlyingConId: int,
    ) -> list[object]: ...

    async def reqScannerDataAsync(
        self,
        subscription: object,
        scannerSubscriptionOptions: list[object] | None = None,
        scannerSubscriptionFilterOptions: list[object] | None = None,
    ) -> list[object]: ...

    async def reqScannerParametersAsync(self) -> str: ...

    async def accountSummaryAsync(self, account: str = "") -> list[object]: ...

    async def reqContractDetailsAsync(self, contract: object) -> list[object]: ...

    def placeOrder(self, contract: object, order: object) -> object: ...

    async def reqAllOpenOrdersAsync(self) -> list[object]: ...

    async def reqCompletedOrdersAsync(self, apiOnly: bool) -> list[object]: ...

    def openTrades(self) -> list[object]: ...

    def trades(self) -> list[object]: ...

    async def reqPositionsAsync(self) -> list[object]: ...

    async def reqExecutionsAsync(self, execFilter: object = None) -> list[object]: ...

    def cancelOrder(self, order: object, manualCancelOrderTime: str = "") -> object: ...


class _QualifiedContract(Protocol):
    symbol: str
    conId: int
    exchange: str
    primaryExchange: str
    currency: str
    secType: str
    lastTradeDateOrContractMonth: str
    strike: float
    right: str
    multiplier: str
    tradingClass: str


class _SourceOptionChain(Protocol):
    exchange: str
    underlyingConId: int
    tradingClass: str
    multiplier: str
    expirations: list[str]
    strikes: list[float]


class _SourceBar(Protocol):
    date: date | datetime
    open: float
    high: float
    low: float
    close: float
    volume: Decimal | float | int


class _SourceTicker(Protocol):
    time: datetime | None
    bid: float
    ask: float
    last: float
    close: float
    marketDataType: int


class _SourceOptionComputation(Protocol):
    impliedVol: float
    delta: float
    gamma: float


class _SourceOptionTicker(Protocol):
    bid: float
    ask: float
    callOpenInterest: float
    putOpenInterest: float
    marketDataType: int
    modelGreeks: _SourceOptionComputation | None


@dataclass(frozen=True, slots=True)
class BrokerSession:
    """Confirmed identity and state for one configured broker session."""

    environment: Environment
    account_id: str
    connected: bool

    @property
    def masked_account_id(self) -> str:
        """Return an account identifier suitable for normal human-readable output."""

        return mask_ibkr_account(self.account_id) or "***"


@dataclass(frozen=True, slots=True)
class MarketDataSubscriptionStatus:
    """One physical streaming market-data request owned by this connection."""

    request_id: int
    con_id: int
    security_type: str
    exchange: str
    purpose: str
    snapshot: bool
    created_at: datetime
    consumer_count: int


@dataclass(frozen=True, slots=True)
class IbkrResourceStatus:
    """Broker-local resource facts Stocker can know without claiming account entitlement."""

    market_data_line_budget: int
    ib_async_max_requests: int | None
    ib_async_requests_interval: float | None
    active_market_data_lines: int
    active_underlying_lines: int
    active_option_lines: int
    subscriptions: tuple[MarketDataSubscriptionStatus, ...]
    market_data_requests_today: int
    deduplicated_requests_today: int
    capacity_rejects_today: int
    active_scanners: int
    pending_historical_work: int
    historical_concurrency_limit: int
    historical_requests_today: int
    scanner_requests_today: int
    pacing_state: str
    pacing_violations_today: int
    last_resource_error: str | None
    market_data_budget_label: str = "Stocker API line budget"
    ibkr_account_line_limit: int | None = None


@dataclass(slots=True)
class _ActiveMarketDataSubscription:
    request_id: int
    contract: object
    ticker: object
    con_id: int
    security_type: str
    exchange: str
    purpose: str
    created_at: datetime
    consumer_count: int = 1


@dataclass(frozen=True, slots=True)
class QualifiedInstrument:
    """Stable Stocker identity copied from one qualified IBKR stock contract."""

    symbol: str
    con_id: int
    exchange: str
    primary_exchange: str | None
    currency: str
    security_type: str


@dataclass(frozen=True, slots=True)
class QualifiedOption:
    """Stable identity copied from one qualified IBKR option contract."""

    symbol: str
    con_id: int
    exchange: str
    currency: str
    security_type: str
    expiry: date
    strike: float
    right: str
    multiplier: str
    trading_class: str


@dataclass(frozen=True, slots=True)
class OptionContractRequest:
    """One option identity advertised by an IBKR security-definition chain."""

    symbol: str
    exchange: str
    currency: str
    expiry: date
    strike: float
    right: str
    multiplier: str
    trading_class: str


@dataclass(frozen=True, slots=True)
class OptionChainDefinition:
    """Minimal IBKR option security-definition parameters for one chain."""

    exchange: str
    underlying_con_id: int
    trading_class: str
    multiplier: str
    expirations: tuple[date, ...]
    strikes: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OptionMarketSnapshot:
    """Contract snapshot using only IBKR tick-13 Model Option Computation Greeks."""

    option: QualifiedOption
    captured_at: datetime
    bid: float | None
    ask: float | None
    open_interest: float | None
    market_data_type: int | None
    model_iv: float | None
    model_delta: float | None
    model_gamma: float | None


@dataclass(frozen=True, slots=True)
class HistoricalBar:
    """Minimal unfilled OHLCV bar copied from an IBKR historical response."""

    timestamp: date | datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def validate_historical_bar(bar: HistoricalBar) -> HistoricalBar:
    """Return one valid normalized bar or fail before trading code can use it."""

    if not isinstance(bar.timestamp, (date, datetime)):
        raise ValueError("invalid historical bar timestamp")
    values = (bar.open, bar.high, bar.low, bar.close, bar.volume)
    if not all(isfinite(value) for value in values) or bar.volume < 0:
        raise ValueError("invalid historical bar values")
    if bar.high < max(bar.open, bar.low, bar.close) or bar.low > min(bar.open, bar.high, bar.close):
        raise ValueError("inconsistent historical bar OHLC values")
    return bar


@dataclass(frozen=True, slots=True)
class CurrentQuote:
    """Small current IBKR snapshot with missing prices represented explicitly."""

    symbol: str
    con_id: int
    timestamp: datetime | None
    bid: float | None
    ask: float | None
    last: float | None
    close: float | None
    market_data_type: int | None = None


@dataclass(frozen=True, slots=True)
class HistoricalVolatilitySnapshot:
    """One temporary generic-tick-104 observation for a qualified stock."""

    symbol: str
    con_id: int
    raw_historical_volatility: float
    observation_timestamp: datetime
    market_data_type: int
    unit: Literal["DECIMAL", "PERCENT"] = "DECIMAL"


def _new_client() -> _IbClient:
    from ib_async import IB

    return cast(_IbClient, IB())


def _no_startup_fetches() -> object:
    from ib_async import StartupFetchNONE

    return StartupFetchNONE


class IbkrConnection:
    """One independent PAPER or LIVE IBKR session with explicit execution opt-in."""

    def __init__(
        self,
        config: IbkrConfig,
        *,
        client: _IbClient | None = None,
        execution_enabled: bool = False,
    ) -> None:
        self.config = config
        self._client = client if client is not None else _new_client()
        self._library_max_requests = self._configure_ib_async_throttle()
        self._account_id: str | None = None
        self._execution_enabled = execution_enabled
        self._connection_epoch = 0
        self._open_orders_loaded = False
        self._completed_orders_loaded = False
        self._scanner_capabilities: ScannerCapabilities | None = None
        self._qualified_stock_cache: dict[tuple[str, str, str, str], QualifiedInstrument] = {}
        self._option_chain_cache: dict[int, tuple[OptionChainDefinition, ...]] = {}
        self._active_market_data: dict[
            tuple[int, str, str, str, int], _ActiveMarketDataSubscription
        ] = {}
        self._next_local_request_id = 1
        self._resource_counter_date = datetime.now(tz=UTC).date()
        self._market_data_requests_today = 0
        self._deduplicated_requests_today = 0
        self._capacity_rejects_today = 0
        self._scanner_semaphore = asyncio.Semaphore(MAX_ACTIVE_SCANNERS)
        self._historical_semaphore = asyncio.Semaphore(HISTORICAL_REQUEST_CONCURRENCY)
        self._active_scanners = 0
        self._pending_historical_work = 0
        self._historical_requests_today = 0
        self._scanner_requests_today = 0
        self._pacing_violations_today = 0
        self._last_resource_error: str | None = None
        self._observe_resource_errors()

    def resource_status(self) -> IbkrResourceStatus:
        """Return only connection-local resource facts Stocker can observe."""

        self._reset_daily_resource_counters()
        client = getattr(self._client, "client", None)
        max_requests = getattr(client, "MaxRequests", None)
        requests_interval = getattr(client, "RequestsInterval", None)
        subscriptions = tuple(
            MarketDataSubscriptionStatus(
                request_id=item.request_id,
                con_id=item.con_id,
                security_type=item.security_type,
                exchange=item.exchange,
                purpose=item.purpose,
                snapshot=False,
                created_at=item.created_at,
                consumer_count=item.consumer_count,
            )
            for item in sorted(
                self._active_market_data.values(), key=lambda value: value.request_id
            )
        )
        return IbkrResourceStatus(
            market_data_line_budget=self.config.market_data_line_budget,
            ib_async_max_requests=(int(max_requests) if isinstance(max_requests, int) else None),
            ib_async_requests_interval=(
                float(requests_interval) if isinstance(requests_interval, (int, float)) else None
            ),
            active_market_data_lines=len(subscriptions),
            active_underlying_lines=sum(item.security_type != "OPT" for item in subscriptions),
            active_option_lines=sum(item.security_type == "OPT" for item in subscriptions),
            subscriptions=subscriptions,
            market_data_requests_today=self._market_data_requests_today,
            deduplicated_requests_today=self._deduplicated_requests_today,
            capacity_rejects_today=self._capacity_rejects_today,
            active_scanners=self._active_scanners,
            pending_historical_work=self._pending_historical_work,
            historical_concurrency_limit=HISTORICAL_REQUEST_CONCURRENCY,
            historical_requests_today=self._historical_requests_today,
            scanner_requests_today=self._scanner_requests_today,
            pacing_state=(
                "THROTTLING"
                if bool(getattr(client, "_isThrottling", False))
                else "VIOLATION_RECORDED"
                if self._pacing_violations_today
                else "OK"
            ),
            pacing_violations_today=self._pacing_violations_today,
            last_resource_error=self._last_resource_error,
        )

    def _configure_ib_async_throttle(self) -> int | None:
        client = getattr(self._client, "client", None)
        if client is None:
            return None
        configured = getattr(client, "MaxRequests", None)
        if not isinstance(configured, int) or configured <= 0:
            return None
        self._set_ib_async_throttle(configured)
        return configured

    def _apply_ib_async_throttle(self) -> None:
        """Reapply Stocker's budget without losing ib_async's native ceiling."""

        if self._library_max_requests is None:
            return
        self._set_ib_async_throttle(self._library_max_requests)

    def _set_ib_async_throttle(self, library_ceiling: int) -> None:
        client = getattr(self._client, "client", None)
        if client is None:
            return
        derived = max(1, self.config.market_data_line_budget // 2)
        client.MaxRequests = min(library_ceiling, derived)

    def _reset_daily_resource_counters(self, today: date | None = None) -> None:
        """Keep operational counters scoped to the current UTC day."""

        current = today or datetime.now(tz=UTC).date()
        if current == self._resource_counter_date:
            return
        self._resource_counter_date = current
        self._market_data_requests_today = 0
        self._deduplicated_requests_today = 0
        self._capacity_rejects_today = 0
        self._historical_requests_today = 0
        self._scanner_requests_today = 0
        self._pacing_violations_today = 0

    def _observe_resource_errors(self) -> None:
        event = getattr(self._client, "errorEvent", None)
        if event is not None:
            event += self._record_resource_error

    def _record_resource_error(
        self,
        _request_id: int,
        code: int,
        message: str,
        _contract: object,
        *_extra: object,
    ) -> None:
        normalized = str(message).lower()
        pacing = (
            code == 100
            or "pacing violation" in normalized
            or (code in {162, 420} and "pacing" in normalized)
        )
        capacity = code == 101 or any(
            phrase in normalized
            for phrase in (
                "market data lines",
                "market data subscriptions reached",
                "scanner subscription limit",
                "too many scanner",
            )
        )
        entitlement = code == 354 or any(
            phrase in normalized
            for phrase in ("not subscribed", "market data permission", "not entitled")
        )
        if not (pacing or capacity or entitlement):
            return
        self._reset_daily_resource_counters()
        if pacing:
            self._pacing_violations_today += 1
        if capacity:
            self._capacity_rejects_today += 1
        self._last_resource_error = f"{int(code)}: {sanitize_ibkr_message(message)}"

    def _acquire_market_data_stream(
        self,
        contract: object,
        *,
        generic_tick_list: str,
        market_data_type: int,
        purpose: str,
    ) -> tuple[tuple[int, str, str, str, int], object]:
        self._reset_daily_resource_counters()
        con_id = int(cast(Any, contract).conId)
        security_type = str(getattr(contract, "secType", "")).upper()
        exchange = str(getattr(contract, "exchange", "")).upper()
        key = (con_id, security_type, exchange, generic_tick_list, market_data_type)
        existing = self._active_market_data.get(key)
        if existing is not None:
            existing.consumer_count += 1
            self._deduplicated_requests_today += 1
            return key, existing.ticker
        if len(self._active_market_data) >= self.config.market_data_line_budget:
            self._capacity_rejects_today += 1
            self._last_resource_error = "IBKR_MARKET_DATA_CAPACITY_UNAVAILABLE"
            raise IbkrError("IBKR_MARKET_DATA_CAPACITY_UNAVAILABLE")
        ticker = self._client.reqMktData(
            contract,
            genericTickList=generic_tick_list,
            snapshot=False,
            regulatorySnapshot=False,
        )
        request_id = self._market_data_request_id(ticker)
        self._active_market_data[key] = _ActiveMarketDataSubscription(
            request_id=request_id,
            contract=contract,
            ticker=ticker,
            con_id=con_id,
            security_type=security_type,
            exchange=exchange,
            purpose=purpose,
            created_at=datetime.now(tz=UTC),
        )
        self._market_data_requests_today += 1
        return key, ticker

    def _release_market_data_stream(self, key: tuple[int, str, str, str, int]) -> None:
        active = self._active_market_data.get(key)
        if active is None:
            return
        active.consumer_count -= 1
        if active.consumer_count > 0:
            return
        try:
            self._client.cancelMktData(active.contract)
        finally:
            self._active_market_data.pop(key, None)

    def _cancel_all_market_data_streams(self) -> None:
        active = tuple(self._active_market_data.values())
        self._active_market_data.clear()
        for item in active:
            with suppress(Exception):
                self._client.cancelMktData(item.contract)

    def _market_data_request_id(self, ticker: object) -> int:
        wrapper = getattr(self._client, "wrapper", None)
        for request_id, candidate in getattr(wrapper, "reqId2Ticker", {}).items():
            if candidate is ticker:
                return int(request_id)
        request_id = self._next_local_request_id
        self._next_local_request_id += 1
        return request_id

    @property
    def environment(self) -> Environment:
        return self.config.environment

    @property
    def account(self) -> str:
        return self._account_id or ""

    @property
    def connection_epoch(self) -> int:
        return self._connection_epoch

    @property
    def is_connected(self) -> bool:
        return self._client.isConnected()

    async def connect(self) -> BrokerSession:
        """Connect and verify the configured environment against the account."""

        try:
            await self._client.connectAsync(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.connect_timeout_seconds,
                readonly=not self._execution_enabled,
                account=self.config.expected_account or "",
                raiseSyncErrors=True,
                fetchFields=_no_startup_fetches(),
            )
            if not self.is_connected:
                raise IbkrError("IBKR reported no active API connection after connecting")
            account_id = self._select_account(self._client.managedAccounts())
            self._verify_environment(account_id)
            self._account_id = account_id
            self._connection_epoch += 1
            self._open_orders_loaded = False
            self._completed_orders_loaded = False
            return BrokerSession(
                environment=self.config.environment,
                account_id=account_id,
                connected=True,
            )
        except Exception as exc:
            self.disconnect()
            if isinstance(exc, IbkrError):
                raise
            raise IbkrError(
                f"IBKR {self.config.environment.value} connection failed at "
                f"{self.config.host}:{self.config.port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Disconnect this instance and clear its confirmed account state."""

        was_connected = self._client.isConnected() or self._account_id is not None
        self._cancel_all_market_data_streams()
        if self._client.isConnected():
            self._client.disconnect()
        self._account_id = None
        self._open_orders_loaded = False
        self._completed_orders_loaded = False
        self._scanner_capabilities = None
        self._qualified_stock_cache.clear()
        self._option_chain_cache.clear()
        if was_connected:
            self._connection_epoch += 1

    def reconfigure(self, config: IbkrConfig) -> None:
        """Replace validated connection settings while this socket is disconnected."""

        if self.is_connected:
            raise IbkrError("IBKR connection must be disconnected before reconfiguration")
        if config.environment is not self.environment:
            raise ValueError("IBKR reconfiguration cannot change execution environment")
        self.config = config
        self._apply_ib_async_throttle()

    @contextmanager
    def capture_api_errors(self) -> Iterator[list[IbkrApiError]]:
        """Capture sanitized IBKR error callbacks for one diagnostic request scope."""

        errors: list[IbkrApiError] = []
        event = getattr(self._client, "errorEvent", None)
        if event is None:
            yield errors
            return

        def capture(request_id: int, code: int, message: str, contract: object) -> None:
            errors.append(
                IbkrApiError(
                    request_id=int(request_id),
                    code=int(code),
                    message=sanitize_ibkr_message(message),
                    con_id=_optional_contract_integer(contract, "conId"),
                    symbol=_optional_contract_text(contract, "symbol"),
                    exchange=_optional_contract_text(contract, "exchange"),
                )
            )

        event += capture
        try:
            yield errors
        finally:
            event -= capture

    async def account_state(self) -> BrokerAccountState:
        """Read authoritative equity, buying power, and positions for this session."""

        self._require_connected()
        try:
            values = await asyncio.wait_for(
                self._client.accountSummaryAsync(self.account),
                timeout=self.config.request_timeout_seconds,
            )
            positions = await self.read_positions()
        except TimeoutError as exc:
            raise IbkrError("IBKR account state request timed out") from exc
        except Exception as exc:
            if isinstance(exc, IbkrError):
                raise
            raise IbkrError(f"IBKR account state request failed: {exc}") from exc
        equity = _account_value(values, self.account, "NetLiquidation")
        buying_power = _account_value(values, self.account, "BuyingPower")
        return BrokerAccountState(
            environment=self.environment,
            account=self.account,
            equity=equity,
            buying_power=buying_power,
            connected=self.is_connected,
            positions=positions,
        )

    async def minimum_tick(self, instrument: QualifiedInstrument) -> float:
        """Read the qualified contract's IBKR minimum price increment."""

        self._require_connected()
        try:
            details = await self._client.reqContractDetailsAsync(_to_ib_contract(instrument))
            if len(details) != 1:
                raise IbkrError(
                    f"IBKR returned {len(details)} contract details for {instrument.symbol}"
                )
            tick = float(cast(Any, details[0]).minTick)
        except Exception as exc:
            if isinstance(exc, IbkrError):
                raise
            raise IbkrError(
                f"IBKR minimum tick request failed for {instrument.symbol}: {exc}"
            ) from exc
        if not isfinite(tick) or tick <= 0.0:
            raise IbkrError(f"IBKR returned invalid minimum tick for {instrument.symbol}")
        return tick

    async def submit_protected_order(
        self, plan: OrderPlan, instrument: QualifiedInstrument
    ) -> BrokerOrderIds:
        """Transmit one protected bracket to this explicitly configured account."""

        self._require_connected()
        if not self._execution_enabled:
            raise IbkrError("IBKR connection is read-only; execution was not enabled")
        if (
            plan.environment is not self.environment
            or self.config.expected_account is None
            or self.account != self.config.expected_account
            or instrument.con_id != plan.con_id
        ):
            raise IbkrError("ACCOUNT_OR_ENVIRONMENT_MISMATCH")
        if plan.side is not OrderAction.SELL:
            raise IbkrError("Stage 7 supports only the first strategy's SHORT order plans")

        from ib_async import LimitOrder, MarketOrder, StopOrder

        parent_id = int(self._client.client.getReqId())
        target_id = int(self._client.client.getReqId())
        stop_id = int(self._client.client.getReqId())
        common = {"orderRef": plan.order_plan_id, "account": self.account}
        parent = MarketOrder(
            "SELL", plan.quantity, orderId=parent_id, transmit=False, tif="DAY", **common
        )
        target = LimitOrder(
            "BUY",
            plan.quantity,
            plan.target_price,
            orderId=target_id,
            parentId=parent_id,
            transmit=False,
            tif="GTC",
            **common,
        )
        stop = StopOrder(
            "BUY",
            plan.quantity,
            plan.stop_price,
            orderId=stop_id,
            parentId=parent_id,
            transmit=True,
            tif="GTC",
            **common,
        )
        contract = _to_ib_contract(instrument)
        placed: list[object] = []
        try:
            for bracket_order in (parent, target, stop):
                self._client.placeOrder(contract, bracket_order)
                placed.append(bracket_order)
        except Exception as exc:
            for placed_order in placed:
                with suppress(Exception):
                    self._client.cancelOrder(placed_order)
            raise IbkrError(
                f"IBKR protected {self.environment.value} order submission failed: {exc}"
            ) from exc
        return BrokerOrderIds(parent=parent_id, stop=stop_id, target=target_id)

    async def read_open_orders(self) -> tuple[BrokerOpenOrder, ...]:
        """Read and normalize all open orders visible to this IBKR session."""

        self._require_connected()
        try:
            if not self._open_orders_loaded:
                trades = await asyncio.wait_for(
                    self._client.reqAllOpenOrdersAsync(),
                    timeout=self.config.request_timeout_seconds,
                )
                self._open_orders_loaded = True
            else:
                trades = self._client.openTrades()
            return tuple(self._normalize_open_order(trade) for trade in trades)
        except TimeoutError as exc:
            raise IbkrError("IBKR open-order request timed out") from exc
        except Exception as exc:
            if isinstance(exc, IbkrError):
                raise
            raise IbkrError(f"IBKR open-order request failed: {exc}") from exc

    async def read_order_statuses(self) -> tuple[BrokerOrderStatus, ...]:
        """Read normalized open and completed order states, including rejections."""

        self._require_connected()
        try:
            if not self._open_orders_loaded:
                await asyncio.wait_for(
                    self._client.reqAllOpenOrdersAsync(),
                    timeout=self.config.request_timeout_seconds,
                )
                self._open_orders_loaded = True
            if not self._completed_orders_loaded:
                await asyncio.wait_for(
                    self._client.reqCompletedOrdersAsync(apiOnly=False),
                    timeout=self.config.request_timeout_seconds,
                )
                self._completed_orders_loaded = True
            trades = self._client.trades()
        except TimeoutError as exc:
            raise IbkrError("IBKR order-status request timed out") from exc
        except Exception as exc:
            raise IbkrError(f"IBKR order-status request failed: {exc}") from exc
        by_order: dict[int, BrokerOrderStatus] = {}
        for trade in trades:
            source = cast(Any, trade)
            order = source.order
            status = source.orderStatus
            if str(order.account) != self.account:
                continue
            messages = [str(item.message) for item in getattr(source, "log", ()) if item.message]
            normalized = BrokerOrderStatus(
                order_id=int(order.orderId),
                order_plan_id=str(order.orderRef),
                account=str(order.account),
                environment=self.environment,
                status=_order_lifecycle(str(status.status)),
                filled_quantity=float(status.filled),
                remaining_quantity=float(status.remaining),
                reason=messages[-1] if messages else "",
            )
            by_order[normalized.order_id] = normalized
        return tuple(by_order[order_id] for order_id in sorted(by_order))

    async def read_positions(self) -> tuple[BrokerPosition, ...]:
        """Read normalized actual positions for the connected account."""

        self._require_connected()
        try:
            sources = await asyncio.wait_for(
                self._client.reqPositionsAsync(),
                timeout=self.config.request_timeout_seconds,
            )
        except TimeoutError as exc:
            raise IbkrError("IBKR position request timed out") from exc
        except Exception as exc:
            raise IbkrError(f"IBKR position request failed: {exc}") from exc
        positions: list[BrokerPosition] = []
        for source_object in sources:
            source = cast(Any, source_object)
            if str(source.account) != self.account:
                continue
            positions.append(
                BrokerPosition(
                    account=str(source.account),
                    con_id=int(source.contract.conId),
                    symbol=str(source.contract.symbol),
                    quantity=float(source.position),
                    average_price=float(source.avgCost),
                )
            )
        return tuple(positions)

    async def read_fills(self) -> tuple[BrokerFill, ...]:
        """Read normalized executions; execution IDs make repeated reads idempotent."""

        self._require_connected()
        try:
            sources = await asyncio.wait_for(
                self._client.reqExecutionsAsync(),
                timeout=self.config.request_timeout_seconds,
            )
        except TimeoutError as exc:
            raise IbkrError("IBKR execution request timed out") from exc
        except Exception as exc:
            raise IbkrError(f"IBKR execution request failed: {exc}") from exc
        fills: list[BrokerFill] = []
        for source_object in sources:
            source = cast(Any, source_object)
            execution = source.execution
            if str(execution.acctNumber) != self.account:
                continue
            timestamp = execution.time
            if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
                raise IbkrError("IBKR returned an execution without a timezone-aware timestamp")
            side_text = str(execution.side).upper()
            side = OrderAction.SELL if side_text in {"SLD", "SELL"} else OrderAction.BUY
            report = getattr(source, "commissionReport", None)
            commission = _optional_number(getattr(report, "commission", None))
            fills.append(
                BrokerFill(
                    execution_id=str(execution.execId),
                    order_id=int(execution.orderId),
                    account=str(execution.acctNumber),
                    environment=self.environment,
                    con_id=int(source.contract.conId),
                    symbol=str(source.contract.symbol),
                    side=side,
                    quantity=float(execution.shares),
                    price=float(execution.price),
                    executed_at=timestamp,
                    commission=commission,
                    order_plan_id=str(getattr(execution, "orderRef", "")),
                )
            )
        return tuple(fills)

    async def cancel_order(self, order_id: int) -> None:
        """Cancel one explicitly named open order."""

        self._require_connected()
        trades = await self._client.reqAllOpenOrdersAsync()
        for trade in trades:
            order = cast(Any, trade).order
            if int(order.orderId) == order_id and str(order.account) == self.account:
                self._client.cancelOrder(order)
                return
        raise IbkrError(f"IBKR open order {order_id} was not found for the connected account")

    def _normalize_open_order(self, trade: object) -> BrokerOpenOrder:
        source = cast(Any, trade)
        order = source.order
        contract = source.contract
        order_type = str(order.orderType).upper()
        if int(order.parentId) == 0:
            role = OrderRole.ENTRY
        elif order_type in {"STP", "STP LMT"}:
            role = OrderRole.STOP
        else:
            role = OrderRole.TARGET
        return BrokerOpenOrder(
            order_id=int(order.orderId),
            order_plan_id=str(order.orderRef),
            account=str(order.account),
            environment=self.environment,
            con_id=int(contract.conId),
            symbol=str(contract.symbol),
            role=role,
            status=_order_lifecycle(str(source.orderStatus.status)),
        )

    async def resolve_stock(
        self,
        symbol: str,
        *,
        exchange: str,
        currency: str,
        primary_exchange: str | None = None,
    ) -> QualifiedInstrument:
        """Qualify exactly one stock contract; never guess between ambiguous matches."""

        self._require_connected()
        normalized_symbol = symbol.strip().upper()
        normalized_exchange = exchange.strip().upper()
        normalized_currency = currency.strip().upper()
        normalized_primary_exchange = (primary_exchange or "").strip().upper()
        if not normalized_symbol or not normalized_exchange or not normalized_currency:
            raise IbkrError("Stock symbol, exchange, and currency are required")
        cache_key = (
            normalized_symbol,
            normalized_exchange,
            normalized_currency,
            normalized_primary_exchange,
        )
        cached = self._qualified_stock_cache.get(cache_key)
        if cached is not None:
            self._reset_daily_resource_counters()
            self._deduplicated_requests_today += 1
            return cached

        from ib_async import Stock

        requested = Stock(
            normalized_symbol,
            normalized_exchange,
            normalized_currency,
            primaryExchange=normalized_primary_exchange,
        )
        try:
            results = await asyncio.wait_for(
                self._client.qualifyContractsAsync(requested, returnAll=True),
                timeout=self.config.request_timeout_seconds,
            )
        except Exception as exc:
            raise IbkrError(
                f"IBKR contract resolution failed for {normalized_symbol}: {exc}"
            ) from exc

        if len(results) != 1 or results[0] is None:
            raise IbkrError(f"IBKR stock contract {normalized_symbol} could not be resolved")
        result = results[0]
        if isinstance(result, list):
            raise IbkrError(
                f"IBKR stock contract {normalized_symbol} resolved ambiguously "
                f"to {len(result)} contracts"
            )

        contract = cast(_QualifiedContract, result)
        if contract.secType != "STK" or contract.conId <= 0:
            raise IbkrError(
                f"IBKR returned an invalid qualified stock contract for {normalized_symbol}"
            )
        qualified = QualifiedInstrument(
            symbol=contract.symbol,
            con_id=contract.conId,
            exchange=contract.exchange,
            primary_exchange=contract.primaryExchange or None,
            currency=contract.currency,
            security_type=contract.secType,
        )
        self._qualified_stock_cache[cache_key] = qualified
        return qualified

    async def hot_us_stocks_by_volume(self, *, max_results: int = 50) -> tuple[str, ...]:
        """Return one finite IBKR US-stock volume scan, ordered by scanner rank."""

        self._require_connected()
        if not 1 <= max_results <= 50:
            raise ValueError("IBKR market scanners support between 1 and 50 results")

        from ib_async import ScannerSubscription

        subscription = ScannerSubscription(
            numberOfRows=max_results,
            instrument="STK",
            locationCode="STK.US.MAJOR",
            scanCode="HOT_BY_VOLUME",
        )
        try:
            rows = await self._bounded_scanner_data(subscription)
        except Exception as exc:
            raise IbkrError(
                f"IBKR HOT_BY_VOLUME scanner request failed: {sanitize_ibkr_message(exc)}"
            ) from exc

        symbols: list[str] = []
        seen: set[str] = set()
        for row in sorted(rows, key=lambda item: int(cast(Any, item).rank)):
            contract = cast(Any, row).contractDetails.contract
            symbol = str(contract.symbol).strip().upper()
            if str(contract.secType).upper() != "STK" or not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
        return tuple(symbols[:max_results])

    async def _bounded_scanner_data(
        self,
        subscription: object,
        filter_options: list[object] | None = None,
    ) -> list[object]:
        """Collect one scan and guarantee broker-side cancellation on every exit."""

        self._reset_daily_resource_counters()
        async with self._scanner_semaphore:
            self._active_scanners += 1
            self._scanner_requests_today += 1
            try:
                request = getattr(self._client, "reqScannerSubscription", None)
                cancel = getattr(self._client, "cancelScannerSubscription", None)
                wrapper = getattr(self._client, "wrapper", None)
                if callable(request) and callable(cancel) and wrapper is not None:
                    data_list = request(subscription, [], filter_options or [])
                    future = wrapper.startReq(data_list.reqId, container=data_list)
                    try:
                        result = await asyncio.wait_for(
                            future,
                            timeout=self.config.request_timeout_seconds,
                        )
                        return list(result)
                    finally:
                        cancel(data_list)
                        end_request = getattr(wrapper, "_endReq", None)
                        if callable(end_request):
                            end_request(data_list.reqId)
                if filter_options is None:
                    result = await asyncio.wait_for(
                        self._client.reqScannerDataAsync(subscription),
                        timeout=self.config.request_timeout_seconds,
                    )
                else:
                    result = await asyncio.wait_for(
                        self._client.reqScannerDataAsync(subscription, [], filter_options),
                        timeout=self.config.request_timeout_seconds,
                    )
                return list(result)
            finally:
                self._active_scanners -= 1

    async def scanner_capabilities(self) -> ScannerCapabilities:
        """Discover and cache the connected broker's finite scanner vocabulary."""

        self._require_connected()
        if self._scanner_capabilities is not None:
            return self._scanner_capabilities
        try:
            payload = await asyncio.wait_for(
                self._client.reqScannerParametersAsync(),
                timeout=self.config.request_timeout_seconds,
            )
            root = ElementTree.fromstring(payload)
        except Exception as exc:
            raise IbkrError(
                f"IBKR scanner-parameter discovery failed: {sanitize_ibkr_message(exc)}"
            ) from exc
        values: dict[str, set[str]] = {
            "locations": set(),
            "scan_codes": set(),
            "filters": set(),
        }
        location_scan_codes: dict[str, set[str]] = {}
        location_filters: dict[str, set[str]] = {}
        location_instruments: dict[str, set[str]] = {}
        for element in root.iter():
            text = (element.text or "").strip()
            tag = element.tag.rsplit("}", maxsplit=1)[-1]
            if tag == "locationCode" and text:
                values["locations"].add(text)
            elif tag == "scanCode" and text:
                values["scan_codes"].add(text)
            elif tag in {"code", "fieldCode", "filterCode"} and text:
                values["filters"].add(text)
            direct_location = next(
                (
                    (child.text or "").strip()
                    for child in element
                    if child.tag.rsplit("}", maxsplit=1)[-1] == "locationCode"
                    and (child.text or "").strip()
                ),
                None,
            )
            if direct_location is not None:
                local_codes = {
                    (child.text or "").strip()
                    for child in element.iter()
                    if child.tag.rsplit("}", maxsplit=1)[-1] == "scanCode"
                    and (child.text or "").strip()
                }
                local_filters = {
                    (child.text or "").strip()
                    for child in element.iter()
                    if child.tag.rsplit("}", maxsplit=1)[-1] in {"code", "fieldCode", "filterCode"}
                    and (child.text or "").strip()
                }
                local_instruments = {
                    value.strip()
                    for child in element
                    if child.tag.rsplit("}", maxsplit=1)[-1] == "instruments"
                    for value in (child.text or "").replace(";", ",").split(",")
                    if value.strip()
                }
                if local_codes:
                    location_scan_codes.setdefault(direct_location, set()).update(local_codes)
                if local_filters:
                    location_filters.setdefault(direct_location, set()).update(local_filters)
                if local_instruments:
                    location_instruments.setdefault(direct_location, set()).update(
                        local_instruments
                    )
        discovered = ScannerCapabilities(
            locations=frozenset(values["locations"]),
            scan_codes=frozenset(values["scan_codes"]),
            filters=frozenset(values["filters"]),
            location_scan_codes={
                location: frozenset(codes) for location, codes in location_scan_codes.items()
            },
            location_filters={
                location: frozenset(filters) for location, filters in location_filters.items()
            },
            location_instruments={
                location: frozenset(instruments)
                for location, instruments in location_instruments.items()
            },
        )
        self._scanner_capabilities = discovered
        return discovered

    async def activity_scan(
        self,
        *,
        market: MarketDefinition,
        cap_bucket: CapBucket,
        component: ActivityScanner,
        max_results: int = 50,
    ) -> tuple[ScannerCandidate, ...]:
        """Run one bounded market/cap-restricted Activity Shortlist component."""

        self._require_connected()
        if not 1 <= max_results <= 50:
            raise ValueError("IBKR market scanners support between 1 and 50 results")
        capabilities = await self.scanner_capabilities()
        if market.scanner_location not in capabilities.locations:
            raise IbkrError("SCANNER_NOT_AVAILABLE")
        if not capabilities.supports_instrument(market.scanner_location, market.scanner_instrument):
            raise IbkrError("SCANNER_NOT_AVAILABLE")
        if component.value not in capabilities.scan_codes_for(market.scanner_location):
            raise IbkrError("SCANNER_NOT_AVAILABLE")
        cap = CAP_BUCKETS_V1.definition(cap_bucket)
        if cap_bucket is not CapBucket.ALL:
            filters = capabilities.filters_for(market.scanner_location)
            above_available = bool(
                {"marketCapAbove", "usdMarketCapAbove", "marketCapAbove1e6"} & filters
            )
            below_available = bool(
                {"marketCapBelow", "usdMarketCapBelow", "marketCapBelow1e6"} & filters
            )
            if not above_available or (
                cap.maximum_usd_exclusive is not None and not below_available
            ):
                raise IbkrError("CAP_FILTER_UNAVAILABLE")

        from ib_async import ScannerSubscription, TagValue

        subscription = ScannerSubscription(
            numberOfRows=max_results,
            instrument=market.scanner_instrument,
            locationCode=market.scanner_location,
            scanCode=component.value,
        )
        filter_options: list[object] = []
        filters = capabilities.filters_for(market.scanner_location)
        if cap.scanner_minimum_millions is not None:
            if "usdMarketCapAbove" in filters:
                filter_options.append(
                    TagValue("usdMarketCapAbove", str(cap.scanner_minimum_millions))
                )
            elif "marketCapAbove1e6" in filters:
                filter_options.append(
                    TagValue("marketCapAbove1e6", f"{cap.scanner_minimum_millions:g}")
                )
            else:
                subscription.marketCapAbove = cap.scanner_minimum_millions
        if cap.scanner_maximum_millions is not None:
            if "usdMarketCapBelow" in filters:
                filter_options.append(
                    TagValue("usdMarketCapBelow", str(cap.scanner_maximum_millions))
                )
            elif "marketCapBelow1e6" in filters:
                filter_options.append(
                    TagValue("marketCapBelow1e6", f"{cap.scanner_maximum_millions:g}")
                )
            else:
                subscription.marketCapBelow = cap.scanner_maximum_millions
        try:
            rows = await self._bounded_scanner_data(subscription, filter_options)
        except Exception as exc:
            raise IbkrError(
                f"IBKR {component.value} scanner request failed: {sanitize_ibkr_message(exc)}"
            ) from exc
        results: list[ScannerCandidate] = []
        seen: set[tuple[str, int | None]] = set()
        for raw in sorted(rows, key=lambda item: int(cast(Any, item).rank)):
            contract = cast(Any, raw).contractDetails.contract
            symbol = str(contract.symbol).strip().upper()
            con_id = int(contract.conId) if int(getattr(contract, "conId", 0)) > 0 else None
            identity = (symbol, con_id)
            if str(contract.secType).upper() != "STK" or not symbol or identity in seen:
                continue
            seen.add(identity)
            results.append(
                ScannerCandidate(
                    component=component,
                    rank=int(cast(Any, raw).rank) + 1,
                    symbol=symbol,
                    con_id=con_id,
                    exchange=str(contract.exchange or "SMART").upper(),
                    primary_exchange=(
                        str(contract.primaryExchange).upper()
                        if getattr(contract, "primaryExchange", "")
                        else None
                    ),
                    currency=str(contract.currency or market.currency).upper(),
                )
            )
        return tuple(results[:max_results])

    async def historical_bars(
        self,
        instrument: QualifiedInstrument,
        *,
        bar_size: str,
        duration: str,
        what_to_show: str,
        regular_trading_hours: bool,
        end_time: date | datetime | None = None,
        minimum_bars: int = 1,
    ) -> tuple[HistoricalBar, ...]:
        """Request and validate one raw IBKR historical-bar response."""

        self._require_connected()
        if not bar_size.strip() or not duration.strip() or not what_to_show.strip():
            raise IbkrError("Historical bar size, duration, and data type are required")
        if minimum_bars < 1:
            raise IbkrError("Historical minimum_bars must be at least 1")

        self._pending_historical_work += 1
        self._reset_daily_resource_counters()
        try:
            async with self._historical_semaphore:
                self._historical_requests_today += 1
                source_bars = await self._client.reqHistoricalDataAsync(
                    _to_ib_contract(instrument),
                    endDateTime=end_time or "",
                    durationStr=duration.strip(),
                    barSizeSetting=bar_size.strip(),
                    whatToShow=what_to_show.strip().upper(),
                    useRTH=regular_trading_hours,
                    formatDate=2,
                    keepUpToDate=False,
                    timeout=self.config.request_timeout_seconds,
                )
        except Exception as exc:
            raise IbkrError(
                f"IBKR historical data request failed for {instrument.symbol}: {exc}"
            ) from exc
        finally:
            self._pending_historical_work -= 1

        if not source_bars:
            raise IbkrError(f"IBKR historical data returned no bars for {instrument.symbol}")
        bars = tuple(
            _normalize_historical_bar(cast(_SourceBar, bar), index=index)
            for index, bar in enumerate(source_bars)
        )
        if len(bars) < minimum_bars:
            raise IbkrError(
                f"IBKR historical data returned {len(bars)} bars; "
                f"at least {minimum_bars} required for {instrument.symbol}"
            )
        try:
            timestamps_increase = all(
                current.timestamp > previous.timestamp for previous, current in pairwise(bars)
            )
        except TypeError as exc:
            raise IbkrError(
                f"IBKR historical timestamps are incompatible for {instrument.symbol}"
            ) from exc
        if not timestamps_increase:
            raise IbkrError(
                f"IBKR historical timestamps are not strictly increasing for {instrument.symbol}"
            )
        return bars

    async def current_quote(
        self, instrument: QualifiedInstrument, *, market_data_type: int = 1
    ) -> CurrentQuote:
        """Request one finite IBKR market-data snapshot for a qualified instrument."""

        self._require_connected()
        if market_data_type not in {1, 2, 3, 4}:
            raise IbkrError("IBKR market data type must be one of 1, 2, 3, or 4")
        try:
            self._client.reqMarketDataType(market_data_type)
            self._reset_daily_resource_counters()
            self._market_data_requests_today += 1
            tickers = await asyncio.wait_for(
                self._client.reqTickersAsync(_to_ib_contract(instrument), regulatorySnapshot=False),
                timeout=self.config.request_timeout_seconds,
            )
        except Exception as exc:
            raise IbkrError(
                f"IBKR current market data request failed for {instrument.symbol}: {exc}"
            ) from exc
        if len(tickers) != 1:
            raise IbkrError(
                f"IBKR current market data returned {len(tickers)} snapshots "
                f"for {instrument.symbol}"
            )

        ticker = cast(_SourceTicker, tickers[0])
        try:
            timestamp = ticker.time
            bid = _available_price(ticker.bid)
            ask = _available_price(ticker.ask)
            last = _available_price(ticker.last)
            close = _available_price(ticker.close)
        except (AttributeError, TypeError, ValueError) as exc:
            raise IbkrError(
                f"IBKR returned an invalid current market data snapshot for {instrument.symbol}"
            ) from exc
        if timestamp is not None and not isinstance(timestamp, datetime):
            raise IbkrError(
                f"IBKR returned an invalid current market data timestamp for {instrument.symbol}"
            )
        if all(price is None for price in (bid, ask, last)):
            raise IbkrError(
                f"IBKR current market data for {instrument.symbol} "
                "contained no current bid, ask, or last"
            )
        return CurrentQuote(
            symbol=instrument.symbol,
            con_id=instrument.con_id,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            last=last,
            close=close,
            market_data_type=_optional_integer(getattr(ticker, "marketDataType", None)),
        )

    async def option_snapshots(
        self,
        options: tuple[QualifiedOption, ...],
        *,
        market_data_type: int = 1,
        purpose: str = "OPTION_PRE_CONTEXT",
    ) -> tuple[OptionMarketSnapshot, ...]:
        """Capture tick-13 model computations plus generic-101 open interest."""

        self._require_connected()
        if not options:
            raise IbkrError("At least one qualified option is required")
        if market_data_type not in {1, 2, 3, 4}:
            raise IbkrError("IBKR market data type must be one of 1, 2, 3, or 4")
        unique_options = tuple({option.con_id: option for option in options}.values())
        contracts = tuple(_to_ib_option_contract(option) for option in unique_options)
        acquired: list[tuple[int, str, str, str, int]] = []
        tickers: tuple[object, ...] = ()
        try:
            self._client.reqMarketDataType(market_data_type)
            requested: list[object] = []
            for contract in contracts:
                key, ticker = self._acquire_market_data_stream(
                    contract,
                    generic_tick_list="101",
                    market_data_type=market_data_type,
                    purpose=purpose,
                )
                acquired.append(key)
                requested.append(ticker)
            tickers = tuple(requested)
            deadline = asyncio.get_running_loop().time() + self.config.request_timeout_seconds
            while not all(
                _option_snapshot_complete(
                    cast(_SourceOptionTicker, ticker),
                    option,
                    allow_delayed_model=market_data_type in {3, 4},
                )
                for ticker, option in zip(tickers, unique_options, strict=True)
            ):
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.05)
        except Exception as exc:
            if isinstance(exc, IbkrError):
                raise
            raise IbkrError(f"IBKR option market data request failed: {exc}") from exc
        finally:
            for key in reversed(acquired):
                self._release_market_data_stream(key)

        captured_at = datetime.now(tz=UTC)
        normalized: list[OptionMarketSnapshot] = []
        for option, source in zip(unique_options, tickers, strict=True):
            ticker = cast(_SourceOptionTicker, source)
            reported_market_data_type = _optional_integer(getattr(ticker, "marketDataType", None))
            accepted_model_types = {1, 2, 3, 4} if market_data_type in {3, 4} else {1, 2}
            model = (
                getattr(ticker, "modelGreeks", None)
                if reported_market_data_type in accepted_model_types
                else None
            )
            normalized.append(
                OptionMarketSnapshot(
                    option=option,
                    captured_at=captured_at,
                    bid=_optional_number(getattr(ticker, "bid", None)),
                    ask=_optional_number(getattr(ticker, "ask", None)),
                    open_interest=_option_open_interest(ticker, option),
                    market_data_type=reported_market_data_type,
                    model_iv=_optional_number(getattr(model, "impliedVol", None)),
                    model_delta=_optional_number(getattr(model, "delta", None)),
                    model_gamma=_optional_number(getattr(model, "gamma", None)),
                )
            )
        return tuple(normalized)

    async def historical_volatility_snapshot(
        self,
        instrument: QualifiedInstrument,
        *,
        market_data_type: int = 1,
        purpose: str = "SESSION_HARD_HV",
    ) -> HistoricalVolatilitySnapshot:
        """Capture stock generic tick 104 and always release its temporary line."""

        self._require_connected()
        if instrument.security_type != "STK":
            raise IbkrError("HV_NOT_READY: historical volatility requires a stock")
        if market_data_type not in {1, 2}:
            raise IbkrError("HV_NOT_READY: historical volatility requires live or frozen data")
        key: tuple[int, str, str, str, int] | None = None
        ticker: object | None = None
        try:
            self._client.reqMarketDataType(market_data_type)
            key, ticker = self._acquire_market_data_stream(
                _to_ib_contract(instrument),
                generic_tick_list="104",
                market_data_type=market_data_type,
                purpose=purpose,
            )
            deadline = asyncio.get_running_loop().time() + self.config.request_timeout_seconds
            while True:
                value = _optional_number(getattr(ticker, "histVolatility", None))
                reported_type = _optional_integer(getattr(ticker, "marketDataType", None))
                if value is not None and value > 0.0 and reported_type in {1, 2}:
                    return HistoricalVolatilitySnapshot(
                        symbol=instrument.symbol,
                        con_id=instrument.con_id,
                        raw_historical_volatility=value,
                        observation_timestamp=datetime.now(tz=UTC),
                        market_data_type=reported_type,
                        unit="DECIMAL",
                    )
                if asyncio.get_running_loop().time() >= deadline:
                    raise IbkrError("HV_NOT_READY: missing, invalid, or unavailable tick 104")
                await asyncio.sleep(0.05)
        except Exception as exc:
            if isinstance(exc, IbkrError):
                raise
            raise IbkrError(
                f"HV_NOT_READY: tick 104 request failed for {instrument.symbol}: {exc}"
            ) from exc
        finally:
            if key is not None:
                self._release_market_data_stream(key)

    async def option_chains(
        self, instrument: QualifiedInstrument
    ) -> tuple[OptionChainDefinition, ...]:
        """Return normalized IBKR option-chain security definitions."""

        self._require_connected()
        cached = self._option_chain_cache.get(instrument.con_id)
        if cached is not None:
            self._reset_daily_resource_counters()
            self._deduplicated_requests_today += 1
            return cached
        try:
            sources = await asyncio.wait_for(
                self._client.reqSecDefOptParamsAsync(
                    instrument.symbol,
                    "",
                    instrument.security_type,
                    instrument.con_id,
                ),
                timeout=self.config.request_timeout_seconds,
            )
        except Exception as exc:
            raise IbkrError(
                f"IBKR option-chain request failed for {instrument.symbol}: {exc}"
            ) from exc
        chains: list[OptionChainDefinition] = []
        for source in sources:
            chain = cast(_SourceOptionChain, source)
            try:
                expirations = tuple(
                    sorted({_parse_option_expiry(value) for value in chain.expirations})
                )
                strikes = tuple(sorted({float(value) for value in chain.strikes}))
                underlying_con_id = int(chain.underlyingConId)
            except (AttributeError, TypeError, ValueError) as exc:
                raise IbkrError(
                    f"IBKR returned an invalid option chain for {instrument.symbol}"
                ) from exc
            if underlying_con_id != instrument.con_id:
                continue
            chains.append(
                OptionChainDefinition(
                    exchange=str(chain.exchange),
                    underlying_con_id=underlying_con_id,
                    trading_class=str(chain.tradingClass),
                    multiplier=str(chain.multiplier),
                    expirations=expirations,
                    strikes=strikes,
                )
            )
        if not chains:
            raise IbkrError(f"IBKR returned no option chain for {instrument.symbol}")
        result = tuple(chains)
        self._option_chain_cache[instrument.con_id] = result
        return result

    async def qualify_options(
        self, requests: tuple[OptionContractRequest, ...]
    ) -> tuple[QualifiedOption, ...]:
        """Qualify the exact option candidates supplied by Stage 4."""

        self._require_connected()
        if not requests:
            raise IbkrError("At least one option contract request is required")
        try:
            contracts = await asyncio.wait_for(
                self._client.qualifyContractsAsync(
                    *(_to_ib_option_request(item) for item in requests),
                    returnAll=False,
                ),
                timeout=self.config.request_timeout_seconds,
            )
        except Exception as exc:
            raise IbkrError(f"IBKR option contract qualification failed: {exc}") from exc
        qualified: list[QualifiedOption] = []
        for source in contracts:
            if source is None or isinstance(source, list):
                continue
            contract = cast(_QualifiedContract, source)
            try:
                expiry = _parse_option_expiry(contract.lastTradeDateOrContractMonth[:8])
                con_id = int(contract.conId)
                strike = float(contract.strike)
            except (AttributeError, TypeError, ValueError) as exc:
                raise IbkrError("IBKR returned an invalid qualified option contract") from exc
            if contract.secType != "OPT" or con_id <= 0:
                raise IbkrError("IBKR returned an invalid qualified option contract")
            qualified.append(
                QualifiedOption(
                    symbol=contract.symbol,
                    con_id=con_id,
                    exchange=contract.exchange,
                    currency=contract.currency,
                    security_type=contract.secType,
                    expiry=expiry,
                    strike=strike,
                    right=contract.right,
                    multiplier=contract.multiplier,
                    trading_class=contract.tradingClass,
                )
            )
        return tuple(qualified)

    def _select_account(self, accounts: list[str]) -> str:
        expected = self.config.expected_account
        if expected is not None:
            if expected not in accounts:
                raise IbkrError("Connected IBKR session does not expose the expected account")
            return expected
        if not accounts:
            raise IbkrError("Connected IBKR session exposed no account identifier")
        if len(accounts) > 1:
            raise IbkrError(
                "Connected IBKR session exposes multiple accounts; configure expected_account"
            )
        return accounts[0]

    def _verify_environment(self, account_id: str) -> None:
        account_is_paper = account_id.upper().startswith("D")
        expected_is_paper = self.config.environment is Environment.PAPER
        if account_is_paper != expected_is_paper:
            actual = Environment.PAPER if account_is_paper else Environment.LIVE
            raise IbkrError(
                f"IBKR session environment mismatch: configured "
                f"{self.config.environment.value}, account appears {actual.value}"
            )

    def _require_connected(self) -> None:
        if not self.is_connected or self._account_id is None:
            raise IbkrError("IBKR operation requires a verified active connection")


def _to_ib_contract(instrument: QualifiedInstrument) -> object:
    from ib_async import Contract

    return Contract(
        conId=instrument.con_id,
        symbol=instrument.symbol,
        secType=instrument.security_type,
        exchange=instrument.exchange,
        primaryExchange=instrument.primary_exchange or "",
        currency=instrument.currency,
    )


def _to_ib_option_contract(option: QualifiedOption) -> object:
    from ib_async import Contract

    return Contract(
        conId=option.con_id,
        symbol=option.symbol,
        secType=option.security_type,
        exchange=option.exchange,
        currency=option.currency,
        lastTradeDateOrContractMonth=option.expiry.strftime("%Y%m%d"),
        strike=option.strike,
        right=option.right,
        multiplier=option.multiplier,
        tradingClass=option.trading_class,
    )


def _to_ib_option_request(request: OptionContractRequest) -> object:
    from ib_async import Option

    return Option(
        request.symbol,
        request.expiry.strftime("%Y%m%d"),
        request.strike,
        request.right,
        request.exchange,
        request.multiplier,
        request.currency,
        tradingClass=request.trading_class,
    )


def _normalize_historical_bar(source: _SourceBar, *, index: int) -> HistoricalBar:
    try:
        timestamp = source.date
        open_price = float(source.open)
        high = float(source.high)
        low = float(source.low)
        close = float(source.close)
        volume = float(source.volume)
    except (AttributeError, TypeError, ValueError) as exc:
        raise IbkrError(f"IBKR returned invalid historical bar at index {index}") from exc

    try:
        return validate_historical_bar(
            HistoricalBar(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    except ValueError as exc:
        raise IbkrError(f"IBKR returned {exc} at index {index}") from exc


def _available_price(value: float) -> float | None:
    price = float(value)
    return price if isfinite(price) and price > 0 else None


def _optional_number(value: object) -> float | None:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _option_open_interest(ticker: _SourceOptionTicker, option: QualifiedOption) -> float | None:
    right_specific = (
        getattr(ticker, "callOpenInterest", None)
        if option.right.upper() in {"C", "CALL"}
        else getattr(ticker, "putOpenInterest", None)
    )
    return _optional_number(right_specific)


def _option_snapshot_complete(
    ticker: _SourceOptionTicker,
    option: QualifiedOption,
    *,
    allow_delayed_model: bool,
) -> bool:
    market_data_type = _optional_integer(getattr(ticker, "marketDataType", None))
    accepted_model_types = {1, 2, 3, 4} if allow_delayed_model else {1, 2}
    model = (
        getattr(ticker, "modelGreeks", None) if market_data_type in accepted_model_types else None
    )
    values = [
        _optional_number(getattr(ticker, "bid", None)),
        _optional_number(getattr(ticker, "ask", None)),
        _option_open_interest(ticker, option),
    ]
    values.append(_optional_number(getattr(model, "impliedVol", None)))
    return all(value is not None for value in values)


def _optional_integer(value: object) -> int | None:
    number = _optional_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _optional_contract_integer(contract: object, field: str) -> int | None:
    return _optional_integer(getattr(contract, field, None))


def _optional_contract_text(contract: object, field: str) -> str | None:
    value = getattr(contract, field, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _account_value(values: list[object], account: str, tag: str) -> float | None:
    matching: list[tuple[str, float]] = []
    for value_object in values:
        value = cast(Any, value_object)
        if str(value.account) != account or str(value.tag) != tag:
            continue
        number = _optional_number(value.value)
        if number is not None:
            matching.append((str(value.currency).upper(), number))
    if not matching:
        return None
    base_values = [number for currency, number in matching if currency == "BASE"]
    if len(base_values) == 1:
        return base_values[0]
    if not base_values and len(matching) == 1:
        return matching[0][1]
    return None


def _order_lifecycle(status: str) -> OrderLifecycle:
    normalized = status.strip().upper().replace(" ", "")
    if normalized in {
        "PENDINGSUBMIT",
        "APIPENDING",
        "PRESUBMITTED",
        "SUBMITTED",
        "PENDINGCANCEL",
    }:
        return OrderLifecycle.SUBMITTED
    if normalized == "FILLED":
        return OrderLifecycle.FILLED
    if normalized in {"CANCELLED", "APICANCELLED"}:
        return OrderLifecycle.CANCELLED
    if normalized == "INACTIVE":
        return OrderLifecycle.REJECTED
    raise IbkrError(f"IBKR returned unsupported order status: {status}")


def _parse_option_expiry(value: str) -> date:
    return datetime.strptime(str(value), "%Y%m%d").date()
