"""Minimal IBKR connection, market-data, and PAPER execution boundary."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import pairwise
from math import isfinite
from typing import Any, Protocol, cast

from stocker_core.config import IbkrConfig
from stocker_core.runs import Environment
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

    async def accountSummaryAsync(self, account: str = "") -> list[object]: ...

    async def reqContractDetailsAsync(self, contract: object) -> list[object]: ...

    def placeOrder(self, contract: object, order: object) -> object: ...

    async def reqAllOpenOrdersAsync(self) -> list[object]: ...

    async def reqCompletedOrdersAsync(self, apiOnly: bool) -> list[object]: ...

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

        if len(self.account_id) <= 5:
            return "***"
        return f"{self.account_id[:2]}***{self.account_id[-3:]}"


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


def _new_client() -> _IbClient:
    from ib_async import IB

    return cast(_IbClient, IB())


def _no_startup_fetches() -> object:
    from ib_async import StartupFetchNONE

    return StartupFetchNONE


class IbkrConnection:
    """One independent IBKR session; execution is opt-in and PAPER-only."""

    def __init__(
        self,
        config: IbkrConfig,
        *,
        client: _IbClient | None = None,
        execution_enabled: bool = False,
    ) -> None:
        self.config = config
        self._client = client if client is not None else _new_client()
        self._account_id: str | None = None
        self._execution_enabled = execution_enabled
        self._connection_epoch = 0

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
        """Connect read-only and verify the configured environment against the account."""

        try:
            await self._client.connectAsync(
                self.config.host,
                self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.connect_timeout_seconds,
                readonly=(
                    not self._execution_enabled or self.environment is Environment.LIVE
                ),
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
        if self._client.isConnected():
            self._client.disconnect()
        self._account_id = None
        if was_connected:
            self._connection_epoch += 1

    async def account_state(self) -> BrokerAccountState:
        """Read authoritative PAPER equity, buying power, and positions."""

        self._require_connected()
        try:
            values = await self._client.accountSummaryAsync(self.account)
            positions = await self.read_positions()
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
        """Transmit one atomic-style IBKR PAPER parent/target/stop bracket."""

        self._require_connected()
        if self.environment is Environment.LIVE or plan.environment is Environment.LIVE:
            raise IbkrError("LIVE_EXECUTION_DISABLED")
        if not self._execution_enabled:
            raise IbkrError("IBKR connection is read-only; PAPER execution was not enabled")
        if not self.account.upper().startswith("D") or instrument.con_id != plan.con_id:
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
            raise IbkrError(f"IBKR protected PAPER order submission failed: {exc}") from exc
        return BrokerOrderIds(parent=parent_id, stop=stop_id, target=target_id)

    async def read_open_orders(self) -> tuple[BrokerOpenOrder, ...]:
        """Read and normalize all open orders visible to this IBKR session."""

        self._require_connected()
        try:
            trades = await self._client.reqAllOpenOrdersAsync()
            return tuple(self._normalize_open_order(trade) for trade in trades)
        except Exception as exc:
            if isinstance(exc, IbkrError):
                raise
            raise IbkrError(f"IBKR open-order request failed: {exc}") from exc

    async def read_order_statuses(self) -> tuple[BrokerOrderStatus, ...]:
        """Read normalized open and completed order states, including rejections."""

        self._require_connected()
        try:
            trades = [
                *(await self._client.reqAllOpenOrdersAsync()),
                *(await self._client.reqCompletedOrdersAsync(apiOnly=False)),
            ]
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
            sources = await self._client.reqPositionsAsync()
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
            sources = await self._client.reqExecutionsAsync()
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
        if not normalized_symbol or not exchange.strip() or not currency.strip():
            raise IbkrError("Stock symbol, exchange, and currency are required")

        from ib_async import Stock

        requested = Stock(
            normalized_symbol,
            exchange.strip().upper(),
            currency.strip().upper(),
            primaryExchange=(primary_exchange or "").strip().upper(),
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
        return QualifiedInstrument(
            symbol=contract.symbol,
            con_id=contract.conId,
            exchange=contract.exchange,
            primary_exchange=contract.primaryExchange or None,
            currency=contract.currency,
            security_type=contract.secType,
        )

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

        try:
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

    async def current_quote(self, instrument: QualifiedInstrument) -> CurrentQuote:
        """Request one finite IBKR market-data snapshot for a qualified instrument."""

        self._require_connected()
        try:
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
        )

    async def option_snapshots(
        self, options: tuple[QualifiedOption, ...]
    ) -> tuple[OptionMarketSnapshot, ...]:
        """Capture tick-13 model computations plus generic-101 open interest."""

        self._require_connected()
        if not options:
            raise IbkrError("At least one qualified option is required")
        contracts = tuple(_to_ib_option_contract(option) for option in options)
        try:
            self._client.reqMarketDataType(1)
            tickers = tuple(
                self._client.reqMktData(
                    contract,
                    genericTickList="101",
                    snapshot=False,
                    regulatorySnapshot=False,
                )
                for contract in contracts
            )
            deadline = asyncio.get_running_loop().time() + self.config.request_timeout_seconds
            while not all(
                _option_snapshot_complete(cast(_SourceOptionTicker, ticker), option)
                for ticker, option in zip(tickers, options, strict=True)
            ):
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.05)
        except Exception as exc:
            raise IbkrError(f"IBKR option market data request failed: {exc}") from exc
        finally:
            for contract in locals().get("contracts", ()):
                self._client.cancelMktData(contract)

        captured_at = datetime.now(tz=UTC)
        normalized: list[OptionMarketSnapshot] = []
        for option, source in zip(options, tickers, strict=True):
            ticker = cast(_SourceOptionTicker, source)
            market_data_type = _optional_integer(getattr(ticker, "marketDataType", None))
            model = getattr(ticker, "modelGreeks", None) if market_data_type in {1, 2} else None
            normalized.append(
                OptionMarketSnapshot(
                    option=option,
                    captured_at=captured_at,
                    bid=_optional_number(getattr(ticker, "bid", None)),
                    ask=_optional_number(getattr(ticker, "ask", None)),
                    open_interest=_option_open_interest(ticker, option),
                    market_data_type=market_data_type,
                    model_iv=_optional_number(getattr(model, "impliedVol", None)),
                    model_delta=_optional_number(getattr(model, "delta", None)),
                    model_gamma=_optional_number(getattr(model, "gamma", None)),
                )
            )
        return tuple(normalized)

    async def option_chains(
        self, instrument: QualifiedInstrument
    ) -> tuple[OptionChainDefinition, ...]:
        """Return normalized IBKR option-chain security definitions."""

        self._require_connected()
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
        return tuple(chains)

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


def _option_snapshot_complete(ticker: _SourceOptionTicker, option: QualifiedOption) -> bool:
    market_data_type = _optional_integer(getattr(ticker, "marketDataType", None))
    model = getattr(ticker, "modelGreeks", None) if market_data_type in {1, 2} else None
    return all(
        value is not None
        for value in (
            _optional_number(getattr(ticker, "bid", None)),
            _optional_number(getattr(ticker, "ask", None)),
            _option_open_interest(ticker, option),
            _optional_number(getattr(model, "impliedVol", None)),
        )
    )


def _optional_integer(value: object) -> int | None:
    number = _optional_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


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
