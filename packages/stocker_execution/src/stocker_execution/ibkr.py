"""Minimal read-only IBKR connection and market-data boundary."""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from itertools import pairwise
from math import isfinite
from typing import Any, Protocol, cast

from stocker_core.config import IbkrConfig
from stocker_core.runs import Environment


class IbkrError(RuntimeError):
    """A clear failure at the IBKR connection or data boundary."""


class _IbClient(Protocol):
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


class _QualifiedContract(Protocol):
    symbol: str
    conId: int
    exchange: str
    primaryExchange: str
    currency: str
    secType: str


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
class HistoricalBar:
    """Minimal unfilled OHLCV bar copied from an IBKR historical response."""

    timestamp: date | datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


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
    """One independent, read-only IBKR Gateway or TWS API session."""

    def __init__(self, config: IbkrConfig, *, client: _IbClient | None = None) -> None:
        self.config = config
        self._client = client if client is not None else _new_client()
        self._account_id: str | None = None

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
                readonly=True,
                account=self.config.expected_account or "",
                raiseSyncErrors=True,
                fetchFields=_no_startup_fetches(),
            )
            if not self.is_connected:
                raise IbkrError("IBKR reported no active API connection after connecting")
            account_id = self._select_account(self._client.managedAccounts())
            self._verify_environment(account_id)
            self._account_id = account_id
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

        if self._client.isConnected():
            self._client.disconnect()
        self._account_id = None

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

    if not isinstance(timestamp, (date, datetime)):
        raise IbkrError(f"IBKR returned invalid historical bar timestamp at index {index}")
    values = (open_price, high, low, close, volume)
    if not all(isfinite(value) for value in values) or volume < 0:
        raise IbkrError(f"IBKR returned invalid historical bar values at index {index}")
    if high < max(open_price, low, close) or low > min(open_price, high, close):
        raise IbkrError(f"IBKR returned inconsistent OHLC values at index {index}")

    return HistoricalBar(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _available_price(value: float) -> float | None:
    price = float(value)
    return price if isfinite(price) and price > 0 else None
