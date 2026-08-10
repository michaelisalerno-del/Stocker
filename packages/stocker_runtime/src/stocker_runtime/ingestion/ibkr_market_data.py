"""Public market-data-only IBKR facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast, runtime_checkable

from stocker_runtime.ingestion.dynamic_market_data import (
    ContractCandidate,
    OptionParameterSet,
)
from stocker_runtime.ingestion.inbox import (
    AdmissionResult,
    CallbackFence,
    MarketDataCallback,
)

MAX_MARKET_DATA_REQUEST_ID = 1_499_999_999


@dataclass(frozen=True)
class MarketDataStatus:
    """Typed, market-data-only interpretation of an official IBKR status."""

    kind: Literal[
        "temporary_disconnect",
        "recovered",
        "farm_degraded",
        "farm_recovered",
        "pacing",
        "request_rejected",
        "snapshot_end",
    ]
    code: int
    request_id: int | None
    message: str
    received_at_us: int
    affected_feed_kinds: tuple[Literal["quotes", "trades", "bars"], ...] = ()


@runtime_checkable
class MarketDataAdapter(Protocol):
    """Only capabilities the recorder may request from an external adapter."""

    def set_callback(
        self, callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult]
    ) -> None: ...

    def set_disconnect_callback(self, callback: Callable[[int], None]) -> None: ...

    def set_status_callback(self, callback: Callable[[MarketDataStatus], None]) -> None: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None:
        """Fence the retired socket so its callbacks cannot affect a future connection."""

        ...

    def configure_subscriptions(self, subscriptions: tuple[IBKRSubscription, ...]) -> None: ...

    def subscribe(self, fence: CallbackFence) -> None: ...

    def cancel(self, request_id: int) -> None: ...


@dataclass(frozen=True)
class IBKRSubscription:
    """Exact security identity for one configured market-data request."""

    request_id: int
    con_id: int
    symbol: str
    security_type: str
    exchange: str
    currency: str
    feed_kind: str
    snapshot: bool = False

    def __post_init__(self) -> None:
        if (
            self.request_id < 0
            or self.request_id > MAX_MARKET_DATA_REQUEST_ID
            or self.con_id <= 0
            or not self.symbol
            or not self.security_type
            or not self.exchange
            or not self.currency
            or self.feed_kind not in {"quotes", "trades", "bars"}
        ):
            raise ValueError("IBKR market-data subscription identity is invalid")


class _OptionMarketDataBridge(MarketDataAdapter, Protocol):
    def option_parameters(
        self, *, underlying_con_id: int, symbol: str
    ) -> tuple[OptionParameterSet, ...]: ...

    def option_contracts(
        self,
        *,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        multiplier: str,
        trading_class: str,
    ) -> tuple[ContractCandidate, ...]: ...


class IBKRMarketData:
    """Narrow facade over the private official API bridge."""

    __slots__ = ("_bridge",)

    def __init__(self, bridge: _OptionMarketDataBridge) -> None:
        self._bridge = bridge

    @classmethod
    def official(
        cls,
        *,
        host: str,
        port: int,
        client_id: int,
        read_only: bool,
        external_read_only_verified: bool,
        subscriptions: tuple[IBKRSubscription, ...] = (),
    ) -> IBKRMarketData:
        """Create the private official bridge after explicit safety verification."""

        from stocker_runtime.ingestion.official_bridge import create_official_bridge

        bridge = create_official_bridge(
            host=host,
            port=port,
            client_id=client_id,
            read_only=read_only,
            external_read_only_verified=external_read_only_verified,
            subscriptions=subscriptions,
        )
        return cls(cast(_OptionMarketDataBridge, bridge))

    def set_callback(
        self, callback: Callable[[CallbackFence, MarketDataCallback], AdmissionResult]
    ) -> None:
        self._bridge.set_callback(callback)

    def set_disconnect_callback(self, callback: Callable[[int], None]) -> None:
        self._bridge.set_disconnect_callback(callback)

    def set_status_callback(self, callback: Callable[[MarketDataStatus], None]) -> None:
        self._bridge.set_status_callback(callback)

    def connect(self) -> None:
        self._bridge.connect()

    def disconnect(self) -> None:
        self._bridge.disconnect()

    def configure_subscriptions(self, subscriptions: tuple[IBKRSubscription, ...]) -> None:
        self._bridge.configure_subscriptions(subscriptions)

    def subscribe(self, fence: CallbackFence) -> None:
        self._bridge.subscribe(fence)

    def cancel(self, request_id: int) -> None:
        self._bridge.cancel(request_id)

    def option_parameters(
        self, *, underlying_con_id: int, symbol: str
    ) -> tuple[OptionParameterSet, ...]:
        return self._bridge.option_parameters(
            underlying_con_id=underlying_con_id,
            symbol=symbol,
        )

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
        return self._bridge.option_contracts(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            multiplier=multiplier,
            trading_class=trading_class,
        )
