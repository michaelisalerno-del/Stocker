"""Credential-free deterministic IBKR engineering adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from stocker_prospective.contract import assert_no_broker_mutation_surface
from stocker_prospective.market_data import (
    ConnectionTracker,
    MarketDataType,
    RequestIdAllocator,
)


class FakeIBKREvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    scenario: str
    kind: str
    timestamp_utc: str
    payload: dict[str, object]


@dataclass(frozen=True)
class FakeRequestResult:
    request_id: int
    items: tuple[Any, ...]


class FakeIBKRAdapter:
    """Replay market-data callbacks and failures with no network or credentials."""

    def __init__(self, *, fixture_id: str, events: tuple[FakeIBKREvent, ...]) -> None:
        self.fixture_id = fixture_id
        self._events = tuple(sorted(events, key=lambda item: item.sequence))
        if [item.sequence for item in self._events] != list(range(len(self._events))):
            raise ValueError("fake IBKR fixture sequence must be contiguous")
        self.request_ids = RequestIdAllocator(start=1)
        self.connection = ConnectionTracker()
        self.market_data_type = MarketDataType.LIVE
        self.active_subscriptions: dict[int, tuple[str, str]] = {}
        self._fixture_cursor = 0
        self._control_events: list[dict[str, Any]] = []
        assert_no_broker_mutation_surface(self)

    @classmethod
    def from_fixture(cls, path: str | Path) -> FakeIBKRAdapter:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("fixture_version") != "1":
            raise ValueError("unsupported fake IBKR fixture")
        events = tuple(FakeIBKREvent.model_validate(item) for item in payload["events"])
        return cls(fixture_id=str(payload["fixture_id"]), events=events)

    @property
    def scenarios(self) -> frozenset[str]:
        return frozenset(item.scenario for item in self._events)

    def connect(self) -> None:
        self.connection.connected(self.market_data_type)

    def disconnect(self) -> None:
        self.connection.connection_lost(code=1100, message="fake_connection_loss")

    @property
    def connected(self) -> bool:
        return self.connection.health().state.value == "connected"

    def subscribe(self, *, kind: str, symbol: str) -> int:
        if not self.connected:
            raise RuntimeError("fake adapter is disconnected")
        request_id = self.request_ids.next()
        self.active_subscriptions[request_id] = (kind, symbol)
        return request_id

    def cancel_subscription(self, request_id: int) -> None:
        self.active_subscriptions.pop(request_id, None)

    def replay(self) -> Iterator[FakeIBKREvent]:
        for event in self._events:
            if event.kind == "connection_loss":
                self.connection.connection_lost(code=1100, message="fake_connection_loss")
            elif event.kind == "reconnect":
                self.connection.connection_restored(
                    data_maintained=bool(event.payload.get("data_maintained", False)),
                    code=self._integer(event.payload.get("code"), default=1101),
                )
            elif event.kind == "market_data_type_change":
                self.market_data_type = MarketDataType(str(event.payload["market_data_type"]))
                self.connection.market_data_type_observed(self.market_data_type)
            yield event

    @staticmethod
    def _attribute(value: object, *names: str) -> object | None:
        if isinstance(value, dict):
            for name in names:
                if name in value:
                    return cast(object, value[name])
            return None
        for name in names:
            candidate = getattr(value, name, None)
            if candidate is not None:
                return cast(object, candidate)
        return None

    @staticmethod
    def _con_id(symbol: str) -> int:
        return int.from_bytes(symbol.encode("utf-8"), "little") % 2_000_000_000 + 1

    @staticmethod
    def _integer(value: object, *, default: int) -> int:
        if value is None or isinstance(value, bool):
            return default
        return int(str(value))

    @staticmethod
    def _floating(value: object, *, default: float) -> float:
        if value is None or isinstance(value, bool):
            return default
        return float(str(value))

    def qualify_exact_contract(self, contract: object) -> FakeRequestResult:
        symbol = str(self._attribute(contract, "symbol") or "")
        security_type = str(self._attribute(contract, "secType", "sec_type") or "STK")
        if not symbol:
            return FakeRequestResult(self.request_ids.next(), ())
        con_id_value = self._attribute(contract, "conId", "con_id")
        con_id = (
            self._con_id(
                "|".join(
                    (
                        symbol,
                        str(
                            self._attribute(
                                contract,
                                "lastTradeDateOrContractMonth",
                                "expiry",
                            )
                            or ""
                        ),
                        str(self._attribute(contract, "strike") or ""),
                        str(self._attribute(contract, "right") or ""),
                    )
                )
            )
            if con_id_value is None
            else self._integer(con_id_value, default=0)
        )
        qualified = SimpleNamespace(
            symbol=symbol,
            secType=security_type,
            conId=con_id,
            exchange=str(self._attribute(contract, "exchange") or "SMART"),
            currency=str(self._attribute(contract, "currency") or "USD"),
            localSymbol=str(self._attribute(contract, "localSymbol", "local_symbol") or symbol),
            lastTradeDateOrContractMonth=str(
                self._attribute(contract, "lastTradeDateOrContractMonth", "expiry") or ""
            ).replace("-", ""),
            strike=self._floating(self._attribute(contract, "strike"), default=0.0),
            right=str(self._attribute(contract, "right") or ""),
            tradingClass=str(self._attribute(contract, "tradingClass", "trading_class") or symbol),
        )
        return FakeRequestResult(self.request_ids.next(), (qualified,))

    def _request(self, kind: str, contract: object) -> int:
        if not self.connected:
            raise RuntimeError("fake adapter is disconnected")
        symbol = str(self._attribute(contract, "symbol") or "")
        request_id = self.request_ids.next()
        self.active_subscriptions[request_id] = (kind, symbol)
        return request_id

    def request_market_data(self, contract: object, **_kwargs: object) -> int:
        return self._request("level1", contract)

    def request_historical_five_minute_updates(
        self,
        contract: object,
        **_kwargs: object,
    ) -> int:
        return self._request("bar", contract)

    def request_tick_by_tick(
        self,
        contract: object,
        tick_type: str,
        **_kwargs: object,
    ) -> int:
        return self._request(f"tick_by_tick:{tick_type}", contract)

    def request_market_depth(self, contract: object, **_kwargs: object) -> int:
        return self._request("depth", contract)

    def _cancel(self, request_id: int) -> None:
        self.active_subscriptions.pop(request_id, None)

    def cancel_market_data(self, request_id: int, **_kwargs: object) -> None:
        self._cancel(request_id)

    def cancel_historical_updates(self, request_id: int, **_kwargs: object) -> None:
        self._cancel(request_id)

    def cancel_tick_by_tick(self, request_id: int, **_kwargs: object) -> None:
        self._cancel(request_id)

    def cancel_market_depth(self, request_id: int, **_kwargs: object) -> None:
        self._cancel(request_id)

    def require_live_market_data(self) -> None:
        self.market_data_type = MarketDataType.LIVE
        self.connection.market_data_type_observed(MarketDataType.LIVE)

    def request_current_time(self) -> int:
        request_id = -1
        observed = datetime.now(UTC)
        self._control_events.append(
            {
                "kind": "current_time",
                "request_id": request_id,
                "provider_timestamp_utc": observed.isoformat(),
                "received_timestamp_utc": observed.isoformat(),
                "received_monotonic_ns": 0,
                "source_sequence": 0,
            }
        )
        return request_id

    def request_depth_exchanges(self) -> int:
        request_id = -2
        observed = datetime.now(UTC)
        self._control_events.append(
            {
                "kind": "depth_exchanges",
                "request_id": request_id,
                "exchanges": (
                    {"exchange": "NYSE"},
                    {"exchange": "NASDAQ"},
                ),
                "received_timestamp_utc": observed.isoformat(),
                "received_monotonic_ns": 1,
                "source_sequence": 1,
            }
        )
        return request_id

    def request_option_chain_metadata(self, **_kwargs: object) -> FakeRequestResult:
        return FakeRequestResult(self.request_ids.next(), ())

    def server_version(self) -> int:
        return 187

    def reconnect(self) -> None:
        self.connection.connected(self.market_data_type)

    def drain_stream_events(self) -> tuple[dict[str, Any], ...]:
        output = list(self._control_events)
        self._control_events.clear()
        for event in self._events[self._fixture_cursor :]:
            self._fixture_cursor += 1
            output.extend(self._translate(event))
        return tuple(output)

    def _request_for(self, symbol: str, prefix: str) -> int | None:
        return next(
            (
                request_id
                for request_id, (kind, candidate) in sorted(self.active_subscriptions.items())
                if candidate == symbol and kind.startswith(prefix)
            ),
            None,
        )

    def _translate(self, event: FakeIBKREvent) -> tuple[dict[str, Any], ...]:
        observed = datetime.fromisoformat(event.timestamp_utc.replace("Z", "+00:00"))
        symbol = str(event.payload.get("symbol", ""))
        base = {
            "received_timestamp_utc": observed.isoformat(),
            "provider_timestamp_utc": observed.isoformat(),
            "received_monotonic_ns": event.sequence * 10 + 1,
            "source_sequence": event.sequence * 10 + 1,
        }
        if event.kind == "connection_loss":
            self.connection.connection_lost(code=1100, message="fake_connection_loss")
            return ()
        if event.kind == "reconnect":
            self.connection.connection_restored(
                data_maintained=bool(event.payload.get("data_maintained", False)),
                code=self._integer(event.payload.get("code"), default=1101),
            )
            return ()
        if event.kind == "market_data_type_change":
            self.market_data_type = MarketDataType(str(event.payload["market_data_type"]))
            self.connection.market_data_type_observed(self.market_data_type)
            return ()
        if event.kind == "level1_quote":
            request_id = self._request_for(symbol, "level1")
            if request_id is None:
                return ()
            return tuple(
                {
                    **base,
                    "kind": "level1_quote_update",
                    "request_id": request_id,
                    "field": field,
                    "value": event.payload[field],
                    "market_data_type": self.market_data_type.value,
                    "received_monotonic_ns": event.sequence * 10 + offset,
                    "source_sequence": event.sequence * 10 + offset,
                }
                for offset, field in enumerate(("bid", "ask"), start=1)
            )
        if event.kind == "tick_by_tick_bidask":
            request_id = self._request_for(symbol, "tick_by_tick:BidAsk")
            if request_id is None:
                return ()
            return (
                {
                    **base,
                    **event.payload,
                    "kind": "tick_by_tick_bidask",
                    "request_id": request_id,
                    "market_data_type": self.market_data_type.value,
                },
            )
        if event.kind == "last_trade":
            request_id = self._request_for(symbol, "tick_by_tick:Last")
            if request_id is None:
                return ()
            return (
                {
                    **base,
                    **event.payload,
                    "kind": "tick_by_tick_trade",
                    "request_id": request_id,
                    "market_data_type": self.market_data_type.value,
                },
            )
        if event.kind == "depth_update":
            request_id = self._request_for(symbol, "depth")
            if request_id is None:
                return ()
            side = str(event.payload.get("side", "bid"))
            operation = str(event.payload.get("operation", "update"))
            return (
                {
                    **base,
                    "kind": "depth_update",
                    "request_id": request_id,
                    # An engineering fixture starts from an empty book, so the
                    # first advertised update is represented as an insert.
                    "operation": "insert" if operation == "update" else operation,
                    "position": 0,
                    "side": side,
                    "price": 100.0 if side == "bid" else 100.1,
                    "size": 500.0,
                    "market_maker_or_exchange": "FAKE",
                    "smart_depth": True,
                },
            )
        if event.kind == "depth_reset":
            request_id = self._request_for(symbol, "depth")
            if request_id is None:
                return ()
            return (
                {
                    **base,
                    "kind": "depth_reset",
                    "request_id": request_id,
                    "smart_depth": True,
                },
            )
        if event.kind == "five_minute_bar":
            request_id = self._request_for(symbol, "bar")
            if request_id is None:
                return ()
            return (
                {
                    **base,
                    "kind": "historical_bar_update",
                    "request_id": request_id,
                    "bar_start_utc": (observed - timedelta(minutes=5)).isoformat(),
                    "open": 100.0,
                    "high": 100.2,
                    "low": 99.9,
                    "close": 100.1,
                    "volume": 1_000.0,
                    "wap": 100.05,
                    "trade_count": 100,
                },
            )
        return ()
