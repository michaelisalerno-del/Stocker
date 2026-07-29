"""Deterministic conversion of official IBKR callbacks into immutable raw events."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, cast
from zoneinfo import ZoneInfo

from stocker_prospective.events import (
    DepthOperation,
    DepthSide,
    OptionQuoteEvent,
    RawEvent,
    UnderlyingDepthEvent,
    UnderlyingLevel1QuoteEvent,
    UnderlyingTickBidAskEvent,
    UnderlyingTickTradeEvent,
)
from stocker_prospective.live_bars import HistoricalBarUpdate
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_ledger import OptionContract

NEW_YORK = ZoneInfo("America/New_York")


class StreamKind(StrEnum):
    UNDERLYING_LEVEL1 = "underlying_level1"
    UNDERLYING_TICK_BIDASK = "underlying_tick_bidask"
    UNDERLYING_TICK_LAST = "underlying_tick_last"
    UNDERLYING_DEPTH = "underlying_depth"
    UNDERLYING_BAR = "underlying_bar"
    OPTION_LEVEL1 = "option_level1"


@dataclass(frozen=True)
class StreamOwner:
    request_id: int
    kind: StreamKind
    symbol: str
    con_id: int
    exchange: str | None = None
    episode_id: str | None = None
    option_contract: OptionContract | None = None

    def __post_init__(self) -> None:
        if self.request_id < 0 or self.con_id <= 0 or not self.symbol:
            raise ValueError("stream ownership identity is invalid")
        if self.kind is StreamKind.OPTION_LEVEL1:
            if self.episode_id is None or self.option_contract is None:
                raise ValueError("option stream ownership is incomplete")
            if self.option_contract.con_id != self.con_id:
                raise ValueError("option stream conId differs from ownership")
        elif self.episode_id is not None or self.option_contract is not None:
            raise ValueError("underlying stream cannot carry option ownership")


def stream_owner_payload(owner: StreamOwner) -> dict[str, Any]:
    """Return a JSON-safe, version-stable stream ownership receipt."""

    option = owner.option_contract
    return {
        "request_id": owner.request_id,
        "kind": owner.kind.value,
        "symbol": owner.symbol,
        "con_id": owner.con_id,
        "exchange": owner.exchange,
        "episode_id": owner.episode_id,
        "option_contract": (
            None
            if option is None
            else {
                "underlying_con_id": option.underlying_con_id,
                "con_id": option.con_id,
                "expiry": option.expiry.isoformat(),
                "dte": option.dte,
                "dte_bucket": option.dte_bucket.value,
                "strike": option.strike,
                "right": option.right,
                "multiplier": option.multiplier,
                "exchange": option.exchange,
                "trading_class": option.trading_class,
            }
        ),
    }


@dataclass(frozen=True)
class NormalizedCallback:
    raw_event: RawEvent | None = None
    historical_bar: HistoricalBarUpdate | None = None
    control_kind: str | None = None
    control_payload: dict[str, Any] | None = None


def _timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("IBKR market field is boolean, not numeric")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError("IBKR market field is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError("IBKR market field is non-finite")
    return number


def _integer(value: object) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _market_data_type(value: object) -> MarketDataType:
    if isinstance(value, MarketDataType):
        return value
    try:
        return MarketDataType(str(value))
    except ValueError:
        return MarketDataType.UNKNOWN


def _event_id(owner: StreamOwner, payload: dict[str, Any]) -> str:
    identity = (
        f"{owner.kind.value}|{owner.request_id}|{owner.con_id}|"
        f"{payload['source_sequence']}|{payload['kind']}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _session(timestamp: datetime) -> date:
    return timestamp.astimezone(NEW_YORK).date()


class IBKRCallbackNormalizer:
    """Own request identities and preserve every state-changing callback."""

    def __init__(self, *, prospective_collection_start: datetime) -> None:
        if (
            prospective_collection_start.tzinfo is None
            or prospective_collection_start.utcoffset() is None
        ):
            raise ValueError("prospective collection start must be timezone-aware")
        self.prospective_collection_start = prospective_collection_start.astimezone(UTC)
        self._owners: dict[int, StreamOwner] = {}
        self._quote_state: dict[int, dict[str, Any]] = {}

    def register(self, owner: StreamOwner) -> None:
        existing = self._owners.get(owner.request_id)
        if existing is not None and existing != owner:
            raise ValueError("request-ID ownership differs")
        self._owners[owner.request_id] = owner

    def unregister(self, request_id: int) -> None:
        self._owners.pop(request_id, None)
        self._quote_state.pop(request_id, None)

    def owner(self, request_id: int) -> StreamOwner | None:
        return self._owners.get(request_id)

    @property
    def owners(self) -> tuple[StreamOwner, ...]:
        return tuple(self._owners.values())

    def normalize(self, payload: dict[str, Any]) -> NormalizedCallback | None:
        kind = str(payload.get("kind", ""))
        received = _timestamp(payload.get("received_timestamp_utc"))
        if received is None:
            raise ValueError("IBKR callback receive timestamp is invalid")
        if received < self.prospective_collection_start:
            return None
        request_id = _integer(payload.get("request_id"))
        if request_id is None:
            raise ValueError("IBKR callback request ID is invalid")
        if request_id < 0:
            return NormalizedCallback(
                control_kind=kind,
                control_payload=dict(payload),
            )
        owner = self._owners.get(request_id)
        if owner is None:
            raise ValueError("IBKR callback has no deterministic request owner")
        if kind == "level1_quote_update":
            return NormalizedCallback(raw_event=self._level1(owner, payload, received))
        if kind == "tick_by_tick_bidask":
            return NormalizedCallback(raw_event=self._bidask(owner, payload, received))
        if kind == "tick_by_tick_trade":
            return NormalizedCallback(raw_event=self._trade(owner, payload, received))
        if kind == "depth":
            return NormalizedCallback(raw_event=self._depth(owner, payload, received))
        if kind == "depth_reset":
            return NormalizedCallback(
                raw_event=UnderlyingDepthEvent(
                    **self._common(owner, payload, received),
                    operation=DepthOperation.REMOVE,
                    position=0,
                    side=DepthSide.BID,
                    price=None,
                    size=None,
                    market_maker_or_exchange=None,
                    smart_depth=bool(payload.get("smart_depth", True)),
                    reset=True,
                ),
                control_kind=kind,
                control_payload={**payload, "symbol": owner.symbol, "con_id": owner.con_id},
            )
        if kind in {"historical_bar", "historical_bar_update"}:
            return NormalizedCallback(historical_bar=self._historical_bar(owner, payload, received))
        if kind in {"historical_backfill_end", "ibkr_error"}:
            return NormalizedCallback(control_kind=kind, control_payload=dict(payload))
        raise ValueError(f"unsupported IBKR stream callback kind: {kind}")

    def _common(
        self,
        owner: StreamOwner,
        payload: dict[str, Any],
        received: datetime,
    ) -> dict[str, Any]:
        provider = _timestamp(payload.get("provider_timestamp_utc"))
        sequence = _integer(payload.get("source_sequence"))
        monotonic = _integer(payload.get("received_monotonic_ns"))
        if sequence is None or sequence < 0 or monotonic is None or monotonic < 0:
            raise ValueError("IBKR callback ordering identity is invalid")
        return {
            "event_id": _event_id(owner, payload),
            "received_timestamp_utc": received,
            "received_monotonic_ns": monotonic,
            "provider_timestamp_utc": provider,
            "source_sequence": sequence,
            "session": _session(provider or received),
            "symbol": owner.symbol,
            "con_id": owner.con_id,
            "request_id": owner.request_id,
        }

    def _level1(
        self,
        owner: StreamOwner,
        payload: dict[str, Any],
        received: datetime,
    ) -> RawEvent:
        if owner.kind not in {
            StreamKind.UNDERLYING_LEVEL1,
            StreamKind.OPTION_LEVEL1,
        }:
            raise ValueError("Level I callback differs from request ownership")
        state = self._quote_state.setdefault(owner.request_id, {})
        field = str(payload.get("field", "unknown"))
        if field == "option_computation":
            source = str(payload.get("computation_source", "unknown"))
            if source in {"model", "last", "bid", "ask"}:
                source_values = cast(
                    dict[str, dict[str, float | None]],
                    state.setdefault("option_computation_by_source", {}),
                )
                snapshot = dict(source_values.get(source, {}))
                for source_name, target_name in (
                    ("option_price", "option_model_price"),
                    ("implied_volatility", "implied_volatility"),
                    ("delta", "delta"),
                    ("gamma", "gamma"),
                    ("theta", "theta"),
                    ("vega", "vega"),
                    ("underlying_reference_price", "underlying_reference_price"),
                ):
                    value = _number(payload.get(source_name))
                    snapshot[source_name] = value
                    if source == "model":
                        state[target_name] = value
                source_values[source] = snapshot
        elif field == "market_data_type":
            state["market_data_type"] = payload.get("value")
        else:
            state[field] = _number(payload.get("value"))
        if payload.get("market_data_type") is not None:
            state["market_data_type"] = payload["market_data_type"]
        common = self._common(owner, payload, received)
        market_data_type = _market_data_type(state.get("market_data_type"))
        if owner.kind is StreamKind.OPTION_LEVEL1:
            contract = owner.option_contract
            episode_id = owner.episode_id
            assert contract is not None and episode_id is not None
            return OptionQuoteEvent(
                **common,
                episode_id=episode_id,
                expiry=contract.expiry,
                dte=contract.dte,
                dte_bucket=contract.dte_bucket,
                strike=contract.strike,
                right=contract.right,
                multiplier=contract.multiplier,
                exchange=contract.exchange,
                trading_class=contract.trading_class,
                bid=_number(state.get("bid")),
                bid_size=_number(state.get("bid_size")),
                ask=_number(state.get("ask")),
                ask_size=_number(state.get("ask_size")),
                last=_number(state.get("last")),
                last_size=_number(state.get("last_size")),
                market_data_type=market_data_type,
                option_model_price=_number(state.get("option_model_price")),
                implied_volatility=_number(state.get("implied_volatility")),
                delta=_number(state.get("delta")),
                gamma=_number(state.get("gamma")),
                theta=_number(state.get("theta")),
                vega=_number(state.get("vega")),
                underlying_reference_price=_number(state.get("underlying_reference_price")),
                volume=_number(state.get("volume")),
                open_interest=_number(
                    state.get(
                        "call_open_interest" if contract.right == "C" else "put_open_interest"
                    )
                ),
                option_computation_by_source=cast(
                    dict[str, dict[str, float | None]],
                    state.get("option_computation_by_source", {}),
                ),
                quote_attributes=cast(
                    dict[str, bool | int | float | str | None],
                    payload.get("attributes", {}),
                ),
            )
        bid = _number(state.get("bid"))
        ask = _number(state.get("ask"))
        quote_valid = bid is not None and ask is not None and bid > 0.0 and ask > 0.0 and ask >= bid
        provider_timestamp = cast(datetime | None, common["provider_timestamp_utc"])
        return UnderlyingLevel1QuoteEvent(
            **common,
            bid=bid,
            bid_size=_number(state.get("bid_size")),
            ask=ask,
            ask_size=_number(state.get("ask_size")),
            last=_number(state.get("last")),
            last_size=_number(state.get("last_size")),
            volume=_number(state.get("volume")),
            market_data_type=market_data_type,
            source="official_ibkr_tws_socket_api",
            quote_valid=quote_valid,
            staleness_ms=(
                None
                if provider_timestamp is None
                else max(0.0, (received - provider_timestamp).total_seconds() * 1000.0)
            ),
            tick_type=field,
            exchange=owner.exchange,
            quote_attributes=cast(
                dict[str, bool | int | float | str | None],
                payload.get("attributes", {}),
            ),
            halted=(bool(state["halted"]) if state.get("halted") is not None else None),
        )

    def _bidask(
        self,
        owner: StreamOwner,
        payload: dict[str, Any],
        received: datetime,
    ) -> UnderlyingTickBidAskEvent:
        if owner.kind is not StreamKind.UNDERLYING_TICK_BIDASK:
            raise ValueError("BidAsk callback differs from request ownership")
        return UnderlyingTickBidAskEvent(
            **self._common(owner, payload, received),
            bid=float(payload["bid"]),
            bid_size=float(payload["bid_size"]),
            ask=float(payload["ask"]),
            ask_size=float(payload["ask_size"]),
            bid_past_low=cast(bool | None, payload.get("bid_past_low")),
            ask_past_high=cast(bool | None, payload.get("ask_past_high")),
            exchange=owner.exchange,
            market_data_type=_market_data_type(payload.get("market_data_type")),
        )

    def _trade(
        self,
        owner: StreamOwner,
        payload: dict[str, Any],
        received: datetime,
    ) -> UnderlyingTickTradeEvent:
        if owner.kind is not StreamKind.UNDERLYING_TICK_LAST:
            raise ValueError("Last callback differs from request ownership")
        conditions = payload.get("conditions", ())
        return UnderlyingTickTradeEvent(
            **self._common(owner, payload, received),
            price=float(payload["price"]),
            size=float(payload["size"]),
            exchange=(None if payload.get("exchange") is None else str(payload["exchange"])),
            conditions=tuple(str(value) for value in cast(tuple[object, ...], conditions)),
            market_data_type=_market_data_type(payload.get("market_data_type")),
            past_limit=cast(bool | None, payload.get("past_limit")),
            unreported=cast(bool | None, payload.get("unreported")),
            halted=cast(bool | None, payload.get("halted")),
        )

    def _depth(
        self,
        owner: StreamOwner,
        payload: dict[str, Any],
        received: datetime,
    ) -> UnderlyingDepthEvent:
        if owner.kind is not StreamKind.UNDERLYING_DEPTH:
            raise ValueError("depth callback differs from request ownership")
        operation = DepthOperation(str(payload["operation"]))
        return UnderlyingDepthEvent(
            **self._common(owner, payload, received),
            operation=operation,
            position=int(payload["position"]),
            side=DepthSide(str(payload["side"])),
            price=(None if operation is DepthOperation.REMOVE else _number(payload.get("price"))),
            size=(None if operation is DepthOperation.REMOVE else _number(payload.get("size"))),
            market_maker_or_exchange=(
                None
                if payload.get("market_maker_or_exchange") is None
                else str(payload["market_maker_or_exchange"])
            ),
            smart_depth=bool(payload.get("smart_depth", False)),
            reset=False,
        )

    def _historical_bar(
        self,
        owner: StreamOwner,
        payload: dict[str, Any],
        received: datetime,
    ) -> HistoricalBarUpdate:
        if owner.kind is not StreamKind.UNDERLYING_BAR:
            raise ValueError("bar callback differs from request ownership")
        start = _timestamp(payload.get("bar_start_utc"))
        if start is None:
            raise ValueError("historical bar timestamp is invalid")
        return HistoricalBarUpdate(
            request_id=owner.request_id,
            symbol=owner.symbol,
            con_id=owner.con_id,
            bar_start_utc=start,
            provider_timestamp_utc=start,
            received_timestamp_utc=received,
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=_number(payload.get("volume")),
            wap=_number(payload.get("wap")),
            trade_count=_integer(payload.get("trade_count")),
            source=str(payload.get("source", "ibkr_historical_keep_up_to_date")),
            explicitly_finalised=False,
        )


__all__ = [
    "IBKRCallbackNormalizer",
    "NormalizedCallback",
    "StreamKind",
    "StreamOwner",
    "stream_owner_payload",
]
