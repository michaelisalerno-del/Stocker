"""Deterministic conversion of official IBKR callbacks into immutable raw events."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal, Self, cast
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
from stocker_prospective.options import DteBucket

NEW_YORK = ZoneInfo("America/New_York")
MAX_RETIRED_QUOTE_STATES = 256


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


class OptionContractReceipt(BaseModel):
    """Strict JSON contract identity embedded in a durable owner receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    underlying_con_id: int = Field(gt=0, strict=True)
    con_id: int = Field(gt=0, strict=True)
    expiry: date
    dte: int = Field(ge=0, strict=True)
    dte_bucket: DteBucket
    strike: float = Field(gt=0.0, allow_inf_nan=False, strict=True)
    right: Literal["C", "P"]
    multiplier: int = Field(gt=0, strict=True)
    exchange: str = Field(min_length=1, strict=True)
    trading_class: str = Field(min_length=1, strict=True)

    @classmethod
    def from_contract(cls, contract: OptionContract) -> Self:
        if contract.con_id is None:
            raise ValueError("PERSISTED_OPTION_CONTRACT_UNRESOLVED")
        return cls(
            underlying_con_id=contract.underlying_con_id,
            con_id=contract.con_id,
            expiry=contract.expiry,
            dte=contract.dte,
            dte_bucket=contract.dte_bucket,
            strike=contract.strike,
            right=contract.right,
            multiplier=contract.multiplier,
            exchange=contract.exchange,
            trading_class=contract.trading_class,
        )

    def to_contract(self) -> OptionContract:
        return OptionContract(
            underlying_con_id=self.underlying_con_id,
            con_id=self.con_id,
            expiry=self.expiry,
            dte=self.dte,
            dte_bucket=self.dte_bucket,
            strike=self.strike,
            right=self.right,
            multiplier=self.multiplier,
            exchange=self.exchange,
            trading_class=self.trading_class,
        )


class StreamOwnerReceipt(BaseModel):
    """Typed admission-time identity for one provider callback stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: int = Field(ge=0, strict=True)
    kind: StreamKind
    symbol: str = Field(min_length=1, strict=True)
    con_id: int = Field(gt=0, strict=True)
    exchange: str | None = Field(default=None, strict=True)
    episode_id: str | None = Field(default=None, min_length=1, strict=True)
    option_contract: OptionContractReceipt | None = None

    @classmethod
    def from_owner(cls, owner: StreamOwner) -> Self:
        option = owner.option_contract
        return cls(
            request_id=owner.request_id,
            kind=owner.kind,
            symbol=owner.symbol,
            con_id=owner.con_id,
            exchange=owner.exchange,
            episode_id=owner.episode_id,
            option_contract=(
                None if option is None else OptionContractReceipt.from_contract(option)
            ),
        )

    def to_owner(self) -> StreamOwner:
        return StreamOwner(
            request_id=self.request_id,
            kind=self.kind,
            symbol=self.symbol,
            con_id=self.con_id,
            exchange=self.exchange,
            episode_id=self.episode_id,
            option_contract=(
                None if self.option_contract is None else self.option_contract.to_contract()
            ),
        )


def stream_owner_payload(owner: StreamOwner) -> dict[str, Any]:
    """Return a JSON-safe stream ownership receipt."""

    return StreamOwnerReceipt.from_owner(owner).model_dump(mode="json")


def stream_owner_from_payload(payload: object) -> StreamOwner:
    """Validate and restore the owner receipt committed at callback admission."""

    try:
        return StreamOwnerReceipt.model_validate(payload).to_owner()
    except (ValidationError, ValueError) as exc:
        raise ValueError("PERSISTED_STREAM_OWNER_INVALID") from exc


@dataclass(frozen=True)
class NormalizedCallback:
    raw_event: RawEvent | None = None
    historical_bar: HistoricalBarUpdate | None = None
    control_kind: str | None = None
    control_payload: dict[str, Any] | None = None
    stream_owner: StreamOwner | None = None


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

    def __init__(
        self,
        *,
        prospective_collection_start: datetime,
        max_retired_quote_states: int = MAX_RETIRED_QUOTE_STATES,
    ) -> None:
        if (
            prospective_collection_start.tzinfo is None
            or prospective_collection_start.utcoffset() is None
        ):
            raise ValueError("prospective collection start must be timezone-aware")
        if max_retired_quote_states <= 0:
            raise ValueError("retired quote-state bound must be positive")
        self.prospective_collection_start = prospective_collection_start.astimezone(UTC)
        self.max_retired_quote_states = max_retired_quote_states
        self._owners: dict[int, StreamOwner] = {}
        self._quote_state: dict[StreamOwner, dict[str, Any]] = {}
        self._retired_quote_owners: dict[StreamOwner, None] = {}
        self._detached_batch_owners: set[StreamOwner] | None = None
        self._detached_batch_original_states: dict[StreamOwner, dict[str, Any] | None] | None = None
        self._detached_batch_newly_retired: set[StreamOwner] | None = None

    def register(self, owner: StreamOwner) -> None:
        existing = self._owners.get(owner.request_id)
        if existing is not None and existing != owner:
            raise ValueError("request-ID ownership differs")
        if existing is None:
            self._retired_quote_owners.pop(owner, None)
            self._quote_state.pop(owner, None)
        self._owners[owner.request_id] = owner

    def unregister(self, request_id: int) -> None:
        owner = self._owners.get(request_id)
        if owner is None:
            return
        if owner in self._quote_state:
            # Poll-time reconciliation releases this state as soon as SQLite
            # proves no admitted callback remains unacknowledged.
            self._retain_retired_quote_owner(owner)
        self._owners.pop(request_id)

    def _retain_retired_quote_owner(self, owner: StreamOwner) -> None:
        if owner in self._retired_quote_owners:
            return
        if len(self._retired_quote_owners) >= self.max_retired_quote_states:
            raise RuntimeError("CALLBACK_RETIRED_QUOTE_STATE_CAPACITY_EXCEEDED")
        self._retired_quote_owners[owner] = None

    def owner(self, request_id: int) -> StreamOwner | None:
        return self._owners.get(request_id)

    @property
    def owners(self) -> tuple[StreamOwner, ...]:
        return tuple(self._owners.values())

    @property
    def retired_quote_owners(self) -> tuple[StreamOwner, ...]:
        return tuple(self._retired_quote_owners)

    def release_retired_quote_owner(self, owner: StreamOwner) -> None:
        if self._owners.get(owner.request_id) == owner:
            return
        self._retired_quote_owners.pop(owner, None)
        self._quote_state.pop(owner, None)

    @contextmanager
    def normalization_batch(self) -> Iterator[None]:
        """Retain detached quote state for one ordered durable-inbox lease."""

        if (
            self._detached_batch_owners is not None
            or self._detached_batch_original_states is not None
            or self._detached_batch_newly_retired is not None
        ):
            raise RuntimeError("CALLBACK_NORMALIZATION_BATCH_ALREADY_ACTIVE")
        self._detached_batch_owners = set()
        self._detached_batch_original_states = {}
        self._detached_batch_newly_retired = set()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            original_states = self._detached_batch_original_states
            newly_retired = self._detached_batch_newly_retired
            self._detached_batch_owners = None
            self._detached_batch_original_states = None
            self._detached_batch_newly_retired = None
            if not succeeded:
                for owner, original in original_states.items():
                    if original is None:
                        self._quote_state.pop(owner, None)
                    else:
                        self._quote_state[owner] = original
                for owner in newly_retired:
                    self._retired_quote_owners.pop(owner, None)

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
        current_owner = self._owners.get(request_id)
        persisted_owner_payload = payload.get("persisted_stream_owner")
        if "persisted_stream_owner" in payload and persisted_owner_payload is None:
            raise ValueError("PERSISTED_STREAM_OWNER_MISSING")
        persisted_owner = (
            None
            if persisted_owner_payload is None
            else stream_owner_from_payload(persisted_owner_payload)
        )
        if persisted_owner is not None and persisted_owner.request_id != request_id:
            raise ValueError("persisted stream owner request ID differs from callback")
        owner = persisted_owner or current_owner
        if owner is None:
            raise ValueError("IBKR callback has no deterministic request owner")
        # The immutable admission receipt wins over mutable ownership. Request
        # IDs may be cancelled and reused before their older lease is polled.
        detached_owner = persisted_owner is not None and current_owner != persisted_owner
        if detached_owner and self._detached_batch_owners is not None:
            self._detached_batch_owners.add(owner)
            if kind == "level1_quote_update":
                was_retired = owner in self._retired_quote_owners
                self._retain_retired_quote_owner(owner)
                if not was_retired:
                    newly_retired = self._detached_batch_newly_retired
                    assert newly_retired is not None
                    newly_retired.add(owner)
            original_states = self._detached_batch_original_states
            assert original_states is not None
            if owner not in original_states:
                current_state = self._quote_state.get(owner)
                original_states[owner] = None if current_state is None else deepcopy(current_state)
        try:
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
                return NormalizedCallback(
                    historical_bar=self._historical_bar(owner, payload, received)
                )
            if kind in {"historical_backfill_end", "ibkr_error"}:
                return NormalizedCallback(
                    control_kind=kind,
                    control_payload=dict(payload),
                    stream_owner=owner,
                )
            raise ValueError(f"unsupported IBKR stream callback kind: {kind}")
        finally:
            # A cancelled stream's admission receipt is lease-local evidence. It
            # must not recreate mutable state beyond one ordered inbox batch.
            if detached_owner and self._detached_batch_owners is None:
                self._quote_state.pop(owner, None)
                self._retired_quote_owners.pop(owner, None)

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
        state = self._quote_state.setdefault(owner, {})
        field = str(payload.get("field", "unknown"))
        if field == "option_computation":
            source = str(payload.get("computation_source", "unknown"))
            if source in {"model", "last", "bid", "ask"}:
                source_values = cast(
                    dict[str, dict[str, float | str | None]],
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
                snapshot["greek_timestamp_utc"] = received.astimezone(UTC).isoformat()
                source_market_data_type = payload.get(
                    "market_data_type",
                    state.get("market_data_type"),
                )
                snapshot["market_data_status"] = (
                    None if source_market_data_type is None else str(source_market_data_type)
                )
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
                    dict[str, dict[str, float | str | None]],
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
    "OptionContractReceipt",
    "StreamKind",
    "StreamOwner",
    "StreamOwnerReceipt",
    "stream_owner_from_payload",
    "stream_owner_payload",
]
