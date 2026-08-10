"""Bounded core planning and exact option discovery for plugin market-data interests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast
from zoneinfo import ZoneInfo

from stocker_runtime.domain import JsonValue, canonical_json_bytes
from stocker_runtime.ideas.contract import DiscoveryReceipt, MarketDataInterest
from stocker_runtime.ingestion.inbox import CallbackFence

_NEW_YORK = ZoneInfo("America/New_York")
MAX_OPTION_PARAMETER_SETS = 32
MAX_OPTION_EXPIRATIONS = 512
MAX_OPTION_STRIKES = 4_096
MAX_EXACT_CONTRACT_CANDIDATES = 16
MAX_MARKET_DATA_LINES = 100
MAX_PLANNING_DEMANDS = 356


@dataclass(frozen=True)
class MarketDataDemand:
    """One static requirement or resolved interest competing for a live data line."""

    source_id: str
    instrument_id: str
    feed_kind: str
    required: bool
    priority: int
    stale_after_us: int
    snapshot: bool = False

    def __post_init__(self) -> None:
        if (
            not self.source_id
            or not self.instrument_id
            or self.feed_kind not in {"quotes", "trades", "bars"}
            or not 0 <= self.priority <= 1_000
            or self.stale_after_us <= 0
        ):
            raise ValueError("market-data demand is invalid")


@dataclass(frozen=True)
class MarketDataCapacity:
    """Hard line ceiling available to one deterministic planner pass."""

    line_limit: int

    def __post_init__(self) -> None:
        if not 1 <= self.line_limit <= MAX_MARKET_DATA_LINES:
            raise ValueError("market-data line limit must be within 1..100")


@dataclass(frozen=True)
class PlannedSubscription:
    instrument_id: str
    feed_kind: str
    required: bool
    priority: int
    stale_after_us: int
    source_ids: tuple[str, ...]
    snapshot: bool


@dataclass(frozen=True)
class MarketDataPlan:
    subscriptions: tuple[PlannedSubscription, ...]
    deferred_source_ids: tuple[str, ...]
    required_complete: bool
    line_count: int


def plan_market_data(
    requirements: Sequence[MarketDataDemand],
    interests: Sequence[MarketDataDemand],
    capacity: MarketDataCapacity,
) -> MarketDataPlan:
    """Deduplicate demands and choose a stable, hard-cap-respecting request set."""

    all_demands = (*requirements, *interests)
    if len(all_demands) > MAX_PLANNING_DEMANDS:
        raise ValueError("market-data planning demand bound exceeded")
    source_ids = tuple(item.source_id for item in all_demands)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("market-data demand source ids must be unique")
    grouped: dict[tuple[str, str], list[MarketDataDemand]] = {}
    for demand in all_demands:
        grouped.setdefault((demand.instrument_id, demand.feed_kind), []).append(demand)
    candidates = tuple(
        PlannedSubscription(
            instrument_id=instrument_id,
            feed_kind=feed_kind,
            required=any(item.required for item in demands),
            priority=max(item.priority for item in demands),
            stale_after_us=min(item.stale_after_us for item in demands),
            source_ids=tuple(sorted(item.source_id for item in demands)),
            snapshot=all(item.snapshot for item in demands),
        )
        for (instrument_id, feed_kind), demands in grouped.items()
    )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                not item.required,
                -item.priority,
                item.instrument_id,
                item.feed_kind,
            ),
        )
    )
    selected = ordered[: capacity.line_limit]
    deferred = ordered[capacity.line_limit :]
    deferred_sources = tuple(source for item in deferred for source in item.source_ids)
    return MarketDataPlan(
        subscriptions=selected,
        deferred_source_ids=deferred_sources,
        required_complete=not any(item.required for item in deferred),
        line_count=len(selected),
    )


@dataclass(frozen=True)
class OptionParameterSet:
    """Bounded metadata returned by IBKR's option-parameter discovery call."""

    exchange: str
    trading_class: str
    multiplier: str
    expirations: tuple[str, ...]
    strikes: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.exchange
            or not self.trading_class
            or not self.multiplier
            or len(self.expirations) > MAX_OPTION_EXPIRATIONS
            or len(self.strikes) > MAX_OPTION_STRIKES
            or any(len(value) != 8 or not value.isdigit() for value in self.expirations)
            or any(not math.isfinite(value) or value <= 0 for value in self.strikes)
        ):
            raise ValueError("option parameter metadata is invalid or unbounded")


@dataclass(frozen=True)
class ContractCandidate:
    """Exact, market-data-only option identity returned by contract details."""

    con_id: int
    symbol: str
    expiry: str
    strike: float
    right: str
    multiplier: str
    exchange: str
    currency: str
    trading_class: str

    def __post_init__(self) -> None:
        if (
            self.con_id <= 0
            or not self.symbol
            or len(self.expiry) != 8
            or not self.expiry.isdigit()
            or not math.isfinite(self.strike)
            or self.strike <= 0
            or self.right not in {"C", "P"}
            or not self.multiplier
            or not self.exchange
            or not self.currency
            or not self.trading_class
        ):
            raise ValueError("option contract candidate is invalid")


class OptionDiscoveryBackend(Protocol):
    """Private market-data-only metadata surface held by the core adapter."""

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


class UnderlyingInstrument(Protocol):
    @property
    def instrument_id(self) -> str: ...

    @property
    def ibkr_con_id(self) -> int | None: ...

    @property
    def kind(self) -> str: ...

    @property
    def symbol(self) -> str: ...

    @property
    def exchange(self) -> str: ...

    @property
    def currency(self) -> str: ...


@dataclass(frozen=True)
class InterestResolutionRequest:
    interest_id: str
    instance_id: str
    interest: MarketDataInterest

    def __post_init__(self) -> None:
        if not self.interest_id or not self.instance_id:
            raise ValueError("interest resolution identity is required")


class InstrumentResolver:
    """Resolve one interest using bounded metadata followed by one exact query."""

    def __init__(
        self,
        backend: OptionDiscoveryBackend,
        *,
        underlyings: Mapping[str, UnderlyingInstrument],
        completed_at_us: Callable[[], int],
        lease_heartbeat: Callable[[], None] | None = None,
    ) -> None:
        self._backend = backend
        self._underlyings = dict(underlyings)
        self._completed_at_us = completed_at_us
        self._lease_heartbeat = lease_heartbeat or (lambda: None)
        self._parameter_cache: dict[tuple[int, str], tuple[OptionParameterSet, ...]] = {}

    def _external_call(self, call: Callable[[], object]) -> object:
        self._lease_heartbeat()
        try:
            return call()
        finally:
            self._lease_heartbeat()

    @staticmethod
    def _receipt_id(material: Mapping[str, JsonValue]) -> str:
        return hashlib.sha256(canonical_json_bytes(material)).hexdigest()

    def _denied(
        self,
        request: InterestResolutionRequest,
        reason: str,
        candidates_inspected: int,
    ) -> DiscoveryReceipt:
        completed = self._completed_at_us()
        material = {
            "interest_id": request.interest_id,
            "interest_key": request.interest.interest_key,
            "instance_id": request.instance_id,
            "status": "denied",
            "reason_code": reason,
            "candidates_inspected": candidates_inspected,
            "completed_at_us": completed,
        }
        return DiscoveryReceipt(
            receipt_id=self._receipt_id(cast(Mapping[str, JsonValue], material)),
            interest_id=request.interest_id,
            interest_key=request.interest.interest_key,
            instance_id=request.instance_id,
            status="denied",
            reason_code=reason,
            candidates_inspected=candidates_inspected,
            completed_at_us=completed,
        )

    def resolve(self, request: InterestResolutionRequest) -> DiscoveryReceipt:
        """Return a deterministic receipt; absence and ambiguity are bounded denials."""

        interest = request.interest
        underlying = self._underlyings.get(interest.underlying_instrument_id)
        if (
            underlying is None
            or underlying.ibkr_con_id is None
            or underlying.kind.lower() != "stock"
        ):
            return self._denied(request, "UNDERLYING_IDENTITY_UNAVAILABLE", 0)
        underlying_con_id = underlying.ibkr_con_id
        parameter_key = (underlying_con_id, underlying.symbol)
        parameter_sets = self._parameter_cache.get(parameter_key)
        if parameter_sets is None:
            parameter_sets = cast(
                tuple[OptionParameterSet, ...],
                self._external_call(
                    lambda: self._backend.option_parameters(
                        underlying_con_id=underlying_con_id,
                        symbol=underlying.symbol,
                    )
                ),
            )
            self._parameter_cache[parameter_key] = parameter_sets
        if len(parameter_sets) > MAX_OPTION_PARAMETER_SETS:
            return self._denied(request, "OPTION_METADATA_BOUND_EXCEEDED", 0)
        as_of_date = datetime.fromtimestamp(interest.as_of_at_us / 1_000_000, _NEW_YORK).date()
        choices: list[tuple[str, float, float, OptionParameterSet]] = []
        for parameters in parameter_sets:
            if parameters.exchange != "SMART":
                continue
            expirations = tuple(sorted(set(parameters.expirations)))
            strikes = tuple(sorted(set(parameters.strikes)))
            if not strikes:
                continue
            nearest = min(
                range(len(strikes)),
                key=lambda index: (
                    abs(strikes[index] - interest.reference_price),
                    strikes[index],
                ),
            )
            selected_index = nearest + interest.strike_offset
            if not 0 <= selected_index < len(strikes):
                continue
            for expiry in expirations:
                expiry_date = datetime.strptime(expiry, "%Y%m%d").date()
                days = (expiry_date - as_of_date).days
                if not (interest.minimum_days_to_expiry <= days <= interest.maximum_days_to_expiry):
                    continue
                choices.append(
                    (
                        expiry,
                        abs(strikes[nearest] - interest.reference_price),
                        strikes[selected_index],
                        parameters,
                    )
                )
        if not choices:
            return self._denied(request, "NO_MATCHING_EXPIRY_OR_STRIKE", 0)
        expiry, _nearest_distance, strike, parameters = min(
            choices,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3].trading_class,
                item[3].multiplier,
            ),
        )
        right = "C" if interest.option_right == "call" else "P"
        candidates = cast(
            tuple[ContractCandidate, ...],
            self._external_call(
                lambda: self._backend.option_contracts(
                    symbol=underlying.symbol,
                    expiry=expiry,
                    strike=strike,
                    right=right,
                    multiplier=parameters.multiplier,
                    trading_class=parameters.trading_class,
                )
            ),
        )
        if len(candidates) > MAX_EXACT_CONTRACT_CANDIDATES:
            return self._denied(request, "EXACT_CONTRACT_BOUND_EXCEEDED", len(candidates))
        exact = tuple(
            item
            for item in candidates
            if item.symbol == underlying.symbol
            and item.expiry == expiry
            and item.strike == strike
            and item.right == right
            and item.multiplier == parameters.multiplier
            and item.trading_class == parameters.trading_class
            and item.exchange == "SMART"
            and item.currency == underlying.currency
        )
        if not exact:
            return self._denied(request, "EXACT_CONTRACT_NOT_FOUND", len(candidates))
        if len({item.con_id for item in exact}) != 1:
            return self._denied(request, "AMBIGUOUS_CONTRACT_IDENTITY", len(candidates))
        selected = min(exact, key=lambda item: item.con_id)
        completed = self._completed_at_us()
        instrument_id = f"ibkr-option-{selected.con_id}"
        material = {
            "interest_id": request.interest_id,
            "interest_key": request.interest.interest_key,
            "instance_id": request.instance_id,
            "status": "resolved",
            "instrument_id": instrument_id,
            "expiry": selected.expiry,
            "strike": selected.strike,
            "option_right": interest.option_right,
            "multiplier": selected.multiplier,
            "candidates_inspected": len(candidates),
            "completed_at_us": completed,
        }
        return DiscoveryReceipt(
            receipt_id=self._receipt_id(cast(Mapping[str, JsonValue], material)),
            interest_id=request.interest_id,
            interest_key=request.interest.interest_key,
            instance_id=request.instance_id,
            status="resolved",
            instrument_id=instrument_id,
            expiry=selected.expiry,
            strike=selected.strike,
            option_right=interest.option_right,
            multiplier=selected.multiplier,
            candidates_inspected=len(candidates),
            completed_at_us=completed,
        )


@dataclass(frozen=True)
class SubscriptionApplyPlan:
    """Externally actionable changes whose durable rows already exist."""

    configured: tuple[object, ...]
    starts: tuple[tuple[object, CallbackFence], ...]
    stops: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            len(self.configured) > MAX_MARKET_DATA_LINES
            or len(self.starts) > MAX_MARKET_DATA_LINES
            or len(self.stops) > MAX_MARKET_DATA_LINES
        ):
            raise ValueError("subscription apply plan bound exceeded")
        configured_ids = tuple(getattr(item, "request_id", None) for item in self.configured)
        start_ids = tuple(fence.request_id for _item, fence in self.starts)
        if (
            any(
                not isinstance(request_id, int) or isinstance(request_id, bool) or request_id < 0
                for request_id in (*configured_ids, *start_ids, *self.stops)
            )
            or len(set(configured_ids)) != len(configured_ids)
            or len(set(start_ids)) != len(start_ids)
            or len(set(self.stops)) != len(self.stops)
            or not set(start_ids).issubset(configured_ids)
        ):
            raise ValueError("subscription apply request ids must be valid and unique")


@dataclass(frozen=True)
class SubscriptionLifecycleResult:
    started_request_ids: tuple[int, ...]
    stopped_request_ids: tuple[int, ...]
    failures: tuple[tuple[str, int, str], ...] = ()


class SubscriptionBackend(Protocol):
    def configure_subscriptions(self, subscriptions: tuple[object, ...]) -> None: ...

    def subscribe(self, fence: CallbackFence) -> None: ...

    def cancel(self, request_id: int) -> None: ...


class SubscriptionController:
    """Apply one pre-persisted lifecycle transition in capacity-safe order."""

    def __init__(self, backend: SubscriptionBackend) -> None:
        self._backend = backend

    def apply(self, plan: SubscriptionApplyPlan) -> SubscriptionLifecycleResult:
        stopped: list[int] = []
        for request_id in plan.stops:
            try:
                self._backend.cancel(request_id)
            except Exception as error:
                return SubscriptionLifecycleResult(
                    (),
                    tuple(stopped),
                    (("cancel", request_id, type(error).__name__),),
                )
            stopped.append(request_id)
        self._backend.configure_subscriptions(plan.configured)
        started: list[int] = []
        failures: list[tuple[str, int, str]] = []
        for _configured, fence in plan.starts:
            if fence.request_id is None:
                raise ValueError("dynamic subscription fence requires a request id")
            try:
                self._backend.subscribe(fence)
            except Exception as error:
                failures.append(("subscribe", fence.request_id, type(error).__name__))
            else:
                started.append(fence.request_id)
        return SubscriptionLifecycleResult(tuple(started), tuple(stopped), tuple(failures))
