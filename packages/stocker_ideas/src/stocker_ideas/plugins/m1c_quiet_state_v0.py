"""Frozen Quiet-State classification and defined-risk option selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

BOTTOM_5_THRESHOLD = 0.115697407847643
BOTTOM_10_THRESHOLD = 0.135896965695626
BOTTOM_20_THRESHOLD = 0.167095528962669
MINIMUM_EPISODE_SPACING_MINUTES = 30.0
MAXIMUM_DELTA_DISTANCE = 0.05
MINIMUM_WING_FRACTION = 0.01

type OptionRight = Literal["call", "put"]
type StructureType = Literal[
    "ATM_IRON_BUTTERFLY",
    "DELTA_IRON_CONDOR",
    "CALL_CREDIT_SPREAD",
    "PUT_CREDIT_SPREAD",
]


@dataclass(frozen=True)
class QuietState:
    probability: float
    previous_probability: float | None
    bottom_5: bool
    bottom_10: bool
    bottom_20: bool
    fresh_episode: bool


@dataclass(frozen=True)
class OptionPanelContract:
    event_id: str
    interest_key: str
    instrument_id: str
    bucket: str
    offset: int
    right: OptionRight
    expiry: str
    strike: float
    multiplier: str
    bid: float
    ask: float
    delta: float | None

    def __post_init__(self) -> None:
        values = (self.strike, self.bid, self.ask)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("option panel values must be finite")
        if self.strike <= 0.0 or self.bid < 0.0 or self.ask < self.bid:
            raise ValueError("option panel quote is invalid")
        if self.delta is not None and not math.isfinite(self.delta):
            raise ValueError("option delta must be finite when present")


@dataclass(frozen=True)
class StructureLeg:
    side: Literal["short", "long"]
    contract: OptionPanelContract
    target_delta: float | None = None


@dataclass(frozen=True)
class StructureAttempt:
    structure_type: StructureType
    available: bool
    legs: tuple[StructureLeg, ...]
    reason: str | None
    opening_credit: float | None


def _probability(value: float) -> float:
    observed = float(value)
    if not math.isfinite(observed) or not 0.0 <= observed <= 1.0:
        raise ValueError("M1C probability must lie in [0, 1]")
    return observed


def classify_quiet_state(
    *,
    probability: float,
    previous_probability: float | None,
    minutes_since_previous_episode: float | None,
) -> QuietState:
    """Apply the frozen inclusive bottom-tail and spacing rules."""

    observed = _probability(probability)
    previous = None if previous_probability is None else _probability(previous_probability)
    if minutes_since_previous_episode is not None and (
        not math.isfinite(minutes_since_previous_episode) or minutes_since_previous_episode < 0.0
    ):
        raise ValueError("episode spacing must be finite and non-negative")
    crossing = observed <= BOTTOM_10_THRESHOLD and (
        previous is None or previous > BOTTOM_10_THRESHOLD
    )
    spaced = (
        minutes_since_previous_episode is None
        or minutes_since_previous_episode >= MINIMUM_EPISODE_SPACING_MINUTES
    )
    return QuietState(
        probability=observed,
        previous_probability=previous,
        bottom_5=observed <= BOTTOM_5_THRESHOLD,
        bottom_10=observed <= BOTTOM_10_THRESHOLD,
        bottom_20=observed <= BOTTOM_20_THRESHOLD,
        fresh_episode=crossing and spaced,
    )


def _unavailable(structure_type: StructureType, reason: str) -> StructureAttempt:
    return StructureAttempt(
        structure_type=structure_type,
        available=False,
        legs=(),
        reason=reason,
        opening_credit=None,
    )


def _available(
    structure_type: StructureType,
    legs: tuple[StructureLeg, ...],
) -> StructureAttempt:
    if any(leg.side == "short" and leg.contract.bid <= 0.0 for leg in legs):
        return _unavailable(structure_type, "short_leg_bid_not_positive")
    if any(leg.side == "long" and leg.contract.ask <= 0.0 for leg in legs):
        return _unavailable(structure_type, "protective_leg_ask_not_positive")
    opening_credit = math.fsum(
        leg.contract.bid if leg.side == "short" else -leg.contract.ask for leg in legs
    )
    if opening_credit <= 0.0:
        return _unavailable(structure_type, "non_positive_opening_credit")
    return StructureAttempt(
        structure_type=structure_type,
        available=True,
        legs=legs,
        reason=None,
        opening_credit=opening_credit,
    )


def _scope_is_valid(contracts: tuple[OptionPanelContract, ...]) -> bool:
    return (
        bool(contracts)
        and len({item.bucket for item in contracts}) == 1
        and len({item.expiry for item in contracts}) == 1
        and len({item.multiplier for item in contracts}) == 1
        and len({(item.offset, item.right) for item in contracts}) == len(contracts)
    )


def _at(
    contracts: tuple[OptionPanelContract, ...],
    *,
    strike: float,
    right: OptionRight,
) -> OptionPanelContract | None:
    return next(
        (item for item in contracts if item.strike == strike and item.right == right),
        None,
    )


def _atm_strike(contracts: tuple[OptionPanelContract, ...], reference: float) -> float | None:
    calls = {item.strike for item in contracts if item.right == "call"}
    puts = {item.strike for item in contracts if item.right == "put"}
    common = calls & puts
    return min(common, key=lambda strike: (abs(strike - reference), strike)) if common else None


def _iron_butterfly(
    contracts: tuple[OptionPanelContract, ...], reference: float
) -> StructureAttempt:
    name: StructureType = "ATM_IRON_BUTTERFLY"
    atm = _atm_strike(contracts, reference)
    if atm is None:
        return _unavailable(name, "atm_pair_unavailable")
    short_call = _at(contracts, strike=atm, right="call")
    short_put = _at(contracts, strike=atm, right="put")
    minimum = MINIMUM_WING_FRACTION * reference
    pairs: list[tuple[float, OptionPanelContract, OptionPanelContract]] = []
    for upper in contracts:
        if upper.right != "call" or upper.strike <= atm:
            continue
        distance = upper.strike - atm
        if distance + 1e-12 < minimum:
            continue
        lower = _at(contracts, strike=atm - distance, right="put")
        if lower is not None:
            pairs.append((distance, upper, lower))
    if short_call is None or short_put is None:
        return _unavailable(name, "atm_pair_unavailable")
    if not pairs:
        return _unavailable(name, "symmetric_wings_unavailable")
    _distance, long_call, long_put = min(
        pairs,
        key=lambda item: (
            item[0],
            item[1].strike,
            item[1].instrument_id,
            item[2].instrument_id,
        ),
    )
    return _available(
        name,
        (
            StructureLeg("short", short_call),
            StructureLeg("short", short_put),
            StructureLeg("long", long_call),
            StructureLeg("long", long_put),
        ),
    )


def _nearest_delta(
    contracts: tuple[OptionPanelContract, ...],
    *,
    right: OptionRight,
    target: float,
) -> tuple[OptionPanelContract, float] | None:
    candidates = tuple(
        (abs(item.delta - target), item)
        for item in contracts
        if item.right == right and item.delta is not None
    )
    if not candidates:
        return None
    distance, contract = min(
        candidates,
        key=lambda item: (item[0], item[1].strike, item[1].instrument_id),
    )
    return contract, distance


def _delta_condor(contracts: tuple[OptionPanelContract, ...]) -> StructureAttempt:
    name: StructureType = "DELTA_IRON_CONDOR"
    selected = (
        _nearest_delta(contracts, right="call", target=0.25),
        _nearest_delta(contracts, right="put", target=-0.25),
        _nearest_delta(contracts, right="call", target=0.10),
        _nearest_delta(contracts, right="put", target=-0.10),
    )
    if any(item is None for item in selected):
        return _unavailable(name, "delta_contract_unavailable")
    resolved = tuple(item for item in selected if item is not None)
    if any(distance > MAXIMUM_DELTA_DISTANCE for _contract, distance in resolved):
        return _unavailable(name, "delta_tolerance_failed")
    short_call, short_put, long_call, long_put = (item[0] for item in resolved)
    if not (long_put.strike < short_put.strike < short_call.strike < long_call.strike):
        return _unavailable(name, "invalid_strike_ordering")
    return _available(
        name,
        (
            StructureLeg("short", short_call, 0.25),
            StructureLeg("short", short_put, -0.25),
            StructureLeg("long", long_call, 0.10),
            StructureLeg("long", long_put, -0.10),
        ),
    )


def _credit_spread(
    contracts: tuple[OptionPanelContract, ...],
    reference: float,
    *,
    right: OptionRight,
) -> StructureAttempt:
    name: StructureType = "CALL_CREDIT_SPREAD" if right == "call" else "PUT_CREDIT_SPREAD"
    atm = _atm_strike(contracts, reference)
    short = None if atm is None else _at(contracts, strike=atm, right=right)
    if short is None:
        return _unavailable(name, "short_leg_unavailable")
    minimum = MINIMUM_WING_FRACTION * reference
    if right == "call":
        candidates = tuple(
            item
            for item in contracts
            if item.right == right and item.strike - short.strike + 1e-12 >= minimum
        )
        wing = min(candidates, key=lambda item: (item.strike, item.instrument_id), default=None)
    else:
        candidates = tuple(
            item
            for item in contracts
            if item.right == right and short.strike - item.strike + 1e-12 >= minimum
        )
        wing = max(candidates, key=lambda item: (item.strike, item.instrument_id), default=None)
    if wing is None:
        return _unavailable(name, "protective_wing_unavailable")
    return _available(
        name,
        (StructureLeg("short", short), StructureLeg("long", wing)),
    )


def select_defined_risk_structures(
    *,
    contracts: tuple[OptionPanelContract, ...],
    underlying_reference_price: float,
) -> tuple[StructureAttempt, ...]:
    """Select the four frozen short-premium attempts for one expiry bucket."""

    reference = float(underlying_reference_price)
    if not math.isfinite(reference) or reference <= 0.0 or not _scope_is_valid(contracts):
        names: tuple[StructureType, ...] = (
            "ATM_IRON_BUTTERFLY",
            "DELTA_IRON_CONDOR",
            "CALL_CREDIT_SPREAD",
            "PUT_CREDIT_SPREAD",
        )
        return tuple(_unavailable(name, "invalid_contract_scope") for name in names)
    return (
        _iron_butterfly(contracts, reference),
        _delta_condor(contracts),
        _credit_spread(contracts, reference, right="call"),
        _credit_spread(contracts, reference, right="put"),
    )


__all__ = [
    "BOTTOM_5_THRESHOLD",
    "BOTTOM_10_THRESHOLD",
    "BOTTOM_20_THRESHOLD",
    "MAXIMUM_DELTA_DISTANCE",
    "MINIMUM_EPISODE_SPACING_MINUTES",
    "MINIMUM_WING_FRACTION",
    "OptionPanelContract",
    "QuietState",
    "StructureAttempt",
    "StructureLeg",
    "classify_quiet_state",
    "select_defined_risk_structures",
]
