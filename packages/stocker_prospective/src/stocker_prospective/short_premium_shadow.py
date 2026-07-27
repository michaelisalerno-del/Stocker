"""Conservative defined-risk option structures for record-only shadow outcomes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from stocker_prospective.events import OptionQuoteEvent
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.option_ledger import OptionContract
from stocker_prospective.options import DteBucket, select_atm_strike

MAXIMUM_DELTA_DISTANCE: Final[float] = 0.05
MINIMUM_WING_FRACTION: Final[float] = 0.01
FIXED_WIDTH_FRACTION: Final[float] = 0.01
MAXIMUM_QUOTE_AGE: Final[timedelta] = timedelta(seconds=5)
MAXIMUM_SPREAD_FRACTION: Final[float] = 1.0


class StructureType(StrEnum):
    ATM_IRON_BUTTERFLY = "ATM_IRON_BUTTERFLY"
    DELTA_IRON_CONDOR = "DELTA_IRON_CONDOR"
    CALL_CREDIT_SPREAD = "CALL_CREDIT_SPREAD"
    PUT_CREDIT_SPREAD = "PUT_CREDIT_SPREAD"


@dataclass(frozen=True)
class StructureLeg:
    side: Literal["short", "long"]
    contract: OptionContract
    target_delta: float | None = None


@dataclass(frozen=True)
class DefinedRiskStructure:
    structure_type: StructureType
    dte_bucket: DteBucket | None
    legs: tuple[StructureLeg, ...]
    available: bool
    quality_flags: tuple[str, ...]
    delta_distances: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        short_legs = tuple(leg for leg in self.legs if leg.side == "short")
        long_legs = tuple(leg for leg in self.legs if leg.side == "long")
        if self.available and (not short_legs or not long_legs):
            raise ValueError("a defined-risk structure requires short and protective long legs")
        if any(leg.contract.con_id is None for leg in self.legs):
            raise ValueError("shadow structure legs must be exact resolved contracts")


def _unavailable(
    structure_type: StructureType,
    contracts: Iterable[OptionContract],
    *flags: str,
) -> DefinedRiskStructure:
    observed = tuple(contracts)
    buckets = {contract.dte_bucket for contract in observed}
    return DefinedRiskStructure(
        structure_type=structure_type,
        dte_bucket=next(iter(buckets)) if len(buckets) == 1 else None,
        legs=(),
        available=False,
        quality_flags=tuple(sorted(set(flags))),
    )


def _validate_contract_scope(
    contracts: tuple[OptionContract, ...],
) -> tuple[DteBucket, object] | None:
    if not contracts:
        return None
    buckets = {contract.dte_bucket for contract in contracts}
    expiries = {contract.expiry for contract in contracts}
    if len(buckets) != 1 or len(expiries) != 1:
        return None
    return next(iter(buckets)), next(iter(expiries))


def _contract_at(
    contracts: tuple[OptionContract, ...],
    *,
    strike: float,
    right: Literal["C", "P"],
) -> OptionContract | None:
    return next(
        (
            contract
            for contract in contracts
            if contract.strike == strike and contract.right == right
        ),
        None,
    )


def select_iron_butterfly(
    *,
    contracts: Iterable[OptionContract],
    underlying_entry_price: float,
) -> DefinedRiskStructure:
    """Freeze ATM shorts and nearest symmetric wings at least one percent away."""

    observed = tuple(contracts)
    if (
        not math.isfinite(underlying_entry_price)
        or underlying_entry_price <= 0.0
        or _validate_contract_scope(observed) is None
    ):
        return _unavailable(
            StructureType.ATM_IRON_BUTTERFLY,
            observed,
            "invalid_contract_scope",
        )
    call_strikes = {item.strike for item in observed if item.right == "C"}
    put_strikes = {item.strike for item in observed if item.right == "P"}
    common = sorted(call_strikes.intersection(put_strikes))
    if not common:
        return _unavailable(
            StructureType.ATM_IRON_BUTTERFLY,
            observed,
            "atm_pair_unavailable",
        )
    atm = select_atm_strike(underlying_entry_price, common)
    short_call = _contract_at(observed, strike=atm, right="C")
    short_put = _contract_at(observed, strike=atm, right="P")
    minimum_distance = MINIMUM_WING_FRACTION * underlying_entry_price
    pairs: list[tuple[float, OptionContract, OptionContract]] = []
    for upper in observed:
        if upper.right != "C" or upper.strike <= atm:
            continue
        distance = upper.strike - atm
        if distance + 1e-12 < minimum_distance:
            continue
        lower = _contract_at(observed, strike=atm - distance, right="P")
        if lower is not None:
            pairs.append((distance, upper, lower))
    if short_call is None or short_put is None:
        return _unavailable(
            StructureType.ATM_IRON_BUTTERFLY,
            observed,
            "atm_pair_unavailable",
        )
    if not pairs:
        return _unavailable(
            StructureType.ATM_IRON_BUTTERFLY,
            observed,
            "asymmetric_wings",
            "symmetric_wings_unavailable",
        )
    _, long_call, long_put = min(
        pairs,
        key=lambda item: (
            item[0],
            item[1].strike,
            item[1].con_id or 0,
            item[2].con_id or 0,
        ),
    )
    return DefinedRiskStructure(
        structure_type=StructureType.ATM_IRON_BUTTERFLY,
        dte_bucket=short_call.dte_bucket,
        legs=(
            StructureLeg(side="short", contract=short_call),
            StructureLeg(side="short", contract=short_put),
            StructureLeg(side="long", contract=long_call),
            StructureLeg(side="long", contract=long_put),
        ),
        available=True,
        quality_flags=(),
    )


def _quote_key(quote: OptionQuoteEvent) -> tuple[object, float, str, int]:
    return quote.expiry, quote.strike, quote.right, quote.con_id


def _contract_key(contract: OptionContract) -> tuple[object, float, str, int]:
    return contract.expiry, contract.strike, contract.right, contract.con_id or 0


def _quote_map(
    quotes: Iterable[OptionQuoteEvent],
) -> dict[tuple[object, float, str, int], OptionQuoteEvent]:
    output: dict[tuple[object, float, str, int], OptionQuoteEvent] = {}
    for quote in sorted(
        quotes,
        key=lambda item: (
            item.ordering_timestamp,
            item.received_monotonic_ns,
            item.source_sequence,
            item.event_id,
        ),
    ):
        output[_quote_key(quote)] = quote
    return output


def _nearest_delta(
    *,
    contracts: tuple[OptionContract, ...],
    quotes: Mapping[tuple[object, float, str, int], OptionQuoteEvent],
    right: Literal["C", "P"],
    target: float,
) -> tuple[OptionContract, float] | None:
    candidates: list[tuple[float, float, int, OptionContract]] = []
    for contract in contracts:
        if contract.right != right:
            continue
        quote = quotes.get(_contract_key(contract))
        if quote is None or quote.delta is None or not math.isfinite(quote.delta):
            continue
        distance = abs(float(quote.delta) - target)
        candidates.append((distance, contract.strike, contract.con_id or 0, contract))
    if not candidates:
        return None
    distance, _strike, _con_id, contract = min(candidates)
    return contract, distance


def select_delta_iron_condor(
    *,
    contracts: Iterable[OptionContract],
    entry_quotes: Iterable[OptionQuoteEvent],
    maximum_delta_distance: float = MAXIMUM_DELTA_DISTANCE,
) -> DefinedRiskStructure:
    """Select exact +/-.25 shorts and +/-.10 protective wings."""

    if maximum_delta_distance != MAXIMUM_DELTA_DISTANCE and (
        maximum_delta_distance <= 0.0 or maximum_delta_distance > MAXIMUM_DELTA_DISTANCE
    ):
        raise ValueError("delta tolerance cannot exceed the frozen maximum")
    observed = tuple(contracts)
    if _validate_contract_scope(observed) is None:
        return _unavailable(
            StructureType.DELTA_IRON_CONDOR,
            observed,
            "invalid_contract_scope",
        )
    quotes = _quote_map(entry_quotes)
    selections = (
        _nearest_delta(contracts=observed, quotes=quotes, right="C", target=0.25),
        _nearest_delta(contracts=observed, quotes=quotes, right="P", target=-0.25),
        _nearest_delta(contracts=observed, quotes=quotes, right="C", target=0.10),
        _nearest_delta(contracts=observed, quotes=quotes, right="P", target=-0.10),
    )
    if any(selection is None for selection in selections):
        return _unavailable(
            StructureType.DELTA_IRON_CONDOR,
            observed,
            "missing_greek",
        )
    resolved = tuple(selection for selection in selections if selection is not None)
    distances = tuple(distance for _contract, distance in resolved)
    if any(distance > maximum_delta_distance for distance in distances):
        unavailable = _unavailable(
            StructureType.DELTA_IRON_CONDOR,
            observed,
            "delta_tolerance_failed",
        )
        return DefinedRiskStructure(
            **{
                **unavailable.__dict__,
                "delta_distances": distances,
            }
        )
    short_call, short_put, long_call, long_put = (selection[0] for selection in resolved)
    if not (long_put.strike < short_put.strike < short_call.strike < long_call.strike):
        return _unavailable(
            StructureType.DELTA_IRON_CONDOR,
            observed,
            "invalid_strike_ordering",
        )
    call_width = long_call.strike - short_call.strike
    put_width = short_put.strike - long_put.strike
    quality_flags = (
        ()
        if math.isclose(call_width, put_width, rel_tol=0.0, abs_tol=1e-12)
        else ("asymmetric_wings",)
    )
    return DefinedRiskStructure(
        structure_type=StructureType.DELTA_IRON_CONDOR,
        dte_bucket=short_call.dte_bucket,
        legs=(
            StructureLeg(side="short", contract=short_call, target_delta=0.25),
            StructureLeg(side="short", contract=short_put, target_delta=-0.25),
            StructureLeg(side="long", contract=long_call, target_delta=0.10),
            StructureLeg(side="long", contract=long_put, target_delta=-0.10),
        ),
        available=True,
        quality_flags=quality_flags,
        delta_distances=distances,
    )


def select_fixed_width_credit_spread(
    *,
    contracts: Iterable[OptionContract],
    underlying_entry_price: float,
    right: Literal["C", "P"],
) -> DefinedRiskStructure:
    """Select an ATM short and nearest same-expiry protective one-percent wing."""

    observed = tuple(contracts)
    structure_type = (
        StructureType.CALL_CREDIT_SPREAD if right == "C" else StructureType.PUT_CREDIT_SPREAD
    )
    if (
        not math.isfinite(underlying_entry_price)
        or underlying_entry_price <= 0.0
        or _validate_contract_scope(observed) is None
    ):
        return _unavailable(structure_type, observed, "invalid_contract_scope")
    right_contracts = tuple(item for item in observed if item.right == right)
    if not right_contracts:
        return _unavailable(structure_type, observed, "short_leg_unavailable")
    short_strike = select_atm_strike(
        underlying_entry_price,
        (contract.strike for contract in right_contracts),
    )
    short = _contract_at(observed, strike=short_strike, right=right)
    minimum_width = FIXED_WIDTH_FRACTION * underlying_entry_price
    if right == "C":
        wings = [
            contract
            for contract in right_contracts
            if contract.strike - short_strike + 1e-12 >= minimum_width
        ]
        wings.sort(key=lambda contract: (contract.strike, contract.con_id or 0))
    else:
        wings = [
            contract
            for contract in right_contracts
            if short_strike - contract.strike + 1e-12 >= minimum_width
        ]
        wings.sort(key=lambda contract: (-contract.strike, contract.con_id or 0))
    if short is None or not wings:
        return _unavailable(structure_type, observed, "protective_wing_unavailable")
    return DefinedRiskStructure(
        structure_type=structure_type,
        dte_bucket=short.dte_bucket,
        legs=(
            StructureLeg(side="short", contract=short),
            StructureLeg(side="long", contract=wings[0]),
        ),
        available=True,
        quality_flags=(),
    )


class CreditShadowOutcome(BaseModel):
    """Conservative bid/ask record for one defined-risk structure and horizon."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structure_type: StructureType
    dte_bucket: DteBucket | None
    horizon_minutes: int
    attempted: bool
    complete_quote_quality: bool
    strict_quote_quality: bool
    quote_quality_status: str
    quote_quality_flags: tuple[str, ...]
    opening_net_credit: float | None
    closing_debit: float | None
    commission_free_pnl: float | None
    configured_commission_pnl: float | None
    maximum_defined_risk: float | None
    return_on_maximum_risk: float | None
    return_on_opening_credit: float | None
    breakeven_strikes: tuple[float, ...]
    underlying_maximum_excursion: float | None
    short_strike_touched: bool | None
    short_strike_crossed: bool | None
    protective_wing_touched: bool | None
    maximum_adverse_marked_pnl: float | None
    maximum_favourable_marked_pnl: float | None
    configured_commission_per_contract: float
    conservative_fill_convention: str = "open_short_bid_long_ask_close_short_ask_long_bid"
    midpoint_pnl_primary: bool = False
    research_only: bool = True
    shadow_outcome_only: bool = True
    defined_risk: bool = True


def _quality_flags(
    *,
    structure: DefinedRiskStructure,
    quotes: Mapping[tuple[object, float, str, int], OptionQuoteEvent],
    reference_timestamp: datetime,
    opening: bool,
) -> set[str]:
    flags = set(structure.quality_flags)
    for leg in structure.legs:
        quote = quotes.get(_contract_key(leg.contract))
        required = (
            quote.bid
            if quote is not None
            and ((opening and leg.side == "short") or (not opening and leg.side == "long"))
            else quote.ask
            if quote is not None
            else None
        )
        if quote is None or required is None:
            flags.add("missing_leg_quote")
            continue
        if quote.bid == 0.0:
            flags.add("zero_bid")
        if quote.bid is not None and quote.ask is not None:
            if quote.bid > quote.ask:
                flags.add("crossed_quote")
            if quote.bid == quote.ask:
                flags.add("locked_quote")
            midpoint = (quote.bid + quote.ask) / 2.0
            if midpoint > 0.0 and (quote.ask - quote.bid) / midpoint > MAXIMUM_SPREAD_FRACTION:
                flags.add("excessive_spread")
        observed = quote.provider_timestamp_utc or quote.received_timestamp_utc
        if abs(reference_timestamp - observed) > MAXIMUM_QUOTE_AGE:
            flags.add("stale_quote")
        if quote.market_data_type is not MarketDataType.LIVE:
            flags.add("market_data_not_live")
        if any(
            value is None
            for value in (
                quote.implied_volatility,
                quote.delta,
                quote.gamma,
                quote.theta,
                quote.vega,
            )
        ):
            flags.add("missing_greek")
    return flags


def _net_value(
    structure: DefinedRiskStructure,
    quotes: Mapping[tuple[object, float, str, int], OptionQuoteEvent],
    *,
    opening: bool,
) -> float | None:
    value = 0.0
    for leg in structure.legs:
        quote = quotes.get(_contract_key(leg.contract))
        if quote is None:
            return None
        if opening:
            price = quote.bid if leg.side == "short" else quote.ask
        else:
            price = quote.ask if leg.side == "short" else quote.bid
        if price is None or not math.isfinite(price):
            return None
        value += price if leg.side == "short" else -price
    multiplier = structure.legs[0].contract.multiplier
    return value * multiplier


def _width(structure: DefinedRiskStructure) -> float | None:
    short_calls = [
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "short" and leg.contract.right == "C"
    ]
    long_calls = [
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "long" and leg.contract.right == "C"
    ]
    short_puts = [
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "short" and leg.contract.right == "P"
    ]
    long_puts = [
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "long" and leg.contract.right == "P"
    ]
    widths: list[float] = []
    if short_calls and long_calls:
        widths.append(min(long_calls) - max(short_calls))
    if short_puts and long_puts:
        widths.append(min(short_puts) - max(long_puts))
    valid = [abs(value) for value in widths if value != 0.0]
    return max(valid) if valid else None


def _breakevens(
    structure: DefinedRiskStructure,
    credit_per_share: float,
) -> tuple[float, ...]:
    short_calls = sorted(
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "short" and leg.contract.right == "C"
    )
    short_puts = sorted(
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "short" and leg.contract.right == "P"
    )
    values: list[float] = []
    if short_puts:
        values.append(short_puts[0] - credit_per_share)
    if short_calls:
        values.append(short_calls[-1] + credit_per_share)
    return tuple(values)


def calculate_credit_shadow(
    *,
    structure: DefinedRiskStructure,
    entry_quotes: Iterable[OptionQuoteEvent],
    exit_quotes: Iterable[OptionQuoteEvent],
    entry_timestamp: datetime,
    exit_timestamp: datetime,
    underlying_path: Iterable[float],
    mark_quote_surfaces: Iterable[Iterable[OptionQuoteEvent]] = (),
    additional_quality_flags: Iterable[str] = (),
    configured_commission_per_contract: float = 0.65,
) -> CreditShadowOutcome:
    """Apply frozen conservative fills without suppressing failed attempts."""

    if entry_timestamp.tzinfo is None or exit_timestamp.tzinfo is None:
        raise ValueError("shadow timestamps must be timezone-aware")
    if exit_timestamp < entry_timestamp:
        raise ValueError("shadow exit cannot precede entry")
    if configured_commission_per_contract < 0.0:
        raise ValueError("commission sensitivity cannot be negative")
    entry_map = _quote_map(entry_quotes)
    exit_map = _quote_map(exit_quotes)
    flags = set((*structure.quality_flags, *additional_quality_flags))
    if not structure.available:
        flags.add("structure_unavailable")
    flags.update(
        _quality_flags(
            structure=structure,
            quotes=entry_map,
            reference_timestamp=entry_timestamp,
            opening=True,
        )
    )
    flags.update(
        _quality_flags(
            structure=structure,
            quotes=exit_map,
            reference_timestamp=exit_timestamp,
            opening=False,
        )
    )
    opening_credit = _net_value(structure, entry_map, opening=True) if structure.available else None
    closing_debit = _net_value(structure, exit_map, opening=False) if structure.available else None
    if structure.available and closing_debit is None:
        flags.add("exit_quote_unavailable")
    if opening_credit is not None and opening_credit <= 0.0:
        flags.add("negative_or_zero_opening_credit")
    pnl = (
        opening_credit - closing_debit
        if opening_credit is not None and closing_debit is not None
        else None
    )
    marked_pnls: list[float] = []
    if opening_credit is not None:
        for raw_surface in mark_quote_surfaces:
            marked_debit = _net_value(
                structure,
                _quote_map(raw_surface),
                opening=False,
            )
            if marked_debit is None:
                flags.add("incomplete_mark_surface")
            else:
                marked_pnls.append(opening_credit - marked_debit)
    if pnl is not None:
        marked_pnls.append(pnl)
    width = _width(structure) if structure.available else None
    multiplier = structure.legs[0].contract.multiplier if structure.legs else 100
    maximum_risk = (
        width * multiplier - opening_credit
        if width is not None and opening_credit is not None
        else None
    )
    if structure.available and (maximum_risk is None or maximum_risk <= 0.0):
        flags.add("maximum_risk_too_small_or_undefined")
        maximum_risk = None
    path = tuple(float(value) for value in underlying_path)
    valid_path = bool(path) and all(math.isfinite(value) for value in path)
    short_calls = [
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "short" and leg.contract.right == "C"
    ]
    short_puts = [
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "short" and leg.contract.right == "P"
    ]
    long_calls = [
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "long" and leg.contract.right == "C"
    ]
    long_puts = [
        leg.contract.strike
        for leg in structure.legs
        if leg.side == "long" and leg.contract.right == "P"
    ]
    minimum = min(path) if valid_path else None
    maximum = max(path) if valid_path else None
    short_touched = (
        any(maximum is not None and maximum >= strike for strike in short_calls)
        or any(minimum is not None and minimum <= strike for strike in short_puts)
        if valid_path
        else None
    )
    short_crossed = (
        any(maximum is not None and maximum > strike for strike in short_calls)
        or any(minimum is not None and minimum < strike for strike in short_puts)
        if valid_path
        else None
    )
    wing_touched = (
        any(maximum is not None and maximum >= strike for strike in long_calls)
        or any(minimum is not None and minimum <= strike for strike in long_puts)
        if valid_path
        else None
    )
    missing_flags = {"missing_leg_quote", "structure_unavailable"}
    complete = not bool(flags.intersection(missing_flags))
    strict = complete and not flags
    status = "strict_quality" if strict else "complete_quote_quality" if complete else "incomplete"
    credit_per_share = opening_credit / multiplier if opening_credit is not None else None
    commission_pnl = (
        pnl - configured_commission_per_contract * len(structure.legs) * 2
        if pnl is not None
        else None
    )
    return CreditShadowOutcome(
        structure_type=structure.structure_type,
        dte_bucket=structure.dte_bucket,
        horizon_minutes=int((exit_timestamp - entry_timestamp).total_seconds() // 60),
        attempted=True,
        complete_quote_quality=complete,
        strict_quote_quality=strict,
        quote_quality_status=status,
        quote_quality_flags=tuple(sorted(flags)),
        opening_net_credit=opening_credit,
        closing_debit=closing_debit,
        commission_free_pnl=pnl,
        configured_commission_pnl=commission_pnl,
        maximum_defined_risk=maximum_risk,
        return_on_maximum_risk=(
            pnl / maximum_risk if pnl is not None and maximum_risk is not None else None
        ),
        return_on_opening_credit=(
            pnl / opening_credit
            if pnl is not None and opening_credit is not None and opening_credit > 0.0
            else None
        ),
        breakeven_strikes=(
            _breakevens(structure, credit_per_share) if credit_per_share is not None else ()
        ),
        underlying_maximum_excursion=(
            max(abs(value - path[0]) for value in path) if valid_path else None
        ),
        short_strike_touched=short_touched,
        short_strike_crossed=short_crossed,
        protective_wing_touched=wing_touched,
        maximum_adverse_marked_pnl=min(marked_pnls) if marked_pnls else None,
        maximum_favourable_marked_pnl=max(marked_pnls) if marked_pnls else None,
        configured_commission_per_contract=configured_commission_per_contract,
    )


__all__ = [
    "FIXED_WIDTH_FRACTION",
    "MAXIMUM_DELTA_DISTANCE",
    "MINIMUM_WING_FRACTION",
    "CreditShadowOutcome",
    "DefinedRiskStructure",
    "StructureLeg",
    "StructureType",
    "calculate_credit_shadow",
    "select_delta_iron_condor",
    "select_fixed_width_credit_spread",
    "select_iron_butterfly",
]
