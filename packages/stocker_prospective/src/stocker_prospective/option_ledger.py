"""Bounded option discovery and frozen ask-entry/bid-exit shadow outcomes."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from stocker_prospective.events import OptionQuoteEvent
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.options import DteBucket, select_atm_strike


@dataclass(frozen=True)
class OptionContract:
    underlying_con_id: int
    con_id: int | None
    expiry: date
    dte: int
    dte_bucket: DteBucket
    strike: float
    right: Literal["C", "P"]
    multiplier: int
    exchange: str
    trading_class: str

    @property
    def con_id_key(self) -> str:
        if self.con_id is not None:
            return f"conid:{self.con_id}"
        return (
            f"{self.underlying_con_id}|{self.expiry.isoformat()}|"
            f"{self.strike:.12g}|{self.right}|{self.exchange}|{self.trading_class}"
        )


class OptionContractPlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    contracts: tuple[OptionContract, ...]
    requested_contract_count: int
    maximum_contracts: int
    capacity_reduced: bool
    missing_buckets: tuple[str, ...]
    selection_rule: str = "nearest_atm_common_strike_then_symmetric_wings_v0"
    selection_roles: dict[str, tuple[str, ...]] = Field(default_factory=dict)


def _bucket_priority() -> tuple[DteBucket, ...]:
    return (
        DteBucket.ONE_DTE,
        DteBucket.ZERO_DTE,
        DteBucket.THREE_TO_FIVE_DTE,
    )


def _strike_priority(
    *,
    available: list[float],
    underlying_reference: float,
    strike_steps: int,
) -> list[float]:
    atm = select_atm_strike(underlying_reference, available)
    index = available.index(atm)
    selected = [atm]
    for step in range(1, strike_steps + 1):
        above = index + step
        below = index - step
        if above < len(available):
            selected.append(available[above])
        if below >= 0:
            selected.append(available[below])
    return selected


def build_contract_plan(
    *,
    underlying_con_id: int,
    session_date: date,
    underlying_reference: float,
    expiries: dict[DteBucket, date | None],
    strikes_by_expiry_right: dict[tuple[date, str], tuple[float, ...]],
    strike_steps: int,
    maximum_contracts: int,
    exchange: str,
    trading_class: str,
    multiplier: int = 100,
) -> OptionContractPlan:
    """Resolve only common call/put strikes and reduce capacity deterministically."""

    if underlying_reference <= 0.0 or not math.isfinite(underlying_reference):
        raise ValueError("underlying reference must be positive and finite")
    if strike_steps < 0 or maximum_contracts < 0:
        raise ValueError("option selection bounds must be nonnegative")
    by_bucket: dict[DteBucket, list[OptionContract]] = {}
    missing: list[str] = []
    allowed_dte = {
        DteBucket.ZERO_DTE: {0},
        DteBucket.ONE_DTE: {1},
        DteBucket.THREE_TO_FIVE_DTE: {3, 4, 5},
    }
    for bucket in _bucket_priority():
        expiry = expiries.get(bucket)
        if expiry is None:
            missing.append(bucket.value)
            by_bucket[bucket] = []
            continue
        dte = (expiry - session_date).days
        if dte not in allowed_dte[bucket]:
            raise ValueError(f"expiry outside {bucket.value} bucket")
        calls = set(strikes_by_expiry_right.get((expiry, "C"), ()))
        puts = set(strikes_by_expiry_right.get((expiry, "P"), ()))
        common = sorted(
            strike for strike in calls.intersection(puts) if strike > 0.0 and math.isfinite(strike)
        )
        if not common:
            missing.append(bucket.value)
            by_bucket[bucket] = []
            continue
        ordered_strikes = _strike_priority(
            available=common,
            underlying_reference=underlying_reference,
            strike_steps=strike_steps,
        )
        rights: tuple[Literal["C", "P"], ...] = ("C", "P")
        by_bucket[bucket] = [
            OptionContract(
                underlying_con_id=underlying_con_id,
                con_id=None,
                expiry=expiry,
                dte=dte,
                dte_bucket=bucket,
                strike=strike,
                right=right,
                multiplier=multiplier,
                exchange=exchange,
                trading_class=trading_class,
            )
            for strike in ordered_strikes
            for right in rights
        ]
    requested = sum(len(items) for items in by_bucket.values())
    selected: list[OptionContract] = []
    maximum_depth = max((len(items) for items in by_bucket.values()), default=0)
    for priority_index in range(maximum_depth):
        for bucket in _bucket_priority():
            candidates = by_bucket[bucket]
            if priority_index < len(candidates) and len(selected) < maximum_contracts:
                selected.append(candidates[priority_index])
    return OptionContractPlan(
        contracts=tuple(selected),
        requested_contract_count=requested,
        maximum_contracts=maximum_contracts,
        capacity_reduced=len(selected) < requested,
        missing_buckets=tuple(missing),
    )


class QuoteQualityFlags(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    missing_bid: bool
    missing_ask: bool
    zero_bid: bool
    crossed_quote: bool
    locked_quote: bool
    stale_quote: bool
    spread_too_wide: bool
    size_missing: bool
    market_data_not_live: bool
    option_computation_missing: bool
    contract_not_resolved: bool
    subscription_capacity_denied: bool
    subscription_started_late: bool

    @property
    def acceptable_executable_quote(self) -> bool:
        return not any(
            (
                self.missing_bid,
                self.missing_ask,
                self.zero_bid,
                self.crossed_quote,
                self.stale_quote,
                self.market_data_not_live,
                self.contract_not_resolved,
            )
        )


class ShadowOptionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    symbol: str
    con_id: int | None
    expiry: date
    dte: int
    dte_bucket: DteBucket
    strike: float
    right: Literal["C", "P"]
    horizon_minutes: int
    entry_timestamp: datetime
    horizon_timestamp: datetime
    entry_ask: float | None
    entry_bid: float | None
    entry_spread: float | None
    entry_quote_age_seconds: float | None
    exit_bid: float | None
    exit_ask: float | None
    exit_spread: float | None
    exit_quote_age_seconds: float | None
    first_bid_after_horizon: float | None
    first_bid_after_horizon_timestamp: datetime | None
    ask_to_bid_return: float | None
    first_after_horizon_sensitivity_return: float | None
    mid_to_mid_return: float | None
    last_to_last_return: float | None
    dollar_pnl_per_contract: float | None
    premium_at_risk: float | None
    maximum_observed_bid: float | None
    minimum_observed_bid: float | None
    maximum_favourable_return: float | None
    maximum_adverse_return: float | None
    quote_quality_flags: tuple[str, ...]
    entry_quality: QuoteQualityFlags | None
    exit_quality: QuoteQualityFlags | None
    crossed_otm_to_itm: bool | None
    underlying_movement: float | None
    iv_change: float | None
    delta_change: float | None
    gamma_change: float | None
    theta_change: float | None
    primary_return_definition: str = (
        "first_valid_ask_after_entry_to_last_valid_bid_at_or_before_horizon"
    )
    first_ask_after_horizon: float | None = None
    research_only: bool = True
    shadow_quote_pnl: bool = True


def quote_quality_flags(
    quote: OptionQuoteEvent,
    *,
    reference_timestamp: datetime,
    maximum_quote_age: timedelta,
    maximum_spread_fraction: float = 1.0,
    subscription_started_late: bool = False,
    subscription_capacity_denied: bool = False,
) -> QuoteQualityFlags:
    observed = quote.provider_timestamp_utc or quote.received_timestamp_utc
    age = abs(reference_timestamp - observed)
    crossed = quote.bid is not None and quote.ask is not None and quote.bid > quote.ask
    locked = quote.bid is not None and quote.ask is not None and quote.bid == quote.ask
    bid = quote.bid
    ask = quote.ask
    if bid is None or ask is None:
        spread_fraction = None
    else:
        midpoint = (bid + ask) / 2.0
        spread_fraction = None if midpoint <= 0.0 else (ask - bid) / midpoint
    return QuoteQualityFlags(
        missing_bid=quote.bid is None,
        missing_ask=quote.ask is None,
        zero_bid=quote.bid == 0.0,
        crossed_quote=crossed,
        locked_quote=locked,
        stale_quote=age > maximum_quote_age,
        spread_too_wide=(spread_fraction is not None and spread_fraction > maximum_spread_fraction),
        size_missing=quote.bid_size is None or quote.ask_size is None,
        market_data_not_live=quote.market_data_type is not MarketDataType.LIVE,
        option_computation_missing=all(
            value is None
            for value in (
                quote.option_model_price,
                quote.implied_volatility,
                quote.delta,
                quote.gamma,
                quote.theta,
                quote.vega,
            )
        ),
        contract_not_resolved=quote.con_id <= 0,
        subscription_capacity_denied=subscription_capacity_denied,
        subscription_started_late=subscription_started_late,
    )


def _timeline(
    quotes: tuple[OptionQuoteEvent, ...],
    contract: OptionContract,
) -> list[OptionQuoteEvent]:
    return sorted(
        (
            quote
            for quote in quotes
            if (
                quote.expiry == contract.expiry
                and quote.strike == contract.strike
                and quote.right == contract.right
                and (contract.con_id is None or quote.con_id == contract.con_id)
            )
        ),
        key=lambda item: (
            item.ordering_timestamp,
            item.received_monotonic_ns,
            item.source_sequence,
            item.event_id,
        ),
    )


def _first_valid_entry(
    timeline: list[OptionQuoteEvent],
    *,
    entry_timestamp: datetime,
    maximum_quote_age: timedelta,
) -> tuple[OptionQuoteEvent, QuoteQualityFlags] | None:
    for quote in timeline:
        if quote.ordering_timestamp < entry_timestamp:
            continue
        flags = quote_quality_flags(
            quote,
            reference_timestamp=entry_timestamp,
            maximum_quote_age=maximum_quote_age,
            subscription_started_late=quote.ordering_timestamp - entry_timestamp
            > maximum_quote_age,
        )
        if (
            quote.ask is not None
            and quote.ask > 0.0
            and not flags.crossed_quote
            and not flags.stale_quote
            and not flags.market_data_not_live
        ):
            return quote, flags
    return None


def _last_valid_exit(
    timeline: list[OptionQuoteEvent],
    *,
    horizon: datetime,
    maximum_quote_age: timedelta,
) -> tuple[OptionQuoteEvent, QuoteQualityFlags] | None:
    for quote in reversed(timeline):
        if quote.ordering_timestamp > horizon:
            continue
        flags = quote_quality_flags(
            quote,
            reference_timestamp=horizon,
            maximum_quote_age=maximum_quote_age,
        )
        if (
            quote.bid is not None
            and quote.bid >= 0.0
            and not flags.crossed_quote
            and not flags.stale_quote
            and not flags.market_data_not_live
        ):
            return quote, flags
    return None


def _first_valid_after(
    timeline: list[OptionQuoteEvent],
    *,
    horizon: datetime,
    maximum_quote_age: timedelta,
) -> OptionQuoteEvent | None:
    for quote in timeline:
        if quote.ordering_timestamp < horizon:
            continue
        flags = quote_quality_flags(
            quote,
            reference_timestamp=horizon,
            maximum_quote_age=maximum_quote_age,
        )
        if (
            quote.bid is not None
            and quote.bid > 0.0
            and not flags.crossed_quote
            and not flags.stale_quote
            and not flags.market_data_not_live
        ):
            return quote
    return None


def _difference(
    entry: OptionQuoteEvent,
    exit_quote: OptionQuoteEvent,
    field: str,
) -> float | None:
    first = getattr(entry, field)
    second = getattr(exit_quote, field)
    if first is None or second is None:
        return None
    return float(second - first)


def _itm(right: Literal["C", "P"], strike: float, underlying: float) -> bool:
    return underlying > strike if right == "C" else underlying < strike


def _quality_names(flags: QuoteQualityFlags | None) -> list[str]:
    if flags is None:
        return []
    return [name for name, value in flags.model_dump().items() if isinstance(value, bool) and value]


def build_shadow_outcomes(
    *,
    episode_id: str,
    symbol: str,
    entry_timestamp: datetime,
    contracts: tuple[OptionContract, ...],
    quotes: tuple[OptionQuoteEvent, ...],
    horizons: tuple[timedelta, ...],
    maximum_quote_age: timedelta,
) -> tuple[ShadowOptionOutcome, ...]:
    """Freeze every contract/horizon outcome without best-quote selection."""

    outcomes: list[ShadowOptionOutcome] = []
    for contract in contracts:
        timeline = _timeline(quotes, contract)
        entry_result = _first_valid_entry(
            timeline,
            entry_timestamp=entry_timestamp,
            maximum_quote_age=maximum_quote_age,
        )
        for horizon_delta in horizons:
            horizon = entry_timestamp + horizon_delta
            exit_result = _last_valid_exit(
                timeline,
                horizon=horizon,
                maximum_quote_age=maximum_quote_age,
            )
            sensitivity = _first_valid_after(
                timeline,
                horizon=horizon,
                maximum_quote_age=maximum_quote_age,
            )
            entry_quote = None if entry_result is None else entry_result[0]
            entry_flags = None if entry_result is None else entry_result[1]
            exit_quote = None if exit_result is None else exit_result[0]
            exit_flags = None if exit_result is None else exit_result[1]
            observed_bids = [
                quote.bid
                for quote in timeline
                if (
                    entry_timestamp <= quote.ordering_timestamp <= horizon
                    and quote.bid is not None
                    and quote.bid >= 0.0
                )
            ]
            entry_ask = None if entry_quote is None else entry_quote.ask
            exit_bid = None if exit_quote is None else exit_quote.bid
            primary_return = (
                None
                if entry_ask is None or exit_bid is None or entry_ask <= 0.0
                else (exit_bid - entry_ask) / entry_ask
            )
            sensitivity_return = (
                None
                if (
                    entry_ask is None
                    or sensitivity is None
                    or sensitivity.bid is None
                    or entry_ask <= 0.0
                )
                else (sensitivity.bid - entry_ask) / entry_ask
            )
            entry_mid = (
                None
                if entry_quote is None or entry_quote.bid is None or entry_quote.ask is None
                else (entry_quote.bid + entry_quote.ask) / 2.0
            )
            exit_mid = (
                None
                if exit_quote is None or exit_quote.bid is None or exit_quote.ask is None
                else (exit_quote.bid + exit_quote.ask) / 2.0
            )
            quality = list(
                dict.fromkeys(
                    [
                        *(
                            ["missing_valid_entry_ask"]
                            if entry_quote is None
                            else _quality_names(entry_flags)
                        ),
                        *(
                            ["missing_valid_exit_bid"]
                            if exit_quote is None
                            else _quality_names(exit_flags)
                        ),
                    ]
                )
            )
            entry_underlying = (
                None if entry_quote is None else entry_quote.underlying_reference_price
            )
            exit_underlying = None if exit_quote is None else exit_quote.underlying_reference_price
            crossed_otm = (
                None
                if entry_underlying is None or exit_underlying is None
                else (
                    not _itm(contract.right, contract.strike, entry_underlying)
                    and _itm(contract.right, contract.strike, exit_underlying)
                )
            )
            outcomes.append(
                ShadowOptionOutcome(
                    episode_id=episode_id,
                    symbol=symbol,
                    con_id=contract.con_id,
                    expiry=contract.expiry,
                    dte=contract.dte,
                    dte_bucket=contract.dte_bucket,
                    strike=contract.strike,
                    right=contract.right,
                    horizon_minutes=int(horizon_delta.total_seconds() // 60),
                    entry_timestamp=entry_timestamp,
                    horizon_timestamp=horizon,
                    entry_ask=entry_ask,
                    entry_bid=None if entry_quote is None else entry_quote.bid,
                    entry_spread=(
                        None
                        if entry_quote is None or entry_quote.bid is None or entry_quote.ask is None
                        else entry_quote.ask - entry_quote.bid
                    ),
                    entry_quote_age_seconds=(
                        None
                        if entry_quote is None
                        else (entry_quote.ordering_timestamp - entry_timestamp).total_seconds()
                    ),
                    exit_bid=exit_bid,
                    exit_ask=None if exit_quote is None else exit_quote.ask,
                    exit_spread=(
                        None
                        if exit_quote is None or exit_quote.bid is None or exit_quote.ask is None
                        else exit_quote.ask - exit_quote.bid
                    ),
                    exit_quote_age_seconds=(
                        None
                        if exit_quote is None
                        else (horizon - exit_quote.ordering_timestamp).total_seconds()
                    ),
                    first_bid_after_horizon=(None if sensitivity is None else sensitivity.bid),
                    first_bid_after_horizon_timestamp=(
                        None if sensitivity is None else sensitivity.ordering_timestamp
                    ),
                    first_ask_after_horizon=(
                        None if sensitivity is None else sensitivity.ask
                    ),
                    ask_to_bid_return=primary_return,
                    first_after_horizon_sensitivity_return=sensitivity_return,
                    mid_to_mid_return=(
                        None
                        if entry_mid is None or exit_mid is None or entry_mid <= 0.0
                        else (exit_mid - entry_mid) / entry_mid
                    ),
                    last_to_last_return=(
                        None
                        if entry_quote is None
                        or exit_quote is None
                        or entry_quote.last is None
                        or exit_quote.last is None
                        or entry_quote.last <= 0.0
                        else (exit_quote.last - entry_quote.last) / entry_quote.last
                    ),
                    dollar_pnl_per_contract=(
                        None
                        if entry_ask is None or exit_bid is None
                        else round(
                            (exit_bid - entry_ask) * contract.multiplier,
                            12,
                        )
                    ),
                    premium_at_risk=(
                        None if entry_ask is None else entry_ask * contract.multiplier
                    ),
                    maximum_observed_bid=max(observed_bids) if observed_bids else None,
                    minimum_observed_bid=min(observed_bids) if observed_bids else None,
                    maximum_favourable_return=(
                        None
                        if entry_ask is None or not observed_bids
                        else (max(observed_bids) - entry_ask) / entry_ask
                    ),
                    maximum_adverse_return=(
                        None
                        if entry_ask is None or not observed_bids
                        else (min(observed_bids) - entry_ask) / entry_ask
                    ),
                    quote_quality_flags=tuple(quality),
                    entry_quality=entry_flags,
                    exit_quality=exit_flags,
                    crossed_otm_to_itm=crossed_otm,
                    underlying_movement=(
                        None
                        if entry_underlying is None or exit_underlying is None
                        else exit_underlying - entry_underlying
                    ),
                    iv_change=(
                        None
                        if entry_quote is None or exit_quote is None
                        else _difference(entry_quote, exit_quote, "implied_volatility")
                    ),
                    delta_change=(
                        None
                        if entry_quote is None or exit_quote is None
                        else _difference(entry_quote, exit_quote, "delta")
                    ),
                    gamma_change=(
                        None
                        if entry_quote is None or exit_quote is None
                        else _difference(entry_quote, exit_quote, "gamma")
                    ),
                    theta_change=(
                        None
                        if entry_quote is None or exit_quote is None
                        else _difference(entry_quote, exit_quote, "theta")
                    ),
                )
            )
    return tuple(outcomes)


class DirectionalShadowSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    archetype: str
    direction: Literal["CALL", "PUT", "ABSTAIN", "ORACLE"]
    selected_contract_key: str | None
    label: str
    live_decision_panel_visible: bool


def map_directional_shadow(
    *,
    archetype: Literal["A1", "C1", "R1"],
    action: Literal["CALL", "PUT", "ABSTAIN"],
    atm_call: OptionContract,
    atm_put: OptionContract,
) -> DirectionalShadowSelection:
    contract = atm_call if action == "CALL" else atm_put if action == "PUT" else None
    label = (
        "prospective hypothesis — not validated"
        if archetype == "A1"
        else "comparison only — not validated"
    )
    return DirectionalShadowSelection(
        archetype=archetype,
        direction=action,
        selected_contract_key=None if contract is None else contract.con_id_key,
        label=label,
        live_decision_panel_visible=True,
    )


def retrospective_oracle(
    call: ShadowOptionOutcome,
    put: ShadowOptionOutcome,
) -> DirectionalShadowSelection:
    candidates = [item for item in (call, put) if item.ask_to_bid_return is not None]
    selected = (
        None
        if not candidates
        else max(candidates, key=lambda item: item.ask_to_bid_return or -math.inf)
    )
    return DirectionalShadowSelection(
        archetype="oracle",
        direction="ORACLE",
        selected_contract_key=(
            None
            if selected is None
            else f"conid:{selected.con_id}"
            if selected.con_id is not None
            else None
        ),
        label="retrospective oracle — not tradeable",
        live_decision_panel_visible=False,
    )


def straddle_outcome(
    call: ShadowOptionOutcome,
    put: ShadowOptionOutcome,
) -> dict[str, float | str | bool | None]:
    entry = (
        None if call.entry_ask is None or put.entry_ask is None else call.entry_ask + put.entry_ask
    )
    exit_value = (
        None if call.exit_bid is None or put.exit_bid is None else call.exit_bid + put.exit_bid
    )
    return {
        "structure": "ATM_straddle",
        "entry_call_ask_plus_put_ask": entry,
        "exit_call_bid_plus_put_bid": exit_value,
        "ask_to_bid_return": (
            None
            if entry is None or exit_value is None or entry <= 0.0
            else (exit_value - entry) / entry
        ),
        "research_only": True,
    }


def median_quote_age(outcomes: tuple[ShadowOptionOutcome, ...]) -> float | None:
    values = [
        item.exit_quote_age_seconds for item in outcomes if item.exit_quote_age_seconds is not None
    ]
    return None if not values else statistics.median(values)
