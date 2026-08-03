"""Record-only option capital, ROI, and Greek-attribution calculations."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stocker_prospective.events import OptionQuoteEvent

MARGIN_UNAVAILABLE: Literal["MARGIN_UNAVAILABLE"] = "MARGIN_UNAVAILABLE"
THETA_ATTRIBUTION_INCOMPLETE: Literal["THETA_ATTRIBUTION_INCOMPLETE"] = (
    "THETA_ATTRIBUTION_INCOMPLETE"
)
type MarginValue = float | Literal["MARGIN_UNAVAILABLE"]
type GreekSource = Literal["bid", "ask", "last", "model"]
type PrimaryCapitalBasis = Literal[
    "premium_paid",
    "cash_secured_capital",
    "maximum_defined_risk",
]


class StrategyType(StrEnum):
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
    DEFINED_RISK_OPTION = "DEFINED_RISK_OPTION"
    LONG_OPTION = "LONG_OPTION"
    SHORT_PUT = "SHORT_PUT"


class TransactionCosts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commissions: float = Field(default=0.0, ge=0.0)
    regulatory_fees: float = Field(default=0.0, ge=0.0)
    exchange_fees: float = Field(default=0.0, ge=0.0)

    @property
    def total(self) -> float:
        return self.commissions + self.regulatory_fees + self.exchange_fees


class MarginEstimate(BaseModel):
    """One provider margin estimate accepted only when explicitly reliable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_margin: float | None = Field(default=None, gt=0.0)
    maintenance_margin: float | None = Field(default=None, gt=0.0)
    reliable: bool = False

    @model_validator(mode="after")
    def _complete_when_reliable(self) -> MarginEstimate:
        if self.reliable and (self.initial_margin is None or self.maintenance_margin is None):
            raise ValueError("reliable margin estimate requires both margin values")
        return self


class GreekSourceSnapshot(BaseModel):
    """One IBKR option-computation source, never filled from another source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    option_model_price: float | None = None
    underlying_model_reference_price: float | None = None
    greek_timestamp: datetime | None = None
    market_data_status: str = Field(min_length=1)

    @field_validator("greek_timestamp")
    @classmethod
    def _aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Greek timestamp must be timezone-aware")
        return value.astimezone(UTC)


class OptionLeg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    leg_id: str = Field(min_length=1)
    con_id: int | None = Field(default=None, gt=0)
    right: Literal["C", "P"]
    strike: float = Field(gt=0.0)
    multiplier: int = Field(gt=0)
    signed_contract_quantity: int

    @model_validator(mode="after")
    def _nonzero_quantity(self) -> OptionLeg:
        if self.signed_contract_quantity == 0:
            raise ValueError("option leg quantity cannot be zero")
        return self


class OptionLegQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    leg_id: str = Field(min_length=1)
    quote_timestamp: datetime
    bid: float | None
    ask: float | None
    last: float | None
    market_data_status: str = Field(min_length=1)
    greeks_by_source: dict[GreekSource, GreekSourceSnapshot] = Field(default_factory=dict)

    @field_validator("quote_timestamp")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("option quote timestamp must be timezone-aware")
        return value.astimezone(UTC)


class PositionGreeks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None


class GreekAttribution(BaseModel):
    """One diagnostic Taylor interval in frozen IBKR units."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    delta_contribution: float | None
    gamma_contribution: float | None
    theta_contribution: float | None
    vega_contribution: float | None
    total_estimated_contribution: float | None
    model_or_midpoint_change: float | None
    change_basis: Literal["model", "midpoint"] | None
    greek_residual: float | None
    status: Literal["COMPLETE", "GREEK_ATTRIBUTION_INCOMPLETE"]
    diagnostic_only: Literal[True] = True


class OptionStrategySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_at: datetime
    legs: tuple[OptionLegQuote, ...]
    margin_estimate: MarginEstimate | None = None
    unexplained_quote_gap: bool = False

    @field_validator("observed_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("strategy observation timestamp must be timezone-aware")
        return value.astimezone(UTC)


class OptionStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str = Field(min_length=1)
    strategy_type: StrategyType
    structure_name: str | None = Field(default=None, min_length=1)
    legs: tuple[OptionLeg, ...]
    costs: TransactionCosts = Field(default_factory=TransactionCosts)

    @model_validator(mode="after")
    def _valid_short_put(self) -> OptionStrategy:
        if len({leg.leg_id for leg in self.legs}) != len(self.legs):
            raise ValueError("option strategy leg IDs must be unique")
        if self.strategy_type is StrategyType.SHORT_PUT and not (
            len(self.legs) == 1
            and self.legs[0].right == "P"
            and self.legs[0].signed_contract_quantity < 0
        ):
            raise ValueError("short put requires one negative-quantity put leg")
        if self.strategy_type is StrategyType.LONG_OPTION and not (
            self.legs and all(leg.signed_contract_quantity > 0 for leg in self.legs)
        ):
            raise ValueError("long option structure requires only positive-quantity legs")
        if self.strategy_type is StrategyType.BULL_PUT_SPREAD:
            short_legs = tuple(leg for leg in self.legs if leg.signed_contract_quantity < 0)
            long_legs = tuple(leg for leg in self.legs if leg.signed_contract_quantity > 0)
            if not (
                len(self.legs) == 2
                and len(short_legs) == 1
                and len(long_legs) == 1
                and all(leg.right == "P" for leg in self.legs)
                and short_legs[0].strike > long_legs[0].strike
                and abs(short_legs[0].signed_contract_quantity)
                == long_legs[0].signed_contract_quantity
            ):
                raise ValueError(
                    "bull put spread requires equal short-higher and long-lower put legs"
                )
        if self.strategy_type is StrategyType.DEFINED_RISK_OPTION and not (
            len(self.legs) >= 2
            and any(leg.signed_contract_quantity < 0 for leg in self.legs)
            and any(leg.signed_contract_quantity > 0 for leg in self.legs)
        ):
            raise ValueError("defined-risk option structure requires short and long legs")
        return self


class UnderlyingStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    costs: TransactionCosts = Field(default_factory=TransactionCosts)


class UnderlyingQuoteSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed_at: datetime
    quote_timestamp: datetime
    bid: float | None
    ask: float | None
    last: float | None
    market_data_status: str = Field(min_length=1)

    @field_validator("observed_at", "quote_timestamp")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("underlying quote timestamp must be timezone-aware")
        return value.astimezone(UTC)


class UnderlyingRiskAccountingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str
    strategy_type: Literal["UNDERLYING_LONG"] = "UNDERLYING_LONG"
    observed_at: datetime
    quantity: int
    entry_underlying_notional: float
    gross_underlying_pnl: float | None
    net_underlying_pnl: float | None
    underlying_roi: float | None
    commissions: float
    regulatory_fees: float
    exchange_fees: float
    midpoint_pnl: float | None
    bid_ask_cost: float | None
    theoretical_maximum_loss: float
    account_independent_notional_exposure: float
    delta_equivalent_underlying_exposure: float
    capital_hours_employed: float
    return_per_1000_reserved_capital: float | None
    quote_timestamp: datetime
    quote_age_seconds: float
    market_data_status: str
    net_delta_exposure: float
    net_gamma_exposure: float = Field(default=0.0, ge=0.0, le=0.0)
    net_vega_exposure: float = Field(default=0.0, ge=0.0, le=0.0)
    estimated_theta_contribution: float = Field(default=0.0, ge=0.0, le=0.0)
    accounting_payload_version: Literal["underlying_risk_accounting_v1"] = (
        "underlying_risk_accounting_v1"
    )
    market_data_source: Literal["ibkr"] = "ibkr"
    research_only: Literal[True] = True
    can_authorize_trade: Literal[False] = False
    order_routing: Literal["disabled"] = "disabled"


class StrategyComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str
    strategy_type: Literal["UNDERLYING_LONG", "SHORT_PUT", "BULL_PUT_SPREAD"]
    net_monetary_pnl: float | None
    primary_capital_basis: str
    primary_capital_amount: float
    primary_roi: float | None
    underlying_roi: float | None
    premium_roi: float | None
    cash_secured_roi: float | None
    entry_margin_roi: float | None
    peak_margin_roi: float | None
    full_risk_roi: float | None
    defined_risk_roi: float | None
    maximum_theoretical_loss: float | None
    maximum_observed_margin: MarginValue
    estimated_theta_contribution: float | None
    theta_attribution_status: str
    position_greek_source: Literal["underlying", "model"]
    net_delta_exposure: float | None
    net_gamma_exposure: float | None
    net_vega_exposure: float | None
    bid_ask_cost: float | None
    greek_residual: float | None
    maximum_drawdown: float
    expected_shortfall: float | None
    expected_shortfall_confidence: float = Field(default=0.95, ge=0.95, le=0.95)
    market_data_source: Literal["ibkr"] = "ibkr"
    research_only: Literal[True] = True
    can_authorize_trade: Literal[False] = False


class StrategyComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategies: tuple[StrategyComparisonRow, StrategyComparisonRow, StrategyComparisonRow]
    accounting_payload_version: Literal["strategy_comparison_v1"] = "strategy_comparison_v1"
    executable_pnl_is_primary: Literal[True] = True
    greek_attribution_is_diagnostic_only: Literal[True] = True
    market_data_source: Literal["ibkr"] = "ibkr"
    research_only: Literal[True] = True
    can_authorize_trade: Literal[False] = False
    order_routing: Literal["disabled"] = "disabled"


class OptionRiskAccountingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str
    strategy_type: StrategyType
    structure_name: str
    legs: tuple[OptionLeg, ...]
    observed_at: datetime
    option_multiplier: int
    gross_entry_credit: float
    gross_entry_debit: float
    commissions: float
    regulatory_fees: float
    exchange_fees: float
    cash_secured_capital: float | None
    total_premium_paid: float | None
    theoretical_maximum_loss: float | None
    short_put_max_loss: float | None
    spread_max_loss: float | None
    defined_risk_capital: float | None
    entry_initial_margin: MarginValue
    entry_maintenance_margin: MarginValue
    ibkr_initial_margin: MarginValue
    ibkr_maintenance_margin: MarginValue
    maximum_observed_initial_margin: MarginValue
    maximum_observed_maintenance_margin: MarginValue
    primary_capital_basis: PrimaryCapitalBasis
    primary_capital_amount: float = Field(ge=0.0)
    primary_roi: float | None
    gross_executable_pnl: float | None
    net_option_pnl: float | None
    executable_entry_prices_by_leg: dict[str, float]
    executable_exit_prices_by_leg: dict[str, float | None]
    executable_pnl_is_primary: Literal[True] = True
    midpoint_pnl: float | None
    model_price_pnl: float | None
    bid_ask_cost: float | None
    premium_roi: float | None
    cash_secured_roi: float | None
    full_risk_roi: float | None
    defined_risk_roi: float | None
    entry_margin_roi: float | None
    peak_margin_roi: float | None
    gross_premium_yield: float | None
    net_premium_yield: float | None
    capital_hours_employed: float | None
    return_per_1000_reserved_capital: float | None
    account_independent_notional_exposure: float
    delta_equivalent_underlying_exposure: float | None
    quote_timestamps_by_leg: dict[str, datetime]
    quote_age_seconds_by_leg: dict[str, float]
    market_data_status_by_leg: dict[str, str]
    greek_observations_by_leg: dict[str, dict[GreekSource, GreekSourceSnapshot]]
    position_greeks_by_source: dict[GreekSource, PositionGreeks]
    net_delta_exposure: float | None
    net_gamma_exposure: float | None
    net_theta_exposure: float | None
    net_vega_exposure: float | None
    position_greek_source: Literal["model"] = "model"
    theta_interval_contribution: float | None
    estimated_theta_contribution: float | None
    theta_attribution_status: Literal["COMPLETE", "THETA_ATTRIBUTION_INCOMPLETE"]
    theta_attribution_label: Literal["estimated_theta_contribution"] = (
        "estimated_theta_contribution"
    )
    greek_attribution_interval: GreekAttribution | None
    greek_residual: float | None
    greek_attribution_diagnostic_only: Literal[True] = True
    greek_attribution_performed: bool
    maximum_adverse_excursion: float = Field(ge=0.0)
    maximum_drawdown: float = Field(ge=0.0)
    unexplained_quote_gap: bool
    quote_quality_flags: tuple[str, ...]
    accounting_payload_version: Literal["option_risk_accounting_v1"] = "option_risk_accounting_v1"
    market_data_source: Literal["ibkr"] = "ibkr"
    research_only: Literal[True] = True
    can_authorize_trade: Literal[False] = False
    order_routing: Literal["disabled"] = "disabled"


def _finite_price(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value < 0.0:
        return None
    return value


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _source_number(
    raw: dict[str, float | str | None],
    key: str,
) -> float | None:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _source_timestamp(raw: dict[str, float | str | None]) -> datetime | None:
    value = raw.get("greek_timestamp_utc")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def option_leg_quote_from_event(
    event: OptionQuoteEvent,
    *,
    leg_id: str,
) -> OptionLegQuote:
    """Translate one immutable live event without collapsing Greek sources."""

    sources: dict[GreekSource, GreekSourceSnapshot] = {}
    for source in ("bid", "ask", "last", "model"):
        raw = event.option_computation_by_source.get(source)
        if raw is None:
            continue
        market_data_status = raw.get("market_data_status")
        sources[source] = GreekSourceSnapshot(
            implied_volatility=_source_number(raw, "implied_volatility"),
            delta=_source_number(raw, "delta"),
            gamma=_source_number(raw, "gamma"),
            theta=_source_number(raw, "theta"),
            vega=_source_number(raw, "vega"),
            option_model_price=_source_number(raw, "option_price"),
            underlying_model_reference_price=_source_number(
                raw,
                "underlying_reference_price",
            ),
            greek_timestamp=_source_timestamp(raw),
            market_data_status=(
                market_data_status
                if isinstance(market_data_status, str) and market_data_status
                else event.market_data_type.value
            ),
        )
    return OptionLegQuote(
        leg_id=leg_id,
        quote_timestamp=event.ordering_timestamp,
        bid=event.bid,
        ask=event.ask,
        last=event.last,
        market_data_status=event.market_data_type.value,
        greeks_by_source=sources,
    )


def _quote_map(
    strategy: OptionStrategy,
    snapshot: OptionStrategySnapshot,
) -> dict[str, OptionLegQuote]:
    quotes = {quote.leg_id: quote for quote in snapshot.legs}
    if len(quotes) != len(snapshot.legs):
        raise ValueError("strategy snapshot leg IDs must be unique")
    if set(quotes) != {leg.leg_id for leg in strategy.legs}:
        raise ValueError("strategy snapshot must contain every configured leg exactly once")
    return quotes


def _entry_cash_flow(
    strategy: OptionStrategy,
    quotes: dict[str, OptionLegQuote],
) -> float | None:
    cash_flow = 0.0
    for leg in strategy.legs:
        quote = quotes[leg.leg_id]
        price = quote.ask if leg.signed_contract_quantity > 0 else quote.bid
        price = _finite_price(price)
        if price is None:
            return None
        cash_flow -= price * leg.multiplier * leg.signed_contract_quantity
    return cash_flow


def _liquidation_value(
    strategy: OptionStrategy,
    quotes: dict[str, OptionLegQuote],
) -> float | None:
    value = 0.0
    for leg in strategy.legs:
        quote = quotes[leg.leg_id]
        price = quote.bid if leg.signed_contract_quantity > 0 else quote.ask
        price = _finite_price(price)
        if price is None:
            return None
        value += price * leg.multiplier * leg.signed_contract_quantity
    return value


def _executable_prices_by_leg(
    strategy: OptionStrategy,
    quotes: dict[str, OptionLegQuote],
    *,
    entry: bool,
) -> dict[str, float | None]:
    prices: dict[str, float | None] = {}
    for leg in strategy.legs:
        quote = quotes[leg.leg_id]
        if entry:
            raw_price = quote.ask if leg.signed_contract_quantity > 0 else quote.bid
        else:
            raw_price = quote.bid if leg.signed_contract_quantity > 0 else quote.ask
        prices[leg.leg_id] = _finite_price(raw_price)
    return prices


def _midpoint_change(
    strategy: OptionStrategy,
    entry_quotes: dict[str, OptionLegQuote],
    current_quotes: dict[str, OptionLegQuote],
) -> float | None:
    pnl = 0.0
    for leg in strategy.legs:
        entry = entry_quotes[leg.leg_id]
        current = current_quotes[leg.leg_id]
        prices = tuple(
            _finite_price(value) for value in (entry.bid, entry.ask, current.bid, current.ask)
        )
        if any(value is None for value in prices):
            return None
        entry_bid, entry_ask, current_bid, current_ask = prices
        assert (
            entry_bid is not None
            and entry_ask is not None
            and current_bid is not None
            and current_ask is not None
        )
        entry_midpoint = (entry_bid + entry_ask) / 2.0
        current_midpoint = (current_bid + current_ask) / 2.0
        pnl += (current_midpoint - entry_midpoint) * leg.multiplier * leg.signed_contract_quantity
    return pnl


def _reliable_margin(
    estimate: MarginEstimate | None,
) -> tuple[float, float] | None:
    if (
        estimate is None
        or not estimate.reliable
        or estimate.initial_margin is None
        or estimate.maintenance_margin is None
    ):
        return None
    return estimate.initial_margin, estimate.maintenance_margin


def _expiry_maximum_loss(
    strategy: OptionStrategy,
    *,
    entry_cash_flow: float,
) -> float | None:
    """Return bounded expiry loss from the exact signed multi-leg payoff."""

    high_price_slope = sum(
        leg.multiplier * leg.signed_contract_quantity for leg in strategy.legs if leg.right == "C"
    )
    if high_price_slope < 0:
        return None
    candidate_underlying_prices = (0.0, *(leg.strike for leg in strategy.legs))
    expiry_pnls: list[float] = []
    for underlying_price in candidate_underlying_prices:
        intrinsic_value = 0.0
        for leg in strategy.legs:
            intrinsic_per_share = (
                max(underlying_price - leg.strike, 0.0)
                if leg.right == "C"
                else max(leg.strike - underlying_price, 0.0)
            )
            intrinsic_value += intrinsic_per_share * leg.multiplier * leg.signed_contract_quantity
        expiry_pnls.append(entry_cash_flow + intrinsic_value - strategy.costs.total)
    return max(0.0, -min(expiry_pnls))


def _position_greeks(
    strategy: OptionStrategy,
    quotes: dict[str, OptionLegQuote],
) -> dict[GreekSource, PositionGreeks]:
    sources = {source for quote in quotes.values() for source in quote.greeks_by_source}
    result: dict[GreekSource, PositionGreeks] = {}
    for source in sorted(sources):
        totals: dict[str, float | None] = {}
        for name in ("delta", "gamma", "theta", "vega"):
            values: list[float] = []
            complete = True
            for leg in strategy.legs:
                source_snapshot = quotes[leg.leg_id].greeks_by_source.get(source)
                contract_value = None if source_snapshot is None else getattr(source_snapshot, name)
                if contract_value is None:
                    complete = False
                    break
                values.append(contract_value * leg.multiplier * leg.signed_contract_quantity)
            totals[name] = sum(values) if complete else None
        result[source] = PositionGreeks.model_validate(totals)
    return result


def _delta_equivalent_underlying_exposure(
    strategy: OptionStrategy,
    quotes: dict[str, OptionLegQuote],
) -> float | None:
    exposure = 0.0
    for leg in strategy.legs:
        model = quotes[leg.leg_id].greeks_by_source.get("model")
        delta = None if model is None else model.delta
        reference = None if model is None else model.underlying_model_reference_price
        if (
            delta is None
            or reference is None
            or not math.isfinite(delta)
            or not math.isfinite(reference)
            or reference <= 0.0
        ):
            return None
        exposure += delta * leg.multiplier * leg.signed_contract_quantity * reference
    return exposure


def _model_theta_timestamp_complete(
    strategy: OptionStrategy,
    snapshot: OptionStrategySnapshot,
    quotes: dict[str, OptionLegQuote],
    maximum_attribution_gap: timedelta,
) -> bool:
    for leg in strategy.legs:
        model = quotes[leg.leg_id].greeks_by_source.get("model")
        if model is None or model.theta is None or model.greek_timestamp is None:
            return False
        greek_age = snapshot.observed_at - model.greek_timestamp
        if greek_age < timedelta(0) or greek_age > maximum_attribution_gap:
            return False
    return True


def _model_price_change(
    strategy: OptionStrategy,
    start_quotes: dict[str, OptionLegQuote],
    end_quotes: dict[str, OptionLegQuote],
) -> float | None:
    change = 0.0
    for leg in strategy.legs:
        start = start_quotes[leg.leg_id].greeks_by_source.get("model")
        end = end_quotes[leg.leg_id].greeks_by_source.get("model")
        start_price = None if start is None else start.option_model_price
        end_price = None if end is None else end.option_model_price
        if (
            start_price is None
            or end_price is None
            or not math.isfinite(start_price)
            or not math.isfinite(end_price)
        ):
            return None
        change += (end_price - start_price) * leg.multiplier * leg.signed_contract_quantity
    return change


def _greek_attribution(
    *,
    strategy: OptionStrategy,
    previous_snapshot: OptionStrategySnapshot,
    current_snapshot: OptionStrategySnapshot,
    previous_quotes: dict[str, OptionLegQuote],
    current_quotes: dict[str, OptionLegQuote],
    theta_contribution: float | None,
    maximum_attribution_gap: timedelta,
) -> GreekAttribution:
    elapsed = current_snapshot.observed_at - previous_snapshot.observed_at
    valid_interval = (
        elapsed <= maximum_attribution_gap
        and not current_snapshot.unexplained_quote_gap
        and theta_contribution is not None
    )
    delta_contribution = 0.0
    gamma_contribution = 0.0
    vega_contribution = 0.0
    if valid_interval:
        for leg in strategy.legs:
            previous = previous_quotes[leg.leg_id].greeks_by_source.get("model")
            current = current_quotes[leg.leg_id].greeks_by_source.get("model")
            required = (
                None if previous is None else previous.delta,
                None if previous is None else previous.gamma,
                None if previous is None else previous.vega,
                None if previous is None else previous.implied_volatility,
                None if current is None else current.implied_volatility,
                None if previous is None else previous.underlying_model_reference_price,
                None if current is None else current.underlying_model_reference_price,
            )
            if any(value is None or not math.isfinite(value) for value in required):
                valid_interval = False
                break
            (
                previous_delta,
                previous_gamma,
                previous_vega,
                previous_iv,
                current_iv,
                previous_underlying,
                current_underlying,
            ) = required
            assert (
                previous_delta is not None
                and previous_gamma is not None
                and previous_vega is not None
                and previous_iv is not None
                and current_iv is not None
                and previous_underlying is not None
                and current_underlying is not None
            )
            underlying_change = current_underlying - previous_underlying
            scale = leg.multiplier * leg.signed_contract_quantity
            delta_contribution += previous_delta * scale * underlying_change
            gamma_contribution += 0.5 * previous_gamma * scale * underlying_change**2
            # IBKR vega is the option-price change for one volatility percentage point.
            vega_contribution += previous_vega * scale * (current_iv - previous_iv) * 100.0
    model_change = _model_price_change(strategy, previous_quotes, current_quotes)
    midpoint_change = _midpoint_change(strategy, previous_quotes, current_quotes)
    marked_change = model_change if model_change is not None else midpoint_change
    basis: Literal["model", "midpoint"] | None = (
        "model" if model_change is not None else "midpoint" if midpoint_change is not None else None
    )
    if not valid_interval or marked_change is None or theta_contribution is None:
        return GreekAttribution(
            delta_contribution=None,
            gamma_contribution=None,
            theta_contribution=None,
            vega_contribution=None,
            total_estimated_contribution=None,
            model_or_midpoint_change=marked_change,
            change_basis=basis,
            greek_residual=None,
            status="GREEK_ATTRIBUTION_INCOMPLETE",
        )
    total = delta_contribution + gamma_contribution + theta_contribution + vega_contribution
    return GreekAttribution(
        delta_contribution=delta_contribution,
        gamma_contribution=gamma_contribution,
        theta_contribution=theta_contribution,
        vega_contribution=vega_contribution,
        total_estimated_contribution=total,
        model_or_midpoint_change=marked_change,
        change_basis=basis,
        greek_residual=marked_change - total,
        status="COMPLETE",
    )


def calculate_option_strategy_path(
    *,
    strategy: OptionStrategy,
    snapshots: tuple[OptionStrategySnapshot, ...],
    maximum_attribution_gap: timedelta,
    include_greek_attribution: bool = False,
) -> tuple[OptionRiskAccountingRecord, ...]:
    """Calculate executable marks without consulting an account or execution surface.

    Raw source-separated Greeks are always retained.  Taylor-path attribution
    is optional because it is diagnostic and is not required to finalize the
    executable P&L record.
    """

    if not snapshots:
        raise ValueError("option strategy path requires at least one observation")
    if maximum_attribution_gap <= timedelta(0):
        raise ValueError("maximum attribution gap must be positive")
    ordered = tuple(sorted(snapshots, key=lambda item: item.observed_at))
    if ordered != snapshots or len({item.observed_at for item in ordered}) != len(ordered):
        raise ValueError("strategy observations must be uniquely ordered")
    multipliers = {leg.multiplier for leg in strategy.legs}
    if len(multipliers) != 1:
        raise ValueError("one option strategy must use one contract multiplier")
    multiplier = next(iter(multipliers))
    entry_quotes = _quote_map(strategy, ordered[0])
    entry_cash_flow = _entry_cash_flow(strategy, entry_quotes)
    if entry_cash_flow is None:
        raise ValueError("strategy path requires every executable entry quote")
    entry_price_values = _executable_prices_by_leg(strategy, entry_quotes, entry=True)
    executable_entry_prices = {
        leg_id: price for leg_id, price in entry_price_values.items() if price is not None
    }
    gross_credit = max(entry_cash_flow, 0.0)
    gross_debit = max(-entry_cash_flow, 0.0)
    costs = strategy.costs.total
    entry_margin = _reliable_margin(ordered[0].margin_estimate)
    entry_initial_margin: MarginValue = (
        MARGIN_UNAVAILABLE if entry_margin is None else entry_margin[0]
    )
    entry_maintenance_margin: MarginValue = (
        MARGIN_UNAVAILABLE if entry_margin is None else entry_margin[1]
    )
    leg = strategy.legs[0]
    contracts = abs(leg.signed_contract_quantity)
    if strategy.strategy_type is StrategyType.SHORT_PUT:
        cash_secured_capital = leg.strike * leg.multiplier * contracts - gross_credit
        total_premium_paid = None
        maximum_loss = cash_secured_capital + costs
        spread_max_loss = None
        defined_risk_capital = None
        reserved_capital = cash_secured_capital
        primary_capital_basis: PrimaryCapitalBasis = "cash_secured_capital"
    elif strategy.strategy_type is StrategyType.BULL_PUT_SPREAD:
        short_leg = next(item for item in strategy.legs if item.signed_contract_quantity < 0)
        long_leg = next(item for item in strategy.legs if item.signed_contract_quantity > 0)
        contracts = abs(short_leg.signed_contract_quantity)
        cash_secured_capital = None
        total_premium_paid = None
        spread_max_loss = max(
            0.0,
            (
                (short_leg.strike - long_leg.strike) * short_leg.multiplier * contracts
                - gross_credit
                + costs
            ),
        )
        maximum_loss = spread_max_loss
        defined_risk_capital = spread_max_loss
        reserved_capital = defined_risk_capital
        primary_capital_basis = "maximum_defined_risk"
    elif strategy.strategy_type is StrategyType.DEFINED_RISK_OPTION:
        cash_secured_capital = None
        total_premium_paid = None
        generic_maximum_loss = _expiry_maximum_loss(
            strategy,
            entry_cash_flow=entry_cash_flow,
        )
        if generic_maximum_loss is None:
            raise ValueError("defined-risk option structure has unbounded expiry loss")
        maximum_loss = generic_maximum_loss
        spread_max_loss = None
        defined_risk_capital = maximum_loss
        reserved_capital = defined_risk_capital
        primary_capital_basis = "maximum_defined_risk"
    else:
        cash_secured_capital = None
        total_premium_paid = gross_debit
        maximum_loss = total_premium_paid + costs
        spread_max_loss = None
        defined_risk_capital = None
        reserved_capital = total_premium_paid
        primary_capital_basis = "premium_paid"
    assert reserved_capital is not None and reserved_capital >= 0.0
    primary_capital_amount = reserved_capital
    records: list[OptionRiskAccountingRecord] = []
    observed_net_pnls: list[float] = []
    observed_initial_margins: list[float] = []
    observed_maintenance_margins: list[float] = []
    theta_complete = include_greek_attribution
    cumulative_theta = 0.0
    previous_snapshot: OptionStrategySnapshot | None = None
    previous_position_greeks: dict[GreekSource, PositionGreeks] | None = None
    cumulative_greek_contributions = 0.0
    cumulative_mark_change = 0.0
    greek_attribution_complete = include_greek_attribution
    for index, snapshot in enumerate(ordered):
        quotes = _quote_map(strategy, snapshot)
        liquidation = _liquidation_value(strategy, quotes)
        gross_pnl = (
            None
            if entry_cash_flow is None or liquidation is None
            else entry_cash_flow + liquidation
        )
        net_pnl = None if gross_pnl is None else gross_pnl - costs
        midpoint_pnl = _midpoint_change(strategy, entry_quotes, quotes)
        elapsed_hours = (snapshot.observed_at - ordered[0].observed_at).total_seconds() / 3600.0
        current_margin = _reliable_margin(snapshot.margin_estimate)
        if current_margin is not None:
            observed_initial_margins.append(current_margin[0])
            observed_maintenance_margins.append(current_margin[1])
        current_initial_margin: MarginValue = (
            MARGIN_UNAVAILABLE if current_margin is None else current_margin[0]
        )
        current_maintenance_margin: MarginValue = (
            MARGIN_UNAVAILABLE if current_margin is None else current_margin[1]
        )
        maximum_initial_margin: MarginValue = (
            max(observed_initial_margins) if observed_initial_margins else MARGIN_UNAVAILABLE
        )
        maximum_maintenance_margin: MarginValue = (
            max(observed_maintenance_margins)
            if observed_maintenance_margins
            else MARGIN_UNAVAILABLE
        )
        position_greeks = _position_greeks(strategy, quotes)
        model_greeks = position_greeks.get("model")
        delta_equivalent_exposure = _delta_equivalent_underlying_exposure(
            strategy,
            quotes,
        )
        current_theta = None if model_greeks is None else model_greeks.theta
        theta_interval: float | None = None
        interval_attribution: GreekAttribution | None = None
        if include_greek_attribution and index == 0:
            if current_theta is None or not _model_theta_timestamp_complete(
                strategy,
                snapshot,
                quotes,
                maximum_attribution_gap,
            ):
                theta_complete = False
        elif include_greek_attribution:
            assert previous_snapshot is not None and previous_position_greeks is not None
            elapsed = snapshot.observed_at - previous_snapshot.observed_at
            previous_model = previous_position_greeks.get("model")
            previous_theta = None if previous_model is None else previous_model.theta
            previous_quotes = _quote_map(strategy, previous_snapshot)
            interval_complete = (
                elapsed <= maximum_attribution_gap
                and not snapshot.unexplained_quote_gap
                and previous_theta is not None
                and current_theta is not None
                and _model_theta_timestamp_complete(
                    strategy,
                    previous_snapshot,
                    previous_quotes,
                    maximum_attribution_gap,
                )
                and _model_theta_timestamp_complete(
                    strategy,
                    snapshot,
                    quotes,
                    maximum_attribution_gap,
                )
            )
            if interval_complete:
                assert previous_theta is not None and current_theta is not None
                theta_interval = (
                    0.5 * (previous_theta + current_theta) * (elapsed.total_seconds() / 86_400.0)
                )
                cumulative_theta += theta_interval
            else:
                theta_complete = False
            interval_attribution = _greek_attribution(
                strategy=strategy,
                previous_snapshot=previous_snapshot,
                current_snapshot=snapshot,
                previous_quotes=previous_quotes,
                current_quotes=quotes,
                theta_contribution=theta_interval,
                maximum_attribution_gap=maximum_attribution_gap,
            )
            if (
                interval_attribution.status == "COMPLETE"
                and interval_attribution.total_estimated_contribution is not None
                and interval_attribution.model_or_midpoint_change is not None
            ):
                cumulative_greek_contributions += interval_attribution.total_estimated_contribution
                cumulative_mark_change += interval_attribution.model_or_midpoint_change
            else:
                greek_attribution_complete = False
        model_price_pnl = (
            _model_price_change(strategy, entry_quotes, quotes)
            if include_greek_attribution
            else None
        )
        executable_exit_prices = _executable_prices_by_leg(strategy, quotes, entry=False)
        quality_flags: list[str] = []
        if snapshot.unexplained_quote_gap:
            quality_flags.append("unexplained_quote_gap")
        for leg_id, price in executable_exit_prices.items():
            quote = quotes[leg_id]
            if price is None:
                quality_flags.append(f"missing_executable_exit_quote:{leg_id}")
            if quote.market_data_status.lower() != "live":
                quality_flags.append(f"non_live_market_data:{leg_id}:{quote.market_data_status}")
            quote_age = (snapshot.observed_at - quote.quote_timestamp).total_seconds()
            if quote_age < 0.0:
                quality_flags.append(f"quote_timestamp_after_observation:{leg_id}")
        if net_pnl is not None:
            observed_net_pnls.append(net_pnl)
        maximum_adverse_excursion = max(
            0.0,
            -min((0.0, *observed_net_pnls)),
        )
        maximum_drawdown = _maximum_drawdown(tuple(observed_net_pnls))
        records.append(
            OptionRiskAccountingRecord(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy.strategy_type,
                structure_name=strategy.structure_name or strategy.strategy_type.value,
                legs=strategy.legs,
                observed_at=snapshot.observed_at,
                option_multiplier=multiplier,
                gross_entry_credit=gross_credit,
                gross_entry_debit=gross_debit,
                commissions=strategy.costs.commissions,
                regulatory_fees=strategy.costs.regulatory_fees,
                exchange_fees=strategy.costs.exchange_fees,
                cash_secured_capital=cash_secured_capital,
                total_premium_paid=total_premium_paid,
                theoretical_maximum_loss=maximum_loss,
                short_put_max_loss=(
                    maximum_loss if strategy.strategy_type is StrategyType.SHORT_PUT else None
                ),
                spread_max_loss=spread_max_loss,
                defined_risk_capital=defined_risk_capital,
                entry_initial_margin=entry_initial_margin,
                entry_maintenance_margin=entry_maintenance_margin,
                ibkr_initial_margin=current_initial_margin,
                ibkr_maintenance_margin=current_maintenance_margin,
                maximum_observed_initial_margin=maximum_initial_margin,
                maximum_observed_maintenance_margin=maximum_maintenance_margin,
                primary_capital_basis=primary_capital_basis,
                primary_capital_amount=primary_capital_amount,
                primary_roi=_safe_ratio(net_pnl, primary_capital_amount),
                gross_executable_pnl=gross_pnl,
                net_option_pnl=net_pnl,
                executable_entry_prices_by_leg=executable_entry_prices,
                executable_exit_prices_by_leg=executable_exit_prices,
                midpoint_pnl=midpoint_pnl,
                model_price_pnl=model_price_pnl,
                bid_ask_cost=(
                    None if midpoint_pnl is None or gross_pnl is None else midpoint_pnl - gross_pnl
                ),
                premium_roi=(
                    None
                    if strategy.strategy_type is not StrategyType.LONG_OPTION
                    else _safe_ratio(net_pnl, total_premium_paid)
                ),
                cash_secured_roi=_safe_ratio(net_pnl, cash_secured_capital),
                full_risk_roi=(
                    None
                    if strategy.strategy_type is not StrategyType.SHORT_PUT
                    else _safe_ratio(net_pnl, maximum_loss)
                ),
                defined_risk_roi=(
                    None
                    if strategy.strategy_type
                    not in {
                        StrategyType.BULL_PUT_SPREAD,
                        StrategyType.DEFINED_RISK_OPTION,
                    }
                    else _safe_ratio(net_pnl, defined_risk_capital)
                ),
                entry_margin_roi=(
                    None
                    if strategy.strategy_type is not StrategyType.SHORT_PUT
                    or not isinstance(entry_initial_margin, float)
                    else _safe_ratio(net_pnl, entry_initial_margin)
                ),
                peak_margin_roi=(
                    None
                    if strategy.strategy_type is not StrategyType.SHORT_PUT
                    or not isinstance(maximum_initial_margin, float)
                    else _safe_ratio(net_pnl, maximum_initial_margin)
                ),
                gross_premium_yield=(
                    None
                    if strategy.strategy_type
                    not in {
                        StrategyType.SHORT_PUT,
                        StrategyType.BULL_PUT_SPREAD,
                        StrategyType.DEFINED_RISK_OPTION,
                    }
                    else _safe_ratio(gross_credit, reserved_capital)
                ),
                net_premium_yield=(
                    None
                    if strategy.strategy_type
                    not in {
                        StrategyType.SHORT_PUT,
                        StrategyType.BULL_PUT_SPREAD,
                        StrategyType.DEFINED_RISK_OPTION,
                    }
                    else _safe_ratio(gross_credit - costs, reserved_capital)
                ),
                capital_hours_employed=reserved_capital * elapsed_hours,
                return_per_1000_reserved_capital=(
                    None
                    if (reserved_return := _safe_ratio(net_pnl, reserved_capital)) is None
                    else reserved_return * 1_000.0
                ),
                account_independent_notional_exposure=(
                    sum(
                        item.strike * item.multiplier * abs(item.signed_contract_quantity)
                        for item in strategy.legs
                    )
                ),
                delta_equivalent_underlying_exposure=(delta_equivalent_exposure),
                quote_timestamps_by_leg={
                    leg_id: quote.quote_timestamp for leg_id, quote in quotes.items()
                },
                quote_age_seconds_by_leg={
                    leg_id: abs((snapshot.observed_at - quote.quote_timestamp).total_seconds())
                    for leg_id, quote in quotes.items()
                },
                market_data_status_by_leg={
                    leg_id: quote.market_data_status for leg_id, quote in quotes.items()
                },
                greek_observations_by_leg={
                    leg_id: quote.greeks_by_source for leg_id, quote in quotes.items()
                },
                position_greeks_by_source=position_greeks,
                net_delta_exposure=(None if model_greeks is None else model_greeks.delta),
                net_gamma_exposure=(None if model_greeks is None else model_greeks.gamma),
                net_theta_exposure=(None if model_greeks is None else model_greeks.theta),
                net_vega_exposure=(None if model_greeks is None else model_greeks.vega),
                theta_interval_contribution=theta_interval,
                estimated_theta_contribution=(cumulative_theta if theta_complete else None),
                theta_attribution_status=(
                    "COMPLETE" if theta_complete else THETA_ATTRIBUTION_INCOMPLETE
                ),
                greek_attribution_interval=interval_attribution,
                greek_residual=(
                    cumulative_mark_change - cumulative_greek_contributions
                    if index > 0 and greek_attribution_complete
                    else None
                ),
                greek_attribution_performed=include_greek_attribution,
                maximum_adverse_excursion=maximum_adverse_excursion,
                maximum_drawdown=maximum_drawdown,
                unexplained_quote_gap=snapshot.unexplained_quote_gap,
                quote_quality_flags=tuple(quality_flags),
            )
        )
        previous_snapshot = snapshot
        previous_position_greeks = position_greeks
    return tuple(records)


def calculate_underlying_strategy_path(
    *,
    strategy: UnderlyingStrategy,
    snapshots: tuple[UnderlyingQuoteSnapshot, ...],
) -> tuple[UnderlyingRiskAccountingRecord, ...]:
    """Mark a long underlying comparator at ask entry and bid liquidation."""

    if not snapshots:
        raise ValueError("underlying strategy path requires at least one observation")
    ordered = tuple(sorted(snapshots, key=lambda item: item.observed_at))
    if ordered != snapshots or len({item.observed_at for item in ordered}) != len(ordered):
        raise ValueError("underlying observations must be uniquely ordered")
    entry_ask = _finite_price(ordered[0].ask)
    entry_bid = _finite_price(ordered[0].bid)
    if entry_ask is None or entry_ask <= 0.0:
        raise ValueError("underlying long requires an executable entry ask")
    entry_notional = entry_ask * strategy.quantity
    entry_midpoint = None if entry_bid is None else (entry_bid + entry_ask) / 2.0
    records: list[UnderlyingRiskAccountingRecord] = []
    for snapshot in ordered:
        current_bid = _finite_price(snapshot.bid)
        current_ask = _finite_price(snapshot.ask)
        gross_pnl = None if current_bid is None else (current_bid - entry_ask) * strategy.quantity
        net_pnl = None if gross_pnl is None else gross_pnl - strategy.costs.total
        current_midpoint = (
            None
            if current_bid is None or current_ask is None
            else (current_bid + current_ask) / 2.0
        )
        midpoint_pnl = (
            None
            if entry_midpoint is None or current_midpoint is None
            else (current_midpoint - entry_midpoint) * strategy.quantity
        )
        elapsed_hours = (snapshot.observed_at - ordered[0].observed_at).total_seconds() / 3_600.0
        records.append(
            UnderlyingRiskAccountingRecord(
                strategy_id=strategy.strategy_id,
                observed_at=snapshot.observed_at,
                quantity=strategy.quantity,
                entry_underlying_notional=entry_notional,
                gross_underlying_pnl=gross_pnl,
                net_underlying_pnl=net_pnl,
                underlying_roi=_safe_ratio(net_pnl, entry_notional),
                commissions=strategy.costs.commissions,
                regulatory_fees=strategy.costs.regulatory_fees,
                exchange_fees=strategy.costs.exchange_fees,
                midpoint_pnl=midpoint_pnl,
                bid_ask_cost=(
                    None if midpoint_pnl is None or gross_pnl is None else midpoint_pnl - gross_pnl
                ),
                theoretical_maximum_loss=entry_notional + strategy.costs.total,
                account_independent_notional_exposure=entry_notional,
                delta_equivalent_underlying_exposure=(
                    snapshot.bid if snapshot.bid is not None else entry_ask
                )
                * strategy.quantity,
                capital_hours_employed=entry_notional * elapsed_hours,
                return_per_1000_reserved_capital=(
                    None
                    if (underlying_return := _safe_ratio(net_pnl, entry_notional)) is None
                    else underlying_return * 1_000.0
                ),
                quote_timestamp=snapshot.quote_timestamp,
                quote_age_seconds=abs(
                    (snapshot.observed_at - snapshot.quote_timestamp).total_seconds()
                ),
                market_data_status=snapshot.market_data_status,
                net_delta_exposure=float(strategy.quantity),
            )
        )
    return tuple(records)


def _maximum_drawdown(values: tuple[float, ...]) -> float:
    peak = 0.0
    maximum = 0.0
    for value in values:
        peak = max(peak, value)
        maximum = max(maximum, peak - value)
    return maximum


def _expected_shortfall(
    values: tuple[float, ...],
    *,
    confidence: float = 0.95,
) -> float | None:
    """Return lower-tail monetary P&L; negative values are losses."""

    if not values:
        return None
    tail_count = max(1, math.ceil(len(values) * (1.0 - confidence)))
    tail = tuple(sorted(values)[:tail_count])
    return sum(tail) / len(tail)


def build_strategy_comparison_report(
    *,
    underlying_long: tuple[UnderlyingRiskAccountingRecord, ...],
    short_put: tuple[OptionRiskAccountingRecord, ...],
    bull_put_spread: tuple[OptionRiskAccountingRecord, ...],
) -> StrategyComparisonReport:
    """Compare three record-only paths while retaining each capital denominator."""

    if not underlying_long or not short_put or not bull_put_spread:
        raise ValueError("comparison report requires all three strategy paths")
    if any(item.strategy_type is not StrategyType.SHORT_PUT for item in short_put):
        raise ValueError("short-put comparison path has the wrong strategy type")
    if any(item.strategy_type is not StrategyType.BULL_PUT_SPREAD for item in bull_put_spread):
        raise ValueError("bull-put comparison path has the wrong strategy type")

    underlying_last = underlying_long[-1]
    underlying_values = tuple(
        item.net_underlying_pnl for item in underlying_long if item.net_underlying_pnl is not None
    )
    underlying_row = StrategyComparisonRow(
        strategy_id=underlying_last.strategy_id,
        strategy_type="UNDERLYING_LONG",
        net_monetary_pnl=underlying_last.net_underlying_pnl,
        primary_capital_basis="underlying_notional",
        primary_capital_amount=underlying_last.entry_underlying_notional,
        primary_roi=underlying_last.underlying_roi,
        underlying_roi=underlying_last.underlying_roi,
        premium_roi=None,
        cash_secured_roi=None,
        entry_margin_roi=None,
        peak_margin_roi=None,
        full_risk_roi=None,
        defined_risk_roi=None,
        maximum_theoretical_loss=underlying_last.theoretical_maximum_loss,
        maximum_observed_margin=MARGIN_UNAVAILABLE,
        estimated_theta_contribution=0.0,
        theta_attribution_status="NOT_APPLICABLE",
        position_greek_source="underlying",
        net_delta_exposure=underlying_last.net_delta_exposure,
        net_gamma_exposure=underlying_last.net_gamma_exposure,
        net_vega_exposure=underlying_last.net_vega_exposure,
        bid_ask_cost=underlying_last.bid_ask_cost,
        greek_residual=None,
        maximum_drawdown=_maximum_drawdown(underlying_values),
        expected_shortfall=_expected_shortfall(underlying_values),
    )

    def option_row(
        path: tuple[OptionRiskAccountingRecord, ...],
        *,
        strategy_type: Literal["SHORT_PUT", "BULL_PUT_SPREAD"],
    ) -> StrategyComparisonRow:
        last = path[-1]
        values = tuple(item.net_option_pnl for item in path if item.net_option_pnl is not None)
        return StrategyComparisonRow(
            strategy_id=last.strategy_id,
            strategy_type=strategy_type,
            net_monetary_pnl=last.net_option_pnl,
            primary_capital_basis=last.primary_capital_basis,
            primary_capital_amount=last.primary_capital_amount,
            primary_roi=last.primary_roi,
            underlying_roi=None,
            premium_roi=last.premium_roi,
            cash_secured_roi=last.cash_secured_roi,
            entry_margin_roi=last.entry_margin_roi,
            peak_margin_roi=last.peak_margin_roi,
            full_risk_roi=last.full_risk_roi,
            defined_risk_roi=last.defined_risk_roi,
            maximum_theoretical_loss=last.theoretical_maximum_loss,
            maximum_observed_margin=last.maximum_observed_initial_margin,
            estimated_theta_contribution=last.estimated_theta_contribution,
            theta_attribution_status=last.theta_attribution_status,
            position_greek_source="model",
            net_delta_exposure=last.net_delta_exposure,
            net_gamma_exposure=last.net_gamma_exposure,
            net_vega_exposure=last.net_vega_exposure,
            bid_ask_cost=last.bid_ask_cost,
            greek_residual=last.greek_residual,
            maximum_drawdown=_maximum_drawdown(values),
            expected_shortfall=_expected_shortfall(values),
        )

    return StrategyComparisonReport(
        strategies=(
            underlying_row,
            option_row(
                short_put,
                strategy_type="SHORT_PUT",
            ),
            option_row(
                bull_put_spread,
                strategy_type="BULL_PUT_SPREAD",
            ),
        )
    )


__all__ = [
    "MARGIN_UNAVAILABLE",
    "THETA_ATTRIBUTION_INCOMPLETE",
    "GreekSourceSnapshot",
    "GreekAttribution",
    "MarginEstimate",
    "OptionLeg",
    "OptionLegQuote",
    "OptionRiskAccountingRecord",
    "OptionStrategy",
    "OptionStrategySnapshot",
    "PositionGreeks",
    "StrategyType",
    "StrategyComparisonReport",
    "StrategyComparisonRow",
    "TransactionCosts",
    "UnderlyingQuoteSnapshot",
    "UnderlyingRiskAccountingRecord",
    "UnderlyingStrategy",
    "calculate_option_strategy_path",
    "calculate_underlying_strategy_path",
    "build_strategy_comparison_report",
    "option_leg_quote_from_event",
]
