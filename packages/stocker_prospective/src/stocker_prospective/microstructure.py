"""Deterministic descriptive microstructure primitives; no fitted direction model."""

from __future__ import annotations

import math
import statistics
import sys
from bisect import bisect_right
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from stocker_prospective.events import (
    UnderlyingLevel1QuoteEvent,
    UnderlyingTickTradeEvent,
)

EPSILON = 1e-12
STANDARD_WINDOWS = (
    timedelta(seconds=1),
    timedelta(seconds=5),
    timedelta(seconds=15),
    timedelta(seconds=30),
    timedelta(seconds=60),
    timedelta(minutes=5),
)


class QuotePrimitives(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    midpoint: float
    spread: float
    spread_bps: float
    quote_size_imbalance: float
    microprice: float
    microprice_edge: float
    microprice_edge_bps: float
    microprice_edge_half_spread_fraction: float


class ProbableTradeSide(StrEnum):
    BUY = "probable_buyer_initiated"
    SELL = "probable_seller_initiated"
    UNKNOWN = "unknown"
    UNCLASSIFIED = "unclassified"


class TradeSideClassification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    side: ProbableTradeSide
    classification_method: str
    classification_confidence: str
    prevailing_bid: float | None
    prevailing_ask: float | None
    quote_age_ms: float | None


class PersistedTradeSideClassification(TradeSideClassification):
    """Auditable classification evidence tied to one immutable Last event."""

    trade_event_id: str
    trade_provider_timestamp_utc: datetime | None
    trade_received_timestamp_utc: datetime
    trade_received_monotonic_ns: int
    trade_source_sequence: int
    trade_price: float
    trade_size: float


class QuoteFlowSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bid_price_improvements: int
    bid_price_deteriorations: int
    ask_price_improvements: int
    ask_price_deteriorations: int
    bid_size_additions: int
    bid_size_removals: int
    ask_size_additions: int
    ask_size_removals: int
    quote_update_count: int
    quote_update_rate: float
    price_moving_updates: int
    size_only_updates: int
    locked_market_count: int
    crossed_market_count: int
    invalid_quote_count: int
    spread_tightening_count: int
    spread_widening_count: int
    time_weighted_quote_size_imbalance: float | None
    time_weighted_microprice_edge: float | None
    midpoint_change: float | None
    best_bid_change: float | None
    best_ask_change: float | None
    bid_displayed_size_removal_proxy: float
    ask_displayed_size_removal_proxy: float


class TradeFlowSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probable_buy_volume: float
    probable_sell_volume: float
    unknown_volume: float
    probable_buy_trade_count: int
    probable_sell_trade_count: int
    unknown_trade_count: int
    trade_imbalance: float | None
    unknown_volume_fraction: float
    classification_valid_fraction: float
    mean_trade_size: float | None
    median_trade_size: float | None
    trade_arrival_rate: float
    buy_arrival_rate: float
    sell_arrival_rate: float
    trade_features_valid: bool


class PriceImpactSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    side: str
    raw_signed_price_impact: float | None
    price_impact_per_share: float | None
    price_impact_per_notional: float | None
    price_impact_bps: float | None
    flow_volume: float
    event_count: int
    window_duration_seconds: float
    support_valid: bool
    native_currency_only: bool = True


class ReplenishmentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    side: str
    response_interval_seconds: int
    trigger_count: int
    size_depleted: float
    size_restored: float
    replenishment_ratio: float | None
    price_survival_time_seconds: float | None
    midpoint_response: float | None
    direct_hidden_liquidity_observed: bool = False


class DescriptiveScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    score_id: str
    label: str = "microstructure descriptive score"
    components: dict[str, float | None]
    composite: float | None
    valid_component_count: int
    fitted_weights: bool = False
    direction_threshold: float | None = None


class MicrostructureWindowSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    window_start: datetime
    window_end: datetime
    duration_seconds: float
    quote_flow: QuoteFlowSummary
    trade_flow: TradeFlowSummary
    trade_classifications: tuple[PersistedTradeSideClassification, ...]
    buy_impact: PriceImpactSummary
    sell_impact: PriceImpactSummary
    replenishment: dict[str, ReplenishmentSummary]
    scores: dict[str, DescriptiveScore]
    causal_as_of: datetime
    formulas_version: str = "microstructure-formulas-v0"


RELATION_STATES = {
    "same_direction",
    "opposite_direction",
    "microstructure_neutral",
    "insufficient_data",
}


def compare_frozen_archetypes(
    *,
    actions: Mapping[str, str],
    summary: MicrostructureWindowSummary,
) -> dict[str, str]:
    """Compare frozen outputs descriptively without changing either output."""

    def score_relation(action: str, bullish: str, bearish: str) -> str:
        if action == "ABSTAIN":
            return "microstructure_neutral"
        up = summary.scores[bullish].composite
        down = summary.scores[bearish].composite
        if up is None or down is None:
            return "insufficient_data"
        if abs(up - down) <= EPSILON:
            return "microstructure_neutral"
        micro_direction = "CALL" if up > down else "PUT"
        return "same_direction" if micro_direction == action else "opposite_direction"

    r1_action = actions.get("R1", "ABSTAIN")
    microprice = summary.scores["MC"].components.get("microprice_edge")
    trade = summary.scores["MC"].components.get("trade_imbalance")
    if r1_action == "ABSTAIN":
        r1_relation = "microstructure_neutral"
    elif microprice is None or trade is None:
        r1_relation = "insufficient_data"
    elif microprice > 0.0 and trade > 0.0:
        r1_relation = "same_direction" if r1_action == "CALL" else "opposite_direction"
    elif microprice < 0.0 and trade < 0.0:
        r1_relation = "same_direction" if r1_action == "PUT" else "opposite_direction"
    else:
        r1_relation = "microstructure_neutral"

    directional = [
        action
        for action in (actions.get("A1"), actions.get("C1"), actions.get("R1"))
        if action in {"CALL", "PUT"}
    ]
    if len(directional) < 2:
        agreement = "insufficient_data"
    elif len(set(directional)) == 1:
        agreement = "same_direction"
    else:
        agreement = "opposite_direction"
    return {
        "A1_absorption": score_relation(actions.get("A1", "ABSTAIN"), "MA", "MB"),
        "C1_continuation": score_relation(actions.get("C1", "ABSTAIN"), "MC", "MD"),
        "R1_quote_trade": r1_relation,
        "archetype_agreement": agreement,
    }


def _valid_quote(quote: UnderlyingLevel1QuoteEvent) -> bool:
    return bool(
        quote.quote_valid
        and quote.bid is not None
        and quote.ask is not None
        and quote.bid_size is not None
        and quote.ask_size is not None
        and quote.bid > 0.0
        and quote.ask > 0.0
        and quote.ask >= quote.bid
        and quote.bid_size >= 0.0
        and quote.ask_size >= 0.0
    )


def quote_primitives(quote: UnderlyingLevel1QuoteEvent) -> QuotePrimitives:
    """Calculate the frozen top-of-book primitives without clipping raw inputs."""

    if not _valid_quote(quote):
        raise ValueError("quote is not valid for primitive calculation")
    assert quote.bid is not None
    assert quote.ask is not None
    assert quote.bid_size is not None
    assert quote.ask_size is not None
    midpoint = (quote.bid + quote.ask) / 2.0
    spread = quote.ask - quote.bid
    size_total = quote.bid_size + quote.ask_size
    imbalance = (quote.bid_size - quote.ask_size) / (size_total + EPSILON)
    microprice = (quote.ask * quote.bid_size + quote.bid * quote.ask_size) / (size_total + EPSILON)
    edge = microprice - midpoint
    half_spread = spread / 2.0
    half_fraction = 0.0 if half_spread <= EPSILON else edge / half_spread
    return QuotePrimitives(
        midpoint=midpoint,
        spread=spread,
        spread_bps=spread / midpoint * 10_000.0,
        quote_size_imbalance=min(max(imbalance, -1.0), 1.0),
        microprice=microprice,
        microprice_edge=edge,
        microprice_edge_bps=edge / midpoint * 10_000.0,
        microprice_edge_half_spread_fraction=min(max(half_fraction, -1.0), 1.0),
    )


def classify_probable_trade_side(
    trade: UnderlyingTickTradeEvent,
    prevailing_quote: UnderlyingLevel1QuoteEvent | None,
    *,
    maximum_quote_age: timedelta,
) -> TradeSideClassification:
    """Approximate side from the latest quote; never claim true aggressor side."""

    if prevailing_quote is None or not _valid_quote(prevailing_quote):
        return TradeSideClassification(
            side=ProbableTradeSide.UNCLASSIFIED,
            classification_method="prevailing_quote_v1",
            classification_confidence="none",
            prevailing_bid=None,
            prevailing_ask=None,
            quote_age_ms=None,
        )
    quote_timestamp = (
        prevailing_quote.provider_timestamp_utc or prevailing_quote.received_timestamp_utc
    )
    trade_timestamp = trade.provider_timestamp_utc or trade.received_timestamp_utc
    age = trade_timestamp - quote_timestamp
    age_ms = age.total_seconds() * 1000.0
    assert prevailing_quote.bid is not None
    assert prevailing_quote.ask is not None
    if age < timedelta(0) or age > maximum_quote_age:
        side = ProbableTradeSide.UNCLASSIFIED
        confidence = "none"
    elif prevailing_quote.ask <= prevailing_quote.bid:
        side = ProbableTradeSide.UNKNOWN
        confidence = "low"
    elif trade.price >= prevailing_quote.ask:
        side = ProbableTradeSide.BUY
        confidence = "probable"
    elif trade.price <= prevailing_quote.bid:
        side = ProbableTradeSide.SELL
        confidence = "probable"
    else:
        side = ProbableTradeSide.UNKNOWN
        confidence = "inside_spread"
    return TradeSideClassification(
        side=side,
        classification_method="prevailing_quote_v1",
        classification_confidence=confidence,
        prevailing_bid=prevailing_quote.bid,
        prevailing_ask=prevailing_quote.ask,
        quote_age_ms=max(0.0, age_ms),
    )


def _ordered_quotes(
    quotes: tuple[UnderlyingLevel1QuoteEvent, ...],
) -> list[UnderlyingLevel1QuoteEvent]:
    return sorted(
        quotes,
        key=lambda item: (
            item.ordering_timestamp,
            item.received_monotonic_ns,
            item.source_sequence,
            item.event_id,
        ),
    )


def _event_order_key(
    event: UnderlyingLevel1QuoteEvent | UnderlyingTickTradeEvent,
) -> tuple[datetime, int, int, str]:
    return (
        event.ordering_timestamp,
        event.received_monotonic_ns,
        event.source_sequence,
        event.event_id,
    )


def _prevailing_quote(
    ordered: list[UnderlyingLevel1QuoteEvent],
    keys: list[tuple[datetime, int, int, str]],
    *,
    timestamp: datetime | None = None,
    before_event: UnderlyingTickTradeEvent | None = None,
) -> UnderlyingLevel1QuoteEvent | None:
    if (timestamp is None) == (before_event is None):
        raise ValueError("prevailing quote requires exactly one causal boundary")
    target = (
        _event_order_key(before_event)
        if before_event is not None
        else (timestamp, sys.maxsize, sys.maxsize, "\U0010ffff")
    )
    position = bisect_right(keys, target)
    return None if position == 0 else ordered[position - 1]


def _quote_flow(
    *,
    ordered_all: list[UnderlyingLevel1QuoteEvent],
    window_start: datetime,
    window_end: datetime,
) -> QuoteFlowSummary:
    duration = max((window_end - window_start).total_seconds(), EPSILON)
    keys = [_event_order_key(item) for item in ordered_all]
    baseline = _prevailing_quote(ordered_all, keys, timestamp=window_start)
    in_window = [
        item for item in ordered_all if window_start <= item.ordering_timestamp <= window_end
    ]
    sequence = (
        [baseline] if baseline is not None and baseline not in in_window else []
    ) + in_window
    improvements_bid = deteriorations_bid = improvements_ask = deteriorations_ask = 0
    bid_add = bid_remove = ask_add = ask_remove = 0
    price_updates = size_updates = 0
    locked = crossed = invalid = tighten = widen = 0
    bid_removed_amount = ask_removed_amount = 0.0
    valid = [item for item in sequence if _valid_quote(item)]
    for current in in_window:
        if current.bid is None or current.ask is None or current.bid <= 0.0 or current.ask <= 0.0:
            invalid += 1
        elif current.bid > current.ask:
            crossed += 1
        elif current.bid == current.ask:
            locked += 1
    for previous, current in zip(sequence, sequence[1:], strict=False):
        prices_changed = previous.bid != current.bid or previous.ask != current.ask
        sizes_changed = (
            previous.bid_size != current.bid_size or previous.ask_size != current.ask_size
        )
        price_updates += int(prices_changed)
        size_updates += int(not prices_changed and sizes_changed)
        if previous.bid is not None and current.bid is not None:
            improvements_bid += int(current.bid > previous.bid)
            deteriorations_bid += int(current.bid < previous.bid)
        if previous.ask is not None and current.ask is not None:
            improvements_ask += int(current.ask < previous.ask)
            deteriorations_ask += int(current.ask > previous.ask)
        if (
            previous.bid == current.bid
            and previous.bid_size is not None
            and current.bid_size is not None
        ):
            difference = current.bid_size - previous.bid_size
            bid_add += int(difference > 0.0)
            bid_remove += int(difference < 0.0)
            bid_removed_amount += max(0.0, -difference)
        if (
            previous.ask == current.ask
            and previous.ask_size is not None
            and current.ask_size is not None
        ):
            difference = current.ask_size - previous.ask_size
            ask_add += int(difference > 0.0)
            ask_remove += int(difference < 0.0)
            ask_removed_amount += max(0.0, -difference)
        if _valid_quote(previous) and _valid_quote(current):
            previous_spread = quote_primitives(previous).spread
            current_spread = quote_primitives(current).spread
            tighten += int(current_spread < previous_spread)
            widen += int(current_spread > previous_spread)

    weighted_imbalance = 0.0
    weighted_edge = 0.0
    valid_seconds = 0.0
    state = baseline
    cursor = window_start
    for event in [*in_window, None]:
        boundary = window_end if event is None else min(event.ordering_timestamp, window_end)
        seconds = max(0.0, (boundary - cursor).total_seconds())
        if state is not None and _valid_quote(state) and seconds > 0.0:
            primitive = quote_primitives(state)
            weighted_imbalance += primitive.quote_size_imbalance * seconds
            weighted_edge += primitive.microprice_edge * seconds
            valid_seconds += seconds
        if event is not None:
            state = event
            cursor = max(cursor, event.ordering_timestamp)
    first = valid[0] if valid else None
    last = valid[-1] if valid else None

    def change(field: str) -> float | None:
        if first is None or last is None:
            return None
        first_value = (
            quote_primitives(first).midpoint if field == "midpoint" else getattr(first, field)
        )
        last_value = (
            quote_primitives(last).midpoint if field == "midpoint" else getattr(last, field)
        )
        if first_value is None or last_value is None:
            return None
        return float(last_value - first_value)

    return QuoteFlowSummary(
        bid_price_improvements=improvements_bid,
        bid_price_deteriorations=deteriorations_bid,
        ask_price_improvements=improvements_ask,
        ask_price_deteriorations=deteriorations_ask,
        bid_size_additions=bid_add,
        bid_size_removals=bid_remove,
        ask_size_additions=ask_add,
        ask_size_removals=ask_remove,
        quote_update_count=len(in_window),
        quote_update_rate=len(in_window) / duration,
        price_moving_updates=price_updates,
        size_only_updates=size_updates,
        locked_market_count=locked,
        crossed_market_count=crossed,
        invalid_quote_count=invalid,
        spread_tightening_count=tighten,
        spread_widening_count=widen,
        time_weighted_quote_size_imbalance=(
            None if valid_seconds <= 0.0 else weighted_imbalance / valid_seconds
        ),
        time_weighted_microprice_edge=(
            None if valid_seconds <= 0.0 else weighted_edge / valid_seconds
        ),
        midpoint_change=change("midpoint"),
        best_bid_change=change("bid"),
        best_ask_change=change("ask"),
        bid_displayed_size_removal_proxy=bid_removed_amount,
        ask_displayed_size_removal_proxy=ask_removed_amount,
    )


def _trade_flow(
    classifications: list[tuple[UnderlyingTickTradeEvent, TradeSideClassification]],
    *,
    duration_seconds: float,
    minimum_classification_valid_fraction: float,
) -> TradeFlowSummary:
    buy = [trade for trade, result in classifications if result.side is ProbableTradeSide.BUY]
    sell = [trade for trade, result in classifications if result.side is ProbableTradeSide.SELL]
    unknown = [
        trade
        for trade, result in classifications
        if result.side in {ProbableTradeSide.UNKNOWN, ProbableTradeSide.UNCLASSIFIED}
    ]
    buy_volume = sum(item.size for item in buy)
    sell_volume = sum(item.size for item in sell)
    unknown_volume = sum(item.size for item in unknown)
    total = buy_volume + sell_volume + unknown_volume
    valid = buy_volume + sell_volume
    valid_fraction = valid / (total + EPSILON) if total > 0.0 else 0.0
    sizes = [trade.size for trade, _ in classifications]
    valid_features = valid_fraction >= minimum_classification_valid_fraction and valid > 0.0
    return TradeFlowSummary(
        probable_buy_volume=buy_volume,
        probable_sell_volume=sell_volume,
        unknown_volume=unknown_volume,
        probable_buy_trade_count=len(buy),
        probable_sell_trade_count=len(sell),
        unknown_trade_count=len(unknown),
        trade_imbalance=(
            (buy_volume - sell_volume) / (valid + EPSILON) if valid_features else None
        ),
        unknown_volume_fraction=unknown_volume / (total + EPSILON) if total > 0.0 else 0.0,
        classification_valid_fraction=valid_fraction,
        mean_trade_size=statistics.fmean(sizes) if sizes else None,
        median_trade_size=statistics.median(sizes) if sizes else None,
        trade_arrival_rate=len(sizes) / max(duration_seconds, EPSILON),
        buy_arrival_rate=len(buy) / max(duration_seconds, EPSILON),
        sell_arrival_rate=len(sell) / max(duration_seconds, EPSILON),
        trade_features_valid=valid_features,
    )


def _impact(
    *,
    side: ProbableTradeSide,
    flow_volume: float,
    event_count: int,
    midpoint_change: float | None,
    starting_midpoint: float | None,
    duration_seconds: float,
) -> PriceImpactSummary:
    signed = (
        None
        if midpoint_change is None
        else midpoint_change
        if side is ProbableTradeSide.BUY
        else -midpoint_change
    )
    support = flow_volume > EPSILON and event_count > 0 and signed is not None
    notional = (
        flow_volume * starting_midpoint
        if starting_midpoint is not None and starting_midpoint > 0.0
        else None
    )
    return PriceImpactSummary(
        side="probable_buy" if side is ProbableTradeSide.BUY else "probable_sell",
        raw_signed_price_impact=signed,
        price_impact_per_share=(
            None if not support else signed / (flow_volume + EPSILON)  # type: ignore[operator]
        ),
        price_impact_per_notional=(
            None if not support or notional is None else signed / (notional + EPSILON)  # type: ignore[operator]
        ),
        price_impact_bps=(
            None
            if not support or starting_midpoint is None
            else signed / starting_midpoint * 10_000.0  # type: ignore[operator]
        ),
        flow_volume=flow_volume,
        event_count=event_count,
        window_duration_seconds=duration_seconds,
        support_valid=support,
    )


def _one_replenishment(
    *,
    side: ProbableTradeSide,
    seconds: int,
    classifications: list[tuple[UnderlyingTickTradeEvent, TradeSideClassification]],
    quotes: list[UnderlyingLevel1QuoteEvent],
    quote_keys: list[tuple[datetime, int, int, str]],
) -> ReplenishmentSummary:
    triggers = [(trade, result) for trade, result in classifications if result.side is side]
    depleted_total = restored_total = 0.0
    survival: list[float] = []
    midpoint_responses: list[float] = []
    for trade, result in triggers:
        start_quote = _prevailing_quote(
            quotes,
            quote_keys,
            before_event=trade,
        )
        if start_quote is None or not _valid_quote(start_quote):
            continue
        start_time = trade.ordering_timestamp
        end_time = start_time + timedelta(seconds=seconds)
        after = [
            item
            for item in quotes
            if _event_order_key(item) > _event_order_key(trade)
            and item.ordering_timestamp <= end_time
            and _valid_quote(item)
        ]
        if not after:
            continue
        start_primitive = quote_primitives(start_quote)
        if side is ProbableTradeSide.SELL:
            assert result.prevailing_bid is not None
            start_price = result.prevailing_bid
            assert start_quote.bid_size is not None
            start_size = start_quote.bid_size
            relevant_sizes = [
                0.0 if item.bid is None or item.bid < start_price else float(item.bid_size or 0.0)
                for item in after
            ]
            survival_event = next(
                (item for item in after if item.bid is not None and item.bid < start_price),
                None,
            )
        else:
            assert result.prevailing_ask is not None
            start_price = result.prevailing_ask
            assert start_quote.ask_size is not None
            start_size = start_quote.ask_size
            relevant_sizes = [
                0.0 if item.ask is None or item.ask > start_price else float(item.ask_size or 0.0)
                for item in after
            ]
            survival_event = next(
                (item for item in after if item.ask is not None and item.ask > start_price),
                None,
            )
        minimum = min([start_size, *relevant_sizes])
        depleted = max(0.0, start_size - minimum)
        minimum_index = relevant_sizes.index(min(relevant_sizes))
        restored = max(0.0, max(relevant_sizes[minimum_index:]) - minimum)
        depleted_total += depleted
        restored_total += restored
        survival.append(
            float(seconds)
            if survival_event is None
            else max(0.0, (survival_event.ordering_timestamp - start_time).total_seconds())
        )
        midpoint_responses.append(quote_primitives(after[-1]).midpoint - start_primitive.midpoint)
    label = "bid" if side is ProbableTradeSide.SELL else "ask"
    return ReplenishmentSummary(
        side=label,
        response_interval_seconds=seconds,
        trigger_count=len(triggers),
        size_depleted=depleted_total,
        size_restored=restored_total,
        replenishment_ratio=(
            None if depleted_total <= EPSILON else restored_total / (depleted_total + EPSILON)
        ),
        price_survival_time_seconds=statistics.fmean(survival) if survival else None,
        midpoint_response=(statistics.fmean(midpoint_responses) if midpoint_responses else None),
    )


def _bounded(value: float | None, *, scale: float = 1.0) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if scale <= EPSILON:
        return 0.0
    return min(max(value / scale, -1.0), 1.0)


def _mean_valid(values: dict[str, float | None]) -> float | None:
    valid = [value for value in values.values() if value is not None and math.isfinite(value)]
    return None if not valid else statistics.fmean(valid)


def _scores(
    *,
    quote_flow: QuoteFlowSummary,
    trade_flow: TradeFlowSummary,
    buy_impact: PriceImpactSummary,
    sell_impact: PriceImpactSummary,
    replenishment: dict[str, ReplenishmentSummary],
    latest_primitive: QuotePrimitives | None,
) -> dict[str, DescriptiveScore]:
    quote_imbalance = (
        quote_flow.time_weighted_quote_size_imbalance
        if quote_flow.time_weighted_quote_size_imbalance is not None
        else None
    )
    micro = (
        latest_primitive.microprice_edge_half_spread_fraction
        if latest_primitive is not None
        else None
    )
    trade = trade_flow.trade_imbalance
    ask_depletion = _bounded(quote_flow.ask_displayed_size_removal_proxy)
    bid_depletion = _bounded(quote_flow.bid_displayed_size_removal_proxy)
    bid_improvement = _bounded(
        float(quote_flow.bid_price_improvements - quote_flow.bid_price_deteriorations)
    )
    ask_deterioration = _bounded(
        float(quote_flow.ask_price_deteriorations - quote_flow.ask_price_improvements)
    )
    midpoint = _bounded(quote_flow.midpoint_change)
    buy_efficiency = _bounded(buy_impact.price_impact_bps)
    sell_efficiency = _bounded(sell_impact.price_impact_bps)
    mc = {
        "quote_size_imbalance": quote_imbalance,
        "microprice_edge": micro,
        "trade_imbalance": trade,
        "ask_size_depletion": ask_depletion,
        "bid_price_improvement": bid_improvement,
        "midpoint_movement": midpoint,
        "buy_impact_efficiency": buy_efficiency,
    }
    md = {
        "quote_size_imbalance": None if quote_imbalance is None else -quote_imbalance,
        "microprice_edge": None if micro is None else -micro,
        "trade_imbalance": None if trade is None else -trade,
        "bid_size_depletion": bid_depletion,
        "ask_price_deterioration": ask_deterioration,
        "midpoint_movement": None if midpoint is None else -midpoint,
        "sell_impact_efficiency": sell_efficiency,
    }
    bid_replenishment = replenishment["bid_3s"]
    ask_replenishment = replenishment["ask_3s"]
    bid_ratio = _bounded(bid_replenishment.replenishment_ratio)
    ask_ratio = _bounded(ask_replenishment.replenishment_ratio)
    bid_survival = _bounded(bid_replenishment.price_survival_time_seconds, scale=3.0)
    ask_survival = _bounded(ask_replenishment.price_survival_time_seconds, scale=3.0)
    bid_response = _bounded(bid_replenishment.midpoint_response)
    ask_response = _bounded(ask_replenishment.midpoint_response)
    ma = {
        "probable_sell_pressure": None if trade is None else -trade,
        "bid_replenishment": bid_ratio,
        "non_negative_midpoint_response": bid_response,
        "bid_price_survival": bid_survival,
        "ask_thinning_after_defence": ask_depletion,
        "positive_microprice_recovery": micro,
    }
    mb = {
        "probable_buy_pressure": trade,
        "ask_replenishment": ask_ratio,
        "non_positive_midpoint_response": None if ask_response is None else -ask_response,
        "ask_price_survival": ask_survival,
        "bid_thinning_after_defence": bid_depletion,
        "negative_microprice_recovery": None if micro is None else -micro,
    }
    return {
        name: DescriptiveScore(
            score_id=name,
            components=components,
            composite=_mean_valid(components),
            valid_component_count=sum(value is not None for value in components.values()),
        )
        for name, components in (("MC", mc), ("MD", md), ("MA", ma), ("MB", mb))
    }


def summarise_microstructure_window(
    *,
    symbol: str,
    window_start: datetime,
    window_end: datetime,
    quotes: tuple[UnderlyingLevel1QuoteEvent, ...],
    trades: tuple[UnderlyingTickTradeEvent, ...],
    maximum_quote_age: timedelta,
    minimum_classification_valid_fraction: float,
) -> MicrostructureWindowSummary:
    """Summarise only events available at ``window_end``."""

    if window_end <= window_start:
        raise ValueError("microstructure window end must follow start")
    if not 0.0 <= minimum_classification_valid_fraction <= 1.0:
        raise ValueError("classification quality threshold must be in [0, 1]")
    ordered_quotes = [
        item
        for item in _ordered_quotes(quotes)
        if item.ordering_timestamp <= window_end and item.received_timestamp_utc <= window_end
    ]
    quote_keys = [_event_order_key(item) for item in ordered_quotes]
    ordered_trades = sorted(
        (item for item in trades if window_start <= item.ordering_timestamp <= window_end),
        key=lambda item: (
            item.ordering_timestamp,
            item.received_monotonic_ns,
            item.source_sequence,
            item.event_id,
        ),
    )
    classifications = [
        (
            item,
            classify_probable_trade_side(
                item,
                _prevailing_quote(
                    ordered_quotes,
                    quote_keys,
                    before_event=item,
                ),
                maximum_quote_age=maximum_quote_age,
            ),
        )
        for item in ordered_trades
    ]
    persisted_classifications = tuple(
        PersistedTradeSideClassification(
            **classification.model_dump(),
            trade_event_id=trade.event_id,
            trade_provider_timestamp_utc=trade.provider_timestamp_utc,
            trade_received_timestamp_utc=trade.received_timestamp_utc,
            trade_received_monotonic_ns=trade.received_monotonic_ns,
            trade_source_sequence=trade.source_sequence,
            trade_price=trade.price,
            trade_size=trade.size,
        )
        for trade, classification in classifications
    )
    duration = (window_end - window_start).total_seconds()
    quote_flow = _quote_flow(
        ordered_all=ordered_quotes,
        window_start=window_start,
        window_end=window_end,
    )
    trade_flow = _trade_flow(
        classifications,
        duration_seconds=duration,
        minimum_classification_valid_fraction=minimum_classification_valid_fraction,
    )
    start_quote = _prevailing_quote(
        ordered_quotes,
        quote_keys,
        timestamp=window_start,
    )
    start_midpoint = (
        quote_primitives(start_quote).midpoint
        if start_quote is not None and _valid_quote(start_quote)
        else None
    )
    buy_impact = _impact(
        side=ProbableTradeSide.BUY,
        flow_volume=trade_flow.probable_buy_volume,
        event_count=trade_flow.probable_buy_trade_count,
        midpoint_change=quote_flow.midpoint_change,
        starting_midpoint=start_midpoint,
        duration_seconds=duration,
    )
    sell_impact = _impact(
        side=ProbableTradeSide.SELL,
        flow_volume=trade_flow.probable_sell_volume,
        event_count=trade_flow.probable_sell_trade_count,
        midpoint_change=quote_flow.midpoint_change,
        starting_midpoint=start_midpoint,
        duration_seconds=duration,
    )
    replenishment = {
        f"{side_name}_{seconds}s": _one_replenishment(
            side=side,
            seconds=seconds,
            classifications=classifications,
            quotes=ordered_quotes,
            quote_keys=quote_keys,
        )
        for side_name, side in (
            ("bid", ProbableTradeSide.SELL),
            ("ask", ProbableTradeSide.BUY),
        )
        for seconds in (1, 3, 5)
    }
    latest_quote = _prevailing_quote(
        ordered_quotes,
        quote_keys,
        timestamp=window_end,
    )
    latest_primitive = (
        quote_primitives(latest_quote)
        if latest_quote is not None and _valid_quote(latest_quote)
        else None
    )
    return MicrostructureWindowSummary(
        symbol=symbol,
        window_start=window_start,
        window_end=window_end,
        duration_seconds=duration,
        quote_flow=quote_flow,
        trade_flow=trade_flow,
        trade_classifications=persisted_classifications,
        buy_impact=buy_impact,
        sell_impact=sell_impact,
        replenishment=replenishment,
        scores=_scores(
            quote_flow=quote_flow,
            trade_flow=trade_flow,
            buy_impact=buy_impact,
            sell_impact=sell_impact,
            replenishment=replenishment,
            latest_primitive=latest_primitive,
        ),
        causal_as_of=window_end,
    )


def standard_window_summaries(
    *,
    symbol: str,
    as_of: datetime,
    quotes: tuple[UnderlyingLevel1QuoteEvent, ...],
    trades: tuple[UnderlyingTickTradeEvent, ...],
    maximum_quote_age: timedelta,
    minimum_classification_valid_fraction: float,
) -> tuple[MicrostructureWindowSummary, ...]:
    return tuple(
        summarise_microstructure_window(
            symbol=symbol,
            window_start=as_of - window,
            window_end=as_of,
            quotes=quotes,
            trades=trades,
            maximum_quote_age=maximum_quote_age,
            minimum_classification_valid_fraction=minimum_classification_valid_fraction,
        )
        for window in STANDARD_WINDOWS
    )


def episode_relative_windows(
    *,
    trigger_timestamp: datetime,
    entry_timestamp: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    return {
        "T-15m_to_T-10m": (
            trigger_timestamp - timedelta(minutes=15),
            trigger_timestamp - timedelta(minutes=10),
        ),
        "T-10m_to_T-5m": (
            trigger_timestamp - timedelta(minutes=10),
            trigger_timestamp - timedelta(minutes=5),
        ),
        "T-5m_to_T": (trigger_timestamp - timedelta(minutes=5), trigger_timestamp),
        "T_to_entry": (trigger_timestamp, entry_timestamp),
        "entry_to_+5m": (entry_timestamp, entry_timestamp + timedelta(minutes=5)),
        "entry_to_+10m": (entry_timestamp, entry_timestamp + timedelta(minutes=10)),
        "entry_to_+15m": (entry_timestamp, entry_timestamp + timedelta(minutes=15)),
        "entry_to_+30m": (entry_timestamp, entry_timestamp + timedelta(minutes=30)),
    }
