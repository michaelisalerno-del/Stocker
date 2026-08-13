"""Deterministic, research-only direction signs over frozen microstructure summaries."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from stocker_prospective.events import UnderlyingLevel1QuoteEvent
from stocker_prospective.market_data import MarketDataType
from stocker_prospective.microstructure import MicrostructureWindowSummary, ProbableTradeSide

RESEARCH_LABEL_V0 = (
    "RESEARCH ONLY — MICROSTRUCTURE DIRECTION V0 — NOT VALIDATED — NO RECOMMENDATION"
)
FORMULAS_VERSION_V0 = "microstructure-direction-v0-sign-methods-2026-08-13"


class DirectionActionV0(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    ABSTAIN = "ABSTAIN"


class DirectionMethodV0(StrEnum):
    M01 = "M01_TRADE_IMBALANCE_SIGN"
    M02 = "M02_QUOTE_IMBALANCE_SIGN"
    M03 = "M03_MICROPRICE_EDGE_SIGN"
    M04 = "M04_MC_MD_DIFFERENCE"
    M05 = "M05_STRICT_CONSENSUS"
    M06 = "M06_MAJORITY_SIGN"


class TickByTickStatusV0(StrEnum):
    BIDASK_AND_LAST_PRESENT = "BIDASK_AND_LAST_PRESENT"
    BIDASK_ONLY = "BIDASK_ONLY"
    LAST_ONLY = "LAST_ONLY"
    ABSENT = "ABSENT"


class DepthStatusV0(StrEnum):
    PRESENT_VALID = "PRESENT_VALID"
    PRESENT_INVALID = "PRESENT_INVALID"
    ABSENT = "ABSENT"


class DirectionMarketDataTypeV0(StrEnum):
    UNKNOWN = MarketDataType.UNKNOWN.value
    LIVE = MarketDataType.LIVE.value
    FROZEN = MarketDataType.FROZEN.value
    DELAYED = MarketDataType.DELAYED.value
    DELAYED_FROZEN = MarketDataType.DELAYED_FROZEN.value
    MIXED = "mixed"


class CausalEntryV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    timestamp_utc: datetime
    information_cutoff_utc: datetime
    delay_seconds: float
    midpoint: float
    market_data_type: DirectionMarketDataTypeV0


class MicrostructureDirectionResultV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    run_id: str
    symbol: str
    trigger_timestamp_utc: datetime
    decision_timestamp_utc: datetime
    information_cutoff_utc: datetime
    confirmation_delay_seconds: float
    window_name: str
    direction_method: DirectionMethodV0
    action: DirectionActionV0
    signed_score: float | None
    component_values: dict[str, float | None]
    component_validity: dict[str, bool]
    quote_count: int
    trade_count: int
    classified_trade_count: int
    trade_classification_valid_fraction: float
    stale_quote_fraction: float | None
    unclassified_trade_fraction: float | None
    unknown_trade_volume_fraction: float
    probable_buyer_initiated_volume: float
    probable_seller_initiated_volume: float
    tick_by_tick_status: TickByTickStatusV0
    depth_status: DepthStatusV0
    market_data_type: DirectionMarketDataTypeV0
    data_quality_flags: tuple[str, ...]
    causal_valid: bool
    formulas_version: str = FORMULAS_VERSION_V0
    research_label: str = RESEARCH_LABEL_V0


def microstructure_direction_windows_v0(
    trigger_timestamp_utc: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    """Return the frozen V0 evidence windows in declared display order."""

    if trigger_timestamp_utc.tzinfo is None or trigger_timestamp_utc.utcoffset() is None:
        raise ValueError("microstructure direction trigger must be timezone-aware")
    return {
        "T-60s_to_T0": (trigger_timestamp_utc - timedelta(seconds=60), trigger_timestamp_utc),
        "T-30s_to_T0": (trigger_timestamp_utc - timedelta(seconds=30), trigger_timestamp_utc),
        "T-15s_to_T0": (trigger_timestamp_utc - timedelta(seconds=15), trigger_timestamp_utc),
        "T-5s_to_T0": (trigger_timestamp_utc - timedelta(seconds=5), trigger_timestamp_utc),
        "T0_to_T+1s": (trigger_timestamp_utc, trigger_timestamp_utc + timedelta(seconds=1)),
        "T0_to_T+5s": (trigger_timestamp_utc, trigger_timestamp_utc + timedelta(seconds=5)),
        "T0_to_T+15s": (trigger_timestamp_utc, trigger_timestamp_utc + timedelta(seconds=15)),
        "T0_to_T+30s": (trigger_timestamp_utc, trigger_timestamp_utc + timedelta(seconds=30)),
        "T0_to_T+60s": (trigger_timestamp_utc, trigger_timestamp_utc + timedelta(seconds=60)),
    }


def select_first_causal_entry_v0(
    *,
    information_cutoff_utc: datetime,
    quotes: tuple[UnderlyingLevel1QuoteEvent, ...],
) -> CausalEntryV0 | None:
    """Select the first valid top-of-book observation strictly after the cutoff."""

    candidates = sorted(
        (
            quote
            for quote in quotes
            if quote.received_timestamp_utc > information_cutoff_utc
            and quote.quote_valid
            and quote.bid is not None
            and quote.ask is not None
            and math.isfinite(quote.bid)
            and math.isfinite(quote.ask)
            and 0.0 < quote.bid <= quote.ask
        ),
        key=lambda quote: (
            quote.received_timestamp_utc,
            quote.received_monotonic_ns,
            quote.source_sequence,
            quote.event_id,
        ),
    )
    if not candidates:
        return None
    selected = candidates[0]
    assert selected.bid is not None
    assert selected.ask is not None
    return CausalEntryV0(
        event_id=selected.event_id,
        timestamp_utc=selected.received_timestamp_utc,
        information_cutoff_utc=information_cutoff_utc,
        delay_seconds=(selected.received_timestamp_utc - information_cutoff_utc).total_seconds(),
        midpoint=(selected.bid + selected.ask) / 2.0,
        market_data_type=selected.market_data_type.value,
    )


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _sign_action(value: float | None, *, valid: bool) -> DirectionActionV0:
    if not valid or value is None or not math.isfinite(value) or value == 0.0:
        return DirectionActionV0.ABSTAIN
    return DirectionActionV0.UP if value > 0.0 else DirectionActionV0.DOWN


def _tick_status(*, bidask: bool, last: bool) -> TickByTickStatusV0:
    if bidask and last:
        return TickByTickStatusV0.BIDASK_AND_LAST_PRESENT
    if bidask:
        return TickByTickStatusV0.BIDASK_ONLY
    if last:
        return TickByTickStatusV0.LAST_ONLY
    return TickByTickStatusV0.ABSENT


def _depth_status(*, present: bool, valid: bool) -> DepthStatusV0:
    if not present:
        return DepthStatusV0.ABSENT
    return DepthStatusV0.PRESENT_VALID if valid else DepthStatusV0.PRESENT_INVALID


def build_microstructure_direction_v0(
    *,
    episode_id: str,
    run_id: str,
    trigger_timestamp_utc: datetime,
    window_name: str,
    summary: MicrostructureWindowSummary,
    tick_bidask_present: bool,
    tick_last_present: bool,
    depth_present: bool,
    depth_valid: bool,
    market_data_type: str,
    data_quality_flags: tuple[str, ...],
) -> tuple[MicrostructureDirectionResultV0, ...]:
    """Apply the six frozen sign methods without fitting, weights, or imputation."""

    decision = summary.window_end
    causal_valid = summary.causal_as_of <= decision
    flags = list(data_quality_flags)
    if not causal_valid:
        flags.append("information_after_decision")
    direction_market_data_type = DirectionMarketDataTypeV0(market_data_type)
    if direction_market_data_type is not DirectionMarketDataTypeV0.LIVE:
        flags.append("non_live_market_data")
    flags = sorted(set(flags))
    evidence_contract_valid = causal_valid and not {
        "data_gap",
        "information_after_decision",
        "non_live_market_data",
    }.intersection(flags)

    trade = summary.trade_flow.trade_imbalance
    quote = summary.quote_flow.time_weighted_quote_size_imbalance
    mc = summary.scores.get("MC")
    md = summary.scores.get("MD")
    micro = None if mc is None else mc.components.get("microprice_edge")
    mc_composite = None if mc is None else mc.composite
    md_composite = None if md is None else md.composite
    mc_md = mc_composite - md_composite if _finite(mc_composite) and _finite(md_composite) else None

    classified_volume = (
        summary.trade_flow.probable_buy_volume + summary.trade_flow.probable_sell_volume
    )
    trade_valid = bool(
        evidence_contract_valid
        and tick_last_present
        and summary.trade_flow.trade_features_valid
        and classified_volume > 0.0
        and _finite(trade)
    )
    quote_valid = bool(evidence_contract_valid and _finite(quote))
    micro_valid = bool(evidence_contract_valid and _finite(micro))
    mc_md_valid = bool(evidence_contract_valid and _finite(mc_md))
    validity = {
        "trade_imbalance": trade_valid,
        "quote_imbalance": quote_valid,
        "microprice_edge": micro_valid,
        "mc_composite": bool(evidence_contract_valid and _finite(mc_composite)),
        "md_composite": bool(evidence_contract_valid and _finite(md_composite)),
    }
    values = {
        "trade_imbalance": trade,
        "quote_imbalance": quote,
        "microprice_edge": micro,
        "mc_composite": mc_composite,
        "md_composite": md_composite,
        "probable_buyer_initiated_volume": summary.trade_flow.probable_buy_volume,
        "probable_seller_initiated_volume": summary.trade_flow.probable_sell_volume,
    }
    core = (trade, quote, micro)
    core_valid = trade_valid and quote_valid and micro_valid
    finite_core = tuple(value for value in core if value is not None)
    consensus_score = statistics.fmean(finite_core) if finite_core else None
    strict_action = DirectionActionV0.ABSTAIN
    majority_action = DirectionActionV0.ABSTAIN
    if core_valid:
        if all(value is not None and value > 0.0 for value in core):
            strict_action = DirectionActionV0.UP
        elif all(value is not None and value < 0.0 for value in core):
            strict_action = DirectionActionV0.DOWN
        positive = sum(value is not None and value > 0.0 for value in core)
        negative = sum(value is not None and value < 0.0 for value in core)
        if positive >= 2:
            majority_action = DirectionActionV0.UP
        elif negative >= 2:
            majority_action = DirectionActionV0.DOWN

    method_values = {
        DirectionMethodV0.M01: (trade, _sign_action(trade, valid=trade_valid)),
        DirectionMethodV0.M02: (quote, _sign_action(quote, valid=quote_valid)),
        DirectionMethodV0.M03: (micro, _sign_action(micro, valid=micro_valid)),
        DirectionMethodV0.M04: (mc_md, _sign_action(mc_md, valid=mc_md_valid)),
        DirectionMethodV0.M05: (
            consensus_score if core_valid else None,
            strict_action,
        ),
        DirectionMethodV0.M06: (
            consensus_score if core_valid else None,
            majority_action,
        ),
    }
    unclassified_count = sum(
        classification.side is ProbableTradeSide.UNCLASSIFIED
        for classification in summary.trade_classifications
    )
    unclassified_fraction = (
        unclassified_count / len(summary.trade_classifications)
        if summary.trade_classifications
        else None
    )
    trade_count = (
        summary.trade_flow.probable_buy_trade_count
        + summary.trade_flow.probable_sell_trade_count
        + summary.trade_flow.unknown_trade_count
    )
    classified_count = (
        summary.trade_flow.probable_buy_trade_count + summary.trade_flow.probable_sell_trade_count
    )
    confirmation_delay = max(0.0, (decision - trigger_timestamp_utc).total_seconds())
    common = {
        "episode_id": episode_id,
        "run_id": run_id,
        "symbol": summary.symbol,
        "trigger_timestamp_utc": trigger_timestamp_utc,
        "decision_timestamp_utc": decision,
        "information_cutoff_utc": decision,
        "confirmation_delay_seconds": confirmation_delay,
        "window_name": window_name,
        "component_values": values,
        "component_validity": validity,
        "quote_count": summary.quote_flow.quote_update_count,
        "trade_count": trade_count,
        "classified_trade_count": classified_count,
        "trade_classification_valid_fraction": (summary.trade_flow.classification_valid_fraction),
        # Maximum quote age is not retained in each summary, so exact stale-only
        # attribution is unavailable; unclassified trades retain the broader signal.
        "stale_quote_fraction": None,
        "unclassified_trade_fraction": unclassified_fraction,
        "unknown_trade_volume_fraction": summary.trade_flow.unknown_volume_fraction,
        "probable_buyer_initiated_volume": summary.trade_flow.probable_buy_volume,
        "probable_seller_initiated_volume": summary.trade_flow.probable_sell_volume,
        "tick_by_tick_status": _tick_status(
            bidask=tick_bidask_present,
            last=tick_last_present,
        ),
        "depth_status": _depth_status(present=depth_present, valid=depth_valid),
        "market_data_type": direction_market_data_type,
        "data_quality_flags": tuple(flags),
        "causal_valid": causal_valid,
    }
    return tuple(
        MicrostructureDirectionResultV0(
            direction_method=method,
            signed_score=score,
            action=action,
            **common,
        )
        for method, (score, action) in method_values.items()
    )


__all__ = [
    "CausalEntryV0",
    "DepthStatusV0",
    "DirectionActionV0",
    "DirectionMarketDataTypeV0",
    "DirectionMethodV0",
    "FORMULAS_VERSION_V0",
    "MicrostructureDirectionResultV0",
    "RESEARCH_LABEL_V0",
    "TickByTickStatusV0",
    "build_microstructure_direction_v0",
    "microstructure_direction_windows_v0",
    "select_first_causal_entry_v0",
]
