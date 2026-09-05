"""Pooled admission arithmetic for the existing Session HARD setup.

No symbol, session, run, or execution-selection conditioning enters the estimate.
Completion timestamps denote when the entire outcome became observable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import fsum, isfinite

from stocker_core.strategies import SESSION_HARD_HV_METHOD
from stocker_execution.session_hard_structure_d import (
    EntryBar,
    SignalStatus,
    StrategySignal,
    nominal_exit_prices,
)


class CostAwareDecision(StrEnum):
    PASS = "TRADE_COST_AWARE_PASS"
    FAIL = "ABSTAIN_COST_AWARE_FAIL"
    WARMUP = "TRADE_BASELINE_WARMUP"


@dataclass(frozen=True, slots=True)
class CompletedPayoff:
    opportunity_id: str
    completion_timestamp: datetime
    gross_r: float

    def __post_init__(self) -> None:
        _aware(self.completion_timestamp)
        if not self.opportunity_id or not isfinite(self.gross_r):
            raise ValueError("completed payoff requires an identity and finite gross R")


@dataclass(frozen=True, slots=True)
class CostAwareAssessment:
    completed_observation_count: int
    estimated_gross_r: float | None
    estimated_round_trip_cost_bps: float
    stop_distance_bps: float
    cost_r: float
    estimated_net_r: float | None
    decision: CostAwareDecision

    @property
    def take_trade(self) -> bool:
        return self.decision in {CostAwareDecision.PASS, CostAwareDecision.WARMUP}


def assess_pooled_payoff(
    *,
    opportunity_id: str,
    signal_timestamp: datetime,
    entry_reference_price: float,
    initial_stop_price: float,
    estimated_round_trip_cost_bps: float,
    observations: Iterable[CompletedPayoff],
) -> CostAwareAssessment:
    """Keep baseline admission for n < 20; afterwards require mean gross R - cost R > 0.

    This function deliberately receives no current outcome or symbol. History can
    be unsorted and overlapping; only strict completion-time causality admits it.
    The current identity is excluded even if incorrectly supplied as completed.
    """
    _aware(signal_timestamp)
    if (
        not isfinite(entry_reference_price)
        or entry_reference_price <= 0
        or not isfinite(initial_stop_price)
        or initial_stop_price <= 0
        or initial_stop_price == entry_reference_price
        or not isfinite(estimated_round_trip_cost_bps)
        or estimated_round_trip_cost_bps < 0
    ):
        raise ValueError("cost hurdle requires valid reference, stop, and cost inputs")
    prior: dict[str, CompletedPayoff] = {}
    for observation in observations:
        if (
            observation.opportunity_id == opportunity_id
            or observation.completion_timestamp >= signal_timestamp
        ):
            continue
        previous = prior.setdefault(observation.opportunity_id, observation)
        if previous != observation:
            raise ValueError("conflicting completed payoff for the same opportunity")
    n = len(prior)
    gross_r = fsum(item.gross_r for item in prior.values()) / n if n else None
    stop_bps = abs(entry_reference_price - initial_stop_price) / entry_reference_price * 10_000
    cost_r = estimated_round_trip_cost_bps / stop_bps
    net_r = gross_r - cost_r if gross_r is not None else None
    decision = (
        CostAwareDecision.WARMUP
        if n < 20
        else CostAwareDecision.PASS
        if net_r is not None and net_r > 0
        else CostAwareDecision.FAIL
    )
    return CostAwareAssessment(
        n, gross_r, estimated_round_trip_cost_bps, stop_bps, cost_r, net_r, decision
    )


def _aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("payoff timestamps must be timezone-aware")


# The historical accounting contract already used by Session HARD research.
# Broker brackets/exits are intentionally independent of this observation horizon.
SHADOW_HORIZON_MINUTES = 15


def is_baseline_payoff_candidate(signal: StrategySignal) -> bool:
    return (
        signal.strategy_id == SESSION_HARD_HV_METHOD.strategy_id
        and signal.strategy_version == SESSION_HARD_HV_METHOD.strategy_version
        and signal.status is SignalStatus.ENTRY_TRIGGERED
        and signal.selected
        and signal.direction == "DOWN"
        and signal.side == "SHORT"
    )


def pooled_opportunity_id(signal: StrategySignal) -> str:
    """Deduplicate the same economic opportunity shared by PAPER/LIVE or runs."""
    identity = (
        signal.strategy_id,
        signal.strategy_version,
        signal.underlying_con_id,
        signal.t0.astimezone(UTC).isoformat(),
        signal.entry_reference,
        signal.m_price,
        signal.stop_distance_m,
        signal.target_distance_m,
        signal.feature_calculation_version,
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()[:32]


def hypothetical_baseline_outcome(
    signal: StrategySignal, bars: Sequence[EntryBar], *, as_of: datetime
) -> CompletedPayoff | None:
    """Frozen Structure D gross accounting, using only a complete minute prefix.

    Reuses the actual Stage 6 fill/reference; never fills an untriggered signal.
    Nominal stop/target touches (including stop-first ambiguity and legacy nominal
    stop-gap accounting) match fixed_exit in the frozen research runner. Timeout
    is the close of T0+14, observable at T0+15, independent of entry delay.
    """
    if not is_baseline_payoff_candidate(signal):
        return None
    _aware(as_of)
    if signal.entry_timestamp is None or signal.entry_reference is None:
        raise ValueError("baseline entry timestamp and reference are required")
    entry = signal.entry_reference
    stop, target = nominal_exit_prices(signal)
    risk = abs(entry - stop)
    start = signal.entry_timestamp.astimezone(UTC)
    end = signal.t0.astimezone(UTC) + timedelta(minutes=SHADOW_HORIZON_MINUTES)
    by_time = {bar.timestamp.astimezone(UTC): bar for bar in bars}
    timestamp = start
    while timestamp < end:
        completion = timestamp + timedelta(minutes=1)
        if completion > as_of:
            return None
        bar = by_time.get(timestamp)
        if bar is None:
            return None
        exit_price = None
        if bar.high >= stop:
            exit_price = stop
        elif bar.low <= target:
            exit_price = target
        elif completion == end:
            exit_price = bar.close
        if exit_price is not None:
            return CompletedPayoff(
                pooled_opportunity_id(signal), completion, (entry - exit_price) / risk
            )
        timestamp = completion
    return None
