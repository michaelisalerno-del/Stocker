"""Operational edge states and new-entry admission decisions."""

from __future__ import annotations

from dataclasses import dataclass

from stocker_research.dynamic_loop_edge_state.online_state import (
    EdgeForecast,
    SupportEvidence,
)


@dataclass(frozen=True)
class DecisionThresholds:
    minimum_independent_sessions: int = 8
    minimum_independent_stocks: int = 5
    minimum_effective_sample_size: float = 12.0
    maximum_posterior_std_net_bps: float = 80.0
    active_probability: float = 0.9
    survival_probability: float = 0.9
    decaying_termination_probability: float = 0.25
    retired_positive_probability: float = 0.35
    change_reset_probability: float = 0.45
    out_of_distribution_threshold: float = 4.0


@dataclass(frozen=True)
class EdgeDecision:
    edge_state: str
    admit_new_entry: bool
    reason_codes: tuple[str, ...]
    existing_position_action: str


def classify_edge_state(
    forecast: EdgeForecast,
    support: SupportEvidence,
    thresholds: DecisionThresholds,
    *,
    previous_state: str,
    has_existing_position: bool = False,
    required_features_available: bool = True,
    unresolved_outcomes: bool = False,
    structural_breadth_collapse: bool = False,
    high_cost_pressure: bool = False,
) -> EdgeDecision:
    """Classify a payoff state; unknown is the default and only active admits."""

    existing_action = "unchanged_existing_exit_rule" if has_existing_position else "not_applicable"
    unknown_reasons: list[str] = []
    if support.effective_sessions < thresholds.minimum_independent_sessions:
        unknown_reasons.append("insufficient_sessions")
    if support.independent_stocks < thresholds.minimum_independent_stocks:
        unknown_reasons.append("insufficient_stocks")
    if support.effective_sample_size < thresholds.minimum_effective_sample_size:
        unknown_reasons.append("insufficient_effective_sample_size")
    if forecast.posterior_std_net_bps > thresholds.maximum_posterior_std_net_bps:
        unknown_reasons.append("posterior_uncertainty_too_high")
    if forecast.out_of_distribution_score > thresholds.out_of_distribution_threshold:
        unknown_reasons.append("out_of_distribution")
    if not required_features_available:
        unknown_reasons.append("missing_features")
    if unresolved_outcomes:
        unknown_reasons.append("unresolved_outcomes")
    if unknown_reasons:
        return EdgeDecision("unknown", False, tuple(unknown_reasons), existing_action)

    retired = (
        forecast.posterior_mean_net_bps <= 0.0
        and forecast.p_edge_positive <= thresholds.retired_positive_probability
    ) or (
        previous_state in {"active", "decaying"}
        and forecast.p_change_now >= thresholds.change_reset_probability
        and forecast.p_edge_positive < 0.5
    )
    if retired:
        reasons = ["retired_state", "edge_probability_too_low"]
        if forecast.posterior_lower_bound_net_bps <= 0.0:
            reasons.append("lower_bound_not_positive")
        return EdgeDecision("retired", False, tuple(reasons), existing_action)

    decaying = previous_state in {"active", "decaying"} and (
        forecast.p_off_next >= thresholds.decaying_termination_probability
        or forecast.posterior_lower_bound_net_bps <= 0.0
        or structural_breadth_collapse
        or high_cost_pressure
    )
    if decaying:
        reasons = ["decaying_state"]
        if forecast.p_off_next >= thresholds.decaying_termination_probability:
            reasons.append("termination_probability_too_high")
        if forecast.posterior_lower_bound_net_bps <= 0.0:
            reasons.append("lower_bound_not_positive")
        if structural_breadth_collapse:
            reasons.append("structural_breadth_collapse")
        if high_cost_pressure:
            reasons.append("high_cost_pressure")
        return EdgeDecision("decaying", False, tuple(reasons), existing_action)

    admission_reasons: list[str] = []
    if forecast.p_edge_active < thresholds.active_probability:
        admission_reasons.append("edge_probability_too_low")
    if forecast.posterior_lower_bound_net_bps <= 0.0:
        admission_reasons.append("lower_bound_not_positive")
    if forecast.p_survive_horizon < thresholds.survival_probability:
        admission_reasons.append("survival_probability_too_low")
    if forecast.p_off_next >= thresholds.decaying_termination_probability:
        admission_reasons.append("termination_probability_too_high")
    if structural_breadth_collapse:
        admission_reasons.append("structural_breadth_collapse")
    if high_cost_pressure:
        admission_reasons.append("high_cost_pressure")
    if admission_reasons:
        return EdgeDecision("unknown", False, tuple(admission_reasons), existing_action)
    return EdgeDecision("active", True, ("admitted_active_edge",), existing_action)


__all__ = [
    "DecisionThresholds",
    "EdgeDecision",
    "SupportEvidence",
    "classify_edge_state",
]
