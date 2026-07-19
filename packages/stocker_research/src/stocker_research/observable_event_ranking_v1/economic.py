"""Sealed post-prospective quote-based economic evaluation boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EconomicEvaluationPrerequisites:
    """Every prerequisite defaults false or absent; no generic cost assumptions exist."""

    prospective_structural_gate_passed: bool = False
    complete_live_quote_coverage: float = 0.0
    entry_exit_timing_rule_passed: bool = False
    commission_fee_configuration_version: str | None = None
    currency_conversion_treatment: str | None = None
    economic_contract_hash: str | None = None


class EconomicEvaluationBlocked(RuntimeError):
    """Raised unless the separately frozen economic layer is authorised."""


@dataclass(frozen=True)
class EconomicContinuationDecision:
    """Later quote-simulation gate result, never an achieved-fill statement."""

    decision: str
    passed: bool
    quote_simulation_is_achieved_fill: bool = False


def require_economic_prerequisites(prerequisites: EconomicEvaluationPrerequisites) -> None:
    """Fail closed until all structural, quote, cost, currency, and freeze gates pass."""

    blockers: list[str] = []
    if not prerequisites.prospective_structural_gate_passed:
        blockers.append("prospective_structural_gate_not_passed")
    if prerequisites.complete_live_quote_coverage < 0.80:
        blockers.append("live_quote_coverage_below_80_percent")
    if not prerequisites.entry_exit_timing_rule_passed:
        blockers.append("entry_exit_quote_timing_not_satisfied")
    if not prerequisites.commission_fee_configuration_version:
        blockers.append("missing_user_supplied_ibkr_fee_configuration")
    if not prerequisites.currency_conversion_treatment:
        blockers.append("missing_currency_conversion_treatment")
    if not prerequisites.economic_contract_hash:
        blockers.append("economic_contract_not_frozen")
    if blockers:
        raise EconomicEvaluationBlocked("|".join(blockers))


def evaluate_economic_continuation_gate(
    *,
    prerequisites: EconomicEvaluationPrerequisites,
    net_top_two_bootstrap_lower: float,
    cost_stress_1_5x_point_estimate: float,
    max_stock_net_contribution_fraction: float,
    short_period_dominated: bool,
    exact_rerun: bool,
    independent_audit: bool,
) -> EconomicContinuationDecision:
    """Apply the sealed eventual gate after explicit quote/cost prerequisites pass."""

    require_economic_prerequisites(prerequisites)
    passed = bool(
        net_top_two_bootstrap_lower > 0.0
        and cost_stress_1_5x_point_estimate > 0.0
        and max_stock_net_contribution_fraction <= 0.15
        and not short_period_dominated
        and exact_rerun
        and independent_audit
    )
    return EconomicContinuationDecision(
        decision=(
            "economic_quote_simulation_continuation_gate_passed"
            if passed
            else "economic_quote_simulation_continuation_gate_failed"
        ),
        passed=passed,
    )
