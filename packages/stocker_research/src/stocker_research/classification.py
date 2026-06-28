"""Conservative research result classification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ResearchClassification = Literal[
    "rejected_data_issue",
    "rejected_insufficient_data",
    "rejected_no_edge",
    "rejected_costs_kill_edge",
    "rejected_unstable_parameters",
    "rejected_walkforward_failure",
    "rejected_too_few_trades",
    "rejected_overnight_risk",
    "rejected_weekend_risk",
    "rejected_holding_policy_violation",
    "interesting_needs_more_tests",
    "interesting_intraday_needs_more_tests",
    "interesting_swing_needs_more_tests",
    "candidate_intraday_test",
    "candidate_swing_exceptional",
    "candidate_paper_test",
]


class ClassificationResult(BaseModel):
    """Classification and concrete reasons."""

    classification: ResearchClassification
    reasons: list[str]


def classify_research_result(
    *,
    net_test_return: float,
    gross_test_return: float,
    trade_count: int,
    stability_score: float,
    profitable_split_pct: float,
    max_drawdown: float,
    cost_drag: float,
    data_errors: int,
    leakage_errors: int,
    minimum_trades: int = 20,
    minimum_profitable_split_pct: float = 0.6,
    minimum_stability_score: float = 0.5,
    max_allowed_drawdown: float = -0.25,
    benchmark_pass: bool = True,
    null_pass: bool = True,
    selection_rejected: bool = False,
    holding_policy_classification: ResearchClassification | None = None,
    holding_policy_reasons: list[str] | None = None,
) -> ClassificationResult:
    """Classify a research result with intentionally conservative gates."""

    reasons: list[str] = []
    if data_errors:
        reasons.append("data_errors")
        return ClassificationResult(classification="rejected_data_issue", reasons=reasons)
    if leakage_errors:
        reasons.append("leakage_errors")
        return ClassificationResult(classification="rejected_data_issue", reasons=reasons)
    if selection_rejected:
        reasons.append("no_train_selected_parameter")
        return ClassificationResult(classification="rejected_no_edge", reasons=reasons)
    if trade_count < minimum_trades:
        reasons.append("too_few_trades")
        return ClassificationResult(classification="rejected_too_few_trades", reasons=reasons)
    if gross_test_return > 0 and net_test_return <= 0 and cost_drag > 0:
        reasons.append("costs_kill_edge")
        return ClassificationResult(classification="rejected_costs_kill_edge", reasons=reasons)
    if net_test_return <= 0:
        reasons.append("no_positive_net_edge")
        return ClassificationResult(classification="rejected_no_edge", reasons=reasons)
    if not benchmark_pass:
        reasons.append("failed_benchmark")
        return ClassificationResult(classification="rejected_no_edge", reasons=reasons)
    if not null_pass:
        reasons.append("failed_null_timing")
        return ClassificationResult(classification="rejected_no_edge", reasons=reasons)
    holding_reasons = holding_policy_reasons or []
    if holding_policy_classification is not None and holding_policy_classification.startswith(
        "rejected_"
    ):
        reasons.extend(holding_reasons)
        return ClassificationResult(
            classification=holding_policy_classification,
            reasons=reasons,
        )
    if profitable_split_pct < minimum_profitable_split_pct:
        reasons.append("walkforward_failure")
        return ClassificationResult(
            classification="rejected_walkforward_failure",
            reasons=reasons,
        )
    if stability_score < minimum_stability_score:
        reasons.append("unstable_parameters")
        return ClassificationResult(
            classification="rejected_unstable_parameters",
            reasons=reasons,
        )
    if max_drawdown < max_allowed_drawdown:
        reasons.append("drawdown_too_large")
        return ClassificationResult(
            classification="interesting_needs_more_tests",
            reasons=reasons,
        )
    if holding_policy_classification in {
        "interesting_intraday_needs_more_tests",
        "interesting_swing_needs_more_tests",
        "candidate_intraday_test",
        "candidate_swing_exceptional",
    }:
        reasons.extend(holding_reasons)
        return ClassificationResult(
            classification=holding_policy_classification,
            reasons=reasons,
        )
    reasons.append("all_candidate_gates_passed")
    return ClassificationResult(classification="candidate_paper_test", reasons=reasons)
