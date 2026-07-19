"""Frozen support, development, and prospective gate decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


@dataclass(frozen=True)
class GateDecision:
    """Machine-readable gate result."""

    decision: str
    passed: bool
    gates: dict[str, Any]
    authorises_prospective_freeze: bool = False


def evaluate_support_gate(
    events: pd.DataFrame,
    slates: pd.DataFrame,
    *,
    exact_event_rerun: bool,
    outcome_free_audit: bool,
    data_blocker: str | None = None,
) -> GateDecision:
    """Apply every mandatory outcome-free support gate without changing the denominator."""

    otherwise_valid = slates.loc[
        slates["otherwise_valid_scheduled_slate"].astype(bool)
        & slates["eligible_stock_count"].ge(50)
    ]
    supported = otherwise_valid.loc[otherwise_valid["candidate_count"].ge(8)]
    supported_ids = set(supported["slate_id"].astype(str))
    supported_events = events.loc[events["slate_id"].astype(str).isin(supported_ids)].copy()
    valid_count = len(otherwise_valid)
    support_fraction = len(supported) / valid_count if valid_count else 0.0
    event_count = supported_events["event_id"].nunique() if not supported_events.empty else 0
    stock_count = supported_events["symbol"].nunique() if not supported_events.empty else 0
    if supported_events.empty:
        month_count = 0
        max_stock_fraction = 0.0
    else:
        sessions = pd.to_datetime(supported_events["session"], utc=True)
        month_count = sessions.dt.tz_localize(None).dt.to_period("M").nunique()
        max_stock_fraction = float(
            supported_events["symbol"].value_counts().max() / len(supported_events)
        )
    checks: dict[str, Any] = {
        "otherwise_valid_scheduled_slates": valid_count,
        "supported_slates": len(supported),
        "unique_event_rows": int(event_count),
        "candidate_support_fraction": support_fraction,
        "distinct_event_stocks": int(stock_count),
        "calendar_months": int(month_count),
        "max_stock_event_fraction": max_stock_fraction,
        "exact_event_rerun": exact_event_rerun,
        "outcome_free_event_audit": outcome_free_audit,
    }
    passed = bool(
        len(supported) >= 1_000
        and event_count >= 5_000
        and support_fraction >= 0.60
        and stock_count >= 30
        and month_count >= 6
        and max_stock_fraction <= 0.075
        and exact_event_rerun
        and outcome_free_audit
        and data_blocker is None
    )
    if data_blocker is not None:
        decision = data_blocker
    elif passed:
        decision = "support_gate_passed"
    else:
        decision = "blocked_insufficient_cross_sectional_support"
    return GateDecision(decision=decision, passed=passed, gates=checks)


def evaluate_development_gate(
    *,
    support_passed: bool,
    candidate_mean_ic: float,
    baseline_mean_ic: float,
    candidate_top_two_minus_median: float,
    baseline_top_two_minus_median: float,
    positive_fold_fraction: float,
    max_stock_event_fraction: float,
    max_stock_top_two_fraction: float,
    exact_rerun: bool,
    independent_audit: bool,
) -> GateDecision:
    """Decide whether historical evidence may only authorise a prospective freeze."""

    checks = {
        "support_passed": support_passed,
        "candidate_ic_exceeds_baseline": candidate_mean_ic > baseline_mean_ic,
        "candidate_top_two_exceeds_baseline": (
            candidate_top_two_minus_median > baseline_top_two_minus_median
        ),
        "positive_fold_fraction": positive_fold_fraction,
        "max_stock_event_fraction": max_stock_event_fraction,
        "max_stock_top_two_fraction": max_stock_top_two_fraction,
        "exact_rerun": exact_rerun,
        "independent_audit": independent_audit,
    }
    evidence_passed = bool(
        checks["candidate_ic_exceeds_baseline"]
        and checks["candidate_top_two_exceeds_baseline"]
        and positive_fold_fraction > 0.5
        and max_stock_event_fraction <= 0.075
        and max_stock_top_two_fraction <= 0.15
    )
    passed = bool(support_passed and evidence_passed and exact_rerun and independent_audit)
    if not support_passed:
        decision = "blocked_insufficient_cross_sectional_support"
    elif not exact_rerun or not independent_audit:
        decision = "blocked_audit_or_reproducibility_failure"
    elif evidence_passed:
        decision = "historical_incremental_ranking_evidence_supports_prospective_freeze"
    else:
        decision = "historical_no_incremental_ranking_evidence"
    return GateDecision(
        decision=decision,
        passed=passed,
        gates=checks,
        authorises_prospective_freeze=passed,
    )


def evaluate_prospective_gate(
    *,
    evidence_type: Literal["prospective", "historical_development"],
    months: int,
    evaluable_supported_slates: int,
    event_rows: int,
    candidate_minus_baseline_ic: float,
    ic_ci_lower: float,
    candidate_top_two_minus_median: float,
    baseline_top_two_minus_median: float,
    top_two_difference_ci_lower: float,
    positive_first_six_months: int,
    positive_leave_one_stock_out_fraction: float,
    max_stock_event_fraction: float,
    top_five_selection_fraction: float,
    exact_rerun: bool,
    independent_audit: bool,
) -> GateDecision:
    """Apply the final structural gate only to post-freeze prospective evidence."""

    checks = dict(locals())
    if evidence_type != "prospective":
        return GateDecision(
            decision="prospective_gate_not_evaluable_from_historical_data",
            passed=False,
            gates=checks,
        )
    passed = bool(
        months >= 6
        and evaluable_supported_slates >= 1_000
        and event_rows >= 5_000
        and candidate_minus_baseline_ic >= 0.01
        and ic_ci_lower > 0.0
        and candidate_top_two_minus_median > baseline_top_two_minus_median
        and top_two_difference_ci_lower > 0.0
        and positive_first_six_months >= 4
        and positive_leave_one_stock_out_fraction >= 0.90
        and max_stock_event_fraction <= 0.075
        and top_five_selection_fraction < 0.25
        and exact_rerun
        and independent_audit
    )
    return GateDecision(
        decision="prospective_structural_gate_passed"
        if passed
        else "prospective_structural_gate_failed",
        passed=passed,
        gates=checks,
    )
