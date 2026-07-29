"""Create the one-time M1C Prospective Opening Reversal V1 activation package.

The builder reads only the frozen 2024/2025 opening-transition predictor
surface and event identifiers needed to reconcile the earlier report labels.
It never reads a 2026 outcome, connects to a broker, or exposes an order path.
Once the activation receipt exists, the builder refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from stocker_prospective.m1c_prospective_opening_reversal_v1 import (
    DESCRIPTIVE_OVERNIGHT_GAP_Q10_V1,
    DESCRIPTIVE_OVERNIGHT_GAP_Q90_V1,
    DESCRIPTIVE_TOTAL_TRANSITION_Q10_V1,
    DESCRIPTIVE_TOTAL_TRANSITION_Q90_V1,
    NEGATIVE_OPENING_RETURN_THRESHOLD_V1,
    OPENING_RANGE_THRESHOLD_V1,
    POSITIVE_OPENING_RETURN_THRESHOLD_V1,
    FrozenOpeningReversalRuleV1,
    build_activation_receipt_v1,
    build_frozen_experiment_config_v1,
    load_activation_receipt_v1,
    load_frozen_experiment_config_v1,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PUBLISHED_ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts" / "primary"
PUBLISHED_REPORT_ROOT = EXPERIMENT_ROOT / "reports"
ARTIFACT_ROOT = PUBLISHED_ARTIFACT_ROOT
REPORT_ROOT = PUBLISHED_REPORT_ROOT
RETROSPECTIVE_ROOT = (
    ROOT
    / "research"
    / "directional-readiness"
    / "20260728-m1c-opening-market-transition-v1"
    / "artifacts"
    / "primary"
)
ACTIVATION_PATH = ARTIFACT_ROOT / "experiment_activation_receipt_v1.json"
_GIT_STATE_OVERRIDE: tuple[str, str, str] | None = None


def _json_ready(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _event_reconciliation() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconcile event counts without loading any return/outcome column."""

    market = pd.read_parquet(
        RETROSPECTIVE_ROOT / "opening_market_state_surface_v1.parquet",
        columns=[
            "partition",
            "session",
            "checkpoint",
            "market_proxy_v1",
            "opening_market_transition_state_v1",
            "opening_transition_sign_v1",
            "opening_transition_event_id_v1",
        ],
    )
    eligible = pd.read_csv(
        RETROSPECTIVE_ROOT / "unique_opening_transition_events_v1.csv",
        usecols=[
            "opening_transition_event_id_v1",
            "stock_episode_count",
        ],
    )
    market["session"] = pd.to_datetime(market["session"], errors="raise")
    if bool((market["session"].dt.year >= 2026).any()):
        raise RuntimeError("protected_2026_market_rows_refused")
    severe = market[
        market["opening_transition_sign_v1"].isin([-1, 1])
        & market["opening_transition_event_id_v1"].notna()
    ].copy()
    if severe["opening_transition_event_id_v1"].duplicated().any():
        raise RuntimeError("retrospective_event_id_is_not_unique")
    eligible_counts: dict[str, int] = {}
    for raw_eligible_row in eligible.itertuples(index=False):
        eligible_row = cast(Any, raw_eligible_row)
        eligible_counts[str(eligible_row.opening_transition_event_id_v1)] = int(
            eligible_row.stock_episode_count
        )
    rows: list[dict[str, Any]] = []
    for raw_row in severe.sort_values(["partition", "session"]).itertuples(index=False):
        row = cast(Any, raw_row)
        event_id = str(row.opening_transition_event_id_v1)
        sign = int(row.opening_transition_sign_v1)
        eligible_count = eligible_counts.get(event_id, 0)
        included = eligible_count > 0
        if included:
            explanation = (
                "Corrected common population: the severe VTI event contains "
                f"{eligible_count} eligible checkpoint-6 stock episode(s), "
                "contributes once to the total, and contributes to exactly one "
                "signed count. The prior signed columns counted all severe VTI "
                "sessions while the total counted only events with eligible stocks."
            )
        else:
            explanation = (
                "All-market severe VTI session with zero eligible checkpoint-6 "
                "stock episodes; excluded from the corrected common-population "
                "total and signed counts. The prior mislabeled signed column "
                "included this session."
            )
        rows.append(
            {
                "period": str(row.partition),
                "session": row.session.date().isoformat(),
                "checkpoint": int(row.checkpoint),
                "vti_proxy": str(row.market_proxy_v1),
                "opening_state": str(row.opening_market_transition_state_v1),
                "opening_sign": sign,
                "event_id": event_id,
                "eligible_stock_count": eligible_count,
                "acted_stock_count": eligible_count,
                "included_in_total_event_count": included,
                "included_in_positive_count": included and sign == 1,
                "included_in_negative_count": included and sign == -1,
                "exact_explanation": explanation,
            }
        )
    if any(
        bool(row["included_in_positive_count"])
        and bool(row["included_in_negative_count"])
        for row in rows
    ):
        raise RuntimeError("one_event_received_two_transition_signs")

    old = pd.read_csv(RETROSPECTIVE_ROOT / "event_accounting_v1.csv")
    corrected: list[dict[str, Any]] = []
    expected = {
        "development": (26, 18, 8, 23, 15),
        "assessment": (43, 23, 20, 28, 30),
        "stress": (13, 5, 8, 7, 11),
    }
    for raw_old_row in old.itertuples(index=False):
        old_row = cast(Any, raw_old_row)
        period = str(old_row.period)
        period_rows = [row for row in rows if row["period"] == period]
        total = sum(bool(row["included_in_total_event_count"]) for row in period_rows)
        negative = sum(bool(row["included_in_negative_count"]) for row in period_rows)
        positive = sum(bool(row["included_in_positive_count"]) for row in period_rows)
        all_negative = sum(int(row["opening_sign"]) == -1 for row in period_rows)
        all_positive = sum(int(row["opening_sign"]) == 1 for row in period_rows)
        if (total, negative, positive, all_negative, all_positive) != expected[period]:
            raise RuntimeError(f"event_reconciliation_changed:{period}")
        corrected.append(
            {
                "period": period,
                "stock_episode_count": int(old_row.stock_episode_count),
                "severe_stock_episode_count": int(old_row.severe_stock_episode_count),
                "unique_session_count": int(old_row.unique_session_count),
                "unique_opening_transition_event_count": total,
                "eligible_negative_opening_transition_event_count": negative,
                "eligible_positive_opening_transition_event_count": positive,
                "all_market_negative_severe_session_count": all_negative,
                "all_market_positive_severe_session_count": all_positive,
                "complete_normal_opening_event_count": int(
                    old_row.complete_normal_opening_event_count
                ),
                "incomplete_event_count": int(old_row.incomplete_event_count),
                "mean_stocks_per_transition_event": float(
                    old_row.mean_stocks_per_transition_event
                ),
                "maximum_stocks_per_transition_event": int(
                    old_row.maximum_stocks_per_transition_event
                ),
                "reconciliation_outcome": "event_label_ambiguity_corrected",
                "prior_scientific_interpretation_changes": False,
            }
        )
    return rows, corrected


EMPTY_TABLES: dict[str, list[str]] = {
    "market_data_capacity_snapshot_v1.csv": [
        "timestamp_utc",
        "configured_budget",
        "reserved_lines",
        "mandatory_lines",
        "optional_lines",
        "estimated_free_lines",
        "active_subscription_identifiers_json",
        "subscription_priority_json",
        "owning_subsystem_json",
        "may_be_dropped_json",
        "drop_order_json",
        "pending_subscriptions_json",
        "cancelled_subscriptions_json",
        "awaiting_cleanup_json",
        "current_promoted_episode_id",
        "capacity_denial_reasons_json",
        "exact_broker_accounting_known",
        "uncertainty",
        "snapshot_hash",
    ],
    "engineering_session_report_v1.csv": [
        "session",
        "valid_transfer_ordinal",
        "cohort_phase",
        "vti_ibkr_opening_return",
        "vti_eodhd_opening_return",
        "vti_ibkr_opening_range",
        "vti_eodhd_opening_range",
        "severe_state_agreement",
        "sign_agreement",
        "bar_timestamp_alignment",
        "threshold_boundary_disagreement",
        "source_completeness",
        "prediction_receipt_timing_pass",
        "checkpoint_6_episode_identity_agreement",
        "capacity_reserve_preserved",
        "primary_option_pair_available",
        "graceful_degradation_pass",
        "universe_uninterrupted",
        "orders_placed",
        "status",
    ],
    "prediction_receipts_v1.csv": [
        "experiment_id",
        "experiment_version",
        "session",
        "stock",
        "checkpoint",
        "signal_timestamp_utc",
        "entry_timestamp_utc",
        "receipt_created_at_utc",
        "m1c_probability",
        "m1c_threshold",
        "high_tail_membership",
        "fresh_episode_id",
        "tail_phase_v1",
        "market_opening_return_v1",
        "market_opening_range_v1",
        "opening_market_transition_state_v1",
        "opening_transition_sign_v1",
        "opening_transition_event_id_v1",
        "negative_opening_return_threshold_v1",
        "positive_opening_return_threshold_v1",
        "opening_range_threshold_v1",
        "data_source",
        "transfer_status",
        "cohort_phase",
        "prediction_v1",
        "prediction_sign_v1",
        "eligibility_v1",
        "ineligibility_reasons_v1",
        "completeness_status_v1",
        "scientific_outcome_eligible_v1",
        "scientific_exclusion_reason_v1",
        "capacity_snapshot_id",
        "previous_close_atm_iv_scale_15m",
        "frozen_comparisons_json",
        "rule_hash_v1",
        "receipt_hash_v1",
    ],
    "eligible_episode_table_v1.csv": [
        "session",
        "event_id",
        "stock",
        "m1c_probability",
        "prediction",
        "prediction_sign",
        "opening_sign",
        "cohort_phase",
        "promoted",
        "receipt_hash",
    ],
    "promoted_episode_table_v1.csv": [
        "session",
        "event_id",
        "stock",
        "m1c_probability",
        "promotion_rank",
        "tie_break_rule",
        "capacity_snapshot_hash",
        "primary_option_status",
        "promotion_hash",
    ],
    "non_promoted_eligible_episode_table_v1.csv": [
        "session",
        "event_id",
        "stock",
        "m1c_probability",
        "prediction",
        "opening_transition",
        "capacity_snapshot_hash",
        "winning_promoted_stock",
        "reason_not_promoted",
    ],
    "underlying_outcomes_v1.csv": [
        "prediction_receipt_hash_v1",
        "event_id",
        "session",
        "stock",
        "cohort_phase",
        "r_15m",
        "absolute_r_15m",
        "opening_reversal_aligned_return_v1",
        "material_threshold_15m",
        "material_outcome",
        "correct_predicted_material_direction",
        "accuracy_material_movers",
        "accuracy_no_moves_as_failures",
        "maximum_favourable_excursion",
        "maximum_adverse_excursion",
        "post_entry_local_range_share",
        "iv_residual",
        "exceed_iv_state",
        "outcome_complete",
        "missing_reason",
        "outcome_timestamp_utc",
        "outcome_hash_v1",
    ],
    "primary_option_bid_ask_outcomes_v1.csv": [
        "prediction_receipt_hash_v1",
        "episode_id",
        "contract_id",
        "underlying",
        "expiry",
        "strike",
        "right",
        "multiplier",
        "selection_timestamp_utc",
        "role",
        "entry_timestamp_utc",
        "subscription_start_utc",
        "subscription_end_utc",
        "capacity_line_owner",
        "entry_bid",
        "entry_ask",
        "entry_midpoint_diagnostic",
        "entry_quote_timestamp_utc",
        "exit_bid",
        "exit_ask",
        "exit_midpoint_diagnostic",
        "exit_quote_timestamp_utc",
        "entry_spread",
        "exit_spread",
        "entry_quote_age_seconds",
        "exit_quote_age_seconds",
        "locked_or_crossed",
        "staleness",
        "missingness",
        "conservative_return_v1",
        "complete",
        "missing_reason",
        "outcome_hash_v1",
    ],
    "optional_comparison_outcomes_v1.csv": [
        "episode_id",
        "comparison_kind",
        "contract_id",
        "subscription_start_utc",
        "subscription_end_utc",
        "continuous_coverage",
        "quote_quality_passed",
        "outcome_complete",
        "missing_reason",
    ],
    "optional_feed_degradation_events_v1.csv": [
        "timestamp_utc",
        "feed",
        "action",
        "reason",
        "raw_capacity_reason",
        "capacity_snapshot_hash",
        "episode_id",
        "primary_direction_evidence_complete",
        "primary_option_evidence_complete",
        "event_hash",
    ],
    "contract_discovery_audit_v1.csv": [
        "episode_id",
        "discovery_timestamp_utc",
        "contract_source",
        "cache_hit",
        "candidates_inspected",
        "call_contract_id",
        "put_contract_id",
        "expiry",
        "strike",
        "frozen_tie_break_rule",
        "metadata_request_ended",
        "full_chain_live_subscription_created",
        "live_market_data_lines_consumed",
        "planned_live_market_data_lines",
        "status",
        "missing_reason",
        "selection_hash",
    ],
    "baseline_comparisons_v1.csv": [
        "cohort_phase",
        "population",
        "baseline",
        "population_episode_count",
        "available_episode_count",
        "unavailable_episode_count",
        "event_count",
        "baseline_mean_aligned_return",
        "reversal_mean_on_identical_episodes",
        "reversal_minus_baseline_mean",
    ],
    "a1_comparisons_v1.csv": [
        "cohort_phase",
        "population",
        "agreement_state",
        "opening_reversal_action",
        "a1_action",
        "episode_count",
        "mean_underlying_aligned_return",
        "complete_option_episode_count",
        "mean_conservative_option_return",
    ],
    "stock_response_descriptive_strata_v1.csv": [
        "cohort_phase",
        "stock_opening_response_class_v1",
        "episode_count",
        "event_count",
        "mean_opening_reversal_aligned_return_v1",
        "material_direction_accuracy",
        "descriptive_only",
    ],
    "session_cluster_bootstrap_results_v1.csv": [
        "cohort_phase",
        "seed",
        "replication",
        "mean_aligned_return",
    ],
    "event_cluster_bootstrap_results_v1.csv": [
        "cohort_phase",
        "seed",
        "replication",
        "mean_aligned_return",
    ],
    "primary_null_results_v1.csv": [
        "cohort_phase",
        "seed",
        "replication",
        "mean_aligned_return",
        "material_direction_accuracy",
        "accuracy_no_moves_as_failures",
        "difference_vs_follow_vti",
        "positive_transition_consistent",
        "negative_transition_consistent",
    ],
    "temporal_placebo_results_v1.csv": [
        "cohort_phase",
        "stock",
        "prediction_session",
        "paired_next_eligible_session",
        "placebo_aligned_return",
        "same_cohort",
        "chronology_preserved",
    ],
    "leave_one_stock_out_results_v1.csv": [
        "cohort_phase",
        "omitted_stock",
        "episode_count",
        "event_count",
        "mean_aligned_return",
    ],
    "leave_one_session_out_results_v1.csv": [
        "cohort_phase",
        "omitted_session",
        "episode_count",
        "event_count",
        "mean_aligned_return",
    ],
    "leave_one_event_out_results_v1.csv": [
        "cohort_phase",
        "omitted_event_id",
        "episode_count",
        "event_count",
        "mean_aligned_return",
    ],
    "concentration_report_v1.csv": [
        "cohort_phase",
        "dimension",
        "value",
        "episode_count",
        "episode_fraction",
        "support_limit",
        "passes",
    ],
}


def _pending_receipt(
    *,
    receipt_type: str,
    status: str,
    activation_timestamp: str,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "experiment_id": "m1c-prospective-opening-reversal-v1",
        "experiment_version": "1",
        "receipt_type": receipt_type,
        "status": status,
        "activation_timestamp_utc": activation_timestamp,
        "reason": reason,
        "scientific_result_issued": False,
    }
    payload["receipt_hash"] = _sha256_value(payload)
    return payload


def _build_activation_package() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    activation_timestamp = datetime.now(UTC)
    if _GIT_STATE_OVERRIDE is None:
        branch = _git("branch", "--show-current")
        commit = _git("rev-parse", "HEAD")
        dirty_status = _git("status", "--short") or "clean"
    else:
        branch, commit, dirty_status = _GIT_STATE_OVERRIDE
    config = build_frozen_experiment_config_v1()
    rule = FrozenOpeningReversalRuleV1()
    receipt = build_activation_receipt_v1(
        activation_timestamp_utc=activation_timestamp,
        new_york_trading_date_at_activation=activation_timestamp.astimezone(
            ZoneInfo("America/New_York")
        ).date(),
        branch=branch,
        commit=commit,
        dirty_working_tree_status=dirty_status,
        configuration_hash=config.configuration_hash,
        m1c_version="frozen-m1c-v0",
        tail_phase_version="m1c-tail-phase-v1",
        a1_version="A1/frozen-m1c-microstructure-recorder-v0",
    )

    reconciliation_rows, corrected_accounting = _event_reconciliation()
    _write_csv(
        ARTIFACT_ROOT / "opening_transition_event_count_reconciliation_v1.csv",
        reconciliation_rows,
        [
            "period",
            "session",
            "checkpoint",
            "vti_proxy",
            "opening_state",
            "opening_sign",
            "event_id",
            "eligible_stock_count",
            "acted_stock_count",
            "included_in_total_event_count",
            "included_in_positive_count",
            "included_in_negative_count",
            "exact_explanation",
        ],
    )
    _write_csv(
        ARTIFACT_ROOT / "event_accounting_v2.csv",
        corrected_accounting,
        list(corrected_accounting[0]),
    )
    _write_json(
        ARTIFACT_ROOT / "event_accounting_reconciliation_decision_v1.json",
        {
            "outcome": "event_label_ambiguity_corrected",
            "cause": (
                "The old total counted unique severe events with at least one "
                "eligible checkpoint-6 stock episode. The old positive and "
                "negative columns counted all severe VTI sessions, including "
                "sessions with no eligible stock episode."
            ),
            "aggregation_bug_found": False,
            "session_with_both_signs_found": False,
            "prior_scientific_interpretation_changes": False,
            "original_artifact_preserved": str(
                (
                    RETROSPECTIVE_ROOT / "event_accounting_v1.csv"
                ).relative_to(ROOT)
            ),
            "corrected_artifact": "event_accounting_v2.csv",
        },
    )
    corrected_retrospective_report = """# Corrected Opening Transition Event Summary V2

This versioned correction preserves the original V1 artifacts.

The previous event-accounting labels mixed two populations:

- `unique_opening_transition_event_count` counted severe VTI events with at
  least one eligible checkpoint-6 stock episode.
- the old negative and positive columns counted every severe VTI session,
  including sessions with no eligible stock episode.

On one common eligible-event population, the corrected counts are:

| Period | Total eligible events | Negative | Positive | All-market negative | All-market positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| development | 26 | 18 | 8 | 23 | 15 |
| assessment | 43 | 23 | 20 | 28 | 30 |
| stress | 13 | 5 | 8 | 7 | 11 |

Each event has exactly one sign, and negative plus positive equals the signed
eligible-event total in every period. The reconciliation outcome is
`event_label_ambiguity_corrected`; no event-construction or aggregation bug was
found. The prior scientific interpretation remains
`blocked_insufficient_support`.
"""
    (
        REPORT_ROOT
        / "corrected_retrospective_opening_transition_event_summary_v2.md"
    ).write_text(corrected_retrospective_report, encoding="utf-8")

    _write_json(
        ARTIFACT_ROOT / "frozen_experiment_configuration_v1.json",
        config,
    )
    _write_json(
        ARTIFACT_ROOT / "frozen_rule_manifest_v1.json",
        {
            "rule": rule,
            "rule_hash": rule.rule_hash,
            "frozen_checkpoints": [
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
            ],
            "cohort": "frozen_20_stock_us_cohort",
            "bars": "five_minute",
            "opening_window_new_york": "09:30:00_inclusive_to_10:00:00_exclusive",
            "included_bar_ordinals": [0, 1, 2, 3, 4, 5],
            "final_included_bar": "09:55:00_to_10:00:00",
            "entry": "10:00_next_bar_open",
            "entry_bar_excluded_from_predictors": True,
            "material_threshold_formula": (
                "prior_close_ATM_IV*sqrt(15/(252*390))*sqrt(2/pi)"
            ),
            "threshold_equality_partition": "NO_MATERIAL_MOVE",
            "statistics": {
                "cluster_bootstrap_replications": 2000,
                "cluster_confidence_level": 0.95,
                "primary_null_replications": 1000,
                "primary_null_seed": config.primary_null_seed,
                "session_cluster_seed": config.session_cluster_seed,
                "event_cluster_seed": config.event_cluster_seed,
                "winsor_fraction": 0.01,
                "temporal_placebo": (
                    "next eligible outcome for same stock and cohort"
                ),
            },
            "descriptive_thresholds": {
                "overnight_gap_q10": DESCRIPTIVE_OVERNIGHT_GAP_Q10_V1,
                "overnight_gap_q90": DESCRIPTIVE_OVERNIGHT_GAP_Q90_V1,
                "total_transition_q10": DESCRIPTIVE_TOTAL_TRANSITION_Q10_V1,
                "total_transition_q90": DESCRIPTIVE_TOTAL_TRANSITION_Q90_V1,
            },
            "retrospective_decision": "blocked_insufficient_support",
            "claims": {
                "research_only": True,
                "shadow_only": True,
                "validated_directional_evidence": False,
                "option_profitability_claim": False,
                "orders_allowed": False,
            },
        },
    )
    _write_json(
        ARTIFACT_ROOT / "api_market_data_capacity_manifest_v1.json",
        {
            "schema_version": "api-market-data-capacity-manifest-v1",
            "configured_budget_at_activation": None,
            "deployment_template_budget": 100,
            "exact_broker_accounting_known": False,
            "accounting_policy": "conservative_local_estimate_fail_closed_optional",
            "reserved_position_monitoring_and_operational_safety_lines": 12,
            "mandatory_persistent_estimated_lines": {
                "VTI_five_minute": 1,
                "frozen_20_stock_m1c_five_minute": 20,
            },
            "mandatory_promoted_episode_estimated_lines": {
                "promoted_underlying_level1": 1,
                "primary_1dte_call": 1,
                "primary_1dte_put": 1,
            },
            "maximum_promoted_underlyings": 1,
            "full_option_chain_streaming": False,
            "level_ii_enabled": False,
            "reserve_may_fund_optional_research": False,
            "primary_option_capacity_failure": (
                "option_economics_blocked_capacity; underlying episode retained"
            ),
            "optional_capacity_failure": (
                "optional_feed_not_started_capacity_reserved"
            ),
        },
    )
    _write_json(
        ARTIFACT_ROOT / "subscription_priority_manifest_v1.json",
        {
            "schema_version": "subscription-priority-manifest-v1",
            "priority": [
                "critical_connection_session_clock_health",
                "VTI_and_required_critical_market_proxies",
                "frozen_20_stock_m1c_five_minute_bars",
                "frozen_m1c_scoring_and_episode_receipts",
                "one_promoted_episode_underlying",
                "primary_1dte_option_call_and_put",
                "optional_comparison_contracts",
                "single_promoted_underlying_tick_by_tick",
                "optional_additional_underlying_quote_detail",
                "level_ii_separate_authorisation_only",
                "neutral_controls_and_additional_expiries",
            ],
            "degradation_order": list(rule.optional_feed_degradation_order),
            "preserved_under_all_optional_exhaustion": [
                "VTI_bars",
                "frozen_20_stock_m1c_bars",
                "m1c_scoring",
                "checkpoint_receipts",
                "eligibility_receipts",
                "promoted_underlying",
                "primary_1dte_call_put_pair",
            ],
        },
    )

    for filename, columns in EMPTY_TABLES.items():
        _write_csv(ARTIFACT_ROOT / filename, [], columns)

    activation_iso = receipt.activation_timestamp_utc.isoformat()
    pending_reason = "awaiting_post_activation_engineering_transfer_sessions"
    _write_json(
        ARTIFACT_ROOT / "transfer_report_v1.json",
        {
            "status": "pending",
            "cohort_phase": "engineering_transfer",
            "valid_sessions_required": 20,
            "valid_sessions_observed": 0,
            "allowed_evaluations": [
                "bar_timing",
                "bar_semantics",
                "vti_opening_return",
                "vti_opening_range",
                "severe_state_agreement",
                "threshold_boundary_disagreement",
                "source_completeness",
                "prediction_receipt_timing",
                "subscription_capacity",
                "contract_discovery",
                "graceful_degradation",
                "recorder_reliability",
                "no_order_safeguards",
            ],
            "outcomes_opened": False,
            "reason": pending_reason,
        },
    )
    decisions = {
        "transfer_decision_receipt_v1.json": (
            "transfer_decision",
            "not_issued_pending_20_valid_sessions",
            pending_reason,
        ),
        "development_decision_receipt_v1.json": (
            "development_decision",
            "not_issued_pending_transfer_acceptance_and_support",
            "engineering transfer has not completed",
        ),
        "confirmation_start_receipt_v1.json": (
            "confirmation_start",
            "not_created",
            "development decision has not supported confirmation",
        ),
        "confirmation_decision_receipt_v1.json": (
            "confirmation_decision",
            "not_issued",
            "untouched confirmation has not started",
        ),
        "option_economics_decision_receipt_v1.json": (
            "option_economics_decision",
            "not_issued",
            "direction is not prospectively supported and no option cohort exists",
        ),
    }
    for filename, (receipt_type, status, reason) in decisions.items():
        _write_json(
            ARTIFACT_ROOT / filename,
            _pending_receipt(
                receipt_type=receipt_type,
                status=status,
                activation_timestamp=activation_iso,
                reason=reason,
            ),
        )

    summary = {
        "experiment_id": receipt.experiment_id,
        "experiment_version": receipt.experiment_version,
        "activation_timestamp_utc": activation_iso,
        "retrospective_decision": "blocked_insufficient_support",
        "retrospective_event_reconciliation": (
            "event_label_ambiguity_corrected"
        ),
        "prior_scientific_interpretation_changes": False,
        "frozen_prediction": {
            "NEGATIVE_SEVERE_OPENING_TRANSITION": "CALL",
            "POSITIVE_SEVERE_OPENING_TRANSITION": "PUT",
            "otherwise": "ABSTAIN",
            "formula": "prediction_sign_v1=-opening_transition_sign_v1",
        },
        "cohort_status": {
            "engineering_transfer_valid_sessions": 0,
            "prospective_development_complete_eligible_episodes": 0,
            "untouched_confirmation_complete_eligible_episodes": 0,
        },
        "scientific_result": "not_issued",
        "option_economics_result": "not_issued",
        "operational_blockers": [
            {
                "code": "checkpoint_6_receipt_timing_contract_conflict",
                "detail": (
                    "The final 09:55-10:00 bar is causal only when complete at "
                    "10:00, while the frozen entry is 10:00 and eligibility "
                    "requires receipt_created_at < entry_timestamp. The live "
                    "implementation fails closed; no timestamp is fabricated."
                ),
            }
        ],
        "protected_pre_activation_2026_outcomes_opened": False,
        "order_routing_enabled": False,
        "orders_placed": 0,
    }
    _write_json(ARTIFACT_ROOT / "summary_v1.json", summary)

    report = f"""# M1C Prospective Opening Reversal V1

**RESEARCH ONLY — SHADOW RECORDING ONLY — NO ORDERS**

Activation: `{activation_iso}`  
Rule hash: `{rule.rule_hash}`  
Configuration hash: `{config.configuration_hash}`

## Event accounting

The prior assessment labels mixed two populations. Its total of 43 counted
severe transition events with at least one eligible stock episode, while 28
negative and 30 positive counted all severe VTI sessions. On the common
eligible-event population, assessment is 23 negative plus 20 positive = 43.
Stress is 5 negative plus 8 positive = 13. Development is 18 negative plus
8 positive = 26. This is `event_label_ambiguity_corrected`, not an event-sign
duplication or aggregation bug. The prior scientific interpretation remains
`blocked_insufficient_support`.

## API and capacity operation

The deployment template has a 100-line local budget, but exact broker capacity
is treated as unknown until runtime. Twelve lines are reserved and unavailable
to optional research. Persistent priority protects VTI and the frozen 20-stock
five-minute universe. Only one deterministically promoted underlying and its
one 1DTE nearest-ATM call/put pair may be mandatory. Optional feeds fail closed
and are dropped in the frozen order recorded in the priority manifest. There
are no post-activation runtime snapshots or degradation events yet.

## Engineering transfer

The first 20 valid sessions are `engineering_transfer`. No future stock return,
M1C outcome, option P&L, or directional score from those sessions may influence
the rule. IBKR/EODHD comparison is predictor-only and does not require exact
OHLC equality. Zero valid sessions have been recorded.

There is one explicit timing blocker to resolve prospectively: the sixth bar
completes at 10:00, the frozen entry is also 10:00, and the receipt contract is
strictly before entry. The recorder therefore fails closed instead of inventing
an earlier timestamp.

## Prospective development

The frozen action is CALL after a negative severe VTI opening transition, PUT
after a positive severe transition, and ABSTAIN otherwise. No development
outcome has been opened. Support, baselines, cluster intervals, fixed-seed
nulls, placebo, and concentration results are pending.

## Prospective confirmation

Confirmation has not started. Its start and decision receipts remain
deliberately unissued.

## Option economics

No option-economics claim is made. Primary evidence requires actual first-valid
ask at/after entry and first-valid bid at/after +15 minutes for the frozen 1DTE
pair. Midpoints remain diagnostics. Optional expiries remain separate and
capacity-gated.

## Structural, movement, and execution evidence

No post-activation movement evidence exists. The activation opened no protected
pre-activation 2026 outcome. Order routing is disabled, no order method is
available in the experiment module, and no order was placed.
"""
    (REPORT_ROOT / "m1c_prospective_opening_reversal_v1.md").write_text(
        report,
        encoding="utf-8",
    )

    _write_json(ACTIVATION_PATH, receipt)
    load_activation_receipt_v1(str(ACTIVATION_PATH))
    load_frozen_experiment_config_v1(
        str(ARTIFACT_ROOT / "frozen_experiment_configuration_v1.json")
    )

    hashed_artifacts: dict[str, str] = {}
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if path.is_file() and path.name not in {
            "provenance_manifest_v1.json",
            "artifact_manifest_v1.json",
        }:
            published = PUBLISHED_ARTIFACT_ROOT / path.relative_to(ARTIFACT_ROOT)
            hashed_artifacts[str(published.relative_to(ROOT))] = _sha256_file(path)
    for path in sorted(REPORT_ROOT.rglob("*")):
        if path.is_file():
            published = PUBLISHED_REPORT_ROOT / path.relative_to(REPORT_ROOT)
            hashed_artifacts[str(published.relative_to(ROOT))] = _sha256_file(path)
    for static_path in (EXPERIMENT_ROOT / "README.md", Path(__file__).resolve()):
        hashed_artifacts[str(static_path.relative_to(ROOT))] = _sha256_file(
            static_path
        )
    provenance = {
        "schema_version": "m1c-opening-reversal-provenance-v1",
        "branch": branch,
        "commit": commit,
        "dirty_working_tree_status_at_activation": dirty_status,
        "activation_timestamp_utc": activation_iso,
        "confirmation_start_timestamp_utc": None,
        "configuration_hash": config.configuration_hash,
        "rule_hash": rule.rule_hash,
        "vti_thresholds": {
            "opening_return_q10": NEGATIVE_OPENING_RETURN_THRESHOLD_V1,
            "opening_return_q90": POSITIVE_OPENING_RETURN_THRESHOLD_V1,
            "opening_range_q75": OPENING_RANGE_THRESHOLD_V1,
        },
        "m1c_version": receipt.m1c_version,
        "tail_phase_version": receipt.tail_phase_version,
        "a1_version": receipt.a1_version,
        "contract_selection_version": receipt.option_selection_version,
        "capacity_manager_version": receipt.capacity_manager_version,
        "configured_line_budget_at_activation": None,
        "deployment_template_line_budget": 100,
        "reserved_line_count": 12,
        "mandatory_base_feed_estimate": 21,
        "optional_feed_counts": {},
        "data_sources": [
            "IBKR_live_recording_post_activation",
            "EODHD_after_session_predictor_transfer",
        ],
        "transfer_status": "pending",
        "episode_counts": {},
        "event_counts": {},
        "missingness": {},
        "capacity_denials": 0,
        "contract_discovery_failures": 0,
        "commands": [
            (
                "rtk uv run python research/prospective/"
                "20260729-m1c-prospective-opening-reversal-v1/"
                "build_activation_artifacts.py --activate"
            ),
            (
                "rtk proxy .venv/bin/pytest -q "
                "tests/test_frozen_m1c_recorder_v0.py "
                "tests/test_ibkr_budget_aware_shadow_v0.py "
                "tests/test_m1c_event_ingest_live_recorder_v0.py "
                "tests/test_m1c_opening_market_transition_v1.py "
                "tests/test_m1c_opening_market_transition_v1_recorder.py "
                "tests/test_m1c_subscriptions_options_v0.py "
                "tests/test_m1c_tail_phase_v1.py "
                "tests/test_quiet_state_options_shadow_v0.py "
                "tests/test_m1c_prospective_opening_reversal_v1.py "
                "tests/test_m1c_opening_reversal_capacity_v1.py "
                "tests/test_m1c_opening_reversal_analysis_v1.py "
                "tests/test_m1c_opening_reversal_recorder_v1.py "
                "tests/test_m1c_opening_reversal_option_discovery_v1.py"
            ),
            (
                "rtk proxy .venv/bin/ruff check "
                "packages/stocker_prospective/src/stocker_prospective/"
                "{config,frozen_live_application,live_recorder,option_budget,"
                "option_discovery,option_ledger,parallel,recorder_repository,"
                "recorder_v0,source_transfer,m1c_opening_reversal_analysis_v1,"
                "m1c_prospective_opening_reversal_v1}.py "
                "research/directional-readiness/"
                "20260728-m1c-opening-market-transition-v1/run_experiment.py "
                "research/prospective/"
                "20260729-m1c-prospective-opening-reversal-v1/"
                "build_activation_artifacts.py tests/test_m1c_opening_reversal_*.py "
                "tests/test_m1c_prospective_opening_reversal_v1.py"
            ),
            (
                "rtk proxy .venv/bin/mypy "
                "packages/stocker_prospective/src/stocker_prospective/"
                "{config,frozen_live_application,live_recorder,option_budget,"
                "option_discovery,option_ledger,parallel,recorder_repository,"
                "recorder_v0,source_transfer,m1c_opening_reversal_analysis_v1,"
                "m1c_prospective_opening_reversal_v1}.py"
            ),
            "rtk git diff --check",
            (
                "rtk uv run python research/prospective/"
                "20260729-m1c-prospective-opening-reversal-v1/"
                "build_activation_artifacts.py --verify"
            ),
        ],
        "seeds": {
            "primary_null": config.primary_null_seed,
            "session_cluster": config.session_cluster_seed,
            "event_cluster": config.event_cluster_seed,
        },
        "protected_pre_activation_2026_outcomes_opened": False,
        "broker_order_routing_enabled": False,
        "orders_placed": 0,
        "artifact_sha256": hashed_artifacts,
    }
    provenance["provenance_hash"] = _sha256_value(provenance)
    _write_json(ARTIFACT_ROOT / "provenance_manifest_v1.json", provenance)
    hashed_artifacts[
        str(
            (
                PUBLISHED_ARTIFACT_ROOT / "provenance_manifest_v1.json"
            ).relative_to(ROOT)
        )
    ] = _sha256_file(ARTIFACT_ROOT / "provenance_manifest_v1.json")
    _write_json(
        ARTIFACT_ROOT / "artifact_manifest_v1.json",
        {
            "schema_version": "m1c-opening-reversal-artifact-manifest-v1",
            "activation_timestamp_utc": activation_iso,
            "artifact_count": len(hashed_artifacts),
            "artifacts": hashed_artifacts,
        },
    )


def _manifest_path_in_package(
    relative_path: str,
    *,
    artifact_root: Path,
    report_root: Path,
) -> Path:
    published = ROOT / relative_path
    if published.is_relative_to(PUBLISHED_ARTIFACT_ROOT):
        return artifact_root / published.relative_to(PUBLISHED_ARTIFACT_ROOT)
    if published.is_relative_to(PUBLISHED_REPORT_ROOT):
        return report_root / published.relative_to(PUBLISHED_REPORT_ROOT)
    return published


def _verify_package(
    *,
    artifact_root: Path,
    report_root: Path,
    emit: bool,
) -> dict[str, Any]:
    required_artifact_names = {
        *EMPTY_TABLES,
        "opening_transition_event_count_reconciliation_v1.csv",
        "event_accounting_v2.csv",
        "event_accounting_reconciliation_decision_v1.json",
        "frozen_experiment_configuration_v1.json",
        "frozen_rule_manifest_v1.json",
        "api_market_data_capacity_manifest_v1.json",
        "subscription_priority_manifest_v1.json",
        "transfer_report_v1.json",
        "transfer_decision_receipt_v1.json",
        "development_decision_receipt_v1.json",
        "confirmation_start_receipt_v1.json",
        "confirmation_decision_receipt_v1.json",
        "option_economics_decision_receipt_v1.json",
        "summary_v1.json",
        "provenance_manifest_v1.json",
        "artifact_manifest_v1.json",
        "experiment_activation_receipt_v1.json",
    }
    actual_artifact_names = {
        str(path.relative_to(artifact_root))
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    if actual_artifact_names != required_artifact_names:
        raise RuntimeError("activation_required_artifact_set_mismatch")
    required_report_names = {
        "corrected_retrospective_opening_transition_event_summary_v2.md",
        "m1c_prospective_opening_reversal_v1.md",
    }
    actual_report_names = {
        str(path.relative_to(report_root))
        for path in report_root.rglob("*")
        if path.is_file()
    }
    if actual_report_names != required_report_names:
        raise RuntimeError("activation_required_report_set_mismatch")
    activation_path = artifact_root / "experiment_activation_receipt_v1.json"
    receipt = load_activation_receipt_v1(str(activation_path))
    config = load_frozen_experiment_config_v1(
        str(artifact_root / "frozen_experiment_configuration_v1.json")
    )
    if receipt.configuration_hash != config.configuration_hash:
        raise RuntimeError("activation_configuration_hash_mismatch")
    rule_manifest = json.loads(
        (artifact_root / "frozen_rule_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    rule_hash = FrozenOpeningReversalRuleV1().rule_hash
    if (
        rule_manifest.get("rule_hash") != rule_hash
        or receipt.frozen_rule_hash != rule_hash
    ):
        raise RuntimeError("activation_rule_hash_mismatch")
    reconciliation = pd.read_csv(
        artifact_root / "opening_transition_event_count_reconciliation_v1.csv"
    )
    if bool(
        (
            reconciliation["included_in_positive_count"].astype(bool)
            & reconciliation["included_in_negative_count"].astype(bool)
        ).any()
    ):
        raise RuntimeError("one_event_received_two_transition_signs")
    included = reconciliation[
        reconciliation["included_in_total_event_count"].astype(bool)
    ]
    counts = {
        period: (
            len(group),
            int((group["opening_sign"] == -1).sum()),
            int((group["opening_sign"] == 1).sum()),
        )
        for period, group in included.groupby("period")
    }
    if counts != {
        "assessment": (43, 23, 20),
        "development": (26, 18, 8),
        "stress": (13, 5, 8),
    }:
        raise RuntimeError("event_reconciliation_count_mismatch")

    provenance_path = artifact_root / "provenance_manifest_v1.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    stored_provenance_hash = provenance.pop("provenance_hash", None)
    if stored_provenance_hash != _sha256_value(provenance):
        raise RuntimeError("provenance_manifest_hash_mismatch")
    if (
        provenance.get("protected_pre_activation_2026_outcomes_opened") is not False
        or provenance.get("broker_order_routing_enabled") is not False
        or provenance.get("orders_placed") != 0
    ):
        raise RuntimeError("provenance_safety_claim_mismatch")

    manifest = json.loads(
        (artifact_root / "artifact_manifest_v1.json").read_text(encoding="utf-8")
    )
    artifacts = cast(dict[str, str], manifest.get("artifacts"))
    if not isinstance(artifacts, dict) or manifest.get("artifact_count") != len(
        artifacts
    ):
        raise RuntimeError("artifact_manifest_count_mismatch")
    actual_keys: set[str] = {
        str(
            (PUBLISHED_ARTIFACT_ROOT / path.relative_to(artifact_root)).relative_to(
                ROOT
            )
        )
        for path in artifact_root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest_v1.json"
    }
    actual_keys.update(
        str(
            (PUBLISHED_REPORT_ROOT / path.relative_to(report_root)).relative_to(
                ROOT
            )
        )
        for path in report_root.rglob("*")
        if path.is_file()
    )
    actual_keys.update(
        str(path.relative_to(ROOT))
        for path in (EXPERIMENT_ROOT / "README.md", Path(__file__).resolve())
    )
    if set(artifacts) != actual_keys:
        raise RuntimeError("artifact_manifest_file_set_mismatch")
    for relative_path, expected_hash in artifacts.items():
        path = _manifest_path_in_package(
            relative_path,
            artifact_root=artifact_root,
            report_root=report_root,
        )
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"artifact_hash_mismatch:{relative_path}")

    result = {
        "activation_timestamp_utc": receipt.activation_timestamp_utc.isoformat(),
        "activation_receipt_hash": receipt.activation_receipt_hash,
        "configuration_hash": config.configuration_hash,
        "rule_hash": rule_hash,
        "event_rows": len(reconciliation),
        "artifact_count": len(artifacts),
        "status": "verified",
    }
    if emit:
        print(json.dumps(result, sort_keys=True))
    return result


def _publish_staged_package(
    *,
    staged_artifact_root: Path,
    staged_report_root: Path,
    published_artifact_root: Path,
    published_report_root: Path,
    before_artifact_publish: Callable[[], None] | None = None,
) -> None:
    """Expose the fully verified artifact directory as the final atomic step."""

    if published_artifact_root.exists() and (
        published_artifact_root.is_symlink()
        or not published_artifact_root.is_dir()
        or any(published_artifact_root.iterdir())
    ):
        raise RuntimeError("activation artifact package already exists")
    published_report_root.mkdir(parents=True, exist_ok=True)
    for source in sorted(staged_report_root.rglob("*")):
        if source.is_file():
            target = published_report_root / source.relative_to(staged_report_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    if before_artifact_publish is not None:
        before_artifact_publish()
    published_artifact_root.parent.mkdir(parents=True, exist_ok=True)
    if published_artifact_root.exists():
        # The repository carries this task-owned empty placeholder. Removing
        # it exposes no receipt; rmdir also fails closed if it was populated
        # after the availability check.
        published_artifact_root.rmdir()
    os.replace(staged_artifact_root, published_artifact_root)


def activate() -> None:
    """Stage, verify, then atomically expose the immutable receipt last."""

    global ACTIVATION_PATH, ARTIFACT_ROOT, REPORT_ROOT, _GIT_STATE_OVERRIDE

    if PUBLISHED_ARTIFACT_ROOT.exists() and (
        PUBLISHED_ARTIFACT_ROOT.is_symlink()
        or not PUBLISHED_ARTIFACT_ROOT.is_dir()
        or any(PUBLISHED_ARTIFACT_ROOT.iterdir())
    ):
        raise RuntimeError(
            "immutable activation package already exists; refusing overwrite"
        )
    original_artifact_root = ARTIFACT_ROOT
    original_report_root = REPORT_ROOT
    original_activation_path = ACTIVATION_PATH
    original_git_state_override = _GIT_STATE_OVERRIDE
    _GIT_STATE_OVERRIDE = (
        _git("branch", "--show-current"),
        _git("rev-parse", "HEAD"),
        _git("status", "--short") or "clean",
    )
    with tempfile.TemporaryDirectory(
        prefix=".m1c-opening-reversal-activation-",
        dir=EXPERIMENT_ROOT,
    ) as temporary:
        staging_root = Path(temporary)
        staged_artifact_root = staging_root / "artifacts" / "primary"
        staged_report_root = staging_root / "reports"
        ARTIFACT_ROOT = staged_artifact_root
        REPORT_ROOT = staged_report_root
        ACTIVATION_PATH = (
            staged_artifact_root / "experiment_activation_receipt_v1.json"
        )
        try:
            _build_activation_package()
            _verify_package(
                artifact_root=staged_artifact_root,
                report_root=staged_report_root,
                emit=False,
            )
            _publish_staged_package(
                staged_artifact_root=staged_artifact_root,
                staged_report_root=staged_report_root,
                published_artifact_root=PUBLISHED_ARTIFACT_ROOT,
                published_report_root=PUBLISHED_REPORT_ROOT,
            )
        finally:
            ARTIFACT_ROOT = original_artifact_root
            REPORT_ROOT = original_report_root
            ACTIVATION_PATH = original_activation_path
            _GIT_STATE_OVERRIDE = original_git_state_override


def verify() -> None:
    _verify_package(
        artifact_root=PUBLISHED_ARTIFACT_ROOT,
        report_root=PUBLISHED_REPORT_ROOT,
        emit=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--activate", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.activate:
        activate()
    else:
        verify()


if __name__ == "__main__":
    main()
