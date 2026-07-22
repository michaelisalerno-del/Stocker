"""Pure helpers for the Broad-Conflict Advance-Hazard Quick Screen V0.2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from stocker_research.route_competition_fixed_lead_v01 import (
    earliest_completion_lead,
    fixed_lead_labels,
    prefix_proximity,
    remaining_required_transitions,
)
from stocker_research.route_competition_hazard_v0 import (
    BASELINE_FEATURES as PREDECESSOR_H0_FEATURES,
)
from stocker_research.route_competition_hazard_v0 import (
    ROUTE_FEATURES,
    assign_route_resolution_state,
    permute_route_bundle,
    session_bootstrap_multiplicities,
)

DENSE_CHECKPOINTS: Final[tuple[int, ...]] = (
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
)
BASELINE_NON_CLOCK_FEATURES: Final[tuple[str, ...]] = tuple(
    feature for feature in PREDECESSOR_H0_FEATURES if not feature.startswith("checkpoint_")
)
DENSE_H0_FEATURES: Final[tuple[str, ...]] = (
    *BASELINE_NON_CLOCK_FEATURES,
    *(f"checkpoint_{checkpoint}" for checkpoint in DENSE_CHECKPOINTS),
)
DENSE_H1_FEATURES: Final[tuple[str, ...]] = (*DENSE_H0_FEATURES, *ROUTE_FEATURES)


def theoretical_raw_population(
    *, eligible_sessions: int, stocks: int = 20, checkpoints: int = len(DENSE_CHECKPOINTS)
) -> int:
    """Return the raw stock-session-checkpoint maximum before advance exclusions."""

    if eligible_sessions < 1 or stocks < 1 or checkpoints < 1:
        raise ValueError("raw checkpoint support dimensions must be positive")
    return int(eligible_sessions) * int(stocks) * int(checkpoints)


def candidate_normalized_weights(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every advance-eligible stock-session equal influence within its session."""

    required = {"period", "session", "symbol", "advance_eligible"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate-weight panel missing columns: {missing}")
    output = frame.copy()
    output["eligible_stocks_in_session"] = np.nan
    output["eligible_advance_rows_for_stock_session"] = np.nan
    output["sequential_row_weight"] = np.nan
    eligible = output["advance_eligible"].astype(int).eq(1)
    if not bool(eligible.any()):
        raise ValueError("candidate-weight panel has no advance-eligible rows")
    subset = output.loc[eligible]
    stock_counts = subset.groupby(["period", "session"], sort=False)["symbol"].transform("nunique")
    row_counts = subset.groupby(["period", "session", "symbol"], sort=False)["symbol"].transform(
        "size"
    )
    weights = 1.0 / (stock_counts.astype(float) * row_counts.astype(float))
    if not np.isfinite(weights.to_numpy(float)).all() or bool(weights.le(0.0).any()):
        raise ValueError("candidate-normalized weights must be finite and positive")
    output.loc[eligible, "eligible_stocks_in_session"] = stock_counts.to_numpy(int)
    output.loc[eligible, "eligible_advance_rows_for_stock_session"] = row_counts.to_numpy(int)
    output.loc[eligible, "sequential_row_weight"] = weights.to_numpy(float)
    output["row_weight"] = output["sequential_row_weight"]
    return output


def predecessor_surface_differences(
    reference: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> dict[str, int | float]:
    """Compare shared-checkpoint causal identities, labels, targets, and features."""

    fixed = {
        "row_id",
        "checkpoint_timestamp_utc",
        "route_resolution_state",
        "registered_completion_next_3_bars",
        *feature_columns,
    }
    for name, frame in (("reference", reference), ("reconstructed", reconstructed)):
        missing = sorted(fixed.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} shared surface missing columns: {missing}")
    left = reference.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    right = reconstructed.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    row_mismatches = abs(len(left) - len(right)) + sum(
        first != second
        for first, second in zip(
            left["row_id"].astype(str), right["row_id"].astype(str), strict=False
        )
    )
    if row_mismatches:
        return {
            "row_identity_mismatches": int(row_mismatches),
            "checkpoint_timestamp_mismatches": int(row_mismatches),
            "target_mismatches": int(row_mismatches),
            "route_resolution_label_mismatches": int(row_mismatches),
            "maximum_shared_feature_difference": float("inf"),
        }
    return {
        "row_identity_mismatches": 0,
        "checkpoint_timestamp_mismatches": int(
            (
                pd.to_datetime(left["checkpoint_timestamp_utc"], utc=True)
                != pd.to_datetime(right["checkpoint_timestamp_utc"], utc=True)
            ).sum()
        ),
        "target_mismatches": int(
            (
                left["registered_completion_next_3_bars"].to_numpy(int)
                != right["registered_completion_next_3_bars"].to_numpy(int)
            ).sum()
        ),
        "route_resolution_label_mismatches": int(
            (
                left["route_resolution_state"].astype(str)
                != right["route_resolution_state"].astype(str)
            ).sum()
        ),
        "maximum_shared_feature_difference": float(
            np.max(
                np.abs(
                    left.loc[:, list(feature_columns)].to_numpy(float)
                    - right.loc[:, list(feature_columns)].to_numpy(float)
                )
            )
        ),
    }


def assign_frozen_route_states(
    frame: pd.DataFrame, thresholds: Mapping[str, Sequence[float]]
) -> pd.Series:
    """Apply the predecessor's frozen route-resolution thresholds unchanged."""

    return assign_route_resolution_state(frame, thresholds)


def route_bundle_permutation(
    frame: pd.DataFrame,
    *,
    route_features: Sequence[str],
    strata: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Permute one intact route bundle within each causal stock slate."""

    return permute_route_bundle(frame, route_features=route_features, strata=strata, seed=seed)


def advance_increment_passes(gates: Mapping[str, object]) -> bool:
    """Apply the preregistered clean advance A1-versus-A0 gate."""

    required = {
        "log_loss_improvement",
        "brier_improvement",
        "auc_improvement",
        "average_precision_improvement",
        "bootstrap_80_log_loss_lower",
        "bootstrap_80_brier_lower",
        "bootstrap_80_average_precision_lower",
        "positive_months",
        "materially_adverse_checkpoint_groups",
        "real_exceeds_all_nulls",
        "support_and_concentration_passed",
    }
    missing = sorted(required.difference(gates))
    if missing:
        raise ValueError(f"advance increment gates missing: {missing}")
    return bool(
        float(cast(Any, gates["log_loss_improvement"])) > 0.0
        and float(cast(Any, gates["brier_improvement"])) > 0.0
        and float(cast(Any, gates["auc_improvement"])) >= 0.0
        and float(cast(Any, gates["average_precision_improvement"])) > 0.0
        and float(cast(Any, gates["bootstrap_80_log_loss_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_brier_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_average_precision_lower"])) >= 0.0
        and int(cast(Any, gates["positive_months"])) >= 5
        and int(cast(Any, gates["materially_adverse_checkpoint_groups"])) == 0
        and bool(gates["real_exceeds_all_nulls"])
        and bool(gates["support_and_concentration_passed"])
    )


def broad_conflict_mechanism_passes(gates: Mapping[str, object]) -> bool:
    """Apply the preregistered frozen BROAD_CONFLICT mechanism gate."""

    required = {
        "assessment_minus_pooled",
        "assessment_minus_low_route_support",
        "bootstrap_80_pooled_difference_lower",
        "bootstrap_80_low_route_difference_lower",
        "development_minus_pooled",
        "positive_assessment_months",
        "materially_adverse_checkpoint_groups",
        "assessment_rows",
        "assessment_positives",
        "assessment_sessions",
        "assessment_stocks",
    }
    missing = sorted(required.difference(gates))
    if missing:
        raise ValueError(f"broad-conflict gates missing: {missing}")
    return bool(
        float(cast(Any, gates["assessment_minus_pooled"])) > 0.0
        and float(cast(Any, gates["assessment_minus_low_route_support"])) > 0.0
        and float(cast(Any, gates["bootstrap_80_pooled_difference_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_low_route_difference_lower"])) >= 0.0
        and float(cast(Any, gates["development_minus_pooled"])) > 0.0
        and int(cast(Any, gates["positive_assessment_months"])) >= 5
        and int(cast(Any, gates["materially_adverse_checkpoint_groups"])) == 0
        and int(cast(Any, gates["assessment_rows"])) >= 3_000
        and int(cast(Any, gates["assessment_positives"])) >= 100
        and int(cast(Any, gates["assessment_sessions"])) >= 100
        and int(cast(Any, gates["assessment_stocks"])) >= 15
    )


def choose_broad_conflict_decision(
    *,
    blocker: str | None,
    advance_passed: bool,
    broad_conflict_passed: bool,
    broad_conflict_descriptively_enriched: bool,
    baseline_meaningful: bool,
) -> str:
    """Choose exactly one preregistered dense advance decision."""

    blockers = {
        "blocked_predecessor_reconstruction_failure",
        "blocked_prefix_proximity_reconstruction_failure",
        "blocked_insufficient_raw_checkpoint_support",
        "blocked_insufficient_dense_advance_support",
        "blocked_insufficient_dense_advance_positive_support",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_quick_broad_conflict_resource_limit",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
    if blocker is not None:
        if blocker not in blockers:
            raise ValueError(f"unknown broad-conflict blocker: {blocker}")
        return blocker
    if advance_passed and broad_conflict_passed:
        return "broad_route_conflict_adds_clean_advance_warning"
    if advance_passed:
        return "route_competition_adds_clean_advance_warning_without_state_specificity"
    if broad_conflict_descriptively_enriched:
        return "descriptive_broad_conflict_structure_only"
    if baseline_meaningful:
        return "compressed_transition_baseline_only"
    return "no_clean_advance_route_increment"


__all__ = [
    "DENSE_H0_FEATURES",
    "DENSE_H1_FEATURES",
    "DENSE_CHECKPOINTS",
    "ROUTE_FEATURES",
    "advance_increment_passes",
    "assign_frozen_route_states",
    "broad_conflict_mechanism_passes",
    "candidate_normalized_weights",
    "choose_broad_conflict_decision",
    "earliest_completion_lead",
    "fixed_lead_labels",
    "prefix_proximity",
    "predecessor_surface_differences",
    "remaining_required_transitions",
    "route_bundle_permutation",
    "session_bootstrap_multiplicities",
    "theoretical_raw_population",
]
