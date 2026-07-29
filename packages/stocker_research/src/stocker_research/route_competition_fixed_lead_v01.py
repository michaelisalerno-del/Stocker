"""Pure helpers for the Route-Competition Fixed-Lead Audit Quick Screen V0.1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd


def earliest_completion_lead(checkpoint: int, completion_ordinals: Sequence[int]) -> int:
    """Return the earliest strict completion lead in the fixed three-bar horizon."""

    future = sorted(
        {
            int(value)
            for value in completion_ordinals
            if int(checkpoint) < int(value) <= int(checkpoint) + 3
        }
    )
    return 0 if not future else int(future[0] - int(checkpoint))


def remaining_required_transitions(
    *,
    progress_states: int,
    canonical_oriented_path: Sequence[int],
    motif_type: str,
    declared_transitions_remaining: int | None = None,
) -> int:
    """Derive remaining transitions from an independently supplied canonical route."""

    if motif_type not in {"primitive", "repeat", "composite"}:
        raise ValueError(f"unknown registered motif type: {motif_type}")
    progress = int(progress_states)
    route = tuple(int(state) for state in canonical_oriented_path)
    if len(route) < 3 or route[0] != route[-1]:
        raise ValueError("canonical registered route must be a closed path")
    total_required = len(route) - 1
    completed = progress - 1
    remaining = total_required - completed
    if progress < 1 or remaining < 0:
        raise ValueError("canonical prefix progress is outside the registered route")
    if (
        declared_transitions_remaining is not None
        and int(declared_transitions_remaining) != remaining
    ):
        raise ValueError("declared prefix remainder differs from canonical route length")
    return remaining


def prefix_proximity(
    prefix_ledger: pd.DataFrame,
    *,
    checkpoint: int,
    canonical_oriented_paths: Mapping[tuple[str, str], Sequence[int]],
) -> dict[str, int | float]:
    """Summarise canonical registered-prefix proximity at one completed-bar checkpoint."""

    required = {
        "bar_ordinal",
        "semantic_loop_id",
        "orientation_id",
        "motif_type",
        "progress_states",
        "transitions_remaining",
    }
    missing = sorted(required.difference(prefix_ledger.columns))
    if missing:
        raise ValueError(f"prefix ledger missing columns: {missing}")
    current = (
        prefix_ledger.loc[prefix_ledger["bar_ordinal"].astype(int).eq(int(checkpoint))]
        .drop_duplicates(["semantic_loop_id", "orientation_id"])
        .copy()
    )
    remaining = [
        remaining_required_transitions(
            progress_states=int(cast(Any, row.progress_states)),
            canonical_oriented_path=canonical_oriented_paths[
                (str(row.semantic_loop_id), str(row.orientation_id))
            ],
            motif_type=str(row.motif_type),
            declared_transitions_remaining=int(cast(Any, row.transitions_remaining)),
        )
        for row in current.itertuples(index=False)
    ]
    one_away = sum(value == 1 for value in remaining)
    return {
        "any_prefix_one_transition_from_completion": int(one_away > 0),
        "minimum_remaining_transitions": (float(min(remaining)) if remaining else float("nan")),
        "number_of_one_transition_away_prefixes": int(one_away),
    }


def fixed_lead_labels(
    *,
    first_completion_lead: int,
    any_prefix_one_transition_from_completion: int,
) -> dict[str, int]:
    """Construct the two fixed targets and the non-imminent advance population flag."""

    lead = int(first_completion_lead)
    near_complete = int(any_prefix_one_transition_from_completion)
    if lead not in {0, 1, 2, 3} or near_complete not in {0, 1}:
        raise ValueError("fixed-lead inputs are outside their registered domains")
    return {
        "completion_next_1_bar": int(lead == 1),
        "completion_in_bars_2_or_3": int(lead in {2, 3}),
        "advance_eligible": int(lead != 1 and near_complete == 0),
    }


def predecessor_surface_differences(
    reference: pd.DataFrame,
    reconstructed: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> dict[str, int | float]:
    """Compare the complete frozen predecessor surface after row-id alignment."""

    fixed_columns = {
        "row_id",
        "checkpoint_timestamp_utc",
        "period",
        "row_weight",
        "registered_completion_next_3_bars",
        "H0_probability",
        "H1_probability",
        *feature_columns,
    }
    for name, frame in (("reference", reference), ("reconstructed", reconstructed)):
        missing = sorted(fixed_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} predecessor surface missing columns: {missing}")
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
            "split_mismatches": int(row_mismatches),
            "target_mismatches": int(row_mismatches),
            "maximum_weight_difference": float("inf"),
            "maximum_feature_difference": float("inf"),
            "maximum_probability_difference": float("inf"),
        }
    checkpoint_mismatches = int(
        (
            pd.to_datetime(left["checkpoint_timestamp_utc"], utc=True)
            != pd.to_datetime(right["checkpoint_timestamp_utc"], utc=True)
        ).sum()
    )
    split_mismatches = int((left["period"].astype(str) != right["period"].astype(str)).sum())
    target_mismatches = int(
        (
            left["registered_completion_next_3_bars"].to_numpy(int)
            != right["registered_completion_next_3_bars"].to_numpy(int)
        ).sum()
    )
    return {
        "row_identity_mismatches": 0,
        "checkpoint_timestamp_mismatches": checkpoint_mismatches,
        "split_mismatches": split_mismatches,
        "target_mismatches": target_mismatches,
        "maximum_weight_difference": float(
            np.max(np.abs(left["row_weight"].to_numpy(float) - right["row_weight"].to_numpy(float)))
        ),
        "maximum_feature_difference": float(
            np.max(
                np.abs(
                    left.loc[:, list(feature_columns)].to_numpy(float)
                    - right.loc[:, list(feature_columns)].to_numpy(float)
                )
            )
        ),
        "maximum_probability_difference": float(
            np.max(
                np.abs(
                    left.loc[:, ["H0_probability", "H1_probability"]].to_numpy(float)
                    - right.loc[:, ["H0_probability", "H1_probability"]].to_numpy(float)
                )
            )
        ),
    }


def theoretical_assessment_support(
    *, sessions: int, stocks: int, checkpoints: int, retained_rows: int
) -> dict[str, int | float]:
    """Calculate the corrected fixed-slate theoretical assessment support."""

    theoretical = int(sessions) * int(stocks) * int(checkpoints)
    retained = int(retained_rows)
    if theoretical <= 0 or retained < 0 or retained > theoretical:
        raise ValueError("theoretical assessment support is invalid")
    return {
        "theoretical_eligible_rows": theoretical,
        "retained_rows": retained,
        "retention": float(retained / theoretical),
    }


def fixed_lead_increment_passes(
    gates: Mapping[str, object], *, require_average_precision: bool
) -> bool:
    """Apply the preregistered immediate or advance proper-score gates."""

    required = {
        "log_loss_improvement",
        "brier_improvement",
        "auc_improvement",
        "bootstrap_80_log_loss_lower",
        "bootstrap_80_brier_lower",
        "positive_months",
        "materially_adverse_checkpoints",
        "real_exceeds_all_nulls",
        "support_and_concentration_passed",
    }
    if require_average_precision:
        required.update(
            {
                "average_precision_improvement",
                "bootstrap_80_average_precision_lower",
            }
        )
    missing = sorted(required.difference(gates))
    if missing:
        raise ValueError(f"fixed-lead increment gates missing: {missing}")
    passed = bool(
        float(cast(Any, gates["log_loss_improvement"])) > 0.0
        and float(cast(Any, gates["brier_improvement"])) > 0.0
        and float(cast(Any, gates["auc_improvement"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_log_loss_lower"])) >= 0.0
        and float(cast(Any, gates["bootstrap_80_brier_lower"])) >= 0.0
        and int(cast(Any, gates["positive_months"])) >= 5
        and int(cast(Any, gates["materially_adverse_checkpoints"])) == 0
        and bool(gates["real_exceeds_all_nulls"])
        and bool(gates["support_and_concentration_passed"])
    )
    if require_average_precision:
        passed = bool(
            passed
            and float(cast(Any, gates["average_precision_improvement"])) > 0.0
            and float(cast(Any, gates["bootstrap_80_average_precision_lower"])) >= 0.0
        )
    return passed


def choose_fixed_lead_decision(
    *,
    blocker: str | None,
    immediate_passed: bool,
    advance_passed: bool,
    descriptive_lead_structure: bool,
    baseline_meaningful: bool,
) -> str:
    """Choose exactly one fixed-lead decision, with preregistered blockers first."""

    blockers = {
        "blocked_predecessor_reconstruction_failure",
        "blocked_prefix_proximity_reconstruction_failure",
        "blocked_insufficient_immediate_support",
        "blocked_insufficient_advance_support",
        "blocked_insufficient_advance_positive_support",
        "blocked_protected_boundary_failure",
        "blocked_chronology_or_leakage_failure",
        "blocked_quick_fixed_lead_resource_limit",
        "blocked_model_convergence_failure",
        "blocked_reproducibility_or_audit_failure",
    }
    if blocker is not None:
        if blocker not in blockers:
            raise ValueError(f"unknown fixed-lead blocker: {blocker}")
        return blocker
    if immediate_passed and advance_passed:
        return "route_competition_adds_immediate_and_advance_warning"
    if advance_passed:
        return "route_competition_adds_advance_warning"
    if immediate_passed:
        return "route_competition_is_imminent_confirmation_only"
    if descriptive_lead_structure:
        return "descriptive_fixed_lead_structure_only"
    if baseline_meaningful:
        return "compressed_transition_baseline_only"
    return "no_fixed_lead_route_increment"
