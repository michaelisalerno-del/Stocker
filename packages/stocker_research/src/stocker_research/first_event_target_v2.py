"""Frozen mutually exclusive first primitive-loop target construction.

Repeat traversals and composite motifs remain auxiliary metadata.  This module
accepts structural event evidence only and has no economic or execution inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from stocker_research.semantic_loop_dictionary_v2 import safety_flags

NON_LOOP_PRIMARY_CLASSES = frozenset(
    {
        "NO_LOOP_WITHIN_HORIZON",
        "SESSION_END",
        "DISTINCT_PRIMITIVE_TIE",
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
    }
)


@dataclass(frozen=True, slots=True)
class FirstEventTargetBundle:
    """Primary mutually exclusive outcomes and separately scoped metadata."""

    outcomes: pd.DataFrame
    tie_details: pd.DataFrame
    auxiliary: pd.DataFrame
    target_contract: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScientificDecisionMetrics:
    """Complete, validated structural evidence required by the frozen gate."""

    dictionary_size: int
    development_coverage: float
    validation_coverage: float
    entries_rate_ratio_above_one_share: float
    entries_threshold_retained_share: float
    top_stock_share: float
    genuine_tie_rate: float
    other_dominated_by_obvious_candidate: bool
    semantic_ids_stable: bool
    exact_dictionary_stable_and_informative: bool
    other_is_diffuse: bool
    family_reduces_residual_entropy: bool
    family_coverage_stable_and_higher: bool
    exact_excess_consistent: bool
    coverage_collapsed: bool
    structural_excess_reversed: bool
    blocked: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ScientificDecisionMetrics:
        expected = frozenset(cls.__dataclass_fields__)
        supplied = frozenset(str(key) for key in values)
        if missing := sorted(expected.difference(supplied)):
            raise ValueError(f"scientific decision metrics are missing: {missing}")
        if unknown := sorted(supplied.difference(expected)):
            raise ValueError(f"scientific decision metrics are unknown: {unknown}")
        return cls(**{key: values[key] for key in expected})

    def __post_init__(self) -> None:
        if self.dictionary_size < 0:
            raise ValueError("dictionary size cannot be negative")
        proportions = (
            self.development_coverage,
            self.validation_coverage,
            self.entries_rate_ratio_above_one_share,
            self.entries_threshold_retained_share,
            self.top_stock_share,
            self.genuine_tie_rate,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in proportions):
            raise ValueError("scientific decision proportions must lie in [0, 1]")


def _length_family(transition_length: int) -> str:
    if transition_length == 2:
        return "TWO_STATE_OSCILLATION"
    if transition_length == 3:
        return "THREE_STATE_CYCLE"
    if transition_length == 4:
        return "FOUR_STATE_CYCLE"
    if transition_length in {5, 6}:
        return "FIVE_TO_SIX_STATE_CYCLE"
    if transition_length > 6:
        return "LONG_PRIMITIVE_CYCLE"
    raise ValueError("primitive transition length must be at least two")


def build_loop_family_mapping(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Map mutually exclusive outcomes to a deterministic topology-only family."""

    required = {"decision_id", "primary_class", "primitive_loop_id"}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"family mapping missing fields: {sorted(missing)}")
    if outcomes["decision_id"].duplicated().any():
        raise ValueError("family mapping requires one primary outcome per decision")
    records: list[dict[str, Any]] = []
    for _, row in outcomes.iterrows():
        primary = str(row["primary_class"])
        primitive = row["primitive_loop_id"]
        if primary == "NO_LOOP_WITHIN_HORIZON":
            family = "NO_LOOP"
        elif primary == "SESSION_END":
            family = "SESSION_END"
        elif primary == "DISTINCT_PRIMITIVE_TIE":
            family = "DISTINCT_PRIMITIVE_TIE"
        elif primary in {"UNAVAILABLE_SOURCE", "UNAVAILABLE_STRUCTURAL_GAP"}:
            family = primary
        elif isinstance(primitive, str) and primitive.startswith("loop_p_"):
            raw_length = row.get("primitive_transition_length")
            if pd.isna(raw_length):
                raw_length = len(primitive.removeprefix("loop_p_").split("-")) - 1
            family = _length_family(int(raw_length))
        else:
            raise ValueError(f"primary outcome lacks deterministic family identity: {primary}")
        same_as_previous = bool(row.get("is_same_as_previous_primitive", False))
        repeat_depth = row.get("current_repeat_depth", row.get("repeat_depth", 1))
        repeat_status = (
            "SAME_PRIMITIVE_REPEAT"
            if same_as_previous or pd.notna(repeat_depth) and int(repeat_depth) > 1
            else "NEW_PRIMITIVE_AFTER_DIFFERENT_LOOP"
        )
        if family in {
            "NO_LOOP",
            "SESSION_END",
            "DISTINCT_PRIMITIVE_TIE",
            "UNAVAILABLE_SOURCE",
            "UNAVAILABLE_STRUCTURAL_GAP",
        }:
            repeat_status = "NOT_APPLICABLE"
        records.append(
            {
                **cast(dict[str, Any], row.to_dict()),
                "loop_family": family,
                "repeat_status": repeat_status,
                **safety_flags(),
            }
        )
    return pd.DataFrame.from_records(records)


def decide_target_tractability(
    metrics: ScientificDecisionMetrics | Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen scientific decision hierarchy without economic labels."""

    evidence = (
        metrics
        if isinstance(metrics, ScientificDecisionMetrics)
        else ScientificDecisionMetrics.from_mapping(metrics)
    )
    if evidence.blocked:
        label = "semantic_dictionary_experiment_blocked"
        reason = "required structural evidence was unavailable"
        decision_rule_gap = False
    elif (
        not evidence.semantic_ids_stable
        or evidence.coverage_collapsed
        or evidence.structural_excess_reversed
    ):
        label = "semantic_loop_dictionary_not_stable"
        reason = "unchanged retrospective structural validation did not retain semantics"
        decision_rule_gap = False
    else:
        exact = (
            evidence.dictionary_size >= 8
            and evidence.development_coverage >= 0.50
            and evidence.validation_coverage >= 0.45
            and evidence.development_coverage - evidence.validation_coverage <= 0.10
            and evidence.entries_rate_ratio_above_one_share >= 0.75
            and evidence.entries_threshold_retained_share >= 0.50
            and evidence.top_stock_share <= 0.20
            and evidence.genuine_tie_rate < 0.05
            and not evidence.other_dominated_by_obvious_candidate
            and evidence.semantic_ids_stable
        )
        hybrid = (
            evidence.exact_dictionary_stable_and_informative
            and evidence.validation_coverage >= 0.30
            and evidence.other_is_diffuse
            and evidence.family_reduces_residual_entropy
        )
        family = (
            evidence.validation_coverage < 0.30
            and evidence.other_is_diffuse
            and evidence.family_coverage_stable_and_higher
            and not evidence.exact_excess_consistent
        )
        if exact:
            label = "exact_next_loop_identity_tractable_for_preregistered_forecast"
            reason = "every preregistered exact-identity gate passed"
            decision_rule_gap = False
        elif hybrid:
            label = "hybrid_exact_dictionary_plus_other_ready_for_forecast"
            reason = "stable exact identities cover a bounded subset and OTHER remains diffuse"
            decision_rule_gap = False
        elif family:
            label = "topological_loop_family_target_preferred"
            reason = "family semantics are materially more stable than diffuse exact identities"
            decision_rule_gap = False
        else:
            label = "semantic_dictionary_experiment_blocked"
            reason = (
                "the frozen hierarchy has no decision for stable low exact coverage with "
                "retained exact structural excess"
            )
            decision_rule_gap = True
    return {
        "decision_label": label,
        "decision_reason": reason,
        "decision_rule_gap": decision_rule_gap,
        "next_loop_predictor_justified": label
        in {
            "exact_next_loop_identity_tractable_for_preregistered_forecast",
            "hybrid_exact_dictionary_plus_other_ready_for_forecast",
            "topological_loop_family_target_preferred",
        },
        **safety_flags(),
    }


def _primary_class(
    row: pd.Series,
    *,
    selected_primitive_ids: frozenset[str],
) -> str:
    event = str(row["primary_event"])
    if event in NON_LOOP_PRIMARY_CLASSES:
        return event
    primitive = row.get("primitive_loop_id")
    if not isinstance(primitive, str) or not primitive.startswith("loop_p_"):
        raise ValueError(f"identified loop event lacks primitive identity: {event}")
    return primitive if primitive in selected_primitive_ids else "OTHER_PRIMITIVE_LOOP"


def build_first_event_target(
    first_events: pd.DataFrame,
    *,
    selected_primitive_ids: set[str] | frozenset[str],
    horizon_bars: int,
    copy_outcomes: bool = True,
) -> FirstEventTargetBundle:
    """Freeze one primary class per decision under the preregistered precedence."""

    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    required = {"decision_id", "primary_event", "primitive_loop_id", "bars_until_completion"}
    missing = required.difference(first_events.columns)
    if missing:
        raise ValueError(f"first-event target missing fields: {sorted(missing)}")
    if first_events["decision_id"].duplicated().any():
        raise ValueError("exactly one first-event row is required per decision")
    selected = frozenset(str(value) for value in selected_primitive_ids)
    if any(not value.startswith("loop_p_") for value in selected):
        raise ValueError("selected dictionary may contain primitive semantic IDs only")
    loop_mask = first_events["primitive_loop_id"].notna()
    completion_bars = pd.to_numeric(
        first_events.loc[loop_mask, "bars_until_completion"], errors="coerce"
    )
    if completion_bars.isna().any() or completion_bars.gt(horizon_bars).any():
        raise ValueError("primitive event occurs after or lacks evidence within the frozen horizon")

    outcomes = first_events.copy() if copy_outcomes else first_events
    outcomes["primary_class"] = outcomes.apply(
        _primary_class, axis=1, selected_primitive_ids=selected
    )
    if outcomes["primary_class"].isna().any():
        raise AssertionError("primary target construction produced a missing class")

    auxiliary_fields = [
        "decision_id",
        "semantic_loop_id",
        "primitive_loop_id",
        "event_timestamp",
        "repeat_depth",
        "current_repeat_depth",
        "previous_same_primitive_completion_timestamp",
        "bars_since_previous_same_primitive",
        "transitions_since_previous_same_primitive",
        "is_consecutive_repeat",
        "is_same_as_previous_primitive",
        "nested_repeat_ids",
        "nested_composite_ids",
        "bars_until_completion",
        "state_events_until_completion",
        "active_prefix_length_at_decision",
        "initiated_before_decision",
        "initiated_after_decision",
        "earliest_composite_completion",
        "first_component_completion",
        "final_component_completion",
        "component_primitive_ids",
        "component_boundaries",
        "component_completion_timestamps",
        "earlier_primitive_completion_already_occurred",
        "composite_adds_information_beyond_primitive_sequence",
        "legacy_overlapping_positive_labels",
        "motif_type",
    ]
    if missing_auxiliary := sorted(set(auxiliary_fields).difference(first_events.columns)):
        raise ValueError(
            f"first-event evidence lacks required auxiliary fields: {missing_auxiliary}"
        )
    auxiliary = first_events[auxiliary_fields].copy()
    tie_details = outcomes.loc[outcomes["primary_class"].eq("DISTINCT_PRIMITIVE_TIE")].copy()
    target_contract: dict[str, Any] = {
        "target_version": "semantic_loop_dictionary_first_event_target_v2",
        "horizon_bars": horizon_bars,
        "horizon_inclusive": True,
        "selected_primitive_ids": sorted(selected),
        "other_class": "OTHER_PRIMITIVE_LOOP",
        "primary_non_loop_classes": sorted(NON_LOOP_PRIMARY_CLASSES),
        "repeat_is_auxiliary": True,
        "composite_is_auxiliary": True,
        **safety_flags(),
    }
    return FirstEventTargetBundle(
        outcomes=outcomes,
        tie_details=tie_details,
        auxiliary=auxiliary,
        target_contract=target_contract,
    )


__all__ = [
    "FirstEventTargetBundle",
    "NON_LOOP_PRIMARY_CLASSES",
    "ScientificDecisionMetrics",
    "build_first_event_target",
    "build_loop_family_mapping",
    "decide_target_tractability",
]
