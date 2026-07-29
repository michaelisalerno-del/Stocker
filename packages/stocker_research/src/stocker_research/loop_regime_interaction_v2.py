"""Fail-closed foundations for a future loop-regime interaction study.

This module defines causal population, chronology, support, and multiplicity
contracts.  It intentionally contains no interaction scorer or forecast model:
Part A did not authorize Part B scoring in the 20260718 experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from stocker_research.regime_validity_v2 import reject_outcome_columns, safety_flags

AUTHORIZED_PART_A_DECISIONS = frozenset(
    {
        "regime_representation_validated_for_loop_dictionary",
        "regime_representation_valid_with_required_sensitivity",
        "hierarchical_market_stock_regime_representation_preferred",
    }
)

NON_LOOP_FIRST_EVENT_CLASSES = frozenset(
    {
        "OTHER_PRIMITIVE_LOOP",
        "NO_LOOP_WITHIN_HORIZON",
        "SESSION_END",
        "DISTINCT_PRIMITIVE_TIE",
        "UNAVAILABLE_SOURCE",
        "UNAVAILABLE_STRUCTURAL_GAP",
    }
)


class PartBGateClosedError(RuntimeError):
    """Raised before any Part B population or score may be accessed."""


@dataclass(frozen=True, slots=True)
class PartAGateState:
    """Hash-bound Part A evidence needed before Part B source access."""

    decision: str
    decision_file_hash: str
    binding_hash: str
    state_model_hash: str
    state_alignment_hash: str
    independent_audit_status: str
    independent_audit_file_hash: str
    exact_rerun_byte_identical: bool
    exact_rerun_manifest_file_hash: str

    @property
    def scoring_authorized(self) -> bool:
        return (
            self.decision in AUTHORIZED_PART_A_DECISIONS
            and self.independent_audit_status == "pass"
            and self.exact_rerun_byte_identical
        )

    def validate_hashes(self) -> None:
        hashes = (
            self.decision_file_hash,
            self.binding_hash,
            self.state_model_hash,
            self.state_alignment_hash,
            self.independent_audit_file_hash,
            self.exact_rerun_manifest_file_hash,
        )
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("Part A binding contains a malformed SHA-256 identity")


def assert_part_b_scoring_authorized(gate: PartAGateState) -> None:
    """Fail before data access unless the independently audited gate is open."""

    gate.validate_hashes()
    if not gate.scoring_authorized:
        raise PartBGateClosedError(f"Part B scoring is closed by Part A decision {gate.decision!r}")


@dataclass(frozen=True, slots=True)
class CompletedStateHistory:
    """Past completed runs available at every decision row."""

    previous_states: np.ndarray
    previous_durations: np.ndarray
    history_tokens: tuple[str, ...]


def completed_state_history(
    states: np.ndarray,
    *,
    session_groups: Sequence[np.ndarray],
    depth: int = 4,
) -> CompletedStateHistory:
    """Construct past completed-run history with a hard session reset."""

    labels = np.asarray(states, dtype=int)
    if labels.ndim != 1 or depth <= 0 or np.any(labels < 0):
        raise ValueError("state history requires nonnegative one-dimensional states")
    previous_states = np.full((len(labels), depth), -1, dtype=np.int16)
    previous_durations = np.full((len(labels), depth), -1, dtype=np.int16)
    tokens = ["START" for _ in range(len(labels))]
    covered = np.zeros(len(labels), dtype=bool)
    for raw_group in session_groups:
        group = np.asarray(raw_group, dtype=int)
        if len(group) == 0:
            continue
        if (
            group.min() < 0
            or group.max() >= len(labels)
            or np.any(np.diff(group) <= 0)
            or covered[group].any()
        ):
            raise ValueError("session history groups overlap or are invalid")
        covered[group] = True
        completed: list[tuple[int, int]] = []
        current_state = int(labels[group[0]])
        current_duration = 0
        for position in group:
            state = int(labels[position])
            if state != current_state:
                completed.insert(0, (current_state, current_duration))
                completed = completed[:depth]
                current_state = state
                current_duration = 0
            current_duration += 1
            for history_index, (past_state, past_duration) in enumerate(completed):
                previous_states[position, history_index] = past_state
                previous_durations[position, history_index] = past_duration
            tokens[position] = (
                "START"
                if not completed
                else "|".join(f"s{state_value}:d{duration}" for state_value, duration in completed)
            )
    if not covered.all():
        raise ValueError("session history groups do not cover every decision row")
    return CompletedStateHistory(
        previous_states=previous_states,
        previous_durations=previous_durations,
        history_tokens=tuple(tokens),
    )


@dataclass(frozen=True, slots=True)
class ChronologicalFold:
    fold_id: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_periods: tuple[str, ...]
    validation_period: str


def expanding_period_folds(
    periods: Sequence[str], *, minimum_training_periods: int = 2
) -> tuple[ChronologicalFold, ...]:
    """Create deterministic expanding folds whose validation is strictly later."""

    values = np.asarray([str(period) for period in periods], dtype=object)
    ordered = tuple(sorted(set(values.tolist())))
    if minimum_training_periods < 1 or len(ordered) <= minimum_training_periods:
        raise ValueError("insufficient chronological periods for expanding folds")
    folds: list[ChronologicalFold] = []
    for validation_ordinal in range(minimum_training_periods, len(ordered)):
        train_periods = ordered[:validation_ordinal]
        validation_period = ordered[validation_ordinal]
        train = np.flatnonzero(np.isin(values, train_periods))
        validation = np.flatnonzero(values == validation_period)
        if not len(train) or not len(validation):
            raise AssertionError("chronological fold contains an empty partition")
        if max(train_periods) >= validation_period:
            raise AssertionError("chronological fold leaks validation into training")
        folds.append(
            ChronologicalFold(
                fold_id=f"expanding_{validation_ordinal:02d}",
                train_indices=train,
                validation_indices=validation,
                train_periods=train_periods,
                validation_period=validation_period,
            )
        )
    return tuple(folds)


@dataclass(frozen=True, slots=True)
class StockDeletionPopulation:
    omitted_symbol: str
    retained_indices: np.ndarray
    decision_signature: str


def stock_deletion_populations(
    symbols: Sequence[str], decision_ids: Sequence[str]
) -> tuple[StockDeletionPopulation, ...]:
    """Recompute each leave-one-stock-out population from the full row set."""

    symbol_values = np.asarray([str(symbol) for symbol in symbols], dtype=object)
    decisions = np.asarray([str(decision) for decision in decision_ids], dtype=object)
    if symbol_values.shape != decisions.shape or symbol_values.ndim != 1:
        raise ValueError("stock deletions require aligned one-dimensional rows")
    if len(set(decisions.tolist())) != len(decisions):
        raise ValueError("stock-deletion decision IDs must be unique")
    outputs: list[StockDeletionPopulation] = []
    for omitted in sorted(set(symbol_values.tolist())):
        retained = np.flatnonzero(symbol_values != omitted)
        retained_decisions = tuple(str(value) for value in decisions[retained])
        encoded = json.dumps(retained_decisions, separators=(",", ":")).encode("utf-8")
        outputs.append(
            StockDeletionPopulation(
                omitted_symbol=omitted,
                retained_indices=retained,
                decision_signature=hashlib.sha256(encoded).hexdigest(),
            )
        )
    return tuple(outputs)


def assert_identical_decision_populations(
    populations: Mapping[str, Sequence[str]],
) -> str:
    """Require every attribution model to use the same ordered decision rows."""

    if not populations:
        raise ValueError("at least one model population is required")
    signatures: dict[str, str] = {}
    for model_id, raw_decisions in populations.items():
        decisions = tuple(str(decision) for decision in raw_decisions)
        if len(decisions) != len(set(decisions)):
            raise ValueError(f"model {model_id} contains duplicate decision IDs")
        encoded = json.dumps(decisions, separators=(",", ":")).encode("utf-8")
        signatures[str(model_id)] = hashlib.sha256(encoded).hexdigest()
    if len(set(signatures.values())) != 1:
        raise ValueError("attribution models do not share identical ordered decision rows")
    return next(iter(signatures.values()))


def validate_interaction_columns(columns: Sequence[str]) -> None:
    """Reject economic and future-price fields before matrix construction."""

    reject_outcome_columns(tuple(str(column) for column in columns))


def validate_first_event_classes(
    primary_classes: Sequence[str], *, selected_primitive_ids: frozenset[str]
) -> None:
    """Retain every corrected structural class and reject legacy labels."""

    selected = frozenset(str(value) for value in selected_primitive_ids)
    if any(not value.startswith("loop_p_") for value in selected):
        raise ValueError("selected first-event classes must be primitive semantic IDs")
    observed = frozenset(str(value) for value in primary_classes)
    unknown = observed.difference(selected | NON_LOOP_FIRST_EVENT_CLASSES)
    if unknown:
        raise ValueError(f"unknown or legacy first-event classes: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class SupportThresholds:
    minimum_decision_rows: int = 200
    minimum_completion_events: int = 40
    minimum_stocks: int = 10
    minimum_sessions: int = 30
    minimum_months: int = 4
    maximum_single_stock_share: float = 0.25


@dataclass(frozen=True, slots=True)
class CounterfactualSupport:
    decision_rows: int
    completion_events: int
    stocks: int
    sessions: int
    months: int
    maximum_single_stock_share: float


@dataclass(frozen=True, slots=True)
class SupportDecision:
    supported: bool
    failures: tuple[str, ...]


def evaluate_counterfactual_support(
    support: CounterfactualSupport,
    *,
    thresholds: SupportThresholds = SupportThresholds(),
) -> SupportDecision:
    """Apply all preregistered support gates; any failed component closes the cell."""

    numeric_counts = (
        support.decision_rows,
        support.completion_events,
        support.stocks,
        support.sessions,
        support.months,
    )
    if any(value < 0 for value in numeric_counts) or not math.isfinite(
        support.maximum_single_stock_share
    ):
        raise ValueError("counterfactual support values must be finite and nonnegative")
    failures: list[str] = []
    checks = (
        (support.decision_rows >= thresholds.minimum_decision_rows, "decision_rows"),
        (
            support.completion_events >= thresholds.minimum_completion_events,
            "completion_events",
        ),
        (support.stocks >= thresholds.minimum_stocks, "stocks"),
        (support.sessions >= thresholds.minimum_sessions, "sessions"),
        (support.months >= thresholds.minimum_months, "months"),
        (
            support.maximum_single_stock_share <= thresholds.maximum_single_stock_share,
            "single_stock_concentration",
        ),
    )
    failures.extend(name for passed, name in checks if not passed)
    return SupportDecision(supported=not failures, failures=tuple(failures))


def benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return deterministic within-family BH q-values with stable tie handling."""

    if not p_values:
        return {}
    ordered = sorted((float(value), str(identifier)) for identifier, value in p_values.items())
    if len(ordered) != len(p_values):
        raise ValueError("BH family identifiers must be unique strings")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value, _ in ordered):
        raise ValueError("BH p-values must be finite probabilities")
    count = len(ordered)
    adjusted = [0.0] * count
    running = 1.0
    for index in range(count - 1, -1, -1):
        p_value, _ = ordered[index]
        running = min(running, p_value * count / (index + 1))
        adjusted[index] = min(1.0, running)
    return {identifier: adjusted[index] for index, (_, identifier) in enumerate(ordered)}


def population_scaffold(gate: PartAGateState, *, proposed_contract_hash: str) -> dict[str, Any]:
    """Describe the unopened population without reading or materializing rows."""

    gate.validate_hashes()
    if len(proposed_contract_hash) != 64:
        raise ValueError("proposed Part B contract hash must be SHA-256")
    return {
        "scaffold_version": "loop_regime_interaction_population_v2_blocked",
        "status": "schema_only_part_b_scoring_not_authorized",
        "part_a_decision": gate.decision,
        "part_a_decision_file_hash": gate.decision_file_hash,
        "part_a_binding_hash": gate.binding_hash,
        "part_a_independent_audit_file_hash": gate.independent_audit_file_hash,
        "part_a_exact_rerun_manifest_file_hash": gate.exact_rerun_manifest_file_hash,
        "state_model_hash": gate.state_model_hash,
        "state_alignment_hash": gate.state_alignment_hash,
        "proposed_contract_hash": proposed_contract_hash,
        "population_rows_read": 0,
        "interaction_results_inspected": False,
        "interaction_models_fit": 0,
        "schema_groups": {
            "identity_and_provenance": [
                "run_id",
                "git_sha",
                "contract_hash",
                "data_snapshot_hash",
                "state_model_version",
                "state_model_hash",
                "state_representation",
                "dictionary_version",
                "dictionary_hash",
                "decision_id",
                "symbol",
                "session",
                "decision_timestamp",
                "primitive_loop_id",
                "orientation_id",
                "prefix_progress",
                "source_artifact",
                "source_hash",
            ],
            "loop_structure": [
                "active_primitive_prefix_ids",
                "transitions_completed",
                "transitions_remaining",
                "required_next_state",
                "competing_prefix_count",
                "required_next_state_agreement",
                "previous_completed_primitive_loop",
                "previous_two_primitive_loops",
                "same_loop_repeat_depth",
                "bars_since_previous_completion",
            ],
            "current_regime": [
                "frozen_current_hard_state",
                "hysteretic_state",
                "state_posterior",
                "posterior_entropy",
                "top_two_margin",
                "hard_state_age",
                "expected_state_age",
                "departure_probability",
                "most_likely_next_state",
            ],
            "regime_history": [
                "previous_completed_state_1",
                "previous_completed_state_2",
                "previous_completed_state_3",
                "previous_completed_state_4",
                "completed_state_durations",
                "history_token",
                "state_transition_surprise",
                "entered_from_expected_loop_leg",
            ],
            "market_context": [
                "market_regime",
                "market_regime_age",
                "market_regime_posterior",
                "breadth",
                "dispersion",
                "stock_minus_market_movement",
                "broad_clock_phase",
            ],
            "first_event_target": sorted(NON_LOOP_FIRST_EVENT_CLASSES) + ["selected_loop_p_*"],
        },
        **safety_flags(),
    }


__all__ = [
    "AUTHORIZED_PART_A_DECISIONS",
    "NON_LOOP_FIRST_EVENT_CLASSES",
    "ChronologicalFold",
    "CompletedStateHistory",
    "CounterfactualSupport",
    "PartBGateClosedError",
    "PartAGateState",
    "SupportDecision",
    "SupportThresholds",
    "StockDeletionPopulation",
    "assert_identical_decision_populations",
    "assert_part_b_scoring_authorized",
    "benjamini_hochberg",
    "completed_state_history",
    "evaluate_counterfactual_support",
    "expanding_period_folds",
    "population_scaffold",
    "stock_deletion_populations",
    "validate_first_event_classes",
    "validate_interaction_columns",
]
