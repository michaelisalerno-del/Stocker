"""Assemble causal completed-bar, prefix, completion, and outcome ledgers V2.

This module joins the bar-time decision surface to the state-event automaton.
Repeated bars in one state share the same automaton position, but each completed
bar receives its own decision timestamp and mutually exclusive outcome.  Any
missing in-session bar fails the whole source sequence closed for structural
labels; the completed decisions remain present and are marked unavailable.

Safety boundary: research only; execution is disabled, order placement is
disabled, no broker is connected, and strategy promotion is disabled.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_research.causal_state_export_v2 import SoftLoopPrefixTracker
from stocker_research.loop_dictionary_v2 import LoopDictionary, MotifType
from stocker_research.loop_events_v2 import (
    LoopCompletionEvent,
    PrimaryOutcomeLabel,
    StructuralOutcomeRow,
    safety_flags,
)
from stocker_research.loop_prefix_automaton_v2 import (
    EventTrace,
    FirstNextLoopEventEngine,
    legacy_compatible_cycle_labels,
)

RESEARCH_ONLY = True
EXECUTION_ENABLED = False
ORDER_PLACEMENT = "disabled"
BROKER_CONNECTED = False
STRATEGY_PROMOTION = False


@dataclass(frozen=True, slots=True)
class LoopEventLedgerBundle:
    """Aligned research-only ledgers returned without writing external state."""

    decisions: pd.DataFrame
    prefixes: pd.DataFrame
    completions: pd.DataFrame
    outcomes: pd.DataFrame
    legacy_targets: pd.DataFrame
    target_comparison: pd.DataFrame


def adapt_legacy_run_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and preserve a legacy hard-run ledger as read-only lineage."""

    required = {
        "run_id",
        "symbol",
        "session",
        "state",
        "duration",
        "start_timestamp",
        "end_timestamp",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"legacy run ledger lacks columns: {missing}")
    output = frame.copy().reset_index(drop=True)
    output = output.rename(columns={"run_id": "legacy_run_id"})
    start = pd.to_datetime(output["start_timestamp"], utc=True, errors="coerce")
    end = pd.to_datetime(output["end_timestamp"], utc=True, errors="coerce")
    duration = pd.to_numeric(output["duration"], errors="coerce")
    state = pd.to_numeric(output["state"], errors="coerce")
    supported = (
        start.notna()
        & end.notna()
        & start.le(end)
        & duration.notna()
        & duration.gt(0)
        & state.notna()
    )
    output["supported"] = supported
    output["migration_status"] = np.where(supported, "compatible_read_only", "unavailable")
    output["ambiguity_reason"] = [
        None if bool(is_supported) else "invalid_run_identity_timing_state_or_duration"
        for is_supported in supported
    ]
    output["legacy_context_timing_semantics"] = "preserved_not_reinterpreted"
    return output


def adapt_legacy_overlapping_target_panel(
    frame: pd.DataFrame, *, dictionary: LoopDictionary
) -> pd.DataFrame:
    """Map long-form legacy binary labels while preserving overlapping rows."""

    required = {"decision_id", "legacy_cycle_id", "target"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"legacy target panel lacks columns: {missing}")
    output = frame.copy().reset_index(drop=True)
    target = pd.to_numeric(output["target"], errors="coerce")
    migrations = {str(row["legacy_cycle_id"]): row for row in dictionary.migration_rows()}
    mapped = output["legacy_cycle_id"].astype(str).map(migrations)
    valid_target = target.isin([0, 1])
    known_id = mapped.notna()
    output["semantic_loop_id"] = [
        value.get("semantic_loop_id") if isinstance(value, dict) else None for value in mapped
    ]
    output["primitive_loop_id"] = [
        value.get("primitive_loop_id") if isinstance(value, dict) else None for value in mapped
    ]
    output["supported"] = valid_target & known_id
    output["migration_status"] = np.where(
        output["supported"], "mapped_overlapping_diagnostic", "unavailable"
    )
    output["ambiguity_reason"] = [
        "unknown_legacy_cycle_id"
        if not bool(identifier_known)
        else "non_binary_legacy_target"
        if not bool(target_valid)
        else None
        for identifier_known, target_valid in zip(known_id, valid_target, strict=True)
    ]
    output["target_semantics"] = "legacy_overlapping_compatible_cycle"
    return output


def compare_legacy_targets_to_v2_outcomes(
    legacy_targets: pd.DataFrame,
    v2_outcomes: pd.DataFrame,
    v2_decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare legacy overlapping labels with the V2 dictionary's first event.

    The two target surfaces can be generated from different dictionaries.  In
    particular, ``legacy_targets`` retains every migrated frozen cycle while
    ``v2_outcomes`` must come from ``semantic_loop_dictionary_v2``.  Combining
    the same-dictionary diagnostic emitted by ``build_loop_event_ledgers``
    would silently substitute a legacy-dictionary first event for the V2
    primary outcome.
    """

    required = {
        "legacy_targets": {
            "decision_id",
            "legacy_positive_semantic_ids",
            "source_available",
        },
        "v2_outcomes": {
            "decision_id",
            "primary_label",
            "earliest_event_ids",
            "source_available",
        },
        "v2_decisions": {
            "decision_id",
            "symbol",
            "session",
            "decision_timestamp",
            "active_prefix_count",
            "is_run_entry",
            "structural_event_eligibility",
        },
    }
    for name, columns in required.items():
        frame = {
            "legacy_targets": legacy_targets,
            "v2_outcomes": v2_outcomes,
            "v2_decisions": v2_decisions,
        }[name]
        missing = sorted(columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} lacks columns: {missing}")
        if frame["decision_id"].duplicated().any():
            raise ValueError(f"{name} decision IDs are not unique")

    merged = v2_decisions[
        [
            "decision_id",
            "symbol",
            "session",
            "decision_timestamp",
            "active_prefix_count",
            "is_run_entry",
            "structural_event_eligibility",
        ]
    ].merge(
        legacy_targets[["decision_id", "legacy_positive_semantic_ids", "source_available"]].rename(
            columns={"source_available": "legacy_source_available"}
        ),
        on="decision_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(
        v2_outcomes[
            ["decision_id", "primary_label", "earliest_event_ids", "source_available"]
        ].rename(columns={"source_available": "v2_source_available"}),
        on="decision_id",
        how="left",
        validate="one_to_one",
    )
    if merged[["legacy_source_available", "v2_source_available"]].isna().any().any():
        raise ValueError("legacy and V2 surfaces do not contain identical decision populations")

    rows: list[dict[str, Any]] = []
    records = cast(list[dict[str, Any]], merged.to_dict(orient="records"))
    for row in records:
        legacy_ids = tuple(str(value) for value in row["legacy_positive_semantic_ids"])
        v2_ids = tuple(str(value) for value in row["earliest_event_ids"])
        available = (
            bool(row["structural_event_eligibility"])
            and bool(row["legacy_source_available"])
            and bool(row["v2_source_available"])
        )
        registered_differs = available and set(legacy_ids) != set(v2_ids)
        primary_label = str(row["primary_label"])
        v2_only = sorted(set(v2_ids).difference(legacy_ids)) if v2_ids else [primary_label]
        rows.append(
            {
                "decision_id": str(row["decision_id"]),
                "symbol": str(row["symbol"]),
                "session": str(row["session"]),
                "timestamp": row["decision_timestamp"],
                "legacy_positive_labels": list(legacy_ids),
                "v2_first_event": primary_label,
                "legacy_only_labels": sorted(set(legacy_ids).difference(v2_ids)),
                "v2_only_events": v2_only,
                "legacy_positive_count": len(legacy_ids),
                "active_prefix_count": int(row["active_prefix_count"]),
                "registered_event_set_differs": registered_differs,
                "semantics_differ": available and (registered_differs or not v2_ids),
                "comparison_available": available,
                "is_run_entry": bool(row["is_run_entry"]),
                **safety_flags(),
            }
        )
    return pd.DataFrame(rows)


def _validate_decision_surface(frame: pd.DataFrame, state_column: str) -> None:
    required = {
        "decision_id",
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "decision_timestamp",
        "posterior_state_probabilities",
        state_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"decision surface lacks required fields: {missing}")
    if frame.empty:
        raise ValueError("decision surface cannot be empty")
    if frame["decision_id"].duplicated().any():
        raise ValueError("decision IDs are not unique")
    if not frame["bar_complete_timestamp"].le(frame["decision_timestamp"]).all():
        raise ValueError("a decision precedes its completed source bar")


def session_source_is_complete(frame: pd.DataFrame) -> bool:
    """Return whether a session is a contiguous causal five-minute sequence."""

    ordinals = frame["bar_ordinal"].to_numpy(dtype=int)
    if not np.array_equal(ordinals, np.arange(len(frame), dtype=int)):
        return False
    starts = pd.to_datetime(frame["bar_start_timestamp"], utc=True)
    if starts.duplicated().any() or not starts.is_monotonic_increasing:
        return False
    if len(starts) > 1:
        deltas = starts.diff().iloc[1:]
        if not deltas.eq(pd.Timedelta(minutes=5)).all():
            return False
    return True


def _state_event_inputs(frame: pd.DataFrame, state_column: str) -> tuple[np.ndarray, np.ndarray]:
    states = frame[state_column].to_numpy(dtype=int)
    event_mask = np.r_[True, states[1:] != states[:-1]]
    event_positions = np.flatnonzero(event_mask)
    event_for_bar = np.cumsum(event_mask, dtype=int) - 1
    return event_positions, event_for_bar


def _prefix_aggregates(prefixes: Sequence[Any]) -> dict[str, Any]:
    by_motif = {
        motif: sorted(
            {prefix.semantic_loop_id for prefix in prefixes if prefix.motif_type is motif}
        )
        for motif in MotifType
    }
    return {
        "active_prefix_count": len(prefixes),
        "active_primitive_prefixes": by_motif[MotifType.PRIMITIVE],
        "active_repeat_prefixes": by_motif[MotifType.REPEAT],
        "active_composite_prefixes": by_motif[MotifType.COMPOSITE],
        "shortest_transitions_remaining": min(
            (prefix.transitions_remaining for prefix in prefixes), default=None
        ),
    }


def _history_features(
    trace: EventTrace,
    *,
    decision_event_index: int,
    decision_bar_ordinal: int,
) -> dict[str, Any]:
    previous_states = [event.state for event in trace.state_events[:decision_event_index]]
    primitive_events = [
        event
        for event in trace.registered_completions
        if event.motif_type is MotifType.PRIMITIVE
        and event.completion_event_index <= decision_event_index
        and event.completion_bar_ordinal <= decision_bar_ordinal
    ]
    primitive_ids = [event.semantic_loop_id for event in primitive_events]
    repeat_depth = 0
    if primitive_ids:
        latest = primitive_ids[-1]
        for value in reversed(primitive_ids):
            if value != latest:
                break
            repeat_depth += 1
    output: dict[str, Any] = {}
    for lag in range(1, 5):
        output[f"previous_completed_state_{lag}"] = (
            previous_states[-lag] if len(previous_states) >= lag else None
        )
    output["previous_primitive_loop_1"] = primitive_ids[-1] if primitive_ids else None
    output["previous_primitive_loop_2"] = primitive_ids[-2] if len(primitive_ids) >= 2 else None
    output["bars_since_previous_loop"] = (
        decision_bar_ordinal - primitive_events[-1].completion_bar_ordinal
        if primitive_events
        else None
    )
    output["same_loop_repeat_depth"] = repeat_depth
    return output


def _prefix_row(decision: pd.Series, prefix: Any) -> dict[str, Any]:
    return {
        "decision_id": decision["decision_id"],
        "run_id": decision.get("run_id"),
        "git_sha": decision.get("git_sha"),
        "contract_hash": decision.get("contract_hash"),
        "data_snapshot_hash": decision.get("data_snapshot_hash"),
        "dictionary_version": decision.get("dictionary_version"),
        "state_model_version": decision.get("state_model_version"),
        "symbol": decision["symbol"],
        "session": decision["session"],
        "timestamp": decision["decision_timestamp"],
        "bar_ordinal": int(decision["bar_ordinal"]),
        "semantic_loop_id": prefix.semantic_loop_id,
        "legacy_loop_id": None,
        "primitive_loop_id": prefix.primitive_loop_id,
        "orientation_id": prefix.orientation_id,
        "motif_type": prefix.motif_type.value,
        "repeat_depth": prefix.repeat_depth,
        "prefix_path": list(prefix.prefix_path),
        "progress_states": prefix.progress_states,
        "transitions_remaining": prefix.transitions_remaining,
        "start_event_index": prefix.start_event_index,
        "start_prefix_timestamp": prefix.start_prefix_timestamp,
        "start_prefix_available_timestamp": prefix.start_prefix_available_timestamp,
        **safety_flags(),
    }


def _completion_row(
    decision: pd.Series,
    event: LoopCompletionEvent,
    *,
    session_end_bar_ordinal: int,
    primary_completion_keys: frozenset[tuple[str, int]],
) -> dict[str, Any]:
    row = asdict(event)
    row["motif_type"] = event.motif_type.value
    row["full_path"] = list(event.full_path)
    row["active_prefix_at_decision"] = list(event.active_prefix_at_decision)
    row["source_hashes"] = [list(item) for item in event.source_hashes]
    row["legacy_loop_id"] = None
    row["run_id"] = decision.get("run_id")
    row["git_sha"] = decision.get("git_sha")
    row["contract_hash"] = decision.get("contract_hash")
    row["data_snapshot_hash"] = decision.get("data_snapshot_hash")
    row["dictionary_version"] = decision.get("dictionary_version")
    row["state_model_version"] = decision.get("state_model_version")
    row["timestamp"] = event.completion_available_timestamp
    row["completed_state_runs_until_completion"] = event.state_events_until_completion
    row["session_terminal"] = event.completion_bar_ordinal == session_end_bar_ordinal
    row["is_primary_completion"] = (
        event.semantic_loop_id,
        event.completion_event_index,
    ) in primary_completion_keys
    row.update(safety_flags())
    return row


def _outcome_row(
    decision: pd.Series,
    outcome: StructuralOutcomeRow,
    *,
    previous_primitive_loop: str | None,
) -> dict[str, Any]:
    earliest_primitive_id = outcome.earliest_primitive_completion
    first_event_primitive_ids = {
        event.primitive_loop_id
        for event in outcome.earliest_registered_events
        if event.primitive_loop_id is not None
    }
    return {
        "decision_id": outcome.decision_id,
        "run_id": decision.get("run_id"),
        "git_sha": decision.get("git_sha"),
        "contract_hash": decision.get("contract_hash"),
        "data_snapshot_hash": decision.get("data_snapshot_hash"),
        "dictionary_version": decision.get("dictionary_version"),
        "state_model_version": decision.get("state_model_version"),
        "symbol": decision["symbol"],
        "session": decision["session"],
        "timestamp": decision["decision_timestamp"],
        "decision_timestamp": decision["decision_timestamp"],
        "decision_available_timestamp": decision["decision_timestamp"],
        "bar_ordinal": int(decision["bar_ordinal"]),
        "primary_label": str(outcome.primary_label),
        "tied_semantic_loop_ids": list(outcome.tied_semantic_loop_ids),
        "earliest_event_ids": [
            event.semantic_loop_id for event in outcome.earliest_registered_events
        ],
        "every_completion_within_horizon": list(outcome.every_registered_completion_within_horizon),
        "earliest_primitive_completion": earliest_primitive_id,
        "earliest_repeated_completion": outcome.earliest_repeated_completion,
        "earliest_composite_completion": outcome.earliest_composite_completion,
        "bars_until_completion": outcome.bars_until_completion,
        "state_events_until_completion": outcome.state_events_until_completion,
        "completed_state_runs_until_completion": outcome.state_events_until_completion,
        "transitions_remaining_at_decision": outcome.transitions_remaining_at_decision,
        "first_event_was_open_prefix": outcome.first_event_was_open_prefix,
        "first_event_began_after_decision": outcome.first_event_began_after_decision,
        "first_completion_same_as_previous_primitive_loop": (
            previous_primitive_loop is not None
            and previous_primitive_loop in first_event_primitive_ids
        ),
        "earliest_registered_primitive_same_as_previous": (
            earliest_primitive_id is not None and earliest_primitive_id == previous_primitive_loop
        ),
        "repeat_depth": outcome.repeat_depth,
        "source_available": outcome.source_available,
        "missing_reason": outcome.missing_reason,
        **safety_flags(),
    }


def _unavailable_outcome(decision: pd.Series) -> dict[str, Any]:
    return {
        "decision_id": decision["decision_id"],
        "run_id": decision.get("run_id"),
        "git_sha": decision.get("git_sha"),
        "contract_hash": decision.get("contract_hash"),
        "data_snapshot_hash": decision.get("data_snapshot_hash"),
        "dictionary_version": decision.get("dictionary_version"),
        "state_model_version": decision.get("state_model_version"),
        "symbol": decision["symbol"],
        "session": decision["session"],
        "timestamp": decision["decision_timestamp"],
        "decision_timestamp": decision["decision_timestamp"],
        "decision_available_timestamp": decision["decision_timestamp"],
        "bar_ordinal": int(decision["bar_ordinal"]),
        "primary_label": str(PrimaryOutcomeLabel.UNAVAILABLE),
        "tied_semantic_loop_ids": [],
        "earliest_event_ids": [],
        "every_completion_within_horizon": [],
        "earliest_primitive_completion": None,
        "earliest_repeated_completion": None,
        "earliest_composite_completion": None,
        "bars_until_completion": None,
        "state_events_until_completion": None,
        "completed_state_runs_until_completion": None,
        "transitions_remaining_at_decision": None,
        "first_event_was_open_prefix": False,
        "first_event_began_after_decision": False,
        "first_completion_same_as_previous_primitive_loop": False,
        "repeat_depth": None,
        "source_available": False,
        "missing_reason": "source_sequence_incomplete_or_ambiguous",
        **safety_flags(),
    }


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def build_loop_event_ledgers(
    decisions: pd.DataFrame,
    *,
    dictionary: LoopDictionary,
    horizon_bars: int,
    allowed_states: frozenset[int],
    state_column: str = "hard_state_legacy",
    source_hashes: Sequence[tuple[str, str]] = (),
    soft_prefix_session_keys: frozenset[tuple[str, str]] | None = None,
) -> LoopEventLedgerBundle:
    """Build deterministic first-event and legacy diagnostic ledgers.

    The returned ``decisions`` frame contains only causal features and prefix
    state known at the decision.  Future structural outcomes live exclusively
    in the separate outcome, completion, and legacy diagnostic ledgers.
    """

    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    frame = decisions.copy()
    _validate_decision_surface(frame, state_column)
    frame = frame.sort_values(
        ["symbol", "session", "bar_ordinal", "decision_id"], kind="mergesort"
    ).reset_index(drop=True)

    decision_updates: dict[int, dict[str, Any]] = {}
    prefix_rows: list[dict[str, Any]] = []
    completion_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=allowed_states)
    semantic_ids = set(dictionary.semantic_ids)
    semantic_to_legacy: dict[str, list[str]] = {}
    for migration in dictionary.migration_rows():
        semantic_to_legacy.setdefault(str(migration["semantic_loop_id"]), []).append(
            str(migration["legacy_cycle_id"])
        )

    for (_, _), group in frame.groupby(["symbol", "session"], sort=False):
        positions = group.index.to_numpy(dtype=int)
        local = group.reset_index(drop=True)
        complete = session_source_is_complete(local)
        if "source_sequence_complete" in local:
            declared = local["source_sequence_complete"].astype(bool)
            if declared.nunique() != 1 or bool(declared.iloc[0]) != complete:
                raise AssertionError("session source-eligibility declarations disagree")
        states = local[state_column].to_numpy(dtype=int)
        complete = complete and bool(np.isin(states, list(allowed_states)).all())
        soft_tracker = SoftLoopPrefixTracker(dictionary, state_count=max(allowed_states) + 1)
        session_key = (str(local.at[0, "symbol"]), str(local.at[0, "session"]))
        compute_soft_prefix = (
            soft_prefix_session_keys is None or session_key in soft_prefix_session_keys
        )

        if not complete:
            for local_index, (_, decision) in enumerate(local.iterrows()):
                update = {
                    **_prefix_aggregates(()),
                    **{f"previous_completed_state_{lag}": None for lag in range(1, 5)},
                    "previous_primitive_loop_1": None,
                    "previous_primitive_loop_2": None,
                    "bars_since_previous_loop": None,
                    "same_loop_repeat_depth": 0,
                    "highest_soft_prefix_probability": None,
                    "soft_completion_probabilities": None,
                    "soft_prefix_missing_reason": "source_sequence_incomplete_or_ambiguous",
                    "structural_event_eligibility": False,
                }
                decision_updates[int(positions[local_index])] = update
                unavailable = _unavailable_outcome(decision)
                outcome_rows.append(unavailable)
                legacy_rows.append(
                    {
                        "decision_id": decision["decision_id"],
                        "symbol": decision["symbol"],
                        "session": decision["session"],
                        "timestamp": decision["decision_timestamp"],
                        "legacy_positive_semantic_ids": [],
                        "legacy_positive_legacy_cycle_ids": [],
                        "legacy_positive_count": 0,
                        "source_available": False,
                        **safety_flags(),
                    }
                )
                comparison_rows.append(
                    {
                        "decision_id": decision["decision_id"],
                        "symbol": decision["symbol"],
                        "session": decision["session"],
                        "timestamp": decision["decision_timestamp"],
                        "legacy_positive_labels": [],
                        "v2_first_event": str(PrimaryOutcomeLabel.UNAVAILABLE),
                        "legacy_only_labels": [],
                        "v2_only_events": [str(PrimaryOutcomeLabel.UNAVAILABLE)],
                        "legacy_positive_count": 0,
                        "active_prefix_count": 0,
                        "registered_event_set_differs": False,
                        "semantics_differ": False,
                        "comparison_available": False,
                        "is_run_entry": bool(decision.get("is_run_entry", False)),
                        **safety_flags(),
                    }
                )
            continue

        event_positions, event_for_bar = _state_event_inputs(local, state_column)
        trace = engine.scan_state_events(
            states[event_positions],
            bar_ordinals=tuple(
                int(value)
                for value in local.iloc[event_positions]["bar_ordinal"].to_numpy(dtype=int)
            ),
            event_timestamps=tuple(
                pd.Timestamp(value).to_pydatetime()
                for value in local.iloc[event_positions]["bar_start_timestamp"]
            ),
            available_timestamps=tuple(
                pd.Timestamp(value).to_pydatetime()
                for value in local.iloc[event_positions]["bar_complete_timestamp"]
            ),
        )
        session_end = int(local["bar_ordinal"].max())

        for local_index, (_, decision) in enumerate(local.iterrows()):
            event_index = int(event_for_bar[local_index])
            active_prefixes = trace.prefixes_after_event[event_index]
            history = _history_features(
                trace,
                decision_event_index=event_index,
                decision_bar_ordinal=int(decision["bar_ordinal"]),
            )
            if compute_soft_prefix:
                posterior = np.asarray(decision["posterior_state_probabilities"], dtype=float)
                soft_snapshot = soft_tracker.update(posterior)
                soft_probability: float | None = soft_snapshot.highest_prefix_probability
                completion_by_semantic: dict[str, float] = defaultdict(float)
                for semantic_id, _, probability in soft_snapshot.completion_probabilities:
                    completion_by_semantic[semantic_id] += probability
                soft_completion_probabilities: dict[str, float] | None = {
                    key: min(1.0, value) for key, value in sorted(completion_by_semantic.items())
                }
                soft_missing_reason = None
            else:
                soft_probability = None
                soft_completion_probabilities = None
                soft_missing_reason = "bounded_soft_posterior_sample_not_selected"
            update = {
                **_prefix_aggregates(active_prefixes),
                **history,
                "highest_soft_prefix_probability": soft_probability,
                "soft_completion_probabilities": soft_completion_probabilities,
                "soft_prefix_missing_reason": soft_missing_reason,
                "structural_event_eligibility": True,
            }
            decision_updates[int(positions[local_index])] = update
            prefix_rows.extend(_prefix_row(decision, prefix) for prefix in active_prefixes)

            outcome = engine.outcome_for_decision(
                trace,
                decision_id=str(decision["decision_id"]),
                decision_event_index=event_index,
                decision_bar_ordinal=int(decision["bar_ordinal"]),
                decision_timestamp=pd.Timestamp(decision["decision_timestamp"]).to_pydatetime(),
                decision_available_timestamp=pd.Timestamp(
                    decision["decision_timestamp"]
                ).to_pydatetime(),
                horizon_bars=horizon_bars,
                session_end_bar_ordinal=session_end,
                source_available=True,
                symbol=str(decision["symbol"]),
                session=str(decision["session"]),
                source_hashes=source_hashes,
            )
            outcome_rows.append(
                _outcome_row(
                    decision,
                    outcome,
                    previous_primitive_loop=history["previous_primitive_loop_1"],
                )
            )
            completion_rows.extend(
                _completion_row(
                    decision,
                    event,
                    session_end_bar_ordinal=session_end,
                    primary_completion_keys=frozenset(
                        (primary.semantic_loop_id, primary.completion_event_index)
                        for primary in outcome.earliest_registered_events
                    ),
                )
                for event in outcome.every_registered_completion_event
            )

            legacy = legacy_compatible_cycle_labels(
                trace, decision_event_index=event_index, dictionary=dictionary
            )
            legacy_ids = sorted(
                {
                    legacy_id
                    for semantic_id in legacy
                    for legacy_id in semantic_to_legacy.get(semantic_id, ())
                }
            )
            legacy_rows.append(
                {
                    "decision_id": decision["decision_id"],
                    "symbol": decision["symbol"],
                    "session": decision["session"],
                    "timestamp": decision["decision_timestamp"],
                    "legacy_positive_semantic_ids": list(legacy),
                    "legacy_positive_legacy_cycle_ids": legacy_ids,
                    "legacy_positive_count": len(legacy),
                    "source_available": True,
                    **safety_flags(),
                }
            )
            if str(outcome.primary_label) in semantic_ids:
                v2_event_ids: tuple[str, ...] = (str(outcome.primary_label),)
                v2_only_export = sorted(set(v2_event_ids).difference(legacy))
            elif outcome.tied_semantic_loop_ids:
                v2_event_ids = outcome.tied_semantic_loop_ids
                v2_only_export = sorted(set(v2_event_ids).difference(legacy))
            else:
                v2_event_ids = ()
                v2_only_export = [str(outcome.primary_label)]
            registered_differs = set(legacy) != set(v2_event_ids)
            comparison_rows.append(
                {
                    "decision_id": decision["decision_id"],
                    "symbol": decision["symbol"],
                    "session": decision["session"],
                    "timestamp": decision["decision_timestamp"],
                    "legacy_positive_labels": list(legacy),
                    "v2_first_event": str(outcome.primary_label),
                    "legacy_only_labels": sorted(set(legacy).difference(v2_event_ids)),
                    "v2_only_events": v2_only_export,
                    "legacy_positive_count": len(legacy),
                    "active_prefix_count": len(active_prefixes),
                    "registered_event_set_differs": registered_differs,
                    "semantics_differ": registered_differs or not v2_event_ids,
                    "comparison_available": True,
                    "is_run_entry": bool(decision.get("is_run_entry", False)),
                    **safety_flags(),
                }
            )

    if len(decision_updates) != len(frame):
        raise AssertionError("not every completed bar received structural features")
    update_frame = pd.DataFrame.from_dict(decision_updates, orient="index").reindex(frame.index)
    for column in update_frame:
        frame[column] = update_frame[column]
    structural_features = (
        "previous_completed_state_1",
        "previous_completed_state_2",
        "previous_completed_state_3",
        "previous_completed_state_4",
        "previous_primitive_loop_1",
        "previous_primitive_loop_2",
        "bars_since_previous_loop",
        "same_loop_repeat_depth",
        "active_prefix_count",
        "active_primitive_prefixes",
        "active_repeat_prefixes",
        "active_composite_prefixes",
        "shortest_transitions_remaining",
        "highest_soft_prefix_probability",
        "soft_completion_probabilities",
    )
    eligibility = frame["structural_event_eligibility"].astype(bool)
    sequence_reason = (
        frame["source_sequence_missing_reason"]
        if "source_sequence_missing_reason" in frame
        else pd.Series(
            [
                None if bool(is_eligible) else "incomplete_or_ambiguous_in_session_source_sequence"
                for is_eligible in eligibility
            ],
            index=frame.index,
            dtype="object",
        )
    )
    for field in structural_features:
        present = frame[field].notna()
        valid = eligibility & present
        frame[f"{field}__causal_valid"] = valid
        default_missing = (
            frame["soft_prefix_missing_reason"]
            if field in {"highest_soft_prefix_probability", "soft_completion_probabilities"}
            else pd.Series("insufficient_causal_structural_history", index=frame.index)
        )
        frame[f"{field}__missing_reason"] = [
            None
            if bool(is_valid)
            else sequence_missing
            if not bool(is_eligible)
            else feature_missing
            for is_valid, is_eligible, sequence_missing, feature_missing in zip(
                valid,
                eligibility,
                sequence_reason,
                default_missing,
                strict=True,
            )
        ]

    prefix_frame = (
        pd.DataFrame(prefix_rows)
        if prefix_rows
        else _empty_frame(
            (
                "decision_id",
                "semantic_loop_id",
                "primitive_loop_id",
                "orientation_id",
                "motif_type",
                "repeat_depth",
                "research_only",
                "execution_enabled",
                "order_placement",
                "broker_connected",
                "strategy_promotion",
            )
        )
    )
    completions = (
        pd.DataFrame(completion_rows)
        if completion_rows
        else _empty_frame(
            (
                "decision_id",
                "semantic_loop_id",
                "primitive_loop_id",
                "orientation_id",
                "motif_type",
                "repeat_depth",
                "tied_completion",
                "research_only",
                "execution_enabled",
                "order_placement",
                "broker_connected",
                "strategy_promotion",
            )
        )
    )
    return LoopEventLedgerBundle(
        decisions=frame,
        prefixes=prefix_frame,
        completions=completions,
        outcomes=pd.DataFrame(outcome_rows),
        legacy_targets=pd.DataFrame(legacy_rows),
        target_comparison=pd.DataFrame(comparison_rows),
    )


__all__ = [
    "LoopEventLedgerBundle",
    "adapt_legacy_overlapping_target_panel",
    "adapt_legacy_run_ledger",
    "build_loop_event_ledgers",
    "compare_legacy_targets_to_v2_outcomes",
    "session_source_is_complete",
]
