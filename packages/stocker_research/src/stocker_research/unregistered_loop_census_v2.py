"""Uncapped minimal first-closure reconstruction for semantic loop research V2."""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from stocker_research.semantic_loop_dictionary_v2 import (
    SemanticPathIdentity,
    decompose_semantic_path,
    safety_flags,
)


@dataclass(frozen=True, slots=True)
class _Closure:
    event_index: int
    start_event_index: int
    event_bar_ordinal: int
    start_bar_ordinal: int
    event_timestamp: pd.Timestamp
    event_available_timestamp: pd.Timestamp
    start_timestamp: pd.Timestamp
    identity: SemanticPathIdentity
    repeat_depth: int
    previous_same_event_index: int | None
    previous_same_bar_ordinal: int | None
    previous_same_timestamp: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class UnregisteredVocabularyBundle:
    """Detailed structural vocabulary summaries for old unregistered outcomes."""

    primitive_census: pd.DataFrame
    repeat_census: pd.DataFrame
    composite_census: pd.DataFrame
    length_distribution: pd.DataFrame
    concentration: pd.DataFrame
    tie_census: pd.DataFrame


def _required_columns() -> set[str]:
    return {
        "decision_id",
        "symbol",
        "session",
        "bar_ordinal",
        "bar_start_timestamp",
        "bar_complete_timestamp",
        "decision_timestamp",
        "hard_state_legacy",
        "structural_event_eligibility",
    }


def _empty_event_fields() -> dict[str, Any]:
    return {
        "event_timestamp": None,
        "event_available_timestamp": None,
        "start_state_event_timestamp": None,
        "event_bar_ordinal": None,
        "full_closed_path": [],
        "open_core": [],
        "transition_length": None,
        "primitive_core": [],
        "primitive_transition_length": None,
        "primitive_loop_id": None,
        "semantic_loop_id": None,
        "semantic_motif_id": None,
        "repeat_depth": None,
        "motif_type": None,
        "component_primitive_ids": [],
        "component_boundaries": [],
        "orientation": [],
        "reverse_path_id": None,
        "initiated_before_decision": False,
        "initiated_after_decision": False,
        "initiated_at_decision": False,
        "active_prefix_length_at_decision": 0,
        "transitions_remaining_at_decision": None,
        "bars_until_completion": None,
        "state_events_until_completion": None,
        "current_repeat_depth": None,
        "previous_same_primitive_completion_timestamp": None,
        "bars_since_previous_same_primitive": None,
        "transitions_since_previous_same_primitive": None,
        "is_consecutive_repeat": False,
        "nested_repeat_ids": [],
        "nested_composite_ids": [],
        "is_same_as_previous_primitive": None,
        "earliest_composite_completion": None,
        "first_component_completion": None,
        "final_component_completion": None,
        "component_completion_timestamps": [],
        "earlier_primitive_completion_already_occurred": False,
        "composite_adds_information_beyond_primitive_sequence": False,
        "legacy_overlapping_positive_labels": [],
        "event_key": None,
        "legacy_dictionary_membership": False,
        "current_v2_dictionary_membership": False,
        "event_confidence_representation": "legacy_hard_map_deterministic",
    }


def _repeat_ids(identity: SemanticPathIdentity, repeat_depth: int) -> list[str]:
    if repeat_depth <= 1:
        return []
    primitive_text = identity.primitive_loop_id.removeprefix("loop_p_")
    return [f"loop_r{depth}_{primitive_text}" for depth in range(2, repeat_depth + 1)]


def _session_closures(
    event_states: list[int],
    event_bars: list[int],
    event_timestamps: list[pd.Timestamp],
    event_available: list[pd.Timestamp],
) -> tuple[list[int], dict[int, _Closure]]:
    last_state_index: dict[int, int] = {}
    previous_by_primitive: dict[str, _Closure] = {}
    previous_primitive: str | None = None
    consecutive_depth = 0
    indices: list[int] = []
    closures: dict[int, _Closure] = {}
    for event_index, state in enumerate(event_states):
        if state not in last_state_index:
            last_state_index[state] = event_index
            continue
        start_index = last_state_index[state]
        path = tuple(event_states[start_index : event_index + 1])
        identity = decompose_semantic_path(path)
        primitive_id = identity.primitive_loop_id
        consecutive_depth = consecutive_depth + 1 if primitive_id == previous_primitive else 1
        previous = previous_by_primitive.get(primitive_id)
        closure = _Closure(
            event_index=event_index,
            start_event_index=start_index,
            event_bar_ordinal=event_bars[event_index],
            start_bar_ordinal=event_bars[start_index],
            event_timestamp=event_timestamps[event_index],
            event_available_timestamp=event_available[event_index],
            start_timestamp=event_timestamps[start_index],
            identity=identity,
            repeat_depth=consecutive_depth,
            previous_same_event_index=previous.event_index if previous else None,
            previous_same_bar_ordinal=previous.event_bar_ordinal if previous else None,
            previous_same_timestamp=previous.event_timestamp if previous else None,
        )
        closures[event_index] = closure
        indices.append(event_index)
        previous_by_primitive[primitive_id] = closure
        previous_primitive = primitive_id
        last_state_index[state] = event_index
    return indices, closures


def _identity_value(row: pd.Series, key: str, default: Any = None) -> Any:
    return row[key] if key in row and pd.notna(row[key]) else default


def reconstruct_first_events(
    decisions: pd.DataFrame,
    *,
    horizon_bars: int,
    legacy_dictionary_ids: Iterable[str] = (),
    current_dictionary_ids: Iterable[str] = (),
    decision_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Assign one uncapped primitive-first structural outcome to every decision."""

    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if missing := sorted(_required_columns().difference(decisions.columns)):
        raise ValueError(f"decision surface lacks first-event fields: {missing}")
    if decisions.empty or decisions["decision_id"].duplicated().any():
        raise ValueError("decision surface must be nonempty with unique decision IDs")
    legacy_ids = frozenset(str(value) for value in legacy_dictionary_ids)
    current_ids = frozenset(str(value) for value in current_dictionary_ids)
    requested_ids = (
        None if decision_ids is None else frozenset(str(value) for value in decision_ids)
    )
    if requested_ids is not None and not requested_ids.issubset(
        frozenset(decisions["decision_id"].astype(str))
    ):
        raise ValueError("requested first-event decisions are absent from the source surface")
    frame = decisions.sort_values(
        ["symbol", "session", "bar_ordinal", "decision_id"], kind="mergesort"
    ).reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for (symbol, session), group in frame.groupby(["symbol", "session"], sort=False):
        local = group.reset_index(drop=True)
        states = local["hard_state_legacy"].astype(int).to_numpy()
        event_mask = np.r_[True, states[1:] != states[:-1]]
        event_positions = np.flatnonzero(event_mask)
        event_for_bar = np.cumsum(event_mask, dtype=int) - 1
        event_states = states[event_positions].astype(int).tolist()
        event_bars = local.iloc[event_positions]["bar_ordinal"].astype(int).tolist()
        event_timestamps = [
            pd.Timestamp(value) for value in local.iloc[event_positions]["bar_start_timestamp"]
        ]
        event_available = [
            pd.Timestamp(value) for value in local.iloc[event_positions]["bar_complete_timestamp"]
        ]
        closure_indices, closures = _session_closures(
            event_states, event_bars, event_timestamps, event_available
        )
        session_end = int(local["bar_ordinal"].max())

        closure_history: list[_Closure] = []
        history_by_event: dict[int, tuple[_Closure, ...]] = {}
        for event_index in range(len(event_states)):
            if event_index in closures:
                closure_history.append(closures[event_index])
            history_by_event[event_index] = tuple(closure_history)

        for local_index, decision in local.iterrows():
            local_position = int(cast(Any, local_index))
            decision_id = str(decision["decision_id"])
            if requested_ids is not None and decision_id not in requested_ids:
                continue
            decision_bar = int(decision["bar_ordinal"])
            current_event = int(event_for_bar[local_position])
            source_available = bool(_identity_value(decision, "source_available", True))
            eligible = bool(decision["structural_event_eligibility"])
            history = history_by_event[current_event]
            previous = history[-1] if history else None
            previous_two = history[-2] if len(history) >= 2 else None
            common: dict[str, Any] = {
                "decision_id": decision_id,
                "symbol": str(symbol),
                "session": str(session),
                "decision_timestamp": pd.Timestamp(decision["decision_timestamp"]),
                "bar_ordinal": decision_bar,
                "source_completeness": source_available and eligible,
                "source_missing_reason": _identity_value(
                    decision, "source_sequence_missing_reason"
                ),
                "previous_completed_primitive_loop": (
                    previous.identity.primitive_loop_id if previous else None
                ),
                "previous_two_completed_primitive_loops": (
                    [
                        value.identity.primitive_loop_id
                        for value in (previous_two, previous)
                        if value is not None
                    ]
                ),
                "same_primitive_repeat_depth": previous.repeat_depth if previous else 0,
                "bars_since_previous_primitive_completion": (
                    decision_bar - previous.event_bar_ordinal if previous else None
                ),
                "state_events_since_previous_primitive_completion": (
                    current_event - previous.event_index if previous else None
                ),
                "run_id": _identity_value(decision, "run_id"),
                "git_sha": _identity_value(decision, "git_sha"),
                "contract_hash": _identity_value(decision, "contract_hash"),
                "data_snapshot_hash": _identity_value(decision, "data_snapshot_hash"),
                "dictionary_version": _identity_value(decision, "dictionary_version"),
                "dictionary_hash": _identity_value(decision, "dictionary_hash"),
                "state_model_version": _identity_value(decision, "state_model_version"),
                "source_artifact": "causal_completed_bar_decisions.parquet",
                "source_hash": _identity_value(decision, "source_artifact_hash"),
                **safety_flags(),
            }
            if not source_available:
                rows.append(
                    {
                        **common,
                        "primary_event": "UNAVAILABLE_SOURCE",
                        **_empty_event_fields(),
                    }
                )
                continue
            if not eligible:
                rows.append(
                    {
                        **common,
                        "primary_event": "UNAVAILABLE_STRUCTURAL_GAP",
                        **_empty_event_fields(),
                    }
                )
                continue

            position = bisect.bisect_right(closure_indices, current_event)
            closure = (
                closures[closure_indices[position]] if position < len(closure_indices) else None
            )
            if closure is None or closure.event_bar_ordinal - decision_bar > horizon_bars:
                primary = (
                    "SESSION_END"
                    if session_end <= decision_bar + horizon_bars
                    else "NO_LOOP_WITHIN_HORIZON"
                )
                rows.append({**common, "primary_event": primary, **_empty_event_fields()})
                continue

            identity = closure.identity
            bars_until = closure.event_bar_ordinal - decision_bar
            if bars_until <= 0:
                raise AssertionError("first future closure is not strictly after the decision")
            active_length = (
                current_event - closure.start_event_index + 1
                if closure.start_event_index <= current_event
                else 0
            )
            nested_composites = (
                [identity.semantic_motif_id] if identity.motif_type.value == "composite" else []
            )
            component_completion_timestamps = [
                event_timestamps[closure.start_event_index + boundary_end]
                for _, boundary_end in identity.component_boundaries
            ]
            is_composite = identity.motif_type.value == "composite"
            earlier_component_completed = bool(
                is_composite
                and identity.component_boundaries
                and any(
                    boundary_end < len(identity.full_closed_path) - 1
                    for _, boundary_end in identity.component_boundaries
                )
            )
            rows.append(
                {
                    **common,
                    "primary_event": identity.primitive_loop_id,
                    "event_timestamp": closure.event_timestamp,
                    "event_available_timestamp": closure.event_available_timestamp,
                    "start_state_event_timestamp": closure.start_timestamp,
                    "event_bar_ordinal": closure.event_bar_ordinal,
                    "full_closed_path": list(identity.full_closed_path),
                    "open_core": list(identity.open_core),
                    "transition_length": identity.transition_length,
                    "primitive_core": list(identity.canonical_primitive_core),
                    "primitive_transition_length": identity.primitive_transition_length,
                    "primitive_loop_id": identity.primitive_loop_id,
                    "semantic_loop_id": identity.primitive_loop_id,
                    "semantic_motif_id": identity.semantic_motif_id,
                    "repeat_depth": identity.repeat_depth,
                    "motif_type": identity.motif_type.value,
                    "component_primitive_ids": list(identity.component_primitive_ids),
                    "component_boundaries": [
                        list(value) for value in identity.component_boundaries
                    ],
                    "orientation": list(identity.orientation),
                    "reverse_path_id": identity.reverse_path_id,
                    "initiated_before_decision": closure.start_bar_ordinal < decision_bar,
                    "initiated_after_decision": closure.start_bar_ordinal > decision_bar,
                    "initiated_at_decision": closure.start_bar_ordinal == decision_bar,
                    "active_prefix_length_at_decision": active_length,
                    "transitions_remaining_at_decision": closure.event_index - current_event,
                    "bars_until_completion": bars_until,
                    "state_events_until_completion": closure.event_index - current_event,
                    "current_repeat_depth": closure.repeat_depth,
                    "previous_same_primitive_completion_timestamp": closure.previous_same_timestamp,
                    "bars_since_previous_same_primitive": (
                        closure.event_bar_ordinal - closure.previous_same_bar_ordinal
                        if closure.previous_same_bar_ordinal is not None
                        else None
                    ),
                    "transitions_since_previous_same_primitive": (
                        closure.event_index - closure.previous_same_event_index
                        if closure.previous_same_event_index is not None
                        else None
                    ),
                    "is_consecutive_repeat": closure.repeat_depth > 1,
                    "nested_repeat_ids": _repeat_ids(identity, closure.repeat_depth),
                    "nested_composite_ids": nested_composites,
                    "is_same_as_previous_primitive": bool(
                        previous is not None
                        and previous.identity.primitive_loop_id == identity.primitive_loop_id
                    ),
                    "earliest_composite_completion": (
                        closure.event_timestamp if is_composite else None
                    ),
                    "first_component_completion": (
                        component_completion_timestamps[0]
                        if component_completion_timestamps
                        else None
                    ),
                    "final_component_completion": (
                        component_completion_timestamps[-1]
                        if component_completion_timestamps
                        else None
                    ),
                    "component_completion_timestamps": component_completion_timestamps,
                    "earlier_primitive_completion_already_occurred": earlier_component_completed,
                    "composite_adds_information_beyond_primitive_sequence": bool(
                        is_composite and len(identity.component_primitive_ids) > 1
                    ),
                    "legacy_overlapping_positive_labels": [],
                    "event_key": f"{symbol}|{session}|{closure.event_index}",
                    "legacy_dictionary_membership": identity.primitive_loop_id in legacy_ids,
                    "current_v2_dictionary_membership": identity.primitive_loop_id in current_ids,
                    "event_confidence_representation": "legacy_hard_map_deterministic",
                }
            )

    output = pd.DataFrame(rows)
    expected_rows = len(frame) if requested_ids is None else len(requested_ids)
    if len(output) != expected_rows or output["decision_id"].duplicated().any():
        raise AssertionError("first-event reconstruction is not one-to-one with decisions")
    return output


def _share_metrics(counts: pd.Series) -> dict[str, float]:
    ordered = counts.sort_values(ascending=False)
    total = float(ordered.sum())
    probabilities = ordered.to_numpy(dtype=float) / total if total else np.asarray([])
    return {
        "unique_primitive_roots": float(len(ordered)),
        "top_one_coverage": float(probabilities[:1].sum()),
        "top_five_coverage": float(probabilities[:5].sum()),
        "top_ten_coverage": float(probabilities[:10].sum()),
        "top_twenty_coverage": float(probabilities[:20].sum()),
        "top_thirty_two_coverage": float(probabilities[:32].sum()),
        "top_sixty_four_coverage": float(probabilities[:64].sum()),
        "herfindahl_concentration": float(np.square(probabilities).sum()),
        "entropy_nats": float(
            -(probabilities * np.log(probabilities)).sum() if len(probabilities) else 0.0
        ),
    }


def summarize_unregistered_vocabulary(events: pd.DataFrame) -> UnregisteredVocabularyBundle:
    """Summarize an already reconstructed old-UNREGISTERED decision population."""

    required = {
        "decision_id",
        "symbol",
        "session",
        "decision_timestamp",
        "primitive_loop_id",
        "semantic_motif_id",
        "primitive_transition_length",
        "transition_length",
        "orientation",
        "motif_type",
        "current_repeat_depth",
        "component_primitive_ids",
        "event_key",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"unregistered vocabulary missing fields: {sorted(missing)}")
    if events.empty or events["decision_id"].duplicated().any():
        raise ValueError("unregistered vocabulary requires unique reconstructed decisions")
    frame = events.copy()
    frame["_month"] = pd.to_datetime(frame["decision_timestamp"], utc=True).dt.strftime("%Y-%m")
    frame["_orientation_key"] = frame["orientation"].map(
        lambda path: "-".join(str(int(value)) for value in path)
    )
    primitive_rows: list[dict[str, Any]] = []
    for primitive_id, group in frame.groupby("primitive_loop_id", sort=True):
        primitive_rows.append(
            {
                "primitive_loop_id": primitive_id,
                "semantic_loop_id": primitive_id,
                "decision_count": len(group),
                "independent_event_count": int(group["event_key"].nunique()),
                "oriented_route_count": int(group["_orientation_key"].nunique()),
                "transition_length": int(group["primitive_transition_length"].iloc[0]),
                "stock_breadth": int(group["symbol"].nunique()),
                "session_breadth": int(
                    (group["symbol"].astype(str) + "|" + group["session"].astype(str)).nunique()
                ),
                "month_breadth": int(group["_month"].nunique()),
                "top_stock_share": float(group["symbol"].value_counts().iloc[0] / len(group)),
                "top_month_share": float(group["_month"].value_counts().iloc[0] / len(group)),
                **safety_flags(),
            }
        )
    primitive_census = pd.DataFrame.from_records(primitive_rows).sort_values(
        ["decision_count", "primitive_loop_id"], ascending=[False, True]
    )
    repeat_census = (
        frame.assign(
            is_same_primitive_repeat=frame["current_repeat_depth"].fillna(1).astype(int).gt(1)
        )
        .groupby(["is_same_primitive_repeat", "current_repeat_depth"], dropna=False)
        .agg(
            decision_count=("decision_id", "size"),
            primitive_roots=("primitive_loop_id", "nunique"),
        )
        .reset_index()
    )
    composites = frame.loc[frame["motif_type"].eq("composite")]
    if composites.empty:
        composite_census = pd.DataFrame(
            columns=[
                "semantic_motif_id",
                "component_primitive_ids",
                "full_transition_length",
                "decision_count",
            ]
        )
    else:
        composite_census = (
            composites.assign(
                _components=composites["component_primitive_ids"].map(
                    lambda values: "|".join(str(value) for value in values)
                )
            )
            .groupby(["semantic_motif_id", "_components", "transition_length"], dropna=False)
            .agg(decision_count=("decision_id", "size"))
            .reset_index()
            .rename(
                columns={
                    "_components": "component_primitive_ids",
                    "transition_length": "full_transition_length",
                }
            )
            .sort_values(["decision_count", "semantic_motif_id"], ascending=[False, True])
        )
    length_distribution = (
        frame.groupby(["primitive_transition_length", "transition_length", "motif_type"])
        .agg(decision_count=("decision_id", "size"))
        .reset_index()
        .sort_values(["primitive_transition_length", "transition_length", "motif_type"])
    )
    concentration_records = [
        {
            "scope": "all_unregistered",
            "group": "all",
            **_share_metrics(frame["primitive_loop_id"].value_counts()),
        }
    ]
    for scope, field in (
        ("stock", "symbol"),
        ("month", "_month"),
        ("clock_phase", "clock_phase"),
        ("state", "hard_state_legacy"),
    ):
        if field not in frame:
            continue
        for group_name, group in frame.groupby(field, dropna=False, sort=True):
            concentration_records.append(
                {
                    "scope": scope,
                    "group": str(group_name),
                    **_share_metrics(group["primitive_loop_id"].value_counts()),
                }
            )
    top_stocks = frame["symbol"].value_counts().index.tolist()
    for removed in (1, 5):
        retained = frame.loc[~frame["symbol"].isin(top_stocks[:removed])]
        concentration_records.append(
            {
                "scope": "stock_deletion",
                "group": f"remove_top_{removed}",
                **_share_metrics(retained["primitive_loop_id"].value_counts()),
            }
        )
    concentration = pd.DataFrame.from_records(concentration_records)
    tie_census = pd.DataFrame(
        [
            {
                "tie_class": "DISTINCT_PRIMITIVE_TIE",
                "count": int(
                    frame.get("primary_event", pd.Series(dtype=str))
                    .eq("DISTINCT_PRIMITIVE_TIE")
                    .sum()
                ),
                "status": "earliest uncapped closures deterministically reduced",
                **safety_flags(),
            }
        ]
    )
    for output in (repeat_census, composite_census, length_distribution, concentration):
        for key, value in safety_flags().items():
            output[key] = cast(Any, value)
    return UnregisteredVocabularyBundle(
        primitive_census=primitive_census,
        repeat_census=repeat_census,
        composite_census=composite_census,
        length_distribution=length_distribution,
        concentration=concentration,
        tie_census=tie_census,
    )


__all__ = [
    "UnregisteredVocabularyBundle",
    "reconstruct_first_events",
    "summarize_unregistered_vocabulary",
]
