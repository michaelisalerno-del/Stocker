"""Registered causal structural checkpoint construction."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from .model import oriented_paths

FIXED_BAR_CHECKPOINTS: tuple[int, ...] = (1, 2, 3, 6)


def _checkpoint_row(
    checkpoint_type: str,
    anchor_timestamp: pd.Timestamp,
    bars_since_anchor: int,
    observed: Sequence[int],
) -> dict[str, object]:
    timestamp = anchor_timestamp + pd.Timedelta(minutes=5 * (bars_since_anchor + 1))
    return {
        "checkpoint_type": checkpoint_type,
        "bars_since_anchor": int(bars_since_anchor),
        "checkpoint_timestamp": timestamp,
        "feature_max_availability_timestamp": timestamp,
        "observed_transitions_json": json.dumps(
            [int(value) for value in observed], separators=(",", ":")
        ),
    }


def build_registered_checkpoints(
    state_runs: pd.DataFrame,
    *,
    anchor_ordinal: int,
    terminal_ordinal: int,
    target_cycle: str,
    anchor_state: int,
    fixed_bars: Sequence[int] = FIXED_BAR_CHECKPOINTS,
) -> pd.DataFrame:
    """Build only checkpoints whose defining bar has completed by the freeze."""

    required = {"bar_ordinal", "state", "start_timestamp"}
    missing = required - set(state_runs.columns)
    if missing:
        raise ValueError(f"missing state-run columns: {sorted(missing)}")
    runs = state_runs.copy()
    runs["start_timestamp"] = pd.to_datetime(runs["start_timestamp"], utc=True, errors="raise")
    runs = runs.sort_values(["bar_ordinal", "start_timestamp"], kind="stable")
    anchor_rows = runs.loc[runs["bar_ordinal"].eq(anchor_ordinal)]
    if len(anchor_rows) != 1 or int(anchor_rows.iloc[0]["state"]) != int(anchor_state):
        raise ValueError("anchor state-run identity is ambiguous")
    anchor_timestamp = pd.Timestamp(anchor_rows.iloc[0]["start_timestamp"])
    transitions = runs.loc[
        runs["bar_ordinal"].gt(anchor_ordinal) & runs["bar_ordinal"].le(terminal_ordinal)
    ].copy()
    transition_rows = [cast(Any, row) for row in transitions.itertuples(index=False)][:2]

    rows: list[dict[str, object]] = [_checkpoint_row("anchor_freeze", anchor_timestamp, 0, ())]
    for bars in fixed_bars:
        bars_int = int(bars)
        if bars_int <= 0 or anchor_ordinal + bars_int > terminal_ordinal:
            continue
        observed = transitions.loc[
            transitions["bar_ordinal"].le(anchor_ordinal + bars_int), "state"
        ].astype(int)
        rows.append(
            _checkpoint_row(f"fixed_bar_{bars_int}", anchor_timestamp, bars_int, tuple(observed))
        )

    observed_states: list[int] = []
    for index, transition in enumerate(transition_rows, start=1):
        observed_states.append(int(transition.state))
        bars = int(transition.bar_ordinal) - anchor_ordinal
        rows.append(
            _checkpoint_row(
                "first_completed_transition" if index == 1 else "second_completed_transition",
                anchor_timestamp,
                bars,
                observed_states,
            )
        )

    target_paths = oriented_paths(target_cycle, anchor_state)
    expected = target_paths[0][1:] if len(target_paths) == 1 else ()
    if transition_rows and expected and int(transition_rows[0].state) != int(expected[0]):
        bars = int(transition_rows[0].bar_ordinal) - anchor_ordinal
        rows.append(
            _checkpoint_row(
                "first_incompatible_transition",
                anchor_timestamp,
                bars,
                (int(transition_rows[0].state),),
            )
        )
    if len(transition_rows) >= 2 and expected:
        first = int(transition_rows[0].state)
        second = int(transition_rows[1].state)
        bars = int(transition_rows[1].bar_ordinal) - anchor_ordinal
        if first == int(expected[0]) and second == int(expected[1]):
            rows.append(
                _checkpoint_row("exact_parent_completion", anchor_timestamp, bars, (first, second))
            )
        elif first == int(expected[0]):
            rows.append(
                _checkpoint_row("first_route_diversion", anchor_timestamp, bars, (first, second))
            )

    result = pd.DataFrame(rows)
    event_order = {
        "anchor_freeze": 0,
        "fixed_bar_1": 1,
        "fixed_bar_2": 2,
        "fixed_bar_3": 3,
        "fixed_bar_6": 4,
        "first_completed_transition": 5,
        "second_completed_transition": 6,
        "exact_parent_completion": 7,
        "first_route_diversion": 8,
        "first_incompatible_transition": 9,
    }
    result["_order"] = result["checkpoint_type"].map(event_order).fillna(99)
    return (
        result.sort_values(["checkpoint_timestamp", "_order"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
