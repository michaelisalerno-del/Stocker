"""Research-only helpers for the corrected behavioural-trajectory screen V0.1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

ANCHORS_BY_CHECKPOINT: Final[dict[int, tuple[int, int, int]]] = {
    6: (2, 4, 6),
    12: (4, 8, 12),
    24: (8, 16, 24),
    36: (12, 24, 36),
}
TRAJECTORY_INTERACTION_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("transition_probability_x_arousal_change", "transition_probability", "arousal_change"),
    ("posterior_entropy_x_frustration_change", "posterior_entropy", "frustration_change"),
    ("top_second_margin_x_conviction_change", "top_second_margin", "conviction_change"),
    (
        "transition_probability_x_signed_pressure_acceleration",
        "transition_probability",
        "signed_pressure_acceleration",
    ),
    (
        "posterior_entropy_x_tension_acceleration",
        "posterior_entropy",
        "tension_acceleration",
    ),
    (
        "top_state_probability_x_signed_exhaustion_change",
        "top_state_probability",
        "signed_exhaustion_change",
    ),
)
TRAJECTORY_INTERACTION_FEATURES: Final[tuple[str, ...]] = tuple(
    specification[0] for specification in TRAJECTORY_INTERACTION_SPECS
)
REGISTERED_RAW_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"REGISTERED_PRIMITIVE", "REGISTERED_REPEAT", "REGISTERED_COMPOSITE"}
)
SCREEN_SCOPES: Final[tuple[str, ...]] = ("pooled", "opening", "later", "late_no_open")
PRIMARY_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        "trajectory_signal_feasible_pooled_and_late",
        "trajectory_signal_feasible_pooled_only",
        "trajectory_signal_feasible_late_only",
        "trajectory_signal_feasible_opening_only",
        "trajectory_main_effects_only",
        "descriptive_trajectory_structure_only",
        "no_behavioural_trajectory_increment",
    }
)


@dataclass(frozen=True, slots=True)
class SessionBootstrapDraw:
    """One deterministic whole-session bootstrap resample."""

    sampled_sessions: tuple[str, ...]
    row_indices: NDArray[np.int64]


def trajectory_anchors(checkpoint: int) -> tuple[int, int, int]:
    """Return the fixed even completed-bar anchor triplet for a checkpoint."""

    try:
        anchors = ANCHORS_BY_CHECKPOINT[int(checkpoint)]
    except KeyError as error:
        raise ValueError(f"unsupported trajectory checkpoint: {checkpoint}") from error
    if any(anchor < 2 or anchor % 2 for anchor in anchors):
        raise AssertionError("trajectory anchors must be positive even completed-bar counts")
    return anchors


def causal_anchor_prefix(
    bars: pd.DataFrame,
    *,
    completed_bar_count: int,
    decision_available_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Return exactly one even completed-bar prefix available by the decision."""

    required = {"bar_start_timestamp", "bar_complete_timestamp"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"anchor bar columns missing: {missing}")
    count = int(completed_bar_count)
    if count < 2 or count % 2:
        raise ValueError("anchor requires an even number of completed bars")
    if len(bars) < count:
        raise ValueError("anchor has fewer bars than requested")
    ordered = bars.copy()
    ordered["bar_start_timestamp"] = pd.to_datetime(
        ordered["bar_start_timestamp"], utc=True, errors="raise"
    )
    ordered["bar_complete_timestamp"] = pd.to_datetime(
        ordered["bar_complete_timestamp"], utc=True, errors="raise"
    )
    ordered = ordered.sort_values("bar_start_timestamp", kind="mergesort").reset_index(drop=True)
    prefix = ordered.iloc[:count].copy()
    available = pd.Timestamp(decision_available_timestamp)
    if available.tzinfo is None:
        available = available.tz_localize("UTC")
    else:
        available = available.tz_convert("UTC")
    if bool(prefix["bar_complete_timestamp"].gt(available).any()):
        raise ValueError("future bar entered an earlier behavioural anchor")
    return prefix.reset_index(drop=True)


def trajectory_feature_values(
    earliest: float,
    middle: float,
    final: float,
) -> dict[str, float | int]:
    """Calculate the fixed primary and descriptive trajectory forms."""

    e0, e1, e2 = float(earliest), float(middle), float(final)
    if not all(math.isfinite(value) for value in (e0, e1, e2)):
        raise ValueError("trajectory anchors must be finite")
    first_change = e1 - e0
    recent_change = e2 - e1
    reversal = int(
        first_change != 0.0
        and recent_change != 0.0
        and ((first_change > 0.0) != (recent_change > 0.0))
    )
    persistence = 1 if e0 < e1 < e2 else -1 if e0 > e1 > e2 else 0
    return {
        "change": e2 - e0,
        "acceleration": recent_change - first_change,
        "reversal": reversal,
        "recent_change": recent_change,
        "monotonic_persistence": persistence,
        "peak_displacement": e2 - max(e0, e1, e2),
    }


def build_trajectory_regime_interactions(
    frame: pd.DataFrame,
    *,
    fit_bounds: bool = False,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Build exactly six trajectory/regime products with optional frozen clipping."""

    if fit_bounds and bounds is not None:
        raise ValueError("cannot fit and supply interaction bounds simultaneously")
    required = {
        column
        for _, regime_column, trajectory_column in TRAJECTORY_INTERACTION_SPECS
        for column in (regime_column, trajectory_column)
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"trajectory interaction columns missing: {missing}")
    result = pd.DataFrame(index=frame.index)
    for feature, regime_column, trajectory_column in TRAJECTORY_INTERACTION_SPECS:
        result[feature] = frame[regime_column].astype(float) * frame[trajectory_column].astype(
            float
        )
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("trajectory interactions must be finite")
    if fit_bounds:
        fitted = {
            feature: (
                float(result[feature].quantile(0.01, interpolation="linear")),
                float(result[feature].quantile(0.99, interpolation="linear")),
            )
            for feature in TRAJECTORY_INTERACTION_FEATURES
        }
    elif bounds is None:
        fitted = {}
    else:
        fitted = {feature: (float(value[0]), float(value[1])) for feature, value in bounds.items()}
        if set(fitted) != set(TRAJECTORY_INTERACTION_FEATURES):
            raise ValueError("interaction bounds differ from the six-feature manifest")
    for feature, (lower, upper) in fitted.items():
        result[feature] = result[feature].clip(lower=lower, upper=upper)
    return result.loc[:, list(TRAJECTORY_INTERACTION_FEATURES)], fitted


def structural_history_controls(
    *,
    registered_completion_bar_ordinals: Sequence[int],
    decision_bar_ordinal: int,
    active_registered_prefix_count: int,
) -> dict[str, float | int]:
    """Create causal registered-completion and active-prefix controls."""

    decision = int(decision_bar_ordinal)
    if decision < 0 or active_registered_prefix_count < 0:
        raise ValueError("structural-history ordinals and counts must be non-negative")
    known = sorted(
        int(value) for value in registered_completion_bar_ordinals if int(value) <= decision
    )
    if any(value < 0 for value in known):
        raise ValueError("registered completion ordinal must be non-negative")
    return {
        "registered_completion_count_before_decision": len(known),
        "bars_since_last_registered_completion": float(decision - known[-1]) if known else 0.0,
        "bars_since_last_registered_completion_missing": int(not known),
        "active_registered_prefix_count_at_decision": int(active_registered_prefix_count),
    }


def phase_label(checkpoint: int) -> str:
    """Classify one preregistered checkpoint as opening or later phase."""

    value = int(checkpoint)
    if value in (6, 12):
        return "OPENING_PHASE"
    if value in (24, 36):
        return "LATER_PHASE"
    raise ValueError(f"unsupported checkpoint: {checkpoint}")


def late_loop_subgroup(
    checkpoint: int,
    *,
    opening_registered_completion_count: int,
) -> str | None:
    """Assign the two preregistered later-session opening-history groups."""

    if opening_registered_completion_count < 0:
        raise ValueError("opening registered-completion count must be non-negative")
    if phase_label(checkpoint) == "OPENING_PHASE":
        return None
    if opening_registered_completion_count == 0:
        return "LATE_NO_OPEN_REGISTERED_LOOP"
    return "LATE_AFTER_OPEN_REGISTERED_LOOP"


def map_six_bar_structural_target(raw_outcome: str, *, horizon_bars: int) -> str | None:
    """Map one frozen first-event outcome into the fixed three-class target."""

    if horizon_bars != 6:
        raise ValueError("the structural target requires exactly a six-bar horizon")
    label = str(raw_outcome)
    if label in REGISTERED_RAW_OUTCOMES:
        return "REGISTERED_COMPLETION"
    if label in {"UNREGISTERED_LOOP", "NO_REGISTERED_COMPLETION"}:
        return label
    if label in {"TIED_REGISTERED_COMPLETION", "SOURCE_UNAVAILABLE"}:
        return None
    raise ValueError(f"unknown structural outcome: {raw_outcome}")


def session_block_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    session_column: str = "session",
) -> tuple[SessionBootstrapDraw, ...]:
    """Return fixed-seed whole-session row-index draws without fitting models."""

    draw_count = int(draws)
    if draw_count < 1 or draw_count > 25:
        raise ValueError("session bootstrap requires between 1 and 25 draws")
    if session_column not in frame.columns:
        raise ValueError(f"session column missing: {session_column}")
    session_values = frame[session_column].astype(str)
    sessions = tuple(sorted(session_values.unique().tolist()))
    if not sessions:
        raise ValueError("session bootstrap requires at least one session")
    positions = {
        session: np.flatnonzero(session_values.to_numpy() == session).astype(np.int64)
        for session in sessions
    }
    rng = np.random.default_rng(int(seed))
    result: list[SessionBootstrapDraw] = []
    for _ in range(draw_count):
        sampled = tuple(
            str(value) for value in rng.choice(sessions, size=len(sessions), replace=True)
        )
        row_indices = np.concatenate([positions[session] for session in sampled]).astype(
            np.int64,
            copy=False,
        )
        result.append(SessionBootstrapDraw(sampled, row_indices))
    return tuple(result)


def permute_trajectory_bundle_within_slates(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    seed: int,
    slate_column: str = "slate_id",
) -> pd.DataFrame:
    """Permute complete trajectory bundles within each session/checkpoint slate."""

    feature_names = tuple(features)
    if not feature_names:
        raise ValueError("trajectory bundle must contain at least one feature")
    missing = sorted({slate_column, *feature_names}.difference(frame.columns))
    if missing:
        raise ValueError(f"trajectory permutation columns missing: {missing}")
    result = frame.copy()
    source = frame.loc[:, list(feature_names)].to_numpy(copy=True)
    target_columns = [list(result.columns).index(feature) for feature in feature_names]
    rng = np.random.default_rng(int(seed))
    for positions in frame.groupby(slate_column, sort=True, observed=True).indices.values():
        target_positions = np.asarray(positions, dtype=np.int64)
        source_positions = target_positions[rng.permutation(len(target_positions))]
        result.iloc[target_positions, target_columns] = source[source_positions]
    return result


def reject_protected_dates(
    frame: pd.DataFrame,
    *,
    session_column: str = "session",
    protected_start: str = "2025-08-23",
) -> None:
    """Fail closed if a protected session is materialised."""

    if session_column not in frame.columns:
        raise ValueError(f"session column missing: {session_column}")
    sessions = pd.to_datetime(frame[session_column], utc=True, errors="raise")
    boundary = pd.Timestamp(protected_start, tz="UTC")
    if bool(sessions.ge(boundary).any()):
        raise ValueError("protected date boundary was materialised")


def decide_quick_screen(
    *,
    t1_positive: Mapping[str, bool],
    t2_positive: Mapping[str, bool],
    point_estimate_improves: bool,
) -> str:
    """Apply the preregistered non-blocked quick-screen decision precedence."""

    for name, values in (("T1", t1_positive), ("T2", t2_positive)):
        if set(values) != set(SCREEN_SCOPES):
            raise ValueError(f"{name} decision scopes differ from the preregistration")
    if any(t1_positive.values()) and not any(t2_positive.values()):
        return "trajectory_main_effects_only"
    combined = {scope: bool(t1_positive[scope] or t2_positive[scope]) for scope in SCREEN_SCOPES}
    if any(t2_positive.values()):
        if combined["pooled"] and (combined["later"] or combined["late_no_open"]):
            return "trajectory_signal_feasible_pooled_and_late"
        if combined["pooled"]:
            return "trajectory_signal_feasible_pooled_only"
        if combined["later"] or combined["late_no_open"]:
            return "trajectory_signal_feasible_late_only"
        if combined["opening"]:
            return "trajectory_signal_feasible_opening_only"
    decision = (
        "descriptive_trajectory_structure_only"
        if point_estimate_improves
        else "no_behavioural_trajectory_increment"
    )
    if decision not in PRIMARY_DECISIONS:
        raise AssertionError("unregistered quick-screen decision")
    return decision


__all__ = [
    "ANCHORS_BY_CHECKPOINT",
    "PRIMARY_DECISIONS",
    "SCREEN_SCOPES",
    "SessionBootstrapDraw",
    "TRAJECTORY_INTERACTION_FEATURES",
    "TRAJECTORY_INTERACTION_SPECS",
    "build_trajectory_regime_interactions",
    "causal_anchor_prefix",
    "decide_quick_screen",
    "late_loop_subgroup",
    "map_six_bar_structural_target",
    "phase_label",
    "permute_trajectory_bundle_within_slates",
    "reject_protected_dates",
    "session_block_bootstrap_draws",
    "structural_history_controls",
    "trajectory_anchors",
    "trajectory_feature_values",
]
