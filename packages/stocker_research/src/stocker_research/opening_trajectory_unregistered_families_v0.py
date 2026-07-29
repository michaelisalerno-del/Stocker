"""Pure structural helpers for the opening unregistered-family quick screen.

The module operates only on frozen structural and behavioural inputs. It has no
economic, trading, broker, execution, or deployment surface.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from stocker_research.loop_dictionary_v2 import UnsupportedLoopError, decompose_closed_path

OPENING_ANCHORS: Final[dict[int, tuple[int, int, int]]] = {
    6: (2, 4, 6),
    12: (4, 8, 12),
}
BEHAVIOURS: Final[tuple[str, ...]] = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "signed_exhaustion",
)
TRAJECTORY_FORMS: Final[tuple[str, ...]] = ("change", "acceleration", "reversal")
PROTECTED_START: Final[pd.Timestamp] = pd.Timestamp("2025-08-23")
OTHER_FAMILY: Final[str] = "OTHER_UNREGISTERED_FAMILY"
MAX_TRANSITIONS: Final[int] = 8

MotifType = Literal["primitive_like", "repeat_like", "composite_like"]


@dataclass(frozen=True, slots=True)
class UnregisteredPathEvent:
    """One causal first unregistered completion within the fixed horizon."""

    full_path: tuple[int, ...]
    start_event_index: int
    completion_event_index: int
    start_bar_ordinal: int
    completion_bar_ordinal: int


@dataclass(frozen=True, slots=True)
class CanonicalUnregisteredPath:
    """Stable forward-rotation identity with explicit orientation metadata."""

    family_id: str
    canonical_path: tuple[int, ...]
    oriented_path: tuple[int, ...]
    orientation_id: str
    rotation_offset: int
    reverse_family_id: str
    reverse_orientation_equivalent: bool
    motif_type: MotifType
    repeat_depth: int
    transition_length: int
    revisit_count: int
    v2_semantic_id: str | None
    v2_compatible: bool


def opening_anchor_triplet(checkpoint: int) -> tuple[int, int, int]:
    """Return the preregistered corrected anchor triplet."""

    try:
        return OPENING_ANCHORS[int(checkpoint)]
    except KeyError as error:
        raise ValueError(f"checkpoint {checkpoint} is not an opening checkpoint") from error


def trajectory_feature_names() -> tuple[str, ...]:
    """Return the frozen 18-feature opening trajectory surface."""

    return tuple(f"{behaviour}_{form}" for behaviour in BEHAVIOURS for form in TRAJECTORY_FORMS)


def reject_protected_dates(frame: pd.DataFrame) -> None:
    """Fail when any materialised row reaches the protected boundary."""

    if "session" not in frame:
        raise ValueError("session column is required")
    sessions = pd.to_datetime(frame["session"], errors="raise")
    if bool(sessions.ge(PROTECTED_START).any()):
        raise ValueError("protected date 2025-08-23 or later materialised")


def opening_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the unique frozen opening population after strict validation."""

    required = {"symbol", "session", "decision_ordinal"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"opening population columns missing: {missing}")
    reject_protected_dates(frame)
    result = frame.loc[frame["decision_ordinal"].isin(OPENING_ANCHORS)].copy()
    keys = ["symbol", "session", "decision_ordinal"]
    if result.duplicated(keys).any():
        raise ValueError("opening population keys are not unique")
    if set(result["decision_ordinal"].astype(int)) != set(OPENING_ANCHORS):
        raise ValueError("both opening checkpoints are required")
    expected_clocks = {6: "10:00", 12: "10:30"}
    if "decision_time_america_new_york" in result:
        actual = result.groupby("decision_ordinal")["decision_time_america_new_york"].unique()
        for checkpoint, clock in expected_clocks.items():
            if actual[checkpoint].tolist() != [clock]:
                raise ValueError(f"checkpoint {checkpoint} does not occur at {clock}")
    return result.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)


def binary_targets(raw_outcomes: pd.Series) -> pd.DataFrame:
    """Map frozen raw outcomes to the two preregistered binary targets."""

    raw = raw_outcomes.astype(str)
    scoring = raw.isin({"UNREGISTERED_LOOP", "REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"})
    unregistered = pd.Series(np.nan, index=raw.index, dtype=float)
    unregistered.loc[scoring] = raw.loc[scoring].eq("UNREGISTERED_LOOP").astype(float)
    diagnostic = pd.Series(np.nan, index=raw.index, dtype=float)
    diagnostic_population = raw.isin({"REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"})
    diagnostic.loc[diagnostic_population] = (
        raw.loc[diagnostic_population].eq("REGISTERED_COMPLETION").astype(float)
    )
    return pd.DataFrame(
        {
            "unregistered_event": unregistered,
            "registered_completion": diagnostic,
        }
    )


def _compressed_events(
    bar_states: Sequence[int], bar_ordinals: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(bar_states) != len(bar_ordinals) or not bar_states:
        raise ValueError("bar states and ordinals must be non-empty and aligned")
    ordinals = tuple(int(value) for value in bar_ordinals)
    if any(right <= left for left, right in zip(ordinals[:-1], ordinals[1:], strict=True)):
        raise ValueError("bar ordinals must be strictly increasing")
    states: list[int] = []
    event_ordinals: list[int] = []
    for state, ordinal in zip(bar_states, ordinals, strict=True):
        value = int(state)
        if not states or states[-1] != value:
            states.append(value)
            event_ordinals.append(ordinal)
    return tuple(states), tuple(event_ordinals)


def first_unregistered_path(
    *,
    bar_states: Sequence[int],
    bar_ordinals: Sequence[int],
    decision_bar_ordinal: int,
    decision_event_index: int,
    registered_paths: Collection[tuple[int, ...]],
    horizon_bars: int = 6,
) -> UnregisteredPathEvent | None:
    """Reproduce the V2 earliest unregistered closed-path completion."""

    if horizon_bars != 6:
        raise ValueError("exactly one six-bar horizon is permitted")
    states, event_ordinals = _compressed_events(bar_states, bar_ordinals)
    if decision_event_index < 0 or decision_event_index >= len(states):
        raise ValueError("decision event index is outside the compressed path")
    causal_index = max(
        (index for index, ordinal in enumerate(event_ordinals) if ordinal <= decision_bar_ordinal),
        default=-1,
    )
    if causal_index != decision_event_index:
        raise ValueError("decision event index differs from the causal compressed path")
    registered = {tuple(int(state) for state in path) for path in registered_paths}
    candidates: list[UnregisteredPathEvent] = []
    for completion in range(decision_event_index + 1, len(states)):
        completion_bar = event_ordinals[completion]
        if completion_bar > decision_bar_ordinal + horizon_bars:
            break
        lower = max(0, completion - MAX_TRANSITIONS)
        for start in range(completion - 2, lower - 1, -1):
            if states[start] != states[completion]:
                continue
            path = states[start : completion + 1]
            if path in registered:
                continue
            candidates.append(
                UnregisteredPathEvent(
                    full_path=path,
                    start_event_index=start,
                    completion_event_index=completion,
                    start_bar_ordinal=event_ordinals[start],
                    completion_bar_ordinal=completion_bar,
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda event: (
            event.completion_bar_ordinal,
            event.completion_event_index,
            event.full_path,
        ),
    )


def _canonical_core(core: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    rotations = tuple(core[index:] + core[:index] for index in range(len(core)))
    canonical = min(rotations)
    return canonical, rotations.index(canonical)


def _primitive_root(core: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    for width in range(2, len(core) // 2 + 1):
        if len(core) % width:
            continue
        root = core[:width]
        depth = len(core) // width
        if root * depth == core:
            return root, depth
    return core, 1


def _fallback_motif(canonical_core: tuple[int, ...]) -> tuple[MotifType, int]:
    root, depth = _primitive_root(canonical_core)
    if depth > 1:
        return "repeat_like", depth
    if len(set(root)) == len(root):
        return "primitive_like", 1
    return "composite_like", 1


def _family_id(motif: MotifType, canonical_core: tuple[int, ...]) -> str:
    closed = canonical_core + (canonical_core[0],)
    return f"unregistered_{motif}__" + "-".join(str(state) for state in closed)


def canonical_unregistered_path(closed_path: Sequence[int]) -> CanonicalUnregisteredPath:
    """Canonicalise a causal closed path under the frozen V2-compatible contract."""

    path = tuple(int(state) for state in closed_path)
    if len(path) < 3 or path[0] != path[-1]:
        raise ValueError("an unregistered path must be closed with at least two transitions")
    if len(path) - 1 > MAX_TRANSITIONS:
        raise ValueError("unregistered path exceeds eight transitions")
    if any(left == right for left, right in zip(path[:-1], path[1:], strict=True)):
        raise ValueError("state-event paths cannot contain adjacent duplicate states")
    core = path[:-1]
    canonical_core, rotation_offset = _canonical_core(core)
    motif, repeat_depth = _fallback_motif(canonical_core)
    v2_id: str | None = None
    v2_compatible = False
    try:
        definition = decompose_closed_path(path)
        motif_by_v2_type: dict[str, MotifType] = {
            "primitive": "primitive_like",
            "repeat": "repeat_like",
            "composite": "composite_like",
        }
        motif = motif_by_v2_type[str(definition.motif_type)]
        repeat_depth = int(definition.repeat_depth)
        canonical_core = tuple(int(value) for value in definition.canonical_orientation[:-1])
        rotation_offset = next(
            index for index in range(len(core)) if core[index:] + core[:index] == canonical_core
        )
        v2_id = str(definition.semantic_loop_id)
        v2_compatible = True
    except (UnsupportedLoopError, StopIteration):
        pass
    canonical_path = canonical_core + (canonical_core[0],)
    family = _family_id(motif, canonical_core)
    reversed_core, _ = _canonical_core(tuple(reversed(path))[:-1])
    reverse_motif, _ = _fallback_motif(reversed_core)
    reverse_family = _family_id(reverse_motif, reversed_core)
    return CanonicalUnregisteredPath(
        family_id=family,
        canonical_path=canonical_path,
        oriented_path=path,
        orientation_id=f"{family}__o_" + "-".join(str(state) for state in path),
        rotation_offset=rotation_offset,
        reverse_family_id=reverse_family,
        reverse_orientation_equivalent=family == reverse_family,
        motif_type=motif,
        repeat_depth=repeat_depth,
        transition_length=len(core),
        revisit_count=len(core) - len(set(core)),
        v2_semantic_id=v2_id,
        v2_compatible=v2_compatible,
    )


def hidden_family_census(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure development support for stable canonical family identities."""

    required = {"family_id", "session", "symbol", "year_month"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"hidden-family census columns missing: {missing}")
    rows: list[dict[str, object]] = []
    for family, group in frame.groupby("family_id", sort=True):
        stock_counts = group.groupby("symbol", sort=True).size()
        outcomes = len(group)
        maximum_share = float(stock_counts.max() / outcomes)
        sessions = int(group["session"].nunique())
        stocks = int(group["symbol"].nunique())
        months = int(group["year_month"].nunique())
        eligible = bool(
            outcomes >= 30
            and sessions >= 20
            and stocks >= 8
            and months >= 4
            and maximum_share <= 0.30
        )
        rows.append(
            {
                "family_id": str(family),
                "outcomes": outcomes,
                "sessions": sessions,
                "stocks": stocks,
                "months": months,
                "maximum_stock_share": maximum_share,
                "eligible": eligible,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["outcomes", "family_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def select_hidden_families(census: pd.DataFrame, *, maximum: int = 4) -> tuple[str, ...]:
    """Freeze up to four eligible development families by support and stable ID."""

    required = {"family_id", "outcomes", "eligible"}
    missing = sorted(required.difference(census.columns))
    if missing:
        raise ValueError(f"family-selection columns missing: {missing}")
    eligible = census.loc[census["eligible"].astype(bool)].sort_values(
        ["outcomes", "family_id"], ascending=[False, True], kind="mergesort"
    )
    return tuple(eligible["family_id"].astype(str).head(maximum))


def pool_hidden_family(family_id: str, selected: Collection[str]) -> str:
    """Map unselected identities to the frozen other-family class."""

    value = str(family_id)
    return value if value in set(selected) else OTHER_FAMILY


def _permuted_columns(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    seed: int,
    include_year: bool,
) -> pd.DataFrame:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"permutation features missing: {missing}")
    if "slate_id" not in frame:
        raise ValueError("slate_id column is required")
    result = frame.copy()
    source = frame.loc[:, list(features)].to_numpy(copy=True)
    column_positions = result.columns.get_indexer(pd.Index(features)).astype(int).tolist()
    rng = np.random.default_rng(seed)
    grouping = ["slate_id"]
    if include_year:
        if "year" not in frame:
            raise ValueError("year column is required for trajectory-null permutation")
        grouping.insert(0, "year")
    for positions in frame.groupby(grouping, sort=True, observed=True).indices.values():
        target = np.asarray(positions, dtype=int)
        selected = target[rng.permutation(len(target))]
        result.iloc[target, column_positions] = source[selected]
    return result


def permute_trajectory_bundle_within_slates(
    frame: pd.DataFrame, features: Sequence[str], *, seed: int
) -> pd.DataFrame:
    """Permute all trajectory fields as one stock bundle, separately by year/slate."""

    return _permuted_columns(frame, features, seed=seed, include_year=True)


def permute_group_within_slates(
    frame: pd.DataFrame, features: Sequence[str], *, seed: int
) -> pd.DataFrame:
    """Permute one fixed attribution group together within assessment slates."""

    return _permuted_columns(frame, features, seed=seed, include_year=False)


def session_block_bootstrap_indices(
    frame: pd.DataFrame, *, draws: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Return whole-session row-index draws preserving every row in sampled sessions."""

    if draws <= 0 or "session" not in frame:
        raise ValueError("positive draws and a session column are required")
    sessions = np.asarray(sorted(frame["session"].astype(str).unique()), dtype=object)
    positions = {
        session: np.flatnonzero(frame["session"].astype(str).to_numpy() == session)
        for session in sessions
    }
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    for _ in range(draws):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        output.append(np.concatenate([positions[str(session)] for session in sampled]))
    return tuple(output)


def binary_brier(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return weighted binary Brier score."""

    truth = np.asarray(targets, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    sample_weight = np.ones(len(truth)) if weights is None else np.asarray(weights, dtype=float)
    return float(np.average((probability - truth) ** 2, weights=sample_weight))


def binary_log_loss(
    targets: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return weighted, numerically stable binary log loss."""

    truth = np.asarray(targets, dtype=float)
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0 - 1e-15)
    sample_weight = np.ones(len(truth)) if weights is None else np.asarray(weights, dtype=float)
    losses = -(truth * np.log(probability) + (1.0 - truth) * np.log(1.0 - probability))
    return float(np.average(losses, weights=sample_weight))


def multiclass_brier(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return weighted multiclass Brier score using the full one-hot sum."""

    truth = np.asarray(targets, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    sample_weight = np.ones(len(truth)) if weights is None else np.asarray(weights, dtype=float)
    expected = np.eye(probability.shape[1], dtype=float)[truth]
    return float(np.average(np.sum((probability - expected) ** 2, axis=1), weights=sample_weight))


def multiclass_log_loss(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return weighted multiclass log loss."""

    truth = np.asarray(targets, dtype=int)
    probability = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0)
    sample_weight = np.ones(len(truth)) if weights is None else np.asarray(weights, dtype=float)
    losses = -np.log(probability[np.arange(len(truth)), truth])
    return float(np.average(losses, weights=sample_weight))


def decide_screen(
    *, stage_a_passes: bool, stage_b_passes: bool, point_estimate_improves: bool
) -> str:
    """Apply the preregistered primary-decision precedence."""

    if stage_a_passes and stage_b_passes:
        return "opening_trajectories_predict_unregistered_events_and_families"
    if stage_a_passes:
        return "opening_trajectories_predict_unregistered_events_only"
    if stage_b_passes:
        return "opening_trajectories_predict_hidden_families_only"
    if point_estimate_improves:
        return "opening_trajectory_signal_descriptive_only"
    return "no_opening_trajectory_unregistered_increment"


__all__ = [
    "BEHAVIOURS",
    "CanonicalUnregisteredPath",
    "OTHER_FAMILY",
    "UnregisteredPathEvent",
    "binary_brier",
    "binary_log_loss",
    "binary_targets",
    "canonical_unregistered_path",
    "decide_screen",
    "first_unregistered_path",
    "hidden_family_census",
    "multiclass_brier",
    "multiclass_log_loss",
    "opening_anchor_triplet",
    "opening_population",
    "permute_group_within_slates",
    "permute_trajectory_bundle_within_slates",
    "pool_hidden_family",
    "reject_protected_dates",
    "select_hidden_families",
    "session_block_bootstrap_indices",
    "trajectory_feature_names",
]
