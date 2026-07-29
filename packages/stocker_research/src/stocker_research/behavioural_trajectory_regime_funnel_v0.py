"""Deterministic helpers for the behavioural-trajectory regime funnel screen.

The module is retrospective, research-only, structural, and has no execution
or production-runtime surface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "behavioural_trajectory_test": True,
    "soft_regime_mixture": True,
    "coarse_loop_family_target": True,
    "structural_outcomes_only": True,
    "economic_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
DECISION_CATEGORIES = (
    "behavioural_trajectories_improve_coarse_structural_forecast",
    "regime_mix_filters_behavioural_trajectories",
    "trajectory_main_effects_only",
    "descriptive_trajectory_structure_only",
    "no_behavioural_trajectory_increment",
    "blocked_predecessor_population_not_reconstructable",
    "blocked_frozen_m2_not_reconstructable",
    "blocked_behavioural_trajectory_not_causal",
    "blocked_insufficient_trajectory_support",
    "blocked_protected_boundary_failure",
    "blocked_chronology_or_leakage_failure",
    "blocked_quick_trajectory_resource_limit",
    "blocked_model_convergence_failure",
    "blocked_reproducibility_or_audit_failure",
)
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")


ANCHORS_BY_DECISION: dict[int, tuple[int, int, int]] = {
    6: (2, 4, 6),
    12: (6, 9, 12),
}
TRAJECTORY_INTERACTION_SPECS: tuple[tuple[str, str, str], ...] = (
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
TRAJECTORY_INTERACTION_FEATURES = tuple(
    specification[0] for specification in TRAJECTORY_INTERACTION_SPECS
)


class BlockedScreen(RuntimeError):
    """Fail-closed screen stop carrying one preregistered decision code."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in DECISION_CATEGORIES or not code.startswith("blocked_"):
            raise ValueError(f"invalid trajectory screen blocker: {code}")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def trajectory_anchors(decision_ordinal: int) -> tuple[int, int, int]:
    """Return the three preregistered completed-bar anchors for a decision."""

    try:
        return ANCHORS_BY_DECISION[int(decision_ordinal)]
    except KeyError as error:
        raise ValueError(f"unsupported decision ordinal: {decision_ordinal}") from error


def anchor_formula_availability(completed_bar_count: int) -> tuple[bool, str | None]:
    """Apply the frozen behavioural constructor's even-window precondition."""

    count = int(completed_bar_count)
    if count < 2:
        return False, "frozen_opening_raw_components_requires_at_least_two_completed_bars"
    if count % 2:
        return False, "frozen_opening_raw_components_requires_even_completed_bar_count"
    return True, None


def causal_anchor_prefix(bars: pd.DataFrame, completed_bar_count: int) -> pd.DataFrame:
    """Return exactly the bars completed by an anchor, never later bars."""

    required = {"bar_start_timestamp", "bar_complete_timestamp"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"anchor bar columns missing: {missing}")
    count = int(completed_bar_count)
    if count < 1 or len(bars) < count:
        raise ValueError("anchor does not have the requested completed bars")
    ordered = bars.copy()
    ordered["bar_start_timestamp"] = pd.to_datetime(
        ordered["bar_start_timestamp"], utc=True, errors="raise"
    )
    ordered["bar_complete_timestamp"] = pd.to_datetime(
        ordered["bar_complete_timestamp"], utc=True, errors="raise"
    )
    ordered = ordered.sort_values("bar_start_timestamp", kind="mergesort").reset_index(drop=True)
    if not ordered["bar_start_timestamp"].is_monotonic_increasing:
        raise ValueError("anchor bars are not chronological")
    prefix = ordered.iloc[:count].copy()
    if not bool(
        prefix["bar_complete_timestamp"].le(prefix.iloc[-1]["bar_complete_timestamp"]).all()
    ):
        raise ValueError("future bar entered an earlier anchor")
    return prefix.reset_index(drop=True)


def trajectory_feature_values(
    earliest: float,
    middle: float,
    final: float,
) -> dict[str, float | int]:
    """Calculate exactly the six preregistered trajectory forms."""

    e0, e1, e2 = float(earliest), float(middle), float(final)
    if not all(math.isfinite(value) for value in (e0, e1, e2)):
        raise ValueError("trajectory anchors must be finite")
    first_change = e1 - e0
    recent_change = e2 - e1
    persistence = 1 if e0 < e1 < e2 else -1 if e0 > e1 > e2 else 0
    reversal = int(
        first_change != 0.0
        and recent_change != 0.0
        and ((first_change > 0.0) != (recent_change > 0.0))
    )
    return {
        "change": e2 - e0,
        "recent_change": recent_change,
        "acceleration": recent_change - first_change,
        "persistence": persistence,
        "reversal": reversal,
        "peak_displacement": e2 - max(e0, e1, e2),
    }


def build_trajectory_interactions(
    frame: pd.DataFrame,
    *,
    fit_bounds: bool = False,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Build exactly six trajectory/regime products and optionally clip them."""

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


def multiclass_brier(
    target_indices: NDArray[np.int64],
    probabilities: FloatArray,
    sample_weight: FloatArray | None = None,
) -> float:
    """Return weighted mean row-wise multiclass squared probability error."""

    targets = np.asarray(target_indices, dtype=np.int64)
    predicted = np.asarray(probabilities, dtype=np.float64)
    if predicted.ndim != 2 or targets.ndim != 1 or len(targets) != len(predicted):
        raise ValueError("multiclass Brier inputs have incompatible shapes")
    if len(targets) == 0 or (targets < 0).any() or (targets >= predicted.shape[1]).any():
        raise ValueError("multiclass Brier target index is invalid")
    if not np.isfinite(predicted).all():
        raise ValueError("multiclass probabilities must be finite")
    one_hot = np.zeros_like(predicted)
    one_hot[np.arange(len(targets)), targets] = 1.0
    row_scores = np.square(predicted - one_hot).sum(axis=1)
    if sample_weight is None:
        return float(row_scores.mean())
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != row_scores.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("multiclass Brier sample weights are invalid")
    return float(np.average(row_scores, weights=weights))


def prediction_entropy(probabilities: FloatArray) -> FloatArray:
    """Return row-wise natural-log entropy with exact zero handling."""

    predicted = np.asarray(probabilities, dtype=np.float64)
    if predicted.ndim != 2 or not np.isfinite(predicted).all() or (predicted < 0.0).any():
        raise ValueError("prediction entropy requires finite non-negative probabilities")
    safe = np.clip(predicted, np.finfo(float).tiny, 1.0)
    terms = np.where(predicted > 0.0, predicted * np.log(safe), 0.0)
    return np.asarray(-terms.sum(axis=1), dtype=np.float64)


@dataclass(frozen=True, slots=True, eq=False)
class SessionBootstrapDraw:
    """One fixed-seed resample of complete assessment sessions."""

    draw: int
    sampled_sessions: tuple[str, ...]
    row_indices: NDArray[np.int64]


def session_block_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    session_column: str = "session",
) -> tuple[SessionBootstrapDraw, ...]:
    """Resample whole sessions while retaining every row in each sampled block."""

    if draws < 1 or draws > 50:
        raise ValueError("session bootstrap draws must be in [1, 50]")
    if session_column not in frame:
        raise ValueError(f"session column missing: {session_column}")
    session_values = frame[session_column].astype(str).reset_index(drop=True)
    sessions = np.asarray(sorted(session_values.unique()), dtype=object)
    if len(sessions) < 2:
        raise ValueError("session bootstrap requires at least two sessions")
    values = session_values.to_numpy(dtype=object)
    by_session = {
        str(session): np.flatnonzero(values == session).astype(np.int64) for session in sessions
    }
    generator = np.random.default_rng(seed)
    result: list[SessionBootstrapDraw] = []
    for draw in range(draws):
        sampled = tuple(
            str(value) for value in generator.choice(sessions, size=len(sessions), replace=True)
        )
        row_indices = np.concatenate([by_session[session] for session in sampled]).astype(np.int64)
        result.append(SessionBootstrapDraw(draw, sampled, row_indices))
    return tuple(result)


def permute_trajectory_bundle_within_slates(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    seed: int,
    slate_column: str = "slate_id",
) -> pd.DataFrame:
    """Permute complete trajectory bundles only within each fixed slate."""

    names = tuple(str(feature) for feature in features)
    missing = sorted({slate_column, *names}.difference(frame.columns))
    if missing:
        raise ValueError(f"trajectory permutation columns missing: {missing}")
    values = frame.loc[:, list(names)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("trajectory permutation bundle must be finite")
    result = frame.copy()
    generator = np.random.default_rng(seed)
    for indices in frame.groupby(slate_column, sort=True).groups.values():
        index = list(indices)
        bundle = frame.loc[index, list(names)].to_numpy(dtype=float, copy=True)
        result.loc[index, list(names)] = bundle[generator.permutation(len(index))]
    return result


def reject_protected_dates(frame: pd.DataFrame) -> None:
    """Reject any decision or source row dated on or after 2025-08-23."""

    if "session" not in frame:
        raise ValueError("session column is required for protected-date rejection")
    sessions = pd.to_datetime(frame["session"], utc=True, errors="raise")
    if bool(sessions.ge(PROTECTED_START).any()):
        raise BlockedScreen(
            "blocked_protected_boundary_failure",
            "a row dated 2025-08-23 or later materialised",
        )


def decide_trajectory_screen(
    *,
    t1_pass: bool,
    t2_pass: bool,
    descriptive_structure: bool,
    blocker: str | None = None,
) -> str:
    """Choose exactly one preregistered trajectory-screen category."""

    if blocker is not None:
        if blocker not in DECISION_CATEGORIES or not blocker.startswith("blocked_"):
            raise ValueError(f"invalid trajectory screen blocker: {blocker}")
        return blocker
    if t2_pass:
        return "regime_mix_filters_behavioural_trajectories"
    if t1_pass:
        return "trajectory_main_effects_only"
    if descriptive_structure:
        return "descriptive_trajectory_structure_only"
    return "no_behavioural_trajectory_increment"


def manual_multinomial_probabilities(
    frame: pd.DataFrame,
    model: Mapping[str, Any],
) -> FloatArray:
    """Reconstruct multinomial probabilities from frozen scaler and coefficients."""

    features = tuple(str(value) for value in cast(Sequence[object], model["features"]))
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"frozen multinomial features missing: {missing}")
    values = frame.loc[:, list(features)].to_numpy(dtype=np.float64)
    means = np.asarray(model["scaler_mean"], dtype=np.float64)
    scales = np.asarray(model["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficient"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    if (
        values.ndim != 2
        or means.shape != (values.shape[1],)
        or scales.shape != means.shape
        or coefficients.shape[1:] != means.shape
        or intercept.shape != (coefficients.shape[0],)
        or not np.isfinite(values).all()
        or not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept).all()
        or bool((scales <= 0.0).any())
    ):
        raise ValueError("frozen multinomial parameters do not align")
    logits = ((values - means) / scales) @ coefficients.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(logits)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return np.asarray(probabilities, dtype=np.float64)


__all__ = [
    "ANCHORS_BY_DECISION",
    "BlockedScreen",
    "DECISION_CATEGORIES",
    "PROTECTED_START",
    "SAFETY_FLAGS",
    "TRAJECTORY_INTERACTION_FEATURES",
    "TRAJECTORY_INTERACTION_SPECS",
    "SessionBootstrapDraw",
    "anchor_formula_availability",
    "build_trajectory_interactions",
    "causal_anchor_prefix",
    "decide_trajectory_screen",
    "multiclass_brier",
    "manual_multinomial_probabilities",
    "permute_trajectory_bundle_within_slates",
    "prediction_entropy",
    "reject_protected_dates",
    "session_block_bootstrap_draws",
    "trajectory_anchors",
    "trajectory_feature_values",
]
