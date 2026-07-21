"""Bounded structural helpers for the emotion/regime loop-funnel quick screen.

This module has no market-data, economic-outcome, execution, order, broker, or
deployment surface.  It only supplies deterministic transformations used by a
retrospective structural feasibility experiment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "pre_loop_prediction": True,
    "soft_regime_mixture": True,
    "behavioural_regime_gating_test": True,
    "structural_outcomes_only": True,
    "economic_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}

DECISION_ORDINALS = (6, 12)
PROTECTED_START = pd.Timestamp("2025-08-23")
BEHAVIOURAL_DIMENSIONS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "pressure_magnitude",
    "exhaustion_magnitude",
    "signed_exhaustion",
    "independence",
    "signed_independence",
)
PRIMARY_BEHAVIOURAL_FEATURES = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "signed_exhaustion",
)
BEHAVIOURAL_JOIN_KEYS = (
    "symbol",
    "session",
    "decision_ordinal",
    "feature_available_timestamp_utc",
)
V2_JOIN_KEYS = BEHAVIOURAL_JOIN_KEYS
STATE_PROBABILITY_FEATURES = tuple(f"state_p_{state}" for state in range(8))
PRESSURE_INTERACTIONS = tuple(f"state_p_{state}_x_signed_pressure" for state in range(8))
EXHAUSTION_INTERACTIONS = tuple(f"state_p_{state}_x_signed_exhaustion" for state in range(8))
UNCERTAINTY_INTERACTIONS = (
    "posterior_entropy_x_frustration",
    "posterior_entropy_x_tension",
    "transition_probability_x_arousal",
    "top_second_margin_x_conviction",
)
INTERACTION_FEATURES = (
    *PRESSURE_INTERACTIONS,
    *EXHAUSTION_INTERACTIONS,
    *UNCERTAINTY_INTERACTIONS,
)


class BlockedScreen(RuntimeError):
    """Fail-closed experiment blocker with one permitted decision code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def reconstruct_behavioural_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the ten frozen dimensions from their audited z-components."""

    result = pd.DataFrame(index=frame.index)
    result["arousal"] = frame[["z_activity_effort", "z_range_effort", "z_travel_effort"]].mean(
        axis=1
    )
    result["conviction"] = frame[
        ["z_absolute_efficiency", "z_close_retention", "z_directional_persistence"]
    ].mean(axis=1)
    result["frustration"] = frame[
        ["z_activity_effort", "z_travel_effort", "z_extreme_rejection"]
    ].mean(axis=1) - frame[["z_absolute_progress", "z_absolute_efficiency"]].mean(axis=1)
    result["tension"] = (
        frame[["z_activity_effort", "z_compression", "z_extreme_rejection"]].mean(axis=1)
        - frame["z_absolute_progress"]
    )
    result["signed_pressure"] = frame[
        [
            "z_signed_progress",
            "z_signed_efficiency",
            "z_mean_close_location",
            "z_boundary_slope",
        ]
    ].mean(axis=1)
    result["pressure_magnitude"] = result["signed_pressure"].abs()
    result["exhaustion_magnitude"] = (
        frame["z_effort_acceleration"]
        - frame["z_aligned_progress_acceleration"]
        + frame["z_directional_rejection"]
    )
    result["signed_exhaustion"] = (
        np.sign(result["signed_pressure"]) * result["exhaustion_magnitude"]
    )
    result["independence"] = (
        frame[["z_return_gap", "z_activity_gap", "z_range_gap"]].abs().mean(axis=1)
    )
    result["signed_independence"] = np.sign(frame["return_gap"]) * result["independence"]
    return result.loc[:, list(BEHAVIOURAL_DIMENSIONS)]


def join_behavioural_ledger(
    decisions: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the frozen ledger exactly and verify all ten dimension formulas."""

    missing_decision = sorted(set(BEHAVIOURAL_JOIN_KEYS).difference(decisions.columns))
    missing_ledger = sorted(set(BEHAVIOURAL_JOIN_KEYS).difference(ledger.columns))
    if missing_decision or missing_ledger:
        raise BlockedScreen(
            "blocked_behavioural_ledger_not_reconstructable",
            f"join keys missing: decisions={missing_decision}, ledger={missing_ledger}",
        )
    left = decisions.copy()
    right = ledger.copy()
    for candidate in (left, right):
        candidate["symbol"] = candidate["symbol"].astype(str)
        candidate["session"] = candidate["session"].astype(str)
        candidate["decision_ordinal"] = candidate["decision_ordinal"].astype(int)
        candidate["feature_available_timestamp_utc"] = pd.to_datetime(
            candidate["feature_available_timestamp_utc"], utc=True, errors="raise"
        )
    if right.duplicated(list(BEHAVIOURAL_JOIN_KEYS)).any():
        raise BlockedScreen(
            "blocked_behavioural_ledger_not_reconstructable", "behavioural keys are not unique"
        )
    try:
        joined = left.merge(
            right,
            on=list(BEHAVIOURAL_JOIN_KEYS),
            how="left",
            validate="one_to_one",
            suffixes=("", "__ledger"),
            indicator=True,
        )
    except (KeyError, ValueError) as error:
        raise BlockedScreen("blocked_behavioural_ledger_not_reconstructable", str(error)) from error
    if len(joined) != len(left) or not joined["_merge"].eq("both").all():
        raise BlockedScreen(
            "blocked_behavioural_ledger_not_reconstructable",
            "decision-to-ledger join is incomplete",
        )
    joined = joined.drop(columns="_merge")
    reconstructed = reconstruct_behavioural_dimensions(joined)
    expected = joined.loc[:, list(BEHAVIOURAL_DIMENSIONS)].to_numpy(dtype=float)
    actual = reconstructed.to_numpy(dtype=float)
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise BlockedScreen(
            "blocked_behavioural_ledger_not_reconstructable", "dimension value is not finite"
        )
    absolute_error = np.abs(actual - expected)
    maximum_error = float(absolute_error.max(initial=0.0))
    if maximum_error > tolerance:
        raise BlockedScreen(
            "blocked_behavioural_ledger_not_reconstructable",
            f"maximum reconstruction error {maximum_error:.17g} exceeds {tolerance:.1e}",
        )
    audit: dict[str, Any] = {
        **SAFETY_FLAGS,
        "rows_requested": len(left),
        "rows_joined": len(joined),
        "join_keys": list(BEHAVIOURAL_JOIN_KEYS),
        "dimensions_reconstructed": list(BEHAVIOURAL_DIMENSIONS),
        "maximum_permitted_absolute_error": tolerance,
        "maximum_absolute_reconstruction_error": maximum_error,
        "passed": True,
    }
    return joined, audit


def join_v2_posteriors(
    archived_decisions: pd.DataFrame,
    reproduced_decisions: pd.DataFrame,
    *,
    tolerance: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join reproduced causal V2 summaries and verify the frozen decision export."""

    archived = archived_decisions.copy()
    reproduced = reproduced_decisions.copy()
    for candidate in (archived, reproduced):
        require_columns(candidate, V2_JOIN_KEYS, context="V2 decision join")
        candidate["symbol"] = candidate["symbol"].astype(str)
        candidate["session"] = candidate["session"].astype(str)
        candidate["decision_ordinal"] = candidate["decision_ordinal"].astype(int)
        candidate["feature_available_timestamp_utc"] = pd.to_datetime(
            candidate["feature_available_timestamp_utc"], utc=True, errors="raise"
        )
    required_archived = [
        *(f"posterior_state_{state}" for state in range(8)),
        "posterior_entropy",
        "maximum_posterior_probability",
        "current_state",
    ]
    required_reproduced = [
        *STATE_PROBABILITY_FEATURES,
        "posterior_entropy_reproduced",
        "top_state_probability",
        "hard_top_state",
        "top_second_margin",
        "expected_state_age",
        "transition_probability",
        "persistence_probability",
    ]
    require_columns(archived, required_archived, context="archived V2")
    require_columns(reproduced, required_reproduced, context="reproduced V2")
    if reproduced.duplicated(list(V2_JOIN_KEYS)).any():
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            "reproduced V2 decision keys are not unique",
        )
    try:
        joined = archived.merge(
            reproduced,
            on=list(V2_JOIN_KEYS),
            how="left",
            validate="one_to_one",
            indicator=True,
        )
    except (KeyError, ValueError) as error:
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable", str(error)
        ) from error
    if len(joined) != len(archived) or not joined["_merge"].eq("both").all():
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            "frozen V2 decision population did not reconstruct exactly",
        )
    joined = joined.drop(columns="_merge")
    frozen_probabilities = joined[[f"posterior_state_{state}" for state in range(8)]].to_numpy(
        dtype=float
    )
    reproduced_probabilities = joined[list(STATE_PROBABILITY_FEATURES)].to_numpy(dtype=float)
    probability_error = np.abs(frozen_probabilities - reproduced_probabilities)
    entropy_error = np.abs(
        joined["posterior_entropy"].to_numpy(dtype=float)
        - joined["posterior_entropy_reproduced"].to_numpy(dtype=float)
    )
    top_error = np.abs(
        joined["maximum_posterior_probability"].to_numpy(dtype=float)
        - joined["top_state_probability"].to_numpy(dtype=float)
    )
    maximum_probability_error = float(probability_error.max(initial=0.0))
    maximum_entropy_error = float(entropy_error.max(initial=0.0))
    maximum_top_error = float(top_error.max(initial=0.0))
    hard_agreement = float(
        np.mean(
            joined["current_state"].to_numpy(dtype=int)
            == joined["hard_top_state"].to_numpy(dtype=int)
        )
    )
    simplex_error = float(np.abs(reproduced_probabilities.sum(axis=1) - 1.0).max(initial=0.0))
    finite = np.isfinite(
        joined[
            [
                *STATE_PROBABILITY_FEATURES,
                "posterior_entropy_reproduced",
                "top_state_probability",
                "top_second_margin",
                "expected_state_age",
                "transition_probability",
                "persistence_probability",
            ]
        ].to_numpy(dtype=float)
    ).all()
    if (
        not finite
        or maximum_probability_error > tolerance
        or maximum_entropy_error > tolerance
        or maximum_top_error > tolerance
        or simplex_error > tolerance
        or hard_agreement != 1.0
    ):
        raise BlockedScreen(
            "blocked_v2_decision_population_not_reconstructable",
            "frozen V2 posterior or hard state differs from causal reconstruction",
        )
    joined["posterior_entropy"] = joined["posterior_entropy_reproduced"]
    audit: dict[str, Any] = {
        **SAFETY_FLAGS,
        "rows_expected": len(archived),
        "rows_reconstructed": len(joined),
        "join_keys": list(V2_JOIN_KEYS),
        "maximum_permitted_absolute_error": tolerance,
        "maximum_posterior_absolute_error": maximum_probability_error,
        "maximum_entropy_absolute_error": maximum_entropy_error,
        "maximum_top_probability_absolute_error": maximum_top_error,
        "maximum_simplex_absolute_error": simplex_error,
        "hard_top_state_agreement": hard_agreement,
        "passed": True,
    }
    return joined, audit


def resolve_first_loop_target(
    engine: Any,
    trace: Any,
    *,
    decision_id: str,
    decision_event_index: int,
    decision_bar_ordinal: int,
    decision_timestamp: datetime,
    session_end_bar_ordinal: int,
    horizon_bars: int = 6,
    source_available: bool = True,
    symbol: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Apply frozen first-event semantics and expose one oriented structural target."""

    if horizon_bars != 6:
        raise BlockedScreen(
            "blocked_quick_funnel_resource_limit", "the quick screen permits only six bars"
        )
    outcome = engine.outcome_for_decision(
        trace,
        decision_id=decision_id,
        decision_event_index=decision_event_index,
        decision_bar_ordinal=decision_bar_ordinal,
        decision_timestamp=decision_timestamp,
        decision_available_timestamp=decision_timestamp,
        horizon_bars=horizon_bars,
        session_end_bar_ordinal=session_end_bar_ordinal,
        source_available=source_available,
        symbol=symbol,
        session=session,
    )
    primary = str(outcome.primary_label)
    base: dict[str, Any] = {
        "raw_outcome": primary,
        "semantic_loop_id": None,
        "orientation": None,
        "oriented_loop_key": None,
        "motif_type": None,
        "bars_until_completion": outcome.bars_until_completion,
        "state_events_until_completion": outcome.state_events_until_completion,
        "target_excluded": False,
        "tied_semantic_loop_ids": list(outcome.tied_semantic_loop_ids),
        "source_available": bool(outcome.source_available),
    }
    if primary == "TIED_REGISTERED_COMPLETION":
        base["target_excluded"] = True
        return base
    if primary == "UNAVAILABLE":
        base["target_excluded"] = True
        return base
    if primary == "UNREGISTERED_LOOP":
        return base
    if primary in {"SESSION_END", "NO_REGISTERED_LOOP_WITHIN_HORIZON"}:
        # A trace is intentionally bounded at the target horizon.  Absence of a
        # later state event in that bounded trace is not an early session end.
        if session_end_bar_ordinal > decision_bar_ordinal + horizon_bars:
            base["raw_outcome"] = "NO_REGISTERED_COMPLETION"
        return base
    if not outcome.earliest_registered_events:
        raise BlockedScreen(
            "blocked_chronology_or_leakage_failure",
            "registered primary outcome has no completion event",
        )
    event = outcome.earliest_registered_events[0]
    motif = str(event.motif_type)
    raw_by_motif = {
        "primitive": "REGISTERED_PRIMITIVE_COMPLETION",
        "repeat": "REGISTERED_REPEAT_COMPLETION",
        "composite": "REGISTERED_COMPOSITE_COMPLETION",
    }
    if motif not in raw_by_motif:
        raise BlockedScreen(
            "blocked_chronology_or_leakage_failure", f"unknown registered motif type: {motif}"
        )
    base.update(
        {
            "raw_outcome": raw_by_motif[motif],
            "semantic_loop_id": event.semantic_loop_id,
            "orientation": event.orientation_id,
            "oriented_loop_key": f"{event.semantic_loop_id}|{event.orientation_id}",
            "motif_type": motif,
        }
    )
    return base


def select_exact_loop_classes(
    frame: pd.DataFrame,
    *,
    maximum_selected: int = 6,
    require_minimum: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Freeze the development-only exact oriented-loop vocabulary."""

    development = frame.loc[pd.to_numeric(frame["year"], errors="raise").eq(2024)].copy()
    registered = development.loc[
        development["raw_outcome"].astype(str).str.startswith("REGISTERED_")
        & development["oriented_loop_key"].notna()
    ].copy()
    rows: list[dict[str, Any]] = []
    for key, group in registered.groupby("oriented_loop_key", sort=True):
        stock_counts = group.groupby("symbol", sort=True).size()
        support = len(group)
        rows.append(
            {
                "oriented_loop_key": str(key),
                "support": support,
                "sessions": int(group["session"].nunique()),
                "stocks": int(group["symbol"].nunique()),
                "months": int(group["year_month"].nunique()),
                "maximum_stock_share": float(stock_counts.max() / support),
            }
        )
    support_frame = pd.DataFrame(rows)
    if support_frame.empty:
        eligible = support_frame.copy()
    else:
        eligible = support_frame.loc[
            support_frame["support"].ge(50)
            & support_frame["sessions"].ge(20)
            & support_frame["stocks"].ge(8)
            & support_frame["months"].ge(4)
            & support_frame["maximum_stock_share"].le(0.30)
        ].copy()
        eligible = eligible.sort_values(
            ["support", "oriented_loop_key"], ascending=[False, True], kind="mergesort"
        )
    selected = eligible.head(maximum_selected).reset_index(drop=True)
    if len(selected) < 4 and require_minimum:
        raise BlockedScreen(
            "blocked_insufficient_loop_class_support",
            f"only {len(selected)} exact oriented loops passed development support",
        )
    selected_records = cast(list[dict[str, Any]], selected.to_dict(orient="records"))
    eligible_records = cast(list[dict[str, Any]], eligible.to_dict(orient="records"))
    mapping = {
        str(row["oriented_loop_key"]): f"LOOP_{index + 1}"
        for index, row in enumerate(selected_records)
    }
    manifest: dict[str, Any] = {
        **SAFETY_FLAGS,
        "selection_interval": "2024-01-01_through_2024-12-31_only",
        "selection_rule": {
            "minimum_outcomes": 50,
            "minimum_sessions": 20,
            "minimum_stocks": 8,
            "minimum_months": 4,
            "maximum_stock_share": 0.30,
            "maximum_selected": maximum_selected,
            "tie_break": "stable_lexicographic_oriented_loop_key",
        },
        "eligible_oriented_loops": {
            str(row["oriented_loop_key"]): {
                "support": int(row["support"]),
                "sessions": int(row["sessions"]),
                "stocks": int(row["stocks"]),
                "months": int(row["months"]),
                "maximum_stock_share": float(row["maximum_stock_share"]),
            }
            for row in eligible_records
        },
        "selected_mapping": mapping,
        "selected_count": len(mapping),
        "minimum_required_selected_count": 4,
        "selection_support_passed": len(mapping) >= 4,
    }
    return mapping, manifest


def pool_target_class(
    raw_outcome: str,
    oriented_loop_key: str | None,
    mapping: Mapping[str, str],
) -> str | None:
    """Map one raw structural event into the frozen multiclass target."""

    if raw_outcome in {"TIED_REGISTERED_COMPLETION", "UNAVAILABLE"}:
        return None
    if raw_outcome.startswith("REGISTERED_"):
        if oriented_loop_key is None:
            raise BlockedScreen(
                "blocked_chronology_or_leakage_failure",
                "registered completion is missing its oriented key",
            )
        return mapping.get(oriented_loop_key, "OTHER_REGISTERED_LOOP")
    if raw_outcome == "UNREGISTERED_LOOP":
        return "UNREGISTERED_LOOP"
    if raw_outcome in {
        "SESSION_END",
        "NO_REGISTERED_COMPLETION",
        "NO_REGISTERED_LOOP_WITHIN_HORIZON",
    }:
        return "NO_REGISTERED_COMPLETION"
    raise BlockedScreen(
        "blocked_chronology_or_leakage_failure", f"unknown raw structural outcome: {raw_outcome}"
    )


def _raw_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for state in range(8):
        result[f"state_p_{state}_x_signed_pressure"] = (
            frame[f"state_p_{state}"] * frame["signed_pressure"]
        )
    for state in range(8):
        result[f"state_p_{state}_x_signed_exhaustion"] = (
            frame[f"state_p_{state}"] * frame["signed_exhaustion"]
        )
    result["posterior_entropy_x_frustration"] = frame["posterior_entropy"] * frame["frustration"]
    result["posterior_entropy_x_tension"] = frame["posterior_entropy"] * frame["tension"]
    result["transition_probability_x_arousal"] = frame["transition_probability"] * frame["arousal"]
    result["top_second_margin_x_conviction"] = frame["top_second_margin"] * frame["conviction"]
    return result.loc[:, list(INTERACTION_FEATURES)]


def build_interactions(
    frame: pd.DataFrame,
    *,
    fit_bounds: bool = False,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Construct exactly 20 preregistered interactions and apply frozen clipping."""

    if fit_bounds and bounds is not None:
        raise ValueError("cannot fit and supply interaction bounds simultaneously")
    result = _raw_interactions(frame)
    fitted: dict[str, tuple[float, float]]
    if fit_bounds:
        fitted = {
            feature: (
                float(result[feature].quantile(0.01, interpolation="linear")),
                float(result[feature].quantile(0.99, interpolation="linear")),
            )
            for feature in INTERACTION_FEATURES
        }
    elif bounds is None:
        fitted = {}
    else:
        fitted = {feature: (float(value[0]), float(value[1])) for feature, value in bounds.items()}
        if set(fitted) != set(INTERACTION_FEATURES):
            raise ValueError("interaction clipping bounds differ from the 20-feature manifest")
    if fitted:
        for feature in INTERACTION_FEATURES:
            lower, upper = fitted[feature]
            result[feature] = result[feature].clip(lower=lower, upper=upper)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise BlockedScreen(
            "blocked_chronology_or_leakage_failure", "interaction value is not finite"
        )
    return result, fitted


def multiclass_brier(
    target_indices: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Return the weighted mean sum of squared multiclass probability errors."""

    targets = np.asarray(target_indices, dtype=np.int64)
    predicted = np.asarray(probabilities, dtype=float)
    if predicted.ndim != 2 or targets.ndim != 1 or len(targets) != len(predicted):
        raise ValueError("multiclass Brier inputs have incompatible shapes")
    if len(targets) == 0:
        raise ValueError("multiclass Brier requires at least one row")
    if (targets < 0).any() or (targets >= predicted.shape[1]).any():
        raise ValueError("target index lies outside the probability columns")
    if not np.isfinite(predicted).all():
        raise ValueError("probabilities must be finite")
    one_hot = np.zeros_like(predicted)
    one_hot[np.arange(len(targets)), targets] = 1.0
    row_scores = np.square(predicted - one_hot).sum(axis=1)
    if sample_weight is None:
        return float(np.mean(row_scores))
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != row_scores.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample weights are invalid")
    return float(np.average(row_scores, weights=weights))


def prediction_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Return row-wise natural-log prediction entropy with exact zero handling."""

    predicted = np.asarray(probabilities, dtype=float)
    if predicted.ndim != 2 or not np.isfinite(predicted).all() or (predicted < 0.0).any():
        raise ValueError("prediction entropy requires a finite non-negative matrix")
    safe = np.clip(predicted, np.finfo(float).tiny, 1.0)
    terms = np.where(predicted > 0.0, predicted * np.log(safe), 0.0)
    return np.asarray(-terms.sum(axis=1), dtype=float)


def session_block_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Resample whole sessions while preserving all stocks and checkpoints."""

    if draws <= 0 or draws > 50:
        raise BlockedScreen(
            "blocked_quick_funnel_resource_limit", "bootstrap draws must be in [1, 50]"
        )
    sessions = sorted(frame["session"].astype(str).unique())
    if not sessions:
        raise ValueError("session bootstrap requires at least one session")
    by_session = {
        session: np.flatnonzero(frame["session"].astype(str).eq(session).to_numpy())
        for session in sessions
    }
    generator = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for _ in range(draws):
        sampled = generator.choice(sessions, size=len(sessions), replace=True)
        result.append(np.concatenate([by_session[str(session)] for session in sampled]))
    return tuple(result)


def permute_behavioural_bundle_within_slates(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    """Permute the complete six-dimensional bundle within each fixed slate."""

    require_columns(frame, ("slate_id", *PRIMARY_BEHAVIOURAL_FEATURES), context="null bundle")
    result = frame.copy()
    generator = np.random.default_rng(seed)
    for _, indices in result.groupby("slate_id", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        permutation = generator.permutation(len(positions))
        bundles = frame.loc[positions, list(PRIMARY_BEHAVIOURAL_FEATURES)].to_numpy(copy=True)
        result.loc[positions, list(PRIMARY_BEHAVIOURAL_FEATURES)] = bundles[permutation]
    return result


def decide_funnel(
    *,
    m1_pass: bool,
    m2_pass: bool,
    descriptive_change: bool,
) -> str:
    """Choose exactly one non-blocked preregistered decision category."""

    if m2_pass:
        return "regime_mix_filters_behaviour_into_loop_distribution"
    if m1_pass:
        return "behaviour_main_effects_only"
    if descriptive_change:
        return "descriptive_funnel_only_no_predictive_increment"
    return "no_behaviour_regime_loop_funnel_increment"


def validate_checkpoint_timing(frame: pd.DataFrame) -> dict[str, Any]:
    """Require ordinal 6/12 to be exactly 10:00/10:30 New York time."""

    ordinals = pd.to_numeric(frame["decision_ordinal"], errors="raise").astype(int)
    timestamps = pd.to_datetime(frame["feature_available_timestamp_utc"], utc=True, errors="raise")
    clocks = timestamps.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    expected = ordinals.map({6: "10:00", 12: "10:30"})
    passed = bool(ordinals.isin(DECISION_ORDINALS).all() and clocks.eq(expected).all())
    if not passed:
        raise BlockedScreen(
            "blocked_chronology_or_leakage_failure", "checkpoint ordinal or local clock differs"
        )
    return {
        **SAFETY_FLAGS,
        "decision_ordinals": sorted(int(value) for value in ordinals.unique()),
        "local_clocks": sorted(str(value) for value in clocks.unique()),
        "passed": True,
    }


def reject_protected_dates(frame: pd.DataFrame) -> None:
    """Reject any decision or market row dated on/after the protected boundary."""

    sessions = pd.to_datetime(frame["session"], errors="raise")
    if sessions.ge(PROTECTED_START).any():
        raise BlockedScreen(
            "blocked_protected_boundary_failure", "a row dated 2025-08-23 or later materialised"
        )


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Typed metric row used later by the experiment runner."""

    name: str
    value: float


def safety_payload(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a fresh mandatory safety payload."""

    result: dict[str, Any] = dict(SAFETY_FLAGS)
    if extra is not None:
        result.update(extra)
    return result


def require_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    """Raise a chronology blocker if a required structural column is absent."""

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise BlockedScreen(
            "blocked_chronology_or_leakage_failure", f"{context} columns missing: {missing}"
        )


__all__ = [
    "BEHAVIOURAL_DIMENSIONS",
    "BEHAVIOURAL_JOIN_KEYS",
    "BlockedScreen",
    "DECISION_ORDINALS",
    "EXHAUSTION_INTERACTIONS",
    "INTERACTION_FEATURES",
    "MetricResult",
    "PRESSURE_INTERACTIONS",
    "PRIMARY_BEHAVIOURAL_FEATURES",
    "PROTECTED_START",
    "SAFETY_FLAGS",
    "STATE_PROBABILITY_FEATURES",
    "UNCERTAINTY_INTERACTIONS",
    "build_interactions",
    "decide_funnel",
    "join_behavioural_ledger",
    "join_v2_posteriors",
    "multiclass_brier",
    "permute_behavioural_bundle_within_slates",
    "pool_target_class",
    "prediction_entropy",
    "reconstruct_behavioural_dimensions",
    "reject_protected_dates",
    "require_columns",
    "resolve_first_loop_target",
    "safety_payload",
    "select_exact_loop_classes",
    "session_block_bootstrap_draws",
    "validate_checkpoint_timing",
]
