#!/usr/bin/env python3
"""Independent lightweight audit for the coarse loop-family funnel V0 screen."""

# ruff: noqa: E402 -- repository package path is installed before local imports.

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
FROZEN_REPO_ROOT = Path(os.environ.get("STOCKER_FROZEN_REPO_ROOT", REPO_ROOT)).resolve()
PACKAGE_SOURCE = REPO_ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from stocker_research.loop_dictionary_v2 import LoopDictionary, decompose_closed_path
from stocker_research.loop_prefix_automaton_v2 import FirstNextLoopEventEngine

SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "pre_loop_prediction": True,
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
BEHAVIOURAL_FEATURES = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "signed_exhaustion",
)
ALL_DIMENSIONS = (
    *BEHAVIOURAL_FEATURES[:5],
    "pressure_magnitude",
    "exhaustion_magnitude",
    "signed_exhaustion",
    "independence",
    "signed_independence",
)
BEHAVIOURAL_Z_COMPONENTS = (
    "z_activity_effort",
    "z_range_effort",
    "z_travel_effort",
    "z_absolute_efficiency",
    "z_close_retention",
    "z_directional_persistence",
    "z_extreme_rejection",
    "z_absolute_progress",
    "z_compression",
    "z_signed_progress",
    "z_signed_efficiency",
    "z_mean_close_location",
    "z_boundary_slope",
    "z_effort_acceleration",
    "z_aligned_progress_acceleration",
    "z_directional_rejection",
    "z_return_gap",
    "z_activity_gap",
    "z_range_gap",
)
STATE_PROBABILITIES = tuple(f"state_p_{state}" for state in range(8))
M0_FEATURES = (
    *STATE_PROBABILITIES,
    "posterior_entropy",
    "top_state_probability",
    "top_second_margin",
    "expected_state_age",
    "persistence_probability",
    "transition_probability",
    "remaining_session_bars",
    "checkpoint",
)
INTERACTIONS = (
    *(f"state_p_{state}_x_signed_pressure" for state in range(8)),
    *(f"state_p_{state}_x_signed_exhaustion" for state in range(8)),
    "posterior_entropy_x_frustration",
    "posterior_entropy_x_tension",
    "transition_probability_x_arousal",
    "top_second_margin_x_conviction",
)
M1_FEATURES = (*M0_FEATURES, *BEHAVIOURAL_FEATURES)
M2_FEATURES = (*M1_FEATURES, *INTERACTIONS)
MODEL_FEATURES = {"M0": M0_FEATURES, "M1": M1_FEATURES, "M2": M2_FEATURES}
BOOTSTRAP_DRAWS = 50
BOOTSTRAP_SEED = 20260722
NULL_DRAWS = 10
NULL_SEED = 20260723
CHECKPOINT_MATERIAL_ADVERSITY = -0.001
DICTIONARY_PATH = (
    FROZEN_REPO_ROOT
    / "research"
    / "slrno-v2"
    / "20260714-regime-loop-handoff"
    / "work"
    / "artifacts"
    / "20260718-loop-event-semantics-v2"
    / "primary"
    / "semantic_loop_dictionary_v2.csv"
)
BEHAVIOURAL_PRIMARY = (
    FROZEN_REPO_ROOT
    / "research"
    / "observable-behavioural-state"
    / "20260721-behavioural-state-dimensions-screen-v0"
    / "artifacts"
    / "primary"
)
BEHAVIOURAL_COMPACT = BEHAVIOURAL_PRIMARY / "compact_decision_panel.parquet"
BEHAVIOURAL_DIMENSION_LEDGER = BEHAVIOURAL_PRIMARY / "behavioural_dimension_ledger.parquet"
OPENING_PANEL = (
    FROZEN_REPO_ROOT
    / "research"
    / "opening-regime-path"
    / "20260720-opening-regime-path-direction-screen-v0"
    / "artifacts"
    / "primary"
    / "opening_decision_panel.parquet"
)
JOIN_KEYS = ("symbol", "session", "decision_ordinal", "feature_available_timestamp_utc")
LEDGER_KEYS = JOIN_KEYS[:3]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    **details: Any,
) -> None:
    checks.append({"check": name, "passed": bool(passed), **details})


def normalise_join_keys(frame: pd.DataFrame, *, include_timestamp: bool = True) -> pd.DataFrame:
    result = frame.copy()
    result["symbol"] = result["symbol"].astype(str)
    result["session"] = result["session"].astype(str).str[:10]
    result["decision_ordinal"] = pd.to_numeric(result["decision_ordinal"], errors="raise").astype(
        int
    )
    if include_timestamp:
        result["feature_available_timestamp_utc"] = pd.to_datetime(
            result["feature_available_timestamp_utc"], utc=True, errors="raise"
        )
    return result


def maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        return float("inf")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("inf")
    return float(np.max(np.abs(left - right)))


def reconstruct_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
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
    return result.loc[:, list(ALL_DIMENSIONS)]


def raw_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for state in range(8):
        result[f"state_p_{state}_x_signed_pressure"] = (
            frame[f"state_p_{state}"] * frame["signed_pressure"]
        )
        result[f"state_p_{state}_x_signed_exhaustion"] = (
            frame[f"state_p_{state}"] * frame["signed_exhaustion"]
        )
    result["posterior_entropy_x_frustration"] = frame["posterior_entropy"] * frame["frustration"]
    result["posterior_entropy_x_tension"] = frame["posterior_entropy"] * frame["tension"]
    result["transition_probability_x_arousal"] = frame["transition_probability"] * frame["arousal"]
    result["top_second_margin_x_conviction"] = frame["top_second_margin"] * frame["conviction"]
    return result.loc[:, list(INTERACTIONS)]


def load_dictionary() -> LoopDictionary:
    table = pd.read_csv(DICTIONARY_PATH)
    definitions = {}
    for row in table.itertuples(index=False):
        definition = decompose_closed_path(json.loads(str(row.canonical_orientation)))
        if definition.semantic_loop_id != str(row.semantic_loop_id):
            raise AssertionError("dictionary semantic ID differs")
        definitions[definition.semantic_loop_id] = definition
    dictionary = LoopDictionary(definitions, (), version=str(table["dictionary_version"].iloc[0]))
    if dictionary.dictionary_hash != str(table["dictionary_hash"].iloc[0]):
        raise AssertionError("dictionary hash differs")
    return dictionary


def independent_raw_target(
    row: Any,
    engine: FirstNextLoopEventEngine,
) -> tuple[str, str | None, str | None]:
    states = [int(value) for value in row.state_path_through_horizon]
    ordinals = [int(value) for value in row.bar_ordinals_through_horizon]
    if not states:
        return "SOURCE_UNAVAILABLE", None, None
    event_indices = [0]
    event_indices.extend(
        index for index in range(1, len(states)) if states[index] != states[index - 1]
    )
    base = datetime(2024, 1, 1, 14, 30, tzinfo=UTC)
    event_ordinals = [ordinals[index] for index in event_indices]
    trace = engine.scan_state_events(
        [states[index] for index in event_indices],
        bar_ordinals=event_ordinals,
        event_timestamps=[base + timedelta(minutes=5 * bar) for bar in event_ordinals],
        available_timestamps=[base + timedelta(minutes=5 * (bar + 1)) for bar in event_ordinals],
    )
    origin = int(row.repo_bar_start_ordinal)
    candidates = [index for index, bar in enumerate(event_ordinals) if bar <= origin]
    outcome = engine.outcome_for_decision(
        trace,
        decision_id="audit",
        decision_event_index=candidates[-1] if candidates else -1,
        decision_bar_ordinal=origin,
        decision_timestamp=base + timedelta(minutes=5 * (origin + 1)),
        decision_available_timestamp=base + timedelta(minutes=5 * (origin + 1)),
        horizon_bars=6,
        session_end_bar_ordinal=77,
        source_available=bool(row.source_available),
    )
    primary = str(outcome.primary_label)
    if primary in {"SESSION_END", "NO_REGISTERED_LOOP_WITHIN_HORIZON"}:
        return "NO_REGISTERED_COMPLETION", None, None
    if primary == "UNAVAILABLE":
        return "SOURCE_UNAVAILABLE", None, None
    if primary in {"TIED_REGISTERED_COMPLETION", "UNREGISTERED_LOOP"}:
        return primary, None, None
    event = outcome.earliest_registered_events[0]
    raw = {
        "primitive": "REGISTERED_PRIMITIVE",
        "repeat": "REGISTERED_REPEAT",
        "composite": "REGISTERED_COMPOSITE",
    }[str(event.motif_type)]
    return raw, event.semantic_loop_id, event.orientation_id


def _support_details(frame: pd.DataFrame) -> dict[str, Any]:
    stock_counts = frame["symbol"].value_counts()
    return {
        "outcomes": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
        "months": int(frame["year_month"].nunique()),
        "maximum_stock_share": (float(stock_counts.max() / len(frame)) if len(frame) else None),
    }


def _final_support_passes(details: dict[str, Any]) -> bool:
    return bool(
        int(details["outcomes"]) >= 75
        and int(details["sessions"]) >= 30
        and int(details["stocks"]) >= 8
        and int(details["months"]) >= 4
    )


def independent_mapping(panel: pd.DataFrame) -> tuple[dict[str, str], dict[str, Any]]:
    development = panel.loc[
        panel["year"].eq(2024)
        & panel["raw_outcome"].isin(
            [
                "REGISTERED_PRIMITIVE",
                "REGISTERED_REPEAT",
                "REGISTERED_COMPOSITE",
                "UNREGISTERED_LOOP",
                "NO_REGISTERED_COMPLETION",
            ]
        )
    ].copy()
    registered = (
        "REGISTERED_PRIMITIVE",
        "REGISTERED_REPEAT",
        "REGISTERED_COMPOSITE",
    )
    subtype_support = {
        subtype: _support_details(development.loc[development["raw_outcome"].eq(subtype)])
        for subtype in registered
    }
    subtype_pass = {
        subtype: bool(
            _final_support_passes(details)
            and details["maximum_stock_share"] is not None
            and float(details["maximum_stock_share"]) <= 0.30
        )
        for subtype, details in subtype_support.items()
    }
    mapping = {
        subtype: subtype if subtype_pass[subtype] else "OTHER_REGISTERED_COMPLETION"
        for subtype in registered
    }
    mapping.update(
        {
            "UNREGISTERED_LOOP": "UNREGISTERED_LOOP",
            "NO_REGISTERED_COMPLETION": "NO_REGISTERED_COMPLETION",
        }
    )
    development["target_class"] = development["raw_outcome"].map(mapping)
    initial_classes = sorted(development["target_class"].dropna().astype(str).unique())
    initial_support = {
        label: _support_details(development.loc[development["target_class"].eq(label)])
        for label in initial_classes
    }
    fallback = len(initial_classes) < 4 or any(
        int(details["outcomes"]) < 75 for details in initial_support.values()
    )
    if fallback:
        mapping = {
            **{subtype: "REGISTERED_COMPLETION" for subtype in registered},
            "UNREGISTERED_LOOP": "UNREGISTERED_LOOP",
            "NO_REGISTERED_COMPLETION": "NO_REGISTERED_COMPLETION",
        }
        development["target_class"] = development["raw_outcome"].map(mapping)
    final_classes = sorted(development["target_class"].dropna().astype(str).unique())
    final_support = {
        label: _support_details(development.loc[development["target_class"].eq(label)])
        for label in final_classes
    }
    return mapping, {
        "target_variant": (
            "three_class_fallback"
            if fallback
            else ("five_classes" if len(final_classes) == 5 else "four_classes")
        ),
        "fallback_required": fallback,
        "registered_subtype_support": subtype_support,
        "registered_subtype_pass": subtype_pass,
        "final_target_classes": final_classes,
        "final_development_class_support": final_support,
        "development_support_passed": len(final_classes) >= 3
        and all(_final_support_passes(details) for details in final_support.values()),
    }


def independent_pool(raw: str, mapping: dict[str, str]) -> str | None:
    if raw in {"TIED_REGISTERED_COMPLETION", "SOURCE_UNAVAILABLE"}:
        return None
    if raw not in mapping:
        raise AssertionError(f"unknown raw target {raw}")
    return mapping[raw]


def manual_probabilities(frame: pd.DataFrame, payload: dict[str, Any]) -> np.ndarray:
    matrix = frame.loc[:, payload["features"]].to_numpy(dtype=float)
    mean = np.asarray(payload["scaler_mean"], dtype=float)
    scale = np.asarray(payload["scaler_scale"], dtype=float)
    coefficient = np.asarray(payload["coefficient"], dtype=float)
    intercept = np.asarray(payload["intercept"], dtype=float)
    logits = ((matrix - mean) / scale) @ coefficient.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    return exponential / exponential.sum(axis=1, keepdims=True)


def metric_values(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    class_order: tuple[str, ...],
) -> dict[str, float]:
    class_index = {label: index for index, label in enumerate(class_order)}
    target = frame["target_class"].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    realised = probabilities[np.arange(len(target)), target]
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.asarray(
        [int(np.flatnonzero(order[index] == value)[0]) + 1 for index, value in enumerate(target)]
    )
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(target)), target] = 1.0
    entropy = -np.where(
        probabilities > 0.0,
        probabilities * np.log(np.clip(probabilities, np.finfo(float).tiny, 1.0)),
        0.0,
    ).sum(axis=1)
    return {
        "multiclass_log_loss": float(np.average(-np.log(realised), weights=weights)),
        "multiclass_brier": float(
            np.average(np.square(probabilities - one_hot).sum(axis=1), weights=weights)
        ),
        "top_one_accuracy": float(np.average(ranks <= 1, weights=weights)),
        "top_two_accuracy": float(np.average(ranks <= 2, weights=weights)),
        "top_three_accuracy": float(np.average(ranks <= 3, weights=weights)),
        "mean_reciprocal_rank": float(np.average(1.0 / ranks, weights=weights)),
        "realised_probability": float(np.average(realised, weights=weights)),
        "prediction_entropy": float(np.average(entropy, weights=weights)),
        "effective_candidate_count": float(np.average(np.exp(entropy), weights=weights)),
    }


def comparison_values(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    class_order: tuple[str, ...],
) -> dict[str, float]:
    """Independently calculate every decision-critical paired model increment."""

    baseline_metrics = metric_values(frame, baseline, class_order)
    candidate_metrics = metric_values(frame, candidate, class_order)
    return {
        "log_loss_improvement": baseline_metrics["multiclass_log_loss"]
        - candidate_metrics["multiclass_log_loss"],
        "brier_improvement": baseline_metrics["multiclass_brier"]
        - candidate_metrics["multiclass_brier"],
        "top_two_improvement": candidate_metrics["top_two_accuracy"]
        - baseline_metrics["top_two_accuracy"],
        "realised_probability_improvement": candidate_metrics["realised_probability"]
        - baseline_metrics["realised_probability"],
        "prediction_entropy_reduction": baseline_metrics["prediction_entropy"]
        - candidate_metrics["prediction_entropy"],
    }


def permutation_preserves_slate_contract(
    original: pd.DataFrame,
    permuted: pd.DataFrame,
) -> bool:
    """Verify a null draw changes only whole behavioural bundles within each slate."""

    immutable = [
        "symbol",
        "session",
        "decision_ordinal",
        "feature_available_timestamp_utc",
        "target_class",
        *M0_FEATURES,
    ]
    if not original.loc[:, immutable].equals(permuted.loc[:, immutable]):
        return False
    for _, indices in original.groupby("slate_id", sort=True).groups.items():
        labels = list(indices)
        before = sorted(
            tuple(float(value) for value in row)
            for row in original.loc[labels, list(BEHAVIOURAL_FEATURES)].to_numpy()
        )
        after = sorted(
            tuple(float(value) for value in row)
            for row in permuted.loc[labels, list(BEHAVIOURAL_FEATURES)].to_numpy()
        )
        if before != after:
            return False
    return True


def fit_independent(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    class_order: tuple[str, ...],
) -> tuple[StandardScaler, LogisticRegression]:
    target_map = {label: index for index, label in enumerate(class_order)}
    matrix = frame.loc[:, list(features)].to_numpy(dtype=float)
    target = frame["target_class"].map(target_map).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    scaler = StandardScaler().fit(matrix)
    model = LogisticRegression(
        penalty="l2",
        C=0.25,
        solver="lbfgs",
        max_iter=300,
        class_weight=None,
        random_state=20260721,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        model.fit(scaler.transform(matrix), target, sample_weight=weights)
    return scaler, model


def permute_bundle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy()
    generator = np.random.default_rng(seed)
    for _, indices in result.groupby("slate_id", sort=True).groups.items():
        labels = list(indices)
        bundles = frame.loc[labels, list(BEHAVIOURAL_FEATURES)].to_numpy(copy=True)
        result.loc[labels, list(BEHAVIOURAL_FEATURES)] = bundles[generator.permutation(len(labels))]
    return result


def apply_interactions(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = development.copy()
    ass = assessment.copy()
    dev_raw = raw_interactions(dev)
    ass_raw = raw_interactions(ass)
    for feature in INTERACTIONS:
        lower = float(dev_raw[feature].quantile(0.01, interpolation="linear"))
        upper = float(dev_raw[feature].quantile(0.99, interpolation="linear"))
        dev[feature] = dev_raw[feature].clip(lower, upper)
        ass[feature] = ass_raw[feature].clip(lower, upper)
    return dev, ass


def audit_artifacts(artifacts: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    required = (
        "contract.json",
        "source_manifest.json",
        "protected_boundary_audit.json",
        "behavioural_ledger_reconstruction.json",
        "v2_population_reconstruction.json",
        "target_class_mapping.json",
        "raw_subtype_census.csv",
        "feature_manifest.json",
        "interaction_manifest.json",
        "decision_panel.parquet",
        "model_configurations.json",
        "model_coefficients.json",
        "assessment_predictions.parquet",
        "pooled_metrics.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "class_metrics.csv",
        "funnel_metrics.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
        "concentration_metrics.csv",
        "decision.json",
        "determinism_check.json",
        "report.md",
    )
    try:
        _check(
            checks,
            "required_artifacts",
            all((artifacts / name).is_file() for name in required),
            missing=[name for name in required if not (artifacts / name).is_file()],
        )
        contract = json.loads((artifacts / "contract.json").read_text(encoding="utf-8"))
        decision = json.loads((artifacts / "decision.json").read_text(encoding="utf-8"))
        safety_passed = all(
            contract.get(key) == expected
            and contract.get("safety", {}).get(key) == expected
            and decision.get(key) == expected
            for key, expected in SAFETY_FLAGS.items()
        )
        _check(checks, "safety_flags", safety_passed)

        panel = pd.read_parquet(artifacts / "decision_panel.parquet")
        predictions = pd.read_parquet(artifacts / "assessment_predictions.parquet")
        _check(
            checks,
            "dates_and_protected_boundary",
            panel["session"].astype(str).min() >= "2024-01-01"
            and panel["session"].astype(str).max() <= "2025-08-22"
            and json.loads(
                (artifacts / "protected_boundary_audit.json").read_text(encoding="utf-8")
            )["protected_rows_materialised"]
            == 0,
            minimum_session=panel["session"].astype(str).min(),
            maximum_session=panel["session"].astype(str).max(),
        )

        compact = normalise_join_keys(
            pd.read_parquet(
                BEHAVIOURAL_COMPACT,
                columns=[*JOIN_KEYS, "return_gap"],
            )
        )
        ledger = normalise_join_keys(
            pd.read_parquet(
                BEHAVIOURAL_DIMENSION_LEDGER,
                columns=[*LEDGER_KEYS, *BEHAVIOURAL_Z_COMPONENTS, *ALL_DIMENSIONS],
            ),
            include_timestamp=False,
        )
        frozen_behaviour = compact.merge(
            ledger,
            on=list(LEDGER_KEYS),
            how="inner",
            validate="one_to_one",
        )
        behaviour_columns = (*BEHAVIOURAL_Z_COMPONENTS, "return_gap", *ALL_DIMENSIONS)
        panel_behaviour = normalise_join_keys(panel.loc[:, [*JOIN_KEYS, *behaviour_columns]])
        behaviour_join = panel_behaviour.merge(
            frozen_behaviour.loc[:, [*JOIN_KEYS, *behaviour_columns]],
            on=list(JOIN_KEYS),
            how="left",
            validate="one_to_one",
            suffixes=("__panel", "__frozen"),
            indicator=True,
        )
        panel_values = behaviour_join.loc[
            :, [f"{feature}__panel" for feature in behaviour_columns]
        ].to_numpy(dtype=float)
        frozen_values = behaviour_join.loc[
            :, [f"{feature}__frozen" for feature in behaviour_columns]
        ].to_numpy(dtype=float)
        frozen_behaviour_error = maximum_absolute_difference(panel_values, frozen_values)
        reconstructed = reconstruct_dimensions(frozen_behaviour)
        formula_error = maximum_absolute_difference(
            reconstructed.to_numpy(dtype=float),
            frozen_behaviour.loc[:, list(ALL_DIMENSIONS)].to_numpy(dtype=float),
        )
        _check(
            checks,
            "frozen_behavioural_values",
            len(frozen_behaviour) == len(panel)
            and behaviour_join["_merge"].eq("both").all()
            and frozen_behaviour_error <= 1e-12
            and formula_error <= 1e-12,
            frozen_source_rows=len(frozen_behaviour),
            joined_panel_rows=len(behaviour_join),
            maximum_frozen_source_error=frozen_behaviour_error,
            maximum_formula_reconstruction_error=formula_error,
        )

        posterior = panel.loc[:, list(STATE_PROBABILITIES)].to_numpy(dtype=float)
        opening_columns = [
            *JOIN_KEYS,
            "current_state",
            "maximum_posterior_probability",
            "posterior_entropy",
            *(f"posterior_state_{state}" for state in range(8)),
        ]
        frozen_opening = normalise_join_keys(
            pd.read_parquet(OPENING_PANEL, columns=opening_columns)
        )
        panel_opening = normalise_join_keys(
            panel.loc[
                :,
                [
                    *JOIN_KEYS,
                    "current_state",
                    "hard_top_state",
                    *STATE_PROBABILITIES,
                    *(f"posterior_state_{state}" for state in range(8)),
                    "posterior_entropy",
                    "posterior_entropy_reproduced",
                    "top_state_probability",
                    "top_second_margin",
                ],
            ]
        )
        opening_join = panel_opening.merge(
            frozen_opening,
            on=list(JOIN_KEYS),
            how="left",
            validate="one_to_one",
            suffixes=("__panel", "__frozen"),
            indicator=True,
        )
        frozen_posterior = opening_join.loc[
            :, [f"posterior_state_{state}__frozen" for state in range(8)]
        ].to_numpy(dtype=float)
        panel_frozen_copy = opening_join.loc[
            :, [f"posterior_state_{state}__panel" for state in range(8)]
        ].to_numpy(dtype=float)
        source_v2_error = max(
            maximum_absolute_difference(posterior, frozen_posterior),
            maximum_absolute_difference(panel_frozen_copy, frozen_posterior),
            maximum_absolute_difference(
                opening_join["posterior_entropy__panel"].to_numpy(dtype=float),
                opening_join["posterior_entropy__frozen"].to_numpy(dtype=float),
            ),
            maximum_absolute_difference(
                opening_join["top_state_probability"].to_numpy(dtype=float),
                opening_join["maximum_posterior_probability"].to_numpy(dtype=float),
            ),
        )
        entropy = -np.where(
            posterior > 0.0,
            posterior * np.log(np.clip(posterior, np.finfo(float).tiny, 1.0)),
            0.0,
        ).sum(axis=1)
        ordered = np.sort(posterior, axis=1)
        derived_v2_error = max(
            float(np.max(np.abs(posterior.sum(axis=1) - 1.0))),
            float(np.max(np.abs(entropy - panel["posterior_entropy"].to_numpy(dtype=float)))),
            float(
                np.max(
                    np.abs(ordered[:, -1] - panel["top_state_probability"].to_numpy(dtype=float))
                )
            ),
            float(
                np.max(
                    np.abs(
                        ordered[:, -1]
                        - ordered[:, -2]
                        - panel["top_second_margin"].to_numpy(dtype=float)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        panel["persistence_probability"].to_numpy(dtype=float)
                        + panel["transition_probability"].to_numpy(dtype=float)
                        - 1.0
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        panel["remaining_session_bars"].to_numpy(dtype=float)
                        - (78 - panel["decision_ordinal"].to_numpy(dtype=float))
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        panel["checkpoint"].to_numpy(dtype=float)
                        - panel["decision_ordinal"].eq(12).to_numpy(dtype=float)
                    )
                )
            ),
        )
        hard_state_matches = bool(
            opening_join["current_state__panel"]
            .astype(int)
            .eq(opening_join["current_state__frozen"].astype(int))
            .all()
            and opening_join["hard_top_state"]
            .astype(int)
            .eq(opening_join["current_state__frozen"].astype(int))
            .all()
        )
        v2_population = json.loads(
            (artifacts / "v2_population_reconstruction.json").read_text(encoding="utf-8")
        )
        _check(
            checks,
            "v2_posterior_values",
            len(frozen_opening) == int(v2_population["rows_expected"])
            and len(frozen_opening) == int(v2_population["rows_reconstructed"])
            and opening_join["_merge"].eq("both").all()
            and hard_state_matches
            and source_v2_error <= 1e-12
            and derived_v2_error <= 1e-12,
            frozen_source_rows=len(frozen_opening),
            joined_panel_rows=len(opening_join),
            maximum_frozen_source_error=source_v2_error,
            maximum_derived_error=derived_v2_error,
            hard_state_matches=hard_state_matches,
        )

        timestamps = pd.to_datetime(panel["feature_available_timestamp_utc"], utc=True)
        clocks = timestamps.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        expected_clocks = panel["decision_ordinal"].map({6: "10:00", 12: "10:30"})
        _check(
            checks,
            "checkpoint_timestamps",
            panel["decision_ordinal"].isin([6, 12]).all() and clocks.eq(expected_clocks).all(),
        )

        dictionary = load_dictionary()
        engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
        target_matches = []
        horizon_matches = []
        for row in panel.itertuples(index=False):
            raw, semantic, orientation = independent_raw_target(row, engine)
            stored_semantic = None if pd.isna(row.semantic_loop_id) else str(row.semantic_loop_id)
            stored_orientation = None if pd.isna(row.orientation) else str(row.orientation)
            target_matches.append(
                raw == str(row.raw_outcome)
                and semantic == stored_semantic
                and orientation == stored_orientation
            )
            ordinals = [int(value) for value in row.bar_ordinals_through_horizon]
            states = [int(value) for value in row.state_path_through_horizon]
            expected_ordinals = list(range(int(row.repo_bar_start_ordinal) + 7))
            horizon_matches.append(
                (not bool(row.source_available) and not ordinals and not states)
                or (
                    bool(row.source_available)
                    and ordinals == expected_ordinals
                    and len(states) == len(expected_ordinals)
                )
            )
        _check(
            checks,
            "six_bar_first_event_target",
            all(target_matches) and all(horizon_matches),
            rows_checked=len(panel),
            target_mismatches=int(len(target_matches) - sum(target_matches)),
            horizon_mismatches=int(len(horizon_matches) - sum(horizon_matches)),
        )
        ties = panel["raw_outcome"].eq("TIED_REGISTERED_COMPLETION")
        unavailable = panel["raw_outcome"].eq("SOURCE_UNAVAILABLE")
        expected_excluded = ties | unavailable
        _check(
            checks,
            "tie_and_unavailable_exclusions",
            panel["target_excluded"].astype(bool).eq(expected_excluded).all()
            and bool((~panel.loc[expected_excluded, "scoring_eligible"]).all())
            and panel.loc[expected_excluded, "target_class"].isna().all()
            and panel["source_available"].astype(bool).eq(~unavailable).all(),
            tied_rows=int(ties.sum()),
            unavailable_rows=int(unavailable.sum()),
        )

        mapping_payload = json.loads(
            (artifacts / "target_class_mapping.json").read_text(encoding="utf-8")
        )
        mapping, independent_mapping_audit = independent_mapping(panel)
        _check(
            checks,
            "development_only_target_pooling_and_fallback",
            mapping == mapping_payload["frozen_mapping"]
            and independent_mapping_audit["target_variant"] == mapping_payload["target_variant"]
            and independent_mapping_audit["fallback_required"]
            == mapping_payload["fallback_required"]
            and bool(independent_mapping_audit["development_support_passed"])
            and set(independent_mapping_audit["final_target_classes"])
            == set(mapping_payload["final_target_classes"]),
            independent_mapping=mapping,
            independent_variant=independent_mapping_audit["target_variant"],
        )
        independently_pooled = [independent_pool(str(raw), mapping) for raw in panel["raw_outcome"]]
        stored_pooled = panel["target_class"].where(panel["target_class"].notna(), None).tolist()
        _check(
            checks,
            "frozen_final_target_mapping",
            independently_pooled == stored_pooled,
        )
        census = pd.read_csv(artifacts / "raw_subtype_census.csv")
        census_matches = True
        for row in census.itertuples(index=False):
            development_subtype = panel.loc[
                panel["year"].eq(2024) & panel["raw_outcome"].eq(row.raw_subtype)
            ]
            assessment_subtype = panel.loc[
                panel["year"].eq(2025) & panel["raw_outcome"].eq(row.raw_subtype)
            ]
            census_matches = census_matches and (
                int(row.development_outcomes) == len(development_subtype)
                and int(row.assessment_outcomes) == len(assessment_subtype)
                and int(row.development_sessions) == development_subtype["session"].nunique()
                and int(row.assessment_sessions) == assessment_subtype["session"].nunique()
                and int(row.development_stocks) == development_subtype["symbol"].nunique()
                and int(row.assessment_stocks) == assessment_subtype["symbol"].nunique()
                and int(row.development_months) == development_subtype["year_month"].nunique()
                and int(row.assessment_months) == assessment_subtype["year_month"].nunique()
            )
        _check(
            checks,
            "raw_subtype_census_support",
            census_matches and len(census) == 5,
        )

        feature_manifest = json.loads(
            (artifacts / "feature_manifest.json").read_text(encoding="utf-8")
        )
        _check(
            checks,
            "model_feature_ladder",
            tuple(feature_manifest["M0"]) == M0_FEATURES
            and tuple(feature_manifest["M1"]) == M1_FEATURES
            and tuple(feature_manifest["M2"]) == M2_FEATURES,
        )
        model_configurations = json.loads(
            (artifacts / "model_configurations.json").read_text(encoding="utf-8")
        )
        requested = model_configurations["requested_configuration"]
        _check(
            checks,
            "fixed_model_configuration",
            requested
            == {
                "penalty": "l2",
                "C": 0.25,
                "solver": "lbfgs",
                "max_iter": 300,
                "multi_class": "multinomial",
                "class_weight": None,
                "random_state": 20260721,
                "n_jobs": 1,
            }
            and int(model_configurations["primary_fitted_model_count"]) == 3
            and int(model_configurations["null_refits"]["draws"]) == 10,
        )
        interaction_manifest = json.loads(
            (artifacts / "interaction_manifest.json").read_text(encoding="utf-8")
        )
        raw = raw_interactions(panel)
        development_mask = panel["year"].eq(2024) & panel["scoring_eligible"]
        clipped = raw.copy()
        bound_error = 0.0
        finite_bounds = bool(development_mask.any())
        for feature in INTERACTIONS:
            stored_bounds = interaction_manifest["clip_bounds"][feature]
            actual_lower = float(
                raw.loc[development_mask, feature].quantile(0.01, interpolation="linear")
            )
            actual_upper = float(
                raw.loc[development_mask, feature].quantile(0.99, interpolation="linear")
            )
            stored_lower = float(stored_bounds["q01"])
            stored_upper = float(stored_bounds["q99"])
            feature_bounds = np.asarray(
                [actual_lower, actual_upper, stored_lower, stored_upper], dtype=float
            )
            if not np.isfinite(feature_bounds).all():
                finite_bounds = False
                bound_error = float("inf")
            else:
                bound_error = max(
                    bound_error,
                    abs(actual_lower - stored_lower),
                    abs(actual_upper - stored_upper),
                )
            clipped[feature] = raw[feature].clip(stored_lower, stored_upper)
        interaction_error = float(
            np.max(
                np.abs(
                    clipped.to_numpy(dtype=float)
                    - panel.loc[:, list(INTERACTIONS)].to_numpy(dtype=float)
                )
            )
        )
        _check(
            checks,
            "all_20_interactions_and_development_clipping",
            len(INTERACTIONS) == 20
            and finite_bounds
            and bound_error <= 1e-12
            and interaction_error <= 1e-12,
            development_rows=int(development_mask.sum()),
            finite_bounds=finite_bounds,
            bound_error=bound_error,
            interaction_error=interaction_error,
        )

        coefficient_payload = json.loads(
            (artifacts / "model_coefficients.json").read_text(encoding="utf-8")
        )["models"]
        scoring = panel.loc[panel["scoring_eligible"]].copy()
        development = scoring.loc[scoring["year"].eq(2024)].copy()
        assessment = scoring.loc[scoring["year"].eq(2025)].copy()
        class_order = tuple(coefficient_payload["M0"]["class_order"])
        assessment_support = (
            assessment["target_class"].value_counts().reindex(class_order, fill_value=0)
        )
        assessment_stock_share = assessment["symbol"].value_counts(normalize=True)
        assessment_class_share = assessment["target_class"].value_counts(normalize=True)
        support_passed = bool(
            len(assessment) >= 3000
            and assessment["session"].nunique() >= 100
            and assessment["symbol"].nunique() >= 15
            and assessment["year_month"].nunique() >= 6
            and len(class_order) >= 3
            and assessment_support.ge(50).all()
            and float(assessment_stock_share.max()) <= 0.10
            and float(assessment_class_share.max()) <= 0.75
        )
        _check(
            checks,
            "assessment_support_and_concentration",
            support_passed and bool(decision["support"]["passed"]),
            rows=len(assessment),
            maximum_stock_share=float(assessment_stock_share.max()),
            maximum_class_share=float(assessment_class_share.max()),
        )
        coefficient_error = 0.0
        preprocessing_error = 0.0
        independent_models: dict[str, tuple[StandardScaler, LogisticRegression]] = {}
        for name, features in MODEL_FEATURES.items():
            scaler, model = fit_independent(development, features, class_order)
            independent_models[name] = (scaler, model)
            stored = coefficient_payload[name]
            coefficient_error = max(
                coefficient_error,
                float(np.max(np.abs(model.coef_ - np.asarray(stored["coefficient"])))),
                float(
                    np.max(np.abs(model.intercept_ - np.asarray(stored["intercept"]))),
                ),
            )
            preprocessing_error = max(
                preprocessing_error,
                float(
                    np.max(np.abs(scaler.mean_ - np.asarray(stored["scaler_mean"]))),
                ),
                float(
                    np.max(np.abs(scaler.scale_ - np.asarray(stored["scaler_scale"]))),
                ),
            )
        _check(
            checks,
            "development_only_preprocessing_fit_and_coefficients",
            coefficient_error <= 1e-12 and preprocessing_error <= 1e-12,
            coefficient_error=coefficient_error,
            preprocessing_error=preprocessing_error,
        )
        _check(
            checks,
            "2024_fit_2025_assessment",
            set(development["year"].unique()) == {2024}
            and set(assessment["year"].unique()) == {2025}
            and predictions["session"].astype(str).str[:4].eq("2025").all(),
        )

        prediction_keys = ["symbol", "session", "decision_ordinal"]
        assessment_ordered = predictions.loc[:, prediction_keys].merge(
            assessment,
            on=prediction_keys,
            how="left",
            validate="one_to_one",
        )
        manual_error = 0.0
        stored_probabilities: dict[str, np.ndarray] = {}
        for name in MODEL_FEATURES:
            manual = manual_probabilities(assessment_ordered, coefficient_payload[name])
            stored_matrix = predictions.loc[
                :, [f"probability__{name}__{label}" for label in class_order]
            ].to_numpy(dtype=float)
            stored_probabilities[name] = stored_matrix
            manual_error = max(manual_error, float(np.max(np.abs(manual - stored_matrix))))
        _check(
            checks,
            "manual_probability_reconstruction",
            manual_error <= 1e-12 and len(predictions) >= 100,
            rows_checked=len(predictions),
            maximum_probability_error=manual_error,
        )
        census_probability_error = 0.0
        for census_row in census.itertuples(index=False):
            mask = assessment_ordered["raw_outcome"].eq(census_row.raw_subtype).to_numpy()
            if not mask.any():
                continue
            class_index = class_order.index(str(census_row.final_pooled_class))
            weights = assessment_ordered.loc[mask, "row_weight"].to_numpy(dtype=float)
            for model in MODEL_FEATURES:
                actual = float(
                    np.average(stored_probabilities[model][mask, class_index], weights=weights)
                )
                stored = float(
                    getattr(
                        census_row,
                        f"assessment_mean_{model}_probability_final_pooled_class",
                    )
                )
                census_probability_error = max(census_probability_error, abs(actual - stored))
        _check(
            checks,
            "raw_subtype_census_probabilities",
            census_probability_error <= 1e-12,
            maximum_probability_error=census_probability_error,
        )

        pooled = pd.read_csv(artifacts / "pooled_metrics.csv")
        metric_error = 0.0
        audited_pooled: dict[str, dict[str, float]] = {}
        metric_pairs = {
            "multiclass_log_loss": "multiclass_log_loss",
            "multiclass_brier": "multiclass_brier",
            "top_one_accuracy": "top_one_accuracy",
            "top_two_accuracy": "top_two_accuracy",
            "top_three_accuracy": "top_three_accuracy",
            "mean_reciprocal_rank": "mean_reciprocal_rank",
            "realised_probability": "mean_probability_realised_class",
            "prediction_entropy": "prediction_entropy",
            "effective_candidate_count": "effective_candidate_count",
        }
        for name in MODEL_FEATURES:
            actual = metric_values(assessment_ordered, stored_probabilities[name], class_order)
            audited_pooled[name] = actual
            stored = pooled.loc[pooled["model"].eq(name)].iloc[0]
            for calculated, stored_name in metric_pairs.items():
                metric_error = max(
                    metric_error,
                    abs(actual[calculated] - float(stored[stored_name])),
                )
        _check(
            checks,
            "primary_multiclass_metrics",
            metric_error <= 1e-12,
            maximum_metric_error=metric_error,
        )

        stability_error = 0.0
        stability_structure = True
        grouped_metrics: dict[str, dict[str | int, dict[str, dict[str, float]]]] = {
            "year_month": {},
            "decision_ordinal": {},
        }
        stability_files = {
            "year_month": pd.read_csv(artifacts / "monthly_metrics.csv"),
            "decision_ordinal": pd.read_csv(artifacts / "checkpoint_metrics.csv"),
        }
        for group_column, stored_frame in stability_files.items():
            group_values = sorted(assessment_ordered[group_column].unique())
            expected_rows = len(group_values) * len(MODEL_FEATURES)
            stability_structure = stability_structure and len(stored_frame) == expected_rows
            for raw_group in group_values:
                group: str | int = (
                    int(raw_group) if group_column == "decision_ordinal" else str(raw_group)
                )
                mask = assessment_ordered[group_column].eq(raw_group).to_numpy()
                subset = assessment_ordered.loc[mask].reset_index(drop=True)
                grouped_metrics[group_column][group] = {}
                for name in MODEL_FEATURES:
                    actual = metric_values(subset, stored_probabilities[name][mask], class_order)
                    grouped_metrics[group_column][group][name] = actual
                    stored_rows = stored_frame.loc[
                        stored_frame[group_column].astype(str).eq(str(group))
                        & stored_frame["model"].eq(name)
                    ]
                    if len(stored_rows) != 1:
                        stability_structure = False
                        continue
                    stored = stored_rows.iloc[0]
                    for calculated, stored_name in metric_pairs.items():
                        stability_error = max(
                            stability_error,
                            abs(actual[calculated] - float(stored[stored_name])),
                        )
        _check(
            checks,
            "monthly_and_checkpoint_metrics",
            stability_structure and stability_error <= 1e-12,
            maximum_metric_error=stability_error,
        )

        bootstrap = pd.read_csv(artifacts / "bootstrap_metrics.csv")
        sessions = sorted(assessment_ordered["session"].astype(str).unique())
        by_session = {
            session: np.flatnonzero(
                assessment_ordered["session"].astype(str).eq(session).to_numpy()
            )
            for session in sessions
        }
        bootstrap_specifications = {
            "m1_minus_m0": (
                "M0",
                "M1",
                (
                    "log_loss_improvement",
                    "brier_improvement",
                    "top_two_improvement",
                    "realised_probability_improvement",
                ),
            ),
            "m2_minus_m1": (
                "M1",
                "M2",
                (
                    "log_loss_improvement",
                    "brier_improvement",
                    "top_two_improvement",
                    "realised_probability_improvement",
                    "prediction_entropy_reduction",
                ),
            ),
        }
        real_comparisons: dict[str, dict[str, float]] = {}
        expected_bootstrap_draws: dict[str, list[float]] = {}
        for prefix, (baseline_name, candidate_name, metrics) in bootstrap_specifications.items():
            real_comparisons[prefix] = comparison_values(
                assessment_ordered,
                stored_probabilities[baseline_name],
                stored_probabilities[candidate_name],
                class_order,
            )
            for metric in metrics:
                expected_bootstrap_draws[f"{prefix}_{metric}"] = []
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        for _ in range(BOOTSTRAP_DRAWS):
            sampled_sessions = generator.choice(sessions, size=len(sessions), replace=True)
            bootstrap_indices = np.concatenate(
                [by_session[str(value)] for value in sampled_sessions]
            )
            sampled = assessment_ordered.iloc[bootstrap_indices].reset_index(drop=True)
            for prefix, (
                baseline_name,
                candidate_name,
                metrics,
            ) in bootstrap_specifications.items():
                comparison = comparison_values(
                    sampled,
                    stored_probabilities[baseline_name][bootstrap_indices],
                    stored_probabilities[candidate_name][bootstrap_indices],
                    class_order,
                )
                for metric in metrics:
                    expected_bootstrap_draws[f"{prefix}_{metric}"].append(comparison[metric])
        bootstrap_error = 0.0
        bootstrap_structure = set(bootstrap["metric"]) == set(expected_bootstrap_draws)
        bootstrap_summaries: dict[str, dict[str, float]] = {}
        for metric, expected_values in expected_bootstrap_draws.items():
            rows = bootstrap.loc[bootstrap["metric"].eq(metric)]
            if len(rows) != 1:
                bootstrap_structure = False
                continue
            row = rows.iloc[0]
            expected = np.asarray(expected_values, dtype=float)
            stored_draws = np.asarray(json.loads(str(row["draw_values"])), dtype=float)
            if stored_draws.shape != expected.shape:
                bootstrap_error = float("inf")
                continue
            prefix = "m1_minus_m0" if metric.startswith("m1_minus_m0") else "m2_minus_m1"
            suffix = metric.removeprefix(f"{prefix}_")
            summary = {
                "real_value": real_comparisons[prefix][suffix],
                "draw_mean": float(expected.mean()),
                "interval_90_lower": float(np.quantile(expected, 0.05)),
                "interval_90_upper": float(np.quantile(expected, 0.95)),
                "interval_95_lower": float(np.quantile(expected, 0.025)),
                "interval_95_upper": float(np.quantile(expected, 0.975)),
            }
            bootstrap_summaries[metric] = summary
            bootstrap_error = max(
                bootstrap_error,
                float(np.max(np.abs(stored_draws - expected))),
                *(abs(float(row[key]) - value) for key, value in summary.items()),
            )
            bootstrap_structure = bootstrap_structure and (
                int(row["draw_count"]) == BOOTSTRAP_DRAWS and int(row["seed"]) == BOOTSTRAP_SEED
            )
        _check(
            checks,
            "paired_session_block_bootstrap",
            bootstrap_structure and bootstrap_error <= 1e-12,
            draws_checked=BOOTSTRAP_DRAWS,
            maximum_draw_or_summary_error=bootstrap_error,
        )

        null = pd.read_csv(artifacts / "null_metrics.csv")
        permutation_contract_passed = True
        first_dev_null: pd.DataFrame | None = None
        first_ass_null: pd.DataFrame | None = None
        for draw in range(NULL_DRAWS):
            dev_null_draw = permute_bundle(development, NULL_SEED + draw * 2)
            ass_null_draw = permute_bundle(assessment_ordered, NULL_SEED + draw * 2 + 1)
            permutation_contract_passed = permutation_contract_passed and (
                permutation_preserves_slate_contract(development, dev_null_draw)
                and permutation_preserves_slate_contract(assessment_ordered, ass_null_draw)
            )
            if draw == 0:
                first_dev_null = dev_null_draw
                first_ass_null = ass_null_draw
        if first_dev_null is None or first_ass_null is None:
            raise AssertionError("null audit did not construct its first draw")
        dev_null, ass_null = first_dev_null, first_ass_null
        dev_null, ass_null = apply_interactions(dev_null, ass_null)
        m1_scaler, m1_model = fit_independent(dev_null, M1_FEATURES, class_order)
        m2_scaler, m2_model = fit_independent(dev_null, M2_FEATURES, class_order)
        m0_probability = manual_probabilities(ass_null, coefficient_payload["M0"])
        m1_probability = m1_model.predict_proba(
            m1_scaler.transform(ass_null.loc[:, list(M1_FEATURES)].to_numpy(dtype=float))
        )
        m2_probability = m2_model.predict_proba(
            m2_scaler.transform(ass_null.loc[:, list(M2_FEATURES)].to_numpy(dtype=float))
        )
        null0 = metric_values(ass_null, m0_probability, class_order)
        null1 = metric_values(ass_null, m1_probability, class_order)
        null2 = metric_values(ass_null, m2_probability, class_order)
        first_null = {
            "m1_minus_m0_log_loss_improvement": null0["multiclass_log_loss"]
            - null1["multiclass_log_loss"],
            "m1_minus_m0_brier_improvement": null0["multiclass_brier"] - null1["multiclass_brier"],
            "m2_minus_m1_log_loss_improvement": null1["multiclass_log_loss"]
            - null2["multiclass_log_loss"],
            "m2_minus_m1_brier_improvement": null1["multiclass_brier"] - null2["multiclass_brier"],
            "m2_minus_m1_top_two_improvement": null2["top_two_accuracy"]
            - null1["top_two_accuracy"],
        }
        real_null_values = {
            "m1_minus_m0_log_loss_improvement": real_comparisons["m1_minus_m0"][
                "log_loss_improvement"
            ],
            "m1_minus_m0_brier_improvement": real_comparisons["m1_minus_m0"]["brier_improvement"],
            "m2_minus_m1_log_loss_improvement": real_comparisons["m2_minus_m1"][
                "log_loss_improvement"
            ],
            "m2_minus_m1_brier_improvement": real_comparisons["m2_minus_m1"]["brier_improvement"],
            "m2_minus_m1_top_two_improvement": real_comparisons["m2_minus_m1"][
                "top_two_improvement"
            ],
        }
        null_error = 0.0
        null_structure = set(null["metric"]) == set(first_null)
        null_summaries: dict[str, dict[str, float]] = {}
        for metric, expected_first in first_null.items():
            rows = null.loc[null["metric"].eq(metric)]
            if len(rows) != 1:
                null_structure = False
                continue
            row = rows.iloc[0]
            stored_draws = np.asarray(json.loads(str(row["draw_values"])), dtype=float)
            if stored_draws.shape != (NULL_DRAWS,) or not np.isfinite(stored_draws).all():
                null_error = float("inf")
                continue
            summary = {
                "real_value": real_null_values[metric],
                "null_mean": float(stored_draws.mean()),
                "null_q90": float(np.quantile(stored_draws, 0.90)),
                "real_percentile": float(np.mean(stored_draws <= real_null_values[metric])),
            }
            null_summaries[metric] = summary
            null_error = max(
                null_error,
                abs(float(stored_draws[0]) - expected_first),
                *(abs(float(row[key]) - value) for key, value in summary.items()),
            )
            null_structure = null_structure and (
                int(row["draw_count"]) == NULL_DRAWS and int(row["seed_base"]) == NULL_SEED
            )
        _check(
            checks,
            "within_slate_behavioural_null",
            permutation_contract_passed and null_structure and null_error <= 1e-12,
            permutation_draws_checked=NULL_DRAWS,
            independently_refitted_draws=1,
            maximum_first_draw_or_summary_error=null_error,
        )

        determinism = json.loads((artifacts / "determinism_check.json").read_text(encoding="utf-8"))
        _check(
            checks,
            "fast_determinism_refit",
            bool(determinism["passed"])
            and float(determinism["maximum_probability_difference"]) <= 1e-12
            and bool(determinism["class_order_equal"])
            and bool(determinism["final_decision_equal"]),
        )

        decision_input_error = 0.0
        decision_payload_matches = True
        audited_decision_comparisons: dict[str, dict[str, Any]] = {}
        decision_specifications = {
            "m1_minus_m0": ("m1_versus_m0", "M0", "M1"),
            "m2_minus_m1": ("m2_versus_m1", "M1", "M2"),
        }
        for prefix, (payload_key, baseline_name, candidate_name) in decision_specifications.items():
            comparison = real_comparisons[prefix]
            month_values = {
                str(group): metrics[baseline_name]["multiclass_log_loss"]
                - metrics[candidate_name]["multiclass_log_loss"]
                for group, metrics in grouped_metrics["year_month"].items()
            }
            checkpoint_values = {
                str(group): metrics[baseline_name]["multiclass_log_loss"]
                - metrics[candidate_name]["multiclass_log_loss"]
                for group, metrics in grouped_metrics["decision_ordinal"].items()
            }
            matching_null_metrics = [
                f"{prefix}_log_loss_improvement",
                f"{prefix}_brier_improvement",
            ]
            if prefix == "m2_minus_m1":
                matching_null_metrics.append(f"{prefix}_top_two_improvement")
            null_details = {
                metric: {
                    key: null_summaries[metric][key]
                    for key in ("real_value", "null_q90", "real_percentile")
                }
                for metric in matching_null_metrics
            }
            gates = {
                "log_loss_improves": comparison["log_loss_improvement"] > 0.0,
                "brier_improves": comparison["brier_improvement"] > 0.0,
                "top_two_not_reduced": comparison["top_two_improvement"] >= 0.0,
                "bootstrap_90_lower_log_loss_non_negative": bootstrap_summaries[
                    f"{prefix}_log_loss_improvement"
                ]["interval_90_lower"]
                >= 0.0,
                "bootstrap_90_lower_brier_non_negative": bootstrap_summaries[
                    f"{prefix}_brier_improvement"
                ]["interval_90_lower"]
                >= 0.0,
                "positive_log_loss_months_at_least_five": sum(
                    value > 0.0 for value in month_values.values()
                )
                >= 5,
                "neither_checkpoint_materially_adverse": min(checkpoint_values.values())
                >= CHECKPOINT_MATERIAL_ADVERSITY,
                "real_log_loss_or_brier_exceeds_matching_null_q90": any(
                    null_details[f"{prefix}_{metric}"]["real_value"]
                    > null_details[f"{prefix}_{metric}"]["null_q90"]
                    for metric in ("log_loss_improvement", "brier_improvement")
                ),
                "concentration_gates_pass": support_passed,
            }
            expected_payload = {
                "baseline": baseline_name,
                "candidate": candidate_name,
                "log_loss_improvement": comparison["log_loss_improvement"],
                "brier_improvement": comparison["brier_improvement"],
                "top_two_improvement": comparison["top_two_improvement"],
                "monthly_log_loss_improvements": month_values,
                "positive_log_loss_months": sum(value > 0.0 for value in month_values.values()),
                "checkpoint_log_loss_improvements": checkpoint_values,
                "null": null_details,
                "gates": gates,
                "passes": all(gates.values()),
            }
            audited_decision_comparisons[prefix] = expected_payload
            stored_payload = decision[payload_key]
            decision_payload_matches = decision_payload_matches and (
                stored_payload["baseline"] == baseline_name
                and stored_payload["candidate"] == candidate_name
                and stored_payload["gates"] == gates
                and bool(stored_payload["passes"]) == all(gates.values())
                and int(stored_payload["positive_log_loss_months"])
                == expected_payload["positive_log_loss_months"]
            )
            for key in (
                "log_loss_improvement",
                "brier_improvement",
                "top_two_improvement",
            ):
                decision_input_error = max(
                    decision_input_error,
                    abs(float(stored_payload[key]) - float(expected_payload[key])),
                )
            for key in ("monthly_log_loss_improvements", "checkpoint_log_loss_improvements"):
                stored_values = stored_payload[key]
                expected_values = expected_payload[key]
                decision_payload_matches = decision_payload_matches and (
                    set(stored_values) == set(expected_values)
                )
                for group, expected_value in expected_values.items():
                    if group in stored_values:
                        decision_input_error = max(
                            decision_input_error,
                            abs(float(stored_values[group]) - float(expected_value)),
                        )
            decision_payload_matches = decision_payload_matches and (
                set(stored_payload["null"]) == set(null_details)
            )
            for metric, expected_values in null_details.items():
                if metric not in stored_payload["null"]:
                    continue
                for key, expected_value in expected_values.items():
                    decision_input_error = max(
                        decision_input_error,
                        abs(float(stored_payload["null"][metric][key]) - expected_value),
                    )

        m1_pass = bool(audited_decision_comparisons["m1_minus_m0"]["passes"])
        m2_pass = bool(audited_decision_comparisons["m2_minus_m1"]["passes"])
        descriptive_change = bool(
            audited_pooled["M1"]["prediction_entropy"] < audited_pooled["M0"]["prediction_entropy"]
            or audited_pooled["M2"]["prediction_entropy"]
            < audited_pooled["M1"]["prediction_entropy"]
            or real_comparisons["m1_minus_m0"]["top_two_improvement"] != 0.0
            or real_comparisons["m2_minus_m1"]["top_two_improvement"] != 0.0
        )
        if m2_pass:
            expected_decision = "regime_mix_filters_behaviour_into_coarse_loop_family"
        elif m1_pass:
            expected_decision = "behaviour_main_effects_only"
        elif descriptive_change:
            expected_decision = "descriptive_coarse_funnel_only"
        else:
            expected_decision = "no_behaviour_regime_family_increment"
        _check(
            checks,
            "decision_logic",
            decision_payload_matches
            and decision_input_error <= 1e-12
            and bool(decision["descriptive_funnel_change"]) == descriptive_change
            and decision["decision"] == expected_decision,
            expected=expected_decision,
            actual=decision["decision"],
            maximum_decision_input_error=decision_input_error,
            stored_gate_payloads_match=decision_payload_matches,
        )
    except Exception as error:  # fail closed and retain a useful audit artifact
        _check(checks, "audit_exception", False, error_type=type(error).__name__, detail=str(error))

    passed = bool(checks and all(check["passed"] for check in checks))
    result = {
        **SAFETY_FLAGS,
        "audit": "independent_lightweight_audit_v0",
        "checks": checks,
        "check_count": len(checks),
        "passed": passed,
    }
    (artifacts / "lightweight_audit.json").write_text(canonical_json(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=EXPERIMENT_DIR / "artifacts" / "primary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_artifacts(args.artifacts.expanduser().resolve())
    print(canonical_json(result), end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
