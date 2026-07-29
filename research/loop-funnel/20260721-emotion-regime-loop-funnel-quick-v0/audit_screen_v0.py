#!/usr/bin/env python3
"""Independent lightweight audit for the loop-funnel quick screen."""

# ruff: noqa: E402 -- repository package path is installed before local imports.

from __future__ import annotations

import argparse
import json
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
    "behavioural_regime_gating_test": True,
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
DICTIONARY_PATH = (
    REPO_ROOT
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
    REPO_ROOT
    / "research"
    / "observable-behavioural-state"
    / "20260721-behavioural-state-dimensions-screen-v0"
    / "artifacts"
    / "primary"
)
BEHAVIOURAL_COMPACT = BEHAVIOURAL_PRIMARY / "compact_decision_panel.parquet"
BEHAVIOURAL_DIMENSION_LEDGER = BEHAVIOURAL_PRIMARY / "behavioural_dimension_ledger.parquet"
OPENING_PANEL = (
    REPO_ROOT
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
        return "UNAVAILABLE", None, None
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
    if primary in {"TIED_REGISTERED_COMPLETION", "UNAVAILABLE", "UNREGISTERED_LOOP"}:
        return primary, None, None
    event = outcome.earliest_registered_events[0]
    raw = {
        "primitive": "REGISTERED_PRIMITIVE_COMPLETION",
        "repeat": "REGISTERED_REPEAT_COMPLETION",
        "composite": "REGISTERED_COMPOSITE_COMPLETION",
    }[str(event.motif_type)]
    return raw, event.semantic_loop_id, event.orientation_id


def independent_mapping(panel: pd.DataFrame) -> dict[str, str]:
    development = panel.loc[
        panel["year"].eq(2024)
        & panel["raw_outcome"].astype(str).str.startswith("REGISTERED_")
        & panel["oriented_loop_key"].notna()
    ]
    rows = []
    for key, group in development.groupby("oriented_loop_key", sort=True):
        stock = group.groupby("symbol", sort=True).size()
        rows.append(
            {
                "key": str(key),
                "support": len(group),
                "sessions": group["session"].nunique(),
                "stocks": group["symbol"].nunique(),
                "months": group["year_month"].nunique(),
                "share": float(stock.max() / len(group)),
            }
        )
    support = pd.DataFrame(rows)
    eligible = support.loc[
        support["support"].ge(50)
        & support["sessions"].ge(20)
        & support["stocks"].ge(8)
        & support["months"].ge(4)
        & support["share"].le(0.30)
    ].sort_values(["support", "key"], ascending=[False, True], kind="mergesort")
    return {
        str(row.key): f"LOOP_{index + 1}"
        for index, row in enumerate(eligible.head(6).itertuples(index=False))
    }


def independent_pool(raw: str, key: Any, mapping: dict[str, str]) -> str | None:
    if raw in {"TIED_REGISTERED_COMPLETION", "UNAVAILABLE"}:
        return None
    if raw.startswith("REGISTERED_"):
        return mapping.get(str(key), "OTHER_REGISTERED_LOOP")
    if raw == "UNREGISTERED_LOOP":
        return raw
    if raw in {"NO_REGISTERED_COMPLETION", "SESSION_END", "NO_REGISTERED_LOOP_WITHIN_HORIZON"}:
        return "NO_REGISTERED_COMPLETION"
    raise AssertionError(f"unknown raw target {raw}")


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
    return {
        "multiclass_log_loss": float(np.average(-np.log(realised), weights=weights)),
        "multiclass_brier": float(
            np.average(np.square(probabilities - one_hot).sum(axis=1), weights=weights)
        ),
        "top_one_accuracy": float(np.average(ranks <= 1, weights=weights)),
        "top_three_accuracy": float(np.average(ranks <= 3, weights=weights)),
        "realised_probability": float(np.average(realised, weights=weights)),
        "prediction_entropy": float(
            np.average(
                -np.where(
                    probabilities > 0.0,
                    probabilities * np.log(np.clip(probabilities, np.finfo(float).tiny, 1.0)),
                    0.0,
                ).sum(axis=1),
                weights=weights,
            )
        ),
    }


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
        blocked_support = decision.get("decision") == "blocked_insufficient_loop_class_support"
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
        unavailable = panel["raw_outcome"].eq("UNAVAILABLE")
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
        mapping = independent_mapping(panel)
        _check(
            checks,
            "development_only_loop_selection",
            mapping == mapping_payload["selected_mapping"],
            independently_selected=mapping,
        )
        if blocked_support:
            _check(
                checks,
                "final_target_mapping_and_pooling",
                panel["target_class"].isna().all()
                and (~panel["scoring_eligible"]).all()
                and mapping_payload["final_target_classes"] == [],
                status="not_formed_after_development_vocabulary_gate_failed",
            )
        else:
            independently_pooled = [
                independent_pool(str(raw), key, mapping)
                for raw, key in zip(panel["raw_outcome"], panel["oriented_loop_key"], strict=True)
            ]
            stored_pooled = (
                panel["target_class"].where(panel["target_class"].notna(), None).tolist()
            )
            _check(
                checks,
                "final_target_mapping_and_pooling",
                independently_pooled == stored_pooled,
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
        interaction_manifest = json.loads(
            (artifacts / "interaction_manifest.json").read_text(encoding="utf-8")
        )
        raw = raw_interactions(panel)
        development_mask = panel["year"].eq(2024) & (
            ~panel["target_excluded"] if blocked_support else panel["scoring_eligible"]
        )
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

        if blocked_support:
            configurations = json.loads(
                (artifacts / "model_configurations.json").read_text(encoding="utf-8")
            )
            coefficients = json.loads(
                (artifacts / "model_coefficients.json").read_text(encoding="utf-8")
            )
            determinism = json.loads(
                (artifacts / "determinism_check.json").read_text(encoding="utf-8")
            )
            eligible_count = len(mapping)
            _check(
                checks,
                "support_blocker_decision_logic",
                eligible_count < 4
                and mapping_payload["selected_count"] == eligible_count
                and decision["decision"] == "blocked_insufficient_loop_class_support"
                and configurations["primary_fitted_model_count"] == 0
                and coefficients["models"] == {}
                and predictions.empty
                and determinism["prescribed_model_determinism_applicable"] is False
                and determinism["maximum_probability_difference"] is None
                and bool(determinism["support_gate_reproducibility_passed"])
                and decision["determinism_check_passed"] is None
                and bool(decision["support_gate_reproducibility_passed"]),
                independently_eligible_exact_oriented_loops=eligible_count,
                required=4,
                prescribed_model_determinism="not_applicable",
                support_gate_reproducibility_passed=bool(
                    determinism["support_gate_reproducibility_passed"]
                ),
            )
            passed = bool(checks and all(check["passed"] for check in checks))
            result = {
                **SAFETY_FLAGS,
                "audit": "independent_lightweight_audit_v0",
                "blocked_support_audit": True,
                "checks": checks,
                "check_count": len(checks),
                "passed": passed,
            }
            (artifacts / "lightweight_audit.json").write_text(
                canonical_json(result), encoding="utf-8"
            )
            return result

        coefficient_payload = json.loads(
            (artifacts / "model_coefficients.json").read_text(encoding="utf-8")
        )["models"]
        scoring = panel.loc[panel["scoring_eligible"]].copy()
        development = scoring.loc[scoring["year"].eq(2024)].copy()
        assessment = scoring.loc[scoring["year"].eq(2025)].copy()
        class_order = tuple(coefficient_payload["M0"]["class_order"])
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

        pooled = pd.read_csv(artifacts / "pooled_metrics.csv")
        metric_error = 0.0
        for name in MODEL_FEATURES:
            actual = metric_values(assessment_ordered, stored_probabilities[name], class_order)
            stored = pooled.loc[pooled["model"].eq(name)].iloc[0]
            for metric in (
                "multiclass_log_loss",
                "multiclass_brier",
                "top_one_accuracy",
                "top_three_accuracy",
            ):
                metric_error = max(metric_error, abs(actual[metric] - float(stored[metric])))
        _check(
            checks,
            "primary_multiclass_metrics",
            metric_error <= 1e-12,
            maximum_metric_error=metric_error,
        )

        bootstrap = pd.read_csv(artifacts / "bootstrap_metrics.csv")
        sessions = sorted(assessment_ordered["session"].astype(str).unique())
        by_session = {
            session: np.flatnonzero(
                assessment_ordered["session"].astype(str).eq(session).to_numpy()
            )
            for session in sessions
        }
        generator = np.random.default_rng(20260722)
        sampled_sessions = generator.choice(sessions, size=len(sessions), replace=True)
        bootstrap_indices = np.concatenate([by_session[str(value)] for value in sampled_sessions])
        sampled = assessment_ordered.iloc[bootstrap_indices].reset_index(drop=True)
        base0 = metric_values(sampled, stored_probabilities["M0"][bootstrap_indices], class_order)
        base1 = metric_values(sampled, stored_probabilities["M1"][bootstrap_indices], class_order)
        base2 = metric_values(sampled, stored_probabilities["M2"][bootstrap_indices], class_order)
        first_bootstrap = {
            "m1_minus_m0_log_loss_improvement": base0["multiclass_log_loss"]
            - base1["multiclass_log_loss"],
            "m1_minus_m0_brier_improvement": base0["multiclass_brier"] - base1["multiclass_brier"],
            "m1_minus_m0_top_three_improvement": base1["top_three_accuracy"]
            - base0["top_three_accuracy"],
            "m2_minus_m1_log_loss_improvement": base1["multiclass_log_loss"]
            - base2["multiclass_log_loss"],
            "m2_minus_m1_brier_improvement": base1["multiclass_brier"] - base2["multiclass_brier"],
            "m2_minus_m1_top_three_improvement": base2["top_three_accuracy"]
            - base1["top_three_accuracy"],
            "m2_minus_m1_realised_probability_improvement": base2["realised_probability"]
            - base1["realised_probability"],
            "m2_minus_m1_prediction_entropy_reduction": base1["prediction_entropy"]
            - base2["prediction_entropy"],
        }
        bootstrap_error = 0.0
        for metric, expected in first_bootstrap.items():
            row = bootstrap.loc[bootstrap["metric"].eq(metric)].iloc[0]
            actual = float(json.loads(str(row["draw_values"]))[0])
            bootstrap_error = max(bootstrap_error, abs(actual - expected))
        _check(
            checks,
            "paired_session_block_bootstrap",
            bootstrap_error <= 1e-12
            and bootstrap["draw_count"].eq(50).all()
            and len(bootstrap_indices) > 0,
            first_draw_error=bootstrap_error,
        )

        null = pd.read_csv(artifacts / "null_metrics.csv")
        dev_null = permute_bundle(development, 20260723)
        ass_null = permute_bundle(assessment_ordered, 20260724)
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
            "m2_minus_m1_top_three_improvement": null2["top_three_accuracy"]
            - null1["top_three_accuracy"],
        }
        null_error = 0.0
        for metric, expected in first_null.items():
            row = null.loc[null["metric"].eq(metric)].iloc[0]
            actual = float(json.loads(str(row["draw_values"]))[0])
            null_error = max(null_error, abs(actual - expected))
        _check(
            checks,
            "within_slate_behavioural_null",
            null_error <= 1e-12 and null["draw_count"].eq(10).all(),
            first_draw_error=null_error,
        )

        m1_pass = all(bool(value) for value in decision["m1_versus_m0"]["gates"].values())
        m2_pass = all(bool(value) for value in decision["m2_versus_m1"]["gates"].values())
        if m2_pass:
            expected_decision = "regime_mix_filters_behaviour_into_loop_distribution"
        elif m1_pass:
            expected_decision = "behaviour_main_effects_only"
        elif decision["descriptive_funnel_change"]:
            expected_decision = "descriptive_funnel_only_no_predictive_increment"
        else:
            expected_decision = "no_behaviour_regime_loop_funnel_increment"
        _check(
            checks,
            "decision_logic",
            decision["decision"] == expected_decision,
            expected=expected_decision,
            actual=decision["decision"],
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
