#!/usr/bin/env python3
"""Independent lightweight auditor for Behavioural-Trajectory Funnel V0.1."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.loop_dictionary_v2 import (  # noqa: E402
    LoopDictionary,
    decompose_closed_path,
)
from stocker_research.loop_prefix_automaton_v2 import (  # noqa: E402
    FirstNextLoopEventEngine,
)

DEFAULT_ARTIFACTS = EXPERIMENT_DIR / "artifacts" / "primary"
COARSE_PANEL = (
    REPO_ROOT
    / "research"
    / "loop-funnel"
    / "20260721-emotion-regime-coarse-loop-family-v0"
    / "artifacts"
    / "primary"
    / "decision_panel.parquet"
)
OBSERVABLE_SCALING = (
    REPO_ROOT
    / "research"
    / "observable-behavioural-state"
    / "20260721-behavioural-state-dimensions-screen-v0"
    / "artifacts"
    / "primary"
    / "behavioural_component_scaling.json"
)
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
SAFETY_FLAGS: dict[str, bool | str] = {
    "research_only": True,
    "quick_feasibility_screen": True,
    "corrected_even_trajectory_anchors": True,
    "opening_and_later_session_checkpoints": True,
    "late_no_open_loop_subgroup": True,
    "structural_outcomes_only": True,
    "economic_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
TARGET_CLASSES = (
    "REGISTERED_COMPLETION",
    "UNREGISTERED_LOOP",
    "NO_REGISTERED_COMPLETION",
)
BEHAVIOURS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "signed_exhaustion",
)
ANCHORS = {6: (2, 4, 6), 12: (4, 8, 12), 24: (8, 16, 24), 36: (12, 24, 36)}
TRAJECTORY_FORMS = (
    "change",
    "acceleration",
    "reversal",
    "recent_change",
    "monotonic_persistence",
    "peak_displacement",
)
TRAJECTORY_INTERACTIONS = {
    "transition_probability_x_arousal_change": (
        "transition_probability",
        "arousal_change",
    ),
    "posterior_entropy_x_frustration_change": (
        "posterior_entropy",
        "frustration_change",
    ),
    "top_second_margin_x_conviction_change": (
        "top_second_margin",
        "conviction_change",
    ),
    "transition_probability_x_signed_pressure_acceleration": (
        "transition_probability",
        "signed_pressure_acceleration",
    ),
    "posterior_entropy_x_tension_acceleration": (
        "posterior_entropy",
        "tension_acceleration",
    ),
    "top_state_probability_x_signed_exhaustion_change": (
        "top_state_probability",
        "signed_exhaustion_change",
    ),
}
BASE_INTERACTIONS = {
    **{
        f"state_p_{state}_x_signed_pressure": (f"state_p_{state}", "signed_pressure")
        for state in range(8)
    },
    **{
        f"state_p_{state}_x_signed_exhaustion": (
            f"state_p_{state}",
            "signed_exhaustion",
        )
        for state in range(8)
    },
    "posterior_entropy_x_frustration": ("posterior_entropy", "frustration"),
    "posterior_entropy_x_tension": ("posterior_entropy", "tension"),
    "transition_probability_x_arousal": ("transition_probability", "arousal"),
    "top_second_margin_x_conviction": ("top_second_margin", "conviction"),
}
BOOTSTRAP_SEED = 20260722
NULL_SEED = 20260723


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safety_audit(artifacts: Path) -> dict[str, Any]:
    checked = (
        "contract.json",
        "source_manifest.json",
        "protected_boundary_audit.json",
        "checkpoint_anchor_manifest.json",
        "trajectory_anchor_scaling.json",
        "feature_manifest.json",
        "interaction_manifest.json",
        "model_configurations.json",
        "model_coefficients.json",
        "decision.json",
        "determinism_check.json",
    )
    for filename in checked:
        document = read_json(artifacts / filename)
        for key, expected in SAFETY_FLAGS.items():
            require(document.get(key) == expected, f"{filename} safety flag differs: {key}")
    return {"files_checked": len(checked), "passed": True}


def chronology_and_anchor_audit(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    artifacts: Path,
) -> dict[str, Any]:
    sessions = pd.to_datetime(panel["session"], utc=True, errors="raise")
    require(not bool(sessions.ge(pd.Timestamp("2025-08-23", tz="UTC")).any()), "protected row")
    require(set(panel["decision_ordinal"].astype(int)) == set(ANCHORS), "checkpoint set differs")
    require(
        set(panel.loc[panel["year"].eq(2024), "session"].astype(str).str[:4]) == {"2024"},
        "development chronology differs",
    )
    require(
        set(panel.loc[panel["year"].eq(2025), "session"].astype(str).str[:4]) == {"2025"},
        "assessment chronology differs",
    )
    maximum_future_seconds = float(
        (
            pd.to_datetime(ledger["latest_input_bar_complete_timestamp_utc"], utc=True)
            - pd.to_datetime(ledger["decision_available_timestamp_utc"], utc=True)
        )
        .dt.total_seconds()
        .max()
    )
    require(maximum_future_seconds <= 0.0, "a future bar entered a behavioural anchor")
    actual = {
        int(checkpoint): tuple(
            rows.sort_values("anchor_role")["anchor_ordinal"].astype(int).unique().tolist()
        )
        for checkpoint, rows in ledger.groupby("decision_ordinal", sort=True)
    }
    for checkpoint, expected in ANCHORS.items():
        # Sorting by E0/E1/E2 is lexicographically stable for these labels.
        require(actual[checkpoint] == expected, f"anchor triplet differs for {checkpoint}")
        require(all(value % 2 == 0 for value in actual[checkpoint]), "odd anchor entered")
    protected = read_json(artifacts / "protected_boundary_audit.json")
    require(protected["protected_rows_materialised"] == 0, "protected count differs")
    return {
        "checkpoints": sorted(actual),
        "anchor_triplets": {str(key): list(value) for key, value in actual.items()},
        "maximum_future_input_seconds": maximum_future_seconds,
        "protected_rows_materialised": 0,
        "passed": True,
    }


def scaling_and_dimension_audit(
    ledger: pd.DataFrame,
    artifacts: Path,
) -> dict[str, Any]:
    scaling = read_json(artifacts / "trajectory_anchor_scaling.json")
    maximum_parameter_difference = 0.0
    maximum_z_difference = 0.0
    for family in ("base_components", "pressure_aligned_components"):
        for key, components in scaling[family].items():
            _, checkpoint, _, anchor = key.split("_")
            mask = ledger["decision_ordinal"].eq(int(checkpoint)) & ledger["anchor_ordinal"].eq(
                int(anchor)
            )
            development = ledger.loc[mask & ledger["year"].eq(2024)]
            all_rows = ledger.loc[mask]
            for component, parameters in components.items():
                values = development[component].astype(float)
                center = float(values.median())
                lower = float(values.quantile(0.25, interpolation="linear"))
                upper = float(values.quantile(0.75, interpolation="linear"))
                scale = upper - lower
                if not np.isfinite(scale) or scale < 1e-12:
                    scale = 1.0
                maximum_parameter_difference = max(
                    maximum_parameter_difference,
                    abs(center - float(parameters["center"])),
                    abs(scale - float(parameters["scale"])),
                )
                reconstructed = np.clip(
                    (all_rows[component].to_numpy(dtype=float) - center) / scale,
                    float(parameters["clip_lower"]),
                    float(parameters["clip_upper"]),
                )
                maximum_z_difference = max(
                    maximum_z_difference,
                    float(
                        np.max(
                            np.abs(reconstructed - all_rows[f"z_{component}"].to_numpy(dtype=float))
                        )
                    ),
                )
    reconstructed_dimensions = pd.DataFrame(index=ledger.index)
    reconstructed_dimensions["arousal"] = ledger[
        ["z_activity_effort", "z_range_effort", "z_travel_effort"]
    ].mean(axis=1)
    reconstructed_dimensions["conviction"] = ledger[
        ["z_absolute_efficiency", "z_close_retention", "z_directional_persistence"]
    ].mean(axis=1)
    reconstructed_dimensions["frustration"] = ledger[
        ["z_activity_effort", "z_travel_effort", "z_extreme_rejection"]
    ].mean(axis=1) - ledger[["z_absolute_progress", "z_absolute_efficiency"]].mean(axis=1)
    reconstructed_dimensions["tension"] = (
        ledger[["z_activity_effort", "z_compression", "z_extreme_rejection"]].mean(axis=1)
        - ledger["z_absolute_progress"]
    )
    reconstructed_dimensions["signed_pressure"] = ledger[
        ["z_signed_progress", "z_signed_efficiency", "z_mean_close_location", "z_boundary_slope"]
    ].mean(axis=1)
    exhaustion_magnitude = (
        ledger["z_effort_acceleration"]
        - ledger["z_aligned_progress_acceleration"]
        + ledger["z_directional_rejection"]
    )
    reconstructed_dimensions["signed_exhaustion"] = (
        np.sign(reconstructed_dimensions["signed_pressure"]) * exhaustion_magnitude
    )
    maximum_dimension_difference = float(
        np.max(
            np.abs(
                reconstructed_dimensions.loc[:, list(BEHAVIOURS)].to_numpy(dtype=float)
                - ledger.loc[:, list(BEHAVIOURS)].to_numpy(dtype=float)
            )
        )
    )
    coarse = pd.read_parquet(
        COARSE_PANEL,
        columns=["symbol", "session", "decision_ordinal", *BEHAVIOURS],
    )
    final = ledger.loc[
        ledger["anchor_role"].eq("E2") & ledger["decision_ordinal"].isin((6, 12)),
        ["symbol", "session", "decision_ordinal", *BEHAVIOURS],
    ]
    comparison = final.merge(
        coarse,
        on=["symbol", "session", "decision_ordinal"],
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )
    maximum_frozen_level_difference = float(
        max(np.max(np.abs(comparison[name] - comparison[f"{name}_frozen"])) for name in BEHAVIOURS)
    )
    predecessor_scaling = read_json(OBSERVABLE_SCALING)
    frozen_parameter_differences: list[float] = []
    for checkpoint in (6, 12):
        key = f"checkpoint_{checkpoint}_anchor_{checkpoint}"
        for family in ("base_components", "pressure_aligned_components"):
            for component, expected in predecessor_scaling[family][str(checkpoint)].items():
                actual = scaling[family][key][component]
                for field in ("center", "scale", "clip_lower", "clip_upper"):
                    frozen_parameter_differences.append(
                        abs(float(actual[field]) - float(expected[field]))
                    )
    maximum_frozen_scaling_difference = float(max(frozen_parameter_differences, default=0.0))
    require(maximum_parameter_difference <= 1e-12, "development scaling differs")
    require(maximum_z_difference <= 1e-12, "scaled component differs")
    require(maximum_dimension_difference <= 1e-12, "behavioural dimension differs")
    require(maximum_frozen_level_difference <= 1e-12, "frozen final level differs")
    require(maximum_frozen_scaling_difference <= 1e-12, "frozen final scaling differs")
    return {
        "maximum_scaling_parameter_difference": maximum_parameter_difference,
        "maximum_scaled_component_difference": maximum_z_difference,
        "maximum_behavioural_dimension_difference": maximum_dimension_difference,
        "frozen_final_rows_compared": len(comparison),
        "maximum_frozen_final_level_difference": maximum_frozen_level_difference,
        "maximum_frozen_final_scaling_difference": maximum_frozen_scaling_difference,
        "new_final_scaling_groups_present": all(
            key in scaling["base_components"]
            for key in ("checkpoint_24_anchor_24", "checkpoint_36_anchor_36")
        ),
        "passed": True,
    }


def trajectory_and_interaction_audit(
    panel: pd.DataFrame,
    ledger: pd.DataFrame,
    artifacts: Path,
) -> dict[str, Any]:
    maximum_trajectory_difference = 0.0
    indexed_panel = panel.set_index(["symbol", "session", "decision_ordinal"])
    for role in ("E0", "E1", "E2"):
        anchor = ledger.loc[ledger["anchor_role"].eq(role)].set_index(
            ["symbol", "session", "decision_ordinal"]
        )
        anchor = anchor.loc[anchor.index.isin(indexed_panel.index)]
        for behaviour in BEHAVIOURS:
            maximum_trajectory_difference = max(
                maximum_trajectory_difference,
                float(
                    np.max(
                        np.abs(
                            anchor[behaviour].to_numpy(dtype=float)
                            - indexed_panel.loc[anchor.index, f"{behaviour}_{role}"].to_numpy(
                                dtype=float
                            )
                        )
                    )
                ),
            )
    for behaviour in BEHAVIOURS:
        e0 = panel[f"{behaviour}_E0"].to_numpy(dtype=float)
        e1 = panel[f"{behaviour}_E1"].to_numpy(dtype=float)
        e2 = panel[f"{behaviour}_E2"].to_numpy(dtype=float)
        first = e1 - e0
        recent = e2 - e1
        calculated: dict[str, np.ndarray] = {
            "change": e2 - e0,
            "acceleration": recent - first,
            "reversal": (
                (first != 0.0) & (recent != 0.0) & ((first > 0.0) != (recent > 0.0))
            ).astype(float),
            "recent_change": recent,
            "monotonic_persistence": np.where(
                (e0 < e1) & (e1 < e2),
                1.0,
                np.where((e0 > e1) & (e1 > e2), -1.0, 0.0),
            ),
            "peak_displacement": e2 - np.maximum.reduce([e0, e1, e2]),
        }
        for form in TRAJECTORY_FORMS:
            maximum_trajectory_difference = max(
                maximum_trajectory_difference,
                float(
                    np.max(
                        np.abs(
                            calculated[form] - panel[f"{behaviour}_{form}"].to_numpy(dtype=float)
                        )
                    )
                ),
            )
    manifest = read_json(artifacts / "interaction_manifest.json")
    maximum_interaction_difference = 0.0
    for family, specifications in (
        ("behavioural_level_regime_interactions", BASE_INTERACTIONS),
        ("trajectory_regime_interactions", TRAJECTORY_INTERACTIONS),
    ):
        for feature, (left, right) in specifications.items():
            bounds = manifest[family][feature]
            calculated = (panel[left].astype(float) * panel[right].astype(float)).clip(
                lower=float(bounds["q01"]), upper=float(bounds["q99"])
            )
            maximum_interaction_difference = max(
                maximum_interaction_difference,
                float(np.max(np.abs(calculated - panel[feature].astype(float)))),
            )
    require(maximum_trajectory_difference <= 1e-12, "trajectory feature differs")
    require(maximum_interaction_difference <= 1e-12, "interaction differs")
    return {
        "trajectory_features_checked": len(BEHAVIOURS) * len(TRAJECTORY_FORMS),
        "interactions_checked": len(BASE_INTERACTIONS) + len(TRAJECTORY_INTERACTIONS),
        "maximum_trajectory_difference": maximum_trajectory_difference,
        "maximum_interaction_difference": maximum_interaction_difference,
        "passed": True,
    }


def load_dictionary() -> LoopDictionary:
    table = pd.read_csv(DICTIONARY_PATH)
    definitions = {}
    for row in table.itertuples(index=False):
        definition = decompose_closed_path(json.loads(str(row.canonical_orientation)))
        definitions[definition.semantic_loop_id] = definition
    return LoopDictionary(definitions, (), version=str(table["dictionary_version"].iloc[0]))


def map_outcome(outcome: Any) -> tuple[str, str | None]:
    primary = str(outcome.primary_label)
    if primary == "TIED_REGISTERED_COMPLETION":
        return primary, None
    if primary == "UNAVAILABLE":
        return "SOURCE_UNAVAILABLE", None
    if primary == "UNREGISTERED_LOOP":
        return primary, primary
    if primary in {"SESSION_END", "NO_REGISTERED_LOOP_WITHIN_HORIZON"}:
        return "NO_REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"
    require(bool(outcome.earliest_registered_events), "registered outcome lacks event")
    motif = str(outcome.earliest_registered_events[0].motif_type)
    raw = {
        "primitive": "REGISTERED_PRIMITIVE",
        "repeat": "REGISTERED_REPEAT",
        "composite": "REGISTERED_COMPOSITE",
    }[motif]
    return raw, "REGISTERED_COMPLETION"


def structural_audit(panel: pd.DataFrame) -> dict[str, Any]:
    dictionary = load_dictionary()
    engine = FirstNextLoopEventEngine(dictionary, allowed_states=frozenset(range(8)))
    mismatches = 0
    checked = 0
    for (symbol, session), rows in panel.groupby(["symbol", "session"], sort=True):
        latest_checkpoint = int(rows["decision_ordinal"].max())
        latest = rows.loc[rows["decision_ordinal"].eq(latest_checkpoint)]
        require(len(latest) == 1, "one latest path per stock/session is required")
        state_path = [int(value) for value in latest.iloc[0]["state_path_through_horizon"]]
        require(
            len(state_path) == latest_checkpoint + 6,
            "latest decision horizon has an unexpected completed-bar count",
        )
        starts = pd.date_range(
            pd.Timestamp(f"{session} 09:30", tz="America/New_York"),
            periods=len(state_path),
            freq="5min",
        )
        event_mask = np.concatenate(([True], np.asarray(state_path[1:]) != state_path[:-1]))
        event_indices = np.flatnonzero(event_mask)
        trace = engine.scan_state_events(
            [state_path[index] for index in event_indices],
            bar_ordinals=event_indices.tolist(),
            event_timestamps=[starts[index].to_pydatetime() for index in event_indices],
            available_timestamps=[
                (starts[index] + pd.Timedelta(minutes=5)).to_pydatetime() for index in event_indices
            ],
        )
        event_ordinals = np.asarray([event.bar_ordinal for event in trace.state_events], dtype=int)
        completion_ordinals = [
            int(event.completion_bar_ordinal) for event in trace.registered_completions
        ]
        opening_count = sum(value <= 11 for value in completion_ordinals)
        for row in rows.itertuples(index=False):
            checkpoint = int(row.decision_ordinal)
            origin = checkpoint - 1
            event_index = int(np.flatnonzero(event_ordinals <= origin)[-1])
            available = pd.Timestamp(row.feature_available_timestamp_utc).to_pydatetime()
            outcome = engine.outcome_for_decision(
                trace,
                decision_id=f"{symbol}|{session}|{checkpoint:02d}",
                decision_event_index=event_index,
                decision_bar_ordinal=origin,
                decision_timestamp=available,
                decision_available_timestamp=available,
                horizon_bars=6,
                session_end_bar_ordinal=77,
                source_available=True,
                symbol=str(symbol),
                session=str(session),
            )
            raw, target = map_outcome(outcome)
            known = sorted(value for value in completion_ordinals if value <= origin)
            expected_controls = {
                "registered_completion_count_before_decision": len(known),
                "bars_since_last_registered_completion": float(origin - known[-1])
                if known
                else 0.0,
                "bars_since_last_registered_completion_missing": int(not known),
                "active_registered_prefix_count_at_decision": len(
                    trace.prefixes_after_event[event_index]
                ),
            }
            expected_phase = "OPENING_PHASE" if checkpoint in (6, 12) else "LATER_PHASE"
            expected_subgroup = (
                None
                if checkpoint in (6, 12)
                else (
                    "LATE_NO_OPEN_REGISTERED_LOOP"
                    if opening_count == 0
                    else "LATE_AFTER_OPEN_REGISTERED_LOOP"
                )
            )
            actual_target = None if pd.isna(row.target_class) else str(row.target_class)
            conditions = [
                raw == str(row.raw_outcome),
                target == actual_target,
                opening_count == int(row.opening_registered_completion_count_through_ordinal_12),
                expected_phase == str(row.phase),
                expected_subgroup
                == (None if pd.isna(row.late_loop_subgroup) else str(row.late_loop_subgroup)),
                *(
                    abs(float(getattr(row, key)) - float(value)) <= 1e-12
                    for key, value in expected_controls.items()
                ),
            ]
            mismatches += int(not all(conditions))
            checked += 1
    require(mismatches == 0, f"structural target/control mismatches: {mismatches}")
    return {"rows_checked": checked, "mismatches": mismatches, "passed": True}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def manual_model_probabilities(frame: pd.DataFrame, model: Mapping[str, Any]) -> np.ndarray:
    features = [str(value) for value in model["features"]]
    matrix = frame.loc[:, features].to_numpy(dtype=float)
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    coefficients = np.asarray(model["coefficient"], dtype=float)
    intercept = np.asarray(model["intercept"], dtype=float)
    return softmax(((matrix - mean) / scale) @ coefficients.T + intercept)


def metric_values(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    class_index = {label: index for index, label in enumerate(TARGET_CLASSES)}
    targets = frame["target_class"].map(class_index).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranks = np.asarray(
        [int(np.flatnonzero(order[index] == target)[0]) + 1 for index, target in enumerate(targets)]
    )
    realised = probabilities[np.arange(len(targets)), targets]
    entropy = -np.sum(
        np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0), axis=1
    )
    mean_entropy = float(np.average(entropy, weights=weights))
    return {
        "multiclass_log_loss": float(
            log_loss(targets, probabilities, labels=np.arange(3), sample_weight=weights)
        ),
        "multiclass_brier": float(
            np.average(np.sum((probabilities - np.eye(3)[targets]) ** 2, axis=1), weights=weights)
        ),
        "top_one_accuracy": float(np.average(ranks <= 1, weights=weights)),
        "top_two_accuracy": float(np.average(ranks <= 2, weights=weights)),
        "mean_reciprocal_rank": float(np.average(1.0 / ranks, weights=weights)),
        "mean_probability_realised_class": float(np.average(realised, weights=weights)),
        "prediction_entropy": mean_entropy,
        "effective_candidate_count": math.exp(mean_entropy),
    }


def model_and_metric_audit(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    artifacts: Path,
) -> dict[str, Any]:
    configuration = read_json(artifacts / "model_configurations.json")
    feature_document = read_json(artifacts / "feature_manifest.json")
    coefficients = read_json(artifacts / "model_coefficients.json")["primary_models"]
    pooled = pd.read_csv(artifacts / "pooled_metrics.csv").set_index("model")
    development = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2024)]
    assessment = panel.loc[panel["scoring_eligible"] & panel["year"].eq(2025)]
    maximum_probability_difference = 0.0
    maximum_metric_difference = 0.0
    maximum_preprocessing_difference = 0.0
    reporting_median_difference = max(
        abs(
            float(feature_document["development_frozen_reporting_medians"]["posterior_entropy"])
            - float(development["posterior_entropy"].median())
        ),
        abs(
            float(
                feature_document["development_frozen_reporting_medians"]["transition_probability"]
            )
            - float(development["transition_probability"].median())
        ),
    )
    for name in ("T0", "T1", "T2"):
        model = coefficients[name]
        require(model["features"] == configuration["models"][name]["features"], "feature list")
        require(max(model["n_iter"]) < 300, "model did not converge")
        development_matrix = development.loc[:, model["features"]].to_numpy(dtype=float)
        expected_mean = development_matrix.mean(axis=0)
        expected_scale = development_matrix.std(axis=0, ddof=0)
        expected_scale[expected_scale == 0.0] = 1.0
        maximum_preprocessing_difference = max(
            maximum_preprocessing_difference,
            float(np.max(np.abs(expected_mean - np.asarray(model["scaler_mean"], dtype=float)))),
            float(np.max(np.abs(expected_scale - np.asarray(model["scaler_scale"], dtype=float)))),
        )
        calculated = manual_model_probabilities(assessment, model)
        stored = predictions[[f"{name}_probability_{label}" for label in TARGET_CLASSES]].to_numpy(
            dtype=float
        )
        maximum_probability_difference = max(
            maximum_probability_difference, float(np.max(np.abs(calculated - stored)))
        )
        metrics = metric_values(assessment, calculated)
        for metric, value in metrics.items():
            maximum_metric_difference = max(
                maximum_metric_difference, abs(value - float(pooled.loc[name, metric]))
            )
    require(len(assessment) >= 100, "manual probability sample is below 100")
    require(
        configuration["configuration"]["preprocessing_fit"] == "2024_only"
        and configuration["configuration"]["coefficients_fit"] == "2024_only",
        "model fit chronology differs",
    )
    require(maximum_preprocessing_difference <= 1e-12, "development preprocessing differs")
    require(reporting_median_difference <= 1e-12, "development reporting median differs")
    require(maximum_probability_difference <= 1e-12, "manual probabilities differ")
    require(maximum_metric_difference <= 1e-12, "pooled metric differs")
    return {
        "manual_probability_rows": len(assessment),
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "maximum_development_preprocessing_difference": maximum_preprocessing_difference,
        "maximum_development_reporting_median_difference": reporting_median_difference,
        "coefficients_and_class_order_checked": True,
        "passed": True,
    }


def comparison(frame: pd.DataFrame, baseline: str, candidate: str) -> dict[str, float]:
    before = metric_values(
        frame,
        frame[[f"{baseline}_probability_{label}" for label in TARGET_CLASSES]].to_numpy(
            dtype=float
        ),
    )
    after = metric_values(
        frame,
        frame[[f"{candidate}_probability_{label}" for label in TARGET_CLASSES]].to_numpy(
            dtype=float
        ),
    )
    return {
        "log_loss_improvement": before["multiclass_log_loss"] - after["multiclass_log_loss"],
        "brier_improvement": before["multiclass_brier"] - after["multiclass_brier"],
        "top_two_change": after["top_two_accuracy"] - before["top_two_accuracy"],
    }


def masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "pooled": pd.Series(True, index=frame.index),
        "opening": frame["phase"].eq("OPENING_PHASE"),
        "later": frame["phase"].eq("LATER_PHASE"),
        "late_no_open": frame["late_loop_subgroup"].eq("LATE_NO_OPEN_REGISTERED_LOOP"),
    }


def bootstrap_audit(predictions: pd.DataFrame, artifacts: Path) -> dict[str, Any]:
    complete = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    stored = complete.loc[complete["record_type"].eq("draw")]
    base = predictions.reset_index(drop=True)
    sessions = tuple(sorted(base["session"].astype(str).unique().tolist()))
    positions = {
        session: np.flatnonzero(base["session"].astype(str).to_numpy() == session)
        for session in sessions
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    maximum_difference = 0.0
    checked = 0
    draw_values: dict[tuple[str, str, str], list[float]] = {}
    for draw in range(25):
        sampled = tuple(str(value) for value in rng.choice(sessions, len(sessions), replace=True))
        sample = base.iloc[np.concatenate([positions[value] for value in sampled])].reset_index(
            drop=True
        )
        for population, mask in masks(sample).items():
            subset = sample.loc[mask]
            for baseline, candidate in (("T0", "T1"), ("T1", "T2")):
                expected = stored.loc[
                    stored["draw"].eq(draw)
                    & stored["population"].eq(population)
                    & stored["comparison"].eq(f"{candidate}_minus_{baseline}")
                ].iloc[0]
                values = comparison(subset, baseline, candidate)
                for metric, value in values.items():
                    maximum_difference = max(
                        maximum_difference, abs(value - float(expected[metric]))
                    )
                    draw_values.setdefault(
                        (population, f"{candidate}_minus_{baseline}", metric), []
                    ).append(value)
                checked += 1
    intervals_checked = 0
    for row in complete.loc[complete["record_type"].eq("interval")].itertuples(index=False):
        values = np.asarray(
            draw_values[(str(row.population), str(row.comparison), str(row.metric))], dtype=float
        )
        tail = (1.0 - float(row.interval_level)) / 2.0
        lower = float(np.quantile(values, tail, method="linear"))
        upper = float(np.quantile(values, 1.0 - tail, method="linear"))
        maximum_difference = max(
            maximum_difference, abs(lower - float(row.lower)), abs(upper - float(row.upper))
        )
        intervals_checked += 1
    require(maximum_difference <= 1e-12, "session bootstrap differs")
    return {
        "draws": 25,
        "comparison_rows_checked": checked,
        "interval_rows_checked": intervals_checked,
        "maximum_difference": maximum_difference,
        "paired_all_checkpoints_and_stocks": True,
        "passed": True,
    }


def permute_bundle(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    seed: int,
) -> pd.DataFrame:
    result = frame.copy()
    source = frame.loc[:, list(features)].to_numpy(copy=True)
    columns = [int(result.columns.get_loc(feature)) for feature in features]
    rng = np.random.default_rng(seed)
    for positions in frame.groupby("slate_id", sort=True, observed=True).indices.values():
        target = np.asarray(positions, dtype=int)
        selected = target[rng.permutation(len(target))]
        result.iloc[target, columns] = source[selected]
    return result


def null_audit(panel: pd.DataFrame, predictions: pd.DataFrame, artifacts: Path) -> dict[str, Any]:
    complete = pd.read_csv(artifacts / "null_metrics.csv")
    stored = complete.loc[complete["record_type"].eq("draw")]
    model_document = read_json(artifacts / "model_coefficients.json")
    null_models = {int(row["draw"]): row for row in model_document["null_models"]}
    manifest = read_json(artifacts / "interaction_manifest.json")
    features = [f"{behaviour}_{form}" for behaviour in BEHAVIOURS for form in TRAJECTORY_FORMS]
    scoring = panel.loc[panel["scoring_eligible"]].copy().reset_index(drop=True)
    maximum_difference = 0.0
    checked = 0
    draw_values: dict[tuple[str, str, str], list[float]] = {}
    for draw in range(5):
        scoring["_row_order"] = np.arange(len(scoring))
        parts = []
        for year in (2024, 2025):
            part = scoring.loc[scoring["year"].eq(year)].copy()
            parts.append(
                permute_bundle(
                    part,
                    features,
                    seed=NULL_SEED + draw * 10 + (year - 2024),
                )
            )
        permuted = pd.concat(parts, ignore_index=True).sort_values("_row_order", kind="mergesort")
        for feature, (left, right) in TRAJECTORY_INTERACTIONS.items():
            bounds = manifest["trajectory_regime_interactions"][feature]
            permuted[feature] = (permuted[left] * permuted[right]).clip(
                lower=float(bounds["q01"]), upper=float(bounds["q99"])
            )
        assessment = permuted.loc[permuted["year"].eq(2025)].copy()
        key_columns = ["symbol", "session", "decision_ordinal"]
        null_predictions = assessment.loc[
            :,
            [
                *key_columns,
                "phase",
                "late_loop_subgroup",
                "target_class",
                "row_weight",
            ],
        ].merge(
            predictions[
                [
                    *key_columns,
                    *[f"T0_probability_{label}" for label in TARGET_CLASSES],
                ]
            ],
            on=key_columns,
            validate="one_to_one",
        )
        for name in ("T1", "T2"):
            calculated = manual_model_probabilities(assessment, null_models[draw][name])
            for index, target in enumerate(TARGET_CLASSES):
                null_predictions[f"{name}_probability_{target}"] = calculated[:, index]
        for population, mask in masks(null_predictions).items():
            subset = null_predictions.loc[mask]
            for baseline, candidate in (("T0", "T1"), ("T1", "T2")):
                expected = stored.loc[
                    stored["draw"].eq(draw)
                    & stored["population"].eq(population)
                    & stored["comparison"].eq(f"{candidate}_minus_{baseline}")
                ].iloc[0]
                values = comparison(subset, baseline, candidate)
                for metric, value in values.items():
                    maximum_difference = max(
                        maximum_difference, abs(value - float(expected[metric]))
                    )
                    draw_values.setdefault(
                        (population, f"{candidate}_minus_{baseline}", metric), []
                    ).append(value)
                checked += 1
    summary_mismatches = 0
    summaries_checked = 0
    for row in complete.loc[complete["record_type"].eq("comparison")].itertuples(index=False):
        population = str(row.population)
        comparison_name = str(row.comparison)
        baseline, candidate = ("T0", "T1") if comparison_name == "T1_minus_T0" else ("T1", "T2")
        real_subset = predictions.loc[masks(predictions)[population]]
        real_value = comparison(real_subset, baseline, candidate)[str(row.metric)]
        values = np.asarray(
            draw_values[(population, comparison_name, str(row.metric))], dtype=float
        )
        count = int((real_value > values).sum())
        conditions = (
            abs(real_value - float(row.real_increment)) <= 1e-12,
            count == int(row.null_draws_exceeded),
            bool(row.exceeds_at_least_0_of_5) == (count >= 0),
            bool(row.exceeds_at_least_3_of_5) == (count >= 3),
            bool(row.exceeds_at_least_4_of_5) == (count >= 4),
            bool(row.exceeds_all_5) == (count == 5),
        )
        summary_mismatches += int(not all(conditions))
        summaries_checked += 1
    require(maximum_difference <= 1e-12, "trajectory-bundle null differs")
    require(summary_mismatches == 0, "trajectory-bundle null summary differs")
    return {
        "draws": 5,
        "comparison_rows_checked": checked,
        "summary_rows_checked": summaries_checked,
        "summary_mismatches": summary_mismatches,
        "maximum_difference": maximum_difference,
        "complete_bundle_permuted_within_slate": True,
        "passed": True,
    }


def decision_audit(artifacts: Path) -> dict[str, Any]:
    decision = read_json(artifacts / "decision.json")
    predictions = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    bootstrap = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    null = pd.read_csv(artifacts / "null_metrics.csv")
    monthly = pd.read_csv(artifacts / "monthly_metrics.csv")
    concentration = pd.read_csv(artifacts / "concentration_metrics.csv")
    gate_mismatches = 0
    for gate in decision["gates"]:
        population = str(gate["population"])
        comparison_name = str(gate["comparison"])
        baseline, candidate = ("T0", "T1") if comparison_name == "T1_minus_T0" else ("T1", "T2")
        subset = predictions.loc[masks(predictions)[population]]
        increment = comparison(subset, baseline, candidate)
        interval = bootstrap.loc[
            bootstrap["record_type"].eq("interval")
            & bootstrap["population"].eq(population)
            & bootstrap["comparison"].eq(comparison_name)
            & bootstrap["interval_level"].eq(0.8)
        ].set_index("metric")
        null_rows = null.loc[
            null["record_type"].eq("comparison")
            & null["population"].eq(population)
            & null["comparison"].eq(comparison_name)
        ].set_index("metric")
        monthly_rows = monthly.loc[
            monthly["group_value"].astype(str).str.startswith(f"{population}|")
        ]
        pivot = monthly_rows.pivot(
            index="group_value", columns="model", values="multiclass_log_loss"
        )
        positive_months = int((pivot[baseline] - pivot[candidate] > 0.0).sum())
        month_threshold = 5 if population in {"pooled", "opening"} else 4
        concentration_pass = bool(
            concentration.loc[concentration["population"].eq(population), "passed"].all()
        )
        expected_conditions = {
            "supported": bool(decision["support"]["screen_supported_populations"][population]),
            "log_loss_improves": increment["log_loss_improvement"] > 0.0,
            "brier_improves": increment["brier_improvement"] > 0.0,
            "bootstrap_80_log_loss_lower_non_negative": float(
                interval.loc["log_loss_improvement", "lower"]
            )
            >= 0.0,
            "bootstrap_80_brier_lower_non_negative": float(
                interval.loc["brier_improvement", "lower"]
            )
            >= 0.0,
            "top_two_decline_within_0_002": increment["top_two_change"] >= -0.002,
            "positive_month_count": positive_months >= month_threshold,
            "real_log_loss_or_brier_exceeds_four_of_five_nulls": (
                int(null_rows.loc["log_loss_improvement", "null_draws_exceeded"]) >= 4
                or int(null_rows.loc["brier_improvement", "null_draws_exceeded"]) >= 4
            ),
            "concentration_gates_pass": concentration_pass,
        }
        gate_mismatches += int(
            expected_conditions != gate["conditions"]
            or bool(gate["rough_screen_positive"]) != all(expected_conditions.values())
            or positive_months != int(gate["positive_log_loss_months"])
        )
    t1 = {str(key): bool(value) for key, value in decision["T1_screen_positive"].items()}
    t2 = {str(key): bool(value) for key, value in decision["T2_screen_positive"].items()}
    if any(t1.values()) and not any(t2.values()):
        expected = "trajectory_main_effects_only"
    elif any(t2.values()):
        combined = {key: t1[key] or t2[key] for key in t1}
        if combined["pooled"] and (combined["later"] or combined["late_no_open"]):
            expected = "trajectory_signal_feasible_pooled_and_late"
        elif combined["pooled"]:
            expected = "trajectory_signal_feasible_pooled_only"
        elif combined["later"] or combined["late_no_open"]:
            expected = "trajectory_signal_feasible_late_only"
        else:
            expected = "trajectory_signal_feasible_opening_only"
    elif decision["point_estimate_improves_somewhere"]:
        expected = "descriptive_trajectory_structure_only"
    else:
        expected = "no_behavioural_trajectory_increment"
    require(expected == decision["decision"], "decision precedence differs")
    require(gate_mismatches == 0, "independently reconstructed decision gate differs")
    for gate in decision["gates"]:
        require(
            bool(gate["rough_screen_positive"]) == all(gate["conditions"].values()),
            "stored gate conjunction differs",
        )
    return {
        "expected_decision": expected,
        "stored_decision": decision["decision"],
        "gates_checked": len(decision["gates"]),
        "gate_mismatches": gate_mismatches,
        "passed": True,
    }


def audit(artifacts: Path) -> dict[str, Any]:
    required = (
        "contract.json",
        "source_manifest.json",
        "protected_boundary_audit.json",
        "checkpoint_anchor_manifest.json",
        "trajectory_anchor_scaling.json",
        "trajectory_missingness.csv",
        "feature_manifest.json",
        "interaction_manifest.json",
        "decision_panel.parquet",
        "trajectory_ledger.parquet",
        "model_configurations.json",
        "model_coefficients.json",
        "assessment_predictions.parquet",
        "pooled_metrics.csv",
        "phase_metrics.csv",
        "checkpoint_metrics.csv",
        "monthly_metrics.csv",
        "class_metrics.csv",
        "late_loop_subgroup_metrics.csv",
        "trajectory_diagnostics.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
        "concentration_metrics.csv",
        "decision.json",
        "determinism_check.json",
    )
    missing = [filename for filename in required if not (artifacts / filename).is_file()]
    require(not missing, f"required artifacts missing: {missing}")
    panel = pd.read_parquet(artifacts / "decision_panel.parquet")
    ledger = pd.read_parquet(artifacts / "trajectory_ledger.parquet")
    predictions = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    checks = {
        "safety": safety_audit(artifacts),
        "chronology_and_anchors": chronology_and_anchor_audit(panel, ledger, artifacts),
        "scaling_and_dimensions": scaling_and_dimension_audit(ledger, artifacts),
        "trajectories_and_interactions": trajectory_and_interaction_audit(panel, ledger, artifacts),
        "structural_target_and_controls": structural_audit(panel),
        "models_and_metrics": model_and_metric_audit(panel, predictions, artifacts),
        "session_block_bootstrap": bootstrap_audit(predictions, artifacts),
        "trajectory_bundle_null": null_audit(panel, predictions, artifacts),
        "decision_logic": decision_audit(artifacts),
    }
    determinism = read_json(artifacts / "determinism_check.json")
    require(bool(determinism["passed"]), "determinism artifact failed")
    result = {
        **SAFETY_FLAGS,
        "passed": all(bool(value["passed"]) for value in checks.values()),
        "independent_of_new_reusable_module": True,
        "checks": checks,
        "determinism_check_passed": True,
        "fail_closed": True,
    }
    (artifacts / "lightweight_audit.json").write_text(canonical_json(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def main() -> int:
    try:
        result = audit(parse_args().artifacts.expanduser().resolve())
        print(canonical_json(result), end="")
        return 0
    except Exception as error:
        print(f"blocked_reproducibility_or_audit_failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
