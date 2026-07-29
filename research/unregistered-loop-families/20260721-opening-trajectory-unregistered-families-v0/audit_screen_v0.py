#!/usr/bin/env python3
"""Independent lightweight audit for the opening unregistered-family screen."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_source))

from stocker_research.loop_dictionary_v2 import (  # noqa: E402
    UnsupportedLoopError,
    decompose_closed_path,
)

DEFAULT_ARTIFACTS = EXPERIMENT_DIR / "artifacts" / "primary"
PREDECESSOR = (
    REPO_ROOT
    / "research"
    / "behavioural-trajectory"
    / "20260721-behavioural-trajectory-late-loops-v01"
    / "artifacts"
    / "primary"
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
    "opening_phase_only": True,
    "unregistered_structural_event_target": True,
    "development_frozen_unregistered_families": True,
    "economic_outcomes_opened": False,
    "directional_outcomes_opened": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}
KEYS = ("symbol", "session", "decision_ordinal")
BEHAVIOURS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "signed_exhaustion",
)
TRAJECTORY_FEATURES = tuple(
    f"{behaviour}_{form}"
    for behaviour in BEHAVIOURS
    for form in ("change", "acceleration", "reversal")
)
RAW_OUTCOME_CLASSES = (
    "REGISTERED_COMPLETION",
    "UNREGISTERED_LOOP",
    "NO_REGISTERED_COMPLETION",
    "TIED_REGISTERED_COMPLETION",
    "SOURCE_UNAVAILABLE",
)
BOOTSTRAP_SEED = 20260724
NULL_SEED = 20260725
OTHER_FAMILY = "OTHER_UNREGISTERED_FAMILY"
ATTRIBUTION_SEED = 20260726
TRAJECTORY_GROUPS = {
    f"{behaviour.upper()}_TRAJECTORY": tuple(
        f"{behaviour}_{form}" for form in ("change", "acceleration", "reversal")
    )
    for behaviour in BEHAVIOURS
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def binary_brier(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average((probabilities - labels) ** 2, weights=weights))


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    values = -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    return float(np.average(values, weights=weights))


def multiclass_brier(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1])[labels]
    return float(np.average(np.sum((probabilities - one_hot) ** 2, axis=1), weights=weights))


def multiclass_log_loss(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return float(np.average(-np.log(clipped[np.arange(len(labels)), labels]), weights=weights))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def manual_probabilities(frame: pd.DataFrame, model: Mapping[str, Any]) -> np.ndarray:
    features = [str(value) for value in model["features"]]
    matrix = frame.loc[:, features].to_numpy(dtype=float)
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    coefficients = np.asarray(model["coefficient"], dtype=float)
    intercept = np.asarray(model["intercept"], dtype=float)
    logits = ((matrix - mean) / scale) @ coefficients.T + intercept
    if str(model["kind"]) == "binary":
        positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        return positive
    return softmax(logits)


def normalise_outcome(raw: object) -> str:
    value = str(raw)
    if value in {"REGISTERED_PRIMITIVE", "REGISTERED_REPEAT", "REGISTERED_COMPOSITE"}:
        return "REGISTERED_COMPLETION"
    return value


def frozen_panel(path_ledger: pd.DataFrame, selected: tuple[str, ...]) -> pd.DataFrame:
    predecessor = pd.read_parquet(PREDECESSOR / "decision_panel.parquet")
    panel = predecessor.loc[predecessor["decision_ordinal"].isin((6, 12))].copy()
    panel = panel.sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    panel = panel.reset_index(drop=True)
    panel["raw_structural_outcome"] = panel["raw_outcome"].map(normalise_outcome)
    scoring = panel["raw_structural_outcome"].isin(
        {"UNREGISTERED_LOOP", "REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"}
    )
    panel["unregistered_event"] = np.nan
    panel.loc[scoring, "unregistered_event"] = (
        panel.loc[scoring, "raw_structural_outcome"].eq("UNREGISTERED_LOOP").astype(float)
    )
    diagnostic = panel["raw_structural_outcome"].isin(
        {"REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"}
    )
    panel["registered_completion"] = np.nan
    panel.loc[diagnostic, "registered_completion"] = (
        panel.loc[diagnostic, "raw_structural_outcome"].eq("REGISTERED_COMPLETION").astype(float)
    )
    families = path_ledger.loc[:, [*KEYS, "family_id"]].copy()
    families["hidden_family_class"] = families["family_id"].map(
        lambda value: str(value) if str(value) in selected else OTHER_FAMILY
    )
    return panel.merge(families, on=list(KEYS), how="left", validate="one_to_one")


def registered_paths() -> frozenset[tuple[int, ...]]:
    table = pd.read_csv(DICTIONARY_PATH)
    output: set[tuple[int, ...]] = set()
    for raw in table["all_valid_oriented_paths"].astype(str):
        output.update(tuple(int(state) for state in path) for path in ast.literal_eval(raw))
    return frozenset(output)


def canonical_core(core: tuple[int, ...]) -> tuple[int, ...]:
    return min(core[index:] + core[:index] for index in range(len(core)))


def primitive_root(core: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    for width in range(2, len(core) // 2 + 1):
        if len(core) % width == 0 and core[:width] * (len(core) // width) == core:
            return core[:width], len(core) // width
    return core, 1


def canonical_identity(path_value: Sequence[int]) -> dict[str, Any]:
    path = tuple(int(value) for value in path_value)
    require(len(path) >= 3 and path[0] == path[-1], "path is not closed")
    core = path[:-1]
    canonical = canonical_core(core)
    root, depth = primitive_root(canonical)
    rotation_offset = next(
        index for index in range(len(core)) if core[index:] + core[:index] == canonical
    )
    if depth > 1:
        motif = "repeat_like"
    elif len(set(root)) == len(root):
        motif = "primitive_like"
    else:
        motif = "composite_like"
    v2_semantic_id: str | None = None
    v2_compatible = False
    try:
        definition = decompose_closed_path(path)
        motif = {
            "primitive": "primitive_like",
            "repeat": "repeat_like",
            "composite": "composite_like",
        }[str(definition.motif_type)]
        canonical = tuple(int(value) for value in definition.canonical_orientation[:-1])
        depth = int(definition.repeat_depth)
        rotation_offset = next(
            index for index in range(len(core)) if core[index:] + core[:index] == canonical
        )
        v2_semantic_id = str(definition.semantic_loop_id)
        v2_compatible = True
    except UnsupportedLoopError:
        pass
    closed = canonical + (canonical[0],)
    family = f"unregistered_{motif}__" + "-".join(str(value) for value in closed)
    reverse = canonical_core(tuple(reversed(path))[:-1])
    reverse_root, reverse_depth = primitive_root(reverse)
    reverse_motif = (
        "repeat_like"
        if reverse_depth > 1
        else "primitive_like"
        if len(set(reverse_root)) == len(reverse_root)
        else "composite_like"
    )
    reverse_closed = reverse + (reverse[0],)
    reverse_family = f"unregistered_{reverse_motif}__" + "-".join(
        str(value) for value in reverse_closed
    )
    return {
        "family_id": family,
        "canonical_path": closed,
        "oriented_path": path,
        "orientation_id": f"{family}__o_" + "-".join(str(value) for value in path),
        "rotation_offset": rotation_offset,
        "reverse_family_id": reverse_family,
        "reverse_orientation_equivalent": family == reverse_family,
        "motif_type": motif,
        "repeat_depth": depth,
        "transition_length": len(core),
        "revisit_count": len(core) - len(set(core)),
        "v2_semantic_id": v2_semantic_id,
        "v2_compatible": v2_compatible,
    }


def first_unregistered(
    states_value: Sequence[int],
    ordinals_value: Sequence[int],
    *,
    decision_bar: int,
    decision_event_index: int,
    known_paths: frozenset[tuple[int, ...]],
) -> tuple[tuple[int, ...], int, int] | None:
    states: list[int] = []
    ordinals: list[int] = []
    for state, ordinal in zip(states_value, ordinals_value, strict=True):
        if not states or states[-1] != int(state):
            states.append(int(state))
            ordinals.append(int(ordinal))
    causal = max(index for index, ordinal in enumerate(ordinals) if ordinal <= decision_bar)
    require(causal == decision_event_index, "decision event index differs")
    candidates: list[tuple[tuple[int, ...], int, int]] = []
    for completion in range(decision_event_index + 1, len(states)):
        if ordinals[completion] > decision_bar + 6:
            break
        for start in range(completion - 2, max(0, completion - 8) - 1, -1):
            if states[start] != states[completion]:
                continue
            path = tuple(states[start : completion + 1])
            if path not in known_paths:
                candidates.append((path, completion, ordinals[completion]))
    return min(candidates, key=lambda row: (row[2], row[1], row[0])) if candidates else None


def safety_audit(artifacts: Path) -> dict[str, Any]:
    files = (
        EXPERIMENT_DIR / "contract.json",
        artifacts / "contract.json",
        artifacts / "decision.json",
    )
    for path in files:
        document = read_json(path)
        for key, value in SAFETY_FLAGS.items():
            require(document.get(key) == value, f"safety flag differs in {path.name}: {key}")
    decision = read_json(artifacts / "decision.json")
    allowed = set(read_json(EXPERIMENT_DIR / "contract.json")["primary_decision_categories"])
    require(str(decision["decision"]) in allowed, "decision category is outside contract")
    return {"files_checked": len(files), "passed": True}


def population_feature_target_audit(
    panel: pd.DataFrame, predictions: pd.DataFrame, artifacts: Path
) -> dict[str, Any]:
    reconstruction = read_json(artifacts / "opening_population_reconstruction.json")
    features = read_json(artifacts / "trajectory_feature_reconstruction.json")
    configuration = read_json(artifacts / "model_configurations.json")
    predecessor_configuration = read_json(PREDECESSOR / "model_configurations.json")
    require(len(panel) == 15_549, "opening population row count differs")
    require(set(panel["decision_ordinal"].astype(int)) == {6, 12}, "checkpoint differs")
    require(pd.to_datetime(panel["session"]).max() < pd.Timestamp("2025-08-23"), "protected row")
    require(reconstruction["protected_rows_materialised"] == 0, "protected audit differs")
    require(float(features["maximum_feature_difference"]) <= 1e-12, "feature reconstruction")
    require(
        configuration["T0_features"] == predecessor_configuration["models"]["T0"]["features"],
        "T0 surface differs",
    )
    require(
        configuration["T1_features"] == predecessor_configuration["models"]["T1"]["features"],
        "T1 surface differs",
    )
    require(
        tuple(configuration["T1_features"][-18:]) == TRAJECTORY_FEATURES,
        "18-feature trajectory surface differs",
    )
    assessment = panel.loc[panel["year"].eq(2025) & panel["unregistered_event"].notna()]
    expected = assessment.loc[:, list(KEYS)].reset_index(drop=True)
    actual = predictions.loc[:, list(KEYS)].reset_index(drop=True)
    require(expected.equals(actual), "assessment prediction population differs")
    outcomes = predictions["raw_structural_outcome"]
    expected_u = outcomes.eq("UNREGISTERED_LOOP").astype(float)
    require(
        np.array_equal(expected_u, predictions["unregistered_event"].to_numpy(dtype=float)),
        "unregistered target differs",
    )
    diagnostic = outcomes.isin({"REGISTERED_COMPLETION", "NO_REGISTERED_COMPLETION"})
    expected_r = outcomes.loc[diagnostic].eq("REGISTERED_COMPLETION").astype(float)
    require(
        np.array_equal(
            expected_r,
            predictions.loc[diagnostic, "registered_completion"].to_numpy(dtype=float),
        ),
        "registered diagnostic target differs",
    )
    census = pd.read_csv(artifacts / "structural_outcome_census.csv")
    for year in (2024, 2025):
        for checkpoint in (6, 12):
            stored = census.loc[
                census["year"].eq(year) & census["decision_ordinal"].eq(checkpoint)
            ].set_index("raw_outcome")
            require(
                set(stored.index.astype(str)) == set(RAW_OUTCOME_CLASSES),
                "raw five-category census is incomplete",
            )
            source = panel.loc[panel["year"].eq(year) & panel["decision_ordinal"].eq(checkpoint)]
            for outcome in RAW_OUTCOME_CLASSES:
                require(
                    int(stored.loc[outcome, "rows"])
                    == int(source["raw_structural_outcome"].eq(outcome).sum()),
                    "raw structural census count differs",
                )
    target_manifest = read_json(artifacts / "binary_target_manifest.json")
    for period, year in (("development", 2024), ("assessment", 2025)):
        source = panel.loc[panel["year"].eq(year)]
        require(
            set(target_manifest["counts"][period]) == set(RAW_OUTCOME_CLASSES),
            "target manifest omits a raw category",
        )
        for outcome in RAW_OUTCOME_CLASSES:
            require(
                int(target_manifest["counts"][period][outcome])
                == int(source["raw_structural_outcome"].eq(outcome).sum()),
                "target manifest count differs",
            )
    eligible = panel["unregistered_event"].notna()
    slate_counts = panel.loc[eligible].groupby("slate_id", sort=True).size()
    expected_weights = (
        panel.loc[eligible, "slate_id"].map((1.0 / slate_counts).to_dict()).to_numpy(dtype=float)
    )
    weight_difference = float(
        np.max(np.abs(panel.loc[eligible, "row_weight"].to_numpy(dtype=float) - expected_weights))
    )
    require(weight_difference <= 1e-15, "slate weighting differs")
    require(
        panel.loc[~eligible, "row_weight"].isna().all(),
        "excluded targets unexpectedly received slate weight",
    )
    ledger = pd.read_parquet(PREDECESSOR / "trajectory_ledger.parquet")
    opening_ledger = ledger.loc[ledger["decision_ordinal"].isin((6, 12))]
    anchors = {
        checkpoint: tuple(
            sorted(
                opening_ledger.loc[
                    opening_ledger["decision_ordinal"].eq(checkpoint), "anchor_ordinal"
                ]
                .astype(int)
                .unique()
            )
        )
        for checkpoint in (6, 12)
    }
    require(anchors == {6: (2, 4, 6), 12: (4, 8, 12)}, "opening anchors differ")
    maximum_trajectory_difference = 0.0
    causality_failures = 0
    for checkpoint, expected_anchors in ((6, (2, 4, 6)), (12, (4, 8, 12))):
        checkpoint_ledger = opening_ledger.loc[opening_ledger["decision_ordinal"].eq(checkpoint)]
        require(
            tuple(sorted(checkpoint_ledger["anchor_ordinal"].astype(int).unique()))
            == expected_anchors,
            f"checkpoint {checkpoint} anchor triplet differs",
        )
        causality_failures += int(
            (
                pd.to_datetime(checkpoint_ledger["latest_input_bar_complete_timestamp_utc"])
                > pd.to_datetime(checkpoint_ledger["anchor_available_timestamp_utc"])
            ).sum()
        )
        causality_failures += int(
            (
                pd.to_datetime(checkpoint_ledger["anchor_available_timestamp_utc"])
                > pd.to_datetime(checkpoint_ledger["decision_available_timestamp_utc"])
            ).sum()
        )
    require(causality_failures == 0, "an anchor used a future bar")
    for behaviour in BEHAVIOURS:
        pivot = opening_ledger.pivot(
            index=list(KEYS), columns="anchor_role", values=behaviour
        ).reset_index()
        reconstructed = panel.loc[:, list(KEYS)].merge(pivot, on=list(KEYS), validate="one_to_one")
        early = reconstructed["E0"].to_numpy(dtype=float)
        middle = reconstructed["E1"].to_numpy(dtype=float)
        final = reconstructed["E2"].to_numpy(dtype=float)
        first = middle - early
        recent = final - middle
        expected_features = {
            behaviour: final,
            f"{behaviour}_change": final - early,
            f"{behaviour}_acceleration": recent - first,
            f"{behaviour}_reversal": (
                (np.sign(first) != np.sign(recent)) & (first != 0.0) & (recent != 0.0)
            ).astype(float),
        }
        for feature, expected_values in expected_features.items():
            maximum_trajectory_difference = max(
                maximum_trajectory_difference,
                float(np.max(np.abs(expected_values - panel[feature].to_numpy(dtype=float)))),
            )
    require(
        maximum_trajectory_difference <= 1e-12,
        "independent trajectory reconstruction differs",
    )
    return {
        "rows": len(panel),
        "assessment_rows": len(predictions),
        "maximum_feature_difference": float(features["maximum_feature_difference"]),
        "maximum_weight_difference": weight_difference,
        "maximum_independent_trajectory_difference": maximum_trajectory_difference,
        "anchor_causality_failures": causality_failures,
        "anchors": {str(key): list(value) for key, value in anchors.items()},
        "passed": True,
    }


def path_and_family_audit(
    panel: pd.DataFrame, path_ledger: pd.DataFrame, artifacts: Path
) -> dict[str, Any]:
    known_paths = registered_paths()
    source = panel.loc[panel["raw_structural_outcome"].eq("UNREGISTERED_LOOP")].merge(
        path_ledger, on=list(KEYS), validate="one_to_one", suffixes=("", "_ledger")
    )
    mismatches = 0
    for row in source.itertuples(index=False):
        event = first_unregistered(
            row.state_path_through_horizon,
            row.bar_ordinals_through_horizon,
            decision_bar=int(row.repo_bar_start_ordinal),
            decision_event_index=int(row.decision_event_index),
            known_paths=known_paths,
        )
        if event is None:
            mismatches += 1
            continue
        path, completion_index, completion_bar = event
        identity = canonical_identity(path)
        stored_path = tuple(int(value) for value in row.completed_state_transition_sequence)
        conditions = (
            path == stored_path,
            completion_index == int(row.completion_event_index),
            completion_bar == int(row.completion_bar_ordinal),
            completion_bar - int(row.repo_bar_start_ordinal) == int(row.bars_until_completion),
            identity["family_id"] == str(row.family_id),
            identity["canonical_path"] == tuple(int(value) for value in row.canonical_path),
            identity["oriented_path"] == tuple(int(value) for value in row.oriented_path),
            identity["orientation_id"] == str(row.orientation_id),
            int(identity["rotation_offset"]) == int(row.rotation_offset),
            identity["reverse_family_id"] == str(row.reverse_family_id),
            bool(identity["reverse_orientation_equivalent"])
            == bool(row.reverse_orientation_equivalent),
            identity["motif_type"] == str(row.motif_type_ledger),
            int(identity["repeat_depth"]) == int(row.repeat_depth),
            int(identity["transition_length"]) == int(row.transition_length),
            int(identity["revisit_count"]) == int(row.revisit_count),
            identity["v2_semantic_id"]
            == (None if pd.isna(row.v2_semantic_id) else str(row.v2_semantic_id)),
            bool(identity["v2_compatible"]) == bool(row.v2_compatible),
            pd.Timestamp(row.event_available_timestamp_utc)
            > pd.Timestamp(row.decision_timestamp_utc),
        )
        mismatches += int(not all(conditions))
    require(mismatches == 0, f"unregistered path mismatches={mismatches}")
    development = path_ledger.loc[path_ledger["year"].eq(2024)]
    census_rows = []
    for family, group in development.groupby("family_id", sort=True):
        stock_counts = group.groupby("symbol").size()
        outcomes = len(group)
        sessions = int(group["session"].nunique())
        stocks = int(group["symbol"].nunique())
        months = int(group["year_month"].nunique())
        share = float(stock_counts.max() / outcomes)
        census_rows.append(
            {
                "family_id": str(family),
                "outcomes": outcomes,
                "eligible": bool(
                    outcomes >= 30
                    and sessions >= 20
                    and stocks >= 8
                    and months >= 4
                    and share <= 0.30
                ),
            }
        )
    census = pd.DataFrame(census_rows)
    expected_selected = tuple(
        census.loc[census["eligible"]]
        .sort_values(["outcomes", "family_id"], ascending=[False, True], kind="mergesort")[
            "family_id"
        ]
        .head(4)
        .astype(str)
    )
    mapping = read_json(artifacts / "hidden_family_mapping.json")
    stored_selected = tuple(str(value) for value in mapping["selected_families"])
    require(expected_selected == stored_selected, "development hidden-family selection differs")
    require(
        bool(mapping["frozen_before_assessment_family_support"]),
        "family mapping was not declared frozen before assessment support",
    )
    support = pd.read_csv(artifacts / "hidden_family_support.csv")
    for row in support.itertuples(index=False):
        year = int(row.year)
        frame = path_ledger.loc[path_ledger["year"].eq(year)].copy()
        frame["target"] = frame["family_id"].map(
            lambda value: str(value) if str(value) in stored_selected else OTHER_FAMILY
        )
        subset = frame.loc[frame["target"].eq(str(row.hidden_family_class))]
        require(len(subset) == int(row.outcomes), "hidden-family support differs")
    return {
        "paths_checked": len(source),
        "path_mismatches": mismatches,
        "development_family_count": int(development["family_id"].nunique()),
        "eligible_family_count": int(census["eligible"].sum()),
        "selected_families": list(stored_selected),
        "selection_used_2024_only": True,
        "passed": True,
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = quantile * float(sorted_weights.sum())
    location = min(
        int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left")),
        len(sorted_values) - 1,
    )
    return float(sorted_values[location])


def binary_values(frame: pd.DataFrame, model: str, target: str) -> dict[str, float]:
    labels = frame[target].to_numpy(dtype=int)
    probabilities = frame[f"{model}_probability"].to_numpy(dtype=float)
    weights = frame["row_weight"].to_numpy(dtype=float)
    return {
        "log_loss": binary_log_loss(labels, probabilities, weights),
        "brier_score": binary_brier(labels, probabilities, weights),
        "auc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(labels, probabilities, sample_weight=weights)
        ),
    }


def binary_increment_values(
    frame: pd.DataFrame,
    baseline: str,
    candidate: str,
    target: str,
    *,
    top_decile_thresholds: Mapping[str, float] | None = None,
) -> dict[str, float]:
    base = binary_values(frame, baseline, target)
    added = binary_values(frame, candidate, target)
    output = {
        "log_loss_improvement": base["log_loss"] - added["log_loss"],
        "brier_improvement": base["brier_score"] - added["brier_score"],
        "auc_improvement": added["auc"] - base["auc"],
    }
    if top_decile_thresholds is not None:
        labels = frame[target].to_numpy(dtype=int)
        weights = frame["row_weight"].to_numpy(dtype=float)
        precisions = {}
        for model in (baseline, candidate):
            probabilities = frame[f"{model}_probability"].to_numpy(dtype=float)
            selected = probabilities >= float(top_decile_thresholds[model])
            precisions[model] = float(np.average(labels[selected], weights=weights[selected]))
        output["top_decile_precision_improvement"] = precisions[candidate] - precisions[baseline]
    return output


def family_values(
    frame: pd.DataFrame, model: str, class_order: tuple[str, ...]
) -> dict[str, float]:
    probabilities = frame.loc[
        :, [f"{model}_probability_{label}" for label in class_order]
    ].to_numpy(dtype=float)
    indices = {label: index for index, label in enumerate(class_order)}
    labels = frame["hidden_family_class"].map(indices).to_numpy(dtype=int)
    weights = frame["row_weight"].to_numpy(dtype=float)
    ranks = np.argsort(-probabilities, axis=1, kind="stable")
    top_two = np.asarray(
        [label in ranks[index, :2] for index, label in enumerate(labels)], dtype=float
    )
    return {
        "multiclass_log_loss": multiclass_log_loss(labels, probabilities, weights),
        "multiclass_brier": multiclass_brier(labels, probabilities, weights),
        "top_two_accuracy": float(np.average(top_two, weights=weights)),
    }


def family_increment_values(frame: pd.DataFrame, class_order: tuple[str, ...]) -> dict[str, float]:
    base = family_values(frame, "F0", class_order)
    added = family_values(frame, "F1", class_order)
    return {
        "multiclass_log_loss_improvement": base["multiclass_log_loss"]
        - added["multiclass_log_loss"],
        "multiclass_brier_improvement": base["multiclass_brier"] - added["multiclass_brier"],
        "top_two_accuracy_improvement": added["top_two_accuracy"] - base["top_two_accuracy"],
    }


def model_and_metric_audit(
    panel: pd.DataFrame, predictions: pd.DataFrame, artifacts: Path
) -> dict[str, Any]:
    configuration = read_json(artifacts / "model_configurations.json")
    coefficient_document = read_json(artifacts / "model_coefficients.json")
    models = cast(dict[str, dict[str, Any]], coefficient_document["primary_models"])
    scoring = panel.loc[panel["unregistered_event"].notna()].reset_index(drop=True)
    development = scoring.loc[scoring["year"].eq(2024)].copy()
    assessment = scoring.loc[scoring["year"].eq(2025)].reset_index(drop=True)
    require(
        len(models) == int(configuration["model_count"]) <= 6,
        "primary model count differs",
    )
    require(
        set(models) in ({"U0", "U1", "R0", "R1"}, {"U0", "U1", "R0", "R1", "F0", "F1"}),
        "primary model names differ",
    )
    maximum_scaler_difference = 0.0
    maximum_probability_difference = 0.0
    rows_manually_reconstructed = 0
    for name, model in models.items():
        if name.startswith("R"):
            development_input = development.loc[development["registered_completion"].notna()]
            assessment_input = assessment.loc[assessment["registered_completion"].notna()]
        elif name.startswith("F"):
            development_input = development.loc[development["hidden_family_class"].notna()]
            assessment_input = assessment.loc[assessment["hidden_family_class"].notna()]
        else:
            development_input = development
            assessment_input = assessment
        features = [str(value) for value in model["features"]]
        matrix = development_input.loc[:, features].to_numpy(dtype=float)
        expected_mean = matrix.mean(axis=0)
        expected_scale = matrix.std(axis=0)
        expected_scale[expected_scale == 0.0] = 1.0
        maximum_scaler_difference = max(
            maximum_scaler_difference,
            float(np.max(np.abs(expected_mean - np.asarray(model["scaler_mean"])))),
            float(np.max(np.abs(expected_scale - np.asarray(model["scaler_scale"])))),
        )
        require(max(int(value) for value in model["n_iter"]) < 300, f"{name} did not converge")
        probabilities = manual_probabilities(assessment_input, model)
        rows_manually_reconstructed += min(100, len(assessment_input))
        if str(model["kind"]) == "binary":
            stored = predictions.loc[
                predictions[f"{name}_probability"].notna(), f"{name}_probability"
            ].to_numpy(dtype=float)
            difference = float(np.max(np.abs(probabilities - stored)))
        else:
            class_order = tuple(str(value) for value in model["class_order"])
            stored = predictions.loc[
                predictions["hidden_family_class"].notna(),
                [f"{name}_probability_{label}" for label in class_order],
            ].to_numpy(dtype=float)
            difference = float(np.max(np.abs(probabilities - stored)))
        maximum_probability_difference = max(maximum_probability_difference, difference)
    require(maximum_scaler_difference <= 1e-12, "development-only scaler differs")
    require(maximum_probability_difference <= 1e-12, "manual probability reconstruction differs")
    thresholds = cast(
        dict[str, dict[str, float]], configuration["development_prediction_thresholds"]
    )
    maximum_threshold_difference = 0.0
    for name in ("U0", "U1"):
        probabilities = manual_probabilities(development, models[name])
        weights = development["row_weight"].to_numpy(dtype=float)
        maximum_threshold_difference = max(
            maximum_threshold_difference,
            abs(
                weighted_quantile(probabilities, weights, 0.90)
                - float(thresholds[name]["top_decile_threshold"])
            ),
            abs(
                weighted_quantile(probabilities, weights, 0.80)
                - float(thresholds[name]["top_quintile_threshold"])
            ),
        )
    require(maximum_threshold_difference <= 1e-12, "development thresholds differ")

    occurrence = pd.read_csv(artifacts / "unregistered_occurrence_metrics.csv")
    diagnostic = pd.read_csv(artifacts / "registered_completion_diagnostic.csv")
    maximum_metric_difference = 0.0
    for name, table, target in (
        ("U0", occurrence, "unregistered_event"),
        ("U1", occurrence, "unregistered_event"),
        ("R0", diagnostic, "registered_completion"),
        ("R1", diagnostic, "registered_completion"),
    ):
        subset = predictions.loc[predictions[target].notna()]
        expected = binary_values(subset, name, target)
        stored = table.loc[table["model"].eq(name)].iloc[0]
        for metric in ("brier_score", "log_loss", "auc", "average_precision"):
            maximum_metric_difference = max(
                maximum_metric_difference,
                abs(float(stored[metric]) - expected[metric]),
            )
    class_order = tuple(str(value) for value in configuration["family_class_order"])
    if class_order:
        family_table = pd.read_csv(artifacts / "hidden_family_metrics.csv")
        pooled = family_table.loc[
            family_table["group_type"].eq("population")
            & family_table["group_value"].eq("POOLED_UNREGISTERED")
        ]
        family_predictions = predictions.loc[predictions["hidden_family_class"].notna()]
        for name in ("F0", "F1"):
            expected = family_values(family_predictions, name, class_order)
            stored = pooled.loc[pooled["model"].eq(name)].iloc[0]
            for metric in (
                "multiclass_log_loss",
                "multiclass_brier",
                "top_two_accuracy",
            ):
                maximum_metric_difference = max(
                    maximum_metric_difference,
                    abs(float(stored[metric]) - expected[metric]),
                )
    require(maximum_metric_difference <= 1e-12, "independent pooled metrics differ")
    return {
        "models_checked": sorted(models),
        "rows_manually_reconstructed": rows_manually_reconstructed,
        "maximum_scaler_difference": maximum_scaler_difference,
        "maximum_threshold_difference": maximum_threshold_difference,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "development_only_preprocessing": True,
        "development_only_coefficients": True,
        "passed": True,
    }


def permute_columns(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    seed: int,
    include_year: bool,
) -> pd.DataFrame:
    result = frame.copy()
    source = frame.loc[:, list(features)].to_numpy(copy=True)
    positions = result.columns.get_indexer(pd.Index(features)).astype(int).tolist()
    rng = np.random.default_rng(seed)
    grouping = ["slate_id"]
    if include_year:
        grouping.insert(0, "year")
    for values in frame.groupby(grouping, sort=True, observed=True).indices.values():
        target = np.asarray(values, dtype=int)
        selected = target[rng.permutation(len(target))]
        result.iloc[target, positions] = source[selected]
    return result


def session_draws(frame: pd.DataFrame, draws: int) -> tuple[np.ndarray, ...]:
    sessions = np.asarray(sorted(frame["session"].astype(str).unique()), dtype=object)
    locations = {
        session: np.flatnonzero(frame["session"].astype(str).to_numpy() == session)
        for session in sessions
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = []
    for _ in range(draws):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        output.append(np.concatenate([locations[str(session)] for session in sampled]))
    return tuple(output)


def bootstrap_audit(predictions: pd.DataFrame, artifacts: Path) -> dict[str, Any]:
    configuration = read_json(artifacts / "model_configurations.json")
    thresholds = cast(
        dict[str, dict[str, float]], configuration["development_prediction_thresholds"]
    )
    top_thresholds = {
        model: float(values["top_decile_threshold"]) for model, values in thresholds.items()
    }
    transition_median = float(configuration["development_transition_probability_median"])
    class_order = tuple(str(value) for value in configuration["family_class_order"])
    expected: dict[tuple[str, str, str, int, str], float] = {}
    for draw, locations in enumerate(session_draws(predictions, 25)):
        sampled = predictions.iloc[locations].reset_index(drop=True)
        scopes = {
            "pooled": pd.Series(True, index=sampled.index),
            "ordinal_6": sampled["decision_ordinal"].eq(6),
            "ordinal_12": sampled["decision_ordinal"].eq(12),
            "low_transition": sampled["transition_probability"].le(transition_median),
        }
        for population, mask in scopes.items():
            primary = sampled.loc[mask & sampled["unregistered_event"].notna()]
            for metric, value in binary_increment_values(
                primary,
                "U0",
                "U1",
                "unregistered_event",
                top_decile_thresholds=top_thresholds,
            ).items():
                expected[("A_primary", population, "U1_minus_U0", draw, metric)] = value
            diagnostic = sampled.loc[mask & sampled["registered_completion"].notna()]
            for metric, value in binary_increment_values(
                diagnostic, "R0", "R1", "registered_completion"
            ).items():
                if metric in {"log_loss_improvement", "brier_improvement"}:
                    expected[("A_diagnostic", population, "R1_minus_R0", draw, metric)] = value
        if class_order:
            family = sampled.loc[sampled["hidden_family_class"].notna()]
            for metric, value in family_increment_values(family, class_order).items():
                expected[("B", "pooled", "F1_minus_F0", draw, metric)] = value
    stored = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    draw_rows = stored.loc[stored["record_type"].eq("draw")]
    require(len(draw_rows) == len(expected), "bootstrap draw row count differs")
    maximum_draw_difference = 0.0
    for row in draw_rows.itertuples(index=False):
        key = (
            str(row.stage),
            str(row.population),
            str(row.comparison),
            int(row.draw),
            str(row.metric),
        )
        require(key in expected, f"unexpected bootstrap key {key}")
        maximum_draw_difference = max(
            maximum_draw_difference, abs(float(row.value) - expected[key])
        )
    require(maximum_draw_difference <= 1e-12, "bootstrap draw differs")
    maximum_interval_difference = 0.0
    interval_rows = stored.loc[stored["record_type"].eq("interval")]
    for row in interval_rows.itertuples(index=False):
        samples = np.asarray(
            [
                value
                for (stage, population, comparison, _draw, metric), value in expected.items()
                if stage == str(row.stage)
                and population == str(row.population)
                and comparison == str(row.comparison)
                and metric == str(row.metric)
            ],
            dtype=float,
        )
        level = float(row.interval_level)
        alpha = (1.0 - level) / 2.0
        lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
        maximum_interval_difference = max(
            maximum_interval_difference,
            abs(float(row.lower) - float(lower)),
            abs(float(row.upper) - float(upper)),
        )
    require(maximum_interval_difference <= 1e-12, "bootstrap interval differs")
    return {
        "draws": 25,
        "resampled_unit": "whole_session",
        "maximum_draw_difference": maximum_draw_difference,
        "maximum_interval_difference": maximum_interval_difference,
        "models_refit_inside_draws": False,
        "passed": True,
    }


def attribution_audit(panel: pd.DataFrame, artifacts: Path) -> dict[str, Any]:
    coefficient_document = read_json(artifacts / "model_coefficients.json")
    model = cast(dict[str, Any], coefficient_document["primary_models"]["U1"])
    assessment = panel.loc[
        panel["year"].eq(2025) & panel["unregistered_event"].notna()
    ].reset_index(drop=True)
    labels = assessment["unregistered_event"].to_numpy(dtype=int)
    weights = assessment["row_weight"].to_numpy(dtype=float)
    original = manual_probabilities(assessment, model)
    original_values = {
        "log_loss": binary_log_loss(labels, original, weights),
        "brier": binary_brier(labels, original, weights),
        "auc": float(roc_auc_score(labels, original, sample_weight=weights)),
    }
    table = pd.read_csv(artifacts / "trajectory_group_attribution.csv")
    maximum_difference = 0.0
    features = [str(value) for value in model["features"]]
    coefficients = np.asarray(model["coefficient"], dtype=float)[0]
    locations = {feature: index for index, feature in enumerate(features)}
    for group_index, (group_name, group_features) in enumerate(TRAJECTORY_GROUPS.items()):
        stored_draws = table.loc[
            table["record_type"].eq("permutation_draw") & table["trajectory_group"].eq(group_name)
        ]
        require(len(stored_draws) == 20, f"{group_name} attribution draw count differs")
        group_coefficients = np.asarray(
            [coefficients[locations[feature]] for feature in group_features]
        )
        summary = table.loc[
            table["record_type"].eq("coefficient_summary")
            & table["trajectory_group"].eq(group_name)
        ].iloc[0]
        maximum_difference = max(
            maximum_difference,
            abs(
                float(summary["sum_absolute_standardised_coefficients"])
                - float(np.abs(group_coefficients).sum())
            ),
            abs(
                float(summary["signed_standardised_coefficient_sum"])
                - float(group_coefficients.sum())
            ),
        )
        for draw in range(20):
            permuted = permute_columns(
                assessment,
                group_features,
                seed=ATTRIBUTION_SEED + group_index * 100 + draw,
                include_year=False,
            )
            probabilities = manual_probabilities(permuted, model)
            expected = {
                "log_loss_deterioration": binary_log_loss(labels, probabilities, weights)
                - original_values["log_loss"],
                "brier_deterioration": binary_brier(labels, probabilities, weights)
                - original_values["brier"],
                "auc_deterioration": original_values["auc"]
                - float(roc_auc_score(labels, probabilities, sample_weight=weights)),
            }
            stored = stored_draws.loc[stored_draws["permutation_draw"].eq(draw)].iloc[0]
            for metric, value in expected.items():
                maximum_difference = max(maximum_difference, abs(float(stored[metric]) - value))
    require(maximum_difference <= 1e-12, "grouped attribution differs")
    return {
        "groups": len(TRAJECTORY_GROUPS),
        "permutations_per_group": 20,
        "model_refits": 0,
        "maximum_difference": maximum_difference,
        "passed": True,
    }


def null_audit(panel: pd.DataFrame, artifacts: Path) -> dict[str, Any]:
    configuration = read_json(artifacts / "model_configurations.json")
    class_order = tuple(str(value) for value in configuration["family_class_order"])
    coefficient_document = read_json(artifacts / "model_coefficients.json")
    primary = cast(dict[str, dict[str, Any]], coefficient_document["primary_models"])
    serialised_draws = {
        int(value["draw"]): cast(dict[str, dict[str, Any]], value["models"])
        for value in coefficient_document["null_models"]
    }
    require(set(serialised_draws) == set(range(5)), "null model draws differ")
    scoring = panel.loc[panel["unregistered_event"].notna()].reset_index(drop=True)
    stored = pd.read_csv(artifacts / "null_metrics.csv")
    stored_draws = stored.loc[stored["record_type"].eq("draw")]
    maximum_difference = 0.0
    maximum_scaler_difference = 0.0
    bundle_failures = 0
    expected_count = 0
    for draw in range(5):
        permuted = permute_columns(
            scoring, TRAJECTORY_FEATURES, seed=NULL_SEED + draw, include_year=True
        )
        for (_year, _slate), source_group in scoring.groupby(
            ["year", "slate_id"], sort=True, observed=True
        ):
            target_group = permuted.loc[source_group.index]
            source_bundles = Counter(
                map(tuple, source_group.loc[:, list(TRAJECTORY_FEATURES)].to_numpy())
            )
            target_bundles = Counter(
                map(tuple, target_group.loc[:, list(TRAJECTORY_FEATURES)].to_numpy())
            )
            bundle_failures += int(source_bundles != target_bundles)
        development = permuted.loc[permuted["year"].eq(2024)]
        assessment = permuted.loc[permuted["year"].eq(2025)].copy()
        null_models = serialised_draws[draw]
        for name, model in null_models.items():
            if name.startswith("R"):
                fit_frame = development.loc[development["registered_completion"].notna()]
            elif name.startswith("F"):
                fit_frame = development.loc[development["hidden_family_class"].notna()]
            else:
                fit_frame = development
            matrix = fit_frame.loc[:, [str(value) for value in model["features"]]].to_numpy(
                dtype=float
            )
            expected_mean = matrix.mean(axis=0)
            expected_scale = matrix.std(axis=0)
            expected_scale[expected_scale == 0.0] = 1.0
            maximum_scaler_difference = max(
                maximum_scaler_difference,
                float(np.max(np.abs(expected_mean - np.asarray(model["scaler_mean"])))),
                float(np.max(np.abs(expected_scale - np.asarray(model["scaler_scale"])))),
            )
        temporary = assessment.loc[
            :,
            [
                "row_weight",
                "unregistered_event",
                "registered_completion",
                "hidden_family_class",
            ],
        ].copy()
        temporary["U0_probability"] = manual_probabilities(assessment, primary["U0"])
        temporary["U1_probability"] = manual_probabilities(assessment, null_models["U1"])
        comparisons: dict[str, dict[str, float]] = {
            "U1_minus_U0": binary_increment_values(temporary, "U0", "U1", "unregistered_event")
        }
        diagnostic_mask = assessment["registered_completion"].notna()
        diagnostic = assessment.loc[diagnostic_mask]
        diagnostic_predictions = temporary.loc[diagnostic_mask].copy()
        diagnostic_predictions["R0_probability"] = manual_probabilities(diagnostic, primary["R0"])
        diagnostic_predictions["R1_probability"] = manual_probabilities(
            diagnostic, null_models["R1"]
        )
        comparisons["R1_minus_R0"] = binary_increment_values(
            diagnostic_predictions, "R0", "R1", "registered_completion"
        )
        if class_order:
            family_mask = assessment["hidden_family_class"].notna()
            family = assessment.loc[family_mask]
            family_predictions = temporary.loc[family_mask].copy()
            for name, model in (("F0", primary["F0"]), ("F1", null_models["F1"])):
                probabilities = manual_probabilities(family, model)
                for index, label in enumerate(class_order):
                    family_predictions[f"{name}_probability_{label}"] = probabilities[:, index]
            comparisons["F1_minus_F0"] = family_increment_values(family_predictions, class_order)
        allowed = {
            "U1_minus_U0": {"log_loss_improvement", "brier_improvement"},
            "R1_minus_R0": {"log_loss_improvement", "brier_improvement"},
            "F1_minus_F0": {
                "multiclass_log_loss_improvement",
                "multiclass_brier_improvement",
            },
        }
        for comparison, values in comparisons.items():
            for metric in allowed[comparison]:
                expected_count += 1
                row = stored_draws.loc[
                    stored_draws["draw"].eq(draw)
                    & stored_draws["comparison"].eq(comparison)
                    & stored_draws["metric"].eq(metric)
                ]
                require(len(row) == 1, "null draw metric is missing")
                maximum_difference = max(
                    maximum_difference, abs(float(row.iloc[0]["value"]) - values[metric])
                )
    require(len(stored_draws) == expected_count, "null draw row count differs")
    require(bundle_failures == 0, "trajectory null split a trajectory bundle")
    require(maximum_scaler_difference <= 1e-12, "null used non-development scaler")
    require(maximum_difference <= 1e-12, "trajectory null increment differs")
    return {
        "draws": 5,
        "bundle_failures": bundle_failures,
        "maximum_scaler_difference": maximum_scaler_difference,
        "maximum_increment_difference": maximum_difference,
        "passed": True,
    }


def support_audit(
    panel: pd.DataFrame, path_ledger: pd.DataFrame, selected: tuple[str, ...]
) -> dict[str, Any]:
    assessment = panel.loc[panel["year"].eq(2025) & panel["unregistered_event"].notna()]
    counts = assessment["raw_structural_outcome"].value_counts()
    maximum_stock_share = float(assessment.groupby("symbol").size().max() / len(assessment))
    maximum_class_share = float(
        assessment["unregistered_event"].value_counts().max() / len(assessment)
    )
    stage_a_conditions = {
        "minimum_rows": len(assessment) >= 5_500,
        "minimum_sessions": assessment["session"].nunique() >= 140,
        "minimum_stocks": assessment["symbol"].nunique() >= 15,
        "eight_months": assessment["year_month"].nunique() == 8,
        "minimum_unregistered": int(counts.get("UNREGISTERED_LOOP", 0)) >= 1_000,
        "minimum_registered": int(counts.get("REGISTERED_COMPLETION", 0)) >= 250,
        "minimum_no_completion": int(counts.get("NO_REGISTERED_COMPLETION", 0)) >= 2_500,
        "maximum_stock_share": maximum_stock_share <= 0.10,
        "maximum_binary_class_share": maximum_class_share <= 0.75,
        "minimum_trajectory_retention": len(panel) / 15_549 >= 0.95,
    }
    assessment_paths = path_ledger.loc[path_ledger["year"].eq(2025)].copy()
    assessment_paths["hidden_family_class"] = assessment_paths["family_id"].map(
        lambda value: str(value) if str(value) in selected else OTHER_FAMILY
    )
    family_counts = assessment_paths["hidden_family_class"].value_counts()
    family_share = float(family_counts.max() / len(assessment_paths))
    stage_b_stock_share = float(
        assessment_paths.groupby("symbol").size().max() / len(assessment_paths)
    )
    selected_with_20 = sum(int(family_counts.get(family, 0)) >= 20 for family in selected)
    stage_b_conditions = {
        "minimum_rows": len(assessment_paths) >= 1_000,
        "minimum_sessions": assessment_paths["session"].nunique() >= 100,
        "minimum_stocks": assessment_paths["symbol"].nunique() >= 15,
        "two_selected_families_with_20": selected_with_20 >= 2,
        "minimum_final_classes": family_counts.size >= 3,
        "maximum_stock_share": stage_b_stock_share <= 0.15,
        "maximum_family_share": family_share <= 0.70,
    }
    return {
        "stage_a": {
            "passed": all(stage_a_conditions.values()),
            "conditions": stage_a_conditions,
            "rows": len(assessment),
            "maximum_stock_share": maximum_stock_share,
            "maximum_binary_class_share": maximum_class_share,
        },
        "stage_b": {
            "passed": len(selected) >= 2 and all(stage_b_conditions.values()),
            "conditions": stage_b_conditions,
            "rows": len(assessment_paths),
            "maximum_stock_share": stage_b_stock_share,
            "maximum_family_share": family_share,
        },
    }


def decision_audit(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    path_ledger: pd.DataFrame,
    artifacts: Path,
) -> dict[str, Any]:
    configuration = read_json(artifacts / "model_configurations.json")
    class_order = tuple(str(value) for value in configuration["family_class_order"])
    selected = tuple(
        str(value)
        for value in read_json(artifacts / "hidden_family_mapping.json")["selected_families"]
    )
    support = support_audit(panel, path_ledger, selected)
    real_u = binary_increment_values(predictions, "U0", "U1", "unregistered_event")
    diagnostic = predictions.loc[predictions["registered_completion"].notna()]
    real_r = binary_increment_values(diagnostic, "R0", "R1", "registered_completion")
    bootstrap = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    null = pd.read_csv(artifacts / "null_metrics.csv")

    def bootstrap_lower(stage: str, comparison: str, metric: str, level: float) -> float:
        rows = bootstrap.loc[
            bootstrap["record_type"].eq("draw")
            & bootstrap["stage"].eq(stage)
            & bootstrap["population"].eq("pooled")
            & bootstrap["comparison"].eq(comparison)
            & bootstrap["metric"].eq(metric),
            "value",
        ].to_numpy(dtype=float)
        return float(np.quantile(rows, (1.0 - level) / 2.0))

    def nulls_exceeded(comparison: str, metric: str, real: float) -> int:
        values = null.loc[
            null["record_type"].eq("draw")
            & null["comparison"].eq(comparison)
            & null["metric"].eq(metric),
            "value",
        ].to_numpy(dtype=float)
        return int((real > values).sum())

    stage_a_months = 0
    for month in sorted(predictions["year_month"].astype(str).unique()):
        frame = predictions.loc[predictions["year_month"].astype(str).eq(month)]
        stage_a_months += int(
            binary_increment_values(frame, "U0", "U1", "unregistered_event")["log_loss_improvement"]
            > 0.0
        )
    checkpoint_increments = {
        checkpoint: binary_increment_values(
            predictions.loc[predictions["decision_ordinal"].eq(checkpoint)],
            "U0",
            "U1",
            "unregistered_event",
        )["log_loss_improvement"]
        for checkpoint in (6, 12)
    }
    stage_a_conditions = {
        "log_loss_improves": real_u["log_loss_improvement"] > 0.0,
        "brier_improves": real_u["brier_improvement"] > 0.0,
        "auc_not_reduced": real_u["auc_improvement"] >= 0.0,
        "bootstrap_90_log_loss_lower_non_negative": bootstrap_lower(
            "A_primary", "U1_minus_U0", "log_loss_improvement", 0.90
        )
        >= 0.0,
        "bootstrap_90_brier_lower_non_negative": bootstrap_lower(
            "A_primary", "U1_minus_U0", "brier_improvement", 0.90
        )
        >= 0.0,
        "positive_log_loss_in_six_of_eight_months": stage_a_months >= 6,
        "neither_checkpoint_materially_adverse": min(checkpoint_increments.values()) >= -0.001,
        "real_log_loss_or_brier_exceeds_four_of_five_nulls": (
            nulls_exceeded("U1_minus_U0", "log_loss_improvement", real_u["log_loss_improvement"])
            >= 4
            or nulls_exceeded("U1_minus_U0", "brier_improvement", real_u["brier_improvement"]) >= 4
        ),
        "concentration_gates_pass": bool(support["stage_a"]["passed"]),
    }
    stage_a_passes = all(stage_a_conditions.values())
    stage_b_conditions: dict[str, bool] = {"support_available": bool(class_order)}
    stage_b_months = 0
    stage_b_checkpoint_increments: dict[int, float] = {}
    real_f: dict[str, float] | None = None
    if class_order:
        family = predictions.loc[predictions["hidden_family_class"].notna()]
        real_f = family_increment_values(family, class_order)
        for month in sorted(family["year_month"].astype(str).unique()):
            frame = family.loc[family["year_month"].astype(str).eq(month)]
            stage_b_months += int(
                family_increment_values(frame, class_order)["multiclass_log_loss_improvement"] > 0.0
            )
        stage_b_checkpoint_increments = {
            checkpoint: family_increment_values(
                family.loc[family["decision_ordinal"].eq(checkpoint)], class_order
            )["multiclass_log_loss_improvement"]
            for checkpoint in (6, 12)
        }
        stage_b_conditions.update(
            {
                "multiclass_log_loss_improves": real_f["multiclass_log_loss_improvement"] > 0.0,
                "multiclass_brier_improves": real_f["multiclass_brier_improvement"] > 0.0,
                "top_two_not_reduced": real_f["top_two_accuracy_improvement"] >= 0.0,
                "bootstrap_80_log_loss_lower_non_negative": bootstrap_lower(
                    "B", "F1_minus_F0", "multiclass_log_loss_improvement", 0.80
                )
                >= 0.0,
                "bootstrap_80_brier_lower_non_negative": bootstrap_lower(
                    "B", "F1_minus_F0", "multiclass_brier_improvement", 0.80
                )
                >= 0.0,
                "positive_log_loss_in_five_months": stage_b_months >= 5,
                "neither_checkpoint_materially_adverse": min(stage_b_checkpoint_increments.values())
                >= -0.001,
                "real_log_loss_or_brier_exceeds_four_of_five_nulls": (
                    nulls_exceeded(
                        "F1_minus_F0",
                        "multiclass_log_loss_improvement",
                        real_f["multiclass_log_loss_improvement"],
                    )
                    >= 4
                    or nulls_exceeded(
                        "F1_minus_F0",
                        "multiclass_brier_improvement",
                        real_f["multiclass_brier_improvement"],
                    )
                    >= 4
                ),
                "concentration_gates_pass": bool(support["stage_b"]["passed"]),
            }
        )
    stage_b_passes = bool(class_order) and all(stage_b_conditions.values())
    point_estimate_improves = bool(
        real_u["log_loss_improvement"] > 0.0
        or real_u["brier_improvement"] > 0.0
        or (
            real_f is not None
            and (
                real_f["multiclass_log_loss_improvement"] > 0.0
                or real_f["multiclass_brier_improvement"] > 0.0
            )
        )
    )
    if stage_a_passes and stage_b_passes:
        expected_decision = "opening_trajectories_predict_unregistered_events_and_families"
    elif stage_a_passes:
        expected_decision = "opening_trajectories_predict_unregistered_events_only"
    elif stage_b_passes:
        expected_decision = "opening_trajectories_predict_hidden_families_only"
    elif point_estimate_improves:
        expected_decision = "opening_trajectory_signal_descriptive_only"
    else:
        expected_decision = "no_opening_trajectory_unregistered_increment"
    expected_stage_b_status = (
        "hidden_family_support_insufficient"
        if not class_order
        else "hidden_family_prediction_supported"
        if stage_b_passes
        else "hidden_family_prediction_not_supported"
    )
    stored = read_json(artifacts / "decision.json")
    require(stored["decision"] == expected_decision, "primary decision differs")
    require(stored["stage_b_status"] == expected_stage_b_status, "Stage B status differs")
    require(bool(stored["stage_a_passes"]) == stage_a_passes, "Stage A pass flag differs")
    require(bool(stored["stage_b_passes"]) == stage_b_passes, "Stage B pass flag differs")
    require(stored["stage_a_conditions"] == stage_a_conditions, "Stage A logic differs")
    require(stored["stage_b_conditions"] == stage_b_conditions, "Stage B logic differs")
    return {
        "decision": expected_decision,
        "stage_b_status": expected_stage_b_status,
        "stage_a_passes": stage_a_passes,
        "stage_b_passes": stage_b_passes,
        "stage_a_conditions": stage_a_conditions,
        "stage_b_conditions": stage_b_conditions,
        "stage_a_positive_months": stage_a_months,
        "stage_b_positive_months": stage_b_months,
        "registered_diagnostic_recomputed": real_r,
        "support": support,
        "passed": True,
    }


def audit(artifacts: Path = DEFAULT_ARTIFACTS) -> dict[str, Any]:
    artifacts = artifacts.expanduser().resolve()
    required = (
        "contract.json",
        "source_manifest.json",
        "protected_boundary_audit.json",
        "opening_population_reconstruction.json",
        "trajectory_feature_reconstruction.json",
        "structural_outcome_census.csv",
        "binary_target_manifest.json",
        "model_configurations.json",
        "model_coefficients.json",
        "assessment_predictions.parquet",
        "unregistered_occurrence_metrics.csv",
        "registered_completion_diagnostic.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "transition_split_metrics.csv",
        "trajectory_group_attribution.csv",
        "unregistered_path_ledger.parquet",
        "unregistered_path_census.csv",
        "hidden_family_mapping.json",
        "hidden_family_support.csv",
        "hidden_family_metrics.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
        "concentration_metrics.csv",
        "decision.json",
        "determinism_check.json",
    )
    missing = [name for name in required if not (artifacts / name).is_file()]
    require(not missing, f"required artifacts missing: {missing}")
    path_ledger = pd.read_parquet(artifacts / "unregistered_path_ledger.parquet")
    mapping = read_json(artifacts / "hidden_family_mapping.json")
    selected = tuple(str(value) for value in mapping["selected_families"])
    panel = frozen_panel(path_ledger, selected)
    predictions = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    source = read_json(artifacts / "source_manifest.json")
    protected = read_json(artifacts / "protected_boundary_audit.json")
    require(source["raw_data_downloaded"] is False, "raw data was downloaded")
    require(source["one_minute_data_opened"] is False, "one-minute data was opened")
    require(int(source["protected_rows_materialised"]) == 0, "protected source row")
    require(int(protected["protected_rows_materialised"]) == 0, "protected row")
    require(protected["development_start"] == "2024-01-01", "development start differs")
    require(
        protected["development_end_inclusive"] == "2024-12-31",
        "development end differs",
    )
    require(protected["assessment_start"] == "2025-01-01", "assessment start differs")
    require(
        protected["assessment_end_inclusive"] == "2025-08-22",
        "assessment end differs",
    )
    development_sessions = pd.to_datetime(panel.loc[panel["year"].eq(2024), "session"])
    assessment_sessions = pd.to_datetime(panel.loc[panel["year"].eq(2025), "session"])
    require(
        development_sessions.between("2024-01-01", "2024-12-31").all(),
        "development assignment leaves frozen dates",
    )
    require(
        assessment_sessions.between("2025-01-01", "2025-08-22").all(),
        "assessment assignment leaves frozen dates",
    )
    require(
        pd.Timestamp(protected["maximum_session"]) < pd.Timestamp("2025-08-23"), "protected date"
    )
    components = {
        "safety": safety_audit(artifacts),
        "population_features_targets": population_feature_target_audit(
            panel, predictions, artifacts
        ),
        "paths_and_families": path_and_family_audit(panel, path_ledger, artifacts),
        "models_and_metrics": model_and_metric_audit(panel, predictions, artifacts),
        "grouped_attribution": attribution_audit(panel, artifacts),
        "bootstrap": bootstrap_audit(predictions, artifacts),
        "trajectory_null": null_audit(panel, artifacts),
        "decision": decision_audit(panel, predictions, path_ledger, artifacts),
    }
    determinism = read_json(artifacts / "determinism_check.json")
    require(bool(determinism["passed"]), "fast determinism check did not pass")
    require(
        float(determinism["maximum_probability_difference"]) <= 1e-12,
        "determinism probability difference exceeds tolerance",
    )
    result = {
        **SAFETY_FLAGS,
        "passed": all(bool(value["passed"]) for value in components.values()),
        "fail_closed": True,
        "independent_from_runner_helpers": True,
        "components": components,
        "determinism_verified": True,
        "protected_rows_materialised": 0,
    }
    require(bool(result["passed"]), "an independent audit component failed")
    (artifacts / "lightweight_audit.json").write_text(canonical_json(result), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def main() -> int:
    result = audit(parse_args().artifacts)
    print(canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
