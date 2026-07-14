#!/usr/bin/env python3
"""Independent audit for the hierarchical loop-quality algorithm V1.

This module intentionally imports no production runner, model helper, feature
builder, selection helper, metric helper, gate helper, grade helper, or
falsification helper.  It reconstructs the frozen design from the contract and
the pinned parent artifacts.

The pre-artifact mode is read-only.  The pre-score mode may independently
refit the 2024 causal development models after a production fit bundle exists,
but it never opens a later-period panel.  Post-score opens the already-scored
2025 and backward-2023 panels only after verifying both sealed phase locks, and
then independently reconstructs predictions, diagnostics, gates, and the
demotion-only transfer decision.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    WORKSPACE
    / "work/contracts/20260711-hierarchical-loop-quality-algorithm-v1.json"
)
CONTRACT_SHA256 = (
    "f6956b6ab0495a49669f714df834d1fd0fdaa13b0ecf4b123d6c54c0fc9b5936"
)
RUNNER_PATH = WORKSPACE / "work/run_hierarchical_loop_quality_algorithm_v1.py"

QUALITY_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
V3_ROOT = Path("/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710")
DEFAULT_ROOT = Path("/private/tmp/stocker_hierarchical_loop_quality_algorithm_v1_20260711")
V1_RUNNER_PATH = WORKSPACE / "work/run_per_loop_movement_quality.py"
V1_RUNNER_SHA256 = "7da5e88e603583d3dba7422569bc8e27837171c7165e69bcaafade472738e2ea"
V1_PROVISIONAL_SUPPORT_PATH = QUALITY_ROOT / "provisional_support_2024.csv"
V1_PROVISIONAL_SUPPORT_SHA256 = (
    "5974edec6960c961182628ca3c854d55c9c6ed7aab402e0dfbb6bcfdea4bddd1"
)

TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
NUMERIC_CONTROLS = (
    "b0_entry_numeric",
    "b0_entry_high_stress",
    "entry_time_sin",
    "entry_time_cos",
    "current_bar_log_return",
    "return_sum_6",
    "mean_abs_return_12",
    "session_return",
    "bar_range_pct",
)
OUTER_MONTHS = (
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
)
INNER_SCHEDULE = {
    "2024-07": ("2024-04", "2024-05", "2024-06"),
    "2024-08": ("2024-05", "2024-06", "2024-07"),
    "2024-09": ("2024-06", "2024-07", "2024-08"),
    "2024-10": ("2024-07", "2024-08", "2024-09"),
    "2024-11": ("2024-08", "2024-09", "2024-10"),
    "2024-12": ("2024-09", "2024-10", "2024-11"),
}
FULL_SELECTION_MONTHS = ("2024-10", "2024-11", "2024-12")
SCALE_GRID = (
    (0.0, 0.0),
    (0.125, 0.0625),
    (0.125, 0.125),
    (0.25, 0.0625),
    (0.25, 0.125),
    (0.25, 0.25),
    (0.5, 0.0625),
    (0.5, 0.125),
    (0.5, 0.25),
    (0.5, 0.5),
    (1.0, 0.0625),
    (1.0, 0.125),
    (1.0, 0.25),
    (1.0, 0.5),
    (1.0, 1.0),
)
EPSILON = 1e-12


PIN_PATHS = {
    "per_loop_movement_quality_v1_contract_sha256": (
        WORKSPACE / "work/contracts/20260710-per-loop-movement-quality-v1.json"
    ),
    "loop_quality_feature_ablation_v3_contract_sha256": (
        WORKSPACE / "work/contracts/20260710-loop-quality-feature-ablation-v3.json"
    ),
    "loop_quality_feature_ablation_v3_runner_sha256": (
        WORKSPACE / "work/run_loop_quality_feature_ablation_v3.py"
    ),
    "loop_quality_feature_ablation_v3_fit_complete_sha256": V3_ROOT
    / "fit_complete.json",
    "loop_quality_feature_ablation_v3_pre_score_audit_sha256": V3_ROOT
    / "pre_score_audit.json",
    "loop_quality_feature_ablation_v3_scoring_complete_sha256": V3_ROOT
    / "scoring_complete.json",
    "loop_quality_feature_ablation_v3_independent_artifact_audit_sha256": V3_ROOT
    / "independent_artifact_audit.json",
    "loop_quality_feature_ablation_v3_topology_manifest_sha256": V3_ROOT
    / "topology_feature_manifest.json",
    "loop_quality_feature_ablation_v3_rotation_mapping_sha256": V3_ROOT
    / "rotation_mapping.csv",
    "loop_quality_feature_ablation_v3_model_parameters_sha256": V3_ROOT
    / "model_parameters.npz",
    "loop_quality_feature_ablation_v3_oof_predictions_sha256": V3_ROOT
    / "oof_predictions_2024.parquet",
    "fixed_cycles_csv_sha256": QUALITY_ROOT / "fixed_cycles.csv",
    "frozen_semimarkov_parameters_npz_sha256": STATE_ROOT
    / "frozen_semimarkov_parameters.npz",
    "quality_thresholds_2024_json_sha256": QUALITY_ROOT
    / "quality_thresholds_2024.json",
    "quality_feature_manifest_json_sha256": QUALITY_ROOT / "feature_manifest.json",
    "quality_fit_manifest_json_sha256": QUALITY_ROOT / "fit_manifest.json",
    "parent_oof_predictions_2024_parquet_sha256": QUALITY_ROOT
    / "oof_predictions_2024.parquet",
    "parent_training_long_2024_parquet_sha256": QUALITY_ROOT
    / "training_long_2024.parquet",
    "anchor_panel_train_2024_parquet_sha256": PRICE_ROOT
    / "anchor_panel_train_2024.parquet",
    "parent_final_cycle_tiers_csv_sha256": QUALITY_ROOT / "final_cycle_tiers.csv",
    "parent_gates_json_sha256": QUALITY_ROOT / "gates.json",
    "parent_summary_json_sha256": QUALITY_ROOT / "summary.json",
}

FIT_ARTIFACTS = (
    "feature_manifest.json",
    "route_mapping.csv",
    "fold_schedule.json",
    "hyperparameter_grid.json",
    "candidate_fit_audit_2024.csv",
    "inner_selection_2024.csv",
    "outer_fold_audit_2024.csv",
    "oof_predictions_2024.parquet",
    "model_parameters.npz",
    "full_fit_audit.json",
    "support_2024.json",
    "cell_diagnostics_2024.csv",
    "calibration_diagnostics_2024.csv",
    "rotation_diagnostics_2024.csv",
    "falsification_diagnostics_2024.json",
    "algorithm_gates_2024.json",
    "per_loop_grades_2024.csv",
    "provisional_decision.json",
    "fit_source_hashes_pre_fit.json",
    "fit_source_hashes.json",
)

SCORING_HASHED_ARTIFACTS = tuple(
    name
    for period in ("2025", "2023")
    for name in (
        f"scoring_predictions_{period}.parquet",
        f"support_{period}.json",
        f"cell_diagnostics_{period}.csv",
        f"calibration_diagnostics_{period}.csv",
        f"rotation_diagnostics_{period}.csv",
        f"algorithm_gates_{period}.json",
        f"per_loop_grades_{period}.csv",
    )
) + (
    "evaluation_source_hashes.json",
    "period_transfer_gates.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


@dataclass
class Audit:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def check(self, name: str, passed: bool, details: Any = None) -> None:
        self.checks.append(
            {"name": name, "pass": bool(passed), "details": json_safe(details)}
        )

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(row["pass"] for row in self.checks)


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text())


def verify_contract_semantics(audit: Audit) -> dict[str, Any]:
    contract = load_contract()
    audit.check("contract_sha256_exact", sha256(CONTRACT_PATH) == CONTRACT_SHA256)
    audit.check(
        "safety_labels_exact",
        contract.get("research_only") is True
        and contract.get("live_ordering_enabled") is False
        and contract.get("order_placement") == "disabled"
        and contract.get("economic_edge_claim") is False
        and contract.get("deployment_enabled") is False,
    )
    periods = contract["periods"]
    audit.check(
        "periods_and_prospective_status_exact",
        periods["causal_outer_oof_months"] == list(OUTER_MONTHS)
        and periods["fit_period"] == 2024
        and periods["development_score"] == 2025
        and periods["backward_portability_score"] == 2023
        and periods["partial_2026_permitted"] is False
        and periods["prospective_claim_permitted"] is False
        and periods["later_period_promotion_permitted"] is False,
    )
    grid = contract["scale_grid"]
    actual_grid = tuple(tuple(float(item) for item in pair) for pair in grid["pairs"])
    audit.check(
        "literal_fifteen_pair_grid_exact",
        actual_grid == SCALE_GRID and grid["pair_count"] == 15,
        actual_grid,
    )
    schedule = {
        month: tuple(values)
        for month, values in contract["causal_nested_selection"][
            "inner_validation_schedule"
        ].items()
    }
    causal = all(
        inner < outer
        for outer, inner_months in schedule.items()
        for inner in inner_months
    )
    audit.check(
        "causal_inner_outer_and_full_schedule_exact",
        schedule == INNER_SCHEDULE
        and causal
        and tuple(
            contract["causal_nested_selection"]["final_full_fit_selection_months"]
        )
        == FULL_SELECTION_MONTHS,
        schedule,
    )
    model = contract["model_specification"]
    audit.check(
        "model_specification_exact",
        model["class"] == "multinomial LogisticRegression"
        and model["C"] == 0.2
        and model["solver"] == "lbfgs"
        and model["max_iter"] == 2000
        and model["tol"] == 1e-10
        and model["random_state"] == 20260711
        and model["temperature"] == 1.0
        and model["class_order"] == [0, 1, 2]
        and model["same_selected_scale_pair_for_all_six_models"] is True,
    )
    feature = contract["feature_construction"]
    audit.check(
        "hierarchical_feature_semantics_exact",
        feature["context_block"]["width"] == 17
        and feature["route_topology_block"]["width"] == 63
        and feature["cycle_block"]["width"] == 20
        and feature["compatible_cycle_state_block"]["width"] == 44
        and feature["compatible_cycle_state_block"]["centering_scope"]
        == "within cycle, never global across the forty-four routes"
        and feature["total_width_nonzero_pair"] == 144,
    )
    primary = contract["primary_comparisons_and_gates"]
    uncertainty = primary["uncertainty"]
    audit.check(
        "primary_multiplicity_and_bootstrap_exact",
        uncertainty["draws"] == 20000
        and uncertainty["seed"] == 20260711
        and uncertainty["endpoint_count"] == 8
        and uncertainty["one_sided_confidence_level_each_endpoint"] == 0.99375
        and "ceil(n_sessions/L)" in uncertainty["resample_algorithm"]
        and uncertainty["quantile_method"] == "numpy.quantile method=linear",
    )
    falsification = contract["falsification_checks"]
    audit.check(
        "falsification_multiplicity_and_rng_exact",
        falsification["draws"] == 999
        and falsification["maximum_p_value_each_of_four_statistics"] == 0.01
        and falsification["minimum_unique_anchors_each_stratum"] == 2
        and "draw-major" in falsification["rng"]
        and falsification["predictions_refitted_per_null_draw"] is False,
    )
    named = contract["named_cycle_development_candidates"]
    audit.check(
        "named_candidate_gatekeeping_exact",
        named["good_hypothesis_family_size"] == 60
        and named["high_hypothesis_family_size"] == 60
        and named["high_is_gatekept_by_good"] is True
        and "Exactly ten components" in named["component_family_each_cycle_horizon_tier"]
        and "ceil(n_calendar_sessions/L)" in named["component_uncertainty"]
        and "qhier75" in named["candidate_probability_substitution"]
        and named["parent_grade_changed"] is False
        and named["prospective_validated"] is False,
    )
    integrity = contract["integrity_and_safety"]
    audit.check(
        "integrity_and_no_shadow_rules_exact",
        integrity["no_2026_rows"] is True
        and integrity["existing_frozen_quality_grades_unchanged"] is True
        and integrity["existing_aggregate_movement_shadow_unchanged"] is True
        and integrity["existing_per_loop_quality_shadow_unchanged"] is True
        and integrity["new_shadow_created_by_this_contract"] is False
        and integrity["provider_volume_name_if_referenced"] == "historical_volume"
        and integrity["provider_volume_is_exchange_wide_or_order_flow"] is False,
    )
    return contract


def verify_source_pins(audit: Audit, contract: Mapping[str, Any]) -> dict[str, str]:
    declared = contract["frozen_lineage"]["source_pins"]
    actual: dict[str, str] = {}
    missing: list[str] = []
    mismatches: dict[str, dict[str, str]] = {}
    for name, path in PIN_PATHS.items():
        if not path.is_file():
            missing.append(name)
            continue
        digest = sha256(path)
        actual[name] = digest
        expected = str(declared.get(name, ""))
        if digest != expected:
            mismatches[name] = {"expected": expected, "actual": digest}
    audit.check(
        "all_twenty_two_direct_source_pins_present",
        not missing and set(declared) == set(PIN_PATHS),
        {"missing": missing, "extra": sorted(set(declared) - set(PIN_PATHS))},
    )
    audit.check("all_twenty_two_direct_source_hashes_exact", not mismatches, mismatches)
    supplementary = {
        "per_loop_runner.py": (
            V1_RUNNER_PATH,
            V1_RUNNER_SHA256,
        ),
        "provisional_support_2024.csv": (
            V1_PROVISIONAL_SUPPORT_PATH,
            V1_PROVISIONAL_SUPPORT_SHA256,
        ),
    }
    supplementary_actual = {
        name: sha256(path) if path.is_file() else "missing"
        for name, (path, _) in supplementary.items()
    }
    audit.check(
        "supplementary_v1_implementation_dependencies_exact",
        all(
            supplementary_actual[name] == expected
            for name, (_, expected) in supplementary.items()
        ),
        supplementary_actual,
    )
    actual.update(supplementary_actual)
    return actual


def compatible_rotations(
    core: tuple[int, ...], current_state: int
) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current_state),)
            for index, state in enumerate(core)
            if int(state) == int(current_state)
        }
    )


def normalized_centroids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(STATE_ROOT / "frozen_semimarkov_parameters.npz") as parameters:
        means = np.asarray(parameters["means"], dtype=float)
    if means.shape != (8, 14):
        raise AssertionError(f"frozen centroid shape changed: {means.shape}")
    center = means.mean(axis=0)
    scale = means.std(axis=0, ddof=0)
    safe = np.where(scale > 0.0, scale, 1.0)
    standardized = (means - center) / safe
    standardized[:, scale == 0.0] = 0.0
    return standardized, center, scale


def topology_column_names() -> list[str]:
    return (
        [f"next_state_p_{state}" for state in range(8)]
        + [f"route_state_fraction_{state}" for state in range(8)]
        + [f"length_is_{length}" for length in (2, 3, 4)]
        + [f"next_centroid_z_{index:02d}" for index in range(14)]
        + [f"route_centroid_z_{index:02d}" for index in range(14)]
        + [f"next_minus_current_centroid_z_{index:02d}" for index in range(14)]
        + ["rotation_count_minus_one", "next_state_entropy_normalized"]
    )


def topology_values(
    core: tuple[int, ...], current_state: int, standardized: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    rotations = compatible_rotations(core, current_state)
    if not rotations:
        raise AssertionError("cycle/current-state unit has no compatible rotation")
    length = len(core)
    next_distribution = np.zeros(8, dtype=float)
    composition = np.zeros(8, dtype=float)
    for route in rotations:
        next_distribution[int(route[1])] += 1.0 / len(rotations)
        for state in route[1:]:
            composition[int(state)] += 1.0 / (len(rotations) * length)
    length_encoding = np.asarray(
        [float(length == value) for value in (2, 3, 4)], dtype=float
    )
    next_centroid = next_distribution @ standardized
    route_centroid = composition @ standardized
    delta = next_centroid - standardized[int(current_state)]
    positive = next_distribution[next_distribution > 0.0]
    entropy = (
        float(-(positive * np.log(positive)).sum() / math.log(8.0))
        if len(positive) > 1
        else 0.0
    )
    vector = np.concatenate(
        (
            next_distribution,
            composition,
            length_encoding,
            next_centroid,
            route_centroid,
            delta,
            np.asarray([len(rotations) - 1.0, entropy]),
        )
    )
    if vector.shape != (63,) or not np.isfinite(vector).all():
        raise AssertionError("independent topology vector is invalid")
    metadata = {
        "compatible_rotations": [
            "->".join(str(item) for item in route) for route in rotations
        ],
        "compatible_rotation_count": len(rotations),
        "transition_length": length,
        "next_state_entropy_normalized": entropy,
    }
    return vector, metadata


def build_rotation_mapping() -> tuple[pd.DataFrame, np.ndarray]:
    standardized, _, _ = normalized_centroids()
    cycles = pd.read_csv(QUALITY_ROOT / "fixed_cycles.csv").sort_values(
        "cycle_index", kind="stable"
    )
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for cycle in cycles.itertuples(index=False):
        closed = tuple(int(item) for item in str(cycle.cycle).split("->"))
        if closed[0] != closed[-1]:
            raise AssertionError(f"frozen cycle is not closed: {cycle.cycle}")
        core = closed[:-1]
        for state in sorted(set(core)):
            vector, metadata = topology_values(core, state, standardized)
            rows.append(
                {
                    "route_index": len(rows),
                    "route_id": f"{cycle.cycle_id}@state_{state}",
                    "cycle_id": str(cycle.cycle_id),
                    "cycle_index": int(cycle.cycle_index),
                    "current_state": int(state),
                    "cycle": str(cycle.cycle),
                    **metadata,
                }
            )
            vectors.append(vector)
    mapping = pd.DataFrame(rows).sort_values(
        ["cycle_index", "current_state"], kind="stable"
    )
    mapping = mapping.reset_index(drop=True)
    mapping["route_index"] = np.arange(len(mapping), dtype=int)
    matrix = np.vstack(vectors)
    if len(mapping) != 44 or matrix.shape != (44, 63):
        raise AssertionError("independent route mapping is not 44 by 63")
    if mapping.duplicated(["cycle_id", "current_state"]).any():
        raise AssertionError("duplicate independent route unit")
    return mapping, matrix


def verify_rotation_mapping(audit: Audit) -> tuple[pd.DataFrame, np.ndarray]:
    mapping, vectors = build_rotation_mapping()
    stored = pd.read_csv(V3_ROOT / "rotation_mapping.csv").sort_values(
        ["cycle_index", "current_state"], kind="stable"
    )
    keys_match = (
        len(stored) == 44
        and np.array_equal(stored["cycle_id"].astype(str), mapping["cycle_id"])
        and np.array_equal(
            stored["current_state"].to_numpy(dtype=int),
            mapping["current_state"].to_numpy(dtype=int),
        )
    )
    audit.check("independent_forty_four_route_keys_exact", keys_match)
    audit.check(
        "independent_rotation_count_structure_exact",
        int(mapping["compatible_rotation_count"].sum()) == 45
        and mapping.loc[
            mapping["compatible_rotation_count"].gt(1), "route_id"
        ].tolist()
        == ["cycle_15@state_1"],
    )
    audit.check(
        "independent_sixty_three_topology_columns_exact",
        len(topology_column_names()) == 63 and vectors.shape == (44, 63),
    )
    stored_vectors = stored[topology_column_names()].to_numpy(dtype=float)
    serialized_error = float(
        np.max(np.abs(stored_vectors - vectors), initial=0.0)
    )
    audit.check(
        "independent_topology_matches_pinned_v3_serialization",
        keys_match and serialized_error <= 1e-12,
        {"maximum_error": serialized_error},
    )
    return mapping, vectors


def pinned_training_topology_vectors(
    mapping: pd.DataFrame, independently_reconstructed: np.ndarray
) -> np.ndarray:
    """Return the exact serialized V3 training representation after verification.

    V3 generated its OOF parquet directly from the structural calculation but
    generated later training matrices by reading the pinned rotation-mapping
    CSV.  Decimal serialization changes a few coordinates by sub-ulp amounts.
    Those values are mathematically equivalent at the contract's 1e-12 feature
    tolerance, but tightly converged LBFGS can amplify the representation-level
    difference.  The audit therefore reconstructs and verifies all 63 features
    independently, then uses the exact pinned CSV floats for numerical replay.
    """

    stored = pd.read_csv(V3_ROOT / "rotation_mapping.csv").sort_values(
        ["cycle_index", "current_state"], kind="stable"
    ).reset_index(drop=True)
    keys_exact = (
        len(stored) == len(mapping) == 44
        and np.array_equal(stored["cycle_id"].astype(str), mapping["cycle_id"].astype(str))
        and np.array_equal(
            stored["cycle_index"].to_numpy(dtype=int),
            mapping["cycle_index"].to_numpy(dtype=int),
        )
        and np.array_equal(
            stored["current_state"].to_numpy(dtype=int),
            mapping["current_state"].to_numpy(dtype=int),
        )
    )
    serialized = stored[topology_column_names()].to_numpy(dtype=float)
    independent = np.asarray(independently_reconstructed, dtype=float)
    maximum = float(np.max(np.abs(serialized - independent), initial=0.0))
    if not keys_exact or serialized.shape != (44, 63) or maximum > 1e-12:
        raise AssertionError(
            "pinned V3 training topology failed independent reconstruction"
        )
    return serialized


def verify_topology_replay(
    audit: Audit, oof: pd.DataFrame
) -> float:
    columns = topology_column_names()
    stored = pd.read_parquet(
        V3_ROOT / "oof_predictions_2024.parquet",
        columns=["anchor_id", "cycle_index", *columns],
    )
    left = oof.sort_values(["anchor_id", "cycle_index"], kind="stable").reset_index(
        drop=True
    )
    right = stored.sort_values(
        ["anchor_id", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    ids_exact = (
        len(left) == len(right)
        and np.array_equal(left["anchor_id"], right["anchor_id"])
        and np.array_equal(left["cycle_index"], right["cycle_index"])
    )
    maximum = math.inf
    if ids_exact:
        maximum = float(
            np.max(
                np.abs(
                    left[columns].to_numpy(dtype=float)
                    - right[columns].to_numpy(dtype=float)
                ),
                initial=0.0,
            )
        )
    audit.check(
        "all_oof_topology_values_independently_exact",
        ids_exact and maximum <= 1e-12,
        {"ids_exact": ids_exact, "maximum_error": maximum},
    )
    return maximum


def numeric_medians() -> dict[str, float]:
    manifest = json.loads((QUALITY_ROOT / "feature_manifest.json").read_text())
    medians = {
        name: float(manifest["numeric_medians"][name]) for name in NUMERIC_CONTROLS
    }
    if set(medians) != set(NUMERIC_CONTROLS):
        raise AssertionError("numeric median keys changed")
    return medians


def add_month_key(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    dates = pd.to_datetime(output["session_date"], errors="raise")
    if not dates.dt.year.eq(2024).all():
        raise AssertionError("non-2024 row reached fit-only audit")
    output["month_key"] = dates.dt.strftime("%Y-%m")
    return output


def add_independent_topology(
    frame: pd.DataFrame, mapping: pd.DataFrame, vectors: np.ndarray
) -> pd.DataFrame:
    route_lookup = {
        (str(row.cycle_id), int(row.current_state)): int(row.route_index)
        for row in mapping.itertuples(index=False)
    }
    keys = list(
        zip(frame["cycle_id"].astype(str), frame["state"].astype(int), strict=True)
    )
    try:
        route_index = np.asarray([route_lookup[key] for key in keys], dtype=int)
    except KeyError as error:
        raise AssertionError(f"row outside frozen route mapping: {error}") from error
    output = frame.copy()
    output["route_index"] = route_index
    output["current_state"] = output["state"].to_numpy(dtype=int)
    output.loc[:, topology_column_names()] = vectors[route_index]
    return output


def load_training_frame(
    mapping: pd.DataFrame, vectors: np.ndarray
) -> pd.DataFrame:
    frame = pd.read_parquet(QUALITY_ROOT / "training_long_2024.parquet")
    if len(frame) != 32677 or not frame["loop_occurs"].eq(1).all():
        raise AssertionError("realized 2024 training cohort changed")
    frame = add_month_key(frame)
    replay_vectors = pinned_training_topology_vectors(mapping, vectors)
    return add_independent_topology(frame, mapping, replay_vectors)


def parent_oof_columns() -> list[str]:
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "cycle_index",
        "cycle_id",
        "state",
        "history_token",
        "loop_probability",
        "first_order_probability",
        "loop_occurs",
        "positive_cycle_count",
        "conditional_weight",
    ]
    for target in TARGETS:
        for horizon in HORIZONS:
            columns.extend(
                (
                    f"quality_class__{target}__h{horizon}",
                    f"joint_good_target__{target}__h{horizon}",
                    f"joint_high_target__{target}__h{horizon}",
                )
            )
            for model in ("qcontext", "qcycle"):
                for tier in TIERS:
                    columns.extend(
                        (
                            f"{model}__{target}__h{horizon}__{tier}",
                            f"joint__{model}__{target}__h{horizon}__{tier}",
                        )
                    )
    return columns


def load_oof_frame(mapping: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
    frame = pd.read_parquet(
        QUALITY_ROOT / "oof_predictions_2024.parquet", columns=parent_oof_columns()
    )
    if len(frame) != 216438:
        raise AssertionError("parent OOF row count changed")
    anchors = pd.read_parquet(
        PRICE_ROOT / "anchor_panel_train_2024.parquet",
        columns=["anchor_id", "state", "history_token", *NUMERIC_CONTROLS],
    )
    if anchors["anchor_id"].duplicated().any():
        raise AssertionError("anchor controls are not unique")
    frame = frame.merge(
        anchors,
        on="anchor_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "__anchor"),
        sort=False,
    )
    if len(frame) != 216438 or frame[list(NUMERIC_CONTROLS)].isna().any().any():
        raise AssertionError("independent causal-control merge failed")
    for column in ("state", "history_token"):
        if not np.array_equal(frame[column], frame[f"{column}__anchor"]):
            raise AssertionError(f"parent {column} disagrees with anchor controls")
    frame = frame.drop(columns=["state__anchor", "history_token__anchor"])
    frame.insert(0, "source_row", np.arange(len(frame), dtype=np.int64))
    frame = add_month_key(frame)
    return add_independent_topology(frame, mapping, vectors)


def verify_unique_oof_support(audit: Audit, oof: pd.DataFrame) -> dict[str, Any]:
    positive = oof.loc[oof["loop_occurs"].eq(1)].copy()
    weights = positive["conditional_weight"].to_numpy(dtype=float)
    by_anchor = positive.groupby("anchor_id", sort=False)["conditional_weight"].sum()
    per_quarter = positive.groupby("quarter", sort=True)["conditional_weight"].sum()
    per_stock = positive.groupby("symbol_norm", sort=True)["conditional_weight"].sum()
    payload = {
        "effective_weight": float(weights.sum()),
        "realized_rows": int(len(positive)),
        "sessions": int(positive["session_date"].nunique()),
        "stocks": int(positive["symbol_norm"].nunique()),
        "quarter_weights": {str(key): float(value) for key, value in per_quarter.items()},
        "minimum_stock_weight": float(per_stock.min()),
        "all_positive_anchor_weights_equal_one": bool(
            np.allclose(by_anchor.to_numpy(dtype=float), 1.0)
        ),
    }
    passed = (
        abs(payload["effective_weight"] - 14167.0) <= 1e-12
        and payload["realized_rows"] == 15584
        and payload["sessions"] == 128
        and payload["stocks"] == 22
        and payload["minimum_stock_weight"] >= 50.0
        and payload["all_positive_anchor_weights_equal_one"]
    )
    audit.check("unique_oof_support_reconstructed", passed, payload)
    return payload


def independent_support_payload(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame.loc[frame["loop_occurs"].eq(1)]
    quarter = positive.groupby("quarter")["conditional_weight"].sum().to_dict()
    stock = positive.groupby("symbol_norm")["conditional_weight"].sum().to_dict()
    checks = {
        "total_effective_weight": float(positive["conditional_weight"].sum())
        >= 10000.0,
        "each_required_quarter_weight": all(
            float(quarter.get(value, 0.0)) >= 5000.0
            for value in ("2024_q3", "2024_q4")
        ),
        "sessions": positive["session_date"].nunique() >= 100,
        "stocks": positive["symbol_norm"].nunique() >= 18,
        "each_stock_effective_weight": bool(stock and min(stock.values()) >= 50.0),
        "realized_rows_reconstruction_integrity": len(positive) >= 15000,
    }
    return {
        "compatible_rows": len(frame),
        "realized_rows": len(positive),
        "unique_realized_anchors": int(positive["anchor_id"].nunique()),
        "total_effective_weight": float(positive["conditional_weight"].sum()),
        "quarter_effective_weight": quarter,
        "sessions": int(positive["session_date"].nunique()),
        "stocks": int(positive["symbol_norm"].nunique()),
        "minimum_stock_effective_weight": float(min(stock.values())),
        "checks": checks,
        "support_pass": bool(all(checks.values())),
        "realized_rows_is_independent_support_gate": False,
    }


def expected_feature_manifest(selected_pair: tuple[float, float]) -> dict[str, Any]:
    return {
        "task_key_template": "qhier__{target}__h{horizon}",
        "targets": list(TARGETS),
        "horizons": list(HORIZONS),
        "context_width": 17,
        "topology_width": 63,
        "cycle_width": 20,
        "compatible_cycle_state_width": 44,
        "nonzero_total_width": 144,
        "numeric_controls": list(NUMERIC_CONTROLS),
        "numeric_medians": numeric_medians(),
        "topology_columns": topology_column_names(),
        "topology_scales": {
            "columns_0_through_18": 1.0,
            "columns_19_through_60": 0.5,
            "columns_61_through_62": 1.0,
        },
        "cycle_centering": "global conditional-weight center fitted within each causal training fold",
        "route_centering": "conditional-weight center within cycle; coordinates outside row cycle are zero",
        "selected_grid_index": SCALE_GRID.index(selected_pair),
        "selected_pair": list(selected_pair),
        "model": {
            "class": "multinomial LogisticRegression",
            "C": 0.2,
            "solver": "lbfgs",
            "max_iter": 2000,
            "tol": 1e-10,
            "random_state": 20260711,
            "temperature": 1.0,
        },
        "zero_pair": "exact sealed V3 qroute_topology; no V4 coefficients",
        "v1_named_grade_runner_sha256": V1_RUNNER_SHA256,
        "v1_provisional_support_sha256": V1_PROVISIONAL_SUPPORT_SHA256,
        "future_realized_feature_used": False,
        "stock_identity_feature_used": False,
        "provider_volume_used": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def raw_context(
    frame: pd.DataFrame, medians: Mapping[str, float] | None = None
) -> sparse.csr_matrix:
    fill = numeric_medians() if medians is None else dict(medians)
    numeric = frame.loc[:, list(NUMERIC_CONTROLS)].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.fillna(pd.Series(fill))
    values = numeric.to_numpy(dtype=float)
    states = frame["state"].to_numpy(dtype=int)
    if (
        not np.isfinite(values).all()
        or states.min(initial=0) < 0
        or states.max(initial=0) > 7
    ):
        raise AssertionError("invalid causal context")
    state_block = sparse.csr_matrix(np.eye(8, dtype=float)[states])
    result = sparse.hstack((state_block, sparse.csr_matrix(values)), format="csr")
    if result.shape[1] != 17:
        raise AssertionError("context width changed")
    return result


def fit_context_scaler(raw: sparse.csr_matrix, weights: np.ndarray) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.fit(raw, sample_weight=np.asarray(weights, dtype=float))
    return scaler


def weighted_hierarchy_centers(
    frame: pd.DataFrame,
    weights: np.ndarray,
    mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = np.asarray(weights, dtype=float)
    if len(frame) != len(weight) or not np.isfinite(weight).all() or weight.sum() <= 0:
        raise AssertionError("invalid hierarchy weights")
    cycles = frame["cycle_index"].to_numpy(dtype=int)
    routes = frame["route_index"].to_numpy(dtype=int)
    mu_cycle = np.bincount(cycles, weights=weight, minlength=20) / weight.sum()
    cycle_weight = np.bincount(cycles, weights=weight, minlength=20)
    if len(mu_cycle) != 20 or np.any(cycle_weight <= 0.0):
        raise AssertionError("one or more frozen cycles has zero training weight")
    route_weight = np.bincount(routes, weights=weight, minlength=44)
    route_cycle = mapping["cycle_index"].to_numpy(dtype=int)
    mu_route = route_weight / cycle_weight[route_cycle]
    for cycle in range(20):
        member = route_cycle == cycle
        if not np.isclose(mu_route[member].sum(), 1.0):
            raise AssertionError("within-cycle route centers do not sum to one")
    return mu_cycle, mu_route, route_cycle


def centered_cycle_block(
    cycles: np.ndarray, mu_cycle: np.ndarray, scale: float
) -> sparse.csr_matrix:
    dense = -np.broadcast_to(mu_cycle, (len(cycles), len(mu_cycle))).copy()
    dense[np.arange(len(cycles)), np.asarray(cycles, dtype=int)] += 1.0
    return sparse.csr_matrix(float(scale) * dense)


def centered_route_block(
    cycles: np.ndarray,
    routes: np.ndarray,
    mu_route: np.ndarray,
    route_cycle: np.ndarray,
    scale: float,
) -> sparse.csr_matrix:
    cycle_array = np.asarray(cycles, dtype=int)
    route_array = np.asarray(routes, dtype=int)
    dense = np.zeros((len(cycle_array), len(mu_route)), dtype=float)
    for cycle in range(20):
        rows = np.flatnonzero(cycle_array == cycle)
        columns = np.flatnonzero(route_cycle == cycle)
        if len(rows) == 0:
            continue
        dense[np.ix_(rows, columns)] = -mu_route[columns]
    dense[np.arange(len(route_array)), route_array] += 1.0
    return sparse.csr_matrix(float(scale) * dense)


@dataclass
class FoldDesign:
    train: sparse.csr_matrix
    validation: sparse.csr_matrix
    scaler: StandardScaler
    mu_cycle: np.ndarray
    mu_route: np.ndarray
    route_cycle: np.ndarray


def build_fold_design(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    pair: tuple[float, float],
    mapping: pd.DataFrame,
) -> FoldDesign:
    a_cycle, a_route = pair
    if pair == (0.0, 0.0):
        raise AssertionError("zero pair must use the sealed V3 endpoint")
    weights = training["conditional_weight"].to_numpy(dtype=float)
    medians = numeric_medians()
    raw_train = raw_context(training, medians)
    raw_validation = raw_context(validation, medians)
    scaler = fit_context_scaler(raw_train, weights)
    train_context = scaler.transform(raw_train).tocsr()
    validation_context = scaler.transform(raw_validation).tocsr()
    topology_columns = topology_column_names()
    train_topology = training[topology_columns].to_numpy(dtype=float).copy()
    validation_topology = validation[topology_columns].to_numpy(dtype=float).copy()
    train_topology[:, 19:61] *= 0.5
    validation_topology[:, 19:61] *= 0.5
    mu_cycle, mu_route, route_cycle = weighted_hierarchy_centers(
        training, weights, mapping
    )
    train_cycle = training["cycle_index"].to_numpy(dtype=int)
    validation_cycle = validation["cycle_index"].to_numpy(dtype=int)
    train_route = training["route_index"].to_numpy(dtype=int)
    validation_route = validation["route_index"].to_numpy(dtype=int)
    train = sparse.hstack(
        (
            train_context,
            sparse.csr_matrix(train_topology),
            centered_cycle_block(train_cycle, mu_cycle, a_cycle),
            centered_route_block(
                train_cycle, train_route, mu_route, route_cycle, a_route
            ),
        ),
        format="csr",
    )
    valid = sparse.hstack(
        (
            validation_context,
            sparse.csr_matrix(validation_topology),
            centered_cycle_block(validation_cycle, mu_cycle, a_cycle),
            centered_route_block(
                validation_cycle,
                validation_route,
                mu_route,
                route_cycle,
                a_route,
            ),
        ),
        format="csr",
    )
    if train.shape[1] != 144 or valid.shape[1] != 144:
        raise AssertionError("hierarchical design width changed")
    if not np.isfinite(train.data).all() or not np.isfinite(valid.data).all():
        raise AssertionError("non-finite hierarchical design")
    return FoldDesign(train, valid, scaler, mu_cycle, mu_route, route_cycle)


def build_qroute_design(
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, StandardScaler]:
    weights = training["conditional_weight"].to_numpy(dtype=float)
    medians = numeric_medians()
    raw_train = raw_context(training, medians)
    raw_validation = raw_context(validation, medians)
    scaler = fit_context_scaler(raw_train, weights)
    train_context = scaler.transform(raw_train).tocsr()
    validation_context = scaler.transform(raw_validation).tocsr()
    columns = topology_column_names()
    train_topology = training[columns].to_numpy(dtype=float).copy()
    validation_topology = validation[columns].to_numpy(dtype=float).copy()
    train_topology[:, 19:61] *= 0.5
    validation_topology[:, 19:61] *= 0.5
    train = sparse.hstack(
        (train_context, sparse.csr_matrix(train_topology)), format="csr"
    )
    valid = sparse.hstack(
        (validation_context, sparse.csr_matrix(validation_topology)), format="csr"
    )
    if train.shape[1] != 80 or valid.shape[1] != 80:
        raise AssertionError("qroute design width changed")
    return train, valid, scaler


def fit_ordered_model(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    qroute_zero: bool = False,
) -> LogisticRegression:
    kwargs = {
        "C": 0.2,
        "solver": "lbfgs",
        "max_iter": 1000 if qroute_zero else 2000,
        "random_state": 20260710 if qroute_zero else 20260711,
    }
    if not qroute_zero:
        kwargs["tol"] = 1e-10
    model = LogisticRegression(**kwargs)
    model.fit(
        matrix,
        np.asarray(target, dtype=int),
        sample_weight=np.asarray(weights, dtype=float),
    )
    if not np.array_equal(model.classes_, np.asarray([0, 1, 2])):
        raise AssertionError("ordered class support changed")
    if int(model.n_iter_[0]) >= int(kwargs["max_iter"]):
        raise AssertionError("ordered model did not converge")
    return model


def class_probabilities(
    model: LogisticRegression, matrix: sparse.csr_matrix
) -> np.ndarray:
    probability = np.asarray(model.predict_proba(matrix), dtype=float)
    if probability.shape != (matrix.shape[0], 3):
        raise AssertionError("ordered probability shape changed")
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or not np.allclose(probability.sum(axis=1), 1.0)
    ):
        raise AssertionError("ordered probabilities are invalid")
    return probability


def task_key(target: str, horizon: int) -> str:
    return f"qhier__{target}__h{horizon}"


def tier_probabilities(class_probability: np.ndarray) -> dict[str, np.ndarray]:
    p75 = class_probability[:, 1] + class_probability[:, 2]
    p90 = class_probability[:, 2]
    if (
        np.any(p90 > p75 + EPSILON)
        or np.any(p75 > 1.0 + EPSILON)
        or np.any(p90 < -EPSILON)
    ):
        raise AssertionError("tier probability nesting failed")
    return {"p75": p75, "p90": p90}


def fit_pair_predictions(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    pair: tuple[float, float],
    mapping: pd.DataFrame,
) -> tuple[dict[tuple[str, int, str], np.ndarray], dict[str, Any]]:
    weights = training["conditional_weight"].to_numpy(dtype=float)
    if pair == (0.0, 0.0):
        train_matrix, validation_matrix, scaler = build_qroute_design(
            training, validation
        )
        design: FoldDesign | None = None
    else:
        design = build_fold_design(training, validation, pair, mapping)
        train_matrix, validation_matrix, scaler = (
            design.train,
            design.validation,
            design.scaler,
        )
    output: dict[tuple[str, int, str], np.ndarray] = {}
    fitted: dict[str, Any] = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            observed = training[
                f"quality_class__{target}__h{horizon}"
            ].to_numpy(dtype=int)
            model = fit_ordered_model(
                train_matrix,
                observed,
                weights,
                qroute_zero=pair == (0.0, 0.0),
            )
            classes = class_probabilities(model, validation_matrix)
            for tier, values in tier_probabilities(classes).items():
                output[(target, horizon, tier)] = values
            fitted[task_key(target, horizon)] = model
    metadata = {
        "models": fitted,
        "scaler": scaler,
        "mu_cycle": None if design is None else design.mu_cycle,
        "mu_route": None if design is None else design.mu_route,
        "route_cycle": None if design is None else design.route_cycle,
        "feature_width": train_matrix.shape[1],
    }
    return output, metadata


def binary_log_loss(observed: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(observed, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def brier_loss(observed: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return (
        np.asarray(probability, dtype=float) - np.asarray(observed, dtype=float)
    ) ** 2


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if len(value) != len(weight) or weight.sum() <= 0.0:
        raise AssertionError("invalid weighted mean")
    return float(np.dot(value, weight) / weight.sum())


def attach_sealed_qroute_probabilities(oof: pd.DataFrame) -> pd.DataFrame:
    columns = ["anchor_id", "cycle_index"] + [
        f"qroute_topology__{target}__h{horizon}__{tier}"
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    sealed = pd.read_parquet(V3_ROOT / "oof_predictions_2024.parquet", columns=columns)
    output = oof.merge(
        sealed,
        on=["anchor_id", "cycle_index"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    probability_columns = columns[2:]
    if len(output) != len(oof) or output[probability_columns].isna().any().any():
        raise AssertionError("sealed qroute probability replay merge failed")
    return output


def observed_binary_target(
    frame: pd.DataFrame,
    surface: str,
    target: str,
    horizon: int,
    tier: str,
) -> np.ndarray:
    if surface == "conditional":
        ordered = frame[f"quality_class__{target}__h{horizon}"].to_numpy(dtype=int)
        return (ordered >= (1 if tier == "p75" else 2)).astype(float)
    if surface == "joint":
        name = "joint_good_target" if tier == "p75" else "joint_high_target"
        return frame[f"{name}__{target}__h{horizon}"].to_numpy(dtype=float)
    raise ValueError(surface)


def model_probability(
    frame: pd.DataFrame,
    model: str,
    surface: str,
    target: str,
    horizon: int,
    tier: str,
) -> np.ndarray:
    if model == "qhier":
        conditional = frame[
            f"qhier__{target}__h{horizon}__{tier}"
        ].to_numpy(dtype=float)
        return (
            conditional
            if surface == "conditional"
            else frame["loop_probability"].to_numpy(dtype=float) * conditional
        )
    if model == "qcontext":
        prefix = "" if surface == "conditional" else "joint__"
        return frame[
            f"{prefix}qcontext__{target}__h{horizon}__{tier}"
        ].to_numpy(dtype=float)
    if model == "qroute_topology":
        conditional = frame[
            f"qroute_topology__{target}__h{horizon}__{tier}"
        ].to_numpy(dtype=float)
        return (
            conditional
            if surface == "conditional"
            else frame["loop_probability"].to_numpy(dtype=float) * conditional
        )
    raise ValueError(model)


def equal_cell_pooled_log_loss(
    panel: pd.DataFrame, model: str, surface: str
) -> float:
    if surface == "conditional":
        frame = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
        weights = frame["conditional_weight"].to_numpy(dtype=float)
    elif surface == "joint":
        frame = panel.reset_index(drop=True)
        weights = np.ones(len(frame), dtype=float)
    else:
        raise ValueError(surface)
    cells: list[float] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                observed = observed_binary_target(
                    frame, surface, target, horizon, tier
                )
                probability = model_probability(
                    frame, model, surface, target, horizon, tier
                )
                cells.append(
                    weighted_mean(binary_log_loss(observed, probability), weights)
                )
    return float(np.mean(cells))


def falsification_observed_statistics(panel: pd.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    for baseline in ("qcontext", "qroute_topology"):
        for surface in ("conditional", "joint"):
            baseline_loss = equal_cell_pooled_log_loss(panel, baseline, surface)
            candidate_loss = equal_cell_pooled_log_loss(panel, "qhier", surface)
            if baseline_loss <= 0.0:
                raise AssertionError("falsification baseline loss is not positive")
            values[f"qhier_vs_{baseline}__{surface}"] = (
                baseline_loss - candidate_loss
            ) / baseline_loss
    return values


def prepare_evaluation_panel(
    oof: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    route_probe = f"qroute_topology__{TARGETS[0]}__h{HORIZONS[0]}__p75"
    panel = (
        oof.copy()
        if route_probe in oof.columns
        else attach_sealed_qroute_probabilities(oof)
    )
    route_count = mapping.set_index("route_index")[
        "compatible_rotation_count"
    ].to_dict()
    panel["compatible_rotation_count"] = panel["route_index"].map(route_count).astype(int)
    cuts = entropy_cutpoints(panel)
    panel["entropy_quartile"] = np.searchsorted(
        cuts,
        panel["next_state_entropy_normalized"].to_numpy(dtype=float),
        side="left",
    ).astype(np.int8)
    structural = panel["loop_probability"].to_numpy(dtype=float)
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                qhier = f"qhier__{target}__h{horizon}__{tier}"
                qroute = f"qroute_topology__{target}__h{horizon}__{tier}"
                qfull = f"qfull__{target}__h{horizon}__{tier}"
                parent_full = f"qcycle__{target}__h{horizon}__{tier}"
                panel[f"joint__{qhier}"] = structural * panel[qhier].to_numpy(float)
                panel[f"joint__{qroute}"] = structural * panel[qroute].to_numpy(float)
                panel[qfull] = panel[parent_full].to_numpy(float)
                panel[f"joint__{qfull}"] = panel[
                    f"joint__{parent_full}"
                ].to_numpy(float)
    probability_columns = [
        ("qhier", target, horizon, tier)
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    for model, target, horizon, tier in probability_columns:
        p75 = panel[f"{model}__{target}__h{horizon}__p75"].to_numpy(float)
        p90 = panel[f"{model}__{target}__h{horizon}__p90"].to_numpy(float)
        if np.any(p90 > p75) or np.any(p90 < 0.0) or np.any(p75 > 1.0):
            raise AssertionError("evaluation qhier probabilities are not nested")
        break
    return panel


def evaluation_surface(
    panel: pd.DataFrame, surface: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if surface == "conditional":
        frame = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
        return frame, frame["conditional_weight"].to_numpy(dtype=float)
    if surface == "joint":
        frame = panel.reset_index(drop=True)
        return frame, np.ones(len(frame), dtype=float)
    raise ValueError(surface)


def evaluation_probability(
    frame: pd.DataFrame,
    model: str,
    target: str,
    horizon: int,
    tier: str,
    surface: str,
) -> np.ndarray:
    prefix = "" if surface == "conditional" else "joint__"
    return frame[f"{prefix}{model}__{target}__h{horizon}__{tier}"].to_numpy(
        dtype=float
    )


def calibration_table(
    period: str,
    surface: str,
    model: str,
    target: str,
    horizon: int,
    tier: str,
    observed: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    minimum_rows: int,
) -> tuple[list[dict[str, Any]], float, float]:
    bins = np.minimum(np.floor(10.0 * np.asarray(probability, float)).astype(int), 9)
    total = float(np.asarray(weights, dtype=float).sum())
    if total <= 0.0:
        raise AssertionError("calibration surface has no weight")
    rows: list[dict[str, Any]] = []
    ece = 0.0
    supported_errors: list[float] = []
    for bin_index in range(10):
        mask = bins == bin_index
        count = int(mask.sum())
        weight = float(weights[mask].sum())
        if weight > 0.0:
            predicted = weighted_mean(probability[mask], weights[mask])
            actual = weighted_mean(observed[mask], weights[mask])
            error = abs(actual - predicted)
            ece += weight / total * error
        else:
            predicted = actual = error = math.nan
        supported = count >= minimum_rows and weight > 0.0
        if supported:
            supported_errors.append(error)
        rows.append(
            {
                "period": period,
                "surface": surface,
                "model": model,
                "target": target,
                "horizon": horizon,
                "tier": tier,
                "bin": bin_index,
                "rows": count,
                "weight": weight,
                "mean_probability": predicted,
                "event_rate": actual,
                "absolute_error": error,
                "supported": supported,
            }
        )
    maximum = max(supported_errors) if supported_errors else math.nan
    return rows, float(ece), float(maximum)


def independent_cell_diagnostics(
    panel: pd.DataFrame, period: str, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []
    for surface in ("conditional", "joint"):
        frame, weights = evaluation_surface(panel, surface)
        minimum = (
            50 if surface == "conditional" else 250
        ) if mode == "oof" else (
            100 if surface == "conditional" else 500
        )
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    observed = observed_binary_target(
                        frame, surface, target, horizon, tier
                    )
                    for model in ("qcontext", "qroute_topology", "qfull", "qhier"):
                        probability = evaluation_probability(
                            frame, model, target, horizon, tier, surface
                        )
                        log = binary_log_loss(observed, probability)
                        brier = brier_loss(observed, probability)
                        rows, ece, maximum = calibration_table(
                            period,
                            surface,
                            model,
                            target,
                            horizon,
                            tier,
                            observed,
                            probability,
                            weights,
                            minimum,
                        )
                        bins.extend(rows)
                        metrics.append(
                            {
                                "period": period,
                                "surface": surface,
                                "model": model,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(frame),
                                "weight": float(weights.sum()),
                                "positives": int(np.asarray(observed).sum()),
                                "weighted_prevalence": weighted_mean(observed, weights),
                                "log_loss": weighted_mean(log, weights),
                                "brier": weighted_mean(brier, weights),
                                "ece": ece,
                                "maximum_supported_bin_error": maximum,
                            }
                        )
    return pd.DataFrame(metrics), pd.DataFrame(bins)


def common_block_positions(n_sessions: int, draws: int = 20000) -> np.ndarray:
    if n_sessions < 1:
        raise AssertionError("bootstrap calendar is empty")
    length = min(5, n_sessions)
    starts_count = n_sessions - length + 1
    required = int(math.ceil(n_sessions / length))
    starts = np.random.Generator(np.random.PCG64(20260711)).integers(
        0, starts_count, size=(draws, required)
    )
    positions = (
        starts[:, :, None] + np.arange(length, dtype=int)[None, None, :]
    ).reshape(draws, -1)[:, :n_sessions]
    return positions.astype(np.int32, copy=False)


def resampled_nanmeans(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if not np.isfinite(array).any(axis=0).all():
        raise AssertionError("observed endpoint lacks a finite session")
    output = np.empty((len(positions), array.shape[1]), dtype=float)
    for start in range(0, len(positions), 500):
        stop = min(start + 500, len(positions))
        sampled = array[positions[start:stop]]
        finite = np.isfinite(sampled)
        count = finite.sum(axis=1)
        if (count == 0).any():
            raise AssertionError("resampled endpoint lacks a finite session")
        output[start:stop] = np.nansum(sampled, axis=1) / count
    return output


def weighted_group_series(
    frame: pd.DataFrame,
    values: np.ndarray,
    weights: np.ndarray,
    column: str,
) -> pd.Series:
    grouped = pd.DataFrame(
        {
            "group": frame[column].astype(str).to_numpy(),
            "weighted": np.asarray(values, dtype=float)
            * np.asarray(weights, dtype=float),
            "weight": np.asarray(weights, dtype=float),
        }
    ).groupby("group", sort=True).sum()
    if (grouped["weight"] <= 0.0).any():
        raise AssertionError("grouped endpoint has non-positive weight")
    return grouped["weighted"] / grouped["weight"]


def independent_comparison_payload(
    panel: pd.DataFrame,
    surface: str,
    baseline: str,
    loss_name: str,
    calendar: Sequence[str],
) -> dict[str, Any]:
    frame, weights = evaluation_surface(panel, surface)
    differences: dict[tuple[str, int, str], np.ndarray] = {}
    baseline_losses: dict[tuple[str, int, str], np.ndarray] = {}
    daily: list[pd.Series] = []
    quarters: list[pd.Series] = []
    cells: list[dict[str, Any]] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                observed = observed_binary_target(frame, surface, target, horizon, tier)
                candidate_probability = evaluation_probability(
                    frame, "qhier", target, horizon, tier, surface
                )
                baseline_probability = evaluation_probability(
                    frame, baseline, target, horizon, tier, surface
                )
                candidate = (
                    binary_log_loss(observed, candidate_probability)
                    if loss_name == "log_loss"
                    else brier_loss(observed, candidate_probability)
                )
                baseline_values = (
                    binary_log_loss(observed, baseline_probability)
                    if loss_name == "log_loss"
                    else brier_loss(observed, baseline_probability)
                )
                difference = candidate - baseline_values
                key = (target, horizon, tier)
                differences[key] = difference
                baseline_losses[key] = baseline_values
                daily.append(
                    weighted_group_series(
                        frame, difference, weights, "session_date"
                    ).rename(f"{target}_{horizon}_{tier}")
                )
                quarters.append(
                    weighted_group_series(frame, difference, weights, "quarter").rename(
                        f"{target}_{horizon}_{tier}"
                    )
                )
                difference_mean = weighted_mean(difference, weights)
                baseline_mean = weighted_mean(baseline_values, weights)
                cells.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "tier": tier,
                        "difference": difference_mean,
                        "baseline_loss": baseline_mean,
                        "relative_degradation": difference_mean / baseline_mean,
                    }
                )
    pooled_difference = float(np.mean([row["difference"] for row in cells]))
    pooled_baseline = float(np.mean([row["baseline_loss"] for row in cells]))
    daily_series = pd.concat(daily, axis=1).reindex(
        pd.Index(calendar, dtype=str)
    ).mean(axis=1, skipna=True)
    quarter_series = pd.concat(quarters, axis=1).mean(axis=1, skipna=True)
    symbols = frame["symbol_norm"].astype(str).to_numpy()
    deletions = {
        symbol: float(
            np.mean(
                [
                    weighted_mean(values[symbols != symbol], weights[symbols != symbol])
                    for values in differences.values()
                ]
            )
        )
        for symbol in sorted(set(symbols))
    }
    target_aggregates = {
        target: float(
            np.mean(
                [
                    weighted_mean(differences[(target, horizon, tier)], weights)
                    for horizon in HORIZONS
                    for tier in TIERS
                ]
            )
        )
        for target in TARGETS
    }
    horizon_aggregates = {
        str(horizon): float(
            np.mean(
                [
                    weighted_mean(differences[(target, horizon, tier)], weights)
                    for target in TARGETS
                    for tier in TIERS
                ]
            )
        )
        for horizon in HORIZONS
    }
    return {
        "pooled_difference": pooled_difference,
        "pooled_baseline": pooled_baseline,
        "relative_improvement": -pooled_difference / pooled_baseline,
        "daily": daily_series.to_numpy(dtype=float),
        "quarter_differences": quarter_series.to_dict(),
        "leave_one_stock_out_differences": deletions,
        "cell_diagnostics": cells,
        "target_aggregates": target_aggregates,
        "horizon_aggregates": horizon_aggregates,
    }


def independent_slice_diagnostics(
    panel: pd.DataFrame, period: str, mode: str
) -> pd.DataFrame:
    positive = panel.loc[panel["loop_occurs"].eq(1)].copy()
    all_groups = {
        "cycle_current_state": panel["cycle_id"].astype(str)
        + "@"
        + panel["state"].astype(str),
        "compatible_rotation_count": panel["compatible_rotation_count"].astype(str),
        "entropy_quartile": panel["entropy_quartile"].astype(str),
    }
    positive_groups = {
        "cycle_current_state": positive["cycle_id"].astype(str)
        + "@"
        + positive["state"].astype(str),
        "compatible_rotation_count": positive["compatible_rotation_count"].astype(str),
        "entropy_quartile": positive["entropy_quartile"].astype(str),
    }
    limits = {
        "minimum_realized_rows": 100 if mode == "oof" else 200,
        "minimum_effective_weight": 75.0 if mode == "oof" else 150.0,
        "minimum_sessions": 40 if mode == "oof" else 80,
        "minimum_stocks": 10,
        "required_quarters": 2 if mode == "oof" else 4,
    }
    rows: list[dict[str, Any]] = []
    for group_type, groups in all_groups.items():
        positive_values = positive_groups[group_type]
        for group_value in sorted(groups.unique()):
            support = positive.loc[positive_values.eq(group_value)]
            effective = float(support["conditional_weight"].sum())
            supported = bool(
                len(support) >= limits["minimum_realized_rows"]
                and effective >= limits["minimum_effective_weight"]
                and support["session_date"].nunique() >= limits["minimum_sessions"]
                and support["symbol_norm"].nunique() >= limits["minimum_stocks"]
                and support["quarter"].nunique() == limits["required_quarters"]
            )
            for surface in ("conditional", "joint"):
                if surface == "conditional":
                    frame = support.reset_index(drop=True)
                    weights = frame["conditional_weight"].to_numpy(float)
                else:
                    frame = panel.loc[groups.eq(group_value)].reset_index(drop=True)
                    weights = np.ones(len(frame), dtype=float)
                for baseline in ("qcontext", "qroute_topology"):
                    pooled: dict[str, float] = {}
                    for loss_name in ("log_loss", "brier"):
                        cell_values: list[float] = []
                        for target in TARGETS:
                            for horizon in HORIZONS:
                                for tier in TIERS:
                                    observed = observed_binary_target(
                                        frame, surface, target, horizon, tier
                                    )
                                    candidate = evaluation_probability(
                                        frame, "qhier", target, horizon, tier, surface
                                    )
                                    base = evaluation_probability(
                                        frame, baseline, target, horizon, tier, surface
                                    )
                                    difference = (
                                        binary_log_loss(observed, candidate)
                                        - binary_log_loss(observed, base)
                                        if loss_name == "log_loss"
                                        else brier_loss(observed, candidate)
                                        - brier_loss(observed, base)
                                    )
                                    cell_values.append(weighted_mean(difference, weights))
                        pooled[loss_name] = float(np.mean(cell_values))
                    rows.append(
                        {
                            "period": period,
                            "group_type": group_type,
                            "group_value": group_value,
                            "surface": surface,
                            "candidate": "qhier",
                            "baseline": baseline,
                            "realized_rows": len(support),
                            "effective_weight": effective,
                            "sessions": int(support["session_date"].nunique()),
                            "stocks": int(support["symbol_norm"].nunique()),
                            "quarters": int(support["quarter"].nunique()),
                            "supported": supported,
                            "pooled_log_loss_difference": pooled["log_loss"],
                            "pooled_brier_difference": pooled["brier"],
                            "sign_reversal": bool(
                                supported
                                and (
                                    pooled["log_loss"] > 0.0
                                    or pooled["brier"] > 0.0
                                )
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def independent_core_evaluation(
    panel: pd.DataFrame, period: str, mode: str
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    np.ndarray,
    tuple[str, ...],
]:
    metrics, calibration = independent_cell_diagnostics(panel, period, mode)
    rotations = independent_slice_diagnostics(panel, period, mode)
    calendar = tuple(sorted(panel["session_date"].astype(str).unique()))
    positions = common_block_positions(len(calendar))
    comparisons: dict[str, Any] = {}
    endpoint_keys: list[tuple[str, str, str]] = []
    endpoints: list[np.ndarray] = []
    for baseline in ("qcontext", "qroute_topology", "qfull"):
        comparisons[baseline] = {}
        for surface in ("conditional", "joint"):
            comparisons[baseline][surface] = {}
            for loss_name in ("log_loss", "brier"):
                payload = independent_comparison_payload(
                    panel, surface, baseline, loss_name, calendar
                )
                comparisons[baseline][surface][loss_name] = payload
                endpoint_keys.append((baseline, surface, loss_name))
                endpoints.append(payload["daily"])
    endpoint_matrix = np.column_stack(endpoints)
    resampled = resampled_nanmeans(endpoint_matrix, positions)
    for index, (baseline, surface, loss_name) in enumerate(endpoint_keys):
        quantile = 0.9875 if baseline == "qfull" else 0.99375
        payload = comparisons[baseline][surface][loss_name]
        payload["daily_mean"] = float(np.nanmean(endpoint_matrix[:, index]))
        payload["bootstrap_upper"] = float(
            np.quantile(resampled[:, index], quantile, method="linear")
        )

    indexed = metrics.set_index(["surface", "model", "target", "horizon", "tier"])
    calibration_gates: dict[str, Any] = {}
    for surface in ("conditional", "joint"):
        maximum_limit = 0.02 if surface == "conditional" else 0.01
        cells: list[dict[str, Any]] = []
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    candidate = indexed.loc[(surface, "qhier", target, horizon, tier)]
                    context = indexed.loc[(surface, "qcontext", target, horizon, tier)]
                    route = indexed.loc[
                        (surface, "qroute_topology", target, horizon, tier)
                    ]
                    maximum = float(candidate["maximum_supported_bin_error"])
                    checks = {
                        "ece_no_greater_than_qcontext": float(candidate["ece"])
                        <= float(context["ece"]),
                        "ece_no_greater_than_qroute": float(candidate["ece"])
                        <= float(route["ece"]),
                        "supported_maximum_exists": np.isfinite(maximum),
                        "absolute_maximum_within_limit": np.isfinite(maximum)
                        and maximum <= maximum_limit,
                    }
                    cells.append(
                        {
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                            "qhier_ece": float(candidate["ece"]),
                            "qcontext_ece": float(context["ece"]),
                            "qroute_ece": float(route["ece"]),
                            "qhier_maximum_supported_bin_error": maximum,
                            "checks": checks,
                            "pass": bool(all(checks.values())),
                        }
                    )
        calibration_gates[surface] = {
            "cells": cells,
            "pass": bool(all(row["pass"] for row in cells)),
        }

    primary: dict[str, Any] = {}
    thresholds = {
        "qcontext": {"conditional": 0.005, "joint": 0.0025},
        "qroute_topology": {"conditional": 0.001, "joint": 0.0005},
    }
    for baseline in ("qcontext", "qroute_topology"):
        primary[baseline] = {}
        for surface in ("conditional", "joint"):
            log = comparisons[baseline][surface]["log_loss"]
            brier = comparisons[baseline][surface]["brier"]
            supported_slices = rotations.loc[
                rotations["baseline"].eq(baseline)
                & rotations["surface"].eq(surface)
                & rotations["supported"]
            ]
            checks = {
                "minimum_relative_log_loss_improvement": log["relative_improvement"]
                >= thresholds[baseline][surface],
                "pooled_brier_below_zero": brier["pooled_difference"] < 0.0,
                "log_loss_bootstrap_upper_below_zero": log["bootstrap_upper"] < 0.0,
                "brier_bootstrap_upper_below_zero": brier["bootstrap_upper"] < 0.0,
                "quarters_no_worse": all(
                    value <= 0.0
                    for payload in (log, brier)
                    for value in payload["quarter_differences"].values()
                ),
                "stock_deletions_no_worse": all(
                    value <= 0.0
                    for payload in (log, brier)
                    for value in payload["leave_one_stock_out_differences"].values()
                ),
                "cells_within_degradation_limit": max(
                    row["relative_degradation"] for row in log["cell_diagnostics"]
                )
                <= 0.0025,
                "target_aggregates_no_worse": all(
                    value <= 0.0
                    for payload in (log, brier)
                    for value in payload["target_aggregates"].values()
                ),
                "horizon_aggregates_no_worse": all(
                    value <= 0.0
                    for payload in (log, brier)
                    for value in payload["horizon_aggregates"].values()
                ),
                "calibration": calibration_gates[surface]["pass"],
                "supported_slices_exist": len(supported_slices) > 0,
                "no_supported_slice_sign_reversal": bool(
                    len(supported_slices) > 0
                    and not supported_slices["sign_reversal"].any()
                ),
            }
            primary[baseline][surface] = {
                "log_loss": log,
                "brier": brier,
                "checks": checks,
                "pass": bool(all(checks.values())),
            }

    secondary: dict[str, Any] = {}
    for surface in ("conditional", "joint"):
        secondary[surface] = {}
        for loss_name in ("log_loss", "brier"):
            payload = comparisons["qfull"][surface][loss_name]
            context_loss = float(
                metrics.loc[
                    metrics["surface"].eq(surface)
                    & metrics["model"].eq("qcontext"),
                    loss_name,
                ].mean()
            )
            full_loss = float(
                metrics.loc[
                    metrics["surface"].eq(surface)
                    & metrics["model"].eq("qfull"),
                    loss_name,
                ].mean()
            )
            gain = context_loss - full_loss
            margin = 0.1 * gain if gain > 0.0 else math.nan
            secondary[surface][loss_name] = {
                **payload,
                "qfull_vs_context_gain": gain,
                "margin": margin,
                "precondition": gain > 0.0,
                "pass": bool(
                    gain > 0.0
                    and payload["pooled_difference"] <= margin
                    and payload["bootstrap_upper"] <= margin
                ),
            }
    positive = panel.loc[panel["loop_occurs"].eq(1)]
    if mode == "oof":
        quarter_weights = positive.groupby("quarter")["conditional_weight"].sum()
        stock_weights = positive.groupby("symbol_norm")["conditional_weight"].sum()
        support_pass = bool(
            float(positive["conditional_weight"].sum()) >= 10000.0
            and quarter_weights.get("2024_q3", 0.0) >= 5000.0
            and quarter_weights.get("2024_q4", 0.0) >= 5000.0
            and positive["session_date"].nunique() >= 100
            and positive["symbol_norm"].nunique() >= 18
            and float(stock_weights.min()) >= 50.0
            and len(positive) >= 15000
        )
    else:
        support_pass = float(positive["conditional_weight"].sum()) >= 25000.0
    gates = {
        "period": period,
        "support_pass": support_pass,
        "calibration": calibration_gates,
        "primary": primary,
        "primary_without_falsification_pass": bool(
            support_pass
            and all(
                primary[baseline][surface]["pass"]
                for baseline in ("qcontext", "qroute_topology")
                for surface in ("conditional", "joint")
            )
        ),
        "secondary_qfull_noninferiority": secondary,
        "secondary_pass": bool(
            all(
                secondary[surface][loss_name]["pass"]
                for surface in ("conditional", "joint")
                for loss_name in ("log_loss", "brier")
            )
        ),
    }
    return metrics, calibration, rotations, gates, positions, calendar


def independent_falsification(panel: pd.DataFrame) -> dict[str, Any]:
    cells = [
        (target, horizon, tier)
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    label_columns = [
        f"quality_class__{target}__h{horizon}"
        for target in TARGETS
        for horizon in HORIZONS
    ]
    anchor_columns = [
        "anchor_id",
        "symbol_norm",
        "quarter",
        "session_date",
        "start_timestamp",
        *label_columns,
    ]
    anchors = panel[anchor_columns].drop_duplicates("anchor_id", keep="first").copy()
    disagreement = panel.groupby("anchor_id", sort=False)[label_columns].nunique()
    if (disagreement > 1).any().any():
        raise AssertionError("movement labels differ across an anchor")
    anchors = anchors.sort_values(
        ["symbol_norm", "quarter", "session_date", "start_timestamp", "anchor_id"],
        kind="stable",
    ).reset_index(drop=True)
    if anchors.duplicated(
        ["symbol_norm", "quarter", "session_date", "start_timestamp", "anchor_id"]
    ).any():
        raise AssertionError("duplicate falsification ordering key")
    anchor_position = pd.Series(np.arange(len(anchors)), index=anchors["anchor_id"])
    row_anchor = panel["anchor_id"].map(anchor_position).to_numpy(dtype=int)
    labels = np.empty((len(anchors), len(cells)), dtype=np.int8)
    for index, (target, horizon, tier) in enumerate(cells):
        ordered = anchors[f"quality_class__{target}__h{horizon}"].to_numpy(int)
        labels[:, index] = ordered >= (1 if tier == "p75" else 2)
    strata: list[np.ndarray] = []
    for _, positions in anchors.groupby(
        ["symbol_norm", "quarter"], sort=True
    ).groups.items():
        indices = np.asarray(positions, dtype=int)
        if len(indices) < 2:
            raise AssertionError("falsification stratum has fewer than two anchors")
        strata.append(indices)

    models = ("qhier", "qcontext", "qroute_topology")
    surfaces = ("conditional", "joint")
    constants: dict[tuple[str, str], np.ndarray] = {}
    coefficients: dict[tuple[str, str], np.ndarray] = {}
    for surface in surfaces:
        if surface == "conditional":
            mask = panel["loop_occurs"].eq(1).to_numpy()
            weights = panel.loc[mask, "conditional_weight"].to_numpy(float)
        else:
            mask = np.ones(len(panel), dtype=bool)
            weights = np.ones(len(panel), dtype=float)
        frame = panel.loc[mask].reset_index(drop=True)
        anchor_indices = row_anchor[mask]
        occurrence = frame["loop_occurs"].to_numpy(float)
        for model in models:
            cell_constant = np.empty(len(cells), dtype=float)
            cell_coefficient = np.empty((len(anchors), len(cells)), dtype=float)
            for index, (target, horizon, tier) in enumerate(cells):
                probability = np.clip(
                    evaluation_probability(
                        frame, model, target, horizon, tier, surface
                    ),
                    EPSILON,
                    1.0 - EPSILON,
                )
                zero_loss = -np.log(1.0 - probability)
                label_slope = np.log((1.0 - probability) / probability)
                if surface == "joint":
                    label_slope *= occurrence
                denominator = float(weights.sum())
                cell_constant[index] = float(
                    np.dot(weights, zero_loss) / denominator
                )
                cell_coefficient[:, index] = np.bincount(
                    anchor_indices,
                    weights=weights * label_slope,
                    minlength=len(anchors),
                ) / denominator
            constants[(surface, model)] = cell_constant
            coefficients[(surface, model)] = cell_coefficient

    def statistics(transformed: np.ndarray) -> np.ndarray:
        losses: dict[tuple[str, str], float] = {}
        for surface in surfaces:
            for model in models:
                cell_loss = constants[(surface, model)] + np.sum(
                    coefficients[(surface, model)] * transformed, axis=0
                )
                losses[(surface, model)] = float(np.mean(cell_loss))
        output: list[float] = []
        for baseline in ("qcontext", "qroute_topology"):
            for surface in surfaces:
                base = losses[(surface, baseline)]
                output.append((base - losses[(surface, "qhier")]) / base)
        return np.asarray(output, dtype=float)

    observed = statistics(labels)
    direct_map = falsification_observed_statistics(panel)
    names = (
        "qhier_vs_qcontext__conditional",
        "qhier_vs_qcontext__joint",
        "qhier_vs_qroute_topology__conditional",
        "qhier_vs_qroute_topology__joint",
    )
    direct = np.asarray([direct_map[name] for name in names], dtype=float)
    replay_error = float(np.max(np.abs(observed - direct)))
    if replay_error > 1e-12:
        raise AssertionError("linear falsification statistic failed direct replay")
    rng = np.random.Generator(np.random.PCG64(20260711))
    null = np.empty((999, 4), dtype=float)
    for draw_zero in range(999):
        transformed = labels.copy()
        draw_number = draw_zero + 1
        for indices in strata:
            original = labels[indices]
            if draw_number % 2 == 1:
                offset = int(rng.integers(1, len(indices)))
                transformed[indices] = np.roll(original, offset, axis=0)
            else:
                while True:
                    permutation = rng.permutation(len(indices))
                    if not np.array_equal(permutation, np.arange(len(indices))):
                        break
                transformed[indices] = original[permutation]
        null[draw_zero] = statistics(transformed)
    p_values = (1 + (null >= observed[None, :]).sum(axis=0)) / 1000.0
    details = {
        name: {
            "observed_relative_log_loss_improvement": float(observed[index]),
            "null_mean": float(null[:, index].mean()),
            "null_q95": float(
                np.quantile(null[:, index], 0.95, method="linear")
            ),
            "p_value": float(p_values[index]),
            "pass": bool(p_values[index] <= 0.01),
        }
        for index, name in enumerate(names)
    }
    return {
        "draws": 999,
        "seed": 20260711,
        "anchors": len(anchors),
        "strata": len(strata),
        "statistics": details,
        "observed_direct_replay_max_error": replay_error,
        "pass": bool(all(item["pass"] for item in details.values())),
    }


def v1_moving_block_bounds(
    values: np.ndarray, seed: int
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return math.nan, math.nan, math.nan
    length = min(5, len(clean))
    blocks = np.asarray(
        [clean[start : start + length] for start in range(len(clean) - length + 1)]
    )
    needed = int(math.ceil(len(clean) / length))
    rng = np.random.default_rng(seed)
    draws = np.empty(5000, dtype=float)
    for index in range(5000):
        chosen = rng.integers(0, len(blocks), size=needed)
        draws[index] = blocks[chosen].reshape(-1)[: len(clean)].mean()
    return (
        float(clean.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def calibration_summary(
    observed: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    minimum_rows: int,
) -> tuple[float, float]:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(probability, dtype=float)
    w = np.asarray(weights, dtype=float)
    bins = np.minimum((p * 10.0).astype(int), 9)
    total = float(w.sum())
    if total <= 0.0:
        return math.nan, math.nan
    ece = 0.0
    supported: list[float] = []
    for bin_index in range(10):
        mask = bins == bin_index
        if not mask.any() or w[mask].sum() <= 0.0:
            continue
        error = abs(
            weighted_mean(p[mask], w[mask]) - weighted_mean(y[mask], w[mask])
        )
        ece += float(w[mask].sum()) / total * error
        if int(mask.sum()) >= minimum_rows:
            supported.append(error)
    return float(ece), float(max(supported)) if supported else math.nan


def v1_base_support(
    cycle: pd.DataFrame, contract: Mapping[str, Any], mode: str
) -> bool:
    if mode == "oof":
        rule = contract["support_gates"][
            "july_december_2024_oof_provisional_tier_each_cycle"
        ]
        required = [f"2024_q{value}" for value in rule["required_quarters"]]
        quarter_minimum_key = "minimum_realized_loop_rows_each_required_quarter"
    elif mode == "scoring":
        rule = contract["support_gates"]["each_full_scoring_period_each_cycle"]
        years = pd.to_datetime(cycle["session_date"], errors="raise").dt.year.unique()
        if len(years) != 1:
            raise AssertionError("V1 scoring support crosses years")
        required = [f"{int(years[0])}_q{value}" for value in range(1, 5)]
        quarter_minimum_key = "minimum_realized_loop_rows_each_quarter"
    else:
        raise ValueError(mode)
    realized = cycle.loc[cycle["loop_occurs"].eq(1)]
    quarter_counts = realized["quarter"].astype(str).value_counts()
    return bool(
        len(realized) >= int(rule["minimum_realized_loop_rows"])
        and realized["symbol_norm"].nunique()
        >= int(rule["minimum_stocks_with_realized_loop"])
        and set(quarter_counts.index) == set(required)
        and all(
            int(quarter_counts.get(quarter, 0))
            >= int(rule[quarter_minimum_key])
            for quarter in required
        )
        and (
            "minimum_compatible_anchor_cycle_rows" not in rule
            or len(cycle) >= int(rule["minimum_compatible_anchor_cycle_rows"])
        )
    )


def v1_structural_reliability(
    cycle: pd.DataFrame,
    contract: Mapping[str, Any],
    minimum_bin_rows: int,
) -> bool:
    observed = cycle["loop_occurs"].to_numpy(dtype=int)
    history_probability = cycle["loop_probability"].to_numpy(dtype=float)
    first_probability = cycle["first_order_probability"].to_numpy(dtype=float)
    history_log = binary_log_loss(observed, history_probability)
    first_log = binary_log_loss(observed, first_probability)
    history_brier = brier_loss(observed, history_probability)
    first_brier = brier_loss(observed, first_probability)
    history_ece, history_max = calibration_summary(
        observed, history_probability, np.ones(len(cycle)), minimum_bin_rows
    )
    first_ece, first_max = calibration_summary(
        observed, first_probability, np.ones(len(cycle)), minimum_bin_rows
    )
    tolerance = float(
        contract["structural_reliability_gate_each_cycle_and_scoring_period"][
            "maximum_supported_bin_error_tolerance"
        ]
    )
    return bool(
        history_log.mean() < first_log.mean()
        and history_brier.mean() < first_brier.mean()
        and history_ece <= first_ece
        and np.isfinite(history_max)
        and np.isfinite(first_max)
        and history_max <= first_max + tolerance
    )


def v1_surface_pass(
    frame: pd.DataFrame,
    observed: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    weights: np.ndarray,
    surface: str,
    minimum_bin_rows: int,
    required_quarters: Sequence[str],
    contract: Mapping[str, Any],
    seed: int,
) -> bool:
    common = contract["common_quality_gates"]
    comparison_rule = common[
        "conditional_candidate_vs_context"
        if surface == "conditional"
        else "joint_chain_candidate_vs_context"
    ]
    tolerance = float(
        common["calibration"][
            "conditional_maximum_supported_bin_error_tolerance"
            if surface == "conditional"
            else "joint_maximum_supported_bin_error_tolerance"
        ]
    )
    relative_improvement = math.nan
    brier_difference_pass = False
    interval_pass = True
    robustness_pass = True
    for loss_index, loss_name in enumerate(("log_loss", "brier")):
        if loss_name == "log_loss":
            candidate = binary_log_loss(observed, candidate_probability)
            baseline = binary_log_loss(observed, baseline_probability)
        else:
            candidate = brier_loss(observed, candidate_probability)
            baseline = brier_loss(observed, baseline_probability)
        difference = candidate - baseline
        mean_difference = weighted_mean(difference, weights)
        baseline_mean = weighted_mean(baseline, weights)
        daily = weighted_group_series(frame, difference, weights, "session_date")
        _, _, upper = v1_moving_block_bounds(daily.to_numpy(float), seed + loss_index)
        interval_pass &= np.isfinite(upper) and upper < 0.0
        quarter = weighted_group_series(frame, difference, weights, "quarter")
        quarter_pass = all(
            value in quarter.index and float(quarter.loc[value]) < 0.0
            for value in required_quarters
        )
        deletion_pass = True
        for symbol in sorted(frame["symbol_norm"].astype(str).unique()):
            keep = frame["symbol_norm"].astype(str).ne(symbol).to_numpy()
            deletion_pass &= (
                keep.any() and weighted_mean(difference[keep], weights[keep]) < 0.0
            )
        robustness_pass &= quarter_pass and deletion_pass
        if loss_name == "log_loss":
            relative_improvement = -mean_difference / baseline_mean
        else:
            brier_difference_pass = mean_difference < 0.0
    baseline_ece, baseline_max = calibration_summary(
        observed, baseline_probability, weights, minimum_bin_rows
    )
    candidate_ece, candidate_max = calibration_summary(
        observed, candidate_probability, weights, minimum_bin_rows
    )
    return bool(
        relative_improvement
        >= float(comparison_rule["minimum_relative_log_loss_improvement"])
        and brier_difference_pass
        and interval_pass
        and robustness_pass
        and candidate_ece <= baseline_ece
        and np.isfinite(candidate_max)
        and np.isfinite(baseline_max)
        and candidate_max <= baseline_max + tolerance
    )


def v1_quality_cell_pass(
    cycle: pd.DataFrame,
    target: str,
    horizon: int,
    tier: str,
    contract: Mapping[str, Any],
    mode: str,
    required_quarters: Sequence[str],
    seed: int,
) -> tuple[bool, float, float]:
    threshold = 1 if tier == "p75" else 2
    quality = cycle[f"quality_class__{target}__h{horizon}"].to_numpy(dtype=int)
    realized_mask = cycle["loop_occurs"].eq(1).to_numpy()
    conditional = cycle.loc[realized_mask].reset_index(drop=True)
    conditional_observed = (quality[realized_mask] >= threshold).astype(int)
    weights = conditional["conditional_weight"].to_numpy(dtype=float)
    context = conditional[
        f"qcontext__{target}__h{horizon}__{tier}"
    ].to_numpy(float)
    candidate = conditional[
        f"qhier__{target}__h{horizon}__{tier}"
    ].to_numpy(float)
    joint_label = "good" if tier == "p75" else "high"
    joint_observed = cycle[
        f"joint_{joint_label}_target__{target}__h{horizon}"
    ].to_numpy(int)
    joint_context = cycle[
        f"joint__qcontext__{target}__h{horizon}__{tier}"
    ].to_numpy(float)
    joint_candidate = cycle[
        f"joint__qhier__{target}__h{horizon}__{tier}"
    ].to_numpy(float)
    calibration_rule = contract["common_quality_gates"]["calibration"]
    if mode == "oof":
        support_rule = contract["support_gates"][
            "july_december_2024_oof_provisional_tier_each_cycle"
        ]
        conditional_bin = int(
            calibration_rule["oof_minimum_supported_conditional_bin_rows"]
        )
        joint_bin = int(calibration_rule["oof_minimum_supported_joint_bin_rows"])
    elif mode == "scoring":
        support_rule = contract["support_gates"][
            "each_full_scoring_period_each_cycle"
        ]
        conditional_bin = int(
            calibration_rule["scoring_minimum_supported_conditional_bin_rows"]
        )
        joint_bin = int(
            calibration_rule["scoring_minimum_supported_joint_bin_rows"]
        )
    else:
        raise ValueError(mode)
    positives = int(conditional_observed.sum())
    negatives = int(len(conditional_observed) - positives)
    if tier == "p75":
        minimum_positive = minimum_negative = int(
            support_rule[
                "good_minimum_p75_positive_and_negative_rows_each_target_horizon"
            ]
        )
    else:
        minimum_positive = int(support_rule["high_minimum_p90_positive_rows_each_target_horizon"])
        minimum_negative = int(support_rule["high_minimum_p90_negative_rows_each_target_horizon"])
    support_pass = positives >= minimum_positive and negatives >= minimum_negative
    conditional_pass = v1_surface_pass(
        conditional,
        conditional_observed,
        context,
        candidate,
        weights,
        "conditional",
        conditional_bin,
        required_quarters,
        contract,
        seed,
    )
    joint_pass = v1_surface_pass(
        cycle,
        joint_observed,
        joint_context,
        joint_candidate,
        np.ones(len(cycle), dtype=float),
        "joint",
        joint_bin,
        required_quarters,
        contract,
        seed + 100,
    )
    observed_rate = weighted_mean(conditional_observed, weights)
    mean_context = weighted_mean(context, weights)
    mean_candidate = weighted_mean(candidate, weights)
    ratio = observed_rate / mean_context if mean_context > 0.0 else math.nan
    residual = conditional_observed - context
    daily_residual = weighted_group_series(
        conditional, residual, weights, "session_date"
    )
    _, residual_low, _ = v1_moving_block_bounds(
        daily_residual.to_numpy(float), seed + 200
    )
    if tier == "p75":
        rule = contract["tier_rules_each_cycle_and_horizon"]["good"]
        rate_pass = bool(
            observed_rate >= float(rule["minimum_observed_conditional_exceedance_rate"])
            and mean_candidate >= float(rule["minimum_mean_calibrated_qcycle_probability"])
            and ratio >= float(rule["minimum_observed_rate_divided_by_mean_qcontext_probability"])
        )
    else:
        rule = contract["tier_rules_each_cycle_and_horizon"]["high"]
        rate_pass = bool(
            observed_rate >= float(rule["p90_minimum_observed_conditional_exceedance_rate"])
            and mean_candidate >= float(rule["p90_minimum_mean_calibrated_qcycle_probability"])
            and ratio >= float(rule["p90_minimum_observed_rate_divided_by_mean_qcontext_probability"])
        )
    passed = bool(
        support_pass
        and conditional_pass
        and joint_pass
        and np.isfinite(residual_low)
        and residual_low > 0.0
        and rate_pass
    )
    return passed, observed_rate, mean_candidate


def independent_v1_horizon_grades(
    panel: pd.DataFrame, mode: str = "oof"
) -> pd.DataFrame:
    contract = json.loads(
        (WORKSPACE / "work/contracts/20260710-per-loop-movement-quality-v1.json").read_text()
    )
    if mode == "oof":
        eligibility_frame = pd.read_csv(V1_PROVISIONAL_SUPPORT_PATH)
        eligibility = eligibility_frame.set_index("cycle_id")[
            "full_2024_fit_eligible"
        ].astype(bool).to_dict()
        required_quarters = ("2024_q3", "2024_q4")
        structural_bin = int(
            contract["common_quality_gates"]["calibration"][
                "oof_minimum_supported_joint_bin_rows"
            ]
        )
    elif mode == "scoring":
        eligibility = {}
        years = pd.to_datetime(panel["session_date"], errors="raise").dt.year.unique()
        if len(years) != 1:
            raise AssertionError("V1 scoring grade frame crosses years")
        required_quarters = tuple(
            f"{int(years[0])}_q{value}" for value in range(1, 5)
        )
        structural_bin = int(
            contract["common_quality_gates"]["calibration"][
                "scoring_minimum_supported_joint_bin_rows"
            ]
        )
    else:
        raise ValueError(mode)
    cycle_indices = pd.read_csv(QUALITY_ROOT / "fixed_cycles.csv").set_index(
        "cycle_id"
    )["cycle_index"].astype(int).to_dict()
    rows: list[dict[str, Any]] = []
    for cycle_number, (cycle_id, cycle) in enumerate(
        panel.groupby("cycle_id", sort=True)
    ):
        cycle = cycle.reset_index(drop=True)
        support = v1_base_support(cycle, contract, mode)
        fit_eligible = (
            bool(eligibility.get(str(cycle_id), False)) if mode == "oof" else True
        )
        structural = v1_structural_reliability(
            cycle, contract, structural_bin
        )
        results: dict[tuple[str, int, str], tuple[bool, float, float]] = {}
        for target_index, target in enumerate(TARGETS):
            for horizon in HORIZONS:
                for tier_index, tier in enumerate(TIERS):
                    results[(target, horizon, tier)] = v1_quality_cell_pass(
                        cycle,
                        target,
                        horizon,
                        tier,
                        contract,
                        mode,
                        required_quarters,
                        20260710
                        + (0 if mode == "oof" else 50000)
                        + cycle_number * 1000
                        + target_index * 200
                        + horizon * 5
                        + tier_index,
                    )
        for horizon in HORIZONS:
            good_cells = [results[(target, horizon, "p75")][0] for target in TARGETS]
            p90_cells = [results[(target, horizon, "p90")][0] for target in TARGETS]
            high_rule = contract["tier_rules_each_cycle_and_horizon"]["high"]
            high_p75_rate = all(
                results[(target, horizon, "p75")][1]
                >= float(high_rule["minimum_p75_observed_conditional_exceedance_rate"])
                and results[(target, horizon, "p75")][2]
                >= float(high_rule["minimum_p75_mean_calibrated_qcycle_probability"])
                for target in TARGETS
            )
            common = bool(
                support
                and fit_eligible
                and (True if mode == "oof" else structural)
            )
            good = bool(common and all(good_cells))
            high = bool(good and high_p75_rate and all(p90_cells))
            grade = (
                "high_movement_quality"
                if high
                else "good_movement_quality"
                if good
                else "unqualified"
            )
            rows.append(
                {
                    "cycle_index": int(cycle_indices[str(cycle_id)]),
                    "cycle_id": str(cycle_id),
                    "horizon": horizon,
                    "grade": grade,
                    "support_pass": support and fit_eligible,
                    "structural_pass": structural,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 60:
        raise AssertionError("independent V1 grades are not sixty horizon rows")
    return result


def named_component_series(
    panel: pd.DataFrame,
    cycle_id: str,
    target: str,
    horizon: int,
    tier: str,
    calendar: Sequence[str],
) -> list[tuple[str, str, np.ndarray]]:
    cycle = panel.loc[panel["cycle_id"].eq(cycle_id)].reset_index(drop=True)
    components: list[tuple[str, str, np.ndarray]] = []
    for surface in ("conditional", "joint"):
        frame, weights = evaluation_surface(cycle, surface)
        observed = observed_binary_target(frame, surface, target, horizon, tier)
        candidate = evaluation_probability(
            frame, "qhier", target, horizon, tier, surface
        )
        context = evaluation_probability(
            frame, "qcontext", target, horizon, tier, surface
        )
        for loss_name in ("log_loss", "brier"):
            difference = (
                binary_log_loss(observed, candidate)
                - binary_log_loss(observed, context)
                if loss_name == "log_loss"
                else brier_loss(observed, candidate) - brier_loss(observed, context)
            )
            daily = weighted_group_series(
                frame, difference, weights, "session_date"
            ).reindex(pd.Index(calendar, dtype=str))
            components.append(
                (
                    f"{target}__{surface}__{loss_name}",
                    "negative",
                    daily.to_numpy(dtype=float),
                )
            )
    frame, weights = evaluation_surface(cycle, "conditional")
    observed = observed_binary_target(frame, "conditional", target, horizon, tier)
    context = evaluation_probability(
        frame, "qcontext", target, horizon, tier, "conditional"
    )
    residual = observed - context
    daily_lift = weighted_group_series(
        frame, residual, weights, "session_date"
    ).reindex(pd.Index(calendar, dtype=str))
    components.append(
        (
            f"{target}__conditional__lift",
            "positive",
            daily_lift.to_numpy(dtype=float),
        )
    )
    return components


def centered_null_p_value(
    values: np.ndarray, alternative: str, positions: np.ndarray
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).any():
        raise AssertionError("named component has no finite observed value")
    observed = float(np.nanmean(array))
    null = resampled_nanmeans(array - observed, positions)[:, 0]
    if alternative == "negative":
        p_value = (1 + int((null <= observed).sum())) / (len(null) + 1)
    elif alternative == "positive":
        p_value = (1 + int((null >= observed).sum())) / (len(null) + 1)
    else:
        raise ValueError(alternative)
    return observed, float(p_value)


def holm_passes(
    frame: pd.DataFrame, p_column: str, alpha: float = 0.025
) -> pd.Series:
    order = frame.sort_values(
        [p_column, "cycle_index", "horizon"], kind="stable"
    ).index
    passed = pd.Series(False, index=frame.index)
    stopped = False
    total = len(frame)
    for rank, index in enumerate(order, start=1):
        if stopped:
            continue
        if float(frame.loc[index, p_column]) <= alpha / (total - rank + 1):
            passed.loc[index] = True
        else:
            stopped = True
    return passed


def independent_named_diagnostics(
    panel: pd.DataFrame,
    rotations: pd.DataFrame,
    positions: np.ndarray,
    calendar: Sequence[str],
    global_primary_pass: bool,
    mapping: pd.DataFrame,
    mode: str = "oof",
) -> pd.DataFrame:
    v1_grades = independent_v1_horizon_grades(panel, mode=mode).set_index(
        ["cycle_id", "horizon"]
    )
    v1_global = aggregate_v1_substitution_grades(
        v1_grades.reset_index().rename(columns={"grade": "v1_substitution_grade"})
    ).set_index("cycle_id")["v1_substitution_global_grade"]
    cycles = pd.read_csv(QUALITY_ROOT / "fixed_cycles.csv").sort_values(
        "cycle_index", kind="stable"
    )
    rows: list[dict[str, Any]] = []
    for cycle in cycles.itertuples(index=False):
        cycle_id = str(cycle.cycle_id)
        required = mapping.loc[mapping["cycle_id"].eq(cycle_id)]
        orientation_pass = True
        for unit in required.itertuples(index=False):
            group_value = f"{cycle_id}@{int(unit.current_state)}"
            unit_rows = rotations.loc[
                rotations["group_type"].eq("cycle_current_state")
                & rotations["group_value"].eq(group_value)
            ]
            if (
                len(unit_rows) != 4
                or not unit_rows["supported"].all()
                or unit_rows["sign_reversal"].any()
            ):
                orientation_pass = False
                break
        for horizon in HORIZONS:
            row: dict[str, Any] = {
                "cycle_index": int(cycle.cycle_index),
                "cycle_id": cycle_id,
                "cycle": str(cycle.cycle),
                "horizon": horizon,
                "global_primary_pass": global_primary_pass,
                "required_orientation_count": len(required),
                "orientation_pass": orientation_pass,
            }
            for tier in TIERS:
                p_values: list[float] = []
                component_payload: dict[str, Any] = {}
                if global_primary_pass:
                    for target in TARGETS:
                        for name, alternative, values in named_component_series(
                            panel, cycle_id, target, horizon, tier, calendar
                        ):
                            observed, p_value = centered_null_p_value(
                                values, alternative, positions
                            )
                            component_payload[name] = {
                                "observed": observed,
                                "alternative": alternative,
                                "p_value": p_value,
                            }
                            p_values.append(p_value)
                    if len(p_values) != 10:
                        raise AssertionError("named unit does not have ten components")
                    row[f"{tier}_unit_p_value"] = max(p_values)
                else:
                    component_payload["not_run"] = (
                        "global_primary_precondition_failed"
                    )
                    row[f"{tier}_unit_p_value"] = 1.0
                row[f"{tier}_component_json"] = json.dumps(
                    json_safe(component_payload),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            grade = str(v1_grades.loc[(cycle_id, horizon), "grade"])
            global_grade = str(v1_global.loc[cycle_id])
            row["v1_substitution_grade"] = grade
            row["v1_substitution_global_grade"] = global_grade
            row["good_point_pass"] = grade in (
                "good_movement_quality",
                "high_movement_quality",
            ) and global_grade in (
                "good_movement_quality",
                "high_movement_quality",
            )
            row["high_point_pass"] = (
                grade == "high_movement_quality"
                and global_grade == "high_movement_quality"
            )
            rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["cycle_index", "horizon"], kind="stable"
    )
    if len(result) != 60:
        raise AssertionError("named diagnostics are not sixty hypotheses")
    if global_primary_pass:
        result["good_holm_pass"] = holm_passes(result, "p75_unit_p_value")
        high_p = result["p90_unit_p_value"].copy()
        good_gate = (
            result["good_point_pass"]
            & result["good_holm_pass"]
            & result["orientation_pass"]
        )
        high_p.loc[~good_gate] = 1.0
        result["high_gatekept_unit_p_value"] = high_p
        result["high_holm_pass"] = holm_passes(
            result, "high_gatekept_unit_p_value"
        )
    else:
        result["good_holm_pass"] = False
        result["high_gatekept_unit_p_value"] = 1.0
        result["high_holm_pass"] = False
    good = (
        result["global_primary_pass"]
        & result["orientation_pass"]
        & result["good_point_pass"]
        & result["good_holm_pass"]
    )
    high = good & result["high_point_pass"] & result["high_holm_pass"]
    result["development_label"] = np.where(
        high,
        "development_high_candidate",
        np.where(good, "development_good_candidate", "development_unqualified"),
    )
    result["parent_grade_changed"] = False
    result["prospective_validated"] = False
    development_global = aggregate_global_cycle_labels(result).set_index("cycle_index")
    result["global_development_label"] = result["cycle_index"].map(
        development_global["global_development_label"]
    )
    return result.reset_index(drop=True)


def aggregate_global_cycle_labels(horizon_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"cycle_index", "cycle_id", "horizon", "development_label"}
    if not required.issubset(horizon_rows.columns):
        raise AssertionError("horizon candidate rows lack global-grade columns")
    rows: list[dict[str, Any]] = []
    for (cycle_index, cycle_id), group in horizon_rows.groupby(
        ["cycle_index", "cycle_id"], sort=True
    ):
        group = group.sort_values("horizon", kind="stable")
        if group["horizon"].tolist() != [6, 12, 24]:
            raise AssertionError("global cycle aggregation lacks three horizons")
        labels = group["development_label"].astype(str).tolist()
        if all(label == "development_high_candidate" for label in labels):
            global_label = "development_high_candidate"
        elif all(
            label
            in ("development_high_candidate", "development_good_candidate")
            for label in labels
        ) and any(label == "development_good_candidate" for label in labels):
            global_label = "development_good_candidate"
        else:
            global_label = "development_unqualified"
        rows.append(
            {
                "cycle_index": int(cycle_index),
                "cycle_id": str(cycle_id),
                "global_development_label": global_label,
            }
        )
    result = pd.DataFrame(rows).sort_values("cycle_index", kind="stable")
    if len(result) != 20:
        raise AssertionError("global cycle aggregation is not twenty cycles")
    return result.reset_index(drop=True)


DEVELOPMENT_LABEL_RANK = {
    "development_unqualified": 0,
    "development_good_candidate": 1,
    "development_high_candidate": 2,
}


def ordered_minimum_development_label(labels: Iterable[str]) -> str:
    values = [str(label) for label in labels]
    if not values or any(value not in DEVELOPMENT_LABEL_RANK for value in values):
        raise AssertionError("unknown or empty development-label transfer set")
    minimum = min(DEVELOPMENT_LABEL_RANK[value] for value in values)
    return next(
        label for label, rank in DEVELOPMENT_LABEL_RANK.items() if rank == minimum
    )


def aggregate_v1_substitution_grades(horizon_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"cycle_index", "cycle_id", "horizon", "v1_substitution_grade"}
    if not required.issubset(horizon_rows.columns):
        raise AssertionError("horizon rows lack V1 substitution grades")
    rows: list[dict[str, Any]] = []
    for (cycle_index, cycle_id), group in horizon_rows.groupby(
        ["cycle_index", "cycle_id"], sort=True
    ):
        group = group.sort_values("horizon", kind="stable")
        if group["horizon"].tolist() != [6, 12, 24]:
            raise AssertionError("V1 global aggregation lacks three horizons")
        grades = group["v1_substitution_grade"].astype(str).tolist()
        if all(grade == "high_movement_quality" for grade in grades):
            global_grade = "high_movement_quality"
        elif all(
            grade in ("high_movement_quality", "good_movement_quality")
            for grade in grades
        ) and any(grade == "good_movement_quality" for grade in grades):
            global_grade = "good_movement_quality"
        else:
            global_grade = "unqualified"
        rows.append(
            {
                "cycle_index": int(cycle_index),
                "cycle_id": str(cycle_id),
                "v1_substitution_global_grade": global_grade,
            }
        )
    result = pd.DataFrame(rows).sort_values("cycle_index", kind="stable")
    if len(result) != 20:
        raise AssertionError("V1 global aggregation is not twenty cycles")
    return result.reset_index(drop=True)


def verify_repeated_global_cycle_columns(
    stored: pd.DataFrame,
) -> tuple[bool, dict[str, Any]]:
    expected_development = aggregate_global_cycle_labels(stored)
    expected_v1 = aggregate_v1_substitution_grades(stored)
    expected = expected_development.merge(
        expected_v1, on=["cycle_index", "cycle_id"], validate="one_to_one"
    )
    observed = (
        stored[
            [
                "cycle_index",
                "cycle_id",
                "global_development_label",
                "v1_substitution_global_grade",
            ]
        ]
        .drop_duplicates()
        .sort_values("cycle_index", kind="stable")
        .reset_index(drop=True)
    )
    expected = expected.sort_values("cycle_index", kind="stable").reset_index(drop=True)
    passed = (
        len(observed) == 20
        and np.array_equal(observed["cycle_index"], expected["cycle_index"])
        and np.array_equal(observed["cycle_id"].astype(str), expected["cycle_id"])
        and np.array_equal(
            observed["global_development_label"].astype(str),
            expected["global_development_label"].astype(str),
        )
        and np.array_equal(
            observed["v1_substitution_global_grade"].astype(str),
            expected["v1_substitution_global_grade"].astype(str),
        )
    )
    return passed, {
        "observed": observed.to_dict(orient="records"),
        "expected": expected.to_dict(orient="records"),
    }


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if len(value) != len(weight) or weight.sum() <= 0.0:
        raise AssertionError("invalid weighted quantile input")
    order = np.argsort(value, kind="stable")
    value = value[order]
    weight = weight[order]
    locations = (np.cumsum(weight) - 0.5 * weight) / weight.sum()
    return np.interp(np.asarray(quantiles, dtype=float), locations, value)


def entropy_cutpoints(oof: pd.DataFrame) -> np.ndarray:
    realized = oof.loc[oof["loop_occurs"].eq(1)]
    return weighted_quantiles(
        realized["next_state_entropy_normalized"].to_numpy(dtype=float),
        realized["conditional_weight"].to_numpy(dtype=float),
        (0.25, 0.5, 0.75),
    )


def selection_loss_cells(
    validation: pd.DataFrame,
    predictions: Mapping[tuple[str, int, str], np.ndarray],
) -> tuple[dict[str, float], dict[str, float], float]:
    weights = validation["conditional_weight"].to_numpy(dtype=float)
    losses: dict[str, float] = {}
    cell_weights: dict[str, float] = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            ordered = validation[
                f"quality_class__{target}__h{horizon}"
            ].to_numpy(dtype=int)
            for tier in TIERS:
                observed = ordered >= (1 if tier == "p75" else 2)
                key = f"{target}__h{horizon}__{tier}"
                losses[key] = weighted_mean(
                    binary_log_loss(observed, predictions[(target, horizon, tier)]),
                    weights,
                )
                cell_weights[key] = float(weights.sum())
    objective = float(np.mean(list(losses.values())))
    return losses, cell_weights, objective


def select_scale_pair(
    objectives: Mapping[int, float], tolerance: float = 1e-6
) -> tuple[int, list[int]]:
    if set(objectives) != set(range(len(SCALE_GRID))):
        raise AssertionError("selection objective grid is incomplete")
    minimum = min(float(value) for value in objectives.values())
    tie = [
        index
        for index, value in objectives.items()
        if float(value) <= minimum + float(tolerance)
    ]
    selected = min(
        tie,
        key=lambda index: (
            SCALE_GRID[index][1],
            SCALE_GRID[index][0],
            index,
        ),
    )
    return selected, sorted(tie)


def validate_schedule() -> None:
    if tuple(INNER_SCHEDULE) != OUTER_MONTHS:
        raise AssertionError("outer schedule order changed")
    for outer, inner_months in INNER_SCHEDULE.items():
        if len(inner_months) != 3 or any(month >= outer for month in inner_months):
            raise AssertionError("non-causal inner schedule")


def verify_runner_separation(audit: Audit) -> dict[str, Any]:
    if not RUNNER_PATH.is_file():
        audit.check("production_runner_not_yet_present", True)
        return {"present": False}
    source = RUNNER_PATH.read_text()
    tree = ast.parse(source)
    imports_audit = any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "audit_hierarchical_loop_quality_algorithm_v1"
        in ast.get_source_segment(source, node)
        for node in ast.walk(tree)
    )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    audit.check("production_runner_does_not_import_independent_audit", not imports_audit)
    audit.check(
        "production_runner_has_separate_fit_and_scoring_entrypoints",
        "run_fit_only" in functions and "run_scoring" in functions,
        sorted(functions),
    )
    return {"present": True, "sha256": sha256(RUNNER_PATH)}


def preartifact_audit() -> dict[str, Any]:
    audit = Audit()
    contract = verify_contract_semantics(audit)
    pins = verify_source_pins(audit, contract)
    mapping, vectors = verify_rotation_mapping(audit)
    validate_schedule()
    oof = load_oof_frame(mapping, vectors)
    verify_topology_replay(audit, oof)
    support = verify_unique_oof_support(audit, oof)
    training = load_training_frame(mapping, vectors)
    audit.check(
        "fit_only_cohorts_are_2024_and_expected_size",
        len(training) == 32677
        and len(oof) == 216438
        and set(training["month_key"].str[:4]) == {"2024"}
        and set(oof["month_key"].str[:4]) == {"2024"},
    )
    runner = verify_runner_separation(audit)
    return {
        "phase": "hierarchical_v1_independent_preartifact",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "contract_sha256": CONTRACT_SHA256,
        "source_hashes": pins,
        "support": support,
        "route_units": len(mapping),
        "topology_width": vectors.shape[1],
        "runner": runner,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scoring_authorized": False,
        "later_period_outcomes_opened_by_audit": False,
        "prospective_validated": False,
        "parent_grade_changed": False,
    }


def fit_artifact_paths(root: Path) -> dict[str, Path]:
    return {name: root / name for name in FIT_ARTIFACTS}


def verify_fit_bundle_lock(audit: Audit, root: Path) -> dict[str, Any]:
    paths = fit_artifact_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    marker_path = root / "fit_complete.json"
    if not marker_path.is_file():
        missing.append("fit_complete.json")
    audit.check("fit_bundle_artifacts_present", not missing, sorted(set(missing)))
    if missing:
        return {}
    marker = json.loads(marker_path.read_text())
    stored_hashes = marker.get("artifact_hashes", {})
    actual_hashes = {name: sha256(path) for name, path in paths.items()}
    checks = {
        "status": marker.get("status")
        == "fit_frozen_pending_independent_pre_score_audit",
        "contract": marker.get("contract_sha256") == CONTRACT_SHA256,
        "research_only": marker.get("research_only") is True,
        "live_disabled": marker.get("live_ordering_enabled") is False,
        "orders_disabled": marker.get("order_placement") == "disabled",
        "later_closed": marker.get("later_period_panels_read") is False
        and marker.get("later_scoring_authorized") is False
        and marker.get("scoring_authorized") is False,
        "shadows_untouched": marker.get("shadow_tree_read") is False
        and marker.get("shadow_tree_written") is False,
        "support": marker.get("support_pass") is True,
        "parent_grade": marker.get("parent_grade_changed") is False,
        "cohort": marker.get("oof_compatible_rows") == 216438
        and marker.get("oof_realized_rows") == 15584
        and abs(float(marker.get("oof_effective_weight", math.nan)) - 14167.0)
        <= 1e-12,
        "zero_chain": float(
            marker.get("zero_pair_chain_replay_max_error", math.inf)
        )
        <= 1e-12,
        "probability_chain": bool(
            marker.get("probability_chain_replay_max_error", {})
        )
        and all(
            float(value) <= 1e-12
            for value in marker.get("probability_chain_replay_max_error", {}).values()
        ),
    }
    audit.check("fit_complete_lock_semantics_exact", all(checks.values()), checks)
    audit.check(
        "fit_artifact_hashes_exact",
        stored_hashes == actual_hashes,
        {"stored": stored_hashes, "actual": actual_hashes},
    )
    fit_sources = json.loads((root / "fit_source_hashes.json").read_text())
    supplementary = {
        "per_loop_runner.py": V1_RUNNER_SHA256,
        "provisional_support_2024.csv": V1_PROVISIONAL_SUPPORT_SHA256,
    }
    audit.check(
        "fit_bundle_records_supplementary_v1_dependency_pins",
        all(fit_sources.get(name) == digest for name, digest in supplementary.items()),
        {
            "expected": supplementary,
            "stored": {name: fit_sources.get(name) for name in supplementary},
        },
    )
    return marker


def _frame_maximum_error(
    stored: pd.DataFrame,
    expected: pd.DataFrame,
    keys: Sequence[str],
    tolerance: float = 1e-12,
) -> tuple[bool, dict[str, Any]]:
    left = stored.sort_values(list(keys), kind="stable").reset_index(drop=True)
    right = expected.sort_values(list(keys), kind="stable").reset_index(drop=True)
    if len(left) != len(right) or set(left.columns) != set(right.columns):
        return False, {
            "stored_rows": len(left),
            "expected_rows": len(right),
            "stored_only": sorted(set(left.columns) - set(right.columns)),
            "expected_only": sorted(set(right.columns) - set(left.columns)),
        }
    maximum = 0.0
    categorical: list[str] = []
    for column in right.columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            actual = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
            wanted = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
            if not np.array_equal(np.isnan(actual), np.isnan(wanted)):
                categorical.append(column)
                continue
            if (
                np.isinf(actual).any()
                or np.isinf(wanted).any()
                or not np.array_equal(np.isfinite(actual), np.isfinite(wanted))
            ):
                categorical.append(column)
                continue
            finite = np.isfinite(actual) & np.isfinite(wanted)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(actual[finite] - wanted[finite]))))
        elif not np.array_equal(left[column].astype(str), right[column].astype(str)):
            categorical.append(column)
    return maximum <= tolerance and not categorical, {
        "maximum_numeric_error": maximum,
        "categorical_mismatches": categorical,
    }


def nested_differences(
    actual: Any,
    expected: Any,
    *,
    tolerance: float = 1e-12,
    path: str = "$",
) -> list[dict[str, Any]]:
    left = json_safe(actual)
    right = json_safe(expected)
    differences: list[dict[str, Any]] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append(
                {
                    "path": path,
                    "actual_keys_only": sorted(set(left) - set(right)),
                    "expected_keys_only": sorted(set(right) - set(left)),
                }
            )
        for key in sorted(set(left) & set(right)):
            differences.extend(
                nested_differences(
                    left[key],
                    right[key],
                    tolerance=tolerance,
                    path=f"{path}.{key}",
                )
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": path, "actual_length": len(left), "expected_length": len(right)}]
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            differences.extend(
                nested_differences(
                    left_item,
                    right_item,
                    tolerance=tolerance,
                    path=f"{path}[{index}]",
                )
            )
        return differences
    if isinstance(left, bool) or isinstance(right, bool):
        if not (
            isinstance(left, bool)
            and isinstance(right, bool)
            and left is right
        ):
            differences.append({"path": path, "actual": left, "expected": right})
        return differences
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        if abs(float(left) - float(right)) > tolerance:
            differences.append({"path": path, "actual": left, "expected": right})
        return differences
    if left != right:
        differences.append({"path": path, "actual": left, "expected": right})
    return differences


def replay_inner_candidates(
    training: pd.DataFrame,
    mapping: pd.DataFrame,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Reconstruct every unique April-November validation-month/grid candidate.

    This is intentionally expensive and is called only by ``--pre-score-only``
    after the production fit bundle is frozen.  The zero endpoint for July and
    later is replaced by the pinned V3 probability in the comparison layer;
    April-June is independently refitted with the exact V3 settings.
    """

    cache: dict[tuple[str, int], dict[str, Any]] = {}
    validation_months = sorted(
        {month for months in INNER_SCHEDULE.values() for month in months}
        | set(FULL_SELECTION_MONTHS)
    )
    v3_columns = ["anchor_id", "cycle_index", "loop_occurs"] + [
        f"qroute_topology__{target}__h{horizon}__{tier}"
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    v3_oof = pd.read_parquet(V3_ROOT / "oof_predictions_2024.parquet", columns=v3_columns)
    for month in validation_months:
        train = training.loc[training["month_key"].lt(month)].reset_index(drop=True)
        validation = training.loc[training["month_key"].eq(month)].reset_index(drop=True)
        if train.empty or validation.empty:
            raise AssertionError(f"empty independent inner fold {month}")
        for grid_index, pair in enumerate(SCALE_GRID):
            if pair == (0.0, 0.0) and month >= "2024-07":
                validation_keys = validation[["anchor_id", "cycle_index"]].copy()
                validation_keys["__validation_order"] = np.arange(
                    len(validation_keys), dtype=int
                )
                subset = v3_oof.loc[v3_oof["loop_occurs"].eq(1)].merge(
                    validation_keys,
                    on=["anchor_id", "cycle_index"],
                    how="right",
                    sort=False,
                    validate="one_to_one",
                ).sort_values("__validation_order", kind="stable")
                if subset.filter(like="qroute_topology__").isna().any().any():
                    raise AssertionError("sealed V3 zero endpoint merge failed")
                predictions = {
                    (target, horizon, tier): subset[
                        f"qroute_topology__{target}__h{horizon}__{tier}"
                    ].to_numpy(dtype=float)
                    for target in TARGETS
                    for horizon in HORIZONS
                    for tier in TIERS
                }
                metadata: dict[str, Any] = {"zero_replay": True}
            else:
                predictions, metadata = fit_pair_predictions(
                    train, validation, pair, mapping
                )
            losses, weights, objective = selection_loss_cells(validation, predictions)
            cache[(month, grid_index)] = {
                "validation": validation,
                "predictions": predictions,
                "losses": losses,
                "weights": weights,
                "objective": objective,
                "metadata": metadata,
            }
    return cache


def combine_selection_scope(
    scope: str,
    months: Sequence[str],
    cache: Mapping[tuple[str, int], Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    objectives: dict[int, float] = {}
    pending: list[dict[str, Any]] = []
    for grid_index, pair in enumerate(SCALE_GRID):
        combined_losses: dict[str, float] = {}
        combined_weights: dict[str, float] = {}
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    cell = f"{target}__h{horizon}__{tier}"
                    numerator = sum(
                        float(cache[(month, grid_index)]["losses"][cell])
                        * float(cache[(month, grid_index)]["weights"][cell])
                        for month in months
                    )
                    denominator = sum(
                        float(cache[(month, grid_index)]["weights"][cell])
                        for month in months
                    )
                    if denominator <= 0.0:
                        raise AssertionError("selection cell has zero combined weight")
                    combined_losses[cell] = numerator / denominator
                    combined_weights[cell] = denominator
        objective = float(np.mean(list(combined_losses.values())))
        objectives[grid_index] = objective
        row: dict[str, Any] = {
            "selection_scope": scope,
            "validation_months_json": json.dumps(list(months), separators=(",", ":")),
            "grid_index": grid_index,
            "a_cycle": pair[0],
            "a_route": pair[1],
            "selection_objective": objective,
        }
        for cell, value in combined_losses.items():
            row[f"loss__{cell}"] = value
            row[f"weight__{cell}"] = combined_weights[cell]
        pending.append(row)
    selected, tie = select_scale_pair(objectives)
    minimum = min(objectives.values())
    for row in pending:
        row["objective_minimum"] = minimum
        row["in_tie_set"] = int(row["grid_index"]) in tie
        row["selected"] = int(row["grid_index"]) == selected
        rows.append(row)
    return pd.DataFrame(rows)


def reconstruct_inner_selection(
    training: pd.DataFrame, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    cache = replay_inner_candidates(training, mapping)
    frames = [
        combine_selection_scope(f"outer:{outer}", months, cache)
        for outer, months in INNER_SCHEDULE.items()
    ]
    frames.append(combine_selection_scope("full_2024", FULL_SELECTION_MONTHS, cache))
    return pd.concat(frames, ignore_index=True), cache


def reconstruct_candidate_fit_audit(
    training: pd.DataFrame,
    mapping: pd.DataFrame,
    cache: Mapping[tuple[str, int], Mapping[str, Any]],
) -> pd.DataFrame:
    months = sorted({month for month, _ in cache})
    rows: list[dict[str, Any]] = []
    for month in months:
        train = training.loc[training["month_key"].lt(month)].reset_index(drop=True)
        validation = training.loc[training["month_key"].eq(month)].reset_index(drop=True)
        weights = train["conditional_weight"].to_numpy(dtype=float)
        mu_cycle, mu_route, _ = weighted_hierarchy_centers(train, weights, mapping)
        for grid_index, pair in enumerate(SCALE_GRID):
            metadata = cache[(month, grid_index)]["metadata"]
            models = metadata.get("models", {})
            maximum_n_iter = max(
                (int(model.n_iter_[0]) for model in models.values()), default=0
            )
            if pair == (0.0, 0.0) and month >= "2024-07":
                zero_source = "sealed_v3_oof"
            elif pair == (0.0, 0.0):
                zero_source = "causal_v3_regeneration"
            else:
                zero_source = "not_zero"
            rows.append(
                {
                    "validation_month": month,
                    "grid_index": grid_index,
                    "a_cycle": pair[0],
                    "a_route": pair[1],
                    "training_month_max": str(train["month_key"].max()),
                    "training_rows": len(train),
                    "training_weight": float(weights.sum()),
                    "validation_rows": len(validation),
                    "feature_width": 80 if pair == (0.0, 0.0) else 144,
                    "zero_source": zero_source,
                    "maximum_n_iter": maximum_n_iter,
                    "mu_cycle_hash": hashlib.sha256(mu_cycle.tobytes()).hexdigest(),
                    "mu_route_hash": hashlib.sha256(mu_route.tobytes()).hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def reconstruct_outer_oof(
    training: pd.DataFrame,
    oof: pd.DataFrame,
    mapping: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = oof.copy()
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                output[f"qhier__{target}__h{horizon}__{tier}"] = np.nan
    audit_rows: list[dict[str, Any]] = []
    v3_columns = ["anchor_id", "cycle_index"] + [
        f"qroute_topology__{target}__h{horizon}__{tier}"
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    v3_oof = pd.read_parquet(V3_ROOT / "oof_predictions_2024.parquet", columns=v3_columns)
    for fold_index, month in enumerate(OUTER_MONTHS, start=1):
        selected_row = selection.loc[
            selection["selection_scope"].eq(f"outer:{month}")
            & selection["selected"].astype(bool)
        ]
        if len(selected_row) != 1:
            raise AssertionError("outer scope does not have exactly one selected pair")
        selected_grid = int(selected_row.iloc[0]["grid_index"])
        pair = SCALE_GRID[selected_grid]
        train = training.loc[training["month_key"].lt(month)].reset_index(drop=True)
        positions = np.flatnonzero(output["month_key"].eq(month).to_numpy())
        validation = output.iloc[positions].reset_index(drop=True)
        if pair == (0.0, 0.0):
            merged = validation[["anchor_id", "cycle_index"]].merge(
                v3_oof,
                on=["anchor_id", "cycle_index"],
                how="left",
                validate="one_to_one",
                sort=False,
            )
            predictions = {
                (target, horizon, tier): merged[
                    f"qroute_topology__{target}__h{horizon}__{tier}"
                ].to_numpy(dtype=float)
                for target in TARGETS
                for horizon in HORIZONS
                for tier in TIERS
            }
            metadata = {"feature_width": 80, "models": {}}
        else:
            predictions, metadata = fit_pair_predictions(train, validation, pair, mapping)
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    output.loc[
                        positions, f"qhier__{target}__h{horizon}__{tier}"
                    ] = predictions[(target, horizon, tier)]
                model = metadata.get("models", {}).get(task_key(target, horizon))
                audit_rows.append(
                    {
                        "fold_index": fold_index,
                        "validation_month": month,
                        "inner_months_json": json.dumps(
                            list(INNER_SCHEDULE[month]), separators=(",", ":")
                        ),
                        "selected_grid_index": selected_grid,
                        "a_cycle": pair[0],
                        "a_route": pair[1],
                        "training_month_max": str(train["month_key"].max()),
                        "training_rows": len(train),
                        "training_weight": float(train["conditional_weight"].sum()),
                        "validation_compatible_rows": len(validation),
                        "validation_realized_rows": int(validation["loop_occurs"].sum()),
                        "target": target,
                        "horizon": horizon,
                        "feature_width": int(metadata["feature_width"]),
                        "n_iter": 0 if model is None else int(model.n_iter_[0]),
                        "zero_fallback": pair == (0.0, 0.0),
                        "max_zero_replay_error": (
                            0.0 if pair == (0.0, 0.0) else math.nan
                        ),
                    }
                )
    probability_columns = [
        f"qhier__{target}__h{horizon}__{tier}"
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    if not np.isfinite(output[probability_columns].to_numpy(dtype=float)).all():
        raise AssertionError("outer OOF reconstruction left non-finite probability")
    return output, pd.DataFrame(audit_rows)


def reconstruct_full_parameters(
    training: pd.DataFrame,
    mapping: pd.DataFrame,
    selection: pd.DataFrame,
    entropy_cuts: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = selection.loc[
        selection["selection_scope"].eq("full_2024")
        & selection["selected"].astype(bool)
    ]
    if len(selected) != 1:
        raise AssertionError("full selection is not unique")
    grid_index = int(selected.iloc[0]["grid_index"])
    pair = SCALE_GRID[grid_index]
    parameters: dict[str, np.ndarray] = {
        "selected_grid_index": np.asarray([grid_index], dtype=int),
        "selected_a_cycle": np.asarray([pair[0]], dtype=float),
        "selected_a_route": np.asarray([pair[1]], dtype=float),
        "entropy_quartile_cutpoints": np.asarray(entropy_cuts, dtype=float),
    }
    weights = training["conditional_weight"].to_numpy(dtype=float)
    raw = raw_context(training, numeric_medians())
    scaler = fit_context_scaler(raw, weights)
    mu_cycle, mu_route, route_cycle = weighted_hierarchy_centers(
        training, weights, mapping
    )
    parameters.update(
        {
            "context_scaler_scale": np.asarray(scaler.scale_, dtype=float),
            "context_scaler_mean": np.asarray(scaler.mean_, dtype=float),
            "context_scaler_var": np.asarray(scaler.var_, dtype=float),
            "context_numeric_medians": np.asarray(
                [numeric_medians()[name] for name in NUMERIC_CONTROLS], dtype=float
            ),
            "mu_cycle": np.asarray(mu_cycle, dtype=float),
            "mu_route_within_cycle": np.asarray(mu_route, dtype=float),
            "route_cycle": np.asarray(route_cycle, dtype=int),
        }
    )
    base_audit: dict[str, Any] = {
        "selected_grid_index": grid_index,
        "selected_pair": list(pair),
        "training_rows": len(training),
        "training_weight": float(weights.sum()),
        "zero_fallback": pair == (0.0, 0.0),
        "models": {},
    }
    if pair == (0.0, 0.0):
        base_audit["sealed_v3_model_parameters_sha256"] = sha256(
            V3_ROOT / "model_parameters.npz"
        )
        base_audit["new_qhier_coefficients_stored"] = False
        return parameters, base_audit
    predictions, metadata = fit_pair_predictions(training, training, pair, mapping)
    del predictions
    model_rows: dict[str, Any] = {}
    for key, model in metadata["models"].items():
        parameters[f"{key}__classes"] = np.asarray(model.classes_, dtype=int)
        parameters[f"{key}__coef"] = np.asarray(model.coef_, dtype=float)
        parameters[f"{key}__intercept"] = np.asarray(model.intercept_, dtype=float)
        parameters[f"{key}__n_iter"] = np.asarray(model.n_iter_, dtype=int)
        parameters[f"{key}__temperature"] = np.asarray([1.0], dtype=float)
        model_rows[key] = {
            "feature_width": int(model.coef_.shape[1]),
            "n_iter": int(model.n_iter_[0]),
        }
    base_audit["models"] = model_rows
    base_audit["new_qhier_coefficients_stored"] = True
    return parameters, base_audit


def compare_npz(
    stored_path: Path, expected: Mapping[str, np.ndarray], tolerance: float = 1e-12
) -> tuple[bool, dict[str, Any]]:
    with np.load(stored_path) as stored:
        stored_keys = set(stored.files)
        expected_keys = set(expected)
        maximum = 0.0
        shape_errors: list[str] = []
        for key in sorted(stored_keys & expected_keys):
            actual = np.asarray(stored[key])
            wanted = np.asarray(expected[key])
            if actual.shape != wanted.shape:
                shape_errors.append(key)
                continue
            if actual.size:
                maximum = max(
                    maximum,
                    float(np.max(np.abs(actual.astype(float) - wanted.astype(float)))),
                )
    passed = (
        stored_keys == expected_keys and not shape_errors and maximum <= tolerance
    )
    return passed, {
        "stored_only": sorted(stored_keys - expected_keys),
        "expected_only": sorted(expected_keys - stored_keys),
        "shape_errors": shape_errors,
        "maximum_error": maximum,
    }


def pre_score_audit(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    audit = Audit()
    contract = verify_contract_semantics(audit)
    source_hashes = verify_source_pins(audit, contract)
    mapping, vectors = verify_rotation_mapping(audit)
    marker = verify_fit_bundle_lock(audit, root)
    if not marker or not audit.all_passed:
        return {
            "phase": "hierarchical_v1_independent_pre_score",
            "all_passed": False,
            "check_count": len(audit.checks),
            "checks": audit.checks,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "scoring_authorized": False,
            "later_period_outcomes_opened_by_audit": False,
        }
    training = load_training_frame(mapping, vectors)
    oof = load_oof_frame(mapping, vectors)
    expected_selection, candidate_cache = reconstruct_inner_selection(training, mapping)
    stored_selection = pd.read_csv(root / "inner_selection_2024.csv")
    passed, details = _frame_maximum_error(
        stored_selection,
        expected_selection,
        ["selection_scope", "grid_index"],
    )
    audit.check("all_inner_and_full_scale_selections_independently_exact", passed, details)
    expected_candidate_audit = reconstruct_candidate_fit_audit(
        training, mapping, candidate_cache
    )
    stored_candidate_audit = pd.read_csv(root / "candidate_fit_audit_2024.csv")
    passed, details = _frame_maximum_error(
        stored_candidate_audit,
        expected_candidate_audit,
        ["validation_month", "grid_index"],
    )
    audit.check("all_candidate_fit_audits_independently_exact", passed, details)
    selected_full = expected_selection.loc[
        expected_selection["selection_scope"].eq("full_2024")
        & expected_selection["selected"].astype(bool)
    ].iloc[0]
    selected_pair = (
        float(selected_full["a_cycle"]),
        float(selected_full["a_route"]),
    )
    audit.check(
        "fit_marker_selected_pair_matches_independent_selection",
        marker.get("selected_full_grid_index") == int(selected_full["grid_index"])
        and marker.get("selected_full_pair") == list(selected_pair),
        {
            "marker_grid": marker.get("selected_full_grid_index"),
            "independent_grid": int(selected_full["grid_index"]),
        },
    )
    expected_json_artifacts = {
        "fold_schedule.json": {
            "outer_months": list(OUTER_MONTHS),
            "inner_validation_schedule": INNER_SCHEDULE,
            "final_selection_months": list(FULL_SELECTION_MONTHS),
            "strict_training_rule": "month_key < validation_month",
        },
        "hyperparameter_grid.json": {
            "ordered_pairs": [list(pair) for pair in SCALE_GRID],
            "pair_order": ["a_cycle", "a_route"],
            "tie_tolerance": 1e-6,
            "tie_break": ["a_route", "a_cycle"],
            "shared_across_all_six_tasks": True,
            "model_C": 0.2,
            "model_max_iter": 2000,
            "model_tol": 1e-10,
            "seed": 20260711,
        },
        "feature_manifest.json": expected_feature_manifest(selected_pair),
        "support_2024.json": independent_support_payload(oof),
    }
    for filename, expected_payload in expected_json_artifacts.items():
        saved_payload = json.loads((root / filename).read_text())
        differences = nested_differences(saved_payload, expected_payload)
        audit.check(
            f"{filename.replace('.', '_')}_independently_exact",
            not differences,
            differences[:20],
        )
    stored_mapping = pd.read_csv(root / "route_mapping.csv").sort_values(
        "route_index", kind="stable"
    )
    mapping_keys_exact = (
        len(stored_mapping) == 44
        and np.array_equal(
            stored_mapping["route_index"].to_numpy(dtype=int), np.arange(44)
        )
        and np.array_equal(
            stored_mapping["cycle_index"].to_numpy(dtype=int),
            mapping["cycle_index"].to_numpy(dtype=int),
        )
        and np.array_equal(
            stored_mapping["current_state"].to_numpy(dtype=int),
            mapping["current_state"].to_numpy(dtype=int),
        )
        and np.array_equal(
            stored_mapping["cycle_id"].astype(str), mapping["cycle_id"].astype(str)
        )
        and np.array_equal(
            stored_mapping["compatible_rotation_count"].to_numpy(dtype=int),
            mapping["compatible_rotation_count"].to_numpy(dtype=int),
        )
    )
    audit.check("fit_route_mapping_independently_exact", mapping_keys_exact)
    expected_fit_sources = {
        **source_hashes,
        "hierarchical_contract_sha256": CONTRACT_SHA256,
        "hierarchical_runner_sha256": sha256(RUNNER_PATH),
    }
    for filename in ("fit_source_hashes_pre_fit.json", "fit_source_hashes.json"):
        saved_sources = json.loads((root / filename).read_text())
        audit.check(
            f"{filename.replace('.', '_')}_independently_exact",
            saved_sources == expected_fit_sources,
            {
                "stored_only": sorted(set(saved_sources) - set(expected_fit_sources)),
                "expected_only": sorted(set(expected_fit_sources) - set(saved_sources)),
            },
        )
    independent_oof, expected_folds = reconstruct_outer_oof(
        training, oof, mapping, expected_selection
    )
    stored_oof = pd.read_parquet(root / "oof_predictions_2024.parquet")
    keys = ["anchor_id", "cycle_index"]
    probability_columns = [
        f"qhier__{target}__h{horizon}__{tier}"
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    left = stored_oof.sort_values(keys, kind="stable").reset_index(drop=True)
    right = independent_oof.sort_values(keys, kind="stable").reset_index(drop=True)
    ids_exact = (
        len(left) == len(right)
        and np.array_equal(left["anchor_id"], right["anchor_id"])
        and np.array_equal(left["cycle_index"], right["cycle_index"])
    )
    maximum_oof_error = math.inf
    if ids_exact:
        maximum_oof_error = float(
            np.max(
                np.abs(
                    left[probability_columns].to_numpy(dtype=float)
                    - right[probability_columns].to_numpy(dtype=float)
                ),
                initial=0.0,
            )
        )
    audit.check(
        "all_causal_outer_oof_predictions_independently_exact",
        ids_exact and maximum_oof_error <= 1e-12,
        {"ids_exact": ids_exact, "maximum_error": maximum_oof_error},
    )
    panel = prepare_evaluation_panel(independent_oof, mapping)
    (
        expected_metrics,
        expected_calibration,
        expected_rotations,
        expected_gates,
        block_positions,
        calendar,
    ) = independent_core_evaluation(panel, "2024_oof", "oof")
    expected_falsification = independent_falsification(panel)
    expected_global_pass = bool(
        expected_gates["support_pass"]
        and expected_gates["primary_without_falsification_pass"]
        and expected_falsification["pass"]
    )
    expected_gates["falsification"] = expected_falsification
    expected_gates["primary_algorithm_pass"] = expected_global_pass
    expected_gates["primary_algorithm_label"] = (
        "development_algorithm_supported"
        if expected_global_pass
        else "development_algorithm_unconfirmed"
    )
    expected_grades = independent_named_diagnostics(
        panel,
        expected_rotations,
        block_positions,
        calendar,
        expected_global_pass,
        mapping,
    )
    frame_artifacts = (
        (
            "cell_diagnostics_2024.csv",
            expected_metrics,
            ["surface", "model", "target", "horizon", "tier"],
            "all_cell_metrics_independently_exact",
        ),
        (
            "calibration_diagnostics_2024.csv",
            expected_calibration,
            ["surface", "model", "target", "horizon", "tier", "bin"],
            "all_fixed_bin_calibration_independently_exact",
        ),
        (
            "rotation_diagnostics_2024.csv",
            expected_rotations,
            ["group_type", "group_value", "surface", "baseline"],
            "all_supported_slice_diagnostics_independently_exact",
        ),
        (
            "per_loop_grades_2024.csv",
            expected_grades,
            ["cycle_index", "horizon"],
            "v1_substitution_holm_and_named_labels_independently_exact",
        ),
    )
    for filename, expected_frame, frame_keys, check_name in frame_artifacts:
        stored_frame = pd.read_csv(root / filename)
        frame_pass, frame_details = _frame_maximum_error(
            stored_frame, expected_frame, frame_keys
        )
        audit.check(check_name, frame_pass, frame_details)
    saved_falsification = json.loads(
        (root / "falsification_diagnostics_2024.json").read_text()
    )
    falsification_differences = nested_differences(
        saved_falsification, expected_falsification
    )
    audit.check(
        "all_999_falsification_draws_and_gates_independently_exact",
        not falsification_differences,
        falsification_differences[:20],
    )
    saved_gates = json.loads((root / "algorithm_gates_2024.json").read_text())
    gate_differences = nested_differences(saved_gates, expected_gates)
    audit.check(
        "all_primary_secondary_bootstrap_calibration_slice_gates_independently_exact",
        not gate_differences,
        gate_differences[:20],
    )
    stored_folds = pd.read_csv(root / "outer_fold_audit_2024.csv")
    passed, details = _frame_maximum_error(
        stored_folds,
        expected_folds,
        ["validation_month", "target", "horizon"],
    )
    audit.check("outer_fold_audit_independently_exact", passed, details)
    expected_parameters, expected_full_audit = reconstruct_full_parameters(
        training, mapping, expected_selection, entropy_cutpoints(oof)
    )
    passed, details = compare_npz(root / "model_parameters.npz", expected_parameters)
    audit.check("full_model_parameters_independently_exact", passed, details)
    saved_full_audit = json.loads((root / "full_fit_audit.json").read_text())
    audit.check(
        "full_fit_audit_independently_exact",
        json_safe(saved_full_audit) == json_safe(expected_full_audit),
        {"stored": saved_full_audit, "expected": expected_full_audit},
    )
    stored_grades = pd.read_csv(root / "per_loop_grades_2024.csv")
    global_pass, global_details = verify_repeated_global_cycle_columns(stored_grades)
    audit.check(
        "global_cycle_aggregation_independently_exact",
        global_pass,
        global_details,
    )
    selected_full_row = expected_selection.loc[
        expected_selection["selection_scope"].eq("full_2024")
        & expected_selection["selected"].astype(bool)
    ].iloc[0]
    expected_provisional = {
        "primary_algorithm_label": expected_gates["primary_algorithm_label"],
        "primary_algorithm_pass": expected_global_pass,
        "selected_full_grid_index": int(selected_full_row["grid_index"]),
        "selected_full_pair": [
            float(selected_full_row["a_cycle"]),
            float(selected_full_row["a_route"]),
        ],
        "development_high_candidate_horizons": int(
            expected_grades["development_label"]
            .eq("development_high_candidate")
            .sum()
        ),
        "development_good_candidate_horizons": int(
            expected_grades["development_label"]
            .eq("development_good_candidate")
            .sum()
        ),
        "parent_grade_changed": False,
        "prospective_validated": False,
    }
    saved_provisional = json.loads((root / "provisional_decision.json").read_text())
    provisional_differences = nested_differences(
        saved_provisional, expected_provisional
    )
    audit.check(
        "provisional_decision_independently_exact",
        not provisional_differences,
        provisional_differences,
    )
    final_artifact_hashes = {
        name: sha256(root / name) for name in FIT_ARTIFACTS
    }
    final_fit_complete_sha256 = sha256(root / "fit_complete.json")
    final_runner_sha256 = sha256(RUNNER_PATH)
    audit.check(
        "fit_bundle_unchanged_through_independent_reconstruction",
        marker.get("artifact_hashes") == final_artifact_hashes
        and marker.get("runner_sha256") == final_runner_sha256
        and marker.get("contract_sha256") == CONTRACT_SHA256,
        {
            "runner_sha256": final_runner_sha256,
            "fit_complete_sha256": final_fit_complete_sha256,
        },
    )
    result = {
        "phase": "hierarchical_v1_independent_pre_score",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": final_runner_sha256,
        "fit_complete_sha256": final_fit_complete_sha256,
        "fit_artifact_hashes": final_artifact_hashes,
        "maximum_oof_probability_error": maximum_oof_error,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scoring_authorized": audit.all_passed,
        "later_period_outcomes_opened_by_audit": False,
        "prospective_validated": False,
        "parent_grade_changed": False,
    }
    write_json(root / "pre_score_audit.json", result)
    return result


def load_npz_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {name: np.asarray(stored[name]).copy() for name in stored.files}


def frozen_context_transform(
    frame: pd.DataFrame, parameters: Mapping[str, np.ndarray]
) -> sparse.csr_matrix:
    medians = {
        name: float(parameters["context_numeric_medians"][index])
        for index, name in enumerate(NUMERIC_CONTROLS)
    }
    raw = raw_context(frame, medians)
    scale = np.asarray(parameters["context_scaler_scale"], dtype=float)
    if scale.shape != (17,) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise AssertionError("invalid frozen context scale")
    transformed = raw.multiply(1.0 / scale).tocsr()
    if transformed.shape != (len(frame), 17) or not np.isfinite(transformed.data).all():
        raise AssertionError("invalid independently transformed context")
    return transformed


def independent_stored_class_probabilities(
    matrix: sparse.csr_matrix,
    parameters: Mapping[str, np.ndarray],
    key: str,
) -> np.ndarray:
    classes = np.asarray(parameters[f"{key}__classes"], dtype=int)
    coefficients = np.asarray(parameters[f"{key}__coef"], dtype=float)
    intercept = np.asarray(parameters[f"{key}__intercept"], dtype=float)
    temperature = np.asarray(parameters[f"{key}__temperature"], dtype=float)
    if (
        not np.array_equal(classes, np.asarray([0, 1, 2]))
        or coefficients.shape != (3, matrix.shape[1])
        or intercept.shape != (3,)
        or temperature.shape != (1,)
        or float(temperature[0]) != 1.0
    ):
        raise AssertionError(f"invalid frozen ordered model bundle: {key}")
    logits = np.asarray(matrix @ coefficients.T, dtype=float)
    logits += intercept
    logits /= float(temperature[0])
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    if (
        probability.shape != (matrix.shape[0], 3)
        or not np.isfinite(probability).all()
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12)
    ):
        raise AssertionError(f"invalid independently replayed probabilities: {key}")
    return probability


def add_independent_qhier_probability(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    class_probability: np.ndarray,
) -> None:
    probability = np.asarray(class_probability, dtype=float)
    if probability.shape != (len(frame), 3):
        raise AssertionError("independent class probability shape changed")
    tier = tier_probabilities(probability)
    structural = frame["loop_probability"].to_numpy(dtype=float)
    key = task_key(target, horizon)
    for tier_name in TIERS:
        values = tier[tier_name]
        frame[f"{key}__{tier_name}"] = values
        frame[f"joint__{key}__{tier_name}"] = structural * values


def independent_score_with_frozen_bundle(
    source: pd.DataFrame,
    mapping: pd.DataFrame,
    vectors: np.ndarray,
    parameters: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if "current_state" in source and not np.array_equal(
        source["state"].to_numpy(dtype=int),
        source["current_state"].to_numpy(dtype=int),
    ):
        raise AssertionError("later source state and current_state disagree")
    topology_columns = topology_column_names()
    if not set(topology_columns).issubset(source.columns):
        raise AssertionError("sealed later source lacks V3 topology columns")
    route_lookup = {
        (str(row.cycle_id), int(row.current_state)): int(row.route_index)
        for row in mapping.itertuples(index=False)
    }
    route_indices = np.asarray(
        [
            route_lookup[(str(cycle), int(state))]
            for cycle, state in zip(
                source["cycle_id"], source["state"], strict=True
            )
        ],
        dtype=int,
    )
    topology_error = float(
        np.max(
            np.abs(
                source[topology_columns].to_numpy(dtype=float)
                - vectors[route_indices]
            ),
            initial=0.0,
        )
    )
    if topology_error > 1e-12:
        raise AssertionError("sealed later topology failed independent replay")
    # Preserve the exact pinned V3 serialized floats used by production after
    # independently reconstructing and verifying their mathematical values.
    output = source.copy()
    output["route_index"] = route_indices
    if "current_state" not in output:
        output["current_state"] = output["state"].to_numpy(dtype=int)
    pair = (
        float(np.asarray(parameters["selected_a_cycle"])[0]),
        float(np.asarray(parameters["selected_a_route"])[0]),
    )
    grid_index = int(np.asarray(parameters["selected_grid_index"])[0])
    if pair not in SCALE_GRID or SCALE_GRID[grid_index] != pair:
        raise AssertionError("frozen selected scale pair is inconsistent")
    if pair == (0.0, 0.0):
        for target in TARGETS:
            for horizon in HORIZONS:
                p75 = output[
                    f"qroute_topology__{target}__h{horizon}__p75"
                ].to_numpy(dtype=float)
                p90 = output[
                    f"qroute_topology__{target}__h{horizon}__p90"
                ].to_numpy(dtype=float)
                classes = np.column_stack((1.0 - p75, p75 - p90, p90))
                add_independent_qhier_probability(output, target, horizon, classes)
    else:
        context = frozen_context_transform(output, parameters)
        topology = output[topology_columns].to_numpy(dtype=float).copy()
        topology[:, 19:61] *= 0.5
        mu_cycle = np.asarray(parameters["mu_cycle"], dtype=float)
        mu_route = np.asarray(parameters["mu_route_within_cycle"], dtype=float)
        route_cycle = np.asarray(parameters["route_cycle"], dtype=int)
        expected_route_cycle = mapping.sort_values(
            "route_index", kind="stable"
        )["cycle_index"].to_numpy(dtype=int)
        if (
            mu_cycle.shape != (20,)
            or mu_route.shape != (44,)
            or not np.array_equal(route_cycle, expected_route_cycle)
        ):
            raise AssertionError("frozen hierarchy centers are invalid")
        matrix = sparse.hstack(
            (
                context,
                sparse.csr_matrix(topology),
                centered_cycle_block(
                    output["cycle_index"].to_numpy(dtype=int), mu_cycle, pair[0]
                ),
                centered_route_block(
                    output["cycle_index"].to_numpy(dtype=int),
                    output["route_index"].to_numpy(dtype=int),
                    mu_route,
                    route_cycle,
                    pair[1],
                ),
            ),
            format="csr",
        )
        if matrix.shape != (len(output), 144):
            raise AssertionError("later independent hierarchy width changed")
        for target in TARGETS:
            for horizon in HORIZONS:
                classes = independent_stored_class_probabilities(
                    matrix, parameters, task_key(target, horizon)
                )
                add_independent_qhier_probability(output, target, horizon, classes)
    errors = independent_probability_chain_errors(output)
    errors["source_topology_maximum_error"] = topology_error
    return output, errors


def independent_probability_chain_errors(panel: pd.DataFrame) -> dict[str, float]:
    structural = panel["loop_probability"].to_numpy(dtype=float)
    errors: dict[str, float] = {}
    for model in ("qcontext", "qroute_topology", "qfull", "qhier"):
        maximum = 0.0
        for target in TARGETS:
            for horizon in HORIZONS:
                p75 = panel[
                    f"{model}__{target}__h{horizon}__p75"
                ].to_numpy(dtype=float)
                p90 = panel[
                    f"{model}__{target}__h{horizon}__p90"
                ].to_numpy(dtype=float)
                if (
                    not np.isfinite(p75).all()
                    or not np.isfinite(p90).all()
                    or np.any(p90 > p75 + EPSILON)
                    or np.any(p90 < -EPSILON)
                    or np.any(p75 > 1.0 + EPSILON)
                ):
                    raise AssertionError(f"invalid later probability bundle: {model}")
                for tier_name, conditional in (("p75", p75), ("p90", p90)):
                    joint = panel[
                        f"joint__{model}__{target}__h{horizon}__{tier_name}"
                    ].to_numpy(dtype=float)
                    maximum = max(
                        maximum,
                        float(
                            np.max(
                                np.abs(joint - structural * conditional), initial=0.0
                            )
                        ),
                    )
        if maximum > 1e-12:
            raise AssertionError(f"later joint chain replay failed: {model}")
        errors[model] = maximum
    return errors


def independent_period_transfer(
    provisional: Mapping[str, Any],
    provisional_grades: pd.DataFrame,
    later_grades: Mapping[str, pd.DataFrame],
    period_payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    indexed = {
        period: frame.set_index(["cycle_index", "horizon"])
        for period, frame in later_grades.items()
    }
    transfer_rows: list[dict[str, Any]] = []
    for row in provisional_grades.itertuples(index=False):
        key = (int(row.cycle_index), int(row.horizon))
        base = str(row.development_label)
        later = {
            period: str(frame.loc[key, "development_label"])
            for period, frame in indexed.items()
        }
        final_label = ordered_minimum_development_label(
            (base, later["2025"], later["2023"])
        )
        transfer_rows.append(
            {
                "cycle_index": key[0],
                "cycle_id": str(row.cycle_id),
                "horizon": key[1],
                "provisional_label": base,
                "development_2025_label": later["2025"],
                "backward_2023_label": later["2023"],
                "final_development_portability_label": final_label,
                "development_portable": final_label
                != "development_unqualified",
                "provisional_label_retained": final_label == base,
                "later_promotion_performed": False,
                "parent_grade_changed": False,
                "prospective_validated": False,
            }
        )
    global_rows: list[dict[str, Any]] = []
    for cycle_id, base_group in provisional_grades.groupby("cycle_id", sort=True):
        base = str(base_group["global_development_label"].iloc[0])
        later = {
            period: str(
                frame.loc[
                    frame["cycle_id"].astype(str).eq(str(cycle_id)),
                    "global_development_label",
                ].iloc[0]
            )
            for period, frame in later_grades.items()
        }
        final_label = ordered_minimum_development_label(
            (base, later["2025"], later["2023"])
        )
        global_rows.append(
            {
                "cycle_id": str(cycle_id),
                "provisional_label": base,
                "development_2025_label": later["2025"],
                "backward_2023_label": later["2023"],
                "final_development_portability_label": final_label,
                "development_portable": final_label
                != "development_unqualified",
                "provisional_label_retained": final_label == base,
                "later_promotion_performed": False,
                "parent_grade_changed": False,
                "prospective_validated": False,
            }
        )
    return {
        "provisional_algorithm_label": provisional["primary_algorithm_label"],
        "periods": dict(period_payloads),
        "algorithm_development_portable": bool(
            provisional["primary_algorithm_pass"]
            and all(
                payload["primary_algorithm_pass"]
                for payload in period_payloads.values()
            )
        ),
        "later_promotion_performed": False,
        "named_transfer": transfer_rows,
        "global_named_transfer": global_rows,
        "prospective_validated": False,
    }


def _closed_post_score_result(
    audit: Audit, details: str, preaudit: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    audit.check("post_score_phase_lock", False, details)
    scoring_was_authorized = bool(
        preaudit
        and preaudit.get("all_passed") is True
        and preaudit.get("scoring_authorized") is True
    )
    return {
        "phase": "hierarchical_v1_independent_post_score",
        "all_passed": False,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scoring_was_authorized": scoring_was_authorized,
        "scoring_authorized": scoring_was_authorized,
        "later_period_outcomes_opened_by_audit": False,
        "prospective_validated": False,
        "parent_grade_changed": False,
    }


def post_score_audit(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Independently replay sealed later scoring after both phase locks exist."""

    audit = Audit()
    required_markers = {
        "fit_complete.json": root / "fit_complete.json",
        "pre_score_audit.json": root / "pre_score_audit.json",
        "scoring_complete.json": root / "scoring_complete.json",
        "summary.json": root / "summary.json",
    }
    missing = [name for name, path in required_markers.items() if not path.is_file()]
    if missing:
        return _closed_post_score_result(
            audit, f"missing sealed phase marker(s): {missing}"
        )
    fit = json.loads(required_markers["fit_complete.json"].read_text())
    preaudit = json.loads(required_markers["pre_score_audit.json"].read_text())
    scoring = json.loads(required_markers["scoring_complete.json"].read_text())
    current_runner_hash = sha256(RUNNER_PATH)
    lock_pass = bool(
        preaudit.get("all_passed") is True
        and preaudit.get("scoring_authorized") is True
        and preaudit.get("contract_sha256") == CONTRACT_SHA256
        and preaudit.get("runner_sha256") == current_runner_hash
        and preaudit.get("fit_complete_sha256")
        == sha256(required_markers["fit_complete.json"])
        and preaudit.get("fit_artifact_hashes") == fit.get("artifact_hashes")
        and fit.get("contract_sha256") == CONTRACT_SHA256
        and fit.get("runner_sha256") == current_runner_hash
        and scoring.get("contract_sha256") == CONTRACT_SHA256
        and scoring.get("runner_sha256") == current_runner_hash
        and scoring.get("pre_score_audit_sha256")
        == sha256(required_markers["pre_score_audit.json"])
        and scoring.get("pre_score_audit_passed") is True
    )
    if not lock_pass:
        return _closed_post_score_result(
            audit, "fit, pre-score, or scoring phase lock mismatch", preaudit
        )
    contract = verify_contract_semantics(audit)
    current_source_hashes = verify_source_pins(audit, contract)
    expected_fit_sources = {
        **current_source_hashes,
        "hierarchical_contract_sha256": CONTRACT_SHA256,
        "hierarchical_runner_sha256": current_runner_hash,
    }
    saved_fit_sources = json.loads((root / "fit_source_hashes.json").read_text())
    audit.check(
        "current_frozen_source_set_still_matches_fit_freeze",
        saved_fit_sources == expected_fit_sources,
        {
            "stored_only": sorted(set(saved_fit_sources) - set(expected_fit_sources)),
            "expected_only": sorted(set(expected_fit_sources) - set(saved_fit_sources)),
        },
    )
    if not audit.all_passed:
        return _closed_post_score_result(
            audit,
            "contract or frozen source changed before post-score reconstruction",
            preaudit,
        )
    fit_hashes = {
        name: sha256(root / name)
        for name in FIT_ARTIFACTS
        if (root / name).is_file()
    }
    audit.check(
        "fit_bundle_still_exact_after_later_scoring",
        fit_hashes == fit.get("artifact_hashes"),
        {
            "stored_count": len(fit.get("artifact_hashes", {})),
            "current_count": len(fit_hashes),
        },
    )
    missing_scoring = [
        name for name in SCORING_HASHED_ARTIFACTS if not (root / name).is_file()
    ]
    if missing_scoring:
        return _closed_post_score_result(
            audit, f"missing scored artifact(s): {missing_scoring}", preaudit
        )
    scoring_hashes = {
        name: sha256(root / name) for name in SCORING_HASHED_ARTIFACTS
    }
    audit.check(
        "all_scoring_artifact_hashes_match_scoring_marker",
        scoring_hashes == scoring.get("artifact_hashes"),
        {
            "stored_count": len(scoring.get("artifact_hashes", {})),
            "current_count": len(scoring_hashes),
        },
    )
    safety_pass = bool(
        scoring.get("research_only") is True
        and scoring.get("live_ordering_enabled") is False
        and scoring.get("order_placement") == "disabled"
        and scoring.get("later_periods_are_prospective") is False
        and scoring.get("parent_grade_changed") is False
        and scoring.get("shadow_tree_read") is False
        and scoring.get("shadow_tree_written") is False
    )
    audit.check("post_score_safety_and_no_shadow_labels_exact", safety_pass)
    v3_complete = json.loads((V3_ROOT / "scoring_complete.json").read_text())
    source_paths = {
        period: V3_ROOT / f"scoring_predictions_{period}.parquet"
        for period in ("2025", "2023")
    }
    source_hashes = {
        f"scoring_predictions_{period}.parquet": sha256(path)
        for period, path in source_paths.items()
    }
    expected_source_hashes = {
        name: v3_complete["artifact_hashes"][name]
        for name in source_hashes
    }
    stored_source_hashes = json.loads(
        (root / "evaluation_source_hashes.json").read_text()
    )
    audit.check(
        "sealed_later_source_hashes_independently_exact",
        source_hashes == expected_source_hashes == stored_source_hashes,
        {
            "actual": source_hashes,
            "v3_marker": expected_source_hashes,
            "stored": stored_source_hashes,
        },
    )
    if not audit.all_passed:
        return _closed_post_score_result(
            audit,
            "fit, scoring, safety, or later-source integrity failed before row-level reconstruction",
            preaudit,
        )
    mapping, vectors = build_rotation_mapping()
    parameters = load_npz_bundle(root / "model_parameters.npz")
    later_grades: dict[str, pd.DataFrame] = {}
    period_payloads: dict[str, dict[str, Any]] = {}
    maximum_prediction_errors: dict[str, float] = {}
    for period in ("2025", "2023"):
        source = pd.read_parquet(source_paths[period])
        years = pd.to_datetime(source["session_date"], errors="raise").dt.year
        audit.check(
            f"{period}_panel_period_is_exact_and_contains_no_2026",
            bool(years.eq(int(period)).all() and not years.eq(2026).any()),
            {"rows": len(source), "years": sorted(years.unique().tolist())},
        )
        independent_panel, probability_errors = independent_score_with_frozen_bundle(
            source, mapping, vectors, parameters
        )
        cuts = np.asarray(parameters["entropy_quartile_cutpoints"], dtype=float)
        expected_entropy = np.searchsorted(
            cuts,
            independent_panel["next_state_entropy_normalized"].to_numpy(dtype=float),
            side="left",
        ).astype(np.int8)
        entropy_exact = np.array_equal(
            independent_panel["entropy_quartile"].to_numpy(dtype=np.int8),
            expected_entropy,
        )
        audit.check(
            f"{period}_frozen_entropy_assignment_independently_exact",
            entropy_exact,
            {"cutpoints": cuts},
        )
        stored_panel = pd.read_parquet(root / f"scoring_predictions_{period}.parquet")
        panel_pass, panel_details = _frame_maximum_error(
            stored_panel,
            independent_panel,
            ["anchor_id", "cycle_index"],
        )
        maximum_prediction_errors[period] = float(
            panel_details.get("maximum_numeric_error", math.inf)
        )
        audit.check(
            f"{period}_all_predictions_independently_exact",
            panel_pass,
            {**panel_details, "chain_errors": probability_errors},
        )
        (
            expected_metrics,
            expected_calibration,
            expected_rotations,
            expected_gates,
            block_positions,
            calendar,
        ) = independent_core_evaluation(independent_panel, period, "scoring")
        expected_falsification = independent_falsification(independent_panel)
        period_pass = bool(
            expected_gates["support_pass"]
            and expected_gates["primary_without_falsification_pass"]
            and expected_falsification["pass"]
        )
        expected_gates["falsification"] = expected_falsification
        expected_gates["primary_algorithm_pass"] = period_pass
        expected_gates["primary_algorithm_label"] = (
            "development_algorithm_supported"
            if period_pass
            else "development_algorithm_unconfirmed"
        )
        expected_gates["probability_chain_replay_max_error"] = {
            name: probability_errors[name]
            for name in ("qcontext", "qroute_topology", "qfull", "qhier")
        }
        expected_grades = independent_named_diagnostics(
            independent_panel,
            expected_rotations,
            block_positions,
            calendar,
            period_pass,
            mapping,
            mode="scoring",
        )
        period_frames = (
            (
                f"cell_diagnostics_{period}.csv",
                expected_metrics,
                ["surface", "model", "target", "horizon", "tier"],
                "all_cell_metrics",
            ),
            (
                f"calibration_diagnostics_{period}.csv",
                expected_calibration,
                ["surface", "model", "target", "horizon", "tier", "bin"],
                "all_calibration_bins",
            ),
            (
                f"rotation_diagnostics_{period}.csv",
                expected_rotations,
                ["group_type", "group_value", "surface", "baseline"],
                "all_supported_slices",
            ),
            (
                f"per_loop_grades_{period}.csv",
                expected_grades,
                ["cycle_index", "horizon"],
                "all_named_holm_and_labels",
            ),
        )
        for filename, expected, keys, label in period_frames:
            stored = pd.read_csv(root / filename)
            passed, details = _frame_maximum_error(stored, expected, keys)
            audit.check(f"{period}_{label}_independently_exact", passed, details)
        stored_gates = json.loads((root / f"algorithm_gates_{period}.json").read_text())
        gate_differences = nested_differences(stored_gates, expected_gates)
        audit.check(
            f"{period}_all_bootstrap_falsification_and_algorithm_gates_independently_exact",
            not gate_differences,
            gate_differences[:20],
        )
        expected_support = {
            "effective_weight": float(
                independent_panel.loc[
                    independent_panel["loop_occurs"].eq(1), "conditional_weight"
                ].sum()
            ),
            "minimum_effective_weight": 25000.0,
            "support_pass": expected_gates["support_pass"],
        }
        stored_support = json.loads((root / f"support_{period}.json").read_text())
        support_differences = nested_differences(stored_support, expected_support)
        audit.check(
            f"{period}_support_independently_exact",
            not support_differences,
            support_differences,
        )
        later_grades[period] = expected_grades
        period_payloads[period] = {
            "primary_algorithm_pass": period_pass,
            "primary_algorithm_label": expected_gates["primary_algorithm_label"],
        }
    provisional = json.loads((root / "provisional_decision.json").read_text())
    provisional_grades = pd.read_csv(root / "per_loop_grades_2024.csv")
    expected_transfer = independent_period_transfer(
        provisional, provisional_grades, later_grades, period_payloads
    )
    stored_transfer = json.loads((root / "period_transfer_gates.json").read_text())
    transfer_differences = nested_differences(stored_transfer, expected_transfer)
    audit.check(
        "all_demotion_only_transfer_decisions_independently_exact",
        not transfer_differences
        and expected_transfer["later_promotion_performed"] is False
        and expected_transfer["prospective_validated"] is False,
        transfer_differences[:20],
    )
    expected_scoring_marker = {
        "status": "scoring_complete_development_and_backward_portability_only",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": current_runner_hash,
        "pre_score_audit_sha256": sha256(root / "pre_score_audit.json"),
        "pre_score_audit_passed": True,
        "transfer": expected_transfer,
        "later_periods_are_prospective": False,
        "parent_grade_changed": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "artifact_hashes": scoring_hashes,
    }
    marker_differences = nested_differences(scoring, expected_scoring_marker)
    audit.check(
        "scoring_complete_marker_independently_exact",
        not marker_differences,
        marker_differences[:20],
    )
    expected_summary = {
        **expected_scoring_marker,
        "fit_complete_sha256": sha256(root / "fit_complete.json"),
        "interpretation": "Movement magnitude and future range only; no directional, economic, ordering, or deployment claim.",
    }
    saved_summary = json.loads((root / "summary.json").read_text())
    summary_differences = nested_differences(saved_summary, expected_summary)
    audit.check(
        "summary_independently_exact",
        not summary_differences,
        summary_differences[:20],
    )
    result = {
        "phase": "hierarchical_v1_independent_post_score",
        "all_passed": audit.all_passed,
        "check_count": len(audit.checks),
        "checks": audit.checks,
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": current_runner_hash,
        "fit_complete_sha256": sha256(root / "fit_complete.json"),
        "pre_score_audit_sha256": sha256(root / "pre_score_audit.json"),
        "scoring_complete_sha256": sha256(root / "scoring_complete.json"),
        "summary_sha256": sha256(root / "summary.json"),
        "fit_artifact_hashes": fit_hashes,
        "scoring_artifact_hashes": scoring_hashes,
        "evaluation_source_hashes": source_hashes,
        "maximum_prediction_errors": maximum_prediction_errors,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "scoring_was_authorized": True,
        "scoring_authorized": True,
        "later_period_outcomes_opened_by_audit": True,
        "later_periods_are_prospective": False,
        "later_promotion_performed": False,
        "prospective_validated": False,
        "parent_grade_changed": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
    }
    write_json(root / "independent_artifact_audit.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preartifact-only", action="store_true")
    modes.add_argument("--pre-score-only", action="store_true")
    modes.add_argument("--post-score", action="store_true")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preartifact_only:
        result = preartifact_audit()
    elif args.pre_score_only:
        result = pre_score_audit(args.root)
    else:
        result = post_score_audit(args.root)
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    if not result.get("all_passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
