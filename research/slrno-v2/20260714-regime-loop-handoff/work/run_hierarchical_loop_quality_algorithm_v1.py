"""Causal hierarchical movement-quality algorithm (research only).

The fit phase uses only 2024.  Development/backward scoring is hard-locked
behind an independent pre-score artifact audit.  This module never opens a
prospective shadow or a 2026 outcome.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-hierarchical-loop-quality-algorithm-v1.json"
CONTRACT_SHA256 = "f6956b6ab0495a49669f714df834d1fd0fdaa13b0ecf4b123d6c54c0fc9b5936"
V3_SOURCE = HERE / "run_loop_quality_feature_ablation_v3.py"
V1_SOURCE = HERE / "run_per_loop_movement_quality.py"
V3_SOURCE_SHA256 = "c3aa481dd880e35cc0cc07baa41b6d6c2ed1c380d935e31ce8c1a9d4ff7f05c8"
V1_SOURCE_SHA256 = "7da5e88e603583d3dba7422569bc8e27837171c7165e69bcaafade472738e2ea"
V1_PROVISIONAL_SUPPORT_SHA256 = "5974edec6960c961182628ca3c854d55c9c6ed7aab402e0dfbb6bcfdea4bddd1"

PARENT_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
V3_ROOT = Path("/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710")
OUT = Path("/private/tmp/stocker_hierarchical_loop_quality_algorithm_v1_20260711")

TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
OUTER_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
CANDIDATE_MONTHS = tuple(f"2024-{month:02d}" for month in range(4, 13))
FINAL_SELECTION_MONTHS = ("2024-10", "2024-11", "2024-12")
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
INNER_SCHEDULE = {
    "2024-07": ("2024-04", "2024-05", "2024-06"),
    "2024-08": ("2024-05", "2024-06", "2024-07"),
    "2024-09": ("2024-06", "2024-07", "2024-08"),
    "2024-10": ("2024-07", "2024-08", "2024-09"),
    "2024-11": ("2024-08", "2024-09", "2024-10"),
    "2024-12": ("2024-09", "2024-10", "2024-11"),
}
MODEL_C = 0.2
MODEL_MAX_ITER = 2000
MODEL_TOL = 1e-10
SEED = 20260711
BOOTSTRAP_DRAWS = 20000
NULL_DRAWS = 999
EPSILON = 1e-12
CONTEXT_WIDTH = 17
TOPOLOGY_WIDTH = 63
CYCLE_WIDTH = 20
ROUTE_WIDTH = 44
HIERARCHY_WIDTH = 144


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256(V3_SOURCE) != V3_SOURCE_SHA256:
    raise AssertionError("pinned V3 source changed")
v3 = _load_module(V3_SOURCE, "frozen_v3_hierarchical_dependency")


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def expected_source_paths() -> dict[str, Path]:
    return {
        "per_loop_movement_quality_v1_contract_sha256": HERE
        / "contracts/20260710-per-loop-movement-quality-v1.json",
        "loop_quality_feature_ablation_v3_contract_sha256": HERE
        / "contracts/20260710-loop-quality-feature-ablation-v3.json",
        "loop_quality_feature_ablation_v3_runner_sha256": V3_SOURCE,
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
        "fixed_cycles_csv_sha256": PARENT_ROOT / "fixed_cycles.csv",
        "frozen_semimarkov_parameters_npz_sha256": STATE_ROOT
        / "frozen_semimarkov_parameters.npz",
        "quality_thresholds_2024_json_sha256": PARENT_ROOT
        / "quality_thresholds_2024.json",
        "quality_feature_manifest_json_sha256": PARENT_ROOT / "feature_manifest.json",
        "quality_fit_manifest_json_sha256": PARENT_ROOT / "fit_manifest.json",
        "parent_oof_predictions_2024_parquet_sha256": PARENT_ROOT
        / "oof_predictions_2024.parquet",
        "parent_training_long_2024_parquet_sha256": PARENT_ROOT
        / "training_long_2024.parquet",
        "anchor_panel_train_2024_parquet_sha256": PRICE_ROOT
        / "anchor_panel_train_2024.parquet",
        "parent_final_cycle_tiers_csv_sha256": PARENT_ROOT / "final_cycle_tiers.csv",
        "parent_gates_json_sha256": PARENT_ROOT / "gates.json",
        "parent_summary_json_sha256": PARENT_ROOT / "summary.json",
    }


def validate_contract_and_sources() -> tuple[dict[str, Any], dict[str, str]]:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise AssertionError("hierarchical algorithm contract changed")
    contract = json.loads(CONTRACT.read_text())
    if contract["research_only"] is not True:
        raise AssertionError("research-only label changed")
    if contract["live_ordering_enabled"] is not False:
        raise AssertionError("ordering safety label changed")
    if contract["order_placement"] != "disabled":
        raise AssertionError("order-placement label changed")
    if contract["execution_authorization"]["authorized"] is not True:
        raise AssertionError("execution is not authorized")
    if sha256(V1_SOURCE) != V1_SOURCE_SHA256:
        raise AssertionError("sealed V1 named-grade implementation changed")
    contract_grid = tuple(
        tuple(float(item) for item in pair) for pair in contract["scale_grid"]["pairs"]
    )
    if contract_grid != SCALE_GRID or len(SCALE_GRID) != 15:
        raise AssertionError("literal scale grid changed")
    expected = contract["frozen_lineage"]["source_pins"]
    paths = expected_source_paths()
    if set(expected) != set(paths):
        raise AssertionError("source-pin path set is incomplete")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatch = {
        name: {"expected": expected[name], "actual": value}
        for name, value in actual.items()
        if value != expected[name]
    }
    if mismatch:
        raise AssertionError(f"pinned source changed: {mismatch}")
    actual["hierarchical_contract_sha256"] = CONTRACT_SHA256
    actual["hierarchical_runner_sha256"] = sha256(Path(__file__))
    actual["per_loop_runner.py"] = sha256(V1_SOURCE)
    provisional_support = PARENT_ROOT / "provisional_support_2024.csv"
    if sha256(provisional_support) != V1_PROVISIONAL_SUPPORT_SHA256:
        raise AssertionError("sealed V1 fit-eligibility artifact changed")
    actual["provisional_support_2024.csv"] = sha256(provisional_support)
    return contract, actual


def route_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(V3_ROOT / "rotation_mapping.csv").sort_values(
        ["cycle_index", "current_state"], kind="stable"
    ).reset_index(drop=True)
    if len(mapping) != ROUTE_WIDTH:
        raise AssertionError("frozen compatible route count changed")
    if mapping.duplicated(["cycle_index", "current_state"]).any():
        raise AssertionError("duplicate compatible route")
    mapping.insert(0, "route_index", np.arange(ROUTE_WIDTH, dtype=int))
    return mapping


def add_route_index(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    if "current_state" in frame:
        if not np.array_equal(
            frame["state"].to_numpy(int), frame["current_state"].to_numpy(int)
        ):
            raise AssertionError("state and frozen current_state disagree")
    lookup = mapping[["route_index", "cycle_index", "current_state"]].rename(
        columns={"current_state": "route_state"}
    )
    output = frame.merge(
        lookup,
        left_on=["cycle_index", "state"],
        right_on=["cycle_index", "route_state"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(output) != len(frame) or output["route_index"].isna().any():
        raise AssertionError("unknown or incompatible cycle-state route")
    output["route_index"] = output["route_index"].astype(int)
    return output.drop(columns="route_state")


def weighted_centers(
    cycle_index: np.ndarray,
    route_index: np.ndarray,
    weights: np.ndarray,
    mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cycle_index = np.asarray(cycle_index, int)
    route_index = np.asarray(route_index, int)
    weights = np.asarray(weights, float)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(weights).all():
        raise AssertionError("non-positive or non-finite training weight")
    mu_cycle = np.bincount(cycle_index, weights=weights, minlength=CYCLE_WIDTH) / total
    cycle_weight = np.bincount(cycle_index, weights=weights, minlength=CYCLE_WIDTH)
    if (cycle_weight <= 0.0).any():
        raise AssertionError("a frozen cycle has no positive training-fold weight")
    route_weight = np.bincount(route_index, weights=weights, minlength=ROUTE_WIDTH)
    route_cycle = mapping.sort_values("route_index")["cycle_index"].to_numpy(int)
    mu_route = route_weight / cycle_weight[route_cycle]
    for cycle in range(CYCLE_WIDTH):
        if not np.isclose(mu_route[route_cycle == cycle].sum(), 1.0, atol=1e-12):
            raise AssertionError("within-cycle route center does not sum to one")
    if not np.isclose(mu_cycle.sum(), 1.0, atol=1e-12):
        raise AssertionError("cycle center does not sum to one")
    return mu_cycle, mu_route


def centered_blocks(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    mu_cycle: np.ndarray,
    mu_route: np.ndarray,
    pair: tuple[float, float],
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    a_cycle, a_route = pair
    cycle = frame["cycle_index"].to_numpy(int)
    route = frame["route_index"].to_numpy(int)
    count = len(frame)
    cycle_values = -np.broadcast_to(np.asarray(mu_cycle), (count, CYCLE_WIDTH)).copy()
    cycle_values[np.arange(count), cycle] += 1.0
    cycle_values *= a_cycle
    route_values = np.zeros((count, ROUTE_WIDTH), float)
    route_cycle = mapping.sort_values("route_index")["cycle_index"].to_numpy(int)
    for cycle_id in range(CYCLE_WIDTH):
        rows = cycle == cycle_id
        columns = route_cycle == cycle_id
        route_values[np.ix_(rows, columns)] = -mu_route[columns]
    route_values[np.arange(count), route] += 1.0
    route_values *= a_route
    return sparse.csr_matrix(cycle_values), sparse.csr_matrix(route_values)


def hierarchy_matrix(
    frame: pd.DataFrame,
    scaled_context: sparse.csr_matrix,
    mapping: pd.DataFrame,
    mu_cycle: np.ndarray,
    mu_route: np.ndarray,
    pair: tuple[float, float],
) -> sparse.csr_matrix:
    base = v3.feature_matrices(frame, scaled_context)["qroute_topology"]
    if pair == (0.0, 0.0):
        return base
    cycle, route = centered_blocks(frame, mapping, mu_cycle, mu_route, pair)
    output = sparse.hstack((base, cycle, route), format="csr")
    if output.shape != (len(frame), HIERARCHY_WIDTH):
        raise AssertionError("hierarchical feature width changed")
    if not np.isfinite(output.data).all():
        raise AssertionError("non-finite hierarchical feature")
    return output


def fit_hierarchical_model(
    matrix: sparse.csr_matrix, labels: np.ndarray, weights: np.ndarray
) -> LogisticRegression:
    labels = np.asarray(labels, int)
    if not np.array_equal(np.unique(labels), np.asarray([0, 1, 2])):
        raise AssertionError("ordered task lacks a class")
    model = LogisticRegression(
        C=MODEL_C,
        solver="lbfgs",
        max_iter=MODEL_MAX_ITER,
        tol=MODEL_TOL,
        random_state=SEED,
    )
    model.fit(matrix, labels, sample_weight=np.asarray(weights, float))
    if not np.array_equal(model.classes_, np.asarray([0, 1, 2])):
        raise AssertionError("class order changed")
    if int(model.n_iter_[0]) >= MODEL_MAX_ITER:
        raise AssertionError("hierarchical model failed convergence")
    return model


def binary_losses(observed: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    observed = np.asarray(observed, float)
    probability = np.clip(np.asarray(probability, float), EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": -(observed * np.log(probability) + (1.0 - observed) * np.log(1.0 - probability)),
        "brier": np.square(probability - observed),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    if not len(values) or float(weights.sum()) <= 0.0:
        return math.nan
    return float(np.average(values, weights=weights))


def choose_scale_pair(objectives: Mapping[tuple[float, float], float]) -> tuple[float, float]:
    if tuple(objectives) != SCALE_GRID:
        raise AssertionError("selection objectives do not follow literal pair order")
    minimum = min(float(value) for value in objectives.values())
    tie_set = [pair for pair in SCALE_GRID if float(objectives[pair]) <= minimum + 1e-6]
    return min(tie_set, key=lambda pair: (pair[1], pair[0]))


def task_key(target: str, horizon: int) -> str:
    return f"qhier__{target}__h{horizon}"


def add_qhier_probability(
    frame: pd.DataFrame, target: str, horizon: int, probability: np.ndarray
) -> None:
    probability = np.asarray(probability, float)
    if probability.shape != (len(frame), 3):
        raise AssertionError("class probability shape changed")
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-12
    ):
        raise AssertionError("invalid class probability")
    p75 = probability[:, 1] + probability[:, 2]
    p90 = probability[:, 2]
    key = task_key(target, horizon)
    frame[f"{key}__p75"] = p75
    frame[f"{key}__p90"] = p90
    structural = frame["loop_probability"].to_numpy(float)
    frame[f"joint__{key}__p75"] = structural * p75
    frame[f"joint__{key}__p90"] = structural * p90
    if (p90 > p75 + EPSILON).any() or p90.min() < -EPSILON or p75.max() > 1 + EPSILON:
        raise AssertionError("ordered qhier probabilities are invalid")


def probability_columns(models: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for model in models:
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    columns.append(f"{model}__{target}__h{horizon}__{tier}")
                    columns.append(f"joint__{model}__{target}__h{horizon}__{tier}")
    return columns


def load_oof_2024(mapping: pd.DataFrame) -> pd.DataFrame:
    # The V3 file is itself pinned and already contains the exact topology,
    # controls, sealed baselines, labels, and entropy cut-point assignment.
    frame = pd.read_parquet(V3_ROOT / "oof_predictions_2024.parquet")
    frame = add_route_index(frame, mapping)
    frame["month_key"] = pd.to_datetime(frame["session_date"], errors="raise").dt.strftime(
        "%Y-%m"
    )
    if len(frame) != 216438 or tuple(sorted(frame["month_key"].unique())) != OUTER_MONTHS:
        raise AssertionError("V3 OOF cohort changed")
    if not np.array_equal(frame["source_row"].to_numpy(int), np.arange(len(frame))):
        raise AssertionError("V3 OOF row order changed")
    return frame


def load_training_2024(mapping: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    training = v3.prepare_training(mapping).reset_index(drop=True)
    training = add_route_index(training, mapping)
    if len(training) != 32677 or not training["loop_occurs"].eq(1).all():
        raise AssertionError("realized-loop training cohort changed")
    sealed_columns = [
        f"qroute_topology__{target}__h{horizon}__{tier}"
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    sealed = oof.loc[
        oof["loop_occurs"].eq(1), ["anchor_id", "cycle_id", *sealed_columns]
    ]
    if sealed.duplicated(["anchor_id", "cycle_id"]).any():
        raise AssertionError("sealed V3 positive OOF key is not unique")
    training = training.merge(
        sealed,
        on=["anchor_id", "cycle_id"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    later = training["month_key"].isin(OUTER_MONTHS)
    if training.loc[later, sealed_columns].isna().any().any():
        raise AssertionError("sealed V3 zero-pair prediction merge failed")
    return training


def validate_zero_pair_replay(oof: pd.DataFrame) -> float:
    maximum = 0.0
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                conditional = oof[f"qroute_topology__{target}__h{horizon}__{tier}"].to_numpy(float)
                joint = oof[f"joint__qroute_topology__{target}__h{horizon}__{tier}"].to_numpy(float)
                maximum = max(
                    maximum,
                    float(
                        np.max(
                            np.abs(
                                joint
                                - oof["loop_probability"].to_numpy(float) * conditional
                            )
                        )
                    ),
                )
    if maximum > 1e-12:
        raise AssertionError("sealed V3 zero-pair chain replay failed")
    return maximum


def _scaled_fold_context(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    train_positions: np.ndarray,
    validation_positions: np.ndarray,
    medians: Mapping[str, float],
) -> tuple[StandardScaler, sparse.csr_matrix, sparse.csr_matrix]:
    raw_training = v3.raw_context(training, medians)
    raw_validation = v3.raw_context(validation, medians)
    weights = training.iloc[train_positions]["conditional_weight"].to_numpy(float)
    scaler = v3.fit_scaler(raw_training[train_positions], weights)
    return (
        scaler,
        scaler.transform(raw_training[train_positions]).tocsr(),
        scaler.transform(raw_validation[validation_positions]).tocsr(),
    )


def _cell_loss_stats(
    frame: pd.DataFrame,
    probabilities: Mapping[tuple[str, int], np.ndarray],
) -> dict[tuple[str, int, str], tuple[float, float]]:
    weights = frame["conditional_weight"].to_numpy(float)
    output: dict[tuple[str, int, str], tuple[float, float]] = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            classes = frame[f"quality_class__{target}__h{horizon}"].to_numpy(int)
            probability = probabilities[(target, horizon)]
            for tier, column in (("p75", 1), ("p90", 2)):
                observed = (classes >= column).astype(int)
                predicted = probability[:, 1:].sum(axis=1) if tier == "p75" else probability[:, 2]
                losses = binary_losses(observed, predicted)["log_loss"]
                output[(target, horizon, tier)] = (
                    float(np.dot(weights, losses)),
                    float(weights.sum()),
                )
    return output


def _sealed_zero_probabilities(
    frame: pd.DataFrame,
) -> dict[tuple[str, int], np.ndarray]:
    output: dict[tuple[str, int], np.ndarray] = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            p75 = frame[f"qroute_topology__{target}__h{horizon}__p75"].to_numpy(float)
            p90 = frame[f"qroute_topology__{target}__h{horizon}__p90"].to_numpy(float)
            output[(target, horizon)] = np.column_stack((1.0 - p75, p75 - p90, p90))
    return output


def generate_candidate_loss_statistics(
    training: pd.DataFrame,
    mapping: pd.DataFrame,
    medians: Mapping[str, float],
) -> tuple[
    dict[tuple[str, tuple[float, float]], dict[tuple[str, int, str], tuple[float, float]]],
    pd.DataFrame,
]:
    """Generate the exact causal April-December selection evidence."""

    statistics: dict[
        tuple[str, tuple[float, float]],
        dict[tuple[str, int, str], tuple[float, float]],
    ] = {}
    audit_rows: list[dict[str, Any]] = []
    raw = v3.raw_context(training, medians)
    months = training["month_key"].astype(str).to_numpy()
    all_weights = training["conditional_weight"].to_numpy(float)
    for validation_month in CANDIDATE_MONTHS:
        train_positions = np.flatnonzero(months < validation_month)
        validation_positions = np.flatnonzero(months == validation_month)
        if not len(train_positions) or not len(validation_positions):
            raise AssertionError("empty causal selection fold")
        if not np.all(months[train_positions] < validation_month):
            raise AssertionError("non-causal selection fold")
        train_frame = training.iloc[train_positions].reset_index(drop=True)
        validation_frame = training.iloc[validation_positions].reset_index(drop=True)
        weights = all_weights[train_positions]
        scaler = v3.fit_scaler(raw[train_positions], weights)
        train_context = scaler.transform(raw[train_positions]).tocsr()
        validation_context = scaler.transform(raw[validation_positions]).tocsr()
        mu_cycle, mu_route = weighted_centers(
            train_frame["cycle_index"],
            train_frame["route_index"],
            weights,
            mapping,
        )
        base_train = v3.feature_matrices(train_frame, train_context)["qroute_topology"]
        base_validation = v3.feature_matrices(validation_frame, validation_context)[
            "qroute_topology"
        ]
        for grid_index, pair in enumerate(SCALE_GRID):
            predictions: dict[tuple[str, int], np.ndarray] = {}
            if pair == (0.0, 0.0) and validation_month >= "2024-07":
                predictions = _sealed_zero_probabilities(validation_frame)
                iterations = {(target, horizon): 0 for target in TARGETS for horizon in HORIZONS}
                width = CONTEXT_WIDTH + TOPOLOGY_WIDTH
                zero_source = "sealed_v3_oof"
            else:
                if pair == (0.0, 0.0):
                    train_matrix = base_train
                    validation_matrix = base_validation
                    fit_function = v3.fit_model
                    zero_source = "causal_v3_regeneration"
                else:
                    train_matrix = hierarchy_matrix(
                        train_frame, train_context, mapping, mu_cycle, mu_route, pair
                    )
                    validation_matrix = hierarchy_matrix(
                        validation_frame,
                        validation_context,
                        mapping,
                        mu_cycle,
                        mu_route,
                        pair,
                    )
                    fit_function = fit_hierarchical_model
                    zero_source = "not_zero"
                width = train_matrix.shape[1]
                iterations = {}
                for target in TARGETS:
                    for horizon in HORIZONS:
                        labels = train_frame[
                            f"quality_class__{target}__h{horizon}"
                        ].to_numpy(int)
                        model = fit_function(train_matrix, labels, weights)
                        predictions[(target, horizon)] = model.predict_proba(validation_matrix)
                        iterations[(target, horizon)] = int(model.n_iter_[0])
            statistics[(validation_month, pair)] = _cell_loss_stats(
                validation_frame, predictions
            )
            audit_rows.append(
                {
                    "validation_month": validation_month,
                    "grid_index": grid_index,
                    "a_cycle": pair[0],
                    "a_route": pair[1],
                    "training_month_max": str(train_frame["month_key"].max()),
                    "training_rows": len(train_frame),
                    "training_weight": float(weights.sum()),
                    "validation_rows": len(validation_frame),
                    "feature_width": width,
                    "zero_source": zero_source,
                    "maximum_n_iter": max(iterations.values()),
                    "mu_cycle_hash": hashlib.sha256(mu_cycle.tobytes()).hexdigest(),
                    "mu_route_hash": hashlib.sha256(mu_route.tobytes()).hexdigest(),
                }
            )
    return statistics, pd.DataFrame(audit_rows)


def selection_objectives(
    statistics: Mapping[
        tuple[str, tuple[float, float]],
        Mapping[tuple[str, int, str], tuple[float, float]],
    ],
    months: Sequence[str],
) -> tuple[dict[tuple[float, float], float], dict[tuple[float, float], dict[str, float]]]:
    objectives: dict[tuple[float, float], float] = {}
    components: dict[tuple[float, float], dict[str, float]] = {}
    for pair in SCALE_GRID:
        cell_values: dict[str, float] = {}
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    numerator = 0.0
                    denominator = 0.0
                    for month in months:
                        value, weight = statistics[(month, pair)][(target, horizon, tier)]
                        numerator += value
                        denominator += weight
                    if denominator <= 0.0:
                        raise AssertionError("selection cell has no weight")
                    cell_values[f"loss__{target}__h{horizon}__{tier}"] = numerator / denominator
                    cell_values[f"weight__{target}__h{horizon}__{tier}"] = denominator
        objectives[pair] = float(
            np.mean([value for key, value in cell_values.items() if key.startswith("loss__")])
        )
        components[pair] = cell_values
    return objectives, components


def build_inner_selection_table(
    statistics: Mapping[
        tuple[str, tuple[float, float]],
        Mapping[tuple[str, int, str], tuple[float, float]],
    ]
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], tuple[float, float]]:
    rows: list[dict[str, Any]] = []
    selections: dict[str, tuple[float, float]] = {}
    schedules: list[tuple[str, Sequence[str]]] = [
        *((f"outer:{month}", INNER_SCHEDULE[month]) for month in OUTER_MONTHS),
        ("full_2024", FINAL_SELECTION_MONTHS),
    ]
    for scope, months in schedules:
        objectives, components = selection_objectives(statistics, months)
        selected = choose_scale_pair(objectives)
        selections[scope] = selected
        minimum = min(objectives.values())
        for grid_index, pair in enumerate(SCALE_GRID):
            rows.append(
                {
                    "selection_scope": scope,
                    "validation_months_json": json.dumps(list(months), separators=(",", ":")),
                    "grid_index": grid_index,
                    "a_cycle": pair[0],
                    "a_route": pair[1],
                    **components[pair],
                    "selection_objective": objectives[pair],
                    "objective_minimum": minimum,
                    "in_tie_set": objectives[pair] <= minimum + 1e-6,
                    "selected": pair == selected,
                }
            )
    return pd.DataFrame(rows), selections, selections["full_2024"]


def fit_outer_predictions(
    training: pd.DataFrame,
    oof: pd.DataFrame,
    mapping: pd.DataFrame,
    medians: Mapping[str, float],
    selections: Mapping[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = oof.copy()
    probabilities = {
        (target, horizon): np.full((len(output), 3), np.nan)
        for target in TARGETS
        for horizon in HORIZONS
    }
    audit_rows: list[dict[str, Any]] = []
    training_raw = v3.raw_context(training, medians)
    output_raw = v3.raw_context(output, medians)
    training_month = training["month_key"].astype(str).to_numpy()
    output_month = output["month_key"].astype(str).to_numpy()
    for fold_index, validation_month in enumerate(OUTER_MONTHS, start=1):
        pair = selections[f"outer:{validation_month}"]
        grid_index = SCALE_GRID.index(pair)
        train_positions = np.flatnonzero(training_month < validation_month)
        validation_positions = np.flatnonzero(output_month == validation_month)
        train_frame = training.iloc[train_positions].reset_index(drop=True)
        validation_frame = output.iloc[validation_positions].reset_index(drop=True)
        weights = train_frame["conditional_weight"].to_numpy(float)
        if pair == (0.0, 0.0):
            sealed = _sealed_zero_probabilities(validation_frame)
            for key, value in sealed.items():
                probabilities[key][validation_positions] = value
            for target in TARGETS:
                for horizon in HORIZONS:
                    audit_rows.append(
                        {
                            "fold_index": fold_index,
                            "validation_month": validation_month,
                            "inner_months_json": json.dumps(
                                list(INNER_SCHEDULE[validation_month]), separators=(",", ":")
                            ),
                            "selected_grid_index": grid_index,
                            "a_cycle": pair[0],
                            "a_route": pair[1],
                            "training_month_max": str(train_frame["month_key"].max()),
                            "training_rows": len(train_frame),
                            "training_weight": float(weights.sum()),
                            "validation_compatible_rows": len(validation_frame),
                            "validation_realized_rows": int(validation_frame["loop_occurs"].sum()),
                            "target": target,
                            "horizon": horizon,
                            "feature_width": CONTEXT_WIDTH + TOPOLOGY_WIDTH,
                            "n_iter": 0,
                            "zero_fallback": True,
                            "max_zero_replay_error": 0.0,
                        }
                    )
            continue
        scaler = v3.fit_scaler(training_raw[train_positions], weights)
        train_context = scaler.transform(training_raw[train_positions]).tocsr()
        validation_context = scaler.transform(output_raw[validation_positions]).tocsr()
        mu_cycle, mu_route = weighted_centers(
            train_frame["cycle_index"], train_frame["route_index"], weights, mapping
        )
        train_matrix = hierarchy_matrix(
            train_frame, train_context, mapping, mu_cycle, mu_route, pair
        )
        validation_matrix = hierarchy_matrix(
            validation_frame, validation_context, mapping, mu_cycle, mu_route, pair
        )
        for target in TARGETS:
            for horizon in HORIZONS:
                labels = train_frame[f"quality_class__{target}__h{horizon}"].to_numpy(int)
                model = fit_hierarchical_model(train_matrix, labels, weights)
                probabilities[(target, horizon)][validation_positions] = model.predict_proba(
                    validation_matrix
                )
                audit_rows.append(
                    {
                        "fold_index": fold_index,
                        "validation_month": validation_month,
                        "inner_months_json": json.dumps(
                            list(INNER_SCHEDULE[validation_month]), separators=(",", ":")
                        ),
                        "selected_grid_index": grid_index,
                        "a_cycle": pair[0],
                        "a_route": pair[1],
                        "training_month_max": str(train_frame["month_key"].max()),
                        "training_rows": len(train_frame),
                        "training_weight": float(weights.sum()),
                        "validation_compatible_rows": len(validation_frame),
                        "validation_realized_rows": int(validation_frame["loop_occurs"].sum()),
                        "target": target,
                        "horizon": horizon,
                        "feature_width": train_matrix.shape[1],
                        "n_iter": int(model.n_iter_[0]),
                        "zero_fallback": False,
                        "max_zero_replay_error": math.nan,
                    }
                )
    for (target, horizon), probability in probabilities.items():
        if not np.isfinite(probability).all():
            raise AssertionError("outer OOF probability has an uncovered row")
        add_qhier_probability(output, target, horizon, probability)
    return output, pd.DataFrame(audit_rows)


def fit_full_model_bundle(
    training: pd.DataFrame,
    mapping: pd.DataFrame,
    medians: Mapping[str, float],
    selected_pair: tuple[float, float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    weights = training["conditional_weight"].to_numpy(float)
    raw = v3.raw_context(training, medians)
    scaler = v3.fit_scaler(raw, weights)
    context = scaler.transform(raw).tocsr()
    mu_cycle, mu_route = weighted_centers(
        training["cycle_index"], training["route_index"], weights, mapping
    )
    route_cycle = mapping.sort_values("route_index")["cycle_index"].to_numpy(int)
    parameters: dict[str, np.ndarray] = {
        "selected_grid_index": np.asarray([SCALE_GRID.index(selected_pair)], int),
        "selected_a_cycle": np.asarray([selected_pair[0]], float),
        "selected_a_route": np.asarray([selected_pair[1]], float),
        "context_scaler_scale": scaler.scale_.copy(),
        "context_scaler_mean": scaler.mean_.copy(),
        "context_scaler_var": scaler.var_.copy(),
        "context_numeric_medians": np.asarray(
            [medians[name] for name in v3.NUMERIC_CONTROLS], float
        ),
        "mu_cycle": mu_cycle.copy(),
        "mu_route_within_cycle": mu_route.copy(),
        "route_cycle": route_cycle,
        "entropy_quartile_cutpoints": np.asarray(
            json.loads((V3_ROOT / "feature_manifest.json").read_text())[
                "entropy_quartile_cutpoints"
            ],
            float,
        ),
    }
    audit: dict[str, Any] = {
        "selected_grid_index": SCALE_GRID.index(selected_pair),
        "selected_pair": list(selected_pair),
        "training_rows": len(training),
        "training_weight": float(weights.sum()),
        "zero_fallback": selected_pair == (0.0, 0.0),
        "models": {},
    }
    if selected_pair == (0.0, 0.0):
        audit["sealed_v3_model_parameters_sha256"] = sha256(
            V3_ROOT / "model_parameters.npz"
        )
        audit["new_qhier_coefficients_stored"] = False
        return parameters, audit
    matrix = hierarchy_matrix(
        training, context, mapping, mu_cycle, mu_route, selected_pair
    )
    for target in TARGETS:
        for horizon in HORIZONS:
            key = task_key(target, horizon)
            labels = training[f"quality_class__{target}__h{horizon}"].to_numpy(int)
            model = fit_hierarchical_model(matrix, labels, weights)
            parameters[f"{key}__classes"] = model.classes_.copy()
            parameters[f"{key}__coef"] = model.coef_.copy()
            parameters[f"{key}__intercept"] = model.intercept_.copy()
            parameters[f"{key}__n_iter"] = model.n_iter_.copy()
            parameters[f"{key}__temperature"] = np.asarray([1.0])
            audit["models"][key] = {
                "feature_width": matrix.shape[1],
                "n_iter": int(model.n_iter_[0]),
            }
    audit["new_qhier_coefficients_stored"] = True
    return parameters, audit


def scaler_from_parameters(parameters: Mapping[str, np.ndarray]) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.scale_ = np.asarray(parameters["context_scaler_scale"], float).copy()
    scaler.mean_ = np.asarray(parameters["context_scaler_mean"], float).copy()
    scaler.var_ = np.asarray(parameters["context_scaler_var"], float).copy()
    scaler.n_features_in_ = len(scaler.scale_)
    scaler.n_samples_seen_ = 1
    return scaler


def predict_stored_model(
    matrix: sparse.csr_matrix, parameters: Mapping[str, np.ndarray], key: str
) -> np.ndarray:
    classes = np.asarray(parameters[f"{key}__classes"], int)
    if not np.array_equal(classes, np.asarray([0, 1, 2])):
        raise AssertionError("stored class order changed")
    logits = np.asarray(matrix @ np.asarray(parameters[f"{key}__coef"]).T)
    logits += np.asarray(parameters[f"{key}__intercept"])
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def score_with_bundle(
    panel: pd.DataFrame,
    mapping: pd.DataFrame,
    parameters: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    output = add_route_index(panel.copy(), mapping)
    selected_pair = (
        float(parameters["selected_a_cycle"][0]),
        float(parameters["selected_a_route"][0]),
    )
    if selected_pair == (0.0, 0.0):
        for target in TARGETS:
            for horizon in HORIZONS:
                probability = _sealed_zero_probabilities(output)[(target, horizon)]
                add_qhier_probability(output, target, horizon, probability)
        return output
    medians = {
        name: float(parameters["context_numeric_medians"][index])
        for index, name in enumerate(v3.NUMERIC_CONTROLS)
    }
    raw = v3.raw_context(output, medians)
    context = scaler_from_parameters(parameters).transform(raw).tocsr()
    matrix = hierarchy_matrix(
        output,
        context,
        mapping,
        np.asarray(parameters["mu_cycle"], float),
        np.asarray(parameters["mu_route_within_cycle"], float),
        selected_pair,
    )
    for target in TARGETS:
        for horizon in HORIZONS:
            probability = predict_stored_model(matrix, parameters, task_key(target, horizon))
            add_qhier_probability(output, target, horizon, probability)
    return output


def probability_column(
    model: str, target: str, horizon: int, tier: str, surface: str
) -> str:
    key = f"{model}__{target}__h{horizon}__{tier}"
    return key if surface == "conditional" else f"joint__{key}"


def surface_frame(panel: pd.DataFrame, surface: str) -> tuple[pd.DataFrame, np.ndarray]:
    if surface == "conditional":
        frame = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
        return frame, frame["conditional_weight"].to_numpy(float)
    if surface == "joint":
        frame = panel.reset_index(drop=True)
        return frame, np.ones(len(frame), float)
    raise ValueError(surface)


def cell_observed(
    frame: pd.DataFrame, surface: str, target: str, horizon: int, tier: str
) -> np.ndarray:
    if surface == "conditional":
        classes = frame[f"quality_class__{target}__h{horizon}"].to_numpy(int)
        return (classes >= (1 if tier == "p75" else 2)).astype(int)
    label = "good" if tier == "p75" else "high"
    return frame[f"joint_{label}_target__{target}__h{horizon}"].to_numpy(int)


def calibration_rows(
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
    total = float(np.asarray(weights, float).sum())
    if total <= 0.0:
        raise AssertionError("calibration surface has no weight")
    rows: list[dict[str, Any]] = []
    ece = 0.0
    supported: list[float] = []
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
        is_supported = count >= minimum_rows and weight > 0.0
        if is_supported:
            supported.append(error)
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
                "supported": is_supported,
            }
        )
    maximum = max(supported) if supported else math.nan
    return rows, float(ece), float(maximum)


def build_cell_diagnostics(
    panel: pd.DataFrame, period: str, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    models = ("qcontext", "qroute_topology", "qfull", "qhier")
    for surface in ("conditional", "joint"):
        frame, weights = surface_frame(panel, surface)
        minimum = (
            50 if surface == "conditional" else 250
        ) if mode == "oof" else (
            100 if surface == "conditional" else 500
        )
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    observed = cell_observed(frame, surface, target, horizon, tier)
                    for model in models:
                        probability = frame[
                            probability_column(model, target, horizon, tier, surface)
                        ].to_numpy(float)
                        losses = binary_losses(observed, probability)
                        rows, ece, maximum = calibration_rows(
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
                        calibration.extend(rows)
                        metric_rows.append(
                            {
                                "period": period,
                                "surface": surface,
                                "model": model,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(frame),
                                "weight": float(weights.sum()),
                                "positives": int(observed.sum()),
                                "weighted_prevalence": weighted_mean(observed, weights),
                                "log_loss": weighted_mean(losses["log_loss"], weights),
                                "brier": weighted_mean(losses["brier"], weights),
                                "ece": ece,
                                "maximum_supported_bin_error": maximum,
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(calibration)


def common_block_positions(n_sessions: int, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    if n_sessions < 1:
        raise AssertionError("bootstrap calendar is empty")
    length = min(5, n_sessions)
    block_count = n_sessions - length + 1
    needed = int(math.ceil(n_sessions / length))
    starts = np.random.Generator(np.random.PCG64(SEED)).integers(
        0, block_count, size=(draws, needed)
    )
    positions = (starts[:, :, None] + np.arange(length)[None, None, :]).reshape(
        draws, -1
    )[:, :n_sessions]
    return positions.astype(np.int32, copy=False)


def bootstrap_means(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).any(axis=0).all():
        raise AssertionError("observed bootstrap endpoint has no finite session")
    result = np.empty((len(positions), values.shape[1]), float)
    batch = 500
    for start in range(0, len(positions), batch):
        stop = min(start + batch, len(positions))
        sampled = values[positions[start:stop]]
        finite = np.isfinite(sampled)
        count = finite.sum(axis=1)
        if (count == 0).any():
            raise AssertionError("resampled bootstrap endpoint has no finite value")
        result[start:stop] = np.nansum(sampled, axis=1) / count
    return result


def _weighted_group_cell(
    frame: pd.DataFrame,
    values: np.ndarray,
    weights: np.ndarray,
    group: str,
) -> pd.Series:
    table = pd.DataFrame(
        {
            "group": frame[group].astype(str).to_numpy(),
            "weighted": np.asarray(values, float) * np.asarray(weights, float),
            "weight": np.asarray(weights, float),
        }
    ).groupby("group", sort=True).sum()
    return table["weighted"] / table["weight"]


def comparison_data(
    panel: pd.DataFrame,
    surface: str,
    candidate: str,
    baseline: str,
    loss: str,
    calendar: Sequence[str],
) -> dict[str, Any]:
    frame, weights = surface_frame(panel, surface)
    cell_differences: dict[tuple[str, int, str], np.ndarray] = {}
    baseline_losses: dict[tuple[str, int, str], np.ndarray] = {}
    daily_cells: list[pd.Series] = []
    quarter_cells: list[pd.Series] = []
    cell_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                observed = cell_observed(frame, surface, target, horizon, tier)
                candidate_probability = frame[
                    probability_column(candidate, target, horizon, tier, surface)
                ].to_numpy(float)
                baseline_probability = frame[
                    probability_column(baseline, target, horizon, tier, surface)
                ].to_numpy(float)
                candidate_loss = binary_losses(observed, candidate_probability)[loss]
                baseline_loss = binary_losses(observed, baseline_probability)[loss]
                difference = candidate_loss - baseline_loss
                key = (target, horizon, tier)
                cell_differences[key] = difference
                baseline_losses[key] = baseline_loss
                daily_cells.append(
                    _weighted_group_cell(frame, difference, weights, "session_date").rename(
                        f"{target}_{horizon}_{tier}"
                    )
                )
                quarter_cells.append(
                    _weighted_group_cell(frame, difference, weights, "quarter").rename(
                        f"{target}_{horizon}_{tier}"
                    )
                )
                cell_mean = weighted_mean(difference, weights)
                base_mean = weighted_mean(baseline_loss, weights)
                if not np.isfinite(cell_mean) or not np.isfinite(base_mean) or base_mean <= 0.0:
                    raise AssertionError("comparison cell loss is non-finite")
                cell_rows.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "tier": tier,
                        "difference": cell_mean,
                        "baseline_loss": base_mean,
                        "relative_degradation": cell_mean / base_mean,
                    }
                )
    pooled_difference = float(np.mean([row["difference"] for row in cell_rows]))
    pooled_baseline = float(np.mean([row["baseline_loss"] for row in cell_rows]))
    daily = pd.concat(daily_cells, axis=1).reindex(pd.Index(calendar, dtype=str)).mean(
        axis=1, skipna=True
    )
    quarters = pd.concat(quarter_cells, axis=1).mean(axis=1, skipna=True)
    deletion: dict[str, float] = {}
    symbols = frame["symbol_norm"].astype(str).to_numpy()
    for symbol in sorted(set(symbols)):
        keep = symbols != symbol
        deletion[symbol] = float(
            np.mean(
                [weighted_mean(values[keep], weights[keep]) for values in cell_differences.values()]
            )
        )
    target_aggregates = {
        target: float(
            np.mean(
                [
                    weighted_mean(cell_differences[(target, horizon, tier)], weights)
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
                    weighted_mean(cell_differences[(target, horizon, tier)], weights)
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
        "daily": daily.to_numpy(float),
        "quarter_differences": quarters.to_dict(),
        "leave_one_stock_out_differences": deletion,
        "cell_diagnostics": cell_rows,
        "target_aggregates": target_aggregates,
        "horizon_aggregates": horizon_aggregates,
    }


def causal_slice_diagnostics(
    panel: pd.DataFrame, period: str, mode: str
) -> pd.DataFrame:
    positive = panel.loc[panel["loop_occurs"].eq(1)].copy()
    group_specs = {
        "cycle_current_state": panel["cycle_id"].astype(str) + "@" + panel["state"].astype(str),
        "compatible_rotation_count": panel["compatible_rotation_count"].astype(str),
        "entropy_quartile": panel["entropy_quartile"].astype(str),
    }
    positive_group_specs = {
        "cycle_current_state": positive["cycle_id"].astype(str)
        + "@"
        + positive["state"].astype(str),
        "compatible_rotation_count": positive["compatible_rotation_count"].astype(str),
        "entropy_quartile": positive["entropy_quartile"].astype(str),
    }
    rule = {
        "minimum_realized_rows": 100 if mode == "oof" else 200,
        "minimum_effective_weight": 75.0 if mode == "oof" else 150.0,
        "minimum_sessions": 40 if mode == "oof" else 80,
        "minimum_stocks": 10,
        "required_quarters": 2 if mode == "oof" else 4,
    }
    rows: list[dict[str, Any]] = []
    for group_type, all_groups in group_specs.items():
        positive_groups = positive_group_specs[group_type]
        for group_value in sorted(all_groups.unique()):
            support_subset = positive.loc[positive_groups.eq(group_value)]
            effective_weight = float(support_subset["conditional_weight"].sum())
            supported = bool(
                len(support_subset) >= rule["minimum_realized_rows"]
                and effective_weight >= rule["minimum_effective_weight"]
                and support_subset["session_date"].nunique() >= rule["minimum_sessions"]
                and support_subset["symbol_norm"].nunique() >= rule["minimum_stocks"]
                and support_subset["quarter"].nunique() == rule["required_quarters"]
            )
            for surface in ("conditional", "joint"):
                if surface == "conditional":
                    frame = support_subset.reset_index(drop=True)
                    weights = frame["conditional_weight"].to_numpy(float)
                else:
                    frame = panel.loc[all_groups.eq(group_value)].reset_index(drop=True)
                    weights = np.ones(len(frame), float)
                for baseline in ("qcontext", "qroute_topology"):
                    pooled: dict[str, float] = {}
                    for loss in ("log_loss", "brier"):
                        cell_values = []
                        for target in TARGETS:
                            for horizon in HORIZONS:
                                for tier in TIERS:
                                    observed = cell_observed(
                                        frame, surface, target, horizon, tier
                                    )
                                    candidate_probability = frame[
                                        probability_column(
                                            "qhier", target, horizon, tier, surface
                                        )
                                    ].to_numpy(float)
                                    baseline_probability = frame[
                                        probability_column(
                                            baseline, target, horizon, tier, surface
                                        )
                                    ].to_numpy(float)
                                    difference = (
                                        binary_losses(observed, candidate_probability)[loss]
                                        - binary_losses(observed, baseline_probability)[loss]
                                    )
                                    value = weighted_mean(difference, weights)
                                    if not np.isfinite(value):
                                        raise AssertionError("supported slice cell is non-finite")
                                    cell_values.append(value)
                        pooled[loss] = float(np.mean(cell_values))
                    rows.append(
                        {
                            "period": period,
                            "group_type": group_type,
                            "group_value": group_value,
                            "surface": surface,
                            "candidate": "qhier",
                            "baseline": baseline,
                            "realized_rows": len(support_subset),
                            "effective_weight": effective_weight,
                            "sessions": int(support_subset["session_date"].nunique()),
                            "stocks": int(support_subset["symbol_norm"].nunique()),
                            "quarters": int(support_subset["quarter"].nunique()),
                            "supported": supported,
                            "pooled_log_loss_difference": pooled["log_loss"],
                            "pooled_brier_difference": pooled["brier"],
                            "sign_reversal": bool(
                                supported
                                and (pooled["log_loss"] > 0.0 or pooled["brier"] > 0.0)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def evaluate_core(
    panel: pd.DataFrame, period: str, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray, tuple[str, ...]]:
    metrics, calibration = build_cell_diagnostics(panel, period, mode)
    rotations = causal_slice_diagnostics(panel, period, mode)
    calendar = tuple(sorted(panel["session_date"].astype(str).unique()))
    positions = common_block_positions(len(calendar))
    comparison_payloads: dict[str, Any] = {}
    endpoint_keys: list[tuple[str, str, str]] = []
    endpoint_values: list[np.ndarray] = []
    for baseline in ("qcontext", "qroute_topology", "qfull"):
        comparison_payloads[baseline] = {}
        for surface in ("conditional", "joint"):
            comparison_payloads[baseline][surface] = {}
            for loss in ("log_loss", "brier"):
                payload = comparison_data(
                    panel, surface, "qhier", baseline, loss, calendar
                )
                comparison_payloads[baseline][surface][loss] = payload
                endpoint_keys.append((baseline, surface, loss))
                endpoint_values.append(payload["daily"])
    endpoint_matrix = np.column_stack(endpoint_values)
    resampled = bootstrap_means(endpoint_matrix, positions)
    for index, (baseline, surface, loss) in enumerate(endpoint_keys):
        quantile = 0.9875 if baseline == "qfull" else 0.99375
        comparison_payloads[baseline][surface][loss]["daily_mean"] = float(
            np.nanmean(endpoint_matrix[:, index])
        )
        comparison_payloads[baseline][surface][loss]["bootstrap_upper"] = float(
            np.quantile(resampled[:, index], quantile, method="linear")
        )

    metric_index = metrics.set_index(["surface", "model", "target", "horizon", "tier"])
    calibration_checks: dict[str, Any] = {}
    for surface in ("conditional", "joint"):
        maximum_limit = 0.02 if surface == "conditional" else 0.01
        cells = []
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    qhier = metric_index.loc[(surface, "qhier", target, horizon, tier)]
                    context = metric_index.loc[(surface, "qcontext", target, horizon, tier)]
                    route = metric_index.loc[
                        (surface, "qroute_topology", target, horizon, tier)
                    ]
                    maximum = float(qhier["maximum_supported_bin_error"])
                    checks = {
                        "ece_no_greater_than_qcontext": float(qhier["ece"])
                        <= float(context["ece"]),
                        "ece_no_greater_than_qroute": float(qhier["ece"])
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
                            "qhier_ece": float(qhier["ece"]),
                            "qcontext_ece": float(context["ece"]),
                            "qroute_ece": float(route["ece"]),
                            "qhier_maximum_supported_bin_error": maximum,
                            "checks": checks,
                            "pass": bool(all(checks.values())),
                        }
                    )
        calibration_checks[surface] = {
            "cells": cells,
            "pass": bool(all(cell["pass"] for cell in cells)),
        }

    primary: dict[str, Any] = {}
    thresholds = {
        "qcontext": {"conditional": 0.005, "joint": 0.0025},
        "qroute_topology": {"conditional": 0.001, "joint": 0.0005},
    }
    for baseline in ("qcontext", "qroute_topology"):
        primary[baseline] = {}
        for surface in ("conditional", "joint"):
            log = comparison_payloads[baseline][surface]["log_loss"]
            brier = comparison_payloads[baseline][surface]["brier"]
            slice_rows = rotations.loc[
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
                "calibration": calibration_checks[surface]["pass"],
                "supported_slices_exist": len(slice_rows) > 0,
                "no_supported_slice_sign_reversal": bool(
                    len(slice_rows) > 0 and not slice_rows["sign_reversal"].any()
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
        for loss in ("log_loss", "brier"):
            payload = comparison_payloads["qfull"][surface][loss]
            context_loss = float(
                metrics.loc[
                    metrics["surface"].eq(surface)
                    & metrics["model"].eq("qcontext"),
                    loss,
                ].mean()
            )
            full_loss = float(
                metrics.loc[
                    metrics["surface"].eq(surface) & metrics["model"].eq("qfull"),
                    loss,
                ].mean()
            )
            gain = context_loss - full_loss
            margin = 0.1 * gain if gain > 0.0 else math.nan
            secondary[surface][loss] = {
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
    gates = {
        "period": period,
        "support_pass": bool(v3.v3_support(panel)["support_pass"])
        if mode == "oof"
        else float(
            panel.loc[panel["loop_occurs"].eq(1), "conditional_weight"].sum()
        )
        >= 25000.0,
        "calibration": calibration_checks,
        "primary": primary,
        "primary_without_falsification_pass": bool(
            (
                bool(v3.v3_support(panel)["support_pass"])
                if mode == "oof"
                else float(
                    panel.loc[panel["loop_occurs"].eq(1), "conditional_weight"].sum()
                )
                >= 25000.0
            )
            and all(
                primary[baseline][surface]["pass"]
                for baseline in ("qcontext", "qroute_topology")
                for surface in ("conditional", "joint")
            )
        ),
        "secondary_qfull_noninferiority": secondary,
        "secondary_pass": bool(
            all(secondary[surface][loss]["pass"] for surface in ("conditional", "joint") for loss in ("log_loss", "brier"))
        ),
    }
    return metrics, calibration, rotations, gates, positions, calendar


def falsification_diagnostics(panel: pd.DataFrame) -> dict[str, Any]:
    movement_columns = [
        (target, horizon, tier)
        for target in TARGETS
        for horizon in HORIZONS
        for tier in TIERS
    ]
    anchor_columns = [
        "anchor_id",
        "symbol_norm",
        "quarter",
        "session_date",
        "start_timestamp",
        *[f"quality_class__{target}__h{horizon}" for target in TARGETS for horizon in HORIZONS],
    ]
    anchors = panel[anchor_columns].drop_duplicates("anchor_id", keep="first").copy()
    duplicate_check = panel.groupby("anchor_id", sort=False)[
        [f"quality_class__{target}__h{horizon}" for target in TARGETS for horizon in HORIZONS]
    ].nunique()
    if (duplicate_check > 1).any().any():
        raise AssertionError("movement vector differs across rows of an anchor")
    anchors = anchors.sort_values(
        ["symbol_norm", "quarter", "session_date", "start_timestamp", "anchor_id"],
        kind="stable",
    ).reset_index(drop=True)
    if anchors.duplicated(
        ["symbol_norm", "quarter", "session_date", "start_timestamp", "anchor_id"]
    ).any():
        raise AssertionError("duplicate canonical anchor ordering key")
    anchor_position = pd.Series(np.arange(len(anchors)), index=anchors["anchor_id"])
    row_anchor = panel["anchor_id"].map(anchor_position).to_numpy(int)
    observed_labels = np.empty((len(anchors), len(movement_columns)), np.int8)
    for index, (target, horizon, tier) in enumerate(movement_columns):
        classes = anchors[f"quality_class__{target}__h{horizon}"].to_numpy(int)
        observed_labels[:, index] = classes >= (1 if tier == "p75" else 2)
    strata: list[np.ndarray] = []
    for _, positions in anchors.groupby(["symbol_norm", "quarter"], sort=True).groups.items():
        indices = np.asarray(positions, int)
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
            mask = np.ones(len(panel), bool)
            weights = np.ones(len(panel), float)
        frame = panel.loc[mask].reset_index(drop=True)
        anchor_indices = row_anchor[mask]
        occurrence = frame["loop_occurs"].to_numpy(float)
        for model in models:
            cell_constant = np.empty(len(movement_columns), float)
            cell_coefficient = np.empty((len(anchors), len(movement_columns)), float)
            for index, (target, horizon, tier) in enumerate(movement_columns):
                probability = np.clip(
                    frame[probability_column(model, target, horizon, tier, surface)].to_numpy(float),
                    EPSILON,
                    1.0 - EPSILON,
                )
                base = -np.log(1.0 - probability)
                slope = np.log((1.0 - probability) / probability)
                if surface == "joint":
                    slope = slope * occurrence
                denominator = float(weights.sum())
                cell_constant[index] = float(np.dot(weights, base) / denominator)
                cell_coefficient[:, index] = np.bincount(
                    anchor_indices,
                    weights=weights * slope,
                    minlength=len(anchors),
                ) / denominator
            constants[(surface, model)] = cell_constant
            coefficients[(surface, model)] = cell_coefficient

    # The expression above deliberately builds one anchor coefficient per cell;
    # assert the linearized loss exactly replays the observed direct statistic.
    def statistics(labels: np.ndarray) -> np.ndarray:
        losses: dict[tuple[str, str], float] = {}
        for surface in surfaces:
            for model in models:
                per_cell = constants[(surface, model)] + np.sum(
                    coefficients[(surface, model)] * labels, axis=0
                )
                losses[(surface, model)] = float(np.mean(per_cell))
        values = []
        for baseline in ("qcontext", "qroute_topology"):
            for surface in surfaces:
                baseline_loss = losses[(surface, baseline)]
                values.append(
                    (baseline_loss - losses[(surface, "qhier")]) / baseline_loss
                )
        return np.asarray(values, float)

    observed = statistics(observed_labels)
    direct = []
    for baseline in ("qcontext", "qroute_topology"):
        for surface in surfaces:
            frame, weights = surface_frame(panel, surface)
            candidate_cells = []
            baseline_cells = []
            for target, horizon, tier in movement_columns:
                y = cell_observed(frame, surface, target, horizon, tier)
                candidate_probability = frame[
                    probability_column("qhier", target, horizon, tier, surface)
                ].to_numpy(float)
                baseline_probability = frame[
                    probability_column(baseline, target, horizon, tier, surface)
                ].to_numpy(float)
                candidate_cells.append(
                    weighted_mean(binary_losses(y, candidate_probability)["log_loss"], weights)
                )
                baseline_cells.append(
                    weighted_mean(binary_losses(y, baseline_probability)["log_loss"], weights)
                )
            candidate_loss = float(np.mean(candidate_cells))
            baseline_loss = float(np.mean(baseline_cells))
            direct.append((baseline_loss - candidate_loss) / baseline_loss)
    direct_values = np.asarray(direct, float)
    replay_error = float(np.max(np.abs(observed - direct_values)))
    if replay_error > 1e-12:
        raise AssertionError("falsification linearized statistic failed direct replay")
    rng = np.random.Generator(np.random.PCG64(SEED))
    null = np.empty((NULL_DRAWS, 4), float)
    for draw_zero in range(NULL_DRAWS):
        transformed = observed_labels.copy()
        draw_number = draw_zero + 1
        for indices in strata:
            original = observed_labels[indices]
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
    p_values = (1 + (null >= observed[None, :]).sum(axis=0)) / (NULL_DRAWS + 1)
    names = (
        "qhier_vs_qcontext__conditional",
        "qhier_vs_qcontext__joint",
        "qhier_vs_qroute_topology__conditional",
        "qhier_vs_qroute_topology__joint",
    )
    details = {
        name: {
            "observed_relative_log_loss_improvement": float(observed[index]),
            "null_mean": float(null[:, index].mean()),
            "null_q95": float(np.quantile(null[:, index], 0.95, method="linear")),
            "p_value": float(p_values[index]),
            "pass": bool(p_values[index] <= 0.01),
        }
        for index, name in enumerate(names)
    }
    return {
        "draws": NULL_DRAWS,
        "seed": SEED,
        "anchors": len(anchors),
        "strata": len(strata),
        "statistics": details,
        "observed_direct_replay_max_error": replay_error,
        "pass": bool(all(item["pass"] for item in details.values())),
    }


def _qhier_as_v1_candidate(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                output[f"qcycle__{target}__h{horizon}__{tier}"] = output[
                    f"qhier__{target}__h{horizon}__{tier}"
                ].to_numpy(float)
                output[f"joint__qcycle__{target}__h{horizon}__{tier}"] = output[
                    f"joint__qhier__{target}__h{horizon}__{tier}"
                ].to_numpy(float)
    return output


def _component_daily_series(
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
        frame, weights = surface_frame(cycle, surface)
        observed = cell_observed(frame, surface, target, horizon, tier)
        candidate_probability = frame[
            probability_column("qhier", target, horizon, tier, surface)
        ].to_numpy(float)
        context_probability = frame[
            probability_column("qcontext", target, horizon, tier, surface)
        ].to_numpy(float)
        for loss in ("log_loss", "brier"):
            difference = (
                binary_losses(observed, candidate_probability)[loss]
                - binary_losses(observed, context_probability)[loss]
            )
            daily = _weighted_group_cell(
                frame, difference, weights, "session_date"
            ).reindex(pd.Index(calendar, dtype=str))
            components.append((f"{target}__{surface}__{loss}", "negative", daily.to_numpy(float)))
    frame, weights = surface_frame(cycle, "conditional")
    observed = cell_observed(frame, "conditional", target, horizon, tier)
    context_probability = frame[
        probability_column("qcontext", target, horizon, tier, "conditional")
    ].to_numpy(float)
    lift = observed - context_probability
    daily_lift = _weighted_group_cell(frame, lift, weights, "session_date").reindex(
        pd.Index(calendar, dtype=str)
    )
    components.append((f"{target}__conditional__lift", "positive", daily_lift.to_numpy(float)))
    return components


def centered_component_p_value(
    values: np.ndarray, alternative: str, positions: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(values, float)
    if not np.isfinite(values).any():
        raise AssertionError("named component has no finite observed session")
    observed = float(np.nanmean(values))
    centered = values - observed
    null = bootstrap_means(centered, positions)[:, 0]
    if alternative == "negative":
        p_value = (1 + int((null <= observed).sum())) / (len(null) + 1)
    elif alternative == "positive":
        p_value = (1 + int((null >= observed).sum())) / (len(null) + 1)
    else:
        raise ValueError(alternative)
    return observed, float(p_value)


def holm_rejections(
    frame: pd.DataFrame, p_column: str, alpha: float = 0.025
) -> pd.Series:
    order = frame.sort_values(
        [p_column, "cycle_index", "horizon"], kind="stable"
    ).index
    rejected = pd.Series(False, index=frame.index)
    stopped = False
    total = len(frame)
    for rank, index in enumerate(order, start=1):
        if stopped:
            continue
        if float(frame.loc[index, p_column]) <= alpha / (total - rank + 1):
            rejected.loc[index] = True
        else:
            stopped = True
    return rejected


def named_cycle_diagnostics(
    panel: pd.DataFrame,
    rotations: pd.DataFrame,
    positions: np.ndarray,
    calendar: Sequence[str],
    global_primary_pass: bool,
    mode: str,
) -> pd.DataFrame:
    mapping = route_mapping()
    v1 = _load_module(V1_SOURCE, "sealed_v1_named_grade_dependency")
    contract = json.loads((HERE / "contracts/20260710-per-loop-movement-quality-v1.json").read_text())
    candidate = _qhier_as_v1_candidate(panel)
    eligibility = None
    if mode == "oof":
        support_path = PARENT_ROOT / "provisional_support_2024.csv"
        if sha256(support_path) != V1_PROVISIONAL_SUPPORT_SHA256:
            raise AssertionError("V1 fit-eligibility artifact changed")
        support = pd.read_csv(support_path)
        if len(support) != 20 or not support["full_2024_fit_eligible"].astype(bool).all():
            raise AssertionError("frozen V1 fit eligibility is not all twenty cycles")
        eligibility = support.set_index("cycle_id")["full_2024_fit_eligible"].astype(bool).to_dict()
    v1_grades = v1.grade_period(
        candidate,
        "2024_oof" if mode == "oof" else str(panel["quarter"].iloc[0])[:4],
        mode,
        contract,
        full_2024_eligibility=eligibility,
    )
    horizons = v1_grades["horizons"].set_index(["cycle_id", "horizon"])
    cycle_grade = v1_grades["cycles"].set_index("cycle_id")["global_grade"]
    cycles = pd.read_csv(PARENT_ROOT / "fixed_cycles.csv").sort_values("cycle_index")
    rows: list[dict[str, Any]] = []
    for _, cycle in cycles.iterrows():
        cycle_id = str(cycle["cycle_id"])
        required = mapping.loc[mapping["cycle_id"].eq(cycle_id)]
        orientation_pass = True
        for _, unit in required.iterrows():
            value = f"{cycle_id}@{int(unit.current_state)}"
            unit_rows = rotations.loc[
                rotations["group_type"].eq("cycle_current_state")
                & rotations["group_value"].eq(value)
            ]
            if len(unit_rows) != 4 or not unit_rows["supported"].all() or unit_rows["sign_reversal"].any():
                orientation_pass = False
                break
        for horizon in HORIZONS:
            row: dict[str, Any] = {
                "cycle_index": int(cycle["cycle_index"]),
                "cycle_id": cycle_id,
                "cycle": str(cycle["cycle"]),
                "horizon": horizon,
                "global_primary_pass": global_primary_pass,
                "required_orientation_count": len(required),
                "orientation_pass": orientation_pass,
            }
            for tier in TIERS:
                component_p: list[float] = []
                component_payload: dict[str, Any] = {}
                if global_primary_pass:
                    for target in TARGETS:
                        for name, alternative, values in _component_daily_series(
                            panel, cycle_id, target, horizon, tier, calendar
                        ):
                            observed, p_value = centered_component_p_value(
                                values, alternative, positions
                            )
                            component_payload[name] = {
                                "observed": observed,
                                "alternative": alternative,
                                "p_value": p_value,
                            }
                            component_p.append(p_value)
                    row[f"{tier}_unit_p_value"] = max(component_p)
                else:
                    component_payload["not_run"] = "global_primary_precondition_failed"
                    row[f"{tier}_unit_p_value"] = 1.0
                row[f"{tier}_component_json"] = json.dumps(
                    safe(component_payload), sort_keys=True, separators=(",", ":")
                )
            grade = str(horizons.loc[(cycle_id, horizon), "grade"])
            global_grade = str(cycle_grade.loc[cycle_id])
            row["v1_substitution_grade"] = grade
            row["v1_substitution_global_grade"] = global_grade
            row["good_point_pass"] = grade in (
                "good_movement_quality",
                "high_movement_quality",
            ) and global_grade in ("good_movement_quality", "high_movement_quality")
            row["high_point_pass"] = (
                grade == "high_movement_quality"
                and global_grade == "high_movement_quality"
            )
            rows.append(row)
    result = pd.DataFrame(rows).sort_values(["cycle_index", "horizon"], kind="stable")
    if len(result) != 60:
        raise AssertionError("named hypothesis count changed")
    if global_primary_pass:
        result["good_holm_pass"] = holm_rejections(result, "p75_unit_p_value")
        high_for_holm = result["p90_unit_p_value"].copy()
        gate = (
            result["good_point_pass"]
            & result["good_holm_pass"]
            & result["orientation_pass"]
        )
        high_for_holm.loc[~gate] = 1.0
        result["high_gatekept_unit_p_value"] = high_for_holm
        result["high_holm_pass"] = holm_rejections(
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
    high = (
        good
        & result["high_point_pass"]
        & result["high_holm_pass"]
    )
    result["development_label"] = np.where(
        high,
        "development_high_candidate",
        np.where(good, "development_good_candidate", "development_unqualified"),
    )
    global_labels: dict[str, str] = {}
    for cycle_id, group in result.groupby("cycle_id", sort=True):
        labels = tuple(group["development_label"].astype(str))
        if all(label == "development_high_candidate" for label in labels):
            global_labels[cycle_id] = "development_high_candidate"
        elif all(label != "development_unqualified" for label in labels) and any(
            label == "development_good_candidate" for label in labels
        ):
            global_labels[cycle_id] = "development_good_candidate"
        else:
            global_labels[cycle_id] = "development_unqualified"
    result["global_development_label"] = result["cycle_id"].map(global_labels)
    result["parent_grade_changed"] = False
    result["prospective_validated"] = False
    return result


def validate_probability_bundle(panel: pd.DataFrame) -> dict[str, float]:
    errors: dict[str, float] = {}
    structural = panel["loop_probability"].to_numpy(float)
    for model in ("qcontext", "qroute_topology", "qfull", "qhier"):
        maximum = 0.0
        for target in TARGETS:
            for horizon in HORIZONS:
                p75 = panel[f"{model}__{target}__h{horizon}__p75"].to_numpy(float)
                p90 = panel[f"{model}__{target}__h{horizon}__p90"].to_numpy(float)
                if (
                    not np.isfinite(p75).all()
                    or not np.isfinite(p90).all()
                    or (p90 > p75 + EPSILON).any()
                    or p90.min() < -EPSILON
                    or p75.max() > 1.0 + EPSILON
                ):
                    raise AssertionError(f"invalid probability bundle for {model}")
                for tier, conditional in (("p75", p75), ("p90", p90)):
                    joint = panel[
                        f"joint__{model}__{target}__h{horizon}__{tier}"
                    ].to_numpy(float)
                    maximum = max(maximum, float(np.max(np.abs(joint - structural * conditional))))
        if maximum > 1e-12:
            raise AssertionError(f"joint-chain replay failed for {model}")
        errors[model] = maximum
    return errors


def feature_manifest(
    medians: Mapping[str, float], selected_pair: tuple[float, float]
) -> dict[str, Any]:
    return {
        "task_key_template": "qhier__{target}__h{horizon}",
        "targets": list(TARGETS),
        "horizons": list(HORIZONS),
        "context_width": CONTEXT_WIDTH,
        "topology_width": TOPOLOGY_WIDTH,
        "cycle_width": CYCLE_WIDTH,
        "compatible_cycle_state_width": ROUTE_WIDTH,
        "nonzero_total_width": HIERARCHY_WIDTH,
        "numeric_controls": list(v3.NUMERIC_CONTROLS),
        "numeric_medians": dict(medians),
        "topology_columns": list(v3.TOPOLOGY_COLUMNS),
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
            "C": MODEL_C,
            "solver": "lbfgs",
            "max_iter": MODEL_MAX_ITER,
            "tol": MODEL_TOL,
            "random_state": SEED,
            "temperature": 1.0,
        },
        "zero_pair": "exact sealed V3 qroute_topology; no V4 coefficients",
        "v1_named_grade_runner_sha256": V1_SOURCE_SHA256,
        "v1_provisional_support_sha256": V1_PROVISIONAL_SUPPORT_SHA256,
        "future_realized_feature_used": False,
        "stock_identity_feature_used": False,
        "provider_volume_used": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def fit_artifact_names() -> tuple[str, ...]:
    return (
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


def file_hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256(root / name) for name in names}


def run_fit_only() -> dict[str, Any]:
    _, source_hashes = validate_contract_and_sources()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            "fit-only requires a pristine artifact root; archive or explicitly remove the old bundle"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "fit_source_hashes_pre_fit.json", source_hashes)
    mapping = route_mapping()
    oof = load_oof_2024(mapping)
    training = load_training_2024(mapping, oof)
    support = v3.v3_support(oof)
    if not support["support_pass"]:
        raise AssertionError("frozen V3 unique-cohort support failed")
    zero_replay = validate_zero_pair_replay(oof)
    medians = v3.load_numeric_medians()
    statistics, candidate_audit = generate_candidate_loss_statistics(
        training, mapping, medians
    )
    inner_selection, selections, selected_full = build_inner_selection_table(statistics)
    oof_predictions, outer_audit = fit_outer_predictions(
        training, oof, mapping, medians, selections
    )
    probability_replay = validate_probability_bundle(oof_predictions)
    metrics, calibration, rotations, gates, positions, calendar = evaluate_core(
        oof_predictions, "2024_oof", "oof"
    )
    falsification = falsification_diagnostics(oof_predictions)
    global_pass = bool(
        gates["support_pass"]
        and gates["primary_without_falsification_pass"]
        and falsification["pass"]
    )
    gates["falsification"] = falsification
    gates["primary_algorithm_pass"] = global_pass
    gates["primary_algorithm_label"] = (
        "development_algorithm_supported"
        if global_pass
        else "development_algorithm_unconfirmed"
    )
    per_loop = named_cycle_diagnostics(
        oof_predictions,
        rotations,
        positions,
        calendar,
        global_pass,
        "oof",
    )
    parameters, full_fit_audit = fit_full_model_bundle(
        training, mapping, medians, selected_full
    )
    current_hashes = validate_contract_and_sources()[1]
    if current_hashes != source_hashes:
        raise AssertionError("source changed during fit-only phase")

    mapping.to_csv(OUT / "route_mapping.csv", index=False)
    candidate_audit.to_csv(OUT / "candidate_fit_audit_2024.csv", index=False)
    inner_selection.to_csv(OUT / "inner_selection_2024.csv", index=False)
    outer_audit.to_csv(OUT / "outer_fold_audit_2024.csv", index=False)
    oof_predictions.to_parquet(OUT / "oof_predictions_2024.parquet", index=False)
    np.savez_compressed(OUT / "model_parameters.npz", **parameters)
    metrics.to_csv(OUT / "cell_diagnostics_2024.csv", index=False)
    calibration.to_csv(OUT / "calibration_diagnostics_2024.csv", index=False)
    rotations.to_csv(OUT / "rotation_diagnostics_2024.csv", index=False)
    per_loop.to_csv(OUT / "per_loop_grades_2024.csv", index=False)
    write_json(OUT / "feature_manifest.json", feature_manifest(medians, selected_full))
    write_json(
        OUT / "fold_schedule.json",
        {
            "outer_months": list(OUTER_MONTHS),
            "inner_validation_schedule": INNER_SCHEDULE,
            "final_selection_months": list(FINAL_SELECTION_MONTHS),
            "strict_training_rule": "month_key < validation_month",
        },
    )
    write_json(
        OUT / "hyperparameter_grid.json",
        {
            "ordered_pairs": [list(pair) for pair in SCALE_GRID],
            "pair_order": ["a_cycle", "a_route"],
            "tie_tolerance": 1e-6,
            "tie_break": ["a_route", "a_cycle"],
            "shared_across_all_six_tasks": True,
            "model_C": MODEL_C,
            "model_max_iter": MODEL_MAX_ITER,
            "model_tol": MODEL_TOL,
            "seed": SEED,
        },
    )
    write_json(OUT / "full_fit_audit.json", full_fit_audit)
    write_json(OUT / "support_2024.json", support)
    write_json(OUT / "falsification_diagnostics_2024.json", falsification)
    write_json(OUT / "algorithm_gates_2024.json", gates)
    provisional = {
        "primary_algorithm_label": gates["primary_algorithm_label"],
        "primary_algorithm_pass": global_pass,
        "selected_full_grid_index": SCALE_GRID.index(selected_full),
        "selected_full_pair": list(selected_full),
        "development_high_candidate_horizons": int(
            per_loop["development_label"].eq("development_high_candidate").sum()
        ),
        "development_good_candidate_horizons": int(
            per_loop["development_label"].eq("development_good_candidate").sum()
        ),
        "parent_grade_changed": False,
        "prospective_validated": False,
    }
    write_json(OUT / "provisional_decision.json", provisional)
    write_json(OUT / "fit_source_hashes.json", source_hashes)
    artifacts = file_hashes(OUT, fit_artifact_names())
    fit_complete = {
        "status": "fit_frozen_pending_independent_pre_score_audit",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "support_pass": bool(support["support_pass"]),
        "oof_compatible_rows": len(oof_predictions),
        "oof_realized_rows": int(oof_predictions["loop_occurs"].sum()),
        "oof_effective_weight": float(
            oof_predictions.loc[
                oof_predictions["loop_occurs"].eq(1), "conditional_weight"
            ].sum()
        ),
        "zero_pair_chain_replay_max_error": zero_replay,
        "probability_chain_replay_max_error": probability_replay,
        "selected_full_grid_index": SCALE_GRID.index(selected_full),
        "selected_full_pair": list(selected_full),
        "provisional_decision": provisional,
        "later_period_panels_read": False,
        "later_scoring_authorized": False,
        "scoring_authorized": False,
        "parent_grade_changed": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "artifact_hashes": artifacts,
    }
    write_json(OUT / "fit_complete.json", fit_complete)
    return fit_complete


def validate_only() -> dict[str, Any]:
    _, source_hashes = validate_contract_and_sources()
    mapping = route_mapping()
    oof = load_oof_2024(mapping)
    training = load_training_2024(mapping, oof)
    support = v3.v3_support(oof)
    weights = training["conditional_weight"].to_numpy(float)
    mu_cycle, mu_route = weighted_centers(
        training["cycle_index"], training["route_index"], weights, mapping
    )
    return {
        "status": "validated_without_fit",
        "contract_sha256": CONTRACT_SHA256,
        "source_hash_count": len(source_hashes),
        "route_mapping_rows": len(mapping),
        "scale_pairs": len(SCALE_GRID),
        "training_rows": len(training),
        "oof_rows": len(oof),
        "support": support,
        "zero_pair_chain_replay_max_error": validate_zero_pair_replay(oof),
        "mu_cycle_sum": float(mu_cycle.sum()),
        "maximum_within_cycle_route_center_sum_error": float(
            max(
                abs(
                    mu_route[
                        mapping.sort_values("route_index")["cycle_index"].to_numpy(int)
                        == cycle
                    ].sum()
                    - 1.0
                )
                for cycle in range(CYCLE_WIDTH)
            )
        ),
        "later_period_panels_read": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def load_parameters() -> dict[str, np.ndarray]:
    with np.load(OUT / "model_parameters.npz") as stored:
        return {name: stored[name].copy() for name in stored.files}


def validate_fit_and_pre_score_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    fit_path = OUT / "fit_complete.json"
    audit_path = OUT / "pre_score_audit.json"
    if not fit_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("fit freeze and independent pre-score audit are required")
    fit = json.loads(fit_path.read_text())
    audit = json.loads(audit_path.read_text())
    if fit["contract_sha256"] != CONTRACT_SHA256:
        raise AssertionError("fit contract hash changed")
    if fit["runner_sha256"] != sha256(Path(__file__)):
        raise AssertionError("runner changed after fit freeze")
    for name, expected in fit["artifact_hashes"].items():
        if sha256(OUT / name) != expected:
            raise AssertionError(f"fit artifact changed after freeze: {name}")
    frozen_sources = json.loads((OUT / "fit_source_hashes.json").read_text())
    if validate_contract_and_sources()[1] != frozen_sources:
        raise AssertionError("fit sources changed after freeze")
    audit_pass = audit.get("all_passed") is True
    if not audit_pass or audit.get("scoring_authorized") is not True:
        raise AssertionError("independent pre-score audit did not authorize scoring")
    if audit.get("contract_sha256") != CONTRACT_SHA256:
        raise AssertionError("independent audit contract mismatch")
    if audit.get("runner_sha256") != sha256(Path(__file__)):
        raise AssertionError("independent audit runner mismatch")
    if audit.get("fit_complete_sha256") != sha256(fit_path):
        raise AssertionError("independent audit does not bind the current fit marker")
    if audit.get("fit_artifact_hashes") != fit["artifact_hashes"]:
        raise AssertionError("independent audit artifact manifest mismatch")
    return fit, audit


def scoring_sources_after_lock() -> dict[str, Path]:
    return {
        "2025": V3_ROOT / "scoring_predictions_2025.parquet",
        "2023": V3_ROOT / "scoring_predictions_2023.parquet",
    }


def validate_scoring_source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    v3_complete = json.loads((V3_ROOT / "scoring_complete.json").read_text())
    hashes: dict[str, str] = {}
    for period, path in paths.items():
        expected = v3_complete["artifact_hashes"][f"scoring_predictions_{period}.parquet"]
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"sealed V3 scoring panel changed for {period}")
        hashes[f"scoring_predictions_{period}.parquet"] = actual
    return hashes


def scoring_phase_output_names() -> tuple[str, ...]:
    period_names = tuple(
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
    )
    return (
        *period_names,
        "evaluation_source_hashes.json",
        "period_transfer_gates.json",
        "scoring_complete.json",
        "summary.json",
        "independent_artifact_audit.json",
    )


def run_scoring() -> dict[str, Any]:
    fit, audit = validate_fit_and_pre_score_lock()
    stale = [name for name in scoring_phase_output_names() if (OUT / name).exists()]
    if stale:
        raise FileExistsError(
            f"score-only requires a pristine scoring phase; stale outputs: {stale}"
        )
    paths = scoring_sources_after_lock()
    evaluation_hashes = validate_scoring_source_hashes(paths)
    mapping = route_mapping()
    parameters = load_parameters()
    provisional = json.loads((OUT / "provisional_decision.json").read_text())
    provisional_grades = pd.read_csv(OUT / "per_loop_grades_2024.csv")
    period_payloads: dict[str, Any] = {}
    later_grades: dict[str, pd.DataFrame] = {}
    scoring_artifacts: list[str] = []
    for period in ("2025", "2023"):
        sealed_panel = pd.read_parquet(paths[period])
        panel = score_with_bundle(sealed_panel, mapping, parameters)
        replay = validate_probability_bundle(panel)
        metrics, calibration, rotations, gates, positions, calendar = evaluate_core(
            panel, period, "scoring"
        )
        falsification = falsification_diagnostics(panel)
        period_pass = bool(
            gates["support_pass"]
            and gates["primary_without_falsification_pass"]
            and falsification["pass"]
        )
        gates["falsification"] = falsification
        gates["primary_algorithm_pass"] = period_pass
        gates["primary_algorithm_label"] = (
            "development_algorithm_supported"
            if period_pass
            else "development_algorithm_unconfirmed"
        )
        gates["probability_chain_replay_max_error"] = replay
        grades = named_cycle_diagnostics(
            panel, rotations, positions, calendar, period_pass, "scoring"
        )
        later_grades[period] = grades
        panel.to_parquet(OUT / f"scoring_predictions_{period}.parquet", index=False)
        metrics.to_csv(OUT / f"cell_diagnostics_{period}.csv", index=False)
        calibration.to_csv(OUT / f"calibration_diagnostics_{period}.csv", index=False)
        rotations.to_csv(OUT / f"rotation_diagnostics_{period}.csv", index=False)
        grades.to_csv(OUT / f"per_loop_grades_{period}.csv", index=False)
        write_json(OUT / f"support_{period}.json", {
            "effective_weight": float(panel.loc[panel["loop_occurs"].eq(1), "conditional_weight"].sum()),
            "minimum_effective_weight": 25000.0,
            "support_pass": gates["support_pass"],
        })
        write_json(OUT / f"algorithm_gates_{period}.json", gates)
        period_payloads[period] = {
            "primary_algorithm_pass": period_pass,
            "primary_algorithm_label": gates["primary_algorithm_label"],
        }
        scoring_artifacts.extend(
            [
                f"scoring_predictions_{period}.parquet",
                f"support_{period}.json",
                f"cell_diagnostics_{period}.csv",
                f"calibration_diagnostics_{period}.csv",
                f"rotation_diagnostics_{period}.csv",
                f"algorithm_gates_{period}.json",
                f"per_loop_grades_{period}.csv",
            ]
        )
    transfer_rows = []
    label_rank = {
        "development_unqualified": 0,
        "development_good_candidate": 1,
        "development_high_candidate": 2,
    }
    rank_label = {value: key for key, value in label_rank.items()}
    for _, row in provisional_grades.iterrows():
        key = (int(row["cycle_index"]), int(row["horizon"]))
        base = str(row["development_label"])
        later = {
            period: str(
                frame.set_index(["cycle_index", "horizon"]).loc[key, "development_label"]
            )
            for period, frame in later_grades.items()
        }
        all_labels = (base, later["2025"], later["2023"])
        if any(value not in label_rank for value in all_labels):
            raise AssertionError("unknown named development label")
        final_label = rank_label[min(label_rank[value] for value in all_labels)]
        portable = final_label != "development_unqualified"
        transfer_rows.append(
            {
                "cycle_index": key[0],
                "cycle_id": row["cycle_id"],
                "horizon": key[1],
                "provisional_label": base,
                "development_2025_label": later["2025"],
                "backward_2023_label": later["2023"],
                "final_development_portability_label": final_label,
                "development_portable": portable,
                "provisional_label_retained": final_label == base,
                "later_promotion_performed": False,
                "parent_grade_changed": False,
                "prospective_validated": False,
            }
        )
    global_transfer_rows = []
    for cycle_id, base_group in provisional_grades.groupby("cycle_id", sort=True):
        base = str(base_group["global_development_label"].iloc[0])
        later = {
            period: str(
                frame.loc[
                    frame["cycle_id"].eq(cycle_id), "global_development_label"
                ].iloc[0]
            )
            for period, frame in later_grades.items()
        }
        all_labels = (base, later["2025"], later["2023"])
        if any(value not in label_rank for value in all_labels):
            raise AssertionError("unknown global named development label")
        final_label = rank_label[min(label_rank[value] for value in all_labels)]
        global_transfer_rows.append(
            {
                "cycle_id": cycle_id,
                "provisional_label": base,
                "development_2025_label": later["2025"],
                "backward_2023_label": later["2023"],
                "final_development_portability_label": final_label,
                "development_portable": final_label != "development_unqualified",
                "provisional_label_retained": final_label == base,
                "later_promotion_performed": False,
                "parent_grade_changed": False,
                "prospective_validated": False,
            }
        )
    transfer = {
        "provisional_algorithm_label": provisional["primary_algorithm_label"],
        "periods": period_payloads,
        "algorithm_development_portable": bool(
            provisional["primary_algorithm_pass"]
            and all(payload["primary_algorithm_pass"] for payload in period_payloads.values())
        ),
        "later_promotion_performed": False,
        "named_transfer": transfer_rows,
        "global_named_transfer": global_transfer_rows,
        "prospective_validated": False,
    }
    write_json(OUT / "evaluation_source_hashes.json", evaluation_hashes)
    write_json(OUT / "period_transfer_gates.json", transfer)
    scoring_artifacts.extend(["evaluation_source_hashes.json", "period_transfer_gates.json"])
    scoring_complete = {
        "status": "scoring_complete_development_and_backward_portability_only",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "pre_score_audit_sha256": sha256(OUT / "pre_score_audit.json"),
        "pre_score_audit_passed": True,
        "transfer": transfer,
        "later_periods_are_prospective": False,
        "parent_grade_changed": False,
        "shadow_tree_read": False,
        "shadow_tree_written": False,
        "artifact_hashes": file_hashes(OUT, scoring_artifacts),
    }
    write_json(OUT / "scoring_complete.json", scoring_complete)
    summary = {
        **scoring_complete,
        "fit_complete_sha256": sha256(OUT / "fit_complete.json"),
        "interpretation": "Movement magnitude and future range only; no directional, economic, ordering, or deployment claim.",
    }
    write_json(OUT / "summary.json", summary)
    return summary


def self_tests() -> dict[str, Any]:
    checks = {
        "contract_hash": sha256(CONTRACT) == CONTRACT_SHA256,
        "v3_source_hash": sha256(V3_SOURCE) == V3_SOURCE_SHA256,
        "v1_source_hash": sha256(V1_SOURCE) == V1_SOURCE_SHA256,
        "scale_grid": len(SCALE_GRID) == 15
        and all(pair == (0.0, 0.0) or 0.0 < pair[1] <= pair[0] for pair in SCALE_GRID),
        "width": CONTEXT_WIDTH + TOPOLOGY_WIDTH + CYCLE_WIDTH + ROUTE_WIDTH
        == HIERARCHY_WIDTH,
        "schedule": tuple(INNER_SCHEDULE) == OUTER_MONTHS,
        "tie_break": choose_scale_pair(
            {pair: (0.0 if pair in ((0.0, 0.0), (0.125, 0.0625)) else 1.0) for pair in SCALE_GRID}
        )
        == (0.0, 0.0),
        "safety": json.loads(CONTRACT.read_text())["research_only"] is True
        and json.loads(CONTRACT.read_text())["live_ordering_enabled"] is False,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"checks": checks, "passed": len(checks), "total": len(checks)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum((args.validate_only, args.fit_only, args.score_only)) > 1:
        raise SystemExit("choose exactly one phase")
    tests = self_tests()
    if args.validate_only:
        result = validate_only()
    elif args.fit_only:
        result = run_fit_only()
    elif args.score_only:
        result = run_scoring()
    else:
        raise SystemExit("choose --validate-only, --fit-only, or --score-only")
    print(json.dumps(safe({"self_tests": tests, **result}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
