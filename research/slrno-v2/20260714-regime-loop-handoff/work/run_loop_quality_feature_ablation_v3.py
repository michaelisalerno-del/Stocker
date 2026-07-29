"""Frozen V3 source attribution for conditional loop movement quality.

Phase one reconstructs causal 2024 OOF rows, fits only the three interior
representations, reuses sealed qcontext/qfull probabilities, evaluates the
frozen comparison family, fits full-2024 interior models, and freezes all fit
artifacts.  Phase two is locked until an independent pre-score audit explicitly
authorizes loading the already-opened 2025 development and backward-2023
portability panels.

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
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CONTRACT = HERE / "contracts/20260710-loop-quality-feature-ablation-v3.json"
CONTRACT_SHA256 = "221a016e78c353a70261fe724cdfc4d312e355febfc353449844b31b8862702d"
PARENT_ROOT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
OUT = Path("/private/tmp/stocker_loop_quality_feature_ablation_v3_20260710")

PARENT_TRAINING = PARENT_ROOT / "training_long_2024.parquet"
PARENT_OOF = PARENT_ROOT / "oof_predictions_2024.parquet"
FIXED_CYCLES = PARENT_ROOT / "fixed_cycles.csv"
STATE_PARAMETERS = STATE_ROOT / "frozen_semimarkov_parameters.npz"
ANCHOR_2024 = PRICE_ROOT / "anchor_panel_train_2024.parquet"

K = 8
END = 8
CENTROID_WIDTH = 14
CYCLE_COUNT = 20
TOKEN_COUNT = 648
CONTEXT_WIDTH = 17
TOPOLOGY_WIDTH = 63
MODEL_C = 0.2
MAX_ITER = 1000
SEED = 20260710
EPSILON = 1e-12
TARGETS = ("absolute_return_bps", "future_range_bps")
HORIZONS = (6, 12, 24)
TIERS = ("p75", "p90")
MODELS = (
    "qcontext",
    "qroute_topology",
    "qcycle_main",
    "qcycle_state",
    "qfull",
)
NEW_MODELS = ("qroute_topology", "qcycle_main", "qcycle_state")
OOF_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
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
COMPARISONS = (
    ("qroute_topology_vs_qcontext", "qroute_topology", "qcontext", "superiority"),
    (
        "qroute_topology_noninferiority_vs_qfull",
        "qroute_topology",
        "qfull",
        "retention",
    ),
    ("qcycle_main_vs_qroute_topology", "qcycle_main", "qroute_topology", "superiority"),
    ("qcycle_state_vs_qcycle_main", "qcycle_state", "qcycle_main", "superiority"),
    ("qfull_vs_qcycle_state", "qfull", "qcycle_state", "superiority"),
)
REFERENCE_COMPARISON = ("qfull_vs_qcontext_reference", "qfull", "qcontext", "reference")

NEXT_COLUMNS = tuple(f"next_state_p_{state}" for state in range(K))
COMPOSITION_COLUMNS = tuple(f"route_state_fraction_{state}" for state in range(K))
LENGTH_COLUMNS = ("length_is_2", "length_is_3", "length_is_4")
NEXT_CENTROID_COLUMNS = tuple(
    f"next_centroid_z_{index:02d}" for index in range(CENTROID_WIDTH)
)
ROUTE_CENTROID_COLUMNS = tuple(
    f"route_centroid_z_{index:02d}" for index in range(CENTROID_WIDTH)
)
DELTA_CENTROID_COLUMNS = tuple(
    f"next_minus_current_centroid_z_{index:02d}"
    for index in range(CENTROID_WIDTH)
)
AMBIGUITY_COLUMNS = (
    "rotation_count_minus_one",
    "next_state_entropy_normalized",
)
TOPOLOGY_COLUMNS = (
    *NEXT_COLUMNS,
    *COMPOSITION_COLUMNS,
    *LENGTH_COLUMNS,
    *NEXT_CENTROID_COLUMNS,
    *ROUTE_CENTROID_COLUMNS,
    *DELTA_CENTROID_COLUMNS,
    *AMBIGUITY_COLUMNS,
)

EXPECTED_PIN_HASHES = {
    "per_loop_contract.json": "67d64c463df52f01f360561ef0a69d5772b7eec0409468c93d6eb5a630dee02e",
    "v2_contract.json": "33d109a1bcc7ee58fb5ee65a5a5c1075a233baa07d50b1219db8358af22f4728",
    "v3_contract.json": CONTRACT_SHA256,
    "fixed_cycles.csv": "bf9292fa51de1e545e5a319fa2e2faf2088926acd5315b9106597b1da318b253",
    "frozen_semimarkov_parameters.npz": "909858ed7c9c02c1c113661202cb5d7c6bfabd243f1cc428b8a5fb1a3c022251",
    "quality_thresholds_2024.json": "f9e2355e36dae28e4279dfabe74645cb3a363b706d95d4179955093b80015b72",
    "quality_feature_manifest.json": "b3db72b43ad15f89ac8fedd182d2c5eee3931786dc600a339dc8714dad89ddd6",
    "quality_fit_manifest.json": "0a911d631fcc98445d60fa098a219313fabf3d88e78aa1137ab8b684c0e9ee58",
    "parent_oof_predictions_2024.parquet": "689b8853ec482c07a46faea48f49665df8c92612ef28bc9934fe2df2e97e7d30",
    "parent_training_long_2024.parquet": "a901d868f65af5ddcaf296221606d2afe5dae1a8a16f3a59ae8990801acec256",
    "anchor_panel_train_2024.parquet": "788fd81909d1c5d3e6ee20e3e36e3ebb74199188e41052ea1b04f61c96fa9932",
}


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(safe(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_paths_fit() -> dict[str, Path]:
    return {
        "per_loop_contract.json": HERE
        / "contracts/20260710-per-loop-movement-quality-v1.json",
        "v2_contract.json": HERE
        / "contracts/20260710-loop-quality-feature-ablation-v2.json",
        "v3_contract.json": CONTRACT,
        "runner.py": Path(__file__),
        "fixed_cycles.csv": FIXED_CYCLES,
        "frozen_semimarkov_parameters.npz": STATE_PARAMETERS,
        "quality_thresholds_2024.json": PARENT_ROOT / "quality_thresholds_2024.json",
        "quality_feature_manifest.json": PARENT_ROOT / "feature_manifest.json",
        "quality_fit_manifest.json": PARENT_ROOT / "fit_manifest.json",
        "parent_oof_predictions_2024.parquet": PARENT_OOF,
        "parent_training_long_2024.parquet": PARENT_TRAINING,
        "anchor_panel_train_2024.parquet": ANCHOR_2024,
        "parent_final_cycle_tiers.csv": PARENT_ROOT / "final_cycle_tiers.csv",
        "parent_gates.json": PARENT_ROOT / "gates.json",
        "parent_summary.json": PARENT_ROOT / "summary.json",
    }


def hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen source: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def validate_contract_and_sources() -> tuple[dict[str, Any], dict[str, str]]:
    if sha256(CONTRACT) != CONTRACT_SHA256:
        raise AssertionError("V3 contract hash changed")
    contract = json.loads(CONTRACT.read_text())
    if not contract["execution_authorization"]["authorized"]:
        raise AssertionError("V3 execution was not authorized")
    if not contract["research_only"] or contract["live_ordering_enabled"]:
        raise AssertionError("V3 safety labels changed")
    if contract["order_placement"] != "disabled":
        raise AssertionError("V3 order label changed")
    actual = hashes(source_paths_fit())
    for name, expected in EXPECTED_PIN_HASHES.items():
        if actual[name] != expected:
            raise AssertionError(f"pinned source changed: {name}")
    return contract, actual


def cycle_core(cycle: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in str(cycle).split("->"))
    if len(values) < 3 or values[0] != values[-1]:
        raise AssertionError(f"invalid closed cycle {cycle}")
    return values[:-1]


def compatible_rotations(
    core: Iterable[int], current_state: int
) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in core)
    return tuple(
        sorted(
            {
                values[index:] + values[:index] + (int(current_state),)
                for index, state in enumerate(values)
                if state == int(current_state)
            }
        )
    )


def centroid_normalization() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(STATE_PARAMETERS) as parameters:
        means = parameters["means"].astype(float)
        semantic_new = parameters["semantic_new_state"].astype(int)
    if means.shape != (K, CENTROID_WIDTH):
        raise AssertionError("centroid shape changed")
    if not np.array_equal(np.sort(semantic_new), np.arange(K)):
        raise AssertionError("semantic state index changed")
    center = means.mean(axis=0)
    raw_scale = means.std(axis=0, ddof=0)
    scale = np.where(raw_scale > 0.0, raw_scale, 1.0)
    z = (means - center) / scale
    z[:, raw_scale == 0.0] = 0.0
    return z, center, scale


def topology_vector(
    core: Iterable[int], current_state: int, centroids: np.ndarray
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    routes = compatible_rotations(core, current_state)
    if not routes:
        raise AssertionError("incompatible cycle-state pair")
    length = len(routes[0]) - 1
    if length not in HORIZONS[:0] + (2, 3, 4):
        raise AssertionError("unsupported cycle length")
    next_probability = np.zeros(K)
    composition = np.zeros(K)
    for route in routes:
        next_probability[route[1]] += 1.0 / len(routes)
        for state in route[1:]:
            composition[state] += 1.0 / (len(routes) * length)
    length_one_hot = np.asarray([length == value for value in (2, 3, 4)], float)
    next_centroid = next_probability @ centroids
    route_centroid = composition @ centroids
    delta = next_centroid - centroids[int(current_state)]
    positive = next_probability[next_probability > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(K)
    result = np.concatenate(
        (
            next_probability,
            composition,
            length_one_hot,
            next_centroid,
            route_centroid,
            delta,
            np.asarray([len(routes) - 1, entropy]),
        )
    )
    if len(result) != TOPOLOGY_WIDTH or not np.isfinite(result).all():
        raise AssertionError("topology vector invalid")
    return result, routes


def load_cycles_and_mapping() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cycles = pd.read_csv(FIXED_CYCLES)
    if len(cycles) != CYCLE_COUNT:
        raise AssertionError("cycle count changed")
    centroids, center, scale = centroid_normalization()
    rows = []
    for _, cycle_row in cycles.sort_values("cycle_index").iterrows():
        core = cycle_core(cycle_row["cycle"])
        if len(core) != int(cycle_row["transition_length"]):
            raise AssertionError("transition length mismatch")
        for current_state in sorted(set(core)):
            vector, routes = topology_vector(core, current_state, centroids)
            row = {
                "cycle_index": int(cycle_row["cycle_index"]),
                "cycle_id": str(cycle_row["cycle_id"]),
                "cycle": str(cycle_row["cycle"]),
                "transition_length": int(cycle_row["transition_length"]),
                "current_state": int(current_state),
                "compatible_rotation_count": len(routes),
                "compatible_rotations": json.dumps(
                    ["->".join(map(str, route)) for route in routes],
                    separators=(",", ":"),
                ),
            }
            row.update(dict(zip(TOPOLOGY_COLUMNS, vector, strict=True)))
            rows.append(row)
    mapping = pd.DataFrame(rows)
    if mapping.duplicated(["cycle_id", "current_state"]).any():
        raise AssertionError("duplicate topology mapping")
    metadata = {
        "centroid_column_center": center.tolist(),
        "centroid_column_scale": scale.tolist(),
        "topology_columns": list(TOPOLOGY_COLUMNS),
    }
    return cycles, mapping, metadata


def merge_anchor_controls(frame: pd.DataFrame, anchor_path: Path) -> pd.DataFrame:
    anchors = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", "state", "history_token", *NUMERIC_CONTROLS],
    )
    if anchors["anchor_id"].duplicated().any():
        raise AssertionError("anchor controls not unique")
    merged = frame.merge(
        anchors,
        on="anchor_id",
        how="left",
        suffixes=("", "__anchor"),
        sort=False,
        validate="many_to_one",
    )
    if len(merged) != len(frame) or merged[list(NUMERIC_CONTROLS)].isna().any().any():
        raise AssertionError("anchor control merge failed")
    if not merged["state"].equals(merged["state__anchor"]):
        raise AssertionError("state merge disagreement")
    if not merged["history_token"].equals(merged["history_token__anchor"]):
        raise AssertionError("history token merge disagreement")
    return merged.drop(columns=["state__anchor", "history_token__anchor"])


def add_topology(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cycle_id",
        "current_state",
        "compatible_rotation_count",
        "compatible_rotations",
        *TOPOLOGY_COLUMNS,
    ]
    output = frame.merge(
        mapping[columns],
        left_on=["cycle_id", "state"],
        right_on=["cycle_id", "current_state"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(output) != len(frame) or output[list(TOPOLOGY_COLUMNS)].isna().any().any():
        raise AssertionError("topology merge failed")
    return output


def raw_context(frame: pd.DataFrame, medians: Mapping[str, float]) -> sparse.csr_matrix:
    numeric = frame.loc[:, list(NUMERIC_CONTROLS)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(pd.Series(medians))
    values = numeric.to_numpy(float)
    if not np.isfinite(values).all():
        raise AssertionError("non-finite causal context")
    state = frame["state"].to_numpy(int)
    state_matrix = sparse.csr_matrix(np.eye(K)[state])
    result = sparse.hstack((state_matrix, sparse.csr_matrix(values)), format="csr")
    if result.shape[1] != CONTEXT_WIDTH:
        raise AssertionError("context width changed")
    return result


def sparse_one_hot(indices: np.ndarray, width: int, scale: float) -> sparse.csr_matrix:
    indices = np.asarray(indices, int)
    if indices.min(initial=0) < 0 or indices.max(initial=0) >= width:
        raise AssertionError("one-hot index outside width")
    return sparse.csr_matrix(
        (
            np.full(len(indices), scale, float),
            (np.arange(len(indices)), indices),
        ),
        shape=(len(indices), width),
    )


def feature_matrices(
    frame: pd.DataFrame, scaled_context: sparse.csr_matrix
) -> dict[str, sparse.csr_matrix]:
    cycle = frame["cycle_index"].to_numpy(int)
    state = frame["state"].to_numpy(int)
    topology = frame.loc[:, list(TOPOLOGY_COLUMNS)].to_numpy(float).copy()
    topology[:, 19:61] *= 0.5
    route = sparse.hstack(
        (scaled_context, sparse.csr_matrix(topology)), format="csr"
    )
    cycle_main_block = sparse_one_hot(cycle, CYCLE_COUNT, 1.0)
    main = sparse.hstack((scaled_context, cycle_main_block), format="csr")
    state_block = sparse_one_hot(cycle * K + state, CYCLE_COUNT * K, 0.5)
    cycle_state = sparse.hstack((main, state_block), format="csr")
    expected = {
        "qroute_topology": CONTEXT_WIDTH + TOPOLOGY_WIDTH,
        "qcycle_main": CONTEXT_WIDTH + CYCLE_COUNT,
        "qcycle_state": CONTEXT_WIDTH + CYCLE_COUNT + CYCLE_COUNT * K,
    }
    output = {
        "qroute_topology": route,
        "qcycle_main": main,
        "qcycle_state": cycle_state,
    }
    for name, matrix in output.items():
        if matrix.shape[1] != expected[name]:
            raise AssertionError(f"{name} width changed")
    return output


def fit_scaler(raw: sparse.csr_matrix, weights: np.ndarray) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.fit(raw, sample_weight=np.asarray(weights, float))
    return scaler


def fit_model(
    matrix: sparse.csr_matrix, target: np.ndarray, weights: np.ndarray
) -> LogisticRegression:
    target = np.asarray(target, int)
    if not np.array_equal(np.unique(target), np.asarray([0, 1, 2])):
        raise AssertionError("ordered target lacks a class")
    model = LogisticRegression(
        C=MODEL_C,
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=SEED,
    )
    model.fit(matrix, target, sample_weight=np.asarray(weights, float))
    if not np.array_equal(model.classes_, np.asarray([0, 1, 2])):
        raise AssertionError("class order changed")
    if int(model.n_iter_[0]) >= MAX_ITER:
        raise AssertionError("model failed convergence gate")
    return model


def task_key(model: str, target: str, horizon: int) -> str:
    return f"{model}__{target}__h{horizon}"


def add_probability_columns(
    frame: pd.DataFrame,
    model: str,
    target: str,
    horizon: int,
    probability: np.ndarray,
) -> None:
    probability = np.asarray(probability, float)
    if probability.shape != (len(frame), 3):
        raise AssertionError("class probability shape changed")
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0):
        raise AssertionError("invalid class probability")
    key = task_key(model, target, horizon)
    p75 = probability[:, 1] + probability[:, 2]
    p90 = probability[:, 2]
    frame[f"{key}__p75"] = p75
    frame[f"{key}__p90"] = p90
    structural = frame["loop_probability"].to_numpy(float)
    frame[f"joint__{key}__p75"] = structural * p75
    frame[f"joint__{key}__p90"] = structural * p90
    if (p90 > p75 + 1e-12).any():
        raise AssertionError("ordered probabilities not nested")


def reuse_parent_probabilities(frame: pd.DataFrame) -> None:
    for target in TARGETS:
        for horizon in HORIZONS:
            for model, parent_model in (("qcontext", "qcontext"), ("qfull", "qcycle")):
                for tier in TIERS:
                    parent = f"{parent_model}__{target}__h{horizon}__{tier}"
                    parent_joint = f"joint__{parent}"
                    key = f"{model}__{target}__h{horizon}__{tier}"
                    if parent not in frame or parent_joint not in frame:
                        raise AssertionError("sealed parent probability missing")
                    frame[key] = frame[parent].to_numpy(float)
                    frame[f"joint__{key}"] = frame[parent_joint].to_numpy(float)
                if (
                    frame[f"{model}__{target}__h{horizon}__p90"]
                    > frame[f"{model}__{target}__h{horizon}__p75"] + 1e-12
                ).any():
                    raise AssertionError("sealed parent nesting failure")


def predict_from_parameters(
    matrix: sparse.csr_matrix, parameters: Mapping[str, np.ndarray], key: str
) -> np.ndarray:
    coef = parameters[f"{key}__coef"]
    intercept = parameters[f"{key}__intercept"]
    logits = np.asarray(matrix @ coef.T) + intercept
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    return probability / probability.sum(axis=1, keepdims=True)


def binary_losses(observed: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    observed = np.asarray(observed, float)
    probability = np.clip(np.asarray(probability, float), EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": -(
            observed * np.log(probability)
            + (1.0 - observed) * np.log(1.0 - probability)
        ),
        "brier": np.square(probability - observed),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    if len(values) == 0 or float(weights.sum()) <= 0.0:
        return math.nan
    return float(np.average(values, weights=weights))


def weighted_group_mean(
    groups: Iterable[Any], values: np.ndarray, weights: np.ndarray
) -> pd.Series:
    table = pd.DataFrame(
        {
            "group": pd.Series(groups).astype(str).to_numpy(),
            "weighted": np.asarray(values, float) * np.asarray(weights, float),
            "weight": np.asarray(weights, float),
        }
    ).groupby("group", sort=True).sum()
    return table["weighted"] / table["weight"]


def moving_block_interval(
    values: np.ndarray, seed: int, confidence: float = 0.99, draws: int = 10000
) -> tuple[float, float, float]:
    clean = np.asarray(values, float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return math.nan, math.nan, math.nan
    block_length = min(5, len(clean))
    blocks = np.asarray(
        [clean[start : start + block_length] for start in range(len(clean) - block_length + 1)]
    )
    needed = int(math.ceil(len(clean) / block_length))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        chosen = rng.integers(0, len(blocks), size=needed)
        samples[index] = blocks[chosen].reshape(-1)[: len(clean)].mean()
    alpha = (1.0 - confidence) / 2.0
    return (
        float(clean.mean()),
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
    )


def calibration_summary(
    observed: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    minimum_bin_rows: int,
) -> tuple[float, float]:
    observed = np.asarray(observed, float)
    probability = np.asarray(probability, float)
    weights = np.asarray(weights, float)
    bins = np.minimum((probability * 10.0).astype(int), 9)
    total = float(weights.sum())
    if total <= 0.0:
        return math.nan, math.nan
    ece = 0.0
    supported_errors = []
    for index in range(10):
        mask = bins == index
        if not mask.any():
            continue
        weight = float(weights[mask].sum())
        error = abs(
            weighted_mean(observed[mask], weights[mask])
            - weighted_mean(probability[mask], weights[mask])
        )
        ece += weight / total * error
        if int(mask.sum()) >= minimum_bin_rows:
            supported_errors.append(error)
    return float(ece), max(supported_errors) if supported_errors else math.nan


def surface_frame(
    panel: pd.DataFrame, surface: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if surface == "conditional":
        output = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
        weights = output["conditional_weight"].to_numpy(float)
    elif surface == "joint":
        output = panel.reset_index(drop=True)
        weights = np.ones(len(output))
    else:
        raise ValueError(surface)
    return output, weights


def cell_observed(frame: pd.DataFrame, surface: str, target: str, horizon: int, tier: str) -> np.ndarray:
    if surface == "conditional":
        classes = frame[f"quality_class__{target}__h{horizon}"].to_numpy(int)
        return (classes >= (1 if tier == "p75" else 2)).astype(int)
    label = "good" if tier == "p75" else "high"
    return frame[f"joint_{label}_target__{target}__h{horizon}"].to_numpy(int)


def probability_column(model: str, target: str, horizon: int, tier: str, surface: str) -> str:
    key = f"{model}__{target}__h{horizon}__{tier}"
    return key if surface == "conditional" else f"joint__{key}"


def build_cell_metrics(panel: pd.DataFrame, period: str, mode: str) -> pd.DataFrame:
    rows = []
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
                    for model in MODELS:
                        probability = frame[
                            probability_column(model, target, horizon, tier, surface)
                        ].to_numpy(float)
                        losses = binary_losses(observed, probability)
                        ece, maximum = calibration_summary(
                            observed, probability, weights, minimum
                        )
                        rows.append(
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
    return pd.DataFrame(rows)


def cell_difference_data(
    frame: pd.DataFrame,
    weights: np.ndarray,
    surface: str,
    candidate: str,
    baseline: str,
    target: str,
    horizon: int,
    tier: str,
    loss: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = cell_observed(frame, surface, target, horizon, tier)
    candidate_probability = frame[
        probability_column(candidate, target, horizon, tier, surface)
    ].to_numpy(float)
    baseline_probability = frame[
        probability_column(baseline, target, horizon, tier, surface)
    ].to_numpy(float)
    candidate_loss = binary_losses(observed, candidate_probability)[loss]
    baseline_loss = binary_losses(observed, baseline_probability)[loss]
    return candidate_loss - baseline_loss, candidate_loss, baseline_loss


def pooled_model_loss(
    panel: pd.DataFrame, surface: str, model: str, loss: str
) -> float:
    frame, weights = surface_frame(panel, surface)
    values = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                observed = cell_observed(frame, surface, target, horizon, tier)
                probability = frame[
                    probability_column(model, target, horizon, tier, surface)
                ].to_numpy(float)
                values.append(weighted_mean(binary_losses(observed, probability)[loss], weights))
    return float(np.mean(values))


def comparison_surface(
    panel: pd.DataFrame,
    cell_metrics: pd.DataFrame,
    period: str,
    surface: str,
    comparison_index: int,
    name: str,
    candidate: str,
    baseline: str,
    kind: str,
) -> dict[str, Any]:
    frame, weights = surface_frame(panel, surface)
    loss_payload: dict[str, Any] = {}
    cell_rows: list[dict[str, Any]] = []
    target_aggregates: dict[str, dict[str, float]] = {}
    horizon_aggregates: dict[str, dict[str, float]] = {}
    calibration_ece_tolerance = 0.005 if surface == "conditional" else 0.0025
    calibration_max_tolerance = 0.01 if surface == "conditional" else 0.005

    cell_differences: dict[tuple[str, int, str, str], np.ndarray] = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                for loss in ("log_loss", "brier"):
                    difference, candidate_loss, baseline_loss = cell_difference_data(
                        frame,
                        weights,
                        surface,
                        candidate,
                        baseline,
                        target,
                        horizon,
                        tier,
                        loss,
                    )
                    cell_differences[(target, horizon, tier, loss)] = difference
                    if loss == "log_loss":
                        candidate_metric = cell_metrics.loc[
                            cell_metrics["surface"].eq(surface)
                            & cell_metrics["model"].eq(candidate)
                            & cell_metrics["target"].eq(target)
                            & cell_metrics["horizon"].eq(horizon)
                            & cell_metrics["tier"].eq(tier)
                        ].iloc[0]
                        baseline_metric = cell_metrics.loc[
                            cell_metrics["surface"].eq(surface)
                            & cell_metrics["model"].eq(baseline)
                            & cell_metrics["target"].eq(target)
                            & cell_metrics["horizon"].eq(horizon)
                            & cell_metrics["tier"].eq(tier)
                        ].iloc[0]
                        mean_difference = weighted_mean(difference, weights)
                        baseline_mean = weighted_mean(baseline_loss, weights)
                        cell_rows.append(
                            {
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "log_loss_difference": mean_difference,
                                "relative_log_loss_degradation": mean_difference / baseline_mean,
                                "candidate_ece": float(candidate_metric["ece"]),
                                "baseline_ece": float(baseline_metric["ece"]),
                                "ece_difference": float(candidate_metric["ece"] - baseline_metric["ece"]),
                                "candidate_maximum_supported_bin_error": float(candidate_metric["maximum_supported_bin_error"]),
                                "baseline_maximum_supported_bin_error": float(baseline_metric["maximum_supported_bin_error"]),
                                "maximum_supported_bin_error_difference": float(candidate_metric["maximum_supported_bin_error"] - baseline_metric["maximum_supported_bin_error"]),
                            }
                        )

    for loss_index, loss in enumerate(("log_loss", "brier")):
        cell_means = []
        baseline_means = []
        daily_columns = []
        quarter_columns = []
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    difference = cell_differences[(target, horizon, tier, loss)]
                    _, _, baseline_loss = cell_difference_data(
                        frame,
                        weights,
                        surface,
                        candidate,
                        baseline,
                        target,
                        horizon,
                        tier,
                        loss,
                    )
                    cell_means.append(weighted_mean(difference, weights))
                    baseline_means.append(weighted_mean(baseline_loss, weights))
                    daily_columns.append(
                        weighted_group_mean(frame["session_date"], difference, weights).rename(
                            f"{target}_{horizon}_{tier}"
                        )
                    )
                    quarter_columns.append(
                        weighted_group_mean(frame["quarter"], difference, weights).rename(
                            f"{target}_{horizon}_{tier}"
                        )
                    )
        pooled_difference = float(np.mean(cell_means))
        pooled_baseline = float(np.mean(baseline_means))
        daily = pd.concat(daily_columns, axis=1).mean(axis=1)
        quarters = pd.concat(quarter_columns, axis=1).mean(axis=1)
        seed = SEED + comparison_index * 100 + (0 if surface == "conditional" else 10) + loss_index
        interval = moving_block_interval(daily.to_numpy(float), seed)
        deletion_values = {}
        symbols = sorted(frame["symbol_norm"].astype(str).unique())
        symbol_values = frame["symbol_norm"].astype(str).to_numpy()
        for symbol in symbols:
            keep = symbol_values != symbol
            per_cell = [
                weighted_mean(values[keep], weights[keep])
                for (key, values) in cell_differences.items()
                if key[3] == loss
            ]
            deletion_values[symbol] = float(np.mean(per_cell))
        loss_payload[loss] = {
            "pooled_candidate_minus_baseline": pooled_difference,
            "pooled_baseline": pooled_baseline,
            "relative_improvement": -pooled_difference / pooled_baseline,
            "daily_mean": interval[0],
            "daily_ci_low": interval[1],
            "daily_ci_high": interval[2],
            "quarter_differences": quarters.to_dict(),
            "leave_one_stock_out_differences": deletion_values,
            "maximum_leave_one_stock_out_difference": max(deletion_values.values()),
        }

    for target in TARGETS:
        target_aggregates[target] = {}
        for loss in ("log_loss", "brier"):
            values = [
                weighted_mean(cell_differences[(target, horizon, tier, loss)], weights)
                for horizon in HORIZONS
                for tier in TIERS
            ]
            target_aggregates[target][loss] = float(np.mean(values))
    for horizon in HORIZONS:
        horizon_aggregates[str(horizon)] = {}
        for loss in ("log_loss", "brier"):
            values = [
                weighted_mean(cell_differences[(target, horizon, tier, loss)], weights)
                for target in TARGETS
                for tier in TIERS
            ]
            horizon_aggregates[str(horizon)][loss] = float(np.mean(values))

    calibration_pass = all(
        row["ece_difference"] <= calibration_ece_tolerance
        and row["maximum_supported_bin_error_difference"] <= calibration_max_tolerance
        for row in cell_rows
    )
    common_checks = {
        "minimum_relative_log_loss_improvement": loss_payload["log_loss"]["relative_improvement"] >= 0.0025,
        "pooled_brier_difference_below_zero": loss_payload["brier"]["pooled_candidate_minus_baseline"] < 0.0,
        "daily_log_loss_upper_below_zero": loss_payload["log_loss"]["daily_ci_high"] < 0.0,
        "daily_brier_upper_below_zero": loss_payload["brier"]["daily_ci_high"] < 0.0,
        "every_quarter_log_loss_below_zero": all(value < 0.0 for value in loss_payload["log_loss"]["quarter_differences"].values()),
        "every_quarter_brier_below_zero": all(value < 0.0 for value in loss_payload["brier"]["quarter_differences"].values()),
        "every_stock_deletion_log_loss_below_zero": loss_payload["log_loss"]["maximum_leave_one_stock_out_difference"] < 0.0,
        "every_stock_deletion_brier_below_zero": loss_payload["brier"]["maximum_leave_one_stock_out_difference"] < 0.0,
        "cell_maximum_relative_log_loss_degradation": max(row["relative_log_loss_degradation"] for row in cell_rows) <= 0.0025,
        "target_aggregates_no_worse": all(value <= 0.0 for payload in target_aggregates.values() for value in payload.values()),
        "horizon_aggregates_no_worse": all(value <= 0.0 for payload in horizon_aggregates.values() for value in payload.values()),
        "calibration_noninferiority": calibration_pass,
    }
    return {
        "period": period,
        "surface": surface,
        "comparison": name,
        "candidate": candidate,
        "baseline": baseline,
        "kind": kind,
        "losses": loss_payload,
        "cell_diagnostics": cell_rows,
        "target_aggregates": target_aggregates,
        "horizon_aggregates": horizon_aggregates,
        "common_checks": common_checks,
        "common_pass": bool(all(common_checks.values())),
    }


def rotation_diagnostics(
    panel: pd.DataFrame,
    period: str,
    mode: str,
    comparison_name: str,
    candidate: str,
    baseline: str,
) -> pd.DataFrame:
    frame = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
    weights = frame["conditional_weight"].to_numpy(float)
    group_specs = {
        "cycle_current_state": frame["cycle_id"].astype(str) + "@" + frame["state"].astype(str),
        "compatible_rotation_count": frame["compatible_rotation_count"].astype(str),
        "next_state_entropy_quartile": frame["entropy_quartile"].astype(str),
    }
    difference_cache: dict[str, list[np.ndarray]] = {"log_loss": [], "brier": []}
    for loss in ("log_loss", "brier"):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    difference, _, _ = cell_difference_data(
                        frame,
                        weights,
                        "conditional",
                        candidate,
                        baseline,
                        target,
                        horizon,
                        tier,
                        loss,
                    )
                    difference_cache[loss].append(difference)
    rows = []
    for group_type, group_values in group_specs.items():
        for group_value in sorted(group_values.unique()):
            mask = group_values.eq(group_value).to_numpy()
            subset = frame.loc[mask]
            required_rows = 100 if mode == "oof" else 200
            required_quarters = 2 if mode == "oof" else 4
            supported = bool(
                int(mask.sum()) >= required_rows
                and subset["symbol_norm"].nunique() >= 10
                and subset["quarter"].nunique() == required_quarters
            )
            pooled = {}
            for loss in ("log_loss", "brier"):
                values = [
                    weighted_mean(difference[mask], weights[mask])
                    for difference in difference_cache[loss]
                ]
                pooled[loss] = float(np.mean(values))
            rows.append(
                {
                    "period": period,
                    "comparison": comparison_name,
                    "candidate": candidate,
                    "baseline": baseline,
                    "group_type": group_type,
                    "group_value": group_value,
                    "rows": int(mask.sum()),
                    "weight": float(weights[mask].sum()),
                    "stocks": int(subset["symbol_norm"].nunique()),
                    "quarters": int(subset["quarter"].nunique()),
                    "supported": supported,
                    "pooled_log_loss_difference": pooled["log_loss"],
                    "pooled_brier_difference": pooled["brier"],
                    "sign_reversal": bool(
                        supported and (pooled["log_loss"] > 0.0 or pooled["brier"] > 0.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def retention_surface_checks(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    surface_payload: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    gains: dict[str, dict[str, float]] = {}
    for loss in ("log_loss", "brier"):
        context_loss = pooled_model_loss(panel, surface, "qcontext", loss)
        full_loss = pooled_model_loss(panel, surface, "qfull", loss)
        route_loss = pooled_model_loss(panel, surface, "qroute_topology", loss)
        full_gain = context_loss - full_loss
        route_gain = context_loss - route_loss
        gains[loss] = {
            "context_loss": context_loss,
            "full_loss": full_loss,
            "route_loss": route_loss,
            "full_gain": full_gain,
            "route_gain": route_gain,
            "retention": route_gain / full_gain if full_gain > 0.0 else math.nan,
            "noninferiority_margin": 0.1 * full_gain if full_gain > 0.0 else math.nan,
        }
    cell_retention = []
    for target in TARGETS:
        for horizon in HORIZONS:
            for tier in TIERS:
                subset = metrics.loc[
                    metrics["surface"].eq(surface)
                    & metrics["target"].eq(target)
                    & metrics["horizon"].eq(horizon)
                    & metrics["tier"].eq(tier)
                ].set_index("model")
                context_loss = float(subset.loc["qcontext", "log_loss"])
                full_loss = float(subset.loc["qfull", "log_loss"])
                route_loss = float(subset.loc["qroute_topology", "log_loss"])
                full_gain = context_loss - full_loss
                route_gain = context_loss - route_loss
                cell_retention.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "tier": tier,
                        "full_gain": full_gain,
                        "route_gain": route_gain,
                        "retention": route_gain / full_gain if full_gain > 0.0 else math.nan,
                        "pass": bool(full_gain > 0.0 and route_gain / full_gain >= 0.75),
                    }
                )
    log_margin = gains["log_loss"]["noninferiority_margin"]
    brier_margin = gains["brier"]["noninferiority_margin"]
    losses = surface_payload["losses"]
    checks = {
        "conditional_or_joint_log_loss_retention_at_least_90pct": gains["log_loss"]["retention"] >= 0.90,
        "conditional_or_joint_brier_retention_at_least_80pct": gains["brier"]["retention"] >= 0.80,
        "log_loss_99pct_upper_within_margin": losses["log_loss"]["daily_ci_high"] <= log_margin,
        "brier_99pct_upper_within_margin": losses["brier"]["daily_ci_high"] <= brier_margin,
        "every_quarter_log_loss_within_margin": all(value <= log_margin for value in losses["log_loss"]["quarter_differences"].values()),
        "every_quarter_brier_within_margin": all(value <= brier_margin for value in losses["brier"]["quarter_differences"].values()),
        "every_stock_log_loss_within_margin": losses["log_loss"]["maximum_leave_one_stock_out_difference"] <= log_margin,
        "every_stock_brier_within_margin": losses["brier"]["maximum_leave_one_stock_out_difference"] <= brier_margin,
        "each_positive_full_gain_cell_retains_75pct": all(row["pass"] for row in cell_retention),
        "calibration_noninferiority_to_full": surface_payload["common_checks"]["calibration_noninferiority"],
    }
    return {
        "gains": gains,
        "cell_retention": cell_retention,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def provisional_attribution(comparison_pass: Mapping[str, bool]) -> dict[str, Any]:
    reference = bool(comparison_pass.get("qfull_vs_qcontext_reference", False))
    topology_signal = bool(comparison_pass.get("qroute_topology_vs_qcontext", False))
    topology_retention = bool(
        comparison_pass.get("qroute_topology_noninferiority_vs_qfull", False)
    )
    identity = bool(comparison_pass.get("qcycle_main_vs_qroute_topology", False))
    state = bool(comparison_pass.get("qcycle_state_vs_qcycle_main", False))
    history = bool(comparison_pass.get("qfull_vs_qcycle_state", False))
    supported_components = []
    if identity:
        supported_components.append("cycle_identity_representation_needed")
    if state:
        supported_components.append("current_state_rotation_needed")
    if history:
        supported_components.append("history_token_needed")
    if not reference:
        label = "no_reference_signal"
    elif topology_signal and topology_retention and not supported_components:
        label = "topology_sufficient"
    elif topology_signal and topology_retention:
        label = "topology_dominant_with_residual_detail"
    elif identity:
        label = "cycle_identity_representation_needed"
    elif state:
        label = "current_state_rotation_needed"
    elif history:
        label = "history_token_needed"
    else:
        label = "unresolved"
    return {
        "label": label,
        "reference_signal_pass": reference,
        "topology_signal_pass": topology_signal,
        "topology_retention_pass": topology_retention,
        "supported_residual_components": supported_components,
        "comparison_pass": dict(comparison_pass),
        "prospective_validated": False,
        "frozen_parent_grade_changed": False,
    }


def evaluate_period(
    panel: pd.DataFrame, period: str, mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    metrics = build_cell_metrics(panel, period, mode)
    payloads: dict[str, Any] = {}
    summary_rows = []
    rotation_frames = []
    all_specs = (*COMPARISONS, REFERENCE_COMPARISON)
    comparison_pass: dict[str, bool] = {}
    for index, (name, candidate, baseline, kind) in enumerate(all_specs):
        surfaces = {}
        for surface in ("conditional", "joint"):
            surface_payload = comparison_surface(
                panel,
                metrics,
                period,
                surface,
                index,
                name,
                candidate,
                baseline,
                kind,
            )
            if kind == "retention":
                retention = retention_surface_checks(
                    panel, metrics, surface_payload, surface
                )
                surface_payload["retention_gate"] = retention
                surface_payload["surface_pass"] = retention["pass"]
            else:
                surface_payload["surface_pass"] = surface_payload["common_pass"]
            surfaces[surface] = surface_payload
        rotation = rotation_diagnostics(
            panel, period, mode, name, candidate, baseline
        )
        rotation_pass = bool(
            not rotation.loc[rotation["supported"], "sign_reversal"].any()
        )
        overall = bool(
            surfaces["conditional"]["surface_pass"]
            and surfaces["joint"]["surface_pass"]
            and rotation_pass
        )
        comparison_pass[name] = overall
        payloads[name] = {
            "candidate": candidate,
            "baseline": baseline,
            "kind": kind,
            "surfaces": surfaces,
            "rotation_no_sign_reversal_pass": rotation_pass,
            "pass": overall,
        }
        summary_rows.append(
            {
                "period": period,
                "comparison": name,
                "candidate": candidate,
                "baseline": baseline,
                "kind": kind,
                "conditional_pass": surfaces["conditional"]["surface_pass"],
                "joint_pass": surfaces["joint"]["surface_pass"],
                "rotation_no_sign_reversal_pass": rotation_pass,
                "pass": overall,
                "conditional_relative_log_loss_improvement": surfaces["conditional"]["losses"]["log_loss"]["relative_improvement"],
                "joint_relative_log_loss_improvement": surfaces["joint"]["losses"]["log_loss"]["relative_improvement"],
            }
        )
        rotation_frames.append(rotation)
    attribution = provisional_attribution(comparison_pass)
    gates = {
        "period": period,
        "mode": mode,
        "comparisons": payloads,
        "comparison_pass": comparison_pass,
        "source_attribution": attribution,
    }
    return (
        metrics,
        pd.DataFrame(summary_rows),
        gates,
        pd.concat(rotation_frames, ignore_index=True),
        attribution,
    )


def parent_probability_columns() -> list[str]:
    columns = []
    for parent_model in ("qcontext", "qcycle"):
        for target in TARGETS:
            for horizon in HORIZONS:
                for tier in TIERS:
                    key = f"{parent_model}__{target}__h{horizon}__{tier}"
                    columns.extend((key, f"joint__{key}"))
    return columns


def label_columns() -> list[str]:
    columns = []
    for target in TARGETS:
        for horizon in HORIZONS:
            columns.append(f"quality_class__{target}__h{horizon}")
            columns.append(f"joint_good_target__{target}__h{horizon}")
            columns.append(f"joint_high_target__{target}__h{horizon}")
    return columns


def prepare_oof(mapping: pd.DataFrame) -> pd.DataFrame:
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
        *label_columns(),
        *parent_probability_columns(),
    ]
    frame = pd.read_parquet(PARENT_OOF, columns=columns).reset_index(drop=True)
    frame.insert(0, "source_row", np.arange(len(frame), dtype=np.int64))
    frame = merge_anchor_controls(frame, ANCHOR_2024)
    frame = add_topology(frame, mapping)
    frame = frame.sort_values("source_row", kind="stable").reset_index(drop=True)
    reuse_parent_probabilities(frame)
    if len(frame) != 216438 or not np.array_equal(frame["source_row"], np.arange(len(frame))):
        raise AssertionError("OOF row alignment changed")
    return frame


def prepare_training(mapping: pd.DataFrame) -> pd.DataFrame:
    training = pd.read_parquet(PARENT_TRAINING).reset_index(drop=True)
    if len(training) != 32677 or not training["loop_occurs"].eq(1).all():
        raise AssertionError("full-2024 training cohort changed")
    training = add_topology(training, mapping)
    training["month_key"] = pd.to_datetime(
        training["session_date"], errors="raise"
    ).dt.strftime("%Y-%m")
    return training


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(np.asarray(list(quantiles), float), cumulative, values)


def entropy_cutpoints(oof: pd.DataFrame) -> np.ndarray:
    positive = oof.loc[oof["loop_occurs"].eq(1)]
    return weighted_quantiles(
        positive["next_state_entropy_normalized"].to_numpy(float),
        positive["conditional_weight"].to_numpy(float),
        (0.25, 0.50, 0.75),
    )


def add_entropy_quartile(frame: pd.DataFrame, cutpoints: np.ndarray) -> None:
    frame["entropy_quartile"] = np.searchsorted(
        np.asarray(cutpoints, float),
        frame["next_state_entropy_normalized"].to_numpy(float),
        side="left",
    ).astype(np.int8)


def v3_support(oof: pd.DataFrame) -> dict[str, Any]:
    positive = oof.loc[oof["loop_occurs"].eq(1)]
    quarter_weight = positive.groupby("quarter")["conditional_weight"].sum().to_dict()
    stock_weight = positive.groupby("symbol_norm")["conditional_weight"].sum().to_dict()
    checks = {
        "total_effective_weight": float(positive["conditional_weight"].sum()) >= 10000.0,
        "each_required_quarter_weight": all(
            float(quarter_weight.get(quarter, 0.0)) >= 5000.0
            for quarter in ("2024_q3", "2024_q4")
        ),
        "sessions": positive["session_date"].nunique() >= 100,
        "stocks": positive["symbol_norm"].nunique() >= 18,
        "each_stock_effective_weight": bool(
            stock_weight and min(stock_weight.values()) >= 50.0
        ),
        "realized_rows_reconstruction_integrity": len(positive) >= 15000,
    }
    return {
        "compatible_rows": len(oof),
        "realized_rows": len(positive),
        "unique_realized_anchors": int(positive["anchor_id"].nunique()),
        "total_effective_weight": float(positive["conditional_weight"].sum()),
        "quarter_effective_weight": quarter_weight,
        "sessions": int(positive["session_date"].nunique()),
        "stocks": int(positive["symbol_norm"].nunique()),
        "minimum_stock_effective_weight": float(min(stock_weight.values())),
        "checks": checks,
        "support_pass": bool(all(checks.values())),
        "realized_rows_is_independent_support_gate": False,
    }


def fit_oof_models(
    training: pd.DataFrame,
    oof: pd.DataFrame,
    medians: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = oof.copy()
    probabilities = {
        (model, target, horizon): np.full((len(output), 3), np.nan)
        for model in NEW_MODELS
        for target in TARGETS
        for horizon in HORIZONS
    }
    fold_rows = []
    training_weights = training["conditional_weight"].to_numpy(float)
    training_raw = raw_context(training, medians)
    output_raw = raw_context(output, medians)
    output_month = pd.to_datetime(output["session_date"], errors="raise").dt.strftime("%Y-%m")
    for fold_index, validation_month in enumerate(OOF_MONTHS, start=1):
        train_positions = np.flatnonzero(training["month_key"].lt(validation_month).to_numpy())
        validation_positions = np.flatnonzero(output_month.eq(validation_month).to_numpy())
        if len(train_positions) == 0 or len(validation_positions) == 0:
            raise AssertionError("empty causal OOF fold")
        weights = training_weights[train_positions]
        scaler = fit_scaler(training_raw[train_positions], weights)
        train_context = scaler.transform(training_raw[train_positions]).tocsr()
        validation_context = scaler.transform(output_raw[validation_positions]).tocsr()
        train_frame = training.iloc[train_positions].reset_index(drop=True)
        validation_frame = output.iloc[validation_positions].reset_index(drop=True)
        train_matrices = feature_matrices(train_frame, train_context)
        validation_matrices = feature_matrices(validation_frame, validation_context)
        for model in NEW_MODELS:
            for target in TARGETS:
                for horizon in HORIZONS:
                    labels = train_frame[
                        f"quality_class__{target}__h{horizon}"
                    ].to_numpy(int)
                    fitted = fit_model(train_matrices[model], labels, weights)
                    probability = fitted.predict_proba(validation_matrices[model])
                    probabilities[(model, target, horizon)][validation_positions] = probability
                    fold_rows.append(
                        {
                            "fold": fold_index,
                            "validation_month": validation_month,
                            "training_rows": len(train_positions),
                            "training_weight": float(weights.sum()),
                            "validation_compatible_rows": len(validation_positions),
                            "model": model,
                            "target": target,
                            "horizon": horizon,
                            "feature_width": train_matrices[model].shape[1],
                            "n_iter": int(fitted.n_iter_[0]),
                            "temperature": 1.0,
                        }
                    )
    for (model, target, horizon), probability in probabilities.items():
        if not np.isfinite(probability).all():
            raise AssertionError("OOF probability has an uncovered row")
        add_probability_columns(output, model, target, horizon, probability)
    return output, pd.DataFrame(fold_rows)


def fit_full_models(
    training: pd.DataFrame, medians: Mapping[str, float], centroid_metadata: Mapping[str, Any], entropy_cuts: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    weights = training["conditional_weight"].to_numpy(float)
    raw = raw_context(training, medians)
    scaler = fit_scaler(raw, weights)
    context = scaler.transform(raw).tocsr()
    matrices = feature_matrices(training, context)
    parameters: dict[str, np.ndarray] = {
        "context_scaler_scale": scaler.scale_.copy(),
        "context_scaler_mean": scaler.mean_.copy(),
        "context_scaler_var": scaler.var_.copy(),
        "context_numeric_medians": np.asarray([medians[name] for name in NUMERIC_CONTROLS]),
        "centroid_column_center": np.asarray(centroid_metadata["centroid_column_center"]),
        "centroid_column_scale": np.asarray(centroid_metadata["centroid_column_scale"]),
        "entropy_quartile_cutpoints": np.asarray(entropy_cuts, float),
    }
    audit: dict[str, Any] = {"models": {}}
    for model in NEW_MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                key = task_key(model, target, horizon)
                labels = training[f"quality_class__{target}__h{horizon}"].to_numpy(int)
                fitted = fit_model(matrices[model], labels, weights)
                parameters[f"{key}__classes"] = fitted.classes_.copy()
                parameters[f"{key}__coef"] = fitted.coef_.copy()
                parameters[f"{key}__intercept"] = fitted.intercept_.copy()
                parameters[f"{key}__n_iter"] = fitted.n_iter_.copy()
                parameters[f"{key}__temperature"] = np.asarray([1.0])
                audit["models"][key] = {
                    "feature_width": matrices[model].shape[1],
                    "n_iter": int(fitted.n_iter_[0]),
                    "temperature": 1.0,
                }
    return parameters, audit


def topology_feature_manifest(
    centroid_metadata: Mapping[str, Any], entropy_cuts: np.ndarray
) -> dict[str, Any]:
    blocks = [
        {"name": "candidate_next_state_distribution", "start": 0, "stop": 8, "scale": 1.0},
        {"name": "future_route_state_composition", "start": 8, "stop": 16, "scale": 1.0},
        {"name": "transition_length_one_hot", "start": 16, "stop": 19, "scale": 1.0},
        {"name": "next_centroid_expectation", "start": 19, "stop": 33, "scale": 0.5},
        {"name": "route_centroid_expectation", "start": 33, "stop": 47, "scale": 0.5},
        {"name": "next_minus_current_centroid", "start": 47, "stop": 61, "scale": 0.5},
        {"name": "rotation_ambiguity", "start": 61, "stop": 63, "scale": 1.0},
    ]
    return {
        "topology_width": TOPOLOGY_WIDTH,
        "topology_columns": list(TOPOLOGY_COLUMNS),
        "blocks": blocks,
        "design_rows_store_unscaled_normalized_expectations": True,
        "entropy_quartile_cutpoints": np.asarray(entropy_cuts).tolist(),
        "entropy_quantile_weighting": "2024 OOF realized-loop inverse-overlap weight",
        "future_realized_feature_used": False,
        "stock_identity_feature_used": False,
        **dict(centroid_metadata),
    }


def feature_manifest(
    medians: Mapping[str, float], entropy_cuts: np.ndarray
) -> dict[str, Any]:
    return {
        "models": {
            "qcontext": {"width": 17, "sealed_reuse": True},
            "qroute_topology": {"width": 80, "fitted": True},
            "qcycle_main": {"width": 37, "fitted": True},
            "qcycle_state": {"width": 197, "fitted": True},
            "qfull": {"width": 13157, "sealed_reuse": True},
        },
        "numeric_controls": list(NUMERIC_CONTROLS),
        "numeric_medians": dict(medians),
        "topology_columns": list(TOPOLOGY_COLUMNS),
        "entropy_quartile_cutpoints": np.asarray(entropy_cuts).tolist(),
        "temperature": 1.0,
        "C": MODEL_C,
        "solver": "lbfgs",
        "seed": SEED,
        "stock_identity_feature_used": False,
        "future_realized_feature_used": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def validate_panel_probabilities(panel: pd.DataFrame) -> None:
    for model in MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                for prefix in ("", "joint__"):
                    p75 = panel[
                        f"{prefix}{model}__{target}__h{horizon}__p75"
                    ].to_numpy(float)
                    p90 = panel[
                        f"{prefix}{model}__{target}__h{horizon}__p90"
                    ].to_numpy(float)
                    if not np.isfinite(p75).all() or not np.isfinite(p90).all():
                        raise AssertionError("non-finite probability")
                    if (p90 > p75 + 1e-12).any() or p90.min() < -1e-12 or p75.max() > 1.0 + 1e-12:
                        raise AssertionError("probability nesting or bounds failure")
                    if prefix == "joint__":
                        conditional_p75 = panel[
                            f"{model}__{target}__h{horizon}__p75"
                        ].to_numpy(float)
                        structural = panel["loop_probability"].to_numpy(float)
                        if not np.allclose(p75, conditional_p75 * structural, atol=1e-12, rtol=0.0):
                            raise AssertionError("joint chain rule failure")


def fit_artifact_names() -> tuple[str, ...]:
    return (
        "topology_feature_manifest.json",
        "feature_manifest.json",
        "rotation_mapping.csv",
        "oof_predictions_2024.parquet",
        "model_parameters.npz",
        "fold_audit_2024.csv",
        "support_2024.json",
        "cell_diagnostics_2024.csv",
        "comparison_summary_2024.csv",
        "paired_pooled_gates_2024.json",
        "rotation_diagnostics_2024.csv",
        "two_axis_cycle_diagnostics_2024.csv",
        "provisional_source_attribution.json",
        "fit_source_hashes.json",
        "full_fit_audit.json",
    )


def hash_named_files(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: sha256(root / name) for name in names}


def load_numeric_medians() -> dict[str, float]:
    manifest = json.loads((PARENT_ROOT / "feature_manifest.json").read_text())
    medians = {name: float(manifest["numeric_medians"][name]) for name in NUMERIC_CONTROLS}
    return medians


def write_fit_artifacts(
    cycles: pd.DataFrame,
    mapping: pd.DataFrame,
    centroid_metadata: Mapping[str, Any],
    oof: pd.DataFrame,
    fold_audit: pd.DataFrame,
    support: Mapping[str, Any],
    metrics: pd.DataFrame,
    comparison_summary: pd.DataFrame,
    gates: Mapping[str, Any],
    rotations: pd.DataFrame,
    attribution: Mapping[str, Any],
    parameters: Mapping[str, np.ndarray],
    full_fit_audit: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    medians: Mapping[str, float],
    entropy_cuts: np.ndarray,
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(OUT / "rotation_mapping.csv", index=False)
    oof.to_parquet(OUT / "oof_predictions_2024.parquet", index=False)
    fold_audit.to_csv(OUT / "fold_audit_2024.csv", index=False)
    metrics.to_csv(OUT / "cell_diagnostics_2024.csv", index=False)
    comparison_summary.to_csv(OUT / "comparison_summary_2024.csv", index=False)
    rotations.to_csv(OUT / "rotation_diagnostics_2024.csv", index=False)
    cycle_axis_diagnostics(oof, "2024_oof").to_csv(
        OUT / "two_axis_cycle_diagnostics_2024.csv", index=False
    )
    np.savez_compressed(OUT / "model_parameters.npz", **parameters)
    write_json(
        OUT / "topology_feature_manifest.json",
        topology_feature_manifest(centroid_metadata, entropy_cuts),
    )
    write_json(OUT / "feature_manifest.json", feature_manifest(medians, entropy_cuts))
    write_json(OUT / "support_2024.json", support)
    write_json(OUT / "paired_pooled_gates_2024.json", gates)
    write_json(OUT / "provisional_source_attribution.json", attribution)
    write_json(OUT / "fit_source_hashes.json", source_hashes)
    write_json(OUT / "full_fit_audit.json", full_fit_audit)
    artifact_hashes = hash_named_files(OUT, fit_artifact_names())
    fit_complete = {
        "status": "fit_frozen_pending_independent_pre_score_audit",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "support_pass": bool(support["support_pass"]),
        "oof_compatible_rows": len(oof),
        "oof_realized_rows": int(oof["loop_occurs"].sum()),
        "oof_effective_weight": float(
            oof.loc[oof["loop_occurs"].eq(1), "conditional_weight"].sum()
        ),
        "provisional_source_attribution": attribution,
        "model_fit_performed": True,
        "oof_predictions_generated": True,
        "later_period_panels_read": False,
        "later_scoring_authorized": False,
        "scoring_authorized": False,
        "parent_grade_changed": False,
        "live_shadow_tree_read": False,
        "live_shadow_tree_written": False,
        "artifact_hashes": artifact_hashes,
    }
    write_json(OUT / "fit_complete.json", fit_complete)
    return fit_complete


def run_fit_only() -> dict[str, Any]:
    contract, source_hashes = validate_contract_and_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "fit_source_hashes_pre_fit.json", source_hashes)
    cycles, mapping, centroid_metadata = load_cycles_and_mapping()
    oof = prepare_oof(mapping)
    training = prepare_training(mapping)
    support = v3_support(oof)
    if not support["support_pass"]:
        raise AssertionError(f"V3 unique-cohort support failed: {support}")
    entropy_cuts = entropy_cutpoints(oof)
    add_entropy_quartile(oof, entropy_cuts)
    add_entropy_quartile(training, entropy_cuts)
    medians = load_numeric_medians()
    oof_predictions, fold_audit = fit_oof_models(training, oof, medians)
    validate_panel_probabilities(oof_predictions)
    metrics, comparison_summary, gates, rotations, attribution = evaluate_period(
        oof_predictions, "2024_oof", "oof"
    )
    parameters, full_fit_audit = fit_full_models(
        training, medians, centroid_metadata, entropy_cuts
    )
    current_hashes = hashes(source_paths_fit())
    if current_hashes != source_hashes:
        raise AssertionError("fit source changed during V3 fit phase")
    fit_complete = write_fit_artifacts(
        cycles,
        mapping,
        centroid_metadata,
        oof_predictions,
        fold_audit,
        support,
        metrics,
        comparison_summary,
        gates,
        rotations,
        attribution,
        parameters,
        full_fit_audit,
        source_hashes,
        medians,
        entropy_cuts,
    )
    return fit_complete


def validate_only() -> dict[str, Any]:
    _, source_hashes = validate_contract_and_sources()
    cycles, mapping, metadata = load_cycles_and_mapping()
    oof = prepare_oof(mapping)
    training = prepare_training(mapping)
    support = v3_support(oof)
    cuts = entropy_cutpoints(oof)
    return {
        "status": "validated_without_fit",
        "contract_sha256": CONTRACT_SHA256,
        "source_hash_count": len(source_hashes),
        "cycles": len(cycles),
        "rotation_mapping_rows": len(mapping),
        "topology_width": len(metadata["topology_columns"]),
        "oof_rows": len(oof),
        "training_rows": len(training),
        "support": support,
        "entropy_quartile_cutpoints": cuts.tolist(),
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def load_parameters() -> dict[str, np.ndarray]:
    with np.load(OUT / "model_parameters.npz") as stored:
        return {name: stored[name].copy() for name in stored.files}


def scaler_from_parameters(parameters: Mapping[str, np.ndarray]) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.scale_ = np.asarray(parameters["context_scaler_scale"]).copy()
    scaler.mean_ = np.asarray(parameters["context_scaler_mean"]).copy()
    scaler.var_ = np.asarray(parameters["context_scaler_var"]).copy()
    scaler.n_features_in_ = len(scaler.scale_)
    scaler.n_samples_seen_ = 1
    return scaler


def scoring_source_paths() -> dict[str, Path]:
    return {
        "quality_scoring_2025.parquet": PARENT_ROOT / "quality_scoring_2025.parquet",
        "quality_scoring_2023.parquet": PARENT_ROOT / "quality_scoring_2023.parquet",
        "anchor_panel_2025.parquet": PRICE_ROOT / "anchor_panel_2025.parquet",
        "anchor_panel_2023.parquet": PRICE_ROOT / "anchor_panel_2023.parquet",
    }


def validate_fit_and_pre_score_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    fit_complete_path = OUT / "fit_complete.json"
    audit_path = OUT / "pre_score_audit.json"
    if not fit_complete_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("fit freeze and independent pre-score audit are required")
    fit_complete = json.loads(fit_complete_path.read_text())
    audit = json.loads(audit_path.read_text())
    if fit_complete["contract_sha256"] != CONTRACT_SHA256:
        raise AssertionError("fit marker contract hash changed")
    if fit_complete["runner_sha256"] != sha256(Path(__file__)):
        raise AssertionError("runner changed after fit freeze")
    if not fit_complete["support_pass"]:
        raise AssertionError("fit support gate did not pass")
    for name, expected in fit_complete["artifact_hashes"].items():
        if sha256(OUT / name) != expected:
            raise AssertionError(f"fit artifact changed after freeze: {name}")
    fit_sources = json.loads((OUT / "fit_source_hashes.json").read_text())
    if hashes(source_paths_fit()) != fit_sources:
        raise AssertionError("fit sources changed after freeze")
    audit_pass = bool(audit.get("all_passed") or audit.get("pass"))
    if not audit_pass or audit.get("scoring_authorized") is not True:
        raise AssertionError("independent audit did not authorize later scoring")
    if audit.get("contract_sha256", CONTRACT_SHA256) != CONTRACT_SHA256:
        raise AssertionError("pre-score audit contract mismatch")
    return fit_complete, audit


def prepare_scoring_panel(
    scoring_path: Path,
    anchor_path: Path,
    mapping: pd.DataFrame,
    parameters: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "cycle_index",
        "cycle_id",
        "cycle",
        "transition_length",
        "state",
        "history_token",
        "loop_probability",
        "first_order_probability",
        "loop_occurs",
        "positive_cycle_count",
        "conditional_weight",
        *label_columns(),
        *parent_probability_columns(),
    ]
    panel = pd.read_parquet(scoring_path, columns=columns).reset_index(drop=True)
    panel.insert(0, "source_row", np.arange(len(panel), dtype=np.int64))
    panel = merge_anchor_controls(panel, anchor_path)
    panel = add_topology(panel, mapping)
    panel = panel.sort_values("source_row", kind="stable").reset_index(drop=True)
    reuse_parent_probabilities(panel)
    cuts = np.asarray(parameters["entropy_quartile_cutpoints"], float)
    add_entropy_quartile(panel, cuts)
    medians = {
        name: float(parameters["context_numeric_medians"][index])
        for index, name in enumerate(NUMERIC_CONTROLS)
    }
    raw = raw_context(panel, medians)
    context = scaler_from_parameters(parameters).transform(raw).tocsr()
    matrices = feature_matrices(panel, context)
    for model in NEW_MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                key = task_key(model, target, horizon)
                probability = predict_from_parameters(matrices[model], parameters, key)
                add_probability_columns(panel, model, target, horizon, probability)
    validate_panel_probabilities(panel)
    return panel


def scoring_support(panel: pd.DataFrame) -> dict[str, Any]:
    positive = panel.loc[panel["loop_occurs"].eq(1)]
    weight = float(positive["conditional_weight"].sum())
    return {
        "compatible_rows": len(panel),
        "realized_rows": len(positive),
        "effective_weight": weight,
        "minimum_effective_weight": 25000.0,
        "support_pass": weight >= 25000.0,
        "quarters": sorted(positive["quarter"].astype(str).unique()),
        "stocks": int(positive["symbol_norm"].nunique()),
    }


def cycle_axis_diagnostics(panel: pd.DataFrame, period: str) -> pd.DataFrame:
    positive = panel.loc[panel["loop_occurs"].eq(1)].reset_index(drop=True)
    rows = []
    final_grades = pd.read_csv(PARENT_ROOT / "final_cycle_tiers.csv").set_index("cycle_id")
    for cycle_id, frame in positive.groupby("cycle_id", sort=True):
        weights = frame["conditional_weight"].to_numpy(float)
        high_level_cells = 0
        observed_rates = []
        mean_qfull = []
        for target in TARGETS:
            for horizon in HORIZONS:
                observed = (
                    frame[f"quality_class__{target}__h{horizon}"].to_numpy(int) >= 1
                ).astype(int)
                rate = weighted_mean(observed, weights)
                probability = frame[
                    f"qfull__{target}__h{horizon}__p75"
                ].to_numpy(float)
                mean_probability = weighted_mean(probability, weights)
                observed_rates.append(rate)
                mean_qfull.append(mean_probability)
                high_level_cells += int(rate >= 0.35 and mean_probability >= 0.35)
        pair_differences = {}
        for name, candidate, baseline in (
            ("qfull_vs_qcontext", "qfull", "qcontext"),
            ("qroute_vs_qcontext", "qroute_topology", "qcontext"),
            ("qmain_vs_qroute", "qcycle_main", "qroute_topology"),
            ("qstate_vs_qmain", "qcycle_state", "qcycle_main"),
            ("qfull_vs_qstate", "qfull", "qcycle_state"),
        ):
            values = []
            for target in TARGETS:
                for horizon in HORIZONS:
                    for tier in TIERS:
                        observed = cell_observed(frame, "conditional", target, horizon, tier)
                        candidate_probability = frame[
                            probability_column(candidate, target, horizon, tier, "conditional")
                        ].to_numpy(float)
                        baseline_probability = frame[
                            probability_column(baseline, target, horizon, tier, "conditional")
                        ].to_numpy(float)
                        difference = (
                            binary_losses(observed, candidate_probability)["log_loss"]
                            - binary_losses(observed, baseline_probability)["log_loss"]
                        )
                        values.append(weighted_mean(difference, weights))
            pair_differences[name] = float(np.mean(values))
        rows.append(
            {
                "period": period,
                "cycle_id": cycle_id,
                "cycle": str(frame["cycle"].iloc[0]) if "cycle" in frame else "",
                "realized_rows": len(frame),
                "weight": float(weights.sum()),
                "p75_high_level_cells_of_6": high_level_cells,
                "minimum_p75_observed_rate": min(observed_rates),
                "minimum_p75_mean_qfull": min(mean_qfull),
                "absolute_high_period": high_level_cells == 6,
                **{f"pooled_log_loss_difference__{key}": value for key, value in pair_differences.items()},
                "frozen_parent_grade": final_grades.loc[cycle_id, "final_grade"],
                "parent_grade_changed": False,
                "prospective_validated": False,
            }
        )
    return pd.DataFrame(rows)


def portable_attribution(
    provisional: Mapping[str, Any], period_gates: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    provisional_pass = provisional["comparison_pass"]
    portable_pass = {
        name: bool(
            passed
            and period_gates["2025"]["comparison_pass"].get(name, False)
            and period_gates["2023"]["comparison_pass"].get(name, False)
        )
        for name, passed in provisional_pass.items()
    }
    provisional_label = provisional["label"]
    required_by_label = {
        "no_reference_signal": (),
        "topology_sufficient": (
            "qfull_vs_qcontext_reference",
            "qroute_topology_vs_qcontext",
            "qroute_topology_noninferiority_vs_qfull",
        ),
        "topology_dominant_with_residual_detail": (
            "qfull_vs_qcontext_reference",
            "qroute_topology_vs_qcontext",
            "qroute_topology_noninferiority_vs_qfull",
        ),
        "cycle_identity_representation_needed": (
            "qfull_vs_qcontext_reference",
            "qcycle_main_vs_qroute_topology",
        ),
        "current_state_rotation_needed": (
            "qfull_vs_qcontext_reference",
            "qcycle_state_vs_qcycle_main",
        ),
        "history_token_needed": (
            "qfull_vs_qcontext_reference",
            "qfull_vs_qcycle_state",
        ),
        "unresolved": ("qfull_vs_qcontext_reference",),
    }
    required = required_by_label[provisional_label]
    retained = bool(all(portable_pass.get(name, False) for name in required))
    if provisional_label == "no_reference_signal":
        final_label = provisional_label
    elif retained:
        final_label = provisional_label
    else:
        final_label = "unresolved_not_portable"
    return {
        "provisional_2024_label": provisional_label,
        "final_development_portability_label": final_label,
        "provisional_label_retained": retained,
        "portable_comparison_pass": portable_pass,
        "later_period_promotion_performed": False,
        "prospective_validated": False,
        "parent_grade_changed": False,
    }


def run_scoring() -> dict[str, Any]:
    fit_complete, pre_score_audit = validate_fit_and_pre_score_lock()
    evaluation_sources = scoring_source_paths()
    evaluation_hashes = hashes(evaluation_sources)
    parameters = load_parameters()
    mapping = pd.read_csv(OUT / "rotation_mapping.csv")
    specs = {
        "2025": (
            evaluation_sources["quality_scoring_2025.parquet"],
            evaluation_sources["anchor_panel_2025.parquet"],
        ),
        "2023": (
            evaluation_sources["quality_scoring_2023.parquet"],
            evaluation_sources["anchor_panel_2023.parquet"],
        ),
    }
    period_gates = {}
    axes = [pd.read_csv(OUT / "two_axis_cycle_diagnostics_2024.csv")]
    supports = {}
    for period, (scoring_path, anchor_path) in specs.items():
        panel = prepare_scoring_panel(scoring_path, anchor_path, mapping, parameters)
        support = scoring_support(panel)
        if not support["support_pass"]:
            raise AssertionError(f"scoring support failed in {period}: {support}")
        metrics, comparisons, gates, rotations, attribution = evaluate_period(
            panel, period, "scoring"
        )
        panel.to_parquet(OUT / f"scoring_predictions_{period}.parquet", index=False)
        metrics.to_csv(OUT / f"cell_diagnostics_{period}.csv", index=False)
        comparisons.to_csv(OUT / f"comparison_summary_{period}.csv", index=False)
        rotations.to_csv(OUT / f"rotation_diagnostics_{period}.csv", index=False)
        write_json(OUT / f"paired_pooled_gates_{period}.json", gates)
        write_json(OUT / f"support_{period}.json", support)
        period_gates[period] = gates
        supports[period] = support
        axes.append(cycle_axis_diagnostics(panel, period))
    provisional = json.loads((OUT / "provisional_source_attribution.json").read_text())
    transfer = portable_attribution(provisional, period_gates)
    write_json(OUT / "period_transfer_gates.json", transfer)
    write_json(OUT / "source_attribution.json", transfer)
    pd.concat(axes, ignore_index=True).to_csv(
        OUT / "two_axis_cycle_diagnostics.csv", index=False
    )
    write_json(OUT / "evaluation_source_hashes.json", evaluation_hashes)
    scoring_names = [
        name
        for period in ("2025", "2023")
        for name in (
            f"scoring_predictions_{period}.parquet",
            f"cell_diagnostics_{period}.csv",
            f"comparison_summary_{period}.csv",
            f"rotation_diagnostics_{period}.csv",
            f"paired_pooled_gates_{period}.json",
            f"support_{period}.json",
        )
    ] + [
        "period_transfer_gates.json",
        "source_attribution.json",
        "two_axis_cycle_diagnostics.csv",
        "evaluation_source_hashes.json",
    ]
    if hashes(evaluation_sources) != evaluation_hashes:
        raise AssertionError("evaluation source changed during scoring")
    scoring_complete = {
        "status": "scoring_complete_development_and_backward_portability_only",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "contract_sha256": CONTRACT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "pre_score_audit_sha256": sha256(OUT / "pre_score_audit.json"),
        "pre_score_audit_passed": True,
        "supports": supports,
        "source_attribution": transfer,
        "later_periods_are_prospective": False,
        "parent_grade_changed": False,
        "live_shadow_tree_read": False,
        "live_shadow_tree_written": False,
        "artifact_hashes": hash_named_files(OUT, scoring_names),
    }
    write_json(OUT / "scoring_complete.json", scoring_complete)
    summary = {
        **scoring_complete,
        "fit_complete_sha256": sha256(OUT / "fit_complete.json"),
        "interpretation": "Movement magnitude and future range source attribution only; no direction, signed return, P&L, economic edge, tradability, order, or deployment claim.",
    }
    write_json(OUT / "summary.json", summary)
    return summary


def self_tests() -> dict[str, Any]:
    checks = {}
    checks["contract_hash"] = sha256(CONTRACT) == CONTRACT_SHA256
    checks["topology_width"] = len(TOPOLOGY_COLUMNS) == TOPOLOGY_WIDTH
    checks["periodic_rotation_dedup"] = compatible_rotations((0, 1, 0, 1), 0) == (
        (0, 1, 0, 1, 0),
    )
    checks["ambiguous_rotation"] = len(
        compatible_rotations((1, 2, 1, 3), 1)
    ) == 2
    values = np.linspace(-0.02, 0.01, 30)
    checks["bootstrap_deterministic"] = moving_block_interval(
        values, 1, draws=100
    ) == moving_block_interval(values, 1, draws=100)
    checks["comparison_family"] = len(COMPARISONS) == 5
    checks["safety_labels"] = (
        json.loads(CONTRACT.read_text())["research_only"] is True
        and json.loads(CONTRACT.read_text())["live_ordering_enabled"] is False
    )
    if not all(checks.values()):
        raise AssertionError(f"V3 self-test failure: {checks}")
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
        raise SystemExit("choose only one phase")
    tests = self_tests()
    if args.validate_only:
        result = validate_only()
    elif args.fit_only:
        result = run_fit_only()
    else:
        result = run_scoring()
    result = {"self_tests": tests, **result}
    print(json.dumps(safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
