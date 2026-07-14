"""Research-only per-loop movement-quality probability experiment.

The frozen last-three-state probability remains the structural loop forecast.
This runner estimates the conditional probability that a *realised* compatible
loop is accompanied by movement above globally frozen 2024 thresholds.  It
then forms the chain-rule joint probability

    P(loop and movement tier | causal information)
      = P_frozen(loop | causal state history)
        * P(movement tier | loop, causal information).

Fitting and temperature selection use 2024 only.  The default scoring path is
sealed behind a completed fit freeze and a passing independent pre-score
audit.  No direction, signed-return claim, P&L, order, broker, runtime, or
deployment surface exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CONTRACT = HERE / "contracts/20260710-per-loop-movement-quality-v1.json"
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
PATH_ROOT = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710")
OUT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")

ANCHOR_2024 = PRICE_ROOT / "anchor_panel_train_2024.parquet"
ANCHOR_2025 = PRICE_ROOT / "anchor_panel_2025.parquet"
ANCHOR_2023 = PRICE_ROOT / "anchor_panel_2023.parquet"
FEATURE_MANIFEST = PRICE_ROOT / "feature_manifest.json"
PRICE_GATES = PRICE_ROOT / "gates.json"
PRICE_AUDIT = PRICE_ROOT / "independent_artifact_audit.json"
PATH_PARAMETERS = PATH_ROOT / "model_parameters.npz"
PATH_GATES = PATH_ROOT / "gates.json"
PATH_AUDIT = PATH_ROOT / "independent_artifact_audit.json"
CYCLE_PATH = PATH_ROOT / "fixed_cycles.csv"

SEED = 20260710
K = 8
END_STATE = K
TOKEN_COUNT = (K + 1) * (K + 1) * K
CYCLE_COUNT = 20
HORIZONS = (6, 12, 24)
TARGETS = ("absolute_return_bps", "future_range_bps")
MODELS = ("qcontext", "qcycle")
TIERS = ("p75", "p90")
QUANTILES = (0.75, 0.90)
TEMPERATURE_GRID = (0.75, 1.0, 1.25, 1.5, 2.0)
OOF_MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
MODEL_C = 0.2
MAX_ITER = 1000
EPSILON = 1e-12
CONTEXT_WIDTH = K + 9
QCYCLE_WIDTH = CONTEXT_WIDTH + CYCLE_COUNT + CYCLE_COUNT * K + CYCLE_COUNT * TOKEN_COUNT

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
LOOP_SCORE_COLUMNS = tuple(f"loop_score_{index:02d}" for index in range(1, 21))
OUTCOME_COLUMNS = tuple(
    f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS
)
FUTURE_STATE_COLUMNS = tuple(f"future_state_{step}" for step in range(1, 5))

PROTECTED_PATHS = (
    WORKSPACE / "work/contracts/20260710-frozen-loop-movement-shadow-v1.json",
    WORKSPACE
    / "work/contracts/20260710-frozen-loop-movement-shadow-v1-manifest.json",
    WORKSPACE / "work/frozen_loop_movement_shadow_core.py",
    WORKSPACE / "work/run_frozen_loop_movement_shadow.py",
    WORKSPACE / "work/tests/test_frozen_loop_movement_shadow.py",
    WORKSPACE / "work/reports/20260710-frozen-loop-movement-shadow-harness.md",
    WORKSPACE
    / "work/reports/20260710-frozen-loop-movement-prospective-shadow-contract.md",
    WORKSPACE / "work/shadow_validation/frozen_loop_movement_shadow_v1",
)


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        safe(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_protected_tree() -> dict[str, Any]:
    """Content-only snapshot of the already frozen prospective movement tree."""

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in PROTECTED_PATHS:
        candidates = [root]
        if root.is_dir():
            candidates.extend(sorted(root.rglob("*")))
        for candidate in candidates:
            path = candidate.resolve()
            if path in seen:
                continue
            seen.add(path)
            if not path.exists() and not path.is_symlink():
                raise FileNotFoundError(path)
            details = path.lstat()
            if path.is_symlink():
                kind = "symlink"
                content_hash = hashlib.sha256(os.readlink(path).encode()).hexdigest()
            elif path.is_dir():
                kind = "directory"
                content_hash = None
            elif path.is_file():
                kind = "file"
                content_hash = sha256(path)
            else:
                kind = "other"
                content_hash = None
            rows.append(
                {
                    "path": str(path.relative_to(WORKSPACE.resolve())),
                    "kind": kind,
                    "mode": stat.S_IMODE(details.st_mode),
                    "size": details.st_size,
                    "sha256": content_hash,
                }
            )
    rows.sort(key=lambda row: row["path"])
    runtime_path = (
        WORKSPACE
        / "work/shadow_validation/frozen_loop_movement_shadow_v1/runtime_metadata.json"
    )
    ledger = (
        WORKSPACE
        / "work/shadow_validation/frozen_loop_movement_shadow_v1/prediction_ledger.jsonl"
    )
    runtime = json.loads(runtime_path.read_text())
    payload = {
        "files": rows,
        "file_count": len(rows),
        "tree_sha256": canonical_json_hash(rows),
        "runtime_outcomes_opened": runtime.get("outcomes_opened"),
        "ledger_size": ledger.stat().st_size,
        "ledger_lines": len(ledger.read_text().splitlines()),
        "ledger_sha256": sha256(ledger),
    }
    if payload["runtime_outcomes_opened"] is not False:
        raise AssertionError("prospective movement runtime has opened outcomes")
    return payload


def canonical_cycle(core: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in core)
    if not values:
        raise ValueError("empty cycle")
    return min(values[index:] + values[:index] for index in range(len(values)))


def oriented_paths(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current),)
            for index, state in enumerate(core)
            if int(state) == int(current)
        }
    )


def load_cycles() -> pd.DataFrame:
    source = pd.read_csv(CYCLE_PATH)
    if len(source) != CYCLE_COUNT:
        raise AssertionError("frozen cycle file must contain twenty cycles")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for index, row in enumerate(source.itertuples(index=False), start=1):
        expected_id = f"cycle_{index:02d}"
        if str(row.cycle_id) != expected_id:
            raise AssertionError("frozen cycle order changed")
        closed = tuple(int(part) for part in str(row.cycle).split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid closed cycle {row.cycle}")
        core = canonical_cycle(closed[:-1])
        if core in seen or len(core) not in (2, 3, 4):
            raise AssertionError(f"invalid or duplicate cycle {row.cycle}")
        if min(core) < 0 or max(core) >= K:
            raise AssertionError("cycle state outside frozen state range")
        seen.add(core)
        rows.append(
            {
                "cycle_index": index - 1,
                "cycle_id": expected_id,
                "cycle": "->".join(str(value) for value in core + (core[0],)),
                "transition_length": len(core),
                "core": core,
            }
        )
    return pd.DataFrame(rows)


def _contract_lookup(contract: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = contract
        try:
            for key in path:
                value = value[key]
            return value
        except (KeyError, TypeError):
            continue
    raise AssertionError(f"contract lacks any accepted field path: {paths}")


def validate_contract() -> dict[str, Any]:
    if not CONTRACT.is_file():
        raise FileNotFoundError(CONTRACT)
    contract = json.loads(CONTRACT.read_text())
    if contract.get("contract_id") != "per_loop_movement_quality_v1":
        raise AssertionError("unexpected movement-quality contract id")
    if contract.get("research_only") is not True:
        raise AssertionError("movement-quality contract is not research-only")
    if contract.get("live_ordering_enabled") is not False:
        raise AssertionError("movement-quality contract enables live ordering")
    if contract.get("order_placement") != "disabled":
        raise AssertionError("movement-quality contract enables order placement")
    if contract.get("economic_edge_claim", False) is not False:
        raise AssertionError("movement-quality contract permits an edge claim")

    if list(contract["cohort"]["horizons_bars"]) != list(HORIZONS):
        raise AssertionError("contract horizon drift")
    if list(contract["tier_rules_each_cycle_and_horizon"]["good"]["required_targets"]) != list(TARGETS):
        raise AssertionError("contract target drift")
    if contract["periods"]["fit_and_internal_forward_validation"] != 2024:
        raise AssertionError("contract fit-period drift")
    if contract["periods"]["internal_forward_validation_months"] != list(OOF_MONTHS):
        raise AssertionError("contract OOF schedule drift")
    if contract["periods"]["2026_permitted"] is not False:
        raise AssertionError("contract permits 2026")
    for model_name in MODELS:
        model = contract["models"][model_name]
        if (
            float(model["C"]) != MODEL_C
            or model["solver"] != "lbfgs"
            or int(model["max_iter"]) != MAX_ITER
            or int(model["random_state"]) != SEED
        ):
            raise AssertionError(f"contract {model_name} model drift")
    calibration = contract["internal_2024_oof_and_calibration"]
    if [float(value) for value in calibration["temperature_grid"]] != list(
        TEMPERATURE_GRID
    ):
        raise AssertionError("contract temperature grid drift")
    if contract["outcomes"]["comparison_operator"] != ">":
        raise AssertionError("contract threshold operator drift")
    blocks = contract["models"]["qcycle"]["cycle_feature_blocks"]
    expected_blocks = [
        ("cycle_one_hot", CYCLE_COUNT, 1.0),
        ("cycle_by_current_state", CYCLE_COUNT * K, 0.5),
        ("cycle_by_history_token", CYCLE_COUNT * TOKEN_COUNT, 0.25),
    ]
    observed_blocks = [
        (str(row["name"]), int(row["width"]), float(row["feature_scale"]))
        for row in blocks
    ]
    if observed_blocks != expected_blocks:
        raise AssertionError("contract hierarchical feature-block drift")
    return contract


def expected_history_token(frame: pd.DataFrame) -> np.ndarray:
    previous_2 = frame["previous_state_2"].to_numpy(dtype=int)
    previous_1 = frame["previous_state_1"].to_numpy(dtype=int)
    current = frame["state"].to_numpy(dtype=int)
    return ((previous_2 * (K + 1) + previous_1) * K + current)


def load_anchor_panel(path: Path, expected_year: int, period: str) -> pd.DataFrame:
    required = {
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "state",
        "previous_state_1",
        "previous_state_2",
        "history_token",
        "bar_index_in_session",
        *NUMERIC_CONTROLS,
        *FUTURE_STATE_COLUMNS,
        *OUTCOME_COLUMNS,
        *LOOP_SCORE_COLUMNS,
        *(f"exact_{horizon}" for horizon in HORIZONS),
    }
    frame = pd.read_parquet(path, columns=sorted(required))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"{period} anchor panel lacks {missing}")
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["start_timestamp"] = pd.to_datetime(
        frame["start_timestamp"], utc=True, errors="raise"
    )
    frame = frame.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {expected_year} or expected_year >= 2026:
        raise AssertionError(f"{period} year boundary failure")
    keys = ["symbol_norm", "session_date", "start_timestamp"]
    if frame.duplicated(keys).any() or frame["anchor_id"].duplicated().any():
        raise AssertionError(f"{period} duplicate anchor")
    if frame["bar_index_in_session"].astype(int).gt(53).any():
        raise AssertionError(f"{period} admitted an anchor after bar 53")
    if not all(frame[f"exact_{horizon}"].astype(bool).all() for horizon in HORIZONS):
        raise AssertionError(f"{period} contains an inexact forward price path")
    state = frame["state"].to_numpy(dtype=int)
    if state.min(initial=0) < 0 or state.max(initial=0) >= K:
        raise AssertionError(f"{period} state outside frozen range")
    history = frame["history_token"].to_numpy(dtype=int)
    if not np.array_equal(history, expected_history_token(frame)):
        raise AssertionError(f"{period} history-token mismatch")
    if history.min(initial=0) < 0 or history.max(initial=0) >= TOKEN_COUNT:
        raise AssertionError(f"{period} history token outside frozen range")
    outcomes = frame.loc[:, list(OUTCOME_COLUMNS)].to_numpy(dtype=float)
    scores = frame.loc[:, list(LOOP_SCORE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(outcomes).all() or not np.isfinite(scores).all():
        raise AssertionError(f"{period} has non-finite outcome or loop score")
    if scores.min(initial=0.0) < 0.0 or scores.max(initial=0.0) > 1.0 + 1e-9:
        raise AssertionError(f"{period} has invalid loop score")
    return frame


def path_occurrence(anchors: pd.DataFrame, path: tuple[int, ...]) -> np.ndarray:
    label = np.ones(len(anchors), dtype=bool)
    for step, destination in enumerate(path[1:], start=1):
        label &= (
            anchors[f"future_state_{step}"].to_numpy(dtype=int)
            == int(destination)
        )
    return label


def first_order_path_probability(
    core: tuple[int, ...], current: int, transition: np.ndarray
) -> float:
    probability = 0.0
    for path in oriented_paths(core, current):
        route_probability = 1.0
        for left, right in zip(path[:-1], path[1:], strict=True):
            route_probability *= float(transition[int(left), int(right)])
        probability += route_probability
    return float(np.clip(probability, 0.0, 1.0))


def expand_compatible_cycles(
    anchors: pd.DataFrame, cycles: pd.DataFrame, first_order: np.ndarray
) -> pd.DataFrame:
    if first_order.shape != (K, K + 1):
        raise AssertionError("invalid frozen first-order transition matrix")
    base_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "state",
        "previous_state_1",
        "previous_state_2",
        "history_token",
        *NUMERIC_CONTROLS,
        *FUTURE_STATE_COLUMNS,
        *OUTCOME_COLUMNS,
    ]
    rows: list[pd.DataFrame] = []
    for cycle in cycles.itertuples(index=False):
        core = tuple(int(value) for value in cycle.core)
        selected = anchors.loc[anchors["state"].isin(set(core)), base_columns].copy()
        selected = selected.reset_index(drop=True)
        occurrence = np.zeros(len(selected), dtype=bool)
        for current in sorted(set(core)):
            mask = selected["state"].eq(current).to_numpy()
            current_anchors = selected.loc[mask].reset_index(drop=True)
            current_occurrence = np.zeros(len(current_anchors), dtype=bool)
            for path in oriented_paths(core, current):
                current_occurrence |= path_occurrence(current_anchors, path)
            occurrence[mask] = current_occurrence
        selected["cycle_index"] = int(cycle.cycle_index)
        selected["cycle_id"] = str(cycle.cycle_id)
        selected["cycle"] = str(cycle.cycle)
        selected["transition_length"] = int(cycle.transition_length)
        selected["loop_probability"] = anchors.loc[
            anchors["state"].isin(set(core)),
            f"loop_score_{int(cycle.cycle_index) + 1:02d}",
        ].to_numpy(dtype=float)
        selected["loop_occurs"] = occurrence.astype(np.int8)
        selected["first_order_probability"] = [
            first_order_path_probability(core, int(current), first_order)
            for current in selected["state"].to_numpy(dtype=int)
        ]
        rows.append(selected)
    expanded = pd.concat(rows, ignore_index=True)
    expanded = expanded.sort_values(
        ["anchor_id", "cycle_index"], kind="stable"
    ).reset_index(drop=True)
    if expanded.duplicated(["anchor_id", "cycle_id"]).any():
        raise AssertionError("duplicate compatible anchor-cycle row")
    positive_count = expanded.groupby("anchor_id", sort=False)["loop_occurs"].transform(
        "sum"
    )
    expanded["positive_cycle_count"] = positive_count.astype(np.int16)
    expanded["conditional_weight"] = np.where(
        expanded["loop_occurs"].eq(1),
        1.0 / positive_count.clip(lower=1).to_numpy(dtype=float),
        0.0,
    )
    weights = expanded.loc[expanded["loop_occurs"].eq(1)].groupby(
        "anchor_id", sort=False
    )["conditional_weight"].sum()
    if not np.allclose(weights.to_numpy(dtype=float), 1.0):
        raise AssertionError("realised-loop weights do not sum to one per anchor")
    return expanded


def compute_thresholds(anchors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            values = anchors[f"{target}_{horizon}"].to_numpy(dtype=float)
            p75, p90 = np.quantile(values, QUANTILES, method="linear")
            if not 0.0 < p75 < p90:
                raise AssertionError("invalid 2024 movement thresholds")
            rows.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "p75_threshold_bps": float(p75),
                    "p90_threshold_bps": float(p90),
                    "quantile_method": "linear",
                    "training_anchors": len(anchors),
                }
            )
    return pd.DataFrame(rows)


def threshold_map(thresholds: pd.DataFrame) -> dict[tuple[str, int], tuple[float, float]]:
    return {
        (str(row.target), int(row.horizon)): (
            float(row.p75_threshold_bps),
            float(row.p90_threshold_bps),
        )
        for row in thresholds.itertuples(index=False)
    }


def quality_class(values: np.ndarray, p75: float, p90: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values > p90, 2, np.where(values > p75, 1, 0)).astype(np.int8)


def add_quality_classes(
    expanded: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    output = expanded.copy()
    mapping = threshold_map(thresholds)
    for target in TARGETS:
        for horizon in HORIZONS:
            p75, p90 = mapping[(target, horizon)]
            output[f"quality_class__{target}__h{horizon}"] = quality_class(
                output[f"{target}_{horizon}"].to_numpy(dtype=float), p75, p90
            )
    return output


def raw_context(
    frame: pd.DataFrame, numeric_medians: dict[str, float]
) -> sparse.csr_matrix:
    numeric = frame.loc[:, list(NUMERIC_CONTROLS)].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.fillna(pd.Series(numeric_medians))
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("non-finite causal context")
    state = frame["state"].to_numpy(dtype=int)
    state_matrix = sparse.csr_matrix(np.eye(K, dtype=np.float64)[state])
    result = sparse.hstack((state_matrix, sparse.csr_matrix(values)), format="csr")
    if result.shape[1] != CONTEXT_WIDTH:
        raise AssertionError("context feature-width drift")
    return result


def sparse_one_hot(indices: np.ndarray, width: int, scale: float = 1.0) -> sparse.csr_matrix:
    indices = np.asarray(indices, dtype=int)
    if indices.min(initial=0) < 0 or indices.max(initial=0) >= width:
        raise AssertionError("one-hot index outside declared width")
    return sparse.csr_matrix(
        (
            np.full(len(indices), float(scale), dtype=np.float64),
            (np.arange(len(indices)), indices),
        ),
        shape=(len(indices), width),
    )


def hierarchical_features(
    frame: pd.DataFrame, scaled_context: sparse.csr_matrix
) -> sparse.csr_matrix:
    cycle = frame["cycle_index"].to_numpy(dtype=int)
    state = frame["state"].to_numpy(dtype=int)
    token = frame["history_token"].to_numpy(dtype=int)
    cycle_block = sparse_one_hot(cycle, CYCLE_COUNT, 1.0)
    cycle_state = sparse_one_hot(cycle * K + state, CYCLE_COUNT * K, 0.5)
    cycle_token = sparse_one_hot(
        cycle * TOKEN_COUNT + token, CYCLE_COUNT * TOKEN_COUNT, 0.25
    )
    result = sparse.hstack(
        (scaled_context, cycle_block, cycle_state, cycle_token), format="csr"
    )
    if result.shape[1] != QCYCLE_WIDTH:
        raise AssertionError("hierarchical feature-width drift")
    return result


def fit_scaler(
    raw: sparse.csr_matrix, weights: np.ndarray
) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.fit(raw, sample_weight=np.asarray(weights, dtype=float))
    return scaler


def fit_multinomial(
    matrix: sparse.csr_matrix, target: np.ndarray, weights: np.ndarray
) -> LogisticRegression:
    target = np.asarray(target, dtype=int)
    if not np.array_equal(np.unique(target), np.asarray([0, 1, 2])):
        raise AssertionError("multinomial quality target lacks a class")
    model = LogisticRegression(
        C=MODEL_C,
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=SEED,
    )
    model.fit(matrix, target, sample_weight=np.asarray(weights, dtype=float))
    if not np.array_equal(model.classes_, np.asarray([0, 1, 2])):
        raise AssertionError("quality model class order changed")
    return model


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 3 or temperature <= 0.0:
        raise AssertionError("invalid temperature-softmax input")
    scaled = logits / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    probability = exponential / exponential.sum(axis=1, keepdims=True)
    if not np.allclose(probability.sum(axis=1), 1.0):
        raise AssertionError("temperature probabilities do not normalize")
    return probability


def temperature_calibrate(
    raw_probability: np.ndarray, temperature: float
) -> np.ndarray:
    """Apply the contract's exact log-probability temperature transform."""

    raw_probability = np.asarray(raw_probability, dtype=float)
    if raw_probability.ndim != 2 or raw_probability.shape[1] != 3:
        raise AssertionError("invalid raw class-probability matrix")
    clipped = np.clip(raw_probability, EPSILON, 1.0)
    return softmax_temperature(np.log(clipped), temperature)


def weighted_multiclass_log_loss(
    target: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> float:
    target = np.asarray(target, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0)
    weights = np.asarray(weights, dtype=float)
    if len(target) != len(probability) or weights.sum() <= 0.0:
        raise AssertionError("invalid weighted loss input")
    losses = -np.log(probability[np.arange(len(target)), target])
    return float(np.average(losses, weights=weights))


def select_temperature(
    model_name: str,
    target_name: str,
    horizon: int,
    raw_probability: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for temperature in TEMPERATURE_GRID:
        probability = temperature_calibrate(raw_probability, temperature)
        rows.append(
            {
                "model": model_name,
                "target": target_name,
                "horizon": horizon,
                "temperature": temperature,
                "weighted_oof_log_loss": weighted_multiclass_log_loss(
                    observed, probability, weights
                ),
            }
        )
    ranking = sorted(
        rows,
        key=lambda row: (
            row["weighted_oof_log_loss"],
            abs(row["temperature"] - 1.0),
            row["temperature"],
        ),
    )
    selected = float(ranking[0]["temperature"])
    for row in rows:
        row["selected"] = bool(row["temperature"] == selected)
    return selected, rows


def model_key(model: str, target: str, horizon: int) -> str:
    return f"{model}__{target}__h{horizon}"


def validate_probability_outputs(frame: pd.DataFrame) -> None:
    probability_columns = [
        column
        for column in frame.columns
        if column == "loop_probability"
        or column == "first_order_probability"
        or "__raw_class_" in column
        or "__calibrated_class_" in column
        or column.endswith("__p75")
        or column.endswith("__p90")
    ]
    values = frame[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("non-finite quality probability")
    if values.min(initial=0.0) < -1e-12 or values.max(initial=0.0) > 1.0 + 1e-12:
        raise AssertionError("quality probability outside unit interval")
    for model_name in MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                key = model_key(model_name, target, horizon)
                for prefix in ("", "joint__"):
                    p75 = frame[f"{prefix}{key}__p75"].to_numpy(dtype=float)
                    p90 = frame[f"{prefix}{key}__p90"].to_numpy(dtype=float)
                    if (p90 > p75 + 1e-12).any():
                        raise AssertionError("ordered quality probabilities are not nested")
                raw_columns = [f"{key}__raw_class_{index}" for index in range(3)]
                calibrated_columns = [
                    f"{key}__calibrated_class_{index}" for index in range(3)
                ]
                if all(column in frame for column in raw_columns) and not np.allclose(
                    frame[raw_columns].sum(axis=1).to_numpy(dtype=float), 1.0
                ):
                    raise AssertionError("raw class probabilities do not normalize")
                if all(
                    column in frame for column in calibrated_columns
                ) and not np.allclose(
                    frame[calibrated_columns].sum(axis=1).to_numpy(dtype=float),
                    1.0,
                ):
                    raise AssertionError("calibrated class probabilities do not normalize")


def _task_target(frame: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    return frame[f"quality_class__{target}__h{horizon}"].to_numpy(dtype=int)


def expanding_month_oof(
    conditional: pd.DataFrame,
    expanded: pd.DataFrame,
    numeric_medians: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], pd.DataFrame]:
    dates = pd.to_datetime(conditional["session_date"], errors="raise")
    month = dates.dt.strftime("%Y-%m")
    expanded_dates = pd.to_datetime(expanded["session_date"], errors="raise")
    expanded_month = expanded_dates.dt.strftime("%Y-%m")
    raw = raw_context(conditional, numeric_medians)
    expanded_raw = raw_context(expanded, numeric_medians)
    weights = conditional["conditional_weight"].to_numpy(dtype=float)
    accum: dict[str, dict[str, list[np.ndarray]]] = {
        model_key(model, target, horizon): {
            "positions": [],
            "logits": [],
            "observed": [],
            "weights": [],
            "all_positions": [],
            "all_logits": [],
        }
        for model in MODELS
        for target in TARGETS
        for horizon in HORIZONS
    }
    fold_rows: list[dict[str, Any]] = []
    for fold_index, validation_month in enumerate(OOF_MONTHS, start=1):
        train_mask = month.lt(validation_month).to_numpy()
        validation_mask = month.eq(validation_month).to_numpy()
        train_positions = np.flatnonzero(train_mask)
        validation_positions = np.flatnonzero(validation_mask)
        all_validation_positions = np.flatnonzero(
            expanded_month.eq(validation_month).to_numpy()
        )
        if len(train_positions) == 0 or len(validation_positions) == 0:
            raise AssertionError(f"empty expanding-month fold {validation_month}")
        train_weights = weights[train_positions]
        scaler = fit_scaler(raw[train_positions], train_weights)
        context_train = scaler.transform(raw[train_positions]).tocsr()
        context_validation = scaler.transform(raw[validation_positions]).tocsr()
        context_all_validation = scaler.transform(
            expanded_raw[all_validation_positions]
        ).tocsr()
        cycle_train = hierarchical_features(
            conditional.iloc[train_positions].reset_index(drop=True), context_train
        )
        cycle_validation = hierarchical_features(
            conditional.iloc[validation_positions].reset_index(drop=True),
            context_validation,
        )
        cycle_all_validation = hierarchical_features(
            expanded.iloc[all_validation_positions].reset_index(drop=True),
            context_all_validation,
        )
        matrices = {
            "qcontext": (
                context_train,
                context_validation,
                context_all_validation,
            ),
            "qcycle": (cycle_train, cycle_validation, cycle_all_validation),
        }
        for target in TARGETS:
            for horizon in HORIZONS:
                observed_all = _task_target(conditional, target, horizon)
                train_target = observed_all[train_positions]
                validation_target = observed_all[validation_positions]
                for model_name, (
                    train_x,
                    validation_x,
                    all_validation_x,
                ) in matrices.items():
                    fitted = fit_multinomial(train_x, train_target, train_weights)
                    key = model_key(model_name, target, horizon)
                    accum[key]["positions"].append(validation_positions.copy())
                    accum[key]["logits"].append(fitted.decision_function(validation_x))
                    accum[key]["observed"].append(validation_target.copy())
                    accum[key]["weights"].append(weights[validation_positions].copy())
                    accum[key]["all_positions"].append(
                        all_validation_positions.copy()
                    )
                    accum[key]["all_logits"].append(
                        fitted.decision_function(all_validation_x)
                    )
                    fold_rows.append(
                        {
                            "fold": fold_index,
                            "validation_month": validation_month,
                            "training_rows": len(train_positions),
                            "validation_rows": len(validation_positions),
                            "training_weight": float(train_weights.sum()),
                            "validation_weight": float(weights[validation_positions].sum()),
                            "model": model_name,
                            "target": target,
                            "horizon": horizon,
                            "n_iter": int(fitted.n_iter_[0]),
                        }
                    )

    selected_temperatures: dict[str, float] = {}
    temperature_rows: list[dict[str, Any]] = []
    oof = expanded.loc[
        expanded_month.isin(OOF_MONTHS),
        [
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
            *OUTCOME_COLUMNS,
        ],
    ].copy()
    oof["source_position"] = oof.index.to_numpy(dtype=int)
    oof = oof.sort_values("source_position", kind="stable").reset_index(drop=True)
    expected_all_positions = oof["source_position"].to_numpy(dtype=int)
    expected_conditional_positions = np.flatnonzero(month.isin(OOF_MONTHS).to_numpy())
    for target in TARGETS:
        for horizon in HORIZONS:
            observed_column = f"quality_class__{target}__h{horizon}"
            oof[observed_column] = _task_target(
                expanded.iloc[expected_all_positions], target, horizon
            )
            observed_class = oof[observed_column].to_numpy(dtype=int)
            loop_occurs = oof["loop_occurs"].eq(1).to_numpy()
            oof[f"joint_good_target__{target}__h{horizon}"] = (
                loop_occurs & (observed_class >= 1)
            ).astype(np.int8)
            oof[f"joint_high_target__{target}__h{horizon}"] = (
                loop_occurs & (observed_class == 2)
            ).astype(np.int8)
            for model_name in MODELS:
                key = model_key(model_name, target, horizon)
                positions = np.concatenate(accum[key]["positions"])
                logits = np.vstack(accum[key]["logits"])
                observed = np.concatenate(accum[key]["observed"])
                task_weights = np.concatenate(accum[key]["weights"])
                order = np.argsort(positions, kind="stable")
                positions = positions[order]
                logits = logits[order]
                observed = observed[order]
                task_weights = task_weights[order]
                if not np.array_equal(positions, expected_conditional_positions):
                    raise AssertionError("OOF row alignment drift")
                raw_probability = softmax_temperature(logits, 1.0)
                selected, rows = select_temperature(
                    model_name,
                    target,
                    horizon,
                    raw_probability,
                    observed,
                    task_weights,
                )
                selected_temperatures[key] = selected
                temperature_rows.extend(rows)
                all_positions = np.concatenate(accum[key]["all_positions"])
                all_logits = np.vstack(accum[key]["all_logits"])
                all_order = np.argsort(all_positions, kind="stable")
                all_positions = all_positions[all_order]
                all_logits = all_logits[all_order]
                if not np.array_equal(all_positions, expected_all_positions):
                    raise AssertionError("all-compatible OOF row alignment drift")
                all_raw_probability = softmax_temperature(all_logits, 1.0)
                probability = temperature_calibrate(all_raw_probability, selected)
                for class_index in range(3):
                    oof[f"{key}__raw_class_{class_index}"] = all_raw_probability[
                        :, class_index
                    ]
                    oof[
                        f"{key}__calibrated_class_{class_index}"
                    ] = probability[:, class_index]
                oof[f"{key}__p75"] = probability[:, 1] + probability[:, 2]
                oof[f"{key}__p90"] = probability[:, 2]
                oof[f"joint__{key}__p75"] = (
                    oof["loop_probability"].to_numpy(dtype=float)
                    * oof[f"{key}__p75"].to_numpy(dtype=float)
                )
                oof[f"joint__{key}__p90"] = (
                    oof["loop_probability"].to_numpy(dtype=float)
                    * oof[f"{key}__p90"].to_numpy(dtype=float)
                )
                if (oof[f"{key}__p90"] > oof[f"{key}__p75"] + 1e-12).any():
                    raise AssertionError("nested OOF quality probability failure")
    oof = oof.drop(columns="source_position")
    temperature = pd.DataFrame(temperature_rows).sort_values(
        ["model", "target", "horizon", "temperature"], kind="stable"
    ).reset_index(drop=True)
    folds = pd.DataFrame(fold_rows)
    validate_probability_outputs(oof)
    return oof, temperature, selected_temperatures, folds


def fit_full_models(
    conditional: pd.DataFrame,
    numeric_medians: dict[str, float],
    temperatures: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    weights = conditional["conditional_weight"].to_numpy(dtype=float)
    raw = raw_context(conditional, numeric_medians)
    scaler = fit_scaler(raw, weights)
    context = scaler.transform(raw).tocsr()
    cycle = hierarchical_features(conditional, context)
    matrices = {"qcontext": context, "qcycle": cycle}
    parameters: dict[str, np.ndarray] = {
        "context_scaler_scale": scaler.scale_.copy(),
        "context_scaler_mean": scaler.mean_.copy(),
        "context_scaler_var": scaler.var_.copy(),
        "context_numeric_medians": np.asarray(
            [numeric_medians[name] for name in NUMERIC_CONTROLS], dtype=float
        ),
    }
    audit: dict[str, Any] = {"models": {}}
    for model_name, matrix in matrices.items():
        for target in TARGETS:
            for horizon in HORIZONS:
                key = model_key(model_name, target, horizon)
                model = fit_multinomial(
                    matrix, _task_target(conditional, target, horizon), weights
                )
                parameters[f"{key}__classes"] = model.classes_.copy()
                parameters[f"{key}__coef"] = model.coef_.copy()
                parameters[f"{key}__intercept"] = model.intercept_.copy()
                parameters[f"{key}__n_iter"] = model.n_iter_.copy()
                parameters[f"{key}__temperature"] = np.asarray(
                    [temperatures[key]], dtype=float
                )
                audit["models"][key] = {
                    "feature_width": matrix.shape[1],
                    "n_iter": int(model.n_iter_[0]),
                    "temperature": temperatures[key],
                }
    return parameters, audit


def training_support(expanded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cycle_id, cycle_frame in expanded.groupby("cycle_id", sort=True):
        occurs = cycle_frame["loop_occurs"].eq(1).to_numpy()
        for target in TARGETS:
            for horizon in HORIZONS:
                quality = _task_target(cycle_frame, target, horizon)
                good = occurs & (quality >= 1)
                high = occurs & (quality == 2)
                rows.append(
                    {
                        "cycle_id": cycle_id,
                        "cycle": str(cycle_frame["cycle"].iloc[0]),
                        "transition_length": int(
                            cycle_frame["transition_length"].iloc[0]
                        ),
                        "target": target,
                        "horizon": horizon,
                        "compatible_rows": len(cycle_frame),
                        "realised_occurrences": int(occurs.sum()),
                        "realised_stocks": int(
                            cycle_frame.loc[occurs, "symbol_norm"].nunique()
                        ),
                        "realised_quarters": int(
                            cycle_frame.loc[occurs, "quarter"].nunique()
                        ),
                        "good_joint_events": int(good.sum()),
                        "high_joint_events": int(high.sum()),
                    }
                )
    return pd.DataFrame(rows)


def _fit_source_paths() -> dict[str, Path]:
    return {
        "contract.json": CONTRACT,
        "runner.py": Path(__file__),
        "anchor_panel_train_2024.parquet": ANCHOR_2024,
        "fixed_cycles.csv": CYCLE_PATH,
        "feature_manifest.json": FEATURE_MANIFEST,
        "path_model_parameters.npz": PATH_PARAMETERS,
        "path_gates.json": PATH_GATES,
        "path_independent_artifact_audit.json": PATH_AUDIT,
        "price_gates.json": PRICE_GATES,
        "price_independent_artifact_audit.json": PRICE_AUDIT,
    }


def self_tests() -> dict[str, Any]:
    assert canonical_cycle((2, 5, 1)) == canonical_cycle((5, 1, 2))
    assert oriented_paths((1, 2, 1), 1) == [(1, 1, 2, 1), (1, 2, 1, 1)]
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    classes = quality_class(values, 2.0, 4.0)
    assert np.array_equal(classes, np.asarray([0, 0, 1, 1, 2]))
    probability = softmax_temperature(
        np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]), 1.25
    )
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.all(probability[:, 2] <= probability[:, 1] + probability[:, 2])
    frame = pd.DataFrame(
        {
            "cycle_index": [0, 19],
            "state": [0, 7],
            "history_token": [0, TOKEN_COUNT - 1],
        }
    )
    context = sparse.csr_matrix(np.ones((2, CONTEXT_WIDTH)))
    hierarchical = hierarchical_features(frame, context)
    assert hierarchical.shape == (2, QCYCLE_WIDTH)
    assert np.allclose(hierarchical.sum(axis=1).A1, CONTEXT_WIDTH + 1.75)
    target = np.asarray([0, 1, 2])
    p = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    assert abs(weighted_multiclass_log_loss(target, p, np.ones(3)) + np.log(0.8)) < 1e-12
    return {
        "cycle_rotation": True,
        "strict_nested_threshold_classes": True,
        "temperature_normalization": True,
        "hierarchical_block_scaling": True,
        "weighted_multiclass_loss": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def fit_only() -> dict[str, Any]:
    contract = validate_contract()
    tests = self_tests()
    OUT.mkdir(parents=True, exist_ok=True)
    path_gates = json.loads(PATH_GATES.read_text())
    path_audit = json.loads(PATH_AUDIT.read_text())
    price_gates = json.loads(PRICE_GATES.read_text())
    price_audit = json.loads(PRICE_AUDIT.read_text())
    if path_gates.get("history_retained") is not True:
        raise AssertionError("retained structural loop model is unavailable")
    if path_audit.get("all_passed") is not True:
        raise AssertionError("structural loop-model audit failed")
    if price_gates.get("movement_consequence_retained") is not True:
        raise AssertionError("retained movement predecessor is unavailable")
    if price_audit.get("all_passed") is not True:
        raise AssertionError("movement predecessor audit failed")

    pre_snapshot = snapshot_protected_tree()
    write_json(OUT / "prospective_shadow_pre_content_snapshot.json", pre_snapshot)
    fit_sources = _fit_source_paths()
    write_json(
        OUT / "fit_source_hashes.json",
        {name: sha256(path) for name, path in fit_sources.items()},
    )

    manifest = json.loads(FEATURE_MANIFEST.read_text())
    if manifest.get("numeric_controls") != list(NUMERIC_CONTROLS):
        raise AssertionError("frozen causal control order changed")
    if manifest.get("loop_score_columns") != list(LOOP_SCORE_COLUMNS):
        raise AssertionError("frozen loop-score order changed")
    numeric_medians = {
        name: float(manifest["numeric_medians"][name]) for name in NUMERIC_CONTROLS
    }
    anchors = load_anchor_panel(ANCHOR_2024, 2024, "train_2024")
    cycles = load_cycles()
    cycles.drop(columns="core").to_csv(OUT / "fixed_cycles.csv", index=False)
    thresholds = compute_thresholds(anchors)
    contracted_thresholds = contract["outcomes"]["thresholds_bps"]
    for row in thresholds.itertuples(index=False):
        expected = contracted_thresholds[str(row.target)][str(int(row.horizon))]
        if not math.isclose(
            float(row.p75_threshold_bps),
            float(expected["p75"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(row.p90_threshold_bps),
            float(expected["p90"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("reconstructed 2024 threshold differs from contract")
    thresholds.to_csv(OUT / "quality_thresholds_2024.csv", index=False)
    write_json(
        OUT / "quality_thresholds_2024.json",
        {
            "fit_period": 2024,
            "quantile_method": "linear",
            "comparison_operator": ">",
            "thresholds_bps": contracted_thresholds,
        },
    )
    with np.load(PATH_PARAMETERS) as path_parameters:
        first_order = path_parameters["first_order"].copy()
    expanded = add_quality_classes(
        expand_compatible_cycles(anchors, cycles, first_order), thresholds
    )
    support = training_support(expanded)
    support.to_csv(OUT / "training_support_2024.csv", index=False)
    conditional = expanded.loc[expanded["loop_occurs"].eq(1)].copy()
    conditional = conditional.reset_index(drop=True)
    if conditional.empty or conditional["conditional_weight"].sum() <= 0.0:
        raise AssertionError("empty realised-loop conditional training surface")
    conditional.to_parquet(OUT / "training_long_2024.parquet", index=False)

    oof, selection, temperatures, folds = expanding_month_oof(
        conditional, expanded, numeric_medians
    )
    oof.to_parquet(OUT / "oof_predictions_2024.parquet", index=False)
    selection.to_csv(OUT / "temperature_selection_2024.csv", index=False)
    folds.to_csv(OUT / "oof_fold_audit_2024.csv", index=False)
    full_fit_eligibility = {
        cycle_id: _base_support_gate(group.reset_index(drop=True), "full_2024", contract)[
            "pass"
        ]
        for cycle_id, group in expanded.groupby("cycle_id", sort=True)
    }
    provisional = grade_period(
        oof,
        "2024_oof",
        "oof",
        contract,
        full_2024_eligibility=full_fit_eligibility,
    )
    provisional["support"].to_csv(
        OUT / "provisional_support_2024.csv", index=False
    )
    provisional["structural"].to_csv(
        OUT / "provisional_structural_2024.csv", index=False
    )
    provisional["cells"].to_csv(
        OUT / "provisional_quality_cells_2024.csv", index=False
    )
    provisional["horizons"].to_csv(
        OUT / "provisional_horizon_grades_2024.csv", index=False
    )
    provisional["cycles"].to_csv(
        OUT / "provisional_tiers_2024.csv", index=False
    )
    provisional_gates = {
        "scientific_status": "2024_internal_forward_provisional_only",
        "high_movement_quality_cycles": int(
            provisional["cycles"]["global_grade"]
            .eq("high_movement_quality")
            .sum()
        ),
        "good_movement_quality_cycles": int(
            provisional["cycles"]["global_grade"]
            .eq("good_movement_quality")
            .sum()
        ),
        "unqualified_cycles": int(
            provisional["cycles"]["global_grade"].eq("unqualified").sum()
        ),
        "promotion_permitted": False,
        "prospective_validated": False,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "provisional_gates_2024.json", provisional_gates)
    parameters, model_audit = fit_full_models(
        conditional, numeric_medians, temperatures
    )
    parameters["frozen_first_order"] = first_order
    np.savez_compressed(OUT / "quality_model_parameters.npz", **parameters)

    fit_manifest = {
        "contract_id": contract["contract_id"],
        "fit_period": 2024,
        "anchors": len(anchors),
        "compatible_anchor_cycle_rows": len(expanded),
        "realised_anchor_cycle_rows": len(conditional),
        "realised_anchor_weight": float(conditional["conditional_weight"].sum()),
        "targets": list(TARGETS),
        "horizons": list(HORIZONS),
        "classes": {
            "0": "movement <= frozen 2024 p75",
            "1": "frozen 2024 p75 < movement <= frozen 2024 p90",
            "2": "movement > frozen 2024 p90",
        },
        "weighting": "each realised anchor-cycle receives 1 / realised cycles at anchor",
        "oof_months": list(OOF_MONTHS),
        "temperature_grid": list(TEMPERATURE_GRID),
        "temperature_ranking": [
            "weighted_oof_log_loss",
            "absolute_temperature_distance_from_one",
            "temperature",
        ],
        "model": {
            "class": "LogisticRegression",
            "C": MODEL_C,
            "solver": "lbfgs",
            "max_iter": MAX_ITER,
            "random_state": SEED,
        },
        "context_feature_width": CONTEXT_WIDTH,
        "qcycle_feature_width": QCYCLE_WIDTH,
        "qcycle_blocks": {
            "scaled_context": CONTEXT_WIDTH,
            "cycle_one_hot_width": CYCLE_COUNT,
            "cycle_one_hot_scale": 1.0,
            "cycle_by_state_width": CYCLE_COUNT * K,
            "cycle_by_state_scale": 0.5,
            "cycle_by_history_token_width": CYCLE_COUNT * TOKEN_COUNT,
            "cycle_by_history_token_scale": 0.25,
            "post_block_rescaling": False,
        },
        "model_audit": model_audit,
        "direct_volume_input": False,
        "volume_note": (
            "Price-quality models use no volume directly. The frozen upstream "
            "state detector uses provider historical_volume features; this is "
            "not exchange-wide volume or order flow."
        ),
        "direction_excluded": True,
        "signed_return_excluded": True,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "fit_manifest.json", fit_manifest)
    write_json(
        OUT / "feature_manifest.json",
        {
            "numeric_controls": list(NUMERIC_CONTROLS),
            "numeric_medians": numeric_medians,
            "state_one_hot_width": K,
            "context_width": CONTEXT_WIDTH,
            "qcycle_width": QCYCLE_WIDTH,
            "cycle_one_hot": {"width": CYCLE_COUNT, "scale": 1.0},
            "cycle_by_current_state": {
                "width": CYCLE_COUNT * K,
                "scale": 0.5,
            },
            "cycle_by_history_token": {
                "width": CYCLE_COUNT * TOKEN_COUNT,
                "scale": 0.25,
            },
            "stock_identity_included": False,
            "future_label_or_outcome_included_as_feature": False,
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
        },
    )
    write_json(OUT / "self_tests.json", tests)
    if snapshot_protected_tree() != pre_snapshot:
        raise AssertionError("prospective movement shadow changed during 2024 fit")
    artifact_names = (
        "fixed_cycles.csv",
        "quality_thresholds_2024.csv",
        "quality_thresholds_2024.json",
        "training_long_2024.parquet",
        "training_support_2024.csv",
        "provisional_tiers_2024.csv",
        "provisional_support_2024.csv",
        "provisional_structural_2024.csv",
        "provisional_quality_cells_2024.csv",
        "provisional_horizon_grades_2024.csv",
        "provisional_gates_2024.json",
        "oof_predictions_2024.parquet",
        "temperature_selection_2024.csv",
        "oof_fold_audit_2024.csv",
        "quality_model_parameters.npz",
        "fit_manifest.json",
        "feature_manifest.json",
        "self_tests.json",
        "fit_source_hashes.json",
        "prospective_shadow_pre_content_snapshot.json",
    )
    fit_complete = {
        "artifact_hashes": {
            name: sha256(OUT / name) for name in artifact_names
        },
        "fit_source_hashes_sha256": sha256(OUT / "fit_source_hashes.json"),
        "contract_sha256": sha256(CONTRACT),
        "runner_sha256": sha256(Path(__file__)),
        "scoring_outcomes_opened": False,
        "scoring_authorized": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "fit_complete.json", fit_complete)
    return fit_complete


def load_frozen_parameters() -> dict[str, np.ndarray]:
    return dict(np.load(OUT / "quality_model_parameters.npz"))


def scaler_from_parameters(parameters: dict[str, np.ndarray]) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.scale_ = parameters["context_scaler_scale"].copy()
    scaler.mean_ = parameters["context_scaler_mean"].copy()
    scaler.var_ = parameters["context_scaler_var"].copy()
    scaler.n_features_in_ = CONTEXT_WIDTH
    return scaler


def predict_from_parameters(
    matrix: sparse.csr_matrix,
    parameters: dict[str, np.ndarray],
    key: str,
) -> np.ndarray:
    classes = parameters[f"{key}__classes"].astype(int)
    if not np.array_equal(classes, np.asarray([0, 1, 2])):
        raise AssertionError("stored quality class order changed")
    logits = matrix @ parameters[f"{key}__coef"].T
    logits = np.asarray(logits) + parameters[f"{key}__intercept"][None, :]
    temperature = float(parameters[f"{key}__temperature"][0])
    raw_probability = softmax_temperature(logits, 1.0)
    return temperature_calibrate(raw_probability, temperature)


def score_probabilities(
    expanded: pd.DataFrame,
    numeric_medians: dict[str, float],
    parameters: dict[str, np.ndarray],
) -> pd.DataFrame:
    raw = raw_context(expanded, numeric_medians)
    scaler = scaler_from_parameters(parameters)
    context = scaler.transform(raw).tocsr()
    cycle = hierarchical_features(expanded, context)
    matrices = {"qcontext": context, "qcycle": cycle}
    output = expanded.loc[
        :,
        [
            "anchor_id",
            "symbol_norm",
            "session_date",
            "quarter",
            "start_timestamp",
            "state",
            "history_token",
            "cycle_index",
            "cycle_id",
            "cycle",
            "transition_length",
            "loop_probability",
            "first_order_probability",
            "loop_occurs",
            "positive_cycle_count",
            "conditional_weight",
        ],
    ].copy()
    for target in TARGETS:
        for horizon in HORIZONS:
            quality_column = f"quality_class__{target}__h{horizon}"
            observed = expanded[quality_column].to_numpy(dtype=int)
            output[quality_column] = observed
            output[f"joint_good_target__{target}__h{horizon}"] = (
                expanded["loop_occurs"].eq(1).to_numpy() & (observed >= 1)
            ).astype(np.int8)
            output[f"joint_high_target__{target}__h{horizon}"] = (
                expanded["loop_occurs"].eq(1).to_numpy() & (observed == 2)
            ).astype(np.int8)
            for model_name, matrix in matrices.items():
                key = model_key(model_name, target, horizon)
                probability = predict_from_parameters(matrix, parameters, key)
                p75 = probability[:, 1] + probability[:, 2]
                p90 = probability[:, 2]
                output[f"{key}__p75"] = p75
                output[f"{key}__p90"] = p90
                output[f"joint__{key}__p75"] = (
                    expanded["loop_probability"].to_numpy(dtype=float) * p75
                )
                output[f"joint__{key}__p90"] = (
                    expanded["loop_probability"].to_numpy(dtype=float) * p90
                )
                if (p90 > p75 + 1e-12).any():
                    raise AssertionError("nested quality probability failure")
    validate_probability_outputs(output)
    return output


def binary_losses(target: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": -(
            target * np.log(probability)
            + (1.0 - target) * np.log(1.0 - probability)
        ),
        "brier": np.square(probability - target),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0.0:
        return math.nan
    return float(np.average(values, weights=weights))


def calibration_rows(
    period: str,
    surface: str,
    model: str,
    target_name: str,
    horizon: int,
    tier: str,
    observed: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    min_bin_support: int,
) -> tuple[list[dict[str, Any]], float, float]:
    observed = np.asarray(observed, dtype=float)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    bin_id = np.minimum((probability * 10.0).astype(int), 9)
    rows = []
    total_weight = weights.sum()
    ece = 0.0
    supported_errors = []
    for index in range(10):
        mask = bin_id == index
        count = int(mask.sum())
        weight = float(weights[mask].sum())
        mean_probability = weighted_mean(probability[mask], weights[mask])
        event_rate = weighted_mean(observed[mask], weights[mask])
        error = abs(mean_probability - event_rate) if weight > 0.0 else math.nan
        supported = count >= int(min_bin_support) and weight > 0.0
        if weight > 0.0:
            ece += weight / total_weight * error
        if supported:
            supported_errors.append(error)
        rows.append(
            {
                "period": period,
                "surface": surface,
                "model": model,
                "target": target_name,
                "horizon": horizon,
                "tier": tier,
                "bin": index,
                "rows": count,
                "weight": weight,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "absolute_error": error,
                "supported": supported,
            }
        )
    maximum = max(supported_errors) if supported_errors else math.nan
    return rows, float(ece), float(maximum)


def moving_block_bounds(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return math.nan, math.nan, math.nan
    block = min(5, len(clean))
    blocks = np.asarray(
        [clean[start : start + block] for start in range(len(clean) - block + 1)]
    )
    needed = int(math.ceil(len(clean) / block))
    rng = np.random.default_rng(seed)
    draws = np.empty(5000, dtype=float)
    for index in range(len(draws)):
        selected = rng.integers(0, len(blocks), size=needed)
        draws[index] = blocks[selected].reshape(-1)[: len(clean)].mean()
    return (
        float(clean.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def evaluate_scoring(scoring: pd.DataFrame, period: str, seed_offset: int) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    per_cycle_rows: list[dict[str, Any]] = []
    for surface in ("conditional", "joint"):
        if surface == "conditional":
            surface_mask = scoring["loop_occurs"].eq(1).to_numpy()
            weights = scoring.loc[surface_mask, "conditional_weight"].to_numpy(float)
        else:
            surface_mask = np.ones(len(scoring), dtype=bool)
            weights = np.ones(len(scoring), dtype=float)
        surface_frame = scoring.loc[surface_mask].reset_index(drop=True)
        for target_name in TARGETS:
            for horizon in HORIZONS:
                quality = surface_frame[
                    f"quality_class__{target_name}__h{horizon}"
                ].to_numpy(dtype=int)
                for tier_index, tier in enumerate(TIERS):
                    if surface == "conditional":
                        observed = quality >= (1 if tier == "p75" else 2)
                    else:
                        label = "good" if tier == "p75" else "high"
                        observed = surface_frame[
                            f"joint_{label}_target__{target_name}__h{horizon}"
                        ].to_numpy(dtype=int)
                    model_losses: dict[str, dict[str, np.ndarray]] = {}
                    for model_index, model_name in enumerate(MODELS):
                        key = model_key(model_name, target_name, horizon)
                        column = (
                            f"{key}__{tier}"
                            if surface == "conditional"
                            else f"joint__{key}__{tier}"
                        )
                        probability = surface_frame[column].to_numpy(dtype=float)
                        losses = binary_losses(observed, probability)
                        model_losses[model_name] = losses
                        rows, ece, maximum = calibration_rows(
                            period,
                            surface,
                            model_name,
                            target_name,
                            horizon,
                            tier,
                            observed,
                            probability,
                            weights,
                            100 if surface == "conditional" else 500,
                        )
                        calibration.extend(rows)
                        metric_rows.append(
                            {
                                "period": period,
                                "surface": surface,
                                "model": model_name,
                                "target": target_name,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(surface_frame),
                                "weight": float(weights.sum()),
                                "positives": int(np.asarray(observed).sum()),
                                "weighted_prevalence": weighted_mean(
                                    observed, weights
                                ),
                                "log_loss": weighted_mean(
                                    losses["log_loss"], weights
                                ),
                                "brier": weighted_mean(losses["brier"], weights),
                                "ece": ece,
                                "maximum_supported_bin_error": maximum,
                            }
                        )
                    for loss_index, loss_name in enumerate(("log_loss", "brier")):
                        difference = (
                            model_losses["qcycle"][loss_name]
                            - model_losses["qcontext"][loss_name]
                        )
                        weighted_difference = difference * weights
                        daily = pd.DataFrame(
                            {
                                "session_date": surface_frame[
                                    "session_date"
                                ].to_numpy(),
                                "weighted_difference": weighted_difference,
                                "weight": weights,
                            }
                        ).groupby("session_date", sort=True).sum()
                        daily_difference = (
                            daily["weighted_difference"] / daily["weight"]
                        ).to_numpy(dtype=float)
                        mean, low, high = moving_block_bounds(
                            daily_difference,
                            SEED
                            + seed_offset
                            + (0 if surface == "conditional" else 10000)
                            + TARGETS.index(target_name) * 1000
                            + horizon * 10
                            + tier_index * 2
                            + loss_index,
                        )
                        baseline = weighted_mean(
                            model_losses["qcontext"][loss_name], weights
                        )
                        comparison_rows.append(
                            {
                                "period": period,
                                "surface": surface,
                                "candidate": "qcycle",
                                "baseline": "qcontext",
                                "target": target_name,
                                "horizon": horizon,
                                "tier": tier,
                                "loss": loss_name,
                                "weighted_mean_difference": weighted_mean(
                                    difference, weights
                                ),
                                "daily_mean_difference": mean,
                                "daily_ci_low": low,
                                "daily_ci_high": high,
                                "baseline_loss": baseline,
                                "relative_improvement": -weighted_mean(
                                    difference, weights
                                )
                                / baseline,
                            }
                        )
                    for cycle_id, positions in surface_frame.groupby(
                        "cycle_id", sort=True
                    ).groups.items():
                        positions = np.asarray(positions, dtype=int)
                        cycle_weights = weights[positions]
                        per_cycle_rows.append(
                            {
                                "period": period,
                                "surface": surface,
                                "cycle_id": cycle_id,
                                "target": target_name,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(positions),
                                "weight": float(cycle_weights.sum()),
                                "positives": int(np.asarray(observed)[positions].sum()),
                                "qcontext_log_loss": weighted_mean(
                                    model_losses["qcontext"]["log_loss"][positions],
                                    cycle_weights,
                                ),
                                "qcycle_log_loss": weighted_mean(
                                    model_losses["qcycle"]["log_loss"][positions],
                                    cycle_weights,
                                ),
                                "qcontext_brier": weighted_mean(
                                    model_losses["qcontext"]["brier"][positions],
                                    cycle_weights,
                                ),
                                "qcycle_brier": weighted_mean(
                                    model_losses["qcycle"]["brier"][positions],
                                    cycle_weights,
                                ),
                            }
                        )
    return {
        "metrics": pd.DataFrame(metric_rows),
        "calibration": pd.DataFrame(calibration),
        "comparisons": pd.DataFrame(comparison_rows),
        "per_cycle": pd.DataFrame(per_cycle_rows),
    }


def _weighted_difference_by_group(
    frame: pd.DataFrame,
    difference: np.ndarray,
    weights: np.ndarray,
    column: str,
) -> pd.Series:
    grouped = pd.DataFrame(
        {
            "group": frame[column].astype(str).to_numpy(),
            "weighted": np.asarray(difference, dtype=float)
            * np.asarray(weights, dtype=float),
            "weight": np.asarray(weights, dtype=float),
        }
    ).groupby("group", sort=True).sum()
    if (grouped["weight"] <= 0.0).any():
        return pd.Series(dtype=float)
    return grouped["weighted"] / grouped["weight"]


def _calibration_summary(
    observed: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray,
    min_bin_support: int,
) -> tuple[float, float]:
    observed = np.asarray(observed, dtype=float)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(weights, dtype=float)
    bin_id = np.minimum((probability * 10.0).astype(int), 9)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return math.nan, math.nan
    ece = 0.0
    supported_errors: list[float] = []
    for index in range(10):
        mask = bin_id == index
        count = int(mask.sum())
        weight = float(weights[mask].sum())
        if weight <= 0.0:
            continue
        mean_probability = weighted_mean(probability[mask], weights[mask])
        event_rate = weighted_mean(observed[mask], weights[mask])
        error = abs(mean_probability - event_rate)
        ece += weight / total_weight * error
        if count >= min_bin_support:
            supported_errors.append(error)
    maximum = max(supported_errors) if supported_errors else math.nan
    return float(ece), float(maximum)


def _base_support_gate(
    frame: pd.DataFrame, mode: str, contract: dict[str, Any]
) -> dict[str, Any]:
    support = contract["support_gates"]
    if mode == "full_2024":
        rule = support["full_2024_fit_eligibility_each_cycle"]
        required_quarters = [f"2024_q{value}" for value in range(1, 5)]
    elif mode == "oof":
        rule = support["july_december_2024_oof_provisional_tier_each_cycle"]
        required_quarters = [f"2024_q{value}" for value in rule["required_quarters"]]
    elif mode == "scoring":
        rule = support["each_full_scoring_period_each_cycle"]
        years = pd.to_datetime(frame["session_date"], errors="raise").dt.year.unique()
        if len(years) != 1:
            raise AssertionError("support frame crosses years")
        required_quarters = [f"{int(years[0])}_q{value}" for value in range(1, 5)]
    else:
        raise ValueError(mode)
    realised = frame.loc[frame["loop_occurs"].eq(1)]
    quarter_counts = realised["quarter"].astype(str).value_counts()
    quarter_minimum_key = (
        "minimum_realized_loop_rows_each_required_quarter"
        if mode == "oof"
        else "minimum_realized_loop_rows_each_quarter"
    )
    checks: dict[str, bool] = {
        "realised_rows": len(realised) >= int(rule["minimum_realized_loop_rows"]),
        "stocks": realised["symbol_norm"].nunique()
        >= int(rule["minimum_stocks_with_realized_loop"]),
        "quarters": bool(
            set(quarter_counts.index) == set(required_quarters)
            and len(quarter_counts) == len(required_quarters)
        ),
        "quarter_rows": all(
            int(quarter_counts.get(value, 0))
            >= int(rule[quarter_minimum_key])
            for value in required_quarters
        ),
    }
    if "minimum_compatible_anchor_cycle_rows" in rule:
        checks["compatible_rows"] = len(frame) >= int(
            rule["minimum_compatible_anchor_cycle_rows"]
        )
    return {
        "compatible_rows": len(frame),
        "realised_rows": len(realised),
        "realised_stocks": int(realised["symbol_norm"].nunique()),
        "required_quarters": required_quarters,
        "minimum_realised_quarter_rows": min(
            (int(quarter_counts.get(value, 0)) for value in required_quarters),
            default=0,
        ),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _structural_gate(
    frame: pd.DataFrame, min_bin_support: int, tolerance: float
) -> dict[str, Any]:
    observed = frame["loop_occurs"].to_numpy(dtype=int)
    history_probability = frame["loop_probability"].to_numpy(dtype=float)
    first_probability = frame["first_order_probability"].to_numpy(dtype=float)
    history = binary_losses(observed, history_probability)
    first = binary_losses(observed, first_probability)
    history_ece, history_maximum = _calibration_summary(
        observed, history_probability, np.ones(len(frame)), min_bin_support
    )
    first_ece, first_maximum = _calibration_summary(
        observed, first_probability, np.ones(len(frame)), min_bin_support
    )
    checks = {
        "log_loss_lower": float(history["log_loss"].mean())
        < float(first["log_loss"].mean()),
        "brier_lower": float(history["brier"].mean()) < float(first["brier"].mean()),
        "ece_no_worse": history_ece <= first_ece,
        "maximum_supported_bin_error": bool(
            np.isfinite(history_maximum)
            and np.isfinite(first_maximum)
            and history_maximum <= first_maximum + tolerance
        ),
    }
    return {
        "history_log_loss": float(history["log_loss"].mean()),
        "first_order_log_loss": float(first["log_loss"].mean()),
        "history_brier": float(history["brier"].mean()),
        "first_order_brier": float(first["brier"].mean()),
        "history_ece": history_ece,
        "first_order_ece": first_ece,
        "history_maximum_supported_bin_error": history_maximum,
        "first_order_maximum_supported_bin_error": first_maximum,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _surface_gate(
    frame: pd.DataFrame,
    observed: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    weights: np.ndarray,
    surface: str,
    required_quarters: list[str],
    min_bin_support: int,
    contract: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    baseline_losses = binary_losses(observed, baseline_probability)
    candidate_losses = binary_losses(observed, candidate_probability)
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
    loss_results: dict[str, Any] = {}
    all_slice_pass = True
    all_interval_pass = True
    brier_difference_pass = True
    relative_log_loss = math.nan
    for loss_index, loss_name in enumerate(("log_loss", "brier")):
        difference = candidate_losses[loss_name] - baseline_losses[loss_name]
        mean_difference = weighted_mean(difference, weights)
        baseline_mean = weighted_mean(baseline_losses[loss_name], weights)
        daily = _weighted_difference_by_group(
            frame, difference, weights, "session_date"
        )
        daily_mean, daily_low, daily_high = moving_block_bounds(
            daily.to_numpy(dtype=float), seed + loss_index
        )
        quarter = _weighted_difference_by_group(frame, difference, weights, "quarter")
        quarter_pass = bool(
            all(
                value in quarter.index and float(quarter.loc[value]) < 0.0
                for value in required_quarters
            )
        )
        deletion_values = {}
        for symbol in sorted(frame["symbol_norm"].astype(str).unique()):
            keep = frame["symbol_norm"].astype(str).ne(symbol).to_numpy()
            deletion_values[symbol] = weighted_mean(difference[keep], weights[keep])
        deletion_pass = bool(
            deletion_values
            and all(np.isfinite(value) and value < 0.0 for value in deletion_values.values())
        )
        interval_pass = bool(np.isfinite(daily_high) and daily_high < 0.0)
        all_interval_pass &= interval_pass
        all_slice_pass &= quarter_pass and deletion_pass
        if loss_name == "brier":
            brier_difference_pass = bool(mean_difference < 0.0)
        else:
            relative_log_loss = -mean_difference / baseline_mean
        loss_results[loss_name] = {
            "candidate": weighted_mean(candidate_losses[loss_name], weights),
            "baseline": baseline_mean,
            "mean_difference": mean_difference,
            "daily_mean_difference": daily_mean,
            "daily_ci_low": daily_low,
            "daily_ci_high": daily_high,
            "interval_pass": interval_pass,
            "quarter_means": quarter.to_dict(),
            "quarter_pass": quarter_pass,
            "leave_one_stock_max_difference": max(deletion_values.values()),
            "stock_deletions_pass": deletion_pass,
        }
    baseline_ece, baseline_maximum = _calibration_summary(
        observed, baseline_probability, weights, min_bin_support
    )
    candidate_ece, candidate_maximum = _calibration_summary(
        observed, candidate_probability, weights, min_bin_support
    )
    calibration_checks = {
        "ece_no_worse": candidate_ece <= baseline_ece,
        "maximum_supported_bin_error": bool(
            np.isfinite(candidate_maximum)
            and np.isfinite(baseline_maximum)
            and candidate_maximum <= baseline_maximum + tolerance
        ),
    }
    checks = {
        "relative_log_loss": relative_log_loss
        >= float(comparison_rule["minimum_relative_log_loss_improvement"]),
        "brier_difference": brier_difference_pass,
        "daily_intervals": all_interval_pass,
        "quarter_and_stock_robustness": all_slice_pass,
        **calibration_checks,
    }
    return {
        "relative_log_loss_improvement": relative_log_loss,
        "losses": loss_results,
        "baseline_ece": baseline_ece,
        "candidate_ece": candidate_ece,
        "baseline_maximum_supported_bin_error": baseline_maximum,
        "candidate_maximum_supported_bin_error": candidate_maximum,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _quality_cell_gate(
    cycle_frame: pd.DataFrame,
    target: str,
    horizon: int,
    tier: str,
    mode: str,
    required_quarters: list[str],
    contract: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    threshold_class = 1 if tier == "p75" else 2
    quality = cycle_frame[f"quality_class__{target}__h{horizon}"].to_numpy(int)
    realised_mask = cycle_frame["loop_occurs"].eq(1).to_numpy()
    conditional_frame = cycle_frame.loc[realised_mask].reset_index(drop=True)
    conditional_observed = (quality[realised_mask] >= threshold_class).astype(int)
    conditional_weights = conditional_frame["conditional_weight"].to_numpy(float)
    key_context = model_key("qcontext", target, horizon)
    key_cycle = model_key("qcycle", target, horizon)
    conditional_context = conditional_frame[f"{key_context}__{tier}"].to_numpy(float)
    conditional_cycle = conditional_frame[f"{key_cycle}__{tier}"].to_numpy(float)
    joint_label = "good" if tier == "p75" else "high"
    joint_observed = cycle_frame[
        f"joint_{joint_label}_target__{target}__h{horizon}"
    ].to_numpy(int)
    joint_context = cycle_frame[f"joint__{key_context}__{tier}"].to_numpy(float)
    joint_cycle = cycle_frame[f"joint__{key_cycle}__{tier}"].to_numpy(float)
    calibration_rule = contract["common_quality_gates"]["calibration"]
    if mode == "oof":
        conditional_bin = int(calibration_rule["oof_minimum_supported_conditional_bin_rows"])
        joint_bin = int(calibration_rule["oof_minimum_supported_joint_bin_rows"])
        support_rule = contract["support_gates"][
            "july_december_2024_oof_provisional_tier_each_cycle"
        ]
    else:
        conditional_bin = int(calibration_rule["scoring_minimum_supported_conditional_bin_rows"])
        joint_bin = int(calibration_rule["scoring_minimum_supported_joint_bin_rows"])
        support_rule = contract["support_gates"]["each_full_scoring_period_each_cycle"]
    positives = int(conditional_observed.sum())
    negatives = int(len(conditional_observed) - positives)
    if tier == "p75":
        minimum_positive = int(
            support_rule["good_minimum_p75_positive_and_negative_rows_each_target_horizon"]
        )
        minimum_negative = minimum_positive
    else:
        minimum_positive = int(
            support_rule["high_minimum_p90_positive_rows_each_target_horizon"]
        )
        minimum_negative = int(
            support_rule["high_minimum_p90_negative_rows_each_target_horizon"]
        )
    support_pass = positives >= minimum_positive and negatives >= minimum_negative
    conditional_gate = _surface_gate(
        conditional_frame,
        conditional_observed,
        conditional_context,
        conditional_cycle,
        conditional_weights,
        "conditional",
        required_quarters,
        conditional_bin,
        contract,
        seed,
    )
    joint_gate = _surface_gate(
        cycle_frame,
        joint_observed,
        joint_context,
        joint_cycle,
        np.ones(len(cycle_frame), dtype=float),
        "joint",
        required_quarters,
        joint_bin,
        contract,
        seed + 100,
    )
    observed_rate = weighted_mean(conditional_observed, conditional_weights)
    mean_context = weighted_mean(conditional_context, conditional_weights)
    mean_cycle = weighted_mean(conditional_cycle, conditional_weights)
    lift_ratio = observed_rate / mean_context if mean_context > 0.0 else math.nan
    residual = conditional_observed - conditional_context
    daily_residual = _weighted_difference_by_group(
        conditional_frame,
        residual,
        conditional_weights,
        "session_date",
    )
    residual_mean, residual_low, residual_high = moving_block_bounds(
        daily_residual.to_numpy(dtype=float), seed + 200
    )
    lift_pass = bool(np.isfinite(residual_low) and residual_low > 0.0)
    tier_rules = contract["tier_rules_each_cycle_and_horizon"]
    if tier == "p75":
        rate_rule = tier_rules["good"]
        rate_checks = {
            "observed_rate": observed_rate
            >= float(rate_rule["minimum_observed_conditional_exceedance_rate"]),
            "mean_qcycle": mean_cycle
            >= float(rate_rule["minimum_mean_calibrated_qcycle_probability"]),
            "observed_over_qcontext": lift_ratio
            >= float(rate_rule["minimum_observed_rate_divided_by_mean_qcontext_probability"]),
        }
    else:
        rate_rule = tier_rules["high"]
        rate_checks = {
            "observed_rate": observed_rate
            >= float(rate_rule["p90_minimum_observed_conditional_exceedance_rate"]),
            "mean_qcycle": mean_cycle
            >= float(rate_rule["p90_minimum_mean_calibrated_qcycle_probability"]),
            "observed_over_qcontext": lift_ratio
            >= float(rate_rule["p90_minimum_observed_rate_divided_by_mean_qcontext_probability"]),
        }
    checks = {
        "support": support_pass,
        "conditional_quality": conditional_gate["pass"],
        "joint_chain": joint_gate["pass"],
        "lift_interval": lift_pass,
        **rate_checks,
    }
    return {
        "target": target,
        "horizon": horizon,
        "tier": tier,
        "conditional_rows": len(conditional_frame),
        "positive_rows": positives,
        "negative_rows": negatives,
        "observed_rate": observed_rate,
        "mean_qcontext_probability": mean_context,
        "mean_qcycle_probability": mean_cycle,
        "observed_rate_divided_by_mean_qcontext": lift_ratio,
        "daily_residual_mean": residual_mean,
        "daily_residual_ci_low": residual_low,
        "daily_residual_ci_high": residual_high,
        "conditional_gate": conditional_gate,
        "joint_gate": joint_gate,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def grade_period(
    scoring: pd.DataFrame,
    period: str,
    mode: str,
    contract: dict[str, Any],
    full_2024_eligibility: dict[str, bool] | None = None,
) -> dict[str, pd.DataFrame]:
    support_rows = []
    structural_rows = []
    cell_rows = []
    horizon_rows = []
    cycle_rows = []
    if mode == "oof":
        required_quarters = ["2024_q3", "2024_q4"]
        structural_bin = int(
            contract["common_quality_gates"]["calibration"][
                "oof_minimum_supported_joint_bin_rows"
            ]
        )
    else:
        years = pd.to_datetime(scoring["session_date"], errors="raise").dt.year.unique()
        if len(years) != 1:
            raise AssertionError("grade frame crosses years")
        required_quarters = [f"{int(years[0])}_q{value}" for value in range(1, 5)]
        structural_bin = int(
            contract["common_quality_gates"]["calibration"][
                "scoring_minimum_supported_joint_bin_rows"
            ]
        )
    tolerance = float(
        contract["structural_reliability_gate_each_cycle_and_scoring_period"][
            "maximum_supported_bin_error_tolerance"
        ]
    )
    for cycle_index, (cycle_id, cycle_frame) in enumerate(
        scoring.groupby("cycle_id", sort=True)
    ):
        cycle_frame = cycle_frame.reset_index(drop=True)
        support = _base_support_gate(cycle_frame, mode, contract)
        fit_eligible = (
            True
            if full_2024_eligibility is None
            else bool(full_2024_eligibility.get(cycle_id, False))
        )
        support_rows.append(
            {
                "period": period,
                "cycle_id": cycle_id,
                **{key: value for key, value in support.items() if key != "checks"},
                "checks": json.dumps(support["checks"], sort_keys=True),
                "full_2024_fit_eligible": fit_eligible,
                "combined_support_pass": bool(support["pass"] and fit_eligible),
            }
        )
        structural = _structural_gate(cycle_frame, structural_bin, tolerance)
        structural_rows.append(
            {
                "period": period,
                "cycle_id": cycle_id,
                **{key: value for key, value in structural.items() if key != "checks"},
                "checks": json.dumps(structural["checks"], sort_keys=True),
            }
        )
        cell_results: dict[tuple[str, int, str], dict[str, Any]] = {}
        for target_index, target in enumerate(TARGETS):
            for horizon in HORIZONS:
                for tier_index, tier in enumerate(TIERS):
                    result = _quality_cell_gate(
                        cycle_frame,
                        target,
                        horizon,
                        tier,
                        mode,
                        required_quarters,
                        contract,
                        SEED
                        + (0 if mode == "oof" else 50000)
                        + cycle_index * 1000
                        + target_index * 200
                        + horizon * 5
                        + tier_index,
                    )
                    cell_results[(target, horizon, tier)] = result
                    cell_rows.append(
                        {
                            "period": period,
                            "cycle_id": cycle_id,
                            "target": target,
                            "horizon": horizon,
                            "tier": tier,
                            "pass": result["pass"],
                            "positive_rows": result["positive_rows"],
                            "negative_rows": result["negative_rows"],
                            "observed_rate": result["observed_rate"],
                            "mean_qcontext_probability": result[
                                "mean_qcontext_probability"
                            ],
                            "mean_qcycle_probability": result[
                                "mean_qcycle_probability"
                            ],
                            "observed_rate_divided_by_mean_qcontext": result[
                                "observed_rate_divided_by_mean_qcontext"
                            ],
                            "daily_residual_ci_low": result["daily_residual_ci_low"],
                            "conditional_relative_log_loss_improvement": result[
                                "conditional_gate"
                            ]["relative_log_loss_improvement"],
                            "joint_relative_log_loss_improvement": result["joint_gate"][
                                "relative_log_loss_improvement"
                            ],
                            "gate_detail": json.dumps(safe(result), sort_keys=True),
                        }
                    )
        horizon_grades = []
        for horizon in HORIZONS:
            good_cells = [
                cell_results[(target, horizon, "p75")]["pass"]
                for target in TARGETS
            ]
            high_p90_cells = [
                cell_results[(target, horizon, "p90")]["pass"]
                for target in TARGETS
            ]
            high_p75_rate = all(
                cell_results[(target, horizon, "p75")]["observed_rate"]
                >= float(
                    contract["tier_rules_each_cycle_and_horizon"]["high"][
                        "minimum_p75_observed_conditional_exceedance_rate"
                    ]
                )
                and cell_results[(target, horizon, "p75")][
                    "mean_qcycle_probability"
                ]
                >= float(
                    contract["tier_rules_each_cycle_and_horizon"]["high"][
                        "minimum_p75_mean_calibrated_qcycle_probability"
                    ]
                )
                for target in TARGETS
            )
            structural_required = True if mode == "oof" else structural["pass"]
            common_required = bool(
                support["pass"] and fit_eligible and structural_required
            )
            good_pass = bool(common_required and all(good_cells))
            high_pass = bool(good_pass and high_p75_rate and all(high_p90_cells))
            grade = (
                "high_movement_quality"
                if high_pass
                else "good_movement_quality"
                if good_pass
                else "unqualified"
            )
            horizon_grades.append(grade)
            horizon_rows.append(
                {
                    "period": period,
                    "cycle_id": cycle_id,
                    "horizon": horizon,
                    "grade": grade,
                    "support_pass": bool(support["pass"] and fit_eligible),
                    "structural_pass": structural["pass"],
                    "structural_required_for_grade": mode != "oof",
                    "both_targets_good_pass": bool(all(good_cells)),
                    "both_targets_high_p75_rate_pass": high_p75_rate,
                    "both_targets_high_p90_pass": bool(all(high_p90_cells)),
                }
            )
        if all(value == "high_movement_quality" for value in horizon_grades):
            global_grade = "high_movement_quality"
        elif all(value != "unqualified" for value in horizon_grades) and any(
            value == "good_movement_quality" for value in horizon_grades
        ):
            global_grade = "good_movement_quality"
        else:
            global_grade = "unqualified"
        cycle_rows.append(
            {
                "period": period,
                "cycle_id": cycle_id,
                "h6_grade": horizon_grades[0],
                "h12_grade": horizon_grades[1],
                "h24_grade": horizon_grades[2],
                "global_grade": global_grade,
                "prospective_validated": False,
                "economic_edge_claim": False,
            }
        )
    return {
        "support": pd.DataFrame(support_rows),
        "structural": pd.DataFrame(structural_rows),
        "cells": pd.DataFrame(cell_rows),
        "horizons": pd.DataFrame(horizon_rows),
        "cycles": pd.DataFrame(cycle_rows),
    }


def derive_final_tiers(
    provisional: pd.DataFrame,
    development: pd.DataFrame,
    backward: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grade_rank = {
        "unqualified": 0,
        "good_movement_quality": 1,
        "high_movement_quality": 2,
    }
    rank_grade = {value: key for key, value in grade_rank.items()}
    sources = {
        "provisional_2024_oof": provisional.set_index("cycle_id"),
        "development_2025": development.set_index("cycle_id"),
        "backward_2023": backward.set_index("cycle_id"),
    }
    expected_cycles = set(sources["provisional_2024_oof"].index)
    if any(set(frame.index) != expected_cycles for frame in sources.values()):
        raise AssertionError("cycle set differs across final-grade sources")
    rows = []
    for cycle_id in sorted(expected_cycles):
        grades = {
            name: str(frame.loc[cycle_id, "global_grade"])
            for name, frame in sources.items()
        }
        if any(value not in grade_rank for value in grades.values()):
            raise AssertionError("unknown cycle grade")
        final_grade = rank_grade[min(grade_rank[value] for value in grades.values())]
        rows.append(
            {
                "cycle_id": cycle_id,
                "provisional_2024_oof_grade": grades["provisional_2024_oof"],
                "development_2025_grade": grades["development_2025"],
                "backward_2023_grade": grades["backward_2023"],
                "final_grade": final_grade,
                "prospective_validated": False,
                "economic_edge_claim": False,
            }
        )
    tiers = pd.DataFrame(rows)
    gates = {
        "qualified_good_or_high_cycles": int(
            tiers["final_grade"]
            .isin(["good_movement_quality", "high_movement_quality"])
            .sum()
        ),
        "high_cycles": int(
            tiers["final_grade"].eq("high_movement_quality").sum()
        ),
        "final_grade_is_minimum_of_2024_oof_2025_2023": True,
        "no_unqualified_cycle_may_surface": True,
        "prospective_validation_pending": True,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    return tiers, gates


def main_score() -> dict[str, Any]:
    contract = validate_contract()
    fit_complete_path = OUT / "fit_complete.json"
    pre_audit_path = OUT / "pre_score_audit.json"
    if not fit_complete_path.is_file():
        raise AssertionError("run --fit-only before sealed scoring")
    if not pre_audit_path.is_file():
        raise AssertionError("independent pre-score audit is required")
    pre_audit = json.loads(pre_audit_path.read_text())
    if pre_audit.get("all_passed") is not True:
        raise AssertionError("independent pre-score audit has not passed")
    fit_complete = json.loads(fit_complete_path.read_text())
    current_sources = {
        name: sha256(path) for name, path in _fit_source_paths().items()
    }
    recorded_sources = json.loads((OUT / "fit_source_hashes.json").read_text())
    if current_sources != recorded_sources:
        raise AssertionError("fit source changed after the 2024 freeze")
    for name, expected in fit_complete["artifact_hashes"].items():
        if sha256(OUT / name) != expected:
            raise AssertionError(f"frozen fit artifact changed: {name}")
    pre_snapshot = json.loads(
        (OUT / "prospective_shadow_pre_content_snapshot.json").read_text()
    )
    if snapshot_protected_tree() != pre_snapshot:
        raise AssertionError("prospective movement shadow changed before scoring")

    # The evaluation panels are opened only after all preceding assertions.
    write_json(
        OUT / "evaluation_source_hashes.json",
        {
            "anchor_panel_2025.parquet": sha256(ANCHOR_2025),
            "anchor_panel_2023.parquet": sha256(ANCHOR_2023),
        },
    )
    thresholds = pd.read_csv(OUT / "quality_thresholds_2024.csv")
    cycles = load_cycles()
    parameters = load_frozen_parameters()
    numeric_medians = {
        name: float(value)
        for name, value in zip(
            NUMERIC_CONTROLS,
            parameters["context_numeric_medians"],
            strict=True,
        )
    }
    all_metrics = []
    all_calibration = []
    all_comparisons = []
    all_per_cycle = []
    period_support = []
    period_structural = []
    period_cells = []
    period_horizons = []
    period_grades: dict[str, pd.DataFrame] = {}
    for period, path, year, seed_offset in (
        ("2025", ANCHOR_2025, 2025, 1000),
        ("2023", ANCHOR_2023, 2023, 2000),
    ):
        anchors = load_anchor_panel(path, year, period)
        expanded = add_quality_classes(
            expand_compatible_cycles(
                anchors, cycles, parameters["frozen_first_order"]
            ),
            thresholds,
        )
        scoring = score_probabilities(expanded, numeric_medians, parameters)
        scoring.to_parquet(OUT / f"quality_scoring_{period}.parquet", index=False)
        evaluation = evaluate_scoring(scoring, period, seed_offset)
        all_metrics.append(evaluation["metrics"])
        all_calibration.append(evaluation["calibration"])
        all_comparisons.append(evaluation["comparisons"])
        all_per_cycle.append(evaluation["per_cycle"])
        grades = grade_period(scoring, period, "scoring", contract)
        period_support.append(grades["support"])
        period_structural.append(grades["structural"])
        period_cells.append(grades["cells"])
        period_horizons.append(grades["horizons"])
        period_grades[period] = grades["cycles"]

    metrics = pd.concat(all_metrics, ignore_index=True)
    calibration = pd.concat(all_calibration, ignore_index=True)
    comparisons = pd.concat(all_comparisons, ignore_index=True)
    per_cycle = pd.concat(all_per_cycle, ignore_index=True)
    support = pd.concat(period_support, ignore_index=True)
    structural = pd.concat(period_structural, ignore_index=True)
    quality_cells = pd.concat(period_cells, ignore_index=True)
    horizon_grades = pd.concat(period_horizons, ignore_index=True)
    cycle_grades = pd.concat(period_grades.values(), ignore_index=True)
    metrics.to_csv(OUT / "quality_metrics.csv", index=False)
    calibration.to_csv(OUT / "quality_calibration.csv", index=False)
    comparisons.to_csv(OUT / "quality_comparisons.csv", index=False)
    per_cycle.to_csv(OUT / "quality_per_cycle_metrics.csv", index=False)
    support.to_csv(OUT / "quality_period_support.csv", index=False)
    structural.to_csv(OUT / "quality_period_structural.csv", index=False)
    quality_cells.to_csv(OUT / "quality_period_cells.csv", index=False)
    horizon_grades.to_csv(OUT / "quality_period_horizon_grades.csv", index=False)
    cycle_grades.to_csv(OUT / "quality_period_cycle_grades.csv", index=False)
    provisional = pd.read_csv(OUT / "provisional_tiers_2024.csv")
    tiers, gates = derive_final_tiers(
        provisional, period_grades["2025"], period_grades["2023"]
    )
    tiers.to_csv(OUT / "final_cycle_tiers.csv", index=False)
    write_json(OUT / "gates.json", gates)
    post_snapshot = snapshot_protected_tree()
    if post_snapshot != pre_snapshot:
        raise AssertionError("prospective movement shadow changed during scoring")
    write_json(OUT / "prospective_shadow_post_content_snapshot.json", post_snapshot)
    summary = {
        "algorithm": "per_loop_conditional_movement_quality",
        "fit_period": 2024,
        "scoring_periods": [2025, 2023],
        "scientific_status": "development_and_backward_portability_not_prospective",
        "qualified_tiers": tiers.to_dict(orient="records"),
        "gates": gates,
        "prospective_shadow_unchanged": True,
        "interpretation": (
            "Movement/range quality only. No direction, signed return, P&L, "
            "economic edge, tradability, order, or deployment claim."
        ),
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    tests = self_tests()
    if args.self_test_only:
        print(json.dumps(tests, indent=2, sort_keys=True))
        return
    result = fit_only() if args.fit_only else main_score()
    print(json.dumps(safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
