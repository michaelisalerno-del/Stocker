"""Independent audit for the research-only per-loop movement-quality layer.

``--pre-score-only`` is deliberately restricted to the frozen 2024 fit and
July--December causal OOF artifacts.  It must pass before the production
runner may open row-level 2025 or backward-2023 movement outcomes.  The
default post-score entry point remains sealed until scoring has completed.

This module does not import the production runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CONTRACT = HERE / "contracts/20260710-per-loop-movement-quality-v1.json"
RUNNER = HERE / "run_per_loop_movement_quality.py"

STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
PATH_ROOT = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710")
PRICE_ROOT = Path("/private/tmp/stocker_frozen_loop_price_consequence_20260710")
ARTIFACT = Path("/private/tmp/stocker_per_loop_movement_quality_20260710")

ANCHOR_2024 = PRICE_ROOT / "anchor_panel_train_2024.parquet"
ANCHOR_2025 = PRICE_ROOT / "anchor_panel_2025.parquet"
ANCHOR_2023 = PRICE_ROOT / "anchor_panel_2023.parquet"
CYCLE_SOURCE = STATE_ROOT / "fixed_cycle_shuffled_nulls.csv"
PATH_PARAMETERS = PATH_ROOT / "model_parameters.npz"
PATH_CYCLES = PATH_ROOT / "fixed_cycles.csv"
FEATURE_MANIFEST = PRICE_ROOT / "feature_manifest.json"
PATH_GATES = PATH_ROOT / "gates.json"
PATH_AUDIT = PATH_ROOT / "independent_artifact_audit.json"
PRICE_GATES = PRICE_ROOT / "gates.json"
PRICE_AUDIT = PRICE_ROOT / "independent_artifact_audit.json"

K = 8
END = 8
TOKEN_WIDTH = 648
CYCLE_COUNT = 20
HORIZONS = (6, 12, 24)
TARGETS = ("absolute_return_bps", "future_range_bps")
MODELS = ("qcontext", "qcycle")
TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
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
EPS = 1e-12
SEED = 20260710

PROTECTED_PATHS = (
    WORKSPACE / "work/contracts/20260710-frozen-loop-movement-shadow-v1.json",
    WORKSPACE / "work/contracts/20260710-frozen-loop-movement-shadow-v1-manifest.json",
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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def content_snapshot() -> dict[str, Any]:
    """Hash protected content only; ignore inode, mode and timestamps."""

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in PROTECTED_PATHS:
        paths = [root]
        if root.is_dir():
            paths.extend(sorted(root.rglob("*")))
        for candidate in paths:
            path = candidate.resolve()
            if path in seen:
                continue
            seen.add(path)
            if candidate.is_symlink():
                kind = "symlink"
                sha = hashlib.sha256(os.readlink(candidate).encode()).hexdigest()
            elif candidate.is_dir():
                kind = "directory"
                sha = None
            else:
                kind = "file"
                sha = digest(candidate)
            rows.append(
                {
                    "path": str(candidate.relative_to(WORKSPACE)),
                    "kind": kind,
                    "mode": stat.S_IMODE(candidate.lstat().st_mode),
                    "size": candidate.lstat().st_size,
                    "sha256": sha,
                }
            )
    rows.sort(key=lambda row: row["path"])
    runtime = json.loads(
        (
            WORKSPACE
            / "work/shadow_validation/frozen_loop_movement_shadow_v1/runtime_metadata.json"
        ).read_text()
    )
    ledger = (
        WORKSPACE
        / "work/shadow_validation/frozen_loop_movement_shadow_v1/prediction_ledger.jsonl"
    )
    return {
        "files": rows,
        "file_count": len(rows),
        "tree_sha256": json_hash(rows),
        "runtime_outcomes_opened": runtime.get("outcomes_opened"),
        "ledger_size": ledger.stat().st_size,
        "ledger_lines": len(ledger.read_text().splitlines()),
        "ledger_sha256": digest(ledger),
    }


def canonical_cycle(values: Iterable[int]) -> tuple[int, ...]:
    core = tuple(int(value) for value in values)
    if not core:
        raise AssertionError("empty cycle")
    return min(core[index:] + core[:index] for index in range(len(core)))


def compatible_routes(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current),)
            for index, state in enumerate(core)
            if int(state) == int(current)
        }
    )


def independent_cycles() -> list[dict[str, Any]]:
    source = pd.read_csv(CYCLE_SOURCE)
    if len(source) != CYCLE_COUNT:
        raise AssertionError("expected twenty frozen cycles")
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for index, value in enumerate(source["cycle"].astype(str), start=1):
        closed = tuple(int(part) for part in value.split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid closed cycle: {value}")
        core = canonical_cycle(closed[:-1])
        if core in seen or len(core) not in (2, 3, 4):
            raise AssertionError(f"duplicate or unsupported cycle: {value}")
        if min(core) < 0 or max(core) >= K:
            raise AssertionError(f"cycle state outside frozen range: {value}")
        if any(left == right for left, right in zip(core, core[1:] + core[:1])):
            raise AssertionError(f"cycle contains self-transition: {value}")
        seen.add(core)
        output.append(
            {
                "cycle_id": f"cycle_{index:02d}",
                "cycle_index": index - 1,
                "cycle": "->".join(str(state) for state in core + (core[0],)),
                "transition_length": len(core),
                "core": core,
            }
        )
    return output


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=float)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def history_tokens(prev2: np.ndarray, prev1: np.ndarray, state: np.ndarray) -> np.ndarray:
    prev2 = np.asarray(prev2, dtype=int)
    prev1 = np.asarray(prev1, dtype=int)
    state = np.asarray(state, dtype=int)
    if (
        prev2.min(initial=0) < 0
        or prev2.max(initial=0) > END
        or prev1.min(initial=0) < 0
        or prev1.max(initial=0) > END
        or state.min(initial=0) < 0
        or state.max(initial=0) >= K
    ):
        raise AssertionError("invalid history state")
    return (prev2 * 9 + prev1) * K + state


def route_probability(
    frame: pd.DataFrame,
    route: tuple[int, ...],
    parameters: dict[str, np.ndarray],
    model: str,
) -> np.ndarray:
    probability = np.ones(len(frame), dtype=float)
    prev2 = frame["previous_state_2"].to_numpy(int)
    prev1 = frame["previous_state_1"].to_numpy(int)
    current = np.full(len(frame), route[0], dtype=int)
    for destination in route[1:]:
        if model == "history":
            token = history_tokens(prev2, prev1, current)
            logits = (
                parameters["history_intercept"][None, :]
                + parameters["history_coef"][:, token].T
            )
            step = softmax(logits)[:, int(destination)]
        elif model == "first_order":
            step = parameters["first_order"][current, int(destination)]
        else:
            raise ValueError(model)
        probability *= step
        prev2, prev1, current = (
            prev1,
            current,
            np.full(len(frame), int(destination), dtype=int),
        )
    return probability


def cycle_probability(
    frame: pd.DataFrame,
    core: tuple[int, ...],
    parameters: dict[str, np.ndarray],
    model: str,
) -> np.ndarray:
    values = np.zeros(len(frame), dtype=float)
    for current in sorted(set(core)):
        mask = frame["state"].eq(current).to_numpy()
        selected = frame.loc[mask].reset_index(drop=True)
        probability = np.zeros(len(selected), dtype=float)
        for route in compatible_routes(core, current):
            probability += route_probability(selected, route, parameters, model)
        values[mask] = probability
    if values.min(initial=0.0) < -1e-12 or values.max(initial=0.0) > 1 + 1e-9:
        raise AssertionError("invalid structural cycle probability")
    return np.clip(values, 0.0, 1.0)


def realized_label(frame: pd.DataFrame, core: tuple[int, ...]) -> np.ndarray:
    label = np.zeros(len(frame), dtype=bool)
    future = frame[
        [f"future_state_{step}" for step in range(1, 5)]
    ].to_numpy(int)
    for current in sorted(set(core)):
        mask = frame["state"].eq(current).to_numpy()
        for route in compatible_routes(core, current):
            required = np.asarray(route[1:], dtype=int)
            label[mask] |= np.all(future[mask, : len(required)] == required, axis=1)
    return label


def load_2024_anchors() -> pd.DataFrame:
    frame = pd.read_parquet(ANCHOR_2024)
    required = {
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "bar_index_in_session",
        "state",
        "previous_state_1",
        "previous_state_2",
        "history_token",
        "quarter",
        *NUMERIC_CONTROLS,
        *(f"future_state_{step}" for step in range(1, 5)),
        *(f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"2024 anchor panel missing columns: {missing}")
    frame = frame.copy().sort_values("anchor_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(frame["anchor_id"].to_numpy(int), np.arange(len(frame))):
        raise AssertionError("2024 anchor ids are not canonical")
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {2024}:
        raise AssertionError("2024 anchor panel crosses year boundary")
    if frame["bar_index_in_session"].astype(int).gt(53).any():
        raise AssertionError("anchor after frozen start-bar cutoff")
    expected_token = history_tokens(
        frame["previous_state_2"].to_numpy(int),
        frame["previous_state_1"].to_numpy(int),
        frame["state"].to_numpy(int),
    )
    if not np.array_equal(expected_token, frame["history_token"].to_numpy(int)):
        raise AssertionError("stored history token mismatch")
    outcome_columns = [
        f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS
    ]
    if not np.isfinite(frame[outcome_columns].to_numpy(float)).all():
        raise AssertionError("non-finite 2024 movement outcome")
    frame["month_key"] = dates.dt.strftime("%Y-%m")
    return frame


def load_evaluation_anchors(path: Path, year: int) -> pd.DataFrame:
    """Load only the frozen causal controls, loop-label trace, and movement targets."""

    columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "quarter",
        "start_timestamp",
        "bar_index_in_session",
        "state",
        "previous_state_1",
        "previous_state_2",
        "history_token",
        *NUMERIC_CONTROLS,
        *(f"future_state_{step}" for step in range(1, 5)),
        *(f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS),
        *(f"exact_{horizon}" for horizon in HORIZONS),
        *(f"loop_score_{index:02d}" for index in range(1, 21)),
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["start_timestamp"] = pd.to_datetime(
        frame["start_timestamp"], utc=True, errors="raise"
    )
    frame = frame.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {year} or year >= 2026:
        raise AssertionError(f"evaluation panel crosses frozen {year} boundary")
    if frame["anchor_id"].duplicated().any():
        raise AssertionError("duplicate evaluation anchor id")
    if frame["bar_index_in_session"].astype(int).gt(53).any():
        raise AssertionError("evaluation anchor after bar 53")
    if not all(frame[f"exact_{horizon}"].astype(bool).all() for horizon in HORIZONS):
        raise AssertionError("evaluation panel has inexact future-bar spacing")
    expected = history_tokens(
        frame["previous_state_2"].to_numpy(int),
        frame["previous_state_1"].to_numpy(int),
        frame["state"].to_numpy(int),
    )
    if not np.array_equal(expected, frame["history_token"].to_numpy(int)):
        raise AssertionError("evaluation history-token mismatch")
    targets = [f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS]
    if not np.isfinite(frame[targets].to_numpy(float)).all():
        raise AssertionError("non-finite evaluation movement target")
    frame["month_key"] = dates.dt.strftime("%Y-%m")
    return frame


def reconstruct_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for horizon in HORIZONS:
            values = frame[f"{target}_{horizon}"].astype(float)
            rows.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "p75_threshold_bps": float(
                        values.quantile(0.75, interpolation="linear")
                    ),
                    "p90_threshold_bps": float(
                        values.quantile(0.90, interpolation="linear")
                    ),
                    "quantile_method": "linear",
                    "training_anchors": len(values),
                }
            )
    return pd.DataFrame(rows)


def movement_class(values: np.ndarray, p75: float, p90: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values > p90, 2, np.where(values > p75, 1, 0)).astype(np.int8)


def oof_splits(frame: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return the six frozen expanding-month masks without touching outcomes."""

    month = frame["month_key"].astype(str).to_numpy()
    output = []
    for validation_month in OOF_MONTHS:
        train = month < validation_month
        validation = month == validation_month
        if not train.any() or not validation.any():
            raise AssertionError(f"empty OOF fold for {validation_month}")
        if not np.all(month[train] < validation_month):
            raise AssertionError("OOF fit includes non-prior month")
        output.append((validation_month, train, validation))
    return output


def reconstruct_long_panel(
    anchors: pd.DataFrame,
    cycles: list[dict[str, Any]],
    parameters: dict[str, np.ndarray],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    base_columns = [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "month_key",
        "quarter",
        "start_timestamp",
        "state",
        "previous_state_1",
        "previous_state_2",
        "history_token",
        *NUMERIC_CONTROLS,
        *(f"future_state_{step}" for step in range(1, 5)),
        *(f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS),
    ]
    for cycle in cycles:
        mask = anchors["state"].isin(set(cycle["core"])).to_numpy()
        selected = anchors.loc[mask, base_columns].copy()
        selected["cycle_id"] = cycle["cycle_id"]
        selected["cycle_index"] = cycle["cycle_index"]
        selected["cycle"] = cycle["cycle"]
        selected["transition_length"] = cycle["transition_length"]
        selected["loop_occurs"] = realized_label(
            anchors.loc[mask].reset_index(drop=True), cycle["core"]
        ).astype(np.int8)
        selected["loop_probability"] = cycle_probability(
            anchors.loc[mask].reset_index(drop=True),
            cycle["core"],
            parameters,
            "history",
        )
        selected["first_order_probability"] = cycle_probability(
            anchors.loc[mask].reset_index(drop=True),
            cycle["core"],
            parameters,
            "first_order",
        )
        parts.append(selected)
    output = pd.concat(parts, ignore_index=True)
    positive_count = (
        output.groupby("anchor_id", sort=False)["loop_occurs"]
        .transform("sum")
        .to_numpy(int)
    )
    positive = output["loop_occurs"].to_numpy(bool)
    weight = np.zeros(len(output), dtype=float)
    weight[positive] = 1.0 / positive_count[positive]
    output["positive_cycle_count"] = positive_count.astype(np.int16)
    output["conditional_weight"] = weight
    if not np.allclose(
        output.loc[positive].groupby("anchor_id")["conditional_weight"].sum(),
        1.0,
        atol=1e-12,
    ):
        raise AssertionError("inverse positive-overlap weights do not sum to one")
    return output.sort_values(["anchor_id", "cycle_index"], kind="stable").reset_index(
        drop=True
    )


def raw_context(frame: pd.DataFrame, medians: pd.Series) -> sparse.csr_matrix:
    numeric = (
        frame.loc[:, list(NUMERIC_CONTROLS)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medians)
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise AssertionError("non-finite causal context")
    state = frame["state"].to_numpy(int)
    one_hot = sparse.csr_matrix(np.eye(K, dtype=float)[state])
    return sparse.hstack(
        (one_hot, sparse.csr_matrix(numeric.to_numpy(float))), format="csr"
    )


def fit_preprocessor(frame: pd.DataFrame) -> tuple[pd.Series, StandardScaler]:
    medians = frame.loc[:, list(NUMERIC_CONTROLS)].apply(
        pd.to_numeric, errors="coerce"
    ).median(axis=0)
    scaler = StandardScaler(with_mean=False).fit(raw_context(frame, medians))
    return medians, scaler


def fit_weighted_preprocessor(
    frame: pd.DataFrame, medians: pd.Series, weights: np.ndarray
) -> StandardScaler:
    scaler = StandardScaler(with_mean=False)
    scaler.fit(raw_context(frame, medians), sample_weight=np.asarray(weights, float))
    return scaler


def cycle_blocks(frame: pd.DataFrame) -> sparse.csr_matrix:
    row = np.arange(len(frame))
    cycle = frame["cycle_index"].to_numpy(int)
    state = frame["state"].to_numpy(int)
    token = frame["history_token"].to_numpy(int)
    main = sparse.csr_matrix(
        (np.ones(len(frame)), (row, cycle)), shape=(len(frame), CYCLE_COUNT)
    )
    by_state = sparse.csr_matrix(
        (np.full(len(frame), 0.5), (row, cycle * K + state)),
        shape=(len(frame), CYCLE_COUNT * K),
    )
    by_history = sparse.csr_matrix(
        (np.full(len(frame), 0.25), (row, cycle * TOKEN_WIDTH + token)),
        shape=(len(frame), CYCLE_COUNT * TOKEN_WIDTH),
    )
    return sparse.hstack((main, by_state, by_history), format="csr")


def feature_matrices(
    train: pd.DataFrame,
    predict: pd.DataFrame,
) -> tuple[dict[str, sparse.csr_matrix], dict[str, sparse.csr_matrix], dict[str, Any]]:
    medians, scaler = fit_preprocessor(train)
    train_context = scaler.transform(raw_context(train, medians)).tocsr()
    predict_context = scaler.transform(raw_context(predict, medians)).tocsr()
    train_x = {
        "qcontext": train_context,
        "qcycle": sparse.hstack((train_context, cycle_blocks(train)), format="csr"),
    }
    predict_x = {
        "qcontext": predict_context,
        "qcycle": sparse.hstack(
            (predict_context, cycle_blocks(predict)), format="csr"
        ),
    }
    metadata = {
        "numeric_medians": medians,
        "context_scaler": scaler,
        "widths": {name: matrix.shape[1] for name, matrix in train_x.items()},
    }
    return train_x, predict_x, metadata


def fit_ordered_model(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    weights: np.ndarray,
) -> LogisticRegression:
    if not np.array_equal(np.unique(labels), np.arange(3)):
        raise AssertionError("ordered training rows do not contain all three classes")
    model = LogisticRegression(
        C=0.2,
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    )
    model.fit(matrix, labels, sample_weight=weights)
    if not np.array_equal(model.classes_, np.arange(3)):
        raise AssertionError("ordered class mapping changed")
    return model


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise AssertionError("temperature input must have three ordered classes")
    return softmax(np.log(np.maximum(probability, EPS)) / float(temperature))


def weighted_multinomial_log_loss(
    labels: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> float:
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(weights, dtype=float)
    losses = -np.log(np.clip(probability[np.arange(len(labels)), labels], EPS, 1.0))
    return float(np.average(losses, weights=weights))


def select_temperature(
    labels: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> tuple[float, pd.DataFrame]:
    rows = []
    for temperature in TEMPERATURES:
        calibrated = apply_temperature(probability, temperature)
        rows.append(
            {
                "temperature": temperature,
                "weighted_log_loss": weighted_multinomial_log_loss(
                    labels, calibrated, weights
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["distance_from_one"] = (table["temperature"] - 1.0).abs()
    selected = table.sort_values(
        ["weighted_log_loss", "distance_from_one", "temperature"], kind="stable"
    ).iloc[0]
    return float(selected["temperature"]), table.drop(columns="distance_from_one")


def ordered_outputs(
    structural_probability: np.ndarray, class_probability: np.ndarray
) -> dict[str, np.ndarray]:
    probability = np.asarray(class_probability, dtype=float)
    structural = np.asarray(structural_probability, dtype=float)
    q90 = probability[:, 2]
    q75 = probability[:, 1] + q90
    output = {
        "class_0": probability[:, 0],
        "class_1": probability[:, 1],
        "class_2": probability[:, 2],
        "q75": q75,
        "q90": q90,
        "j75": structural * q75,
        "j90": structural * q90,
    }
    if (
        not all(np.isfinite(values).all() for values in output.values())
        or np.any(q90 < -1e-12)
        or np.any(q90 > q75 + 1e-12)
        or np.any(q75 > 1 + 1e-12)
    ):
        raise AssertionError("ordered probability nesting failed")
    return output


def model_key(model: str, target: str, horizon: int) -> str:
    return f"{model}__{target}__h{horizon}"


def add_quality_classes(
    expanded: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    output = expanded.copy()
    lookup = {
        (str(row.target), int(row.horizon)): (
            float(row.p75_threshold_bps),
            float(row.p90_threshold_bps),
        )
        for row in thresholds.itertuples(index=False)
    }
    for target in TARGETS:
        for horizon in HORIZONS:
            p75, p90 = lookup[(target, horizon)]
            output[f"quality_class__{target}__h{horizon}"] = movement_class(
                output[f"{target}_{horizon}"].to_numpy(float), p75, p90
            )
    return output


def numeric_medians_from_manifest() -> pd.Series:
    manifest = json.loads(FEATURE_MANIFEST.read_text())
    if manifest.get("numeric_controls") != list(NUMERIC_CONTROLS):
        raise AssertionError("upstream causal-control order changed")
    medians = pd.Series(
        {
            name: float(manifest["numeric_medians"][name])
            for name in NUMERIC_CONTROLS
        }
    )
    return medians.loc[list(NUMERIC_CONTROLS)]


def reconstruct_oof(
    conditional: pd.DataFrame,
    expanded: pd.DataFrame,
    medians: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Independently refit every causal 2024 fold and reconstruct its outputs."""

    conditional_month = pd.to_datetime(
        conditional["session_date"], errors="raise"
    ).dt.strftime("%Y-%m")
    expanded_month = pd.to_datetime(
        expanded["session_date"], errors="raise"
    ).dt.strftime("%Y-%m")
    conditional_raw = raw_context(conditional, medians)
    expanded_raw = raw_context(expanded, medians)
    weights = conditional["conditional_weight"].to_numpy(float)
    accumulator: dict[str, dict[str, list[np.ndarray]]] = {
        model_key(model, target, horizon): {
            "conditional_positions": [],
            "conditional_logits": [],
            "conditional_labels": [],
            "conditional_weights": [],
            "all_positions": [],
            "all_logits": [],
        }
        for model in MODELS
        for target in TARGETS
        for horizon in HORIZONS
    }
    fold_rows: list[dict[str, Any]] = []
    for fold_index, validation_month in enumerate(OOF_MONTHS, start=1):
        train_positions = np.flatnonzero(
            conditional_month.lt(validation_month).to_numpy()
        )
        validation_positions = np.flatnonzero(
            conditional_month.eq(validation_month).to_numpy()
        )
        all_positions = np.flatnonzero(expanded_month.eq(validation_month).to_numpy())
        if not len(train_positions) or not len(validation_positions) or not len(all_positions):
            raise AssertionError(f"empty independent OOF fold {validation_month}")
        train_weight = weights[train_positions]
        scaler = StandardScaler(with_mean=False)
        scaler.fit(
            conditional_raw[train_positions],
            sample_weight=train_weight,
        )
        train_context = scaler.transform(conditional_raw[train_positions]).tocsr()
        validation_context = scaler.transform(
            conditional_raw[validation_positions]
        ).tocsr()
        all_context = scaler.transform(expanded_raw[all_positions]).tocsr()
        matrices = {
            "qcontext": (train_context, validation_context, all_context),
            "qcycle": (
                sparse.hstack(
                    (
                        train_context,
                        cycle_blocks(
                            conditional.iloc[train_positions].reset_index(drop=True)
                        ),
                    ),
                    format="csr",
                ),
                sparse.hstack(
                    (
                        validation_context,
                        cycle_blocks(
                            conditional.iloc[validation_positions].reset_index(
                                drop=True
                            )
                        ),
                    ),
                    format="csr",
                ),
                sparse.hstack(
                    (
                        all_context,
                        cycle_blocks(
                            expanded.iloc[all_positions].reset_index(drop=True)
                        ),
                    ),
                    format="csr",
                ),
            ),
        }
        for target in TARGETS:
            for horizon in HORIZONS:
                labels = conditional[
                    f"quality_class__{target}__h{horizon}"
                ].to_numpy(int)
                for model, (train_x, validation_x, all_x) in matrices.items():
                    fitted = fit_ordered_model(
                        train_x,
                        labels[train_positions],
                        train_weight,
                    )
                    key = model_key(model, target, horizon)
                    item = accumulator[key]
                    item["conditional_positions"].append(validation_positions.copy())
                    item["conditional_logits"].append(
                        fitted.decision_function(validation_x)
                    )
                    item["conditional_labels"].append(labels[validation_positions])
                    item["conditional_weights"].append(
                        weights[validation_positions].copy()
                    )
                    item["all_positions"].append(all_positions.copy())
                    item["all_logits"].append(fitted.decision_function(all_x))
                    fold_rows.append(
                        {
                            "fold": fold_index,
                            "validation_month": validation_month,
                            "training_rows": len(train_positions),
                            "validation_rows": len(validation_positions),
                            "training_weight": float(train_weight.sum()),
                            "validation_weight": float(
                                weights[validation_positions].sum()
                            ),
                            "model": model,
                            "target": target,
                            "horizon": horizon,
                            "n_iter": int(fitted.n_iter_[0]),
                        }
                    )

    base_columns = [
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
        *(f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS),
    ]
    expected_all = np.flatnonzero(expanded_month.isin(OOF_MONTHS).to_numpy())
    expected_conditional = np.flatnonzero(
        conditional_month.isin(OOF_MONTHS).to_numpy()
    )
    oof = expanded.iloc[expected_all][base_columns].copy().reset_index(drop=True)
    temperature_rows: list[dict[str, Any]] = []
    temperatures: dict[str, float] = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            observed = expanded.iloc[expected_all][
                f"quality_class__{target}__h{horizon}"
            ].to_numpy(int)
            loop = oof["loop_occurs"].eq(1).to_numpy()
            oof[f"quality_class__{target}__h{horizon}"] = observed
            oof[f"joint_good_target__{target}__h{horizon}"] = (
                loop & (observed >= 1)
            ).astype(np.int8)
            oof[f"joint_high_target__{target}__h{horizon}"] = (
                loop & (observed == 2)
            ).astype(np.int8)
            for model in MODELS:
                key = model_key(model, target, horizon)
                item = accumulator[key]
                positions = np.concatenate(item["conditional_positions"])
                order = np.argsort(positions, kind="stable")
                positions = positions[order]
                if not np.array_equal(positions, expected_conditional):
                    raise AssertionError(f"conditional OOF alignment drift for {key}")
                logits = np.vstack(item["conditional_logits"])[order]
                labels = np.concatenate(item["conditional_labels"])[order]
                task_weights = np.concatenate(item["conditional_weights"])[order]
                raw_conditional = softmax(logits)
                selected, table = select_temperature(
                    labels, raw_conditional, task_weights
                )
                temperatures[key] = selected
                for row in table.itertuples(index=False):
                    temperature_rows.append(
                        {
                            "model": model,
                            "target": target,
                            "horizon": horizon,
                            "temperature": float(row.temperature),
                            "weighted_oof_log_loss": float(row.weighted_log_loss),
                            "selected": bool(float(row.temperature) == selected),
                        }
                    )
                all_positions = np.concatenate(item["all_positions"])
                all_order = np.argsort(all_positions, kind="stable")
                all_positions = all_positions[all_order]
                if not np.array_equal(all_positions, expected_all):
                    raise AssertionError(f"all-compatible OOF alignment drift for {key}")
                raw_probability = softmax(np.vstack(item["all_logits"])[all_order])
                calibrated = apply_temperature(raw_probability, selected)
                for class_index in range(3):
                    oof[f"{key}__raw_class_{class_index}"] = raw_probability[
                        :, class_index
                    ]
                    oof[f"{key}__calibrated_class_{class_index}"] = calibrated[
                        :, class_index
                    ]
                outputs = ordered_outputs(
                    oof["loop_probability"].to_numpy(float), calibrated
                )
                oof[f"{key}__p75"] = outputs["q75"]
                oof[f"{key}__p90"] = outputs["q90"]
                oof[f"joint__{key}__p75"] = outputs["j75"]
                oof[f"joint__{key}__p90"] = outputs["j90"]
    temperature = pd.DataFrame(temperature_rows).sort_values(
        ["model", "target", "horizon", "temperature"], kind="stable"
    ).reset_index(drop=True)
    folds = pd.DataFrame(fold_rows)
    return oof, temperature, folds, temperatures


def reconstruct_full_parameters(
    conditional: pd.DataFrame,
    medians: pd.Series,
    temperatures: dict[str, float],
    first_order: np.ndarray,
) -> dict[str, np.ndarray]:
    weights = conditional["conditional_weight"].to_numpy(float)
    raw = raw_context(conditional, medians)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(raw, sample_weight=weights)
    context = scaler.transform(raw).tocsr()
    matrices = {
        "qcontext": context,
        "qcycle": sparse.hstack((context, cycle_blocks(conditional)), format="csr"),
    }
    parameters: dict[str, np.ndarray] = {
        "context_scaler_scale": scaler.scale_.copy(),
        "context_scaler_mean": scaler.mean_.copy(),
        "context_scaler_var": scaler.var_.copy(),
        "context_numeric_medians": medians.loc[list(NUMERIC_CONTROLS)].to_numpy(
            float
        ),
        "frozen_first_order": first_order.copy(),
    }
    for model, matrix in matrices.items():
        for target in TARGETS:
            for horizon in HORIZONS:
                key = model_key(model, target, horizon)
                labels = conditional[
                    f"quality_class__{target}__h{horizon}"
                ].to_numpy(int)
                fitted = fit_ordered_model(matrix, labels, weights)
                parameters[f"{key}__classes"] = fitted.classes_.copy()
                parameters[f"{key}__coef"] = fitted.coef_.copy()
                parameters[f"{key}__intercept"] = fitted.intercept_.copy()
                parameters[f"{key}__n_iter"] = fitted.n_iter_.copy()
                parameters[f"{key}__temperature"] = np.asarray(
                    [temperatures[key]], dtype=float
                )
    return parameters


def reconstruct_scoring_probabilities(
    expanded: pd.DataFrame,
    parameters: dict[str, np.ndarray],
) -> pd.DataFrame:
    medians = pd.Series(
        parameters["context_numeric_medians"], index=list(NUMERIC_CONTROLS)
    )
    raw = raw_context(expanded, medians)
    scaler = StandardScaler(with_mean=False)
    scaler.scale_ = parameters["context_scaler_scale"].copy()
    scaler.mean_ = parameters["context_scaler_mean"].copy()
    scaler.var_ = parameters["context_scaler_var"].copy()
    scaler.n_features_in_ = K + len(NUMERIC_CONTROLS)
    context = scaler.transform(raw).tocsr()
    matrices = {
        "qcontext": context,
        "qcycle": sparse.hstack((context, cycle_blocks(expanded)), format="csr"),
    }
    columns = [
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
    ]
    output = expanded.loc[:, columns].copy()
    for target in TARGETS:
        for horizon in HORIZONS:
            quality_column = f"quality_class__{target}__h{horizon}"
            quality = expanded[quality_column].to_numpy(int)
            occurrence = expanded["loop_occurs"].eq(1).to_numpy()
            output[quality_column] = quality
            output[f"joint_good_target__{target}__h{horizon}"] = (
                occurrence & (quality >= 1)
            ).astype(np.int8)
            output[f"joint_high_target__{target}__h{horizon}"] = (
                occurrence & (quality == 2)
            ).astype(np.int8)
            for model, matrix in matrices.items():
                key = model_key(model, target, horizon)
                classes = parameters[f"{key}__classes"].astype(int)
                if not np.array_equal(classes, np.arange(3)):
                    raise AssertionError(f"stored class order changed for {key}")
                logits = np.asarray(matrix @ parameters[f"{key}__coef"].T)
                logits += parameters[f"{key}__intercept"][None, :]
                raw_probability = softmax(logits)
                probability = apply_temperature(
                    raw_probability, float(parameters[f"{key}__temperature"][0])
                )
                ordered = ordered_outputs(
                    expanded["loop_probability"].to_numpy(float), probability
                )
                output[f"{key}__p75"] = ordered["q75"]
                output[f"{key}__p90"] = ordered["q90"]
                output[f"joint__{key}__p75"] = ordered["j75"]
                output[f"joint__{key}__p90"] = ordered["j90"]
    return output


def compare_frame(
    expected: pd.DataFrame,
    path: Path,
    sort_columns: list[str],
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    observed = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    expected = expected.copy().sort_values(sort_columns, kind="stable").reset_index(
        drop=True
    )
    observed = observed.copy().sort_values(sort_columns, kind="stable").reset_index(
        drop=True
    )
    if list(expected.columns) != list(observed.columns):
        raise AssertionError(
            f"column mismatch for {path.name}: expected {list(expected.columns)}, "
            f"observed {list(observed.columns)}"
        )
    maximum_error = 0.0
    for column in expected.columns:
        if pd.api.types.is_datetime64_any_dtype(expected[column]) or column.endswith(
            "timestamp"
        ):
            left = pd.to_datetime(expected[column], utc=True)
            right = pd.to_datetime(observed[column], utc=True)
            if not left.equals(right):
                raise AssertionError(f"datetime mismatch in {path.name}:{column}")
        elif pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(observed[column], errors="coerce").to_numpy(float)
            if not np.array_equal(np.isnan(left), np.isnan(right)):
                raise AssertionError(f"NaN mismatch in {path.name}:{column}")
            finite = np.isfinite(left) & np.isfinite(right)
            error = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
            maximum_error = max(maximum_error, error)
            if error > tolerance:
                raise AssertionError(
                    f"numeric mismatch in {path.name}:{column}: {error}"
                )
        else:
            if not expected[column].astype(str).equals(observed[column].astype(str)):
                raise AssertionError(f"value mismatch in {path.name}:{column}")
    return {"rows": len(expected), "maximum_numeric_error": maximum_error}


def expected_source_hashes() -> dict[str, str]:
    paths = {
        "contract.json": CONTRACT,
        "runner.py": RUNNER,
        "anchor_panel_train_2024.parquet": ANCHOR_2024,
        "fixed_cycles.csv": PATH_CYCLES,
        "feature_manifest.json": FEATURE_MANIFEST,
        "path_model_parameters.npz": PATH_PARAMETERS,
        "path_gates.json": PATH_GATES,
        "path_independent_artifact_audit.json": PATH_AUDIT,
        "price_gates.json": PRICE_GATES,
        "price_independent_artifact_audit.json": PRICE_AUDIT,
    }
    return {name: digest(path) for name, path in paths.items()}


def reconstruct_training_support(expanded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cycle_id, group in expanded.groupby("cycle_id", sort=True):
        occurrence = group["loop_occurs"].eq(1).to_numpy()
        for target in TARGETS:
            for horizon in HORIZONS:
                quality = group[
                    f"quality_class__{target}__h{horizon}"
                ].to_numpy(int)
                rows.append(
                    {
                        "cycle_id": cycle_id,
                        "cycle": str(group["cycle"].iloc[0]),
                        "transition_length": int(group["transition_length"].iloc[0]),
                        "target": target,
                        "horizon": horizon,
                        "compatible_rows": len(group),
                        "realised_occurrences": int(occurrence.sum()),
                        "realised_stocks": int(
                            group.loc[occurrence, "symbol_norm"].nunique()
                        ),
                        "realised_quarters": int(
                            group.loc[occurrence, "quarter"].nunique()
                        ),
                        "good_joint_events": int((occurrence & (quality >= 1)).sum()),
                        "high_joint_events": int((occurrence & (quality == 2)).sum()),
                    }
                )
    return pd.DataFrame(rows)


def binary_loss_arrays(observed: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return {
        "log_loss": -(
            observed * np.log(probability)
            + (1.0 - observed) * np.log(1.0 - probability)
        ),
        "brier": np.square(probability - observed),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.average(values, weights=weights)) if weights.sum() > 0 else math.nan


def grouped_weighted_mean(
    frame: pd.DataFrame, values: np.ndarray, weights: np.ndarray, column: str
) -> pd.Series:
    table = pd.DataFrame(
        {
            "group": frame[column].astype(str).to_numpy(),
            "weighted": np.asarray(values, float) * np.asarray(weights, float),
            "weight": np.asarray(weights, float),
        }
    ).groupby("group", sort=True).sum()
    if (table["weight"] <= 0).any():
        return pd.Series(dtype=float)
    return table["weighted"] / table["weight"]


def moving_block_interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return math.nan, math.nan, math.nan
    length = min(5, len(clean))
    blocks = np.asarray(
        [clean[start : start + length] for start in range(len(clean) - length + 1)]
    )
    required = int(math.ceil(len(clean) / length))
    generator = np.random.default_rng(seed)
    draws = np.empty(5000, dtype=float)
    for draw in range(len(draws)):
        selected = generator.integers(0, len(blocks), size=required)
        draws[draw] = blocks[selected].reshape(-1)[: len(clean)].mean()
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
    observed = np.asarray(observed, float)
    probability = np.asarray(probability, float)
    weights = np.asarray(weights, float)
    bins = np.minimum((probability * 10).astype(int), 9)
    total = float(weights.sum())
    if total <= 0:
        return math.nan, math.nan
    ece = 0.0
    supported: list[float] = []
    for index in range(10):
        mask = bins == index
        weight = float(weights[mask].sum())
        if weight <= 0:
            continue
        error = abs(
            weighted_mean(probability[mask], weights[mask])
            - weighted_mean(observed[mask], weights[mask])
        )
        ece += weight / total * error
        if int(mask.sum()) >= minimum_rows:
            supported.append(error)
    return float(ece), max(supported) if supported else math.nan


def calibration_table_rows(
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
    observed = np.asarray(observed, float)
    probability = np.asarray(probability, float)
    weights = np.asarray(weights, float)
    bin_id = np.minimum((probability * 10).astype(int), 9)
    total = float(weights.sum())
    ece = 0.0
    supported_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for index in range(10):
        mask = bin_id == index
        count = int(mask.sum())
        weight = float(weights[mask].sum())
        mean_probability = weighted_mean(probability[mask], weights[mask])
        event_rate = weighted_mean(observed[mask], weights[mask])
        error = abs(mean_probability - event_rate) if weight > 0 else math.nan
        supported = count >= minimum_rows and weight > 0
        if weight > 0:
            ece += weight / total * error
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
                "bin": index,
                "rows": count,
                "weight": weight,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "absolute_error": error,
                "supported": supported,
            }
        )
    return rows, float(ece), max(supported_errors) if supported_errors else math.nan


def reconstruct_evaluation(
    scoring: pd.DataFrame, period: str, seed_offset: int
) -> dict[str, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    per_cycle_rows: list[dict[str, Any]] = []
    for surface in ("conditional", "joint"):
        if surface == "conditional":
            mask = scoring["loop_occurs"].eq(1).to_numpy()
            weights = scoring.loc[mask, "conditional_weight"].to_numpy(float)
        else:
            mask = np.ones(len(scoring), dtype=bool)
            weights = np.ones(len(scoring), dtype=float)
        frame = scoring.loc[mask].reset_index(drop=True)
        for target in TARGETS:
            for horizon in HORIZONS:
                quality = frame[f"quality_class__{target}__h{horizon}"].to_numpy(int)
                for tier_index, tier in enumerate(("p75", "p90")):
                    if surface == "conditional":
                        observed = quality >= (1 if tier == "p75" else 2)
                    else:
                        label = "good" if tier == "p75" else "high"
                        observed = frame[
                            f"joint_{label}_target__{target}__h{horizon}"
                        ].to_numpy(int)
                    losses: dict[str, dict[str, np.ndarray]] = {}
                    for model in MODELS:
                        key = model_key(model, target, horizon)
                        column = (
                            f"{key}__{tier}"
                            if surface == "conditional"
                            else f"joint__{key}__{tier}"
                        )
                        probability = frame[column].to_numpy(float)
                        model_losses = binary_loss_arrays(observed, probability)
                        losses[model] = model_losses
                        rows, ece, maximum = calibration_table_rows(
                            period,
                            surface,
                            model,
                            target,
                            horizon,
                            tier,
                            observed,
                            probability,
                            weights,
                            100 if surface == "conditional" else 500,
                        )
                        calibration_rows.extend(rows)
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
                                "positives": int(np.asarray(observed).sum()),
                                "weighted_prevalence": weighted_mean(observed, weights),
                                "log_loss": weighted_mean(
                                    model_losses["log_loss"], weights
                                ),
                                "brier": weighted_mean(model_losses["brier"], weights),
                                "ece": ece,
                                "maximum_supported_bin_error": maximum,
                            }
                        )
                    for loss_index, loss_name in enumerate(("log_loss", "brier")):
                        difference = losses["qcycle"][loss_name] - losses["qcontext"][loss_name]
                        daily = pd.DataFrame(
                            {
                                "session_date": frame["session_date"].to_numpy(),
                                "weighted": difference * weights,
                                "weight": weights,
                            }
                        ).groupby("session_date", sort=True).sum()
                        daily_difference = (daily["weighted"] / daily["weight"]).to_numpy(float)
                        mean, low, high = moving_block_interval(
                            daily_difference,
                            SEED
                            + seed_offset
                            + (0 if surface == "conditional" else 10000)
                            + TARGETS.index(target) * 1000
                            + horizon * 10
                            + tier_index * 2
                            + loss_index,
                        )
                        baseline = weighted_mean(losses["qcontext"][loss_name], weights)
                        mean_difference = weighted_mean(difference, weights)
                        comparison_rows.append(
                            {
                                "period": period,
                                "surface": surface,
                                "candidate": "qcycle",
                                "baseline": "qcontext",
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "loss": loss_name,
                                "weighted_mean_difference": mean_difference,
                                "daily_mean_difference": mean,
                                "daily_ci_low": low,
                                "daily_ci_high": high,
                                "baseline_loss": baseline,
                                "relative_improvement": -mean_difference / baseline,
                            }
                        )
                    for cycle_id, positions in frame.groupby("cycle_id", sort=True).groups.items():
                        index = np.asarray(positions, dtype=int)
                        cycle_weights = weights[index]
                        per_cycle_rows.append(
                            {
                                "period": period,
                                "surface": surface,
                                "cycle_id": cycle_id,
                                "target": target,
                                "horizon": horizon,
                                "tier": tier,
                                "rows": len(index),
                                "weight": float(cycle_weights.sum()),
                                "positives": int(np.asarray(observed)[index].sum()),
                                "qcontext_log_loss": weighted_mean(
                                    losses["qcontext"]["log_loss"][index], cycle_weights
                                ),
                                "qcycle_log_loss": weighted_mean(
                                    losses["qcycle"]["log_loss"][index], cycle_weights
                                ),
                                "qcontext_brier": weighted_mean(
                                    losses["qcontext"]["brier"][index], cycle_weights
                                ),
                                "qcycle_brier": weighted_mean(
                                    losses["qcycle"]["brier"][index], cycle_weights
                                ),
                            }
                        )
    return {
        "metrics": pd.DataFrame(metric_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "comparisons": pd.DataFrame(comparison_rows),
        "per_cycle": pd.DataFrame(per_cycle_rows),
    }


def base_support_gate(
    frame: pd.DataFrame, mode: str, contract: dict[str, Any]
) -> dict[str, Any]:
    rules = contract["support_gates"]
    if mode == "full_2024":
        rule = rules["full_2024_fit_eligibility_each_cycle"]
        quarters = [f"2024_q{value}" for value in range(1, 5)]
    elif mode == "oof":
        rule = rules["july_december_2024_oof_provisional_tier_each_cycle"]
        quarters = [f"2024_q{value}" for value in rule["required_quarters"]]
    elif mode == "scoring":
        rule = rules["each_full_scoring_period_each_cycle"]
        years = pd.to_datetime(frame["session_date"], errors="raise").dt.year.unique()
        if len(years) != 1 or int(years[0]) >= 2026:
            raise AssertionError("scoring support crosses frozen year boundary")
        quarters = [f"{int(years[0])}_q{value}" for value in range(1, 5)]
    else:
        raise ValueError(mode)
    realised = frame.loc[frame["loop_occurs"].eq(1)]
    counts = realised["quarter"].astype(str).value_counts()
    quarter_minimum_key = (
        "minimum_realized_loop_rows_each_required_quarter"
        if mode == "oof"
        else "minimum_realized_loop_rows_each_quarter"
    )
    checks = {
        "realised_rows": len(realised) >= int(rule["minimum_realized_loop_rows"]),
        "stocks": realised["symbol_norm"].nunique()
        >= int(rule["minimum_stocks_with_realized_loop"]),
        "quarters": set(counts.index) == set(quarters) and len(counts) == len(quarters),
        "quarter_rows": all(
            int(counts.get(quarter, 0))
            >= int(rule[quarter_minimum_key])
            for quarter in quarters
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
        "required_quarters": quarters,
        "minimum_realised_quarter_rows": min(
            (int(counts.get(quarter, 0)) for quarter in quarters), default=0
        ),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def structural_gate(
    frame: pd.DataFrame, minimum_rows: int, tolerance: float
) -> dict[str, Any]:
    observed = frame["loop_occurs"].to_numpy(int)
    history_probability = frame["loop_probability"].to_numpy(float)
    first_probability = frame["first_order_probability"].to_numpy(float)
    history = binary_loss_arrays(observed, history_probability)
    first = binary_loss_arrays(observed, first_probability)
    history_ece, history_max = calibration_summary(
        observed, history_probability, np.ones(len(frame)), minimum_rows
    )
    first_ece, first_max = calibration_summary(
        observed, first_probability, np.ones(len(frame)), minimum_rows
    )
    checks = {
        "log_loss_lower": float(history["log_loss"].mean())
        < float(first["log_loss"].mean()),
        "brier_lower": float(history["brier"].mean())
        < float(first["brier"].mean()),
        "ece_no_worse": history_ece <= first_ece,
        "maximum_supported_bin_error": bool(
            np.isfinite(history_max)
            and np.isfinite(first_max)
            and history_max <= first_max + tolerance
        ),
    }
    return {
        "history_log_loss": float(history["log_loss"].mean()),
        "first_order_log_loss": float(first["log_loss"].mean()),
        "history_brier": float(history["brier"].mean()),
        "first_order_brier": float(first["brier"].mean()),
        "history_ece": history_ece,
        "first_order_ece": first_ece,
        "history_maximum_supported_bin_error": history_max,
        "first_order_maximum_supported_bin_error": first_max,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def surface_gate(
    frame: pd.DataFrame,
    observed: np.ndarray,
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    weights: np.ndarray,
    surface: str,
    required_quarters: list[str],
    minimum_bin_rows: int,
    contract: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    baseline = binary_loss_arrays(observed, baseline_probability)
    candidate = binary_loss_arrays(observed, candidate_probability)
    common = contract["common_quality_gates"]
    comparison = common[
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
    loss_detail: dict[str, Any] = {}
    slices_pass = True
    intervals_pass = True
    brier_pass = True
    relative = math.nan
    for loss_index, loss_name in enumerate(("log_loss", "brier")):
        difference = candidate[loss_name] - baseline[loss_name]
        mean_difference = weighted_mean(difference, weights)
        baseline_mean = weighted_mean(baseline[loss_name], weights)
        daily = grouped_weighted_mean(frame, difference, weights, "session_date")
        daily_mean, daily_low, daily_high = moving_block_interval(
            daily.to_numpy(float), seed + loss_index
        )
        quarter = grouped_weighted_mean(frame, difference, weights, "quarter")
        quarter_pass = all(
            quarter_name in quarter.index and float(quarter.loc[quarter_name]) < 0
            for quarter_name in required_quarters
        )
        deletion: dict[str, float] = {}
        symbol_values = frame["symbol_norm"].astype(str)
        for symbol in sorted(symbol_values.unique()):
            keep = symbol_values.ne(symbol).to_numpy()
            deletion[symbol] = weighted_mean(difference[keep], weights[keep])
        deletion_pass = bool(
            deletion
            and all(np.isfinite(value) and value < 0 for value in deletion.values())
        )
        interval_pass = bool(np.isfinite(daily_high) and daily_high < 0)
        slices_pass &= quarter_pass and deletion_pass
        intervals_pass &= interval_pass
        if loss_name == "log_loss":
            relative = -mean_difference / baseline_mean
        else:
            brier_pass = mean_difference < 0
        loss_detail[loss_name] = {
            "candidate": weighted_mean(candidate[loss_name], weights),
            "baseline": baseline_mean,
            "mean_difference": mean_difference,
            "daily_mean_difference": daily_mean,
            "daily_ci_low": daily_low,
            "daily_ci_high": daily_high,
            "interval_pass": interval_pass,
            "quarter_means": quarter.to_dict(),
            "quarter_pass": quarter_pass,
            "leave_one_stock_max_difference": max(deletion.values()),
            "stock_deletions_pass": deletion_pass,
        }
    baseline_ece, baseline_max = calibration_summary(
        observed, baseline_probability, weights, minimum_bin_rows
    )
    candidate_ece, candidate_max = calibration_summary(
        observed, candidate_probability, weights, minimum_bin_rows
    )
    checks = {
        "relative_log_loss": relative
        >= float(comparison["minimum_relative_log_loss_improvement"]),
        "brier_difference": bool(brier_pass),
        "daily_intervals": bool(intervals_pass),
        "quarter_and_stock_robustness": bool(slices_pass),
        "ece_no_worse": candidate_ece <= baseline_ece,
        "maximum_supported_bin_error": bool(
            np.isfinite(candidate_max)
            and np.isfinite(baseline_max)
            and candidate_max <= baseline_max + tolerance
        ),
    }
    return {
        "relative_log_loss_improvement": relative,
        "losses": loss_detail,
        "baseline_ece": baseline_ece,
        "candidate_ece": candidate_ece,
        "baseline_maximum_supported_bin_error": baseline_max,
        "candidate_maximum_supported_bin_error": candidate_max,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def quality_cell_gate(
    cycle_frame: pd.DataFrame,
    target: str,
    horizon: int,
    tier: str,
    contract: dict[str, Any],
    seed: int,
    mode: str = "oof",
) -> dict[str, Any]:
    threshold_class = 1 if tier == "p75" else 2
    quality = cycle_frame[f"quality_class__{target}__h{horizon}"].to_numpy(int)
    realised = cycle_frame["loop_occurs"].eq(1).to_numpy()
    conditional = cycle_frame.loc[realised].reset_index(drop=True)
    conditional_observed = (quality[realised] >= threshold_class).astype(int)
    conditional_weights = conditional["conditional_weight"].to_numpy(float)
    context_key = model_key("qcontext", target, horizon)
    cycle_key = model_key("qcycle", target, horizon)
    context_probability = conditional[f"{context_key}__{tier}"].to_numpy(float)
    cycle_probability_values = conditional[f"{cycle_key}__{tier}"].to_numpy(float)
    joint_label = "good" if tier == "p75" else "high"
    joint_observed = cycle_frame[
        f"joint_{joint_label}_target__{target}__h{horizon}"
    ].to_numpy(int)
    calibration = contract["common_quality_gates"]["calibration"]
    if mode == "oof":
        required_quarters = ["2024_q3", "2024_q4"]
        conditional_minimum = int(
            calibration["oof_minimum_supported_conditional_bin_rows"]
        )
        joint_minimum = int(calibration["oof_minimum_supported_joint_bin_rows"])
        support = contract["support_gates"][
            "july_december_2024_oof_provisional_tier_each_cycle"
        ]
    elif mode == "scoring":
        years = pd.to_datetime(cycle_frame["session_date"], errors="raise").dt.year.unique()
        if len(years) != 1 or int(years[0]) >= 2026:
            raise AssertionError("quality cell crosses scoring year")
        required_quarters = [f"{int(years[0])}_q{value}" for value in range(1, 5)]
        conditional_minimum = int(
            calibration["scoring_minimum_supported_conditional_bin_rows"]
        )
        joint_minimum = int(calibration["scoring_minimum_supported_joint_bin_rows"])
        support = contract["support_gates"]["each_full_scoring_period_each_cycle"]
    else:
        raise ValueError(mode)
    conditional_gate = surface_gate(
        conditional,
        conditional_observed,
        context_probability,
        cycle_probability_values,
        conditional_weights,
        "conditional",
        required_quarters,
        conditional_minimum,
        contract,
        seed,
    )
    joint_gate = surface_gate(
        cycle_frame,
        joint_observed,
        cycle_frame[f"joint__{context_key}__{tier}"].to_numpy(float),
        cycle_frame[f"joint__{cycle_key}__{tier}"].to_numpy(float),
        np.ones(len(cycle_frame)),
        "joint",
        required_quarters,
        joint_minimum,
        contract,
        seed + 100,
    )
    positives = int(conditional_observed.sum())
    negatives = int(len(conditional_observed) - positives)
    if tier == "p75":
        minimum_positive = int(
            support[
                "good_minimum_p75_positive_and_negative_rows_each_target_horizon"
            ]
        )
        minimum_negative = minimum_positive
    else:
        minimum_positive = int(support["high_minimum_p90_positive_rows_each_target_horizon"])
        minimum_negative = int(support["high_minimum_p90_negative_rows_each_target_horizon"])
    support_pass = positives >= minimum_positive and negatives >= minimum_negative
    observed_rate = weighted_mean(conditional_observed, conditional_weights)
    mean_context = weighted_mean(context_probability, conditional_weights)
    mean_cycle = weighted_mean(cycle_probability_values, conditional_weights)
    ratio = observed_rate / mean_context if mean_context > 0 else math.nan
    residual = conditional_observed - context_probability
    daily_residual = grouped_weighted_mean(
        conditional, residual, conditional_weights, "session_date"
    )
    residual_mean, residual_low, residual_high = moving_block_interval(
        daily_residual.to_numpy(float), seed + 200
    )
    if tier == "p75":
        rate = contract["tier_rules_each_cycle_and_horizon"]["good"]
        rate_checks = {
            "observed_rate": observed_rate
            >= float(rate["minimum_observed_conditional_exceedance_rate"]),
            "mean_qcycle": mean_cycle
            >= float(rate["minimum_mean_calibrated_qcycle_probability"]),
            "observed_over_qcontext": ratio
            >= float(rate["minimum_observed_rate_divided_by_mean_qcontext_probability"]),
        }
    else:
        rate = contract["tier_rules_each_cycle_and_horizon"]["high"]
        rate_checks = {
            "observed_rate": observed_rate
            >= float(rate["p90_minimum_observed_conditional_exceedance_rate"]),
            "mean_qcycle": mean_cycle
            >= float(rate["p90_minimum_mean_calibrated_qcycle_probability"]),
            "observed_over_qcontext": ratio
            >= float(rate["p90_minimum_observed_rate_divided_by_mean_qcontext_probability"]),
        }
    checks = {
        "support": support_pass,
        "conditional_quality": conditional_gate["pass"],
        "joint_chain": joint_gate["pass"],
        "lift_interval": bool(np.isfinite(residual_low) and residual_low > 0),
        **rate_checks,
    }
    return {
        "target": target,
        "horizon": horizon,
        "tier": tier,
        "conditional_rows": len(conditional),
        "positive_rows": positives,
        "negative_rows": negatives,
        "observed_rate": observed_rate,
        "mean_qcontext_probability": mean_context,
        "mean_qcycle_probability": mean_cycle,
        "observed_rate_divided_by_mean_qcontext": ratio,
        "daily_residual_mean": residual_mean,
        "daily_residual_ci_low": residual_low,
        "daily_residual_ci_high": residual_high,
        "conditional_gate": conditional_gate,
        "joint_gate": joint_gate,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def reconstruct_provisional_grades(
    oof: pd.DataFrame,
    full_expanded: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    full_eligibility = {
        cycle_id: base_support_gate(group.reset_index(drop=True), "full_2024", contract)[
            "pass"
        ]
        for cycle_id, group in full_expanded.groupby("cycle_id", sort=True)
    }
    support_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    structural_minimum = int(
        contract["common_quality_gates"]["calibration"][
            "oof_minimum_supported_joint_bin_rows"
        ]
    )
    structural_tolerance = float(
        contract["structural_reliability_gate_each_cycle_and_scoring_period"][
            "maximum_supported_bin_error_tolerance"
        ]
    )
    for cycle_index, (cycle_id, raw_cycle) in enumerate(
        oof.groupby("cycle_id", sort=True)
    ):
        cycle = raw_cycle.reset_index(drop=True)
        support = base_support_gate(cycle, "oof", contract)
        fit_eligible = bool(full_eligibility[cycle_id])
        support_rows.append(
            {
                "period": "2024_oof",
                "cycle_id": cycle_id,
                **{key: value for key, value in support.items() if key != "checks"},
                "checks": json.dumps(support["checks"], sort_keys=True),
                "full_2024_fit_eligible": fit_eligible,
                "combined_support_pass": bool(support["pass"] and fit_eligible),
            }
        )
        structural = structural_gate(cycle, structural_minimum, structural_tolerance)
        structural_rows.append(
            {
                "period": "2024_oof",
                "cycle_id": cycle_id,
                **{key: value for key, value in structural.items() if key != "checks"},
                "checks": json.dumps(structural["checks"], sort_keys=True),
            }
        )
        cells: dict[tuple[str, int, str], dict[str, Any]] = {}
        for target_index, target in enumerate(TARGETS):
            for horizon in HORIZONS:
                for tier_index, tier in enumerate(("p75", "p90")):
                    result = quality_cell_gate(
                        cycle,
                        target,
                        horizon,
                        tier,
                        contract,
                        SEED
                        + cycle_index * 1000
                        + target_index * 200
                        + horizon * 5
                        + tier_index,
                    )
                    cells[(target, horizon, tier)] = result
                    cell_rows.append(
                        {
                            "period": "2024_oof",
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
        horizon_grades: list[str] = []
        high_rule = contract["tier_rules_each_cycle_and_horizon"]["high"]
        for horizon in HORIZONS:
            good_cells = [cells[(target, horizon, "p75")]["pass"] for target in TARGETS]
            high_cells = [cells[(target, horizon, "p90")]["pass"] for target in TARGETS]
            high_p75 = all(
                cells[(target, horizon, "p75")]["observed_rate"]
                >= float(high_rule["minimum_p75_observed_conditional_exceedance_rate"])
                and cells[(target, horizon, "p75")]["mean_qcycle_probability"]
                >= float(high_rule["minimum_p75_mean_calibrated_qcycle_probability"])
                for target in TARGETS
            )
            good_pass = bool(support["pass"] and fit_eligible and all(good_cells))
            high_pass = bool(good_pass and high_p75 and all(high_cells))
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
                    "period": "2024_oof",
                    "cycle_id": cycle_id,
                    "horizon": horizon,
                    "grade": grade,
                    "support_pass": bool(support["pass"] and fit_eligible),
                    "structural_pass": structural["pass"],
                    "structural_required_for_grade": False,
                    "both_targets_good_pass": bool(all(good_cells)),
                    "both_targets_high_p75_rate_pass": high_p75,
                    "both_targets_high_p90_pass": bool(all(high_cells)),
                }
            )
        if all(grade == "high_movement_quality" for grade in horizon_grades):
            global_grade = "high_movement_quality"
        elif all(grade != "unqualified" for grade in horizon_grades) and any(
            grade == "good_movement_quality" for grade in horizon_grades
        ):
            global_grade = "good_movement_quality"
        else:
            global_grade = "unqualified"
        cycle_rows.append(
            {
                "period": "2024_oof",
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


def reconstruct_scoring_grades(
    scoring: pd.DataFrame, period: str, contract: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    years = pd.to_datetime(scoring["session_date"], errors="raise").dt.year.unique()
    if len(years) != 1 or str(int(years[0])) != period or int(years[0]) >= 2026:
        raise AssertionError("scoring period/year mismatch")
    support_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    structural_minimum = int(
        contract["common_quality_gates"]["calibration"][
            "scoring_minimum_supported_joint_bin_rows"
        ]
    )
    structural_tolerance = float(
        contract["structural_reliability_gate_each_cycle_and_scoring_period"][
            "maximum_supported_bin_error_tolerance"
        ]
    )
    high_rule = contract["tier_rules_each_cycle_and_horizon"]["high"]
    for cycle_index, (cycle_id, raw_cycle) in enumerate(
        scoring.groupby("cycle_id", sort=True)
    ):
        cycle = raw_cycle.reset_index(drop=True)
        support = base_support_gate(cycle, "scoring", contract)
        support_rows.append(
            {
                "period": period,
                "cycle_id": cycle_id,
                **{key: value for key, value in support.items() if key != "checks"},
                "checks": json.dumps(support["checks"], sort_keys=True),
                "full_2024_fit_eligible": True,
                "combined_support_pass": bool(support["pass"]),
            }
        )
        structural = structural_gate(cycle, structural_minimum, structural_tolerance)
        structural_rows.append(
            {
                "period": period,
                "cycle_id": cycle_id,
                **{key: value for key, value in structural.items() if key != "checks"},
                "checks": json.dumps(structural["checks"], sort_keys=True),
            }
        )
        cells: dict[tuple[str, int, str], dict[str, Any]] = {}
        for target_index, target in enumerate(TARGETS):
            for horizon in HORIZONS:
                for tier_index, tier in enumerate(("p75", "p90")):
                    result = quality_cell_gate(
                        cycle,
                        target,
                        horizon,
                        tier,
                        contract,
                        SEED
                        + 50000
                        + cycle_index * 1000
                        + target_index * 200
                        + horizon * 5
                        + tier_index,
                        mode="scoring",
                    )
                    cells[(target, horizon, tier)] = result
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
        horizon_grades: list[str] = []
        for horizon in HORIZONS:
            good_cells = [cells[(target, horizon, "p75")]["pass"] for target in TARGETS]
            high_cells = [cells[(target, horizon, "p90")]["pass"] for target in TARGETS]
            high_p75 = all(
                cells[(target, horizon, "p75")]["observed_rate"]
                >= float(high_rule["minimum_p75_observed_conditional_exceedance_rate"])
                and cells[(target, horizon, "p75")]["mean_qcycle_probability"]
                >= float(high_rule["minimum_p75_mean_calibrated_qcycle_probability"])
                for target in TARGETS
            )
            good_pass = bool(support["pass"] and structural["pass"] and all(good_cells))
            high_pass = bool(good_pass and high_p75 and all(high_cells))
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
                    "support_pass": bool(support["pass"]),
                    "structural_pass": structural["pass"],
                    "structural_required_for_grade": True,
                    "both_targets_good_pass": bool(all(good_cells)),
                    "both_targets_high_p75_rate_pass": high_p75,
                    "both_targets_high_p90_pass": bool(all(high_cells)),
                }
            )
        if all(grade == "high_movement_quality" for grade in horizon_grades):
            global_grade = "high_movement_quality"
        elif all(grade != "unqualified" for grade in horizon_grades) and any(
            grade == "good_movement_quality" for grade in horizon_grades
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


def reconstruct_final_tiers(
    provisional: pd.DataFrame,
    development: pd.DataFrame,
    backward: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rank = {
        "unqualified": 0,
        "good_movement_quality": 1,
        "high_movement_quality": 2,
    }
    inverse = {value: key for key, value in rank.items()}
    sources = {
        "provisional_2024_oof_grade": provisional.set_index("cycle_id"),
        "development_2025_grade": development.set_index("cycle_id"),
        "backward_2023_grade": backward.set_index("cycle_id"),
    }
    cycles = set(next(iter(sources.values())).index)
    if any(set(frame.index) != cycles for frame in sources.values()):
        raise AssertionError("cycle set changed across grade periods")
    rows = []
    for cycle_id in sorted(cycles):
        grades = {
            output_name: str(frame.loc[cycle_id, "global_grade"])
            for output_name, frame in sources.items()
        }
        rows.append(
            {
                "cycle_id": cycle_id,
                **grades,
                "final_grade": inverse[min(rank[value] for value in grades.values())],
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
        "high_cycles": int(tiers["final_grade"].eq("high_movement_quality").sum()),
        "final_grade_is_minimum_of_2024_oof_2025_2023": True,
        "no_unqualified_cycle_may_surface": True,
        "prospective_validation_pending": True,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    return tiers, gates


def _compare_support_core(expected: pd.DataFrame, observed: pd.DataFrame) -> None:
    keys = ["cycle_id", "target", "horizon"]
    core = [
        "cycle",
        "transition_length",
        "compatible_rows",
        "realised_occurrences",
        "realised_stocks",
        "realised_quarters",
        "good_joint_events",
        "high_joint_events",
    ]
    missing = sorted(set(keys + core).difference(observed.columns))
    if missing:
        raise AssertionError(f"training support lacks independently required columns: {missing}")
    left = expected[keys + core].sort_values(keys, kind="stable").reset_index(drop=True)
    right = observed[keys + core].sort_values(keys, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)


def _validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("contract_id") != "per_loop_movement_quality_v1":
        raise AssertionError("unexpected contract id")
    if not (
        contract.get("research_only") is True
        and contract.get("live_ordering_enabled") is False
        and contract.get("order_placement") == "disabled"
        and contract.get("economic_edge_claim") is False
    ):
        raise AssertionError("contract safety labels changed")
    if contract["periods"]["fit_and_internal_forward_validation"] != 2024:
        raise AssertionError("fit period changed")
    if contract["periods"]["internal_forward_validation_months"] != list(
        OOF_MONTHS
    ):
        raise AssertionError("OOF schedule changed")
    if contract["periods"]["2026_permitted"] is not False:
        raise AssertionError("contract permits 2026")
    if contract["cohort"]["horizons_bars"] != list(HORIZONS):
        raise AssertionError("horizons changed")
    if contract["outcomes"]["comparison_operator"] != ">":
        raise AssertionError("movement event is no longer strict greater-than")
    if [float(value) for value in contract["internal_2024_oof_and_calibration"]["temperature_grid"]] != list(TEMPERATURES):
        raise AssertionError("temperature grid changed")
    expected_blocks = [
        ("cycle_one_hot", 20, 1.0),
        ("cycle_by_current_state", 160, 0.5),
        ("cycle_by_history_token", 12960, 0.25),
    ]
    blocks = [
        (str(row["name"]), int(row["width"]), float(row["feature_scale"]))
        for row in contract["models"]["qcycle"]["cycle_feature_blocks"]
    ]
    if blocks != expected_blocks:
        raise AssertionError("hierarchical block specification changed")


def _artifact_hashes_exact(fit_complete: dict[str, Any]) -> dict[str, str]:
    current = {
        name: digest(ARTIFACT / name)
        for name in fit_complete.get("artifact_hashes", {})
    }
    if current != fit_complete.get("artifact_hashes"):
        raise AssertionError("one or more frozen fit artifacts changed")
    return current


def pre_score_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, details: Any = None) -> None:
        checks.append({"name": name, "pass": bool(condition), "details": safe(details)})
        if not condition:
            raise AssertionError(f"pre-score audit failed: {name}: {details}")

    contract = json.loads(CONTRACT.read_text())
    _validate_contract(contract)
    check("contract_and_safety_exact", True)
    check("contract_no_2026", contract["periods"]["2026_permitted"] is False)

    fit_complete = json.loads((ARTIFACT / "fit_complete.json").read_text())
    check(
        "fit_complete_safety",
        fit_complete.get("scoring_outcomes_opened") is False
        and fit_complete.get("scoring_authorized") is False
        and fit_complete.get("research_only") is True
        and fit_complete.get("live_ordering_enabled") is False
        and fit_complete.get("order_placement") == "disabled",
    )
    artifact_hashes = _artifact_hashes_exact(fit_complete)
    check("fit_artifact_hashes_frozen", True, artifact_hashes)
    recorded_sources = json.loads((ARTIFACT / "fit_source_hashes.json").read_text())
    current_sources = expected_source_hashes()
    check("fit_source_hashes_exact", recorded_sources == current_sources, current_sources)
    check(
        "contract_and_runner_hashes_in_fit_complete",
        fit_complete.get("contract_sha256") == digest(CONTRACT)
        and fit_complete.get("runner_sha256") == digest(RUNNER),
    )
    scoring_artifacts = sorted(
        str(path.name)
        for pattern in (
            "quality_scoring_*.parquet",
            "quality_metrics.csv",
            "quality_comparisons.csv",
            "final_cycle_tiers.csv",
            "gates.json",
            "summary.json",
            "evaluation_source_hashes.json",
        )
        for path in ARTIFACT.glob(pattern)
    )
    check("no_scoring_artifacts_or_outcomes", not scoring_artifacts, scoring_artifacts)

    expected_snapshot = json.loads(
        (ARTIFACT / "prospective_shadow_pre_content_snapshot.json").read_text()
    )
    current_snapshot = content_snapshot()
    check(
        "aggregate_movement_shadow_content_unchanged",
        expected_snapshot == current_snapshot,
        {
            "stored_tree": expected_snapshot.get("tree_sha256"),
            "current_tree": current_snapshot.get("tree_sha256"),
        },
    )
    check(
        "aggregate_shadow_outcomes_closed_and_ledger_empty",
        current_snapshot["runtime_outcomes_opened"] is False
        and current_snapshot["ledger_lines"] == 0
        and current_snapshot["ledger_size"] == 0,
    )

    path_gates = json.loads(PATH_GATES.read_text())
    path_audit = json.loads(PATH_AUDIT.read_text())
    price_gates = json.loads(PRICE_GATES.read_text())
    price_audit = json.loads(PRICE_AUDIT.read_text())
    check(
        "frozen_lineage_retained_and_audited",
        path_gates.get("history_retained") is True
        and path_audit.get("all_passed") is True
        and price_gates.get("movement_consequence_retained") is True
        and price_audit.get("all_passed") is True,
    )

    anchors = load_2024_anchors()
    check(
        "anchor_panel_2024_only_and_exact_cohort",
        len(anchors) == 70374
        and anchors["bar_index_in_session"].astype(int).le(53).all(),
        len(anchors),
    )
    thresholds = reconstruct_thresholds(anchors)
    saved_thresholds = pd.read_csv(ARTIFACT / "quality_thresholds_2024.csv")
    threshold_comparison = compare_frame(
        thresholds,
        ARTIFACT / "quality_thresholds_2024.csv",
        ["target", "horizon"],
        tolerance=1e-12,
    )
    contract_errors = []
    for row in thresholds.itertuples(index=False):
        expected = contract["outcomes"]["thresholds_bps"][str(row.target)][
            str(int(row.horizon))
        ]
        contract_errors.extend(
            [
                abs(float(row.p75_threshold_bps) - float(expected["p75"])),
                abs(float(row.p90_threshold_bps) - float(expected["p90"])),
            ]
        )
    check(
        "thresholds_reconstructed_and_contract_exact",
        max(contract_errors) < 1e-12
        and threshold_comparison["maximum_numeric_error"] < 1e-12,
        threshold_comparison,
    )

    cycles = independent_cycles()
    expected_cycle_frame = pd.DataFrame(cycles).drop(columns="core")[
        ["cycle_index", "cycle_id", "cycle", "transition_length"]
    ]
    cycle_comparison = compare_frame(
        expected_cycle_frame,
        ARTIFACT / "fixed_cycles.csv",
        ["cycle_index"],
        tolerance=0.0,
    )
    check("twenty_cycles_reconstructed_exact", len(cycles) == 20, cycle_comparison)
    path_cycle = pd.read_csv(PATH_CYCLES)
    check(
        "cycle_source_lineage_exact",
        path_cycle["cycle_id"].tolist() == expected_cycle_frame["cycle_id"].tolist()
        and path_cycle["cycle"].tolist() == expected_cycle_frame["cycle"].tolist(),
    )

    path_parameters = dict(np.load(PATH_PARAMETERS))
    expanded = add_quality_classes(
        reconstruct_long_panel(anchors, cycles, path_parameters), thresholds
    )
    stored_loop_columns = [f"loop_score_{index:02d}" for index in range(1, 21)]
    reconstructed_loop_error = 0.0
    for cycle in cycles:
        selected = expanded["cycle_id"].eq(cycle["cycle_id"])
        expected = expanded.loc[selected, "loop_probability"].to_numpy(float)
        observed = anchors.loc[
            anchors["state"].isin(set(cycle["core"])),
            stored_loop_columns[cycle["cycle_index"]],
        ].to_numpy(float)
        reconstructed_loop_error = max(
            reconstructed_loop_error, float(np.max(np.abs(expected - observed)))
        )
    check(
        "frozen_structural_probabilities_reconstructed",
        reconstructed_loop_error < 1e-11,
        reconstructed_loop_error,
    )
    conditional = expanded.loc[expanded["loop_occurs"].eq(1)].reset_index(drop=True)
    stored_training_columns = pd.read_parquet(
        ARTIFACT / "training_long_2024.parquet"
    ).columns.tolist()
    missing_training_columns = sorted(set(stored_training_columns).difference(conditional.columns))
    check("training_long_schema_reconstructable", not missing_training_columns, missing_training_columns)
    training_comparison = compare_frame(
        conditional[stored_training_columns],
        ARTIFACT / "training_long_2024.parquet",
        ["anchor_id", "cycle_index"],
        tolerance=1e-10,
    )
    check(
        "labels_overlap_weights_and_training_rows_exact",
        training_comparison["maximum_numeric_error"] < 1e-10,
        training_comparison,
    )
    check(
        "positive_overlap_weights_exact",
        np.allclose(
            conditional.groupby("anchor_id", sort=False)["conditional_weight"].sum(),
            1.0,
            atol=1e-12,
        ),
    )

    support = reconstruct_training_support(expanded)
    stored_support = pd.read_csv(ARTIFACT / "training_support_2024.csv")
    _compare_support_core(support, stored_support)
    check("training_support_core_exact", True, {"rows": len(support)})

    medians = numeric_medians_from_manifest()
    oof, temperatures, folds, selected = reconstruct_oof(
        conditional, expanded, medians
    )
    oof_comparison = compare_frame(
        oof,
        ARTIFACT / "oof_predictions_2024.parquet",
        ["anchor_id", "cycle_index"],
        tolerance=2e-9,
    )
    temperature_comparison = compare_frame(
        temperatures,
        ARTIFACT / "temperature_selection_2024.csv",
        ["model", "target", "horizon", "temperature"],
        tolerance=2e-10,
    )
    fold_comparison = compare_frame(
        folds,
        ARTIFACT / "oof_fold_audit_2024.csv",
        ["fold", "model", "target", "horizon"],
        tolerance=2e-10,
    )
    check("oof_fold_causality_exact", all(name in OOF_MONTHS for name in folds["validation_month"]), fold_comparison)
    check("oof_predictions_exact", oof_comparison["maximum_numeric_error"] < 2e-9, oof_comparison)
    check("temperature_selection_exact", temperature_comparison["maximum_numeric_error"] < 2e-10, temperature_comparison)

    reconstructed_parameters = reconstruct_full_parameters(
        conditional,
        medians,
        selected,
        path_parameters["first_order"],
    )
    saved_parameters = dict(np.load(ARTIFACT / "quality_model_parameters.npz"))
    check("full_parameter_keys_exact", set(reconstructed_parameters) == set(saved_parameters))
    parameter_errors: dict[str, float] = {}
    for name, expected in reconstructed_parameters.items():
        observed = saved_parameters[name]
        if expected.shape != observed.shape:
            raise AssertionError(f"parameter shape mismatch: {name}")
        parameter_errors[name] = float(np.max(np.abs(expected - observed))) if expected.size else 0.0
    check(
        "full_scaler_and_model_parameters_exact",
        max(parameter_errors.values(), default=0.0) < 2e-9,
        {"maximum_error": max(parameter_errors.values(), default=0.0)},
    )

    probability_columns = [
        column
        for column in oof.columns
        if "__p75" in column
        or "__p90" in column
        or "__raw_class_" in column
        or "__calibrated_class_" in column
    ]
    probability = oof[probability_columns].to_numpy(float)
    nested = True
    chain_error = 0.0
    for model in MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                key = model_key(model, target, horizon)
                q75 = oof[f"{key}__p75"].to_numpy(float)
                q90 = oof[f"{key}__p90"].to_numpy(float)
                nested &= bool(np.all((0 <= q90) & (q90 <= q75) & (q75 <= 1)))
                structural = oof["loop_probability"].to_numpy(float)
                chain_error = max(
                    chain_error,
                    float(
                        np.max(
                            np.abs(oof[f"joint__{key}__p75"].to_numpy(float) - structural * q75)
                        )
                    ),
                    float(
                        np.max(
                            np.abs(oof[f"joint__{key}__p90"].to_numpy(float) - structural * q90)
                        )
                    ),
                )
    check("all_probabilities_finite_and_unit_interval", np.isfinite(probability).all() and probability.min() >= -1e-12 and probability.max() <= 1 + 1e-12)
    check("ordered_nesting_and_chain_rule_exact", nested and chain_error < 1e-12, chain_error)

    reconstructed_grades = reconstruct_provisional_grades(oof, expanded, contract)
    provisional_specs = (
        ("support", "provisional_support_2024.csv", ["cycle_id"]),
        ("structural", "provisional_structural_2024.csv", ["cycle_id"]),
        (
            "cells",
            "provisional_quality_cells_2024.csv",
            ["cycle_id", "target", "horizon", "tier"],
        ),
        (
            "horizons",
            "provisional_horizon_grades_2024.csv",
            ["cycle_id", "horizon"],
        ),
        ("cycles", "provisional_tiers_2024.csv", ["cycle_id"]),
    )
    provisional_comparisons: dict[str, Any] = {}
    for name, filename, sort_columns in provisional_specs:
        details = compare_frame(
            reconstructed_grades[name],
            ARTIFACT / filename,
            sort_columns,
            tolerance=2e-9,
        )
        provisional_comparisons[name] = details
        check(
            f"provisional_{name}_exact",
            details["maximum_numeric_error"] < 2e-9,
            details,
        )
    provisional = reconstructed_grades["cycles"]
    expected_gates = {
        "scientific_status": "2024_internal_forward_provisional_only",
        "high_movement_quality_cycles": int(
            provisional["global_grade"].eq("high_movement_quality").sum()
        ),
        "good_movement_quality_cycles": int(
            provisional["global_grade"].eq("good_movement_quality").sum()
        ),
        "unqualified_cycles": int(
            provisional["global_grade"].eq("unqualified").sum()
        ),
        "promotion_permitted": False,
        "prospective_validated": False,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    saved_provisional_gates = json.loads(
        (ARTIFACT / "provisional_gates_2024.json").read_text()
    )
    check(
        "provisional_gate_counts_and_safety_exact",
        expected_gates == saved_provisional_gates,
        expected_gates,
    )
    check(
        "no_provisional_grade_claims_prospective_validation",
        not provisional["prospective_validated"].astype(bool).any()
        and not provisional["economic_edge_claim"].astype(bool).any(),
    )

    runner_text = RUNNER.read_text().lower()
    forbidden_execution = (
        "place_order(",
        "submit_order(",
        "broker_client",
        "position_size(",
        "paper_trade(",
    )
    check(
        "runner_has_no_execution_surface",
        not any(fragment in runner_text for fragment in forbidden_execution),
    )
    check(
        "runner_does_not_import_aggregate_shadow",
        "import frozen_loop_movement_shadow" not in runner_text
        and "from frozen_loop_movement_shadow" not in runner_text,
    )

    payload = {
        "all_passed": all(item["pass"] for item in checks),
        "check_count": len(checks),
        "checks": checks,
        "threshold_reconstruction": threshold_comparison,
        "training_reconstruction": training_comparison,
        "oof_reconstruction": oof_comparison,
        "temperature_reconstruction": temperature_comparison,
        "fold_reconstruction": fold_comparison,
        "maximum_structural_probability_error": reconstructed_loop_error,
        "maximum_parameter_error": max(parameter_errors.values(), default=0.0),
        "maximum_chain_rule_error": chain_error,
        "provisional_reconstruction": provisional_comparisons,
        "scoring_outcomes_opened": False,
        "no_2026_rows": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(ARTIFACT / "pre_score_audit.json", payload)
    return payload


def post_score_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, details: Any = None) -> None:
        checks.append({"name": name, "pass": bool(condition), "details": safe(details)})
        if not condition:
            raise AssertionError(f"post-score audit failed: {name}: {details}")

    required_outputs = (
        "evaluation_source_hashes.json",
        "quality_scoring_2025.parquet",
        "quality_scoring_2023.parquet",
        "quality_metrics.csv",
        "quality_calibration.csv",
        "quality_comparisons.csv",
        "quality_per_cycle_metrics.csv",
        "quality_period_support.csv",
        "quality_period_structural.csv",
        "quality_period_cells.csv",
        "quality_period_horizon_grades.csv",
        "quality_period_cycle_grades.csv",
        "final_cycle_tiers.csv",
        "gates.json",
        "prospective_shadow_post_content_snapshot.json",
        "summary.json",
    )
    missing = [name for name in required_outputs if not (ARTIFACT / name).is_file()]
    check("all_sealed_scoring_outputs_present", not missing, missing)

    contract = json.loads(CONTRACT.read_text())
    _validate_contract(contract)
    check("contract_safety_and_periods_exact", True)
    pre_audit = json.loads((ARTIFACT / "pre_score_audit.json").read_text())
    check(
        "independent_pre_score_audit_passed_before_scoring",
        pre_audit.get("all_passed") is True
        and pre_audit.get("scoring_outcomes_opened") is False
        and int(pre_audit.get("check_count", 0)) >= 35,
    )
    fit_complete = json.loads((ARTIFACT / "fit_complete.json").read_text())
    fit_hashes = _artifact_hashes_exact(fit_complete)
    check("frozen_fit_artifacts_still_exact", True, {"files": len(fit_hashes)})
    current_fit_sources = expected_source_hashes()
    recorded_fit_sources = json.loads((ARTIFACT / "fit_source_hashes.json").read_text())
    check("fit_sources_still_exact", current_fit_sources == recorded_fit_sources)

    evaluation_sources = {
        "anchor_panel_2025.parquet": digest(ANCHOR_2025),
        "anchor_panel_2023.parquet": digest(ANCHOR_2023),
    }
    recorded_evaluation_sources = json.loads(
        (ARTIFACT / "evaluation_source_hashes.json").read_text()
    )
    check(
        "evaluation_source_hashes_exact",
        evaluation_sources == recorded_evaluation_sources,
        evaluation_sources,
    )

    pre_snapshot = json.loads(
        (ARTIFACT / "prospective_shadow_pre_content_snapshot.json").read_text()
    )
    post_snapshot = json.loads(
        (ARTIFACT / "prospective_shadow_post_content_snapshot.json").read_text()
    )
    current_snapshot = content_snapshot()
    check("aggregate_shadow_pre_post_exact", pre_snapshot == post_snapshot)
    check("aggregate_shadow_current_exact", pre_snapshot == current_snapshot)
    check(
        "aggregate_shadow_ledger_empty_and_outcomes_closed",
        current_snapshot["runtime_outcomes_opened"] is False
        and current_snapshot["ledger_size"] == 0
        and current_snapshot["ledger_lines"] == 0,
    )

    parameters = dict(np.load(ARTIFACT / "quality_model_parameters.npz"))
    path_parameters = dict(np.load(PATH_PARAMETERS))
    check(
        "frozen_first_order_parameter_exact",
        np.array_equal(parameters["frozen_first_order"], path_parameters["first_order"]),
    )
    for model in MODELS:
        for target in TARGETS:
            for horizon in HORIZONS:
                key = model_key(model, target, horizon)
                check(
                    f"parameter_class_temperature_{key}",
                    np.array_equal(parameters[f"{key}__classes"], np.arange(3))
                    and float(parameters[f"{key}__temperature"][0])
                    in TEMPERATURES
                    and np.isfinite(parameters[f"{key}__coef"]).all()
                    and np.isfinite(parameters[f"{key}__intercept"]).all(),
                )
    cycles = independent_cycles()
    thresholds = pd.read_csv(ARTIFACT / "quality_thresholds_2024.csv")

    period_reconstruction: dict[str, Any] = {}
    reconstructed_evaluations: dict[str, dict[str, pd.DataFrame]] = {}
    reconstructed_grades: dict[str, dict[str, pd.DataFrame]] = {}
    forbidden_fragments = (
        "direction",
        "signed_return",
        "pnl",
        "cost",
        "spread",
        "slippage",
        "position",
        "broker",
        "strategy",
        "deployment",
        "volume",
    )
    for period, path, year, seed_offset in (
        ("2025", ANCHOR_2025, 2025, 1000),
        ("2023", ANCHOR_2023, 2023, 2000),
    ):
        anchors = load_evaluation_anchors(path, year)
        expanded = add_quality_classes(
            reconstruct_long_panel(anchors, cycles, path_parameters), thresholds
        )
        scoring = reconstruct_scoring_probabilities(expanded, parameters)
        saved_path = ARTIFACT / f"quality_scoring_{period}.parquet"
        saved_columns = pd.read_parquet(saved_path).columns.tolist()
        unexpected = [
            column
            for column in saved_columns
            if any(fragment in column.lower() for fragment in forbidden_fragments)
        ]
        check(f"{period}_scoring_has_no_forbidden_columns", not unexpected, unexpected)
        scoring_comparison = compare_frame(
            scoring[saved_columns],
            saved_path,
            ["anchor_id", "cycle_index"],
            tolerance=2e-10,
        )
        check(
            f"{period}_labels_features_and_probabilities_exact",
            scoring_comparison["maximum_numeric_error"] < 2e-10,
            scoring_comparison,
        )
        loop_error = 0.0
        for cycle in cycles:
            mask = expanded["cycle_id"].eq(cycle["cycle_id"])
            reconstructed = expanded.loc[mask, "loop_probability"].to_numpy(float)
            stored_source = anchors.loc[
                anchors["state"].isin(set(cycle["core"])),
                f"loop_score_{cycle['cycle_index'] + 1:02d}",
            ].to_numpy(float)
            loop_error = max(
                loop_error, float(np.max(np.abs(reconstructed - stored_source)))
            )
        check(
            f"{period}_structural_probability_reconstructed",
            loop_error < 1e-11,
            loop_error,
        )
        probability_columns = [
            column
            for column in scoring.columns
            if column.endswith("__p75") or column.endswith("__p90")
        ]
        values = scoring[probability_columns].to_numpy(float)
        nesting = True
        chain_error = 0.0
        structural = scoring["loop_probability"].to_numpy(float)
        for model in MODELS:
            for target in TARGETS:
                for horizon in HORIZONS:
                    key = model_key(model, target, horizon)
                    q75 = scoring[f"{key}__p75"].to_numpy(float)
                    q90 = scoring[f"{key}__p90"].to_numpy(float)
                    nesting &= bool(np.all((0 <= q90) & (q90 <= q75) & (q75 <= 1)))
                    chain_error = max(
                        chain_error,
                        float(
                            np.max(
                                np.abs(
                                    scoring[f"joint__{key}__p75"].to_numpy(float)
                                    - structural * q75
                                )
                            )
                        ),
                        float(
                            np.max(
                                np.abs(
                                    scoring[f"joint__{key}__p90"].to_numpy(float)
                                    - structural * q90
                                )
                            )
                        ),
                    )
        check(
            f"{period}_probability_finiteness_nesting_chain_rule",
            np.isfinite(values).all()
            and values.min() >= -1e-12
            and values.max() <= 1 + 1e-12
            and nesting
            and chain_error < 1e-12,
            chain_error,
        )
        reconstructed_evaluations[period] = reconstruct_evaluation(
            scoring, period, seed_offset
        )
        reconstructed_grades[period] = reconstruct_scoring_grades(
            scoring, period, contract
        )
        period_reconstruction[period] = {
            "anchors": len(anchors),
            "compatible_rows": len(scoring),
            "realised_rows": int(scoring["loop_occurs"].sum()),
            "effective_realised_anchor_weight": float(
                scoring["conditional_weight"].sum()
            ),
            "maximum_scoring_error": scoring_comparison[
                "maximum_numeric_error"
            ],
            "maximum_structural_error": loop_error,
            "maximum_chain_rule_error": chain_error,
        }
        check(
            f"{period}_year_and_cohort_exact",
            set(pd.to_datetime(scoring["session_date"]).dt.year) == {year}
            and scoring["cycle_id"].nunique() == CYCLE_COUNT,
            period_reconstruction[period],
        )

    evaluation_specs = (
        (
            "metrics",
            "quality_metrics.csv",
            ["period", "surface", "model", "target", "horizon", "tier"],
        ),
        (
            "calibration",
            "quality_calibration.csv",
            ["period", "surface", "model", "target", "horizon", "tier", "bin"],
        ),
        (
            "comparisons",
            "quality_comparisons.csv",
            ["period", "surface", "target", "horizon", "tier", "loss"],
        ),
        (
            "per_cycle",
            "quality_per_cycle_metrics.csv",
            ["period", "surface", "cycle_id", "target", "horizon", "tier"],
        ),
    )
    evaluation_checks: dict[str, Any] = {}
    for key, filename, sort_columns in evaluation_specs:
        expected = pd.concat(
            [reconstructed_evaluations[period][key] for period in ("2025", "2023")],
            ignore_index=True,
        )
        details = compare_frame(
            expected, ARTIFACT / filename, sort_columns, tolerance=2e-9
        )
        evaluation_checks[key] = details
        check(
            f"aggregate_{key}_exact",
            details["maximum_numeric_error"] < 2e-9,
            details,
        )

    grade_specs = (
        ("support", "quality_period_support.csv", ["period", "cycle_id"]),
        ("structural", "quality_period_structural.csv", ["period", "cycle_id"]),
        (
            "cells",
            "quality_period_cells.csv",
            ["period", "cycle_id", "target", "horizon", "tier"],
        ),
        (
            "horizons",
            "quality_period_horizon_grades.csv",
            ["period", "cycle_id", "horizon"],
        ),
        ("cycles", "quality_period_cycle_grades.csv", ["period", "cycle_id"]),
    )
    grade_checks: dict[str, Any] = {}
    for key, filename, sort_columns in grade_specs:
        expected = pd.concat(
            [reconstructed_grades[period][key] for period in ("2025", "2023")],
            ignore_index=True,
        )
        details = compare_frame(
            expected, ARTIFACT / filename, sort_columns, tolerance=2e-9
        )
        grade_checks[key] = details
        check(
            f"period_{key}_and_gates_exact",
            details["maximum_numeric_error"] < 2e-9,
            details,
        )

    provisional = pd.read_csv(ARTIFACT / "provisional_tiers_2024.csv")
    final_tiers, final_gates = reconstruct_final_tiers(
        provisional,
        reconstructed_grades["2025"]["cycles"],
        reconstructed_grades["2023"]["cycles"],
    )
    final_comparison = compare_frame(
        final_tiers,
        ARTIFACT / "final_cycle_tiers.csv",
        ["cycle_id"],
        tolerance=0.0,
    )
    check("final_minimum_tier_logic_exact", final_comparison["maximum_numeric_error"] == 0.0, final_comparison)
    saved_gates = json.loads((ARTIFACT / "gates.json").read_text())
    check("final_gate_decisions_exact", saved_gates == final_gates, final_gates)
    check(
        "no_cycle_qualified_as_good_or_high",
        final_gates["qualified_good_or_high_cycles"] == 0
        and final_gates["high_cycles"] == 0
        and final_tiers["final_grade"].eq("unqualified").all(),
    )

    summary = json.loads((ARTIFACT / "summary.json").read_text())
    check(
        "summary_safety_and_gate_copy_exact",
        summary.get("gates") == final_gates
        and summary.get("qualified_tiers") == safe(final_tiers.to_dict(orient="records"))
        and summary.get("research_only") is True
        and summary.get("live_ordering_enabled") is False
        and summary.get("order_placement") == "disabled"
        and summary.get("prospective_shadow_unchanged") is True,
    )
    check(
        "summary_period_interpretation_exact",
        summary.get("fit_period") == 2024
        and summary.get("scoring_periods") == [2025, 2023]
        and summary.get("scientific_status")
        == "development_and_backward_portability_not_prospective",
    )

    scoring_hashes = {name: digest(ARTIFACT / name) for name in required_outputs}
    check("scoring_artifact_hash_manifest_complete", len(scoring_hashes) == len(required_outputs))
    runner_text = RUNNER.read_text().lower()
    check(
        "no_execution_surface",
        not any(
            fragment in runner_text
            for fragment in (
                "place_order(",
                "submit_order(",
                "broker_client",
                "position_size(",
                "paper_trade(",
            )
        ),
    )

    payload = {
        "all_passed": all(item["pass"] for item in checks),
        "check_count": len(checks),
        "checks": checks,
        "fit_source_hashes": current_fit_sources,
        "evaluation_source_hashes": evaluation_sources,
        "scoring_artifact_hashes": scoring_hashes,
        "period_reconstruction": period_reconstruction,
        "aggregate_evaluation_reconstruction": evaluation_checks,
        "period_grade_reconstruction": grade_checks,
        "final_tier_reconstruction": final_comparison,
        "final_decision": {
            "qualified_good_or_high_cycles": final_gates[
                "qualified_good_or_high_cycles"
            ],
            "high_cycles": final_gates["high_cycles"],
            "all_twenty_unqualified": bool(
                final_tiers["final_grade"].eq("unqualified").all()
            ),
        },
        "prospective_shadow": {
            "tree_sha256": current_snapshot["tree_sha256"],
            "outcomes_opened": current_snapshot["runtime_outcomes_opened"],
            "ledger_lines": current_snapshot["ledger_lines"],
        },
        "no_2026_rows": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(ARTIFACT / "independent_artifact_audit.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-score-only", action="store_true")
    args = parser.parse_args()
    result = pre_score_audit() if args.pre_score_only else post_score_audit()
    print(json.dumps(safe(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
