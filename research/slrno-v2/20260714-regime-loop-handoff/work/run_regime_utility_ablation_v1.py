#!/usr/bin/env python3
"""Causal 2024-only ablation of regime, history, departure, loop, and burst utility.

Research only. This runner has no direction, P&L, order, broker, or deployment path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


WORK = Path(__file__).resolve().parent
CONTRACT_PATH = WORK / "contracts/20260711-regime-utility-ablation-v1.json"
PRE_SCORE_PATH = WORK / "contracts/20260711-regime-utility-ablation-v1-pre-score.json"
ANCHOR_PATH = Path(
    "/private/tmp/stocker_frozen_loop_price_consequence_20260710/"
    "anchor_panel_train_2024.parquet"
)
RUN_PATH = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/"
    "train_2024_filtered_runs.csv"
)
CYCLE_PATH = Path(
    "/private/tmp/stocker_per_loop_movement_quality_20260710/fixed_cycles.csv"
)
OUT = Path("/private/tmp/stocker_regime_utility_ablation_v1_20260711")

SEED = 20260711
K = 8
END_STATE = 8
TOKEN_COUNT = (K + 1) * (K + 1) * K
HORIZONS = (6, 12, 24)
TARGETS = ("absolute_return_bps", "future_range_bps")
MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
LAYERS = ("context", "state", "history", "departure", "loops", "burst")
LAYER_WIDTHS = {
    "context": 9,
    "state": 17,
    "history": 665,
    "departure": 666,
    "loops": 686,
    "burst": 738,
}
INCREMENTAL_PAIRS = (
    ("state_vs_context", "context", "state"),
    ("history_vs_state", "state", "history"),
    ("departure_vs_history", "history", "departure"),
    ("loops_vs_departure", "departure", "loops"),
    ("burst_vs_loops", "loops", "burst"),
)
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
DEPARTURE_ANCHOR_CONTROLS = (
    "b0_entry_numeric",
    "b0_entry_high_stress",
    "entry_time_sin",
    "entry_time_cos",
)
DEPARTURE_RUN_CONTROLS = (
    "b0_state_numeric",
    "b0_high_stress",
    "time_sin",
    "time_cos",
)
ACTUAL_COLUMNS = tuple(
    f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS
)
EPSILON = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def history_tokens(
    previous_state_2: np.ndarray,
    previous_state_1: np.ndarray,
    current_state: np.ndarray,
) -> np.ndarray:
    previous_state_2 = np.asarray(previous_state_2, dtype=int)
    previous_state_1 = np.asarray(previous_state_1, dtype=int)
    current_state = np.asarray(current_state, dtype=int)
    if (
        previous_state_2.min(initial=0) < 0
        or previous_state_2.max(initial=0) > END_STATE
        or previous_state_1.min(initial=0) < 0
        or previous_state_1.max(initial=0) > END_STATE
        or current_state.min(initial=0) < 0
        or current_state.max(initial=0) >= K
    ):
        raise AssertionError("invalid state in history token")
    return ((previous_state_2 * (K + 1) + previous_state_1) * K + current_state)


def token_matrix(tokens: np.ndarray) -> sparse.csr_matrix:
    tokens = np.asarray(tokens, dtype=int)
    if tokens.min(initial=0) < 0 or tokens.max(initial=0) >= TOKEN_COUNT:
        raise AssertionError("history token outside frozen range")
    return sparse.csr_matrix(
        (
            np.ones(len(tokens), dtype=np.float32),
            (np.arange(len(tokens)), tokens),
        ),
        shape=(len(tokens), TOKEN_COUNT),
    )


def state_matrix(states: np.ndarray) -> sparse.csr_matrix:
    states = np.asarray(states, dtype=int)
    if states.min(initial=0) < 0 or states.max(initial=0) >= K:
        raise AssertionError("state outside frozen range")
    return sparse.csr_matrix(
        (
            np.ones(len(states), dtype=np.float32),
            (np.arange(len(states)), states),
        ),
        shape=(len(states), K),
    )


def canonical_cycle(core: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in core)
    return min(values[index:] + values[:index] for index in range(len(values)))


def oriented_paths(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current),)
            for index, state in enumerate(core)
            if int(state) == int(current)
        }
    )


def load_contract_and_pre_score() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(CONTRACT_PATH.read_text())
    pre_score = json.loads(PRE_SCORE_PATH.read_text())
    if contract["research_only"] is not True:
        raise AssertionError("research-only boundary drift")
    if contract["live_ordering_enabled"] is not False:
        raise AssertionError("live-ordering boundary drift")
    if contract["order_placement"] != "disabled":
        raise AssertionError("order-placement boundary drift")
    if contract["periods"]["validation_months"] != list(MONTHS):
        raise AssertionError("validation schedule drift")
    expected_widths = {
        row["name"]: int(row["width"])
        for row in contract["nested_feature_layers"]
    }
    if expected_widths != LAYER_WIDTHS:
        raise AssertionError("feature-width contract drift")
    paths = {
        "contract": CONTRACT_PATH,
        "runner": Path(__file__).resolve(),
        "anchor_panel": ANCHOR_PATH,
        "causal_runs": RUN_PATH,
        "fixed_cycles": CYCLE_PATH,
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    if actual != pre_score["sha256"]:
        raise AssertionError(f"pre-score hash mismatch: {actual} != {pre_score['sha256']}")
    return contract, pre_score


def load_cycles() -> pd.DataFrame:
    raw = pd.read_csv(CYCLE_PATH)
    if len(raw) != 20 or not np.array_equal(raw["cycle_index"], np.arange(20)):
        raise AssertionError("fixed twenty-cycle lineage changed")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for row in raw.itertuples(index=False):
        closed = tuple(int(part) for part in str(row.cycle).split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid cycle {row.cycle}")
        core = canonical_cycle(closed[:-1])
        if core in seen or len(core) not in (2, 3, 4):
            raise AssertionError(f"duplicate or unsupported cycle {row.cycle}")
        seen.add(core)
        rows.append(
            {
                "cycle_index": int(row.cycle_index),
                "cycle_id": str(row.cycle_id),
                "cycle": "->".join(str(value) for value in core + (core[0],)),
                "transition_length": len(core),
                "core": core,
            }
        )
    cycles = pd.DataFrame(rows)
    if int(cycles["transition_length"].eq(2).sum()) != 13:
        raise AssertionError("expected thirteen two-state cycles")
    return cycles


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor_columns = {
        "anchor_id",
        "symbol_norm",
        "session_date",
        "month",
        "state",
        "duration",
        "start_timestamp",
        "previous_state_1",
        "previous_state_2",
        "history_token",
        "bar_index_in_session",
        *NUMERIC_CONTROLS,
        *ACTUAL_COLUMNS,
        *(f"exact_{horizon}" for horizon in HORIZONS),
    }
    anchors = pd.read_parquet(ANCHOR_PATH, columns=sorted(anchor_columns))
    anchors["start_timestamp"] = pd.to_datetime(
        anchors["start_timestamp"], utc=True, errors="raise"
    )
    anchors["session_date"] = anchors["session_date"].astype(str)
    anchors["symbol_norm"] = anchors["symbol_norm"].astype(str)
    anchors["month_key"] = anchors["session_date"].str.slice(0, 7)
    anchors = anchors.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if anchors["anchor_id"].duplicated().any():
        raise AssertionError("duplicate price anchor")
    if set(pd.to_datetime(anchors["session_date"]).dt.year.unique()) != {2024}:
        raise AssertionError("anchor year boundary failure")
    if anchors["bar_index_in_session"].astype(int).gt(53).any():
        raise AssertionError("anchor after frozen bar-53 cutoff")
    if not all(anchors[f"exact_{horizon}"].astype(bool).all() for horizon in HORIZONS):
        raise AssertionError("inexact future price support")
    expected_tokens = history_tokens(
        anchors["previous_state_2"].to_numpy(int),
        anchors["previous_state_1"].to_numpy(int),
        anchors["state"].to_numpy(int),
    )
    if not np.array_equal(expected_tokens, anchors["history_token"].to_numpy(int)):
        raise AssertionError("anchor history token mismatch")
    outcomes = anchors.loc[:, ACTUAL_COLUMNS].to_numpy(float)
    if not np.isfinite(outcomes).all() or (outcomes < 0.0).any():
        raise AssertionError("invalid movement outcome")

    run_columns = [
        "run_id",
        "symbol_norm",
        "session_date",
        "month",
        "state",
        "duration",
        "start_timestamp",
        "previous_state_1",
        "previous_state_2",
        "b0_state_numeric",
        "b0_high_stress",
        "time_sin",
        "time_cos",
        "next_state",
        "has_next_state",
    ]
    runs = pd.read_csv(RUN_PATH, usecols=run_columns)
    runs["start_timestamp"] = pd.to_datetime(
        runs["start_timestamp"], utc=True, errors="raise"
    )
    runs["session_date"] = runs["session_date"].astype(str)
    runs["symbol_norm"] = runs["symbol_norm"].astype(str)
    runs = runs.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)
    if set(pd.to_datetime(runs["session_date"]).dt.year.unique()) != {2024}:
        raise AssertionError("run year boundary failure")
    runs["history_token"] = history_tokens(
        runs["previous_state_2"].to_numpy(int),
        runs["previous_state_1"].to_numpy(int),
        runs["state"].to_numpy(int),
    )
    runs["next_outcome"] = np.where(
        runs["has_next_state"].astype(bool),
        runs["next_state"].fillna(END_STATE).to_numpy(int),
        END_STATE,
    ).astype(int)
    if runs["next_outcome"].min() < 0 or runs["next_outcome"].max() > END_STATE:
        raise AssertionError("invalid destination outcome")
    runs["session_run_index"] = runs.groupby(
        ["symbol_norm", "session_date"], sort=False
    ).cumcount()
    cycles = load_cycles()
    return anchors, runs, cycles


def build_causal_phase_block(
    anchors: pd.DataFrame, runs: pd.DataFrame, cycles: pd.DataFrame
) -> tuple[np.ndarray, list[str], np.ndarray]:
    keys = ["symbol_norm", "session_date", "start_timestamp"]
    lookup = runs[
        keys + ["state", "session_run_index"]
    ].rename(columns={"state": "run_state"})
    positioned = (
        anchors.reset_index(names="anchor_position")
        .merge(lookup, on=keys, how="left", validate="one_to_one")
        .sort_values("anchor_position", kind="stable")
    )
    if positioned["session_run_index"].isna().any():
        raise AssertionError("anchor-to-run phase join failed")
    if not np.array_equal(
        positioned["state"].to_numpy(int), positioned["run_state"].to_numpy(int)
    ):
        raise AssertionError("anchor-to-run state mismatch")

    sequences = {
        key: (
            group["state"].to_numpy(int),
            group["duration"].to_numpy(int),
        )
        for key, group in runs.groupby(["symbol_norm", "session_date"], sort=False)
    }
    two_state = cycles.loc[cycles["transition_length"].eq(2)].reset_index(drop=True)
    two_cycle_indices = two_state["cycle_index"].to_numpy(int)
    repeat = np.zeros((len(anchors), len(two_state)), dtype=np.float32)
    prior_pair = np.zeros_like(repeat)
    durable = np.zeros_like(repeat)
    pair_by_state: dict[int, list[tuple[int, int, int]]] = {state: [] for state in range(K)}
    for local_index, row in enumerate(two_state.itertuples(index=False)):
        left, right = (int(value) for value in row.core)
        pair_by_state[left].append((local_index, left, right))
        pair_by_state[right].append((local_index, right, left))

    for row in positioned[
        ["anchor_position", "symbol_norm", "session_date", "session_run_index", "state"]
    ].itertuples(index=False):
        states, durations = sequences[(row.symbol_norm, row.session_date)]
        index = int(row.session_run_index)
        current = int(row.state)
        for local_index, _, other in pair_by_state[current]:
            count = 0
            cursor = index
            while (
                cursor >= 2
                and int(states[cursor - 1]) == other
                and int(states[cursor - 2]) == current
            ):
                count += 1
                cursor -= 2
            if count:
                previous_current = int(durations[index - 2])
                previous_other = int(durations[index - 1])
                repeat[int(row.anchor_position), local_index] = np.log1p(count)
                prior_pair[int(row.anchor_position), local_index] = np.log1p(
                    previous_current + previous_other
                )
                durable[int(row.anchor_position), local_index] = float(
                    previous_current >= 2 and previous_other >= 2
                )
    names = (
        [f"burst_log_repeat__{value}" for value in two_state["cycle_id"]]
        + [f"burst_log_prior_pair__{value}" for value in two_state["cycle_id"]]
        + [f"burst_prior_durable__{value}" for value in two_state["cycle_id"]]
    )
    static = np.hstack((repeat, prior_pair, durable)).astype(np.float32)
    if static.shape != (len(anchors), 39) or not np.isfinite(static).all():
        raise AssertionError("invalid causal phase block")
    return static, names, two_cycle_indices


def fit_destination_model(train_runs: pd.DataFrame) -> LogisticRegression:
    model = LogisticRegression(
        C=0.2,
        solver="lbfgs",
        max_iter=500,
        random_state=SEED,
    )
    model.fit(
        token_matrix(train_runs["history_token"].to_numpy(int)),
        train_runs["next_outcome"].to_numpy(int),
    )
    if not np.array_equal(model.classes_, np.arange(K + 1)):
        raise AssertionError("destination model missing a class")
    return model


def desired_destination_probability(
    model: LogisticRegression, tokens: np.ndarray, destination: int
) -> np.ndarray:
    probabilities = model.predict_proba(token_matrix(tokens))
    column = int(np.flatnonzero(model.classes_ == int(destination))[0])
    return np.clip(probabilities[:, column], EPSILON, 1.0 - EPSILON)


def predict_loop_scores(
    model: LogisticRegression, anchors: pd.DataFrame, cycles: pd.DataFrame
) -> np.ndarray:
    count = len(anchors)
    output = np.zeros((count, len(cycles)), dtype=np.float64)
    state_all = anchors["state"].to_numpy(int)
    previous_2_all = anchors["previous_state_2"].to_numpy(int)
    previous_1_all = anchors["previous_state_1"].to_numpy(int)
    for cycle in cycles.itertuples(index=False):
        core = tuple(int(value) for value in cycle.core)
        for current in sorted(set(core)):
            positions = np.flatnonzero(state_all == current)
            if not len(positions):
                continue
            total = np.zeros(len(positions), dtype=np.float64)
            for path in oriented_paths(core, current):
                probability = np.ones(len(positions), dtype=np.float64)
                previous_2 = previous_2_all[positions].copy()
                previous_1 = previous_1_all[positions].copy()
                current_state = np.full(len(positions), current, dtype=int)
                for destination in path[1:]:
                    tokens = history_tokens(previous_2, previous_1, current_state)
                    probability *= desired_destination_probability(
                        model, tokens, int(destination)
                    )
                    previous_2, previous_1, current_state = (
                        previous_1,
                        current_state,
                        np.full(len(positions), int(destination), dtype=int),
                    )
                total += probability
            output[positions, int(cycle.cycle_index)] = np.clip(
                total, EPSILON, 1.0 - EPSILON
            )
    if not np.isfinite(output).all() or output.min() < 0.0 or output.max() > 1.0:
        raise AssertionError("invalid fold-local loop probability")
    return output


def departure_raw_matrix(
    frame: pd.DataFrame,
    numeric_columns: tuple[str, ...],
    medians: pd.Series,
) -> sparse.csr_matrix:
    numeric = (
        frame.loc[:, numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .set_axis(DEPARTURE_ANCHOR_CONTROLS, axis=1)
        .fillna(medians)
        .to_numpy(np.float32)
    )
    return sparse.hstack(
        (
            state_matrix(frame["state"].to_numpy(int)),
            token_matrix(frame["history_token"].to_numpy(int)),
            sparse.csr_matrix(numeric),
        ),
        format="csr",
    )


def fit_and_predict_departure(
    train_runs: pd.DataFrame, frames: list[pd.DataFrame]
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    run_numeric = train_runs.loc[:, DEPARTURE_RUN_CONTROLS].apply(
        pd.to_numeric, errors="coerce"
    )
    run_numeric.columns = DEPARTURE_ANCHOR_CONTROLS
    medians = run_numeric.median(axis=0)
    raw_train = departure_raw_matrix(train_runs, DEPARTURE_RUN_CONTROLS, medians)
    scaler = StandardScaler(with_mean=False)
    train_x = scaler.fit_transform(raw_train).tocsr()
    model = LogisticRegression(
        C=0.2,
        solver="lbfgs",
        max_iter=500,
        random_state=SEED,
    )
    target = train_runs["duration"].to_numpy(int) <= 3
    model.fit(train_x, target.astype(np.int8))
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    predictions: list[np.ndarray] = []
    for frame in frames:
        raw = departure_raw_matrix(frame, DEPARTURE_ANCHOR_CONTROLS, medians)
        values = model.predict_proba(scaler.transform(raw))[:, positive_column]
        predictions.append(np.clip(values, EPSILON, 1.0 - EPSILON))
    parameters = {
        "departure_medians": medians.to_numpy(float),
        "departure_scaler_scale": scaler.scale_.copy(),
        "departure_scaler_mean": scaler.mean_.copy(),
        "departure_scaler_var": scaler.var_.copy(),
        "departure_classes": model.classes_.copy(),
        "departure_coef": model.coef_.copy(),
        "departure_intercept": model.intercept_.copy(),
        "departure_n_iter": model.n_iter_.copy(),
    }
    return predictions, parameters


def numeric_block(
    frame: pd.DataFrame, medians: pd.Series
) -> np.ndarray:
    numeric = (
        frame.loc[:, NUMERIC_CONTROLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(medians)
        .to_numpy(np.float32)
    )
    if not np.isfinite(numeric).all():
        raise AssertionError("non-finite price context")
    return numeric


def build_max_design(
    frame: pd.DataFrame,
    medians: pd.Series,
    departure: np.ndarray,
    loops: np.ndarray,
    static_phase: np.ndarray,
    two_cycle_indices: np.ndarray,
) -> sparse.csr_matrix:
    numeric = sparse.csr_matrix(numeric_block(frame, medians))
    state = state_matrix(frame["state"].to_numpy(int))
    history = token_matrix(frame["history_token"].to_numpy(int))
    departure_column = sparse.csr_matrix(np.asarray(departure, dtype=np.float32)[:, None])
    loop_block = sparse.csr_matrix(np.asarray(loops, dtype=np.float32))
    repeat = static_phase[:, : len(two_cycle_indices)]
    interaction = loops[:, two_cycle_indices].astype(np.float32) * repeat
    burst = sparse.csr_matrix(np.hstack((static_phase, interaction)).astype(np.float32))
    matrix = sparse.hstack(
        (numeric, state, history, departure_column, loop_block, burst), format="csr"
    )
    if matrix.shape[1] != LAYER_WIDTHS["burst"]:
        raise AssertionError(f"maximum design width drift: {matrix.shape}")
    if not np.isfinite(matrix.data).all():
        raise AssertionError("non-finite maximum design")
    return matrix


def prediction_column(layer: str, target: str, horizon: int) -> str:
    return f"prediction__{layer}__{target}__h{horizon}"


def actual_column(target: str, horizon: int) -> str:
    return f"actual__{target}__h{horizon}"


def score_folds(
    anchors: pd.DataFrame,
    runs: pd.DataFrame,
    cycles: pd.DataFrame,
    static_phase: np.ndarray,
    static_phase_names: list[str],
    two_cycle_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], sparse.csr_matrix]:
    prediction_frames: list[pd.DataFrame] = []
    structural_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    validation_designs: list[sparse.csr_matrix] = []
    parameters: dict[str, np.ndarray] = {}
    outcome_names = list(ACTUAL_COLUMNS)

    for fold_index, validation_month in enumerate(MONTHS, start=1):
        anchor_train_mask = anchors["month_key"].lt(validation_month).to_numpy()
        anchor_validation_mask = anchors["month_key"].eq(validation_month).to_numpy()
        eligible_mask = anchor_train_mask | anchor_validation_mask
        run_train_mask = runs["month"].astype(str).lt(validation_month).to_numpy()
        if not anchor_train_mask.any() or not anchor_validation_mask.any() or not run_train_mask.any():
            raise AssertionError(f"empty fold {validation_month}")
        train_runs = runs.loc[run_train_mask].reset_index(drop=True)
        fold_anchors = anchors.loc[eligible_mask].reset_index(names="global_position")
        fold_train_mask = fold_anchors["month_key"].lt(validation_month).to_numpy()
        fold_validation_mask = fold_anchors["month_key"].eq(validation_month).to_numpy()

        destination = fit_destination_model(train_runs)
        loops = predict_loop_scores(destination, fold_anchors, cycles)
        departure_list, departure_parameters = fit_and_predict_departure(
            train_runs, [fold_anchors]
        )
        departure = departure_list[0]
        train_numeric = fold_anchors.loc[fold_train_mask, NUMERIC_CONTROLS].apply(
            pd.to_numeric, errors="coerce"
        )
        medians = train_numeric.median(axis=0)
        phase = static_phase[fold_anchors["global_position"].to_numpy(int)]
        raw_design = build_max_design(
            fold_anchors,
            medians,
            departure,
            loops,
            phase,
            two_cycle_indices,
        )
        scaler = StandardScaler(with_mean=False)
        train_x_max = scaler.fit_transform(raw_design[fold_train_mask]).tocsr()
        validation_x_max = scaler.transform(raw_design[fold_validation_mask]).tocsr()
        validation_designs.append(validation_x_max)
        y_train = fold_anchors.loc[fold_train_mask, outcome_names].to_numpy(float)
        y_validation = fold_anchors.loc[fold_validation_mask, outcome_names].to_numpy(float)

        prefix = validation_month.replace("-", "_")
        parameters[f"{prefix}__destination_classes"] = destination.classes_.copy()
        parameters[f"{prefix}__destination_coef"] = destination.coef_.copy()
        parameters[f"{prefix}__destination_intercept"] = destination.intercept_.copy()
        parameters[f"{prefix}__destination_n_iter"] = destination.n_iter_.copy()
        for name, value in departure_parameters.items():
            parameters[f"{prefix}__{name}"] = value
        parameters[f"{prefix}__outcome_numeric_medians"] = medians.to_numpy(float)
        parameters[f"{prefix}__outcome_scaler_scale"] = scaler.scale_.copy()
        parameters[f"{prefix}__outcome_scaler_mean"] = scaler.mean_.copy()
        parameters[f"{prefix}__outcome_scaler_var"] = scaler.var_.copy()

        selected = fold_anchors.loc[
            fold_validation_mask,
            [
                "anchor_id",
                "symbol_norm",
                "session_date",
                "start_timestamp",
                "month_key",
                "state",
                "history_token",
            ],
        ].reset_index(drop=True)
        for column_index, (target, horizon) in enumerate(
            (item for target in TARGETS for item in ((target, h) for h in HORIZONS))
        ):
            selected[actual_column(target, horizon)] = y_validation[:, column_index]

        for layer in LAYERS:
            width = LAYER_WIDTHS[layer]
            model = Ridge(alpha=10.0, solver="lsqr")
            model.fit(train_x_max[:, :width], y_train)
            prediction = model.predict(validation_x_max[:, :width])
            parameters[f"{prefix}__{layer}__coef"] = model.coef_.copy()
            parameters[f"{prefix}__{layer}__intercept"] = np.asarray(model.intercept_)
            for column_index, (target, horizon) in enumerate(
                (item for target in TARGETS for item in ((target, h) for h in HORIZONS))
            ):
                selected[prediction_column(layer, target, horizon)] = prediction[
                    :, column_index
                ]
        prediction_frames.append(selected)

        structural = selected[
            ["anchor_id", "symbol_norm", "session_date", "start_timestamp", "month_key"]
        ].copy()
        structural["departure_probability_3bar_proxy"] = departure[fold_validation_mask]
        validation_loops = loops[fold_validation_mask]
        for cycle_index in range(20):
            structural[f"loop_probability_{cycle_index + 1:02d}"] = validation_loops[
                :, cycle_index
            ]
        validation_phase = phase[fold_validation_mask]
        for column_index, name in enumerate(static_phase_names):
            structural[name] = validation_phase[:, column_index]
        repeat = validation_phase[:, : len(two_cycle_indices)]
        for local_index, cycle_index in enumerate(two_cycle_indices):
            cycle_id = str(cycles.iloc[int(cycle_index)]["cycle_id"])
            structural[f"burst_loop_x_log_repeat__{cycle_id}"] = (
                validation_loops[:, int(cycle_index)] * repeat[:, local_index]
            )
        structural_frames.append(structural)

        fold_rows.append(
            {
                "fold": fold_index,
                "validation_month": validation_month,
                "maximum_train_anchor_month": str(
                    fold_anchors.loc[fold_train_mask, "month_key"].max()
                ),
                "maximum_train_run_month": str(train_runs["month"].max()),
                "train_anchor_rows": int(fold_train_mask.sum()),
                "validation_anchor_rows": int(fold_validation_mask.sum()),
                "train_run_rows": int(len(train_runs)),
                "destination_iterations_max": int(destination.n_iter_.max()),
                "departure_iterations_max": int(
                    departure_parameters["departure_n_iter"].max()
                ),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    structural = pd.concat(structural_frames, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    design = sparse.vstack(validation_designs, format="csr")
    if predictions["anchor_id"].duplicated().any() or len(predictions) != 34169:
        raise AssertionError("validation prediction cohort drift")
    if design.shape != (len(predictions), LAYER_WIDTHS["burst"]):
        raise AssertionError("stored validation design alignment failure")
    if not np.isfinite(
        predictions.filter(like="prediction__").to_numpy(float)
    ).all():
        raise AssertionError("non-finite outcome prediction")
    return predictions, structural, folds, parameters, design


def layer_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    symbols = sorted(predictions["symbol_norm"].unique())
    for layer in LAYERS:
        for target in TARGETS:
            for horizon in HORIZONS:
                actual = predictions[actual_column(target, horizon)].to_numpy(float)
                predicted = predictions[prediction_column(layer, target, horizon)].to_numpy(float)
                absolute = np.abs(predicted - actual)
                squared = np.square(predicted - actual)
                pooled_rows.append(
                    {
                        "layer": layer,
                        "target": target,
                        "horizon": horizon,
                        "rows": len(predictions),
                        "mse": float(squared.mean()),
                        "mae": float(absolute.mean()),
                    }
                )
                for month in MONTHS:
                    mask = predictions["month_key"].eq(month).to_numpy()
                    monthly_rows.append(
                        {
                            "layer": layer,
                            "target": target,
                            "horizon": horizon,
                            "month": month,
                            "rows": int(mask.sum()),
                            "mse": float(squared[mask].mean()),
                            "mae": float(absolute[mask].mean()),
                        }
                    )
                for deleted_symbol in symbols:
                    mask = predictions["symbol_norm"].ne(deleted_symbol).to_numpy()
                    deletion_rows.append(
                        {
                            "layer": layer,
                            "target": target,
                            "horizon": horizon,
                            "deleted_symbol": deleted_symbol,
                            "rows": int(mask.sum()),
                            "mse": float(squared[mask].mean()),
                            "mae": float(absolute[mask].mean()),
                        }
                    )
    return (
        pd.DataFrame(pooled_rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(deletion_rows),
    )


def moving_block_interval(
    daily_values: np.ndarray, seed_offset: int, draws: int = 5000, block: int = 5
) -> tuple[float, float, float]:
    values = np.asarray(daily_values, dtype=float)
    if len(values) < block:
        raise AssertionError("insufficient daily support for block bootstrap")
    rng = np.random.default_rng(SEED + int(seed_offset))
    starts = np.arange(len(values) - block + 1)
    block_count = math.ceil(len(values) / block)
    samples = np.empty(draws, dtype=float)
    offsets = np.arange(block)
    for draw in range(draws):
        selected_starts = rng.choice(starts, size=block_count, replace=True)
        positions = (selected_starts[:, None] + offsets[None, :]).ravel()[: len(values)]
        samples[draw] = float(values[positions].mean())
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(values.mean()), float(lower), float(upper)


def compare_pair(
    predictions: pd.DataFrame,
    pair_name: str,
    baseline: str,
    candidate: str,
    seed_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cell_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    symbols = sorted(predictions["symbol_norm"].unique())

    baseline_cell_mse: dict[tuple[str, int], float] = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            actual = predictions[actual_column(target, horizon)].to_numpy(float)
            base = predictions[prediction_column(baseline, target, horizon)].to_numpy(float)
            candidate_prediction = predictions[
                prediction_column(candidate, target, horizon)
            ].to_numpy(float)
            base_squared = np.square(base - actual)
            candidate_squared = np.square(candidate_prediction - actual)
            base_absolute = np.abs(base - actual)
            candidate_absolute = np.abs(candidate_prediction - actual)
            baseline_cell_mse[(target, horizon)] = float(base_squared.mean())
            cell_rows.append(
                {
                    "pair": pair_name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "target": target,
                    "horizon": horizon,
                    "baseline_mse": float(base_squared.mean()),
                    "candidate_mse": float(candidate_squared.mean()),
                    "mse_improvement_fraction": float(
                        (base_squared.mean() - candidate_squared.mean())
                        / base_squared.mean()
                    ),
                    "baseline_mae": float(base_absolute.mean()),
                    "candidate_mae": float(candidate_absolute.mean()),
                    "mae_improvement_fraction": float(
                        (base_absolute.mean() - candidate_absolute.mean())
                        / base_absolute.mean()
                    ),
                }
            )
            for deleted_symbol in symbols:
                mask = predictions["symbol_norm"].ne(deleted_symbol).to_numpy()
                base_mse = float(base_squared[mask].mean())
                candidate_mse = float(candidate_squared[mask].mean())
                deletion_rows.append(
                    {
                        "pair": pair_name,
                        "baseline": baseline,
                        "candidate": candidate,
                        "target": target,
                        "horizon": horizon,
                        "deleted_symbol": deleted_symbol,
                        "mse_improvement_fraction": (base_mse - candidate_mse)
                        / base_mse,
                    }
                )

        base_squared_all: list[np.ndarray] = []
        candidate_squared_all: list[np.ndarray] = []
        base_absolute_all: list[np.ndarray] = []
        candidate_absolute_all: list[np.ndarray] = []
        for horizon in HORIZONS:
            actual = predictions[actual_column(target, horizon)].to_numpy(float)
            base = predictions[prediction_column(baseline, target, horizon)].to_numpy(float)
            cand = predictions[prediction_column(candidate, target, horizon)].to_numpy(float)
            base_squared_all.append(np.square(base - actual))
            candidate_squared_all.append(np.square(cand - actual))
            base_absolute_all.append(np.abs(base - actual))
            candidate_absolute_all.append(np.abs(cand - actual))
        base_squared_flat = np.concatenate(base_squared_all)
        candidate_squared_flat = np.concatenate(candidate_squared_all)
        base_absolute_flat = np.concatenate(base_absolute_all)
        candidate_absolute_flat = np.concatenate(candidate_absolute_all)
        target_rows.append(
            {
                "pair": pair_name,
                "baseline": baseline,
                "candidate": candidate,
                "target": target,
                "baseline_mse": float(base_squared_flat.mean()),
                "candidate_mse": float(candidate_squared_flat.mean()),
                "mse_improvement_fraction": float(
                    (base_squared_flat.mean() - candidate_squared_flat.mean())
                    / base_squared_flat.mean()
                ),
                "baseline_mae": float(base_absolute_flat.mean()),
                "candidate_mae": float(candidate_absolute_flat.mean()),
                "mae_improvement_fraction": float(
                    (base_absolute_flat.mean() - candidate_absolute_flat.mean())
                    / base_absolute_flat.mean()
                ),
            }
        )
        for month in MONTHS:
            mask = predictions["month_key"].eq(month).to_numpy()
            base_month = np.concatenate([values[mask] for values in base_squared_all])
            candidate_month = np.concatenate(
                [values[mask] for values in candidate_squared_all]
            )
            month_rows.append(
                {
                    "pair": pair_name,
                    "baseline": baseline,
                    "candidate": candidate,
                    "target": target,
                    "month": month,
                    "baseline_mse": float(base_month.mean()),
                    "candidate_mse": float(candidate_month.mean()),
                    "mse_improvement_fraction": float(
                        (base_month.mean() - candidate_month.mean()) / base_month.mean()
                    ),
                }
            )

        normalized = np.column_stack(
            [
                (candidate_squared_all[index] - base_squared_all[index])
                / baseline_cell_mse[(target, horizon)]
                for index, horizon in enumerate(HORIZONS)
            ]
        ).mean(axis=1)
        daily = (
            pd.DataFrame(
                {
                    "session_date": predictions["session_date"].to_numpy(),
                    "normalized_difference": normalized,
                }
            )
            .groupby("session_date", sort=True)["normalized_difference"]
            .mean()
        )
        observed, lower, upper = moving_block_interval(
            daily.to_numpy(float), seed_index * 10 + TARGETS.index(target)
        )
        bootstrap_rows.append(
            {
                "pair": pair_name,
                "baseline": baseline,
                "candidate": candidate,
                "target": target,
                "session_dates": len(daily),
                "normalized_mse_difference": observed,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

    cells = pd.DataFrame(cell_rows)
    targets = pd.DataFrame(target_rows)
    months = pd.DataFrame(month_rows)
    deletions = pd.DataFrame(deletion_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    checks = {
        "pooled_mse_positive_both_targets": bool(
            targets["mse_improvement_fraction"].gt(0.0).all()
        ),
        "pooled_mae_positive_both_targets": bool(
            targets["mae_improvement_fraction"].gt(0.0).all()
        ),
        "all_six_cells_mse_positive": bool(
            cells["mse_improvement_fraction"].gt(0.0).all()
        ),
        "at_least_four_months_each_target": bool(
            months.assign(positive=months["mse_improvement_fraction"].gt(0.0))
            .groupby("target")["positive"]
            .sum()
            .ge(4)
            .all()
        ),
        "every_stock_deletion_all_six_cells_positive": bool(
            deletions["mse_improvement_fraction"].gt(0.0).all()
        ),
        "bootstrap_upper_below_zero_both_targets": bool(
            bootstraps["ci_upper"].lt(0.0).all()
        ),
    }
    checks["retained"] = bool(all(checks.values()))
    decision = {
        "pair": pair_name,
        "baseline": baseline,
        "candidate": candidate,
        "checks": checks,
        "minimum_cell_mse_improvement_fraction": float(
            cells["mse_improvement_fraction"].min()
        ),
        "minimum_stock_deletion_mse_improvement_fraction": float(
            deletions["mse_improvement_fraction"].min()
        ),
        "positive_months_by_target": {
            target: int(
                months.loc[
                    months["target"].eq(target), "mse_improvement_fraction"
                ].gt(0.0).sum()
            )
            for target in TARGETS
        },
    }
    return cells, targets, months, deletions, bootstraps, decision


def evaluate_comparisons(
    predictions: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    all_cells: list[pd.DataFrame] = []
    all_targets: list[pd.DataFrame] = []
    all_months: list[pd.DataFrame] = []
    all_deletions: list[pd.DataFrame] = []
    all_bootstraps: list[pd.DataFrame] = []
    decisions: dict[str, Any] = {}
    pairs = list(INCREMENTAL_PAIRS) + [("burst_vs_context", "context", "burst")]
    for index, (pair_name, baseline, candidate) in enumerate(pairs):
        cells, targets, months, deletions, bootstraps, decision = compare_pair(
            predictions, pair_name, baseline, candidate, index
        )
        all_cells.append(cells)
        all_targets.append(targets)
        all_months.append(months)
        all_deletions.append(deletions)
        all_bootstraps.append(bootstraps)
        decisions[pair_name] = decision
    tables = {
        "cells": pd.concat(all_cells, ignore_index=True),
        "targets": pd.concat(all_targets, ignore_index=True),
        "months": pd.concat(all_months, ignore_index=True),
        "deletions": pd.concat(all_deletions, ignore_index=True),
        "bootstraps": pd.concat(all_bootstraps, ignore_index=True),
    }
    return tables, decisions


def final_decision(
    comparison_tables: dict[str, pd.DataFrame], decisions: dict[str, Any]
) -> dict[str, Any]:
    regime = decisions["state_vs_context"]
    final_targets = comparison_tables["targets"].loc[
        comparison_tables["targets"]["pair"].eq("burst_vs_context")
    ]
    final_cells = comparison_tables["cells"].loc[
        comparison_tables["cells"]["pair"].eq("burst_vs_context")
    ]
    final_deletions = comparison_tables["deletions"].loc[
        comparison_tables["deletions"]["pair"].eq("burst_vs_context")
    ]
    final_bootstraps = comparison_tables["bootstraps"].loc[
        comparison_tables["bootstraps"]["pair"].eq("burst_vs_context")
    ]
    abs_improvement = float(
        final_targets.loc[
            final_targets["target"].eq("absolute_return_bps"),
            "mse_improvement_fraction",
        ].iloc[0]
    )
    range_improvement = float(
        final_targets.loc[
            final_targets["target"].eq("future_range_bps"),
            "mse_improvement_fraction",
        ].iloc[0]
    )
    magnitude_checks = {
        "absolute_return_mse_at_least_one_percent": abs_improvement >= 0.01,
        "future_range_mse_at_least_three_percent": range_improvement >= 0.03,
        "all_six_horizons_positive": bool(
            final_cells["mse_improvement_fraction"].gt(0.0).all()
        ),
        "every_stock_deletion_all_six_horizons_positive": bool(
            final_deletions["mse_improvement_fraction"].gt(0.0).all()
        ),
        "bootstrap_upper_below_zero_both_targets": bool(
            final_bootstraps["ci_upper"].lt(0.0).all()
        ),
    }
    magnitude_checks["pass"] = bool(all(magnitude_checks.values()))
    if regime["checks"]["retained"]:
        regime_label = "reliably_useful_for_movement_magnitude_in_2024_internal_forward_test"
    elif (
        regime["checks"]["pooled_mse_positive_both_targets"]
        and comparison_tables["cells"]
        .loc[comparison_tables["cells"]["pair"].eq("state_vs_context")]
        ["mse_improvement_fraction"]
        .gt(0.0)
        .sum()
        >= 5
    ):
        regime_label = "weak_incremental_utility_not_robust"
    else:
        regime_label = "incremental_movement_utility_not_supported"
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "regime_utility_label": regime_label,
        "regime_reliably_useful": bool(regime["checks"]["retained"]),
        "incremental_layer_decisions": {
            name: decisions[name] for name, _, _ in INCREMENTAL_PAIRS
        },
        "final_stack_vs_context": {
            "absolute_return_mse_improvement_fraction": abs_improvement,
            "future_range_mse_improvement_fraction": range_improvement,
            "magnitude_checks": magnitude_checks,
        },
        "prospective_validation_claim": False,
        "direction_or_signed_return_claim": False,
        "high_or_good_loop_grade_inferred": False,
    }


def feature_manifest(
    cycles: pd.DataFrame, static_phase_names: list[str]
) -> dict[str, Any]:
    two_state = cycles.loc[cycles["transition_length"].eq(2), "cycle_id"].tolist()
    burst_names = static_phase_names + [
        f"burst_loop_x_log_repeat__{cycle_id}" for cycle_id in two_state
    ]
    return {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "provider_volume_label": "historical_volume_not_used",
        "numeric_controls": list(NUMERIC_CONTROLS),
        "layer_widths": LAYER_WIDTHS,
        "layer_order": list(LAYERS),
        "history_token_width": TOKEN_COUNT,
        "loop_probability_columns": [
            f"loop_probability_{index:02d}" for index in range(1, 21)
        ],
        "two_state_cycle_ids": two_state,
        "burst_columns": burst_names,
        "burst_current_or_future_duration_used": False,
        "stock_identity_used": False,
        "future_price_or_state_label_used_as_feature": False,
    }


def main() -> None:
    contract, pre_score = load_contract_and_pre_score()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {OUT}")
    OUT.mkdir(parents=True)
    anchors, runs, cycles = load_inputs()
    static_phase, static_phase_names, two_cycle_indices = build_causal_phase_block(
        anchors, runs, cycles
    )
    predictions, structural, folds, parameters, validation_design = score_folds(
        anchors,
        runs,
        cycles,
        static_phase,
        static_phase_names,
        two_cycle_indices,
    )
    pooled, monthly, layer_deletions = layer_metrics(predictions)
    comparison_tables, decisions = evaluate_comparisons(predictions)
    decision = final_decision(comparison_tables, decisions)

    predictions.to_parquet(OUT / "oof_predictions_2024.parquet", index=False)
    structural.to_parquet(OUT / "oof_structural_features_2024.parquet", index=False)
    folds.to_csv(OUT / "fold_audit.csv", index=False)
    pooled.to_csv(OUT / "layer_metrics_pooled.csv", index=False)
    monthly.to_csv(OUT / "layer_metrics_monthly.csv", index=False)
    layer_deletions.to_csv(OUT / "layer_metrics_stock_deletions.csv", index=False)
    comparison_tables["cells"].to_csv(OUT / "incremental_cells.csv", index=False)
    comparison_tables["targets"].to_csv(OUT / "incremental_targets.csv", index=False)
    comparison_tables["months"].to_csv(OUT / "incremental_months.csv", index=False)
    comparison_tables["deletions"].to_csv(
        OUT / "incremental_stock_deletions.csv", index=False
    )
    comparison_tables["bootstraps"].to_csv(
        OUT / "incremental_bootstraps.csv", index=False
    )
    np.savez_compressed(OUT / "model_parameters.npz", **parameters)
    sparse.save_npz(OUT / "validation_design_matrix.npz", validation_design)
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "feature_manifest.json", feature_manifest(cycles, static_phase_names))
    write_json(
        OUT / "source_hashes.json",
        {
            **pre_score,
            "pre_score_manifest_sha256": sha256(PRE_SCORE_PATH),
        },
    )
    summary = {
        "contract_id": contract["contract_id"],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "anchor_rows_total_2024": len(anchors),
        "validation_rows_july_december": len(predictions),
        "run_rows_total_2024": len(runs),
        "stocks": int(predictions["symbol_norm"].nunique()),
        "session_dates": int(predictions["session_date"].nunique()),
        "folds": folds.to_dict(orient="records"),
        "decision": decision,
        "incremental_targets": comparison_tables["targets"].to_dict(orient="records"),
        "incremental_bootstraps": comparison_tables["bootstraps"].to_dict(
            orient="records"
        ),
    }
    write_json(OUT / "summary.json", summary)
    artifact_files = sorted(path for path in OUT.iterdir() if path.is_file())
    write_json(
        OUT / "artifact_manifest.json",
        {
            "files": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in artifact_files
            ],
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
