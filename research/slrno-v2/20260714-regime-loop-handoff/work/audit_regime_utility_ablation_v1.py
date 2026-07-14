#!/usr/bin/env python3
"""Independent artifact audit for regime-utility ablation V1."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, softmax


WORK = Path(__file__).resolve().parent
CONTRACT = WORK / "contracts/20260711-regime-utility-ablation-v1.json"
PRE_SCORE = WORK / "contracts/20260711-regime-utility-ablation-v1-pre-score.json"
RUNNER = WORK / "run_regime_utility_ablation_v1.py"
ANCHORS = Path(
    "/private/tmp/stocker_frozen_loop_price_consequence_20260710/"
    "anchor_panel_train_2024.parquet"
)
RUNS = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/"
    "train_2024_filtered_runs.csv"
)
CYCLES = Path(
    "/private/tmp/stocker_per_loop_movement_quality_20260710/fixed_cycles.csv"
)
ROOT = Path("/private/tmp/stocker_regime_utility_ablation_v1_20260711")

SEED = 20260711
K = 8
END_STATE = 8
TOKEN_COUNT = 648
MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
HORIZONS = (6, 12, 24)
TARGETS = ("absolute_return_bps", "future_range_bps")
LAYERS = ("context", "state", "history", "departure", "loops", "burst")
WIDTHS = {
    "context": 9,
    "state": 17,
    "history": 665,
    "departure": 666,
    "loops": 686,
    "burst": 738,
}
PAIRS = (
    ("state_vs_context", "context", "state"),
    ("history_vs_state", "state", "history"),
    ("departure_vs_history", "history", "departure"),
    ("loops_vs_departure", "departure", "loops"),
    ("burst_vs_loops", "loops", "burst"),
    ("burst_vs_context", "context", "burst"),
)
NUMERIC = (
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
DEPARTURE_NUMERIC = (
    "b0_entry_numeric",
    "b0_entry_high_stress",
    "entry_time_sin",
    "entry_time_cos",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def history_token(previous_2: np.ndarray, previous_1: np.ndarray, state: np.ndarray) -> np.ndarray:
    return ((np.asarray(previous_2, int) * 9 + np.asarray(previous_1, int)) * 8 + np.asarray(state, int))


def canonical_core(value: str) -> tuple[int, ...]:
    closed = tuple(int(part) for part in value.split("->"))
    raw = closed[:-1]
    return min(raw[index:] + raw[:index] for index in range(len(raw)))


def rotations(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (current,)
            for index, state in enumerate(core)
            if state == current
        }
    )


def actual_column(target: str, horizon: int) -> str:
    return f"actual__{target}__h{horizon}"


def prediction_column(layer: str, target: str, horizon: int) -> str:
    return f"prediction__{layer}__{target}__h{horizon}"


def assert_close(left: np.ndarray, right: np.ndarray, name: str, atol: float = 1e-9) -> float:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if left.shape != right.shape:
        raise AssertionError(f"{name} shape mismatch: {left.shape} != {right.shape}")
    maximum = float(np.max(np.abs(left - right), initial=0.0))
    if not np.allclose(left, right, rtol=1e-10, atol=atol, equal_nan=True):
        raise AssertionError(f"{name} mismatch; max abs={maximum}")
    return maximum


def reconstruct_loop_probabilities(
    frame: pd.DataFrame,
    cycle_frame: pd.DataFrame,
    classes: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    if not np.array_equal(classes, np.arange(9)) or coefficients.shape != (9, 648):
        raise AssertionError("invalid saved destination parameters")

    def probability(tokens: np.ndarray, destination: int) -> np.ndarray:
        logits = coefficients[:, np.asarray(tokens, int)].T + intercepts[None, :]
        return np.clip(softmax(logits, axis=1)[:, int(destination)], 1e-12, 1.0 - 1e-12)

    output = np.zeros((len(frame), 20), dtype=float)
    states = frame["state"].to_numpy(int)
    prior_2_all = frame["previous_state_2"].to_numpy(int)
    prior_1_all = frame["previous_state_1"].to_numpy(int)
    for cycle_index, value in enumerate(cycle_frame["cycle"].astype(str)):
        core = canonical_core(value)
        for current in sorted(set(core)):
            positions = np.flatnonzero(states == current)
            total = np.zeros(len(positions), dtype=float)
            for path in rotations(core, current):
                path_probability = np.ones(len(positions), dtype=float)
                prior_2 = prior_2_all[positions].copy()
                prior_1 = prior_1_all[positions].copy()
                current_state = np.full(len(positions), current, dtype=int)
                for destination in path[1:]:
                    tokens = history_token(prior_2, prior_1, current_state)
                    path_probability *= probability(tokens, destination)
                    prior_2, prior_1, current_state = (
                        prior_1,
                        current_state,
                        np.full(len(positions), destination, dtype=int),
                    )
                total += path_probability
            output[positions, cycle_index] = np.clip(total, 1e-12, 1.0 - 1e-12)
    return output


def reconstruct_phase(
    validation: pd.DataFrame, runs: pd.DataFrame, cycle_frame: pd.DataFrame
) -> tuple[np.ndarray, list[int], list[str]]:
    run_frame = runs.copy()
    run_frame["start_timestamp"] = pd.to_datetime(run_frame["start_timestamp"], utc=True)
    run_frame = run_frame.sort_values(
        ["symbol_norm", "session_date", "start_timestamp"], kind="stable"
    ).reset_index(drop=True)
    run_frame["session_run_index"] = run_frame.groupby(
        ["symbol_norm", "session_date"], sort=False
    ).cumcount()
    lookup = run_frame[
        ["symbol_norm", "session_date", "start_timestamp", "state", "session_run_index"]
    ].rename(columns={"state": "run_state"})
    positioned = (
        validation.reset_index(names="position")
        .merge(
            lookup,
            on=["symbol_norm", "session_date", "start_timestamp"],
            how="left",
            validate="one_to_one",
        )
        .sort_values("position", kind="stable")
    )
    if positioned["session_run_index"].isna().any():
        raise AssertionError("audit phase join failure")
    sequences = {
        key: (group["state"].to_numpy(int), group["duration"].to_numpy(int))
        for key, group in run_frame.groupby(["symbol_norm", "session_date"], sort=False)
    }
    two_indices = [
        index
        for index, value in enumerate(cycle_frame["cycle"].astype(str))
        if len(canonical_core(value)) == 2
    ]
    identifiers = [str(cycle_frame.iloc[index]["cycle_id"]) for index in two_indices]
    repeat = np.zeros((len(validation), 13), dtype=float)
    pair = np.zeros_like(repeat)
    durable = np.zeros_like(repeat)
    compatible: dict[int, list[tuple[int, int]]] = {state: [] for state in range(8)}
    for local, cycle_index in enumerate(two_indices):
        left, right = canonical_core(str(cycle_frame.iloc[cycle_index]["cycle"]))
        compatible[left].append((local, right))
        compatible[right].append((local, left))
    for row in positioned[
        ["position", "symbol_norm", "session_date", "session_run_index", "state"]
    ].itertuples(index=False):
        states, durations = sequences[(row.symbol_norm, row.session_date)]
        index = int(row.session_run_index)
        current = int(row.state)
        for local, other in compatible[current]:
            count = 0
            cursor = index
            while cursor >= 2 and states[cursor - 1] == other and states[cursor - 2] == current:
                count += 1
                cursor -= 2
            if count:
                prior_current = int(durations[index - 2])
                prior_other = int(durations[index - 1])
                repeat[int(row.position), local] = np.log1p(count)
                pair[int(row.position), local] = np.log1p(prior_current + prior_other)
                durable[int(row.position), local] = float(prior_current >= 2 and prior_other >= 2)
    return np.hstack((repeat, pair, durable)), two_indices, identifiers


def moving_block(values: np.ndarray, seed_offset: int) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    block = 5
    draws = 5000
    rng = np.random.default_rng(SEED + seed_offset)
    starts = np.arange(len(values) - block + 1)
    block_count = math.ceil(len(values) / block)
    sampled = np.empty(draws, dtype=float)
    offsets = np.arange(block)
    for draw in range(draws):
        chosen = rng.choice(starts, size=block_count, replace=True)
        positions = (chosen[:, None] + offsets).ravel()[: len(values)]
        sampled[draw] = values[positions].mean()
    lower, upper = np.quantile(sampled, [0.025, 0.975], method="linear")
    return float(values.mean()), float(lower), float(upper)


def recompute_tables(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pooled: list[dict[str, Any]] = []
    monthly: list[dict[str, Any]] = []
    layer_deletions: list[dict[str, Any]] = []
    symbols = sorted(predictions["symbol_norm"].unique())
    for layer in LAYERS:
        for target in TARGETS:
            for horizon in HORIZONS:
                actual = predictions[actual_column(target, horizon)].to_numpy(float)
                forecast = predictions[prediction_column(layer, target, horizon)].to_numpy(float)
                squared = np.square(forecast - actual)
                absolute = np.abs(forecast - actual)
                pooled.append(
                    dict(layer=layer, target=target, horizon=horizon, rows=len(predictions), mse=squared.mean(), mae=absolute.mean())
                )
                for month in MONTHS:
                    mask = predictions["month_key"].eq(month).to_numpy()
                    monthly.append(
                        dict(layer=layer, target=target, horizon=horizon, month=month, rows=mask.sum(), mse=squared[mask].mean(), mae=absolute[mask].mean())
                    )
                for symbol in symbols:
                    mask = predictions["symbol_norm"].ne(symbol).to_numpy()
                    layer_deletions.append(
                        dict(layer=layer, target=target, horizon=horizon, deleted_symbol=symbol, rows=mask.sum(), mse=squared[mask].mean(), mae=absolute[mask].mean())
                    )

    cells: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    month_rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for pair_index, (pair, baseline, candidate) in enumerate(PAIRS):
        baseline_cell: dict[tuple[str, int], float] = {}
        for target in TARGETS:
            base_sq_all: list[np.ndarray] = []
            cand_sq_all: list[np.ndarray] = []
            base_abs_all: list[np.ndarray] = []
            cand_abs_all: list[np.ndarray] = []
            for horizon in HORIZONS:
                actual = predictions[actual_column(target, horizon)].to_numpy(float)
                base = predictions[prediction_column(baseline, target, horizon)].to_numpy(float)
                cand = predictions[prediction_column(candidate, target, horizon)].to_numpy(float)
                base_sq = np.square(base - actual)
                cand_sq = np.square(cand - actual)
                base_abs = np.abs(base - actual)
                cand_abs = np.abs(cand - actual)
                baseline_cell[(target, horizon)] = base_sq.mean()
                base_sq_all.append(base_sq)
                cand_sq_all.append(cand_sq)
                base_abs_all.append(base_abs)
                cand_abs_all.append(cand_abs)
                cells.append(
                    dict(
                        pair=pair,
                        baseline=baseline,
                        candidate=candidate,
                        target=target,
                        horizon=horizon,
                        baseline_mse=base_sq.mean(),
                        candidate_mse=cand_sq.mean(),
                        mse_improvement_fraction=(base_sq.mean() - cand_sq.mean()) / base_sq.mean(),
                        baseline_mae=base_abs.mean(),
                        candidate_mae=cand_abs.mean(),
                        mae_improvement_fraction=(base_abs.mean() - cand_abs.mean()) / base_abs.mean(),
                    )
                )
                for symbol in symbols:
                    mask = predictions["symbol_norm"].ne(symbol).to_numpy()
                    base_mse = base_sq[mask].mean()
                    cand_mse = cand_sq[mask].mean()
                    deletion_rows.append(
                        dict(pair=pair, baseline=baseline, candidate=candidate, target=target, horizon=horizon, deleted_symbol=symbol, mse_improvement_fraction=(base_mse - cand_mse) / base_mse)
                    )
            base_sq_flat = np.concatenate(base_sq_all)
            cand_sq_flat = np.concatenate(cand_sq_all)
            base_abs_flat = np.concatenate(base_abs_all)
            cand_abs_flat = np.concatenate(cand_abs_all)
            target_rows.append(
                dict(
                    pair=pair,
                    baseline=baseline,
                    candidate=candidate,
                    target=target,
                    baseline_mse=base_sq_flat.mean(),
                    candidate_mse=cand_sq_flat.mean(),
                    mse_improvement_fraction=(base_sq_flat.mean() - cand_sq_flat.mean()) / base_sq_flat.mean(),
                    baseline_mae=base_abs_flat.mean(),
                    candidate_mae=cand_abs_flat.mean(),
                    mae_improvement_fraction=(base_abs_flat.mean() - cand_abs_flat.mean()) / base_abs_flat.mean(),
                )
            )
            for month in MONTHS:
                mask = predictions["month_key"].eq(month).to_numpy()
                base_month = np.concatenate([values[mask] for values in base_sq_all])
                cand_month = np.concatenate([values[mask] for values in cand_sq_all])
                month_rows.append(
                    dict(pair=pair, baseline=baseline, candidate=candidate, target=target, month=month, baseline_mse=base_month.mean(), candidate_mse=cand_month.mean(), mse_improvement_fraction=(base_month.mean() - cand_month.mean()) / base_month.mean())
                )
            normalized = np.column_stack(
                [
                    (cand_sq_all[index] - base_sq_all[index]) / baseline_cell[(target, horizon)]
                    for index, horizon in enumerate(HORIZONS)
                ]
            ).mean(axis=1)
            daily = (
                pd.DataFrame(dict(session_date=predictions["session_date"], value=normalized))
                .groupby("session_date", sort=True)["value"]
                .mean()
            )
            observed, lower, upper = moving_block(
                daily.to_numpy(float), pair_index * 10 + TARGETS.index(target)
            )
            bootstrap_rows.append(
                dict(pair=pair, baseline=baseline, candidate=candidate, target=target, session_dates=len(daily), normalized_mse_difference=observed, ci_lower=lower, ci_upper=upper)
            )
    return {
        "layer_metrics_pooled.csv": pd.DataFrame(pooled),
        "layer_metrics_monthly.csv": pd.DataFrame(monthly),
        "layer_metrics_stock_deletions.csv": pd.DataFrame(layer_deletions),
        "incremental_cells.csv": pd.DataFrame(cells),
        "incremental_targets.csv": pd.DataFrame(target_rows),
        "incremental_months.csv": pd.DataFrame(month_rows),
        "incremental_stock_deletions.csv": pd.DataFrame(deletion_rows),
        "incremental_bootstraps.csv": pd.DataFrame(bootstrap_rows),
    }


def dataframe_close(observed: pd.DataFrame, expected: pd.DataFrame, name: str) -> float:
    if list(observed.columns) != list(expected.columns) or len(observed) != len(expected):
        raise AssertionError(f"{name} frame shape/column mismatch")
    maximum = 0.0
    for column in observed.columns:
        if pd.api.types.is_numeric_dtype(observed[column]):
            maximum = max(
                maximum,
                assert_close(observed[column], expected[column], f"{name}:{column}", atol=1e-10),
            )
        elif not observed[column].astype(str).equals(expected[column].astype(str)):
            raise AssertionError(f"{name}:{column} value mismatch")
    return maximum


def main() -> None:
    checks: list[dict[str, Any]] = []

    def record(name: str, detail: Any) -> None:
        checks.append({"name": name, "pass": True, "detail": detail})

    contract = json.loads(CONTRACT.read_text())
    pre_score = json.loads(PRE_SCORE.read_text())
    source_hashes = json.loads((ROOT / "source_hashes.json").read_text())
    actual_hashes = {
        "contract": digest(CONTRACT),
        "runner": digest(RUNNER),
        "anchor_panel": digest(ANCHORS),
        "causal_runs": digest(RUNS),
        "fixed_cycles": digest(CYCLES),
    }
    if actual_hashes != pre_score["sha256"] or actual_hashes != source_hashes["sha256"]:
        raise AssertionError("frozen source hash mismatch")
    record("frozen_source_hashes", actual_hashes)
    if digest(PRE_SCORE) != source_hashes["pre_score_manifest_sha256"]:
        raise AssertionError("pre-score manifest hash mismatch")
    record("pre_score_manifest_hash", digest(PRE_SCORE))
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
    ):
        raise AssertionError("contract safety boundary failure")
    record("contract_safety_boundary", "research only; ordering disabled")

    predictions = pd.read_parquet(ROOT / "oof_predictions_2024.parquet")
    structural = pd.read_parquet(ROOT / "oof_structural_features_2024.parquet")
    folds = pd.read_csv(ROOT / "fold_audit.csv")
    parameters = np.load(ROOT / "model_parameters.npz")
    design = sparse.load_npz(ROOT / "validation_design_matrix.npz").tocsr()
    if len(predictions) != 34169 or predictions["anchor_id"].duplicated().any():
        raise AssertionError("prediction cohort mismatch")
    record("prediction_cohort", {"rows": len(predictions), "anchors": predictions.anchor_id.nunique()})
    if tuple(predictions["month_key"].drop_duplicates()) != MONTHS:
        raise AssertionError("prediction month order mismatch")
    if not predictions["session_date"].astype(str).str.startswith("2024-").all():
        raise AssertionError("later-period prediction found")
    record("period_boundary", list(predictions["month_key"].drop_duplicates()))
    if not np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise AssertionError("non-finite prediction artifact")
    record("prediction_finiteness", True)
    forbidden = ("direction", "signed_return", "pnl", "profit", "cost", "spread", "slippage", "position", "order", "broker", "strategy", "deployment")
    bad = [column for column in predictions.columns if any(token in column.lower() for token in forbidden)]
    if bad:
        raise AssertionError(f"forbidden prediction fields: {bad}")
    record("forbidden_fields_absent", True)
    if design.shape != (34169, 738) or not np.isfinite(design.data).all():
        raise AssertionError("validation design matrix mismatch")
    record("validation_design", {"shape": list(design.shape), "nonzero": int(design.nnz)})
    if not all(str(row.maximum_train_anchor_month) < str(row.validation_month) and str(row.maximum_train_run_month) < str(row.validation_month) for row in folds.itertuples(index=False)):
        raise AssertionError("fold chronology failure")
    record("fold_chronology", folds[["validation_month", "maximum_train_anchor_month", "maximum_train_run_month"]].to_dict(orient="records"))

    anchor_columns = [
        "anchor_id", "symbol_norm", "session_date", "start_timestamp", "state",
        "previous_state_1", "previous_state_2", "history_token", *NUMERIC,
        *(f"{target}_{horizon}" for target in TARGETS for horizon in HORIZONS),
    ]
    source = pd.read_parquet(ANCHORS, columns=anchor_columns)
    source["start_timestamp"] = pd.to_datetime(source["start_timestamp"], utc=True)
    source["month_key"] = source["session_date"].astype(str).str.slice(0, 7)
    source = predictions[["anchor_id"]].merge(source, on="anchor_id", validate="one_to_one")
    if not predictions[["symbol_norm", "session_date", "state", "history_token"]].reset_index(drop=True).equals(
        source[["symbol_norm", "session_date", "state", "history_token"]].reset_index(drop=True)
    ):
        raise AssertionError("source identifier/state mismatch")
    record("source_identifier_state_match", True)
    maximum_actual = 0.0
    for target in TARGETS:
        for horizon in HORIZONS:
            maximum_actual = max(
                maximum_actual,
                assert_close(
                    predictions[actual_column(target, horizon)],
                    source[f"{target}_{horizon}"],
                    f"source outcome {target} h{horizon}",
                ),
            )
    record("source_outcome_match", maximum_actual)
    expected_history = history_token(
        source["previous_state_2"], source["previous_state_1"], source["state"]
    )
    if not np.array_equal(expected_history, source["history_token"].to_numpy(int)):
        raise AssertionError("history token reconstruction failure")
    record("history_token_reconstruction", True)

    outcome_order = [(target, horizon) for target in TARGETS for horizon in HORIZONS]
    max_prediction_error = 0.0
    for month in MONTHS:
        prefix = month.replace("-", "_")
        positions = np.flatnonzero(predictions["month_key"].eq(month).to_numpy())
        for layer in LAYERS:
            coefficients = parameters[f"{prefix}__{layer}__coef"]
            intercept = parameters[f"{prefix}__{layer}__intercept"]
            reconstructed = design[positions, : WIDTHS[layer]] @ coefficients.T + intercept
            observed = np.column_stack(
                [predictions.loc[positions, prediction_column(layer, target, horizon)] for target, horizon in outcome_order]
            )
            max_prediction_error = max(
                max_prediction_error,
                assert_close(reconstructed, observed, f"outcome prediction {month} {layer}", atol=2e-8),
            )
    record("exact_outcome_prediction_reconstruction", max_prediction_error)

    cycles = pd.read_csv(CYCLES)
    loop_columns = [f"loop_probability_{index:02d}" for index in range(1, 21)]
    max_loop_error = 0.0
    max_departure_error = 0.0
    for month in MONTHS:
        prefix = month.replace("-", "_")
        mask = predictions["month_key"].eq(month).to_numpy()
        month_source = source.loc[mask].reset_index(drop=True)
        month_structural = structural.loc[structural["month_key"].eq(month)].reset_index(drop=True)
        reconstructed_loops = reconstruct_loop_probabilities(
            month_source,
            cycles,
            parameters[f"{prefix}__destination_classes"],
            parameters[f"{prefix}__destination_coef"],
            parameters[f"{prefix}__destination_intercept"],
        )
        max_loop_error = max(
            max_loop_error,
            assert_close(reconstructed_loops, month_structural[loop_columns], f"loop probabilities {month}", atol=2e-9),
        )

        numeric = month_source.loc[:, DEPARTURE_NUMERIC].apply(pd.to_numeric, errors="coerce")
        medians = parameters[f"{prefix}__departure_medians"]
        for index, column in enumerate(DEPARTURE_NUMERIC):
            numeric[column] = numeric[column].fillna(medians[index])
        scale = parameters[f"{prefix}__departure_scaler_scale"]
        coef = parameters[f"{prefix}__departure_coef"][0]
        state = month_source["state"].to_numpy(int)
        token = month_source["history_token"].to_numpy(int)
        logit = (
            parameters[f"{prefix}__departure_intercept"][0]
            + coef[state] / scale[state]
            + coef[8 + token] / scale[8 + token]
            + (numeric.to_numpy(float) * (coef[656:] / scale[656:])).sum(axis=1)
        )
        departure = np.clip(expit(logit), 1e-12, 1.0 - 1e-12)
        max_departure_error = max(
            max_departure_error,
            assert_close(departure, month_structural["departure_probability_3bar_proxy"], f"departure {month}", atol=2e-7),
        )
    record("exact_loop_probability_reconstruction", max_loop_error)
    record("exact_departure_probability_reconstruction", max_departure_error)
    probability_values = structural[["departure_probability_3bar_proxy", *loop_columns]].to_numpy(float)
    if not np.isfinite(probability_values).all() or probability_values.min() < 0 or probability_values.max() > 1:
        raise AssertionError("structural probability bounds failure")
    record("structural_probability_bounds", [float(probability_values.min()), float(probability_values.max())])

    run_frame = pd.read_csv(RUNS, usecols=["symbol_norm", "session_date", "start_timestamp", "state", "duration"])
    phase, two_indices, cycle_ids = reconstruct_phase(source, run_frame, cycles)
    phase_columns = (
        [f"burst_log_repeat__{value}" for value in cycle_ids]
        + [f"burst_log_prior_pair__{value}" for value in cycle_ids]
        + [f"burst_prior_durable__{value}" for value in cycle_ids]
    )
    max_phase_error = assert_close(phase, structural[phase_columns], "causal burst phase", atol=2e-7)
    record("exact_past_only_burst_phase_reconstruction", max_phase_error)
    max_interaction_error = 0.0
    for local, cycle_index in enumerate(two_indices):
        expected = structural[f"loop_probability_{cycle_index + 1:02d}"].to_numpy(float) * phase[:, local]
        observed = structural[f"burst_loop_x_log_repeat__{cycle_ids[local]}"].to_numpy(float)
        max_interaction_error = max(max_interaction_error, assert_close(expected, observed, f"burst interaction {cycle_ids[local]}", atol=2e-7))
    record("exact_loop_phase_interactions", max_interaction_error)

    recomputed = recompute_tables(predictions)
    metric_errors: dict[str, float] = {}
    for filename, expected in recomputed.items():
        observed = pd.read_csv(ROOT / filename)
        metric_errors[filename] = dataframe_close(observed, expected, filename)
    record("exact_all_metric_tables", metric_errors)

    incremental_cells = recomputed["incremental_cells.csv"]
    incremental_targets = recomputed["incremental_targets.csv"]
    incremental_months = recomputed["incremental_months.csv"]
    incremental_deletions = recomputed["incremental_stock_deletions.csv"]
    incremental_bootstraps = recomputed["incremental_bootstraps.csv"]
    decision = json.loads((ROOT / "decision.json").read_text())
    reconstructed_decisions: dict[str, bool] = {}
    for pair, _, _ in PAIRS[:5]:
        cells = incremental_cells[incremental_cells.pair.eq(pair)]
        targets = incremental_targets[incremental_targets.pair.eq(pair)]
        months = incremental_months[incremental_months.pair.eq(pair)]
        deletions = incremental_deletions[incremental_deletions.pair.eq(pair)]
        bootstraps = incremental_bootstraps[incremental_bootstraps.pair.eq(pair)]
        gate_checks = {
            "pooled_mse_positive_both_targets": bool(targets.mse_improvement_fraction.gt(0).all()),
            "pooled_mae_positive_both_targets": bool(targets.mae_improvement_fraction.gt(0).all()),
            "all_six_cells_mse_positive": bool(cells.mse_improvement_fraction.gt(0).all()),
            "at_least_four_months_each_target": bool(months.assign(positive=months.mse_improvement_fraction.gt(0)).groupby("target").positive.sum().ge(4).all()),
            "every_stock_deletion_all_six_cells_positive": bool(deletions.mse_improvement_fraction.gt(0).all()),
            "bootstrap_upper_below_zero_both_targets": bool(bootstraps.ci_upper.lt(0).all()),
        }
        gate_checks["retained"] = all(gate_checks.values())
        observed_checks = decision["incremental_layer_decisions"][pair]["checks"]
        if gate_checks != observed_checks:
            raise AssertionError(f"decision gate mismatch {pair}: {gate_checks} != {observed_checks}")
        reconstructed_decisions[pair] = gate_checks["retained"]
    record("exact_incremental_decisions", reconstructed_decisions)
    if decision["regime_reliably_useful"] is not reconstructed_decisions["state_vs_context"]:
        raise AssertionError("regime decision mismatch")
    record("regime_decision", decision["regime_utility_label"])
    final_targets = incremental_targets[incremental_targets.pair.eq("burst_vs_context")]
    final_cells = incremental_cells[incremental_cells.pair.eq("burst_vs_context")]
    final_deletions = incremental_deletions[incremental_deletions.pair.eq("burst_vs_context")]
    final_bootstraps = incremental_bootstraps[incremental_bootstraps.pair.eq("burst_vs_context")]
    final_checks = {
        "absolute_return_mse_at_least_one_percent": float(final_targets.loc[final_targets.target.eq("absolute_return_bps"), "mse_improvement_fraction"].iloc[0]) >= 0.01,
        "future_range_mse_at_least_three_percent": float(final_targets.loc[final_targets.target.eq("future_range_bps"), "mse_improvement_fraction"].iloc[0]) >= 0.03,
        "all_six_horizons_positive": bool(final_cells.mse_improvement_fraction.gt(0).all()),
        "every_stock_deletion_all_six_horizons_positive": bool(final_deletions.mse_improvement_fraction.gt(0).all()),
        "bootstrap_upper_below_zero_both_targets": bool(final_bootstraps.ci_upper.lt(0).all()),
    }
    final_checks["pass"] = all(final_checks.values())
    if final_checks != decision["final_stack_vs_context"]["magnitude_checks"]:
        raise AssertionError("final magnitude gate mismatch")
    record("exact_final_magnitude_gate", final_checks)
    if not (decision["research_only"] and not decision["live_ordering_enabled"] and decision["order_placement"] == "disabled"):
        raise AssertionError("decision safety boundary failure")
    record("decision_safety_boundary", True)

    result = {
        "audit": "regime_utility_ablation_v1_independent",
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "passed": len(checks),
        "failed": 0,
        "all_passed": True,
        "checks": checks,
    }
    audit_path = ROOT / "independent_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    files = sorted(path for path in ROOT.iterdir() if path.is_file() and path.name != "artifact_manifest.json")
    manifest = {
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in files
        ],
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    (ROOT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
