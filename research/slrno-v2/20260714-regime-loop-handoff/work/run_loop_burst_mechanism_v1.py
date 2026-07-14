"""Causal loop-burst mechanism and continuation model V1.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-loop-burst-mechanism-v1.json"
OOF_SOURCE = Path(
    "/private/tmp/stocker_factor_conditioned_loop_occurrence_v1_20260711/"
    "oof_predictions_2024.parquet"
)
OOF_AUDIT = Path(
    "/private/tmp/stocker_factor_conditioned_loop_occurrence_v1_20260711/"
    "pre_score_audit.json"
)
RUN_SOURCE = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/"
    "train_2024_filtered_runs.csv"
)
CYCLE_SOURCE = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710/"
    "fixed_cycle_shuffled_nulls.csv"
)
FACTOR_CONTRACT = HERE / "contracts/20260711-factor-conditioned-loop-occurrence-v1.json"
FACTOR_RUNNER = HERE / "run_factor_conditioned_loop_occurrence_v1.py"
ROOT = Path("/private/tmp/stocker_loop_burst_mechanism_v1_20260711")

EXPECTED_HASHES = {
    "contract": "29a6e219d3886f7617bc417797c7cc8b7f66f02347d93e866276f40db3c90360",
    "oof_source": "422a7cd24f7e797daef6e5a81756460308bb50a6bf9e2d179dd64abe0b07c6bc",
    "oof_audit": "18d4290c50f749ce6ec5434324afa82cd7bebafcd8be198ed8b1c6c7361eedb1",
    "run_source": "9557298a1a1bc32d47e15a3be31c453f81e019c5a9b1cf76401e4ad4613614d0",
    "cycle_source": "5695f09a7573a110034d251b5abdc40c2f37a11cc7198b196636a624c7d1ad22",
    "factor_contract": "ef8b61bdd4f6671fa64713551a9991f6e4591c3c96bc1ccc324c81b7195bfe7d",
    "factor_runner": "aafb89c6046b752335c7da664c0e8f35062eb66014b86148931fbc92180fa9ff",
}

MONTHS = ("2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12")
VALIDATION_MONTHS = ("2024-10", "2024-11", "2024-12")
TRAINING_MONTHS = {
    "2024-10": ("2024-07", "2024-08", "2024-09"),
    "2024-11": ("2024-07", "2024-08", "2024-09", "2024-10"),
    "2024-12": ("2024-07", "2024-08", "2024-09", "2024-10", "2024-11"),
}
MODELS = (
    "qhistory",
    "qfull9",
    "qoffset_calibration",
    "qburst_global",
    "qburst_orientation",
)
LEARNED_MODELS = MODELS[2:]
PHASE_FEATURES = (
    "log1p_repeat_count",
    "log1p_prior_current_duration",
    "log1p_prior_other_duration",
    "log1p_prior_pair_duration",
    "scheduled_bars_remaining",
)
LOSS_EPSILON = 1e-12
RIDGE_LAMBDA = 0.01
SEED = 20260711
BOOTSTRAP_DRAWS = 4999
SIGN_FLIP_DRAWS = 9999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")


def verify_contract_and_sources() -> tuple[dict[str, Any], dict[str, str]]:
    hashes = {
        "contract": sha256(CONTRACT),
        "oof_source": sha256(OOF_SOURCE),
        "oof_audit": sha256(OOF_AUDIT),
        "run_source": sha256(RUN_SOURCE),
        "cycle_source": sha256(CYCLE_SOURCE),
        "factor_contract": sha256(FACTOR_CONTRACT),
        "factor_runner": sha256(FACTOR_RUNNER),
    }
    if hashes != EXPECTED_HASHES:
        raise AssertionError(f"frozen hash mismatch: {hashes}")
    contract = json.loads(CONTRACT.read_text())
    if not (
        contract["research_only"] is True
        and contract["live_ordering_enabled"] is False
        and contract["order_placement"] == "disabled"
        and contract["economic_edge_claim"] is False
        and contract["population_and_target"]["later_period_paths_permitted"]
        is False
        and contract["population_and_target"][
            "prospective_shadow_read_or_write_permitted"
        ]
        is False
    ):
        raise AssertionError("safety contract changed")
    audit = json.loads(OOF_AUDIT.read_text())
    if not (
        audit["all_passed"] is True
        and audit["check_count"] == 47
    ):
        raise AssertionError("parent OOF audit is not fully passing")
    return contract, hashes


def oof_columns() -> list[str]:
    return [
        "anchor_id",
        "symbol_norm",
        "session_date",
        "start_timestamp",
        "month",
        "cycle_index",
        "cycle_id",
        "cycle",
        "transition_length",
        "state",
        "current_state",
        "target",
        "inverse_compatible_weight",
        "bar_ordinal",
        "entry_minutes",
        "entry_clock_quartile",
        "future_state_1",
        "future_state_2",
        "qhistory",
        "qfull9",
    ]


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oof = pd.read_parquet(OOF_SOURCE, columns=oof_columns())
    oof["month"] = oof["month"].astype(str)
    oof["session_date"] = oof["session_date"].astype(str)
    oof["symbol_norm"] = oof["symbol_norm"].astype(str)
    if len(oof) != 361220 or tuple(sorted(oof["month"].unique())) != MONTHS:
        raise AssertionError("parent OOF population changed")
    oof = oof.loc[oof["transition_length"].eq(2)].copy()
    cycles = pd.read_csv(CYCLE_SOURCE)
    two_state = cycles.loc[cycles["transition_length"].eq(2), "cycle"].tolist()
    if len(two_state) != 13 or set(oof["cycle"].unique()) != set(two_state):
        raise AssertionError("two-state cycle dictionary changed")
    runs = pd.read_csv(
        RUN_SOURCE,
        usecols=[
            "run_id",
            "symbol_norm",
            "session_date",
            "month",
            "state",
            "duration",
            "start_pos",
            "start_timestamp",
        ],
    )
    runs["session_date"] = runs["session_date"].astype(str)
    runs["symbol_norm"] = runs["symbol_norm"].astype(str)
    runs["start_timestamp"] = pd.to_datetime(runs["start_timestamp"], utc=True)
    runs = runs.sort_values(
        ["symbol_norm", "session_date", "run_id"], kind="stable"
    ).reset_index(drop=True)
    runs["session_run_index"] = runs.groupby(
        ["symbol_norm", "session_date"], sort=False
    ).cumcount()
    if len(runs) != 110949:
        raise AssertionError("run population changed")
    return oof, runs, cycles


def other_state(cycle: str, current_state: int) -> int:
    states = {int(value) for value in cycle.split("->")}
    others = states - {int(current_state)}
    if len(states) != 2 or len(others) != 1:
        raise AssertionError(f"not a two-state orientation: {cycle}@{current_state}")
    return next(iter(others))


def sequence_feature(
    states: np.ndarray,
    durations: np.ndarray,
    index: int,
    current_state: int,
    alternate_state: int,
) -> tuple[int, int, int, int, float, float, float]:
    repeat_count = 0
    cursor = index
    while (
        cursor >= 2
        and int(states[cursor - 1]) == alternate_state
        and int(states[cursor - 2]) == current_state
    ):
        repeat_count += 1
        cursor -= 2
    prior_current = int(durations[index - 2]) if repeat_count else 0
    prior_other = int(durations[index - 1]) if repeat_count else 0
    current_duration = int(durations[index])
    next_duration = (
        float(durations[index + 1]) if index + 1 < len(durations) else math.nan
    )
    return_duration = (
        float(durations[index + 2]) if index + 2 < len(durations) else math.nan
    )
    return (
        repeat_count,
        prior_current,
        prior_other,
        current_duration,
        next_duration,
        return_duration,
        float(prior_current + prior_other),
    )


def build_feature_ledger(
    oof: pd.DataFrame, runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_lookup = runs[
        [
            "symbol_norm",
            "session_date",
            "start_timestamp",
            "state",
            "duration",
            "start_pos",
            "session_run_index",
        ]
    ].rename(
        columns={
            "state": "run_state",
            "duration": "realized_current_duration",
            "start_pos": "run_start_pos",
        }
    )
    frame = oof.merge(
        run_lookup,
        on=["symbol_norm", "session_date", "start_timestamp"],
        how="left",
        validate="many_to_one",
    )
    if frame["session_run_index"].isna().any() or not (
        frame["current_state"].to_numpy(int)
        == frame["run_state"].to_numpy(int)
    ).all():
        raise AssertionError("OOF-to-run join failed")
    frame["run_to_session_position_offset"] = (
        frame["run_start_pos"].to_numpy(int)
        - frame["bar_ordinal"].to_numpy(int)
    )
    offset_counts = frame.groupby(
        ["symbol_norm", "session_date"], sort=False
    )["run_to_session_position_offset"].nunique()
    if not offset_counts.eq(1).all():
        raise AssertionError("run global position does not map uniquely to session ordinal")

    sequences = {
        key: (
            group["state"].to_numpy(int),
            group["duration"].to_numpy(int),
        )
        for key, group in runs.groupby(["symbol_norm", "session_date"], sort=False)
    }
    repeats = np.zeros(len(frame), dtype=int)
    prior_current = np.zeros(len(frame), dtype=int)
    prior_other = np.zeros(len(frame), dtype=int)
    realized_current = np.zeros(len(frame), dtype=int)
    realized_next = np.full(len(frame), np.nan)
    realized_return = np.full(len(frame), np.nan)
    prior_pair = np.zeros(len(frame), dtype=float)
    alternate = np.zeros(len(frame), dtype=int)
    for position, row in enumerate(
        frame[
            [
                "symbol_norm",
                "session_date",
                "session_run_index",
                "current_state",
                "cycle",
            ]
        ].itertuples(index=False)
    ):
        states, durations = sequences[(row.symbol_norm, row.session_date)]
        index = int(row.session_run_index)
        alt = other_state(row.cycle, int(row.current_state))
        alternate[position] = alt
        values = sequence_feature(
            states, durations, index, int(row.current_state), alt
        )
        (
            repeats[position],
            prior_current[position],
            prior_other[position],
            realized_current[position],
            realized_next[position],
            realized_return[position],
            prior_pair[position],
        ) = values
    frame["other_state"] = alternate
    frame["repeat_count"] = repeats
    frame["prior_current_duration"] = prior_current
    frame["prior_other_duration"] = prior_other
    frame["prior_pair_duration"] = prior_pair
    frame["prior_durable"] = (prior_current >= 2) & (prior_other >= 2)
    frame["log1p_repeat_count"] = np.log1p(repeats)
    frame["log1p_prior_current_duration"] = np.log1p(prior_current)
    frame["log1p_prior_other_duration"] = np.log1p(prior_other)
    frame["log1p_prior_pair_duration"] = np.log1p(prior_pair)
    frame["scheduled_bars_remaining"] = np.maximum(
        1, 78 - frame["bar_ordinal"].to_numpy(int)
    ).astype(float)
    frame["realized_current_duration"] = realized_current
    frame["realized_next_duration"] = realized_next
    frame["realized_return_duration"] = realized_return
    frame["two_destination_eligible"] = frame["future_state_1"].ne(8) & frame[
        "future_state_2"
    ].ne(8)
    orientation = (
        frame[["cycle_id", "current_state"]]
        .drop_duplicates()
        .sort_values(["cycle_id", "current_state"], kind="stable")
        .reset_index(drop=True)
    )
    orientation["orientation_index"] = np.arange(len(orientation))
    if len(orientation) != 26:
        raise AssertionError("orientation count changed")
    frame = frame.merge(
        orientation,
        on=["cycle_id", "current_state"],
        validate="many_to_one",
    )
    predictor_columns = {
        "repeat_count",
        "prior_current_duration",
        "prior_other_duration",
        "prior_pair_duration",
        *PHASE_FEATURES,
        "orientation_index",
        "qfull9",
    }
    forbidden = {
        "realized_current_duration",
        "realized_next_duration",
        "realized_return_duration",
        "future_state_1",
        "future_state_2",
        "two_destination_eligible",
        "target",
    }
    if predictor_columns & forbidden:
        raise AssertionError("future outcome entered predictor set")
    audit = {
        "rows": len(frame),
        "continuation_rows": int(frame["repeat_count"].ge(1).sum()),
        "continuation_positives": int(
            frame.loc[frame["repeat_count"].ge(1), "target"].sum()
        ),
        "cycles": int(frame["cycle_id"].nunique()),
        "orientations": len(orientation),
        "stocks": int(frame["symbol_norm"].nunique()),
        "sessions": int(frame["session_date"].nunique()),
        "maximum_repeat_count": int(frame["repeat_count"].max()),
        "predictor_columns": sorted(predictor_columns),
        "forbidden_outcome_columns": sorted(forbidden),
        "future_predictor_intersection": sorted(predictor_columns & forbidden),
    }
    return frame, orientation, audit


def weighted_center_scale(
    frame: pd.DataFrame, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = frame[list(PHASE_FEATURES)].to_numpy(float)
    center = np.average(values, axis=0, weights=weights)
    variance = np.average((values - center) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return center, scale


def phase_matrix(
    frame: pd.DataFrame,
    center: np.ndarray,
    scale: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    phase = (frame[list(PHASE_FEATURES)].to_numpy(float) - center) / scale
    intercept = np.ones((len(frame), 1), dtype=float)
    if model == "qoffset_calibration":
        return intercept, np.zeros(1)
    if model == "qburst_global":
        return np.column_stack((intercept, phase)), np.asarray([0.0, *([1.0] * 5)])
    if model != "qburst_orientation":
        raise ValueError(model)
    orientation = frame["orientation_index"].to_numpy(int)
    dummy = np.zeros((len(frame), 25), dtype=float)
    selected = orientation < 25
    dummy[np.arange(len(frame))[selected], orientation[selected]] = 1.0
    interactions = np.hstack([dummy * phase[:, [index]] for index in range(5)])
    matrix = np.column_stack((intercept, phase, dummy, interactions))
    penalties = np.asarray(
        [0.0, *([1.0] * 5), *([4.0] * 25), *([8.0] * 125)]
    )
    if matrix.shape[1] != 156 or len(penalties) != 156:
        raise AssertionError("orientation design width changed")
    return matrix, penalties


def fit_offset_model(
    matrix: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    penalties: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    weights = np.asarray(weights, float)
    weight_sum = weights.sum()

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offset + matrix @ beta
        loss = np.logaddexp(0.0, eta) - y * eta
        value = float(np.sum(weights * loss) / weight_sum)
        value += 0.5 * ridge_lambda * float(np.sum(penalties * beta * beta))
        gradient = matrix.T @ (weights * (expit(eta) - y)) / weight_sum
        gradient += ridge_lambda * penalties * beta
        return value, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(matrix.shape[1]),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or result.nit >= 1000 or not np.isfinite(result.x).all():
        raise AssertionError(f"optimizer failed: {result.message}")
    return result.x, {
        "optimizer_success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "gradient_max_abs": float(np.max(np.abs(result.jac))),
        "feature_width": matrix.shape[1],
        "ridge_lambda": ridge_lambda,
    }


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, float), LOSS_EPSILON, 1 - LOSS_EPSILON)
    return np.log(values / (1 - values))


def fit_predict_models(
    feature_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    continuation = feature_ledger.loc[feature_ledger["repeat_count"].ge(1)].copy()
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    parameters: dict[str, np.ndarray] = {}
    for validation_month in VALIDATION_MONTHS:
        training = continuation.loc[
            continuation["month"].isin(TRAINING_MONTHS[validation_month])
        ]
        validation = continuation.loc[
            continuation["month"].eq(validation_month)
        ].copy()
        if training.empty or validation.empty:
            raise AssertionError("empty causal fold")
        weights = training["inverse_compatible_weight"].to_numpy(float)
        y = training["target"].to_numpy(int)
        center, scale = weighted_center_scale(training, weights)
        train_offset = logit(training["qfull9"].to_numpy(float))
        validation_offset = logit(validation["qfull9"].to_numpy(float))
        parameters[f"{validation_month}__center"] = center
        parameters[f"{validation_month}__scale"] = scale
        for model in LEARNED_MODELS:
            train_matrix, penalties = phase_matrix(training, center, scale, model)
            validation_matrix, validation_penalties = phase_matrix(
                validation, center, scale, model
            )
            if not np.array_equal(penalties, validation_penalties):
                raise AssertionError("penalty vector changed across fold")
            ridge = 0.0 if model == "qoffset_calibration" else RIDGE_LAMBDA
            beta, fit_audit = fit_offset_model(
                train_matrix,
                train_offset,
                y,
                weights,
                penalties,
                ridge,
            )
            probability = np.clip(
                expit(validation_offset + validation_matrix @ beta),
                LOSS_EPSILON,
                1 - LOSS_EPSILON,
            )
            validation[model] = probability
            parameters[f"{validation_month}__{model}__beta"] = beta
            audits.append(
                {
                    "validation_month": validation_month,
                    "training_months_json": json.dumps(
                        TRAINING_MONTHS[validation_month]
                    ),
                    "model": model,
                    "training_rows": len(training),
                    "training_positives": int(y.sum()),
                    "validation_rows": len(validation),
                    "validation_positives": int(validation["target"].sum()),
                    **fit_audit,
                }
            )
        predictions.append(validation)
    result = pd.concat(predictions, ignore_index=True)
    probability_values = result[list(MODELS)].to_numpy(float)
    if not np.isfinite(probability_values).all() or (
        (probability_values <= 0) | (probability_values >= 1)
    ).any():
        raise AssertionError("invalid continuation probabilities")
    return result, pd.DataFrame(audits), parameters


def binary_losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), LOSS_EPSILON, 1 - LOSS_EPSILON)
    return (-(y * np.log(p) + (1 - y) * np.log(1 - p)), (y - p) ** 2)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() else math.nan


def calibration(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray, minimum_rows: int = 250
) -> tuple[float, float, int]:
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    supported: list[tuple[float, float]] = []
    for index in range(10):
        selected = bins == index
        if selected.sum() < minimum_rows or weights[selected].sum() <= 0:
            continue
        error = abs(
            weighted_mean(y[selected], weights[selected])
            - weighted_mean(p[selected], weights[selected])
        )
        supported.append((float(weights[selected].sum()), error))
    if not supported:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in supported)
    return (
        float(sum(weight * error for weight, error in supported) / total),
        float(max(error for _, error in supported)),
        len(supported),
    )


def metric_row(frame: pd.DataFrame, model: str, surface: str) -> dict[str, Any]:
    weights = (
        frame["inverse_compatible_weight"].to_numpy(float)
        if surface == "inverse_compatible"
        else np.ones(len(frame))
    )
    y = frame["target"].to_numpy(int)
    p = frame[model].to_numpy(float)
    ll, br = binary_losses(y, p)
    ece, maximum, bins = calibration(y, p, weights)
    return {
        "rows": len(frame),
        "positives": int(y.sum()),
        "weight_sum": float(weights.sum()),
        "log_loss": weighted_mean(ll, weights),
        "brier": weighted_mean(br, weights),
        "ece": ece,
        "maximum_supported_bin_error": maximum,
        "supported_bins": bins,
    }


def daily_difference(
    frame: pd.DataFrame, candidate: str, baseline: str, endpoint: str
) -> np.ndarray:
    y = frame["target"].to_numpy(int)
    candidate_loss = binary_losses(y, frame[candidate].to_numpy(float))[
        0 if endpoint == "log_loss" else 1
    ]
    baseline_loss = binary_losses(y, frame[baseline].to_numpy(float))[
        0 if endpoint == "log_loss" else 1
    ]
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    grouped = pd.DataFrame(
        {
            "date": frame["session_date"],
            "weighted": (candidate_loss - baseline_loss) * weights,
            "weight": weights,
        }
    ).groupby("date", sort=True).sum()
    return (grouped["weighted"] / grouped["weight"]).to_numpy(float)


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    blocks = np.asarray(
        [
            values[index : index + 5].mean()
            for index in range(0, len(values), 5)
            if len(values[index : index + 5]) == 5
        ]
    )
    sampled = np.random.default_rng(seed).choice(
        blocks, size=(BOOTSTRAP_DRAWS, len(blocks)), replace=True
    ).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip(values: np.ndarray, seed: int) -> float:
    null = (
        np.random.default_rng(seed).choice(
            np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values))
        )
        @ values
    ) / len(values)
    return float((1 + np.sum(null <= values.mean())) / (SIGN_FLIP_DRAWS + 1))


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    ordered = sorted(
        output.index.tolist(), key=lambda index: output.loc[index, "p_value"]
    )
    running = 0.0
    for rank, index in enumerate(ordered, start=1):
        adjusted = min(
            1.0,
            max(running, (len(ordered) - rank + 1) * output.loc[index, "p_value"]),
        )
        running = adjusted
        output.loc[index, "holm_adjusted_p"] = adjusted
        output.loc[index, "holm_pass"] = adjusted <= 0.05
        output.loc[index, "holm_rank"] = rank
        output.loc[index, "family_size"] = len(ordered)
    output["holm_rank"] = output["holm_rank"].astype(int)
    output["family_size"] = output["family_size"].astype(int)
    return output


def recurrence_diagnostics(
    feature_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    primary = feature_ledger.loc[
        feature_ledger["month"].isin(VALIDATION_MONTHS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (cycle_id, state), selected in primary.groupby(
        ["cycle_id", "current_state"], sort=True
    ):
        recurrent = selected["repeat_count"].ge(1)
        weights = selected["inverse_compatible_weight"].to_numpy(float)
        y = selected["target"].to_numpy(int)
        recurrent_rate = weighted_mean(y[recurrent], weights[recurrent])
        initiation_rate = weighted_mean(y[~recurrent], weights[~recurrent])
        rows.append(
            {
                "cycle_id": cycle_id,
                "current_state": int(state),
                "rows": len(selected),
                "positives": int(y.sum()),
                "recurrent_rows": int(recurrent.sum()),
                "recurrent_positives": int(y[recurrent].sum()),
                "initiation_rows": int((~recurrent).sum()),
                "initiation_positives": int(y[~recurrent].sum()),
                "recurrent_rate": recurrent_rate,
                "initiation_rate": initiation_rate,
                "rate_difference": recurrent_rate - initiation_rate,
                "rate_ratio": recurrent_rate / initiation_rate
                if initiation_rate > 0
                else math.inf,
                "supported": int(recurrent.sum()) >= 100
                and int(y[recurrent].sum()) >= 20,
            }
        )
    orientation = pd.DataFrame(rows)
    recurrent = primary["repeat_count"].ge(1).to_numpy()
    weights = primary["inverse_compatible_weight"].to_numpy(float)
    y = primary["target"].to_numpy(int)
    recurrent_rate = weighted_mean(y[recurrent], weights[recurrent])
    initiation_rate = weighted_mean(y[~recurrent], weights[~recurrent])
    daily_rows: list[dict[str, Any]] = []
    for date, selected in primary.groupby("session_date", sort=True):
        mask = selected["repeat_count"].ge(1).to_numpy()
        if not mask.any() or (~mask).sum() == 0:
            continue
        day_weights = selected["inverse_compatible_weight"].to_numpy(float)
        day_y = selected["target"].to_numpy(int)
        daily_rows.append(
            {
                "session_date": date,
                "risk_difference": weighted_mean(day_y[mask], day_weights[mask])
                - weighted_mean(day_y[~mask], day_weights[~mask]),
            }
        )
    daily_frame = pd.DataFrame(daily_rows)
    lower, upper = bootstrap_interval(
        daily_frame["risk_difference"].to_numpy(float), SEED + 7000
    )
    summary = {
        "rows": len(primary),
        "recurrent_rows": int(recurrent.sum()),
        "recurrent_rate": recurrent_rate,
        "initiation_rate": initiation_rate,
        "pooled_rate_difference": recurrent_rate - initiation_rate,
        "pooled_rate_ratio": recurrent_rate / initiation_rate,
        "supported_orientations": int(orientation["supported"].sum()),
        "supported_orientations_ratio_above_one": int(
            (orientation.loc[orientation["supported"], "rate_ratio"] > 1).sum()
        ),
        "daily_sessions": len(daily_frame),
        "bootstrap_lower": lower,
        "bootstrap_upper": upper,
    }
    return orientation, summary, daily_frame


def session_boundary_diagnostic(feature_ledger: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = feature_ledger.loc[
        feature_ledger["month"].isin(VALIDATION_MONTHS)
        & feature_ledger["cycle_id"].eq("cycle_13")
        & feature_ledger["current_state"].eq(5)
    ].copy()
    selected["boundary_before_exit"] = selected["future_state_1"].eq(8)
    selected["boundary_after_other"] = selected["future_state_1"].eq(7) & selected[
        "future_state_2"
    ].eq(8)
    rows: list[dict[str, Any]] = []
    for clock, group in selected.groupby("entry_clock_quartile", sort=True):
        eligible = group["two_destination_eligible"]
        rows.append(
            {
                "entry_clock_quartile": int(clock),
                "rows": len(group),
                "positives": int(group["target"].sum()),
                "rate": float(group["target"].mean()),
                "boundary_before_exit": int(group["boundary_before_exit"].sum()),
                "boundary_after_other": int(group["boundary_after_other"].sum()),
                "boundary_fraction": float(
                    (group["boundary_before_exit"] | group["boundary_after_other"]).mean()
                ),
                "eligible_rows": int(eligible.sum()),
                "eligible_positives": int(group.loc[eligible, "target"].sum()),
                "eligible_rate": float(group.loc[eligible, "target"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    indexed = frame.set_index("entry_clock_quartile")
    summary = {
        "cycle_13_state_5_rows": len(selected),
        "late_boundary_fraction": float(indexed.loc[3, "boundary_fraction"]),
        "mid_eligible_rate": float(indexed.loc[1, "eligible_rate"]),
        "late_eligible_rate": float(indexed.loc[3, "eligible_rate"]),
        "mid_minus_late_eligible_rate": float(
            indexed.loc[1, "eligible_rate"] - indexed.loc[3, "eligible_rate"]
        ),
    }
    return frame, summary


def chatter_diagnostic(feature_ledger: pd.DataFrame) -> dict[str, Any]:
    selected = feature_ledger.loc[
        feature_ledger["month"].isin(VALIDATION_MONTHS)
        & feature_ledger["cycle_id"].eq("cycle_13")
        & feature_ledger["current_state"].eq(5)
        & feature_ledger["target"].eq(1)
    ].copy()
    durations = selected[
        [
            "realized_current_duration",
            "realized_next_duration",
            "realized_return_duration",
        ]
    ]
    if durations.isna().any().any():
        raise AssertionError("positive cycle-13 loop lacks realized duration")
    total = durations.sum(axis=1)
    return {
        "rows": len(selected),
        "all_three_legs_at_least_two_fraction": float((durations >= 2).all(axis=1).mean()),
        "any_one_bar_leg_fraction": float((durations == 1).any(axis=1).mean()),
        "total_duration_median_bars": float(total.median()),
        "total_duration_mean_bars": float(total.mean()),
        "total_duration_p25_bars": float(total.quantile(0.25)),
        "total_duration_p75_bars": float(total.quantile(0.75)),
    }


def repeat_count_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["repeat_bucket"] = np.minimum(frame["repeat_count"].to_numpy(int), 4)
    rows: list[dict[str, Any]] = []
    for bucket, selected in frame.groupby("repeat_bucket", sort=True):
        weights = selected["inverse_compatible_weight"].to_numpy(float)
        rows.append(
            {
                "repeat_bucket": "4+" if int(bucket) == 4 else str(int(bucket)),
                "rows": len(selected),
                "positives": int(selected["target"].sum()),
                "event_rate": weighted_mean(selected["target"].to_numpy(int), weights),
                "mean_scheduled_bars_remaining": weighted_mean(
                    selected["scheduled_bars_remaining"].to_numpy(float), weights
                ),
                "mean_prior_pair_duration": weighted_mean(
                    selected["prior_pair_duration"].to_numpy(float), weights
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    feature_ledger: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, Any]:
    pooled_rows: list[dict[str, Any]] = []
    for surface in ("inverse_compatible", "unweighted"):
        for model in MODELS:
            pooled_rows.append(
                {"surface": surface, "model": model, **metric_row(predictions, model, surface)}
            )
    pooled = pd.DataFrame(pooled_rows)
    primary = pooled.loc[pooled["surface"].eq("inverse_compatible")].set_index("model")

    comparison_rows: list[dict[str, Any]] = []
    multiplicity_rows: list[dict[str, Any]] = []
    for comparison_index, baseline in enumerate(("qfull9", "qoffset_calibration")):
        for endpoint_index, endpoint in enumerate(("log_loss", "brier")):
            values = daily_difference(
                predictions, "qburst_orientation", baseline, endpoint
            )
            seed = SEED + comparison_index * 100 + endpoint_index
            lower, upper = bootstrap_interval(values, seed)
            p_value = sign_flip(values, seed + 10)
            comparison_rows.append(
                {
                    "candidate": "qburst_orientation",
                    "baseline": baseline,
                    "endpoint": endpoint,
                    "daily_sessions": len(values),
                    "mean_difference": float(values.mean()),
                    "bootstrap_lower": lower,
                    "bootstrap_upper": upper,
                    "p_value": p_value,
                }
            )
            multiplicity_rows.append(
                {"baseline": baseline, "endpoint": endpoint, "p_value": p_value}
            )
    comparisons = pd.DataFrame(comparison_rows)
    multiplicity = holm(pd.DataFrame(multiplicity_rows))

    temporal_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    for month in VALIDATION_MONTHS:
        selected = predictions.loc[predictions["month"].eq(month)]
        for model in MODELS:
            temporal_rows.append(
                {"month": month, "model": model, **metric_row(selected, model, "inverse_compatible")}
            )
    for symbol in sorted(predictions["symbol_norm"].unique()):
        selected = predictions.loc[predictions["symbol_norm"].ne(symbol)]
        for model in ("qoffset_calibration", "qburst_orientation"):
            stock_rows.append(
                {"deleted_symbol": symbol, "model": model, **metric_row(selected, model, "inverse_compatible")}
            )
    for (cycle_id, state), selected in predictions.groupby(
        ["cycle_id", "current_state"], sort=True
    ):
        supported = len(selected) >= 100 and int(selected["target"].sum()) >= 20
        for model in ("qoffset_calibration", "qburst_orientation"):
            orientation_rows.append(
                {
                    "cycle_id": cycle_id,
                    "current_state": int(state),
                    "supported": supported,
                    "model": model,
                    **metric_row(selected, model, "inverse_compatible"),
                }
            )
    temporal = pd.DataFrame(temporal_rows)
    stocks = pd.DataFrame(stock_rows)
    orientations = pd.DataFrame(orientation_rows)
    durable = predictions.loc[predictions["prior_durable"]]
    durable_rows = [
        {"model": model, **metric_row(durable, model, "inverse_compatible")}
        for model in ("qoffset_calibration", "qburst_orientation")
    ]
    durable_frame = pd.DataFrame(durable_rows)

    recurrence_orientation, recurrence_summary, recurrence_daily = recurrence_diagnostics(
        feature_ledger
    )
    boundary_frame, boundary_summary = session_boundary_diagnostic(feature_ledger)
    chatter = chatter_diagnostic(feature_ledger)
    repeat_counts = repeat_count_table(predictions)

    temporal_index = temporal.set_index(["month", "model"])
    stock_index = stocks.set_index(["deleted_symbol", "model"])
    orientation_index = orientations.set_index(["cycle_id", "current_state", "model"])
    supported_keys = (
        orientations.loc[orientations["supported"], ["cycle_id", "current_state"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    orientation_differences = []
    for cycle_id, state in supported_keys:
        candidate = orientation_index.loc[(cycle_id, state, "qburst_orientation")]
        baseline = orientation_index.loc[(cycle_id, state, "qoffset_calibration")]
        orientation_differences.append(candidate["log_loss"] - baseline["log_loss"])
    orientation_differences = np.asarray(orientation_differences, float)
    durable_index = durable_frame.set_index("model")
    primary_checks = {
        "minimum_pooled_log_loss_improvement": bool(
            (primary.loc["qoffset_calibration", "log_loss"] - primary.loc["qburst_orientation", "log_loss"])
            / primary.loc["qoffset_calibration", "log_loss"]
            >= 0.005
        ),
        "pooled_brier": bool(
            primary.loc["qburst_orientation", "brier"]
            <= primary.loc["qoffset_calibration", "brier"]
        ),
        "bootstrap_both_baselines": bool((comparisons["bootstrap_upper"] <= 0).all()),
        "Holm_all_four": bool(multiplicity["holm_pass"].all()),
        "every_month": bool(
            all(
                temporal_index.loc[(month, "qburst_orientation"), endpoint]
                <= temporal_index.loc[(month, "qoffset_calibration"), endpoint]
                for month in VALIDATION_MONTHS
                for endpoint in ("log_loss", "brier")
            )
        ),
        "every_stock_deletion": bool(
            all(
                stock_index.loc[(symbol, "qburst_orientation"), endpoint]
                <= stock_index.loc[(symbol, "qoffset_calibration"), endpoint]
                for symbol in sorted(predictions["symbol_norm"].unique())
                for endpoint in ("log_loss", "brier")
            )
        ),
        "orientation_count": bool((orientation_differences < 0).sum() >= 20),
        "orientation_maximum_harm": bool(orientation_differences.max() <= 0.005),
        "calibration": bool(
            primary.loc["qburst_orientation", "ece"]
            <= primary.loc["qoffset_calibration", "ece"]
            and primary.loc["qburst_orientation", "maximum_supported_bin_error"] <= 0.03
        ),
        "durable_prior": bool(
            durable_index.loc["qburst_orientation", "log_loss"]
            <= durable_index.loc["qoffset_calibration", "log_loss"]
            and durable_index.loc["qburst_orientation", "brier"]
            <= durable_index.loc["qoffset_calibration", "brier"]
        ),
    }
    model_gate = {
        "pass": bool(all(primary_checks.values())),
        "checks": primary_checks,
        "pooled_relative_log_loss_improvement_vs_offset": float(
            (primary.loc["qoffset_calibration", "log_loss"] - primary.loc["qburst_orientation", "log_loss"])
            / primary.loc["qoffset_calibration", "log_loss"]
        ),
        "pooled_brier_difference_vs_offset": float(
            primary.loc["qburst_orientation", "brier"]
            - primary.loc["qoffset_calibration", "brier"]
        ),
        "supported_orientations": len(orientation_differences),
        "orientations_with_lower_log_loss": int((orientation_differences < 0).sum()),
        "maximum_orientation_log_loss_harm": float(orientation_differences.max()),
        "durable_rows": len(durable),
        "durable_positives": int(durable["target"].sum()),
    }
    mechanism_checks = {
        "H1_pooled_rate_ratio": recurrence_summary["pooled_rate_ratio"] >= 1.5,
        "H1_orientation_count": recurrence_summary[
            "supported_orientations_ratio_above_one"
        ]
        >= 24,
        "H1_bootstrap": recurrence_summary["bootstrap_lower"] > 0,
        "H2_late_boundary": boundary_summary["late_boundary_fraction"] >= 0.25,
        "H2_mid_late_eligible": boundary_summary[
            "mid_minus_late_eligible_rate"
        ]
        >= 0.03,
        "H4_durable_support": len(durable) >= 2000
        and int(durable["target"].sum()) >= 400,
        "H4_durable_model": primary_checks["durable_prior"],
        "H5_durable_realized_fraction": chatter[
            "all_three_legs_at_least_two_fraction"
        ]
        >= 0.35,
        "H5_duration_median": chatter["total_duration_median_bars"] >= 10,
    }
    mechanism_gate = {
        "pass": bool(all(mechanism_checks.values())),
        "checks": mechanism_checks,
        "recurrence": recurrence_summary,
        "session_boundary": boundary_summary,
        "chatter": chatter,
    }
    if model_gate["pass"]:
        label = "development_burst_continuation_candidate_pending_unseen_validation"
    elif mechanism_gate["pass"]:
        label = "burst_mechanism_retained_as_post_inspection_hypothesis_without_forecaster"
    else:
        label = "burst_continuation_model_rejected_or_unconfirmed"
    decision = {
        "label": label,
        "primary_model_pass": model_gate["pass"],
        "mechanism_gate_pass": mechanism_gate["pass"],
        "retained_forecaster": "qburst_orientation" if model_gate["pass"] else None,
        "named_loop_good_or_high_promoted": False,
        "later_period_scoring_performed": False,
        "prospective_validated": False,
        "economic_edge_claim": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    return {
        "pooled": pooled,
        "comparisons": comparisons,
        "multiplicity": multiplicity,
        "temporal": temporal,
        "stocks": stocks,
        "orientations": orientations,
        "durable": durable_frame,
        "recurrence_orientation": recurrence_orientation,
        "recurrence_daily": recurrence_daily,
        "boundary": boundary_frame,
        "repeat_counts": repeat_counts,
        "model_gate": model_gate,
        "mechanism_gate": mechanism_gate,
        "decision": decision,
    }


def artifact_manifest(root: Path, names: Iterable[str]) -> dict[str, Any]:
    return {
        "files": {
            name: {
                "sha256": sha256(root / name),
                "bytes": (root / name).stat().st_size,
            }
            for name in sorted(names)
        }
    }


def run() -> None:
    contract, hashes = verify_contract_and_sources()
    oof, runs, _ = load_sources()
    feature_ledger, orientation, feature_audit = build_feature_ledger(oof, runs)
    predictions, fit_audit, parameters = fit_predict_models(feature_ledger)
    results = evaluate(feature_ledger, predictions)
    ROOT.mkdir(parents=True, exist_ok=False)
    feature_ledger.to_parquet(ROOT / "burst_feature_ledger_2024_jul_dec.parquet", index=False)
    predictions.to_parquet(ROOT / "continuation_predictions_2024_oct_dec.parquet", index=False)
    orientation.to_csv(ROOT / "orientation_dictionary.csv", index=False)
    fit_audit.to_csv(ROOT / "fit_audit.csv", index=False)
    np.savez_compressed(ROOT / "model_parameters.npz", **parameters)
    results["pooled"].to_csv(ROOT / "pooled_metrics.csv", index=False)
    results["comparisons"].to_csv(ROOT / "comparisons.csv", index=False)
    results["multiplicity"].to_csv(ROOT / "multiplicity.csv", index=False)
    results["temporal"].to_csv(ROOT / "temporal_slices.csv", index=False)
    results["stocks"].to_csv(ROOT / "stock_deletions.csv", index=False)
    results["orientations"].to_csv(ROOT / "orientation_slices.csv", index=False)
    results["durable"].to_csv(ROOT / "durable_prior_slice.csv", index=False)
    results["recurrence_orientation"].to_csv(ROOT / "recurrence_orientations.csv", index=False)
    results["recurrence_daily"].to_csv(ROOT / "recurrence_daily.csv", index=False)
    results["boundary"].to_csv(ROOT / "cycle13_session_boundary.csv", index=False)
    results["repeat_counts"].to_csv(ROOT / "repeat_count_diagnostic.csv", index=False)
    write_json(ROOT / "feature_audit.json", feature_audit)
    write_json(ROOT / "model_gate.json", results["model_gate"])
    write_json(ROOT / "mechanism_gate.json", results["mechanism_gate"])
    write_json(ROOT / "decision.json", results["decision"])
    summary = {
        "contract_id": contract["contract_id"],
        "contract_sha256": hashes["contract"],
        "runner_sha256": sha256(Path(__file__)),
        "scientific_status": contract["scientific_status"],
        "source_hashes": hashes,
        "feature_audit": feature_audit,
        "prediction_rows": len(predictions),
        "prediction_positives": int(predictions["target"].sum()),
        "validation_months": list(VALIDATION_MONTHS),
        "fit_count": len(fit_audit),
        "model_gate": results["model_gate"],
        "mechanism_gate": results["mechanism_gate"],
        "decision": results["decision"],
        "direct_volume_fields_used": [],
        "volume_label": "historical_volume_not_used",
        "direction_or_signed_return_used": False,
        "price_consequence_used": False,
        "later_period_scoring_performed": False,
        "prospective_shadow_read_or_write_performed": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(ROOT / "summary.json", summary)
    names = [
        "burst_feature_ledger_2024_jul_dec.parquet",
        "continuation_predictions_2024_oct_dec.parquet",
        "orientation_dictionary.csv",
        "fit_audit.csv",
        "model_parameters.npz",
        "pooled_metrics.csv",
        "comparisons.csv",
        "multiplicity.csv",
        "temporal_slices.csv",
        "stock_deletions.csv",
        "orientation_slices.csv",
        "durable_prior_slice.csv",
        "recurrence_orientations.csv",
        "recurrence_daily.csv",
        "cycle13_session_boundary.csv",
        "repeat_count_diagnostic.csv",
        "feature_audit.json",
        "model_gate.json",
        "mechanism_gate.json",
        "decision.json",
        "summary.json",
    ]
    write_json(ROOT / "artifact_manifest.json", artifact_manifest(ROOT, names))
    print(json.dumps(safe(summary), indent=2, sort_keys=True))


def self_test() -> None:
    states = np.asarray([5, 7, 5, 7, 5, 6])
    durations = np.asarray([3, 4, 5, 6, 7, 8])
    values = sequence_feature(states, durations, 4, 5, 7)
    assert values[:4] == (2, 5, 6, 7)
    assert values[4] == 8
    assert np.isnan(values[5])
    assert values[6] == 11
    frame = pd.DataFrame(
        {
            **{feature: [0.0, 1.0] for feature in PHASE_FEATURES},
            "orientation_index": [0, 25],
        }
    )
    center = np.zeros(5)
    scale = np.ones(5)
    assert phase_matrix(frame, center, scale, "qoffset_calibration")[0].shape == (2, 1)
    assert phase_matrix(frame, center, scale, "qburst_global")[0].shape == (2, 6)
    assert phase_matrix(frame, center, scale, "qburst_orientation")[0].shape == (2, 156)
    ll, br = binary_losses(np.asarray([0, 1]), np.asarray([0.25, 0.75]))
    assert np.allclose(ll, -np.log(0.75))
    assert np.allclose(br, 0.0625)


if __name__ == "__main__":
    self_test()
    run()
