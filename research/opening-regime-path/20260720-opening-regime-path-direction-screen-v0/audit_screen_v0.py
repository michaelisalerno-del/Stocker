#!/usr/bin/env python3
"""Independent audit for Opening Regime-Path Direction Screen V0.

This auditor deliberately does not import the experiment runner or its reusable
screen module.  It reconstructs targets, topology, interactions, predictions,
metrics, resampling, the structure null, support, and the decision from artifacts
and bounded provider rows.
"""

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.

from __future__ import annotations

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from stocker_research.regime_gap_segmentation_v2 import causal_segment_groups
from stocker_research.regime_panel_v2 import EMISSION_FEATURES
from stocker_research.regime_validity_v2 import (
    EmissionPreprocessing,
    SemiMarkovParameters,
    causal_filter_summary,
    gaussian_log_emissions,
    transform_emissions,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
START = pd.Timestamp("2024-01-01T00:00:00Z")
DEVELOPMENT_END = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
SAFE_END = PROTECTED_START - pd.Timedelta(microseconds=1)
DECISION_ORDINALS = (6, 12)
STATE_COUNT = 8
BOOTSTRAP_DRAWS = 300
NULL_DRAWS = 100
BOOTSTRAP_SEED = 20260720
NULL_SEED = 20260721
EXPECTED_MODEL_HASH = "4fc1a02dce9ac2311dabaeb4623a559d37286dfe58baffef53828cc7415a3425"

SLRNO_WORK = REPO_ROOT / "research" / "slrno-v2" / "20260714-regime-loop-handoff" / "work"
REFIT_DIR = SLRNO_WORK / "artifacts" / "20260719-right-censored-regime-refit-v2" / "primary"
PARAMETERS_PATH = REFIT_DIR / "full_refit_parameters.npz"
PREPROCESSING_PATH = REFIT_DIR / "full_refit_preprocessing.csv"
POSTERIOR_AUDIT_PATH = REFIT_DIR / "posterior_audit_input.parquet"

SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "feasibility_screen": True,
    "representation_specific": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
}

CONTEXT_SYMBOLS = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "AXTI",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "OKLO",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
DECISION_SYMBOLS = tuple(symbol for symbol in CONTEXT_SYMBOLS if symbol not in {"AXTI", "OKLO"})
TARGETS = (
    "large_remaining_move",
    "up_given_large_move",
    "remaining_direction_up",
)
FORBIDDEN_FRAGMENTS = (
    "future_state",
    "future_run",
    "future_closure",
    "future_loop",
    "payoff",
    "profitable_loop",
    "exact_loop",
    "economic_history",
    "outcome_selected",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value), encoding="utf-8")


def _arrow_hash(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _provider_path(root: Path, symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return root / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def _bounded_source(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        filters=[
            ("timestamp", ">=", START.to_pydatetime()),
            ("timestamp", "<=", SAFE_END.to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].lt(START).any() or frame["timestamp"].ge(PROTECTED_START).any():
        raise AssertionError("bounded provider read admitted a protected or early row")
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _regular_grid(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    output = frame.copy()
    local = output["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    keep = (
        minute.ge(570)
        & minute.lt(960)
        & ((minute - 570) % 5).eq(0)
        & local.dt.second.eq(0)
        & local.dt.microsecond.eq(0)
    )
    output = output.loc[keep].copy()
    local = output["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    output["symbol"] = symbol
    output["session"] = local.dt.strftime("%Y-%m-%d")
    output["bar_ordinal"] = ((minute - 570) // 5).astype(int)
    return output.sort_values(["session", "bar_ordinal"], kind="mergesort")


def _reproduce_frozen_posterior() -> dict[str, Any]:
    preprocessing_frame = pd.read_csv(PREPROCESSING_PATH)
    preprocessing = EmissionPreprocessing(
        feature_names=tuple(preprocessing_frame["feature"].astype(str)),
        medians=preprocessing_frame["imputer_median"].to_numpy(dtype=float),
        centers=preprocessing_frame["scaler_center"].to_numpy(dtype=float),
        scales=preprocessing_frame["scaler_scale"].to_numpy(dtype=float),
    )
    preprocessing.validate()
    if preprocessing.feature_names != tuple(EMISSION_FEATURES):
        raise AssertionError("frozen preprocessing order differs")
    with np.load(PARAMETERS_PATH) as stored:
        parameters = SemiMarkovParameters(
            means=np.asarray(stored["means"]).copy(),
            variances=np.asarray(stored["variances"]).copy(),
            duration_hazard=np.asarray(stored["duration_hazard"]).copy(),
            transitions=np.asarray(stored["transitions"]).copy(),
            initial=np.asarray(stored["initial"]).copy(),
            occupancy=np.asarray(stored["occupancy"]).copy(),
        )
        model_hash = str(np.asarray(stored["state_model_hash"]).item())
    parameters.validate()
    if model_hash != EXPECTED_MODEL_HASH:
        raise AssertionError("frozen state model hash differs")
    frame = pd.read_parquet(POSTERIOR_AUDIT_PATH)
    summary = causal_filter_summary(
        gaussian_log_emissions(transform_emissions(frame, preprocessing), parameters),
        groups=causal_segment_groups(frame),
        model=parameters.as_dict(),
    )
    expected = frame[[f"state_probability_{state}" for state in range(STATE_COUNT)]].to_numpy(
        dtype=float
    )
    return {
        "rows": len(frame),
        "hard_state_agreement": float(
            np.mean(summary.hard_states == frame["state"].to_numpy(dtype=int))
        ),
        "maximum_probability_absolute_error": float(
            np.max(np.abs(summary.state_probabilities - expected))
        ),
        "maximum_expected_age_absolute_error": float(
            np.max(np.abs(summary.expected_age - frame["age"].to_numpy(dtype=float)))
        ),
        "maximum_entropy_absolute_error": float(
            np.max(
                np.abs(summary.posterior_entropy - frame["posterior_entropy"].to_numpy(dtype=float))
            )
        ),
    }


def _runs(states: Sequence[int]) -> tuple[list[int], list[int]]:
    values = [int(value) for value in states]
    if not values:
        raise AssertionError("empty state path")
    run_states = [values[0]]
    run_durations = [1]
    for value in values[1:]:
        if value == run_states[-1]:
            run_durations[-1] += 1
        else:
            run_states.append(value)
            run_durations.append(1)
    return run_states, run_durations


def _topology(states: Sequence[int]) -> dict[str, float]:
    values = [int(value) for value in states]
    run_states, durations = _runs(values)
    transitions = len(run_states) - 1
    revisits = sum(run_states[index] in run_states[:index] for index in range(1, len(run_states)))
    returns = sum(value == run_states[0] for value in run_states[1:])
    two = sum(
        run_states[index] == run_states[index + 2] and run_states[index] != run_states[index + 1]
        for index in range(max(0, len(run_states) - 2))
    )
    three = sum(
        run_states[index] == run_states[index + 3] and len(set(run_states[index : index + 3])) == 3
        for index in range(max(0, len(run_states) - 3))
    )
    recent_two = bool(
        len(run_states) >= 3
        and run_states[-3] == run_states[-1]
        and run_states[-3] != run_states[-2]
    )
    recent_three = bool(
        len(run_states) >= 4
        and run_states[-4] == run_states[-1]
        and len(set(run_states[-4:-1])) == 3
    )
    alternations = sum(
        run_states[index] == run_states[index + 2] for index in range(max(0, len(run_states) - 2))
    )
    completed = durations[:-1]
    _, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    occupancy = counts / len(values)
    return {
        "opening_transition_count": float(transitions),
        "opening_unique_state_count": float(len(set(run_states))),
        "opening_state_revisit_count": float(revisits),
        "opening_return_to_origin_count": float(returns),
        "opening_two_state_closure_count": float(two),
        "opening_three_state_closure_count": float(three),
        "opening_any_short_closure": float(two + three > 0),
        "opening_most_recent_path_was_closure": float(recent_two or recent_three),
        "opening_alternation_ratio": float(alternations / max(1, len(run_states) - 2)),
        "opening_transition_rate": float(transitions / len(values)),
        "opening_mean_completed_run_duration": (float(np.mean(completed)) if completed else 0.0),
        "opening_maximum_completed_run_duration": (float(max(completed)) if completed else 0.0),
        "opening_minimum_completed_run_duration": (float(min(completed)) if completed else 0.0),
        "opening_most_recent_completed_run_duration": (float(completed[-1]) if completed else 0.0),
        "opening_time_since_latest_transition": (
            float(durations[-1] - 1) if transitions else float(len(values))
        ),
        "opening_state_occupancy_entropy": float(-np.sum(occupancy * np.log(occupancy))),
        "opening_largest_state_occupancy_fraction": float(occupancy.max()),
        "current_state_age": float(durations[-1]),
    }


def _target_population(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "up_given_large_move":
        return frame.loc[frame["large_remaining_move"].eq(1)].copy()
    return frame.copy()


def _manual_probability(model: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    names = [str(name) for name in model["feature_names"]]
    values = frame[names].to_numpy(dtype=float)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    linear = float(model["intercept"]) + ((values - means) / scales) @ coefficients
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _calibration(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return math.nan, math.nan
    predictor = _logit(probabilities)
    if float(np.std(predictor)) < 1e-12:
        return float(_logit(np.asarray([np.mean(labels)]))[0]), 0.0

    def calculation(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + parameters[1] * predictor
        prediction = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        loss = float(
            -np.sum(
                labels * np.log(np.clip(prediction, 1e-12, 1.0))
                + (1 - labels) * np.log(np.clip(1.0 - prediction, 1e-12, 1.0))
            )
        )
        gradient = np.asarray(
            [np.sum(prediction - labels), np.sum((prediction - labels) * predictor)]
        )
        return loss, gradient

    result = minimize(
        lambda parameters: calculation(parameters)[0],
        np.asarray([0.0, 1.0]),
        jac=lambda parameters: calculation(parameters)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    return float(result.x[0]), float(result.x[1])


def _calibration_bins(frame: pd.DataFrame, target: str, model: str, scope: str) -> pd.DataFrame:
    population = _target_population(frame, target)
    labels = population[target].to_numpy(dtype=int)
    probabilities = population[f"p__{target}__{model}"].to_numpy(dtype=float)
    numbers = np.minimum((np.clip(probabilities, 0.0, 1.0) * 10).astype(int), 9)
    rows: list[dict[str, Any]] = []
    for number in range(10):
        mask = numbers == number
        rows.append(
            {
                "scope": scope,
                "target": target,
                "model": model,
                "bin": number + 1,
                "probability_lower": number / 10.0,
                "probability_upper": (number + 1) / 10.0,
                "row_count": int(mask.sum()),
                "mean_predicted_probability": (
                    float(np.mean(probabilities[mask])) if mask.any() else math.nan
                ),
                "observed_rate": float(np.mean(labels[mask])) if mask.any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _metric(frame: pd.DataFrame, target: str, model: str, scope: str) -> dict[str, Any]:
    population = _target_population(frame, target)
    labels = population[target].to_numpy(dtype=int)
    probabilities = np.clip(
        population[f"p__{target}__{model}"].to_numpy(dtype=float),
        1e-12,
        1.0 - 1e-12,
    )
    intercept, slope = _calibration(labels, probabilities)
    bins = _calibration_bins(population, target, model, scope)
    populated = bins.loc[bins["row_count"].gt(0)]
    ece = float(
        np.sum(
            populated["row_count"]
            / len(population)
            * (populated["observed_rate"] - populated["mean_predicted_probability"]).abs()
        )
    )
    return {
        "scope": scope,
        "target": target,
        "model": model,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "auc": (
            float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else math.nan
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
        "base_rate": float(np.mean(labels)),
        "row_count": len(population),
        "session_count": int(population["session"].nunique()),
        "stock_count": int(population["symbol"].nunique()),
    }


def _all_metrics(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    months: list[dict[str, Any]] = []
    bins: list[pd.DataFrame] = []
    for target in TARGETS:
        for model in ("M0", "M1", "M2", "M3"):
            pooled.append(_metric(assessment, target, model, "pooled"))
            bins.append(_calibration_bins(assessment, target, model, "pooled"))
            for ordinal in DECISION_ORDINALS:
                subset = assessment.loc[assessment["decision_ordinal"].eq(ordinal)]
                checkpoints.append(_metric(subset, target, model, f"checkpoint_{ordinal}"))
            for month in sorted(assessment["year_month"].astype(str).unique()):
                subset = assessment.loc[assessment["year_month"].astype(str).eq(month)]
                months.append(_metric(subset, target, model, month))
    return (
        pd.DataFrame(pooled),
        pd.DataFrame(checkpoints),
        pd.DataFrame(months),
        pd.concat(bins, ignore_index=True),
    )


def _selection(assessment: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slate_id, slate in assessment.groupby("slate_id", sort=True):
        ordered = slate.sort_values("symbol", kind="mergesort")
        choices: dict[str, tuple[pd.Series | None, bool]] = {}
        for model in ("M0", "M1", "M2", "M3"):
            column = f"expected_signed_remaining_move_score__{model}"
            candidate = ordered.sort_values(
                [column, "symbol"], ascending=[False, True], kind="mergesort"
            ).iloc[0]
            positive = float(candidate[column]) > 0.0
            choices[model] = (candidate if positive else None, positive)
        choices["highest_opening_relative_momentum"] = (
            ordered.sort_values(
                ["open_to_decision_cohort_relative_return_bps", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0],
            True,
        )
        choices["strongest_opening_reversal"] = (
            ordered.sort_values(
                ["open_to_decision_cohort_relative_return_bps", "symbol"],
                ascending=[True, True],
                kind="mergesort",
            ).iloc[0],
            True,
        )
        choices["highest_M1_large_move_probability"] = (
            ordered.sort_values(
                ["p__large_remaining_move__M1", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0],
            True,
        )
        seed = int.from_bytes(hashlib.sha256(f"20260722|{slate_id}".encode()).digest()[:8], "big")
        choices["random_within_slate"] = (ordered.iloc[seed % len(ordered)], True)
        for name, (selected, selected_flag) in choices.items():
            rows.append(
                {
                    "slate_id": str(slate_id),
                    "session": str(ordered.iloc[0]["session"]),
                    "decision_ordinal": int(ordered.iloc[0]["decision_ordinal"]),
                    "candidate": name,
                    "selected": selected_flag,
                    "selected_symbol": str(selected["symbol"]) if selected is not None else None,
                    "raw_remaining_return_bps": (
                        float(selected["raw_remaining_return_bps"]) if selected is not None else 0.0
                    ),
                    "cohort_relative_remaining_return_bps": (
                        float(selected["residual_remaining_return_bps"])
                        if selected is not None
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _economic(selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in selection.groupby("candidate", sort=True):
        selected = group.loc[group["selected"]]
        for friction in (0.0, 10.0, 20.0):
            rows.append(
                {
                    "record_type": "candidate_summary",
                    "candidate": candidate,
                    "friction_bps": friction,
                    "slate_count": int(group["slate_id"].nunique()),
                    "selected_slate_count": len(selected),
                    "selection_rate": float(len(selected) / len(group)),
                    "mean_raw_remaining_return_bps": (
                        float((selected["raw_remaining_return_bps"] - friction).mean())
                        if not selected.empty
                        else math.nan
                    ),
                    "mean_cohort_relative_remaining_return_bps": (
                        float((selected["cohort_relative_remaining_return_bps"] - friction).mean())
                        if not selected.empty
                        else math.nan
                    ),
                    "paired_mean_cohort_relative_difference_vs_M1_bps": math.nan,
                }
            )
    pivot = selection.pivot(
        index="slate_id", columns="candidate", values="cohort_relative_remaining_return_bps"
    )
    for candidate in ("M0", "M2", "M3"):
        rows.append(
            {
                "record_type": "candidate_minus_M1_paired",
                "candidate": candidate,
                "friction_bps": 0.0,
                "slate_count": len(pivot),
                "selected_slate_count": int(
                    selection.loc[selection["candidate"].eq(candidate), "selected"].sum()
                ),
                "selection_rate": math.nan,
                "mean_raw_remaining_return_bps": math.nan,
                "mean_cohort_relative_remaining_return_bps": math.nan,
                "paired_mean_cohort_relative_difference_vs_M1_bps": float(
                    (pivot[candidate] - pivot["M1"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _loss(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    values = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    if metric == "brier":
        return float(np.mean((labels - values) ** 2))
    return float(log_loss(labels, values, labels=[0, 1]))


def _bootstrap(assessment: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("movement_M2_minus_M1_brier_improvement", "large_remaining_move", "M2", "M1", "brier"),
        ("movement_M3_minus_M2_brier_improvement", "large_remaining_move", "M3", "M2", "brier"),
        ("direction_M2_minus_M1_brier_improvement", "up_given_large_move", "M2", "M1", "brier"),
        ("direction_M3_minus_M2_brier_improvement", "up_given_large_move", "M3", "M2", "brier"),
        (
            "movement_M2_minus_M1_log_loss_improvement",
            "large_remaining_move",
            "M2",
            "M1",
            "log_loss",
        ),
        (
            "movement_M3_minus_M2_log_loss_improvement",
            "large_remaining_move",
            "M3",
            "M2",
            "log_loss",
        ),
        (
            "direction_M2_minus_M1_log_loss_improvement",
            "up_given_large_move",
            "M2",
            "M1",
            "log_loss",
        ),
        (
            "direction_M3_minus_M2_log_loss_improvement",
            "up_given_large_move",
            "M3",
            "M2",
            "log_loss",
        ),
    )
    pivot = selection.pivot(
        index="slate_id", columns="candidate", values="cohort_relative_remaining_return_bps"
    )
    pivot["session"] = pivot.index.to_series().str.split("|").str[0]
    economic_by_session = (pivot["M3"] - pivot["M1"]).groupby(pivot["session"]).mean()
    sessions = tuple(sorted(assessment["session"].astype(str).unique()))
    session_frames = {
        session: group.copy() for session, group in assessment.groupby("session", sort=True)
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_sessions = tuple(
            str(value) for value in rng.choice(sessions, len(sessions), replace=True)
        )
        sampled = pd.concat(
            [session_frames[session] for session in sampled_sessions], ignore_index=True
        )
        for metric_name, target, candidate, baseline, metric in specs:
            population = _target_population(sampled, target)
            labels = population[target].to_numpy(dtype=int)
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric_name,
                    "value": _loss(
                        labels,
                        population[f"p__{target}__{baseline}"].to_numpy(dtype=float),
                        metric,
                    )
                    - _loss(
                        labels,
                        population[f"p__{target}__{candidate}"].to_numpy(dtype=float),
                        metric,
                    ),
                    "confidence_level": math.nan,
                    "lower": math.nan,
                    "upper": math.nan,
                }
            )
        values = economic_by_session.loc[list(sampled_sessions)].to_numpy(dtype=float)
        rows.append(
            {
                "record_type": "draw",
                "draw": draw,
                "metric": "M3_minus_M1_top_one_cohort_relative_bps",
                "value": float(np.mean(values)),
                "confidence_level": math.nan,
                "lower": math.nan,
                "upper": math.nan,
            }
        )
    draws = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for metric, values in draws.groupby("metric", sort=True)["value"]:
        array = values.to_numpy(dtype=float)
        for confidence, low, high in ((0.90, 0.05, 0.95), (0.95, 0.025, 0.975)):
            summaries.append(
                {
                    "record_type": "interval",
                    "draw": -1,
                    "metric": metric,
                    "value": float(np.mean(array)),
                    "confidence_level": confidence,
                    "lower": float(np.quantile(array, low)),
                    "upper": float(np.quantile(array, high)),
                }
            )
    return pd.concat([draws, pd.DataFrame(summaries)], ignore_index=True)


def _slate_weights(slate_ids: pd.Series) -> np.ndarray:
    values = slate_ids.astype(str).reset_index(drop=True)
    sizes = values.groupby(values, sort=True).transform("size").to_numpy(dtype=float)
    return 1.0 / sizes


def _fit_null(
    frame: pd.DataFrame, target: str, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    values = frame[list(features)].to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260720,
        n_jobs=1,
    )
    estimator.fit(
        (values - means) / scales,
        frame[target].to_numpy(dtype=int),
        sample_weight=_slate_weights(frame["slate_id"]),
    )
    if int(np.max(estimator.n_iter_)) >= 250:
        raise AssertionError("auditor null model did not converge")
    return means, scales, np.asarray(estimator.coef_[0]), float(estimator.intercept_[0])


def _predict_null(
    frame: pd.DataFrame,
    features: Sequence[str],
    fitted: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    means, scales, coefficients, intercept = fitted
    values = frame[list(features)].to_numpy(dtype=float)
    linear = intercept + ((values - means) / scales) @ coefficients
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))


def _permute(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    seed: int,
    draw: int,
) -> pd.DataFrame:
    output = frame.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, draw]))
    for indices in frame.groupby("slate_id", sort=True).groups.values():
        positions = np.asarray(list(indices), dtype=int)
        source = frame.loc[positions, list(columns)].to_numpy(copy=True)
        output.loc[positions, list(columns)] = source[rng.permutation(len(positions))]
    return output


def _null(
    development: pd.DataFrame,
    assessment: pd.DataFrame,
    model_features: Mapping[str, Sequence[str]],
    structural: Sequence[str],
    real_metrics: pd.DataFrame,
) -> pd.DataFrame:
    def improvement(target: str, candidate: str, baseline: str) -> float:
        subset = real_metrics.loc[real_metrics["target"].eq(target)]
        candidate_value = float(subset.loc[subset["model"].eq(candidate), "brier_score"].iloc[0])
        baseline_value = float(subset.loc[subset["model"].eq(baseline), "brier_score"].iloc[0])
        return baseline_value - candidate_value

    real = {
        "movement_M2_minus_M1_brier_improvement": improvement("large_remaining_move", "M2", "M1"),
        "movement_M3_minus_M2_brier_improvement": improvement("large_remaining_move", "M3", "M2"),
        "direction_M2_minus_M1_brier_improvement": improvement("up_given_large_move", "M2", "M1"),
        "direction_M3_minus_M2_brier_improvement": improvement("up_given_large_move", "M3", "M2"),
    }
    rows: list[dict[str, Any]] = []
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for draw in range(NULL_DRAWS):
        dev = _permute(development, structural, seed=NULL_SEED, draw=draw)
        assess = _permute(assessment, structural, seed=NULL_SEED + 1, draw=draw)
        for target, label in (
            ("large_remaining_move", "movement"),
            ("up_given_large_move", "direction"),
        ):
            train = _target_population(dev, target).reset_index(drop=True)
            scored = _target_population(assess, target)
            m2 = _fit_null(train, target, model_features["M2"])
            m3 = _fit_null(train, target, model_features["M3"])
            labels = scored[target].to_numpy(dtype=int)
            p1 = assessment.loc[scored.index, f"p__{target}__M1"].to_numpy(dtype=float)
            p2 = _predict_null(scored, model_features["M2"], m2)
            p3 = _predict_null(scored, model_features["M3"], m3)
            values = {
                f"{label}_M2_minus_M1_brier_improvement": _loss(labels, p1, "brier")
                - _loss(labels, p2, "brier"),
                f"{label}_M3_minus_M2_brier_improvement": _loss(labels, p2, "brier")
                - _loss(labels, p3, "brier"),
            }
            for metric, value in values.items():
                rows.append(
                    {
                        "record_type": "draw",
                        "draw": draw,
                        "metric": metric,
                        "value": value,
                        "real_value": real[metric],
                        "null_90th_percentile": math.nan,
                        "real_percentile_under_null": math.nan,
                    }
                )
    draws = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for metric, values in draws.groupby("metric", sort=True)["value"]:
        array = values.to_numpy(dtype=float)
        summaries.append(
            {
                "record_type": "summary",
                "draw": -1,
                "metric": metric,
                "value": float(np.mean(array)),
                "real_value": real[metric],
                "null_90th_percentile": float(np.quantile(array, 0.90)),
                "real_percentile_under_null": float(np.mean(array <= real[metric])),
            }
        )
    return pd.concat([draws, pd.DataFrame(summaries)], ignore_index=True)


def _compare_frames(actual: pd.DataFrame, expected: pd.DataFrame) -> tuple[bool, str]:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=2e-9,
            atol=2e-9,
            check_like=False,
        )
        return True, "matched"
    except AssertionError as error:
        return False, str(error)[:500]


def _concentration(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(assessment)
    for symbol, count in assessment["symbol"].value_counts(sort=False).sort_index().items():
        rows.append(
            {
                "dimension": "stock_row_share",
                "value": str(symbol),
                "row_count": int(count),
                "share": float(count / total),
                "gate": 0.10,
                "passes": bool(count / total <= 0.10),
            }
        )
    for state, count in assessment["current_state"].value_counts(sort=False).sort_index().items():
        rows.append(
            {
                "dimension": "current_state_row_share",
                "value": str(state),
                "row_count": int(count),
                "share": float(count / total),
                "gate": 0.40,
                "passes": bool(count / total <= 0.40),
            }
        )
    large = assessment.loc[assessment["large_remaining_move"].eq(1)]
    closure_rows = int(assessment["opening_any_short_closure"].eq(1.0).sum())
    support: dict[str, Any] = {
        "assessment_rows": total,
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "actual_large_remaining_moves": len(large),
        "actual_large_moves_by_checkpoint": {
            str(ordinal): int(large["decision_ordinal"].eq(ordinal).sum())
            for ordinal in DECISION_ORDINALS
        },
        "represented_months": int(assessment["year_month"].nunique()),
        "maximum_stock_row_share": float(assessment["symbol"].value_counts().max() / total),
        "maximum_current_state_row_share": float(
            assessment["current_state"].value_counts().max() / total
        ),
        "opening_short_closure_rows": closure_rows,
        "interaction_support_status": (
            "sufficient" if closure_rows >= 100 else "interaction_support_insufficient"
        ),
    }
    support["primary_support_passes"] = bool(
        total >= 3000
        and support["assessment_sessions"] >= 100
        and support["assessment_stocks"] >= 15
        and len(large) >= 600
        and all(value >= 250 for value in support["actual_large_moves_by_checkpoint"].values())
        and support["represented_months"] >= 6
        and support["maximum_stock_row_share"] <= 0.10
        and support["maximum_current_state_row_share"] <= 0.40
    )
    rows.extend(
        [
            {
                "dimension": "opening_short_closure_support",
                "value": "all",
                "row_count": closure_rows,
                "share": float(closure_rows / total),
                "gate": 100.0,
                "passes": closure_rows >= 100,
            },
            {
                "dimension": "primary_support",
                "value": "all",
                "row_count": total,
                "share": 1.0,
                "gate": math.nan,
                "passes": support["primary_support_passes"],
            },
        ]
    )
    return pd.DataFrame(rows), support


def _lookup(frame: pd.DataFrame, target: str, model: str, metric: str) -> float:
    return float(frame.loc[frame["target"].eq(target) & frame["model"].eq(model), metric].iloc[0])


def _increment(
    frame: pd.DataFrame, target: str, candidate: str, baseline: str, metric: str
) -> float:
    return _lookup(frame, target, baseline, metric) - _lookup(frame, target, candidate, metric)


def _bootstrap_lower(frame: pd.DataFrame, metric: str) -> float:
    return float(
        frame.loc[
            frame["record_type"].eq("interval")
            & frame["metric"].eq(metric)
            & frame["confidence_level"].eq(0.90),
            "lower",
        ].iloc[0]
    )


def _null_percentile(frame: pd.DataFrame, metric: str) -> float:
    return float(
        frame.loc[
            frame["record_type"].eq("summary") & frame["metric"].eq(metric),
            "real_percentile_under_null",
        ].iloc[0]
    )


def _positive_months(frame: pd.DataFrame, target: str, candidate: str, baseline: str) -> int:
    candidate_rows = frame.loc[frame["target"].eq(target) & frame["model"].eq(candidate)].set_index(
        "scope"
    )
    baseline_rows = frame.loc[frame["target"].eq(target) & frame["model"].eq(baseline)].set_index(
        "scope"
    )
    return int((baseline_rows["brier_score"] - candidate_rows["brier_score"]).gt(0).sum())


def _decision_category(
    pooled: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoints: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    support: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    def checkpoint_passes(target: str, candidate: str, baseline: str) -> bool:
        values = []
        for ordinal in DECISION_ORDINALS:
            subset = checkpoints.loc[checkpoints["scope"].eq(f"checkpoint_{ordinal}")]
            values.append(_increment(subset, target, candidate, baseline, "brier_score"))
        return min(values) >= -0.001

    movement = bool(
        _increment(pooled, "large_remaining_move", "M2", "M1", "brier_score") > 0
        and _increment(pooled, "large_remaining_move", "M2", "M1", "log_loss") > 0
        and _bootstrap_lower(bootstrap, "movement_M2_minus_M1_brier_improvement") >= 0
        and _bootstrap_lower(bootstrap, "movement_M2_minus_M1_log_loss_improvement") >= 0
        and _positive_months(monthly, "large_remaining_move", "M2", "M1") >= 5
        and _null_percentile(null, "movement_M2_minus_M1_brier_improvement") > 0.90
        and checkpoint_passes("large_remaining_move", "M2", "M1")
        and support["primary_support_passes"]
    )
    direction = bool(
        _increment(pooled, "up_given_large_move", "M2", "M1", "brier_score") > 0
        and _increment(pooled, "up_given_large_move", "M2", "M1", "log_loss") > 0
        and _lookup(pooled, "up_given_large_move", "M2", "auc")
        >= _lookup(pooled, "up_given_large_move", "M1", "auc")
        and _bootstrap_lower(bootstrap, "direction_M2_minus_M1_brier_improvement") >= 0
        and _bootstrap_lower(bootstrap, "direction_M2_minus_M1_log_loss_improvement") >= 0
        and _positive_months(monthly, "up_given_large_move", "M2", "M1") >= 5
        and _null_percentile(null, "direction_M2_minus_M1_brier_improvement") > 0.90
        and checkpoint_passes("up_given_large_move", "M2", "M1")
        and support["primary_support_passes"]
    )
    interaction = False
    for target, label in (
        ("large_remaining_move", "movement"),
        ("up_given_large_move", "direction"),
    ):
        interaction |= bool(
            _increment(pooled, target, "M3", "M2", "brier_score") > 0
            and _increment(pooled, target, "M3", "M2", "log_loss") > 0
            and _bootstrap_lower(bootstrap, f"{label}_M3_minus_M2_brier_improvement") >= 0
            and _bootstrap_lower(bootstrap, f"{label}_M3_minus_M2_log_loss_improvement") >= 0
            and _null_percentile(null, f"{label}_M3_minus_M2_brier_improvement") > 0.90
            and _positive_months(monthly, target, "M3", "M2") >= 5
            and support["interaction_support_status"] == "sufficient"
        )
    if not support["primary_support_passes"]:
        category = "blocked_insufficient_opening_path_support"
    elif movement and direction:
        category = "opening_regime_path_adds_movement_and_direction"
    elif movement:
        category = "opening_regime_path_adds_movement_only"
    elif direction:
        category = "opening_regime_path_adds_direction_only"
    elif interaction:
        category = "opening_loop_regime_interaction_only"
    else:
        category = "opening_structure_no_increment_over_price"
    return category, {"movement": movement, "direction": direction, "interaction": interaction}


def audit_artifacts(artifacts: Path, provider_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    contract = json.loads((artifacts / "contract.json").read_text(encoding="utf-8"))
    decision = json.loads((artifacts / "decision.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((artifacts / "feature_manifest.json").read_text(encoding="utf-8"))
    model_config = json.loads((artifacts / "model_configurations.json").read_text(encoding="utf-8"))
    coefficients = json.loads((artifacts / "model_coefficients.json").read_text(encoding="utf-8"))[
        "models"
    ]
    input_hashes = json.loads(
        (artifacts / "input_artifact_hashes.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads((artifacts / "source_manifest.json").read_text(encoding="utf-8"))
    boundary = json.loads((artifacts / "protected_boundary_audit.json").read_text(encoding="utf-8"))
    thresholds_artifact = json.loads(
        (artifacts / "development_movement_thresholds.json").read_text(encoding="utf-8")
    )
    panel = pd.read_parquet(artifacts / "opening_decision_panel.parquet")
    ledger = pd.read_parquet(artifacts / "opening_state_path_ledger.parquet")
    predictions = pd.read_parquet(artifacts / "assessment_predictions.parquet")

    safety_ok = all(
        contract.get(key) == value
        and contract.get("safety", {}).get(key) == value
        and decision.get(key) == value
        for key, value in SAFETY_FLAGS.items()
    )
    record("safety_flags", safety_ok, SAFETY_FLAGS)

    hash_results = []
    for item in input_hashes["artifacts"]:
        path = REPO_ROOT / str(item["repository_relative_path"])
        actual = sha256_file(path) if path.is_file() else "missing"
        hash_results.append(actual == item["sha256"])
    record("input_hashes", all(hash_results), {"artifact_count": len(hash_results)})
    posterior_reproduction = _reproduce_frozen_posterior()
    record(
        "frozen_regime_and_posterior_reproduction",
        posterior_reproduction["hard_state_agreement"] == 1.0
        and posterior_reproduction["maximum_probability_absolute_error"] <= 1e-8
        and posterior_reproduction["maximum_expected_age_absolute_error"] <= 1e-8
        and posterior_reproduction["maximum_entropy_absolute_error"] <= 1e-8,
        posterior_reproduction,
    )

    source_by_symbol = {row["symbol"]: row for row in source_manifest["sources"]}
    grids: dict[str, pd.DataFrame] = {}
    source_hash_ok = True
    source_minimum: pd.Timestamp | None = None
    source_maximum: pd.Timestamp | None = None
    protected_rows = 0
    for symbol in (*CONTEXT_SYMBOLS, "VTI"):
        safe = _bounded_source(_provider_path(provider_root, symbol))
        source_minimum = (
            safe["timestamp"].min()
            if source_minimum is None
            else min(source_minimum, safe["timestamp"].min())
        )
        source_maximum = (
            safe["timestamp"].max()
            if source_maximum is None
            else max(source_maximum, safe["timestamp"].max())
        )
        protected_rows += int(safe["timestamp"].ge(PROTECTED_START).sum())
        development_source = safe.loc[safe["timestamp"].lt(DEVELOPMENT_END)].reset_index(drop=True)
        declared = source_by_symbol[symbol]
        source_hash_ok &= bool(
            len(safe) == int(declared["complete_safe_bounded_rows"])
            and _arrow_hash(safe) == declared["complete_safe_bounded_hash"]
            and len(development_source) == int(declared["development_bounded_rows"])
            and _arrow_hash(development_source) == declared["development_bounded_hash"]
        )
        if symbol in DECISION_SYMBOLS:
            grids[symbol] = _regular_grid(safe, symbol)
    record(
        "bounded_source_hashes_and_protected_boundary",
        source_hash_ok
        and protected_rows == 0
        and boundary["protected_rows_opened"] == 0
        and boundary["protected_rows_materialised"] == 0
        and boundary["protected_files_touched"] == [],
        {
            "minimum_timestamp": str(source_minimum),
            "maximum_timestamp": str(source_maximum),
            "protected_rows_materialised": protected_rows,
        },
    )
    qa_ok = all(row["status"] != "fail" for row in source_manifest["vendor_qa"])
    record(
        "source_gap_QA_and_corporate_action_ledger",
        qa_ok and all("adjusted_close_present" in row for row in source_manifest["vendor_qa"]),
        {"qa_symbols": len(source_manifest["vendor_qa"])},
    )

    timestamps = pd.to_datetime(panel["decision_bar_start_timestamp_utc"], utc=True)
    available = pd.to_datetime(panel["feature_available_timestamp_utc"], utc=True)
    local_available = available.dt.tz_convert("America/New_York")
    expected_times = panel["decision_ordinal"].map({6: "10:00", 12: "10:30"})
    actual_times = local_available.dt.strftime("%H:%M")
    chronology_ok = bool(
        panel["decision_ordinal"].isin(DECISION_ORDINALS).all()
        and panel["repo_bar_start_ordinal"].eq(panel["decision_ordinal"] - 1).all()
        and available.eq(timestamps + pd.Timedelta(minutes=5)).all()
        and actual_times.eq(expected_times).all()
        and panel["entry_bar_ordinal"].eq(panel["repo_bar_start_ordinal"] + 2).all()
        and panel["terminal_bar_ordinal"].eq(77).all()
        and timestamps.lt(PROTECTED_START).all()
        and available.lt(PROTECTED_START).all()
    )
    record("checkpoints_bar_completion_and_t_plus_2_index", chronology_ok)

    keys = ["symbol", "session", "decision_ordinal"]
    panel_sorted = panel.sort_values(keys, kind="mergesort").reset_index(drop=True)
    ledger_sorted = ledger.sort_values(keys, kind="mergesort").reset_index(drop=True)
    state_ok = panel_sorted[keys].equals(ledger_sorted[keys])
    topology_columns = feature_manifest["opening_path_topology_features"]
    interaction_columns = feature_manifest["interaction_features"]
    for panel_row, ledger_row in zip(
        panel_sorted.itertuples(index=False),
        ledger_sorted.itertuples(index=False),
        strict=True,
    ):
        states = [int(value) for value in str(ledger_row.opening_state_path).split(",")]
        expected_ordinals = list(range(int(panel_row.repo_bar_start_ordinal) + 1))
        stored_ordinals = [int(value) for value in str(ledger_row.opening_bar_ordinals).split(",")]
        state_ok &= bool(
            stored_ordinals == expected_ordinals
            and len(states) == int(panel_row.decision_ordinal)
            and states[-1] == int(panel_row.current_state)
            and states[0] == int(panel_row.opening_state)
        )
        topology = _topology(states)
        for column in topology_columns:
            state_ok &= bool(
                np.isclose(float(getattr(panel_row, column)), topology[column], atol=1e-12)
                and np.isclose(float(getattr(ledger_row, column)), topology[column], atol=1e-12)
            )
        run_states, durations = _runs(states)
        previous = run_states[-2] if len(run_states) >= 2 else None
        posterior = np.asarray(
            [float(value) for value in str(ledger_row.current_posterior).split(",")]
        )
        state_ok &= bool(
            posterior.shape == (STATE_COUNT,)
            and np.isclose(posterior.sum(), 1.0, atol=1e-8)
            and np.isclose(
                float(panel_row.maximum_posterior_probability), posterior.max(), atol=1e-12
            )
            and np.isclose(
                float(panel_row.posterior_entropy),
                -np.sum(posterior[posterior > 0] * np.log(posterior[posterior > 0])),
                atol=1e-12,
            )
            and np.isclose(float(panel_row.current_state_age), durations[-1], atol=1e-12)
            and float(panel_row.opening_state_equals_current) == float(states[0] == states[-1])
        )
        for state in range(STATE_COUNT):
            state_ok &= bool(
                float(getattr(panel_row, f"current_state_{state}")) == float(states[-1] == state)
                and float(getattr(panel_row, f"previous_completed_state_{state}"))
                == float(previous == state)
                and np.isclose(
                    float(getattr(panel_row, f"posterior_state_{state}")),
                    posterior[state],
                    atol=1e-12,
                )
            )
            interaction_sources = {
                "any_short_closure": topology["opening_any_short_closure"],
                "opening_return_to_origin_count": topology["opening_return_to_origin_count"],
                "transition_rate": topology["opening_transition_rate"],
                "current_state_age": topology["current_state_age"],
            }
            for source, value in interaction_sources.items():
                column = f"current_state_{state}_x_{source}"
                state_ok &= bool(
                    column in interaction_columns
                    and np.isclose(
                        float(getattr(panel_row, column)),
                        float(states[-1] == state) * value,
                        atol=1e-12,
                    )
                )
    record(
        "current_state_posterior_topology_and_frozen_interactions",
        state_ok,
        {"rows_reconstructed": len(panel_sorted)},
    )

    anchor_ok = True
    expected_raw = np.empty(len(panel_sorted), dtype=float)
    for symbol, indices in panel_sorted.groupby("symbol", sort=True).groups.items():
        grid = grids[str(symbol)]
        needed_sessions = set(panel_sorted.loc[list(indices), "session"].astype(str))
        needed = grid.loc[grid["session"].isin(needed_sessions)].copy()
        for _, session_grid in needed.groupby("session", sort=True):
            anchor_ok &= bool(
                len(session_grid) == 78
                and session_grid["bar_ordinal"].tolist() == list(range(78))
                and session_grid[["open", "high", "low", "close"]].gt(0.0).all().all()
            )
        indexed = needed.set_index(["session", "bar_ordinal"], verify_integrity=True)
        for index in indices:
            row = panel_sorted.loc[index]
            session = str(row["session"])
            origin = int(row["repo_bar_start_ordinal"])
            entry_ordinal = origin + 2
            entry = float(indexed.loc[(session, entry_ordinal), "open"])
            terminal = float(indexed.loc[(session, 77), "close"])
            origin_timestamp = pd.Timestamp(indexed.loc[(session, origin), "timestamp"])
            anchor_ok &= bool(
                np.isclose(entry, float(row["delayed_entry_open"]), atol=1e-12)
                and np.isclose(terminal, float(row["terminal_close"]), atol=1e-12)
                and origin_timestamp == pd.Timestamp(row["decision_bar_start_timestamp_utc"])
            )
            expected_raw[index] = 10_000.0 * (terminal / entry - 1.0)
    anchor_ok &= bool(
        np.allclose(
            expected_raw,
            panel_sorted["raw_remaining_return_bps"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
        )
    )
    record(
        "delayed_entry_session_close_terminal_and_raw_outcome",
        anchor_ok,
        {"rows_reconstructed": len(panel_sorted)},
    )

    reconstructed_medians = np.empty(len(panel_sorted), dtype=float)
    reconstructed_residuals = np.empty(len(panel_sorted), dtype=float)
    for _, indices in panel_sorted.groupby("slate_id", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        values = expected_raw[positions]
        for offset, position in enumerate(positions):
            median = float(np.median(np.delete(values, offset)))
            reconstructed_medians[position] = median
            reconstructed_residuals[position] = expected_raw[position] - median
    outcome_ok = bool(
        np.allclose(
            reconstructed_medians,
            panel_sorted["cohort_median_return_minus_i_bps"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
        )
        and np.allclose(
            reconstructed_residuals,
            panel_sorted["residual_remaining_return_bps"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-10,
        )
    )
    reconstructed_thresholds: dict[str, float] = {}
    for ordinal in DECISION_ORDINALS:
        mask = panel_sorted["year"].eq(2024) & panel_sorted["decision_ordinal"].eq(ordinal)
        reconstructed_thresholds[str(ordinal)] = float(
            pd.Series(np.abs(reconstructed_residuals[mask.to_numpy()])).quantile(
                0.75, interpolation="linear"
            )
        )
    expected_threshold = panel_sorted["decision_ordinal"].map(
        {int(key): value for key, value in reconstructed_thresholds.items()}
    )
    expected_large = (
        np.abs(reconstructed_residuals) >= expected_threshold.to_numpy(dtype=float)
    ).astype(int)
    expected_up = (reconstructed_residuals > 0.0).astype(int)
    outcome_ok &= bool(
        reconstructed_thresholds == thresholds_artifact["thresholds_bps"]
        and np.allclose(
            expected_threshold,
            panel_sorted["movement_threshold_bps"],
            rtol=0.0,
            atol=1e-12,
        )
        and np.array_equal(expected_large, panel_sorted["large_remaining_move"])
        and np.array_equal(expected_up, panel_sorted["up_given_large_move"])
        and np.array_equal(expected_up, panel_sorted["remaining_direction_up"])
    )
    record(
        "cohort_relative_outcome_and_development_only_thresholds",
        outcome_ok,
        reconstructed_thresholds,
    )

    development = panel.loc[panel["year"].eq(2024)].reset_index(drop=True)
    assessment_panel = panel.loc[panel["year"].eq(2025)].reset_index(drop=True)
    identity_columns = ["symbol", "session", "decision_ordinal", "slate_id"]
    model_ok = assessment_panel[identity_columns].equals(predictions[identity_columns])
    predictions["current_state"] = assessment_panel["current_state"].to_numpy(dtype=int)
    all_model_features: list[str] = []
    for target in TARGETS:
        training = _target_population(development, target).reset_index(drop=True)
        weight_totals = (
            pd.Series(_slate_weights(training["slate_id"]))
            .groupby(training["slate_id"].astype(str).reset_index(drop=True))
            .sum()
        )
        model_ok &= bool(np.allclose(weight_totals, 1.0, atol=1e-12))
        for model_name in ("M0", "M1", "M2", "M3"):
            model = coefficients[target][model_name]
            features = [str(value) for value in model["feature_names"]]
            all_model_features.extend(features)
            values = training[features].to_numpy(dtype=float)
            means = values.mean(axis=0)
            scales = values.std(axis=0, ddof=0)
            scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
            manual = _manual_probability(model, predictions)
            model_ok &= bool(
                features == model_config["models"][model_name]
                and int(model["training_rows"]) == len(training)
                and int(model["training_slates"]) == training["slate_id"].nunique()
                and bool(model["converged"])
                and int(model["iterations"]) < 250
                and np.allclose(means, model["means"], rtol=0.0, atol=1e-12)
                and np.allclose(scales, model["scales"], rtol=0.0, atol=1e-12)
                and np.allclose(
                    manual,
                    predictions[f"p__{target}__{model_name}"],
                    rtol=0.0,
                    atol=1e-12,
                )
            )
    forbidden = sorted(
        {
            name
            for name in all_model_features
            if any(fragment in name.lower() for fragment in FORBIDDEN_FRAGMENTS)
        }
    )
    model_ok &= bool(
        not forbidden
        and model_config["preprocessing_fit_interval"] == "2024_only"
        and model_config["model_row_weight"] == "1 / valid_slate_size"
        and model_config["logistic"]
        == {
            "C": 1.0,
            "class_weight": None,
            "max_iter": 250,
            "n_jobs": 1,
            "penalty": "l2",
            "solver": "liblinear",
        }
        and feature_manifest["provider_volume_label"] == "historical_activity_proxy"
    )
    record(
        "2024_only_normalization_equal_slate_weight_and_manual_logistic_predictions",
        model_ok,
        {"models_reconstructed": 12, "forbidden_model_features": forbidden},
    )

    scale_ok = True
    expected_scale = np.empty(len(predictions), dtype=float)
    scale_manifest = model_config["movement_scale_reference"]["checkpoints"]
    movement_model = coefficients["large_remaining_move"]["M1"]
    development_m1 = _manual_probability(movement_model, development)
    for ordinal in DECISION_ORDINALS:
        dev_mask = development["decision_ordinal"].eq(ordinal).to_numpy()
        assess_mask = predictions["decision_ordinal"].eq(ordinal).to_numpy()
        probabilities = development_m1[dev_mask]
        edges = np.quantile(probabilities, np.linspace(0.0, 1.0, 11))
        edges = np.maximum.accumulate(edges)
        bins = np.minimum(np.searchsorted(edges[1:-1], probabilities, side="right"), 9)
        dev = development.loc[dev_mask]
        overall = float(dev["residual_remaining_return_bps"].abs().median())
        medians = []
        for number in range(10):
            values = dev.loc[bins == number, "residual_remaining_return_bps"].abs()
            medians.append(float(values.median()) if not values.empty else overall)
        declared = scale_manifest[str(ordinal)]
        scale_ok &= bool(
            np.allclose(edges, declared["probability_edges"], rtol=0.0, atol=1e-12)
            and np.allclose(
                medians,
                declared["median_absolute_remaining_movement_bps"],
                rtol=0.0,
                atol=1e-12,
            )
        )
        score_probabilities = predictions.loc[assess_mask, "p__large_remaining_move__M1"].to_numpy(
            dtype=float
        )
        score_bins = np.minimum(np.searchsorted(edges[1:-1], score_probabilities, side="right"), 9)
        expected_scale[assess_mask] = np.asarray(medians)[score_bins]
    scale_ok &= bool(
        np.allclose(
            expected_scale,
            predictions["predicted_remaining_movement_scale_bps"],
            rtol=0.0,
            atol=1e-12,
        )
    )
    for model in ("M0", "M1", "M2", "M3"):
        expected_score = (
            predictions[f"p__large_remaining_move__{model}"]
            * (2.0 * predictions[f"p__up_given_large_move__{model}"] - 1.0)
            * expected_scale
        )
        scale_ok &= bool(
            np.allclose(
                expected_score,
                predictions[f"expected_signed_remaining_move_score__{model}"],
                rtol=0.0,
                atol=1e-12,
            )
        )
    record("delayed_economic_movement_scale_and_signed_scores", scale_ok)

    pooled_expected, checkpoint_expected, monthly_expected, bins_expected = _all_metrics(
        predictions
    )
    movement_actual = pd.read_csv(artifacts / "movement_metrics.csv")
    direction_actual = pd.read_csv(artifacts / "direction_metrics.csv")
    checkpoint_actual = pd.read_csv(artifacts / "checkpoint_metrics.csv")
    monthly_actual = pd.read_csv(artifacts / "monthly_metrics.csv")
    bins_actual = pd.read_csv(artifacts / "calibration_bins.csv")
    movement_expected = pooled_expected.loc[
        pooled_expected["target"].eq("large_remaining_move")
    ].reset_index(drop=True)
    direction_expected = pooled_expected.loc[
        pooled_expected["target"].ne("large_remaining_move")
    ].reset_index(drop=True)
    metric_checks = {
        "movement": _compare_frames(movement_actual, movement_expected),
        "direction": _compare_frames(direction_actual, direction_expected),
        "checkpoint": _compare_frames(checkpoint_actual, checkpoint_expected),
        "monthly": _compare_frames(monthly_actual, monthly_expected),
        "calibration_bins": _compare_frames(bins_actual, bins_expected),
    }
    record(
        "brier_log_loss_auc_calibration_monthly_and_checkpoint_metrics",
        all(result[0] for result in metric_checks.values()),
        {name: detail for name, (_, detail) in metric_checks.items()},
    )

    selection = _selection(predictions)
    economic_expected = _economic(selection)
    economic_actual = pd.read_csv(artifacts / "economic_reference_metrics.csv")
    economic_match = _compare_frames(economic_actual, economic_expected)
    record("economic_reference_ranking_and_paired_M1_comparison", *economic_match)

    bootstrap_expected = _bootstrap(predictions, selection)
    bootstrap_actual = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    bootstrap_match = _compare_frames(bootstrap_actual, bootstrap_expected)
    record(
        "session_block_bootstrap_300_draws_preserving_complete_sessions",
        bootstrap_match[0]
        and bootstrap_actual.loc[bootstrap_actual["record_type"].eq("draw"), "draw"].nunique()
        == 300,
        bootstrap_match[1],
    )

    structural = [
        *feature_manifest["current_regime_features"],
        *feature_manifest["opening_path_topology_features"],
        *feature_manifest["interaction_features"],
    ]
    null_expected = _null(
        development,
        predictions,
        model_config["models"],
        structural,
        pooled_expected,
    )
    null_actual = pd.read_csv(artifacts / "null_metrics.csv")
    null_match = _compare_frames(null_actual, null_expected)
    record(
        "within_slate_complete_structural_bundle_permutation_100_draws",
        null_match[0]
        and null_actual.loc[null_actual["record_type"].eq("draw"), "draw"].nunique() == 100,
        null_match[1],
    )

    concentration_expected, support = _concentration(predictions)
    concentration_actual = pd.read_csv(artifacts / "concentration_metrics.csv")
    concentration_match = _compare_frames(concentration_actual, concentration_expected)
    record(
        "support_and_concentration_gates",
        concentration_match[0] and support["primary_support_passes"],
        {**support, "comparison": concentration_match[1]},
    )
    category, pass_flags = _decision_category(
        pooled_expected,
        monthly_expected,
        checkpoint_expected,
        bootstrap_expected,
        null_expected,
        support,
    )
    decision_ok = bool(
        decision["decision"] == category
        and decision["evidence"]["movement_increment_passes"] == pass_flags["movement"]
        and decision["evidence"]["direction_increment_passes"] == pass_flags["direction"]
        and decision["evidence"]["interaction_increment_passes"] == pass_flags["interaction"]
    )
    record(
        "registered_decision_logic_and_no_economic_rescue",
        decision_ok and not decision["economic_reference_can_rescue_probability_failure"],
        {"reconstructed_category": category, **pass_flags},
    )

    text_artifacts = [
        path
        for path in artifacts.iterdir()
        if path.suffix in {".json", ".csv", ".md"} and path.name != "independent_audit.json"
    ]
    local_path_hits = [
        path.name
        for path in text_artifacts
        if "/Users/" in path.read_text(encoding="utf-8", errors="replace")
    ]
    record(
        "no_local_absolute_paths_or_forbidden_future_payoff_loop_fields",
        not local_path_hits and not forbidden,
        {"local_path_hits": local_path_hits, "forbidden_fields": forbidden},
    )

    passed = all(check["passed"] for check in checks)
    return {
        **SAFETY_FLAGS,
        "audit_id": "opening-regime-path-direction-screen-v0-independent-audit",
        "auditor_imports_runner": False,
        "auditor_imports_screen_module": False,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "checks": checks,
        "reconstructed_decision": category,
        "rows_reconstructed": len(panel),
        "assessment_rows_reconstructed": len(predictions),
        "protected_rows_materialised": protected_rows,
        "bootstrap_draws_reconstructed": BOOTSTRAP_DRAWS,
        "null_draws_reconstructed": NULL_DRAWS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=Path.home()
        / "StockerLocal"
        / "data"
        / "processed"
        / "source=eodhd"
        / "instrument_type=stock",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts.resolve()
    try:
        result = audit_artifacts(artifacts, args.provider_root.expanduser().resolve())
    except Exception as error:  # noqa: BLE001 - fail-closed artifact is required.
        result = {
            **SAFETY_FLAGS,
            "audit_id": "opening-regime-path-direction-screen-v0-independent-audit",
            "auditor_imports_runner": False,
            "auditor_imports_screen_module": False,
            "passed": False,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    write_json(artifacts / "independent_audit.json", result)
    print(canonical_json(result), end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
