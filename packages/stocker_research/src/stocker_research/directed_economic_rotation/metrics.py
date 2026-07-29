"""Calibration, paired predictive, and system activation metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

PAIR_KEYS = (
    "period",
    "forecast_session",
    "destination_family",
    "target_window_sessions",
)


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["target_available"].fillna(False)
        & frame["activation_target"].notna()
        & frame["predicted_activation_probability"].notna()
    ].copy()


def _log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))


def _calibration_fit(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    if len(np.unique(target)) < 2:
        return math.nan, math.nan
    predictor = logit(np.clip(probability, 1e-6, 1.0 - 1e-6))
    design = np.column_stack([np.ones(len(predictor)), predictor])
    coefficients = np.zeros(2, dtype=float)
    for _ in range(50):
        fitted = expit(design @ coefficients)
        weight = np.clip(fitted * (1.0 - fitted), 1e-6, None)
        gradient = design.T @ (target - fitted)
        hessian = (design.T * weight) @ design + np.eye(2) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(coefficients[0]), float(coefficients[1])


def calibration_table(frame: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """Reliability table for observable activation targets only."""

    eligible = _eligible(frame)
    columns = [
        "probability_bin",
        "bin_lower",
        "bin_upper",
        "forecasts",
        "mean_probability",
        "activation_rate",
        "absolute_calibration_error",
    ]
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    edges = np.linspace(0.0, 1.0, bins + 1)
    eligible["probability_bin"] = pd.cut(
        eligible["predicted_activation_probability"],
        bins=edges.tolist(),
        labels=False,
        include_lowest=True,
        right=True,
    ).astype(int)
    records: list[dict[str, float | int]] = []
    for bin_index in range(bins):
        selected = eligible.loc[eligible["probability_bin"].eq(bin_index)]
        if selected.empty:
            continue
        mean_probability = float(selected["predicted_activation_probability"].mean())
        activation_rate = float(selected["activation_target"].astype(float).mean())
        records.append(
            {
                "probability_bin": bin_index,
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]),
                "forecasts": int(len(selected)),
                "mean_probability": mean_probability,
                "activation_rate": activation_rate,
                "absolute_calibration_error": abs(mean_probability - activation_rate),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def activation_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Cause-specific activation metrics with abstention kept explicit."""

    eligible = _eligible(frame)
    if eligible.empty:
        return {"eligible_forecasts": 0}
    target = eligible["activation_target"].astype(float).to_numpy()
    probability = eligible["predicted_activation_probability"].to_numpy(float)
    active = eligible["prediction_state"].eq("nominated").to_numpy(bool)
    true = target.astype(bool)
    precision = float(np.mean(true[active])) if active.any() else math.nan
    recall = float(np.sum(active & true) / np.sum(true)) if true.any() else math.nan
    false_activation = float(np.mean(~true[active])) if active.any() else math.nan
    table = calibration_table(eligible)
    ece = (
        float(
            np.average(
                table["absolute_calibration_error"],
                weights=table["forecasts"],
            )
        )
        if not table.empty
        else math.nan
    )
    intercept, slope = _calibration_fit(target, probability)
    auc = float(roc_auc_score(true, probability)) if len(np.unique(target)) == 2 else math.nan
    return {
        "eligible_forecasts": int(len(eligible)),
        "positive_activations": int(true.sum()),
        "base_activation_rate": float(target.mean()),
        "brier_score": float(np.mean(np.square(probability - target))),
        "log_loss": _log_loss(target, probability),
        "ece": ece,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "false_activation_rate": false_activation,
        "coverage": float(active.mean()),
        "abstention": float(1.0 - active.mean()),
    }


def paired_model_comparison(
    forecasts: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    bootstrap_resamples: int = 2000,
    seed: int = 20260716,
) -> dict[str, float | int | str]:
    """Control-minus-treatment losses on an exactly paired destination population."""

    selected = forecasts.loc[forecasts["model_name"].isin([treatment, control])].copy()
    treatment_keys = set(
        map(
            tuple,
            selected.loc[selected["model_name"].eq(treatment), list(PAIR_KEYS)].itertuples(
                index=False, name=None
            ),
        )
    )
    control_keys = set(
        map(
            tuple,
            selected.loc[selected["model_name"].eq(control), list(PAIR_KEYS)].itertuples(
                index=False, name=None
            ),
        )
    )
    if treatment_keys != control_keys:
        raise ValueError("paired population differs between models")
    eligible = _eligible(selected)
    probability = eligible.pivot(
        index=list(PAIR_KEYS),
        columns="model_name",
        values="predicted_activation_probability",
    )
    target = eligible.pivot(index=list(PAIR_KEYS), columns="model_name", values="activation_target")
    paired_index = probability.dropna(subset=[treatment, control]).index
    probability = probability.loc[paired_index]
    target = target.loc[paired_index]
    if not target[treatment].astype(bool).eq(target[control].astype(bool)).all():
        raise ValueError("paired models have different targets")
    y = target[treatment].astype(float).to_numpy()
    treated = probability[treatment].to_numpy(float)
    controlled = probability[control].to_numpy(float)
    row_improvement = np.square(controlled - y) - np.square(treated - y)
    brier_improvement = float(np.mean(row_improvement)) if len(y) else math.nan
    sessions = [f"{period}|{session}" for period, session, _, _ in paired_index]
    unique_sessions = sorted(set(sessions))
    values = np.full(max(bootstrap_resamples, 1), brier_improvement, dtype=float)
    if bootstrap_resamples > 0 and unique_sessions:
        rng = np.random.default_rng(seed)
        block = min(5, len(unique_sessions))
        blocks = int(math.ceil(len(unique_sessions) / block))
        session_array = np.asarray(sessions)
        for draw in range(bootstrap_resamples):
            starts = rng.integers(0, len(unique_sessions), size=blocks)
            sampled_sessions = [
                unique_sessions[(int(start) + offset) % len(unique_sessions)]
                for start in starts
                for offset in range(block)
            ][: len(unique_sessions)]
            sampled = np.concatenate(
                [row_improvement[session_array == session] for session in sampled_sessions]
            )
            values[draw] = float(np.mean(sampled))
    return {
        "treatment_model": treatment,
        "control_model": control,
        "paired_rows": int(len(y)),
        "treatment_brier": float(np.mean(np.square(treated - y))) if len(y) else math.nan,
        "control_brier": float(np.mean(np.square(controlled - y))) if len(y) else math.nan,
        "brier_improvement": brier_improvement,
        "treatment_log_loss": _log_loss(y, treated) if len(y) else math.nan,
        "control_log_loss": _log_loss(y, controlled) if len(y) else math.nan,
        "log_loss_improvement": (
            _log_loss(y, controlled) - _log_loss(y, treated) if len(y) else math.nan
        ),
        "brier_interval_lower": float(np.quantile(values, 0.025)),
        "brier_interval_upper": float(np.quantile(values, 0.975)),
    }


def system_activation_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Score no-activation and multiple-activation probabilities separately."""

    keys = ["period", "forecast_session", "target_window_sessions"]
    rows = frame.sort_values(keys, kind="stable").drop_duplicates(keys)
    rows = rows.loc[rows["target_available"].fillna(False)].copy()
    no_target = rows["no_activation_flag"].astype(float).to_numpy()
    multiple_target = rows["multiple_activation_flag"].astype(float).to_numpy()
    return {
        "system_windows": int(len(rows)),
        "no_activation_observations": int(no_target.sum()),
        "multiple_activation_observations": int(multiple_target.sum()),
        "no_activation_brier": float(
            np.mean(np.square(rows["probability_no_activation"].to_numpy(float) - no_target))
        ),
        "multiple_activation_brier": float(
            np.mean(
                np.square(rows["probability_multiple_activation"].to_numpy(float) - multiple_target)
            )
        ),
    }


__all__ = [
    "activation_metrics",
    "calibration_table",
    "paired_model_comparison",
    "system_activation_metrics",
]
