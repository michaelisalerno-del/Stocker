"""Independent replay audit for loop-burst mechanism V1.

This auditor does not import the production runner.

research_only: true
live_ordering_enabled: false
order_placement: disabled
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contracts/20260711-loop-burst-mechanism-v1.json"
RUNNER = HERE / "run_loop_burst_mechanism_v1.py"
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
    "runner": "ce0899590698b16281e9f4cb4f732940ea6d82097594b6143c7e5531ea8b9c46",
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
MODELS = ("qhistory", "qfull9", "qoffset_calibration", "qburst_global", "qburst_orientation")
LEARNED_MODELS = MODELS[2:]
PHASE_FEATURES = (
    "log1p_repeat_count",
    "log1p_prior_current_duration",
    "log1p_prior_other_duration",
    "log1p_prior_pair_duration",
    "scheduled_bars_remaining",
)
EPSILON = 1e-12
RIDGE = 0.01
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


def oof_columns() -> list[str]:
    return [
        "anchor_id", "symbol_norm", "session_date", "start_timestamp", "month",
        "cycle_index", "cycle_id", "cycle", "transition_length", "state",
        "current_state", "target", "inverse_compatible_weight", "bar_ordinal",
        "entry_minutes", "entry_clock_quartile", "future_state_1", "future_state_2",
        "qhistory", "qfull9",
    ]


def other_state(cycle: str, current: int) -> int:
    states = {int(value) for value in cycle.split("->")}
    other = states - {int(current)}
    if len(states) != 2 or len(other) != 1:
        raise AssertionError("invalid two-state cycle")
    return next(iter(other))


def rebuild_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_parquet(OOF_SOURCE, columns=oof_columns())
    oof["month"] = oof["month"].astype(str)
    oof["session_date"] = oof["session_date"].astype(str)
    oof["symbol_norm"] = oof["symbol_norm"].astype(str)
    oof = oof.loc[oof["transition_length"].eq(2)].copy()
    runs = pd.read_csv(
        RUN_SOURCE,
        usecols=["run_id", "symbol_norm", "session_date", "state", "duration", "start_pos", "start_timestamp"],
    )
    runs["session_date"] = runs["session_date"].astype(str)
    runs["symbol_norm"] = runs["symbol_norm"].astype(str)
    runs["start_timestamp"] = pd.to_datetime(runs["start_timestamp"], utc=True)
    runs = runs.sort_values(["symbol_norm", "session_date", "run_id"], kind="stable").reset_index(drop=True)
    runs["session_run_index"] = runs.groupby(["symbol_norm", "session_date"], sort=False).cumcount()
    lookup = runs[["symbol_norm", "session_date", "start_timestamp", "state", "duration", "start_pos", "session_run_index"]].rename(
        columns={"state": "run_state", "duration": "realized_current_duration", "start_pos": "run_start_pos"}
    )
    frame = oof.merge(lookup, on=["symbol_norm", "session_date", "start_timestamp"], how="left", validate="many_to_one")
    frame["run_to_session_position_offset"] = frame["run_start_pos"].to_numpy(int) - frame["bar_ordinal"].to_numpy(int)
    if not frame.groupby(["symbol_norm", "session_date"])["run_to_session_position_offset"].nunique().eq(1).all():
        raise AssertionError("position mapping failed")
    sequences = {
        key: (group["state"].to_numpy(int), group["duration"].to_numpy(int))
        for key, group in runs.groupby(["symbol_norm", "session_date"], sort=False)
    }
    n = len(frame)
    repeat = np.zeros(n, int)
    prior_current = np.zeros(n, int)
    prior_other = np.zeros(n, int)
    current_duration = np.zeros(n, int)
    next_duration = np.full(n, np.nan)
    return_duration = np.full(n, np.nan)
    alternate = np.zeros(n, int)
    for position, row in enumerate(
        frame[["symbol_norm", "session_date", "session_run_index", "current_state", "cycle"]].itertuples(index=False)
    ):
        states, durations = sequences[(row.symbol_norm, row.session_date)]
        index = int(row.session_run_index)
        alt = other_state(row.cycle, int(row.current_state))
        alternate[position] = alt
        cursor = index
        count = 0
        while cursor >= 2 and int(states[cursor - 1]) == alt and int(states[cursor - 2]) == int(row.current_state):
            count += 1
            cursor -= 2
        repeat[position] = count
        if count:
            prior_current[position] = durations[index - 2]
            prior_other[position] = durations[index - 1]
        current_duration[position] = durations[index]
        if index + 1 < len(durations):
            next_duration[position] = durations[index + 1]
        if index + 2 < len(durations):
            return_duration[position] = durations[index + 2]
    frame["other_state"] = alternate
    frame["repeat_count"] = repeat
    frame["prior_current_duration"] = prior_current
    frame["prior_other_duration"] = prior_other
    frame["prior_pair_duration"] = prior_current + prior_other
    frame["prior_durable"] = (prior_current >= 2) & (prior_other >= 2)
    frame["log1p_repeat_count"] = np.log1p(repeat)
    frame["log1p_prior_current_duration"] = np.log1p(prior_current)
    frame["log1p_prior_other_duration"] = np.log1p(prior_other)
    frame["log1p_prior_pair_duration"] = np.log1p(prior_current + prior_other)
    frame["scheduled_bars_remaining"] = np.maximum(1, 78 - frame["bar_ordinal"].to_numpy(int)).astype(float)
    frame["realized_current_duration"] = current_duration
    frame["realized_next_duration"] = next_duration
    frame["realized_return_duration"] = return_duration
    frame["two_destination_eligible"] = frame["future_state_1"].ne(8) & frame["future_state_2"].ne(8)
    orientation = frame[["cycle_id", "current_state"]].drop_duplicates().sort_values(["cycle_id", "current_state"], kind="stable").reset_index(drop=True)
    orientation["orientation_index"] = np.arange(len(orientation))
    frame = frame.merge(orientation, on=["cycle_id", "current_state"], validate="many_to_one")
    return frame, orientation


def weighted_center_scale(frame: pd.DataFrame, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = frame[list(PHASE_FEATURES)].to_numpy(float)
    center = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values - center) ** 2, axis=0, weights=weights))
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return center, scale


def design(frame: pd.DataFrame, center: np.ndarray, scale: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    phase = (frame[list(PHASE_FEATURES)].to_numpy(float) - center) / scale
    intercept = np.ones((len(frame), 1))
    if model == "qoffset_calibration":
        return intercept, np.zeros(1)
    if model == "qburst_global":
        return np.column_stack((intercept, phase)), np.asarray([0.0, *([1.0] * 5)])
    orientation = frame["orientation_index"].to_numpy(int)
    dummy = np.zeros((len(frame), 25))
    mask = orientation < 25
    dummy[np.arange(len(frame))[mask], orientation[mask]] = 1.0
    interactions = np.hstack([dummy * phase[:, [index]] for index in range(5)])
    return (
        np.column_stack((intercept, phase, dummy, interactions)),
        np.asarray([0.0, *([1.0] * 5), *([4.0] * 25), *([8.0] * 125)]),
    )


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, float), EPSILON, 1 - EPSILON)
    return np.log(values / (1 - values))


def fit_model(matrix: np.ndarray, offset: np.ndarray, y: np.ndarray, weights: np.ndarray, penalties: np.ndarray, ridge: float) -> tuple[np.ndarray, dict[str, Any]]:
    total = weights.sum()

    def both(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offset + matrix @ beta
        loss = np.logaddexp(0.0, eta) - y * eta
        value = np.sum(weights * loss) / total + 0.5 * ridge * np.sum(penalties * beta * beta)
        gradient = matrix.T @ (weights * (expit(eta) - y)) / total + ridge * penalties * beta
        return float(value), gradient

    result = minimize(lambda beta: both(beta)[0], np.zeros(matrix.shape[1]), jac=lambda beta: both(beta)[1], method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise AssertionError(result.message)
    return result.x, {
        "optimizer_success": bool(result.success), "iterations": int(result.nit),
        "objective": float(result.fun), "gradient_max_abs": float(np.max(np.abs(result.jac))),
        "feature_width": matrix.shape[1], "ridge_lambda": ridge,
    }


def refit_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    continuation = frame.loc[frame["repeat_count"].ge(1)]
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    parameters: dict[str, np.ndarray] = {}
    for month in VALIDATION_MONTHS:
        training = continuation.loc[continuation["month"].isin(TRAINING_MONTHS[month])]
        validation = continuation.loc[continuation["month"].eq(month)].copy()
        weights = training["inverse_compatible_weight"].to_numpy(float)
        y = training["target"].to_numpy(int)
        center, scale = weighted_center_scale(training, weights)
        parameters[f"{month}__center"] = center
        parameters[f"{month}__scale"] = scale
        training_offset = logit(training["qfull9"])
        validation_offset = logit(validation["qfull9"])
        for model in LEARNED_MODELS:
            train_matrix, penalties = design(training, center, scale, model)
            validation_matrix, _ = design(validation, center, scale, model)
            ridge = 0.0 if model == "qoffset_calibration" else RIDGE
            beta, audit = fit_model(train_matrix, training_offset, y, weights, penalties, ridge)
            parameters[f"{month}__{model}__beta"] = beta
            validation[model] = np.clip(expit(validation_offset + validation_matrix @ beta), EPSILON, 1 - EPSILON)
            audits.append({
                "validation_month": month, "training_months_json": json.dumps(TRAINING_MONTHS[month]),
                "model": model, "training_rows": len(training), "training_positives": int(y.sum()),
                "validation_rows": len(validation), "validation_positives": int(validation["target"].sum()), **audit,
            })
        outputs.append(validation)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audits), parameters


def losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPSILON, 1 - EPSILON)
    return (-(y * np.log(p) + (1 - y) * np.log(1 - p)), (y - p) ** 2)


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) and weights.sum() else math.nan


def calibration(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> tuple[float, float, int]:
    bins = np.minimum((np.clip(p, 0, 1) * 10).astype(int), 9)
    values: list[tuple[float, float]] = []
    for index in range(10):
        mask = bins == index
        if mask.sum() < 250 or weights[mask].sum() <= 0:
            continue
        values.append((weights[mask].sum(), abs(wmean(y[mask], weights[mask]) - wmean(p[mask], weights[mask]))))
    if not values:
        return math.inf, math.inf, 0
    total = sum(weight for weight, _ in values)
    return float(sum(weight * error for weight, error in values) / total), float(max(error for _, error in values)), len(values)


def metric(frame: pd.DataFrame, model: str, surface: str = "inverse_compatible") -> dict[str, Any]:
    weights = frame["inverse_compatible_weight"].to_numpy(float) if surface == "inverse_compatible" else np.ones(len(frame))
    y = frame["target"].to_numpy(int)
    ll, br = losses(y, frame[model].to_numpy(float))
    ece, maximum, bins = calibration(y, frame[model].to_numpy(float), weights)
    return {"rows": len(frame), "positives": int(y.sum()), "weight_sum": float(weights.sum()), "log_loss": wmean(ll, weights), "brier": wmean(br, weights), "ece": ece, "maximum_supported_bin_error": maximum, "supported_bins": bins}


def daily_difference(frame: pd.DataFrame, candidate: str, baseline: str, endpoint: str) -> np.ndarray:
    y = frame["target"].to_numpy(int)
    candidate_loss = losses(y, frame[candidate])[0 if endpoint == "log_loss" else 1]
    baseline_loss = losses(y, frame[baseline])[0 if endpoint == "log_loss" else 1]
    weights = frame["inverse_compatible_weight"].to_numpy(float)
    grouped = pd.DataFrame({"date": frame["session_date"], "weighted": (candidate_loss - baseline_loss) * weights, "weight": weights}).groupby("date", sort=True).sum()
    return (grouped["weighted"] / grouped["weight"]).to_numpy(float)


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    blocks = np.asarray([values[index:index + 5].mean() for index in range(0, len(values), 5) if len(values[index:index + 5]) == 5])
    sampled = np.random.default_rng(seed).choice(blocks, size=(BOOTSTRAP_DRAWS, len(blocks)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def sign_flip(values: np.ndarray, seed: int) -> float:
    null = (np.random.default_rng(seed).choice(np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_DRAWS, len(values))) @ values) / len(values)
    return float((1 + np.sum(null <= values.mean())) / (SIGN_FLIP_DRAWS + 1))


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    ordered = sorted(output.index, key=lambda index: output.loc[index, "p_value"])
    running = 0.0
    for rank, index in enumerate(ordered, start=1):
        adjusted = min(1.0, max(running, (len(ordered) - rank + 1) * output.loc[index, "p_value"]))
        running = adjusted
        output.loc[index, "holm_adjusted_p"] = adjusted
        output.loc[index, "holm_pass"] = adjusted <= 0.05
        output.loc[index, "holm_rank"] = rank
        output.loc[index, "family_size"] = len(ordered)
    output["holm_rank"] = output["holm_rank"].astype(int)
    output["family_size"] = output["family_size"].astype(int)
    return output


def recurrence(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    primary = frame.loc[frame["month"].isin(VALIDATION_MONTHS)]
    rows = []
    for (cycle, state), selected in primary.groupby(["cycle_id", "current_state"], sort=True):
        repeat = selected["repeat_count"].ge(1).to_numpy()
        weights = selected["inverse_compatible_weight"].to_numpy(float)
        y = selected["target"].to_numpy(int)
        rr, ir = wmean(y[repeat], weights[repeat]), wmean(y[~repeat], weights[~repeat])
        rows.append({"cycle_id": cycle, "current_state": int(state), "rows": len(selected), "positives": int(y.sum()), "recurrent_rows": int(repeat.sum()), "recurrent_positives": int(y[repeat].sum()), "initiation_rows": int((~repeat).sum()), "initiation_positives": int(y[~repeat].sum()), "recurrent_rate": rr, "initiation_rate": ir, "rate_difference": rr - ir, "rate_ratio": rr / ir if ir > 0 else math.inf, "supported": repeat.sum() >= 100 and y[repeat].sum() >= 20})
    orientations = pd.DataFrame(rows)
    repeat = primary["repeat_count"].ge(1).to_numpy()
    weights = primary["inverse_compatible_weight"].to_numpy(float)
    y = primary["target"].to_numpy(int)
    rr, ir = wmean(y[repeat], weights[repeat]), wmean(y[~repeat], weights[~repeat])
    daily_rows = []
    for date, selected in primary.groupby("session_date", sort=True):
        mask = selected["repeat_count"].ge(1).to_numpy()
        if not mask.any() or not (~mask).any():
            continue
        day_weights = selected["inverse_compatible_weight"].to_numpy(float)
        day_y = selected["target"].to_numpy(int)
        daily_rows.append({"session_date": date, "risk_difference": wmean(day_y[mask], day_weights[mask]) - wmean(day_y[~mask], day_weights[~mask])})
    daily = pd.DataFrame(daily_rows)
    lower, upper = bootstrap(daily["risk_difference"].to_numpy(float), SEED + 7000)
    summary = {"rows": len(primary), "recurrent_rows": int(repeat.sum()), "recurrent_rate": rr, "initiation_rate": ir, "pooled_rate_difference": rr - ir, "pooled_rate_ratio": rr / ir, "supported_orientations": int(orientations["supported"].sum()), "supported_orientations_ratio_above_one": int((orientations.loc[orientations["supported"], "rate_ratio"] > 1).sum()), "daily_sessions": len(daily), "bootstrap_lower": lower, "bootstrap_upper": upper}
    return orientations, summary, daily


def boundary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = frame.loc[frame["month"].isin(VALIDATION_MONTHS) & frame["cycle_id"].eq("cycle_13") & frame["current_state"].eq(5)].copy()
    selected["before"] = selected["future_state_1"].eq(8)
    selected["after"] = selected["future_state_1"].eq(7) & selected["future_state_2"].eq(8)
    rows = []
    for clock, group in selected.groupby("entry_clock_quartile", sort=True):
        eligible = group["two_destination_eligible"]
        rows.append({"entry_clock_quartile": int(clock), "rows": len(group), "positives": int(group["target"].sum()), "rate": group["target"].mean(), "boundary_before_exit": int(group["before"].sum()), "boundary_after_other": int(group["after"].sum()), "boundary_fraction": (group["before"] | group["after"]).mean(), "eligible_rows": int(eligible.sum()), "eligible_positives": int(group.loc[eligible, "target"].sum()), "eligible_rate": group.loc[eligible, "target"].mean()})
    table = pd.DataFrame(rows)
    indexed = table.set_index("entry_clock_quartile")
    return table, {"cycle_13_state_5_rows": len(selected), "late_boundary_fraction": float(indexed.loc[3, "boundary_fraction"]), "mid_eligible_rate": float(indexed.loc[1, "eligible_rate"]), "late_eligible_rate": float(indexed.loc[3, "eligible_rate"]), "mid_minus_late_eligible_rate": float(indexed.loc[1, "eligible_rate"] - indexed.loc[3, "eligible_rate"])}


def chatter(frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame.loc[frame["month"].isin(VALIDATION_MONTHS) & frame["cycle_id"].eq("cycle_13") & frame["current_state"].eq(5) & frame["target"].eq(1)]
    durations = selected[["realized_current_duration", "realized_next_duration", "realized_return_duration"]]
    total = durations.sum(axis=1)
    return {"rows": len(selected), "all_three_legs_at_least_two_fraction": float((durations >= 2).all(axis=1).mean()), "any_one_bar_leg_fraction": float((durations == 1).any(axis=1).mean()), "total_duration_median_bars": float(total.median()), "total_duration_mean_bars": float(total.mean()), "total_duration_p25_bars": float(total.quantile(.25)), "total_duration_p75_bars": float(total.quantile(.75))}


def repeat_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["repeat_bucket"] = np.minimum(frame["repeat_count"], 4)
    rows = []
    for bucket, selected in frame.groupby("repeat_bucket", sort=True):
        weights = selected["inverse_compatible_weight"].to_numpy(float)
        rows.append({"repeat_bucket": "4+" if int(bucket) == 4 else str(int(bucket)), "rows": len(selected), "positives": int(selected["target"].sum()), "event_rate": wmean(selected["target"].to_numpy(int), weights), "mean_scheduled_bars_remaining": wmean(selected["scheduled_bars_remaining"].to_numpy(float), weights), "mean_prior_pair_duration": wmean(selected["prior_pair_duration"].to_numpy(float), weights)})
    return pd.DataFrame(rows)


def compare_frames(calculated: pd.DataFrame, stored: pd.DataFrame, keys: Sequence[str], tolerance: float = 2e-10) -> tuple[bool, float, int]:
    joined = calculated.merge(stored, on=list(keys), suffixes=("__c", "__s"), validate="one_to_one")
    if len(joined) != len(calculated) or len(joined) != len(stored):
        return False, math.inf, 1
    maximum, errors = 0.0, 0
    for column in calculated.columns:
        if column in keys or column not in stored.columns:
            continue
        left, right = joined[f"{column}__c"], joined[f"{column}__s"]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            a, b = left.to_numpy(float), right.to_numpy(float)
            finite = np.isfinite(a) & np.isfinite(b)
            same_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
            errors += int((~(finite | same_inf)).sum())
            if finite.any():
                difference = np.abs(a[finite] - b[finite])
                maximum = max(maximum, float(difference.max()))
                errors += int((difference > tolerance).sum())
        else:
            left_text = left.astype("string").fillna("")
            right_text = right.astype("string").fillna("")
            errors += int((left_text != right_text).sum())
    return errors == 0, maximum, errors


def run_audit() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    hashes = {
        "contract": sha256(CONTRACT), "runner": sha256(RUNNER), "oof_source": sha256(OOF_SOURCE),
        "oof_audit": sha256(OOF_AUDIT), "run_source": sha256(RUN_SOURCE), "cycle_source": sha256(CYCLE_SOURCE),
        "factor_contract": sha256(FACTOR_CONTRACT), "factor_runner": sha256(FACTOR_RUNNER),
    }
    record("frozen_hashes", hashes == EXPECTED_HASHES, hashes)
    contract = json.loads(CONTRACT.read_text())
    record("safety_labels", contract["research_only"] is True and contract["live_ordering_enabled"] is False and contract["order_placement"] == "disabled", {key: contract[key] for key in ("research_only", "live_ordering_enabled", "order_placement")})
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text())["files"]
    mismatches = {name: sha256(ROOT / name) for name, descriptor in manifest.items() if sha256(ROOT / name) != descriptor["sha256"]}
    record("artifact_hashes", not mismatches, mismatches)

    features, orientations = rebuild_features()
    stored_features = pd.read_parquet(ROOT / "burst_feature_ledger_2024_jul_dec.parquet")
    feature_columns = ["anchor_id", "cycle_index", "repeat_count", "prior_current_duration", "prior_other_duration", "prior_pair_duration", "prior_durable", *PHASE_FEATURES, "realized_current_duration", "realized_next_duration", "realized_return_duration", "two_destination_eligible", "orientation_index"]
    feature_error = 0.0
    categorical_errors = 0
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(features[column]) and pd.api.types.is_numeric_dtype(stored_features[column]):
            a, b = features[column].to_numpy(float), stored_features[column].to_numpy(float)
            finite = np.isfinite(a) & np.isfinite(b)
            feature_error = max(feature_error, float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0)
            categorical_errors += int((~(finite | (np.isnan(a) & np.isnan(b)))).sum())
        else:
            categorical_errors += int((features[column].astype(str) != stored_features[column].astype(str)).sum())
    record("sequence_feature_replay", feature_error == 0 and categorical_errors == 0 and len(features) == len(stored_features), {"max": feature_error, "errors": categorical_errors, "rows": len(features)})
    stored_orientations = pd.read_csv(ROOT / "orientation_dictionary.csv")
    record("orientation_dictionary", orientations.to_dict("records") == stored_orientations.to_dict("records"), len(orientations))

    replay_predictions, fit_audit, parameters = refit_predictions(features)
    stored_predictions = pd.read_parquet(ROOT / "continuation_predictions_2024_oct_dec.parquet")
    key_equal = replay_predictions[["anchor_id", "cycle_index"]].equals(stored_predictions[["anchor_id", "cycle_index"]])
    prediction_error = max(float(np.max(np.abs(replay_predictions[model].to_numpy(float) - stored_predictions[model].to_numpy(float)))) for model in LEARNED_MODELS)
    stored_parameters = np.load(ROOT / "model_parameters.npz")
    parameter_error = max(float(np.max(np.abs(value - stored_parameters[name]))) for name, value in parameters.items())
    stored_parameters.close()
    record("model_parameter_replay", parameter_error < 2e-10, parameter_error)
    record("model_prediction_replay", key_equal and prediction_error < 2e-10, {"keys": key_equal, "max": prediction_error})
    passed, maximum, errors = compare_frames(fit_audit, pd.read_csv(ROOT / "fit_audit.csv"), ["validation_month", "model"])
    record("fit_audit_replay", passed, {"max": maximum, "errors": errors})

    pooled_rows = []
    for surface in ("inverse_compatible", "unweighted"):
        for model in MODELS:
            pooled_rows.append({"surface": surface, "model": model, **metric(stored_predictions, model, surface)})
    pooled = pd.DataFrame(pooled_rows)
    passed, maximum, errors = compare_frames(pooled, pd.read_csv(ROOT / "pooled_metrics.csv"), ["surface", "model"])
    record("pooled_calibration_replay", passed, {"max": maximum, "errors": errors})

    comparison_rows, multi_rows = [], []
    for comparison_index, baseline in enumerate(("qfull9", "qoffset_calibration")):
        for endpoint_index, endpoint in enumerate(("log_loss", "brier")):
            values = daily_difference(stored_predictions, "qburst_orientation", baseline, endpoint)
            seed = SEED + comparison_index * 100 + endpoint_index
            lower, upper = bootstrap(values, seed)
            p = sign_flip(values, seed + 10)
            comparison_rows.append({"candidate": "qburst_orientation", "baseline": baseline, "endpoint": endpoint, "daily_sessions": len(values), "mean_difference": float(values.mean()), "bootstrap_lower": lower, "bootstrap_upper": upper, "p_value": p})
            multi_rows.append({"baseline": baseline, "endpoint": endpoint, "p_value": p})
    comparisons = pd.DataFrame(comparison_rows)
    multiplicity = holm(pd.DataFrame(multi_rows))
    passed1, maximum1, errors1 = compare_frames(comparisons, pd.read_csv(ROOT / "comparisons.csv"), ["candidate", "baseline", "endpoint"])
    passed2, maximum2, errors2 = compare_frames(multiplicity, pd.read_csv(ROOT / "multiplicity.csv"), ["baseline", "endpoint"])
    record("bootstrap_Holm_replay", passed1 and passed2, {"max": max(maximum1, maximum2), "errors": errors1 + errors2})

    temporal_rows, stock_rows, orientation_rows = [], [], []
    for month in VALIDATION_MONTHS:
        selected = stored_predictions.loc[stored_predictions["month"].eq(month)]
        for model in MODELS:
            temporal_rows.append({"month": month, "model": model, **metric(selected, model)})
    for symbol in sorted(stored_predictions["symbol_norm"].unique()):
        selected = stored_predictions.loc[stored_predictions["symbol_norm"].ne(symbol)]
        for model in ("qoffset_calibration", "qburst_orientation"):
            stock_rows.append({"deleted_symbol": symbol, "model": model, **metric(selected, model)})
    for (cycle, state), selected in stored_predictions.groupby(["cycle_id", "current_state"], sort=True):
        supported = len(selected) >= 100 and selected["target"].sum() >= 20
        for model in ("qoffset_calibration", "qburst_orientation"):
            orientation_rows.append({"cycle_id": cycle, "current_state": int(state), "supported": supported, "model": model, **metric(selected, model)})
    tables = [
        ("temporal_replay", pd.DataFrame(temporal_rows), pd.read_csv(ROOT / "temporal_slices.csv"), ["month", "model"]),
        ("stock_replay", pd.DataFrame(stock_rows), pd.read_csv(ROOT / "stock_deletions.csv"), ["deleted_symbol", "model"]),
        ("orientation_replay", pd.DataFrame(orientation_rows), pd.read_csv(ROOT / "orientation_slices.csv"), ["cycle_id", "current_state", "model"]),
    ]
    for name, calculated, saved, keys in tables:
        passed, maximum, errors = compare_frames(calculated, saved, keys)
        record(name, passed, {"max": maximum, "errors": errors})
    durable = stored_predictions.loc[stored_predictions["prior_durable"]]
    durable_table = pd.DataFrame([{"model": model, **metric(durable, model)} for model in ("qoffset_calibration", "qburst_orientation")])
    passed, maximum, errors = compare_frames(durable_table, pd.read_csv(ROOT / "durable_prior_slice.csv"), ["model"])
    record("durable_replay", passed, {"max": maximum, "errors": errors})

    recurrence_table, recurrence_summary, recurrence_daily = recurrence(features)
    boundary_table, boundary_summary = boundary(features)
    chatter_summary = chatter(features)
    repeat_counts = repeat_table(stored_predictions)
    diagnostics = [
        ("recurrence_orientation_replay", recurrence_table, pd.read_csv(ROOT / "recurrence_orientations.csv"), ["cycle_id", "current_state"]),
        ("recurrence_daily_replay", recurrence_daily, pd.read_csv(ROOT / "recurrence_daily.csv"), ["session_date"]),
        ("boundary_replay", boundary_table, pd.read_csv(ROOT / "cycle13_session_boundary.csv"), ["entry_clock_quartile"]),
        ("repeat_count_replay", repeat_counts, pd.read_csv(ROOT / "repeat_count_diagnostic.csv"), ["repeat_bucket"]),
    ]
    for name, calculated, saved, keys in diagnostics:
        passed, maximum, errors = compare_frames(calculated, saved, keys)
        record(name, passed, {"max": maximum, "errors": errors})

    primary = pooled.loc[pooled["surface"].eq("inverse_compatible")].set_index("model")
    temporal = pd.DataFrame(temporal_rows).set_index(["month", "model"])
    stocks = pd.DataFrame(stock_rows).set_index(["deleted_symbol", "model"])
    orientation_frame = pd.DataFrame(orientation_rows)
    orientation_index = orientation_frame.set_index(["cycle_id", "current_state", "model"])
    supported_keys = orientation_frame.loc[orientation_frame["supported"], ["cycle_id", "current_state"]].drop_duplicates().itertuples(index=False, name=None)
    orientation_differences = np.asarray([orientation_index.loc[(cycle, state, "qburst_orientation"), "log_loss"] - orientation_index.loc[(cycle, state, "qoffset_calibration"), "log_loss"] for cycle, state in supported_keys])
    durable_index = durable_table.set_index("model")
    primary_checks = {
        "minimum_pooled_log_loss_improvement": (primary.loc["qoffset_calibration", "log_loss"] - primary.loc["qburst_orientation", "log_loss"]) / primary.loc["qoffset_calibration", "log_loss"] >= .005,
        "pooled_brier": primary.loc["qburst_orientation", "brier"] <= primary.loc["qoffset_calibration", "brier"],
        "bootstrap_both_baselines": bool((comparisons["bootstrap_upper"] <= 0).all()),
        "Holm_all_four": bool(multiplicity["holm_pass"].all()),
        "every_month": all(temporal.loc[(month, "qburst_orientation"), endpoint] <= temporal.loc[(month, "qoffset_calibration"), endpoint] for month in VALIDATION_MONTHS for endpoint in ("log_loss", "brier")),
        "every_stock_deletion": all(stocks.loc[(symbol, "qburst_orientation"), endpoint] <= stocks.loc[(symbol, "qoffset_calibration"), endpoint] for symbol in sorted(stored_predictions["symbol_norm"].unique()) for endpoint in ("log_loss", "brier")),
        "orientation_count": int((orientation_differences < 0).sum()) >= 20,
        "orientation_maximum_harm": float(orientation_differences.max()) <= .005,
        "calibration": primary.loc["qburst_orientation", "ece"] <= primary.loc["qoffset_calibration", "ece"] and primary.loc["qburst_orientation", "maximum_supported_bin_error"] <= .03,
        "durable_prior": durable_index.loc["qburst_orientation", "log_loss"] <= durable_index.loc["qoffset_calibration", "log_loss"] and durable_index.loc["qburst_orientation", "brier"] <= durable_index.loc["qoffset_calibration", "brier"],
    }
    stored_model_gate = json.loads((ROOT / "model_gate.json").read_text())
    model_gate_match = primary_checks == stored_model_gate["checks"] and bool(all(primary_checks.values())) == stored_model_gate["pass"]
    record("model_gate_replay", model_gate_match, primary_checks)
    mechanism_checks = {
        "H1_pooled_rate_ratio": recurrence_summary["pooled_rate_ratio"] >= 1.5,
        "H1_orientation_count": recurrence_summary["supported_orientations_ratio_above_one"] >= 24,
        "H1_bootstrap": recurrence_summary["bootstrap_lower"] > 0,
        "H2_late_boundary": boundary_summary["late_boundary_fraction"] >= .25,
        "H2_mid_late_eligible": boundary_summary["mid_minus_late_eligible_rate"] >= .03,
        "H4_durable_support": len(durable) >= 2000 and durable["target"].sum() >= 400,
        "H4_durable_model": primary_checks["durable_prior"],
        "H5_durable_realized_fraction": chatter_summary["all_three_legs_at_least_two_fraction"] >= .35,
        "H5_duration_median": chatter_summary["total_duration_median_bars"] >= 10,
    }
    stored_mechanism = json.loads((ROOT / "mechanism_gate.json").read_text())
    mechanism_match = mechanism_checks == stored_mechanism["checks"] and recurrence_summary == stored_mechanism["recurrence"] and boundary_summary == stored_mechanism["session_boundary"] and chatter_summary == stored_mechanism["chatter"]
    record("mechanism_gate_replay", mechanism_match, mechanism_checks)
    decision = json.loads((ROOT / "decision.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    record("decision_replay", decision["primary_model_pass"] is False and decision["mechanism_gate_pass"] is False and decision["retained_forecaster"] is None and decision["named_loop_good_or_high_promoted"] is False, decision)
    record("summary_reconciliation", summary["decision"] == decision and summary["prediction_rows"] == len(stored_predictions) and summary["fit_count"] == 9 and summary["direct_volume_fields_used"] == [] and summary["later_period_scoring_performed"] is False, {"rows": len(stored_predictions), "fits": summary["fit_count"]})
    all_passed = all(check["passed"] for check in checks.values())
    result = {
        "audit_id": "loop_burst_mechanism_v1_independent_audit",
        "all_passed": all_passed,
        "checks_passed": sum(check["passed"] for check in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "auditor_imported_runner": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    (ROOT / "independent_audit.json").write_text(json.dumps(safe(result), indent=2, sort_keys=True) + "\n")
    if not all_passed:
        raise AssertionError("loop burst audit failed: " + str([name for name, check in checks.items() if not check["passed"]]))
    print(json.dumps(safe(result), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run_audit()
