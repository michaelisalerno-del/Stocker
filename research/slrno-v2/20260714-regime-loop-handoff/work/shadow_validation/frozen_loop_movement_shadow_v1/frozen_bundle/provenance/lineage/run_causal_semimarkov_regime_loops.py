"""Research-only causal semi-Markov regime and loop investigation.

Fit is restricted to 2024.  2025 is a fixed out-of-sample development check.
No 2026 row, return-direction target, P&L, order, or runtime path is available.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler


HERE = Path(__file__).resolve().parent
OUT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
SEED = 20260710
K = 8
MAX_DURATION = 24
MIN_CLEAN_RUN = 2
LEAVE_HORIZON = 3
HISTORY_UNKNOWN = K
SHUFFLES = 200
TOP_CYCLES = 20
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-01-01", tz="UTC")
EMISSION_FEATURES = (
    "regime_log_activity_3",
    "regime_log_activity_12",
    "regime_activity_acceleration",
    "signed_efficiency_6",
    "signed_efficiency_12",
    "regime_log_bar_range",
    "close_location_value",
    "regime_wick_balance",
    "log_relative_historical_volume",
    "log_relative_cumulative_historical_volume",
    "regime_log_market_dispersion",
    "regime_stock_minus_market_scaled",
    "vti__signed_efficiency_12",
    "regime_market_breadth_centered",
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sealed = load_module(
    "semimarkov_panel_base", HERE / "run_sealed_2025_sec_raw_activity_validation.py"
)
motifs = sealed.pre.base.activity.motifs


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


def stable_phase(value: str, modulus: int) -> int:
    digest = hashlib.sha256(f"semimarkov:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def add_emission_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    activity_3 = pd.to_numeric(frame["mean_abs_return_3"], errors="coerce").clip(
        lower=0.0
    )
    activity_12 = pd.to_numeric(frame["mean_abs_return_12"], errors="coerce").clip(
        lower=0.0
    )
    frame["regime_log_activity_3"] = np.log1p(10000.0 * activity_3)
    frame["regime_log_activity_12"] = np.log1p(10000.0 * activity_12)
    frame["regime_activity_acceleration"] = (
        frame["regime_log_activity_3"] - frame["regime_log_activity_12"]
    )
    frame["regime_log_bar_range"] = np.log1p(
        10000.0
        * pd.to_numeric(frame["bar_range_pct"], errors="coerce").clip(lower=0.0)
    )
    frame["regime_wick_balance"] = pd.to_numeric(
        frame["upper_wick_pct_of_range"], errors="coerce"
    ) - pd.to_numeric(frame["lower_wick_pct_of_range"], errors="coerce")
    frame["regime_log_market_dispersion"] = np.log1p(
        10000.0
        * pd.to_numeric(frame["market_dispersion_return_6"], errors="coerce")
        .abs()
        .clip(lower=0.0)
    )
    denominator = (
        6.0 * activity_12.replace(0.0, np.nan)
    ).clip(lower=1e-8)
    frame["regime_stock_minus_market_scaled"] = np.tanh(
        pd.to_numeric(frame["stock_minus_market_return_6"], errors="coerce")
        / denominator
    )
    frame["regime_market_breadth_centered"] = pd.to_numeric(
        frame["market_breadth_return_6_positive"], errors="coerce"
    ) - 0.5
    forbidden = [name for name in EMISSION_FEATURES if "future" in name.lower()]
    missing = [name for name in EMISSION_FEATURES if name not in frame]
    if forbidden or missing:
        raise AssertionError(
            f"invalid emission manifest; forbidden={forbidden}, missing={missing}"
        )
    return frame


def group_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        group.index.to_numpy(dtype=int)
        for _, group in frame.groupby(["symbol_norm", "session_date"], sort=False)
    ]


def rle(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [
        (int(start), int(end), int(labels[start]))
        for start, end in zip(starts, ends, strict=True)
    ]


def clean_short_runs(
    labels: np.ndarray,
    scaled: np.ndarray,
    groups: list[np.ndarray],
    centroids: np.ndarray,
) -> np.ndarray:
    output = labels.copy()
    for _ in range(2):
        changes = 0
        for positions in groups:
            local = output[positions].copy()
            runs = rle(local)
            for run_index, (start, end, label) in enumerate(runs):
                if end - start >= MIN_CLEAN_RUN:
                    continue
                candidates = []
                if run_index > 0:
                    candidates.append(runs[run_index - 1][2])
                if run_index + 1 < len(runs):
                    candidates.append(runs[run_index + 1][2])
                candidates = sorted(set(candidate for candidate in candidates if candidate != label))
                if not candidates:
                    continue
                values = scaled[positions[start:end]]
                best = min(
                    candidates,
                    key=lambda candidate: float(
                        np.mean(np.square(values - centroids[candidate]))
                    ),
                )
                local[start:end] = best
                changes += 1
            output[positions] = local
        if changes == 0:
            break
    return output


def semantic_remap(labels: np.ndarray, raw: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    summary = pd.DataFrame(
        {
            "state": labels,
            "activity": pd.to_numeric(raw["regime_log_activity_12"], errors="coerce"),
            "direction": pd.to_numeric(raw["signed_efficiency_12"], errors="coerce"),
        }
    ).groupby("state", sort=True).mean()
    order = summary.sort_values(["activity", "direction"], kind="mergesort").index.tolist()
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    remapped = np.asarray([mapping[int(label)] for label in labels], dtype=np.int16)
    return remapped, mapping


def offline_runs(
    frame: pd.DataFrame, labels: np.ndarray, positions_by_session: list[np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_id = 0
    for positions in positions_by_session:
        local_labels = labels[positions]
        for start, end, state in rle(local_labels):
            first = int(positions[start])
            last = int(positions[end - 1])
            rows.append(
                {
                    "run_id": run_id,
                    "symbol_norm": str(frame.at[first, "symbol_norm"]),
                    "session_date": str(frame.at[first, "session_date"]),
                    "state": state,
                    "duration": int(end - start),
                    "start_pos": first,
                    "end_pos": last,
                }
            )
            run_id += 1
    return pd.DataFrame(rows)


def estimate_semimarkov(
    scaled: np.ndarray,
    labels: np.ndarray,
    runs: pd.DataFrame,
) -> dict[str, np.ndarray]:
    means = np.zeros((K, scaled.shape[1]), dtype=float)
    variances = np.zeros_like(means)
    occupancy = np.zeros(K, dtype=float)
    for state in range(K):
        values = scaled[labels == state]
        if len(values) == 0:
            raise AssertionError(f"empty training state {state}")
        means[state] = values.mean(axis=0)
        variances[state] = np.maximum(values.var(axis=0), 0.05)
        occupancy[state] = len(values)
    occupancy = (occupancy + 0.5) / (occupancy.sum() + 0.5 * K)
    duration_hazard = np.zeros((K, MAX_DURATION), dtype=float)
    for state in range(K):
        durations = runs.loc[runs["state"].eq(state), "duration"].to_numpy(dtype=int)
        for age in range(1, MAX_DURATION + 1):
            at_risk = int(np.sum(durations >= age))
            exits = int(np.sum(durations == age))
            if age == MAX_DURATION:
                exits = at_risk
            duration_hazard[state, age - 1] = np.clip(
                (exits + 0.5) / (at_risk + 1.0), 0.01, 1.0
            )
        duration_hazard[state, -1] = 1.0
    transition_counts = np.full((K, K), 0.5, dtype=float)
    np.fill_diagonal(transition_counts, 0.0)
    initial_counts = np.full(K, 0.5, dtype=float)
    for _, group in runs.groupby(["symbol_norm", "session_date"], sort=False):
        states = group["state"].to_numpy(dtype=int)
        if len(states):
            initial_counts[states[0]] += 1.0
        for origin, destination in zip(states[:-1], states[1:], strict=True):
            if origin != destination:
                transition_counts[origin, destination] += 1.0
    transitions = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    initial = initial_counts / initial_counts.sum()
    return {
        "means": means,
        "variances": variances,
        "duration_hazard": duration_hazard,
        "transitions": transitions,
        "initial": initial,
        "occupancy": occupancy,
    }


def log_emission(scaled: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    means = model["means"]
    variances = model["variances"]
    output = np.empty((len(scaled), K), dtype=np.float64)
    constant = np.log(2.0 * np.pi * variances)
    for state in range(K):
        output[:, state] = -0.5 * np.sum(
            constant[state] + np.square(scaled - means[state]) / variances[state], axis=1
        )
    return output


def propagate(alpha: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    hazard = model["duration_hazard"]
    stay = alpha * (1.0 - hazard)
    predicted = np.zeros_like(alpha)
    predicted[:, 1:] += stay[:, :-1]
    predicted[:, -1] += stay[:, -1]
    exit_mass = np.sum(alpha * hazard, axis=1)
    predicted[:, 0] += exit_mass @ model["transitions"]
    total = predicted.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise AssertionError("semi-Markov propagation lost probability mass")
    return predicted / total


def causal_filter(
    emissions: np.ndarray,
    positions_by_session: list[np.ndarray],
    model: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    labels = np.full(len(emissions), -1, dtype=np.int16)
    ages = np.zeros(len(emissions), dtype=np.int16)
    log_likelihood = np.full(len(emissions), np.nan, dtype=float)
    iid_log_likelihood = np.full(len(emissions), np.nan, dtype=float)
    log_iid = np.log(np.clip(model["occupancy"], 1e-15, 1.0))
    for positions in positions_by_session:
        alpha: np.ndarray | None = None
        causal_age = 0
        previous_state = -1
        for position in positions:
            if alpha is None:
                prior = np.zeros((K, MAX_DURATION), dtype=float)
                prior[:, 0] = model["initial"]
            else:
                prior = propagate(alpha, model)
            state_prior = prior.sum(axis=1)
            emission = emissions[position]
            ll = float(
                logsumexp(
                    np.log(np.clip(state_prior, 1e-300, 1.0)) + emission
                )
            )
            iid_ll = float(logsumexp(log_iid + emission))
            scaled_emission = np.exp(emission - np.max(emission))
            posterior = prior * scaled_emission[:, None]
            posterior_sum = posterior.sum()
            if not np.isfinite(posterior_sum) or posterior_sum <= 0.0:
                raise AssertionError("semi-Markov posterior underflow")
            alpha = posterior / posterior_sum
            state = int(np.argmax(alpha.sum(axis=1)))
            causal_age = causal_age + 1 if state == previous_state else 1
            previous_state = state
            labels[position] = state
            ages[position] = min(causal_age, MAX_DURATION)
            log_likelihood[position] = ll
            iid_log_likelihood[position] = iid_ll
    if (labels < 0).any() or np.isnan(log_likelihood).any():
        raise AssertionError("causal filter left an unassigned row")
    hsmm_nll = float(-np.mean(log_likelihood))
    iid_nll = float(-np.mean(iid_log_likelihood))
    return labels, ages, log_likelihood, {
        "semimarkov_nll": hsmm_nll,
        "iid_mixture_nll": iid_nll,
        "relative_nll_improvement": float((iid_nll - hsmm_nll) / iid_nll),
    }


def build_online_runs(
    frame: pd.DataFrame,
    labels: np.ndarray,
    ages: np.ndarray,
    positions_by_session: list[np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    row_run = np.full(len(frame), -1, dtype=np.int32)
    run_id = 0
    for positions in positions_by_session:
        local = labels[positions]
        previous_states: list[int] = []
        for start, end, state in rle(local):
            first = int(positions[start])
            last = int(positions[end - 1])
            row_run[positions[start:end]] = run_id
            rows.append(
                {
                    "run_id": run_id,
                    "symbol_norm": str(frame.at[first, "symbol_norm"]),
                    "session_date": str(frame.at[first, "session_date"]),
                    "month": str(frame.at[first, "session_date"])[:7],
                    "year": int(str(frame.at[first, "session_date"])[:4]),
                    "state": state,
                    "duration": int(end - start),
                    "start_pos": first,
                    "end_pos": last,
                    "start_timestamp": frame.at[first, "timestamp"],
                    "end_timestamp": frame.at[last, "timestamp"],
                    "previous_state_1": previous_states[-1]
                    if len(previous_states) >= 1
                    else HISTORY_UNKNOWN,
                    "previous_state_2": previous_states[-2]
                    if len(previous_states) >= 2
                    else HISTORY_UNKNOWN,
                    "b0_state_numeric": frame.at[last, "b0_state_numeric"],
                    "b0_high_stress": frame.at[last, "b0_high_stress"],
                    "time_sin": frame.at[last, "time_sin"],
                    "time_cos": frame.at[last, "time_cos"],
                }
            )
            previous_states.append(state)
            run_id += 1
    runs = pd.DataFrame(rows)
    runs["next_state"] = runs.groupby(
        ["symbol_norm", "session_date"], sort=False
    )["state"].shift(-1)
    runs["has_next_state"] = runs["next_state"].notna()
    if (row_run < 0).any():
        raise AssertionError("online run mapping left an unassigned row")
    if not np.array_equal(
        np.minimum(ages, MAX_DURATION),
        np.asarray(
            [
                min(
                    int(position - runs.at[row_run[position], "start_pos"] + 1),
                    MAX_DURATION,
                )
                for position in range(len(frame))
            ],
            dtype=np.int16,
        ),
    ):
        raise AssertionError("causal age disagrees with filtered run mapping")
    return runs, row_run


def state_quality(
    frame: pd.DataFrame,
    scaled: np.ndarray,
    labels: np.ndarray,
    runs: pd.DataFrame,
    model: dict[str, np.ndarray],
    likelihood: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    occupancy = pd.Series(labels).value_counts(normalize=True).reindex(range(K), fill_value=0.0)
    for state in range(K):
        values = scaled[labels == state]
        centroid = values.mean(axis=0) if len(values) else np.full(scaled.shape[1], np.nan)
        drift = float(np.sqrt(np.mean(np.square(centroid - model["means"][state]))))
        durations = runs.loc[runs["state"].eq(state), "duration"]
        rows.append(
            {
                "state": state,
                "occupancy": float(occupancy.loc[state]),
                "runs": int(len(durations)),
                "median_duration": float(durations.median()) if len(durations) else math.nan,
                "mean_duration": float(durations.mean()) if len(durations) else math.nan,
                "centroid_drift_scaled_rms": drift,
            }
        )
    report = pd.DataFrame(rows)
    quality = {
        **likelihood,
        "minimum_state_occupancy": float(report["occupancy"].min()),
        "median_run_duration": float(runs["duration"].median()),
        "median_state_centroid_drift": float(report["centroid_drift_scaled_rms"].median()),
        "maximum_state_centroid_drift": float(report["centroid_drift_scaled_rms"].max()),
    }
    return report, quality


def build_hazard_rows(
    frame: pd.DataFrame,
    labels: np.ndarray,
    ages: np.ndarray,
    runs: pd.DataFrame,
    row_run: np.ndarray,
) -> pd.DataFrame:
    future = []
    exact = np.ones(len(frame), dtype=bool)
    grouped_state = pd.Series(labels, index=frame.index).groupby(
        [frame["symbol_norm"], frame["session_date"]], sort=False
    )
    grouped_time = frame.groupby(["symbol_norm", "session_date"], sort=False)[
        "timestamp"
    ]
    for step in range(1, LEAVE_HORIZON + 1):
        future_state = grouped_state.shift(-step)
        future_time = grouped_time.shift(-step)
        future.append(future_state.to_numpy(float))
        exact &= (future_time - frame["timestamp"]).eq(
            pd.Timedelta(minutes=5 * step)
        ).fillna(False).to_numpy(bool)
    matrix = np.column_stack(future)
    leave = np.any(matrix != labels[:, None], axis=1)
    phase = frame["session_date"].astype(str).map(
        lambda value: stable_phase(value, LEAVE_HORIZON)
    )
    primary = frame["bar_index_in_session"].astype(int).mod(LEAVE_HORIZON).eq(phase)
    eligible = exact & primary.to_numpy(bool)
    positions = np.flatnonzero(eligible)
    current = labels[positions].astype(int)
    state_onehot = np.eye(K, dtype=np.float32)[current]
    age = ages[positions].astype(float)
    state_age = np.column_stack(
        (state_onehot, np.log1p(age), age / MAX_DURATION)
    ).astype(np.float32)
    b0 = pd.to_numeric(frame.loc[positions, "b0_state_numeric"], errors="coerce").fillna(0.0)
    stress = pd.to_numeric(frame.loc[positions, "b0_high_stress"], errors="coerce").fillna(0.0)
    time_sin = pd.to_numeric(frame.loc[positions, "time_sin"], errors="coerce").fillna(0.0)
    time_cos = pd.to_numeric(frame.loc[positions, "time_cos"], errors="coerce").fillna(0.0)
    context = np.column_stack((b0, stress, time_sin, time_cos)).astype(np.float32)
    output = frame.loc[
        positions,
        ["symbol_norm", "session_date", "timestamp", "bar_index_in_session"],
    ].copy()
    output["year"] = output["session_date"].astype(str).str[:4].astype(int)
    output["quarter"] = (
        output["year"].astype(str)
        + "_q"
        + pd.to_datetime(output["session_date"]).dt.quarter.astype(str)
    )
    output["leave_within_3"] = leave[positions].astype(int)
    output["state"] = current
    output["age"] = age.astype(int)
    output["run_id"] = row_run[positions]
    output.attrs["state_age_matrix"] = state_age
    output.attrs["context_matrix"] = np.column_stack((state_age, context)).astype(np.float32)
    return output


def binary_probabilities(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    model = LogisticRegression(
        C=0.20, solver="lbfgs", max_iter=500, random_state=SEED
    )
    model.fit(train_x, train_y)
    probability = model.predict_proba(test_x)
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return np.clip(probability[:, positive_index], 1e-9, 1.0 - 1e-9)


def binary_losses(y: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "log_loss": -(
            y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)
        ),
        "brier": np.square(probability - y),
    }


def paired_gate(
    meta: pd.DataFrame,
    candidate_losses: dict[str, np.ndarray],
    baseline_losses: dict[str, np.ndarray],
    comparison: str,
    seed_offset: int,
) -> tuple[pd.DataFrame, bool]:
    rows = []
    for offset, loss_name in enumerate(("log_loss", "brier")):
        difference = candidate_losses[loss_name] - baseline_losses[loss_name]
        daily = pd.DataFrame(
            {
                "session_date": meta["session_date"].astype(str).to_numpy(),
                "difference": difference,
            }
        ).groupby("session_date", sort=True, as_index=False).mean()
        mean, low, high = motifs.moving_block_bounds(
            daily["difference"].to_numpy(float), SEED + seed_offset + offset
        )
        quarter_means = pd.Series(difference).groupby(
            meta["quarter"].astype(str).reset_index(drop=True)
        ).mean()
        deletion = {
            symbol: float(
                np.mean(difference[meta["symbol_norm"].astype(str).ne(symbol).to_numpy()])
            )
            for symbol in sorted(meta["symbol_norm"].astype(str).unique())
        }
        baseline_mean = float(np.mean(baseline_losses[loss_name]))
        rows.append(
            {
                "comparison": comparison,
                "loss": loss_name,
                "row_mean_difference": float(np.mean(difference)),
                "daily_mean_difference": mean,
                "daily_ci_low": low,
                "daily_ci_high": high,
                "baseline_mean_loss": baseline_mean,
                "relative_improvement": float(-np.mean(difference) / baseline_mean),
                "negative_quarter_count": int((quarter_means < 0.0).sum()),
                "leave_one_symbol_max_difference": max(deletion.values()),
                "leave_one_symbol_all_negative": bool(max(deletion.values()) < 0.0),
            }
        )
    result = pd.DataFrame(rows).set_index("loss")
    passed = bool(
        result["daily_ci_high"].lt(0.0).all()
        and result["negative_quarter_count"].ge(3).all()
        and result["leave_one_symbol_all_negative"].all()
        and result.loc["log_loss", "relative_improvement"] >= 0.005
    )
    return result.reset_index(), passed


def destination_matrices(
    runs: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    eligible = runs.loc[runs["has_next_state"]].copy().reset_index(drop=True)
    current = eligible["state"].astype(int).to_numpy()
    onehot = sparse.csr_matrix(np.eye(K, dtype=np.float32)[current])
    duration = eligible["duration"].to_numpy(float)
    numeric = np.column_stack(
        (
            np.log1p(duration),
            np.minimum(duration, MAX_DURATION) / MAX_DURATION,
            pd.to_numeric(eligible["b0_state_numeric"], errors="coerce").fillna(0.0),
            pd.to_numeric(eligible["b0_high_stress"], errors="coerce").fillna(0.0),
            pd.to_numeric(eligible["time_sin"], errors="coerce").fillna(0.0),
            pd.to_numeric(eligible["time_cos"], errors="coerce").fillna(0.0),
        )
    ).astype(np.float32)
    context = sparse.hstack((onehot, sparse.csr_matrix(numeric)), format="csr")
    prev1 = eligible["previous_state_1"].astype(int).to_numpy()
    prev2 = eligible["previous_state_2"].astype(int).to_numpy()
    token = ((prev2 * (K + 1) + prev1) * K + current).astype(int)
    token_count = (K + 1) * (K + 1) * K
    history_token = sparse.csr_matrix(
        (np.ones(len(eligible), dtype=np.float32), (np.arange(len(eligible)), token)),
        shape=(len(eligible), token_count),
    )
    history = sparse.hstack((context, history_token), format="csr")
    target = eligible["next_state"].astype(int).to_numpy()
    return context, history, target, eligible


def multiclass_probability(
    train_x: sparse.csr_matrix,
    train_y: np.ndarray,
    test_x: sparse.csr_matrix,
) -> np.ndarray:
    model = LogisticRegression(
        C=0.20, solver="lbfgs", max_iter=500, random_state=SEED
    )
    model.fit(train_x, train_y)
    raw = model.predict_proba(test_x)
    output = np.full((test_x.shape[0], K), 1e-9)
    for index, state in enumerate(model.classes_):
        output[:, int(state)] = raw[:, index]
    output /= output.sum(axis=1, keepdims=True)
    return np.clip(output, 1e-9, 1.0)


def multiclass_losses(y: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    chosen = probability[np.arange(len(y)), y]
    onehot = np.eye(K)[y]
    return {
        "log_loss": -np.log(chosen),
        "brier": np.sum(np.square(probability - onehot), axis=1),
    }


def canonical_cycle(core: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in core)
    rotations = [values[index:] + values[:index] for index in range(len(values))]
    return min(rotations)


def session_sequences(runs: pd.DataFrame) -> list[dict[str, Any]]:
    sequences = []
    for (symbol, session_date), group in runs.groupby(
        ["symbol_norm", "session_date"], sort=False
    ):
        sequences.append(
            {
                "symbol_norm": str(symbol),
                "session_date": str(session_date),
                "month": str(session_date)[:7],
                "states": group["state"].astype(int).tolist(),
                "durations": group["duration"].astype(int).tolist(),
            }
        )
    return sequences


def count_cycles(
    sequences: list[dict[str, Any]], candidates: set[tuple[int, ...]] | None = None
) -> tuple[Counter[tuple[int, ...]], dict[tuple[int, ...], set[str]], dict[tuple[int, ...], set[str]]]:
    counts: Counter[tuple[int, ...]] = Counter()
    symbols: dict[tuple[int, ...], set[str]] = defaultdict(set)
    months: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for sequence in sequences:
        states = sequence["states"]
        for transitions in range(2, 6):
            width = transitions + 1
            for start in range(0, len(states) - width + 1):
                window = states[start : start + width]
                if window[0] != window[-1]:
                    continue
                key = canonical_cycle(window[:-1])
                if candidates is not None and key not in candidates:
                    continue
                counts[key] += 1
                symbols[key].add(sequence["symbol_norm"])
                months[key].add(sequence["month"])
    return counts, symbols, months


def shuffled_cycle_null(
    sequences: list[dict[str, Any]], candidates: list[tuple[int, ...]]
) -> dict[tuple[int, ...], np.ndarray]:
    output = {candidate: np.zeros(SHUFFLES, dtype=int) for candidate in candidates}
    candidate_set = set(candidates)
    rng = np.random.default_rng(SEED + 31000)
    for shuffle in range(SHUFFLES):
        shuffled = []
        for sequence in sequences:
            order = rng.permutation(len(sequence["states"]))
            states = [sequence["states"][int(index)] for index in order]
            compressed = [states[0]] if states else []
            for state in states[1:]:
                if state != compressed[-1]:
                    compressed.append(state)
            shuffled.append({**sequence, "states": compressed})
        counts, _, _ = count_cycles(shuffled, candidate_set)
        for candidate in candidates:
            output[candidate][shuffle] = int(counts.get(candidate, 0))
    return output


def self_tests() -> None:
    assert canonical_cycle((2, 5, 1)) == canonical_cycle((5, 1, 2))
    model = {
        "duration_hazard": np.asarray([[0.1, 0.5, 1.0], [0.2, 0.4, 1.0]]),
        "transitions": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
    }
    alpha = np.zeros((2, 3), dtype=float)
    alpha[0, 0] = 1.0
    propagated = propagate(alpha, model)
    assert abs(float(propagated.sum()) - 1.0) < 1e-12
    assert propagated[0, 1] > 0.0 and propagated[1, 0] > 0.0


def main() -> None:
    self_tests()
    OUT.mkdir(parents=True, exist_ok=True)
    panel, _, _, panel_audit = sealed.prepare_panel()
    panel = add_emission_features(panel)
    if panel["timestamp"].max() >= TEST_END:
        raise AssertionError("2026 row entered regime-loop harness")
    train_mask = panel["timestamp"].lt(TRAIN_END).to_numpy(bool)
    test_mask = panel["timestamp"].ge(TRAIN_END).to_numpy(bool)
    if not train_mask.any() or not test_mask.any():
        raise AssertionError("empty 2024/2025 split")
    raw_features = panel.loc[:, list(EMISSION_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    train_imputed = imputer.fit_transform(raw_features.loc[train_mask])
    train_scaled = scaler.fit_transform(train_imputed).astype(np.float32)
    all_scaled = scaler.transform(imputer.transform(raw_features)).astype(np.float32)
    train_frame = panel.loc[train_mask].reset_index(drop=True)
    test_frame = panel.loc[test_mask].reset_index(drop=True)
    train_groups = group_positions(train_frame)
    test_groups = group_positions(test_frame)
    sample_step = max(1, len(train_scaled) // 200000)
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=4096,
        n_init=10,
        max_iter=300,
        random_state=SEED,
    )
    kmeans.fit(train_scaled[::sample_step])
    raw_train_labels = kmeans.predict(train_scaled).astype(np.int16)
    clean_train_labels = clean_short_runs(
        raw_train_labels, train_scaled, train_groups, kmeans.cluster_centers_
    )
    clean_train_labels, mapping = semantic_remap(clean_train_labels, train_frame)
    offline = offline_runs(train_frame, clean_train_labels, train_groups)
    model = estimate_semimarkov(train_scaled, clean_train_labels, offline)
    train_emissions = log_emission(train_scaled, model)
    test_scaled = all_scaled[test_mask]
    test_emissions = log_emission(test_scaled, model)
    train_labels, train_ages, _, train_likelihood = causal_filter(
        train_emissions, train_groups, model
    )
    test_labels, test_ages, _, test_likelihood = causal_filter(
        test_emissions, test_groups, model
    )
    train_runs, train_row_run = build_online_runs(
        train_frame, train_labels, train_ages, train_groups
    )
    test_runs, test_row_run = build_online_runs(
        test_frame, test_labels, test_ages, test_groups
    )
    train_state_report, train_quality = state_quality(
        train_frame, train_scaled, train_labels, train_runs, model, train_likelihood
    )
    test_state_report, test_quality = state_quality(
        test_frame, test_scaled, test_labels, test_runs, model, test_likelihood
    )
    h1_pass = bool(
        train_quality["minimum_state_occupancy"] >= 0.01
        and test_quality["minimum_state_occupancy"] >= 0.01
        and train_quality["median_run_duration"] >= 2.0
        and test_quality["median_run_duration"] >= 2.0
        and test_quality["relative_nll_improvement"] >= 0.005
        and test_quality["median_state_centroid_drift"] <= 1.5
        and test_quality["maximum_state_centroid_drift"] <= 3.0
    )

    hazard_train = build_hazard_rows(
        train_frame, train_labels, train_ages, train_runs, train_row_run
    )
    hazard_test = build_hazard_rows(
        test_frame, test_labels, test_ages, test_runs, test_row_run
    )
    hazard_train_state = hazard_train.attrs["state_age_matrix"]
    hazard_train_context = hazard_train.attrs["context_matrix"]
    hazard_test_state = hazard_test.attrs["state_age_matrix"]
    hazard_test_context = hazard_test.attrs["context_matrix"]
    hazard_y_train = hazard_train["leave_within_3"].to_numpy(dtype=int)
    hazard_y_test = hazard_test["leave_within_3"].to_numpy(dtype=int)
    hazard_state_probability = binary_probabilities(
        hazard_train_state, hazard_y_train, hazard_test_state
    )
    hazard_context_probability = binary_probabilities(
        hazard_train_context, hazard_y_train, hazard_test_context
    )
    hazard_state_losses = binary_losses(hazard_y_test, hazard_state_probability)
    hazard_context_losses = binary_losses(hazard_y_test, hazard_context_probability)
    hazard_gate, h2_pass = paired_gate(
        hazard_test.reset_index(drop=True),
        hazard_context_losses,
        hazard_state_losses,
        "context_vs_state_age",
        33000,
    )

    combined_runs = pd.concat((train_runs, test_runs), ignore_index=True)
    context_matrix, history_matrix, destination, destination_meta = destination_matrices(
        combined_runs
    )
    destination_train_mask = destination_meta["year"].eq(2024).to_numpy()
    destination_test_mask = destination_meta["year"].eq(2025).to_numpy()
    destination_context_probability = multiclass_probability(
        context_matrix[destination_train_mask],
        destination[destination_train_mask],
        context_matrix[destination_test_mask],
    )
    destination_history_probability = multiclass_probability(
        history_matrix[destination_train_mask],
        destination[destination_train_mask],
        history_matrix[destination_test_mask],
    )
    destination_y_test = destination[destination_test_mask]
    destination_context_losses = multiclass_losses(
        destination_y_test, destination_context_probability
    )
    destination_history_losses = multiclass_losses(
        destination_y_test, destination_history_probability
    )
    destination_test_meta = destination_meta.loc[destination_test_mask].reset_index(drop=True)
    destination_test_meta["quarter"] = (
        destination_test_meta["year"].astype(str)
        + "_q"
        + pd.to_datetime(destination_test_meta["session_date"]).dt.quarter.astype(str)
    )
    destination_gate, h3_pass = paired_gate(
        destination_test_meta,
        destination_history_losses,
        destination_context_losses,
        "order3_history_vs_context",
        34000,
    )

    train_sequences = session_sequences(train_runs)
    test_sequences = session_sequences(test_runs)
    train_counts, train_symbols, train_months = count_cycles(train_sequences)
    eligible_cycles = [
        cycle
        for cycle, count in train_counts.most_common()
        if count >= 30
        and len(train_symbols[cycle]) >= 5
        and len(train_months[cycle]) >= 6
    ][:TOP_CYCLES]
    test_counts, test_symbols, test_months = count_cycles(
        test_sequences, set(eligible_cycles)
    )
    null = shuffled_cycle_null(test_sequences, eligible_cycles)
    cycle_rows = []
    for cycle in eligible_cycles:
        distribution = null[cycle]
        null_q99 = float(np.quantile(distribution, 0.99, method="higher"))
        observed = int(test_counts.get(cycle, 0))
        support_pass = bool(
            observed >= 30
            and len(test_symbols[cycle]) >= 8
            and len(test_months[cycle]) >= 6
            and observed > null_q99
        )
        cycle_rows.append(
            {
                "cycle": "->".join(map(str, (*cycle, cycle[0]))),
                "transition_length": len(cycle),
                "train_occurrences": int(train_counts[cycle]),
                "train_symbols": int(len(train_symbols[cycle])),
                "train_months": int(len(train_months[cycle])),
                "test_occurrences": observed,
                "test_symbols": int(len(test_symbols[cycle])),
                "test_months": int(len(test_months[cycle])),
                "null_mean": float(distribution.mean()),
                "null_q95": float(np.quantile(distribution, 0.95)),
                "null_q99": null_q99,
                "shuffle_exceedance_p": float(
                    (1 + np.sum(distribution >= observed)) / (1 + SHUFFLES)
                ),
                "support_and_null_pass": support_pass,
            }
        )
    cycles = pd.DataFrame(cycle_rows)
    cycle_null_pass = bool(
        not cycles.empty and cycles["support_and_null_pass"].any()
    )
    h4_pass = bool(h3_pass and cycle_null_pass)
    loop_pass = bool(h1_pass and h3_pass and h4_pass)

    state_assignments = pd.concat(
        (
            train_frame[
                ["symbol_norm", "session_date", "timestamp", "bar_index_in_session", "b0_state_numeric"]
            ].assign(year=2024, state=train_labels, age=train_ages),
            test_frame[
                ["symbol_norm", "session_date", "timestamp", "bar_index_in_session", "b0_state_numeric"]
            ].assign(year=2025, state=test_labels, age=test_ages),
        ),
        ignore_index=True,
    )
    state_assignments.to_parquet(OUT / "causal_state_assignments.parquet", index=False)
    train_runs.to_csv(OUT / "train_2024_filtered_runs.csv", index=False)
    test_runs.to_csv(OUT / "test_2025_filtered_runs.csv", index=False)
    train_state_report.assign(year=2024).to_csv(OUT / "train_state_quality.csv", index=False)
    test_state_report.assign(year=2025).to_csv(OUT / "test_state_quality.csv", index=False)
    hazard_gate.to_csv(OUT / "leave_hazard_gates.csv", index=False)
    destination_gate.to_csv(OUT / "destination_history_gates.csv", index=False)
    cycles.to_csv(OUT / "fixed_cycle_shuffled_nulls.csv", index=False)
    pd.DataFrame(
        {
            "feature": EMISSION_FEATURES,
            "imputer_median": imputer.statistics_,
            "scaler_center": scaler.center_,
            "scaler_scale": scaler.scale_,
        }
    ).to_csv(OUT / "frozen_emission_preprocessing.csv", index=False)
    np.savez_compressed(
        OUT / "frozen_semimarkov_parameters.npz",
        **model,
        semantic_old_state=np.asarray(list(mapping.keys()), dtype=int),
        semantic_new_state=np.asarray(list(mapping.values()), dtype=int),
    )
    result = {
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
        "edge_claimed": False,
        "repo_runtime_files_changed": False,
        "deployment_performed": False,
        "training_period": "2024",
        "test_period": "2025",
        "2026_outcomes_opened": False,
        "data_source": "EODHD regular-session five-minute historical provider OHLCV plus VTI",
        "volume_label": "historical_volume",
        "volume_is_exchange_wide": False,
        "state_count": K,
        "maximum_duration_bars": MAX_DURATION,
        "emission_features": list(EMISSION_FEATURES),
        "panel_audit": panel_audit,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_runs": int(len(train_runs)),
        "test_runs": int(len(test_runs)),
        "train_state_quality": train_quality,
        "test_state_quality": test_quality,
        "h1_stable_causal_regimes_pass": h1_pass,
        "hazard_train_rows": int(len(hazard_train)),
        "hazard_test_rows": int(len(hazard_test)),
        "h2_duration_context_hazard_pass": h2_pass,
        "destination_train_transitions": int(destination_train_mask.sum()),
        "destination_test_transitions": int(destination_test_mask.sum()),
        "h3_order3_destination_history_pass": h3_pass,
        "train_selected_cycles": int(len(eligible_cycles)),
        "cycles_passing_support_and_null": int(
            cycles["support_and_null_pass"].sum() if len(cycles) else 0
        ),
        "h4_recurrent_cycle_pass": h4_pass,
        "regime_loop_hypothesis_pass": loop_pass,
        "decision": (
            "retain_causal_regime_loop_hypothesis_for_sealed_2026_validation"
            if loop_pass
            else "reject_causal_regime_loop_hypothesis_under_frozen_2025_gates"
        ),
        "duration_hazard_decision": (
            "retain_duration_context_hazard_as_separate_research_hypothesis"
            if h2_pass
            else "reject_duration_context_hazard_increment"
        ),
        "null": {
            "type": "within-session shuffled filtered-run order",
            "shuffles": SHUFFLES,
            "cycle_transition_lengths": [2, 3, 4, 5],
            "train_selected_top_n": TOP_CYCLES,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(safe(result), indent=2))
    (OUT / "summary.md").write_text(
        "# Causal semi-Markov regime and loop detection\n\n"
        "Research-only 2024-train/2025-test investigation. 2026 remained sealed.\n\n"
        f"- Stable causal regimes passed: `{h1_pass}`\n"
        f"- Duration/context leave hazard passed: `{h2_pass}`\n"
        f"- Order-3 destination history passed: `{h3_pass}`\n"
        f"- Fixed recurrent cycle/null gate passed: `{h4_pass}`\n"
        f"- Overall regime-loop hypothesis passed: `{loop_pass}`\n"
        f"- Decision: `{result['decision']}`\n"
        "- Safety: live ordering and order placement disabled; no deployment\n"
    )
    print(json.dumps(safe(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
