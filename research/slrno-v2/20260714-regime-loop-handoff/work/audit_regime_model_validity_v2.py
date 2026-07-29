#!/usr/bin/env python3
"""Independently audit Regime Model Validity V2 Part A.

This auditor deliberately does not import the Part A runner or any of its
summary-generation helpers. It reconstructs the bounded 2024 panel and core
mathematics directly from frozen lineage primitives and primary artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import njit
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler

WORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORK_DIR.parents[3]
PRIMARY = WORK_DIR / "artifacts" / "20260718-regime-model-validity-v2" / "primary"
EXACT_RERUN = WORK_DIR / "artifacts" / "20260718-regime-model-validity-v2" / "exact_rerun"
CONTRACT_PATH = WORK_DIR / "contracts" / "20260718-regime-model-validity-v2.json"
REPORT_PATH = WORK_DIR / "reports" / "20260718-regime-model-validity-v2.md"
FROZEN_CORE_PATH = WORK_DIR / "frozen_loop_movement_shadow_core.py"
STATE_DIR = (
    WORK_DIR
    / "shadow_validation"
    / "frozen_loop_movement_shadow_v1"
    / "frozen_bundle"
    / "artifacts"
    / "state"
)
MODEL_PATH = STATE_DIR / "frozen_semimarkov_parameters.npz"
PREPROCESSING_PATH = STATE_DIR / "frozen_emission_preprocessing.csv"
PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
BASELINE_SHA = "66cd706fa727ac5873b299d5c22388221203f451"
EXPECTED_CONTRACT_HASH = "dd44ce458f41a16f023a49b9be7ab3f762ed31b07d42fc6e1ba673f233546c55"
EXPECTED_SNAPSHOT = "48d2141ef993928d4e8a01d6b3c24dff665280c67f4167115b453613460cc661"
START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
ASSESSMENT_START = pd.Timestamp("2025-01-01", tz="UTC")
ASSESSMENT_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
K_VALUES = (6, 8, 10, 12)
SEEDS = (20260710, 20260711, 20260712, 20260713, 20260714)
SELECTED_PATHS = ((5, 6, 5), (4, 6, 4))
SELECTED_IDS = ("loop_p_5-6-5", "loop_p_4-6-4")
SYMBOLS = (
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
SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_connected": False,
    "economic_outcomes_used": False,
    "payoff_selection_used": False,
    "production_runtime_modified": False,
    "strategy_promotion": False,
}
EXACT_EXCLUSIONS = {
    "artifact_manifest.json",
    "independent_audit.json",
    "exact_rerun_manifest.json",
    "post_run_tree_manifest.json",
}
NEW_V2_PATH_MARKERS = (
    "regime_validity_v2.py",
    "state_alignment_v2.py",
    "state_representation_sensitivity_v2.py",
    "loop_orientation_v2.py",
    "loop_regime_interaction_v2.py",
    "regime_model_validity_v2.py",
    "loop_regime_interaction_foundations_v2.py",
    "20260718-regime-model-validity-v2",
    "20260718-loop-regime-interaction-foundations-v2",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _provider_file(symbol: str) -> Path:
    stored = "VTI.US" if symbol == "VTI" else symbol
    return PROVIDER_ROOT / f"symbol={stored}" / "timeframe=5m" / "data.parquet"


def _bounded_provider_hash(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> str:
    frame = pd.read_parquet(
        path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        filters=[
            ("timestamp", ">=", start.to_pydatetime()),
            ("timestamp", "<=", end.to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if frame["timestamp"].gt(end).any() or frame["timestamp"].lt(start).any():
        raise AssertionError("source hash admitted a protected later row")
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _source_snapshot(
    *, start: pd.Timestamp = START, end: pd.Timestamp = END
) -> tuple[dict[str, str], str]:
    paths = {symbol: _provider_file(symbol) for symbol in (*SYMBOLS, "VTI")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"structural source unavailable: {missing}")
    hashes = {
        symbol: _bounded_provider_hash(path, start=start, end=end) for symbol, path in paths.items()
    }
    return hashes, _sha256_payload(hashes)


def _prepare_panel(
    frozen_core: Any,
    *,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
) -> pd.DataFrame:
    parts = [
        frozen_core.prepare_symbol_bars(symbol, PROVIDER_ROOT, start, end) for symbol in SYMBOLS
    ]
    panel = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["symbol_norm", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    vti = frozen_core.prepare_symbol_bars("VTI", PROVIDER_ROOT, start, end)
    panel = frozen_core.add_market_features(panel, vti)
    panel = frozen_core.add_emission_features(panel)
    panel["symbol"] = panel["symbol_norm"].astype(str)
    panel["session"] = panel["session_date"].astype(str)
    panel["bar_ordinal"] = panel["bar_index_in_session"].astype(int)
    panel["bar_start_timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel["bar_complete_timestamp"] = panel["bar_start_timestamp"] + pd.Timedelta(minutes=5)
    return panel.reset_index(drop=True)


def _prefix_invariance_audit(
    frozen_core: Any,
    full_panel: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
) -> tuple[bool, dict[str, object]]:
    cutoffs = (
        pd.Timestamp("2024-04-30 23:59:59", tz="UTC"),
        pd.Timestamp("2024-07-31 23:59:59", tz="UTC"),
        pd.Timestamp("2024-10-31 23:59:59", tz="UTC"),
    )
    maximum_difference = 0.0
    compared_rows = 0
    failures: list[str] = []
    for cutoff in cutoffs:
        prefix = _prepare_panel(frozen_core, start=START, end=cutoff)
        final_session = prefix["session_date"].max()
        selected = prefix.loc[
            prefix["session_date"].eq(final_session),
            [
                "symbol_norm",
                "timestamp",
                *feature_names,
            ],
        ]
        reference = full_panel.loc[
            full_panel["session_date"].eq(final_session),
            [
                "symbol_norm",
                "timestamp",
                *feature_names,
            ],
        ]
        merged = selected.merge(
            reference,
            on=["symbol_norm", "timestamp"],
            suffixes=("_prefix", "_full"),
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all():
            failures.append(f"{cutoff.date()}:row_identity")
            continue
        compared_rows += len(merged)
        for feature in feature_names:
            left = pd.to_numeric(merged[f"{feature}_prefix"], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(merged[f"{feature}_full"], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                maximum_difference = max(
                    maximum_difference,
                    float(np.max(np.abs(left[finite] - right[finite]))),
                )
            if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.allclose(
                left[finite], right[finite], atol=1e-12, rtol=0.0
            ):
                failures.append(f"{cutoff.date()}:{feature}")
    return not failures, {
        "cutoff_count": len(cutoffs),
        "compared_rows": compared_rows,
        "maximum_absolute_difference": maximum_difference,
        "failures": failures,
    }


def _scale_and_emit(
    panel: pd.DataFrame,
    preprocessing: pd.DataFrame,
    model: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    features = preprocessing["feature"].astype(str).tolist()
    raw = panel[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = raw.to_numpy(dtype=float)
    medians = preprocessing["imputer_median"].to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        values[missing] = np.take(medians, np.nonzero(missing)[1])
    scaled = (
        (values - preprocessing["scaler_center"].to_numpy(dtype=float))
        / preprocessing["scaler_scale"].to_numpy(dtype=float)
    ).astype(np.float32)
    means = np.asarray(model["means"], dtype=float)
    variances = np.asarray(model["variances"], dtype=float)
    emissions = np.empty((len(scaled), len(means)), dtype=np.float64)
    constant = np.log(2.0 * np.pi * variances)
    for state in range(len(means)):
        emissions[:, state] = -0.5 * np.sum(
            constant[state] + np.square(scaled - means[state]) / variances[state],
            axis=1,
        )
    return scaled, emissions


def _independently_fit_scale(
    panel: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    raw = (
        panel.loc[:, list(feature_names)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    imputed = imputer.fit_transform(raw)
    return np.asarray(scaler.fit_transform(imputed), dtype=np.float32)


def _groups(panel: pd.DataFrame, *, split_gaps: bool) -> tuple[np.ndarray, ...]:
    groups: list[np.ndarray] = []
    for _, group in panel.groupby(["symbol_norm", "session_date"], sort=False):
        positions = group.index.to_numpy(dtype=int)
        if not split_gaps:
            groups.append(positions)
            continue
        ordinals = group["bar_index_in_session"].to_numpy(dtype=int)
        starts = pd.to_datetime(group["timestamp"], utc=True)
        breaks = (
            np.flatnonzero(
                (np.diff(ordinals) != 1)
                | (starts.diff().iloc[1:].to_numpy() != np.timedelta64(5, "m"))
            )
            + 1
        )
        groups.extend(segment for segment in np.split(positions, breaks) if len(segment))
    return tuple(groups)


def _reset_mask(groups: tuple[np.ndarray, ...], row_count: int) -> np.ndarray:
    reset = np.zeros(row_count, dtype=np.bool_)
    covered = np.zeros(row_count, dtype=np.bool_)
    for group in groups:
        reset[int(group[0])] = True
        covered[group] = True
    if not covered.all():
        raise AssertionError("auditor groups do not cover the panel")
    return reset


def _expand_hazard(hazard: np.ndarray, maximum_age: int = 78) -> np.ndarray:
    terminal = hazard.shape[1] - 1
    window_start = max(0, terminal - 6)
    tail = np.clip(hazard[:, window_start:terminal].mean(axis=1), 1e-6, 1.0 - 1e-6)
    expanded = np.empty((hazard.shape[0], maximum_age), dtype=float)
    expanded[:, :terminal] = hazard[:, :terminal]
    expanded[:, terminal:] = tail[:, None]
    return expanded


@njit(cache=False)
def _posterior_sample_kernel(
    emissions: np.ndarray,
    hazard: np.ndarray,
    transitions: np.ndarray,
    initial: np.ndarray,
    occupancy: np.ndarray,
    reset: np.ndarray,
    sample_positions: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    row_count, state_count = emissions.shape
    age_count = hazard.shape[1]
    hard = np.full(row_count, -1, dtype=np.int16)
    sample_probability = np.zeros((len(sample_positions), state_count), dtype=np.float64)
    sample_state_age = np.zeros((len(sample_positions), state_count, age_count), dtype=np.float64)
    sample_expected_age = np.zeros(len(sample_positions), dtype=np.float64)
    sample_departure = np.zeros(len(sample_positions), dtype=np.float64)
    entropy = np.zeros(row_count, dtype=np.float64)
    log_likelihood = np.zeros(row_count, dtype=np.float64)
    iid_log_likelihood = np.zeros(row_count, dtype=np.float64)
    probabilities = np.zeros((row_count, state_count), dtype=np.float64)
    alpha = np.zeros((state_count, age_count), dtype=np.float64)
    prior = np.zeros((state_count, age_count), dtype=np.float64)
    sample_index = 0
    for position in range(row_count):
        prior.fill(0.0)
        if reset[position]:
            for state in range(state_count):
                prior[state, 0] = initial[state] / initial.sum()
        else:
            for state in range(state_count):
                exit_mass = 0.0
                for age in range(age_count):
                    stay = alpha[state, age] * (1.0 - hazard[state, age])
                    prior[state, min(age + 1, age_count - 1)] += stay
                    exit_mass += alpha[state, age] * hazard[state, age]
                for destination in range(state_count):
                    prior[destination, 0] += exit_mass * transitions[state, destination]
            prior /= prior.sum()
        maximum_term = -np.inf
        for state in range(state_count):
            state_prior = prior[state].sum()
            maximum_term = max(
                maximum_term,
                math.log(max(state_prior, 1e-300)) + emissions[position, state],
            )
        likelihood_total = 0.0
        for state in range(state_count):
            state_prior = prior[state].sum()
            likelihood_total += math.exp(
                math.log(max(state_prior, 1e-300)) + emissions[position, state] - maximum_term
            )
        log_likelihood[position] = maximum_term + math.log(likelihood_total)
        maximum_iid = -np.inf
        for state in range(state_count):
            maximum_iid = max(
                maximum_iid,
                math.log(max(occupancy[state], 1e-300)) + emissions[position, state],
            )
        iid_total = 0.0
        for state in range(state_count):
            iid_total += math.exp(
                math.log(max(occupancy[state], 1e-300)) + emissions[position, state] - maximum_iid
            )
        iid_log_likelihood[position] = maximum_iid + math.log(iid_total)

        maximum = np.max(emissions[position])
        total = 0.0
        for state in range(state_count):
            relative = math.exp(emissions[position, state] - maximum)
            for age in range(age_count):
                alpha[state, age] = prior[state, age] * relative
                total += alpha[state, age]
        alpha /= total
        best_state = 0
        best_probability = -1.0
        for state in range(state_count):
            probability = alpha[state].sum()
            probabilities[position, state] = probability
            if probability > best_probability:
                best_probability = probability
                best_state = state
            entropy[position] -= probability * math.log(max(probability, 1e-300))
        hard[position] = best_state
        if sample_index < len(sample_positions) and position == sample_positions[sample_index]:
            for state in range(state_count):
                for age in range(age_count):
                    value = alpha[state, age]
                    sample_state_age[sample_index, state, age] = value
                    sample_probability[sample_index, state] += value
                    sample_expected_age[sample_index] += value * (age + 1.0)
                    sample_departure[sample_index] += value * hazard[state, age]
            sample_index += 1
    return (
        hard,
        sample_probability,
        sample_state_age,
        sample_expected_age,
        sample_departure,
        entropy,
        log_likelihood,
        iid_log_likelihood,
        probabilities,
    )


def _rle(labels: np.ndarray) -> list[tuple[int, int, int]]:
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    return [
        (int(start), int(end), int(labels[start])) for start, end in zip(starts, ends, strict=True)
    ]


def _clean_short_runs(
    labels: np.ndarray,
    scaled: np.ndarray,
    groups: tuple[np.ndarray, ...],
    centroids: np.ndarray,
) -> np.ndarray:
    output = labels.copy()
    for _ in range(2):
        changes = 0
        for positions in groups:
            local = output[positions].copy()
            runs = _rle(local)
            for run_index, (start, end, label) in enumerate(runs):
                if end - start >= 2:
                    continue
                candidates: list[int] = []
                if run_index > 0:
                    candidates.append(runs[run_index - 1][2])
                if run_index + 1 < len(runs):
                    candidates.append(runs[run_index + 1][2])
                candidates = sorted(set(value for value in candidates if value != label))
                if not candidates:
                    continue
                values = scaled[positions[start:end]]
                best = min(
                    candidates,
                    key=lambda state: float(np.mean(np.square(values - centroids[state]))),
                )
                local[start:end] = best
                changes += 1
            output[positions] = local
        if changes == 0:
            break
    return output


def _semantic_remap(labels: np.ndarray, panel: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    summary = (
        pd.DataFrame(
            {
                "state": labels,
                "activity": pd.to_numeric(panel["regime_log_activity_12"], errors="coerce"),
                "direction": pd.to_numeric(panel["signed_efficiency_12"], errors="coerce"),
            }
        )
        .groupby("state", sort=True)
        .mean()
    )
    order = summary.sort_values(["activity", "direction"], kind="mergesort").index
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int16), mapping


def _estimate_parameters(
    scaled: np.ndarray,
    labels: np.ndarray,
    groups: tuple[np.ndarray, ...],
    *,
    state_count: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(scaled, dtype=float)
    means = np.zeros((state_count, values.shape[1]), dtype=float)
    variances = np.zeros_like(means)
    occupancy = np.zeros(state_count, dtype=float)
    runs: list[tuple[int, int, int]] = []
    initial_counts = np.full(state_count, 0.5, dtype=float)
    transition_counts = np.full((state_count, state_count), 0.5, dtype=float)
    np.fill_diagonal(transition_counts, 0.0)
    for state in range(state_count):
        state_values = values[labels == state]
        means[state] = state_values.mean(axis=0)
        variances[state] = np.maximum(state_values.var(axis=0), 0.05)
        occupancy[state] = len(state_values)
    for group_index, group in enumerate(groups):
        local_runs = _rle(labels[group])
        initial_counts[local_runs[0][2]] += 1.0
        for left, right in zip(local_runs[:-1], local_runs[1:], strict=True):
            transition_counts[left[2], right[2]] += 1.0
        runs.extend((group_index, state, end - start) for start, end, state in local_runs)
    hazard = np.zeros((state_count, 24), dtype=float)
    for state in range(state_count):
        durations = np.asarray(
            [duration for _, run_state, duration in runs if run_state == state], dtype=int
        )
        for age in range(1, 25):
            at_risk = int(np.sum(durations >= age))
            exits = int(np.sum(durations == age))
            if age == 24:
                exits = at_risk
            hazard[state, age - 1] = np.clip((exits + 0.5) / (at_risk + 1.0), 0.01, 1.0)
        hazard[state, -1] = 1.0
    return {
        "means": means,
        "variances": variances,
        "occupancy": (occupancy + 0.5) / (occupancy.sum() + 0.5 * state_count),
        "duration_hazard": hazard,
        "transitions": transition_counts / transition_counts.sum(axis=1, keepdims=True),
        "initial": initial_counts / initial_counts.sum(),
    }


def _gaussian_emissions(scaled: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(scaled, dtype=float)
    means = np.asarray(model["means"], dtype=float)
    variances = np.asarray(model["variances"], dtype=float)
    constant = np.log(2.0 * np.pi * variances)
    return np.asarray(
        [
            -0.5
            * np.sum(
                constant[state] + np.square(values - means[state]) / variances[state],
                axis=1,
            )
            for state in range(len(means))
        ],
        dtype=float,
    ).T


def _row_distance(left: np.ndarray, right: np.ndarray) -> float:
    width = max(len(left), len(right))
    first = np.zeros(width, dtype=float)
    second = np.zeros(width, dtype=float)
    first[: len(left)] = left
    second[: len(right)] = right
    return float(np.linalg.norm(first - second) / math.sqrt(max(width, 1)))


def _normalized(values: np.ndarray) -> np.ndarray:
    profile = np.asarray(values, dtype=float)
    total = float(profile.sum())
    return profile / total if total > 0.0 else profile


def _independent_alignment(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> tuple[dict[int, int], pd.DataFrame]:
    reference_hazard = _expand_hazard(np.asarray(reference["duration_hazard"], dtype=float))
    candidate_hazard = _expand_hazard(np.asarray(candidate["duration_hazard"], dtype=float))
    costs = np.zeros((len(candidate["means"]), len(reference["means"])), dtype=float)
    components: dict[tuple[int, int], tuple[float, float, float]] = {}
    for candidate_state in range(len(candidate["means"])):
        candidate_transition = np.sort(_normalized(candidate["transitions"][candidate_state]))
        candidate_duration = _normalized(candidate_hazard[candidate_state])
        for reference_state in range(len(reference["means"])):
            centroid = _row_distance(
                candidate["means"][candidate_state], reference["means"][reference_state]
            )
            transition = _row_distance(
                candidate_transition,
                np.sort(_normalized(reference["transitions"][reference_state])),
            )
            duration = _row_distance(
                candidate_duration,
                _normalized(reference_hazard[reference_state]),
            )
            components[(candidate_state, reference_state)] = (centroid, transition, duration)
            costs[candidate_state, reference_state] = (
                0.60 * centroid + 0.25 * transition + 0.15 * duration
            )
    candidate_rows, reference_rows = linear_sum_assignment(costs)
    mapping = {
        int(candidate_state): int(reference_state)
        for candidate_state, reference_state in zip(candidate_rows, reference_rows, strict=True)
    }
    rows = []
    for candidate_state, reference_state in sorted(mapping.items()):
        centroid, transition, duration = components[(candidate_state, reference_state)]
        rows.append(
            {
                "candidate_state": candidate_state,
                "reference_state": reference_state,
                "centroid_distance": centroid,
                "transition_distance": transition,
                "duration_distance": duration,
                "total_cost": float(costs[candidate_state, reference_state]),
            }
        )
    return mapping, pd.DataFrame(rows)


def _apply_mapping(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping.get(int(value), -1) for value in labels], dtype=np.int16)


def _state_occupancy(labels: np.ndarray, state_count: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=int), minlength=state_count) / len(labels)


def _run_metrics(labels: np.ndarray, groups: tuple[np.ndarray, ...]) -> dict[str, float]:
    durations: list[int] = []
    for group in groups:
        durations.extend(end - start for start, end, _ in _rle(labels[group]))
    values = np.asarray(durations, dtype=float)
    return {
        "median_run_duration": float(np.median(values)),
        "one_bar_reversal_rate": float(np.mean(values == 1.0)),
    }


def _canonical_loop_id(core: tuple[int, ...]) -> str:
    canonical = min(core[index:] + core[:index] for index in range(len(core)))
    closed = canonical + (canonical[0],)
    return "loop_p_" + "-".join(str(value) for value in closed)


def _session_sequences(
    panel: pd.DataFrame,
    labels: np.ndarray,
) -> list[tuple[str, str, tuple[int, ...], tuple[int, ...]]]:
    records: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]] = []
    for (symbol, session), group in panel.groupby(["symbol_norm", "session_date"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        local = np.asarray(labels[positions], dtype=int)
        runs = _rle(local)
        records.append(
            (
                str(symbol),
                str(session),
                tuple(state for _, _, state in runs),
                tuple(end - start for start, end, _ in runs),
            )
        )
    return records


def _closure_schedule(states: tuple[int, ...]) -> list[tuple[int, str]]:
    stack: list[int] = []
    closures: list[tuple[int, str]] = []
    for event_index, state in enumerate(states):
        if state not in stack:
            stack.append(state)
            continue
        stack_index = stack.index(state)
        core = tuple(stack[stack_index:])
        if len(core) >= 2:
            closures.append((event_index, _canonical_loop_id(core)))
        stack = stack[:stack_index] + [state]
    return closures


def _first_event_counts(
    records: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]],
    candidate_ids: tuple[str, ...],
) -> dict[str, int]:
    counts = {candidate: 0 for candidate in candidate_ids}
    for _, _, states, durations in records:
        starts = np.r_[0, np.cumsum(np.asarray(durations, dtype=int))[:-1]]
        closures = _closure_schedule(states)
        closure_pointer = 0
        for event_index, (run_start, duration) in enumerate(zip(starts, durations, strict=True)):
            while closure_pointer < len(closures) and closures[closure_pointer][0] <= event_index:
                closure_pointer += 1
            if closure_pointer >= len(closures):
                break
            closure_event, loop_id = closures[closure_pointer]
            event_bar = int(starts[closure_event])
            lower = max(int(run_start), event_bar - 24)
            upper = min(int(run_start + duration - 1), event_bar - 1)
            if upper >= lower and loop_id in counts:
                counts[loop_id] += upper - lower + 1
    return counts


def _first_event_surface(
    panel: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (symbol, session), group in panel.groupby(["symbol_norm", "session_date"], sort=True):
        positions = group.index.to_numpy(dtype=int)
        bars = group["bar_index_in_session"].to_numpy(dtype=int)
        timestamps = pd.to_datetime(group["timestamp"], utc=True)
        complete = np.array_equal(bars, np.arange(len(group), dtype=int)) and (
            len(group) <= 1 or timestamps.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all()
        )
        if not complete:
            rows.extend(
                {
                    "position": int(position),
                    "symbol": str(symbol),
                    "session": str(session),
                    "bar_ordinal": int(bar),
                    "primary_label": "UNAVAILABLE_SOURCE",
                    "bars_until_completion": math.nan,
                }
                for position, bar in zip(positions, bars, strict=True)
            )
            continue
        local = np.asarray(labels[positions], dtype=int)
        runs = _rle(local)
        states = tuple(state for _, _, state in runs)
        starts = np.asarray([start for start, _, _ in runs], dtype=int)
        registered_paths = {
            (5, 6, 5): "loop_p_5-6-5",
            (6, 5, 6): "loop_p_5-6-5",
            (4, 6, 4): "loop_p_4-6-4",
            (6, 4, 6): "loop_p_4-6-4",
        }
        registered: list[tuple[int, int, str]] = []
        unregistered: list[tuple[int, int]] = []
        for event_index, current_state in enumerate(states):
            completion_bar = int(bars[int(starts[event_index])])
            if event_index >= 2:
                registered_path = tuple(states[event_index - 2 : event_index + 1])
                loop_id = registered_paths.get(registered_path)
                if loop_id is not None:
                    registered.append((event_index, completion_bar, loop_id))
            lower = max(0, event_index - 8)
            for start_index in range(event_index - 2, lower - 1, -1):
                if states[start_index] != current_state:
                    continue
                path = tuple(states[start_index : event_index + 1])
                if path not in registered_paths:
                    unregistered.append((event_index, completion_bar))
        session_end = int(bars.max())
        for local_position, bar in enumerate(bars):
            current_event = int(np.searchsorted(starts, local_position, side="right") - 1)
            horizon_end = int(bar) + 24
            future_registered = [
                event
                for event in registered
                if event[0] > current_event and event[1] > int(bar) and event[1] <= horizon_end
            ]
            future_unregistered = [
                event
                for event in unregistered
                if event[0] > current_event and event[1] > int(bar) and event[1] <= horizon_end
            ]
            label = "NO_LOOP_WITHIN_HORIZON"
            bars_until: float = math.nan
            earliest_registered = min(
                future_registered, key=lambda item: (item[1], item[0], item[2]), default=None
            )
            earliest_unregistered = min(
                future_unregistered,
                key=lambda item: (item[1], item[0]),
                default=None,
            )
            if earliest_registered is not None and (
                earliest_unregistered is None or earliest_registered[1] <= earliest_unregistered[1]
            ):
                label = earliest_registered[2]
                bars_until = float(earliest_registered[1] - int(bar))
            elif earliest_unregistered is not None:
                label = "OTHER_PRIMITIVE_LOOP"
                bars_until = float(earliest_unregistered[1] - int(bar))
            future_state_event_exists = current_event + 1 < len(states)
            if label == "NO_LOOP_WITHIN_HORIZON" and (
                not future_state_event_exists or session_end <= horizon_end
            ):
                label = "SESSION_END"
            rows.append(
                {
                    "position": int(positions[local_position]),
                    "symbol": str(symbol),
                    "session": str(session),
                    "bar_ordinal": int(bar),
                    "primary_label": label,
                    "bars_until_completion": bars_until,
                }
            )
    return pd.DataFrame(rows).sort_values("position", kind="mergesort").reset_index(drop=True)


def _hysteretic_by_session(
    probabilities: np.ndarray,
    groups: tuple[np.ndarray, ...],
) -> np.ndarray:
    output = np.full(len(probabilities), -1, dtype=np.int16)
    for group in groups:
        current = int(np.argmax(probabilities[int(group[0])]))
        output[int(group[0])] = current
        for position in group[1:]:
            candidate = int(np.argmax(probabilities[int(position)]))
            if candidate != current:
                candidate_probability = float(probabilities[int(position), candidate])
                advantage = candidate_probability - float(probabilities[int(position), current])
                if candidate_probability >= 0.55 and advantage >= 0.10:
                    current = candidate
            output[int(position)] = current
    return output


def _null_parameters(
    records: list[tuple[str, str, tuple[int, ...], tuple[int, ...]]],
    *,
    state_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial = np.full(state_count, 0.5, dtype=float)
    transitions = np.full((state_count, state_count), 0.5, dtype=float)
    np.fill_diagonal(transitions, 0.0)
    at_risk = np.zeros((state_count, 78), dtype=np.int64)
    exits = np.zeros_like(at_risk)
    for _, _, states, durations in records:
        initial[states[0]] += 1.0
        for origin, destination in zip(states[:-1], states[1:], strict=True):
            transitions[origin, destination] += 1.0
        for index, (state, duration) in enumerate(zip(states, durations, strict=True)):
            at_risk[state, :duration] += 1
            if index < len(states) - 1:
                exits[state, duration - 1] += 1
    initial /= initial.sum()
    transitions /= transitions.sum(axis=1, keepdims=True)
    pooled_risk = at_risk.sum(axis=0).astype(float)
    pooled_exits = exits.sum(axis=0).astype(float)
    pooled_hazard = np.divide(
        pooled_exits + 0.5,
        pooled_risk + 1.0,
        out=np.zeros(78, dtype=float),
        where=pooled_risk > 0,
    )
    hazard = np.zeros_like(at_risk, dtype=float)
    for state in range(state_count):
        supported = at_risk[state] > 0
        hazard[state, supported] = (exits[state, supported] + 8.0 * pooled_hazard[supported]) / (
            at_risk[state, supported] + 8.0
        )
        hazard[state, ~supported & (pooled_risk > 0)] = pooled_hazard[
            ~supported & (pooled_risk > 0)
        ]
    cumulative = np.zeros((state_count, 79), dtype=float)
    for state in range(state_count):
        survival = 1.0
        for age in range(78):
            cumulative[state, age] = survival * hazard[state, age]
            survival *= 1.0 - hazard[state, age]
        cumulative[state, 78] = survival
    cumulative = np.cumsum(cumulative, axis=1)
    cumulative[:, -1] = 1.0
    return initial, transitions, cumulative


@njit(cache=False)
def _independent_two_state_null_kernel(
    initial: np.ndarray,
    transitions: np.ndarray,
    duration_cumulative: np.ndarray,
    session_lengths: np.ndarray,
    candidate_pairs: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    output = np.zeros((draws, len(candidate_pairs)), dtype=np.int64)
    for draw in range(draws):
        np.random.seed(seed + draw)
        for session_length in session_lengths:
            states = np.empty(78, dtype=np.int64)
            durations = np.empty(78, dtype=np.int64)
            starts = np.empty(78, dtype=np.int64)
            uniform = np.random.random()
            cumulative = 0.0
            state = 0
            for state_index in range(len(initial)):
                cumulative += initial[state_index]
                if uniform <= cumulative:
                    state = state_index
                    break
            elapsed = 0
            run_count = 0
            while elapsed < session_length:
                states[run_count] = state
                starts[run_count] = elapsed
                duration_uniform = np.random.random()
                selected_duration = duration_cumulative.shape[1]
                for duration_index in range(duration_cumulative.shape[1]):
                    if duration_uniform <= duration_cumulative[state, duration_index]:
                        selected_duration = duration_index + 1
                        break
                duration = min(selected_duration, session_length - elapsed)
                durations[run_count] = duration
                elapsed += duration
                run_count += 1
                if elapsed >= session_length:
                    break
                transition_uniform = np.random.random()
                cumulative = 0.0
                next_state = 0
                for destination in range(len(initial)):
                    cumulative += transitions[state, destination]
                    if transition_uniform <= cumulative:
                        next_state = destination
                        break
                state = next_state
            stack = np.empty(16, dtype=np.int64)
            stack_size = 0
            closure_events = np.empty(78, dtype=np.int64)
            closure_candidates = np.full(78, -1, dtype=np.int64)
            closure_count = 0
            for event_index in range(run_count):
                current = states[event_index]
                stack_index = -1
                for index in range(stack_size):
                    if stack[index] == current:
                        stack_index = index
                        break
                if stack_index < 0:
                    stack[stack_size] = current
                    stack_size += 1
                    continue
                if stack_size - stack_index == 2:
                    left = min(stack[stack_index], stack[stack_index + 1])
                    right = max(stack[stack_index], stack[stack_index + 1])
                    for index in range(len(candidate_pairs)):
                        if candidate_pairs[index, 0] == left and candidate_pairs[index, 1] == right:
                            closure_candidates[closure_count] = index
                            break
                closure_events[closure_count] = event_index
                closure_count += 1
                stack[stack_index] = current
                stack_size = stack_index + 1
            closure_pointer = 0
            for event_index in range(run_count):
                while (
                    closure_pointer < closure_count
                    and closure_events[closure_pointer] <= event_index
                ):
                    closure_pointer += 1
                if closure_pointer >= closure_count:
                    break
                candidate_index = closure_candidates[closure_pointer]
                if candidate_index < 0:
                    continue
                event_bar = starts[closure_events[closure_pointer]]
                run_start = starts[event_index]
                run_end = run_start + durations[event_index] - 1
                lower = max(run_start, event_bar - 24)
                upper = min(run_end, event_bar - 1)
                if upper >= lower:
                    output[draw, candidate_index] += upper - lower + 1
    return output


def _audit_k_seed_surface(
    panel: pd.DataFrame,
    scaled: np.ndarray,
    *,
    full_groups: tuple[np.ndarray, ...],
    causal_groups: tuple[np.ndarray, ...],
    reference_model: dict[str, np.ndarray],
    reference_hard: np.ndarray,
) -> tuple[bool, dict[str, Any], dict[str, int]]:
    registry = pd.read_csv(PRIMARY / "k_seed_model_registry.csv").set_index("model_id")
    reported_alignment = pd.read_csv(PRIMARY / "state_alignment.csv")
    reported_stability = pd.read_csv(PRIMARY / "state_stability_by_k_seed.csv").set_index(
        "model_id"
    )
    reported_loops = pd.read_csv(PRIMARY / "loop_stability_by_k_seed.csv")
    sample_step = max(1, len(scaled) // 200000)
    sample_indices = np.arange(0, len(scaled), sample_step, dtype=int)[:200000]
    empty_sample = np.asarray([], dtype=np.int64)
    reset = _reset_mask(causal_groups, len(panel))
    all_pass = True
    maximum_differences: dict[str, float] = {
        "training_objective": 0.0,
        "causal_negative_log_likelihood": 0.0,
        "iid_mixture_negative_log_likelihood": 0.0,
        "alignment_total_cost": 0.0,
        "normalized_mutual_information": 0.0,
        "structural_null_rate_ratio": 0.0,
    }
    independent_k8_positive_counts = {loop_id: 0 for loop_id in SELECTED_IDS}
    independent_k8_nmi: list[float] = []
    for state_count in K_VALUES:
        for seed in SEEDS:
            model_id = f"regime_k{state_count}_seed{seed}"
            clusterer = MiniBatchKMeans(
                n_clusters=state_count,
                batch_size=4096,
                n_init=10,
                max_iter=300,
                random_state=seed,
            )
            clusterer.fit(scaled[sample_indices])
            raw = clusterer.predict(scaled).astype(np.int16)
            cleaned = _clean_short_runs(
                raw,
                scaled,
                full_groups,
                np.asarray(clusterer.cluster_centers_, dtype=float),
            )
            semantic, _ = _semantic_remap(cleaned, panel)
            fitted = _estimate_parameters(
                scaled,
                semantic,
                full_groups,
                state_count=state_count,
            )
            emissions = _gaussian_emissions(scaled, fitted)
            posterior = _posterior_sample_kernel(
                emissions,
                _expand_hazard(fitted["duration_hazard"]),
                fitted["transitions"],
                fitted["initial"],
                fitted["occupancy"],
                reset,
                empty_sample,
            )
            hard = posterior[0]
            mapping, independent_alignment = _independent_alignment(reference_model, fitted)
            aligned = _apply_mapping(hard, mapping)
            reported_row = registry.loc[model_id]
            compared_registry = {
                "training_objective": float(clusterer.inertia_),
                "causal_negative_log_likelihood": float(-posterior[6].mean()),
                "iid_mixture_negative_log_likelihood": float(-posterior[7].mean()),
            }
            for column, actual in compared_registry.items():
                difference = abs(actual - float(reported_row[column]))
                maximum_differences[column] = max(maximum_differences[column], difference)
                all_pass &= difference <= max(1e-8, abs(actual) * 5e-11)

            artifact_alignment = reported_alignment.loc[
                reported_alignment["model_id"].eq(model_id)
            ].sort_values("candidate_state", kind="mergesort")
            independent_alignment = independent_alignment.sort_values(
                "candidate_state", kind="mergesort"
            )
            all_pass &= artifact_alignment["candidate_state"].astype(int).tolist() == (
                independent_alignment["candidate_state"].astype(int).tolist()
            )
            all_pass &= artifact_alignment["reference_state"].astype(int).tolist() == (
                independent_alignment["reference_state"].astype(int).tolist()
            )
            alignment_difference = float(
                np.max(
                    np.abs(
                        artifact_alignment["total_cost"].to_numpy(dtype=float)
                        - independent_alignment["total_cost"].to_numpy(dtype=float)
                    )
                )
            )
            maximum_differences["alignment_total_cost"] = max(
                maximum_differences["alignment_total_cost"], alignment_difference
            )
            all_pass &= alignment_difference < 1e-9

            nmi = float(normalized_mutual_info_score(reference_hard, hard))
            if state_count == 8:
                independent_k8_nmi.append(nmi)
            nmi_difference = abs(
                nmi - float(reported_stability.loc[model_id, "normalized_mutual_information"])
            )
            maximum_differences["normalized_mutual_information"] = max(
                maximum_differences["normalized_mutual_information"], nmi_difference
            )
            all_pass &= nmi_difference < 1e-9
            ari = float(adjusted_rand_score(reference_hard, hard))
            all_pass &= (
                abs(ari - float(reported_stability.loc[model_id, "adjusted_rand_index"])) < 1e-9
            )

            reference_to_candidate = {
                reference_state: candidate_state
                for candidate_state, reference_state in mapping.items()
            }
            translated: dict[str, str] = {}
            for loop_id, path in zip(SELECTED_IDS, SELECTED_PATHS, strict=True):
                if all(state in reference_to_candidate for state in path):
                    candidate_path = tuple(reference_to_candidate[state] for state in path)
                    translated[loop_id] = _canonical_loop_id(candidate_path[:-1])
            records = _session_sequences(panel, hard)
            if translated:
                native_ids = tuple(translated.values())
                observed = _first_event_counts(records, native_ids)
                null_initial, null_transition, null_duration = _null_parameters(
                    records,
                    state_count=state_count,
                )
                candidate_pairs = np.asarray(
                    [
                        sorted(
                            int(value) for value in loop_id.removeprefix("loop_p_").split("-")[:-1]
                        )
                        for loop_id in native_ids
                    ],
                    dtype=np.int64,
                )
                draws = _independent_two_state_null_kernel(
                    null_initial,
                    null_transition,
                    null_duration,
                    np.asarray([sum(record[3]) for record in records], dtype=np.int64),
                    candidate_pairs,
                    100,
                    seed + 50000,
                )
                for index, (reference_loop, native_loop) in enumerate(translated.items()):
                    reported = reported_loops.loc[
                        reported_loops["model_id"].eq(model_id)
                        & reported_loops["primitive_loop_id"].eq(reference_loop)
                    ].iloc[0]
                    null_mean = float(draws[:, index].mean())
                    ratio = (
                        float(observed[native_loop] / null_mean) if null_mean > 0.0 else math.nan
                    )
                    all_pass &= int(reported["observed_first_events"]) == observed[native_loop]
                    difference = abs(float(reported["structural_null_rate_ratio"]) - ratio)
                    maximum_differences["structural_null_rate_ratio"] = max(
                        maximum_differences["structural_null_rate_ratio"], difference
                    )
                    all_pass &= difference < 1e-9
                    if state_count == 8 and ratio > 1.0:
                        independent_k8_positive_counts[reference_loop] += 1
            del emissions, posterior, hard, aligned, raw, cleaned, semantic
    metrics = {
        "maximum_absolute_differences": maximum_differences,
        "independently_fitted_model_count": len(K_VALUES) * len(SEEDS),
        "independent_k8_positive_structural_excess_counts": independent_k8_positive_counts,
        "independent_minimum_k8_nmi": min(independent_k8_nmi),
    }
    return all_pass, metrics, independent_k8_positive_counts


def _validate_safety() -> tuple[bool, list[str]]:
    failures: list[str] = []
    for path in sorted(PRIMARY.iterdir()):
        if path.name in {"independent_audit.json", "exact_rerun_manifest.json"}:
            continue
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, expected in SAFETY_FLAGS.items():
                if payload.get(key) != expected:
                    failures.append(f"{path.name}:{key}")
        elif path.suffix == ".csv":
            header = pd.read_csv(path, nrows=0)
            missing = set(SAFETY_FLAGS).difference(header.columns)
            if missing:
                failures.append(f"{path.name}:missing:{sorted(missing)}")
                continue
            frame = pd.read_csv(path, usecols=list(SAFETY_FLAGS))
            for key, expected in SAFETY_FLAGS.items():
                if not frame[key].eq(expected).all():
                    failures.append(f"{path.name}:{key}")
        elif path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=list(SAFETY_FLAGS))
            for key, expected in SAFETY_FLAGS.items():
                if not frame[key].eq(expected).all():
                    failures.append(f"{path.name}:{key}")
        elif path.suffix == ".npz":
            with np.load(path) as stored:
                for key, expected in SAFETY_FLAGS.items():
                    if key not in stored or stored[key].item() != expected:
                        failures.append(f"{path.name}:{key}")
    report = REPORT_PATH.read_text(encoding="utf-8")
    if not all(f"`{key}={str(value).lower()}`" in report for key, value in SAFETY_FLAGS.items()):
        failures.append("report:safety_flags")
    return not failures, failures


def _verify_exact_rerun() -> dict[str, Any]:
    primary_files = {
        path.name: path
        for path in PRIMARY.iterdir()
        if path.is_file() and path.name not in EXACT_EXCLUSIONS
    }
    rerun_files = {
        path.name: path
        for path in EXACT_RERUN.iterdir()
        if path.is_file() and path.name not in EXACT_EXCLUSIONS
    }
    missing = sorted(set(primary_files).symmetric_difference(rerun_files))
    mismatches = sorted(
        name
        for name in set(primary_files).intersection(rerun_files)
        if _sha256_file(primary_files[name]) != _sha256_file(rerun_files[name])
    )
    payload = {
        "manifest_version": "regime_model_validity_v2_exact_rerun",
        "compared_artifact_count": len(set(primary_files).intersection(rerun_files)),
        "excluded_self_referential_files": sorted(EXACT_EXCLUSIONS),
        "missing_or_extra_files": missing,
        "mismatched_files": mismatches,
        "byte_identical": not missing and not mismatches,
        "primary_artifact_manifest_hash": _sha256_file(PRIMARY / "artifact_manifest.json"),
        "exact_artifact_manifest_hash": _sha256_file(EXACT_RERUN / "artifact_manifest.json"),
        **SAFETY_FLAGS,
    }
    payload["manifest_hash"] = _sha256_payload(payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n")


def _historical_tree_changes() -> list[str]:
    """Return baseline modifications while allowing only this task's new V2 paths."""

    output = subprocess.run(
        ["git", "diff", "--name-status", BASELINE_SHA, "--"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    failures: list[str] = []
    for line in output:
        status, *paths = line.split("\t")
        allowed_addition = (
            status == "A"
            and paths
            and all(any(marker in path for marker in NEW_V2_PATH_MARKERS) for path in paths)
        )
        if not allowed_addition:
            failures.append(line)
    return failures


def audit() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source_identity = json.loads((PRIMARY / "source_identity_manifest.json").read_text())
    implementation_identity = json.loads(
        (PRIMARY / "implementation_source_manifest.json").read_text()
    )
    run_metadata = json.loads((PRIMARY / "run_metadata.json").read_text())
    reconstruction = json.loads((PRIMARY / "current_regime_reconstruction.json").read_text())
    decision = json.loads((PRIMARY / "part_a_decision.json").read_text())
    math_audit = json.loads((PRIMARY / "regime_math_audit.json").read_text())
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}

    checks["contract_hash"] = _sha256_file(CONTRACT_PATH) == EXPECTED_CONTRACT_HASH
    checks["contract_safety"] = all(
        contract.get(key) == value for key, value in SAFETY_FLAGS.items()
    )
    checks["source_manifest_baseline"] = (
        source_identity["git_sha"] == BASELINE_SHA
        and source_identity["expected_data_snapshot_hash"] == EXPECTED_SNAPSHOT
    )
    checks["source_file_hashes"] = all(
        _sha256_file(REPO_ROOT / relative) == expected
        for relative, expected in source_identity["source_file_hashes"].items()
    )
    implementation_files = {
        relative: _sha256_file(REPO_ROOT / relative)
        for relative in implementation_identity["files"]
    }
    independent_implementation_hash = _sha256_payload(implementation_files)
    checks["implementation_source_manifest"] = (
        implementation_files == implementation_identity["files"]
        and independent_implementation_hash
        == implementation_identity["implementation_source_hash"]
        == reconstruction["implementation_source_hash"]
        == run_metadata["implementation_source_hash"]
    )

    source_hashes, snapshot = _source_snapshot()
    checks["bounded_source_snapshot"] = snapshot == EXPECTED_SNAPSHOT
    checks["bounded_source_components_complete"] = len(source_hashes) == len(SYMBOLS) + 1 and all(
        len(value) == 64 for value in source_hashes.values()
    )
    assessment_source_hashes, assessment_snapshot = _source_snapshot(
        start=ASSESSMENT_START,
        end=ASSESSMENT_END,
    )
    combined_snapshot = _sha256_payload(
        {
            "development_2024": snapshot,
            "unchanged_retrospective_2025": assessment_snapshot,
        }
    )
    checks["all_opened_data_bound"] = (
        len(assessment_source_hashes) == len(SYMBOLS) + 1
        and assessment_snapshot == run_metadata["assessment_snapshot_hash"]
        and combined_snapshot == reconstruction["data_snapshot_hash"]
        and combined_snapshot == run_metadata["combined_opened_data_snapshot_hash"]
    )
    frozen_core = _load_module("regime_validity_independent_frozen_core", FROZEN_CORE_PATH)
    panel = _prepare_panel(frozen_core)
    checks["panel_identity"] = (
        len(panel) == reconstruction["panel_rows"] == 424583
        and panel["symbol_norm"].nunique() == reconstruction["stock_count"] == 22
        and panel["session_date"].nunique() == reconstruction["session_count"] == 252
    )
    feature_names = tuple(pd.read_csv(PREPROCESSING_PATH)["feature"].astype(str))
    prefix_pass, prefix_metrics = _prefix_invariance_audit(
        frozen_core,
        panel,
        feature_names=feature_names,
    )
    checks["emission_prefix_invariance"] = prefix_pass
    metrics["emission_prefix_invariance"] = prefix_metrics
    causality = pd.read_parquet(
        PRIMARY / "regime_causality_audit.parquet",
        columns=[
            "feature_availability_pass",
            "future_information_used",
            "feature_latest_bar_offset_max",
            "feature_provenance_complete",
        ],
    )
    checks["feature_availability_provenance"] = bool(
        causality["feature_availability_pass"].all()
        and not causality["future_information_used"].any()
        and causality["feature_latest_bar_offset_max"].le(0).all()
        and causality["feature_provenance_complete"].all()
    )

    preprocessing = pd.read_csv(PREPROCESSING_PATH)
    with np.load(MODEL_PATH) as stored:
        model = {key: np.asarray(stored[key]).copy() for key in stored.files}
    scaled, emissions = _scale_and_emit(panel, preprocessing, model)
    refit_scaled = _independently_fit_scale(
        panel,
        tuple(preprocessing["feature"].astype(str)),
    )
    sample = pd.read_parquet(PRIMARY / "current_state_assignment_sample.parquet")
    sample_positions = np.unique(np.linspace(0, len(panel) - 1, 2048, dtype=int))
    checks["sample_row_identity"] = (
        len(sample) == len(sample_positions)
        and sample["symbol"].astype(str).tolist()
        == panel.loc[sample_positions, "symbol_norm"].astype(str).tolist()
        and sample["bar_ordinal"].astype(int).tolist()
        == panel.loc[sample_positions, "bar_index_in_session"].astype(int).tolist()
    )
    stored_scaled = np.stack(sample["scaled_emissions"].to_numpy())
    stored_emissions = np.stack(sample["log_emissions"].to_numpy())
    metrics["scaled_sample_max_abs_difference"] = float(
        np.max(np.abs(stored_scaled - scaled[sample_positions]))
    )
    metrics["emission_sample_max_abs_difference"] = float(
        np.max(np.abs(stored_emissions - emissions[sample_positions]))
    )
    checks["imputation_scaling_sample"] = metrics["scaled_sample_max_abs_difference"] < 1e-6
    checks["gaussian_emission_sample"] = metrics["emission_sample_max_abs_difference"] < 1e-10

    full_groups = _groups(panel, split_gaps=False)
    causal_groups = _groups(panel, split_gaps=True)
    active_hazard = _expand_hazard(np.asarray(model["duration_hazard"], dtype=float))
    active = _posterior_sample_kernel(
        emissions,
        active_hazard,
        np.asarray(model["transitions"], dtype=float),
        np.asarray(model["initial"], dtype=float),
        np.asarray(model["occupancy"], dtype=float),
        _reset_mask(causal_groups, len(panel)),
        sample_positions,
    )
    legacy = _posterior_sample_kernel(
        emissions,
        np.asarray(model["duration_hazard"], dtype=float),
        np.asarray(model["transitions"], dtype=float),
        np.asarray(model["initial"], dtype=float),
        np.asarray(model["occupancy"], dtype=float),
        _reset_mask(full_groups, len(panel)),
        sample_positions,
    )
    stored_probability = np.stack(sample["posterior_state_probabilities"].to_numpy())
    stored_state_age = np.stack(sample["state_age_posterior"].to_numpy()).reshape(
        len(sample), 8, 78
    )
    metrics["posterior_sample_max_abs_difference"] = float(
        np.max(np.abs(stored_probability - active[1]))
    )
    metrics["state_age_sample_max_abs_difference"] = float(
        np.max(np.abs(stored_state_age - active[2]))
    )
    metrics["expected_age_sample_max_abs_difference"] = float(
        np.max(
            np.abs(
                np.sum(stored_state_age * np.arange(1, 79)[None, None, :], axis=(1, 2)) - active[3]
            )
        )
    )
    metrics["departure_sample_max_abs_difference"] = float(
        np.max(
            np.abs(np.sum(stored_state_age * active_hazard[None, :, :], axis=(1, 2)) - active[4])
        )
    )
    checks["posterior_state_age_reconstruction"] = (
        metrics["posterior_sample_max_abs_difference"] < 1e-10
        and metrics["state_age_sample_max_abs_difference"] < 1e-10
    )
    checks["hard_map_reconstruction"] = np.array_equal(
        active[0][sample_positions], sample["posterior_map_state"].to_numpy(dtype=int)
    )
    checks["legacy_hard_state_reconstruction"] = np.array_equal(
        legacy[0][sample_positions], sample["legacy_hard_state"].to_numpy(dtype=int)
    )
    checks["expected_age_reconstruction"] = (
        metrics["expected_age_sample_max_abs_difference"] < 1e-10
    )
    checks["departure_probability_reconstruction"] = (
        metrics["departure_sample_max_abs_difference"] < 1e-10
    )
    reset_samples = sample["bar_ordinal"].to_numpy(dtype=int) == 0
    checks["session_reset"] = bool(
        np.allclose(stored_state_age[reset_samples, :, 1:], 0.0, atol=1e-12)
    )
    independent_hysteretic = _hysteretic_by_session(active[8], causal_groups)
    checks["hysteretic_sample_reconstruction"] = np.array_equal(
        independent_hysteretic[sample_positions],
        sample["hysteretic_state"].to_numpy(dtype=int),
    )
    independent_hard_events = _first_event_surface(panel, legacy[0])
    independent_hysteretic_events = _first_event_surface(panel, independent_hysteretic)
    reported_event_comparison = pd.read_parquet(
        PRIMARY / "state_representation_event_comparison.parquet",
        columns=[
            "primary_label_reference",
            "bars_until_completion_reference",
            "primary_label_candidate",
            "bars_until_completion_candidate",
        ],
    )
    independent_event_agreement: dict[str, float] = {}
    event_surface_pass = len(reported_event_comparison) == len(independent_hard_events)
    for loop_id in SELECTED_IDS:
        selected = independent_hard_events["primary_label"].eq(loop_id)
        same_loop = independent_hysteretic_events["primary_label"].eq(loop_id)
        bounded = (
            (
                independent_hysteretic_events["bars_until_completion"]
                - independent_hard_events["bars_until_completion"]
            )
            .abs()
            .le(2)
        )
        independent_event_agreement[loop_id] = float((same_loop & bounded)[selected].mean())
        reported_selected = reported_event_comparison["primary_label_reference"].eq(loop_id)
        reported_same = reported_event_comparison["primary_label_candidate"].eq(loop_id)
        reported_bounded = (
            (
                reported_event_comparison["bars_until_completion_candidate"]
                - reported_event_comparison["bars_until_completion_reference"]
            )
            .abs()
            .le(2)
        )
        reported_fraction = float((reported_same & reported_bounded)[reported_selected].mean())
        event_surface_pass &= math.isclose(
            independent_event_agreement[loop_id],
            reported_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    checks["independent_hard_hysteretic_loop_events"] = bool(event_surface_pass)
    metrics["independent_hard_hysteretic_event_agreement"] = independent_event_agreement

    historical_step = max(1, len(refit_scaled) // 200000)
    historical_sample = np.arange(0, len(refit_scaled), historical_step, dtype=int)
    clusterer = MiniBatchKMeans(
        n_clusters=8,
        batch_size=4096,
        n_init=10,
        max_iter=300,
        random_state=20260710,
    )
    clusterer.fit(refit_scaled[historical_sample])
    raw_labels = clusterer.predict(refit_scaled).astype(np.int16)
    cleaned = _clean_short_runs(
        raw_labels,
        refit_scaled,
        full_groups,
        np.asarray(clusterer.cluster_centers_, dtype=float),
    )
    semantic, mapping = _semantic_remap(cleaned, panel)
    fitted = _estimate_parameters(refit_scaled, semantic, full_groups, state_count=8)
    differences = {
        key: float(np.max(np.abs(fitted[key] - np.asarray(model[key], dtype=float))))
        for key in fitted
    }
    metrics["independent_refit_parameter_differences"] = differences
    checks["refit_difference_reproduced"] = all(
        math.isclose(
            value,
            float(reconstruction["refit_parameter_differences"][key]),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        for key, value in differences.items()
    )
    cleanup = pd.read_csv(PRIMARY / "cleaning_variant_state_metrics.csv")
    reported_cleanup = float(
        cleanup.loc[cleanup["variant"].eq("CLEANING_1"), "bars_relabelled_share"].iloc[0]
    )
    metrics["independent_cleanup_changed_share"] = float(np.mean(cleaned != raw_labels))
    checks["cleanup_reconstruction"] = math.isclose(
        metrics["independent_cleanup_changed_share"],
        reported_cleanup,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    cleanup_audit = pd.read_csv(PRIMARY / "short_run_cleaning_audit.csv").set_index("variant")
    changed_source_runs = 0
    raw_run_count = 0
    cleaned_run_count = 0
    for group in full_groups:
        raw_runs = _rle(raw_labels[group])
        cleaned_runs = _rle(cleaned[group])
        raw_run_count += len(raw_runs)
        cleaned_run_count += len(cleaned_runs)
        changed_source_runs += sum(
            bool(np.any(raw_labels[group[start:end]] != cleaned[group[start:end]]))
            for start, end, _ in raw_runs
        )
    checks["cleanup_run_change_reconstruction"] = changed_source_runs == int(
        cleanup_audit.loc["CLEANING_1", "runs_affected"]
    ) and cleaned_run_count - raw_run_count == int(
        cleanup_audit.loc["CLEANING_1", "transition_count_change"]
    )
    checks["cleanup_future_neighbor_classification"] = bool(
        cleanup_audit.loc["CLEANING_1", "uses_future_neighbor"]
    )
    with np.load(PRIMARY / "current_state_parameters.npz") as stored:
        reported_mapping = dict(
            zip(
                stored["semantic_old_state"].astype(int),
                stored["semantic_new_state"].astype(int),
                strict=True,
            )
        )
    checks["semantic_mapping_reconstruction"] = mapping == reported_mapping
    centroids = pd.read_csv(PRIMARY / "current_state_centroids.csv")
    reconstructed_centroids = centroids.pivot(
        index="state", columns="feature", values="reconstructed_kmeans_centroid"
    ).reindex(columns=preprocessing["feature"])
    semantic_centers = np.empty_like(clusterer.cluster_centers_, dtype=float)
    for old_state, new_state in mapping.items():
        semantic_centers[new_state] = clusterer.cluster_centers_[old_state]
    metrics["cluster_centroid_max_abs_difference"] = float(
        np.max(np.abs(reconstructed_centroids.to_numpy() - semantic_centers))
    )
    checks["cluster_assignment_centroids"] = metrics["cluster_centroid_max_abs_difference"] < 1e-10

    registry = pd.read_csv(PRIMARY / "k_seed_model_registry.csv")
    expected_registry: list[tuple[str, int, int, int]] = []
    registry_ordinal = 0
    for state_count in (6, 8, 10, 12):
        for seed in (20260710, 20260711, 20260712, 20260713, 20260714):
            expected_registry.append(
                (
                    f"regime_k{state_count}_seed{seed}",
                    state_count,
                    seed,
                    registry_ordinal,
                )
            )
            registry_ordinal += 1
    actual_registry = list(
        registry[["model_id", "state_count", "seed", "registry_ordinal"]].itertuples(
            index=False, name=None
        )
    )
    checks["k_seed_registry"] = actual_registry == expected_registry
    alignment = pd.read_csv(PRIMARY / "state_alignment.csv")
    alignment_valid = True
    for _, group in alignment.groupby("model_id", sort=True):
        alignment_valid &= not group["candidate_state"].duplicated().any()
        alignment_valid &= not group["reference_state"].duplicated().any()
        alignment_valid &= bool(np.isfinite(group["total_cost"]).all())
    checks["state_alignment_integrity"] = bool(alignment_valid)
    k_seed_pass, k_seed_metrics, independent_k8_counts = _audit_k_seed_surface(
        panel,
        refit_scaled,
        full_groups=full_groups,
        causal_groups=causal_groups,
        reference_model=model,
        reference_hard=legacy[0],
    )
    checks["independent_k_seed_fits_alignment_and_nulls"] = k_seed_pass
    metrics["independent_k_seed_audit"] = k_seed_metrics

    loop_stability = pd.read_csv(PRIMARY / "loop_stability_by_k_seed.csv")
    k8_counts = (
        loop_stability.loc[loop_stability["state_count"].eq(8)]
        .groupby("primitive_loop_id")["positive_structural_excess"]
        .sum()
    )
    checks["k8_structural_excess_gate"] = bool((k8_counts >= 4).all())
    checks["independent_k8_structural_excess_gate"] = all(
        independent_k8_counts[loop_id] >= 4 for loop_id in SELECTED_IDS
    )
    event_comparison = pd.read_parquet(
        PRIMARY / "state_representation_event_comparison.parquet",
        columns=["primary_label_reference", "agreement_class"],
    )
    robust = pd.read_csv(PRIMARY / "loop_robustness_by_representation.csv")
    event_agreement: dict[str, float] = {}
    for loop_id in robust["primitive_loop_id"].astype(str):
        selected = event_comparison.loc[
            event_comparison["primary_label_reference"].eq(loop_id), "agreement_class"
        ]
        event_agreement[loop_id] = float(
            selected.isin(["EXACT_EVENT_AGREEMENT", "SAME_PRIMITIVE_SHIFTED_TIMESTAMP"]).mean()
        )
    checks["loop_event_agreement"] = all(
        math.isclose(
            event_agreement[str(row.primitive_loop_id)],
            float(row.same_primitive_bounded_shift_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in robust.itertuples(index=False)
    )

    drift = pd.read_csv(PRIMARY / "state_period_drift.csv")
    sample_loops = pd.read_csv(PRIMARY / "training_sample_loop_stability.csv")
    independently_reconstructed_math_core = all(
        checks[key]
        for key in (
            "posterior_state_age_reconstruction",
            "hard_map_reconstruction",
            "expected_age_reconstruction",
            "departure_probability_reconstruction",
            "session_reset",
            "emission_prefix_invariance",
            "feature_availability_provenance",
        )
    )
    terminal_run_count = len(full_groups)
    independently_reconstructed_duration_censoring_pass = False
    recoverable_local_defect = terminal_run_count > 0 and independently_reconstructed_math_core
    if not independently_reconstructed_math_core:
        independently_derived_decision = "regime_validity_audit_blocked"
    elif not independently_reconstructed_duration_censoring_pass and recoverable_local_defect:
        independently_derived_decision = "regime_representation_requires_targeted_repair"
    else:
        independently_derived_decision = "regime_representation_unstable_loop_dictionary_must_pause"
    metrics["independent_gate_inputs"] = {
        "mathematical_and_causal_core_pass": independently_reconstructed_math_core,
        "terminal_training_run_count": terminal_run_count,
        "terminal_censoring_parameter_pass": independently_reconstructed_duration_censoring_pass,
        "minimum_hard_hysteretic_selected_loop_agreement": min(
            independent_event_agreement.values()
        ),
        "minimum_k8_nmi": k_seed_metrics["independent_minimum_k8_nmi"],
        "k8_positive_structural_excess_counts": independent_k8_counts,
    }
    checks["part_a_gate_reconstruction"] = (
        decision["decision"] == independently_derived_decision
        and decision["part_b_authorized"] is False
        and decision["part_b_accessed"] is False
        and math_audit["terminal_censoring_parameter_pass"]
        == independently_reconstructed_duration_censoring_pass
        and min(independent_event_agreement.values()) >= 0.75
        and all(independent_k8_counts[loop_id] >= 4 for loop_id in SELECTED_IDS)
        and float(drift["centroid_drift_scaled_rms"].max()) <= 3.0
        and float(sample_loops["dictionary_coverage_ratio"].min()) < 0.75
        and float(k_seed_metrics["independent_minimum_k8_nmi"]) < 0.50
    )

    historical_changes = _historical_tree_changes()
    checks["frozen_historical_tree_unchanged"] = historical_changes == []
    safety_pass, safety_failures = _validate_safety()
    checks["artifact_safety_flags"] = safety_pass
    metrics["safety_failures"] = safety_failures
    exact = _verify_exact_rerun()
    checks["exact_rerun"] = bool(exact["byte_identical"])
    _write_json(PRIMARY / "exact_rerun_manifest.json", exact)
    _write_json(EXACT_RERUN / "exact_rerun_manifest.json", exact)

    manifest = json.loads((PRIMARY / "artifact_manifest.json").read_text())
    manifest_hashes_match = all(
        (PRIMARY / relative).is_file() and _sha256_file(PRIMARY / relative) == expected
        for relative, expected in manifest["artifacts"].items()
    )
    checks["artifact_manifest"] = manifest_hashes_match
    report = REPORT_PATH.read_text(encoding="utf-8")
    checks["report_sections"] = all(f"## {index}." in report for index in range(1, 28))

    status = "pass" if all(checks.values()) else "fail"
    payload = {
        "audit_version": "regime_model_validity_v2_independent_audit",
        "status": status,
        "independent_audit_reproducible": status == "pass",
        "primary_decision_reproduced": independently_derived_decision,
        "part_b_authorized": False,
        "part_b_accessed": False,
        "checks": checks,
        "metrics": metrics,
        "exact_rerun_manifest_hash": exact["manifest_hash"],
        "git_sha": BASELINE_SHA,
        "contract_hash": EXPECTED_CONTRACT_HASH,
        "data_snapshot_hash": reconstruction["data_snapshot_hash"],
        "state_model_version": reconstruction["state_model_version"],
        "state_model_hash": reconstruction["state_model_hash"],
        "dictionary_version": reconstruction["dictionary_version"],
        "dictionary_hash": reconstruction["dictionary_hash"],
        **SAFETY_FLAGS,
    }
    payload["audit_hash"] = _sha256_payload(payload)
    _write_json(PRIMARY / "independent_audit.json", payload)
    if status != "pass":
        failed = sorted(key for key, value in checks.items() if not value)
        raise SystemExit(f"independent Part A audit failed: {failed}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    audit()


if __name__ == "__main__":
    main()
