"""Research-only joint history/dwell fixed-loop completion forecast.

The destination marginal is the retained 2024 last-three-state model.  A
2024-only hierarchical dwell model conditions duration on the same history and
the hypothesised destination.  Dynamic programming rolls this normalized joint
kernel through the twenty frozen cycles at 6, 12, and 24 bars.

No 2026 row, price, return, direction, range, P&L, order, broker, runtime, or
deployment path is available.
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


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CONTRACT = HERE / "contracts/20260710-joint-history-semimarkov-loop-completion-v1.json"
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
BACKWARD_ROOT = Path(
    "/private/tmp/stocker_sealed_backward_2023_complete_detector_20260710"
)
PATH_ROOT = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710")
OUT = Path("/private/tmp/stocker_joint_history_semimarkov_loop_completion_20260710")
RUN_2024 = STATE_ROOT / "train_2024_filtered_runs.csv"
RUN_2025 = STATE_ROOT / "test_2025_filtered_runs.csv"
RUN_2023 = BACKWARD_ROOT / "backward_2023_filtered_runs.parquet"
PATH_PARAMETERS = PATH_ROOT / "model_parameters.npz"
PATH_GATES = PATH_ROOT / "gates.json"
PATH_AUDIT = PATH_ROOT / "independent_artifact_audit.json"
SEMIMARKOV_PARAMETERS = STATE_ROOT / "frozen_semimarkov_parameters.npz"
CYCLE_PATH = STATE_ROOT / "fixed_cycle_shuffled_nulls.csv"

SEED = 20260710
K = 8
END_STATE = K
DESTINATIONS = K + 1
HISTORY_VALUES = K + 1
DURATION_BUCKETS = 24
OVERFLOW_DURATION = 24
HORIZONS = (6, 12, 24)
MAX_HORIZON = max(HORIZONS)
MAX_START_BAR = 53
MIN_BIN_SUPPORT = 500
EPSILON = 1e-12
TAU_STATE_DEST = 256.0
TAU_ORDER2 = 256.0
TAU_ORDER3 = 1024.0
SMOOTHING_GRID = (64.0, 256.0, 1024.0)
EXPECTED_SMOOTHING = (TAU_STATE_DEST, TAU_ORDER2, TAU_ORDER3)

MODEL_COLUMNS = {
    "history_path_only": "probability_history_path_only",
    "history_frozen_state_timed": "probability_history_frozen_state_timed",
    "history_destination_timed": "probability_history_destination_timed",
    "history_order2_timed": "probability_history_order2_timed",
    "history_joint_timed": "probability_history_joint_timed",
}
COMPARISON_SPECS = (
    (
        "history_joint_timed",
        "history_frozen_state_timed",
        "joint_vs_frozen_state_timed",
        0.005,
        0.005,
    ),
    (
        "history_joint_timed",
        "history_destination_timed",
        "joint_vs_destination_timed",
        0.0025,
        0.0,
    ),
    (
        "history_destination_timed",
        "history_frozen_state_timed",
        "destination_vs_frozen_state_timed",
        0.005,
        0.005,
    ),
)

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
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in PROTECTED_PATHS:
        candidates = [root]
        if root.is_dir():
            candidates.extend(sorted(root.rglob("*")))
        for path in candidates:
            path = path.resolve()
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
                    "inode": details.st_ino,
                    "mtime_ns": details.st_mtime_ns,
                    "ctime_ns": details.st_ctime_ns,
                    "sha256": content_hash,
                }
            )
    rows.sort(key=lambda row: row["path"])
    runtime_metadata = json.loads(
        (
            WORKSPACE
            / "work/shadow_validation/frozen_loop_movement_shadow_v1/runtime_metadata.json"
        ).read_text()
    )
    ledger = (
        WORKSPACE
        / "work/shadow_validation/frozen_loop_movement_shadow_v1/prediction_ledger.jsonl"
    )
    payload = {
        "files": rows,
        "file_count": len(rows),
        "tree_sha256": canonical_json_hash(rows),
        "runtime_outcomes_opened": runtime_metadata.get("outcomes_opened"),
        "ledger_size": ledger.stat().st_size,
        "ledger_lines": len(ledger.read_text().splitlines()),
        "ledger_sha256": sha256(ledger),
    }
    if payload["runtime_outcomes_opened"] is not False:
        raise AssertionError("prospective movement runtime has opened outcomes")
    return payload


def snapshots_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def canonical_cycle(core: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in core)
    if not values:
        raise ValueError("empty cycle")
    return min(values[index:] + values[:index] for index in range(len(values)))


def oriented_paths(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    paths = {
        core[index:] + core[:index] + (int(current),)
        for index, state in enumerate(core)
        if int(state) == int(current)
    }
    return sorted(paths)


def load_cycles() -> pd.DataFrame:
    source = pd.read_csv(CYCLE_PATH)
    if len(source) != 20:
        raise AssertionError("frozen cycle source must contain twenty rows")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for index, value in enumerate(source["cycle"].astype(str), start=1):
        closed = tuple(int(part) for part in value.split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid closed cycle: {value}")
        core = canonical_cycle(closed[:-1])
        if core in seen or len(core) not in (2, 3, 4):
            raise AssertionError(f"invalid or duplicate frozen cycle: {value}")
        if any(left == right for left, right in zip(core, core[1:] + core[:1])):
            raise AssertionError(f"cycle contains an impossible self transition: {value}")
        seen.add(core)
        rows.append(
            {
                "cycle_id": f"cycle_{index:02d}",
                "cycle": "->".join(str(state) for state in core + (core[0],)),
                "transition_length": len(core),
                "core": core,
            }
        )
    return pd.DataFrame(rows)


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
        raise AssertionError("invalid history state")
    return ((previous_state_2 * HISTORY_VALUES + previous_state_1) * K + current_state)


def load_runs(
    path: Path,
    expected_year: int,
    period: str,
    scoring: bool,
) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {
        "symbol_norm",
        "session_date",
        "state",
        "duration",
        "start_pos",
        "start_timestamp",
        "previous_state_1",
        "previous_state_2",
        "next_state",
        "has_next_state",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"{period} run file lacks {missing}")
    output = frame.copy()
    output["symbol_norm"] = output["symbol_norm"].astype(str)
    output["session_date"] = output["session_date"].astype(str)
    output["state"] = pd.to_numeric(output["state"], errors="raise").astype(int)
    output["duration"] = pd.to_numeric(output["duration"], errors="raise").astype(int)
    output["start_pos"] = pd.to_numeric(output["start_pos"], errors="raise").astype(int)
    output["start_timestamp"] = pd.to_datetime(
        output["start_timestamp"], utc=True, errors="raise"
    )
    output = output.sort_values(
        ["symbol_norm", "session_date", "start_pos"], kind="stable"
    ).reset_index(drop=True)
    dates = pd.to_datetime(output["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {expected_year} or expected_year >= 2026:
        raise AssertionError(f"{period} date boundary failure")
    if output["duration"].le(0).any() or output["state"].min() < 0 or output["state"].max() >= K:
        raise AssertionError(f"{period} invalid state or duration")
    grouped_state = output.groupby(["symbol_norm", "session_date"], sort=False)["state"]
    grouped_duration = output.groupby(
        ["symbol_norm", "session_date"], sort=False
    )["duration"]
    expected_prev1 = grouped_state.shift(1).fillna(END_STATE).astype(int)
    expected_prev2 = grouped_state.shift(2).fillna(END_STATE).astype(int)
    if not np.array_equal(expected_prev1, output["previous_state_1"].astype(int)):
        raise AssertionError(f"{period} previous_state_1 mismatch")
    if not np.array_equal(expected_prev2, output["previous_state_2"].astype(int)):
        raise AssertionError(f"{period} previous_state_2 mismatch")
    expected_next = grouped_state.shift(-1)
    stored_has_next = output["has_next_state"].astype(bool)
    if not np.array_equal(expected_next.notna(), stored_has_next):
        raise AssertionError(f"{period} terminal destination mismatch")
    stored_next = pd.to_numeric(output["next_state"], errors="coerce")
    if not np.array_equal(
        expected_next.loc[stored_has_next].astype(int),
        stored_next.loc[stored_has_next].astype(int),
    ):
        raise AssertionError(f"{period} next state mismatch")
    output["next_outcome"] = expected_next.fillna(END_STATE).astype(int)
    if output["next_outcome"].eq(output["state"]).any():
        raise AssertionError("compressed runs contain a self transition")
    for step in range(1, 5):
        output[f"future_state_{step}"] = (
            grouped_state.shift(-step).fillna(END_STATE).astype(int)
        )
    for step in range(1, 4):
        output[f"future_duration_{step}"] = (
            grouped_duration.shift(-step).fillna(0).astype(int)
        )
    local = output["start_timestamp"].dt.tz_convert("America/New_York")
    offset_minutes = local.dt.hour * 60 + local.dt.minute - 570
    valid_grid = (
        offset_minutes.ge(0)
        & offset_minutes.lt(390)
        & offset_minutes.mod(5).eq(0)
        & local.dt.second.eq(0)
        & local.dt.microsecond.eq(0)
    )
    output["bar_index_in_session"] = -1
    output.loc[valid_grid, "bar_index_in_session"] = (
        offset_minutes.loc[valid_grid] // 5
    ).astype(int)
    output["clock_grid_valid"] = valid_grid.to_numpy(bool)
    output["quarter"] = (
        dates.dt.year.astype(str) + "_q" + dates.dt.quarter.astype(str)
    )
    output["period"] = period
    if scoring:
        output = output.loc[
            output["clock_grid_valid"]
            & output["bar_index_in_session"].le(MAX_START_BAR)
        ].reset_index(drop=True)
    output["anchor_id"] = np.arange(len(output), dtype=np.int64)
    return output


def dwell_bucket(duration: np.ndarray) -> np.ndarray:
    values = np.asarray(duration, dtype=int)
    if (values <= 0).any():
        raise AssertionError("dwell duration must be positive")
    return np.minimum(values, OVERFLOW_DURATION) - 1


def _smooth_counts(
    counts: np.ndarray, prior: np.ndarray, strength: float
) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    output = (counts + strength * prior) / (totals + strength)
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=-1), 1.0):
        raise AssertionError("invalid hierarchical dwell smoothing")
    return output


def dwell_count_tensors(train: pd.DataFrame) -> dict[str, np.ndarray]:
    fit = train.loc[train["next_outcome"].ne(END_STATE)].copy()
    if fit.empty:
        raise AssertionError("no non-terminal dwell rows")
    previous_state_1 = fit["previous_state_1"].to_numpy(dtype=int)
    previous_state_2 = fit["previous_state_2"].to_numpy(dtype=int)
    current = fit["state"].to_numpy(dtype=int)
    destination = fit["next_outcome"].to_numpy(dtype=int)
    bucket = dwell_bucket(fit["duration"].to_numpy(dtype=int))
    state_dest_counts = np.zeros(
        (K, DESTINATIONS, DURATION_BUCKETS), dtype=np.int64
    )
    order2_counts = np.zeros(
        (HISTORY_VALUES, K, DESTINATIONS, DURATION_BUCKETS), dtype=np.int64
    )
    order3_counts = np.zeros(
        (
            HISTORY_VALUES,
            HISTORY_VALUES,
            K,
            DESTINATIONS,
            DURATION_BUCKETS,
        ),
        dtype=np.int64,
    )
    np.add.at(state_dest_counts, (current, destination, bucket), 1)
    np.add.at(
        order2_counts,
        (previous_state_1, current, destination, bucket),
        1,
    )
    np.add.at(
        order3_counts,
        (previous_state_2, previous_state_1, current, destination, bucket),
        1,
    )
    return {
        "state_dest_counts": state_dest_counts,
        "order2_counts": order2_counts,
        "order3_counts": order3_counts,
        "fit_rows": np.asarray([len(fit)], dtype=np.int64),
        "terminal_rows_excluded": np.asarray(
            [int(train["next_outcome"].eq(END_STATE).sum())], dtype=np.int64
        ),
    }


def smooth_dwell_counts(
    counts: dict[str, np.ndarray],
    frozen_pmf: np.ndarray,
    strengths: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    tau_state_dest, tau_order2, tau_order3 = strengths
    state_dest_pmf = _smooth_counts(
        counts["state_dest_counts"], frozen_pmf[:, None, :], tau_state_dest
    )
    order2_pmf = _smooth_counts(
        counts["order2_counts"], state_dest_pmf[None, :, :, :], tau_order2
    )
    order3_pmf = _smooth_counts(
        counts["order3_counts"], order2_pmf[None, :, :, :, :], tau_order3
    )
    return {
        "state_pmf": frozen_pmf.copy(),
        "state_dest_counts": counts["state_dest_counts"],
        "state_dest_pmf": state_dest_pmf,
        "order2_counts": counts["order2_counts"],
        "order2_pmf": order2_pmf,
        "order3_counts": counts["order3_counts"],
        "order3_pmf": order3_pmf,
        "training_rows": counts["fit_rows"],
        "terminal_rows_excluded": counts["terminal_rows_excluded"],
        "tau_state_dest": np.asarray([tau_state_dest]),
        "tau_order2": np.asarray([tau_order2]),
        "tau_order3": np.asarray([tau_order3]),
    }


def fit_dwell_kernel(
    train: pd.DataFrame, frozen_pmf: np.ndarray
) -> dict[str, np.ndarray]:
    counts = dwell_count_tensors(train)
    arrays = smooth_dwell_counts(counts, frozen_pmf, EXPECTED_SMOOTHING)
    verify_kernel_arrays(arrays)
    return arrays


def select_smoothing_2024(
    train: pd.DataFrame, frozen_pmf: np.ndarray
) -> pd.DataFrame:
    dates = pd.to_datetime(train["session_date"], errors="raise")
    fit = train.loc[train["next_outcome"].ne(END_STATE)].copy()
    fit_dates = dates.loc[fit.index]
    totals = {
        (alpha1, alpha2, alpha3): [0.0, 0]
        for alpha1 in SMOOTHING_GRID
        for alpha2 in SMOOTHING_GRID
        for alpha3 in SMOOTHING_GRID
    }
    month_rows = []
    for month in range(7, 13):
        start = pd.Timestamp(year=2024, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        history = train.loc[dates.lt(start)].copy()
        validation = fit.loc[fit_dates.ge(start) & fit_dates.lt(end)].copy()
        if history.empty or validation.empty:
            raise AssertionError(f"empty expanding-month smoothing split: {month}")
        counts = dwell_count_tensors(history)
        prev2 = validation["previous_state_2"].to_numpy(dtype=int)
        prev1 = validation["previous_state_1"].to_numpy(dtype=int)
        state = validation["state"].to_numpy(dtype=int)
        destination = validation["next_outcome"].to_numpy(dtype=int)
        bucket = dwell_bucket(validation["duration"].to_numpy(dtype=int))
        for strengths in totals:
            kernel = smooth_dwell_counts(counts, frozen_pmf, strengths)
            probability = kernel["order3_pmf"][
                prev2, prev1, state, destination, bucket
            ]
            loss_sum = float(-np.log(np.clip(probability, EPSILON, 1.0)).sum())
            totals[strengths][0] += loss_sum
            totals[strengths][1] += len(validation)
            month_rows.append(
                {
                    "month": f"2024-{month:02d}",
                    "alpha_state_destination": strengths[0],
                    "alpha_order2": strengths[1],
                    "alpha_order3": strengths[2],
                    "rows": len(validation),
                    "log_loss": loss_sum / len(validation),
                }
            )
    pooled = []
    for strengths, (loss_sum, rows) in totals.items():
        pooled.append(
            {
                "month": "pooled",
                "alpha_state_destination": strengths[0],
                "alpha_order2": strengths[1],
                "alpha_order3": strengths[2],
                "rows": rows,
                "log_loss": loss_sum / rows,
            }
        )
    result = pd.DataFrame(month_rows + pooled)
    ranked = result.loc[result["month"].eq("pooled")].sort_values(
        [
            "log_loss",
            "alpha_state_destination",
            "alpha_order2",
            "alpha_order3",
        ],
        kind="stable",
    )
    best = tuple(
        float(ranked.iloc[0][column])
        for column in (
            "alpha_state_destination",
            "alpha_order2",
            "alpha_order3",
        )
    )
    if best != EXPECTED_SMOOTHING:
        raise AssertionError(
            f"2024-only smoothing selection did not reproduce: {best}"
        )
    result["selected"] = (
        result["alpha_state_destination"].eq(EXPECTED_SMOOTHING[0])
        & result["alpha_order2"].eq(EXPECTED_SMOOTHING[1])
        & result["alpha_order3"].eq(EXPECTED_SMOOTHING[2])
    )
    return result


def verify_kernel_arrays(arrays: dict[str, np.ndarray]) -> None:
    probability_names = (
        "state_pmf",
        "state_dest_pmf",
        "order2_pmf",
        "order3_pmf",
    )
    for name in probability_names:
        values = arrays[name]
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise AssertionError(f"invalid kernel array {name}")
        if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-12):
            raise AssertionError(f"kernel PMF does not normalize: {name}")


def frozen_duration_pmf(hazard: np.ndarray) -> np.ndarray:
    hazard = np.asarray(hazard, dtype=float)
    if hazard.shape != (K, 24):
        raise AssertionError("frozen duration hazard shape drifted")
    pmf = np.zeros((K, DURATION_BUCKETS), dtype=float)
    for state in range(K):
        survival = 1.0
        for index in range(24):
            pmf[state, index] = survival * hazard[state, index]
            survival *= 1.0 - hazard[state, index]
    if not np.allclose(pmf.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("frozen state duration PMF does not normalize")
    return pmf


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def destination_probability(
    previous_state_2: np.ndarray,
    previous_state_1: np.ndarray,
    current_state: np.ndarray,
    destination: int,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    tokens = history_tokens(previous_state_2, previous_state_1, current_state)
    logits = (
        parameters["history_intercept"][None, :]
        + parameters["history_coef"][:, tokens].T
    )
    return softmax(logits)[:, int(destination)]


def advance_distribution(
    distribution: np.ndarray, transition: np.ndarray, horizon: int = MAX_HORIZON
) -> np.ndarray:
    if distribution.ndim != 2 or transition.ndim != 2:
        raise AssertionError("batch convolution requires two-dimensional arrays")
    output = np.zeros_like(distribution)
    maximum_exact = min(horizon, OVERFLOW_DURATION - 1)
    for duration in range(1, maximum_exact + 1):
        output[:, duration:] += (
            distribution[:, : horizon + 1 - duration]
            * transition[:, duration - 1][:, None]
        )
    return output


def route_probabilities(
    anchors: pd.DataFrame,
    route: tuple[int, ...],
    parameters: dict[str, np.ndarray],
    kernel: dict[str, np.ndarray],
    frozen_pmf: np.ndarray,
) -> dict[str, Any]:
    count = len(anchors)
    path_probability = np.ones(count, dtype=float)
    distributions = {
        "history_frozen_state_timed": np.pad(
            np.ones((count, 1), dtype=float), ((0, 0), (0, MAX_HORIZON))
        ),
        "history_destination_timed": np.pad(
            np.ones((count, 1), dtype=float), ((0, 0), (0, MAX_HORIZON))
        ),
        "history_order2_timed": np.pad(
            np.ones((count, 1), dtype=float), ((0, 0), (0, MAX_HORIZON))
        ),
        "history_joint_timed": np.pad(
            np.ones((count, 1), dtype=float), ((0, 0), (0, MAX_HORIZON))
        ),
    }
    previous_state_2 = anchors["previous_state_2"].to_numpy(dtype=int)
    previous_state_1 = anchors["previous_state_1"].to_numpy(dtype=int)
    current_state = np.full(count, int(route[0]), dtype=int)
    for destination in route[1:]:
        p_destination = destination_probability(
            previous_state_2,
            previous_state_1,
            current_state,
            int(destination),
            parameters,
        )
        path_probability *= p_destination
        pmfs = {
            "history_frozen_state_timed": frozen_pmf[current_state],
            "history_destination_timed": kernel["state_dest_pmf"][
                current_state, int(destination)
            ],
            "history_order2_timed": kernel["order2_pmf"][
                previous_state_1, current_state, int(destination)
            ],
            "history_joint_timed": kernel["order3_pmf"][
                previous_state_2,
                previous_state_1,
                current_state,
                int(destination),
            ],
        }
        for model, pmf in pmfs.items():
            transition = p_destination[:, None] * pmf
            distributions[model] = advance_distribution(distributions[model], transition)
        previous_state_2, previous_state_1, current_state = (
            previous_state_1,
            current_state,
            np.full(count, int(destination), dtype=int),
        )
    probabilities: dict[int, dict[str, np.ndarray]] = {}
    for horizon in HORIZONS:
        probabilities[horizon] = {
            "history_path_only": path_probability.copy(),
            **{
                model: distribution[:, : horizon + 1].sum(axis=1)
                for model, distribution in distributions.items()
            },
        }
    return {
        "path_probability": path_probability,
        "distributions": distributions,
        "probabilities": probabilities,
    }


def oriented_actual_completion(
    anchors: pd.DataFrame, route: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    label = np.ones(len(anchors), dtype=bool)
    for step, destination in enumerate(route[1:], start=1):
        label &= anchors[f"future_state_{step}"].to_numpy(dtype=int) == int(destination)
    completion = anchors["duration"].to_numpy(dtype=int).copy()
    for step in range(1, len(route) - 1):
        completion += anchors[f"future_duration_{step}"].to_numpy(dtype=int)
    return label, completion


def score_period(
    anchors: pd.DataFrame,
    cycles: pd.DataFrame,
    parameters: dict[str, np.ndarray],
    kernel: dict[str, np.ndarray],
    frozen_pmf: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    route_rows: list[dict[str, Any]] = []
    for cycle in cycles.itertuples(index=False):
        cycle_core = tuple(int(state) for state in cycle.core)
        selected = anchors.loc[anchors["state"].isin(set(cycle_core))].copy()
        selected = selected.reset_index(drop=True)
        eventual = np.zeros(len(selected), dtype=bool)
        completion = np.full(len(selected), np.nan, dtype=float)
        targets = {horizon: np.zeros(len(selected), dtype=bool) for horizon in HORIZONS}
        probabilities = {
            horizon: {model: np.zeros(len(selected), dtype=float) for model in MODEL_COLUMNS}
            for horizon in HORIZONS
        }
        for current in sorted(set(cycle_core)):
            mask = selected["state"].eq(current).to_numpy()
            state_anchors = selected.loc[mask].reset_index(drop=True)
            local_eventual = np.zeros(len(state_anchors), dtype=bool)
            local_completion = np.full(len(state_anchors), np.nan, dtype=float)
            local_targets = {
                horizon: np.zeros(len(state_anchors), dtype=bool) for horizon in HORIZONS
            }
            local_probabilities = {
                horizon: {
                    model: np.zeros(len(state_anchors), dtype=float)
                    for model in MODEL_COLUMNS
                }
                for horizon in HORIZONS
            }
            routes = oriented_paths(cycle_core, current)
            if not routes:
                raise AssertionError("compatible cycle has no oriented route")
            for route in routes:
                truth, route_completion = oriented_actual_completion(state_anchors, route)
                local_eventual |= truth
                local_completion[truth] = route_completion[truth]
                forecast = route_probabilities(
                    state_anchors, route, parameters, kernel, frozen_pmf
                )
                for horizon in HORIZONS:
                    local_targets[horizon] |= truth & (route_completion <= horizon)
                    for model in MODEL_COLUMNS:
                        local_probabilities[horizon][model] += forecast["probabilities"][
                            horizon
                        ][model]
                route_rows.append(
                    {
                        "period": str(state_anchors["period"].iloc[0]),
                        "cycle_id": cycle.cycle_id,
                        "cycle": cycle.cycle,
                        "current_state": current,
                        "route": "->".join(str(value) for value in route),
                        "transition_count": len(route) - 1,
                    }
                )
            eventual[mask] = local_eventual
            completion[mask] = local_completion
            for horizon in HORIZONS:
                targets[horizon][mask] = local_targets[horizon]
                for model in MODEL_COLUMNS:
                    probabilities[horizon][model][mask] = local_probabilities[horizon][
                        model
                    ]
        metadata = [
            "anchor_id",
            "period",
            "symbol_norm",
            "session_date",
            "quarter",
            "start_timestamp",
            "bar_index_in_session",
            "state",
            "previous_state_1",
            "previous_state_2",
            "duration",
        ]
        for horizon in HORIZONS:
            output = selected[metadata].copy()
            output["cycle_id"] = cycle.cycle_id
            output["cycle"] = cycle.cycle
            output["transition_length"] = int(cycle.transition_length)
            output["horizon"] = horizon
            output["eventual_target"] = eventual.astype(np.int8)
            output["completion_bars"] = completion
            output["target"] = targets[horizon].astype(np.int8)
            for model, column in MODEL_COLUMNS.items():
                values = probabilities[horizon][model]
                if (
                    not np.isfinite(values).all()
                    or values.min(initial=0.0) < 0.0
                    or values.max(initial=0.0) > 1.0 + 1e-9
                ):
                    raise AssertionError(f"invalid cycle probability for {model}")
                output[column] = np.clip(values, EPSILON, 1.0 - EPSILON)
            if not (
                output[MODEL_COLUMNS["history_joint_timed"]]
                <= output[MODEL_COLUMNS["history_path_only"]] + 1e-12
            ).all():
                raise AssertionError("joint completion mass exceeds path marginal")
            rows.append(output)
    scoring = pd.concat(rows, ignore_index=True).sort_values(
        ["anchor_id", "horizon", "cycle_id"], kind="stable"
    ).reset_index(drop=True)
    if scoring.duplicated(["anchor_id", "horizon", "cycle_id"]).any():
        raise AssertionError("duplicate anchor/horizon/cycle score")
    for column in MODEL_COLUMNS.values():
        pivot = scoring.pivot_table(
            index=["anchor_id", "cycle_id"], columns="horizon", values=column
        )
        if column == MODEL_COLUMNS["history_path_only"]:
            if not np.allclose(pivot[6], pivot[12]) or not np.allclose(pivot[12], pivot[24]):
                raise AssertionError("path-only marginal changed with horizon")
        elif not ((pivot[6] <= pivot[12] + 1e-12) & (pivot[12] <= pivot[24] + 1e-12)).all():
            raise AssertionError(f"non-monotonic completion probability: {column}")
    route_manifest = pd.DataFrame(route_rows).drop_duplicates().sort_values(
        ["period", "cycle_id", "current_state", "route"], kind="stable"
    )
    return scoring, route_manifest.reset_index(drop=True)


def binary_losses(target: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    target = np.asarray(target, dtype=float)
    return {
        "log_loss": -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability)),
        "brier": np.square(probability - target),
    }


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


def calibration_rows(
    scoring: pd.DataFrame, period: str, model: str, column: str
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        selected = scoring["horizon"].eq(horizon).to_numpy()
        target = scoring.loc[selected, "target"].to_numpy(dtype=float)
        probability = scoring.loc[selected, column].to_numpy(dtype=float)
        bin_id = np.minimum((probability * 10.0).astype(int), 9)
        ece = 0.0
        supported_errors = []
        for index in range(10):
            mask = bin_id == index
            count = int(mask.sum())
            mean_probability = float(probability[mask].mean()) if count else math.nan
            event_rate = float(target[mask].mean()) if count else math.nan
            error = abs(mean_probability - event_rate) if count else math.nan
            supported = count >= MIN_BIN_SUPPORT
            if count:
                ece += count / len(target) * error
            if supported:
                supported_errors.append(error)
            rows.append(
                {
                    "period": period,
                    "model": model,
                    "horizon": horizon,
                    "bin": index,
                    "count": count,
                    "mean_probability": mean_probability,
                    "event_rate": event_rate,
                    "absolute_error": error,
                    "supported": supported,
                }
            )
        if not supported_errors:
            raise AssertionError("calibration has no supported bin")
        summary[horizon] = {
            "ece": float(ece),
            "maximum_supported_bin_error": float(max(supported_errors)),
        }
    return rows, summary


def top_three_metrics(
    scoring: pd.DataFrame, period: str, model: str, column: str
) -> dict[str, Any]:
    ranked = scoring.sort_values(
        ["anchor_id", "horizon", column, "cycle_id"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    selected = ranked.groupby(["anchor_id", "horizon"], sort=False).cumcount().lt(3)
    positives = int(ranked["target"].sum())
    hits = int(ranked.loc[selected, "target"].sum())
    return {
        "period": period,
        "model": model,
        "anchor_horizons": int(
            ranked[["anchor_id", "horizon"]].drop_duplicates().shape[0]
        ),
        "positive_labels": positives,
        "selected_labels": int(selected.sum()),
        "hits": hits,
        "recall": float(hits / positives) if positives else math.nan,
        "precision": float(hits / selected.sum()) if selected.any() else math.nan,
    }


def evaluate_period(scoring: pd.DataFrame, period: str, seed_offset: int) -> dict[str, Any]:
    target = scoring["target"].to_numpy(dtype=int)
    losses: dict[str, dict[str, np.ndarray]] = {}
    calibration: list[dict[str, Any]] = []
    calibration_summary: dict[str, dict[int, dict[str, float]]] = {}
    metrics: list[dict[str, Any]] = []
    tops: list[dict[str, Any]] = []
    top_summary: dict[str, dict[str, Any]] = {}
    for model, column in MODEL_COLUMNS.items():
        probability = scoring[column].to_numpy(dtype=float)
        losses[model] = binary_losses(target, probability)
        rows, summary = calibration_rows(scoring, period, model, column)
        calibration.extend(rows)
        calibration_summary[model] = summary
        for horizon in HORIZONS:
            mask = scoring["horizon"].eq(horizon).to_numpy()
            metrics.append(
                {
                    "period": period,
                    "model": model,
                    "horizon": horizon,
                    "rows": int(mask.sum()),
                    "positives": int(target[mask].sum()),
                    "prevalence": float(target[mask].mean()),
                    "log_loss": float(losses[model]["log_loss"][mask].mean()),
                    "brier": float(losses[model]["brier"][mask].mean()),
                    "ece": summary[horizon]["ece"],
                    "maximum_supported_bin_error": summary[horizon][
                        "maximum_supported_bin_error"
                    ],
                }
            )
        top = top_three_metrics(scoring, period, model, column)
        tops.append(top)
        top_summary[model] = top

    per_cycle_rows = []
    for cycle_id, indices in scoring.groupby("cycle_id", sort=True).groups.items():
        positions = np.asarray(indices, dtype=int)
        row: dict[str, Any] = {
            "period": period,
            "cycle_id": cycle_id,
            "cycle": str(scoring.loc[positions[0], "cycle"]),
            "rows": len(positions),
            "positives": int(target[positions].sum()),
        }
        for model in MODEL_COLUMNS:
            row[f"{model}_log_loss"] = float(
                losses[model]["log_loss"][positions].mean()
            )
            row[f"{model}_brier"] = float(losses[model]["brier"][positions].mean())
        per_cycle_rows.append(row)
    per_cycle = pd.DataFrame(per_cycle_rows)

    comparisons: list[dict[str, Any]] = []
    gates: dict[str, dict[str, Any]] = {}
    for comparison_index, (
        candidate,
        baseline,
        name,
        relative_requirement,
        recall_requirement,
    ) in enumerate(COMPARISON_SPECS):
        loss_rows = []
        for loss_index, loss_name in enumerate(("log_loss", "brier")):
            difference = losses[candidate][loss_name] - losses[baseline][loss_name]
            daily = (
                pd.DataFrame(
                    {"session_date": scoring["session_date"], "difference": difference}
                )
                .groupby("session_date", sort=True)["difference"]
                .mean()
                .to_numpy(dtype=float)
            )
            mean, low, high = moving_block_bounds(
                daily, SEED + seed_offset + comparison_index * 100 + loss_index
            )
            horizons = pd.Series(difference).groupby(
                scoring["horizon"].reset_index(drop=True)
            ).mean()
            quarters = pd.Series(difference).groupby(
                scoring["quarter"].reset_index(drop=True)
            ).mean()
            deletions = {
                symbol: float(
                    difference[
                        scoring["symbol_norm"].astype(str).ne(symbol).to_numpy()
                    ].mean()
                )
                for symbol in sorted(scoring["symbol_norm"].astype(str).unique())
            }
            baseline_mean = float(losses[baseline][loss_name].mean())
            row = {
                "period": period,
                "comparison": name,
                "candidate": candidate,
                "baseline": baseline,
                "loss": loss_name,
                "row_mean_difference": float(difference.mean()),
                "daily_mean_difference": mean,
                "daily_ci_low": low,
                "daily_ci_high": high,
                "baseline_mean_loss": baseline_mean,
                "relative_improvement": float(-difference.mean() / baseline_mean),
                "negative_horizon_count": int((horizons < 0.0).sum()),
                "negative_quarter_count": int((quarters < 0.0).sum()),
                "leave_one_symbol_max_difference": max(deletions.values()),
                "leave_one_symbol_all_negative": bool(max(deletions.values()) < 0.0),
            }
            comparisons.append(row)
            loss_rows.append(row)
        negative_cycles = int(
            (
                per_cycle[f"{candidate}_log_loss"]
                < per_cycle[f"{baseline}_log_loss"]
            ).sum()
        )
        recall_difference = float(
            top_summary[candidate]["recall"] - top_summary[baseline]["recall"]
        )
        ece_pass = all(
            calibration_summary[candidate][horizon]["ece"]
            <= calibration_summary[baseline][horizon]["ece"]
            for horizon in HORIZONS
        )
        maximum_error_pass = all(
            calibration_summary[candidate][horizon]["maximum_supported_bin_error"]
            <= calibration_summary[baseline][horizon]["maximum_supported_bin_error"]
            + 0.01
            for horizon in HORIZONS
        )
        gate = {
            "intervals_pass": all(row["daily_ci_high"] < 0.0 for row in loss_rows),
            "relative_log_loss_improvement": loss_rows[0]["relative_improvement"],
            "relative_log_loss_requirement": relative_requirement,
            "relative_log_loss_pass": loss_rows[0]["relative_improvement"]
            >= relative_requirement,
            "horizons_pass": all(row["negative_horizon_count"] == 3 for row in loss_rows),
            "quarters_pass": all(row["negative_quarter_count"] == 4 for row in loss_rows),
            "stock_deletions_pass": all(
                row["leave_one_symbol_all_negative"] for row in loss_rows
            ),
            "ece_pass": ece_pass,
            "maximum_supported_bin_error_pass": maximum_error_pass,
            "negative_cycle_count": negative_cycles,
            "per_cycle_pass": negative_cycles >= 15,
            "top_three_recall_difference": recall_difference,
            "top_three_recall_requirement": recall_requirement,
            "top_three_recall_pass": recall_difference >= recall_requirement,
        }
        gate["pass"] = bool(
            gate["intervals_pass"]
            and gate["relative_log_loss_pass"]
            and gate["horizons_pass"]
            and gate["quarters_pass"]
            and gate["stock_deletions_pass"]
            and gate["ece_pass"]
            and gate["maximum_supported_bin_error_pass"]
            and gate["per_cycle_pass"]
            and gate["top_three_recall_pass"]
        )
        gates[name] = gate

    path_baseline = "history_path_only"
    path_horizon_pass = {}
    for loss_name in ("log_loss", "brier"):
        difference = losses["history_joint_timed"][loss_name] - losses[path_baseline][
            loss_name
        ]
        horizon_means = pd.Series(difference).groupby(
            scoring["horizon"].reset_index(drop=True)
        ).mean()
        path_horizon_pass[loss_name] = bool((horizon_means < 0.0).all())
    path_recall_difference = float(
        top_summary["history_joint_timed"]["recall"]
        - top_summary[path_baseline]["recall"]
    )
    gates["joint_vs_path_only_sanity"] = {
        "log_loss_lower_every_horizon": path_horizon_pass["log_loss"],
        "brier_lower_every_horizon": path_horizon_pass["brier"],
        "top_three_recall_difference": path_recall_difference,
        "top_three_recall_pass": path_recall_difference >= 0.005,
        "pass": bool(
            path_horizon_pass["log_loss"]
            and path_horizon_pass["brier"]
            and path_recall_difference >= 0.005
        ),
    }

    support_rows = []
    support_pass = True
    for horizon in HORIZONS:
        selected = scoring.loc[scoring["horizon"].eq(horizon)]
        positives = selected.groupby("cycle_id", sort=True)["target"].sum()
        row = {
            "period": period,
            "horizon": horizon,
            "rows": len(selected),
            "positives": int(selected["target"].sum()),
            "cycles": int(selected["cycle_id"].nunique()),
            "minimum_cycle_positives": int(positives.min()),
            "stocks": int(selected["symbol_norm"].nunique()),
            "quarters": int(selected["quarter"].nunique()),
            "current_states": int(selected["state"].nunique()),
        }
        row["pass"] = bool(
            row["rows"] >= 300_000
            and row["positives"] >= 8_000
            and row["cycles"] == 20
            and row["minimum_cycle_positives"] >= 40
            and row["stocks"] >= 18
            and row["quarters"] == 4
            and row["current_states"] == K
        )
        support_pass &= row["pass"]
        support_rows.append(row)
    return {
        "metrics": pd.DataFrame(metrics),
        "calibration": pd.DataFrame(calibration),
        "top_three": pd.DataFrame(tops),
        "per_cycle": per_cycle,
        "comparisons": pd.DataFrame(comparisons),
        "support": pd.DataFrame(support_rows),
        "support_pass": bool(support_pass),
        "gates": gates,
    }


def self_tests() -> dict[str, Any]:
    assert canonical_cycle((2, 5, 1)) == canonical_cycle((5, 1, 2))
    assert history_tokens(np.asarray([8]), np.asarray([8]), np.asarray([0]))[0] == 640
    distribution = np.zeros((1, MAX_HORIZON + 1), dtype=float)
    distribution[:, 0] = 1.0
    first = np.zeros((1, DURATION_BUCKETS), dtype=float)
    second = np.zeros_like(first)
    first[:, 0] = 0.8
    first[:, 1] = 0.2
    second[:, 1] = 1.0
    result = advance_distribution(distribution, first)
    result = advance_distribution(result, second)
    assert abs(result[0, 3] - 0.8) < 1e-12
    assert abs(result[0, 4] - 0.2) < 1e-12
    assert result[0, :3].sum() == 0.0

    rng = np.random.default_rng(SEED)
    left = rng.random(4)
    left /= left.sum()
    right = rng.random(5)
    right /= right.sum()
    left_transition = np.zeros((1, DURATION_BUCKETS))
    right_transition = np.zeros((1, DURATION_BUCKETS))
    left_transition[0, :4] = left
    right_transition[0, :5] = right
    dynamic = advance_distribution(distribution, left_transition)
    dynamic = advance_distribution(dynamic, right_transition)
    brute = np.convolve(left, right)
    assert np.allclose(dynamic[0, 2 : 2 + len(brute)], brute)

    coupled_a = np.zeros((DURATION_BUCKETS,))
    coupled_b = np.zeros((DURATION_BUCKETS,))
    coupled_a[0] = 1.0
    coupled_b[5] = 1.0
    mixture = 0.5 * coupled_a + 0.5 * coupled_b
    assert coupled_a[:3].sum() != mixture[:3].sum()
    return {
        "synthetic_convolution": True,
        "brute_force_match": True,
        "destination_duration_coupling_distinguishable": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }


def fit_only() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    tests = self_tests()
    path_gates = json.loads(PATH_GATES.read_text())
    path_audit = json.loads(PATH_AUDIT.read_text())
    if path_gates.get("history_retained") is not True or path_audit.get("all_passed") is not True:
        raise AssertionError("retained path predecessor or audit failed")
    pre_snapshot = snapshot_protected_tree()
    write_json(OUT / "prospective_shadow_pre_snapshot.json", pre_snapshot)
    fit_sources = {
        "contract.json": CONTRACT,
        "runner.py": Path(__file__),
        "train_2024_filtered_runs.csv": RUN_2024,
        "path_model_parameters.npz": PATH_PARAMETERS,
        "path_gates.json": PATH_GATES,
        "path_independent_artifact_audit.json": PATH_AUDIT,
        "frozen_semimarkov_parameters.npz": SEMIMARKOV_PARAMETERS,
        "fixed_cycle_shuffled_nulls.csv": CYCLE_PATH,
    }
    write_json(
        OUT / "fit_source_hashes.json",
        {name: sha256(path) for name, path in fit_sources.items()},
    )
    train = load_runs(RUN_2024, 2024, "train_2024", scoring=False)
    with np.load(SEMIMARKOV_PARAMETERS) as archive:
        frozen_pmf = frozen_duration_pmf(archive["duration_hazard"])
    smoothing_selection = select_smoothing_2024(train, frozen_pmf)
    smoothing_selection.to_csv(OUT / "smoothing_selection_2024.csv", index=False)
    kernel = fit_dwell_kernel(train, frozen_pmf)
    np.savez_compressed(OUT / "kernel_parameters.npz", **kernel)
    cycles = load_cycles()
    cycles.drop(columns="core").to_csv(OUT / "fixed_cycles.csv", index=False)
    training_audit = {
        "rows": len(train),
        "stocks": int(train["symbol_norm"].nunique()),
        "dates": int(train["session_date"].nunique()),
        "states": int(train["state"].nunique()),
        "destination_classes": sorted(train["next_outcome"].astype(int).unique()),
        "nonterminal_fit_rows": int(kernel["training_rows"][0]),
        "terminal_rows_excluded": int(kernel["terminal_rows_excluded"][0]),
        "overflow_durations": int(train["duration"].ge(OVERFLOW_DURATION).sum()),
        "maximum_duration": int(train["duration"].max()),
        "order3_supported_cells": int(
            (kernel["order3_counts"].sum(axis=-1) > 0).sum()
        ),
        "order2_supported_cells": int(
            (kernel["order2_counts"].sum(axis=-1) > 0).sum()
        ),
        "state_destination_supported_cells": int(
            (kernel["state_dest_counts"].sum(axis=-1) > 0).sum()
        ),
    }
    write_json(OUT / "training_audit.json", training_audit)
    write_json(OUT / "self_tests.json", tests)
    manifest = {
        "contract_id": json.loads(CONTRACT.read_text())["contract_id"],
        "dwell_buckets": "1..23 exact; 24 means duration >= 24",
        "terminal_session_end_rows_excluded": True,
        "base_duration_pmf": "frozen 2024 semi-Markov current-state PMF",
        "tau_state_dest": TAU_STATE_DEST,
        "tau_order2": TAU_ORDER2,
        "tau_order3": TAU_ORDER3,
        "smoothing_grid": list(SMOOTHING_GRID),
        "smoothing_selection_sha256": sha256(OUT / "smoothing_selection_2024.csv"),
        "kernel_shapes": {
            name: list(values.shape)
            for name, values in kernel.items()
            if name.endswith("_pmf") or name.endswith("_counts")
        },
        "training_audit": training_audit,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "kernel_manifest.json", manifest)
    fit_complete = {
        "kernel_parameters_sha256": sha256(OUT / "kernel_parameters.npz"),
        "kernel_manifest_sha256": sha256(OUT / "kernel_manifest.json"),
        "smoothing_selection_sha256": sha256(OUT / "smoothing_selection_2024.csv"),
        "fit_source_hashes_sha256": sha256(OUT / "fit_source_hashes.json"),
        "pre_snapshot_sha256": sha256(OUT / "prospective_shadow_pre_snapshot.json"),
        "scoring_outcomes_opened": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json(OUT / "fit_complete.json", fit_complete)
    return fit_complete


def main_score() -> dict[str, Any]:
    fit_complete_path = OUT / "fit_complete.json"
    if not fit_complete_path.is_file():
        raise AssertionError("run --fit-only and the independent pre-score audit first")
    fit_complete = json.loads(fit_complete_path.read_text())
    pre_audit_path = OUT / "pre_score_audit.json"
    if not pre_audit_path.is_file() or json.loads(pre_audit_path.read_text()).get(
        "all_passed"
    ) is not True:
        raise AssertionError("independent pre-score audit has not passed")
    if sha256(OUT / "kernel_parameters.npz") != fit_complete["kernel_parameters_sha256"]:
        raise AssertionError("frozen kernel parameters changed after fit-only")
    recorded_fit_sources = json.loads((OUT / "fit_source_hashes.json").read_text())
    current_fit_sources = {
        "contract.json": sha256(CONTRACT),
        "runner.py": sha256(Path(__file__)),
        "train_2024_filtered_runs.csv": sha256(RUN_2024),
        "path_model_parameters.npz": sha256(PATH_PARAMETERS),
        "path_gates.json": sha256(PATH_GATES),
        "path_independent_artifact_audit.json": sha256(PATH_AUDIT),
        "frozen_semimarkov_parameters.npz": sha256(SEMIMARKOV_PARAMETERS),
        "fixed_cycle_shuffled_nulls.csv": sha256(CYCLE_PATH),
    }
    if current_fit_sources != recorded_fit_sources:
        raise AssertionError("fit source changed after the pre-score freeze")
    pre_snapshot = json.loads((OUT / "prospective_shadow_pre_snapshot.json").read_text())
    current_snapshot = snapshot_protected_tree()
    if not snapshots_equal(pre_snapshot, current_snapshot):
        raise AssertionError("prospective movement tree changed before joint scoring")
    evaluation_sources = {
        "test_2025_filtered_runs.csv": RUN_2025,
        "backward_2023_filtered_runs.parquet": RUN_2023,
    }
    write_json(
        OUT / "evaluation_source_hashes.json",
        {name: sha256(path) for name, path in evaluation_sources.items()},
    )
    kernel = dict(np.load(OUT / "kernel_parameters.npz"))
    verify_kernel_arrays(kernel)
    parameters = dict(np.load(PATH_PARAMETERS))
    if not np.array_equal(parameters["history_classes"], np.arange(DESTINATIONS)):
        raise AssertionError("retained destination class order changed")
    with np.load(SEMIMARKOV_PARAMETERS) as archive:
        frozen_pmf = frozen_duration_pmf(archive["duration_hazard"])
    cycles = load_cycles()

    all_metrics = []
    all_calibration = []
    all_top = []
    all_cycle = []
    all_comparisons = []
    all_support = []
    all_routes = []
    gates: dict[str, Any] = {"periods": {}}
    for period, path, year, seed_offset in (
        ("2025", RUN_2025, 2025, 1000),
        ("2023", RUN_2023, 2023, 2000),
    ):
        anchors = load_runs(path, year, period, scoring=True)
        scoring, routes = score_period(
            anchors, cycles, parameters, kernel, frozen_pmf
        )
        scoring.to_parquet(OUT / f"joint_completion_scoring_{period}.parquet", index=False)
        routes.to_csv(OUT / f"route_manifest_{period}.csv", index=False)
        evaluation = evaluate_period(scoring, period, seed_offset)
        all_metrics.append(evaluation["metrics"])
        all_calibration.append(evaluation["calibration"])
        all_top.append(evaluation["top_three"])
        all_cycle.append(evaluation["per_cycle"])
        all_comparisons.append(evaluation["comparisons"])
        all_support.append(evaluation["support"])
        all_routes.append(routes)
        gates["periods"][period] = {
            "support_pass": evaluation["support_pass"],
            **evaluation["gates"],
        }
    metrics = pd.concat(all_metrics, ignore_index=True)
    calibration = pd.concat(all_calibration, ignore_index=True)
    top = pd.concat(all_top, ignore_index=True)
    per_cycle = pd.concat(all_cycle, ignore_index=True)
    comparisons = pd.concat(all_comparisons, ignore_index=True)
    support = pd.concat(all_support, ignore_index=True)
    routes = pd.concat(all_routes, ignore_index=True).drop_duplicates()
    metrics.to_csv(OUT / "joint_completion_metrics.csv", index=False)
    calibration.to_csv(OUT / "joint_completion_calibration.csv", index=False)
    top.to_csv(OUT / "joint_completion_top_three.csv", index=False)
    per_cycle.to_csv(OUT / "joint_completion_per_cycle.csv", index=False)
    comparisons.to_csv(OUT / "joint_completion_comparisons.csv", index=False)
    support.to_csv(OUT / "joint_completion_support.csv", index=False)
    routes.to_csv(OUT / "route_manifest.csv", index=False)

    gates["joint_completion_retained"] = bool(
        all(gates["periods"][period]["support_pass"] for period in ("2025", "2023"))
        and all(
            gates["periods"][period][comparison]["pass"]
            for period in ("2025", "2023")
            for comparison in (
                "joint_vs_frozen_state_timed",
                "joint_vs_destination_timed",
            )
        )
        and all(
            gates["periods"][period]["joint_vs_path_only_sanity"]["pass"]
            for period in ("2025", "2023")
        )
    )
    gates["destination_conditioned_timing_retained"] = bool(
        all(gates["periods"][period]["support_pass"] for period in ("2025", "2023"))
        and all(
            gates["periods"][period]["destination_vs_frozen_state_timed"]["pass"]
            for period in ("2025", "2023")
        )
    )
    gates["research_only"] = True
    gates["live_ordering_enabled"] = False
    gates["order_placement"] = "disabled"
    gates["economic_edge_claim"] = False
    write_json(OUT / "gates.json", gates)
    post_snapshot = snapshot_protected_tree()
    if not snapshots_equal(pre_snapshot, post_snapshot):
        raise AssertionError("prospective movement tree changed during joint scoring")
    write_json(OUT / "prospective_shadow_post_snapshot.json", post_snapshot)
    summary = {
        "algorithm": "joint_history_destination_conditioned_dwell_loop_completion",
        "fit_period": 2024,
        "scoring_periods": [2025, 2023],
        "horizons": list(HORIZONS),
        "gates": gates,
        "metrics": metrics.to_dict(orient="records"),
        "top_three": top.to_dict(orient="records"),
        "prospective_shadow_unchanged": True,
        "interpretation": (
            "Loop completion identity/timing research only. No price direction, "
            "return, range, economic edge, P&L, order, or deployment claim."
        ),
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
