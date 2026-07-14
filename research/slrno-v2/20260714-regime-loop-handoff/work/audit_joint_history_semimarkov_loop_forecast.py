"""Independent audit for the research-only joint history/dwell loop forecast.

``--pre-score-only`` uses 2024 inputs only and must pass before the production
runner is allowed to open 2025 or backward-2023 run outcomes.  The default mode
performs the later full scoring/metric/gate reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
CONTRACT = HERE / "contracts/20260710-joint-history-semimarkov-loop-completion-v1.json"
RUNNER = HERE / "run_joint_history_semimarkov_loop_forecast.py"
STATE_ROOT = Path("/private/tmp/stocker_causal_semimarkov_regime_loops_20260710")
PATH_ROOT = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710")
BACKWARD_ROOT = Path(
    "/private/tmp/stocker_sealed_backward_2023_complete_detector_20260710"
)
ARTIFACT = Path("/private/tmp/stocker_joint_history_semimarkov_loop_completion_20260710")
TRAIN = STATE_ROOT / "train_2024_filtered_runs.csv"
RUN_2025 = STATE_ROOT / "test_2025_filtered_runs.csv"
RUN_2023 = BACKWARD_ROOT / "backward_2023_filtered_runs.parquet"
SEMIMARKOV = STATE_ROOT / "frozen_semimarkov_parameters.npz"
CYCLES = STATE_ROOT / "fixed_cycle_shuffled_nulls.csv"
PATH_PARAMETERS = PATH_ROOT / "model_parameters.npz"
PATH_GATES = PATH_ROOT / "gates.json"
PATH_AUDIT = PATH_ROOT / "independent_artifact_audit.json"

K = 8
END = 8
HISTORY_VALUES = 9
DESTINATIONS = 9
BUCKETS = 24
OVERFLOW = 24
GRID = (64.0, 256.0, 1024.0)
SELECTED = (256.0, 256.0, 1024.0)
EPS = 1e-12
SEED = 20260710
HORIZONS = (6, 12, 24)
MAX_HORIZON = max(HORIZONS)
MAX_START_BAR = 53
MIN_BIN_SUPPORT = 500

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


def snapshot() -> dict[str, Any]:
    rows = []
    seen: set[Path] = set()
    for root in PROTECTED_PATHS:
        paths = [root]
        if root.is_dir():
            paths.extend(sorted(root.rglob("*")))
        for path in paths:
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            details = path.lstat()
            if path.is_symlink():
                kind = "symlink"
                content_hash = hashlib.sha256(os.readlink(path).encode()).hexdigest()
            elif path.is_dir():
                kind = "directory"
                content_hash = None
            else:
                kind = "file"
                content_hash = digest(path)
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


def load_training() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN)
    frame["symbol_norm"] = frame["symbol_norm"].astype(str)
    frame["session_date"] = frame["session_date"].astype(str)
    frame["state"] = pd.to_numeric(frame["state"], errors="raise").astype(int)
    frame["duration"] = pd.to_numeric(frame["duration"], errors="raise").astype(int)
    frame["start_pos"] = pd.to_numeric(frame["start_pos"], errors="raise").astype(int)
    frame = frame.sort_values(
        ["symbol_norm", "session_date", "start_pos"], kind="stable"
    ).reset_index(drop=True)
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    assert set(dates.dt.year.unique()) == {2024}
    grouped = frame.groupby(["symbol_norm", "session_date"], sort=False)["state"]
    previous1 = grouped.shift(1).fillna(END).astype(int)
    previous2 = grouped.shift(2).fillna(END).astype(int)
    next_state = grouped.shift(-1)
    assert np.array_equal(previous1, frame["previous_state_1"].astype(int))
    assert np.array_equal(previous2, frame["previous_state_2"].astype(int))
    assert np.array_equal(next_state.notna(), frame["has_next_state"].astype(bool))
    stored_next = pd.to_numeric(frame["next_state"], errors="coerce")
    assert np.array_equal(
        next_state.loc[next_state.notna()].astype(int),
        stored_next.loc[next_state.notna()].astype(int),
    )
    frame["next_outcome"] = next_state.fillna(END).astype(int)
    assert not frame["next_outcome"].eq(frame["state"]).any()
    return frame


def base_pmf() -> np.ndarray:
    with np.load(SEMIMARKOV) as archive:
        hazard = archive["duration_hazard"].copy()
    output = np.zeros((K, BUCKETS), dtype=float)
    for state in range(K):
        survival = 1.0
        for index in range(BUCKETS):
            output[state, index] = survival * hazard[state, index]
            survival *= 1.0 - hazard[state, index]
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-12)
    return output


def counts(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    fit = frame.loc[frame["next_outcome"].ne(END)]
    prev2 = fit["previous_state_2"].to_numpy(int)
    prev1 = fit["previous_state_1"].to_numpy(int)
    state = fit["state"].to_numpy(int)
    destination = fit["next_outcome"].to_numpy(int)
    bucket = np.minimum(fit["duration"].to_numpy(int), OVERFLOW) - 1
    q1 = np.zeros((K, DESTINATIONS, BUCKETS), dtype=np.int64)
    q2 = np.zeros((HISTORY_VALUES, K, DESTINATIONS, BUCKETS), dtype=np.int64)
    q3 = np.zeros(
        (HISTORY_VALUES, HISTORY_VALUES, K, DESTINATIONS, BUCKETS),
        dtype=np.int64,
    )
    np.add.at(q1, (state, destination, bucket), 1)
    np.add.at(q2, (prev1, state, destination, bucket), 1)
    np.add.at(q3, (prev2, prev1, state, destination, bucket), 1)
    return {"q1": q1, "q2": q2, "q3": q3}


def posterior(count: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    result = (count + strength * prior) / (
        count.sum(axis=-1, keepdims=True) + strength
    )
    assert np.isfinite(result).all()
    assert np.allclose(result.sum(axis=-1), 1.0, atol=1e-12)
    return result


def kernel(
    count: dict[str, np.ndarray], base: np.ndarray, strengths: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    q1 = posterior(count["q1"], base[:, None, :], strengths[0])
    q2 = posterior(count["q2"], q1[None, :, :, :], strengths[1])
    q3 = posterior(count["q3"], q2[None, :, :, :, :], strengths[2])
    return {"q1": q1, "q2": q2, "q3": q3}


def smoothing_grid(frame: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    fit = frame.loc[frame["next_outcome"].ne(END)].copy()
    fit_dates = dates.loc[fit.index]
    aggregate = {
        (a1, a2, a3): [0.0, 0]
        for a1 in GRID
        for a2 in GRID
        for a3 in GRID
    }
    for month in range(7, 13):
        start = pd.Timestamp(2024, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        history = frame.loc[dates.lt(start)]
        validation = fit.loc[fit_dates.ge(start) & fit_dates.lt(end)]
        count = counts(history)
        prev2 = validation["previous_state_2"].to_numpy(int)
        prev1 = validation["previous_state_1"].to_numpy(int)
        state = validation["state"].to_numpy(int)
        destination = validation["next_outcome"].to_numpy(int)
        bucket = np.minimum(validation["duration"].to_numpy(int), OVERFLOW) - 1
        for strengths in aggregate:
            q3 = kernel(count, base, strengths)["q3"]
            probability = q3[prev2, prev1, state, destination, bucket]
            aggregate[strengths][0] += float(-np.log(np.clip(probability, EPS, 1.0)).sum())
            aggregate[strengths][1] += len(validation)
    rows = [
        {
            "alpha_state_destination": key[0],
            "alpha_order2": key[1],
            "alpha_order3": key[2],
            "rows": value[1],
            "log_loss": value[0] / value[1],
        }
        for key, value in aggregate.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["log_loss", "alpha_state_destination", "alpha_order2", "alpha_order3"],
        kind="stable",
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def check_joint_normalization(saved: dict[str, np.ndarray]) -> dict[str, float]:
    parameters = dict(np.load(PATH_PARAMETERS))
    contexts = np.indices((HISTORY_VALUES, HISTORY_VALUES, K)).reshape(3, -1).T
    prev2, prev1, state = contexts.T
    token = ((prev2 * HISTORY_VALUES + prev1) * K + state).astype(int)
    logits = parameters["history_intercept"][None, :] + parameters["history_coef"][:, token].T
    destination = softmax(logits)
    q3 = saved["order3_pmf"][prev2, prev1, state]
    joint = destination[:, :, None] * q3
    normalization_error = float(np.max(np.abs(joint.sum(axis=(1, 2)) - 1.0)))
    marginal_error = float(np.max(np.abs(joint.sum(axis=2) - destination)))
    return {
        "normalization_error": normalization_error,
        "destination_marginal_error": marginal_error,
    }


def dp_self_test() -> float:
    left = np.asarray([0.25, 0.75])
    right = np.asarray([0.6, 0.4])
    brute = np.convolve(left, right)
    distribution = np.zeros(8)
    distribution[0] = 1.0
    for transition in (left, right):
        updated = np.zeros_like(distribution)
        for duration, mass in enumerate(transition, start=1):
            updated[duration:] += distribution[:-duration] * mass
        distribution = updated
    return float(np.max(np.abs(distribution[2 : 2 + len(brute)] - brute)))


def pre_score_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, details: Any = None) -> None:
        checks.append({"name": name, "pass": bool(condition), "details": details})
        if not condition:
            raise AssertionError(f"pre-score audit failed: {name}: {details}")

    contract = json.loads(CONTRACT.read_text())
    check("contract_research_only", contract["research_only"] is True)
    check("contract_no_2026", contract["periods"]["2026_permitted"] is False)
    check(
        "contract_selected_smoothing",
        tuple(
            contract["joint_kernel"]["smoothing"]["selection"]["selected_tuple"]
        )
        == SELECTED,
    )
    recorded_sources = json.loads((ARTIFACT / "fit_source_hashes.json").read_text())
    current_sources = {
        "contract.json": digest(CONTRACT),
        "runner.py": digest(RUNNER),
        "train_2024_filtered_runs.csv": digest(TRAIN),
        "path_model_parameters.npz": digest(PATH_PARAMETERS),
        "path_gates.json": digest(PATH_ROOT / "gates.json"),
        "path_independent_artifact_audit.json": digest(
            PATH_ROOT / "independent_artifact_audit.json"
        ),
        "frozen_semimarkov_parameters.npz": digest(SEMIMARKOV),
        "fixed_cycle_shuffled_nulls.csv": digest(CYCLES),
    }
    check("fit_source_hashes", recorded_sources == current_sources)
    check(
        "no_scoring_artifacts_yet",
        not any(ARTIFACT.glob("joint_completion_scoring_*.parquet"))
        and not (ARTIFACT / "summary.json").exists()
        and not (ARTIFACT / "gates.json").exists(),
    )
    train = load_training()
    frozen = base_pmf()
    reconstructed_counts = counts(train)
    reconstructed = kernel(reconstructed_counts, frozen, SELECTED)
    saved = dict(np.load(ARTIFACT / "kernel_parameters.npz"))
    count_errors = {
        "state_dest": int(
            np.max(np.abs(saved["state_dest_counts"] - reconstructed_counts["q1"]))
        ),
        "order2": int(
            np.max(np.abs(saved["order2_counts"] - reconstructed_counts["q2"]))
        ),
        "order3": int(
            np.max(np.abs(saved["order3_counts"] - reconstructed_counts["q3"]))
        ),
    }
    probability_errors = {
        "state": float(np.max(np.abs(saved["state_pmf"] - frozen))),
        "state_dest": float(
            np.max(np.abs(saved["state_dest_pmf"] - reconstructed["q1"]))
        ),
        "order2": float(np.max(np.abs(saved["order2_pmf"] - reconstructed["q2"]))),
        "order3": float(np.max(np.abs(saved["order3_pmf"] - reconstructed["q3"]))),
    }
    check("kernel_counts_exact", max(count_errors.values()) == 0, count_errors)
    check(
        "kernel_probabilities_exact",
        max(probability_errors.values()) < 1e-12,
        probability_errors,
    )
    check(
        "terminal_rows_excluded",
        int(saved["terminal_rows_excluded"][0])
        == int(train["next_outcome"].eq(END).sum()),
    )
    grid = smoothing_grid(train, frozen)
    best = tuple(
        float(grid.iloc[0][column])
        for column in (
            "alpha_state_destination",
            "alpha_order2",
            "alpha_order3",
        )
    )
    check("smoothing_selection_reproduced", best == SELECTED, best)
    stored_grid = pd.read_csv(ARTIFACT / "smoothing_selection_2024.csv")
    stored_pooled = stored_grid.loc[stored_grid["month"].eq("pooled")].sort_values(
        ["log_loss", "alpha_state_destination", "alpha_order2", "alpha_order3"],
        kind="stable",
    )
    check(
        "smoothing_loss_reproduced",
        abs(float(stored_pooled.iloc[0]["log_loss"]) - float(grid.iloc[0]["log_loss"]))
        < 1e-12,
    )
    joint = check_joint_normalization(saved)
    check("joint_normalizes", joint["normalization_error"] < 1e-12, joint)
    check("destination_marginal_exact", joint["destination_marginal_error"] < 1e-12, joint)
    check("dp_bruteforce", dp_self_test() < 1e-12, dp_self_test())
    cycle_source = pd.read_csv(CYCLES)
    check("twenty_cycles", len(cycle_source) == 20)
    check("prospective_shadow_unchanged", snapshot() == json.loads((ARTIFACT / "prospective_shadow_pre_snapshot.json").read_text()))
    runner_text = RUNNER.read_text()
    check("joint_runner_does_not_import_shadow", "import frozen_loop_movement_shadow" not in runner_text)
    forbidden = ("place_order(", "submit_order(", "broker_client", "position_size(")
    check(
        "no_execution_surface",
        not any(fragment in runner_text.lower() for fragment in forbidden),
    )
    result = {
        "all_passed": True,
        "check_count": len(checks),
        "checks": checks,
        "count_errors": count_errors,
        "probability_errors": probability_errors,
        "smoothing_best": best,
        "joint_errors": joint,
        "scoring_outcomes_opened": False,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    (ARTIFACT / "pre_score_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def canonical_cycle(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        raise AssertionError("empty cycle")
    return min(values[index:] + values[:index] for index in range(len(values)))


def compatible_routes(core: tuple[int, ...], current: int) -> list[tuple[int, ...]]:
    return sorted(
        {
            core[index:] + core[:index] + (int(current),)
            for index, state in enumerate(core)
            if int(state) == int(current)
        }
    )


def independent_cycles() -> list[dict[str, Any]]:
    source = pd.read_csv(CYCLES)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for index, value in enumerate(source["cycle"].astype(str), start=1):
        closed = tuple(int(part) for part in value.split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid cycle {value}")
        core = canonical_cycle(closed[:-1])
        if core in seen or len(core) not in (2, 3, 4):
            raise AssertionError(f"duplicate or invalid cycle {value}")
        if any(left == right for left, right in zip(core, core[1:] + core[:1])):
            raise AssertionError(f"self-transition in cycle {value}")
        seen.add(core)
        rows.append(
            {
                "cycle_id": f"cycle_{index:02d}",
                "cycle": "->".join(str(state) for state in core + (core[0],)),
                "transition_length": len(core),
                "core": core,
            }
        )
    if len(rows) != 20:
        raise AssertionError("expected twenty frozen cycles")
    copied = pd.read_csv(ARTIFACT / "fixed_cycles.csv")
    expected = pd.DataFrame(rows).drop(columns="core")
    pd.testing.assert_frame_equal(expected, copied, check_dtype=False)
    return rows


def load_scoring_runs(path: Path, expected_year: int, period: str) -> pd.DataFrame:
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
    if not required.issubset(frame.columns):
        raise AssertionError(f"{period} missing run columns: {sorted(required - set(frame.columns))}")
    output = frame.copy()
    output["symbol_norm"] = output["symbol_norm"].astype(str)
    output["session_date"] = output["session_date"].astype(str)
    for column in ("state", "duration", "start_pos", "previous_state_1", "previous_state_2"):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(int)
    output["start_timestamp"] = pd.to_datetime(
        output["start_timestamp"], utc=True, errors="raise"
    )
    output = output.sort_values(
        ["symbol_norm", "session_date", "start_pos"], kind="stable"
    ).reset_index(drop=True)
    dates = pd.to_datetime(output["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {expected_year} or expected_year >= 2026:
        raise AssertionError(f"{period} date boundary failed")
    if output["duration"].le(0).any() or not output["state"].between(0, K - 1).all():
        raise AssertionError(f"{period} invalid duration/state")
    groups = [output["symbol_norm"], output["session_date"]]
    grouped_states = output.groupby(groups, sort=False)["state"]
    grouped_durations = output.groupby(groups, sort=False)["duration"]
    previous1 = grouped_states.shift(1).fillna(END).astype(int)
    previous2 = grouped_states.shift(2).fillna(END).astype(int)
    next_state = grouped_states.shift(-1)
    if not np.array_equal(previous1, output["previous_state_1"].to_numpy(int)):
        raise AssertionError(f"{period} previous_state_1 mismatch")
    if not np.array_equal(previous2, output["previous_state_2"].to_numpy(int)):
        raise AssertionError(f"{period} previous_state_2 mismatch")
    has_next = output["has_next_state"].astype(bool)
    if not np.array_equal(next_state.notna(), has_next):
        raise AssertionError(f"{period} terminal marker mismatch")
    stored_next = pd.to_numeric(output["next_state"], errors="coerce")
    if not np.array_equal(
        next_state.loc[has_next].astype(int), stored_next.loc[has_next].astype(int)
    ):
        raise AssertionError(f"{period} next-state mismatch")
    output["next_outcome"] = next_state.fillna(END).astype(int)
    if output["next_outcome"].eq(output["state"]).any():
        raise AssertionError(f"{period} compressed run self-transition")
    for step in range(1, 5):
        output[f"future_state_{step}"] = (
            grouped_states.shift(-step).fillna(END).astype(int)
        )
    for step in range(1, 4):
        output[f"future_duration_{step}"] = (
            grouped_durations.shift(-step).fillna(0).astype(int)
        )
    local = output["start_timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute - 570
    grid = (
        minute.ge(0)
        & minute.lt(390)
        & minute.mod(5).eq(0)
        & local.dt.second.eq(0)
        & local.dt.microsecond.eq(0)
    )
    output["bar_index_in_session"] = -1
    output.loc[grid, "bar_index_in_session"] = (minute.loc[grid] // 5).astype(int)
    output = output.loc[grid & output["bar_index_in_session"].le(MAX_START_BAR)].copy()
    output = output.reset_index(drop=True)
    dates = pd.to_datetime(output["session_date"], errors="raise")
    output["quarter"] = dates.dt.year.astype(str) + "_q" + dates.dt.quarter.astype(str)
    output["period"] = period
    output["anchor_id"] = np.arange(len(output), dtype=np.int64)
    return output


def retained_destination(
    previous2: np.ndarray,
    previous1: np.ndarray,
    current: np.ndarray,
    destination: int,
    parameters: dict[str, np.ndarray],
) -> np.ndarray:
    token = ((previous2 * HISTORY_VALUES + previous1) * K + current).astype(int)
    logits = parameters["history_intercept"][None, :] + parameters[
        "history_coef"
    ][:, token].T
    return softmax(logits)[:, int(destination)]


def independent_convolution(
    distribution: np.ndarray, transition: np.ndarray
) -> np.ndarray:
    """Convolve exact durations 1..23, dropping the >=24 overflow mass."""
    output = np.zeros_like(distribution)
    for bucket in range(OVERFLOW - 1):
        duration = bucket + 1
        output[:, duration:] += (
            distribution[:, : MAX_HORIZON + 1 - duration]
            * transition[:, bucket][:, None]
        )
    return output


def independent_route_forecast(
    anchors: pd.DataFrame,
    route: tuple[int, ...],
    parameters: dict[str, np.ndarray],
    saved_kernel: dict[str, np.ndarray],
    frozen: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    size = len(anchors)
    previous2 = anchors["previous_state_2"].to_numpy(int)
    previous1 = anchors["previous_state_1"].to_numpy(int)
    current = np.full(size, route[0], dtype=int)
    path_probability = np.ones(size, dtype=float)
    timed_models = tuple(model for model in MODEL_COLUMNS if model != "history_path_only")
    distributions = {
        model: np.zeros((size, MAX_HORIZON + 1), dtype=float)
        for model in timed_models
    }
    for values in distributions.values():
        values[:, 0] = 1.0
    for destination in route[1:]:
        destination_mass = retained_destination(
            previous2, previous1, current, int(destination), parameters
        )
        path_probability *= destination_mass
        pmfs = {
            "history_frozen_state_timed": frozen[current],
            "history_destination_timed": saved_kernel["state_dest_pmf"][
                current, int(destination)
            ],
            "history_order2_timed": saved_kernel["order2_pmf"][
                previous1, current, int(destination)
            ],
            "history_joint_timed": saved_kernel["order3_pmf"][
                previous2, previous1, current, int(destination)
            ],
        }
        for model in timed_models:
            transition = destination_mass[:, None] * pmfs[model]
            distributions[model] = independent_convolution(
                distributions[model], transition
            )
        previous2, previous1, current = (
            previous1,
            current,
            np.full(size, int(destination), dtype=int),
        )
    return {
        horizon: {
            "history_path_only": path_probability.copy(),
            **{
                model: values[:, : horizon + 1].sum(axis=1)
                for model, values in distributions.items()
            },
        }
        for horizon in HORIZONS
    }


def independent_route_truth(
    anchors: pd.DataFrame, route: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    matches = np.ones(len(anchors), dtype=bool)
    for step, destination in enumerate(route[1:], start=1):
        matches &= anchors[f"future_state_{step}"].to_numpy(int) == int(destination)
    completion = anchors["duration"].to_numpy(int).copy()
    for step in range(1, len(route) - 1):
        completion += anchors[f"future_duration_{step}"].to_numpy(int)
    return matches, completion


def reconstruct_period(
    period: str,
    anchors: pd.DataFrame,
    cycles: list[dict[str, Any]],
    parameters: dict[str, np.ndarray],
    saved_kernel: dict[str, np.ndarray],
    frozen: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    scoring_path = ARTIFACT / f"joint_completion_scoring_{period}.parquet"
    scoring = pd.read_parquet(scoring_path).sort_values(
        ["anchor_id", "horizon", "cycle_id"], kind="stable"
    ).reset_index(drop=True)
    if scoring.duplicated(["anchor_id", "horizon", "cycle_id"]).any():
        raise AssertionError(f"{period} duplicate scoring keys")
    if set(scoring["period"].astype(str)) != {period}:
        raise AssertionError(f"{period} scoring period mismatch")
    scoring_dates = pd.to_datetime(scoring["session_date"], errors="raise")
    if set(scoring_dates.dt.year.unique()) != {int(period)} or scoring_dates.dt.year.ge(2026).any():
        raise AssertionError(f"{period} scoring includes forbidden year")
    if not set(scoring["horizon"].astype(int).unique()) == set(HORIZONS):
        raise AssertionError(f"{period} horizon drift")

    maximum_probability_error = {model: 0.0 for model in MODEL_COLUMNS}
    label_mismatches = 0
    eventual_mismatches = 0
    completion_mismatches = 0
    metadata_mismatches = 0
    route_rows: list[dict[str, Any]] = []
    expected_rows = 0

    for cycle in cycles:
        core = cycle["core"]
        selected = anchors.loc[anchors["state"].isin(set(core))].reset_index(drop=True)
        expected_rows += len(selected) * len(HORIZONS)
        eventual = np.zeros(len(selected), dtype=bool)
        completion = np.full(len(selected), np.nan, dtype=float)
        targets = {horizon: np.zeros(len(selected), dtype=bool) for horizon in HORIZONS}
        probabilities = {
            horizon: {model: np.zeros(len(selected), dtype=float) for model in MODEL_COLUMNS}
            for horizon in HORIZONS
        }
        for current_state in sorted(set(core)):
            mask = selected["state"].eq(current_state).to_numpy()
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
            truth_count = np.zeros(len(state_anchors), dtype=int)
            for route in compatible_routes(core, current_state):
                truth, route_completion = independent_route_truth(state_anchors, route)
                truth_count += truth.astype(int)
                local_eventual |= truth
                local_completion[truth] = route_completion[truth]
                forecast = independent_route_forecast(
                    state_anchors, route, parameters, saved_kernel, frozen
                )
                for horizon in HORIZONS:
                    local_targets[horizon] |= truth & (route_completion <= horizon)
                    for model in MODEL_COLUMNS:
                        local_probabilities[horizon][model] += forecast[horizon][model]
                route_rows.append(
                    {
                        "period": period,
                        "cycle_id": cycle["cycle_id"],
                        "cycle": cycle["cycle"],
                        "current_state": current_state,
                        "route": "->".join(str(value) for value in route),
                        "transition_count": len(route) - 1,
                    }
                )
            if (truth_count > 1).any():
                raise AssertionError(f"{period} non-exclusive cycle rotations")
            eventual[mask] = local_eventual
            completion[mask] = local_completion
            for horizon in HORIZONS:
                targets[horizon][mask] = local_targets[horizon]
                for model in MODEL_COLUMNS:
                    probabilities[horizon][model][mask] = local_probabilities[horizon][model]

        cycle_scoring = scoring.loc[scoring["cycle_id"].eq(cycle["cycle_id"])]
        for horizon in HORIZONS:
            observed = cycle_scoring.loc[cycle_scoring["horizon"].eq(horizon)].sort_values(
                "anchor_id", kind="stable"
            ).reset_index(drop=True)
            if len(observed) != len(selected):
                raise AssertionError(f"{period} {cycle['cycle_id']} row-count mismatch")
            exact_columns = (
                "anchor_id",
                "symbol_norm",
                "session_date",
                "quarter",
                "bar_index_in_session",
                "state",
                "previous_state_1",
                "previous_state_2",
                "duration",
            )
            for column in exact_columns:
                if not np.array_equal(
                    observed[column].astype(str).to_numpy(),
                    selected[column].astype(str).to_numpy(),
                ):
                    metadata_mismatches += 1
            if not np.array_equal(
                pd.to_datetime(observed["start_timestamp"], utc=True).astype("int64"),
                selected["start_timestamp"].astype("int64"),
            ):
                metadata_mismatches += 1
            if not observed["cycle"].astype(str).eq(cycle["cycle"]).all():
                metadata_mismatches += 1
            if not observed["transition_length"].astype(int).eq(
                cycle["transition_length"]
            ).all():
                metadata_mismatches += 1
            label_mismatches += int(
                np.sum(observed["target"].to_numpy(int) != targets[horizon].astype(int))
            )
            eventual_mismatches += int(
                np.sum(
                    observed["eventual_target"].to_numpy(int)
                    != eventual.astype(int)
                )
            )
            completion_mismatches += int(
                np.sum(
                    ~np.isclose(
                        observed["completion_bars"].to_numpy(float),
                        completion,
                        atol=0.0,
                        rtol=0.0,
                        equal_nan=True,
                    )
                )
            )
            for model, column in MODEL_COLUMNS.items():
                expected = np.clip(probabilities[horizon][model], EPS, 1.0 - EPS)
                error = float(
                    np.max(np.abs(observed[column].to_numpy(float) - expected), initial=0.0)
                )
                maximum_probability_error[model] = max(
                    maximum_probability_error[model], error
                )

    if len(scoring) != expected_rows:
        raise AssertionError(f"{period} total scoring row mismatch")
    if metadata_mismatches or label_mismatches or eventual_mismatches or completion_mismatches:
        raise AssertionError(
            f"{period} reconstruction mismatch: metadata={metadata_mismatches}, "
            f"labels={label_mismatches}, eventual={eventual_mismatches}, "
            f"completion={completion_mismatches}"
        )
    if max(maximum_probability_error.values()) >= 1e-11:
        raise AssertionError(
            f"{period} probability mismatch: {maximum_probability_error}"
        )
    for column in MODEL_COLUMNS.values():
        pivot = scoring.pivot_table(
            index=["anchor_id", "cycle_id"], columns="horizon", values=column
        )
        if column == MODEL_COLUMNS["history_path_only"]:
            if not np.allclose(pivot[6], pivot[12]) or not np.allclose(pivot[12], pivot[24]):
                raise AssertionError(f"{period} path probability changed by horizon")
        elif not ((pivot[6] <= pivot[12] + 1e-12) & (pivot[12] <= pivot[24] + 1e-12)).all():
            raise AssertionError(f"{period} non-monotonic {column}")
    if not (
        scoring[MODEL_COLUMNS["history_joint_timed"]]
        <= scoring[MODEL_COLUMNS["history_path_only"]] + 1e-12
    ).all():
        raise AssertionError(f"{period} completion mass exceeds path mass")
    route_manifest = pd.DataFrame(route_rows).drop_duplicates().sort_values(
        ["period", "cycle_id", "current_state", "route"], kind="stable"
    ).reset_index(drop=True)
    observed_routes = pd.read_csv(ARTIFACT / f"route_manifest_{period}.csv")
    observed_routes["period"] = observed_routes["period"].astype(str)
    pd.testing.assert_frame_equal(
        route_manifest,
        observed_routes.sort_values(
            ["period", "cycle_id", "current_state", "route"], kind="stable"
        ).reset_index(drop=True),
        check_dtype=False,
    )
    return scoring, route_manifest, {
        "anchors": len(anchors),
        "scoring_rows": len(scoring),
        "label_mismatches": label_mismatches,
        "eventual_mismatches": eventual_mismatches,
        "completion_mismatches": completion_mismatches,
        "metadata_mismatches": metadata_mismatches,
        "maximum_probability_error": maximum_probability_error,
    }


def independent_losses(target: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    y = np.asarray(target, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return {
        "log_loss": -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)),
        "brier": np.square(p - y),
    }


def independent_calibration(
    scoring: pd.DataFrame, period: str, model: str, column: str
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        mask = scoring["horizon"].eq(horizon).to_numpy()
        target = scoring.loc[mask, "target"].to_numpy(float)
        probability = scoring.loc[mask, column].to_numpy(float)
        bins = np.minimum((probability * 10.0).astype(int), 9)
        ece = 0.0
        supported_errors: list[float] = []
        for bin_id in range(10):
            selected = bins == bin_id
            count = int(selected.sum())
            mean_probability = float(probability[selected].mean()) if count else math.nan
            event_rate = float(target[selected].mean()) if count else math.nan
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
                    "bin": bin_id,
                    "count": count,
                    "mean_probability": mean_probability,
                    "event_rate": event_rate,
                    "absolute_error": error,
                    "supported": supported,
                }
            )
        if not supported_errors:
            raise AssertionError(f"{period} {model} horizon {horizon} has no supported bin")
        summary[horizon] = {
            "ece": float(ece),
            "maximum_supported_bin_error": float(max(supported_errors)),
        }
    return rows, summary


def independent_top_three(
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
    selected_count = int(selected.sum())
    return {
        "period": period,
        "model": model,
        "anchor_horizons": int(
            ranked[["anchor_id", "horizon"]].drop_duplicates().shape[0]
        ),
        "positive_labels": positives,
        "selected_labels": selected_count,
        "hits": hits,
        "recall": float(hits / positives) if positives else math.nan,
        "precision": float(hits / selected_count) if selected_count else math.nan,
    }


def independent_block_interval(
    values: np.ndarray, seed: int
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 10:
        return math.nan, math.nan, math.nan
    width = min(5, len(clean))
    blocks = np.asarray(
        [clean[start : start + width] for start in range(len(clean) - width + 1)]
    )
    needed = int(math.ceil(len(clean) / width))
    rng = np.random.default_rng(seed)
    draws = np.empty(5000, dtype=float)
    for index in range(len(draws)):
        choices = rng.integers(0, len(blocks), size=needed)
        draws[index] = blocks[choices].reshape(-1)[: len(clean)].mean()
    return (
        float(clean.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def independently_evaluate(
    scoring: pd.DataFrame, period: str, seed_offset: int
) -> dict[str, Any]:
    target = scoring["target"].to_numpy(int)
    losses: dict[str, dict[str, np.ndarray]] = {}
    metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    calibration_summary: dict[str, dict[int, dict[str, float]]] = {}
    top_rows: list[dict[str, Any]] = []
    top_summary: dict[str, dict[str, Any]] = {}
    for model, column in MODEL_COLUMNS.items():
        probability = scoring[column].to_numpy(float)
        losses[model] = independent_losses(target, probability)
        bins, summary = independent_calibration(scoring, period, model, column)
        calibration_rows.extend(bins)
        calibration_summary[model] = summary
        for horizon in HORIZONS:
            mask = scoring["horizon"].eq(horizon).to_numpy()
            metric_rows.append(
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
        top = independent_top_three(scoring, period, model, column)
        top_rows.append(top)
        top_summary[model] = top

    per_cycle_rows: list[dict[str, Any]] = []
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

    comparison_rows: list[dict[str, Any]] = []
    gates: dict[str, dict[str, Any]] = {}
    for comparison_index, (
        candidate,
        baseline,
        comparison,
        relative_requirement,
        recall_requirement,
    ) in enumerate(COMPARISON_SPECS):
        paired_rows = []
        for loss_index, loss_name in enumerate(("log_loss", "brier")):
            difference = losses[candidate][loss_name] - losses[baseline][loss_name]
            daily = (
                pd.DataFrame(
                    {
                        "session_date": scoring["session_date"],
                        "difference": difference,
                    }
                )
                .groupby("session_date", sort=True)["difference"]
                .mean()
                .to_numpy(float)
            )
            mean, low, high = independent_block_interval(
                daily, SEED + seed_offset + comparison_index * 100 + loss_index
            )
            horizon_means = pd.Series(difference).groupby(
                scoring["horizon"].reset_index(drop=True)
            ).mean()
            quarter_means = pd.Series(difference).groupby(
                scoring["quarter"].reset_index(drop=True)
            ).mean()
            deletion_means = [
                float(
                    difference[
                        scoring["symbol_norm"].astype(str).ne(symbol).to_numpy()
                    ].mean()
                )
                for symbol in sorted(scoring["symbol_norm"].astype(str).unique())
            ]
            baseline_mean = float(losses[baseline][loss_name].mean())
            row = {
                "period": period,
                "comparison": comparison,
                "candidate": candidate,
                "baseline": baseline,
                "loss": loss_name,
                "row_mean_difference": float(difference.mean()),
                "daily_mean_difference": mean,
                "daily_ci_low": low,
                "daily_ci_high": high,
                "baseline_mean_loss": baseline_mean,
                "relative_improvement": float(-difference.mean() / baseline_mean),
                "negative_horizon_count": int((horizon_means < 0.0).sum()),
                "negative_quarter_count": int((quarter_means < 0.0).sum()),
                "leave_one_symbol_max_difference": max(deletion_means),
                "leave_one_symbol_all_negative": bool(max(deletion_means) < 0.0),
            }
            comparison_rows.append(row)
            paired_rows.append(row)
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
            "intervals_pass": all(row["daily_ci_high"] < 0.0 for row in paired_rows),
            "relative_log_loss_improvement": paired_rows[0]["relative_improvement"],
            "relative_log_loss_requirement": relative_requirement,
            "relative_log_loss_pass": paired_rows[0]["relative_improvement"]
            >= relative_requirement,
            "horizons_pass": all(
                row["negative_horizon_count"] == len(HORIZONS) for row in paired_rows
            ),
            "quarters_pass": all(
                row["negative_quarter_count"] == 4 for row in paired_rows
            ),
            "stock_deletions_pass": all(
                row["leave_one_symbol_all_negative"] for row in paired_rows
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
        gates[comparison] = gate

    path_checks = {}
    for loss_name in ("log_loss", "brier"):
        difference = (
            losses["history_joint_timed"][loss_name]
            - losses["history_path_only"][loss_name]
        )
        horizon_means = pd.Series(difference).groupby(
            scoring["horizon"].reset_index(drop=True)
        ).mean()
        path_checks[loss_name] = bool((horizon_means < 0.0).all())
    path_recall = float(
        top_summary["history_joint_timed"]["recall"]
        - top_summary["history_path_only"]["recall"]
    )
    gates["joint_vs_path_only_sanity"] = {
        "log_loss_lower_every_horizon": path_checks["log_loss"],
        "brier_lower_every_horizon": path_checks["brier"],
        "top_three_recall_difference": path_recall,
        "top_three_recall_pass": path_recall >= 0.005,
        "pass": bool(
            path_checks["log_loss"]
            and path_checks["brier"]
            and path_recall >= 0.005
        ),
    }

    support_rows: list[dict[str, Any]] = []
    support_pass = True
    for horizon in HORIZONS:
        selected = scoring.loc[scoring["horizon"].eq(horizon)]
        cycle_positives = selected.groupby("cycle_id", sort=True)["target"].sum()
        row = {
            "period": period,
            "horizon": horizon,
            "rows": len(selected),
            "positives": int(selected["target"].sum()),
            "cycles": int(selected["cycle_id"].nunique()),
            "minimum_cycle_positives": int(cycle_positives.min()),
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
        "metrics": pd.DataFrame(metric_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "top_three": pd.DataFrame(top_rows),
        "per_cycle": per_cycle,
        "comparisons": pd.DataFrame(comparison_rows),
        "support": pd.DataFrame(support_rows),
        "support_pass": bool(support_pass),
        "gates": gates,
    }


def compare_saved_frame(
    expected: pd.DataFrame, path: Path, sort_columns: list[str]
) -> dict[str, Any]:
    observed = pd.read_csv(path)
    if set(expected.columns) != set(observed.columns):
        raise AssertionError(
            f"{path.name} column mismatch: expected={list(expected.columns)}, "
            f"observed={list(observed.columns)}"
        )
    observed = observed[list(expected.columns)]
    if "period" in expected:
        expected = expected.copy()
        observed = observed.copy()
        expected["period"] = expected["period"].astype(str)
        observed["period"] = observed["period"].astype(str)
    expected = expected.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    observed = observed.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected,
        observed,
        check_dtype=False,
        check_exact=False,
        atol=1e-11,
        rtol=0.0,
    )
    maximum_error = 0.0
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(
            expected[column]
        ):
            left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(observed[column], errors="coerce").to_numpy(float)
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                maximum_error = max(
                    maximum_error, float(np.max(np.abs(left[finite] - right[finite])))
                )
    return {"rows": len(expected), "maximum_numeric_error": maximum_error}


def compare_nested(expected: Any, observed: Any, path: str = "root") -> float:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            raise AssertionError(f"nested keys differ at {path}")
        return max(
            (compare_nested(expected[key], observed[key], f"{path}.{key}") for key in expected),
            default=0.0,
        )
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise AssertionError(f"nested list differs at {path}")
        return max(
            (
                compare_nested(left, right, f"{path}[{index}]")
                for index, (left, right) in enumerate(zip(expected, observed))
            ),
            default=0.0,
        )
    if isinstance(expected, (bool, np.bool_)) or isinstance(observed, (bool, np.bool_)):
        if bool(expected) is not bool(observed):
            raise AssertionError(f"boolean differs at {path}: {expected} != {observed}")
        return 0.0
    if isinstance(expected, (int, float, np.number)) and isinstance(
        observed, (int, float, np.number)
    ):
        if pd.isna(expected) and pd.isna(observed):
            return 0.0
        error = abs(float(expected) - float(observed))
        if error > 1e-11:
            raise AssertionError(f"numeric value differs at {path}: {expected} != {observed}")
        return error
    if expected != observed:
        raise AssertionError(f"value differs at {path}: {expected!r} != {observed!r}")
    return 0.0


def verify_shadow_bundle() -> dict[str, Any]:
    shadow = WORKSPACE / "work/shadow_validation/frozen_loop_movement_shadow_v1"
    manifest_path = shadow / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bundle = shadow / "frozen_bundle"
    errors = []
    for item in manifest["files"]:
        bundled = bundle / item["bundle_path"]
        if not bundled.is_file() or digest(bundled) != item["sha256"]:
            errors.append(item["name"])
    external_manifest = (
        WORKSPACE
        / "work/contracts/20260710-frozen-loop-movement-shadow-v1-manifest.json"
    )
    external_contract = (
        WORKSPACE / "work/contracts/20260710-frozen-loop-movement-shadow-v1.json"
    )
    runtime = json.loads((shadow / "runtime_metadata.json").read_text())
    if digest(external_manifest) != digest(manifest_path):
        errors.append("external_manifest_copy")
    if digest(external_contract) != manifest["contract"]["sha256"]:
        errors.append("external_contract")
    if digest(shadow / "contract.json") != manifest["contract"]["sha256"]:
        errors.append("runtime_contract")
    if runtime.get("freeze_manifest_sha256") != digest(manifest_path):
        errors.append("runtime_manifest_hash")
    if runtime.get("contract_sha256") != manifest["contract"]["sha256"]:
        errors.append("runtime_contract_hash")
    if runtime.get("outcomes_opened") is not False:
        errors.append("outcomes_opened")
    if errors:
        raise AssertionError(f"prospective shadow bundle mismatch: {errors}")
    return {
        "bundle_files": len(manifest["files"]),
        "manifest_sha256": digest(manifest_path),
        "contract_sha256": manifest["contract"]["sha256"],
        "outcomes_opened": False,
    }


def post_score_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, details: Any = None) -> None:
        checks.append({"name": name, "pass": bool(condition), "details": details})
        if not condition:
            raise AssertionError(f"post-score audit failed: {name}: {details}")

    contract = json.loads(CONTRACT.read_text())
    check("contract_research_only", contract.get("research_only") is True)
    check("contract_live_ordering_disabled", contract.get("live_ordering_enabled") is False)
    check("contract_order_placement_disabled", contract.get("order_placement") == "disabled")
    check("contract_no_2026", contract["periods"].get("2026_permitted") is False)
    check(
        "contract_periods_exact",
        contract["periods"]["fit"] == 2024
        and contract["periods"]["development_score"] == 2025
        and contract["periods"]["backward_portability_score"] == 2023,
    )
    check(
        "contract_kernel_exact",
        tuple(contract["joint_kernel"]["smoothing"]["selection"]["selected_tuple"])
        == SELECTED
        and contract["joint_kernel"]["duration_fit_excludes_session_end_destination"]
        is True,
    )

    pre_audit = json.loads((ARTIFACT / "pre_score_audit.json").read_text())
    check("pre_score_audit_passed", pre_audit.get("all_passed") is True)
    check("pre_score_outcomes_closed", pre_audit.get("scoring_outcomes_opened") is False)
    fit_complete = json.loads((ARTIFACT / "fit_complete.json").read_text())
    fit_hashes = {
        "kernel_parameters_sha256": digest(ARTIFACT / "kernel_parameters.npz"),
        "kernel_manifest_sha256": digest(ARTIFACT / "kernel_manifest.json"),
        "smoothing_selection_sha256": digest(ARTIFACT / "smoothing_selection_2024.csv"),
        "fit_source_hashes_sha256": digest(ARTIFACT / "fit_source_hashes.json"),
        "pre_snapshot_sha256": digest(ARTIFACT / "prospective_shadow_pre_snapshot.json"),
    }
    check(
        "fit_artifact_hashes_frozen",
        all(fit_complete.get(name) == value for name, value in fit_hashes.items()),
        fit_hashes,
    )
    check(
        "fit_safety_metadata",
        fit_complete.get("research_only") is True
        and fit_complete.get("live_ordering_enabled") is False
        and fit_complete.get("order_placement") == "disabled"
        and fit_complete.get("scoring_outcomes_opened") is False,
    )

    current_fit_sources = {
        "contract.json": digest(CONTRACT),
        "runner.py": digest(RUNNER),
        "train_2024_filtered_runs.csv": digest(TRAIN),
        "path_model_parameters.npz": digest(PATH_PARAMETERS),
        "path_gates.json": digest(PATH_GATES),
        "path_independent_artifact_audit.json": digest(PATH_AUDIT),
        "frozen_semimarkov_parameters.npz": digest(SEMIMARKOV),
        "fixed_cycle_shuffled_nulls.csv": digest(CYCLES),
    }
    recorded_fit_sources = json.loads((ARTIFACT / "fit_source_hashes.json").read_text())
    check("fit_source_hashes_exact", current_fit_sources == recorded_fit_sources)
    current_evaluation_sources = {
        "test_2025_filtered_runs.csv": digest(RUN_2025),
        "backward_2023_filtered_runs.parquet": digest(RUN_2023),
    }
    recorded_evaluation_sources = json.loads(
        (ARTIFACT / "evaluation_source_hashes.json").read_text()
    )
    check(
        "evaluation_source_hashes_exact",
        current_evaluation_sources == recorded_evaluation_sources,
        current_evaluation_sources,
    )

    pre_snapshot = json.loads(
        (ARTIFACT / "prospective_shadow_pre_snapshot.json").read_text()
    )
    post_snapshot = json.loads(
        (ARTIFACT / "prospective_shadow_post_snapshot.json").read_text()
    )
    current_snapshot = snapshot()
    check("protected_shadow_pre_post_exact", pre_snapshot == post_snapshot)
    check("protected_shadow_current_exact", pre_snapshot == current_snapshot)
    shadow_bundle = verify_shadow_bundle()
    check("protected_shadow_bundle_hashes", shadow_bundle["bundle_files"] == 18, shadow_bundle)

    runner_text = RUNNER.read_text().lower()
    check(
        "runner_does_not_import_shadow",
        "import frozen_loop_movement_shadow" not in runner_text
        and "from frozen_loop_movement_shadow" not in runner_text,
    )
    execution_fragments = (
        "place_order(",
        "submit_order(",
        "broker_client",
        "position_size(",
        "paper_trade(",
    )
    check(
        "runner_has_no_execution_surface",
        not any(fragment in runner_text for fragment in execution_fragments),
    )

    train = load_training()
    frozen = base_pmf()
    reconstructed_counts = counts(train)
    reconstructed_kernel = kernel(reconstructed_counts, frozen, SELECTED)
    saved_kernel = dict(np.load(ARTIFACT / "kernel_parameters.npz"))
    count_errors = {
        "state_destination": int(
            np.max(
                np.abs(saved_kernel["state_dest_counts"] - reconstructed_counts["q1"])
            )
        ),
        "order2": int(
            np.max(np.abs(saved_kernel["order2_counts"] - reconstructed_counts["q2"]))
        ),
        "order3": int(
            np.max(np.abs(saved_kernel["order3_counts"] - reconstructed_counts["q3"]))
        ),
    }
    probability_errors = {
        "state": float(np.max(np.abs(saved_kernel["state_pmf"] - frozen))),
        "state_destination": float(
            np.max(np.abs(saved_kernel["state_dest_pmf"] - reconstructed_kernel["q1"]))
        ),
        "order2": float(
            np.max(np.abs(saved_kernel["order2_pmf"] - reconstructed_kernel["q2"]))
        ),
        "order3": float(
            np.max(np.abs(saved_kernel["order3_pmf"] - reconstructed_kernel["q3"]))
        ),
    }
    check("training_2024_only", set(pd.to_datetime(train["session_date"]).dt.year) == {2024})
    check("kernel_counts_reconstructed", max(count_errors.values()) == 0, count_errors)
    check(
        "kernel_probabilities_reconstructed",
        max(probability_errors.values()) < 1e-12,
        probability_errors,
    )
    check(
        "terminal_rows_excluded_exact",
        int(saved_kernel["terminal_rows_excluded"][0])
        == int(train["next_outcome"].eq(END).sum()),
    )
    check(
        "smoothing_strengths_exact",
        float(saved_kernel["tau_state_dest"][0]) == SELECTED[0]
        and float(saved_kernel["tau_order2"][0]) == SELECTED[1]
        and float(saved_kernel["tau_order3"][0]) == SELECTED[2],
    )
    pmf_errors = {
        name: float(np.max(np.abs(saved_kernel[name].sum(axis=-1) - 1.0)))
        for name in ("state_pmf", "state_dest_pmf", "order2_pmf", "order3_pmf")
    }
    check("all_dwell_pmfs_normalize", max(pmf_errors.values()) < 1e-12, pmf_errors)
    joint_errors = check_joint_normalization(saved_kernel)
    check(
        "joint_kernel_normalizes",
        joint_errors["normalization_error"] < 1e-12
        and joint_errors["destination_marginal_error"] < 1e-12,
        joint_errors,
    )
    check("independent_dp_self_test", dp_self_test() < 1e-12)

    path_parameters = dict(np.load(PATH_PARAMETERS))
    check(
        "retained_destination_classes_exact",
        np.array_equal(path_parameters["history_classes"], np.arange(DESTINATIONS)),
    )
    cycles = independent_cycles()
    check("twenty_frozen_cycles_exact", len(cycles) == 20)

    period_results: dict[str, Any] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    route_manifests: list[pd.DataFrame] = []
    forbidden_tokens = {
        "price",
        "return",
        "direction",
        "range",
        "pnl",
        "cost",
        "spread",
        "order",
        "broker",
        "position",
        "volume",
        "deployment",
    }
    for period, path, year, seed_offset in (
        ("2025", RUN_2025, 2025, 1000),
        ("2023", RUN_2023, 2023, 2000),
    ):
        anchors = load_scoring_runs(path, year, period)
        scoring, routes, reconstruction = reconstruct_period(
            period, anchors, cycles, path_parameters, saved_kernel, frozen
        )
        forbidden_columns = [
            column
            for column in scoring.columns
            if forbidden_tokens.intersection(column.lower().split("_"))
        ]
        check(f"{period}_no_forbidden_columns", not forbidden_columns, forbidden_columns)
        check(
            f"{period}_cohort_causal_grid",
            scoring["bar_index_in_session"].between(0, MAX_START_BAR).all()
            and pd.to_datetime(scoring["session_date"]).dt.year.eq(year).all(),
        )
        check(
            f"{period}_labels_completion_exact",
            reconstruction["label_mismatches"] == 0
            and reconstruction["eventual_mismatches"] == 0
            and reconstruction["completion_mismatches"] == 0
            and reconstruction["metadata_mismatches"] == 0,
            reconstruction,
        )
        check(
            f"{period}_five_probabilities_exact",
            max(reconstruction["maximum_probability_error"].values()) < 1e-11,
            reconstruction["maximum_probability_error"],
        )
        evaluation = independently_evaluate(scoring, period, seed_offset)
        period_results[period] = reconstruction
        evaluations[period] = evaluation
        route_manifests.append(routes)

    combined_routes = pd.concat(route_manifests, ignore_index=True).drop_duplicates()
    route_frame_check = compare_saved_frame(
        combined_routes,
        ARTIFACT / "route_manifest.csv",
        ["period", "cycle_id", "current_state", "route"],
    )
    check("combined_route_manifest_exact", route_frame_check["maximum_numeric_error"] < 1e-11, route_frame_check)

    frame_specs = (
        (
            "metrics",
            "joint_completion_metrics.csv",
            ["period", "model", "horizon"],
        ),
        (
            "calibration",
            "joint_completion_calibration.csv",
            ["period", "model", "horizon", "bin"],
        ),
        (
            "top_three",
            "joint_completion_top_three.csv",
            ["period", "model"],
        ),
        (
            "per_cycle",
            "joint_completion_per_cycle.csv",
            ["period", "cycle_id"],
        ),
        (
            "comparisons",
            "joint_completion_comparisons.csv",
            ["period", "comparison", "loss"],
        ),
        (
            "support",
            "joint_completion_support.csv",
            ["period", "horizon"],
        ),
    )
    aggregate_checks: dict[str, Any] = {}
    for key, filename, sort_columns in frame_specs:
        expected = pd.concat(
            [evaluations[period][key] for period in ("2025", "2023")],
            ignore_index=True,
        )
        details = compare_saved_frame(expected, ARTIFACT / filename, sort_columns)
        aggregate_checks[key] = details
        check(f"aggregate_{key}_exact", details["maximum_numeric_error"] < 1e-11, details)

    reconstructed_gates: dict[str, Any] = {"periods": {}}
    for period in ("2025", "2023"):
        reconstructed_gates["periods"][period] = {
            "support_pass": evaluations[period]["support_pass"],
            **evaluations[period]["gates"],
        }
    reconstructed_gates["joint_completion_retained"] = bool(
        all(
            reconstructed_gates["periods"][period]["support_pass"]
            for period in ("2025", "2023")
        )
        and all(
            reconstructed_gates["periods"][period][comparison]["pass"]
            for period in ("2025", "2023")
            for comparison in (
                "joint_vs_frozen_state_timed",
                "joint_vs_destination_timed",
            )
        )
        and all(
            reconstructed_gates["periods"][period]["joint_vs_path_only_sanity"][
                "pass"
            ]
            for period in ("2025", "2023")
        )
    )
    reconstructed_gates["destination_conditioned_timing_retained"] = bool(
        all(
            reconstructed_gates["periods"][period]["support_pass"]
            for period in ("2025", "2023")
        )
        and all(
            reconstructed_gates["periods"][period][
                "destination_vs_frozen_state_timed"
            ]["pass"]
            for period in ("2025", "2023")
        )
    )
    reconstructed_gates.update(
        {
            "research_only": True,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "economic_edge_claim": False,
        }
    )
    saved_gates = json.loads((ARTIFACT / "gates.json").read_text())
    gate_error = compare_nested(reconstructed_gates, saved_gates)
    check("all_gate_values_and_decisions_exact", gate_error < 1e-11, gate_error)
    summary = json.loads((ARTIFACT / "summary.json").read_text())
    summary_gate_error = compare_nested(reconstructed_gates, summary["gates"])
    check("summary_gate_copy_exact", summary_gate_error < 1e-11, summary_gate_error)
    check(
        "saved_safety_decisions_exact",
        saved_gates.get("research_only") is True
        and saved_gates.get("live_ordering_enabled") is False
        and saved_gates.get("order_placement") == "disabled"
        and saved_gates.get("economic_edge_claim") is False,
    )
    check(
        "scientific_decisions_reconstructed",
        saved_gates["joint_completion_retained"]
        == reconstructed_gates["joint_completion_retained"]
        and saved_gates["destination_conditioned_timing_retained"]
        == reconstructed_gates["destination_conditioned_timing_retained"],
        {
            "joint_completion_retained": reconstructed_gates[
                "joint_completion_retained"
            ],
            "destination_conditioned_timing_retained": reconstructed_gates[
                "destination_conditioned_timing_retained"
            ],
        },
    )

    payload = {
        "all_passed": all(item["pass"] for item in checks),
        "check_count": len(checks),
        "checks": checks,
        "fit_source_hashes": current_fit_sources,
        "evaluation_source_hashes": current_evaluation_sources,
        "kernel_reconstruction": {
            "count_errors": count_errors,
            "probability_errors": probability_errors,
            "pmf_normalization_errors": pmf_errors,
            "joint_errors": joint_errors,
        },
        "period_reconstruction": period_results,
        "aggregate_reconstruction": aggregate_checks,
        "reconstructed_decisions": {
            "joint_completion_retained": reconstructed_gates[
                "joint_completion_retained"
            ],
            "destination_conditioned_timing_retained": reconstructed_gates[
                "destination_conditioned_timing_retained"
            ],
        },
        "prospective_shadow": shadow_bundle,
        "no_2026_rows": True,
        "research_only": True,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    (ARTIFACT / "independent_artifact_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-score-only", action="store_true")
    args = parser.parse_args()
    result = pre_score_audit() if args.pre_score_only else post_score_audit()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
