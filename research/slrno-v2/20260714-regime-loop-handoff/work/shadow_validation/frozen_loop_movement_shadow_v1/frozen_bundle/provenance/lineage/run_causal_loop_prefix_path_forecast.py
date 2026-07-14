"""Research-only causal fixed-loop path forecaster.

The transition models are fitted on frozen 2024 state runs and scored on 2025
and backward-portability 2023 runs.  Session end is an explicit destination.
No price direction, future return, P&L, order, or runtime path is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression


HERE = Path(__file__).resolve().parent
SOURCE_2024_2025 = Path(
    "/private/tmp/stocker_causal_semimarkov_regime_loops_20260710"
)
SOURCE_2023 = Path(
    "/private/tmp/stocker_sealed_backward_2023_complete_detector_20260710"
)
TRAIN_PATH = SOURCE_2024_2025 / "train_2024_filtered_runs.csv"
TEST_2025_PATH = SOURCE_2024_2025 / "test_2025_filtered_runs.csv"
TEST_2023_PATH = SOURCE_2023 / "backward_2023_filtered_runs.parquet"
CYCLE_PATH = SOURCE_2024_2025 / "fixed_cycle_shuffled_nulls.csv"
OUT = Path("/private/tmp/stocker_causal_loop_prefix_path_forecast_20260710")

SEED = 20260710
K = 8
END_STATE = K
TOKEN_COUNT = (K + 1) * (K + 1) * K
MODEL_NAMES = ("first_order", "history", "context")
PROBABILITY_COLUMNS = {
    "first_order": "probability_first_order",
    "history": "probability_history",
    "context": "probability_context",
}
EPSILON = 1e-12
MIN_BIN_SUPPORT = 500


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise AssertionError("fixed cycle file must contain exactly twenty rows")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for index, value in enumerate(source["cycle"].astype(str), start=1):
        closed = tuple(int(part) for part in value.split("->"))
        if len(closed) < 3 or closed[0] != closed[-1]:
            raise AssertionError(f"invalid closed cycle: {value}")
        core = canonical_cycle(closed[:-1])
        if core in seen:
            raise AssertionError(f"duplicate canonical cycle: {value}")
        if len(core) not in (2, 3, 4):
            raise AssertionError(f"unsupported transition length: {value}")
        if min(core) < 0 or max(core) >= K:
            raise AssertionError(f"cycle state outside frozen range: {value}")
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


def load_runs(path: Path, expected_year: int, period: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {
        "symbol_norm",
        "session_date",
        "state",
        "start_pos",
        "start_timestamp",
        "previous_state_1",
        "previous_state_2",
        "b0_state_numeric",
        "b0_high_stress",
        "next_state",
        "has_next_state",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"missing run columns: {missing}")
    output = frame.copy()
    output["symbol_norm"] = output["symbol_norm"].astype(str)
    output["session_date"] = output["session_date"].astype(str)
    output["state"] = pd.to_numeric(output["state"], errors="raise").astype(int)
    output["start_pos"] = pd.to_numeric(
        output["start_pos"], errors="raise"
    ).astype(int)
    output = output.sort_values(
        ["symbol_norm", "session_date", "start_pos"], kind="stable"
    ).reset_index(drop=True)
    dates = pd.to_datetime(output["session_date"], errors="raise")
    if set(dates.dt.year.unique()) != {expected_year}:
        raise AssertionError(f"{period} contains a year outside {expected_year}")
    if dates.max() >= pd.Timestamp("2026-01-01"):
        raise AssertionError("2026 row entered loop-prefix harness")
    if output["state"].min() < 0 or output["state"].max() >= K:
        raise AssertionError("state outside frozen eight-state range")

    groups = output.groupby(["symbol_norm", "session_date"], sort=False)["state"]
    expected_prev1 = groups.shift(1).fillna(END_STATE).astype(int)
    expected_prev2 = groups.shift(2).fillna(END_STATE).astype(int)
    if not np.array_equal(
        expected_prev1.to_numpy(), output["previous_state_1"].astype(int).to_numpy()
    ):
        raise AssertionError(f"{period} previous_state_1 mismatch")
    if not np.array_equal(
        expected_prev2.to_numpy(), output["previous_state_2"].astype(int).to_numpy()
    ):
        raise AssertionError(f"{period} previous_state_2 mismatch")

    expected_next = groups.shift(-1)
    stored_has_next = output["has_next_state"].astype(bool)
    if not np.array_equal(expected_next.notna().to_numpy(), stored_has_next.to_numpy()):
        raise AssertionError(f"{period} has_next_state mismatch")
    observed_next = pd.to_numeric(output["next_state"], errors="coerce")
    if not np.array_equal(
        expected_next.loc[stored_has_next].astype(int).to_numpy(),
        observed_next.loc[stored_has_next].astype(int).to_numpy(),
    ):
        raise AssertionError(f"{period} next_state mismatch")
    output["next_outcome"] = expected_next.fillna(END_STATE).astype(int)

    timestamp = pd.to_datetime(output["start_timestamp"], utc=True, errors="raise")
    local = timestamp.dt.tz_convert("America/New_York")
    entry_minutes = (
        local.dt.hour.to_numpy(float) * 60.0
        + local.dt.minute.to_numpy(float)
        + local.dt.second.to_numpy(float) / 60.0
        - 570.0
    )
    if np.min(entry_minutes) < 0.0 or np.max(entry_minutes) >= 390.0:
        raise AssertionError(f"{period} has a run entry outside regular session")
    phase = 2.0 * np.pi * entry_minutes / 390.0
    output["entry_time_sin"] = np.sin(phase)
    output["entry_time_cos"] = np.cos(phase)
    output["b0_entry_numeric"] = pd.to_numeric(
        output["b0_state_numeric"], errors="coerce"
    ).fillna(0.0)
    output["b0_entry_high_stress"] = pd.to_numeric(
        output["b0_high_stress"], errors="coerce"
    ).fillna(0.0)
    output["quarter"] = (
        dates.dt.year.astype(str) + "_q" + dates.dt.quarter.astype(str)
    )
    output["period"] = period
    output["anchor_id"] = np.arange(len(output), dtype=np.int64)
    for step in range(1, 5):
        output[f"future_state_{step}"] = (
            groups.shift(-step).fillna(END_STATE).astype(int)
        )
    return output


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
        raise AssertionError("invalid history-token state")
    return ((previous_state_2 * (K + 1) + previous_state_1) * K + current_state)


def token_matrix(tokens: np.ndarray) -> sparse.csr_matrix:
    tokens = np.asarray(tokens, dtype=int)
    return sparse.csr_matrix(
        (
            np.ones(len(tokens), dtype=np.float32),
            (np.arange(len(tokens)), tokens),
        ),
        shape=(len(tokens), TOKEN_COUNT),
    )


def context_numeric(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[
        :,
        [
            "b0_entry_numeric",
            "b0_entry_high_stress",
            "entry_time_sin",
            "entry_time_cos",
        ],
    ].to_numpy(dtype=np.float32)


def fit_models(train: pd.DataFrame) -> dict[str, Any]:
    current = train["state"].to_numpy(dtype=int)
    target = train["next_outcome"].to_numpy(dtype=int)
    counts = np.ones((K, K + 1), dtype=float)
    np.add.at(counts, (current, target), 1.0)
    first_order = counts / counts.sum(axis=1, keepdims=True)

    tokens = history_tokens(
        train["previous_state_2"].to_numpy(dtype=int),
        train["previous_state_1"].to_numpy(dtype=int),
        current,
    )
    history_x = token_matrix(tokens)
    history_model = LogisticRegression(
        C=0.20,
        solver="lbfgs",
        max_iter=500,
        random_state=SEED,
    )
    history_model.fit(history_x, target)
    context_x = sparse.hstack(
        (history_x, sparse.csr_matrix(context_numeric(train))), format="csr"
    )
    context_model = LogisticRegression(
        C=0.20,
        solver="lbfgs",
        max_iter=500,
        random_state=SEED,
    )
    context_model.fit(context_x, target)
    required_classes = np.arange(K + 1)
    if not np.array_equal(history_model.classes_, required_classes):
        raise AssertionError("history model lacks a destination class")
    if not np.array_equal(context_model.classes_, required_classes):
        raise AssertionError("context model lacks a destination class")
    return {
        "first_order": first_order,
        "history": history_model,
        "context": context_model,
    }


def desired_probabilities(
    model: LogisticRegression,
    tokens: np.ndarray,
    destination: int,
    numeric: np.ndarray | None,
) -> np.ndarray:
    matrix: sparse.csr_matrix = token_matrix(tokens)
    if numeric is not None:
        matrix = sparse.hstack(
            (matrix, sparse.csr_matrix(numeric.astype(np.float32))), format="csr"
        )
    raw = model.predict_proba(matrix)
    column = int(np.flatnonzero(model.classes_ == int(destination))[0])
    return np.clip(raw[:, column], EPSILON, 1.0 - EPSILON)


def oriented_path_probabilities(
    anchors: pd.DataFrame,
    path: tuple[int, ...],
    models: dict[str, Any],
) -> dict[str, np.ndarray]:
    count = len(anchors)
    probabilities = {
        "first_order": np.ones(count, dtype=float),
        "history": np.ones(count, dtype=float),
        "context": np.ones(count, dtype=float),
    }
    previous_state_2 = anchors["previous_state_2"].to_numpy(dtype=int)
    previous_state_1 = anchors["previous_state_1"].to_numpy(dtype=int)
    current_state = np.full(count, path[0], dtype=int)
    numeric = context_numeric(anchors)
    for destination in path[1:]:
        tokens = history_tokens(previous_state_2, previous_state_1, current_state)
        probabilities["first_order"] *= models["first_order"][
            current_state, int(destination)
        ]
        probabilities["history"] *= desired_probabilities(
            models["history"], tokens, int(destination), None
        )
        probabilities["context"] *= desired_probabilities(
            models["context"], tokens, int(destination), numeric
        )
        previous_state_2, previous_state_1, current_state = (
            previous_state_1,
            current_state,
            np.full(count, int(destination), dtype=int),
        )
    return probabilities


def oriented_path_label(anchors: pd.DataFrame, path: tuple[int, ...]) -> np.ndarray:
    label = np.ones(len(anchors), dtype=bool)
    for step, destination in enumerate(path[1:], start=1):
        label &= anchors[f"future_state_{step}"].to_numpy(dtype=int) == int(
            destination
        )
    return label


def score_period(
    runs: pd.DataFrame, cycles: pd.DataFrame, models: dict[str, Any]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cycle in cycles.itertuples(index=False):
        core = tuple(int(state) for state in cycle.core)
        compatible = runs["state"].isin(set(core)).to_numpy()
        anchors = runs.loc[compatible].copy().reset_index(drop=True)
        target = np.zeros(len(anchors), dtype=bool)
        probabilities = {
            name: np.zeros(len(anchors), dtype=float) for name in MODEL_NAMES
        }
        for current in sorted(set(core)):
            state_mask = anchors["state"].eq(current).to_numpy()
            state_anchors = anchors.loc[state_mask].reset_index(drop=True)
            state_target = np.zeros(len(state_anchors), dtype=bool)
            state_probabilities = {
                name: np.zeros(len(state_anchors), dtype=float)
                for name in MODEL_NAMES
            }
            paths = oriented_paths(core, current)
            if not paths:
                raise AssertionError("compatible cycle has no oriented path")
            for path in paths:
                state_target |= oriented_path_label(state_anchors, path)
                path_probabilities = oriented_path_probabilities(
                    state_anchors, path, models
                )
                for name in MODEL_NAMES:
                    state_probabilities[name] += path_probabilities[name]
            target[state_mask] = state_target
            for name in MODEL_NAMES:
                probabilities[name][state_mask] = state_probabilities[name]
        for name in MODEL_NAMES:
            if not np.isfinite(probabilities[name]).all():
                raise AssertionError(f"non-finite {name} path probability")
            if probabilities[name].min(initial=0.0) < 0.0:
                raise AssertionError(f"negative {name} path probability")
            if probabilities[name].max(initial=0.0) > 1.0 + 1e-9:
                raise AssertionError(f"{name} rotations exceed unit probability")
            probabilities[name] = np.clip(
                probabilities[name], EPSILON, 1.0 - EPSILON
            )
        output = anchors.loc[
            :,
            [
                "anchor_id",
                "period",
                "symbol_norm",
                "session_date",
                "quarter",
                "state",
                "previous_state_1",
                "previous_state_2",
            ],
        ].copy()
        output["cycle_id"] = cycle.cycle_id
        output["cycle"] = cycle.cycle
        output["transition_length"] = int(cycle.transition_length)
        output["target"] = target.astype(np.int8)
        for name, column in PROBABILITY_COLUMNS.items():
            output[column] = probabilities[name]
        rows.append(output)
    scoring = pd.concat(rows, ignore_index=True)
    scoring = scoring.sort_values(["anchor_id", "cycle_id"], kind="stable").reset_index(
        drop=True
    )
    if scoring.duplicated(["anchor_id", "cycle_id"]).any():
        raise AssertionError("duplicate anchor-cycle score")
    return scoring


def binary_loss(target: np.ndarray, probability: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=float)
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return {
        "log_loss": -(
            target * np.log(probability)
            + (1.0 - target) * np.log(1.0 - probability)
        ),
        "brier": np.square(probability - target),
    }


def calibration_rows(
    period: str, model: str, target: np.ndarray, probability: np.ndarray
) -> tuple[list[dict[str, Any]], float, float]:
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    bin_id = np.minimum((probability * 10.0).astype(int), 9)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    supported_errors: list[float] = []
    for index in range(10):
        mask = bin_id == index
        count = int(mask.sum())
        mean_probability = float(np.mean(probability[mask])) if count else math.nan
        event_rate = float(np.mean(target[mask])) if count else math.nan
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
                "bin": index,
                "lower": index / 10.0,
                "upper": (index + 1) / 10.0,
                "count": count,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "absolute_error": error,
                "supported": supported,
            }
        )
    maximum = max(supported_errors) if supported_errors else math.nan
    return rows, float(ece), float(maximum)


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


def top_three_metrics(
    scoring: pd.DataFrame, period: str, model: str, probability_column: str
) -> dict[str, Any]:
    ranked = scoring.sort_values(
        ["anchor_id", probability_column, "cycle_id"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()
    ranked["rank"] = ranked.groupby("anchor_id", sort=False).cumcount() + 1
    selected = ranked["rank"].le(3)
    positives = int(ranked["target"].sum())
    hits = int(ranked.loc[selected, "target"].sum())
    anchors_with_positive = int(
        ranked.groupby("anchor_id", sort=False)["target"].max().sum()
    )
    anchors_hit = int(
        ranked.assign(hit=(selected & ranked["target"].eq(1)).astype(int))
        .groupby("anchor_id", sort=False)["hit"]
        .max()
        .sum()
    )
    return {
        "period": period,
        "model": model,
        "anchors": int(ranked["anchor_id"].nunique()),
        "positive_labels": positives,
        "selected_labels": int(selected.sum()),
        "hits": hits,
        "recall": float(hits / positives) if positives else math.nan,
        "precision": float(hits / selected.sum()) if selected.any() else math.nan,
        "anchors_with_positive": anchors_with_positive,
        "positive_anchor_hit_rate": (
            float(anchors_hit / anchors_with_positive)
            if anchors_with_positive
            else math.nan
        ),
    }


def evaluate_period(
    scoring: pd.DataFrame, period: str, seed_offset: int
) -> dict[str, Any]:
    target = scoring["target"].to_numpy(dtype=int)
    calibration: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    top_three: list[dict[str, Any]] = []
    losses: dict[str, dict[str, np.ndarray]] = {}
    calibration_summary: dict[str, dict[str, float]] = {}
    top_summary: dict[str, dict[str, Any]] = {}
    for model, probability_column in PROBABILITY_COLUMNS.items():
        probability = scoring[probability_column].to_numpy(dtype=float)
        model_losses = binary_loss(target, probability)
        losses[model] = model_losses
        rows, ece, maximum = calibration_rows(period, model, target, probability)
        calibration.extend(rows)
        calibration_summary[model] = {
            "ece": ece,
            "maximum_supported_bin_error": maximum,
        }
        overall.append(
            {
                "period": period,
                "model": model,
                "rows": len(scoring),
                "positives": int(target.sum()),
                "prevalence": float(target.mean()),
                "log_loss": float(model_losses["log_loss"].mean()),
                "brier": float(model_losses["brier"].mean()),
                "ece": ece,
                "maximum_supported_bin_error": maximum,
            }
        )
        top = top_three_metrics(scoring, period, model, probability_column)
        top_three.append(top)
        top_summary[model] = top

    per_cycle: list[dict[str, Any]] = []
    for cycle_id, indices in scoring.groupby("cycle_id", sort=True).groups.items():
        positions = np.asarray(indices, dtype=int)
        row: dict[str, Any] = {
            "period": period,
            "cycle_id": cycle_id,
            "cycle": str(scoring.loc[positions[0], "cycle"]),
            "transition_length": int(
                scoring.loc[positions[0], "transition_length"]
            ),
            "rows": len(positions),
            "positives": int(target[positions].sum()),
            "prevalence": float(target[positions].mean()),
        }
        for model in MODEL_NAMES:
            row[f"{model}_log_loss"] = float(
                losses[model]["log_loss"][positions].mean()
            )
            row[f"{model}_brier"] = float(losses[model]["brier"][positions].mean())
        per_cycle.append(row)
    per_cycle_frame = pd.DataFrame(per_cycle)

    comparison_specs = (
        ("history", "first_order", "history_vs_first_order", 0.05),
        ("context", "history", "context_vs_history", 0.0),
    )
    comparisons: list[dict[str, Any]] = []
    comparison_gates: dict[str, dict[str, Any]] = {}
    for comparison_index, (
        candidate,
        baseline,
        comparison,
        recall_requirement,
    ) in enumerate(comparison_specs):
        loss_rows: list[dict[str, Any]] = []
        for loss_index, loss_name in enumerate(("log_loss", "brier")):
            difference = losses[candidate][loss_name] - losses[baseline][loss_name]
            daily = (
                pd.DataFrame(
                    {
                        "session_date": scoring["session_date"].to_numpy(),
                        "difference": difference,
                    }
                )
                .groupby("session_date", sort=True, as_index=False)["difference"]
                .mean()
            )
            daily_mean, ci_low, ci_high = moving_block_bounds(
                daily["difference"].to_numpy(dtype=float),
                SEED + seed_offset + comparison_index * 100 + loss_index,
            )
            quarter_means = pd.Series(difference).groupby(
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
                "comparison": comparison,
                "candidate": candidate,
                "baseline": baseline,
                "loss": loss_name,
                "row_mean_difference": float(difference.mean()),
                "daily_mean_difference": daily_mean,
                "daily_ci_low": ci_low,
                "daily_ci_high": ci_high,
                "baseline_mean_loss": baseline_mean,
                "relative_improvement": float(-difference.mean() / baseline_mean),
                "negative_quarter_count": int((quarter_means < 0.0).sum()),
                "leave_one_symbol_max_difference": max(deletions.values()),
                "leave_one_symbol_all_negative": bool(
                    max(deletions.values()) < 0.0
                ),
            }
            comparisons.append(row)
            loss_rows.append(row)
        candidate_cycle = per_cycle_frame[f"{candidate}_log_loss"]
        baseline_cycle = per_cycle_frame[f"{baseline}_log_loss"]
        negative_cycle_count = int((candidate_cycle < baseline_cycle).sum())
        recall_difference = float(
            top_summary[candidate]["recall"] - top_summary[baseline]["recall"]
        )
        intervals_pass = all(row["daily_ci_high"] < 0.0 for row in loss_rows)
        quarters_pass = all(
            row["negative_quarter_count"] == 4 for row in loss_rows
        )
        deletions_pass = all(
            row["leave_one_symbol_all_negative"] for row in loss_rows
        )
        relative_pass = loss_rows[0]["relative_improvement"] >= 0.005
        ece_pass = (
            calibration_summary[candidate]["ece"]
            <= calibration_summary[baseline]["ece"]
        )
        maximum_error_pass = (
            calibration_summary[candidate]["maximum_supported_bin_error"]
            <= calibration_summary[baseline]["maximum_supported_bin_error"] + 0.01
        )
        cycle_pass = negative_cycle_count >= 15
        recall_pass = recall_difference >= recall_requirement
        comparison_gates[comparison] = {
            "intervals_pass": intervals_pass,
            "relative_log_loss_pass": relative_pass,
            "quarters_pass": quarters_pass,
            "stock_deletions_pass": deletions_pass,
            "ece_pass": ece_pass,
            "maximum_supported_bin_error_pass": maximum_error_pass,
            "negative_cycle_count": negative_cycle_count,
            "per_cycle_pass": cycle_pass,
            "top_three_recall_difference": recall_difference,
            "top_three_recall_requirement": recall_requirement,
            "top_three_recall_pass": recall_pass,
            "pass": bool(
                intervals_pass
                and relative_pass
                and quarters_pass
                and deletions_pass
                and ece_pass
                and maximum_error_pass
                and cycle_pass
                and recall_pass
            ),
        }

    cycle_positive = scoring.groupby("cycle_id", sort=True)["target"].sum()
    support = {
        "rows": len(scoring),
        "positives": int(target.sum()),
        "cycles": int(scoring["cycle_id"].nunique()),
        "minimum_cycle_positives": int(cycle_positive.min()),
        "stocks": int(scoring["symbol_norm"].nunique()),
        "quarters": int(scoring["quarter"].nunique()),
        "current_states": int(scoring["state"].nunique()),
    }
    support["pass"] = bool(
        support["rows"] >= 300_000
        and support["positives"] >= 10_000
        and support["cycles"] == 20
        and support["minimum_cycle_positives"] >= 100
        and support["stocks"] >= 18
        and support["quarters"] == 4
        and support["current_states"] == K
    )
    return {
        "overall": pd.DataFrame(overall),
        "calibration": pd.DataFrame(calibration),
        "top_three": pd.DataFrame(top_three),
        "per_cycle": per_cycle_frame,
        "comparisons": pd.DataFrame(comparisons),
        "support": support,
        "comparison_gates": comparison_gates,
    }


def self_tests() -> None:
    assert canonical_cycle((2, 5, 1)) == canonical_cycle((5, 1, 2))
    assert oriented_paths((0, 1, 0, 1), 0) == [(0, 1, 0, 1, 0)]
    assert oriented_paths((1, 2, 1, 3), 1) == [
        (1, 2, 1, 3, 1),
        (1, 3, 1, 2, 1),
    ]
    tokens = history_tokens(
        np.asarray([8, 2]), np.asarray([8, 1]), np.asarray([0, 3])
    )
    assert np.array_equal(tokens, np.asarray([640, 155]))
    matrix = token_matrix(tokens)
    assert matrix.shape == (2, TOKEN_COUNT)
    assert np.array_equal(matrix.sum(axis=1).A1, np.ones(2))
    example = pd.DataFrame(
        {
            "future_state_1": [1, 1],
            "future_state_2": [0, END_STATE],
        }
    )
    assert np.array_equal(
        oriented_path_label(example, (0, 1, 0)), np.asarray([True, False])
    )
    target = np.asarray([0, 1])
    probability = np.asarray([0.25, 0.75])
    losses = binary_loss(target, probability)
    assert np.allclose(losses["log_loss"], -np.log(0.75))
    assert np.allclose(losses["brier"], 0.0625)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    self_tests()
    if args.self_test_only:
        print("self-tests passed")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "train_2024_filtered_runs.csv": sha256(TRAIN_PATH),
        "test_2025_filtered_runs.csv": sha256(TEST_2025_PATH),
        "backward_2023_filtered_runs.parquet": sha256(TEST_2023_PATH),
        "fixed_cycle_shuffled_nulls.csv": sha256(CYCLE_PATH),
    }
    write_json(OUT / "source_hashes.json", source_hashes)
    cycles = load_cycles()
    cycles.drop(columns="core").to_csv(OUT / "fixed_cycles.csv", index=False)

    train = load_runs(TRAIN_PATH, 2024, "train_2024")
    models = fit_models(train)
    feature_audit = {
        "training_rows": len(train),
        "training_stocks": int(train["symbol_norm"].nunique()),
        "training_dates": int(train["session_date"].nunique()),
        "training_outcome_counts": {
            str(key): int(value)
            for key, value in train["next_outcome"].value_counts().sort_index().items()
        },
        "token_count": TOKEN_COUNT,
        "history_iterations": models["history"].n_iter_.tolist(),
        "context_iterations": models["context"].n_iter_.tolist(),
        "context_columns": [
            "b0_state_numeric_at_entry",
            "b0_high_stress_at_entry",
            "entry_time_sin",
            "entry_time_cos",
        ],
        "explicitly_excluded": [
            "duration",
            "end_timestamp",
            "stored_time_sin",
            "stored_time_cos",
            "price_direction",
            "future_return",
            "pnl",
        ],
    }
    write_json(OUT / "feature_audit.json", feature_audit)
    np.savez_compressed(
        OUT / "model_parameters.npz",
        first_order=models["first_order"],
        history_classes=models["history"].classes_,
        history_coef=models["history"].coef_,
        history_intercept=models["history"].intercept_,
        history_n_iter=models["history"].n_iter_,
        context_classes=models["context"].classes_,
        context_coef=models["context"].coef_,
        context_intercept=models["context"].intercept_,
        context_n_iter=models["context"].n_iter_,
    )

    all_overall: list[pd.DataFrame] = []
    all_calibration: list[pd.DataFrame] = []
    all_top_three: list[pd.DataFrame] = []
    all_per_cycle: list[pd.DataFrame] = []
    all_comparisons: list[pd.DataFrame] = []
    gates: dict[str, Any] = {"periods": {}}
    period_specs = (
        ("2025", TEST_2025_PATH, 2025, 1000),
        ("2023", TEST_2023_PATH, 2023, 2000),
    )
    for period, path, year, seed_offset in period_specs:
        runs = load_runs(path, year, period)
        scoring = score_period(runs, cycles, models)
        scoring.to_parquet(OUT / f"scoring_{period}.parquet", index=False)
        evaluation = evaluate_period(scoring, period, seed_offset)
        all_overall.append(evaluation["overall"])
        all_calibration.append(evaluation["calibration"])
        all_top_three.append(evaluation["top_three"])
        all_per_cycle.append(evaluation["per_cycle"])
        all_comparisons.append(evaluation["comparisons"])
        gates["periods"][period] = {
            "support": evaluation["support"],
            **evaluation["comparison_gates"],
        }

    overall = pd.concat(all_overall, ignore_index=True)
    calibration = pd.concat(all_calibration, ignore_index=True)
    top_three = pd.concat(all_top_three, ignore_index=True)
    per_cycle = pd.concat(all_per_cycle, ignore_index=True)
    comparisons = pd.concat(all_comparisons, ignore_index=True)
    overall.to_csv(OUT / "overall_metrics.csv", index=False)
    calibration.to_csv(OUT / "calibration.csv", index=False)
    top_three.to_csv(OUT / "top_three_metrics.csv", index=False)
    per_cycle.to_csv(OUT / "per_cycle_metrics.csv", index=False)
    comparisons.to_csv(OUT / "comparisons.csv", index=False)

    gates["history_retained"] = bool(
        all(gates["periods"][period]["support"]["pass"] for period in ("2025", "2023"))
        and all(
            gates["periods"][period]["history_vs_first_order"]["pass"]
            for period in ("2025", "2023")
        )
    )
    gates["context_retained"] = bool(
        gates["history_retained"]
        and all(
            gates["periods"][period]["context_vs_history"]["pass"]
            for period in ("2025", "2023")
        )
    )
    gates["live_ordering_enabled"] = False
    gates["order_placement"] = "disabled"
    write_json(OUT / "gates.json", gates)

    summary = {
        "algorithm": "causal_fixed_loop_path_probability",
        "training_period": 2024,
        "scoring_periods": [2025, 2023],
        "models": list(MODEL_NAMES),
        "session_end_destination": END_STATE,
        "multi_label": True,
        "overall_metrics": overall.to_dict(orient="records"),
        "top_three_metrics": top_three.to_dict(orient="records"),
        "gates": gates,
        "interpretation": (
            "State-path forecasting only; no direction, return, economic edge, "
            "tradability, order, or deployment claim."
        ),
    }
    write_json(OUT / "summary.json", summary)
    lines = [
        "# Causal fixed-loop path forecast",
        "",
        f"- History retained: {gates['history_retained']}",
        f"- B0/stress/clock context retained: {gates['context_retained']}",
        "- Session end is explicit destination 8; outputs are overlapping multi-label cycle probabilities.",
        "- Research only; live ordering is disabled and no directional or economic claim was tested.",
        "",
        "## Overall metrics",
        "",
        "```text",
        overall.to_string(index=False),
        "```",
        "",
        "## Top-three ranking",
        "",
        "```text",
        top_three.to_string(index=False),
        "```",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(json.dumps(safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
