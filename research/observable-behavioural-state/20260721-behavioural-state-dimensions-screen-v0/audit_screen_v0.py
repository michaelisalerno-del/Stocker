#!/usr/bin/env python3
"""Independent auditor for Observable Behavioural-State Dimensions Screen V0.

This program deliberately does not import ``run_screen_v0``.  It reconstructs
the scientific calculations from frozen inputs and emitted ledgers and fails
closed when an unexplained difference is found.
"""

from __future__ import annotations

# ruff: noqa: E402 -- numerical thread limits must be fixed before imports.
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
import sys
import warnings
from collections import Counter
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

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PREDECESSOR = (
    REPO_ROOT
    / "research"
    / "opening-regime-path"
    / "20260720-opening-regime-path-direction-screen-v0"
    / "artifacts"
    / "primary"
)
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
START = pd.Timestamp("2024-01-01T00:00:00Z")
EPSILON = 1e-12
BOOTSTRAP_SEED = 20260721
NULL_SEED = 20260722
RANDOM_SELECTION_SEED = 20260724

SAFETY_FLAGS: dict[str, Any] = {
    "research_only": True,
    "feasibility_screen": True,
    "observable_only": True,
    "continuous_behavioural_dimensions": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "loops_regimes_states_and_structural_paths_forbidden": True,
}

SYMBOLS = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)

BASE_COMPONENTS = (
    "activity_effort",
    "range_effort",
    "travel_effort",
    "signed_progress",
    "absolute_progress",
    "signed_efficiency",
    "absolute_efficiency",
    "close_retention",
    "directional_persistence",
    "new_high_fraction",
    "new_low_fraction",
    "up_extreme_rejection",
    "down_extreme_rejection",
    "extreme_rejection",
    "compression",
    "normalised_high_slope",
    "normalised_low_slope",
    "boundary_slope",
    "activity_acceleration",
    "range_acceleration",
    "effort_acceleration",
    "signed_progress_acceleration",
    "return_gap",
    "activity_gap",
    "range_gap",
    "mean_close_location",
)
DERIVED_COMPONENTS = ("aligned_progress_acceleration", "directional_rejection")
DIMENSIONS = (
    "arousal",
    "conviction",
    "frustration",
    "tension",
    "signed_pressure",
    "pressure_magnitude",
    "exhaustion_magnitude",
    "signed_exhaustion",
    "independence",
    "signed_independence",
)
CONJUNCTIONS = (
    "active_conviction",
    "active_frustration",
    "pressurised_tension",
    "pressurised_exhaustion",
    "independent_pressure",
)
LABELS = (
    "CALM",
    "TENSE",
    "CONFLICTED",
    "BULLISH_PRESSURE",
    "BEARISH_PRESSURE",
    "UPWARD_PRESSURE_EXHAUSTING",
    "DOWNWARD_PRESSURE_EXHAUSTING",
    "INDEPENDENT",
)
FORBIDDEN = (
    "regime",
    "state",
    "loop",
    "closure",
    "excursion",
    "transition",
    "posterior",
    "structural_score",
    "future_price",
    "future_activity",
    "future_return",
    "mfe",
    "mae",
    "p&l",
    "pnl",
    "profit_history",
    "news",
    "bid_ask",
    "order_book",
    "broker",
    "symbol",
    "month",
    "behavioural_label",
)


class Audit:
    """Ordered fail-closed check ledger."""

    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.details: dict[str, Any] = {}
        self.failures: list[str] = []

    def check(self, name: str, condition: bool, detail: Any | None = None) -> None:
        passed = bool(condition)
        self.checks[name] = passed
        if detail is not None:
            self.details[name] = detail
        if not passed:
            self.failures.append(name)

    def close(self, name: str, actual: Any, expected: Any, tolerance: float = 1e-10) -> None:
        left = np.asarray(actual, dtype=float)
        right = np.asarray(expected, dtype=float)
        same_shape = left.shape == right.shape
        error = (
            float(np.nanmax(np.abs(left - right)))
            if same_shape and left.size and np.isfinite(left - right).any()
            else (0.0 if same_shape and left.size == 0 else math.inf)
        )
        equal_nan = same_shape and np.allclose(
            left, right, rtol=0.0, atol=tolerance, equal_nan=True
        )
        self.check(name, equal_nan, {"maximum_absolute_error": error, "tolerance": tolerance})


def canonical_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrow_hash(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def manual_probability(model: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    names = [str(value) for value in model["feature_names"]]
    values = frame.loc[:, names].to_numpy(dtype=float)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    linear = float(model["intercept"]) + ((values - means) / scales) @ coefficients
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))


def bounded_source(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        filters=[
            ("timestamp", ">=", START.to_pydatetime()),
            ("timestamp", "<", PROTECTED_START.to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def prepare_bars(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    local = raw["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    in_regular = minute.ge(570) & minute.lt(960)
    on_grid = ((minute - 570) % 5).eq(0) & local.dt.second.eq(0) & local.dt.microsecond.eq(0)
    invalid_sessions = set(local.loc[in_regular & ~on_grid].dt.strftime("%Y-%m-%d"))
    regular = raw.loc[in_regular & on_grid].copy()
    local = regular["timestamp"].dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    regular["symbol"] = symbol
    regular["session"] = local.dt.strftime("%Y-%m-%d")
    regular["bar_ordinal"] = ((minute - 570) // 5).astype(np.int16)
    valid: list[pd.DataFrame] = []
    for session, part in regular.groupby("session", sort=True):
        part = part.sort_values("bar_ordinal", kind="mergesort")
        prices = part[["open", "high", "low", "close"]].to_numpy(dtype=float)
        if (
            str(session) not in invalid_sessions
            and len(part) == 78
            and part["bar_ordinal"].astype(int).tolist() == list(range(78))
            and np.isfinite(prices).all()
            and bool((prices > 0.0).all())
        ):
            valid.append(part)
    frame = pd.concat(valid, ignore_index=True).sort_values(
        ["session", "bar_ordinal"], kind="mergesort"
    )
    baseline = frame.groupby("bar_ordinal", sort=False)["volume"].transform(
        lambda values: values.expanding(min_periods=10).mean().shift(1)
    )
    frame["historical_relative_activity"] = frame["volume"] / baseline.replace(0.0, np.nan)
    return frame.reset_index(drop=True)


def range_baselines(bars: pd.DataFrame) -> dict[tuple[str, int], float]:
    rows: list[dict[str, Any]] = []
    for session, part in bars.groupby("session", sort=True):
        ordered = part.sort_values("bar_ordinal", kind="mergesort")
        for checkpoint in (6, 12):
            opening = ordered.iloc[:checkpoint]
            opening_range = (
                10_000.0
                * (float(opening["high"].max()) - float(opening["low"].min()))
                / float(opening.iloc[0]["open"])
            )
            rows.append(
                {
                    "session": str(session),
                    "decision_ordinal": checkpoint,
                    "opening_range_bps": opening_range,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["decision_ordinal", "session"], kind="mergesort")
    frame["baseline"] = frame.groupby("decision_ordinal", sort=False)[
        "opening_range_bps"
    ].transform(lambda values: values.expanding(min_periods=1).median().shift(1))
    return {
        (str(row.session), int(row.decision_ordinal)): float(row.baseline)
        for row in frame.itertuples(index=False)
        if np.isfinite(float(row.baseline)) and float(row.baseline) > 0.0
    }


def leave_one_out(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.asarray(
        [value - np.median(np.delete(array, index)) for index, value in enumerate(array)]
    )


def cohort_median_gap(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array - float(np.median(array))


def slope(values: np.ndarray) -> float:
    x = np.arange(len(values), dtype=float)
    centered = x - x.mean()
    return float(centered @ (values - values.mean()) / (centered @ centered))


def audit_inputs_and_sources(
    audit: Audit,
    artifacts: Path,
    provider_root: Path,
) -> dict[str, pd.DataFrame]:
    contract = json.loads((artifacts / "contract.json").read_text(encoding="utf-8"))
    safety_ok = all(
        contract.get(key) == value and contract.get("safety", {}).get(key) == value
        for key, value in SAFETY_FLAGS.items()
    )
    audit.check("safety_flags", safety_ok)
    input_hashes = json.loads(
        (artifacts / "input_artifact_hashes.json").read_text(encoding="utf-8")
    )["predecessor_artifacts"]
    input_results = []
    for record in input_hashes:
        path = REPO_ROOT / record["repository_relative_path"]
        actual = sha256_file(path) if path.is_file() else None
        input_results.append(actual == record["sha256"])
    audit.check("input_artifact_hashes", all(input_results), {"artifacts": len(input_results)})

    manifest = json.loads((artifacts / "source_manifest.json").read_text(encoding="utf-8"))
    protected_audit = json.loads(
        (artifacts / "protected_boundary_audit.json").read_text(encoding="utf-8")
    )
    source_frames: dict[str, pd.DataFrame] = {}
    source_results = []
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    for record in manifest["sources"]:
        symbol = str(record["symbol"])
        path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        raw = bounded_source(path)
        source_frames[symbol] = prepare_bars(raw, symbol)
        source_results.append(
            arrow_hash(raw) == record["bounded_safe_hash"]
            and len(raw) == int(record["bounded_safe_rows"])
        )
        value_min = pd.Timestamp(raw["timestamp"].min())
        value_max = pd.Timestamp(raw["timestamp"].max())
        minimum = value_min if minimum is None else min(minimum, value_min)
        maximum = value_max if maximum is None else max(maximum, value_max)
    audit.check("source_hashes_and_counts", all(source_results), {"symbols": len(source_results)})
    qa_root = Path.home() / "StockerLocal" / "data" / "reports" / "vendor_qa"
    qa_results = []
    for record in manifest["vendor_qa"]:
        path = qa_root / Path(str(record["logical_path"])).name
        qa_results.append(
            path.is_file()
            and sha256_file(path) == record["sha256"]
            and record["status"] in {"pass", "warning"}
            and int(record["validation_error_count"]) == 0
            and int(record["adjusted_close_differences"] or 0) == 0
            and bool(record["corporate_action_check_passed"])
        )
    audit.check("qa_and_corporate_action_ledgers", all(qa_results), {"records": len(qa_results)})
    audit.check(
        "protected_boundary",
        maximum is not None
        and maximum < PROTECTED_START
        and manifest.get(
            "protected_rows_materialised",
            protected_audit["protected_rows_materialised"],
        )
        == 0
        and manifest["maximum_timestamp_read"] == str(maximum),
        {"minimum_timestamp_read": str(minimum), "maximum_timestamp_read": str(maximum)},
    )
    return source_frames


def audit_expected_population(
    audit: Audit,
    compact: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame],
    artifacts: Path,
) -> None:
    predecessor = pd.read_parquet(PREDECESSOR / "opening_decision_panel.parquet")
    candidate_keys: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_history_ok = True
    for symbol in SYMBOLS:
        bars = source_frames[symbol]
        sessions = {
            str(session): rows.sort_values("bar_ordinal", kind="mergesort")
            for session, rows in bars.groupby("session", sort=True)
        }
        baselines = range_baselines(bars)
        requested = predecessor.loc[predecessor["symbol"].eq(symbol)]
        for row in requested.itertuples(index=False):
            session = str(row.session)
            checkpoint = int(row.decision_ordinal)
            source_history_ok = (
                source_history_ok
                and session in sessions
                and (
                    session,
                    checkpoint,
                )
                in baselines
            )
            if session not in sessions or (session, checkpoint) not in baselines:
                continue
            activity = (
                sessions[session]
                .iloc[:checkpoint]["historical_relative_activity"]
                .to_numpy(dtype=float)
            )
            if not np.isfinite(activity).all() or bool((activity < 0.0).any()):
                exclusions.append(
                    {
                        "symbol": symbol,
                        "session": session,
                        "decision_ordinal": checkpoint,
                        "reason": "incomplete_causal_historical_activity_proxy",
                    }
                )
                continue
            candidate_keys.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "decision_ordinal": checkpoint,
                    "slate_id": str(row.slate_id),
                }
            )
    audit.check("predecessor_source_history_available", source_history_ok)
    candidates = pd.DataFrame(candidate_keys)
    sizes = candidates.groupby("slate_id", sort=False)["symbol"].transform("size")
    undersized = candidates.loc[sizes.lt(15)]
    exclusions.extend(
        {
            "symbol": str(row.symbol),
            "session": str(row.session),
            "decision_ordinal": int(row.decision_ordinal),
            "reason": "parent_slate_below_15_valid_stocks",
        }
        for row in undersized.itertuples(index=False)
    )
    expected = candidates.loc[sizes.ge(15)].sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    actual = compact[["symbol", "session", "decision_ordinal", "slate_id"]].sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    audit.check(
        "independent_eligibility_population",
        expected.reset_index(drop=True).equals(actual.reset_index(drop=True)),
        {"expected_rows": len(expected), "actual_rows": len(actual)},
    )
    manifest = json.loads((artifacts / "source_manifest.json").read_text(encoding="utf-8"))

    def identity(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
        return (
            str(row["symbol"]),
            str(row["session"]),
            int(row["decision_ordinal"]),
            str(row["reason"]),
        )

    audit.check(
        "eligibility_exclusion_ledger",
        Counter(identity(row) for row in exclusions)
        == Counter(identity(row) for row in manifest["causal_feature_exclusions"]),
        {"expected_exclusions": len(exclusions)},
    )


def audit_predecessor(audit: Audit, artifacts: Path) -> None:
    panel = pd.read_parquet(PREDECESSOR / "opening_decision_panel.parquet")
    archived = pd.read_parquet(PREDECESSOR / "assessment_predictions.parquet")
    models = json.loads((PREDECESSOR / "model_coefficients.json").read_text(encoding="utf-8"))[
        "models"
    ]
    reconstruction = json.loads(
        (artifacts / "predecessor_reconstruction.json").read_text(encoding="utf-8")
    )
    assessment = panel.loc[panel["year"].eq(2025)].sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    archived = archived.sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    maximum_errors: dict[str, float] = {}
    metric_errors: dict[str, float] = {}
    for target, metric_name in (
        ("large_remaining_move", "movement_metrics.csv"),
        ("up_given_large_move", "direction_metrics.csv"),
    ):
        all_probability = manual_probability(models[target]["M1"], assessment)
        maximum_errors[target] = float(
            np.max(np.abs(all_probability - archived[f"p__{target}__M1"].to_numpy(dtype=float)))
        )
        metric_mask = (
            np.ones(len(assessment), dtype=bool)
            if target == "large_remaining_move"
            else assessment["large_remaining_move"].eq(1).to_numpy(dtype=bool)
        )
        probability = all_probability[metric_mask]
        population = (
            assessment
            if target == "large_remaining_move"
            else assessment.loc[assessment["large_remaining_move"].eq(1)]
        )
        labels = population[target].to_numpy(dtype=int)
        metrics = {
            "brier": float(brier_score_loss(labels, probability)),
            "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
            "auc": float(roc_auc_score(labels, probability)),
        }
        archived_metric = pd.read_csv(PREDECESSOR / metric_name)
        archived_metric = archived_metric.loc[
            archived_metric["scope"].eq("pooled") & archived_metric["model"].eq("M1")
        ].iloc[0]
        metric_errors[target] = max(
            abs(metrics["brier"] - float(archived_metric["brier_score"])),
            abs(metrics["log_loss"] - float(archived_metric["log_loss"])),
            abs(metrics["auc"] - float(archived_metric["auc"])),
        )
        recorded = reconstruction["targets"][target]
        audit.check(
            f"predecessor_record_{target}",
            bool(recorded["passed"])
            and int(recorded["probability_reconstruction_rows"]) == len(assessment)
            and abs(float(recorded["maximum_prediction_absolute_error"]) - maximum_errors[target])
            <= 1e-15,
        )
    audit.check(
        "exact_predecessor_reconstruction",
        max(maximum_errors.values()) <= 1e-12 and max(metric_errors.values()) <= 1e-12,
        {"probability_errors": maximum_errors, "metric_errors": metric_errors},
    )


def recompute_raw_components(
    audit: Audit,
    compact: pd.DataFrame,
    ledger: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    keys = ["symbol", "session", "decision_ordinal", "slate_id"]
    audit.check("component_ledger_keys", compact[keys].equals(ledger[keys]))
    baselines = {symbol: range_baselines(frame) for symbol, frame in source_frames.items()}
    sessions = {
        symbol: {
            str(session): rows.sort_values("bar_ordinal", kind="mergesort")
            for session, rows in frame.groupby("session", sort=True)
        }
        for symbol, frame in source_frames.items()
    }
    records: list[dict[str, Any]] = []
    path_error = 0.0
    for row in ledger.itertuples(index=False):
        symbol = str(row.symbol)
        session = str(row.session)
        checkpoint = int(row.decision_ordinal)
        opening = sessions[symbol][session].iloc[:checkpoint]
        open_ = opening["open"].to_numpy(dtype=float)
        high = opening["high"].to_numpy(dtype=float)
        low = opening["low"].to_numpy(dtype=float)
        close = opening["close"].to_numpy(dtype=float)
        activity = opening["historical_relative_activity"].to_numpy(dtype=float)
        for source, emitted in (
            (open_, row.bar_open),
            (high, row.bar_high),
            (low, row.bar_low),
            (close, row.bar_close),
            (activity, row.historical_relative_activity),
        ):
            path_error = max(path_error, float(np.max(np.abs(source - np.asarray(emitted)))))
        previous = np.roll(close, 1)
        previous[0] = open_[0]
        returns = 10_000.0 * (close / previous - 1.0)
        width = high - low
        true_range = (
            10_000.0
            * np.maximum.reduce([width, np.abs(high - previous), np.abs(low - previous)])
            / previous
        )
        nonzero = width > EPSILON
        close_location = np.full(checkpoint, 0.5, dtype=float)
        upper = np.zeros(checkpoint, dtype=float)
        lower = np.zeros(checkpoint, dtype=float)
        close_location[nonzero] = (close[nonzero] - low[nonzero]) / width[nonzero]
        upper[nonzero] = (high[nonzero] - np.maximum(open_[nonzero], close[nonzero])) / width[
            nonzero
        ]
        lower[nonzero] = (np.minimum(open_[nonzero], close[nonzero]) - low[nonzero]) / width[
            nonzero
        ]
        new_high = np.ones(checkpoint, dtype=bool)
        new_low = np.ones(checkpoint, dtype=bool)
        new_high[1:] = high[1:] > np.maximum.accumulate(high)[:-1]
        new_low[1:] = low[1:] < np.minimum.accumulate(low)[:-1]
        half = checkpoint // 2
        opening_range = float(high.max() - low.min())
        opening_range_bps = 10_000.0 * opening_range / open_[0]
        baseline = baselines[symbol][(session, checkpoint)]
        signed_efficiency = float(returns.sum() / max(float(np.abs(returns).sum()), EPSILON))
        cumulative = 10_000.0 * (close[-1] / open_[0] - 1.0)
        persistence = (
            0.5
            if abs(cumulative) <= EPSILON
            else float(np.mean(np.sign(returns) == np.sign(cumulative)))
        )
        activity_acceleration = float(
            np.log1p(activity[half:]).mean() - np.log1p(activity[:half]).mean()
        )
        range_acceleration = float(true_range[half:].mean() - true_range[:half].mean())
        high_slope = slope(high) / max(opening_range, EPSILON)
        low_slope = slope(low) / max(opening_range, EPSILON)
        record = {
            "activity_effort": float(np.log1p(activity.mean())),
            "range_effort": float(np.log1p(true_range.sum())),
            "travel_effort": float(np.log1p(np.abs(returns).sum())),
            "signed_efficiency": signed_efficiency,
            "absolute_efficiency": abs(signed_efficiency),
            "close_retention": abs(close[-1] - open_[0]) / max(float(width.sum()), EPSILON),
            "directional_persistence": persistence,
            "new_high_fraction": float(new_high.mean()),
            "new_low_fraction": float(new_low.mean()),
            "up_extreme_rejection": float(upper[new_high].mean()),
            "down_extreme_rejection": float(lower[new_low].mean()),
            "compression": -float(np.log(max(opening_range_bps / baseline, EPSILON))),
            "normalised_high_slope": high_slope,
            "normalised_low_slope": low_slope,
            "activity_acceleration": activity_acceleration,
            "range_acceleration": range_acceleration,
            "mean_close_location": float(close_location.mean()),
            "earlier_half_return_bps": float(returns[:half].sum()),
            "recent_half_return_bps": float(returns[half:].sum()),
        }
        record["extreme_rejection"] = 0.5 * (
            record["up_extreme_rejection"] + record["down_extreme_rejection"]
        )
        record["boundary_slope"] = 0.5 * (high_slope + low_slope)
        record["effort_acceleration"] = 0.5 * (activity_acceleration + range_acceleration)
        records.append(record)
    calculated = pd.DataFrame(records)
    calculated["symbol"] = ledger["symbol"].to_numpy()
    calculated["slate_id"] = ledger["slate_id"].to_numpy()
    calculated["decision_ordinal"] = ledger["decision_ordinal"].to_numpy()
    audit.check("source_paths_and_activity_normalisation", path_error <= 1e-12, path_error)

    for _, indices in calculated.groupby("slate_id", sort=True).groups.items():
        index = list(indices)
        current = compact.loc[index, "open_to_decision_raw_return_bps"].to_numpy(dtype=float)
        return_gap = cohort_median_gap(current)
        earlier = cohort_median_gap(calculated.loc[index, "earlier_half_return_bps"])
        recent = cohort_median_gap(calculated.loc[index, "recent_half_return_bps"])
        activity_gap = cohort_median_gap(calculated.loc[index, "activity_effort"])
        range_gap = cohort_median_gap(calculated.loc[index, "range_effort"])
        signed_progress = compact.loc[
            index, "open_to_decision_cohort_relative_return_bps"
        ].to_numpy(dtype=float)
        calculated.loc[index, "signed_progress"] = signed_progress
        calculated.loc[index, "absolute_progress"] = np.abs(signed_progress)
        calculated.loc[index, "return_gap"] = return_gap
        calculated.loc[index, "signed_progress_acceleration"] = recent - earlier
        calculated.loc[index, "activity_gap"] = activity_gap
        calculated.loc[index, "range_gap"] = range_gap
    for component in BASE_COMPONENTS:
        audit.close(
            f"raw_component_{component}",
            calculated[component],
            ledger[component],
            1e-9,
        )
    return calculated


def robust_scale(values: pd.Series) -> tuple[float, float]:
    center = float(values.median())
    scale = float(values.quantile(0.75, interpolation="linear")) - float(
        values.quantile(0.25, interpolation="linear")
    )
    return center, scale if np.isfinite(scale) and scale >= EPSILON else 1.0


def audit_scaling_dimensions_and_labels(
    audit: Audit,
    component: pd.DataFrame,
    dimensions: pd.DataFrame,
    scaling_manifest: Mapping[str, Any],
    dimension_manifest: Mapping[str, Any],
    label_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    work = component.copy()
    for checkpoint in (6, 12):
        checkpoint_mask = work["decision_ordinal"].eq(checkpoint)
        development_mask = checkpoint_mask & work["year"].eq(2024)
        for component_name in BASE_COMPONENTS:
            center, scale = robust_scale(work.loc[development_mask, component_name])
            frozen = scaling_manifest["base_components"][str(checkpoint)][component_name]
            audit.close(f"scale_center_{checkpoint}_{component_name}", center, frozen["center"])
            audit.close(f"scale_iqr_{checkpoint}_{component_name}", scale, frozen["scale"])
            expected = np.clip(
                (work.loc[checkpoint_mask, component_name].to_numpy(dtype=float) - center) / scale,
                -5.0,
                5.0,
            )
            target = f"z_{component_name}"
            work.loc[checkpoint_mask, target] = expected
            audit.close(
                f"standardised_{checkpoint}_{component_name}",
                expected,
                dimensions.loc[checkpoint_mask, target],
            )
    work["signed_pressure"] = work[
        [
            "z_signed_progress",
            "z_signed_efficiency",
            "z_mean_close_location",
            "z_boundary_slope",
        ]
    ].mean(axis=1)
    pressure = work["signed_pressure"].to_numpy(dtype=float)
    pressure_sign = np.sign(pressure)
    pressure_sign[np.abs(pressure) <= EPSILON] = 0.0
    work["aligned_progress_acceleration"] = pressure_sign * work[
        "signed_progress_acceleration"
    ].to_numpy(dtype=float)
    work["directional_rejection"] = np.where(
        pressure_sign > 0.0,
        work["up_extreme_rejection"],
        np.where(
            pressure_sign < 0.0,
            work["down_extreme_rejection"],
            0.5 * (work["up_extreme_rejection"] + work["down_extreme_rejection"]),
        ),
    )
    audit.close(
        "aligned_progress_acceleration",
        work["aligned_progress_acceleration"],
        component["aligned_progress_acceleration"],
    )
    audit.close(
        "directional_rejection", work["directional_rejection"], component["directional_rejection"]
    )
    for checkpoint in (6, 12):
        checkpoint_mask = work["decision_ordinal"].eq(checkpoint)
        development_mask = checkpoint_mask & work["year"].eq(2024)
        for component_name in DERIVED_COMPONENTS:
            center, scale = robust_scale(work.loc[development_mask, component_name])
            frozen = scaling_manifest["pressure_aligned_components"][str(checkpoint)][
                component_name
            ]
            audit.close(
                f"derived_scale_center_{checkpoint}_{component_name}", center, frozen["center"]
            )
            audit.close(f"derived_scale_iqr_{checkpoint}_{component_name}", scale, frozen["scale"])
            expected = np.clip(
                (work.loc[checkpoint_mask, component_name].to_numpy(dtype=float) - center) / scale,
                -5.0,
                5.0,
            )
            work.loc[checkpoint_mask, f"z_{component_name}"] = expected
            audit.close(
                f"standardised_{checkpoint}_{component_name}",
                expected,
                dimensions.loc[checkpoint_mask, f"z_{component_name}"],
            )

    expected_dimensions = pd.DataFrame(index=work.index)
    expected_dimensions["arousal"] = work[
        ["z_activity_effort", "z_range_effort", "z_travel_effort"]
    ].mean(axis=1)
    expected_dimensions["conviction"] = work[
        ["z_absolute_efficiency", "z_close_retention", "z_directional_persistence"]
    ].mean(axis=1)
    expected_dimensions["frustration"] = work[
        ["z_activity_effort", "z_travel_effort", "z_extreme_rejection"]
    ].mean(axis=1) - work[["z_absolute_progress", "z_absolute_efficiency"]].mean(axis=1)
    expected_dimensions["tension"] = (
        work[["z_activity_effort", "z_compression", "z_extreme_rejection"]].mean(axis=1)
        - work["z_absolute_progress"]
    )
    expected_dimensions["signed_pressure"] = work["signed_pressure"]
    expected_dimensions["pressure_magnitude"] = work["signed_pressure"].abs()
    expected_dimensions["exhaustion_magnitude"] = (
        work["z_effort_acceleration"]
        - work["z_aligned_progress_acceleration"]
        + work["z_directional_rejection"]
    )
    expected_dimensions["signed_exhaustion"] = (
        pressure_sign * expected_dimensions["exhaustion_magnitude"]
    )
    expected_dimensions["independence"] = (
        work[["z_return_gap", "z_activity_gap", "z_range_gap"]].abs().mean(axis=1)
    )
    expected_dimensions["signed_independence"] = (
        np.sign(work["return_gap"]) * expected_dimensions["independence"]
    )
    for name in DIMENSIONS:
        audit.close(f"dimension_{name}", expected_dimensions[name], dimensions[name])
        work[name] = expected_dimensions[name]

    raw_conjunctions = {
        "active_conviction": work["arousal"] * work["conviction"],
        "active_frustration": work["arousal"] * work["frustration"],
        "pressurised_tension": work["tension"] * work["pressure_magnitude"],
        "pressurised_exhaustion": work["exhaustion_magnitude"] * work["pressure_magnitude"],
        "independent_pressure": work["independence"] * work["signed_pressure"],
    }
    for checkpoint in (6, 12):
        mask = work["decision_ordinal"].eq(checkpoint)
        development = mask & work["year"].eq(2024)
        for name, values in raw_conjunctions.items():
            lower = float(values.loc[development].quantile(0.01, interpolation="linear"))
            upper = float(values.loc[development].quantile(0.99, interpolation="linear"))
            frozen = dimension_manifest["conjunction_clip_bounds"][str(checkpoint)][name]
            audit.close(f"conjunction_q01_{checkpoint}_{name}", lower, frozen["q01"])
            audit.close(f"conjunction_q99_{checkpoint}_{name}", upper, frozen["q99"])
            expected = values.loc[mask].clip(lower, upper)
            work.loc[mask, name] = expected
            audit.close(f"conjunction_{checkpoint}_{name}", expected, dimensions.loc[mask, name])

    threshold_specs = {
        "arousal_q30": ("arousal", 0.30),
        "tension_q70": ("tension", 0.70),
        "frustration_q70": ("frustration", 0.70),
        "signed_pressure_q30": ("signed_pressure", 0.30),
        "signed_pressure_q70": ("signed_pressure", 0.70),
        "conviction_q60": ("conviction", 0.60),
        "exhaustion_magnitude_q70": ("exhaustion_magnitude", 0.70),
        "independence_q70": ("independence", 0.70),
    }
    for checkpoint in (6, 12):
        mask = work["decision_ordinal"].eq(checkpoint)
        development = mask & work["year"].eq(2024)
        thresholds: dict[str, float] = {}
        for threshold, (name, quantile) in threshold_specs.items():
            value = float(work.loc[development, name].quantile(quantile, interpolation="linear"))
            thresholds[threshold] = value
            audit.close(
                f"label_threshold_{checkpoint}_{threshold}",
                value,
                label_manifest["thresholds"][str(checkpoint)][threshold],
            )
        expected_labels = {
            "CALM": work.loc[mask, "arousal"].le(thresholds["arousal_q30"]),
            "TENSE": work.loc[mask, "tension"].ge(thresholds["tension_q70"]),
            "CONFLICTED": work.loc[mask, "frustration"].ge(thresholds["frustration_q70"]),
            "BULLISH_PRESSURE": work.loc[mask, "signed_pressure"].ge(
                thresholds["signed_pressure_q70"]
            )
            & work.loc[mask, "conviction"].ge(thresholds["conviction_q60"]),
            "BEARISH_PRESSURE": work.loc[mask, "signed_pressure"].le(
                thresholds["signed_pressure_q30"]
            )
            & work.loc[mask, "conviction"].ge(thresholds["conviction_q60"]),
            "UPWARD_PRESSURE_EXHAUSTING": work.loc[mask, "signed_pressure"].gt(0.0)
            & work.loc[mask, "exhaustion_magnitude"].ge(thresholds["exhaustion_magnitude_q70"]),
            "DOWNWARD_PRESSURE_EXHAUSTING": work.loc[mask, "signed_pressure"].lt(0.0)
            & work.loc[mask, "exhaustion_magnitude"].ge(thresholds["exhaustion_magnitude_q70"]),
            "INDEPENDENT": work.loc[mask, "independence"].ge(thresholds["independence_q70"]),
        }
        for label, expected in expected_labels.items():
            audit.check(
                f"label_{checkpoint}_{label}",
                np.array_equal(
                    expected.to_numpy(dtype=bool),
                    dimensions.loc[mask, f"label__{label}"].to_numpy(dtype=bool),
                ),
            )
    return work


def equal_slate_weights(frame: pd.DataFrame) -> np.ndarray:
    sizes = (
        frame.groupby("slate_id", sort=False)["slate_id"].transform("size").to_numpy(dtype=float)
    )
    return 1.0 / sizes


def refit_model(frame: pd.DataFrame, target: str, features: Sequence[str]) -> dict[str, Any]:
    values = frame.loc[:, list(features)].to_numpy(dtype=float)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales >= EPSILON), scales, 1.0)
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=20260721,
        n_jobs=1,
    )
    estimator.fit(
        (values - means) / scales,
        frame[target].to_numpy(dtype=int),
        sample_weight=equal_slate_weights(frame),
    )
    return {
        "means": means,
        "scales": scales,
        "coefficients": estimator.coef_[0],
        "intercept": float(estimator.intercept_[0]),
        "iterations": int(np.max(estimator.n_iter_)),
    }


def audit_models(
    audit: Audit, full: pd.DataFrame, assessment: pd.DataFrame, artifacts: Path
) -> None:
    configs = json.loads((artifacts / "model_configurations.json").read_text(encoding="utf-8"))
    coefficients = json.loads((artifacts / "model_coefficients.json").read_text(encoding="utf-8"))[
        "models"
    ]
    configuration = configs["configuration"]
    audit.check(
        "fixed_model_configuration",
        configuration
        == {
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 250,
            "class_weight": None,
            "n_jobs": 1,
        }
        and configs["primary_fitted_model_count"] == 6,
    )
    forbidden_hits: list[str] = []
    for ladder in configs["features"].values():
        for features in ladder.values():
            for feature in features:
                normalised = str(feature).casefold().replace("/", "_").replace(" ", "_")
                if (
                    any(fragment in normalised for fragment in FORBIDDEN)
                    or normalised.startswith("label__")
                    or normalised in {label.casefold() for label in LABELS}
                ):
                    forbidden_hits.append(str(feature))
    audit.check("forbidden_feature_absence", not forbidden_hits, sorted(set(forbidden_hits)))
    audit.check(
        "fixed_new_feature_sets",
        set(configs["features"]["movement"]["P2"]) - set(configs["features"]["movement"]["P1"])
        == set(DIMENSIONS)
        and set(configs["features"]["movement"]["P3"]) - set(configs["features"]["movement"]["P2"])
        == set(CONJUNCTIONS),
    )
    development = full.loc[full["year"].eq(2024)].reset_index(drop=True)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for target_name, target, model_names in (
        ("movement", "large_remaining_move", ("P0", "P2", "P3")),
        ("direction", "up_given_large_move", ("D0", "D2", "D3")),
    ):
        training = (
            development
            if target_name == "movement"
            else development.loc[development["large_remaining_move"].eq(1)].reset_index(drop=True)
        )
        for model_name in model_names:
            model = coefficients[target_name][model_name]
            fitted = refit_model(training, target, model["feature_names"])
            audit.close(f"model_means_{model_name}", fitted["means"], model["means"], 1e-12)
            audit.close(f"model_scales_{model_name}", fitted["scales"], model["scales"], 1e-12)
            audit.close(
                f"model_coefficients_{model_name}",
                fitted["coefficients"],
                model["coefficients"],
                1e-12,
            )
            audit.close(
                f"model_intercept_{model_name}", fitted["intercept"], model["intercept"], 1e-12
            )
            audit.check(
                f"model_convergence_{model_name}",
                fitted["iterations"] < 250 and bool(model["converged"]),
            )
    for target_name, models, prefix in (
        ("movement", ("P0", "P1", "P2", "P3"), "p_large_remaining_move__"),
        ("direction", ("D0", "D1", "D2", "D3"), "p_up_given_large_move__"),
    ):
        for model_name in models:
            probability = manual_probability(coefficients[target_name][model_name], assessment)
            audit.close(
                f"manual_prediction_{model_name}",
                probability,
                assessment[f"{prefix}{model_name}"],
                1e-12,
            )


def calibration(labels: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    predictor = np.log(
        np.clip(probability, 1e-9, 1.0 - 1e-9) / (1.0 - np.clip(probability, 1e-9, 1.0 - 1e-9))
    )
    if float(np.std(predictor)) < EPSILON:
        rate = float(labels.mean())
        return float(np.log(rate / (1.0 - rate))), 0.0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + parameters[1] * predictor
        predicted = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        loss_value = -float(
            np.sum(
                labels * np.log(np.clip(predicted, 1e-12, 1.0))
                + (1 - labels) * np.log(np.clip(1.0 - predicted, 1e-12, 1.0))
            )
        )
        gradient = np.asarray(
            [np.sum(predicted - labels), np.sum((predicted - labels) * predictor)]
        )
        return loss_value, gradient

    result = minimize(
        lambda parameters: objective(parameters)[0],
        np.asarray([0.0, 1.0]),
        jac=lambda parameters: objective(parameters)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    return float(result.x[0]), float(result.x[1])


def metric_record(frame: pd.DataFrame, target: str, model: str, column: str) -> dict[str, float]:
    population = (
        frame
        if target == "large_remaining_move"
        else frame.loc[frame["large_remaining_move"].eq(1)]
    )
    labels = population[target].to_numpy(dtype=int)
    probability = population[column].to_numpy(dtype=float)
    intercept, slope_value = calibration(labels, probability)
    bins = np.minimum((np.clip(probability, 0.0, 1.0) * 10).astype(int), 9)
    ece = sum(
        float((bins == number).mean())
        * abs(float(probability[bins == number].mean()) - float(labels[bins == number].mean()))
        for number in range(10)
        if bool((bins == number).any())
    )
    return {
        "brier_score": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "auc": float(roc_auc_score(labels, probability)),
        "calibration_intercept": intercept,
        "calibration_slope": slope_value,
        "expected_calibration_error": float(ece),
        "base_rate": float(labels.mean()),
        "row_count": float(len(population)),
        "session_count": float(population["session"].nunique()),
        "stock_count": float(population["symbol"].nunique()),
    }


def audit_metrics(audit: Audit, assessment: pd.DataFrame, artifacts: Path) -> None:
    files = [
        pd.read_csv(artifacts / "movement_metrics.csv"),
        pd.read_csv(artifacts / "direction_metrics.csv"),
        pd.read_csv(artifacts / "monthly_metrics.csv"),
        pd.read_csv(artifacts / "checkpoint_metrics.csv"),
    ]
    all_metrics = pd.concat(files, ignore_index=True)
    numeric = (
        "brier_score",
        "log_loss",
        "auc",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "base_rate",
        "row_count",
        "session_count",
        "stock_count",
    )
    maximum_error = 0.0
    for row in all_metrics.itertuples(index=False):
        if row.scope_type == "pooled":
            scope = assessment
        elif row.scope_type == "month":
            scope = assessment.loc[assessment["year_month"].eq(str(row.scope_value))]
        else:
            scope = assessment.loc[assessment["decision_ordinal"].eq(int(row.scope_value))]
        column = (
            f"p_large_remaining_move__{row.model}"
            if row.target == "large_remaining_move"
            else f"p_up_given_large_move__{row.model}"
        )
        expected = metric_record(scope, str(row.target), str(row.model), column)
        maximum_error = max(
            maximum_error,
            *(abs(float(getattr(row, name)) - expected[name]) for name in numeric),
        )
    audit.check("predictive_metrics", maximum_error <= 1e-9, maximum_error)
    bins = pd.read_csv(artifacts / "calibration_bins.csv")
    valid_bins = len(bins) == 880 and set(bins["bin"].unique()) == set(range(1, 11))
    audit.check("ten_bin_calibration", valid_bins, {"rows": len(bins)})


def selection_ledger(assessment: pd.DataFrame) -> pd.DataFrame:
    scored = assessment.copy()
    systems = {
        "predecessor": ("P1", "D1"),
        "behavioural_dimensions": ("P2", "D2"),
        "behavioural_conjunctions": ("P3", "D3"),
    }
    for system, (movement, direction) in systems.items():
        scored[f"score__{system}"] = (
            scored[f"p_large_remaining_move__{movement}"]
            * (2.0 * scored[f"p_up_given_large_move__{direction}"] - 1.0)
            * scored["predicted_remaining_movement_scale_bps"]
        )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SELECTION_SEED)

    def append(system: str, row: pd.Series, side: float) -> None:
        rows.append(
            {
                "system": system,
                "session": str(row["session"]),
                "year_month": str(row["year_month"]),
                "decision_ordinal": int(row["decision_ordinal"]),
                "slate_id": str(row["slate_id"]),
                "selected_symbol": str(row["symbol"]),
                "signed_gross_return_bps": side * float(row["raw_remaining_return_bps"]),
                "signed_cohort_relative_return_bps": side
                * float(row["residual_remaining_return_bps"]),
            }
        )

    for _, slate in scored.groupby("slate_id", sort=True):
        ordered = slate.sort_values("symbol", kind="mergesort")
        for system in systems:
            selected = (
                ordered.assign(absolute=ordered[f"score__{system}"].abs())
                .sort_values(["absolute", "symbol"], ascending=[False, True], kind="mergesort")
                .iloc[0]
            )
            append(system, selected, 1.0 if float(selected[f"score__{system}"]) >= 0.0 else -1.0)
        selected = ordered.sort_values(
            ["p_large_remaining_move__P1", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        append(
            "highest_frozen_movement_probability",
            selected,
            1.0 if float(selected["score__predecessor"]) >= 0.0 else -1.0,
        )
        selected = ordered.sort_values(
            ["open_to_decision_cohort_relative_return_bps", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        append("highest_open_to_decision_relative_momentum", selected, 1.0)
        selected = (
            ordered.assign(absolute=ordered["open_to_decision_cohort_relative_return_bps"].abs())
            .sort_values(["absolute", "symbol"], ascending=[False, True], kind="mergesort")
            .iloc[0]
        )
        append(
            "strongest_reversal",
            selected,
            -1.0 if float(selected["open_to_decision_cohort_relative_return_bps"]) > 0.0 else 1.0,
        )
        selected = ordered.iloc[int(rng.integers(0, len(ordered)))]
        append(
            "random_within_slate",
            selected,
            1.0 if float(selected["score__predecessor"]) >= 0.0 else -1.0,
        )
    return pd.DataFrame(rows)


def economic_metrics(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("pooled", "all", selections)]
    scopes.extend(
        ("checkpoint", str(int(value)), frame)
        for value, frame in selections.groupby("decision_ordinal", sort=True)
    )
    scopes.extend(
        ("month", str(value), frame) for value, frame in selections.groupby("year_month", sort=True)
    )
    for scope_type, scope_value, scope in scopes:
        for system, part in scope.groupby("system", sort=True):
            for friction in (0.0, 10.0, 20.0):
                rows.append(
                    {
                        "record_type": "system",
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "system": system,
                        "friction_bps": friction,
                        "selection_count": len(part),
                        "mean_signed_gross_return_bps": float(
                            (part["signed_gross_return_bps"] - friction).mean()
                        ),
                        "mean_signed_cohort_relative_return_bps": float(
                            (part["signed_cohort_relative_return_bps"] - friction).mean()
                        ),
                    }
                )
        for candidate, baseline in (
            ("behavioural_dimensions", "predecessor"),
            ("behavioural_conjunctions", "predecessor"),
            ("behavioural_conjunctions", "behavioural_dimensions"),
        ):
            paired = scope.loc[scope["system"].isin([candidate, baseline])].pivot(
                index="slate_id",
                columns="system",
                values=["signed_gross_return_bps", "signed_cohort_relative_return_bps"],
            )
            for friction in (0.0, 10.0, 20.0):
                rows.append(
                    {
                        "record_type": "paired_difference",
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "system": f"{candidate}_minus_{baseline}",
                        "friction_bps": friction,
                        "selection_count": len(paired),
                        "mean_signed_gross_return_bps": float(
                            (
                                paired[("signed_gross_return_bps", candidate)]
                                - paired[("signed_gross_return_bps", baseline)]
                            ).mean()
                        ),
                        "mean_signed_cohort_relative_return_bps": float(
                            (
                                paired[("signed_cohort_relative_return_bps", candidate)]
                                - paired[("signed_cohort_relative_return_bps", baseline)]
                            ).mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def improvement(
    frame: pd.DataFrame,
    target: str,
    baseline: str,
    candidate: str,
    metric: str,
) -> float:
    population = (
        frame
        if target == "large_remaining_move"
        else frame.loc[frame["large_remaining_move"].eq(1)]
    )
    labels = population[target].to_numpy(dtype=int)
    base = population[baseline].to_numpy(dtype=float)
    new = population[candidate].to_numpy(dtype=float)
    if metric == "brier":
        return float(np.mean((base - labels) ** 2) - np.mean((new - labels) ** 2))
    return float(log_loss(labels, base, labels=[0, 1]) - log_loss(labels, new, labels=[0, 1]))


BOOTSTRAP_SPECS = (
    (
        "P2_minus_P1_brier_improvement",
        "large_remaining_move",
        "p_large_remaining_move__P1",
        "p_large_remaining_move__P2",
        "brier",
    ),
    (
        "P2_minus_P1_log_loss_improvement",
        "large_remaining_move",
        "p_large_remaining_move__P1",
        "p_large_remaining_move__P2",
        "log_loss",
    ),
    (
        "P3_minus_P2_brier_improvement",
        "large_remaining_move",
        "p_large_remaining_move__P2",
        "p_large_remaining_move__P3",
        "brier",
    ),
    (
        "D2_minus_D1_brier_improvement",
        "up_given_large_move",
        "p_up_given_large_move__D1",
        "p_up_given_large_move__D2",
        "brier",
    ),
    (
        "D2_minus_D1_log_loss_improvement",
        "up_given_large_move",
        "p_up_given_large_move__D1",
        "p_up_given_large_move__D2",
        "log_loss",
    ),
    (
        "D3_minus_D2_brier_improvement",
        "up_given_large_move",
        "p_up_given_large_move__D2",
        "p_up_given_large_move__D3",
        "brier",
    ),
)


def selection_pair(selections: pd.DataFrame, candidate: str, baseline: str) -> float:
    paired = selections.loc[selections["system"].isin([candidate, baseline])].pivot(
        index="slate_id", columns="system", values="signed_cohort_relative_return_bps"
    )
    return float((paired[candidate] - paired[baseline]).mean())


def audit_bootstrap(
    audit: Audit,
    assessment: pd.DataFrame,
    selections: pd.DataFrame,
    artifacts: Path,
) -> None:
    archived = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    archived = archived.loc[archived["record_type"].eq("draw")].sort_values(
        ["metric", "draw"], kind="mergesort"
    )
    unique = np.asarray(sorted(assessment["session"].astype(str).unique()), dtype=object)
    positions = {
        session: np.flatnonzero(assessment["session"].astype(str).to_numpy() == session)
        for session in unique
    }
    by_session = {session: rows for session, rows in selections.groupby("session", sort=True)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for draw in range(200):
        sampled_sessions = tuple(
            str(value) for value in rng.choice(unique, size=len(unique), replace=True)
        )
        sampled = assessment.iloc[
            np.concatenate([positions[session] for session in sampled_sessions])
        ].reset_index(drop=True)
        selected_parts = []
        for occurrence, session in enumerate(sampled_sessions):
            part = by_session[session].copy()
            part["slate_id"] = f"{occurrence:04d}|" + part["slate_id"].astype(str)
            selected_parts.append(part)
        sampled_selection = pd.concat(selected_parts, ignore_index=True)
        for name, target, baseline, candidate, metric in BOOTSTRAP_SPECS:
            rows.append(
                {
                    "draw": draw,
                    "metric": name,
                    "value": improvement(sampled, target, baseline, candidate, metric),
                }
            )
        rows.append(
            {
                "draw": draw,
                "metric": "behavioural_dimensions_minus_predecessor_return_after_20bps",
                "value": selection_pair(sampled_selection, "behavioural_dimensions", "predecessor"),
            }
        )
        rows.append(
            {
                "draw": draw,
                "metric": "conjunction_minus_behavioural_dimensions_return_after_20bps",
                "value": selection_pair(
                    sampled_selection,
                    "behavioural_conjunctions",
                    "behavioural_dimensions",
                ),
            }
        )
    expected = pd.DataFrame(rows).sort_values(["metric", "draw"], kind="mergesort")
    expected_values = expected["value"].to_numpy(dtype=float)
    archived_values = archived["value"].to_numpy(dtype=float)
    absolute_error = np.abs(expected_values - archived_values)
    worst = int(np.nanargmax(absolute_error))
    audit.check(
        "session_block_bootstrap",
        bool(np.allclose(expected_values, archived_values, rtol=0.0, atol=1e-10)),
        {
            "maximum_absolute_error": float(absolute_error[worst]),
            "tolerance": 1e-10,
            "worst_draw": int(expected.iloc[worst]["draw"]),
            "worst_metric": str(expected.iloc[worst]["metric"]),
            "expected_value": float(expected_values[worst]),
            "archived_value": float(archived_values[worst]),
        },
    )


def permute_bundle(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    output = frame.copy()
    for indices in output.groupby("slate_id", sort=True).groups.values():
        index = list(indices)
        bundle = frame.loc[index, list(DIMENSIONS)].to_numpy(dtype=float)
        output.loc[index, list(DIMENSIONS)] = bundle[rng.permutation(len(index))]
    return output


def recompute_conjunctions(frame: pd.DataFrame, bounds: Mapping[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    raw = {
        "active_conviction": output["arousal"] * output["conviction"],
        "active_frustration": output["arousal"] * output["frustration"],
        "pressurised_tension": output["tension"] * output["pressure_magnitude"],
        "pressurised_exhaustion": output["exhaustion_magnitude"] * output["pressure_magnitude"],
        "independent_pressure": output["independence"] * output["signed_pressure"],
    }
    for checkpoint in (6, 12):
        mask = output["decision_ordinal"].eq(checkpoint)
        for name, values in raw.items():
            frozen = bounds[str(checkpoint)][name]
            output.loc[mask, name] = values.loc[mask].clip(frozen["q01"], frozen["q99"])
    return output


def paired_null_economic(frame: pd.DataFrame) -> float:
    rows: list[dict[str, Any]] = []
    for _, slate in frame.groupby("slate_id", sort=True):
        ordered = slate.sort_values("symbol", kind="mergesort")
        for system, movement, direction in (
            ("predecessor", "P1", "D1"),
            ("behavioural_dimensions", "P2", "D2"),
        ):
            score = (
                ordered[f"p_large_remaining_move__{movement}"]
                * (2.0 * ordered[f"p_up_given_large_move__{direction}"] - 1.0)
                * ordered["predicted_remaining_movement_scale_bps"]
            )
            selected = (
                ordered.assign(absolute=score.abs(), score=score)
                .sort_values(["absolute", "symbol"], ascending=[False, True], kind="mergesort")
                .iloc[0]
            )
            side = 1.0 if float(selected["score"]) >= 0.0 else -1.0
            rows.append(
                {
                    "slate_id": str(selected["slate_id"]),
                    "system": system,
                    "return": side * float(selected["residual_remaining_return_bps"]) - 20.0,
                }
            )
    paired = pd.DataFrame(rows).pivot(index="slate_id", columns="system", values="return")
    return float((paired["behavioural_dimensions"] - paired["predecessor"]).mean())


def audit_null(
    audit: Audit,
    full: pd.DataFrame,
    assessment: pd.DataFrame,
    artifacts: Path,
) -> None:
    archived = pd.read_csv(artifacts / "null_metrics.csv")
    archived = archived.loc[archived["record_type"].eq("draw")].sort_values(
        ["metric", "draw"], kind="mergesort"
    )
    configs = json.loads((artifacts / "model_configurations.json").read_text(encoding="utf-8"))
    bounds = json.loads(
        (artifacts / "behavioural_dimension_manifest.json").read_text(encoding="utf-8")
    )["conjunction_clip_bounds"]
    development = full.loc[full["year"].eq(2024)].reset_index(drop=True)
    assessment_base = assessment.reset_index(drop=True)
    rng = np.random.default_rng(NULL_SEED)
    rows: list[dict[str, Any]] = []
    for draw in range(50):
        dev = recompute_conjunctions(permute_bundle(development, rng), bounds)
        assess = recompute_conjunctions(permute_bundle(assessment_base, rng), bounds)
        for target_name, target, model_names in (
            ("movement", "large_remaining_move", ("P2", "P3")),
            ("direction", "up_given_large_move", ("D2", "D3")),
        ):
            training = (
                dev
                if target_name == "movement"
                else dev.loc[dev["large_remaining_move"].eq(1)].reset_index(drop=True)
            )
            for model_name in model_names:
                fitted = refit_model(
                    training,
                    target,
                    configs["features"][target_name][model_name],
                )
                model = {
                    "feature_names": configs["features"][target_name][model_name],
                    **fitted,
                }
                prefix = (
                    "p_large_remaining_move__"
                    if target_name == "movement"
                    else "p_up_given_large_move__"
                )
                assess[f"{prefix}{model_name}"] = manual_probability(model, assess)
        draw_values = {
            name: improvement(assess, target, baseline, candidate, metric)
            for name, target, baseline, candidate, metric in BOOTSTRAP_SPECS
        }
        draw_values["behavioural_system_minus_predecessor_delayed_return_after_20bps"] = (
            paired_null_economic(assess)
        )
        rows.extend(
            {"draw": draw, "metric": name, "null_value": value}
            for name, value in draw_values.items()
        )
    expected = pd.DataFrame(rows).sort_values(["metric", "draw"], kind="mergesort")
    audit.close("behavioural_bundle_null", expected["null_value"], archived["null_value"], 1e-10)


def audit_increment_gate(
    audit: Audit,
    *,
    recorded: Mapping[str, Any],
    pooled: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    target: str,
    baseline: str,
    candidate: str,
    prefix: str,
    concentration_passes: bool,
    require_log_null: bool,
    adversity_threshold: float,
) -> bool:
    pooled = pooled.set_index("model")
    brier = float(pooled.loc[baseline, "brier_score"] - pooled.loc[candidate, "brier_score"])
    log_increment = float(pooled.loc[baseline, "log_loss"] - pooled.loc[candidate, "log_loss"])
    auc_change = float(pooled.loc[candidate, "auc"] - pooled.loc[baseline, "auc"])
    monthly_target = monthly.loc[monthly["target"].eq(target)].copy()
    monthly_target["scope_value"] = monthly_target["scope_value"].astype(str)
    monthly_pivot = monthly_target.pivot(index="scope_value", columns="model", values="brier_score")
    positive_months = int((monthly_pivot[baseline] - monthly_pivot[candidate] > 0.0).sum())
    checkpoint_target = checkpoint.loc[checkpoint["target"].eq(target)].copy()
    checkpoint_target["scope_value"] = checkpoint_target["scope_value"].astype(str)
    checkpoint_pivot = checkpoint_target.pivot(
        index="scope_value", columns="model", values="brier_score"
    )
    checkpoint_improvements = {
        str(value): float(
            checkpoint_pivot.loc[str(value), baseline] - checkpoint_pivot.loc[str(value), candidate]
        )
        for value in (6, 12)
    }
    brier_bootstrap = bootstrap.loc[
        bootstrap["record_type"].eq("summary")
        & bootstrap["metric"].eq(f"{prefix}_brier_improvement")
    ].iloc[0]
    log_bootstrap = (
        bootstrap.loc[
            bootstrap["record_type"].eq("summary")
            & bootstrap["metric"].eq(f"{prefix}_log_loss_improvement")
        ].iloc[0]
        if require_log_null
        else None
    )
    brier_null = null.loc[
        null["record_type"].eq("summary") & null["metric"].eq(f"{prefix}_brier_improvement")
    ].iloc[0]
    log_null = (
        null.loc[
            null["record_type"].eq("summary") & null["metric"].eq(f"{prefix}_log_loss_improvement")
        ].iloc[0]
        if require_log_null
        else None
    )
    expected_gates = {
        "brier_improvement_positive": brier > 0.0,
        "log_loss_improvement_positive": log_increment > 0.0,
        "auc_not_reduced": auc_change >= 0.0,
        "bootstrap_90_lower_brier_non_negative": float(brier_bootstrap["interval_90_lower"]) >= 0.0,
        "bootstrap_90_lower_log_loss_non_negative": (
            log_bootstrap is not None and float(log_bootstrap["interval_90_lower"]) >= 0.0
        )
        if require_log_null
        else True,
        "positive_brier_months_at_least_five": positive_months >= 5,
        "neither_checkpoint_materially_adverse": min(checkpoint_improvements.values())
        >= adversity_threshold,
        "real_brier_increment_exceeds_null_q90": float(brier_null["real_value"])
        > float(brier_null["null_q90"]),
        "real_log_loss_increment_exceeds_null_q90": (
            log_null is not None and float(log_null["real_value"]) > float(log_null["null_q90"])
        )
        if require_log_null
        else True,
        "concentration_gates_pass": concentration_passes,
    }
    label = f"{candidate}_minus_{baseline}"
    audit.close(f"decision_{label}_brier", brier, recorded["brier_improvement"])
    audit.close(f"decision_{label}_log_loss", log_increment, recorded["log_loss_improvement"])
    audit.close(f"decision_{label}_auc", auc_change, recorded["auc_change"])
    audit.check(f"decision_{label}_gates", dict(recorded["gates"]) == expected_gates)
    expected_pass = all(expected_gates.values())
    audit.check(f"decision_{label}_pass", bool(recorded["passes"]) == expected_pass)
    return expected_pass


def audit_economic_concentration_decision(
    audit: Audit,
    full: pd.DataFrame,
    assessment: pd.DataFrame,
    artifacts: Path,
) -> pd.DataFrame:
    selections = selection_ledger(assessment)
    expected = economic_metrics(selections).sort_values(
        ["record_type", "scope_type", "scope_value", "system", "friction_bps"],
        kind="mergesort",
    )
    archived = pd.read_csv(artifacts / "economic_reference_metrics.csv").sort_values(
        ["record_type", "scope_type", "scope_value", "system", "friction_bps"],
        kind="mergesort",
    )
    audit.check("economic_reference_shape", expected.shape == archived.shape)
    for column in (
        "friction_bps",
        "selection_count",
        "mean_signed_gross_return_bps",
        "mean_signed_cohort_relative_return_bps",
    ):
        audit.close(f"economic_reference_{column}", expected[column], archived[column], 1e-9)
    row_share = float(assessment.groupby("symbol").size().max() / len(assessment))
    primary = selections.loc[
        selections["system"].isin(
            ["predecessor", "behavioural_dimensions", "behavioural_conjunctions"]
        )
    ]
    selection_share = max(
        float(part.groupby("selected_symbol").size().max() / len(part))
        for _, part in primary.groupby("system", sort=True)
    )
    decision = json.loads((artifacts / "decision.json").read_text(encoding="utf-8"))
    contract = json.loads((artifacts / "contract.json").read_text(encoding="utf-8"))
    model_config = json.loads((artifacts / "model_configurations.json").read_text(encoding="utf-8"))
    gate_constants = contract["decision_gate_constants"]
    audit.check(
        "decision_gate_constants",
        decision["gate_constants"] == gate_constants
        and model_config["decision_gate_constants"] == gate_constants,
    )
    audit.close(
        "decision_row_concentration",
        row_share,
        decision["support"]["maximum_stock_decision_row_share"],
    )
    audit.close(
        "economic_selection_concentration",
        selection_share,
        decision["support"]["maximum_stock_economic_selection_share"],
    )
    large_by_checkpoint = {
        str(int(checkpoint)): int(rows["large_remaining_move"].sum())
        for checkpoint, rows in assessment.groupby("decision_ordinal", sort=True)
    }
    development = full.loc[full["year"].eq(2024)]
    source_manifest = json.loads((artifacts / "source_manifest.json").read_text(encoding="utf-8"))
    expected_support: dict[str, Any] = {
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "actual_large_moves": int(assessment["large_remaining_move"].sum()),
        "actual_large_moves_by_checkpoint": large_by_checkpoint,
        "represented_months": int(assessment["year_month"].nunique()),
        "maximum_stock_decision_row_share": row_share,
        "maximum_stock_economic_selection_share": selection_share,
        "predecessor_total_rows": 15_617,
        "eligible_total_rows": len(full),
        "development_rows": len(development),
        "development_sessions": int(development["session"].nunique()),
        "development_stocks": int(development["symbol"].nunique()),
        "development_large_moves": int(development["large_remaining_move"].sum()),
        "causal_feature_exclusion_count": int(source_manifest["causal_feature_exclusion_count"]),
    }
    expected_support["movement_support_passes"] = bool(
        expected_support["assessment_rows"] >= 3000
        and expected_support["assessment_sessions"] >= 100
        and expected_support["assessment_stocks"] >= 15
        and expected_support["actual_large_moves"] >= 600
        and expected_support["represented_months"] >= 6
        and row_share <= 0.10
        and selection_share <= 0.20
    )
    expected_support["direction_support_passes"] = bool(
        expected_support["movement_support_passes"]
        and all(value >= 250 for value in large_by_checkpoint.values())
    )
    for key, value in expected_support.items():
        if isinstance(value, float):
            audit.close(f"support_{key}", value, decision["support"][key])
        else:
            audit.check(f"support_{key}", decision["support"][key] == value)

    movement = pd.read_csv(artifacts / "movement_metrics.csv")
    direction = pd.read_csv(artifacts / "direction_metrics.csv")
    monthly = pd.read_csv(artifacts / "monthly_metrics.csv")
    checkpoint = pd.read_csv(artifacts / "checkpoint_metrics.csv")
    bootstrap = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    null = pd.read_csv(artifacts / "null_metrics.csv")
    concentration_passes = row_share <= 0.10 and selection_share <= 0.20
    adversity = float(gate_constants["checkpoint_material_adversity_brier"])
    movement_pass = audit_increment_gate(
        audit,
        recorded=decision["movement_increment"],
        pooled=movement,
        monthly=monthly,
        checkpoint=checkpoint,
        bootstrap=bootstrap,
        null=null,
        target="large_remaining_move",
        baseline="P1",
        candidate="P2",
        prefix="P2_minus_P1",
        concentration_passes=concentration_passes,
        require_log_null=True,
        adversity_threshold=adversity,
    )
    direction_pass = audit_increment_gate(
        audit,
        recorded=decision["direction_increment"],
        pooled=direction,
        monthly=monthly,
        checkpoint=checkpoint,
        bootstrap=bootstrap,
        null=null,
        target="up_given_large_move",
        baseline="D1",
        candidate="D2",
        prefix="D2_minus_D1",
        concentration_passes=concentration_passes,
        require_log_null=True,
        adversity_threshold=adversity,
    )
    movement_conjunction_pass = audit_increment_gate(
        audit,
        recorded=decision["movement_conjunction_increment"],
        pooled=movement,
        monthly=monthly,
        checkpoint=checkpoint,
        bootstrap=bootstrap,
        null=null,
        target="large_remaining_move",
        baseline="P2",
        candidate="P3",
        prefix="P3_minus_P2",
        concentration_passes=concentration_passes,
        require_log_null=False,
        adversity_threshold=adversity,
    )
    direction_conjunction_pass = audit_increment_gate(
        audit,
        recorded=decision["direction_conjunction_increment"],
        pooled=direction,
        monthly=monthly,
        checkpoint=checkpoint,
        bootstrap=bootstrap,
        null=null,
        target="up_given_large_move",
        baseline="D2",
        candidate="D3",
        prefix="D3_minus_D2",
        concentration_passes=concentration_passes,
        require_log_null=False,
        adversity_threshold=adversity,
    )
    conjunction_pass = movement_conjunction_pass or direction_conjunction_pass
    audit.check(
        "decision_conjunction_pass",
        bool(decision["conjunction_increment_passes"]) == conjunction_pass,
    )
    census = pd.read_csv(artifacts / "behavioural_state_census.csv")
    summaries = census.loc[census["record_type"].eq("label_summary")]
    overall = float(assessment["large_remaining_move"].mean())
    descriptive = bool(
        (
            summaries["row_count"].ge(100)
            & summaries["represented_months"].ge(6)
            & summaries["stock_count"].ge(15)
            & summaries["movement_rate"].sub(overall).abs().ge(0.01)
        ).any()
    )
    audit.check(
        "descriptive_difference_gate",
        bool(decision["descriptive_differences_gate"]) == descriptive,
    )
    expected_decision = (
        "behavioural_dimensions_add_movement_and_direction"
        if movement_pass and direction_pass
        else "behavioural_dimensions_add_movement_only"
        if movement_pass
        else "behavioural_dimensions_add_direction_only"
        if direction_pass
        else "behavioural_conjunctions_only"
        if conjunction_pass
        else "behavioural_descriptions_only_no_predictive_increment"
        if descriptive
        else "no_behavioural_state_increment"
    )
    audit.check("decision_logic", decision["decision"] == expected_decision, expected_decision)
    return selections


def audit_population_and_outcomes(
    audit: Audit,
    compact: pd.DataFrame,
    assessment: pd.DataFrame,
) -> None:
    timestamps = pd.to_datetime(compact["feature_available_timestamp_utc"], utc=True)
    local = timestamps.dt.tz_convert("America/New_York")
    checkpoint_ok = (
        compact["decision_ordinal"].isin([6, 12]).all()
        and compact.loc[compact["decision_ordinal"].eq(6), "entry_bar_ordinal"].eq(7).all()
        and compact.loc[compact["decision_ordinal"].eq(12), "entry_bar_ordinal"].eq(13).all()
        and compact["terminal_bar_ordinal"].eq(77).all()
        and local.loc[compact["decision_ordinal"].eq(6)].dt.strftime("%H:%M").eq("10:00").all()
        and local.loc[compact["decision_ordinal"].eq(12)].dt.strftime("%H:%M").eq("10:30").all()
    )
    audit.check("decision_checkpoints_and_t_plus_2", checkpoint_ok)
    audit.check(
        "development_and_assessment_partition",
        set(compact["year"].unique()) == {2024, 2025}
        and compact.loc[compact["year"].eq(2024), "session"].max() <= "2024-12-31"
        and assessment["session"].min() >= "2025-01-01"
        and assessment["session"].max() <= "2025-08-22"
        and len(compact) <= 20_000,
    )
    raw_return = 10_000.0 * (
        compact["terminal_close"].to_numpy(dtype=float)
        / compact["delayed_entry_open"].to_numpy(dtype=float)
        - 1.0
    )
    audit.close("delayed_remaining_return", raw_return, compact["raw_remaining_return_bps"], 1e-9)
    predecessor = pd.read_parquet(PREDECESSOR / "opening_decision_panel.parquet")
    keys = ["symbol", "session", "decision_ordinal"]
    outcome_columns = [
        "cohort_median_return_minus_i_bps",
        "residual_remaining_return_bps",
        "large_remaining_move",
        "up_given_large_move",
    ]
    frozen = compact[keys + outcome_columns].merge(
        predecessor[keys + outcome_columns],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_frozen"),
    )
    audit.close(
        "cohort_relative_outcome",
        frozen["residual_remaining_return_bps"],
        frozen["residual_remaining_return_bps_frozen"],
        1e-12,
    )
    audit.check(
        "outcome_labels_unchanged",
        np.array_equal(frozen["large_remaining_move"], frozen["large_remaining_move_frozen"])
        and np.array_equal(frozen["up_given_large_move"], frozen["up_given_large_move_frozen"]),
    )
    slate_sizes = compact.groupby("slate_id").size()
    audit.check("minimum_parent_slate", int(slate_sizes.min()) >= 15, int(slate_sizes.min()))


def run_audit(artifacts: Path, provider_root: Path) -> int:
    audit = Audit()
    required = (
        "contract.json",
        "source_manifest.json",
        "input_artifact_hashes.json",
        "protected_boundary_audit.json",
        "predecessor_reconstruction.json",
        "raw_component_manifest.json",
        "behavioural_component_scaling.json",
        "behavioural_dimension_manifest.json",
        "behavioural_label_thresholds.json",
        "forbidden_feature_audit.json",
        "compact_decision_panel.parquet",
        "behavioural_component_ledger.parquet",
        "behavioural_dimension_ledger.parquet",
        "behavioural_state_census.csv",
        "model_configurations.json",
        "model_coefficients.json",
        "assessment_predictions.parquet",
        "movement_metrics.csv",
        "direction_metrics.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "calibration_bins.csv",
        "dimension_diagnostics.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
        "economic_reference_metrics.csv",
        "concentration_metrics.csv",
        "decision.json",
        "report.md",
    )
    audit.check("required_artifacts", all((artifacts / name).is_file() for name in required))
    source_frames = audit_inputs_and_sources(audit, artifacts, provider_root)
    audit_predecessor(audit, artifacts)
    compact = pd.read_parquet(artifacts / "compact_decision_panel.parquet")
    ledger = pd.read_parquet(artifacts / "behavioural_component_ledger.parquet")
    dimension_ledger = pd.read_parquet(artifacts / "behavioural_dimension_ledger.parquet")
    predictions = pd.read_parquet(artifacts / "assessment_predictions.parquet")
    assessment = compact.loc[compact["year"].eq(2025)].merge(
        predictions,
        on=["symbol", "session", "decision_ordinal", "slate_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    for column in predictions.columns:
        if (
            column not in {"symbol", "session", "decision_ordinal", "slate_id"}
            and f"{column}_prediction" in assessment
        ):
            assessment[column] = assessment[f"{column}_prediction"]
    audit_expected_population(audit, compact, source_frames, artifacts)
    audit_population_and_outcomes(audit, compact, assessment)
    raw = recompute_raw_components(audit, compact, ledger, source_frames)
    raw["year"] = ledger["year"].to_numpy()
    raw["session"] = ledger["session"].to_numpy()
    raw["aligned_progress_acceleration"] = ledger["aligned_progress_acceleration"].to_numpy(
        dtype=float
    )
    raw["directional_rejection"] = ledger["directional_rejection"].to_numpy(dtype=float)
    scaling = json.loads(
        (artifacts / "behavioural_component_scaling.json").read_text(encoding="utf-8")
    )
    dimension_manifest = json.loads(
        (artifacts / "behavioural_dimension_manifest.json").read_text(encoding="utf-8")
    )
    labels = json.loads(
        (artifacts / "behavioural_label_thresholds.json").read_text(encoding="utf-8")
    )
    reconstructed = audit_scaling_dimensions_and_labels(
        audit, raw, dimension_ledger, scaling, dimension_manifest, labels
    )
    full = compact.copy()
    for column in [*DIMENSIONS, *CONJUNCTIONS]:
        full[column] = reconstructed[column].to_numpy(dtype=float)
    audit_models(audit, full, assessment, artifacts)
    audit_metrics(audit, assessment, artifacts)
    selections = audit_economic_concentration_decision(audit, full, assessment, artifacts)
    audit_bootstrap(audit, assessment, selections, artifacts)
    audit_null(audit, full, assessment, artifacts)
    census = pd.read_csv(artifacts / "behavioural_state_census.csv")
    summaries = census.loc[census["record_type"].eq("label_summary")]
    audit.check(
        "behavioural_state_census",
        set(summaries["label"]) == set(LABELS)
        and int(summaries["row_count"].sum())
        == int(assessment[[f"label__{label}" for label in LABELS]].astype(bool).sum().sum()),
    )
    diagnostics = pd.read_csv(artifacts / "dimension_diagnostics.csv")
    audit.check(
        "dimension_diagnostics",
        {
            "standardised_model_coefficient",
            "assessment_dimension_correlation",
            "condition_number",
            "variance_inflation_factor",
            "development_frozen_dimension_quintile",
            "assessment_group_permutation_importance",
        }.issubset(set(diagnostics["diagnostic_type"])),
    )
    payload = {
        **SAFETY_FLAGS,
        "auditor_imports_runner": False,
        "audited_rows": len(compact),
        "audited_assessment_rows": len(assessment),
        "checks": audit.checks,
        "details": audit.details,
        "failures": audit.failures,
        "passed": not audit.failures,
    }
    (artifacts / "independent_audit.json").write_text(canonical_json(payload), encoding="utf-8")
    if audit.failures:
        print("independent audit failures: " + ", ".join(audit.failures), file=sys.stderr)
        return 1
    print(canonical_json({"passed": True, "checks": len(audit.checks)}), end="")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_audit(args.artifacts.resolve(), args.provider_root.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
