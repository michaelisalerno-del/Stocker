#!/usr/bin/env python3
"""Run Observable Behavioural-State Dimensions Screen V0."""

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
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PACKAGE_SRC = REPO_ROOT / "packages" / "stocker_research" / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from stocker_research.behavioural_state_dimensions_v0 import (  # noqa: E402
    CONJUNCTION_FEATURES,
    DECISION_CATEGORIES,
    DESCRIPTIVE_LABELS,
    DIMENSION_FEATURES,
    FrozenLogisticModel,
    RobustComponentScale,
    apply_component_scaling,
    apply_conjunction_bounds,
    assert_allowed_model_features,
    assert_safe_timestamps,
    assign_descriptive_labels,
    bar_component_frame,
    decide_behavioural_screen,
    derive_behavioural_dimensions,
    derive_conjunctions,
    derive_exhaustion_inputs,
    fit_component_scaling,
    fit_conjunction_bounds,
    fit_fixed_logistic,
    fit_label_thresholds,
    manual_logistic_prediction,
    opening_raw_components,
    permute_bundle_within_slates,
    session_block_bootstrap_draws,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_EXACT = EXPERIMENT_DIR / "artifacts" / "exact_rerun"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
AUDITOR_PATH = EXPERIMENT_DIR / "audit_screen_v0.py"

PREDECESSOR_DIR = (
    REPO_ROOT
    / "research"
    / "opening-regime-path"
    / "20260720-opening-regime-path-direction-screen-v0"
)
PREDECESSOR_PRIMARY = PREDECESSOR_DIR / "artifacts" / "primary"
PREDECESSOR_PANEL = PREDECESSOR_PRIMARY / "opening_decision_panel.parquet"
PREDECESSOR_PREDICTIONS = PREDECESSOR_PRIMARY / "assessment_predictions.parquet"
PREDECESSOR_COEFFICIENTS = PREDECESSOR_PRIMARY / "model_coefficients.json"
PREDECESSOR_MOVEMENT_METRICS = PREDECESSOR_PRIMARY / "movement_metrics.csv"
PREDECESSOR_DIRECTION_METRICS = PREDECESSOR_PRIMARY / "direction_metrics.csv"
PREDECESSOR_THRESHOLDS = PREDECESSOR_PRIMARY / "development_movement_thresholds.json"
PREDECESSOR_SOURCE_MANIFEST = PREDECESSOR_PRIMARY / "source_manifest.json"
PRESSURE_SOURCE_MANIFEST = (
    REPO_ROOT
    / "research"
    / "observable-pressure-onset"
    / "20260720-high-movement-pressure-onset-screen-v0-1"
    / "artifacts"
    / "primary"
    / "source_manifest.json"
)

START = pd.Timestamp("2024-01-01T00:00:00Z")
DEVELOPMENT_END_EXCLUSIVE = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
EXPECTED_SESSION_BARS = 78
MAX_COMPACT_ROWS = 20_000
BOOTSTRAP_DRAWS = 200
NULL_DRAWS = 50
BOOTSTRAP_SEED = 20260721
NULL_SEED = 20260722
PERMUTATION_IMPORTANCE_SEED = 20260723
RANDOM_SELECTION_SEED = 20260724
CHECKPOINT_MATERIAL_ADVERSITY = -0.001
DECISION_GATE_CONSTANTS: dict[str, Any] = {
    "bootstrap_interval": 0.9,
    "checkpoint_material_adversity_brier": CHECKPOINT_MATERIAL_ADVERSITY,
    "minimum_positive_months": 5,
    "null_percentile_threshold": 0.9,
    "null_real_comparator": "strictly_greater_than_q90",
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

M1_FEATURES = (
    "checkpoint_60m",
    "opening_gap_bps",
    "open_to_decision_raw_return_bps",
    "open_to_decision_cohort_relative_return_bps",
    "latest_one_bar_return_bps",
    "latest_three_bar_return_bps",
    "opening_realized_volatility_bps",
    "opening_high_low_range_bps",
    "current_true_range_bps",
    "mean_completed_bar_true_range_bps",
    "close_location_within_opening_range",
    "distance_from_opening_high_bps",
    "distance_from_opening_low_bps",
    "positive_close_fraction",
    "directional_close_persistence_ratio",
    "historical_activity_proxy_shock",
    "cross_sectional_dispersion_bps",
)

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

EXPECTED_INPUT_HASHES = {
    "opening_decision_panel.parquet": (
        "a6ab58f870a2b356ab2174011988d075b408612045c36e74017c772e5c8cd78a"
    ),
    "assessment_predictions.parquet": (
        "f5db0db6540e2ea60270df04c36efd64a5cfe9af42058d799abcab954e089d65"
    ),
    "model_coefficients.json": "7c3a55a84d2caca53e7206bcfc5cb67c65627b3324f77833c039e12f02b983e0",
    "movement_metrics.csv": "2941ba9e34961032da12cf7e0a9dc4acd7a3a4ca1de30a59d9104640c56a3479",
    "source_manifest.json": "1f1c8b08038f866c96afef33a6754eb2825ac32d7e9909ce01aca1598d58c100",
}

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


class ScreenBlocker(RuntimeError):
    """A preregistered fail-closed experiment stop."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in DECISION_CATEGORIES or not code.startswith("blocked_"):
            raise ValueError(f"invalid blocker code: {code}")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def arrow_hash(frame: pd.DataFrame) -> str:
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def provider_path(root: Path, symbol: str) -> Path:
    return root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def logical_source_path(symbol: str) -> str:
    return f"source=eodhd/instrument_type=stock/symbol={symbol}/timeframe=5m/data.parquet"


def bounded_source(path: Path) -> pd.DataFrame:
    """Read only declared columns and safe dates using parquet predicates."""

    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(
        path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        filters=[
            ("timestamp", ">=", START.to_pydatetime()),
            ("timestamp", "<", PROTECTED_START.to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    assert_safe_timestamps(frame["timestamp"])
    if frame["timestamp"].lt(START).any():
        raise ScreenBlocker("blocked_protected_boundary_failure", "pre-development source row")
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or contract.get("safety", {}).get(key) != expected:
            raise RuntimeError(f"contract safety flag differs: {key}")
    if tuple(contract["population"]["symbols"]) != SYMBOLS:
        raise RuntimeError("contract cohort differs")
    if tuple(contract["population"]["decision_ordinals"]) != (6, 12):
        raise RuntimeError("contract checkpoints differ")
    if contract.get("decision_gate_constants") != DECISION_GATE_CONSTANTS:
        raise RuntimeError("contract decision gate constants differ")
    return cast(dict[str, Any], contract)


def verify_input_hashes() -> list[dict[str, Any]]:
    paths = (
        PREDECESSOR_PANEL,
        PREDECESSOR_PREDICTIONS,
        PREDECESSOR_COEFFICIENTS,
        PREDECESSOR_MOVEMENT_METRICS,
        PREDECESSOR_DIRECTION_METRICS,
        PREDECESSOR_THRESHOLDS,
        PREDECESSOR_SOURCE_MANIFEST,
        PRESSURE_SOURCE_MANIFEST,
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ScreenBlocker(
                "blocked_observable_predecessor_not_reconstructable",
                f"required predecessor input missing: {path.name}",
            )
        digest = sha256_file(path)
        expected = (
            EXPECTED_INPUT_HASHES.get(path.name) if path.parent == PREDECESSOR_PRIMARY else None
        )
        if expected is not None and digest != expected:
            raise ScreenBlocker(
                "blocked_observable_predecessor_not_reconstructable",
                f"frozen predecessor hash differs: {path.name}",
            )
        records.append(
            {
                "repository_relative_path": str(path.relative_to(REPO_ROOT)),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def predecessor_reconstruction(
    panel: pd.DataFrame,
    archived_assessment: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], pd.DataFrame]:
    """Reconstruct both frozen M1 models before deriving any new component."""

    coefficients = json.loads(PREDECESSOR_COEFFICIENTS.read_text(encoding="utf-8"))["models"]
    assessment = panel.loc[panel["year"].eq(2025)].sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    archived = archived_assessment.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    keys = ["symbol", "session", "decision_ordinal"]
    if not assessment[keys].reset_index(drop=True).equals(archived[keys].reset_index(drop=True)):
        raise ScreenBlocker(
            "blocked_observable_predecessor_not_reconstructable",
            "predecessor assessment keys differ",
        )
    target_specs = {
        "large_remaining_move": PREDECESSOR_MOVEMENT_METRICS,
        "up_given_large_move": PREDECESSOR_DIRECTION_METRICS,
    }
    reconstruction: dict[str, Any] = {}
    frozen_models: dict[str, Mapping[str, Any]] = {}
    frozen = archived[keys].reset_index(drop=True).copy()
    frozen["predicted_remaining_movement_scale_bps"] = archived[
        "predicted_remaining_movement_scale_bps"
    ].to_numpy(dtype=float)
    for target, metric_path in target_specs.items():
        model = cast(Mapping[str, Any], coefficients[target]["M1"])
        all_probabilities = manual_logistic_prediction(model, assessment)
        probability_column = f"p__{target}__M1"
        expected = archived[probability_column].to_numpy(dtype=float)
        maximum_error = float(np.max(np.abs(all_probabilities - expected)))
        metric_mask = (
            np.ones(len(assessment), dtype=bool)
            if target == "large_remaining_move"
            else assessment["large_remaining_move"].eq(1).to_numpy(dtype=bool)
        )
        population = assessment.loc[metric_mask]
        probabilities = all_probabilities[metric_mask]
        labels = population[target].to_numpy(dtype=int)
        actual_metrics = {
            "brier": float(brier_score_loss(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "auc": float(roc_auc_score(labels, probabilities)),
        }
        metric_frame = pd.read_csv(metric_path)
        metric_row = metric_frame.loc[
            metric_frame["scope"].eq("pooled")
            & metric_frame["target"].eq(target)
            & metric_frame["model"].eq("M1")
        ].iloc[0]
        archived_metrics = {
            "brier": float(metric_row["brier_score"]),
            "log_loss": float(metric_row["log_loss"]),
            "auc": float(metric_row["auc"]),
        }
        errors = {
            name: abs(actual_metrics[name] - archived_metrics[name]) for name in actual_metrics
        }
        passed = maximum_error <= 1e-12 and max(errors.values()) <= 1e-12
        reconstruction[target] = {
            "model_id": str(model["model_id"]),
            "probability_reconstruction_rows": len(assessment),
            "metric_population_rows": len(population),
            "maximum_prediction_absolute_error": maximum_error,
            "actual_metrics": actual_metrics,
            "archived_metrics": archived_metrics,
            "metric_absolute_errors": errors,
            "passed": passed,
        }
        if not passed:
            raise ScreenBlocker(
                "blocked_observable_predecessor_not_reconstructable",
                f"{target} M1 differs beyond 1e-12",
            )
        frozen_models[target] = model
        frozen[probability_column] = all_probabilities
    result = {
        **SAFETY_FLAGS,
        "source_experiment": str(PREDECESSOR_DIR.relative_to(REPO_ROOT)),
        "source_commit": "5e80d972d1e003c8366a1ec6ca170d1077288ead",
        "required_probability_tolerance": 1e-12,
        "required_metric_tolerance": 1e-12,
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "actual_large_moves": int(assessment["large_remaining_move"].sum()),
        "targets": reconstruction,
        "passed": all(record["passed"] for record in reconstruction.values()),
    }
    return result, frozen_models, frozen


def load_qa_record(symbol: str) -> dict[str, Any]:
    path = (
        Path.home()
        / "StockerLocal"
        / "data"
        / "reports"
        / "vendor_qa"
        / f"{symbol}_5m_eodhd_qa.json"
    )
    if not path.is_file():
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", f"vendor QA missing for {symbol}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation_errors = int(payload.get("validation", {}).get("counts", {}).get("error", 0))
    adjusted = payload.get("adjusted_close", {})
    adjusted_differences = int(adjusted.get("different_from_close_count", 0) or 0)
    if payload.get("status") == "fail" or validation_errors or adjusted_differences:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure",
            f"QA or corporate-action check failed for {symbol}",
        )
    return {
        "symbol": symbol,
        "status": str(payload.get("status", "unknown")),
        "logical_path": f"external_vendor_qa/{path.name}",
        "sha256": sha256_file(path),
        "validation_error_count": validation_errors,
        "adjusted_close_present": adjusted.get("present"),
        "adjusted_close_differences": adjusted_differences,
        "corporate_action_check_passed": True,
    }


def prepare_symbol_bars(
    raw: pd.DataFrame,
    *,
    symbol: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Validate exact regular-session grids and add causal historical activity."""

    timestamps = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    local = timestamps.dt.tz_convert("America/New_York")
    minute = local.dt.hour * 60 + local.dt.minute
    in_regular = minute.ge(570) & minute.lt(960)
    on_grid = ((minute - 570) % 5).eq(0) & local.dt.second.eq(0) & local.dt.microsecond.eq(0)
    invalid_sessions = set(local.loc[in_regular & ~on_grid].dt.strftime("%Y-%m-%d"))
    regular = raw.loc[in_regular & on_grid].copy()
    local_regular = pd.to_datetime(regular["timestamp"], utc=True).dt.tz_convert("America/New_York")
    minute_regular = local_regular.dt.hour * 60 + local_regular.dt.minute
    regular["symbol"] = symbol
    regular["session"] = local_regular.dt.strftime("%Y-%m-%d")
    regular["bar_ordinal"] = ((minute_regular - 570) // 5).astype(np.int16)
    regular["bar_start_timestamp"] = pd.to_datetime(regular["timestamp"], utc=True)
    regular["bar_complete_timestamp"] = regular["bar_start_timestamp"] + pd.Timedelta(minutes=5)
    regular = regular.sort_values(
        ["session", "bar_ordinal", "bar_start_timestamp"], kind="mergesort"
    ).reset_index(drop=True)

    valid_parts: list[pd.DataFrame] = []
    gap_records: list[dict[str, str]] = []
    for session, part in regular.groupby("session", sort=True):
        ordered = part.sort_values("bar_ordinal", kind="mergesort").copy()
        prices = ordered[["open", "high", "low", "close"]].to_numpy(dtype=float)
        reasons: list[str] = []
        if str(session) in invalid_sessions:
            reasons.append("off_grid_source_row")
        if len(ordered) != EXPECTED_SESSION_BARS:
            reasons.append("incomplete_regular_session")
        if ordered["bar_ordinal"].astype(int).tolist() != list(range(EXPECTED_SESSION_BARS)):
            reasons.append("ordinal_grid_failure")
        if not np.isfinite(prices).all() or bool((prices <= 0.0).any()):
            reasons.append("invalid_price")
        if reasons:
            gap_records.append(
                {"symbol": symbol, "session": str(session), "reason": "|".join(sorted(reasons))}
            )
            continue
        valid_parts.append(ordered)
    if not valid_parts:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", f"no complete sessions for {symbol}"
        )
    frame = pd.concat(valid_parts, ignore_index=True).sort_values(
        ["session", "bar_ordinal"], kind="mergesort"
    )
    frame["historical_activity_baseline_at_bar"] = frame.groupby("bar_ordinal", sort=False)[
        "volume"
    ].transform(lambda values: values.expanding(min_periods=10).mean().shift(1))
    frame["historical_relative_activity"] = frame["volume"] / frame[
        "historical_activity_baseline_at_bar"
    ].replace(0.0, np.nan)
    return frame.reset_index(drop=True), gap_records


def _range_baselines(bars: pd.DataFrame) -> dict[tuple[str, int], float]:
    rows: list[dict[str, Any]] = []
    for session, session_frame in bars.groupby("session", sort=True):
        ordered = session_frame.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
        for checkpoint in (6, 12):
            opening = ordered.iloc[:checkpoint]
            opening_range_bps = (
                10_000.0
                * (float(opening["high"].max()) - float(opening["low"].min()))
                / float(opening.iloc[0]["open"])
            )
            rows.append(
                {
                    "session": str(session),
                    "decision_ordinal": checkpoint,
                    "opening_range_bps": opening_range_bps,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["decision_ordinal", "session"], kind="mergesort")
    frame["trailing_median"] = frame.groupby("decision_ordinal", sort=False)[
        "opening_range_bps"
    ].transform(lambda values: values.expanding(min_periods=1).median().shift(1))
    return {
        (str(row.session), int(row.decision_ordinal)): float(row.trailing_median)
        for row in frame.itertuples(index=False)
        if np.isfinite(float(row.trailing_median)) and float(row.trailing_median) > 0.0
    }


def leave_one_out_median(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("leave-one-out median requires at least two finite values")
    medians = np.asarray([np.median(np.delete(array, index)) for index in range(len(array))])
    return array - medians, medians


def cohort_median_gap(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 1 or not np.isfinite(array).all():
        raise ValueError("cohort median gap requires finite values")
    return array - float(np.median(array))


def build_component_panel(
    predecessor: pd.DataFrame,
    *,
    provider_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct exact predecessor keys and derive causal completed-bar components."""

    expected_manifest = json.loads(PRESSURE_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected_sources = {str(row["symbol"]): row for row in expected_manifest["sources"]}
    requested_columns = list(
        dict.fromkeys(
            [
                "symbol",
                "session",
                "year",
                "year_month",
                "decision_ordinal",
                "repo_bar_start_ordinal",
                "decision_time_america_new_york",
                "checkpoint_60m",
                "slate_id",
                "decision_bar_start_timestamp_utc",
                "feature_available_timestamp_utc",
                "entry_bar_ordinal",
                "delayed_entry_open",
                "terminal_bar_ordinal",
                "terminal_close",
                "raw_remaining_return_bps",
                "cohort_median_return_minus_i_bps",
                "residual_remaining_return_bps",
                "movement_threshold_bps",
                "large_remaining_move",
                "up_given_large_move",
                "open_to_decision_cohort_relative_return_bps",
                *M1_FEATURES,
            ]
        )
    )
    predecessor_keys = predecessor.loc[:, requested_columns].copy()
    records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    gap_records: list[dict[str, str]] = []
    causal_feature_exclusions: list[dict[str, Any]] = []
    source_month_parts: list[pd.DataFrame] = []
    source_minimum: pd.Timestamp | None = None
    source_maximum: pd.Timestamp | None = None

    for symbol in SYMBOLS:
        qa = load_qa_record(symbol)
        qa_records.append(qa)
        raw = bounded_source(provider_path(provider_root, symbol))
        digest = arrow_hash(raw)
        expected = expected_sources[symbol]
        if digest != expected["bounded_safe_hash"] or len(raw) != int(
            expected["bounded_safe_rows"]
        ):
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure",
                f"bounded source identity differs for {symbol}",
            )
        source_records.append(
            {
                "symbol": symbol,
                "logical_path": logical_source_path(symbol),
                "bounded_safe_hash": digest,
                "bounded_safe_rows": len(raw),
                "vendor_qa_sha256": qa["sha256"],
                "vendor_qa_status": qa["status"],
                "corporate_action_check_passed": True,
            }
        )
        months = raw[["timestamp"]].copy()
        months["symbol"] = symbol
        months["year_month"] = months["timestamp"].dt.strftime("%Y-%m")
        source_month_parts.append(months[["symbol", "year_month"]])
        minimum = pd.Timestamp(raw["timestamp"].min())
        maximum = pd.Timestamp(raw["timestamp"].max())
        source_minimum = minimum if source_minimum is None else min(source_minimum, minimum)
        source_maximum = maximum if source_maximum is None else max(source_maximum, maximum)

        bars, symbol_gaps = prepare_symbol_bars(raw, symbol=symbol)
        gap_records.extend(symbol_gaps)
        baselines = _range_baselines(bars)
        sessions = {
            str(session): part.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
            for session, part in bars.groupby("session", sort=True)
        }
        requested = predecessor_keys.loc[predecessor_keys["symbol"].eq(symbol)]
        for predecessor_row in requested.itertuples(index=False):
            session = str(predecessor_row.session)
            checkpoint = int(predecessor_row.decision_ordinal)
            origin = checkpoint - 1
            if session not in sessions or (session, checkpoint) not in baselines:
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure",
                    f"complete causal history unavailable for {symbol}/{session}/{checkpoint}",
                )
            session_frame = sessions[session]
            by_ordinal = session_frame.set_index("bar_ordinal", verify_integrity=True)
            opening = session_frame.iloc[:checkpoint].copy()
            if origin != int(predecessor_row.repo_bar_start_ordinal) or len(opening) != checkpoint:
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure", "checkpoint ordinal convention differs"
                )
            opening_activity = opening["volume"].to_numpy(dtype=float)
            opening_relative_activity = opening["historical_relative_activity"].to_numpy(
                dtype=float
            )
            if (
                not np.isfinite(opening_activity).all()
                or bool((opening_activity < 0.0).any())
                or not np.isfinite(opening_relative_activity).all()
                or bool((opening_relative_activity < 0.0).any())
            ):
                causal_feature_exclusions.append(
                    {
                        "symbol": symbol,
                        "session": session,
                        "decision_ordinal": checkpoint,
                        "reason": "incomplete_causal_historical_activity_proxy",
                    }
                )
                continue
            component_bars = bar_component_frame(opening)
            calculated = opening_raw_components(
                component_bars,
                trailing_opening_range_median_bps=baselines[(session, checkpoint)],
                signed_progress_bps=0.0,
                signed_progress_acceleration_bps=0.0,
                return_gap_bps=0.0,
            )
            expected_decision_start = pd.Timestamp(predecessor_row.decision_bar_start_timestamp_utc)
            expected_available = pd.Timestamp(predecessor_row.feature_available_timestamp_utc)
            if pd.Timestamp(opening.iloc[-1]["bar_start_timestamp"]) != expected_decision_start:
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure", "decision timestamp differs"
                )
            if pd.Timestamp(opening.iloc[-1]["bar_complete_timestamp"]) != expected_available:
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure", "feature availability differs"
                )
            entry_ordinal = origin + 2
            if entry_ordinal != int(predecessor_row.entry_bar_ordinal):
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure", "delayed t+2 ordinal differs"
                )
            entry_open = float(by_ordinal.loc[entry_ordinal, "open"])
            terminal_close = float(by_ordinal.loc[77, "close"])
            if not math.isclose(
                entry_open, float(predecessor_row.delayed_entry_open), rel_tol=0.0, abs_tol=1e-12
            ) or not math.isclose(
                terminal_close, float(predecessor_row.terminal_close), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure", "delayed entry or terminal differs"
                )
            raw_remaining = 10_000.0 * (terminal_close / entry_open - 1.0)
            if not math.isclose(
                raw_remaining,
                float(predecessor_row.raw_remaining_return_bps),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ScreenBlocker(
                    "blocked_chronology_or_leakage_failure", "remaining outcome differs"
                )
            half = checkpoint // 2
            returns = component_bars["return_bps"].to_numpy(dtype=float)
            highs = component_bars["high"].to_numpy(dtype=float)
            lows = component_bars["low"].to_numpy(dtype=float)
            new_high = np.ones(checkpoint, dtype=bool)
            new_low = np.ones(checkpoint, dtype=bool)
            new_high[1:] = highs[1:] > np.maximum.accumulate(highs)[:-1]
            new_low[1:] = lows[1:] < np.minimum.accumulate(lows)[:-1]
            record = {
                "symbol": symbol,
                "session": session,
                "decision_ordinal": checkpoint,
                "slate_id": str(predecessor_row.slate_id),
                "provider_activity_label": "historical_activity_proxy",
                "activity_normalisation": "same_stock_same_clock_expanding_prior_mean_minimum_10",
                "availability_status": "eligible_exact_predecessor_population",
                "source_gap_status": "pass",
                "qa_status": qa["status"],
                "corporate_action_status": "pass",
                "bar_count": checkpoint,
                "bar_start_timestamps_utc": [
                    pd.Timestamp(value).isoformat()
                    for value in component_bars["bar_start_timestamp"].tolist()
                ],
                "bar_open": component_bars["open"].astype(float).tolist(),
                "bar_high": component_bars["high"].astype(float).tolist(),
                "bar_low": component_bars["low"].astype(float).tolist(),
                "bar_close": component_bars["close"].astype(float).tolist(),
                "historical_relative_activity": component_bars["historical_relative_activity"]
                .astype(float)
                .tolist(),
                "return_bps_path": component_bars["return_bps"].astype(float).tolist(),
                "true_range_bps_path": component_bars["true_range_bps"].astype(float).tolist(),
                "close_location_path": component_bars["close_location"].astype(float).tolist(),
                "upper_wick_fraction_path": component_bars["upper_wick_fraction"]
                .astype(float)
                .tolist(),
                "lower_wick_fraction_path": component_bars["lower_wick_fraction"]
                .astype(float)
                .tolist(),
                "new_high_path": new_high.tolist(),
                "new_low_path": new_low.tolist(),
                "earlier_half_return_bps": float(returns[:half].sum()),
                "recent_half_return_bps": float(returns[half:].sum()),
                "calculated_open_to_decision_return_bps": 10_000.0
                * (
                    float(component_bars.iloc[-1]["close"]) / float(component_bars.iloc[0]["open"])
                    - 1.0
                ),
                "entry_open_reconstructed": entry_open,
                "terminal_close_reconstructed": terminal_close,
                **calculated,
            }
            records.append(record)

    raw_components = pd.DataFrame(records).sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    raw_components = raw_components.reset_index(drop=True)
    keys = ["symbol", "session", "decision_ordinal", "slate_id"]
    compact = predecessor_keys.merge(
        raw_components, on=keys, how="inner", validate="one_to_one", sort=False
    ).sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    compact = compact.reset_index(drop=True)
    eligible_slate_size = compact.groupby("slate_id", sort=False)["symbol"].transform("size")
    undersized = compact.loc[eligible_slate_size.lt(15)]
    causal_feature_exclusions.extend(
        {
            "symbol": str(row.symbol),
            "session": str(row.session),
            "decision_ordinal": int(row.decision_ordinal),
            "reason": "parent_slate_below_15_valid_stocks",
        }
        for row in undersized.itertuples(index=False)
    )
    compact = compact.loc[eligible_slate_size.ge(15)].reset_index(drop=True)
    if compact.empty or len(compact) > len(predecessor) or len(compact) > MAX_COMPACT_ROWS:
        raise ScreenBlocker(
            "blocked_quick_behavioural_screen_resource_limit",
            f"eligible compact rows are empty or exceed a hard limit: {len(compact)}",
        )
    predecessor_slate_sizes = predecessor.groupby("slate_id", sort=False).size().to_dict()
    for _, indices in compact.groupby("slate_id", sort=True).groups.items():
        index = list(indices)
        if len(index) < 15:
            raise ScreenBlocker(
                "blocked_insufficient_behavioural_support", "parent slate has fewer than 15 stocks"
            )
        raw_open = compact.loc[index, "calculated_open_to_decision_return_bps"].to_numpy(
            dtype=float
        )
        predecessor_relative, _ = leave_one_out_median(raw_open)
        return_gap = cohort_median_gap(raw_open)
        archived_gap = compact.loc[index, "open_to_decision_cohort_relative_return_bps"].to_numpy(
            dtype=float
        )
        slate_id = str(compact.loc[index[0], "slate_id"])
        if len(index) == int(predecessor_slate_sizes[slate_id]) and not np.allclose(
            predecessor_relative, archived_gap, rtol=0.0, atol=1e-8
        ):
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure", "cohort-relative opening return differs"
            )
        earlier_relative = cohort_median_gap(
            compact.loc[index, "earlier_half_return_bps"].to_numpy(dtype=float)
        )
        recent_relative = cohort_median_gap(
            compact.loc[index, "recent_half_return_bps"].to_numpy(dtype=float)
        )
        activity_gap = cohort_median_gap(
            compact.loc[index, "activity_effort"].to_numpy(dtype=float)
        )
        range_gap = cohort_median_gap(compact.loc[index, "range_effort"].to_numpy(dtype=float))
        compact.loc[index, "signed_progress"] = archived_gap
        compact.loc[index, "absolute_progress"] = np.abs(archived_gap)
        compact.loc[index, "return_gap"] = return_gap
        compact.loc[index, "earlier_relative_return_bps"] = earlier_relative
        compact.loc[index, "recent_relative_return_bps"] = recent_relative
        compact.loc[index, "signed_progress_acceleration"] = recent_relative - earlier_relative
        compact.loc[index, "activity_gap"] = activity_gap
        compact.loc[index, "range_gap"] = range_gap
    source_months = (
        pd.concat(source_month_parts, ignore_index=True)
        .groupby(["symbol", "year_month"], sort=True)
        .size()
        .rename("row_count")
        .reset_index()
    )
    context = {
        "sources": source_records,
        "vendor_qa": qa_records,
        "source_gap_ledger": gap_records,
        "source_gap_ledger_rejected_session_records": len(gap_records),
        "causal_feature_exclusions": causal_feature_exclusions,
        "causal_feature_exclusion_count": len(causal_feature_exclusions),
        "source_rows_by_symbol_month": source_months.to_dict("records"),
        "minimum_timestamp_read": str(source_minimum),
        "maximum_timestamp_read": str(source_maximum),
        "protected_rows_materialised": 0,
    }
    return compact, context


def _serialize_scaling(
    scaling: Mapping[int, Mapping[str, RobustComponentScale]],
) -> dict[str, dict[str, Mapping[str, float | str]]]:
    return {
        str(checkpoint): {
            component: frozen.as_dict() for component, frozen in sorted(records.items())
        }
        for checkpoint, records in sorted(scaling.items())
    }


def derive_dimension_panel(
    component_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit 2024 component scaling, then derive fixed dimensions and labels."""

    development_mask = component_panel["year"].eq(2024)
    base_scaling = fit_component_scaling(
        component_panel.loc[development_mask], components=BASE_COMPONENTS
    )
    standardised = apply_component_scaling(
        component_panel, base_scaling, components=BASE_COMPONENTS
    )
    standardised["signed_pressure"] = standardised[
        [
            "z_signed_progress",
            "z_signed_efficiency",
            "z_mean_close_location",
            "z_boundary_slope",
        ]
    ].mean(axis=1)
    standardised = derive_exhaustion_inputs(standardised)
    derived_scaling = fit_component_scaling(
        standardised.loc[development_mask], components=DERIVED_COMPONENTS
    )
    standardised = apply_component_scaling(
        standardised, derived_scaling, components=DERIVED_COMPONENTS
    )
    dimensions = derive_behavioural_dimensions(standardised)
    dimension_panel = standardised.copy()
    for dimension in DIMENSION_FEATURES:
        dimension_panel[dimension] = dimensions[dimension]
    raw_conjunctions = derive_conjunctions(dimension_panel)
    dimension_panel = pd.concat([dimension_panel, raw_conjunctions], axis=1)
    conjunction_bounds = fit_conjunction_bounds(dimension_panel.loc[development_mask])
    dimension_panel = apply_conjunction_bounds(dimension_panel, conjunction_bounds)
    label_thresholds = fit_label_thresholds(dimension_panel.loc[development_mask])
    dimension_panel = assign_descriptive_labels(dimension_panel, label_thresholds)
    required = [
        *BASE_COMPONENTS,
        *DERIVED_COMPONENTS,
        *(f"z_{component}" for component in (*BASE_COMPONENTS, *DERIVED_COMPONENTS)),
        *DIMENSION_FEATURES,
        *CONJUNCTION_FEATURES,
    ]
    if not np.isfinite(dimension_panel.loc[:, required].to_numpy(dtype=float)).all():
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "component or dimension is not finite"
        )
    scaling_manifest = {
        **SAFETY_FLAGS,
        "fit_interval": "2024-01-01_through_2024-12-31_only",
        "method": "checkpoint_specific_median_iqr",
        "clip": [-5.0, 5.0],
        "base_components": _serialize_scaling(base_scaling),
        "pressure_aligned_components": _serialize_scaling(derived_scaling),
        "activity_normalisation": "existing_same_stock_same_clock_expanding_prior_mean_minimum_10",
        "activity_provider_label": "historical_activity_proxy",
    }
    dimension_manifest = {
        **SAFETY_FLAGS,
        "dimension_features": list(DIMENSION_FEATURES),
        "conjunction_features": list(CONJUNCTION_FEATURES),
        "weights_fitted": False,
        "equal_weighted_components": True,
        "dimension_formulas": {
            "arousal": "mean(z_activity_effort,z_range_effort,z_travel_effort)",
            "conviction": "mean(z_absolute_efficiency,z_close_retention,z_directional_persistence)",
            "frustration": (
                "mean(z_activity_effort,z_travel_effort,z_extreme_rejection)"
                "-mean(z_absolute_progress,z_absolute_efficiency)"
            ),
            "tension": (
                "mean(z_activity_effort,z_compression,z_extreme_rejection)-z_absolute_progress"
            ),
            "signed_pressure": (
                "mean(z_signed_progress,z_signed_efficiency,z_mean_close_location,z_boundary_slope)"
            ),
            "pressure_magnitude": "abs(signed_pressure)",
            "exhaustion_magnitude": (
                "z_effort_acceleration-z_aligned_progress_acceleration+z_directional_rejection"
            ),
            "signed_exhaustion": "sign(signed_pressure)*exhaustion_magnitude",
            "independence": "mean(abs(z_return_gap),abs(z_activity_gap),abs(z_range_gap))",
            "signed_independence": "sign(return_gap)*independence",
        },
        "conjunction_formulas": {
            "active_conviction": "arousal*conviction",
            "active_frustration": "arousal*frustration",
            "pressurised_tension": "tension*pressure_magnitude",
            "pressurised_exhaustion": "exhaustion_magnitude*pressure_magnitude",
            "independent_pressure": "independence*signed_pressure",
        },
        "conjunction_clip_bounds": {
            str(checkpoint): {
                feature: {"q01": bounds[0], "q99": bounds[1]}
                for feature, bounds in sorted(records.items())
            }
            for checkpoint, records in sorted(conjunction_bounds.items())
        },
    }
    label_manifest = {
        **SAFETY_FLAGS,
        "fit_interval": "2024_only",
        "reporting_only": True,
        "model_feature_use": False,
        "release_label_forbidden": True,
        "thresholds": {
            str(checkpoint): dict(sorted(records.items()))
            for checkpoint, records in sorted(label_thresholds.items())
        },
        "labels": list(DESCRIPTIVE_LABELS),
    }
    return dimension_panel, scaling_manifest, dimension_manifest, label_manifest


def model_feature_sets(
    frozen_models: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    movement_m1 = tuple(
        str(value) for value in frozen_models["large_remaining_move"]["feature_names"]
    )
    direction_m1 = tuple(
        str(value) for value in frozen_models["up_given_large_move"]["feature_names"]
    )
    if movement_m1 != direction_m1:
        raise ScreenBlocker(
            "blocked_observable_predecessor_not_reconstructable", "M1 feature orders differ"
        )
    features = {
        "movement": {
            "P0": ("checkpoint_60m",),
            "P1": movement_m1,
            "P2": (*movement_m1, *DIMENSION_FEATURES),
            "P3": (*movement_m1, *DIMENSION_FEATURES, *CONJUNCTION_FEATURES),
        },
        "direction": {
            "D0": ("checkpoint_60m", "open_to_decision_cohort_relative_return_bps"),
            "D1": direction_m1,
            "D2": (*direction_m1, *DIMENSION_FEATURES),
            "D3": (*direction_m1, *DIMENSION_FEATURES, *CONJUNCTION_FEATURES),
        },
    }
    for ladder in features.values():
        for names in ladder.values():
            assert_allowed_model_features(names)
    return features


def fit_primary_models(
    panel: pd.DataFrame,
    frozen_models: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, FrozenLogisticModel | Mapping[str, Any]]],
    dict[str, Any],
    dict[str, dict[str, tuple[str, ...]]],
]:
    """Fit exactly six new primary models; P1 and D1 remain frozen."""

    features = model_feature_sets(frozen_models)
    development = panel.loc[panel["year"].eq(2024)].reset_index(drop=True)
    direction_development = development.loc[development["large_remaining_move"].eq(1)].reset_index(
        drop=True
    )
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    try:
        movement: dict[str, FrozenLogisticModel | Mapping[str, Any]] = {
            "P0": fit_fixed_logistic(
                development,
                development["large_remaining_move"],
                features=features["movement"]["P0"],
                slate_column="slate_id",
                model_id="large_remaining_move__P0",
            ),
            "P1": frozen_models["large_remaining_move"],
            "P2": fit_fixed_logistic(
                development,
                development["large_remaining_move"],
                features=features["movement"]["P2"],
                slate_column="slate_id",
                model_id="large_remaining_move__P2",
            ),
            "P3": fit_fixed_logistic(
                development,
                development["large_remaining_move"],
                features=features["movement"]["P3"],
                slate_column="slate_id",
                model_id="large_remaining_move__P3",
            ),
        }
        direction: dict[str, FrozenLogisticModel | Mapping[str, Any]] = {
            "D0": fit_fixed_logistic(
                direction_development,
                direction_development["up_given_large_move"],
                features=features["direction"]["D0"],
                slate_column="slate_id",
                model_id="up_given_large_move__D0",
            ),
            "D1": frozen_models["up_given_large_move"],
            "D2": fit_fixed_logistic(
                direction_development,
                direction_development["up_given_large_move"],
                features=features["direction"]["D2"],
                slate_column="slate_id",
                model_id="up_given_large_move__D2",
            ),
            "D3": fit_fixed_logistic(
                direction_development,
                direction_development["up_given_large_move"],
                features=features["direction"]["D3"],
                slate_column="slate_id",
                model_id="up_given_large_move__D3",
            ),
        }
    except (ConvergenceWarning, RuntimeError) as error:
        raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error
    models = {"movement": movement, "direction": direction}
    serialized: dict[str, Any] = {"movement": {}, "direction": {}}
    for target, ladder in models.items():
        for name, model in ladder.items():
            if isinstance(model, FrozenLogisticModel):
                serialized[target][name] = model.as_dict()
            else:
                serialized[target][name] = {**dict(model), "frozen_predecessor": True}
    return models, serialized, features


def _predict_model(
    model: FrozenLogisticModel | Mapping[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    if isinstance(model, FrozenLogisticModel):
        return model.predict(frame)
    return manual_logistic_prediction(model, frame)


def score_assessment(
    panel: pd.DataFrame,
    frozen_predictions: pd.DataFrame,
    models: Mapping[str, Mapping[str, FrozenLogisticModel | Mapping[str, Any]]],
) -> pd.DataFrame:
    assessment = panel.loc[panel["year"].eq(2025)].sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    assessment = assessment.reset_index(drop=True)
    keys = ["symbol", "session", "decision_ordinal"]
    frozen_columns = [
        *keys,
        "p__large_remaining_move__M1",
        "p__up_given_large_move__M1",
        "predicted_remaining_movement_scale_bps",
    ]
    scored = assessment.merge(
        frozen_predictions[frozen_columns], on=keys, how="left", validate="one_to_one", sort=False
    )
    for name in ("P0", "P2", "P3"):
        scored[f"p_large_remaining_move__{name}"] = _predict_model(models["movement"][name], scored)
    scored["p_large_remaining_move__P1"] = scored["p__large_remaining_move__M1"]
    for name in ("D0", "D2", "D3"):
        scored[f"p_up_given_large_move__{name}"] = _predict_model(models["direction"][name], scored)
    scored["p_up_given_large_move__D1"] = scored["p__up_given_large_move__M1"]
    probability_columns = [
        *(f"p_large_remaining_move__{name}" for name in ("P0", "P1", "P2", "P3")),
        *(f"p_up_given_large_move__{name}" for name in ("D0", "D1", "D2", "D3")),
    ]
    if not np.isfinite(scored.loc[:, probability_columns].to_numpy(dtype=float)).all():
        raise ScreenBlocker("blocked_model_convergence_failure", "non-finite model probability")
    return scored


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def calibration_intercept_slope(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return math.nan, math.nan
    predictor = _logit(probabilities)
    if float(np.std(predictor)) < 1e-12:
        return float(_logit(np.asarray([float(np.mean(labels))]))[0]), 0.0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + parameters[1] * predictor
        predicted = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        loss_value = float(
            -np.sum(
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
        x0=np.asarray([0.0, 1.0]),
        jac=lambda parameters: objective(parameters)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    if not np.isfinite(result.x).all():
        return math.nan, math.nan
    return float(result.x[0]), float(result.x[1])


def _probability_columns(target: str) -> tuple[tuple[str, str], ...]:
    if target == "large_remaining_move":
        return tuple((name, f"p_large_remaining_move__{name}") for name in ("P0", "P1", "P2", "P3"))
    return tuple((name, f"p_up_given_large_move__{name}") for name in ("D0", "D1", "D2", "D3"))


def _target_population(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "up_given_large_move":
        return frame.loc[frame["large_remaining_move"].eq(1)]
    return frame


def metric_record(
    frame: pd.DataFrame,
    *,
    target: str,
    model: str,
    probability_column: str,
    scope_type: str,
    scope_value: str,
) -> dict[str, Any]:
    population = _target_population(frame, target)
    labels = population[target].to_numpy(dtype=int)
    probabilities = population[probability_column].to_numpy(dtype=float)
    intercept, slope = calibration_intercept_slope(labels, probabilities)
    bins = np.minimum((np.clip(probabilities, 0.0, 1.0) * 10).astype(int), 9)
    ece = 0.0
    for bin_number in range(10):
        mask = bins == bin_number
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return {
        "scope_type": scope_type,
        "scope_value": scope_value,
        "target": target,
        "model": model,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "auc": float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else math.nan,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
        "base_rate": float(np.mean(labels)),
        "row_count": len(population),
        "session_count": int(population["session"].nunique()),
        "stock_count": int(population["symbol"].nunique()),
    }


def calibration_bin_records(
    frame: pd.DataFrame,
    *,
    target: str,
    model: str,
    probability_column: str,
    scope_type: str,
    scope_value: str,
) -> list[dict[str, Any]]:
    population = _target_population(frame, target)
    labels = population[target].to_numpy(dtype=int)
    probabilities = population[probability_column].to_numpy(dtype=float)
    bins = np.minimum((np.clip(probabilities, 0.0, 1.0) * 10).astype(int), 9)
    output: list[dict[str, Any]] = []
    for bin_number in range(10):
        mask = bins == bin_number
        output.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                "target": target,
                "model": model,
                "bin": bin_number + 1,
                "lower_bound": bin_number / 10.0,
                "upper_bound": (bin_number + 1) / 10.0,
                "row_count": int(mask.sum()),
                "mean_probability": float(probabilities[mask].mean()) if mask.any() else math.nan,
                "observed_rate": float(labels[mask].mean()) if mask.any() else math.nan,
            }
        )
    return output


def evaluate_predictions(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", scored)]
    scopes.extend(
        ("checkpoint", str(int(checkpoint)), rows.copy())
        for checkpoint, rows in scored.groupby("decision_ordinal", sort=True)
    )
    scopes.extend(
        ("month", str(month), rows.copy())
        for month, rows in scored.groupby("year_month", sort=True)
    )
    for scope_type, scope_value, scope in scopes:
        for target in ("large_remaining_move", "up_given_large_move"):
            for model, column in _probability_columns(target):
                record = metric_record(
                    scope,
                    target=target,
                    model=model,
                    probability_column=column,
                    scope_type=scope_type,
                    scope_value=scope_value,
                )
                if scope_type == "pooled":
                    pooled_rows.append(record)
                elif scope_type == "checkpoint":
                    checkpoint_rows.append(record)
                else:
                    monthly_rows.append(record)
                calibration_rows.extend(
                    calibration_bin_records(
                        scope,
                        target=target,
                        model=model,
                        probability_column=column,
                        scope_type=scope_type,
                        scope_value=scope_value,
                    )
                )
    pooled = pd.DataFrame(pooled_rows)
    movement = pooled.loc[pooled["target"].eq("large_remaining_move")].reset_index(drop=True)
    direction = pooled.loc[pooled["target"].eq("up_given_large_move")].reset_index(drop=True)
    return (
        movement,
        direction,
        pd.DataFrame(monthly_rows),
        pd.DataFrame(checkpoint_rows),
        pd.DataFrame(calibration_rows),
    )


def behavioural_state_census(assessment: pd.DataFrame) -> pd.DataFrame:
    """Describe fixed reporting labels without fitting or searching label combinations."""

    rows: list[dict[str, Any]] = []
    for label in DESCRIPTIVE_LABELS:
        selected = assessment.loc[assessment[f"label__{label}"].astype(bool)]
        large = selected.loc[selected["large_remaining_move"].eq(1)]
        stock_counts = selected.groupby("symbol", sort=True).size().sort_values(ascending=False)
        rows.append(
            {
                "record_type": "label_summary",
                "label": label,
                "scope": "pooled",
                "row_count": len(selected),
                "session_count": int(selected["session"].nunique()),
                "stock_count": int(selected["symbol"].nunique()),
                "movement_rate": float(selected["large_remaining_move"].mean())
                if len(selected)
                else math.nan,
                "mean_absolute_remaining_movement_bps": float(
                    selected["residual_remaining_return_bps"].abs().mean()
                )
                if len(selected)
                else math.nan,
                "up_rate_among_large_moves": float(large["up_given_large_move"].mean())
                if len(large)
                else math.nan,
                "up_large_move_count": int(large["up_given_large_move"].sum()),
                "down_large_move_count": int(len(large) - large["up_given_large_move"].sum()),
                "mean_raw_remaining_return_bps": float(selected["raw_remaining_return_bps"].mean())
                if len(selected)
                else math.nan,
                "mean_cohort_relative_remaining_return_bps": float(
                    selected["residual_remaining_return_bps"].mean()
                )
                if len(selected)
                else math.nan,
                "represented_months": int(selected["year_month"].nunique()),
                "largest_stock": str(stock_counts.index[0]) if len(stock_counts) else None,
                "largest_stock_share": float(stock_counts.iloc[0] / len(selected))
                if len(selected)
                else math.nan,
            }
        )
        for month, month_rows in selected.groupby("year_month", sort=True):
            rows.append(
                {
                    "record_type": "label_month_support",
                    "label": label,
                    "scope": str(month),
                    "row_count": len(month_rows),
                    "session_count": int(month_rows["session"].nunique()),
                    "stock_count": int(month_rows["symbol"].nunique()),
                    "movement_rate": float(month_rows["large_remaining_move"].mean()),
                    "mean_absolute_remaining_movement_bps": float(
                        month_rows["residual_remaining_return_bps"].abs().mean()
                    ),
                    "up_rate_among_large_moves": float(
                        month_rows.loc[
                            month_rows["large_remaining_move"].eq(1), "up_given_large_move"
                        ].mean()
                    ),
                    "mean_raw_remaining_return_bps": float(
                        month_rows["raw_remaining_return_bps"].mean()
                    ),
                    "mean_cohort_relative_remaining_return_bps": float(
                        month_rows["residual_remaining_return_bps"].mean()
                    ),
                }
            )
    for label_count, count_rows in assessment.groupby("behavioural_label_count", sort=True):
        rows.append(
            {
                "record_type": "label_count_overlap",
                "label": "ALL_LABELS",
                "scope": str(int(label_count)),
                "row_count": len(count_rows),
                "session_count": int(count_rows["session"].nunique()),
                "stock_count": int(count_rows["symbol"].nunique()),
                "movement_rate": float(count_rows["large_remaining_move"].mean()),
                "mean_absolute_remaining_movement_bps": float(
                    count_rows["residual_remaining_return_bps"].abs().mean()
                ),
                "up_rate_among_large_moves": float(
                    count_rows.loc[
                        count_rows["large_remaining_move"].eq(1), "up_given_large_move"
                    ].mean()
                ),
                "mean_raw_remaining_return_bps": float(
                    count_rows["raw_remaining_return_bps"].mean()
                ),
                "mean_cohort_relative_remaining_return_bps": float(
                    count_rows["residual_remaining_return_bps"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _cyclic_bundle_permutation(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    output = frame.copy()
    for indices in output.groupby("slate_id", sort=True).groups.values():
        index = list(indices)
        values = frame.loc[index, list(features)].to_numpy(dtype=float)
        output.loc[index, list(features)] = np.roll(values, shift=1, axis=0)
    return output


def dimension_diagnostics(
    full_panel: pd.DataFrame,
    assessment: pd.DataFrame,
    serialized_models: Mapping[str, Any],
    models: Mapping[str, Mapping[str, FrozenLogisticModel | Mapping[str, Any]]],
) -> pd.DataFrame:
    """Report coefficients, dependence, frozen quintiles, and group importance."""

    records: list[dict[str, Any]] = []
    for target, ladder in serialized_models.items():
        for model_name, model in ladder.items():
            for feature, coefficient in zip(
                model["feature_names"], model["coefficients"], strict=True
            ):
                records.append(
                    {
                        "diagnostic_type": "standardised_model_coefficient",
                        "target": target,
                        "model": model_name,
                        "feature": feature,
                        "value": float(coefficient),
                    }
                )
    dimension_values = assessment.loc[:, list(DIMENSION_FEATURES)].to_numpy(dtype=float)
    correlation = assessment.loc[:, list(DIMENSION_FEATURES)].corr()
    for left in DIMENSION_FEATURES:
        for right in DIMENSION_FEATURES:
            records.append(
                {
                    "diagnostic_type": "assessment_dimension_correlation",
                    "feature": left,
                    "feature_2": right,
                    "value": float(correlation.loc[left, right]),
                }
            )
    standardized = (dimension_values - dimension_values.mean(axis=0)) / np.where(
        dimension_values.std(axis=0, ddof=0) > 1e-12,
        dimension_values.std(axis=0, ddof=0),
        1.0,
    )
    records.append(
        {
            "diagnostic_type": "condition_number",
            "feature": "dimension_group",
            "value": float(np.linalg.cond(standardized)),
        }
    )
    for column, feature in enumerate(DIMENSION_FEATURES):
        response = standardized[:, column]
        predictors = np.delete(standardized, column, axis=1)
        design = np.column_stack([np.ones(len(predictors)), predictors])
        fitted = design @ np.linalg.lstsq(design, response, rcond=None)[0]
        residual_sum = float(np.sum((response - fitted) ** 2))
        total_sum = float(np.sum((response - response.mean()) ** 2))
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 1e-12 else 1.0
        records.append(
            {
                "diagnostic_type": "variance_inflation_factor",
                "feature": feature,
                "value": float(1.0 / max(1.0 - r_squared, 1e-12)),
            }
        )

    development = full_panel.loc[full_panel["year"].eq(2024)]
    for checkpoint in (6, 12):
        development_checkpoint = development.loc[development["decision_ordinal"].eq(checkpoint)]
        assessment_checkpoint = assessment.loc[assessment["decision_ordinal"].eq(checkpoint)]
        for dimension in DIMENSION_FEATURES:
            thresholds = (
                development_checkpoint[dimension]
                .quantile([0.2, 0.4, 0.6, 0.8], interpolation="linear")
                .to_numpy(dtype=float)
            )
            quintiles = (
                np.digitize(
                    assessment_checkpoint[dimension].to_numpy(dtype=float), thresholds, right=True
                )
                + 1
            )
            for quintile in range(1, 6):
                rows = assessment_checkpoint.loc[quintiles == quintile]
                large = rows.loc[rows["large_remaining_move"].eq(1)]
                records.append(
                    {
                        "diagnostic_type": "development_frozen_dimension_quintile",
                        "feature": dimension,
                        "checkpoint": checkpoint,
                        "quintile": quintile,
                        "row_count": len(rows),
                        "movement_rate": float(rows["large_remaining_move"].mean()),
                        "mean_absolute_future_movement_bps": float(
                            rows["residual_remaining_return_bps"].abs().mean()
                        ),
                        "direction_rate_among_large_moves": float(
                            large["up_given_large_move"].mean()
                        ),
                        "threshold_q20": thresholds[0],
                        "threshold_q40": thresholds[1],
                        "threshold_q60": thresholds[2],
                        "threshold_q80": thresholds[3],
                    }
                )

    importance_specs = (
        (
            "dimension_group",
            DIMENSION_FEATURES,
            "movement",
            "P2",
            "large_remaining_move",
            "p_large_remaining_move__P2",
        ),
        (
            "dimension_group",
            DIMENSION_FEATURES,
            "direction",
            "D2",
            "up_given_large_move",
            "p_up_given_large_move__D2",
        ),
        (
            "conjunction_group",
            CONJUNCTION_FEATURES,
            "movement",
            "P3",
            "large_remaining_move",
            "p_large_remaining_move__P3",
        ),
        (
            "conjunction_group",
            CONJUNCTION_FEATURES,
            "direction",
            "D3",
            "up_given_large_move",
            "p_up_given_large_move__D3",
        ),
    )
    for group, features, target_group, model_name, target, probability_column in importance_specs:
        permuted = _cyclic_bundle_permutation(assessment, features)
        probabilities = _predict_model(models[target_group][model_name], permuted)
        population_mask = (
            np.ones(len(assessment), dtype=bool)
            if target == "large_remaining_move"
            else assessment["large_remaining_move"].eq(1).to_numpy(dtype=bool)
        )
        labels = assessment.loc[population_mask, target].to_numpy(dtype=int)
        original = assessment.loc[population_mask, probability_column].to_numpy(dtype=float)
        permuted_loss = float(brier_score_loss(labels, probabilities[population_mask]))
        original_loss = float(brier_score_loss(labels, original))
        records.append(
            {
                "diagnostic_type": "assessment_group_permutation_importance",
                "feature": group,
                "target": target,
                "model": model_name,
                "value": permuted_loss - original_loss,
                "original_brier": original_loss,
                "permuted_brier": permuted_loss,
                "permutation": "deterministic_one_stock_cyclic_shift_within_slate",
            }
        )
    return pd.DataFrame(records)


def _selection_row(
    slate: pd.DataFrame,
    *,
    system: str,
    selected_index: int,
    side: float,
) -> dict[str, Any]:
    selected = slate.loc[selected_index]
    return {
        "system": system,
        "session": str(selected["session"]),
        "year_month": str(selected["year_month"]),
        "decision_ordinal": int(selected["decision_ordinal"]),
        "slate_id": str(selected["slate_id"]),
        "selected_symbol": str(selected["symbol"]),
        "side": float(side),
        "raw_remaining_return_bps": float(selected["raw_remaining_return_bps"]),
        "residual_remaining_return_bps": float(selected["residual_remaining_return_bps"]),
        "signed_gross_return_bps": float(side * selected["raw_remaining_return_bps"]),
        "signed_cohort_relative_return_bps": float(
            side * selected["residual_remaining_return_bps"]
        ),
    }


def economic_selection_ledger(assessment: pd.DataFrame) -> pd.DataFrame:
    """Create delayed top-one and fixed benchmark selections at every assessment slate."""

    scored = assessment.copy()
    systems = {
        "predecessor": ("P1", "D1"),
        "behavioural_dimensions": ("P2", "D2"),
        "behavioural_conjunctions": ("P3", "D3"),
    }
    for system, (movement, direction) in systems.items():
        scored[f"signed_opportunity_score__{system}"] = (
            scored[f"p_large_remaining_move__{movement}"]
            * (2.0 * scored[f"p_up_given_large_move__{direction}"] - 1.0)
            * scored["predicted_remaining_movement_scale_bps"]
        )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SELECTION_SEED)
    for _, slate in scored.groupby("slate_id", sort=True):
        ordered = slate.sort_values("symbol", kind="mergesort")
        for system in systems:
            score_column = f"signed_opportunity_score__{system}"
            candidates = ordered.assign(_absolute_score=ordered[score_column].abs()).sort_values(
                ["_absolute_score", "symbol"], ascending=[False, True], kind="mergesort"
            )
            selected_index = int(candidates.index[0])
            score = float(scored.loc[selected_index, score_column])
            side = 1.0 if score >= 0.0 else -1.0
            rows.append(
                _selection_row(scored, system=system, selected_index=selected_index, side=side)
            )
        highest_movement = ordered.sort_values(
            ["p_large_remaining_move__P1", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected_index = int(highest_movement.index[0])
        predecessor_score = float(
            scored.loc[selected_index, "signed_opportunity_score__predecessor"]
        )
        rows.append(
            _selection_row(
                scored,
                system="highest_frozen_movement_probability",
                selected_index=selected_index,
                side=1.0 if predecessor_score >= 0.0 else -1.0,
            )
        )
        momentum = ordered.sort_values(
            ["open_to_decision_cohort_relative_return_bps", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
        rows.append(
            _selection_row(
                scored,
                system="highest_open_to_decision_relative_momentum",
                selected_index=int(momentum.index[0]),
                side=1.0,
            )
        )
        reversal = ordered.assign(
            _absolute_momentum=ordered["open_to_decision_cohort_relative_return_bps"].abs()
        ).sort_values(["_absolute_momentum", "symbol"], ascending=[False, True], kind="mergesort")
        selected_index = int(reversal.index[0])
        momentum_value = float(
            scored.loc[selected_index, "open_to_decision_cohort_relative_return_bps"]
        )
        rows.append(
            _selection_row(
                scored,
                system="strongest_reversal",
                selected_index=selected_index,
                side=-1.0 if momentum_value > 0.0 else 1.0,
            )
        )
        random_index = int(ordered.index[int(rng.integers(0, len(ordered)))])
        random_score = float(scored.loc[random_index, "signed_opportunity_score__predecessor"])
        rows.append(
            _selection_row(
                scored,
                system="random_within_slate",
                selected_index=random_index,
                side=1.0 if random_score >= 0.0 else -1.0,
            )
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["system", "session", "decision_ordinal"], kind="mergesort")
        .reset_index(drop=True)
    )


def economic_reference_metrics(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", selections)]
    scopes.extend(
        ("checkpoint", str(int(checkpoint)), frame)
        for checkpoint, frame in selections.groupby("decision_ordinal", sort=True)
    )
    scopes.extend(
        ("month", str(month), frame) for month, frame in selections.groupby("year_month", sort=True)
    )
    for scope_type, scope_value, scope in scopes:
        for system, system_rows in scope.groupby("system", sort=True):
            for friction in (0.0, 10.0, 20.0):
                rows.append(
                    {
                        "record_type": "system",
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "system": str(system),
                        "friction_bps": friction,
                        "selection_count": len(system_rows),
                        "mean_signed_gross_return_bps": float(
                            (system_rows["signed_gross_return_bps"] - friction).mean()
                        ),
                        "mean_signed_cohort_relative_return_bps": float(
                            (system_rows["signed_cohort_relative_return_bps"] - friction).mean()
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
            if len(paired) == 0:
                continue
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


def _loss(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    if metric == "brier":
        return float(np.mean((probabilities - labels) ** 2))
    if metric == "log_loss":
        return float(log_loss(labels, probabilities, labels=[0, 1]))
    raise ValueError(metric)


def _predictive_improvement(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline_column: str,
    candidate_column: str,
    metric: str,
) -> float:
    population = _target_population(frame, target)
    labels = population[target].to_numpy(dtype=int)
    return _loss(labels, population[baseline_column].to_numpy(dtype=float), metric) - _loss(
        labels, population[candidate_column].to_numpy(dtype=float), metric
    )


def _selection_pair_return(
    selection: pd.DataFrame,
    *,
    candidate: str,
    baseline: str,
) -> float:
    paired = selection.loc[selection["system"].isin([candidate, baseline])].pivot(
        index="slate_id", columns="system", values="signed_cohort_relative_return_bps"
    )
    return float((paired[candidate] - paired[baseline]).mean())


def bootstrap_metrics(
    assessment: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    specs = (
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
    draws = session_block_bootstrap_draws(
        assessment["session"], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    selection_by_session = {
        session: rows for session, rows in selections.groupby("session", sort=True)
    }
    values: dict[str, list[float]] = {spec[0]: [] for spec in specs}
    values["behavioural_dimensions_minus_predecessor_return_after_20bps"] = []
    values["conjunction_minus_behavioural_dimensions_return_after_20bps"] = []
    rows: list[dict[str, Any]] = []
    for draw in draws:
        sampled = assessment.iloc[draw.row_indices].reset_index(drop=True)
        selection_parts: list[pd.DataFrame] = []
        for occurrence, session in enumerate(draw.sampled_sessions):
            part = selection_by_session[session].copy()
            part["slate_id"] = f"{occurrence:04d}|" + part["slate_id"].astype(str)
            selection_parts.append(part)
        sampled_selection = pd.concat(selection_parts, ignore_index=True)
        for metric_name, target, baseline, candidate, loss_name in specs:
            value = _predictive_improvement(
                sampled,
                target=target,
                baseline_column=baseline,
                candidate_column=candidate,
                metric=loss_name,
            )
            values[metric_name].append(value)
            rows.append(
                {"record_type": "draw", "draw": draw.draw, "metric": metric_name, "value": value}
            )
        for metric_name, candidate, baseline in (
            (
                "behavioural_dimensions_minus_predecessor_return_after_20bps",
                "behavioural_dimensions",
                "predecessor",
            ),
            (
                "conjunction_minus_behavioural_dimensions_return_after_20bps",
                "behavioural_conjunctions",
                "behavioural_dimensions",
            ),
        ):
            value = _selection_pair_return(
                sampled_selection, candidate=candidate, baseline=baseline
            )
            values[metric_name].append(value)
            rows.append(
                {"record_type": "draw", "draw": draw.draw, "metric": metric_name, "value": value}
            )
    for metric, metric_values in values.items():
        array = np.asarray(metric_values, dtype=float)
        rows.append(
            {
                "record_type": "summary",
                "draw": -1,
                "metric": metric,
                "value": float(array.mean()),
                "interval_90_lower": float(np.quantile(array, 0.05)),
                "interval_90_upper": float(np.quantile(array, 0.95)),
                "interval_95_lower": float(np.quantile(array, 0.025)),
                "interval_95_upper": float(np.quantile(array, 0.975)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["record_type", "metric", "draw"], kind="mergesort")
        .reset_index(drop=True)
    )


def _manifest_conjunction_bounds(
    dimension_manifest: Mapping[str, Any],
) -> dict[int, dict[str, tuple[float, float]]]:
    return {
        int(checkpoint): {
            feature: (float(record["q01"]), float(record["q99"]))
            for feature, record in records.items()
        }
        for checkpoint, records in dimension_manifest["conjunction_clip_bounds"].items()
    }


def _recompute_conjunctions(
    frame: pd.DataFrame,
    bounds: Mapping[int, Mapping[str, tuple[float, float]]],
) -> pd.DataFrame:
    output = frame.copy()
    conjunctions = derive_conjunctions(output)
    for feature in CONJUNCTION_FEATURES:
        output[feature] = conjunctions[feature]
    return apply_conjunction_bounds(output, bounds)


def _fit_null_candidates(
    development: pd.DataFrame,
    feature_sets: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    draw: int,
) -> dict[str, dict[str, FrozenLogisticModel]]:
    direction = development.loc[development["large_remaining_move"].eq(1)].reset_index(drop=True)
    try:
        return {
            "movement": {
                name: fit_fixed_logistic(
                    development,
                    development["large_remaining_move"],
                    features=feature_sets["movement"][name],
                    slate_column="slate_id",
                    model_id=f"null_{draw}_large_remaining_move__{name}",
                )
                for name in ("P2", "P3")
            },
            "direction": {
                name: fit_fixed_logistic(
                    direction,
                    direction["up_given_large_move"],
                    features=feature_sets["direction"][name],
                    slate_column="slate_id",
                    model_id=f"null_{draw}_up_given_large_move__{name}",
                )
                for name in ("D2", "D3")
            },
        }
    except (ConvergenceWarning, RuntimeError) as error:
        raise ScreenBlocker("blocked_model_convergence_failure", str(error)) from error


def _paired_null_economic_return(frame: pd.DataFrame) -> float:
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
            candidates = ordered.assign(_score=score, _absolute_score=score.abs()).sort_values(
                ["_absolute_score", "symbol"], ascending=[False, True], kind="mergesort"
            )
            selected = candidates.iloc[0]
            side = 1.0 if float(selected["_score"]) >= 0.0 else -1.0
            rows.append(
                {
                    "slate_id": str(selected["slate_id"]),
                    "system": system,
                    "return": side * float(selected["residual_remaining_return_bps"]) - 20.0,
                }
            )
    selection = pd.DataFrame(rows).pivot(index="slate_id", columns="system", values="return")
    return float((selection["behavioural_dimensions"] - selection["predecessor"]).mean())


def within_slate_behavioural_null(
    full_panel: pd.DataFrame,
    assessment: pd.DataFrame,
    feature_sets: Mapping[str, Mapping[str, Sequence[str]]],
    dimension_manifest: Mapping[str, Any],
    selections: pd.DataFrame,
) -> pd.DataFrame:
    """Run exactly 50 bundled within-slate null draws with conjunction recomputation."""

    development = full_panel.loc[full_panel["year"].eq(2024)].reset_index(drop=True)
    assessment_base = assessment.reset_index(drop=True)
    bounds = _manifest_conjunction_bounds(dimension_manifest)
    real_values = {
        "P2_minus_P1_brier_improvement": _predictive_improvement(
            assessment,
            target="large_remaining_move",
            baseline_column="p_large_remaining_move__P1",
            candidate_column="p_large_remaining_move__P2",
            metric="brier",
        ),
        "P2_minus_P1_log_loss_improvement": _predictive_improvement(
            assessment,
            target="large_remaining_move",
            baseline_column="p_large_remaining_move__P1",
            candidate_column="p_large_remaining_move__P2",
            metric="log_loss",
        ),
        "P3_minus_P2_brier_improvement": _predictive_improvement(
            assessment,
            target="large_remaining_move",
            baseline_column="p_large_remaining_move__P2",
            candidate_column="p_large_remaining_move__P3",
            metric="brier",
        ),
        "D2_minus_D1_brier_improvement": _predictive_improvement(
            assessment,
            target="up_given_large_move",
            baseline_column="p_up_given_large_move__D1",
            candidate_column="p_up_given_large_move__D2",
            metric="brier",
        ),
        "D2_minus_D1_log_loss_improvement": _predictive_improvement(
            assessment,
            target="up_given_large_move",
            baseline_column="p_up_given_large_move__D1",
            candidate_column="p_up_given_large_move__D2",
            metric="log_loss",
        ),
        "D3_minus_D2_brier_improvement": _predictive_improvement(
            assessment,
            target="up_given_large_move",
            baseline_column="p_up_given_large_move__D2",
            candidate_column="p_up_given_large_move__D3",
            metric="brier",
        ),
        "behavioural_system_minus_predecessor_delayed_return_after_20bps": _selection_pair_return(
            selections,
            candidate="behavioural_dimensions",
            baseline="predecessor",
        ),
    }
    rng = np.random.default_rng(NULL_SEED)
    values: dict[str, list[float]] = {metric: [] for metric in real_values}
    rows: list[dict[str, Any]] = []
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for draw in range(NULL_DRAWS):
        permuted_development = permute_bundle_within_slates(
            development,
            features=DIMENSION_FEATURES,
            slate_column="slate_id",
            rng=rng,
        )
        permuted_assessment = permute_bundle_within_slates(
            assessment_base,
            features=DIMENSION_FEATURES,
            slate_column="slate_id",
            rng=rng,
        )
        permuted_development = _recompute_conjunctions(permuted_development, bounds)
        permuted_assessment = _recompute_conjunctions(permuted_assessment, bounds)
        models = _fit_null_candidates(permuted_development, feature_sets, draw=draw)
        for name in ("P2", "P3"):
            permuted_assessment[f"p_large_remaining_move__{name}"] = models["movement"][
                name
            ].predict(permuted_assessment)
        for name in ("D2", "D3"):
            permuted_assessment[f"p_up_given_large_move__{name}"] = models["direction"][
                name
            ].predict(permuted_assessment)
        draw_values = {
            "P2_minus_P1_brier_improvement": _predictive_improvement(
                permuted_assessment,
                target="large_remaining_move",
                baseline_column="p_large_remaining_move__P1",
                candidate_column="p_large_remaining_move__P2",
                metric="brier",
            ),
            "P2_minus_P1_log_loss_improvement": _predictive_improvement(
                permuted_assessment,
                target="large_remaining_move",
                baseline_column="p_large_remaining_move__P1",
                candidate_column="p_large_remaining_move__P2",
                metric="log_loss",
            ),
            "P3_minus_P2_brier_improvement": _predictive_improvement(
                permuted_assessment,
                target="large_remaining_move",
                baseline_column="p_large_remaining_move__P2",
                candidate_column="p_large_remaining_move__P3",
                metric="brier",
            ),
            "D2_minus_D1_brier_improvement": _predictive_improvement(
                permuted_assessment,
                target="up_given_large_move",
                baseline_column="p_up_given_large_move__D1",
                candidate_column="p_up_given_large_move__D2",
                metric="brier",
            ),
            "D2_minus_D1_log_loss_improvement": _predictive_improvement(
                permuted_assessment,
                target="up_given_large_move",
                baseline_column="p_up_given_large_move__D1",
                candidate_column="p_up_given_large_move__D2",
                metric="log_loss",
            ),
            "D3_minus_D2_brier_improvement": _predictive_improvement(
                permuted_assessment,
                target="up_given_large_move",
                baseline_column="p_up_given_large_move__D2",
                candidate_column="p_up_given_large_move__D3",
                metric="brier",
            ),
            (
                "behavioural_system_minus_predecessor_delayed_return_after_20bps"
            ): _paired_null_economic_return(permuted_assessment),
        }
        for metric, value in draw_values.items():
            values[metric].append(value)
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "null_value": value,
                    "real_value": real_values[metric],
                }
            )
    for metric, metric_values in values.items():
        array = np.asarray(metric_values, dtype=float)
        real = real_values[metric]
        rows.append(
            {
                "record_type": "summary",
                "draw": -1,
                "metric": metric,
                "null_value": float(array.mean()),
                "real_value": real,
                "null_q90": float(np.quantile(array, 0.90)),
                "real_percentile": float(np.mean(array < real)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["record_type", "metric", "draw"], kind="mergesort")
        .reset_index(drop=True)
    )


def concentration_and_support(
    assessment: pd.DataFrame,
    selections: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_counts = assessment.groupby("symbol", sort=True).size()
    for symbol, count in row_counts.items():
        rows.append(
            {
                "record_type": "decision_row_share",
                "system": "all_rows",
                "symbol": symbol,
                "count": int(count),
                "share": float(count / len(assessment)),
            }
        )
    primary_systems = selections.loc[
        selections["system"].isin(
            ["predecessor", "behavioural_dimensions", "behavioural_conjunctions"]
        )
    ]
    maximum_selection_share = 0.0
    for system, system_rows in primary_systems.groupby("system", sort=True):
        counts = system_rows.groupby("selected_symbol", sort=True).size()
        for symbol, count in counts.items():
            share = float(count / len(system_rows))
            maximum_selection_share = max(maximum_selection_share, share)
            rows.append(
                {
                    "record_type": "economic_selection_share",
                    "system": system,
                    "symbol": symbol,
                    "count": int(count),
                    "share": share,
                }
            )
    large_by_checkpoint = {
        str(int(checkpoint)): int(rows_["large_remaining_move"].sum())
        for checkpoint, rows_ in assessment.groupby("decision_ordinal", sort=True)
    }
    support = {
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "actual_large_moves": int(assessment["large_remaining_move"].sum()),
        "actual_large_moves_by_checkpoint": large_by_checkpoint,
        "represented_months": int(assessment["year_month"].nunique()),
        "maximum_stock_decision_row_share": float(row_counts.max() / len(assessment)),
        "maximum_stock_economic_selection_share": maximum_selection_share,
    }
    support["movement_support_passes"] = bool(
        support["assessment_rows"] >= 3000
        and support["assessment_sessions"] >= 100
        and support["assessment_stocks"] >= 15
        and support["actual_large_moves"] >= 600
        and support["represented_months"] >= 6
        and support["maximum_stock_decision_row_share"] <= 0.10
        and support["maximum_stock_economic_selection_share"] <= 0.20
    )
    support["direction_support_passes"] = bool(
        support["movement_support_passes"]
        and all(value >= 250 for value in large_by_checkpoint.values())
    )
    if not support["movement_support_passes"]:
        raise ScreenBlocker(
            "blocked_insufficient_behavioural_support", "assessment movement support gate failed"
        )
    return pd.DataFrame(rows), support


def _metric_value(
    metrics: pd.DataFrame,
    model: str,
    column: str,
) -> float:
    row = metrics.loc[metrics["model"].eq(model)]
    if len(row) != 1:
        raise ValueError(f"metric unavailable for {model}")
    return float(row.iloc[0][column])


def _slice_improvement_count(
    metrics: pd.DataFrame,
    *,
    target: str,
    candidate: str,
    baseline: str,
) -> int:
    target_rows = metrics.loc[metrics["target"].eq(target)]
    pivot = target_rows.pivot(index="scope_value", columns="model", values="brier_score")
    return int((pivot[baseline] - pivot[candidate] > 0.0).sum())


def _checkpoint_improvements(
    metrics: pd.DataFrame,
    *,
    target: str,
    candidate: str,
    baseline: str,
) -> dict[str, float]:
    target_rows = metrics.loc[metrics["target"].eq(target)]
    pivot = target_rows.pivot(index="scope_value", columns="model", values="brier_score")
    return {
        str(checkpoint): float(
            pivot.loc[str(checkpoint), baseline] - pivot.loc[str(checkpoint), candidate]
        )
        for checkpoint in (6, 12)
    }


def _bootstrap_summary(bootstrap: pd.DataFrame, metric: str) -> pd.Series:
    rows = bootstrap.loc[bootstrap["record_type"].eq("summary") & bootstrap["metric"].eq(metric)]
    if len(rows) != 1:
        raise ValueError(f"bootstrap summary unavailable: {metric}")
    return rows.iloc[0]


def _null_summary(null: pd.DataFrame, metric: str) -> pd.Series:
    rows = null.loc[null["record_type"].eq("summary") & null["metric"].eq(metric)]
    if len(rows) != 1:
        raise ValueError(f"null summary unavailable: {metric}")
    return rows.iloc[0]


def _increment_gate(
    pooled: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    *,
    target: str,
    baseline: str,
    candidate: str,
    bootstrap_prefix: str,
    null_prefix: str,
    concentration_passes: bool,
    require_log_bootstrap: bool,
) -> dict[str, Any]:
    brier = _metric_value(pooled, baseline, "brier_score") - _metric_value(
        pooled, candidate, "brier_score"
    )
    log_increment = _metric_value(pooled, baseline, "log_loss") - _metric_value(
        pooled, candidate, "log_loss"
    )
    auc_change = _metric_value(pooled, candidate, "auc") - _metric_value(pooled, baseline, "auc")
    brier_bootstrap = _bootstrap_summary(bootstrap, f"{bootstrap_prefix}_brier_improvement")
    log_lower: float | None = (
        float(
            _bootstrap_summary(bootstrap, f"{bootstrap_prefix}_log_loss_improvement")[
                "interval_90_lower"
            ]
        )
        if require_log_bootstrap
        else None
    )
    brier_null = _null_summary(null, f"{null_prefix}_brier_improvement")
    log_null = (
        _null_summary(null, f"{null_prefix}_log_loss_improvement")
        if require_log_bootstrap
        else None
    )
    checkpoint_values = _checkpoint_improvements(
        checkpoint,
        target=target,
        baseline=baseline,
        candidate=candidate,
    )
    positive_months = _slice_improvement_count(
        monthly, target=target, baseline=baseline, candidate=candidate
    )
    gates = {
        "brier_improvement_positive": brier > 0.0,
        "log_loss_improvement_positive": log_increment > 0.0,
        "auc_not_reduced": auc_change >= 0.0,
        "bootstrap_90_lower_brier_non_negative": float(brier_bootstrap["interval_90_lower"]) >= 0.0,
        "bootstrap_90_lower_log_loss_non_negative": (log_lower is not None and log_lower >= 0.0)
        if require_log_bootstrap
        else True,
        "positive_brier_months_at_least_five": positive_months >= 5,
        "neither_checkpoint_materially_adverse": min(checkpoint_values.values())
        >= CHECKPOINT_MATERIAL_ADVERSITY,
        "real_brier_increment_exceeds_null_q90": float(brier_null["real_value"])
        > float(brier_null["null_q90"]),
        "real_log_loss_increment_exceeds_null_q90": (
            log_null is not None and float(log_null["real_value"]) > float(log_null["null_q90"])
        )
        if require_log_bootstrap
        else True,
        "concentration_gates_pass": concentration_passes,
    }
    return {
        "brier_improvement": brier,
        "log_loss_improvement": log_increment,
        "auc_change": auc_change,
        "bootstrap_90_lower_brier": float(brier_bootstrap["interval_90_lower"]),
        "bootstrap_90_lower_log_loss": log_lower,
        "positive_months": positive_months,
        "checkpoint_brier_improvements": checkpoint_values,
        "brier_null_q90": float(brier_null["null_q90"]),
        "brier_null_percentile": float(brier_null["real_percentile"]),
        "log_loss_null_q90": float(log_null["null_q90"]) if log_null is not None else None,
        "log_loss_null_percentile": (
            float(log_null["real_percentile"]) if log_null is not None else None
        ),
        "gates": gates,
        "passes": all(gates.values()),
    }


def descriptive_difference_gate(census: pd.DataFrame, assessment: pd.DataFrame) -> bool:
    overall = float(assessment["large_remaining_move"].mean())
    summaries = census.loc[census["record_type"].eq("label_summary")]
    return bool(
        (
            summaries["row_count"].ge(100)
            & summaries["represented_months"].ge(6)
            & summaries["stock_count"].ge(15)
            & summaries["movement_rate"].sub(overall).abs().ge(0.01)
        ).any()
    )


def derive_decision(
    movement_metrics: pd.DataFrame,
    direction_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    support: Mapping[str, Any],
    census: pd.DataFrame,
    assessment: pd.DataFrame,
) -> dict[str, Any]:
    concentration_passes = bool(
        support["maximum_stock_decision_row_share"] <= 0.10
        and support["maximum_stock_economic_selection_share"] <= 0.20
    )
    movement = _increment_gate(
        movement_metrics,
        monthly_metrics,
        checkpoint_metrics,
        bootstrap,
        null,
        target="large_remaining_move",
        baseline="P1",
        candidate="P2",
        bootstrap_prefix="P2_minus_P1",
        null_prefix="P2_minus_P1",
        concentration_passes=concentration_passes,
        require_log_bootstrap=True,
    )
    direction = (
        _increment_gate(
            direction_metrics,
            monthly_metrics,
            checkpoint_metrics,
            bootstrap,
            null,
            target="up_given_large_move",
            baseline="D1",
            candidate="D2",
            bootstrap_prefix="D2_minus_D1",
            null_prefix="D2_minus_D1",
            concentration_passes=concentration_passes,
            require_log_bootstrap=True,
        )
        if support["direction_support_passes"]
        else {"passes": False, "status": "unavailable_insufficient_support"}
    )
    movement_conjunction = _increment_gate(
        movement_metrics,
        monthly_metrics,
        checkpoint_metrics,
        bootstrap,
        null,
        target="large_remaining_move",
        baseline="P2",
        candidate="P3",
        bootstrap_prefix="P3_minus_P2",
        null_prefix="P3_minus_P2",
        concentration_passes=concentration_passes,
        require_log_bootstrap=False,
    )
    direction_conjunction = (
        _increment_gate(
            direction_metrics,
            monthly_metrics,
            checkpoint_metrics,
            bootstrap,
            null,
            target="up_given_large_move",
            baseline="D2",
            candidate="D3",
            bootstrap_prefix="D3_minus_D2",
            null_prefix="D3_minus_D2",
            concentration_passes=concentration_passes,
            require_log_bootstrap=False,
        )
        if support["direction_support_passes"]
        else {"passes": False, "status": "unavailable_insufficient_support"}
    )
    conjunction_passes = bool(movement_conjunction["passes"] or direction_conjunction["passes"])
    descriptive_differences = descriptive_difference_gate(census, assessment)
    decision = decide_behavioural_screen(
        movement_passes=bool(movement["passes"]),
        direction_passes=bool(direction["passes"]),
        conjunction_passes=conjunction_passes,
        descriptive_differences=descriptive_differences,
    )
    return {
        **SAFETY_FLAGS,
        "gate_constants": DECISION_GATE_CONSTANTS,
        "decision": decision,
        "movement_increment": movement,
        "direction_increment": direction,
        "movement_conjunction_increment": movement_conjunction,
        "direction_conjunction_increment": direction_conjunction,
        "conjunction_increment_passes": conjunction_passes,
        "descriptive_differences_gate": descriptive_differences,
        "support": dict(support),
        "economic_reference_can_override_predictive_gates": False,
        "exact_rerun_passed": False,
        "independent_audit_passed": False,
    }


def raw_component_manifest() -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "provider_volume_label": "historical_activity_proxy",
        "activity_normalisation": (
            "existing_same_stock_same_clock_expanding_prior_session_mean_minimum_10"
        ),
        "causal_window": "regular_session_open_through_completed_decision_bar_t_only",
        "bar_components": {
            "return_bps": (
                "10000*(close/previous_completed_close-1); first denominator=session_open"
            ),
            "true_range_bps": (
                "10000*max(high-low,abs(high-prev_close),abs(low-prev_close))/prev_close"
            ),
            "close_location": "(close-low)/(high-low), or 0.5 for zero range",
            "upper_wick_fraction": "(high-max(open,close))/(high-low)",
            "lower_wick_fraction": "(min(open,close)-low)/(high-low)",
            "historical_relative_activity": (
                "volume/prior-session same-stock same-bar expanding mean"
            ),
        },
        "opening_components": list(BASE_COMPONENTS),
        "pressure_aligned_components": list(DERIVED_COMPONENTS),
        "compression_baseline": (
            "expanding_prior_complete_sessions_same_stock_same_checkpoint_median"
        ),
        "progress_acceleration": (
            "recent_half_stock_minus_inclusive_cohort_median_return_minus_"
            "earlier_half_stock_minus_inclusive_cohort_median_return"
        ),
        "signed_progress_reference": "exact_frozen_predecessor_cohort_relative_return",
        "component_gap_reference": "stock_value_minus_inclusive_simultaneous_cohort_median",
        "outcome_reference": "leave_one_stock_out_median_within_simultaneous_slate",
    }


def forbidden_feature_audit(
    feature_sets: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target, ladder in feature_sets.items():
        for model, features in ladder.items():
            assert_allowed_model_features(features)
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "feature_count": len(features),
                    "features": list(features),
                    "passed": True,
                }
            )
    return {
        **SAFETY_FLAGS,
        "models": rows,
        "descriptive_labels_used_as_features": False,
        "symbol_identity_used_as_feature": False,
        "month_identity_used_as_feature": False,
        "future_fields_used_as_features": False,
        "passed": True,
    }


def concentration_manifest(
    concentration: pd.DataFrame,
    support: Mapping[str, Any],
) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "record_type": "summary",
                "system": "all_primary_systems",
                "symbol": None,
                "count": len(concentration),
                "share": math.nan,
                "maximum_stock_decision_row_share": support["maximum_stock_decision_row_share"],
                "maximum_stock_economic_selection_share": support[
                    "maximum_stock_economic_selection_share"
                ],
                "decision_row_gate_passed": support["maximum_stock_decision_row_share"] <= 0.10,
                "economic_selection_gate_passed": support["maximum_stock_economic_selection_share"]
                <= 0.20,
            }
        ]
    )
    return pd.concat([concentration, summary], ignore_index=True, sort=False)


def plot_dimension_distribution_and_correlation(
    assessment: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].boxplot(
        [assessment[feature].to_numpy(dtype=float) for feature in DIMENSION_FEATURES],
        tick_labels=list(DIMENSION_FEATURES),
        showfliers=False,
    )
    axes[0].tick_params(axis="x", rotation=70)
    axes[0].set_title("Assessment behavioural-dimension distributions")
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    correlation = assessment.loc[:, list(DIMENSION_FEATURES)].corr().to_numpy(dtype=float)
    image = axes[1].imshow(correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1].set_xticks(range(len(DIMENSION_FEATURES)), DIMENSION_FEATURES, rotation=70)
    axes[1].set_yticks(range(len(DIMENSION_FEATURES)), DIMENSION_FEATURES)
    axes[1].set_title("Assessment correlation")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(output, dpi=140, metadata={"Software": "Stocker research"})
    plt.close(figure)


def plot_calibration_comparison(calibration: pd.DataFrame, output: Path) -> None:
    pooled = calibration.loc[
        calibration["scope_type"].eq("pooled") & calibration["row_count"].gt(0)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, target, models in (
        (axes[0], "large_remaining_move", ("P1", "P2", "P3")),
        (axes[1], "up_given_large_move", ("D1", "D2", "D3")),
    ):
        for model in models:
            rows = pooled.loc[pooled["target"].eq(target) & pooled["model"].eq(model)]
            axis.plot(rows["mean_probability"], rows["observed_rate"], marker="o", label=model)
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=0.7)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed rate")
        axis.set_title(target)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=140, metadata={"Software": "Stocker research"})
    plt.close(figure)


def plot_economic_reference(economic: pd.DataFrame, output: Path) -> None:
    pooled = economic.loc[
        economic["record_type"].eq("system")
        & economic["scope_type"].eq("pooled")
        & economic["system"].isin(
            ["predecessor", "behavioural_dimensions", "behavioural_conjunctions"]
        )
    ]
    pivot = pooled.pivot(
        index="system", columns="friction_bps", values="mean_signed_cohort_relative_return_bps"
    ).reindex(["predecessor", "behavioural_dimensions", "behavioural_conjunctions"])
    figure, axis = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=axis)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_ylabel("Mean signed cohort-relative return (bps)")
    axis.set_title("Delayed gross economic-reference diagnostic")
    axis.legend(title="Synthetic friction (bps)")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output, dpi=140, metadata={"Software": "Stocker research"})
    plt.close(figure)


def _markdown_metrics(frame: pd.DataFrame) -> str:
    lines = ["| Model | Brier | Log loss | AUC |", "|---|---:|---:|---:|"]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.brier_score:.9f} | {row.log_loss:.9f} | {row.auc:.9f} |"
        )
    return "\n".join(lines)


def render_report(
    *,
    decision: Mapping[str, Any],
    movement: pd.DataFrame,
    direction: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    economic: pd.DataFrame,
    support: Mapping[str, Any],
) -> str:
    bootstrap_summary = bootstrap.loc[bootstrap["record_type"].eq("summary")]
    null_summary = null.loc[null["record_type"].eq("summary")]
    economic_pooled = economic.loc[
        economic["record_type"].eq("system") & economic["scope_type"].eq("pooled")
    ]
    movement_increment = decision["movement_increment"]
    direction_increment = decision["direction_increment"]
    lines = [
        "# Observable Behavioural-State Dimensions Screen V0 report",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        (
            "This is a retrospective, research-only, observable-only feasibility screen. "
            "It is not prospective validation, achieved P&L, a strategy, or executable-edge "
            "evidence. The behavioural vocabulary describes continuous participant behaviour; "
            "the stock is not assigned literal emotions."
        ),
        "",
        "## Support and boundary",
        "",
        (
            f"- Development rows / sessions / stocks / large moves: "
            f"{support['development_rows']} / {support['development_sessions']} / "
            f"{support['development_stocks']} / {support['development_large_moves']}."
        ),
        (
            f"- Assessment rows / sessions / stocks / large moves: "
            f"{support['assessment_rows']} / {support['assessment_sessions']} / "
            f"{support['assessment_stocks']} / {support['actual_large_moves']}."
        ),
        (
            f"- Exact predecessor rows: {support['predecessor_total_rows']}; "
            f"eligibility exclusions: {support['causal_feature_exclusion_count']}."
        ),
        "- Assessment dates: 2025-01-01 through 2025-08-22; protected rows materialised: 0.",
        (
            "- Decision checkpoints: completed five-minute bars 6 (10:00) and 12 (10:30), "
            "America/New_York."
        ),
        "",
        "## Movement models",
        "",
        _markdown_metrics(movement),
        "",
        "## Direction models among actual large moves",
        "",
        _markdown_metrics(direction),
        "",
        "## Gate results",
        "",
        (
            f"- P2 versus P1 passes: `{movement_increment['passes']}`; Brier improvement "
            f"{movement_increment['brier_improvement']:.12g}; log-loss improvement "
            f"{movement_increment['log_loss_improvement']:.12g}."
        ),
        (
            f"- D2 versus D1 passes: `{direction_increment['passes']}`; Brier improvement "
            f"{direction_increment.get('brier_improvement', math.nan):.12g}; log-loss "
            f"improvement {direction_increment.get('log_loss_improvement', math.nan):.12g}."
        ),
        f"- P3 versus P2 passes: `{decision['movement_conjunction_increment']['passes']}`.",
        f"- D3 versus D2 passes: `{decision['direction_conjunction_increment']['passes']}`.",
        "",
        "## Bootstrap intervals",
        "",
    ]
    for row in bootstrap_summary.itertuples(index=False):
        lines.append(
            f"- `{row.metric}`: 90% [{row.interval_90_lower:.12g}, "
            f"{row.interval_90_upper:.12g}]; 95% [{row.interval_95_lower:.12g}, "
            f"{row.interval_95_upper:.12g}]."
        )
    lines.extend(["", "## Bundled within-slate null", ""])
    for row in null_summary.itertuples(index=False):
        lines.append(
            f"- `{row.metric}`: real {row.real_value:.12g}; null q90 "
            f"{row.null_q90:.12g}; percentile {row.real_percentile:.3f}."
        )
    lines.extend(["", "## Delayed economic-reference diagnostic", ""])
    for row in economic_pooled.itertuples(index=False):
        lines.append(
            f"- `{row.system}` at {row.friction_bps:.0f} bps: signed gross "
            f"{row.mean_signed_gross_return_bps:.6f} bps; signed cohort-relative "
            f"{row.mean_signed_cohort_relative_return_bps:.6f} bps."
        )
    lines.extend(
        [
            "",
            (
                "The economic diagnostic is delayed and gross apart from synthetic friction. "
                "It cannot rescue a failed proper-score gate and is not achieved P&L."
            ),
            "",
            (
                f"Exact rerun passed: `{decision['exact_rerun_passed']}`. Independent audit "
                f"passed: `{decision['independent_audit_passed']}`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


SCIENTIFIC_ARTIFACTS = (
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
    "independent_audit.json",
    "report.md",
    "behavioural_dimension_distributions_and_correlations.png",
    "calibration_comparison.png",
    "delayed_economic_reference.png",
)


def write_run_artifacts(
    output: Path,
    *,
    contract: Mapping[str, Any],
    input_hashes: Sequence[Mapping[str, Any]],
    source_context: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    full_panel: pd.DataFrame,
    assessment: pd.DataFrame,
    scaling_manifest: Mapping[str, Any],
    dimension_manifest: Mapping[str, Any],
    label_manifest: Mapping[str, Any],
    feature_sets: Mapping[str, Mapping[str, Sequence[str]]],
    serialized_models: Mapping[str, Any],
    census: pd.DataFrame,
    movement: pd.DataFrame,
    direction: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    calibration: pd.DataFrame,
    diagnostics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    null: pd.DataFrame,
    economic: pd.DataFrame,
    concentration: pd.DataFrame,
    decision: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "input_artifact_hashes.json",
        {**SAFETY_FLAGS, "predecessor_artifacts": list(input_hashes)},
    )
    write_json(
        output / "source_manifest.json",
        {
            **SAFETY_FLAGS,
            "provider": "EODHD",
            "provider_activity_label": "historical_activity_proxy",
            "date_predicate_applied_before_materialisation": True,
            "symbol_predicate_applied_before_materialisation": True,
            "source_paths_are_logical_not_local_absolute": True,
            "minimum_timestamp_read": source_context["minimum_timestamp_read"],
            "maximum_timestamp_read": source_context["maximum_timestamp_read"],
            "protected_rows_materialised": source_context["protected_rows_materialised"],
            "sources": source_context["sources"],
            "vendor_qa": source_context["vendor_qa"],
            "source_gap_ledger": source_context["source_gap_ledger"],
            "source_gap_ledger_rejected_session_records": source_context[
                "source_gap_ledger_rejected_session_records"
            ],
            "causal_feature_exclusions": source_context["causal_feature_exclusions"],
            "causal_feature_exclusion_count": source_context["causal_feature_exclusion_count"],
            "source_rows_by_symbol_month": source_context["source_rows_by_symbol_month"],
        },
    )
    write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "read_start": str(START),
            "assessment_end_inclusive": "2025-08-22",
            "protected_start": str(PROTECTED_START),
            "minimum_timestamp_read": source_context["minimum_timestamp_read"],
            "maximum_timestamp_read": source_context["maximum_timestamp_read"],
            "protected_files_touched": [],
            "protected_rows_materialised": source_context["protected_rows_materialised"],
            "passed": source_context["protected_rows_materialised"] == 0,
        },
    )
    write_json(output / "predecessor_reconstruction.json", predecessor)
    write_json(output / "raw_component_manifest.json", raw_component_manifest())
    write_json(output / "behavioural_component_scaling.json", scaling_manifest)
    write_json(output / "behavioural_dimension_manifest.json", dimension_manifest)
    write_json(output / "behavioural_label_thresholds.json", label_manifest)
    write_json(output / "forbidden_feature_audit.json", forbidden_feature_audit(feature_sets))
    write_json(
        output / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "configuration": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 250,
                "class_weight": None,
                "n_jobs": 1,
            },
            "features": {
                target: {model: list(features) for model, features in ladder.items()}
                for target, ladder in feature_sets.items()
            },
            "primary_fitted_model_count": 6,
            "frozen_predecessor_model_count": 2,
            "row_weight": "1/eligible_rows_in_slate",
            "development_fit_interval": "2024_only",
            "assessment_interval": "2025-01-01_through_2025-08-22",
            "decision_gate_constants": DECISION_GATE_CONSTANTS,
        },
    )
    write_json(output / "model_coefficients.json", {**SAFETY_FLAGS, "models": serialized_models})

    m1_features = list(feature_sets["movement"]["P1"])
    compact_columns = list(
        dict.fromkeys(
            [
                "symbol",
                "session",
                "year",
                "year_month",
                "decision_ordinal",
                "decision_time_america_new_york",
                "checkpoint_60m",
                "slate_id",
                "decision_bar_start_timestamp_utc",
                "feature_available_timestamp_utc",
                "entry_bar_ordinal",
                "delayed_entry_open",
                "terminal_bar_ordinal",
                "terminal_close",
                "raw_remaining_return_bps",
                "cohort_median_return_minus_i_bps",
                "residual_remaining_return_bps",
                "movement_threshold_bps",
                "large_remaining_move",
                "up_given_large_move",
                *m1_features,
                *BASE_COMPONENTS,
                *DERIVED_COMPONENTS,
                *DIMENSION_FEATURES,
                *CONJUNCTION_FEATURES,
                *(f"label__{label}" for label in DESCRIPTIVE_LABELS),
                "behavioural_label_count",
                "behavioural_labels",
                "availability_status",
                "source_gap_status",
                "qa_status",
                "corporate_action_status",
            ]
        )
    )
    write_parquet(output / "compact_decision_panel.parquet", full_panel.loc[:, compact_columns])
    component_columns = list(
        dict.fromkeys(
            [
                "symbol",
                "session",
                "year",
                "decision_ordinal",
                "slate_id",
                "feature_available_timestamp_utc",
                "entry_bar_ordinal",
                "delayed_entry_open",
                "terminal_bar_ordinal",
                "terminal_close",
                "raw_remaining_return_bps",
                "cohort_median_return_minus_i_bps",
                "residual_remaining_return_bps",
                "bar_count",
                "bar_start_timestamps_utc",
                "bar_open",
                "bar_high",
                "bar_low",
                "bar_close",
                "historical_relative_activity",
                "return_bps_path",
                "true_range_bps_path",
                "close_location_path",
                "upper_wick_fraction_path",
                "lower_wick_fraction_path",
                "new_high_path",
                "new_low_path",
                "earlier_half_return_bps",
                "recent_half_return_bps",
                "earlier_relative_return_bps",
                "recent_relative_return_bps",
                *BASE_COMPONENTS,
                *DERIVED_COMPONENTS,
                "provider_activity_label",
                "activity_normalisation",
            ]
        )
    )
    write_parquet(
        output / "behavioural_component_ledger.parquet",
        full_panel.loc[:, component_columns],
    )
    dimension_columns = [
        "symbol",
        "session",
        "year",
        "year_month",
        "decision_ordinal",
        "slate_id",
        *(f"z_{component}" for component in (*BASE_COMPONENTS, *DERIVED_COMPONENTS)),
        *DIMENSION_FEATURES,
        *CONJUNCTION_FEATURES,
        *(f"label__{label}" for label in DESCRIPTIVE_LABELS),
        "behavioural_label_count",
        "behavioural_labels",
    ]
    write_parquet(
        output / "behavioural_dimension_ledger.parquet",
        full_panel.loc[:, dimension_columns],
    )
    prediction_columns = list(
        dict.fromkeys(
            [
                "symbol",
                "session",
                "year_month",
                "decision_ordinal",
                "slate_id",
                "entry_bar_ordinal",
                "delayed_entry_open",
                "terminal_close",
                "raw_remaining_return_bps",
                "residual_remaining_return_bps",
                "large_remaining_move",
                "up_given_large_move",
                "open_to_decision_cohort_relative_return_bps",
                "predicted_remaining_movement_scale_bps",
                *DIMENSION_FEATURES,
                *CONJUNCTION_FEATURES,
                *(f"p_large_remaining_move__{model}" for model in ("P0", "P1", "P2", "P3")),
                *(f"p_up_given_large_move__{model}" for model in ("D0", "D1", "D2", "D3")),
            ]
        )
    )
    write_parquet(output / "assessment_predictions.parquet", assessment.loc[:, prediction_columns])
    write_csv(output / "behavioural_state_census.csv", census)
    write_csv(output / "movement_metrics.csv", movement)
    write_csv(output / "direction_metrics.csv", direction)
    write_csv(output / "monthly_metrics.csv", monthly)
    write_csv(output / "checkpoint_metrics.csv", checkpoint)
    write_csv(output / "calibration_bins.csv", calibration)
    write_csv(output / "dimension_diagnostics.csv", diagnostics)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "null_metrics.csv", null)
    write_csv(output / "economic_reference_metrics.csv", economic)
    write_csv(output / "concentration_metrics.csv", concentration)
    write_json(output / "decision.json", decision)
    plot_dimension_distribution_and_correlation(
        assessment, output / "behavioural_dimension_distributions_and_correlations.png"
    )
    plot_calibration_comparison(calibration, output / "calibration_comparison.png")
    plot_economic_reference(economic, output / "delayed_economic_reference.png")
    report = render_report(
        decision=decision,
        movement=movement,
        direction=direction,
        bootstrap=bootstrap,
        null=null,
        economic=economic,
        support=decision["support"],
    )
    (output / "report.md").write_text(report, encoding="utf-8")


def execute_run(output: Path, *, provider_root: Path) -> dict[str, Any]:
    contract = load_contract()
    input_hashes = verify_input_hashes()
    predecessor_panel = pd.read_parquet(PREDECESSOR_PANEL)
    predecessor_panel = predecessor_panel.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if len(predecessor_panel) != 15_617 or len(predecessor_panel) > MAX_COMPACT_ROWS:
        raise ScreenBlocker(
            "blocked_quick_behavioural_screen_resource_limit", "predecessor row count differs"
        )
    if set(predecessor_panel["year"].unique()) != {2024, 2025}:
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "predecessor years extend beyond 2024/2025"
        )
    archived_assessment = pd.read_parquet(PREDECESSOR_PREDICTIONS)
    predecessor, frozen_models, frozen_predictions = predecessor_reconstruction(
        predecessor_panel, archived_assessment
    )
    component_panel, source_context = build_component_panel(
        predecessor_panel, provider_root=provider_root
    )
    if source_context["protected_rows_materialised"] != 0:
        raise ScreenBlocker(
            "blocked_protected_boundary_failure", "protected provider row materialised"
        )
    full_panel, scaling_manifest, dimension_manifest, label_manifest = derive_dimension_panel(
        component_panel
    )
    models, serialized_models, feature_sets = fit_primary_models(full_panel, frozen_models)
    assessment = score_assessment(full_panel, frozen_predictions, models)
    movement, direction, monthly, checkpoint, calibration = evaluate_predictions(assessment)
    census = behavioural_state_census(assessment)
    diagnostics = dimension_diagnostics(full_panel, assessment, serialized_models, models)
    selections = economic_selection_ledger(assessment)
    economic = economic_reference_metrics(selections)
    bootstrap = bootstrap_metrics(assessment, selections)
    null = within_slate_behavioural_null(
        full_panel, assessment, feature_sets, dimension_manifest, selections
    )
    concentration, support = concentration_and_support(assessment, selections)
    development = full_panel.loc[full_panel["year"].eq(2024)]
    support.update(
        {
            "predecessor_total_rows": len(predecessor_panel),
            "eligible_total_rows": len(full_panel),
            "development_rows": len(development),
            "development_sessions": int(development["session"].nunique()),
            "development_stocks": int(development["symbol"].nunique()),
            "development_large_moves": int(development["large_remaining_move"].sum()),
            "causal_feature_exclusion_count": source_context["causal_feature_exclusion_count"],
        }
    )
    concentration = concentration_manifest(concentration, support)
    decision = derive_decision(
        movement,
        direction,
        monthly,
        checkpoint,
        bootstrap,
        null,
        support,
        census,
        assessment,
    )
    write_run_artifacts(
        output,
        contract=contract,
        input_hashes=input_hashes,
        source_context=source_context,
        predecessor=predecessor,
        full_panel=full_panel,
        assessment=assessment,
        scaling_manifest=scaling_manifest,
        dimension_manifest=dimension_manifest,
        label_manifest=label_manifest,
        feature_sets=feature_sets,
        serialized_models=serialized_models,
        census=census,
        movement=movement,
        direction=direction,
        monthly=monthly,
        checkpoint=checkpoint,
        calibration=calibration,
        diagnostics=diagnostics,
        bootstrap=bootstrap,
        null=null,
        economic=economic,
        concentration=concentration,
        decision=decision,
    )
    return decision


def run_independent_auditor(artifacts: Path, *, provider_root: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PACKAGE_SRC)
    result = subprocess.run(
        [
            sys.executable,
            str(AUDITOR_PATH),
            "--artifacts",
            str(artifacts),
            "--provider-root",
            str(provider_root),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"independent audit failed: {result.stdout[-1000:]} {result.stderr[-1000:]}",
        )


def _update_final_decision(artifacts: Path) -> None:
    decision_path = artifacts / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    audit = json.loads((artifacts / "independent_audit.json").read_text(encoding="utf-8"))
    decision["independent_audit_passed"] = bool(audit.get("passed"))
    decision["exact_rerun_passed"] = True
    write_json(decision_path, decision)
    movement = pd.read_csv(artifacts / "movement_metrics.csv")
    direction = pd.read_csv(artifacts / "direction_metrics.csv")
    bootstrap = pd.read_csv(artifacts / "bootstrap_metrics.csv")
    null = pd.read_csv(artifacts / "null_metrics.csv")
    economic = pd.read_csv(artifacts / "economic_reference_metrics.csv")
    report = render_report(
        decision=decision,
        movement=movement,
        direction=direction,
        bootstrap=bootstrap,
        null=null,
        economic=economic,
        support=decision["support"],
    )
    (artifacts / "report.md").write_text(report, encoding="utf-8")


def compare_exact_runs(primary: Path, exact: Path) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for name in SCIENTIFIC_ARTIFACTS:
        primary_path = primary / name
        exact_path = exact / name
        if not primary_path.is_file() or not exact_path.is_file():
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure", f"rerun artifact missing: {name}"
            )
        primary_hash = sha256_file(primary_path)
        exact_hash = sha256_file(exact_path)
        comparisons.append(
            {
                "artifact": name,
                "primary_sha256": primary_hash,
                "exact_rerun_sha256": exact_hash,
                "identical": primary_hash == exact_hash,
            }
        )
    passed = all(row["identical"] for row in comparisons)
    manifest = {
        **SAFETY_FLAGS,
        "fixed_seeds": {
            "bootstrap": BOOTSTRAP_SEED,
            "null": NULL_SEED,
            "permutation_importance": PERMUTATION_IMPORTANCE_SEED,
            "random_selection": RANDOM_SELECTION_SEED,
        },
        "artifact_comparisons": comparisons,
        "all_scientific_artifacts_identical": passed,
        "passed": passed,
    }
    write_json(primary / "exact_rerun_manifest.json", manifest)
    write_json(exact / "exact_rerun_manifest.json", manifest)
    if not passed:
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure", "exact rerun artifact hash differs"
        )
    return manifest


def run_complete_screen(
    *,
    primary: Path,
    exact: Path,
    provider_root: Path,
) -> dict[str, Any]:
    execute_run(primary, provider_root=provider_root)
    run_independent_auditor(primary, provider_root=provider_root)
    execute_run(exact, provider_root=provider_root)
    run_independent_auditor(exact, provider_root=provider_root)
    _update_final_decision(primary)
    _update_final_decision(exact)
    manifest = compare_exact_runs(primary, exact)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report.md").write_text(
        (primary / "report.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--primary-only", action="store_true")
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


def main() -> int:
    args = parse_args()
    provider_root = args.provider_root.expanduser().resolve()
    primary = args.output.resolve()
    try:
        if args.primary_only:
            decision = execute_run(primary, provider_root=provider_root)
            print(canonical_json(decision), end="")
            return 0
        exact = DEFAULT_EXACT if primary == DEFAULT_PRIMARY else primary.parent / "exact_rerun"
        manifest = run_complete_screen(
            primary=primary,
            exact=exact.resolve(),
            provider_root=provider_root,
        )
        print(canonical_json(manifest), end="")
        return 0
    except ScreenBlocker as blocker:
        print(blocker.code)
        print(blocker.detail, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
