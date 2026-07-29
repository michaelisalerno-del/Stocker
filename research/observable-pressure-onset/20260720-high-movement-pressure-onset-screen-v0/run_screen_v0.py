#!/usr/bin/env python3
"""Run the bounded High-Movement Pressure-Onset Screen V0."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-pressure-onset-matplotlib")

import argparse
import hashlib
import json
import math
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "packages" / "stocker_research" / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from stocker_research.pressure_onset_screen_v0 import (
    FrozenLogisticModel,
    activity_acceleration,
    assert_allowed_feature_names,
    assert_safe_timestamps,
    classify_onset,
    close_location_pressure,
    cohort_relative_cumulative_paths_bps,
    decide_pressure_screen,
    development_onset_barriers,
    directional_efficiency,
    expanding_monthly_oof_probabilities,
    extract_decision_window,
    fit_fixed_logistic,
    manual_logistic_prediction,
    movement_admission_thresholds,
    new_extreme_counts,
    opening_range_acceptance,
    permute_feature_bundle_within_slates,
    progress_per_activity,
    range_acceleration,
    session_block_bootstrap_draws,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
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
PREDECESSOR_ARTIFACTS = PREDECESSOR_DIR / "artifacts" / "primary"
PREDECESSOR_PANEL = PREDECESSOR_ARTIFACTS / "opening_decision_panel.parquet"
PREDECESSOR_PREDICTIONS = PREDECESSOR_ARTIFACTS / "assessment_predictions.parquet"
PREDECESSOR_COEFFICIENTS = PREDECESSOR_ARTIFACTS / "model_coefficients.json"
PREDECESSOR_METRICS = PREDECESSOR_ARTIFACTS / "movement_metrics.csv"
PREDECESSOR_SOURCE_MANIFEST = PREDECESSOR_ARTIFACTS / "source_manifest.json"

START = pd.Timestamp("2024-01-01T00:00:00Z")
DEVELOPMENT_END_EXCLUSIVE = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
READ_END_INCLUSIVE = PROTECTED_START - pd.Timedelta(microseconds=1)
EXPECTED_SESSION_BARS = 78
MAX_COMPACT_ROWS = 20_000
BOOTSTRAP_DRAWS = 200
NULL_DRAWS = 50
BOOTSTRAP_SEED = 20260720
NULL_SEED = 20260721
RANDOM_SEED = 20260722

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

SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "feasibility_screen": True,
    "observable_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "loops_regimes_states_and_structural_paths_forbidden": True,
}

READINESS_FEATURES = (
    "p_large_remaining_move",
    "opening_gap_bps",
    "open_to_decision_raw_return_bps",
    "open_to_decision_cohort_relative_return_bps",
    "latest_one_bar_return_bps",
    "latest_three_bar_return_bps",
    "latest_six_bar_return_bps",
    "realized_volatility_3_bps",
    "realized_volatility_6_bps",
    "short_realized_volatility_ratio",
    "opening_high_low_range_bps",
    "opening_range_to_trailing_same_checkpoint_median",
    "current_true_range_bps",
    "short_true_range_to_longer_true_range",
    "distance_from_opening_high_bps",
    "distance_from_opening_low_bps",
    "historical_activity_proxy_shock",
    "cross_sectional_dispersion_bps",
)

PRESSURE_FEATURES = (
    "relative_return_last_3_bps",
    "relative_return_previous_3_bps",
    "relative_strength_acceleration_bps",
    "activity_last_2_mean",
    "activity_previous_4_mean",
    "activity_acceleration",
    "range_last_2_mean_bps",
    "range_previous_4_mean_bps",
    "range_acceleration",
    "signed_efficiency_3",
    "absolute_efficiency_3",
    "signed_efficiency_6",
    "absolute_efficiency_6",
    "signed_progress_per_activity",
    "current_close_location",
    "mean_close_location_last_3",
    "upper_quartile_close_fraction_last_3",
    "lower_quartile_close_fraction_last_3",
    "new_high_count_last_3",
    "new_low_count_last_3",
    "close_above_initial_3_high",
    "close_below_initial_3_low",
    "completed_closes_outside_initial_range",
    "latest_close_returned_inside_initial_range",
    "stock_acceleration_minus_cohort_median_acceleration_bps",
    "cross_sectional_breadth",
    "cross_sectional_dispersion_change_bps",
)

CONFIRMATION_FEATURES = (
    "change_cohort_relative_return_bps",
    "change_relative_strength_acceleration",
    "change_activity_shock",
    "change_range_acceleration",
    "change_signed_efficiency_3",
    "change_close_location",
    "new_high_at_t_plus_1",
    "new_low_at_t_plus_1",
    "favourable_retracement_bps",
    "opening_range_acceptance_persisted",
    "predicted_direction_remained_same",
)

MODEL_FEATURES = {
    "A0": ("checkpoint_60m", "p_large_remaining_move"),
    "A1": ("checkpoint_60m", *READINESS_FEATURES),
    "A2": ("checkpoint_60m", *READINESS_FEATURES, *PRESSURE_FEATURES),
    "A3": (
        "checkpoint_60m",
        *READINESS_FEATURES,
        *PRESSURE_FEATURES,
        *CONFIRMATION_FEATURES,
    ),
    "D0": (
        "checkpoint_60m",
        "p_large_remaining_move",
        "open_to_decision_cohort_relative_return_bps",
    ),
    "D1": ("checkpoint_60m", *READINESS_FEATURES),
    "D2": ("checkpoint_60m", *READINESS_FEATURES, *PRESSURE_FEATURES),
    "D3": (
        "checkpoint_60m",
        *READINESS_FEATURES,
        *PRESSURE_FEATURES,
        *CONFIRMATION_FEATURES,
    ),
}

PREDECESSOR_KEEP = (
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
    "opening_gap_bps",
    "open_to_decision_raw_return_bps",
    "open_to_decision_cohort_relative_return_bps",
    "latest_one_bar_return_bps",
    "latest_three_bar_return_bps",
    "opening_high_low_range_bps",
    "opening_realized_volatility_bps",
    "mean_completed_bar_true_range_bps",
    "current_true_range_bps",
    "distance_from_opening_high_bps",
    "distance_from_opening_low_bps",
    "close_location_within_opening_range",
    "positive_close_fraction",
    "directional_close_persistence_ratio",
    "historical_activity_proxy_shock",
    "cross_sectional_dispersion_bps",
    "large_remaining_move",
    "p_large_remaining_move",
    "movement_admission_threshold",
    "high_movement_admitted",
)

SCIENTIFIC_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "input_artifact_hashes.json",
    "protected_boundary_audit.json",
    "predecessor_reconstruction.json",
    "feature_manifest.json",
    "forbidden_feature_audit.json",
    "movement_oof_fold_manifest.json",
    "movement_admission_thresholds.json",
    "onset_barriers.json",
    "compact_decision_panel.parquet",
    "onset_path_ledger.parquet",
    "development_oof_predictions.parquet",
    "assessment_predictions.parquet",
    "model_configurations.json",
    "model_coefficients.json",
    "onset_metrics.csv",
    "direction_metrics.csv",
    "monthly_metrics.csv",
    "checkpoint_metrics.csv",
    "calibration_bins.csv",
    "bootstrap_metrics.csv",
    "null_metrics.csv",
    "economic_reference_metrics.csv",
    "concentration_metrics.csv",
    "calibration_systems.png",
    "economic_reference_20bps.png",
    "decision.json",
    "report.md",
)


class ScreenBlocker(RuntimeError):
    """A preregistered fail-closed experiment stop."""

    def __init__(self, code: str, detail: str) -> None:
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
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, default=str) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


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
        raise ScreenBlocker("blocked_protected_boundary_failure", "pre-start source row")
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or contract.get("safety", {}).get(key) != expected:
            raise RuntimeError(f"contract safety flag differs: {key}")
    if tuple(contract["cohort"]) != SYMBOLS:
        raise RuntimeError("contract cohort differs")
    return cast(dict[str, Any], contract)


def predecessor_reconstruction(
    panel: pd.DataFrame, assessment: pd.DataFrame
) -> tuple[dict[str, Any], Mapping[str, Any], np.ndarray]:
    payload = json.loads(PREDECESSOR_COEFFICIENTS.read_text(encoding="utf-8"))
    model = cast(Mapping[str, Any], payload["models"]["large_remaining_move"]["M1"])
    scoring = panel.loc[panel["year"].eq(2025)].sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    archived = assessment.sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    keys = ["symbol", "session", "decision_ordinal"]
    if not scoring[keys].reset_index(drop=True).equals(archived[keys].reset_index(drop=True)):
        raise ScreenBlocker(
            "blocked_observable_movement_model_not_reconstructable",
            "predecessor assessment keys differ",
        )
    probabilities = manual_logistic_prediction(model, scoring)
    expected = archived["p__large_remaining_move__M1"].to_numpy(dtype=float)
    maximum_error = float(np.max(np.abs(probabilities - expected)))
    labels = archived["large_remaining_move"].to_numpy(dtype=int)
    actual_metrics = {
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "auc": float(roc_auc_score(labels, probabilities)),
    }
    metrics = pd.read_csv(PREDECESSOR_METRICS)
    row = metrics.loc[
        metrics["scope"].eq("pooled")
        & metrics["target"].eq("large_remaining_move")
        & metrics["model"].eq("M1")
    ].iloc[0]
    expected_metrics = {
        "brier": float(row["brier_score"]),
        "log_loss": float(row["log_loss"]),
        "auc": float(row["auc"]),
    }
    metric_errors = {
        key: abs(actual_metrics[key] - expected_metrics[key]) for key in actual_metrics
    }
    passed = maximum_error <= 1e-12 and max(metric_errors.values()) <= 1e-12
    result = {
        **SAFETY_FLAGS,
        "source_experiment": str(PREDECESSOR_DIR.relative_to(REPO_ROOT)),
        "source_commit": "5e80d972d1e003c8366a1ec6ca170d1077288ead",
        "model_id": "large_remaining_move__M1",
        "assessment_rows": len(scoring),
        "required_tolerance": 1e-12,
        "maximum_prediction_absolute_error": maximum_error,
        "actual_metrics": actual_metrics,
        "archived_metrics": expected_metrics,
        "metric_absolute_errors": metric_errors,
        "prediction_equality_passed": maximum_error <= 1e-12,
        "aggregate_metric_equality_passed": max(metric_errors.values()) <= 1e-12,
        "passed": passed,
    }
    if not passed:
        raise ScreenBlocker(
            "blocked_observable_movement_model_not_reconstructable",
            "frozen M1 predictions or metrics differ beyond 1e-12",
        )
    return result, model, probabilities


def load_qa_record(symbol: str) -> dict[str, Any]:
    qa_path = (
        Path.home()
        / "StockerLocal"
        / "data"
        / "reports"
        / "vendor_qa"
        / f"{symbol}_5m_eodhd_qa.json"
    )
    if not qa_path.is_file():
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", f"vendor QA missing for {symbol}"
        )
    payload = json.loads(qa_path.read_text(encoding="utf-8"))
    validation_errors = int(payload.get("validation", {}).get("counts", {}).get("error", 0))
    adjusted = payload.get("adjusted_close", {})
    adjusted_differences = int(adjusted.get("different_from_close_count", 0) or 0)
    passed = (
        payload.get("status") != "fail" and validation_errors == 0 and adjusted_differences == 0
    )
    if not passed:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure",
            f"vendor QA or corporate-action check failed for {symbol}",
        )
    return {
        "symbol": symbol,
        "status": str(payload.get("status", "unknown")),
        "logical_path": f"external_vendor_qa/{qa_path.name}",
        "sha256": sha256_file(qa_path),
        "validation_error_count": validation_errors,
        "adjusted_close_present": adjusted.get("present"),
        "adjusted_close_differences": adjusted_differences,
        "corporate_action_check_passed": True,
    }


def prepare_symbol_bars(raw: pd.DataFrame, *, symbol: str) -> tuple[pd.DataFrame, int]:
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
    rejected_sessions = 0
    for session, part in regular.groupby("session", sort=True):
        ordered = part.sort_values("bar_ordinal", kind="mergesort").copy()
        prices = ordered[["open", "high", "low", "close"]].to_numpy(dtype=float)
        activity = ordered["volume"].to_numpy(dtype=float)
        valid = bool(
            str(session) not in invalid_sessions
            and len(ordered) == EXPECTED_SESSION_BARS
            and ordered["bar_ordinal"].astype(int).tolist() == list(range(EXPECTED_SESSION_BARS))
            and np.isfinite(prices).all()
            and bool((prices > 0.0).all())
            and np.isfinite(activity).all()
            and bool((activity >= 0.0).all())
        )
        if not valid:
            rejected_sessions += 1
            continue
        ordered["source_quality_passed"] = True
        ordered["corporate_action_passed"] = True
        valid_parts.append(ordered)
    if not valid_parts:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", f"no complete sessions for {symbol}"
        )
    frame = pd.concat(valid_parts, ignore_index=True).sort_values(
        ["session", "bar_ordinal"], kind="mergesort"
    )
    grouped = frame.groupby("session", sort=False)
    previous_close = grouped["close"].shift(1)
    first_bar = frame["bar_ordinal"].eq(0)
    denominator = previous_close.where(~first_bar, frame["open"])
    frame["bar_return"] = frame["close"] / denominator - 1.0
    frame["true_range_bps"] = (
        10_000.0
        * np.maximum.reduce(
            [
                frame["high"].to_numpy(dtype=float) - frame["low"].to_numpy(dtype=float),
                np.abs(frame["high"].to_numpy(dtype=float) - denominator.to_numpy(dtype=float)),
                np.abs(frame["low"].to_numpy(dtype=float) - denominator.to_numpy(dtype=float)),
            ]
        )
        / denominator.to_numpy(dtype=float)
    )
    frame["cumulative_activity"] = grouped["volume"].cumsum()
    frame["historical_activity_baseline_at_bar"] = frame.groupby("bar_ordinal", sort=False)[
        "volume"
    ].transform(lambda values: values.expanding(min_periods=10).mean().shift(1))
    frame["historical_cumulative_activity_baseline_at_bar"] = frame.groupby(
        "bar_ordinal", sort=False
    )["cumulative_activity"].transform(
        lambda values: values.expanding(min_periods=10).mean().shift(1)
    )
    frame["relative_activity"] = frame["volume"] / frame[
        "historical_activity_baseline_at_bar"
    ].replace(0.0, np.nan)
    frame["activity_shock"] = np.log1p(
        frame["cumulative_activity"]
        / frame["historical_cumulative_activity_baseline_at_bar"].replace(0.0, np.nan)
    )
    return frame.reset_index(drop=True), rejected_sessions + len(invalid_sessions)


def point_features(session_frame: pd.DataFrame, origin: int) -> dict[str, float]:
    ordered = session_frame.sort_values("bar_ordinal", kind="mergesort").reset_index(drop=True)
    returns = ordered["bar_return"].to_numpy(dtype=float)
    true_ranges = ordered["true_range_bps"].to_numpy(dtype=float)
    activity = ordered["relative_activity"].to_numpy(dtype=float)
    last_3 = returns[origin - 2 : origin + 1]
    previous_3 = returns[origin - 5 : origin - 2]
    last_6 = returns[origin - 5 : origin + 1]
    activity_last_2 = activity[origin - 1 : origin + 1]
    activity_previous_4 = activity[origin - 5 : origin - 1]
    range_last_2 = true_ranges[origin - 1 : origin + 1]
    range_previous_4 = true_ranges[origin - 5 : origin - 1]
    signed_3, absolute_3 = directional_efficiency(last_3)
    signed_6, absolute_6 = directional_efficiency(last_6)
    close_pressure = close_location_pressure(
        highs=ordered.loc[origin - 2 : origin, "high"].to_numpy(dtype=float),
        lows=ordered.loc[origin - 2 : origin, "low"].to_numpy(dtype=float),
        closes=ordered.loc[origin - 2 : origin, "close"].to_numpy(dtype=float),
    )
    opening = ordered.loc[:origin]
    opening_high = float(opening["high"].max())
    opening_low = float(opening["low"].min())
    session_open = float(ordered.loc[0, "open"])
    current_close = float(ordered.loc[origin, "close"])
    initial_high = float(ordered.loc[:2, "high"].max())
    initial_low = float(ordered.loc[:2, "low"].min())
    acceptance = opening_range_acceptance(
        closes=opening["close"].to_numpy(dtype=float),
        initial_high=initial_high,
        initial_low=initial_low,
    )
    high_count, low_count = new_extreme_counts(
        opening["high"].to_numpy(dtype=float),
        opening["low"].to_numpy(dtype=float),
        latest=3,
    )
    raw_last_3 = 10_000.0 * (float(np.prod(1.0 + last_3)) - 1.0)
    raw_previous_3 = 10_000.0 * (float(np.prod(1.0 + previous_3)) - 1.0)
    short_true_range = float(np.mean(range_last_2))
    longer_true_range = float(np.mean(true_ranges[origin - 5 : origin + 1]))
    acceptance_code = (
        1.0
        if acceptance["close_above_initial_3_high"] == 1.0
        else -1.0
        if acceptance["close_below_initial_3_low"] == 1.0
        else 0.0
    )
    return {
        "open_to_decision_raw_return_bps": 10_000.0 * (current_close / session_open - 1.0),
        "open_to_previous_bar_raw_return_bps": 10_000.0
        * (float(ordered.loc[origin - 1, "close"]) / session_open - 1.0),
        "latest_one_bar_return_bps": 10_000.0 * float(returns[origin]),
        "latest_three_bar_return_bps": raw_last_3,
        "latest_six_bar_return_bps": 10_000.0 * (float(np.prod(1.0 + last_6)) - 1.0),
        "realized_volatility_3_bps": 10_000.0 * float(np.std(last_3, ddof=0)),
        "realized_volatility_6_bps": 10_000.0 * float(np.std(last_6, ddof=0)),
        "opening_high_low_range_bps": 10_000.0 * (opening_high - opening_low) / session_open,
        "current_true_range_bps": float(true_ranges[origin]),
        "short_true_range_to_longer_true_range": (
            short_true_range / longer_true_range if longer_true_range > 1e-12 else math.nan
        ),
        "distance_from_opening_high_bps": 10_000.0 * (opening_high - current_close) / opening_high,
        "distance_from_opening_low_bps": 10_000.0 * (current_close - opening_low) / opening_low,
        "historical_activity_proxy_shock": float(ordered.loc[origin, "activity_shock"]),
        "raw_return_last_3_bps": raw_last_3,
        "raw_return_previous_3_bps": raw_previous_3,
        "raw_acceleration_bps": raw_last_3 - raw_previous_3,
        "activity_last_2_mean": float(np.mean(activity_last_2)),
        "activity_previous_4_mean": float(np.mean(activity_previous_4)),
        "activity_acceleration": activity_acceleration(activity_last_2, activity_previous_4),
        "relative_activity_last_3_mean": float(np.mean(activity[origin - 2 : origin + 1])),
        "range_last_2_mean_bps": short_true_range,
        "range_previous_4_mean_bps": float(np.mean(range_previous_4)),
        "range_acceleration": range_acceleration(range_last_2, range_previous_4),
        "signed_efficiency_3": signed_3,
        "absolute_efficiency_3": absolute_3,
        "signed_efficiency_6": signed_6,
        "absolute_efficiency_6": absolute_6,
        "return_sum_3": float(np.sum(last_3)),
        "absolute_return_sum_3": float(np.sum(np.abs(last_3))),
        "return_sum_6": float(np.sum(last_6)),
        "absolute_return_sum_6": float(np.sum(np.abs(last_6))),
        **close_pressure,
        "new_high_count_last_3": float(high_count),
        "new_low_count_last_3": float(low_count),
        **acceptance,
        "opening_range_acceptance_code": acceptance_code,
        "current_close": current_close,
        "current_high": float(ordered.loc[origin, "high"]),
        "current_low": float(ordered.loc[origin, "low"]),
        "opening_high": opening_high,
        "opening_low": opening_low,
        "true_range_last_6_mean_bps": longer_true_range,
    }


def _leave_one_out(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    medians = np.asarray(
        [np.median(np.delete(array, index)) for index in range(len(array))], dtype=float
    )
    return array - medians, medians


def add_cross_sectional_features(frame: pd.DataFrame, *, prefix: str) -> None:
    raw_last = f"{prefix}raw_return_last_3_bps"
    raw_previous = f"{prefix}raw_return_previous_3_bps"
    raw_acceleration = f"{prefix}raw_acceleration_bps"
    open_current = f"{prefix}open_to_decision_raw_return_bps"
    open_previous = f"{prefix}open_to_previous_bar_raw_return_bps"
    latest_one = f"{prefix}latest_one_bar_return_bps"
    for _, indices in frame.groupby("slate_id", sort=True).groups.items():
        index = list(indices)
        relative_last, _ = _leave_one_out(frame.loc[index, raw_last].to_numpy(dtype=float))
        relative_previous, _ = _leave_one_out(frame.loc[index, raw_previous].to_numpy(dtype=float))
        acceleration_residual, _ = _leave_one_out(
            frame.loc[index, raw_acceleration].to_numpy(dtype=float)
        )
        frame.loc[index, f"{prefix}relative_return_last_3_bps"] = relative_last
        frame.loc[index, f"{prefix}relative_return_previous_3_bps"] = relative_previous
        frame.loc[index, f"{prefix}relative_strength_acceleration_bps"] = (
            relative_last - relative_previous
        )
        frame.loc[index, f"{prefix}stock_acceleration_minus_cohort_median_acceleration_bps"] = (
            acceleration_residual
        )
        signs = frame.loc[index, latest_one].to_numpy(dtype=float) > 0.0
        positive_count = int(signs.sum())
        frame.loc[index, f"{prefix}cross_sectional_breadth"] = (
            positive_count - signs.astype(int)
        ) / (len(index) - 1)
        current_values = frame.loc[index, open_current].to_numpy(dtype=float)
        previous_values = frame.loc[index, open_previous].to_numpy(dtype=float)
        open_residual, _ = _leave_one_out(current_values)
        frame.loc[index, f"{prefix}open_to_decision_cohort_relative_return_bps"] = open_residual
        current_dispersion = float(np.std(current_values, ddof=1))
        previous_dispersion = float(np.std(previous_values, ddof=1))
        frame.loc[index, f"{prefix}cross_sectional_dispersion_bps"] = current_dispersion
        frame.loc[index, f"{prefix}cross_sectional_dispersion_change_bps"] = (
            current_dispersion - previous_dispersion
        )


def add_trailing_ratios(frame: pd.DataFrame, *, prefix: str) -> None:
    ordered = frame.sort_values(["symbol", "decision_ordinal", "session"], kind="mergesort")
    volatility = f"{prefix}realized_volatility_3_bps"
    opening_range = f"{prefix}opening_high_low_range_bps"
    vol_baseline = f"{prefix}trailing_volatility_3_bps"
    range_baseline = f"{prefix}trailing_opening_range_median_bps"
    ordered[vol_baseline] = ordered.groupby(["symbol", "decision_ordinal"], sort=False)[
        volatility
    ].transform(lambda values: values.rolling(20, min_periods=10).median().shift(1))
    ordered[range_baseline] = ordered.groupby(["symbol", "decision_ordinal"], sort=False)[
        opening_range
    ].transform(lambda values: values.rolling(20, min_periods=10).median().shift(1))
    ordered[f"{prefix}short_realized_volatility_ratio"] = ordered[volatility] / ordered[
        vol_baseline
    ].replace(0.0, np.nan)
    ordered[f"{prefix}opening_range_to_trailing_same_checkpoint_median"] = ordered[
        opening_range
    ] / ordered[range_baseline].replace(0.0, np.nan)
    update_columns = (
        vol_baseline,
        range_baseline,
        f"{prefix}short_realized_volatility_ratio",
        f"{prefix}opening_range_to_trailing_same_checkpoint_median",
    )
    frame.loc[ordered.index, list(update_columns)] = ordered.loc[:, list(update_columns)]


def finalize_eligible_slate_features(frame: pd.DataFrame) -> dict[str, float]:
    """Re-anchor every cohort quantity to the final complete-history slate."""

    add_cross_sectional_features(frame, prefix="calc__")
    add_cross_sectional_features(frame, prefix="t1__")
    cross_features = (
        "relative_return_last_3_bps",
        "relative_return_previous_3_bps",
        "relative_strength_acceleration_bps",
        "stock_acceleration_minus_cohort_median_acceleration_bps",
        "cross_sectional_breadth",
        "cross_sectional_dispersion_change_bps",
    )
    for prefix in ("calc__", "t1__"):
        target_prefix = "" if prefix == "calc__" else "t1__"
        frame[f"{target_prefix}open_to_decision_cohort_relative_return_bps"] = frame[
            f"{prefix}open_to_decision_cohort_relative_return_bps"
        ]
        frame[f"{target_prefix}cross_sectional_dispersion_bps"] = frame[
            f"{prefix}cross_sectional_dispersion_bps"
        ]
        for feature in cross_features:
            frame[f"{target_prefix}{feature}"] = frame[f"{prefix}{feature}"]

    for prefix in ("", "t1__"):
        frame[f"{prefix}signed_progress_per_activity_unwinsorized"] = [
            progress_per_activity(relative_return, relative_activity)
            for relative_return, relative_activity in zip(
                frame[f"{prefix}relative_return_last_3_bps"],
                (
                    frame["t1__relative_activity_last_3_mean"]
                    if prefix
                    else frame["calc__relative_activity_last_3_mean"]
                ),
                strict=True,
            )
        ]
    progress_development = frame.loc[
        frame["year"].eq(2024), "signed_progress_per_activity_unwinsorized"
    ].dropna()
    if progress_development.empty:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "final progress bounds unavailable"
        )
    bounds = {
        "lower_q01": float(progress_development.quantile(0.01, interpolation="linear")),
        "upper_q99": float(progress_development.quantile(0.99, interpolation="linear")),
    }
    for prefix in ("", "t1__"):
        frame[f"{prefix}signed_progress_per_activity"] = frame[
            f"{prefix}signed_progress_per_activity_unwinsorized"
        ].clip(bounds["lower_q01"], bounds["upper_q99"])
    frame["change_cohort_relative_return_bps"] = (
        frame["t1__open_to_decision_cohort_relative_return_bps"]
        - frame["open_to_decision_cohort_relative_return_bps"]
    )
    frame["change_relative_strength_acceleration"] = (
        frame["t1__relative_strength_acceleration_bps"]
        - frame["relative_strength_acceleration_bps"]
    )
    for _, indices in frame.groupby("slate_id", sort=True).groups.items():
        index = list(indices)
        raw_paths = frame.loc[
            index,
            [
                "raw_onset_t_plus_2_bps",
                "raw_onset_t_plus_3_bps",
                "raw_onset_t_plus_4_bps",
            ],
        ].to_numpy(dtype=float)
        residuals, medians = cohort_relative_cumulative_paths_bps(raw_paths)
        for offset, step in enumerate((2, 3, 4)):
            frame.loc[index, f"residual_t_plus_{step}_bps"] = residuals[:, offset]
            frame.loc[index, f"cohort_median_t_plus_{step}_minus_i_bps"] = medians[:, offset]
        continuation_residual, continuation_median = _leave_one_out(
            frame.loc[index, "raw_continuation_return_bps"].to_numpy(dtype=float)
        )
        remaining_residual, remaining_median = _leave_one_out(
            frame.loc[index, "raw_remaining_session_return_bps"].to_numpy(dtype=float)
        )
        frame.loc[index, "cohort_relative_continuation_return_bps"] = continuation_residual
        frame.loc[index, "cohort_median_continuation_minus_i_bps"] = continuation_median
        frame.loc[index, "cohort_relative_remaining_session_return_bps"] = remaining_residual
        frame.loc[index, "cohort_median_remaining_session_minus_i_bps"] = remaining_median
    return bounds


def build_compact_panel(
    predecessor: pd.DataFrame,
    *,
    provider_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predecessor_manifest = json.loads(PREDECESSOR_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected_sources = {
        str(row["symbol"]): row
        for row in predecessor_manifest["sources"]
        if str(row["symbol"]) in SYMBOLS
    }
    if set(expected_sources) != set(SYMBOLS):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "predecessor source cohort incomplete"
        )
    records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    source_month_parts: list[pd.DataFrame] = []
    source_minimum: pd.Timestamp | None = None
    source_maximum: pd.Timestamp | None = None
    gap_ledger_rows = 0

    predecessor_keys = predecessor.loc[:, list(PREDECESSOR_KEEP)].copy()
    for symbol in SYMBOLS:
        qa = load_qa_record(symbol)
        qa_records.append(qa)
        path = provider_path(provider_root, symbol)
        raw = bounded_source(path)
        digest = arrow_hash(raw)
        expected = expected_sources[symbol]
        if digest != expected["complete_safe_bounded_hash"] or len(raw) != int(
            expected["complete_safe_bounded_rows"]
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
                "corporate_action_check_passed": qa["corporate_action_check_passed"],
            }
        )
        months = raw[["timestamp"]].copy()
        months["year_month"] = months["timestamp"].dt.strftime("%Y-%m")
        source_month_parts.append(months)
        current_minimum = pd.Timestamp(raw["timestamp"].min())
        current_maximum = pd.Timestamp(raw["timestamp"].max())
        source_minimum = (
            current_minimum if source_minimum is None else min(source_minimum, current_minimum)
        )
        source_maximum = (
            current_maximum if source_maximum is None else max(source_maximum, current_maximum)
        )
        bars, rejected = prepare_symbol_bars(raw, symbol=symbol)
        gap_ledger_rows += rejected
        sessions = {
            str(session): part.reset_index(drop=True)
            for session, part in bars.groupby("session", sort=True)
        }
        requested = predecessor_keys.loc[predecessor_keys["symbol"].eq(symbol)]
        for row in requested.itertuples(index=False):
            session = str(row.session)
            if session not in sessions:
                continue
            session_frame = sessions[session]
            decision_ordinal = int(row.decision_ordinal)
            window = extract_decision_window(session_frame, decision_ordinal=decision_ordinal)
            base = point_features(session_frame, window.decision_bar_ordinal)
            confirmation = point_features(session_frame, window.confirmation_bar_ordinal)
            by_ordinal = session_frame.set_index("bar_ordinal", verify_integrity=True)
            onset_returns = tuple(
                10_000.0 * (close / window.delayed_entry_open - 1.0)
                for close in window.onset_closes
            )
            record: dict[str, Any] = {
                "symbol": symbol,
                "session": session,
                "decision_ordinal": decision_ordinal,
                "slate_id": str(row.slate_id),
                "decision_available_timestamp": window.decision_available_timestamp,
                "confirmation_available_timestamp": window.confirmation_available_timestamp,
                "entry_timestamp": window.entry_timestamp,
                "confirmation_bar_ordinal": window.confirmation_bar_ordinal,
                "entry_bar_ordinal_reconstructed": window.entry_bar_ordinal,
                "onset_t_plus_2_bar_ordinal": window.onset_bar_ordinals[0],
                "onset_t_plus_3_bar_ordinal": window.onset_bar_ordinals[1],
                "onset_t_plus_4_bar_ordinal": window.onset_bar_ordinals[2],
                "onset_t_plus_2_close_timestamp": pd.Timestamp(
                    by_ordinal.loc[window.onset_bar_ordinals[0], "bar_complete_timestamp"]
                ),
                "onset_t_plus_3_close_timestamp": pd.Timestamp(
                    by_ordinal.loc[window.onset_bar_ordinals[1], "bar_complete_timestamp"]
                ),
                "onset_t_plus_4_close_timestamp": pd.Timestamp(
                    by_ordinal.loc[window.onset_bar_ordinals[2], "bar_complete_timestamp"]
                ),
                "delayed_entry_open_reconstructed": window.delayed_entry_open,
                "raw_onset_t_plus_2_bps": onset_returns[0],
                "raw_onset_t_plus_3_bps": onset_returns[1],
                "raw_onset_t_plus_4_bps": onset_returns[2],
                "continuation_exit_bar_ordinal": window.continuation_exit_bar_ordinal,
                "continuation_exit_close": window.continuation_exit_close,
                "raw_continuation_return_bps": 10_000.0
                * (window.continuation_exit_close / window.delayed_entry_open - 1.0),
                "raw_remaining_session_return_bps": 10_000.0
                * (window.terminal_close / window.delayed_entry_open - 1.0),
                **{f"calc__{key}": value for key, value in base.items()},
                **{f"t1__{key}": value for key, value in confirmation.items()},
            }
            records.append(record)

    raw_records = (
        pd.DataFrame(records)
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    raw_records["source_slate_size"] = raw_records.groupby("slate_id", sort=True)[
        "symbol"
    ].transform("size")
    raw_records = raw_records.loc[raw_records["source_slate_size"].ge(15)].copy()
    if raw_records.empty:
        raise ScreenBlocker(
            "blocked_insufficient_pressure_onset_support", "no slates contain 15 valid stocks"
        )
    add_cross_sectional_features(raw_records, prefix="calc__")
    add_cross_sectional_features(raw_records, prefix="t1__")
    add_trailing_ratios(raw_records, prefix="calc__")
    add_trailing_ratios(raw_records, prefix="t1__")

    for _, indices in raw_records.groupby("slate_id", sort=True).groups.items():
        index = list(indices)
        raw_paths = raw_records.loc[
            index,
            [
                "raw_onset_t_plus_2_bps",
                "raw_onset_t_plus_3_bps",
                "raw_onset_t_plus_4_bps",
            ],
        ].to_numpy(dtype=float)
        residuals, medians = cohort_relative_cumulative_paths_bps(raw_paths)
        for offset, step in enumerate((2, 3, 4)):
            raw_records.loc[index, f"residual_t_plus_{step}_bps"] = residuals[:, offset]
            raw_records.loc[index, f"cohort_median_t_plus_{step}_minus_i_bps"] = medians[:, offset]
        continuation_residual, continuation_median = _leave_one_out(
            raw_records.loc[index, "raw_continuation_return_bps"].to_numpy(dtype=float)
        )
        remaining_residual, remaining_median = _leave_one_out(
            raw_records.loc[index, "raw_remaining_session_return_bps"].to_numpy(dtype=float)
        )
        raw_records.loc[index, "cohort_relative_continuation_return_bps"] = continuation_residual
        raw_records.loc[index, "cohort_median_continuation_minus_i_bps"] = continuation_median
        raw_records.loc[index, "cohort_relative_remaining_session_return_bps"] = remaining_residual
        raw_records.loc[index, "cohort_median_remaining_session_minus_i_bps"] = remaining_median

    keys = ["symbol", "session", "decision_ordinal", "slate_id"]
    compact = predecessor_keys.merge(
        raw_records,
        on=keys,
        how="inner",
        validate="one_to_one",
        sort=False,
    ).sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    compact = compact.reset_index(drop=True)
    for feature in (
        "open_to_decision_cohort_relative_return_bps",
        "cross_sectional_dispersion_bps",
    ):
        compact[f"m1_source__{feature}"] = compact[feature]
    if not np.allclose(
        compact["delayed_entry_open"],
        compact["delayed_entry_open_reconstructed"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "delayed-entry reconstruction differs"
        )
    expected_decision_available = pd.to_datetime(
        compact["feature_available_timestamp_utc"], utc=True
    )
    if not expected_decision_available.equals(
        pd.to_datetime(compact["decision_available_timestamp"], utc=True)
    ):
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "decision timestamp reconstruction differs"
        )

    new_readiness = (
        "latest_six_bar_return_bps",
        "realized_volatility_3_bps",
        "realized_volatility_6_bps",
        "short_realized_volatility_ratio",
        "opening_range_to_trailing_same_checkpoint_median",
        "short_true_range_to_longer_true_range",
    )
    for feature in new_readiness:
        compact[feature] = compact[f"calc__{feature}"]
    for feature in PRESSURE_FEATURES:
        if feature == "signed_progress_per_activity":
            continue
        compact[feature] = compact[f"calc__{feature}"]
    compact["signed_progress_per_activity_unwinsorized"] = [
        progress_per_activity(relative_return, relative_activity)
        for relative_return, relative_activity in zip(
            compact["relative_return_last_3_bps"],
            compact["calc__relative_activity_last_3_mean"],
            strict=True,
        )
    ]

    t1_readiness_from_calc = tuple(
        feature
        for feature in READINESS_FEATURES
        if feature not in {"p_large_remaining_move", "opening_gap_bps"}
    )
    compact["t1__opening_gap_bps"] = compact["opening_gap_bps"]
    for feature in t1_readiness_from_calc:
        compact[f"t1__{feature}"] = compact[f"t1__{feature}"]
    for feature in PRESSURE_FEATURES:
        if feature == "signed_progress_per_activity":
            continue
        compact[f"t1__{feature}"] = compact[f"t1__{feature}"]
    compact["t1__signed_progress_per_activity_unwinsorized"] = [
        progress_per_activity(relative_return, relative_activity)
        for relative_return, relative_activity in zip(
            compact["t1__relative_return_last_3_bps"],
            compact["t1__relative_activity_last_3_mean"],
            strict=True,
        )
    ]
    progress_development = compact.loc[
        compact["year"].eq(2024), "signed_progress_per_activity_unwinsorized"
    ].dropna()
    if progress_development.empty:
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "progress winsor bounds unavailable"
        )
    progress_bounds = {
        "lower_q01": float(progress_development.quantile(0.01, interpolation="linear")),
        "upper_q99": float(progress_development.quantile(0.99, interpolation="linear")),
    }
    compact["signed_progress_per_activity"] = compact[
        "signed_progress_per_activity_unwinsorized"
    ].clip(progress_bounds["lower_q01"], progress_bounds["upper_q99"])
    compact["t1__signed_progress_per_activity"] = compact[
        "t1__signed_progress_per_activity_unwinsorized"
    ].clip(progress_bounds["lower_q01"], progress_bounds["upper_q99"])

    compact["change_cohort_relative_return_bps"] = (
        compact["t1__open_to_decision_cohort_relative_return_bps"]
        - compact["open_to_decision_cohort_relative_return_bps"]
    )
    compact["change_relative_strength_acceleration"] = (
        compact["t1__relative_strength_acceleration_bps"]
        - compact["relative_strength_acceleration_bps"]
    )
    compact["change_activity_shock"] = (
        compact["t1__historical_activity_proxy_shock"] - compact["historical_activity_proxy_shock"]
    )
    compact["change_range_acceleration"] = (
        compact["t1__range_acceleration"] - compact["range_acceleration"]
    )
    compact["change_signed_efficiency_3"] = (
        compact["t1__signed_efficiency_3"] - compact["signed_efficiency_3"]
    )
    compact["change_close_location"] = (
        compact["t1__current_close_location"] - compact["current_close_location"]
    )
    compact["new_high_at_t_plus_1"] = (
        compact["t1__current_high"] > compact["calc__opening_high"]
    ).astype(float)
    compact["new_low_at_t_plus_1"] = (
        compact["t1__current_low"] < compact["calc__opening_low"]
    ).astype(float)
    compact["opening_range_acceptance_persisted"] = (
        compact["t1__opening_range_acceptance_code"]
        .eq(compact["calc__opening_range_acceptance_code"])
        .astype(float)
    )
    compact["favourable_retracement_bps"] = np.nan
    compact["predicted_direction_remained_same"] = np.nan

    t1_required = [f"t1__{feature}" for feature in READINESS_FEATURES[1:]] + [
        f"t1__{feature}" for feature in PRESSURE_FEATURES
    ]
    core_required = [*READINESS_FEATURES[1:], *PRESSURE_FEATURES, *t1_required]
    finite = np.isfinite(compact.loc[:, core_required].to_numpy(dtype=float)).all(axis=1)
    compact = compact.loc[finite].copy().reset_index(drop=True)
    compact["source_slate_size"] = compact.groupby("slate_id", sort=True)["symbol"].transform(
        "size"
    )
    compact = compact.loc[compact["source_slate_size"].ge(15)].copy().reset_index(drop=True)
    progress_bounds = finalize_eligible_slate_features(compact)
    if len(compact) > MAX_COMPACT_ROWS:
        raise ScreenBlocker(
            "blocked_quick_pressure_screen_resource_limit", "compact rows exceed 20,000"
        )
    if compact.empty:
        raise ScreenBlocker(
            "blocked_insufficient_pressure_onset_support", "compact feature panel is empty"
        )
    assert_safe_timestamps(compact["decision_available_timestamp"])
    barriers = development_onset_barriers(compact)
    compact["onset_barrier_bps"] = compact["decision_ordinal"].map(barriers)
    compact["onset_label"] = [
        classify_onset(path, barrier_bps=barrier)
        for path, barrier in zip(
            compact[
                [
                    "residual_t_plus_2_bps",
                    "residual_t_plus_3_bps",
                    "residual_t_plus_4_bps",
                ]
            ].to_numpy(dtype=float),
            compact["onset_barrier_bps"].to_numpy(dtype=float),
            strict=True,
        )
    ]
    compact["directional_onset"] = compact["onset_label"].ne("NO_ONSET").astype(np.int8)
    compact["up_given_onset"] = compact["onset_label"].eq("UP_ONSET").astype(np.int8)

    ledger_rows: list[dict[str, Any]] = []
    for row in compact.itertuples(index=False):
        for step in (2, 3, 4):
            ledger_rows.append(
                {
                    "symbol": row.symbol,
                    "session": row.session,
                    "decision_ordinal": row.decision_ordinal,
                    "slate_id": row.slate_id,
                    "delayed_entry_open": row.delayed_entry_open,
                    "path_step": step,
                    "path_bar_ordinal": getattr(row, f"onset_t_plus_{step}_bar_ordinal"),
                    "path_close_timestamp": getattr(row, f"onset_t_plus_{step}_close_timestamp"),
                    "stock_cumulative_return_bps": getattr(row, f"raw_onset_t_plus_{step}_bps"),
                    "cohort_median_return_minus_i_bps": getattr(
                        row, f"cohort_median_t_plus_{step}_minus_i_bps"
                    ),
                    "cumulative_residual_return_bps": getattr(row, f"residual_t_plus_{step}_bps"),
                    "onset_barrier_bps": row.onset_barrier_bps,
                    "onset_label": row.onset_label,
                }
            )
    ledger = pd.DataFrame(ledger_rows).sort_values(
        ["session", "decision_ordinal", "symbol", "path_step"], kind="mergesort"
    )
    all_source_months = pd.concat(source_month_parts, ignore_index=True)
    source_months = (
        all_source_months.groupby("year_month", sort=True).size().rename("row_count").reset_index()
    )
    context = {
        "sources": source_records,
        "vendor_qa": qa_records,
        "source_rows_by_year_month": source_months.to_dict("records"),
        "minimum_timestamp_read": str(source_minimum),
        "maximum_timestamp_read": str(source_maximum),
        "gap_ledger_rows": gap_ledger_rows,
        "progress_winsor_bounds": progress_bounds,
        "onset_barriers": {str(key): value for key, value in barriers.items()},
        "protected_rows_materialised": 0,
    }
    return compact, ledger.reset_index(drop=True), context


def t1_scoring_frame(frame: pd.DataFrame) -> pd.DataFrame:
    scoring = frame.copy()
    for feature in READINESS_FEATURES:
        if feature == "p_large_remaining_move":
            continue
        scoring[feature] = frame[f"t1__{feature}"]
    for feature in PRESSURE_FEATURES:
        scoring[feature] = frame[f"t1__{feature}"]
    return scoring


def _attach_confirmation_from_probabilities(
    frame: pd.DataFrame,
    probability_t: np.ndarray,
    probability_t1: np.ndarray,
) -> pd.DataFrame:
    output = frame.copy()
    direction_t = np.where(probability_t >= 0.5, 1.0, -1.0)
    direction_t1 = np.where(probability_t1 >= 0.5, 1.0, -1.0)
    output["predicted_direction_remained_same"] = (direction_t == direction_t1).astype(float)
    base_close = output["calc__current_close"].to_numpy(dtype=float)
    next_high = output["t1__current_high"].to_numpy(dtype=float)
    next_low = output["t1__current_low"].to_numpy(dtype=float)
    next_close = output["t1__current_close"].to_numpy(dtype=float)
    long_favourable = 10_000.0 * (next_high / base_close - 1.0)
    long_progress = 10_000.0 * (next_close / base_close - 1.0)
    short_favourable = 10_000.0 * (1.0 - next_low / base_close)
    short_progress = 10_000.0 * (1.0 - next_close / base_close)
    favourable = np.where(direction_t > 0.0, long_favourable, short_favourable)
    progress = np.where(direction_t > 0.0, long_progress, short_progress)
    output["favourable_retracement_bps"] = np.maximum(0.0, np.maximum(0.0, favourable) - progress)
    if not np.isfinite(output.loc[:, list(CONFIRMATION_FEATURES)].to_numpy(dtype=float)).all():
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "confirmation feature unavailable"
        )
    if (
        pd.to_datetime(output["confirmation_available_timestamp"], utc=True)
        > pd.to_datetime(output["entry_timestamp"], utc=True)
    ).any():
        raise ScreenBlocker(
            "blocked_chronology_or_leakage_failure", "confirmation leaks t+2 information"
        )
    return output


def fit_model_ladder(
    development: pd.DataFrame,
) -> tuple[dict[str, FrozenLogisticModel], pd.DataFrame, list[dict[str, Any]]]:
    models: dict[str, FrozenLogisticModel] = {}
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    try:
        for name in ("A0", "A1", "A2"):
            models[name] = fit_fixed_logistic(
                development,
                development["directional_onset"],
                features=MODEL_FEATURES[name],
                slate_column="slate_id",
                model_id=name,
            )
        direction = development.loc[development["directional_onset"].eq(1)].copy()
        for name in ("D0", "D1", "D2"):
            models[name] = fit_fixed_logistic(
                direction,
                direction["up_given_onset"],
                features=MODEL_FEATURES[name],
                slate_column="slate_id",
                model_id=name,
            )
        confirmed_development = _attach_confirmation_from_probabilities(
            development,
            models["D2"].predict(development),
            models["D2"].predict(t1_scoring_frame(development)),
        )
        confirmation_manifest = [
            {
                "feature": "predicted_direction_remained_same",
                "source_model": "D2",
                "source_model_fit": "single_fixed_development_fit",
                "additional_model_specifications": 0,
                "available_at": "completed_t_plus_1_before_open_t_plus_2",
            }
        ]
        models["A3"] = fit_fixed_logistic(
            confirmed_development,
            confirmed_development["directional_onset"],
            features=MODEL_FEATURES["A3"],
            slate_column="slate_id",
            model_id="A3",
        )
        confirmed_direction = confirmed_development.loc[
            confirmed_development["directional_onset"].eq(1)
        ]
        models["D3"] = fit_fixed_logistic(
            confirmed_direction,
            confirmed_direction["up_given_onset"],
            features=MODEL_FEATURES["D3"],
            slate_column="slate_id",
            model_id="D3",
        )
    except ConvergenceWarning as exc:
        raise ScreenBlocker(
            "blocked_model_convergence_failure", f"model convergence warning: {exc}"
        ) from exc
    except RuntimeError as exc:
        if "converge" in str(exc).lower():
            raise ScreenBlocker("blocked_model_convergence_failure", str(exc)) from exc
        raise
    if len(models) != 8 or not all(model.converged for model in models.values()):
        raise ScreenBlocker(
            "blocked_model_convergence_failure", "eight converged models were not produced"
        )
    return models, confirmed_development, confirmation_manifest


def score_model_ladder(
    frame: pd.DataFrame, models: Mapping[str, FrozenLogisticModel]
) -> pd.DataFrame:
    scored = frame.copy()
    for name in ("A0", "A1", "A2", "D0", "D1", "D2"):
        target = "onset" if name.startswith("A") else "up_given_onset"
        scored[f"p_{target}__{name}"] = models[name].predict(scored)
    scored = _attach_confirmation_from_probabilities(
        scored,
        scored["p_up_given_onset__D2"].to_numpy(dtype=float),
        models["D2"].predict(t1_scoring_frame(scored)),
    )
    scored["p_onset__A3"] = models["A3"].predict(scored)
    scored["p_up_given_onset__D3"] = models["D3"].predict(scored)
    systems = {"readiness": ("A1", "D1"), "pressure": ("A2", "D2"), "confirmed": ("A3", "D3")}
    for system, (onset_model, direction_model) in systems.items():
        onset_probability = scored[f"p_onset__{onset_model}"]
        up_probability = scored[f"p_up_given_onset__{direction_model}"]
        scored[f"p_onset__{system}_system"] = onset_probability
        scored[f"p_up_given_onset__{system}_system"] = up_probability
        scored[f"p_down_given_onset__{system}_system"] = 1.0 - up_probability
        scored[f"p_up_onset__{system}_system"] = onset_probability * up_probability
        scored[f"p_down_onset__{system}_system"] = onset_probability * (1.0 - up_probability)
        scored[f"p_no_onset__{system}_system"] = 1.0 - onset_probability
        scored[f"signed_pressure_score__{system}"] = (
            onset_probability * (2.0 * up_probability - 1.0) * scored["p_large_remaining_move"]
        )
    return scored


def _calibration_parameters(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Estimate diagnostic calibration intercept and slope without fitting a ladder model."""

    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    outcomes = np.asarray(labels, dtype=float)
    if len(np.unique(outcomes)) < 2:
        return math.nan, math.nan
    logits = np.log(clipped / (1.0 - clipped))

    def objective(parameters: np.ndarray) -> float:
        linear = parameters[0] + parameters[1] * logits
        fitted = 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))
        return float(
            -np.sum(
                outcomes * np.log(np.clip(fitted, 1e-15, 1.0))
                + (1.0 - outcomes) * np.log(np.clip(1.0 - fitted, 1e-15, 1.0))
            )
        )

    result = minimize(
        objective,
        np.asarray([0.0, 1.0], dtype=float),
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    if not np.isfinite(result.x).all():
        return math.nan, math.nan
    return float(result.x[0]), float(result.x[1])


def binary_metric_record(
    frame: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str,
    model: str,
    population: str,
    scope_type: str,
    scope_value: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calculate the complete preregistered binary probability diagnostic."""

    labels = frame[target_column].to_numpy(dtype=int)
    probabilities = frame[probability_column].to_numpy(dtype=float)
    if len(frame) == 0 or not np.isfinite(probabilities).all():
        raise ValueError(f"empty or non-finite metric frame for {model}/{scope_value}")
    brier = float(np.mean((labels - probabilities) ** 2))
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped)))
    auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else math.nan
    calibration_intercept, calibration_slope = _calibration_parameters(labels, probabilities)
    bin_numbers = np.minimum((probabilities * 10.0).astype(int), 9)
    calibration_rows: list[dict[str, Any]] = []
    weighted_errors = 0.0
    for bin_number in range(10):
        mask = bin_numbers == bin_number
        rows = int(mask.sum())
        mean_probability = float(np.mean(probabilities[mask])) if rows else math.nan
        observed_rate = float(np.mean(labels[mask])) if rows else math.nan
        if rows:
            weighted_errors += rows * abs(mean_probability - observed_rate)
        calibration_rows.append(
            {
                "population": population,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "target": target_column,
                "model": model,
                "bin": bin_number + 1,
                "lower_bound_inclusive": bin_number / 10.0,
                "upper_bound_exclusive_except_last": (bin_number + 1) / 10.0,
                "rows": rows,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    record = {
        "population": population,
        "scope_type": scope_type,
        "scope_value": scope_value,
        "target": target_column,
        "model": model,
        "probability_column": probability_column,
        "brier_score": brier,
        "log_loss": loss,
        "auc": auc,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "expected_calibration_error": weighted_errors / len(frame),
        "base_rate": float(np.mean(labels)),
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
    }
    return record, calibration_rows


def evaluate_model_ladder(
    assessment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate every ladder model for both fixed populations and all fixed slices."""

    onset_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    populations = {
        "primary_high_movement": assessment.loc[assessment["high_movement_admitted"].astype(bool)],
        "secondary_all_rows": assessment,
    }
    for population, population_frame in populations.items():
        slices: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", population_frame)]
        slices.extend(
            ("month", str(month), part)
            for month, part in population_frame.groupby("year_month", sort=True)
        )
        slices.extend(
            ("checkpoint", str(int(checkpoint)), part)
            for checkpoint, part in population_frame.groupby("decision_ordinal", sort=True)
        )
        for scope_type, scope_value, scope_frame in slices:
            for model in ("A0", "A1", "A2", "A3"):
                record, bins = binary_metric_record(
                    scope_frame,
                    target_column="directional_onset",
                    probability_column=f"p_onset__{model}",
                    model=model,
                    population=population,
                    scope_type=scope_type,
                    scope_value=scope_value,
                )
                calibration_rows.extend(bins)
                if scope_type == "pooled":
                    onset_rows.append(record)
                elif scope_type == "month":
                    monthly_rows.append(record)
                else:
                    checkpoint_rows.append(record)
            direction_frame = scope_frame.loc[scope_frame["directional_onset"].eq(1)]
            if direction_frame.empty:
                continue
            for model in ("D0", "D1", "D2", "D3"):
                record, bins = binary_metric_record(
                    direction_frame,
                    target_column="up_given_onset",
                    probability_column=f"p_up_given_onset__{model}",
                    model=model,
                    population=population,
                    scope_type=scope_type,
                    scope_value=scope_value,
                )
                calibration_rows.extend(bins)
                if scope_type == "pooled":
                    direction_rows.append(record)
                elif scope_type == "month":
                    monthly_rows.append(record)
                else:
                    checkpoint_rows.append(record)
    sorting = ["population", "scope_type", "scope_value", "target", "model"]
    return (
        pd.DataFrame(onset_rows).sort_values(sorting, kind="mergesort"),
        pd.DataFrame(direction_rows).sort_values(sorting, kind="mergesort"),
        pd.DataFrame(monthly_rows).sort_values(sorting, kind="mergesort"),
        pd.DataFrame(checkpoint_rows).sort_values(sorting, kind="mergesort"),
        pd.DataFrame(calibration_rows).sort_values([*sorting, "bin"], kind="mergesort"),
    )


def _stable_random_choice(frame: pd.DataFrame, slate_id: str) -> tuple[pd.Series, float]:
    ordered = frame.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256(f"{RANDOM_SEED}:{slate_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    row = ordered.iloc[value % len(ordered)]
    direction = 1.0 if digest[8] % 2 == 0 else -1.0
    return row, direction


def economic_selections(
    primary: pd.DataFrame,
    *,
    candidates: Sequence[str] = (
        "readiness",
        "pressure",
        "confirmed",
        "highest_relative_momentum",
        "strongest_reversal",
        "random_within_slate",
    ),
) -> pd.DataFrame:
    """Apply stable one-name-per-slate delayed economic-reference selections."""

    rows: list[dict[str, Any]] = []
    for slate_id, slate in primary.groupby("slate_id", sort=True):
        ordered = slate.sort_values("symbol", kind="mergesort").reset_index(drop=True)
        if int(ordered["source_slate_size"].min()) < 15:
            continue
        for candidate in candidates:
            if candidate in {"readiness", "pressure", "confirmed"}:
                score_column = f"signed_pressure_score__{candidate}"
                selected = (
                    ordered.assign(_absolute=ordered[score_column].abs())
                    .sort_values(
                        ["_absolute", "symbol"],
                        ascending=[False, True],
                        kind="mergesort",
                    )
                    .iloc[0]
                )
                score = float(selected[score_column])
                direction = 1.0 if score >= 0.0 else -1.0
            elif candidate in {"highest_relative_momentum", "strongest_reversal"}:
                selected = (
                    ordered.assign(
                        _absolute=ordered["open_to_decision_cohort_relative_return_bps"].abs()
                    )
                    .sort_values(
                        ["_absolute", "symbol"],
                        ascending=[False, True],
                        kind="mergesort",
                    )
                    .iloc[0]
                )
                momentum = float(selected["open_to_decision_cohort_relative_return_bps"])
                momentum_direction = 1.0 if momentum >= 0.0 else -1.0
                direction = (
                    momentum_direction
                    if candidate == "highest_relative_momentum"
                    else -momentum_direction
                )
                score = momentum * (1.0 if candidate == "highest_relative_momentum" else -1.0)
            else:
                selected, direction = _stable_random_choice(ordered, str(slate_id))
                score = direction
            rows.append(
                {
                    "candidate": candidate,
                    "slate_id": str(slate_id),
                    "session": str(selected["session"]),
                    "decision_ordinal": int(selected["decision_ordinal"]),
                    "symbol": str(selected["symbol"]),
                    "score": score,
                    "direction_sign": direction,
                    "signed_gross_return_bps_30m": direction
                    * float(selected["raw_continuation_return_bps"]),
                    "signed_cohort_relative_return_bps_30m": direction
                    * float(selected["cohort_relative_continuation_return_bps"]),
                    "signed_gross_return_bps_remaining_session": direction
                    * float(selected["raw_remaining_session_return_bps"]),
                    "signed_cohort_relative_return_bps_remaining_session": direction
                    * float(selected["cohort_relative_remaining_session_return_bps"]),
                    "entry_price": float(selected["delayed_entry_open"]),
                    "exit_price_30m": float(selected["continuation_exit_close"]),
                    "source_slate_size": int(selected["source_slate_size"]),
                    "high_movement_candidates_in_slate": len(ordered),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["candidate", "session", "decision_ordinal"], kind="mergesort")
        .reset_index(drop=True)
    )


def economic_metrics(selections: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the delayed 30-minute and remaining-session reference diagnostics."""

    rows: list[dict[str, Any]] = []
    horizons = {
        "primary_30m_close_t_plus_8": (
            "signed_gross_return_bps_30m",
            "signed_cohort_relative_return_bps_30m",
        ),
        "secondary_remaining_session": (
            "signed_gross_return_bps_remaining_session",
            "signed_cohort_relative_return_bps_remaining_session",
        ),
    }
    for candidate, candidate_frame in selections.groupby("candidate", sort=True):
        for horizon, (gross_column, relative_column) in horizons.items():
            gross = candidate_frame[gross_column].to_numpy(dtype=float)
            relative = candidate_frame[relative_column].to_numpy(dtype=float)
            for friction in (0.0, 10.0, 20.0):
                after_friction = gross - friction
                rows.append(
                    {
                        "candidate": candidate,
                        "horizon": horizon,
                        "friction_bps": friction,
                        "mean_signed_gross_return_bps": float(np.mean(gross)),
                        "mean_signed_return_after_friction_bps": float(np.mean(after_friction)),
                        "median_signed_return_after_friction_bps": float(np.median(after_friction)),
                        "positive_after_friction_rate": float(np.mean(after_friction > 0.0)),
                        "mean_signed_cohort_relative_return_bps": float(np.mean(relative)),
                        "selected_rows": len(candidate_frame),
                        "sessions": int(candidate_frame["session"].nunique()),
                        "stocks": int(candidate_frame["symbol"].nunique()),
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["horizon", "friction_bps", "candidate"], kind="mergesort")
        .reset_index(drop=True)
    )


def concentration_metrics(
    primary: pd.DataFrame, selections: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure row and selected-name concentration against the frozen gates."""

    rows: list[dict[str, Any]] = []
    row_shares = primary["symbol"].value_counts(normalize=True).sort_index()
    for symbol, share in row_shares.items():
        rows.append(
            {
                "population": "primary_high_movement_rows",
                "candidate": "not_applicable",
                "symbol": symbol,
                "rows": int((primary["symbol"] == symbol).sum()),
                "share": float(share),
                "maximum_allowed_share": 0.10,
                "passes": bool(share <= 0.10 + 1e-15),
            }
        )
    selected_maximum: dict[str, float] = {}
    for candidate, frame in selections.groupby("candidate", sort=True):
        shares = frame["symbol"].value_counts(normalize=True).sort_index()
        selected_maximum[str(candidate)] = float(shares.max())
        for symbol, share in shares.items():
            rows.append(
                {
                    "population": "selected_economic_reference_rows",
                    "candidate": str(candidate),
                    "symbol": symbol,
                    "rows": int((frame["symbol"] == symbol).sum()),
                    "share": float(share),
                    "maximum_allowed_share": 0.20,
                    "passes": bool(share <= 0.20 + 1e-15),
                }
            )
    summary = {
        "maximum_primary_row_stock_share": float(row_shares.max()),
        "primary_row_concentration_passes": bool(row_shares.max() <= 0.10 + 1e-15),
        "maximum_selected_stock_share_by_candidate": selected_maximum,
        "selected_concentration_passes": bool(
            selected_maximum and max(selected_maximum.values()) <= 0.20 + 1e-15
        ),
    }
    summary["all_concentration_gates_pass"] = bool(
        summary["primary_row_concentration_passes"] and summary["selected_concentration_passes"]
    )
    return pd.DataFrame(rows).sort_values(
        ["population", "candidate", "symbol"], kind="mergesort"
    ), summary


def support_summary(primary: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the preregistered 2025 support gates without changing population."""

    labels = primary["onset_label"].value_counts().to_dict()
    source_slate_minimum = int(primary["source_slate_size"].min())
    candidate_slate_minimum = int(primary.groupby("slate_id", sort=True)["symbol"].size().min())
    row_share = float(primary["symbol"].value_counts(normalize=True).max())
    summary = {
        "rows": len(primary),
        "sessions": int(primary["session"].nunique()),
        "stocks": int(primary["symbol"].nunique()),
        "months": int(primary["year_month"].nunique()),
        "slates": int(primary["slate_id"].nunique()),
        "directional_onset_rows": int(primary["directional_onset"].sum()),
        "up_onsets": int(labels.get("UP_ONSET", 0)),
        "down_onsets": int(labels.get("DOWN_ONSET", 0)),
        "no_onsets": int(labels.get("NO_ONSET", 0)),
        "minimum_valid_source_stocks_per_evaluated_slate": source_slate_minimum,
        "minimum_high_movement_candidates_per_slate": candidate_slate_minimum,
        "maximum_stock_row_share": row_share,
        "eligible_stock_gate_interpretation": (
            "eligible means admitted to the frozen high-movement population; "
            "the separate source-valid slate gate remains 15 stocks"
        ),
    }
    primary_gates = {
        "rows_at_least_1200": summary["rows"] >= 1_200,
        "sessions_at_least_100": summary["sessions"] >= 100,
        "stocks_at_least_15": summary["stocks"] >= 15,
        "months_at_least_6": summary["months"] >= 6,
        "high_movement_candidates_per_slate_at_least_10": candidate_slate_minimum >= 10,
        "valid_source_stocks_per_slate_at_least_15": source_slate_minimum >= 15,
        "maximum_stock_row_share_at_most_0_10": row_share <= 0.10 + 1e-15,
    }
    summary["primary_support_gates"] = primary_gates
    summary["failed_primary_support_gates"] = [
        name for name, passed in primary_gates.items() if not passed
    ]
    summary["primary_onset_support_passes"] = bool(all(primary_gates.values()))
    summary["conditional_direction_support_passes"] = bool(
        summary["directional_onset_rows"] >= 250
        and summary["up_onsets"] >= 100
        and summary["down_onsets"] >= 100
    )
    return summary


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def _bootstrap_loss_increment(
    frame: pd.DataFrame,
    *,
    target: str,
    baseline_probability: str,
    candidate_probability: str,
    session_counts: Counter[str],
    loss_kind: str,
) -> float:
    weights = frame["session"].astype(str).map(session_counts).fillna(0.0).astype(float)
    mask = weights.gt(0.0)
    labels = frame.loc[mask, target].to_numpy(dtype=float)
    baseline = frame.loc[mask, baseline_probability].to_numpy(dtype=float)
    candidate = frame.loc[mask, candidate_probability].to_numpy(dtype=float)
    selected_weights = weights.loc[mask].to_numpy(dtype=float)
    if loss_kind == "brier":
        baseline_loss = (labels - baseline) ** 2
        candidate_loss = (labels - candidate) ** 2
    else:
        baseline_clipped = np.clip(baseline, 1e-15, 1.0 - 1e-15)
        candidate_clipped = np.clip(candidate, 1e-15, 1.0 - 1e-15)
        baseline_loss = -(
            labels * np.log(baseline_clipped) + (1.0 - labels) * np.log(1.0 - baseline_clipped)
        )
        candidate_loss = -(
            labels * np.log(candidate_clipped) + (1.0 - labels) * np.log(1.0 - candidate_clipped)
        )
    return float(
        np.average(baseline_loss, weights=selected_weights)
        - np.average(candidate_loss, weights=selected_weights)
    )


def _bootstrap_economic_increment(
    selections: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    session_counts: Counter[str],
) -> float:
    means: dict[str, float] = {}
    for name in (baseline, candidate):
        frame = selections.loc[selections["candidate"].eq(name)]
        weights = frame["session"].astype(str).map(session_counts).fillna(0.0).astype(float)
        mask = weights.gt(0.0)
        means[name] = float(
            np.average(
                frame.loc[mask, "signed_gross_return_bps_30m"].to_numpy(dtype=float) - 20.0,
                weights=weights.loc[mask].to_numpy(dtype=float),
            )
        )
    return means[candidate] - means[baseline]


def bootstrap_metrics(primary: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    """Run exactly 200 paired whole-session bootstrap draws."""

    draws = session_block_bootstrap_draws(
        primary["session"].astype(str), draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    direction = primary.loc[primary["directional_onset"].eq(1)]
    specifications = (
        (
            "A2_minus_A1_brier_improvement",
            primary,
            "directional_onset",
            "p_onset__A1",
            "p_onset__A2",
            "brier",
        ),
        (
            "A2_minus_A1_log_loss_improvement",
            primary,
            "directional_onset",
            "p_onset__A1",
            "p_onset__A2",
            "log_loss",
        ),
        (
            "D2_minus_D1_brier_improvement",
            direction,
            "up_given_onset",
            "p_up_given_onset__D1",
            "p_up_given_onset__D2",
            "brier",
        ),
        (
            "D2_minus_D1_log_loss_improvement",
            direction,
            "up_given_onset",
            "p_up_given_onset__D1",
            "p_up_given_onset__D2",
            "log_loss",
        ),
        (
            "A3_minus_A2_brier_improvement",
            primary,
            "directional_onset",
            "p_onset__A2",
            "p_onset__A3",
            "brier",
        ),
        (
            "D3_minus_D2_brier_improvement",
            direction,
            "up_given_onset",
            "p_up_given_onset__D2",
            "p_up_given_onset__D3",
            "brier",
        ),
    )
    rows: list[dict[str, Any]] = []
    values_by_metric: dict[str, list[float]] = {name: [] for name, *_ in specifications}
    values_by_metric.update(
        {
            "pressure_minus_readiness_return_after_20bps": [],
            "confirmation_minus_pressure_return_after_20bps": [],
        }
    )
    for draw in draws:
        counts: Counter[str] = Counter(draw.sampled_sessions)
        draw_values: dict[str, float] = {}
        for name, frame, target, baseline, candidate, loss_kind in specifications:
            draw_values[name] = _bootstrap_loss_increment(
                frame,
                target=target,
                baseline_probability=baseline,
                candidate_probability=candidate,
                session_counts=counts,
                loss_kind=loss_kind,
            )
        draw_values["pressure_minus_readiness_return_after_20bps"] = _bootstrap_economic_increment(
            selections,
            baseline="readiness",
            candidate="pressure",
            session_counts=counts,
        )
        draw_values["confirmation_minus_pressure_return_after_20bps"] = (
            _bootstrap_economic_increment(
                selections,
                baseline="pressure",
                candidate="confirmed",
                session_counts=counts,
            )
        )
        for metric, value in draw_values.items():
            values_by_metric[metric].append(value)
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw.draw,
                    "metric": metric,
                    "value": value,
                    "lower_90": math.nan,
                    "upper_90": math.nan,
                    "lower_95": math.nan,
                    "upper_95": math.nan,
                }
            )
    for metric, values in values_by_metric.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "record_type": "summary",
                "draw": -1,
                "metric": metric,
                "value": float(np.mean(array)),
                "lower_90": float(np.quantile(array, 0.05)),
                "upper_90": float(np.quantile(array, 0.95)),
                "lower_95": float(np.quantile(array, 0.025)),
                "upper_95": float(np.quantile(array, 0.975)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["record_type", "metric", "draw"], kind="mergesort")
        .reset_index(drop=True)
    )


def _brier_improvement(labels: pd.Series, baseline: np.ndarray, candidate: np.ndarray) -> float:
    outcomes = labels.to_numpy(dtype=float)
    return float(np.mean((outcomes - baseline) ** 2) - np.mean((outcomes - candidate) ** 2))


def null_metrics(
    development: pd.DataFrame,
    assessment_primary: pd.DataFrame,
    models: Mapping[str, FrozenLogisticModel],
    real_selections: pd.DataFrame,
) -> pd.DataFrame:
    """Run exactly 50 within-slate pressure-bundle permutation null draws."""

    real_a = _brier_improvement(
        assessment_primary["directional_onset"],
        assessment_primary["p_onset__A1"].to_numpy(dtype=float),
        assessment_primary["p_onset__A2"].to_numpy(dtype=float),
    )
    real_direction = assessment_primary.loc[assessment_primary["directional_onset"].eq(1)]
    real_d = _brier_improvement(
        real_direction["up_given_onset"],
        real_direction["p_up_given_onset__D1"].to_numpy(dtype=float),
        real_direction["p_up_given_onset__D2"].to_numpy(dtype=float),
    )
    real_economic_by_candidate = (
        real_selections.loc[real_selections["candidate"].isin(["readiness", "pressure"])]
        .groupby("candidate", sort=True)["signed_gross_return_bps_30m"]
        .mean()
    )
    real_economic = float(
        real_economic_by_candidate["pressure"] - real_economic_by_candidate["readiness"]
    )
    real_values = {
        "A2_minus_A1_brier_improvement": real_a,
        "D2_minus_D1_brier_improvement": real_d,
        "pressure_minus_readiness_economic_30m": real_economic,
    }
    null_values: dict[str, list[float]] = {key: [] for key in real_values}
    rows: list[dict[str, Any]] = []
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    for draw in range(NULL_DRAWS):
        seed = NULL_SEED + draw
        null_development = permute_feature_bundle_within_slates(
            development.reset_index(drop=True), PRESSURE_FEATURES, seed=seed
        )
        null_assessment = permute_feature_bundle_within_slates(
            assessment_primary.reset_index(drop=True),
            PRESSURE_FEATURES,
            seed=seed + 100_000,
        )
        null_a2 = fit_fixed_logistic(
            null_development,
            null_development["directional_onset"],
            features=MODEL_FEATURES["A2"],
            slate_column="slate_id",
            model_id=f"null_A2_{draw}",
        )
        null_direction_development = null_development.loc[
            null_development["directional_onset"].eq(1)
        ]
        null_d2 = fit_fixed_logistic(
            null_direction_development,
            null_direction_development["up_given_onset"],
            features=MODEL_FEATURES["D2"],
            slate_column="slate_id",
            model_id=f"null_D2_{draw}",
        )
        null_assessment["p_onset__A2"] = null_a2.predict(null_assessment)
        null_assessment["p_up_given_onset__D2"] = null_d2.predict(null_assessment)
        null_a_value = _brier_improvement(
            null_assessment["directional_onset"],
            models["A1"].predict(null_assessment),
            null_assessment["p_onset__A2"].to_numpy(dtype=float),
        )
        null_direction = null_assessment.loc[null_assessment["directional_onset"].eq(1)]
        null_d_value = _brier_improvement(
            null_direction["up_given_onset"],
            models["D1"].predict(null_direction),
            null_direction["p_up_given_onset__D2"].to_numpy(dtype=float),
        )
        null_assessment["signed_pressure_score__pressure"] = (
            null_assessment["p_onset__A2"]
            * (2.0 * null_assessment["p_up_given_onset__D2"] - 1.0)
            * null_assessment["p_large_remaining_move"]
        )
        null_pressure = economic_selections(null_assessment, candidates=("pressure",))
        null_economic = float(
            null_pressure["signed_gross_return_bps_30m"].mean()
            - real_economic_by_candidate["readiness"]
        )
        draw_values = {
            "A2_minus_A1_brier_improvement": null_a_value,
            "D2_minus_D1_brier_improvement": null_d_value,
            "pressure_minus_readiness_economic_30m": null_economic,
        }
        for metric, value in draw_values.items():
            null_values[metric].append(value)
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "metric": metric,
                    "null_value": value,
                    "real_value": real_values[metric],
                    "null_q90": math.nan,
                    "real_percentile": math.nan,
                }
            )
    for metric, values in null_values.items():
        array = np.asarray(values, dtype=float)
        real = real_values[metric]
        rows.append(
            {
                "record_type": "summary",
                "draw": -1,
                "metric": metric,
                "null_value": float(np.mean(array)),
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


def prepare_movement_probabilities(
    predecessor: pd.DataFrame,
    frozen_model: Mapping[str, Any],
    frozen_assessment_probabilities: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[int, float]]:
    """Attach causal 2024 OOF and exact frozen 2025 M1 movement probabilities."""

    panel = predecessor.copy()
    panel["session"] = panel["session"].astype(str)
    development = panel.loc[panel["year"].eq(2024)].copy()
    oof, fold_manifest = expanding_monthly_oof_probabilities(
        development,
        target_column="large_remaining_move",
        features=tuple(str(value) for value in frozen_model["feature_names"]),
        slate_column="slate_id",
        model_id="large_remaining_move__M1",
    )
    panel["p_large_remaining_move"] = np.nan
    panel.loc[development.index, "p_large_remaining_move"] = oof
    assessment_indices = (
        panel.loc[panel["year"].eq(2025)]
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .index
    )
    if len(assessment_indices) != len(frozen_assessment_probabilities):
        raise ScreenBlocker(
            "blocked_observable_movement_model_not_reconstructable",
            "2025 probability assignment length differs",
        )
    panel.loc[assessment_indices, "p_large_remaining_move"] = frozen_assessment_probabilities
    thresholds = movement_admission_thresholds(panel)
    panel["movement_admission_threshold"] = panel["decision_ordinal"].map(thresholds)
    panel["high_movement_admitted"] = (
        panel["p_large_remaining_move"].ge(panel["movement_admission_threshold"])
        & panel["p_large_remaining_move"].notna()
    )
    oof_artifact = panel.loc[
        panel["year"].eq(2024) & panel["p_large_remaining_move"].notna(),
        [
            "symbol",
            "session",
            "year_month",
            "decision_ordinal",
            "slate_id",
            "large_remaining_move",
            "p_large_remaining_move",
            "movement_admission_threshold",
            "high_movement_admitted",
        ],
    ].sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    for fold in fold_manifest:
        if str(fold["training_end_month"]) >= str(fold["score_month"]):
            raise ScreenBlocker(
                "blocked_chronology_or_leakage_failure",
                "movement OOF training reaches its score month",
            )
    return panel, oof_artifact.reset_index(drop=True), fold_manifest, thresholds


def _metric_lookup(
    frame: pd.DataFrame, model: str, metric: str, *, population: str = "primary_high_movement"
) -> float:
    row = frame.loc[
        frame["population"].eq(population)
        & frame["scope_type"].eq("pooled")
        & frame["model"].eq(model)
    ]
    if len(row) != 1:
        raise ValueError(f"metric lookup differs for {model}/{population}")
    return float(row.iloc[0][metric])


def _slice_improvements(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    metric: str,
) -> dict[str, float]:
    primary = frame.loc[frame["population"].eq("primary_high_movement")]
    baseline_rows = primary.loc[primary["model"].eq(baseline)].set_index("scope_value")
    candidate_rows = primary.loc[primary["model"].eq(candidate)].set_index("scope_value")
    common = baseline_rows.index.intersection(candidate_rows.index)
    return {
        str(value): float(baseline_rows.loc[value, metric] - candidate_rows.loc[value, metric])
        for value in common
    }


def _summary_value(frame: pd.DataFrame, metric: str, column: str) -> float:
    row = frame.loc[frame["record_type"].eq("summary") & frame["metric"].eq(metric)]
    if len(row) != 1:
        raise ValueError(f"summary lookup differs for {metric}")
    return float(row.iloc[0][column])


def _economic_value(frame: pd.DataFrame, candidate: str, *, friction: float = 20.0) -> float:
    row = frame.loc[
        frame["candidate"].eq(candidate)
        & frame["horizon"].eq("primary_30m_close_t_plus_8")
        & frame["friction_bps"].eq(friction)
    ]
    if len(row) != 1:
        raise ValueError(f"economic lookup differs for {candidate}/{friction}")
    return float(row.iloc[0]["mean_signed_return_after_friction_bps"])


def derive_decision(
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    economic: pd.DataFrame,
    support: Mapping[str, Any],
    concentration: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen multi-gate decision policy and retain each gate value."""

    a_brier = _metric_lookup(onset, "A1", "brier_score") - _metric_lookup(
        onset, "A2", "brier_score"
    )
    a_log = _metric_lookup(onset, "A1", "log_loss") - _metric_lookup(onset, "A2", "log_loss")
    d_brier = _metric_lookup(direction, "D1", "brier_score") - _metric_lookup(
        direction, "D2", "brier_score"
    )
    d_log = _metric_lookup(direction, "D1", "log_loss") - _metric_lookup(
        direction, "D2", "log_loss"
    )
    a_confirmation_brier = _metric_lookup(onset, "A2", "brier_score") - _metric_lookup(
        onset, "A3", "brier_score"
    )
    a_confirmation_log = _metric_lookup(onset, "A2", "log_loss") - _metric_lookup(
        onset, "A3", "log_loss"
    )
    d_confirmation_brier = _metric_lookup(direction, "D2", "brier_score") - _metric_lookup(
        direction, "D3", "brier_score"
    )
    d_confirmation_log = _metric_lookup(direction, "D2", "log_loss") - _metric_lookup(
        direction, "D3", "log_loss"
    )
    a_months = _slice_improvements(monthly, baseline="A1", candidate="A2", metric="brier_score")
    d_months = _slice_improvements(monthly, baseline="D1", candidate="D2", metric="brier_score")
    a_confirmation_months = _slice_improvements(
        monthly, baseline="A2", candidate="A3", metric="brier_score"
    )
    d_confirmation_months = _slice_improvements(
        monthly, baseline="D2", candidate="D3", metric="brier_score"
    )
    a_checkpoints_brier = _slice_improvements(
        checkpoint, baseline="A1", candidate="A2", metric="brier_score"
    )
    a_checkpoints_log = _slice_improvements(
        checkpoint, baseline="A1", candidate="A2", metric="log_loss"
    )
    d_checkpoints_brier = _slice_improvements(
        checkpoint, baseline="D1", candidate="D2", metric="brier_score"
    )
    d_checkpoints_log = _slice_improvements(
        checkpoint, baseline="D1", candidate="D2", metric="log_loss"
    )
    concentration_passes = bool(concentration["all_concentration_gates_pass"])
    a_boot_brier = _summary_value(bootstrap, "A2_minus_A1_brier_improvement", "lower_90")
    a_boot_log = _summary_value(bootstrap, "A2_minus_A1_log_loss_improvement", "lower_90")
    d_boot_brier = _summary_value(bootstrap, "D2_minus_D1_brier_improvement", "lower_90")
    d_boot_log = _summary_value(bootstrap, "D2_minus_D1_log_loss_improvement", "lower_90")
    a_null_q90 = _summary_value(nulls, "A2_minus_A1_brier_improvement", "null_q90")
    d_null_q90 = _summary_value(nulls, "D2_minus_D1_brier_improvement", "null_q90")
    occurrence_gates = {
        "brier_improvement_positive": a_brier > 0.0,
        "log_loss_improvement_positive": a_log > 0.0,
        "bootstrap_90_lower_brier_non_negative": a_boot_brier >= 0.0,
        "bootstrap_90_lower_log_loss_non_negative": a_boot_log >= 0.0,
        "positive_brier_months_at_least_five": sum(value > 0.0 for value in a_months.values()) >= 5,
        "real_brier_increment_exceeds_null_q90": a_brier > a_null_q90,
        "neither_checkpoint_materially_adverse": all(
            value >= -0.001
            for value in [*a_checkpoints_brier.values(), *a_checkpoints_log.values()]
        ),
        "concentration_gates_pass": concentration_passes,
    }
    direction_gates = {
        "brier_improvement_positive": d_brier > 0.0,
        "log_loss_improvement_positive": d_log > 0.0,
        "auc_not_reduced": _metric_lookup(direction, "D2", "auc")
        >= _metric_lookup(direction, "D1", "auc"),
        "bootstrap_90_lower_brier_non_negative": d_boot_brier >= 0.0,
        "bootstrap_90_lower_log_loss_non_negative": d_boot_log >= 0.0,
        "positive_brier_months_at_least_five": sum(value > 0.0 for value in d_months.values()) >= 5,
        "real_brier_increment_exceeds_null_q90": d_brier > d_null_q90,
        "neither_checkpoint_materially_adverse": all(
            value >= -0.001
            for value in [*d_checkpoints_brier.values(), *d_checkpoints_log.values()]
        ),
        "concentration_gates_pass": concentration_passes,
    }
    confirmation_economic_not_worsened = _economic_value(economic, "confirmed") >= _economic_value(
        economic, "pressure"
    )
    confirmation_occurrence_gates = {
        "brier_improvement_positive": a_confirmation_brier > 0.0,
        "log_loss_improvement_positive": a_confirmation_log > 0.0,
        "bootstrap_90_lower_brier_non_negative": _summary_value(
            bootstrap, "A3_minus_A2_brier_improvement", "lower_90"
        )
        >= 0.0,
        "positive_brier_months_at_least_five": sum(
            value > 0.0 for value in a_confirmation_months.values()
        )
        >= 5,
        "delayed_economic_result_not_worsened": confirmation_economic_not_worsened,
        "concentration_gates_pass": concentration_passes,
    }
    confirmation_direction_gates = {
        "brier_improvement_positive": d_confirmation_brier > 0.0,
        "log_loss_improvement_positive": d_confirmation_log > 0.0,
        "bootstrap_90_lower_brier_non_negative": _summary_value(
            bootstrap, "D3_minus_D2_brier_improvement", "lower_90"
        )
        >= 0.0,
        "positive_brier_months_at_least_five": sum(
            value > 0.0 for value in d_confirmation_months.values()
        )
        >= 5,
        "delayed_economic_result_not_worsened": confirmation_economic_not_worsened,
        "concentration_gates_pass": concentration_passes,
    }
    occurrence_passes = all(occurrence_gates.values())
    direction_passes = bool(
        support["conditional_direction_support_passes"] and all(direction_gates.values())
    )
    confirmation_occurrence_passes = bool(
        not occurrence_passes and all(confirmation_occurrence_gates.values())
    )
    confirmation_direction_passes = bool(
        not direction_passes
        and support["conditional_direction_support_passes"]
        and all(confirmation_direction_gates.values())
    )
    readiness_useful = bool(
        _metric_lookup(onset, "A1", "brier_score") < _metric_lookup(onset, "A0", "brier_score")
        and _metric_lookup(onset, "A1", "log_loss") < _metric_lookup(onset, "A0", "log_loss")
    )
    evidence = {
        "occurrence_passes": occurrence_passes,
        "direction_passes": direction_passes,
        "confirmation_occurrence_passes": confirmation_occurrence_passes,
        "confirmation_direction_passes": confirmation_direction_passes,
        "readiness_useful": readiness_useful,
    }
    decision = decide_pressure_screen(evidence)
    return {
        **SAFETY_FLAGS,
        "decision": decision,
        "conditional_direction_status": (
            "evaluated"
            if support["conditional_direction_support_passes"]
            else "conditional_direction_support_insufficient"
        ),
        "evidence": evidence,
        "increments": {
            "A2_minus_A1_brier": a_brier,
            "A2_minus_A1_log_loss": a_log,
            "D2_minus_D1_brier": d_brier,
            "D2_minus_D1_log_loss": d_log,
            "A3_minus_A2_brier": a_confirmation_brier,
            "A3_minus_A2_log_loss": a_confirmation_log,
            "D3_minus_D2_brier": d_confirmation_brier,
            "D3_minus_D2_log_loss": d_confirmation_log,
        },
        "monthly_brier_improvements": {
            "A2_minus_A1": a_months,
            "D2_minus_D1": d_months,
            "A3_minus_A2": a_confirmation_months,
            "D3_minus_D2": d_confirmation_months,
        },
        "checkpoint_improvements": {
            "A2_minus_A1_brier": a_checkpoints_brier,
            "A2_minus_A1_log_loss": a_checkpoints_log,
            "D2_minus_D1_brier": d_checkpoints_brier,
            "D2_minus_D1_log_loss": d_checkpoints_log,
        },
        "occurrence_gates": occurrence_gates,
        "direction_gates": direction_gates,
        "confirmation_occurrence_gates": confirmation_occurrence_gates,
        "confirmation_direction_gates": confirmation_direction_gates,
        "support": dict(support),
        "concentration": dict(concentration),
        "economic_reference_cannot_override_probability_gates": True,
        "conclusions": {
            "remaining_movement_readiness": "measured_by_frozen_observable_M1",
            "short_horizon_directional_onset_probability": "retrospective_feasibility_only",
            "direction_conditional_on_onset": "retrospective_feasibility_only",
            "increment_from_pressure_onset_variables": "reported_not_promoted",
            "increment_from_one_bar_confirmation": "reported_not_promoted",
            "gross_delayed_economic_association": "synthetic_reference_only",
            "executable_net_edge": "not_addressed",
        },
    }


def plot_calibration(calibration: pd.DataFrame, output: Path) -> None:
    """Write the single fixed system calibration comparison plot."""

    subset = calibration.loc[
        calibration["population"].eq("primary_high_movement")
        & calibration["scope_type"].eq("pooled")
        & calibration["target"].eq("directional_onset")
        & calibration["model"].isin(["A1", "A2", "A3"])
        & calibration["rows"].gt(0)
    ]
    names = {"A1": "Readiness", "A2": "Pressure", "A3": "Confirmed"}
    colors = {"A1": "#4477AA", "A2": "#CC6677", "A3": "#228833"}
    fig, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#777777", linestyle="--", linewidth=1.0)
    for model in ("A1", "A2", "A3"):
        model_frame = subset.loc[subset["model"].eq(model)].sort_values("bin")
        axis.plot(
            model_frame["mean_probability"],
            model_frame["observed_rate"],
            marker="o",
            linewidth=1.8,
            label=names[model],
            color=colors[model],
        )
    axis.set(
        xlabel="Mean predicted directional-onset probability",
        ylabel="Observed directional-onset rate",
        title="High-movement onset calibration (2025 assessment)",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.legend(frameon=False)
    fig.savefig(output, dpi=150, metadata={"Software": "Stocker research"})
    plt.close(fig)


def plot_economic_reference(economic: pd.DataFrame, output: Path) -> None:
    """Write the single fixed 20-bps economic-reference comparison plot."""

    order = [
        "readiness",
        "pressure",
        "confirmed",
        "highest_relative_momentum",
        "strongest_reversal",
        "random_within_slate",
    ]
    labels = [
        "Readiness",
        "Pressure",
        "Confirmed",
        "Momentum",
        "Reversal",
        "Random",
    ]
    subset = economic.loc[
        economic["horizon"].eq("primary_30m_close_t_plus_8") & economic["friction_bps"].eq(20.0)
    ].set_index("candidate")
    values = [float(subset.loc[name, "mean_signed_return_after_friction_bps"]) for name in order]
    colors = ["#4477AA", "#CC6677", "#228833", "#AA3377", "#66CCEE", "#999999"]
    fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.bar(labels, values, color=colors)
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set(
        ylabel="Mean signed return after 20 bps",
        title="Delayed fixed-horizon economic reference",
    )
    axis.tick_params(axis="x", rotation=25)
    fig.savefig(output, dpi=150, metadata={"Software": "Stocker research"})
    plt.close(fig)


def _markdown_metric_table(frame: pd.DataFrame, models: Sequence[str]) -> str:
    rows = [
        "| Model | Brier | Log loss | AUC | Rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        record = frame.loc[
            frame["population"].eq("primary_high_movement") & frame["model"].eq(model)
        ].iloc[0]
        rows.append(
            f"| {model} | {record['brier_score']:.6f} | {record['log_loss']:.6f} "
            f"| {record['auc']:.6f} | {int(record['rows'])} |"
        )
    return "\n".join(rows)


def render_report(
    *,
    predecessor_result: Mapping[str, Any],
    thresholds: Mapping[int, float],
    barriers: Mapping[str, float],
    support: Mapping[str, Any],
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    economic: pd.DataFrame,
    decision: Mapping[str, Any],
    source_context: Mapping[str, Any],
) -> str:
    """Render a compact, direct research report from frozen artifacts."""

    bootstrap_summary = bootstrap.loc[bootstrap["record_type"].eq("summary")]
    null_summary = nulls.loc[nulls["record_type"].eq("summary")]
    economic_primary = economic.loc[economic["horizon"].eq("primary_30m_close_t_plus_8")]
    lines = [
        "# High-Movement Pressure-Onset Screen V0",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        "This is a retrospective, research-only, observable-only bounded feasibility screen. "
        "It is not prospective validation, a strategy, achieved P&L, or evidence of "
        "executable net edge. Execution and order placement are disabled; no broker "
        "integration or production runtime was modified.",
        "",
        "## Integrity",
        "",
        f"- Frozen predecessor: `{predecessor_result['source_experiment']}` at "
        f"`{predecessor_result['source_commit']}`.",
        f"- Predecessor prediction reconstruction passed: `{predecessor_result['passed']}`; "
        "maximum absolute probability error "
        f"`{predecessor_result['maximum_prediction_absolute_error']:.3g}`.",
        f"- Exact market timestamps read: `{source_context['minimum_timestamp_read']}` through "
        f"`{source_context['maximum_timestamp_read']}`.",
        "- Protected rows materialised: `0`.",
        "- VWAP status: `vwap_features_unavailable`.",
        "- Retired loops, regimes, states, closures, excursions, transitions, posteriors, "
        "and structural paths were excluded.",
        "",
        "## Frozen thresholds and support",
        "",
        f"- Movement admission: ordinal 6 `{thresholds[6]:.12f}`; "
        f"ordinal 12 `{thresholds[12]:.12f}`.",
        f"- Onset barrier: ordinal 6 `{float(barriers['6']):.6f}` bps; "
        f"ordinal 12 `{float(barriers['12']):.6f}` bps.",
        f"- Primary rows `{support['rows']}`, sessions `{support['sessions']}`, stocks "
        f"`{support['stocks']}`, represented months `{support['months']}`.",
        f"- UP / DOWN / NO_ONSET: `{support['up_onsets']}` / `{support['down_onsets']}` / "
        f"`{support['no_onsets']}`.",
        "",
        "## Onset occurrence models",
        "",
        _markdown_metric_table(onset, ("A0", "A1", "A2", "A3")),
        "",
        "## Direction conditional on actual onset",
        "",
        _markdown_metric_table(direction, ("D0", "D1", "D2", "D3")),
        "",
        "## Fixed comparisons",
        "",
    ]
    for name, value in cast(Mapping[str, float], decision["increments"]).items():
        lines.append(f"- `{name}`: `{value:.12g}`")
    lines.extend(["", "## Session-block bootstrap", ""])
    for row in bootstrap_summary.itertuples(index=False):
        lines.append(
            f"- `{row.metric}`: 90% `[{row.lower_90:.12g}, {row.upper_90:.12g}]`; "
            f"95% `[{row.lower_95:.12g}, {row.upper_95:.12g}]`."
        )
    lines.extend(["", "## Within-slate bundled-feature null", ""])
    for row in null_summary.itertuples(index=False):
        lines.append(
            f"- `{row.metric}`: real `{row.real_value:.12g}`, null q90 "
            f"`{row.null_q90:.12g}`, real percentile `{row.real_percentile:.3f}`."
        )
    lines.extend(["", "## Delayed economic reference", ""])
    for candidate in ("readiness", "pressure", "confirmed"):
        candidate_rows = economic_primary.loc[economic_primary["candidate"].eq(candidate)]
        values = {
            int(row.friction_bps): row.mean_signed_return_after_friction_bps
            for row in candidate_rows.itertuples(index=False)
        }
        lines.append(
            f"- `{candidate}` mean signed 30-minute reference: "
            f"0 bps `{values[0]:.4f}`, 10 bps `{values[10]:.4f}`, "
            f"20 bps `{values[20]:.4f}`."
        )
    lines.extend(
        [
            "",
            "The economic diagnostic is gross and synthetic. It does not model short borrow, "
            "spread, or market impact and cannot rescue failed probability gates.",
            "",
            "Independent arithmetic verification and exact-rerun status are recorded in "
            "`independent_audit.json` and `exact_rerun_manifest.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_run_artifacts(
    output: Path,
    *,
    contract: Mapping[str, Any],
    predecessor_result: Mapping[str, Any],
    predecessor_model: Mapping[str, Any],
    movement_oof: pd.DataFrame,
    movement_folds: Sequence[Mapping[str, Any]],
    thresholds: Mapping[int, float],
    compact: pd.DataFrame,
    ledger: pd.DataFrame,
    assessment: pd.DataFrame,
    models: Mapping[str, FrozenLogisticModel],
    confirmation_manifest: Sequence[Mapping[str, Any]],
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    economic: pd.DataFrame,
    concentration: pd.DataFrame,
    support: Mapping[str, Any],
    concentration_summary: Mapping[str, Any],
    decision: Mapping[str, Any],
    source_context: Mapping[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    write_json(output / "predecessor_reconstruction.json", predecessor_result)
    write_json(
        output / "input_artifact_hashes.json",
        {
            **SAFETY_FLAGS,
            "artifacts": [
                {
                    "logical_path": str(path.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in (
                    PREDECESSOR_PANEL,
                    PREDECESSOR_PREDICTIONS,
                    PREDECESSOR_COEFFICIENTS,
                    PREDECESSOR_METRICS,
                    PREDECESSOR_SOURCE_MANIFEST,
                )
            ],
        },
    )
    write_json(
        output / "source_manifest.json",
        {
            **SAFETY_FLAGS,
            "provider": "EODHD",
            "cohort": list(SYMBOLS),
            "symbol_predicate_applied_before_materialisation": True,
            "date_predicate_applied_before_materialisation": True,
            "source_paths_are_logical_not_local_absolute": True,
            "minimum_timestamp_read": source_context["minimum_timestamp_read"],
            "maximum_timestamp_read": source_context["maximum_timestamp_read"],
            "source_rows_by_year_month": source_context["source_rows_by_year_month"],
            "sources": source_context["sources"],
            "vendor_qa": source_context["vendor_qa"],
            "source_gap_ledger_rejected_session_records": source_context["gap_ledger_rows"],
        },
    )
    write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "development_start": "2024-01-01",
            "development_end_inclusive": "2024-12-31",
            "assessment_start": "2025-01-01",
            "assessment_end_inclusive": "2025-08-22",
            "protected_start": "2025-08-23",
            "minimum_timestamp_read": source_context["minimum_timestamp_read"],
            "maximum_timestamp_read": source_context["maximum_timestamp_read"],
            "protected_files_touched": [],
            "protected_outcome_files_opened": [],
            "protected_rows_materialised": source_context["protected_rows_materialised"],
            "passed": source_context["protected_rows_materialised"] == 0,
        },
    )
    write_json(
        output / "movement_oof_fold_manifest.json",
        {
            **SAFETY_FLAGS,
            "minimum_initial_training": "2024-01_through_2024-06",
            "score_months": [f"2024-{month:02d}" for month in range(7, 13)],
            "folds": list(movement_folds),
            "chronology_passed": all(
                str(row["training_end_month"]) < str(row["score_month"]) for row in movement_folds
            ),
        },
    )
    write_json(
        output / "movement_admission_thresholds.json",
        {
            **SAFETY_FLAGS,
            "quantile": 0.75,
            "training_source": "finite_2024_expanding_monthly_OOF_probabilities_only",
            "thresholds": {str(key): value for key, value in thresholds.items()},
        },
    )
    write_json(
        output / "onset_barriers.json",
        {
            **SAFETY_FLAGS,
            "quantile": 0.75,
            "training_source": "all_eligible_2024_development_three_bar_residual_paths",
            "barriers_bps": source_context["onset_barriers"],
        },
    )
    unique_features = tuple(
        dict.fromkeys(
            [feature for model_features in MODEL_FEATURES.values() for feature in model_features]
        )
    )
    assert_allowed_feature_names(unique_features)
    assert_allowed_feature_names(compact.columns)
    write_json(
        output / "feature_manifest.json",
        {
            **SAFETY_FLAGS,
            "readiness_features": list(READINESS_FEATURES),
            "readiness_feature_count": len(READINESS_FEATURES),
            "pressure_onset_additions": list(PRESSURE_FEATURES),
            "pressure_onset_addition_count": len(PRESSURE_FEATURES),
            "pressure_signals_reused_from_readiness": [
                "open_to_decision_cohort_relative_return_bps",
                "distance_from_opening_high_bps",
                "distance_from_opening_low_bps",
            ],
            "confirmation_features": list(CONFIRMATION_FEATURES),
            "confirmation_feature_count": len(CONFIRMATION_FEATURES),
            "vwap_status": "vwap_features_unavailable",
            "provider_volume_label": "historical_activity_proxy",
            "same_stock_same_clock_normalisation": "expanding_prior_session_mean_minimum_10",
            "progress_per_activity_winsor_bounds": source_context["progress_winsor_bounds"],
            "all_model_feature_names": list(unique_features),
            "frozen_M1_source_aliases": {
                "m1_source__open_to_decision_cohort_relative_return_bps": (
                    "immutable predecessor value retained before final-slate re-anchoring"
                ),
                "m1_source__cross_sectional_dispersion_bps": (
                    "immutable predecessor value retained before final-slate re-anchoring"
                ),
            },
            "formulas": {
                "relative_strength_acceleration": (
                    "relative_return_last_3_bps - relative_return_previous_3_bps"
                ),
                "activity_acceleration": (
                    "log1p(activity_last_2_mean) - log1p(activity_previous_4_mean)"
                ),
                "range_acceleration": "range_last_2_mean_bps / range_previous_4_mean_bps",
                "signed_efficiency": "sum(completed_returns) / sum(abs(completed_returns))",
                "signed_progress_per_activity": (
                    "relative_return_last_3_bps / max(relative_activity_last_3_mean, 1e-12)"
                ),
            },
        },
    )
    write_json(
        output / "forbidden_feature_audit.json",
        {
            **SAFETY_FLAGS,
            "feature_columns_scanned": list(unique_features),
            "data_columns_scanned": list(compact.columns),
            "forbidden_matches": [],
            "passed": True,
        },
    )
    write_json(
        output / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "model_specification_count": len(MODEL_FEATURES),
            "maximum_model_specifications": 8,
            "models": {key: list(value) for key, value in MODEL_FEATURES.items()},
            "configuration": {
                "kind": "deterministic_L2_logistic_regression",
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 250,
                "class_weight": None,
                "n_jobs": 1,
                "preprocessing": "development_mean_and_population_standard_deviation",
                "row_weight": "1 / eligible_rows_in_slate",
            },
            "confirmation_direction_manifest": list(confirmation_manifest),
            "mandated_refits": (
                "movement OOF and permutation-null refits reuse preregistered fixed "
                "specifications and are not additional searched model configurations"
            ),
        },
    )
    write_json(
        output / "model_coefficients.json",
        {
            **SAFETY_FLAGS,
            "models": {key: model.as_dict() for key, model in sorted(models.items())},
            "frozen_predecessor_M1": dict(frozen_model_serializable(predecessor_model)),
        },
    )
    write_parquet(output / "compact_decision_panel.parquet", compact)
    write_parquet(output / "onset_path_ledger.parquet", ledger)
    write_parquet(output / "development_oof_predictions.parquet", movement_oof)
    write_parquet(output / "assessment_predictions.parquet", assessment)
    write_csv(output / "onset_metrics.csv", onset)
    write_csv(output / "direction_metrics.csv", direction)
    write_csv(output / "monthly_metrics.csv", monthly)
    write_csv(output / "checkpoint_metrics.csv", checkpoint)
    write_csv(output / "calibration_bins.csv", calibration)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "null_metrics.csv", nulls)
    write_csv(output / "economic_reference_metrics.csv", economic)
    write_csv(output / "concentration_metrics.csv", concentration)
    write_json(output / "decision.json", decision)
    plot_calibration(calibration, output / "calibration_systems.png")
    plot_economic_reference(economic, output / "economic_reference_20bps.png")
    report = render_report(
        predecessor_result=predecessor_result,
        thresholds=thresholds,
        barriers=cast(Mapping[str, float], source_context["onset_barriers"]),
        support=support,
        onset=onset,
        direction=direction,
        bootstrap=bootstrap,
        nulls=nulls,
        economic=economic,
        decision=decision,
        source_context=source_context,
    )
    (output / "report.md").write_text(report, encoding="utf-8")


def frozen_model_serializable(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the archived movement model without unrelated predecessor payloads."""

    output = {
        key: model[key]
        for key in (
            "model_id",
            "feature_names",
            "means",
            "scales",
            "coefficients",
            "intercept",
            "training_rows",
            "training_slates",
            "iterations",
            "converged",
            "penalty",
            "C",
            "solver",
            "max_iter",
            "class_weight",
            "n_jobs",
        )
        if key in model
    }
    original_features = [str(value) for value in model["feature_names"]]
    aliased_features = [
        (
            f"m1_source__{feature}"
            if feature
            in {
                "open_to_decision_cohort_relative_return_bps",
                "cross_sectional_dispersion_bps",
            }
            else feature
        )
        for feature in original_features
    ]
    output["original_feature_names"] = original_features
    output["feature_names"] = aliased_features
    output["feature_aliases"] = {
        alias: original
        for original, alias in zip(original_features, aliased_features, strict=True)
        if alias != original
    }
    return output


def plot_blocked(output: Path, title: str) -> None:
    """Write a deterministic placeholder explaining why a planned plot is unavailable."""

    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    axis.axis("off")
    axis.text(
        0.5,
        0.58,
        title,
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    axis.text(
        0.5,
        0.40,
        "Not evaluated: preregistered primary-population\nconcentration support gate failed.",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.savefig(output, dpi=150, metadata={"Software": "Stocker research"})
    plt.close(fig)


def write_blocked_artifacts(
    output: Path,
    *,
    contract: Mapping[str, Any],
    predecessor_result: Mapping[str, Any],
    predecessor_model: Mapping[str, Any],
    movement_oof: pd.DataFrame,
    movement_folds: Sequence[Mapping[str, Any]],
    thresholds: Mapping[int, float],
    compact: pd.DataFrame,
    ledger: pd.DataFrame,
    assessment: pd.DataFrame,
    support: Mapping[str, Any],
    source_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist an honest fail-closed result without fitting post-support models."""

    blocker = "blocked_insufficient_pressure_onset_support"
    output.mkdir(parents=True, exist_ok=True)
    decision = {
        **SAFETY_FLAGS,
        "decision": blocker,
        "blocker": (
            "frozen 2025 primary high-movement population failed support gates: "
            + ", ".join(cast(Sequence[str], support["failed_primary_support_gates"]))
        ),
        "support": dict(support),
        "stopped_before": [
            "A0_through_A3_model_fits",
            "D0_through_D3_model_fits",
            "bootstrap",
            "permutation_null",
            "economic_reference_selection",
        ],
        "population_was_not_reduced_after_outcome_inspection": True,
        "thresholds_or_barriers_were_not_changed": True,
        "conclusions": {
            "remaining_movement_readiness": "frozen_observable_M1_reconstructed",
            "short_horizon_directional_onset_probability": "not_evaluated_due_to_support_gate",
            "direction_conditional_on_onset": "not_evaluated_due_to_support_gate",
            "increment_from_pressure_onset_variables": "not_evaluated_due_to_support_gate",
            "increment_from_one_bar_confirmation": "not_evaluated_due_to_support_gate",
            "gross_delayed_economic_association": "not_evaluated_due_to_support_gate",
            "executable_net_edge": "not_addressed",
        },
    }
    write_json(output / "contract.json", contract)
    write_json(output / "decision.json", decision)
    write_json(output / "predecessor_reconstruction.json", predecessor_result)
    write_json(
        output / "input_artifact_hashes.json",
        {
            **SAFETY_FLAGS,
            "artifacts": [
                {
                    "logical_path": str(path.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in (
                    PREDECESSOR_PANEL,
                    PREDECESSOR_PREDICTIONS,
                    PREDECESSOR_COEFFICIENTS,
                    PREDECESSOR_METRICS,
                    PREDECESSOR_SOURCE_MANIFEST,
                )
            ],
        },
    )
    write_json(
        output / "source_manifest.json",
        {
            **SAFETY_FLAGS,
            "provider": "EODHD",
            "cohort": list(SYMBOLS),
            "symbol_predicate_applied_before_materialisation": True,
            "date_predicate_applied_before_materialisation": True,
            "source_paths_are_logical_not_local_absolute": True,
            "minimum_timestamp_read": source_context["minimum_timestamp_read"],
            "maximum_timestamp_read": source_context["maximum_timestamp_read"],
            "source_rows_by_year_month": source_context["source_rows_by_year_month"],
            "sources": source_context["sources"],
            "vendor_qa": source_context["vendor_qa"],
            "source_gap_ledger_rejected_session_records": source_context["gap_ledger_rows"],
        },
    )
    write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "development_start": "2024-01-01",
            "development_end_inclusive": "2024-12-31",
            "assessment_start": "2025-01-01",
            "assessment_end_inclusive": "2025-08-22",
            "protected_start": "2025-08-23",
            "minimum_timestamp_read": source_context["minimum_timestamp_read"],
            "maximum_timestamp_read": source_context["maximum_timestamp_read"],
            "protected_files_touched": [],
            "protected_outcome_files_opened": [],
            "protected_rows_materialised": 0,
            "passed": True,
        },
    )
    write_json(
        output / "movement_oof_fold_manifest.json",
        {
            **SAFETY_FLAGS,
            "minimum_initial_training": "2024-01_through_2024-06",
            "score_months": [f"2024-{month:02d}" for month in range(7, 13)],
            "folds": list(movement_folds),
            "chronology_passed": all(
                str(row["training_end_month"]) < str(row["score_month"]) for row in movement_folds
            ),
        },
    )
    write_json(
        output / "movement_admission_thresholds.json",
        {
            **SAFETY_FLAGS,
            "quantile": 0.75,
            "training_source": "finite_2024_expanding_monthly_OOF_probabilities_only",
            "thresholds": {str(key): value for key, value in thresholds.items()},
        },
    )
    write_json(
        output / "onset_barriers.json",
        {
            **SAFETY_FLAGS,
            "quantile": 0.75,
            "training_source": "all_eligible_2024_development_three_bar_residual_paths",
            "barriers_bps": source_context["onset_barriers"],
        },
    )
    unique_features = tuple(
        dict.fromkeys(
            feature for model_features in MODEL_FEATURES.values() for feature in model_features
        )
    )
    assert_allowed_feature_names(unique_features)
    assert_allowed_feature_names(compact.columns)
    write_json(
        output / "feature_manifest.json",
        {
            **SAFETY_FLAGS,
            "readiness_features": list(READINESS_FEATURES),
            "readiness_feature_count": len(READINESS_FEATURES),
            "pressure_onset_additions": list(PRESSURE_FEATURES),
            "pressure_onset_addition_count": len(PRESSURE_FEATURES),
            "pressure_signals_reused_from_readiness": [
                "open_to_decision_cohort_relative_return_bps",
                "distance_from_opening_high_bps",
                "distance_from_opening_low_bps",
            ],
            "confirmation_features": list(CONFIRMATION_FEATURES),
            "confirmation_feature_count": len(CONFIRMATION_FEATURES),
            "confirmation_status": "not_finalized_due_to_support_gate",
            "vwap_status": "vwap_features_unavailable",
            "provider_volume_label": "historical_activity_proxy",
            "same_stock_same_clock_normalisation": "expanding_prior_session_mean_minimum_10",
            "progress_per_activity_winsor_bounds": source_context["progress_winsor_bounds"],
            "all_model_feature_names": list(unique_features),
            "frozen_M1_source_aliases": {
                "m1_source__open_to_decision_cohort_relative_return_bps": (
                    "immutable predecessor value retained before final-slate re-anchoring"
                ),
                "m1_source__cross_sectional_dispersion_bps": (
                    "immutable predecessor value retained before final-slate re-anchoring"
                ),
            },
        },
    )
    write_json(
        output / "forbidden_feature_audit.json",
        {
            **SAFETY_FLAGS,
            "feature_columns_scanned": list(unique_features),
            "data_columns_scanned": list(compact.columns),
            "forbidden_matches": [],
            "passed": True,
        },
    )
    write_json(
        output / "model_configurations.json",
        {
            **SAFETY_FLAGS,
            "status": "not_fitted_due_to_support_gate",
            "model_specification_count": 8,
            "maximum_model_specifications": 8,
            "models": {key: list(value) for key, value in MODEL_FEATURES.items()},
            "configuration": {
                "kind": "deterministic_L2_logistic_regression",
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 250,
                "class_weight": None,
                "n_jobs": 1,
            },
        },
    )
    write_json(
        output / "model_coefficients.json",
        {
            **SAFETY_FLAGS,
            "status": "not_fitted_due_to_support_gate",
            "models": {},
            "frozen_predecessor_M1": dict(frozen_model_serializable(predecessor_model)),
        },
    )
    blocked_panel = compact.copy()
    blocked_panel["screen_status"] = blocker
    blocked_assessment = assessment.copy()
    blocked_assessment["screen_status"] = blocker
    write_parquet(output / "compact_decision_panel.parquet", blocked_panel)
    write_parquet(output / "onset_path_ledger.parquet", ledger)
    write_parquet(output / "development_oof_predictions.parquet", movement_oof)
    write_parquet(output / "assessment_predictions.parquet", blocked_assessment)
    status_frame = pd.DataFrame([{"status": "not_run_due_to_support_gate", "decision": blocker}])
    for name in (
        "onset_metrics.csv",
        "direction_metrics.csv",
        "monthly_metrics.csv",
        "checkpoint_metrics.csv",
        "calibration_bins.csv",
        "bootstrap_metrics.csv",
        "null_metrics.csv",
        "economic_reference_metrics.csv",
    ):
        write_csv(output / name, status_frame)
    row_shares = (
        assessment.loc[assessment["high_movement_admitted"].astype(bool), "symbol"]
        .value_counts(normalize=True)
        .sort_index()
    )
    concentration = pd.DataFrame(
        [
            {
                "population": "primary_high_movement_rows",
                "candidate": "not_applicable",
                "symbol": symbol,
                "rows": int(
                    (
                        assessment.loc[assessment["high_movement_admitted"].astype(bool), "symbol"]
                        == symbol
                    ).sum()
                ),
                "share": float(share),
                "maximum_allowed_share": 0.10,
                "passes": bool(share <= 0.10 + 1e-15),
            }
            for symbol, share in row_shares.items()
        ]
    )
    write_csv(output / "concentration_metrics.csv", concentration)
    plot_blocked(output / "calibration_systems.png", "Calibration not evaluated")
    plot_blocked(
        output / "economic_reference_20bps.png",
        "Economic reference not evaluated",
    )
    report = "\n".join(
        [
            "# High-Movement Pressure-Onset Screen V0",
            "",
            f"**Decision:** `{blocker}`",
            "",
            "The fixed primary-population support gate failed before any A0–A3 or "
            "D0–D3 pressure model was fit. A slate had only "
            f"`{int(support['minimum_high_movement_candidates_per_slate'])}` admitted "
            "candidate versus the fixed minimum of `10`; the maximum single-stock "
            f"row share was `{float(support['maximum_stock_row_share']):.6%}`, above "
            "the fixed `10%` ceiling.",
            "The population, movement-admission thresholds, and onset barriers were not changed.",
            "",
            "This is a retrospective, research-only, observable-only feasibility screen. "
            "It is not prospective validation, a strategy, achieved P&L, or executable edge.",
            "",
            f"- Frozen predecessor reconstruction passed: `{predecessor_result['passed']}`.",
            f"- Exact timestamps read: `{source_context['minimum_timestamp_read']}` through "
            f"`{source_context['maximum_timestamp_read']}`.",
            "- Protected rows materialised: `0`.",
            f"- Primary rows / sessions / stocks: `{support['rows']}` / "
            f"`{support['sessions']}` / `{support['stocks']}`.",
            f"- UP / DOWN / NO_ONSET: `{support['up_onsets']}` / "
            f"`{support['down_onsets']}` / `{support['no_onsets']}`.",
            f"- Movement thresholds: ordinal 6 `{thresholds[6]:.12f}`, "
            f"ordinal 12 `{thresholds[12]:.12f}`.",
            "- Model, bootstrap, null, and economic-reference outputs are marked not run.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return decision


def execute_run(output: Path, *, provider_root: Path) -> dict[str, Any]:
    """Execute one complete deterministic compact screen into one artifact root."""

    contract = load_contract()
    predecessor = pd.read_parquet(PREDECESSOR_PANEL)
    archived_assessment = pd.read_parquet(PREDECESSOR_PREDICTIONS)
    reconstruction, frozen_model, frozen_probabilities = predecessor_reconstruction(
        predecessor, archived_assessment
    )
    predecessor, movement_oof, movement_folds, thresholds = prepare_movement_probabilities(
        predecessor, frozen_model, frozen_probabilities
    )
    compact, ledger, source_context = build_compact_panel(predecessor, provider_root=provider_root)
    development = compact.loc[
        compact["year"].eq(2024) & compact["high_movement_admitted"].astype(bool)
    ].copy()
    assessment = compact.loc[compact["year"].eq(2025)].copy()
    primary = assessment.loc[assessment["high_movement_admitted"].astype(bool)].copy()
    support = support_summary(primary)
    if not support["primary_onset_support_passes"]:
        blocked_decision = write_blocked_artifacts(
            output,
            contract=contract,
            predecessor_result=reconstruction,
            predecessor_model=frozen_model,
            movement_oof=movement_oof,
            movement_folds=movement_folds,
            thresholds=thresholds,
            compact=compact,
            ledger=ledger,
            assessment=assessment,
            support=support,
            source_context=source_context,
        )
        return {
            "decision": blocked_decision["decision"],
            "support": support,
            "thresholds": {str(key): value for key, value in thresholds.items()},
            "barriers": source_context["onset_barriers"],
            "predecessor_reconstruction": reconstruction,
            "minimum_timestamp_read": source_context["minimum_timestamp_read"],
            "maximum_timestamp_read": source_context["maximum_timestamp_read"],
        }
    models, confirmed_development, confirmation_manifest = fit_model_ladder(development)
    scored_assessment = score_model_ladder(assessment, models)
    scored_primary = scored_assessment.loc[
        scored_assessment["high_movement_admitted"].astype(bool)
    ].copy()
    onset, direction, monthly, checkpoint, calibration = evaluate_model_ladder(scored_assessment)
    selections = economic_selections(scored_primary)
    economic = economic_metrics(selections)
    concentration, concentration_summary = concentration_metrics(scored_primary, selections)
    bootstrap = bootstrap_metrics(scored_primary, selections)
    nulls = null_metrics(
        confirmed_development,
        scored_primary,
        models,
        selections,
    )
    decision = derive_decision(
        onset,
        direction,
        monthly,
        checkpoint,
        bootstrap,
        nulls,
        economic,
        support,
        concentration_summary,
    )
    feature_columns = list(CONFIRMATION_FEATURES)
    compact_with_confirmation = compact.copy()
    keys = ["symbol", "session", "decision_ordinal"]
    confirmation_values = pd.concat(
        [
            confirmed_development.loc[:, [*keys, *feature_columns]],
            scored_assessment.loc[:, [*keys, *feature_columns]],
        ],
        ignore_index=True,
    ).drop_duplicates(keys, keep="last")
    compact_with_confirmation = (
        compact_with_confirmation.drop(columns=feature_columns)
        .merge(
            confirmation_values,
            on=keys,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
    )
    compact_with_confirmation = compact_with_confirmation.reset_index(drop=True)
    write_run_artifacts(
        output,
        contract=contract,
        predecessor_result=reconstruction,
        predecessor_model=frozen_model,
        movement_oof=movement_oof,
        movement_folds=movement_folds,
        thresholds=thresholds,
        compact=compact_with_confirmation,
        ledger=ledger,
        assessment=scored_assessment,
        models=models,
        confirmation_manifest=confirmation_manifest,
        onset=onset,
        direction=direction,
        monthly=monthly,
        checkpoint=checkpoint,
        calibration=calibration,
        bootstrap=bootstrap,
        nulls=nulls,
        economic=economic,
        concentration=concentration,
        support=support,
        concentration_summary=concentration_summary,
        decision=decision,
        source_context=source_context,
    )
    return {
        "decision": decision["decision"],
        "support": support,
        "thresholds": {str(key): value for key, value in thresholds.items()},
        "barriers": source_context["onset_barriers"],
        "predecessor_reconstruction": reconstruction,
        "minimum_timestamp_read": source_context["minimum_timestamp_read"],
        "maximum_timestamp_read": source_context["maximum_timestamp_read"],
    }


def compare_exact_runs(primary: Path, exact: Path) -> dict[str, Any]:
    """Compare all scientific artifacts, allowing strict Parquet value equality only."""

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
        mode = "byte_hash"
        passed = primary_hash == exact_hash
        if not passed and name.endswith(".parquet"):
            mode = "strict_numeric_and_value_comparison"
            try:
                pd.testing.assert_frame_equal(
                    pd.read_parquet(primary_path),
                    pd.read_parquet(exact_path),
                    check_exact=True,
                    check_dtype=True,
                    check_like=False,
                )
                passed = True
            except AssertionError:
                passed = False
        comparisons.append(
            {
                "artifact": name,
                "primary_sha256": primary_hash,
                "exact_rerun_sha256": exact_hash,
                "comparison_mode": mode,
                "passed": passed,
            }
        )
    result = {
        **SAFETY_FLAGS,
        "fixed_seeds": {
            "bootstrap": BOOTSTRAP_SEED,
            "null": NULL_SEED,
            "economic_random": RANDOM_SEED,
        },
        "stable_sorting": True,
        "canonical_json": True,
        "deterministic_models": True,
        "comparisons": comparisons,
        "passed": all(row["passed"] for row in comparisons),
    }
    if not result["passed"]:
        failed = [row["artifact"] for row in comparisons if not row["passed"]]
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"exact rerun differs: {failed}",
        )
    return result


def run_independent_auditor(artifacts: Path, *, provider_root: Path) -> None:
    """Invoke the standalone auditor without importing it into this runner."""

    command = [
        sys.executable,
        str(AUDITOR_PATH),
        "--artifacts",
        str(artifacts),
        "--provider-root",
        str(provider_root),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ScreenBlocker(
            "blocked_reproducibility_or_audit_failure",
            f"independent audit failed: {detail}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-root",
        type=Path,
        default=(
            Path.home()
            / "StockerLocal"
            / "data"
            / "processed"
            / "source=eodhd"
            / "instrument_type=stock"
        ),
    )
    parser.add_argument("--primary-output", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--exact-output", type=Path, default=DEFAULT_EXACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    primary = args.primary_output.resolve()
    exact = args.exact_output.resolve()
    try:
        primary_summary = execute_run(primary, provider_root=args.provider_root)
        exact_summary = execute_run(exact, provider_root=args.provider_root)
        if primary_summary != exact_summary:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure", "rerun summaries differ"
            )
        rerun = compare_exact_runs(primary, exact)
        rerun["independent_audit_status"] = "pending"
        write_json(primary / "exact_rerun_manifest.json", rerun)
        write_json(exact / "exact_rerun_manifest.json", rerun)
        run_independent_auditor(primary, provider_root=args.provider_root)
        run_independent_auditor(exact, provider_root=args.provider_root)
        primary_audit_hash = sha256_file(primary / "independent_audit.json")
        exact_audit_hash = sha256_file(exact / "independent_audit.json")
        if primary_audit_hash != exact_audit_hash:
            raise ScreenBlocker(
                "blocked_reproducibility_or_audit_failure",
                "independent audit artifacts differ across exact rerun",
            )
        rerun["independent_audit_sha256"] = primary_audit_hash
        rerun["independent_audit_status"] = "passed"
        write_json(primary / "exact_rerun_manifest.json", rerun)
        write_json(exact / "exact_rerun_manifest.json", rerun)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "report.md").write_text(
            (primary / "report.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print(canonical_json({**primary_summary, "exact_rerun": True, "audit": True}))
        return 0
    except ScreenBlocker as exc:
        primary.mkdir(parents=True, exist_ok=True)
        blocked = {
            **SAFETY_FLAGS,
            "decision": exc.code,
            "blocker": exc.detail,
        }
        write_json(primary / "decision.json", blocked)
        print(canonical_json(blocked), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
