#!/usr/bin/env python3
"""Run the bounded One-Minute Activity-Price Lead Screen V0."""

# ruff: noqa: E402 -- the repository-local research package path is resolved first.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stocker-one-minute-activity-matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PACKAGE_SRC = REPO_ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from stocker_research.one_minute_activity_price_lead_v0 import (
    FORBIDDEN_FEATURE_TOKENS,
    activity_absorption_interactions,
    activity_acceleration,
    activity_continuation_interactions,
    activity_lead_price_response,
    activity_peak_lead,
    activity_persistence,
    activity_range_response,
    activity_slope,
    bar_sign_weighted_activity_proxy,
    classify_onset,
    decide_activity_screen,
    development_onset_barriers,
    directional_efficiency,
    forbidden_feature_names,
    progress_per_activity,
)

CONTRACT_PATH = EXPERIMENT_DIR / "contract.json"
DEFAULT_PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_EXACT = EXPERIMENT_DIR / "artifacts" / "exact_rerun"
DEFAULT_REPORT = EXPERIMENT_DIR / "reports" / "report.md"
AUDITOR_PATH = EXPERIMENT_DIR / "audit_screen_v0.py"
PREDECESSOR_DIR = (
    REPO_ROOT
    / "research"
    / "observable-pressure-onset"
    / "20260720-high-movement-pressure-onset-screen-v0-1"
)
PREDECESSOR_PRIMARY = PREDECESSOR_DIR / "artifacts" / "primary"
PREDECESSOR_PANEL = PREDECESSOR_PRIMARY / "compact_decision_panel.parquet"
PREDECESSOR_SOURCE_MANIFEST = PREDECESSOR_PRIMARY / "source_manifest.json"
PREDECESSOR_DECISION = PREDECESSOR_PRIMARY / "decision.json"
PREDECESSOR_THRESHOLDS = PREDECESSOR_PRIMARY / "movement_admission_thresholds.json"
PREDECESSOR_BOUNDARY = PREDECESSOR_PRIMARY / "protected_boundary_audit.json"

DEVELOPMENT_START = pd.Timestamp("2024-01-01T00:00:00Z")
ASSESSMENT_START = pd.Timestamp("2025-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2025-08-23T00:00:00Z")
ASSESSMENT_END = pd.Timestamp("2025-08-22T23:59:59.999999Z")
HISTORY_BLOCKER = "blocked_one_minute_history_unavailable"
MAX_COMPACT_ROWS = 20_000
EXPECTED_ASSESSMENT_ROWS = 1_560
EXPECTED_ASSESSMENT_SESSIONS = 153
EXPECTED_ASSESSMENT_STOCKS = 20
EXPECTED_THRESHOLDS = {6: 0.302886936850, 12: 0.300349339178}
BOOTSTRAP_DRAWS = 200
NULL_DRAWS = 50
BOOTSTRAP_SEED = 20260720
NULL_SEED = 20260721
ECONOMIC_RANDOM_SEED = 20260722
WINDOW_ORDINALS = frozenset((*range(20, 30), *range(50, 60)))
REQUIRED_ORDINALS = {
    6: frozenset((*range(20, 30), *range(31, 36), 45, 60)),
    12: frozenset((*range(50, 60), *range(61, 66), 75, 90)),
}
MODEL_FEATURES: dict[str, tuple[str, ...]] = {}

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
    "one_minute_sequence_test": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "loops_regimes_states_and_structural_paths_forbidden": True,
}

PRICE_FEATURES = (
    "one_minute_return_minus_1",
    "one_minute_return_minus_2",
    "one_minute_return_minus_3",
    "cumulative_return_2",
    "cumulative_return_3",
    "cumulative_return_5",
    "cumulative_return_10",
    "cohort_relative_return_2",
    "cohort_relative_return_3",
    "cohort_relative_return_5",
    "cohort_relative_return_10",
    "realised_volatility_3",
    "realised_volatility_5",
    "realised_volatility_10",
    "mean_true_range_3",
    "mean_true_range_5",
    "range_acceleration",
    "signed_efficiency_3",
    "signed_efficiency_5",
    "absolute_efficiency_3",
    "absolute_efficiency_5",
    "mean_close_location_3",
    "upper_minus_lower_wick_imbalance_3",
    "new_one_minute_high_count_5",
    "new_one_minute_low_count_5",
    "maximum_retracement_from_favourable_ten_minute_extreme",
)

ACTIVITY_FEATURES = (
    "relative_activity_minus_1",
    "mean_relative_activity_2",
    "mean_relative_activity_3",
    "mean_relative_activity_5",
    "maximum_relative_activity_5",
    "activity_acceleration",
    "activity_slope_5",
    "elevated_activity_count_5",
    "same_clock_p90_activity_count_5",
    "longest_consecutive_elevated_activity_run_5",
    "latest_minute_share_of_ten_minute_activity",
    "maximum_minute_share_of_ten_minute_activity",
    "activity_coefficient_of_variation_10",
    "bar_sign_weighted_activity_proxy_3",
    "bar_sign_weighted_activity_proxy_5",
    "maximum_relative_activity_minute_index",
    "maximum_absolute_return_minute_index",
    "price_peak_index_minus_activity_peak_index",
)

INTERACTION_FEATURES = (
    "activity_continuation_3",
    "activity_continuation_5",
    "activity_absorption_3",
    "activity_absorption_wick",
    "signed_progress_per_activity_3",
    "absolute_progress_per_activity_3",
    "activity_lead_price_response",
    "activity_range_response",
)

MODEL_FEATURES.update(
    {
        "A0": ("checkpoint_60m", "p_large_remaining_move"),
        "A1": ("checkpoint_60m", "p_large_remaining_move", *PRICE_FEATURES),
        "A2": (
            "checkpoint_60m",
            "p_large_remaining_move",
            *PRICE_FEATURES,
            *ACTIVITY_FEATURES,
        ),
        "A3": (
            "checkpoint_60m",
            "p_large_remaining_move",
            *PRICE_FEATURES,
            *ACTIVITY_FEATURES,
            *INTERACTION_FEATURES,
        ),
        "D0": (
            "checkpoint_60m",
            "p_large_remaining_move",
            "open_to_decision_cohort_relative_return_bps",
        ),
        "D1": (
            "checkpoint_60m",
            "p_large_remaining_move",
            "open_to_decision_cohort_relative_return_bps",
            *PRICE_FEATURES,
        ),
        "D2": (
            "checkpoint_60m",
            "p_large_remaining_move",
            "open_to_decision_cohort_relative_return_bps",
            *PRICE_FEATURES,
            *ACTIVITY_FEATURES,
        ),
        "D3": (
            "checkpoint_60m",
            "p_large_remaining_move",
            "open_to_decision_cohort_relative_return_bps",
            *PRICE_FEATURES,
            *ACTIVITY_FEATURES,
            *INTERACTION_FEATURES,
        ),
    }
)

CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "onset_metrics.csv": (
        "population",
        "scope_type",
        "scope_value",
        "model",
        "brier_score",
        "log_loss",
        "auc",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "base_rate",
        "rows",
        "sessions",
        "stocks",
    ),
    "direction_metrics.csv": (
        "population",
        "scope_type",
        "scope_value",
        "model",
        "brier_score",
        "log_loss",
        "auc",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "base_rate",
        "rows",
        "sessions",
        "stocks",
    ),
    "checkpoint_metrics.csv": (
        "population",
        "scope_type",
        "scope_value",
        "stage",
        "model",
        "brier_score",
        "log_loss",
        "auc",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "base_rate",
        "rows",
        "sessions",
        "stocks",
    ),
    "monthly_metrics.csv": (
        "population",
        "scope_type",
        "scope_value",
        "stage",
        "model",
        "brier_score",
        "log_loss",
        "auc",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
        "base_rate",
        "rows",
        "sessions",
        "stocks",
    ),
    "calibration_bins.csv": (
        "stage",
        "model",
        "scope_type",
        "scope_value",
        "bin",
        "mean_probability",
        "outcome_rate",
        "rows",
    ),
    "feature_group_diagnostics.csv": (
        "stage",
        "model",
        "feature_group",
        "diagnostic",
        "scope",
        "value",
    ),
    "bootstrap_metrics.csv": (
        "record_type",
        "draw",
        "comparison",
        "metric",
        "estimate",
        "lower_90",
        "upper_90",
        "lower_95",
        "upper_95",
        "draws",
    ),
    "null_metrics.csv": (
        "record_type",
        "draw",
        "comparison",
        "metric",
        "real_value",
        "null_value",
        "null_q90",
        "real_percentile",
        "draws",
        "activity_bundle_sha256",
    ),
    "economic_reference_metrics.csv": (
        "candidate",
        "scope_type",
        "scope_value",
        "horizon",
        "friction_bps",
        "mean_signed_return_bps",
        "mean_cohort_relative_signed_return_bps",
        "positive_selection_pct",
        "rows",
    ),
    "concentration_metrics.csv": (
        "scope",
        "candidate",
        "symbol",
        "row_share",
        "selection_share",
        "passes",
    ),
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 of UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical, finite JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministically ordered CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministic compact Parquet artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path.name}")
    return payload


def verify_contract() -> dict[str, Any]:
    """Fail closed if a required safety declaration differs."""

    contract = read_json(CONTRACT_PATH)
    for key, expected in SAFETY_FLAGS.items():
        if contract.get(key) != expected or contract.get("safety", {}).get(key) != expected:
            raise RuntimeError(f"contract safety flag differs: {key}")
    return contract


def canonical_frame_hash(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    """Hash selected sorted values without depending on Parquet metadata."""

    selected = frame.loc[:, list(columns)].copy()
    for column in selected.columns:
        if isinstance(selected[column].dtype, pd.DatetimeTZDtype):
            selected[column] = selected[column].astype(str)
    return sha256_text(selected.to_csv(index=False, lineterminator="\n", float_format="%.17g"))


def reconstruct_frozen_population() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse, never recompute, the frozen V0.1 admission population."""

    panel = pd.read_parquet(PREDECESSOR_PANEL)
    required = {
        "symbol",
        "session",
        "year",
        "year_month",
        "decision_ordinal",
        "decision_time_america_new_york",
        "feature_available_timestamp_utc",
        "p_large_remaining_move",
        "open_to_decision_cohort_relative_return_bps",
        "movement_admission_threshold",
        "high_movement_admitted",
        "parent_slate_id",
        "parent_slate_eligible",
        "parent_valid_stock_count",
        "admitted_stock_count",
        "support_status",
        "primary_eligible",
        "row_weight",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise RuntimeError(f"frozen population columns missing: {missing}")
    if len(panel) > MAX_COMPACT_ROWS:
        raise RuntimeError("blocked_quick_activity_screen_resource_limit")
    timestamps = pd.to_datetime(panel["feature_available_timestamp_utc"], utc=True)
    if timestamps.gt(ASSESSMENT_END).any():
        raise RuntimeError("blocked_protected_boundary_failure")

    thresholds = read_json(PREDECESSOR_THRESHOLDS)["thresholds"]
    threshold_errors = {
        str(ordinal): abs(round(float(thresholds[str(ordinal)]), 12) - expected)
        for ordinal, expected in EXPECTED_THRESHOLDS.items()
    }
    if max(threshold_errors.values()) != 0.0:
        raise RuntimeError("blocked_frozen_high_movement_population_mismatch")

    source_manifest = read_json(PREDECESSOR_SOURCE_MANIFEST)
    qa_by_symbol = {
        str(record["symbol"]): (
            str(record["vendor_qa_status"]),
            bool(record["corporate_action_check_passed"]),
        )
        for record in source_manifest["sources"]
    }
    admitted = panel.loc[panel["high_movement_admitted"].astype(bool)].copy()
    admitted["decision_timestamp_utc"] = pd.to_datetime(
        admitted["feature_available_timestamp_utc"], utc=True
    )
    admitted["qa_status"] = admitted["symbol"].map(
        {symbol: status for symbol, (status, _) in qa_by_symbol.items()}
    )
    admitted["corporate_action_check_passed"] = admitted["symbol"].map(
        {symbol: passed for symbol, (_, passed) in qa_by_symbol.items()}
    )
    admitted["availability_gate_passed"] = False
    admitted["one_minute_source_status"] = "missing_source_file"
    keep = [
        "symbol",
        "session",
        "year",
        "year_month",
        "decision_ordinal",
        "decision_time_america_new_york",
        "decision_timestamp_utc",
        "p_large_remaining_move",
        "open_to_decision_cohort_relative_return_bps",
        "movement_admission_threshold",
        "high_movement_admitted",
        "parent_slate_id",
        "parent_slate_eligible",
        "parent_valid_stock_count",
        "admitted_stock_count",
        "support_status",
        "primary_eligible",
        "row_weight",
        "qa_status",
        "corporate_action_check_passed",
        "availability_gate_passed",
        "one_minute_source_status",
    ]
    compact = (
        admitted.loc[:, keep]
        .sort_values(["session", "decision_ordinal", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    development = compact.loc[compact["year"].eq(2024)]
    assessment = compact.loc[compact["year"].eq(2025)]
    counts = (
        len(assessment),
        int(assessment["session"].nunique()),
        int(assessment["symbol"].nunique()),
    )
    if counts != (
        EXPECTED_ASSESSMENT_ROWS,
        EXPECTED_ASSESSMENT_SESSIONS,
        EXPECTED_ASSESSMENT_STOCKS,
    ):
        raise RuntimeError("blocked_frozen_high_movement_population_mismatch")
    identity_columns = [
        "symbol",
        "session",
        "decision_ordinal",
        "decision_timestamp_utc",
        "p_large_remaining_move",
        "high_movement_admitted",
        "parent_slate_id",
        "movement_admission_threshold",
        "qa_status",
    ]
    reconstruction = {
        **SAFETY_FLAGS,
        "passed": True,
        "source_experiment": str(PREDECESSOR_DIR.relative_to(REPO_ROOT)),
        "source_commit": "cda387c6e05abfd5f9c5cc7cd22ad78224185e03",
        "source_panel_sha256": sha256_file(PREDECESSOR_PANEL),
        "frozen_panel_rows_read": len(panel),
        "frozen_source_minimum_timestamp_read": source_manifest["minimum_timestamp_read"],
        "frozen_source_maximum_timestamp_read": source_manifest["maximum_timestamp_read"],
        "frozen_admitted_minimum_decision_timestamp": str(compact["decision_timestamp_utc"].min()),
        "frozen_admitted_maximum_decision_timestamp": str(compact["decision_timestamp_utc"].max()),
        "predecessor_source_gap_ledger_rejected_session_records": source_manifest[
            "source_gap_ledger_rejected_session_records"
        ],
        "predecessor_corporate_action_checks_passed": all(
            passed for _, passed in qa_by_symbol.values()
        ),
        "predecessor_qa_status_by_symbol": {
            symbol: status for symbol, (status, _) in sorted(qa_by_symbol.items())
        },
        "identity_columns": identity_columns,
        "identity_sha256": canonical_frame_hash(compact, identity_columns),
        "development_admitted_rows": len(development),
        "assessment_admitted_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_checkpoint_rows": {
            str(key): int(value)
            for key, value in assessment.groupby("decision_ordinal", sort=True).size().items()
        },
        "thresholds_from_frozen_artifact": {
            str(key): float(value) for key, value in thresholds.items()
        },
        "required_thresholds_12dp": {str(key): value for key, value in EXPECTED_THRESHOLDS.items()},
        "threshold_absolute_errors_after_12dp_freeze": threshold_errors,
        "admission_rule_recomputed": False,
        "frozen_admission_flag_reused": True,
    }
    return compact, reconstruction


def provider_path(provider_root: Path, symbol: str) -> Path:
    """Return the expected local one-minute EODHD file."""

    return provider_root / f"symbol={symbol}" / "timeframe=1m" / "data.parquet"


def compact_ordinal_ranges(values: Iterable[int]) -> str:
    """Render sorted integer ordinals as stable inclusive ranges."""

    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def read_safe_one_minute_timestamps(path: Path) -> pd.Series:
    """Materialise only the unprotected timestamp column from a local source."""

    frame = pd.read_parquet(
        path,
        columns=["timestamp"],
        filters=[
            ("timestamp", ">=", DEVELOPMENT_START.to_pydatetime()),
            ("timestamp", "<", PROTECTED_START.to_pydatetime()),
        ],
    )
    timestamps = pd.Series(
        pd.to_datetime(frame["timestamp"], utc=True, errors="raise"),
        dtype="datetime64[ns, UTC]",
    ).sort_values(kind="mergesort", ignore_index=True)
    if timestamps.lt(DEVELOPMENT_START).any() or timestamps.ge(PROTECTED_START).any():
        raise RuntimeError("blocked_protected_boundary_failure")
    return timestamps


def safe_timestamp_hash(timestamps: pd.Series) -> str:
    """Hash only materialised safe timestamp labels, never protected rows."""

    canonical = "\n".join(value.isoformat() for value in timestamps.tolist())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def timestamp_convention_candidate_coverage(
    timestamps: pd.DatetimeIndex,
    *,
    market_open: pd.Timestamp,
    expected_minutes: int,
    convention: str,
) -> dict[str, Any]:
    """Map timestamp labels under one explicit convention candidate."""

    if convention not in {"bar_start", "bar_end"}:
        raise ValueError("unknown timestamp convention candidate")
    minute_ns = pd.Timedelta(minutes=1).value
    candidate_starts = timestamps.asi8 - (minute_ns if convention == "bar_end" else 0)
    deltas_ns = candidate_starts - market_open.value
    on_grid = (
        (deltas_ns >= 0)
        & (deltas_ns < expected_minutes * minute_ns)
        & ((deltas_ns % minute_ns) == 0)
    )
    grid_ordinals = (deltas_ns[on_grid] // minute_ns).astype(int).tolist()
    observed = sorted(set(grid_ordinals))
    missing = sorted(set(range(expected_minutes)).difference(observed))
    duplicates = len(grid_ordinals) - len(observed)
    return {
        "observed": observed,
        "missing": missing,
        "duplicate_count": duplicates,
        "off_grid_count": len(timestamps) - len(grid_ordinals),
        "complete": len(observed) == expected_minutes and duplicates == 0,
    }


def build_availability_audit(provider_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inspect local safe timestamps and report exact session/minute coverage."""

    calendar = mcal.get_calendar("XNYS")
    schedule = calendar.schedule(start_date="2024-01-01", end_date="2025-08-22")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    all_safe_timestamps: list[pd.Timestamp] = []
    for symbol in SYMBOLS:
        path = provider_path(provider_root, symbol)
        exists = path.is_file()
        logical_path = (
            f"source=eodhd/instrument_type=stock/symbol={symbol}/timeframe=1m/data.parquet"
        )
        read_error_code: str | None = None
        if exists:
            try:
                timestamps = read_safe_one_minute_timestamps(path)
            except RuntimeError:
                raise
            except Exception as exc:  # noqa: BLE001 - fail closed into the availability ledger.
                timestamps = pd.Series([], dtype="datetime64[ns, UTC]")
                read_error_code = type(exc).__name__
        else:
            timestamps = pd.Series([], dtype="datetime64[ns, UTC]")
        all_safe_timestamps.extend(timestamps.tolist())
        local_dates = timestamps.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        timestamps_by_session = {
            str(session): pd.DatetimeIndex(values)
            for session, values in timestamps.groupby(local_dates, sort=False)
        }
        matched_safe_rows = 0
        complete_sessions = 0
        bar_start_complete_sessions = 0
        bar_end_complete_sessions = 0
        sources.append(
            {
                "symbol": symbol,
                "logical_path": logical_path,
                "source_file_present": exists,
                "source_status": (
                    "unreadable_source"
                    if read_error_code
                    else "present_coverage_scanned"
                    if exists
                    else "missing_source_file"
                ),
                "source_read_error_code": read_error_code,
                "bounded_safe_rows_materialised": len(timestamps),
                "protected_rows_materialised": 0,
                "minimum_safe_timestamp_read": (
                    timestamps.iloc[0].isoformat() if not timestamps.empty else None
                ),
                "maximum_safe_timestamp_read": (
                    timestamps.iloc[-1].isoformat() if not timestamps.empty else None
                ),
                "bounded_safe_timestamp_sha256": safe_timestamp_hash(timestamps),
                "vendor_qa_logical_path": f"external_vendor_qa/{symbol}_1m_eodhd_qa.json",
                "vendor_qa_status": "not_opened_availability_gate_only",
            }
        )
        for session, schedule_row in schedule.iterrows():
            market_open = pd.Timestamp(schedule_row["market_open"])
            market_close = pd.Timestamp(schedule_row["market_close"])
            expected_minutes = int((market_close - market_open).total_seconds() // 60)
            session_text = pd.Timestamp(session).strftime("%Y-%m-%d")
            session_timestamps = timestamps_by_session.get(
                session_text, pd.DatetimeIndex([], tz="UTC")
            )
            bar_start = timestamp_convention_candidate_coverage(
                session_timestamps,
                market_open=market_open,
                expected_minutes=expected_minutes,
                convention="bar_start",
            )
            bar_end = timestamp_convention_candidate_coverage(
                session_timestamps,
                market_open=market_open,
                expected_minutes=expected_minutes,
                convention="bar_end",
            )
            matched_safe_rows += len(session_timestamps)
            if not exists:
                source_status = "missing_source_file"
                qa_status = "source_file_unavailable"
            elif read_error_code:
                source_status = "unreadable_source"
                qa_status = "source_unreadable"
            elif bar_start["complete"] and bar_end["complete"]:
                source_status = "complete_under_both_candidates"
                qa_status = "coverage_complete_timestamp_semantics_pending"
                complete_sessions += 1
                bar_start_complete_sessions += 1
                bar_end_complete_sessions += 1
            elif bar_start["complete"]:
                source_status = "complete_under_bar_start_candidate"
                qa_status = "coverage_complete_timestamp_semantics_pending"
                complete_sessions += 1
                bar_start_complete_sessions += 1
            elif bar_end["complete"]:
                source_status = "complete_under_bar_end_candidate"
                qa_status = "coverage_complete_timestamp_semantics_pending"
                complete_sessions += 1
                bar_end_complete_sessions += 1
            elif bar_start["observed"] or bar_end["observed"]:
                source_status = "partial"
                qa_status = "coverage_incomplete_timestamp_semantics_pending"
            else:
                source_status = "missing_session"
                qa_status = "session_unavailable"
            rows.append(
                {
                    "symbol": symbol,
                    "year_month": session_text[:7],
                    "population_period": (
                        "development" if session_text < "2025-01-01" else "assessment"
                    ),
                    "session": session_text,
                    "session_open_utc": market_open.isoformat(),
                    "session_close_utc": market_close.isoformat(),
                    "session_open_america_new_york": market_open.tz_convert(
                        "America/New_York"
                    ).isoformat(),
                    "session_close_america_new_york": market_close.tz_convert(
                        "America/New_York"
                    ).isoformat(),
                    "expected_minute_count": expected_minutes,
                    "expected_minute_ordinal_first": 0,
                    "expected_minute_ordinal_last": expected_minutes - 1,
                    "ordinal_basis": "dual_convention_candidates_pending_empirical_proof",
                    "bar_start_candidate_observed_minute_count": len(bar_start["observed"]),
                    "bar_start_candidate_observed_minute_ordinals": compact_ordinal_ranges(
                        bar_start["observed"]
                    ),
                    "bar_start_candidate_missing_minute_count": len(bar_start["missing"]),
                    "bar_start_candidate_missing_minute_ordinals": compact_ordinal_ranges(
                        bar_start["missing"]
                    ),
                    "bar_start_candidate_duplicate_minute_count": bar_start["duplicate_count"],
                    "bar_start_candidate_off_grid_minute_count": bar_start["off_grid_count"],
                    "bar_end_candidate_observed_minute_count": len(bar_end["observed"]),
                    "bar_end_candidate_observed_minute_ordinals": compact_ordinal_ranges(
                        bar_end["observed"]
                    ),
                    "bar_end_candidate_missing_minute_count": len(bar_end["missing"]),
                    "bar_end_candidate_missing_minute_ordinals": compact_ordinal_ranges(
                        bar_end["missing"]
                    ),
                    "bar_end_candidate_duplicate_minute_count": bar_end["duplicate_count"],
                    "bar_end_candidate_off_grid_minute_count": bar_end["off_grid_count"],
                    "source_status": source_status,
                    "source_file_identity": logical_path,
                    "qa_status": qa_status,
                }
            )
        sources[-1]["complete_regular_sessions"] = complete_sessions
        sources[-1]["bar_start_candidate_complete_regular_sessions"] = bar_start_complete_sessions
        sources[-1]["bar_end_candidate_complete_regular_sessions"] = bar_end_complete_sessions
        sources[-1]["unmatched_safe_timestamp_count"] = len(timestamps) - matched_safe_rows
    audit = (
        pd.DataFrame(rows)
        .sort_values(["symbol", "session"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(audit) > MAX_COMPACT_ROWS:
        raise RuntimeError("blocked_quick_activity_screen_resource_limit")
    complete_symbol_sessions = int(audit["source_status"].str.startswith("complete_under_").sum())
    bar_start_complete = int(
        audit["source_status"]
        .isin(
            {
                "complete_under_bar_start_candidate",
                "complete_under_both_candidates",
            }
        )
        .sum()
    )
    bar_end_complete = int(
        audit["source_status"]
        .isin(
            {
                "complete_under_bar_end_candidate",
                "complete_under_both_candidates",
            }
        )
        .sum()
    )
    minimum_timestamp = min(all_safe_timestamps) if all_safe_timestamps else None
    maximum_timestamp = max(all_safe_timestamps) if all_safe_timestamps else None
    manifest = {
        **SAFETY_FLAGS,
        "provider": "EODHD",
        "timeframe": "1m",
        "provider_activity_label": "historical_activity_proxy",
        "source_paths_are_logical_not_local_absolute": True,
        "external_data_downloaded": False,
        "external_api_called": False,
        "credentials_read": False,
        "symbols": list(SYMBOLS),
        "calendar": "XNYS",
        "calendar_sessions": int(audit["session"].nunique()),
        "availability_rows": len(audit),
        "complete_symbol_sessions": complete_symbol_sessions,
        "bar_start_candidate_complete_symbol_sessions": bar_start_complete,
        "bar_end_candidate_complete_symbol_sessions": bar_end_complete,
        "missing_symbol_sessions": int(len(audit) - complete_symbol_sessions),
        "partial_symbol_sessions": int(audit["source_status"].eq("partial").sum()),
        "sources_present": sum(bool(source["source_file_present"]) for source in sources),
        "one_minute_rows_materialised": len(all_safe_timestamps),
        "minimum_one_minute_timestamp_read": (
            minimum_timestamp.isoformat() if minimum_timestamp is not None else None
        ),
        "maximum_one_minute_timestamp_read": (
            maximum_timestamp.isoformat() if maximum_timestamp is not None else None
        ),
        "protected_rows_materialised": 0,
        "protected_filter": {
            "minimum_inclusive": DEVELOPMENT_START.isoformat(),
            "maximum_exclusive": PROTECTED_START.isoformat(),
            "columns_materialised": ["timestamp"],
        },
        "timestamp_convention_candidates_checked": ["bar_start", "bar_end"],
        "availability_gate_passed": max(bar_start_complete, bar_end_complete) == len(audit),
        "availability_gate_candidate_conventions": [
            convention
            for convention, count in (
                ("bar_start", bar_start_complete),
                ("bar_end", bar_end_complete),
            )
            if count == len(audit)
        ],
        "sources": sources,
    }
    return audit, manifest


def five_minute_provider_path(provider_root: Path, symbol: str) -> Path:
    """Return the frozen predecessor's local five-minute source path."""

    return provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"


def read_safe_bars(path: Path) -> pd.DataFrame:
    """Read only safe OHLCV rows using a Parquet predicate."""

    frame = pd.read_parquet(
        path,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        filters=[
            ("timestamp", ">=", DEVELOPMENT_START.to_pydatetime()),
            ("timestamp", "<", PROTECTED_START.to_pydatetime()),
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if (
        frame["timestamp"].lt(DEVELOPMENT_START).any()
        or frame["timestamp"].ge(PROTECTED_START).any()
    ):
        raise RuntimeError("blocked_protected_boundary_failure")
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _regular_session_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    local = output["timestamp"].dt.tz_convert("America/New_York")
    output["session"] = local.dt.strftime("%Y-%m-%d")
    output["minute_of_session_ordinal"] = (local.dt.hour * 60 + local.dt.minute - 570).astype(int)
    return output.loc[output["minute_of_session_ordinal"].between(0, 389)].copy()


def prove_local_timestamp_semantics(provider_root: Path) -> dict[str, Any]:
    """Empirically distinguish start labels by alignment with frozen local 5m bars."""

    records: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        one_path = provider_path(provider_root, symbol)
        five_path = five_minute_provider_path(provider_root, symbol)
        if not one_path.is_file() or not five_path.is_file():
            raise RuntimeError("blocked_one_minute_timestamp_semantics_unproven")
        one = _regular_session_coordinates(read_safe_bars(one_path))
        five = _regular_session_coordinates(read_safe_bars(five_path))
        five = five.loc[five["minute_of_session_ordinal"].between(0, 385)].copy()
        for convention in ("bar_start", "bar_end"):
            shifted = one["timestamp"] - (
                pd.Timedelta(minutes=1) if convention == "bar_end" else pd.Timedelta(0)
            )
            working = one.assign(_bucket=shifted.dt.floor("5min"))
            aggregated = (
                working.groupby("_bucket", sort=True)
                .agg(
                    open_1m=("open", "first"),
                    high_1m=("high", "max"),
                    low_1m=("low", "min"),
                    close_1m=("close", "last"),
                    minute_count=("close", "size"),
                )
                .reset_index(names="timestamp")
            )
            aggregated = aggregated.loc[aggregated["minute_count"].eq(5)]
            aligned = aggregated.merge(
                five.loc[:, ["timestamp", "open", "high", "low", "close"]],
                on="timestamp",
                how="inner",
                validate="one_to_one",
            ).dropna()
            if len(aligned) < 1_000:
                raise RuntimeError("blocked_one_minute_timestamp_semantics_unproven")
            row: dict[str, Any] = {
                "symbol": symbol,
                "candidate": convention,
                "aligned_complete_five_minute_bars": len(aligned),
                "represented_sessions": int(
                    aligned["timestamp"]
                    .dt.tz_convert("America/New_York")
                    .dt.strftime("%Y-%m-%d")
                    .nunique()
                ),
            }
            for column in ("open", "high", "low", "close"):
                difference = np.abs(
                    aligned[f"{column}_1m"].to_numpy(dtype=float)
                    - aligned[column].to_numpy(dtype=float)
                )
                denominator = np.maximum(np.abs(aligned[column].to_numpy(dtype=float)), 1e-12)
                row[f"median_relative_error_{column}"] = float(np.median(difference / denominator))
                row[f"within_one_mill_{column}_fraction"] = float(np.mean(difference <= 0.001))
            records.append(row)
    evidence = pd.DataFrame(records)
    start = evidence.loc[evidence["candidate"].eq("bar_start")].set_index("symbol")
    end = evidence.loc[evidence["candidate"].eq("bar_end")].set_index("symbol")
    ratios: dict[str, float] = {}
    for column in ("open", "close"):
        start_error = float(start[f"median_relative_error_{column}"].median())
        end_error = float(end[f"median_relative_error_{column}"].median())
        ratios[column] = end_error / max(start_error, 1e-15)
    passed = bool(
        len(start) == len(SYMBOLS)
        and int(start["represented_sessions"].min()) >= 100
        and all(
            float(start[f"median_relative_error_{column}"].max()) <= 1e-5
            for column in ("open", "close")
        )
        and min(ratios.values()) >= 50.0
        and all(
            float(start[f"within_one_mill_{column}_fraction"].median())
            > float(end[f"within_one_mill_{column}_fraction"].median())
            for column in ("open", "close")
        )
    )
    if not passed:
        raise RuntimeError("blocked_one_minute_timestamp_semantics_unproven")
    return {
        **SAFETY_FLAGS,
        "passed": True,
        "status": "proved_by_cross_timeframe_ohlc_alignment",
        "timestamp_convention": "bar_start",
        "bar_start_or_end_proved": True,
        "session_relative_ordinal_mapping_performed": True,
        "causal_window_rule": "bar_start_plus_one_minute_less_than_or_equal_to_decision",
        "proof_rule": {
            "minimum_aligned_bars_per_symbol_candidate": 1000,
            "minimum_represented_sessions_per_symbol": 100,
            "maximum_bar_start_open_close_median_relative_error": 1e-5,
            "minimum_bar_end_to_bar_start_open_close_error_ratio": 50.0,
        },
        "bar_end_to_bar_start_median_error_ratios": ratios,
        "evidence": evidence.sort_values(["symbol", "candidate"], kind="mergesort").to_dict(
            "records"
        ),
        "five_minute_anchor": "unchanged local EODHD inputs used by the frozen predecessor",
        "timestamp_guessing_used": False,
    }


def load_relevant_one_minute_bars(
    provider_root: Path, parent_sessions: set[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load only clock minutes needed for causal normalisation, features, and outcomes."""

    required_union = frozenset().union(*REQUIRED_ORDINALS.values())
    parts: list[pd.DataFrame] = []
    qa_records: list[dict[str, Any]] = []
    qa_root = provider_root.parents[2] / "reports" / "vendor_qa"
    for symbol in SYMBOLS:
        path = provider_path(provider_root, symbol)
        frame = _regular_session_coordinates(read_safe_bars(path))
        keep = frame["minute_of_session_ordinal"].isin(WINDOW_ORDINALS) | (
            frame["session"].isin(parent_sessions)
            & frame["minute_of_session_ordinal"].isin(required_union)
        )
        frame = frame.loc[keep].copy()
        frame["symbol"] = symbol
        frame["timestamp_utc"] = frame["timestamp"]
        frame["timestamp_america_new_york"] = frame["timestamp"].dt.tz_convert("America/New_York")
        frame["source_file_identity"] = (
            f"source=eodhd/instrument_type=stock/symbol={symbol}/timeframe=1m/data.parquet"
        )
        qa_path = qa_root / f"{symbol}_1m_eodhd_qa.json"
        qa = read_json(qa_path) if qa_path.is_file() else {"status": "unavailable"}
        qa_status = str(qa.get("status", "unavailable"))
        frame["qa_status"] = qa_status
        qa_records.append(
            {
                "symbol": symbol,
                "logical_path": f"external_vendor_qa/{symbol}_1m_eodhd_qa.json",
                "status": qa_status,
                "sha256": sha256_file(qa_path) if qa_path.is_file() else None,
                "validation_counts": qa.get("validation", {}).get("counts", {}),
                "issue_codes": qa.get("issue_codes", []),
            }
        )
        parts.append(frame)
    bars = pd.concat(parts, ignore_index=True).sort_values(
        ["symbol", "session", "minute_of_session_ordinal"], kind="mergesort"
    )
    duplicate = bars.duplicated(["symbol", "session", "minute_of_session_ordinal"])
    if duplicate.any():
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    numeric = bars.loc[:, ["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or bool((bars["volume"] < 0).any()):
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    return bars.reset_index(drop=True), qa_records


def causal_activity_normalisation(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit prior-session same-symbol/same-clock activity baselines and freeze them in 2025."""

    output = bars.copy()
    row_count = len(output)
    historical_median = np.full(row_count, np.nan, dtype=float)
    prior_observations = np.zeros(row_count, dtype=int)
    relative_activity = np.full(row_count, np.nan, dtype=float)
    log_relative_activity = np.full(row_count, np.nan, dtype=float)
    same_clock_p90 = np.full(row_count, np.nan, dtype=float)
    volumes = output["volume"].to_numpy(dtype=float)
    sessions = output["session"].astype(str).to_numpy()
    window_mask = output["minute_of_session_ordinal"].isin(WINDOW_ORDINALS)
    window = output.loc[window_mask].copy()
    for _, positions in window.groupby(
        ["symbol", "minute_of_session_ordinal"], sort=True
    ).groups.items():
        ordered_indices = np.asarray(
            sorted(positions, key=lambda index: sessions[int(index)]), dtype=int
        )
        development_indices = ordered_indices[sessions[ordered_indices] < "2025-01-01"]
        assessment_indices = ordered_indices[sessions[ordered_indices] >= "2025-01-01"]
        development_volumes = volumes[development_indices]
        valid_relative: list[float] = []
        for offset, row_index in enumerate(development_indices):
            prior_observations[row_index] = offset
            if offset < 20:
                continue
            median = float(np.median(development_volumes[:offset]))
            relative = float(development_volumes[offset] / median) if median > 0.0 else math.nan
            historical_median[row_index] = median
            relative_activity[row_index] = relative
            log_relative_activity[row_index] = math.log1p(relative)
            if len(valid_relative) >= 20:
                same_clock_p90[row_index] = float(np.quantile(valid_relative, 0.90))
            if math.isfinite(relative):
                valid_relative.append(relative)
        if len(development_volumes) < 20 or len(valid_relative) < 20:
            continue
        frozen_median = float(np.median(development_volumes))
        frozen_p90 = float(np.quantile(valid_relative, 0.90))
        assessment_relative = volumes[assessment_indices] / frozen_median
        historical_median[assessment_indices] = frozen_median
        prior_observations[assessment_indices] = len(development_volumes)
        relative_activity[assessment_indices] = assessment_relative
        log_relative_activity[assessment_indices] = np.log1p(assessment_relative)
        same_clock_p90[assessment_indices] = frozen_p90
    output["historical_median_volume"] = historical_median
    output["activity_normalisation_prior_observations"] = prior_observations
    output["relative_activity"] = relative_activity
    output["log_relative_activity"] = log_relative_activity
    output["same_clock_relative_activity_p90"] = same_clock_p90
    fitted = output.loc[window_mask & output["relative_activity"].notna()]
    manifest = {
        **SAFETY_FLAGS,
        "status": "fitted",
        "provider_volume_label": "historical_activity_proxy",
        "formula": "volume / trailing_same_stock_same_minute_median_volume",
        "minimum_prior_observations": 20,
        "log_score": "log1p(relative_activity)",
        "development_only": True,
        "development_rows_use_strictly_prior_sessions": True,
        "assessment_baselines_frozen_from_2024": True,
        "future_sessions_used": False,
        "assessment_outcomes_used": False,
        "normalisation_input_rows": int(window_mask.sum()),
        "normalisation_rows_fitted": len(fitted),
        "same_clock_p90_rule": "expanding_prior_2024_relative_activity_then_frozen_for_2025",
    }
    return output, manifest


def build_valid_parent_population(
    frozen_admitted: pd.DataFrame, bars: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve exact minute availability for every stock in each admitted parent slate."""

    predecessor = pd.read_parquet(PREDECESSOR_PANEL)
    slate_ids = set(frozen_admitted["parent_slate_id"].astype(str))
    parents = predecessor.loc[predecessor["parent_slate_id"].astype(str).isin(slate_ids)].copy()
    parents["decision_timestamp_utc"] = pd.to_datetime(
        parents["feature_available_timestamp_utc"], utc=True, errors="raise"
    )
    available = {
        (str(symbol), str(session)): frozenset(group["minute_of_session_ordinal"].astype(int))
        for (symbol, session), group in bars.groupby(["symbol", "session"], sort=True)
    }
    parents["required_one_minute_window_available"] = [
        REQUIRED_ORDINALS[int(checkpoint)].issubset(
            available.get((str(symbol), str(session)), frozenset())
        )
        for symbol, session, checkpoint in parents[
            ["symbol", "session", "decision_ordinal"]
        ].itertuples(index=False, name=None)
    ]
    valid_counts = parents.groupby("parent_slate_id", sort=True)[
        "required_one_minute_window_available"
    ].transform("sum")
    parents["one_minute_parent_valid_stock_count"] = valid_counts.astype(int)
    parents["one_minute_parent_slate_supported"] = valid_counts.ge(15)
    valid = parents.loc[
        parents["required_one_minute_window_available"]
        & parents["one_minute_parent_slate_supported"]
    ].copy()
    valid = valid.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    admitted_valid = valid.loc[valid["high_movement_admitted"].astype(bool)]
    assessment = admitted_valid.loc[admitted_valid["year"].eq(2025)]
    coverage = {
        **SAFETY_FLAGS,
        "parent_rows_checked": len(parents),
        "parent_rows_with_exact_required_minutes": int(
            parents["required_one_minute_window_available"].sum()
        ),
        "parent_slates_checked": int(parents["parent_slate_id"].nunique()),
        "parent_slates_with_at_least_15_valid_stocks": int(
            parents.loc[parents["one_minute_parent_slate_supported"], "parent_slate_id"].nunique()
        ),
        "minimum_valid_stocks_in_retained_parent_slate": int(
            valid["one_minute_parent_valid_stock_count"].min()
        ),
        "development_admitted_rows_with_exact_windows": int(admitted_valid["year"].eq(2024).sum()),
        "assessment_admitted_rows_with_exact_windows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "assessment_rows_excluded_for_exact_minute_unavailability": int(
            len(frozen_admitted.loc[frozen_admitted["year"].eq(2025)]) - len(assessment)
        ),
    }
    return valid, coverage


def _safe_ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return default
    return default if abs(denominator) < 1e-12 else numerator / denominator


def _one_row_sequence_features(window: pd.DataFrame) -> dict[str, float]:
    """Calculate the fixed price and raw-activity features for one causal window."""

    ordered = window.sort_index(kind="mergesort")
    if len(ordered) != 10:
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    open_values = ordered["open"].to_numpy(dtype=float)
    high_values = ordered["high"].to_numpy(dtype=float)
    low_values = ordered["low"].to_numpy(dtype=float)
    close_values = ordered["close"].to_numpy(dtype=float)
    activity = ordered["relative_activity"].to_numpy(dtype=float)
    log_activity = ordered["log_relative_activity"].to_numpy(dtype=float)
    p90 = ordered["same_clock_relative_activity_p90"].to_numpy(dtype=float)
    arrays = np.concatenate(
        [open_values, high_values, low_values, close_values, activity, log_activity, p90]
    )
    if not np.isfinite(arrays).all():
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    bar_returns = 10_000.0 * (close_values / open_values - 1.0)

    def cumulative(length: int) -> float:
        return float(10_000.0 * (close_values[-1] / open_values[-length] - 1.0))

    previous_close = np.concatenate(([open_values[0]], close_values[:-1]))
    true_range = (
        np.maximum.reduce(
            [
                high_values - low_values,
                np.abs(high_values - previous_close),
                np.abs(low_values - previous_close),
            ]
        )
        / np.maximum(previous_close, 1e-12)
        * 10_000.0
    )
    efficiencies_3 = directional_efficiency(bar_returns[-3:])
    efficiencies_5 = directional_efficiency(bar_returns[-5:])
    ranges = high_values - low_values
    close_locations = np.divide(
        close_values - low_values,
        ranges,
        out=np.full(10, 0.5, dtype=float),
        where=ranges > 1e-12,
    )
    upper_wicks = high_values - np.maximum(open_values, close_values)
    lower_wicks = np.minimum(open_values, close_values) - low_values
    wick_imbalance = np.divide(
        upper_wicks - lower_wicks,
        ranges,
        out=np.zeros(10, dtype=float),
        where=ranges > 1e-12,
    )
    new_high_count = sum(
        bool(high_values[index] > np.max(high_values[:index])) for index in range(5, 10)
    )
    new_low_count = sum(
        bool(low_values[index] < np.min(low_values[:index])) for index in range(5, 10)
    )
    ten_progress = cumulative(10)
    if ten_progress >= 0.0:
        retracement = 10_000.0 * max(0.0, np.max(high_values) - close_values[-1]) / open_values[0]
    else:
        retracement = 10_000.0 * max(0.0, close_values[-1] - np.min(low_values)) / open_values[0]
    persistence = activity_persistence(activity[-5:], same_clock_p90=p90[-5:])
    timing = activity_peak_lead(activity, bar_returns)
    total_activity = float(activity.sum())
    activity_mean = float(activity.mean())
    features: dict[str, float] = {
        "one_minute_return_minus_1": float(bar_returns[-1]),
        "one_minute_return_minus_2": float(bar_returns[-2]),
        "one_minute_return_minus_3": float(bar_returns[-3]),
        "cumulative_return_2": cumulative(2),
        "cumulative_return_3": cumulative(3),
        "cumulative_return_5": cumulative(5),
        "cumulative_return_10": ten_progress,
        "realised_volatility_3": float(np.std(bar_returns[-3:], ddof=0)),
        "realised_volatility_5": float(np.std(bar_returns[-5:], ddof=0)),
        "realised_volatility_10": float(np.std(bar_returns, ddof=0)),
        "mean_true_range_3": float(np.mean(true_range[-3:])),
        "mean_true_range_5": float(np.mean(true_range[-5:])),
        "range_acceleration": _safe_ratio(
            float(np.mean(true_range[-2:])), float(np.mean(true_range[-6:-2]))
        ),
        "signed_efficiency_3": float(efficiencies_3["signed_efficiency"]),
        "signed_efficiency_5": float(efficiencies_5["signed_efficiency"]),
        "absolute_efficiency_3": float(efficiencies_3["absolute_efficiency"]),
        "absolute_efficiency_5": float(efficiencies_5["absolute_efficiency"]),
        "mean_close_location_3": float(np.mean(close_locations[-3:])),
        "upper_minus_lower_wick_imbalance_3": float(np.mean(wick_imbalance[-3:])),
        "new_one_minute_high_count_5": float(new_high_count),
        "new_one_minute_low_count_5": float(new_low_count),
        "maximum_retracement_from_favourable_ten_minute_extreme": float(retracement),
        "relative_activity_minus_1": float(activity[-1]),
        "mean_relative_activity_2": float(np.mean(activity[-2:])),
        "mean_relative_activity_3": float(np.mean(activity[-3:])),
        "mean_relative_activity_5": float(np.mean(activity[-5:])),
        "maximum_relative_activity_5": float(np.max(activity[-5:])),
        "activity_acceleration": activity_acceleration(log_activity[-5:]),
        "activity_slope_5": activity_slope(log_activity[-5:]),
        "elevated_activity_count_5": float(persistence["above_one_count"]),
        "same_clock_p90_activity_count_5": float(persistence["above_same_clock_p90_count"]),
        "longest_consecutive_elevated_activity_run_5": float(persistence["longest_elevated_run"]),
        "latest_minute_share_of_ten_minute_activity": _safe_ratio(
            float(activity[-1]), total_activity
        ),
        "maximum_minute_share_of_ten_minute_activity": _safe_ratio(
            float(np.max(activity)), total_activity
        ),
        "activity_coefficient_of_variation_10": _safe_ratio(
            float(np.std(activity, ddof=0)), activity_mean
        ),
        "bar_sign_weighted_activity_proxy_3": bar_sign_weighted_activity_proxy(
            bar_returns[-3:], activity[-3:]
        ),
        "bar_sign_weighted_activity_proxy_5": bar_sign_weighted_activity_proxy(
            bar_returns[-5:], activity[-5:]
        ),
        "maximum_relative_activity_minute_index": float(timing["activity_peak_index"]),
        "maximum_absolute_return_minute_index": float(timing["price_peak_index"]),
        "price_peak_index_minus_activity_peak_index": float(
            timing["price_peak_index_minus_activity_peak_index"]
        ),
        "calculation_total_relative_activity_3": float(activity[-3:].sum()),
        "calculation_early_activity": float(np.mean(activity[-5:-2])),
        "calculation_cumulative_return_minus_5_through_minus_3": float(
            10_000.0 * (close_values[-3] / open_values[-5] - 1.0)
        ),
    }
    for length in (2, 3, 5, 10):
        features[f"calculation_raw_cumulative_return_{length}"] = features[
            f"cumulative_return_{length}"
        ]
    return features


def _bar_groups(bars: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    return {
        (str(symbol), str(session)): group.set_index("minute_of_session_ordinal", drop=False)
        for (symbol, session), group in bars.groupby(["symbol", "session"], sort=True)
    }


def build_feature_panel(
    valid_parents: pd.DataFrame, bars: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Materialise fixed causal features for valid parent rows and admitted sequence ledger."""

    groups = _bar_groups(bars)
    feature_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for row in valid_parents.itertuples(index=False):
        checkpoint = int(row.decision_ordinal)
        decision_start = 30 if checkpoint == 6 else 60
        ordinals = list(range(decision_start - 10, decision_start))
        window = groups[(str(row.symbol), str(row.session))].loc[ordinals].copy()
        decision = pd.Timestamp(row.decision_timestamp_utc)
        if not (
            window["timestamp_utc"].max() < decision
            and (window["timestamp_utc"] + pd.Timedelta(minutes=1)).max() <= decision
        ):
            raise RuntimeError("blocked_chronology_or_leakage_failure")
        calculated = _one_row_sequence_features(window)
        base = {
            "symbol": str(row.symbol),
            "session": str(row.session),
            "year": int(row.year),
            "year_month": str(row.year_month),
            "decision_ordinal": checkpoint,
            "decision_timestamp_utc": decision,
            "parent_slate_id": str(row.parent_slate_id),
            "one_minute_parent_valid_stock_count": int(row.one_minute_parent_valid_stock_count),
            "high_movement_admitted": bool(row.high_movement_admitted),
            "p_large_remaining_move": float(row.p_large_remaining_move),
            "movement_admission_threshold": float(row.movement_admission_threshold),
            "open_to_decision_cohort_relative_return_bps": float(
                row.open_to_decision_cohort_relative_return_bps
            ),
            "admitted_stock_count": int(row.admitted_stock_count),
            "checkpoint_60m": float(checkpoint == 12),
        }
        feature_rows.append({**base, **calculated})
        if bool(row.high_movement_admitted):
            bar_returns = 10_000.0 * (
                window["close"].to_numpy(dtype=float) / window["open"].to_numpy(dtype=float) - 1.0
            )
            for relative, (_, minute), one_return in zip(
                range(-10, 0), window.iterrows(), bar_returns, strict=True
            ):
                ledger_rows.append(
                    {
                        "symbol": str(row.symbol),
                        "session": str(row.session),
                        "decision_ordinal_5m": checkpoint,
                        "parent_slate_id": str(row.parent_slate_id),
                        "relative_minute": relative,
                        "minute_of_session_ordinal": int(minute["minute_of_session_ordinal"]),
                        "timestamp_utc": minute["timestamp_utc"],
                        "timestamp_america_new_york": minute["timestamp_america_new_york"],
                        "source_file_identity": minute["source_file_identity"],
                        "qa_status": minute["qa_status"],
                        "open": float(minute["open"]),
                        "high": float(minute["high"]),
                        "low": float(minute["low"]),
                        "close": float(minute["close"]),
                        "historical_activity_proxy": float(minute["volume"]),
                        "historical_median_volume": float(minute["historical_median_volume"]),
                        "activity_normalisation_prior_observations": int(
                            minute["activity_normalisation_prior_observations"]
                        ),
                        "relative_activity": float(minute["relative_activity"]),
                        "log_relative_activity": float(minute["log_relative_activity"]),
                        "same_clock_relative_activity_p90": float(
                            minute["same_clock_relative_activity_p90"]
                        ),
                        "one_minute_return_bps": float(one_return),
                    }
                )
    features = pd.DataFrame(feature_rows).sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    )
    for length in (2, 3, 5, 10):
        raw_column = f"calculation_raw_cumulative_return_{length}"
        relative_column = f"cohort_relative_return_{length}"
        features[relative_column] = np.nan
        for _, positions in features.groupby("parent_slate_id", sort=True).groups.items():
            group = features.loc[positions]
            values = group[raw_column].to_numpy(dtype=float)
            for offset, row_index in enumerate(group.index):
                features.loc[row_index, relative_column] = values[offset] - float(
                    np.median(np.delete(values, offset))
                )
    continuation_rows: list[dict[str, float]] = []
    for row in features.itertuples(index=False):
        continuation = activity_continuation_interactions(
            mean_relative_activity_3=float(row.mean_relative_activity_3),
            signed_efficiency_3=float(row.signed_efficiency_3),
            mean_relative_activity_5=float(row.mean_relative_activity_5),
            signed_efficiency_5=float(row.signed_efficiency_5),
        )
        absorption = activity_absorption_interactions(
            mean_relative_activity_3=float(row.mean_relative_activity_3),
            absolute_efficiency_3=float(row.absolute_efficiency_3),
            absolute_wick_imbalance_3=float(row.upper_minus_lower_wick_imbalance_3),
        )
        progress = progress_per_activity(
            float(row.cohort_relative_return_3),
            [float(row.calculation_total_relative_activity_3) / 3.0] * 3,
        )
        continuation_rows.append(
            {
                **continuation,
                **absorption,
                **progress,
                "activity_lead_price_response": activity_lead_price_response(
                    early_activity=float(row.calculation_early_activity),
                    cumulative_return_last_2=float(row.cumulative_return_2),
                    cumulative_return_minutes_minus_5_through_minus_3=float(
                        row.calculation_cumulative_return_minus_5_through_minus_3
                    ),
                ),
                "activity_range_response": activity_range_response(
                    activity_acceleration_value=float(row.activity_acceleration),
                    range_acceleration=float(row.range_acceleration),
                ),
            }
        )
    interactions = pd.DataFrame(continuation_rows, index=features.index)
    features = pd.concat([features, interactions], axis=1)
    development = features.loc[features["year"].eq(2024) & features["high_movement_admitted"]]
    winsor_bounds: dict[str, list[float]] = {}
    for column in (
        "signed_progress_per_activity_3",
        "absolute_progress_per_activity_3",
    ):
        lower, upper = development[column].quantile([0.01, 0.99]).to_numpy(dtype=float)
        winsor_bounds[column] = [float(lower), float(upper)]
        features[column] = features[column].clip(lower=lower, upper=upper)
    predictor_names = [*PRICE_FEATURES, *ACTIVITY_FEATURES, *INTERACTION_FEATURES]
    values = features.loc[:, predictor_names].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    ledger = pd.DataFrame(ledger_rows).sort_values(
        ["session", "decision_ordinal_5m", "symbol", "relative_minute"],
        kind="mergesort",
    )
    manifest = {
        **SAFETY_FLAGS,
        "status": "materialised",
        "predictor_window": "completed_minutes_minus_10_through_minus_1",
        "price_sequence_features": list(PRICE_FEATURES),
        "price_scalar_feature_count": len(PRICE_FEATURES),
        "price_preregistered_feature_family_count": 15,
        "raw_activity_sequence_features": list(ACTIVITY_FEATURES),
        "activity_price_response_interactions": list(INTERACTION_FEATURES),
        "provider_volume_label": "historical_activity_proxy",
        "signed_activity_proxy_label": "bar_sign_weighted_activity_proxy",
        "one_minute_return_definition": "10000 * (bar_close / bar_open - 1)",
        "return_and_range_units": "basis_points_except_dimensionless_ratios",
        "activity_lead_clip": [-9, 9],
        "progress_winsor_quantiles": [0.01, 0.99],
        "progress_winsor_bounds": winsor_bounds,
        "assessment_information_used_in_features": False,
        "future_information_used_in_features": False,
        "feature_rows_materialised": len(features),
        "admitted_sequence_ledger_rows": len(ledger),
    }
    return features.reset_index(drop=True), ledger.reset_index(drop=True), manifest


def build_outcomes(
    features: pd.DataFrame, bars: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, float]]:
    """Construct delayed residual onset paths and fixed terminal diagnostics."""

    groups = _bar_groups(bars)
    path_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    for row in features.itertuples(index=False):
        checkpoint = int(row.decision_ordinal)
        decision_completed = 29 if checkpoint == 6 else 59
        entry_ordinal = decision_completed + 2
        onset_ordinals = list(range(entry_ordinal, entry_ordinal + 5))
        terminal_15 = decision_completed + 16
        terminal_30 = decision_completed + 31
        symbol_bars = groups[(str(row.symbol), str(row.session))]
        entry_timestamp = pd.Timestamp(symbol_bars.loc[entry_ordinal, "timestamp_utc"])
        if entry_timestamp != pd.Timestamp(row.decision_timestamp_utc) + pd.Timedelta(minutes=1):
            raise RuntimeError("blocked_chronology_or_leakage_failure")
        entry_open = float(symbol_bars.loc[entry_ordinal, "open"])
        decision_id = f"{row.parent_slate_id}|{row.symbol}"
        for step, ordinal in enumerate(onset_ordinals, start=2):
            close = float(symbol_bars.loc[ordinal, "close"])
            path_rows.append(
                {
                    "decision_id": decision_id,
                    "parent_slate_id": str(row.parent_slate_id),
                    "symbol": str(row.symbol),
                    "session": str(row.session),
                    "year": int(row.year),
                    "year_month": str(row.year_month),
                    "decision_ordinal": checkpoint,
                    "high_movement_admitted": bool(row.high_movement_admitted),
                    "relative_minute": step,
                    "minute_of_session_ordinal": ordinal,
                    "entry_minute_ordinal": entry_ordinal,
                    "entry_timestamp_utc": entry_timestamp,
                    "entry_open": entry_open,
                    "path_close_timestamp_utc": pd.Timestamp(
                        symbol_bars.loc[ordinal, "timestamp_utc"]
                    )
                    + pd.Timedelta(minutes=1),
                    "path_close": close,
                    "cumulative_return_bps": 10_000.0 * (close / entry_open - 1.0),
                }
            )
        close_15 = float(symbol_bars.loc[terminal_15, "close"])
        close_30 = float(symbol_bars.loc[terminal_30, "close"])
        terminal_rows.append(
            {
                "decision_id": decision_id,
                "parent_slate_id": str(row.parent_slate_id),
                "symbol": str(row.symbol),
                "entry_minute_ordinal": entry_ordinal,
                "entry_timestamp_utc": entry_timestamp,
                "entry_open": entry_open,
                "fifteen_minute_terminal_ordinal": terminal_15,
                "fifteen_minute_terminal_close_timestamp_utc": pd.Timestamp(
                    symbol_bars.loc[terminal_15, "timestamp_utc"]
                )
                + pd.Timedelta(minutes=1),
                "fifteen_minute_terminal_close": close_15,
                "raw_long_return_15_bps": 10_000.0 * (close_15 / entry_open - 1.0),
                "raw_short_return_15_bps": -10_000.0 * (close_15 / entry_open - 1.0),
                "thirty_minute_terminal_ordinal": terminal_30,
                "thirty_minute_terminal_close_timestamp_utc": pd.Timestamp(
                    symbol_bars.loc[terminal_30, "timestamp_utc"]
                )
                + pd.Timedelta(minutes=1),
                "thirty_minute_terminal_close": close_30,
                "raw_long_return_30_bps": 10_000.0 * (close_30 / entry_open - 1.0),
                "raw_short_return_30_bps": -10_000.0 * (close_30 / entry_open - 1.0),
            }
        )
    paths = pd.DataFrame(path_rows)
    paths["cohort_median_cumulative_return_bps"] = np.nan
    for _, positions in paths.groupby(
        ["parent_slate_id", "relative_minute"], sort=True
    ).groups.items():
        group = paths.loc[positions]
        values = group["cumulative_return_bps"].to_numpy(dtype=float)
        for offset, row_index in enumerate(group.index):
            paths.loc[row_index, "cohort_median_cumulative_return_bps"] = float(
                np.median(np.delete(values, offset))
            )
    paths["cumulative_residual_return_bps"] = (
        paths["cumulative_return_bps"] - paths["cohort_median_cumulative_return_bps"]
    )
    terminals = pd.DataFrame(terminal_rows)
    for horizon in (15, 30):
        raw = f"raw_long_return_{horizon}_bps"
        relative = f"cohort_relative_return_{horizon}_bps"
        terminals[relative] = np.nan
        for _, positions in terminals.groupby("parent_slate_id", sort=True).groups.items():
            group = terminals.loc[positions]
            values = group[raw].to_numpy(dtype=float)
            for offset, row_index in enumerate(group.index):
                terminals.loc[row_index, relative] = values[offset] - float(
                    np.median(np.delete(values, offset))
                )
    barriers = development_onset_barriers(paths.loc[paths["high_movement_admitted"]].copy())
    labels: list[dict[str, Any]] = []
    for decision_id, group in paths.groupby("decision_id", sort=True):
        ordered = group.sort_values("relative_minute", kind="mergesort")
        checkpoint = int(ordered["decision_ordinal"].iloc[0])
        label = classify_onset(
            ordered["cumulative_residual_return_bps"].to_numpy(dtype=float),
            barrier_bps=barriers[checkpoint],
        )
        labels.append(
            {
                "decision_id": decision_id,
                "onset_barrier_bps": barriers[checkpoint],
                "onset_label": label,
                "directional_onset": int(label != "NO_ONSET"),
                "up_given_onset": 1 if label == "UP_ONSET" else 0 if label == "DOWN_ONSET" else -1,
            }
        )
    label_frame = pd.DataFrame(labels)
    paths = paths.merge(label_frame, on="decision_id", how="left", validate="many_to_one")
    output = features.copy()
    output["decision_id"] = (
        output["parent_slate_id"].astype(str) + "|" + output["symbol"].astype(str)
    )
    output = output.merge(label_frame, on="decision_id", how="left", validate="one_to_one")
    output = output.merge(terminals, on=["decision_id", "parent_slate_id", "symbol"], how="left")
    output = output.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    paths = paths.sort_values(
        ["session", "decision_ordinal", "symbol", "relative_minute"], kind="mergesort"
    ).reset_index(drop=True)
    return output, paths, barriers


def fit_fixed_model(
    frame: pd.DataFrame, target: str, features: tuple[str, ...], model_id: str
) -> dict[str, Any]:
    """Fit one deterministic standardized fixed L2 logistic model."""

    values = frame.loc[:, list(features)].to_numpy(dtype=float)
    labels = frame[target].to_numpy(dtype=int)
    if not np.isfinite(values).all() or set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("blocked_model_convergence_failure")
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(np.isfinite(scales) & (scales >= 1e-12), scales, 1.0)
    counts = frame.groupby("parent_slate_id", sort=False)["parent_slate_id"].transform("size")
    sample_weight = 1.0 / counts.to_numpy(dtype=float)
    totals = (
        pd.Series(sample_weight)
        .groupby(frame["parent_slate_id"].astype(str).reset_index(drop=True), sort=True)
        .sum()
    )
    if not np.allclose(totals.to_numpy(dtype=float), 1.0, atol=1e-12):
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    estimator = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=250,
        class_weight=None,
        random_state=BOOTSTRAP_SEED,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("error", category=ConvergenceWarning)
        try:
            estimator.fit((values - means) / scales, labels, sample_weight=sample_weight)
        except ConvergenceWarning as exc:
            raise RuntimeError("blocked_model_convergence_failure") from exc
    iterations = int(np.max(estimator.n_iter_))
    if iterations >= 250:
        raise RuntimeError("blocked_model_convergence_failure")
    return {
        "model_id": model_id,
        "target": target,
        "feature_names": list(features),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": estimator.coef_[0].astype(float).tolist(),
        "intercept": float(estimator.intercept_[0]),
        "training_rows": len(frame),
        "training_sessions": int(frame["session"].nunique()),
        "training_stocks": int(frame["symbol"].nunique()),
        "training_slates": int(frame["parent_slate_id"].nunique()),
        "iterations": iterations,
        "converged": True,
        "penalty": "l2",
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 250,
        "class_weight": None,
        "n_jobs": 1,
        "row_weight": "1 / admitted_rows_in_slate",
    }


def predict_fixed_model(model: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    names = [str(value) for value in model["feature_names"]]
    values = frame.loc[:, names].to_numpy(dtype=float)
    means = np.asarray(model["means"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    linear = float(model["intercept"]) + ((values - means) / scales) @ coefficients
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))))


def fit_and_score_ladder(
    panel: pd.DataFrame, *, conditional_direction_supported: bool
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    """Fit at most eight primary models on 2024 and score only 2025."""

    development = panel.loc[panel["year"].eq(2024)].copy()
    assessment = panel.loc[panel["year"].eq(2025)].copy()
    models: dict[str, dict[str, Any]] = {}
    for model_id in ("A0", "A1", "A2", "A3"):
        models[model_id] = fit_fixed_model(
            development, "directional_onset", MODEL_FEATURES[model_id], model_id
        )
    if conditional_direction_supported:
        direction = development.loc[development["directional_onset"].eq(1)].copy()
        for model_id in ("D0", "D1", "D2", "D3"):
            models[model_id] = fit_fixed_model(
                direction, "up_given_onset", MODEL_FEATURES[model_id], model_id
            )
    if len(models) > 8:
        raise RuntimeError("blocked_quick_activity_screen_resource_limit")
    scored = assessment.copy()
    prediction_rows: list[dict[str, Any]] = []
    for model_id, model in models.items():
        stage = "onset" if model_id.startswith("A") else "direction_given_onset"
        column = f"p_onset__{model_id}" if model_id.startswith("A") else f"p_up__{model_id}"
        probabilities = predict_fixed_model(model, scored)
        scored[column] = probabilities
        target_column = "directional_onset" if model_id.startswith("A") else "up_given_onset"
        for row, probability in zip(scored.itertuples(index=False), probabilities, strict=True):
            prediction_rows.append(
                {
                    "symbol": str(row.symbol),
                    "session": str(row.session),
                    "year_month": str(row.year_month),
                    "decision_ordinal": int(row.decision_ordinal),
                    "parent_slate_id": str(row.parent_slate_id),
                    "admitted_stock_count": int(row.admitted_stock_count),
                    "model": model_id,
                    "stage": stage,
                    "target": target_column,
                    "outcome": int(getattr(row, target_column)),
                    "probability": float(probability),
                }
            )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["session", "decision_ordinal", "symbol", "model"], kind="mergesort"
    )
    return scored.reset_index(drop=True), models, predictions.reset_index(drop=True)


def _calibration_parameters(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return math.nan, math.nan
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    logits = np.log(clipped / (1.0 - clipped))

    def objective(parameters: np.ndarray) -> float:
        linear = parameters[0] + parameters[1] * logits
        fitted = 1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0)))
        return float(
            -np.sum(
                labels * np.log(np.clip(fitted, 1e-15, 1.0))
                + (1.0 - labels) * np.log(np.clip(1.0 - fitted, 1e-15, 1.0))
            )
        )

    result = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    return float(result.x[0]), float(result.x[1])


def binary_metric(
    frame: pd.DataFrame,
    *,
    target: str,
    probability: str,
    stage: str,
    model: str,
    scope_type: str,
    scope_value: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = frame[target].to_numpy(dtype=int)
    probabilities = frame[probability].to_numpy(dtype=float)
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    brier = float(np.mean((labels - probabilities) ** 2))
    loss = float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
    auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else math.nan
    intercept, slope = _calibration_parameters(labels, probabilities)
    bins = np.minimum((probabilities * 10.0).astype(int), 9)
    calibration_rows: list[dict[str, Any]] = []
    ece = 0.0
    for bin_number in range(10):
        mask = bins == bin_number
        rows = int(mask.sum())
        mean_probability = float(np.mean(probabilities[mask])) if rows else math.nan
        outcome_rate = float(np.mean(labels[mask])) if rows else math.nan
        if rows:
            ece += rows * abs(mean_probability - outcome_rate)
        calibration_rows.append(
            {
                "stage": stage,
                "model": model,
                "scope_type": scope_type,
                "scope_value": scope_value,
                "bin": bin_number + 1,
                "mean_probability": mean_probability,
                "outcome_rate": outcome_rate,
                "rows": rows,
            }
        )
    record = {
        "population": "assessment_admitted",
        "scope_type": scope_type,
        "scope_value": scope_value,
        "stage": stage,
        "model": model,
        "brier_score": brier,
        "log_loss": loss,
        "auc": auc,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece / len(frame),
        "base_rate": float(np.mean(labels)),
        "rows": len(frame),
        "sessions": int(frame["session"].nunique()),
        "stocks": int(frame["symbol"].nunique()),
    }
    return record, calibration_rows


def evaluate_ladder(
    scored: pd.DataFrame, *, conditional_direction_supported: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate pooled, checkpoint, month, singleton, and multi-candidate slices."""

    onset_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    slices: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", scored)]
    slices.extend(
        ("checkpoint", str(int(value)), part)
        for value, part in scored.groupby("decision_ordinal", sort=True)
    )
    slices.extend(
        ("month", str(value), part) for value, part in scored.groupby("year_month", sort=True)
    )
    slices.extend(
        [
            ("slate_type", "singleton", scored.loc[scored["admitted_stock_count"].eq(1)]),
            (
                "slate_type",
                "multi_candidate",
                scored.loc[scored["admitted_stock_count"].gt(1)],
            ),
        ]
    )
    for scope_type, scope_value, scope in slices:
        if scope.empty:
            continue
        for model in ("A0", "A1", "A2", "A3"):
            record, bins = binary_metric(
                scope,
                target="directional_onset",
                probability=f"p_onset__{model}",
                stage="onset",
                model=model,
                scope_type=scope_type,
                scope_value=scope_value,
            )
            calibration_rows.extend(bins)
            if scope_type == "checkpoint":
                checkpoint_rows.append(record)
            elif scope_type == "month":
                monthly_rows.append(record)
            else:
                onset_rows.append(record)
        if not conditional_direction_supported:
            continue
        direction = scope.loc[scope["directional_onset"].eq(1)]
        if direction.empty:
            continue
        for model in ("D0", "D1", "D2", "D3"):
            record, bins = binary_metric(
                direction,
                target="up_given_onset",
                probability=f"p_up__{model}",
                stage="direction",
                model=model,
                scope_type=scope_type,
                scope_value=scope_value,
            )
            calibration_rows.extend(bins)
            if scope_type == "checkpoint":
                checkpoint_rows.append(record)
            elif scope_type == "month":
                monthly_rows.append(record)
            else:
                direction_rows.append(record)
    sorting = ["scope_type", "scope_value", "stage", "model"]
    return (
        pd.DataFrame(onset_rows).sort_values(sorting, kind="mergesort").reset_index(drop=True),
        pd.DataFrame(direction_rows).sort_values(sorting, kind="mergesort").reset_index(drop=True),
        pd.DataFrame(checkpoint_rows).sort_values(sorting, kind="mergesort").reset_index(drop=True),
        pd.DataFrame(monthly_rows).sort_values(sorting, kind="mergesort").reset_index(drop=True),
        pd.DataFrame(calibration_rows)
        .sort_values([*sorting, "bin"], kind="mergesort")
        .reset_index(drop=True),
    )


def support_summary(panel: pd.DataFrame) -> dict[str, Any]:
    assessment = panel.loc[panel["year"].eq(2025)]
    counts = assessment["onset_label"].value_counts().to_dict()
    row_shares = assessment["symbol"].value_counts(normalize=True)
    aggregate_gates = {
        "assessment_rows_at_least_1200": len(assessment) >= 1_200,
        "assessment_sessions_at_least_100": assessment["session"].nunique() >= 100,
        "assessment_stocks_at_least_15": assessment["symbol"].nunique() >= 15,
        "represented_months_at_least_6": assessment["year_month"].nunique() >= 6,
        "each_parent_slate_has_at_least_15_valid_stocks": bool(
            assessment["one_minute_parent_valid_stock_count"].ge(15).all()
        ),
        "maximum_stock_row_share_at_most_12_5_pct": float(row_shares.max()) <= 0.125,
    }
    conditional_gates = {
        "directional_onsets_at_least_250": int(assessment["directional_onset"].sum()) >= 250,
        "up_onsets_at_least_100": int(counts.get("UP_ONSET", 0)) >= 100,
        "down_onsets_at_least_100": int(counts.get("DOWN_ONSET", 0)) >= 100,
    }
    return {
        **SAFETY_FLAGS,
        "development_rows": int(panel["year"].eq(2024).sum()),
        "development_sessions": int(panel.loc[panel["year"].eq(2024), "session"].nunique()),
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "assessment_checkpoint_rows": {
            str(key): int(value)
            for key, value in assessment.groupby("decision_ordinal", sort=True).size().items()
        },
        "up_onsets": int(counts.get("UP_ONSET", 0)),
        "down_onsets": int(counts.get("DOWN_ONSET", 0)),
        "no_onsets": int(counts.get("NO_ONSET", 0)),
        "directional_onsets": int(assessment["directional_onset"].sum()),
        "maximum_stock_row_share": float(row_shares.max()),
        "largest_row_contributor": str(row_shares.idxmax()),
        "aggregate_gates": aggregate_gates,
        "conditional_direction_gates": conditional_gates,
        "aggregate_support_passes": all(aggregate_gates.values()),
        "conditional_direction_support_passes": all(conditional_gates.values()),
    }


def _stable_random_selection(frame: pd.DataFrame, slate_id: str) -> tuple[pd.Series, float]:
    ordered = frame.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256(f"{ECONOMIC_RANDOM_SEED}:{slate_id}".encode()).digest()
    row = ordered.iloc[int.from_bytes(digest[:8], "big") % len(ordered)]
    direction = 1.0 if digest[8] % 2 == 0 else -1.0
    return row, direction


def build_economic_selections(scored: pd.DataFrame) -> pd.DataFrame:
    """Select one delayed diagnostic stock per slate for six fixed comparators."""

    rows: list[dict[str, Any]] = []
    for slate_id, slate in scored.groupby("parent_slate_id", sort=True):
        working = slate.copy()
        specifications = {
            "price_system": working["p_onset__A1"]
            * (2.0 * working["p_up__D1"] - 1.0)
            * working["p_large_remaining_move"],
            "activity_system": working["p_onset__A2"]
            * (2.0 * working["p_up__D2"] - 1.0)
            * working["p_large_remaining_move"],
            "interaction_system": working["p_onset__A3"]
            * (2.0 * working["p_up__D3"] - 1.0)
            * working["p_large_remaining_move"],
            "highest_one_minute_relative_momentum": working["cohort_relative_return_10"],
            "strongest_one_minute_reversal": -working["cohort_relative_return_3"],
        }
        for candidate, score in specifications.items():
            ranked = working.assign(_score=score, _absolute_score=np.abs(score)).sort_values(
                ["_absolute_score", "symbol"],
                ascending=[False, True],
                kind="mergesort",
            )
            selected = ranked.iloc[0]
            direction = 1.0 if float(selected["_score"]) >= 0.0 else -1.0
            rows.append(
                {
                    "candidate": candidate,
                    "parent_slate_id": str(slate_id),
                    "symbol": str(selected["symbol"]),
                    "session": str(selected["session"]),
                    "year_month": str(selected["year_month"]),
                    "decision_ordinal": int(selected["decision_ordinal"]),
                    "score": float(selected["_score"]),
                    "direction": direction,
                    "signed_return_15_bps": direction * float(selected["raw_long_return_15_bps"]),
                    "signed_return_30_bps": direction * float(selected["raw_long_return_30_bps"]),
                    "cohort_relative_signed_return_15_bps": direction
                    * float(selected["cohort_relative_return_15_bps"]),
                    "cohort_relative_signed_return_30_bps": direction
                    * float(selected["cohort_relative_return_30_bps"]),
                }
            )
        selected, direction = _stable_random_selection(working, str(slate_id))
        rows.append(
            {
                "candidate": "random_admitted_stock",
                "parent_slate_id": str(slate_id),
                "symbol": str(selected["symbol"]),
                "session": str(selected["session"]),
                "year_month": str(selected["year_month"]),
                "decision_ordinal": int(selected["decision_ordinal"]),
                "score": direction,
                "direction": direction,
                "signed_return_15_bps": direction * float(selected["raw_long_return_15_bps"]),
                "signed_return_30_bps": direction * float(selected["raw_long_return_30_bps"]),
                "cohort_relative_signed_return_15_bps": direction
                * float(selected["cohort_relative_return_15_bps"]),
                "cohort_relative_signed_return_30_bps": direction
                * float(selected["cohort_relative_return_30_bps"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["candidate", "session", "decision_ordinal"], kind="mergesort")
        .reset_index(drop=True)
    )


def economic_metrics(selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slices: list[tuple[str, str, pd.DataFrame]] = [("pooled", "all", selections)]
    slices.extend(
        ("month", str(value), part) for value, part in selections.groupby("year_month", sort=True)
    )
    slices.extend(
        ("checkpoint", str(int(value)), part)
        for value, part in selections.groupby("decision_ordinal", sort=True)
    )
    for scope_type, scope_value, scope in slices:
        for candidate, candidate_rows in scope.groupby("candidate", sort=True):
            for horizon in (15, 30):
                gross = candidate_rows[f"signed_return_{horizon}_bps"].to_numpy(dtype=float)
                relative = candidate_rows[f"cohort_relative_signed_return_{horizon}_bps"].to_numpy(
                    dtype=float
                )
                for friction in (0.0, 10.0, 20.0):
                    net = gross - friction
                    rows.append(
                        {
                            "candidate": str(candidate),
                            "scope_type": scope_type,
                            "scope_value": scope_value,
                            "horizon": f"{horizon}_minute_terminal",
                            "friction_bps": friction,
                            "mean_signed_return_bps": float(np.mean(net)),
                            "mean_cohort_relative_signed_return_bps": float(
                                np.mean(relative - friction)
                            ),
                            "positive_selection_pct": float(np.mean(net > 0.0)),
                            "rows": len(candidate_rows),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["scope_type", "scope_value", "candidate", "horizon", "friction_bps"],
        kind="mergesort",
    )


def concentration_metrics(
    assessment: pd.DataFrame, selections: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_shares = assessment["symbol"].value_counts(normalize=True)
    for symbol in SYMBOLS:
        share = float(row_shares.get(symbol, 0.0))
        rows.append(
            {
                "scope": "assessment_admitted_rows",
                "candidate": "all",
                "symbol": symbol,
                "row_share": share,
                "selection_share": math.nan,
                "passes": share <= 0.125,
            }
        )
    candidate_maxima: dict[str, float] = {}
    for candidate, group in selections.groupby("candidate", sort=True):
        shares = group["symbol"].value_counts(normalize=True)
        candidate_maxima[str(candidate)] = float(shares.max())
        for symbol in SYMBOLS:
            share = float(shares.get(symbol, 0.0))
            rows.append(
                {
                    "scope": "economic_selections",
                    "candidate": str(candidate),
                    "symbol": symbol,
                    "row_share": math.nan,
                    "selection_share": share,
                    "passes": share <= 0.20,
                }
            )
    summary = {
        "maximum_assessment_row_share": float(row_shares.max()),
        "assessment_row_concentration_passes": float(row_shares.max()) <= 0.125,
        "maximum_selection_share_by_candidate": candidate_maxima,
        "all_selection_concentration_gates_pass": max(candidate_maxima.values()) <= 0.20,
    }
    return pd.DataFrame(rows), summary


def _loss_improvement(
    frame: pd.DataFrame, target: str, baseline: str, candidate: str, metric: str
) -> float:
    labels = frame[target].to_numpy(dtype=float)
    baseline_probability = frame[baseline].to_numpy(dtype=float)
    candidate_probability = frame[candidate].to_numpy(dtype=float)
    if metric == "brier":
        baseline_loss = (labels - baseline_probability) ** 2
        candidate_loss = (labels - candidate_probability) ** 2
    else:
        baseline_clipped = np.clip(baseline_probability, 1e-15, 1.0 - 1e-15)
        candidate_clipped = np.clip(candidate_probability, 1e-15, 1.0 - 1e-15)
        baseline_loss = -(
            labels * np.log(baseline_clipped) + (1.0 - labels) * np.log(1.0 - baseline_clipped)
        )
        candidate_loss = -(
            labels * np.log(candidate_clipped) + (1.0 - labels) * np.log(1.0 - candidate_clipped)
        )
    return float(np.mean(baseline_loss) - np.mean(candidate_loss))


def _weighted_loss_improvement(
    frame: pd.DataFrame,
    target: str,
    baseline: str,
    candidate: str,
    metric: str,
    session_counts: Counter[str],
) -> float:
    weights = frame["session"].astype(str).map(session_counts).fillna(0.0).to_numpy(dtype=float)
    mask = weights > 0.0
    subset = frame.loc[mask]
    labels = subset[target].to_numpy(dtype=float)
    baseline_probability = subset[baseline].to_numpy(dtype=float)
    candidate_probability = subset[candidate].to_numpy(dtype=float)
    if metric == "brier":
        baseline_loss = (labels - baseline_probability) ** 2
        candidate_loss = (labels - candidate_probability) ** 2
    else:
        base = np.clip(baseline_probability, 1e-15, 1.0 - 1e-15)
        candidate_values = np.clip(candidate_probability, 1e-15, 1.0 - 1e-15)
        baseline_loss = -(labels * np.log(base) + (1.0 - labels) * np.log(1.0 - base))
        candidate_loss = -(
            labels * np.log(candidate_values) + (1.0 - labels) * np.log(1.0 - candidate_values)
        )
    selected_weights = weights[mask]
    return float(
        np.average(baseline_loss, weights=selected_weights)
        - np.average(candidate_loss, weights=selected_weights)
    )


def _economic_increment(
    selections: pd.DataFrame,
    baseline: str,
    candidate: str,
    session_counts: Counter[str] | None = None,
) -> float:
    means: dict[str, float] = {}
    for name in (baseline, candidate):
        frame = selections.loc[selections["candidate"].eq(name)]
        if session_counts is None:
            means[name] = float(frame["signed_return_15_bps"].mean() - 20.0)
        else:
            weights = frame["session"].astype(str).map(session_counts).fillna(0.0)
            mask = weights.gt(0.0)
            means[name] = float(
                np.average(
                    frame.loc[mask, "signed_return_15_bps"].to_numpy(dtype=float) - 20.0,
                    weights=weights.loc[mask].to_numpy(dtype=float),
                )
            )
    return means[candidate] - means[baseline]


def bootstrap_metrics(
    scored: pd.DataFrame, selections: pd.DataFrame, *, direction_supported: bool
) -> pd.DataFrame:
    """Run exactly 200 paired whole-session bootstrap draws."""

    specifications: list[tuple[str, pd.DataFrame, str, str, str, str]] = []
    for baseline, candidate in (("A0", "A1"), ("A1", "A2"), ("A2", "A3")):
        for metric in ("brier", "log_loss"):
            specifications.append(
                (
                    f"{candidate}_minus_{baseline}",
                    scored,
                    "directional_onset",
                    f"p_onset__{baseline}",
                    f"p_onset__{candidate}",
                    metric,
                )
            )
    if direction_supported:
        direction = scored.loc[scored["directional_onset"].eq(1)]
        for baseline, candidate in (("D0", "D1"), ("D1", "D2"), ("D2", "D3")):
            for metric in ("brier", "log_loss"):
                specifications.append(
                    (
                        f"{candidate}_minus_{baseline}",
                        direction,
                        "up_given_onset",
                        f"p_up__{baseline}",
                        f"p_up__{candidate}",
                        metric,
                    )
                )
    unique_sessions = np.asarray(sorted(scored["session"].astype(str).unique()), dtype=object)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [
        tuple(
            str(value)
            for value in generator.choice(unique_sessions, size=len(unique_sessions), replace=True)
        )
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    values: dict[tuple[str, str], list[float]] = {
        (comparison, metric): [] for comparison, _, _, _, _, metric in specifications
    }
    values[("activity_system_minus_price_system", "economic_15m_after_20bps")] = []
    values[("interaction_system_minus_activity_system", "economic_15m_after_20bps")] = []
    rows: list[dict[str, Any]] = []
    for draw_number, draw in enumerate(draws):
        counts: Counter[str] = Counter(draw)
        for comparison, frame, target, baseline, candidate, metric in specifications:
            value = _weighted_loss_improvement(frame, target, baseline, candidate, metric, counts)
            values[(comparison, metric)].append(value)
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw_number,
                    "comparison": comparison,
                    "metric": metric,
                    "estimate": value,
                    "lower_90": math.nan,
                    "upper_90": math.nan,
                    "lower_95": math.nan,
                    "upper_95": math.nan,
                    "draws": BOOTSTRAP_DRAWS,
                }
            )
        for baseline, candidate in (
            ("price_system", "activity_system"),
            ("activity_system", "interaction_system"),
        ):
            comparison = f"{candidate}_minus_{baseline}"
            value = _economic_increment(selections, baseline, candidate, counts)
            values[(comparison, "economic_15m_after_20bps")].append(value)
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw_number,
                    "comparison": comparison,
                    "metric": "economic_15m_after_20bps",
                    "estimate": value,
                    "lower_90": math.nan,
                    "upper_90": math.nan,
                    "lower_95": math.nan,
                    "upper_95": math.nan,
                    "draws": BOOTSTRAP_DRAWS,
                }
            )
    for (comparison, metric), draw_values in values.items():
        array = np.asarray(draw_values, dtype=float)
        if metric == "economic_15m_after_20bps":
            baseline, candidate = (
                ("price_system", "activity_system")
                if comparison.startswith("activity_system")
                else ("activity_system", "interaction_system")
            )
            estimate = _economic_increment(selections, baseline, candidate)
        else:
            specification = next(
                item for item in specifications if item[0] == comparison and item[5] == metric
            )
            _, frame, target, baseline, candidate, _ = specification
            estimate = _loss_improvement(frame, target, baseline, candidate, metric)
        rows.append(
            {
                "record_type": "summary",
                "draw": -1,
                "comparison": comparison,
                "metric": metric,
                "estimate": estimate,
                "lower_90": float(np.quantile(array, 0.05)),
                "upper_90": float(np.quantile(array, 0.95)),
                "lower_95": float(np.quantile(array, 0.025)),
                "upper_95": float(np.quantile(array, 0.975)),
                "draws": BOOTSTRAP_DRAWS,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["record_type", "comparison", "metric", "draw"], kind="mergesort"
    )


def _permute_activity_bundle(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, str]:
    columns = [*ACTIVITY_FEATURES, *INTERACTION_FEATURES]
    output = frame.copy().reset_index(drop=True)
    generator = np.random.default_rng(seed)
    for _, positions in output.groupby("parent_slate_id", sort=True).groups.items():
        indices = list(positions)
        if len(indices) < 2:
            continue
        source = output.loc[indices, columns].to_numpy(copy=True)
        output.loc[indices, columns] = source[generator.permutation(len(indices))]
    digest = sha256_text(
        output.loc[:, ["parent_slate_id", "symbol", *columns]].to_csv(
            index=False, lineterminator="\n", float_format="%.17g"
        )
    )
    return output, digest


def within_slate_activity_null(
    panel: pd.DataFrame,
    scored: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    direction_supported: bool,
) -> pd.DataFrame:
    """Run exactly 50 fixed-seed within-admitted-slate activity-bundle null draws."""

    if not direction_supported:
        return pd.DataFrame(columns=list(CSV_SCHEMAS["null_metrics.csv"]))
    real_values = {
        "A2_minus_A1": _loss_improvement(
            scored, "directional_onset", "p_onset__A1", "p_onset__A2", "brier"
        ),
        "A3_minus_A2": _loss_improvement(
            scored, "directional_onset", "p_onset__A2", "p_onset__A3", "brier"
        ),
        "D2_minus_D1": _loss_improvement(
            scored.loc[scored["directional_onset"].eq(1)],
            "up_given_onset",
            "p_up__D1",
            "p_up__D2",
            "brier",
        ),
        "D3_minus_D2": _loss_improvement(
            scored.loc[scored["directional_onset"].eq(1)],
            "up_given_onset",
            "p_up__D2",
            "p_up__D3",
            "brier",
        ),
        "activity_system_minus_price_system": _economic_increment(
            selections, "price_system", "activity_system"
        ),
    }
    values = {key: [] for key in real_values}
    hashes: list[str] = []
    rows: list[dict[str, Any]] = []
    for draw in range(NULL_DRAWS):
        permuted, bundle_hash = _permute_activity_bundle(panel, NULL_SEED + draw)
        hashes.append(bundle_hash)
        development = permuted.loc[permuted["year"].eq(2024)]
        assessment = permuted.loc[permuted["year"].eq(2025)].copy()
        null_models: dict[str, dict[str, Any]] = {}
        for model_id in ("A2", "A3"):
            null_models[model_id] = fit_fixed_model(
                development,
                "directional_onset",
                MODEL_FEATURES[model_id],
                f"null_{draw}_{model_id}",
            )
            assessment[f"p_onset__{model_id}"] = predict_fixed_model(
                null_models[model_id], assessment
            )
        development_direction = development.loc[development["directional_onset"].eq(1)]
        for model_id in ("D2", "D3"):
            null_models[model_id] = fit_fixed_model(
                development_direction,
                "up_given_onset",
                MODEL_FEATURES[model_id],
                f"null_{draw}_{model_id}",
            )
            assessment[f"p_up__{model_id}"] = predict_fixed_model(null_models[model_id], assessment)
        fixed_columns = [
            "symbol",
            "session",
            "decision_ordinal",
            "p_onset__A1",
            "p_up__D1",
        ]
        assessment = assessment.merge(
            scored.loc[:, fixed_columns],
            on=["symbol", "session", "decision_ordinal"],
            how="left",
            validate="one_to_one",
        )
        null_direction = assessment.loc[assessment["directional_onset"].eq(1)]
        draw_values = {
            "A2_minus_A1": _loss_improvement(
                assessment,
                "directional_onset",
                "p_onset__A1",
                "p_onset__A2",
                "brier",
            ),
            "A3_minus_A2": _loss_improvement(
                assessment,
                "directional_onset",
                "p_onset__A2",
                "p_onset__A3",
                "brier",
            ),
            "D2_minus_D1": _loss_improvement(
                null_direction, "up_given_onset", "p_up__D1", "p_up__D2", "brier"
            ),
            "D3_minus_D2": _loss_improvement(
                null_direction, "up_given_onset", "p_up__D2", "p_up__D3", "brier"
            ),
        }
        null_selections = build_economic_selections(assessment)
        draw_values["activity_system_minus_price_system"] = _economic_increment(
            null_selections, "price_system", "activity_system"
        )
        for comparison, value in draw_values.items():
            values[comparison].append(value)
            rows.append(
                {
                    "record_type": "draw",
                    "draw": draw,
                    "comparison": comparison,
                    "metric": (
                        "economic_15m_after_20bps"
                        if comparison == "activity_system_minus_price_system"
                        else "brier"
                    ),
                    "real_value": real_values[comparison],
                    "null_value": value,
                    "null_q90": math.nan,
                    "real_percentile": math.nan,
                    "draws": NULL_DRAWS,
                    "activity_bundle_sha256": bundle_hash,
                }
            )
    for comparison, draw_values in values.items():
        array = np.asarray(draw_values, dtype=float)
        real = real_values[comparison]
        rows.append(
            {
                "record_type": "summary",
                "draw": -1,
                "comparison": comparison,
                "metric": (
                    "economic_15m_after_20bps"
                    if comparison == "activity_system_minus_price_system"
                    else "brier"
                ),
                "real_value": real,
                "null_value": float(np.mean(array)),
                "null_q90": float(np.quantile(array, 0.90)),
                "real_percentile": float(np.mean(array < real)),
                "draws": NULL_DRAWS,
                "activity_bundle_sha256": sha256_text("\n".join(hashes)),
            }
        )
    return pd.DataFrame(rows).sort_values(["record_type", "comparison", "draw"], kind="mergesort")


def feature_group_diagnostics(
    scored: pd.DataFrame, models: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    """Describe fixed feature groups without fitting candidate models."""

    groups = {
        "price_only_sequence": list(PRICE_FEATURES),
        "raw_activity": list(ACTIVITY_FEATURES),
        "activity_timing": [
            "maximum_relative_activity_minute_index",
            "maximum_absolute_return_minute_index",
            "price_peak_index_minus_activity_peak_index",
        ],
        "absorption_interactions": ["activity_absorption_3", "activity_absorption_wick"],
        "continuation_interactions": [
            "activity_continuation_3",
            "activity_continuation_5",
            "signed_progress_per_activity_3",
            "absolute_progress_per_activity_3",
            "activity_lead_price_response",
            "activity_range_response",
        ],
    }
    rows: list[dict[str, Any]] = []
    generator = np.random.default_rng(20260723)
    for model_id in ("A3", "D3"):
        model = models[model_id]
        stage = "onset" if model_id == "A3" else "direction"
        frame = scored if stage == "onset" else scored.loc[scored["directional_onset"].eq(1)]
        target = "directional_onset" if stage == "onset" else "up_given_onset"
        original = predict_fixed_model(model, frame)
        original_brier = float(np.mean((frame[target].to_numpy(dtype=float) - original) ** 2))
        coefficient_map = dict(zip(model["feature_names"], model["coefficients"], strict=True))
        for group_name, columns in groups.items():
            present = [column for column in columns if column in coefficient_map]
            rows.append(
                {
                    "stage": stage,
                    "model": model_id,
                    "feature_group": group_name,
                    "diagnostic": "mean_absolute_standardised_coefficient",
                    "scope": "pooled",
                    "value": float(
                        np.mean([abs(float(coefficient_map[column])) for column in present])
                    ),
                }
            )
            permuted = frame.copy()
            order = generator.permutation(len(permuted))
            permuted.loc[:, present] = permuted.loc[:, present].to_numpy()[order]
            probability = predict_fixed_model(model, permuted)
            permuted_brier = float(
                np.mean((frame[target].to_numpy(dtype=float) - probability) ** 2)
            )
            rows.append(
                {
                    "stage": stage,
                    "model": model_id,
                    "feature_group": group_name,
                    "diagnostic": "assessment_permutation_brier_increase",
                    "scope": "pooled",
                    "value": permuted_brier - original_brier,
                }
            )
    activity_quintile = pd.qcut(
        scored["mean_relative_activity_5"].rank(method="first"), 5, labels=False
    )
    for quintile in range(5):
        frame = scored.loc[activity_quintile.eq(quintile)]
        for stage, target, baseline, candidate in (
            ("onset", "directional_onset", "p_onset__A1", "p_onset__A2"),
            ("direction", "up_given_onset", "p_up__D1", "p_up__D2"),
        ):
            if stage == "direction":
                frame = frame.loc[frame["directional_onset"].eq(1)]
            rows.append(
                {
                    "stage": stage,
                    "model": "activity_vs_price",
                    "feature_group": "raw_activity",
                    "diagnostic": "brier_improvement_by_activity_quintile",
                    "scope": f"quintile_{quintile + 1}",
                    "value": _loss_improvement(frame, target, baseline, candidate, "brier"),
                }
            )
    pattern_values = {
        "price_only_sequence": scored["cohort_relative_return_10"].abs(),
        "raw_activity": scored["mean_relative_activity_5"],
        "activity_timing": scored["price_peak_index_minus_activity_peak_index"],
        "absorption_interactions": scored[
            ["activity_absorption_3", "activity_absorption_wick"]
        ].mean(axis=1),
        "continuation_interactions": scored[
            ["activity_continuation_3", "activity_continuation_5"]
        ].mean(axis=1),
    }
    for group_name, pattern in pattern_values.items():
        quintiles = pd.qcut(pattern.rank(method="first"), 5, labels=False)
        for quintile in range(5):
            mask = quintiles.eq(quintile)
            rows.append(
                {
                    "stage": "onset",
                    "model": "descriptive",
                    "feature_group": group_name,
                    "diagnostic": "directional_onset_rate_by_pattern_quintile",
                    "scope": f"quintile_{quintile + 1}",
                    "value": float(scored.loc[mask, "directional_onset"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["stage", "model", "feature_group", "diagnostic", "scope"], kind="mergesort"
    )


def _metric_row(frame: pd.DataFrame, stage: str, model: str) -> pd.Series:
    return frame.loc[
        frame["scope_type"].eq("pooled")
        & frame["scope_value"].eq("all")
        & frame["stage"].eq(stage)
        & frame["model"].eq(model)
    ].iloc[0]


def _summary_value(frame: pd.DataFrame, comparison: str, metric: str, column: str) -> float:
    return float(
        frame.loc[
            frame["record_type"].eq("summary")
            & frame["comparison"].eq(comparison)
            & frame["metric"].eq(metric),
            column,
        ].iloc[0]
    )


def increment_evidence(
    *,
    stage: str,
    baseline: str,
    candidate: str,
    pooled: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    requires_null: bool,
    requires_concentration: bool,
    concentration_passes: bool,
) -> dict[str, Any]:
    baseline_row = _metric_row(pooled, stage, baseline)
    candidate_row = _metric_row(pooled, stage, candidate)
    comparison = f"{candidate}_minus_{baseline}"
    brier = float(baseline_row["brier_score"] - candidate_row["brier_score"])
    loss = float(baseline_row["log_loss"] - candidate_row["log_loss"])
    month_values: dict[str, dict[str, float]] = {}
    for month in sorted(monthly["scope_value"].astype(str).unique()):
        subset = monthly.loc[
            monthly["scope_value"].astype(str).eq(month) & monthly["stage"].eq(stage)
        ]
        base = subset.loc[subset["model"].eq(baseline)].iloc[0]
        richer = subset.loc[subset["model"].eq(candidate)].iloc[0]
        month_values[month] = {
            "brier_improvement": float(base["brier_score"] - richer["brier_score"]),
            "log_loss_improvement": float(base["log_loss"] - richer["log_loss"]),
        }
    positive_months = sum(
        values["brier_improvement"] > 0.0 and values["log_loss_improvement"] > 0.0
        for values in month_values.values()
    )
    checkpoint_values: dict[str, dict[str, float]] = {}
    for value in ("6", "12"):
        subset = checkpoint.loc[
            checkpoint["scope_value"].astype(str).eq(value) & checkpoint["stage"].eq(stage)
        ]
        base = subset.loc[subset["model"].eq(baseline)].iloc[0]
        richer = subset.loc[subset["model"].eq(candidate)].iloc[0]
        checkpoint_values[value] = {
            "brier_improvement": float(base["brier_score"] - richer["brier_score"]),
            "log_loss_improvement": float(base["log_loss"] - richer["log_loss"]),
        }
    neither_checkpoint_materially_adverse = all(
        values["brier_improvement"] >= -0.001 and values["log_loss_improvement"] >= -0.001
        for values in checkpoint_values.values()
    )
    null_q90 = None
    real_null_percentile = None
    exceeds_null = True
    if requires_null:
        null_q90 = _summary_value(nulls, comparison, "brier", "null_q90")
        real_null_percentile = _summary_value(nulls, comparison, "brier", "real_percentile")
        exceeds_null = brier > null_q90
    gates = {
        "improves_brier": brier > 0.0,
        "improves_log_loss": loss > 0.0,
        "bootstrap_90_lower_brier_non_negative": _summary_value(
            bootstrap, comparison, "brier", "lower_90"
        )
        >= 0.0,
        "bootstrap_90_lower_log_loss_non_negative": _summary_value(
            bootstrap, comparison, "log_loss", "lower_90"
        )
        >= 0.0,
        "auc_not_reduced": float(candidate_row["auc"]) >= float(baseline_row["auc"]),
        "positive_in_at_least_five_months": positive_months >= 5,
        "neither_checkpoint_materially_adverse": neither_checkpoint_materially_adverse,
        "exceeds_activity_null_90th_percentile": exceeds_null,
        "concentration_gates_pass": concentration_passes or not requires_concentration,
    }
    return {
        "comparison": comparison,
        "brier_improvement": brier,
        "log_loss_improvement": loss,
        "auc_change": float(candidate_row["auc"] - baseline_row["auc"]),
        "bootstrap_lower_90": {
            "brier": _summary_value(bootstrap, comparison, "brier", "lower_90"),
            "log_loss": _summary_value(bootstrap, comparison, "log_loss", "lower_90"),
        },
        "positive_months": positive_months,
        "monthly_improvements": month_values,
        "checkpoint_improvements": checkpoint_values,
        "material_adversity_tolerance": -0.001,
        "null_q90_brier": null_q90,
        "real_null_percentile_brier": real_null_percentile,
        "gates": gates,
        "passes": all(gates.values()),
    }


def screen_decision(
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    monthly: pd.DataFrame,
    checkpoint: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    support: Mapping[str, Any],
    concentration: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(support["aggregate_support_passes"]):
        return {
            **SAFETY_FLAGS,
            "decision": "blocked_insufficient_one_minute_support",
            "support": dict(support),
            "probability_gates_evaluated": False,
        }
    concentration_passes = bool(
        concentration["assessment_row_concentration_passes"]
        and concentration["all_selection_concentration_gates_pass"]
    )
    increments: dict[str, Any] = {}
    increments["A1_minus_A0"] = increment_evidence(
        stage="onset",
        baseline="A0",
        candidate="A1",
        pooled=onset,
        monthly=monthly,
        checkpoint=checkpoint,
        bootstrap=bootstrap,
        nulls=nulls,
        requires_null=False,
        requires_concentration=False,
        concentration_passes=concentration_passes,
    )
    increments["A2_minus_A1"] = increment_evidence(
        stage="onset",
        baseline="A1",
        candidate="A2",
        pooled=onset,
        monthly=monthly,
        checkpoint=checkpoint,
        bootstrap=bootstrap,
        nulls=nulls,
        requires_null=True,
        requires_concentration=True,
        concentration_passes=concentration_passes,
    )
    increments["A3_minus_A2"] = increment_evidence(
        stage="onset",
        baseline="A2",
        candidate="A3",
        pooled=onset,
        monthly=monthly,
        checkpoint=checkpoint,
        bootstrap=bootstrap,
        nulls=nulls,
        requires_null=True,
        requires_concentration=True,
        concentration_passes=concentration_passes,
    )
    direction_supported = bool(support["conditional_direction_support_passes"])
    if direction_supported:
        for baseline, candidate in (("D0", "D1"), ("D1", "D2"), ("D2", "D3")):
            increments[f"{candidate}_minus_{baseline}"] = increment_evidence(
                stage="direction",
                baseline=baseline,
                candidate=candidate,
                pooled=direction,
                monthly=monthly,
                checkpoint=checkpoint,
                bootstrap=bootstrap,
                nulls=nulls,
                requires_null=candidate in {"D2", "D3"},
                requires_concentration=candidate in {"D2", "D3"},
                concentration_passes=concentration_passes,
            )
    flags = {
        "price_onset": bool(increments["A1_minus_A0"]["passes"]),
        "price_direction": bool(direction_supported and increments["D1_minus_D0"]["passes"]),
        "raw_activity_onset": bool(increments["A2_minus_A1"]["passes"]),
        "raw_activity_direction": bool(direction_supported and increments["D2_minus_D1"]["passes"]),
        "interaction_onset": bool(increments["A3_minus_A2"]["passes"]),
        "interaction_direction": bool(direction_supported and increments["D3_minus_D2"]["passes"]),
    }
    decision = decide_activity_screen(**flags)
    return {
        **SAFETY_FLAGS,
        "decision": decision,
        "conditional_direction_status": (
            "evaluated" if direction_supported else "conditional_direction_support_insufficient"
        ),
        "support": dict(support),
        "concentration": dict(concentration),
        "increments": increments,
        "pass_flags": flags,
        "probability_gates_evaluated": True,
        "economic_reference_cannot_override_probability_gates": True,
        "models_fitted": 8 if direction_supported else 4,
        "bootstrap_draws_run": BOOTSTRAP_DRAWS,
        "null_draws_run": NULL_DRAWS if direction_supported else 0,
        "protected_rows_opened": 0,
        "not_prospective_validation": True,
        "not_achieved_pnl": True,
        "not_a_deployable_strategy": True,
        "not_executable_edge_evidence": True,
    }


def plot_calibration(calibration: pd.DataFrame, *, stage: str, output: Path) -> None:
    prefix = "A" if stage == "onset" else "D"
    subset = calibration.loc[
        calibration["stage"].eq(stage)
        & calibration["scope_type"].eq("pooled")
        & calibration["scope_value"].eq("all")
        & calibration["rows"].gt(0)
    ]
    fig, axis = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    axis.plot([0.0, 1.0], [0.0, 1.0], color="#777777", linestyle="--", linewidth=1.0)
    colors = ["#4477AA", "#EE6677", "#228833", "#AA3377"]
    for model, color in zip((f"{prefix}{i}" for i in range(4)), colors, strict=True):
        frame = subset.loc[subset["model"].eq(model)].sort_values("bin")
        axis.plot(
            frame["mean_probability"],
            frame["outcome_rate"],
            marker="o",
            linewidth=1.6,
            label=model,
            color=color,
        )
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed rate",
        title=(
            "Directional-onset calibration (2025)"
            if stage == "onset"
            else "Direction-given-onset calibration (2025)"
        ),
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    axis.legend(frameon=False)
    fig.savefig(output, dpi=150, metadata={"Software": "Stocker research"})
    plt.close(fig)


def plot_economic_comparison(economic: pd.DataFrame, output: Path) -> None:
    candidates = ["price_system", "activity_system", "interaction_system"]
    subset = economic.loc[
        economic["scope_type"].eq("pooled")
        & economic["horizon"].eq("15_minute_terminal")
        & economic["friction_bps"].eq(20.0)
    ].set_index("candidate")
    values = [float(subset.loc[candidate, "mean_signed_return_bps"]) for candidate in candidates]
    fig, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    axis.bar(["Price", "Activity", "Interaction"], values, color=["#4477AA", "#EE6677", "#228833"])
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.set(
        ylabel="Mean signed return after 20 bps",
        title="Delayed 15-minute economic-reference diagnostic",
    )
    fig.savefig(output, dpi=150, metadata={"Software": "Stocker research"})
    plt.close(fig)


def feature_manifest() -> dict[str, Any]:
    """Return the preregistered, not-materialised feature declaration."""

    return {
        **SAFETY_FLAGS,
        "status": "not_materialised_due_to_history_blocker",
        "predictor_window": "completed_minutes_minus_10_through_minus_1",
        "price_sequence_features": list(PRICE_FEATURES),
        "raw_activity_sequence_features": list(ACTIVITY_FEATURES),
        "activity_price_response_interactions": list(INTERACTION_FEATURES),
        "provider_volume_label": "historical_activity_proxy",
        "signed_activity_proxy_label": "bar_sign_weighted_activity_proxy",
        "activity_lead_clip": [-9, 9],
        "assessment_information_used_in_features": False,
        "future_information_used_in_features": False,
        "features_materialised": 0,
    }


def forbidden_feature_audit() -> dict[str, Any]:
    """Audit the preregistered predictor names before any feature is materialised."""

    feature_names = [*PRICE_FEATURES, *ACTIVITY_FEATURES, *INTERACTION_FEATURES]
    violations = forbidden_feature_names(feature_names)
    return {
        **SAFETY_FLAGS,
        "passed": not violations,
        "status": "names_only_history_gate_failed_before_materialisation",
        "predictor_names_checked": feature_names,
        "forbidden_tokens_checked": list(FORBIDDEN_FEATURE_TOKENS),
        "violations": violations,
        "materialised_predictor_columns_checked": 0,
    }


def input_hash_manifest(
    contract: Mapping[str, Any], source_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Record immutable inputs read before the availability stop."""

    paths = (
        CONTRACT_PATH,
        PREDECESSOR_PANEL,
        PREDECESSOR_SOURCE_MANIFEST,
        PREDECESSOR_DECISION,
        PREDECESSOR_THRESHOLDS,
        PREDECESSOR_BOUNDARY,
    )
    safe_source_hashes = [
        {
            "logical_path": source["logical_path"],
            "bounded_safe_rows_materialised": source["bounded_safe_rows_materialised"],
            "bounded_safe_timestamp_sha256": source["bounded_safe_timestamp_sha256"],
        }
        for source in source_manifest["sources"]
        if source["source_file_present"] and source["source_read_error_code"] is None
    ]
    return {
        **SAFETY_FLAGS,
        "contract_id": contract["contract_id"],
        "artifacts": [
            {
                "logical_path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
        "one_minute_source_artifacts_hashed": len(safe_source_hashes),
        "one_minute_safe_timestamp_hashes": safe_source_hashes,
    }


def empty_model_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    """Declare the fixed ladder while proving that zero models were fitted."""

    configuration = {
        **SAFETY_FLAGS,
        "status": "not_fitted_due_to_history_blocker",
        "fixed_model": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 250,
            "class_weight": None,
            "n_jobs": 1,
        },
        "requested_models": ["A0", "A1", "A2", "A3", "D0", "D1", "D2", "D3"],
        "fitted_models": [],
        "fitted_model_count": 0,
        "feature_or_hyperparameter_search": False,
    }
    coefficients = {
        **SAFETY_FLAGS,
        "status": "not_fitted_due_to_history_blocker",
        "models": {},
        "fitted_model_count": 0,
    }
    return configuration, coefficients


def report_text(reconstruction: Mapping[str, Any], source_manifest: Mapping[str, Any]) -> str:
    """Render the fail-closed scientific report."""

    development_rows = reconstruction["development_admitted_rows"]
    assessment_rows = reconstruction["assessment_admitted_rows"]
    assessment_sessions = reconstruction["assessment_sessions"]
    assessment_stocks = reconstruction["assessment_stocks"]
    return f"""# One-Minute Activity–Price Lead Screen V0

**Decision:** `{HISTORY_BLOCKER}`

This retrospective, research-only, observable-only feasibility screen stopped at the
first one-minute data gate. It is not prospective validation, achieved P&L, a
strategy, or evidence of executable edge.

## Availability gate

- Frozen cohort: `{len(SYMBOLS)}` stocks.
- Required XNYS sessions: `{source_manifest["calendar_sessions"]}`.
- Symbol-session audit rows: `{source_manifest["availability_rows"]}`.
- Complete symbol-sessions: `{source_manifest["complete_symbol_sessions"]}`.
- Missing symbol-sessions: `{source_manifest["missing_symbol_sessions"]}`.
- Local one-minute files present: `{source_manifest["sources_present"]}`.
- Safe one-minute timestamp rows materialised: `{source_manifest["one_minute_rows_materialised"]}`.
- Minimum safe one-minute timestamp: `{source_manifest["minimum_one_minute_timestamp_read"]}`.
- Maximum safe one-minute timestamp: `{source_manifest["maximum_one_minute_timestamp_read"]}`.
- Protected rows opened: `0`.
- External data downloaded: `False`.
- External API called: `False`.
- Credentials read: `False`.

Every session is reported in `one_minute_availability_audit.csv` with its symbol,
month, XNYS open/close, separate bar-start and bar-end candidate ordinals, exact
missing ordinals, duplicates, off-grid rows, source identity, and QA status. Neither
candidate is promoted to a timestamp convention without empirical proof.

## Frozen nomination population

- Source: High-Movement Pressure-Onset Screen V0.1 at `cda387c`.
- Frozen development admitted rows: `{development_rows}`.
- Frozen assessment rows / sessions / stocks:
  `{assessment_rows}` / `{assessment_sessions}` / `{assessment_stocks}`.
- Assessment checkpoint rows: `{reconstruction["assessment_checkpoint_rows"]}`.
- Admission rule recomputed: `False`.
- Exact frozen identity reconstruction: `True`.

## Downstream work not opened

Timestamp semantics could not be empirically proved because complete one-minute
history did not pass the availability gate. No one-minute normalisation,
price/activity feature, interaction,
onset barrier, label, model, bootstrap, activity null, permutation importance,
economic reference, concentration selection, or plot was produced. Zero models were
fitted. This is the required fail-closed behavior; five-minute volume was not used as
a substitute.
"""


def execute_blocked_run(output: Path, *, provider_root: Path) -> dict[str, Any]:
    """Emit one deterministic blocked run."""

    contract = verify_contract()
    output.mkdir(parents=True, exist_ok=True)
    compact, reconstruction = reconstruct_frozen_population()
    availability, source_manifest = build_availability_audit(provider_root)
    if source_manifest["availability_gate_passed"]:
        raise RuntimeError("blocked_reproducibility_or_audit_failure")

    write_json(output / "contract.json", contract)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(
        output / "input_artifact_hashes.json",
        input_hash_manifest(contract, source_manifest),
    )
    write_csv(output / "one_minute_availability_audit.csv", availability)
    write_json(
        output / "timestamp_semantics_audit.json",
        {
            **SAFETY_FLAGS,
            "passed": False,
            "status": "not_evaluated_history_unavailable",
            "timestamp_convention": None,
            "bar_start_or_end_proved": False,
            "session_relative_ordinal_mapping_performed": False,
            "availability_timestamp_label_ordinals_mapped": True,
            "causal_window_materialised": False,
            "reason": "complete local one-minute history failed the availability gate",
            "decision_precedence": HISTORY_BLOCKER,
        },
    )
    write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "passed": True,
            "protected_start": "2025-08-23",
            "one_minute_source_files_opened": source_manifest["sources_present"],
            "one_minute_rows_materialised": source_manifest["one_minute_rows_materialised"],
            "protected_rows_opened": 0,
            "protected_files_touched": [],
            "source_columns_materialised": ["timestamp"],
            "parquet_predicate_maximum_exclusive": PROTECTED_START.isoformat(),
            "minimum_one_minute_timestamp_read": source_manifest[
                "minimum_one_minute_timestamp_read"
            ],
            "maximum_one_minute_timestamp_read": source_manifest[
                "maximum_one_minute_timestamp_read"
            ],
            "frozen_predecessor_maximum_timestamp": "2025-08-22T20:00:00+00:00",
        },
    )
    write_json(output / "frozen_population_reconstruction.json", reconstruction)
    write_json(
        output / "normalisation_manifest.json",
        {
            **SAFETY_FLAGS,
            "status": "not_fitted_due_to_history_blocker",
            "provider_volume_label": "historical_activity_proxy",
            "formula": "volume / trailing_same_stock_same_minute_median_volume",
            "minimum_prior_observations": 20,
            "log_score": "log1p(relative_activity)",
            "development_only": True,
            "future_sessions_used": False,
            "assessment_outcomes_used": False,
            "normalisation_rows_fitted": 0,
        },
    )
    write_json(
        output / "onset_barriers.json",
        {
            **SAFETY_FLAGS,
            "status": "not_calculated_due_to_history_blocker",
            "quantile": 0.75,
            "training_period": "2024",
            "barriers_bps": {"6": None, "12": None},
            "development_rows_used": 0,
        },
    )
    write_json(output / "feature_manifest.json", feature_manifest())
    write_json(output / "forbidden_feature_audit.json", forbidden_feature_audit())
    write_parquet(output / "compact_decision_panel.parquet", compact)
    write_parquet(
        output / "one_minute_sequence_ledger.parquet",
        pd.DataFrame(
            columns=[
                "symbol",
                "session",
                "decision_ordinal_5m",
                "relative_minute",
                "minute_of_session_ordinal",
                "timestamp_utc",
                "timestamp_america_new_york",
                "source_file_identity",
                "qa_status",
            ]
        ),
    )
    write_parquet(
        output / "onset_path_ledger.parquet",
        pd.DataFrame(
            columns=[
                "symbol",
                "session",
                "decision_ordinal_5m",
                "relative_minute",
                "cumulative_residual_return_bps",
                "onset_label",
            ]
        ),
    )
    model_configurations, model_coefficients = empty_model_artifacts()
    write_json(output / "model_configurations.json", model_configurations)
    write_json(output / "model_coefficients.json", model_coefficients)
    write_parquet(
        output / "assessment_predictions.parquet",
        pd.DataFrame(
            columns=[
                "symbol",
                "session",
                "decision_ordinal",
                "model",
                "target",
                "probability",
            ]
        ),
    )
    for filename, columns in CSV_SCHEMAS.items():
        write_csv(output / filename, pd.DataFrame(columns=list(columns)))

    decision = {
        **SAFETY_FLAGS,
        "decision": HISTORY_BLOCKER,
        "reason": "complete local one-minute history is unavailable for the frozen population",
        "decision_gate": "one_minute_data_availability",
        "frozen_population_reconstructed": True,
        "frozen_assessment_rows": reconstruction["assessment_admitted_rows"],
        "frozen_assessment_sessions": reconstruction["assessment_sessions"],
        "frozen_assessment_stocks": reconstruction["assessment_stocks"],
        "required_symbol_sessions": source_manifest["availability_rows"],
        "complete_symbol_sessions": source_manifest["complete_symbol_sessions"],
        "one_minute_rows_materialised": source_manifest["one_minute_rows_materialised"],
        "timestamp_semantics_proved": False,
        "models_fitted": 0,
        "bootstrap_draws_run": 0,
        "null_draws_run": 0,
        "plots_created": 0,
        "protected_rows_opened": 0,
        "external_data_downloaded": False,
        "external_api_called": False,
        "credentials_read": False,
        "five_minute_substitution_used": False,
        "probability_gates_evaluated": False,
        "economic_reference_evaluated": False,
    }
    write_json(output / "decision.json", decision)
    report = report_text(reconstruction, source_manifest)
    (output / "report.md").write_text(report, encoding="utf-8")
    write_json(
        output / "independent_audit.json",
        {
            **SAFETY_FLAGS,
            "passed": False,
            "status": "pending_independent_audit",
            "auditor_imported_runner": False,
        },
    )
    return {
        "decision": HISTORY_BLOCKER,
        "frozen_identity_sha256": reconstruction["identity_sha256"],
        "availability_rows": source_manifest["availability_rows"],
        "models_fitted": 0,
        "protected_rows_opened": 0,
    }


def full_input_hash_manifest(
    contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    bars: pd.DataFrame,
    provider_root: Path,
) -> dict[str, Any]:
    paths = (
        CONTRACT_PATH,
        PREDECESSOR_PANEL,
        PREDECESSOR_SOURCE_MANIFEST,
        PREDECESSOR_DECISION,
        PREDECESSOR_THRESHOLDS,
        PREDECESSOR_BOUNDARY,
    )
    source_hashes: list[dict[str, Any]] = []
    for symbol, group in bars.groupby("symbol", sort=True):
        canonical = group.loc[
            :,
            [
                "timestamp_utc",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "minute_of_session_ordinal",
            ],
        ].copy()
        canonical["timestamp_utc"] = canonical["timestamp_utc"].astype(str)
        source_hashes.append(
            {
                "symbol": str(symbol),
                "one_minute_logical_path": (
                    f"source=eodhd/instrument_type=stock/symbol={symbol}/timeframe=1m/data.parquet"
                ),
                "safe_relevant_rows": len(group),
                "safe_relevant_ohlcv_sha256": sha256_text(
                    canonical.to_csv(index=False, lineterminator="\n", float_format="%.17g")
                ),
                "safe_timestamp_sha256": next(
                    source["bounded_safe_timestamp_sha256"]
                    for source in source_manifest["sources"]
                    if source["symbol"] == symbol
                ),
                "five_minute_anchor_logical_path": (
                    f"source=eodhd/instrument_type=stock/symbol={symbol}/timeframe=5m/data.parquet"
                ),
                "five_minute_anchor_sha256": sha256_file(
                    five_minute_provider_path(provider_root, str(symbol))
                ),
            }
        )
    return {
        **SAFETY_FLAGS,
        "contract_id": contract["contract_id"],
        "artifacts": [
            {"logical_path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
            for path in paths
        ],
        "one_minute_source_artifacts_hashed": len(source_hashes),
        "scientifically_relevant_safe_source_hashes": source_hashes,
        "protected_rows_hashed": 0,
    }


def full_report_text(
    decision: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    semantics: Mapping[str, Any],
    support: Mapping[str, Any],
    barriers: Mapping[int, float],
    onset: pd.DataFrame,
    direction: pd.DataFrame,
    bootstrap: pd.DataFrame,
    nulls: pd.DataFrame,
    economic: pd.DataFrame,
    concentration: Mapping[str, Any],
) -> str:
    def model_lines(frame: pd.DataFrame, stage: str) -> str:
        pooled = frame.loc[frame["scope_type"].eq("pooled") & frame["scope_value"].eq("all")]
        return "\n".join(
            f"- {row.model}: Brier `{row.brier_score:.6f}`, log loss `{row.log_loss:.6f}`, "
            f"AUC `{row.auc:.6f}`."
            for row in pooled.itertuples(index=False)
            if row.stage == stage
        )

    increment_lines = "\n".join(
        f"- {name}: Brier `{value['brier_improvement']:.6g}`, log loss "
        f"`{value['log_loss_improvement']:.6g}`, passes `{value['passes']}`."
        for name, value in decision.get("increments", {}).items()
    )
    bootstrap_summary = bootstrap.loc[bootstrap["record_type"].eq("summary")]
    bootstrap_lines = "\n".join(
        f"- {row.comparison} / {row.metric}: 90% "
        f"`[{row.lower_90:.6g}, {row.upper_90:.6g}]`, 95% "
        f"`[{row.lower_95:.6g}, {row.upper_95:.6g}]`."
        for row in bootstrap_summary.itertuples(index=False)
    )
    null_summary = nulls.loc[nulls["record_type"].eq("summary")]
    null_lines = "\n".join(
        f"- {row.comparison}: real percentile `{row.real_percentile:.3f}`, null q90 "
        f"`{row.null_q90:.6g}`."
        for row in null_summary.itertuples(index=False)
    )
    economic_pooled = economic.loc[
        economic["scope_type"].eq("pooled")
        & economic["friction_bps"].eq(20.0)
        & economic["candidate"].isin(["price_system", "activity_system", "interaction_system"])
    ]
    economic_lines = "\n".join(
        f"- {row.candidate}, {row.horizon}: `{row.mean_signed_return_bps:.3f}` bps "
        "after 20 bps synthetic friction."
        for row in economic_pooled.itertuples(index=False)
    )
    return f"""# One-Minute Activity–Price Lead Screen V0

**Decision:** `{decision["decision"]}`

This is a retrospective, research-only, observable-only bounded feasibility screen.
It is not prospective validation, achieved P&L, a deployable strategy, or evidence
of executable edge.

## Inputs and chronology

- Frozen predecessor: High-Movement Pressure-Onset Screen V0.1 at `cda387c`.
- Frozen admitted development rows: `{reconstruction["development_admitted_rows"]}`.
- Frozen admitted assessment rows: `{reconstruction["assessment_admitted_rows"]}`.
- Safe local one-minute timestamps read: `{source_manifest["one_minute_rows_materialised"]}`.
- Safe timestamp range: `{source_manifest["minimum_one_minute_timestamp_read"]}` through
  `{source_manifest["maximum_one_minute_timestamp_read"]}`.
- Timestamp convention: `{semantics["timestamp_convention"]}`, proved by local 1m-to-5m
  OHLC alignment.
- Protected rows opened: `0`.
- Predictor window: ten fully completed bars, minute -10 through minute -1.
- Entry: open of minute +2; onset closes: +2 through +6; terminals: +16 and +31.

## Support and barriers

- Analysed development rows: `{support["development_rows"]}`.
- Analysed assessment rows / sessions / stocks / months:
  `{support["assessment_rows"]}` / `{support["assessment_sessions"]}` /
  `{support["assessment_stocks"]}` / `{support["assessment_months"]}`.
- Assessment UP / DOWN / NO_ONSET: `{support["up_onsets"]}` /
  `{support["down_onsets"]}` / `{support["no_onsets"]}`.
- Onset barriers: checkpoint 30m `{barriers[6]:.6f}` bps; checkpoint 60m
  `{barriers[12]:.6f}` bps.

## Onset ladder

{model_lines(onset, "onset")}

## Conditional-direction ladder

{model_lines(direction, "direction")}

## Fixed increments

{increment_lines}

## Session-block bootstrap

{bootstrap_lines}

## Within-slate activity null

{null_lines}

## Delayed economic-reference diagnostic

{economic_lines}

The economic values are synthetic-friction diagnostics, not achieved P&L. They
cannot rescue a failed probability gate.

## Concentration

- Maximum assessment row share: `{concentration["maximum_assessment_row_share"]:.4f}`.
- All economic-selection concentration gates pass:
  `{concentration["all_selection_concentration_gates_pass"]}`.
"""


def execute_full_run(output: Path, *, provider_root: Path) -> dict[str, Any]:
    """Execute the bounded full screen from local frozen inputs."""

    contract = verify_contract()
    output.mkdir(parents=True, exist_ok=True)
    frozen, reconstruction = reconstruct_frozen_population()
    availability, source_manifest = build_availability_audit(provider_root)
    semantics = prove_local_timestamp_semantics(provider_root)
    parent_sessions = set(frozen["session"].astype(str))
    bars, qa_records = load_relevant_one_minute_bars(provider_root, parent_sessions)
    bars, normalisation = causal_activity_normalisation(bars)
    valid_parents, exact_coverage = build_valid_parent_population(frozen, bars)
    if (
        exact_coverage["assessment_admitted_rows_with_exact_windows"] < 1_200
        or exact_coverage["assessment_sessions"] < 100
        or exact_coverage["assessment_stocks"] < 15
    ):
        raise RuntimeError("blocked_one_minute_history_unavailable")
    features, sequence_ledger, feature_info = build_feature_panel(valid_parents, bars)
    with_outcomes, onset_paths, barriers = build_outcomes(features, bars)
    analysis = with_outcomes.loc[with_outcomes["high_movement_admitted"]].copy()
    analysis["analysis_admitted_rows_in_slate"] = analysis.groupby("parent_slate_id", sort=False)[
        "parent_slate_id"
    ].transform("size")
    analysis["row_weight"] = 1.0 / analysis["analysis_admitted_rows_in_slate"]
    support = support_summary(analysis)
    if not support["aggregate_support_passes"]:
        raise RuntimeError("blocked_insufficient_one_minute_support")
    scored, models, predictions = fit_and_score_ladder(
        analysis,
        conditional_direction_supported=bool(support["conditional_direction_support_passes"]),
    )
    onset, direction, checkpoint, monthly, calibration = evaluate_ladder(
        scored,
        conditional_direction_supported=bool(support["conditional_direction_support_passes"]),
    )
    if not support["conditional_direction_support_passes"]:
        selections = pd.DataFrame()
        economic = pd.DataFrame(columns=list(CSV_SCHEMAS["economic_reference_metrics.csv"]))
        concentration_frame = pd.DataFrame(columns=list(CSV_SCHEMAS["concentration_metrics.csv"]))
        concentration = {
            "maximum_assessment_row_share": support["maximum_stock_row_share"],
            "assessment_row_concentration_passes": True,
            "maximum_selection_share_by_candidate": {},
            "all_selection_concentration_gates_pass": True,
        }
    else:
        selections = build_economic_selections(scored)
        economic = economic_metrics(selections)
        concentration_frame, concentration = concentration_metrics(scored, selections)
    bootstrap = bootstrap_metrics(
        scored,
        selections,
        direction_supported=bool(support["conditional_direction_support_passes"]),
    )
    nulls = within_slate_activity_null(
        analysis,
        scored,
        selections,
        direction_supported=bool(support["conditional_direction_support_passes"]),
    )
    diagnostics = feature_group_diagnostics(scored, models)
    decision = screen_decision(
        onset,
        direction,
        monthly,
        checkpoint,
        bootstrap,
        nulls,
        support,
        concentration,
    )
    forbidden_names = sorted({name for names in MODEL_FEATURES.values() for name in names})
    violations = forbidden_feature_names(forbidden_names)
    if violations:
        raise RuntimeError("blocked_chronology_or_leakage_failure")

    frozen_output = frozen.rename(columns={"row_weight": "frozen_predecessor_row_weight"})
    analysis_keys = ["symbol", "session", "decision_ordinal"]
    analysis["analysis_eligible"] = True
    new_columns = [
        column
        for column in analysis.columns
        if column not in frozen_output.columns or column in analysis_keys
    ]
    compact = frozen_output.merge(
        analysis.loc[:, new_columns],
        on=analysis_keys,
        how="left",
        validate="one_to_one",
    )
    compact["analysis_eligible"] = compact["analysis_eligible"].eq(True)
    compact["availability_gate_passed"] = compact["analysis_eligible"]
    compact["one_minute_source_status"] = np.where(
        compact["analysis_eligible"], "exact_required_minutes_available", "required_minutes_missing"
    )
    compact = compact.sort_values(
        ["session", "decision_ordinal", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    if len(compact) > MAX_COMPACT_ROWS or len(predictions) > MAX_COMPACT_ROWS:
        raise RuntimeError("blocked_quick_activity_screen_resource_limit")

    availability["proved_timestamp_convention"] = "bar_start"
    source_manifest = dict(source_manifest)
    source_manifest.update(
        {
            "availability_gate_passed": True,
            "availability_gate_definition": (
                "exact causal predictor and delayed outcome ordinals with at least 15 valid "
                "stocks per admitted parent slate"
            ),
            "full_regular_session_completeness_required": False,
            "exact_required_window_coverage": exact_coverage,
            "timestamp_convention": "bar_start",
            "vendor_qa": qa_records,
            "external_data_downloaded": True,
            "input_acquisition_authorized_by_user": True,
            "input_acquisition_performed_before_runner": True,
            "external_api_called": False,
            "credentials_read": False,
            "runner_network_access": False,
            "download_request_range": {
                "from_inclusive": "2024-01-01",
                "to_exclusive": "2025-08-23",
            },
        }
    )
    model_configurations = {
        **SAFETY_FLAGS,
        "status": "fitted",
        "fixed_model": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 250,
            "class_weight": None,
            "n_jobs": 1,
        },
        "model_features": {key: list(value) for key, value in MODEL_FEATURES.items()},
        "requested_models": ["A0", "A1", "A2", "A3", "D0", "D1", "D2", "D3"],
        "fitted_models": sorted(models),
        "fitted_model_count": len(models),
        "feature_or_hyperparameter_search": False,
        "preprocessing_fit_period": "2024_only",
        "assessment_period": "2025-01-01_through_2025-08-22",
    }
    coefficient_artifact = {
        **SAFETY_FLAGS,
        "status": "fitted",
        "models": models,
        "fitted_model_count": len(models),
    }
    report = full_report_text(
        decision,
        reconstruction,
        source_manifest,
        semantics,
        support,
        barriers,
        onset,
        direction,
        bootstrap,
        nulls,
        economic,
        concentration,
    )
    write_json(output / "contract.json", contract)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(
        output / "input_artifact_hashes.json",
        full_input_hash_manifest(contract, source_manifest, bars, provider_root),
    )
    write_csv(output / "one_minute_availability_audit.csv", availability)
    write_json(output / "timestamp_semantics_audit.json", semantics)
    write_json(
        output / "protected_boundary_audit.json",
        {
            **SAFETY_FLAGS,
            "passed": True,
            "protected_start": "2025-08-23",
            "source_predicate_maximum_exclusive": PROTECTED_START.isoformat(),
            "parquet_predicate_maximum_exclusive": PROTECTED_START.isoformat(),
            "one_minute_source_files_opened": source_manifest["sources_present"],
            "one_minute_rows_materialised": source_manifest["one_minute_rows_materialised"],
            "minimum_one_minute_timestamp_read": source_manifest[
                "minimum_one_minute_timestamp_read"
            ],
            "maximum_one_minute_timestamp_read": source_manifest[
                "maximum_one_minute_timestamp_read"
            ],
            "dates_actually_read": ["2024-01-02", "2025-08-22"],
            "protected_rows_opened": 0,
            "protected_files_touched": [],
            "frozen_predecessor_maximum_timestamp": "2025-08-22T20:00:00+00:00",
        },
    )
    write_json(output / "frozen_population_reconstruction.json", reconstruction)
    write_json(output / "normalisation_manifest.json", normalisation)
    write_json(
        output / "onset_barriers.json",
        {
            **SAFETY_FLAGS,
            "status": "frozen_from_2024_development",
            "quantile": 0.75,
            "training_period": "2024",
            "barriers_bps": {str(key): value for key, value in barriers.items()},
            "development_rows_used": support["development_rows"],
        },
    )
    write_json(output / "feature_manifest.json", feature_info)
    write_json(
        output / "forbidden_feature_audit.json",
        {
            **SAFETY_FLAGS,
            "passed": not violations,
            "status": "all_materialised_model_predictor_names_checked",
            "predictor_names_checked": forbidden_names,
            "forbidden_tokens_checked": list(FORBIDDEN_FEATURE_TOKENS),
            "violations": violations,
            "materialised_predictor_columns_checked": len(forbidden_names),
            "symbol_or_month_predictor_used": False,
        },
    )
    write_parquet(output / "compact_decision_panel.parquet", compact)
    write_parquet(output / "one_minute_sequence_ledger.parquet", sequence_ledger)
    write_parquet(
        output / "onset_path_ledger.parquet",
        onset_paths.loc[onset_paths["high_movement_admitted"]].reset_index(drop=True),
    )
    write_json(output / "model_configurations.json", model_configurations)
    write_json(output / "model_coefficients.json", coefficient_artifact)
    write_parquet(output / "assessment_predictions.parquet", predictions)
    write_csv(output / "onset_metrics.csv", onset)
    write_csv(output / "direction_metrics.csv", direction)
    write_csv(output / "checkpoint_metrics.csv", checkpoint)
    write_csv(output / "monthly_metrics.csv", monthly)
    write_csv(output / "calibration_bins.csv", calibration)
    write_csv(output / "feature_group_diagnostics.csv", diagnostics)
    write_csv(output / "bootstrap_metrics.csv", bootstrap)
    write_csv(output / "null_metrics.csv", nulls)
    write_csv(output / "economic_reference_metrics.csv", economic)
    write_csv(output / "concentration_metrics.csv", concentration_frame)
    write_json(output / "decision.json", decision)
    (output / "report.md").write_text(report, encoding="utf-8")
    plot_calibration(calibration, stage="onset", output=output / "onset_calibration.png")
    if support["conditional_direction_support_passes"]:
        plot_calibration(
            calibration, stage="direction", output=output / "direction_calibration.png"
        )
        plot_economic_comparison(economic, output / "delayed_return_comparison.png")
    write_json(
        output / "independent_audit.json",
        {
            **SAFETY_FLAGS,
            "passed": False,
            "status": "pending_independent_audit",
            "auditor_imported_runner": False,
        },
    )
    return {
        "decision": decision["decision"],
        "frozen_identity_sha256": reconstruction["identity_sha256"],
        "availability_rows": source_manifest["availability_rows"],
        "models_fitted": len(models),
        "protected_rows_opened": 0,
    }


def execute_run(output: Path, *, provider_root: Path) -> dict[str, Any]:
    """Route to the history blocker or the full local-data screen."""

    present = sum(provider_path(provider_root, symbol).is_file() for symbol in SYMBOLS)
    if present < len(SYMBOLS):
        return execute_blocked_run(output, provider_root=provider_root)
    return execute_full_run(output, provider_root=provider_root)


def compared_artifact_names() -> list[str]:
    """Return deterministic artifacts compared before the independent audit."""

    return sorted(
        [
            "contract.json",
            "source_manifest.json",
            "input_artifact_hashes.json",
            "one_minute_availability_audit.csv",
            "timestamp_semantics_audit.json",
            "protected_boundary_audit.json",
            "frozen_population_reconstruction.json",
            "normalisation_manifest.json",
            "onset_barriers.json",
            "feature_manifest.json",
            "forbidden_feature_audit.json",
            "compact_decision_panel.parquet",
            "one_minute_sequence_ledger.parquet",
            "onset_path_ledger.parquet",
            "model_configurations.json",
            "model_coefficients.json",
            "assessment_predictions.parquet",
            *CSV_SCHEMAS.keys(),
            "decision.json",
            "report.md",
        ]
    )


def compare_exact_runs(primary: Path, exact: Path) -> dict[str, Any]:
    """Compare every scientific artifact by byte hash."""

    comparisons: list[dict[str, Any]] = []
    names = compared_artifact_names()
    names.extend(
        name
        for name in (
            "onset_calibration.png",
            "direction_calibration.png",
            "delayed_return_comparison.png",
        )
        if (primary / name).is_file() or (exact / name).is_file()
    )
    for name in sorted(names):
        if not (primary / name).is_file() or not (exact / name).is_file():
            raise RuntimeError("blocked_reproducibility_or_audit_failure")
        primary_hash = sha256_file(primary / name)
        exact_hash = sha256_file(exact / name)
        comparisons.append(
            {
                "artifact": name,
                "comparison_mode": "byte_hash",
                "primary_sha256": primary_hash,
                "exact_rerun_sha256": exact_hash,
                "passed": primary_hash == exact_hash,
            }
        )
    passed = all(bool(row["passed"]) for row in comparisons)
    decision = read_json(primary / "decision.json")
    return {
        **SAFETY_FLAGS,
        "decision": decision["decision"],
        "passed": passed,
        "stable_sorting": True,
        "canonical_json": True,
        "fixed_seeds": {
            "bootstrap": BOOTSTRAP_SEED,
            "null": NULL_SEED,
            "economic_random": ECONOMIC_RANDOM_SEED,
        },
        "models_fitted": int(decision.get("models_fitted", 0)),
        "independent_audit_status": "pending",
        "comparisons": comparisons,
    }


def run_independent_auditor(artifacts: Path, provider_root: Path) -> None:
    """Run the standalone auditor without importing it."""

    result = subprocess.run(
        [
            sys.executable,
            str(AUDITOR_PATH),
            "--artifacts",
            str(artifacts),
            "--provider-root",
            str(provider_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"blocked_reproducibility_or_audit_failure: {detail}")


def parse_args() -> argparse.Namespace:
    """Parse the bounded runner arguments."""

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
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run the primary screen and its deterministic rerun."""

    args = parse_args()
    primary = args.primary_output.resolve()
    exact = args.exact_output.resolve()
    first = execute_run(primary, provider_root=args.provider_root)
    second = execute_run(exact, provider_root=args.provider_root)
    if first != second:
        raise RuntimeError("blocked_reproducibility_or_audit_failure")
    rerun = compare_exact_runs(primary, exact)
    if not rerun["passed"]:
        raise RuntimeError("blocked_reproducibility_or_audit_failure")
    write_json(primary / "exact_rerun_manifest.json", rerun)
    write_json(exact / "exact_rerun_manifest.json", rerun)
    run_independent_auditor(primary, args.provider_root)
    run_independent_auditor(exact, args.provider_root)
    primary_audit_hash = sha256_file(primary / "independent_audit.json")
    exact_audit_hash = sha256_file(exact / "independent_audit.json")
    if primary_audit_hash != exact_audit_hash:
        raise RuntimeError("blocked_reproducibility_or_audit_failure")
    rerun["independent_audit_status"] = "passed"
    rerun["independent_audit_sha256"] = primary_audit_hash
    rerun["comparisons"].append(
        {
            "artifact": "independent_audit.json",
            "comparison_mode": "byte_hash",
            "primary_sha256": primary_audit_hash,
            "exact_rerun_sha256": exact_audit_hash,
            "passed": True,
        }
    )
    write_json(primary / "exact_rerun_manifest.json", rerun)
    write_json(exact / "exact_rerun_manifest.json", rerun)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(primary / "report.md", args.report_output)
    decision = str(first["decision"])
    print(decision)
    return 2 if decision.startswith("blocked_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
