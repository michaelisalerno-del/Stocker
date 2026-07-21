#!/usr/bin/env python3
"""Run the bounded One-Minute Activity-Price Lead Screen V0."""

# ruff: noqa: E402 -- the repository-local research package path is resolved first.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PACKAGE_SRC = REPO_ROOT / "packages" / "stocker_research" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from stocker_research.one_minute_activity_price_lead_v0 import (
    FORBIDDEN_FEATURE_TOKENS,
    forbidden_feature_names,
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
DECISION = "blocked_one_minute_history_unavailable"
MAX_COMPACT_ROWS = 20_000
EXPECTED_ASSESSMENT_ROWS = 1_560
EXPECTED_ASSESSMENT_SESSIONS = 153
EXPECTED_ASSESSMENT_STOCKS = 20
EXPECTED_THRESHOLDS = {6: 0.302886936850, 12: 0.300349339178}

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

CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "onset_metrics.csv": (
        "status",
        "population",
        "model",
        "brier",
        "log_loss",
        "auc",
        "rows",
        "sessions",
        "stocks",
    ),
    "direction_metrics.csv": (
        "status",
        "population",
        "model",
        "brier",
        "log_loss",
        "auc",
        "rows",
        "sessions",
        "stocks",
    ),
    "checkpoint_metrics.csv": ("status", "stage", "model", "checkpoint", "metric", "value"),
    "monthly_metrics.csv": ("status", "stage", "model", "year_month", "metric", "value"),
    "calibration_bins.csv": (
        "status",
        "stage",
        "model",
        "scope",
        "bin",
        "mean_probability",
        "outcome_rate",
        "rows",
    ),
    "feature_group_diagnostics.csv": (
        "status",
        "stage",
        "model",
        "feature_group",
        "diagnostic",
        "value",
    ),
    "bootstrap_metrics.csv": (
        "status",
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
        "status",
        "comparison",
        "metric",
        "real_value",
        "null_q90",
        "real_percentile",
        "draws",
    ),
    "economic_reference_metrics.csv": (
        "status",
        "candidate",
        "horizon",
        "friction_bps",
        "mean_signed_return_bps",
        "rows",
    ),
    "concentration_metrics.csv": (
        "status",
        "scope",
        "symbol",
        "row_share",
        "selection_share",
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
            session_timestamps = pd.DatetimeIndex(timestamps.loc[local_dates.eq(session_text)])
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

**Decision:** `{DECISION}`

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


def execute_run(output: Path, *, provider_root: Path) -> dict[str, Any]:
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
            "decision_precedence": DECISION,
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
        "decision": DECISION,
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
        "decision": DECISION,
        "frozen_identity_sha256": reconstruction["identity_sha256"],
        "availability_rows": source_manifest["availability_rows"],
        "models_fitted": 0,
        "protected_rows_opened": 0,
    }


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
    """Compare every scientific blocker artifact by byte hash."""

    comparisons: list[dict[str, Any]] = []
    for name in compared_artifact_names():
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
    return {
        **SAFETY_FLAGS,
        "decision": DECISION,
        "passed": passed,
        "stable_sorting": True,
        "canonical_json": True,
        "fixed_seeds": {"bootstrap": 20260720, "null": 20260721, "economic_random": 20260722},
        "models_fitted": 0,
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
    """Run the primary blocker screen and its deterministic rerun."""

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
    print(DECISION)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
