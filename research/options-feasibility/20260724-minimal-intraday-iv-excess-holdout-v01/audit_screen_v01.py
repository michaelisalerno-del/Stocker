#!/usr/bin/env python3
"""Independently audit the strict V0.1 resume and frozen holdout artifacts."""

from __future__ import annotations

# ruff: noqa: E402 -- deterministic numerical limits precede imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
V0_EXPERIMENT_DIR = (
    REPO_ROOT / "research/options-feasibility/20260723-minimal-intraday-iv-excess-holdout-v0"
)
V0_PRIMARY = V0_EXPERIMENT_DIR / "artifacts" / "primary"
V0_AUDITOR = V0_EXPERIMENT_DIR / "audit_screen_v0.py"
V0_RUNNER = V0_EXPERIMENT_DIR / "run_screen_v0.py"
V0_DOWNLOADER = V0_EXPERIMENT_DIR / "download_holdout_options.py"
V01_DOWNLOADER = EXPERIMENT_DIR / "download_holdout_options.py"
V0_REQUEST_PLAN = V0_PRIMARY / "holdout_options_request_plan.json"
PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
V0_CACHE_ROOT = REPO_ROOT / "data/vendor/eodhd/options/minimal-intraday-iv-excess-holdout-v0"
RESUME_CACHE_ROOT = REPO_ROOT / "data/vendor/eodhd/options/minimal-intraday-iv-excess-holdout-v01"
OPTIONS_CACHE = RESUME_CACHE_ROOT / "canonical/exact_holdout_options.parquet"
STOCK_CACHE = (
    REPO_ROOT
    / "data/cache/minimal-intraday-iv-excess-holdout-v0"
    / "frozen_h0_stock_surface.parquet"
)
STATE_CACHE = STOCK_CACHE.with_name("frozen_state_surface.parquet")

for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.minimal_intraday_iv_excess_holdout_v01 import (
    EXPECTED_FROZEN_REQUESTS,
    EXPECTED_V0_COMPLETE_RECEIPTS,
    MAXIMUM_ADDITIONAL_BYTES,
    MAXIMUM_ADDITIONAL_RECORDS,
    MAXIMUM_CUMULATIVE_RECORDS,
    SAFETY_FLAGS,
    add_movement_outcomes_with_optional_30m,
    assert_v01_safety_flags,
    attach_movement_prices_with_optional_30m,
    coverage_preflight,
    identify_interrupted_request,
    inventory_complete_receipts,
    movement_timing_metrics_with_optional_30m,
    remaining_resume_requests,
)


def load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ValueError(f"cannot load audit dependency: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_nullable_difference(left: Sequence[float], right: Sequence[float]) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape:
        return math.inf
    both_nan = np.isnan(first) & np.isnan(second)
    finite = np.isfinite(first) & np.isfinite(second)
    if bool((~both_nan & ~finite).any()):
        return math.inf
    return float(np.max(np.abs(first[finite] - second[finite]))) if bool(finite.any()) else 0.0


def unreceipted_raw_paths(cache_root: Path) -> list[Path]:
    referenced: set[Path] = set()
    for receipt in (cache_root / "manifests/completed").glob("*.json"):
        stored = read_json(receipt)
        for row in cast(list[dict[str, Any]], stored["manifest_rows"]):
            referenced.add(Path(str(row["cache_path"])).resolve())
    all_raw = {path.resolve() for path in (cache_root / "raw").glob("*.json")}
    return sorted(all_raw.difference(referenced))


def rebuild_plan(downloader: ModuleType) -> list[dict[str, object]]:
    rebuilt = [
        cast(dict[str, object], asdict(row))
        for row in cast(Sequence[Any], downloader.build_request_plan(PROVIDER_ROOT))
    ]
    frozen = cast(list[dict[str, object]], read_json(V0_REQUEST_PLAN)["requests"])
    if rebuilt != frozen or len(rebuilt) != EXPECTED_FROZEN_REQUESTS:
        raise ValueError("independent request-plan reconstruction differs from V0")
    return rebuilt


def audit_resume_scope(plan: Sequence[Mapping[str, object]]) -> dict[str, Any]:
    v0_inventory = inventory_complete_receipts(
        plan,
        cache_roots=[V0_CACHE_ROOT],
        canonical_cache_path=OPTIONS_CACHE,
    )
    resume_inventory = inventory_complete_receipts(
        plan,
        cache_roots=[RESUME_CACHE_ROOT],
        canonical_cache_path=OPTIONS_CACHE,
    )
    overlap = v0_inventory.verified_request_ids.intersection(resume_inventory.verified_request_ids)
    union = v0_inventory.verified_request_ids.union(resume_inventory.verified_request_ids)
    original_missing = remaining_resume_requests(
        plan,
        v0_inventory.verified_request_ids,
    )
    remaining = remaining_resume_requests(plan, frozenset(union))
    v0_orphans = unreceipted_raw_paths(V0_CACHE_ROOT)
    resume_orphans = unreceipted_raw_paths(RESUME_CACHE_ROOT)
    if len(v0_orphans) != 1:
        raise ValueError("V0 interrupted-page count differs")
    interrupted = identify_interrupted_request(v0_orphans[0], original_missing)
    stored_repair = read_json(PRIMARY / "interrupted_request_repair.json")
    stored_manifest = read_json(PRIMARY / "resume_download_manifest.json")
    stored_reuse = pd.read_csv(PRIMARY / "complete_receipt_reuse_audit.csv")

    new_records = int(
        pd.to_numeric(resume_inventory.audit["records_returned"], errors="raise").sum()
    )
    new_bytes = int(pd.to_numeric(resume_inventory.audit["response_bytes"], errors="raise").sum())
    new_exact = int(
        pd.to_numeric(resume_inventory.audit["exact_date_record_count"], errors="raise").sum()
    )
    new_extra = int(
        pd.to_numeric(resume_inventory.audit["extra_date_rejection_count"], errors="raise").sum()
    )
    new_protected = int(
        pd.to_numeric(
            resume_inventory.audit["protected_date_rejection_count"],
            errors="raise",
        ).sum()
    )
    reuse_hash_match = bool(
        len(stored_reuse) == len(v0_inventory.audit)
        and set(stored_reuse["request_id"].astype(str))
        == set(v0_inventory.audit["request_id"].astype(str))
        and set(stored_reuse["receipt_sha256"].astype(str))
        == set(v0_inventory.audit["receipt_sha256"].astype(str))
    )
    repair_gate = bool(
        stored_repair["request_id"] == interrupted.request_id
        and stored_repair["incomplete_page_identity"] == interrupted.incomplete_page_identity
        and stored_repair["incomplete_page_admitted"] is False
        and stored_repair["resume_method"] == "redownload_logical_request_from_beginning"
        and stored_repair["status"] == "complete"
        and int(
            cast(Mapping[str, Any], stored_repair["deduplication_result"])[
                "incomplete_page_rows_admitted"
            ]
        )
        == 0
    )
    accounting_gate = bool(
        stored_manifest["status"] == "complete"
        and int(stored_manifest["complete_receipts_reused"]) == EXPECTED_V0_COMPLETE_RECEIPTS
        and int(stored_manifest["missing_requests_at_resume_start"]) == len(original_missing) == 250
        and int(stored_manifest["new_requests_completed"])
        == resume_inventory.complete_receipts_reused
        == 250
        and int(stored_manifest["new_provider_records"]) == new_records
        and int(stored_manifest["new_exact_date_records"]) == new_exact
        and int(stored_manifest["new_extra_date_records_rejected"]) == new_extra
        and int(stored_manifest["new_2026_or_later_records_rejected"]) == new_protected
        and int(stored_manifest["new_bytes_downloaded"]) == new_bytes
        and new_records <= MAXIMUM_ADDITIONAL_RECORDS
        and new_bytes <= MAXIMUM_ADDITIONAL_BYTES
        and int(stored_manifest["cumulative_provider_records"]) <= MAXIMUM_CUMULATIVE_RECORDS
        and int(stored_manifest["protected_or_unauthorised_records_materialised"]) == 0
        and int(stored_manifest["requests_remaining"]) == 0
        and sha256_file(OPTIONS_CACHE) == stored_manifest["canonical_cache_sha256"]
    )
    passed = bool(
        v0_inventory.complete_receipts_found == EXPECTED_V0_COMPLETE_RECEIPTS
        and v0_inventory.complete_receipts_reused == EXPECTED_V0_COMPLETE_RECEIPTS
        and v0_inventory.corrupt_receipts == 0
        and resume_inventory.complete_receipts_found == 250
        and resume_inventory.complete_receipts_reused == 250
        and resume_inventory.corrupt_receipts == 0
        and not overlap
        and len(union) == EXPECTED_FROZEN_REQUESTS
        and not remaining
        and not resume_orphans
        and reuse_hash_match
        and repair_gate
        and accounting_gate
    )
    return {
        "v0_complete_receipts_found": v0_inventory.complete_receipts_found,
        "v0_complete_receipts_reused": v0_inventory.complete_receipts_reused,
        "v0_corrupt_receipts": v0_inventory.corrupt_receipts,
        "resume_complete_receipts": resume_inventory.complete_receipts_reused,
        "complete_request_redownload_intersections": len(overlap),
        "original_missing_requests": len(original_missing),
        "requests_remaining": len(remaining),
        "v0_unreceipted_pages": len(v0_orphans),
        "resume_unreceipted_pages": len(resume_orphans),
        "interrupted_request_id": interrupted.request_id,
        "interrupted_page_admitted": False,
        "new_provider_records": new_records,
        "new_exact_date_records": new_exact,
        "new_extra_date_records_rejected": new_extra,
        "new_protected_date_records_rejected": new_protected,
        "new_bytes": new_bytes,
        "reuse_artifact_hash_match": reuse_hash_match,
        "interrupted_repair_gate": repair_gate,
        "download_accounting_gate": accounting_gate,
        "passed": passed,
    }


def audit_canonical_rebuild(
    plan: Sequence[Mapping[str, object]],
    *,
    v0_downloader: ModuleType,
    v01_downloader: ModuleType,
    rebuilt_path: Path,
) -> dict[str, Any]:
    """Rebuild exact-date canonical rows from receipts without network access."""

    v0_inventory = inventory_complete_receipts(
        plan,
        cache_roots=[V0_CACHE_ROOT],
        canonical_cache_path=rebuilt_path,
    )
    resume_inventory = inventory_complete_receipts(
        plan,
        cache_roots=[RESUME_CACHE_ROOT],
        canonical_cache_path=rebuilt_path,
    )
    overlap = set(v0_inventory.receipt_paths).intersection(resume_inventory.receipt_paths)
    receipt_paths = {
        **dict(v0_inventory.receipt_paths),
        **dict(resume_inventory.receipt_paths),
    }
    rebuilt, request_audit, totals = v01_downloader.materialize_complete_canonical_cache(
        plan,
        receipt_paths=receipt_paths,
        canonical_path=rebuilt_path,
        v0=v0_downloader,
    )
    stored = pd.read_parquet(OPTIONS_CACHE)
    column_match = list(rebuilt.columns) == list(stored.columns)
    dtype_match = rebuilt.dtypes.astype(str).tolist() == stored.dtypes.astype(str).tolist()
    content_match = bool(column_match and dtype_match and rebuilt.equals(stored))
    stored_manifest = read_json(PRIMARY / "resume_download_manifest.json")
    accounting_match = bool(
        len(receipt_paths) == EXPECTED_FROZEN_REQUESTS
        and not overlap
        and len(request_audit) == EXPECTED_FROZEN_REQUESTS
        and int(totals["canonical_cache_rows"]) == len(stored)
        and int(totals["provider_records"])
        == int(stored_manifest["provider_records_from_all_complete_receipts"])
        and int(totals["exact_date_records"])
        == int(stored_manifest["exact_date_records_from_all_complete_receipts"])
        and int(totals["extra_date_records_rejected"])
        == int(stored_manifest["extra_date_records_rejected_from_all_complete_receipts"])
        and int(totals["protected_date_records_rejected"])
        == int(stored_manifest["protected_date_records_rejected_from_all_complete_receipts"])
        and int(totals["canonical_cache_rows"]) == int(stored_manifest["exact_date_cache_rows"])
    )
    passed = bool(content_match and accounting_match)
    return {
        "receipts_reloaded": len(receipt_paths),
        "network_requests_issued": 0,
        "rebuilt_provider_records": int(totals["provider_records"]),
        "rebuilt_exact_date_records": int(totals["exact_date_records"]),
        "rebuilt_extra_date_records_rejected": int(totals["extra_date_records_rejected"]),
        "rebuilt_protected_date_records_rejected": int(totals["protected_date_records_rejected"]),
        "rebuilt_canonical_rows": int(totals["canonical_cache_rows"]),
        "stored_canonical_rows": len(stored),
        "canonical_column_match": column_match,
        "canonical_dtype_match": dtype_match,
        "canonical_content_mismatches": 0 if content_match else 1,
        "canonical_accounting_match": accounting_match,
        "rebuilt_canonical_sha256": sha256_file(rebuilt_path),
        "stored_canonical_sha256": sha256_file(OPTIONS_CACHE),
        "passed": passed,
    }


def configure_v0_modules(options_cache: Path) -> tuple[ModuleType, ModuleType]:
    auditor = load_module(V0_AUDITOR, "minimal_holdout_v0_auditor_for_v01")
    runner = load_module(V0_RUNNER, "minimal_holdout_v0_runner_for_v01_audit")
    auditor.PRIMARY = PRIMARY
    auditor.REPORTS = REPORTS
    auditor.OPTIONS_CACHE = options_cache
    auditor.STOCK_CACHE = STOCK_CACHE
    auditor.STATE_CACHE = STATE_CACHE
    auditor.SAFETY_FLAGS = SAFETY_FLAGS
    runner.PRIMARY = PRIMARY
    runner.REPORTS = REPORTS
    runner.SAFETY_FLAGS = SAFETY_FLAGS

    original_decide = auditor.decide_experiment

    def decide_with_v01_support_block(*, model: Any, tail: Any) -> dict[str, Any]:
        decision = cast(dict[str, Any], original_decide(model=model, tail=tail))
        decision.update(SAFETY_FLAGS)
        if model.support_passed and not tail.support_passed:
            decision["overall_decision"] = "blocked_insufficient_frozen_tail_support"
            decision["frozen_top_5pct_status"] = "insufficient_support"
            decision["options_only_tail_comparison_status"] = "insufficient_support"
            decision["movement_timing_status"] = "insufficient_support"
        return decision

    auditor.decide_experiment = decide_with_v01_support_block
    return auditor, runner


def audit_preoutcome_order(
    rebuilt_features: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    planned_stock_sessions: int,
    planned_session_count: int,
) -> dict[str, Any]:
    preflight = read_json(PRIMARY / "holdout_coverage_preflight.json")
    freeze = read_json(PRIMARY / "pre_outcome_freeze_manifest.json")
    authorization = read_json(PRIMARY / "holdout_data_authorisation.json")
    recalculated = coverage_preflight(
        rebuilt_features,
        pairs,
        planned_stock_sessions=planned_stock_sessions,
        planned_session_count=planned_session_count,
        planned_stock_month_cells=80,
    )
    numeric_fields = (
        "total_holdout_stock_sessions",
        "planned_session_count",
        "pair_selected_stock_sessions",
        "expected_joined_rows",
        "expected_session_count",
        "expected_stock_count",
        "expected_month_count",
        "represented_stock_month_cells",
        "maximum_expected_stock_weight_share",
        "maximum_expected_month_weight_share",
    )
    preflight_difference = max(
        abs(float(preflight[name]) - float(recalculated[name])) for name in numeric_fields
    )
    coverage_path = PRIMARY / "holdout_coverage_preflight.json"
    freeze_path = PRIMARY / "pre_outcome_freeze_manifest.json"
    outcome_path = PRIMARY / "holdout_panel.parquet"
    observed_filesystem_order = bool(
        coverage_path.stat().st_mtime_ns
        <= freeze_path.stat().st_mtime_ns
        <= outcome_path.stat().st_mtime_ns
    )
    correction = read_json(PRIMARY / "post_run_reporting_corrections.json")
    prior_audit = read_json(PRIMARY / "lightweight_audit.json")
    prior_order = bool(
        cast(Mapping[str, Any], prior_audit["preoutcome_order_audit"])[
            "filesystem_artifact_order_passed"
        ]
    )
    correction_gate = bool(
        correction["reporting_only"] is True
        and correction["scientific_surfaces_changed"] is False
        and correction["predictions_changed"] is False
        and correction["tail_membership_changed"] is False
        and correction["bootstrap_or_null_repeated"] is False
        and int(preflight["total_holdout_stock_sessions"]) == planned_stock_sessions
        and int(preflight["planned_session_count"]) == planned_session_count
        and int(preflight["pair_selected_stock_sessions"]) == len(pairs)
        and int(authorization["actual_xnys_sessions"]) == planned_session_count
        and int(authorization["joined_outcome_sessions"])
        == int(rebuilt_features["session"].nunique())
    )
    filesystem_order = bool(observed_filesystem_order or (prior_order and correction_gate))
    hashes_gate = bool(
        freeze["feature_manifest_sha256"] == sha256_file(PRIMARY / "minimal_feature_manifest.json")
        and freeze["model_configurations_sha256"]
        == sha256_file(PRIMARY / "model_configurations.json")
        and freeze["model_coefficients_sha256"] == sha256_file(PRIMARY / "model_coefficients.json")
        and freeze["frozen_tail_thresholds_sha256"]
        == sha256_file(PRIMARY / "frozen_tail_thresholds.json")
        and authorization["threshold_artifact_sha256"]
        == sha256_file(PRIMARY / "frozen_tail_thresholds.json")
    )
    passed = bool(
        preflight["passed"] is True
        and preflight["outcome_columns_read"] is False
        and recalculated["passed"] is True
        and preflight_difference <= 1e-12
        and freeze["frozen"] is True
        and freeze["holdout_outcomes_read_before_freeze"] is False
        and freeze["coverage_preflight_passed_before_freeze"] is True
        and authorization["coverage_preflight_passed_before_outcomes"] is True
        and authorization["holdout_outcomes_read_before_freeze"] is False
        and filesystem_order
        and hashes_gate
    )
    return {
        "coverage_preflight_recalculation_maximum_difference": preflight_difference,
        "coverage_preflight_passed": bool(preflight["passed"]),
        "outcome_columns_read_during_preflight": bool(preflight["outcome_columns_read"]),
        "freeze_manifest_frozen": bool(freeze["frozen"]),
        "holdout_outcomes_read_before_freeze": bool(freeze["holdout_outcomes_read_before_freeze"]),
        "filesystem_artifact_order_passed": filesystem_order,
        "filesystem_artifact_order_observed_after_reporting_correction": (
            observed_filesystem_order
        ),
        "prior_filesystem_order_audit_passed": prior_order,
        "post_run_reporting_correction_audited": correction_gate,
        "freeze_hashes_passed": hashes_gate,
        "passed": passed,
    }


def audit_join_and_movement(
    runner: ModuleType,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_parquet(PRIMARY / "holdout_selected_option_pairs.parquet")
    panel = pd.read_parquet(PRIMARY / "holdout_panel.parquet")
    h0 = pd.read_parquet(
        STOCK_CACHE,
        columns=[
            "row_id",
            "symbol",
            "session",
            "period",
            "checkpoint",
            "row_weight",
            *runner.GROUP_I,
        ],
    )
    rebuilt, _audit = runner.join_holdout_panel(h0, pairs)
    rebuilt = rebuilt.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    stored = panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    row_mismatches = abs(len(rebuilt) - len(stored)) + sum(
        first != second
        for first, second in zip(
            rebuilt["row_id"].astype(str),
            stored["row_id"].astype(str),
            strict=False,
        )
    )
    feature_difference = math.inf
    if row_mismatches == 0:
        feature_difference = maximum_nullable_difference(
            rebuilt.loc[:, [*runner.GROUP_O, *runner.GROUP_I, "row_weight"]]
            .to_numpy(float)
            .ravel(),
            stored.loc[:, [*runner.GROUP_O, *runner.GROUP_I, "row_weight"]].to_numpy(float).ravel(),
        )

    states = pd.read_parquet(
        STATE_CACHE,
        columns=["symbol", "session", "bar_ordinal", "open", "close"],
    )
    movement = attach_movement_prices_with_optional_30m(rebuilt, states)
    movement = add_movement_outcomes_with_optional_30m(movement)
    movement = movement.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    movement_difference = 0.0
    movement_columns = [
        "entry_price",
        "close_5m",
        "close_10m",
        "close_15m",
        "close_30m",
        *(
            name
            for horizon in (5, 10, 15, 30)
            for name in (
                f"absolute_log_return_{horizon}m",
                f"iv_expected_absolute_{horizon}m",
                f"iv_absolute_residual_{horizon}m",
                f"movement_exceeds_prior_close_iv_{horizon}m",
            )
        ),
    ]
    for column in movement_columns:
        movement_difference = max(
            movement_difference,
            maximum_nullable_difference(movement[column], stored[column]),
        )
    timing = movement_timing_metrics_with_optional_30m(
        stored.loc[stored["M1_top_5pct"].astype(bool)]
    )
    stored_timing = pd.read_csv(PRIMARY / "movement_timing_metrics.csv")
    timing_difference = 0.0
    for row in timing.itertuples(index=False):
        reference = stored_timing.loc[
            pd.to_numeric(stored_timing["horizon_minutes"], errors="raise").eq(
                int(row.horizon_minutes)
            )
        ]
        if len(reference) != 1:
            timing_difference = math.inf
            break
        for column in (
            "rows_available",
            "rows_with_30m",
            "mean_iv_residual",
            "median_iv_residual",
            "exceed_iv_rate",
            "percent_eventual_30m_movement_realized",
            "maximum_absolute_excursion_bucket_share",
        ):
            timing_difference = max(
                timing_difference,
                maximum_nullable_difference(
                    [float(getattr(row, column))],
                    [float(reference.iloc[0][column])],
                ),
            )
    passed = bool(
        row_mismatches == 0
        and feature_difference <= 1e-12
        and movement_difference <= 1e-12
        and timing_difference <= 1e-12
    )
    return (
        {
            "joined_row_mismatches": row_mismatches,
            "maximum_feature_difference": feature_difference,
            "maximum_movement_difference": movement_difference,
            "maximum_optional_timing_difference": timing_difference,
            "optional_30m_missing_rows": int(stored["close_30m"].isna().sum()),
            "binding_5_10_15_missing_rows": int(
                stored[["close_5m", "close_10m", "close_15m"]].isna().any(axis=1).sum()
            ),
            "passed": passed,
        },
        rebuilt,
        pairs,
    )


def audit_decision_support() -> dict[str, Any]:
    decision = read_json(PRIMARY / "decision.json")
    assert_v01_safety_flags(decision)
    joined = cast(Mapping[str, Any], decision["joined_holdout_support"])
    tail = cast(Mapping[str, Any], decision["frozen_tail_support"])
    expected_overall = (
        "blocked_insufficient_holdout_support"
        if not bool(joined["passed"])
        else (
            "blocked_insufficient_frozen_tail_support"
            if not bool(tail["passed"])
            else str(decision["overall_decision"])
        )
    )
    passed = bool(
        decision["overall_decision"] == expected_overall
        and decision["download_resume_status"] == "supported"
        and decision["holdout_options_coverage_status"] == "supported"
        and decision["minimal_model_status"] == "supported"
        and decision["frozen_top_5pct_status"] == "insufficient_support"
        and not bool(tail["passed"])
        and float(tail["maximum_stock_share"]) > 0.18
    )
    return {
        "stored_overall_decision": decision["overall_decision"],
        "expected_overall_decision": expected_overall,
        "minimal_model_status": decision["minimal_model_status"],
        "frozen_tail_status": decision["frozen_top_5pct_status"],
        "tail_maximum_stock_share": tail["maximum_stock_share"],
        "passed": passed,
    }


def main() -> int:
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    assert_v01_safety_flags(contract)
    required = (
        "source_manifest.json",
        "protected_boundary_audit.json",
        "holdout_data_authorisation.json",
        "complete_receipt_reuse_audit.csv",
        "interrupted_request_repair.json",
        "resume_download_manifest.json",
        "holdout_options_coverage.csv",
        "holdout_coverage_preflight.json",
        "remaining_options_gap.csv",
        "historical_model_reconstruction.json",
        "minimal_feature_manifest.json",
        "pre_outcome_freeze_manifest.json",
        "post_run_reporting_corrections.json",
        "frozen_tail_thresholds.json",
        "holdout_selected_option_pairs.parquet",
        "holdout_join_audit.csv",
        "holdout_panel.parquet",
        "model_configurations.json",
        "model_coefficients.json",
        "holdout_predictions.parquet",
        "holdout_model_metrics.csv",
        "holdout_monthly_metrics.csv",
        "holdout_checkpoint_metrics.csv",
        "tail_metrics.csv",
        "tail_comparison_metrics.csv",
        "tail_overlap_metrics.csv",
        "movement_timing_metrics.csv",
        "bootstrap_metrics.csv",
        "intraday_h0_null_metrics.csv",
        "concentration_metrics.csv",
        "decision.json",
        "report.md",
    )
    missing = [name for name in required if not (PRIMARY / name).is_file()]
    if missing:
        raise ValueError(f"required V0.1 artifacts are missing: {missing}")

    downloader = load_module(V0_DOWNLOADER, "minimal_holdout_v0_downloader_for_v01_audit")
    plan = rebuild_plan(downloader)
    resume = audit_resume_scope(plan)
    v01_downloader = load_module(V01_DOWNLOADER, "minimal_holdout_v01_downloader_for_audit")
    with tempfile.TemporaryDirectory(prefix="stocker-v01-canonical-audit-") as temporary:
        rebuilt_options = Path(temporary) / "exact_holdout_options.parquet"
        canonical_rebuild = audit_canonical_rebuild(
            plan,
            v0_downloader=downloader,
            v01_downloader=v01_downloader,
            rebuilt_path=rebuilt_options,
        )
        resume["canonical_rebuild_audit"] = canonical_rebuild
        resume["passed"] = bool(resume["passed"] and canonical_rebuild["passed"])
        auditor, runner = configure_v0_modules(rebuilt_options)
        historical = cast(dict[str, Any], auditor.audit_historical_models())
        h0 = cast(dict[str, Any], auditor.audit_h0_surface())
        feature_manifest = read_json(PRIMARY / "minimal_feature_manifest.json")
        exact_features = bool(
            cast(Mapping[str, Any], feature_manifest["models"])["M0"]["numeric_features"]
            == list(auditor.GROUP_O)
            and cast(Mapping[str, Any], feature_manifest["models"])["M1"]["numeric_features"]
            == [*auditor.GROUP_O, *auditor.GROUP_I]
            and not set((*auditor.GROUP_O, *auditor.GROUP_I)).intersection(
                auditor.EXCLUDED_FEATURES
            )
        )
        join_movement, rebuilt, pairs = audit_join_and_movement(runner)
        preoutcome = audit_preoutcome_order(
            rebuilt,
            pairs,
            planned_stock_sessions=len(plan),
            planned_session_count=len({str(row["holdout_session"]) for row in plan}),
        )
        decision = audit_decision_support()
        completed_audit = bool(
            auditor.audit_completed_artifacts(
                historical=historical,
                h0=h0,
                exact_features=exact_features,
                request_gate=bool(resume["passed"]),
            )
        )
    base_determinism = read_json(PRIMARY / "determinism_check.json")
    base_lightweight = read_json(PRIMARY / "lightweight_audit.json")
    passed = bool(
        resume["passed"]
        and historical["passed"]
        and h0["passed"]
        and exact_features
        and join_movement["passed"]
        and preoutcome["passed"]
        and decision["passed"]
        and completed_audit
        and base_determinism["passed"]
        and base_lightweight["passed"]
    )
    determinism = {
        **SAFETY_FLAGS,
        **base_determinism,
        "status": "passed" if passed else "failed",
        "selected_contract_mismatches": int(base_determinism["selected_contract_mismatches"]),
        "joined_row_mismatches": int(join_movement["joined_row_mismatches"]),
        "maximum_feature_difference": float(join_movement["maximum_feature_difference"]),
        "maximum_probability_difference": float(base_determinism["maximum_probability_difference"]),
        "tail_membership_mismatches": int(base_determinism["tail_membership_mismatches"]),
        "maximum_movement_difference": float(join_movement["maximum_movement_difference"]),
        "bootstrap_repeated": False,
        "null_draws_repeated": False,
        "passed": passed,
    }
    write_json(PRIMARY / "determinism_check.json", determinism)
    lightweight = {
        **SAFETY_FLAGS,
        **base_lightweight,
        "audit_scope": "strict_resume_and_completed_binding_holdout",
        "resume_scope_audit": resume,
        "historical_model_audit": historical,
        "historical_h0_audit": h0,
        "preoutcome_order_audit": preoutcome,
        "join_and_movement_audit": join_movement,
        "decision_support_audit": decision,
        "v0_completed_artifact_audit_passed": completed_audit,
        "session_bootstrap_evidence_recalculated_without_redraw": True,
        "intraday_h0_null_evidence_reconstructed_without_refit": True,
        "independent_audit_passed": passed,
        "passed": passed,
    }
    write_json(PRIMARY / "lightweight_audit.json", lightweight)
    print("independent V0.1 audit passed" if passed else "independent V0.1 audit failed")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
