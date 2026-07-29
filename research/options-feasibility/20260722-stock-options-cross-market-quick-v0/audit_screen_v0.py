#!/usr/bin/env python3
"""Independent fail-closed audit for the bounded cross-market quick screen."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
RUNNER_PATH = EXPERIMENT_DIR / "run_screen_v0.py"
DENSE_PANEL = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts"
    / "primary"
    / "dense_advance_panel.parquet"
)
TRACE_PANEL = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-route-competition-hazard-quick-v0"
    / "artifacts"
    / "primary"
    / "causal_state_trace.parquet"
)
PROBE_RESULTS = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260722-broad-conflict-prior-close-iv-v01-probe"
    / "artifacts"
    / "primary"
    / "contract_history_probe_results.csv"
)

for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))

from stocker_research.stock_options_cross_market_quick_v0 import (  # noqa: E402
    BASE_OPTIONS_FEATURES,
    BOOTSTRAP_SEED,
    CROSS_MARKET_FEATURES,
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    FROZEN_COHORT,
    OPTIONS_MODEL_FEATURES,
    OPTIONS_NULL_SEEDS,
    RIDGE_R0_NUMERIC,
    RIDGE_R1_NUMERIC,
    ROUTE_FEATURES,
    ROUTE_NULL_SEEDS,
    SAFETY_FLAGS,
    STOCK_RELATIVE_OPTIONS_FEATURES,
    TEST_A_S0_NUMERIC,
    TEST_A_S1_NUMERIC,
    TEST_A_S2_NUMERIC,
    TEST_B_O0_NUMERIC,
    TEST_B_O1_NUMERIC,
    TEST_B_O2_NUMERIC,
    route_state_movement_metrics,
)

BLOCKER = "blocked_insufficient_cached_options_coverage"
AUDIT_BLOCKER = "blocked_reproducibility_or_audit_failure"
REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "structural_panel_reconstruction.json",
    "option_pair_reconstruction.json",
    "options_coverage_gap.csv",
    "cross_market_feature_manifest.json",
    "cross_market_panel.parquet",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "test_a_metrics.csv",
    "test_a_monthly_metrics.csv",
    "test_a_subgroup_metrics.csv",
    "test_b_metrics.csv",
    "test_b_monthly_metrics.csv",
    "test_b_route_state_metrics.csv",
    "continuous_residual_metrics.csv",
    "bootstrap_metrics.csv",
    "options_null_metrics.csv",
    "route_null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "lightweight_audit.json",
    "determinism_check.json",
    "report.md",
)


class AuditFailure(RuntimeError):
    """Raised at the first unexplained audit discrepancy."""


def read_json(path: Path) -> dict[str, Any]:
    """Read an object-valued JSON artifact."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditFailure(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write the audit result atomically."""

    content = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    """Hash a file without loading the complete file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner() -> ModuleType:
    """Load the runner in a separate module namespace for cache reconstruction."""

    specification = importlib.util.spec_from_file_location(
        "stock_options_cross_market_quick_v0_audit_runner",
        RUNNER_PATH,
    )
    if specification is None or specification.loader is None:
        raise AuditFailure("runner module could not be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def maximum_numeric_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    """Return an exacting maximum numeric difference, including missingness."""

    if len(left) != len(right):
        return math.inf
    maximum = 0.0
    for column in columns:
        first = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        second = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        if np.logical_xor(np.isnan(first), np.isnan(second)).any():
            return math.inf
        difference = np.abs(first - second)
        difference[np.isnan(first) & np.isnan(second)] = 0.0
        finite = difference[np.isfinite(difference)]
        if finite.size:
            maximum = max(maximum, float(finite.max()))
    return maximum


def audit() -> dict[str, Any]:
    """Independently recompute and verify every applicable requirement."""

    checks: dict[str, dict[str, Any]] = {}

    def require(name: str, condition: bool, **details: Any) -> None:
        checks[name] = {"passed": bool(condition), **details}
        if not condition:
            raise AuditFailure(name)

    missing_artifacts = [name for name in REQUIRED_ARTIFACTS if not (PRIMARY / name).is_file()]
    require("required_artifacts", not missing_artifacts, missing=missing_artifacts)

    contract = read_json(PRIMARY / "contract.json")
    decision = read_json(PRIMARY / "decision.json")
    source_manifest = read_json(PRIMARY / "source_manifest.json")
    boundary = read_json(PRIMARY / "protected_boundary_audit.json")
    structural_audit = read_json(PRIMARY / "structural_panel_reconstruction.json")
    pair_audit = read_json(PRIMARY / "option_pair_reconstruction.json")
    feature_manifest = read_json(PRIMARY / "cross_market_feature_manifest.json")
    model_config = read_json(PRIMARY / "model_configurations.json")
    coefficients = read_json(PRIMARY / "model_coefficients.json")
    determinism = read_json(PRIMARY / "determinism_check.json")
    safety_mismatches: dict[str, dict[str, Any]] = {}
    for artifact_name, artifact in {
        "contract": contract,
        "decision": decision,
        "source_manifest": source_manifest,
        "protected_boundary_audit": boundary,
        "structural_panel_reconstruction": structural_audit,
        "option_pair_reconstruction": pair_audit,
        "cross_market_feature_manifest": feature_manifest,
        "model_configurations": model_config,
        "model_coefficients": coefficients,
        "determinism_check": determinism,
    }.items():
        mismatches = {
            key: {"expected": expected, "observed": artifact.get(key)}
            for key, expected in SAFETY_FLAGS.items()
            if artifact.get(key) != expected
        }
        if mismatches:
            safety_mismatches[artifact_name] = mismatches
    require("safety_flags", not safety_mismatches, mismatches=safety_mismatches)

    source_hash_mismatches: dict[str, dict[str, str]] = {}
    for path_key, hash_key in (
        ("frozen_dense_panel", "frozen_dense_panel_sha256"),
        ("frozen_trace_panel", "frozen_trace_panel_sha256"),
        ("frozen_pair_ledger", "frozen_pair_ledger_sha256"),
        ("options_schema_mapping", "options_schema_mapping_sha256"),
    ):
        source_path = REPO_ROOT / str(source_manifest[path_key])
        observed_hash = sha256_file(source_path)
        expected_hash = str(source_manifest[hash_key])
        if observed_hash != expected_hash:
            source_hash_mismatches[path_key] = {
                "expected": expected_hash,
                "observed": observed_hash,
            }
    require("source_hashes", not source_hash_mismatches, mismatches=source_hash_mismatches)

    gap = pd.read_csv(PRIMARY / "options_coverage_gap.csv")
    split_boundary = gap["split_boundary_ambiguous"].astype(str).str.casefold().eq("true")
    exact_chain = gap["exact_chain_cached"].astype(str).str.casefold().eq("true")
    valid_pair = gap["valid_atm_pair_cached"].astype(str).str.casefold().eq("true")
    eligible_gap = gap.loc[~split_boundary]
    missing_exact_chain = int((~exact_chain & ~split_boundary).sum())
    request_gap = cast(dict[str, Any], source_manifest["request_gap"])
    request_plan_mismatches = {
        "eligible_stock_dates": (
            len(eligible_gap),
            int(request_gap["eligible_stock_dates"]),
        ),
        "missing_exact_chain_stock_dates": (
            missing_exact_chain,
            int(request_gap["missing_exact_chain_stock_dates"]),
        ),
        "cached_exact_chain_stock_dates": (
            int(exact_chain.sum()),
            int(request_gap["cached_exact_chain_stock_dates"]),
        ),
        "cached_valid_atm_pair_stock_dates": (
            int(valid_pair.sum()),
            int(request_gap["cached_valid_atm_pair_stock_dates"]),
        ),
        "estimated_additional_requests": (
            int(gap["eodhd_total_requests_estimate"].sum()),
            int(request_gap["estimated_additional_requests"]),
        ),
        "estimated_additional_records": (
            int(gap["estimated_additional_records"].sum()),
            int(request_gap["estimated_additional_records"]),
        ),
    }
    request_plan_mismatches = {
        key: {"recomputed": values[0], "reported": values[1]}
        for key, values in request_plan_mismatches.items()
        if values[0] != values[1]
    }
    require(
        "coverage_gap_and_request_plan",
        not request_plan_mismatches
        and len(source_manifest["coverage_by_stock"]) == 20
        and len(source_manifest["coverage_by_date"]) > 0
        and len(source_manifest["coverage_by_month"]) >= 5
        and source_manifest["exact_stock_date_gap_manifest"] == "options_coverage_gap.csv",
        stock_date_rows=len(gap),
        missing_exact_chain_stock_dates=missing_exact_chain,
        estimated_additional_requests=int(gap["eodhd_total_requests_estimate"].sum()),
        estimated_additional_records=int(gap["estimated_additional_records"].sum()),
        mismatches=request_plan_mismatches,
    )

    panel = pd.read_parquet(PRIMARY / "cross_market_panel.parquet").sort_values(
        "row_id", kind="mergesort"
    )
    panel = panel.reset_index(drop=True)
    dense = pd.read_parquet(DENSE_PANEL)
    trace = pd.read_parquet(TRACE_PANEL)
    for frame_name, frame, columns in (
        ("cross_market_panel", panel, ("session", "required_options_date")),
        ("dense_structural_panel", dense, ("session",)),
        ("causal_state_trace", trace, ("session",)),
    ):
        for column in columns:
            dates = pd.to_datetime(frame[column], errors="raise")
            require(
                f"protected_date_{frame_name}_{column}",
                bool(dates.lt(pd.Timestamp("2025-08-23")).all()),
                maximum=str(dates.max().date()),
            )
    same_day_or_future = int(
        (pd.to_datetime(panel["required_options_date"]) >= pd.to_datetime(panel["session"])).sum()
    )
    require(
        "no_same_day_or_future_option_join",
        same_day_or_future == 0,
        same_day_or_future_rows=same_day_or_future,
    )
    require(
        "protected_boundary_artifact",
        boundary.get("passed") is True
        and int(boundary.get("protected_rows_materialised", -1)) == 0
        and int(boundary.get("protected_history_records_materialised", -1)) == 0
        and int(
            boundary.get(
                "protected_cached_history_records_skipped_before_materialisation",
                -1,
            )
        )
        >= 0
        and int(boundary.get("same_day_or_future_options_joins", -1)) == 0,
        protected_rows_materialised=boundary.get("protected_rows_materialised"),
        protected_cached_history_records_skipped_before_materialisation=(
            boundary.get("protected_cached_history_records_skipped_before_materialisation")
        ),
    )

    trace_calendar = {
        str(symbol): sorted(
            pd.to_datetime(group["session"], errors="raise").dt.date.unique().tolist()
        )
        for symbol, group in trace.groupby("symbol", sort=True)
    }
    previous_session_mismatches = 0
    for row in panel.drop_duplicates(["symbol", "session"]).itertuples(index=False):
        signal_date = pd.Timestamp(cast(Any, row).session).date()
        earlier = [
            value for value in trace_calendar[str(cast(Any, row).symbol)] if value < signal_date
        ]
        expected_previous = max(earlier)
        observed_previous = pd.Timestamp(cast(Any, row).required_options_date).date()
        previous_session_mismatches += int(expected_previous != observed_previous)
    require(
        "exact_previous_trading_session_join",
        previous_session_mismatches == 0,
        mismatches=previous_session_mismatches,
    )

    next_bar = dense["registered_completion_next_1_bar"].fillna(0).astype(int)
    one_transition = dense["any_prefix_one_transition_from_completion"].fillna(0).astype(int)
    eligible = next_bar.eq(0) & one_transition.eq(0)
    clean_reference = dense.loc[eligible].copy()
    clean_reference["audit_target"] = (
        pd.to_numeric(clean_reference["first_completion_lead"], errors="raise")
        .isin([2, 3])
        .astype(int)
    )
    eligibility_mismatches = int(eligible.ne(dense["advance_eligible"].astype(int).eq(1)).sum())
    target_mismatches = int(
        clean_reference["audit_target"]
        .ne(clean_reference["completion_in_bars_2_or_3"].astype(int))
        .sum()
    )
    development_dates = pd.to_datetime(
        clean_reference.loc[
            clean_reference["period"].astype(str).eq("development"),
            "session",
        ],
        errors="raise",
    )
    assessment_dates = pd.to_datetime(
        clean_reference.loc[
            clean_reference["period"].astype(str).eq("assessment"),
            "session",
        ],
        errors="raise",
    )
    require(
        "frozen_dates_cohort_and_checkpoints",
        set(clean_reference["period"].astype(str)) == {"development", "assessment"}
        and bool(
            development_dates.between(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-12-31"),
            ).all()
        )
        and bool(
            assessment_dates.between(
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-08-22"),
            ).all()
        )
        and tuple(sorted(clean_reference["symbol"].astype(str).unique())) == FROZEN_COHORT
        and tuple(sorted(clean_reference["checkpoint"].astype(int).unique())) == DENSE_CHECKPOINTS
        and contract["development_start"] == "2024-01-01"
        and contract["development_end_inclusive"] == "2024-12-31"
        and contract["assessment_start"] == "2025-01-01"
        and contract["assessment_end_inclusive"] == "2025-08-22",
        development_minimum=str(development_dates.min().date()),
        development_maximum=str(development_dates.max().date()),
        assessment_minimum=str(assessment_dates.min().date()),
        assessment_maximum=str(assessment_dates.max().date()),
        symbols=sorted(clean_reference["symbol"].astype(str).unique().tolist()),
        checkpoints=sorted(clean_reference["checkpoint"].astype(int).unique().tolist()),
    )
    require(
        "structural_eligibility_and_target",
        eligibility_mismatches == 0
        and target_mismatches == 0
        and len(clean_reference) == int(structural_audit["clean_advance_rows"]),
        rows=len(clean_reference),
        eligibility_mismatches=eligibility_mismatches,
        target_mismatches=target_mismatches,
    )
    reference_counts = clean_reference.groupby(["period", "session", "symbol"], sort=False)[
        "symbol"
    ].transform("size")
    session_stock_counts = clean_reference.groupby(["period", "session"], sort=False)[
        "symbol"
    ].transform("nunique")
    audit_weights = 1.0 / (reference_counts.to_numpy(float) * session_stock_counts.to_numpy(float))
    weight_difference = float(
        np.max(
            np.abs(
                audit_weights
                - pd.to_numeric(clean_reference["row_weight"], errors="raise").to_numpy(float)
            )
        )
    )
    require(
        "candidate_normalized_weights",
        weight_difference <= 1e-12,
        maximum_difference=weight_difference,
    )
    joined_reference = clean_reference.merge(
        panel[["row_id"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("row_id", kind="mergesort")
    panel_ids = panel["row_id"].astype(str).tolist()
    reference_ids = joined_reference["row_id"].astype(str).tolist()
    row_identity_mismatches = abs(len(panel_ids) - len(reference_ids)) + sum(
        left != right for left, right in zip(panel_ids, reference_ids, strict=False)
    )
    aligned_reference = clean_reference.loc[
        clean_reference["row_id"].astype(str).isin(panel_ids)
    ].sort_values("row_id", kind="mergesort")
    aligned_reference = aligned_reference.reset_index(drop=True)
    shared_difference = maximum_numeric_difference(
        aligned_reference,
        panel,
        [*DENSE_H0_FEATURES, *ROUTE_FEATURES, "row_weight"],
    )
    route_mismatches = int(
        aligned_reference["route_resolution_state"]
        .astype(str)
        .ne(panel["route_resolution_state"].astype(str))
        .sum()
    )
    require(
        "structural_panel_reconstruction",
        structural_audit.get("passed") is True
        and int(structural_audit["row_identity_mismatches"]) == 0
        and int(structural_audit["route_state_mismatches"]) == 0
        and float(structural_audit["maximum_shared_feature_difference"]) <= 1e-12
        and row_identity_mismatches == 0
        and route_mismatches == 0
        and shared_difference <= 1e-12,
        joined_row_identity_mismatches=row_identity_mismatches,
        joined_route_state_mismatches=route_mismatches,
        joined_maximum_shared_feature_difference=shared_difference,
    )

    raw_hash_failures = 0
    raw_record_mismatches = 0
    protected_cached_history_records = 0
    for raw_file in cast(list[dict[str, Any]], pair_audit["raw_cache_files"]):
        raw_path = REPO_ROOT / str(raw_file["path"])
        raw_hash_failures += int(sha256_file(raw_path) != str(raw_file["sha256"]))
        if str(raw_file["endpoint"]).endswith("/eod"):
            raw_text = raw_path.read_text(encoding="utf-8")
            resource_ids = re.findall(
                r'"id"\s*:\s*"([^"]+-\d{4}-\d{2}-\d{2})"',
                raw_text,
            )
            raw_record_mismatches += int(len(resource_ids) != int(raw_file["record_count"]))
            protected_cached_history_records += sum(
                value[-10:] >= "2025-08-23" for value in resource_ids
            )
        else:
            payload = read_json(raw_path)
            records = payload.get("data")
            raw_record_mismatches += int(
                not isinstance(records, list) or len(records) != int(raw_file["record_count"])
            )
    require(
        "cached_options_integrity",
        raw_hash_failures == 0
        and raw_record_mismatches == 0
        and int(pair_audit["cached_vendor_records_used"]) == 3_624
        and int(pair_audit["cached_contract_history_records_materialised"]) == 18
        and int(pair_audit["protected_history_records_materialised"]) == 0
        and protected_cached_history_records
        == int(pair_audit["protected_cached_history_records_skipped_before_materialisation"])
        and int(pair_audit["network_requests_made"]) == 0,
        raw_hash_failures=raw_hash_failures,
        raw_record_count_mismatches=raw_record_mismatches,
        cached_vendor_records_used=pair_audit["cached_vendor_records_used"],
        history_records_materialised=pair_audit["cached_contract_history_records_materialised"],
        protected_cached_history_records_skipped_before_materialisation=(
            protected_cached_history_records
        ),
        protected_history_records_materialised=pair_audit["protected_history_records_materialised"],
    )

    runner = load_runner()
    rebuilt_pairs, rebuilt_pair_audit = runner.rebuild_cached_pairs()
    require(
        "frozen_pair_selection",
        rebuilt_pair_audit["passed"] is True
        and int(rebuilt_pair_audit["selected_contract_mismatches"]) == 0
        and int(rebuilt_pair_audit["valid_primary_pairs"]) == 8
        and int(rebuilt_pair_audit["exact_previous_close_chains"]) == 9
        and int(rebuilt_pair_audit["protected_history_records_materialised"]) == 0,
        exact_chains=rebuilt_pair_audit["exact_previous_close_chains"],
        valid_pairs=rebuilt_pair_audit["valid_primary_pairs"],
        selected_contract_mismatches=rebuilt_pair_audit["selected_contract_mismatches"],
    )
    expected_ledger = pd.read_csv(PROBE_RESULTS)
    valid_ledger = expected_ledger.loc[
        expected_ledger["pair_available"].astype(str).str.casefold().eq("true")
    ].sort_values(["symbol", "signal_date"], kind="mergesort")
    panel_pairs = panel.drop_duplicates(["symbol", "signal_date"]).sort_values(
        ["symbol", "signal_date"], kind="mergesort"
    )
    rebuilt_valid = rebuilt_pairs.loc[rebuilt_pairs["pair_available"].astype(bool)].sort_values(
        ["symbol", "signal_date"], kind="mergesort"
    )
    identity_columns = (
        "symbol",
        "signal_date",
        "required_options_date",
        "selected_expiry",
        "selected_strike",
        "call_contract_id",
        "put_contract_id",
    )
    pair_identity_mismatches = 0
    for expected_frame in (valid_ledger, rebuilt_valid):
        expected_frame = expected_frame.reset_index(drop=True)
        observed = panel_pairs.reset_index(drop=True)
        pair_identity_mismatches += abs(len(expected_frame) - len(observed))
        for column in identity_columns:
            pair_identity_mismatches += int(
                expected_frame[column]
                .fillna("<missing>")
                .astype(str)
                .ne(observed[column].fillna("<missing>").astype(str))
                .sum()
            )
    rebuilt_feature_difference = maximum_numeric_difference(
        rebuilt_valid.reset_index(drop=True),
        panel_pairs.reset_index(drop=True),
        BASE_OPTIONS_FEATURES,
    )
    require(
        "option_pair_and_feature_reconstruction",
        pair_identity_mismatches == 0 and rebuilt_feature_difference <= 1e-12,
        identity_mismatches=pair_identity_mismatches,
        maximum_option_feature_difference=rebuilt_feature_difference,
    )

    trace_for_volatility = trace.sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort")
    daily = (
        trace_for_volatility.groupby(["symbol", "session"], sort=True, as_index=False)
        .agg(close=("close", "last"))
        .sort_values(["symbol", "session"], kind="mergesort")
    )
    daily["log_return"] = daily.groupby("symbol", sort=False)["close"].transform(
        lambda values: np.log(values).diff()
    )
    daily["audit_realised_volatility"] = daily.groupby("symbol", sort=False)[
        "log_return"
    ].transform(lambda values: values.rolling(20, min_periods=15).std(ddof=1) * math.sqrt(252.0))
    daily["audit_valid_sessions"] = daily.groupby("symbol", sort=False)["log_return"].transform(
        lambda values: values.rolling(20, min_periods=1).count()
    )
    volatility_check = panel.merge(
        daily.rename(columns={"session": "required_options_date"})[
            [
                "symbol",
                "required_options_date",
                "audit_realised_volatility",
                "audit_valid_sessions",
            ]
        ],
        on=["symbol", "required_options_date"],
        how="left",
        validate="many_to_one",
    )
    expected_volatility = volatility_check[
        ["audit_realised_volatility", "audit_valid_sessions"]
    ].rename(
        columns={
            "audit_realised_volatility": "realised_volatility_20d",
            "audit_valid_sessions": "valid_trailing_return_sessions",
        }
    )
    volatility_difference = maximum_numeric_difference(
        panel,
        expected_volatility,
        ("realised_volatility_20d", "valid_trailing_return_sessions"),
    )
    residual_difference = float(
        np.nanmax(
            np.abs(
                (
                    pd.to_numeric(panel["atm_iv"], errors="raise")
                    - pd.to_numeric(panel["realised_volatility_20d"], errors="coerce")
                ).to_numpy(float)
                - pd.to_numeric(panel["iv_minus_realised_20d"], errors="coerce").to_numpy(float)
            )
        )
    )
    require(
        "trailing_realised_volatility",
        volatility_difference <= 1e-12 and residual_difference <= 1e-12,
        maximum_volatility_difference=volatility_difference,
        maximum_iv_minus_realised_difference=residual_difference,
    )

    stock_scale_difference = 0.0
    stock_parameters = cast(
        dict[str, dict[str, dict[str, Any]]],
        feature_manifest["development_only_stock_relative_parameters"],
    )
    stock_source_map = {
        "atm_iv_stock_robust_z": "atm_iv",
        "straddle_move_stock_robust_z": "straddle_mid_pct",
        "skew_stock_robust_z": "skew_25d",
        "term_structure_stock_robust_z": "term_structure",
    }
    for row in panel.itertuples(index=False):
        symbol = str(cast(Any, row).symbol)
        parameters = stock_parameters[symbol]
        atm_fit = parameters.get("atm_iv")
        if atm_fit is not None:
            sorted_values = np.asarray(atm_fit["sorted_values"], dtype=float)
            expected_percentile = float(
                np.searchsorted(
                    sorted_values,
                    float(cast(Any, row).atm_iv),
                    side="right",
                )
                / len(sorted_values)
            )
            stock_scale_difference = max(
                stock_scale_difference,
                abs(expected_percentile - float(cast(Any, row).atm_iv_stock_percentile)),
            )
        for target, source in stock_source_map.items():
            value = float(getattr(cast(Any, row), source))
            fitted = parameters.get(source)
            expected_value = (
                0.0
                if fitted is None or not math.isfinite(value)
                else (value - float(fitted["median"])) / float(fitted["scale"])
            )
            stock_scale_difference = max(
                stock_scale_difference,
                abs(expected_value - float(getattr(cast(Any, row), target))),
            )
    require(
        "development_only_stock_relative_scaling",
        stock_scale_difference <= 1e-12
        and feature_manifest["fit_period"] == "2024 development only",
        maximum_difference=stock_scale_difference,
    )

    standardization = cast(
        dict[str, dict[str, float]],
        feature_manifest["development_only_standardization"],
    )
    standard_source = {
        "standardised_tension": "tension",
        "standardised_prefix_family_entropy": "prefix_family_entropy",
        "standardised_signed_pressure": "signed_pressure",
        "standardised_call_put_iv_gap": "call_put_iv_gap",
        "standardised_transition_probability": "transition_probability",
    }
    standard_difference = 0.0
    for target, source in standard_source.items():
        fitted = standardization[target]
        expected_values = (
            pd.to_numeric(panel[source], errors="raise").to_numpy(float) - float(fitted["mean"])
        ) / float(fitted["scale"])
        standard_difference = max(
            standard_difference,
            float(
                np.max(
                    np.abs(
                        expected_values
                        - pd.to_numeric(panel[target], errors="raise").to_numpy(float)
                    )
                )
            ),
        )
    expected_disagreement = pd.DataFrame(
        {
            "complacent_conflict": panel["BROAD_CONFLICT"].astype(float)
            * (-panel["atm_iv_stock_robust_z"]),
            "structural_tension_gap": panel["standardised_tension"]
            - panel["atm_iv_stock_robust_z"],
            "route_vs_priced_move": panel["standardised_prefix_family_entropy"]
            - panel["straddle_move_stock_robust_z"],
            "directional_agreement": panel["standardised_signed_pressure"]
            * panel["standardised_call_put_iv_gap"],
            "transition_vs_term_urgency": panel["standardised_transition_probability"]
            - panel["term_structure_stock_robust_z"],
        }
    )
    disagreement_difference = maximum_numeric_difference(
        expected_disagreement,
        panel,
        CROSS_MARKET_FEATURES,
    )
    require(
        "five_cross_market_disagreement_features",
        tuple(feature_manifest["cross_market_disagreement_features"]) == CROSS_MARKET_FEATURES
        and standard_difference <= 1e-12
        and disagreement_difference <= 1e-12,
        maximum_standardization_difference=standard_difference,
        maximum_disagreement_difference=disagreement_difference,
    )

    test_a_expected = (
        pd.to_numeric(panel["first_completion_lead"], errors="raise").isin([2, 3]).astype(int)
    )
    test_a_mismatches = int(
        test_a_expected.ne(panel["registered_completion_clean_bars_2_or_3"].astype(int)).sum()
    )
    excluded_target_rows = int(
        (
            panel["registered_completion_next_1_bar"].fillna(0).astype(int).ne(0)
            | panel["any_prefix_one_transition_from_completion"].fillna(0).astype(int).ne(0)
        ).sum()
    )
    require(
        "test_a_target",
        test_a_mismatches == 0 and excluded_target_rows == 0,
        target_mismatches=test_a_mismatches,
        ineligible_rows_materialised=excluded_target_rows,
    )

    trace_groups = {
        (str(symbol), str(session)): group.set_index("bar_ordinal", drop=False)
        for (symbol, session), group in trace.sort_values(
            ["symbol", "session", "bar_ordinal"], kind="mergesort"
        ).groupby(["symbol", "session"], sort=False)
    }
    movement_difference = 0.0
    target_mismatches_b = 0
    for row in panel.itertuples(index=False):
        bars = trace_groups[(str(cast(Any, row).symbol), str(cast(Any, row).session))]
        checkpoint = int(cast(Any, row).checkpoint_bar_ordinal_zero_based)
        entry = float(bars.loc[checkpoint + 1, "open"])
        third_close = float(bars.loc[checkpoint + 3, "close"])
        movement = abs(math.log(third_close / entry))
        sigma = float(cast(Any, row).atm_iv) * math.sqrt(15.0 / (252.0 * 390.0))
        expected_absolute = sigma * math.sqrt(2.0 / math.pi)
        residual = movement - expected_absolute
        movement_difference = max(
            movement_difference,
            abs(entry - float(cast(Any, row).entry_price)),
            abs(movement - float(cast(Any, row).absolute_log_return_15m)),
            abs(sigma - float(cast(Any, row).iv_sigma_15m)),
            abs(expected_absolute - float(cast(Any, row).iv_expected_absolute_15m)),
            abs(residual - float(cast(Any, row).iv_absolute_residual_15m)),
        )
        target_mismatches_b += int(
            int(movement > expected_absolute) != int(cast(Any, row).movement_exceeds_prior_close_iv)
        )
    require(
        "test_b_underlying_movement_target",
        movement_difference <= 1e-12 and target_mismatches_b == 0,
        maximum_difference=movement_difference,
        target_mismatches=target_mismatches_b,
    )

    expected_model_surfaces = {
        "S0": (TEST_A_S0_NUMERIC, ["stock"]),
        "S1": (TEST_A_S1_NUMERIC, ["stock"]),
        "S2": (TEST_A_S2_NUMERIC, ["stock"]),
        "O0": (TEST_B_O0_NUMERIC, ["stock", "checkpoint", "month_of_year"]),
        "O1": (TEST_B_O1_NUMERIC, ["stock", "checkpoint", "month_of_year"]),
        "O2": (
            TEST_B_O2_NUMERIC,
            ["stock", "checkpoint", "month_of_year", "route_state"],
        ),
        "R0": (RIDGE_R0_NUMERIC, []),
        "R1": (RIDGE_R1_NUMERIC, ["route_state"]),
    }
    surface_mismatches: list[str] = []
    for model, (numeric, categoricals) in expected_model_surfaces.items():
        specification = cast(dict[str, Any], model_config["models"][model])
        if tuple(specification["numeric_features"]) != tuple(numeric):
            surface_mismatches.append(f"{model}:numeric")
        if list(specification["categorical_controls"]) != categoricals:
            surface_mismatches.append(f"{model}:categorical")
    require(
        "model_feature_surfaces",
        not surface_mismatches
        and tuple(feature_manifest["base_options_features"]) == BASE_OPTIONS_FEATURES
        and tuple(feature_manifest["stock_relative_options_features"])
        == STOCK_RELATIVE_OPTIONS_FEATURES
        and tuple(feature_manifest["options_model_features"]) == OPTIONS_MODEL_FEATURES,
        mismatches=surface_mismatches,
    )
    fixed_model = cast(dict[str, Any], model_config["model"])
    require(
        "model_and_preprocessing_configuration",
        fixed_model
        == {
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
        }
        and model_config["preprocessing_fit_period"] == "2024 development only"
        and model_config["probability_quantiles_fit_period"] == "2024 development predictions only"
        and cast(dict[str, Any], model_config["ridge"])["alpha"] == 10.0,
        primary_model=fixed_model,
        ridge=model_config["ridge"],
    )

    assessment = panel.loc[panel["period"].eq("assessment")]
    assessment_reference_rows = int(clean_reference["period"].astype(str).eq("assessment").sum())
    stock_weight = assessment.groupby("symbol", sort=True)["row_weight"].sum()
    maximum_stock_share = float((stock_weight / stock_weight.sum()).max())
    coverage_evidence = {
        "assessment_rows": int(len(assessment)),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_stocks": int(assessment["symbol"].nunique()),
        "assessment_months": int(pd.to_datetime(assessment["session"]).dt.to_period("M").nunique()),
        "test_a_positives": int(assessment["registered_completion_clean_bars_2_or_3"].sum()),
        "test_b_positives": int(assessment["movement_exceeds_prior_close_iv"].sum()),
        "broad_conflict_rows": int(assessment["route_resolution_state"].eq("BROAD_CONFLICT").sum()),
        "low_route_support_rows": int(
            assessment["route_resolution_state"].eq("LOW_ROUTE_SUPPORT").sum()
        ),
        "maximum_stock_weight_share": maximum_stock_share,
        "exact_pair_row_coverage": float(len(assessment) / assessment_reference_rows),
    }
    audit_gates = {
        "assessment_rows": coverage_evidence["assessment_rows"] >= 1_500,
        "assessment_sessions": coverage_evidence["assessment_sessions"] >= 50,
        "assessment_stocks": coverage_evidence["assessment_stocks"] >= 10,
        "assessment_months": coverage_evidence["assessment_months"] >= 5,
        "test_a_positives": coverage_evidence["test_a_positives"] >= 100,
        "test_b_positives": coverage_evidence["test_b_positives"] >= 300,
        "broad_conflict_rows": coverage_evidence["broad_conflict_rows"] >= 200,
        "low_route_support_rows": coverage_evidence["low_route_support_rows"] >= 200,
        "maximum_stock_weight_share": maximum_stock_share <= 0.15,
        "exact_pair_row_coverage": coverage_evidence["exact_pair_row_coverage"] >= 0.50,
    }
    require(
        "quick_support_gates",
        not any(audit_gates.values())
        and audit_gates == cast(dict[str, bool], decision["coverage_gates"]),
        evidence=coverage_evidence,
        gates=audit_gates,
    )

    model_csvs = (
        "test_a_metrics.csv",
        "test_a_monthly_metrics.csv",
        "test_a_subgroup_metrics.csv",
        "test_b_metrics.csv",
        "test_b_monthly_metrics.csv",
        "continuous_residual_metrics.csv",
        "bootstrap_metrics.csv",
        "options_null_metrics.csv",
        "route_null_metrics.csv",
    )
    nonempty_blocked_outputs = {
        name: len(pd.read_csv(PRIMARY / name))
        for name in model_csvs
        if len(pd.read_csv(PRIMARY / name)) != 0
    }
    predictions = pd.read_parquet(PRIMARY / "assessment_predictions.parquet")
    require(
        "blocked_model_outputs",
        not nonempty_blocked_outputs
        and predictions.empty
        and coefficients["models"] == {}
        and int(coefficients["primary_models_fitted"]) == 0
        and int(coefficients["ridge_models_fitted"]) == 0
        and int(model_config["primary_model_fits_run"]) == 0
        and int(model_config["ridge_fits_run"]) == 0,
        nonempty_outputs=nonempty_blocked_outputs,
        prediction_rows=len(predictions),
        fitted_models=[],
        manual_probability_reconstruction_rows_per_fitted_model={},
    )
    require(
        "bootstrap_and_null_designs",
        int(model_config["bootstrap_draws_planned"]) == 10
        and int(model_config["bootstrap_draws_run"]) == 0
        and int(model_config["bootstrap_seed"]) == BOOTSTRAP_SEED
        and model_config["bootstrap_interval_levels"] == [0.8, 0.9, 0.95]
        and int(model_config["options_null_refits_planned"]) == 3
        and int(model_config["options_null_refits_run"]) == 0
        and tuple(model_config["options_null_seeds"]) == OPTIONS_NULL_SEEDS
        and int(model_config["route_null_refits_planned"]) == 3
        and int(model_config["route_null_refits_run"]) == 0
        and tuple(model_config["route_null_seeds"]) == ROUTE_NULL_SEEDS
        and tuple(model_config["null_designs"]["options"]["permuted_as_one_bundle"])
        == OPTIONS_MODEL_FEATURES
        and tuple(model_config["null_designs"]["options"]["recomputed"]) == CROSS_MARKET_FEATURES
        and tuple(model_config["null_designs"]["route"]["permuted_as_one_bundle"])
        == (*ROUTE_FEATURES, "route_resolution_state")
        and tuple(model_config["null_designs"]["route"]["recomputed"]) == CROSS_MARKET_FEATURES,
        bootstrap_draws_planned=10,
        bootstrap_draws_run=0,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_interval_levels=[0.8, 0.9, 0.95],
        options_null_refits_planned=3,
        options_null_refits_run=0,
        options_null_seeds=list(OPTIONS_NULL_SEEDS),
        route_null_refits_planned=3,
        route_null_refits_run=0,
        route_null_seeds=list(ROUTE_NULL_SEEDS),
        reason=BLOCKER,
    )

    route_expected, route_contrast = route_state_movement_metrics(assessment)
    for key, value in route_contrast.items():
        route_expected[key] = value
    route_observed = pd.read_csv(PRIMARY / "test_b_route_state_metrics.csv")
    route_expected = route_expected.sort_values("route_state", kind="mergesort").reset_index(
        drop=True
    )
    route_observed = route_observed.sort_values("route_state", kind="mergesort").reset_index(
        drop=True
    )
    route_numeric = [column for column in route_expected.columns if column != "route_state"]
    route_difference = maximum_numeric_difference(
        route_expected,
        route_observed,
        route_numeric,
    )
    require(
        "route_state_residual_statistics",
        route_expected["route_state"].tolist() == route_observed["route_state"].tolist()
        and route_difference <= 1e-12,
        maximum_difference=route_difference,
        broad_minus_low=route_contrast,
    )

    expected_decision = {
        "decision": BLOCKER,
        "blocker": BLOCKER,
        "test_a_options_to_stock_status": "insufficient_support",
        "test_a_disagreement_status": "insufficient_support",
        "test_b_stock_to_options_status": "insufficient_support",
        "test_b_route_increment_status": "insufficient_support",
        "binding_questions_answered": False,
        "models_fit": [],
        "bootstrap_draws_run": 0,
        "options_null_refits_run": 0,
        "route_null_refits_run": 0,
    }
    decision_mismatches = {
        key: {"expected": expected, "observed": decision.get(key)}
        for key, expected in expected_decision.items()
        if decision.get(key) != expected
    }
    require(
        "decision_logic",
        not decision_mismatches,
        mismatches=decision_mismatches,
    )
    require(
        "determinism",
        determinism.get("passed") is True
        and int(determinism["selected_contract_mismatches"]) == 0
        and int(determinism["joined_row_mismatches"]) == 0
        and float(determinism["maximum_feature_difference"]) <= 1e-12
        and float(determinism["maximum_probability_difference"]) <= 1e-12
        and int(determinism["decision_mismatches"]) == 0,
        selected_contract_mismatches=determinism["selected_contract_mismatches"],
        joined_row_mismatches=determinism["joined_row_mismatches"],
        maximum_feature_difference=determinism["maximum_feature_difference"],
        maximum_probability_difference=determinism["maximum_probability_difference"],
    )

    return {
        **SAFETY_FLAGS,
        "passed": True,
        "status": "passed",
        "decision": BLOCKER,
        "checks": checks,
        "fitted_models_audited": 0,
        "manual_probability_reconstruction_rows_per_fitted_model": {},
        "model_metric_rows_audited": 0,
        "ridge_models_audited": 0,
        "bootstrap_draws_audited": 0,
        "options_null_refits_audited": 0,
        "route_null_refits_audited": 0,
        "blocked_outputs_verified_empty": True,
        "note": (
            "The support gate bound before fitting; model coefficients, manual "
            "probabilities, proper-score metrics, Ridge diagnostics, bootstrap "
            "draws, and null refits are therefore correctly absent."
        ),
    }


def fail_closed(error: Exception) -> None:
    """Persist the audit failure and replace the scientific decision."""

    audit_result = {
        **SAFETY_FLAGS,
        "passed": False,
        "status": AUDIT_BLOCKER,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    write_json(PRIMARY / "lightweight_audit.json", audit_result)
    decision_path = PRIMARY / "decision.json"
    decision = read_json(decision_path)
    decision.update(
        {
            "decision": AUDIT_BLOCKER,
            "blocker": AUDIT_BLOCKER,
            "binding_questions_answered": False,
            "test_a_options_to_stock_status": "blocked",
            "test_a_disagreement_status": "blocked",
            "test_b_stock_to_options_status": "blocked",
            "test_b_route_increment_status": "blocked",
        }
    )
    write_json(decision_path, decision)


def main() -> int:
    """Run the audit and fail closed."""

    try:
        result = audit()
    except Exception as error:
        fail_closed(error)
        print(f"{AUDIT_BLOCKER}: {error}")
        return 1
    write_json(PRIMARY / "lightweight_audit.json", result)
    report = (PRIMARY / "report.md").read_text(encoding="utf-8")
    report = report.replace(
        "Independent lightweight audit: `pending`.",
        "Independent lightweight audit: `passed`.",
    )
    (PRIMARY / "report.md").write_text(report, encoding="utf-8")
    (EXPERIMENT_DIR / "reports" / "report.md").write_text(
        report,
        encoding="utf-8",
    )
    print("passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
