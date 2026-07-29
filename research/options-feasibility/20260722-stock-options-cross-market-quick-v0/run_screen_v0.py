#!/usr/bin/env python3
"""Run the bounded Stock/Options Cross-Market Information Quick Screen V0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
PREDECESSOR_PRIMARY = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts"
    / "primary"
)
DENSE_PANEL = PREDECESSOR_PRIMARY / "dense_advance_panel.parquet"
TRACE_PANEL = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-route-competition-hazard-quick-v0"
    / "artifacts"
    / "primary"
    / "causal_state_trace.parquet"
)
PRIOR_OPTIONS_DIR = (
    REPO_ROOT / "research" / "options-feasibility" / "20260722-broad-conflict-prior-close-iv-v0"
)
PRICE_AUDIT = PRIOR_OPTIONS_DIR / "artifacts" / "primary" / "option_underlying_price_audit.csv"
PRIOR_REQUEST_PLAN = PRIOR_OPTIONS_DIR / "artifacts" / "primary" / "options_request_plan.json"
SCHEMA_MAPPING = PRIOR_OPTIONS_DIR / "artifacts" / "primary" / "eodhd_options_schema_mapping.json"
PROBE_DIR = (
    REPO_ROOT
    / "research"
    / "options-feasibility"
    / "20260722-broad-conflict-prior-close-iv-v01-probe"
)
PROBE_RESULTS = PROBE_DIR / "artifacts" / "primary" / "contract_history_probe_results.csv"
PROBE_MANIFEST = PROBE_DIR / "artifacts" / "primary" / "contract_history_probe_manifest.json"
PROBE_AUDIT = PROBE_DIR / "artifacts" / "primary" / "lightweight_audit.json"
PROBE_SCRIPT = PROBE_DIR / "contract_history_probe.py"
OPTIONS_CACHE = REPO_ROOT / "data" / "vendor" / "eodhd" / "options" / "contract-history-probe-v01"

for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))

from stocker_research.stock_options_cross_market_quick_v0 import (  # noqa: E402
    BASE_OPTIONS_FEATURES,
    BOOTSTRAP_SEED,
    CROSS_MARKET_FEATURES,
    DENSE_H0_FEATURES,
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
    add_cross_market_disagreement,
    add_test_b_target,
    apply_stock_relative_options,
    assert_protected_dates,
    assert_safety_flags,
    calculate_optional_option_features,
    calculate_primary_option_features,
    choose_overall_decision,
    compute_underlying_movement_outcomes,
    coverage_gates,
    extract_exact_history_records,
    fit_cross_market_standardization,
    fit_stock_relative_options,
    reconstruct_clean_structural_panel,
    route_state_movement_metrics,
    select_primary_atm_pair,
    trailing_realised_volatility_20d,
    validate_individual_statuses,
)

BLOCKER = "blocked_insufficient_cached_options_coverage"
STARTING_BRANCH = "agent/options-prior-close-iv-screen-v0"
STARTING_SHA = "b0ad90a0ee01635877ab1bc2ddec6c751feca26e"
FINAL_BRANCH = "agent/stock-options-cross-market-quick-v0"


def sha256_file(path: Path) -> str:
    """Hash a source file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic credential-free JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministic CSV atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write a deterministic Parquet artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _load_probe_module() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "stock_options_cross_market_probe_source", PROBE_SCRIPT
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen probe module: {PROBE_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _local_cache_path(manifest_row: Mapping[str, Any]) -> Path:
    endpoint = str(manifest_row["endpoint"])
    name = Path(str(manifest_row["cache_path"])).name
    if endpoint.endswith("/contracts"):
        return OPTIONS_CACHE / "raw" / "contract-discovery" / name
    if endpoint.endswith("/eod"):
        return OPTIONS_CACHE / "contract-histories" / "raw" / name
    raise ValueError(f"unsupported cached options endpoint: {endpoint}")


def _normalise_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text not in {"true", "false"}:
        raise ValueError(f"not a boolean value: {value}")
    return text == "true"


def rebuild_cached_pairs() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild every sampled exact-date ATM pair from raw cached provider records."""

    probe = _load_probe_module()
    expected = pd.read_csv(PROBE_RESULTS)
    manifest = read_json(PROBE_MANIFEST)
    audit = read_json(PROBE_AUDIT)
    if not audit.get("passed"):
        raise RuntimeError("blocked_options_pair_reconstruction_failure")
    rows = cast(list[dict[str, Any]], manifest["manifest_rows"])
    discovery_rows = [row for row in rows if str(row["endpoint"]).endswith("/contracts")]
    history_rows = {
        str(row["contract_id"]): row for row in rows if str(row["endpoint"]).endswith("/eod")
    }
    raw_file_hash_failures = 0
    local_cache_files: list[dict[str, Any]] = []
    for row in [*discovery_rows, *history_rows.values()]:
        path = _local_cache_path(row)
        observed_hash = sha256_file(path)
        raw_file_hash_failures += int(observed_hash != str(row["response_hash"]))
        local_cache_files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": observed_hash,
                "record_count": int(row["record_count"]),
                "endpoint": str(row["endpoint"]),
            }
        )
    if raw_file_hash_failures:
        raise RuntimeError("blocked_options_pair_reconstruction_failure")

    result_rows: list[dict[str, Any]] = []
    history_cache: dict[tuple[str, str], Any] = {}
    history_records_scanned = 0
    history_records_materialised = 0
    history_records_skipped_nonmatching = 0
    protected_history_records_skipped = 0
    for expected_row in expected.itertuples(index=False):
        symbol = str(expected_row.symbol)
        signal_date = date.fromisoformat(str(expected_row.signal_date))
        required_date = date.fromisoformat(str(expected_row.required_options_date))
        previous_close = float(expected_row.previous_close_underlying_price)
        target = probe.ProbeTarget(symbol, signal_date, required_date, previous_close)
        matching_discovery = [
            row
            for row in discovery_rows
            if str(row["underlying_symbol"]) == symbol
            and str(row["expiration_from"]) == (required_date + timedelta(days=7)).isoformat()
        ]
        if len(matching_discovery) != 1:
            raise RuntimeError("blocked_options_pair_reconstruction_failure")
        discovery_row = matching_discovery[0]
        discovery_payload = read_json(_local_cache_path(discovery_row))
        data = discovery_payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("blocked_options_pair_reconstruction_failure")
        descriptors = tuple(probe._contract_descriptor(item, target) for item in data)

        def load_history(
            contract_id: str,
            *,
            _required_date: date = required_date,
        ) -> Any:
            nonlocal history_records_materialised
            nonlocal history_records_scanned
            nonlocal history_records_skipped_nonmatching
            nonlocal protected_history_records_skipped
            if contract_id not in history_rows:
                raise RuntimeError("blocked_options_pair_reconstruction_failure")
            cache_key = (contract_id, _required_date.isoformat())
            if cache_key in history_cache:
                return history_cache[cache_key]
            history_row = history_rows[contract_id]
            extraction = extract_exact_history_records(
                _local_cache_path(history_row).read_text(encoding="utf-8"),
                required_date=_required_date,
            )
            if extraction.cached_records_scanned != int(history_row["record_count"]):
                raise RuntimeError("blocked_options_pair_reconstruction_failure")
            history_records_scanned += extraction.cached_records_scanned
            history_records_materialised += len(extraction.records)
            history_records_skipped_nonmatching += extraction.nonmatching_records_skipped
            protected_history_records_skipped += (
                extraction.protected_records_skipped_before_materialisation
            )
            history = probe.ContractHistory(
                contract_id,
                str(history_row["request_id"]),
                extraction.records,
            )
            history_cache[cache_key] = history
            return history

        pair = probe.select_probe_pair(
            target=target,
            contracts=descriptors,
            load_history=load_history,
        )
        features: dict[str, Any] = {}
        if pair.available:
            chain = pd.DataFrame(pair.exact_records)
            selection = select_primary_atm_pair(chain, previous_close=previous_close)
            features.update(
                calculate_primary_option_features(selection, previous_close=previous_close)
            )
            features.update(
                calculate_optional_option_features(
                    chain,
                    front_selection=selection,
                    previous_close=previous_close,
                )
            )
        result_rows.append(
            {
                "symbol": symbol,
                "signal_date": signal_date.isoformat(),
                "required_options_date": required_date.isoformat(),
                "previous_close_underlying_price": previous_close,
                "contracts_discovered": len(descriptors),
                "exact_observation_rows": len(pair.exact_records),
                "pair_available": bool(pair.available),
                "reason": str(pair.reason),
                "selected_expiry": (
                    None
                    if pair.selected_expiration_date is None
                    else pair.selected_expiration_date.isoformat()
                ),
                "selected_strike": pair.selected_strike,
                "call_contract_id": pair.call_contract_id,
                "put_contract_id": pair.put_contract_id,
                "front_dte": features.get("front_dte"),
                "atm_iv": features.get("atm_iv"),
                "call_iv": features.get("call_iv"),
                "put_iv": features.get("put_iv"),
                "call_put_iv_gap": features.get("call_put_iv_gap"),
                "straddle_mid_pct": features.get("straddle_mid_pct"),
                "combined_relative_spread": features.get("combined_relative_spread"),
                "log1p_combined_open_interest": features.get("log1p_combined_open_interest"),
                "atm_log_moneyness": features.get("atm_log_moneyness"),
                "skew_25d": features.get("skew_25d"),
                "skew_25d_missing": features.get("skew_25d_missing"),
                "term_structure": features.get("term_structure"),
                "term_structure_missing": features.get("term_structure_missing"),
                "history_contract_ids_requested": ";".join(pair.histories_requested),
                "canonical_rejections": int(pair.canonical_rejections),
            }
        )

    rebuilt = (
        pd.DataFrame(result_rows)
        .sort_values(["symbol", "signal_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected_sorted = expected.sort_values(["symbol", "signal_date"], kind="mergesort").reset_index(
        drop=True
    )
    identity_columns = [
        "pair_available",
        "reason",
        "selected_expiry",
        "selected_strike",
        "call_contract_id",
        "put_contract_id",
    ]
    selected_contract_mismatches = 0
    for column in identity_columns:
        left = expected_sorted[column].fillna("<missing>").astype(str)
        right = rebuilt[column].fillna("<missing>").astype(str)
        selected_contract_mismatches += int(left.ne(right).sum())
    numeric_differences: list[float] = []
    for column in ("front_dte", "atm_iv"):
        left = pd.to_numeric(expected_sorted[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(rebuilt[column], errors="coerce").to_numpy(float)
        difference = np.abs(left - right)
        difference[np.isnan(left) & np.isnan(right)] = 0.0
        numeric_differences.extend(difference[np.isfinite(difference)].tolist())
        selected_contract_mismatches += int(np.logical_xor(np.isnan(left), np.isnan(right)).sum())
    maximum_feature_difference = max(numeric_differences, default=0.0)
    reconstruction = {
        **SAFETY_FLAGS,
        "passed": bool(
            selected_contract_mismatches == 0
            and maximum_feature_difference <= 1e-12
            and raw_file_hash_failures == 0
            and history_records_scanned == int(manifest["raw_history_records"])
            and history_records_materialised == len(history_rows)
        ),
        "cached_vendor_records_used": int(manifest["raw_records"]),
        "cached_contract_discovery_records_used": int(manifest["raw_contract_records"]),
        "cached_contract_history_records_used": int(manifest["raw_history_records"]),
        "cached_contract_history_records_scanned_as_text": int(history_records_scanned),
        "cached_contract_history_records_materialised": int(history_records_materialised),
        "cached_contract_history_records_skipped_nonmatching": int(
            history_records_skipped_nonmatching
        ),
        "protected_cached_history_records_skipped_before_materialisation": int(
            protected_history_records_skipped
        ),
        "protected_history_records_materialised": 0,
        "cached_stock_dates": int(len(rebuilt)),
        "cached_symbols": sorted(rebuilt["symbol"].astype(str).unique().tolist()),
        "exact_previous_close_chains": int(rebuilt["exact_observation_rows"].gt(0).sum()),
        "valid_primary_pairs": int(rebuilt["pair_available"].astype(bool).sum()),
        "selected_contract_mismatches": int(selected_contract_mismatches),
        "maximum_option_feature_difference": float(maximum_feature_difference),
        "raw_file_hash_failures": int(raw_file_hash_failures),
        "same_day_or_future_options_rows": int(
            (
                pd.to_datetime(rebuilt["required_options_date"])
                >= pd.to_datetime(rebuilt["signal_date"])
            ).sum()
        ),
        "frozen_selection_rule": {
            "expiry": "nearest expiry satisfying 7 <= DTE <= 45",
            "strike": (
                "common call/put strike minimizing "
                "abs(log(strike / previous_close_underlying_price))"
            ),
            "tie_break": [
                "abs_log_moneyness ascending",
                "minimum_pair_open_interest descending",
                "combined_relative_spread ascending",
                "absolute_call_put_iv_gap ascending",
                "strike ascending",
                "call_contract_id ascending",
                "put_contract_id ascending",
            ],
            "pair_quality": "frozen predecessor requirements",
            "fallback_after_quality_failure": False,
        },
        "network_requests_made": 0,
        "redownloaded": False,
        "raw_cache_files": local_cache_files,
        "optional_feature_scope_note": (
            "The bounded cache stores selected ATM histories only; skew and term structure "
            "are marked missing rather than inferred."
        ),
    }
    if not reconstruction["passed"]:
        raise RuntimeError("blocked_options_pair_reconstruction_failure")
    return rebuilt, reconstruction


def build_cross_market_panel() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Rebuild structural rows, cached pairs, outcomes, and fixed cross-market features."""

    dense = pd.read_parquet(DENSE_PANEL)
    structural, structural_reconstruction = reconstruct_clean_structural_panel(dense)
    pairs, pair_reconstruction = rebuild_cached_pairs()
    available_pairs = pairs.loc[pairs["pair_available"].astype(bool)].copy()
    pair_columns = [
        "symbol",
        "signal_date",
        "required_options_date",
        "previous_close_underlying_price",
        "selected_expiry",
        "selected_strike",
        "call_contract_id",
        "put_contract_id",
        *BASE_OPTIONS_FEATURES,
    ]
    joined = structural.merge(
        available_pairs.loc[:, pair_columns],
        left_on=["symbol", "session"],
        right_on=["symbol", "signal_date"],
        how="inner",
        validate="many_to_one",
    )
    trace = pd.read_parquet(TRACE_PANEL)
    trace["session"] = trace["session"].astype(str)
    joined["session"] = joined["session"].astype(str)
    movement = compute_underlying_movement_outcomes(joined, trace)
    panel = add_test_b_target(movement)
    daily_volatility = trailing_realised_volatility_20d(
        trace.sort_values(["symbol", "session", "bar_ordinal"], kind="mergesort")
    )
    daily_volatility["required_options_date"] = daily_volatility["required_options_date"].astype(
        str
    )
    panel["required_options_date"] = panel["required_options_date"].astype(str)
    panel = panel.merge(
        daily_volatility,
        on=["symbol", "required_options_date"],
        how="left",
        validate="many_to_one",
    )
    panel["iv_minus_realised_20d"] = pd.to_numeric(panel["atm_iv"], errors="raise") - pd.to_numeric(
        panel["realised_volatility_20d"], errors="coerce"
    )
    development = panel.loc[panel["period"].eq("development")]
    stock_parameters = fit_stock_relative_options(development)
    panel = apply_stock_relative_options(panel, stock_parameters)
    standardization = fit_cross_market_standardization(panel.loc[panel["period"].eq("development")])
    panel = add_cross_market_disagreement(panel, standardization)
    assert_protected_dates(
        panel,
        columns=("session", "required_options_date"),
    )
    panel = panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    feature_manifest = {
        **SAFETY_FLAGS,
        "base_options_features": list(BASE_OPTIONS_FEATURES),
        "stock_relative_options_features": list(STOCK_RELATIVE_OPTIONS_FEATURES),
        "options_model_features": list(OPTIONS_MODEL_FEATURES),
        "cross_market_disagreement_features": list(CROSS_MARKET_FEATURES),
        "frozen_h0_features": list(DENSE_H0_FEATURES),
        "frozen_route_features": list(ROUTE_FEATURES),
        "development_only_stock_relative_parameters": {
            symbol: {feature: asdict(parameters) for feature, parameters in values.items()}
            for symbol, values in stock_parameters.items()
        },
        "development_only_standardization": {
            feature: asdict(parameters) for feature, parameters in standardization.items()
        },
        "realised_volatility": {
            "returns": "daily close-to-close log returns",
            "trailing_sessions": 20,
            "minimum_valid_sessions": 15,
            "annualization": "sqrt(252)",
            "information_time": "required previous options-session close",
        },
        "disagreement_formulas": {
            "complacent_conflict": "BROAD_CONFLICT * (-atm_iv_stock_robust_z)",
            "structural_tension_gap": ("standardised_tension - atm_iv_stock_robust_z"),
            "route_vs_priced_move": (
                "standardised_prefix_family_entropy - straddle_move_stock_robust_z"
            ),
            "directional_agreement": (
                "standardised_signed_pressure * standardised_call_put_iv_gap"
            ),
            "transition_vs_term_urgency": (
                "standardised_transition_probability - term_structure_stock_robust_z"
            ),
        },
        "fit_period": "2024 development only",
        "feature_search": False,
        "interaction_search": False,
    }
    return (
        panel,
        pairs,
        structural_reconstruction,
        pair_reconstruction,
        feature_manifest,
    )


def build_coverage_gap(
    structural: pd.DataFrame, pairs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write the exact stock/date/month cached-data gap and request estimates."""

    price = pd.read_csv(PRICE_AUDIT)
    price["signal_date"] = price["signal_date"].astype(str)
    price["required_options_date"] = price["required_options_date"].astype(str)
    price["split_boundary_ambiguous"] = price["split_boundary_ambiguous"].map(_normalise_bool)
    pair_status = pairs[
        [
            "symbol",
            "signal_date",
            "exact_observation_rows",
            "pair_available",
            "reason",
        ]
    ].copy()
    pair_status["exact_chain_cached"] = pair_status["exact_observation_rows"].gt(0)
    pair_status["valid_atm_pair_cached"] = pair_status["pair_available"].astype(bool)
    structural_counts = (
        structural.groupby(["period", "symbol", "session"], sort=True)
        .agg(
            eligible_structural_rows=("row_id", "size"),
            test_a_positives=("registered_completion_clean_bars_2_or_3", "sum"),
        )
        .reset_index()
        .rename(columns={"session": "signal_date"})
    )
    structural_counts["signal_date"] = structural_counts["signal_date"].astype(str)
    gap = price.merge(
        pair_status.drop(columns=["exact_observation_rows", "pair_available"]),
        on=["symbol", "signal_date"],
        how="left",
        validate="one_to_one",
    ).merge(
        structural_counts,
        on=["symbol", "signal_date"],
        how="left",
        validate="one_to_one",
    )
    period_fallback = pd.Series(
        np.where(
            pd.to_datetime(gap["signal_date"]).dt.year.eq(2024),
            "development",
            "assessment",
        ),
        index=gap.index,
    )
    gap["period"] = gap["period"].fillna(period_fallback)
    gap["year_month"] = pd.to_datetime(gap["signal_date"]).dt.strftime("%Y-%m")
    gap["eligible_structural_rows"] = gap["eligible_structural_rows"].fillna(0).astype(int)
    gap["test_a_positives"] = gap["test_a_positives"].fillna(0).astype(int)
    gap["exact_chain_cached"] = (
        gap["exact_chain_cached"].astype("boolean").fillna(False).astype(bool)
    )
    gap["valid_atm_pair_cached"] = (
        gap["valid_atm_pair_cached"].astype("boolean").fillna(False).astype(bool)
    )
    gap["reason"] = gap["reason"].fillna("missing_cached_exact_chain")
    gap["coverage_status"] = np.select(
        [
            gap["split_boundary_ambiguous"],
            gap["valid_atm_pair_cached"],
            gap["exact_chain_cached"],
        ],
        [
            "excluded_split_boundary",
            "cached_valid_atm_pair",
            "cached_pair_quality_failure",
        ],
        default="missing_cached_exact_chain",
    )
    request_eligible = ~gap["split_boundary_ambiguous"]
    missing_chain = request_eligible & ~gap["exact_chain_cached"]
    gap["eodhd_contract_discovery_requests_estimate"] = np.where(missing_chain, 1, 0)
    gap["eodhd_contract_history_requests_estimate"] = np.where(missing_chain, 6, 0)
    gap["eodhd_total_requests_estimate"] = (
        gap["eodhd_contract_discovery_requests_estimate"]
        + gap["eodhd_contract_history_requests_estimate"]
    )
    discovery_records_per_target = 2.0 * 3_120 / 9
    history_records_per_contract = 504 / 18
    gap["estimated_additional_records"] = np.where(
        missing_chain,
        np.ceil(discovery_records_per_target + 6 * history_records_per_contract),
        0,
    ).astype(int)
    frozen_bulk_plan = read_json(PRIOR_REQUEST_PLAN)
    request_gap = {
        "eligible_stock_dates": int(request_eligible.sum()),
        "split_boundary_exclusions": int(gap["split_boundary_ambiguous"].sum()),
        "cached_exact_chain_stock_dates": int(gap["exact_chain_cached"].sum()),
        "cached_valid_atm_pair_stock_dates": int(gap["valid_atm_pair_cached"].sum()),
        "missing_exact_chain_stock_dates": int(missing_chain.sum()),
        "cached_pair_quality_failures": int(
            (gap["exact_chain_cached"] & ~gap["valid_atm_pair_cached"]).sum()
        ),
        "exact_missing_targets_manifest": "options_coverage_gap.csv",
        "viable_endpoints": [
            "/mp/unicornbay/options/contracts",
            "/mp/unicornbay/options/eod with filter[contract]",
        ],
        "estimated_additional_records": int(gap["estimated_additional_records"].sum()),
        "estimated_additional_requests": int(gap["eodhd_total_requests_estimate"].sum()),
        "estimate_assumptions": {
            "contract_discovery_window_dte": "7-90",
            "history_contracts_per_stock_date": 6,
            "history_contract_roles": (
                "front ATM call/put, front approximately-25-delta call/put, back ATM call/put"
            ),
            "records_calibrated_from_cached_probe": True,
            "quality_failure_fallback_allowed": False,
        },
        "frozen_bulk_plan_not_executed": {
            "symbol_month_chunks": int(frozen_bulk_plan["symbol_month_chunks"]),
            "estimated_records": int(frozen_bulk_plan["estimated_records"]),
            "estimated_requests": int(frozen_bulk_plan["estimated_requests"]),
            "reason_not_executed": (
                "historical observation-date filtering is unavailable on the bulk endpoint"
            ),
        },
        "new_downloads_this_experiment": 0,
    }
    return (
        gap.sort_values(["period", "year_month", "symbol", "signal_date"], kind="mergesort"),
        request_gap,
    )


def coverage_evidence(
    structural: pd.DataFrame, panel: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate every quick support and concentration gate."""

    assessment_structural = structural.loc[structural["period"].eq("assessment")]
    assessment = panel.loc[panel["period"].eq("assessment")].copy()
    stock_weights = assessment.groupby("symbol", sort=True)["row_weight"].sum().rename("weight_sum")
    total_weight = float(stock_weights.sum())
    maximum_share = math.inf if total_weight <= 0.0 else float((stock_weights / total_weight).max())
    evidence = {
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
        "maximum_stock_weight_share": maximum_share,
        "exact_pair_row_coverage": (
            0.0
            if assessment_structural.empty
            else float(len(assessment) / len(assessment_structural))
        ),
        "eligible_assessment_structural_rows": int(len(assessment_structural)),
    }
    passed, gates = coverage_gates(evidence)
    evidence["passed"] = passed
    evidence["gates"] = gates
    concentration_rows: list[dict[str, Any]] = []
    for period, period_frame in panel.groupby("period", sort=True):
        period_total = float(period_frame["row_weight"].sum())
        for symbol, group in period_frame.groupby("symbol", sort=True):
            weight = float(group["row_weight"].sum())
            concentration_rows.append(
                {
                    "period": str(period),
                    "symbol": str(symbol),
                    "rows": int(len(group)),
                    "sessions": int(group["session"].nunique()),
                    "weight_sum": weight,
                    "weighted_joined_row_share": weight / period_total,
                }
            )
    return evidence, pd.DataFrame(concentration_rows)


def coverage_summary(
    gap: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarise the exact stock-date manifest without losing its row-level path."""

    grouped = (
        gap.groupby(list(group_columns), sort=True, dropna=False)
        .agg(
            stock_dates=("signal_date", "size"),
            exact_chains_cached=("exact_chain_cached", "sum"),
            valid_atm_pairs_cached=("valid_atm_pair_cached", "sum"),
            eligible_structural_rows=("eligible_structural_rows", "sum"),
            test_a_positives=("test_a_positives", "sum"),
            estimated_additional_requests=("eodhd_total_requests_estimate", "sum"),
            estimated_additional_records=("estimated_additional_records", "sum"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for source in grouped.to_dict(orient="records"):
        row: dict[str, Any] = {column: str(source[column]) for column in group_columns}
        for column in (
            "stock_dates",
            "exact_chains_cached",
            "valid_atm_pairs_cached",
            "eligible_structural_rows",
            "test_a_positives",
            "estimated_additional_requests",
            "estimated_additional_records",
        ):
            row[column] = int(source[column])
        rows.append(row)
    return rows


BINARY_METRIC_COLUMNS = [
    "model",
    "log_loss",
    "brier_score",
    "auc",
    "average_precision",
    "expected_calibration_error",
    "calibration_intercept",
    "calibration_slope",
    "base_rate",
    "mean_probability_realised_class",
    "top_decile_precision",
    "top_decile_lift",
    "top_quintile_precision",
    "top_quintile_lift",
    "rows",
    "sessions",
    "stocks",
    "positive_outcomes",
]


def model_configurations() -> dict[str, Any]:
    """Declare all fixed model surfaces even when support blocks fitting."""

    return {
        **SAFETY_FLAGS,
        "primary_model_fits_allowed": 6,
        "primary_model_fits_run": 0,
        "ridge_fits_allowed": 2,
        "ridge_fits_run": 0,
        "model": {
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
        },
        "ridge": {"alpha": 10.0, "solver": "cholesky"},
        "models": {
            "S0": {
                "numeric_features": list(TEST_A_S0_NUMERIC),
                "categorical_controls": ["stock"],
                "target": "registered_completion_clean_bars_2_or_3",
            },
            "S1": {
                "numeric_features": list(TEST_A_S1_NUMERIC),
                "categorical_controls": ["stock"],
                "target": "registered_completion_clean_bars_2_or_3",
            },
            "S2": {
                "numeric_features": list(TEST_A_S2_NUMERIC),
                "categorical_controls": ["stock"],
                "target": "registered_completion_clean_bars_2_or_3",
            },
            "O0": {
                "numeric_features": list(TEST_B_O0_NUMERIC),
                "categorical_controls": ["stock", "checkpoint", "month_of_year"],
                "target": "movement_exceeds_prior_close_iv",
            },
            "O1": {
                "numeric_features": list(TEST_B_O1_NUMERIC),
                "categorical_controls": ["stock", "checkpoint", "month_of_year"],
                "target": "movement_exceeds_prior_close_iv",
            },
            "O2": {
                "numeric_features": list(TEST_B_O2_NUMERIC),
                "categorical_controls": [
                    "stock",
                    "checkpoint",
                    "month_of_year",
                    "route_state",
                ],
                "target": "movement_exceeds_prior_close_iv",
            },
            "R0": {
                "numeric_features": list(RIDGE_R0_NUMERIC),
                "categorical_controls": [],
                "target": "iv_absolute_residual_15m",
            },
            "R1": {
                "numeric_features": list(RIDGE_R1_NUMERIC),
                "categorical_controls": ["route_state"],
                "target": "iv_absolute_residual_15m",
            },
        },
        "preprocessing_fit_period": "2024 development only",
        "probability_quantiles_fit_period": "2024 development predictions only",
        "bootstrap_draws_planned": 10,
        "bootstrap_draws_run": 0,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_interval_levels": [0.80, 0.90, 0.95],
        "options_null_refits_planned": 3,
        "options_null_refits_run": 0,
        "options_null_seeds": list(OPTIONS_NULL_SEEDS),
        "route_null_refits_planned": 3,
        "route_null_refits_run": 0,
        "route_null_seeds": list(ROUTE_NULL_SEEDS),
        "null_designs": {
            "options": {
                "strata": ["period", "session", "checkpoint"],
                "permuted_as_one_bundle": list(OPTIONS_MODEL_FEATURES),
                "preserved": [
                    "stock structural features",
                    "outcomes",
                    "row weights",
                ],
                "recomputed": list(CROSS_MARKET_FEATURES),
            },
            "route": {
                "strata": ["period", "session", "checkpoint"],
                "permuted_as_one_bundle": [
                    *ROUTE_FEATURES,
                    "route_resolution_state",
                ],
                "preserved": [
                    "options features",
                    "H0 features",
                    "outcomes",
                    "row weights",
                ],
                "recomputed": list(CROSS_MARKET_FEATURES),
            },
        },
        "not_run_reason": BLOCKER,
    }


def write_empty_model_artifacts() -> None:
    """Write schema-bearing empty outputs after the binding support blocker."""

    write_json(
        PRIMARY / "model_coefficients.json",
        {
            **SAFETY_FLAGS,
            "models": {},
            "primary_models_fitted": 0,
            "ridge_models_fitted": 0,
            "reason": BLOCKER,
        },
    )
    predictions = pd.DataFrame(
        columns=[
            "row_id",
            "symbol",
            "session",
            "checkpoint",
            "registered_completion_clean_bars_2_or_3",
            "movement_exceeds_prior_close_iv",
            "S0_probability",
            "S1_probability",
            "S2_probability",
            "O0_probability",
            "O1_probability",
            "O2_probability",
            "R0_prediction",
            "R1_prediction",
        ]
    )
    write_parquet(PRIMARY / "assessment_predictions.parquet", predictions)
    write_csv(PRIMARY / "test_a_metrics.csv", pd.DataFrame(columns=BINARY_METRIC_COLUMNS))
    write_csv(
        PRIMARY / "test_a_monthly_metrics.csv",
        pd.DataFrame(columns=["month", *BINARY_METRIC_COLUMNS]),
    )
    write_csv(
        PRIMARY / "test_a_subgroup_metrics.csv",
        pd.DataFrame(columns=["subgroup", "level", *BINARY_METRIC_COLUMNS]),
    )
    write_csv(PRIMARY / "test_b_metrics.csv", pd.DataFrame(columns=BINARY_METRIC_COLUMNS))
    write_csv(
        PRIMARY / "test_b_monthly_metrics.csv",
        pd.DataFrame(columns=["month", *BINARY_METRIC_COLUMNS]),
    )
    write_csv(
        PRIMARY / "continuous_residual_metrics.csv",
        pd.DataFrame(
            columns=[
                "model",
                "weighted_mae",
                "weighted_rmse",
                "weighted_r_squared",
                "mae_improvement",
                "rmse_improvement",
            ]
        ),
    )
    write_csv(
        PRIMARY / "bootstrap_metrics.csv",
        pd.DataFrame(
            columns=[
                "draw",
                "test",
                "comparison",
                "metric",
                "improvement",
                "interval_level",
                "lower",
                "upper",
            ]
        ),
    )
    write_csv(
        PRIMARY / "options_null_metrics.csv",
        pd.DataFrame(
            columns=[
                "seed",
                "comparison",
                "metric",
                "real_increment",
                "null_increment",
                "real_exceeds_null",
            ]
        ),
    )
    write_csv(
        PRIMARY / "route_null_metrics.csv",
        pd.DataFrame(
            columns=[
                "seed",
                "comparison",
                "metric",
                "real_increment",
                "null_increment",
                "real_exceeds_null",
            ]
        ),
    )


def _numeric_maximum_difference(
    expected: pd.DataFrame, observed: pd.DataFrame, columns: Sequence[str]
) -> float:
    maximum = 0.0
    for column in columns:
        left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
        right = pd.to_numeric(observed[column], errors="coerce").to_numpy(float)
        if np.logical_xor(np.isnan(left), np.isnan(right)).any():
            return math.inf
        difference = np.abs(left - right)
        difference[np.isnan(left) & np.isnan(right)] = 0.0
        finite = difference[np.isfinite(difference)]
        if finite.size:
            maximum = max(maximum, float(finite.max()))
    return maximum


def determinism_check(
    first_panel: pd.DataFrame,
    first_pairs: pd.DataFrame,
    first_decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Reload the cache and refit every model that ran (none after the support gate)."""

    (
        second_panel,
        second_pairs,
        second_structural,
        second_pair_reconstruction,
        _second_features,
    ) = build_cross_market_panel()
    first_panel_ordered = first_panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    second_panel_ordered = second_panel.sort_values("row_id", kind="mergesort").reset_index(
        drop=True
    )
    joined_row_mismatches = abs(len(first_panel_ordered) - len(second_panel_ordered)) + sum(
        left != right
        for left, right in zip(
            first_panel_ordered["row_id"].astype(str),
            second_panel_ordered["row_id"].astype(str),
            strict=False,
        )
    )
    pair_keys = ["symbol", "signal_date"]
    first_pairs_ordered = first_pairs.sort_values(pair_keys, kind="mergesort").reset_index(
        drop=True
    )
    second_pairs_ordered = second_pairs.sort_values(pair_keys, kind="mergesort").reset_index(
        drop=True
    )
    contract_columns = [
        "pair_available",
        "selected_expiry",
        "selected_strike",
        "call_contract_id",
        "put_contract_id",
    ]
    selected_contract_mismatches = abs(len(first_pairs_ordered) - len(second_pairs_ordered))
    if selected_contract_mismatches == 0:
        selected_contract_mismatches += int(
            first_pairs_ordered[contract_columns]
            .fillna("<missing>")
            .astype(str)
            .ne(second_pairs_ordered[contract_columns].fillna("<missing>").astype(str))
            .sum()
            .sum()
        )
    feature_columns = [
        *BASE_OPTIONS_FEATURES,
        *STOCK_RELATIVE_OPTIONS_FEATURES,
        "realised_volatility_20d",
        "iv_minus_realised_20d",
        *CROSS_MARKET_FEATURES,
    ]
    maximum_feature_difference = (
        math.inf
        if joined_row_mismatches
        else _numeric_maximum_difference(first_panel_ordered, second_panel_ordered, feature_columns)
    )
    repeated_decision = choose_overall_decision(
        blocker=BLOCKER,
        test_a_supported=False,
        test_b_supported=False,
        disagreement_descriptive=False,
    )
    decision_mismatches = int(repeated_decision != str(first_decision["decision"]))
    result = {
        **SAFETY_FLAGS,
        "passed": bool(
            selected_contract_mismatches == 0
            and joined_row_mismatches == 0
            and maximum_feature_difference <= 1e-12
            and decision_mismatches == 0
            and second_structural["passed"]
            and second_pair_reconstruction["passed"]
        ),
        "redownloaded": False,
        "network_requests_made": 0,
        "cached_options_records_reloaded": int(
            second_pair_reconstruction["cached_vendor_records_used"]
        ),
        "selected_contract_mismatches": int(selected_contract_mismatches),
        "joined_row_mismatches": int(joined_row_mismatches),
        "maximum_feature_difference": float(maximum_feature_difference),
        "models_refit": [],
        "maximum_coefficient_difference": 0.0,
        "maximum_probability_difference": 0.0,
        "metrics_mismatches": 0,
        "decision_mismatches": int(decision_mismatches),
        "bootstrap_repeated": False,
        "null_draws_repeated": False,
    }
    if not result["passed"]:
        raise RuntimeError("blocked_reproducibility_or_audit_failure")
    return result


def _report(
    *,
    panel: pd.DataFrame,
    pairs: pd.DataFrame,
    coverage: Mapping[str, Any],
    request_gap: Mapping[str, Any],
    route_metrics: pd.DataFrame,
    decision: Mapping[str, Any],
    structural_reconstruction: Mapping[str, Any],
    determinism: Mapping[str, Any],
    concentration: pd.DataFrame,
) -> str:
    assessment = panel.loc[panel["period"].eq("assessment")]
    assessment_months = sorted(
        pd.to_datetime(assessment["session"]).dt.strftime("%Y-%m").unique().tolist()
    )
    broad = route_metrics.loc[route_metrics["route_state"].eq("BROAD_CONFLICT")]
    low = route_metrics.loc[route_metrics["route_state"].eq("LOW_ROUTE_SUPPORT")]
    broad_text = (
        "unsupported"
        if broad.empty or not int(broad.iloc[0]["rows"])
        else (
            f"{int(broad.iloc[0]['rows'])} rows; mean residual "
            f"{float(broad.iloc[0]['mean_iv_residual']):.8f}; "
            f"exceed rate {float(broad.iloc[0]['exceed_iv_rate']):.4%}"
        )
    )
    low_text = (
        "unsupported"
        if low.empty or not int(low.iloc[0]["rows"])
        else (
            f"{int(low.iloc[0]['rows'])} rows; mean residual "
            f"{float(low.iloc[0]['mean_iv_residual']):.8f}; "
            f"exceed rate {float(low.iloc[0]['exceed_iv_rate']):.4%}"
        )
    )
    broad_low = (
        "unavailable"
        if broad.empty or low.empty
        else (
            f"mean residual {float(broad.iloc[0]['mean_iv_residual_difference']):.8f}; "
            f"median residual "
            f"{float(broad.iloc[0]['median_iv_residual_difference']):.8f}; "
            f"exceed-rate difference "
            f"{float(broad.iloc[0]['exceed_iv_rate_difference']):.4%}; "
            f"upper-decile residual "
            f"{float(broad.iloc[0]['upper_decile_iv_residual_difference']):.8f}"
        )
    )
    assessment_concentration = concentration.loc[concentration["period"].eq("assessment")]
    maximum_concentration = float(assessment_concentration["weighted_joined_row_share"].max())
    return f"""# Stock ↔ Options Cross-Market Information Quick Screen V0

Overall decision: `{decision["decision"]}`.

Research-only boundary: previous-close options only; no intraday option quotes,
option P&L, execution, broker integration, prospective validation, strategy
promotion, or production-runtime change.

The current canonical cache contains {int(decision["cached_options_records_used"]):,}
vendor records. Only
{int(decision["cached_history_records_materialised"]):,} exact-date history
observations were materialised;
{int(decision["protected_cached_history_records_skipped_before_materialisation"]):,}
cached observations at or beyond the protected boundary were skipped as raw JSON
text before row decoding. It reconstructs
{int(pairs["exact_observation_rows"].gt(0).sum())}
exact previous-close chains and {int(pairs["pair_available"].sum())} valid ATM pairs
across {pairs["symbol"].nunique()} symbols and {pairs["signal_date"].nunique()} signal
dates. Those pairs join {len(panel):,} frozen clean structural rows, including
{len(assessment):,} assessment rows from {assessment["session"].nunique()} session,
{assessment["symbol"].nunique()} stocks, and {len(assessment_months)} month
({", ".join(assessment_months)}).

The exact frozen structural reconstruction contains
{int(structural_reconstruction["clean_advance_rows"]):,} eligible rows:
{int(structural_reconstruction["development_clean_rows"]):,} development and
{int(structural_reconstruction["assessment_clean_rows"]):,} assessment, with
{int(structural_reconstruction["assessment_clean_positives"]):,} assessment
Test A positives. Row-identity and route-state mismatches are zero and the
maximum shared-feature difference is
{float(structural_reconstruction["maximum_shared_feature_difference"]):.1e}.
Protected rows materialised: 0.

Exact valid-pair assessment row coverage is
{float(coverage["exact_pair_row_coverage"]):.6%}, below the fixed 50% gate.
The joined assessment sample also misses the row, session, stock, month, Test A
positive, Test B positive, BROAD_CONFLICT, LOW_ROUTE_SUPPORT, and concentration
gates. Therefore S0/S1/S2, O0/O1/O2, R0/R1, the ten bootstrap draws, the three
options-null refits, and the three route-null refits were not run.

## Exact cache gap

- Missing exact-chain stock-dates: {int(request_gap["missing_exact_chain_stock_dates"]):,}.
- Cached pair-quality failures with no permitted fallback:
  {int(request_gap["cached_pair_quality_failures"]):,}.
- Estimated additional viable contract-discovery/history requests:
  {int(request_gap["estimated_additional_requests"]):,}.
- Estimated additional provider records:
  {int(request_gap["estimated_additional_records"]):,}.
- New requests or downloads in this experiment: 0.

The row-level stock/date/month request manifest is `options_coverage_gap.csv`.

## Test A

Options-to-stock status: `insufficient_support`.
Disagreement status: `insufficient_support`.
No S0, S1, or S2 metrics, monthly/subgroup results, increments, bootstrap
intervals, or options-feature null comparisons exist. The
binding question—whether previous-close options information improves the stock
system's clean two-to-three-bar registered-loop completion forecast—remains
unanswered.

## Test B

Stock-to-options-movement status: `insufficient_support`.
Route-increment status: `insufficient_support`.
No O0, O1, O2, R0, or R1 metrics, increments, monthly results, bootstrap
intervals, or route-feature null comparisons exist. The binding
question—whether compressed-transition and route-competition features improve
prediction that 15-minute underlying movement exceeds the previous-close
options expectation—remains unanswered.

The below-cache-threshold route outcomes are descriptive diagnostics only:

- BROAD_CONFLICT: {broad_text}.
- LOW_ROUTE_SUPPORT: {low_text}.
- BROAD_CONFLICT minus LOW_ROUTE_SUPPORT: {broad_low}.
- BROAD_CONFLICT top-5%-row positive-residual contribution:
  {float(broad.iloc[0]["top_5pct_positive_residual_contribution"]):.4%}.
- LOW_ROUTE_SUPPORT top-5%-row positive-residual contribution:
  {float(low.iloc[0]["top_5pct_positive_residual_contribution"]):.4%}.

## Concentration and reproducibility

Maximum assessment stock share of weighted joined rows:
{maximum_concentration:.4%}, above the 15% ceiling.

Determinism check: `{"passed" if determinism["passed"] else "failed"}`.
Selected-contract mismatches: {int(determinism["selected_contract_mismatches"])}.
Joined-row mismatches: {int(determinism["joined_row_mismatches"])}.
Maximum feature difference:
{float(determinism["maximum_feature_difference"]):.1e}.
Maximum probability difference:
{float(determinism["maximum_probability_difference"]):.1e}.

Independent lightweight audit: `pending`.

No result is option profitability, an intraday option fill, executable option
return, economic edge, prospective validation, trading utility, or a deployable
strategy.
"""


def run(output: Path = PRIMARY) -> dict[str, Any]:
    """Run the screen and emit the prescribed cached-coverage blocker."""

    global PRIMARY
    PRIMARY = output
    contract = read_json(EXPERIMENT_DIR / "contract.json")
    assert_safety_flags(contract)
    (
        panel,
        pairs,
        structural_reconstruction,
        pair_reconstruction,
        feature_manifest,
    ) = build_cross_market_panel()
    dense = pd.read_parquet(DENSE_PANEL)
    structural, _ = reconstruct_clean_structural_panel(dense)
    gap, request_gap = build_coverage_gap(structural, pairs)
    coverage, concentration = coverage_evidence(structural, panel)
    if bool(coverage["passed"]):
        raise RuntimeError(
            "cached support unexpectedly passed; this bounded blocker run must be reviewed"
        )

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "contract.json", contract)
    source_manifest = {
        **SAFETY_FLAGS,
        "starting_branch": STARTING_BRANCH,
        "starting_sha": STARTING_SHA,
        "final_branch": FINAL_BRANCH,
        "frozen_dense_panel": str(DENSE_PANEL.relative_to(REPO_ROOT)),
        "frozen_dense_panel_sha256": sha256_file(DENSE_PANEL),
        "frozen_trace_panel": str(TRACE_PANEL.relative_to(REPO_ROOT)),
        "frozen_trace_panel_sha256": sha256_file(TRACE_PANEL),
        "frozen_pair_ledger": str(PROBE_RESULTS.relative_to(REPO_ROOT)),
        "frozen_pair_ledger_sha256": sha256_file(PROBE_RESULTS),
        "options_schema_mapping": str(SCHEMA_MAPPING.relative_to(REPO_ROOT)),
        "options_schema_mapping_sha256": sha256_file(SCHEMA_MAPPING),
        "cached_options_records_used": int(pair_reconstruction["cached_vendor_records_used"]),
        "cached_history_records_materialised": int(
            pair_reconstruction["cached_contract_history_records_materialised"]
        ),
        "protected_cached_history_records_skipped_before_materialisation": int(
            pair_reconstruction["protected_cached_history_records_skipped_before_materialisation"]
        ),
        "cached_symbols": pair_reconstruction["cached_symbols"],
        "cached_stock_dates": int(pair_reconstruction["cached_stock_dates"]),
        "network_requests_made": 0,
        "bulk_options_downloads_made": 0,
        "request_gap": request_gap,
        "coverage_evidence": coverage,
        "coverage_by_stock": coverage_summary(gap, group_columns=("symbol",)),
        "coverage_by_date": coverage_summary(gap, group_columns=("signal_date",)),
        "coverage_by_month": coverage_summary(
            gap,
            group_columns=("period", "year_month"),
        ),
        "exact_stock_date_gap_manifest": "options_coverage_gap.csv",
    }
    boundary_audit = {
        **SAFETY_FLAGS,
        "passed": True,
        "protected_start": "2025-08-23",
        "maximum_structural_session": str(structural["session"].max()),
        "maximum_joined_session": str(panel["session"].max()),
        "maximum_options_date": str(panel["required_options_date"].max()),
        "maximum_selected_expiry": str(panel["selected_expiry"].max()),
        "selected_expiry_is_contract_metadata_not_an_observation_date": True,
        "protected_cached_history_records_skipped_before_materialisation": int(
            pair_reconstruction["protected_cached_history_records_skipped_before_materialisation"]
        ),
        "protected_history_records_materialised": int(
            pair_reconstruction["protected_history_records_materialised"]
        ),
        "protected_rows_materialised": int(
            pair_reconstruction["protected_history_records_materialised"]
        ),
        "same_day_or_future_options_joins": int(
            (
                pd.to_datetime(panel["required_options_date"]) >= pd.to_datetime(panel["session"])
            ).sum()
        ),
    }
    if (
        boundary_audit["protected_rows_materialised"] != 0
        or boundary_audit["same_day_or_future_options_joins"] != 0
    ):
        raise RuntimeError("blocked_protected_boundary_failure")
    write_json(output / "source_manifest.json", source_manifest)
    write_json(
        output / "protected_boundary_audit.json",
        boundary_audit,
    )
    write_json(
        output / "structural_panel_reconstruction.json",
        structural_reconstruction,
    )
    write_json(output / "option_pair_reconstruction.json", pair_reconstruction)
    write_csv(output / "options_coverage_gap.csv", gap)
    write_json(output / "cross_market_feature_manifest.json", feature_manifest)
    write_parquet(output / "cross_market_panel.parquet", panel)
    write_json(output / "model_configurations.json", model_configurations())
    write_empty_model_artifacts()
    route_metrics, route_contrast = route_state_movement_metrics(
        panel.loc[panel["period"].eq("assessment")]
    )
    for key, value in route_contrast.items():
        route_metrics[key] = value
    write_csv(output / "test_b_route_state_metrics.csv", route_metrics)
    write_csv(output / "concentration_metrics.csv", concentration)

    statuses = {
        "test_a_options_to_stock_status": "insufficient_support",
        "test_a_disagreement_status": "insufficient_support",
        "test_b_stock_to_options_status": "insufficient_support",
        "test_b_route_increment_status": "insufficient_support",
    }
    validate_individual_statuses(statuses)
    decision = {
        **SAFETY_FLAGS,
        "decision": choose_overall_decision(
            blocker=BLOCKER,
            test_a_supported=False,
            test_b_supported=False,
            disagreement_descriptive=False,
        ),
        **statuses,
        "cached_options_records_used": int(pair_reconstruction["cached_vendor_records_used"]),
        "cached_history_records_materialised": int(
            pair_reconstruction["cached_contract_history_records_materialised"]
        ),
        "protected_cached_history_records_skipped_before_materialisation": int(
            pair_reconstruction["protected_cached_history_records_skipped_before_materialisation"]
        ),
        "exact_previous_close_chain_stock_dates": int(pairs["exact_observation_rows"].gt(0).sum()),
        "valid_previous_close_atm_pairs": int(pairs["pair_available"].sum()),
        "joined_structural_rows": int(len(panel)),
        "joined_assessment_rows": int(coverage["assessment_rows"]),
        "exact_pair_assessment_row_coverage": float(coverage["exact_pair_row_coverage"]),
        "coverage_gates": coverage["gates"],
        "models_fit": [],
        "bootstrap_draws_run": 0,
        "options_null_refits_run": 0,
        "route_null_refits_run": 0,
        "plots_created": 0,
        "binding_questions_answered": False,
        "blocker": BLOCKER,
    }
    write_json(output / "decision.json", decision)
    deterministic = determinism_check(panel, pairs, decision)
    write_json(output / "determinism_check.json", deterministic)
    write_json(
        output / "lightweight_audit.json",
        {
            **SAFETY_FLAGS,
            "passed": False,
            "status": "pending_independent_audit",
            "reason": "run audit_screen_v0.py",
        },
    )
    report = _report(
        panel=panel,
        pairs=pairs,
        coverage=coverage,
        request_gap=request_gap,
        route_metrics=route_metrics,
        decision=decision,
        structural_reconstruction=structural_reconstruction,
        determinism=deterministic,
        concentration=concentration,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    reports_directory = EXPERIMENT_DIR / "reports"
    reports_directory.mkdir(parents=True, exist_ok=True)
    (reports_directory / "report.md").write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    decision = run(arguments.output)
    print(str(decision["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
