#!/usr/bin/env python3
"""Prepare and run the bounded prior-close options IV movement screen V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
for _package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(_REPO_ROOT / "packages" / _package / "src"))

from stocker_research.broad_conflict_options_iv_screen_v0 import (  # noqa: E402
    DENSE_CHECKPOINTS,
    DENSE_H0_FEATURES,
    FROZEN_COHORT,
    OPTIONS_PRIMARY_FEATURES,
    ROUTE_FEATURES,
    SAFETY_FLAGS,
    previous_trading_session,
    verify_structural_reconstruction,
)

EXPERIMENT_DIR = _SCRIPT_DIR
REPO_ROOT = _REPO_ROOT
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
REPORTS = EXPERIMENT_DIR / "reports"
PREDECESSOR = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-broad-conflict-advance-hazard-v02"
    / "artifacts"
    / "primary"
)
DENSE_PANEL = PREDECESSOR / "dense_advance_panel.parquet"
TRACE_PANEL = (
    REPO_ROOT
    / "research"
    / "route-competition"
    / "20260722-route-competition-hazard-quick-v0"
    / "artifacts"
    / "primary"
    / "causal_state_trace.parquet"
)
PROTECTED_START = date(2025, 8, 23)
STARTING_SHA = "2d900cc10af564ff5d783dc65f40f4f3d79874d9"
OPENAPI_SHA = "786448faebd9d3b3c870c95ad86a4a955cee53d6"
OPENAPI_VERSION = "2.0.0"
OPTIONS_FIELDS = (
    "contract",
    "underlying_symbol",
    "exp_date",
    "type",
    "strike",
    "last",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "moneyness",
    "volume",
    "open_interest",
    "volatility",
    "theoretical",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "tradetime",
    "dte",
    "midpoint",
)
MAX_RAW_RECORDS = 3_000_000
MAX_OPTIONS_BYTES = 20_000_000_000
ESTIMATED_RECORD_BYTES = 700
ESTIMATED_RECORDS_PER_SYMBOL_SESSION = 250


def sha256_file(path: Path) -> str:
    """Hash one source or artifact file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Mapping[str, Any]) -> str:
    """Hash a JSON-safe structure using stable compact encoding."""

    content = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one stable, credential-free JSON artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write one small deterministic CSV artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write one deterministic local artifact parquet atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def default_provider_root() -> Path:
    """Use the repository's established local EODHD five-minute cache convention."""

    configured = os.environ.get("STOCKER_EODHD_PROVIDER_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home()
        / "StockerLocal"
        / "data"
        / "processed"
        / "source=eodhd"
        / "instrument_type=stock"
    )


def options_data_dir() -> Path:
    """Return the untracked options cache root without reading a tracked credential file."""

    configured = os.environ.get("EODHD_OPTIONS_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPO_ROOT / "data" / "vendor" / "eodhd" / "options").resolve()


def load_clean_advance_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the exact clean population from the immutable V0.2 panel."""

    if not DENSE_PANEL.is_file():
        raise FileNotFoundError(f"frozen predecessor panel is missing: {DENSE_PANEL}")
    source = pd.read_parquet(DENSE_PANEL)
    explicit_eligibility = source["registered_completion_next_1_bar"].fillna(0).astype(int).eq(
        0
    ) & source["any_prefix_one_transition_from_completion"].fillna(0).astype(int).eq(0)
    frozen_eligibility = source["advance_eligible"].astype(int).eq(1)
    eligibility_mismatches = int(explicit_eligibility.ne(frozen_eligibility).sum())
    reference = source.loc[frozen_eligibility].copy()
    reconstructed = source.loc[explicit_eligibility].copy()
    feature_columns = (*DENSE_H0_FEATURES, *ROUTE_FEATURES)
    comparison = verify_structural_reconstruction(
        reference,
        reconstructed,
        feature_columns=feature_columns,
    )
    comparison.update(
        {
            "eligibility_mismatches": eligibility_mismatches,
            "source_rows": len(source),
            "clean_advance_rows": len(reference),
            "development_clean_rows": int(reference["period"].eq("development").sum()),
            "assessment_clean_rows": int(reference["period"].eq("assessment").sum()),
            "assessment_clean_positives": int(
                reference.loc[
                    reference["period"].eq("assessment"),
                    "completion_in_bars_2_or_3",
                ].sum()
            ),
            "frozen_h0_features": list(DENSE_H0_FEATURES),
            "frozen_route_features": list(ROUTE_FEATURES),
            "frozen_checkpoints": list(DENSE_CHECKPOINTS),
            "a0_predictions_available": bool(reference["A0_probability"].notna().all()),
            "a1_predictions_available": bool(reference["A1_probability"].notna().all()),
            "passed": bool(
                cast(bool, comparison["passed"])
                and eligibility_mismatches == 0
                and len(reference) == 87_443
            ),
        }
    )
    if not comparison["passed"]:
        raise RuntimeError("blocked_structural_panel_reconstruction_failure")
    return reference.reset_index(drop=True), comparison


def required_session_rows(clean: pd.DataFrame) -> pd.DataFrame:
    """Derive every exact prior US session from frozen signal dates."""

    signal_dates = pd.to_datetime(clean["session"], errors="raise").dt.date
    if max(signal_dates) >= PROTECTED_START:
        raise RuntimeError("blocked_protected_boundary_failure")
    signals = sorted(set(signal_dates))
    prior_by_signal = {signal: previous_trading_session(signal) for signal in signals}
    rows = [
        {
            "signal_date": signal,
            "required_options_date": prior_by_signal[signal],
        }
        for signal in signals
    ]
    frame = pd.DataFrame(rows)
    if (frame["required_options_date"] >= frame["signal_date"]).any():
        raise RuntimeError("blocked_chronology_or_leakage_failure")
    return frame


def _load_symbol_daily(provider_root: Path, symbol: str) -> pd.DataFrame:
    path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"underlying source missing for {symbol}: {path}")
    raw = pd.read_parquet(
        path,
        columns=["timestamp", "open", "close"],
        filters=[("timestamp", "<", pd.Timestamp("2025-08-23T00:00:00Z"))],
    )
    timestamp = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
    working = raw.assign(
        timestamp=timestamp,
        session=timestamp.dt.tz_convert("America/New_York").dt.date,
    ).sort_values("timestamp", kind="mergesort")
    daily = (
        working.groupby("session", sort=True)
        .agg(first_open=("open", "first"), unadjusted_close=("close", "last"))
        .reset_index()
    )
    daily["previous_unadjusted_close"] = daily["unadjusted_close"].shift(1)
    daily["overnight_price_ratio"] = daily["first_open"] / daily["previous_unadjusted_close"]
    daily["inferred_corporate_action_boundary"] = daily["overnight_price_ratio"].lt(0.55) | daily[
        "overnight_price_ratio"
    ].gt(1.80)
    return daily


def underlying_price_audit(
    clean: pd.DataFrame,
    required: pd.DataFrame,
    *,
    provider_root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Resolve exact unadjusted closes and infer ambiguous split boundaries."""

    prior_by_signal = {
        cast(date, row.signal_date): cast(date, row.required_options_date)
        for row in required.itertuples(index=False)
    }
    signal_dates = sorted(prior_by_signal)
    rows: list[dict[str, object]] = []
    source_hashes: dict[str, str] = {}
    for symbol in FROZEN_COHORT:
        path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        source_hashes[symbol] = sha256_file(path)
        daily = _load_symbol_daily(provider_root, symbol).set_index("session")
        for signal_date in signal_dates:
            options_date = prior_by_signal[signal_date]
            if options_date not in daily.index or signal_date not in daily.index:
                rows.append(
                    {
                        "symbol": symbol,
                        "signal_date": signal_date.isoformat(),
                        "required_options_date": options_date.isoformat(),
                        "previous_close_underlying_price": math.nan,
                        "price_source": "repository_unadjusted_eodhd_5m_close",
                        "source_available": False,
                        "inferred_split_on_signal_date": False,
                        "split_boundary_ambiguous": False,
                        "provider_moneyness_check": "pending_options_download",
                    }
                )
                continue
            signal_row = daily.loc[signal_date]
            rows.append(
                {
                    "symbol": symbol,
                    "signal_date": signal_date.isoformat(),
                    "required_options_date": options_date.isoformat(),
                    "previous_close_underlying_price": float(
                        daily.loc[options_date, "unadjusted_close"]
                    ),
                    "price_source": "repository_unadjusted_eodhd_5m_close",
                    "source_available": True,
                    "inferred_split_on_signal_date": bool(
                        signal_row["inferred_corporate_action_boundary"]
                    ),
                    "split_boundary_ambiguous": bool(
                        signal_row["inferred_corporate_action_boundary"]
                    ),
                    "provider_moneyness_check": "pending_options_download",
                }
            )
    audit = (
        pd.DataFrame(rows)
        .sort_values(["symbol", "signal_date"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected = len(FROZEN_COHORT) * len(signal_dates)
    if len(audit) != expected or not audit["source_available"].all():
        raise RuntimeError("blocked_structural_panel_reconstruction_failure")
    if (audit["previous_close_underlying_price"] <= 0.0).any():
        raise RuntimeError("blocked_options_data_integrity_failure")
    return audit, source_hashes


def build_request_plan(price_audit: pd.DataFrame) -> dict[str, Any]:
    """Build deterministic bounded symbol-month EOD requests and resource estimates."""

    valid = price_audit.loc[~price_audit["split_boundary_ambiguous"].astype(bool)].copy()
    valid["required_options_date"] = pd.to_datetime(
        valid["required_options_date"], errors="raise"
    ).dt.date
    valid["month"] = valid["required_options_date"].map(
        lambda value: cast(date, value).strftime("%Y-%m")
    )
    chunks: list[dict[str, Any]] = []
    estimated_records = 0
    estimated_eod_page_requests = 0
    for (symbol, month), group in valid.groupby(["symbol", "month"], sort=True):
        dates = sorted(set(cast(Sequence[date], group["required_options_date"])))
        minimum_close = float(group["previous_close_underlying_price"].min())
        maximum_close = float(group["previous_close_underlying_price"].max())
        chunk_records = len(dates) * ESTIMATED_RECORDS_PER_SYMBOL_SESSION
        chunk_requests = max(1, math.ceil(chunk_records / 1000))
        estimated_records += chunk_records
        estimated_eod_page_requests += chunk_requests
        chunks.append(
            {
                "chunk_id": hashlib.sha256(
                    f"{symbol}|{month}|{dates[0]}|{dates[-1]}".encode()
                ).hexdigest(),
                "underlying_symbol": str(symbol),
                "calendar_month": str(month),
                "trade_date_from": dates[0].isoformat(),
                "trade_date_to": dates[-1].isoformat(),
                "required_trade_dates": [value.isoformat() for value in dates],
                "required_trade_date_count": len(dates),
                "minimum_unadjusted_close": minimum_close,
                "maximum_unadjusted_close": maximum_close,
                "strike_from": round(0.70 * minimum_close, 8),
                "strike_to": round(1.30 * maximum_close, 8),
                "expiration_from": (dates[0] + timedelta(days=7)).isoformat(),
                "expiration_to": (dates[-1] + timedelta(days=90)).isoformat(),
                "local_dte_filter": {"minimum": 7, "maximum": 90},
                "fields": list(OPTIONS_FIELDS),
                "compact": True,
                "estimated_records": chunk_records,
                "estimated_requests": chunk_requests,
            }
        )
    estimated_bytes = estimated_records * ESTIMATED_RECORD_BYTES
    unique_dates = sorted(set(cast(Sequence[date], valid["required_options_date"])))
    resource_gate = bool(
        estimated_records <= MAX_RAW_RECORDS and estimated_bytes <= MAX_OPTIONS_BYTES
    )
    return {
        "status": "ready" if resource_gate else "blocked_options_download_resource_limit",
        "endpoint": "/mp/unicornbay/options/eod",
        "one_process": True,
        "uncontrolled_concurrency": False,
        "symbols": list(FROZEN_COHORT),
        "symbol_count": len(FROZEN_COHORT),
        "symbol_month_chunks": len(chunks),
        "estimated_eod_page_requests": estimated_eod_page_requests,
        "estimated_setup_requests": 2,
        "estimated_requests": estimated_eod_page_requests + 2,
        "estimated_records": estimated_records,
        "estimated_storage_bytes": estimated_bytes,
        "maximum_raw_option_records": MAX_RAW_RECORDS,
        "maximum_options_bytes": MAX_OPTIONS_BYTES,
        "required_date_count": len(unique_dates),
        "required_date_start": unique_dates[0].isoformat(),
        "required_date_end": unique_dates[-1].isoformat(),
        "required_date_coverage": [value.isoformat() for value in unique_dates],
        "excluded_split_boundary_symbol_dates": int(
            price_audit["split_boundary_ambiguous"].astype(bool).sum()
        ),
        "resource_gate_passed": resource_gate,
        "chunks": chunks,
    }


def schema_mapping() -> dict[str, Any]:
    """Return the credential-free official OpenAPI schema summary used by V0."""

    canonical_mapping = {
        "contract": "contract_id",
        "underlying_symbol": "underlying_symbol",
        "type": "option_type",
        "exp_date": "expiration_date",
        "strike": "strike",
        "tradetime": "trade_date and trade_timestamp in America/New_York",
        "last": "last",
        "bid": "bid",
        "ask": "ask",
        "bid_size": "bid_size",
        "ask_size": "ask_size",
        "midpoint": "midpoint",
        "volume": "volume",
        "open_interest": "open_interest",
        "volatility": "implied_volatility",
        "theoretical": "theoretical_value",
        "delta": "delta",
        "gamma": "gamma",
        "theta": "theta",
        "vega": "vega",
        "rho": "rho",
        "dte": "dte",
        "moneyness": "moneyness",
    }
    return {
        "verified": True,
        "verified_on": "2026-07-22",
        "official_openapi_repository": "https://github.com/EodHistoricalData/EODHD-openapi",
        "official_openapi_commit": OPENAPI_SHA,
        "official_openapi_version": OPENAPI_VERSION,
        "base_url": "https://eodhd.com/api",
        "authentication": {
            "location": "query",
            "parameter": "api_token",
            "artifact_redaction_required": True,
        },
        "endpoints": {
            "underlying_symbol_coverage": "/mp/unicornbay/options/underlying-symbols",
            "contract_discovery": "/mp/unicornbay/options/contracts",
            "historical_end_of_day": "/mp/unicornbay/options/eod",
        },
        "historical_eod": {
            "pagination": {"offset_minimum": 0, "offset_maximum": 10000, "limit_maximum": 1000},
            "filters": [
                "underlying_symbol",
                "tradetime",
                "tradetime_from",
                "tradetime_to",
                "exp_date",
                "exp_date_from",
                "exp_date_to",
                "strike",
                "strike_from",
                "strike_to",
                "type",
            ],
            "field_selection": "fields[options-eod]",
            "compact_mode": "compact=1 with meta.fields",
            "rate_limit_response": 429,
            "transient_retries": [429, 500, 502, 503, 504],
            "permanent_no_retry": [401, 403, 404, 422],
        },
        "canonical_mapping": canonical_mapping,
        "undocumented_fields_required": False,
    }


def empty_result_artifacts(primary: Path) -> None:
    """Materialize explicit schema-bearing blocked outputs without fabricating observations."""

    write_parquet(
        primary / "selected_option_pairs.parquet",
        pd.DataFrame(
            columns=[
                "symbol",
                "signal_date",
                "required_options_date",
                "selected_expiry",
                "selected_strike",
                "call_contract_id",
                "put_contract_id",
                *OPTIONS_PRIMARY_FEATURES,
            ]
        ),
    )
    write_parquet(
        primary / "options_movement_panel.parquet",
        pd.DataFrame(
            columns=[
                "row_id",
                "symbol",
                "session",
                "checkpoint",
                "required_options_date",
                "absolute_log_return_15m",
                "iv_expected_absolute_15m",
                "iv_absolute_residual_15m",
                "movement_exceeds_iv_expected_absolute",
            ]
        ),
    )
    write_parquet(
        primary / "assessment_predictions.parquet",
        pd.DataFrame(columns=["row_id", "O0_probability", "O1_probability"]),
    )
    csv_schemas = {
        "options_data_quality.csv": ["metric", "value", "status"],
        "options_rejections.csv": [
            "request_id",
            "record_index",
            "provider_record_id",
            "reason_code",
            "raw_record_hash",
        ],
        "options_coverage.csv": [
            "symbol",
            "period",
            "required_rows",
            "valid_pair_rows",
            "coverage",
            "status",
        ],
        "options_structural_join_audit.csv": [
            "scope",
            "key",
            "rows",
            "exact_chain_rows",
            "valid_pair_rows",
            "coverage",
            "status",
        ],
        "pooled_metrics.csv": ["model", "metric", "value", "status"],
        "route_state_movement_metrics.csv": ["route_resolution_state", "metric", "value", "status"],
        "matched_control_metrics.csv": ["contrast", "metric", "value", "status"],
        "monthly_metrics.csv": ["month", "model_or_state", "metric", "value", "status"],
        "checkpoint_metrics.csv": ["checkpoint", "model_or_state", "metric", "value", "status"],
        "subgroup_metrics.csv": [
            "subgroup",
            "level",
            "model_or_state",
            "metric",
            "value",
            "status",
        ],
        "continuous_residual_metrics.csv": ["model", "metric", "value", "status"],
        "bootstrap_metrics.csv": ["draw", "metric", "value", "status"],
        "route_null_metrics.csv": ["draw", "metric", "real_increment", "null_increment", "status"],
        "concentration_metrics.csv": ["symbol", "weighted_share", "status"],
    }
    for name, columns in csv_schemas.items():
        write_csv(primary / name, pd.DataFrame(columns=columns))
    write_json(
        primary / "option_pair_selection_manifest.json",
        {"status": "blocked", "reason": "blocked_missing_eodhd_api_token", "pairs": []},
    )
    write_json(
        primary / "model_coefficients.json",
        {"status": "blocked", "reason": "blocked_missing_eodhd_api_token", "models": {}},
    )


def feature_manifest() -> dict[str, Any]:
    """Return the frozen O0/O1 model surface."""

    return {
        "O0": {
            "prior_close_options": list(OPTIONS_PRIMARY_FEATURES),
            "frozen_compressed_transition_h0": list(DENSE_H0_FEATURES),
            "controls": [
                "stock_fixed_effect",
                "checkpoint_fixed_effect",
                "month_of_year_fixed_effect",
            ],
        },
        "O1": {"inherits": "O0", "frozen_route_competition_bundle": list(ROUTE_FEATURES)},
        "preprocessing_fit_period": "2024_development_only",
        "same_day_options_features": False,
        "future_options_features": False,
    }


def outcome_manifest() -> dict[str, Any]:
    """Return the underlying-only primary and secondary outcome contract."""

    return {
        "entry_price": "open of first completed five-minute bar after checkpoint",
        "primary_horizon": "next three completed five-minute bars only",
        "primary_binary": "movement_exceeds_iv_expected_absolute",
        "primary_continuous": "iv_absolute_residual_15m",
        "primary_underlying_outcomes": [
            "absolute_log_return_15m",
            "realised_range_15m",
            "maximum_absolute_excursion_15m",
            "realised_variance_15m",
        ],
        "secondary_descriptive_horizons_minutes": [10, 30, 60],
        "annual_trading_minutes": 252 * 390,
        "intraday_option_fill_simulated": False,
        "option_pnl_calculated": False,
        "directional_outcomes_primary": False,
    }


def model_configurations() -> dict[str, Any]:
    """Return all fitted-model and resampling bounds."""

    return {
        "binary_models": ["O0", "O1"],
        "binary": {
            "penalty": "l2",
            "C": 0.25,
            "solver": "liblinear",
            "max_iter": 300,
            "class_weight": None,
            "n_jobs": 1,
        },
        "continuous_models": ["R0", "R1"],
        "continuous": {"kind": "Ridge", "alpha": 10.0},
        "weights": "exact_frozen_candidate_normalized_row_weight",
        "session_bootstrap": {"draws": 25, "refit": False, "seed": 20260722},
        "route_bundle_null": {
            "refits": 5,
            "strata": ["development_or_assessment", "session", "checkpoint"],
            "bundle_features": list(ROUTE_FEATURES),
            "preserve_O0": True,
        },
        "materially_adverse_checkpoint_thresholds": {
            "O1_log_loss_improvement_below": -0.005,
            "O1_brier_improvement_below": -0.002,
            "broad_minus_low_iv_residual_below": -0.0005,
        },
        "hyperparameter_tuning": False,
        "threshold_tuning": False,
    }


def blocked_report(
    *,
    plan: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    price_audit: pd.DataFrame,
) -> str:
    """Render the direct blocker report without options or economic claims."""

    ambiguous = int(price_audit["split_boundary_ambiguous"].astype(bool).sum())
    clean_rows = f"{int(reconstruction['clean_advance_rows']):,}"
    assessment_rows = f"{int(reconstruction['assessment_clean_rows']):,}"
    chunk_count = f"{int(plan['symbol_month_chunks']):,}"
    estimated_requests = f"{int(plan['estimated_requests']):,}"
    estimated_records = f"{int(plan['estimated_records']):,}"
    estimated_bytes = f"{int(plan['estimated_storage_bytes']):,}"
    return f"""# Prior-Close Options IV Movement Screen V0 — blocked pre-download

Primary decision: `blocked_missing_eodhd_api_token`

The frozen V0.2 clean-advance population reconstructed exactly ({clean_rows} rows;
{assessment_rows} assessment rows). The request plan contains {chunk_count} symbol-month chunks for
{plan["symbol_count"]} frozen symbols and {plan["required_date_count"]} exact prior trading sessions
from {plan["required_date_start"]} through {plan["required_date_end"]}. It estimates
{estimated_requests} EOD page requests and {estimated_records} raw records, totalling approximately
{estimated_bytes} bytes within the frozen resource caps. {ambiguous} symbol-date joins cross an
inferred unadjusted-price corporate-action boundary and are preregistered as unavailable.

`EODHD_API_TOKEN` was not present. No provider preflight, mapping call, cohort download, option-pair
selection, structural join, underlying movement inference, model fit, bootstrap draw, route-null
refit, or plot was performed. The public demo token was not substituted.

This is a retrospective research-only feasibility screen. It contains no intraday option fill,
option P&L, executable return, profitable-straddle claim, directional edge, prospective validation,
trading utility, or deployable strategy.
"""


def prepare_blocked(*, provider_root: Path, primary: Path = PRIMARY) -> dict[str, Any]:
    """Prepare exact sources/request plan and emit the required missing-token blocker."""

    primary.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    clean, reconstruction = load_clean_advance_panel()
    required = required_session_rows(clean)
    price_audit, source_hashes = underlying_price_audit(
        clean,
        required,
        provider_root=provider_root,
    )
    plan = build_request_plan(price_audit)
    if not plan["resource_gate_passed"]:
        raise RuntimeError("blocked_options_download_resource_limit")
    token_available = bool(os.environ.get("EODHD_API_TOKEN"))
    if token_available:
        raise RuntimeError("prepare_blocked called while EODHD_API_TOKEN is available")
    shutil.copyfile(EXPERIMENT_DIR / "contract.json", primary / "contract.json")
    schema = schema_mapping()
    write_json(primary / "eodhd_options_schema_mapping.json", schema)
    write_json(primary / "options_request_plan.json", cast(dict[str, Any], plan))
    write_csv(primary / "option_underlying_price_audit.csv", price_audit)
    mapping = pd.DataFrame(
        [
            {
                "stocker_symbol": symbol,
                "eodhd_underlying_symbol": symbol,
                "coverage_available": False,
                "earliest_option_date": "",
                "latest_option_date": "",
                "records_returned": 0,
                "mapping_method": "exact_or_us_suffix_candidate_pending_coverage_preflight",
                "status": "blocked_missing_eodhd_api_token",
            }
            for symbol in FROZEN_COHORT
        ]
    )
    write_csv(primary / "underlying_symbol_mapping.csv", mapping)
    write_json(
        primary / "eodhd_options_api_preflight.json",
        {
            "status": "blocked_missing_eodhd_api_token",
            "schema_verified": True,
            "endpoint": "/mp/unicornbay/options/eod",
            "planned_symbol": FROZEN_COHORT[0],
            "record_limit": 10,
            "requests_completed": 0,
            "records_received": 0,
            "authentication_redacted": True,
            "historical_session_dates_confirmed_from_response": False,
        },
    )
    write_json(
        primary / "options_download_manifest.json",
        {
            "status": "blocked_missing_eodhd_api_token",
            "requests_completed": 0,
            "raw_records": 0,
            "download_bytes": 0,
            "pagination_complete": False,
            "unexplained_truncations": 0,
            "credential_exposures": 0,
            "manifest_rows": [],
        },
    )
    write_json(primary / "structural_panel_reconstruction.json", reconstruction)
    write_json(primary / "feature_manifest.json", feature_manifest())
    write_json(primary / "outcome_manifest.json", outcome_manifest())
    write_json(primary / "model_configurations.json", model_configurations())
    clean_dates = pd.to_datetime(clean["session"], errors="raise").dt.date
    boundary = {
        "protected_start": PROTECTED_START.isoformat(),
        "maximum_signal_date": max(clean_dates).isoformat(),
        "maximum_required_options_date": max(required["required_options_date"]).isoformat(),
        "underlying_source_read_filter": "timestamp < 2025-08-23T00:00:00Z",
        "maximum_underlying_price_date": max(
            pd.to_datetime(price_audit["signal_date"], errors="raise").dt.date
        ).isoformat(),
        "protected_rows_materialised": 0,
        "same_day_options_joins": 0,
        "future_options_joins": 0,
        "passed": True,
    }
    write_json(primary / "protected_boundary_audit.json", boundary)
    source_manifest = {
        "starting_branch": "agent/broad-conflict-advance-hazard-v02",
        "starting_sha": STARTING_SHA,
        "frozen_predecessor_panel": str(DENSE_PANEL.relative_to(REPO_ROOT)),
        "frozen_predecessor_panel_sha256": sha256_file(DENSE_PANEL),
        "frozen_underlying_trace": str(TRACE_PANEL.relative_to(REPO_ROOT)),
        "frozen_underlying_trace_sha256": sha256_file(TRACE_PANEL),
        "underlying_provider_root": str(provider_root),
        "underlying_source_hashes": source_hashes,
        "unadjusted_underlying_close": True,
        "frozen_cohort": list(FROZEN_COHORT),
        "signal_date_start": min(clean_dates).isoformat(),
        "signal_date_end": max(clean_dates).isoformat(),
        "required_options_date_start": plan["required_date_start"],
        "required_options_date_end": plan["required_date_end"],
        "underlying_source_read_filter": "timestamp < 2025-08-23T00:00:00Z",
        "protected_rows_materialised": 0,
        "official_openapi_commit": OPENAPI_SHA,
        "options_cache_path": str(options_data_dir()),
        "raw_vendor_data_tracked": False,
    }
    write_json(primary / "source_manifest.json", source_manifest)
    empty_result_artifacts(primary)
    join_summary = pd.DataFrame(
        [
            {
                "scope": "pooled",
                "key": period,
                "rows": int(clean["period"].eq(period).sum()),
                "exact_chain_rows": 0,
                "valid_pair_rows": 0,
                "coverage": 0.0,
                "status": "blocked_missing_eodhd_api_token",
            }
            for period in ("development", "assessment")
        ]
    )
    write_csv(primary / "options_structural_join_audit.csv", join_summary)
    coverage = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "period": period,
                "required_rows": int(
                    (clean["symbol"].eq(symbol) & clean["period"].eq(period)).sum()
                ),
                "valid_pair_rows": 0,
                "coverage": 0.0,
                "status": "blocked_missing_eodhd_api_token",
            }
            for symbol in FROZEN_COHORT
            for period in ("development", "assessment")
        ]
    )
    write_csv(primary / "options_coverage.csv", coverage)
    quality = pd.DataFrame(
        [
            {"metric": metric, "value": value, "status": "blocked_missing_eodhd_api_token"}
            for metric, value in (
                ("raw_records", 0),
                ("canonical_records", 0),
                ("duplicate_records", 0),
                ("rejected_records", 0),
            )
        ]
    )
    write_csv(primary / "options_data_quality.csv", quality)
    decision = {
        **SAFETY_FLAGS,
        "decision": "blocked_missing_eodhd_api_token",
        "options_download_status": "blocked",
        "options_coverage_status": "blocked",
        "iv_excess_model_status": "blocked",
        "broad_conflict_movement_status": "blocked",
        "matched_control_status": "blocked",
        "blocker": "EODHD_API_TOKEN unavailable; no public demo-token substitution",
        "provider_requests_completed": 0,
        "bootstrap_draws_executed": 0,
        "route_null_refits_executed": 0,
    }
    write_json(primary / "decision.json", decision)
    plan_again = build_request_plan(price_audit)
    determinism = {
        "status": "blocked_before_options_data",
        "reason": "blocked_missing_eodhd_api_token",
        "request_plan_rebuild_match": stable_json_hash(cast(dict[str, Any], plan))
        == stable_json_hash(cast(dict[str, Any], plan_again)),
        "structural_reconstruction_repeatable": bool(reconstruction["passed"]),
        "selected_contract_mismatches": None,
        "joined_row_mismatches": None,
        "maximum_option_feature_difference": None,
        "maximum_probability_difference": None,
        "maximum_movement_difference": None,
        "bootstrap_repeated": False,
        "route_null_refits_repeated": False,
    }
    write_json(primary / "determinism_check.json", determinism)
    report = blocked_report(plan=plan, reconstruction=reconstruction, price_audit=price_audit)
    (primary / "report.md").write_text(report, encoding="utf-8")
    (REPORTS / "blocked_pre_download_report.md").write_text(report, encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="prepare bounded request plan")
    parser.add_argument(
        "--analyse", action="store_true", help="analyse a completed canonical cache"
    )
    parser.add_argument("--provider-root", type=Path, default=default_provider_root())
    parser.add_argument("--output", type=Path, default=PRIMARY)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.analyse:
        from build_options_panel import (
            build_and_analyse_cached_options,
            classify_analysis_blocker,
            write_analysis_blocker,
        )

        output = arguments.output.expanduser().resolve()
        try:
            build_and_analyse_cached_options(
                provider_root=arguments.provider_root.expanduser().resolve(),
                output=output,
            )
        except Exception as error:
            decision = write_analysis_blocker(output, classify_analysis_blocker(error))
            print(decision["decision"])
            return 1
        return 0
    decision = prepare_blocked(
        provider_root=arguments.provider_root.expanduser().resolve(),
        primary=arguments.output.expanduser().resolve(),
    )
    print(decision["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
