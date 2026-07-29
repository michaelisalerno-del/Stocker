#!/usr/bin/env python3
"""Independent lightweight audit for the prior-close options IV screen V0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from run_screen_v0 import (  # noqa: E402
    DENSE_PANEL,
    OPENAPI_SHA,
    OPTIONS_FIELDS,
    PRIMARY,
    PROTECTED_START,
    TRACE_PANEL,
    schema_mapping,
    write_json,
)

from stocker_data.calendars import get_market_calendar  # noqa: E402
from stocker_research.broad_conflict_options_iv_screen_v0 import (  # noqa: E402
    DENSE_H0_FEATURES,
    FROZEN_COHORT,
    OPTIONS_PRIMARY_FEATURES,
    ROUTE_FEATURES,
    SAFETY_FLAGS,
    broad_conflict_iv_gate_passes,
    build_matched_control_relations,
    calculate_optional_option_features,
    calculate_primary_option_features,
    choose_options_movement_decision,
    coverage_gates_pass,
    fixed_session_bootstrap_multiplicities,
    iv_movement_approximations,
    o1_model_gate_passes,
    permute_intact_route_bundle,
    select_primary_atm_pair,
)
from stocker_research.eodhd_options_downloader_v0 import (  # noqa: E402
    canonicalize_response_records,
    resolve_canonical_duplicates,
    stable_request_id,
)

REQUIRED_ARTIFACTS = (
    "contract.json",
    "source_manifest.json",
    "protected_boundary_audit.json",
    "eodhd_options_api_preflight.json",
    "eodhd_options_schema_mapping.json",
    "underlying_symbol_mapping.csv",
    "options_request_plan.json",
    "options_download_manifest.json",
    "options_data_quality.csv",
    "options_rejections.csv",
    "options_coverage.csv",
    "option_underlying_price_audit.csv",
    "option_pair_selection_manifest.json",
    "selected_option_pairs.parquet",
    "options_structural_join_audit.csv",
    "structural_panel_reconstruction.json",
    "options_movement_panel.parquet",
    "feature_manifest.json",
    "outcome_manifest.json",
    "model_configurations.json",
    "model_coefficients.json",
    "assessment_predictions.parquet",
    "pooled_metrics.csv",
    "route_state_movement_metrics.csv",
    "matched_control_metrics.csv",
    "monthly_metrics.csv",
    "checkpoint_metrics.csv",
    "subgroup_metrics.csv",
    "continuous_residual_metrics.csv",
    "bootstrap_metrics.csv",
    "route_null_metrics.csv",
    "concentration_metrics.csv",
    "decision.json",
    "determinism_check.json",
    "report.md",
)
AUTH_QUERY = re.compile(r"[?&](?:api_token|token)=[^&\s]+", re.IGNORECASE)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"JSON artifact is not an object: {path.name}")
    return cast(dict[str, Any], value)


def _previous_session(value: date) -> date:
    calendar = get_market_calendar("NYSE")
    sessions = calendar.valid_days(
        start_date=value - timedelta(days=30),
        end_date=value - timedelta(days=1),
        tz="America/New_York",
    )
    if len(sessions) == 0:
        raise AssertionError(f"prior session unavailable: {value}")
    return cast(date, sessions[-1].date())


def _credential_scan(output: Path) -> dict[str, Any]:
    exposures: list[str] = []
    token = os.environ.get("EODHD_API_TOKEN", "")
    for path in sorted(output.iterdir()):
        if path.suffix not in {".json", ".csv", ".md"}:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if AUTH_QUERY.search(content) or (token and token in content):
            exposures.append(path.name)
    return {"passed": not exposures, "exposure_files": exposures}


def _required_dates() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(
        DENSE_PANEL,
        columns=[
            "row_id",
            "symbol",
            "session",
            "period",
            "checkpoint",
            "route_resolution_state",
            "registered_completion_next_1_bar",
            "any_prefix_one_transition_from_completion",
            "advance_eligible",
            "A0_probability",
            "A1_probability",
            "sequential_row_weight",
            "row_weight",
            *DENSE_H0_FEATURES,
            *ROUTE_FEATURES,
        ],
    )
    independent = source["registered_completion_next_1_bar"].fillna(0).astype(int).eq(0) & source[
        "any_prefix_one_transition_from_completion"
    ].fillna(0).astype(int).eq(0)
    frozen = source["advance_eligible"].astype(int).eq(1)
    if int(independent.ne(frozen).sum()) != 0:
        raise AssertionError("independent clean-advance eligibility differs from frozen panel")
    clean = source.loc[independent].copy()
    signal_dates = sorted(set(pd.to_datetime(clean["session"], errors="raise").dt.date))
    required = pd.DataFrame(
        {
            "signal_date": signal_dates,
            "required_options_date": [_previous_session(value) for value in signal_dates],
        }
    )
    return clean, required


def _audit_structural_reconstruction(output: Path, clean: pd.DataFrame) -> dict[str, Any]:
    artifact = _read_json(output / "structural_panel_reconstruction.json")
    feature = _read_json(output / "feature_manifest.json")
    row_identity_mismatches = int(clean["row_id"].duplicated().sum())
    route_state_mismatches = int(
        (
            ~clean["route_resolution_state"]
            .astype(str)
            .isin(
                [
                    "BROAD_CONFLICT",
                    "LOW_ROUTE_SUPPORT",
                    "OTHER",
                    "NARROWING",
                    "DOMINANT_ROUTE",
                ]
            )
        ).sum()
    )
    weights_valid = bool(
        np.isfinite(clean[["sequential_row_weight", "row_weight"]].to_numpy(float)).all()
        and clean[["sequential_row_weight", "row_weight"]].gt(0.0).all().all()
    )
    probabilities_valid = bool(
        np.isfinite(clean[["A0_probability", "A1_probability"]].to_numpy(float)).all()
    )
    feature_surface_valid = bool(
        feature["O0"]["frozen_compressed_transition_h0"] == list(DENSE_H0_FEATURES)
        and feature["O1"]["frozen_route_competition_bundle"] == list(ROUTE_FEATURES)
    )
    passed = bool(
        artifact["passed"]
        and int(artifact["source_rows"]) == 119_395
        and int(artifact["clean_advance_rows"]) == len(clean) == 87_443
        and int(artifact["development_clean_rows"]) == int(clean["period"].eq("development").sum())
        and int(artifact["assessment_clean_rows"]) == int(clean["period"].eq("assessment").sum())
        and int(artifact["row_identity_mismatches"]) == row_identity_mismatches == 0
        and int(artifact["route_state_mismatches"]) == route_state_mismatches == 0
        and float(artifact["maximum_difference"]) <= 1e-12
        and probabilities_valid
        and weights_valid
        and feature_surface_valid
    )
    return {
        "passed": passed,
        "rows": len(clean),
        "row_identity_mismatches": row_identity_mismatches,
        "route_state_mismatches": route_state_mismatches,
        "A0_A1_predictions_finite": probabilities_valid,
        "slate_and_dense_checkpoint_weights_valid": weights_valid,
        "frozen_feature_surfaces_match": feature_surface_valid,
        "maximum_shared_field_difference": float(artifact["maximum_difference"]),
    }


def _audit_plan(output: Path, required: pd.DataFrame) -> dict[str, Any]:
    plan = _read_json(output / "options_request_plan.json")
    required_dates = sorted(value.isoformat() for value in set(required["required_options_date"]))
    chunks = cast(list[dict[str, Any]], plan["chunks"])
    chunk_hashes_valid = True
    for chunk in chunks:
        expected_id = hashlib.sha256(
            (
                f"{chunk['underlying_symbol']}|{chunk['calendar_month']}|"
                f"{chunk['trade_date_from']}|{chunk['trade_date_to']}"
            ).encode()
        ).hexdigest()
        chunk_hashes_valid &= expected_id == chunk["chunk_id"]
        chunk_hashes_valid &= math_isclose(
            float(chunk["strike_from"]),
            0.70 * float(chunk["minimum_unadjusted_close"]),
        )
        chunk_hashes_valid &= math_isclose(
            float(chunk["strike_to"]),
            1.30 * float(chunk["maximum_unadjusted_close"]),
        )
    return {
        "passed": bool(
            plan["symbols"] == list(FROZEN_COHORT)
            and plan["required_date_coverage"] == required_dates
            and int(plan["required_date_count"]) == len(required_dates)
            and bool(plan["resource_gate_passed"])
            and int(plan["estimated_records"]) <= 3_000_000
            and int(plan["estimated_storage_bytes"]) <= 20_000_000_000
            and chunk_hashes_valid
        ),
        "required_dates": len(required_dates),
        "chunks": len(chunks),
        "chunk_hashes_and_strike_bounds_valid": chunk_hashes_valid,
    }


def _audit_unadjusted_prices(output: Path) -> dict[str, Any]:
    source = _read_json(output / "source_manifest.json")
    provider_root = Path(str(source["underlying_provider_root"]))
    prices = pd.read_csv(output / "option_underlying_price_audit.csv")
    expected: dict[tuple[str, str], float] = {}
    for symbol in FROZEN_COHORT:
        path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
        raw = pd.read_parquet(
            path,
            columns=["timestamp", "close"],
            filters=[("timestamp", "<", pd.Timestamp("2025-08-23T00:00:00Z"))],
        )
        timestamp = pd.to_datetime(raw["timestamp"], utc=True, errors="raise")
        daily = (
            raw.assign(
                timestamp=timestamp,
                session=timestamp.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"),
            )
            .sort_values("timestamp", kind="mergesort")
            .groupby("session", sort=True)["close"]
            .last()
        )
        for options_date in prices.loc[prices["symbol"].eq(symbol), "required_options_date"].astype(
            str
        ):
            expected[(symbol, options_date)] = float(daily.loc[options_date])
    differences = [
        abs(
            float(row.previous_close_underlying_price)
            - expected[(str(row.symbol), str(row.required_options_date))]
        )
        for row in prices.itertuples(index=False)
    ]
    maximum = max(differences, default=math.inf)
    return {
        "passed": maximum <= 1e-12,
        "rows": len(prices),
        "maximum_unadjusted_close_difference": maximum,
        "protected_rows_materialised": 0,
    }


def math_isclose(first: float, second: float) -> bool:
    return abs(first - second) <= 1e-8 * max(1.0, abs(first), abs(second))


def _audit_blocked_run(
    output: Path,
    *,
    clean: pd.DataFrame,
    required: pd.DataFrame,
) -> dict[str, Any]:
    decision = _read_json(output / "decision.json")
    download = _read_json(output / "options_download_manifest.json")
    preflight = _read_json(output / "eodhd_options_api_preflight.json")
    reconstruction = _read_json(output / "structural_panel_reconstruction.json")
    prices = pd.read_csv(output / "option_underlying_price_audit.csv")
    pairs = pd.read_parquet(output / "selected_option_pairs.parquet")
    movement = pd.read_parquet(output / "options_movement_panel.parquet")
    predictions = pd.read_parquet(output / "assessment_predictions.parquet")
    exact_date_pairs = set(
        zip(
            required["signal_date"].map(date.isoformat),
            required["required_options_date"].map(date.isoformat),
            strict=True,
        )
    )
    price_date_pairs = set(
        zip(
            prices["signal_date"].astype(str),
            prices["required_options_date"].astype(str),
            strict=True,
        )
    )
    date_mapping_passed = bool(
        len(prices) == len(FROZEN_COHORT) * len(required)
        and price_date_pairs == exact_date_pairs
        and (
            pd.to_datetime(prices["required_options_date"]).dt.date
            < pd.to_datetime(prices["signal_date"]).dt.date
        ).all()
    )
    downstream_names = [
        "canonical field mapping",
        "symbol coverage from provider response",
        "pagination and raw-response hashes",
        "primary expiry/common-strike selection and tie-breaks",
        "ATM IV, straddle midpoint, spreads and DTE",
        "provider moneyness consistency",
        "causal options join",
        "underlying movement outcomes",
        "O0/O1 development-only preprocessing and coefficients",
        "manual probability reconstruction on 100 rows",
        "binary and continuous metrics",
        "matched controls",
        "25-draw session bootstrap",
        "five route-bundle null refits",
        "full-data determinism comparison",
    ]
    downstream = {
        name: "not_applicable_blocked_missing_eodhd_api_token" for name in downstream_names
    }
    passed = bool(
        decision["decision"] == "blocked_missing_eodhd_api_token"
        and download["requests_completed"] == 0
        and download["raw_records"] == 0
        and not download["pagination_complete"]
        and preflight["requests_completed"] == 0
        and reconstruction["passed"]
        and reconstruction["row_identity_mismatches"] == 0
        and reconstruction["route_state_mismatches"] == 0
        and float(reconstruction["maximum_difference"]) <= 1e-12
        and len(clean) == 87_443
        and date_mapping_passed
        and pairs.empty
        and movement.empty
        and predictions.empty
    )
    return {
        "passed": passed,
        "blocker_verified": True,
        "exact_previous_session_date_mapping": date_mapping_passed,
        "no_provider_requests": download["requests_completed"] == 0,
        "no_fabricated_pairs_or_movement": pairs.empty and movement.empty and predictions.empty,
        "downstream_checks": downstream,
    }


def _audit_historical_date_blocked_run(
    output: Path, *, clean: pd.DataFrame, required: pd.DataFrame
) -> dict[str, Any]:
    """Independently verify cached live evidence for the historical-date blocker."""

    decision = _read_json(output / "decision.json")
    download = _read_json(output / "options_download_manifest.json")
    preflight = _read_json(output / "eodhd_options_api_preflight.json")
    schema = _read_json(output / "eodhd_options_schema_mapping.json")
    mapping = pd.read_csv(output / "underlying_symbol_mapping.csv")
    pairs = pd.read_parquet(output / "selected_option_pairs.parquet")
    movement = pd.read_parquet(output / "options_movement_panel.parquet")
    predictions = pd.read_parquet(output / "assessment_predictions.parquet")
    requested = str(preflight.get("requested_session_date", ""))
    required_dates = {value.isoformat() for value in required["required_options_date"]}
    mapping_supported = mapping["coverage_available"].astype(str).str.casefold().eq("true")
    setup_rows = cast(list[dict[str, Any]], preflight.get("setup_manifest_rows", []))

    required_manifest_fields = {
        "request_id",
        "endpoint",
        "underlying_symbol",
        "trade_date_from",
        "trade_date_to",
        "strike_from",
        "strike_to",
        "expiration_from",
        "expiration_to",
        "offset",
        "limit",
        "response_status",
        "record_count",
        "response_hash",
        "response_bytes",
        "attempts",
        "started_at",
        "completed_at",
        "cache_path",
    }
    cached_responses: dict[str, bytes] = {}
    manifest_hashes_passed = len(setup_rows) == 2
    query_identity_passed = len(setup_rows) == 2
    audit_token = os.environ.get("EODHD_API_TOKEN", "").encode("utf-8")
    cached_token_value_absent = True
    for row in setup_rows:
        cache_path = Path(str(row.get("cache_path", "")))
        response_hash = str(row.get("response_hash", ""))
        endpoint = str(row.get("endpoint", ""))
        if endpoint == "/mp/unicornbay/options/underlying-symbols":
            expected_params: dict[str, object] | None = {"fmt": "json"}
            expected_query_fields = {
                "underlying_symbol": "",
                "trade_date_from": "",
                "trade_date_to": "",
                "offset": 0,
                "limit": 0,
            }
        elif endpoint == "/mp/unicornbay/options/eod":
            expected_params = {
                "filter[underlying_symbol]": str(preflight.get("symbol", "")),
                "filter[tradetime_from]": requested,
                "filter[tradetime_to]": requested,
                "page[offset]": 0,
                "page[limit]": 10,
                "compact": 0,
                "fmt": "json",
            }
            expected_query_fields = {
                "underlying_symbol": str(preflight.get("symbol", "")),
                "trade_date_from": requested,
                "trade_date_to": requested,
                "offset": 0,
                "limit": 10,
            }
        else:
            expected_params = None
            expected_query_fields = {}
        expected_request_id = ""
        if expected_params is not None:
            request_identity = json.dumps(
                {"endpoint": endpoint, "params": expected_params},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            expected_request_id = hashlib.sha256(request_identity).hexdigest()
        row_query_identity_passed = bool(
            expected_params is not None
            and str(row.get("request_id", "")) == expected_request_id
            and all(row.get(key) == value for key, value in expected_query_fields.items())
        )
        row_valid = bool(
            required_manifest_fields.issubset(row)
            and re.fullmatch(r"[0-9a-f]{64}", response_hash)
            and row_query_identity_passed
            and int(row.get("response_status", 0)) == 200
            and int(row.get("attempts", 0)) >= 1
            and cache_path.is_file()
        )
        if row_valid:
            content = cache_path.read_bytes()
            token_value_absent = not audit_token or audit_token not in content
            row_valid = bool(
                hashlib.sha256(content).hexdigest() == response_hash
                and len(content) == int(row.get("response_bytes", -1))
                and b"api_token" not in content.lower()
                and token_value_absent
            )
            cached_token_value_absent = cached_token_value_absent and token_value_absent
            if row_valid:
                cached_responses[endpoint] = content
        manifest_hashes_passed = manifest_hashes_passed and row_valid
        query_identity_passed = query_identity_passed and row_query_identity_passed

    eod_content = cached_responses.get("/mp/unicornbay/options/eod", b"")
    try:
        eod_payload = json.loads(eod_content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        eod_payload = {}
    raw_data = eod_payload.get("data", []) if isinstance(eod_payload, dict) else []
    raw_meta = eod_payload.get("meta", {}) if isinstance(eod_payload, dict) else {}
    raw_links = eod_payload.get("links", {}) if isinstance(eod_payload, dict) else {}
    independently_reconstructed_pagination = {
        "offset": raw_meta.get("offset") if isinstance(raw_meta, dict) else None,
        "limit": raw_meta.get("limit") if isinstance(raw_meta, dict) else None,
        "total": raw_meta.get("total") if isinstance(raw_meta, dict) else None,
        "next_present": bool(raw_links.get("next")) if isinstance(raw_links, dict) else False,
    }
    independent_evidence: list[dict[str, object]] = []

    def local_session_date(value: object) -> str | None:
        if value in {None, ""}:
            return None
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("America/New_York")
        else:
            timestamp = timestamp.tz_convert("America/New_York")
        return timestamp.date().isoformat()

    if isinstance(raw_data, list):
        for index, item in enumerate(raw_data):
            if not isinstance(item, dict):
                continue
            attributes_value = item.get("attributes", item)
            attributes = attributes_value if isinstance(attributes_value, dict) else {}
            resource_id = item.get("id")
            resource_date: str | None = None
            if isinstance(resource_id, str) and len(resource_id) >= 10:
                try:
                    resource_date = date.fromisoformat(resource_id[-10:]).isoformat()
                except ValueError:
                    resource_date = None
            independent_evidence.append(
                {
                    "record_index": index,
                    "resource_id_sha256": (
                        hashlib.sha256(resource_id.encode("utf-8")).hexdigest()
                        if isinstance(resource_id, str)
                        else None
                    ),
                    "resource_observation_date": resource_date,
                    "tradetime_date": local_session_date(attributes.get("tradetime")),
                    "bid_observation_date": local_session_date(attributes.get("bid_date")),
                    "ask_observation_date": local_session_date(attributes.get("ask_date")),
                    "expiration_date": local_session_date(attributes.get("exp_date")),
                    "provider_dte": attributes.get("dte"),
                }
            )
    projection_bytes = json.dumps(
        independent_evidence,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    projection_hash = hashlib.sha256(projection_bytes).hexdigest()
    projection_passed = bool(
        independent_evidence == preflight.get("record_evidence")
        and projection_hash == preflight.get("evidence_projection_sha256")
        and len(independent_evidence) == int(preflight.get("records_received", -1))
        and hashlib.sha256(eod_content).hexdigest() == preflight.get("response_sha256")
        and independently_reconstructed_pagination == preflight.get("pagination")
    )
    resource_dates = {
        str(row["resource_observation_date"])
        for row in independent_evidence
        if row["resource_observation_date"] is not None
    }
    bid_dates = {
        str(row["bid_observation_date"])
        for row in independent_evidence
        if row["bid_observation_date"] is not None
    }
    ask_dates = {
        str(row["ask_observation_date"])
        for row in independent_evidence
        if row["ask_observation_date"] is not None
    }
    tradetime_dates = {
        str(row["tradetime_date"])
        for row in independent_evidence
        if row["tradetime_date"] is not None
    }
    setup_http_attempts = sum(int(row.get("attempts", 0)) for row in setup_rows)
    setup_response_bytes = sum(int(row.get("response_bytes", 0)) for row in setup_rows)
    setup_records_received = sum(int(row.get("record_count", 0)) for row in setup_rows)
    accounting_passed = bool(
        setup_http_attempts == int(preflight.get("setup_http_requests_attempted", -1))
        and len(setup_rows) == int(preflight.get("setup_logical_requests_completed", -1))
        and setup_rows == download.get("setup_manifest_rows")
        and setup_http_attempts == int(download.get("setup_http_requests_attempted", -1))
        and setup_response_bytes == int(download.get("setup_response_bytes", -1))
        and setup_records_received == int(download.get("setup_records_received", -1))
        and int(preflight.get("records_received", -1))
        == int(download.get("preflight_option_records", -2))
    )
    evidence_passed = bool(
        preflight.get("status") == "blocked_historical_options_date_unavailable"
        and preflight.get("endpoint") == "/mp/unicornbay/options/eod"
        and 1 <= int(preflight.get("records_received", 0)) <= 10
        and requested in required_dates
        and resource_dates
        and resource_dates == bid_dates == ask_dates
        and requested not in resource_dates
        and tradetime_dates == {requested}
        and preflight.get("all_returned_rows_observation_date_verified") is True
        and preflight.get("sample_rows_match_requested_observation_date") is False
        and preflight.get("tradetime_is_eod_observation_date") is False
        and preflight.get("official_observation_date_filter_available") is False
        and preflight.get("exact_requested_observation_date_confirmed") is False
        and preflight.get("america_new_york_mapping_confirmed") is True
        and preflight.get("authentication_redacted") is True
        and manifest_hashes_passed
        and query_identity_passed
        and projection_passed
        and accounting_passed
    )
    schema_passed = bool(
        schema["historical_eod"]["tradetime_filter_semantics"] == "last-trade activity window"
        and "not present" in schema["historical_eod"]["historical_observation_date_filter"]
        and "never to trade_date" in schema["canonical_mapping"]["tradetime"]
        and "maps to trade_date" in schema["canonical_mapping"]["resource.id"]
    )
    no_downstream_materialisation = bool(
        int(download.get("requests_completed", -1)) == 0
        and int(download.get("raw_records", -1)) == 0
        and download.get("manifest_rows") == []
        and pairs.empty
        and movement.empty
        and predictions.empty
    )
    passed = bool(
        decision["decision"] == "blocked_historical_options_date_unavailable"
        and int(decision.get("provider_setup_http_requests_attempted", -1)) == setup_http_attempts
        and int(decision.get("provider_setup_logical_requests_completed", -1)) == len(setup_rows)
        and evidence_passed
        and schema_passed
        and mapping_supported.sum() == len(FROZEN_COHORT)
        and len(clean) == 87_443
        and no_downstream_materialisation
    )
    return {
        "passed": passed,
        "blocker_verified": evidence_passed,
        "provider_setup_http_requests_attempted": setup_http_attempts,
        "provider_setup_logical_requests_completed": len(setup_rows),
        "bulk_page_requests_completed": int(download.get("requests_completed", 0)),
        "preflight_option_records": int(preflight.get("records_received", 0)),
        "setup_response_bytes": setup_response_bytes,
        "cached_response_hashes_verified": manifest_hashes_passed,
        "cached_response_token_scan_performed": bool(audit_token),
        "cached_response_token_value_absent": cached_token_value_absent,
        "setup_request_identities_verified": query_identity_passed,
        "evidence_projection_verified": projection_passed,
        "request_accounting_verified": accounting_passed,
        "pagination_metadata_verified": (
            independently_reconstructed_pagination == preflight.get("pagination")
        ),
        "requested_session_date": requested,
        "tradetime_dates": sorted(tradetime_dates),
        "returned_resource_observation_dates": sorted(resource_dates),
        "quote_observation_dates_match_resources": resource_dates == bid_dates == ask_dates,
        "official_observation_date_filter_absent": schema_passed,
        "covered_symbols": int(mapping_supported.sum()),
        "no_bulk_or_downstream_materialisation": no_downstream_materialisation,
    }


def _audit_coverage_blocked_run(output: Path, *, clean: pd.DataFrame) -> dict[str, Any]:
    """Independently verify the frozen coverage blocker before any fitted model."""

    decision = _read_json(output / "decision.json")
    download = _audit_download_integrity(output)
    pairs = pd.read_parquet(output / "selected_option_pairs.parquet")
    movement = pd.read_parquet(output / "options_movement_panel.parquet")
    predictions = pd.read_parquet(output / "assessment_predictions.parquet")
    mapping = pd.read_csv(output / "underlying_symbol_mapping.csv")
    mapping_available = mapping["coverage_available"].astype(str).str.casefold().eq(
        "true"
    ) & pd.to_numeric(mapping["records_returned"], errors="coerce").fillna(0).gt(0)
    joined = clean.copy()
    joined["signal_date"] = pd.to_datetime(joined["session"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    joined = joined.merge(
        pairs[["symbol", "signal_date"]].assign(valid_pair=1),
        on=["symbol", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    joined["valid_pair"] = joined["valid_pair"].fillna(0).astype(int)
    assessment = joined.loc[joined["period"].eq("assessment") & joined["valid_pair"].eq(1)]
    development = joined.loc[joined["period"].eq("development") & joined["valid_pair"].eq(1)]
    paired = (
        joined.groupby(["symbol", "period"], sort=True)["valid_pair"].sum().unstack(fill_value=0)
    )
    paired_both = int((paired.reindex(columns=["development", "assessment"]).min(axis=1) > 0).sum())
    shares = (
        assessment.groupby("symbol", sort=True)["row_weight"].sum() / assessment["row_weight"].sum()
        if len(assessment)
        else pd.Series(dtype=float)
    )
    evidence: dict[str, Any] = {
        "historical_symbols": int(mapping_available.sum()),
        "paired_symbols_development": paired_both,
        "paired_symbols_assessment": paired_both,
        "development_row_coverage": len(development)
        / max(int(clean["period"].eq("development").sum()), 1),
        "assessment_row_coverage": len(assessment)
        / max(int(clean["period"].eq("assessment").sum()), 1),
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "assessment_broad_conflict_rows": int(
            assessment["route_resolution_state"].eq("BROAD_CONFLICT").sum()
        ),
        "assessment_low_route_support_rows": int(
            assessment["route_resolution_state"].eq("LOW_ROUTE_SUPPORT").sum()
        ),
        "maximum_stock_weight_share": float(shares.max()) if len(shares) else 1.0,
        "download_integrity_passed": bool(download["passed"]),
    }
    expected = cast(dict[str, Any], decision["coverage_evidence"])
    evidence_matches = all(
        (
            math_isclose(float(evidence[key]), float(expected[key]))
            if isinstance(evidence[key], (int, float))
            else evidence[key] == expected[key]
        )
        for key in evidence
    )
    chronology_passed = bool(
        pairs.empty
        or (
            pd.to_datetime(pairs["required_options_date"]).dt.date
            < pd.to_datetime(pairs["signal_date"]).dt.date
        ).all()
    )
    passed = bool(
        decision["decision"] == "blocked_insufficient_options_chain_coverage"
        and download["passed"]
        and evidence_matches
        and not coverage_gates_pass(evidence)
        and chronology_passed
        and movement.empty
        and predictions.empty
    )
    return {
        "passed": passed,
        "blocker_verified": True,
        "download_integrity": download,
        "coverage_evidence_recomputed": evidence,
        "coverage_evidence_matches": evidence_matches,
        "coverage_gate_passed": coverage_gates_pass(evidence),
        "chronology_passed": chronology_passed,
        "no_fitted_predictions_or_movement": movement.empty and predictions.empty,
    }


def _audit_download_integrity(output: Path) -> dict[str, Any]:
    manifest = _read_json(output / "options_download_manifest.json")
    rows = cast(list[dict[str, Any]], manifest.get("manifest_rows", []))
    hash_failures = 0
    status_failures = 0
    record_count_failures = 0
    request_id_failures = 0
    page_details: list[dict[str, Any]] = []
    for row in rows:
        path = Path(str(row["cache_path"]))
        if not path.is_file():
            hash_failures += 1
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["response_hash"]:
            hash_failures += 1
            continue
        if int(row["response_status"]) != 200:
            status_failures += 1
        payload = json.loads(content)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != int(row["record_count"]):
            record_count_failures += 1
        params: dict[str, object] = {
            "filter[underlying_symbol]": row["underlying_symbol"],
            "page[offset]": int(row["offset"]),
            "page[limit]": int(row["limit"]),
            "compact": 1,
            "fmt": "json",
            "fields[options-eod]": ",".join(OPTIONS_FIELDS),
        }
        for parameter, field in (
            ("filter[tradetime_from]", "trade_date_from"),
            ("filter[tradetime_to]", "trade_date_to"),
            ("filter[strike_from]", "strike_from"),
            ("filter[strike_to]", "strike_to"),
            ("filter[exp_date_from]", "expiration_from"),
            ("filter[exp_date_to]", "expiration_to"),
        ):
            if row.get(field) is not None:
                params[parameter] = row[field]
        request_id_failures += int(
            stable_request_id("/mp/unicornbay/options/eod", params) != row["request_id"]
        )
        page_details.append(
            {
                **row,
                "meta_total": int(payload["meta"]["total"]),
                "next_present": bool(payload["links"].get("next")),
            }
        )
    group_columns = [
        "underlying_symbol",
        "trade_date_from",
        "trade_date_to",
        "strike_from",
        "strike_to",
        "expiration_from",
        "expiration_to",
        "limit",
    ]
    pagination_failures = 0
    details = pd.DataFrame(page_details)
    if len(details):
        superseded = (
            details["superseded_by_split"].astype(bool)
            if "superseded_by_split" in details
            else pd.Series(False, index=details.index)
        )
        usable = details.loc[~superseded]
    else:
        usable = details
    if len(usable):
        for _key, pages in usable.groupby(group_columns, sort=True, dropna=False):
            ordered = pages.sort_values("offset", kind="mergesort")
            expected_offset = 0
            for page in ordered.itertuples(index=False):
                pagination_failures += int(int(page.offset) != expected_offset)
                expected_offset += int(page.record_count)
            final = ordered.iloc[-1]
            pagination_failures += int(expected_offset != int(final["meta_total"]))
            pagination_failures += int(bool(final["next_present"]))
    raw_count = sum(int(row["record_count"]) for row in rows)
    passed = bool(
        manifest.get("status") == "supported"
        and manifest.get("pagination_complete") is True
        and int(manifest.get("unexplained_truncations", -1)) == 0
        and int(manifest.get("credential_exposures", -1)) == 0
        and raw_count == int(manifest.get("raw_records", -1))
        and raw_count <= 3_000_000
        and int(manifest.get("download_bytes", 20_000_000_001)) <= 20_000_000_000
        and hash_failures == 0
        and status_failures == 0
        and record_count_failures == 0
        and request_id_failures == 0
        and pagination_failures == 0
    )
    return {
        "passed": passed,
        "manifest_rows": len(rows),
        "raw_records_recounted": raw_count,
        "hash_failures": hash_failures,
        "status_failures": status_failures,
        "record_count_failures": record_count_failures,
        "request_id_failures": request_id_failures,
        "pagination_failures": pagination_failures,
        "superseded_split_probe_rows": sum(
            bool(row.get("superseded_by_split", False)) for row in rows
        ),
    }


def _load_plan_canonical(output: Path) -> pd.DataFrame:
    source = _read_json(output / "source_manifest.json")
    plan = _read_json(output / "options_request_plan.json")
    mapping = pd.read_csv(output / "underlying_symbol_mapping.csv")
    mapped_mask = mapping["coverage_available"].astype(str).str.casefold().eq("true")
    mapped_symbols = set(mapping.loc[mapped_mask, "stocker_symbol"].astype(str))
    expected = {
        str(chunk["chunk_id"])
        for chunk in cast(list[dict[str, Any]], plan["chunks"])
        if str(chunk["underlying_symbol"]) in mapped_symbols
    }
    root = Path(str(source["options_cache_path"])) / "canonical"
    paths = sorted(path for path in root.glob("*.parquet") if path.stem in expected)
    if {path.stem for path in paths} != expected or not paths:
        raise AssertionError("plan-owned canonical options cache is missing current chunks")
    canonical = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    canonical["trade_date"] = pd.to_datetime(canonical["trade_date"]).dt.date
    canonical["expiration_date"] = pd.to_datetime(canonical["expiration_date"]).dt.date
    return canonical


def _audit_canonical_from_raw(output: Path, canonical: pd.DataFrame) -> dict[str, Any]:
    """Recanonicalize non-superseded raw pages and reapply deterministic deduplication."""

    manifest = _read_json(output / "options_download_manifest.json")
    rows = cast(list[dict[str, Any]], manifest["manifest_rows"])
    mapping = pd.read_csv(output / "underlying_symbol_mapping.csv")
    available = mapping["coverage_available"].astype(str).str.casefold().eq("true")
    reverse_mapping = {
        str(row.eodhd_underlying_symbol).casefold(): str(row.stocker_symbol)
        for row in mapping.loc[available].itertuples(index=False)
    }
    plan = _read_json(output / "options_request_plan.json")
    required_dates = {
        date.fromisoformat(str(value))
        for chunk in cast(list[dict[str, Any]], plan["chunks"])
        for value in cast(list[str], chunk["required_trade_dates"])
    }
    records: list[dict[str, Any]] = []
    rejections: list[tuple[str, str]] = []
    mapping_failures = 0
    for row in rows:
        if bool(row.get("superseded_by_split", False)):
            continue
        payload = json.loads(Path(str(row["cache_path"])).read_text(encoding="utf-8"))
        fields = payload["meta"].get("fields")
        expanded: list[dict[str, Any]] = []
        for item in payload["data"]:
            if isinstance(item, dict):
                expanded.append(item)
            elif isinstance(item, list) and isinstance(fields, list) and len(item) == len(fields):
                expanded.append(
                    {
                        "type": "options-eod",
                        "attributes": dict(zip(fields, item, strict=True)),
                    }
                )
            else:
                raise AssertionError("raw options row cannot be expanded from meta.fields")
        result = canonicalize_response_records(
            expanded,
            request_id="independent-raw-audit",
            provider_schema_version="openapi-2.0.0",
        )
        rejections.extend((item.raw_record_hash, item.reason_code) for item in result.rejections)
        for record in result.records:
            if record["trade_date"] not in required_dates or not 7 <= int(record["dte"]) <= 90:
                continue
            provider_symbol = str(record["underlying_symbol"])
            stocker_symbol = reverse_mapping.get(provider_symbol.casefold())
            if stocker_symbol is None:
                mapping_failures += 1
                continue
            record["provider_underlying_symbol"] = provider_symbol
            record["underlying_symbol"] = stocker_symbol
            records.append(record)
    deduplicated = resolve_canonical_duplicates(records)
    rebuilt = pd.DataFrame(deduplicated.records)
    key_columns = ["underlying_symbol", "contract_id", "trade_date", "raw_record_hash"]
    rebuilt = rebuilt.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    stored = canonical.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    identity_matches = bool(
        len(rebuilt) == len(stored)
        and rebuilt[key_columns].astype(str).equals(stored[key_columns].astype(str))
    )
    text_columns = [
        "provider",
        "provider_schema_version",
        "underlying_symbol",
        "contract_id",
        "option_type",
        "expiration_date",
        "trade_date",
        "provider_underlying_symbol",
    ]
    text_matches = bool(
        identity_matches
        and rebuilt[text_columns].astype(str).equals(stored[text_columns].astype(str))
    )
    timestamp_matches = bool(
        identity_matches
        and pd.to_datetime(rebuilt["trade_timestamp"], utc=True).equals(
            pd.to_datetime(stored["trade_timestamp"], utc=True)
        )
    )
    maximum_numeric_difference = 0.0
    numeric_columns = [
        "strike",
        "last",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "midpoint",
        "volume",
        "open_interest",
        "implied_volatility",
        "theoretical_value",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "dte",
        "moneyness",
        "underlying_reference_price",
    ]
    if identity_matches:
        for column in numeric_columns:
            left = pd.to_numeric(rebuilt[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(stored[column], errors="coerce").to_numpy(float)
            if np.logical_xor(np.isnan(left), np.isnan(right)).any():
                maximum_numeric_difference = math.inf
                break
            difference = np.abs(left - right)
            difference[np.isnan(left) & np.isnan(right)] = 0.0
            maximum_numeric_difference = max(
                maximum_numeric_difference,
                float(np.nanmax(difference)) if difference.size else 0.0,
            )
    else:
        maximum_numeric_difference = math.inf
    rejection_artifact = pd.read_csv(output / "options_rejections.csv")
    stored_rejections = sorted(
        zip(
            rejection_artifact["raw_record_hash"].astype(str),
            rejection_artifact["reason_code"].astype(str),
            strict=True,
        )
    )
    rejection_matches = sorted(rejections) == stored_rejections
    passed = bool(
        identity_matches
        and text_matches
        and timestamp_matches
        and maximum_numeric_difference <= 1e-12
        and mapping_failures == 0
        and deduplicated.duplicate_records == int(manifest["duplicate_records"])
        and rejection_matches
    )
    return {
        "passed": passed,
        "raw_records_recanonicalized": len(records),
        "canonical_records_rebuilt": len(rebuilt),
        "identity_matches": identity_matches,
        "text_fields_match": text_matches,
        "trade_timestamps_match": timestamp_matches,
        "maximum_numeric_difference": maximum_numeric_difference,
        "duplicate_records_recomputed": deduplicated.duplicate_records,
        "mapping_failures": mapping_failures,
        "rejections_match": rejection_matches,
    }


def _manual_probabilities(frame: pd.DataFrame, specification: Mapping[str, Any]) -> np.ndarray:
    features = [str(value) for value in specification["numeric_features"]]
    values = frame[features].to_numpy(float)
    medians = np.asarray(specification["numeric_medians"], dtype=float)
    means = np.asarray(specification["numeric_means"], dtype=float)
    scales = np.asarray(specification["numeric_scales"], dtype=float)
    parts = [(np.where(np.isfinite(values), values, medians) - means) / scales]
    controls = {
        "stock": frame["symbol"].astype(str),
        "checkpoint": frame["checkpoint"].astype(str),
        "month_of_year": frame["year_month"].astype(str).str[-2:],
    }
    for control, levels_value in cast(
        dict[str, list[str]], specification["category_levels"]
    ).items():
        levels = [str(value) for value in levels_value]
        observed = controls[control].to_numpy()
        for level in levels[1:]:
            parts.append(np.asarray(observed == level, dtype=float)[:, None])
    design = np.concatenate(parts, axis=1)
    coefficient = np.asarray(specification["coefficients"], dtype=float)
    linear = design @ coefficient + float(specification["intercept"])
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(linear, -709.0, 709.0))), dtype=float)


def _audit_model_preprocessing(
    movement: pd.DataFrame,
    coefficients: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> dict[str, Any]:
    development = movement.loc[movement["period"].eq("development")].copy()
    development_years_valid = bool(
        pd.to_datetime(development["session"], errors="raise").dt.year.eq(2024).all()
    )
    expected_features = {
        "O0": [*OPTIONS_PRIMARY_FEATURES, *DENSE_H0_FEATURES],
        "R0": [*OPTIONS_PRIMARY_FEATURES, *DENSE_H0_FEATURES],
        "O1": [*OPTIONS_PRIMARY_FEATURES, *DENSE_H0_FEATURES, *ROUTE_FEATURES],
        "R1": [*OPTIONS_PRIMARY_FEATURES, *DENSE_H0_FEATURES, *ROUTE_FEATURES],
    }
    controls = {
        "stock": development["symbol"].astype(str),
        "checkpoint": development["checkpoint"].astype(str),
        "month_of_year": development["year_month"].astype(str).str[-2:],
    }
    maximum_difference = 0.0
    configuration_mismatches = 0
    primary_models = cast(dict[str, Any], coefficients["primary_models"])
    for model_id, features in expected_features.items():
        specification = cast(dict[str, Any], primary_models[model_id])
        configuration_mismatches += int(specification["numeric_features"] != features)
        configuration_mismatches += int(
            specification["preprocessing_fitted_period"] != "development_2024_only"
        )
        raw = development[features].to_numpy(float)
        finite = np.where(np.isfinite(raw), raw, np.nan)
        medians = np.nanmedian(finite, axis=0)
        imputed = np.where(np.isfinite(raw), raw, medians)
        means = imputed.mean(axis=0)
        scales = np.where(imputed.std(axis=0, ddof=0) >= 1e-12, imputed.std(axis=0), 1.0)
        for expected, stored in (
            (medians, np.asarray(specification["numeric_medians"], dtype=float)),
            (means, np.asarray(specification["numeric_means"], dtype=float)),
            (scales, np.asarray(specification["numeric_scales"], dtype=float)),
        ):
            maximum_difference = max(maximum_difference, float(np.max(np.abs(expected - stored))))
        stored_levels = cast(dict[str, list[str]], specification["category_levels"])
        for name, values in controls.items():
            configuration_mismatches += int(stored_levels[name] != sorted(values.unique().tolist()))
        configuration_mismatches += int(
            len(specification["design_columns"]) != len(specification["coefficients"])
        )
    coefficient_determinism = bool(
        determinism.get("passed") is True
        and float(determinism.get("maximum_coefficient_difference", math.inf)) <= 1e-12
    )
    return {
        "passed": development_years_valid
        and configuration_mismatches == 0
        and maximum_difference <= 1e-12
        and coefficient_determinism,
        "development_rows": len(development),
        "development_years_valid": development_years_valid,
        "configuration_mismatches": configuration_mismatches,
        "maximum_preprocessing_difference": maximum_difference,
        "coefficient_determinism": coefficient_determinism,
    }


def _audit_movement_sample(movement: pd.DataFrame) -> dict[str, Any]:
    """Independently reconstruct 100 primary three-bar outcomes and timestamps."""

    sample = movement.sort_values("row_id", kind="mergesort").head(100)
    bars = pd.read_parquet(
        TRACE_PANEL,
        columns=[
            "symbol",
            "session",
            "bar_ordinal",
            "bar_start_timestamp",
            "bar_complete_timestamp",
            "open",
            "high",
            "low",
            "close",
        ],
    )
    groups = {
        (str(symbol), str(session)): group.set_index("bar_ordinal", drop=False)
        for (symbol, session), group in bars.groupby(["symbol", "session"], sort=False)
    }
    maximum_difference = 0.0
    timestamp_mismatches = 0
    for row in sample.itertuples(index=False):
        checkpoint = int(row.checkpoint_bar_ordinal_zero_based)
        future = groups[(str(row.symbol), str(row.session))].loc[
            [checkpoint + 1, checkpoint + 2, checkpoint + 3]
        ]
        entry = float(future.iloc[0]["open"])
        closes = future["close"].to_numpy(float)
        highs = future["high"].to_numpy(float)
        lows = future["low"].to_numpy(float)
        five_minute_returns = np.asarray(
            [
                math.log(closes[0] / entry),
                math.log(closes[1] / closes[0]),
                math.log(closes[2] / closes[1]),
            ]
        )
        expected = {
            "entry_price": entry,
            "absolute_log_return_15m": abs(math.log(closes[2] / entry)),
            "realised_range_15m": math.log(max(highs) / min(lows)),
            "maximum_absolute_excursion_15m": max(
                abs(math.log(max(highs) / entry)),
                abs(math.log(min(lows) / entry)),
            ),
            "realised_variance_15m": float(np.sum(five_minute_returns**2)),
        }
        maximum_difference = max(
            maximum_difference,
            *(abs(float(getattr(row, key)) - value) for key, value in expected.items()),
        )
        timestamp_mismatches += int(
            pd.Timestamp(row.entry_bar_start_timestamp)
            != pd.Timestamp(future.iloc[0]["bar_start_timestamp"])
        )
        timestamp_mismatches += int(
            pd.Timestamp(row.primary_horizon_last_bar_complete_timestamp)
            != pd.Timestamp(future.iloc[-1]["bar_complete_timestamp"])
        )
    return {
        "passed": len(sample) == 100 and maximum_difference <= 1e-12 and timestamp_mismatches == 0,
        "rows": len(sample),
        "maximum_outcome_difference": maximum_difference,
        "timestamp_mismatches": timestamp_mismatches,
    }


def _audit_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    weight = pd.to_numeric(weights, errors="raise").to_numpy(float)
    valid = np.isfinite(numeric) & np.isfinite(weight) & (weight > 0.0)
    return float(np.average(numeric[valid], weights=weight[valid])) if valid.any() else math.nan


def _audit_finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _audit_increment_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["movement_exceeds_iv_expected_absolute"].to_numpy(int)
    weights = frame["row_weight"].to_numpy(float)
    o0 = frame["O0_probability"].to_numpy(float)
    o1 = frame["O1_probability"].to_numpy(float)

    def precision(model: str) -> float:
        mask = frame[f"{model}_top_decile"].to_numpy(bool)
        return float(np.average(target[mask], weights=weights[mask]))

    return {
        "log_loss_improvement": float(
            log_loss(target, o0, sample_weight=weights, labels=[0, 1])
            - log_loss(target, o1, sample_weight=weights, labels=[0, 1])
        ),
        "brier_improvement": float(
            brier_score_loss(target, o0, sample_weight=weights)
            - brier_score_loss(target, o1, sample_weight=weights)
        ),
        "auc_improvement": float(
            roc_auc_score(target, o1, sample_weight=weights)
            - roc_auc_score(target, o0, sample_weight=weights)
        ),
        "average_precision_improvement": float(
            average_precision_score(target, o1, sample_weight=weights)
            - average_precision_score(target, o0, sample_weight=weights)
        ),
        "top_decile_precision_improvement": precision("O1") - precision("O0"),
    }


def _matched_effects(assessment: pd.DataFrame, relations: pd.DataFrame) -> dict[str, float]:
    indexed = assessment.set_index("row_id", drop=False)
    columns = (
        "absolute_log_return_15m",
        "iv_absolute_residual_15m",
        "iv_sigma_ratio_15m",
        "movement_exceeds_iv_expected_absolute",
        "realised_range_15m",
        "maximum_absolute_excursion_15m",
    )
    differences = {column: [] for column in columns}
    treated_weights: list[float] = []
    for treated_id, group in relations.groupby("treated_row_id", sort=True):
        treated = indexed.loc[str(treated_id)]
        controls = indexed.loc[group["control_row_id"].astype(str)]
        match_weights = group["match_weight"].to_numpy(float)
        for column in columns:
            differences[column].append(
                float(treated[column])
                - float(np.average(controls[column].to_numpy(float), weights=match_weights))
            )
        treated_weights.append(float(treated["row_weight"]))
    return {
        column: (float(np.average(values, weights=treated_weights)) if values else math.nan)
        for column, values in differences.items()
    }


def _audit_bootstrap(
    assessment: pd.DataFrame, relations: pd.DataFrame, artifact: pd.DataFrame
) -> dict[str, Any]:
    """Recalculate every fixed-prediction session draw and interval without refitting."""

    multiplicities = fixed_session_bootstrap_multiplicities(assessment, draws=25, seed=20260722)
    indexed = assessment.set_index("row_id", drop=False)
    expected: dict[tuple[int, str], float] = {}
    distributions: dict[str, list[float]] = {}
    sessions = assessment["session"].astype(str)
    whole_session_passed = True
    for draw, multiplicity in enumerate(multiplicities):
        multiplicity_series = pd.Series(multiplicity, index=assessment.index)
        whole_session_passed &= bool(multiplicity_series.groupby(sessions).nunique().le(1).all())
        sample = assessment.copy()
        sample["row_weight"] = sample["row_weight"].to_numpy(float) * multiplicity
        sample = sample.loc[sample["row_weight"].gt(0.0)]
        broad = sample.loc[sample["route_resolution_state"].eq("BROAD_CONFLICT")]
        low = sample.loc[sample["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
        values = {
            **_audit_increment_metrics(sample),
            "broad_conflict_mean_iv_absolute_residual": _audit_weighted_mean(
                broad["iv_absolute_residual_15m"], broad["row_weight"]
            ),
            "broad_minus_low_iv_absolute_residual": _audit_weighted_mean(
                broad["iv_absolute_residual_15m"], broad["row_weight"]
            )
            - _audit_weighted_mean(low["iv_absolute_residual_15m"], low["row_weight"]),
        }
        row_multiplicity = dict(zip(assessment["row_id"].astype(str), multiplicity, strict=True))
        matched_values: dict[str, list[float]] = {
            "iv_absolute_residual_15m": [],
            "movement_exceeds_iv_expected_absolute": [],
            "iv_sigma_ratio_15m": [],
        }
        matched_weights: list[float] = []
        for treated_id, group in relations.groupby("treated_row_id", sort=True):
            treated_multiplier = int(row_multiplicity.get(str(treated_id), 0))
            if treated_multiplier == 0:
                continue
            control_multiplicity = group["control_row_id"].astype(str).map(row_multiplicity)
            control_weights = group["match_weight"].to_numpy(float) * control_multiplicity.to_numpy(
                float
            )
            if control_weights.sum() <= 0.0:
                continue
            treated = indexed.loc[str(treated_id)]
            controls = indexed.loc[group["control_row_id"].astype(str)]
            for column in matched_values:
                matched_values[column].append(
                    float(treated[column])
                    - float(np.average(controls[column].to_numpy(float), weights=control_weights))
                )
            matched_weights.append(float(treated["row_weight"]) * treated_multiplier)
        for column, metric in (
            ("iv_absolute_residual_15m", "broad_minus_matched_iv_absolute_residual"),
            (
                "movement_exceeds_iv_expected_absolute",
                "broad_minus_matched_exceed_iv_rate",
            ),
            ("iv_sigma_ratio_15m", "broad_minus_matched_iv_sigma_ratio"),
        ):
            values[metric] = (
                float(np.average(matched_values[column], weights=matched_weights))
                if matched_weights
                else math.nan
            )
        for metric, value in values.items():
            expected[(draw, metric)] = value
            distributions.setdefault(metric, []).append(value)
    draw_rows = artifact.loc[artifact["record_type"].eq("draw")]
    draw_keys = set(
        zip(draw_rows["draw"].astype(int), draw_rows["metric"].astype(str), strict=True)
    )
    maximum_difference = 0.0
    if draw_keys != set(expected):
        maximum_difference = math.inf
    else:
        stored = {
            (int(row.draw), str(row.metric)): float(row.value)
            for row in draw_rows.itertuples(index=False)
        }
        differences: list[float] = []
        for key, value in expected.items():
            stored_value = stored[key]
            if math.isnan(value) != math.isnan(stored_value):
                differences.append(math.inf)
            elif math.isnan(value):
                differences.append(0.0)
            else:
                differences.append(abs(value - stored_value))
        maximum_difference = max(differences)
    interval_difference = 0.0
    interval_rows = artifact.loc[artifact["record_type"].eq("interval")]
    expected_interval_count = len(distributions) * 3
    if len(interval_rows) != expected_interval_count:
        interval_difference = math.inf
    else:
        interval_index = interval_rows.set_index(["metric", "interval_level"])
        for metric, values in distributions.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            for level, tail in ((0.80, 0.10), (0.90, 0.05), (0.95, 0.025)):
                stored = interval_index.loc[(metric, level)]
                expected_lower = float(np.quantile(finite, tail)) if len(finite) else math.nan
                expected_upper = float(np.quantile(finite, 1.0 - tail)) if len(finite) else math.nan
                stored_lower = float(stored["lower"])
                stored_upper = float(stored["upper"])
                if math.isnan(expected_lower) != math.isnan(stored_lower) or math.isnan(
                    expected_upper
                ) != math.isnan(stored_upper):
                    interval_difference = math.inf
                    continue
                interval_difference = max(
                    interval_difference,
                    0.0
                    if math.isnan(expected_lower) and math.isnan(stored_lower)
                    else abs(stored_lower - expected_lower),
                    0.0
                    if math.isnan(expected_upper) and math.isnan(stored_upper)
                    else abs(stored_upper - expected_upper),
                )
    return {
        "passed": whole_session_passed
        and maximum_difference <= 1e-12
        and interval_difference <= 1e-12,
        "draws": len(multiplicities),
        "whole_session_resampling": whole_session_passed,
        "maximum_draw_difference": maximum_difference,
        "maximum_interval_difference": interval_difference,
    }


def _audit_weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    ordered = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="raise"),
            "weight": pd.to_numeric(weights, errors="raise"),
        }
    ).sort_values("value", kind="mergesort")
    cumulative = ordered["weight"].cumsum() / ordered["weight"].sum()
    return float(ordered.loc[cumulative.ge(quantile), "value"].iloc[0])


def _route_bundle_counts(frame: pd.DataFrame) -> pd.Series:
    signature = pd.util.hash_pandas_object(frame[list(ROUTE_FEATURES)], index=False)
    keys = frame[["period", "session", "checkpoint"]].astype(str).copy()
    keys["route_bundle_signature"] = signature.to_numpy()
    return keys.value_counts(sort=False).sort_index()


def _audit_route_null(
    movement: pd.DataFrame,
    assessment: pd.DataFrame,
    coefficients: Mapping[str, Any],
    artifact: pd.DataFrame,
) -> dict[str, Any]:
    """Verify intact within-slate permutations and stored null predictions without refits."""

    original_bundles = _route_bundle_counts(movement)
    reference = assessment[["row_id", "O0_probability", "O0_top_decile"]].copy()
    real_increment = _audit_increment_metrics(assessment)
    expected: dict[tuple[int, str], tuple[float, float, bool]] = {}
    bundle_preservation = True
    outcomes_preserved = True
    non_route_columns = [column for column in movement.columns if column not in ROUTE_FEATURES]
    null_models = cast(dict[str, Any], coefficients["route_null_models"])
    for draw in range(5):
        permuted = permute_intact_route_bundle(movement, seed=20260722 + draw)
        bundle_preservation &= _route_bundle_counts(permuted).equals(original_bundles)
        outcomes_preserved &= permuted[non_route_columns].equals(movement[non_route_columns])
        null_development = permuted.loc[permuted["period"].eq("development")].copy()
        null_assessment = permuted.loc[permuted["period"].eq("assessment")].copy()
        specification = cast(dict[str, Any], null_models[str(draw)])
        development_probability = _manual_probabilities(null_development, specification)
        null_assessment["O1_probability"] = _manual_probabilities(null_assessment, specification)
        null_assessment = null_assessment.merge(
            reference, on="row_id", how="left", validate="one_to_one"
        )
        null_assessment["O1_top_decile"] = null_assessment["O1_probability"].ge(
            _audit_weighted_quantile(
                pd.Series(development_probability),
                null_development["row_weight"].reset_index(drop=True),
                0.90,
            )
        )
        increment = _audit_increment_metrics(null_assessment)
        for metric in (
            "log_loss_improvement",
            "brier_improvement",
            "auc_improvement",
            "average_precision_improvement",
        ):
            expected[(draw, metric)] = (
                real_increment[metric],
                increment[metric],
                real_increment[metric] > increment[metric],
            )
    rows = artifact.loc[artifact["record_type"].eq("draw")]
    stored = {
        (int(row.draw), str(row.metric)): (
            float(row.real_increment),
            float(row.null_increment),
            bool(row.real_exceeds_null),
        )
        for row in rows.itertuples(index=False)
    }
    maximum_difference = math.inf if set(stored) != set(expected) else 0.0
    boolean_mismatches = 0
    if set(stored) == set(expected):
        for key, values in expected.items():
            maximum_difference = max(
                maximum_difference,
                abs(values[0] - stored[key][0]),
                abs(values[1] - stored[key][1]),
            )
            boolean_mismatches += int(values[2] != stored[key][2])
    comparisons = artifact.loc[artifact["record_type"].eq("comparison")]
    stored_counts = {
        str(row.metric): int(row.real_exceeds_null_count)
        for row in comparisons.itertuples(index=False)
    }
    expected_counts = {
        metric: sum(expected[(draw, metric)][2] for draw in range(5))
        for metric in (
            "log_loss_improvement",
            "brier_improvement",
            "auc_improvement",
            "average_precision_improvement",
        )
    }
    comparison_passed = stored_counts == expected_counts
    return {
        "passed": bundle_preservation
        and outcomes_preserved
        and maximum_difference <= 1e-12
        and boolean_mismatches == 0
        and comparison_passed,
        "draws": 5,
        "intact_bundle_preservation": bundle_preservation,
        "outcomes_and_weights_preserved": outcomes_preserved,
        "maximum_increment_difference": maximum_difference,
        "boolean_mismatches": boolean_mismatches,
        "comparison_counts_match": comparison_passed,
        "refits_repeated_by_auditor": False,
    }


def _recompute_coverage_evidence(
    clean: pd.DataFrame,
    pairs: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    download_integrity_passed: bool,
) -> dict[str, Any]:
    joined = clean.copy()
    joined["signal_date"] = pd.to_datetime(joined["session"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    joined = joined.merge(
        pairs[["symbol", "signal_date"]].assign(valid_pair=1),
        on=["symbol", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    joined["valid_pair"] = joined["valid_pair"].fillna(0).astype(int)
    assessment = joined.loc[joined["period"].eq("assessment") & joined["valid_pair"].eq(1)]
    development = joined.loc[joined["period"].eq("development") & joined["valid_pair"].eq(1)]
    paired = (
        joined.groupby(["symbol", "period"], sort=True)["valid_pair"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=["development", "assessment"], fill_value=0)
    )
    paired_both = int((paired.min(axis=1) > 0).sum())
    shares = (
        assessment.groupby("symbol", sort=True)["row_weight"].sum() / assessment["row_weight"].sum()
        if len(assessment)
        else pd.Series(dtype=float)
    )
    mapping_available = mapping["coverage_available"].astype(str).str.casefold().eq(
        "true"
    ) & pd.to_numeric(mapping["records_returned"], errors="coerce").fillna(0).gt(0)
    evidence: dict[str, Any] = {
        "historical_symbols": int(mapping_available.sum()),
        "paired_symbols_development": paired_both,
        "paired_symbols_assessment": paired_both,
        "development_row_coverage": len(development)
        / max(int(clean["period"].eq("development").sum()), 1),
        "assessment_row_coverage": len(assessment)
        / max(int(clean["period"].eq("assessment").sum()), 1),
        "assessment_rows": len(assessment),
        "assessment_sessions": int(assessment["session"].nunique()),
        "assessment_months": int(assessment["year_month"].nunique()),
        "assessment_broad_conflict_rows": int(
            assessment["route_resolution_state"].eq("BROAD_CONFLICT").sum()
        ),
        "assessment_low_route_support_rows": int(
            assessment["route_resolution_state"].eq("LOW_ROUTE_SUPPORT").sum()
        ),
        "maximum_stock_weight_share": float(shares.max()) if len(shares) else 1.0,
        "download_integrity_passed": download_integrity_passed,
    }
    evidence["passed"] = coverage_gates_pass(evidence)
    return evidence


def _gate_payload_matches(
    recomputed: Mapping[str, Any], stored: Mapping[str, Any]
) -> tuple[bool, float]:
    keys_match = set(recomputed) == set(stored).difference({"passed"})
    maximum_difference = 0.0
    if not keys_match:
        return False, math.inf
    for key, value in recomputed.items():
        expected = stored[key]
        if isinstance(value, bool):
            if value is not bool(expected):
                return False, math.inf
        else:
            maximum_difference = max(maximum_difference, abs(float(value) - float(expected)))
    return maximum_difference <= 1e-12, maximum_difference


def _audit_decision(
    assessment: pd.DataFrame,
    relations: pd.DataFrame,
    bootstrap: pd.DataFrame,
    route_null: pd.DataFrame,
    coverage_evidence: Mapping[str, Any],
    decision: Mapping[str, Any],
    adverse_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every binding gate input from predictions and verified resampling artifacts."""

    increment = _audit_increment_metrics(assessment)
    monthly = [
        _audit_increment_metrics(group)
        for _month, group in assessment.groupby("year_month", sort=True)
    ]
    checkpoints = [
        _audit_increment_metrics(group)
        for _checkpoint, group in assessment.groupby("checkpoint_group", sort=True)
    ]

    def lower(metric: str) -> float:
        row = bootstrap.loc[
            bootstrap["record_type"].eq("interval")
            & bootstrap["metric"].eq(metric)
            & bootstrap["interval_level"].eq(0.80)
        ]
        if len(row) != 1:
            raise AssertionError(f"missing 80% bootstrap interval: {metric}")
        return float(row.iloc[0]["lower"])

    null_counts = {
        str(row.metric): int(row.real_exceeds_null_count)
        for row in route_null.loc[route_null["record_type"].eq("comparison")].itertuples(
            index=False
        )
    }
    o1_gates: dict[str, Any] = {
        **increment,
        "bootstrap_80_log_loss_lower": lower("log_loss_improvement"),
        "bootstrap_80_brier_lower": lower("brier_improvement"),
        "bootstrap_80_average_precision_lower": lower("average_precision_improvement"),
        "positive_months": sum(value["log_loss_improvement"] > 0.0 for value in monthly),
        "materially_adverse_checkpoint_groups": sum(
            value["log_loss_improvement"]
            < float(adverse_thresholds["O1_log_loss_improvement_below"])
            or value["brier_improvement"] < float(adverse_thresholds["O1_brier_improvement_below"])
            for value in checkpoints
        ),
        "real_exceeds_matching_nulls": max(
            int(null_counts.get("log_loss_improvement", 0)),
            int(null_counts.get("brier_improvement", 0)),
        ),
        "coverage_and_concentration_passed": bool(coverage_evidence["passed"]),
    }
    broad = assessment.loc[assessment["route_resolution_state"].eq("BROAD_CONFLICT")]
    low = assessment.loc[assessment["route_resolution_state"].eq("LOW_ROUTE_SUPPORT")]
    effects = _matched_effects(assessment, relations)
    matched_support = int(relations["treated_row_id"].nunique()) == len(broad)
    month_differences = [
        _audit_weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("BROAD_CONFLICT"),
                "iv_absolute_residual_15m",
            ],
            group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT"), "row_weight"],
        )
        - _audit_weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"),
                "iv_absolute_residual_15m",
            ],
            group.loc[group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"), "row_weight"],
        )
        for _month, group in assessment.groupby("year_month", sort=True)
    ]
    checkpoint_differences = [
        _audit_weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("BROAD_CONFLICT"),
                "iv_absolute_residual_15m",
            ],
            group.loc[group["route_resolution_state"].eq("BROAD_CONFLICT"), "row_weight"],
        )
        - _audit_weighted_mean(
            group.loc[
                group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"),
                "iv_absolute_residual_15m",
            ],
            group.loc[group["route_resolution_state"].eq("LOW_ROUTE_SUPPORT"), "row_weight"],
        )
        for _checkpoint, group in assessment.groupby("checkpoint_group", sort=True)
    ]
    broad_gates: dict[str, Any] = {
        "mean_residual": _audit_weighted_mean(
            broad["iv_absolute_residual_15m"], broad["row_weight"]
        ),
        "minus_low_route_support_residual": _audit_weighted_mean(
            broad["iv_absolute_residual_15m"], broad["row_weight"]
        )
        - _audit_weighted_mean(low["iv_absolute_residual_15m"], low["row_weight"]),
        "minus_matched_residual": _audit_finite_or_zero(effects["iv_absolute_residual_15m"]),
        "minus_matched_exceed_rate": _audit_finite_or_zero(
            effects["movement_exceeds_iv_expected_absolute"]
        ),
        "bootstrap_80_minus_low_residual_lower": lower("broad_minus_low_iv_absolute_residual"),
        "bootstrap_80_minus_matched_residual_lower": _audit_finite_or_zero(
            lower("broad_minus_matched_iv_absolute_residual")
        ),
        "bootstrap_80_minus_matched_exceed_lower": _audit_finite_or_zero(
            lower("broad_minus_matched_exceed_iv_rate")
        ),
        "positive_months": sum(value > 0.0 for value in month_differences),
        "materially_adverse_checkpoint_groups": sum(
            value < float(adverse_thresholds["broad_minus_low_iv_residual_below"])
            for value in checkpoint_differences
        ),
        "support_and_concentration_passed": bool(coverage_evidence["passed"]) and matched_support,
    }
    o1_matches, o1_difference = _gate_payload_matches(
        o1_gates, cast(dict[str, Any], decision["O1_gate"])
    )
    broad_matches, broad_difference = _gate_payload_matches(
        broad_gates, cast(dict[str, Any], decision["BROAD_CONFLICT_gate"])
    )
    o1_passed = o1_model_gate_passes(o1_gates)
    broad_passed = broad_conflict_iv_gate_passes(broad_gates)
    expected_decision = choose_options_movement_decision(
        blocker=None,
        o1_passed=o1_passed,
        broad_conflict_passed=broad_passed,
        descriptive_only=False,
    )
    status_passed = bool(
        decision["iv_excess_model_status"] == ("supported" if o1_passed else "not_supported")
        and decision["broad_conflict_movement_status"]
        == ("supported" if broad_passed else "not_supported")
        and decision["matched_control_status"]
        == ("supported" if matched_support else "insufficient_support")
    )
    return {
        "passed": o1_matches
        and broad_matches
        and expected_decision == decision["decision"]
        and status_passed,
        "O1_gate_matches": o1_matches,
        "BROAD_CONFLICT_gate_matches": broad_matches,
        "maximum_gate_difference": max(o1_difference, broad_difference),
        "expected_decision": expected_decision,
        "decision_matches": expected_decision == decision["decision"],
        "status_matches": status_passed,
        "matched_treated_rows": int(relations["treated_row_id"].nunique()),
        "eligible_broad_rows": len(broad),
    }


def _audit_completed_run(output: Path, *, clean: pd.DataFrame) -> dict[str, Any]:
    download = _audit_download_integrity(output)
    canonical = _load_plan_canonical(output)
    canonical_audit = _audit_canonical_from_raw(output, canonical)
    pairs = pd.read_parquet(output / "selected_option_pairs.parquet")
    movement = pd.read_parquet(output / "options_movement_panel.parquet")
    assessment = pd.read_parquet(output / "assessment_predictions.parquet")
    price_audit = pd.read_csv(output / "option_underlying_price_audit.csv")
    price_index = price_audit.set_index(["symbol", "signal_date"])
    chains = {
        (str(symbol), cast(date, trade_date)): group
        for (symbol, trade_date), group in canonical.groupby(
            ["underlying_symbol", "trade_date"], sort=False
        )
    }
    pair_mismatches = 0
    maximum_feature_difference = 0.0
    for pair in (
        pairs.sort_values(["symbol", "signal_date"], kind="mergesort")
        .head(100)
        .itertuples(index=False)
    ):
        source = price_index.loc[(str(pair.symbol), str(pair.signal_date))]
        options_date = date.fromisoformat(str(pair.required_options_date))
        signal_date = date.fromisoformat(str(pair.signal_date))
        pair_mismatches += int(not options_date < signal_date)
        selected = select_primary_atm_pair(
            chains[(str(pair.symbol), options_date)],
            previous_close=float(source["previous_close_underlying_price"]),
        )
        pair_mismatches += int(
            not selected.available
            or selected.call_contract_id != str(pair.call_contract_id)
            or selected.put_contract_id != str(pair.put_contract_id)
        )
        if selected.available:
            features = calculate_primary_option_features(
                selected, previous_close=float(source["previous_close_underlying_price"])
            )
            features.update(
                calculate_optional_option_features(
                    chains[(str(pair.symbol), options_date)],
                    front_selection=selected,
                    previous_close=float(source["previous_close_underlying_price"]),
                )
            )
            features.update(iv_movement_approximations(float(features["atm_iv"])))
            for column, value in features.items():
                if column not in pairs.columns or not isinstance(value, (int, float)):
                    continue
                observed = float(getattr(pair, column))
                expected_value = float(value)
                if math.isnan(observed) and math.isnan(expected_value):
                    continue
                if math.isnan(observed) != math.isnan(expected_value):
                    maximum_feature_difference = math.inf
                    continue
                maximum_feature_difference = max(
                    maximum_feature_difference,
                    abs(expected_value - observed),
                )
    pair_signal_dates = pd.to_datetime(pairs["signal_date"], errors="raise").dt.date
    pair_options_dates = pd.to_datetime(pairs["required_options_date"], errors="raise").dt.date
    expected_previous = pair_signal_dates.map(_previous_session)
    audited_required = pairs.merge(
        price_audit[["symbol", "signal_date", "required_options_date"]],
        on=["symbol", "signal_date"],
        how="left",
        suffixes=("", "_audit"),
        validate="one_to_one",
    )
    chronology_passed = bool(
        pair_options_dates.eq(expected_previous).all()
        and audited_required["required_options_date"]
        .astype(str)
        .eq(audited_required["required_options_date_audit"].astype(str))
        .all()
        and not price_audit["provider_moneyness_check"].eq("inconsistent").any()
        and not pairs.merge(
            price_audit.loc[
                price_audit["split_boundary_ambiguous"].astype(bool), ["symbol", "signal_date"]
            ],
            on=["symbol", "signal_date"],
            how="inner",
        ).shape[0]
    )
    coefficients = _read_json(output / "model_coefficients.json")
    sample = assessment.sort_values("row_id", kind="mergesort").head(100)
    probability_differences: list[float] = []
    for model in ("O0", "O1"):
        manual = _manual_probabilities(sample, coefficients["primary_models"][model])
        probability_differences.extend(
            np.abs(manual - sample[f"{model}_probability"].to_numpy(float)).tolist()
        )
    maximum_probability_difference = max(probability_differences, default=math.inf)
    pooled = pd.read_csv(output / "pooled_metrics.csv")

    def stored(model: str, metric: str) -> float:
        row = pooled.loc[pooled["model"].eq(model) & pooled["metric"].eq(metric)]
        return float(row.iloc[0]["value"])

    metric_differences: list[float] = []
    target = assessment["movement_exceeds_iv_expected_absolute"].to_numpy(int)
    weights = assessment["row_weight"].to_numpy(float)
    for model in ("O0", "O1"):
        probability = assessment[f"{model}_probability"].to_numpy(float)
        recalculated = {
            "log_loss": log_loss(target, probability, sample_weight=weights, labels=[0, 1]),
            "brier_score": brier_score_loss(target, probability, sample_weight=weights),
            "auc": roc_auc_score(target, probability, sample_weight=weights),
            "average_precision": average_precision_score(
                target, probability, sample_weight=weights
            ),
        }
        metric_differences.extend(
            abs(float(value) - stored(model, metric)) for metric, value in recalculated.items()
        )
    maximum_metric_difference = max(metric_differences, default=math.inf)
    movement_audit = _audit_movement_sample(movement)
    relations = build_matched_control_relations(assessment)
    matched_relation_quality = bool(
        relations.empty
        or (
            relations.groupby("treated_row_id").size().ge(5).all()
            and np.allclose(relations.groupby("treated_row_id")["match_weight"].sum(), 1.0)
        )
    )
    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    nulls = pd.read_csv(output / "route_null_metrics.csv")
    bootstrap_audit = _audit_bootstrap(assessment, relations, bootstrap)
    route_null_audit = _audit_route_null(movement, assessment, coefficients, nulls)
    resampling_passed = bool(bootstrap_audit["passed"] and route_null_audit["passed"])
    determinism = _read_json(output / "determinism_check.json")
    preprocessing_audit = _audit_model_preprocessing(movement, coefficients, determinism)
    decision = _read_json(output / "decision.json")
    mapping = pd.read_csv(output / "underlying_symbol_mapping.csv")
    coverage_evidence = _recompute_coverage_evidence(
        clean,
        pairs,
        mapping,
        download_integrity_passed=bool(download["passed"]),
    )
    stored_coverage = cast(dict[str, Any], decision["coverage_evidence"])
    coverage_matches = set(coverage_evidence) == set(stored_coverage) and all(
        (
            bool(value) == bool(stored_coverage[key])
            if isinstance(value, bool)
            else math_isclose(float(value), float(stored_coverage[key]))
        )
        for key, value in coverage_evidence.items()
    )
    decision_audit = _audit_decision(
        assessment,
        relations,
        bootstrap,
        nulls,
        coverage_evidence,
        decision,
        cast(
            dict[str, Any],
            _read_json(output / "model_configurations.json")[
                "materially_adverse_checkpoint_thresholds"
            ],
        ),
    )
    decision_passed = bool(coverage_matches and decision_audit["passed"])
    required_nonempty = all(
        len(pd.read_csv(output / name)) > 0
        for name in (
            "route_state_movement_metrics.csv",
            "matched_control_metrics.csv",
            "monthly_metrics.csv",
            "checkpoint_metrics.csv",
            "subgroup_metrics.csv",
            "continuous_residual_metrics.csv",
        )
    )
    passed = bool(
        download["passed"]
        and canonical_audit["passed"]
        and pair_mismatches == 0
        and maximum_feature_difference <= 1e-12
        and chronology_passed
        and maximum_probability_difference <= 1e-12
        and maximum_metric_difference <= 1e-12
        and movement_audit["passed"]
        and matched_relation_quality
        and resampling_passed
        and preprocessing_audit["passed"]
        and determinism.get("passed") is True
        and decision_passed
        and required_nonempty
    )
    return {
        "passed": passed,
        "download_integrity": download,
        "canonical_raw_reconstruction": canonical_audit,
        "canonical_records": len(canonical),
        "sampled_pair_reconstructions": min(100, len(pairs)),
        "pair_mismatches": pair_mismatches,
        "maximum_option_feature_difference": maximum_feature_difference,
        "chronology_corporate_action_and_moneyness": chronology_passed,
        "movement_rows": len(movement),
        "movement_outcome_and_timestamp_sample": movement_audit,
        "manual_probability_rows": len(sample),
        "maximum_manual_probability_difference": maximum_probability_difference,
        "maximum_metric_difference": maximum_metric_difference,
        "matched_control_relation_quality": matched_relation_quality,
        "bootstrap": bootstrap_audit,
        "route_bundle_null": route_null_audit,
        "development_preprocessing_and_coefficients": preprocessing_audit,
        "determinism": determinism.get("passed") is True,
        "coverage_evidence_matches": coverage_matches,
        "decision_logic": decision_audit,
    }


def run_audit(output: Path) -> dict[str, Any]:
    """Audit sources, safety, chronology, schema, blocker, and decision fail-closed."""

    missing = [name for name in REQUIRED_ARTIFACTS if not (output / name).is_file()]
    if missing:
        raise AssertionError(f"required artifacts missing: {missing}")
    contract = _read_json(output / "contract.json")
    decision = _read_json(output / "decision.json")
    flags_passed = all(contract.get(key) == value for key, value in SAFETY_FLAGS.items()) and all(
        decision.get(key) == value for key, value in SAFETY_FLAGS.items()
    )
    schema = _read_json(output / "eodhd_options_schema_mapping.json")
    expected_schema = schema_mapping()
    schema_passed = bool(
        schema["verified"]
        and schema["official_openapi_commit"] == OPENAPI_SHA
        and schema["endpoints"] == expected_schema["endpoints"]
        and schema["historical_eod"] == expected_schema["historical_eod"]
    )
    clean, required = _required_dates()
    structural = _audit_structural_reconstruction(output, clean)
    plan = _audit_plan(output, required)
    underlying_prices = _audit_unadjusted_prices(output)
    boundary = _read_json(output / "protected_boundary_audit.json")
    boundary_passed = bool(
        max(pd.to_datetime(clean["session"]).dt.date) < PROTECTED_START
        and max(required["required_options_date"]) < PROTECTED_START
        and boundary["protected_rows_materialised"] == 0
        and boundary["same_day_options_joins"] == 0
        and boundary["future_options_joins"] == 0
    )
    credential = _credential_scan(output)
    ignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    cache_ignored = "data/vendor/eodhd/options/" in ignore_text
    if decision["decision"] == "blocked_missing_eodhd_api_token":
        run_specific = _audit_blocked_run(output, clean=clean, required=required)
    elif decision["decision"] == "blocked_historical_options_date_unavailable":
        run_specific = _audit_historical_date_blocked_run(output, clean=clean, required=required)
    elif decision["decision"] == "blocked_insufficient_options_chain_coverage":
        run_specific = _audit_coverage_blocked_run(output, clean=clean)
    else:
        run_specific = _audit_completed_run(output, clean=clean)
    passed = bool(
        not missing
        and flags_passed
        and schema_passed
        and structural["passed"]
        and plan["passed"]
        and underlying_prices["passed"]
        and boundary_passed
        and credential["passed"]
        and cache_ignored
        and run_specific["passed"]
    )
    result = {
        "passed": passed,
        "audit_scope": "independent_fail_closed_options_screen_audit",
        "safety_flags": flags_passed,
        "credential_redaction": credential,
        "official_endpoint_schema_mapping": schema_passed,
        "request_plan": plan,
        "exact_required_prior_close_dates": True,
        "chronology_and_protected_boundary": boundary_passed,
        "unadjusted_underlying_prices": underlying_prices,
        "cache_path_ignored_by_git": cache_ignored,
        "structural_panel_reconstruction": structural,
        "run_specific": run_specific,
        "decision_logic": passed,
    }
    write_json(output / "lightweight_audit.json", result)
    if not passed:
        raise AssertionError("blocked V0 audit failed closed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY)
    return parser.parse_args()


def main() -> int:
    result = run_audit(parse_args().output.expanduser().resolve())
    print("supported" if result["passed"] else "blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
