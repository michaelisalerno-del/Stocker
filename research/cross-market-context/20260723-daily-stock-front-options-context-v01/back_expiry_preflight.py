#!/usr/bin/env python3
"""Make the single bounded non-compact EODHD back-expiry schema preflight."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

import requests

EXPERIMENT_DIR = Path(__file__).resolve().parent
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
RESULT_PATH = PRIMARY / "back_expiry_schema_preflight.json"
PLAN_PATH = PRIMARY / "back_expiry_future_request_plan.json"
BASE_URL = "https://eodhd.com/api"
ENDPOINT = "/mp/unicornbay/options/eod"
SYMBOL = "AAL"
OBSERVATION_DATE = date(2025, 8, 21)
UNDERLYING_CLOSE = 12.564999
MAXIMUM_RECORDS = 100
PROTECTED_BOUNDARY = date(2025, 8, 23)

SAFETY_FLAGS: dict[str, object] = {
    "research_only": True,
    "quick_context_screen": True,
    "branches_run_independently": True,
    "daily_stock_context_test": True,
    "front_options_only_context_test": True,
    "back_expiry_bulk_download_enabled": False,
    "back_expiry_schema_preflight_only": True,
    "previous_close_options_only": True,
    "intraday_option_quotes_used": False,
    "option_pnl_calculated": False,
    "underlying_movement_outcomes_opened": True,
    "directional_outcomes_primary": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "prospective_validation": False,
}


class ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def json(self) -> object: ...


Get = Callable[..., ResponseLike]


def request_parameters() -> dict[str, object]:
    """Return the credential-free, exact-date, non-compact request surface."""

    return {
        "filter[underlying_symbol]": SYMBOL,
        "filter[tradetime_from]": OBSERVATION_DATE.isoformat(),
        "filter[tradetime_to]": OBSERVATION_DATE.isoformat(),
        "filter[strike_from]": round(UNDERLYING_CLOSE * 0.80, 4),
        "filter[strike_to]": round(UNDERLYING_CLOSE * 1.20, 4),
        "filter[exp_date_from]": (OBSERVATION_DATE + timedelta(days=46)).isoformat(),
        "filter[exp_date_to]": (OBSERVATION_DATE + timedelta(days=90)).isoformat(),
        "page[offset]": 0,
        "page[limit]": MAXIMUM_RECORDS,
        "compact": 0,
        "fmt": "json",
    }


def _date_from_record(record: Mapping[str, Any]) -> date | None:
    identity = record.get("id")
    if isinstance(identity, str) and len(identity) >= 10:
        try:
            return date.fromisoformat(identity[-10:])
        except ValueError:
            pass
    attributes = record.get("attributes")
    if isinstance(attributes, Mapping):
        for field in ("trade_date", "date", "tradetime"):
            value = attributes.get(field)
            if isinstance(value, str):
                try:
                    return date.fromisoformat(value[:10])
                except ValueError:
                    continue
    return None


def _expiration_from_record(record: Mapping[str, Any]) -> date | None:
    attributes = record.get("attributes")
    if not isinstance(attributes, Mapping):
        return None
    for field in ("exp_date", "expiration_date", "expiration"):
        value = attributes.get(field)
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                continue
    return None


def _response_fields(records: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    top = sorted({str(key) for record in records for key in record})
    attributes = sorted(
        {
            str(key)
            for record in records
            for value in (record.get("attributes"),)
            if isinstance(value, Mapping)
            for key in value
        }
    )
    return {"record": top, "attributes": attributes}


def perform_preflight(
    *,
    token: str,
    get: Get = requests.get,
) -> dict[str, Any]:
    """Execute one request without persisting provider observation records."""

    if not token:
        return {
            **SAFETY_FLAGS,
            "status": "blocked_missing_eodhd_api_token",
            "request_attempted": False,
            "request_count": 0,
            "record_count": 0,
            "credential_recorded": False,
        }
    parameters = request_parameters()
    authenticated = {**parameters, "api_token": token}
    try:
        response = get(
            f"{BASE_URL}{ENDPOINT}",
            params=authenticated,
            timeout=30.0,
        )
    except Exception as error:
        return {
            **SAFETY_FLAGS,
            "status": "blocked_schema_or_endpoint_failure",
            "request_attempted": True,
            "request_count": 1,
            "endpoint": ENDPOINT,
            "parameters": parameters,
            "error_type": type(error).__name__,
            "credential_recorded": False,
        }
    base: dict[str, Any] = {
        **SAFETY_FLAGS,
        "request_attempted": True,
        "request_count": 1,
        "endpoint": ENDPOINT,
        "base_url": BASE_URL,
        "parameters": parameters,
        "symbol": SYMBOL,
        "requested_observation_date": OBSERVATION_DATE.isoformat(),
        "underlying_close": UNDERLYING_CLOSE,
        "maximum_records": MAXIMUM_RECORDS,
        "http_status": int(response.status_code),
        "content_type": response.headers.get("content-type"),
        "raw_response_bytes": len(response.content),
        "raw_response_sha256": hashlib.sha256(response.content).hexdigest(),
        "raw_response_cache_path": None,
        "raw_response_persisted": False,
        "raw_response_canonicalised": False,
        "canonical_cache_modified": False,
        "protected_records_persisted": 0,
        "credential_recorded": False,
    }
    if response.status_code != 200:
        return {
            **base,
            "status": "blocked_schema_or_endpoint_failure",
            "record_count": 0,
        }
    try:
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("response is not a JSON object")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("response data is not an array")
        if len(data) > MAXIMUM_RECORDS:
            raise ValueError("response exceeds the preflight record limit")
        records = [
            cast(Mapping[str, Any], record) for record in data if isinstance(record, Mapping)
        ]
        if len(records) != len(data):
            raise ValueError("response data contains a non-object record")
        dates = [_date_from_record(record) for record in records]
        returned_dates = sorted({value.isoformat() for value in dates if value is not None})
        protected_records_returned = sum(
            value is not None and value >= PROTECTED_BOUNDARY for value in dates
        )
        every_record_identifiable = all(value is not None for value in dates)
        exact_date_records = [
            record
            for record, record_date in zip(records, dates, strict=True)
            if record_date == OBSERVATION_DATE
        ]
        back_records = []
        for record in exact_date_records:
            expiration = _expiration_from_record(record)
            if expiration is None:
                continue
            dte = (expiration - OBSERVATION_DATE).days
            if 46 <= dte <= 90:
                back_records.append(record)
        pagination = payload.get("meta", {})
        status = "supported_noncompact_schema"
        if records and not every_record_identifiable:
            status = "blocked_authoritative_date_missing"
        elif not back_records:
            status = "blocked_back_expiry_records_absent"
        return {
            **base,
            "status": status,
            "record_count": len(records),
            "response_fields": _response_fields(records),
            "returned_dates": returned_dates,
            "protected_records_returned": protected_records_returned,
            "every_record_has_authoritative_observation_identity": (every_record_identifiable),
            "exact_date_filtering_possible": bool(every_record_identifiable and exact_date_records),
            "exact_requested_date_records": len(exact_date_records),
            "back_expiry_dte_records": len(back_records),
            "pagination_metadata": pagination,
            "noncompact_response": True,
        }
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            **base,
            "status": "blocked_schema_or_endpoint_failure",
            "record_count": 0,
            "error_type": type(error).__name__,
        }


def future_request_plan(result: Mapping[str, Any]) -> dict[str, Any]:
    supported = result.get("status") == "supported_noncompact_schema"
    return {
        **SAFETY_FLAGS,
        "preflight_status": result.get("status"),
        "future_acquisition_supported": supported,
        "current_experiment_bulk_download": False,
        "current_experiment_additional_requests": 0,
        "proposed_endpoint": ENDPOINT,
        "proposed_response_mode": "noncompact",
        "proposed_exact_date_filters": [
            "filter[tradetime_from]",
            "filter[tradetime_to]",
        ],
        "proposed_sequence": [
            "construct one exact previous-session stock-date request",
            "bound expiration to 46–90 calendar DTE",
            "bound strikes to an ATM neighbourhood",
            "retain noncompact resource identity",
            "parse in memory and reject non-exact or protected observation dates",
            "persist no unfiltered provider observation records",
            "validate the protected boundary before any filtered append",
            "write an idempotent credential-free receipt",
        ],
        "option_strategy_or_dte_recommendation": False,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if RESULT_PATH.is_file():
        existing = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if isinstance(existing, Mapping) and existing.get("request_attempted") is True:
            print(str(existing.get("status", "blocked_schema_or_endpoint_failure")))
            return
    result = perform_preflight(token=os.environ.get("EODHD_API_TOKEN", ""))
    _write_json(RESULT_PATH, result)
    _write_json(PLAN_PATH, future_request_plan(result))
    print(str(result["status"]))


if __name__ == "__main__":
    main()
