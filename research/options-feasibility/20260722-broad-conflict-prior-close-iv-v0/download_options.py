#!/usr/bin/env python3
"""Sequential, resumable EODHD options downloader for the bounded V0 request plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import requests

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
for package in ("stocker_research", "stocker_data"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from run_screen_v0 import (  # noqa: E402
    FROZEN_COHORT,
    MAX_OPTIONS_BYTES,
    MAX_RAW_RECORDS,
    OPENAPI_VERSION,
    PRIMARY,
    default_provider_root,
    options_data_dir,
    prepare_blocked,
    schema_mapping,
    write_csv,
    write_json,
    write_parquet,
)

from stocker_research.broad_conflict_options_iv_screen_v0 import SAFETY_FLAGS  # noqa: E402
from stocker_research.eodhd_options_downloader_v0 import (  # noqa: E402
    CANONICAL_OPTION_COLUMNS,
    DownloadConfig,
    EODHDOptionsDownloader,
    OptionsDownloadError,
    OptionsRequest,
    OptionsResourceLimitExceeded,
    OptionsSchemaError,
    canonicalize_response_records,
    deterministic_symbol_mapping,
    redact_secrets,
    resolve_canonical_duplicates,
)

UNDERLYING_SYMBOLS_ENDPOINT = "/mp/unicornbay/options/underlying-symbols"
OPTIONS_EOD_ENDPOINT = "/mp/unicornbay/options/eod"


def _write_blocker(output: Path, blocker: str) -> dict[str, Any]:
    preflight_path = output / "eodhd_options_api_preflight.json"
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    )
    setup_requests_completed = int(preflight.get("setup_requests_completed", 0))
    decision = {
        **SAFETY_FLAGS,
        "status": blocker,
        "decision": blocker,
        "options_download_status": "blocked",
        "options_coverage_status": "blocked",
        "iv_excess_model_status": "blocked",
        "broad_conflict_movement_status": "blocked",
        "matched_control_status": "blocked",
        "provider_setup_requests_completed": setup_requests_completed,
    }
    write_json(output / "decision.json", decision)
    existing_manifest = output / "options_download_manifest.json"
    manifest = (
        json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing_manifest.is_file()
        else {}
    )
    manifest["status"] = blocker
    manifest["setup_requests_completed"] = setup_requests_completed
    manifest.setdefault("pagination_complete", False)
    manifest.setdefault("unexplained_truncations", 0)
    manifest.setdefault("credential_exposures", 0)
    manifest.setdefault("manifest_rows", [])
    write_json(output / "options_download_manifest.json", manifest)
    write_json(
        output / "lightweight_audit.json",
        {
            "passed": False,
            "status": blocker,
            "audit_scope": "download_blocked_before_full_independent_audit",
            "provider_requests_completed": int(manifest.get("requests_completed", 0)),
            "provider_setup_requests_completed": setup_requests_completed,
        },
    )
    write_json(
        output / "determinism_check.json",
        {
            "passed": False,
            "status": "blocked_before_options_data",
            "reason": blocker,
            "request_plan_rebuild_match": True,
            "structural_reconstruction_repeatable": True,
            "selected_contract_mismatches": None,
            "joined_row_mismatches": None,
            "maximum_option_feature_difference": None,
            "maximum_probability_difference": None,
            "maximum_movement_difference": None,
            "bootstrap_repeated": False,
            "route_null_refits_repeated": False,
        },
    )
    if int(manifest.get("requests_completed", 0)) == 0:
        for name in (
            "options_data_quality.csv",
            "options_coverage.csv",
            "options_structural_join_audit.csv",
        ):
            path = output / name
            if path.is_file():
                frame = pd.read_csv(path)
                if "status" in frame:
                    frame["status"] = blocker
                    write_csv(path, frame)
        for name in ("option_pair_selection_manifest.json", "model_coefficients.json"):
            path = output / name
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["status"] = blocker
                if "reason" in payload:
                    payload["reason"] = blocker
                write_json(path, payload)
    observation_text = ""
    if blocker == "blocked_historical_options_date_unavailable":
        resource_dates = cast(list[str], preflight.get("resource_observation_dates", []))
        resource_range = (
            f"{resource_dates[0]} through {resource_dates[-1]}" if resource_dates else "unavailable"
        )
        observation_text = (
            f" The corrected preflight requested {preflight.get('requested_session_date')} "
            f"but the returned historical EOD resources and quote timestamps covered "
            f"{resource_range}; `tradetime` reflected last-trade activity rather than the "
            "EOD observation date. The official schema exposes no observation-date filter."
        )
    report_text = (
        "# Prior-Close Options IV Movement Screen V0\n\n"
        f"Primary decision: `{blocker}`\n\n"
        f"Provider bulk page requests recorded: {int(manifest.get('requests_completed', 0))}. "
        f"Provider setup requests recorded: {setup_requests_completed}."
        f"{observation_text} The fail-closed download did not produce an options-movement "
        "inference. No intraday option fill or option P&L was calculated.\n"
    )
    (output / "report.md").write_text(report_text, encoding="utf-8")
    if output.resolve() == PRIMARY.resolve():
        reports_dir = EXPERIMENT_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "blocked_pre_download_report.md").write_text(report_text, encoding="utf-8")
    return decision


class RequestsTransport:
    """Small synchronous requests boundary used by the reusable downloader."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def get(self, url: str, *, params: dict[str, object], timeout: float) -> requests.Response:
        return self.session.get(url, params=params, timeout=timeout, stream=True)


class SetupRequester:
    """Credential-safe, paced retry boundary for coverage and preflight calls."""

    def __init__(
        self,
        transport: RequestsTransport,
        *,
        token: str,
        requests_per_minute: int,
        max_attempts: int = 4,
    ) -> None:
        self.transport = transport
        self.token = token
        self.minimum_interval = 60.0 / float(requests_per_minute)
        self.max_attempts = max_attempts
        self.last_request_monotonic: float | None = None

    def wait_before_next_request(self) -> None:
        if self.last_request_monotonic is None:
            return
        remaining = self.minimum_interval - (time.monotonic() - self.last_request_monotonic)
        if remaining > 0.0:
            time.sleep(remaining)

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return float(2 ** (attempt - 1))

    def get_json(self, endpoint: str, *, params: dict[str, object], timeout: float) -> object:
        authenticated = {**params, "api_token": self.token}
        url = "https://eodhd.com/api" + endpoint
        for attempt in range(1, self.max_attempts + 1):
            self.wait_before_next_request()
            self.last_request_monotonic = time.monotonic()
            try:
                response = self.transport.get(url, params=authenticated, timeout=timeout)
            except Exception as exc:
                if attempt >= self.max_attempts:
                    message = redact_secrets(str(exc), secrets=(self.token,))
                    raise RuntimeError(f"EODHD setup transport failure: {message}") from None
                time.sleep(float(2 ** (attempt - 1)))
                continue
            if response.status_code == 200:
                try:
                    payload = response.json()
                except (
                    requests.ConnectionError,
                    requests.Timeout,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ContentDecodingError,
                ) as exc:
                    response.close()
                    if attempt >= self.max_attempts:
                        message = redact_secrets(str(exc), secrets=(self.token,))
                        raise RuntimeError(
                            f"EODHD setup response-body failure: {message}"
                        ) from None
                    time.sleep(float(2 ** (attempt - 1)))
                    continue
                except ValueError as exc:
                    response.close()
                    raise OptionsSchemaError("EODHD setup response is not valid JSON") from exc
                response.close()
                return payload
            if response.status_code in {401, 403}:
                response.close()
                raise RuntimeError("EODHD options authentication or entitlement failed")
            if response.status_code not in {429, 500, 502, 503, 504}:
                status = response.status_code
                response.close()
                raise RuntimeError(f"permanent EODHD setup response status {status}")
            delay = self._retry_delay(response, attempt)
            response.close()
            if attempt < self.max_attempts:
                time.sleep(delay)
        raise RuntimeError("transient EODHD setup response exhausted retries")


def _coverage_symbols(requester: SetupRequester, *, timeout: float) -> set[str]:
    payload = requester.get_json(
        UNDERLYING_SYMBOLS_ENDPOINT,
        params={"fmt": "json"},
        timeout=timeout,
    )
    data: object = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise OptionsSchemaError("underlying-symbol coverage response is not a list")
    symbols: set[str] = set()
    for item in data:
        if isinstance(item, str):
            symbols.add(item.upper())
        elif isinstance(item, dict):
            candidate = item.get("symbol", item.get("id"))
            if isinstance(candidate, str):
                symbols.add(candidate.upper())
    if not symbols:
        raise OptionsSchemaError("underlying-symbol coverage response contains no symbols")
    return symbols


def _provider_session_date(value: object) -> str | None:
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


def _resource_observation_date(record: dict[str, Any]) -> str | None:
    resource_id = record.get("id")
    if not isinstance(resource_id, str) or len(resource_id) < 10:
        return None
    candidate = resource_id[-10:]
    try:
        return date.fromisoformat(candidate).isoformat()
    except ValueError:
        return None


def _preflight(
    requester: SetupRequester,
    *,
    symbol: str,
    trade_date: str,
    output: Path,
) -> None:
    params: dict[str, object] = {
        "filter[underlying_symbol]": symbol,
        "filter[tradetime_from]": trade_date,
        "filter[tradetime_to]": trade_date,
        "page[offset]": 0,
        "page[limit]": 10,
        "compact": 0,
        "fmt": "json",
    }
    payload = requester.get_json(
        OPTIONS_EOD_ENDPOINT,
        params=params,
        timeout=30.0,
    )
    if not isinstance(payload, dict):
        raise OptionsSchemaError("preflight response must be an object")
    meta = payload.get("meta")
    data = payload.get("data")
    links = payload.get("links")
    if not isinstance(meta, dict) or not isinstance(data, list) or not isinstance(links, dict):
        raise OptionsSchemaError("preflight response lacks meta/data/links")
    if len(data) > 10:
        raise OptionsSchemaError("preflight exceeded its ten-record cap")
    provider_fields = {
        "contract",
        "underlying_symbol",
        "type",
        "exp_date",
        "strike",
        "tradetime",
        "bid",
        "ask",
        "volatility",
        "open_interest",
        "dte",
    }
    for item in data:
        if not isinstance(item, dict):
            raise OptionsSchemaError("preflight record is not an object")
        attributes = item.get("attributes", item)
        if not isinstance(attributes, dict) or not provider_fields.issubset(attributes):
            raise OptionsSchemaError("preflight response omits required provider fields")
    object_rows = [cast(dict[str, Any], item) for item in data]
    attribute_rows = [cast(dict[str, Any], item.get("attributes", item)) for item in object_rows]
    resource_observation_dates = sorted(
        {value for item in object_rows if (value := _resource_observation_date(item)) is not None}
    )
    tradetime_dates = sorted(
        {
            value
            for item in attribute_rows
            if (value := _provider_session_date(item.get("tradetime"))) is not None
        }
    )
    bid_observation_dates = sorted(
        {
            value
            for item in attribute_rows
            if (value := _provider_session_date(item.get("bid_date"))) is not None
        }
    )
    ask_observation_dates = sorted(
        {
            value
            for item in attribute_rows
            if (value := _provider_session_date(item.get("ask_date"))) is not None
        }
    )
    exact_observation_date_available = bool(resource_observation_dates) and all(
        value == trade_date for value in resource_observation_dates
    )
    if not exact_observation_date_available:
        write_json(
            output / "eodhd_options_api_preflight.json",
            {
                "status": "blocked_historical_options_date_unavailable",
                "endpoint": OPTIONS_EOD_ENDPOINT,
                "symbol": symbol,
                "requested_session_date": trade_date,
                "record_limit": 10,
                "records_received": len(data),
                "setup_requests_completed": 2,
                "pagination": {
                    "offset": meta.get("offset"),
                    "limit": meta.get("limit"),
                    "total": meta.get("total"),
                    "next_present": bool(links.get("next")),
                },
                "resource_observation_dates": resource_observation_dates,
                "tradetime_dates": tradetime_dates,
                "bid_observation_dates": bid_observation_dates,
                "ask_observation_dates": ask_observation_dates,
                "tradetime_is_eod_observation_date": False,
                "official_observation_date_filter_available": False,
                "exact_requested_observation_date_confirmed": False,
                "america_new_york_mapping_confirmed": True,
                "authentication_redacted": True,
                "blocker_reason": (
                    "the documented tradetime filter selects last-trade activity, while "
                    "the returned EOD resource observation dates differ from the requested "
                    "prior-close session"
                ),
            },
        )
        raise RuntimeError("blocked_historical_options_date_unavailable")
    canonical = canonicalize_response_records(
        object_rows,
        request_id="preflight-redacted",
        provider_schema_version=f"openapi-{OPENAPI_VERSION}",
    )
    historical_dates = sorted({record["trade_date"].isoformat() for record in canonical.records})
    if not historical_dates or any(value != trade_date for value in historical_dates):
        raise RuntimeError("blocked_historical_options_date_unavailable")
    required_fields = {
        "contract_id",
        "option_type",
        "expiration_date",
        "strike",
        "trade_date",
        "bid",
        "ask",
        "implied_volatility",
        "open_interest",
        "dte",
    }
    if not canonical.records:
        raise OptionsSchemaError("preflight produced no valid canonical records")
    available_fields = set(canonical.records[0])
    missing = sorted(required_fields.difference(available_fields))
    if missing:
        raise OptionsSchemaError(f"preflight canonical fields unavailable: {missing}")
    write_json(
        output / "eodhd_options_api_preflight.json",
        {
            "status": "supported",
            "endpoint": OPTIONS_EOD_ENDPOINT,
            "symbol": symbol,
            "trade_date_from": trade_date,
            "trade_date_to": trade_date,
            "record_limit": 10,
            "records_received": len(data),
            "setup_requests_completed": 2,
            "canonical_records": len(canonical.records),
            "rejected_records": len(canonical.rejections),
            "pagination": {
                "offset": meta.get("offset"),
                "limit": meta.get("limit"),
                "total": meta.get("total"),
                "next_present": bool(links.get("next")),
            },
            "historical_session_dates": historical_dates,
            "historical_eod_confirmed": True,
            "america_new_york_mapping_confirmed": True,
            "authentication_redacted": True,
        },
    )


def _request_from_chunk(chunk: dict[str, Any]) -> OptionsRequest:
    return OptionsRequest(
        underlying_symbol=str(chunk["underlying_symbol"]),
        trade_date_from=date.fromisoformat(str(chunk["trade_date_from"])),
        trade_date_to=date.fromisoformat(str(chunk["trade_date_to"])),
        strike_from=float(chunk["strike_from"]),
        strike_to=float(chunk["strike_to"]),
        expiration_from=date.fromisoformat(str(chunk["expiration_from"])),
        expiration_to=date.fromisoformat(str(chunk["expiration_to"])),
        fields=tuple(str(value) for value in chunk["fields"]),
        compact=bool(chunk["compact"]),
    )


def execute_download(*, output: Path, data_dir: Path) -> dict[str, Any]:
    """Run coverage, preflight, and every bounded request sequentially."""

    token = os.environ.get("EODHD_API_TOKEN")
    if not token:
        prepare_blocked(provider_root=default_provider_root(), primary=output)
        return {"status": "blocked_missing_eodhd_api_token"}
    plan = json.loads((output / "options_request_plan.json").read_text(encoding="utf-8"))
    if not plan.get("resource_gate_passed"):
        return _write_blocker(output, "blocked_options_download_resource_limit")
    write_json(output / "eodhd_options_schema_mapping.json", schema_mapping())
    pace_value = os.environ.get("EODHD_OPTIONS_REQUESTS_PER_MINUTE", "20")
    try:
        pace = int(pace_value)
    except ValueError as exc:
        raise ValueError("EODHD_OPTIONS_REQUESTS_PER_MINUTE must be an integer") from exc
    if pace <= 0:
        raise ValueError("EODHD_OPTIONS_REQUESTS_PER_MINUTE must be positive")
    transport = RequestsTransport()
    setup_requester = SetupRequester(transport, token=token, requests_per_minute=pace)
    coverage = _coverage_symbols(setup_requester, timeout=30.0)
    mapping_result = deterministic_symbol_mapping(FROZEN_COHORT, provider_coverage=coverage)
    mapping_frame = pd.DataFrame(
        [
            {
                "stocker_symbol": row.stocker_symbol,
                "eodhd_underlying_symbol": row.eodhd_underlying_symbol or "",
                "coverage_available": row.coverage_available,
                "earliest_option_date": "pending_canonical_summary",
                "latest_option_date": "pending_canonical_summary",
                "records_returned": 0,
                "mapping_method": row.mapping_method,
                "status": (
                    "ambiguous"
                    if row.stocker_symbol in mapping_result.ambiguous_symbols
                    else "supported"
                    if row.coverage_available
                    else "not_supported"
                ),
            }
            for row in mapping_result.rows
        ]
    )
    write_csv(output / "underlying_symbol_mapping.csv", mapping_frame)
    if mapping_result.ambiguous_symbols:
        return _write_blocker(output, "blocked_underlying_symbol_mapping_failure")
    mapped = {
        row.stocker_symbol: row.eodhd_underlying_symbol
        for row in mapping_result.rows
        if row.coverage_available and row.eodhd_underlying_symbol is not None
    }
    if len(mapped) < 15:
        return _write_blocker(output, "blocked_underlying_symbol_mapping_failure")
    preflight_chunk = next(
        cast(dict[str, Any], value)
        for value in reversed(plan["chunks"])
        if str(cast(dict[str, Any], value)["underlying_symbol"]) in mapped
    )
    preflight_symbol = str(preflight_chunk["underlying_symbol"])
    _preflight(
        setup_requester,
        symbol=mapped[preflight_symbol],
        trade_date=str(preflight_chunk["required_trade_dates"][-1]),
        output=output,
    )
    setup_requester.wait_before_next_request()
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token=token,
            data_dir=data_dir,
            requests_per_minute=pace,
        ),
        transport=transport,
    )
    all_manifest: list[dict[str, object]] = []
    total_raw_records = 0
    total_canonical_records = 0
    total_rejections = 0
    total_duplicates = 0
    canonical_dir = data_dir / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    rejection_rows: list[dict[str, object]] = []
    for chunk_value in plan["chunks"]:
        chunk = cast(dict[str, Any], chunk_value)
        symbol = str(chunk["underlying_symbol"])
        if symbol not in mapped:
            continue
        provider_symbol = mapped[symbol]
        request = _request_from_chunk(chunk).replace(underlying_symbol=provider_symbol)
        try:
            result = downloader.download_with_splitting(request)
        except OptionsDownloadError as error:
            partial_rows = [row.to_dict() for row in error.manifest_rows]
            all_manifest.extend(partial_rows)
            total_raw_records += sum(int(row["record_count"]) for row in partial_rows)
            download_bytes = sum(
                Path(str(row["cache_path"])).stat().st_size
                for row in {str(value["response_hash"]): value for value in all_manifest}.values()
            )
            write_json(
                output / "options_download_manifest.json",
                {
                    "status": "blocked_options_download_resource_limit",
                    "requests_completed": len(all_manifest),
                    "raw_records": total_raw_records,
                    "canonical_records": total_canonical_records,
                    "download_bytes": download_bytes,
                    "pagination_complete": False,
                    "unexplained_truncations": 0,
                    "credential_exposures": 0,
                    "manifest_rows": all_manifest,
                },
            )
            raise
        total_raw_records += sum(row.record_count for row in result.manifest_rows)
        all_manifest.extend(row.to_dict() for row in result.manifest_rows)
        if total_raw_records > MAX_RAW_RECORDS:
            return _write_blocker(output, "blocked_options_download_resource_limit")
        completed_download_bytes = sum(
            Path(str(row["cache_path"])).stat().st_size
            for row in {str(value["response_hash"]): value for value in all_manifest}.values()
        )
        write_json(
            output / "options_download_manifest.json",
            {
                "status": "in_progress",
                "requests_completed": len(all_manifest),
                "raw_records": total_raw_records,
                "canonical_records": total_canonical_records,
                "download_bytes": completed_download_bytes,
                "pagination_complete": False,
                "unexplained_truncations": 0,
                "credential_exposures": 0,
                "manifest_rows": all_manifest,
            },
        )
        canonicalized = canonicalize_response_records(
            result.records,
            request_id=str(chunk["chunk_id"]),
            provider_schema_version=f"openapi-{OPENAPI_VERSION}",
        )
        for record in canonicalized.records:
            observed_provider_symbol = str(record["underlying_symbol"])
            if observed_provider_symbol.casefold() != provider_symbol.casefold():
                raise OptionsSchemaError(
                    "historical options response underlying differs from requested mapping"
                )
            record["provider_underlying_symbol"] = observed_provider_symbol
            record["underlying_symbol"] = symbol
        required_dates = {date.fromisoformat(str(value)) for value in chunk["required_trade_dates"]}
        bounded = [
            record
            for record in canonicalized.records
            if record["trade_date"] in required_dates and 7 <= int(record["dte"]) <= 90
        ]
        deduplicated = resolve_canonical_duplicates(bounded)
        total_canonical_records += len(deduplicated.records)
        total_duplicates += deduplicated.duplicate_records
        total_rejections += len(canonicalized.rejections)
        rejection_rows.extend(
            {
                "request_id": item.request_id,
                "record_index": item.record_index,
                "provider_record_id": item.provider_record_id,
                "reason_code": item.reason_code,
                "raw_record_hash": item.raw_record_hash,
            }
            for item in canonicalized.rejections
        )
        write_parquet(
            canonical_dir / f"{chunk['chunk_id']}.parquet",
            pd.DataFrame(
                deduplicated.records,
                columns=[*CANONICAL_OPTION_COLUMNS, "provider_underlying_symbol"],
            ),
        )
        download_bytes = sum(
            Path(str(row["cache_path"])).stat().st_size
            for row in {str(value["response_hash"]): value for value in all_manifest}.values()
        )
        if download_bytes > MAX_OPTIONS_BYTES:
            return _write_blocker(output, "blocked_options_download_resource_limit")
        write_json(
            output / "options_download_manifest.json",
            {
                "status": "in_progress",
                "requests_completed": len(all_manifest),
                "raw_records": total_raw_records,
                "canonical_records": total_canonical_records,
                "download_bytes": download_bytes,
                "pagination_complete": False,
                "unexplained_truncations": 0,
                "credential_exposures": 0,
                "manifest_rows": all_manifest,
            },
        )
    download_bytes = sum(
        Path(str(row["cache_path"])).stat().st_size
        for row in {str(value["response_hash"]): value for value in all_manifest}.values()
    )
    manifest = {
        "status": "supported",
        "requests_completed": len(all_manifest),
        "raw_records": total_raw_records,
        "canonical_records": total_canonical_records,
        "rejected_records": total_rejections,
        "duplicate_records": total_duplicates,
        "download_bytes": download_bytes,
        "pagination_complete": True,
        "unexplained_truncations": 0,
        "credential_exposures": 0,
        "manifest_rows": all_manifest,
    }
    serialized = json.dumps(manifest, sort_keys=True)
    if token in serialized or "api_token=" in serialized.casefold():
        raise RuntimeError("credential exposure detected in download manifest")
    write_json(output / "options_download_manifest.json", manifest)
    write_csv(
        output / "options_rejections.csv",
        pd.DataFrame(
            rejection_rows,
            columns=[
                "request_id",
                "record_index",
                "provider_record_id",
                "reason_code",
                "raw_record_hash",
            ],
        ),
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY)
    parser.add_argument("--data-dir", type=Path, default=options_data_dir())
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        result = execute_download(
            output=arguments.output.expanduser().resolve(),
            data_dir=arguments.data_dir.expanduser().resolve(),
        )
    except Exception as error:
        safe = redact_secrets(str(error), secrets=(os.environ.get("EODHD_API_TOKEN", ""),))
        if isinstance(error, OptionsResourceLimitExceeded):
            blocker = "blocked_options_download_resource_limit"
        elif safe == "blocked_historical_options_date_unavailable":
            blocker = safe
        elif isinstance(error, OptionsSchemaError):
            blocker = "blocked_eodhd_options_schema_unverified"
        else:
            blocker = "blocked_options_download_incomplete"
        _write_blocker(arguments.output.expanduser().resolve(), blocker)
        print(safe)
        return 1
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
