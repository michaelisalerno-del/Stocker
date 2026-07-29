#!/usr/bin/env python3
"""Sequential, resumable EODHD options downloader for the bounded V0 request plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, date, datetime
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
    sha256_bytes,
    stable_request_id,
)

UNDERLYING_SYMBOLS_ENDPOINT = "/mp/unicornbay/options/underlying-symbols"
OPTIONS_EOD_ENDPOINT = "/mp/unicornbay/options/eod"


def _write_blocker(output: Path, blocker: str) -> dict[str, Any]:
    preflight_path = output / "eodhd_options_api_preflight.json"
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
    )
    setup_http_requests_attempted = int(preflight.get("setup_http_requests_attempted", 0))
    setup_logical_requests_completed = int(preflight.get("setup_logical_requests_completed", 0))
    setup_manifest_rows = cast(list[dict[str, Any]], preflight.get("setup_manifest_rows", []))
    setup_response_bytes = sum(int(row.get("response_bytes", 0)) for row in setup_manifest_rows)
    setup_records_received = sum(int(row.get("record_count", 0)) for row in setup_manifest_rows)
    preflight_option_records = int(preflight.get("records_received", 0))
    decision = {
        **SAFETY_FLAGS,
        "status": blocker,
        "decision": blocker,
        "options_download_status": "blocked",
        "options_coverage_status": "blocked",
        "iv_excess_model_status": "blocked",
        "broad_conflict_movement_status": "blocked",
        "matched_control_status": "blocked",
        "provider_setup_http_requests_attempted": setup_http_requests_attempted,
        "provider_setup_logical_requests_completed": setup_logical_requests_completed,
        "provider_preflight_option_records": preflight_option_records,
    }
    write_json(output / "decision.json", decision)
    existing_manifest = output / "options_download_manifest.json"
    manifest = (
        json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing_manifest.is_file()
        else {}
    )
    manifest["status"] = blocker
    manifest.pop("setup_requests_completed", None)
    bulk_manifest_rows = cast(list[dict[str, Any]], manifest.get("manifest_rows", []))
    bulk_http_requests_attempted = sum(int(row.get("attempts", 0)) for row in bulk_manifest_rows)
    manifest["setup_http_requests_attempted"] = setup_http_requests_attempted
    manifest["setup_logical_requests_completed"] = setup_logical_requests_completed
    manifest["setup_manifest_rows"] = setup_manifest_rows
    manifest["setup_response_bytes"] = setup_response_bytes
    manifest["setup_records_received"] = setup_records_received
    manifest["preflight_option_records"] = preflight_option_records
    manifest["bulk_http_requests_attempted"] = bulk_http_requests_attempted
    manifest["total_http_requests_attempted"] = (
        setup_http_requests_attempted + bulk_http_requests_attempted
    )
    manifest["total_provider_records_received"] = setup_records_received + int(
        manifest.get("raw_records", 0)
    )
    manifest["total_download_bytes"] = setup_response_bytes + int(manifest.get("download_bytes", 0))
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
            "provider_setup_http_requests_attempted": setup_http_requests_attempted,
            "provider_setup_logical_requests_completed": setup_logical_requests_completed,
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
        f"Provider setup HTTP attempts recorded: {setup_http_requests_attempted}; "
        f"successful logical setup requests: {setup_logical_requests_completed}; "
        f"preflight option records: {preflight_option_records}."
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
        cache_dir: Path,
        max_attempts: int = 4,
        maximum_download_bytes: int | None = None,
    ) -> None:
        if maximum_download_bytes is not None and maximum_download_bytes < 1:
            raise ValueError("maximum_download_bytes must be positive")
        self.transport = transport
        self.token = token
        self.minimum_interval = 60.0 / float(requests_per_minute)
        self.max_attempts = max_attempts
        self.cache_dir = cache_dir
        self.maximum_download_bytes = maximum_download_bytes
        self._download_bytes_accounted = 0
        self.last_request_monotonic: float | None = None
        self.http_requests_attempted = 0
        self.logical_requests_completed = 0
        self.manifest_rows: list[dict[str, object]] = []
        self.last_response_sha256: str | None = None

    def account_cached_bytes(self, response_bytes: int) -> None:
        """Count a verified resumed response against the same fixed byte ceiling."""

        if response_bytes < 0:
            raise ValueError("response_bytes must be non-negative")
        if (
            self.maximum_download_bytes is not None
            and self._download_bytes_accounted + response_bytes > self.maximum_download_bytes
        ):
            raise OptionsResourceLimitExceeded(
                "blocked_options_download_resource_limit: download-byte ceiling"
            )
        self._download_bytes_accounted += response_bytes

    @staticmethod
    def _close_response(response: requests.Response) -> None:
        response.close()

    def _bounded_response_content(self, response: requests.Response) -> bytes:
        """Stream no more than the remaining setup-response byte allowance."""

        if self.maximum_download_bytes is None:
            content = response.content
            self._download_bytes_accounted += len(content)
            return content
        remaining = self.maximum_download_bytes - self._download_bytes_accounted
        content_length_value = response.headers.get("Content-Length")
        content_length: int | None = None
        if content_length_value is not None:
            try:
                content_length = int(content_length_value)
            except ValueError:
                content_length = None
        if content_length is not None and content_length > remaining:
            self._close_response(response)
            raise OptionsResourceLimitExceeded(
                "blocked_options_download_resource_limit: download-byte ceiling"
            )
        chunks: list[bytes] = []
        consumed = 0
        chunk_size = max(min(65_536, remaining), 1)
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            available = remaining - consumed
            if len(chunk) > available:
                self._close_response(response)
                raise OptionsResourceLimitExceeded(
                    "blocked_options_download_resource_limit: download-byte ceiling"
                )
            chunks.append(bytes(chunk))
            consumed += len(chunk)
            self._download_bytes_accounted += len(chunk)
            if consumed == remaining and content_length is None:
                self._close_response(response)
                raise OptionsResourceLimitExceeded(
                    "blocked_options_download_resource_limit: download-byte ceiling"
                )
        self._close_response(response)
        return b"".join(chunks)

    def _cache_response(self, content: bytes) -> tuple[str, Path]:
        response_hash = sha256_bytes(content)
        destination = self.cache_dir / f"{response_hash}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        return response_hash, destination

    @staticmethod
    def _record_count(payload: object) -> int:
        if isinstance(payload, dict):
            data = payload.get("data")
            return len(data) if isinstance(data, list) else 0
        return len(payload) if isinstance(payload, list) else 0

    def _record_success(
        self,
        *,
        endpoint: str,
        params: dict[str, object],
        payload: object,
        content: bytes,
        attempts: int,
        started_at: str,
    ) -> None:
        response_hash, cache_path = self._cache_response(content)
        self.last_response_sha256 = response_hash
        self.logical_requests_completed += 1
        self.manifest_rows.append(
            {
                "request_id": stable_request_id(endpoint, params),
                "endpoint": endpoint,
                "underlying_symbol": str(params.get("filter[underlying_symbol]", "")),
                "trade_date_from": str(params.get("filter[tradetime_from]", "")),
                "trade_date_to": str(params.get("filter[tradetime_to]", "")),
                "strike_from": str(params.get("filter[strike_from]", "")),
                "strike_to": str(params.get("filter[strike_to]", "")),
                "expiration_from": str(params.get("filter[exp_date_from]", "")),
                "expiration_to": str(params.get("filter[exp_date_to]", "")),
                "offset": int(params.get("page[offset]", 0)),
                "limit": int(params.get("page[limit]", 0)),
                "response_status": 200,
                "record_count": self._record_count(payload),
                "response_hash": response_hash,
                "response_bytes": len(content),
                "attempts": attempts,
                "started_at": started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "cache_path": str(cache_path.resolve()),
            }
        )

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
        started_at = datetime.now(UTC).isoformat()
        for attempt in range(1, self.max_attempts + 1):
            self.wait_before_next_request()
            self.last_request_monotonic = time.monotonic()
            try:
                self.http_requests_attempted += 1
                response = self.transport.get(url, params=authenticated, timeout=timeout)
            except Exception as exc:
                if attempt >= self.max_attempts:
                    message = redact_secrets(str(exc), secrets=(self.token,))
                    raise RuntimeError(f"EODHD setup transport failure: {message}") from None
                time.sleep(float(2 ** (attempt - 1)))
                continue
            if response.status_code == 200:
                try:
                    content = self._bounded_response_content(response)
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
                token_bytes = self.token.encode("utf-8")
                if token_bytes and token_bytes in content:
                    response.close()
                    raise RuntimeError("EODHD setup response contains credential material")
                try:
                    payload = json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    response.close()
                    raise OptionsSchemaError("EODHD setup response is not valid JSON") from exc
                self._record_success(
                    endpoint=endpoint,
                    params=params,
                    payload=payload,
                    content=content,
                    attempts=attempt,
                    started_at=started_at,
                )
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
    record_evidence: list[dict[str, object]] = []
    for index, (resource, attributes) in enumerate(zip(object_rows, attribute_rows, strict=True)):
        resource_id = resource.get("id")
        record_evidence.append(
            {
                "record_index": index,
                "resource_id_sha256": (
                    sha256_bytes(resource_id.encode("utf-8"))
                    if isinstance(resource_id, str)
                    else None
                ),
                "resource_observation_date": _resource_observation_date(resource),
                "tradetime_date": _provider_session_date(attributes.get("tradetime")),
                "bid_observation_date": _provider_session_date(attributes.get("bid_date")),
                "ask_observation_date": _provider_session_date(attributes.get("ask_date")),
                "expiration_date": _provider_session_date(attributes.get("exp_date")),
                "provider_dte": attributes.get("dte"),
            }
        )
    evidence_projection_sha256 = sha256_bytes(
        json.dumps(
            record_evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
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
    all_returned_rows_observation_date_verified = bool(record_evidence) and all(
        row["resource_observation_date"] is not None
        and row["resource_observation_date"]
        == row["bid_observation_date"]
        == row["ask_observation_date"]
        for row in record_evidence
    )
    sample_rows_match_requested_observation_date = bool(
        all_returned_rows_observation_date_verified
        and all(row["resource_observation_date"] == trade_date for row in record_evidence)
    )
    setup_manifest_rows = [dict(row) for row in requester.manifest_rows]
    preflight_manifest = next(
        (
            row
            for row in reversed(setup_manifest_rows)
            if row.get("endpoint") == OPTIONS_EOD_ENDPOINT
        ),
        {},
    )
    america_new_york_mapping_confirmed = bool(record_evidence) and all(
        row["tradetime_date"] is not None
        and row["bid_observation_date"] is not None
        and row["ask_observation_date"] is not None
        for row in record_evidence
    )
    blocker_reason = (
        "the official historical EOD schema exposes no EOD observation-date filter; "
        "matching sample rows cannot establish exact historical chain retrieval"
        if sample_rows_match_requested_observation_date
        else "the documented tradetime filter selects last-trade activity, while returned "
        "EOD resource observations cannot establish the requested prior-close session"
    )
    write_json(
        output / "eodhd_options_api_preflight.json",
        {
            "status": "blocked_historical_options_date_unavailable",
            "endpoint": OPTIONS_EOD_ENDPOINT,
            "symbol": symbol,
            "requested_session_date": trade_date,
            "record_limit": 10,
            "records_received": len(data),
            "setup_http_requests_attempted": requester.http_requests_attempted,
            "setup_logical_requests_completed": requester.logical_requests_completed,
            "setup_manifest_rows": setup_manifest_rows,
            "response_sha256": preflight_manifest.get("response_hash"),
            "response_bytes": preflight_manifest.get("response_bytes", 0),
            "response_cache_path": preflight_manifest.get("cache_path", ""),
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
            "record_evidence": record_evidence,
            "evidence_projection_sha256": evidence_projection_sha256,
            "all_returned_rows_observation_date_verified": (
                all_returned_rows_observation_date_verified
            ),
            "sample_rows_match_requested_observation_date": (
                sample_rows_match_requested_observation_date
            ),
            "tradetime_is_eod_observation_date": False,
            "official_observation_date_filter_available": False,
            "exact_requested_observation_date_confirmed": False,
            "america_new_york_mapping_confirmed": america_new_york_mapping_confirmed,
            "authentication_redacted": True,
            "blocker_reason": blocker_reason,
        },
    )
    raise RuntimeError("blocked_historical_options_date_unavailable")


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
    setup_requester = SetupRequester(
        transport,
        token=token,
        requests_per_minute=pace,
        cache_dir=data_dir / "raw" / "setup",
    )
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
