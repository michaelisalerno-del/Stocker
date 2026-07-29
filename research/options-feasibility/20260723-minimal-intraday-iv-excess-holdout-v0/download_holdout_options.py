#!/usr/bin/env python3
"""Download bounded exact-D-1 EODHD chains for the frozen 2025 holdout."""

from __future__ import annotations

# ruff: noqa: E402 -- numerical/network limits are fixed before imports.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import requests
from pandas_market_calendars import get_calendar

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PRIMARY = EXPERIMENT_DIR / "artifacts" / "primary"
DEFAULT_PROVIDER_ROOT = Path(
    "/Users/michaelsalerno/StockerLocal/data/processed/source=eodhd/instrument_type=stock"
)
DEFAULT_CACHE_ROOT = REPO_ROOT / "data/vendor/eodhd/options/minimal-intraday-iv-excess-holdout-v0"

for _package in ("stocker_research", "stocker_data", "stocker_core"):
    _source = REPO_ROOT / "packages" / _package / "src"
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))

from stocker_research.eodhd_options_downloader_v0 import (
    OPTIONS_EOD_ENDPOINT,
    DownloadConfig,
    EODHDOptionsDownloader,
    OptionsAuthenticationError,
    OptionsDownloadError,
    OptionsRequest,
    OptionsResourceLimitExceeded,
    canonicalize_response_records,
    provider_eod_observation_date,
    resolve_canonical_duplicates,
    stable_request_id,
)
from stocker_research.minimal_intraday_iv_excess_holdout_v0 import (
    HOLDOUT_END,
    HOLDOUT_START,
    PROTECTED_START,
    SAFETY_FLAGS,
    assert_safety_flags,
)
from stocker_research.stock_layer_iv_excess_attribution_v0 import FROZEN_COHORT

MAXIMUM_RECORDS = 350_000
MAXIMUM_BYTES = 1_000_000_000
_STANDARD_OCC = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True)
class PlannedRequest:
    """One frozen exact-date chain request."""

    symbol: str
    holdout_session: str
    required_options_date: str
    previous_close: float
    strike_from: float
    strike_to: float
    expiration_from: str
    expiration_to: str


class DownloadBlocked(RuntimeError):
    """One authorized fail-closed download blocker."""

    def __init__(self, decision: str, detail: str) -> None:
        super().__init__(detail)
        self.decision = decision
        self.detail = detail


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, Path, date)):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actual_holdout_sessions() -> tuple[date, ...]:
    """Return only actual XNYS sessions inside the authorized calendar range."""

    calendar = get_calendar("XNYS")
    valid = calendar.valid_days(HOLDOUT_START, HOLDOUT_END, tz="America/New_York")
    sessions = tuple(value.date() for value in valid)
    if len(sessions) != 85 or sessions[0] != date(2025, 9, 2):
        raise DownloadBlocked(
            "blocked_reproducibility_or_audit_failure",
            f"unexpected XNYS holdout session surface: {len(sessions)}",
        )
    return sessions


def prior_session_map(sessions: Sequence[date]) -> dict[date, date]:
    calendar = get_calendar("XNYS")
    valid = calendar.valid_days(
        start_date=min(sessions) - timedelta(days=14),
        end_date=max(sessions),
        tz="America/New_York",
    )
    ordered = [value.date() for value in valid]
    positions = {session: index for index, session in enumerate(ordered)}
    return {session: ordered[positions[session] - 1] for session in sessions}


def regular_session_closes(provider_root: Path, symbol: str) -> dict[date, float]:
    """Read only pre-2026 bars and return the final completed regular-session close."""

    path = provider_root / f"symbol={symbol}" / "timeframe=5m" / "data.parquet"
    if not path.is_file():
        raise DownloadBlocked(
            "blocked_insufficient_holdout_support",
            f"missing five-minute source for {symbol}: {path}",
        )
    raw = pd.read_parquet(path, columns=["timestamp", "close"])
    timestamp = pd.to_datetime(raw["timestamp"], errors="raise", utc=True)
    raw = raw.loc[timestamp.lt(PROTECTED_START.tz_localize("UTC"))].copy()
    raw["timestamp"] = timestamp.loc[raw.index]
    raw["session"] = raw["timestamp"].dt.tz_convert("America/New_York").dt.date
    calendar = get_calendar("XNYS")
    schedule = calendar.schedule(
        start_date=min(raw["session"]),
        end_date=max(raw["session"]),
        tz="UTC",
    )
    bounds = {
        index.date(): (pd.Timestamp(row.market_open), pd.Timestamp(row.market_close))
        for index, row in schedule.iterrows()
    }
    selected: dict[date, float] = {}
    for session, group in raw.groupby("session", sort=True, observed=True):
        boundary = bounds.get(cast(date, session))
        if boundary is None:
            continue
        regular = group.loc[
            group["timestamp"].ge(boundary[0]) & group["timestamp"].lt(boundary[1])
        ].sort_values("timestamp", kind="mergesort")
        regular = regular.loc[
            np.isfinite(pd.to_numeric(regular["close"], errors="coerce"))
            & pd.to_numeric(regular["close"], errors="coerce").gt(0.0)
        ]
        if regular.empty:
            continue
        close = float(regular.iloc[-1]["close"])
        if math.isfinite(close) and close > 0.0:
            selected[cast(date, session)] = close
    return selected


def build_request_plan(provider_root: Path) -> list[PlannedRequest]:
    """Build the immutable 20-stock by 85-session request surface."""

    sessions = actual_holdout_sessions()
    previous = prior_session_map(sessions)
    rows: list[PlannedRequest] = []
    for symbol in FROZEN_COHORT:
        closes = regular_session_closes(provider_root, symbol)
        for session in sessions:
            required = previous[session]
            close = closes.get(required)
            if close is None:
                raise DownloadBlocked(
                    "blocked_insufficient_holdout_support",
                    f"missing prior-session close for {symbol}/{required}",
                )
            rows.append(
                PlannedRequest(
                    symbol=symbol,
                    holdout_session=session.isoformat(),
                    required_options_date=required.isoformat(),
                    previous_close=close,
                    strike_from=round(close * 0.70, 6),
                    strike_to=round(close * 1.30, 6),
                    expiration_from=(required + timedelta(days=7)).isoformat(),
                    expiration_to=(required + timedelta(days=45)).isoformat(),
                )
            )
    if len(rows) != 1_700:
        raise AssertionError("request plan must contain exactly 1,700 stock-session requests")
    return rows


def write_request_plan(rows: Sequence[PlannedRequest], provider_root: Path) -> None:
    by_month = pd.Series([row.holdout_session[:7] for row in rows]).value_counts().sort_index()
    write_json(
        PRIMARY / "holdout_options_request_plan.json",
        {
            **SAFETY_FLAGS,
            "endpoint": OPTIONS_EOD_ENDPOINT,
            "compact": False,
            "provider_root": str(provider_root),
            "requests_planned": len(rows),
            "stocks": len({row.symbol for row in rows}),
            "sessions": len({row.holdout_session for row in rows}),
            "requests_by_month": {str(key): int(value) for key, value in by_month.items()},
            "front_dte_minimum": 7,
            "front_dte_maximum": 45,
            "back_expiry_46_to_90_requested": False,
            "strike_bounds": "70_to_130_percent_of_exact_prior_session_unadjusted_close",
            "requests": [asdict(row) for row in rows],
        },
    )


def load_token(env_file: Path | None) -> str:
    """Load the named token without copying or printing it."""

    token = os.environ.get("EODHD_API_TOKEN", "").strip()
    if token:
        return token
    if env_file is not None and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "EODHD_API_TOKEN":
                token = value.strip().strip("\"'")
                break
    if not token:
        raise DownloadBlocked(
            "blocked_missing_eodhd_api_token",
            "EODHD_API_TOKEN is unavailable",
        )
    return token


def request_from_plan(row: PlannedRequest) -> OptionsRequest:
    return OptionsRequest(
        underlying_symbol=row.symbol,
        trade_date_from=date.fromisoformat(row.required_options_date),
        trade_date_to=date.fromisoformat(row.required_options_date),
        strike_from=row.strike_from,
        strike_to=row.strike_to,
        expiration_from=date.fromisoformat(row.expiration_from),
        expiration_to=date.fromisoformat(row.expiration_to),
        compact=False,
        endpoint=OPTIONS_EOD_ENDPOINT,
    )


def resume_path(cache_root: Path, request: OptionsRequest) -> Path:
    request_id = stable_request_id(
        request.endpoint,
        request.parameters(offset=0, limit=1000),
    )
    return cache_root / "manifests" / "completed" / f"{request_id}.json"


def is_standard_occ(record: Mapping[str, Any], *, symbol: str) -> bool:
    """Reject adjusted, unresolved, or internally inconsistent contract identities."""

    attributes = record.get("attributes", record)
    if not isinstance(attributes, Mapping):
        return False
    contract = attributes.get("contract")
    strike = attributes.get("strike")
    if not isinstance(contract, str) or not isinstance(strike, (int, float)):
        return False
    match = _STANDARD_OCC.fullmatch(contract)
    if match is None or match.group(1) != symbol.upper():
        return False
    if not math.isclose(int(match.group(4)) / 1000.0, float(strike), abs_tol=1e-9):
        return False
    adjusted = attributes.get("adjusted")
    if adjusted not in (None, False, 0, "0", "false", "False"):
        return False
    multiplier = attributes.get("contract_size", attributes.get("multiplier"))
    if multiplier is not None:
        try:
            if float(cast(Any, multiplier)) != 100.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def run_download(
    rows: Sequence[PlannedRequest],
    *,
    token: str,
    cache_root: Path,
) -> dict[str, Any]:
    """Execute the one-process bounded download and canonical materialisation."""

    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token=token,
            data_dir=cache_root,
            page_limit=1000,
            maximum_offset=10000,
            request_timeout_seconds=30.0,
            max_attempts=4,
            exponential_backoff_seconds=1.0,
            requests_per_minute=300,
            maximum_raw_records=MAXIMUM_RECORDS,
            maximum_download_bytes=MAXIMUM_BYTES,
        ),
        transport=requests.Session(),
    )
    canonical_records: list[dict[str, Any]] = []
    request_audit: list[dict[str, Any]] = []
    totals = {
        "requests_completed": 0,
        "cached_complete_receipts_reused": 0,
        "network_requests_completed": 0,
        "records_returned": 0,
        "exact_date_records_retained": 0,
        "extra_date_records_rejected": 0,
        "protected_date_records_rejected": 0,
        "nonstandard_contract_records_rejected": 0,
        "canonical_records_accepted": 0,
        "canonical_records_rejected": 0,
        "bytes_downloaded": 0,
    }
    for index, planned in enumerate(rows, start=1):
        request = request_from_plan(planned)
        receipt = resume_path(cache_root, request)
        was_cached = receipt.is_file()
        result = downloader.download_with_splitting(request)
        required = date.fromisoformat(planned.required_options_date)
        exact: list[Mapping[str, Any]] = []
        protected_rejected = 0
        for record in result.records:
            observed = provider_eod_observation_date(record)
            if observed >= PROTECTED_START.date():
                protected_rejected += 1
            if observed == required and observed < PROTECTED_START.date():
                exact.append(record)
        standard = [record for record in exact if is_standard_occ(record, symbol=planned.symbol)]
        nonstandard = len(exact) - len(standard)
        root_request_id = stable_request_id(
            request.endpoint,
            request.parameters(offset=0, limit=1000),
        )
        canonical = canonicalize_response_records(
            standard,
            request_id=root_request_id,
            provider_schema_version="eodhd_options_eod_noncompact_exact_date_v1",
        )
        canonical_records.extend(canonical.records)
        returned = len(result.records)
        bytes_for_request = sum(
            Path(manifest.cache_path).stat().st_size
            for manifest in result.manifest_rows
            if not manifest.superseded_by_split
        )
        new_bytes = 0 if was_cached else bytes_for_request
        audit = {
            **asdict(planned),
            "request_id": root_request_id,
            "cached_complete_receipt_reused": was_cached,
            "records_returned": returned,
            "exact_date_records_retained_before_contract_gate": len(exact),
            "extra_date_records_rejected": returned - len(exact),
            "protected_date_records_rejected": protected_rejected,
            "nonstandard_contract_records_rejected": nonstandard,
            "canonical_records_accepted": len(canonical.records),
            "canonical_records_rejected": len(canonical.rejections),
            "new_bytes_downloaded": new_bytes,
        }
        request_audit.append(audit)
        totals["requests_completed"] += 1
        totals["cached_complete_receipts_reused"] += int(was_cached)
        totals["network_requests_completed"] += int(not was_cached)
        totals["records_returned"] += returned
        totals["exact_date_records_retained"] += len(exact)
        totals["extra_date_records_rejected"] += returned - len(exact)
        totals["protected_date_records_rejected"] += protected_rejected
        totals["nonstandard_contract_records_rejected"] += nonstandard
        totals["canonical_records_accepted"] += len(canonical.records)
        totals["canonical_records_rejected"] += len(canonical.rejections)
        totals["bytes_downloaded"] += new_bytes
        if index % 100 == 0:
            print(f"completed {index}/{len(rows)} exact-date requests", flush=True)
    deduplicated = resolve_canonical_duplicates(canonical_records)
    canonical_frame = pd.DataFrame(deduplicated.records)
    canonical_path = cache_root / "canonical" / "exact_holdout_options.parquet"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_frame.to_parquet(canonical_path, index=False)
    request_audit_path = cache_root / "canonical" / "request_audit.parquet"
    pd.DataFrame(request_audit).to_parquet(request_audit_path, index=False)
    coverage = pd.DataFrame(request_audit)
    coverage["month"] = coverage["holdout_session"].str[:7]
    coverage_table = (
        coverage.groupby(["symbol", "month"], sort=True, observed=True)
        .agg(
            requests=("request_id", "size"),
            requests_with_exact_records=(
                "exact_date_records_retained_before_contract_gate",
                lambda values: int((values > 0).sum()),
            ),
            exact_records=("canonical_records_accepted", "sum"),
        )
        .reset_index()
    )
    coverage_table.to_csv(PRIMARY / "holdout_options_coverage.csv", index=False)
    manifest = {
        **SAFETY_FLAGS,
        "status": "complete",
        "endpoint": OPTIONS_EOD_ENDPOINT,
        "compact": False,
        "requests_planned": len(rows),
        **totals,
        "maximum_new_records": MAXIMUM_RECORDS,
        "maximum_new_bytes": MAXIMUM_BYTES,
        "record_limit_passed": totals["records_returned"] <= MAXIMUM_RECORDS,
        "byte_limit_passed": totals["bytes_downloaded"] <= MAXIMUM_BYTES,
        "duplicate_records": deduplicated.duplicate_records,
        "conflicting_duplicate_groups": deduplicated.conflicting_duplicate_groups,
        "canonical_rows": len(canonical_frame),
        "canonical_cache_path": str(canonical_path),
        "canonical_cache_sha256": sha256_file(canonical_path),
        "request_audit_cache_path": str(request_audit_path),
        "request_audit_sha256": sha256_file(request_audit_path),
        "pair_coverage_by_stock_and_month": coverage_table.to_dict(orient="records"),
    }
    assert_safety_flags(manifest)
    return manifest


def blocked_manifest(
    *,
    decision: str,
    detail: str,
    requests_planned: int,
) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "status": "blocked",
        "overall_decision": decision,
        "detail": detail,
        "requests_planned": requests_planned,
        "requests_completed": 0,
        "records_returned": 0,
        "exact_date_records_retained": 0,
        "extra_date_records_rejected": 0,
        "protected_date_records_rejected": 0,
        "bytes_downloaded": 0,
    }


def summarize_completed_cache(
    rows: Sequence[PlannedRequest],
    *,
    cache_root: Path,
    decision: str,
    detail: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit completed receipts after a fail-closed bounded stop."""

    plan_by_request_id = {resume_path(cache_root, request_from_plan(row)).stem: row for row in rows}
    completed_root = cache_root / "manifests/completed"
    request_rows: list[dict[str, object]] = []
    referenced_raw_paths: set[Path] = set()
    totals = {
        "records_returned": 0,
        "exact_date_records_retained": 0,
        "extra_date_records_rejected": 0,
        "protected_date_records_rejected": 0,
        "nonstandard_contract_records_rejected": 0,
        "canonical_records_accepted": 0,
        "canonical_records_rejected": 0,
        "bytes_downloaded": 0,
    }
    for receipt in sorted(completed_root.glob("*.json")):
        planned = plan_by_request_id.get(receipt.stem)
        if planned is None:
            raise DownloadBlocked(
                "blocked_reproducibility_or_audit_failure",
                f"completed request is absent from frozen plan: {receipt.stem}",
            )
        stored = cast(dict[str, Any], json.loads(receipt.read_text(encoding="utf-8")))
        manifest_rows = cast(list[dict[str, Any]], stored["manifest_rows"])
        raw_records: list[Mapping[str, Any]] = []
        response_bytes = 0
        for manifest_row in manifest_rows:
            if bool(manifest_row.get("superseded_by_split", False)):
                continue
            raw_path = Path(str(manifest_row["cache_path"]))
            referenced_raw_paths.add(raw_path.resolve())
            content = raw_path.read_bytes()
            response_bytes += len(content)
            payload = cast(dict[str, Any], json.loads(content))
            data = cast(list[Any], payload["data"])
            fields = cast(dict[str, Any], payload["meta"]).get("fields")
            for item in data:
                if isinstance(item, Mapping):
                    raw_records.append(cast(Mapping[str, Any], item))
                elif isinstance(item, list) and isinstance(fields, list):
                    raw_records.append(
                        {
                            "type": "options-eod",
                            "attributes": dict(
                                zip((str(field) for field in fields), item, strict=True)
                            ),
                        }
                    )
                else:
                    raise DownloadBlocked(
                        "blocked_holdout_options_download_failure",
                        "cached provider response schema differs",
                    )
        required = date.fromisoformat(planned.required_options_date)
        exact: list[Mapping[str, Any]] = []
        protected = 0
        for record in raw_records:
            observed = provider_eod_observation_date(record)
            protected += int(observed >= PROTECTED_START.date())
            if observed == required and observed < PROTECTED_START.date():
                exact.append(record)
        standard = [record for record in exact if is_standard_occ(record, symbol=planned.symbol)]
        canonical = canonicalize_response_records(
            standard,
            request_id=receipt.stem,
            provider_schema_version="eodhd_options_eod_noncompact_exact_date_v1",
        )
        values = {
            "records_returned": len(raw_records),
            "exact_date_records_retained": len(exact),
            "extra_date_records_rejected": len(raw_records) - len(exact),
            "protected_date_records_rejected": protected,
            "nonstandard_contract_records_rejected": len(exact) - len(standard),
            "canonical_records_accepted": len(canonical.records),
            "canonical_records_rejected": len(canonical.rejections),
            "bytes_downloaded": response_bytes,
        }
        for name, value in values.items():
            totals[name] += int(value)
        request_rows.append(
            {
                **asdict(planned),
                "request_id": receipt.stem,
                **values,
            }
        )
    request_frame = pd.DataFrame(request_rows)
    completed_coverage = pd.DataFrame(
        columns=[
            "symbol",
            "month",
            "requests_completed",
            "requests_with_exact_records",
            "canonical_records",
        ]
    )
    if not request_frame.empty:
        request_frame["month"] = request_frame["holdout_session"].str[:7]
        completed_coverage = (
            request_frame.groupby(["symbol", "month"], sort=True, observed=True)
            .agg(
                requests_completed=("request_id", "size"),
                requests_with_exact_records=(
                    "exact_date_records_retained",
                    lambda values: int((values > 0).sum()),
                ),
                canonical_records=("canonical_records_accepted", "sum"),
            )
            .reset_index()
        )
    plan_frame = pd.DataFrame([asdict(row) for row in rows])
    plan_frame["month"] = plan_frame["holdout_session"].str[:7]
    planned_coverage = (
        plan_frame.groupby(["symbol", "month"], sort=True, observed=True)
        .size()
        .rename("requests_planned")
        .reset_index()
    )
    coverage = planned_coverage.merge(
        completed_coverage,
        on=["symbol", "month"],
        how="left",
        validate="one_to_one",
    )
    integer_columns = [
        "requests_completed",
        "requests_with_exact_records",
        "canonical_records",
    ]
    for column in integer_columns:
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0).astype(int)
    coverage["requests_remaining"] = (
        coverage["requests_planned"] - coverage["requests_completed"]
    ).astype(int)
    coverage["selected_pair_sessions"] = pd.NA
    coverage["pair_coverage_status"] = "blocked_before_pair_selection"
    all_raw_paths = {raw_path.resolve() for raw_path in (cache_root / "raw").glob("**/*.json")}
    incomplete_raw_paths = sorted(all_raw_paths - referenced_raw_paths)
    if len(incomplete_raw_paths) > 1:
        raise DownloadBlocked(
            "blocked_reproducibility_or_audit_failure",
            "more than one unreceipted raw response cannot be assigned deterministically",
        )
    completed_request_ids = set(request_frame.get("request_id", pd.Series(dtype=str)).astype(str))
    incomplete_plan = [
        row
        for row in rows
        if resume_path(cache_root, request_from_plan(row)).stem not in completed_request_ids
    ]
    incomplete_request_records = 0
    incomplete_request_bytes = 0
    incomplete_exact_date_records = 0
    incomplete_extra_date_records = 0
    incomplete_protected_date_records = 0
    for raw_path in incomplete_raw_paths:
        if not incomplete_plan:
            raise DownloadBlocked(
                "blocked_reproducibility_or_audit_failure",
                "unreceipted raw response has no remaining frozen request",
            )
        planned = incomplete_plan[0]
        required = date.fromisoformat(planned.required_options_date)
        content = raw_path.read_bytes()
        incomplete_request_bytes += len(content)
        payload = cast(dict[str, Any], json.loads(content))
        data = payload.get("data")
        if not isinstance(data, list):
            raise DownloadBlocked(
                "blocked_holdout_options_download_failure",
                "incomplete cached provider response schema differs",
            )
        fields = cast(dict[str, Any], payload.get("meta", {})).get("fields")
        decoded: list[Mapping[str, Any]] = []
        for item in data:
            if isinstance(item, Mapping):
                decoded.append(cast(Mapping[str, Any], item))
            elif isinstance(item, list) and isinstance(fields, list):
                decoded.append(
                    {
                        "type": "options-eod",
                        "attributes": dict(
                            zip((str(field) for field in fields), item, strict=True)
                        ),
                    }
                )
            else:
                raise DownloadBlocked(
                    "blocked_holdout_options_download_failure",
                    "incomplete cached provider response row differs",
                )
        incomplete_request_records += len(decoded)
        for record in decoded:
            observed = provider_eod_observation_date(record)
            incomplete_protected_date_records += int(observed >= PROTECTED_START.date())
            if observed == required and observed < PROTECTED_START.date():
                incomplete_exact_date_records += 1
            else:
                incomplete_extra_date_records += 1
    complete_records = totals["records_returned"]
    complete_exact = totals["exact_date_records_retained"]
    complete_extra = totals["extra_date_records_rejected"]
    complete_protected = totals["protected_date_records_rejected"]
    pair_coverage = [
        {
            "symbol": str(row.symbol),
            "month": str(row.month),
            "selected_pair_sessions": None,
            "status": "blocked_before_pair_selection",
        }
        for row in coverage.itertuples(index=False)
    ]
    manifest = {
        **SAFETY_FLAGS,
        "status": "blocked",
        "overall_decision": decision,
        "detail": detail,
        "endpoint": OPTIONS_EOD_ENDPOINT,
        "compact": False,
        "requests_planned": len(rows),
        "requests_completed": len(request_frame),
        "requests_remaining": len(rows) - len(request_frame),
        **totals,
        "exact_date_records_returned": complete_exact + incomplete_exact_date_records,
        "exact_date_records_retained": complete_exact,
        "exact_date_records_rejected_incomplete": incomplete_exact_date_records,
        "extra_date_records_rejected": complete_extra + incomplete_extra_date_records,
        "protected_date_records_rejected": (complete_protected + incomplete_protected_date_records),
        "complete_receipt_bytes_downloaded": totals["bytes_downloaded"],
        "incomplete_request_bytes_downloaded": incomplete_request_bytes,
        "bytes_downloaded": totals["bytes_downloaded"] + incomplete_request_bytes,
        "maximum_new_records": MAXIMUM_RECORDS,
        "maximum_new_bytes": MAXIMUM_BYTES,
        "resource_ceiling_reached": decision == "blocked_quick_resource_limit",
        "partial_cache_not_used_for_modeling": True,
        "partial_stock_subgroup_not_selected": True,
        "request_coverage_by_stock_and_month": coverage.drop(
            columns=["selected_pair_sessions", "pair_coverage_status"]
        ).to_dict(orient="records"),
        "pair_coverage_by_stock_and_month": pair_coverage,
    }
    if decision == "blocked_quick_resource_limit":
        manifest["complete_receipt_records_returned"] = complete_records
        manifest["incomplete_request_records_returned"] = incomplete_request_records
        manifest["incomplete_request_records_excluded"] = incomplete_request_records
        manifest["incomplete_request_exact_date_records"] = incomplete_exact_date_records
        manifest["incomplete_request_extra_date_records"] = incomplete_extra_date_records
        manifest["incomplete_request_protected_date_records"] = incomplete_protected_date_records
        manifest["incomplete_raw_responses"] = len(incomplete_raw_paths)
        manifest["records_returned"] = complete_records + incomplete_request_records
    return manifest, coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-root", type=Path, default=DEFAULT_PROVIDER_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summarize-blocked-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    PRIMARY.mkdir(parents=True, exist_ok=True)
    rows: list[PlannedRequest] = []
    try:
        rows = build_request_plan(arguments.provider_root)
        write_request_plan(rows, arguments.provider_root)
        if arguments.plan_only:
            print(f"planned {len(rows)} bounded exact-date requests")
            return 0
        if arguments.summarize_blocked_cache:
            manifest, coverage = summarize_completed_cache(
                rows,
                cache_root=arguments.cache_root,
                decision="blocked_quick_resource_limit",
                detail="blocked_options_download_resource_limit: raw-record ceiling",
            )
            coverage.to_csv(PRIMARY / "holdout_options_coverage.csv", index=False)
            write_json(PRIMARY / "holdout_options_download_manifest.json", manifest)
            print(
                f"audited {manifest['requests_completed']} complete receipts after "
                "the raw-record ceiling"
            )
            return 2
        token = load_token(arguments.env_file)
        manifest = run_download(rows, token=token, cache_root=arguments.cache_root)
        write_json(PRIMARY / "holdout_options_download_manifest.json", manifest)
        print(
            f"completed {manifest['requests_completed']} requests; "
            f"retained {manifest['canonical_rows']} canonical exact-date records"
        )
        return 0
    except DownloadBlocked as error:
        manifest, coverage = summarize_completed_cache(
            rows,
            cache_root=arguments.cache_root,
            decision=error.decision,
            detail=error.detail,
        )
        coverage.to_csv(PRIMARY / "holdout_options_coverage.csv", index=False)
    except OptionsAuthenticationError as error:
        manifest = blocked_manifest(
            decision="blocked_missing_eodhd_api_token",
            detail=str(error),
            requests_planned=len(rows),
        )
    except OptionsResourceLimitExceeded as error:
        manifest, coverage = summarize_completed_cache(
            rows,
            cache_root=arguments.cache_root,
            decision="blocked_quick_resource_limit",
            detail=str(error),
        )
        coverage.to_csv(PRIMARY / "holdout_options_coverage.csv", index=False)
    except OptionsDownloadError as error:
        manifest = blocked_manifest(
            decision="blocked_holdout_options_download_failure",
            detail=str(error),
            requests_planned=len(rows),
        )
    write_json(PRIMARY / "holdout_options_download_manifest.json", manifest)
    print(f"{manifest['overall_decision']}: {manifest['detail']}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
