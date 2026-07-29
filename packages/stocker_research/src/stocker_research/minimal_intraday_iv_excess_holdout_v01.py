"""Strict-resume controls for minimal intraday-H0 IV-excess holdout V0.1.

The scientific model, target, tail, bootstrap, null, and decision mechanics
remain in :mod:`minimal_intraday_iv_excess_holdout_v0`.  This module adds only
the fail-closed receipt overlay and pre-outcome authorization needed to resume
the previously blocked acquisition without reopening completed requests.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from stocker_research.eodhd_options_downloader_v0 import (
    OPTIONS_EOD_ENDPOINT,
    provider_eod_observation_date,
    stable_request_id,
)
from stocker_research.minimal_intraday_iv_excess_holdout_v0 import (
    ANNUAL_TRADING_MINUTES,
    HORIZONS,
    weighted_quantile,
)

MAXIMUM_ADDITIONAL_RECORDS: Final[int] = 150_000
MAXIMUM_CUMULATIVE_RECORDS: Final[int] = 500_000
MAXIMUM_ADDITIONAL_BYTES: Final[int] = 150_000_000
EXPECTED_V0_COMPLETE_RECEIPTS: Final[int] = 1_450
EXPECTED_FROZEN_REQUESTS: Final[int] = 1_700

SAFETY_FLAGS: Final[dict[str, object]] = {
    "research_only": True,
    "strict_resume": True,
    "frozen_holdout_validation": True,
    "holdout_start": "2025-09-01",
    "holdout_end": "2025-12-31",
    "training_end": "2024-12-31",
    "completed_receipts_reused": True,
    "complete_requests_redownloaded": False,
    "partial_cohort_model_allowed": False,
    "holdout_outcomes_opened_only_after_coverage_preflight": True,
    "previous_close_options_only": True,
    "minimal_options_plus_intraday_h0_model": True,
    "daily_stock_features_excluded": True,
    "route_competition_features_excluded": True,
    "route_state_features_excluded": True,
    "mismatch_features_excluded": True,
    "top_5_percent_threshold_frozen": True,
    "option_pnl_calculated": False,
    "intraday_option_quotes_used": False,
    "execution_enabled": False,
    "order_placement": "disabled",
    "broker_integration_required": False,
    "strategy_promotion": False,
    "production_runtime_modified": False,
    # Compatibility aliases retained for the frozen V0 scientific helpers.
    "prior_reference_period_not_used_for_tuning": True,
    "hand_built_mismatch_features_excluded": True,
    "top_5_percent_tail_frozen": True,
    "directional_outcomes_primary": False,
}

_OUTCOME_COLUMNS: Final[tuple[str, ...]] = (
    "entry_price",
    "close_5m",
    "close_10m",
    "close_15m",
    "close_30m",
)
_OUTCOME_PREFIXES: Final[tuple[str, ...]] = (
    "absolute_log_return_",
    "iv_sigma_",
    "iv_expected_absolute_",
    "iv_absolute_residual_",
    "movement_exceeds_prior_close_iv_",
)


class ResumeResourceLimitError(RuntimeError):
    """The single authorized V0.1 acquisition allowance was exceeded."""


@dataclass(frozen=True)
class ReceiptInventory:
    """Verified complete-receipt inventory over one or more cache roots."""

    audit: pd.DataFrame
    verified_request_ids: frozenset[str]
    receipt_paths: Mapping[str, Path]
    complete_receipts_found: int
    complete_receipts_reused: int
    corrupt_receipts: int
    corrupt_reasons: Mapping[str, str]


@dataclass(frozen=True)
class InterruptedRequest:
    """Audited identity and exclusion result for one unreceipted page."""

    request_id: str
    symbol: str
    holdout_session: str
    required_options_date: str
    original_complete_pages: int
    incomplete_page_identity: str
    incomplete_page_records: int
    incomplete_page_bytes: int
    incomplete_exact_date_records: int
    incomplete_extra_date_records: int
    incomplete_protected_date_records: int
    next_offset: int | None
    resume_method: str
    incomplete_page_admitted: bool


def assert_v01_safety_flags(value: Mapping[str, object]) -> None:
    """Require every strict-resume scientific and execution boundary flag."""

    mismatches = {
        key: {"expected": expected, "actual": value.get(key)}
        for key, expected in SAFETY_FLAGS.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"minimal holdout V0.1 safety flags differ: {mismatches}")


def _required_text(row: Mapping[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"planned request lacks {name}")
    return value


def _required_float(row: Mapping[str, object], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"planned request lacks numeric {name}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"planned request has non-finite {name}")
    return number


def request_parameters(
    row: Mapping[str, object],
    *,
    offset: int = 0,
    limit: int = 1_000,
) -> dict[str, object]:
    """Reconstruct the exact credential-free request parameters."""

    return {
        "page[offset]": offset,
        "page[limit]": limit,
        "compact": 0,
        "fmt": "json",
        "filter[underlying_symbol]": _required_text(row, "symbol"),
        "filter[tradetime_from]": _required_text(row, "required_options_date"),
        "filter[tradetime_to]": _required_text(row, "required_options_date"),
        "filter[strike_from]": _required_float(row, "strike_from"),
        "filter[strike_to]": _required_float(row, "strike_to"),
        "filter[exp_date_from]": _required_text(row, "expiration_from"),
        "filter[exp_date_to]": _required_text(row, "expiration_to"),
    }


def request_identity(row: Mapping[str, object]) -> str:
    """Return the immutable root identity for one frozen logical request."""

    return stable_request_id(OPTIONS_EOD_ENDPOINT, request_parameters(row))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_payload(content: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("raw response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("raw response is not an object")
    meta = payload.get("meta")
    data = payload.get("data")
    links = payload.get("links")
    if not isinstance(meta, dict) or not isinstance(data, list) or not isinstance(links, dict):
        raise ValueError("raw response lacks meta/data/links")
    fields = meta.get("fields")
    records: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            records.append(cast(dict[str, Any], item))
        elif isinstance(item, list) and isinstance(fields, list) and len(item) == len(fields):
            records.append(
                {
                    "type": "options-eod",
                    "attributes": dict(zip((str(field) for field in fields), item, strict=True)),
                }
            )
        else:
            raise ValueError("raw response row does not match meta.fields")
    return cast(dict[str, Any], payload), records


def _next_page_offset(links: Mapping[str, object]) -> int | None:
    value = links.get("next")
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("pagination next link is invalid")
    values = parse_qs(urlparse(value).query).get("page[offset]")
    if values is None or len(values) != 1:
        raise ValueError("pagination next link lacks one offset")
    try:
        result = int(values[0])
    except ValueError as error:
        raise ValueError("pagination next offset is invalid") from error
    if result < 0:
        raise ValueError("pagination next offset is negative")
    return result


def _manifest_request_identity(row: Mapping[str, object]) -> str:
    planned = {
        "symbol": row.get("underlying_symbol"),
        "required_options_date": row.get("trade_date_from"),
        "strike_from": row.get("strike_from"),
        "strike_to": row.get("strike_to"),
        "expiration_from": row.get("expiration_from"),
        "expiration_to": row.get("expiration_to"),
    }
    if row.get("trade_date_to") != row.get("trade_date_from"):
        raise ValueError("completed receipt is not one exact-date request")
    return stable_request_id(
        OPTIONS_EOD_ENDPOINT,
        request_parameters(
            planned,
            offset=int(cast(Any, row.get("offset"))),
            limit=int(cast(Any, row.get("limit"))),
        ),
    )


def _validate_receipt(
    planned: Mapping[str, object],
    *,
    receipt_path: Path,
    cache_root: Path,
    canonical_cache_path: Path,
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    expected_id = request_identity(planned)
    if receipt_path.stem != expected_id:
        raise ValueError("receipt filename differs from frozen request identity")
    try:
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("completed receipt is unreadable") from error
    rows_value = stored.get("manifest_rows") if isinstance(stored, dict) else None
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("completed receipt has no manifest pages")

    expected_offset = 0
    expected_total: int | None = None
    all_records: list[dict[str, Any]] = []
    response_hashes: list[str] = []
    response_bytes = 0
    raw_root = (cache_root / "raw").resolve()
    ordered = sorted(
        (cast(dict[str, Any], value) for value in rows_value),
        key=lambda value: int(value.get("offset", -1)),
    )
    for page_index, manifest in enumerate(ordered):
        if bool(manifest.get("superseded_by_split", False)):
            raise ValueError("completed logical receipt contains a superseded page")
        if int(manifest.get("response_status", 0)) != 200:
            raise ValueError("completed receipt contains a non-200 page")
        if int(manifest.get("offset", -1)) != expected_offset:
            raise ValueError("completed receipt pagination offsets are not contiguous")
        if manifest.get("underlying_symbol") != planned.get("symbol"):
            raise ValueError("receipt symbol differs from frozen request")
        if manifest.get("trade_date_from") != planned.get("required_options_date") or manifest.get(
            "trade_date_to"
        ) != planned.get("required_options_date"):
            raise ValueError("receipt observation date differs from frozen request")
        if _manifest_request_identity(manifest) != manifest.get("request_id"):
            raise ValueError("page request identity mismatch")

        raw_path = Path(str(manifest.get("cache_path", ""))).resolve()
        if raw_path.parent != raw_root or not raw_path.is_file():
            raise ValueError("receipt raw response is missing or outside its cache")
        content = raw_path.read_bytes()
        response_hash = hashlib.sha256(content).hexdigest()
        if response_hash != manifest.get("response_hash") or raw_path.stem != response_hash:
            raise ValueError("receipt raw response content identity mismatch")
        payload, records = _decode_payload(content)
        meta = cast(dict[str, Any], payload["meta"])
        links = cast(dict[str, Any], payload["links"])
        if int(meta.get("offset", -1)) != expected_offset:
            raise ValueError("provider offset differs from completed receipt")
        if int(meta.get("limit", -1)) != int(manifest.get("limit", -2)):
            raise ValueError("provider limit differs from completed receipt")
        if len(records) != int(manifest.get("record_count", -1)):
            raise ValueError("provider record count differs from completed receipt")
        total_value = meta.get("total")
        if total_value is not None:
            total = int(total_value)
            if expected_total is None:
                expected_total = total
            elif expected_total != total:
                raise ValueError("provider pagination total changed")
        all_records.extend(records)
        expected_offset += len(records)
        response_hashes.append(response_hash)
        response_bytes += len(content)
        next_offset = _next_page_offset(links)
        final = page_index == len(ordered) - 1
        if final:
            if next_offset is not None:
                raise ValueError("completed receipt still advertises a next page")
            if expected_total is not None and expected_total != expected_offset:
                raise ValueError("completed receipt does not reach provider total")
        elif next_offset != expected_offset:
            raise ValueError("completed receipt next offset is not contiguous")

    required = date.fromisoformat(_required_text(planned, "required_options_date"))
    exact = 0
    protected = 0
    for record in all_records:
        observed = provider_eod_observation_date(record)
        protected += int(observed >= date(2026, 1, 1))
        exact += int(observed == required and observed < date(2026, 1, 1))
    extra = len(all_records) - exact
    content_identity = hashlib.sha256(
        json.dumps(response_hashes, separators=(",", ":")).encode()
    ).hexdigest()
    audit: dict[str, object] = {
        "request_id": expected_id,
        "symbol": _required_text(planned, "symbol"),
        "holdout_session": _required_text(planned, "holdout_session"),
        "requested_observation_date": required.isoformat(),
        "pagination_complete": True,
        "page_count": len(ordered),
        "response_hashes": ";".join(response_hashes),
        "receipt_sha256": _sha256_file(receipt_path),
        "deterministic_content_identity": content_identity,
        "records_returned": len(all_records),
        "exact_date_record_count": exact,
        "extra_date_rejection_count": extra,
        "protected_date_rejection_count": protected,
        "response_bytes": response_bytes,
        "receipt_path": str(receipt_path.resolve()),
        "source_cache_root": str(cache_root.resolve()),
        "canonical_cache_path": str(canonical_cache_path.resolve()),
        "reused": True,
        "corrupt": False,
        "corruption_reason": "",
    }
    return audit, all_records


def inventory_complete_receipts(
    plan_rows: Sequence[Mapping[str, object]],
    *,
    cache_roots: Sequence[Path],
    canonical_cache_path: Path,
) -> ReceiptInventory:
    """Verify receipts and hashes without admitting unreceipted raw pages."""

    plan_by_id: dict[str, Mapping[str, object]] = {}
    for row in plan_rows:
        identity = request_identity(row)
        if identity in plan_by_id:
            raise ValueError("frozen request plan contains a duplicate logical identity")
        plan_by_id[identity] = row

    audit_rows: list[dict[str, object]] = []
    valid: dict[str, Path] = {}
    corrupt_reasons: dict[str, str] = {}
    found = 0
    for cache_root in cache_roots:
        completed = cache_root / "manifests" / "completed"
        if not completed.is_dir():
            continue
        for receipt_path in sorted(completed.glob("*.json")):
            found += 1
            planned = plan_by_id.get(receipt_path.stem)
            if planned is None:
                corrupt_reasons[receipt_path.stem] = "receipt_absent_from_frozen_plan"
                audit_rows.append(
                    {
                        "request_id": receipt_path.stem,
                        "receipt_path": str(receipt_path.resolve()),
                        "source_cache_root": str(cache_root.resolve()),
                        "reused": False,
                        "corrupt": True,
                        "corruption_reason": "receipt_absent_from_frozen_plan",
                    }
                )
                continue
            try:
                audit, _records = _validate_receipt(
                    planned,
                    receipt_path=receipt_path,
                    cache_root=cache_root,
                    canonical_cache_path=canonical_cache_path,
                )
            except (KeyError, TypeError, ValueError, OSError) as error:
                reason = f"{type(error).__name__}:{error}"
                corrupt_reasons[receipt_path.stem] = reason
                audit_rows.append(
                    {
                        "request_id": receipt_path.stem,
                        "symbol": planned.get("symbol"),
                        "holdout_session": planned.get("holdout_session"),
                        "requested_observation_date": planned.get("required_options_date"),
                        "receipt_path": str(receipt_path.resolve()),
                        "source_cache_root": str(cache_root.resolve()),
                        "canonical_cache_path": str(canonical_cache_path.resolve()),
                        "pagination_complete": False,
                        "reused": False,
                        "corrupt": True,
                        "corruption_reason": reason,
                    }
                )
                continue
            if receipt_path.stem in valid:
                raise ValueError("a verified completed request exists in more than one cache root")
            valid[receipt_path.stem] = receipt_path.resolve()
            audit_rows.append(audit)

    audit_frame = pd.DataFrame(audit_rows)
    if not audit_frame.empty:
        audit_frame = audit_frame.sort_values(
            ["symbol", "requested_observation_date", "request_id"],
            kind="mergesort",
            na_position="last",
        ).reset_index(drop=True)
    return ReceiptInventory(
        audit=audit_frame,
        verified_request_ids=frozenset(valid),
        receipt_paths=valid,
        complete_receipts_found=found,
        complete_receipts_reused=len(valid),
        corrupt_receipts=len(corrupt_reasons),
        corrupt_reasons=corrupt_reasons,
    )


def load_verified_receipt_records(
    planned: Mapping[str, object],
    *,
    receipt_path: Path,
    cache_root: Path,
    canonical_cache_path: Path,
) -> list[dict[str, Any]]:
    """Reload only rows referenced by one verified complete receipt."""

    _audit, records = _validate_receipt(
        planned,
        receipt_path=receipt_path,
        cache_root=cache_root,
        canonical_cache_path=canonical_cache_path,
    )
    return records


def remaining_resume_requests(
    plan_rows: Sequence[Mapping[str, object]],
    verified_request_ids: frozenset[str],
) -> list[dict[str, object]]:
    """Return only unverified requests in deterministic stock/date order."""

    remaining: dict[str, dict[str, object]] = {}
    for row in plan_rows:
        identity = request_identity(row)
        if identity not in verified_request_ids:
            remaining[identity] = dict(row)
    return sorted(
        remaining.values(),
        key=lambda row: (
            _required_text(row, "symbol"),
            _required_text(row, "required_options_date"),
            _required_text(row, "holdout_session"),
            request_identity(row),
        ),
    )


def _query_one(query: Mapping[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values is not None and len(values) == 1 else None


def identify_interrupted_request(
    raw_path: Path,
    missing_rows: Sequence[Mapping[str, object]],
) -> InterruptedRequest:
    """Map one unreceipted page and keep every row outside canonical materialization."""

    content = raw_path.read_bytes()
    payload, records = _decode_payload(content)
    links = cast(dict[str, Any], payload["links"])
    next_value = links.get("next")
    if not isinstance(next_value, str):
        raise ValueError("interrupted page lacks a deterministic next link")
    query = parse_qs(urlparse(next_value).query)

    def matches(row: Mapping[str, object]) -> bool:
        text_pairs = (
            ("filter[underlying_symbol]", "symbol"),
            ("filter[tradetime_from]", "required_options_date"),
            ("filter[tradetime_to]", "required_options_date"),
            ("filter[exp_date_from]", "expiration_from"),
            ("filter[exp_date_to]", "expiration_to"),
        )
        if any(
            _query_one(query, query_name) != _required_text(row, row_name)
            for query_name, row_name in text_pairs
        ):
            return False
        for query_name, row_name in (
            ("filter[strike_from]", "strike_from"),
            ("filter[strike_to]", "strike_to"),
        ):
            query_value = _query_one(query, query_name)
            if query_value is None or not math.isclose(
                float(query_value),
                _required_float(row, row_name),
                abs_tol=1e-12,
            ):
                return False
        return True

    candidates = [row for row in missing_rows if matches(row)]
    if len(candidates) != 1:
        raise ValueError(f"interrupted page maps to {len(candidates)} frozen logical requests")
    planned = candidates[0]
    required = date.fromisoformat(_required_text(planned, "required_options_date"))
    exact = 0
    protected = 0
    for record in records:
        observed = provider_eod_observation_date(record)
        protected += int(observed >= date(2026, 1, 1))
        exact += int(observed == required and observed < date(2026, 1, 1))
    next_offset = _next_page_offset(links)
    return InterruptedRequest(
        request_id=request_identity(planned),
        symbol=_required_text(planned, "symbol"),
        holdout_session=_required_text(planned, "holdout_session"),
        required_options_date=required.isoformat(),
        original_complete_pages=0,
        incomplete_page_identity=hashlib.sha256(content).hexdigest(),
        incomplete_page_records=len(records),
        incomplete_page_bytes=len(content),
        incomplete_exact_date_records=exact,
        incomplete_extra_date_records=len(records) - exact,
        incomplete_protected_date_records=protected,
        next_offset=next_offset,
        resume_method="redownload_logical_request_from_beginning",
        incomplete_page_admitted=False,
    )


def validate_additional_resource_usage(
    *,
    provider_records: int,
    raw_bytes: int,
    cumulative_records: int,
) -> None:
    """Fail closed at either the V0.1 or cumulative frozen acquisition ceiling."""

    if (
        provider_records < 0
        or raw_bytes < 0
        or cumulative_records < 0
        or provider_records > MAXIMUM_ADDITIONAL_RECORDS
        or raw_bytes > MAXIMUM_ADDITIONAL_BYTES
        or cumulative_records > MAXIMUM_CUMULATIVE_RECORDS
    ):
        raise ResumeResourceLimitError(
            "blocked_resume_resource_limit: frozen V0.1 acquisition allowance exceeded"
        )


def coverage_preflight(
    joined_features: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    *,
    planned_stock_sessions: int,
    planned_session_count: int,
    planned_stock_month_cells: int,
) -> dict[str, object]:
    """Evaluate pair/join support without accepting any movement-outcome column."""

    forbidden = sorted(
        column
        for column in joined_features.columns
        if column in _OUTCOME_COLUMNS
        or any(column.startswith(prefix) for prefix in _OUTCOME_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"coverage preflight received holdout outcome columns: {forbidden}")
    required = {"symbol", "session", "checkpoint", "row_weight"}
    missing = sorted(required.difference(joined_features.columns))
    if missing:
        raise ValueError(f"coverage preflight inputs missing: {missing}")
    if not {"symbol", "session"}.issubset(selected_pairs.columns):
        raise ValueError("selected-pair coverage inputs are incomplete")
    if joined_features.empty:
        raise ValueError("coverage preflight has no prospective joined rows")
    if planned_stock_sessions <= 0 or planned_session_count <= 0 or planned_stock_month_cells <= 0:
        raise ValueError("coverage preflight planned-population counts must be positive")

    features = joined_features.copy()
    features["_month"] = (
        pd.to_datetime(features["session"], errors="raise").dt.to_period("M").astype(str)
    )
    pairs = selected_pairs.loc[:, ["symbol", "session"]].drop_duplicates().copy()
    pairs["_month"] = pd.to_datetime(pairs["session"], errors="raise").dt.to_period("M").astype(str)
    represented_cells = int(pairs[["symbol", "_month"]].drop_duplicates().shape[0])
    cell_coverage = (
        represented_cells / planned_stock_month_cells if planned_stock_month_cells > 0 else math.nan
    )
    weights = pd.to_numeric(features["row_weight"], errors="raise")
    if not weights.gt(0.0).all() or not math.isfinite(float(weights.sum())):
        raise ValueError("coverage preflight weights must be finite and positive")
    total_weight = float(weights.sum())
    stock_weight = features.assign(_weight=weights).groupby("symbol")["_weight"].sum()
    month_weight = features.assign(_weight=weights).groupby("_month")["_weight"].sum()
    maximum_stock_share = float(stock_weight.max() / total_weight)
    maximum_month_share = float(month_weight.max() / total_weight)
    months = int(features["_month"].nunique())
    gates = {
        "expected_rows_at_least_5000": len(features) >= 5_000,
        "expected_sessions_at_least_60": features["session"].nunique() >= 60,
        "expected_stocks_at_least_15": features["symbol"].nunique() >= 15,
        "all_four_holdout_months": months == 4,
        "planned_stock_month_cells_at_least_70pct": cell_coverage >= 0.70,
        "maximum_stock_weight_share_at_most_0_12": maximum_stock_share <= 0.12,
        "maximum_month_weight_share_at_most_0_35": maximum_month_share <= 0.35,
    }
    return {
        **SAFETY_FLAGS,
        "outcome_columns_read": False,
        "total_holdout_stock_sessions": planned_stock_sessions,
        "planned_session_count": planned_session_count,
        "pair_selected_stock_sessions": int(
            features[["symbol", "session"]].drop_duplicates().shape[0]
        ),
        "valid_atm_pairs": int(len(pairs)),
        "expected_joined_rows": int(len(features)),
        "expected_session_count": int(features["session"].nunique()),
        "expected_stock_count": int(features["symbol"].nunique()),
        "expected_month_count": months,
        "planned_stock_month_cells": planned_stock_month_cells,
        "represented_stock_month_cells": represented_cells,
        "represented_stock_month_cell_rate": float(cell_coverage),
        "maximum_expected_stock_weight_share": maximum_stock_share,
        "maximum_expected_month_weight_share": maximum_month_share,
        "gates": gates,
        "passed": all(gates.values()),
    }


def authorize_outcome_access(
    preflight: Mapping[str, object],
    freeze_manifest: Mapping[str, object],
) -> None:
    """Authorize movement construction only after coverage and model freeze."""

    if preflight.get("passed") is not True or preflight.get("outcome_columns_read") is not False:
        raise ValueError("holdout outcome access denied: coverage preflight did not pass")
    if freeze_manifest.get("frozen") is not True:
        raise ValueError("holdout outcome access denied: model/threshold freeze is incomplete")
    if freeze_manifest.get("holdout_outcomes_read_before_freeze") is not False:
        raise ValueError("holdout outcome access denied: chronology/leakage flag failed")


def add_movement_outcomes_with_optional_30m(frame: pd.DataFrame) -> pd.DataFrame:
    """Build frozen movement outcomes while allowing an unavailable sixth bar."""

    required = {
        "entry_price",
        "atm_iv",
        "close_5m",
        "close_10m",
        "close_15m",
        "close_30m",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"movement construction inputs missing: {missing}")
    output = frame.copy()
    entry = pd.to_numeric(output["entry_price"], errors="raise").to_numpy(float)
    atm_iv = pd.to_numeric(output["atm_iv"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(entry).all()
        or bool((entry <= 0.0).any())
        or not np.isfinite(atm_iv).all()
        or bool((atm_iv <= 0.0).any())
    ):
        raise ValueError("entry prices and prior-close ATM IV must be finite and positive")
    for horizon in HORIZONS:
        close = pd.to_numeric(output[f"close_{horizon}m"], errors="coerce").to_numpy(float)
        available = np.isfinite(close) & (close > 0.0)
        if horizon != 30 and not bool(available.all()):
            raise ValueError(f"binding {horizon}-minute closes must be finite and positive")
        movement = np.full(len(output), np.nan, dtype=float)
        movement[available] = np.abs(np.log(close[available] / entry[available]))
        sigma = atm_iv * math.sqrt(horizon / ANNUAL_TRADING_MINUTES)
        expectation = sigma * math.sqrt(2.0 / math.pi)
        residual = np.full(len(output), np.nan, dtype=float)
        residual[available] = movement[available] - expectation[available]
        exceeds = np.full(len(output), np.nan, dtype=float)
        exceeds[available] = (movement[available] > expectation[available]).astype(float)
        output[f"absolute_log_return_{horizon}m"] = movement
        output[f"iv_sigma_{horizon}m"] = sigma
        output[f"iv_expected_absolute_{horizon}m"] = expectation
        output[f"iv_absolute_residual_{horizon}m"] = residual
        output[f"movement_exceeds_prior_close_iv_{horizon}m"] = (
            exceeds.astype(int) if horizon != 30 else exceeds
        )
    return output


def attach_movement_prices_with_optional_30m(
    panel: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """Attach causal entry/close prices and retain a missing optional sixth bar."""

    required_state = {"symbol", "session", "bar_ordinal", "open", "close"}
    missing = sorted(required_state.difference(states.columns))
    if missing:
        raise ValueError(f"movement state inputs missing: {missing}")
    output = panel.copy()
    state_index = states.set_index(["symbol", "session", "bar_ordinal"])
    if not state_index.index.is_unique:
        raise ValueError("movement state identity is not unique")
    for column, offset, price_column in (
        ("entry_price", 0, "open"),
        ("close_5m", 0, "close"),
        ("close_10m", 1, "close"),
        ("close_15m", 2, "close"),
        ("close_30m", 5, "close"),
    ):
        keys = pd.MultiIndex.from_arrays(
            [
                output["symbol"].astype(str),
                output["session"].astype(str),
                output["checkpoint"].astype(int) + offset,
            ],
            names=["symbol", "session", "bar_ordinal"],
        )
        output[column] = pd.to_numeric(
            state_index[price_column].reindex(keys),
            errors="coerce",
        ).to_numpy(float)
    primary = ["entry_price", "close_5m", "close_10m", "close_15m"]
    values = output[primary].to_numpy(float)
    if not np.isfinite(values).all() or not output[primary].gt(0.0).all(axis=None):
        raise ValueError("a binding 5/10/15-minute movement price is unavailable")
    invalid_30m = (
        ~np.isfinite(output["close_30m"].to_numpy(float)) | output["close_30m"].le(0.0).to_numpy()
    )
    output.loc[invalid_30m, "close_30m"] = np.nan
    return output


def movement_timing_metrics_with_optional_30m(
    frame: pd.DataFrame,
    *,
    model: str = "M1",
) -> pd.DataFrame:
    """Report all frozen horizons using available rows at 30 minutes."""

    if frame.empty:
        raise ValueError("movement timing requires a non-empty frozen tail")
    weights = pd.to_numeric(frame["row_weight"], errors="raise").to_numpy(float)
    if not np.isfinite(weights).all() or bool((weights <= 0.0).any()):
        raise ValueError("movement timing weights must be finite and positive")
    movement_matrix = np.column_stack(
        [
            pd.to_numeric(frame[f"absolute_log_return_{horizon}m"], errors="coerce").to_numpy(float)
            for horizon in HORIZONS
        ]
    )
    complete_excursion = np.isfinite(movement_matrix).all(axis=1)
    maximum_bucket = np.full(len(frame), -1, dtype=int)
    if bool(complete_excursion.any()):
        maximum_bucket[complete_excursion] = np.argmax(
            movement_matrix[complete_excursion],
            axis=1,
        )
    eventual = movement_matrix[:, HORIZONS.index(30)]
    rows: list[dict[str, object]] = []
    for index, horizon in enumerate(HORIZONS):
        residual = pd.to_numeric(
            frame[f"iv_absolute_residual_{horizon}m"], errors="coerce"
        ).to_numpy(float)
        exceeds = pd.to_numeric(
            frame[f"movement_exceeds_prior_close_iv_{horizon}m"], errors="coerce"
        ).to_numpy(float)
        movement = movement_matrix[:, index]
        available = np.isfinite(residual) & np.isfinite(exceeds) & np.isfinite(movement)
        if not bool(available.any()):
            raise ValueError(f"movement timing has no available {horizon}-minute rows")
        horizon_weights = weights[available]
        paired_30m = available & np.isfinite(eventual)
        eventual_denominator = float(np.sum(weights[paired_30m] * eventual[paired_30m]))
        excursion_weight = float(weights[complete_excursion].sum())
        rows.append(
            {
                "model": model,
                "horizon_minutes": horizon,
                "rows_available": int(available.sum()),
                "rows_with_30m": int(paired_30m.sum()),
                "mean_iv_residual": float(
                    np.sum(horizon_weights * residual[available]) / horizon_weights.sum()
                ),
                "median_iv_residual": weighted_quantile(
                    residual[available],
                    horizon_weights,
                    0.50,
                ),
                "exceed_iv_rate": float(
                    np.sum(horizon_weights * exceeds[available]) / horizon_weights.sum()
                ),
                "percent_eventual_30m_movement_realized": (
                    float(np.sum(weights[paired_30m] * movement[paired_30m]) / eventual_denominator)
                    if eventual_denominator > 0.0
                    else math.nan
                ),
                "maximum_absolute_excursion_bucket_share": (
                    float(
                        np.sum(
                            weights[complete_excursion]
                            * (maximum_bucket[complete_excursion] == index)
                        )
                        / excursion_weight
                    )
                    if excursion_weight > 0.0
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "EXPECTED_FROZEN_REQUESTS",
    "EXPECTED_V0_COMPLETE_RECEIPTS",
    "InterruptedRequest",
    "MAXIMUM_ADDITIONAL_BYTES",
    "MAXIMUM_ADDITIONAL_RECORDS",
    "MAXIMUM_CUMULATIVE_RECORDS",
    "ReceiptInventory",
    "ResumeResourceLimitError",
    "SAFETY_FLAGS",
    "add_movement_outcomes_with_optional_30m",
    "assert_v01_safety_flags",
    "authorize_outcome_access",
    "attach_movement_prices_with_optional_30m",
    "coverage_preflight",
    "identify_interrupted_request",
    "inventory_complete_receipts",
    "load_verified_receipt_records",
    "movement_timing_metrics_with_optional_30m",
    "remaining_resume_requests",
    "request_identity",
    "request_parameters",
    "validate_additional_resource_usage",
]
