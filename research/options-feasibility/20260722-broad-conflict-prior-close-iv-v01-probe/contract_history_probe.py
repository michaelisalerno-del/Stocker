#!/usr/bin/env python3
"""Bounded retrieval-only probe for exact-date EODHD contract histories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import parse_qs, urlparse

import pandas as pd

PROBE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROBE_DIR.parents[2]
V0_DIR = PROBE_DIR.parent / "20260722-broad-conflict-prior-close-iv-v0"
PRIMARY = PROBE_DIR / "artifacts" / "primary"
sys.path.insert(0, str(REPO_ROOT / "packages" / "stocker_research" / "src"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "stocker_data" / "src"))
sys.path.insert(0, str(V0_DIR))

from download_options import RequestsTransport, SetupRequester  # noqa: E402

from stocker_research.broad_conflict_options_iv_screen_v0 import (  # noqa: E402
    SAFETY_FLAGS,
    calculate_primary_option_features,
    select_primary_atm_pair,
)
from stocker_research.eodhd_options_downloader_v0 import (  # noqa: E402
    CanonicalRejection,
    DownloadConfig,
    EODHDOptionsDownloader,
    OptionsAuthenticationError,
    OptionsDownloadError,
    OptionsRequest,
    OptionsResourceLimitExceeded,
    OptionsSchemaError,
    canonicalize_response_records,
    redact_secrets,
    sha256_bytes,
    stable_request_id,
)

PROBE_SYMBOLS = ("AAL", "MSTR", "WULF")
MAX_REQUIRED_OPTIONS_DATE = date(2025, 8, 21)
CONTRACTS_ENDPOINT = "/mp/unicornbay/options/contracts"
OPTIONS_EOD_ENDPOINT = "/mp/unicornbay/options/eod"
PROBE_PAGE_LIMIT = 1000
MAX_PROBE_RAW_RECORDS = 50_000
MAX_PROBE_BYTES = 250_000_000
MAX_CONTRACTS_PER_TARGET = 5_000
MAX_UNIQUE_HISTORY_CONTRACTS = 72


class ProbeResourceLimitError(RuntimeError):
    """The fixed probe would exceed its explicit records or bytes ceiling."""


class JsonRequester(Protocol):
    """Credential-safe JSON request seam used by contract discovery."""

    def get_json(self, endpoint: str, *, params: dict[str, object], timeout: float) -> object: ...


class LiveJsonRequester(JsonRequester, Protocol):
    """Live requester state needed to persist discovery completion records."""

    http_requests_attempted: int
    manifest_rows: list[dict[str, object]]

    def account_cached_bytes(self, response_bytes: int) -> None: ...


class ResumableDiscoveryRequester:
    """Verify and replay completed contract-discovery responses without redownloading."""

    def __init__(self, live: LiveJsonRequester, *, completion_dir: Path, token: str) -> None:
        self.live = live
        self.completion_dir = completion_dir
        self.token = token
        self.logical_requests_completed = 0
        self.manifest_rows: list[dict[str, object]] = []

    @property
    def http_requests_attempted(self) -> int:
        return self.live.http_requests_attempted

    def _completion_path(self, endpoint: str, params: Mapping[str, object]) -> Path:
        return self.completion_dir / f"{stable_request_id(endpoint, params)}.json"

    def get_json(self, endpoint: str, *, params: dict[str, object], timeout: float) -> object:
        request_id = stable_request_id(endpoint, params)
        completion_path = self._completion_path(endpoint, params)
        if completion_path.is_file():
            try:
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                row = dict(completion["manifest_row"])
                content = Path(row["cache_path"]).read_bytes()
            except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
                raise OptionsSchemaError("contract discovery completion cache is invalid") from exc
            if row.get("request_id") != request_id or row.get("endpoint") != endpoint:
                raise OptionsSchemaError("contract discovery completion identity changed")
            if sha256_bytes(content) != row.get("response_hash"):
                raise OptionsSchemaError("contract discovery completion hash mismatch")
            if self.token.encode("utf-8") in content:
                raise OptionsAuthenticationError(
                    "contract discovery cache contains credential material"
                )
            self.live.account_cached_bytes(len(content))
            try:
                payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise OptionsSchemaError("contract discovery cache is not valid JSON") from exc
            self.manifest_rows.append({**row, "resumed_from_cache": True})
            self.logical_requests_completed += 1
            return payload
        before = len(self.live.manifest_rows)
        payload = self.live.get_json(endpoint, params=params, timeout=timeout)
        new_rows = self.live.manifest_rows[before:]
        if len(new_rows) != 1:
            raise OptionsSchemaError("live contract discovery did not record one response")
        row = {**new_rows[0], "resumed_from_cache": False}
        if row.get("request_id") != request_id:
            raise OptionsSchemaError("live contract discovery request identity changed")
        self.manifest_rows.append(row)
        self.logical_requests_completed += 1
        _atomic_json(completion_path, {"manifest_row": row})
        return payload


@dataclass(frozen=True)
class ProbeTarget:
    """One frozen stock/date/underlying-close target for the retrieval probe."""

    symbol: str
    signal_date: date
    required_options_date: date
    previous_close: float


@dataclass(frozen=True)
class ContractDescriptor:
    """Immutable fields used to identify one candidate option contract."""

    contract_id: str
    underlying_symbol: str
    expiration_date: date
    option_type: str
    strike: float


@dataclass(frozen=True)
class ContractDiscoveryResult:
    """Completely paginated immutable contract metadata for one target."""

    contracts: tuple[ContractDescriptor, ...]
    requests_completed: int


@dataclass(frozen=True)
class CandidateGroup:
    """All immutable call/put identities at one candidate expiry and strike."""

    expiration_date: date
    strike: float
    call_contract_ids: tuple[str, ...]
    put_contract_ids: tuple[str, ...]


@dataclass(frozen=True)
class ContractHistory:
    """One fully paginated raw contract history and its request identity."""

    contract_id: str
    request_id: str
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ProbePairResult:
    """Exact-date primary-pair feasibility outcome for one probe target."""

    available: bool
    reason: str
    selected_expiration_date: date | None
    selected_strike: float | None
    call_contract_id: str | None
    put_contract_id: str | None
    exact_records: tuple[dict[str, Any], ...]
    histories_requested: tuple[str, ...]
    canonical_rejections: int


@dataclass(frozen=True)
class ExactObservationSelection:
    """Canonical history rows selected at one exact prior-close date."""

    records: tuple[dict[str, Any], ...]
    available_observation_dates: tuple[date, ...]
    rejections: tuple[CanonicalRejection, ...]


def _is_true_literal(value: str) -> bool:
    return value.strip().casefold() == "true"


def _manifest_integer(row: Mapping[str, object], field: str) -> int:
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise OptionsSchemaError(f"manifest field is not an integer: {field}")
    return int(value)


def build_probe_targets(price_audit: Path) -> tuple[ProbeTarget, ...]:
    """Freeze three symbols at the first, midpoint, and last shared option dates."""

    by_key: dict[tuple[str, date], ProbeTarget] = {}
    invalid_by_key: dict[tuple[str, date], str] = {}
    dates_by_symbol: dict[str, set[date]] = {symbol: set() for symbol in PROBE_SYMBOLS}
    with price_audit.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            symbol = row["symbol"].upper()
            if symbol not in dates_by_symbol:
                continue
            required_date = date.fromisoformat(row["required_options_date"])
            signal_date = date.fromisoformat(row["signal_date"])
            previous_close = float(row["previous_close_underlying_price"])
            key = (symbol, required_date)
            if not _is_true_literal(row["source_available"]):
                invalid_by_key[key] = "underlying_close_unavailable"
            if _is_true_literal(row["split_boundary_ambiguous"]):
                invalid_by_key[key] = "split_boundary_ambiguous"
            if not math.isfinite(previous_close) or previous_close <= 0.0:
                raise ValueError(f"probe underlying close invalid: {symbol} {required_date}")
            if required_date >= signal_date or required_date > MAX_REQUIRED_OPTIONS_DATE:
                raise ValueError(f"probe chronology or protected boundary invalid: {symbol}")
            target = ProbeTarget(symbol, signal_date, required_date, previous_close)
            if key in by_key and by_key[key] != target:
                raise ValueError(f"conflicting probe target: {symbol} {required_date}")
            by_key[key] = target
            dates_by_symbol[symbol].add(required_date)
    shared_dates = sorted(set.intersection(*(dates_by_symbol[s] for s in PROBE_SYMBOLS)))
    if len(shared_dates) < 3:
        raise ValueError("probe requires at least three shared prior-close dates")
    selected_dates = (shared_dates[0], shared_dates[len(shared_dates) // 2], shared_dates[-1])
    selected_keys = tuple((symbol, day) for day in selected_dates for symbol in PROBE_SYMBOLS)
    selected_invalid = {key: invalid_by_key[key] for key in selected_keys if key in invalid_by_key}
    if selected_invalid:
        raise ValueError(f"selected probe targets invalid: {selected_invalid}")
    return tuple(by_key[key] for key in selected_keys)


def _contract_descriptor(item: object, target: ProbeTarget) -> ContractDescriptor:
    if not isinstance(item, Mapping):
        raise ValueError("contract discovery row is not an object")
    attributes = item.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("contract discovery row lacks attributes")
    contract_id = attributes.get("contract")
    underlying = attributes.get("underlying_symbol")
    option_type = attributes.get("type")
    if not isinstance(contract_id, str) or not contract_id:
        raise ValueError("contract discovery row lacks stable contract identity")
    if not isinstance(underlying, str) or underlying.upper() != target.symbol:
        raise ValueError("contract discovery underlying mismatch")
    if not isinstance(option_type, str) or option_type.casefold() not in {"call", "put"}:
        raise ValueError("contract discovery option type invalid")
    try:
        expiration = date.fromisoformat(str(attributes.get("exp_date"))[:10])
        strike = float(attributes["strike"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("contract discovery expiry or strike invalid") from exc
    minimum_expiry = target.required_options_date + timedelta(days=7)
    maximum_expiry = target.required_options_date + timedelta(days=45)
    if not minimum_expiry <= expiration <= maximum_expiry:
        raise ValueError("contract discovery expiry escaped frozen window")
    if (
        not math.isfinite(strike)
        or not 0.70 * target.previous_close <= strike <= 1.30 * target.previous_close
    ):
        raise ValueError("contract discovery strike escaped frozen buffer")
    return ContractDescriptor(
        contract_id=contract_id,
        underlying_symbol=underlying.upper(),
        expiration_date=expiration,
        option_type=option_type.casefold(),
        strike=strike,
    )


def discover_contracts(
    requester: JsonRequester,
    *,
    target: ProbeTarget,
    page_limit: int,
    maximum_records: int,
    timeout: float = 30.0,
) -> ContractDiscoveryResult:
    """Retrieve every immutable contract page inside the fixed target window."""

    if not 1 <= page_limit <= 1000 or maximum_records < 1:
        raise ValueError("invalid contract discovery bounds")
    offset = 0
    expected_total: int | None = None
    pagination_has_total: bool | None = None
    requests_completed = 0
    contracts: list[ContractDescriptor] = []
    while True:
        remaining = maximum_records - len(contracts)
        if remaining <= 0:
            raise ProbeResourceLimitError("blocked_options_download_resource_limit")
        request_limit = min(page_limit, remaining)
        params: dict[str, object] = {
            "filter[underlying_symbol]": target.symbol,
            "filter[exp_date_from]": (target.required_options_date + timedelta(days=7)).isoformat(),
            "filter[exp_date_to]": (target.required_options_date + timedelta(days=45)).isoformat(),
            "filter[strike_from]": 0.70 * target.previous_close,
            "filter[strike_to]": 1.30 * target.previous_close,
            "sort": "exp_date",
            "page[offset]": offset,
            "page[limit]": request_limit,
            "fields[options-contracts]": "contract,underlying_symbol,exp_date,type,strike",
            "fmt": "json",
        }
        payload = requester.get_json(CONTRACTS_ENDPOINT, params=params, timeout=timeout)
        requests_completed += 1
        if not isinstance(payload, Mapping):
            raise ValueError("contract discovery response is not an object")
        meta = payload.get("meta")
        data = payload.get("data")
        links = payload.get("links")
        if (
            not isinstance(meta, Mapping)
            or not isinstance(data, list)
            or not isinstance(links, Mapping)
        ):
            raise ValueError("contract discovery response lacks meta/data/links")
        try:
            page_offset = int(meta["offset"])
            returned_limit = int(meta["limit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("contract discovery pagination invalid") from exc
        has_total = meta.get("total") is not None
        if pagination_has_total is None:
            pagination_has_total = has_total
        elif pagination_has_total != has_total:
            raise ValueError("contract discovery pagination mode changed")
        total: int | None = None
        if has_total:
            try:
                total = int(meta["total"])
            except (TypeError, ValueError) as exc:
                raise ValueError("contract discovery total invalid") from exc
        if page_offset != offset or returned_limit < 1 or (total is not None and total < 0):
            raise ValueError("contract discovery pagination disagrees with request")
        if total is not None and (total > maximum_records or total > 10_000):
            raise ProbeResourceLimitError("blocked_options_download_resource_limit")
        if total is not None and expected_total is None:
            expected_total = total
        elif total is not None and total != expected_total:
            raise ValueError("contract discovery total changed during pagination")
        page = [_contract_descriptor(item, target) for item in data]
        if len(page) > request_limit:
            raise ValueError("contract discovery returned more than requested")
        contracts.extend(page)
        consumed = offset + len(page)
        next_value = links.get("next")
        next_offset: int | None = None
        if next_value not in {None, ""}:
            if not isinstance(next_value, str):
                raise ValueError("contract discovery next link invalid")
            values = parse_qs(urlparse(next_value).query).get("page[offset]")
            if values is None or len(values) != 1:
                raise ValueError("contract discovery next link lacks one offset")
            try:
                next_offset = int(values[0])
            except ValueError as exc:
                raise ValueError("contract discovery next offset invalid") from exc
        if total is not None and consumed == total:
            if next_offset is not None:
                raise ValueError("contract discovery final page advertises next")
            break
        if total is None and next_offset is None:
            break
        if (total is not None and consumed > total) or not page or next_offset is None:
            raise ValueError("contract discovery silently truncated")
        if next_offset != consumed:
            raise ValueError("contract discovery next offset is not contiguous")
        if consumed > 10_000:
            raise ProbeResourceLimitError("blocked_options_download_resource_limit")
        offset = consumed
    unique = {contract.contract_id: contract for contract in contracts}
    if len(unique) != len(contracts):
        raise ValueError("contract discovery returned duplicate identities")
    return ContractDiscoveryResult(tuple(contracts), requests_completed)


def candidate_groups(
    contracts: Sequence[ContractDescriptor], *, target: ProbeTarget
) -> tuple[CandidateGroup, ...]:
    """Order common-strike identities by the frozen expiry/ATM priority."""

    grouped: dict[tuple[date, float], dict[str, list[str]]] = {}
    for contract in contracts:
        if contract.underlying_symbol != target.symbol:
            raise ValueError("candidate contract underlying mismatch")
        sides = grouped.setdefault(
            (contract.expiration_date, contract.strike), {"call": [], "put": []}
        )
        sides[contract.option_type].append(contract.contract_id)
    output = [
        CandidateGroup(expiry, strike, tuple(sorted(sides["call"])), tuple(sorted(sides["put"])))
        for (expiry, strike), sides in grouped.items()
        if sides["call"] and sides["put"]
    ]
    return tuple(
        sorted(
            output,
            key=lambda group: (
                group.expiration_date,
                abs(math.log(group.strike / target.previous_close)),
                group.strike,
                group.call_contract_ids,
                group.put_contract_ids,
            ),
        )
    )


def select_probe_pair(
    *,
    target: ProbeTarget,
    contracts: Sequence[ContractDescriptor],
    load_history: Callable[[str], ContractHistory],
) -> ProbePairResult:
    """Progressively retrieve histories until the frozen nearest exact pair resolves."""

    requested: list[str] = []
    rejection_count = 0
    groups = candidate_groups(contracts, target=target)
    for expiration in sorted({group.expiration_date for group in groups}):
        expiry_groups = [group for group in groups if group.expiration_date == expiration]
        position = 0
        while position < len(expiry_groups):
            distance = abs(math.log(expiry_groups[position].strike / target.previous_close))
            tied_groups: list[CandidateGroup] = []
            while position < len(expiry_groups):
                candidate = expiry_groups[position]
                candidate_distance = abs(math.log(candidate.strike / target.previous_close))
                if candidate_distance != distance:
                    break
                tied_groups.append(candidate)
                position += 1
            exact_rows: list[dict[str, Any]] = []
            complete_strikes: set[float] = set()
            for group in tied_groups:
                group_rows: list[dict[str, Any]] = []
                for contract_id in (*group.call_contract_ids, *group.put_contract_ids):
                    history = load_history(contract_id)
                    if history.contract_id != contract_id:
                        raise ValueError("history loader returned a different contract identity")
                    requested.append(contract_id)
                    exact = exact_contract_observations(
                        history.records,
                        required_date=target.required_options_date,
                        request_id=history.request_id,
                    )
                    rejection_count += len(exact.rejections)
                    if any(row["contract_id"] != contract_id for row in exact.records):
                        raise ValueError("contract history contains another contract identity")
                    group_rows.extend(exact.records)
                exact_sides = {str(row["option_type"]) for row in group_rows}
                if {"call", "put"}.issubset(exact_sides):
                    complete_strikes.add(group.strike)
                    exact_rows.extend(group_rows)
            if not complete_strikes:
                continue
            selection = select_primary_atm_pair(
                pd.DataFrame(exact_rows), previous_close=target.previous_close
            )
            return ProbePairResult(
                available=selection.available,
                reason=selection.reason,
                selected_expiration_date=selection.expiration_date,
                selected_strike=selection.strike,
                call_contract_id=selection.call_contract_id,
                put_contract_id=selection.put_contract_id,
                exact_records=tuple(exact_rows),
                histories_requested=tuple(requested),
                canonical_rejections=rejection_count,
            )
    return ProbePairResult(
        available=False,
        reason="no_exact_previous_session_common_strike_pair",
        selected_expiration_date=None,
        selected_strike=None,
        call_contract_id=None,
        put_contract_id=None,
        exact_records=(),
        histories_requested=tuple(requested),
        canonical_rejections=rejection_count,
    )


def exact_contract_observations(
    records: Sequence[Mapping[str, Any]], *, required_date: date, request_id: str
) -> ExactObservationSelection:
    """Canonicalize a contract history and keep only its exact EOD observation date."""

    canonical = canonicalize_response_records(
        records,
        request_id=request_id,
        provider_schema_version="openapi-2.0.0-contract-history",
    )
    dates = tuple(sorted({row["trade_date"] for row in canonical.records}))
    exact = tuple(row for row in canonical.records if row["trade_date"] == required_date)
    return ExactObservationSelection(
        records=exact,
        available_observation_dates=dates,
        rejections=tuple(canonical.rejections),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(rows[0]) if rows else ["status"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _cache_root(argument: Path | None) -> Path:
    if argument is not None:
        base = argument
    elif configured := os.environ.get("EODHD_OPTIONS_DATA_DIR"):
        base = Path(configured)
    else:
        base = REPO_ROOT / "data" / "vendor" / "eodhd" / "options"
    return base / "contract-history-probe-v01"


def _probe_plan(targets: Sequence[ProbeTarget], cache_root: Path) -> dict[str, Any]:
    return {
        **SAFETY_FLAGS,
        "experiment": "Broad-Conflict Prior-Close IV Contract-History Probe V0.1",
        "purpose": "retrieval_feasibility_only",
        "primary_v0_decision_unchanged": "blocked_historical_options_date_unavailable",
        "symbols": list(PROBE_SYMBOLS),
        "required_options_dates": sorted(
            {target.required_options_date.isoformat() for target in targets}
        ),
        "stock_dates": len(targets),
        "contract_discovery_endpoint": CONTRACTS_ENDPOINT,
        "contract_history_endpoint": OPTIONS_EOD_ENDPOINT,
        "contract_history_filter": "filter[contract]",
        "observation_date_identity": "resource_id_suffix_verified_by_bid_date_and_ask_date",
        "contract_discovery_minimum_logical_requests": len(targets),
        "maximum_unique_contract_histories": MAX_UNIQUE_HISTORY_CONTRACTS,
        "maximum_raw_provider_records": MAX_PROBE_RAW_RECORDS,
        "maximum_download_bytes": MAX_PROBE_BYTES,
        "page_limit": PROBE_PAGE_LIMIT,
        "processes": 1,
        "n_jobs": 1,
        "gpu": False,
        "cache_root": str(cache_root.resolve()),
        "raw_vendor_data_tracked": False,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _write_blocked_probe(
    *, output: Path, plan: Mapping[str, Any], status: str, reason: str
) -> None:
    manifest: dict[str, Any] = {
        **SAFETY_FLAGS,
        "probe_status": "blocked",
        "status": status,
        "reason": reason,
        "pagination_complete": False,
        "credential_exposures": 0,
        "raw_records": 0,
        "download_bytes": 0,
        "manifest_rows": [],
    }
    _atomic_json(output / "contract_history_probe_plan.json", plan)
    _atomic_json(output / "contract_history_probe_manifest.json", manifest)
    _atomic_csv(output / "contract_history_probe_results.csv", [{"status": status}])
    report = (
        "# EODHD contract-history feasibility probe\n\n"
        f"Status: `{status}`. {reason}\n\n"
        "The completed V0 decision is unchanged. No movement model, option P&L, executable "
        "fill, strategy result, or trading-utility claim was produced.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")


def run_live_probe(
    *,
    token: str,
    output: Path,
    cache_root: Path,
    price_audit: Path,
    requests_per_minute: int,
) -> dict[str, Any]:
    """Run the fixed nine-stock-date retrieval probe and write aggregate-only artifacts."""

    targets = build_probe_targets(price_audit)
    plan = _probe_plan(targets, cache_root)
    _atomic_json(output / "contract_history_probe_plan.json", plan)
    live_requester = SetupRequester(
        RequestsTransport(),
        token=token,
        requests_per_minute=requests_per_minute,
        cache_dir=cache_root / "raw" / "contract-discovery",
        maximum_download_bytes=MAX_PROBE_BYTES,
    )
    requester = ResumableDiscoveryRequester(
        live_requester,
        completion_dir=cache_root / "manifests" / "contract-discovery" / "completed",
        token=token,
    )
    discoveries: dict[tuple[str, date], ContractDiscoveryResult] = {}
    discovery_records = 0
    for target in targets:
        remaining = MAX_PROBE_RAW_RECORDS - discovery_records
        result = discover_contracts(
            requester,
            target=target,
            page_limit=PROBE_PAGE_LIMIT,
            maximum_records=min(MAX_CONTRACTS_PER_TARGET, remaining),
        )
        discoveries[(target.symbol, target.required_options_date)] = result
        discovery_records += len(result.contracts)
        discovery_bytes = sum(
            _manifest_integer(row, "response_bytes") for row in requester.manifest_rows
        )
        if discovery_records > MAX_PROBE_RAW_RECORDS or discovery_bytes > MAX_PROBE_BYTES:
            raise ProbeResourceLimitError("blocked_options_download_resource_limit")

    discovery_bytes = sum(
        _manifest_integer(row, "response_bytes") for row in requester.manifest_rows
    )
    history_record_budget = MAX_PROBE_RAW_RECORDS - discovery_records
    history_byte_budget = MAX_PROBE_BYTES - discovery_bytes
    if history_record_budget < 1 or history_byte_budget < 1:
        raise ProbeResourceLimitError("blocked_options_download_resource_limit")
    downloader = EODHDOptionsDownloader(
        DownloadConfig(
            token=token,
            data_dir=cache_root / "contract-histories",
            page_limit=PROBE_PAGE_LIMIT,
            requests_per_minute=requests_per_minute,
            maximum_raw_records=history_record_budget,
            maximum_download_bytes=history_byte_budget,
        ),
        transport=RequestsTransport(),
    )
    descriptors: dict[str, ContractDescriptor] = {}
    for discovery in discoveries.values():
        for descriptor in discovery.contracts:
            existing = descriptors.get(descriptor.contract_id)
            if existing is not None and existing != descriptor:
                raise ValueError("contract identity mapped to conflicting immutable metadata")
            descriptors[descriptor.contract_id] = descriptor
    history_cache: dict[str, ContractHistory] = {}
    history_manifest: list[dict[str, Any]] = []

    def load_history(contract_id: str) -> ContractHistory:
        if contract_id in history_cache:
            return history_cache[contract_id]
        if len(history_cache) >= MAX_UNIQUE_HISTORY_CONTRACTS:
            raise ProbeResourceLimitError("blocked_options_download_resource_limit")
        descriptor = descriptors[contract_id]
        request = OptionsRequest(
            underlying_symbol=descriptor.underlying_symbol,
            contract_id=contract_id,
            compact=False,
        )
        result = downloader.download(request)
        for row in result.manifest_rows:
            history_manifest.append(
                {
                    **row.to_dict(),
                    "endpoint": OPTIONS_EOD_ENDPOINT,
                    "contract_id": contract_id,
                }
            )
        request_id = stable_request_id(
            request.endpoint,
            request.parameters(offset=0, limit=PROBE_PAGE_LIMIT),
        )
        history = ContractHistory(contract_id, request_id, tuple(result.records))
        history_cache[contract_id] = history
        return history

    result_rows: list[dict[str, Any]] = []
    for target in targets:
        discovery = discoveries[(target.symbol, target.required_options_date)]
        pair = select_probe_pair(
            target=target,
            contracts=discovery.contracts,
            load_history=load_history,
        )
        features: dict[str, Any] = {}
        if pair.available:
            selection = select_primary_atm_pair(
                pd.DataFrame(pair.exact_records), previous_close=target.previous_close
            )
            features = calculate_primary_option_features(
                selection, previous_close=target.previous_close
            )
        result_rows.append(
            {
                "symbol": target.symbol,
                "signal_date": target.signal_date.isoformat(),
                "required_options_date": target.required_options_date.isoformat(),
                "previous_close_underlying_price": target.previous_close,
                "contracts_discovered": len(discovery.contracts),
                "common_strike_candidate_groups": len(
                    candidate_groups(discovery.contracts, target=target)
                ),
                "history_contracts_requested": len(pair.histories_requested),
                "history_contract_ids_requested": ";".join(pair.histories_requested),
                "exact_observation_rows": len(pair.exact_records),
                "pair_available": pair.available,
                "reason": pair.reason,
                "selected_expiry": pair.selected_expiration_date,
                "selected_strike": pair.selected_strike,
                "call_contract_id": pair.call_contract_id,
                "put_contract_id": pair.put_contract_id,
                "front_dte": features.get("front_dte"),
                "atm_iv": features.get("atm_iv"),
                "call_relative_spread": features.get("call_relative_spread"),
                "put_relative_spread": features.get("put_relative_spread"),
                "combined_open_interest": features.get("combined_open_interest"),
                "canonical_rejections": pair.canonical_rejections,
            }
        )

    history_records = sum(int(row["record_count"]) for row in history_manifest)
    history_bytes = sum(Path(str(row["cache_path"])).stat().st_size for row in history_manifest)
    total_records = discovery_records + history_records
    total_bytes = discovery_bytes + history_bytes
    if total_records > MAX_PROBE_RAW_RECORDS or total_bytes > MAX_PROBE_BYTES:
        raise ProbeResourceLimitError("blocked_options_download_resource_limit")
    valid_pairs = sum(bool(row["pair_available"]) for row in result_rows)
    exact_date_targets = sum(int(row["exact_observation_rows"]) > 0 for row in result_rows)
    setup_manifest = [dict(row) for row in requester.manifest_rows]
    setup_attempts_recorded = sum(
        _manifest_integer(cast(Mapping[str, object], row), "attempts") for row in setup_manifest
    )
    all_manifest_rows = [*setup_manifest, *history_manifest]
    serialized = json.dumps(all_manifest_rows, sort_keys=True, default=str)
    if token in serialized:
        raise OptionsAuthenticationError("credential material reached probe manifest")
    manifest = {
        **SAFETY_FLAGS,
        "probe_status": "supported" if valid_pairs else "insufficient_support",
        "status": "contract_history_probe_complete",
        "symbols_requested": list(PROBE_SYMBOLS),
        "stock_dates_requested": len(targets),
        "contract_discovery_logical_requests": requester.logical_requests_completed,
        "contract_discovery_http_attempts": setup_attempts_recorded,
        "contract_discovery_http_attempts_current_run": requester.http_requests_attempted,
        "unique_contract_histories_requested": len(history_cache),
        "history_pages": len(history_manifest),
        "provider_http_attempts_recorded": setup_attempts_recorded
        + sum(int(row["attempts"]) for row in history_manifest),
        "raw_contract_records": discovery_records,
        "raw_history_records": history_records,
        "raw_records": total_records,
        "download_bytes": total_bytes,
        "pagination_complete": True,
        "unexplained_truncations": 0,
        "credential_exposures": 0,
        "exact_date_targets": exact_date_targets,
        "valid_primary_pairs": valid_pairs,
        "manifest_rows": all_manifest_rows,
    }
    _atomic_json(output / "contract_history_probe_manifest.json", manifest)
    _atomic_csv(output / "contract_history_probe_results.csv", result_rows)
    recorded_http_attempts = setup_attempts_recorded + sum(
        int(row["attempts"]) for row in history_manifest
    )
    report = (
        "# EODHD contract-history feasibility probe\n\n"
        f"The bounded probe completed for {len(PROBE_SYMBOLS)} frozen symbols and "
        f"{len(targets)} stock-dates. Exact-date option observations were recovered for "
        f"{exact_date_targets} targets and {valid_pairs} passed the frozen primary-pair rules. "
        f"It retrieved {total_records:,} provider records ({total_bytes:,} bytes) through "
        f"{recorded_http_attempts} "
        "recorded HTTP attempts with complete next-link pagination.\n\n"
        "This is retrieval feasibility only. The completed V0 decision remains unchanged; no "
        "movement model, intraday option fill, option P&L, executable return, strategy result, "
        "or trading-utility claim was produced.\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PRIMARY)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--price-audit",
        type=Path,
        default=V0_DIR / "artifacts" / "primary" / "option_underlying_price_audit.csv",
    )
    args = parser.parse_args()
    targets = build_probe_targets(args.price_audit)
    cache_root = _cache_root(args.cache_dir)
    plan = _probe_plan(targets, cache_root)
    token = os.environ.get("EODHD_API_TOKEN", "")
    if not token:
        _write_blocked_probe(
            output=args.output,
            plan=plan,
            status="blocked_missing_eodhd_api_token",
            reason="EODHD_API_TOKEN was unavailable; no public demo token was substituted.",
        )
        return 2
    try:
        requests_per_minute = int(os.environ.get("EODHD_OPTIONS_REQUESTS_PER_MINUTE", "30"))
        run_live_probe(
            token=token,
            output=args.output,
            cache_root=cache_root,
            price_audit=args.price_audit,
            requests_per_minute=requests_per_minute,
        )
    except (ProbeResourceLimitError, OptionsResourceLimitExceeded):
        _write_blocked_probe(
            output=args.output,
            plan=plan,
            status="blocked_options_download_resource_limit",
            reason="The fixed probe subcap was reached; scope was not broadened.",
        )
        return 2
    except OptionsAuthenticationError:
        _write_blocked_probe(
            output=args.output,
            plan=plan,
            status="blocked_options_download_incomplete",
            reason="EODHD authentication or entitlement failed.",
        )
        return 2
    except (OptionsSchemaError, ValueError) as error:
        _write_blocked_probe(
            output=args.output,
            plan=plan,
            status="blocked_options_data_integrity_failure",
            reason=str(error),
        )
        return 2
    except OptionsDownloadError as error:
        _write_blocked_probe(
            output=args.output,
            plan=plan,
            status="blocked_options_download_incomplete",
            reason=str(error),
        )
        return 2
    except RuntimeError as error:
        _write_blocked_probe(
            output=args.output,
            plan=plan,
            status="blocked_options_download_incomplete",
            reason=redact_secrets(str(error), secrets=(token,)),
        )
        return 2
    return 0


__all__ = [
    "ExactObservationSelection",
    "ContractDescriptor",
    "ContractDiscoveryResult",
    "ContractHistory",
    "CandidateGroup",
    "ProbeTarget",
    "ProbeResourceLimitError",
    "ProbePairResult",
    "build_probe_targets",
    "candidate_groups",
    "discover_contracts",
    "exact_contract_observations",
    "select_probe_pair",
]


if __name__ == "__main__":
    raise SystemExit(main())
