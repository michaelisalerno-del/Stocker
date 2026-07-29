"""Bounded, credential-safe EODHD historical options download helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path
from typing import Any, Final, Protocol, Self, cast
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

SECRET_PARAMETER_NAMES: Final[frozenset[str]] = frozenset({"api_token", "token", "authorization"})
_TOKEN_QUERY = re.compile(r"([?&]api_token=)[^&\s]+", flags=re.IGNORECASE)
OPTIONS_EOD_ENDPOINT: Final[str] = "/mp/unicornbay/options/eod"
TRANSIENT_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})
NEW_YORK: Final[ZoneInfo] = ZoneInfo("America/New_York")
CANONICAL_OPTION_COLUMNS: Final[tuple[str, ...]] = (
    "provider",
    "provider_schema_version",
    "request_id",
    "underlying_symbol",
    "contract_id",
    "option_type",
    "expiration_date",
    "strike",
    "trade_date",
    "trade_timestamp",
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
    "raw_record_hash",
)


class ResponseLike(Protocol):
    """Small HTTP response surface used at the external boundary."""

    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def json(self) -> object: ...


class TransportLike(Protocol):
    """Synchronous GET transport accepted by the bounded downloader."""

    def get(self, url: str, *, params: dict[str, object], timeout: float) -> ResponseLike: ...


class _MaterializedResponse:
    def __init__(self, response: ResponseLike, content: bytes) -> None:
        self.status_code = response.status_code
        self.headers = response.headers
        self.content = content

    def json(self) -> object:
        return json.loads(self.content)


class OptionsDownloadError(RuntimeError):
    """Base class for fail-closed options download errors."""

    def __init__(self, message: str, *, manifest_rows: Sequence[RequestManifestRow] = ()) -> None:
        super().__init__(message)
        self.manifest_rows = list(manifest_rows)


class OptionsAuthenticationError(OptionsDownloadError):
    """Permanent authentication or entitlement failure."""


class OptionsSchemaError(OptionsDownloadError):
    """Permanent response-schema or pagination failure."""


class OffsetLimitExceeded(OptionsDownloadError):
    """A request would exceed the provider's documented offset ceiling."""

    def __init__(self, message: str, *, manifest_rows: Sequence[RequestManifestRow] = ()) -> None:
        super().__init__(message)
        self.manifest_rows = list(manifest_rows)


class OptionsResourceLimitExceeded(OptionsDownloadError):
    """The frozen raw-record or byte ceiling would be exceeded."""

    def __init__(self, message: str, *, manifest_rows: Sequence[RequestManifestRow] = ()) -> None:
        super().__init__(message)
        self.manifest_rows = list(manifest_rows)


@dataclass(frozen=True)
class OptionsRequest:
    """One deterministic, bounded historical options request chunk."""

    underlying_symbol: str
    contract_id: str | None = None
    trade_date_from: date | None = None
    trade_date_to: date | None = None
    strike_from: float | None = None
    strike_to: float | None = None
    expiration_from: date | None = None
    expiration_to: date | None = None
    fields: tuple[str, ...] = ()
    compact: bool = True
    endpoint: str = OPTIONS_EOD_ENDPOINT

    def __post_init__(self) -> None:
        if not self.underlying_symbol.strip():
            raise ValueError("underlying symbol is required")
        if self.contract_id is not None:
            if not self.contract_id.strip():
                raise ValueError("contract identity cannot be empty")
            if self.compact:
                raise ValueError("contract-history request requires compact=False")
            chain_filters = (
                self.trade_date_from,
                self.trade_date_to,
                self.strike_from,
                self.strike_to,
                self.expiration_from,
                self.expiration_to,
            )
            if any(value is not None for value in chain_filters):
                raise ValueError("contract-history request cannot include chain filters")

    def replace(self, **changes: Any) -> Self:
        """Return a copy with explicit field changes."""

        return replace(self, **changes)

    def parameters(self, *, offset: int, limit: int) -> dict[str, object]:
        """Return provider parameters without authentication."""

        params: dict[str, object] = {
            "page[offset]": offset,
            "page[limit]": limit,
            "compact": int(self.compact),
            "fmt": "json",
        }
        if self.contract_id is not None:
            params["filter[contract]"] = self.contract_id
        else:
            params["filter[underlying_symbol]"] = self.underlying_symbol
        optional = {
            "filter[tradetime_from]": self.trade_date_from,
            "filter[tradetime_to]": self.trade_date_to,
            "filter[strike_from]": self.strike_from,
            "filter[strike_to]": self.strike_to,
            "filter[exp_date_from]": self.expiration_from,
            "filter[exp_date_to]": self.expiration_to,
        }
        for key, value in () if self.contract_id is not None else optional.items():
            if value is not None:
                params[key] = value.isoformat() if isinstance(value, date) else value
        if self.fields:
            params["fields[options-eod]"] = ",".join(self.fields)
        return params


@dataclass(frozen=True)
class DownloadConfig:
    """Resource and retry bounds for one-process historical retrieval."""

    token: str
    data_dir: Path
    base_url: str = "https://eodhd.com/api"
    page_limit: int = 1000
    maximum_offset: int = 10000
    request_timeout_seconds: float = 30.0
    max_attempts: int = 4
    exponential_backoff_seconds: float = 1.0
    requests_per_minute: int | None = None
    maximum_raw_records: int = 3_000_000
    maximum_download_bytes: int = 20_000_000_000

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("EODHD_API_TOKEN is required")
        if not 1 <= self.page_limit <= 1000:
            raise ValueError("page_limit must be between 1 and 1000")
        if self.maximum_offset != 10000:
            raise ValueError("maximum_offset must remain at the documented 10000")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.requests_per_minute is not None and self.requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        if self.maximum_raw_records < 1:
            raise ValueError("maximum_raw_records must be positive")
        if self.maximum_download_bytes < 1:
            raise ValueError("maximum_download_bytes must be positive")


@dataclass(frozen=True)
class RequestManifestRow:
    """Credential-free audit row for one provider page request."""

    request_id: str
    underlying_symbol: str
    trade_date_from: str | None
    trade_date_to: str | None
    strike_from: float | None
    strike_to: float | None
    expiration_from: str | None
    expiration_to: str | None
    offset: int
    limit: int
    response_status: int
    record_count: int
    response_hash: str
    attempts: int
    started_at: str
    completed_at: str
    cache_path: str
    superseded_by_split: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe manifest representation."""

        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class DownloadResult:
    """Complete records and page-level provenance for one request chunk."""

    records: list[dict[str, Any]]
    manifest_rows: list[RequestManifestRow]


@dataclass(frozen=True)
class CanonicalRejection:
    """One provider record excluded from the canonical options table."""

    request_id: str
    record_index: int
    provider_record_id: str | None
    reason_code: str
    raw_record_hash: str


@dataclass(frozen=True)
class CanonicalizationResult:
    """Accepted canonical records and explicit rejection provenance."""

    records: list[dict[str, Any]]
    rejections: list[CanonicalRejection]


@dataclass(frozen=True)
class DeduplicationResult:
    """Deterministically unique canonical observations."""

    records: list[dict[str, Any]]
    duplicate_records: int
    conflicting_duplicate_groups: int


@dataclass(frozen=True)
class UnderlyingSymbolMapping:
    """Auditable Stocker-to-provider underlying identity."""

    stocker_symbol: str
    eodhd_underlying_symbol: str | None
    coverage_available: bool
    earliest_option_date: str | None = None
    latest_option_date: str | None = None
    records_returned: int = 0
    mapping_method: str = "unmapped"


@dataclass(frozen=True)
class SymbolMappingResult:
    """Mapping rows plus identities that require human resolution."""

    rows: list[UnderlyingSymbolMapping]
    ambiguous_symbols: list[str]


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of an exact raw response body."""

    return hashlib.sha256(content).hexdigest()


def _canonical_provider_payload(record: Mapping[str, Any]) -> bytes:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _optional_number(value: object, field: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid_numeric_field:{field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"invalid_numeric_field:{field}")
    return number


def _optional_integer(value: object, field: str) -> int | None:
    number = _optional_number(value, field)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"invalid_numeric_field:{field}")
    return int(number)


def _parse_expiration(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("invalid_expiration_date")
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError("invalid_expiration_date") from exc


def _parse_trade_time(value: object) -> tuple[date, datetime]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_trade_timestamp")
    raw = value.strip()
    if len(raw) == 10:
        try:
            trade_day = date.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError("invalid_trade_timestamp") from exc
        local_close = datetime.combine(trade_day, datetime_time(16, 0), tzinfo=NEW_YORK)
        return trade_day, local_close.astimezone(UTC)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid_trade_timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NEW_YORK)
    local = parsed.astimezone(NEW_YORK)
    return local.date(), local.astimezone(UTC)


def provider_eod_observation_date(record: Mapping[str, Any]) -> date:
    """Read the documented EOD resource's observation-date identity suffix."""

    resource_id = record.get("id")
    if not isinstance(resource_id, str) or len(resource_id) < 10:
        raise ValueError("invalid_eod_observation_date")
    try:
        return date.fromisoformat(resource_id[-10:])
    except ValueError as exc:
        raise ValueError("invalid_eod_observation_date") from exc


def _quote_observation_date(value: object, side: str) -> date:
    try:
        quote_date, _ = _parse_trade_time(value)
    except ValueError as exc:
        raise ValueError(f"invalid_{side}_observation_timestamp") from exc
    return quote_date


def _reason_from_error(error: ValueError) -> str:
    message = str(error)
    if message.startswith("invalid_numeric_field:"):
        return message.replace(":", "_", 1)
    return message


def canonicalize_response_records(
    records: Sequence[Mapping[str, Any]],
    *,
    request_id: str,
    provider_schema_version: str,
) -> CanonicalizationResult:
    """Map documented EODHD EOD fields without treating last trade as chain date."""

    accepted: list[dict[str, Any]] = []
    rejected: list[CanonicalRejection] = []
    for index, item in enumerate(records):
        raw_hash = sha256_bytes(_canonical_provider_payload(item))
        provider_id_value = item.get("id")
        provider_id = None if provider_id_value is None else str(provider_id_value)
        attributes_value = item.get("attributes", item)
        try:
            if not isinstance(attributes_value, Mapping):
                raise ValueError("invalid_record_attributes")
            attributes = cast(Mapping[str, Any], attributes_value)
            contract_value = attributes.get("contract")
            if not isinstance(contract_value, str) or not contract_value.strip():
                raise ValueError("missing_contract_id")
            contract_id = contract_value.strip()
            underlying_value = attributes.get("underlying_symbol")
            if not isinstance(underlying_value, str) or not underlying_value.strip():
                raise ValueError("missing_underlying_symbol")
            option_value = attributes.get("type")
            if not isinstance(option_value, str) or option_value.casefold() not in {"call", "put"}:
                raise ValueError("invalid_option_type")
            option_type = option_value.casefold()
            strike = _optional_number(attributes.get("strike"), "strike")
            if strike is None or strike <= 0.0:
                raise ValueError("invalid_strike")
            expiration = _parse_expiration(attributes.get("exp_date"))
            observation_day = provider_eod_observation_date(item)
            bid_observation_day = _quote_observation_date(attributes.get("bid_date"), "bid")
            ask_observation_day = _quote_observation_date(attributes.get("ask_date"), "ask")
            if not (observation_day == bid_observation_day == ask_observation_day):
                raise ValueError("eod_observation_date_mismatch")
            _, trade_timestamp = _parse_trade_time(attributes.get("tradetime"))
            trade_day = observation_day
            if expiration < trade_day:
                raise ValueError("expiration_before_trade_date")
            calculated_dte = (expiration - trade_day).days
            provider_dte = _optional_integer(attributes.get("dte"), "dte")
            if provider_dte is not None and provider_dte < 0:
                raise ValueError("negative_dte")
            if provider_dte is not None and provider_dte != calculated_dte:
                raise ValueError("contract_date_dte_inconsistency")
            bid = _optional_number(attributes.get("bid"), "bid")
            ask = _optional_number(attributes.get("ask"), "ask")
            midpoint = _optional_number(attributes.get("midpoint"), "midpoint")
            if midpoint is None and bid is not None and ask is not None:
                midpoint = (bid + ask) / 2.0
            canonical = {
                "provider": "EODHD/UnicornBay",
                "provider_schema_version": provider_schema_version,
                "request_id": request_id,
                "underlying_symbol": underlying_value.strip(),
                "contract_id": contract_id,
                "option_type": option_type,
                "expiration_date": expiration,
                "strike": strike,
                "trade_date": trade_day,
                "trade_timestamp": trade_timestamp,
                "last": _optional_number(attributes.get("last"), "last"),
                "bid": bid,
                "ask": ask,
                "bid_size": _optional_integer(attributes.get("bid_size"), "bid_size"),
                "ask_size": _optional_integer(attributes.get("ask_size"), "ask_size"),
                "midpoint": midpoint,
                "volume": _optional_integer(attributes.get("volume"), "volume"),
                "open_interest": _optional_integer(
                    attributes.get("open_interest"), "open_interest"
                ),
                "implied_volatility": _optional_number(attributes.get("volatility"), "volatility"),
                "theoretical_value": _optional_number(attributes.get("theoretical"), "theoretical"),
                "delta": _optional_number(attributes.get("delta"), "delta"),
                "gamma": _optional_number(attributes.get("gamma"), "gamma"),
                "theta": _optional_number(attributes.get("theta"), "theta"),
                "vega": _optional_number(attributes.get("vega"), "vega"),
                "rho": _optional_number(attributes.get("rho"), "rho"),
                "dte": calculated_dte,
                "moneyness": _optional_number(attributes.get("moneyness"), "moneyness"),
                "underlying_reference_price": _optional_number(
                    attributes.get("underlying_price"), "underlying_price"
                ),
                "raw_record_hash": raw_hash,
            }
            if tuple(canonical) != CANONICAL_OPTION_COLUMNS:
                raise AssertionError("canonical option column order drifted")
            accepted.append(canonical)
        except ValueError as error:
            rejected.append(
                CanonicalRejection(
                    request_id=request_id,
                    record_index=index,
                    provider_record_id=provider_id,
                    reason_code=_reason_from_error(error),
                    raw_record_hash=raw_hash,
                )
            )
    return CanonicalizationResult(records=accepted, rejections=rejected)


def resolve_canonical_duplicates(
    records: Sequence[Mapping[str, Any]],
) -> DeduplicationResult:
    """Keep the lexicographically smallest raw hash for each contract-date identity."""

    grouped: dict[tuple[str, str, date], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["provider"]),
            str(record["contract_id"]),
            cast(date, record["trade_date"]),
        )
        grouped.setdefault(key, []).append(dict(record))
    output: list[dict[str, Any]] = []
    duplicate_records = 0
    conflicting_groups = 0
    for key in sorted(grouped, key=lambda value: (value[0], value[1], value[2])):
        group = grouped[key]
        duplicate_records += len(group) - 1
        hashes = {str(record["raw_record_hash"]) for record in group}
        if len(hashes) > 1:
            conflicting_groups += 1
        output.append(min(group, key=lambda record: str(record["raw_record_hash"])))
    return DeduplicationResult(
        records=output,
        duplicate_records=duplicate_records,
        conflicting_duplicate_groups=conflicting_groups,
    )


def deterministic_symbol_mapping(
    stocker_symbols: Sequence[str], *, provider_coverage: set[str]
) -> SymbolMappingResult:
    """Map only auditable exact, share-class, and US-suffix transforms."""

    coverage = {symbol.upper() for symbol in provider_coverage}
    rows: list[UnderlyingSymbolMapping] = []
    ambiguous: list[str] = []
    for raw_symbol in stocker_symbols:
        stocker = raw_symbol.upper()
        base = stocker.removesuffix(".US")
        candidates: list[tuple[str, str]] = [(stocker, "exact_symbol")]
        if stocker.endswith(".US"):
            candidates.append((base, "strip_us_suffix"))
        else:
            candidates.append((f"{base}.US", "add_us_suffix"))
        if "." in base:
            hyphenated = base.replace(".", "-")
            candidates.extend(
                [
                    (hyphenated, "dot_to_hyphen"),
                    (f"{hyphenated}.US", "dot_to_hyphen_add_us_suffix"),
                ]
            )
        if "-" in base:
            dotted = base.replace("-", ".")
            candidates.extend(
                [
                    (dotted, "hyphen_to_dot"),
                    (f"{dotted}.US", "hyphen_to_dot_add_us_suffix"),
                ]
            )
        candidates = list(dict.fromkeys(candidates))
        matches = [(candidate, method) for candidate, method in candidates if candidate in coverage]
        unique_matches = {candidate for candidate, _method in matches}
        if len(unique_matches) > 1:
            ambiguous.append(stocker)
            rows.append(
                UnderlyingSymbolMapping(
                    stocker_symbol=stocker,
                    eodhd_underlying_symbol=None,
                    coverage_available=False,
                    mapping_method="ambiguous_exact_transforms",
                )
            )
        elif matches:
            candidate, method = matches[0]
            rows.append(
                UnderlyingSymbolMapping(
                    stocker_symbol=stocker,
                    eodhd_underlying_symbol=candidate,
                    coverage_available=True,
                    mapping_method=method,
                )
            )
        else:
            rows.append(
                UnderlyingSymbolMapping(
                    stocker_symbol=stocker,
                    eodhd_underlying_symbol=None,
                    coverage_available=False,
                    mapping_method="no_exact_coverage_match",
                )
            )
    return SymbolMappingResult(rows=rows, ambiguous_symbols=ambiguous)


def redact_secrets(text: str, *, secrets: Sequence[str] = ()) -> str:
    """Redact known values and authenticated query parameters from arbitrary text."""

    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return _TOKEN_QUERY.sub(r"\1%5BREDACTED%5D", redacted)


def safe_parameters(params: Mapping[str, object]) -> dict[str, object]:
    """Return request parameters with every credential-bearing key removed."""

    return {
        str(key): value
        for key, value in params.items()
        if str(key).casefold() not in SECRET_PARAMETER_NAMES
    }


def stable_request_id(endpoint: str, params: Mapping[str, object]) -> str:
    """Identify a request from its endpoint and non-secret semantic parameters."""

    payload = json.dumps(
        {"endpoint": endpoint, "params": safe_parameters(params)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _midpoint_date(start: date, end: date) -> date:
    return start + timedelta(days=(end - start).days // 2)


def split_request_for_offset_limit(
    request: OptionsRequest,
) -> tuple[OptionsRequest, OptionsRequest]:
    """Split by trade date, then expiry, then strike without changing semantics."""

    if (
        request.trade_date_from is not None
        and request.trade_date_to is not None
        and request.trade_date_from < request.trade_date_to
    ):
        midpoint = _midpoint_date(request.trade_date_from, request.trade_date_to)
        return (
            request.replace(trade_date_to=midpoint),
            request.replace(trade_date_from=midpoint + timedelta(days=1)),
        )
    if (
        request.expiration_from is not None
        and request.expiration_to is not None
        and request.expiration_from < request.expiration_to
    ):
        midpoint = _midpoint_date(request.expiration_from, request.expiration_to)
        return (
            request.replace(expiration_to=midpoint),
            request.replace(expiration_from=midpoint + timedelta(days=1)),
        )
    if (
        request.strike_from is not None
        and request.strike_to is not None
        and request.strike_from < request.strike_to
    ):
        midpoint_strike = (request.strike_from + request.strike_to) / 2.0
        right_start = math.nextafter(midpoint_strike, math.inf)
        if right_start > request.strike_to:
            raise OffsetLimitExceeded("single strike bucket exceeds the provider offset limit")
        return (
            request.replace(strike_to=midpoint_strike),
            request.replace(strike_from=right_start),
        )
    raise OffsetLimitExceeded("request cannot be narrowed below the provider offset limit")


def _date_text(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _next_page_offset(links: Mapping[str, Any]) -> int | None:
    next_value = links.get("next")
    if next_value in {None, ""}:
        return None
    if not isinstance(next_value, str):
        raise OptionsSchemaError("pagination next link is invalid")
    values = parse_qs(urlparse(next_value).query).get("page[offset]")
    if values is None or len(values) != 1:
        raise OptionsSchemaError("pagination next link lacks one offset")
    try:
        offset = int(values[0])
    except ValueError as exc:
        raise OptionsSchemaError("pagination next-link offset is invalid") from exc
    if offset < 0:
        raise OptionsSchemaError("pagination next-link offset is negative")
    return offset


class EODHDOptionsDownloader:
    """Sequential, resumable, content-addressed options EOD downloader."""

    def __init__(
        self,
        config: DownloadConfig,
        *,
        transport: TransportLike,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport
        self.sleep = sleep
        self._last_request_monotonic: float | None = None
        self._raw_records_accounted = 0
        self._download_bytes_accounted = 0

    def _account_resources(self, *, records: int, response_bytes: int) -> None:
        if self._raw_records_accounted + records > self.config.maximum_raw_records:
            raise OptionsResourceLimitExceeded(
                "blocked_options_download_resource_limit: raw-record ceiling"
            )
        if self._download_bytes_accounted + response_bytes > self.config.maximum_download_bytes:
            raise OptionsResourceLimitExceeded(
                "blocked_options_download_resource_limit: download-byte ceiling"
            )
        self._raw_records_accounted += records
        self._download_bytes_accounted += response_bytes

    def _pace(self) -> None:
        if self.config.requests_per_minute is None:
            return
        minimum_interval = 60.0 / float(self.config.requests_per_minute)
        now = time.monotonic()
        if self._last_request_monotonic is not None:
            remaining = minimum_interval - (now - self._last_request_monotonic)
            if remaining > 0.0:
                self.sleep(remaining)
        self._last_request_monotonic = time.monotonic()

    @staticmethod
    def _close_response(response: ResponseLike) -> None:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    def _bounded_response_content(self, response: ResponseLike) -> bytes:
        """Read at most the remaining frozen payload-byte allowance."""

        remaining = self.config.maximum_download_bytes - self._download_bytes_accounted
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
        iterator = getattr(response, "iter_content", None)
        if callable(iterator):
            chunks: list[bytes] = []
            consumed = 0
            bounded_chunk_size = max(min(65_536, remaining), 1)
            for chunk in iterator(chunk_size=bounded_chunk_size):
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
        content = response.content
        if len(content) > remaining:
            self._close_response(response)
            raise OptionsResourceLimitExceeded(
                "blocked_options_download_resource_limit: download-byte ceiling"
            )
        self._download_bytes_accounted += len(content)
        self._close_response(response)
        return content

    def _retry_delay(self, response: ResponseLike, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return self.config.exponential_backoff_seconds * (2.0 ** max(attempt - 1, 0))

    def _request(
        self, request: OptionsRequest, params: dict[str, object]
    ) -> tuple[ResponseLike, int]:
        authenticated = {**params, "api_token": self.config.token}
        url = f"{self.config.base_url.rstrip('/')}{request.endpoint}"
        response: ResponseLike | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            self._pace()
            try:
                response = self.transport.get(
                    url,
                    params=authenticated,
                    timeout=self.config.request_timeout_seconds,
                )
            except Exception as exc:
                if attempt >= self.config.max_attempts:
                    message = redact_secrets(str(exc), secrets=(self.config.token,))
                    raise OptionsDownloadError(f"transport failure: {message}") from None
                self.sleep(self.config.exponential_backoff_seconds * (2.0 ** (attempt - 1)))
                continue
            if response.status_code == 200:
                try:
                    content = self._bounded_response_content(response)
                except OptionsResourceLimitExceeded:
                    raise
                except Exception as exc:
                    self._close_response(response)
                    if attempt >= self.config.max_attempts:
                        message = redact_secrets(str(exc), secrets=(self.config.token,))
                        raise OptionsDownloadError(
                            f"response-body transport failure: {message}"
                        ) from None
                    self.sleep(self.config.exponential_backoff_seconds * (2.0 ** (attempt - 1)))
                    continue
                token_bytes = self.config.token.encode("utf-8")
                if token_bytes and token_bytes in content:
                    raise OptionsAuthenticationError(
                        "EODHD options response contains credential material"
                    )
                return _MaterializedResponse(response, content), attempt
            if response.status_code in {401, 403}:
                self._close_response(response)
                raise OptionsAuthenticationError(
                    f"EODHD options authentication or entitlement failed ({response.status_code})"
                )
            if response.status_code not in TRANSIENT_STATUSES:
                self._close_response(response)
                raise OptionsSchemaError(
                    f"permanent EODHD options response status {response.status_code}"
                )
            if attempt < self.config.max_attempts:
                delay = self._retry_delay(response, attempt)
                self._close_response(response)
                self.sleep(delay)
            else:
                self._close_response(response)
        status = None if response is None else response.status_code
        raise OptionsDownloadError(f"transient EODHD response exhausted retries ({status})")

    def _cache_response(self, content: bytes) -> tuple[str, Path]:
        response_hash = sha256_bytes(content)
        destination = self.config.data_dir / "raw" / f"{response_hash}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        return response_hash, destination

    def _resume_path(self, request: OptionsRequest) -> Path:
        root_id = stable_request_id(
            request.endpoint,
            request.parameters(offset=0, limit=self.config.page_limit),
        )
        return self.config.data_dir / "manifests" / "completed" / f"{root_id}.json"

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    def _load_resumed(self, request: OptionsRequest) -> DownloadResult | None:
        path = self._resume_path(request)
        if not path.exists():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            rows_value = stored["manifest_rows"]
            if not isinstance(rows_value, list) or not rows_value:
                raise ValueError("empty completed manifest")
            rows = [RequestManifestRow(**cast(dict[str, Any], value)) for value in rows_value]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OptionsSchemaError("completed request manifest is invalid") from exc
        expected_offset = 0
        expected_total: int | None = None
        pagination_has_total: bool | None = None
        records: list[dict[str, Any]] = []
        ordered_rows = sorted(rows, key=lambda item: item.offset)
        for row_index, row in enumerate(ordered_rows):
            if row.offset != expected_offset:
                raise OptionsSchemaError("resumed pagination offsets are not contiguous")
            cache_path = Path(row.cache_path)
            if not cache_path.is_file():
                raise OptionsSchemaError("resumed raw response is missing")
            content = cache_path.read_bytes()
            if sha256_bytes(content) != row.response_hash:
                raise OptionsSchemaError("resumed raw response hash mismatch")
            try:
                payload_value = json.loads(content)
            except json.JSONDecodeError as exc:
                raise OptionsSchemaError("resumed raw response is not valid JSON") from exc
            if not isinstance(payload_value, dict):
                raise OptionsSchemaError("resumed raw response must be an object")

            class _CachedResponse:
                status_code = 200
                headers: Mapping[str, str] = {}

                def __init__(self, payload: dict[str, Any], raw: bytes) -> None:
                    self.payload = payload
                    self.content = raw

                def json(self) -> object:
                    return self.payload

            payload, page_records = self._decode_page(
                _CachedResponse(cast(dict[str, Any], payload_value), content)
            )
            meta = cast(dict[str, Any], payload["meta"])
            links = cast(dict[str, Any], payload["links"])
            try:
                page_offset = int(meta["offset"])
                page_limit = int(meta["limit"])
            except (KeyError, TypeError, ValueError) as exc:
                raise OptionsSchemaError("resumed pagination metadata is invalid") from exc
            if page_offset != row.offset or page_limit < 1:
                raise OptionsSchemaError("resumed pagination metadata changed")
            has_total = meta.get("total") is not None
            if pagination_has_total is None:
                pagination_has_total = has_total
            elif pagination_has_total != has_total:
                raise OptionsSchemaError("resumed pagination mode changed")
            if has_total:
                try:
                    total = int(meta["total"])
                except (TypeError, ValueError) as exc:
                    raise OptionsSchemaError("resumed pagination total is invalid") from exc
                if total < 0:
                    raise OptionsSchemaError("resumed pagination total is negative")
                if expected_total is None:
                    expected_total = total
                elif total != expected_total:
                    raise OptionsSchemaError("resumed pagination total changed")
            if len(page_records) != row.record_count:
                raise OptionsSchemaError("resumed record count differs from manifest")
            records.extend(page_records)
            expected_offset += len(page_records)
            next_offset = _next_page_offset(links)
            is_final_manifest_page = row_index == len(ordered_rows) - 1
            if is_final_manifest_page:
                if next_offset is not None:
                    raise OptionsSchemaError("resumed download is incomplete")
                if expected_total is not None and expected_offset != expected_total:
                    raise OptionsSchemaError("resumed download total is incomplete")
            elif next_offset != expected_offset:
                raise OptionsSchemaError("resumed pagination next offset is not contiguous")
        return DownloadResult(records=records, manifest_rows=rows)

    def _save_resume(
        self, request: OptionsRequest, manifest_rows: Sequence[RequestManifestRow]
    ) -> None:
        self._atomic_json(
            self._resume_path(request),
            {"manifest_rows": [row.to_dict() for row in manifest_rows]},
        )

    @staticmethod
    def _decode_page(response: ResponseLike) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            payload = response.json()
        except Exception as exc:
            raise OptionsSchemaError("EODHD options response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise OptionsSchemaError("EODHD options response must be an object")
        meta = payload.get("meta")
        data = payload.get("data")
        links = payload.get("links")
        if not isinstance(meta, dict) or not isinstance(data, list) or not isinstance(links, dict):
            raise OptionsSchemaError("EODHD options response lacks meta/data/links")
        records: list[dict[str, Any]] = []
        fields = meta.get("fields")
        for item in data:
            if isinstance(item, dict):
                records.append(cast(dict[str, Any], item))
            elif isinstance(item, list) and isinstance(fields, list) and len(item) == len(fields):
                records.append(
                    {
                        "type": "options-eod",
                        "attributes": dict(
                            zip((str(field) for field in fields), item, strict=True)
                        ),
                    }
                )
            else:
                raise OptionsSchemaError("EODHD options data row does not match meta.fields")
        return cast(dict[str, Any], payload), records

    def download(self, request: OptionsRequest) -> DownloadResult:
        """Retrieve every page once, cache exact bodies, and verify completeness."""

        resumed = self._load_resumed(request)
        if resumed is not None:
            try:
                self._account_resources(
                    records=sum(row.record_count for row in resumed.manifest_rows),
                    response_bytes=sum(
                        Path(row.cache_path).stat().st_size for row in resumed.manifest_rows
                    ),
                )
            except OptionsResourceLimitExceeded as error:
                raise OptionsResourceLimitExceeded(
                    str(error), manifest_rows=resumed.manifest_rows
                ) from None
            return resumed
        offset = 0
        all_records: list[dict[str, Any]] = []
        manifest: list[RequestManifestRow] = []
        expected_total: int | None = None
        pagination_has_total: bool | None = None
        while True:
            if offset > self.config.maximum_offset:
                raise OffsetLimitExceeded(
                    "provider pagination would exceed offset 10000",
                    manifest_rows=manifest,
                )
            remaining_records = self.config.maximum_raw_records - self._raw_records_accounted
            if remaining_records <= 0:
                raise OptionsResourceLimitExceeded(
                    "blocked_options_download_resource_limit: raw-record ceiling",
                    manifest_rows=manifest,
                )
            if self._download_bytes_accounted >= self.config.maximum_download_bytes:
                raise OptionsResourceLimitExceeded(
                    "blocked_options_download_resource_limit: download-byte ceiling",
                    manifest_rows=manifest,
                )
            request_limit = min(self.config.page_limit, remaining_records)
            params = request.parameters(offset=offset, limit=request_limit)
            request_id = stable_request_id(request.endpoint, params)
            started_at = datetime.now(UTC).isoformat()
            try:
                response, attempts = self._request(request, params)
            except OptionsDownloadError as error:
                error.manifest_rows = [*manifest, *error.manifest_rows]
                raise
            completed_at = datetime.now(UTC).isoformat()
            content = response.content
            try:
                payload, records = self._decode_page(response)
            except OptionsDownloadError as error:
                error.manifest_rows = [*manifest, *error.manifest_rows]
                raise
            meta = cast(dict[str, Any], payload["meta"])
            links = cast(dict[str, Any], payload["links"])
            try:
                page_offset = int(meta["offset"])
                page_limit = int(meta["limit"])
            except (KeyError, TypeError, ValueError) as exc:
                raise OptionsSchemaError(
                    "pagination metadata is invalid", manifest_rows=manifest
                ) from exc
            has_total = meta.get("total") is not None
            if pagination_has_total is None:
                pagination_has_total = has_total
            elif pagination_has_total != has_total:
                raise OptionsSchemaError(
                    "pagination mode changed within one request", manifest_rows=manifest
                )
            total: int | None = None
            if has_total:
                try:
                    total = int(meta["total"])
                except (TypeError, ValueError) as exc:
                    raise OptionsSchemaError(
                        "pagination total is invalid", manifest_rows=manifest
                    ) from exc
            if page_offset != offset or page_limit < 1 or (total is not None and total < 0):
                raise OptionsSchemaError(
                    "pagination metadata disagrees with request", manifest_rows=manifest
                )
            if len(records) > request_limit:
                raise OptionsSchemaError(
                    "pagination returned more rows than requested limit",
                    manifest_rows=manifest,
                )
            response_hash, cache_path = self._cache_response(content)
            current_row = RequestManifestRow(
                request_id=request_id,
                underlying_symbol=request.underlying_symbol,
                trade_date_from=_date_text(request.trade_date_from),
                trade_date_to=_date_text(request.trade_date_to),
                strike_from=request.strike_from,
                strike_to=request.strike_to,
                expiration_from=_date_text(request.expiration_from),
                expiration_to=_date_text(request.expiration_to),
                offset=offset,
                limit=request_limit,
                response_status=response.status_code,
                record_count=len(records),
                response_hash=response_hash,
                attempts=attempts,
                started_at=started_at,
                completed_at=completed_at,
                cache_path=str(cache_path),
            )
            try:
                self._account_resources(records=len(records), response_bytes=0)
            except OptionsResourceLimitExceeded as error:
                raise OptionsResourceLimitExceeded(
                    str(error), manifest_rows=[*manifest, current_row]
                ) from None
            manifest.append(current_row)
            if total is not None and expected_total is None:
                expected_total = total
                if total > self.config.maximum_offset + self.config.page_limit:
                    raise OffsetLimitExceeded(
                        "response total exceeds the provider's retrievable offset window",
                        manifest_rows=manifest,
                    )
            elif total is not None and total != expected_total:
                raise OptionsSchemaError(
                    "pagination total changed within one request", manifest_rows=manifest
                )
            all_records.extend(records)
            consumed = offset + len(records)
            try:
                next_offset = _next_page_offset(links)
            except OptionsSchemaError as error:
                error.manifest_rows = manifest
                raise
            if total is not None and consumed >= total:
                if consumed != total:
                    raise OptionsSchemaError(
                        "pagination returned more rows than meta.total",
                        manifest_rows=manifest,
                    )
                if next_offset is not None:
                    raise OptionsSchemaError(
                        "final page unexpectedly advertises a next link",
                        manifest_rows=manifest,
                    )
                break
            if total is None and next_offset is None:
                break
            if not records or next_offset is None:
                raise OptionsSchemaError(
                    "pagination truncated before meta.total", manifest_rows=manifest
                )
            if next_offset != consumed:
                raise OptionsSchemaError(
                    "pagination next offset is not contiguous", manifest_rows=manifest
                )
            offset = consumed
        result = DownloadResult(records=all_records, manifest_rows=manifest)
        self._save_resume(request, manifest)
        return result

    def download_with_splitting(self, request: OptionsRequest) -> DownloadResult:
        """Download completely, narrowing oversized requests by the frozen split order."""

        try:
            return self.download(request)
        except OffsetLimitExceeded as error:
            left, right = split_request_for_offset_limit(request)
            superseded = [replace(row, superseded_by_split=True) for row in error.manifest_rows]
            try:
                left_result = self.download_with_splitting(left)
            except OptionsDownloadError as child_error:
                child_error.manifest_rows = [
                    *superseded,
                    *child_error.manifest_rows,
                ]
                raise
            try:
                right_result = self.download_with_splitting(right)
            except OptionsDownloadError as child_error:
                child_error.manifest_rows = [
                    *superseded,
                    *left_result.manifest_rows,
                    *child_error.manifest_rows,
                ]
                raise
            return DownloadResult(
                records=[*left_result.records, *right_result.records],
                manifest_rows=[
                    *superseded,
                    *left_result.manifest_rows,
                    *right_result.manifest_rows,
                ],
            )


__all__ = [
    "DownloadConfig",
    "DownloadResult",
    "EODHDOptionsDownloader",
    "OffsetLimitExceeded",
    "OptionsAuthenticationError",
    "OptionsDownloadError",
    "OptionsRequest",
    "OptionsResourceLimitExceeded",
    "OptionsSchemaError",
    "RequestManifestRow",
    "CANONICAL_OPTION_COLUMNS",
    "CanonicalRejection",
    "CanonicalizationResult",
    "DeduplicationResult",
    "SymbolMappingResult",
    "UnderlyingSymbolMapping",
    "canonicalize_response_records",
    "deterministic_symbol_mapping",
    "provider_eod_observation_date",
    "redact_secrets",
    "resolve_canonical_duplicates",
    "safe_parameters",
    "sha256_bytes",
    "split_request_for_offset_limit",
    "stable_request_id",
]
