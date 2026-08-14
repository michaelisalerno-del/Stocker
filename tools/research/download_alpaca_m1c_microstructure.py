#!/usr/bin/env python3
"""Download raw Alpaca SIP ticks around frozen historical M1C events.

This standalone research utility uses only Alpaca's historical stock market-data
GET endpoints. It has no brokerage, account, position, execution, portfolio, or
order operation and is not imported by the Stocker application.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from tools.research.download_ibkr_m1c_microstructure import (
    DEFAULT_CANONICAL_SOURCE,
    DEFAULT_FAMILY_DEFINITION,
    PERIOD_BOUNDS,
    REPOSITORY_ROOT,
    HistoricalEvent,
    event_output_directory,
    event_window,
    family_definition_metadata,
    load_frozen_events,
    select_validation_events,
    sha256_file,
    write_json_atomic,
)

SOURCE = "alpaca_historical_sip"
DATASET_VERSION = "alpaca-m1c-microstructure-v0"
ALPACA_BASE_URL = "https://data.alpaca.markets"
PAGE_LIMIT = 10_000
_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<second>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)


@dataclass(frozen=True)
class AlpacaCredentials:
    """Historical market-data credentials sourced only from the environment."""

    key_id: str
    secret_key: str

    @classmethod
    def from_environment(cls) -> AlpacaCredentials:
        key_id = os.environ.get("APCA_API_KEY_ID", "").strip()
        secret_key = os.environ.get("APCA_API_SECRET_KEY", "").strip()
        if not key_id or not secret_key:
            raise RuntimeError(
                "blocked_alpaca_connection: credentials are absent; set "
                "APCA_API_KEY_ID and APCA_API_SECRET_KEY"
            )
        return cls(key_id=key_id, secret_key=secret_key)


@dataclass(frozen=True)
class ProviderFeed:
    """One fully paginated provider response."""

    feed: str
    records: tuple[dict[str, object], ...]
    request_count: int


class ProviderPaginationError(RuntimeError):
    """The provider returned an unsafe or non-progressing pagination sequence."""


class ProviderRequestError(RuntimeError):
    """A bounded Alpaca market-data request failed."""

    def __init__(self, *, status_code: int | None, message: str, request_count: int) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.request_count = request_count


class HistoricalFeedClient(Protocol):
    """The only provider operation used by the dataset writer."""

    def download_feed(
        self,
        *,
        symbol: str,
        feed: str,
        start_utc: datetime,
        end_utc: datetime,
        max_pages: int = 20_000,
    ) -> ProviderFeed: ...


def _epoch_nanoseconds(value: object) -> int:
    text = str(value)
    match = _TIMESTAMP_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"unsupported Alpaca timestamp: {text}")
    second = datetime.fromisoformat(match.group("second")).replace(tzinfo=UTC)
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return int(second.timestamp()) * 1_000_000_000 + int(fraction or "0")


def _rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _conditions(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("provider conditions must be a list")
    return [str(item) for item in value]


def _float_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("provider numeric value has an invalid type")
    return float(value)


def _int_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("provider integer value has an invalid type")
    return int(value)


def normalize_trade(
    record: Mapping[str, object],
    event: HistoricalEvent,
    *,
    sequence: int,
) -> dict[str, object]:
    """Normalize one raw Alpaca SIP trade without classifying its side."""

    timestamp_ns = _epoch_nanoseconds(record["t"])
    return {
        "m1c_event_id": event.event_id,
        "symbol": event.symbol,
        "m1c_t0_utc": event.t0_utc.astimezone(UTC),
        "provider_timestamp_ns": timestamp_ns,
        "price": _float_value(record["p"]),
        "size": _int_value(record["s"]),
        "exchange": str(record.get("x", "")),
        "trade_id": str(record.get("i", "")),
        "conditions": _conditions(record.get("c")),
        "tape": str(record.get("z", "")),
        "provider_sequence": sequence,
        "source": SOURCE,
        "feed": "TRADES",
    }


def normalize_quote(
    record: Mapping[str, object],
    event: HistoricalEvent,
    *,
    sequence: int,
) -> dict[str, object]:
    """Normalize one raw Alpaca SIP NBBO quote, retaining provider size units."""

    timestamp_ns = _epoch_nanoseconds(record["t"])
    return {
        "m1c_event_id": event.event_id,
        "symbol": event.symbol,
        "m1c_t0_utc": event.t0_utc.astimezone(UTC),
        "provider_timestamp_ns": timestamp_ns,
        "bid": _float_value(record["bp"]),
        "ask": _float_value(record["ap"]),
        "bid_size": _int_value(record["bs"]),
        "ask_size": _int_value(record["as"]),
        "bid_size_unit": "round_lots",
        "ask_size_unit": "round_lots",
        "bid_exchange": str(record.get("bx", "")),
        "ask_exchange": str(record.get("ax", "")),
        "conditions": _conditions(record.get("c")),
        "tape": str(record.get("z", "")),
        "provider_sequence": sequence,
        "source": SOURCE,
        "feed": "BID_ASK",
    }


def assign_provider_identities(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Add deterministic full-content-and-order identities to provider events."""

    result: list[dict[str, object]] = []
    for row in rows:
        canonical = {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in sorted(row.items())
            if key != "provider_event_identity"
        }
        identity = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        result.append({**row, "provider_event_identity": identity})
    return result


TRADE_SCHEMA = pa.schema(
    [
        pa.field("m1c_event_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("m1c_t0_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("provider_timestamp_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("provider_timestamp_ns", pa.int64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("size", pa.int64(), nullable=False),
        pa.field("exchange", pa.string(), nullable=False),
        pa.field("trade_id", pa.string(), nullable=False),
        pa.field("conditions", pa.list_(pa.string()), nullable=False),
        pa.field("tape", pa.string(), nullable=False),
        pa.field("provider_sequence", pa.int64(), nullable=False),
        pa.field("provider_event_identity", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("feed", pa.string(), nullable=False),
    ]
)

QUOTE_SCHEMA = pa.schema(
    [
        pa.field("m1c_event_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("m1c_t0_utc", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("provider_timestamp_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("provider_timestamp_ns", pa.int64(), nullable=False),
        pa.field("bid", pa.float64(), nullable=False),
        pa.field("ask", pa.float64(), nullable=False),
        pa.field("bid_size", pa.int64(), nullable=False),
        pa.field("ask_size", pa.int64(), nullable=False),
        pa.field("bid_size_unit", pa.string(), nullable=False),
        pa.field("ask_size_unit", pa.string(), nullable=False),
        pa.field("bid_exchange", pa.string(), nullable=False),
        pa.field("ask_exchange", pa.string(), nullable=False),
        pa.field("conditions", pa.list_(pa.string()), nullable=False),
        pa.field("tape", pa.string(), nullable=False),
        pa.field("provider_sequence", pa.int64(), nullable=False),
        pa.field("provider_event_identity", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("feed", pa.string(), nullable=False),
    ]
)


def _schema(feed: str) -> pa.Schema:
    if feed == "TRADES":
        return TRADE_SCHEMA
    if feed == "BID_ASK":
        return QUOTE_SCHEMA
    raise ValueError("feed must be TRADES or BID_ASK")


def _table(feed: str, rows: Sequence[dict[str, object]]) -> pa.Table:
    schema = _schema(feed)
    arrays: list[pa.Array] = []
    for field in schema:
        if field.name == "provider_timestamp_utc":
            values = pa.array(
                [_int_value(row["provider_timestamp_ns"]) for row in rows],
                type=pa.int64(),
            ).cast(field.type)
        else:
            values = pa.array([row[field.name] for row in rows], type=field.type)
        arrays.append(values)
    return pa.Table.from_arrays(arrays, schema=schema)


def write_feed_parquet(
    path: str | Path,
    feed: str,
    rows: Sequence[dict[str, object]],
) -> None:
    """Write one deterministic raw provider artifact atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pq.write_table(  # type: ignore[no-untyped-call]
        _table(feed, rows),
        temporary,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )
    temporary.replace(destination)


class AlpacaHistoricalClient:
    """Narrow client exposing only historical SIP trades and quotes."""

    _ENDPOINTS = {
        "TRADES": ("trades", "trades"),
        "BID_ASK": ("quotes", "quotes"),
    }

    def __init__(
        self,
        credentials: AlpacaCredentials,
        *,
        base_url: str = ALPACA_BASE_URL,
        requests_per_minute: int = 180,
        timeout_seconds: float = 60.0,
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= requests_per_minute <= 200:
            raise ValueError("requests_per_minute must be between 1 and 200")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("request bounds are invalid")
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "APCA-API-KEY-ID": credentials.key_id,
                "APCA-API-SECRET-KEY": credentials.secret_key,
                "Accept": "application/json",
                "User-Agent": f"stocker-{DATASET_VERSION}",
            },
            timeout=timeout_seconds,
            transport=transport,
        )
        self._minimum_interval = 60.0 / requests_per_minute
        self._max_retries = max_retries
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._last_request: float | None = None
        self._pace_lock = threading.Lock()

    def __enter__(self) -> AlpacaHistoricalClient:
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _pace(self) -> None:
        with self._pace_lock:
            now = self._monotonic()
            if self._last_request is not None:
                delay = self._minimum_interval - (now - self._last_request)
                if delay > 0:
                    self._sleeper(delay)
                    now = self._monotonic()
            self._last_request = now

    def _get(
        self,
        path: str,
        params: dict[str, str | int | float | bool | None],
        prior_request_count: int,
    ) -> tuple[httpx.Response, int]:
        for attempt in range(self._max_retries + 1):
            attempt_count = attempt + 1
            self._pace()
            try:
                response = self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    raise ProviderRequestError(
                        status_code=None,
                        message=f"connection: {exc}",
                        request_count=prior_request_count + attempt_count,
                    ) from exc
                self._sleeper(min(2**attempt, 30))
                continue
            if response.status_code == 200:
                return response, attempt_count
            message = response.text[:500]
            if response.status_code not in {429, 500, 502, 503, 504}:
                raise ProviderRequestError(
                    status_code=response.status_code,
                    message=message,
                    request_count=prior_request_count + attempt_count,
                )
            if attempt >= self._max_retries:
                raise ProviderRequestError(
                    status_code=response.status_code,
                    message=message,
                    request_count=prior_request_count + attempt_count,
                )
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(2**attempt, 30)
            except ValueError:
                delay = min(2**attempt, 30)
            self._sleeper(max(delay, self._minimum_interval))
        raise AssertionError("bounded retry loop exhausted")

    def download_feed(
        self,
        *,
        symbol: str,
        feed: str,
        start_utc: datetime,
        end_utc: datetime,
        max_pages: int = 20_000,
    ) -> ProviderFeed:
        """Download all opaque-token pages for one exact event interval."""

        try:
            endpoint, payload_key = self._ENDPOINTS[feed]
        except KeyError as exc:
            raise ValueError("feed must be TRADES or BID_ASK") from exc
        if start_utc.tzinfo is None or end_utc.tzinfo is None or start_utc >= end_utc:
            raise ValueError("requested interval must be a positive aware interval")
        if max_pages <= 0:
            raise ValueError("max_pages must be positive")
        path = f"/v2/stocks/{symbol}/{endpoint}"
        page_token: str | None = None
        seen_tokens: set[str] = set()
        records: list[dict[str, object]] = []
        request_count = 0
        for _page_number in range(max_pages):
            params: dict[str, str | int | float | bool | None] = {
                "start": _rfc3339(start_utc),
                "end": _rfc3339(end_utc),
                "feed": "sip",
                "sort": "asc",
                "limit": PAGE_LIMIT,
            }
            if page_token is not None:
                params["page_token"] = page_token
            response, attempts = self._get(path, params, request_count)
            request_count += attempts
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderRequestError(
                    status_code=200,
                    message="provider returned invalid JSON",
                    request_count=request_count,
                ) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get(payload_key), list):
                raise ProviderRequestError(
                    status_code=200,
                    message=f"provider response missing {payload_key}",
                    request_count=request_count,
                )
            for raw in cast(list[object], payload[payload_key]):
                if not isinstance(raw, dict):
                    raise ProviderRequestError(
                        status_code=200,
                        message="provider record is not an object",
                        request_count=request_count,
                    )
                records.append({str(key): value for key, value in raw.items()})
            next_token = payload.get("next_page_token")
            if next_token in {None, ""}:
                return ProviderFeed(feed, tuple(records), request_count)
            if not isinstance(next_token, str):
                raise ProviderPaginationError("provider returned a non-string page token")
            if next_token in seen_tokens:
                raise ProviderPaginationError("provider returned a repeated page token")
            seen_tokens.add(next_token)
            page_token = next_token
        raise ProviderPaginationError(f"provider exceeded {max_pages} pages")


def _feed_filename(feed: str) -> str:
    return "trades.parquet" if feed == "TRADES" else "bid_ask.parquet"


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _iso_from_ns(value: int | None) -> str | None:
    if value is None:
        return None
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    base = datetime.fromtimestamp(seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{nanoseconds:09d}+00:00"


def _valid_resume_feed(
    directory: Path,
    *,
    event: HistoricalEvent,
    feed: str,
    start_utc: datetime,
    end_utc: datetime,
    source_hash: str,
    family_hash: str,
) -> bool:
    metadata = _read_json(directory / "metadata.json")
    details = metadata.get(feed)
    if (
        metadata.get("dataset_version") != DATASET_VERSION
        or metadata.get("canonical_M1C_source_sha256") != source_hash
        or not isinstance((definition := metadata.get("family_definition")), dict)
        or definition.get("sha256") != family_hash
        or metadata.get("event_id") != event.event_id
        or metadata.get("requested_start_utc") != start_utc.isoformat()
        or metadata.get("requested_end_utc") != end_utc.isoformat()
        or not isinstance(details, dict)
        or details.get("completion_status") != "COMPLETE"
    ):
        return False
    path = directory / _feed_filename(feed)
    try:
        table = pq.read_table(path)  # type: ignore[no-untyped-call]
    except (OSError, pa.ArrowException):
        return False
    if table.schema != _schema(feed) or table.num_rows != int(details.get("final_rows", -1)):
        return False
    if table.num_rows == 0:
        return True
    timestamps = table.column("provider_timestamp_ns").to_pylist()
    identities = table.column("provider_event_identity").to_pylist()
    start_ns = int(start_utc.timestamp()) * 1_000_000_000
    end_ns = int(end_utc.timestamp()) * 1_000_000_000
    return (
        set(table.column("m1c_event_id").to_pylist()) == {event.event_id}
        and set(table.column("source").to_pylist()) == {SOURCE}
        and set(table.column("feed").to_pylist()) == {feed}
        and all(isinstance(value, int) and start_ns <= value <= end_ns for value in timestamps)
        and all(isinstance(value, str) and len(value) == 64 for value in identities)
        and len(set(identities)) == len(identities)
    )


def _completed_details(
    *,
    request_count: int,
    raw_rows: int,
    rows: Sequence[dict[str, object]],
    output_file: str,
) -> dict[str, object]:
    timestamps = [_int_value(row["provider_timestamp_ns"]) for row in rows]
    return {
        "request_count": request_count,
        "raw_rows": raw_rows,
        "final_rows": len(rows),
        "earliest_timestamp": _iso_from_ns(min(timestamps) if timestamps else None),
        "latest_timestamp": _iso_from_ns(max(timestamps) if timestamps else None),
        "completion_status": "COMPLETE",
        "permission_status": "PASS",
        "provider_error": None,
        "duplicate_overlap_removed": 0,
        "output_file": output_file,
    }


def _blocked_details(error: Exception, *, request_count: int = 0) -> dict[str, object]:
    status = error.status_code if isinstance(error, ProviderRequestError) else None
    permission = "BLOCKED" if status in {401, 403} else "UNKNOWN"
    return {
        "request_count": request_count,
        "raw_rows": 0,
        "final_rows": 0,
        "earliest_timestamp": None,
        "latest_timestamp": None,
        "completion_status": "BLOCKED_PERMISSION" if permission == "BLOCKED" else "BLOCKED",
        "permission_status": permission,
        "provider_error": f"{type(error).__name__}:{status}:{error}",
        "duplicate_overlap_removed": 0,
        "output_file": None,
    }


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _event_status(metadata: Mapping[str, object]) -> str:
    statuses = [
        details.get("completion_status")
        for feed in ("TRADES", "BID_ASK")
        if isinstance((details := metadata.get(feed)), dict)
    ]
    if statuses == ["COMPLETE", "COMPLETE"]:
        return "complete"
    if "COMPLETE" in statuses:
        return "partial"
    return "blocked"


def _write_summary(root: Path, metadata_rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "period",
        "event_id",
        "symbol",
        "feed",
        "family",
        "T0",
        "requested_start",
        "requested_end",
        "status",
        "request_count",
        "raw_rows",
        "final_rows",
        "first_tick",
        "last_tick",
        "duplicate_overlap_removed",
        "permission_status",
        "provider_error",
        "output_file",
    ]
    rows: list[dict[str, object]] = []
    for metadata in metadata_rows:
        for feed in ("TRADES", "BID_ASK"):
            details = metadata.get(feed)
            if not isinstance(details, dict):
                continue
            rows.append(
                {
                    "period": metadata.get("period"),
                    "event_id": metadata.get("event_id"),
                    "symbol": metadata.get("symbol"),
                    "feed": feed,
                    "family": metadata.get("family"),
                    "T0": metadata.get("t0_utc"),
                    "requested_start": metadata.get("requested_start_utc"),
                    "requested_end": metadata.get("requested_end_utc"),
                    "status": details.get("completion_status"),
                    "request_count": details.get("request_count"),
                    "raw_rows": details.get("raw_rows"),
                    "final_rows": details.get("final_rows"),
                    "first_tick": details.get("earliest_timestamp"),
                    "last_tick": details.get("latest_timestamp"),
                    "duplicate_overlap_removed": details.get("duplicate_overlap_removed"),
                    "permission_status": details.get("permission_status"),
                    "provider_error": details.get("provider_error"),
                    "output_file": details.get("output_file"),
                }
            )
    rows.sort(key=lambda row: (str(row["period"]), str(row["event_id"]), str(row["feed"])))
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "download_summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "download_summary.csv")


def _write_manifest(
    root: Path,
    *,
    source: Path,
    source_hash: str,
    family_hash: str,
    include_r01: bool,
) -> dict[str, object]:
    metadata_rows = [_read_json(path) for path in sorted(root.glob("*/*/*/metadata.json"))]
    statuses = [_event_status(row) for row in metadata_rows]
    manifest: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "canonical_M1C_source": str(source.resolve()),
        "canonical_M1C_source_sha256": source_hash,
        "family_definition": family_definition_metadata(family_hash),
        "development_dates": {
            "start": PERIOD_BOUNDS["development"][0],
            "end": PERIOD_BOUNDS["development"][1],
        },
        "assessment_dates": {
            "start": PERIOD_BOUNDS["assessment"][0],
            "end": PERIOD_BOUNDS["assessment"][1],
        },
        "families": sorted(
            {str(row["family"]) for row in metadata_rows if row.get("family")},
            key=("HARD_M1C", "P01_NEAR_M1", "R01").index,
        ),
        "window": {"pre_minutes": 5, "post_minutes": 20},
        "feed_types": ["TRADES", "BID_ASK"],
        "provider": "Alpaca",
        "provider_feed": "SIP",
        "provider_api": "v2 historical stocks",
        "event_counts": {
            period: sum(row.get("period") == period for row in metadata_rows)
            for period in PERIOD_BOUNDS
        }
        | {"total": len(metadata_rows)},
        "complete_events": statuses.count("complete"),
        "partial_events": statuses.count("partial"),
        "blocked_events": statuses.count("blocked"),
        "total_TRADES_rows": sum(
            int(details.get("final_rows", 0))
            for row in metadata_rows
            if isinstance((details := row.get("TRADES")), dict)
        ),
        "total_BID_ASK_rows": sum(
            int(details.get("final_rows", 0))
            for row in metadata_rows
            if isinstance((details := row.get("BID_ASK")), dict)
        ),
        "symbols": sorted({str(row["symbol"]) for row in metadata_rows if row.get("symbol")}),
        "download_timestamp_utc": datetime.now(UTC).isoformat(),
        "no_order_invariant": "PASS",
        "direction_analysis_performed": False,
        "research_only": True,
        "execution_enabled": False,
        "live_ordering_enabled": False,
        "order_placement": "disabled",
    }
    write_json_atomic(root / "download_manifest.json", manifest)
    _write_summary(root, metadata_rows)
    return manifest


def run_download(
    *,
    period: str,
    source: Path,
    output_root: Path,
    include_r01: bool,
    resume: bool,
    validation_hard_count: int | None,
    event_limit: int | None,
    client: HistoricalFeedClient,
) -> dict[str, int]:
    """Download selected frozen events and persist a resumable raw dataset."""

    source_hash = sha256_file(source)
    family_hash = sha256_file(DEFAULT_FAMILY_DEFINITION)
    allowed_families = {"HARD_M1C", "P01_NEAR_M1"} | ({"R01"} if include_r01 else set())
    for path in sorted(output_root.glob("*/*/*/metadata.json")):
        existing = _read_json(path)
        definition = existing.get("family_definition")
        if (
            existing.get("dataset_version") != DATASET_VERSION
            or existing.get("canonical_M1C_source_sha256") != source_hash
            or not isinstance(definition, dict)
            or definition.get("sha256") != family_hash
            or existing.get("family") not in allowed_families
        ):
            raise ValueError(f"existing dataset has incompatible provenance: {path}")
    events = load_frozen_events(source, period=period, include_r01=include_r01)
    if validation_hard_count is not None:
        events = select_validation_events(
            events,
            count=validation_hard_count,
            family="HARD_M1C",
        )
    if event_limit is not None:
        if event_limit <= 0:
            raise ValueError("event_limit must be positive")
        events = events[:event_limit]
    if any(event.session_date.startswith("2026-") for event in events):
        raise ValueError("2026 events are excluded")

    complete = partial = blocked = skipped = 0
    git_commit = _git_commit()
    family_metadata = family_definition_metadata(family_hash)
    for index, event in enumerate(events, start=1):
        start_utc, end_utc = event_window(event)
        directory = event_output_directory(output_root, event)
        previous = _read_json(directory / "metadata.json")
        details_by_feed: dict[str, dict[str, object]] = {}
        for feed in ("TRADES", "BID_ASK"):
            if resume and _valid_resume_feed(
                directory,
                event=event,
                feed=feed,
                start_utc=start_utc,
                end_utc=end_utc,
                source_hash=source_hash,
                family_hash=family_hash,
            ):
                details = previous.get(feed)
                assert isinstance(details, dict)
                details_by_feed[feed] = cast(dict[str, object], details)
                print(
                    f"[{index}/{len(events)}] {event.symbol} {event.family} {feed}: SKIP",
                    flush=True,
                )
                continue
            try:
                provider = client.download_feed(
                    symbol=event.symbol,
                    feed=feed,
                    start_utc=start_utc,
                    end_utc=end_utc,
                )
                start_ns = int(start_utc.timestamp()) * 1_000_000_000
                end_ns = int(end_utc.timestamp()) * 1_000_000_000
                normalized = [
                    (
                        normalize_trade(record, event, sequence=sequence)
                        if feed == "TRADES"
                        else normalize_quote(record, event, sequence=sequence)
                    )
                    for sequence, record in enumerate(provider.records)
                ]
                rows = assign_provider_identities(
                    [
                        row
                        for row in normalized
                        if start_ns
                        <= _int_value(row["provider_timestamp_ns"])
                        <= end_ns
                    ]
                )
                filename = _feed_filename(feed)
                write_feed_parquet(directory / filename, feed, rows)
                details_by_feed[feed] = _completed_details(
                    request_count=provider.request_count,
                    raw_rows=len(provider.records),
                    rows=rows,
                    output_file=filename,
                )
                print(
                    f"[{index}/{len(events)}] {event.symbol} {event.family} "
                    f"{feed}: COMPLETE {len(rows):,} rows",
                    flush=True,
                )
            except (ProviderRequestError, ProviderPaginationError, ValueError) as error:
                request_count = (
                    error.request_count if isinstance(error, ProviderRequestError) else 0
                )
                details_by_feed[feed] = _blocked_details(error, request_count=request_count)
                print(
                    f"[{index}/{len(events)}] {event.symbol} {event.family} "
                    f"{feed}: BLOCKED {error}",
                    flush=True,
                )

        metadata: dict[str, object] = {
            "dataset_version": DATASET_VERSION,
            "canonical_M1C_source_sha256": source_hash,
            "family_definition": family_metadata,
            "period": event.period,
            "event_id": event.event_id,
            "symbol": event.symbol,
            "session_date": event.session_date,
            "family": event.family,
            "m1c_probability": event.probability,
            "checkpoint": event.checkpoint,
            "t0_utc": event.t0_utc.isoformat(),
            "requested_start_utc": start_utc.isoformat(),
            "requested_end_utc": end_utc.isoformat(),
            "instrument": {
                "symbol": event.symbol,
                "asset_class": "US_EQUITY",
                "currency": "USD",
                "provider_feed": "SIP",
            },
            "provider": "Alpaca",
            "provider_api": "v2 historical stocks",
            "download_timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "research_only": True,
            "execution_enabled": False,
            "live_ordering_enabled": False,
            "order_placement": "disabled",
            "direction_analysis_performed": False,
            **details_by_feed,
        }
        write_json_atomic(directory / "metadata.json", metadata)
        status = _event_status(metadata)
        if status == "complete":
            complete += 1
            if all(
                resume
                and _valid_resume_feed(
                    directory,
                    event=event,
                    feed=feed,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    source_hash=source_hash,
                    family_hash=family_hash,
                )
                for feed in ("TRADES", "BID_ASK")
            ) and all(previous.get(feed) == details_by_feed[feed] for feed in details_by_feed):
                skipped += 1
        elif status == "partial":
            partial += 1
        else:
            blocked += 1

    _write_manifest(
        output_root,
        source=source,
        source_hash=source_hash,
        family_hash=family_hash,
        include_r01=include_r01,
    )
    return {
        "selected_events": len(events),
        "complete_events": complete,
        "partial_events": partial,
        "blocked_events": blocked,
        "skipped_events": skipped,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", required=True, choices=tuple(PERIOD_BOUNDS))
    parser.add_argument("--source", type=Path, default=DEFAULT_CANONICAL_SOURCE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "research-data" / DATASET_VERSION,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-r01", action="store_true")
    parser.add_argument("--validation-hard-count", type=int)
    parser.add_argument("--event-limit", type=int)
    parser.add_argument("--requests-per-minute", type=int, default=180)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    credentials = AlpacaCredentials.from_environment()
    with AlpacaHistoricalClient(
        credentials,
        requests_per_minute=args.requests_per_minute,
        timeout_seconds=args.request_timeout_seconds,
        max_retries=args.max_retries,
    ) as client:
        result = run_download(
            period=args.period,
            source=args.source,
            output_root=args.output_root,
            include_r01=args.include_r01,
            resume=args.resume,
            validation_hard_count=args.validation_hard_count,
            event_limit=args.event_limit,
            client=client,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result["blocked_events"] else 0


if __name__ == "__main__":
    sys.exit(main())
