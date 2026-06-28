"""EODHD data-vendor adapter for Stocker's local data pipeline."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd

from stocker_core.config import EODHDConfig
from stocker_data.audit import create_audit_report
from stocker_data.catalog import write_catalog
from stocker_data.schema import ALL_SCHEMA_COLUMNS
from stocker_data.storage import DatasetKey, dataset_path, read_parquet, write_parquet
from stocker_data.universe import UniverseSymbol
from stocker_data.validate import ValidationIssue, validate_ohlcv

EOD_PERIOD_TO_TIMEFRAME: dict[str, str] = {"d": "1d", "w": "1w", "m": "1mo"}
INTRADAY_MAX_SPAN_DAYS: dict[str, int] = {"1m": 120, "5m": 600, "1h": 7200}
SCREENER_MAX_LIMIT = 100
SCREENER_MAX_OFFSET = 999


class EODHDError(Exception):
    """Base class for EODHD adapter errors."""


class EODHDMissingTokenError(EODHDError):
    """Raised when a live EODHD request lacks an API token."""


class EODHDHTTPError(EODHDError):
    """Raised when EODHD returns a non-success HTTP response."""


class EODHDRateLimitError(EODHDHTTPError):
    """Raised when EODHD rate limits a request after retries."""


class EODHDTemporaryHTTPError(EODHDHTTPError):
    """Raised when temporary EODHD failures are exhausted."""


class EODHDPermanentHTTPError(EODHDHTTPError):
    """Raised when EODHD returns a non-retryable HTTP response."""


class EODHDEmptyResponseError(EODHDError):
    """Raised when EODHD returns no rows."""


class EODHDSchemaError(EODHDError):
    """Raised when an EODHD payload cannot be normalized."""


class EODHDUnsupportedIntervalError(EODHDError):
    """Raised when an intraday interval is not supported."""


class EODHDInvalidDateRangeError(EODHDError):
    """Raised when a requested date range is invalid."""


class EODHDUnsupportedScreenerError(EODHDError):
    """Raised when a screener request would violate EODHD guardrails."""


@dataclass(frozen=True)
class EODHDFetchPlan:
    """Dry-run details for one EODHD fetch command."""

    endpoint: str
    params: dict[str, str]
    timeframe: str
    output_path: Path
    save_raw: bool
    raw_paths: list[Path]
    chunks: list[tuple[datetime, datetime]]

    def to_dict(self) -> dict[str, Any]:
        """Return a printable plan."""

        return {
            "endpoint": self.endpoint,
            "params": self.params,
            "timeframe": self.timeframe,
            "output_path": str(self.output_path),
            "output_filename": self.output_path.name,
            "save_raw": self.save_raw,
            "raw_paths": [str(path) for path in self.raw_paths],
            "chunks": [
                {
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                    "from_unix": int(start.timestamp()),
                    "to_unix": int(end.timestamp()),
                }
                for start, end in self.chunks
            ],
        }


@dataclass(frozen=True)
class EODHDFetchResult:
    """Result of fetching, normalizing, validating, and storing EODHD data."""

    output_path: Path
    catalog_path: Path
    raw_paths: list[Path]
    rows_fetched: int
    rows_saved: int
    min_timestamp: str | None
    max_timestamp: str | None
    issues: list[ValidationIssue]
    audit_markdown_path: Path | None = None
    audit_json_path: Path | None = None

    @property
    def validation_warning_count(self) -> int:
        """Return validation warning count."""

        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def validation_error_count(self) -> int:
        """Return validation error count."""

        return sum(1 for issue in self.issues if issue.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        """Return a printable result summary."""

        return {
            "output_path": str(self.output_path),
            "catalog_path": str(self.catalog_path),
            "raw_paths": [str(path) for path in self.raw_paths],
            "rows_fetched": self.rows_fetched,
            "rows_saved": self.rows_saved,
            "date_range": {
                "from": self.min_timestamp,
                "to": self.max_timestamp,
            },
            "validation_warnings": self.validation_warning_count,
            "validation_errors": self.validation_error_count,
            "audit_markdown_path": str(self.audit_markdown_path)
            if self.audit_markdown_path is not None
            else None,
            "audit_json_path": str(self.audit_json_path)
            if self.audit_json_path is not None
            else None,
        }


class EODHDClient:
    """Small synchronous EODHD HTTP client."""

    def __init__(
        self,
        *,
        config: EODHDConfig | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or EODHDConfig()
        self._sleep = sleep or time.sleep
        self._client = httpx.Client(
            timeout=self.config.request_timeout_seconds,
            transport=transport,
        )

    def require_token(self) -> str:
        """Read the EODHD API token from the configured environment variable."""

        token = os.getenv(self.config.api_token_env)
        if token is None or not token.strip():
            raise EODHDMissingTokenError(
                f"Missing EODHD API token. Set {self.config.api_token_env} before live fetches."
            )
        return token.strip()

    def build_eod_request(
        self,
        *,
        symbol: str,
        from_date: str,
        to_date: str,
        period: str,
        api_token: str,
    ) -> httpx.Request:
        """Build an EODHD EOD request."""

        timeframe_for_eod_period(period)
        return self._client.build_request(
            "GET",
            f"{self.config.base_url.rstrip('/')}/eod/{symbol.upper()}",
            params={
                "from": from_date,
                "to": to_date,
                "period": period,
                "api_token": api_token,
                "fmt": self.config.default_fmt,
            },
        )

    def build_intraday_request(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        api_token: str,
    ) -> httpx.Request:
        """Build an EODHD intraday request."""

        validate_intraday_interval(interval)
        return self._client.build_request(
            "GET",
            f"{self.config.base_url.rstrip('/')}/intraday/{symbol.upper()}",
            params={
                "interval": interval,
                "from": str(int(start.timestamp())),
                "to": str(int(end.timestamp())),
                "api_token": api_token,
                "fmt": self.config.default_fmt,
            },
        )

    def build_screener_request(
        self,
        *,
        filters: list[list[Any]],
        signals: list[str],
        sort: str,
        limit: int,
        offset: int,
        api_token: str,
    ) -> httpx.Request:
        """Build an EODHD screener request without leaking the token to callers."""

        _validate_screener_page(limit=limit, offset=offset)
        params: dict[str, str] = {
            "filters": json.dumps(filters, separators=(",", ":")),
            "sort": sort,
            "limit": str(limit),
            "offset": str(offset),
            "api_token": api_token,
            "fmt": self.config.default_fmt,
        }
        if signals:
            params["signals"] = json.dumps(signals, separators=(",", ":"))
        return self._client.build_request(
            "GET",
            f"{self.config.base_url.rstrip('/')}/screener",
            params=params,
        )

    def fetch_screener_page(
        self,
        *,
        filters: list[list[Any]],
        signals: list[str],
        sort: str,
        limit: int,
        offset: int,
    ) -> Any:
        """Fetch one EODHD screener page."""

        request = self.build_screener_request(
            filters=filters,
            signals=signals,
            sort=sort,
            limit=limit,
            offset=offset,
            api_token=self.require_token(),
        )
        return self._send_json(request)

    def fetch_eod(
        self,
        *,
        symbol: str,
        from_date: str,
        to_date: str,
        period: str,
    ) -> Any:
        """Fetch EOD historical rows from EODHD."""

        validate_date_range(from_date, to_date)
        request = self.build_eod_request(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            period=period,
            api_token=self.require_token(),
        )
        return self._send_json(request)

    def fetch_intraday(
        self,
        *,
        symbol: str,
        from_date: str,
        to_date: str,
        interval: str,
    ) -> list[Any]:
        """Fetch intraday rows over a safely chunked range."""

        rows: list[Any] = []
        for start, end in chunk_intraday_range(from_date, to_date, interval=interval):
            payload = self.fetch_intraday_chunk(
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
            )
            rows.extend(_records_from_response(payload))
        return rows

    def fetch_intraday_chunk(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> Any:
        """Fetch one intraday chunk."""

        request = self.build_intraday_request(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            api_token=self.require_token(),
        )
        return self._send_json(request)

    def _send_json(self, request: httpx.Request) -> Any:
        max_attempts = max(1, self.config.max_retries)
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.send(request)
            except httpx.RequestError as exc:
                if attempt >= max_attempts:
                    raise EODHDTemporaryHTTPError(
                        _request_error_message(
                            request,
                            error=exc,
                            retries_exhausted=True,
                        )
                    ) from exc
                self._sleep(_backoff_delay(attempt))
                continue

            if response.status_code < 400:
                return response.json()

            if _is_retryable_status(response.status_code):
                if attempt >= max_attempts:
                    error_type = (
                        EODHDRateLimitError
                        if response.status_code == 429
                        else EODHDTemporaryHTTPError
                    )
                    raise error_type(
                        _http_error_message(
                            response,
                            retries_exhausted=True,
                        )
                    )
                self._sleep(_retry_delay(response, attempt))
                continue

            raise EODHDPermanentHTTPError(
                _http_error_message(
                    response,
                    retries_exhausted=False,
                )
            )

        raise EODHDError("EODHD retry loop exhausted without a response")


def _validate_screener_page(*, limit: int, offset: int) -> None:
    if limit < 1 or limit > SCREENER_MAX_LIMIT:
        raise EODHDUnsupportedScreenerError(
            f"EODHD screener page limit must be between 1 and {SCREENER_MAX_LIMIT}"
        )
    if offset < 0 or offset > SCREENER_MAX_OFFSET:
        raise EODHDUnsupportedScreenerError(
            f"EODHD screener offset must be between 0 and {SCREENER_MAX_OFFSET}"
        )


def build_screener_filters(
    *,
    exchange: str | None = None,
    min_price: float | None = None,
    min_market_cap: float | None = None,
    min_avgvol_200d: float | None = None,
    sectors: list[str] | None = None,
    industries: list[str] | None = None,
) -> list[list[Any]]:
    """Build first-pass EODHD screener filters from user-facing options."""

    filters: list[list[Any]] = []
    if exchange:
        filters.append(["exchange", "=", exchange])
    if min_price is not None:
        filters.append(["adjusted_close", ">=", min_price])
    if min_market_cap is not None:
        filters.append(["market_capitalization", ">=", min_market_cap])
    if min_avgvol_200d is not None:
        filters.append(["avgvol_200d", ">=", min_avgvol_200d])
    for sector in sectors or []:
        filters.append(["sector", "=", sector])
    for industry in industries or []:
        filters.append(["industry", "=", industry])
    return filters


def normalize_screener_response(payload: Any) -> list[UniverseSymbol]:
    """Normalize an EODHD screener payload to universe symbols."""

    records = _records_from_response(payload)
    symbols: list[UniverseSymbol] = []
    for record in records:
        code = record.get("code") or record.get("symbol")
        if code is None or not str(code).strip():
            raise EODHDSchemaError("EODHD screener response row is missing code/symbol")
        symbols.append(
            UniverseSymbol(
                symbol=str(code),
                name=_optional_str(record.get("name")),
                exchange=_optional_str(record.get("exchange")),
                currency=_optional_str(record.get("currency")),
                instrument_type="stock",
                sector=_optional_str(record.get("sector")),
                industry=_optional_str(record.get("industry")),
                market_capitalization=_optional_float(record.get("market_capitalization")),
                adjusted_close=_optional_float(record.get("adjusted_close")),
                avgvol_200d=_optional_float(record.get("avgvol_200d")),
            )
        )
    return symbols


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def fetch_screener_all(
    *,
    client: EODHDClient,
    filters: list[list[Any]],
    signals: list[str],
    sort: str,
    limit: int,
    max_pages: int,
) -> list[UniverseSymbol]:
    """Fetch EODHD screener pages up to total limit and max page guardrails."""

    if limit < 1:
        raise EODHDUnsupportedScreenerError("EODHD screener limit must be positive")
    if max_pages < 1:
        raise EODHDUnsupportedScreenerError("EODHD screener max_pages must be positive")
    rows: list[UniverseSymbol] = []
    offset = 0
    pages = 0
    requested = 0
    while requested < limit and pages < max_pages:
        page_limit = min(SCREENER_MAX_LIMIT, limit - requested)
        _validate_screener_page(limit=page_limit, offset=offset)
        payload = client.fetch_screener_page(
            filters=filters,
            signals=signals,
            sort=sort,
            limit=page_limit,
            offset=offset,
        )
        page_rows = normalize_screener_response(payload)
        if not page_rows:
            break
        rows.extend(page_rows)
        pages += 1
        requested += page_limit
        offset += page_limit
        if offset > SCREENER_MAX_OFFSET and requested < limit and pages < max_pages:
            raise EODHDUnsupportedScreenerError(
                f"EODHD screener offset would exceed {SCREENER_MAX_OFFSET}"
            )
    return rows[:limit]


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {429, 500, 502, 503, 504}


def _backoff_delay(attempt: int) -> float:
    delay = 0.25 * float(2 ** (attempt - 1))
    return float(min(2.0, max(0.25, delay)))


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(0.0, float(str(retry_after)))
        except ValueError:
            pass
    return _backoff_delay(attempt)


def _safe_response_preview(response: httpx.Response, *, limit: int = 300) -> str:
    preview = response.text[:limit]
    token = response.request.url.params.get("api_token")
    if token:
        preview = preview.replace(token, "<redacted>")
    return preview


def _http_error_message(response: httpx.Response, *, retries_exhausted: bool) -> str:
    path = response.request.url.path
    preview = _safe_response_preview(response)
    exhausted = "true" if retries_exhausted else "false"
    return (
        f"EODHD HTTP {response.status_code} for {path}; "
        f"retries exhausted: {exhausted}; response preview: {preview}"
    )


def _request_error_message(
    request: httpx.Request,
    *,
    error: httpx.RequestError,
    retries_exhausted: bool,
) -> str:
    exhausted = "true" if retries_exhausted else "false"
    return f"EODHD request failed for {request.url.path}; retries exhausted: {exhausted}; {error}"


def timeframe_for_eod_period(period: str) -> str:
    """Return Stocker's timeframe for an EODHD EOD period."""

    try:
        return EOD_PERIOD_TO_TIMEFRAME[period]
    except KeyError as exc:
        raise EODHDSchemaError(f"Unsupported EODHD EOD period: {period}") from exc


def validate_intraday_interval(interval: str) -> None:
    """Raise when the intraday interval is unsupported."""

    if interval not in INTRADAY_MAX_SPAN_DAYS:
        supported = ", ".join(sorted(INTRADAY_MAX_SPAN_DAYS))
        raise EODHDUnsupportedIntervalError(
            f"Unsupported EODHD intraday interval {interval!r}; supported: {supported}"
        )


def _parse_utc_datetime(value: str) -> datetime:
    parsed = pd.Timestamp(value)
    parsed = parsed.tz_localize(UTC) if parsed.tzinfo is None else parsed.tz_convert(UTC)
    return parsed.to_pydatetime()


def validate_date_range(from_date: str, to_date: str) -> tuple[datetime, datetime]:
    """Parse and validate a UTC date range."""

    start = _parse_utc_datetime(from_date)
    end = _parse_utc_datetime(to_date)
    if end < start:
        raise EODHDInvalidDateRangeError(
            f"Invalid EODHD date range: from {from_date} is after to {to_date}"
        )
    return start, end


def chunk_intraday_range(
    from_date: str,
    to_date: str,
    *,
    interval: str,
) -> list[tuple[datetime, datetime]]:
    """Split an intraday UTC date range into EODHD-safe request chunks."""

    validate_intraday_interval(interval)
    start, end = validate_date_range(from_date, to_date)
    if end <= start:
        raise EODHDInvalidDateRangeError(
            f"Invalid EODHD intraday range: from {from_date} must be before to {to_date}"
        )
    max_span = timedelta(days=INTRADAY_MAX_SPAN_DAYS[interval])
    chunks: list[tuple[datetime, datetime]] = []
    current = start
    while current < end:
        chunk_end = min(current + max_span, end)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def _records_from_response(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("data"), list):
        records = payload["data"]
    else:
        raise EODHDSchemaError("EODHD response must be a list of row objects")
    if not records:
        raise EODHDEmptyResponseError("EODHD response contained no rows")
    if not all(isinstance(row, dict) for row in records):
        raise EODHDSchemaError("EODHD response rows must be objects")
    return cast(list[Mapping[str, Any]], records)


def _require_columns(records: list[Mapping[str, Any]], required: set[str]) -> None:
    available = set().union(*(record.keys() for record in records))
    missing = sorted(required.difference(available))
    if missing:
        raise EODHDSchemaError(f"EODHD response is missing required fields: {', '.join(missing)}")


def _sorted_deduped(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = [column for column in ALL_SCHEMA_COLUMNS if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining]


def normalize_eod_response(
    payload: Any,
    *,
    symbol: str,
    instrument_type: str,
    period: str,
    currency: str = "USD",
) -> pd.DataFrame:
    """Normalize an EODHD EOD response to Stocker's canonical OHLCV schema."""

    records = _records_from_response(payload)
    _require_columns(records, {"date", "open", "high", "low", "close"})
    raw = pd.DataFrame(records)
    frame = pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(raw["date"], errors="coerce", utc=True)
    _raise_if_unparseable_timestamps(frame)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(raw[column], errors="coerce")
    frame["volume"] = pd.to_numeric(raw["volume"], errors="coerce") if "volume" in raw else pd.NA
    if "adjusted_close" in raw:
        frame["adjusted_close"] = pd.to_numeric(raw["adjusted_close"], errors="coerce")
    _add_common_columns(
        frame,
        symbol=symbol,
        instrument_type=instrument_type,
        timeframe=timeframe_for_eod_period(period),
        currency=currency,
    )
    return _order_columns(_sorted_deduped(frame))


def normalize_intraday_response(
    payload: Any,
    *,
    symbol: str,
    instrument_type: str,
    interval: str,
    currency: str = "USD",
) -> pd.DataFrame:
    """Normalize an EODHD intraday response to Stocker's canonical OHLCV schema."""

    validate_intraday_interval(interval)
    records = _records_from_response(payload)
    _require_columns(records, {"open", "high", "low", "close"})
    if not any("timestamp" in record or "datetime" in record for record in records):
        raise EODHDSchemaError("EODHD intraday response is missing timestamp/datetime")
    raw = pd.DataFrame(records)
    frame = pd.DataFrame()
    if "timestamp" in raw:
        frame["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", unit="s", utc=True)
    else:
        frame["timestamp"] = pd.to_datetime(raw["datetime"], errors="coerce", utc=True)
    _raise_if_unparseable_timestamps(frame)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(raw[column], errors="coerce")
    frame["volume"] = pd.to_numeric(raw["volume"], errors="coerce") if "volume" in raw else pd.NA
    _add_common_columns(
        frame,
        symbol=symbol,
        instrument_type=instrument_type,
        timeframe=interval,
        currency=currency,
    )
    return _order_columns(_sorted_deduped(frame))


def _raise_if_unparseable_timestamps(frame: pd.DataFrame) -> None:
    if frame["timestamp"].isna().any():
        raise EODHDSchemaError("EODHD response contains unparseable timestamps")


def _add_common_columns(
    frame: pd.DataFrame,
    *,
    symbol: str,
    instrument_type: str,
    timeframe: str,
    currency: str,
) -> None:
    frame["source"] = "eodhd"
    frame["symbol"] = symbol.upper()
    frame["instrument_type"] = instrument_type
    frame["timeframe"] = timeframe
    frame["currency"] = currency
    frame["timezone"] = "UTC"


def _raw_response_path(
    *,
    data_dir: str | Path,
    endpoint: str,
    symbol: str,
    selector_name: str,
    selector_value: str,
    filename: str,
) -> Path:
    return (
        Path(data_dir).expanduser()
        / "raw"
        / "source=eodhd"
        / f"endpoint={endpoint}"
        / f"symbol={symbol.upper()}"
        / f"{selector_name}={selector_value}"
        / filename
    )


def _write_raw_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _merge_or_replace(
    frame: pd.DataFrame,
    *,
    key: DatasetKey,
    data_dir: str | Path,
    overwrite: bool,
    merge: bool,
) -> pd.DataFrame:
    if overwrite and merge:
        raise EODHDError("Use either overwrite or merge, not both")
    output_path = dataset_path(key, data_dir=data_dir)
    if output_path.exists() and not overwrite and not merge:
        raise EODHDError(
            f"Dataset already exists at {output_path}; pass merge=True or overwrite=True"
        )
    if merge and output_path.exists():
        existing = read_parquet(output_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    return _order_columns(_sorted_deduped(frame))


def _validate_before_write(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    market_calendar: str | None,
) -> list[ValidationIssue]:
    issues = validate_ohlcv(
        frame,
        timeframe=timeframe,
        timezone="UTC",
        require_timezone=True,
        market_calendar=market_calendar,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        messages = "; ".join(f"{issue.code}: {issue.message}" for issue in errors)
        raise EODHDSchemaError(f"EODHD normalized data failed validation: {messages}")
    return issues


def _date_range(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty:
        return None, None
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    min_value = timestamps.min()
    max_value = timestamps.max()
    return (
        None if pd.isna(min_value) else str(min_value),
        None if pd.isna(max_value) else str(max_value),
    )


def _fetch_to_storage_from_payload(
    *,
    payload: Any,
    endpoint: str,
    data_dir: str | Path,
    symbol: str,
    timeframe: str,
    instrument_type: str,
    raw_filename: str,
    normalize: Callable[[Any], pd.DataFrame],
    save_raw: bool,
    overwrite: bool,
    merge: bool,
    audit: bool,
    market_calendar: str | None = None,
    raw_selector_name: str = "timeframe",
    raw_selector_value: str | None = None,
) -> EODHDFetchResult:
    """Normalize one payload and store it through the Stocker data pipeline."""

    raw_paths: list[Path] = []
    if save_raw:
        raw_path = _raw_response_path(
            data_dir=data_dir,
            endpoint=endpoint,
            symbol=symbol,
            selector_name=raw_selector_name,
            selector_value=raw_selector_value or timeframe,
            filename=raw_filename,
        )
        raw_paths.append(_write_raw_json(payload, raw_path))

    fetched_rows = len(_records_from_response(payload))
    normalized = normalize(payload)
    key = DatasetKey(
        source="eodhd",
        instrument_type=instrument_type,
        symbol=symbol.upper(),
        timeframe=timeframe,
    )
    final_frame = _merge_or_replace(
        normalized,
        key=key,
        data_dir=data_dir,
        overwrite=overwrite,
        merge=merge,
    )
    issues = _validate_before_write(
        final_frame,
        timeframe=timeframe,
        market_calendar=market_calendar,
    )
    output_path = dataset_path(key, data_dir=data_dir)
    write_parquet(final_frame, output_path)
    catalog_path = write_catalog(data_dir=data_dir)
    audit_markdown_path: Path | None = None
    audit_json_path: Path | None = None
    if audit:
        audit_result = create_audit_report(
            data_dir=data_dir,
            symbol=symbol,
            timeframe=timeframe,
            source="eodhd",
            instrument_type=instrument_type,
            market_calendar=market_calendar,
        )
        audit_markdown_path = audit_result.markdown_path
        audit_json_path = audit_result.json_path
    min_timestamp, max_timestamp = _date_range(final_frame)
    return EODHDFetchResult(
        output_path=output_path,
        catalog_path=catalog_path,
        raw_paths=raw_paths,
        rows_fetched=fetched_rows,
        rows_saved=int(len(final_frame)),
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        issues=issues,
        audit_markdown_path=audit_markdown_path,
        audit_json_path=audit_json_path,
    )


def fetch_eod_to_storage(
    *,
    client: EODHDClient,
    data_dir: str | Path,
    symbol: str,
    from_date: str,
    to_date: str,
    period: str,
    instrument_type: str,
    currency: str = "USD",
    save_raw: bool = True,
    overwrite: bool = False,
    merge: bool = False,
    audit: bool = False,
    market_calendar: str | None = None,
) -> EODHDFetchResult:
    """Fetch EODHD EOD data and store normalized Parquet."""

    payload = client.fetch_eod(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        period=period,
    )
    timeframe = timeframe_for_eod_period(period)
    return _fetch_to_storage_from_payload(
        payload=payload,
        endpoint="eod",
        data_dir=data_dir,
        symbol=symbol,
        timeframe=timeframe,
        instrument_type=instrument_type,
        raw_filename=f"{from_date}_{to_date}.json",
        normalize=lambda raw_payload: normalize_eod_response(
            raw_payload,
            symbol=symbol,
            instrument_type=instrument_type,
            period=period,
            currency=currency,
        ),
        save_raw=save_raw,
        overwrite=overwrite,
        merge=merge,
        audit=audit,
        market_calendar=market_calendar,
        raw_selector_name="period",
        raw_selector_value=period,
    )


def fetch_intraday_to_storage(
    *,
    client: EODHDClient,
    data_dir: str | Path,
    symbol: str,
    from_date: str,
    to_date: str,
    interval: str,
    instrument_type: str,
    currency: str = "USD",
    save_raw: bool = True,
    overwrite: bool = False,
    merge: bool = False,
    audit: bool = False,
    market_calendar: str | None = None,
) -> EODHDFetchResult:
    """Fetch chunked EODHD intraday data and store normalized Parquet."""

    chunks = chunk_intraday_range(from_date, to_date, interval=interval)
    payloads: list[Any] = []
    raw_paths: list[Path] = []
    for start, end in chunks:
        payload = client.fetch_intraday_chunk(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
        )
        payloads.append(payload)
        if save_raw:
            raw_filename = f"{start.date().isoformat()}_{end.date().isoformat()}.json"
            raw_paths.append(
                _write_raw_json(
                    payload,
                    _raw_response_path(
                        data_dir=data_dir,
                        endpoint="intraday",
                        symbol=symbol,
                        selector_name="interval",
                        selector_value=interval,
                        filename=raw_filename,
                    ),
                )
            )

    rows: list[Mapping[str, Any]] = []
    for payload in payloads:
        rows.extend(_records_from_response(payload))

    result = _fetch_to_storage_from_payload(
        payload=rows,
        endpoint="intraday",
        data_dir=data_dir,
        symbol=symbol,
        timeframe=interval,
        instrument_type=instrument_type,
        raw_filename=f"{from_date}_{to_date}.json",
        normalize=lambda raw_payload: normalize_intraday_response(
            raw_payload,
            symbol=symbol,
            instrument_type=instrument_type,
            interval=interval,
            currency=currency,
        ),
        save_raw=False,
        overwrite=overwrite,
        merge=merge,
        audit=audit,
        market_calendar=market_calendar,
    )
    return EODHDFetchResult(
        output_path=result.output_path,
        catalog_path=result.catalog_path,
        raw_paths=raw_paths,
        rows_fetched=len(rows),
        rows_saved=result.rows_saved,
        min_timestamp=result.min_timestamp,
        max_timestamp=result.max_timestamp,
        issues=result.issues,
        audit_markdown_path=result.audit_markdown_path,
        audit_json_path=result.audit_json_path,
    )


def plan_eod_fetch(
    *,
    symbol: str,
    from_date: str,
    to_date: str,
    period: str,
    instrument_type: str,
    data_dir: str | Path,
    save_raw: bool,
) -> EODHDFetchPlan:
    """Build dry-run details for an EOD fetch."""

    validate_date_range(from_date, to_date)
    timeframe = timeframe_for_eod_period(period)
    key = DatasetKey(
        source="eodhd",
        instrument_type=instrument_type,
        symbol=symbol.upper(),
        timeframe=timeframe,
    )
    raw_paths = [
        _raw_response_path(
            data_dir=data_dir,
            endpoint="eod",
            symbol=symbol,
            selector_name="period",
            selector_value=period,
            filename=f"{from_date}_{to_date}.json",
        )
    ]
    return EODHDFetchPlan(
        endpoint=f"/eod/{symbol.upper()}",
        params={"from": from_date, "to": to_date, "period": period, "fmt": "json"},
        timeframe=timeframe,
        output_path=dataset_path(key, data_dir=data_dir),
        save_raw=save_raw,
        raw_paths=raw_paths if save_raw else [],
        chunks=[],
    )


def plan_intraday_fetch(
    *,
    symbol: str,
    from_date: str,
    to_date: str,
    interval: str,
    instrument_type: str,
    data_dir: str | Path,
    save_raw: bool,
) -> EODHDFetchPlan:
    """Build dry-run details for an intraday fetch."""

    chunks = chunk_intraday_range(from_date, to_date, interval=interval)
    key = DatasetKey(
        source="eodhd",
        instrument_type=instrument_type,
        symbol=symbol.upper(),
        timeframe=interval,
    )
    raw_paths = [
        _raw_response_path(
            data_dir=data_dir,
            endpoint="intraday",
            symbol=symbol,
            selector_name="interval",
            selector_value=interval,
            filename=f"{start.date().isoformat()}_{end.date().isoformat()}.json",
        )
        for start, end in chunks
    ]
    return EODHDFetchPlan(
        endpoint=f"/intraday/{symbol.upper()}",
        params={"interval": interval, "fmt": "json"},
        timeframe=interval,
        output_path=dataset_path(key, data_dir=data_dir),
        save_raw=save_raw,
        raw_paths=raw_paths if save_raw else [],
        chunks=chunks,
    )
