"""EODHD data-vendor adapter for Stocker's local data pipeline."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pandas as pd
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from stocker_core.config import EODHDConfig
from stocker_data.audit import create_audit_report
from stocker_data.catalog import write_catalog
from stocker_data.schema import ALL_SCHEMA_COLUMNS
from stocker_data.storage import DatasetKey, dataset_path, read_parquet, write_parquet
from stocker_data.validate import ValidationIssue, validate_ohlcv

EOD_PERIOD_TO_TIMEFRAME: dict[str, str] = {"d": "1d", "w": "1w", "m": "1mo"}
INTRADAY_MAX_SPAN_DAYS: dict[str, int] = {"1m": 120, "5m": 600, "1h": 7200}


class EODHDError(Exception):
    """Base class for EODHD adapter errors."""


class EODHDMissingTokenError(EODHDError):
    """Raised when a live EODHD request lacks an API token."""


class EODHDHTTPError(EODHDError):
    """Raised when EODHD returns a non-success HTTP response."""


class EODHDEmptyResponseError(EODHDError):
    """Raised when EODHD returns no rows."""


class EODHDSchemaError(EODHDError):
    """Raised when an EODHD payload cannot be normalized."""


class EODHDUnsupportedIntervalError(EODHDError):
    """Raised when an intraday interval is not supported."""


class EODHDInvalidDateRangeError(EODHDError):
    """Raised when a requested date range is invalid."""


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
    ) -> None:
        self.config = config or EODHDConfig()
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
        retryer = Retrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=0.25, min=0.25, max=2.0),
            retry=retry_if_exception_type(httpx.RequestError),
            reraise=True,
        )
        for attempt in retryer:
            with attempt:
                response = self._client.send(request)
                if response.status_code >= 400:
                    raise EODHDHTTPError(
                        f"EODHD HTTP {response.status_code}: {response.text[:300]}"
                    )
                return response.json()
        raise EODHDError("EODHD retry loop exhausted without a response")


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


def _validate_before_write(frame: pd.DataFrame, *, timeframe: str) -> list[ValidationIssue]:
    issues = validate_ohlcv(frame, timeframe=timeframe, timezone="UTC", require_timezone=True)
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
    issues = _validate_before_write(final_frame, timeframe=timeframe)
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
