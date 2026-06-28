"""CSV ingestion for local OHLCV research datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from stocker_data.catalog import write_catalog
from stocker_data.schema import ALL_SCHEMA_COLUMNS, MarketDataSpec
from stocker_data.storage import DataLayer, DatasetKey, dataset_path, write_parquet
from stocker_data.validate import ValidationIssue, validate_ohlcv

ColumnMap = dict[str, str]

COMMON_COLUMN_NAMES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "datetime", "date", "time"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "last"),
    "volume": ("volume", "vol", "v"),
    "bid": ("bid",),
    "ask": ("ask",),
    "spread": ("spread",),
    "spread_bps": ("spread_bps", "spreadbps"),
    "adjusted_close": ("adjusted_close", "adjusted close", "adj_close", "adj close"),
    "corporate_action_flag": ("corporate_action_flag", "corporate action flag"),
    "session": ("session",),
}


@dataclass(frozen=True)
class CsvImportResult:
    """Result of one CSV import."""

    path: Path
    catalog_path: Path
    rows: int
    issues: list[ValidationIssue]

    @property
    def error_count(self) -> int:
        """Return validation error count."""

        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        """Return validation warning count."""

        return sum(1 for issue in self.issues if issue.severity == "warning")


def ingest_not_configured_message(data_dir: Path) -> str:
    """Return a clear message for commands that try to ingest before a source exists."""

    return f"No market data ingestion source is configured. Raw data directory: {data_dir}"


def _normalize_column_name(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def parse_column_mapping(mapping: str | None) -> ColumnMap:
    """Parse explicit column mapping in `canonical=source,canonical=source` form."""

    if mapping is None or not mapping.strip():
        return {}
    parsed: ColumnMap = {}
    for part in mapping.split(","):
        if "=" not in part:
            raise ValueError(f"Invalid column mapping entry: {part}")
        canonical, source = part.split("=", 1)
        parsed[canonical.strip()] = source.strip()
    return parsed


def infer_column_mapping(columns: list[str], explicit: ColumnMap | None = None) -> ColumnMap:
    """Infer common OHLCV column names, allowing explicit overrides."""

    normalized_lookup = {_normalize_column_name(column): column for column in columns}
    mapping: ColumnMap = {}
    for canonical, candidates in COMMON_COLUMN_NAMES.items():
        for candidate in candidates:
            match = normalized_lookup.get(_normalize_column_name(candidate))
            if match is not None:
                mapping[canonical] = match
                break
    if explicit:
        mapping.update(explicit)
    return mapping


def _parse_timestamp_column(series: pd.Series, timezone: str) -> pd.Series:
    zone = ZoneInfo(timezone)
    values = series.dropna().astype(str)
    has_explicit_offset = values.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any()
    if has_explicit_offset:
        parsed_utc = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed_utc.dt.tz_convert(zone)
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        return parsed
    try:
        current_tz = getattr(parsed.dt, "tz", None)
    except AttributeError:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_convert(zone)
    if current_tz is None:
        return parsed.dt.tz_localize(zone)
    return parsed.dt.tz_convert(zone)


def _clean_csv_frame(
    raw: pd.DataFrame,
    *,
    spec: MarketDataSpec,
    column_mapping: ColumnMap,
) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required.difference(column_mapping))
    if missing:
        raise ValueError(f"Missing required columns after mapping: {', '.join(missing)}")

    cleaned = pd.DataFrame()
    cleaned["timestamp"] = _parse_timestamp_column(raw[column_mapping["timestamp"]], spec.timezone)
    for column in ("open", "high", "low", "close"):
        cleaned[column] = pd.to_numeric(raw[column_mapping[column]], errors="coerce")

    if "volume" in column_mapping:
        cleaned["volume"] = pd.to_numeric(raw[column_mapping["volume"]], errors="coerce")

    for optional in (
        "bid",
        "ask",
        "spread",
        "spread_bps",
        "adjusted_close",
        "corporate_action_flag",
        "session",
    ):
        if optional in column_mapping:
            cleaned[optional] = raw[column_mapping[optional]]

    cleaned["source"] = spec.source
    cleaned["symbol"] = spec.normalized_symbol()
    cleaned["instrument_type"] = spec.instrument_type
    cleaned["timeframe"] = spec.timeframe
    cleaned["currency"] = spec.currency
    cleaned["timezone"] = spec.timezone

    ordered_columns = [column for column in ALL_SCHEMA_COLUMNS if column in cleaned.columns]
    cleaned = cleaned[ordered_columns].sort_values("timestamp").reset_index(drop=True)
    return cleaned


def import_csv(
    *,
    file_path: str | Path,
    data_dir: str | Path = "data",
    symbol: str,
    source: str,
    timeframe: str,
    instrument_type: str,
    timezone: str,
    currency: str = "USD",
    layer: DataLayer = "processed",
    column_mapping: ColumnMap | str | None = None,
    fail_on_error: bool = True,
) -> CsvImportResult:
    """Import a local CSV into canonical partitioned Parquet storage."""

    path = Path(file_path).expanduser()
    explicit_mapping = (
        parse_column_mapping(column_mapping) if isinstance(column_mapping, str) else column_mapping
    )
    raw = pd.read_csv(path)
    mapping = infer_column_mapping(list(raw.columns), explicit=explicit_mapping)
    spec = MarketDataSpec(
        source=source,
        symbol=symbol,
        instrument_type=instrument_type,
        timeframe=timeframe,
        timezone=timezone,
        currency=currency,
    )
    cleaned = _clean_csv_frame(raw, spec=spec, column_mapping=mapping)
    issues = validate_ohlcv(cleaned, timeframe=timeframe, timezone=timezone, require_timezone=True)
    if fail_on_error and any(issue.severity == "error" for issue in issues):
        messages = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise ValueError(f"CSV import failed validation: {messages}")

    key = DatasetKey(
        source=source,
        instrument_type=instrument_type,
        symbol=spec.normalized_symbol(),
        timeframe=timeframe,
    )
    output_path = dataset_path(key, data_dir=data_dir, layer=layer)
    write_parquet(cleaned, output_path)
    catalog = write_catalog(data_dir=data_dir, layer=layer)
    return CsvImportResult(
        path=output_path,
        catalog_path=catalog,
        rows=int(len(cleaned)),
        issues=issues,
    )
