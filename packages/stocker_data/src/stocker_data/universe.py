"""Universe definitions, batch data orchestration, and research-ready exports."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_core.config import EODHDConfig
from stocker_data.storage import DatasetKey, dataset_path, load_dataset
from stocker_data.validate import validate_ohlcv

UniverseIssueSeverity = Literal["info", "warning", "error"]
UniverseFetchStatus = Literal["planned", "fetched", "skipped", "failed"]


class UniverseValidationIssue(BaseModel):
    """One validation issue for a universe definition."""

    severity: UniverseIssueSeverity
    code: str
    message: str
    symbol: str | None = None


class UniverseSymbol(BaseModel):
    """One symbol and optional vendor metadata inside a universe."""

    model_config = ConfigDict(extra="allow")

    symbol: str
    name: str | None = None
    exchange: str | None = None
    currency: str | None = None
    instrument_type: str = "stock"
    sector: str | None = None
    industry: str | None = None
    market_capitalization: float | None = None
    adjusted_close: float | None = None
    avgvol_200d: float | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        """Preserve EODHD suffixes while enforcing uppercase symbols."""

        return value.strip().upper()


class UniverseFilters(BaseModel):
    """Screener or manual filters that produced a universe."""

    exchange: str | None = None
    min_price: float | None = None
    min_market_cap: float | None = None
    min_avgvol_200d: float | None = None
    sectors: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)


class UniverseDefinition(BaseModel):
    """A reproducible stock universe definition."""

    id: str
    name: str
    description: str = ""
    source: str
    created_at: str
    symbols: list[UniverseSymbol]
    filters: UniverseFilters = Field(default_factory=UniverseFilters)


class UniverseBuildResult(BaseModel):
    """Result of building a universe from a vendor screener."""

    universe_id: str
    output_path: str
    symbol_count: int
    report_markdown_path: str
    report_json_path: str


class UniverseFetchItemResult(BaseModel):
    """Fetch outcome for one universe symbol."""

    symbol: str
    status: UniverseFetchStatus
    rows_fetched: int = 0
    rows_saved: int = 0
    min_timestamp: str | None = None
    max_timestamp: str | None = None
    output_path: str | None = None
    audit_markdown_path: str | None = None
    audit_json_path: str | None = None
    qa_status: str | None = None
    qa_markdown_path: str | None = None
    qa_json_path: str | None = None
    error_message: str | None = None
    duration_seconds: float = 0.0
    skip_reason: str | None = None


class UniverseFetchResult(BaseModel):
    """Batch fetch report for a universe."""

    universe_id: str
    timeframe: str
    source: str
    started_at: str
    completed_at: str
    items: list[UniverseFetchItemResult]
    markdown_path: str
    json_path: str

    @property
    def fetched_count(self) -> int:
        """Return number of fetched symbols."""

        return sum(1 for item in self.items if item.status == "fetched")

    @property
    def failed_count(self) -> int:
        """Return number of failed symbols."""

        return sum(1 for item in self.items if item.status == "failed")


class UniverseQualificationRules(BaseModel):
    """Local data rules for admitting symbols into research-ready universes."""

    min_history_days: int = Field(default=750, ge=0)
    min_last_close: float = Field(default=5.0, ge=0.0)
    min_median_dollar_volume_60d: float = Field(default=10_000_000.0, ge=0.0)
    max_validation_errors: int = Field(default=0, ge=0)
    max_missing_session_warnings: int = Field(default=5, ge=0)


class QualifiedSymbol(BaseModel):
    """Qualified symbol summary."""

    symbol: str


class RejectedUniverseSymbol(BaseModel):
    """Rejected symbol and concrete reasons."""

    symbol: str
    reasons: list[str]


class UniverseQualificationResult(BaseModel):
    """Qualification result for one universe/timeframe/source."""

    universe_id: str
    timeframe: str
    source: str
    qualified_symbols: list[str]
    rejected_symbols: list[RejectedUniverseSymbol]
    rules: UniverseQualificationRules
    created_at: str
    output_path: str
    markdown_path: str
    json_path: str


class QualifiedUniverse(BaseModel):
    """JSON payload handed to future Stage 3 universe research."""

    universe_id: str
    timeframe: str
    source: str
    qualified_symbols: list[QualifiedSymbol]
    rejected_symbols: list[RejectedUniverseSymbol]
    rules: UniverseQualificationRules
    created_at: str


class UniverseHealthReport(BaseModel):
    """Universe data health summary."""

    universe_id: str
    timeframe: str
    source: str
    total_symbols: int
    datasets_present: int
    datasets_missing: int
    fetched_successfully: int
    failed_fetches: int
    audit_pass: int
    audit_warning: int
    audit_fail: int
    qa_pass: int
    qa_warning: int
    qa_fail: int
    average_row_count: float
    min_timestamp: str | None
    max_timestamp: str | None
    research_ready_count: int
    rejected_count: int
    top_rejection_reasons: dict[str, int]
    next_recommended_command: str
    markdown_path: Path
    json_path: Path


@dataclass(frozen=True)
class BatchFetchOptions:
    """Options for batch universe fetch orchestration."""

    from_date: str
    to_date: str
    timeframe: str
    source: str = "eodhd"
    instrument_type: str = "stock"
    currency: str = "USD"
    merge: bool = False
    overwrite: bool = False
    audit: bool = False
    qa: bool = False
    market_calendar: str | None = None
    max_symbols: int | None = None
    fail_fast: bool = False
    sleep_seconds_between_symbols: float = 0.0
    resume: bool = False
    skip_existing: bool = False
    dry_run: bool = False


def _now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reports_dir(data_dir: str | Path) -> Path:
    return Path(data_dir).expanduser() / "reports" / "universes"


def _safe_yaml_dump(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def load_universe(path: str | Path) -> UniverseDefinition:
    """Load a YAML or JSON universe file."""

    universe_path = Path(path)
    raw_text = universe_path.read_text(encoding="utf-8")
    if universe_path.suffix.lower() == ".json":
        payload = json.loads(raw_text)
    else:
        payload = yaml.safe_load(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Universe file must contain a mapping: {universe_path}")
    return UniverseDefinition.model_validate(payload)


def _sorted_symbols(symbols: Iterable[UniverseSymbol]) -> list[UniverseSymbol]:
    return sorted(symbols, key=lambda item: item.symbol)


def save_universe(universe: UniverseDefinition, path: str | Path) -> Path:
    """Save a universe deterministically as YAML."""

    output_path = Path(path)
    payload = universe.model_copy(update={"symbols": _sorted_symbols(universe.symbols)}).model_dump(
        mode="json",
        exclude_none=True,
    )
    return _safe_yaml_dump(payload, output_path)


def validate_universe(universe: UniverseDefinition) -> list[UniverseValidationIssue]:
    """Return structured validation issues for a universe definition."""

    issues: list[UniverseValidationIssue] = []
    if not universe.id.strip():
        issues.append(
            UniverseValidationIssue(
                severity="error",
                code="missing_id",
                message="Missing id",
            )
        )
    if not universe.name.strip():
        issues.append(
            UniverseValidationIssue(severity="error", code="missing_name", message="Missing name")
        )
    if not universe.source.strip():
        issues.append(
            UniverseValidationIssue(
                severity="error",
                code="missing_source",
                message="Missing source",
            )
        )
    if not universe.symbols:
        issues.append(
            UniverseValidationIssue(
                severity="error",
                code="missing_symbols",
                message="Universe must contain at least one symbol",
            )
        )

    counts = Counter(symbol.symbol for symbol in universe.symbols)
    for duplicate_symbol, count in sorted(counts.items()):
        if count > 1:
            issues.append(
                UniverseValidationIssue(
                    severity="error",
                    code="duplicate_symbol",
                    message=f"Duplicate symbol appears {count} times",
                    symbol=duplicate_symbol,
                )
            )

    for symbol in universe.symbols:
        if symbol.symbol != symbol.symbol.upper():
            issues.append(
                UniverseValidationIssue(
                    severity="error",
                    code="symbol_not_uppercase",
                    message="Symbol must be uppercase",
                    symbol=symbol.symbol,
                )
            )
        if symbol.market_capitalization is not None and symbol.market_capitalization < 0:
            issues.append(
                UniverseValidationIssue(
                    severity="warning",
                    code="negative_market_capitalization",
                    message="Market capitalization should be non-negative",
                    symbol=symbol.symbol,
                )
            )
        if symbol.adjusted_close is not None and symbol.adjusted_close <= 0:
            issues.append(
                UniverseValidationIssue(
                    severity="warning",
                    code="non_positive_adjusted_close",
                    message="Adjusted close should be positive",
                    symbol=symbol.symbol,
                )
            )
        if symbol.avgvol_200d is not None and symbol.avgvol_200d < 0:
            issues.append(
                UniverseValidationIssue(
                    severity="warning",
                    code="negative_avgvol_200d",
                    message="Average volume should be non-negative",
                    symbol=symbol.symbol,
                )
            )
    return issues


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def write_universe_build_report(
    *,
    universe: UniverseDefinition,
    output_path: str | Path,
    data_dir: str | Path,
    filters: dict[str, Any],
) -> UniverseBuildResult:
    """Write Markdown/JSON report for a universe build."""

    report_dir = _reports_dir(data_dir)
    stem = f"{universe.id}_build"
    json_path = report_dir / f"{stem}.json"
    markdown_path = report_dir / f"{stem}.md"
    payload = {
        "universe_id": universe.id,
        "name": universe.name,
        "source": universe.source,
        "output_path": str(output_path),
        "symbol_count": len(universe.symbols),
        "filters": filters,
        "symbols": [
            symbol.model_dump(mode="json", exclude_none=True) for symbol in universe.symbols
        ],
        "created_at": _now_iso(),
    }
    _write_json(json_path, payload)
    rows = [
        [symbol.symbol, symbol.name or "", symbol.exchange or ""] for symbol in universe.symbols
    ]
    _write_markdown(
        markdown_path,
        f"""# Universe Build: {universe.id}

## Summary

- Name: {universe.name}
- Source: {universe.source}
- Symbols: {len(universe.symbols)}
- Output: `{output_path}`

## Symbols

{_markdown_table(["Symbol", "Name", "Exchange"], rows)}
""",
    )
    return UniverseBuildResult(
        universe_id=universe.id,
        output_path=str(output_path),
        symbol_count=len(universe.symbols),
        report_markdown_path=str(markdown_path),
        report_json_path=str(json_path),
    )


def _timeframe_to_eod_period(timeframe: str) -> str | None:
    return {"1d": "d", "1w": "w", "1mo": "m"}.get(timeframe)


def _previous_fetch_successes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list):
        return set()
    return {
        str(item.get("symbol", "")).upper()
        for item in items
        if isinstance(item, dict) and item.get("status") == "fetched"
    }


def _fetch_report_paths(
    *,
    universe_id: str,
    timeframe: str,
    data_dir: str | Path,
    suffix: str,
) -> tuple[Path, Path]:
    report_dir = _reports_dir(data_dir)
    stem = f"{universe_id}_{timeframe}_{suffix}"
    return report_dir / f"{stem}.md", report_dir / f"{stem}.json"


def _fetch_markdown(payload: dict[str, Any]) -> str:
    rows = [
        [
            item["symbol"],
            item["status"],
            item.get("rows_saved", 0),
            item.get("min_timestamp") or "",
            item.get("max_timestamp") or "",
            item.get("error_message") or item.get("skip_reason") or "",
        ]
        for item in payload["items"]
    ]
    return f"""# Universe Fetch: {payload["universe_id"]} {payload["timeframe"]}

## Summary

- Source: `{payload["source"]}`
- Started: {payload["started_at"]}
- Completed: {payload["completed_at"]}
- Symbols: {len(payload["items"])}

## Results

{_markdown_table(["Symbol", "Status", "Rows Saved", "Min", "Max", "Message"], rows)}
"""


def run_universe_fetch(
    *,
    universe: UniverseDefinition,
    data_dir: str | Path,
    options: BatchFetchOptions,
    eodhd_config: EODHDConfig,
) -> UniverseFetchResult:
    """Fetch history for a universe through the configured vendor adapter."""

    if options.source != "eodhd":
        raise ValueError("Universe fetch currently supports only source=eodhd")
    if options.overwrite and options.merge:
        raise ValueError("Use either overwrite or merge, not both")

    from stocker_data.vendors import eodhd
    from stocker_data.vendors.eodhd_qa import create_eodhd_qa_report

    markdown_path, json_path = _fetch_report_paths(
        universe_id=universe.id,
        timeframe=options.timeframe,
        data_dir=data_dir,
        suffix="fetch",
    )
    prior_successes = _previous_fetch_successes(json_path) if options.resume else set()
    symbols = _sorted_symbols(universe.symbols)
    if options.max_symbols is not None:
        symbols = symbols[: options.max_symbols]

    started_at = _now_iso()
    items: list[UniverseFetchItemResult] = []
    client = eodhd.EODHDClient(config=eodhd_config)
    for symbol in symbols:
        start = time.monotonic()
        key = DatasetKey(
            source=options.source,
            instrument_type=symbol.instrument_type or options.instrument_type,
            symbol=symbol.symbol,
            timeframe=options.timeframe,
        )
        output_path = dataset_path(key, data_dir=data_dir)
        if options.dry_run:
            items.append(
                UniverseFetchItemResult(
                    symbol=symbol.symbol,
                    status="planned",
                    output_path=str(output_path),
                    duration_seconds=0.0,
                )
            )
            continue
        if symbol.symbol in prior_successes:
            items.append(
                UniverseFetchItemResult(
                    symbol=symbol.symbol,
                    status="skipped",
                    output_path=str(output_path),
                    skip_reason="resume_success",
                )
            )
            continue
        if options.skip_existing and output_path.exists():
            items.append(
                UniverseFetchItemResult(
                    symbol=symbol.symbol,
                    status="skipped",
                    output_path=str(output_path),
                    skip_reason="existing_dataset",
                )
            )
            continue

        try:
            period = _timeframe_to_eod_period(options.timeframe)
            if period is not None:
                result = eodhd.fetch_eod_to_storage(
                    client=client,
                    data_dir=data_dir,
                    symbol=symbol.symbol,
                    from_date=options.from_date,
                    to_date=options.to_date,
                    period=period,
                    instrument_type=symbol.instrument_type or options.instrument_type,
                    currency=symbol.currency or options.currency,
                    save_raw=True,
                    overwrite=options.overwrite,
                    merge=options.merge,
                    audit=options.audit,
                    market_calendar=options.market_calendar,
                )
            else:
                result = eodhd.fetch_intraday_to_storage(
                    client=client,
                    data_dir=data_dir,
                    symbol=symbol.symbol,
                    from_date=options.from_date,
                    to_date=options.to_date,
                    interval=options.timeframe,
                    instrument_type=symbol.instrument_type or options.instrument_type,
                    currency=symbol.currency or options.currency,
                    save_raw=True,
                    overwrite=options.overwrite,
                    merge=options.merge,
                    audit=options.audit,
                    market_calendar=options.market_calendar,
                )
            qa_status: str | None = None
            qa_markdown_path: str | None = None
            qa_json_path: str | None = None
            if options.qa:
                qa_result = create_eodhd_qa_report(
                    data_dir=data_dir,
                    symbol=symbol.symbol,
                    timeframe=options.timeframe,
                    instrument_type=symbol.instrument_type or options.instrument_type,
                    market_calendar=options.market_calendar,
                    require_raw=True,
                )
                qa_status = qa_result.status
                qa_markdown_path = str(qa_result.markdown_path)
                qa_json_path = str(qa_result.json_path)
            items.append(
                UniverseFetchItemResult(
                    symbol=symbol.symbol,
                    status="fetched",
                    rows_fetched=result.rows_fetched,
                    rows_saved=result.rows_saved,
                    min_timestamp=result.min_timestamp,
                    max_timestamp=result.max_timestamp,
                    output_path=str(result.output_path),
                    audit_markdown_path=str(result.audit_markdown_path)
                    if result.audit_markdown_path is not None
                    else None,
                    audit_json_path=str(result.audit_json_path)
                    if result.audit_json_path is not None
                    else None,
                    qa_status=qa_status,
                    qa_markdown_path=qa_markdown_path,
                    qa_json_path=qa_json_path,
                    duration_seconds=round(time.monotonic() - start, 6),
                )
            )
        except Exception as exc:
            item = UniverseFetchItemResult(
                symbol=symbol.symbol,
                status="failed",
                output_path=str(output_path),
                error_message=str(exc),
                duration_seconds=round(time.monotonic() - start, 6),
            )
            items.append(item)
            payload = _fetch_payload(
                universe_id=universe.id,
                timeframe=options.timeframe,
                source=options.source,
                started_at=started_at,
                completed_at=_now_iso(),
                items=items,
            )
            _write_fetch_report(markdown_path, json_path, payload)
            if options.fail_fast:
                raise RuntimeError(str(exc)) from exc
        if options.sleep_seconds_between_symbols > 0:
            time.sleep(options.sleep_seconds_between_symbols)

    payload = _fetch_payload(
        universe_id=universe.id,
        timeframe=options.timeframe,
        source=options.source,
        started_at=started_at,
        completed_at=_now_iso(),
        items=items,
    )
    _write_fetch_report(markdown_path, json_path, payload)
    return UniverseFetchResult(
        universe_id=universe.id,
        timeframe=options.timeframe,
        source=options.source,
        started_at=payload["started_at"],
        completed_at=payload["completed_at"],
        items=items,
        markdown_path=str(markdown_path),
        json_path=str(json_path),
    )


def _fetch_payload(
    *,
    universe_id: str,
    timeframe: str,
    source: str,
    started_at: str,
    completed_at: str,
    items: list[UniverseFetchItemResult],
) -> dict[str, Any]:
    return {
        "universe_id": universe_id,
        "timeframe": timeframe,
        "source": source,
        "started_at": started_at,
        "completed_at": completed_at,
        "items": [item.model_dump(mode="json", exclude_none=True) for item in items],
    }


def _write_fetch_report(markdown_path: Path, json_path: Path, payload: dict[str, Any]) -> None:
    _write_json(json_path, payload)
    _write_markdown(markdown_path, _fetch_markdown(payload))


def _dataset_stats(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = (
        pd.to_datetime(frame["timestamp"], errors="coerce") if "timestamp" in frame else None
    )
    close = pd.to_numeric(frame["close"], errors="coerce") if "close" in frame else pd.Series()
    volume = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame else pd.Series()
    dollar_volume = close * volume if not close.empty and not volume.empty else pd.Series()
    min_timestamp = None if timestamps is None or timestamps.empty else timestamps.min()
    max_timestamp = None if timestamps is None or timestamps.empty else timestamps.max()
    history_days = 0
    if min_timestamp is not None and max_timestamp is not None:
        history_days = int((max_timestamp - min_timestamp).days)
    return {
        "row_count": int(len(frame)),
        "min_timestamp": None if min_timestamp is None else str(min_timestamp),
        "max_timestamp": None if max_timestamp is None else str(max_timestamp),
        "history_days": history_days,
        "last_close": float(close.iloc[-1]) if not close.empty else 0.0,
        "median_volume_60d": float(volume.tail(60).median()) if not volume.empty else 0.0,
        "median_dollar_volume_60d": float(dollar_volume.tail(60).median())
        if not dollar_volume.empty
        else 0.0,
        "median_dollar_volume_200d": float(dollar_volume.tail(200).median())
        if len(dollar_volume) >= 200
        else None,
    }


def qualify_universe(
    *,
    universe: UniverseDefinition,
    data_dir: str | Path,
    timeframe: str,
    source: str,
    rules: UniverseQualificationRules,
    output_path: str | Path,
    market_calendar: str | None = None,
) -> UniverseQualificationResult:
    """Apply local history/liquidity/data-quality filters and export qualified symbols."""

    qualified: list[str] = []
    rejected: list[RejectedUniverseSymbol] = []
    rows: list[list[Any]] = []
    for symbol in _sorted_symbols(universe.symbols):
        key = DatasetKey(
            source=source,
            instrument_type=symbol.instrument_type,
            symbol=symbol.symbol,
            timeframe=timeframe,
        )
        path = dataset_path(key, data_dir=data_dir)
        reasons: list[str] = []
        if not path.exists():
            rejected.append(
                RejectedUniverseSymbol(symbol=symbol.symbol, reasons=["missing_dataset"])
            )
            rows.append([symbol.symbol, "rejected", "missing_dataset"])
            continue
        frame = load_dataset(key, data_dir=data_dir)
        stats = _dataset_stats(frame)
        issues = validate_ohlcv(
            frame,
            timeframe=timeframe,
            timezone="UTC",
            require_timezone=True,
            market_calendar=market_calendar,
        )
        validation_errors = sum(1 for issue in issues if issue.severity == "error")
        missing_session_warnings = sum(
            issue.count for issue in issues if issue.code == "missing_market_session"
        )
        if stats["history_days"] < rules.min_history_days:
            reasons.append("insufficient_history")
        if stats["last_close"] < rules.min_last_close:
            reasons.append("low_last_close")
        if stats["median_dollar_volume_60d"] < rules.min_median_dollar_volume_60d:
            reasons.append("low_dollar_volume")
        if validation_errors > rules.max_validation_errors:
            reasons.append("validation_errors")
        if missing_session_warnings > rules.max_missing_session_warnings:
            reasons.append("missing_sessions")
        if reasons:
            rejected.append(RejectedUniverseSymbol(symbol=symbol.symbol, reasons=reasons))
            rows.append([symbol.symbol, "rejected", ", ".join(reasons)])
        else:
            qualified.append(symbol.symbol)
            rows.append([symbol.symbol, "qualified", ""])

    created_at = _now_iso()
    output = Path(output_path)
    markdown_path, json_path = _fetch_report_paths(
        universe_id=universe.id,
        timeframe=timeframe,
        data_dir=data_dir,
        suffix="qualification",
    )
    qualified_payload = QualifiedUniverse(
        universe_id=universe.id,
        timeframe=timeframe,
        source=source,
        qualified_symbols=[QualifiedSymbol(symbol=symbol) for symbol in qualified],
        rejected_symbols=rejected,
        rules=rules,
        created_at=created_at,
    ).model_dump(mode="json")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, qualified_payload)
    _write_json(json_path, qualified_payload)
    _write_markdown(
        markdown_path,
        f"""# Universe Qualification: {universe.id} {timeframe}

## Summary

- Source: `{source}`
- Qualified: {len(qualified)}
- Rejected: {len(rejected)}
- Output: `{output}`

## Results

{_markdown_table(["Symbol", "Status", "Reasons"], rows)}
""",
    )
    return UniverseQualificationResult(
        universe_id=universe.id,
        timeframe=timeframe,
        source=source,
        qualified_symbols=qualified,
        rejected_symbols=rejected,
        rules=rules,
        created_at=created_at,
        output_path=str(output),
        markdown_path=str(markdown_path),
        json_path=str(json_path),
    )


def load_research_ready_universe(path: str | Path) -> list[str]:
    """Load a research-ready universe export and return qualified symbols."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    symbols = payload.get("qualified_symbols", [])
    if not isinstance(symbols, list):
        return []
    result: list[str] = []
    for item in symbols:
        if isinstance(item, dict) and "symbol" in item:
            result.append(str(item["symbol"]).upper())
        elif isinstance(item, str):
            result.append(item.upper())
    return result


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else None


def universe_health_report(
    *,
    universe: UniverseDefinition,
    data_dir: str | Path,
    timeframe: str,
    source: str,
) -> UniverseHealthReport:
    """Write a health report for a universe/timeframe/source."""

    row_counts: list[int] = []
    min_values: list[str] = []
    max_values: list[str] = []
    datasets_present = 0
    datasets_missing = 0
    audit_counts = {"pass": 0, "warning": 0, "fail": 0}
    qa_counts = {"pass": 0, "warning": 0, "fail": 0}
    for symbol in _sorted_symbols(universe.symbols):
        key = DatasetKey(
            source=source,
            instrument_type=symbol.instrument_type,
            symbol=symbol.symbol,
            timeframe=timeframe,
        )
        path = dataset_path(key, data_dir=data_dir)
        if path.exists():
            datasets_present += 1
            frame = load_dataset(key, data_dir=data_dir)
            stats = _dataset_stats(frame)
            row_counts.append(int(stats["row_count"]))
            if stats["min_timestamp"] is not None:
                min_values.append(str(stats["min_timestamp"]))
            if stats["max_timestamp"] is not None:
                max_values.append(str(stats["max_timestamp"]))
        else:
            datasets_missing += 1
        audit_path = (
            Path(data_dir).expanduser()
            / "reports"
            / "audits"
            / f"{symbol.symbol}_{timeframe}_audit.json"
        )
        audit = _load_json_if_exists(audit_path)
        if audit is not None:
            if audit.get("passed") is True:
                audit_counts["pass"] += 1
            elif audit.get("issue_counts", {}).get("error", 0):
                audit_counts["fail"] += 1
            else:
                audit_counts["warning"] += 1
        qa_path = (
            Path(data_dir).expanduser()
            / "reports"
            / "vendor_qa"
            / f"{symbol.symbol}_{timeframe}_eodhd_qa.json"
        )
        qa = _load_json_if_exists(qa_path)
        if qa is not None and qa.get("status") in qa_counts:
            qa_counts[str(qa["status"])] += 1

    fetch_path = _reports_dir(data_dir) / f"{universe.id}_{timeframe}_fetch.json"
    fetch = _load_json_if_exists(fetch_path)
    fetched_successfully = 0
    failed_fetches = 0
    if fetch is not None:
        fetched_successfully = sum(
            1 for item in fetch.get("items", []) if item.get("status") == "fetched"
        )
        failed_fetches = sum(1 for item in fetch.get("items", []) if item.get("status") == "failed")

    ready_path = (
        Path(data_dir).expanduser()
        / "universes"
        / "research_ready"
        / f"{universe.id}_{timeframe}.json"
    )
    ready = _load_json_if_exists(ready_path)
    research_ready_count = 0
    rejected_count = 0
    rejection_counter: Counter[str] = Counter()
    if ready is not None:
        research_ready_count = len(ready.get("qualified_symbols", []))
        rejected = ready.get("rejected_symbols", [])
        rejected_count = len(rejected)
        for item in rejected:
            if isinstance(item, dict):
                rejection_counter.update(str(reason) for reason in item.get("reasons", []))

    markdown_path, json_path = _fetch_report_paths(
        universe_id=universe.id,
        timeframe=timeframe,
        data_dir=data_dir,
        suffix="health",
    )
    next_command = (
        "uv run stocker universe fetch "
        f"--universe <path> --from YYYY-MM-DD --to YYYY-MM-DD --timeframe {timeframe} "
        f"--source {source} --merge --audit --qa"
    )
    report = UniverseHealthReport(
        universe_id=universe.id,
        timeframe=timeframe,
        source=source,
        total_symbols=len(universe.symbols),
        datasets_present=datasets_present,
        datasets_missing=datasets_missing,
        fetched_successfully=fetched_successfully,
        failed_fetches=failed_fetches,
        audit_pass=audit_counts["pass"],
        audit_warning=audit_counts["warning"],
        audit_fail=audit_counts["fail"],
        qa_pass=qa_counts["pass"],
        qa_warning=qa_counts["warning"],
        qa_fail=qa_counts["fail"],
        average_row_count=float(sum(row_counts) / len(row_counts)) if row_counts else 0.0,
        min_timestamp=min(min_values) if min_values else None,
        max_timestamp=max(max_values) if max_values else None,
        research_ready_count=research_ready_count,
        rejected_count=rejected_count,
        top_rejection_reasons=dict(rejection_counter.most_common(10)),
        next_recommended_command=next_command,
        markdown_path=markdown_path,
        json_path=json_path,
    )
    payload = report.model_dump(mode="json")
    _write_json(json_path, payload)
    _write_markdown(
        markdown_path,
        f"""# Universe Health: {universe.id} {timeframe}

## Summary

- Total symbols: {report.total_symbols}
- Datasets present: {report.datasets_present}
- Datasets missing: {report.datasets_missing}
- Research-ready count: {report.research_ready_count}
- Rejected count: {report.rejected_count}

## Next Command

```bash
{next_command}
```
""",
    )
    return report


def list_universe_files(*, root: str | Path = ".") -> list[Path]:
    """List committed/generated/research-ready universe files."""

    base = Path(root)
    candidates = [
        base / "universes",
        base / "universes" / "generated",
        base / "universes" / "manual",
        base / "data" / "universes" / "research_ready",
    ]
    files: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            files.extend(sorted(candidate.glob("*.yaml")))
            files.extend(sorted(candidate.glob("*.yml")))
            files.extend(sorted(candidate.glob("*.json")))
    return sorted(set(files))
