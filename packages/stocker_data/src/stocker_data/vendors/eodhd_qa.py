"""Vendor QA reports for EODHD-backed Stocker datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stocker_data.storage import DatasetKey, dataset_path, load_dataset
from stocker_data.validate import ValidationIssue, validate_ohlcv

QAStatus = Literal["pass", "warning", "fail"]


class AdjustedPricePolicy(StrEnum):
    """How EODHD adjusted-close availability should be treated."""

    RAW_CLOSE = "raw_close"
    ADJUSTED_AVAILABLE = "adjusted_available"
    REQUIRE_ADJUSTED_CLOSE = "require_adjusted_close"


@dataclass(frozen=True)
class EODHDQAReportResult:
    """Paths and status from an EODHD QA report."""

    markdown_path: Path
    json_path: Path
    status: QAStatus
    issue_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a printable summary."""

        return {
            "markdown_path": str(self.markdown_path),
            "json_path": str(self.json_path),
            "status": self.status,
            "issue_codes": self.issue_codes,
        }


@dataclass(frozen=True)
class RawFileCoverage:
    """Dataset-specific raw response coverage details."""

    endpoint: str
    selector_name: str
    selector_value: str
    paths: list[Path]
    first_date: str | None
    last_date: str | None
    covers_dataset_range: bool


def _reports_dir(data_dir: str | Path) -> Path:
    return Path(data_dir).expanduser() / "reports" / "vendor_qa"


def _raw_root(data_dir: str | Path) -> Path:
    return Path(data_dir).expanduser() / "raw" / "source=eodhd"


def _raw_selector_for_timeframe(timeframe: str) -> tuple[str, str, str]:
    normalized = timeframe.strip().lower()
    if normalized in {"1d", "d", "day", "daily"}:
        return "eod", "period", "d"
    if normalized in {"1w", "w", "week", "weekly"}:
        return "eod", "period", "w"
    if normalized in {"1mo", "m", "month", "monthly"}:
        return "eod", "period", "m"
    return "intraday", "interval", timeframe


def _filename_date_range(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    parts = path.stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return pd.Timestamp(parts[0]), pd.Timestamp(parts[1])
    except ValueError:
        return None


def _raw_file_coverage(
    data_dir: str | Path,
    *,
    symbol: str,
    timeframe: str,
    min_timestamp: str | None,
    max_timestamp: str | None,
) -> RawFileCoverage:
    endpoint, selector_name, selector_value = _raw_selector_for_timeframe(timeframe)
    root = _raw_root(data_dir)
    if not root.exists():
        return RawFileCoverage(
            endpoint=endpoint,
            selector_name=selector_name,
            selector_value=selector_value,
            paths=[],
            first_date=None,
            last_date=None,
            covers_dataset_range=False,
        )
    paths = sorted(
        root.glob(
            f"endpoint={endpoint}/symbol={symbol.upper()}/{selector_name}={selector_value}/*.json"
        )
    )
    ranges = [date_range for path in paths if (date_range := _filename_date_range(path))]
    first_date = min((date_range[0] for date_range in ranges), default=None)
    last_date = max((date_range[1] for date_range in ranges), default=None)
    dataset_start = None if min_timestamp is None else pd.Timestamp(min_timestamp)
    dataset_end = None if max_timestamp is None else pd.Timestamp(max_timestamp)
    covers_dataset_range = False
    if (
        first_date is not None
        and last_date is not None
        and dataset_start is not None
        and dataset_end is not None
    ):
        covers_dataset_range = first_date.date() <= dataset_start.date() and (
            last_date.date() >= dataset_end.date()
        )
    return RawFileCoverage(
        endpoint=endpoint,
        selector_name=selector_name,
        selector_value=selector_value,
        paths=paths,
        first_date=None if first_date is None else first_date.date().isoformat(),
        last_date=None if last_date is None else last_date.date().isoformat(),
        covers_dataset_range=covers_dataset_range,
    )


def _validation_counts(issues: list[ValidationIssue]) -> dict[str, int]:
    counts = {"info": 0, "warning": 0, "error": 0}
    for issue in issues:
        counts[issue.severity] += 1
    return counts


def _adjusted_close_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if "adjusted_close" not in frame:
        return {
            "present": False,
            "missing_count": len(frame),
            "different_from_close_count": 0,
            "max_abs_adjustment_pct": 0.0,
            "first_adjustment_timestamp": None,
            "last_adjustment_timestamp": None,
        }
    close = pd.to_numeric(frame["close"], errors="coerce")
    adjusted = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    difference = (adjusted - close).abs()
    ratio = difference / close.abs()
    changed = ratio.fillna(0.0) > 0.000001
    timestamps = pd.to_datetime(frame.loc[changed, "timestamp"], errors="coerce")
    return {
        "present": True,
        "missing_count": int(adjusted.isna().sum()),
        "different_from_close_count": int(changed.sum()),
        "max_abs_adjustment_pct": float(ratio.max()) if not ratio.dropna().empty else 0.0,
        "first_adjustment_timestamp": None if timestamps.empty else str(timestamps.dropna().min()),
        "last_adjustment_timestamp": None if timestamps.empty else str(timestamps.dropna().max()),
    }


def _refresh_plan(
    *,
    symbol: str,
    timeframe: str,
    instrument_type: str,
    max_timestamp: str | None,
) -> dict[str, Any]:
    next_from = "YYYY-MM-DD"
    if max_timestamp is not None:
        parsed = pd.Timestamp(max_timestamp)
        next_from = parsed.date().isoformat()
    if timeframe in {"1d", "1w", "1mo"}:
        period = {"1d": "d", "1w": "w", "1mo": "m"}[timeframe]
        command = (
            "uv run stocker data fetch-eodhd-eod "
            f"--symbol {symbol.upper()} --from {next_from} --to YYYY-MM-DD "
            f"--period {period} --instrument-type {instrument_type} --merge --save-raw --audit"
        )
    else:
        command = (
            "uv run stocker data fetch-eodhd-intraday "
            f"--symbol {symbol.upper()} --interval {timeframe} --from {next_from} "
            f"--to YYYY-MM-DD --instrument-type {instrument_type} --merge --save-raw --audit"
        )
    return {
        "recommended_mode": "merge",
        "next_from": next_from,
        "rule": "Refresh with --merge, keep raw JSON, validate, audit, then QA before research.",
        "eod_command": command,
    }


def _status_and_issue_codes(
    *,
    validation_counts: dict[str, int],
    adjusted_summary: dict[str, Any],
    policy: AdjustedPricePolicy,
    raw_coverage: RawFileCoverage,
    require_raw: bool,
) -> tuple[QAStatus, list[str]]:
    issue_codes: list[str] = []
    if validation_counts["error"]:
        issue_codes.append("validation_errors")
    if validation_counts["warning"]:
        issue_codes.append("validation_warnings")
    if policy == AdjustedPricePolicy.REQUIRE_ADJUSTED_CLOSE and not adjusted_summary["present"]:
        issue_codes.append("missing_adjusted_close")
    if require_raw and len(raw_coverage.paths) == 0:
        issue_codes.append("missing_raw_responses")
    if raw_coverage.paths and not raw_coverage.covers_dataset_range:
        issue_codes.append("raw_date_range_incomplete")
    if adjusted_summary["different_from_close_count"]:
        issue_codes.append("adjusted_close_differs_from_close")

    fail_codes = {"validation_errors", "missing_adjusted_close", "missing_raw_responses"}
    if fail_codes.intersection(issue_codes):
        return "fail", issue_codes
    if issue_codes:
        return "warning", issue_codes
    return "pass", issue_codes


def _markdown(payload: dict[str, Any]) -> str:
    issues = "\n".join(f"- `{code}`" for code in payload["issue_codes"]) or "- None"
    return f"""# EODHD Vendor QA: {payload["symbol"]} {payload["timeframe"]}

## Summary

- Status: `{payload["status"]}`
- Rows: {payload["row_count"]}
- Date range: {payload["min_timestamp"]} to {payload["max_timestamp"]}
- Raw files: {payload["raw_files"]["count"]}
- Raw selector: `{payload["raw_files"]["endpoint"]}` `{payload["raw_files"]["selector"]}`

## Adjusted Close

```json
{json.dumps(payload["adjusted_close"], indent=2)}
```

## Calendar

```json
{json.dumps(payload["calendar"], indent=2)}
```

## Validation

```json
{json.dumps(payload["validation"], indent=2)}
```

## Refresh Plan

```json
{json.dumps(payload["refresh_plan"], indent=2)}
```

## Issues

{issues}
"""


def create_eodhd_qa_report(
    *,
    data_dir: str | Path = "data",
    symbol: str,
    timeframe: str,
    instrument_type: str = "stock",
    market_calendar: str | None = None,
    adjusted_price_policy: AdjustedPricePolicy | str = AdjustedPricePolicy.ADJUSTED_AVAILABLE,
    require_raw: bool = False,
) -> EODHDQAReportResult:
    """Create a vendor-specific QA report for a normalized EODHD dataset."""

    policy = AdjustedPricePolicy(adjusted_price_policy)
    key = DatasetKey(
        source="eodhd",
        instrument_type=instrument_type,
        symbol=symbol.upper(),
        timeframe=timeframe,
    )
    frame = load_dataset(key, data_dir=data_dir).sort_values("timestamp").reset_index(drop=True)
    issues = validate_ohlcv(
        frame,
        timeframe=timeframe,
        timezone="UTC",
        require_timezone=True,
        market_calendar=market_calendar,
    )
    validation_counts = _validation_counts(issues)
    adjusted_summary = _adjusted_close_summary(frame)
    timestamps = (
        pd.to_datetime(frame["timestamp"], errors="coerce") if "timestamp" in frame else None
    )
    min_timestamp = None if timestamps is None or timestamps.empty else str(timestamps.min())
    max_timestamp = None if timestamps is None or timestamps.empty else str(timestamps.max())
    raw_coverage = _raw_file_coverage(
        data_dir,
        symbol=symbol,
        timeframe=timeframe,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
    )
    status, issue_codes = _status_and_issue_codes(
        validation_counts=validation_counts,
        adjusted_summary=adjusted_summary,
        policy=policy,
        raw_coverage=raw_coverage,
        require_raw=require_raw,
    )
    payload: dict[str, Any] = {
        "source": "eodhd",
        "symbol": key.symbol,
        "timeframe": key.timeframe,
        "instrument_type": key.instrument_type,
        "dataset_path": str(dataset_path(key, data_dir=data_dir)),
        "status": status,
        "issue_codes": issue_codes,
        "row_count": int(len(frame)),
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "adjusted_price_policy": policy.value,
        "adjusted_close": adjusted_summary,
        "raw_files": {
            "required": require_raw,
            "endpoint": raw_coverage.endpoint,
            "selector": {raw_coverage.selector_name: raw_coverage.selector_value},
            "count": len(raw_coverage.paths),
            "first_raw_file": str(raw_coverage.paths[0]) if raw_coverage.paths else None,
            "last_raw_file": str(raw_coverage.paths[-1]) if raw_coverage.paths else None,
            "paths": [str(path) for path in raw_coverage.paths],
            "date_coverage": {
                "first_raw_date": raw_coverage.first_date,
                "last_raw_date": raw_coverage.last_date,
                "covers_dataset_range": raw_coverage.covers_dataset_range,
            },
        },
        "calendar": {
            "market_calendar": market_calendar,
            "missing_session_issues": [
                issue.to_dict() for issue in issues if issue.code == "missing_market_session"
            ],
        },
        "validation": {
            "counts": validation_counts,
            "issues": [issue.to_dict() for issue in issues],
        },
        "refresh_plan": _refresh_plan(
            symbol=symbol,
            timeframe=timeframe,
            instrument_type=instrument_type,
            max_timestamp=max_timestamp,
        ),
    }
    output_dir = _reports_dir(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{key.symbol}_{key.timeframe}_eodhd_qa"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return EODHDQAReportResult(
        markdown_path=markdown_path,
        json_path=json_path,
        status=status,
        issue_codes=issue_codes,
    )
