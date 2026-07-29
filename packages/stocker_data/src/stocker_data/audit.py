"""Dataset audit report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from stocker_data.storage import DataLayer, DatasetKey, dataset_metadata, load_dataset
from stocker_data.validate import ValidationIssue, validate_ohlcv


@dataclass(frozen=True)
class AuditReportResult:
    """Paths and summary produced by an audit report."""

    markdown_path: Path
    json_path: Path
    passed: bool
    issues: list[ValidationIssue]


def _reports_dir(data_dir: str | Path) -> Path:
    return Path(data_dir).expanduser() / "reports" / "audits"


def _return_summary(frame: pd.DataFrame) -> dict[str, float]:
    close = pd.to_numeric(frame["close"], errors="coerce")
    returns = close.pct_change(fill_method=None).dropna()
    if returns.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(returns.mean()),
        "std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "min": float(returns.min()),
        "max": float(returns.max()),
    }


def _largest_bars(frame: pd.DataFrame, count: int = 3) -> dict[str, list[dict[str, Any]]]:
    data = frame.copy()
    data["return"] = pd.to_numeric(data["close"], errors="coerce").pct_change(fill_method=None)
    up = data.nlargest(count, "return")[["timestamp", "open", "high", "low", "close", "return"]]
    down = data.nsmallest(count, "return")[["timestamp", "open", "high", "low", "close", "return"]]
    up_records = cast(list[dict[str, Any]], up.to_dict("records"))
    down_records = cast(list[dict[str, Any]], down.to_dict("records"))
    return {
        "largest_up": [{**record, "timestamp": str(record["timestamp"])} for record in up_records],
        "largest_down": [
            {**record, "timestamp": str(record["timestamp"])} for record in down_records
        ],
    }


def _volume_summary(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    if "volume" not in frame:
        return {"present": False}
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    return {
        "present": True,
        "missing_count": int(volume.isna().sum()),
        "zero_count": int((volume == 0).sum()),
        "min": float(volume.min()) if not volume.dropna().empty else 0.0,
        "max": float(volume.max()) if not volume.dropna().empty else 0.0,
        "mean": float(volume.mean()) if not volume.dropna().empty else 0.0,
    }


def _issue_summary(issues: list[ValidationIssue]) -> dict[str, int]:
    summary = {"info": 0, "warning": 0, "error": 0}
    for issue in issues:
        summary[issue.severity] += 1
    return summary


def _markdown(payload: dict[str, Any]) -> str:
    issue_lines = "\n".join(
        f"- `{issue['severity']}` `{issue['code']}`: {issue['message']} ({issue['count']})"
        for issue in payload["issues"]
    )
    if not issue_lines:
        issue_lines = "- No validation issues."

    return f"""# Data Audit: {payload["symbol"]} {payload["timeframe"]}

## Summary

- Source: `{payload["source"]}`
- Instrument type: `{payload["instrument_type"]}`
- Rows: {payload["row_count"]}
- Date range: {payload["min_timestamp"]} to {payload["max_timestamp"]}
- Passed: {payload["passed"]}

## Validation Issues

{issue_lines}

## Volume Summary

```json
{json.dumps(payload["volume_summary"], indent=2)}
```

## Return Distribution

```json
{json.dumps(payload["return_summary"], indent=2)}
```

## Volatility

- Daily/sample volatility: {payload["return_summary"]["std"]}

## Largest Up Bars

```json
{json.dumps(payload["largest_bars"]["largest_up"], indent=2, default=str)}
```

## Largest Down Bars

```json
{json.dumps(payload["largest_bars"]["largest_down"], indent=2, default=str)}
```
"""


def create_audit_report(
    *,
    data_dir: str | Path = "data",
    symbol: str,
    timeframe: str,
    source: str = "manual",
    instrument_type: str = "stock",
    layer: DataLayer = "processed",
    market_calendar: str | None = None,
) -> AuditReportResult:
    """Create Markdown and JSON audit reports for one dataset."""

    key = DatasetKey(
        source=source,
        instrument_type=instrument_type,
        symbol=symbol.upper(),
        timeframe=timeframe,
    )
    frame = load_dataset(key, data_dir=data_dir, layer=layer)
    metadata = dataset_metadata(key, data_dir=data_dir, layer=layer)
    timezone = str(frame["timezone"].iloc[0]) if "timezone" in frame and not frame.empty else "UTC"
    issues = validate_ohlcv(
        frame,
        timeframe=timeframe,
        timezone=timezone,
        require_timezone=True,
        market_calendar=market_calendar,
    )
    issue_counts = _issue_summary(issues)
    passed = issue_counts["warning"] == 0 and issue_counts["error"] == 0
    payload: dict[str, Any] = {
        "symbol": key.symbol,
        "timeframe": key.timeframe,
        "source": key.source,
        "instrument_type": key.instrument_type,
        "row_count": metadata.row_count,
        "min_timestamp": metadata.min_timestamp,
        "max_timestamp": metadata.max_timestamp,
        "missing_fields": metadata.missing_fields,
        "issues": [issue.to_dict() for issue in issues],
        "issue_counts": issue_counts,
        "volume_summary": _volume_summary(frame),
        "return_summary": _return_summary(frame),
        "largest_bars": _largest_bars(frame),
        "passed": passed,
    }

    output_dir = _reports_dir(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{key.symbol}_{key.timeframe}_audit"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    return AuditReportResult(
        markdown_path=markdown_path,
        json_path=json_path,
        passed=passed,
        issues=issues,
    )
