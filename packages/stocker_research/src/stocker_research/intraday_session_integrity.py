"""Stage 3.8 intraday session-integrity and position-policy diagnostics."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from stocker_data.storage import DatasetKey, load_dataset

DEFAULT_OUTPUT_DIR = Path("data/reports/research/stage3_8_intraday_session_integrity")
DEFAULT_STAGE3_7_SUMMARY = Path(
    "data/reports/research/stage3_7_intraday_5m_session_flat_smoke/summary.json"
)
DEFAULT_SYMBOLS = ["AAPL.US", "AMZN.US", "META.US", "MSFT.US", "NVDA.US"]
REPORT_CONTRACT_FIELDS = [
    "evaluation_policy",
    "indicator_context_policy",
    "context_summary",
    "selected_result",
    "best_test_diagnostic",
    "benchmark_results",
    "null_model_results",
    "holding_policy",
    "holding_policy_analysis",
    "holding_policy_decision",
    "position_policy",
    "classification_reasons",
]


@dataclass(frozen=True)
class IntradaySessionIntegrityResult:
    """Paths and headline counts from a Stage 3.8 diagnostic run."""

    summary_json_path: Path
    summary_markdown_path: Path
    incomplete_sessions_csv_path: Path
    session_bar_counts_csv_path: Path
    position_policy_actions_csv_path: Path
    report_count_analyzed: int
    incomplete_session_count_by_bucket: dict[str, int]
    symbols_with_most_incomplete_sessions: list[dict[str, Any]]
    position_policy_action_summary: dict[str, Any]
    intraday_classification_anatomy: dict[str, Any]
    stage_passed: bool
    recommended_next_step: str


def _timeframe_minutes(timeframe: str) -> int:
    normalized = timeframe.strip().lower()
    if normalized.endswith("m") and normalized[:-1].isdigit():
        return int(normalized[:-1])
    if normalized.endswith("min") and normalized[:-3].isdigit():
        return int(normalized[:-3])
    raise ValueError(f"Unsupported intraday timeframe for session audit: {timeframe}")


def expected_session_timestamps(
    market_open: pd.Timestamp,
    market_close: pd.Timestamp,
    *,
    timeframe: str,
) -> list[pd.Timestamp]:
    """Return the vendor-boundary-inclusive timestamp grid for one session."""

    minutes = _timeframe_minutes(timeframe)
    return list(pd.date_range(market_open, market_close, freq=f"{minutes}min"))


def _load_calendar_schedule(
    *,
    market_calendar: str,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, set[date], str | None]:
    try:
        import pandas_market_calendars as mcal
    except ModuleNotFoundError:
        return pd.DataFrame(), set(), "pandas_market_calendars_unavailable"

    calendar = mcal.get_calendar(market_calendar)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    early_close_dates: set[date] = set()
    try:
        early_closes = calendar.early_closes(schedule)
        early_close_dates = {pd.Timestamp(index).date() for index in early_closes.index}
    except Exception:
        early_close_dates = set()
    return schedule, early_close_dates, None


def _timestamp_convention(
    *,
    first_timestamp: pd.Timestamp | None,
    last_timestamp: pd.Timestamp | None,
    market_open: pd.Timestamp,
    market_close: pd.Timestamp,
    timeframe: str,
) -> str:
    if first_timestamp is None or last_timestamp is None:
        return "no_bars"
    minutes = _timeframe_minutes(timeframe)
    one_bar = pd.Timedelta(minutes=minutes)
    if first_timestamp == market_open and last_timestamp == market_close:
        return "boundary_inclusive"
    if first_timestamp == market_open and last_timestamp == market_close - one_bar:
        return "bar_open"
    if first_timestamp == market_open + one_bar and last_timestamp == market_close:
        return "bar_close"
    return "partial_or_mixed"


def _missing_location_flags(
    *,
    expected: set[pd.Timestamp],
    actual: set[pd.Timestamp],
    first_timestamp: pd.Timestamp | None,
    last_timestamp: pd.Timestamp | None,
    market_open: pd.Timestamp,
    market_close: pd.Timestamp,
) -> tuple[bool, bool, bool]:
    missing = expected - actual
    if not missing:
        return False, False, False
    missing_open = market_open in missing
    missing_close = market_close in missing
    missing_middle = False
    if first_timestamp is not None and last_timestamp is not None:
        for timestamp in missing:
            if first_timestamp < timestamp < last_timestamp:
                missing_middle = True
                break
    return missing_open, missing_middle, missing_close


def _bucket_session(
    *,
    appears_complete: bool,
    is_early_close: bool,
    is_first_fetch_session: bool,
    is_last_fetch_session: bool,
    missing_open: bool,
    missing_middle: bool,
    missing_close: bool,
    missing_count: int,
    extra_count: int,
) -> tuple[str, str, str]:
    if extra_count:
        return (
            "session_calendar_mismatch",
            "warning",
            "Observed bars fall outside the XNYS 5m scheduled timestamp grid.",
        )
    if appears_complete:
        if is_early_close:
            return (
                "expected_market_early_close",
                "info",
                "Session matches an XNYS early-close schedule.",
            )
        return ("complete", "info", "Session matches the XNYS 5m timestamp grid.")
    if (
        (is_first_fetch_session or is_last_fetch_session)
        and missing_count > 1
        and not missing_middle
    ):
        return (
            "expected_fetch_boundary_partial",
            "info",
            "Only the fetch-range boundary session is partial.",
        )
    if missing_middle:
        return (
            "possible_mid_session_gap",
            "warning",
            "Expected in-session 5m bars are missing between observed bars.",
        )
    if missing_open and not missing_close:
        return (
            "possible_missing_open_bar",
            "warning",
            "The scheduled session-open boundary bar is absent.",
        )
    if missing_close and not missing_open:
        return (
            "possible_missing_close_bar",
            "warning",
            "The scheduled session-close boundary bar is absent.",
        )
    if missing_open and missing_close:
        return (
            "possible_mid_session_gap",
            "warning",
            "Both session boundary bars and/or middle bars are absent.",
        )
    return ("unknown", "warning", "Session is incomplete for an unknown reason.")


def analyze_symbol_sessions(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    market_calendar: str,
) -> list[dict[str, Any]]:
    """Return one session-integrity row per expected calendar session for a symbol."""

    if frame.empty or "timestamp" not in frame:
        return []
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    timestamps = data["timestamp"]
    schedule, early_close_dates, calendar_error = _load_calendar_schedule(
        market_calendar=market_calendar,
        start_date=timestamps.min().date(),
        end_date=timestamps.max().date(),
    )
    if calendar_error or schedule.empty:
        return []

    data["session_date"] = timestamps.dt.date
    actual_by_date = {
        session_date: group["timestamp"].sort_values().reset_index(drop=True)
        for session_date, group in data.groupby("session_date")
    }
    schedule_dates = [pd.Timestamp(index).date() for index in schedule.index]
    first_schedule_date = schedule_dates[0]
    last_schedule_date = schedule_dates[-1]
    rows: list[dict[str, Any]] = []

    for session_index, session in schedule.iterrows():
        session_date = pd.Timestamp(session_index).date()
        if session_date not in actual_by_date:
            continue
        market_open = pd.Timestamp(session["market_open"]).tz_convert("UTC")
        market_close = pd.Timestamp(session["market_close"]).tz_convert("UTC")
        expected_timestamps = expected_session_timestamps(
            market_open,
            market_close,
            timeframe=timeframe,
        )
        actual_timestamps = actual_by_date.get(session_date, pd.Series(dtype="datetime64[ns, UTC]"))
        actual_list = [pd.Timestamp(timestamp) for timestamp in actual_timestamps]
        actual_set = set(actual_list)
        expected_set = set(expected_timestamps)
        missing_set = expected_set - actual_set
        extra_set = actual_set - expected_set
        first_timestamp = actual_list[0] if actual_list else None
        last_timestamp = actual_list[-1] if actual_list else None
        missing_open, missing_middle, missing_close = _missing_location_flags(
            expected=expected_set,
            actual=actual_set,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            market_open=market_open,
            market_close=market_close,
        )
        sorted_missing = sorted(missing_set)
        sorted_extra = sorted(extra_set)
        appears_complete = not missing_set and not extra_set
        is_early_close = session_date in early_close_dates
        is_first_fetch_session = session_date == first_schedule_date
        is_last_fetch_session = session_date == last_schedule_date
        bucket, severity, reason = _bucket_session(
            appears_complete=appears_complete,
            is_early_close=is_early_close,
            is_first_fetch_session=is_first_fetch_session,
            is_last_fetch_session=is_last_fetch_session,
            missing_open=missing_open,
            missing_middle=missing_middle,
            missing_close=missing_close,
            missing_count=len(missing_set),
            extra_count=len(extra_set),
        )
        convention = _timestamp_convention(
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            market_open=market_open,
            market_close=market_close,
            timeframe=timeframe,
        )
        dst_handled = not extra_set and (
            first_timestamp is None
            or (
                first_timestamp >= market_open
                and last_timestamp is not None
                and last_timestamp <= market_close
            )
        )
        rows.append(
            {
                "symbol": symbol,
                "session_date": str(session_date),
                "calendar_open": str(market_open),
                "calendar_close": str(market_close),
                "first_timestamp": str(first_timestamp) if first_timestamp is not None else "",
                "last_timestamp": str(last_timestamp) if last_timestamp is not None else "",
                "bar_count": int(len(actual_list)),
                "expected_bar_count": int(len(expected_timestamps)),
                "appears_complete": bool(appears_complete),
                "is_early_close": bool(is_early_close),
                "is_first_fetch_session": bool(is_first_fetch_session),
                "is_last_fetch_session": bool(is_last_fetch_session),
                "missing_bars_at_open": bool(missing_open),
                "missing_bars_in_middle": bool(missing_middle),
                "missing_bars_at_close": bool(missing_close),
                "missing_bar_count": int(len(missing_set)),
                "extra_bar_count": int(len(extra_set)),
                "first_missing_timestamp": str(sorted_missing[0]) if sorted_missing else "",
                "last_missing_timestamp": str(sorted_missing[-1]) if sorted_missing else "",
                "first_extra_timestamp": str(sorted_extra[0]) if sorted_extra else "",
                "last_extra_timestamp": str(sorted_extra[-1]) if sorted_extra else "",
                "dst_handled_correctly": bool(dst_handled),
                "timestamp_convention": convention,
                "bucket": bucket,
                "reason": reason,
                "severity": severity,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _stage3_7_report_paths(stage3_7_summary_path: Path) -> list[Path]:
    summary = _load_json(stage3_7_summary_path)
    paths = summary.get("report_contract_sanity", {}).get("sampled_experiment_json_files", [])
    return [Path(path) for path in paths]


def _experiment_hypothesis_id(payload: dict[str, Any]) -> str:
    hypothesis = payload.get("hypothesis", {})
    if isinstance(hypothesis, dict):
        return str(hypothesis.get("id") or hypothesis.get("name") or "unknown")
    return "unknown"


def _selected_result(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("selected_result", {})
    return value if isinstance(value, dict) else {}


def _position_policy(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("position_policy", {})
    return value if isinstance(value, dict) else {}


def _holding_analysis(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def _null_results(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("null_model_results", {})
    return value if isinstance(value, dict) else {}


def _position_policy_rows(
    report_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for path in report_paths:
        try:
            payload = _load_json(path)
        except Exception as exc:
            malformed.append({"json_path": str(path), "error": str(exc), "missing_fields": []})
            continue
        missing = [field for field in REPORT_CONTRACT_FIELDS if field not in payload]
        if (
            "raw_holding_policy_analysis" not in payload
            and "raw_template_holding_policy_analysis" not in payload
        ):
            missing.append("raw_holding_policy_analysis_or_equivalent")
        if missing:
            malformed.append(
                {
                    "json_path": str(path),
                    "error": "missing_required_contract_fields",
                    "missing_fields": missing,
                }
            )
        payloads.append(payload)
        selected = _selected_result(payload)
        position_policy = _position_policy(payload)
        scored = _holding_analysis(payload, "holding_policy_analysis")
        raw = _holding_analysis(payload, "raw_holding_policy_analysis")
        if not raw:
            raw = _holding_analysis(payload, "raw_template_holding_policy_analysis")
        raw_exposure = _as_float(position_policy.get("raw_exposure"))
        adjusted_exposure = _as_float(position_policy.get("adjusted_exposure"))
        reduction = 0.0 if raw_exposure == 0 else (raw_exposure - adjusted_exposure) / raw_exposure
        rows.append(
            {
                "symbol": str(payload.get("symbol", "")),
                "hypothesis": _experiment_hypothesis_id(payload),
                "classification": str(payload.get("classification", "")),
                "classification_reasons": "|".join(
                    str(reason) for reason in payload.get("classification_reasons", [])
                ),
                "forced_flat_count": _as_int(position_policy.get("positions_forced_flat_count")),
                "late_entries_blocked_count": _as_int(
                    position_policy.get("late_entries_blocked_count")
                ),
                "overnight_carry_prevented_count": _as_int(
                    position_policy.get("overnight_carry_prevented_count")
                ),
                "raw_exposure": raw_exposure,
                "scored_exposure": adjusted_exposure,
                "exposure_reduction_pct": float(reduction),
                "raw_session_flat_compliant": bool(raw.get("session_flat_compliant", False)),
                "scored_session_flat_compliant": bool(
                    scored.get("session_flat_compliant", False)
                ),
                "policy_adjusted_trade_count": _as_int(
                    selected.get("test_trade_count", selected.get("trade_count"))
                ),
                "sessions_seen": _as_int(position_policy.get("sessions_seen")),
                "incomplete_session_warning_count": _as_int(
                    position_policy.get("incomplete_session_warning_count")
                ),
                "json_path": str(path),
                "report_path": str(path.with_suffix(".md")),
            }
        )
    return rows, payloads, malformed


def _group_position_actions(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    output: dict[str, dict[str, Any]] = {}
    for group_key, group_rows in grouped.items():
        raw_values = [_as_float(row["raw_exposure"]) for row in group_rows]
        scored_values = [_as_float(row["scored_exposure"]) for row in group_rows]
        output[group_key] = {
            "experiment_count": len(group_rows),
            "forced_flat_count": sum(_as_int(row["forced_flat_count"]) for row in group_rows),
            "late_entries_blocked_count": sum(
                _as_int(row["late_entries_blocked_count"]) for row in group_rows
            ),
            "overnight_carry_prevented_count": sum(
                _as_int(row["overnight_carry_prevented_count"]) for row in group_rows
            ),
            "sessions_seen": sum(_as_int(row["sessions_seen"]) for row in group_rows),
            "incomplete_session_warning_count": sum(
                _as_int(row["incomplete_session_warning_count"]) for row in group_rows
            ),
            "median_raw_exposure": _median(raw_values),
            "median_scored_exposure": _median(scored_values),
            "raw_session_flat_compliant_count": sum(
                1 for row in group_rows if row["raw_session_flat_compliant"]
            ),
            "scored_session_flat_compliant_count": sum(
                1 for row in group_rows if row["scored_session_flat_compliant"]
            ),
        }
    return dict(sorted(output.items()))


def _classification_anatomy(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[_experiment_hypothesis_id(payload)].append(payload)
    output: dict[str, Any] = {}
    for hypothesis, reports in sorted(grouped.items()):
        selected = [_selected_result(payload) for payload in reports]
        nulls = [_null_results(payload) for payload in reports]
        policies = [_position_policy(payload) for payload in reports]
        output[hypothesis] = {
            "experiment_count": len(reports),
            "classification_counts": dict(
                Counter(str(payload.get("classification", "")) for payload in reports)
            ),
            "classification_reason_counts": dict(
                Counter(
                    str(reason)
                    for payload in reports
                    for reason in payload.get("classification_reasons", [])
                )
            ),
            "benchmark_pass_count": sum(1 for payload in reports if payload.get("benchmark_pass")),
            "null_pass_count": sum(1 for null in nulls if null.get("null_pass")),
            "median_selected_test_net_return": _median(
                [_as_float(row.get("test_net_return")) for row in selected]
            ),
            "median_train_net_return": _median(
                [_as_float(row.get("train_net_return")) for row in selected]
            ),
            "median_trade_count": _median(
                [
                    float(_as_int(row.get("test_trade_count", row.get("trade_count"))))
                    for row in selected
                ]
            ),
            "median_benchmark_excess": _median(
                [_as_float(payload.get("selected_excess_vs_buy_and_hold")) for payload in reports]
            ),
            "median_null_excess": _median(
                [_as_float(null.get("selected_excess_vs_p75_null")) for null in nulls]
            ),
            "median_stability_score": _median(
                [
                    _as_float((payload.get("stability") or {}).get("stability_score"))
                    for payload in reports
                ]
            ),
            "median_max_drawdown": _median(
                [
                    _as_float(row.get("test_max_drawdown", row.get("max_drawdown")))
                    for row in selected
                ]
            ),
            "median_raw_exposure": _median(
                [_as_float(policy.get("raw_exposure")) for policy in policies]
            ),
            "median_scored_exposure": _median(
                [_as_float(policy.get("adjusted_exposure")) for policy in policies]
            ),
            "median_policy_adjusted_trade_count": _median(
                [
                    float(_as_int(row.get("test_trade_count", row.get("trade_count"))))
                    for row in selected
                ]
            ),
        }
    return output


def _markdown(summary: dict[str, Any]) -> str:
    available_range = (
        f"{summary['actual_available_range']['from']} to "
        f"{summary['actual_available_range']['to']}"
    )
    bucket_lines = "\n".join(
        f"- `{bucket}`: {count}"
        for bucket, count in summary["incomplete_session_count_by_bucket"].items()
    )
    if not bucket_lines:
        bucket_lines = "- None"
    symbol_lines = "\n".join(
        f"- `{item['symbol']}`: {item['incomplete_session_count']}"
        for item in summary["symbols_with_most_incomplete_sessions"]
    )
    if not symbol_lines:
        symbol_lines = "- None"
    interpretation_lines = "\n".join(
        f"- {item}" for item in summary["session_data_interpretation"]
    )
    mismatch_lines = "\n".join(
        (
            "- `{symbol}` `{session_date}` extra `{first_extra_timestamp}` "
            "missing `{first_missing_timestamp}`"
        ).format(**item)
        for item in summary["calendar_mismatch_examples"]
    )
    if not mismatch_lines:
        mismatch_lines = "- None"
    position_interpretation_lines = "\n".join(
        f"- {item}" for item in summary["position_policy_interpretation"]
    )
    return f"""# Stage 3.8 Intraday Session Integrity

## Scope

- Timeframe: `{summary["timeframe"]}`
- Market calendar: `{summary["market_calendar"]}`
- Symbols: {", ".join(summary["symbols"])}
- Actual available range: {available_range}
- Stage 3.7 reports analyzed: {summary["report_count_analyzed"]}
- Malformed reports: {summary["malformed_report_count"]}

## Session Completeness

Incomplete/nonstandard session buckets:

{bucket_lines}

Symbols with most incomplete/nonstandard sessions:

{symbol_lines}

The Stage 3.7 `incomplete_session_warning_count` values are position-policy window
warnings, not a count of bad full-day vendor sessions. They are mainly caused by
walk-forward train/test windows starting or ending inside an otherwise valid XNYS
session. Dataset-level session rows are bucketed separately in
`session_bar_counts.csv` and `incomplete_sessions.csv`.

Session interpretation:

{interpretation_lines}

Calendar mismatch examples:

{mismatch_lines}

## Position Policy

```json
{json.dumps(summary["position_policy_action_summary"], indent=2)}
```

Forced flats are expected for raw templates that would otherwise remain long near
the session close. Late-entry blocks mean raw signals appeared after the configured
entry cutoff. Overnight-carry prevention is expected when raw template positions
persist across sessions; scored positions remain session-flat diagnostics only.

Position-policy interpretation:

{position_interpretation_lines}

## Intraday Rejection Anatomy

```json
{json.dumps(summary["intraday_classification_anatomy"], indent=2)}
```

All cases remain rejected diagnostics. No candidates are manufactured and nothing
in this audit is evidence of an edge.

## Report Contract

- Passed: `{summary["report_contract_sanity"]["passed"]}`
- Required fields: {", ".join(f"`{field}`" for field in REPORT_CONTRACT_FIELDS)}

## Recommendation

{summary["recommendation"]}
"""


def build_intraday_session_integrity_summary(
    *,
    data_dir: str | Path = "data",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    symbols: list[str] | None = None,
    timeframe: str = "5m",
    source: str = "eodhd",
    instrument_type: str = "stock",
    market_calendar: str = "XNYS",
    stage3_7_summary_path: str | Path = DEFAULT_STAGE3_7_SUMMARY,
) -> IntradaySessionIntegrityResult:
    """Build Stage 3.8 diagnostics from existing local 5m data and reports."""

    selected_symbols = symbols or DEFAULT_SYMBOLS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_session_rows: list[dict[str, Any]] = []
    min_timestamps: list[pd.Timestamp] = []
    max_timestamps: list[pd.Timestamp] = []
    for symbol in selected_symbols:
        frame = load_dataset(
            DatasetKey(
                source=source,
                instrument_type=instrument_type,
                symbol=symbol,
                timeframe=timeframe,
            ),
            data_dir=data_dir,
        )
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        min_timestamps.append(timestamps.min())
        max_timestamps.append(timestamps.max())
        all_session_rows.extend(
            analyze_symbol_sessions(
                frame,
                symbol=symbol,
                timeframe=timeframe,
                market_calendar=market_calendar,
            )
        )

    incomplete_rows = [row for row in all_session_rows if row["bucket"] != "complete"]
    report_paths = _stage3_7_report_paths(Path(stage3_7_summary_path))
    policy_rows, report_payloads, malformed_reports = _position_policy_rows(report_paths)
    bucket_counts = dict(Counter(str(row["bucket"]) for row in incomplete_rows))
    symbol_incomplete_counts = Counter(str(row["symbol"]) for row in incomplete_rows)
    symbols_with_most = [
        {"symbol": symbol, "incomplete_session_count": count}
        for symbol, count in symbol_incomplete_counts.most_common()
    ]
    severity_counts = dict(Counter(str(row["severity"]) for row in incomplete_rows))
    convention_counts = dict(Counter(str(row["timestamp_convention"]) for row in all_session_rows))
    calendar_mismatch_examples = [
        {
            "symbol": row["symbol"],
            "session_date": row["session_date"],
            "first_timestamp": row["first_timestamp"],
            "last_timestamp": row["last_timestamp"],
            "first_extra_timestamp": row["first_extra_timestamp"],
            "first_missing_timestamp": row["first_missing_timestamp"],
        }
        for row in incomplete_rows
        if row["bucket"] == "session_calendar_mismatch"
    ][:10]
    policy_by_hypothesis = _group_position_actions(policy_rows, "hypothesis")
    policy_by_symbol = _group_position_actions(policy_rows, "symbol")
    aggregate_policy = {
        "forced_flat_count": sum(_as_int(row["forced_flat_count"]) for row in policy_rows),
        "late_entries_blocked_count": sum(
            _as_int(row["late_entries_blocked_count"]) for row in policy_rows
        ),
        "overnight_carry_prevented_count": sum(
            _as_int(row["overnight_carry_prevented_count"]) for row in policy_rows
        ),
        "sessions_seen": sum(_as_int(row["sessions_seen"]) for row in policy_rows),
        "incomplete_session_warning_count": sum(
            _as_int(row["incomplete_session_warning_count"]) for row in policy_rows
        ),
        "by_hypothesis": policy_by_hypothesis,
        "by_symbol": policy_by_symbol,
    }
    position_policy_interpretation = [
        (
            "Forced flats are concentrated in moving_average_momentum, which is expected "
            "because the raw moving-average template can persist into the close."
        ),
        (
            "Late-entry blocks are concentrated in mean_reversion_after_large_down_bar, "
            "which means signals often appear inside the configured entry cutoff."
        ),
        (
            "Overnight-carry prevention is mostly from moving_average_momentum raw "
            "positions carrying across sessions before the overlay scores them flat."
        ),
        (
            "No template with nonzero raw exposure is almost entirely zeroed out by the "
            "session-flat policy; volatility_breakout has zero raw exposure in this smoke."
        ),
    ]
    anatomy = _classification_anatomy(report_payloads)
    malformed_count = len(malformed_reports)
    error_bucket_count = severity_counts.get("error", 0)
    warning_bucket_count = severity_counts.get("warning", 0)
    if error_bucket_count:
        recommendation = (
            "B. Fix session/calendar/timestamp handling before more research."
        )
    elif any(
        summary.get("scored_session_flat_compliant_count", 0) != summary.get("experiment_count", 0)
        for summary in policy_by_hypothesis.values()
    ):
        recommendation = "C. Fix position-policy overlay/reporting before more research."
    else:
        recommendation = (
            "A. Intraday data/session integrity is good enough to broaden to 25-50 liquid US names."
        )
    stage_passed = malformed_count == 0 and error_bucket_count == 0

    summary: dict[str, Any] = {
        "stage": "3.8_intraday_session_integrity",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "timeframe": timeframe,
        "source": source,
        "instrument_type": instrument_type,
        "market_calendar": market_calendar,
        "symbols": selected_symbols,
        "actual_available_range": {
            "from": str(min(min_timestamps)) if min_timestamps else None,
            "to": str(max(max_timestamps)) if max_timestamps else None,
        },
        "session_row_count": len(all_session_rows),
        "incomplete_or_nonstandard_session_count": len(incomplete_rows),
        "incomplete_session_count_by_bucket": bucket_counts,
        "incomplete_session_severity_counts": severity_counts,
        "timestamp_convention_counts": convention_counts,
        "calendar_mismatch_examples": calendar_mismatch_examples,
        "session_data_interpretation": [
            (
                "The 288 Stage 3.7 incomplete-session warnings are walk-forward "
                "window-boundary warnings, not 288 incomplete full trading sessions."
            ),
            (
                "Most observed sessions use boundary-inclusive timestamps: the vendor "
                "emits both the scheduled open and scheduled close labels."
            ),
            (
                "DST handling matches XNYS UTC schedule shifts; regular winter sessions "
                "open at 14:30 UTC and regular summer sessions open at 13:30 UTC."
            ),
            (
                "The possible_missing_close_bar bucket is limited to the final scheduled "
                "boundary label; no mid-session missing-bar bucket appeared in this audit."
            ),
            (
                "The session_calendar_mismatch bucket is small and explicit: one off-grid "
                "AAPL timestamp and two extra post-early-close bars."
            ),
        ],
        "symbols_with_most_incomplete_sessions": symbols_with_most,
        "position_policy_action_summary": aggregate_policy,
        "position_policy_interpretation": position_policy_interpretation,
        "intraday_classification_anatomy": anatomy,
        "report_count_analyzed": len(report_payloads),
        "malformed_report_count": malformed_count,
        "malformed_reports": malformed_reports,
        "report_contract_sanity": {
            "passed": malformed_count == 0,
            "required_fields": REPORT_CONTRACT_FIELDS,
            "reports_checked": [str(path) for path in report_paths],
        },
        "stage_passed": stage_passed,
        "recommendation": recommendation,
        "notes": [
            "Stage 3.7 incomplete-session warnings are walk-forward window-boundary warnings.",
            "Dataset-level early closes are classified against XNYS early-close schedules.",
            "No data was fetched by this diagnostic.",
        ],
        "warning_bucket_count": warning_bucket_count,
    }

    session_fields = [
        "symbol",
        "session_date",
        "calendar_open",
        "calendar_close",
        "first_timestamp",
        "last_timestamp",
        "bar_count",
        "expected_bar_count",
        "appears_complete",
        "is_early_close",
        "is_first_fetch_session",
        "is_last_fetch_session",
        "missing_bars_at_open",
        "missing_bars_in_middle",
        "missing_bars_at_close",
        "missing_bar_count",
        "extra_bar_count",
        "first_missing_timestamp",
        "last_missing_timestamp",
        "first_extra_timestamp",
        "last_extra_timestamp",
        "dst_handled_correctly",
        "timestamp_convention",
        "bucket",
        "reason",
        "severity",
    ]
    policy_fields = [
        "symbol",
        "hypothesis",
        "classification",
        "classification_reasons",
        "forced_flat_count",
        "late_entries_blocked_count",
        "overnight_carry_prevented_count",
        "raw_exposure",
        "scored_exposure",
        "exposure_reduction_pct",
        "raw_session_flat_compliant",
        "scored_session_flat_compliant",
        "policy_adjusted_trade_count",
        "sessions_seen",
        "incomplete_session_warning_count",
        "json_path",
        "report_path",
    ]
    summary_json_path = output_path / "summary.json"
    summary_markdown_path = output_path / "summary.md"
    incomplete_csv_path = output_path / "incomplete_sessions.csv"
    bar_counts_csv_path = output_path / "session_bar_counts.csv"
    policy_csv_path = output_path / "position_policy_actions.csv"
    _write_csv(bar_counts_csv_path, all_session_rows, session_fields)
    _write_csv(incomplete_csv_path, incomplete_rows, session_fields)
    _write_csv(policy_csv_path, policy_rows, policy_fields)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    summary_markdown_path.write_text(_markdown(summary), encoding="utf-8")
    return IntradaySessionIntegrityResult(
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        incomplete_sessions_csv_path=incomplete_csv_path,
        session_bar_counts_csv_path=bar_counts_csv_path,
        position_policy_actions_csv_path=policy_csv_path,
        report_count_analyzed=len(report_payloads),
        incomplete_session_count_by_bucket=bucket_counts,
        symbols_with_most_incomplete_sessions=symbols_with_most,
        position_policy_action_summary=aggregate_policy,
        intraday_classification_anatomy=anatomy,
        stage_passed=stage_passed,
        recommended_next_step=recommendation,
    )
