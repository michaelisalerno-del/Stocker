"""Read-only discovery workflow summaries for Stocker MCP."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from stocker_mcp.security import StockerMCPContext, clamp_limit, default_context, redact_secrets
from stocker_mcp.tools import database, reports

QUESTION_BENCHMARK_FAILURE = (
    "Benchmark failure dominates; inspect regime and benchmark attribution."
)
QUESTION_NULL_FAILURE = (
    "Null-model failure dominates; inspect whether timing edge survives shuffles."
)
QUESTION_COST_FAILURE = (
    "Costs appear to kill this setup; inspect spread and slippage sensitivity."
)
QUESTION_LOW_SAMPLE = (
    "Trade count is too low; inspect whether filters are over-constraining samples."
)
QUESTION_NO_CANDIDATES = (
    "No candidates survived; inspect which gate rejects most completed symbols."
)


def _context(context: StockerMCPContext | None) -> StockerMCPContext:
    return context or default_context()


def _run_id(run_id: str | None, context: StockerMCPContext) -> str:
    if run_id:
        return run_id
    latest = reports.get_latest_universe_run(context=context)
    return str(latest["run_id"])


def _run_payload(run_id: str | None, context: StockerMCPContext) -> dict[str, Any]:
    return reports.read_universe_run(_run_id(run_id, context), context=context)


def _rows(run_id: str | None, context: StockerMCPContext) -> list[dict[str, Any]]:
    return [
        item
        for item in _run_payload(run_id, context).get("per_symbol", [])
        if isinstance(item, dict)
    ]


def _is_rejected(item: dict[str, Any]) -> bool:
    return str(item.get("classification", "")).startswith("rejected_")


def _float_value(item: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(item.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _int_value(item: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(item.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _symbol_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": item.get("symbol"),
        "classification": item.get("classification"),
        "net_return": item.get("net_return"),
        "trade_count": item.get("trade_count"),
        "benchmark_pass": item.get("benchmark_pass"),
        "null_pass": item.get("null_pass"),
        "classification_reasons": item.get("classification_reasons", []),
        "status": item.get("status"),
    }


def summarise_latest_research_state(context: StockerMCPContext | None = None) -> dict[str, Any]:
    """Summarise the latest research state without trading recommendations."""

    resolved = _context(context)
    latest = reports.get_latest_universe_run(context=resolved)
    common = find_common_rejection_reasons(str(latest["run_id"]), context=resolved)
    positives = find_positive_rejected_symbols(str(latest["run_id"]), context=resolved)
    resolved.log_tool_call("summarise_latest_research_state")
    return {
        "latest_run": latest,
        "common_rejection_reasons": common["reasons"],
        "positive_rejected_count": positives["count"],
        "candidate_count": latest.get("candidate_count", 0),
        "interesting_count": latest.get("interesting_count", 0),
        "rejected_count": latest.get("rejected_count", 0),
        "failed_count": latest.get("failed_count", 0),
        "note": "Historical research summary only. No trading, execution, or live signal.",
    }


def find_positive_rejected_symbols(
    run_id: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Find rejected symbols with positive historical net return."""

    resolved = _context(context)
    selected = [
        _symbol_summary(item)
        for item in _rows(run_id, resolved)
        if _is_rejected(item) and _float_value(item, "net_return") > 0
    ]
    resolved.log_tool_call("find_positive_rejected_symbols", {"run_id": run_id})
    return {"symbols": selected, "count": len(selected), "run_id": _run_id(run_id, resolved)}


def find_null_pass_benchmark_fail_symbols(
    run_id: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Find symbols that passed null checks but failed benchmark checks."""

    resolved = _context(context)
    selected = [
        _symbol_summary(item)
        for item in _rows(run_id, resolved)
        if bool(item.get("null_pass")) and not bool(item.get("benchmark_pass"))
    ]
    resolved.log_tool_call("find_null_pass_benchmark_fail_symbols", {"run_id": run_id})
    return {"symbols": selected, "count": len(selected), "run_id": _run_id(run_id, resolved)}


def find_benchmark_pass_rejected_symbols(
    run_id: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Find rejected symbols that passed benchmark checks."""

    resolved = _context(context)
    selected = [
        _symbol_summary(item)
        for item in _rows(run_id, resolved)
        if _is_rejected(item) and bool(item.get("benchmark_pass"))
    ]
    resolved.log_tool_call("find_benchmark_pass_rejected_symbols", {"run_id": run_id})
    return {"symbols": selected, "count": len(selected), "run_id": _run_id(run_id, resolved)}


def find_common_rejection_reasons(
    run_id: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Count common rejection reasons in a universe run."""

    resolved = _context(context)
    payload = _run_payload(run_id, resolved)
    counter: Counter[str] = Counter()
    for item in payload.get("per_symbol", []):
        reasons = item.get("classification_reasons", [])
        if isinstance(reasons, list):
            counter.update(str(reason) for reason in reasons if reason)
    if not counter:
        raw = payload.get("top_rejection_reasons", {})
        if isinstance(raw, dict):
            counter.update({str(key): int(value) for key, value in raw.items()})
    reasons = [{"reason": key, "count": count} for key, count in counter.most_common(20)]
    resolved.log_tool_call("find_common_rejection_reasons", {"run_id": run_id})
    return {"reasons": reasons, "count": len(reasons), "run_id": _run_id(run_id, resolved)}


def get_symbol_bar_summary(
    symbol: str,
    timeframe: str = "5m",
    start: str | None = None,
    end: str | None = None,
    database_name: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Summarise bounded historical bars for one symbol and timeframe."""

    resolved = _context(context)
    result = database.db_get_symbol_bars(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        limit=500,
        database=database_name,
        context=resolved,
    )
    rows = [item for item in result.get("rows", []) if isinstance(item, dict)]
    closes = [_float_value(item, "close") for item in rows if item.get("close") is not None]
    volumes = [_float_value(item, "volume") for item in rows if item.get("volume") is not None]
    timestamps = [str(item.get("timestamp")) for item in rows if item.get("timestamp")]
    resolved.log_tool_call(
        "get_symbol_bar_summary",
        {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end},
    )
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "row_count": len(rows),
        "first_timestamp": timestamps[0] if timestamps else None,
        "latest_timestamp": timestamps[-1] if timestamps else None,
        "first_close": closes[0] if closes else None,
        "latest_close": closes[-1] if closes else None,
        "min_close": min(closes) if closes else None,
        "max_close": max(closes) if closes else None,
        "average_close": mean(closes) if closes else None,
        "total_volume": sum(volumes) if volumes else None,
        "database": result.get("database"),
        "note": "Historical bar summary only. No trading or execution signal.",
    }


def get_symbol_recent_sessions(
    symbol: str,
    timeframe: str = "5m",
    limit: int = 20,
    database_name: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Summarise recent historical sessions for one symbol and timeframe."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=20, maximum=100)
    result = database.db_get_symbol_bars(
        symbol=symbol,
        timeframe=timeframe,
        limit=500,
        database=database_name,
        context=resolved,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in result.get("rows", []):
        if not isinstance(item, dict):
            continue
        timestamp = str(item.get("timestamp", ""))
        if not timestamp:
            continue
        grouped[timestamp[:10]].append(item)
    sessions: list[dict[str, Any]] = []
    for date in sorted(grouped, reverse=True)[:safe_limit]:
        rows = grouped[date]
        closes = [_float_value(item, "close") for item in rows if item.get("close") is not None]
        sessions.append(
            {
                "date": date,
                "bar_count": len(rows),
                "first_timestamp": rows[0].get("timestamp"),
                "latest_timestamp": rows[-1].get("timestamp"),
                "first_close": closes[0] if closes else None,
                "latest_close": closes[-1] if closes else None,
            }
        )
    resolved.log_tool_call(
        "get_symbol_recent_sessions",
        {"symbol": symbol, "timeframe": timeframe, "limit": safe_limit},
    )
    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "sessions": sessions,
        "count": len(sessions),
        "note": "Historical session summary only. No trading or execution signal.",
    }


def get_trade_feature_buckets(
    run_id: str | None = None,
    feature: str | None = None,
    symbol: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Summarise trade-attribution feature buckets when available."""

    resolved = _context(context)
    result = database.db_get_trade_attribution(
        run_id=run_id,
        symbol=symbol,
        limit=500,
        context=resolved,
    )
    rows = [item for item in result.get("rows", []) if isinstance(item, dict)]
    bucket_feature = feature or "classification"
    buckets: dict[str, dict[str, Any]] = {}
    for item in rows:
        raw_value = item.get(bucket_feature)
        bucket = redact_secrets(str(raw_value if raw_value is not None else "unknown"))
        current = buckets.setdefault(
            bucket,
            {"bucket": bucket, "trade_count": 0, "net_returns": []},
        )
        current["trade_count"] += 1
        if item.get("net_return") is not None:
            current["net_returns"].append(_float_value(item, "net_return"))
    summaries: list[dict[str, Any]] = []
    for bucket in buckets.values():
        net_returns = bucket.pop("net_returns")
        bucket["average_net_return"] = mean(net_returns) if net_returns else None
        summaries.append(bucket)
    summaries.sort(key=lambda item: int(item["trade_count"]), reverse=True)
    resolved.log_tool_call(
        "get_trade_feature_buckets",
        {"run_id": run_id, "feature": feature, "symbol": symbol},
    )
    return {
        "feature": bucket_feature,
        "buckets": summaries,
        "count": len(summaries),
        "source_row_count": len(rows),
        "message": result.get("message"),
        "note": "Historical attribution summary only. No trading or execution signal.",
    }


def compare_template_runs(
    run_a: str,
    run_b: str,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Compare two Stocker research/template universe runs."""

    resolved = _context(context)
    resolved.log_tool_call("compare_template_runs", {"run_a": run_a, "run_b": run_b})
    return reports.compare_universe_runs(run_a=run_a, run_b=run_b, context=resolved)


def suggest_research_questions(
    run_id: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Suggest conservative evidence-based research questions from existing results."""

    resolved = _context(context)
    payload = _run_payload(run_id, resolved)
    common = find_common_rejection_reasons(run_id, context=resolved)
    questions: list[dict[str, str]] = []
    for reason in common["reasons"][:5]:
        name = str(reason["reason"])
        if "benchmark" in name:
            questions.append(
                {
                    "question": QUESTION_BENCHMARK_FAILURE,
                    "evidence": name,
                }
            )
        elif "null" in name:
            questions.append(
                {
                    "question": QUESTION_NULL_FAILURE,
                    "evidence": name,
                }
            )
        elif "cost" in name or "spread" in name:
            questions.append(
                {
                    "question": QUESTION_COST_FAILURE,
                    "evidence": name,
                }
            )
        elif "trade" in name or "sample" in name:
            questions.append(
                {
                    "question": QUESTION_LOW_SAMPLE,
                    "evidence": name,
                }
            )
        else:
            questions.append(
                {
                    "question": (
                        f"{name} is common; inspect symbol-level attribution before "
                        "template changes."
                    ),
                    "evidence": name,
                }
            )
    counts = payload.get("classification_counts", {})
    if isinstance(counts, dict):
        rejected_total = sum(
            int(value) for key, value in counts.items() if str(key).startswith("rejected_")
        )
        candidate_total = sum(
            int(value) for key, value in counts.items() if str(key).startswith("candidate_")
        )
        if rejected_total and candidate_total == 0:
            questions.append(
                {
                    "question": QUESTION_NO_CANDIDATES,
                    "evidence": "classification_counts",
                }
            )
    if not questions:
        questions.append(
            {
                "question": "Needs ORB and VWAP attribution before changing template assumptions.",
                "evidence": "no dominant rejection reason",
            }
        )
    unique_by_question = {item["question"]: item for item in questions}
    resolved.log_tool_call("suggest_research_questions", {"run_id": run_id})
    return {
        "run_id": _run_id(run_id, resolved),
        "questions": list(unique_by_question.values())[:8],
        "basis": {
            "classification_counts": payload.get("classification_counts", {}),
            "top_rejection_reasons": common["reasons"][:5],
        },
        "note": "Research questions only. No trading, execution, or deployment recommendation.",
    }
