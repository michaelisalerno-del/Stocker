"""Read-only research report inspection tools."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from stocker_mcp.security import (
    MAX_REPORT_BYTES,
    SecurityError,
    StockerMCPContext,
    clamp_limit,
    default_context,
    is_blocked_path,
    redact_secrets,
)

CANDIDATE_PREFIX = "candidate_"
INTERESTING_PREFIX = "interesting_"
REJECTED_PREFIX = "rejected_"


def _context(context: StockerMCPContext | None) -> StockerMCPContext:
    return context or default_context()


def _json_report_paths(context: StockerMCPContext) -> list[Path]:
    paths: list[Path] = []
    for root in context.report_roots():
        paths.extend(path for path in root.rglob("*.json") if not is_blocked_path(path))
    return sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)


def _universe_paths(context: StockerMCPContext) -> list[Path]:
    return [
        path
        for path in _json_report_paths(context)
        if path.parent.name == "universe" or "symbol_results" in _safe_peek(path)
    ]


def _safe_peek(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:4_000]
    except OSError:
        return ""


def _load_payload(path: Path, context: StockerMCPContext) -> dict[str, Any]:
    roots = context.report_roots()
    if not roots:
        raise SecurityError("no report roots exist")
    resolved = context.resolve_allowed_path(path, roots)
    content, _, _, truncated = context.read_text_file(
        resolved,
        root=_containing_root(resolved, roots),
        max_bytes=MAX_REPORT_BYTES,
    )
    if truncated:
        raise SecurityError(f"report exceeds MCP read limit: {path}")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise SecurityError(f"report JSON must contain an object: {path}")
    return payload


def _containing_root(path: Path, roots: list[Path]) -> Path:
    for root in roots:
        try:
            path.resolve().relative_to(root.resolve())
            return root
        except ValueError:
            continue
    raise SecurityError(f"path is outside report roots: {path}")


def _find_universe_path(run_id_or_path: str | Path, context: StockerMCPContext) -> Path:
    requested = Path(run_id_or_path)
    roots = context.report_roots()
    if requested.suffix == ".json" or requested.is_absolute() or len(requested.parts) > 1:
        return context.resolve_allowed_path(requested, roots)
    run_id = str(run_id_or_path)
    for path in _universe_paths(context):
        if path.stem == run_id:
            return path
        try:
            payload = _load_payload(path, context)
        except (OSError, ValueError, SecurityError, json.JSONDecodeError):
            continue
        if str(payload.get("run_id", "")) == run_id:
            return path
    raise SecurityError(f"universe run not found: {run_id_or_path}")


def _symbol_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("symbol_results", [])
    return [item for item in rows if isinstance(item, dict)]


def _classification_counts(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("classification_counts")
    if isinstance(raw, dict):
        return {str(key): int(value) for key, value in raw.items()}
    return dict(
        Counter(
            str(item.get("classification"))
            for item in _symbol_results(payload)
            if item.get("classification")
        )
    )


def _candidate_count(counts: dict[str, int]) -> int:
    return sum(count for name, count in counts.items() if name.startswith(CANDIDATE_PREFIX))


def _interesting_count(counts: dict[str, int]) -> int:
    return sum(count for name, count in counts.items() if name.startswith(INTERESTING_PREFIX))


def _rejected_count(counts: dict[str, int]) -> int:
    return sum(count for name, count in counts.items() if name.startswith(REJECTED_PREFIX))


def _per_symbol(rows: list[dict[str, Any]], *, limit: int = 100) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for item in rows[:limit]:
        summaries.append(
            {
                "symbol": item.get("symbol"),
                "status": item.get("status"),
                "classification": item.get("classification"),
                "net_return": item.get("net_return"),
                "trade_count": item.get("trade_count"),
                "benchmark_pass": item.get("benchmark_pass"),
                "null_pass": item.get("null_pass"),
                "report_path": item.get("report_path"),
                "json_path": item.get("json_path"),
                "classification_reasons": item.get("classification_reasons", []),
                "error_message": redact_secrets(str(item.get("error_message", ""))),
            }
        )
    return summaries


def _run_summary(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    rows = _symbol_results(payload)
    counts = _classification_counts(payload)
    failed_count = int(
        payload.get("failed_count", sum(1 for item in rows if item.get("status") == "failed"))
    )
    md_path = path.with_suffix(".md")
    return {
        "run_id": payload.get("run_id", path.stem),
        "hypothesis_id": payload.get("hypothesis_id"),
        "universe_id": payload.get("universe_id"),
        "symbol_count": int(payload.get("symbol_count", len(rows))),
        "completed_count": int(payload.get("completed_count", 0)),
        "failed_count": failed_count,
        "classification_counts": counts,
        "candidate_count": int(payload.get("candidate_count", _candidate_count(counts))),
        "interesting_count": _interesting_count(counts),
        "rejected_count": int(payload.get("rejected_count", _rejected_count(counts))),
        "top_rejection_reasons": payload.get("top_rejection_reasons")
        or payload.get("classification_reason_counts", {}),
        "json_path": str(path),
        "markdown_path": str(md_path) if md_path.exists() else None,
    }


def workspace_doctor(context: StockerMCPContext | None = None) -> dict[str, Any]:
    """Return report-root diagnostics."""

    resolved = _context(context)
    resolved.log_tool_call("reports.workspace_doctor")
    return {
        "repo_root": str(resolved.repo_root),
        "stocker_home": str(resolved.stocker_home),
        "report_roots": [str(root) for root in resolved.report_roots()],
        "universe_run_count": len(_universe_paths(resolved)),
    }


def list_recent_research_runs(
    limit: int = 20, context: StockerMCPContext | None = None
) -> dict[str, Any]:
    """List recent universe research runs."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=20, maximum=100)
    runs: list[dict[str, Any]] = []
    for path in _universe_paths(resolved)[:safe_limit]:
        try:
            runs.append(_run_summary(_load_payload(path, resolved), path))
        except (OSError, ValueError, SecurityError, json.JSONDecodeError):
            continue
    resolved.log_tool_call("list_recent_research_runs", {"limit": safe_limit})
    return {"runs": runs, "count": len(runs)}


def get_latest_universe_run(
    hypothesis_id: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Return the latest universe run, optionally filtered by hypothesis id."""

    resolved = _context(context)
    for path in _universe_paths(resolved):
        payload = _load_payload(path, resolved)
        if hypothesis_id is not None and payload.get("hypothesis_id") != hypothesis_id:
            continue
        resolved.log_tool_call("get_latest_universe_run", {"hypothesis_id": hypothesis_id})
        return _run_summary(payload, path)
    raise SecurityError("no matching universe run found")


def read_universe_run(
    run_id_or_path: str,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Read a universe run JSON report with bounded symbol summaries."""

    resolved = _context(context)
    path = _find_universe_path(run_id_or_path, resolved)
    payload = _load_payload(path, resolved)
    summary = _run_summary(payload, path)
    summary["per_symbol"] = _per_symbol(_symbol_results(payload), limit=200)
    resolved.log_tool_call("read_universe_run", {"run_id_or_path": run_id_or_path})
    return summary


def summarise_universe_run(
    run_id_or_path: str,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Summarise classification and per-symbol results for a universe run."""

    return read_universe_run(run_id_or_path, context=context)


def list_symbol_reports(
    run_id_or_path: str | None = None,
    symbol: str | None = None,
    limit: int = 100,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """List single-symbol reports from a run or the global research index."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=100, maximum=500)
    rows: list[dict[str, Any]] = []
    if run_id_or_path:
        payload = _load_payload(_find_universe_path(run_id_or_path, resolved), resolved)
        rows = _symbol_results(payload)
    else:
        for root in resolved.report_roots():
            index = root / "index.json"
            if not index.exists():
                continue
            payload = _load_payload(index, resolved)
            entries = payload.get("experiments", [])
            rows.extend(item for item in entries if isinstance(item, dict))
    if symbol:
        rows = [item for item in rows if str(item.get("symbol", "")).upper() == symbol.upper()]
    resolved.log_tool_call(
        "list_symbol_reports",
        {"run_id_or_path": run_id_or_path, "symbol": symbol, "limit": safe_limit},
    )
    return {"reports": _per_symbol(rows, limit=safe_limit), "count": min(len(rows), safe_limit)}


def read_symbol_report(
    symbol: str | None = None,
    path: str | None = None,
    experiment_id: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Read a bounded symbol report by path, experiment id, or symbol."""

    resolved = _context(context)
    roots = resolved.report_roots()
    if path is not None:
        report_path = resolved.resolve_allowed_path(path, roots)
    else:
        matches = list_symbol_reports(symbol=symbol, limit=500, context=resolved)["reports"]
        if experiment_id is not None:
            matches = [item for item in matches if item.get("experiment_id") == experiment_id]
        if not matches:
            raise SecurityError("no matching symbol report found")
        selected = matches[0]
        report_path = resolved.resolve_allowed_path(
            str(selected.get("json_path") or selected.get("report_path")),
            roots,
        )
    root = _containing_root(report_path, roots)
    content, _, _, truncated = resolved.read_text_file(
        report_path,
        root=root,
        max_bytes=MAX_REPORT_BYTES,
    )
    resolved.log_tool_call(
        "read_symbol_report",
        {"symbol": symbol, "path": path, "experiment_id": experiment_id},
    )
    return {"path": str(report_path), "truncated": truncated, "content": content}


def compare_universe_runs(
    run_a: str,
    run_b: str,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Compare classification counts and symbol membership between two runs."""

    resolved = _context(context)
    path_a = _find_universe_path(run_a, resolved)
    path_b = _find_universe_path(run_b, resolved)
    payload_a = _load_payload(path_a, resolved)
    payload_b = _load_payload(path_b, resolved)
    symbols_a = {str(item.get("symbol")) for item in _symbol_results(payload_a)}
    symbols_b = {str(item.get("symbol")) for item in _symbol_results(payload_b)}
    counts_a = _classification_counts(payload_a)
    counts_b = _classification_counts(payload_b)
    classes = sorted(set(counts_a) | set(counts_b))
    resolved.log_tool_call("compare_universe_runs", {"run_a": run_a, "run_b": run_b})
    return {
        "run_a": _run_summary(payload_a, path_a),
        "run_b": _run_summary(payload_b, path_b),
        "classification_delta": {
            name: counts_b.get(name, 0) - counts_a.get(name, 0) for name in classes
        },
        "added_symbols": sorted(symbols_b - symbols_a),
        "removed_symbols": sorted(symbols_a - symbols_b),
    }


def find_candidate_symbols(
    run_id_or_path: str,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Return candidate symbols from a universe run."""

    return filter_symbol_results(
        run_id_or_path, classification_prefix=CANDIDATE_PREFIX, context=context
    )


def find_interesting_symbols(
    run_id_or_path: str,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Return interesting symbols from a universe run."""

    return filter_symbol_results(
        run_id_or_path, classification_prefix=INTERESTING_PREFIX, context=context
    )


def filter_symbol_results(
    run_id_or_path: str,
    classification: str | None = None,
    null_pass: bool | None = None,
    benchmark_pass: bool | None = None,
    min_net_return: float | None = None,
    min_trade_count: int | None = None,
    classification_prefix: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Filter symbol rows from one universe run."""

    resolved = _context(context)
    payload = _load_payload(_find_universe_path(run_id_or_path, resolved), resolved)
    rows = _symbol_results(payload)
    filtered: list[dict[str, Any]] = []
    for item in rows:
        item_classification = str(item.get("classification", ""))
        if classification is not None and item_classification != classification:
            continue
        if classification_prefix is not None and not item_classification.startswith(
            classification_prefix
        ):
            continue
        if null_pass is not None and bool(item.get("null_pass")) is not null_pass:
            continue
        if benchmark_pass is not None and bool(item.get("benchmark_pass")) is not benchmark_pass:
            continue
        if min_net_return is not None and float(item.get("net_return", 0.0)) < min_net_return:
            continue
        if min_trade_count is not None and int(item.get("trade_count", 0)) < min_trade_count:
            continue
        filtered.append(item)
    resolved.log_tool_call(
        "filter_symbol_results",
        {
            "run_id_or_path": run_id_or_path,
            "classification": classification,
            "null_pass": null_pass,
            "benchmark_pass": benchmark_pass,
        },
    )
    return {"symbols": _per_symbol(filtered, limit=500), "count": len(filtered)}
