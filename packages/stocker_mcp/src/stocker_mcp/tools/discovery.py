"""ChatGPT-compatible search and fetch tools for Stocker MCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from stocker_mcp.security import (
    MAX_LIST_RESULTS,
    MAX_REPORT_BYTES,
    MAX_TEXT_CHARS,
    SecurityError,
    StockerMCPContext,
    clamp_limit,
    default_context,
    is_blocked_path,
    redact_secrets,
)
from stocker_mcp.tools import code, database, reports, workspace

MAX_FETCH_CHARS = 24_000
MAX_SEARCH_TEXT_BYTES = 80_000
SUPPORTED_PREFIXES = {
    "reports",
    "runs",
    "symbols",
    "code",
    "hypotheses",
    "db",
    "workspace",
}


def _context(context: StockerMCPContext | None) -> StockerMCPContext:
    return context or default_context()


def _encode(value: str) -> str:
    return quote(value, safe="")


def _decode(value: str) -> str:
    return unquote(value)


def _short_text(text: str, limit: int = MAX_FETCH_CHARS) -> tuple[str, bool]:
    redacted = redact_secrets(text)
    if len(redacted) <= limit:
        return redacted, False
    return redacted[:limit], True


def _safe_json_text(payload: Any) -> str:
    return redact_secrets(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _add_result(
    results: list[dict[str, Any]],
    seen: set[str],
    *,
    item_id: str,
    title: str,
    source: str,
    limit: int,
) -> None:
    if len(results) >= limit or item_id in seen:
        return
    seen.add(item_id)
    results.append({"id": item_id, "title": redact_secrets(title), "url": "", "source": source})


def _report_root_key(path: Path, context: StockerMCPContext) -> tuple[str, Path]:
    home_root = context.stocker_home / "data" / "reports" / "research"
    repo_root = context.repo_root / "data" / "reports" / "research"
    for key, root in (("home", home_root), ("repo", repo_root)):
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        return key, root.resolve()
    raise SecurityError(f"report path outside report roots: {path}")


def _hypothesis_paths(context: StockerMCPContext) -> list[Path]:
    roots = [
        context.repo_root / "research" / "hypotheses",
        context.repo_root / "configs",
        context.repo_root / "packages",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for suffix in ("*.yaml", "*.yml", "*.json"):
            paths.extend(path for path in root.rglob(suffix) if path.is_file())
    return sorted(path for path in paths if not is_blocked_path(path))


def _search_reports(
    query: str,
    context: StockerMCPContext,
    results: list[dict[str, Any]],
    seen: set[str],
    limit: int,
) -> None:
    lowered = query.lower()
    try:
        recent = reports.list_recent_research_runs(limit=20, context=context)
    except Exception:
        recent = {"runs": []}
    for run in recent.get("runs", []):
        haystack = _safe_json_text(run).lower()
        run_id = str(run.get("run_id", ""))
        run_matches = lowered in haystack or lowered in run_id.lower()
        if run_matches:
            _add_result(
                results,
                seen,
                item_id=f"stocker://runs/{_encode(run_id)}",
                title=f"Universe run {run_id}",
                source="runs",
                limit=limit,
            )
        for key in ("json_path", "markdown_path"):
            raw_path = run.get(key)
            if not raw_path:
                continue
            path = Path(str(raw_path))
            if not run_matches and lowered not in f"{path.name} {path}".lower():
                continue
            try:
                root_key, root = _report_root_key(path, context)
                rel = path.resolve().relative_to(root).as_posix()
            except (OSError, SecurityError, ValueError):
                continue
            _add_result(
                results,
                seen,
                item_id=f"stocker://reports/{root_key}/{_encode(rel)}",
                title=f"{path.name} ({run_id})",
                source="reports",
                limit=limit,
            )
    for root in context.report_roots():
        for path in sorted(root.rglob("*")):
            if len(results) >= limit:
                return
            if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt"}:
                continue
            if is_blocked_path(path):
                continue
            try:
                rel = path.resolve().relative_to(root.resolve()).as_posix()
                content, _, _, _ = context.read_text_file(
                    rel,
                    root=root,
                    max_bytes=min(MAX_REPORT_BYTES, MAX_SEARCH_TEXT_BYTES),
                )
            except (OSError, SecurityError, UnicodeError):
                continue
            match_text = f"{rel} {path.name} {content}".lower()
            if lowered not in match_text:
                continue
            root_key, _ = _report_root_key(path, context)
            _add_result(
                results,
                seen,
                item_id=f"stocker://reports/{root_key}/{_encode(rel)}",
                title=path.name,
                source="reports",
                limit=limit,
            )


def _search_hypotheses(
    query: str,
    context: StockerMCPContext,
    results: list[dict[str, Any]],
    seen: set[str],
    limit: int,
) -> None:
    lowered = query.lower()
    for path in _hypothesis_paths(context):
        if len(results) >= limit:
            return
        try:
            rel = path.resolve().relative_to(context.repo_root).as_posix()
            content, _, _, _ = context.read_text_file(
                rel,
                root=context.repo_root,
                max_bytes=MAX_SEARCH_TEXT_BYTES,
            )
        except (OSError, SecurityError, UnicodeError, ValueError):
            continue
        if lowered not in rel.lower() and lowered not in content.lower():
            continue
        _add_result(
            results,
            seen,
            item_id=f"stocker://hypotheses/{_encode(rel)}",
            title=path.name,
            source="hypotheses",
            limit=limit,
        )


def _search_code(
    query: str,
    context: StockerMCPContext,
    results: list[dict[str, Any]],
    seen: set[str],
    limit: int,
) -> None:
    try:
        matches = code.search_code(query=query, limit=min(40, limit), context=context)
    except SecurityError:
        return
    for match in matches.get("matches", []):
        if len(results) >= limit:
            return
        path = str(match.get("path", ""))
        line = match.get("line")
        title = f"{path}:{line}" if line else path
        suffix = f"#L{line}" if line else ""
        _add_result(
            results,
            seen,
            item_id=f"stocker://code/{_encode(path)}{suffix}",
            title=title,
            source="code",
            limit=limit,
        )


def _search_db(
    query: str,
    context: StockerMCPContext,
    results: list[dict[str, Any]],
    seen: set[str],
    limit: int,
) -> None:
    lowered = query.lower()
    try:
        dbs = database.db_list_databases(context=context)
    except Exception:
        return
    for db_item in dbs.get("databases", []):
        if len(results) >= limit:
            return
        db_name = str(db_item.get("name", ""))
        try:
            tables = database.db_list_tables(database=db_name, context=context)
        except Exception:
            continue
        for table in tables.get("tables", []):
            if len(results) >= limit:
                return
            table_name = str(table)
            haystack = f"{db_name} {table_name}".lower()
            try:
                described = database.db_describe_table(
                    table_name,
                    database=db_name,
                    context=context,
                )
                haystack += " " + _safe_json_text(described).lower()
            except Exception:
                described = {}
            if lowered not in haystack:
                continue
            _add_result(
                results,
                seen,
                item_id=f"stocker://db/{_encode(db_name)}/{_encode(table_name)}",
                title=f"{db_name}: {table_name}",
                source="db",
                limit=limit,
            )
            if lowered in "catalyst news events":
                _add_result(
                    results,
                    seen,
                    item_id=f"stocker://db/{_encode(db_name)}/{_encode(table_name)}",
                    title=f"{db_name}: {table_name}",
                    source="db",
                    limit=limit,
                )


def search(
    query: str,
    limit: int = 20,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Search Stocker code, reports, hypotheses, and database schemas."""

    if not query or not query.strip():
        raise SecurityError("query must not be empty")
    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=20, maximum=MAX_LIST_RESULTS)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    _search_reports(query, resolved, results, seen, safe_limit)
    _search_hypotheses(query, resolved, results, seen, safe_limit)
    _search_code(query, resolved, results, seen, safe_limit)
    _search_db(query, resolved, results, seen, safe_limit)
    if query.lower() in "workspace doctor status":
        _add_result(
            results,
            seen,
            item_id="stocker://workspace/doctor",
            title="Stocker workspace doctor",
            source="workspace",
            limit=safe_limit,
        )
    resolved.log_tool_call("search", {"query": query, "limit": safe_limit})
    return {"results": results, "count": len(results), "truncated": len(results) >= safe_limit}


def _parse_id(item_id: str) -> tuple[str, list[str], str | None]:
    parsed = urlparse(item_id)
    if parsed.scheme != "stocker" or parsed.netloc not in SUPPORTED_PREFIXES:
        raise SecurityError(f"unsupported Stocker id: {item_id}")
    if parsed.params or parsed.query:
        raise SecurityError(f"unsupported Stocker id syntax: {item_id}")
    parts = [_decode(part) for part in parsed.path.lstrip("/").split("/") if part]
    line_anchor = None
    if parsed.fragment:
        if not parsed.fragment.startswith("L") or not parsed.fragment[1:].isdigit():
            raise SecurityError(f"unsupported Stocker id anchor: {item_id}")
        line_anchor = parsed.fragment
    return parsed.netloc, parts, line_anchor


def _fetch_code(
    parts: list[str],
    line_anchor: str | None,
    context: StockerMCPContext,
) -> dict[str, Any]:
    if len(parts) != 1:
        raise SecurityError("code fetch id must contain one encoded repo-relative path")
    path = parts[0]
    start_line = int(line_anchor[1:]) if line_anchor else None
    end_line = start_line + 80 if start_line else None
    result = code.read_code_file(path, start_line=start_line, end_line=end_line, context=context)
    text, truncated = _short_text(str(result.get("content", "")))
    return {
        "id": f"stocker://code/{_encode(path)}" + (f"#{line_anchor}" if line_anchor else ""),
        "title": path,
        "text": text,
        "url": "",
        "metadata": {
            "source": "code",
            "path": result.get("path"),
            "start_line": result.get("start_line"),
            "end_line": result.get("end_line"),
            "truncated": bool(result.get("truncated")) or truncated,
        },
    }


def _fetch_hypothesis(parts: list[str], context: StockerMCPContext) -> dict[str, Any]:
    if len(parts) != 1:
        raise SecurityError("hypothesis fetch id must contain one encoded repo-relative path")
    path = parts[0]
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts or "hypotheses" not in rel.parts:
        raise SecurityError(f"unsupported hypothesis id path: {path}")
    content, _, _, truncated = context.read_text_file(path, root=context.repo_root)
    text, text_truncated = _short_text(content)
    return {
        "id": f"stocker://hypotheses/{_encode(path)}",
        "title": Path(path).name,
        "text": text,
        "url": "",
        "metadata": {
            "source": "hypothesis",
            "path": path,
            "truncated": truncated or text_truncated,
        },
    }


def _fetch_report(parts: list[str], context: StockerMCPContext) -> dict[str, Any]:
    if len(parts) != 2:
        raise SecurityError("report fetch id must contain root key and encoded relative path")
    root_key, rel = parts
    if root_key == "home":
        root = context.stocker_home / "data" / "reports" / "research"
    elif root_key == "repo":
        root = context.repo_root / "data" / "reports" / "research"
    else:
        raise SecurityError(f"unsupported report root key: {root_key}")
    content, _, _, truncated = context.read_text_file(
        rel,
        root=root,
        max_bytes=min(MAX_REPORT_BYTES, MAX_TEXT_CHARS),
    )
    text, text_truncated = _short_text(content)
    return {
        "id": f"stocker://reports/{root_key}/{_encode(rel)}",
        "title": Path(rel).name,
        "text": text,
        "url": "",
        "metadata": {
            "source": "reports",
            "path": rel,
            "truncated": truncated or text_truncated,
        },
    }


def _fetch_run(parts: list[str], context: StockerMCPContext) -> dict[str, Any]:
    if len(parts) != 1:
        raise SecurityError("run fetch id must contain one run id")
    run_id = parts[0]
    summary = reports.summarise_universe_run(run_id, context=context)
    text, truncated = _short_text(_safe_json_text(summary))
    return {
        "id": f"stocker://runs/{_encode(run_id)}",
        "title": f"Universe run {run_id}",
        "text": text,
        "url": "",
        "metadata": {"source": "reports", "run_id": run_id, "truncated": truncated},
    }


def _fetch_symbol(parts: list[str], context: StockerMCPContext) -> dict[str, Any]:
    if len(parts) != 1:
        raise SecurityError("symbol fetch id must contain one symbol")
    symbol = parts[0].upper()
    result = reports.list_symbol_reports(symbol=symbol, limit=20, context=context)
    text, truncated = _short_text(_safe_json_text(result))
    return {
        "id": f"stocker://symbols/{_encode(symbol)}",
        "title": f"Symbol reports for {symbol}",
        "text": text,
        "url": "",
        "metadata": {"source": "reports", "symbol": symbol, "truncated": truncated},
    }


def _fetch_db(parts: list[str], context: StockerMCPContext) -> dict[str, Any]:
    if len(parts) > 2:
        raise SecurityError("database fetch id may contain database and optional table")
    if not parts:
        result = database.db_list_databases(context=context)
        title = "Stocker databases"
    elif len(parts) == 1:
        result = database.db_list_tables(database=parts[0], context=context)
        title = f"Database {parts[0]}"
    else:
        result = database.db_describe_table(parts[1], database=parts[0], context=context)
        title = f"{parts[0]}: {parts[1]}"
    text, truncated = _short_text(_safe_json_text(result))
    return {
        "id": "stocker://db/" + "/".join(_encode(part) for part in parts),
        "title": title,
        "text": text,
        "url": "",
        "metadata": {"source": "db", "path": "/".join(parts), "truncated": truncated},
    }


def _fetch_workspace(parts: list[str], context: StockerMCPContext) -> dict[str, Any]:
    if parts != ["doctor"]:
        raise SecurityError("workspace fetch only supports stocker://workspace/doctor")
    result = workspace.workspace_doctor(context=context)
    text, truncated = _short_text(_safe_json_text(result))
    return {
        "id": "stocker://workspace/doctor",
        "title": "Stocker workspace doctor",
        "text": text,
        "url": "",
        "metadata": {"source": "workspace", "path": "doctor", "truncated": truncated},
    }


def fetch(id: str, context: StockerMCPContext | None = None) -> dict[str, Any]:
    """Fetch one safe Stocker item by stocker:// id."""

    resolved = _context(context)
    prefix, parts, line_anchor = _parse_id(id)
    handlers = {
        "code": _fetch_code,
        "hypotheses": _fetch_hypothesis,
        "reports": _fetch_report,
        "runs": _fetch_run,
        "symbols": _fetch_symbol,
        "db": _fetch_db,
        "workspace": _fetch_workspace,
    }
    if prefix == "code":
        result = handlers[prefix](parts, line_anchor, resolved)
    else:
        if line_anchor is not None:
            raise SecurityError(f"anchors are unsupported for {prefix} ids")
        result = handlers[prefix](parts, resolved)
    resolved.log_tool_call("fetch", {"id": id})
    result["text"] = redact_secrets(str(result.get("text", "")))[:MAX_FETCH_CHARS]
    return result
