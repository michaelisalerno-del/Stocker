"""Read-only code and git inspection tools scoped to the Stocker repo."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from stocker_mcp.security import (
    MAX_LIST_RESULTS,
    MAX_SEARCH_RESULTS,
    SecurityError,
    StockerMCPContext,
    clamp_limit,
    default_context,
    is_blocked_path,
    redact_secrets,
)

SAFE_GIT_REF = re.compile(r"^[A-Za-z0-9_./@{}^~:+-]+$")


def _context(context: StockerMCPContext | None) -> StockerMCPContext:
    return context or default_context()


def _run_git(
    context: StockerMCPContext, args: list[str], *, max_lines: int = 1_000
) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "-C", str(context.repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    text = result.stdout if result.returncode == 0 else result.stderr
    lines = redact_secrets(text).splitlines()
    return {
        "returncode": result.returncode,
        "output": "\n".join(lines[:max_lines]),
        "truncated": len(lines) > max_lines,
    }


def _validate_glob(pattern: str | None) -> None:
    if pattern is None:
        return
    path = Path(pattern)
    if path.is_absolute() or ".." in path.parts:
        raise SecurityError(f"glob pattern is outside the repo root: {pattern}")


def _validate_git_ref(ref: str | None) -> None:
    if ref is None:
        return
    if ref.startswith("-") or not SAFE_GIT_REF.match(ref):
        raise SecurityError(f"unsafe git ref: {ref}")


def get_repo_info(context: StockerMCPContext | None = None) -> dict[str, Any]:
    """Return basic Stocker repository metadata."""

    resolved = _context(context)
    resolved.log_tool_call("get_repo_info")
    branch = _run_git(resolved, ["branch", "--show-current"], max_lines=5)
    commit = _run_git(resolved, ["rev-parse", "HEAD"], max_lines=5)
    return {
        "repo_root": str(resolved.repo_root),
        "stocker_home": str(resolved.stocker_home),
        "branch": branch["output"].strip(),
        "commit": commit["output"].strip(),
        "has_pyproject": (resolved.repo_root / "pyproject.toml").exists(),
    }


def git_status(context: StockerMCPContext | None = None) -> dict[str, Any]:
    """Return read-only short git status."""

    resolved = _context(context)
    resolved.log_tool_call("git_status")
    return _run_git(resolved, ["status", "--short"], max_lines=500)


def git_log(limit: int = 10, context: StockerMCPContext | None = None) -> dict[str, Any]:
    """Return recent commit subjects."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=10, maximum=100)
    resolved.log_tool_call("git_log", {"limit": safe_limit})
    return _run_git(resolved, ["log", "--oneline", f"-n{safe_limit}"], max_lines=safe_limit)


def git_current_commit(context: StockerMCPContext | None = None) -> dict[str, Any]:
    """Return the current commit SHA."""

    resolved = _context(context)
    resolved.log_tool_call("git_current_commit")
    return _run_git(resolved, ["rev-parse", "HEAD"], max_lines=5)


def list_files(
    path: str = "",
    glob: str | None = None,
    limit: int = 200,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """List files under the repo root while skipping blocked paths."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=200, maximum=MAX_LIST_RESULTS)
    _validate_glob(glob)
    root = resolved.resolve_repo_path(path)
    if root.is_file():
        files = [root]
    elif glob:
        files = sorted(item for item in root.glob(glob) if item.is_file())
    else:
        files = sorted(item for item in root.rglob("*") if item.is_file())
    visible: list[str] = []
    skipped = 0
    for file_path in files:
        if is_blocked_path(file_path):
            skipped += 1
            continue
        try:
            visible.append(resolved.safe_relative(file_path))
        except OSError:
            skipped += 1
            continue
        if len(visible) >= safe_limit:
            break
    resolved.log_tool_call("list_files", {"path": path, "glob": glob, "limit": safe_limit})
    return {
        "root": resolved.safe_relative(root),
        "files": visible,
        "count": len(visible),
        "skipped": skipped,
        "truncated": len(visible) >= safe_limit,
    }


def search_code(
    query: str,
    path_glob: str | None = None,
    limit: int = 100,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Search repo text files for a literal query."""

    if not query:
        raise SecurityError("query must not be empty")
    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=100, maximum=MAX_SEARCH_RESULTS)
    _validate_glob(path_glob)
    paths = (
        sorted(resolved.repo_root.glob(path_glob))
        if path_glob
        else sorted(resolved.repo_root.rglob("*"))
    )
    matches: list[dict[str, Any]] = []
    lowered = query.lower()
    for path in paths:
        if not path.is_file() or is_blocked_path(path):
            continue
        try:
            content, _, _, _ = resolved.read_text_file(path, root=resolved.repo_root)
        except (OSError, UnicodeError, SecurityError):
            continue
        for number, line in enumerate(content.splitlines(), start=1):
            if lowered not in line.lower():
                continue
            matches.append(
                {
                    "path": resolved.safe_relative(path),
                    "line": number,
                    "text": redact_secrets(line),
                }
            )
            if len(matches) >= safe_limit:
                resolved.log_tool_call(
                    "search_code",
                    {"query": query, "path_glob": path_glob, "limit": safe_limit},
                )
                return {"matches": matches, "match_count": len(matches), "truncated": True}
    resolved.log_tool_call(
        "search_code", {"query": query, "path_glob": path_glob, "limit": safe_limit}
    )
    return {"matches": matches, "match_count": len(matches), "truncated": False}


def read_code_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Read a bounded repo file, optionally by one-indexed line range."""

    resolved = _context(context)
    file_path = resolved.resolve_repo_path(path)
    content, start, end, truncated = resolved.read_text_file(
        file_path,
        root=resolved.repo_root,
        start_line=start_line,
        end_line=end_line,
    )
    resolved.log_tool_call(
        "read_code_file",
        {"path": path, "start_line": start_line, "end_line": end_line},
    )
    return {
        "path": resolved.safe_relative(file_path),
        "start_line": start,
        "end_line": end,
        "truncated": truncated,
        "content": content,
    }


def git_diff(
    ref: str | None = None,
    path: str | None = None,
    max_lines: int = 1_000,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Return a bounded read-only git diff."""

    resolved = _context(context)
    safe_lines = clamp_limit(max_lines, default=1_000, maximum=5_000)
    _validate_git_ref(ref)
    args = ["diff", "--no-ext-diff"]
    if ref:
        args.append(ref)
    if path:
        file_path = resolved.resolve_repo_path(path)
        if not file_path.exists():
            raise SecurityError(f"path does not exist for git diff: {path}")
        args.extend(["--", resolved.safe_relative(file_path)])
    resolved.log_tool_call("git_diff", {"ref": ref, "path": path, "max_lines": safe_lines})
    return _run_git(resolved, args, max_lines=safe_lines)
