"""Central read-only safety helpers for the Stocker MCP server."""

from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 128_000
MAX_REPORT_BYTES = 2_000_000
MAX_TEXT_CHARS = 120_000
MAX_LINES = 1_000
MAX_SEARCH_RESULTS = 100
MAX_LIST_RESULTS = 500
MAX_DB_ROWS = 500

BLOCKED_NAME_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "secrets.*",
    "credentials.*",
    "token.*",
    "*apikey*",
    "*api_key*",
    "*api-key*",
)

BLOCKED_PATH_PARTS = {
    ".aws",
    ".azure",
    ".git",
    ".gcloud",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ssh",
    ".uv-cache",
    ".venv",
    "__pycache__",
}

SECRET_KEY_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b"
    r"(:\s*bearer\s+|\s*[:=]\s*)"
    r"([^\s,;\"']+)"
)
BEARER_PATTERN = re.compile(r"(?i)(Authorization:\s*Bearer\s+)([A-Za-z0-9._~+/=-]+)")
HIGH_ENTROPY_PATTERN = re.compile(
    r"\b(?=[A-Za-z0-9+=]{40,}\b)(?=.*[A-Z])(?=.*[a-z])(?=.*\d)[A-Za-z0-9+=]{40,}\b"
)


class SecurityError(ValueError):
    """Raised when a requested MCP operation violates the local safety model."""


def _resolve_existing_or_parent(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    return path.parent.resolve() / path.name


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def is_blocked_path(path: Path) -> bool:
    """Return whether a path should never be exposed by the MCP server."""

    parts = [part.lower() for part in path.parts]
    if any(part in BLOCKED_PATH_PARTS for part in parts):
        return True
    for part in parts:
        for pattern in BLOCKED_NAME_PATTERNS:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def is_sensitive_column(name: str) -> bool:
    """Return whether a database column name looks credential-like."""

    lowered = name.lower()
    return any(token in lowered for token in ("api_key", "apikey", "token", "secret", "password"))


def redact_secrets(text: str) -> str:
    """Mask secret-looking values in text output."""

    def _replace_key(match: re.Match[str]) -> str:
        separator = match.group(2)
        if "bearer" in separator.lower():
            return f"{match.group(1)}: Bearer [REDACTED]"
        return f"{match.group(1)}{separator}[REDACTED]"

    redacted = SECRET_KEY_PATTERN.sub(_replace_key, text)
    redacted = BEARER_PATTERN.sub(r"\1[REDACTED]", redacted)
    return HIGH_ENTROPY_PATTERN.sub("[REDACTED]", redacted)


def redact_value(column: str, value: Any) -> Any:
    """Redact a scalar database value if its column or content looks sensitive."""

    if value is None:
        return None
    if is_sensitive_column(column):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_secrets(value)
    return value


def clamp_limit(value: int | None, *, default: int, maximum: int) -> int:
    """Clamp user-provided limits to a safe positive range."""

    if value is None:
        return default
    return max(1, min(int(value), maximum))


def find_repo_root(start: Path | None = None) -> Path:
    """Find the Stocker project root without trusting a broad home-level git root."""

    current = (start or Path.cwd()).resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        pyproject = candidate / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'name = "stocker"' in text and (candidate / "packages").is_dir():
            return candidate
    return current


def resolve_stocker_home(value: str | Path | None = None) -> Path:
    """Resolve STOCKER_HOME, defaulting to a local user workspace."""

    raw = value if value is not None else os.environ.get("STOCKER_HOME")
    if raw is None:
        raw = Path.home() / "StockerLocal"
    return Path(raw).expanduser().resolve()


def db_enabled_from_env() -> bool:
    """Return whether DB tools are enabled by environment."""

    value = os.environ.get("STOCKER_MCP_DISABLE_DB", "")
    return value.lower() not in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StockerMCPContext:
    """Resolved roots and safety methods shared by all MCP tools."""

    repo_root: Path = field(default_factory=find_repo_root)
    stocker_home: Path = field(default_factory=resolve_stocker_home)
    log_enabled: bool = True
    db_enabled: bool = field(default_factory=db_enabled_from_env)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root).expanduser().resolve())
        object.__setattr__(self, "stocker_home", Path(self.stocker_home).expanduser().resolve())

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return (self.repo_root, self.stocker_home)

    @property
    def db_root(self) -> Path:
        return self.stocker_home / "db"

    @property
    def export_root(self) -> Path:
        return self.stocker_home / "exports"

    @property
    def log_path(self) -> Path:
        return self.stocker_home / "logs" / "mcp_tool_calls.jsonl"

    def report_roots(self) -> list[Path]:
        roots = [
            self.stocker_home / "data" / "reports" / "research",
            self.repo_root / "data" / "reports" / "research",
        ]
        return [root.resolve() for root in roots if root.exists()]

    def resolve_under_root(self, root: Path, user_path: str | Path = "") -> Path:
        """Resolve a user path under a specific root after symlink normalization."""

        root_resolved = Path(root).expanduser().resolve()
        requested = Path(user_path).expanduser()
        candidate = requested if requested.is_absolute() else root_resolved / requested
        resolved = _resolve_existing_or_parent(candidate)
        if not _is_relative_to(resolved, root_resolved):
            raise SecurityError(f"path is outside allowed root: {user_path}")
        if is_blocked_path(resolved):
            raise SecurityError(f"path is blocked by Stocker MCP policy: {user_path}")
        return resolved

    def resolve_repo_path(self, path: str | Path = "") -> Path:
        return self.resolve_under_root(self.repo_root, path)

    def resolve_allowed_path(self, path: str | Path, roots: list[Path] | tuple[Path, ...]) -> Path:
        """Resolve a path under one of the supplied roots."""

        requested = Path(path).expanduser()
        for root in roots:
            try:
                return self.resolve_under_root(root, requested)
            except SecurityError:
                continue
        raise SecurityError(f"path is outside allowed roots: {path}")

    def safe_relative(self, path: Path, root: Path | None = None) -> str:
        base = (root or self.repo_root).resolve()
        try:
            return path.resolve().relative_to(base).as_posix()
        except ValueError:
            return str(path.resolve())

    def read_text_file(
        self,
        path: str | Path,
        *,
        root: Path,
        start_line: int | None = None,
        end_line: int | None = None,
        max_bytes: int = MAX_FILE_BYTES,
    ) -> tuple[str, int, int, bool]:
        """Read a bounded text file and return content plus range metadata."""

        resolved = self.resolve_under_root(root, path)
        if not resolved.is_file():
            raise SecurityError(f"not a readable file: {path}")
        size = resolved.stat().st_size
        truncated = size > max_bytes
        with resolved.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
            truncated = True
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total_lines = len(lines)
        start = 1 if start_line is None else max(1, int(start_line))
        end = total_lines if end_line is None else max(start, int(end_line))
        if end - start + 1 > MAX_LINES:
            end = start + MAX_LINES - 1
            truncated = True
        selected = lines[start - 1 : end]
        content = "\n".join(selected)
        if text.endswith("\n") and selected:
            content += "\n"
        return redact_secrets(content[:MAX_TEXT_CHARS]), start, min(end, total_lines), truncated

    def read_json_file(
        self, path: str | Path, *, root: Path, max_bytes: int = MAX_REPORT_BYTES
    ) -> Any:
        content, _, _, truncated = self.read_text_file(path, root=root, max_bytes=max_bytes)
        if truncated:
            raise SecurityError(f"JSON file exceeds MCP read limit: {path}")
        return json.loads(content)

    def log_tool_call(self, tool: str, params: dict[str, Any] | None = None) -> None:
        """Write a redacted local audit record, ignoring logging failures."""

        if not self.log_enabled:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
                "tool": tool,
                "params": json.loads(redact_secrets(json.dumps(params or {}, default=str))),
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            return


def default_context() -> StockerMCPContext:
    """Build a context from the current process environment."""

    return StockerMCPContext()
