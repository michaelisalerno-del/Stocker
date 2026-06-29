"""Read-only database tools scoped to STOCKER_HOME/db."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from stocker_mcp.security import (
    MAX_DB_ROWS,
    SecurityError,
    StockerMCPContext,
    clamp_limit,
    default_context,
    is_blocked_path,
    redact_value,
)

SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
DUCKDB_SUFFIXES = {".duckdb", ".ddb"}
TABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
BANNED_SQL_PATTERN = re.compile(
    r"\b(attach|detach|copy|export|install|load|pragma|create|insert|update|delete|drop|alter|"
    r"truncate|replace|vacuum|analyze)\b",
    re.IGNORECASE,
)
FILE_FUNCTION_PATTERN = re.compile(
    r"\b(read_csv|read_parquet|read_json|read_text|glob|sqlite_scan|parquet_scan)\s*\(",
    re.IGNORECASE,
)


def _context(context: StockerMCPContext | None) -> StockerMCPContext:
    return context or default_context()


def _db_files(context: StockerMCPContext) -> list[Path]:
    if not context.db_root.exists():
        return []
    suffixes = SQLITE_SUFFIXES | DUCKDB_SUFFIXES
    return sorted(
        path
        for path in context.db_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and not is_blocked_path(path)
    )


def _db_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SQLITE_SUFFIXES:
        return "sqlite"
    if suffix in DUCKDB_SUFFIXES:
        return "duckdb"
    raise SecurityError(f"unsupported database type: {path.name}")


def _resolve_database(database: str | None, context: StockerMCPContext) -> Path:
    if not context.db_enabled:
        raise SecurityError("database tools are disabled by Stocker MCP configuration")
    files = _db_files(context)
    if database is None:
        if not files:
            raise SecurityError(f"no databases found under {context.db_root}")
        return files[0]
    requested = Path(database)
    if requested.is_absolute() or len(requested.parts) > 1:
        path = context.resolve_under_root(context.db_root, requested)
        if path not in files:
            raise SecurityError(f"database is not registered under db root: {database}")
        return path
    for path in files:
        if database in {path.name, path.stem}:
            return path
    raise SecurityError(f"database not found: {database}")


def _sqlite_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _duckdb_connection(path: Path) -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise SecurityError("duckdb is not installed in this environment") from exc
    return duckdb.connect(str(path), read_only=True)


def _quote_identifier(name: str) -> str:
    parts = name.split(".")
    if not parts or any(not TABLE_PATTERN.match(part) for part in parts):
        raise SecurityError(f"unsafe table identifier: {name}")
    return ".".join(f'"{part}"' for part in parts)


def _redact_rows(columns: Iterable[str], rows: Iterable[Iterable[Any]]) -> list[dict[str, Any]]:
    names = [str(column) for column in columns]
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {name: redact_value(name, value) for name, value in zip(names, row, strict=True)}
        )
    return output


def _execute(path: Path, sql: str, params: tuple[Any, ...] = (), *, limit: int) -> dict[str, Any]:
    limited_sql = f"select * from ({sql}) as stocker_mcp_query limit {limit}"
    db_type = _db_type(path)
    if db_type == "sqlite":
        with _sqlite_connection(path) as connection:
            cursor = connection.execute(limited_sql, params)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description or []]
    else:
        with _duckdb_connection(path) as connection:
            cursor = connection.execute(limited_sql, params)
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description or []]
    return {
        "database": str(path),
        "database_type": db_type,
        "columns": columns,
        "rows": _redact_rows(columns, rows),
        "row_count": len(rows),
        "limit": limit,
    }


def _table_names(path: Path) -> list[str]:
    if _db_type(path) == "sqlite":
        with _sqlite_connection(path) as connection:
            rows = connection.execute(
                "select name from sqlite_master where type in ('table', 'view') "
                "and name not like 'sqlite_%' order by name"
            ).fetchall()
            return [str(row[0]) for row in rows]
    with _duckdb_connection(path) as connection:
        rows = connection.execute("show tables").fetchall()
        return [str(row[0]) for row in rows]


def _require_known_table(path: Path, table: str) -> str:
    if table not in _table_names(path):
        raise SecurityError(f"table not found: {table}")
    return _quote_identifier(table)


def _validate_select_sql(sql: str) -> str:
    stripped = sql.strip()
    if not stripped:
        raise SecurityError("sql must not be empty")
    if ";" in stripped:
        raise SecurityError("semicolon-delimited SQL is not allowed")
    lowered = stripped.lower()
    if not lowered.startswith("select "):
        raise SecurityError("only SELECT statements are allowed")
    if BANNED_SQL_PATTERN.search(stripped):
        raise SecurityError("SQL contains a blocked statement or keyword")
    if FILE_FUNCTION_PATTERN.search(stripped):
        raise SecurityError("SQL file-reading functions are blocked")
    if "--" in stripped or "/*" in stripped:
        raise SecurityError("SQL comments are blocked")
    return stripped


def db_list_databases(context: StockerMCPContext | None = None) -> dict[str, Any]:
    """List supported local database files under STOCKER_HOME/db."""

    resolved = _context(context)
    if not resolved.db_enabled:
        return {
            "db_root": str(resolved.db_root),
            "databases": [],
            "count": 0,
            "enabled": False,
        }
    databases = [
        {
            "name": path.name,
            "path": str(path),
            "type": _db_type(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _db_files(resolved)
    ]
    resolved.log_tool_call("db_list_databases")
    return {
        "db_root": str(resolved.db_root),
        "databases": databases,
        "count": len(databases),
        "enabled": True,
    }


def db_list_tables(
    database: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """List tables in a selected database."""

    resolved = _context(context)
    path = _resolve_database(database, resolved)
    tables = _table_names(path)
    resolved.log_tool_call("db_list_tables", {"database": database})
    return {"database": str(path), "database_type": _db_type(path), "tables": tables}


def db_describe_table(
    table: str,
    database: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Describe columns for one known table."""

    resolved = _context(context)
    path = _resolve_database(database, resolved)
    quoted = _require_known_table(path, table)
    if _db_type(path) == "sqlite":
        with _sqlite_connection(path) as connection:
            rows = connection.execute(f"pragma table_info({quoted})").fetchall()
            columns = [
                {"name": row["name"], "type": row["type"], "not_null": bool(row["notnull"])}
                for row in rows
            ]
    else:
        result = _execute(path, f"describe select * from {quoted}", limit=MAX_DB_ROWS)
        columns = result["rows"]
    resolved.log_tool_call("db_describe_table", {"database": database, "table": table})
    return {"database": str(path), "table": table, "columns": columns}


def db_preview_table(
    table: str,
    database: str | None = None,
    limit: int = 50,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Preview rows from a known table."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=50, maximum=MAX_DB_ROWS)
    path = _resolve_database(database, resolved)
    quoted = _require_known_table(path, table)
    resolved.log_tool_call(
        "db_preview_table",
        {"database": database, "table": table, "limit": safe_limit},
    )
    return _execute(path, f"select * from {quoted}", limit=safe_limit)


def _first_existing_table(path: Path, candidates: tuple[str, ...]) -> str | None:
    tables = set(_table_names(path))
    for table in candidates:
        if table in tables:
            return table
    return None


def db_get_symbol_bars(
    symbol: str,
    timeframe: str,
    start: str | None = None,
    end: str | None = None,
    limit: int = 500,
    database: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Return recent bars from a conventional bars table."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=500, maximum=MAX_DB_ROWS)
    path = _resolve_database(database, resolved)
    table = _first_existing_table(path, ("bars", "ohlcv", "prices", "market_bars"))
    if table is None:
        return {"database": str(path), "rows": [], "row_count": 0, "message": "no bars table found"}
    filters = ["symbol = ?", "timeframe = ?"]
    params: list[Any] = [symbol.upper(), timeframe]
    if start is not None:
        filters.append("timestamp >= ?")
        params.append(start)
    if end is not None:
        filters.append("timestamp <= ?")
        params.append(end)
    query = (
        f"select * from {_quote_identifier(table)} where {' and '.join(filters)} order by timestamp"
    )
    resolved.log_tool_call(
        "db_get_symbol_bars",
        {"database": database, "symbol": symbol, "timeframe": timeframe, "limit": safe_limit},
    )
    return _execute(path, query, tuple(params), limit=safe_limit)


def db_get_latest_catalysts(
    symbol: str | None = None,
    limit: int = 100,
    database: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Return latest catalyst-like rows when a conventional table exists."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=100, maximum=MAX_DB_ROWS)
    path = _resolve_database(database, resolved)
    table = _first_existing_table(path, ("catalysts", "events", "news"))
    if table is None:
        return {
            "database": str(path),
            "rows": [],
            "row_count": 0,
            "message": "no catalysts table found",
        }
    params: tuple[Any, ...] = ()
    where = ""
    if symbol is not None:
        where = " where symbol = ?"
        params = (symbol.upper(),)
    query = f"select * from {_quote_identifier(table)}{where} order by created_at desc"
    resolved.log_tool_call(
        "db_get_latest_catalysts",
        {"database": database, "symbol": symbol, "limit": safe_limit},
    )
    return _execute(path, query, params, limit=safe_limit)


def db_get_trade_attribution(
    run_id: str | None = None,
    symbol: str | None = None,
    limit: int = 500,
    database: str | None = None,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Return trade-attribution rows when a conventional table exists."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=500, maximum=MAX_DB_ROWS)
    path = _resolve_database(database, resolved)
    table = _first_existing_table(path, ("trade_attribution", "trades_attribution", "trades"))
    if table is None:
        return {
            "database": str(path),
            "rows": [],
            "row_count": 0,
            "message": "no attribution table found",
        }
    filters: list[str] = []
    params: list[Any] = []
    if run_id is not None:
        filters.append("run_id = ?")
        params.append(run_id)
    if symbol is not None:
        filters.append("symbol = ?")
        params.append(symbol.upper())
    where = f" where {' and '.join(filters)}" if filters else ""
    query = f"select * from {_quote_identifier(table)}{where}"
    resolved.log_tool_call(
        "db_get_trade_attribution",
        {"database": database, "run_id": run_id, "symbol": symbol, "limit": safe_limit},
    )
    return _execute(path, query, tuple(params), limit=safe_limit)


def db_select(
    sql: str,
    database: str | None = None,
    limit: int = 500,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Run a heavily restricted SELECT-only query."""

    resolved = _context(context)
    safe_limit = clamp_limit(limit, default=500, maximum=MAX_DB_ROWS)
    path = _resolve_database(database, resolved)
    safe_sql = _validate_select_sql(sql)
    resolved.log_tool_call(
        "db_select", {"database": database, "sql": safe_sql, "limit": safe_limit}
    )
    return _execute(path, safe_sql, limit=safe_limit)
