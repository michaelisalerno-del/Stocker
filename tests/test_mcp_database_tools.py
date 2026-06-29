import sqlite3
from pathlib import Path

import pytest

from stocker_mcp.security import SecurityError, StockerMCPContext
from stocker_mcp.tools import database


def _context_with_sqlite(tmp_path: Path) -> StockerMCPContext:
    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    db_dir = home / "db"
    repo.mkdir()
    db_dir.mkdir(parents=True)
    db_path = db_dir / "stocker.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "create table bars ("
            "symbol text, timeframe text, timestamp text, close real, api_key text"
            ")"
        )
        connection.execute(
            "insert into bars values ('AAPL', '1d', '2026-01-01T00:00:00Z', 101.5, 'secret')"
        )
        connection.execute("create table catalysts (symbol text, created_at text, title text)")
        connection.execute(
            "insert into catalysts values ('AAPL', '2026-01-02T00:00:00Z', 'earnings')"
        )
    return StockerMCPContext(repo_root=repo, stocker_home=home)


def _context_with_duckdb(tmp_path: Path) -> StockerMCPContext:
    import duckdb

    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    db_dir = home / "db"
    repo.mkdir()
    db_dir.mkdir(parents=True)
    db_path = db_dir / "research.duckdb"
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("create table bars (symbol varchar, timeframe varchar, close double)")
        connection.execute("insert into bars values ('AAPL', '1d', 101.5)")
    return StockerMCPContext(repo_root=repo, stocker_home=home)


def test_database_table_listing_and_preview_are_read_only(tmp_path: Path) -> None:
    context = _context_with_sqlite(tmp_path)

    listed = database.db_list_tables(context=context)
    preview = database.db_preview_table("bars", limit=10, context=context)

    assert "bars" in listed["tables"]
    assert preview["rows"][0]["symbol"] == "AAPL"
    assert preview["rows"][0]["api_key"] == "[REDACTED]"


def test_named_symbol_bars_query_forces_limit(tmp_path: Path) -> None:
    context = _context_with_sqlite(tmp_path)

    result = database.db_get_symbol_bars("AAPL", "1d", limit=1, context=context)

    assert result["row_count"] == 1
    assert result["rows"][0]["close"] == 101.5


def test_arbitrary_sql_rejects_write_statements(tmp_path: Path) -> None:
    context = _context_with_sqlite(tmp_path)

    with pytest.raises(SecurityError):
        database.db_select("delete from bars", context=context)


def test_database_tools_can_be_disabled(tmp_path: Path) -> None:
    context = _context_with_sqlite(tmp_path)
    disabled = StockerMCPContext(
        repo_root=context.repo_root,
        stocker_home=context.stocker_home,
        db_enabled=False,
    )

    with pytest.raises(SecurityError):
        database.db_list_tables(context=disabled)


def test_arbitrary_sql_select_is_limited_and_redacted(tmp_path: Path) -> None:
    context = _context_with_sqlite(tmp_path)

    result = database.db_select("select symbol, api_key from bars", limit=5, context=context)

    assert result["row_count"] == 1
    assert result["rows"] == [{"symbol": "AAPL", "api_key": "[REDACTED]"}]


def test_duckdb_table_listing_description_and_preview(tmp_path: Path) -> None:
    context = _context_with_duckdb(tmp_path)

    listed = database.db_list_tables(context=context)
    described = database.db_describe_table("bars", context=context)
    preview = database.db_preview_table("bars", context=context)

    assert listed["database_type"] == "duckdb"
    assert listed["tables"] == ["bars"]
    assert any(column["column_name"] == "symbol" for column in described["columns"])
    assert preview["rows"] == [{"symbol": "AAPL", "timeframe": "1d", "close": 101.5}]
