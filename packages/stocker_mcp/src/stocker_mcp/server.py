"""Stdio MCP server for safe local Stocker inspection."""

from __future__ import annotations

import argparse
import json
from typing import Any

from stocker_mcp import __version__
from stocker_mcp.schemas import TOOL_NAMES
from stocker_mcp.security import SecurityError, redact_secrets
from stocker_mcp.tools import code, database, diagnostics, reports, workspace


def tool_names() -> tuple[str, ...]:
    """Return the tools registered by the Stocker MCP server."""

    return TOOL_NAMES


def _json(payload: Any) -> str:
    return redact_secrets(json.dumps(payload, indent=2, sort_keys=True, default=str))


def build_server() -> Any:
    """Build a FastMCP server with only read-only Stocker tools."""

    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("Stocker")

    @mcp.tool()
    def get_repo_info() -> dict[str, Any]:
        return code.get_repo_info()

    @mcp.tool()
    def git_status() -> dict[str, Any]:
        return code.git_status()

    @mcp.tool()
    def git_log(limit: int = 10) -> dict[str, Any]:
        return code.git_log(limit=limit)

    @mcp.tool()
    def git_current_commit() -> dict[str, Any]:
        return code.git_current_commit()

    @mcp.tool()
    def list_files(path: str = "", glob: str | None = None, limit: int = 200) -> dict[str, Any]:
        return code.list_files(path=path, glob=glob, limit=limit)

    @mcp.tool()
    def search_code(query: str, path_glob: str | None = None, limit: int = 100) -> dict[str, Any]:
        return code.search_code(query=query, path_glob=path_glob, limit=limit)

    @mcp.tool()
    def read_code_file(
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        return code.read_code_file(path=path, start_line=start_line, end_line=end_line)

    @mcp.tool()
    def git_diff(
        ref: str | None = None,
        path: str | None = None,
        max_lines: int = 1_000,
    ) -> dict[str, Any]:
        return code.git_diff(ref=ref, path=path, max_lines=max_lines)

    @mcp.tool()
    def workspace_doctor() -> dict[str, Any]:
        return workspace.workspace_doctor()

    @mcp.tool()
    def list_recent_research_runs(limit: int = 20) -> dict[str, Any]:
        return reports.list_recent_research_runs(limit=limit)

    @mcp.tool()
    def get_latest_universe_run(hypothesis_id: str | None = None) -> dict[str, Any]:
        return reports.get_latest_universe_run(hypothesis_id=hypothesis_id)

    @mcp.tool()
    def read_universe_run(run_id_or_path: str) -> dict[str, Any]:
        return reports.read_universe_run(run_id_or_path=run_id_or_path)

    @mcp.tool()
    def summarise_universe_run(run_id_or_path: str) -> dict[str, Any]:
        return reports.summarise_universe_run(run_id_or_path=run_id_or_path)

    @mcp.tool()
    def list_symbol_reports(
        run_id_or_path: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return reports.list_symbol_reports(
            run_id_or_path=run_id_or_path, symbol=symbol, limit=limit
        )

    @mcp.tool()
    def read_symbol_report(
        symbol: str | None = None,
        path: str | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        return reports.read_symbol_report(symbol=symbol, path=path, experiment_id=experiment_id)

    @mcp.tool()
    def compare_universe_runs(run_a: str, run_b: str) -> dict[str, Any]:
        return reports.compare_universe_runs(run_a=run_a, run_b=run_b)

    @mcp.tool()
    def find_candidate_symbols(run_id_or_path: str) -> dict[str, Any]:
        return reports.find_candidate_symbols(run_id_or_path=run_id_or_path)

    @mcp.tool()
    def find_interesting_symbols(run_id_or_path: str) -> dict[str, Any]:
        return reports.find_interesting_symbols(run_id_or_path=run_id_or_path)

    @mcp.tool()
    def filter_symbol_results(
        run_id_or_path: str,
        classification: str | None = None,
        null_pass: bool | None = None,
        benchmark_pass: bool | None = None,
        min_net_return: float | None = None,
        min_trade_count: int | None = None,
    ) -> dict[str, Any]:
        return reports.filter_symbol_results(
            run_id_or_path=run_id_or_path,
            classification=classification,
            null_pass=null_pass,
            benchmark_pass=benchmark_pass,
            min_net_return=min_net_return,
            min_trade_count=min_trade_count,
        )

    @mcp.tool()
    def db_list_databases() -> dict[str, Any]:
        return database.db_list_databases()

    @mcp.tool()
    def db_list_tables(database: str | None = None) -> dict[str, Any]:
        return database.db_list_tables(database=database)

    @mcp.tool()
    def db_describe_table(table: str, database: str | None = None) -> dict[str, Any]:
        return database.db_describe_table(table=table, database=database)

    @mcp.tool()
    def db_preview_table(
        table: str,
        database: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return database.db_preview_table(table=table, database=database, limit=limit)

    @mcp.tool()
    def db_get_symbol_bars(
        symbol: str,
        timeframe: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
        database: str | None = None,
    ) -> dict[str, Any]:
        return database.db_get_symbol_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit,
            database=database,
        )

    @mcp.tool()
    def db_get_latest_catalysts(
        symbol: str | None = None,
        limit: int = 100,
        database: str | None = None,
    ) -> dict[str, Any]:
        return database.db_get_latest_catalysts(symbol=symbol, limit=limit, database=database)

    @mcp.tool()
    def db_get_trade_attribution(
        run_id: str | None = None,
        symbol: str | None = None,
        limit: int = 500,
        database: str | None = None,
    ) -> dict[str, Any]:
        return database.db_get_trade_attribution(
            run_id=run_id,
            symbol=symbol,
            limit=limit,
            database=database,
        )

    @mcp.tool()
    def db_select(sql: str, database: str | None = None, limit: int = 500) -> dict[str, Any]:
        return database.db_select(sql=sql, database=database, limit=limit)

    @mcp.tool()
    def export_diagnostics_zip(
        output_path: str | None = None,
        include_code_summary: bool = True,
        include_reports: bool = True,
        include_db_schema: bool = True,
    ) -> dict[str, Any]:
        return diagnostics.export_diagnostics_zip(
            output_path=output_path,
            include_code_summary=include_code_summary,
            include_reports=include_reports,
            include_db_schema=include_db_schema,
        )

    @mcp.resource("stocker://workspace/doctor")
    def workspace_doctor_resource() -> str:
        return _json(workspace.workspace_doctor())

    @mcp.resource("stocker://reports/latest")
    def reports_latest_resource() -> str:
        try:
            return _json(reports.get_latest_universe_run())
        except SecurityError as exc:
            return _json({"error": str(exc)})

    @mcp.resource("stocker://repo/status")
    def repo_status_resource() -> str:
        return _json(code.git_status())

    @mcp.prompt()
    def summarise_latest_stocker_scan() -> str:
        return "Summarise the latest Stocker universe scan using get_latest_universe_run."

    @mcp.prompt()
    def compare_two_stocker_universe_runs() -> str:
        return "Compare two Stocker universe runs using compare_universe_runs."

    @mcp.prompt()
    def investigate_symbol_gate_failure() -> str:
        return "Investigate why a symbol failed Stocker gates using read_symbol_report."

    return mcp


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stocker-mcp",
        description="Run the read-only local Stocker MCP server over stdio.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="MCP transport. Only stdio is enabled.",
    )
    parser.add_argument("--version", action="version", version=f"stocker-mcp {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="Print safe workspace diagnostics and exit.")
    subparsers.add_parser("tools", help="Print registered MCP tool names and exit.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Console script entry point."""

    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(_json(workspace.workspace_doctor()))
        return
    if args.command == "tools":
        print(_json({"tools": list(tool_names())}))
        return
    build_server().run(transport=args.transport)


if __name__ == "__main__":
    main()
