"""Public MCP tool metadata for the read-only Stocker connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

READ_ONLY_ANNOTATIONS: dict[str, bool] = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


@dataclass(frozen=True)
class ToolSpec:
    """Stable public metadata for one Stocker MCP tool."""

    name: str
    title: str
    description: str


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "search",
        "Search Stocker",
        "Search Stocker code, hypotheses, research reports, run summaries, and database schemas.",
    ),
    ToolSpec(
        "fetch",
        "Fetch Stocker Item",
        "Fetch one safe Stocker item by stocker:// identifier returned from search.",
    ),
    ToolSpec("workspace_doctor", "Workspace Doctor", "Inspect safe Stocker workspace diagnostics."),
    ToolSpec("get_repo_info", "Repository Info", "Read basic Stocker repository metadata."),
    ToolSpec("git_status", "Git Status", "Read short git status for the Stocker repository."),
    ToolSpec("git_log", "Git Log", "Read recent Stocker repository commit subjects."),
    ToolSpec(
        "git_current_commit",
        "Current Commit",
        "Read the current Stocker repository commit SHA.",
    ),
    ToolSpec("list_files", "List Code Files", "List safe Stocker repository files."),
    ToolSpec("search_code", "Search Code", "Search Stocker repository text files."),
    ToolSpec(
        "read_code_file",
        "Read Code File",
        "Read a bounded Stocker repository file, optionally by line range.",
    ),
    ToolSpec("git_diff", "Git Diff", "Read a bounded git diff from the Stocker repository."),
    ToolSpec(
        "list_recent_research_runs",
        "List Research Runs",
        "List recent Stocker universe research run summaries.",
    ),
    ToolSpec(
        "get_latest_universe_run",
        "Latest Universe Run",
        "Read the latest Stocker universe run summary.",
    ),
    ToolSpec(
        "read_universe_run",
        "Read Universe Run",
        "Read a bounded Stocker universe run with per-symbol summaries.",
    ),
    ToolSpec(
        "summarise_universe_run",
        "Summarise Universe Run",
        "Summarise classification counts and per-symbol results for one universe run.",
    ),
    ToolSpec(
        "compare_universe_runs",
        "Compare Universe Runs",
        "Compare two universe run summaries.",
    ),
    ToolSpec("list_symbol_reports", "List Symbol Reports", "List Stocker symbol reports."),
    ToolSpec("read_symbol_report", "Read Symbol Report", "Read a bounded Stocker symbol report."),
    ToolSpec(
        "find_candidate_symbols",
        "Find Candidate Symbols",
        "Find candidate symbols in one Stocker universe run.",
    ),
    ToolSpec(
        "find_interesting_symbols",
        "Find Interesting Symbols",
        "Find interesting symbols in one Stocker universe run.",
    ),
    ToolSpec(
        "filter_symbol_results",
        "Filter Symbol Results",
        "Filter symbol rows from a Stocker universe run.",
    ),
    ToolSpec(
        "db_list_databases",
        "List Databases",
        "List Stocker databases under STOCKER_HOME/db.",
    ),
    ToolSpec("db_list_tables", "List Tables", "List tables in a Stocker database."),
    ToolSpec("db_describe_table", "Describe Table", "Describe columns for a Stocker table."),
    ToolSpec("db_preview_table", "Preview Table", "Preview bounded rows from a Stocker table."),
    ToolSpec(
        "db_get_symbol_bars",
        "Get Symbol Bars",
        "Read bounded historical bar rows from DB tables or canonical Parquet partitions.",
    ),
    ToolSpec(
        "db_get_latest_catalysts",
        "Get Latest Catalysts",
        "Read bounded catalyst or news rows when present.",
    ),
    ToolSpec(
        "db_get_trade_attribution",
        "Get Trade Attribution",
        "Read bounded trade-attribution rows when present.",
    ),
    ToolSpec(
        "db_select",
        "Restricted Select",
        "Run a heavily restricted SELECT-only database query.",
    ),
    ToolSpec(
        "summarise_latest_research_state",
        "Summarise Latest Research State",
        "Summarise the latest Stocker research state without trading recommendations.",
    ),
    ToolSpec(
        "find_positive_rejected_symbols",
        "Find Positive Rejected Symbols",
        "Find rejected symbols with positive historical net return.",
    ),
    ToolSpec(
        "find_null_pass_benchmark_fail_symbols",
        "Find Null-Pass Benchmark-Fail Symbols",
        "Find symbols that passed null checks but failed benchmark checks.",
    ),
    ToolSpec(
        "find_benchmark_pass_rejected_symbols",
        "Find Benchmark-Pass Rejected Symbols",
        "Find rejected symbols that passed benchmark checks.",
    ),
    ToolSpec(
        "find_common_rejection_reasons",
        "Find Common Rejection Reasons",
        "Count common rejection reasons in a universe run.",
    ),
    ToolSpec(
        "get_symbol_bar_summary",
        "Get Symbol Bar Summary",
        "Summarise bounded historical bars for one symbol and timeframe.",
    ),
    ToolSpec(
        "get_symbol_recent_sessions",
        "Get Symbol Recent Sessions",
        "Summarise recent historical sessions for one symbol and timeframe.",
    ),
    ToolSpec(
        "get_trade_feature_buckets",
        "Get Trade Feature Buckets",
        "Summarise trade-attribution feature buckets when available.",
    ),
    ToolSpec(
        "compare_template_runs",
        "Compare Template Runs",
        "Compare two Stocker research/template universe runs.",
    ),
    ToolSpec(
        "suggest_research_questions",
        "Suggest Research Questions",
        "Suggest conservative evidence-based research questions from existing results.",
    ),
    ToolSpec(
        "export_diagnostics_zip",
        "Export Diagnostics Zip",
        "Create a redacted diagnostics zip under STOCKER_HOME/exports.",
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in TOOL_SPECS)
_TOOL_SPEC_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}


def get_tool_spec(name: str) -> ToolSpec:
    """Return public metadata for a registered tool name."""

    return _TOOL_SPEC_BY_NAME[name]


def tool_metadata() -> list[dict[str, Any]]:
    """Return ChatGPT-friendly tool metadata without implementation details."""

    return [
        {
            "name": spec.name,
            "title": spec.title,
            "description": spec.description,
            "annotations": dict(READ_ONLY_ANNOTATIONS),
        }
        for spec in TOOL_SPECS
    ]
