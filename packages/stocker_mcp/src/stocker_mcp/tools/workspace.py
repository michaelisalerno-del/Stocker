"""Workspace diagnostics for the Stocker MCP server."""

from __future__ import annotations

from typing import Any

from stocker_mcp.security import StockerMCPContext, default_context


def _context(context: StockerMCPContext | None) -> StockerMCPContext:
    return context or default_context()


def workspace_doctor(context: StockerMCPContext | None = None) -> dict[str, Any]:
    """Return safe local workspace diagnostics."""

    resolved = _context(context)
    report_roots = resolved.report_roots()
    doctor = {
        "repo_root": str(resolved.repo_root),
        "repo_root_exists": resolved.repo_root.exists(),
        "stocker_home": str(resolved.stocker_home),
        "stocker_home_exists": resolved.stocker_home.exists(),
        "db_root": str(resolved.db_root),
        "db_root_exists": resolved.db_root.exists(),
        "db_enabled": resolved.db_enabled,
        "export_root": str(resolved.export_root),
        "report_roots": [str(root) for root in report_roots],
        "repo_report_root_exists": (resolved.repo_root / "data" / "reports" / "research").exists(),
        "home_report_root_exists": (
            resolved.stocker_home / "data" / "reports" / "research"
        ).exists(),
        "read_only": True,
        "blocked_actions": [
            "broker access",
            "IBKR access",
            "order placement",
            "live or paper execution",
            "arbitrary shell",
            "arbitrary file writes",
            "secret file reads",
            "deletion",
            "vendor data fetching",
        ],
    }
    resolved.log_tool_call("workspace_doctor")
    return doctor
