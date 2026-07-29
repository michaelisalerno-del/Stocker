"""Safe diagnostics export for sharing Stocker context with AI tools."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from stocker_mcp.security import (
    SecurityError,
    StockerMCPContext,
    default_context,
    is_blocked_path,
    redact_secrets,
)
from stocker_mcp.tools import code, database, reports, workspace


def _context(context: StockerMCPContext | None) -> StockerMCPContext:
    return context or default_context()


def _json_bytes(payload: Any) -> bytes:
    return redact_secrets(json.dumps(payload, indent=2, sort_keys=True, default=str)).encode(
        "utf-8"
    )


def _safe_output_path(output_path: str | None, context: StockerMCPContext) -> Path:
    context.export_root.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        return context.export_root / "stocker_diagnostics.zip"
    path = context.resolve_under_root(context.export_root, output_path)
    if path.suffix.lower() != ".zip":
        raise SecurityError(
            "diagnostics output_path must be a .zip file under STOCKER_HOME/exports"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(archive: zipfile.ZipFile, name: str, payload: Any) -> None:
    archive.writestr(name, _json_bytes(payload))


def _include_latest_reports(archive: zipfile.ZipFile, context: StockerMCPContext) -> None:
    recent = reports.list_recent_research_runs(limit=3, context=context)
    _write_json(archive, "latest_universe_runs.json", recent)
    for item in recent["runs"]:
        json_path = item.get("json_path")
        markdown_path = item.get("markdown_path")
        if json_path:
            path = Path(str(json_path))
            root = _report_root_for(path, context)
            content, _, _, _ = context.read_text_file(path, root=root)
            archive.writestr(f"reports/{path.name}", content.encode("utf-8"))
        if markdown_path:
            path = Path(str(markdown_path))
            if not path.exists() or is_blocked_path(path):
                continue
            root = _report_root_for(path, context)
            content, _, _, _ = context.read_text_file(path, root=root)
            archive.writestr(f"reports/{path.name}", content.encode("utf-8"))


def _report_root_for(path: Path, context: StockerMCPContext) -> Path:
    for root in context.report_roots():
        try:
            path.resolve().relative_to(root.resolve())
            return root
        except ValueError:
            continue
    raise SecurityError(f"report path outside report roots: {path}")


def _include_db_schema(archive: zipfile.ZipFile, context: StockerMCPContext) -> None:
    databases = database.db_list_databases(context=context)
    schemas: list[dict[str, Any]] = []
    for item in databases["databases"]:
        db_name = str(item["name"])
        try:
            tables = database.db_list_tables(database=db_name, context=context)
        except Exception:
            continue
        table_schemas = []
        for table in tables["tables"]:
            try:
                table_schemas.append(
                    database.db_describe_table(table, database=db_name, context=context)
                )
            except Exception:
                continue
        schemas.append({"database": item, "tables": table_schemas})
    _write_json(archive, "db_schema.json", {"databases": schemas})


def _include_config_examples(archive: zipfile.ZipFile, context: StockerMCPContext) -> None:
    config_root = context.repo_root / "configs"
    if not config_root.exists():
        return
    for path in sorted(config_root.glob("*.yaml")):
        if is_blocked_path(path):
            continue
        content, _, _, _ = context.read_text_file(path, root=context.repo_root)
        archive.writestr(f"configs/{path.name}", content.encode("utf-8"))


def export_diagnostics_zip(
    output_path: str | None = None,
    include_code_summary: bool = True,
    include_reports: bool = True,
    include_db_schema: bool = True,
    context: StockerMCPContext | None = None,
) -> dict[str, Any]:
    """Create a redacted diagnostics zip under STOCKER_HOME/exports."""

    resolved = _context(context)
    target = _safe_output_path(output_path, resolved)
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_json(archive, "workspace_doctor.json", workspace.workspace_doctor(context=resolved))
        _write_json(archive, "git_status.json", code.git_status(context=resolved))
        _write_json(archive, "git_log_5.json", code.git_log(limit=5, context=resolved))
        if include_code_summary:
            _write_json(archive, "repo_info.json", code.get_repo_info(context=resolved))
            _write_json(
                archive,
                "repo_files.json",
                code.list_files(path="", glob=None, limit=200, context=resolved),
            )
        if include_reports:
            _include_latest_reports(archive, resolved)
        if include_db_schema:
            _include_db_schema(archive, resolved)
        _include_config_examples(archive, resolved)
    resolved.log_tool_call(
        "export_diagnostics_zip",
        {
            "output_path": str(target),
            "include_code_summary": include_code_summary,
            "include_reports": include_reports,
            "include_db_schema": include_db_schema,
        },
    )
    return {"output_path": str(target), "size_bytes": target.stat().st_size}
