import json
import zipfile
from pathlib import Path

from stocker_mcp.security import StockerMCPContext
from stocker_mcp.server import build_server, tool_names
from stocker_mcp.tools import diagnostics


def test_diagnostics_zip_excludes_env_and_database_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    repo.mkdir()
    (repo / "configs").mkdir()
    (repo / "configs" / "research.example.yaml").write_text(
        "token: should-redact\n", encoding="utf-8"
    )
    (repo / ".env").write_text("API_KEY=leak", encoding="utf-8")
    report_dir = home / "data" / "reports" / "research" / "universe"
    report_dir.mkdir(parents=True)
    (report_dir / "run.json").write_text(
        json.dumps({"run_id": "run", "classification_counts": {}, "symbol_results": []}),
        encoding="utf-8",
    )
    db_dir = home / "db"
    db_dir.mkdir()
    (db_dir / "stocker.sqlite").write_text("not a real db", encoding="utf-8")
    context = StockerMCPContext(repo_root=repo, stocker_home=home)

    result = diagnostics.export_diagnostics_zip(context=context)

    zip_path = Path(result["output_path"])
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "workspace_doctor.json" in names
        assert "reports/run.json" in names
        assert ".env" not in names
        assert "db/stocker.sqlite" not in names
        config_text = archive.read("configs/research.example.yaml").decode()
    assert "should-redact" not in config_text


def test_mcp_server_exports_expected_tool_names() -> None:
    names = set(tool_names())

    assert "read_code_file" in names
    assert "summarise_universe_run" in names
    assert "db_select" in names
    assert "export_diagnostics_zip" in names


def test_mcp_server_builds_with_sdk() -> None:
    server = build_server()

    assert type(server).__name__ == "FastMCP"
