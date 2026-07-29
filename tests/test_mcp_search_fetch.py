import json
from pathlib import Path

import pytest

from stocker_mcp.security import SecurityError, StockerMCPContext
from stocker_mcp.tools import discovery


def _context(tmp_path: Path) -> StockerMCPContext:
    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    repo.mkdir()
    home.mkdir()
    (repo / "packages").mkdir()
    (repo / "pyproject.toml").write_text('name = "stocker"\n', encoding="utf-8")
    (repo / "example.py").write_text("def vwap_signal():\n    return 'VWAP'\n", encoding="utf-8")
    (repo / ".env").write_text("API_KEY=leak", encoding="utf-8")
    hypotheses = repo / "research" / "hypotheses" / "examples"
    hypotheses.mkdir(parents=True)
    (hypotheses / "vwap.yaml").write_text(
        "id: vwap-test\nname: VWAP mean reversion\nexpected_edge_reason: VWAP reclaim\n",
        encoding="utf-8",
    )
    report_dir = home / "data" / "reports" / "research" / "universe"
    report_dir.mkdir(parents=True)
    (report_dir / "run-vwap.json").write_text(
        json.dumps(
            {
                "run_id": "run-vwap",
                "hypothesis_id": "vwap-test",
                "classification_counts": {"rejected_no_edge": 1},
                "symbol_results": [
                    {
                        "symbol": "NVDA",
                        "status": "completed",
                        "classification": "rejected_no_edge",
                        "classification_reasons": ["failed_benchmark"],
                        "net_return": 0.02,
                        "trade_count": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "run-vwap.md").write_text("# VWAP run\nfailed benchmark\n", encoding="utf-8")
    return StockerMCPContext(repo_root=repo, stocker_home=home)


def test_search_returns_chatgpt_result_shape(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = discovery.search("VWAP", context=context)

    assert result["results"]
    assert all({"id", "title", "url"}.issubset(item) for item in result["results"])
    assert all(item["url"] == "" for item in result["results"])
    assert any(item["id"].startswith("stocker://code/") for item in result["results"])
    assert any(item["id"].startswith("stocker://hypotheses/") for item in result["results"])
    assert any(item["id"].startswith("stocker://reports/") for item in result["results"])


def test_search_prioritizes_matching_nested_report_files(tmp_path: Path) -> None:
    context = _context(tmp_path)
    report_dir = (
        context.stocker_home
        / "data"
        / "reports"
        / "research"
        / "role_aware_event_cutter_v0"
        / "role_aware_event_cutter_v0_20260630T225926Z"
    )
    report_dir.mkdir(parents=True)
    (report_dir / "summary.md").write_text("role-aware summary", encoding="utf-8")

    result = discovery.search("role_aware_event_cutter_v0", limit=1, context=context)

    assert result["results"][0]["id"].startswith(
        "stocker://reports/home/role_aware_event_cutter_v0"
    )


def test_fetch_returns_safe_text_and_metadata(tmp_path: Path) -> None:
    context = _context(tmp_path)
    search_result = discovery.search("VWAP", context=context)
    code_id = next(item["id"] for item in search_result["results"] if item["source"] == "code")

    fetched = discovery.fetch(code_id, context=context)

    assert fetched["id"] == code_id
    assert fetched["url"] == ""
    assert "VWAP" in fetched["text"]
    assert fetched["metadata"]["source"] == "code"
    assert "API_KEY" not in fetched["text"]


def test_fetch_rejects_invalid_or_blocked_ids(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(SecurityError):
        discovery.fetch("file:///etc/passwd", context=context)
    with pytest.raises(SecurityError):
        discovery.fetch("stocker://code/../.env", context=context)
    with pytest.raises(SecurityError):
        discovery.fetch("stocker://code/.env", context=context)
