import json
from pathlib import Path

from stocker_mcp.security import StockerMCPContext
from stocker_mcp.tools import reports


def _write_universe_run(root: Path, run_id: str, symbols: list[dict[str, object]]) -> Path:
    universe_dir = root / "data" / "reports" / "research" / "universe"
    universe_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "hypothesis_id": "hypothesis-a",
        "universe_id": "test-universe",
        "symbol_count": len(symbols),
        "completed_count": sum(1 for item in symbols if item["status"] == "completed"),
        "failed_count": sum(1 for item in symbols if item["status"] == "failed"),
        "classification_counts": {
            "candidate_intraday_test": 1,
            "interesting_needs_more_tests": 1,
            "rejected_no_edge": 1,
        },
        "candidate_count": 1,
        "rejected_count": 1,
        "top_rejection_reasons": {"failed_benchmark": 2},
        "symbol_results": symbols,
    }
    path = universe_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    (universe_dir / f"{run_id}.md").write_text("# report\n", encoding="utf-8")
    return path


def _context(tmp_path: Path) -> StockerMCPContext:
    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    repo.mkdir()
    home.mkdir()
    return StockerMCPContext(repo_root=repo, stocker_home=home)


def test_list_and_summarise_universe_reports(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_universe_run(
        context.stocker_home,
        "run-1",
        [
            {
                "symbol": "AAPL",
                "status": "completed",
                "classification": "candidate_intraday_test",
                "net_return": 0.12,
                "trade_count": 42,
                "benchmark_pass": True,
                "null_pass": True,
            },
            {
                "symbol": "MSFT",
                "status": "completed",
                "classification": "rejected_no_edge",
                "classification_reasons": ["failed_benchmark"],
                "net_return": -0.02,
                "trade_count": 12,
                "benchmark_pass": False,
                "null_pass": False,
            },
            {"symbol": "TSLA", "status": "failed", "error_message": "bad data"},
        ],
    )

    listed = reports.list_recent_research_runs(limit=10, context=context)
    summary = reports.summarise_universe_run("run-1", context=context)

    assert listed["runs"][0]["run_id"] == "run-1"
    assert summary["candidate_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["classification_counts"]["candidate_intraday_test"] == 1
    assert summary["per_symbol"][0]["symbol"] == "AAPL"


def test_filter_symbol_results(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _write_universe_run(
        context.stocker_home,
        "run-2",
        [
            {
                "symbol": "AAPL",
                "status": "completed",
                "classification": "candidate_intraday_test",
                "net_return": 0.12,
                "trade_count": 42,
                "benchmark_pass": True,
                "null_pass": True,
            },
            {
                "symbol": "MSFT",
                "status": "completed",
                "classification": "rejected_no_edge",
                "net_return": -0.02,
                "trade_count": 12,
                "benchmark_pass": False,
                "null_pass": False,
            },
        ],
    )

    result = reports.filter_symbol_results(
        "run-2",
        classification="candidate_intraday_test",
        null_pass=True,
        benchmark_pass=True,
        min_net_return=0.1,
        min_trade_count=20,
        context=context,
    )

    assert [item["symbol"] for item in result["symbols"]] == ["AAPL"]
