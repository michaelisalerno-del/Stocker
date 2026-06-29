import json
import sqlite3
from pathlib import Path

from stocker_mcp.security import StockerMCPContext
from stocker_mcp.tools import research


def _context(tmp_path: Path) -> StockerMCPContext:
    repo = tmp_path / "repo"
    home = tmp_path / "StockerLocal"
    repo.mkdir()
    db_dir = home / "db"
    report_dir = home / "data" / "reports" / "research" / "universe"
    db_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    payload = {
        "run_id": "run-a",
        "hypothesis_id": "template-a",
        "classification_counts": {
            "rejected_no_edge": 2,
            "candidate_intraday_test": 1,
        },
        "classification_reason_counts": {
            "failed_benchmark": 2,
            "failed_null_timing": 1,
        },
        "symbol_results": [
            {
                "symbol": "AAPL",
                "status": "completed",
                "classification": "rejected_no_edge",
                "classification_reasons": ["failed_benchmark"],
                "net_return": 0.04,
                "trade_count": 24,
                "benchmark_pass": False,
                "null_pass": True,
            },
            {
                "symbol": "MSFT",
                "status": "completed",
                "classification": "rejected_no_edge",
                "classification_reasons": ["failed_null_timing"],
                "net_return": 0.01,
                "trade_count": 9,
                "benchmark_pass": True,
                "null_pass": False,
            },
        ],
    }
    (report_dir / "run-a.json").write_text(json.dumps(payload), encoding="utf-8")
    db_path = db_dir / "stocker.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "create table bars ("
            "symbol text, timeframe text, timestamp text, open real, high real, "
            "low real, close real, volume real"
            ")"
        )
        connection.executemany(
            "insert into bars values (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("AAPL", "5m", "2026-01-01T14:30:00Z", 10, 11, 9, 10.5, 100),
                ("AAPL", "5m", "2026-01-01T14:35:00Z", 10.5, 12, 10, 11.5, 200),
            ],
        )
    return StockerMCPContext(repo_root=repo, stocker_home=home)


def test_discovery_workflow_tools_use_report_evidence(tmp_path: Path) -> None:
    context = _context(tmp_path)

    positive = research.find_positive_rejected_symbols(context=context)
    null_pass_benchmark_fail = research.find_null_pass_benchmark_fail_symbols(context=context)
    benchmark_pass_rejected = research.find_benchmark_pass_rejected_symbols(context=context)
    reasons = research.find_common_rejection_reasons(context=context)
    questions = research.suggest_research_questions(context=context)

    assert [item["symbol"] for item in positive["symbols"]] == ["AAPL", "MSFT"]
    assert [item["symbol"] for item in null_pass_benchmark_fail["symbols"]] == ["AAPL"]
    assert [item["symbol"] for item in benchmark_pass_rejected["symbols"]] == ["MSFT"]
    assert reasons["reasons"][0]["reason"] == "failed_benchmark"
    assert questions["questions"]
    assert all("live" not in item["question"].lower() for item in questions["questions"])


def test_bar_summary_reads_historical_db_only(tmp_path: Path) -> None:
    context = _context(tmp_path)

    summary = research.get_symbol_bar_summary("AAPL", timeframe="5m", context=context)
    sessions = research.get_symbol_recent_sessions("AAPL", timeframe="5m", context=context)

    assert summary["symbol"] == "AAPL"
    assert summary["row_count"] == 2
    assert summary["latest_timestamp"] == "2026-01-01T14:35:00Z"
    assert sessions["sessions"][0]["date"] == "2026-01-01"
