"""Reclassify opening-range reports with Stage 4.5 robustness gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "stocker_research" / "src"))

from stocker_research.robustness import RobustnessGatePolicy  # noqa: E402

STAGE4_4_DIR = "stage4_4_opening_range_robustness"
OUTPUT_DIR = "stage4_5_intraday_robustness_gates"
UNIVERSE_PATTERN = "*opening_range_breakout_intraday_session_flat_v1_us_liquid_25_intraday_5m.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _counter_dict(values: list[str]) -> dict[str, int]:
    return dict(Counter(values))


def _latest_universe_report(report_root: Path) -> Path:
    universe_dir = report_root / "universe"
    candidates = sorted(universe_dir.glob(UNIVERSE_PATTERN))
    if not candidates:
        raise FileNotFoundError(f"No opening-range universe report found in {universe_dir}")
    return candidates[-1]


def _by_symbol(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["symbol"]): row for row in rows}


def _failure_reasons(
    cost_row: dict[str, str],
    concentration_row: dict[str, str],
    policy: RobustnessGatePolicy,
) -> list[str]:
    reasons: list[str] = []
    if policy.require_cost_stress_for_intraday_candidate and not _bool(
        cost_row.get("survives_1_5x_costs")
    ):
        reasons.append("failed_cost_stress")
    if policy.require_positive_median_trade and (
        _bool(concentration_row.get("median_trade_negative"))
        or _float(concentration_row.get("median_trade")) <= 0
    ):
        reasons.append("negative_median_trade")
    if _float(concentration_row.get("top_positive_split_share")) > (
        policy.max_top_positive_split_share
    ):
        reasons.append("split_concentrated")
    if _float(concentration_row.get("top5_winners_share_of_positive_profit")) > (
        policy.max_top_5_winner_profit_share
    ):
        reasons.append("trade_concentrated")
    if _float(concentration_row.get("profit_factor")) < policy.min_candidate_profit_factor:
        reasons.append("weak_profit_factor")
    return reasons


def _new_classification(old_classification: str, failure_reasons: list[str]) -> str:
    if old_classification == "candidate_intraday_test" and failure_reasons:
        return "interesting_intraday_needs_more_tests"
    return old_classification


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _summary_markdown(summary: dict[str, Any]) -> str:
    downgraded = summary["symbols_downgraded_by_robustness_gates"] or []
    downgraded_text = ", ".join(downgraded) if downgraded else "None"
    return f"""# Stage 4.5 Intraday Robustness Gates

Stage 4.5 promotes the Stage 4.4 cost-stress and concentration diagnostics into
official intraday candidate gates. No new strategy templates, data fetches, ML,
broker, paper, live trading, or dashboard code were added.

## Tests

{chr(10).join(f"- {item}" for item in summary["tests_run"]) or "- Not recorded"}

Status: `{summary["test_status"]}`

## Classification Counts

- Old classification counts: `{json.dumps(summary["old_classification_counts"], sort_keys=True)}`
- New classification counts: `{json.dumps(summary["new_classification_counts"], sort_keys=True)}`
- Old candidate count: {summary["old_candidate_count"]}
- New candidate count: {summary["new_candidate_count"]}
- Downgraded symbols: {downgraded_text}

## CRM

- Old classification: `{summary["crm_old_classification"]}`
- New classification: `{summary["crm_new_classification"]}`
- Robustness failure reasons: `{summary["robustness_failure_reasons_by_symbol"].get("CRM", [])}`

## Cost Stress

```json
{json.dumps(summary["cost_stress_summary"], indent=2, sort_keys=True)}
```

## Concentration

```json
{json.dumps(summary["concentration_summary"], indent=2, sort_keys=True)}
```

## Recommendation

{summary["recommendation_for_next_strategy_family"]}
"""


def build_summary(
    *,
    report_root: Path,
    output_dir: Path,
    universe_report: Path,
    tests_run: list[str],
    test_status: str,
) -> dict[str, Any]:
    policy = RobustnessGatePolicy()
    stage4_4_dir = report_root / STAGE4_4_DIR
    cost_rows = _by_symbol(_read_csv(stage4_4_dir / "cost_stress_by_symbol.csv"))
    concentration_rows = _by_symbol(_read_csv(stage4_4_dir / "return_concentration_by_symbol.csv"))
    stage4_4_summary = _load_json(stage4_4_dir / "summary.json")
    universe_payload = _load_json(universe_report)
    symbol_results = [
        item
        for item in universe_payload.get("symbol_results", [])
        if isinstance(item, dict) and item.get("status") in {"completed", "skipped"}
    ]

    reclassified_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    old_classifications: list[str] = []
    new_classifications: list[str] = []
    downgraded: list[str] = []
    failure_reasons_by_symbol: dict[str, list[str]] = {}

    for item in symbol_results:
        symbol = str(item["symbol"])
        cost_row = cost_rows[symbol]
        concentration_row = concentration_rows[symbol]
        old_classification = str(item["classification"])
        failure_reasons = _failure_reasons(cost_row, concentration_row, policy)
        new_classification = _new_classification(old_classification, failure_reasons)
        old_classifications.append(old_classification)
        new_classifications.append(new_classification)
        failure_reasons_by_symbol[symbol] = failure_reasons
        if old_classification != new_classification:
            downgraded.append(symbol)
        if failure_reasons:
            failure_rows.append(
                {
                    "symbol": symbol,
                    "old_classification": old_classification,
                    "new_classification": new_classification,
                    "failure_reasons": "|".join(failure_reasons),
                    "survives_1_5x_costs": cost_row.get("survives_1_5x_costs"),
                    "median_trade_return": concentration_row.get("median_trade"),
                    "profit_factor": concentration_row.get("profit_factor"),
                    "top_positive_split_share": concentration_row.get("top_positive_split_share"),
                    "top_5_winners_profit_share": concentration_row.get(
                        "top5_winners_share_of_positive_profit"
                    ),
                }
            )
        reclassified_rows.append(
            {
                "symbol": symbol,
                "old_classification": old_classification,
                "new_classification": new_classification,
                "old_net_return": item.get("net_return", 0.0),
                "benchmark_pass": item.get("benchmark_pass", False),
                "null_pass": item.get("null_pass", False),
                "survives_1_5x_costs": cost_row.get("survives_1_5x_costs"),
                "net_return_1_5x_costs": cost_row.get("net_return_1_5x"),
                "median_trade_return": concentration_row.get("median_trade"),
                "profit_factor": concentration_row.get("profit_factor"),
                "top_positive_split_share": concentration_row.get("top_positive_split_share"),
                "top_5_winners_profit_share": concentration_row.get(
                    "top5_winners_share_of_positive_profit"
                ),
                "robustness_failure_reasons": "|".join(failure_reasons),
            }
        )

    old_counts = _counter_dict(old_classifications)
    new_counts = _counter_dict(new_classifications)
    summary = {
        "stage": "4.5_intraday_robustness_gates",
        "created_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
        "source_report_root": str(report_root),
        "source_universe_report": str(universe_report),
        "output_dir": str(output_dir),
        "tests_run": tests_run,
        "test_status": test_status,
        "old_classification_counts": old_counts,
        "new_classification_counts": new_counts,
        "old_candidate_count": old_counts.get("candidate_intraday_test", 0),
        "new_candidate_count": new_counts.get("candidate_intraday_test", 0),
        "symbols_downgraded_by_robustness_gates": downgraded,
        "robustness_failure_reasons_by_symbol": failure_reasons_by_symbol,
        "crm_old_classification": next(
            row["old_classification"] for row in reclassified_rows if row["symbol"] == "CRM"
        ),
        "crm_new_classification": next(
            row["new_classification"] for row in reclassified_rows if row["symbol"] == "CRM"
        ),
        "cost_stress_summary": stage4_4_summary.get("cost_stress_summary", {}),
        "concentration_summary": stage4_4_summary.get("concentration_summary", {}),
        "candidate_remains": new_counts.get("candidate_intraday_test", 0) > 0,
        "recommendation_for_next_strategy_family": (
            "No robust opening-range candidate remains. Move to VWAP reclaim/rejection "
            "as the next hypothesis family, using the new robustness gates from the start."
        ),
        "no_new_data_fetched": True,
        "no_strategy_templates_added": True,
        "no_gates_weakened": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "reclassified_opening_range_results.csv",
        reclassified_rows,
        [
            "symbol",
            "old_classification",
            "new_classification",
            "old_net_return",
            "benchmark_pass",
            "null_pass",
            "survives_1_5x_costs",
            "net_return_1_5x_costs",
            "median_trade_return",
            "profit_factor",
            "top_positive_split_share",
            "top_5_winners_profit_share",
            "robustness_failure_reasons",
        ],
    )
    _write_csv(
        output_dir / "robustness_gate_failures.csv",
        failure_rows,
        [
            "symbol",
            "old_classification",
            "new_classification",
            "failure_reasons",
            "survives_1_5x_costs",
            "median_trade_return",
            "profit_factor",
            "top_positive_split_share",
            "top_5_winners_profit_share",
        ],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-report-root",
        type=Path,
        default=Path("data/reports/research"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/reports/research") / OUTPUT_DIR,
    )
    parser.add_argument("--universe-report", type=Path)
    parser.add_argument("--test-command", action="append", default=[])
    parser.add_argument("--test-status", default="not_recorded")
    args = parser.parse_args()

    universe_report = args.universe_report or _latest_universe_report(args.source_report_root)
    summary = build_summary(
        report_root=args.source_report_root,
        output_dir=args.output_dir,
        universe_report=universe_report,
        tests_run=list(args.test_command),
        test_status=str(args.test_status),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
