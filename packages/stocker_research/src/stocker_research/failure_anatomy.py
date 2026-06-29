"""Stage 3.6 failure-anatomy diagnostics for existing research reports."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HOLDING_POLICY_REJECTION_CLASSIFICATIONS = {
    "rejected_overnight_risk",
    "rejected_weekend_risk",
    "rejected_holding_policy_violation",
}

EXPERIMENT_CONTRACT_FIELDS = [
    "evaluation_policy",
    "indicator_context_policy",
    "context_summary",
    "selected_result",
    "best_test_diagnostic",
    "benchmark_results",
    "null_model_results",
    "holding_policy",
    "holding_policy_analysis",
    "holding_policy_decision",
    "classification_reasons",
]


@dataclass(frozen=True)
class FailureAnatomySummaryResult:
    """Paths and headline counts for a Stage 3.6 diagnostic run."""

    summary_json_path: Path
    summary_markdown_path: Path
    selected_cases_csv_path: Path | None
    report_count_analyzed: int
    malformed_report_count: int
    classification_counts: dict[str, int]
    top_diagnostic_findings: list[str]
    recommended_next_step: str


@dataclass(frozen=True)
class ExperimentReport:
    """Normalized view of one experiment JSON report."""

    payload: dict[str, Any]
    json_path: Path
    report_path: Path
    hypothesis: str
    symbol: str
    classification: str
    classification_reasons: list[str]
    selected_result: dict[str, Any]
    best_test_diagnostic: dict[str, Any]
    selection: dict[str, Any]
    benchmark_pass: bool
    null_pass: bool
    holding_policy_analysis: dict[str, Any]
    holding_policy_decision: dict[str, Any]
    stability: dict[str, Any]
    minimum_required_trades: int


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "JSON file must contain an object"
    return payload, None


def _is_output_path(path: Path, output_dir: Path) -> bool:
    try:
        path.relative_to(output_dir)
    except ValueError:
        return False
    return True


def _is_universe_report(payload: dict[str, Any]) -> bool:
    return "symbol_results" in payload and "classification_counts" in payload


def _looks_like_experiment_report(payload: dict[str, Any]) -> bool:
    return bool(
        {"classification", "classification_reasons", "selected_result", "hypothesis"} & set(payload)
    )


def _missing_contract_fields(payload: dict[str, Any]) -> list[str]:
    missing = [field for field in EXPERIMENT_CONTRACT_FIELDS if field not in payload]
    for field in ("classification", "hypothesis", "symbol"):
        if field not in payload:
            missing.append(field)
    return missing


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _hypothesis_id(payload: dict[str, Any]) -> str:
    hypothesis = _dict(payload.get("hypothesis"))
    return str(hypothesis.get("id") or hypothesis.get("name") or "unknown_hypothesis")


def _minimum_required_trades(payload: dict[str, Any]) -> int:
    hypothesis = _dict(payload.get("hypothesis"))
    minimum_evidence = _dict(hypothesis.get("minimum_evidence"))
    return _as_int(minimum_evidence.get("min_trades"), 20)


def _selected_trade_count(selected_result: dict[str, Any]) -> int:
    return _as_int(
        selected_result.get("test_trade_count", selected_result.get("trade_count")),
        0,
    )


def _selected_max_drawdown(selected_result: dict[str, Any]) -> float:
    return _as_float(
        selected_result.get("test_max_drawdown", selected_result.get("max_drawdown")),
        0.0,
    )


def _selected_test_net_return(selected_result: dict[str, Any]) -> float:
    return _as_float(selected_result.get("test_net_return"), 0.0)


def _selected_train_net_return(selected_result: dict[str, Any]) -> float:
    return _as_float(selected_result.get("train_net_return"), 0.0)


def _null_pass(payload: dict[str, Any]) -> bool:
    null_model_results = _dict(payload.get("null_model_results"))
    return _as_bool(null_model_results.get("null_pass", False))


def _excess_vs_null(payload: dict[str, Any]) -> float:
    null_model_results = _dict(payload.get("null_model_results"))
    return _as_float(null_model_results.get("selected_excess_vs_p75_null"), 0.0)


def _selection_succeeded(report: ExperimentReport) -> bool:
    if "no_train_selected_parameter" in report.classification_reasons:
        return False
    selected_parameter_set_id = str(report.selected_result.get("parameter_set_id", "none"))
    selection_method = str(report.selection.get("selection_method", ""))
    return selected_parameter_set_id != "none" and selection_method != "fallback_for_reporting_only"


def _holding_policy_rejected(report: ExperimentReport) -> bool:
    return report.classification in HOLDING_POLICY_REJECTION_CLASSIFICATIONS


def _test_return_bucket(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _quantile(values: list[float], percentile: float) -> float | None:
    clean = sorted(values)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(clean) - 1)
    fraction = rank - lower
    return round(clean[lower] + (clean[upper] - clean[lower]) * fraction, 12)


def _median(values: list[float]) -> float | None:
    return _quantile(values, 0.5)


def _normalize_experiment(payload: dict[str, Any], json_path: Path) -> ExperimentReport:
    selected_result = _dict(payload.get("selected_result"))
    return ExperimentReport(
        payload=payload,
        json_path=json_path,
        report_path=json_path.with_suffix(".md"),
        hypothesis=_hypothesis_id(payload),
        symbol=str(payload.get("symbol", "")),
        classification=str(payload.get("classification", "")),
        classification_reasons=[
            str(reason) for reason in _as_list(payload.get("classification_reasons"))
        ],
        selected_result=selected_result,
        best_test_diagnostic=_dict(payload.get("best_test_diagnostic")),
        selection=_dict(payload.get("selection")),
        benchmark_pass=_as_bool(payload.get("benchmark_pass", False)),
        null_pass=_null_pass(payload),
        holding_policy_analysis=_dict(payload.get("holding_policy_analysis")),
        holding_policy_decision=_dict(payload.get("holding_policy_decision")),
        stability=_dict(payload.get("stability")),
        minimum_required_trades=_minimum_required_trades(payload),
    )


def _discover_reports(
    report_root: Path,
    output_dir: Path,
) -> tuple[list[ExperimentReport], list[dict[str, Any]], list[dict[str, Any]]]:
    experiments: list[ExperimentReport] = []
    universe_reports: list[dict[str, Any]] = []
    malformed_reports: list[dict[str, Any]] = []
    for path in sorted(report_root.rglob("*.json")):
        if path.name == "index.json" or _is_output_path(path, output_dir):
            continue
        payload, error = _load_json(path)
        if payload is None:
            malformed_reports.append(
                {"json_path": str(path), "missing_fields": [], "error": error or "load_error"}
            )
            continue
        if _is_universe_report(payload):
            universe_reports.append({"json_path": str(path), "payload": payload})
            continue
        if not _looks_like_experiment_report(payload):
            continue
        missing_fields = _missing_contract_fields(payload)
        if missing_fields:
            malformed_reports.append(
                {
                    "json_path": str(path),
                    "missing_fields": missing_fields,
                    "error": "missing_required_contract_fields",
                }
            )
            continue
        experiments.append(_normalize_experiment(payload, path))
    return experiments, universe_reports, malformed_reports


def _classification_anatomy(reports: list[ExperimentReport]) -> dict[str, Any]:
    classification_counts = Counter(report.classification for report in reports)
    reason_counts = Counter(
        reason for report in reports for reason in report.classification_reasons
    )
    return {
        "total_experiments": len(reports),
        "classification_counts": dict(classification_counts),
        "classification_reason_counts": dict(reason_counts),
        "rejected_no_edge_count": classification_counts.get("rejected_no_edge", 0),
        "rejected_too_few_trades_count": classification_counts.get(
            "rejected_too_few_trades",
            0,
        ),
        "rejected_costs_kill_edge_count": classification_counts.get(
            "rejected_costs_kill_edge",
            0,
        ),
        "holding_policy_rejection_count": sum(
            1 for report in reports if _holding_policy_rejected(report)
        ),
        "failed_benchmark_count": sum(1 for report in reports if not report.benchmark_pass),
        "failed_null_timing_count": sum(1 for report in reports if not report.null_pass),
        "no_train_selected_parameter_count": reason_counts.get(
            "no_train_selected_parameter",
            0,
        ),
    }


def _partial_pass_matrix(reports: list[ExperimentReport]) -> list[dict[str, Any]]:
    counts: Counter[tuple[bool, bool, str, bool, bool]] = Counter()
    for report in reports:
        counts[
            (
                report.benchmark_pass,
                report.null_pass,
                _test_return_bucket(_selected_test_net_return(report.selected_result)),
                _selection_succeeded(report),
                _holding_policy_rejected(report),
            )
        ] += 1
    rows = []
    for (
        benchmark_pass,
        null_pass,
        test_net_return,
        train_selection_succeeded,
        holding_policy_rejected,
    ), count in sorted(counts.items(), key=lambda item: (str(item[0]), item[1]), reverse=True):
        rows.append(
            {
                "benchmark_pass": benchmark_pass,
                "null_pass": null_pass,
                "test_net_return": test_net_return,
                "train_selection_succeeded": train_selection_succeeded,
                "holding_policy_rejected": holding_policy_rejected,
                "count": count,
                "diagnostic_only": True,
            }
        )
    return rows


def _metric_distributions(reports: list[ExperimentReport]) -> dict[str, float | None]:
    test_returns = [_selected_test_net_return(report.selected_result) for report in reports]
    return {
        "median_selected_test_net_return": _median(test_returns),
        "p25_selected_test_net_return": _quantile(test_returns, 0.25),
        "p75_selected_test_net_return": _quantile(test_returns, 0.75),
        "median_train_net_return": _median(
            [_selected_train_net_return(report.selected_result) for report in reports]
        ),
        "median_benchmark_excess": _median(
            [
                _as_float(report.payload.get("selected_excess_vs_buy_and_hold"), 0.0)
                for report in reports
            ]
        ),
        "median_null_p75_excess": _median([_excess_vs_null(report.payload) for report in reports]),
        "median_trade_count": _median(
            [float(_selected_trade_count(report.selected_result)) for report in reports]
        ),
        "median_max_drawdown": _median(
            [_selected_max_drawdown(report.selected_result) for report in reports]
        ),
        "median_stability_score": _median(
            [_as_float(report.stability.get("stability_score"), 0.0) for report in reports]
        ),
        "median_gap_contribution_pct": _median(
            [
                _as_float(report.holding_policy_analysis.get("gap_return_contribution_pct"), 0.0)
                for report in reports
            ]
        ),
        "median_overnight_exposure_count": _median(
            [
                float(_as_int(report.holding_policy_analysis.get("overnight_exposure_count"), 0))
                for report in reports
            ]
        ),
        "median_weekend_exposure_count": _median(
            [
                float(_as_int(report.holding_policy_analysis.get("weekend_exposure_count"), 0))
                for report in reports
            ]
        ),
        "median_max_holding_sessions": _median(
            [
                float(_as_int(report.holding_policy_analysis.get("estimated_holding_sessions"), 0))
                for report in reports
            ]
        ),
    }


def _diagnostic_case(report: ExperimentReport) -> dict[str, Any]:
    return {
        "diagnostic_case_type": "rejected_diagnostic_case",
        "diagnostic_only": True,
        "symbol": report.symbol,
        "hypothesis": report.hypothesis,
        "classification": report.classification,
        "classification_reasons": report.classification_reasons,
        "test_net_return": _selected_test_net_return(report.selected_result),
        "trade_count": _selected_trade_count(report.selected_result),
        "benchmark_pass": report.benchmark_pass,
        "null_pass": report.null_pass,
        "selected_excess_vs_buy_and_hold": _as_float(
            report.payload.get("selected_excess_vs_buy_and_hold"),
            0.0,
        ),
        "selected_excess_vs_p75_null": _excess_vs_null(report.payload),
        "max_drawdown": _selected_max_drawdown(report.selected_result),
        "stability_score": _as_float(report.stability.get("stability_score"), 0.0),
        "gap_return_contribution_pct": _as_float(
            report.holding_policy_analysis.get("gap_return_contribution_pct"),
            0.0,
        ),
        "overnight_exposure_count": _as_int(
            report.holding_policy_analysis.get("overnight_exposure_count"),
            0,
        ),
        "weekend_exposure_count": _as_int(
            report.holding_policy_analysis.get("weekend_exposure_count"),
            0,
        ),
        "report_path": str(report.report_path),
        "json_path": str(report.json_path),
    }


def _best_rejected_cases(
    reports_by_hypothesis: dict[str, list[ExperimentReport]],
) -> dict[str, Any]:
    score_functions = {
        "by_selected_excess_vs_buy_and_hold": lambda report: _as_float(
            report.payload.get("selected_excess_vs_buy_and_hold"),
            0.0,
        ),
        "by_selected_excess_vs_p75_null": lambda report: _excess_vs_null(report.payload),
        "by_selected_test_net_return": lambda report: _selected_test_net_return(
            report.selected_result
        ),
        "by_stability_score": lambda report: _as_float(
            report.stability.get("stability_score"),
            0.0,
        ),
    }
    output: dict[str, Any] = {}
    for hypothesis, reports in reports_by_hypothesis.items():
        rejected = [report for report in reports if report.classification.startswith("rejected_")]
        output[hypothesis] = {}
        for key, score in score_functions.items():
            output[hypothesis][key] = [
                _diagnostic_case(report)
                for report in sorted(rejected, key=score, reverse=True)[:10]
            ]
    return output


def _too_few_trades_drilldown(reports: list[ExperimentReport]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        if report.classification != "rejected_too_few_trades" and (
            "too_few_trades" not in report.classification_reasons
        ):
            continue
        rows.append(
            {
                "symbol": report.symbol,
                "hypothesis": report.hypothesis,
                "selected_parameter_set": {
                    "parameter_set_id": str(report.selected_result.get("parameter_set_id", "")),
                    "params": _dict(report.selected_result.get("params")),
                },
                "trade_count": _selected_trade_count(report.selected_result),
                "minimum_required_trades": report.minimum_required_trades,
                "test_net_return": _selected_test_net_return(report.selected_result),
                "benchmark_pass": report.benchmark_pass,
                "null_pass": report.null_pass,
                "report_path": str(report.report_path),
            }
        )
    return rows


def _train_selection_failure_drilldown(reports: list[ExperimentReport]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        if _selection_succeeded(report):
            continue
        rows.append(
            {
                "symbol": report.symbol,
                "hypothesis": report.hypothesis,
                "selected_result": {
                    "parameter_set_id": str(report.selected_result.get("parameter_set_id", "")),
                    "test_net_return": _selected_test_net_return(report.selected_result),
                },
                "best_test_diagnostic": {
                    "parameter_set_id": str(
                        report.best_test_diagnostic.get("parameter_set_id", "")
                    ),
                    "test_net_return": _as_float(
                        report.best_test_diagnostic.get("test_net_return"),
                        0.0,
                    ),
                },
                "selection_diagnostics": report.selection.get("diagnostics", {}),
                "report_path": str(report.report_path),
            }
        )
    return rows


def _holding_policy_drilldown(reports: list[ExperimentReport]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        violations = [
            str(item)
            for item in _as_list(report.holding_policy_analysis.get("holding_policy_violations"))
        ]
        warnings = [
            str(item)
            for item in _as_list(
                report.holding_policy_analysis.get("holding_policy_warning_reasons")
            )
        ]
        decision_reasons = [
            str(item) for item in _as_list(report.holding_policy_decision.get("reasons"))
        ]
        if not (_holding_policy_rejected(report) or violations or warnings or decision_reasons):
            continue
        rows.append(
            {
                "symbol": report.symbol,
                "hypothesis": report.hypothesis,
                "classification": report.classification,
                "evidence_tier": str(
                    report.holding_policy_decision.get(
                        "evidence_tier",
                        report.holding_policy_analysis.get("evidence_tier", ""),
                    )
                ),
                "holding_policy_decision": report.holding_policy_decision,
                "max_holding_sessions": _as_int(
                    report.holding_policy_analysis.get("estimated_holding_sessions"),
                    0,
                ),
                "overnight_exposure_count": _as_int(
                    report.holding_policy_analysis.get("overnight_exposure_count"),
                    0,
                ),
                "weekend_exposure_count": _as_int(
                    report.holding_policy_analysis.get("weekend_exposure_count"),
                    0,
                ),
                "gap_return_contribution_pct": _as_float(
                    report.holding_policy_analysis.get("gap_return_contribution_pct"),
                    0.0,
                ),
                "holding_policy_violations": violations,
                "holding_policy_warning_reasons": warnings,
                "report_path": str(report.report_path),
            }
        )
    return rows


def _classification_counts(reports: list[ExperimentReport]) -> dict[str, int]:
    return dict(Counter(report.classification for report in reports))


def _top_reason_counts(reports: list[ExperimentReport], limit: int = 10) -> dict[str, int]:
    return dict(
        Counter(
            reason for report in reports for reason in report.classification_reasons
        ).most_common(limit)
    )


def _candidate_count(classification_counts: dict[str, int]) -> int:
    return sum(
        classification_counts.get(name, 0)
        for name in (
            "candidate_intraday_test",
            "candidate_swing_exceptional",
            "candidate_paper_test",
        )
    )


def _recommended_next_step(
    *,
    reports: list[ExperimentReport],
    malformed_count: int,
    classification_counts: dict[str, int],
    reason_counts: Counter[str],
) -> str:
    if malformed_count and malformed_count >= max(1, len(reports) // 10):
        return "d) fix a harness/reporting issue"
    if reason_counts.get("session_flat_unproven", 0) == len(reports) and reports:
        return "b) move to intraday/session-flat hypothesis variants"
    rejected_no_edge = classification_counts.get("rejected_no_edge", 0)
    if rejected_no_edge >= max(1, len(reports) // 2):
        return "a) write better daily/swing hypotheses"
    holding_rejections = sum(1 for report in reports if _holding_policy_rejected(report))
    if holding_rejections >= max(1, len(reports) // 4):
        return "b) move to intraday/session-flat hypothesis variants"
    sparse_count = classification_counts.get("rejected_too_few_trades", 0)
    if sparse_count >= max(1, len(reports) // 4):
        return "c) improve data/universe context filters"
    return "a) write better daily/swing hypotheses"


def _top_findings(
    *,
    reports: list[ExperimentReport],
    malformed_count: int,
    classification_counts: dict[str, int],
    reason_counts: Counter[str],
) -> list[str]:
    findings = [
        (
            f"{len(reports)} experiment reports analyzed; "
            f"{malformed_count} malformed reports recorded."
        ),
    ]
    if classification_counts:
        top_classification, top_count = Counter(classification_counts).most_common(1)[0]
        findings.append(f"Largest classification bucket: {top_classification} ({top_count}).")
    if reason_counts:
        top_reason, top_reason_count = reason_counts.most_common(1)[0]
        findings.append(f"Most common reason: {top_reason} ({top_reason_count}).")
    candidate_count = _candidate_count(classification_counts)
    findings.append(
        f"Candidate-labelled reports remain {candidate_count}; "
        "rejected cases are diagnostic only."
    )
    holding_rejections = sum(1 for report in reports if _holding_policy_rejected(report))
    findings.append(f"Holding-policy rejection count: {holding_rejections}.")
    return findings


def _group_by_hypothesis(reports: list[ExperimentReport]) -> dict[str, list[ExperimentReport]]:
    grouped: dict[str, list[ExperimentReport]] = defaultdict(list)
    for report in reports:
        grouped[report.hypothesis].append(report)
    return dict(sorted(grouped.items()))


def _hypothesis_summaries(
    reports_by_hypothesis: dict[str, list[ExperimentReport]],
) -> dict[str, Any]:
    return {
        hypothesis: {
            "classification_anatomy": _classification_anatomy(reports),
            "partial_pass_matrix": _partial_pass_matrix(reports),
            "metric_distributions": _metric_distributions(reports),
        }
        for hypothesis, reports in reports_by_hypothesis.items()
    }


def _report_contract_check(
    experiments: list[ExperimentReport],
    malformed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "required_fields": EXPERIMENT_CONTRACT_FIELDS,
        "checked_report_count": len(experiments) + len(malformed_reports),
        "valid_report_count": len(experiments),
        "malformed_report_count": len(malformed_reports),
        "passed": not malformed_reports,
        "malformed_reports": malformed_reports,
    }


def _final_interpretation(
    *,
    reports: list[ExperimentReport],
    classification_counts: dict[str, int],
    reason_counts: Counter[str],
    malformed_count: int,
    recommended_next_step: str,
) -> dict[str, str]:
    candidate_count = _candidate_count(classification_counts)
    if candidate_count:
        continue_as_is = (
            "Some reports have candidate labels, but Stage 3.6 does not relabel or promote them; "
            "review the original gates before treating any result as actionable research."
        )
    else:
        continue_as_is = (
            "The current daily templates do not show a reason to continue as-is. "
            "The best-ranked rows in this summary are rejected diagnostic cases, not candidates."
        )

    no_edge = classification_counts.get("rejected_no_edge", 0)
    sparse = classification_counts.get("rejected_too_few_trades", 0)
    train_selection = reason_counts.get("no_train_selected_parameter", 0)
    holding = sum(1 for report in reports if _holding_policy_rejected(report))
    failure_mix = (
        "Failures are mostly "
        f"no-edge ({no_edge}), too-few-trades ({sparse}), "
        f"train-selection failures ({train_selection}), and holding-risk rejections ({holding})."
    )
    if malformed_count:
        failure_mix += (
            f" {malformed_count} malformed reports were also recorded for contract follow-up."
        )

    return {
        "continue_as_is": continue_as_is,
        "failure_mix": failure_mix,
        "recommended_next_step": recommended_next_step,
    }


def _build_summary_payload(
    *,
    report_root: Path,
    output_dir: Path,
    experiments: list[ExperimentReport],
    universe_reports: list[dict[str, Any]],
    malformed_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    reports_by_hypothesis = _group_by_hypothesis(experiments)
    classification_counts = _classification_counts(experiments)
    reason_counts = Counter(
        reason for report in experiments for reason in report.classification_reasons
    )
    recommended_next_step = _recommended_next_step(
        reports=experiments,
        malformed_count=len(malformed_reports),
        classification_counts=classification_counts,
        reason_counts=reason_counts,
    )
    top_findings = _top_findings(
        reports=experiments,
        malformed_count=len(malformed_reports),
        classification_counts=classification_counts,
        reason_counts=reason_counts,
    )
    return {
        "stage": "3.6",
        "generated_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
        "diagnostic_only": True,
        "report_root": str(report_root),
        "output_dir": str(output_dir),
        "report_count_analyzed": len(experiments),
        "universe_report_count_analyzed": len(universe_reports),
        "malformed_report_count": len(malformed_reports),
        "classification_counts": classification_counts,
        "classification_reason_counts": dict(reason_counts),
        "top_rejection_context_reasons": _top_reason_counts(experiments),
        "hypotheses": _hypothesis_summaries(reports_by_hypothesis),
        "best_rejected_cases": _best_rejected_cases(reports_by_hypothesis),
        "too_few_trades_drilldown": _too_few_trades_drilldown(experiments),
        "train_selection_failure_drilldown": _train_selection_failure_drilldown(experiments),
        "holding_policy_drilldown": _holding_policy_drilldown(experiments),
        "report_contract_check": _report_contract_check(experiments, malformed_reports),
        "top_diagnostic_findings": top_findings,
        "final_interpretation": _final_interpretation(
            reports=experiments,
            classification_counts=classification_counts,
            reason_counts=reason_counts,
            malformed_count=len(malformed_reports),
            recommended_next_step=recommended_next_step,
        ),
    }


def _markdown_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "- None"
    return "\n".join(f"- `{key}`: {value}" for key, value in sorted(counts.items()))


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _summary_markdown(payload: dict[str, Any]) -> str:
    hypothesis_sections = []
    for hypothesis, summary in payload["hypotheses"].items():
        anatomy = summary["classification_anatomy"]
        metrics = summary["metric_distributions"]
        partial_pass_rows = summary["partial_pass_matrix"]
        partial_pass_columns = [
            "benchmark_pass",
            "null_pass",
            "test_net_return",
            "train_selection_succeeded",
            "holding_policy_rejected",
            "count",
            "diagnostic_only",
        ]
        hypothesis_sections.append(
            f"""## Hypothesis: `{hypothesis}`

### Classification Anatomy

- Total experiments: {anatomy["total_experiments"]}
- Rejected no-edge: {anatomy["rejected_no_edge_count"]}
- Rejected too-few-trades: {anatomy["rejected_too_few_trades_count"]}
- Rejected costs-kill-edge: {anatomy["rejected_costs_kill_edge_count"]}
- Holding-policy rejections: {anatomy["holding_policy_rejection_count"]}
- Failed benchmark: {anatomy["failed_benchmark_count"]}
- Failed null timing: {anatomy["failed_null_timing_count"]}
- No train-selected parameter: {anatomy["no_train_selected_parameter_count"]}

Classification counts:

{_markdown_counts(anatomy["classification_counts"])}

Classification reason counts:

{_markdown_counts(anatomy["classification_reason_counts"])}

### Partial-Pass Matrix

{_markdown_table(partial_pass_rows, partial_pass_columns)}

### Metric Distributions

{_markdown_counts({key: value for key, value in metrics.items() if value is not None})}
"""
        )

    findings = "\n".join(f"- {finding}" for finding in payload["top_diagnostic_findings"])
    interpretation = payload["final_interpretation"]
    contract = payload["report_contract_check"]
    return f"""# Stage 3.6 Failure Anatomy

This report is diagnostic only. It analyzes existing research reports and does not
promote rejected daily-bar partial passes into candidates.

## Run Summary

- Experiment reports analyzed: {payload["report_count_analyzed"]}
- Universe reports observed: {payload["universe_report_count_analyzed"]}
- Malformed experiment reports: {payload["malformed_report_count"]}
- Output directory: `{payload["output_dir"]}`

## Classification Counts

{_markdown_counts(payload["classification_counts"])}

## Top Diagnostic Findings

{findings}

{"".join(hypothesis_sections)}

## Drilldowns

- Too-few-trades rows: {len(payload["too_few_trades_drilldown"])}
- Train-selection failure rows: {len(payload["train_selection_failure_drilldown"])}
- Holding-policy rows: {len(payload["holding_policy_drilldown"])}

## Report Contract Check

- Passed: {contract["passed"]}
- Valid experiment reports: {contract["valid_report_count"]}
- Malformed experiment reports: {contract["malformed_report_count"]}

## Final Interpretation

{interpretation["continue_as_is"]}

{interpretation["failure_mix"]}

Recommended next step: {interpretation["recommended_next_step"]}
"""


def _flatten_selected_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hypothesis, grouped in payload["best_rejected_cases"].items():
        for ranking, cases in grouped.items():
            for rank, case in enumerate(cases, start=1):
                row = {
                    "hypothesis": hypothesis,
                    "ranking": ranking,
                    "rank": rank,
                    **case,
                    "classification_reasons": ";".join(case.get("classification_reasons", [])),
                }
                rows.append(row)
    return rows


def _write_selected_cases_csv(output_dir: Path, payload: dict[str, Any]) -> Path | None:
    rows = _flatten_selected_cases(payload)
    if not rows:
        return None
    path = output_dir / "selected_cases.csv"
    fieldnames = [
        "hypothesis",
        "ranking",
        "rank",
        "diagnostic_case_type",
        "diagnostic_only",
        "symbol",
        "classification",
        "classification_reasons",
        "test_net_return",
        "trade_count",
        "benchmark_pass",
        "null_pass",
        "selected_excess_vs_buy_and_hold",
        "selected_excess_vs_p75_null",
        "max_drawdown",
        "stability_score",
        "gap_return_contribution_pct",
        "overnight_exposure_count",
        "weekend_exposure_count",
        "report_path",
        "json_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_failure_anatomy_summary(
    *,
    report_root: Path = Path("data/reports/research"),
    output_dir: Path | None = None,
) -> FailureAnatomySummaryResult:
    """Build Stage 3.6 diagnostics from existing research JSON reports."""

    resolved_report_root = report_root.expanduser()
    resolved_output_dir = (
        output_dir.expanduser()
        if output_dir is not None
        else resolved_report_root / "stage3_6_failure_anatomy"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    experiments, universe_reports, malformed_reports = _discover_reports(
        resolved_report_root,
        resolved_output_dir,
    )
    payload = _build_summary_payload(
        report_root=resolved_report_root,
        output_dir=resolved_output_dir,
        experiments=experiments,
        universe_reports=universe_reports,
        malformed_reports=malformed_reports,
    )

    summary_json_path = resolved_output_dir / "summary.json"
    summary_markdown_path = resolved_output_dir / "summary.md"
    summary_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    summary_markdown_path.write_text(_summary_markdown(payload), encoding="utf-8")
    selected_cases_csv_path = _write_selected_cases_csv(resolved_output_dir, payload)

    return FailureAnatomySummaryResult(
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        selected_cases_csv_path=selected_cases_csv_path,
        report_count_analyzed=len(experiments),
        malformed_report_count=len(malformed_reports),
        classification_counts=payload["classification_counts"],
        top_diagnostic_findings=payload["top_diagnostic_findings"],
        recommended_next_step=payload["final_interpretation"]["recommended_next_step"],
    )
