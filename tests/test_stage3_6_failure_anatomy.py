import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.failure_anatomy import build_failure_anatomy_summary


def _experiment_payload(
    *,
    symbol: str,
    hypothesis_id: str,
    classification: str,
    reasons: list[str],
    test_net_return: float,
    train_net_return: float,
    trade_count: int,
    benchmark_pass: bool,
    null_pass: bool,
    selection_method: str = "train_side_evidence",
    selected_parameter_set_id: str = "ps_0001",
    best_test_parameter_set_id: str = "ps_0002",
    best_test_net_return: float = 0.12,
    holding_classification: str = "rejected_holding_policy_violation",
    holding_reasons: list[str] | None = None,
) -> dict[str, Any]:
    selected_result = {
        "parameter_set_id": selected_parameter_set_id,
        "params": {"fast_window": 5, "slow_window": 20},
        "train_net_return": train_net_return,
        "test_net_return": test_net_return,
        "test_trade_count": trade_count,
        "trade_count": trade_count,
        "test_max_drawdown": -0.08,
        "max_drawdown": -0.08,
    }
    best_test_diagnostic = {
        "parameter_set_id": best_test_parameter_set_id,
        "params": {"fast_window": 8, "slow_window": 30},
        "test_net_return": best_test_net_return,
        "test_trade_count": max(trade_count, 25),
    }
    return {
        "experiment_id": f"20260628_{hypothesis_id}_{symbol}_1d",
        "symbol": symbol,
        "timeframe": "1d",
        "classification": classification,
        "classification_reasons": reasons,
        "hypothesis": {
            "id": hypothesis_id,
            "name": hypothesis_id.replace("_", " ").title(),
            "minimum_evidence": {"min_trades": 20},
        },
        "evaluation_policy": "walk_forward_with_indicator_context",
        "indicator_context_policy": "historical_indicator_context_before_window_not_scored",
        "context_summary": {"context_rows_are_scored": False},
        "selected_result": selected_result,
        "best_test_diagnostic": best_test_diagnostic,
        "benchmark_results": {"buy_and_hold": {"net_return": 0.03}},
        "selected_excess_vs_buy_and_hold": 0.04,
        "null_model_results": {
            "null_pass": null_pass,
            "selected_excess_vs_p75_null": 0.02,
        },
        "benchmark_pass": benchmark_pass,
        "holding_policy": {"preferred_style": "intraday"},
        "holding_policy_analysis": {
            "evidence_tier": "rejected_holding_risk",
            "estimated_holding_sessions": 4,
            "overnight_exposure_count": 3,
            "weekend_exposure_count": 1,
            "gap_return_contribution_pct": 0.55,
            "holding_policy_violations": ["max_holding_sessions_exceeded"],
            "holding_policy_warning_reasons": [
                "daily_bars_are_swing_research_vehicle",
                "held_overnight",
            ],
        },
        "holding_policy_decision": {
            "classification": holding_classification,
            "evidence_tier": "rejected_holding_risk",
            "reasons": holding_reasons or ["max_holding_sessions_exceeded"],
        },
        "selection": {
            "selection_method": selection_method,
            "selected_parameter_set_id": selected_parameter_set_id,
            "diagnostics": {
                "reason": "no parameter passed train-side evidence"
                if selection_method == "fallback_for_reporting_only"
                else "selected on train-side evidence"
            },
        },
        "stability": {"stability_score": 0.42},
    }


def _write_experiment(report_root: Path, payload: dict[str, Any]) -> Path:
    json_path = report_root / f"{payload['experiment_id']}.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    json_path.with_suffix(".md").write_text("# Experiment report\n", encoding="utf-8")
    return json_path


def test_failure_anatomy_summary_groups_rejected_diagnostics(tmp_path: Path) -> None:
    report_root = tmp_path / "data" / "reports" / "research"
    report_root.mkdir(parents=True)
    _write_experiment(
        report_root,
        _experiment_payload(
            symbol="AAPL.US",
            hypothesis_id="moving_average_momentum",
            classification="rejected_too_few_trades",
            reasons=["too_few_trades"],
            test_net_return=0.07,
            train_net_return=0.03,
            trade_count=7,
            benchmark_pass=True,
            null_pass=True,
        ),
    )
    _write_experiment(
        report_root,
        _experiment_payload(
            symbol="MSFT.US",
            hypothesis_id="moving_average_momentum",
            classification="rejected_no_edge",
            reasons=["no_train_selected_parameter"],
            test_net_return=-0.02,
            train_net_return=-0.01,
            trade_count=26,
            benchmark_pass=False,
            null_pass=False,
            selection_method="fallback_for_reporting_only",
            selected_parameter_set_id="none",
            best_test_parameter_set_id="ps_0042",
            best_test_net_return=0.19,
        ),
    )
    _write_experiment(
        report_root,
        _experiment_payload(
            symbol="META.US",
            hypothesis_id="pullback_in_uptrend",
            classification="rejected_holding_policy_violation",
            reasons=["max_holding_sessions_exceeded", "daily_bars_are_swing_research_vehicle"],
            test_net_return=0.11,
            train_net_return=0.08,
            trade_count=34,
            benchmark_pass=True,
            null_pass=True,
        ),
    )
    malformed_path = report_root / "malformed_experiment.json"
    malformed_path.write_text(json.dumps({"classification": "rejected_no_edge"}), encoding="utf-8")

    result = build_failure_anatomy_summary(report_root=report_root)

    assert result.report_count_analyzed == 3
    assert result.malformed_report_count == 1
    payload = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    ma = payload["hypotheses"]["moving_average_momentum"]
    assert ma["classification_anatomy"]["total_experiments"] == 2
    assert ma["classification_anatomy"]["rejected_too_few_trades_count"] == 1
    assert ma["classification_anatomy"]["failed_benchmark_count"] == 1
    assert ma["classification_anatomy"]["failed_null_timing_count"] == 1
    assert ma["classification_anatomy"]["no_train_selected_parameter_count"] == 1
    assert ma["partial_pass_matrix"][0]["diagnostic_only"] is True
    assert ma["metric_distributions"]["median_selected_test_net_return"] == 0.025
    assert payload["best_rejected_cases"]["moving_average_momentum"][
        "by_selected_test_net_return"
    ][0]["diagnostic_case_type"] == "rejected_diagnostic_case"
    assert payload["too_few_trades_drilldown"][0]["minimum_required_trades"] == 20
    assert payload["train_selection_failure_drilldown"][0]["best_test_diagnostic"][
        "parameter_set_id"
    ] == "ps_0042"
    assert payload["holding_policy_drilldown"][0]["holding_policy_decision"]["evidence_tier"]
    assert payload["report_contract_check"]["malformed_report_count"] == 1
    assert "daily templates" in result.summary_markdown_path.read_text(encoding="utf-8")
    assert result.selected_cases_csv_path is not None
    assert result.selected_cases_csv_path.exists()


def test_failure_anatomy_cli_writes_stage3_6_outputs(tmp_path: Path) -> None:
    report_root = tmp_path / "data" / "reports" / "research"
    report_root.mkdir(parents=True)
    _write_experiment(
        report_root,
        _experiment_payload(
            symbol="AAPL.US",
            hypothesis_id="volatility_breakout",
            classification="rejected_no_edge",
            reasons=["no_positive_net_edge"],
            test_net_return=-0.03,
            train_net_return=0.01,
            trade_count=40,
            benchmark_pass=False,
            null_pass=True,
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "research",
            "failure-anatomy",
            "--reports-dir",
            str(report_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "stage3_6_failure_anatomy" in result.output
    assert "report_count_analyzed" in result.output
    assert "malformed_report_count" in result.output
    assert "classification_counts" in result.output
    assert "recommended_next_step" in result.output
    assert (report_root / "stage3_6_failure_anatomy" / "summary.json").exists()
    assert (report_root / "stage3_6_failure_anatomy" / "summary.md").exists()
