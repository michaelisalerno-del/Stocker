"""Research experiment runner and conservative result classification."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import DirectionMode, VectorizedBacktestResult, evaluate_positions
from stocker_data.audit import create_audit_report
from stocker_data.storage import DatasetKey, dataset_metadata, load_dataset
from stocker_data.universe import load_research_ready_universe
from stocker_research.benchmarks import compare_with_benchmarks
from stocker_research.classification import ResearchClassification, classify_research_result
from stocker_research.hypothesis import Hypothesis, load_hypothesis
from stocker_research.leakage import (
    LeakageIssue,
    check_timestamp_integrity,
    collect_research_leakage_issues,
)
from stocker_research.null_models import run_null_timing_test
from stocker_research.parameters import ParameterGrid, ParameterSet
from stocker_research.regime import label_regimes, performance_by_regime
from stocker_research.selection import SelectionResult, select_parameter_set
from stocker_research.stability import StabilityReport, analyze_stability
from stocker_research.templates import StrategyTemplate, get_template
from stocker_research.walkforward import (
    WalkForwardConfig,
    WalkForwardSplit,
    generate_walk_forward_splits,
)

Classification = (
    ResearchClassification
    | Literal[
        "rejected_data_issue",
        "rejected_insufficient_data",
        "rejected_no_edge",
        "rejected_costs_kill_edge",
        "rejected_unstable_parameters",
        "rejected_walkforward_failure",
        "rejected_too_few_trades",
        "interesting_needs_more_tests",
        "candidate_paper_test",
    ]
)


@dataclass(frozen=True)
class ExperimentRunResult:
    """Paths and classification from a research experiment."""

    experiment_id: str
    classification: Classification
    markdown_path: Path
    json_path: Path


@dataclass(frozen=True)
class UniverseResearchRunResult:
    """Paths and aggregate counts from a universe research run."""

    run_id: str
    markdown_path: Path
    json_path: Path
    classification_counts: dict[str, int]
    failed_count: int


def classify_experiment(
    *,
    test_net_return: float,
    train_net_return: float,
    stability_score: float,
    profitable_split_pct: float,
    trade_count: int,
    max_drawdown: float,
    regime_count: int,
    warnings: list[str],
) -> Classification:
    """Classify an experiment conservatively."""

    if any("data" in warning.lower() for warning in warnings):
        return "rejected_data_issue"
    if train_net_return > 0 and test_net_return <= 0:
        return "rejected_walkforward_failure"
    if test_net_return <= 0:
        return "rejected_no_edge"
    if train_net_return > 0 and test_net_return < train_net_return * 0.25:
        return "rejected_costs_kill_edge"
    if stability_score < 0.5:
        return "rejected_unstable_parameters"
    if profitable_split_pct < 0.6:
        return "rejected_walkforward_failure"
    if trade_count < 20 or max_drawdown < -0.25 or regime_count < 2:
        return "interesting_needs_more_tests"
    return "candidate_paper_test"


def _template_for(hypothesis: Hypothesis) -> StrategyTemplate:
    return get_template(hypothesis.template)


def _cost_model(hypothesis: Hypothesis) -> CostModel:
    return CostModel(
        spread_bps=hypothesis.costs.spread_bps,
        commission_bps=hypothesis.costs.commission_bps,
        slippage_bps=hypothesis.costs.slippage_bps,
    )


def _walk_forward_config(hypothesis: Hypothesis) -> WalkForwardConfig:
    method = hypothesis.walkforward.to_validation_method()
    return WalkForwardConfig(
        mode=method.mode,
        train_size=method.train_size,
        test_size=method.test_size,
        step_size=method.step_size,
        embargo_bars=method.embargo_bars,
        min_rows=method.min_rows,
    )


def _evaluate_slice(
    frame: pd.DataFrame,
    template: StrategyTemplate,
    params: dict[str, Any],
    cost_model: CostModel,
    direction: DirectionMode,
) -> VectorizedBacktestResult:
    positions = template.generate_positions(frame.reset_index(drop=True), params)
    direction_mode: DirectionMode = (
        direction if direction in {"long_only", "short_only"} else "long_short"
    )
    return evaluate_positions(
        frame.reset_index(drop=True),
        positions,
        cost_model=cost_model,
        direction=direction_mode,
    )


def _metric_float(row: dict[str, Any], key: str, fallback_key: str | None = None) -> float:
    value = row.get(key)
    if value is None and fallback_key is not None:
        value = row.get(fallback_key)
    if value is None:
        return 0.0
    return float(value)


def _metric_int(row: dict[str, Any], key: str, fallback_key: str | None = None) -> int:
    value = row.get(key)
    if value is None and fallback_key is not None:
        value = row.get(fallback_key)
    if value is None:
        return 0
    return int(value)


def _deduplicate_leakage_issues(issues: list[LeakageIssue]) -> list[LeakageIssue]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[LeakageIssue] = []
    for issue in issues:
        key = (issue.code, issue.message, issue.severity)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _run_grid(
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    parameter_sets: list[ParameterSet],
    hypothesis: Hypothesis,
) -> list[dict[str, Any]]:
    template = _template_for(hypothesis)
    cost_model = _cost_model(hypothesis)
    rows: list[dict[str, Any]] = []
    for parameter_set in parameter_sets:
        train_returns: list[float] = []
        train_gross_returns: list[float] = []
        train_trade_counts: list[int] = []
        train_drawdowns: list[float] = []
        test_returns: list[float] = []
        test_gross_returns: list[float] = []
        test_trade_counts: list[int] = []
        test_drawdowns: list[float] = []
        for split in splits:
            train = frame.iloc[split.train_start : split.train_end].copy()
            test = frame.iloc[split.test_start : split.test_end].copy()
            train_result = _evaluate_slice(
                train,
                template,
                parameter_set.params,
                cost_model,
                hypothesis.direction,
            )
            test_result = _evaluate_slice(
                test,
                template,
                parameter_set.params,
                cost_model,
                hypothesis.direction,
            )
            train_returns.append(train_result.net_return)
            train_gross_returns.append(train_result.gross_return)
            train_trade_counts.append(train_result.number_of_trades)
            train_drawdowns.append(train_result.max_drawdown)
            test_returns.append(test_result.net_return)
            test_gross_returns.append(test_result.gross_return)
            test_trade_counts.append(test_result.number_of_trades)
            test_drawdowns.append(test_result.max_drawdown)
        train_net_return = float(sum(train_returns) / len(train_returns))
        train_gross_return = float(sum(train_gross_returns) / len(train_gross_returns))
        train_profitable_split_pct = float(
            sum(value > 0 for value in train_returns) / len(train_returns)
        )
        train_trade_count = int(sum(train_trade_counts))
        train_max_drawdown = float(min(train_drawdowns)) if train_drawdowns else 0.0
        test_net_return = float(sum(test_returns) / len(test_returns))
        test_gross_return = float(sum(test_gross_returns) / len(test_gross_returns))
        test_profitable_split_pct = float(
            sum(value > 0 for value in test_returns) / len(test_returns)
        )
        test_trade_count = int(sum(test_trade_counts))
        test_max_drawdown = float(min(test_drawdowns)) if test_drawdowns else 0.0
        rows.append(
            {
                "parameter_set_id": parameter_set.parameter_set_id,
                "params": parameter_set.params,
                "train_gross_return": train_gross_return,
                "train_net_return": train_net_return,
                "train_profitable_split_pct": train_profitable_split_pct,
                "train_trade_count": train_trade_count,
                "train_max_drawdown": train_max_drawdown,
                "test_gross_return": test_gross_return,
                "test_net_return": test_net_return,
                "test_profitable_split_pct": test_profitable_split_pct,
                "test_trade_count": test_trade_count,
                "test_max_drawdown": test_max_drawdown,
                "profitable_split_pct": test_profitable_split_pct,
                "trade_count": test_trade_count,
                "max_drawdown": test_max_drawdown,
            }
        )
    return rows


def _experiment_id(hypothesis: Hypothesis, symbol: str, timeframe: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    return f"{stamp}_{hypothesis.id}_{symbol.upper()}_{timeframe}"


def _markdown(payload: dict[str, Any]) -> str:
    warnings = "\n".join(f"- {warning}" for warning in payload["warnings"]) or "- None"
    reasons = "\n".join(f"- {reason}" for reason in payload.get("classification_reasons", []))
    if not reasons:
        reasons = "- None"
    leakage_issues = payload.get("leakage_issues", [])
    leakage_lines = "\n".join(
        f"- `{issue['severity']}` `{issue['code']}`: {issue['message']}" for issue in leakage_issues
    )
    if not leakage_lines:
        leakage_lines = "- No leakage issues."
    selected_result = payload.get("selected_result", {})
    best_test_diagnostic = payload.get("best_test_diagnostic", {})
    selection = payload.get("selection", {})
    benchmark_summary = {
        "benchmark_results": payload.get("benchmark_results", {}),
        "selected_excess_vs_buy_and_hold": payload.get("selected_excess_vs_buy_and_hold", 0.0),
        "selected_excess_drawdown_vs_buy_and_hold": payload.get(
            "selected_excess_drawdown_vs_buy_and_hold", 0.0
        ),
        "benchmark_pass": payload.get("benchmark_pass", False),
    }
    return f"""# Research Experiment: {payload["experiment_id"]}

## Classification

`{payload["classification"]}`

## Classification Reasons

{reasons}

## Hypothesis

- Name: {payload["hypothesis"]["name"]}
- Template: `{payload["hypothesis"]["template"]}`
- Expected edge reason: {payload["hypothesis"]["expected_edge_reason"]}

## Data

- Symbol: `{payload["symbol"]}`
- Timeframe: `{payload["timeframe"]}`
- Date range: {payload["data"]["min_timestamp"]} to {payload["data"]["max_timestamp"]}
- Rows: {payload["data"]["row_count"]}

## Walk-Forward

- Splits: {len(payload["splits"])}
- Selected parameter set: `{selected_result.get("parameter_set_id", "none")}`
- Best-test diagnostic parameter set: `{best_test_diagnostic.get("parameter_set_id", "none")}`
- Median test return: {payload["stability"]["median_test_return"]}

## Parameter Selection

Selection uses train-side evidence only. Best-by-test is diagnostic only.

```json
{json.dumps(selection, indent=2)}
```

## Selected Result

```json
{json.dumps(selected_result, indent=2)}
```

## Best-Test Diagnostic

```json
{json.dumps(best_test_diagnostic, indent=2)}
```

## Benchmarks

```json
{json.dumps(benchmark_summary, indent=2)}
```

## Null Timing

```json
{json.dumps(payload.get("null_model_results", {}), indent=2)}
```

## Leakage Checks

{leakage_lines}

## Stability

```json
{json.dumps(payload["stability"], indent=2)}
```

## Regime Performance

```json
{json.dumps(payload["regime_performance"], indent=2)}
```

## Warnings

{warnings}
"""


def _load_index_payload(report_dir: Path) -> dict[str, Any]:
    index_json = report_dir / "index.json"
    if index_json.exists():
        payload = json.loads(index_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.setdefault("experiments", [])
            payload.setdefault("universe_runs", [])
            return payload
    return {"experiments": [], "universe_runs": []}


def _write_index(report_dir: Path, payload: dict[str, Any]) -> None:
    experiments = payload.get("experiments", [])
    universe_runs = payload.get("universe_runs", [])
    lines = [
        "# Research Experiments",
        "",
        "| Experiment | Hypothesis | Symbol | Timeframe | Classification | "
        "Net Return | Max DD | Trades | Stability |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in experiments:
        line_template = (
            "| {experiment_id} | {hypothesis_name} | {symbol} | {timeframe} | "
            "{classification} | {net_return:.6f} | {max_drawdown:.6f} | "
            "{trade_count} | {stability_score:.3f} |"
        )
        lines.append(line_template.format(**item))
    if universe_runs:
        lines.extend(
            [
                "",
                "# Universe Research Runs",
                "",
                "| Run | Hypothesis | Universe | Symbols | Candidates | Rejected | Report |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for item in universe_runs:
            lines.append(
                "| {run_id} | {hypothesis_id} | {universe_id} | {symbol_count} | "
                "{candidate_count} | {rejected_count} | {report_path} |".format(**item)
            )
    (report_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _update_index(report_dir: Path, entry: dict[str, Any]) -> None:
    payload = _load_index_payload(report_dir)
    payload["experiments"].append(entry)
    _write_index(report_dir, payload)


def _update_universe_index(report_dir: Path, entry: dict[str, Any]) -> None:
    payload = _load_index_payload(report_dir)
    payload["universe_runs"].append(entry)
    _write_index(report_dir, payload)


def run_research_experiment(
    *,
    hypothesis_path: Path,
    data_dir: str | Path = "data",
    symbol: str,
    timeframe: str,
    source: str = "manual",
    instrument_type: str = "stock",
    max_parameter_sets: int = 100,
    market_calendar: str | None = None,
) -> ExperimentRunResult:
    """Run one disciplined research experiment and save reports."""

    hypothesis = load_hypothesis(hypothesis_path)
    key = DatasetKey(
        source=source, instrument_type=instrument_type, symbol=symbol.upper(), timeframe=timeframe
    )
    raw_frame = load_dataset(key, data_dir=data_dir)
    initial_leakage_issues = check_timestamp_integrity(raw_frame)
    if "timestamp" in raw_frame:
        frame = raw_frame.sort_values("timestamp").reset_index(drop=True)
    else:
        frame = raw_frame.reset_index(drop=True)
    metadata = dataset_metadata(key, data_dir=data_dir)
    audit = create_audit_report(
        data_dir=data_dir,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        instrument_type=instrument_type,
        market_calendar=market_calendar,
    )
    warnings = [f"data issue: {issue.code}" for issue in audit.issues if issue.severity == "error"]
    splits = generate_walk_forward_splits(frame, _walk_forward_config(hypothesis))
    if not splits:
        warnings.append("insufficient data: no walk-forward splits generated")
    parameter_sets = ParameterGrid(
        parameter_space=hypothesis.parameter_space,
        maximum_parameter_sets=min(max_parameter_sets, hypothesis.maximum_parameter_sets),
    ).expand()
    grid_results = _run_grid(frame, splits, parameter_sets, hypothesis) if splits else []
    if grid_results:
        selection = select_parameter_set(
            grid_results,
            minimum_train_trades=hypothesis.minimum_evidence.min_trades,
            max_train_drawdown=-abs(float(hypothesis.risk_model.max_drawdown)),
            minimum_train_profitable_split_pct=0.0,
        )
        selected_result = dict(selection.selected_result)
        best_test_diagnostic = max(grid_results, key=lambda row: float(row["test_net_return"]))
        stability = analyze_stability(
            grid_results,
            best_parameter_set_id=selection.selected_parameter_set_id,
        )
    else:
        selection = select_parameter_set([])
        selected_result = {
            "parameter_set_id": "none",
            "params": parameter_sets[0].params if parameter_sets else {},
            "train_gross_return": 0.0,
            "train_net_return": 0.0,
            "train_profitable_split_pct": 0.0,
            "train_trade_count": 0,
            "train_max_drawdown": 0.0,
            "test_gross_return": 0.0,
            "test_net_return": 0.0,
            "test_profitable_split_pct": 0.0,
            "test_trade_count": 0,
            "test_max_drawdown": 0.0,
            "profitable_split_pct": 0.0,
            "trade_count": 0,
            "max_drawdown": 0.0,
        }
        selection = SelectionResult(
            selected_parameter_set_id="none",
            selected_result=selected_result,
            selection_method=selection.selection_method,
            rejected_parameter_set_ids=selection.rejected_parameter_set_ids,
            diagnostics=selection.diagnostics,
        )
        best_test_diagnostic = selected_result
        stability = StabilityReport(
            best_parameter_set_id="none",
            best_test_return=0.0,
            median_test_return=0.0,
            profitable_neighbour_pct=0.0,
            train_to_test_degradation=0.0,
            stability_score=0.0,
            isolated_warning=True,
        )
    if stability.isolated_warning:
        warnings.append("parameter stability warning: best setting is isolated")

    template = _template_for(hypothesis)
    selected_params = dict(selected_result.get("params", {}))
    selected_params["parameter_set_id"] = str(selected_result["parameter_set_id"])
    full_positions = template.generate_positions(frame, selected_params)
    full_result = evaluate_positions(frame, full_positions, cost_model=_cost_model(hypothesis))
    signals = template.generate_signals(frame, selected_params)
    leakage_issues = _deduplicate_leakage_issues(
        [
            *initial_leakage_issues,
            *collect_research_leakage_issues(
                frame=frame,
                splits=splits,
                signals=signals,
                embargo_bars=hypothesis.walkforward.embargo_bars,
            ),
        ]
    )
    for issue in leakage_issues:
        warnings.append(f"leakage {issue.severity}: {issue.code}")
    regimes = label_regimes(frame, window=min(20, max(3, len(frame) // 4)))
    regime_performance = performance_by_regime(
        pd.Series(full_result.net_returns),
        regimes["trend_regime"],
    )
    cost_model = _cost_model(hypothesis)
    benchmark_comparison = compare_with_benchmarks(
        frame,
        splits=splits,
        selected_result=selected_result,
        cost_model=cost_model,
    )
    null_model_results = run_null_timing_test(
        frame,
        full_positions,
        cost_model=cost_model,
        hypothesis_id=hypothesis.id,
        symbol=key.symbol,
        timeframe=key.timeframe,
        parameter_set_id=str(selected_result["parameter_set_id"]),
        selected_net_return=_metric_float(selected_result, "test_net_return"),
        null_count=7,
        direction=hypothesis.direction,
    )
    gross_test_return = _metric_float(
        selected_result,
        "test_gross_return",
        "test_net_return",
    )
    classification_result = classify_research_result(
        net_test_return=_metric_float(selected_result, "test_net_return"),
        gross_test_return=gross_test_return,
        trade_count=_metric_int(selected_result, "test_trade_count", "trade_count"),
        stability_score=stability.stability_score,
        profitable_split_pct=_metric_float(
            selected_result,
            "test_profitable_split_pct",
            "profitable_split_pct",
        ),
        max_drawdown=_metric_float(selected_result, "test_max_drawdown", "max_drawdown"),
        cost_drag=max(0.0, gross_test_return - _metric_float(selected_result, "test_net_return")),
        data_errors=sum(1 for warning in warnings if "data issue" in warning.lower()),
        leakage_errors=sum(1 for issue in leakage_issues if issue.severity == "error"),
        minimum_trades=hypothesis.minimum_evidence.min_trades,
        minimum_profitable_split_pct=hypothesis.minimum_evidence.min_profitable_split_pct,
        minimum_stability_score=hypothesis.minimum_evidence.min_stability_score,
        max_allowed_drawdown=-abs(float(hypothesis.risk_model.max_drawdown)),
        benchmark_pass=bool(benchmark_comparison["benchmark_pass"]),
        null_pass=bool(null_model_results["null_pass"]),
        selection_rejected=selection.selection_method == "fallback_for_reporting_only",
    )
    classification: Classification = classification_result.classification
    classification_reasons = list(classification_result.reasons)
    if any(issue.severity == "error" for issue in leakage_issues) and (
        "leakage_errors" not in classification_reasons
    ):
        classification_reasons.append("leakage_errors")
    if any("insufficient data" in warning.lower() for warning in warnings):
        classification = "rejected_insufficient_data"
        classification_reasons.insert(0, "insufficient_data")

    experiment_id = _experiment_id(hypothesis, symbol, timeframe)
    report_dir = Path(data_dir).expanduser() / "reports" / "research"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{experiment_id}.json"
    markdown_path = report_dir / f"{experiment_id}.md"
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "classification": classification,
        "hypothesis": hypothesis.model_dump(mode="json"),
        "symbol": key.symbol,
        "timeframe": key.timeframe,
        "source": key.source,
        "instrument_type": key.instrument_type,
        "data": metadata.to_dict(),
        "cost_assumptions": hypothesis.costs.model_dump(),
        "splits": [split.model_dump() for split in splits],
        "selection": selection.model_dump(mode="json"),
        "selected_result": selected_result,
        "best_test_diagnostic": best_test_diagnostic,
        "grid_results": grid_results,
        "best_result": selected_result,
        "worst_result": min(grid_results, key=lambda row: float(row["test_net_return"]))
        if grid_results
        else selected_result,
        "stability": stability.to_dict(),
        "regime_performance": regime_performance,
        "leakage_issues": [issue.to_dict() for issue in leakage_issues],
        "benchmark_results": benchmark_comparison["benchmark_results"],
        "selected_excess_vs_buy_and_hold": benchmark_comparison["selected_excess_vs_buy_and_hold"],
        "selected_excess_drawdown_vs_buy_and_hold": benchmark_comparison[
            "selected_excess_drawdown_vs_buy_and_hold"
        ],
        "benchmark_pass": benchmark_comparison["benchmark_pass"],
        "null_model_results": null_model_results,
        "warnings": warnings,
        "classification_reasons": classification_reasons,
        "full_sample_result": full_result.to_dict(),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    _update_index(
        report_dir,
        {
            "experiment_id": experiment_id,
            "hypothesis_name": hypothesis.name,
            "hypothesis_id": hypothesis.id,
            "symbol": key.symbol,
            "timeframe": key.timeframe,
            "date_range": f"{metadata.min_timestamp} to {metadata.max_timestamp}",
            "classification": classification,
            "classification_reasons": classification_reasons,
            "net_return": _metric_float(selected_result, "test_net_return"),
            "max_drawdown": _metric_float(selected_result, "test_max_drawdown", "max_drawdown"),
            "trade_count": _metric_int(selected_result, "test_trade_count", "trade_count"),
            "stability_score": stability.stability_score,
            "benchmark_pass": benchmark_comparison["benchmark_pass"],
            "null_pass": null_model_results["null_pass"],
            "selected_excess_vs_buy_and_hold": benchmark_comparison[
                "selected_excess_vs_buy_and_hold"
            ],
            "selected_excess_vs_p75_null": null_model_results["selected_excess_vs_p75_null"],
            "report_path": str(markdown_path),
        },
    )
    return ExperimentRunResult(
        experiment_id=experiment_id,
        classification=classification,
        markdown_path=markdown_path,
        json_path=json_path,
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _universe_run_id(hypothesis: Hypothesis, universe_id: str, timeframe: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    return f"{stamp}_{hypothesis.id}_{universe_id}_{timeframe}"


def _median(values: list[float]) -> float:
    clean = sorted(values)
    if not clean:
        return 0.0
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return (clean[midpoint - 1] + clean[midpoint]) / 2


def _existing_experiment_entry(
    *,
    report_root: Path,
    hypothesis_id: str,
    symbol: str,
    timeframe: str,
) -> dict[str, Any] | None:
    payload = _load_index_payload(report_root)
    matches = [
        item
        for item in payload.get("experiments", [])
        if item.get("hypothesis_id") == hypothesis_id
        and item.get("symbol") == symbol
        and item.get("timeframe") == timeframe
    ]
    return matches[-1] if matches else None


def _universe_markdown(payload: dict[str, Any]) -> str:
    rows = [
        [
            item["symbol"],
            item["status"],
            item.get("classification", ""),
            item.get("net_return", ""),
            item.get("trade_count", ""),
            item.get("benchmark_pass", ""),
            item.get("null_pass", ""),
            item.get("error_message", ""),
        ]
        for item in payload["symbol_results"]
    ]
    table = "\n".join(
        [
            "| Symbol | Status | Classification | Net Return | Trades | "
            "Benchmark | Null | Message |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )
    return f"""# Universe Research Run: {payload["run_id"]}

## Summary

- Hypothesis: `{payload["hypothesis_id"]}`
- Universe: `{payload["universe_id"]}`
- Symbols tested: {payload["symbol_count"]}
- Failed: {payload["failed_count"]}
- Candidates: {payload["candidate_count"]}
- Rejected: {payload["rejected_count"]}

## Classification Counts

```json
{json.dumps(payload["classification_counts"], indent=2, sort_keys=True)}
```

## Classification Reason Counts

```json
{json.dumps(payload["classification_reason_counts"], indent=2, sort_keys=True)}
```

## Benchmark And Null Summary

- Benchmark pass count: {payload["benchmark_pass_count"]}
- Null pass count: {payload["null_pass_count"]}
- Median excess vs benchmark: {payload["median_excess_vs_benchmark"]}
- Median excess vs null: {payload["median_excess_vs_null"]}

## Symbols

{table}
"""


def run_universe_research(
    *,
    hypothesis_path: Path,
    qualified_universe_path: Path,
    data_dir: str | Path = "data",
    source: str | None = None,
    timeframe: str | None = None,
    instrument_type: str = "stock",
    max_symbols: int | None = None,
    fail_fast: bool = False,
    resume: bool = False,
    skip_existing: bool = False,
    market_calendar: str | None = None,
) -> UniverseResearchRunResult:
    """Run one written hypothesis across a research-ready universe export."""

    hypothesis = load_hypothesis(hypothesis_path)
    universe_payload = _load_json(qualified_universe_path)
    universe_id = str(universe_payload.get("universe_id", qualified_universe_path.stem))
    resolved_source = source or str(universe_payload.get("source", hypothesis.data_source))
    resolved_timeframe = timeframe or str(universe_payload.get("timeframe", hypothesis.timeframe))
    symbols = load_research_ready_universe(qualified_universe_path)
    if max_symbols is not None:
        symbols = symbols[:max_symbols]

    report_root = Path(data_dir).expanduser() / "reports" / "research"
    report_dir = report_root / "universe"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = _universe_run_id(hypothesis, universe_id, resolved_timeframe)
    json_path = report_dir / f"{run_id}.json"
    markdown_path = report_dir / f"{run_id}.md"

    symbol_results: list[dict[str, Any]] = []
    for symbol in symbols:
        existing = (
            _existing_experiment_entry(
                report_root=report_root,
                hypothesis_id=hypothesis.id,
                symbol=symbol,
                timeframe=resolved_timeframe,
            )
            if resume or skip_existing
            else None
        )
        if existing is not None:
            symbol_results.append(
                {
                    "symbol": symbol,
                    "status": "skipped",
                    "skip_reason": "existing_index_entry",
                    "classification": existing.get("classification", "rejected_data_issue"),
                    "classification_reasons": existing.get("classification_reasons", []),
                    "report_path": existing.get("report_path"),
                    "net_return": float(existing.get("net_return", 0.0)),
                    "max_drawdown": float(existing.get("max_drawdown", 0.0)),
                    "trade_count": int(existing.get("trade_count", 0)),
                    "stability_score": float(existing.get("stability_score", 0.0)),
                    "benchmark_pass": bool(existing.get("benchmark_pass", False)),
                    "null_pass": bool(existing.get("null_pass", False)),
                    "selected_excess_vs_buy_and_hold": float(
                        existing.get("selected_excess_vs_buy_and_hold", 0.0)
                    ),
                    "selected_excess_vs_p75_null": float(
                        existing.get("selected_excess_vs_p75_null", 0.0)
                    ),
                }
            )
            continue
        try:
            result = run_research_experiment(
                hypothesis_path=hypothesis_path,
                data_dir=data_dir,
                symbol=symbol,
                timeframe=resolved_timeframe,
                source=resolved_source,
                instrument_type=instrument_type,
                max_parameter_sets=hypothesis.maximum_parameter_sets,
                market_calendar=market_calendar,
            )
            experiment_payload = _load_json(result.json_path)
            selected_result = experiment_payload.get("selected_result", {})
            stability = experiment_payload.get("stability", {})
            null_model_results = experiment_payload.get("null_model_results", {})
            symbol_results.append(
                {
                    "symbol": symbol,
                    "status": "completed",
                    "classification": result.classification,
                    "classification_reasons": experiment_payload.get("classification_reasons", []),
                    "experiment_id": result.experiment_id,
                    "report_path": str(result.markdown_path),
                    "json_path": str(result.json_path),
                    "net_return": float(selected_result.get("test_net_return", 0.0)),
                    "max_drawdown": float(
                        selected_result.get(
                            "test_max_drawdown",
                            selected_result.get("max_drawdown", 0.0),
                        )
                    ),
                    "trade_count": int(
                        selected_result.get(
                            "test_trade_count",
                            selected_result.get("trade_count", 0),
                        )
                    ),
                    "stability_score": float(stability.get("stability_score", 0.0)),
                    "benchmark_pass": bool(experiment_payload.get("benchmark_pass", False)),
                    "null_pass": bool(null_model_results.get("null_pass", False)),
                    "selected_excess_vs_buy_and_hold": float(
                        experiment_payload.get("selected_excess_vs_buy_and_hold", 0.0)
                    ),
                    "selected_excess_vs_p75_null": float(
                        null_model_results.get("selected_excess_vs_p75_null", 0.0)
                    ),
                }
            )
        except Exception as exc:
            symbol_results.append(
                {
                    "symbol": symbol,
                    "status": "failed",
                    "classification": "rejected_data_issue",
                    "error_message": str(exc),
                    "net_return": 0.0,
                    "max_drawdown": 0.0,
                    "trade_count": 0,
                    "stability_score": 0.0,
                    "classification_reasons": ["symbol_failed"],
                    "benchmark_pass": False,
                    "null_pass": False,
                    "selected_excess_vs_buy_and_hold": 0.0,
                    "selected_excess_vs_p75_null": 0.0,
                }
            )
            if fail_fast:
                break

    classifications = [
        str(item["classification"])
        for item in symbol_results
        if item.get("status") in {"completed", "skipped"} and item.get("classification")
    ]
    aggregate_items = [
        item for item in symbol_results if item.get("status") in {"completed", "skipped"}
    ]
    classification_reasons = [
        str(reason) for item in aggregate_items for reason in item.get("classification_reasons", [])
    ]
    classification_counts = dict(Counter(classifications))
    classification_reason_counts = dict(Counter(classification_reasons))
    failed_count = sum(1 for item in symbol_results if item["status"] == "failed")
    candidate_count = classification_counts.get("candidate_paper_test", 0)
    rejected_count = sum(
        count
        for classification_name, count in classification_counts.items()
        if classification_name.startswith("rejected_")
    )
    payload: dict[str, Any] = {
        "run_id": run_id,
        "hypothesis_id": hypothesis.id,
        "hypothesis_name": hypothesis.name,
        "universe_id": universe_id,
        "qualified_universe_path": str(qualified_universe_path),
        "timeframe": resolved_timeframe,
        "source": resolved_source,
        "symbol_count": len(symbols),
        "failed_count": failed_count,
        "candidate_count": candidate_count,
        "rejected_count": rejected_count,
        "classification_counts": classification_counts,
        "classification_reason_counts": classification_reason_counts,
        "benchmark_pass_count": sum(1 for item in aggregate_items if item.get("benchmark_pass")),
        "null_pass_count": sum(1 for item in aggregate_items if item.get("null_pass")),
        "median_net_return": _median([float(item["net_return"]) for item in aggregate_items]),
        "median_excess_vs_benchmark": _median(
            [float(item["selected_excess_vs_buy_and_hold"]) for item in aggregate_items]
        ),
        "median_excess_vs_null": _median(
            [float(item["selected_excess_vs_p75_null"]) for item in aggregate_items]
        ),
        "median_max_drawdown": _median([float(item["max_drawdown"]) for item in aggregate_items]),
        "median_trade_count": _median([float(item["trade_count"]) for item in aggregate_items]),
        "median_stability_score": _median(
            [float(item["stability_score"]) for item in aggregate_items]
        ),
        "top_candidates": [
            item for item in symbol_results if item.get("classification") == "candidate_paper_test"
        ][:10],
        "top_rejection_reasons": dict(Counter(classification_reasons).most_common(10)),
        "symbol_results": symbol_results,
        "created_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    markdown_path.write_text(_universe_markdown(payload), encoding="utf-8")
    _update_universe_index(
        report_root,
        {
            "run_id": run_id,
            "hypothesis_id": hypothesis.id,
            "universe_id": universe_id,
            "symbol_count": len(symbols),
            "candidate_count": candidate_count,
            "rejected_count": rejected_count,
            "report_path": str(markdown_path),
        },
    )
    return UniverseResearchRunResult(
        run_id=run_id,
        markdown_path=markdown_path,
        json_path=json_path,
        classification_counts=classification_counts,
        failed_count=failed_count,
    )
