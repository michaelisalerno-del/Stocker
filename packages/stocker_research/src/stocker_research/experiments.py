"""Research experiment runner and conservative result classification."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_data.audit import create_audit_report
from stocker_data.storage import DatasetKey, dataset_metadata, load_dataset
from stocker_data.universe import load_research_ready_universe
from stocker_research.benchmarks import compare_with_benchmarks
from stocker_research.classification import ResearchClassification, classify_research_result
from stocker_research.holding_policy import (
    analyze_holding_policy,
    build_holding_policy_decision,
)
from stocker_research.hypothesis import Hypothesis, load_hypothesis
from stocker_research.leakage import (
    LeakageIssue,
    check_timestamp_integrity,
    collect_research_leakage_issues,
)
from stocker_research.null_models import run_null_timing_test_for_splits
from stocker_research.parameters import ParameterGrid, ParameterSet
from stocker_research.position_policy import (
    PositionPolicyResult,
    apply_holding_policy_to_positions,
    summarize_position_policy_effect,
)
from stocker_research.regime import label_regimes, performance_by_regime
from stocker_research.robustness import (
    RobustnessGatePolicy,
    build_intraday_robustness_diagnostics,
)
from stocker_research.selection import SelectionResult, select_parameter_set
from stocker_research.stability import StabilityReport, analyze_stability
from stocker_research.templates import StrategyTemplate, get_template
from stocker_research.walkforward import (
    WalkForwardConfig,
    WalkForwardSplit,
    generate_walk_forward_splits,
)
from stocker_research.windows import (
    EVALUATION_POLICY_WITH_INDICATOR_CONTEXT,
    GRID_CONTEXT_POLICY_WITH_INDICATOR_CONTEXT,
    INDICATOR_CONTEXT_POLICY,
    build_evaluation_window,
    evaluate_window_with_context,
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
        "rejected_overnight_risk",
        "rejected_weekend_risk",
        "rejected_holding_policy_violation",
        "interesting_needs_more_tests",
        "interesting_intraday_needs_more_tests",
        "interesting_swing_needs_more_tests",
        "candidate_intraday_test",
        "candidate_swing_exceptional",
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


def _safe_required_lookback(template: StrategyTemplate, row: dict[str, Any]) -> int:
    value = row.get("required_lookback_bars")
    if value is not None:
        return max(0, int(value))
    params = row.get("params", {})
    if not isinstance(params, dict):
        return 0
    try:
        return max(0, int(template.required_lookback_bars(params)))
    except (KeyError, TypeError, ValueError):
        return 0


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
    *,
    timeframe: str | None = None,
    market_calendar: str | None = None,
) -> list[dict[str, Any]]:
    template = _template_for(hypothesis)
    cost_model = _cost_model(hypothesis)
    rows: list[dict[str, Any]] = []
    for parameter_set in parameter_sets:
        required_lookback_bars = max(
            0,
            int(template.required_lookback_bars(parameter_set.params)),
        )
        train_returns: list[float] = []
        train_gross_returns: list[float] = []
        train_trade_counts: list[int] = []
        train_drawdowns: list[float] = []
        train_context_rows: list[int] = []
        test_returns: list[float] = []
        test_gross_returns: list[float] = []
        test_trade_counts: list[int] = []
        test_drawdowns: list[float] = []
        test_context_rows: list[int] = []
        train_position_policy_results: list[PositionPolicyResult] = []
        test_position_policy_results: list[PositionPolicyResult] = []
        for split in splits:
            train_evaluation = evaluate_window_with_context(
                frame,
                template,
                parameter_set.params,
                cost_model=cost_model,
                direction=hypothesis.direction,
                eval_start=split.train_start,
                eval_end=split.train_end,
                holding_policy=hypothesis.holding_policy,
                timeframe=timeframe or hypothesis.timeframe,
                market_calendar=market_calendar,
            )
            test_evaluation = evaluate_window_with_context(
                frame,
                template,
                parameter_set.params,
                cost_model=cost_model,
                direction=hypothesis.direction,
                eval_start=split.test_start,
                eval_end=split.test_end,
                holding_policy=hypothesis.holding_policy,
                timeframe=timeframe or hypothesis.timeframe,
                market_calendar=market_calendar,
            )
            train_result = train_evaluation.result
            test_result = test_evaluation.result
            train_returns.append(train_result.net_return)
            train_gross_returns.append(train_result.gross_return)
            train_trade_counts.append(train_result.number_of_trades)
            train_drawdowns.append(train_result.max_drawdown)
            train_context_rows.append(train_evaluation.window.context_rows_used)
            if train_evaluation.position_policy is not None:
                train_position_policy_results.append(train_evaluation.position_policy)
            test_returns.append(test_result.net_return)
            test_gross_returns.append(test_result.gross_return)
            test_trade_counts.append(test_result.number_of_trades)
            test_drawdowns.append(test_result.max_drawdown)
            test_context_rows.append(test_evaluation.window.context_rows_used)
            if test_evaluation.position_policy is not None:
                test_position_policy_results.append(test_evaluation.position_policy)
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
                "required_lookback_bars": required_lookback_bars,
                "train_context_rows_used": int(max(train_context_rows, default=0)),
                "test_context_rows_used": int(max(test_context_rows, default=0)),
                "context_policy": GRID_CONTEXT_POLICY_WITH_INDICATOR_CONTEXT,
                "profitable_split_pct": test_profitable_split_pct,
                "trade_count": test_trade_count,
                "max_drawdown": test_max_drawdown,
                "train_position_policy": summarize_position_policy_effect(
                    train_position_policy_results
                ),
                "position_policy": summarize_position_policy_effect(
                    test_position_policy_results
                ),
            }
        )
    return rows


def _selected_test_window_positions(
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    template: StrategyTemplate,
    params: dict[str, Any],
    *,
    holding_policy: Any | None = None,
    timeframe: str | None = None,
    market_calendar: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    position_policy_results: list[PositionPolicyResult] = []
    if not splits:
        reset_frame = frame.reset_index(drop=True)
        window = build_evaluation_window(
            frame,
            template,
            params,
            eval_start=0,
            eval_end=len(frame),
            holding_policy=holding_policy,
            timeframe=timeframe,
            market_calendar=market_calendar,
        )
        if holding_policy is not None and timeframe is not None:
            policy_result = apply_holding_policy_to_positions(
                window.eval_frame,
                window.raw_eval_positions,
                policy=holding_policy,
                timeframe=timeframe,
                market_calendar=market_calendar,
            )
            position_policy_results.append(policy_result)
        return (
            reset_frame,
            window.raw_eval_positions,
            window.eval_positions,
            pd.Series(["full_sample"] * len(reset_frame)),
            summarize_position_policy_effect(position_policy_results),
        )

    frames: list[pd.DataFrame] = []
    raw_position_series: list[pd.Series] = []
    scored_position_series: list[pd.Series] = []
    window_ids: list[str] = []
    for split in splits:
        if split.test_end <= split.test_start:
            continue
        window = build_evaluation_window(
            frame,
            template,
            params,
            eval_start=split.test_start,
            eval_end=split.test_end,
            holding_policy=holding_policy,
            timeframe=timeframe,
            market_calendar=market_calendar,
        )
        frames.append(window.eval_frame)
        raw_position_series.append(window.raw_eval_positions)
        scored_position_series.append(window.eval_positions)
        window_ids.extend([split.split_id] * len(window.eval_frame))
        if holding_policy is not None and timeframe is not None:
            policy_result = apply_holding_policy_to_positions(
                window.eval_frame,
                window.raw_eval_positions,
                policy=holding_policy,
                timeframe=timeframe,
                market_calendar=market_calendar,
            )
            position_policy_results.append(policy_result)
    if not frames:
        empty = frame.iloc[0:0].reset_index(drop=True)
        return (
            empty,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.Series(dtype=str),
            summarize_position_policy_effect([]),
        )
    return (
        pd.concat(frames, ignore_index=True),
        pd.concat(raw_position_series, ignore_index=True),
        pd.concat(scored_position_series, ignore_index=True),
        pd.Series(window_ids),
        summarize_position_policy_effect(position_policy_results),
    )


def _robustness_gate_policy(hypothesis: Hypothesis) -> RobustnessGatePolicy:
    return RobustnessGatePolicy(**hypothesis.robustness_policy.model_dump())


def _multiply_cost_model(cost_model: CostModel, multiplier: float) -> CostModel:
    return CostModel(
        spread_bps=cost_model.spread_bps * multiplier,
        commission_bps=cost_model.commission_bps * multiplier,
        slippage_bps=cost_model.slippage_bps * multiplier,
    )


def _reconstruct_round_trip_returns(
    positions: pd.Series,
    net_returns: Sequence[float],
) -> list[float]:
    position_values = positions.reset_index(drop=True).astype(float).tolist()
    returns = [float(value) for value in net_returns]
    trade_returns: list[float] = []
    active = False
    running_return = 0.0
    previous_position = 0.0

    for index, current_position in enumerate(position_values):
        bar_return = returns[index] if index < len(returns) else 0.0
        opens_trade = not active and abs(previous_position) == 0.0 and abs(current_position) > 0.0
        flips_trade = active and previous_position * current_position < 0.0
        closes_trade = active and abs(previous_position) > 0.0 and abs(current_position) == 0.0

        if opens_trade:
            active = True
            running_return = bar_return
        elif active:
            running_return += bar_return

        if flips_trade:
            trade_returns.append(running_return)
            running_return = bar_return
            active = abs(current_position) > 0.0
        elif closes_trade:
            trade_returns.append(running_return)
            running_return = 0.0
            active = False

        previous_position = current_position

    if active:
        trade_returns.append(running_return)
    return trade_returns


def _selected_intraday_robustness_diagnostics(
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    template: StrategyTemplate,
    selected_params: dict[str, Any],
    hypothesis: Hypothesis,
    cost_model: CostModel,
    *,
    timeframe: str,
    market_calendar: str | None,
    symbol: str,
    benchmark_pass: bool,
    null_pass: bool,
    selected_net_return: float,
    stability_score: float,
    train_selection_succeeded: bool,
    session_flat_compliant: bool,
    minimum_trades: int,
    session_quality_warning: bool,
) -> dict[str, Any]:
    policy = _robustness_gate_policy(hypothesis)
    if not splits:
        return build_intraday_robustness_diagnostics(
            cost_stress_rows=None,
            trade_returns=None,
            split_rows=None,
            policy=policy,
            symbol=symbol,
            benchmark_pass=benchmark_pass,
            null_pass=null_pass,
            net_return=selected_net_return,
            stability_score=stability_score,
            train_selection_succeeded=train_selection_succeeded,
            session_flat_compliant=session_flat_compliant,
            trade_count=0,
            min_trades=minimum_trades,
            session_quality_warning=session_quality_warning,
        )

    cost_stress_rows: list[dict[str, float]] = []
    split_rows: list[dict[str, float | str]] = []
    trade_returns: list[float] = []
    for multiplier in (1.0, 1.5, 2.0, 3.0):
        split_net_returns: list[float] = []
        stressed_cost_model = _multiply_cost_model(cost_model, multiplier)
        for split in splits:
            evaluation = evaluate_window_with_context(
                frame,
                template,
                selected_params,
                cost_model=stressed_cost_model,
                direction=hypothesis.direction,
                eval_start=split.test_start,
                eval_end=split.test_end,
                holding_policy=hypothesis.holding_policy,
                timeframe=timeframe,
                market_calendar=market_calendar,
            )
            split_net_returns.append(float(evaluation.result.net_return))
            if math.isclose(multiplier, 1.0):
                split_rows.append(
                    {
                        "split_id": split.split_id,
                        "test_net_return": float(evaluation.result.net_return),
                    }
                )
                trade_returns.extend(
                    _reconstruct_round_trip_returns(
                        evaluation.window.eval_positions,
                        evaluation.result.net_returns,
                    )
                )
        cost_stress_rows.append(
            {
                "cost_multiplier": multiplier,
                "net_return": float(sum(split_net_returns) / len(split_net_returns))
                if split_net_returns
                else 0.0,
            }
        )

    return build_intraday_robustness_diagnostics(
        cost_stress_rows=cost_stress_rows,
        trade_returns=trade_returns,
        split_rows=split_rows,
        policy=policy,
        symbol=symbol,
        benchmark_pass=benchmark_pass,
        null_pass=null_pass,
        net_return=selected_net_return,
        stability_score=stability_score,
        train_selection_succeeded=train_selection_succeeded,
        session_flat_compliant=session_flat_compliant,
        trade_count=len(trade_returns),
        min_trades=minimum_trades,
        session_quality_warning=session_quality_warning,
    )


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
        "benchmark_policy": payload.get("benchmark_policy", ""),
        "strategy_direction": payload.get("strategy_direction", ""),
    }
    holding_summary = {
        "holding_policy": payload.get("holding_policy", {}),
        "position_policy": payload.get("position_policy", {}),
        "raw_template_holding_policy_analysis": payload.get(
            "raw_template_holding_policy_analysis", {}
        ),
        "scored_holding_policy_analysis": payload.get("scored_holding_policy_analysis", {}),
        "holding_policy_analysis": payload.get("holding_policy_analysis", {}),
        "holding_policy_decision": payload.get("holding_policy_decision", {}),
    }
    return f"""# Research Experiment: {payload["experiment_id"]}

## Classification

`{payload["classification"]}`

Classification is conservative. Rejection is the common and expected research outcome.

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

## Evaluation Policy

- Policy: `{payload.get("evaluation_policy", "")}`
- Indicator context: `{payload.get("indicator_context_policy", "")}`
- Context rows are scored: `{payload.get("context_rows_are_scored", False)}`
- Max required lookback bars: {payload.get("max_required_lookback_bars", 0)}

Historical rows before each train/test window may be used only to warm up indicators.
Metrics, trades, drawdown, exposure, returns, and costs are scored only inside the
actual train/test rows. Future rows after the evaluation window are not used. This
avoids false rejection of rolling indicators while keeping rejection the expected
research outcome.

```json
{json.dumps(payload.get("context_summary", {}), indent=2)}
```

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

Benchmarks are measured over the same walk-forward test windows as the selected
result. Buy-and-hold is a long market baseline for every hypothesis direction.

```json
{json.dumps(benchmark_summary, indent=2)}
```

## Holding Policy

The default preference is intraday and session-flat. Swing results are research
vehicles unless they pass stricter evidence gates. Daily data cannot prove
session-flat tradability; overnight, weekend, and gap contribution are reported
separately where measurable.

For intraday/session-flat hypotheses, raw template target positions remain visible
for diagnostics, but scored positions are adjusted by the research-side position
policy before returns, costs, holding analysis, and classification are computed.

```json
{json.dumps(holding_summary, indent=2)}
```

## Robustness Diagnostics

Intraday candidate promotion requires modest cost-stress survival, positive
median reconstructed trade return, acceptable split and winner concentration,
and minimum profit factor. These diagnostics harden classification only; they do
not add strategy templates or fetch data.

```json
{json.dumps(payload.get("robustness_diagnostics", {}), indent=2)}
```

## Null Timing

Null timing checks use deterministic circular shifts over the same walk-forward test
windows and indicator-context policy as the selected result.

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
    grid_results = (
        _run_grid(
            frame,
            splits,
            parameter_sets,
            hypothesis,
            timeframe=key.timeframe,
            market_calendar=market_calendar,
        )
        if splits
        else []
    )
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
    selected_required_lookback = _safe_required_lookback(template, selected_result)
    selected_result.setdefault("required_lookback_bars", selected_required_lookback)
    selected_result.setdefault("train_context_rows_used", 0)
    selected_result.setdefault("test_context_rows_used", 0)
    selected_result.setdefault("context_policy", GRID_CONTEXT_POLICY_WITH_INDICATOR_CONTEXT)
    cost_model = _cost_model(hypothesis)
    full_evaluation = evaluate_window_with_context(
        frame,
        template,
        selected_params,
        cost_model=cost_model,
        direction=hypothesis.direction,
        eval_start=0,
        eval_end=len(frame),
        holding_policy=hypothesis.holding_policy,
        timeframe=key.timeframe,
        market_calendar=market_calendar,
    )
    full_result = full_evaluation.result
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
    benchmark_comparison = compare_with_benchmarks(
        frame,
        splits=splits,
        selected_result=selected_result,
        cost_model=cost_model,
        direction=hypothesis.direction,
    )
    null_model_results = run_null_timing_test_for_splits(
        frame,
        splits=splits,
        template=template,
        selected_params=selected_params,
        cost_model=cost_model,
        hypothesis_id=hypothesis.id,
        symbol=key.symbol,
        timeframe=key.timeframe,
        parameter_set_id=str(selected_result["parameter_set_id"]),
        selected_net_return=_metric_float(selected_result, "test_net_return"),
        null_count=7,
        direction=hypothesis.direction,
        holding_policy=hypothesis.holding_policy,
        market_calendar=market_calendar,
    )
    (
        holding_frame,
        raw_holding_positions,
        scored_holding_positions,
        holding_window_ids,
        selected_position_policy,
    ) = _selected_test_window_positions(
        frame,
        splits,
        template,
        selected_params,
        holding_policy=hypothesis.holding_policy,
        timeframe=key.timeframe,
        market_calendar=market_calendar,
    )
    raw_template_holding_policy_analysis = analyze_holding_policy(
        holding_frame,
        raw_holding_positions,
        result=None,
        selected_net_return=_metric_float(selected_result, "test_net_return"),
        timeframe=key.timeframe,
        policy=hypothesis.holding_policy,
        window_ids=holding_window_ids,
        market_calendar=market_calendar,
    )
    scored_holding_policy_analysis = analyze_holding_policy(
        holding_frame,
        scored_holding_positions,
        result=None,
        selected_net_return=_metric_float(selected_result, "test_net_return"),
        timeframe=key.timeframe,
        policy=hypothesis.holding_policy,
        window_ids=holding_window_ids,
        market_calendar=market_calendar,
    )
    holding_policy_analysis = scored_holding_policy_analysis
    holding_policy_decision = build_holding_policy_decision(
        holding_policy_analysis,
        hypothesis.holding_policy,
        selected_excess_vs_benchmark=float(benchmark_comparison["selected_excess_vs_buy_and_hold"]),
        selected_excess_vs_null=float(null_model_results["selected_excess_vs_p75_null"]),
        trade_count=_metric_int(selected_result, "test_trade_count", "trade_count"),
        max_drawdown=_metric_float(selected_result, "test_max_drawdown", "max_drawdown"),
    )
    robustness_policy = _robustness_gate_policy(hypothesis)
    robustness_diagnostics = _selected_intraday_robustness_diagnostics(
        frame,
        splits,
        template,
        selected_params,
        hypothesis,
        cost_model,
        timeframe=key.timeframe,
        market_calendar=market_calendar,
        symbol=key.symbol,
        benchmark_pass=bool(benchmark_comparison["benchmark_pass"]),
        null_pass=bool(null_model_results["null_pass"]),
        selected_net_return=_metric_float(selected_result, "test_net_return"),
        stability_score=stability.stability_score,
        train_selection_succeeded=selection.selection_method != "fallback_for_reporting_only",
        session_flat_compliant=holding_policy_analysis.session_flat_compliant,
        minimum_trades=hypothesis.minimum_evidence.min_trades,
        session_quality_warning=bool(holding_policy_analysis.holding_policy_warning_reasons),
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
        holding_policy_classification=holding_policy_decision.classification,
        holding_policy_reasons=holding_policy_decision.reasons,
        intraday_robustness_diagnostics=robustness_diagnostics,
        intraday_robustness_policy=robustness_policy,
    )
    classification: Classification = classification_result.classification
    classification_reasons = list(classification_result.reasons)
    for reason in holding_policy_decision.reasons:
        if reason not in classification_reasons:
            classification_reasons.append(reason)
    if any(issue.severity == "error" for issue in leakage_issues) and (
        "leakage_errors" not in classification_reasons
    ):
        classification_reasons.append("leakage_errors")
    if any("insufficient data" in warning.lower() for warning in warnings):
        classification = "rejected_insufficient_data"
        classification_reasons.insert(0, "insufficient_data")

    required_lookback_bars_by_parameter_set = {
        str(row.get("parameter_set_id", "unknown")): _safe_required_lookback(template, row)
        for row in grid_results
    }
    selected_parameter_set_id = str(selected_result["parameter_set_id"])
    required_lookback_bars_by_parameter_set.setdefault(
        selected_parameter_set_id,
        selected_required_lookback,
    )
    max_required_lookback_bars = max(
        required_lookback_bars_by_parameter_set.values(),
        default=selected_required_lookback,
    )
    context_summary = {
        "context_policy": GRID_CONTEXT_POLICY_WITH_INDICATOR_CONTEXT,
        "indicator_context_policy": INDICATOR_CONTEXT_POLICY,
        "context_rows_are_scored": False,
        "max_train_context_rows_used": int(
            max(
                (int(row.get("train_context_rows_used", 0) or 0) for row in grid_results),
                default=0,
            )
        ),
        "max_test_context_rows_used": int(
            max(
                (int(row.get("test_context_rows_used", 0) or 0) for row in grid_results),
                default=0,
            )
        ),
        "selected_parameter_set_id": selected_parameter_set_id,
        "selected_required_lookback_bars": selected_required_lookback,
        "selected_train_context_rows_used": int(
            selected_result.get("train_context_rows_used", 0) or 0
        ),
        "selected_test_context_rows_used": int(
            selected_result.get("test_context_rows_used", 0) or 0
        ),
    }

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
        "evaluation_policy": EVALUATION_POLICY_WITH_INDICATOR_CONTEXT,
        "indicator_context_policy": INDICATOR_CONTEXT_POLICY,
        "context_rows_are_scored": False,
        "max_required_lookback_bars": max_required_lookback_bars,
        "required_lookback_bars_by_parameter_set": required_lookback_bars_by_parameter_set,
        "context_summary": context_summary,
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
        "benchmark_policy": benchmark_comparison["benchmark_policy"],
        "strategy_direction": benchmark_comparison["strategy_direction"],
        "null_model_results": null_model_results,
        "holding_policy": hypothesis.holding_policy.model_dump(mode="json"),
        "position_policy": selected_position_policy,
        "raw_template_holding_policy_analysis": raw_template_holding_policy_analysis.model_dump(
            mode="json"
        ),
        "scored_holding_policy_analysis": scored_holding_policy_analysis.model_dump(mode="json"),
        "holding_policy_analysis": holding_policy_analysis.model_dump(mode="json"),
        "holding_policy_decision": holding_policy_decision.model_dump(mode="json"),
        "robustness_policy": hypothesis.robustness_policy.model_dump(mode="json"),
        "robustness_diagnostics": robustness_diagnostics,
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
            "holding_policy_evidence_tier": holding_policy_decision.evidence_tier,
            "session_flat_compliant": holding_policy_analysis.session_flat_compliant,
            "overnight_exposure_count": holding_policy_analysis.overnight_exposure_count,
            "weekend_exposure_count": holding_policy_analysis.weekend_exposure_count,
            "gap_return_contribution_pct": holding_policy_analysis.gap_return_contribution_pct,
            "holding_policy_rejection": classification
            in {
                "rejected_overnight_risk",
                "rejected_weekend_risk",
                "rejected_holding_policy_violation",
            },
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
            item.get("holding_policy_evidence_tier", ""),
            item.get("error_message", ""),
        ]
        for item in payload["symbol_results"]
    ]
    table = "\n".join(
        [
            "| Symbol | Status | Classification | Net Return | Trades | "
            "Benchmark | Null | Holding Tier | Message |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
            *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
        ]
    )
    return f"""# Universe Research Run: {payload["run_id"]}

## Summary

- Hypothesis: `{payload["hypothesis_id"]}`
- Universe: `{payload["universe_id"]}`
- Symbols tested: {payload["symbol_count"]}
- Completed: {payload["completed_count"]}
- Skipped: {payload["skipped_count"]}
- Failed: {payload["failed_count"]}
- Rejected: {payload["rejected_count"]}
- Candidate paper test: {payload["candidate_count"]}
- Intraday candidates: {payload["intraday_candidate_count"]}
- Swing exceptional candidates: {payload["swing_exceptional_candidate_count"]}

Rejected results are expected. Failed symbols are data or harness failures; rejected
symbols completed research and did not pass the conservative gates.

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

## Holding Policy Summary

- Holding policy rejection count: {payload["holding_policy_rejection_count"]}
- Overnight violation count: {payload["overnight_violation_count"]}
- Weekend violation count: {payload["weekend_violation_count"]}
- Median gap contribution pct: {payload["median_gap_return_contribution_pct"]}
- Median overnight exposure count: {payload["median_overnight_exposure_count"]}
- Median weekend exposure count: {payload["median_weekend_exposure_count"]}

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
                    "holding_policy_evidence_tier": str(
                        existing.get("holding_policy_evidence_tier", "")
                    ),
                    "session_flat_compliant": bool(existing.get("session_flat_compliant", False)),
                    "overnight_exposure_count": int(existing.get("overnight_exposure_count", 0)),
                    "weekend_exposure_count": int(existing.get("weekend_exposure_count", 0)),
                    "gap_return_contribution_pct": float(
                        existing.get("gap_return_contribution_pct", 0.0)
                    ),
                    "holding_policy_rejection": bool(
                        existing.get("holding_policy_rejection", False)
                    ),
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
            holding_policy_analysis = experiment_payload.get("holding_policy_analysis", {})
            holding_policy_decision = experiment_payload.get("holding_policy_decision", {})
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
                    "holding_policy_evidence_tier": str(
                        holding_policy_decision.get("evidence_tier", "")
                    ),
                    "session_flat_compliant": bool(
                        holding_policy_analysis.get("session_flat_compliant", False)
                    ),
                    "overnight_exposure_count": int(
                        holding_policy_analysis.get("overnight_exposure_count", 0)
                    ),
                    "weekend_exposure_count": int(
                        holding_policy_analysis.get("weekend_exposure_count", 0)
                    ),
                    "gap_return_contribution_pct": float(
                        holding_policy_analysis.get("gap_return_contribution_pct", 0.0)
                    ),
                    "holding_policy_rejection": bool(
                        str(result.classification)
                        in {
                            "rejected_overnight_risk",
                            "rejected_weekend_risk",
                            "rejected_holding_policy_violation",
                        }
                    ),
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
                    "holding_policy_evidence_tier": "",
                    "session_flat_compliant": False,
                    "overnight_exposure_count": 0,
                    "weekend_exposure_count": 0,
                    "gap_return_contribution_pct": 0.0,
                    "holding_policy_rejection": False,
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
    completed_count = sum(1 for item in symbol_results if item["status"] == "completed")
    skipped_count = sum(1 for item in symbol_results if item["status"] == "skipped")
    failed_count = sum(1 for item in symbol_results if item["status"] == "failed")
    intraday_candidate_count = classification_counts.get("candidate_intraday_test", 0)
    swing_exceptional_candidate_count = classification_counts.get(
        "candidate_swing_exceptional",
        0,
    )
    candidate_count = (
        classification_counts.get("candidate_paper_test", 0)
        + intraday_candidate_count
        + swing_exceptional_candidate_count
    )
    rejected_count = sum(
        count
        for classification_name, count in classification_counts.items()
        if classification_name.startswith("rejected_")
    )
    holding_policy_rejection_count = sum(
        1 for item in aggregate_items if item.get("holding_policy_rejection")
    )
    overnight_violation_count = sum(
        1
        for item in aggregate_items
        if item.get("classification") == "rejected_overnight_risk"
        or "overnight_risk_too_high" in item.get("classification_reasons", [])
    )
    weekend_violation_count = sum(
        1
        for item in aggregate_items
        if item.get("classification") == "rejected_weekend_risk"
        or "weekend_risk_too_high" in item.get("classification_reasons", [])
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
        "completed_count": completed_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "candidate_count": candidate_count,
        "intraday_candidate_count": intraday_candidate_count,
        "swing_exceptional_candidate_count": swing_exceptional_candidate_count,
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
        "holding_policy_rejection_count": holding_policy_rejection_count,
        "overnight_violation_count": overnight_violation_count,
        "weekend_violation_count": weekend_violation_count,
        "median_gap_return_contribution_pct": _median(
            [float(item["gap_return_contribution_pct"]) for item in aggregate_items]
        ),
        "median_overnight_exposure_count": _median(
            [float(item["overnight_exposure_count"]) for item in aggregate_items]
        ),
        "median_weekend_exposure_count": _median(
            [float(item["weekend_exposure_count"]) for item in aggregate_items]
        ),
        "median_max_drawdown": _median([float(item["max_drawdown"]) for item in aggregate_items]),
        "median_trade_count": _median([float(item["trade_count"]) for item in aggregate_items]),
        "median_stability_score": _median(
            [float(item["stability_score"]) for item in aggregate_items]
        ),
        "top_candidates": [
            item
            for item in symbol_results
            if item.get("classification")
            in {
                "candidate_paper_test",
                "candidate_intraday_test",
                "candidate_swing_exceptional",
            }
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
