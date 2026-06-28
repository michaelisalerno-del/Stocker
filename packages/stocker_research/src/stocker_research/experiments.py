"""Research experiment runner and conservative result classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import DirectionMode, VectorizedBacktestResult, evaluate_positions
from stocker_data.audit import create_audit_report
from stocker_data.storage import DatasetKey, dataset_metadata, load_dataset
from stocker_research.hypothesis import Hypothesis, load_hypothesis
from stocker_research.parameters import ParameterSet, generate_parameter_grid
from stocker_research.regime import label_regimes, performance_by_regime
from stocker_research.stability import StabilityReport, analyze_stability
from stocker_research.templates import (
    MeanReversionTemplate,
    MovingAverageMomentumTemplate,
    StrategyTemplate,
    VolatilityBreakoutTemplate,
)
from stocker_research.walkforward import (
    WalkForwardConfig,
    WalkForwardSplit,
    generate_walk_forward_splits,
)

Classification = Literal[
    "rejected_data_issue",
    "rejected_no_edge",
    "rejected_costs_kill_edge",
    "rejected_unstable_parameters",
    "rejected_walkforward_failure",
    "interesting_needs_more_tests",
    "candidate_paper_test",
]


@dataclass(frozen=True)
class ExperimentRunResult:
    """Paths and classification from a research experiment."""

    experiment_id: str
    classification: Classification
    markdown_path: Path
    json_path: Path


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
    if hypothesis.signal_family == "moving_average_momentum":
        return MovingAverageMomentumTemplate()
    if hypothesis.signal_family == "mean_reversion_after_large_down_day":
        return MeanReversionTemplate()
    if hypothesis.signal_family == "volatility_breakout":
        return VolatilityBreakoutTemplate()
    raise ValueError(f"Unsupported signal family: {hypothesis.signal_family}")


def _cost_model(hypothesis: Hypothesis) -> CostModel:
    return CostModel(
        spread_bps=hypothesis.cost_model.spread_bps,
        commission_bps=hypothesis.cost_model.commission_bps,
        slippage_bps=hypothesis.cost_model.slippage_bps,
    )


def _walk_forward_config(hypothesis: Hypothesis) -> WalkForwardConfig:
    method = hypothesis.validation_method
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
    direction_mode: DirectionMode = "long_only" if direction != "long_short" else "long_short"
    return evaluate_positions(
        frame.reset_index(drop=True),
        positions,
        cost_model=cost_model,
        direction=direction_mode,
    )


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
        test_returns: list[float] = []
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
            test_returns.append(test_result.net_return)
            test_trade_counts.append(test_result.number_of_trades)
            test_drawdowns.append(test_result.max_drawdown)
        rows.append(
            {
                "parameter_set_id": parameter_set.parameter_set_id,
                "params": parameter_set.params,
                "train_net_return": float(sum(train_returns) / len(train_returns)),
                "test_net_return": float(sum(test_returns) / len(test_returns)),
                "profitable_split_pct": float(
                    sum(value > 0 for value in test_returns) / len(test_returns)
                ),
                "trade_count": int(sum(test_trade_counts)),
                "max_drawdown": float(min(test_drawdowns)) if test_drawdowns else 0.0,
            }
        )
    return rows


def _experiment_id(hypothesis: Hypothesis, symbol: str, timeframe: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    return f"{stamp}_{hypothesis.id}_{symbol.upper()}_{timeframe}"


def _markdown(payload: dict[str, Any]) -> str:
    warnings = "\n".join(f"- {warning}" for warning in payload["warnings"]) or "- None"
    return f"""# Research Experiment: {payload["experiment_id"]}

## Classification

`{payload["classification"]}`

## Hypothesis

- Name: {payload["hypothesis"]["name"]}
- Signal family: `{payload["hypothesis"]["signal_family"]}`
- Expected edge reason: {payload["hypothesis"]["expected_edge_reason"]}

## Data

- Symbol: `{payload["symbol"]}`
- Timeframe: `{payload["timeframe"]}`
- Date range: {payload["data"]["min_timestamp"]} to {payload["data"]["max_timestamp"]}
- Rows: {payload["data"]["row_count"]}

## Walk-Forward

- Splits: {len(payload["splits"])}
- Best parameter set: `{payload["stability"]["best_parameter_set_id"]}`
- Median test return: {payload["stability"]["median_test_return"]}

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


def _update_index(report_dir: Path, entry: dict[str, Any]) -> None:
    index_json = report_dir / "index.json"
    if index_json.exists():
        payload = json.loads(index_json.read_text(encoding="utf-8"))
    else:
        payload = {"experiments": []}
    payload["experiments"].append(entry)
    index_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Research Experiments",
        "",
        "| Experiment | Hypothesis | Symbol | Timeframe | Classification | "
        "Net Return | Max DD | Trades | Stability |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["experiments"]:
        line_template = (
            "| {experiment_id} | {hypothesis_name} | {symbol} | {timeframe} | "
            "{classification} | {net_return:.6f} | {max_drawdown:.6f} | "
            "{trade_count} | {stability_score:.3f} |"
        )
        lines.append(line_template.format(**item))
    (report_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research_experiment(
    *,
    hypothesis_path: Path,
    data_dir: str | Path = "data",
    symbol: str,
    timeframe: str,
    source: str = "manual",
    instrument_type: str = "stock",
    max_parameter_sets: int = 100,
) -> ExperimentRunResult:
    """Run one disciplined research experiment and save reports."""

    hypothesis = load_hypothesis(hypothesis_path)
    key = DatasetKey(
        source=source, instrument_type=instrument_type, symbol=symbol.upper(), timeframe=timeframe
    )
    frame = load_dataset(key, data_dir=data_dir).sort_values("timestamp").reset_index(drop=True)
    metadata = dataset_metadata(key, data_dir=data_dir)
    audit = create_audit_report(
        data_dir=data_dir,
        symbol=symbol,
        timeframe=timeframe,
        source=source,
        instrument_type=instrument_type,
    )
    warnings = [f"data issue: {issue.code}" for issue in audit.issues if issue.severity == "error"]
    splits = generate_walk_forward_splits(frame, _walk_forward_config(hypothesis))
    if not splits:
        warnings.append("data issue: no walk-forward splits generated")
    parameter_sets = generate_parameter_grid(
        hypothesis.parameter_space, max_size=max_parameter_sets
    )
    grid_results = _run_grid(frame, splits, parameter_sets, hypothesis) if splits else []
    if grid_results:
        best = max(grid_results, key=lambda row: float(row["test_net_return"]))
        stability = analyze_stability(
            grid_results,
            best_parameter_set_id=str(best["parameter_set_id"]),
        )
    else:
        best = {
            "parameter_set_id": "none",
            "params": {},
            "train_net_return": 0.0,
            "test_net_return": 0.0,
            "profitable_split_pct": 0.0,
            "trade_count": 0,
            "max_drawdown": 0.0,
        }
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
    full_positions = template.generate_positions(frame, dict(best["params"]))
    full_result = evaluate_positions(frame, full_positions, cost_model=_cost_model(hypothesis))
    regimes = label_regimes(frame, window=min(20, max(3, len(frame) // 4)))
    regime_performance = performance_by_regime(
        pd.Series(full_result.net_returns),
        regimes["trend_regime"],
    )
    non_unknown_regimes = [name for name in regime_performance if name != "unknown"]
    classification = classify_experiment(
        test_net_return=float(best["test_net_return"]),
        train_net_return=float(best["train_net_return"]),
        stability_score=stability.stability_score,
        profitable_split_pct=float(best["profitable_split_pct"]),
        trade_count=int(best["trade_count"]),
        max_drawdown=float(best["max_drawdown"]),
        regime_count=len(non_unknown_regimes),
        warnings=warnings,
    )

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
        "cost_assumptions": hypothesis.cost_model.model_dump(),
        "splits": [split.model_dump() for split in splits],
        "grid_results": grid_results,
        "best_result": best,
        "worst_result": min(grid_results, key=lambda row: float(row["test_net_return"]))
        if grid_results
        else best,
        "stability": stability.to_dict(),
        "regime_performance": regime_performance,
        "warnings": warnings,
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
            "symbol": key.symbol,
            "timeframe": key.timeframe,
            "date_range": f"{metadata.min_timestamp} to {metadata.max_timestamp}",
            "classification": classification,
            "net_return": float(best["test_net_return"]),
            "max_drawdown": float(best["max_drawdown"]),
            "trade_count": int(best["trade_count"]),
            "stability_score": stability.stability_score,
            "report_path": str(markdown_path),
        },
    )
    return ExperimentRunResult(
        experiment_id=experiment_id,
        classification=classification,
        markdown_path=markdown_path,
        json_path=json_path,
    )
