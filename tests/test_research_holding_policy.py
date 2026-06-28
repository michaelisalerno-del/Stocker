import json
from pathlib import Path

import pandas as pd
import pytest

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import evaluate_positions
from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.classification import classify_research_result
from stocker_research.experiments import run_research_experiment
from stocker_research.holding_policy import analyze_holding_policy, build_holding_policy_decision
from stocker_research.hypothesis import HypothesisHoldingPolicy


def _daily_frame(rows: int = 6) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
        ],
        utc=True,
    )[:rows]
    close = pd.Series([100.0, 106.0, 120.0, 121.0, 122.0, 123.0][:rows])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 105.0, 118.0, 120.5, 121.5, 122.5][:rows],
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000] * rows,
        }
    )


def _weekday_daily_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"], utc=True)
    close = pd.Series([100.0, 112.0, 120.0])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0, 110.0, 113.0],
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000] * len(timestamps),
        }
    )


def _intraday_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30",
            "2024-01-02 10:00",
            "2024-01-02 15:40",
            "2024-01-02 15:55",
            "2024-01-03 09:30",
            "2024-01-03 10:00",
            "2024-01-03 15:40",
            "2024-01-03 15:55",
        ],
        utc=True,
    )
    close = pd.Series([100.0, 101.0, 102.0, 102.5, 103.0, 104.0, 104.5, 105.0])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [1000] * len(timestamps),
        }
    )


def test_daily_multi_bar_hold_is_swing_not_session_flat() -> None:
    frame = _daily_frame()
    positions = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    result = evaluate_positions(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        direction="long_only",
    )

    report = analyze_holding_policy(
        frame,
        positions,
        result=result,
        timeframe="1d",
        policy=HypothesisHoldingPolicy(),
    )

    assert report.max_holding_bars == 3
    assert report.estimated_holding_sessions == 3
    assert report.overnight_exposure_count >= 2
    assert report.weekend_exposure_count == 1
    assert report.session_flat_compliant is False
    assert "daily_bars_are_swing_research_vehicle" in report.holding_policy_warning_reasons
    assert report.intraday_return_contribution is None
    assert "daily data cannot prove session-flat tradability" in report.attribution_note


def test_intraday_flat_by_close_passes_session_flat_policy() -> None:
    frame = _intraday_frame()
    positions = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    result = evaluate_positions(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        direction="long_only",
    )

    report = analyze_holding_policy(
        frame,
        positions,
        result=result,
        timeframe="30m",
        policy=HypothesisHoldingPolicy(),
    )

    assert report.session_flat_compliant is True
    assert report.overnight_exposure_count == 0
    assert report.weekend_exposure_count == 0
    assert report.holding_policy_violations == []
    assert "session_flat_compliant" in report.holding_policy_warning_reasons


def test_intraday_position_held_past_close_creates_violation() -> None:
    frame = _intraday_frame()
    positions = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    result = evaluate_positions(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        direction="long_only",
    )

    report = analyze_holding_policy(
        frame,
        positions,
        result=result,
        timeframe="30m",
        policy=HypothesisHoldingPolicy(),
    )

    assert report.session_flat_compliant is False
    assert report.overnight_exposure_count == 1
    assert "failed_flatten_before_close" in report.holding_policy_violations
    assert "held_overnight" in report.holding_policy_warning_reasons


def test_weekend_hold_creates_exceptional_only_warning() -> None:
    frame = _daily_frame()
    positions = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    result = evaluate_positions(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        direction="long_only",
    )

    report = analyze_holding_policy(
        frame,
        positions,
        result=result,
        timeframe="1d",
        policy=HypothesisHoldingPolicy(),
    )

    assert report.weekend_exposure_count == 1
    assert "weekend_exceptional_only" in report.holding_policy_warning_reasons


def test_gap_contribution_pct_uses_selected_test_return_denominator() -> None:
    frame = _weekday_daily_frame()
    positions = pd.Series([1.0, 0.0, 0.0])

    report = analyze_holding_policy(
        frame,
        positions,
        result=None,
        selected_net_return=0.20,
        timeframe="1d",
        policy=HypothesisHoldingPolicy(max_overnight_exposure_count=1),
    )

    assert report.gap_return_contribution == pytest.approx(0.10)
    assert report.gap_return_contribution_pct == pytest.approx(0.50)
    assert report.gap_return_contribution_pct != 1.0
    assert "daily data cannot prove session-flat tradability" in report.attribution_note


def test_profitable_gap_dependent_candidate_is_rejected_for_overnight_risk() -> None:
    classification = classify_research_result(
        net_test_return=0.12,
        gross_test_return=0.13,
        trade_count=80,
        stability_score=0.8,
        profitable_split_pct=0.8,
        max_drawdown=-0.08,
        cost_drag=0.01,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="rejected_overnight_risk",
        holding_policy_reasons=["gap_dependent_returns", "overnight_risk_too_high"],
    )

    assert classification.classification == "rejected_overnight_risk"
    assert "gap_dependent_returns" in classification.reasons


def test_swing_strategy_needs_exceptional_holding_gate_to_be_candidate() -> None:
    weak = classify_research_result(
        net_test_return=0.12,
        gross_test_return=0.13,
        trade_count=80,
        stability_score=0.8,
        profitable_split_pct=0.8,
        max_drawdown=-0.08,
        cost_drag=0.01,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="interesting_swing_needs_more_tests",
        holding_policy_reasons=["swing_not_exceptional"],
    )
    strong = classify_research_result(
        net_test_return=0.18,
        gross_test_return=0.2,
        trade_count=120,
        stability_score=0.85,
        profitable_split_pct=0.85,
        max_drawdown=-0.05,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        benchmark_pass=True,
        null_pass=True,
        holding_policy_classification="candidate_swing_exceptional",
        holding_policy_reasons=["swing_exceptional_evidence"],
    )

    assert weak.classification == "interesting_swing_needs_more_tests"
    assert strong.classification == "candidate_swing_exceptional"


def test_conditional_overnight_weak_swing_evidence_needs_more_tests() -> None:
    frame = _weekday_daily_frame()
    positions = pd.Series([1.0, 1.0, 0.0])
    policy = HypothesisHoldingPolicy(
        allow_overnight="conditional",
        allow_weekend="exceptional_only",
        max_overnight_exposure_count=2,
        max_weekend_exposure_count=0,
        max_gap_return_contribution_pct=0.75,
        min_swing_excess_vs_benchmark=0.05,
        min_swing_excess_vs_null=0.03,
        min_swing_trade_count=50,
        max_swing_drawdown=0.15,
    )
    report = analyze_holding_policy(
        frame,
        positions,
        result=None,
        selected_net_return=0.25,
        timeframe="1d",
        policy=policy,
    )

    decision = build_holding_policy_decision(
        report,
        policy,
        selected_excess_vs_benchmark=0.02,
        selected_excess_vs_null=0.01,
        trade_count=20,
        max_drawdown=-0.10,
    )

    assert report.weekend_exposure_count == 0
    assert decision.classification == "interesting_swing_needs_more_tests"
    assert "swing_not_exceptional" in decision.reasons


def test_conditional_overnight_exceptional_swing_evidence_can_be_candidate() -> None:
    frame = _weekday_daily_frame()
    positions = pd.Series([1.0, 1.0, 0.0])
    policy = HypothesisHoldingPolicy(
        allow_overnight="conditional",
        allow_weekend="exceptional_only",
        max_overnight_exposure_count=2,
        max_weekend_exposure_count=0,
        max_gap_return_contribution_pct=0.75,
        min_swing_excess_vs_benchmark=0.05,
        min_swing_excess_vs_null=0.03,
        min_swing_trade_count=50,
        max_swing_drawdown=0.15,
    )
    report = analyze_holding_policy(
        frame,
        positions,
        result=None,
        selected_net_return=0.25,
        timeframe="1d",
        policy=policy,
    )

    decision = build_holding_policy_decision(
        report,
        policy,
        selected_excess_vs_benchmark=0.08,
        selected_excess_vs_null=0.04,
        trade_count=80,
        max_drawdown=-0.08,
    )

    assert report.weekend_exposure_count == 0
    assert decision.classification == "candidate_swing_exceptional"
    assert decision.swing_exceptional_pass is True


def test_weekend_exposure_under_strict_policy_is_rejected() -> None:
    frame = _daily_frame(rows=4)
    positions = pd.Series([0.0, 1.0, 1.0, 0.0])
    policy = HypothesisHoldingPolicy(
        allow_overnight="conditional",
        allow_weekend="exceptional_only",
        max_overnight_exposure_count=5,
        max_weekend_exposure_count=0,
        max_gap_return_contribution_pct=0.75,
        min_swing_trade_count=0,
    )
    report = analyze_holding_policy(
        frame,
        positions,
        result=None,
        selected_net_return=0.25,
        timeframe="1d",
        policy=policy,
    )

    decision = build_holding_policy_decision(
        report,
        policy,
        selected_excess_vs_benchmark=1.0,
        selected_excess_vs_null=1.0,
        trade_count=100,
        max_drawdown=0.0,
    )

    assert report.weekend_exposure_count == 1
    assert decision.classification == "rejected_weekend_risk"
    assert "weekend_risk_too_high" in decision.reasons


def test_gap_dependent_returns_are_rejected_for_overnight_risk() -> None:
    frame = _weekday_daily_frame()
    positions = pd.Series([1.0, 0.0, 0.0])
    policy = HypothesisHoldingPolicy(
        allow_overnight="conditional",
        allow_weekend="exceptional_only",
        max_overnight_exposure_count=2,
        max_gap_return_contribution_pct=0.25,
        min_swing_trade_count=0,
    )
    report = analyze_holding_policy(
        frame,
        positions,
        result=None,
        selected_net_return=0.20,
        timeframe="1d",
        policy=policy,
    )

    decision = build_holding_policy_decision(
        report,
        policy,
        selected_excess_vs_benchmark=1.0,
        selected_excess_vs_null=1.0,
        trade_count=100,
        max_drawdown=0.0,
    )

    assert report.gap_return_contribution_pct == pytest.approx(0.50)
    assert decision.classification == "rejected_overnight_risk"
    assert "gap_dependent_returns" in decision.reasons


def _write_dataset(data_dir: Path, frame: pd.DataFrame, symbol: str = "AAPL.US") -> None:
    export = frame.copy()
    export["source"] = "manual"
    export["symbol"] = symbol
    export["instrument_type"] = "stock"
    export["timeframe"] = "1d"
    export["currency"] = "USD"
    export["timezone"] = "UTC"
    key = DatasetKey(source="manual", instrument_type="stock", symbol=symbol, timeframe="1d")
    write_parquet(export, dataset_path(key, data_dir=data_dir))


def _write_hypothesis(tmp_path: Path) -> Path:
    path = tmp_path / "holding_policy_hypothesis.yaml"
    path.write_text(
        json.dumps(
            {
                "id": "holding_policy_test_hypothesis",
                "name": "Holding Policy Test Hypothesis",
                "description": "Synthetic hypothesis for holding policy report tests.",
                "hypothesis_version": 1,
                "market_universe": "unit_test",
                "instrument_type": "stock",
                "timeframe": "1d",
                "data_source": "manual",
                "template": "moving_average_momentum",
                "direction": "long_only",
                "entry_logic": "Fast moving average above slow moving average.",
                "exit_logic": "Flat when fast moving average is no longer above slow average.",
                "holding_period": "Signal persistence only.",
                "expected_edge_reason": "Unit test fixture.",
                "invalidation_rules": ["Unit test invalidation."],
                "minimum_evidence": {
                    "min_trades": 0,
                    "min_profitable_split_pct": 0.0,
                    "min_stability_score": 0.0,
                },
                "holding_policy": {
                    "preferred_style": "intraday",
                    "allow_intraday": True,
                    "allow_overnight": "conditional",
                    "allow_weekend": "exceptional_only",
                    "max_holding_sessions": 5,
                    "require_exceptional_evidence_for_swing": True,
                    "require_gap_risk_report": True,
                    "min_swing_excess_vs_benchmark": 0.01,
                    "min_swing_trade_count": 0,
                    "max_gap_return_contribution_pct": 0.6,
                    "max_weekend_exposure_count": 0,
                    "max_overnight_exposure_count": 10,
                    "max_swing_drawdown": 0.25,
                },
                "parameter_space": {
                    "fast_window": [2],
                    "holding_period": [1],
                    "slow_window": [5],
                },
                "maximum_parameter_sets": 1,
                "costs": {"spread_bps": 0.0, "commission_bps": 0.0, "slippage_bps": 0.0},
                "risk": {"max_drawdown": 0.25},
                "walkforward": {
                    "mode": "rolling",
                    "train_bars": 20,
                    "test_bars": 10,
                    "embargo_bars": 0,
                    "step_bars": 10,
                    "minimum_rows": 30,
                },
                "created_at": "2026-06-28T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_runner_report_includes_holding_policy_section(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    rows = 52
    dates = pd.date_range("2024-01-02", periods=rows, freq="B", tz="UTC")
    close = pd.Series([100.0 + index * 0.5 for index in range(rows)])
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close.shift(1).fillna(close.iloc[0]) + 0.25,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000] * rows,
        }
    )
    _write_dataset(data_dir, frame)

    result = run_research_experiment(
        hypothesis_path=_write_hypothesis(tmp_path),
        data_dir=data_dir,
        symbol="AAPL.US",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    markdown = result.markdown_path.read_text(encoding="utf-8")

    for key in {
        "evaluation_policy",
        "indicator_context_policy",
        "context_summary",
        "holding_policy",
        "holding_policy_analysis",
        "holding_policy_decision",
        "classification_reasons",
    }:
        assert key in payload
    assert "holding_policy" in payload
    assert "holding_policy_analysis" in payload
    assert "holding_policy_decision" in payload
    assert payload["holding_policy_analysis"]["session_flat_compliant"] is False
    gap_contribution = float(payload["holding_policy_analysis"]["gap_return_contribution"])
    selected_net_return = float(payload["selected_result"]["test_net_return"])
    if gap_contribution and selected_net_return:
        assert payload["holding_policy_analysis"]["gap_return_contribution_pct"] == pytest.approx(
            abs(gap_contribution) / abs(selected_net_return)
        )
        assert payload["holding_policy_analysis"]["gap_return_contribution_pct"] != 1.0
    assert "daily_bars_are_swing_research_vehicle" in payload["classification_reasons"]
    assert "## Holding Policy" in markdown
    assert "daily data cannot prove session-flat tradability" in markdown
