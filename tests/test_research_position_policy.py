import json
from pathlib import Path

import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.experiments import run_research_experiment
from stocker_research.holding_policy import analyze_holding_policy
from stocker_research.hypothesis import HypothesisHoldingPolicy
from stocker_research.position_policy import (
    PositionPolicyResult,
    apply_holding_policy_to_positions,
    summarize_position_policy_effect,
)
from stocker_research.templates.base import StrategyTemplate
from stocker_research.windows import evaluate_window_with_context


def _two_session_intraday_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-06-25 13:30",
            "2026-06-25 18:55",
            "2026-06-25 19:30",
            "2026-06-25 19:55",
            "2026-06-25 20:00",
            "2026-06-26 13:30",
            "2026-06-26 18:55",
            "2026-06-26 19:30",
            "2026-06-26 19:55",
            "2026-06-26 20:00",
        ],
        utc=True,
    )
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 104.5, 105.0, 105.5, 106.0, 106.5])
    return pd.DataFrame(
        {
            "source": "eodhd",
            "symbol": "AAPL.US",
            "instrument_type": "stock",
            "timeframe": "5m",
            "timestamp": timestamps,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "volume": 1000,
            "currency": "USD",
            "timezone": "UTC",
        }
    )


def _intraday_policy() -> HypothesisHoldingPolicy:
    return HypothesisHoldingPolicy(
        preferred_style="intraday",
        allow_intraday=True,
        allow_overnight=False,
        allow_weekend=False,
        max_holding_sessions=1,
        flatten_before_close_minutes=10,
        entry_cutoff_before_close_minutes=30,
    )


def test_intraday_policy_forces_flat_before_close_blocks_late_entries_and_prevents_carry() -> None:
    frame = _two_session_intraday_frame()
    positions = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])

    result = apply_holding_policy_to_positions(
        frame,
        positions,
        policy=_intraday_policy(),
        timeframe="5m",
        market_calendar="XNYS",
    )

    assert result.adjusted_positions.tolist() == [
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert result.positions_forced_flat_count >= 2
    assert result.late_entries_blocked_count == 1
    assert result.overnight_carry_prevented_count == 1
    assert result.sessions_seen == 2
    assert result.adjusted_exposure < result.raw_exposure


def test_daily_timeframe_does_not_pretend_to_be_session_flat() -> None:
    frame = _two_session_intraday_frame().iloc[[0, 5]].reset_index(drop=True)
    positions = pd.Series([1.0, 1.0])

    result = apply_holding_policy_to_positions(
        frame,
        positions,
        policy=_intraday_policy(),
        timeframe="1d",
    )

    assert result.adjusted_positions.tolist() == [1.0, 1.0]
    assert "daily_timeframe_not_session_flat_adjusted" in result.warnings
    assert result.policy_applied is False


class _ContextEntryTemplate(StrategyTemplate):
    name = "context_entry_template"

    def required_lookback_bars(self, params: dict[str, object]) -> int:
        return 1

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, object]) -> pd.Series:
        return pd.Series([0.0, 1.0, 1.0, 1.0], index=frame.index).reindex(frame.index).fillna(1.0)


def test_context_rows_can_create_first_scored_bar_position_with_normal_entry_cost() -> None:
    frame = _two_session_intraday_frame().iloc[:4].reset_index(drop=True)

    evaluation = evaluate_window_with_context(
        frame,
        _ContextEntryTemplate(),
        {},
        cost_model=CostModel(spread_bps=2.0, commission_bps=0.0, slippage_bps=2.0),
        direction="long_only",
        eval_start=1,
        eval_end=4,
        holding_policy=_intraday_policy(),
        timeframe="5m",
        market_calendar="XNYS",
    )

    assert evaluation.window.context_rows_used == 1
    assert evaluation.window.raw_eval_positions.iloc[0] == 1.0
    assert evaluation.window.eval_positions.iloc[0] == 1.0
    assert evaluation.result.number_of_trades >= 1
    assert evaluation.position_policy is not None
    assert evaluation.position_policy.adjusted_exposure <= evaluation.position_policy.raw_exposure


def test_holding_analysis_uses_market_calendar_for_incomplete_intraday_windows() -> None:
    frame = _two_session_intraday_frame().iloc[:3].reset_index(drop=True)
    positions = pd.Series([0.0, 1.0, 1.0])

    report = analyze_holding_policy(
        frame,
        positions,
        result=None,
        selected_net_return=0.01,
        timeframe="5m",
        policy=_intraday_policy(),
        market_calendar="XNYS",
    )

    assert report.session_flat_compliant is True
    assert "failed_flatten_before_close" not in report.holding_policy_violations
    assert "entry_after_cutoff" not in report.holding_policy_violations


def test_position_policy_summary_caps_timestamp_action_samples() -> None:
    results = [
        PositionPolicyResult(
            adjusted_positions=pd.Series([0.0]),
            policy_applied=True,
            policy_name="session_flat_intraday",
            policy_actions=[
                {
                    "action": "forced_flat_before_close",
                    "timestamp": f"2026-06-25 19:{minute:02d}:00+00:00",
                    "session": "2026-06-25",
                }
                for minute in range(60)
            ],
            positions_forced_flat_count=60,
        )
    ]

    summary = summarize_position_policy_effect(results)

    assert summary["policy_action_count"] == 60
    assert len(summary["policy_actions"]) == 50
    assert summary["policy_actions_omitted_count"] == 10
    assert summary["positions_forced_flat_count"] == 60


def _write_intraday_hypothesis(path: Path) -> Path:
    path.write_text(
        """
id: test_intraday_moving_average
name: Test intraday moving average
description: Intraday session-flat test hypothesis.
hypothesis_version: 1
market_universe: unit_test
instrument_type: stock
symbol_filter: "*"
timeframe: 5m
data_source: eodhd
template: moving_average_momentum
signal_family: moving_average_momentum
entry_logic: Existing moving-average template, scored with a session-flat overlay.
exit_logic: Session-flat overlay forces flat before close.
holding_period: Intraday only, no overnight carry.
direction: long_only
costs:
  spread_bps: 2.0
  commission_bps: 0.5
  slippage_bps: 2.0
risk:
  max_drawdown: 0.30
  min_trades: 1
parameter_space:
  fast_window: [1]
  slow_window: [2]
maximum_parameter_sets: 1
walkforward:
  mode: rolling
  train_bars: 4
  test_bars: 4
  embargo_bars: 0
  step_bars: 4
  minimum_rows: 8
expected_edge_reason: Test-only report contract check.
invalidation_rules:
  - Reject if session-flat scoring is not applied.
minimum_evidence:
  min_trades: 1
  min_profitable_split_pct: 0.0
  min_stability_score: 0.0
holding_policy:
  preferred_style: intraday
  allow_intraday: true
  allow_overnight: false
  allow_weekend: false
  max_holding_sessions: 1
  flatten_before_close_minutes: 10
  entry_cutoff_before_close_minutes: 30
  require_exceptional_evidence_for_swing: true
created_at: "2026-06-28T00:00:00Z"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_position_policy_action_counts_appear_in_experiment_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    key = DatasetKey(source="eodhd", instrument_type="stock", symbol="AAPL.US", timeframe="5m")
    write_parquet(_two_session_intraday_frame(), dataset_path(key, data_dir=data_dir))
    hypothesis_path = _write_intraday_hypothesis(tmp_path / "intraday.yaml")

    result = run_research_experiment(
        hypothesis_path=hypothesis_path,
        data_dir=data_dir,
        symbol="AAPL.US",
        timeframe="5m",
        source="eodhd",
        instrument_type="stock",
        max_parameter_sets=1,
        market_calendar="XNYS",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert "position_policy" in payload
    assert payload["position_policy"]["policy_applied"] is True
    assert payload["position_policy"]["positions_forced_flat_count"] >= 1
    assert (
        payload["position_policy"]["raw_exposure"]
        >= payload["position_policy"]["adjusted_exposure"]
    )
    assert payload["raw_template_holding_policy_analysis"]
    assert payload["scored_holding_policy_analysis"] == payload["holding_policy_analysis"]
    assert payload["holding_policy_analysis"]["session_flat_compliant"] is True
