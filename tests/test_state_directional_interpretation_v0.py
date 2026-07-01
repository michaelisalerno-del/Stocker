from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.state_directional_interpretation_v0 import (
    DirectionalInterpretationConfig,
    add_aligned_return_columns,
    build_directional_decision,
    estimate_expected_direction,
    run_random_blocker_baseline,
    run_state_directional_interpretation_report,
    summarize_blocker_quality,
    summarize_no_trade_quality,
    summarize_short_candidates,
)


def _events() -> pd.DataFrame:
    rows = []
    for index in range(40):
        timestamp = pd.Timestamp("2026-01-02", tz="UTC") + pd.Timedelta(days=index)
        if index < 20:
            state = "failed_open_down_continuation"
            forward = -0.006 if index % 5 else 0.004
        elif index < 30:
            state = "controlled_pullback_after_bullish_impulse"
            forward = 0.005 if index % 4 else -0.003
        else:
            state = "dead_chop_blocker"
            forward = 0.0005 if index % 2 else -0.0004
        rows.append(
            {
                "symbol": "AAA" if index % 2 else "BBB",
                "timestamp": timestamp,
                "session_date": timestamp.date().isoformat(),
                "bar_index_in_session": 12 + index,
                "time_of_day_bucket": "morning" if index < 24 else "midday",
                "event_state": state,
                "forward_6_bar_return": forward,
                "forward_6_bar_mfe": abs(forward) + 0.002,
                "forward_6_bar_mae": -abs(forward) - 0.001,
                "forward_9_bar_return": forward * 1.1,
                "forward_9_bar_mfe": abs(forward) + 0.003,
                "forward_9_bar_mae": -abs(forward) - 0.001,
                "forward_24_bar_return": forward * 1.5,
                "forward_24_bar_mfe": abs(forward) + 0.004,
                "forward_24_bar_mae": -abs(forward) - 0.002,
            }
        )
    return pd.DataFrame(rows)


def _write_minimal_state_report(input_dir: Path, events: pd.DataFrame) -> None:
    input_dir.mkdir(parents=True)
    events.to_csv(input_dir / "event_rows.csv", index=False)
    (input_dir / "summary.md").write_text("# State Event Detector V0\n", encoding="utf-8")
    (input_dir / "summary.json").write_text(
        json.dumps({"decision": {"decision": "continue_research"}}),
        encoding="utf-8",
    )
    (input_dir / "decision.json").write_text(
        json.dumps({"decision": "continue_research"}),
        encoding="utf-8",
    )


def test_negative_raw_return_is_positive_aligned_for_short_direction() -> None:
    frame = pd.DataFrame(
        {
            "event_state": ["failed_open_down_continuation"],
            "forward_24_bar_return": [-0.01],
        }
    )

    aligned = add_aligned_return_columns(frame, horizon=24, expected_direction=-1)

    assert aligned.iloc[0]["aligned_24_bar_return"] == 0.01
    assert aligned.iloc[0]["directional_consistent_24"] is True


def test_train_only_expected_direction_ignores_test_period() -> None:
    train = pd.DataFrame({"forward_24_bar_return": [-0.01, -0.02, -0.03]})
    test = pd.DataFrame({"forward_24_bar_return": [0.50, 0.60, 0.70]})

    direction = estimate_expected_direction(
        event_state="failed_open_down_continuation",
        train_rows=train,
        horizon=24,
    )

    assert direction["expected_direction"] == -1
    assert direction["train_median_return"] < 0
    assert test["forward_24_bar_return"].median() > 0


def test_no_trade_expected_direction_stays_zero_even_with_directional_train_return() -> None:
    train = pd.DataFrame({"forward_12_bar_return": [-0.01, -0.02, -0.03]})

    direction = estimate_expected_direction(
        event_state="dead_chop_blocker",
        train_rows=train,
        horizon=12,
    )

    assert direction["expected_direction"] == 0
    assert direction["train_evidence_direction"] == -1


def test_blocker_quality_metrics_capture_bad_longs_and_false_blocks() -> None:
    frame = pd.DataFrame({"forward_24_bar_return": [-0.01, -0.02, 0.03, 0.04]})

    summary = summarize_blocker_quality(
        frame,
        event_state="failed_open_down_continuation",
        horizon=24,
    )

    assert summary["bad_long_capture_rate"] == 0.5
    assert summary["good_long_false_block_rate"] == 0.5
    assert summary["avoided_long_loss_bps"] == 150.0
    assert summary["missed_long_profit_bps"] == pytest.approx(350.0)
    assert summary["blocker_net_value_bps"] == pytest.approx(-200.0)


def test_short_candidate_metrics_treat_negative_forward_as_win() -> None:
    frame = pd.DataFrame(
        {
            "forward_24_bar_return": [-0.01, -0.02, 0.03],
            "forward_24_bar_mfe": [0.005, 0.004, 0.04],
            "forward_24_bar_mae": [-0.02, -0.03, -0.001],
        }
    )

    summary = summarize_short_candidates(
        frame,
        event_state="failed_bullish_impulse_recoil",
        horizon=24,
    )

    assert summary["short_win_rate"] == 2 / 3
    assert summary["short_median_return"] == 0.01
    assert summary["short_directional_accuracy"] == 2 / 3


def test_no_trade_quality_scores_low_movement_and_big_move_false_blocks() -> None:
    frame = pd.DataFrame(
        {
            "forward_6_bar_return": [0.0002, -0.0003, 0.01],
            "forward_6_bar_mfe": [0.001, 0.0012, 0.012],
            "forward_6_bar_mae": [-0.001, -0.0011, -0.002],
        }
    )

    summary = summarize_no_trade_quality(
        frame,
        event_state="dead_chop_blocker",
        horizon=6,
        low_movement_threshold=0.001,
    )

    assert summary["low_movement_rate"] == 2 / 3
    assert summary["false_block_big_move_rate"] == 1 / 3
    assert summary["no_trade_quality_score"] == 1 / 3


def test_random_blocker_baseline_keeps_same_count() -> None:
    frame = _events().head(20)

    baseline = run_random_blocker_baseline(
        frame,
        retained_count=7,
        horizon=24,
        seed=3,
        iterations=10,
    )

    assert baseline["retained_count"] == 7
    assert baseline["baseline"] == "random_blocker_same_count"


def test_directional_decision_keeps_consistent_short_state_despite_negative_raw_return() -> None:
    directional = pd.DataFrame(
        {
            "event_state": ["failed_open_down_continuation"],
            "horizon": [24],
            "role": ["long_blocker_or_short"],
            "event_count": [120],
            "symbol_count": [4],
            "single_symbol_share": [0.35],
            "single_session_share": [0.02],
            "expected_direction": [-1],
            "raw_median_return": [-0.01],
            "aligned_median_return": [0.01],
            "aligned_win_rate": [0.65],
            "random_aligned_median_return": [0.001],
            "generic_aligned_median_return": [0.0],
            "role_evidence_conflict": [False],
        }
    )

    decision = build_directional_decision(
        directional_summary=directional,
        blocker_quality=pd.DataFrame(),
        short_summary=pd.DataFrame(),
        no_trade_summary=pd.DataFrame(),
        config=DirectionalInterpretationConfig(min_events=30),
    )

    assert decision["decision"] == "continue_research_short_candidate"
    assert decision["state_decisions"][0]["state_decision"] == "continue_research_short_candidate"


def test_directional_decision_uses_oos_gate_over_full_sample_result() -> None:
    directional = pd.DataFrame(
        {
            "event_state": ["failed_open_down_continuation"],
            "horizon": [24],
            "role": ["long_blocker_or_short"],
            "event_count": [120],
            "symbol_count": [4],
            "single_symbol_share": [0.35],
            "single_session_share": [0.02],
            "expected_direction": [-1],
            "raw_median_return": [-0.01],
            "aligned_median_return": [0.01],
            "aligned_win_rate": [0.65],
            "random_aligned_median_return": [0.001],
            "generic_aligned_median_return": [0.0],
            "role_evidence_conflict": [False],
        }
    )
    oos = pd.DataFrame(
        {
            "event_state": ["failed_open_down_continuation"],
            "horizon": [24],
            "test_aligned_median_return": [-0.002],
            "test_aligned_win_rate": [0.40],
            "random_aligned_median_return": [0.0],
            "generic_aligned_median_return": [0.0],
        }
    )

    decision = build_directional_decision(
        directional_summary=directional,
        blocker_quality=pd.DataFrame(),
        short_summary=pd.DataFrame(),
        no_trade_summary=pd.DataFrame(),
        oos_response=oos,
        config=DirectionalInterpretationConfig(min_events=30),
    )

    assert decision["decision"] == "reject_random_baseline_better"
    assert decision["state_decisions"][0]["state_decision_basis"] == "oos_60_40"
    assert decision["state_decisions"][0]["state_decision"] == "reject_random_baseline_better"


def test_directional_interpretation_report_writes_expected_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "state_event_detector_v0_20260630T000000Z"
    _write_minimal_state_report(input_dir, _events())

    result = run_state_directional_interpretation_report(
        input_dir=input_dir,
        config=DirectionalInterpretationConfig(horizons=(6, 9, 24), min_events=3),
    )

    expected = {
        "directional_state_summary.csv",
        "blocker_quality_summary.csv",
        "short_candidate_summary.csv",
        "no_trade_quality_summary.csv",
        "oos_directional_state_response.csv",
        "decision.json",
        "summary.md",
    }
    assert expected.issubset({path.name for path in input_dir.iterdir()})
    decision = json.loads((input_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["research_only"] is True
    assert result.output_dir == input_dir


def test_directional_interpretation_cli_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "state_event_detector_v0_20260630T000000Z"
    _write_minimal_state_report(input_dir, _events())

    result = CliRunner().invoke(
        app,
        [
            "research",
            "state-event-directional-interpretation",
            "--input-dir",
            str(input_dir),
            "--min-events",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "state_directional_interpretation_v0" in result.output
