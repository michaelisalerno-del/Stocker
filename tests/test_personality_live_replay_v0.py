from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.personality_live_replay_v0 import (
    LiveReplayConfig,
    _parse_filter_rule,
    run_personality_live_replay_lab,
    simulate_trade_outcome,
)


def _write_replay_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    event_dir = tmp_path / "events"
    template_dir = tmp_path / "templates"
    stop_dir = tmp_path / "stop_report"
    event_dir.mkdir()
    template_dir.mkdir()
    stop_dir.mkdir()

    template_path = template_dir / "templates.yaml"
    template_path.write_text(
        """
version: live_replay_test
research_only: true
templates:
  - template_id: replay_long
    personality: replay_long
    parent_event_state: controlled_pullback_after_bullish_impulse
    role: long_reversal_candidate
    expected_direction: 1
    horizon: 12
    base_conditions:
      - feature: close_location_value
        operator: ">="
        threshold: 0.5
    regime_fields: [time_regime]
    filter_features: [distance_from_session_low_pct]
  - template_id: replay_short
    personality: replay_short
    parent_event_state: failed_bullish_impulse_recoil
    role: short_or_long_blocker
    expected_direction: -1
    horizon: 24
    base_conditions:
      - feature: close_location_value
        operator: "<="
        threshold: 0.4
    regime_fields: [time_regime]
    filter_features: [distance_from_session_high_pct]
""",
        encoding="utf-8",
    )
    rows = [
        {
            "symbol": "AAA",
            "timestamp": "2026-06-03T14:30:00Z",
            "session_date": "2026-06-03",
            "event_state": "controlled_pullback_after_bullish_impulse",
            "close_location_value": 0.7,
            "time_regime": "morning",
            "distance_from_session_low_pct": 0.006,
            "distance_from_session_high_pct": -0.010,
            "distance_from_recent_low_pct": 0.005,
            "distance_from_recent_high_pct": -0.010,
            "distance_from_opening_range_low_pct": 0.006,
            "distance_from_opening_range_high_pct": -0.010,
            "forward_12_bar_return": 0.003,
            "forward_12_bar_mfe": 0.010,
            "forward_12_bar_mae": -0.001,
        },
        {
            "symbol": "AAA",
            "timestamp": "2026-06-03T14:35:00Z",
            "session_date": "2026-06-03",
            "event_state": "controlled_pullback_after_bullish_impulse",
            "close_location_value": 0.7,
            "time_regime": "morning",
            "distance_from_session_low_pct": 0.007,
            "distance_from_session_high_pct": -0.010,
            "distance_from_recent_low_pct": 0.005,
            "distance_from_recent_high_pct": -0.010,
            "distance_from_opening_range_low_pct": 0.006,
            "distance_from_opening_range_high_pct": -0.010,
            "forward_12_bar_return": -0.004,
            "forward_12_bar_mfe": 0.003,
            "forward_12_bar_mae": -0.006,
        },
        {
            "symbol": "BBB",
            "timestamp": "2026-06-04T15:00:00Z",
            "session_date": "2026-06-04",
            "event_state": "failed_bullish_impulse_recoil",
            "close_location_value": 0.2,
            "time_regime": "midday",
            "distance_from_session_low_pct": 0.010,
            "distance_from_session_high_pct": -0.006,
            "distance_from_recent_low_pct": 0.010,
            "distance_from_recent_high_pct": -0.004,
            "distance_from_opening_range_low_pct": 0.010,
            "distance_from_opening_range_high_pct": -0.006,
            "forward_24_bar_return": -0.006,
            "forward_24_bar_mfe": 0.001,
            "forward_24_bar_mae": -0.009,
        },
        {
            "symbol": "CCC",
            "timestamp": "2026-05-31T15:00:00Z",
            "session_date": "2026-05-31",
            "event_state": "controlled_pullback_after_bullish_impulse",
            "close_location_value": 0.8,
            "time_regime": "morning",
            "distance_from_session_low_pct": 0.007,
            "forward_12_bar_return": 0.005,
            "forward_12_bar_mfe": 0.006,
            "forward_12_bar_mae": -0.001,
        },
        {
            "symbol": "AAA",
            "timestamp": "2026-06-03T16:00:00Z",
            "session_date": "2026-06-03",
            "event_state": "dead_chop_blocker",
            "close_location_value": 0.5,
            "time_regime": "midday",
            "distance_from_session_low_pct": 0.001,
            "forward_12_bar_return": 0.0001,
            "forward_12_bar_mfe": 0.001,
            "forward_12_bar_mae": -0.001,
        },
    ]
    pd.DataFrame(rows).to_csv(event_dir / "event_rows.csv", index=False)
    candidate_book = pd.DataFrame(
        [
            {
                "template_id": "replay_long",
                "personality": "replay_long",
                "parent_event_state": "controlled_pullback_after_bullish_impulse",
                "role": "long_reversal_candidate",
                "horizon": 12,
                "expected_direction": 1,
                "regime_field": "time_regime",
                "regime_value": "morning",
                "filter_rule": "distance_from_session_low_pct >= 0.005",
                "stop_model": "fixed_50bps",
                "target_r": 1.0,
                "candidate_status": "candidate_continue_research",
            },
            {
                "template_id": "replay_long",
                "personality": "replay_long",
                "parent_event_state": "controlled_pullback_after_bullish_impulse",
                "role": "long_reversal_candidate",
                "horizon": 12,
                "expected_direction": 1,
                "regime_field": "time_regime",
                "regime_value": "morning",
                "filter_rule": "distance_from_session_low_pct >= 0.005",
                "stop_model": "fixed_50bps",
                "target_r": 2.0,
                "candidate_status": "candidate_continue_research",
            },
            {
                "template_id": "replay_short",
                "personality": "replay_short",
                "parent_event_state": "failed_bullish_impulse_recoil",
                "role": "short_or_long_blocker",
                "horizon": 24,
                "expected_direction": -1,
                "regime_field": "time_regime",
                "regime_value": "midday",
                "filter_rule": "distance_from_session_high_pct <= -0.003",
                "stop_model": "structure_session_extreme_10bps",
                "target_r": 1.0,
                "candidate_status": "candidate_continue_research",
            },
        ]
    )
    candidate_book.to_csv(stop_dir / "candidate_book.csv", index=False)
    (stop_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_event_dir": str(event_dir),
                "input_template_dir": str(tmp_path / "template_report"),
                "research_only": True,
            }
        ),
        encoding="utf-8",
    )
    return event_dir, template_path, stop_dir / "candidate_book.csv"


def test_parse_filter_rule_supports_simple_numeric_comparison() -> None:
    parsed = _parse_filter_rule("distance_from_session_low_pct >= 0.005")

    assert parsed["feature"] == "distance_from_session_low_pct"
    assert parsed["operator"] == ">="
    assert parsed["threshold"] == 0.005


def test_simulate_trade_uses_conservative_target_stop_order() -> None:
    row = pd.Series(
        {
            "forward_12_bar_return": 0.004,
            "forward_12_bar_mfe": 0.010,
            "forward_12_bar_mae": -0.006,
        }
    )

    result = simulate_trade_outcome(
        row,
        horizon=12,
        expected_direction=1,
        risk_bps=50.0,
        target_r=1.0,
        cost_bps=10.0,
    )

    assert result["exit_reason"] == "ambiguous_stop_first"
    assert result["net_r"] == -1.2


def test_live_replay_writes_trade_ledger_and_dedupes_duplicate_candidates(tmp_path: Path) -> None:
    event_dir, template_path, candidate_book = _write_replay_inputs(tmp_path)

    result = run_personality_live_replay_lab(
        input_event_dir=event_dir,
        candidate_book_path=candidate_book,
        template_path=template_path,
        output_dir=tmp_path / "out",
        replay_start="2026-06-01",
        replay_end="2026-06-30",
        config=LiveReplayConfig(cost_bps=10.0, random_iterations=5, random_seed=11),
    )

    for name in [
        "summary.md",
        "summary.json",
        "decision.json",
        "signals.csv",
        "trades.csv",
        "daily_pnl.csv",
        "symbol_summary.csv",
        "personality_summary.csv",
        "random_live_baseline.csv",
        "missed_candidates.csv",
        "blocked_by_dead_chop.csv",
    ]:
        assert (result.output_dir / name).exists()
    signals = pd.read_csv(result.signals_csv_path)
    trades = pd.read_csv(result.trades_csv_path)
    assert len(signals) == 3
    assert len(trades) == 2
    assert trades["timestamp"].min().startswith("2026-06")
    assert set(trades["template_id"]) == {"replay_long", "replay_short"}


def test_personality_live_replay_cli_smoke(tmp_path: Path) -> None:
    event_dir, template_path, candidate_book = _write_replay_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "personality-live-replay",
            "--input-event-dir",
            str(event_dir),
            "--candidate-book",
            str(candidate_book),
            "--template-path",
            str(template_path),
            "--replay-start",
            "2026-06-01",
            "--replay-end",
            "2026-06-30",
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--cost-bps",
            "10",
            "--random-iterations",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "personality_live_replay_v0" in result.output
