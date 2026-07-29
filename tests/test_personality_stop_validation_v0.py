from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.personality_stop_validation_v0 import (
    PersonalityStopValidationConfig,
    _apply_cost_to_scored_events,
    _risk_bps_for_model,
    _split_train_test_by_time,
    run_personality_stop_validation_lab,
    score_stop_model_events,
)


def _write_input_report(tmp_path: Path) -> tuple[Path, Path]:
    event_dir = tmp_path / "state_event_detector_v0" / "run"
    template_dir = tmp_path / "personality_template_v0" / "run"
    config_dir = tmp_path / "configs"
    event_dir.mkdir(parents=True)
    template_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    for index in range(30):
        symbol = ["AAA", "BBB", "CCC", "DDD", "EEE"][index % 5]
        rows.append(
            {
                "symbol": symbol,
                "timestamp": f"2026-01-{1 + index // 5:02d}T14:{index % 5:02d}:00Z",
                "session_date": f"2026-01-{1 + index // 5:02d}",
                "event_state": "controlled_pullback_after_bullish_impulse",
                "close_location_value": 0.62,
                "distance_from_session_low_pct": 0.004 + 0.0001 * index,
                "distance_from_session_high_pct": -0.012 - 0.0001 * index,
                "distance_from_recent_low_pct": 0.003 + 0.0001 * index,
                "distance_from_recent_high_pct": -0.009 - 0.0001 * index,
                "distance_from_opening_range_low_pct": 0.005 + 0.0001 * index,
                "distance_from_opening_range_high_pct": -0.011 - 0.0001 * index,
                "time_regime": "morning" if index < 18 else "midday",
                "forward_12_bar_return": 0.006 if index < 16 else -0.002,
                "forward_12_bar_mfe": 0.009 if index < 16 else 0.003,
                "forward_12_bar_mae": -0.002 if index < 16 else -0.007,
            }
        )
    for index in range(30):
        symbol = ["AAA", "BBB", "CCC", "DDD", "EEE"][index % 5]
        rows.append(
            {
                "symbol": symbol,
                "timestamp": f"2026-02-{1 + index // 5:02d}T15:{index % 5:02d}:00Z",
                "session_date": f"2026-02-{1 + index // 5:02d}",
                "event_state": "failed_bullish_impulse_recoil",
                "close_location_value": 0.28,
                "distance_from_session_low_pct": 0.010 + 0.0001 * index,
                "distance_from_session_high_pct": -0.004 - 0.0001 * index,
                "distance_from_recent_low_pct": 0.008 + 0.0001 * index,
                "distance_from_recent_high_pct": -0.003 - 0.0001 * index,
                "distance_from_opening_range_low_pct": 0.011 + 0.0001 * index,
                "distance_from_opening_range_high_pct": -0.006 - 0.0001 * index,
                "time_regime": "midday",
                "forward_24_bar_return": -0.008 if index < 18 else 0.002,
                "forward_24_bar_mfe": 0.002 if index < 18 else 0.008,
                "forward_24_bar_mae": -0.011 if index < 18 else -0.001,
            }
        )
    pd.DataFrame(rows).to_csv(event_dir / "event_rows.csv", index=False)

    template_path = config_dir / "templates.yaml"
    template_path.write_text(
        """
version: test
research_only: true
templates:
  - template_id: pullback_template
    personality: pullback_template
    parent_event_state: controlled_pullback_after_bullish_impulse
    role: long_continuation_candidate
    expected_direction: 1
    horizon: 12
    base_conditions:
      - feature: close_location_value
        operator: ">="
        threshold: 0.5
    regime_fields: [time_regime]
    filter_features: [distance_from_session_low_pct]
    minimums:
      base_events: 5
      retained_events: 5
      symbols: 3
      months: 1
  - template_id: recoil_template
    personality: recoil_template
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
    minimums:
      base_events: 5
      retained_events: 5
      symbols: 3
      months: 1
""",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "template_id": "pullback_template",
                "role": "long_continuation_candidate",
                "horizon": 12,
                "expected_direction": 1,
                "regime_field": "time_regime",
                "regime_value": "morning",
                "feature": "distance_from_session_low_pct",
                "operator": ">=",
                "threshold": 0.004,
                "filter_rule": "distance_from_session_low_pct >= 0.004",
                "retained_event_count": 18,
            },
            {
                "template_id": "recoil_template",
                "role": "short_or_long_blocker",
                "horizon": 24,
                "expected_direction": -1,
                "regime_field": "time_regime",
                "regime_value": "midday",
                "feature": "distance_from_session_high_pct",
                "operator": "<=",
                "threshold": -0.003,
                "filter_rule": "distance_from_session_high_pct <= -0.003",
                "retained_event_count": 30,
            },
        ]
    ).to_csv(template_dir / "selected_template_rules.csv", index=False)
    (template_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_event_dir": str(event_dir),
                "template_path": str(template_path),
                "research_only": True,
            }
        ),
        encoding="utf-8",
    )
    return template_dir, event_dir


def test_short_blocker_negative_return_scores_positive_r_multiple() -> None:
    rows = pd.DataFrame(
        [
            {
                "forward_24_bar_return": -0.006,
                "forward_24_bar_mfe": 0.001,
                "forward_24_bar_mae": -0.010,
            }
        ]
    )

    scored = score_stop_model_events(
        rows,
        horizon=24,
        expected_direction=-1,
        risk_bps=pd.Series([50.0]),
        target_r=1.5,
    )

    assert scored["aligned_return_bps"].iloc[0] == 60.0
    assert scored["stop_hit"].iloc[0] is False
    assert scored["final_r_conservative"].iloc[0] == 1.2


def test_stop_hit_for_long_is_conservative_minus_one_r() -> None:
    rows = pd.DataFrame(
        [
            {
                "forward_12_bar_return": 0.006,
                "forward_12_bar_mfe": 0.010,
                "forward_12_bar_mae": -0.007,
            }
        ]
    )

    scored = score_stop_model_events(
        rows,
        horizon=12,
        expected_direction=1,
        risk_bps=pd.Series([50.0]),
        target_r=1.5,
    )

    assert scored["stop_hit"].iloc[0] is True
    assert scored["target_hit"].iloc[0] is True
    assert scored["target_stop_order_ambiguous"].iloc[0] is True
    assert scored["final_r_conservative"].iloc[0] == -1.0


def test_structure_risk_uses_current_distance_not_forward_columns() -> None:
    rows = pd.DataFrame(
        {
            "distance_from_session_low_pct": [0.004],
            "forward_12_bar_mae": [-0.20],
        }
    )

    risk = _risk_bps_for_model(
        rows,
        model_name="structure_session_extreme_10bps",
        expected_direction=1,
    )

    assert risk.iloc[0] == 50.0


def test_train_test_split_uses_earlier_timestamps_for_train() -> None:
    rows = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-03T14:30:00Z",
                "2026-01-01T14:30:00Z",
                "2026-01-02T14:30:00Z",
                "2026-01-04T14:30:00Z",
            ],
            "value": [3, 1, 2, 4],
        }
    )

    train, test = _split_train_test_by_time(rows, train_fraction=0.5)

    assert train["value"].tolist() == [1, 2]
    assert test["value"].tolist() == [3, 4]


def test_cost_sensitivity_reduces_r_multiple() -> None:
    rows = pd.DataFrame(
        [
            {
                "forward_12_bar_return": 0.006,
                "forward_12_bar_mfe": 0.008,
                "forward_12_bar_mae": -0.001,
            }
        ]
    )
    scored = score_stop_model_events(
        rows,
        horizon=12,
        expected_direction=1,
        risk_bps=pd.Series([50.0]),
        target_r=1.5,
    )

    costed = _apply_cost_to_scored_events(scored, cost_bps=10.0)

    assert costed["final_r_after_cost"].iloc[0] == 1.0
    assert costed["final_r_after_cost"].iloc[0] < scored["final_r_conservative"].iloc[0]


def test_report_generation_writes_stop_validation_outputs(tmp_path: Path) -> None:
    template_dir, _ = _write_input_report(tmp_path)

    result = run_personality_stop_validation_lab(
        input_template_dir=template_dir,
        output_dir=tmp_path / "out",
        config=PersonalityStopValidationConfig(
            stop_loss_bps=(50.0,),
            target_r_multiples=(1.5,),
            random_iterations=5,
            random_seed=7,
            cost_bps=(0.0, 10.0),
        ),
    )

    assert result.decision in {
        "continue_research_stop_model",
        "reject_no_stop_model_improvement",
        "reject_concentrated",
    }
    for name in [
        "summary.md",
        "summary.json",
        "decision.json",
        "stop_model_results.csv",
        "selected_stop_models.csv",
        "rejected_stop_models.csv",
        "random_stop_baseline.csv",
        "oos_stop_results.csv",
        "cost_sensitivity.csv",
        "frequency_summary.csv",
        "candidate_book.csv",
        "concentration_warnings.csv",
        "stop_event_examples.csv",
    ]:
        assert (result.output_dir / name).exists()
    selected = pd.read_csv(result.selected_stop_models_csv_path)
    assert "median_final_r_conservative" in selected.columns
    assert "random_median_final_r_conservative" in selected.columns
    oos = pd.read_csv(result.oos_stop_results_csv_path)
    assert "test_median_final_r_conservative" in oos.columns
    costs = pd.read_csv(result.cost_sensitivity_csv_path)
    assert "median_final_r_after_cost" in costs.columns
    frequency = pd.read_csv(result.frequency_summary_csv_path)
    assert "events_per_session" in frequency.columns
    candidate_book = pd.read_csv(result.candidate_book_csv_path)
    assert "candidate_status" in candidate_book.columns


def test_personality_stop_validation_cli_smoke(tmp_path: Path) -> None:
    template_dir, _ = _write_input_report(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "personality-stop-validation",
            "--input-template-dir",
            str(template_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--stop-loss-bps",
            "50",
            "--target-r-multiples",
            "1.5",
            "--random-iterations",
            "5",
            "--cost-bps",
            "0,10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "personality_stop_validation_v0" in result.output
