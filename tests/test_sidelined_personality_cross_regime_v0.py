from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.sidelined_personality_cross_regime_v0 import (
    SidelinedPersonalityCrossRegimeConfig,
    run_sidelined_personality_cross_regime_lab,
)


def _write_sidelined_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    event_dir = tmp_path / "state_event_detector_v0" / "run"
    filter_dir = tmp_path / "combined_regime_filter_test_v0" / "run"
    external_dir = tmp_path / "exhaustion_extension_regime_filter_scan_v0" / "run"
    event_dir.mkdir(parents=True)
    filter_dir.mkdir(parents=True)
    external_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    for month in ("2026-01", "2026-02"):
        for index in range(30):
            rows.append(
                {
                    "symbol": symbols[index % len(symbols)],
                    "timestamp": f"{month}-{1 + index // 5:02d}T15:{index % 5:02d}:00Z",
                    "session_date": f"{month}-{1 + index // 5:02d}",
                    "bar_index_in_session": 25 + index,
                    "time_of_day_bucket": "midday",
                    "event_state": "controlled_pullback_after_bullish_impulse",
                    "distance_from_vwap_pct": 0.006,
                    "distance_from_opening_range_mid_pct": 0.005,
                    "distance_from_session_open_pct": 0.006,
                    "distance_from_session_high_pct": -0.010,
                    "distance_from_session_low_pct": 0.010,
                    "distance_from_recent_high_pct": -0.009,
                    "distance_from_recent_low_pct": 0.009,
                    "close_location_value": 0.7,
                    "upper_wick_pct_of_range": 0.1,
                    "lower_wick_pct_of_range": 0.1,
                    "bar_return": 0.002,
                    "prior_3_bar_return": 0.004,
                    "prior_6_bar_return": 0.006,
                    "prior_12_bar_return": 0.010,
                    "directional_efficiency_6": 0.6,
                    "directional_efficiency_12": 0.6,
                    "rolling_intraday_range_pct": 0.016,
                    "compression_zscore": 0.2,
                    "range_zscore": 0.1,
                    "relative_volume_at_bar_index": 1.0,
                    "relative_cumulative_volume": 1.0,
                    "forward_12_bar_return": 0.010,
                    "forward_12_bar_mfe": 0.012,
                    "forward_12_bar_mae": -0.002,
                }
            )
    for month_index, month in enumerate(("2026-01", "2026-02", "2026-03")):
        for index in range(24):
            low_move = index % 4 != 0
            rows.append(
                {
                    "symbol": symbols[index % len(symbols)],
                    "timestamp": f"{month}-{10 + index // 4:02d}T16:{index % 4:02d}:00Z",
                    "session_date": f"{month}-{10 + index // 4:02d}",
                    "bar_index_in_session": 30 + index,
                    "time_of_day_bucket": "midday",
                    "event_state": "dead_chop_blocker",
                    "distance_from_vwap_pct": 0.0002,
                    "distance_from_opening_range_mid_pct": 0.0001,
                    "distance_from_session_open_pct": 0.0002,
                    "distance_from_session_high_pct": -0.004,
                    "distance_from_session_low_pct": 0.004,
                    "distance_from_recent_high_pct": -0.003,
                    "distance_from_recent_low_pct": 0.003,
                    "close_location_value": 0.45,
                    "upper_wick_pct_of_range": 0.2,
                    "lower_wick_pct_of_range": 0.2,
                    "bar_return": 0.0001,
                    "prior_3_bar_return": 0.0003,
                    "prior_6_bar_return": 0.0005,
                    "prior_12_bar_return": 0.0007,
                    "directional_efficiency_6": 0.10,
                    "directional_efficiency_12": 0.12,
                    "rolling_intraday_range_pct": 0.005 + (0.0001 * (index % 6)),
                    "compression_zscore": -0.8,
                    "range_zscore": -0.7,
                    "relative_volume_at_bar_index": 0.8,
                    "relative_cumulative_volume": 0.8,
                    "forward_12_bar_return": 0.0004
                    if low_move
                    else 0.006 + month_index * 0.001,
                    "forward_12_bar_mfe": 0.0008 if low_move else 0.008,
                    "forward_12_bar_mae": -0.0007 if low_move else -0.002,
                }
            )
    pd.DataFrame(rows).to_csv(event_dir / "event_rows.csv", index=False)
    (event_dir / "summary.json").write_text("{}", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "personality": "pullback_continuation",
                "event_state": "controlled_pullback_after_bullish_impulse",
                "horizon": 12,
                "regime_field": "time_x_vwap_regime",
                "regime_value": "midday|above",
                "filter_rule": "bar_return >= 0",
            }
        ]
    ).to_csv(filter_dir / "selected_filters.csv", index=False)

    pd.DataFrame(
        [
            {
                "horizon": 6,
                "regime_field": "time_regime",
                "regime_value": "morning",
                "filter_rule": "role_bar_reversal <= -0.001",
                "retained_test_count": 51,
                "filtered_test_same_result_rate": 0.68,
                "test_lift_vs_regime": 0.15,
                "filtered_test_median_aligned_return": 0.0037,
                "symbol_count": 5,
                "single_symbol_share": 0.29,
                "verdict": "pass_exhaustion_regime_filter",
            }
        ]
    ).to_csv(external_dir / "selected_exhaustion_regime_filter_results.csv", index=False)
    (external_dir / "summary.json").write_text(
        '{"decision": "continue_research_exhaustion_regime_filter"}',
        encoding="utf-8",
    )
    return event_dir, filter_dir, external_dir


def test_sidelined_screen_excludes_promoted_personalities_and_scores_no_trade(
    tmp_path: Path,
) -> None:
    event_dir, filter_dir, external_dir = _write_sidelined_inputs(tmp_path)

    result = run_sidelined_personality_cross_regime_lab(
        input_event_dir=event_dir,
        input_selected_filter_dir=filter_dir,
        output_dir=tmp_path / "out",
        external_report_dirs=(external_dir,),
        config=SidelinedPersonalityCrossRegimeConfig(
            horizons=(12,),
            regime_fields=("time_x_vwap_regime",),
            filter_features=("rolling_intraday_range_pct",),
            quantiles=(0.5,),
            min_train_events=20,
            min_test_events=8,
            min_retained_events=4,
            random_iterations=3,
            low_movement_threshold=0.0015,
        ),
    )

    summary = pd.read_csv(result.sidelined_personality_summary_csv_path)
    filters = pd.read_csv(result.candidate_filter_results_csv_path)
    external = pd.read_csv(result.external_report_evidence_csv_path)
    assert set(summary["personality"]) == {"dead_chop_noise"}
    assert "pullback_continuation" not in set(summary["personality"])
    assert summary["role"].iloc[0] == "no_trade_filter"
    assert summary["test_low_movement_rate"].iloc[0] > 0.5
    assert (filters["personality"] == "dead_chop_noise").all()
    assert filters["role_objective"].eq("low_movement").all()
    assert external["personality"].tolist() == ["exhaustion_extension"]
    assert result.decision in {
        "continue_research_sidelined_personality",
        "reject_no_sidelined_personality_promoted",
    }


def test_sidelined_screen_report_writes_expected_files(tmp_path: Path) -> None:
    event_dir, filter_dir, external_dir = _write_sidelined_inputs(tmp_path)

    result = run_sidelined_personality_cross_regime_lab(
        input_event_dir=event_dir,
        input_selected_filter_dir=filter_dir,
        output_dir=tmp_path / "out",
        external_report_dirs=(external_dir,),
        config=SidelinedPersonalityCrossRegimeConfig(
            horizons=(12,),
            regime_fields=("time_x_vwap_regime",),
            filter_features=("rolling_intraday_range_pct",),
            quantiles=(0.5,),
            min_train_events=20,
            min_test_events=8,
            min_retained_events=4,
            random_iterations=3,
            low_movement_threshold=0.0015,
        ),
    )

    for name in [
        "summary.md",
        "summary.json",
        "decision.json",
        "sidelined_personality_summary.csv",
        "crossed_regime_summary.csv",
        "candidate_filter_results.csv",
        "selected_sidelined_candidates.csv",
        "rejected_sidelined_candidates.csv",
        "external_report_evidence.csv",
        "concentration_warnings.csv",
    ]:
        assert (result.output_dir / name).exists()


def test_sidelined_screen_cli_smoke(tmp_path: Path) -> None:
    event_dir, filter_dir, external_dir = _write_sidelined_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "sidelined-personality-cross-regime",
            "--input-event-dir",
            str(event_dir),
            "--input-selected-filter-dir",
            str(filter_dir),
            "--external-report-dir",
            str(external_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--horizons",
            "12",
            "--regime-fields",
            "time_x_vwap_regime",
            "--filter-features",
            "rolling_intraday_range_pct",
            "--quantiles",
            "0.5",
            "--min-train-events",
            "20",
            "--min-test-events",
            "8",
            "--min-retained-events",
            "4",
            "--random-iterations",
            "3",
            "--low-movement-threshold",
            "0.0015",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "sidelined_personality_cross_regime_v0" in result.output
