from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.exhaustion_extension_exit_replay_v0 import (
    ExhaustionExtensionExitReplayConfig,
    run_exhaustion_extension_exit_replay_lab,
)


def _write_exhaustion_replay_inputs(tmp_path: Path) -> tuple[Path, Path]:
    event_dir = tmp_path / "sparse_exhaustion_extension_v0" / "run"
    filter_dir = tmp_path / "exhaustion_extension_regime_filter_scan_v0" / "run"
    event_dir.mkdir(parents=True)
    filter_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    for month in ("2026-01", "2026-02"):
        for index in range(24):
            strong = index % 3 != 0
            rows.append(
                {
                    "symbol": symbols[index % len(symbols)],
                    "timestamp": f"{month}-{1 + index // 4:02d}T15:{index % 4:02d}:00Z",
                    "session_date": f"{month}-{1 + index // 4:02d}",
                    "bar_index_in_session": 20 + index,
                    "event_state": "exhaustion_extension",
                    "expected_direction": -1,
                    "extension_x_time_regime": "upside_exhaustion|midday",
                    "role_rejection_wick": 0.7 if strong else 0.2,
                    "distance_from_session_high_pct": -0.001,
                    "distance_from_session_low_pct": 0.04,
                    "distance_from_recent_high_pct": -0.001,
                    "distance_from_recent_low_pct": 0.035,
                    "distance_from_opening_range_high_pct": -0.002,
                    "distance_from_opening_range_low_pct": 0.03,
                    "forward_12_bar_return": -0.012 if strong else 0.006,
                    "forward_12_bar_mfe": 0.002 if strong else 0.010,
                    "forward_12_bar_mae": -0.018 if strong else -0.003,
                }
            )
    for index in range(12):
        rows.append(
            {
                "symbol": symbols[index % len(symbols)],
                "timestamp": f"2026-03-{1 + index // 2:02d}T15:{index % 2:02d}:00Z",
                "session_date": f"2026-03-{1 + index // 2:02d}",
                "bar_index_in_session": 24 + index,
                "event_state": "exhaustion_extension",
                "expected_direction": -1,
                "extension_x_time_regime": "upside_exhaustion|midday",
                "role_rejection_wick": 0.8,
                "distance_from_session_high_pct": -0.001,
                "distance_from_session_low_pct": 0.04,
                "distance_from_recent_high_pct": -0.001,
                "distance_from_recent_low_pct": 0.035,
                "distance_from_opening_range_high_pct": -0.002,
                "distance_from_opening_range_low_pct": 0.03,
                "forward_12_bar_return": -0.012,
                "forward_12_bar_mfe": 0.002,
                "forward_12_bar_mae": -0.018,
            }
        )
    for index in range(12):
        rows.append(
            {
                "symbol": symbols[index % len(symbols)],
                "timestamp": f"2026-03-{8 + index // 2:02d}T15:{index % 2:02d}:00Z",
                "session_date": f"2026-03-{8 + index // 2:02d}",
                "bar_index_in_session": 40 + index,
                "event_state": "exhaustion_extension",
                "expected_direction": -1,
                "extension_x_time_regime": "upside_exhaustion|midday",
                "role_rejection_wick": 0.2,
                "distance_from_session_high_pct": -0.001,
                "distance_from_session_low_pct": 0.04,
                "distance_from_recent_high_pct": -0.001,
                "distance_from_recent_low_pct": 0.035,
                "distance_from_opening_range_high_pct": -0.002,
                "distance_from_opening_range_low_pct": 0.03,
                "forward_12_bar_return": 0.006,
                "forward_12_bar_mfe": 0.010,
                "forward_12_bar_mae": -0.003,
            }
        )
    pd.DataFrame(rows).to_csv(event_dir / "exhaustion_event_rows.csv", index=False)

    pd.DataFrame(
        [
            {
                "horizon": 12,
                "regime_field": "extension_x_time_regime",
                "regime_value": "upside_exhaustion|midday",
                "filter_rule": "role_rejection_wick >= 0.5",
                "feature": "role_rejection_wick",
                "operator": ">=",
                "threshold": 0.5,
                "retained_test_count": 12,
                "filtered_test_same_result_rate": 1.0,
                "test_median_lift_vs_exhaustion": 0.01,
                "verdict": "pass_exhaustion_regime_filter",
            }
        ]
    ).to_csv(filter_dir / "selected_exhaustion_regime_filter_results.csv", index=False)
    return event_dir, filter_dir


def test_exhaustion_exit_replay_uses_prior_rows_only_and_row_direction(
    tmp_path: Path,
) -> None:
    event_dir, filter_dir = _write_exhaustion_replay_inputs(tmp_path)

    result = run_exhaustion_extension_exit_replay_lab(
        input_exhaustion_event_dir=event_dir,
        input_filter_report_dir=filter_dir,
        output_dir=tmp_path / "out",
        config=ExhaustionExtensionExitReplayConfig(
            replay_months=("2026-03",),
            stop_models=("fixed_50bps",),
            target_r_multiples=(1.0,),
            min_train_events=20,
            min_train_symbols=3,
            min_train_months=2,
            min_total_trades=1,
            max_single_month_share=1.0,
            random_iterations=3,
        ),
    )

    selected = pd.read_csv(result.selected_monthly_candidates_csv_path)
    trades = pd.read_csv(result.trades_csv_path)
    assert result.decision == "continue_research_exhaustion_exit_replay"
    assert selected["train_end_timestamp"].iloc[0].startswith("2026-02")
    assert trades["expected_direction"].eq(-1).all()
    assert trades["session_date"].str.startswith("2026-03").all()
    assert trades["net_r"].sum() > 0


def test_exhaustion_exit_replay_writes_expected_files(tmp_path: Path) -> None:
    event_dir, filter_dir = _write_exhaustion_replay_inputs(tmp_path)

    result = run_exhaustion_extension_exit_replay_lab(
        input_exhaustion_event_dir=event_dir,
        input_filter_report_dir=filter_dir,
        output_dir=tmp_path / "out",
        config=ExhaustionExtensionExitReplayConfig(
            replay_months=("2026-03",),
            stop_models=("fixed_50bps",),
            target_r_multiples=(1.0,),
            min_train_events=20,
            min_train_symbols=3,
            min_train_months=2,
            min_total_trades=1,
            random_iterations=3,
        ),
    )

    for name in [
        "summary.md",
        "summary.json",
        "decision.json",
        "selected_filter_book.csv",
        "monthly_exit_sweep.csv",
        "selected_monthly_candidates.csv",
        "monthly_summary.csv",
        "random_monthly_baseline.csv",
        "signals.csv",
        "trades.csv",
        "missed_signals.csv",
        "daily_pnl.csv",
        "exhaustion_exit_summary.csv",
        "concentration_warnings.csv",
    ]:
        assert (result.output_dir / name).exists()


def test_exhaustion_exit_replay_tolerates_missing_structure_distance_columns(
    tmp_path: Path,
) -> None:
    event_dir, filter_dir = _write_exhaustion_replay_inputs(tmp_path)
    path = event_dir / "exhaustion_event_rows.csv"
    rows = pd.read_csv(path).drop(
        columns=["distance_from_recent_high_pct", "distance_from_recent_low_pct"]
    )
    rows.to_csv(path, index=False)

    result = run_exhaustion_extension_exit_replay_lab(
        input_exhaustion_event_dir=event_dir,
        input_filter_report_dir=filter_dir,
        output_dir=tmp_path / "out-missing-distance",
        config=ExhaustionExtensionExitReplayConfig(
            replay_months=("2026-03",),
            stop_models=("structure_recent_extreme_10bps",),
            target_r_multiples=(1.0,),
            min_train_events=20,
            min_train_symbols=3,
            min_train_months=2,
            min_total_trades=1,
            max_single_month_share=1.0,
            random_iterations=3,
        ),
    )

    assert result.summary_json_path.exists()
    assert result.decision == "reject_low_sample"


def test_exhaustion_exit_replay_cli_smoke(tmp_path: Path) -> None:
    event_dir, filter_dir = _write_exhaustion_replay_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "exhaustion-extension-exit-replay",
            "--input-exhaustion-event-dir",
            str(event_dir),
            "--input-filter-report-dir",
            str(filter_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--replay-months",
            "2026-03",
            "--stop-models",
            "fixed_50bps",
            "--target-r-multiples",
            "1",
            "--min-train-events",
            "20",
            "--min-train-symbols",
            "3",
            "--min-train-months",
            "2",
            "--min-total-trades",
            "1",
            "--random-iterations",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "exhaustion_extension_exit_replay_v0" in result.output
