from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.walk_forward_personality_filter_exit_v0 import (
    WalkForwardPersonalityFilterExitConfig,
    WalkForwardSelectedFilterExitConfig,
    run_walk_forward_personality_filter_exit_lab,
    run_walk_forward_selected_filter_exit_lab,
)


def _write_walk_forward_inputs(tmp_path: Path) -> tuple[Path, Path]:
    event_dir = tmp_path / "state_event_detector_v0" / "run"
    combined_dir = tmp_path / "combined_regime_personality_rank_v0" / "run"
    event_dir.mkdir(parents=True)
    combined_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    for month_index, month in enumerate(["2026-01", "2026-02"]):
        for index in range(24):
            strong = index % 3 != 0
            rows.append(
                {
                    "symbol": symbols[index % len(symbols)],
                    "timestamp": f"{month}-{1 + index // 4:02d}T14:{30 + index % 4:02d}:00Z",
                    "session_date": f"{month}-{1 + index // 4:02d}",
                    "bar_index_in_session": 10 + index,
                    "time_of_day_bucket": "morning",
                    "event_state": "controlled_pullback_after_bullish_impulse",
                    "distance_from_vwap_pct": 0.004,
                    "distance_from_session_low_pct": 0.006,
                    "distance_from_session_high_pct": -0.012,
                    "distance_from_recent_low_pct": 0.005,
                    "distance_from_recent_high_pct": -0.011,
                    "distance_from_opening_range_low_pct": 0.006,
                    "distance_from_opening_range_high_pct": -0.012,
                    "bar_return": 0.006 + month_index * 0.0001 if strong else -0.002,
                    "close_location_value": 0.72 if strong else 0.35,
                    "upper_wick_pct_of_range": 0.1,
                    "lower_wick_pct_of_range": 0.1,
                    "body_pct_of_range": 0.7,
                    "directional_efficiency_6": 0.6,
                    "directional_efficiency_12": 0.65,
                    "rolling_intraday_range_pct": 0.015,
                    "compression_zscore": 0.1,
                    "range_zscore": 0.2,
                    "relative_volume_at_bar_index": 1.0,
                    "relative_cumulative_volume": 1.0,
                    "forward_12_bar_return": 0.010 if strong else -0.004,
                    "forward_12_bar_mfe": 0.014 if strong else 0.002,
                    "forward_12_bar_mae": -0.001 if strong else -0.010,
                }
            )
    for index in range(10):
        rows.append(
            {
                "symbol": symbols[index % len(symbols)],
                "timestamp": f"2026-03-{1 + index // 2:02d}T14:{30 + index % 2:02d}:00Z",
                "session_date": f"2026-03-{1 + index // 2:02d}",
                "bar_index_in_session": 10 + index,
                "time_of_day_bucket": "morning",
                "event_state": "controlled_pullback_after_bullish_impulse",
                "distance_from_vwap_pct": 0.004,
                "distance_from_session_low_pct": 0.006,
                "distance_from_session_high_pct": -0.012,
                "distance_from_recent_low_pct": 0.005,
                "distance_from_recent_high_pct": -0.011,
                "distance_from_opening_range_low_pct": 0.006,
                "distance_from_opening_range_high_pct": -0.012,
                "bar_return": 0.50,
                "close_location_value": 0.75,
                "upper_wick_pct_of_range": 0.1,
                "lower_wick_pct_of_range": 0.1,
                "body_pct_of_range": 0.7,
                "directional_efficiency_6": 0.6,
                "directional_efficiency_12": 0.65,
                "rolling_intraday_range_pct": 0.015,
                "compression_zscore": 0.1,
                "range_zscore": 0.2,
                "relative_volume_at_bar_index": 1.0,
                "relative_cumulative_volume": 1.0,
                "forward_12_bar_return": 0.009,
                "forward_12_bar_mfe": 0.014,
                "forward_12_bar_mae": -0.001,
            }
        )
    for index in range(20):
        rows.append(
            {
                "symbol": symbols[index % len(symbols)],
                "timestamp": f"2026-03-{8 + index // 4:02d}T15:{30 + index % 4:02d}:00Z",
                "session_date": f"2026-03-{8 + index // 4:02d}",
                "bar_index_in_session": 30 + index,
                "time_of_day_bucket": "midday",
                "event_state": "dead_chop_blocker",
                "distance_from_vwap_pct": 0.0,
                "distance_from_session_low_pct": 0.002,
                "distance_from_session_high_pct": -0.002,
                "distance_from_recent_low_pct": 0.002,
                "distance_from_recent_high_pct": -0.002,
                "distance_from_opening_range_low_pct": 0.002,
                "distance_from_opening_range_high_pct": -0.002,
                "bar_return": -0.002,
                "close_location_value": 0.40,
                "upper_wick_pct_of_range": 0.1,
                "lower_wick_pct_of_range": 0.1,
                "body_pct_of_range": 0.4,
                "directional_efficiency_6": 0.1,
                "directional_efficiency_12": 0.1,
                "rolling_intraday_range_pct": 0.006,
                "compression_zscore": -0.6,
                "range_zscore": -0.5,
                "relative_volume_at_bar_index": 0.8,
                "relative_cumulative_volume": 0.8,
                "forward_12_bar_return": -0.006,
                "forward_12_bar_mfe": 0.001,
                "forward_12_bar_mae": -0.010,
            }
        )
    pd.DataFrame(rows).to_csv(event_dir / "event_rows.csv", index=False)

    pd.DataFrame(
        [
            {
                "event_state": "controlled_pullback_after_bullish_impulse",
                "horizon": 12,
                "regime_field": "time_x_vwap_regime",
                "regime_value": "morning|above",
                "direction": 1,
                "test_event_count": 10,
                "combined_regime_pass": True,
                "personality": "pullback_continuation",
                "combined_score": 10.0,
            }
        ]
    ).to_csv(combined_dir / "best_combined_regimes_by_personality.csv", index=False)
    return event_dir, combined_dir


def _write_selected_filter_report(tmp_path: Path) -> Path:
    report_dir = tmp_path / "combined_regime_filter_test_v0" / "run"
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "personality": "pullback_continuation",
                "event_state": "controlled_pullback_after_bullish_impulse",
                "horizon": 12,
                "regime_field": "time_x_vwap_regime",
                "regime_value": "morning|above",
                "filter_feature": "bar_return",
                "filter_operator": ">=",
                "filter_threshold": 0.0,
                "filter_rule": "bar_return >= 0",
                "selection_score": 10.0,
            }
        ]
    ).to_csv(report_dir / "selected_filters.csv", index=False)
    return report_dir


def _write_dead_chop_blocker_report(tmp_path: Path) -> Path:
    report_dir = tmp_path / "sidelined_personality_cross_regime_v0" / "run"
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "personality": "dead_chop_noise",
                "event_state": "dead_chop_blocker",
                "role": "no_trade_filter",
                "role_objective": "low_movement",
                "horizon": 9,
                "regime_field": "time_x_vwap_regime",
                "regime_value": "midday|near",
                "filter_rule": "rolling_intraday_range_pct <= 0.004",
                "feature": "rolling_intraday_range_pct",
                "operator": "<=",
                "threshold": 0.004,
                "retained_test_count": 4,
                "filtered_test_same_result_rate": 0.75,
                "excess_vs_random_same_count": 0.20,
                "verdict": "promote_for_retest",
            }
        ]
    ).to_csv(report_dir / "selected_sidelined_candidates.csv", index=False)
    return report_dir


def _append_same_timestamp_dead_chop_blockers(event_dir: Path) -> None:
    path = event_dir / "event_rows.csv"
    rows = pd.read_csv(path)
    march_signals = rows[
        rows["event_state"].eq("controlled_pullback_after_bullish_impulse")
        & rows["session_date"].astype(str).str.startswith("2026-03")
    ].head(3)
    blockers = march_signals.copy()
    blockers["event_state"] = "dead_chop_blocker"
    blockers["distance_from_vwap_pct"] = 0.0
    blockers["time_of_day_bucket"] = "morning"
    blockers["rolling_intraday_range_pct"] = 0.003
    blockers["directional_efficiency_12"] = 0.1
    blockers["forward_12_bar_return"] = 0.0002
    blockers["forward_12_bar_mfe"] = 0.0006
    blockers["forward_12_bar_mae"] = -0.0005
    pd.concat([rows, blockers], ignore_index=True).to_csv(path, index=False)


def test_walk_forward_rediscovers_filters_from_prior_months_only(tmp_path: Path) -> None:
    event_dir, combined_dir = _write_walk_forward_inputs(tmp_path)

    result = run_walk_forward_personality_filter_exit_lab(
        input_event_dir=event_dir,
        input_combined_regime_dir=combined_dir,
        output_dir=tmp_path / "out",
        config=WalkForwardPersonalityFilterExitConfig(
            replay_months=("2026-03",),
            filter_features=("bar_return",),
            stop_models=("fixed_50bps",),
            target_r_multiples=(1.0,),
            quantiles=(0.50,),
            min_train_events=20,
            min_train_symbols=3,
            min_train_months=2,
            min_replay_signals=1,
            min_total_trades=1,
            max_single_month_share=1.0,
            random_iterations=3,
        ),
    )

    selected = pd.read_csv(result.selected_monthly_candidates_csv_path)
    trades = pd.read_csv(result.trades_csv_path)
    assert result.decision == "continue_research_walk_forward_filter_exit"
    assert selected["month"].tolist() == ["2026-03"]
    assert selected["filter_feature"].iloc[0] == "bar_return"
    assert selected["filter_threshold"].iloc[0] < 0.10
    assert selected["train_end_timestamp"].iloc[0].startswith("2026-02")
    assert trades["session_date"].str.startswith("2026-03").all()
    assert trades["net_r"].sum() > 0


def test_walk_forward_report_writes_expected_files(tmp_path: Path) -> None:
    event_dir, combined_dir = _write_walk_forward_inputs(tmp_path)

    result = run_walk_forward_personality_filter_exit_lab(
        input_event_dir=event_dir,
        input_combined_regime_dir=combined_dir,
        output_dir=tmp_path / "out",
        config=WalkForwardPersonalityFilterExitConfig(
            replay_months=("2026-03",),
            filter_features=("bar_return",),
            stop_models=("fixed_50bps",),
            target_r_multiples=(1.0,),
            quantiles=(0.50,),
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
        "monthly_filter_candidates.csv",
        "monthly_exit_sweep.csv",
        "selected_monthly_candidates.csv",
        "monthly_summary.csv",
        "random_monthly_baseline.csv",
        "signals.csv",
        "trades.csv",
        "missed_signals.csv",
        "daily_pnl.csv",
        "personality_summary.csv",
    ]:
        assert (result.output_dir / name).exists()


def test_walk_forward_personality_filter_exit_cli_smoke(tmp_path: Path) -> None:
    event_dir, combined_dir = _write_walk_forward_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "walk-forward-personality-filter-exit",
            "--input-event-dir",
            str(event_dir),
            "--input-combined-regime-dir",
            str(combined_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--replay-months",
            "2026-03",
            "--filter-features",
            "bar_return",
            "--stop-models",
            "fixed_50bps",
            "--target-r-multiples",
            "1",
            "--quantiles",
            "0.5",
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
    assert "walk_forward_personality_filter_exit_v0" in result.output


def test_selected_filter_exit_uses_frozen_filters_and_prior_exit_selection(
    tmp_path: Path,
) -> None:
    event_dir, _ = _write_walk_forward_inputs(tmp_path)
    filter_dir = _write_selected_filter_report(tmp_path)

    result = run_walk_forward_selected_filter_exit_lab(
        input_event_dir=event_dir,
        input_filter_report_dir=filter_dir,
        output_dir=tmp_path / "selected-out",
        config=WalkForwardSelectedFilterExitConfig(
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
    assert result.decision == "continue_research_walk_forward_selected_filter_exit"
    assert selected["filter_rule"].tolist() == ["bar_return >= 0"]
    assert selected["train_end_timestamp"].iloc[0].startswith("2026-02")
    assert trades["session_date"].str.startswith("2026-03").all()
    assert trades["net_r"].sum() > 0


def test_selected_filter_exit_cli_smoke(tmp_path: Path) -> None:
    event_dir, _ = _write_walk_forward_inputs(tmp_path)
    filter_dir = _write_selected_filter_report(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "walk-forward-selected-filter-exit",
            "--input-event-dir",
            str(event_dir),
            "--input-filter-report-dir",
            str(filter_dir),
            "--output-dir",
            str(tmp_path / "selected-cli-out"),
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
            "--max-single-month-share",
            "1",
            "--random-iterations",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "walk_forward_selected_filter_exit_v0" in result.output


def test_selected_filter_exit_applies_dead_chop_blockers(tmp_path: Path) -> None:
    event_dir, _ = _write_walk_forward_inputs(tmp_path)
    _append_same_timestamp_dead_chop_blockers(event_dir)
    filter_dir = _write_selected_filter_report(tmp_path)
    blocker_dir = _write_dead_chop_blocker_report(tmp_path)

    result = run_walk_forward_selected_filter_exit_lab(
        input_event_dir=event_dir,
        input_filter_report_dir=filter_dir,
        output_dir=tmp_path / "selected-blocked-out",
        input_blocker_report_dir=blocker_dir,
        config=WalkForwardSelectedFilterExitConfig(
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

    blocked = pd.read_csv(result.blocked_signals_csv_path)
    blocker_summary = pd.read_csv(result.blocker_caveat_summary_csv_path)
    summary = pd.read_csv(result.monthly_summary_csv_path)
    assert len(blocked) == 3
    assert blocked["blocker_personality"].eq("dead_chop_noise").all()
    assert blocker_summary["blocked_signal_count"].sum() == 3
    assert summary["blocked_signal_count"].iloc[0] == 3
    assert summary["signal_count"].iloc[0] == 5
    assert result.trade_count == 5
