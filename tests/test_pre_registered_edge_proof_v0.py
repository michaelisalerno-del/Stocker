from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.pre_registered_edge_proof_v0 import (
    PreRegisteredEdgeProofConfig,
    run_pre_registered_edge_proof,
)


def _event_row(
    *,
    symbol: str,
    timestamp: str,
    session_date: str,
    forward_return: float,
    forward_mfe: float,
    forward_mae: float,
    event_state: str = "failed_bounce_active_liquidation",
    compression_zscore: float = 0.0,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "session_date": session_date,
        "bar_index_in_session": 20,
        "time_of_day_bucket": "midday",
        "event_state": event_state,
        "opening_mid_x_range_regime": "below|mid_range",
        "compression_zscore": compression_zscore,
        "bar_return": -0.002,
        "distance_from_session_high_pct": -0.01,
        "distance_from_recent_high_pct": -0.01,
        "close_location_value": 0.2,
        "upper_wick_pct_of_range": 0.1,
        "lower_wick_pct_of_range": 0.1,
        "relative_volume_at_bar_index": 1.0,
        "relative_cumulative_volume": 1.0,
        "same_direction_other_symbol_count_15m": 1,
        "same_personality_other_symbol_count_15m": 1,
        "same_direction_other_symbol_count_30m": 1,
        "same_personality_other_symbol_count_30m": 1,
        "forward_24_bar_return": forward_return,
        "forward_24_bar_mfe": forward_mfe,
        "forward_24_bar_mae": forward_mae,
    }


def _write_registration_inputs(tmp_path: Path, *, include_future: bool) -> tuple[Path, Path]:
    event_dir = tmp_path / "state_event_detector_v0" / "run"
    staged_dir = tmp_path / "staged" / "run"
    event_dir.mkdir(parents=True)
    staged_dir.mkdir(parents=True)

    event_rows = [
        _event_row(
            symbol="AAA",
            timestamp="2026-06-10T15:00:00Z",
            session_date="2026-06-10",
            forward_return=-0.006,
            forward_mfe=0.001,
            forward_mae=-0.006,
        )
    ]
    if include_future:
        event_rows.extend(
            [
                _event_row(
                    symbol="AAA",
                    timestamp="2026-07-02T15:00:00Z",
                    session_date="2026-07-02",
                    forward_return=-0.010,
                    forward_mfe=0.001,
                    forward_mae=-0.011,
                ),
                _event_row(
                    symbol="BBB",
                    timestamp="2026-07-03T15:00:00Z",
                    session_date="2026-07-03",
                    forward_return=-0.012,
                    forward_mfe=0.001,
                    forward_mae=-0.013,
                ),
                _event_row(
                    symbol="CCC",
                    timestamp="2026-07-05T15:00:00Z",
                    session_date="2026-07-05",
                    forward_return=-0.014,
                    forward_mfe=0.001,
                    forward_mae=-0.015,
                ),
            ]
        )
    pd.DataFrame(event_rows).to_csv(event_dir / "event_rows.csv", index=False)

    pd.DataFrame(
        [
            {
                "personality": "active_liquidation",
                "event_state": "failed_bounce_active_liquidation",
                "horizon": 24,
                "expected_direction": -1,
                "regime_field": "opening_mid_x_range_regime",
                "regime_value": "below|mid_range",
                "filter_feature": "compression_zscore",
                "filter_operator": ">=",
                "filter_threshold": -0.614603,
                "filter_rule": "compression_zscore >= -0.614603",
                "rule_kind": "single",
                "feature_b": "",
                "operator_b": "",
                "threshold_b": "",
                "selection_score": 28.7,
            }
        ]
    ).to_csv(staged_dir / "mixed_regime_filter_book.csv", index=False)
    pd.DataFrame(
        [
            {
                "month": "2026-06",
                "selected_filter_rank": 0,
                "personality": "active_liquidation",
                "event_state": "failed_bounce_active_liquidation",
                "horizon": 24,
                "expected_direction": -1,
                "regime_field": "opening_mid_x_range_regime",
                "regime_value": "below|mid_range",
                "filter_feature": "compression_zscore",
                "filter_operator": ">=",
                "filter_threshold": -0.614603,
                "filter_rule": "compression_zscore >= -0.614603",
                "filter_selection_score": 28.7,
                "train_end_timestamp": "2026-06-29T20:00:00+00:00",
                "stop_model": "fixed_50bps",
                "target_r": 2.0,
                "eligible_symbols": "AAA|BBB",
                "eligible_symbol_count": 2,
                "train_exit_count": 60,
                "train_exit_total_net_r": 12.0,
                "train_exit_mean_net_r": 0.2,
                "train_exit_win_rate": 0.62,
                "exit_selection_score": 0.5,
                "monthly_candidate_rank": 0,
            }
        ]
    ).to_csv(staged_dir / "selected_monthly_candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "caveat_rule_id": 0,
                "month": "2026-05",
                "rule_name": "stale old caveat",
                "rule_family": "train_selected_numeric",
                "strict_status": "train_selected_staged_supported",
                "feature": "relative_volume_at_bar_index",
                "operator": ">=",
                "selected_threshold": 99.0,
            }
        ]
    ).to_csv(staged_dir / "caveat_rule_book.csv", index=False)
    return event_dir, staged_dir


def test_pre_registered_edge_proof_waits_when_no_future_rows(tmp_path: Path) -> None:
    event_dir, staged_dir = _write_registration_inputs(tmp_path, include_future=False)

    result = run_pre_registered_edge_proof(
        input_event_dir=event_dir,
        input_staged_report_dir=staged_dir,
        output_dir=tmp_path / "out",
        config=PreRegisteredEdgeProofConfig(
            registration_cutoff_month="2026-06",
            evaluation_months=("2026-07",),
            random_iterations=5,
        ),
    )

    summary = json.loads(result.summary_json_path.read_text())
    registration = json.loads(result.registration_json_path.read_text())
    frozen = pd.read_csv(result.frozen_candidates_csv_path)

    assert result.decision == "registered_waiting_for_future_data"
    assert summary["edge_claimed"] is False
    assert summary["order_placement"] == "disabled"
    assert registration["registration_cutoff_month"] == "2026-06"
    assert registration["frozen_candidate_count"] == 1
    assert frozen["source_month"].tolist() == ["2026-06"]
    assert frozen["personality"].tolist() == ["active_liquidation"]


def test_pre_registered_edge_proof_evaluates_future_without_reselection(
    tmp_path: Path,
) -> None:
    event_dir, staged_dir = _write_registration_inputs(tmp_path, include_future=True)

    result = run_pre_registered_edge_proof(
        input_event_dir=event_dir,
        input_staged_report_dir=staged_dir,
        output_dir=tmp_path / "out",
        config=PreRegisteredEdgeProofConfig(
            registration_cutoff_month="2026-06",
            evaluation_months=("2026-07",),
            min_forward_trades=1,
            random_iterations=5,
            random_seed=7,
        ),
    )

    trades = pd.read_csv(result.evaluation_trades_csv_path)
    monthly = pd.read_csv(result.evaluation_monthly_summary_csv_path)
    caveats = pd.read_csv(result.frozen_caveats_csv_path)

    assert set(trades["symbol"]) == {"AAA", "BBB"}
    assert "CCC" not in set(trades["symbol"])
    assert trades["month"].tolist() == ["2026-07", "2026-07"]
    assert monthly.loc[0, "source_month"] == "2026-06"
    assert monthly.loc[0, "selected_candidate_count"] == 1
    assert caveats.empty


def test_pre_registered_edge_proof_cli_smoke(tmp_path: Path) -> None:
    event_dir, staged_dir = _write_registration_inputs(tmp_path, include_future=False)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "pre-registered-edge-proof",
            "--input-event-dir",
            str(event_dir),
            "--input-staged-report-dir",
            str(staged_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--registration-cutoff-month",
            "2026-06",
            "--evaluation-months",
            "2026-07",
            "--random-iterations",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "pre_registered_edge_proof_v0" in result.output
    run_dirs = sorted((tmp_path / "cli-out").glob("pre_registered_edge_proof_v0_*"))
    assert run_dirs
