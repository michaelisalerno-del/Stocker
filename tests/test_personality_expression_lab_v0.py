from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.personality_expression_lab_v0 import (
    PersonalityExpressionLabConfig,
    run_personality_expression_lab,
)


def _event_row(
    *,
    symbol: str,
    timestamp: str,
    session_date: str,
    event_state: str,
    direction: int,
    forward_win: bool,
    directional_efficiency_6: float = 0.25,
    distance_from_opening_mid: float = 0.006,
) -> dict[str, object]:
    if direction > 0:
        forward_return = 0.02 if forward_win else -0.01
        forward_mfe = 0.02 if forward_win else 0.002
        forward_mae = -0.002 if forward_win else -0.012
    else:
        forward_return = -0.02 if forward_win else 0.01
        forward_mfe = 0.002 if forward_win else 0.012
        forward_mae = -0.02 if forward_win else -0.002
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "session_date": session_date,
        "bar_index_in_session": 14,
        "bar_index_bucket": "morning",
        "time_of_day_bucket": "morning",
        "event_state": event_state,
        "open": 10.0,
        "high": 10.2,
        "low": 9.9,
        "close": 10.1,
        "volume": 1000,
        "bar_return": -0.004 if direction < 0 else 0.004,
        "bar_range_pct": 0.03,
        "body_pct_of_range": 0.5,
        "close_location_value": 0.5,
        "upper_wick_pct_of_range": 0.2,
        "lower_wick_pct_of_range": 0.2,
        "prior_3_bar_return": -0.004,
        "prior_6_bar_return": -0.006,
        "prior_12_bar_return": -0.010,
        "directional_efficiency_3": 0.25,
        "directional_efficiency_6": directional_efficiency_6,
        "directional_efficiency_12": 0.30,
        "distance_from_vwap_pct": 0.0,
        "distance_from_opening_range_mid_pct": distance_from_opening_mid,
        "distance_from_opening_range_high_pct": -0.010,
        "distance_from_opening_range_low_pct": 0.010,
        "distance_from_session_open_pct": 0.006,
        "distance_from_session_high_pct": -0.015,
        "distance_from_session_low_pct": 0.012,
        "distance_from_recent_high_pct": -0.020,
        "distance_from_recent_low_pct": 0.012,
        "rolling_intraday_range_pct": 0.014,
        "compression_zscore": 0.0,
        "range_zscore": 0.0,
        "return_zscore": 0.0,
        "relative_volume_at_bar_index": 1.0,
        "relative_cumulative_volume": 1.0,
        "forward_6_bar_return": forward_return,
        "forward_6_bar_mfe": forward_mfe,
        "forward_6_bar_mae": forward_mae,
        "forward_6_bar_abs_return": abs(forward_return),
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    event_dir = tmp_path / "events"
    discovery_dir = tmp_path / "discovery"
    event_dir.mkdir()
    discovery_dir.mkdir()

    rows: list[dict[str, object]] = []
    train_months = ["2026-01", "2026-02", "2026-03", "2026-04"]
    test_months = ["2026-05", "2026-06"]
    for month_index, month in enumerate(train_months):
        for index in range(2):
            symbol = "AAA" if index == 0 else "BBB"
            day = 1 + month_index * 2 + index
            slow_day = day + 10
            open_day = day + 20
            rows.append(
                _event_row(
                    symbol=symbol,
                    timestamp=f"{month}-{day:02d}T14:30:00Z",
                    session_date=f"{month}-{day:02d}",
                    event_state="failed_bounce_active_liquidation",
                    direction=-1,
                    forward_win=True,
                )
            )
            rows.append(
                _event_row(
                    symbol=symbol,
                    timestamp=f"{month}-{slow_day:02d}T15:00:00Z",
                    session_date=f"{month}-{slow_day:02d}",
                    event_state="slow_snapback_after_dip",
                    direction=1,
                    forward_win=True,
                )
            )
            rows.append(
                _event_row(
                    symbol=symbol,
                    timestamp=f"{month}-{open_day:02d}T15:30:00Z",
                    session_date=f"{month}-{open_day:02d}",
                    event_state="failed_open_down_continuation",
                    direction=-1,
                    forward_win=True,
                )
            )
    for month_index, month in enumerate(test_months):
        for index in range(2):
            symbol = "AAA" if index == 0 else "BBB"
            day = 1 + month_index * 2 + index
            slow_day = day + 10
            rows.append(
                _event_row(
                    symbol=symbol,
                    timestamp=f"{month}-{day:02d}T14:30:00Z",
                    session_date=f"{month}-{day:02d}",
                    event_state="failed_bounce_active_liquidation",
                    direction=-1,
                    forward_win=index == 0,
                )
            )
            rows.append(
                _event_row(
                    symbol=symbol,
                    timestamp=f"{month}-{slow_day:02d}T15:00:00Z",
                    session_date=f"{month}-{slow_day:02d}",
                    event_state="slow_snapback_after_dip",
                    direction=1,
                    forward_win=True,
                )
            )
    pd.DataFrame(rows).to_csv(event_dir / "event_rows.csv", index=False)

    rules = pd.DataFrame(
        [
            {
                "personality": "active_liquidation",
                "horizon": 6,
                "regime_field": "time_regime",
                "regime_value": "morning",
                "filter_rule": "directional_efficiency_6 <= 0.4",
                "rule_kind": "single",
                "feature": "directional_efficiency_6",
                "operator": "<=",
                "threshold": 0.4,
            },
            {
                "personality": "slow_repair",
                "horizon": 6,
                "regime_field": "opening_mid_side_regime",
                "regime_value": "above",
                "filter_rule": "directional_efficiency_6 <= 0.4",
                "rule_kind": "single",
                "feature": "directional_efficiency_6",
                "operator": "<=",
                "threshold": 0.4,
            },
            {
                "personality": "open_down_pressure",
                "horizon": 6,
                "regime_field": "time_regime",
                "regime_value": "morning",
                "filter_rule": "directional_efficiency_6 <= 0.4",
                "rule_kind": "single",
                "feature": "directional_efficiency_6",
                "operator": "<=",
                "threshold": 0.4,
            },
        ]
    )
    rules.to_csv(discovery_dir / "passed_personality_rules.csv", index=False)
    return event_dir, discovery_dir


def test_personality_expression_lab_selects_train_supported_expressions(tmp_path: Path) -> None:
    event_dir, discovery_dir = _write_inputs(tmp_path)

    result = run_personality_expression_lab(
        input_event_dir=event_dir,
        input_personality_discovery_dir=discovery_dir,
        output_dir=tmp_path / "out",
        config=PersonalityExpressionLabConfig(
            train_months=("2026-01", "2026-02", "2026-03", "2026-04"),
            test_months=("2026-05", "2026-06"),
            stop_models=("fixed_100bps",),
            target_r_multiples=(1.0,),
            min_train_trades=4,
            min_train_months=2,
            min_train_total_net_r=0.0,
            min_train_win_rate=0.55,
            max_expressions_per_personality=1,
            min_oos_trades=1,
        ),
    )

    selected = pd.read_csv(result.selected_expressions_csv_path)
    test_trades = pd.read_csv(result.test_trades_csv_path)
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    assert set(selected["personality"]) == {"active_liquidation", "slow_repair"}
    assert "open_down_pressure" not in set(selected["personality"])
    assert set(test_trades["personality"]) == {"active_liquidation", "slow_repair"}
    assert result.decision == "continue_research_personality_expression_lab"
    assert summary["research_only"] is True
    assert summary["edge_claimed"] is False
    assert summary["test_trade_count"] == len(test_trades)
    assert summary["test_total_net_r"] > 0


def test_personality_expression_lab_cli_smoke(tmp_path: Path) -> None:
    event_dir, discovery_dir = _write_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "personality-expression-lab",
            "--input-event-dir",
            str(event_dir),
            "--input-personality-discovery-dir",
            str(discovery_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--train-months",
            "2026-01,2026-02,2026-03,2026-04",
            "--test-months",
            "2026-05,2026-06",
            "--stop-models",
            "fixed_100bps",
            "--target-r-multiples",
            "1",
            "--min-train-trades",
            "4",
            "--min-train-months",
            "2",
            "--min-oos-trades",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "personality_expression_lab_v0" in result.output
    run_dirs = sorted((tmp_path / "cli-out").glob("personality_expression_lab_v0_*"))
    assert run_dirs
    summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "continue_research_personality_expression_lab"
