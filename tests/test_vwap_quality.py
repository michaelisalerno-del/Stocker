import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.vwap_quality import (
    add_vwap_quality_features,
    build_bucket_summary,
    build_vwap_quality_report,
    reconstruct_round_trip_trades,
)


def _feature_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 14:30", periods=8, freq="5min", tz="UTC")
    close = [99.0, 101.0, 102.0, 98.0, 99.0, 101.0, 100.5, 102.0]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_date": ["2024-01-02"] * 8,
            "bar_index_in_session": list(range(8)),
            "minutes_from_session_open": [i * 5.0 for i in range(8)],
            "minutes_to_session_close": [390.0 - i * 5.0 for i in range(8)],
            "close": close,
            "session_vwap": [100.0] * 8,
            "distance_from_vwap": [(value / 100.0) - 1.0 for value in close],
            "relative_volume_at_bar_index": [1.0] * 8,
            "relative_cumulative_volume": [1.0, 1.05, 1.10, 1.20, 1.15, 1.30, 1.25, 1.40],
            "opening_range_high": [101.0] * 8,
            "opening_range_low": [99.0] * 8,
            "opening_range_mid": [100.0] * 8,
            "opening_range_width": [2.0] * 8,
            "session_complete_warning": [False] * 8,
        }
    )


def test_vwap_cross_count_bars_since_cross_and_reclaim_count() -> None:
    enriched = add_vwap_quality_features(_feature_frame(), near_vwap_threshold=0.001)

    assert enriched["vwap_cross_count_so_far"].tolist() == [0, 1, 1, 2, 2, 3, 3, 3]
    assert enriched["bars_since_last_vwap_cross"].tolist() == [0, 0, 1, 0, 1, 0, 1, 2]
    assert enriched["vwap_reclaim_count_so_far"].tolist() == [0, 1, 1, 1, 1, 2, 2, 2]


def test_near_vwap_percentage_and_slopes_are_entry_time_features() -> None:
    frame = _feature_frame()
    frame["session_vwap"] = [100.0, 100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0]
    frame["distance_from_vwap"] = pd.Series(frame["close"]) / frame["session_vwap"] - 1.0

    enriched = add_vwap_quality_features(frame, near_vwap_threshold=0.02, near_vwap_window=3)

    assert enriched.loc[2, "near_vwap_pct_30_bars"] == 1.0
    assert round(float(enriched.loc[5, "vwap_slope_3_bars"]), 6) == round(102.0 / 100.5 - 1.0, 6)
    assert round(float(enriched.loc[5, "close_momentum_3_bars"]), 6) == round(
        101.0 / 102.0 - 1.0, 6
    )


def test_no_lookahead_future_mutation_does_not_change_entry_features() -> None:
    frame = _feature_frame()
    original = add_vwap_quality_features(frame)
    mutated = frame.copy()
    mutated.loc[6:, ["close", "session_vwap", "distance_from_vwap"]] = [500.0, 400.0, 0.25]
    changed = add_vwap_quality_features(mutated)

    columns = [
        "vwap_cross_count_so_far",
        "bars_since_last_vwap_cross",
        "vwap_reclaim_count_so_far",
        "near_vwap_pct_30_bars",
        "vwap_slope_3_bars",
        "close_momentum_3_bars",
    ]
    pd.testing.assert_frame_equal(original.loc[:5, columns], changed.loc[:5, columns])


def test_reconstruct_round_trip_trades_from_target_positions() -> None:
    frame = _feature_frame().iloc[:5].copy()
    frame["close"] = [100.0, 101.0, 103.0, 102.0, 104.0]
    positions = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0])
    signals = pd.DataFrame(
        {
            "entry_mode": ["", "reclaim", "", "", ""],
            "time_stop_exit": [False, False, False, True, False],
            "vwap_lost_exit": [False] * 5,
        }
    )

    trades = reconstruct_round_trip_trades(
        frame,
        positions,
        signals=signals,
        cost_bps=10.0,
        symbol="AAPL.US",
        experiment_id="exp",
        parameter_set_id="ps_0001",
        exit_mode="time_stop",
    )

    assert len(trades) == 1
    trade = trades[0]
    assert trade["entry_mode"] == "reclaim"
    assert trade["holding_bars"] == 2
    assert round(float(trade["trade_gross_return"]), 6) == round(
        (103.0 / 101.0 - 1.0) + (102.0 / 103.0 - 1.0), 6
    )
    assert round(float(trade["trade_net_return"]), 6) == round(
        float(trade["trade_gross_return"]) - 0.002, 6
    )
    assert trade["win_loss"] == "win"


def test_bucket_summary_calculates_outcome_statistics() -> None:
    trades = pd.DataFrame(
        [
            {
                "entry_mode": "reclaim",
                "trade_gross_return": 0.02,
                "trade_net_return": 0.01,
                "holding_bars": 2,
                "time_of_day_bucket": "opening",
            },
            {
                "entry_mode": "reclaim",
                "trade_gross_return": -0.01,
                "trade_net_return": -0.02,
                "holding_bars": 4,
                "time_of_day_bucket": "opening",
            },
            {
                "entry_mode": "rejection",
                "trade_gross_return": -0.03,
                "trade_net_return": -0.04,
                "holding_bars": 3,
                "time_of_day_bucket": "midday",
            },
        ]
    )

    summary = build_bucket_summary(trades)
    reclaim = summary[(summary["feature"] == "entry_mode") & (summary["bucket"] == "reclaim")].iloc[
        0
    ]

    assert int(reclaim["trade_count"]) == 2
    assert float(reclaim["win_rate"]) == 0.5
    assert float(reclaim["median_holding_bars"]) == 3.0
    assert float(reclaim["top_5_winner_contribution_share"]) == 1.0


def _write_quality_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02 14:30", periods=20, freq="5min", tz="UTC"),
            "open": [100.0, 99.0, 98.0, 101.0, 102.0] * 4,
            "high": [100.0, 99.0, 98.0, 101.0, 102.0] * 4,
            "low": [100.0, 99.0, 98.0, 101.0, 102.0] * 4,
            "close": [100.0, 99.0, 98.0, 101.0, 102.0] * 4,
            "volume": [1000.0] * 20,
        }
    )
    key = DatasetKey(
        source="manual",
        instrument_type="stock",
        symbol="AAPL.US",
        timeframe="5m",
    )
    write_parquet(frame, dataset_path(key, data_dir=data_dir))
    report_dir = data_dir / "reports" / "research"
    report_dir.mkdir(parents=True)
    experiment_id = "20240101000000_vwap_reclaim_rejection_intraday_session_flat_v1_AAPL.US_5m"
    report_path = report_dir / f"{experiment_id}.json"
    report_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "symbol": "AAPL.US",
                "timeframe": "5m",
                "source": "manual",
                "instrument_type": "stock",
                "hypothesis": {"template": "vwap_reclaim_rejection"},
                "cost_assumptions": {"spread_bps": 3.0, "commission_bps": 0.5, "slippage_bps": 2.0},
                "holding_policy": {
                    "preferred_style": "intraday",
                    "allow_intraday": True,
                    "allow_overnight": False,
                    "allow_weekend": False,
                    "max_holding_sessions": 1,
                    "flatten_before_close_minutes": 0,
                    "entry_cutoff_before_close_minutes": 0,
                },
                "splits": [
                    {
                        "split_id": "split_001",
                        "train_start": 0,
                        "train_end": 5,
                        "test_start": 5,
                        "test_end": 20,
                    }
                ],
                "grid_results": [
                    {
                        "parameter_set_id": "ps_0001",
                        "params": {
                            "entry_mode": "reclaim",
                            "entry_buffer_bps": 0,
                            "min_reclaim_distance_bps": 0,
                            "reclaim_lookback_bars": 3,
                            "rejection_lookback_bars": 3,
                            "max_reclaim_distance_from_vwap_pct": 0.10,
                            "max_rejection_distance_from_vwap_pct": 0.02,
                            "min_bounce_distance_bps": 0,
                            "min_bars_after_open": 1,
                            "max_hold_bars": 3,
                            "min_relative_volume": 0.0,
                            "exit_mode": "time_stop",
                            "relative_volume_lookback_sessions": 0,
                            "entry_cutoff_before_close_minutes": 0,
                            "flatten_before_close_minutes": 0,
                            "market_calendar": None,
                        },
                    }
                ],
                "selected_result": {"parameter_set_id": "ps_0001"},
                "robustness_diagnostics": {"robustness_flags": ["failed_cost_stress"]},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    return data_dir, report_path


def test_build_vwap_quality_report_writes_expected_files(tmp_path: Path) -> None:
    data_dir, report_path = _write_quality_fixture(tmp_path)

    result = build_vwap_quality_report(
        reports_dir=data_dir / "reports" / "research",
        output_dir=data_dir / "reports" / "research" / "quality",
        report_paths=[report_path],
        data_dir=data_dir,
        timeframe="5m",
        source="manual",
        market_calendar=None,
    )

    assert result.summary_json_path.exists()
    assert result.summary_markdown_path.exists()
    assert result.trade_attribution_csv_path.exists()
    assert result.feature_bucket_summary_csv_path.exists()
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert summary["trade_count"] > 0
    assert summary["report_count_analyzed"] == 1


def test_vwap_quality_cli_smoke(tmp_path: Path) -> None:
    data_dir, report_path = _write_quality_fixture(tmp_path)
    output_dir = data_dir / "reports" / "research" / "quality_cli"

    result = CliRunner().invoke(
        app,
        [
            "research",
            "vwap-quality",
            "--reports-dir",
            str(data_dir / "reports" / "research"),
            "--report-path",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--data-dir",
            str(data_dir),
            "--timeframe",
            "5m",
            "--source",
            "manual",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "trade_attribution.csv" in result.output
