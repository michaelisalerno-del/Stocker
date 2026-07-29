from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_data.storage import DatasetKey, dataset_path
from stocker_research.state_lifecycle_context_lab_v0 import (
    StateLifecycleContextConfig,
    _scan_numeric_features,
    _write_csv,
    run_state_lifecycle_context_lab,
)


def _trade_row(
    *,
    symbol: str,
    session_date: str,
    split_month: str,
    personality: str,
    event_state: str,
    net_r: float,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": f"{session_date}T15:20:00Z",
        "session_date": session_date,
        "bar_index_in_session": 10,
        "personality": personality,
        "event_state": event_state,
        "expected_direction": -1,
        "month": split_month,
        "net_r": net_r,
    }


def _dense_rows(*, symbol: str, session_date: str, compressed: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(12):
        timestamp = pd.Timestamp(f"{session_date}T14:30:00Z") + pd.Timedelta(
            minutes=index * 5
        )
        prior = index < 10
        compression = "compressed" if compressed or not prior else "expanded"
        efficiency = "choppy_efficiency" if compressed or not prior else "directional_efficiency"
        zscore = 0.0 if compressed or not prior else (0.0 if index % 2 == 0 else 2.0)
        rows.append(
            {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "session_date": session_date,
                "bar_index_in_session": index,
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1000,
                "vwap_side_regime": "below",
                "opening_mid_side_regime": "below",
                "session_open_side_regime": "below",
                "range_regime": "high_range",
                "compression_regime": compression,
                "efficiency_regime": efficiency,
                "relative_volume_regime": "normal_relative_volume",
                "time_regime": "morning",
                "vwap_x_efficiency_regime": f"below|{efficiency}",
                "vwap_x_range_regime": "below|high_range",
                "compression_x_efficiency_regime": f"{compression}|{efficiency}",
                "opening_mid_x_range_regime": "below|high_range",
                "time_x_vwap_regime": "morning|below",
                "volume_x_vwap_regime": "normal_relative_volume|below",
                "distance_from_vwap_pct": -0.01,
                "distance_from_opening_range_mid_pct": -0.01,
                "distance_from_session_open_pct": -0.01,
                "rolling_intraday_range_pct": 0.015,
                "compression_zscore": zscore,
                "range_zscore": 0.0,
                "return_zscore": 0.0,
                "relative_volume_at_bar_index": 1.0,
                "relative_cumulative_volume": 1.0,
                "directional_efficiency_6": 0.2 if compressed else 0.8,
                "directional_efficiency_12": 0.2 if compressed else 0.8,
                "vwap_cross_count_12": 0,
                "range_cross_count_12": 0,
                "bar_return": -0.001,
                "bar_range_pct": 0.01,
                "close_location_value": 0.4,
            }
        )
    return rows


def _write_dataset(data_dir: Path, symbol: str, rows: list[dict[str, object]]) -> None:
    path = dataset_path(
        DatasetKey(source="eodhd", instrument_type="stock", symbol=symbol, timeframe="5m"),
        data_dir=data_dir,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_context_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    report_dir = tmp_path / "expression_report"
    event_dir = tmp_path / "events"
    data_dir = tmp_path / "data"
    report_dir.mkdir()
    event_dir.mkdir()

    train_rows = [
        _trade_row(
            symbol="AAA",
            session_date="2026-01-02",
            split_month="2026-01",
            personality="active_liquidation",
            event_state="failed_bounce_active_liquidation",
            net_r=1.0,
        ),
        _trade_row(
            symbol="AAA",
            session_date="2026-02-03",
            split_month="2026-02",
            personality="active_liquidation",
            event_state="failed_bounce_active_liquidation",
            net_r=0.8,
        ),
        _trade_row(
            symbol="BBB",
            session_date="2026-03-04",
            split_month="2026-03",
            personality="active_liquidation",
            event_state="failed_bounce_active_liquidation",
            net_r=-0.6,
        ),
        _trade_row(
            symbol="BBB",
            session_date="2026-04-05",
            split_month="2026-04",
            personality="active_liquidation",
            event_state="failed_bounce_active_liquidation",
            net_r=-0.4,
        ),
    ]
    test_rows = [
        _trade_row(
            symbol="AAA",
            session_date="2026-05-06",
            split_month="2026-05",
            personality="active_liquidation",
            event_state="failed_bounce_active_liquidation",
            net_r=1.1,
        ),
        _trade_row(
            symbol="BBB",
            session_date="2026-06-07",
            split_month="2026-06",
            personality="active_liquidation",
            event_state="failed_bounce_active_liquidation",
            net_r=-0.5,
        ),
    ]
    pd.DataFrame(train_rows).to_csv(report_dir / "train_trades.csv", index=False)
    pd.DataFrame(test_rows).to_csv(report_dir / "test_trades.csv", index=False)

    event_rows: list[dict[str, object]] = []
    for row in train_rows + test_rows:
        good = row["symbol"] == "AAA"
        if good:
            for bar in (6, 8):
                event_rows.append(
                    {
                        "symbol": row["symbol"],
                        "timestamp": f"{row['session_date']}T15:{bar * 5:02d}:00Z",
                        "session_date": row["session_date"],
                        "bar_index_in_session": bar,
                        "event_state": "failed_open_down_continuation",
                        "event_direction": "down",
                        "event_confidence_score": 1.0,
                    }
                )
        event_rows.append(
            {
                "symbol": row["symbol"],
                "timestamp": row["timestamp"],
                "session_date": row["session_date"],
                "bar_index_in_session": row["bar_index_in_session"],
                "event_state": row["event_state"],
                "event_direction": "down",
                "event_confidence_score": 1.0,
            }
        )
    pd.DataFrame(event_rows).to_csv(event_dir / "event_rows.csv", index=False)

    rows_by_symbol: dict[str, list[dict[str, object]]] = {"AAA": [], "BBB": []}
    for row in train_rows + test_rows:
        rows_by_symbol[str(row["symbol"])].extend(
            _dense_rows(
                symbol=str(row["symbol"]),
                session_date=str(row["session_date"]),
                compressed=row["symbol"] == "AAA",
            )
        )
    for symbol, rows in rows_by_symbol.items():
        _write_dataset(data_dir, symbol, rows)
    return report_dir, event_dir, data_dir


def test_state_lifecycle_context_lab_finds_prior_regime_and_cluster_candidates(
    tmp_path: Path,
) -> None:
    report_dir, event_dir, data_dir = _write_context_inputs(tmp_path)

    result = run_state_lifecycle_context_lab(
        input_expression_report_dir=report_dir,
        input_event_dir=event_dir,
        data_dir=data_dir,
        output_dir=tmp_path / "out",
        config=StateLifecycleContextConfig(
            lookback_bars=(6,),
            min_train_count=2,
            min_oos_count=1,
            market_calendar=None,
        ),
    )

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    numeric = pd.read_csv(result.prior_regime_numeric_scan_csv_path)
    clusters = pd.read_csv(result.prior_event_cluster_scan_csv_path)
    trade_context = pd.read_csv(result.trade_context_features_csv_path)

    assert result.decision == "continue_research_state_lifecycle_context"
    assert summary["research_only"] is True
    assert summary["edge_claimed"] is False
    assert summary["order_placement"] == "disabled"
    assert (
        numeric["classification"].eq("train_and_oos_candidate")
        & numeric["feature"].str.contains("compression_zscore_std")
    ).any()
    assert (
        clusters["classification"].eq("train_and_oos_candidate")
        & clusters["feature"].eq("cluster_down_pressure_count_6")
    ).any()
    assert {
        "shadow_candidate_index",
        "planned_exit_shadow_net_r",
        "prior_20_shadow_candidate_net_r",
        "prior_20_weak_context_share",
        "weak_cluster_shadow_deterioration_trigger",
    }.issubset(set(trade_context.columns))


def test_state_lifecycle_context_lab_cli_smoke(tmp_path: Path) -> None:
    report_dir, event_dir, data_dir = _write_context_inputs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "state-lifecycle-context",
            "--input-expression-report-dir",
            str(report_dir),
            "--input-event-dir",
            str(event_dir),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "cli-out"),
            "--lookback-bars",
            "6",
            "--min-train-count",
            "2",
            "--min-oos-count",
            "1",
            "--market-calendar",
            "none",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "state_lifecycle_context_lab_v0" in result.output
    run_dirs = sorted((tmp_path / "cli-out").glob("state_lifecycle_context_lab_v0_*"))
    assert run_dirs
    summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "continue_research_state_lifecycle_context"


def test_numeric_scan_counts_match_csv_replayed_threshold_flags(tmp_path: Path) -> None:
    feature = "prev_12_compression_zscore_std"
    lower = 0.10465262806285
    csv_collapsing_upper = 0.10465262806285002
    rows = []
    for split in ("train", "test"):
        rows.extend(
            [
                {
                    "split": split,
                    "symbol": "AAA",
                    "month": "2026-01" if split == "train" else "2026-05",
                    "personality": "active_liquidation",
                    "net_r": 1.0,
                    feature: lower,
                },
                {
                    "split": split,
                    "symbol": "BBB",
                    "month": "2026-01" if split == "train" else "2026-05",
                    "personality": "active_liquidation",
                    "net_r": 1.0,
                    feature: csv_collapsing_upper,
                },
                {
                    "split": split,
                    "symbol": "CCC",
                    "month": "2026-01" if split == "train" else "2026-05",
                    "personality": "active_liquidation",
                    "net_r": -1.0,
                    feature: 0.2,
                },
            ]
        )
    trades = pd.DataFrame(rows)

    scan = _scan_numeric_features(
        trades,
        feature_columns=[feature],
        family="prior_regime_numeric",
        config=StateLifecycleContextConfig(min_train_count=1, min_oos_count=1),
    )
    boundary_row = scan[
        scan["personality"].eq("active_liquidation")
        & scan["feature"].eq(feature)
        & scan["operator"].eq("<=")
        & scan["threshold"].eq(lower)
    ].iloc[0]

    csv_path = tmp_path / "trade_context_features.csv"
    _write_csv(csv_path, trades)
    reloaded = pd.read_csv(csv_path)
    replay_mask = (
        reloaded["personality"].eq("active_liquidation")
        & (pd.to_numeric(reloaded[feature], errors="coerce") <= float(boundary_row["threshold"]))
    )

    assert int(boundary_row["train_count"]) == int(
        (replay_mask & reloaded["split"].eq("train")).sum()
    )
    assert int(boundary_row["test_count"]) == int(
        (replay_mask & reloaded["split"].eq("test")).sum()
    )
