from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.state_event_detector_v0 import (
    StateEventDetectorConfig,
    audit_manual_state_examples,
    build_decision,
    detect_state_events,
    run_same_event_cross_symbol_similarity,
    run_state_event_detector_lab,
)


def _session_timestamps(session_date: str, bars: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{session_date} 13:30", periods=bars, freq="5min", tz="UTC")


def _frame_from_closes(
    closes: list[float],
    *,
    session_date: str = "2026-06-23",
    symbol: str = "TEST",
    volume: float = 10_000.0,
) -> pd.DataFrame:
    close = pd.Series(closes, dtype="float")
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.05
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.05
    return pd.DataFrame(
        {
            "source": "eodhd",
            "symbol": symbol,
            "instrument_type": "stock",
            "timeframe": "5m",
            "timestamp": _session_timestamps(session_date, len(closes)),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "currency": "USD",
            "timezone": "UTC",
        }
    )


def _controlled_pullback_frame(symbol: str = "TEST") -> pd.DataFrame:
    closes = [
        100.00,
        100.10,
        100.24,
        100.42,
        100.68,
        100.95,
        101.18,
        101.40,
        101.62,
        101.82,
        101.96,
        102.08,
        102.18,
        102.05,
        101.96,
        102.04,
        102.12,
        102.18,
        102.24,
        102.30,
        102.35,
        102.40,
        102.44,
        102.48,
        102.52,
        102.56,
        102.60,
        102.64,
        102.68,
        102.72,
        102.76,
        102.80,
        102.84,
        102.88,
        102.92,
        102.96,
    ]
    frame = _frame_from_closes(closes, symbol=symbol)
    frame.loc[3:12, "volume"] = 25_000.0
    frame.loc[13:15, "volume"] = 8_000.0
    return frame


def _with_volume_pattern(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[3:12, "volume"] = 25_000.0
    frame.loc[13:, "volume"] = 8_000.0
    return frame


def test_detect_state_events_outputs_clean_entry_rows_with_forward_targets() -> None:
    config = StateEventDetectorConfig(
        horizons=(6, 9, 12),
        min_bars_after_open=3,
        entry_cutoff_before_close_minutes=0,
        relative_volume_lookback_sessions=2,
    )

    events = detect_state_events(_controlled_pullback_frame(), symbol="TEST", config=config)

    assert "controlled_pullback_after_bullish_impulse" in set(events["event_state"])
    row = events[events["event_state"].eq("controlled_pullback_after_bullish_impulse")].iloc[0]
    assert row["event_family"] == "bullish_impulse_pullback"
    assert row["event_direction"] == "up"
    assert row["event_confidence_score"] > 0.0
    assert "overlap_candidates" in events.columns
    assert "prior_12_bar_return" in events.columns
    assert "impulse_midpoint" in events.columns
    assert pd.notna(row["forward_6_bar_return"])
    assert pd.notna(row["forward_9_bar_mfe"])
    assert pd.notna(row["forward_12_bar_mae"])
    event_index = int(row["raw_row_index"])
    source = _controlled_pullback_frame()
    expected_forward = source.loc[event_index + 6, "close"] / source.loc[event_index, "close"] - 1.0
    assert row["forward_6_bar_return"] == expected_forward
    assert bool(row["state_entry"]) is True


def test_event_detection_does_not_use_future_rows() -> None:
    config = StateEventDetectorConfig(
        horizons=(6,),
        min_bars_after_open=3,
        entry_cutoff_before_close_minutes=0,
        relative_volume_lookback_sessions=2,
    )
    original = _controlled_pullback_frame()
    mutated = original.copy()
    mutated.loc[20:, ["open", "high", "low", "close", "volume"]] = 10_000.0

    original_events = detect_state_events(original, symbol="TEST", config=config)
    mutated_events = detect_state_events(mutated, symbol="TEST", config=config)
    compared_columns = [
        "symbol",
        "timestamp",
        "session_date",
        "bar_index_in_session",
        "event_state",
        "trigger_reason",
        "overlap_candidates",
    ]
    original_prefix = original_events[original_events["bar_index_in_session"] < 20]
    mutated_prefix = mutated_events[mutated_events["bar_index_in_session"] < 20]

    pd.testing.assert_frame_equal(
        original_prefix[compared_columns].reset_index(drop=True),
        mutated_prefix[compared_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_each_event_detector_fires_on_minimal_synthetic_example() -> None:
    config = StateEventDetectorConfig(
        horizons=(6,),
        min_bars_after_open=3,
        entry_cutoff_before_close_minutes=0,
        relative_volume_lookback_sessions=2,
    )
    cases: dict[str, pd.DataFrame] = {
        "controlled_pullback_after_bullish_impulse": _controlled_pullback_frame(),
        "failed_bullish_impulse_recoil": _with_volume_pattern(
            _frame_from_closes(
                [100 + index * 0.20 for index in range(13)]
                + [101.20]
                + [102.40] * 20,
            )
        ),
        "liquidation_failed_low_reclaim": _frame_from_closes(
            [100 - index * 0.18 for index in range(13)]
            + [98.00, 97.80, 98.00]
            + [98.10] * 20,
        ),
        "failed_bounce_active_liquidation": _frame_from_closes(
            [
                100.00,
                99.90,
                99.80,
                99.70,
                99.60,
                99.50,
                99.20,
                98.90,
                98.60,
                98.30,
                98.00,
                97.80,
                97.60,
                97.75,
                97.70,
                97.65,
            ]
            + [97.60] * 20,
        ),
        "failed_open_down_continuation": _frame_from_closes(
            [
                100.00,
                100.05,
                99.98,
                100.02,
                99.96,
                99.94,
                99.70,
                99.50,
                99.30,
                99.10,
                98.90,
                98.80,
                98.70,
            ]
            + [98.60] * 20,
        ),
        "slow_snapback_after_dip": _frame_from_closes(
            [
                100.00,
                99.85,
                99.65,
                99.50,
                99.30,
                99.15,
                98.95,
                98.80,
                98.60,
                98.45,
                98.25,
                98.10,
                98.00,
                98.08,
                98.00,
                98.14,
                98.06,
                98.20,
                98.12,
                98.28,
                98.20,
                98.36,
            ]
            + [98.38] * 20,
        ),
        "dead_chop_blocker": _frame_from_closes(
            [100.00 + (0.03 if index % 2 else -0.03) for index in range(40)]
        ),
    }
    failed_recoil = cases["failed_bullish_impulse_recoil"]
    failed_recoil.loc[13, "high"] = 103.30
    failed_recoil.loc[13, "close"] = 101.20
    reclaim = cases["liquidation_failed_low_reclaim"]
    reclaim.loc[13, "low"] = 97.20
    reclaim.loc[13, "close"] = 98.00

    for expected_state, frame in cases.items():
        events = detect_state_events(frame, symbol="TEST", config=config)
        assert expected_state in set(events["event_state"]), expected_state
        assert len(events) < len(frame)


def test_manual_audit_reports_pass_fail_and_nearest_alternative() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["HOOD", "HOOD"],
            "timestamp": pd.to_datetime(
                ["2026-04-15 14:30", "2026-07-01 15:00"],
                utc=True,
            ),
            "session_date": ["2026-04-15", "2026-07-01"],
            "bar_index_in_session": [12, 18],
            "event_state": [
                "controlled_pullback_after_bullish_impulse",
                "dead_chop_blocker",
            ],
        }
    )
    examples = [
        {
            "symbol": "HOOD",
            "session_date": "2026-04-15",
            "expected_event_states": ("controlled_pullback_after_bullish_impulse",),
            "manual_note": "expected pass",
        },
        {
            "symbol": "HOOD",
            "session_date": "2026-07-01",
            "expected_event_states": ("failed_bullish_impulse_recoil",),
            "manual_note": "expected fail",
        },
    ]

    audit = audit_manual_state_examples(events, examples=examples)

    assert audit["detected_expected_event"].tolist() == [True, False]
    failed = audit[audit["detected_expected_event"].eq(False)].iloc[0]
    assert failed["nearest_detected_alternative_event"] == "dead_chop_blocker"
    assert failed["pass_fail"] == "fail"
    assert failed["status"] == "manual_reproduction_failed"


def test_state_event_detector_lab_writes_required_report_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for symbol in ["AAA", "BBB", "CCC"]:
        key = DatasetKey(source="eodhd", instrument_type="stock", symbol=symbol, timeframe="5m")
        write_parquet(
            _controlled_pullback_frame(symbol=symbol),
            dataset_path(key, data_dir=data_dir),
        )

    result = run_state_event_detector_lab(
        data_dir=data_dir,
        symbols=["AAA", "BBB", "CCC"],
        output_dir=tmp_path / "reports",
        config=StateEventDetectorConfig(
            horizons=(6, 9, 12),
            min_bars_after_open=3,
            entry_cutoff_before_close_minutes=0,
            relative_volume_lookback_sessions=2,
            min_events_for_similarity=1,
        ),
        manual_examples=[],
    )

    run_dir = result.output_dir
    expected_files = {
        "summary.md",
        "summary.json",
        "event_rows.csv",
        "manual_state_audit.csv",
        "event_state_summary.csv",
        "same_event_cross_symbol_similarity.csv",
        "random_baseline.csv",
        "oos_event_response.csv",
        "concentration_warnings.csv",
        "decision.json",
    }
    assert expected_files.issubset({path.name for path in run_dir.iterdir()})
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert payload["symbols_completed"] == ["AAA", "BBB", "CCC"]
    assert payload["total_event_rows"] >= 3
    assert decision["research_only"] is True
    assert decision["live_ordering_enabled"] is False
    assert decision["order_placement"] == "disabled"
    event_rows = pd.read_csv(run_dir / "event_rows.csv")
    manual_audit = pd.read_csv(run_dir / "manual_state_audit.csv")
    event_summary = pd.read_csv(run_dir / "event_state_summary.csv")
    oos = pd.read_csv(run_dir / "oos_event_response.csv")
    assert "overlap_candidates" in event_rows.columns
    assert {
        "expected_event_state",
        "detected_expected_event",
        "detected_event_timestamps",
        "nearest_detected_alternative_event",
        "nearest_alternative_timestamp",
        "pass_fail",
        "failure_notes",
    }.issubset(manual_audit.columns)
    assert {
        "median_forward_return",
        "mean_forward_return",
        "p25_return",
        "p75_return",
        "single_symbol_share",
        "single_session_share",
        "concentration_warning",
    }.issubset(event_summary.columns)
    assert {
        "test_directional_accuracy",
        "median_return_excess_vs_generic_bps",
    }.issubset(oos.columns)


def test_state_event_detector_cli_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for symbol in ["AAA", "BBB"]:
        key = DatasetKey(source="eodhd", instrument_type="stock", symbol=symbol, timeframe="5m")
        write_parquet(
            _controlled_pullback_frame(symbol=symbol),
            dataset_path(key, data_dir=data_dir),
        )
    config_path = tmp_path / "research.yaml"
    config_path.write_text(f"data:\n  data_dir: {data_dir}\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "research",
            "state-event-detector",
            "--symbols",
            "AAA,BBB",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--min-events-for-similarity",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "state_event_detector_v0" in result.output
    assert "manual_audit_status" in result.output


def test_cross_symbol_similarity_respects_sample_cap() -> None:
    rows = []
    for index in range(30):
        rows.append(
            {
                "symbol": "AAA" if index % 2 == 0 else "BBB",
                "timestamp": pd.Timestamp("2026-06-23 14:30", tz="UTC")
                + pd.Timedelta(minutes=5 * index),
                "session_date": "2026-06-23",
                "bar_index_in_session": index,
                "bar_index_bucket": "morning",
                "time_of_day_bucket": "morning",
                "primary_state_label": "dead_chop_blocker"
                if index % 5
                else "failed_open_down_continuation",
                "event_state": "dead_chop_blocker"
                if index % 5
                else "failed_open_down_continuation",
                "response_horizon": 6,
                "response_return": 0.001 if index % 3 else -0.001,
                "response_mfe": 0.002,
                "response_mae": -0.002,
                "path_return_1": 0.0001,
                "path_return_2": 0.0002,
                "path_return_3": 0.0003,
                "path_return_4": 0.0004,
                "path_return_5": 0.0005,
                "path_return_6": 0.0006,
            }
        )

    summary, _ = run_same_event_cross_symbol_similarity(
        pd.DataFrame(rows),
        config=StateEventDetectorConfig(max_similarity_events=8),
    )

    same_event = summary[summary["baseline"].eq("same_event_cross_symbol")]
    assert int(same_event["source_event_count"].sum()) <= 8
    assert {
        "random_cross_symbol_same_time_bucket",
        "different_event_cross_symbol_same_time_bucket",
        "same_symbol_random_event",
    }.issubset(set(summary["baseline"]))


def test_same_event_cross_symbol_similarity_excludes_same_symbol_matches() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "symbol": "AAA",
                "timestamp": pd.Timestamp("2026-06-23 14:30", tz="UTC")
                + pd.Timedelta(minutes=5 * index),
                "session_date": "2026-06-23",
                "bar_index_in_session": index,
                "bar_index_bucket": "morning",
                "time_of_day_bucket": "morning",
                "primary_state_label": "dead_chop_blocker",
                "event_state": "dead_chop_blocker",
                "response_horizon": 6,
                "response_return": 0.001,
                "response_mfe": 0.002,
                "response_mae": -0.002,
                "path_return_1": 0.0001,
                "path_return_2": 0.0002,
                "path_return_3": 0.0003,
                "path_return_4": 0.0004,
                "path_return_5": 0.0005,
                "path_return_6": 0.0006,
            }
        )

    summary, _ = run_same_event_cross_symbol_similarity(
        pd.DataFrame(rows),
        config=StateEventDetectorConfig(max_similarity_events=8),
    )

    assert "same_event_cross_symbol" not in set(summary["baseline"])


def test_decision_requires_same_state_horizon_for_transfer_and_oos() -> None:
    config = StateEventDetectorConfig(min_events_for_similarity=2)
    manual = pd.DataFrame({"detected_expected_event": [True]})
    state_summary = pd.DataFrame(
        {
            "event_state": ["state_a", "state_b"],
            "horizon": [6, 6],
            "event_count": [10, 10],
            "symbol_count": [3, 3],
        }
    )
    similarity = pd.DataFrame(
        {
            "baseline": [
                "same_event_cross_symbol",
                "random_cross_symbol_same_time_bucket",
                "different_event_cross_symbol_same_time_bucket",
            ],
            "event_state": ["state_a", "state_a", "state_a"],
            "horizon": [6, 6, 6],
            "response_sign_agreement": [0.70, 0.50, 0.55],
            "median_abs_return_difference": [0.01, 0.02, 0.03],
        }
    )
    oos = pd.DataFrame(
        {
            "event_state": ["state_b"],
            "horizon": [6],
            "gate_passed": [True],
        }
    )

    decision = build_decision(
        manual_audit=manual,
        event_state_summary=state_summary,
        similarity=similarity,
        oos_response=oos,
        concentration_warnings=pd.DataFrame(),
        config=config,
    )

    assert decision["decision"] == "reject_no_oos_edge"
