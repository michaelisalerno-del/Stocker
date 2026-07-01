from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.event_failure_cutter_v0 import (
    EventFailureCutterConfig,
    add_rolling_symbol_state_efficacy,
    build_candidate_filters,
    build_decision,
    calculate_blocker_quality,
    run_event_failure_cutter_lab,
    run_random_same_count_baseline,
)


def _event_frame() -> pd.DataFrame:
    rows = []
    states = [
        "controlled_pullback_after_bullish_impulse",
        "failed_open_down_continuation",
    ]
    for index in range(80):
        state = states[index % 2]
        symbol = "AAA" if index % 4 < 2 else "BBB"
        session = pd.Timestamp("2026-01-02") + pd.Timedelta(days=index // 2)
        forward_6 = 0.004 if index % 5 else -0.006
        forward_24 = -0.004 if state == "failed_open_down_continuation" else forward_6
        if state == "failed_open_down_continuation" and index % 7 == 0:
            forward_24 = 0.007
        rows.append(
            {
                "symbol": symbol,
                "timestamp": session + pd.Timedelta(hours=14, minutes=30),
                "session_date": session.date().isoformat(),
                "bar_index_in_session": 10 + index,
                "time_of_day_bucket": "morning" if index < 50 else "midday",
                "event_state": state,
                "event_family": "test",
                "event_direction": "up" if "controlled" in state else "down",
                "event_confidence_score": 0.8,
                "distance_from_vwap_pct": -0.002 if index % 3 else 0.004,
                "distance_from_opening_range_mid_pct": -0.003 if index % 4 else 0.002,
                "distance_from_opening_range_low_pct": -0.001 if index % 5 else 0.003,
                "distance_from_session_high_pct": -0.006 if index % 6 else -0.001,
                "distance_from_session_low_pct": 0.004 if index % 4 else 0.001,
                "upper_wick_pct_of_range": 0.65 if index % 6 == 0 else 0.20,
                "lower_wick_pct_of_range": 0.25,
                "close_location_value": 0.25 if index % 6 == 0 else 0.75,
                "bar_return": -0.002 if index % 6 == 0 else 0.002,
                "prior_3_bar_return": -0.003 if index % 6 == 0 else 0.003,
                "prior_6_bar_return": 0.004,
                "prior_12_bar_return": 0.006,
                "directional_efficiency_6": 0.25 if index % 5 == 0 else 0.70,
                "directional_efficiency_12": 0.30 if index % 5 == 0 else 0.75,
                "pullback_depth_from_recent_high": 0.35 if index % 5 else 0.75,
                "impulse_volume_ratio": 0.70 if index % 5 else 1.50,
                "relative_volume_at_bar_index": 1.20 if index % 7 else 2.20,
                "rolling_intraday_range_pct": 0.012,
                "range_zscore": 0.2,
                "compression_zscore": -0.2,
                "vwap_cross_count_12": 1 if index % 4 else 4,
                "range_cross_count_12": 1 if index % 4 else 4,
                "forward_6_bar_return": forward_6,
                "forward_6_bar_mfe": max(forward_6, 0.006),
                "forward_6_bar_mae": min(forward_6, -0.004),
                "forward_24_bar_return": forward_24,
                "forward_24_bar_mfe": max(forward_24, 0.006),
                "forward_24_bar_mae": min(forward_24, -0.005),
            }
        )
    return pd.DataFrame(rows)


def test_rolling_symbol_state_efficacy_uses_prior_sessions_only() -> None:
    frame = _event_frame().head(6).copy()
    frame["symbol"] = "AAA"
    frame["event_state"] = "controlled_pullback_after_bullish_impulse"
    frame["session_date"] = [f"2026-01-0{index + 1}" for index in range(6)]
    frame["timestamp"] = pd.to_datetime(frame["session_date"], utc=True)
    frame["forward_6_bar_return"] = [0.01, -0.02, 0.03, -0.04, 0.05, 0.50]

    features = add_rolling_symbol_state_efficacy(frame, horizons=(6,), windows=(2,))
    original = features.loc[features["session_date"].eq("2026-01-05")].iloc[0][
        "symbol_state_h6_prior_2_session_median_return"
    ]

    mutated = frame.copy()
    mutated.loc[mutated["session_date"].eq("2026-01-06"), "forward_6_bar_return"] = -0.99
    mutated_features = add_rolling_symbol_state_efficacy(mutated, horizons=(6,), windows=(2,))
    unchanged = mutated_features.loc[mutated_features["session_date"].eq("2026-01-05")].iloc[0][
        "symbol_state_h6_prior_2_session_median_return"
    ]

    assert pd.isna(features.iloc[0]["symbol_state_h6_prior_2_session_median_return"])
    assert original == unchanged
    assert original == (-0.04 + 0.03) / 2


def test_candidate_filter_thresholds_are_train_only() -> None:
    frame = _event_frame()
    train = frame.iloc[:40].copy()
    test = frame.iloc[40:].copy()
    test["upper_wick_pct_of_range"] = 99.0

    candidates = build_candidate_filters(
        train,
        event_state="controlled_pullback_after_bullish_impulse",
        horizon=6,
        config=EventFailureCutterConfig(
            min_train_events=5,
            min_test_events=5,
            max_candidates_per_state_horizon=80,
        ),
    )

    upper_wick = candidates[candidates["feature_1"].eq("upper_wick_pct_of_range")]
    assert not upper_wick.empty
    assert upper_wick["threshold_1"].max() <= train["upper_wick_pct_of_range"].max()


def test_random_same_count_baseline_matches_retained_count() -> None:
    frame = _event_frame().head(30)
    retained = frame.iloc[:7]

    baseline = run_random_same_count_baseline(
        test_rows=frame,
        retained_rows=retained,
        horizon=6,
        objective_mode="long",
        seed=7,
        iterations=25,
    )

    assert int(baseline["retained_count"]) == 7
    assert baseline["baseline"] == "random_same_count"
    assert pd.notna(baseline["median_objective_after"])


def test_blocker_quality_calculation_reports_capture_and_false_block_rates() -> None:
    frame = pd.DataFrame(
        {
            "forward_24_bar_return": [-0.01, -0.02, 0.03, 0.04],
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "session_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        }
    )
    retained_mask = pd.Series([True, False, True, False])

    quality = calculate_blocker_quality(frame, retained_mask, horizon=24)

    assert quality["bad_long_capture_rate"] == 0.5
    assert quality["good_long_false_block_rate"] == 0.5
    assert quality["retained_count"] == 2


def test_event_failure_cutter_report_writes_expected_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "state_event_detector_v0_20260630T000000Z"
    input_dir.mkdir()
    _event_frame().to_csv(input_dir / "event_rows.csv", index=False)
    for name in [
        "manual_state_audit.csv",
        "event_state_summary.csv",
        "same_event_cross_symbol_similarity.csv",
        "random_baseline.csv",
        "oos_event_response.csv",
        "concentration_warnings.csv",
    ]:
        pd.DataFrame({"ok": [True]}).to_csv(input_dir / name, index=False)
    (input_dir / "summary.json").write_text("{}", encoding="utf-8")
    (input_dir / "summary.md").write_text("# test\n", encoding="utf-8")
    (input_dir / "decision.json").write_text('{"decision": "continue_research"}', encoding="utf-8")

    result = run_event_failure_cutter_lab(
        input_dir=input_dir,
        output_dir=tmp_path / "reports",
        config=EventFailureCutterConfig(
            horizons=(6, 24),
            min_train_events=5,
            min_test_events=5,
            random_iterations=10,
        ),
    )

    expected_files = {
        "summary.md",
        "summary.json",
        "decision.json",
        "failure_attribution_summary.csv",
        "candidate_bad_trade_filters.csv",
        "filter_oos_results.csv",
        "random_filter_baseline.csv",
        "blocker_quality_summary.csv",
        "state_failure_examples.csv",
        "feature_distribution_good_vs_bad.csv",
        "concentration_warnings.csv",
    }
    assert expected_files.issubset({path.name for path in result.output_dir.iterdir()})
    decision = json.loads((result.output_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["research_only"] is True
    assert decision["order_placement"] == "disabled"


def test_decision_rejects_when_random_beats_candidate_filter() -> None:
    filter_results = pd.DataFrame(
        {
            "gate_passed": [False],
            "random_beaten": [False],
            "oos_objective_lift_bps": [1.0],
            "concentration_warning": [False],
        }
    )

    decision = build_decision(filter_results, pd.DataFrame())

    assert decision["decision"] == "reject_random_filter_beats_state_filter"


def test_event_failure_cutter_cli_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "state_event_detector_v0_20260630T000000Z"
    input_dir.mkdir()
    _event_frame().to_csv(input_dir / "event_rows.csv", index=False)
    for name in [
        "manual_state_audit.csv",
        "event_state_summary.csv",
        "same_event_cross_symbol_similarity.csv",
        "random_baseline.csv",
        "oos_event_response.csv",
        "concentration_warnings.csv",
    ]:
        pd.DataFrame({"ok": [True]}).to_csv(input_dir / name, index=False)
    (input_dir / "summary.json").write_text("{}", encoding="utf-8")
    (input_dir / "summary.md").write_text("# test\n", encoding="utf-8")
    (input_dir / "decision.json").write_text('{"decision": "continue_research"}', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "research",
            "event-failure-cutter",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "reports"),
            "--min-train-events",
            "5",
            "--min-test-events",
            "5",
            "--random-iterations",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "event_failure_cutter_v0" in result.output
