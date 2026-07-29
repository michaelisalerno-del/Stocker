from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.role_aware_event_cutter_v0 import (
    EVENT_STATE_ROLES,
    RoleAwareEventCutterConfig,
    add_aligned_return_column,
    add_rolling_symbol_state_efficacy,
    build_candidate_filters,
    build_decision,
    estimate_role_direction,
    evaluate_role_rows,
    run_random_role_baseline,
    run_role_aware_event_cutter_lab,
)


def _event_frame() -> pd.DataFrame:
    rows = []
    states = [
        "controlled_pullback_after_bullish_impulse",
        "failed_open_down_continuation",
        "failed_bullish_impulse_recoil",
        "dead_chop_blocker",
    ]
    for index in range(96):
        state = states[index % len(states)]
        session = pd.Timestamp("2026-01-02") + pd.Timedelta(days=index // len(states))
        is_bad_feature = index % 3 == 0
        if state == "controlled_pullback_after_bullish_impulse":
            forward_6 = -0.006 if is_bad_feature else 0.005
            forward_24 = forward_6 * 1.2
        elif state == "dead_chop_blocker":
            forward_6 = 0.0003 if is_bad_feature else 0.006
            forward_24 = forward_6 * 1.5
        else:
            forward_6 = 0.006 if is_bad_feature else -0.005
            forward_24 = forward_6 * 1.3
        rows.append(
            {
                "symbol": ["AAA", "BBB", "CCC"][index % 3],
                "timestamp": session + pd.Timedelta(hours=14, minutes=30),
                "session_date": session.date().isoformat(),
                "bar_index_in_session": 10 + index,
                "time_of_day_bucket": "morning" if index < 56 else "midday",
                "event_state": state,
                "distance_from_vwap_pct": -0.003 if is_bad_feature else 0.004,
                "distance_from_opening_range_mid_pct": -0.002 if is_bad_feature else 0.003,
                "distance_from_opening_range_high_pct": -0.004 if is_bad_feature else 0.001,
                "distance_from_opening_range_low_pct": -0.001 if is_bad_feature else 0.004,
                "distance_from_session_open_pct": -0.002 if is_bad_feature else 0.002,
                "distance_from_session_high_pct": -0.008 if is_bad_feature else -0.002,
                "distance_from_session_low_pct": 0.002 if is_bad_feature else 0.008,
                "distance_from_recent_high_pct": -0.007 if is_bad_feature else -0.001,
                "distance_from_recent_low_pct": 0.001 if is_bad_feature else 0.007,
                "close_location_value": 0.25 if is_bad_feature else 0.80,
                "upper_wick_pct_of_range": 0.70 if is_bad_feature else 0.20,
                "lower_wick_pct_of_range": 0.20 if is_bad_feature else 0.55,
                "bar_return": -0.003 if is_bad_feature else 0.003,
                "prior_3_bar_return": -0.004 if is_bad_feature else 0.004,
                "prior_6_bar_return": -0.002 if is_bad_feature else 0.005,
                "prior_12_bar_return": 0.001 if is_bad_feature else 0.006,
                "vwap_cross_count_12": 5 if is_bad_feature else 1,
                "range_cross_count_12": 4 if is_bad_feature else 1,
                "rolling_intraday_range_pct": 0.003 if state == "dead_chop_blocker" else 0.012,
                "compression_zscore": -0.5 if is_bad_feature else 0.5,
                "directional_efficiency_6": 0.20 if is_bad_feature else 0.70,
                "directional_efficiency_12": 0.25 if is_bad_feature else 0.75,
                "relative_volume_at_bar_index": 0.8 if is_bad_feature else 1.3,
                "relative_cumulative_volume": 0.9 if is_bad_feature else 1.2,
                "forward_6_bar_return": forward_6,
                "forward_6_bar_mfe": max(forward_6, 0.006),
                "forward_6_bar_mae": min(forward_6, -0.004),
                "forward_24_bar_return": forward_24,
                "forward_24_bar_mfe": max(forward_24, 0.007),
                "forward_24_bar_mae": min(forward_24, -0.005),
            }
        )
    return pd.DataFrame(rows)


def _write_state_event_report(input_dir: Path, frame: pd.DataFrame) -> None:
    input_dir.mkdir(parents=True)
    frame.to_csv(input_dir / "event_rows.csv", index=False)
    for name in [
        "manual_state_audit.csv",
        "event_state_summary.csv",
        "same_event_cross_symbol_similarity.csv",
        "random_baseline.csv",
        "oos_event_response.csv",
        "concentration_warnings.csv",
    ]:
        pd.DataFrame({"ok": [True]}).to_csv(input_dir / name, index=False)
    (input_dir / "summary.json").write_text(
        json.dumps({"manual_audit_status": "manual_reproduced"}),
        encoding="utf-8",
    )
    (input_dir / "summary.md").write_text("# test\n", encoding="utf-8")
    (input_dir / "decision.json").write_text(
        json.dumps({"decision": "continue_research"}),
        encoding="utf-8",
    )


def test_event_role_mapping_uses_expected_roles() -> None:
    assert EVENT_STATE_ROLES["controlled_pullback_after_bullish_impulse"]["role"] == (
        "long_candidate"
    )
    assert EVENT_STATE_ROLES["failed_open_down_continuation"][
        "default_expected_direction"
    ] == -1
    assert EVENT_STATE_ROLES["dead_chop_blocker"]["default_expected_direction"] == 0


def test_negative_forward_return_is_positive_aligned_for_short_role() -> None:
    frame = pd.DataFrame({"forward_24_bar_return": [-0.012]})

    aligned = add_aligned_return_column(frame, horizon=24, expected_direction=-1)

    assert aligned.iloc[0]["aligned_24_bar_return"] == 0.012


def test_role_evidence_conflict_flags_train_direction_against_default() -> None:
    train = pd.DataFrame({"forward_24_bar_return": [0.01, 0.02, 0.03]})

    direction = estimate_role_direction(
        event_state="failed_open_down_continuation",
        train_rows=train,
        horizon=24,
    )

    assert direction["default_expected_direction"] == -1
    assert direction["train_inferred_direction"] == 1
    assert direction["role_evidence_conflict"] is True


def test_role_metrics_score_long_blocker_short_and_no_trade() -> None:
    blocker = pd.DataFrame({"forward_24_bar_return": [-0.01, -0.02, 0.03, 0.04]})
    blocker_metrics = evaluate_role_rows(
        blocker,
        pd.Series([True, True, False, False]),
        event_state="failed_open_down_continuation",
        horizon=24,
        expected_direction=-1,
        role="long_blocker_or_short_candidate",
    )
    assert blocker_metrics["aligned_median_return_after"] == 0.015
    assert blocker_metrics["bad_long_capture_rate"] == 1.0
    assert blocker_metrics["good_long_false_block_rate"] == 0.0
    assert blocker_metrics["short_median_return_after"] == 0.015

    no_trade = pd.DataFrame({"forward_6_bar_return": [0.0002, -0.0003, 0.01]})
    no_trade_metrics = evaluate_role_rows(
        no_trade,
        pd.Series([True, True, False]),
        event_state="dead_chop_blocker",
        horizon=6,
        expected_direction=0,
        role="no_trade_filter",
    )
    assert no_trade_metrics["low_movement_rate_after"] == 1.0
    assert no_trade_metrics["false_block_big_move_rate_after"] == 0.0


def test_random_role_baseline_keeps_same_count_and_scores_aligned() -> None:
    frame = pd.DataFrame({"forward_24_bar_return": [-0.01, -0.02, 0.03, 0.04]})
    retained = frame.iloc[:2]

    baseline = run_random_role_baseline(
        test_rows=frame,
        retained_rows=retained,
        horizon=24,
        expected_direction=-1,
        role="long_blocker_or_short_candidate",
        seed=3,
        iterations=10,
    )

    assert baseline["baseline"] == "random_same_count"
    assert baseline["retained_count"] == 2
    assert pd.notna(baseline["aligned_median_return_after"])


def test_candidate_filter_thresholds_are_selected_from_train_only() -> None:
    frame = _event_frame()
    train = frame.iloc[:56].copy()
    test = frame.iloc[56:].copy()
    test["upper_wick_pct_of_range"] = 99.0

    candidates = build_candidate_filters(
        train,
        event_state="controlled_pullback_after_bullish_impulse",
        horizon=6,
        config=RoleAwareEventCutterConfig(
            min_train_events=5,
            min_retained_events=2,
            max_candidates_per_state_horizon=40,
        ),
    )

    upper_wick = candidates[candidates["feature_1"].eq("upper_wick_pct_of_range")]
    assert not upper_wick.empty
    assert upper_wick["threshold_1"].max() <= train["upper_wick_pct_of_range"].max()


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


def test_decision_rejects_concentrated_even_when_filter_passes() -> None:
    selected = pd.DataFrame(
        {
            "filter_id": ["f1"],
            "event_state": ["controlled_pullback_after_bullish_impulse"],
            "horizon": [6],
            "role": ["long_candidate"],
            "gate_passed": [True],
            "concentration_warning": [True],
            "selected_decision": ["continue_research_long_candidate"],
        }
    )
    warnings = pd.DataFrame({"warning": ["single_symbol_dominates"]})

    decision = build_decision(selected, warnings)

    assert decision["decision"] == "reject_concentrated"


def test_role_aware_event_cutter_report_writes_expected_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "state_event_detector_v0_20260630T000000Z"
    _write_state_event_report(input_dir, _event_frame())

    result = run_role_aware_event_cutter_lab(
        input_dir=input_dir,
        output_dir=tmp_path / "reports",
        config=RoleAwareEventCutterConfig(
            horizons=(6, 24),
            min_train_events=5,
            min_test_events=5,
            min_retained_events=2,
            random_iterations=10,
            max_candidates_per_state_horizon=12,
        ),
    )

    expected_files = {
        "summary.md",
        "summary.json",
        "decision.json",
        "role_aware_state_summary.csv",
        "aligned_directional_results.csv",
        "long_candidate_filter_results.csv",
        "blocker_quality_results.csv",
        "short_candidate_results.csv",
        "no_trade_quality_results.csv",
        "role_evidence_conflicts.csv",
        "filter_oos_results.csv",
        "random_role_baselines.csv",
        "concentration_warnings.csv",
        "selected_filters.csv",
        "rejected_filters.csv",
    }
    assert expected_files.issubset({path.name for path in result.output_dir.iterdir()})
    decision = json.loads((result.output_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["research_only"] is True
    assert decision["order_placement"] == "disabled"


def test_role_aware_event_cutter_cli_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "state_event_detector_v0_20260630T000000Z"
    _write_state_event_report(input_dir, _event_frame())

    result = CliRunner().invoke(
        app,
        [
            "research",
            "role-aware-event-cutter",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(tmp_path / "reports"),
            "--min-train-events",
            "5",
            "--min-test-events",
            "5",
            "--min-retained-events",
            "2",
            "--random-iterations",
            "10",
            "--max-candidates-per-state-horizon",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "role_aware_event_cutter_v0" in result.output
