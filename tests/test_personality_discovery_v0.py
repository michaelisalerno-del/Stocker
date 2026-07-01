from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.personality_discovery_v0 import (
    PersonalityDiscoveryConfig,
    add_discovery_features,
    build_filter_candidates,
    load_personality_specs,
    run_personality_discovery_lab,
)


def _event_frame() -> pd.DataFrame:
    rows = []
    state_map = [
        ("failed_open_down_continuation", "open_down_pressure", -1),
        ("failed_bounce_active_liquidation", "active_liquidation", -1),
        ("liquidation_failed_low_reclaim", "reclaim_reversal", 1),
        ("controlled_pullback_after_bullish_impulse", "pullback_continuation", 1),
    ]
    for index in range(96):
        event_state, _, direction = state_map[index % len(state_map)]
        session = pd.Timestamp("2026-01-02", tz="UTC") + pd.Timedelta(days=index // 4)
        is_good = index % 3 != 0
        forward = 0.006 if direction > 0 else -0.006
        if not is_good:
            forward *= -1
        rows.append(
            {
                "symbol": ["AAA", "BBB", "CCC", "DDD"][index % 4],
                "timestamp": session + pd.Timedelta(hours=14, minutes=30 + index % 6),
                "session_date": session.date().isoformat(),
                "bar_index_in_session": 8 + index % 50,
                "time_of_day_bucket": "morning" if index % 20 < 10 else "midday",
                "event_state": event_state,
                "close": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "open": 100.0 + index - 0.2,
                "distance_from_vwap_pct": -0.003 if is_good else 0.004,
                "distance_from_opening_range_mid_pct": -0.002 if is_good else 0.003,
                "distance_from_opening_range_high_pct": -0.004,
                "distance_from_opening_range_low_pct": 0.004,
                "distance_from_session_open_pct": -0.002 if is_good else 0.002,
                "distance_from_session_high_pct": -0.006 if is_good else -0.001,
                "distance_from_session_low_pct": 0.006 if is_good else 0.001,
                "distance_from_recent_high_pct": -0.007 if is_good else -0.001,
                "distance_from_recent_low_pct": 0.007 if is_good else 0.001,
                "close_location_value": 0.30 if is_good else 0.80,
                "upper_wick_pct_of_range": 0.65 if is_good else 0.10,
                "lower_wick_pct_of_range": 0.55 if is_good else 0.15,
                "bar_return": -0.003 if is_good else 0.003,
                "prior_3_bar_return": 0.002 if is_good else -0.004,
                "prior_6_bar_return": -0.006 if is_good else 0.003,
                "prior_12_bar_return": -0.011 if is_good else 0.003,
                "vwap_cross_count_12": 1 if is_good else 5,
                "range_cross_count_12": 1 if is_good else 4,
                "rolling_intraday_range_pct": 0.018 if is_good else 0.006,
                "compression_zscore": 0.7 if is_good else -0.8,
                "range_zscore": 0.6 if is_good else -0.7,
                "directional_efficiency_6": 0.65 if is_good else 0.20,
                "directional_efficiency_12": 0.70 if is_good else 0.25,
                "relative_volume_at_bar_index": 1.25 if is_good else 0.75,
                "relative_cumulative_volume": 1.15 if is_good else 0.85,
                "forward_6_bar_return": forward,
                "forward_6_bar_mfe": max(forward, 0.005),
                "forward_6_bar_mae": min(forward, -0.004),
                "forward_12_bar_return": forward * 1.2,
                "forward_12_bar_mfe": max(forward * 1.2, 0.006),
                "forward_12_bar_mae": min(forward * 1.2, -0.005),
            }
        )
    return pd.DataFrame(rows)


def _write_state_event_report(input_dir: Path, frame: pd.DataFrame) -> None:
    input_dir.mkdir(parents=True)
    frame.to_csv(input_dir / "event_rows.csv", index=False)
    (input_dir / "summary.json").write_text(
        json.dumps({"manual_audit_status": "manual_reproduced"}),
        encoding="utf-8",
    )
    (input_dir / "summary.md").write_text("# test\n", encoding="utf-8")
    (input_dir / "decision.json").write_text(
        json.dumps({"decision": "continue_research"}),
        encoding="utf-8",
    )


def _write_specs(spec_dir: Path) -> None:
    spec_dir.mkdir(parents=True)
    (spec_dir / "open_down_pressure.yaml").write_text(
        """
personality: open_down_pressure
version: v0
role: short_or_long_blocker
default_expected_direction: -1
base_event_states:
  - failed_open_down_continuation
preferred_horizons: [6, 12]
regime_families:
  - vwap_side
  - compression_x_efficiency
filter_families:
  - prior_return
  - wick_quality
caveat_families:
  - auction_location
  - event_quality
minimums:
  train_count: 8
  test_count: 6
  retained_count: 4
  symbol_count: 3
max_single_symbol_share: 0.75
""",
        encoding="utf-8",
    )
    (spec_dir / "reclaim_reversal.yaml").write_text(
        """
personality: reclaim_reversal
version: v0
role: long_reversal
default_expected_direction: 1
base_event_states:
  - liquidation_failed_low_reclaim
preferred_horizons: [6, 12]
regime_families:
  - vwap_side
filter_families:
  - wick_quality
caveat_families:
  - event_quality
minimums:
  train_count: 8
  test_count: 6
  retained_count: 4
  symbol_count: 3
max_single_symbol_share: 0.75
""",
        encoding="utf-8",
    )


def test_load_personality_specs_maps_yaml_fields(tmp_path: Path) -> None:
    spec_dir = tmp_path / "specs"
    _write_specs(spec_dir)

    specs = load_personality_specs(spec_dir)

    assert {spec.personality for spec in specs} == {
        "open_down_pressure",
        "reclaim_reversal",
    }
    open_down = next(spec for spec in specs if spec.personality == "open_down_pressure")
    assert open_down.default_expected_direction == -1
    assert "auction_location" in open_down.caveat_families


def test_add_discovery_features_includes_caveat_columns() -> None:
    features = add_discovery_features(_event_frame())

    expected = {
        "vwap_side_regime",
        "compression_x_efficiency_regime",
        "auction_prior_close_side",
        "event_quality_regime",
        "role_rejection_wick",
        "role_bar_reversal",
    }
    assert expected.issubset(features.columns)


def test_filter_candidates_use_train_only_thresholds(tmp_path: Path) -> None:
    spec_dir = tmp_path / "specs"
    _write_specs(spec_dir)
    spec = next(
        spec
        for spec in load_personality_specs(spec_dir)
        if spec.personality == "open_down_pressure"
    )
    frame = add_discovery_features(_event_frame())
    train = frame.iloc[:48].copy()
    test = frame.iloc[48:].copy()
    test["upper_wick_pct_of_range"] = 99.0

    candidates = build_filter_candidates(
        train,
        spec=spec,
        horizon=6,
        config=PersonalityDiscoveryConfig(
            random_iterations=5,
            max_filters_per_personality_horizon=20,
        ),
    )

    upper_wick = candidates[candidates["feature"].eq("upper_wick_pct_of_range")]
    assert not upper_wick.empty
    assert upper_wick["threshold"].max() <= train["upper_wick_pct_of_range"].max()


def test_personality_discovery_report_writes_expected_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "state_event_detector_v0_20260701T000000Z"
    spec_dir = tmp_path / "specs"
    _write_state_event_report(input_dir, _event_frame())
    _write_specs(spec_dir)

    result = run_personality_discovery_lab(
        input_dir=input_dir,
        spec_dir=spec_dir,
        output_dir=tmp_path / "reports",
        config=PersonalityDiscoveryConfig(
            horizons=(6, 12),
            random_iterations=5,
            max_filters_per_personality_horizon=20,
            default_min_train_events=8,
            default_min_test_events=6,
            default_min_retained_events=4,
            default_min_symbols=3,
        ),
    )

    expected = {
        "summary.md",
        "summary.json",
        "decision.json",
        "loaded_personality_specs.csv",
        "personality_base_summary.csv",
        "candidate_personality_rules.csv",
        "selected_personality_rules.csv",
        "passed_personality_rules.csv",
        "rejected_personality_rules.csv",
        "random_personality_baseline.csv",
        "concentration_warnings.csv",
        "personality_discovery_examples.csv",
        "personality_decision_matrix.csv",
    }
    assert expected.issubset({path.name for path in result.output_dir.iterdir()})
    payload = json.loads((result.output_dir / "decision.json").read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert payload["order_placement"] == "disabled"


def test_personality_discovery_cli_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "state_event_detector_v0_20260701T000000Z"
    spec_dir = tmp_path / "specs"
    _write_state_event_report(input_dir, _event_frame())
    _write_specs(spec_dir)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "personality-discovery",
            "--input-dir",
            str(input_dir),
            "--spec-dir",
            str(spec_dir),
            "--output-dir",
            str(tmp_path / "reports"),
            "--horizons",
            "6,12",
            "--random-iterations",
            "5",
            "--max-filters-per-personality-horizon",
            "20",
            "--default-min-train-events",
            "8",
            "--default-min-test-events",
            "6",
            "--default-min-retained-events",
            "4",
            "--default-min-symbols",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "personality_discovery_v0" in result.output
