from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.personality_discovery_v0 import add_discovery_features
from stocker_research.personality_rulebook_v0 import (
    PersonalityRulebookConfig,
    apply_fixed_rulebook,
    load_fixed_rulebook,
    run_personality_rulebook_lab,
)


def _write_rulebook(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v0_test",
        "research_only": True,
        "rules": [
            {
                "rule_id": "open_down_fixed",
                "personality": "open_down_pressure",
                "role": "short_or_long_blocker",
                "horizon": 6,
                "expected_direction": -1,
                "regime_field": "vwap_side_regime",
                "regime_value": "below",
                "rule_kind": "single",
                "feature": "distance_from_vwap_pct",
                "operator": "<=",
                "threshold": -0.002,
                "filter_rule": "distance_from_vwap_pct <= -0.002",
            },
            {
                "rule_id": "pullback_fixed",
                "personality": "pullback_continuation",
                "role": "long_continuation",
                "horizon": 6,
                "expected_direction": 1,
                "regime_field": "auction_session_open_location",
                "regime_value": "near",
                "rule_kind": "and",
                "feature": "distance_from_vwap_pct",
                "operator": "<=",
                "threshold": 0.005,
                "feature_b": "distance_from_recent_high_pct",
                "operator_b": "<=",
                "threshold_b": -0.003,
                "filter_rule": "vwap near AND pullback shallow",
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _event_rows() -> pd.DataFrame:
    rows = []
    states = [
        ("failed_open_down_continuation", -1),
        ("controlled_pullback_after_bullish_impulse", 1),
    ]
    for index in range(96):
        event_state, direction = states[index % 2]
        is_good = index % 4 != 0
        session = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index // 4)
        forward = 0.006 * direction
        if not is_good:
            forward *= -1
        rows.append(
            {
                "symbol": ["AAA", "BBB", "CCC", "DDD"][index % 4],
                "timestamp": session + pd.Timedelta(hours=14, minutes=30 + index % 6),
                "session_date": session.date().isoformat(),
                "bar_index_in_session": 8 + index % 30,
                "event_state": event_state,
                "distance_from_vwap_pct": -0.004 if direction < 0 else 0.003,
                "distance_from_session_open_pct": 0.0002,
                "distance_from_opening_range_mid_pct": -0.004,
                "distance_from_opening_range_high_pct": -0.006,
                "distance_from_opening_range_low_pct": 0.006,
                "distance_from_session_high_pct": -0.006,
                "distance_from_session_low_pct": 0.006,
                "distance_from_recent_high_pct": -0.004,
                "distance_from_recent_low_pct": 0.004,
                "close_location_value": 0.4,
                "upper_wick_pct_of_range": 0.4,
                "lower_wick_pct_of_range": 0.4,
                "bar_return": -0.002 if direction < 0 else 0.002,
                "prior_3_bar_return": -0.003 if direction < 0 else 0.003,
                "prior_6_bar_return": -0.004 if direction < 0 else 0.004,
                "prior_12_bar_return": -0.005 if direction < 0 else 0.005,
                "directional_efficiency_6": 0.6,
                "directional_efficiency_12": 0.6,
                "rolling_intraday_range_pct": 0.015,
                "compression_zscore": 0.2,
                "forward_6_bar_return": forward,
            }
        )
    return pd.DataFrame(rows)


def _write_event_report(path: Path) -> None:
    path.mkdir(parents=True)
    _event_rows().to_csv(path / "event_rows.csv", index=False)
    (path / "summary.json").write_text(json.dumps({"decision": "continue_research"}))


def test_load_fixed_rulebook_preserves_literal_thresholds(tmp_path: Path) -> None:
    rulebook_path = tmp_path / "rulebook.yaml"
    _write_rulebook(rulebook_path)

    rulebook = load_fixed_rulebook(rulebook_path)

    assert rulebook.version == "v0_test"
    assert len(rulebook.rules) == 2
    assert rulebook.rules[0].threshold == -0.002


def test_apply_fixed_rulebook_uses_existing_rule_values(tmp_path: Path) -> None:
    frame = add_discovery_features(_event_rows())
    rulebook_path = tmp_path / "rulebook.yaml"
    _write_rulebook(rulebook_path)
    rulebook = load_fixed_rulebook(rulebook_path)

    matches, summary, _random = apply_fixed_rulebook(
        frame,
        rulebook,
        config=PersonalityRulebookConfig(random_iterations=5, min_events=8, min_symbols=3),
    )

    assert not matches.empty
    assert set(matches["rule_id"]) == {"open_down_fixed", "pullback_fixed"}
    assert summary["threshold"].tolist() == [-0.002, 0.005]
    assert summary["event_count"].min() >= 8


def test_run_personality_rulebook_lab_writes_reports(tmp_path: Path) -> None:
    input_dir = tmp_path / "events"
    rulebook_path = tmp_path / "rulebook.yaml"
    _write_event_report(input_dir)
    _write_rulebook(rulebook_path)

    result = run_personality_rulebook_lab(
        input_event_dir=input_dir,
        rulebook_path=rulebook_path,
        output_dir=tmp_path / "out",
        config=PersonalityRulebookConfig(random_iterations=5, min_events=8, min_symbols=3),
    )

    expected = {
        "summary.md",
        "summary.json",
        "decision.json",
        "fixed_rulebook.csv",
        "rule_matches.csv",
        "rule_summary.csv",
        "personality_summary.csv",
        "random_baseline.csv",
        "concentration_warnings.csv",
    }
    assert expected.issubset({path.name for path in result.output_dir.iterdir()})
    payload = json.loads((result.output_dir / "decision.json").read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert payload["order_placement"] == "disabled"


def test_personality_rulebook_cli_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "events"
    rulebook_path = tmp_path / "rulebook.yaml"
    _write_event_report(input_dir)
    _write_rulebook(rulebook_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "personality-rulebook",
            "--input-event-dir",
            str(input_dir),
            "--rulebook-path",
            str(rulebook_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--random-iterations",
            "5",
            "--min-events",
            "8",
            "--min-symbols",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "personality_rulebook_v0" in result.output
