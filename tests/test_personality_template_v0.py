from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_research.personality_template_v0 import (
    PersonalityTemplateConfig,
    evaluate_personality_templates,
    load_personality_templates,
    run_personality_template_lab,
)


def _write_templates(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "template_test_v0",
        "research_only": True,
        "templates": [
            {
                "template_id": "pullback_session_open_hold",
                "personality": "pullback_near_vwap_structure_hold",
                "parent_event_state": "controlled_pullback_after_bullish_impulse",
                "role": "long_candidate",
                "expected_direction": 1,
                "horizon": 12,
                "base_conditions": [
                    {
                        "feature": "distance_from_vwap_pct",
                        "operator": "between",
                        "lower": -0.002,
                        "upper": 0.006,
                    }
                ],
                "regime_fields": ["session_open_side_regime"],
                "filter_features": ["distance_from_opening_range_low_pct"],
                "minimums": {
                    "base_events": 8,
                    "retained_events": 4,
                    "symbols": 2,
                    "months": 1,
                },
                "max_single_month_share": 1.0,
            },
            {
                "template_id": "dead_chop_near_vwap_low_range",
                "personality": "near_vwap_dead_chop_low_range",
                "parent_event_state": "dead_chop_blocker",
                "role": "no_trade_filter",
                "expected_direction": 0,
                "horizon": 12,
                "base_conditions": [
                    {
                        "feature": "distance_from_vwap_pct",
                        "operator": "between",
                        "lower": -0.002,
                        "upper": 0.002,
                    }
                ],
                "regime_fields": ["range_regime"],
                "filter_features": ["prior_6_bar_return"],
                "minimums": {
                    "base_events": 8,
                    "retained_events": 4,
                    "symbols": 2,
                    "months": 1,
                },
                "max_single_month_share": 1.0,
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _event_rows() -> pd.DataFrame:
    rows = []
    symbols = ["AAA", "BBB", "CCC"]
    for index in range(72):
        symbol = symbols[index % len(symbols)]
        session = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index // 3)
        is_pullback = index % 2 == 0
        if is_pullback:
            event_state = "controlled_pullback_after_bullish_impulse"
            is_good = index % 6 != 0
            forward = 0.009 if is_good else -0.004
            abs_forward = abs(forward)
            distance_from_vwap = 0.001
            opening_low_dist = 0.005 if is_good else 0.018
            prior_6 = 0.002
        else:
            event_state = "dead_chop_blocker"
            forward = 0.001 if index % 5 else 0.006
            abs_forward = abs(forward)
            distance_from_vwap = 0.0005
            opening_low_dist = 0.003
            prior_6 = 0.002 if index % 3 else -0.003
        rows.append(
            {
                "symbol": symbol,
                "timestamp": session + pd.Timedelta(hours=14, minutes=index % 30),
                "session_date": session.date().isoformat(),
                "bar_index_in_session": 10 + index % 20,
                "bar_index_bucket": "morning",
                "time_of_day_bucket": "morning",
                "event_state": event_state,
                "distance_from_vwap_pct": distance_from_vwap,
                "distance_from_opening_range_mid_pct": 0.002,
                "distance_from_opening_range_high_pct": -0.004,
                "distance_from_opening_range_low_pct": opening_low_dist,
                "distance_from_session_open_pct": 0.0002,
                "distance_from_session_high_pct": -0.006,
                "distance_from_session_low_pct": 0.008,
                "distance_from_recent_high_pct": -0.004,
                "distance_from_recent_low_pct": 0.006,
                "close_location_value": 0.65,
                "upper_wick_pct_of_range": 0.2,
                "lower_wick_pct_of_range": 0.25,
                "bar_return": 0.001,
                "prior_3_bar_return": 0.001,
                "prior_6_bar_return": prior_6,
                "prior_12_bar_return": -0.002,
                "directional_efficiency_6": 0.3,
                "directional_efficiency_12": 0.2,
                "rolling_intraday_range_pct": 0.008,
                "compression_zscore": -0.2,
                "vwap_cross_count_12": 2,
                "range_cross_count_12": 1,
                "forward_12_bar_return": forward,
                "forward_12_bar_abs_return": abs_forward,
            }
        )
    return pd.DataFrame(rows)


def _write_event_report(path: Path) -> None:
    path.mkdir(parents=True)
    _event_rows().to_csv(path / "event_rows.csv", index=False)
    (path / "summary.json").write_text(
        json.dumps({"decision": "continue_research"}),
        encoding="utf-8",
    )


def test_load_personality_templates_reads_base_conditions(tmp_path: Path) -> None:
    template_path = tmp_path / "templates.yaml"
    _write_templates(template_path)

    book = load_personality_templates(template_path)

    assert book.version == "template_test_v0"
    assert len(book.templates) == 2
    assert book.templates[0].base_conditions[0].operator == "between"


def test_evaluate_templates_uses_role_aware_scoring_and_no_trade_quality(tmp_path: Path) -> None:
    template_path = tmp_path / "templates.yaml"
    _write_templates(template_path)
    book = load_personality_templates(template_path)

    base, candidates, selected, random_baseline = evaluate_personality_templates(
        _event_rows(),
        book,
        config=PersonalityTemplateConfig(random_iterations=5, max_candidates_per_template=16),
    )

    assert set(base["template_id"]) == {
        "pullback_session_open_hold",
        "dead_chop_near_vwap_low_range",
    }
    assert not candidates.empty
    assert not selected.empty
    assert not random_baseline.empty
    pullback = base[base["template_id"].eq("pullback_session_open_hold")].iloc[0]
    dead_chop = base[base["template_id"].eq("dead_chop_near_vwap_low_range")].iloc[0]
    assert pullback["base_score_mode"] == "directional"
    assert dead_chop["base_score_mode"] == "no_trade_low_abs_move"


def test_run_personality_template_lab_writes_expected_reports(tmp_path: Path) -> None:
    input_dir = tmp_path / "events"
    template_path = tmp_path / "templates.yaml"
    _write_event_report(input_dir)
    _write_templates(template_path)

    result = run_personality_template_lab(
        input_event_dir=input_dir,
        template_path=template_path,
        output_dir=tmp_path / "out",
        config=PersonalityTemplateConfig(random_iterations=5, max_candidates_per_template=16),
    )

    expected = {
        "summary.md",
        "summary.json",
        "decision.json",
        "personality_templates.csv",
        "template_base_summary.csv",
        "candidate_template_rules.csv",
        "selected_template_rules.csv",
        "rejected_template_rules.csv",
        "template_matches.csv",
        "random_baseline.csv",
        "concentration_warnings.csv",
    }
    assert expected.issubset({path.name for path in result.output_dir.iterdir()})
    payload = json.loads((result.output_dir / "decision.json").read_text(encoding="utf-8"))
    assert payload["research_only"] is True
    assert payload["order_placement"] == "disabled"


def test_personality_template_cli_smoke(tmp_path: Path) -> None:
    input_dir = tmp_path / "events"
    template_path = tmp_path / "templates.yaml"
    _write_event_report(input_dir)
    _write_templates(template_path)

    result = CliRunner().invoke(
        app,
        [
            "research",
            "personality-template",
            "--input-event-dir",
            str(input_dir),
            "--template-path",
            str(template_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--random-iterations",
            "5",
            "--max-candidates-per-template",
            "16",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "personality_template_v0" in result.output
