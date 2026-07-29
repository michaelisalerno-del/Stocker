from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocker_research.personality_discovery_v0 import add_discovery_features
from stocker_research.personality_rulebook_validation_v0 import (
    RulebookValidationConfig,
    collapse_personality_rules,
    run_personality_rulebook_validation,
)


def _source_rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "personality": "open_down_pressure",
                "role": "short_or_long_blocker",
                "horizon": 6,
                "regime_field": "vwap_side_regime",
                "regime_value": "below",
                "rule_kind": "single",
                "feature": "distance_from_vwap_pct",
                "operator": "<=",
                "threshold": -0.002,
                "feature_b": "",
                "operator_b": "",
                "threshold_b": float("nan"),
                "filter_rule": "distance_from_vwap_pct <= -0.002",
                "retained_test_count": 12,
                "filtered_test_same_result_rate": 0.75,
                "test_lift_vs_personality": 0.20,
                "random_same_count_p95_rate": 0.60,
            },
            {
                "personality": "open_down_pressure",
                "role": "short_or_long_blocker",
                "horizon": 6,
                "regime_field": "vwap_side_regime",
                "regime_value": "below",
                "rule_kind": "single",
                "feature": "distance_from_vwap_pct",
                "operator": "<=",
                "threshold": -0.0025,
                "feature_b": "",
                "operator_b": "",
                "threshold_b": float("nan"),
                "filter_rule": "distance_from_vwap_pct <= -0.0025",
                "retained_test_count": 10,
                "filtered_test_same_result_rate": 0.70,
                "test_lift_vs_personality": 0.15,
                "random_same_count_p95_rate": 0.58,
            },
            {
                "personality": "reclaim_reversal",
                "role": "long_reversal",
                "horizon": 6,
                "regime_field": "event_quality_regime",
                "regime_value": "high_event_quality",
                "rule_kind": "and",
                "feature": "distance_from_session_low_pct",
                "operator": "<=",
                "threshold": 0.005,
                "feature_b": "lower_wick_pct_of_range",
                "operator_b": ">=",
                "threshold_b": 0.4,
                "filter_rule": "session low hold AND lower wick",
                "retained_test_count": 11,
                "filtered_test_same_result_rate": 0.72,
                "test_lift_vs_personality": 0.16,
                "random_same_count_p95_rate": 0.61,
            },
        ]
    )


def _event_rows() -> pd.DataFrame:
    rows = []
    for index in range(80):
        symbol = ["AAA", "BBB", "CCC", "DDD"][index % 4]
        is_good = index % 4 != 0
        session = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index // 4)
        rows.append(
            {
                "symbol": symbol,
                "timestamp": session + pd.Timedelta(hours=14, minutes=30 + index % 6),
                "session_date": session.date().isoformat(),
                "bar_index_in_session": 10 + index % 20,
                "event_state": "failed_open_down_continuation",
                "distance_from_vwap_pct": -0.004 if is_good else -0.001,
                "distance_from_session_low_pct": 0.006,
                "lower_wick_pct_of_range": 0.2,
                "upper_wick_pct_of_range": 0.5,
                "close_location_value": 0.3,
                "bar_return": -0.002,
                "prior_3_bar_return": -0.003,
                "prior_6_bar_return": -0.004,
                "prior_12_bar_return": -0.005,
                "directional_efficiency_6": 0.6,
                "directional_efficiency_12": 0.6,
                "rolling_intraday_range_pct": 0.015,
                "compression_zscore": 0.2,
                "forward_6_bar_return": -0.006 if is_good else 0.006,
            }
        )
    return add_discovery_features(pd.DataFrame(rows))


def test_collapse_personality_rules_keeps_best_structural_rule() -> None:
    collapsed = collapse_personality_rules(_source_rules(), top_per_personality=3)

    assert len(collapsed) == 2
    open_down = collapsed[collapsed["personality"].eq("open_down_pressure")].iloc[0]
    assert open_down["threshold"] == -0.002
    assert open_down["source_duplicate_count"] == 2


def test_rulebook_validation_uses_holdout_symbols_and_random_baseline(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    validation_dir = tmp_path / "validation"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    validation_dir.mkdir()
    _source_rules().to_csv(source_dir / "passed_personality_rules.csv", index=False)
    pd.DataFrame({"symbol": ["SRC"], "event_state": ["failed_open_down_continuation"]}).to_csv(
        source_dir / "event_rows.csv",
        index=False,
    )
    _event_rows().to_csv(validation_dir / "event_rows.csv", index=False)

    result = run_personality_rulebook_validation(
        source_personality_dir=source_dir,
        validation_event_dir=validation_dir,
        output_dir=output_dir,
        config=RulebookValidationConfig(
            random_iterations=10,
            min_validation_events=10,
            min_validation_symbols=3,
            max_single_symbol_share=0.5,
        ),
    )

    assert result.decision in {
        "continue_research_rulebook_transfer",
        "reject_no_rulebook_transfer",
    }
    assert (result.output_dir / "rulebook_validation_results.csv").exists()
    validation = pd.read_csv(result.output_dir / "rulebook_validation_results.csv")
    assert validation["validation_symbol_count"].max() >= 3
    assert "random_same_count_p95_rate" in validation.columns
