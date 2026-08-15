from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

from stocker_prospective.microstructure_direction_v0 import (
    DepthStatusV0,
    DirectionActionV0,
    DirectionMethodV0,
    MicrostructureDirectionResultV0,
    TickByTickStatusV0,
)
from stocker_prospective.microstructure_direction_v0_research import (
    PricePointV0,
    evaluate_forward_outcomes_v0,
    run_census_only_research_v0,
    summarise_method_results_v0,
)

T0 = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)


def _direction(action: DirectionActionV0 = DirectionActionV0.UP) -> MicrostructureDirectionResultV0:
    return MicrostructureDirectionResultV0(
        episode_id="episode-1",
        run_id="run-1",
        symbol="AAL",
        trigger_timestamp_utc=T0,
        decision_timestamp_utc=T0,
        information_cutoff_utc=T0,
        confirmation_delay_seconds=0.0,
        window_name="T-5s_to_T0",
        direction_method=DirectionMethodV0.M01,
        action=action,
        signed_score=0.5 if action is not DirectionActionV0.ABSTAIN else None,
        component_values={"trade_imbalance": 0.5},
        component_validity={"trade_imbalance": True},
        quote_count=5,
        trade_count=4,
        classified_trade_count=3,
        trade_classification_valid_fraction=0.75,
        stale_quote_fraction=None,
        unclassified_trade_fraction=0.25,
        unknown_trade_volume_fraction=0.1,
        probable_buyer_initiated_volume=75.0,
        probable_seller_initiated_volume=25.0,
        tick_by_tick_status=TickByTickStatusV0.BIDASK_AND_LAST_PRESENT,
        depth_status=DepthStatusV0.ABSENT,
        market_data_type="live",
        data_quality_flags=(),
        causal_valid=True,
    )


def test_forward_outcome_starts_strictly_after_cutoff_and_uses_log_returns() -> None:
    points = (
        PricePointV0(received_timestamp_utc=T0, midpoint=999.0, event_id="at-cutoff"),
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(seconds=1), midpoint=100.0, event_id="entry"
        ),
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(minutes=2), midpoint=102.0, event_id="high"
        ),
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(minutes=3), midpoint=99.0, event_id="low"
        ),
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(minutes=5, seconds=1),
            midpoint=101.0,
            event_id="terminal-5m",
        ),
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(minutes=10, seconds=1),
            midpoint=103.0,
            event_id="terminal-10m",
        ),
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(minutes=15, seconds=1),
            midpoint=104.0,
            event_id="terminal-15m",
        ),
    )

    outcomes = evaluate_forward_outcomes_v0(direction=_direction(), price_points=points)

    assert tuple(row.horizon_minutes for row in outcomes) == (5, 10, 15)
    first = outcomes[0]
    assert first.entry_event_id == "entry"
    assert first.entry_timestamp_utc > first.information_cutoff_utc
    assert first.entry_delay_seconds == 1.0
    assert first.forward_log_return == math.log(101.0 / 100.0)
    assert first.signed_forward_log_return == first.forward_log_return
    assert first.realized_direction == "UP"
    assert first.direction_correct is True
    assert first.signed_mfe == math.log(102.0 / 100.0)
    assert first.signed_mae == math.log(99.0 / 100.0)


def test_method_summary_keeps_abstentions_in_coverage_denominator() -> None:
    points = (
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(seconds=1), midpoint=100.0, event_id="entry"
        ),
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(minutes=15, seconds=1),
            midpoint=101.0,
            event_id="terminal",
        ),
    )
    resolved = evaluate_forward_outcomes_v0(
        direction=_direction(),
        price_points=points,
        m1c_genuine_forward_move=True,
    )[2]

    summary = summarise_method_results_v0(
        rows=(resolved,),
        eligible_episode_count=2,
        abstain_episode_count=1,
    )

    assert summary["eligible_m1c_episodes"] == 2
    assert summary["direction_resolved_n"] == 1
    assert summary["coverage"] == 0.5
    assert summary["direction_accuracy"] == 1.0
    assert summary["useful_joint_rate"] == 1.0
    assert summary["joint_rate_over_all_m1c_episodes"] == 0.5


def test_joint_metrics_are_unavailable_without_the_canonical_m1c_move_label() -> None:
    points = (
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(seconds=1),
            midpoint=100.0,
            event_id="entry",
        ),
        PricePointV0(
            received_timestamp_utc=T0 + timedelta(minutes=15, seconds=1),
            midpoint=101.0,
            event_id="terminal",
        ),
    )
    resolved = evaluate_forward_outcomes_v0(direction=_direction(), price_points=points)[2]

    summary = summarise_method_results_v0(
        rows=(resolved,),
        eligible_episode_count=1,
        abstain_episode_count=0,
    )

    assert summary["useful_joint_rate"] is None
    assert summary["joint_rate_over_all_m1c_episodes"] is None


def test_zero_episode_census_writes_full_artifact_set_and_decision_d(tmp_path) -> None:
    census = tmp_path / "source_census.json"
    census.write_text(
        json.dumps(
            {
                "source_label": "prospective_archives",
                "databases": [
                    {
                        "database_identity": "archive-1",
                        "eligible_m1c_episodes": 0,
                        "episode_microstructure_rows": 0,
                        "microstructure_episodes": 0,
                        "first_episode_date": None,
                        "last_episode_date": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"

    decision = run_census_only_research_v0(census_path=census, output_root=output)

    assert decision["decision"] == "D_INSUFFICIENT_PROSPECTIVE_MICROSTRUCTURE_DATA"
    for name in (
        "README.md",
        "report.md",
        "run_manifest.json",
        "episode_level_results.csv",
        "method_summary.csv",
        "timing_window_comparison.csv",
        "baseline_comparison.csv",
        "first_bar_agreement_analysis.csv",
        "data_quality_summary.csv",
        "assessment_results.csv",
        "decision.json",
    ):
        assert (output / name).is_file(), name
