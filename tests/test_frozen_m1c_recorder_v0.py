from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stocker_prospective.direction import FrozenDirectionRuntime
from stocker_prospective.frozen_m1c import (
    M1C_THRESHOLD,
    FreshEpisodeTracker,
    FrozenM1CRuntime,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHETYPE_ROOT = (
    ROOT
    / "research"
    / "directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0"
    / "artifacts"
    / "primary"
)


def _m1c_runtime() -> FrozenM1CRuntime:
    return FrozenM1CRuntime.from_artifacts(
        feature_manifest_path=ARCHETYPE_ROOT / "causal_movement_feature_manifest.json",
        threshold_path=ARCHETYPE_ROOT / "causal_movement_threshold.json",
    )


def _direction_runtime() -> FrozenDirectionRuntime:
    return FrozenDirectionRuntime.from_artifacts(
        model_configurations_path=ARCHETYPE_ROOT / "model_configurations.json",
        normalisation_path=ARCHETYPE_ROOT / "stock_local_normalisation_parameters.json",
        thresholds_path=ARCHETYPE_ROOT / "frozen_archetype_thresholds.json",
    )


def test_m1c_runtime_pins_exact_causal_feature_order_and_probability() -> None:
    runtime = _m1c_runtime()

    assert runtime.threshold == M1C_THRESHOLD == 0.488333710794033
    assert runtime.numeric_features[-12:] == (
        "arousal",
        "conviction",
        "prior_6_mean_range",
        "prior_6_price_travel",
        "prior_6_absolute_net_movement",
        "prior_6_activity_proxy",
        "recent_vs_earlier_range_ratio",
        "recent_vs_earlier_activity_ratio",
        "current_bar_range_vs_prior_6",
        "current_bar_activity_vs_prior_6",
        "current_bar_body_fraction",
        "current_bar_extreme_wick_fraction",
    )
    assert "signed_pressure" not in runtime.numeric_features
    assert "tension" not in runtime.numeric_features

    score = runtime.score(
        symbol="AAL",
        checkpoint=6,
        group_o_context={
            name: 0.0
            for name in runtime.numeric_features
            if not name.startswith("checkpoint_") and name not in runtime.causal_group_i_features
        },
        causal_group_i={name: 0.0 for name in runtime.causal_group_i_features},
    )

    assert score.probability == pytest.approx(0.3791098724444006, abs=1e-15)
    assert score.feature_order == runtime.numeric_features
    assert score.threshold_passed is False
    assert score.missing_feature_count == 0


def test_m1c_runtime_distinguishes_absent_group_o_keys_from_explicit_missing_values() -> None:
    runtime = _m1c_runtime()
    context = {name: None for name in runtime.required_group_o_features}
    absent = runtime.required_group_o_features[0]

    assert runtime.missing_group_o_features(context) == ()
    context.pop(absent)
    assert runtime.missing_group_o_features(context) == (absent,)


def test_fresh_episode_tracker_starts_on_first_above_and_enforces_thirty_minutes() -> None:
    tracker = FreshEpisodeTracker(threshold=M1C_THRESHOLD)
    session = date(2026, 7, 27)
    first = datetime(2026, 7, 27, 14, 0, tzinfo=UTC)

    initial = tracker.evaluate(
        symbol="AAL",
        session=session,
        checkpoint=6,
        trigger_bar_end=first,
        probability=0.60,
    )
    continued = tracker.evaluate(
        symbol="AAL",
        session=session,
        checkpoint=8,
        trigger_bar_end=first + timedelta(minutes=10),
        probability=0.61,
    )
    tracker.evaluate(
        symbol="AAL",
        session=session,
        checkpoint=10,
        trigger_bar_end=first + timedelta(minutes=20),
        probability=0.20,
    )
    suppressed = tracker.evaluate(
        symbol="AAL",
        session=session,
        checkpoint=12,
        trigger_bar_end=first + timedelta(minutes=25),
        probability=0.62,
    )
    tracker.evaluate(
        symbol="AAL",
        session=session,
        checkpoint=14,
        trigger_bar_end=first + timedelta(minutes=30),
        probability=0.20,
    )
    second = tracker.evaluate(
        symbol="AAL",
        session=session,
        checkpoint=16,
        trigger_bar_end=first + timedelta(minutes=35),
        probability=0.63,
    )

    assert initial.fresh_episode is True
    assert initial.previous_probability is None
    assert initial.episode_number == 1
    assert initial.prospective_entry_timestamp == first
    assert continued.raw_above_threshold is True
    assert continued.fresh_episode is False
    assert suppressed.fresh_episode is False
    assert suppressed.rejection_reason == "minimum_episode_spacing_not_met"
    assert second.fresh_episode is True
    assert second.episode_number == 2
    assert second.minutes_since_previous_episode == 35.0
    assert second.episode_id != initial.episode_id


@pytest.mark.parametrize("model_id", ["A1", "C1", "R1"])
def test_direction_runtime_reproduces_committed_assessment_rows(model_id: str) -> None:
    runtime = _direction_runtime()
    row = pd.read_parquet(ARCHETYPE_ROOT / "assessment_predictions.parquet").iloc[0]
    raw = {feature: row[f"raw__{feature}"] for feature in runtime.feature_names(model_id)}

    result = runtime.classify_one(
        model_id=model_id,
        raw_features=raw,
        symbol=str(row["stock"]),
        checkpoint=int(row["checkpoint"]),
        checkpoint_category=str(row["checkpoint_category"]),
        day_of_week=str(row["day_of_week"]),
    )

    assert result.probability_up == pytest.approx(
        float(row[f"{model_id}_probability"]),
        abs=1e-12,
    )
    assert result.action == str(row[f"{model_id}_action"])
    assert result.label.endswith("not validated")
