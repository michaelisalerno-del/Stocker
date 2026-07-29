from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.direction import FrozenDirectionRuntime
from stocker_prospective.direction_features import (
    DirectionFeatureBar,
    FrozenDirectionFeatureBuilder,
)

ROOT = Path(__file__).parents[1]
PRIMARY = (
    ROOT
    / "research/directional-readiness"
    / "20260726-stock-local-directional-archetypes-v0/artifacts/primary"
)


def bars() -> tuple[DirectionFeatureBar, ...]:
    start = datetime(2026, 7, 24, 13, 30, tzinfo=UTC)
    return tuple(
        DirectionFeatureBar(
            symbol="AAL",
            session=date(2026, 7, 24),
            bar_ordinal=index,
            bar_start_timestamp=start + timedelta(minutes=5 * index),
            bar_complete_timestamp=start + timedelta(minutes=5 * (index + 1)),
            open=100.0 + index * 0.1,
            high=100.3 + index * 0.1,
            low=99.8 + index * 0.1,
            close=100.1 + index * 0.1,
            volume=1_000.0 + index * 20,
            historical_relative_activity=0.8 + index * 0.02,
            stock_log_return=0.001 if index % 2 == 0 else -0.0002,
            market_log_return=0.0002 if index % 2 == 0 else -0.0001,
            finalised=True,
        )
        for index in range(8)
    )


def test_direction_feature_builder_ends_at_t_minus_one_and_excludes_trigger_bar() -> None:
    builder = FrozenDirectionFeatureBuilder.from_beta_artifact(
        PRIMARY / "stock_market_beta_parameters.csv"
    )
    original = bars()
    first = builder.build(symbol="AAL", checkpoint=8, completed_bars=original)
    changed_trigger = (*original[:-1], original[-1].model_copy(update={"close": 999.0}))
    second = builder.build(
        symbol="AAL",
        checkpoint=8,
        completed_bars=changed_trigger,
    )

    assert first.marker_bar_ordinal == 6
    assert first.trigger_bar_ordinal == 7
    assert first.maximum_direction_feature_timestamp == original[6].bar_complete_timestamp
    assert first.trigger_bar_excluded is True
    assert first.raw_features == second.raw_features


def test_direction_builder_emits_every_frozen_archetype_input() -> None:
    builder = FrozenDirectionFeatureBuilder.from_beta_artifact(
        PRIMARY / "stock_market_beta_parameters.csv"
    )
    runtime = FrozenDirectionRuntime.from_artifacts(
        model_configurations_path=PRIMARY / "model_configurations.json",
        normalisation_path=PRIMARY / "stock_local_normalisation_parameters.json",
        thresholds_path=PRIMARY / "frozen_archetype_thresholds.json",
    )
    result = builder.build(symbol="AAL", checkpoint=8, completed_bars=bars())

    for model_id in ("A1", "C1", "R1"):
        assert set(runtime.feature_names(model_id)).issubset(result.raw_features)
    classifications = runtime.classify(
        raw_features=result.raw_features,
        symbol="AAL",
        checkpoint=8,
        checkpoint_category=result.checkpoint_category,
        day_of_week=result.day_of_week,
    )
    assert set(classifications) == {"A1", "C1", "R1"}
    assert classifications["A1"].label == "prospective hypothesis — not validated"


def test_direction_builder_requires_exact_trigger_and_marker_bars() -> None:
    builder = FrozenDirectionFeatureBuilder.from_beta_artifact(
        PRIMARY / "stock_market_beta_parameters.csv"
    )
    with pytest.raises(ValueError, match="exactly checkpoint"):
        builder.build(symbol="AAL", checkpoint=8, completed_bars=bars()[:-1])
    partial = list(bars())
    partial[6] = partial[6].model_copy(update={"finalised": False})
    with pytest.raises(ValueError, match="finalised"):
        builder.build(symbol="AAL", checkpoint=8, completed_bars=tuple(partial))
