from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from stocker_prospective.live_bars import (
    AuditedFiveMinuteBarAdapter,
    HistoricalBarUpdate,
    KeepUpToDateBarFinalizer,
    checkpoint_for_bar,
)
from stocker_prospective.m1c_features import (
    HistoricalActivityBaseline,
    LiveFeatureBar,
    M1CCausalFeatureBuilder,
)

ROOT = Path(__file__).parents[1]
SCALING = (
    ROOT
    / "research/route-competition/20260722-broad-conflict-advance-hazard-v02"
    / "artifacts/primary/model_configurations.json"
)


def test_live_m1c_builder_uses_exact_frozen_feature_order_and_no_peer_inputs() -> None:
    builder = M1CCausalFeatureBuilder.from_scaling_artifact(SCALING)
    bars = tuple(
        LiveFeatureBar(
            symbol="AAL",
            session=date(2026, 7, 24),
            bar_ordinal=index,
            bar_start_timestamp=datetime(2026, 7, 24, 13, 30, tzinfo=UTC)
            + timedelta(minutes=5 * index),
            bar_complete_timestamp=datetime(2026, 7, 24, 13, 35, tzinfo=UTC)
            + timedelta(minutes=5 * index),
            open=100.0 + index * 0.1,
            high=100.3 + index * 0.1,
            low=99.8 + index * 0.1,
            close=100.1 + index * 0.1,
            volume=1_000.0 + index * 10,
            historical_relative_activity=0.8 + index * 0.02,
            finalised=True,
            source="fixture",
        )
        for index in range(6)
    )

    result = builder.build(symbol="AAL", checkpoint=6, completed_bars=bars)

    assert tuple(result.scaled_features) == (
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
    assert not {
        "signed_pressure",
        "tension",
        "posterior_entropy",
        "transition_probability",
    }.intersection(result.scaled_features)
    assert result.feature_available_timestamp_utc == bars[-1].bar_complete_timestamp


def test_live_m1c_builder_rejects_partial_missing_and_noncontiguous_bars() -> None:
    builder = M1CCausalFeatureBuilder.from_scaling_artifact(SCALING)
    bars = [
        LiveFeatureBar(
            symbol="AAL",
            session=date(2026, 7, 24),
            bar_ordinal=index,
            bar_start_timestamp=datetime(2026, 7, 24, 13, 30, tzinfo=UTC)
            + timedelta(minutes=5 * index),
            bar_complete_timestamp=datetime(2026, 7, 24, 13, 35, tzinfo=UTC)
            + timedelta(minutes=5 * index),
            open=100.0,
            high=100.2,
            low=99.8,
            close=100.1,
            volume=1_000.0,
            historical_relative_activity=1.0,
            finalised=True,
            source="fixture",
        )
        for index in range(6)
    ]
    bars[-1] = bars[-1].model_copy(update={"finalised": False})
    with pytest.raises(ValueError, match="finalised"):
        builder.build(symbol="AAL", checkpoint=6, completed_bars=tuple(bars))

    bars[-1] = bars[-1].model_copy(update={"finalised": True, "historical_relative_activity": None})
    with pytest.raises(ValueError, match="historical relative activity"):
        builder.build(symbol="AAL", checkpoint=6, completed_bars=tuple(bars))

    bars[-1] = bars[-1].model_copy(update={"historical_relative_activity": 1.0, "bar_ordinal": 7})
    with pytest.raises(ValueError, match="contiguous"):
        builder.build(symbol="AAL", checkpoint=6, completed_bars=tuple(bars))


def test_historical_activity_baseline_never_uses_same_session() -> None:
    baseline = HistoricalActivityBaseline(minimum_sessions=2)
    baseline.commit_session(
        symbol="AAL",
        session=date(2026, 7, 22),
        volume_by_ordinal={0: 100.0},
    )
    baseline.commit_session(
        symbol="AAL",
        session=date(2026, 7, 23),
        volume_by_ordinal={0: 300.0},
    )

    assert (
        baseline.relative_activity(
            symbol="AAL",
            session=date(2026, 7, 24),
            bar_ordinal=0,
            volume=400.0,
        )
        == 2.0
    )
    with pytest.raises(ValueError, match="chronology"):
        baseline.relative_activity(
            symbol="AAL",
            session=date(2026, 7, 23),
            bar_ordinal=0,
            volume=400.0,
        )


def update(
    start: datetime,
    *,
    finalised: bool,
    close: float = 100.1,
) -> HistoricalBarUpdate:
    return HistoricalBarUpdate(
        request_id=1,
        symbol="AAL",
        con_id=265598,
        bar_start_utc=start,
        provider_timestamp_utc=start + timedelta(minutes=5),
        received_timestamp_utc=start + timedelta(minutes=5, seconds=1),
        open=100.0,
        high=100.2,
        low=99.9,
        close=close,
        volume=1_000.0,
        wap=100.05,
        trade_count=20,
        source="ibkr_historical_keep_up_to_date",
        explicitly_finalised=finalised,
    )


def test_audited_bar_adapter_scores_only_final_rth_bars_and_indexes_checkpoint() -> None:
    adapter = AuditedFiveMinuteBarAdapter()
    first_start = datetime(2026, 7, 24, 13, 30, tzinfo=UTC)

    assert adapter.add(update(first_start, finalised=False)) == ()
    emitted = adapter.add(update(first_start, finalised=True))

    assert len(emitted) == 1
    assert emitted[0].checkpoint == 1
    assert emitted[0].bar_end_utc == first_start + timedelta(minutes=5)
    assert emitted[0].finalised is True
    assert emitted[0].source_completeness == "complete"
    assert adapter.add(update(first_start, finalised=True)) == ()

    with pytest.raises(ValueError, match="finalised bar changed"):
        adapter.add(update(first_start, finalised=True, close=100.2))


def test_new_york_checkpoint_mapping_rejects_non_rth_and_misalignment() -> None:
    assert checkpoint_for_bar(datetime(2026, 7, 24, 13, 30, tzinfo=UTC)) == 1
    assert checkpoint_for_bar(datetime(2026, 7, 24, 13, 55, tzinfo=UTC)) == 6
    with pytest.raises(ValueError, match="outside XNYS"):
        checkpoint_for_bar(datetime(2026, 7, 24, 13, 25, tzinfo=UTC))
    with pytest.raises(ValueError, match="aligned"):
        checkpoint_for_bar(datetime(2026, 7, 24, 13, 31, tzinfo=UTC))


def test_keep_up_to_date_waits_for_next_bar_before_finalising() -> None:
    activation = datetime(2026, 7, 24, 13, 30, tzinfo=UTC)
    finalizer = KeepUpToDateBarFinalizer(
        prospective_collection_start=activation,
    )
    finalizer.register(7, symbol="AAL", con_id=265598)

    def observe(start: datetime, close: float) -> tuple[HistoricalBarUpdate, ...]:
        return finalizer.add(
            request_id=7,
            bar_start_utc=start,
            provider_timestamp_utc=start + timedelta(minutes=5),
            received_timestamp_utc=start + timedelta(minutes=5, seconds=1),
            open=12.0,
            high=max(12.1, close),
            low=min(11.9, close),
            close=close,
            volume=100.0,
            wap=12.0,
            trade_count=10,
        )

    assert observe(activation, 12.01) == ()
    assert observe(activation, 12.02) == ()
    completed = observe(activation + timedelta(minutes=5), 12.03)
    assert len(completed) == 1
    assert completed[0].close == 12.02
    assert completed[0].explicitly_finalised is True
