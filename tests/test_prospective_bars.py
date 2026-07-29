from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from stocker_prospective.bars import (
    CompletedBar,
    DiagnosticFiveMinuteBarAggregator,
    assess_bar_for_features,
)
from stocker_prospective.market_data import RealtimeBarUpdate

END = datetime(2026, 7, 24, 14, 35, tzinfo=UTC)


def bar(**updates: object) -> CompletedBar:
    payload: dict[str, object] = {
        "symbol": "AAL",
        "permanent_contract_id": 265598,
        "bar_start_utc": END - timedelta(minutes=5),
        "bar_end_utc": END,
        "session_date": date(2026, 7, 24),
        "open": 12.0,
        "high": 12.1,
        "low": 11.95,
        "close": 12.05,
        "activity_value": 101.0,
        "activity_semantic_label": "ibkr_trade_volume_not_eodhd_activity_proxy",
        "bar_source": "ibkr",
        "source_timestamp_utc": END,
        "receive_timestamp_utc": END + timedelta(seconds=1),
        "complete": True,
        "feature_as_of_utc": END,
        "scoring_checkpoint_utc": END + timedelta(seconds=2),
        "regular_trading_hours": True,
    }
    payload.update(updates)
    return CompletedBar.model_validate(payload)


def test_only_complete_exact_five_minute_bars_reach_feature_gate() -> None:
    accepted = assess_bar_for_features(
        bar(),
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=True,
    )
    partial = assess_bar_for_features(
        bar(complete=False),
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=True,
    )
    wrong_width = assess_bar_for_features(
        bar(bar_start_utc=END - timedelta(minutes=4)),
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=True,
    )

    assert accepted.eligible is True
    assert partial.rejection_reason == "partial_bar"
    assert wrong_width.rejection_reason == "non_five_minute_bar"


def test_late_callback_stale_feature_and_semantic_substitution_are_blocked() -> None:
    late = assess_bar_for_features(
        bar(receive_timestamp_utc=END + timedelta(seconds=3)),
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=True,
    )
    stale = assess_bar_for_features(
        bar(
            feature_as_of_utc=END - timedelta(minutes=1),
            scoring_checkpoint_utc=END + timedelta(seconds=2),
        ),
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=True,
    )
    semantic = assess_bar_for_features(
        bar(),
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=False,
    )

    assert late.rejection_reason == "callback_received_after_scoring_checkpoint"
    assert stale.rejection_reason == "stale_feature"
    assert semantic.rejection_reason == "blocked_feature_source_semantics_mismatch"


def test_non_rth_and_missing_values_are_rejections_not_forward_fills() -> None:
    outside = assess_bar_for_features(
        bar(regular_trading_hours=False),
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=True,
    )
    missing = assess_bar_for_features(
        bar(activity_value=None),
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=True,
    )

    assert outside.rejection_reason == "outside_frozen_regular_session"
    assert missing.rejection_reason == "missing_required_bar_value"


def test_ibkr_five_second_bars_form_one_completed_diagnostic_bar() -> None:
    aggregator = DiagnosticFiveMinuteBarAggregator()
    aggregator.register(7, symbol="AAL", permanent_contract_id=265598)
    completed: tuple[CompletedBar, ...] = ()
    start = datetime(2026, 7, 24, 14, 30, tzinfo=UTC)
    for index in range(60):
        timestamp = start + timedelta(seconds=index * 5)
        completed += aggregator.add(
            RealtimeBarUpdate(
                request_id=7,
                source_timestamp_utc=timestamp,
                receive_timestamp_utc=timestamp + timedelta(seconds=1),
                open=12.0 + index / 1000,
                high=12.1 + index / 1000,
                low=11.9 + index / 1000,
                close=12.05 + index / 1000,
                volume=10.0,
                wap=12.03,
                trade_count=2,
            )
        )
    completed += aggregator.add(
        RealtimeBarUpdate(
            request_id=7,
            source_timestamp_utc=start + timedelta(minutes=5),
            receive_timestamp_utc=start + timedelta(minutes=5, seconds=1),
            open=12.1,
            high=12.2,
            low=12.0,
            close=12.15,
            volume=10.0,
            wap=12.13,
            trade_count=2,
        )
    )

    assert len(completed) == 1
    result = completed[0]
    assert result.complete is True
    assert result.bar_start_utc == start
    assert result.bar_end_utc == start + timedelta(minutes=5)
    assert result.activity_value == 600.0
    assert result.activity_semantic_label.endswith("not_eodhd_historical_activity_proxy")
    assessment = assess_bar_for_features(
        result,
        maximum_feature_age=timedelta(seconds=15),
        source_semantics_allowed=False,
    )
    assert assessment.rejection_reason == "blocked_feature_source_semantics_mismatch"


def test_missing_five_second_bar_remains_partial_and_is_never_filled() -> None:
    aggregator = DiagnosticFiveMinuteBarAggregator()
    aggregator.register(7, symbol="AAL", permanent_contract_id=265598)
    start = datetime(2026, 7, 24, 14, 30, tzinfo=UTC)
    aggregator.add(
        RealtimeBarUpdate(
            request_id=7,
            source_timestamp_utc=start,
            receive_timestamp_utc=start + timedelta(seconds=1),
            open=12.0,
            high=12.1,
            low=11.9,
            close=12.05,
            volume=None,
            wap=None,
            trade_count=None,
        )
    )

    result = aggregator.flush()[0]
    assert result.complete is False
    assert result.activity_value is None
    assert result.source_timestamp_utc == start
    assert result.feature_as_of_utc == start
    assert (
        assess_bar_for_features(
            result,
            maximum_feature_age=timedelta(seconds=15),
            source_semantics_allowed=False,
        ).rejection_reason
        == "partial_bar"
    )


def test_fully_observed_final_rth_bucket_can_complete_without_a_later_bucket() -> None:
    aggregator = DiagnosticFiveMinuteBarAggregator()
    aggregator.register(7, symbol="AAL", permanent_contract_id=265598)
    start = datetime(2026, 7, 24, 19, 55, tzinfo=UTC)
    for index in range(60):
        timestamp = start + timedelta(seconds=index * 5)
        aggregator.add(
            RealtimeBarUpdate(
                request_id=7,
                source_timestamp_utc=timestamp,
                receive_timestamp_utc=timestamp + timedelta(seconds=1),
                open=12.0,
                high=12.1,
                low=11.9,
                close=12.05,
                volume=10.0,
                wap=12.03,
                trade_count=2,
            )
        )

    result = aggregator.flush(completed_through_utc=start + timedelta(minutes=5))[0]

    assert result.complete is True
    assert result.bar_end_utc == start + timedelta(minutes=5)
    assert result.source_timestamp_utc == result.bar_end_utc
