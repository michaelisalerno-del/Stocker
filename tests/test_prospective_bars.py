from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from stocker_prospective.bars import CompletedBar, assess_bar_for_features

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
