"""Completed-bar evidence contract and fail-closed feature gate."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel, ConfigDict, field_validator


class CompletedBar(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    permanent_contract_id: int
    bar_start_utc: datetime
    bar_end_utc: datetime
    session_date: date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    activity_value: float | None
    activity_semantic_label: str
    bar_source: str
    source_timestamp_utc: datetime
    receive_timestamp_utc: datetime
    complete: bool
    feature_as_of_utc: datetime
    scoring_checkpoint_utc: datetime
    regular_trading_hours: bool

    @field_validator(
        "bar_start_utc",
        "bar_end_utc",
        "source_timestamp_utc",
        "receive_timestamp_utc",
        "feature_as_of_utc",
        "scoring_checkpoint_utc",
    )
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bar timestamps must be timezone-aware")
        return value.astimezone(UTC)


class BarAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    eligible: bool
    rejection_reason: str | None


def assess_bar_for_features(
    bar: CompletedBar,
    *,
    maximum_feature_age: timedelta,
    source_semantics_allowed: bool,
) -> BarAssessment:
    """Assess without filling, repairing, or changing any observed bar value."""

    if not bar.complete:
        return BarAssessment(eligible=False, rejection_reason="partial_bar")
    if bar.bar_end_utc - bar.bar_start_utc != timedelta(minutes=5):
        return BarAssessment(eligible=False, rejection_reason="non_five_minute_bar")
    if not bar.regular_trading_hours:
        return BarAssessment(
            eligible=False,
            rejection_reason="outside_frozen_regular_session",
        )
    values = (bar.open, bar.high, bar.low, bar.close, bar.activity_value)
    if any(value is None for value in values):
        return BarAssessment(
            eligible=False,
            rejection_reason="missing_required_bar_value",
        )
    if (
        bar.receive_timestamp_utc > bar.scoring_checkpoint_utc
        or bar.source_timestamp_utc > bar.scoring_checkpoint_utc
        or bar.feature_as_of_utc > bar.scoring_checkpoint_utc
    ):
        return BarAssessment(
            eligible=False,
            rejection_reason="callback_received_after_scoring_checkpoint",
        )
    if bar.scoring_checkpoint_utc - bar.feature_as_of_utc > maximum_feature_age:
        return BarAssessment(eligible=False, rejection_reason="stale_feature")
    if not source_semantics_allowed:
        return BarAssessment(
            eligible=False,
            rejection_reason="blocked_feature_source_semantics_mismatch",
        )
    assert bar.high is not None
    assert bar.low is not None
    assert bar.open is not None
    assert bar.close is not None
    if bar.high < max(bar.open, bar.close, bar.low) or bar.low > min(
        bar.open,
        bar.close,
        bar.high,
    ):
        return BarAssessment(eligible=False, rejection_reason="invalid_ohlc")
    return BarAssessment(eligible=True, rejection_reason=None)
