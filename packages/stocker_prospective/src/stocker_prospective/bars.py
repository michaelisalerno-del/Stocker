"""Completed-bar evidence contract and fail-closed feature gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, field_validator

from stocker_prospective.market_data import CallbackRequestError, RealtimeBarUpdate


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


@dataclass
class _FiveMinuteBucket:
    symbol: str
    permanent_contract_id: int
    start_utc: datetime
    updates: dict[datetime, RealtimeBarUpdate] = field(default_factory=dict)


class DiagnosticFiveMinuteBarAggregator:
    """Aggregate official 5-second bars without claiming frozen feature parity."""

    def __init__(self) -> None:
        self._identities: dict[int, tuple[str, int]] = {}
        self._buckets: dict[int, _FiveMinuteBucket] = {}

    def register(self, request_id: int, *, symbol: str, permanent_contract_id: int) -> None:
        identity = (symbol, permanent_contract_id)
        existing = self._identities.get(request_id)
        if existing is not None and existing != identity:
            raise CallbackRequestError("realtime_bar_request_identity_changed")
        self._identities[request_id] = identity

    @staticmethod
    def _bucket_start(timestamp: datetime) -> datetime:
        aware = timestamp.astimezone(UTC)
        epoch = int(aware.timestamp())
        return datetime.fromtimestamp((epoch // 300) * 300, tz=UTC)

    def add(self, update: RealtimeBarUpdate) -> tuple[CompletedBar, ...]:
        identity = self._identities.get(update.request_id)
        if identity is None:
            raise CallbackRequestError("unknown_realtime_bar_request")
        start = self._bucket_start(update.source_timestamp_utc)
        bucket = self._buckets.get(update.request_id)
        completed: tuple[CompletedBar, ...] = ()
        if bucket is not None and start < bucket.start_utc:
            raise CallbackRequestError("out_of_order_realtime_bar")
        if bucket is not None and start > bucket.start_utc:
            completed = (self._finalize(bucket, force_partial=False),)
            bucket = None
        if bucket is None:
            bucket = _FiveMinuteBucket(
                symbol=identity[0],
                permanent_contract_id=identity[1],
                start_utc=start,
            )
            self._buckets[update.request_id] = bucket
        bucket.updates[update.source_timestamp_utc.astimezone(UTC)] = update
        return completed

    def flush(
        self,
        *,
        completed_through_utc: datetime | None = None,
    ) -> tuple[CompletedBar, ...]:
        bars = tuple(
            self._finalize(
                bucket,
                force_partial=(
                    completed_through_utc is None
                    or completed_through_utc.astimezone(UTC)
                    < bucket.start_utc + timedelta(minutes=5)
                ),
            )
            for _, bucket in sorted(self._buckets.items())
        )
        self._buckets.clear()
        return bars

    @staticmethod
    def _finalize(bucket: _FiveMinuteBucket, *, force_partial: bool) -> CompletedBar:
        ordered = [bucket.updates[key] for key in sorted(bucket.updates)]
        end = bucket.start_utc + timedelta(minutes=5)
        expected = {bucket.start_utc + timedelta(seconds=5 * index) for index in range(60)}
        observed = set(bucket.updates)
        prices_complete = bool(ordered) and all(
            value is not None
            for update in ordered
            for value in (update.open, update.high, update.low, update.close)
        )
        complete = not force_partial and observed == expected and prices_complete
        opens = [update.open for update in ordered]
        highs = [update.high for update in ordered]
        lows = [update.low for update in ordered]
        closes = [update.close for update in ordered]
        volume_values = [update.volume for update in ordered]
        receive_timestamp = (
            max(update.receive_timestamp_utc for update in ordered) if ordered else bucket.start_utc
        )
        last_source_timestamp = (
            max(update.source_timestamp_utc for update in ordered) if ordered else bucket.start_utc
        )
        feature_as_of = end if complete else last_source_timestamp
        return CompletedBar(
            symbol=bucket.symbol,
            permanent_contract_id=bucket.permanent_contract_id,
            bar_start_utc=bucket.start_utc,
            bar_end_utc=end,
            session_date=bucket.start_utc.astimezone(ZoneInfo("America/New_York")).date(),
            open=opens[0] if prices_complete else None,
            high=max(value for value in highs if value is not None) if prices_complete else None,
            low=min(value for value in lows if value is not None) if prices_complete else None,
            close=closes[-1] if prices_complete else None,
            activity_value=(
                sum(value for value in volume_values if value is not None)
                if volume_values and all(value is not None for value in volume_values)
                else None
            ),
            activity_semantic_label=(
                "ibkr_realtime_bar_trade_volume_not_eodhd_historical_activity_proxy"
            ),
            bar_source="ibkr_realtime_bar_5_second_aggregation",
            source_timestamp_utc=end if complete else last_source_timestamp,
            receive_timestamp_utc=receive_timestamp,
            complete=complete,
            feature_as_of_utc=feature_as_of,
            scoring_checkpoint_utc=max(feature_as_of, receive_timestamp),
            regular_trading_hours=True,
        )


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
