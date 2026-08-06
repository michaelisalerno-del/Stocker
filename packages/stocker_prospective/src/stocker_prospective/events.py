"""Immutable causal market-data event contracts."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from stocker_prospective.market_data import MarketDataType
from stocker_prospective.options import DteBucket


class DepthOperation(StrEnum):
    INSERT = "insert"
    UPDATE = "update"
    REMOVE = "remove"


class DepthSide(StrEnum):
    BID = "bid"
    ASK = "ask"


class RawMarketEvent(BaseModel):
    """Ordering identity shared by every append-only event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    received_timestamp_utc: datetime
    received_monotonic_ns: int = Field(ge=0)
    provider_timestamp_utc: datetime | None
    source_sequence: int = Field(ge=0)
    session: date
    symbol: str = Field(min_length=1)
    con_id: int
    request_id: int = Field(ge=0)

    @field_validator("received_timestamp_utc", "provider_timestamp_utc")
    @classmethod
    def _aware_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @property
    def ordering_timestamp(self) -> datetime:
        return self.provider_timestamp_utc or self.received_timestamp_utc


class RawCallbackEnvelopeEvent(BaseModel):
    """Original callback evidence retained independently of derived projection.

    This deliberately does not inherit ``RawMarketEvent``: provider and control
    callbacks may use a negative request ID or lack a provider monotonic
    timestamp. ``received_monotonic_ns`` is a deterministic ordering surrogate
    in that case; the original nullable value remains explicit in
    ``original_received_monotonic_ns``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    inbox_event_id: str = Field(min_length=1)
    received_timestamp_utc: datetime
    received_monotonic_ns: int = Field(ge=0)
    original_received_monotonic_ns: int | None = Field(default=None, ge=0)
    provider_timestamp_utc: datetime | None
    source_sequence: int = Field(gt=0)
    session: date
    symbol: str = Field(min_length=1)
    subscription_symbol: str | None
    con_id: int
    request_id: int
    callback_kind: str = Field(min_length=1)
    connection_generation: int = Field(ge=0)
    callback_classification: str = Field(min_length=1)
    subscription_owner: str | None
    stream_owner: dict[str, Any] | None
    original_payload: dict[str, Any]
    admission_run_id: str | None
    admission_recorder_generation: int | None
    recovery_disposition: Literal[
        "original_provider_callback",
        "scientifically_blocked_raw_only",
    ]

    @field_validator("received_timestamp_utc", "provider_timestamp_utc")
    @classmethod
    def _callback_timestamps_are_aware_utc(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("callback evidence timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @property
    def ordering_timestamp(self) -> datetime:
        return self.provider_timestamp_utc or self.received_timestamp_utc


class UnderlyingLevel1QuoteEvent(RawMarketEvent):
    bid: float | None
    bid_size: float | None
    ask: float | None
    ask_size: float | None
    last: float | None
    last_size: float | None
    volume: float | None = None
    market_data_type: MarketDataType
    source: str
    quote_valid: bool
    staleness_ms: float | None = Field(default=None, ge=0.0)
    tick_type: str
    exchange: str | None
    quote_attributes: dict[str, bool | int | float | str | None] = Field(default_factory=dict)
    halted: bool | None = None


class UnderlyingTickBidAskEvent(RawMarketEvent):
    bid: float
    bid_size: float
    ask: float
    ask_size: float
    bid_past_low: bool | None = None
    ask_past_high: bool | None = None
    exchange: str | None = None
    market_data_type: MarketDataType


class UnderlyingTickTradeEvent(RawMarketEvent):
    price: float
    size: float
    exchange: str | None
    conditions: tuple[str, ...]
    market_data_type: MarketDataType
    past_limit: bool | None = None
    unreported: bool | None = None
    halted: bool | None = None


class UnderlyingDepthEvent(RawMarketEvent):
    operation: DepthOperation
    position: int = Field(ge=0)
    side: DepthSide
    price: float | None
    size: float | None
    market_maker_or_exchange: str | None
    smart_depth: bool
    reset: bool = False


class DepthRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    position: int
    price: float
    size: float
    market_maker_or_exchange: str | None


class UnderlyingDepthSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_timestamp_utc: datetime
    received_monotonic_ns: int
    session: date
    symbol: str
    con_id: int
    bid_rows: tuple[DepthRow, ...]
    ask_rows: tuple[DepthRow, ...]
    total_bid_size: float | None
    total_ask_size: float | None
    depth_imbalance: float | None
    weighted_depth_imbalance: float | None
    distance_weighted_bid_liquidity: float | None
    distance_weighted_ask_liquidity: float | None
    bid_depth_additions: float
    bid_depth_removals: float
    ask_depth_additions: float
    ask_depth_removals: float
    bid_side_replenishment: float | None
    ask_side_replenishment: float | None
    depth_centroid_shift: float | None
    book_slope: float | None
    active_venues: int
    near_touch_bid_liquidity: float | None
    near_touch_ask_liquidity: float | None
    book_valid: bool
    reset_count: int
    smart_depth: bool

    @field_validator("snapshot_timestamp_utc")
    @classmethod
    def _snapshot_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot timestamp must be timezone-aware")
        return value.astimezone(UTC)


class UnderlyingDepthSnapshotEvent(RawMarketEvent):
    """Periodic derived state retained beside immutable raw depth callbacks."""

    trigger_event_id: str = Field(min_length=1)
    source: Literal["deterministic_depth_book_v0"] = "deterministic_depth_book_v0"
    snapshot: UnderlyingDepthSnapshot


class OptionQuoteEvent(RawMarketEvent):
    episode_id: str
    expiry: date
    dte: int = Field(ge=0)
    dte_bucket: DteBucket
    strike: float
    right: Literal["C", "P"]
    multiplier: int = Field(gt=0)
    exchange: str
    trading_class: str
    bid: float | None
    bid_size: float | None
    ask: float | None
    ask_size: float | None
    last: float | None
    last_size: float | None
    market_data_type: MarketDataType
    option_model_price: float | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    underlying_reference_price: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    option_computation_by_source: dict[
        str,
        dict[str, float | str | None],
    ] = Field(default_factory=dict)
    quote_attributes: dict[str, bool | int | float | str | None] = Field(default_factory=dict)


class FiveMinuteBarEvent(RawMarketEvent):
    bar_start_utc: datetime
    bar_end_utc: datetime
    checkpoint: int = Field(ge=1)
    open: float
    high: float
    low: float
    close: float
    volume_or_activity_field: float | None
    wap_where_available: float | None
    trade_count_where_available: int | None
    source: str
    source_completeness: str
    finalised: bool

    @field_validator("bar_start_utc", "bar_end_utc")
    @classmethod
    def _bar_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bar timestamps must be timezone-aware")
        return value.astimezone(UTC)


RawEvent = (
    RawCallbackEnvelopeEvent
    | UnderlyingLevel1QuoteEvent
    | UnderlyingTickBidAskEvent
    | UnderlyingTickTradeEvent
    | UnderlyingDepthEvent
    | UnderlyingDepthSnapshotEvent
    | OptionQuoteEvent
    | FiveMinuteBarEvent
)
