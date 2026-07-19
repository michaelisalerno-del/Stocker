"""Order-free IBKR observability records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum


class MarketDataType(IntEnum):
    """Official TWS market-data type callback values."""

    LIVE = 1
    FROZEN = 2
    DELAYED = 3
    DELAYED_FROZEN = 4


class ObservationClassification(StrEnum):
    """Quote-observability classes; only one is current complete top of book."""

    LIVE_TOP_OF_BOOK_OBSERVED = "LIVE_TOP_OF_BOOK_OBSERVED"
    LIVE_PARTIAL_QUOTE = "LIVE_PARTIAL_QUOTE"
    FROZEN_NON_CURRENT = "FROZEN_NON_CURRENT"
    DELAYED_NON_EXECUTABLE = "DELAYED_NON_EXECUTABLE"
    STALE = "STALE"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


class ReferenceQuoteUncertainty(StrEnum):
    """Labels for later quote-reference work, never fill certainty."""

    EXACT_REFERENCE_QUOTE_OBSERVED = "EXACT_REFERENCE_QUOTE_OBSERVED"
    GAP_REFERENCE_QUOTE_OBSERVED = "GAP_REFERENCE_QUOTE_OBSERVED"
    BOUNDED_NOT_EXACT = "BOUNDED_NOT_EXACT"
    ASSUMED = "ASSUMED"
    AMBIGUOUS = "AMBIGUOUS"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class ContractIdentity:
    """Stable current IBKR contract-resolution evidence."""

    research_symbol: str
    source_provider_symbol: str
    con_id: int | None
    symbol: str | None
    local_symbol: str | None
    security_type: str | None
    currency: str | None
    routing_exchange: str
    primary_exchange: str | None
    trading_class: str | None
    valid_exchanges: str | None
    minimum_tick: float | None
    timezone_identifier: str | None
    trading_hours: str | None
    liquid_hours: str | None
    resolution_timestamp: datetime
    api_tws_version: str | None
    resolution_status: str
    resolution_error: str | None


@dataclass(frozen=True)
class ContractRequest:
    """US stock request using SMART plus an explicit primary exchange when known."""

    research_symbol: str
    source_provider_symbol: str
    symbol: str
    currency: str = "USD"
    security_type: str = "STK"
    routing_exchange: str = "SMART"
    primary_exchange: str | None = None


@dataclass(frozen=True)
class QuoteSnapshot:
    """Raw callback aggregation returned by an observability client."""

    request_id: int
    requested_timestamp: datetime
    server_time_observation: datetime | None
    local_send_timestamp: datetime
    first_response_timestamp: datetime | None
    snapshot_completion_timestamp: datetime | None
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    last: float | None
    last_size: float | None
    market_data_type: MarketDataType | None
    snapshot_complete: bool
    error_code: int | None
    error_message: str | None
    connection_status: str


@dataclass(frozen=True)
class ObservationPlanItem:
    """One immutable prospective entry or exit quote-observation request."""

    observation_id: str
    event_id: str | None
    decision_id: str | None
    decision_timestamp: datetime
    planned_entry_reference_timestamp: datetime
    planned_exit_reference_timestamp: datetime
    planned_observation_timestamp: datetime
    symbol: str
    con_id: int
    required_observation_type: str
    maximum_collection_delay_seconds: float
    completion_status: str

    @property
    def planned_timestamp(self) -> datetime:
        """Compatibility alias for the specific observation timestamp."""

        return self.planned_observation_timestamp


@dataclass(frozen=True)
class QuoteObservationRecord:
    """Append-only top-of-book observation with no fill claim."""

    observation_id: str
    event_id: str | None
    decision_id: str | None
    request_id: int
    requested_timestamp: datetime
    ibkr_server_time_observation: datetime | None
    local_send_timestamp_utc: datetime
    first_response_timestamp_utc: datetime | None
    snapshot_completion_timestamp_utc: datetime | None
    symbol: str
    con_id: int
    exchange: str
    primary_exchange: str | None
    bid: float | None
    ask: float | None
    bid_size: float | None
    ask_size: float | None
    last: float | None
    last_size: float | None
    market_data_type: MarketDataType | None
    classification: ObservationClassification
    quote_age_or_timing_uncertainty_seconds: float | None
    subscription_status: str
    snapshot_complete: bool
    error_code: int | None
    error_message: str | None
    connection_status: str
    api_tws_version: str | None
    source_identifier: str
    collector_version: str
    collector_hash: str
    reference_uncertainty: ReferenceQuoteUncertainty
    fill_claim: bool = False
