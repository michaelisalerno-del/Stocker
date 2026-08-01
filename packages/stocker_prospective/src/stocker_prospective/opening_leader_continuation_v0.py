"""Frozen record-only opening-leader continuation prospective recorder V0.

The module's public functions are deliberately causal and order-free.  M1C is
accepted only as attached context and is absent from every ranking key.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stocker_prospective.database import EvidenceMetadata, ProspectiveRepository
from stocker_prospective.live_bars import xnys_session_bounds

RECORDER_VERSION_V0: Final[Literal["opening-leader-continuation-recorder-v0"]] = (
    "opening-leader-continuation-recorder-v0"
)
RECORDER_TITLE_V0: Final[str] = "Opening Leader Continuation Prospective Recorder V0"
FROZEN_HYPOTHESIS_V0: Final[str] = (
    "The single strongest stock in the frozen cohort at the causal opening checkpoint "
    "may continue outperforming by the end of the regular session."
)
RETROSPECTIVE_RESULT_V0: Final[str] = "session_close_persistence_only"
CANONICAL_COHORT_V0: Final[tuple[str, ...]] = (
    "AAL",
    "AAOI",
    "APLD",
    "ASTS",
    "CIFR",
    "HIMS",
    "IONQ",
    "IREN",
    "MARA",
    "MP",
    "MRNA",
    "MSTR",
    "NVTS",
    "QBTS",
    "RGTI",
    "RIOT",
    "RIVN",
    "SMCI",
    "SOFI",
    "WULF",
)
CANONICAL_COHORT_HASH_V0: Final[str] = (
    "ab2a55b55329b83e410ebdf46fa5d47f5a75a44798d01374a4d7f072da57b634"
)
FROZEN_SIGNAL_CHECKPOINTS_V0: Final[dict[int, Literal["primary", "secondary"]]] = {
    6: "primary",
    12: "secondary",
}
MINIMUM_COMPLETE_SLATE_V0: Final[int] = 15


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def checkpoint_timestamp_v0(session: date, checkpoint: int) -> datetime:
    """Map a checkpoint through the repository's canonical XNYS bar grid."""

    if checkpoint not in FROZEN_SIGNAL_CHECKPOINTS_V0:
        raise ValueError("opening leader V0 permits only C6 and C12")
    market_open, _ = xnys_session_bounds(session)
    return market_open + timedelta(minutes=5 * checkpoint)


class M1CContextV0(BaseModel):
    """Logging-only M1C context; no field is an admission or ranking input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    high_low_state: str
    tail_phase: str
    qualified_fresh_event_status: str
    movement_consumed: float | None
    source_completeness: str

    @field_validator("movement_consumed")
    @classmethod
    def _finite_movement(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("movement consumed must be finite when present")
        return value


class CausalCheckpointBarV0(BaseModel):
    """Exact causal input used to rank one stock at C6 or C12."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    session: date
    checkpoint: int
    bar_start_utc: datetime
    bar_end_utc: datetime
    available_at_utc: datetime
    regular_session_open: float
    checkpoint_open: float
    checkpoint_high: float
    checkpoint_low: float
    checkpoint_close: float
    session_open_source_id: str = Field(min_length=1)
    checkpoint_source_id: str = Field(min_length=1)
    source_timestamp_utc: datetime
    received_timestamp_utc: datetime
    source_completeness: Literal["complete", "partial"]
    duplicate_resolution: Literal["unique", "causally_resolved", "unresolved"]

    @field_validator(
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "source_timestamp_utc",
        "received_timestamp_utc",
    )
    @classmethod
    def _timestamp_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="causal bar timestamp")

    @model_validator(mode="after")
    def _valid_causal_ohlc(self) -> CausalCheckpointBarV0:
        prices = (
            self.regular_session_open,
            self.checkpoint_open,
            self.checkpoint_high,
            self.checkpoint_low,
            self.checkpoint_close,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in prices):
            raise ValueError("causal OHLC values must be finite and positive")
        if self.checkpoint_high < max(
            self.checkpoint_open,
            self.checkpoint_close,
            self.checkpoint_low,
        ) or self.checkpoint_low > min(
            self.checkpoint_open,
            self.checkpoint_close,
            self.checkpoint_high,
        ):
            raise ValueError("causal checkpoint OHLC is inconsistent")
        if self.bar_end_utc - self.bar_start_utc != timedelta(minutes=5):
            raise ValueError("checkpoint input must identify one exact five-minute bar")
        if self.available_at_utc < self.bar_end_utc:
            raise ValueError("bar cannot be available before its close")
        return self


class RankedStockV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    symbol: str
    open_to_checkpoint_return_bps: float
    regular_session_open: float
    checkpoint_close: float
    input_identity: str
    causal_input: CausalCheckpointBarV0
    m1c_context: M1CContextV0 | None


class OpeningLeaderRankingV0(BaseModel):
    """Complete deterministic slate result, including every exclusion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recorder_version: Literal["opening-leader-continuation-recorder-v0"]
    session: date
    checkpoint: int
    checkpoint_role: Literal["primary", "secondary"]
    signal_timestamp_utc: datetime
    evaluated_at_utc: datetime
    cohort_hash: str
    slate_hash: str
    eligible: bool
    failure_reasons: tuple[str, ...]
    exclusions: dict[str, str]
    exact_slate_membership: tuple[str, ...]
    slate_size: int
    ranking: tuple[RankedStockV0, ...]
    rank_1: RankedStockV0 | None
    rank_2: RankedStockV0 | None
    rank_1_minus_rank_2_bps: float | None
    selected_identity: Literal["rank_1"] = "rank_1"
    direction: Literal["LONG"] = "LONG"

    @field_validator("signal_timestamp_utc", "evaluated_at_utc")
    @classmethod
    def _ranking_timestamp_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="ranking timestamp")


class ObservationTargetV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    target_timestamp_utc: datetime
    source_kind: Literal["level1_quote", "official_bar_close_reference"]
    executable: bool
    rank_persistence_required: bool = False

    @field_validator("target_timestamp_utc")
    @classmethod
    def _target_timestamp_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="observation target")


class ObservationScheduleV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session: date
    signal_timestamp_utc: datetime
    e0_timestamp_utc: datetime
    underlying_targets: tuple[ObservationTargetV0, ...]
    option_snapshot_names: tuple[str, ...]


class UnderlyingQuoteV0(BaseModel):
    """One immutable IBKR top-of-book observation with explicit freshness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quote_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    target_timestamp_utc: datetime
    actual_quote_timestamp_utc: datetime
    provider_timestamp_utc: datetime | None
    received_timestamp_utc: datetime
    timestamp_provenance: Literal["provider", "receive"]
    last: float | None
    bid: float | None
    ask: float | None
    midpoint: float | None
    bid_size: float | None
    ask_size: float | None
    spread_dollars: float | None
    spread_bps: float | None
    source: str
    market_data_status: Literal["live", "frozen", "delayed", "delayed_frozen", "unknown"]
    quote_age_seconds: float | None
    halted: bool | None
    data_quality_flags: tuple[str, ...]
    valid_for_signal: bool

    @field_validator(
        "target_timestamp_utc",
        "actual_quote_timestamp_utc",
        "provider_timestamp_utc",
        "received_timestamp_utc",
    )
    @classmethod
    def _quote_timestamp_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_utc(value, label="quote timestamp")


class UnderlyingShadowReturnV0(BaseModel):
    """Long-only record calculation; no position or order semantics exist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_quote_id: str
    exit_quote_id: str
    conservative_ask_to_bid_gross_bps: float | None
    conservative_ask_to_bid_net_bps: float | None
    midpoint_to_midpoint_diagnostic_bps: float | None
    last_to_last_diagnostic_bps: float | None
    configured_fee_bps: float
    friction_diagnostics_bps: dict[str, float | None]
    official_close_reference: float | None
    official_close_reference_bps: float | None
    official_close_executable: Literal[False] = False


class OptionContractRequestV0(BaseModel):
    """One exact bounded qualification request, never an order contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    underlying: str
    underlying_con_id: int
    expiry: date
    strike: float
    right: Literal["C", "P"]
    multiplier: int = 100
    exchange: str
    trading_class: str


class OptionChainSelectionV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["AVAILABLE", "UNAVAILABLE"]
    reason: str | None
    underlying: str
    spot: float
    selected_expiries: tuple[date, ...]
    selected_strikes_by_expiry: dict[str, tuple[float, ...]]
    requests: tuple[OptionContractRequestV0, ...]
    maximum_expiries: Literal[2] = 2
    maximum_strikes_per_expiry: Literal[5] = 5
    spot_band_fraction: float = Field(default=0.15, ge=0.15, le=0.15)
    full_chain_streaming: Literal[False] = False


class OptionQuoteV0(BaseModel):
    """One exact-contract option snapshot with no executable recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str
    underlying: str
    con_id: int
    right: Literal["C", "P"]
    strike: float
    expiry: date
    multiplier: int
    trading_class: str
    exchange: str
    captured_at_utc: datetime
    provider_timestamp_utc: datetime | None
    received_timestamp_utc: datetime
    quote_timestamp_utc: datetime
    timestamp_provenance: Literal["provider", "receive", "capture_fallback"]
    bid: float | None
    ask: float | None
    midpoint: float | None
    bid_size: float | None
    ask_size: float | None
    last: float | None
    volume: float | None
    open_interest: float | None
    implied_volatility: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    underlying_reference_price: float | None
    greeks_source: Literal["model"]
    option_computation_by_source: dict[str, dict[str, float | None]]
    quote_age_seconds: float | None
    market_data_status: Literal["live", "frozen", "delayed", "delayed_frozen", "unknown"]
    stale: bool
    available: bool
    data_quality_flags: tuple[str, ...]

    @field_validator(
        "captured_at_utc",
        "provider_timestamp_utc",
        "received_timestamp_utc",
        "quote_timestamp_utc",
    )
    @classmethod
    def _option_timestamp_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware_utc(value, label="option quote timestamp")

    @classmethod
    def from_snapshot(
        cls,
        *,
        snapshot_id: str,
        underlying: str,
        con_id: int,
        right: Literal["C", "P"],
        strike: float,
        expiry: date,
        multiplier: int,
        trading_class: str,
        exchange: str,
        captured_at_utc: datetime,
        provider_timestamp_utc: datetime | None,
        received_timestamp_utc: datetime | None = None,
        values: dict[str, object],
        maximum_quote_age_seconds: float,
    ) -> OptionQuoteV0:
        if maximum_quote_age_seconds <= 0.0:
            raise ValueError("maximum option quote age must be positive")
        captured = _aware_utc(captured_at_utc, label="option capture timestamp")
        provider = (
            None
            if provider_timestamp_utc is None
            else _aware_utc(provider_timestamp_utc, label="option provider timestamp")
        )
        received = (
            captured
            if received_timestamp_utc is None
            else _aware_utc(received_timestamp_utc, label="option receive timestamp")
        )
        if provider is not None:
            quote_timestamp = provider
            timestamp_provenance: Literal["provider", "receive", "capture_fallback"] = (
                "provider"
            )
        elif received_timestamp_utc is not None:
            quote_timestamp = received
            timestamp_provenance = "receive"
        else:
            quote_timestamp = captured
            timestamp_provenance = "capture_fallback"
        bid = _finite_optional(values.get("bid"))
        ask = _finite_optional(values.get("ask"))
        flags: list[str] = []
        if bid is None or bid < 0.0:
            flags.append("bid_unavailable")
        if ask is None or ask <= 0.0:
            flags.append("ask_unavailable")
        if bid is not None and ask is not None and bid > ask:
            flags.append("quote_crossed")
        midpoint = (
            None
            if bid is None or ask is None or bid < 0.0 or ask <= 0.0 or bid > ask
            else (bid + ask) / 2.0
        )
        age = max(0.0, (captured - quote_timestamp).total_seconds())
        if provider is None and timestamp_provenance == "receive":
            flags.append("provider_timestamp_unavailable_receive_fallback")
        elif provider is None:
            flags.append("source_timestamp_unavailable_capture_fallback")
        if quote_timestamp > captured:
            flags.append("source_timestamp_in_future")
        elif age > maximum_quote_age_seconds:
            flags.append("quote_stale")
        raw_status = str(values.get("market_data_type", "unknown")).lower()
        allowed_statuses = {"live", "frozen", "delayed", "delayed_frozen"}
        status = raw_status if raw_status in allowed_statuses else "unknown"
        if status != "live":
            flags.append("market_data_not_live")
        unavailable_flags = {
            "bid_unavailable",
            "ask_unavailable",
            "quote_crossed",
            "source_timestamp_unavailable_capture_fallback",
            "source_timestamp_in_future",
            "quote_stale",
            "market_data_not_live",
        }
        return cls(
            snapshot_id=snapshot_id,
            underlying=underlying,
            con_id=con_id,
            right=right,
            strike=strike,
            expiry=expiry,
            multiplier=multiplier,
            trading_class=trading_class,
            exchange=exchange,
            captured_at_utc=captured,
            provider_timestamp_utc=provider,
            received_timestamp_utc=received,
            quote_timestamp_utc=quote_timestamp,
            timestamp_provenance=timestamp_provenance,
            bid=bid,
            ask=ask,
            midpoint=midpoint,
            bid_size=_finite_optional(values.get("bid_size")),
            ask_size=_finite_optional(values.get("ask_size")),
            last=_finite_optional(values.get("last")),
            volume=_finite_optional(values.get("volume")),
            open_interest=_finite_optional(values.get("open_interest")),
            implied_volatility=_finite_optional(values.get("implied_volatility")),
            delta=_finite_optional(values.get("delta")),
            gamma=_finite_optional(values.get("gamma")),
            theta=_finite_optional(values.get("theta")),
            vega=_finite_optional(values.get("vega")),
            underlying_reference_price=_finite_optional(
                values.get("underlying_reference_price")
            ),
            greeks_source="model",
            option_computation_by_source=cast(
                dict[str, dict[str, float | None]],
                values.get("option_computation_by_source", {}),
            ),
            quote_age_seconds=age,
            market_data_status=cast(Any, status),
            stale="quote_stale" in flags,
            available=not unavailable_flags.intersection(flags),
            data_quality_flags=tuple(flags),
        )


class OptionDiagnosticV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["P20", "P30", "BPS20"]
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    reason: str | None
    short_con_id: int | None
    long_con_id: int | None
    short_target_delta: float
    entry_short_mark: float | None
    entry_long_mark: float | None
    entry_credit: float | None
    order_authorized: Literal[False] = False


class OpeningLeaderEvidenceRecordV0(BaseModel):
    """One immutable row in the existing prospective evidence database."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_id: str
    run_id: str
    recorder_version: Literal["opening-leader-continuation-recorder-v0"]
    deployment_receipt_id: str
    session: date
    checkpoint: Literal[6, 12]
    signal_timestamp_utc: datetime
    selected_symbol: str | None
    record_type: str
    observation_name: str
    observed_at_utc: datetime
    original_stable_id: str | None
    cohort_hash: str
    contract_hash: str
    code_hash: str
    data_quality_flags: tuple[str, ...]
    payload: dict[str, Any]
    content_hash: str

    @field_validator("signal_timestamp_utc", "observed_at_utc")
    @classmethod
    def _evidence_timestamp_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="evidence timestamp")


class RankPersistenceV0(BaseModel):
    """Diagnostic state for the original leader; never a signal gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_leader: str
    current_rank: int
    remains_rank_1: bool
    remains_top_2: bool
    remains_above_cohort_median: bool
    return_since_signal_bps: float
    drawdown_from_signal_bps: float
    maximum_favourable_excursion_bps: float
    maximum_adverse_excursion_bps: float
    signal_admission_changed: Literal[False] = False


class OpeningLeaderFreezeIdentityV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    deployment_receipt_id: str
    freeze_completed_at_utc: datetime
    contract_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    code_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    cohort_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_hashes_signed: Literal[True]
    order_routing_disabled: Literal[True]
    protected_historical_outcomes_accessed: Literal[False]

    @field_validator("freeze_completed_at_utc")
    @classmethod
    def _freeze_timestamp_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="deployment freeze timestamp")


class OptionSnapshotCaptureV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str
    observation_name: str
    captured_at_utc: datetime
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    reason: str | None
    selection: OptionChainSelectionV0 | None
    quotes: tuple[OptionQuoteV0, ...]

    @field_validator("captured_at_utc")
    @classmethod
    def _snapshot_timestamp_aware(cls, value: datetime) -> datetime:
        return _aware_utc(value, label="option snapshot timestamp")


class OpeningLeaderPollResultV0(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    created_signal_receipts: tuple[str, ...]
    created_failures: tuple[str, ...]
    created_observations: tuple[str, ...]
    created_option_snapshots: tuple[str, ...]


class ProspectiveBoundaryErrorV0(RuntimeError):
    """The recorder was asked to use pre-freeze or historical input."""


MetadataFactoryV0 = Callable[[datetime, tuple[datetime, ...]], EvidenceMetadata]
BarProviderV0 = Callable[[date, int], tuple[CausalCheckpointBarV0, ...]]
UnderlyingQuoteProviderV0 = Callable[
    [str, int, str, datetime, datetime],
    UnderlyingQuoteV0 | None,
]
OptionSnapshotProviderV0 = Callable[
    [str, int, str, float, datetime],
    OptionSnapshotCaptureV0,
]
RankPersistenceProviderV0 = Callable[
    [str, date, int, datetime, float],
    RankPersistenceV0 | None,
]
OfficialCloseProviderV0 = Callable[
    [str, date, datetime],
    tuple[float, str, datetime] | None,
]


class OpeningLeaderContinuationRecorderV0:
    """Restart-idempotent orchestrator integrated into the existing recorder loop."""

    def __init__(
        self,
        *,
        store: OpeningLeaderEvidenceStoreV0,
        freeze_identity: OpeningLeaderFreezeIdentityV0,
        prospective_start_utc: datetime,
        metadata_factory: MetadataFactoryV0,
        bar_provider: BarProviderV0,
        underlying_quote_provider: UnderlyingQuoteProviderV0,
        option_snapshot_provider: OptionSnapshotProviderV0,
        rank_persistence_provider: RankPersistenceProviderV0,
        official_close_provider: OfficialCloseProviderV0,
    ) -> None:
        prospective_start = _aware_utc(
            prospective_start_utc,
            label="opening-leader prospective start",
        )
        if freeze_identity.deployment_receipt_id != store.deployment_receipt_id:
            raise ValueError("deployment receipt identity differs from evidence store")
        if freeze_identity.contract_hash != store.contract_hash:
            raise ValueError("deployment contract hash differs from evidence store")
        if freeze_identity.code_hash != store.code_hash:
            raise ValueError("deployment code hash differs from evidence store")
        if freeze_identity.cohort_hash != store.cohort_hash:
            raise ValueError("deployment cohort hash differs from evidence store")
        self.store = store
        self.freeze_identity = freeze_identity
        self.prospective_start_utc = prospective_start
        self.boundary_utc = max(
            prospective_start,
            freeze_identity.freeze_completed_at_utc,
        )
        self.metadata_factory = metadata_factory
        self.bar_provider = bar_provider
        self.underlying_quote_provider = underlying_quote_provider
        self.option_snapshot_provider = option_snapshot_provider
        self.rank_persistence_provider = rank_persistence_provider
        self.official_close_provider = official_close_provider

    def _metadata(self, observed: datetime, *sources: datetime) -> EvidenceMetadata:
        source_values = sources or (observed,)
        metadata = self.metadata_factory(observed, tuple(source_values))
        if metadata.recorded_at_utc < self.boundary_utc:
            raise ProspectiveBoundaryErrorV0("historical backfill is forbidden")
        return metadata

    def _assert_quote_after_boundary(self, quote: UnderlyingQuoteV0 | None) -> None:
        if quote is None:
            return
        source_timestamp = quote.provider_timestamp_utc or quote.actual_quote_timestamp_utc
        if (
            source_timestamp < self.boundary_utc
            or quote.received_timestamp_utc < self.boundary_utc
        ):
            raise ProspectiveBoundaryErrorV0(
                "pre-freeze market data observed; a new recorder version is required"
            )

    @staticmethod
    def _index(
        records: tuple[OpeningLeaderEvidenceRecordV0, ...],
    ) -> dict[tuple[int, str, str], OpeningLeaderEvidenceRecordV0]:
        return {
            (record.checkpoint, record.record_type, record.observation_name): record
            for record in records
            if record.original_stable_id is None
        }

    def _append(
        self,
        *,
        observed: datetime,
        session: date,
        checkpoint: int,
        selected_symbol: str | None,
        record_type: str,
        observation_name: str,
        payload: Mapping[str, object],
        flags: tuple[str, ...] = (),
        source_timestamps: tuple[datetime, ...] = (),
    ) -> OpeningLeaderEvidenceRecordV0:
        metadata = self._metadata(observed, *source_timestamps)
        return self.store.append(
            metadata,
            stable_id=stable_evidence_id_v0(
                deployment_receipt_id=self.freeze_identity.deployment_receipt_id,
                session=session,
                checkpoint=checkpoint,
                record_type=record_type,
                observation_name=observation_name,
            ),
            session=session,
            checkpoint=checkpoint,
            signal_timestamp_utc=checkpoint_timestamp_v0(session, checkpoint),
            selected_symbol=selected_symbol,
            record_type=record_type,
            observation_name=observation_name,
            payload=payload,
            data_quality_flags=flags,
        )

    def _capture_option_snapshot(
        self,
        *,
        observed: datetime,
        session: date,
        checkpoint: int,
        symbol: str,
        observation_name: str,
        quote: UnderlyingQuoteV0,
    ) -> bool:
        reference = quote.midpoint or quote.last
        if reference is None:
            capture = OptionSnapshotCaptureV0(
                snapshot_id=f"unavailable-{checkpoint}-{observation_name}",
                observation_name=observation_name,
                captured_at_utc=observed,
                status="UNAVAILABLE",
                reason="underlying_reference_unavailable",
                selection=None,
                quotes=(),
            )
        else:
            try:
                capture = self.option_snapshot_provider(
                    symbol,
                    checkpoint,
                    observation_name,
                    reference,
                    observed,
                )
            except (RuntimeError, TimeoutError, ValueError) as exc:
                capture = OptionSnapshotCaptureV0(
                    snapshot_id=f"unavailable-{checkpoint}-{observation_name}",
                    observation_name=observation_name,
                    captured_at_utc=observed,
                    status="UNAVAILABLE",
                    reason=f"{type(exc).__name__}:{exc}",
                    selection=None,
                    quotes=(),
                )
        recorded_at = max(observed, capture.captured_at_utc)
        self._append(
            observed=recorded_at,
            session=session,
            checkpoint=checkpoint,
            selected_symbol=symbol,
            record_type="option_snapshot",
            observation_name=observation_name,
            payload=capture.model_dump(mode="json"),
            flags=(
                ()
                if capture.reason is None
                else (capture.reason,)
            ),
            source_timestamps=(capture.captured_at_utc,),
        )
        for option_quote in capture.quotes:
            self._append(
                observed=max(recorded_at, option_quote.captured_at_utc),
                session=session,
                checkpoint=checkpoint,
                selected_symbol=symbol,
                record_type="option_quote",
                observation_name=f"{observation_name}:{option_quote.con_id}",
                payload=option_quote.model_dump(mode="json"),
                flags=option_quote.data_quality_flags,
                source_timestamps=(
                    option_quote.quote_timestamp_utc,
                    option_quote.received_timestamp_utc,
                ),
            )
        return True

    def _record_unavailable_option_snapshot(
        self,
        *,
        observed: datetime,
        session: date,
        checkpoint: int,
        symbol: str,
        observation_name: str,
        reason: str,
    ) -> None:
        capture = OptionSnapshotCaptureV0(
            snapshot_id=f"unavailable-{checkpoint}-{observation_name}",
            observation_name=observation_name,
            captured_at_utc=observed,
            status="UNAVAILABLE",
            reason=reason,
            selection=None,
            quotes=(),
        )
        self._append(
            observed=observed,
            session=session,
            checkpoint=checkpoint,
            selected_symbol=symbol,
            record_type="option_snapshot",
            observation_name=observation_name,
            payload=capture.model_dump(mode="json"),
            flags=(reason,),
            source_timestamps=(observed,),
        )

    def _record_quote_observation(
        self,
        *,
        observed: datetime,
        session: date,
        checkpoint: int,
        symbol: str,
        observation_name: str,
        quote: UnderlyingQuoteV0,
        signal_quote: UnderlyingQuoteV0,
        entry_quote: UnderlyingQuoteV0 | None,
    ) -> OpeningLeaderEvidenceRecordV0:
        payload = self._quote_observation_payload(
            session=session,
            checkpoint=checkpoint,
            symbol=symbol,
            observation_name=observation_name,
            quote=quote,
            signal_quote=signal_quote,
            entry_quote=entry_quote,
        )
        return self._append(
            observed=observed,
            session=session,
            checkpoint=checkpoint,
            selected_symbol=symbol,
            record_type="underlying_observation",
            observation_name=observation_name,
            payload=payload,
            flags=quote.data_quality_flags,
            source_timestamps=(
                quote.actual_quote_timestamp_utc,
                quote.received_timestamp_utc,
            ),
        )

    def _quote_observation_payload(
        self,
        *,
        session: date,
        checkpoint: int,
        symbol: str,
        observation_name: str,
        quote: UnderlyingQuoteV0,
        signal_quote: UnderlyingQuoteV0,
        entry_quote: UnderlyingQuoteV0 | None,
    ) -> dict[str, object]:
        persistence = (
            None
            if entry_quote is None
            or observation_name not in {"E0", "E1", "E2", "H30", "H60", "H120"}
            else self.rank_persistence_provider(
                symbol,
                session,
                checkpoint,
                quote.actual_quote_timestamp_utc,
                signal_quote.midpoint
                or signal_quote.last
                or cast(float, signal_quote.ask),
            )
        )
        shadow = (
            None
            if entry_quote is None
            else calculate_underlying_shadow_return_v0(
                entry=entry_quote,
                exit_quote=quote,
                configured_fee_bps=1.0,
                friction_diagnostics_bps=(0.0, 1.0, 5.0, 10.0),
            )
        )
        return {
            "quote": quote.model_dump(mode="json"),
            "shadow_return": None if shadow is None else shadow.model_dump(mode="json"),
            "rank_persistence": (
                None if persistence is None else persistence.model_dump(mode="json")
            ),
        }

    def _append_late_quote_observation(
        self,
        *,
        original: OpeningLeaderEvidenceRecordV0,
        observed: datetime,
        session: date,
        checkpoint: int,
        symbol: str,
        observation_name: str,
        quote: UnderlyingQuoteV0,
        signal_quote: UnderlyingQuoteV0,
        entry_quote: UnderlyingQuoteV0 | None,
    ) -> OpeningLeaderEvidenceRecordV0:
        payload = self._quote_observation_payload(
            session=session,
            checkpoint=checkpoint,
            symbol=symbol,
            observation_name=observation_name,
            quote=quote,
            signal_quote=signal_quote,
            entry_quote=entry_quote,
        )
        return self.store.append_late_or_correction(
            self._metadata(
                observed,
                quote.actual_quote_timestamp_utc,
                quote.received_timestamp_utc,
            ),
            original_stable_id=original.stable_id,
            correction_kind="late_observation",
            payload=payload,
            data_quality_flags=(
                *quote.data_quality_flags,
                "late_data_not_rewriting_original",
            ),
        )

    def _reconcile_official_close(
        self,
        *,
        observed: datetime,
        market_close: datetime,
        session: date,
        checkpoint: int,
        symbol: str,
        entry_quote: UnderlyingQuoteV0 | None,
        index: dict[tuple[int, str, str], OpeningLeaderEvidenceRecordV0],
        linked_by_original: Mapping[str, OpeningLeaderEvidenceRecordV0],
    ) -> bool:
        if observed < market_close:
            return False
        official_key = (checkpoint, "official_close_reference", "OFFICIAL_CLOSE")
        official_record = index.get(official_key)
        effective_official = (
            None
            if official_record is None
            else linked_by_original.get(official_record.stable_id, official_record)
        )
        official = self.official_close_provider(symbol, session, observed)
        if official is not None:
            close, source_id, source_timestamp = official
            official_payload = {
                "official_close": close,
                "source_id": source_id,
                "source_timestamp_utc": source_timestamp.isoformat(),
                "executable": False,
                "entry_quote_id": None if entry_quote is None else entry_quote.quote_id,
                "ask_entry_to_official_close_reference_bps": (
                    None if entry_quote is None else _return_bps(entry_quote.ask, close)
                ),
            }
            if (
                effective_official is not None
                and effective_official.payload.get("source_id") == source_id
                and effective_official.payload.get("official_close") == close
            ):
                return False
            if official_record is None:
                official_record = self._append(
                    observed=observed,
                    session=session,
                    checkpoint=checkpoint,
                    selected_symbol=symbol,
                    record_type="official_close_reference",
                    observation_name="OFFICIAL_CLOSE",
                    payload=official_payload,
                    source_timestamps=(source_timestamp,),
                )
                index[official_key] = official_record
            else:
                correction_kind: Literal["late_observation", "correction"] = (
                    "correction"
                    if effective_official is not None
                    and effective_official.payload.get("official_close") is not None
                    else "late_observation"
                )
                self.store.append_late_or_correction(
                    self._metadata(observed, source_timestamp),
                    original_stable_id=official_record.stable_id,
                    correction_kind=correction_kind,
                    payload=official_payload,
                    data_quality_flags=(
                        (
                            "correction_not_rewriting_original"
                            if correction_kind == "correction"
                            else "late_data_not_rewriting_original"
                        ),
                        "official_close_non_executable",
                    ),
                )
            return True
        if (
            official_record is None
            and observed >= market_close + timedelta(minutes=31)
        ):
            official_record = self._append(
                observed=observed,
                session=session,
                checkpoint=checkpoint,
                selected_symbol=symbol,
                record_type="official_close_reference",
                observation_name="OFFICIAL_CLOSE",
                payload={
                    "status": "UNAVAILABLE",
                    "reason": "official_close_not_causally_available",
                    "executable": False,
                },
                flags=("official_close_not_causally_available",),
            )
            index[official_key] = official_record
            return True
        return False

    @staticmethod
    def _quote_from_record(record: OpeningLeaderEvidenceRecordV0) -> UnderlyingQuoteV0:
        payload = record.payload.get("quote")
        if not isinstance(payload, dict):
            raise ValueError("underlying observation quote payload is missing")
        return UnderlyingQuoteV0.model_validate(payload)

    def outstanding_sessions(self, *, now: datetime) -> tuple[date, ...]:
        """Return post-freeze sessions whose existing receipts need terminal recovery."""

        observed = _aware_utc(now, label="opening-leader recovery timestamp")
        run_id = self._metadata(observed).run_id
        records = self.store.records_for_run(run_id)
        linked = {
            record.original_stable_id: record
            for record in records
            if record.original_stable_id is not None
        }
        sessions: set[date] = set()
        for signal in records:
            if signal.record_type != "signal_receipt" or signal.original_stable_id is not None:
                continue
            _, market_close = xnys_session_bounds(signal.session)
            if observed < market_close:
                continue
            matching = tuple(
                record
                for record in records
                if record.session == signal.session
                and record.checkpoint == signal.checkpoint
                and record.original_stable_id is None
            )
            e0 = next(
                (
                    record
                    for record in matching
                    if record.record_type == "underlying_observation"
                    and record.observation_name == "E0"
                ),
                None,
            )
            official = any(
                record.record_type == "official_close_reference"
                and record.observation_name == "OFFICIAL_CLOSE"
                for record in matching
            )
            final = any(
                record.record_type == "underlying_observation"
                and record.observation_name == "FINAL_CONTINUOUS"
                for record in matching
            )
            effective_e0 = None if e0 is None else linked.get(e0.stable_id, e0)
            e0_has_quote = effective_e0 is not None and isinstance(
                effective_e0.payload.get("quote"),
                dict,
            )
            observation_names = {
                record.observation_name
                for record in matching
                if record.record_type == "underlying_observation"
            }
            option_names = {
                record.observation_name
                for record in matching
                if record.record_type == "option_snapshot"
            }
            required_observations = {
                "SIGNAL",
                "E0",
                "E1",
                "E2",
                "H30",
                "H60",
                "H120",
                "PRE_CLOSE_30",
                "PRE_CLOSE_15",
                "PRE_CLOSE_5",
                "PRE_CLOSE_1",
                "FINAL_CONTINUOUS",
            }
            required_options = {
                "SIGNAL",
                "E0",
                "H60",
                "H120",
                "PRE_CLOSE_30",
                "FINAL_CONTINUOUS",
            }
            if (
                e0 is None
                or not official
                or not {"SIGNAL", "E0"}.issubset(observation_names)
                or not {"SIGNAL", "E0"}.issubset(option_names)
                or (
                    e0_has_quote
                    and (
                        not final
                        or not required_observations.issubset(observation_names)
                        or not required_options.issubset(option_names)
                    )
                )
            ):
                sessions.add(signal.session)
        return tuple(sorted(sessions))

    def poll(
        self,
        *,
        session: date,
        now: datetime,
        m1c_context_by_checkpoint: Mapping[int, Mapping[str, M1CContextV0]],
    ) -> OpeningLeaderPollResultV0:
        observed = _aware_utc(now, label="opening-leader poll timestamp")
        market_open, market_close = xnys_session_bounds(session)
        if market_open < self.boundary_utc:
            raise ProspectiveBoundaryErrorV0("historical backfill is forbidden")
        if observed < self.boundary_utc:
            raise ProspectiveBoundaryErrorV0("recorder freeze boundary has not been reached")
        run_id = self._metadata(observed).run_id
        records = tuple(
            record
            for record in self.store.records_for_run(run_id)
            if record.session == session
        )
        index = self._index(records)
        linked_by_original = {
            record.original_stable_id: record
            for record in records
            if record.original_stable_id is not None
        }
        late_by_original = {
            original_id: record
            for original_id, record in linked_by_original.items()
            if record.record_type == "late_observation"
            and isinstance(record.payload.get("quote"), dict)
        }
        created_receipts: list[str] = []
        created_failures: list[str] = []
        created_observations: list[str] = []
        created_options: list[str] = []
        new_signal_recording_allowed = (
            market_open <= observed <= market_close + timedelta(minutes=31)
        )

        for checkpoint in FROZEN_SIGNAL_CHECKPOINTS_V0:
            if not new_signal_recording_allowed:
                continue
            label = f"C{checkpoint}"
            if (checkpoint, "signal_receipt", "SIGNAL") in index or (
                checkpoint,
                "signal_failure",
                "SIGNAL",
            ) in index:
                continue
            signal_timestamp = checkpoint_timestamp_v0(session, checkpoint)
            if observed < signal_timestamp:
                continue
            bars = self.bar_provider(session, checkpoint)
            if any(bar.received_timestamp_utc < self.boundary_utc for bar in bars):
                raise ProspectiveBoundaryErrorV0(
                    "pre-freeze market data observed; a new recorder version is required"
                )
            checkpoint_context = dict(m1c_context_by_checkpoint.get(checkpoint, {}))
            for symbol in CANONICAL_COHORT_V0:
                checkpoint_context.setdefault(
                    symbol,
                    M1CContextV0(
                        probability=None,
                        high_low_state="UNKNOWN_INCOMPLETE",
                        tail_phase="UNKNOWN_INCOMPLETE",
                        qualified_fresh_event_status="UNKNOWN_INCOMPLETE",
                        movement_consumed=None,
                        source_completeness="unavailable_at_opening_leader_receipt",
                    ),
                )
            ranking = rank_opening_leader_v0(
                session=session,
                checkpoint=checkpoint,
                bars=bars,
                m1c_context_by_symbol=checkpoint_context,
                cohort_hash=self.freeze_identity.cohort_hash,
                evaluated_at_utc=observed,
            )
            deadline = signal_timestamp + timedelta(seconds=420)
            if not ranking.eligible:
                if observed < deadline:
                    continue
                self._append(
                    observed=observed,
                    session=session,
                    checkpoint=checkpoint,
                    selected_symbol=None,
                    record_type="signal_failure",
                    observation_name="SIGNAL",
                    payload={"ranking": ranking.model_dump(mode="json")},
                    flags=ranking.failure_reasons,
                )
                created_failures.append(label)
                continue
            if observed > deadline:
                self._append(
                    observed=observed,
                    session=session,
                    checkpoint=checkpoint,
                    selected_symbol=None,
                    record_type="signal_failure",
                    observation_name="SIGNAL",
                    payload={"ranking": ranking.model_dump(mode="json")},
                    flags=("signal_capture_deadline_missed",),
                )
                created_failures.append(label)
                continue
            assert ranking.rank_1 is not None
            quote = self.underlying_quote_provider(
                ranking.rank_1.symbol,
                checkpoint,
                "SIGNAL",
                signal_timestamp,
                observed,
            )
            self._assert_quote_after_boundary(quote)
            if quote is None or not quote.valid_for_signal:
                if observed < deadline:
                    continue
                flags = (
                    ("signal_quote_unavailable",)
                    if quote is None
                    else quote.data_quality_flags
                )
                self._append(
                    observed=observed,
                    session=session,
                    checkpoint=checkpoint,
                    selected_symbol=ranking.rank_1.symbol,
                    record_type="signal_failure",
                    observation_name="SIGNAL",
                    payload={
                        "ranking": ranking.model_dump(mode="json"),
                        "quote": None if quote is None else quote.model_dump(mode="json"),
                    },
                    flags=flags,
                )
                created_failures.append(label)
                continue
            self._append(
                observed=observed,
                session=session,
                checkpoint=checkpoint,
                selected_symbol=ranking.rank_1.symbol,
                record_type="signal_receipt",
                observation_name="SIGNAL",
                payload={
                    "valid": True,
                    "selected_identity": "rank_1",
                    "direction": "LONG",
                    "ranking": ranking.model_dump(mode="json"),
                    "signal_quote": quote.model_dump(mode="json"),
                    "causal_signal_available_at_utc": observed.isoformat(),
                    "maximum_rank_input_available_at_utc": max(
                        item.causal_input.available_at_utc for item in ranking.ranking
                    ).isoformat(),
                    "m1c_context_role": "context_only",
                    "order_routing_enabled": False,
                },
                source_timestamps=(
                    quote.actual_quote_timestamp_utc,
                    *(
                        timestamp
                        for item in ranking.ranking
                        for timestamp in (
                            item.causal_input.available_at_utc,
                            item.causal_input.source_timestamp_utc,
                            item.causal_input.received_timestamp_utc,
                        )
                    ),
                ),
            )
            self._record_quote_observation(
                observed=observed,
                session=session,
                checkpoint=checkpoint,
                symbol=ranking.rank_1.symbol,
                observation_name="SIGNAL",
                quote=quote,
                signal_quote=quote,
                entry_quote=None,
            )
            self._capture_option_snapshot(
                observed=observed,
                session=session,
                checkpoint=checkpoint,
                symbol=ranking.rank_1.symbol,
                observation_name="SIGNAL",
                quote=quote,
            )
            created_receipts.append(label)
            created_observations.append(f"{label}:SIGNAL")
            created_options.append(f"{label}:SIGNAL")

        for record in records:
            if record.record_type != "signal_receipt":
                continue
            checkpoint = record.checkpoint
            label = f"C{checkpoint}"
            selected_symbol = record.selected_symbol
            raw_signal_quote = record.payload.get("signal_quote")
            if selected_symbol is None or not isinstance(raw_signal_quote, dict):
                continue
            signal_quote = UnderlyingQuoteV0.model_validate(raw_signal_quote)
            signal_key = (checkpoint, "underlying_observation", "SIGNAL")
            signal_observation = index.get(signal_key)
            if signal_observation is None:
                signal_observation = self._record_quote_observation(
                    observed=observed,
                    session=session,
                    checkpoint=checkpoint,
                    symbol=selected_symbol,
                    observation_name="SIGNAL",
                    quote=signal_quote,
                    signal_quote=signal_quote,
                    entry_quote=None,
                )
                index[signal_key] = signal_observation
                created_observations.append(f"{label}:SIGNAL")
            signal_option_key = (checkpoint, "option_snapshot", "SIGNAL")
            if signal_option_key not in index:
                if observed - record.observed_at_utc <= timedelta(seconds=90):
                    self._capture_option_snapshot(
                        observed=observed,
                        session=session,
                        checkpoint=checkpoint,
                        symbol=selected_symbol,
                        observation_name="SIGNAL",
                        quote=signal_quote,
                    )
                else:
                    self._record_unavailable_option_snapshot(
                        observed=observed,
                        session=session,
                        checkpoint=checkpoint,
                        symbol=selected_symbol,
                        observation_name="SIGNAL",
                        reason="restart_missed_option_snapshot_window",
                    )
                created_options.append(f"{label}:SIGNAL")

            raw_causal_available = record.payload.get(
                "causal_signal_available_at_utc",
                record.observed_at_utc,
            )
            causal_signal_available = _aware_utc(
                raw_causal_available
                if isinstance(raw_causal_available, datetime)
                else datetime.fromisoformat(str(raw_causal_available)),
                label="causal signal availability",
            )
            e0_key = (checkpoint, "underlying_observation", "E0")
            e0_record = index.get(e0_key)
            entry_quote: UnderlyingQuoteV0 | None = None
            if e0_record is not None:
                effective_e0 = late_by_original.get(e0_record.stable_id, e0_record)
                try:
                    entry_quote = self._quote_from_record(effective_e0)
                except ValueError:
                    entry_quote = None
            if entry_quote is None and observed > causal_signal_available:
                e0_quote = self.underlying_quote_provider(
                    selected_symbol,
                    checkpoint,
                    "E0",
                    causal_signal_available,
                    observed,
                )
                self._assert_quote_after_boundary(e0_quote)
                e0_valid = (
                    e0_quote is not None
                    and e0_quote.valid_for_signal
                    and causal_signal_available
                    < e0_quote.actual_quote_timestamp_utc
                    <= causal_signal_available + timedelta(seconds=90)
                    and market_open
                    <= e0_quote.actual_quote_timestamp_utc
                    < market_close
                )
                if e0_valid and e0_quote is not None:
                    if e0_record is None:
                        e0_record = self._record_quote_observation(
                            observed=observed,
                            session=session,
                            checkpoint=checkpoint,
                            symbol=selected_symbol,
                            observation_name="E0",
                            quote=e0_quote,
                            signal_quote=signal_quote,
                            entry_quote=e0_quote,
                        )
                        index[e0_key] = e0_record
                    else:
                        self._append_late_quote_observation(
                            original=e0_record,
                            observed=observed,
                            session=session,
                            checkpoint=checkpoint,
                            symbol=selected_symbol,
                            observation_name="E0",
                            quote=e0_quote,
                            signal_quote=signal_quote,
                            entry_quote=e0_quote,
                        )
                    entry_quote = e0_quote
                    created_observations.append(f"{label}:E0")
                elif (
                    e0_record is None
                    and observed > causal_signal_available + timedelta(seconds=90)
                ):
                    e0_record = self._append(
                        observed=observed,
                        session=session,
                        checkpoint=checkpoint,
                        selected_symbol=selected_symbol,
                        record_type="underlying_observation",
                        observation_name="E0",
                        payload={
                            "status": "UNAVAILABLE",
                            "target_timestamp_utc": causal_signal_available.isoformat(),
                            "reason": "next_causal_quote_window_missed",
                        },
                        flags=("next_causal_quote_window_missed",),
                    )
                    index[e0_key] = e0_record
                    created_observations.append(f"{label}:E0")
            e0_option_key = (checkpoint, "option_snapshot", "E0")
            if e0_record is not None and e0_option_key not in index:
                if (
                    entry_quote is not None
                    and observed - entry_quote.actual_quote_timestamp_utc
                    <= timedelta(seconds=90)
                ):
                    self._capture_option_snapshot(
                        observed=observed,
                        session=session,
                        checkpoint=checkpoint,
                        symbol=selected_symbol,
                        observation_name="E0",
                        quote=entry_quote,
                    )
                else:
                    self._record_unavailable_option_snapshot(
                        observed=observed,
                        session=session,
                        checkpoint=checkpoint,
                        symbol=selected_symbol,
                        observation_name="E0",
                        reason="entry_quote_or_option_snapshot_window_unavailable",
                    )
                created_options.append(f"{label}:E0")
            if entry_quote is None:
                if self._reconcile_official_close(
                    observed=observed,
                    market_close=market_close,
                    session=session,
                    checkpoint=checkpoint,
                    symbol=selected_symbol,
                    entry_quote=None,
                    index=index,
                    linked_by_original=linked_by_original,
                ):
                    created_observations.append(f"{label}:OFFICIAL_CLOSE")
                continue

            schedule = build_observation_schedule_v0(
                session=session,
                signal_timestamp_utc=record.signal_timestamp_utc,
                e0_timestamp_utc=entry_quote.actual_quote_timestamp_utc,
            )
            for target in schedule.underlying_targets:
                if target.name in {"SIGNAL", "E0", "OFFICIAL_CLOSE"}:
                    continue
                target_key = (checkpoint, "underlying_observation", target.name)
                original_observation = index.get(target_key)
                effective_quote: UnderlyingQuoteV0 | None = None
                if original_observation is not None:
                    effective = late_by_original.get(
                        original_observation.stable_id,
                        original_observation,
                    )
                    try:
                        effective_quote = self._quote_from_record(effective)
                    except ValueError:
                        effective_quote = None
                if target.name == "FINAL_CONTINUOUS" and observed < market_close:
                    final_option_key = (checkpoint, "option_snapshot", target.name)
                    if (
                        observed >= target.target_timestamp_utc
                        and final_option_key not in index
                    ):
                        option_reference_quote = self.underlying_quote_provider(
                            selected_symbol,
                            checkpoint,
                            target.name,
                            target.target_timestamp_utc,
                            observed,
                        )
                        self._assert_quote_after_boundary(option_reference_quote)
                        if (
                            option_reference_quote is not None
                            and option_reference_quote.valid_for_signal
                            and market_open
                            <= option_reference_quote.actual_quote_timestamp_utc
                            < market_close
                        ):
                            self._capture_option_snapshot(
                                observed=observed,
                                session=session,
                                checkpoint=checkpoint,
                                symbol=selected_symbol,
                                observation_name=target.name,
                                quote=option_reference_quote,
                            )
                            created_options.append(f"{label}:{target.name}")
                    continue
                if observed < target.target_timestamp_utc:
                    continue
                if effective_quote is None:
                    quote = self.underlying_quote_provider(
                        selected_symbol,
                        checkpoint,
                        target.name,
                        target.target_timestamp_utc,
                        observed,
                    )
                    self._assert_quote_after_boundary(quote)
                    quote_lag = (
                        None
                        if quote is None
                        else (
                            quote.actual_quote_timestamp_utc
                            - target.target_timestamp_utc
                        ).total_seconds()
                    )
                    quote_within_window = (
                        quote is not None
                        and quote.valid_for_signal
                        and market_open <= quote.actual_quote_timestamp_utc < market_close
                        and (
                            target.name == "FINAL_CONTINUOUS"
                            or (
                                quote_lag is not None
                                and 0.0 <= quote_lag <= 90.0
                            )
                        )
                    )
                    if quote_within_window and quote is not None:
                        if original_observation is None:
                            original_observation = self._record_quote_observation(
                                observed=observed,
                                session=session,
                                checkpoint=checkpoint,
                                symbol=selected_symbol,
                                observation_name=target.name,
                                quote=quote,
                                signal_quote=signal_quote,
                                entry_quote=entry_quote,
                            )
                            index[target_key] = original_observation
                        else:
                            self._append_late_quote_observation(
                                original=original_observation,
                                observed=observed,
                                session=session,
                                checkpoint=checkpoint,
                                symbol=selected_symbol,
                                observation_name=target.name,
                                quote=quote,
                                signal_quote=signal_quote,
                                entry_quote=entry_quote,
                            )
                        effective_quote = quote
                        created_observations.append(f"{label}:{target.name}")
                    else:
                        lag = (observed - target.target_timestamp_utc).total_seconds()
                        terminal = target.name == "FINAL_CONTINUOUS" or lag > 90.0
                        if original_observation is None and terminal:
                            original_observation = self._append(
                                observed=observed,
                                session=session,
                                checkpoint=checkpoint,
                                selected_symbol=selected_symbol,
                                record_type="underlying_observation",
                                observation_name=target.name,
                                payload={
                                    "status": "UNAVAILABLE",
                                    "target": target.model_dump(mode="json"),
                                    "reason": "scheduled_quote_window_missed",
                                },
                                flags=("scheduled_quote_window_missed",),
                            )
                            index[target_key] = original_observation
                            created_observations.append(f"{label}:{target.name}")
                option_key = (checkpoint, "option_snapshot", target.name)
                if (
                    target.name in schedule.option_snapshot_names
                    and original_observation is not None
                    and option_key not in index
                ):
                    if (
                        target.name != "FINAL_CONTINUOUS"
                        and effective_quote is not None
                        and observed - effective_quote.actual_quote_timestamp_utc
                        <= timedelta(seconds=90)
                    ):
                        self._capture_option_snapshot(
                            observed=observed,
                            session=session,
                            checkpoint=checkpoint,
                            symbol=selected_symbol,
                            observation_name=target.name,
                            quote=effective_quote,
                        )
                    else:
                        self._record_unavailable_option_snapshot(
                            observed=observed,
                            session=session,
                            checkpoint=checkpoint,
                            symbol=selected_symbol,
                            observation_name=target.name,
                            reason=(
                                "final_continuous_option_snapshot_window_missed"
                                if target.name == "FINAL_CONTINUOUS"
                                else "underlying_or_option_snapshot_window_unavailable"
                            ),
                        )
                    created_options.append(f"{label}:{target.name}")

            if self._reconcile_official_close(
                observed=observed,
                market_close=market_close,
                session=session,
                checkpoint=checkpoint,
                symbol=selected_symbol,
                entry_quote=entry_quote,
                index=index,
                linked_by_original=linked_by_original,
            ):
                created_observations.append(f"{label}:OFFICIAL_CLOSE")
        return OpeningLeaderPollResultV0(
            created_signal_receipts=tuple(created_receipts),
            created_failures=tuple(created_failures),
            created_observations=tuple(created_observations),
            created_option_snapshots=tuple(created_options),
        )


def calculate_rank_persistence_v0(
    *,
    original_leader: str,
    signal_price: float,
    current_price: float,
    current_return_bps_by_symbol: Mapping[str, float],
    observed_path_prices: tuple[float, ...],
) -> RankPersistenceV0:
    if original_leader not in current_return_bps_by_symbol:
        raise ValueError("original leader is missing from the current causal slate")
    prices = (signal_price, current_price, *observed_path_prices)
    if any(not math.isfinite(value) or value <= 0.0 for value in prices):
        raise ValueError("rank-persistence prices must be finite and positive")
    if any(not math.isfinite(value) for value in current_return_bps_by_symbol.values()):
        raise ValueError("current rank returns must be finite")
    ordered = sorted(
        current_return_bps_by_symbol.items(),
        key=lambda item: (-item[1], item[0]),
    )
    rank = next(index for index, item in enumerate(ordered, start=1) if item[0] == original_leader)
    sorted_returns = sorted(current_return_bps_by_symbol.values())
    count = len(sorted_returns)
    median = (
        sorted_returns[count // 2]
        if count % 2
        else (sorted_returns[count // 2 - 1] + sorted_returns[count // 2]) / 2.0
    )
    path_returns = tuple((price / signal_price - 1.0) * 10_000.0 for price in observed_path_prices)
    current = (current_price / signal_price - 1.0) * 10_000.0
    return RankPersistenceV0(
        original_leader=original_leader,
        current_rank=rank,
        remains_rank_1=rank == 1,
        remains_top_2=rank <= 2,
        remains_above_cohort_median=(
            current_return_bps_by_symbol[original_leader] > median
        ),
        return_since_signal_bps=current,
        drawdown_from_signal_bps=min(0.0, current),
        maximum_favourable_excursion_bps=max((0.0, *path_returns)),
        maximum_adverse_excursion_bps=min((0.0, *path_returns)),
    )


def stable_evidence_id_v0(
    *,
    deployment_receipt_id: str,
    session: date,
    checkpoint: int,
    record_type: str,
    observation_name: str,
    revision_discriminator: str = "original",
) -> str:
    if checkpoint not in FROZEN_SIGNAL_CHECKPOINTS_V0:
        raise ValueError("opening-leader evidence permits only C6 and C12")
    identity = "|".join(
        (
            RECORDER_VERSION_V0,
            deployment_receipt_id,
            session.isoformat(),
            f"C{checkpoint}",
            record_type,
            observation_name,
            revision_discriminator,
        )
    )
    return f"olc-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"


class OpeningLeaderEvidenceStoreV0:
    """Append-only repository extension; it exposes no update or delete API."""

    def __init__(
        self,
        repository: ProspectiveRepository,
        *,
        deployment_receipt_id: str,
        contract_hash: str,
        code_hash: str,
        cohort_hash: str,
    ) -> None:
        self.repository = repository
        self.deployment_receipt_id = deployment_receipt_id
        self.contract_hash = contract_hash
        self.code_hash = code_hash
        self.cohort_hash = cohort_hash

    @staticmethod
    def _from_row(row: sqlite3.Row) -> OpeningLeaderEvidenceRecordV0:
        return OpeningLeaderEvidenceRecordV0(
            stable_id=str(row["stable_id"]),
            run_id=str(row["run_id"]),
            recorder_version=cast(
                Literal["opening-leader-continuation-recorder-v0"],
                str(row["recorder_version"]),
            ),
            deployment_receipt_id=str(row["deployment_receipt_id"]),
            session=date.fromisoformat(str(row["session_date"])),
            checkpoint=cast(Literal[6, 12], int(row["checkpoint"])),
            signal_timestamp_utc=datetime.fromisoformat(str(row["signal_timestamp_utc"])),
            selected_symbol=(
                None if row["selected_symbol"] is None else str(row["selected_symbol"])
            ),
            record_type=str(row["record_type"]),
            observation_name=str(row["observation_name"]),
            observed_at_utc=datetime.fromisoformat(str(row["observed_at_utc"])),
            original_stable_id=(
                None if row["original_stable_id"] is None else str(row["original_stable_id"])
            ),
            cohort_hash=str(row["cohort_hash"]),
            contract_hash=str(row["contract_hash"]),
            code_hash=str(row["code_hash"]),
            data_quality_flags=tuple(json.loads(str(row["data_quality_flags_json"]))),
            payload=dict(json.loads(str(row["payload_json"]))),
            content_hash=str(row["content_hash"]),
        )

    def append(
        self,
        metadata: EvidenceMetadata,
        *,
        stable_id: str,
        session: date,
        checkpoint: int,
        signal_timestamp_utc: datetime,
        selected_symbol: str | None,
        record_type: str,
        observation_name: str,
        payload: Mapping[str, object],
        data_quality_flags: tuple[str, ...],
        original_stable_id: str | None = None,
    ) -> OpeningLeaderEvidenceRecordV0:
        if checkpoint not in FROZEN_SIGNAL_CHECKPOINTS_V0:
            raise ValueError("opening-leader evidence permits only C6 and C12")
        if metadata.recorded_at_utc < metadata.prospective_start_utc:
            raise ValueError("historical backfill is forbidden")
        signal = _aware_utc(signal_timestamp_utc, label="evidence signal timestamp")
        if signal != checkpoint_timestamp_v0(session, checkpoint):
            raise ValueError("evidence signal timestamp differs from canonical checkpoint")
        payload_dict = dict(payload)
        content = {
            "stable_id": stable_id,
            "run_id": metadata.run_id,
            "recorder_version": RECORDER_VERSION_V0,
            "deployment_receipt_id": self.deployment_receipt_id,
            "session": session.isoformat(),
            "checkpoint": checkpoint,
            "signal_timestamp_utc": signal.isoformat(),
            "selected_symbol": selected_symbol,
            "record_type": record_type,
            "observation_name": observation_name,
            "observed_at_utc": metadata.recorded_at_utc.astimezone(UTC).isoformat(),
            "original_stable_id": original_stable_id,
            "cohort_hash": self.cohort_hash,
            "contract_hash": self.contract_hash,
            "code_hash": self.code_hash,
            "data_quality_flags": data_quality_flags,
            "payload": payload_dict,
        }
        content_hash = _content_hash(content)
        with self.repository._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM opening_leader_evidence_v0 WHERE stable_id = ?",
                (stable_id,),
            ).fetchone()
            if existing is not None:
                recorded = self._from_row(existing)
                if recorded.content_hash != content_hash:
                    raise ValueError("immutable opening-leader evidence differs")
                return recorded
            if original_stable_id is not None:
                original = connection.execute(
                    "SELECT 1 FROM opening_leader_evidence_v0 WHERE stable_id = ?",
                    (original_stable_id,),
                ).fetchone()
                if original is None:
                    raise ValueError("late observation original evidence is absent")
            envelope_id = self.repository._insert_envelope(connection, metadata)
            connection.execute(
                """
                INSERT INTO opening_leader_evidence_v0(
                    envelope_id, run_id, stable_id, recorder_version,
                    deployment_receipt_id, session_date, checkpoint,
                    signal_timestamp_utc, selected_symbol, record_type,
                    observation_name, observed_at_utc, original_stable_id,
                    cohort_hash, contract_hash, code_hash,
                    data_quality_flags_json, payload_json, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    metadata.run_id,
                    stable_id,
                    RECORDER_VERSION_V0,
                    self.deployment_receipt_id,
                    session.isoformat(),
                    checkpoint,
                    signal.isoformat(),
                    selected_symbol,
                    record_type,
                    observation_name,
                    metadata.recorded_at_utc.astimezone(UTC).isoformat(),
                    original_stable_id,
                    self.cohort_hash,
                    self.contract_hash,
                    self.code_hash,
                    _canonical_json(data_quality_flags),
                    _canonical_json(payload_dict),
                    content_hash,
                ),
            )
            row = connection.execute(
                "SELECT * FROM opening_leader_evidence_v0 WHERE stable_id = ?",
                (stable_id,),
            ).fetchone()
        assert row is not None
        return self._from_row(row)

    def append_late_or_correction(
        self,
        metadata: EvidenceMetadata,
        *,
        original_stable_id: str,
        correction_kind: Literal["late_observation", "correction"],
        payload: Mapping[str, object],
        data_quality_flags: tuple[str, ...],
    ) -> OpeningLeaderEvidenceRecordV0:
        with self.repository._connect() as connection:
            row = connection.execute(
                "SELECT * FROM opening_leader_evidence_v0 WHERE stable_id = ?",
                (original_stable_id,),
            ).fetchone()
        if row is None:
            raise ValueError("late observation original evidence is absent")
        original = self._from_row(row)
        revision = _content_hash(
            {
                "kind": correction_kind,
                "payload": dict(payload),
                "flags": data_quality_flags,
            }
        )
        stable_id = stable_evidence_id_v0(
            deployment_receipt_id=self.deployment_receipt_id,
            session=original.session,
            checkpoint=original.checkpoint,
            record_type=correction_kind,
            observation_name=original.observation_name,
            revision_discriminator=revision,
        )
        return self.append(
            metadata,
            stable_id=stable_id,
            session=original.session,
            checkpoint=original.checkpoint,
            signal_timestamp_utc=original.signal_timestamp_utc,
            selected_symbol=original.selected_symbol,
            record_type=correction_kind,
            observation_name=original.observation_name,
            payload=payload,
            data_quality_flags=data_quality_flags,
            original_stable_id=original_stable_id,
        )

    def records_for_run(self, run_id: str) -> tuple[OpeningLeaderEvidenceRecordV0, ...]:
        with self.repository._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM opening_leader_evidence_v0 WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def recorded_identities(self, run_id: str) -> set[str]:
        return {record.stable_id for record in self.records_for_run(run_id)}


def select_option_chain_requests_v0(
    *,
    session: date,
    underlying: str,
    underlying_con_id: int,
    spot: float,
    available_expiries: tuple[date, ...],
    available_strikes: tuple[float, ...],
    exchange: str,
    trading_class: str,
) -> OptionChainSelectionV0:
    """Freeze at most 20 requests across two expiries and five band anchors."""

    if not math.isfinite(spot) or spot <= 0.0:
        raise ValueError("option selection spot must be finite and positive")
    expiries = tuple(sorted({expiry for expiry in available_expiries if expiry >= session})[:2])
    usable_strikes = tuple(
        sorted(
            {
                strike
                for strike in available_strikes
                if math.isfinite(strike)
                and strike > 0.0
                and abs(strike / spot - 1.0) <= 0.15 + 1e-12
            }
        )
    )
    target_fractions = (0.85, 0.925, 1.0, 1.075, 1.15)
    selected: set[float] = set()
    for target_fraction in target_fractions:
        if not usable_strikes:
            break
        selected.add(
            min(
                usable_strikes,
                key=lambda strike: (abs(strike - target_fraction * spot), strike),
            )
        )
    if len(selected) < 5:
        selected.update(
            sorted(
                (strike for strike in usable_strikes if strike not in selected),
                key=lambda strike: (abs(strike - spot), strike),
            )[: 5 - len(selected)]
        )
    selected_strikes = tuple(sorted(selected))
    if len(expiries) < 2:
        return OptionChainSelectionV0(
            status="UNAVAILABLE",
            reason="fewer_than_two_usable_expiries",
            underlying=underlying,
            spot=spot,
            selected_expiries=expiries,
            selected_strikes_by_expiry={
                expiry.isoformat(): selected_strikes for expiry in expiries
            },
            requests=(),
        )
    if not selected_strikes:
        return OptionChainSelectionV0(
            status="UNAVAILABLE",
            reason="no_strikes_within_frozen_spot_band",
            underlying=underlying,
            spot=spot,
            selected_expiries=expiries,
            selected_strikes_by_expiry={
                expiry.isoformat(): selected_strikes for expiry in expiries
            },
            requests=(),
        )
    rights: tuple[Literal["C", "P"], ...] = ("C", "P")
    requests = tuple(
        OptionContractRequestV0(
            underlying=underlying,
            underlying_con_id=underlying_con_id,
            expiry=expiry,
            strike=strike,
            right=right,
            exchange=exchange,
            trading_class=trading_class,
        )
        for expiry in expiries
        for strike in selected_strikes
        for right in rights
    )
    return OptionChainSelectionV0(
        status="AVAILABLE",
        reason=None,
        underlying=underlying,
        spot=spot,
        selected_expiries=expiries,
        selected_strikes_by_expiry={
            expiry.isoformat(): selected_strikes for expiry in expiries
        },
        requests=requests,
    )


def _target_put(
    quotes: tuple[OptionQuoteV0, ...],
    *,
    target: float,
) -> OptionQuoteV0 | None:
    candidates = tuple(
        quote
        for quote in quotes
        if quote.right == "P" and quote.available and quote.delta is not None
    )
    return min(
        candidates,
        key=lambda quote: (
            abs(abs(cast(float, quote.delta)) - target),
            quote.expiry,
            quote.strike,
            quote.con_id,
        ),
        default=None,
    )


def select_option_diagnostics_v0(
    quotes: tuple[OptionQuoteV0, ...],
) -> dict[str, OptionDiagnosticV0]:
    """Select frozen shadow examples; unavailable inputs never trigger substitution."""

    output: dict[str, OptionDiagnosticV0] = {}
    selected_by_name: dict[str, OptionQuoteV0 | None] = {
        "P20": _target_put(quotes, target=0.20),
        "P30": _target_put(quotes, target=0.30),
    }
    for name, target in (("P20", 0.20), ("P30", 0.30)):
        selected = selected_by_name[name]
        output[name] = OptionDiagnosticV0(
            name=cast(Any, name),
            status="UNAVAILABLE" if selected is None else "AVAILABLE",
            reason="target_delta_put_unavailable" if selected is None else None,
            short_con_id=None if selected is None else selected.con_id,
            long_con_id=None,
            short_target_delta=target,
            entry_short_mark=None if selected is None else selected.bid,
            entry_long_mark=None,
            entry_credit=None if selected is None else selected.bid,
        )
    short = selected_by_name["P20"]
    lower: OptionQuoteV0 | None = None
    reason: str | None = None
    if short is None:
        reason = "target_delta_put_unavailable"
    else:
        same_expiry = sorted(
            {
                quote.strike
                for quote in quotes
                if quote.right == "P"
                and quote.expiry == short.expiry
                and quote.strike < short.strike
            },
            reverse=True,
        )
        if same_expiry:
            required_strike = same_expiry[0]
            lower = next(
                (
                    quote
                    for quote in quotes
                    if quote.right == "P"
                    and quote.expiry == short.expiry
                    and quote.strike == required_strike
                    and quote.available
                ),
                None,
            )
        if lower is None:
            reason = "next_lower_strike_quote_unavailable"
    credit = (
        None
        if short is None or lower is None or short.bid is None or lower.ask is None
        else short.bid - lower.ask
    )
    output["BPS20"] = OptionDiagnosticV0(
        name="BPS20",
        status="AVAILABLE" if short is not None and lower is not None else "UNAVAILABLE",
        reason=reason,
        short_con_id=None if short is None else short.con_id,
        long_con_id=None if lower is None else lower.con_id,
        short_target_delta=0.20,
        entry_short_mark=None if short is None else short.bid,
        entry_long_mark=None if lower is None else lower.ask,
        entry_credit=credit,
    )
    return output


def _finite_optional(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_underlying_quote_v0(
    *,
    quote_id: str,
    symbol: str,
    target_timestamp_utc: datetime,
    captured_at_utc: datetime,
    provider_timestamp_utc: datetime | None,
    values: dict[str, object],
    source: str,
    maximum_quote_age_seconds: float,
) -> UnderlyingQuoteV0:
    """Normalize a bounded IBKR snapshot without imputing a quote field."""

    if maximum_quote_age_seconds <= 0.0:
        raise ValueError("maximum quote age must be positive")
    target = _aware_utc(target_timestamp_utc, label="quote target")
    captured = _aware_utc(captured_at_utc, label="quote capture")
    provider = (
        None
        if provider_timestamp_utc is None
        else _aware_utc(provider_timestamp_utc, label="quote provider timestamp")
    )
    bid = _finite_optional(values.get("bid"))
    ask = _finite_optional(values.get("ask"))
    last = _finite_optional(values.get("last"))
    bid_size = _finite_optional(values.get("bid_size"))
    ask_size = _finite_optional(values.get("ask_size"))
    status_value = str(values.get("market_data_type", "unknown")).lower()
    allowed_statuses = {"live", "frozen", "delayed", "delayed_frozen"}
    market_data_status = status_value if status_value in allowed_statuses else "unknown"
    flags: list[str] = []
    if bid is None or bid <= 0.0:
        flags.append("bid_missing_or_nonpositive")
    if ask is None or ask <= 0.0:
        flags.append("ask_missing_or_nonpositive")
    if bid is not None and ask is not None and bid > ask:
        flags.append("quote_crossed")
    if last is not None and last <= 0.0:
        flags.append("last_nonpositive")
        last = None
    if bid_size is not None and bid_size < 0.0:
        flags.append("bid_size_negative")
        bid_size = None
    if ask_size is not None and ask_size < 0.0:
        flags.append("ask_size_negative")
        ask_size = None
    if bid is None or ask is None or bid <= 0.0 or ask <= 0.0 or bid > ask:
        midpoint = None
        spread = None
    else:
        midpoint = (bid + ask) / 2.0
        spread = ask - bid
    spread_bps = None if midpoint is None or spread is None else 10_000.0 * spread / midpoint
    quote_timestamp = provider or captured
    timestamp_provenance: Literal["provider", "receive"] = (
        "provider" if provider is not None else "receive"
    )
    quote_age = max(0.0, (captured - quote_timestamp).total_seconds())
    if provider is None:
        flags.append("provider_timestamp_unavailable_receive_fallback")
    if quote_timestamp > captured:
        flags.append("source_timestamp_in_future")
    elif quote_age > maximum_quote_age_seconds:
        flags.append("quote_stale")
    if market_data_status != "live":
        flags.append("market_data_not_live")
    halted_value = values.get("halted")
    halted = halted_value if isinstance(halted_value, bool) else None
    if halted is True:
        flags.append("underlying_halted")
    blocking = {
        "bid_missing_or_nonpositive",
        "ask_missing_or_nonpositive",
        "quote_crossed",
        "source_timestamp_in_future",
        "quote_stale",
        "market_data_not_live",
        "underlying_halted",
    }
    return UnderlyingQuoteV0(
        quote_id=quote_id,
        symbol=symbol,
        target_timestamp_utc=target,
        actual_quote_timestamp_utc=quote_timestamp,
        provider_timestamp_utc=provider,
        received_timestamp_utc=captured,
        timestamp_provenance=timestamp_provenance,
        last=last,
        bid=bid,
        ask=ask,
        midpoint=midpoint,
        bid_size=bid_size,
        ask_size=ask_size,
        spread_dollars=spread,
        spread_bps=spread_bps,
        source=source,
        market_data_status=cast(Any, market_data_status),
        quote_age_seconds=quote_age,
        halted=halted,
        data_quality_flags=tuple(flags),
        valid_for_signal=not blocking.intersection(flags),
    )


def _return_bps(entry: float | None, exit_value: float | None) -> float | None:
    if (
        entry is None
        or exit_value is None
        or not math.isfinite(entry)
        or not math.isfinite(exit_value)
        or entry <= 0.0
        or exit_value <= 0.0
    ):
        return None
    return (exit_value / entry - 1.0) * 10_000.0


def calculate_underlying_shadow_return_v0(
    *,
    entry: UnderlyingQuoteV0,
    exit_quote: UnderlyingQuoteV0,
    configured_fee_bps: float,
    friction_diagnostics_bps: tuple[float, ...],
    official_close_reference: float | None = None,
) -> UnderlyingShadowReturnV0:
    """Calculate fixed long shadow returns from observed sides only."""

    if entry.symbol != exit_quote.symbol:
        raise ValueError("entry and exit symbols differ")
    if not math.isfinite(configured_fee_bps) or configured_fee_bps < 0.0:
        raise ValueError("configured fee bps must be finite and nonnegative")
    if any(not math.isfinite(value) or value < 0.0 for value in friction_diagnostics_bps):
        raise ValueError("friction diagnostics must be finite and nonnegative")
    if official_close_reference is not None and (
        not math.isfinite(official_close_reference) or official_close_reference <= 0.0
    ):
        raise ValueError("official close reference must be finite and positive")
    conservative = _return_bps(entry.ask, exit_quote.bid)
    return UnderlyingShadowReturnV0(
        entry_quote_id=entry.quote_id,
        exit_quote_id=exit_quote.quote_id,
        conservative_ask_to_bid_gross_bps=conservative,
        conservative_ask_to_bid_net_bps=(
            None if conservative is None else conservative - configured_fee_bps
        ),
        midpoint_to_midpoint_diagnostic_bps=_return_bps(entry.midpoint, exit_quote.midpoint),
        last_to_last_diagnostic_bps=_return_bps(entry.last, exit_quote.last),
        configured_fee_bps=configured_fee_bps,
        friction_diagnostics_bps={
            f"{friction:g}": None if conservative is None else conservative - friction
            for friction in friction_diagnostics_bps
        },
        official_close_reference=official_close_reference,
        official_close_reference_bps=_return_bps(entry.ask, official_close_reference),
    )


def build_observation_schedule_v0(
    *,
    session: date,
    signal_timestamp_utc: datetime,
    e0_timestamp_utc: datetime,
) -> ObservationScheduleV0:
    """Freeze all quote targets without introducing an adaptive horizon."""

    signal = _aware_utc(signal_timestamp_utc, label="signal timestamp")
    e0 = _aware_utc(e0_timestamp_utc, label="E0 timestamp")
    if signal not in {
        checkpoint_timestamp_v0(session, checkpoint)
        for checkpoint in FROZEN_SIGNAL_CHECKPOINTS_V0
    }:
        raise ValueError("signal timestamp is not the canonical C6 or C12 close")
    if e0 < signal:
        raise ValueError("E0 cannot precede the causal signal checkpoint")
    _, market_close = xnys_session_bounds(session)
    targets = (
        ObservationTargetV0(
            name="SIGNAL",
            target_timestamp_utc=signal,
            source_kind="level1_quote",
            executable=True,
        ),
        ObservationTargetV0(
            name="E0",
            target_timestamp_utc=e0,
            source_kind="level1_quote",
            executable=True,
            rank_persistence_required=True,
        ),
        ObservationTargetV0(
            name="E1",
            target_timestamp_utc=e0 + timedelta(minutes=5),
            source_kind="level1_quote",
            executable=True,
            rank_persistence_required=True,
        ),
        ObservationTargetV0(
            name="E2",
            target_timestamp_utc=e0 + timedelta(minutes=10),
            source_kind="level1_quote",
            executable=True,
            rank_persistence_required=True,
        ),
        ObservationTargetV0(
            name="H30",
            target_timestamp_utc=e0 + timedelta(minutes=30),
            source_kind="level1_quote",
            executable=True,
            rank_persistence_required=True,
        ),
        ObservationTargetV0(
            name="H60",
            target_timestamp_utc=e0 + timedelta(minutes=60),
            source_kind="level1_quote",
            executable=True,
            rank_persistence_required=True,
        ),
        ObservationTargetV0(
            name="H120",
            target_timestamp_utc=e0 + timedelta(minutes=120),
            source_kind="level1_quote",
            executable=True,
            rank_persistence_required=True,
        ),
        ObservationTargetV0(
            name="PRE_CLOSE_30",
            target_timestamp_utc=market_close - timedelta(minutes=30),
            source_kind="level1_quote",
            executable=True,
        ),
        ObservationTargetV0(
            name="PRE_CLOSE_15",
            target_timestamp_utc=market_close - timedelta(minutes=15),
            source_kind="level1_quote",
            executable=True,
        ),
        ObservationTargetV0(
            name="PRE_CLOSE_5",
            target_timestamp_utc=market_close - timedelta(minutes=5),
            source_kind="level1_quote",
            executable=True,
        ),
        ObservationTargetV0(
            name="PRE_CLOSE_1",
            target_timestamp_utc=market_close - timedelta(minutes=1),
            source_kind="level1_quote",
            executable=True,
        ),
        ObservationTargetV0(
            name="FINAL_CONTINUOUS",
            target_timestamp_utc=market_close - timedelta(seconds=1),
            source_kind="level1_quote",
            executable=True,
        ),
        ObservationTargetV0(
            name="OFFICIAL_CLOSE",
            target_timestamp_utc=market_close,
            source_kind="official_bar_close_reference",
            executable=False,
        ),
    )
    return ObservationScheduleV0(
        session=session,
        signal_timestamp_utc=signal,
        e0_timestamp_utc=e0,
        underlying_targets=targets,
        option_snapshot_names=(
            "SIGNAL",
            "E0",
            "H60",
            "H120",
            "PRE_CLOSE_30",
            "FINAL_CONTINUOUS",
        ),
    )


def _bar_exclusion_reason(
    bar: CausalCheckpointBarV0,
    *,
    session: date,
    checkpoint: int,
    signal_timestamp: datetime,
    evaluated_at: datetime,
) -> str | None:
    if bar.session != session:
        return "session_mismatch"
    if bar.checkpoint != checkpoint:
        return "checkpoint_mismatch"
    if bar.bar_end_utc != signal_timestamp:
        return "checkpoint_timestamp_mismatch"
    if bar.bar_start_utc != signal_timestamp - timedelta(minutes=5):
        return "bar_start_timestamp_mismatch"
    if bar.available_at_utc > evaluated_at:
        return "future_data_not_causally_available"
    if bar.source_timestamp_utc > evaluated_at or bar.received_timestamp_utc > evaluated_at:
        return "future_source_timestamp"
    if bar.source_completeness != "complete":
        return "source_incomplete"
    if bar.duplicate_resolution == "unresolved":
        return "duplicate_bar_unresolved"
    return None


def rank_opening_leader_v0(
    *,
    session: date,
    checkpoint: int,
    bars: tuple[CausalCheckpointBarV0, ...],
    m1c_context_by_symbol: dict[str, M1CContextV0] | None,
    cohort_hash: str,
    evaluated_at_utc: datetime,
) -> OpeningLeaderRankingV0:
    """Rank only causal open-to-checkpoint returns; M1C is attached afterwards."""

    signal_timestamp = checkpoint_timestamp_v0(session, checkpoint)
    evaluated_at = _aware_utc(evaluated_at_utc, label="ranking evaluation timestamp")
    contexts = {} if m1c_context_by_symbol is None else dict(m1c_context_by_symbol)
    counts = Counter(bar.symbol for bar in bars)
    by_symbol = {bar.symbol: bar for bar in bars if counts[bar.symbol] == 1}
    exclusions: dict[str, str] = {}
    eligible_bars: list[CausalCheckpointBarV0] = []
    for symbol in CANONICAL_COHORT_V0:
        if counts[symbol] == 0:
            exclusions[symbol] = "missing_checkpoint_bar"
            continue
        if counts[symbol] > 1:
            exclusions[symbol] = "duplicate_checkpoint_bar"
            continue
        bar = by_symbol[symbol]
        reason = _bar_exclusion_reason(
            bar,
            session=session,
            checkpoint=checkpoint,
            signal_timestamp=signal_timestamp,
            evaluated_at=evaluated_at,
        )
        if reason is not None:
            exclusions[symbol] = reason
            continue
        eligible_bars.append(bar)

    failure_reasons: list[str] = []
    unexpected = sorted(set(counts).difference(CANONICAL_COHORT_V0))
    if unexpected:
        failure_reasons.append("non_canonical_symbols_present:" + ",".join(unexpected))
    if cohort_hash != CANONICAL_COHORT_HASH_V0:
        failure_reasons.append("canonical_cohort_hash_mismatch")
    if evaluated_at < signal_timestamp:
        failure_reasons.append("checkpoint_time_not_reached")
    if len(eligible_bars) < MINIMUM_COMPLETE_SLATE_V0:
        failure_reasons.append("fewer_than_15_complete_stocks")

    rank_inputs = [
        (
            bar,
            10_000.0 * (bar.checkpoint_close / bar.regular_session_open - 1.0),
        )
        for bar in eligible_bars
    ]
    rank_inputs.sort(key=lambda item: (-item[1], item[0].symbol))
    ranking = tuple(
        RankedStockV0(
            rank=index,
            symbol=bar.symbol,
            open_to_checkpoint_return_bps=return_bps,
            regular_session_open=bar.regular_session_open,
            checkpoint_close=bar.checkpoint_close,
            input_identity=bar.checkpoint_source_id,
            causal_input=bar,
            m1c_context=contexts.get(bar.symbol),
        )
        for index, (bar, return_bps) in enumerate(rank_inputs, start=1)
    )
    exact_slate = tuple(item.symbol for item in ranking)
    slate_hash = _content_hash(
        {
            "session": session.isoformat(),
            "checkpoint": checkpoint,
            "members": [
                {
                    "symbol": item.symbol,
                    "regular_session_open": item.regular_session_open,
                    "checkpoint_close": item.checkpoint_close,
                    "input_identity": item.input_identity,
                    "causal_input": item.causal_input.model_dump(mode="json"),
                }
                for item in ranking
            ],
            "exclusions": exclusions,
        }
    )
    rank_1 = ranking[0] if ranking else None
    rank_2 = ranking[1] if len(ranking) > 1 else None
    eligible = not failure_reasons and rank_1 is not None and rank_2 is not None
    return OpeningLeaderRankingV0(
        recorder_version=RECORDER_VERSION_V0,
        session=session,
        checkpoint=checkpoint,
        checkpoint_role=FROZEN_SIGNAL_CHECKPOINTS_V0[checkpoint],
        signal_timestamp_utc=signal_timestamp,
        evaluated_at_utc=evaluated_at,
        cohort_hash=cohort_hash,
        slate_hash=slate_hash,
        eligible=eligible,
        failure_reasons=tuple(failure_reasons),
        exclusions=exclusions,
        exact_slate_membership=exact_slate,
        slate_size=len(ranking),
        ranking=ranking,
        rank_1=rank_1,
        rank_2=rank_2,
        rank_1_minus_rank_2_bps=(
            None
            if rank_1 is None or rank_2 is None
            else rank_1.open_to_checkpoint_return_bps - rank_2.open_to_checkpoint_return_bps
        ),
    )


__all__ = [
    "CANONICAL_COHORT_HASH_V0",
    "CANONICAL_COHORT_V0",
    "CausalCheckpointBarV0",
    "M1CContextV0",
    "OptionChainSelectionV0",
    "OptionContractRequestV0",
    "OptionDiagnosticV0",
    "OptionQuoteV0",
    "ObservationScheduleV0",
    "ObservationTargetV0",
    "OpeningLeaderContinuationRecorderV0",
    "OpeningLeaderRankingV0",
    "OpeningLeaderEvidenceRecordV0",
    "OpeningLeaderEvidenceStoreV0",
    "OpeningLeaderFreezeIdentityV0",
    "OpeningLeaderPollResultV0",
    "OptionSnapshotCaptureV0",
    "ProspectiveBoundaryErrorV0",
    "UnderlyingQuoteV0",
    "UnderlyingShadowReturnV0",
    "build_observation_schedule_v0",
    "calculate_underlying_shadow_return_v0",
    "calculate_rank_persistence_v0",
    "checkpoint_timestamp_v0",
    "normalize_underlying_quote_v0",
    "rank_opening_leader_v0",
    "RankPersistenceV0",
    "select_option_chain_requests_v0",
    "select_option_diagnostics_v0",
    "stable_evidence_id_v0",
]
