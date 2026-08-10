"""First-party idea-plugin protocol and bounded evaluation contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from stocker_runtime.domain import (
    DomainModel,
    IdeaOutput,
    JsonValue,
    MarketEvent,
    OutputKind,
    ProtectedDataClass,
    RuntimeMode,
    canonical_json_bytes,
)

MAX_PLUGIN_STATE_BYTES = 64 * 1024
MAX_OUTPUTS_PER_BATCH = 256
MAX_EVENTS_PER_BATCH = 256
MAX_INTERESTS_PER_BATCH = 64
MAX_INTEREST_LIFETIME_US = 7 * 86_400_000_000


class IdeaManifest(DomainModel):
    """Immutable identity, capability, and resource limits declared by a plugin."""

    api_version: Literal[1] = 1
    idea_id: str = Field(min_length=1)
    idea_version: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str
    modes: tuple[RuntimeMode, ...] = Field(min_length=1)
    output_kinds: tuple[OutputKind, ...] = Field(min_length=1)
    parameter_schema_version: str = Field(min_length=1)
    parameter_schema: Mapping[str, JsonValue]
    maximum_state_bytes: int = Field(ge=1, le=MAX_PLUGIN_STATE_BYTES)
    maximum_outputs_per_batch: int = Field(ge=1, le=MAX_OUTPUTS_PER_BATCH)
    maximum_interests_per_batch: int = Field(ge=0, le=MAX_INTERESTS_PER_BATCH)

    @model_validator(mode="after")
    def declarations_are_unique(self) -> Self:
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")
        if len(set(self.output_kinds)) != len(self.output_kinds):
            raise ValueError("output_kinds must not contain duplicates")
        return self


class IdeaActivation(DomainModel):
    """Core-owned immutable activation details passed to one plugin instance."""

    instance_id: str = Field(min_length=1)
    parameters: Mapping[str, JsonValue]
    parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plugin_code_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    activated_at_us: int = Field(ge=0)
    run_id: str = Field(min_length=1)
    protected_data_class: ProtectedDataClass
    universe: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def parameters_hash_matches(self) -> Self:
        expected_hash = hashlib.sha256(canonical_json_bytes(self.parameters)).hexdigest()
        if self.parameters_hash != expected_hash:
            raise ValueError("parameters_hash does not match canonical parameters")
        return self


class MarketDataRequirement(DomainModel):
    """Market-data need declared by a plugin and fulfilled by the core."""

    feed_kind: str = Field(min_length=1)
    event_kind: str | None = Field(default=None, min_length=1)
    instrument_id: str = Field(min_length=1)
    cadence: str = Field(min_length=1)
    gaps_block: bool
    staleness_block: bool


class MarketDataInterest(DomainModel):
    """One bounded, causal request for core-owned option discovery and recording."""

    interest_key: str = Field(min_length=1, max_length=128)
    underlying_instrument_id: str = Field(min_length=1, max_length=128)
    asset_kind: Literal["option"] = "option"
    minimum_days_to_expiry: int = Field(ge=0, le=365)
    maximum_days_to_expiry: int = Field(ge=0, le=365)
    option_right: Literal["call", "put"]
    strike_offset: int = Field(ge=-32, le=32)
    reference_price: float = Field(gt=0)
    feed_kind: Literal["quotes"] = "quotes"
    cadence: Literal["snapshot", "stream"]
    as_of_at_us: int = Field(ge=0)
    expires_at_us: int = Field(ge=0)
    required: bool
    priority: int = Field(ge=0, le=1_000)
    maximum_contracts: Literal[1] = 1
    input_event_ids: tuple[str, ...] = Field(min_length=1, max_length=MAX_EVENTS_PER_BATCH)

    @model_validator(mode="after")
    def bounded_causal_window(self) -> Self:
        if self.minimum_days_to_expiry > self.maximum_days_to_expiry:
            raise ValueError("minimum expiry must not exceed maximum expiry")
        if self.expires_at_us <= self.as_of_at_us:
            raise ValueError("interest expiry must follow its causal as-of time")
        if self.expires_at_us - self.as_of_at_us > MAX_INTEREST_LIFETIME_US:
            raise ValueError("interest lifetime must not exceed seven days")
        if len(set(self.input_event_ids)) != len(self.input_event_ids):
            raise ValueError("interest input event ids must be unique")
        return self


class DiscoveryReceipt(DomainModel):
    """Authority-free result of one bounded core-owned instrument discovery."""

    receipt_id: str = Field(min_length=1)
    interest_id: str = Field(min_length=1)
    interest_key: str = Field(min_length=1, max_length=128)
    instance_id: str = Field(min_length=1)
    status: Literal["resolved", "denied"]
    reason_code: str | None = Field(default=None, min_length=1, max_length=128)
    instrument_id: str | None = Field(default=None, min_length=1)
    expiry: str | None = Field(default=None, pattern=r"^[0-9]{8}$")
    strike: float | None = Field(default=None, gt=0)
    option_right: Literal["call", "put"] | None = None
    multiplier: str | None = Field(default=None, min_length=1, max_length=16)
    candidates_inspected: int = Field(ge=0, le=4_096)
    completed_at_us: int = Field(ge=0)

    @model_validator(mode="after")
    def resolved_fields_are_complete(self) -> Self:
        resolved = (
            self.instrument_id,
            self.expiry,
            self.strike,
            self.option_right,
            self.multiplier,
        )
        if self.status == "resolved" and (
            self.reason_code is not None or any(value is None for value in resolved)
        ):
            raise ValueError("resolved discovery receipt requires complete identity")
        if self.status == "denied" and (
            self.reason_code is None or any(value is not None for value in resolved)
        ):
            raise ValueError("denied discovery receipt requires only a reason")
        return self


class IdeaBatch(DomainModel):
    """Immutable causal input of at most 256 market events."""

    mode: RuntimeMode
    events: tuple[MarketEvent, ...] = Field(min_length=1, max_length=MAX_EVENTS_PER_BATCH)
    input_watermark: str = Field(min_length=1)
    causal_from_at_us: int = Field(ge=0)
    causal_through_at_us: int = Field(ge=0)
    prior_state_input_event_ids: tuple[str, ...] = Field(
        default=(), max_length=MAX_EVENTS_PER_BATCH
    )
    discovery_receipts: tuple[DiscoveryReceipt, ...] = Field(
        default=(), max_length=MAX_INTERESTS_PER_BATCH
    )

    @model_validator(mode="after")
    def causal_range_is_ordered(self) -> Self:
        if self.causal_from_at_us > self.causal_through_at_us:
            raise ValueError("causal_from_at_us must not exceed causal_through_at_us")
        return self


type BoundedIdeaOutput = Annotated[IdeaOutput, Field(discriminator="kind")]


class IdeaEvaluation(DomainModel):
    """Bounded state and evidence returned by one plugin evaluation."""

    state: JsonValue
    outputs: tuple[BoundedIdeaOutput, ...] = Field(max_length=MAX_OUTPUTS_PER_BATCH)
    retained_input_event_ids: tuple[str, ...] = Field(default=(), max_length=MAX_EVENTS_PER_BATCH)
    output_input_event_ids: tuple[tuple[str, ...], ...] = Field(
        default=(), max_length=MAX_OUTPUTS_PER_BATCH
    )
    interests: tuple[MarketDataInterest, ...] = Field(max_length=MAX_INTERESTS_PER_BATCH)

    @model_validator(mode="after")
    def state_fits_bound(self) -> Self:
        if len(self.state_json()) > MAX_PLUGIN_STATE_BYTES:
            raise ValueError(f"state exceeds {MAX_PLUGIN_STATE_BYTES} bytes")
        if len(set(self.retained_input_event_ids)) != len(self.retained_input_event_ids):
            raise ValueError("retained input event ids must be unique")
        if any(
            not lineage or len(lineage) > MAX_EVENTS_PER_BATCH or len(set(lineage)) != len(lineage)
            for lineage in self.output_input_event_ids
        ):
            raise ValueError("every output lineage must contain 1..256 unique event ids")
        if len({interest.interest_key for interest in self.interests}) != len(self.interests):
            raise ValueError("interest keys must be unique within one evaluation")
        return self

    def state_json(self) -> bytes:
        """Return canonical state bytes used for persistence and size checks."""

        return canonical_json_bytes(self.state)


@runtime_checkable
class IdeaPlugin(Protocol):
    """Public protocol implemented by reviewed, first-party idea plugins."""

    @property
    def manifest(self) -> IdeaManifest: ...

    def requirements(
        self,
        activation: IdeaActivation,
    ) -> tuple[MarketDataRequirement, ...]: ...

    def evaluate(
        self,
        batch: IdeaBatch,
        state: JsonValue,
    ) -> IdeaEvaluation: ...
