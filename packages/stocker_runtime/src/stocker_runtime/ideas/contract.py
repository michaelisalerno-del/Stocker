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
