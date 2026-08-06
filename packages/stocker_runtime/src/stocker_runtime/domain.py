"""Public, authority-free domain objects for the Stocker V2 runtime."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_validator,
)

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]
)

MAX_OUTPUT_PAYLOAD_BYTES = 16 * 1024
MAX_MARKET_EVENT_PAYLOAD_BYTES = 64 * 1024
FORBIDDEN_AUTHORITY_FIELD_NAMES = frozenset(
    {
        "account",
        "account_id",
        "approval",
        "approval_id",
        "approved",
        "authority",
        "broker_account",
        "broker_account_id",
        "broker_order",
        "broker_order_id",
        "executable",
        "execution",
        "execution_id",
        "is_approved",
        "live_mode",
        "mode",
        "operating_mode",
        "order",
        "order_id",
        "order_type",
        "paper_mode",
        "risk",
        "risk_approval",
        "risk_decision",
        "runtime_mode",
        "should_transmit",
        "transmit",
        "trading_mode",
    }
)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw_json(nested) for nested in value]
    return value


class RuntimeMode(StrEnum):
    """Operating modes authorized for the immediate V2 runtime."""

    PROSPECTIVE_RECORD = "prospective_record"
    SHADOW = "shadow"


class OutputKind(StrEnum):
    """Evidence kinds a first-party idea plugin may emit."""

    OBSERVATION = "observation"
    SIGNAL = "signal"
    PROPOSED_POSITION = "proposed_position"
    PROPOSED_TRADE = "proposed_trade"


class ProtectedDataClass(StrEnum):
    """Protected evidence classes permitted by the immediate V2 runtime."""

    PROSPECTIVE = "prospective_protected"
    SHADOW = "shadow_protected"


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Serialize a JSON value deterministically as compact UTF-8 bytes."""

    return json.dumps(
        _thaw_json(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _normalized_json_key(key: str) -> str:
    acronym_boundaries = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    word_boundaries = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", acronym_boundaries)
    return re.sub(r"[^A-Za-z0-9]+", "_", word_boundaries).strip("_").lower()


def ensure_authority_free_json(value: JsonValue) -> None:
    """Reject JSON keys that would smuggle trading authority through extension data."""

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = _normalized_json_key(key)
            if normalized_key in FORBIDDEN_AUTHORITY_FIELD_NAMES:
                raise ValueError(f"forbidden authority field: {key}")
            ensure_authority_free_json(nested_value)
    elif isinstance(value, list | tuple):
        for nested_value in value:
            ensure_authority_free_json(nested_value)


class DomainModel(BaseModel):
    """Immutable DTO with strict fields and deterministic JSON serialization."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    _JSON_FIELD_NAMES = frozenset({"parameter_schema", "parameters", "payload", "state"})

    @model_validator(mode="after")
    def freeze_nested_json(self) -> Self:
        for field_name in self._JSON_FIELD_NAMES & type(self).model_fields.keys():
            value = getattr(self, field_name)
            frozen_value = _freeze_json(value)
            if frozen_value is not value:
                object.__setattr__(self, field_name, frozen_value)
        return self

    @field_serializer("*", mode="wrap", check_fields=False)
    def serialize_nested_json(
        self,
        value: object,
        handler: SerializerFunctionWrapHandler,
        info: FieldSerializationInfo,
    ) -> object:
        if info.field_name in self._JSON_FIELD_NAMES:
            return _thaw_json(value)
        return handler(value)

    def to_canonical_json(self) -> bytes:
        """Serialize the DTO as compact, key-sorted UTF-8 JSON."""

        return canonical_json_bytes(self.model_dump(mode="json"))


class MarketEvent(DomainModel):
    """Immutable normalized market evidence with a 64 KiB payload ceiling."""

    event_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    feed_kind: str = Field(min_length=1)
    event_kind: str = Field(min_length=1)
    event_at_us: int = Field(ge=0)
    received_at_us: int = Field(ge=0)
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def payload_fits_bound(self) -> Self:
        if len(self.payload_json()) > MAX_MARKET_EVENT_PAYLOAD_BYTES:
            raise ValueError(f"market event payload exceeds {MAX_MARKET_EVENT_PAYLOAD_BYTES} bytes")
        return self

    def payload_json(self) -> bytes:
        """Return canonical payload bytes used for admission-size checks."""

        return canonical_json_bytes(self.payload)


class IdeaOutputBase(DomainModel):
    """Fields common to every bounded plugin output."""

    subject_instrument_id: str = Field(min_length=1)
    as_of_at_us: int = Field(ge=0)
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def payload_fits_bound(self) -> Self:
        if len(self.payload_json()) > MAX_OUTPUT_PAYLOAD_BYTES:
            raise ValueError(f"payload exceeds {MAX_OUTPUT_PAYLOAD_BYTES} bytes")
        return self

    def payload_json(self) -> bytes:
        """Return the canonical payload bytes used for size and identity checks."""

        return canonical_json_bytes(self.payload)


class Observation(IdeaOutputBase):
    """A plugin's descriptive evidence about an instrument at a causal time."""

    kind: Literal["observation"] = "observation"


class Signal(IdeaOutputBase):
    """A plugin's directional or classificatory evidence."""

    kind: Literal["signal"] = "signal"


class ProposedOutputBase(IdeaOutputBase):
    """Common authority guard for every unapproved proposal."""

    status: Literal["unapproved"] = "unapproved"

    @model_validator(mode="after")
    def payload_has_no_authority(self) -> Self:
        ensure_authority_free_json(self.payload)
        return self


class ProposedPosition(ProposedOutputBase):
    """An unapproved desired-position proposal with no execution authority."""

    kind: Literal["proposed_position"] = "proposed_position"


class ProposedTrade(ProposedOutputBase):
    """An unapproved trade proposal with no execution authority."""

    kind: Literal["proposed_trade"] = "proposed_trade"


type IdeaOutput = Observation | Signal | ProposedPosition | ProposedTrade
