"""Public, authority-free domain objects for the Stocker V2 runtime."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]

MAX_OUTPUT_PAYLOAD_BYTES = 16 * 1024
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
        "order",
        "order_id",
        "order_type",
        "paper_mode",
        "risk",
        "risk_approval",
        "risk_decision",
        "should_transmit",
        "transmit",
    }
)


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
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def ensure_authority_free_json(value: JsonValue) -> None:
    """Reject JSON keys that would smuggle trading authority through extension data."""

    if isinstance(value, dict):
        for key, nested_value in value.items():
            normalized_key = key.lower().replace("-", "_")
            if normalized_key in FORBIDDEN_AUTHORITY_FIELD_NAMES:
                raise ValueError(f"forbidden authority field: {key}")
            ensure_authority_free_json(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            ensure_authority_free_json(nested_value)


class DomainModel(BaseModel):
    """Immutable DTO with strict fields and deterministic JSON serialization."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    def to_canonical_json(self) -> bytes:
        """Serialize the DTO as compact, key-sorted UTF-8 JSON."""

        return canonical_json_bytes(self.model_dump(mode="json"))


class MarketEvent(DomainModel):
    """Immutable normalized market evidence supplied to an idea plugin."""

    event_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    feed_kind: str = Field(min_length=1)
    event_kind: str = Field(min_length=1)
    event_at_us: int = Field(ge=0)
    received_at_us: int = Field(ge=0)
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def payload_has_no_authority(self) -> Self:
        ensure_authority_free_json(self.payload)
        return self


class IdeaOutputBase(DomainModel):
    """Fields common to every bounded plugin output."""

    subject_instrument_id: str = Field(min_length=1)
    as_of_at_us: int = Field(ge=0)
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def payload_fits_bound(self) -> Self:
        ensure_authority_free_json(self.payload)
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


class ProposedPosition(IdeaOutputBase):
    """An unapproved desired-position proposal with no execution authority."""

    kind: Literal["proposed_position"] = "proposed_position"
    status: Literal["unapproved"] = "unapproved"


class ProposedTrade(IdeaOutputBase):
    """An unapproved trade proposal with no execution authority."""

    kind: Literal["proposed_trade"] = "proposed_trade"
    status: Literal["unapproved"] = "unapproved"


type IdeaOutput = Observation | Signal | ProposedPosition | ProposedTrade
