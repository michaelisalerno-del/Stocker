"""Public, authority-free domain objects for the Stocker V2 runtime."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldSerializationInfo,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_serializer,
    model_validator,
)

type JsonValue = None | bool | int | float | str | tuple[JsonValue, ...] | Mapping[str, JsonValue]

MAX_OUTPUT_PAYLOAD_BYTES = 16 * 1024
MAX_MARKET_EVENT_PAYLOAD_BYTES = 64 * 1024
_JSON_FIELD_NAMES = frozenset({"parameter_schema", "parameters", "payload", "state"})
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


class FrozenJsonObject(Mapping[str, object]):
    """Immutable JSON object with copied, constant-time mapping storage."""

    __slots__ = ("_data",)
    _data: Mapping[str, object]

    def __init__(self, value: Mapping[str, object]) -> None:
        copied = {key: _freeze_json(nested) for key, nested in value.items()}
        object.__setattr__(self, "_data", MappingProxyType(copied))

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("immutable JSON objects cannot be changed")

    def __deepcopy__(self, _memo: dict[int, object]) -> FrozenJsonObject:
        return self

    def __reduce__(self) -> tuple[type[FrozenJsonObject], tuple[dict[str, object]]]:
        return type(self), (dict(self._data),)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self._data) == dict(other.items())

    def __repr__(self) -> str:
        return repr(dict(self._data))


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return FrozenJsonObject(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _normalize_json_input(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _normalize_json_input(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return tuple(_normalize_json_input(nested) for nested in value)
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
    """Strict immutable DTO; unvalidated ``model_construct`` is outside this contract."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_nested_json_input(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        if info.mode == "json":
            normalized = {
                field_name: tuple(field_value) if isinstance(field_value, list) else field_value
                for field_name, field_value in normalized.items()
            }
        for field_name in _JSON_FIELD_NAMES & normalized.keys():
            normalized[field_name] = _normalize_json_input(normalized[field_name])
        return normalized

    @model_validator(mode="after")
    def freeze_nested_json(self) -> Self:
        for field_name in _JSON_FIELD_NAMES & type(self).model_fields.keys():
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
        if info.field_name in _JSON_FIELD_NAMES:
            return _thaw_json(value)
        return handler(value)

    def to_canonical_json(self) -> bytes:
        """Serialize the DTO as compact, key-sorted UTF-8 JSON."""

        return canonical_json_bytes(self.model_dump(mode="json"))

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy an immutable DTO without allowing validation-bypassing updates."""

        if update:
            raise TypeError("immutable DTO copies cannot be updated")
        return super().model_copy(update=update, deep=deep)

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Apply the same fail-closed policy to Pydantic's deprecated copy API."""

        if include is not None or exclude is not None:
            raise TypeError("immutable DTO copies cannot include or exclude fields")
        return self.model_copy(update=update, deep=deep)


class ShadowFillPolicy(DomainModel):
    """Versioned, conservative virtual-price convention; it is not a broker fill."""

    model_id: str = Field(min_length=1, max_length=128)
    entry_convention: Literal["buy_ask_sell_bid"] = "buy_ask_sell_bid"
    exit_convention: Literal["buy_bid_sell_ask"] = "buy_bid_sell_ask"
    max_quote_age_us: int = Field(default=60_000_000, gt=0)


class ShadowCostPolicy(DomainModel):
    """Versioned virtual cost policy, expressed as a non-negative basis-point charge."""

    model_id: str = Field(min_length=1, max_length=128)
    per_side_bps: float = Field(default=0.0, ge=0.0, le=10_000.0)


class MarketEvent(DomainModel):
    """Immutable normalized market evidence with a 64 KiB payload ceiling."""

    event_id: str = Field(min_length=1)
    instrument_id: str = Field(min_length=1)
    feed_kind: str = Field(min_length=1)
    event_kind: str = Field(min_length=1)
    event_at_us: int = Field(ge=0)
    received_at_us: int = Field(ge=0)
    payload: Mapping[str, JsonValue]

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
    payload: Mapping[str, JsonValue]

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


class ProposedTradeLeg(DomainModel):
    """One typed, authority-free leg of an unapproved trade proposal."""

    instrument_id: str = Field(min_length=1)
    action: Literal["buy", "sell"]
    target: Literal["long", "short", "reduce", "close"]
    quantity_value: float | None = Field(default=None, gt=0)
    notional_value: float | None = Field(default=None, gt=0)
    currency: str | None = Field(default=None, min_length=1)
    price_hint: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def has_sizing_hint(self) -> Self:
        if self.quantity_value is None and self.notional_value is None:
            raise ValueError("trade leg requires quantity_value or notional_value")
        return self


class ProposedTrade(ProposedOutputBase):
    """An unapproved trade proposal with no execution authority."""

    kind: Literal["proposed_trade"] = "proposed_trade"
    legs: tuple[ProposedTradeLeg, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def subjects_one_of_its_legs(self) -> Self:
        if self.subject_instrument_id not in {leg.instrument_id for leg in self.legs}:
            raise ValueError("proposed trade subject must identify one of its legs")
        return self


type IdeaOutput = Observation | Signal | ProposedPosition | ProposedTrade
