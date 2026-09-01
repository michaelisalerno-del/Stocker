"""Small, broker-independent universe definitions and lookup."""

from collections.abc import Iterable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class InstrumentReference(BaseModel):
    """Unresolved stock identity sufficient for later IBKR qualification."""

    model_config = ConfigDict(frozen=True)

    symbol: Identifier
    exchange: Identifier
    primary_exchange: Identifier | None = None
    currency: Identifier
    security_type: Identifier = "STK"

    @field_validator("symbol", "exchange", "primary_exchange", "currency", "security_type")
    @classmethod
    def normalize_market_identity(cls, value: str | None) -> str | None:
        """Normalize identity fields so equal references compare equal across universes."""

        return value.upper() if value is not None else None


class UniverseDefinition(BaseModel):
    """One immutable named collection of eligible instrument references."""

    model_config = ConfigDict(frozen=True)

    universe_id: Identifier
    name: Identifier
    members: tuple[InstrumentReference, ...] = Field(min_length=1)

    @field_validator("members")
    @classmethod
    def remove_duplicate_members(
        cls, members: tuple[InstrumentReference, ...]
    ) -> tuple[InstrumentReference, ...]:
        """Keep first occurrence order while removing exact normalized duplicates."""

        return tuple(dict.fromkeys(members))


class UniverseCatalog:
    """Explicit in-memory lookup for configured universe definitions."""

    def __init__(self, universes: Iterable[UniverseDefinition]) -> None:
        self._universes: dict[str, UniverseDefinition] = {}
        for universe in universes:
            if universe.universe_id in self._universes:
                raise ValueError(f"Duplicate universe_id: {universe.universe_id}")
            self._universes[universe.universe_id] = universe

    def get_universe(self, universe_id: str) -> UniverseDefinition:
        """Return one universe or reject an unknown identifier clearly."""

        try:
            return self._universes[universe_id]
        except KeyError as exc:
            raise ValueError(f"Unknown universe: {universe_id}") from exc

    def get_members(self, universe_id: str) -> tuple[InstrumentReference, ...]:
        """Return the immutable members of one configured universe."""

        return self.get_universe(universe_id).members

    def list_universes(self) -> tuple[UniverseDefinition, ...]:
        """List definitions in deterministic configuration order."""

        return tuple(self._universes.values())
