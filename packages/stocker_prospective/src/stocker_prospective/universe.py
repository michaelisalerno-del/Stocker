"""Registered frozen-universe loading with provenance and membership checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from stocker_prospective.bundle import ANCHOR_COHORT


class UniverseError(RuntimeError):
    """The registered universe failed its identity contract."""


class RegisteredUniverse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    universe_id: str
    cohort: str
    symbols: tuple[str, ...]
    universe_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_artifact: str
    source_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_field: str
    registration_note: str


def _symbol_hash(symbols: list[str] | tuple[str, ...]) -> str:
    canonical = json.dumps(list(symbols), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256((canonical + "\n").encode("utf-8")).hexdigest()


def load_registered_universe(path: str | Path) -> RegisteredUniverse:
    """Load the actual registered 20-stock artifact; never infer membership."""

    source = Path(path)
    if not source.is_file():
        raise UniverseError(
            f"blocked_frozen_universe_mismatch: missing universe artifact at {source}"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        universe = RegisteredUniverse.model_validate(payload)
    except Exception as exc:
        raise UniverseError("blocked_frozen_universe_mismatch: invalid artifact") from exc
    if (
        universe.cohort != ANCHOR_COHORT
        or len(universe.symbols) != 20
        or len(set(universe.symbols)) != 20
        or any(symbol != symbol.upper() for symbol in universe.symbols)
        or _symbol_hash(universe.symbols) != universe.universe_hash
    ):
        raise UniverseError(
            "blocked_frozen_universe_mismatch: anchor_frozen_20 membership/hash changed"
        )
    return universe
