"""Frozen, outcome-free structural family mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_FORBIDDEN_MAPPING_TOKENS = (
    "net_payoff",
    "episode_result",
    "activation_result",
    "profitability_class",
)


@dataclass(frozen=True)
class FamilyTaxonomy:
    """Map frozen loop/orientation pairs to topology-only destination families."""

    mapping_id: str
    destination_families: tuple[str, ...]
    unknown_family: str
    pair_to_family: dict[tuple[str, str], str]

    @classmethod
    def from_json(cls, path: Path) -> FamilyTaxonomy:
        raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        canonical = json.dumps(raw, sort_keys=True).lower()
        present = [token for token in _FORBIDDEN_MAPPING_TOKENS if token in canonical]
        if present:
            raise ValueError(f"family mapping contains outcome-derived tokens: {present}")
        if raw.get("registered_before_scoring") is not True:
            raise ValueError("family mapping was not registered before scoring")
        families = tuple(str(value) for value in raw["destination_families"])
        if len(families) != len(set(families)):
            raise ValueError("duplicate destination family")
        pair_to_family: dict[tuple[str, str], str] = {}
        for row in raw["pairs"]:
            key = (str(row["loop_id"]), str(row["orientation"]))
            family = str(row["destination_family"])
            if key in pair_to_family:
                raise ValueError(f"duplicate family mapping: {key}")
            if family not in families:
                raise ValueError(f"pair maps outside registered families: {key}")
            pair_to_family[key] = family
        return cls(
            mapping_id=str(raw["mapping_id"]),
            destination_families=families,
            unknown_family=str(raw["unknown_family"]),
            pair_to_family=pair_to_family,
        )

    def family_for(self, loop_id: object, orientation: object) -> str:
        return self.pair_to_family.get(
            (str(loop_id), str(orientation)),
            self.unknown_family,
        )

    def map_pairs(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted({"loop_id", "orientation"} - set(frame.columns))
        if missing:
            raise ValueError(f"missing pair columns: {missing}")
        result = frame.copy()
        result["destination_family"] = [
            self.family_for(loop_id, orientation)
            for loop_id, orientation in result[["loop_id", "orientation"]].itertuples(
                index=False, name=None
            )
        ]
        result["family_mapping_status"] = result["destination_family"].map(
            lambda value: "mapped" if value != self.unknown_family else "unknown_topology"
        )
        return result


__all__ = ["FamilyTaxonomy"]
