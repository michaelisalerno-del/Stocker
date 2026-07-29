"""Frozen-contract loading and identity checks for the signature atlas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def contract_sha256(path: Path) -> str:
    """Return the byte identity of a frozen contract."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(contract: dict[str, Any]) -> None:
    """Fail closed when an atlas contract crosses its research boundary."""

    required = {
        "contract_id",
        "experiment",
        "research_only",
        "execution_enabled",
        "population",
        "costs",
        "chronology",
        "movement_permission",
        "search",
        "support",
        "prospective_completion",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise ValueError(f"contract missing required fields: {missing}")
    if contract["research_only"] is not True or contract["execution_enabled"] is not False:
        raise ValueError("contract violates research-only safety flags")
    population = contract["population"]
    if population.get("decision_ordinals") != [12, 36]:
        raise ValueError("frozen decision ordinals drifted")
    if population.get("horizon_bars") != 24:
        raise ValueError("frozen 24-bar horizon drifted")
    costs = contract["costs"]
    if float(costs.get("entry_bps", -1)) + float(costs.get("exit_bps", -1)) != float(
        costs.get("round_trip_bps", -2)
    ):
        raise ValueError("round-trip cost is internally inconsistent")
    if contract["search"].get("maximum_conditions") != 3:
        raise ValueError("signature complexity ceiling drifted")


def load_contract(path: Path) -> dict[str, Any]:
    """Load and validate a versioned JSON contract."""

    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise TypeError("contract must be a JSON object")
    validate_contract(contract)
    return contract
