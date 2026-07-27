"""Binding scientific and broker-safety claims for recorder V0."""

from __future__ import annotations

import hashlib
import json
from typing import Final

M1C_FROZEN_THRESHOLD: Final[float] = 0.488333710794033
CONTRACT_VERSION: Final[str] = "frozen-m1c-microstructure-recorder-v0"

CLAIMS_BOUNDARY: Final[dict[str, bool | float]] = {
    "research_only": True,
    "prospective_collection": True,
    "record_only": True,
    "m1c_frozen": True,
    "m1c_threshold": M1C_FROZEN_THRESHOLD,
    "a1_frozen_prospective_hypothesis": True,
    "c1_frozen_comparison": True,
    "r1_frozen_comparison": True,
    "microstructure_features_descriptive_only": True,
    "microstructure_direction_model_fitted": False,
    "option_quotes_recorded": True,
    "option_pnl_is_shadow_quote_pnl": True,
    "paper_orders_allowed": False,
    "live_orders_allowed": False,
    "place_order_method_available": False,
    "broker_account_mutation_allowed": False,
    "position_access_required": False,
    "account_balance_access_required": False,
    "execution_enabled": False,
    "strategy_promotion": False,
}

FORBIDDEN_BROKER_METHODS: Final[frozenset[str]] = frozenset(
    {
        "placeOrder",
        "cancelOrder",
        "reqOpenOrders",
        "reqPositions",
        "reqAccountUpdates",
        "reqAccountSummary",
        "reqGlobalCancel",
        "exerciseOptions",
    }
)


def claims_boundary() -> dict[str, bool | float]:
    """Return a fresh JSON-safe copy for contracts and exported datasets."""

    return dict(CLAIMS_BOUNDARY)


def claims_hash() -> str:
    payload = json.dumps(CLAIMS_BOUNDARY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_no_broker_mutation_surface(adapter: object) -> None:
    """Fail if the recorder-visible adapter exposes any mutation/account method."""

    exposed = sorted(name for name in FORBIDDEN_BROKER_METHODS if hasattr(adapter, name))
    if exposed:
        raise RuntimeError("blocked_order_capable_recorder_surface: " + ", ".join(exposed))
