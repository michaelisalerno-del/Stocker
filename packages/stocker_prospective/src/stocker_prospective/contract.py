"""Binding scientific and broker-safety claims for recorder V0."""

from __future__ import annotations

import hashlib
import json
from typing import Final

M1C_FROZEN_THRESHOLD: Final[float] = 0.488333710794033
M1C_BOTTOM_5_THRESHOLD: Final[float] = 0.115697407847643
M1C_BOTTOM_10_THRESHOLD: Final[float] = 0.135896965695626
M1C_BOTTOM_20_THRESHOLD: Final[float] = 0.167095528962669
M1C_FEATURE_MANIFEST_SHA256: Final[str] = (
    "6f59177a58973d33a24741e3c265e1831bfb6dc07afac17ae371501019bdc5cc"
)
M1C_THRESHOLD_ARTIFACT_SHA256: Final[str] = (
    "1aae6b7b28bf0f51b914d069bb31ac2e209b43ddaaa938fd373c55a2e65cbabe"
)
M1C_SCALING_ARTIFACT_SHA256: Final[str] = (
    "9521b093f01313a4993a9e101ef0e214ab32933809585ef267551747762b49c2"
)
SECTOR_PROXY_BY_SYMBOL: Final[dict[str, str]] = {
    "AAL": "XLI",
    "AAOI": "XLK",
    "APLD": "XLK",
    "ASTS": "XLC",
    "CIFR": "XLK",
    "HIMS": "XLV",
    "IONQ": "XLK",
    "IREN": "XLK",
    "MARA": "XLK",
    "MP": "XLB",
    "MRNA": "XLV",
    "MSTR": "XLK",
    "NVTS": "XLK",
    "QBTS": "XLK",
    "RGTI": "XLK",
    "RIOT": "XLK",
    "RIVN": "XLY",
    "SMCI": "XLK",
    "SOFI": "XLF",
    "WULF": "XLK",
}
CONTRACT_VERSION: Final[str] = "frozen-m1c-microstructure-recorder-v0"
BUDGET_AWARE_RECORDER_CONTRACT_VERSION: Final[str] = "ibkr-budget-aware-shadow-recorder-v0"
ORIGINAL_LOW_MOVEMENT_DECISION: Final[str] = "blocked_insufficient_low_tail_support"
type ClaimValue = bool | float | str

CLAIMS_BOUNDARY: Final[dict[str, ClaimValue]] = {
    "research_only": True,
    "record_only": True,
    "frozen_m1c": True,
    "source_transfer_monitoring": True,
    "exact_vendor_bar_equality_required": False,
    "market_data_source": "ibkr",
    "historical_research_source": "eodhd",
    "cross_vendor_validation_diagnostic_only": True,
    "cross_vendor_validation_required_for_science": False,
    "prospective_evidence_description": (
        "prospective evaluation of the frozen implementation using IBKR market data"
    ),
    "option_shadow_outcomes_only": True,
    "historical_engineering_phase_sessions": 20,
    "market_data_budget_enforced": True,
    "market_data_limits_runtime_discovered": True,
    "full_option_chain_streaming_allowed": False,
    "tick_by_tick_universe_streaming_allowed": False,
    "level2_universe_streaming_allowed": False,
    "reserved_future_trading_capacity": True,
    "paper_orders_allowed": False,
    "live_orders_allowed": False,
    "order_methods_available": False,
    "account_access_required": False,
    "position_access_required": False,
    "strategy_promotion": False,
    "original_low_movement_decision_preserved": True,
    "original_decision": ORIGINAL_LOW_MOVEMENT_DECISION,
    "retrospective_gate_relaxation_allowed": False,
    "prospective_collection": True,
    "prospective_record_only": True,
    "m1c_frozen": True,
    "m1c_threshold": M1C_FROZEN_THRESHOLD,
    "m1c_bottom_5_threshold": M1C_BOTTOM_5_THRESHOLD,
    "m1c_bottom_10_threshold": M1C_BOTTOM_10_THRESHOLD,
    "m1c_bottom_20_threshold": M1C_BOTTOM_20_THRESHOLD,
    "primary_quiet_state": "bottom_10_percent",
    "a1_frozen_prospective_hypothesis": True,
    "c1_frozen_comparison": True,
    "r1_frozen_comparison": True,
    "microstructure_features_descriptive_only": True,
    "microstructure_direction_model_fitted": False,
    "option_quotes_recorded": True,
    "option_pnl_is_shadow_quote_pnl": True,
    "defined_risk_short_premium_only": True,
    "naked_short_options_allowed": False,
    "broker_order_methods_allowed": False,
    "place_order_method_available": False,
    "broker_account_mutation_allowed": False,
    "account_balance_access_required": False,
    "execution_enabled": False,
    "protected_historical_start": "2026-01-01",
}

FORBIDDEN_BROKER_METHODS: Final[frozenset[str]] = frozenset(
    {
        "placeOrder",
        "cancelOrder",
        "reqOpenOrders",
        "reqPositions",
        "reqAccountUpdates",
        "reqAccountSummary",
        "reqAccountUpdatesMulti",
        "reqPositionsMulti",
        "reqExecutions",
        "reqCompletedOrders",
        "reqGlobalCancel",
        "exerciseOptions",
    }
)


def claims_boundary() -> dict[str, ClaimValue]:
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
