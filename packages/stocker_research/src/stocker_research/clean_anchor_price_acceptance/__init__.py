"""Causal clean-anchor, first-bar price-acceptance research helpers."""

from stocker_research.clean_anchor_price_acceptance.checkpoint import (
    CheckpointBar,
    PriceAcceptance,
    calculate_price_acceptance,
    select_first_post_anchor_bar,
)
from stocker_research.clean_anchor_price_acceptance.immutable_ledger import (
    ProspectiveAcceptanceLedger,
)
from stocker_research.clean_anchor_price_acceptance.outcomes import (
    RemainingPayoff,
    calculate_remaining_payoff,
)
from stocker_research.clean_anchor_price_acceptance.variants import (
    VARIANT_RULES,
    build_variant_decisions,
    variant_population_identity,
)

__all__ = [
    "CheckpointBar",
    "PriceAcceptance",
    "ProspectiveAcceptanceLedger",
    "calculate_price_acceptance",
    "calculate_remaining_payoff",
    "select_first_post_anchor_bar",
    "build_variant_decisions",
    "variant_population_identity",
    "RemainingPayoff",
    "VARIANT_RULES",
]
