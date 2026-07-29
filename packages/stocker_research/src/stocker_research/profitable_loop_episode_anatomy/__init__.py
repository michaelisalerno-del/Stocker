"""Read-only descriptive anatomy of profitable loop episodes.

The package deliberately exposes data transformations only.  It has no broker,
order, position, strategy, or application-runtime integration.
"""

from .analysis import (
    attach_episode_membership,
    block_circular_pair_shift,
    build_episode_ledgers,
    build_synchronized_panel,
    classify_episode,
    collapse_stock_contributions,
    common_factor_diagnostic,
    concentration_attribution,
    decode_history_token,
    decompose_payoff_components,
    early_leader_checkpoints,
    exact_rerun_identity,
    four_way_counterfactual,
    frozen_run_git_head,
    poisson_binomial_null,
    recompute_component_summary_after_stock_removal,
    validate_causal_indicators,
)
from .census import reproduce_exploratory_census

__all__ = [
    "attach_episode_membership",
    "block_circular_pair_shift",
    "build_episode_ledgers",
    "build_synchronized_panel",
    "classify_episode",
    "collapse_stock_contributions",
    "common_factor_diagnostic",
    "concentration_attribution",
    "decode_history_token",
    "decompose_payoff_components",
    "early_leader_checkpoints",
    "exact_rerun_identity",
    "four_way_counterfactual",
    "frozen_run_git_head",
    "poisson_binomial_null",
    "recompute_component_summary_after_stock_removal",
    "reproduce_exploratory_census",
    "validate_causal_indicators",
]
