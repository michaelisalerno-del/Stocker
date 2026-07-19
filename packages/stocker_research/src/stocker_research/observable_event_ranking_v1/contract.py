"""Frozen scientific and safety contract for Observable Event Ranking V1."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Final

EXPERIMENT_ID: Final = "20260719-observable-event-cross-sectional-ranking-v1"
EVENT_FAMILY: Final = "E1_POSITIVE_RELATIVE_ACCELERATION"
MODEL_ID: Final = "M1_POOLED_LINEAR_RANKER"
DEVELOPMENT_CUTOFF: Final = "2025-12-31"

REQUIRED_SAFETY_FLAGS: Final[dict[str, bool | str]] = {
    "research_only": True,
    "execution_enabled": False,
    "order_placement": "disabled",
    "strategy_promotion": False,
    "production_runtime_modified": False,
    "live_trading_enabled": False,
    "paper_order_submission_enabled": False,
    "account_data_requested": False,
    "positions_requested": False,
    "ibkr_orders_permitted": False,
}

DECISION_CLOCKS: Final[tuple[str, ...]] = (
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
)

PRIMARY_FEATURES: Final[tuple[str, ...]] = (
    "event_strength",
    "market_relative_return_5m",
    "market_relative_return_15m",
    "market_relative_return_30m",
    "sector_relative_return_15m",
    "sector_relative_return_30m",
    "market_relative_acceleration_z",
    "sector_relative_acceleration_z",
    "realized_volatility_30m",
    "activity_shock_z",
    "distance_from_session_high",
    "session_fraction",
)

BASELINES: Final[tuple[str, ...]] = (
    "B0_RANDOM_ELIGIBLE",
    "B1_EVENT_STRENGTH",
    "B2_PREVIOUS_5M_MARKET_RELATIVE_RETURN",
    "B3_15M_MARKET_RELATIVE_STRENGTH",
    "B4_30M_MARKET_RELATIVE_STRENGTH",
    "B5_15M_SECTOR_RELATIVE_STRENGTH",
    "B6_ACTIVITY_SHOCK",
    "B7_REALIZED_VOLATILITY",
    "B8_TRAINING_ONLY_STOCK_CLOCK_EVENT_FREQUENCY",
    "B9_TRAINING_ONLY_STOCK_CLOCK_MEAN_TARGET_PRIOR",
)

FORBIDDEN_PRIMARY_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "regime",
        "regime_id",
        "state",
        "state_id",
        "posterior",
        "loop",
        "loop_id",
        "loop_prefix",
        "loop_completion",
        "loop_route",
        "excursion_origin",
        "excursion_distance",
        "excursion_resolution",
        "excursion_timing",
        "slrno_prediction",
        "profitable_period",
        "personality",
        "template_label",
        "payoff_candidate",
        "mfe",
        "mae",
        "target_before_stop",
        "stop_before_target",
        "future_price",
        "future_return",
        "rank_target",
        "pnl",
        "cost",
        "spread",
        "slippage",
    }
)

_FROZEN_CONTRACT: Final[dict[str, Any]] = {
    "experiment_id": EXPERIMENT_ID,
    "contract_version": "observable_event_cross_sectional_ranking_v1.0.0",
    "frozen_on": "2026-07-19",
    "development_cutoff": DEVELOPMENT_CUTOFF,
    "protected_period_policy": {
        "post_cutoff_data_opening_permitted": False,
        "casual_bypass_cli_flag": False,
    },
    "safety": REQUIRED_SAFETY_FLAGS,
    "scientific_levels": [
        "descriptive_market_structure",
        "structural_predictability",
        "directional_prediction",
        "gross_economic_payoff",
        "executable_net_trading_edge",
    ],
    "interpretation_boundaries": {
        "positive_information_coefficient_is_executable_edge": False,
        "provider_bar_price_is_achieved_ibkr_fill": False,
        "observed_ibkr_bid_ask_proves_fill": False,
    },
    "retired_research": {
        "regime_or_state_inputs": False,
        "loop_or_route_inputs": False,
        "excursion_inputs": False,
        "prior_slrno_predictions_or_labels": False,
        "prior_payoff_selection": False,
        "mfe_mae_or_target_stop_labels": False,
        "stock_identifier_primary_input": False,
        "current_ibkr_metadata_primary_input": False,
    },
    "primary_hypothesis": {
        "text": (
            "Among simultaneously eligible US stocks exhibiting a newly confirmed positive "
            "market-and-sector-relative acceleration event, a pooled linear ranker using only "
            "causally observable price and activity features predicts the 60-minute future "
            "return rank, after a complete five-minute recognition and dispatch delay, better "
            "than the strongest frozen simple observable baseline."
        ),
        "event_family": EVENT_FAMILY,
        "model": MODEL_ID,
        "side": "long_only_structural_ranking",
        "additional_event_families_permitted": False,
    },
    "source": {
        "historical_provider": "EODHD",
        "timeframe": "5m",
        "timezone_session": "America/New_York",
        "timezone_storage": "UTC",
        "volume_interpretation": "provider_reported_activity_proxy",
        "source_mixing_permitted": False,
        "bar_label_convention": "must_be_verified_per_source_before_scientific_run",
        "required_row_identity": [
            "source_provider",
            "source_dataset_id",
            "source_hash",
            "symbol",
            "session",
            "bar_start",
            "bar_end",
            "source_availability_timestamp",
            "timezone",
            "adjustment_status",
            "corporate_action_status",
            "gap_status",
            "volume_is_provider_activity_proxy",
        ],
    },
    "universe": {
        "cadence": "monthly_point_in_time",
        "security_type": "US_common_stock",
        "currency": "USD",
        "previous_close_min_usd": 5.0,
        "prior_valid_sessions_min": 60,
        "trailing_sessions": 20,
        "median_daily_dollar_activity_min_usd": 20_000_000.0,
        "bar_coverage_min": 0.95,
        "same_sector_other_peers_min": 5,
        "current_constituent_backfill_permitted": False,
        "current_ibkr_availability_historical_filter_permitted": False,
        "market_cap_filter": "absent_unless_causally_point_in_time_data_exists",
    },
    "decision_grid_new_york": list(DECISION_CLOCKS),
    "grid_assignment": {
        "ordinary": "first_grid_strictly_after_confirmation_and_availability",
        "exact_grid": "same_grid_only_with_proven_pre_scoring_source_availability",
        "early_close": "eligible_only_when_entry_and_primary_outcome_interval_exists",
    },
    "event": {
        "id": EVENT_FAMILY,
        "recent_return_bars": 3,
        "preceding_return_bars": 3,
        "peer_aggregation": "leave_one_stock_out_median",
        "market_stocks_min": 20,
        "sector_other_stocks_min": 5,
        "robust_scaling": {
            "sessions": 60,
            "exclude_current_session": True,
            "location": "median",
            "scale": "1.4826_times_MAD",
            "epsilon": 1e-8,
            "fallback": "IQR_divided_by_1.349_then_epsilon",
            "population": "same_clock_pooled_eligible_rows_then_prior_session_pool",
        },
        "strength": "min(market_relative_acceleration_z,sector_relative_acceleration_z)",
        "threshold": "single_outcome_free_q90_earliest_six_valid_months",
        "threshold_inclusive": True,
        "deduplication": "first_event_per_stock_session",
        "return_formulas": {
            "recent_15m_return": "close_t/close_t_minus_3-1",
            "preceding_15m_return": "close_t_minus_3/close_t_minus_6-1",
            "market_relative_acceleration": ("recent_market_relative-preceding_market_relative"),
            "sector_relative_acceleration": ("recent_sector_relative-preceding_sector_relative"),
        },
        "trigger": [
            "recent_market_relative>0",
            "recent_sector_relative>0",
            "event_strength>=frozen_q90_threshold",
            "all_contributing_bars_contiguous_complete_and_available",
            "point_in_time_universe_eligible",
            "primary_outcome_interval_fits_regular_session",
        ],
    },
    "support_gate": {
        "valid_slate_eligible_stocks_min": 50,
        "supported_slate_candidates_min": 8,
        "supported_slates_min": 1_000,
        "unique_event_rows_min": 5_000,
        "candidate_support_fraction_min": 0.60,
        "event_stocks_min": 30,
        "calendar_months_min": 6,
        "max_stock_event_fraction": 0.075,
        "requires_exact_event_rerun": True,
        "requires_outcome_free_event_audit": True,
    },
    "features": list(PRIMARY_FEATURES),
    "feature_processing": {
        "imputation_fit": "training_only",
        "clipping_fit": "training_only",
        "standardisation_fit": "training_only",
        "activity_shock_population": "prior_stock_by_clock_provider_dollar_activity",
    },
    "targets": {
        "entry_reference": "open_of_t_plus_2",
        "recognition_dispatch_delay_minutes": 5,
        "primary": "within_slate_percentile_rank_future_return_60m",
        "primary_rank_range": [0.0, 1.0],
        "tie_method": "average",
        "secondary": ["future_return_rank_30m", "session_close_future_return_rank"],
        "descriptive": ["future_return_rank_15m", "future_absolute_movement_rank"],
        "minimum_valid_targets": 8,
        "maximum_unavailable_fraction": 0.10,
    },
    "folds": {
        "kind": "expanding_chronological",
        "initial_training_months_min": 6,
        "evaluation_months": 3,
        "random_cross_validation": False,
    },
    "baselines": list(BASELINES),
    "baseline_selection": {
        "statistic": "pooled_historical_oof_mean_per_slate_spearman_ic",
        "random_baseline_selectable": False,
        "stock_clock_priors": "training_only_with_global_shrinkage_fallback",
    },
    "model": {
        "id": MODEL_ID,
        "kind": "regularized_linear_regression",
        "alpha": 1.0,
        "stock_identifier_input": False,
        "sector_identifier_input": False,
        "hyperparameter_search": False,
        "sample_weight": "1_over_slate_size",
        "preprocessing_fit": "training_only",
    },
    "uncertainty": {
        "primary": "paired_session_block_bootstrap",
        "draws": 2_000,
        "seed": 20260719,
        "confidence": 0.95,
    },
    "historical_development_gate": {
        "candidate_mean_ic_exceeds_strongest_baseline": True,
        "candidate_top_two_minus_median_exceeds_strongest_baseline": True,
        "positive_evaluation_fold_fraction_strictly_above": 0.5,
        "max_stock_event_fraction": 0.075,
        "max_stock_top_two_selection_fraction": 0.15,
        "requires_exact_rerun": True,
        "requires_independent_audit": True,
        "authorises_only": "prospective_freeze",
    },
    "prospective_gate": {
        "historical_data_may_satisfy": False,
        "months_min": 6,
        "evaluable_supported_slates_min": 1_000,
        "unique_event_rows_min": 5_000,
        "candidate_minus_baseline_ic_min": 0.01,
        "candidate_minus_baseline_ic_ci_lower_above_zero": True,
        "top_two_difference_ci_lower_above_zero": True,
        "positive_first_six_months_min": 4,
        "positive_leave_one_stock_out_fraction_min": 0.90,
        "max_stock_event_fraction": 0.075,
        "top_five_selection_fraction_max": 0.25,
        "requires_exact_rerun": True,
        "requires_independent_audit": True,
    },
    "economic_layer": {
        "enabled_by_default": False,
        "generic_cost_assumption_permitted": False,
        "requires_passed_prospective_gate": True,
        "requires_live_ibkr_bid_ask": True,
        "quote_simulation_is_fill_claim": False,
        "eventual_continuation_gate": {
            "complete_live_quote_coverage_min": 0.80,
            "net_top_two_session_bootstrap_lower_above_zero": True,
            "positive_under_cost_stress_multiplier": 1.5,
            "max_stock_net_contribution_fraction": 0.15,
            "single_short_period_domination_permitted": False,
            "requires_exact_rerun_and_audit": True,
        },
    },
    "ibkr_observability": {
        "orders_permitted": False,
        "account_or_position_requests_permitted": False,
        "connection_default": "disabled_localhost_only",
        "maximum_observation_delay_seconds": 10.0,
        "primary_quote": "first_complete_live_bid_ask_within_window",
        "delayed_or_delayed_frozen_executable": False,
        "frozen_is_current": False,
        "partial_quote_is_complete_top_of_book": False,
        "historical_universe_filter_permitted": False,
    },
    "forbidden_primary_columns": sorted(FORBIDDEN_PRIMARY_COLUMNS),
}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for scientific identity hashes."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Return a SHA-256 over canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def frozen_contract() -> dict[str, Any]:
    """Return an isolated copy of the immutable V1 contract."""

    return deepcopy(_FROZEN_CONTRACT)
