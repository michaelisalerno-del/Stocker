"""Fail-closed scientific validity gates."""

from __future__ import annotations

from dataclasses import dataclass

from stocker_prospective.market_data import MarketDataType


@dataclass(frozen=True)
class EpisodeSafetyInputs:
    capability_preflight_passed: bool
    m1c_parity_passed: bool
    direction_parity_passed: bool
    market_data_type: MarketDataType
    previous_close_group_o_valid: bool
    trigger_bar_complete: bool
    clock_drift_within_tolerance: bool
    underlying_quote_fresh: bool
    unresolved_bar_gap: bool
    deterministic_episode_identity: bool
    raw_event_storage_writable: bool
    scientific_recording_authorized: bool


@dataclass(frozen=True)
class EpisodeSafetyDecision:
    scientific_recording_valid: bool
    rejection_reasons: tuple[str, ...]


def evaluate_episode_safety(inputs: EpisodeSafetyInputs) -> EpisodeSafetyDecision:
    """Validate causal signal evidence against the frozen safety contract."""

    checks = (
        (inputs.capability_preflight_passed, "ibkr_capability_preflight_failed"),
        (inputs.m1c_parity_passed, "m1c_parity_failed"),
        (inputs.direction_parity_passed, "direction_parity_failed"),
        (inputs.market_data_type is MarketDataType.LIVE, "market_data_not_live"),
        (inputs.previous_close_group_o_valid, "previous_close_group_o_invalid"),
        (inputs.trigger_bar_complete, "trigger_bar_incomplete"),
        (inputs.clock_drift_within_tolerance, "clock_drift_outside_tolerance"),
        (inputs.underlying_quote_fresh, "underlying_quote_stale"),
        (not inputs.unresolved_bar_gap, "unresolved_bar_gap"),
        (inputs.deterministic_episode_identity, "episode_identity_not_deterministic"),
        (inputs.raw_event_storage_writable, "raw_event_storage_not_writable"),
        (
            inputs.scientific_recording_authorized,
            "scientific_recording_not_authorized",
        ),
    )
    reasons = tuple(reason for passed, reason in checks if not passed)
    return EpisodeSafetyDecision(
        scientific_recording_valid=not reasons,
        rejection_reasons=reasons,
    )


@dataclass(frozen=True)
class OptionOutcomeSafetyInputs:
    contract_resolved: bool
    market_data_type: MarketDataType
    valid_entry_ask: bool
    valid_exit_bid: bool
    quote_freshness_recorded: bool
    subscription_gap_spans_horizon: bool


def evaluate_option_outcome_safety(
    inputs: OptionOutcomeSafetyInputs,
) -> EpisodeSafetyDecision:
    checks = (
        (inputs.contract_resolved, "contract_not_resolved"),
        (inputs.market_data_type is MarketDataType.LIVE, "market_data_not_live"),
        (inputs.valid_entry_ask, "missing_valid_entry_ask"),
        (inputs.valid_exit_bid, "missing_valid_exit_bid"),
        (inputs.quote_freshness_recorded, "quote_freshness_not_recorded"),
        (not inputs.subscription_gap_spans_horizon, "subscription_gap_spans_horizon"),
    )
    reasons = tuple(reason for passed, reason in checks if not passed)
    return EpisodeSafetyDecision(
        scientific_recording_valid=not reasons,
        rejection_reasons=reasons,
    )
