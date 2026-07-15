from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stocker_research.dynamic_loop_edge_state_lead_lag.matching import (
    STRUCTURAL_LINEAGE_FIELDS,
    build_trade_delay_tables,
    match_next_session_setups,
    reconstruct_v2_shifted_policy,
)

ROOT = Path(__file__).resolve().parents[1]
V2_PRIMARY = (
    ROOT
    / "research/slrno-v2/20260714-regime-loop-handoff/work/artifacts"
    / "20260714-dynamic-loop-edge-state-v2/primary"
)


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": [2025, 2025, 2025],
            "score_session": ["2025-01-03", "2025-01-06", "2025-01-08"],
            "session_index": [0, 1, 2],
        }
    )


def _opportunities() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "period": [2025, 2025, 2025],
            "score_session": ["2025-01-03", "2025-01-06", "2025-01-06"],
            "session_index": [0, 1, 1],
            "opportunity_id": ["original", "later-lineage", "same-loop-replacement"],
            "anchor_id": [1, 2, 3],
            "symbol_norm": ["AAA", "AAA", "BBB"],
            "loop_id": ["cycle_01"] * 3,
            "orientation": ["state_2"] * 3,
            "horizon": [24] * 3,
            "history_token": [42, 42, 99],
            "top_loop_cycle": ["1->2->1", "1->2->1", "1->2->1"],
            "strategy": ["frozen"] * 3,
            "family": ["causal"] * 3,
            "representation": ["loop_scores"] * 3,
            "direction": [1, 1, 1],
            "status": ["filled"] * 3,
            "accepted": [True, False, False],
            "entry_timestamp": pd.to_datetime(
                ["2025-01-03T15:00:00Z", "2025-01-06T15:00:00Z", "2025-01-06T16:00:00Z"]
            ),
            "exit_timestamp": pd.to_datetime(
                ["2025-01-03T17:00:00Z", "2025-01-06T17:00:00Z", "2025-01-06T18:00:00Z"]
            ),
            "entry_price": [100.0, 101.0, 50.0],
            "exit_price": [102.0, 104.0, 51.0],
            "gross_return_bps": [200.0, 297.0297029703, 200.0],
            "entry_cost_bps": [5.0] * 3,
            "exit_cost_bps": [5.0] * 3,
            "primary_total_cost_bps": [10.0] * 3,
            "primary_net_payoff_bps": [190.0, 287.0297029703, 190.0],
            "holding_bars": [24] * 3,
            "existing_position_action": ["unchanged_existing_exit_rule"] * 3,
        }
    )
    return frame


def test_exact_match_rejects_different_setup_with_same_loop_label() -> None:
    opportunities = _opportunities()

    matches = match_next_session_setups(opportunities.iloc[:1], opportunities, _calendar())
    match = matches.iloc[0]

    assert match["match_category"] == "same_structural_lineage_not_exact_setup"
    assert match["exact_match"] is False or not bool(match["exact_match"])
    assert pd.isna(match["matched_opportunity_id"])
    assert match["structural_lineage_opportunity_id"] == "later-lineage"
    assert match["structural_lineage_opportunity_id"] != "same-loop-replacement"


def test_session_local_anchor_reuse_is_not_persistent_exact_identity() -> None:
    opportunities = _opportunities()
    opportunities.loc[1, "anchor_id"] = opportunities.loc[0, "anchor_id"]

    matches = match_next_session_setups(opportunities.iloc[:1], opportunities, _calendar())

    assert not bool(matches.iloc[0]["exact_match"])
    assert pd.isna(matches.iloc[0]["matched_opportunity_id"])


def test_replacement_opportunity_never_enters_exact_matched_population() -> None:
    opportunities = _opportunities()
    matches = match_next_session_setups(opportunities.iloc[:1], opportunities, _calendar())
    tables = build_trade_delay_tables(matches, opportunities)

    assert tables.exact_matches.empty
    assert len(tables.restarted_horizon) == 1
    assert tables.restarted_horizon["match_basis"].eq("structural_lineage_diagnostic").all()
    assert tables.restarted_horizon["delayed_opportunity_id"].iloc[0] == "later-lineage"


def test_restarted_horizon_and_constant_terminal_time_are_separate_and_costed() -> None:
    opportunities = _opportunities()
    matches = match_next_session_setups(opportunities.iloc[:1], opportunities, _calendar())
    tables = build_trade_delay_tables(matches, opportunities)

    restarted = tables.restarted_horizon.iloc[0]
    constant = tables.constant_terminal.iloc[0]
    assert restarted["delayed_net_payoff_bps"] == pytest.approx(
        restarted["delayed_gross_payoff_bps"] - 10.0
    )
    assert restarted["delayed_horizon_bars"] == 24
    assert constant["constant_terminal_available"] is False or not bool(
        constant["constant_terminal_available"]
    )
    assert constant["unavailable_reason"] == "original_terminal_precedes_delayed_entry"
    assert pd.isna(constant["delayed_constant_terminal_net_bps"])
    assert restarted["existing_position_action"] == "unchanged_existing_exit_rule"


def test_structural_lineage_fields_are_stronger_than_loop_label() -> None:
    assert {
        "symbol_norm",
        "history_token",
        "top_loop_cycle",
        "direction",
    } <= set(STRUCTURAL_LINEAGE_FIELDS)


def test_v2_shift_reconstruction_uses_prior_opportunity_session_not_calendar_session() -> None:
    decisions = pd.DataFrame(
        {
            "period": [2025, 2025],
            "loop_id": ["cycle_01", "cycle_01"],
            "orientation": ["state_2", "state_2"],
            "horizon": [24, 24],
            "score_session": ["2025-01-03", "2025-01-08"],
            "session_index": [0, 2],
            "opportunity_id": ["a", "b"],
            "accepted": [True, False],
            "status": ["filled", "filled"],
            "primary_net_payoff_bps": [-10.0, 20.0],
        }
    )

    reconstructed = reconstruct_v2_shifted_policy(decisions)

    second = reconstructed.loc[reconstructed["opportunity_id"].eq("b")].iloc[0]
    assert bool(second["delayed_accepted"])
    assert second["policy_source_session"] == "2025-01-03"
    assert second["policy_gap_sessions"] == 2
    assert second["population_category"] == "introduced"


def test_frozen_v2_delayed_result_reconstructs_exactly_before_decomposition() -> None:
    decisions = pd.read_parquet(V2_PRIMARY / "trade_decisions.parquet")
    full = decisions.loc[decisions["model_name"].eq("hierarchical_change_point")]

    reconstructed = reconstruct_v2_shifted_policy(full)
    immediate = reconstructed.loc[
        reconstructed["immediate_accepted"] & reconstructed["status"].eq("filled")
    ]
    delayed = reconstructed.loc[
        reconstructed["delayed_accepted"] & reconstructed["status"].eq("filled")
    ]

    assert len(immediate) == 275
    assert len(delayed) == 259
    assert immediate["primary_net_payoff_bps"].sum() == pytest.approx(-9416.383037515745)
    assert delayed["primary_net_payoff_bps"].sum() == pytest.approx(9014.535299458603)
    categories = reconstructed.groupby("population_category")["primary_net_payoff_bps"].sum()
    assert categories["introduced"] - categories["dropped"] == pytest.approx(18430.918336974348)
