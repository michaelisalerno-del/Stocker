from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.clean_anchor_price_acceptance.outcomes import (
    calculate_remaining_payoff,
)
from stocker_research.clean_anchor_price_acceptance.variants import (
    VARIANT_RULES,
    build_variant_decisions,
    variant_population_identity,
)

ANCHOR = pd.Timestamp("2025-06-02 14:30:00+00:00")
TERMINAL = ANCHOR + pd.Timedelta(minutes=125)


def _path(*, remove_minute: int | None = None) -> pd.DataFrame:
    rows = []
    for minute in range(0, 250, 5):
        if minute == remove_minute:
            continue
        price = 100.0 + minute / 100.0
        rows.append(
            {
                "timestamp": ANCHOR + pd.Timedelta(minutes=minute),
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.5,
            }
        )
    return pd.DataFrame(rows)


def test_constant_terminal_uses_exact_next_open_and_original_terminal() -> None:
    result = calculate_remaining_payoff(
        _path(),
        anchor_timestamp=ANCHOR,
        original_terminal_timestamp=TERMINAL,
        direction=1,
    )

    assert result.status == "available"
    assert result.entry_timestamp == ANCHOR + pd.Timedelta(minutes=10)
    assert result.entry_price == pytest.approx(100.1)
    assert result.exit_timestamp == TERMINAL
    assert result.exit_price == pytest.approx(101.7)


def test_entry_and_exit_costs_are_both_charged() -> None:
    result = calculate_remaining_payoff(
        _path(),
        anchor_timestamp=ANCHOR,
        original_terminal_timestamp=TERMINAL,
        direction=1,
    )

    assert result.entry_cost_bps == 5.0
    assert result.exit_cost_bps == 5.0
    assert result.total_cost_bps == 10.0
    assert result.net_payoff_bps == pytest.approx(result.gross_payoff_bps - 10.0)  # type: ignore[operator]


def test_twice_cost_changes_cost_only() -> None:
    base = calculate_remaining_payoff(
        _path(), anchor_timestamp=ANCHOR, original_terminal_timestamp=TERMINAL, direction=1
    )
    stressed = calculate_remaining_payoff(
        _path(),
        anchor_timestamp=ANCHOR,
        original_terminal_timestamp=TERMINAL,
        direction=1,
        cost_bps_per_side=10.0,
    )

    assert stressed.gross_payoff_bps == base.gross_payoff_bps
    assert stressed.net_payoff_bps == pytest.approx(base.net_payoff_bps - 10.0)  # type: ignore[operator]


def test_missing_exact_entry_does_not_shift_to_later_open() -> None:
    result = calculate_remaining_payoff(
        _path(remove_minute=10),
        anchor_timestamp=ANCHOR,
        original_terminal_timestamp=TERMINAL,
        direction=1,
    )

    assert result.status == "missing_exact_entry_bar"
    assert result.entry_timestamp is None


def test_missing_terminal_is_unavailable_not_zero() -> None:
    result = calculate_remaining_payoff(
        _path(remove_minute=120),
        anchor_timestamp=ANCHOR,
        original_terminal_timestamp=TERMINAL,
        direction=1,
    )

    assert result.status == "missing_exact_terminal_bar"
    assert result.net_payoff_bps is None


def test_additional_bar_delay_keeps_original_terminal() -> None:
    result = calculate_remaining_payoff(
        _path(),
        anchor_timestamp=ANCHOR,
        original_terminal_timestamp=TERMINAL,
        direction=-1,
        additional_delay_bars=1,
    )

    assert result.entry_timestamp == ANCHOR + pd.Timedelta(minutes=15)
    assert result.exit_timestamp == TERMINAL


def test_entry_at_terminal_is_too_late_not_zero() -> None:
    result = calculate_remaining_payoff(
        _path(),
        anchor_timestamp=ANCHOR,
        original_terminal_timestamp=ANCHOR + pd.Timedelta(minutes=10),
        direction=1,
    )

    assert result.status == "too_late"
    assert result.net_payoff_bps is None


def test_restarted_horizon_is_separate_from_constant_terminal() -> None:
    result = calculate_remaining_payoff(
        _path(),
        anchor_timestamp=ANCHOR,
        original_terminal_timestamp=TERMINAL,
        direction=1,
    )

    assert result.restarted_exit_timestamp == ANCHOR + pd.Timedelta(minutes=130)
    assert result.restarted_net_payoff_bps != result.net_payoff_bps


def _variant_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "opportunity_id": "one",
                "source_available": True,
                "availability_status": "available",
                "static_anchor_veto_pass": True,
                "price_acceptance_pass": True,
                "range_permission_available": False,
                "range_permission_pass": False,
                "entry_timestamp": ANCHOR + pd.Timedelta(minutes=10),
                "original_terminal_timestamp": TERMINAL,
                "net_payoff_bps": 30.0,
            },
            {
                "opportunity_id": "two",
                "source_available": True,
                "availability_status": "available",
                "static_anchor_veto_pass": False,
                "price_acceptance_pass": True,
                "range_permission_available": False,
                "range_permission_pass": False,
                "entry_timestamp": ANCHOR + pd.Timedelta(minutes=10),
                "original_terminal_timestamp": TERMINAL,
                "net_payoff_bps": -40.0,
            },
        ]
    )


def test_all_variants_use_identical_source_population_and_clocks() -> None:
    decisions = build_variant_decisions(_variant_source())
    identities = variant_population_identity(decisions)

    assert set(identities) == set(VARIANT_RULES)
    assert len(set(identities.values())) == 1
    assert decisions.groupby("opportunity_id")["entry_timestamp"].nunique().eq(1).all()
    assert decisions.groupby("opportunity_id")["original_terminal_timestamp"].nunique().eq(1).all()


def test_variant_d_is_exact_intersection_and_e_fails_unavailable() -> None:
    decisions = build_variant_decisions(_variant_source())
    d = decisions.loc[decisions["variant"].eq("D_anchor_veto_plus_price_acceptance")]
    e = decisions.loc[decisions["variant"].eq("E_anchor_veto_plus_price_acceptance_plus_range")]

    assert d.set_index("opportunity_id")["admitted"].to_dict() == {"one": True, "two": False}
    assert e["decision"].eq("unavailable").all()


def test_rejections_do_not_replace_or_refill_opportunities() -> None:
    decisions = build_variant_decisions(_variant_source())

    assert decisions["replacement_opportunity_id"].isna().all()
    assert not decisions["overlap_or_capacity_refilled"].any()
    assert decisions["existing_position_action"].eq("unchanged").all()
