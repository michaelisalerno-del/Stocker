from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.fixed_one_bar_entry_latency import score_fixed_latency

ANCHOR = pd.Timestamp("2025-06-02 14:30:00+00:00")
T0 = ANCHOR + pd.Timedelta(minutes=15)
TERMINAL = ANCHOR + pd.Timedelta(minutes=125)


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": ANCHOR + pd.Timedelta(minutes=minute),
                "open": 100.0 + minute / 100.0,
                "high": 101.0 + minute / 100.0,
                "low": 99.0 + minute / 100.0,
                "close": 100.5 + minute / 100.0,
            }
            for minute in range(0, 260, 5)
        ]
    )


def _score(
    bars: pd.DataFrame | None = None,
    *,
    direction: int = 1,
    latency_bars: int = 1,
    t0: pd.Timestamp = T0,
    entry_step: int = 3,
    t0_price: float = 100.15,
    terminal: pd.Timestamp = TERMINAL,
    terminal_price: float = 101.7,
    cost_bps_per_side: float = 5.0,
):
    gross = 10_000.0 * direction * (terminal_price / t0_price - 1.0)
    return score_fixed_latency(
        _bars() if bars is None else bars,
        anchor_timestamp=ANCHOR,
        entry_step=entry_step,
        t0_entry_timestamp=t0,
        t0_entry_price=t0_price,
        original_terminal_timestamp=terminal,
        original_terminal_price=terminal_price,
        direction=direction,
        source_t0_gross_return_bps=gross,
        source_t0_net_return_bps=gross - 2.0 * cost_bps_per_side,
        latency_bars=latency_bars,
        cost_bps_per_side=cost_bps_per_side,
    )


def test_t1_is_exactly_one_completed_bar_after_frozen_t0() -> None:
    result = score_fixed_latency(
        _bars(),
        anchor_timestamp=ANCHOR,
        entry_step=3,
        t0_entry_timestamp=T0,
        t0_entry_price=100.15,
        original_terminal_timestamp=TERMINAL,
        original_terminal_price=101.7,
        direction=1,
        source_t0_gross_return_bps=154.767848227658,
        source_t0_net_return_bps=144.767848227658,
    )

    assert result.status == "available"
    assert result.t0_entry_timestamp == T0
    assert result.t1_entry_timestamp == T0 + pd.Timedelta(minutes=5)
    assert result.t1_entry_price == pytest.approx(100.2)
    assert result.original_terminal_timestamp == TERMINAL


def test_t0_uses_exact_stored_timestamp_and_rejects_reconstruction_drift() -> None:
    result = _score(t0=T0 + pd.Timedelta(minutes=5))

    assert result.status == "t0_timestamp_mismatch"
    assert result.t0_entry_timestamp == T0 + pd.Timedelta(minutes=5)


def test_t1_selection_uses_timestamp_not_row_position() -> None:
    shuffled = _bars().sample(frac=1.0, random_state=17).reset_index(drop=True)

    result = _score(shuffled)

    assert result.status == "available"
    assert result.t1_entry_timestamp == T0 + pd.Timedelta(minutes=5)
    assert result.t1_entry_price == pytest.approx(100.2)


def test_missing_t1_bar_does_not_shift_later_open() -> None:
    bars = _bars().loc[lambda frame: frame["timestamp"].ne(T0 + pd.Timedelta(minutes=5))]

    result = _score(bars)

    assert result.status == "missing_exact_t1_open"
    assert result.t1_entry_timestamp is None
    assert result.t1_net_return_bps is None


def test_t1_at_terminal_is_unavailable_not_zero() -> None:
    result = _score(
        t0=ANCHOR + pd.Timedelta(minutes=120),
        entry_step=24,
        t0_price=101.2,
    )

    assert result.status == "latency_entry_too_late"
    assert result.t1_expected_timestamp == TERMINAL
    assert result.t1_net_return_bps is None


def test_primary_and_restarted_h24_terminals_are_separate() -> None:
    result = _score()

    assert result.original_terminal_timestamp == TERMINAL
    assert result.restarted_exit_timestamp == T0 + pd.Timedelta(minutes=125)
    assert result.restarted_exit_timestamp != result.original_terminal_timestamp
    assert result.restarted_net_return_bps != result.t1_net_return_bps


def test_entry_and_exit_costs_are_charged_to_both_timings() -> None:
    result = _score()

    assert result.t0_total_cost_bps == 10.0
    assert result.t1_total_cost_bps == 10.0
    assert result.t0_net_return_bps == pytest.approx(result.t0_gross_return_bps - 10.0)  # type: ignore[operator]
    assert result.t1_net_return_bps == pytest.approx(result.t1_gross_return_bps - 10.0)  # type: ignore[operator]


def test_twice_cost_stress_changes_only_cost_levels_not_paired_delta() -> None:
    base = _score()
    stressed = _score(cost_bps_per_side=10.0)

    assert stressed.t0_gross_return_bps == base.t0_gross_return_bps
    assert stressed.t1_gross_return_bps == base.t1_gross_return_bps
    assert stressed.t0_net_return_bps == pytest.approx(base.t0_net_return_bps - 10.0)  # type: ignore[operator]
    assert stressed.t1_net_return_bps == pytest.approx(base.t1_net_return_bps - 10.0)  # type: ignore[operator]
    assert stressed.paired_difference_bps == pytest.approx(base.paired_difference_bps)  # type: ignore[arg-type]


def test_long_payoff_and_entry_move_reconcile_exactly() -> None:
    result = _score()

    assert result.t0_gross_return_bps == pytest.approx(154.767848227658)
    assert result.t1_gross_return_bps == pytest.approx(149.700598802395)
    assert result.direction_adjusted_entry_move_bps == pytest.approx(4.9925112331497)
    assert result.paired_difference_bps == pytest.approx(-5.067249425263)
    assert result.paired_difference_bps == pytest.approx(result.exact_entry_price_effect_bps)  # type: ignore[arg-type]
    assert result.reconciliation_error_bps == pytest.approx(0.0, abs=1e-10)


def test_short_payoff_and_entry_move_reconcile_exactly() -> None:
    result = _score(direction=-1)

    assert result.t0_gross_return_bps == pytest.approx(-154.767848227658)
    assert result.t1_gross_return_bps == pytest.approx(-149.700598802395)
    assert result.direction_adjusted_entry_move_bps == pytest.approx(-4.9925112331497)
    assert result.paired_difference_bps == pytest.approx(5.067249425263)
    assert result.reconciliation_error_bps == pytest.approx(0.0, abs=1e-10)


def test_adverse_first_bar_improves_delayed_long_entry() -> None:
    bars = _bars()
    bars.loc[bars["timestamp"].eq(T0 + pd.Timedelta(minutes=5)), "open"] = 99.5

    result = _score(bars)

    assert result.direction_adjusted_entry_move_bps < 0.0  # type: ignore[operator]
    assert result.paired_difference_bps > 0.0  # type: ignore[operator]


def test_favourable_first_bar_harms_delayed_long_entry() -> None:
    bars = _bars()
    bars.loc[bars["timestamp"].eq(T0 + pd.Timedelta(minutes=5)), "open"] = 101.0

    result = _score(bars)

    assert result.direction_adjusted_entry_move_bps > 0.0  # type: ignore[operator]
    assert result.paired_difference_bps < 0.0  # type: ignore[operator]


def test_symmetric_short_examples_reverse_the_entry_effect() -> None:
    adverse = _bars()
    adverse.loc[adverse["timestamp"].eq(T0 + pd.Timedelta(minutes=5)), "open"] = 101.0
    favourable = _bars()
    favourable.loc[favourable["timestamp"].eq(T0 + pd.Timedelta(minutes=5)), "open"] = 99.5

    improved = _score(adverse, direction=-1)
    harmed = _score(favourable, direction=-1)

    assert improved.direction_adjusted_entry_move_bps < 0.0  # type: ignore[operator]
    assert improved.paired_difference_bps > 0.0  # type: ignore[operator]
    assert harmed.direction_adjusted_entry_move_bps > 0.0  # type: ignore[operator]
    assert harmed.paired_difference_bps < 0.0  # type: ignore[operator]


def test_unchanged_entry_price_has_zero_delta_before_cost_variation() -> None:
    bars = _bars()
    bars.loc[bars["timestamp"].eq(T0 + pd.Timedelta(minutes=5)), "open"] = 100.15

    result = _score(bars)

    assert result.direction_adjusted_entry_move_bps == pytest.approx(0.0)
    assert result.paired_difference_bps == pytest.approx(0.0)


def test_appending_bars_after_terminal_does_not_change_primary_result() -> None:
    base = _score()
    future = pd.concat(
        [
            _bars(),
            pd.DataFrame(
                [
                    {
                        "timestamp": TERMINAL + pd.Timedelta(days=1),
                        "open": 1.0,
                        "high": 2.0,
                        "low": 0.5,
                        "close": 1.5,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    appended = _score(future)

    assert appended.t1_entry_price == base.t1_entry_price
    assert appended.t1_net_return_bps == base.t1_net_return_bps
    assert appended.paired_difference_bps == base.paired_difference_bps
