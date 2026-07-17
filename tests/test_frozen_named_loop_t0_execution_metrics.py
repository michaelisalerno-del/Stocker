from __future__ import annotations

import pandas as pd
import pytest

from stocker_research.frozen_named_loop_t0_execution import score_fill_envelope
from stocker_research.frozen_named_loop_t0_execution.metrics import (
    break_even_adverse_slippage_bps,
    concentration_summary,
    direction_flipped_diagnostic,
    family_metrics,
    leave_one_stock_out,
    named_control_comparisons,
    remove_top_contributors,
    session_block_bootstrap,
    session_block_break_even_bootstrap,
)


def payoff_frame(
    *, terminal_price: float = 101.0, named: int = 20, controls: int = 20
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    families = [
        ("cycle_04|state_4", "named", named),
        ("cycle_04|state_2", "control", controls),
    ]
    for family, classification, count in families:
        for index in range(count):
            for payoff in score_fill_envelope(
                opportunity_id=f"{family}-{index}",
                direction=1,
                reference_entry_price=100.0,
                terminal_timestamp=pd.Timestamp("2026-07-17T15:35:00Z"),
                terminal_price=terminal_price,
                cost_bps=10.0,
            ):
                rows.append(
                    {
                        **payoff.__dict__,
                        "family": family,
                        "classification": classification,
                        "symbol": f"S{index % 10}",
                        "session": f"2026-07-{1 + index % 15:02d}",
                        "month": "2026-07",
                        "hindsight_episode_id": f"episode-{index // 5}",
                    }
                )
    return pd.DataFrame(rows)


def test_true_positive_named_effect_survives_f10() -> None:
    metrics = family_metrics(payoff_frame(terminal_price=101.0))
    row = metrics.loc[
        metrics["family"].eq("cycle_04|state_4") & metrics["fill_model"].eq("F10")
    ].iloc[0]

    assert row["mean_net_payoff_bps"] > 0.0
    assert row["opportunities"] == 20


def test_false_reference_edge_disappears_under_f10() -> None:
    metrics = family_metrics(payoff_frame(terminal_price=100.15))
    named = metrics.loc[metrics["family"].eq("cycle_04|state_4")].set_index("fill_model")

    assert named.loc["F0", "mean_net_payoff_bps"] > 0.0
    assert named.loc["F10", "mean_net_payoff_bps"] < 0.0


def test_controls_match_named_candidates_under_a_null_process() -> None:
    comparisons = named_control_comparisons(payoff_frame(terminal_price=101.0))
    row = comparisons.loc[
        comparisons["comparison"].eq("cycle_04|state_4-minus-cycle_04|state_2")
        & comparisons["fill_model"].eq("F10")
    ].iloc[0]

    assert row["mean_difference_bps"] == pytest.approx(0.0)
    assert row["named_family"] == "cycle_04|state_4"
    assert row["control_family"] == "cycle_04|state_2"


def test_session_block_metrics_are_deterministic_under_fixed_seed() -> None:
    frame = payoff_frame().loc[lambda value: value["fill_model"].eq("F10")]

    first = session_block_bootstrap(frame, resamples=200, block_length=5, seed=20260717)
    second = session_block_bootstrap(frame, resamples=200, block_length=5, seed=20260717)

    assert first == second
    assert first["bootstrap_lower_95_bps"] <= first["observed_session_mean_bps"]
    assert first["bootstrap_upper_95_bps"] >= first["observed_session_mean_bps"]


def test_break_even_slippage_uses_exact_return_convention() -> None:
    frame = payoff_frame(named=5, controls=0).drop_duplicates("opportunity_id")

    result = break_even_adverse_slippage_bps(frame)

    assert result == pytest.approx(89.9100899, abs=1e-6)
    assert result > 10.0


def test_break_even_session_block_uncertainty_is_deterministic() -> None:
    frame = payoff_frame(named=20, controls=0).drop_duplicates("opportunity_id")

    first = session_block_break_even_bootstrap(frame, resamples=50, block_length=5, seed=20260717)
    second = session_block_break_even_bootstrap(frame, resamples=50, block_length=5, seed=20260717)

    assert first == second
    assert first["bootstrap_lower_95_bps"] <= first["break_even_adverse_slippage_bps"]
    assert first["bootstrap_upper_95_bps"] >= first["break_even_adverse_slippage_bps"]


def test_contribution_dominated_by_one_stock_is_classified_concentrated() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["dominant", "small-a", "small-b"],
            "net_payoff_bps": [1000.0, 10.0, -5.0],
        }
    )

    result = concentration_summary(frame, dimension="symbol")

    assert result["top_one_absolute_contribution_share"] > 0.95
    assert result["concentrated_or_unstable"] is True


def test_stock_removal_attribution_is_recomputed_not_subtracted_from_old_mean() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["best", "best", "other", "other"],
            "net_payoff_bps": [100.0, 100.0, -10.0, 30.0],
        }
    )

    result = remove_top_contributors(frame, dimension="symbol", top_n=1)

    assert result["removed"] == ["best"]
    assert result["remaining_opportunities"] == 2
    assert result["remaining_total_net_payoff_bps"] == pytest.approx(20.0)
    assert result["remaining_mean_net_payoff_bps"] == pytest.approx(10.0)


def test_leave_one_stock_out_recomputes_each_deterministic_fill_rule() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["a", "a", "b", "b"],
            "net_payoff_bps": [10.0, 20.0, -5.0, 5.0],
        }
    )

    result = leave_one_stock_out(frame).set_index("removed_symbol")

    assert result.loc["a", "remaining_mean_net_payoff_bps"] == pytest.approx(0.0)
    assert result.loc["b", "remaining_mean_net_payoff_bps"] == pytest.approx(15.0)
    assert result["attribution_method"].eq("deterministic_row_deletion_not_model_refit").all()


def test_direction_flipped_null_uses_the_same_entry_and_terminal() -> None:
    source = pd.DataFrame(
        {
            "opportunity_id": ["one"],
            "direction": [1],
            "reference_entry_price": [100.0],
            "terminal_timestamp": [pd.Timestamp("2026-07-17T15:35:00Z")],
            "terminal_price": [101.0],
            "cost_bps": [10.0],
        }
    )

    result = direction_flipped_diagnostic(source).iloc[0]

    assert result["opportunity_id"] == "one"
    assert result["direction"] == -1
    assert result["reference_entry_price"] == 100.0
    assert result["terminal_price"] == 101.0
    assert result["net_payoff_bps"] == pytest.approx(-110.0)
