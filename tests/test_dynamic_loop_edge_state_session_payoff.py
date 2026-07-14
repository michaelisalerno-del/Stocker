from __future__ import annotations

import pandas as pd

from stocker_research.dynamic_loop_edge_state.session_payoff import (
    AggregationSettings,
    aggregate_session_payoffs,
    settled_before,
)


def _trade(
    *,
    stock: str = "AAA",
    session: str = "2025-01-02",
    gross_bps: float = 30.0,
    settlement: str = "2025-01-02T17:00:00Z",
    fill_id: str = "fill-1",
    **costs: float,
) -> dict[str, object]:
    values: dict[str, object] = {
        "session": session,
        "loop_id": "cycle_04",
        "orientation": "state_4",
        "horizon": 24,
        "stock_id": stock,
        "fill_id": fill_id,
        "decision_timestamp": pd.Timestamp(f"{session}T14:35:00Z"),
        "entry_timestamp": pd.Timestamp(f"{session}T14:35:00Z"),
        "exit_timestamp": pd.Timestamp(settlement),
        "settlement_timestamp": pd.Timestamp(settlement),
        "feature_availability_timestamp": pd.Timestamp(f"{session}T14:35:00Z"),
        "gross_payoff_bps": gross_bps,
        "entry_cost_bps": 5.0,
        "exit_cost_bps": 5.0,
        "spread_cost_bps": 0.0,
        "slippage_cost_bps": 0.0,
        "commission_cost_bps": 0.0,
        "financing_cost_bps": 0.0,
        "fx_cost_bps": 0.0,
        "other_cost_bps": 0.0,
        "structural_breadth": 0.25,
        "top_loop_score": 0.4,
        "top_second_margin": 0.1,
        "loop_score_entropy": 0.5,
        "transition_surprise": 0.2,
        "market_return": 0.01,
        "market_volatility": 0.02,
        "liquidity_pressure": 0.3,
    }
    values.update(costs)
    return values


def test_multiple_fills_from_one_stock_count_as_one_independent_contribution() -> None:
    trades = pd.DataFrame(
        [
            _trade(stock="AAA", fill_id="a1", gross_bps=30.0),
            _trade(stock="AAA", fill_id="a2", gross_bps=50.0),
            _trade(stock="AAA", fill_id="a3", gross_bps=70.0),
            _trade(stock="BBB", fill_id="b1", gross_bps=50.0),
        ]
    )

    panel = aggregate_session_payoffs(trades, AggregationSettings())

    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["raw_fill_count"] == 4
    assert row["independent_stock_count"] == 2
    assert row["effective_sample_size"] == 2.0
    assert row["robust_gross_payoff_bps"] == 50.0
    assert row["robust_net_payoff_bps"] == 40.0


def test_no_opportunity_is_missing_evidence_not_a_zero_observation() -> None:
    empty = pd.DataFrame(columns=["session", "loop_id", "orientation", "horizon"])

    panel = aggregate_session_payoffs(empty, AggregationSettings())

    assert panel.empty
    assert "robust_net_payoff_bps" in panel.columns


def test_all_applicable_cost_components_are_included_in_net_payoff() -> None:
    trade = _trade(
        gross_bps=100.0,
        entry_cost_bps=1.0,
        exit_cost_bps=2.0,
        spread_cost_bps=3.0,
        slippage_cost_bps=4.0,
        commission_cost_bps=5.0,
        financing_cost_bps=6.0,
        fx_cost_bps=7.0,
        other_cost_bps=8.0,
    )

    panel = aggregate_session_payoffs(pd.DataFrame([trade]), AggregationSettings())

    row = panel.iloc[0]
    assert row["cost_contribution_bps"] == 36.0
    assert row["robust_net_payoff_bps"] == 64.0


def test_only_outcomes_settled_strictly_before_the_decision_are_visible() -> None:
    trades = pd.DataFrame(
        [
            _trade(fill_id="past", settlement="2025-01-02T16:00:00Z"),
            _trade(fill_id="equal", settlement="2025-01-03T14:30:00Z"),
            _trade(fill_id="future", settlement="2025-01-03T15:00:00Z"),
        ]
    )

    visible = settled_before(trades, pd.Timestamp("2025-01-03T14:30:00Z"))

    assert visible["fill_id"].tolist() == ["past"]


def test_session_boundaries_are_distinct_statistical_units() -> None:
    trades = pd.DataFrame(
        [
            _trade(session="2025-01-02", settlement="2025-01-02T17:00:00Z"),
            _trade(session="2025-01-03", settlement="2025-01-03T17:00:00Z"),
        ]
    )

    panel = aggregate_session_payoffs(trades, AggregationSettings())

    assert panel["session"].tolist() == ["2025-01-02", "2025-01-03"]


def test_independent_stocks_supply_more_evidence_than_correlated_fill_cluster() -> None:
    clustered = pd.DataFrame([_trade(stock="AAA", fill_id=f"a{index}") for index in range(5)])
    broad = pd.DataFrame([_trade(stock=f"S{index}", fill_id=f"s{index}") for index in range(5)])

    clustered_row = aggregate_session_payoffs(clustered, AggregationSettings()).iloc[0]
    broad_row = aggregate_session_payoffs(broad, AggregationSettings()).iloc[0]

    assert clustered_row["raw_fill_count"] == broad_row["raw_fill_count"] == 5
    assert clustered_row["independent_stock_count"] == 1
    assert broad_row["independent_stock_count"] == 5
    assert clustered_row["effective_sample_size"] == 1.0
    assert broad_row["effective_sample_size"] == 5.0
