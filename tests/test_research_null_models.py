import pandas as pd

from stocker_backtest.costs import CostModel
from stocker_research.null_models import run_null_timing_test


def test_null_timing_results_are_deterministic_and_gate_against_p75() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
            "open": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
            "close": [100 + i for i in range(20)],
            "volume": [1000] * 20,
        }
    )
    positions = pd.Series([1.0 if index % 2 == 0 else 0.0 for index in range(len(frame))])

    first = run_null_timing_test(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        hypothesis_id="hypothesis_a",
        symbol="AAPL.US",
        timeframe="1d",
        parameter_set_id="ps_0001",
        selected_net_return=-0.01,
        null_count=7,
    )
    second = run_null_timing_test(
        frame,
        positions,
        cost_model=CostModel(spread_bps=0.0, commission_bps=0.0, slippage_bps=0.0),
        hypothesis_id="hypothesis_a",
        symbol="AAPL.US",
        timeframe="1d",
        parameter_set_id="ps_0001",
        selected_net_return=-0.01,
        null_count=7,
    )

    assert first == second
    assert first["count"] == 7
    assert "p75_null_net_return" in first
    assert "p90_null_net_return" in first
    assert first["selected_excess_vs_p75_null"] < 0.0
    assert first["null_pass"] is False
