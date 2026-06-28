import pandas as pd

from stocker_research.baselines import ohlc_baseline_summary


def test_ohlc_baseline_summary_returns_basic_metrics_for_pandas() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 101.0],
            "volume": [1000, 1200, 900],
        }
    )

    summary = ohlc_baseline_summary(frame)

    assert summary["rows"] == 3
    assert summary["close_start"] == 101.0
    assert summary["close_end"] == 101.0
    assert "mean_return" in summary
    assert "volatility" in summary
