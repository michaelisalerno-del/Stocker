from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "research"
    / "realized_m_20_ibkr_fast_v0"
    / "run_experiment.py"
)
SPEC = importlib.util.spec_from_file_location("realized_m_20_ibkr_fast_v0", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_experiment
SPEC.loader.exec_module(run_experiment)

MINIMUM_VALID = run_experiment.MINIMUM_VALID
calculate_realized_m_20 = run_experiment.calculate_realized_m_20
normalize_prices = run_experiment.normalize_prices


def prices(sessions: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2025, 1, 2, 15, 0, tzinfo=UTC)
    session_count = 0
    day = start
    while session_count < sessions:
        if day.weekday() < 5:
            for minute in range(18):
                timestamp = day + timedelta(minutes=minute)
                rows.append(
                    {
                        "timestamp": timestamp,
                        "open": 100.0,
                        "high": 102.0,
                        "low": 99.0,
                        "close": 101.0 if minute == 14 else 100.0,
                    }
                )
            session_count += 1
        day += timedelta(days=1)
    day -= timedelta(minutes=3)
    for minute in range(18):
        rows.append(
            {
                "timestamp": day,
                "open": 99.0 if minute == 0 else 100.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
            }
        )
        day += timedelta(minutes=1)
    return normalize_prices(pd.DataFrame(rows))


def test_frozen_realized_m_uses_20_prior_same_minute_plus_14_bar_returns() -> None:
    source = prices()
    signal = pd.Timestamp(source.index.max()) - pd.Timedelta(minutes=14)

    result = calculate_realized_m_20(source, signal_timestamp=signal, current_p0=100.0)

    assert result.valid_prior_sessions == 20
    assert result.realized_return == pytest.approx(0.01)
    assert result.realized_price == pytest.approx(1.0)
    assert result.pre_move_m == pytest.approx(1.0)


def test_frozen_realized_m_requires_at_least_10_prior_valid_sessions() -> None:
    source = prices(MINIMUM_VALID - 1)
    signal = pd.Timestamp(source.index.max()) - pd.Timedelta(minutes=14)

    with pytest.raises(ValueError, match="valid prior sessions"):
        calculate_realized_m_20(source, signal_timestamp=signal, current_p0=100.0)


def test_research_harness_exposes_no_order_path() -> None:
    assert run_experiment.RESEARCH_ONLY is True
    assert run_experiment.ORDER_PLACEMENT == "disabled"
    source = Path(run_experiment.__file__).read_text(encoding="utf-8")
    assert "placeOrder" not in source
    assert "submit_order" not in source
