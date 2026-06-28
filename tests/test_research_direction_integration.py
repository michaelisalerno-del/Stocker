import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.experiments import run_research_experiment
from stocker_research.templates.base import StrategyTemplate


class AlwaysShortTemplate(StrategyTemplate):
    name = "always_short"

    def generate_positions(self, frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        return pd.Series([-1.0] * len(frame))


def _falling_frame(symbol: str = "AAPL.US", rows: int = 48) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B", tz="UTC")
    close = pd.Series([100.0 - idx * 0.5 for idx in range(rows)])
    return pd.DataFrame(
        {
            "source": "manual",
            "symbol": symbol,
            "instrument_type": "stock",
            "timeframe": "1d",
            "timestamp": dates,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000,
            "currency": "USD",
            "timezone": "UTC",
        }
    )


def _write_dataset(data_dir: Path, frame: pd.DataFrame, symbol: str = "AAPL.US") -> None:
    key = DatasetKey(source="manual", instrument_type="stock", symbol=symbol, timeframe="1d")
    write_parquet(frame, dataset_path(key, data_dir=data_dir))


def _write_short_hypothesis(tmp_path: Path) -> Path:
    path = tmp_path / "short_hypothesis.yaml"
    path.write_text(
        json.dumps(
            {
                "id": "short_direction_test_hypothesis",
                "name": "Short Direction Test Hypothesis",
                "description": "Small deterministic hypothesis for direction integration tests.",
                "hypothesis_version": 1,
                "market_universe": "unit_test",
                "instrument_type": "stock",
                "timeframe": "1d",
                "data_source": "manual",
                "template": "moving_average_momentum",
                "direction": "short_only",
                "entry_logic": "Synthetic short target position.",
                "exit_logic": "Synthetic short target position remains open.",
                "holding_period": "Synthetic fixture.",
                "expected_edge_reason": "Unit test fixture.",
                "invalidation_rules": ["Unit test invalidation."],
                "minimum_evidence": {
                    "min_trades": 0,
                    "min_profitable_split_pct": 0.0,
                    "min_stability_score": 0.0,
                },
                "parameter_space": {
                    "fast_window": [2],
                    "holding_period": [1],
                    "slow_window": [5],
                },
                "maximum_parameter_sets": 1,
                "costs": {"spread_bps": 0.0, "commission_bps": 0.0, "slippage_bps": 0.0},
                "risk": {"max_drawdown": 0.25},
                "walkforward": {
                    "mode": "rolling",
                    "train_bars": 20,
                    "test_bars": 10,
                    "embargo_bars": 1,
                    "step_bars": 10,
                    "minimum_rows": 31,
                },
                "created_at": "2026-06-28T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_runner_uses_hypothesis_direction_for_selected_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, _falling_frame())
    monkeypatch.setattr(
        "stocker_research.experiments._template_for",
        lambda _hypothesis: AlwaysShortTemplate(),
    )

    result = run_research_experiment(
        hypothesis_path=_write_short_hypothesis(tmp_path),
        data_dir=data_dir,
        symbol="AAPL.US",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert payload["full_sample_result"]["net_return"] > 0.0
    assert payload["benchmark_policy"] == "long_buy_and_hold_market_baseline"
    assert payload["benchmark_results"]["buy_and_hold"]["net_return"] < 0.0
    assert payload["null_model_results"]["direction"] == "short_only"
