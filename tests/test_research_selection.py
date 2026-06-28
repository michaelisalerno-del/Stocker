import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.experiments import run_research_experiment
from stocker_research.selection import select_parameter_set


def _grid_row(
    parameter_set_id: str,
    *,
    train_net_return: float,
    test_net_return: float,
    train_trade_count: int = 30,
    train_max_drawdown: float = -0.05,
    train_profitable_split_pct: float = 0.75,
) -> dict[str, object]:
    return {
        "parameter_set_id": parameter_set_id,
        "params": {"window": int(parameter_set_id[-1])},
        "train_gross_return": train_net_return + 0.01,
        "train_net_return": train_net_return,
        "train_profitable_split_pct": train_profitable_split_pct,
        "train_trade_count": train_trade_count,
        "train_max_drawdown": train_max_drawdown,
        "test_gross_return": test_net_return + 0.01,
        "test_net_return": test_net_return,
        "test_profitable_split_pct": 0.75,
        "test_trade_count": 20,
        "test_max_drawdown": -0.07,
    }


def test_selection_uses_train_evidence_not_luckiest_test_return() -> None:
    grid_results = [
        _grid_row(
            "ps_0001",
            train_net_return=-0.04,
            test_net_return=0.40,
            train_trade_count=8,
            train_profitable_split_pct=0.25,
        ),
        _grid_row("ps_0002", train_net_return=0.05, test_net_return=0.08),
        _grid_row("ps_0003", train_net_return=0.04, test_net_return=0.06),
    ]

    result = select_parameter_set(
        grid_results,
        minimum_train_trades=20,
        max_train_drawdown=-0.25,
    )

    assert result.selected_parameter_set_id == "ps_0002"
    assert result.selected_result["parameter_set_id"] == "ps_0002"
    assert result.selection_method == "train_gated"
    assert "ps_0001" in result.rejected_parameter_set_ids
    assert result.diagnostics["best_test_parameter_set_id"] == "ps_0001"
    assert not result.diagnostics["fallback_for_reporting_only"]


def test_selection_falls_back_deterministically_when_train_gates_fail() -> None:
    grid_results = [
        _grid_row("ps_0002", train_net_return=-0.02, test_net_return=0.10, train_trade_count=2),
        _grid_row("ps_0001", train_net_return=-0.03, test_net_return=0.05, train_trade_count=3),
    ]

    result = select_parameter_set(
        grid_results,
        minimum_train_trades=20,
        max_train_drawdown=-0.25,
    )

    assert result.selected_parameter_set_id == "ps_0001"
    assert result.selection_method == "fallback_for_reporting_only"
    assert set(result.rejected_parameter_set_ids) == {"ps_0001", "ps_0002"}
    assert result.diagnostics["fallback_for_reporting_only"]


def _sample_frame(symbol: str = "AAPL.US", rows: int = 48) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B", tz="UTC")
    close = pd.Series([100 + idx * 0.25 + ((idx % 7) - 3) * 0.1 for idx in range(rows)])
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


def _write_small_hypothesis(tmp_path: Path) -> Path:
    path = tmp_path / "hypothesis.yaml"
    path.write_text(
        json.dumps(
            {
                "id": "selection_test_hypothesis",
                "name": "Selection Test Hypothesis",
                "description": "Small deterministic hypothesis for selection integration tests.",
                "hypothesis_version": 1,
                "market_universe": "unit_test",
                "instrument_type": "stock",
                "timeframe": "1d",
                "data_source": "manual",
                "template": "moving_average_momentum",
                "direction": "long_only",
                "entry_logic": "Fast moving average above slow moving average.",
                "exit_logic": "Flat when fast moving average is no longer above slow average.",
                "holding_period": "Signal persistence only.",
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


def test_runner_reports_train_selected_result_separately_from_best_test_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, _sample_frame())
    grid_results = [
        _grid_row(
            "ps_0001",
            train_net_return=-0.03,
            test_net_return=0.40,
            train_trade_count=5,
            train_profitable_split_pct=0.25,
        ),
        {
            **_grid_row("ps_0002", train_net_return=0.05, test_net_return=-0.01),
            "params": {"fast_window": 2, "holding_period": 1, "slow_window": 5},
        },
        {
            **_grid_row("ps_0003", train_net_return=0.04, test_net_return=0.02),
            "params": {"fast_window": 2, "holding_period": 1, "slow_window": 5},
        },
    ]

    def fake_run_grid(*_args: Any, **_kwargs: Any) -> list[dict[str, object]]:
        return grid_results

    monkeypatch.setattr("stocker_research.experiments._run_grid", fake_run_grid)

    result = run_research_experiment(
        hypothesis_path=_write_small_hypothesis(tmp_path),
        data_dir=data_dir,
        symbol="AAPL.US",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
    )
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert payload["selection"]["selected_parameter_set_id"] == "ps_0002"
    assert payload["selected_result"]["parameter_set_id"] == "ps_0002"
    assert payload["selected_result"]["test_net_return"] == -0.01
    assert payload["best_test_diagnostic"]["parameter_set_id"] == "ps_0001"
    assert result.classification == "rejected_no_edge"
