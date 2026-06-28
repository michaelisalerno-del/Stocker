import json
from pathlib import Path

import pandas as pd

from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.experiments import run_research_experiment
from stocker_research.leakage import collect_research_leakage_issues
from stocker_research.walkforward import WalkForwardSplit


def _sample_frame(symbol: str = "AAPL.US", rows: int = 48) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B", tz="UTC")
    close = pd.Series([100 + idx * 0.3 + ((idx % 5) - 2) * 0.1 for idx in range(rows)])
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
                "id": "leakage_test_hypothesis",
                "name": "Leakage Test Hypothesis",
                "description": "Small deterministic hypothesis for leakage integration tests.",
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


def test_duplicate_timestamp_leakage_issue_rejects_experiment(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    frame = _sample_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    _write_dataset(data_dir, frame)

    result = run_research_experiment(
        hypothesis_path=_write_small_hypothesis(tmp_path),
        data_dir=data_dir,
        symbol="AAPL.US",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    markdown = result.markdown_path.read_text(encoding="utf-8")

    assert result.classification == "rejected_data_issue"
    assert any(issue["code"] == "duplicate_timestamps" for issue in payload["leakage_issues"])
    assert "leakage_errors" in payload["classification_reasons"]
    assert "## Leakage Checks" in markdown
    assert "duplicate_timestamps" in markdown


def test_non_monotonic_raw_timestamps_are_reported_before_sorting(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    frame = _sample_frame()
    frame = frame.iloc[[0, 2, 1, *range(3, len(frame))]].reset_index(drop=True)
    _write_dataset(data_dir, frame)

    result = run_research_experiment(
        hypothesis_path=_write_small_hypothesis(tmp_path),
        data_dir=data_dir,
        symbol="AAPL.US",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert result.classification == "rejected_data_issue"
    assert any(issue["code"] == "non_monotonic_timestamps" for issue in payload["leakage_issues"])
    assert "leakage_errors" in payload["classification_reasons"]


def test_clean_fixture_has_no_leakage_errors_and_serializes_checks(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, _sample_frame())

    result = run_research_experiment(
        hypothesis_path=_write_small_hypothesis(tmp_path),
        data_dir=data_dir,
        symbol="AAPL.US",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    markdown = result.markdown_path.read_text(encoding="utf-8")

    assert not [issue for issue in payload["leakage_issues"] if issue["severity"] == "error"]
    assert "## Leakage Checks" in markdown


def test_invalid_split_and_embargo_violation_are_reported() -> None:
    frame = _sample_frame(rows=20)
    bad_split = WalkForwardSplit(
        split_id="split_bad",
        train_start=0,
        train_end=12,
        test_start=10,
        test_end=18,
    )
    signals = pd.DataFrame(
        {
            "signal": [0.0] * len(frame),
            "target_position": [0.0] * len(frame),
        }
    )

    issues = collect_research_leakage_issues(
        frame=frame,
        splits=[bad_split],
        signals=signals,
        embargo_bars=3,
    )
    issue_codes = {issue.code for issue in issues}

    assert "train_test_overlap" in issue_codes
    assert "embargo_violation" in issue_codes
