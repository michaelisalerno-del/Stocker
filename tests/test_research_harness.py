import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from stocker_backtest.costs import CostModel
from stocker_backtest.vectorized import evaluate_positions
from stocker_core.cli import app
from stocker_data.ingest import import_csv
from stocker_research.experiments import classify_experiment, run_research_experiment
from stocker_research.hypothesis import Hypothesis, load_hypothesis
from stocker_research.leakage import (
    check_embargo_violation,
    check_feature_target_overlap,
    check_same_bar_close_signal,
    check_timestamp_integrity,
    check_train_test_overlap,
)
from stocker_research.parameters import generate_parameter_grid
from stocker_research.regime import label_regimes, performance_by_regime
from stocker_research.stability import analyze_stability
from stocker_research.templates import (
    MeanReversionTemplate,
    MovingAverageMomentumTemplate,
    VolatilityBreakoutTemplate,
)
from stocker_research.walkforward import WalkForwardConfig, generate_walk_forward_splits


def _sample_ohlcv(rows: int = 72) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B", tz="America/New_York")
    close = pd.Series([100 + i * 0.2 + ((i % 7) - 3) * 0.15 for i in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1000 + i for i in range(rows)],
        }
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    export = frame.copy()
    export["timestamp"] = export["timestamp"].astype(str)
    export.to_csv(path, index=False)
    return path


def test_hypothesis_yaml_loading_and_invalid_rejection() -> None:
    hypothesis = load_hypothesis("research/hypotheses/examples/moving_average_momentum.yaml")

    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.template == "moving_average_momentum"
    assert hypothesis.signal_family == "moving_average_momentum"
    assert hypothesis.cost_model.round_trip_bps() == 7.0

    invalid = hypothesis.model_dump()
    invalid["parameter_space"] = {}
    try:
        Hypothesis.model_validate(invalid)
    except ValueError as exc:
        assert "parameter_space" in str(exc)
    else:
        raise AssertionError("invalid hypothesis should be rejected")


def test_walk_forward_splits_are_ordered_deterministic_and_embargoed() -> None:
    frame = _sample_ohlcv(50)
    config = WalkForwardConfig(
        mode="rolling",
        train_size=20,
        test_size=10,
        step_size=10,
        embargo_bars=2,
        min_rows=30,
    )

    first = generate_walk_forward_splits(frame, config)
    second = generate_walk_forward_splits(frame, config)

    assert first == second
    assert first[0].split_id == "split_001"
    assert first[0].train_end < first[0].test_start
    assert first[0].test_start - first[0].train_end == 2
    assert all(split.train_end <= split.test_start for split in first)


def test_expanding_walk_forward_uses_no_future_training_rows() -> None:
    frame = _sample_ohlcv(60)
    splits = generate_walk_forward_splits(
        frame,
        WalkForwardConfig(
            mode="expanding",
            train_size=20,
            test_size=10,
            step_size=10,
            embargo_bars=1,
            min_rows=30,
        ),
    )

    assert splits[1].train_start == 0
    assert splits[1].train_end > splits[0].train_end
    assert all(split.train_end < split.test_start for split in splits)


def test_parameter_grid_guardrail_ids_and_stability_scoring() -> None:
    grid = generate_parameter_grid({"fast": [2, 3], "slow": [5, 8]}, max_size=8)

    assert [item.parameter_set_id for item in grid] == ["ps_0001", "ps_0002", "ps_0003", "ps_0004"]

    try:
        generate_parameter_grid({"x": list(range(20)), "y": list(range(20))}, max_size=100)
    except ValueError as exc:
        assert "exceeds max_size" in str(exc)
    else:
        raise AssertionError("large grid should be rejected")

    stability = analyze_stability(
        [
            {
                "parameter_set_id": "ps_0001",
                "params": {"fast": 2},
                "test_net_return": 0.03,
                "train_net_return": 0.05,
            },
            {
                "parameter_set_id": "ps_0002",
                "params": {"fast": 3},
                "test_net_return": 0.02,
                "train_net_return": 0.04,
            },
            {
                "parameter_set_id": "ps_0003",
                "params": {"fast": 4},
                "test_net_return": -0.01,
                "train_net_return": 0.03,
            },
        ],
        best_parameter_set_id="ps_0001",
    )

    assert stability.best_parameter_set_id == "ps_0001"
    assert stability.profitable_neighbour_pct > 0
    assert stability.stability_score > 0


def test_templates_generate_deterministic_positions() -> None:
    frame = _sample_ohlcv(30)

    ma = MovingAverageMomentumTemplate().generate_positions(
        frame, {"fast_window": 2, "slow_window": 5}
    )
    mr = MeanReversionTemplate().generate_positions(
        frame, {"down_threshold": -0.001, "hold_bars": 2}
    )
    bo = VolatilityBreakoutTemplate().generate_positions(
        frame, {"lookback": 5, "range_multiplier": 0.25}
    )

    assert len(ma) == len(frame)
    assert ma.equals(
        MovingAverageMomentumTemplate().generate_positions(
            frame, {"fast_window": 2, "slow_window": 5}
        )
    )
    assert set(mr.dropna().unique()).issubset({0.0, 1.0})
    assert set(bo.dropna().unique()).issubset({0.0, 1.0})


def test_vectorized_evaluation_applies_costs_and_outputs_curves() -> None:
    frame = _sample_ohlcv(30)
    positions = pd.Series([1.0 if i >= 5 else 0.0 for i in range(len(frame))])

    result = evaluate_positions(
        frame,
        positions,
        cost_model=CostModel(spread_bps=1.0, commission_bps=0.5, slippage_bps=0.5),
        initial_capital=100_000,
    )

    assert result.number_of_trades == 1
    assert result.total_costs > 0
    assert result.net_return <= result.gross_return
    assert len(result.equity_curve) == len(frame)
    assert len(result.drawdown_curve) == len(frame)
    assert isinstance(result.trades, list)


def test_leakage_checks_fail_loudly_for_suspicious_inputs() -> None:
    split = generate_walk_forward_splits(
        _sample_ohlcv(40),
        WalkForwardConfig(mode="rolling", train_size=20, test_size=10, embargo_bars=0),
    )[0]

    assert (
        check_same_bar_close_signal(uses_close=True, execution_lag_bars=0)[0].code
        == "same_bar_close"
    )
    assert (
        check_feature_target_overlap(["close", "future_return"], ["future_return"])[0].code
        == "target_in_features"
    )
    assert not check_train_test_overlap(split)

    bad_split = split.model_copy(update={"test_start": split.train_end - 1})
    assert check_train_test_overlap(bad_split)[0].code == "train_test_overlap"
    assert check_embargo_violation(split, embargo_bars=2)[0].code == "embargo_violation"

    duplicate_frame = _sample_ohlcv(5)
    duplicate_frame.loc[1, "timestamp"] = duplicate_frame.loc[0, "timestamp"]
    assert check_timestamp_integrity(duplicate_frame)[0].code == "duplicate_timestamps"

    reversed_frame = _sample_ohlcv(5).iloc[::-1].reset_index(drop=True)
    assert check_timestamp_integrity(reversed_frame)[0].code == "non_monotonic_timestamps"


def test_regime_labels_and_performance_by_regime_use_historical_windows() -> None:
    frame = _sample_ohlcv(40)
    labels = label_regimes(frame, window=5)

    assert {"volatility_regime", "trend_regime", "range_regime", "drawdown_regime"}.issubset(
        labels.columns
    )
    assert labels.iloc[:4]["volatility_regime"].eq("unknown").all()

    returns = pd.Series([0.001] * len(frame))
    summary = performance_by_regime(returns, labels["trend_regime"])
    assert "uptrend" in summary or "unknown" in summary


def test_experiment_runner_creates_reports_index_and_conservative_classification(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    csv_path = _write_csv(_sample_ohlcv(72), tmp_path / "sample.csv")
    import_csv(
        file_path=csv_path,
        data_dir=data_dir,
        symbol="AAPL",
        source="manual",
        timeframe="1d",
        instrument_type="stock",
        timezone="America/New_York",
        currency="USD",
    )

    result = run_research_experiment(
        hypothesis_path=Path("research/hypotheses/examples/moving_average_momentum.yaml"),
        data_dir=data_dir,
        symbol="AAPL",
        timeframe="1d",
        source="manual",
        instrument_type="stock",
    )

    assert result.json_path.exists()
    assert result.markdown_path.exists()
    assert result.classification in {
        "rejected_no_edge",
        "rejected_costs_kill_edge",
        "rejected_insufficient_data",
        "rejected_unstable_parameters",
        "rejected_walkforward_failure",
        "interesting_needs_more_tests",
        "candidate_paper_test",
    }
    index_payload = json.loads((data_dir / "reports" / "research" / "index.json").read_text())
    assert index_payload["experiments"][0]["experiment_id"] == result.experiment_id
    assert (
        classify_experiment(
            test_net_return=0.05,
            train_net_return=0.06,
            stability_score=0.8,
            profitable_split_pct=1.0,
            trade_count=50,
            max_drawdown=-0.05,
            regime_count=3,
            warnings=[],
        )
        == "candidate_paper_test"
    )


def test_research_run_cli_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    csv_path = _write_csv(_sample_ohlcv(72), tmp_path / "sample.csv")
    import_csv(
        file_path=csv_path,
        data_dir=data_dir,
        symbol="AAPL",
        source="manual",
        timeframe="1d",
        instrument_type="stock",
        timezone="America/New_York",
        currency="USD",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "research",
            "run",
            "--hypothesis",
            "research/hypotheses/examples/moving_average_momentum.yaml",
            "--symbol",
            "AAPL",
            "--timeframe",
            "1d",
            "--data-dir",
            str(data_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "classification" in result.output
