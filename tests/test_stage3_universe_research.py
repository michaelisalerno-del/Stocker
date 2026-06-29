import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stocker_core.cli import app
from stocker_data.storage import DatasetKey, dataset_path, write_parquet
from stocker_research.classification import classify_research_result
from stocker_research.experiments import run_universe_research
from stocker_research.hypothesis import Hypothesis, load_hypothesis
from stocker_research.parameters import ParameterGrid
from stocker_research.templates import get_template


def _sample_frame(symbol: str, rows: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B", tz="UTC")
    close = pd.Series([100 + idx * 0.25 + ((idx % 9) - 4) * 0.12 for idx in range(rows)])
    return pd.DataFrame(
        {
            "source": "eodhd",
            "symbol": symbol,
            "instrument_type": "stock",
            "timeframe": "1d",
            "timestamp": dates,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 2_000_000,
            "currency": "USD",
            "timezone": "UTC",
        }
    )


def _write_dataset(data_dir: Path, symbol: str, rows: int = 90) -> None:
    key = DatasetKey(source="eodhd", instrument_type="stock", symbol=symbol, timeframe="1d")
    write_parquet(_sample_frame(symbol, rows=rows), dataset_path(key, data_dir=data_dir))


def test_stage3_hypothesis_contract_and_examples() -> None:
    hypothesis = load_hypothesis("research/hypotheses/examples/moving_average_momentum.yaml")

    assert hypothesis.template == "moving_average_momentum"
    assert hypothesis.hypothesis_version == 1
    assert hypothesis.maximum_parameter_sets == 100
    assert hypothesis.costs.round_trip_bps() == 7.0
    assert hypothesis.walkforward.train_bars == 500
    assert hypothesis.holding_policy.preferred_style == "intraday"
    assert hypothesis.holding_policy.allow_overnight == "conditional"
    assert hypothesis.holding_policy.allow_weekend == "exceptional_only"
    assert hypothesis.robustness_policy.require_cost_stress_for_intraday_candidate is True
    assert hypothesis.robustness_policy.cost_stress_candidate_multiplier == pytest.approx(1.5)
    assert hypothesis.robustness_policy.min_candidate_profit_factor == pytest.approx(1.10)

    payload = hypothesis.model_dump(mode="json")
    payload["expected_edge_reason"] = ""
    with pytest.raises(ValueError):
        Hypothesis.model_validate(payload)

    payload = hypothesis.model_dump(mode="json")
    payload["template"] = "unknown_template"
    with pytest.raises(ValueError):
        Hypothesis.model_validate(payload)

    payload = hypothesis.model_dump(mode="json")
    payload["costs"] = None
    with pytest.raises(ValueError):
        Hypothesis.model_validate(payload)

    for path in sorted(Path("research/hypotheses/examples").glob("*.yaml")):
        loaded = load_hypothesis(path)
        assert loaded.template
        assert loaded.costs.round_trip_bps() > 0
        assert loaded.walkforward.train_bars > loaded.walkforward.test_bars


def test_parameter_grid_class_has_stable_ids_and_guardrails() -> None:
    grid = ParameterGrid(
        parameter_space={"fast_window": [10, 20], "slow_window": [100, 150]},
        maximum_parameter_sets=10,
    )
    expanded = grid.expand()

    assert [item.parameter_set_id for item in expanded] == [
        "ps_0001",
        "ps_0002",
        "ps_0003",
        "ps_0004",
    ]
    assert expanded[0].params == {"fast_window": 10, "slow_window": 100}

    with pytest.raises(ValueError, match="exceeds"):
        ParameterGrid(
            parameter_space={"x": list(range(20)), "y": list(range(20))},
            maximum_parameter_sets=100,
        ).expand()

    with pytest.raises(ValueError, match="invalid"):
        ParameterGrid(parameter_space={"window": [-1]}, maximum_parameter_sets=10).expand()


def test_template_registry_and_pullback_shifted_signal_output() -> None:
    frame = _sample_frame("AAPL.US", rows=40)
    template = get_template("pullback_in_uptrend")

    signals = template.generate_signals(
        frame,
        {
            "trend_window": 10,
            "pullback_threshold": -0.005,
            "holding_period": 3,
            "parameter_set_id": "ps_test",
        },
    )

    assert {
        "timestamp",
        "signal",
        "target_position",
        "entry",
        "exit",
        "template_name",
        "parameter_set_id",
    }.issubset(signals.columns)
    assert signals["template_name"].unique().tolist() == ["pullback_in_uptrend"]
    assert signals["parameter_set_id"].unique().tolist() == ["ps_test"]
    assert (
        signals["target_position"]
        .shift(-1)
        .fillna(0)
        .equals(
            template.generate_positions(
                frame,
                {"trend_window": 10, "pullback_threshold": -0.005, "holding_period": 3},
            )
            .shift(-1)
            .fillna(0)
        )
    )

    with pytest.raises(ValueError):
        get_template("does_not_exist")


def test_classification_returns_reasons_and_is_conservative() -> None:
    no_edge = classify_research_result(
        net_test_return=-0.01,
        gross_test_return=0.02,
        trade_count=50,
        stability_score=0.8,
        profitable_split_pct=0.8,
        max_drawdown=-0.1,
        cost_drag=0.03,
        data_errors=0,
        leakage_errors=0,
    )
    assert no_edge.classification == "rejected_costs_kill_edge"
    assert "costs_kill_edge" in no_edge.reasons

    candidate = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.1,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.1,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
    )
    assert candidate.classification == "candidate_paper_test"

    intraday_candidate = classify_research_result(
        net_test_return=0.08,
        gross_test_return=0.1,
        trade_count=80,
        stability_score=0.75,
        profitable_split_pct=0.75,
        max_drawdown=-0.1,
        cost_drag=0.02,
        data_errors=0,
        leakage_errors=0,
        holding_policy_classification="candidate_intraday_test",
        holding_policy_reasons=["session_flat_compliant"],
    )
    assert intraday_candidate.classification == "interesting_intraday_needs_more_tests"
    assert "missing_trade_reconstruction" in intraday_candidate.reasons


def test_run_universe_research_writes_aggregate_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, "AAPL.US")
    _write_dataset(data_dir, "MSFT.US")
    ready = data_dir / "universes" / "research_ready" / "us_test_5_1d.json"
    ready.parent.mkdir(parents=True)
    ready.write_text(
        json.dumps(
            {
                "universe_id": "us_test_5",
                "timeframe": "1d",
                "source": "eodhd",
                "qualified_symbols": [{"symbol": "AAPL.US"}, {"symbol": "MSFT.US"}],
                "rejected_symbols": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_universe_research(
        hypothesis_path=Path("research/hypotheses/examples/moving_average_momentum.yaml"),
        qualified_universe_path=ready,
        data_dir=data_dir,
        source="eodhd",
        timeframe="1d",
        instrument_type="stock",
        max_symbols=2,
    )

    assert result.json_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["universe_id"] == "us_test_5"
    assert payload["symbol_count"] == 2
    assert payload["failed_count"] == 0
    assert payload["completed_count"] == 2
    assert payload["skipped_count"] == 0
    assert set(payload["classification_counts"])
    assert set(payload["classification_reason_counts"])
    assert payload["benchmark_pass_count"] <= 2
    assert payload["null_pass_count"] <= 2
    assert "intraday_candidate_count" in payload
    assert "swing_exceptional_candidate_count" in payload
    assert "holding_policy_rejection_count" in payload
    assert "overnight_violation_count" in payload
    assert "weekend_violation_count" in payload
    assert "median_gap_return_contribution_pct" in payload
    assert "median_overnight_exposure_count" in payload
    assert "median_weekend_exposure_count" in payload
    assert "median_excess_vs_benchmark" in payload
    assert "median_excess_vs_null" in payload
    assert len(payload["symbol_results"]) == 2
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert "Classification Counts" in markdown
    assert "Classification Reason Counts" in markdown
    assert "Completed:" in markdown
    assert "Skipped:" in markdown
    assert "Failed:" in markdown
    assert "Rejected:" in markdown
    assert "Candidate paper test:" in markdown
    assert "Holding Policy Summary" in markdown
    assert "Intraday candidates:" in markdown
    assert "Swing exceptional candidates:" in markdown
    assert (data_dir / "reports" / "research" / "index.json").exists()


def test_run_universe_research_keeps_going_after_symbol_failure(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, "AAPL.US")
    ready = data_dir / "universes" / "research_ready" / "us_test_5_1d.json"
    ready.parent.mkdir(parents=True)
    ready.write_text(
        json.dumps(
            {
                "universe_id": "us_test_5",
                "timeframe": "1d",
                "source": "eodhd",
                "qualified_symbols": [{"symbol": "AAPL.US"}, {"symbol": "MISSING.US"}],
                "rejected_symbols": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_universe_research(
        hypothesis_path=Path("research/hypotheses/examples/moving_average_momentum.yaml"),
        qualified_universe_path=ready,
        data_dir=data_dir,
        source="eodhd",
        timeframe="1d",
        instrument_type="stock",
        max_symbols=2,
        fail_fast=False,
    )

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))

    assert payload["failed_count"] == 1
    assert [item["status"] for item in payload["symbol_results"]] == ["completed", "failed"]


def test_run_universe_cli_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, "AAPL.US")
    ready = data_dir / "universes" / "research_ready" / "us_test_5_1d.json"
    ready.parent.mkdir(parents=True)
    ready.write_text(
        json.dumps({"universe_id": "us_test_5", "qualified_symbols": [{"symbol": "AAPL.US"}]}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "research",
            "run-universe",
            "--hypothesis",
            "research/hypotheses/examples/moving_average_momentum.yaml",
            "--qualified-universe",
            str(ready),
            "--data-dir",
            str(data_dir),
            "--source",
            "eodhd",
            "--timeframe",
            "1d",
            "--max-symbols",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "classification_counts" in result.output
