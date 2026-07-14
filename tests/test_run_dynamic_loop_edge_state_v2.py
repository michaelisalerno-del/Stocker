from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "research/slrno-v2/20260714-regime-loop-handoff/work"
RUNNER_PATH = WORK / "run_dynamic_loop_edge_state_v2.py"
CONFIG_PATH = WORK / "contracts/20260714-dynamic-loop-edge-state-v2.json"
SPEC = importlib.util.spec_from_file_location("run_dynamic_loop_edge_state_v2", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_registered_contract_is_research_only_and_freezes_main_choices() -> None:
    config = json.loads(CONFIG_PATH.read_text())

    assert config["safety"]["research_only"] is True
    assert config["safety"]["live_ordering_enabled"] is False
    assert config["safety"]["order_placement"] == "disabled"
    assert config["safety"]["application_position_or_exit_logic_changed"] is False
    assert config["registered_target"]["fixed_horizon_bars"] == 24
    assert config["change_point"]["primary_hazard_probability_per_observed_session"] == 0.05
    assert len(config["change_point"]["predeclared_hazard_sensitivities"]) == 2
    assert config["stress_tests"]["unbounded_parameter_search_allowed"] is False


def test_derived_execution_clock_waits_for_completed_bars_and_settlement() -> None:
    frame = pd.DataFrame(
        {
            "start_timestamp": [pd.Timestamp("2025-01-02T14:30:00Z")],
            "entry_step": [2],
            "horizon": [24],
            "status": ["filled"],
        }
    )

    result = RUNNER.derive_execution_clock(frame)

    assert result["decision_timestamp"].iloc[0] == pd.Timestamp("2025-01-02T14:35:00Z")
    assert result["entry_timestamp"].iloc[0] == pd.Timestamp("2025-01-02T14:40:00Z")
    assert result["exit_timestamp"].iloc[0] == pd.Timestamp("2025-01-02T16:35:00Z")
    assert result["settlement_timestamp"].iloc[0] == pd.Timestamp("2025-01-02T16:35:00Z")
    assert result["decision_timestamp"].iloc[0] < result["entry_timestamp"].iloc[0]


def test_required_artifact_contract_covers_every_requested_machine_readable_output() -> None:
    names = set(RUNNER.required_artifact_names())

    assert {
        "session_payoff_panel.parquet",
        "causal_edge_state_forecasts.parquet",
        "trade_decisions.parquet",
        "model_comparison_metrics.csv",
        "calibration_results.csv",
        "change_point_diagnostics.csv",
        "hindsight_episode_diagnostics.parquet",
        "stress_test_results.csv",
        "run_metadata.json",
    } <= names


def test_predictive_evaluation_accepts_one_frozen_row_per_model_for_one_target() -> None:
    forecasts = pd.DataFrame(
        {
            "period": [2025, 2025],
            "score_session": ["2025-04-01", "2025-04-01"],
            "loop_id": ["cycle_01", "cycle_01"],
            "orientation": ["state_1", "state_1"],
            "horizon": [24, 24],
            "model_name": ["v1_60_session_selector", "ewma_short_memory"],
            "p_edge_positive": [0.6, 0.7],
            "edge_state": ["retired", "active"],
            "posterior_mean_net_bps": [1.0, 2.0],
            "posterior_std_net_bps": [10.0, 10.0],
        }
    )
    payoff_panel = pd.DataFrame(
        {
            "period": [2025],
            "session": ["2025-04-01"],
            "loop_id": ["cycle_01"],
            "orientation": ["state_1"],
            "horizon": [24],
            "robust_net_payoff_bps": [5.0],
        }
    )

    metrics, _, scored = RUNNER.evaluate_prediction_models(
        forecasts,
        payoff_panel,
        RUNNER.load_config(),
    )

    assert len(scored) == 2
    assert set(metrics["model_name"]) == {
        "v1_60_session_selector",
        "ewma_short_memory",
    }
