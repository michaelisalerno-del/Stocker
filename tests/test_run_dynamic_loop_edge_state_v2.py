from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

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
            "p_edge_positive": [0.99, 0.99],
            "p_next_payoff_positive": [0.1, 0.2],
            "edge_state": ["retired", "active"],
            "posterior_mean_net_bps": [1.0, 2.0],
            "posterior_std_net_bps": [10.0, 10.0],
            "posterior_predictive_std_net_bps": [20.0, 20.0],
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
    pooled = metrics.loc[metrics["scope"].eq("pooled")].set_index("model_name")
    assert pooled.loc["v1_60_session_selector", "brier_score"] == pytest.approx(0.81)
    assert pooled.loc["ewma_short_memory", "brier_score"] == pytest.approx(0.64)


def test_hierarchy_without_features_is_an_isolated_pooling_ablation() -> None:
    config = RUNNER.load_config()

    payoff_only = RUNNER.hierarchy_settings_from_config(
        config,
        enable_hierarchy=False,
        include_leading_features=False,
    )
    hierarchy_only = RUNNER.hierarchy_settings_from_config(
        config,
        enable_hierarchy=True,
        include_leading_features=False,
    )
    full = RUNNER.hierarchy_settings_from_config(
        config,
        enable_hierarchy=True,
        include_leading_features=True,
    )

    assert payoff_only.pooling_strength_sessions == 0.0
    assert hierarchy_only.pooling_strength_sessions > 0.0
    assert hierarchy_only.feature_logit_weights == {}
    assert full.feature_logit_weights


def test_population_context_is_recomputed_after_removing_a_stock() -> None:
    surface = pd.DataFrame(
        {
            "period": [2025, 2025, 2025, 2025],
            "session_date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "session": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "start_timestamp": pd.to_datetime(
                [
                    "2025-01-02T14:30:00Z",
                    "2025-01-02T14:35:00Z",
                    "2025-01-03T14:30:00Z",
                    "2025-01-03T14:35:00Z",
                ]
            ),
            "symbol_norm": ["AAA", "BBB", "AAA", "BBB"],
            "loop_id": ["cycle_01"] * 4,
            "orientation": ["state_1"] * 4,
            "horizon": [24] * 4,
            "state": [1, 1, 2, 1],
            "previous_state_1": [0, 0, 1, 1],
            "session_return": [0.01, 0.03, -0.02, 0.04],
            "mean_abs_return_12": [0.1, 0.3, 0.2, 0.4],
        }
    )

    reduced = RUNNER.rebuild_surface_context_for_universe(
        surface.loc[surface["symbol_norm"].eq("BBB")],
        universe_size=1,
    )

    assert reduced["structural_breadth"].eq(1.0).all()
    assert reduced.loc[reduced["session"].eq("2025-01-02"), "market_return"].iloc[0] == 0.03
    assert reduced.loc[reduced["session"].eq("2025-01-03"), "market_volatility"].iloc[0] == 0.4
    assert reduced["transition_surprise"].notna().all()


def test_episode_decay_features_use_the_decay_boundary_not_episode_onset() -> None:
    sessions = pd.bdate_range("2025-01-02", periods=16).strftime("%Y-%m-%d").tolist()
    payoffs = [-20] * 6 + [20, 30, 35, 25, 15, 0, -15, -20, -20, -20]
    panel = pd.DataFrame(
        {
            "period": 2025,
            "session": sessions,
            "loop_id": "cycle_01",
            "orientation": "state_1",
            "horizon": 24,
            "robust_net_payoff_bps": payoffs,
        }
    )
    calendar = pd.DataFrame(
        {
            "period": 2025,
            "score_session": sessions,
            "session_index": range(len(sessions)),
        }
    )
    forecasts = pd.DataFrame(
        {
            "period": 2025,
            "score_session": sessions,
            "session_index": range(len(sessions)),
            "loop_id": "cycle_01",
            "orientation": "state_1",
            "model_name": "hierarchical_change_point",
            "edge_state": "unknown",
        }
    )
    dispersion = [1.0] * len(sessions)
    surprise = [1.0] * len(sessions)
    dispersion[7:9] = [9.0, 12.0]
    surprise[7:9] = [8.0, 11.0]
    features = pd.DataFrame(
        {
            "period": 2025,
            "score_session": sessions,
            "loop_id": "cycle_01",
            "orientation": "state_1",
            "structural_breadth": 0.4,
            "top_second_margin": 0.2,
            "payoff_dispersion": dispersion,
            "transition_surprise": surprise,
        }
    )
    config = RUNNER.load_config()
    config["support"]["warmup_completed_sessions"] = 0
    config["evaluation"]["episode_smoothing_sessions"] = 3
    config["evaluation"]["episode_neutral_band_bps"] = 5.0

    episodes, _ = RUNNER.identify_hindsight_episodes(
        panel,
        forecasts,
        features,
        {2025: calendar},
        config,
    )

    episode = episodes.iloc[0]
    assert episode["hindsight_estimated_onset"] == sessions[6]
    assert episode["hindsight_estimated_decay_onset"] == sessions[8]
    assert episode["dispersion_change_before_decay"] > 0.0
    assert episode["structural_surprise_change_before_decay"] > 0.0


def test_operational_state_change_is_not_counted_as_bocpd_change_point() -> None:
    forecasts = pd.DataFrame(
        {
            "period": [2025, 2025],
            "score_session": ["2025-01-02", "2025-01-03"],
            "session_index": [0, 1],
            "loop_id": ["cycle_01", "cycle_01"],
            "orientation": ["state_1", "state_1"],
            "horizon": [24, 24],
            "model_name": ["hierarchical_change_point"] * 2,
            "edge_state": ["unknown", "active"],
            "p_change_now": [0.1, 0.1],
        }
    )
    hindsight_states = pd.DataFrame(
        columns=[
            "period",
            "score_session",
            "loop_id",
            "orientation",
            "horizon",
            "robust_net_payoff_bps",
        ]
    )

    diagnostics = RUNNER.change_point_diagnostics(
        forecasts,
        pd.DataFrame(),
        hindsight_states,
        RUNNER.load_config(),
    ).set_index("model_name")

    assert diagnostics.loc["hierarchical_change_point", "detected_change_points"] == 0
    assert diagnostics.loc["hierarchical_change_point", "operational_state_transitions"] == 1
