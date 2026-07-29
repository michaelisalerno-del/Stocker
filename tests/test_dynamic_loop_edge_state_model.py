from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from stocker_research.dynamic_loop_edge_state.decision import (
    DecisionThresholds,
    SupportEvidence,
    classify_edge_state,
)
from stocker_research.dynamic_loop_edge_state.online_state import (
    BOCPDSettings,
    EdgeForecast,
    HierarchicalPayoffModel,
    HierarchicalSettings,
    PayoffObservation,
    RobustBOCPD,
)


def _forecast(**overrides: float) -> EdgeForecast:
    values: dict[str, float] = {
        "p_change_now": 0.05,
        "posterior_run_length_mean": 12.0,
        "posterior_run_length_mode": 12.0,
        "posterior_mean_net_bps": 20.0,
        "posterior_std_net_bps": 10.0,
        "posterior_lower_bound_net_bps": 7.0,
        "p_edge_positive": 0.97,
        "p_edge_active": 0.95,
        "p_on_next": 0.2,
        "p_off_next": 0.05,
        "p_survive_horizon": 0.98,
        "out_of_distribution_score": 0.5,
    }
    values.update(overrides)
    return EdgeForecast(**values)


def _support(**overrides: float | int) -> SupportEvidence:
    values: dict[str, float | int] = {
        "effective_sessions": 12.0,
        "independent_stocks": 8,
        "raw_fills": 20,
        "effective_sample_size": 24.0,
    }
    values.update(overrides)
    return SupportEvidence(**values)


def test_bocpd_shortens_effective_memory_after_a_synthetic_mean_shift() -> None:
    model = RobustBOCPD(BOCPDSettings(hazard_probability=0.05))
    for value in [-30.0] * 40:
        model.update(value, effective_sample_size=5.0)
    before = model.snapshot()

    shifts = [model.update(90.0, effective_sample_size=5.0) for _ in range(8)]

    assert before.posterior_run_length_mean > 20.0
    assert max(item.p_change_now for item in shifts) > 0.2
    assert shifts[-1].posterior_run_length_mean < 15.0
    assert shifts[-1].posterior_mean_net_bps > 0.0


def test_posterior_uncertainty_falls_with_more_independent_evidence() -> None:
    sparse = RobustBOCPD(BOCPDSettings(hazard_probability=0.02))
    broad = RobustBOCPD(BOCPDSettings(hazard_probability=0.02))
    for value in [15.0, 12.0, 18.0, 14.0, 16.0] * 3:
        sparse.update(value, effective_sample_size=1.0)
        broad.update(value, effective_sample_size=8.0)

    assert broad.snapshot().posterior_std_net_bps < sparse.snapshot().posterior_std_net_bps


def test_one_extreme_outlier_does_not_create_a_long_false_activation() -> None:
    model = RobustBOCPD(BOCPDSettings(hazard_probability=0.05))
    for _ in range(30):
        model.update(0.0, effective_sample_size=5.0)
    model.update(2_000.0, effective_sample_size=1.0)
    after = [model.update(0.0, effective_sample_size=5.0) for _ in range(5)]

    assert sum(item.p_edge_positive >= 0.9 for item in after) <= 1
    assert after[-1].posterior_mean_net_bps < 10.0


def test_unknown_is_default_when_independent_support_is_insufficient() -> None:
    decision = classify_edge_state(
        _forecast(),
        _support(effective_sessions=2.0, independent_stocks=1, effective_sample_size=2.0),
        DecisionThresholds(),
        previous_state="unknown",
    )

    assert decision.edge_state == "unknown"
    assert decision.admit_new_entry is False
    assert "insufficient_sessions" in decision.reason_codes
    assert "insufficient_stocks" in decision.reason_codes


def test_active_requires_every_frozen_admission_condition() -> None:
    active = classify_edge_state(
        _forecast(), _support(), DecisionThresholds(), previous_state="unknown"
    )
    no_survival = classify_edge_state(
        _forecast(p_survive_horizon=0.7),
        _support(),
        DecisionThresholds(),
        previous_state="unknown",
    )

    assert active.edge_state == "active"
    assert active.admit_new_entry is True
    assert active.reason_codes == ("admitted_active_edge",)
    assert no_survival.edge_state == "unknown"
    assert no_survival.admit_new_entry is False
    assert "survival_probability_too_low" in no_survival.reason_codes


def test_decaying_and_retired_states_block_new_entries() -> None:
    decaying = classify_edge_state(
        _forecast(p_off_next=0.4, posterior_lower_bound_net_bps=-1.0),
        _support(),
        DecisionThresholds(),
        previous_state="active",
    )
    retired = classify_edge_state(
        _forecast(
            p_change_now=0.6,
            posterior_mean_net_bps=-10.0,
            posterior_lower_bound_net_bps=-20.0,
            p_edge_positive=0.2,
            p_edge_active=0.15,
            p_off_next=0.7,
        ),
        _support(),
        DecisionThresholds(),
        previous_state="active",
    )

    assert decaying.edge_state == "decaying"
    assert decaying.admit_new_entry is False
    assert "decaying_state" in decaying.reason_codes
    assert retired.edge_state == "retired"
    assert retired.admit_new_entry is False
    assert "retired_state" in retired.reason_codes


def test_admission_state_never_closes_an_existing_position() -> None:
    decision = classify_edge_state(
        _forecast(p_off_next=0.5),
        _support(),
        DecisionThresholds(),
        previous_state="active",
        has_existing_position=True,
    )

    assert decision.admit_new_entry is False
    assert decision.existing_position_action == "unchanged_existing_exit_rule"


def _observation(
    session_index: int,
    cell: tuple[str, str, int],
    value: float,
    stocks: tuple[str, ...] = ("A", "B", "C", "D"),
) -> PayoffObservation:
    return PayoffObservation(
        cell_key=cell,
        session=f"s{session_index:03d}",
        net_payoff_bps=value,
        effective_sample_size=float(len(stocks)),
        independent_stocks=stocks,
        raw_fills=len(stocks),
        availability_timestamp=pd.Timestamp("2025-01-01", tz="UTC")
        + pd.Timedelta(days=session_index),
    )


def test_hierarchical_pooling_helps_sparse_cell_only_when_shared_evidence_is_relevant() -> None:
    settings = HierarchicalSettings(pooling_strength_sessions=12.0)
    positive = HierarchicalPayoffModel(BOCPDSettings(), settings)
    negative = HierarchicalPayoffModel(BOCPDSettings(), settings)
    common_a = ("cycle_01", "state_1", 24)
    common_b = ("cycle_02", "state_2", 24)
    sparse = ("cycle_03", "state_3", 24)
    for index in range(12):
        positive.update_session(
            f"s{index:03d}",
            [
                _observation(index, common_a, 40.0),
                _observation(index, common_b, 35.0),
            ],
        )
        negative.update_session(
            f"s{index:03d}",
            [
                _observation(index, common_a, -40.0),
                _observation(index, common_b, -35.0),
            ],
        )
    positive.update_session("s012", [_observation(12, sparse, 10.0, ("A",))])
    negative.update_session("s012", [_observation(12, sparse, 10.0, ("A",))])

    positive_forecast, _ = positive.forecast(sparse, horizon_bars=24, session_bars=78)
    negative_forecast, _ = negative.forecast(sparse, horizon_bars=24, session_bars=78)

    assert positive_forecast.p_edge_positive > negative_forecast.p_edge_positive
    assert positive_forecast.posterior_mean_net_bps > negative_forecast.posterior_mean_net_bps
    assert positive_forecast.posterior_std_net_bps > 0.0


def test_pooling_weight_uses_independent_stocks_and_effective_sample_size() -> None:
    common = ("cycle_01", "state_1", 24)
    sparse = ("cycle_02", "state_2", 24)
    settings = HierarchicalSettings(
        pooling_strength_sessions=12.0,
        independent_stock_reference=5.0,
        effective_sample_size_per_session_reference=5.0,
    )
    one_stock = HierarchicalPayoffModel(BOCPDSettings(), settings)
    five_stocks = HierarchicalPayoffModel(BOCPDSettings(), settings)
    for index in range(8):
        one_stock.update_session(
            f"s{index:03d}",
            [
                _observation(index, common, -30.0),
                _observation(index, sparse, 30.0, ("A",)),
            ],
        )
        five_stocks.update_session(
            f"s{index:03d}",
            [
                _observation(index, common, -30.0),
                _observation(index, sparse, 30.0, ("A", "B", "C", "D", "E")),
            ],
        )

    sparse_forecast, _ = one_stock.forecast(sparse, horizon_bars=24, session_bars=78)
    broad_forecast, _ = five_stocks.forecast(sparse, horizon_bars=24, session_bars=78)

    assert broad_forecast.hierarchical_cell_weight > sparse_forecast.hierarchical_cell_weight


def test_event_probability_uses_predictive_observation_uncertainty() -> None:
    cell = ("cycle_08", "state_3", 24)
    model = HierarchicalPayoffModel(BOCPDSettings(), HierarchicalSettings())
    for index, value in enumerate([20.0, -10.0, 25.0, -15.0, 30.0]):
        model.update_session(f"s{index:03d}", [_observation(index, cell, value)])

    forecast, _ = model.forecast(cell, horizon_bars=24, session_bars=78)

    assert forecast.posterior_predictive_std_net_bps >= forecast.posterior_std_net_bps
    assert 0.0 < forecast.p_next_payoff_positive < 1.0


def test_onset_and_termination_probabilities_are_conditional_transitions() -> None:
    cell = ("cycle_09", "state_4", 24)
    model = HierarchicalPayoffModel(
        BOCPDSettings(hazard_probability=0.05),
        HierarchicalSettings(feature_logit_weights={"breadth": 0.5}),
    )
    model.update_session("s000", [_observation(0, cell, 20.0)])

    payoff_history_only, _ = model.forecast(
        cell,
        horizon_bars=24,
        session_bars=78,
        include_leading_features=False,
    )
    forecast, _ = model.forecast(
        cell,
        horizon_bars=24,
        session_bars=78,
        leading_features={"breadth": 1.0},
    )

    assert forecast.p_on_next + forecast.p_off_next < 0.2
    assert forecast.p_next_payoff_positive > payoff_history_only.p_next_payoff_positive
    assert forecast.p_survive_horizon == pytest.approx((1.0 - forecast.p_off_next) ** (24.0 / 78.0))


def test_zero_pooling_payoff_only_model_has_a_finite_cold_start_forecast() -> None:
    model = HierarchicalPayoffModel(
        BOCPDSettings(),
        HierarchicalSettings(
            pooling_strength_sessions=0.0,
            sparse_uncertainty_inflation_bps=0.0,
        ),
    )
    model.update_session(
        "s000",
        [_observation(0, ("cycle_02", "state_2", 24), 10.0)],
    )

    forecast, support = model.forecast(
        ("cycle_01", "state_1", 24),
        horizon_bars=24,
        session_bars=78,
        include_leading_features=False,
    )

    assert np.isfinite(forecast.posterior_mean_net_bps)
    assert np.isfinite(forecast.posterior_std_net_bps)
    assert support.effective_sessions == 0.0


def test_synthetic_temporary_edge_activates_and_terminates_without_sixty_session_lag() -> None:
    cell = ("cycle_04", "state_4", 24)
    model = HierarchicalPayoffModel(
        BOCPDSettings(hazard_probability=0.05),
        HierarchicalSettings(pooling_strength_sessions=4.0),
    )
    thresholds = DecisionThresholds(
        minimum_independent_sessions=5,
        minimum_independent_stocks=4,
        minimum_effective_sample_size=10.0,
        maximum_posterior_std_net_bps=100.0,
        active_probability=0.85,
        survival_probability=0.8,
    )
    values = [0.0] * 40 + [70.0] * 20 + list(np.linspace(50.0, -30.0, 15)) + [-40.0] * 30
    states: list[str] = []
    previous = "unknown"
    for index, value in enumerate(values):
        model.update_session(f"s{index:03d}", [_observation(index, cell, float(value))])
        forecast, support = model.forecast(cell, horizon_bars=24, session_bars=78)
        decision = classify_edge_state(forecast, support, thresholds, previous_state=previous)
        states.append(decision.edge_state)
        previous = decision.edge_state

    positive_states = states[40:60]
    post_edge_states = states[75:]
    assert states[:35].count("active") <= 3
    assert "active" in positive_states[:12]
    assert "decaying" in states[55:80] or "retired" in states[55:80]
    assert "retired" in post_edge_states[:15]


def test_model_predictions_are_reproducible_for_identical_input_order() -> None:
    cell = ("cycle_07", "state_5", 24)
    models = [
        HierarchicalPayoffModel(BOCPDSettings(), HierarchicalSettings()),
        HierarchicalPayoffModel(BOCPDSettings(), HierarchicalSettings()),
    ]
    for model in models:
        for index, value in enumerate([0.0, 10.0, 20.0, -5.0, 30.0]):
            model.update_session(f"s{index:03d}", [_observation(index, cell, value)])

    first, first_support = models[0].forecast(cell, horizon_bars=24, session_bars=78)
    second, second_support = models[1].forecast(cell, horizon_bars=24, session_bars=78)

    assert asdict(first) == asdict(second)
    assert asdict(first_support) == asdict(second_support)
