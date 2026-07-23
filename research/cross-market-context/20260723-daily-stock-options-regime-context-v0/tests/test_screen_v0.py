from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from stocker_research.broad_conflict_advance_hazard_v02 import (
    DENSE_H0_FEATURES,
    ROUTE_FEATURES,
    candidate_normalized_weights,
)
from stocker_research.daily_soft_regimes_v0 import (
    DAILY_OPTIONS_DIMENSIONS,
    DAILY_STOCK_DIMENSIONS,
    apply_options_dimensions,
    apply_soft_regime,
    apply_stock_dimensions,
    fit_options_dimension_parameters,
    fit_soft_regime,
    fit_stock_dimension_parameters,
)
from stocker_research.daily_stock_options_context_v0 import (
    DAILY_OPTIONS_RAW_FEATURES,
    DAILY_STOCK_RAW_FEATURES,
    MISMATCH_FEATURES,
    add_mismatch_features,
    calculate_daily_stock_raw_features,
    choose_daily_context_decision,
    fit_mismatch_standardization,
    iv_horizon_outcomes,
    permute_bundle_within_slates,
    previous_us_trading_session,
    reject_protected_observations,
    select_daily_options_surface,
    validate_daily_context_chronology,
)
from stocker_research.stock_options_cross_market_quick_v0 import (
    fixed_session_bootstrap_multiplicities,
    reconstruct_clean_structural_panel,
)


def load_runner() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "run_screen_v0.py"
    specification = importlib.util.spec_from_file_location("daily_context_runner_test", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_daily_context_uses_exact_previous_trading_session_and_rejects_same_day_options() -> None:
    signal_date = date(2025, 1, 6)
    previous_session = previous_us_trading_session(signal_date)

    assert previous_session == date(2025, 1, 3)
    validate_daily_context_chronology(
        signal_date=signal_date,
        stock_information_date=previous_session,
        options_observation_date=previous_session,
    )
    with pytest.raises(ValueError, match="exact previous US trading session"):
        validate_daily_context_chronology(
            signal_date=signal_date,
            stock_information_date=previous_session,
            options_observation_date=signal_date,
        )


def test_daily_stock_raw_features_follow_the_frozen_trailing_definitions() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=25)
    close = np.arange(100.0, 125.0)
    bars = pd.DataFrame(
        {
            "symbol": "AAL",
            "session": sessions,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "activity": 1_000.0,
        }
    )

    raw = calculate_daily_stock_raw_features(bars)
    last = raw.iloc[-1]
    returns = np.diff(np.log(close))

    assert tuple(DAILY_STOCK_RAW_FEATURES) == (
        "daily_range_5_to_20",
        "daily_rv_5_to_20",
        "daily_range_overlap_5",
        "daily_efficiency_5",
        "daily_efficiency_10",
        "daily_sign_persistence_5",
        "daily_extension_20",
        "daily_extreme_wick_3",
        "daily_close_location_5",
        "daily_relative_return_5",
        "daily_activity_5_to_20",
    )
    assert last["daily_range_5_to_20"] == pytest.approx(1.0)
    assert last["daily_rv_5_to_20"] == pytest.approx(
        np.std(returns[-5:], ddof=1) / np.std(returns[-20:], ddof=1)
    )
    assert last["daily_range_overlap_5"] == pytest.approx(0.5)
    assert last["daily_efficiency_5"] == pytest.approx(1.0)
    assert last["daily_efficiency_10"] == pytest.approx(1.0)
    assert last["daily_sign_persistence_5"] == pytest.approx(1.0)
    assert last["daily_extreme_wick_3"] == pytest.approx(0.5)
    assert last["daily_close_location_5"] == pytest.approx(5.0 / 6.0)
    assert last["daily_relative_return_5"] == pytest.approx(0.0)
    assert last["daily_activity_5_to_20"] == pytest.approx(1.0)
    assert last["valid_trailing_sessions_20"] == 20
    assert all(last[f"{feature}_missing"] == 0 for feature in DAILY_STOCK_RAW_FEATURES)


def test_all_eight_daily_stock_dimensions_use_development_frozen_robust_scaling() -> None:
    values = np.arange(4.0)
    raw = pd.DataFrame(
        {
            "symbol": "AAL",
            "session": pd.date_range("2024-01-02", periods=4),
            **{feature: values for feature in DAILY_STOCK_RAW_FEATURES},
        }
    )
    parameters = fit_stock_dimension_parameters(raw)

    dimensions = apply_stock_dimensions(raw, parameters)
    last = dimensions.iloc[-1]

    assert tuple(DAILY_STOCK_DIMENSIONS) == (
        "daily_compression",
        "daily_directional_efficiency",
        "daily_trend_persistence",
        "daily_extension",
        "daily_rejection",
        "daily_volatility_acceleration",
        "daily_relative_strength",
        "daily_activity_acceleration",
    )
    assert last["daily_compression"] == pytest.approx(-1.0 / 3.0)
    assert last["daily_directional_efficiency"] == pytest.approx(1.0)
    assert last["daily_trend_persistence"] == pytest.approx(1.0)
    assert last["daily_extension"] == pytest.approx(1.0)
    assert last["daily_rejection"] == pytest.approx(0.0)
    assert last["daily_volatility_acceleration"] == pytest.approx(1.0)
    assert last["daily_relative_strength"] == pytest.approx(1.0)
    assert last["daily_activity_acceleration"] == pytest.approx(1.0)


def test_soft_regime_canonical_order_is_lexicographic_and_probabilities_are_stable() -> None:
    rows: list[dict[str, object]] = []
    for component, center in enumerate((-6.0, -2.0, 2.0, 6.0)):
        for repeat in range(12):
            rows.append(
                {
                    "symbol": f"S{repeat % 8}",
                    "session": f"2024-{component + 1:02d}-{repeat + 1:02d}",
                    **{dimension: center + 0.01 * repeat for dimension in DAILY_STOCK_DIMENSIONS},
                }
            )
    development = pd.DataFrame(rows)

    fitted = fit_soft_regime(
        development,
        dimensions=DAILY_STOCK_DIMENSIONS,
        missing_indicators=(),
        canonical_dimensions=(
            "daily_compression",
            "daily_volatility_acceleration",
            "daily_directional_efficiency",
            "daily_extension",
            "daily_rejection",
            "daily_relative_strength",
        ),
        prefix="daily_stock_regime",
    )
    assigned = apply_soft_regime(development, fitted)

    canonical_first_dimension = [
        centroid["daily_compression"] for centroid in fitted.canonical_centroids
    ]
    assert canonical_first_dimension == sorted(canonical_first_dimension)
    probability_columns = [f"daily_stock_regime_p_{value}" for value in range(4)]
    assert np.allclose(assigned[probability_columns].sum(axis=1), 1.0)
    assert set(assigned["daily_stock_regime"].astype(int)) == {0, 1, 2, 3}
    assert tuple(DAILY_OPTIONS_DIMENSIONS) == (
        "options_implied_tension",
        "options_premium_richness",
        "options_downside_asymmetry",
        "options_front_urgency",
        "options_liquidity_stress",
        "options_positioning_concentration",
        "options_directional_positioning",
        "options_surface_disagreement",
    )


def _option_row(
    option_type: str,
    expiration: str,
    dte: int,
    strike: float,
    contract_id: str,
    *,
    iv: float,
    delta: float,
    open_interest: float = 100.0,
) -> dict[str, object]:
    return {
        "underlying_symbol": "AAL",
        "trade_date": date(2025, 1, 3),
        "expiration_date": expiration,
        "dte": dte,
        "option_type": option_type,
        "strike": strike,
        "contract_id": contract_id,
        "bid": 1.0,
        "ask": 1.2,
        "midpoint": 1.1,
        "implied_volatility": iv,
        "delta": delta,
        "gamma": 0.01,
        "open_interest": open_interest,
        "volume": 10.0,
        "request_id": "request-1",
    }


def test_daily_options_surface_uses_front_back_skew_and_full_front_open_interest() -> None:
    rows: list[dict[str, object]] = []
    for strike in (95.0, 100.0, 105.0):
        rows.append(
            _option_row(
                "call",
                "2025-01-17",
                14,
                strike,
                f"front-call-{strike}",
                iv=0.35 if strike == 105.0 else 0.40,
                delta=0.25 if strike == 105.0 else 0.50,
            )
        )
        rows.append(
            _option_row(
                "put",
                "2025-01-17",
                14,
                strike,
                f"front-put-{strike}",
                iv=0.55 if strike == 95.0 else 0.50,
                delta=-0.25 if strike == 95.0 else -0.50,
            )
        )
    for option_type, delta, iv in (("call", 0.50, 0.30), ("put", -0.50, 0.40)):
        rows.append(
            _option_row(
                option_type,
                "2025-03-04",
                60,
                100.0,
                f"back-{option_type}",
                iv=iv,
                delta=delta,
            )
        )
    for option_type, delta in (("call", 0.99), ("put", -0.01)):
        rows.append(
            _option_row(
                option_type,
                "2025-01-17",
                14,
                80.0,
                f"far-{option_type}",
                iv=0.80,
                delta=delta,
            )
        )

    selected = select_daily_options_surface(
        pd.DataFrame(rows),
        previous_close=100.0,
        realised_volatility_20d=0.30,
    )

    assert selected["pair_available"] is True
    assert tuple(DAILY_OPTIONS_RAW_FEATURES) == (
        "atm_iv",
        "straddle_mid_pct",
        "call_put_iv_gap",
        "skew_25d",
        "front_term_urgency",
        "combined_relative_spread",
        "iv_minus_realised_20d",
        "near_spot_oi_concentration",
        "call_put_oi_imbalance",
    )
    assert selected["atm_iv"] == pytest.approx(0.45)
    assert selected["straddle_mid_pct"] == pytest.approx(0.022)
    assert selected["call_put_iv_gap"] == pytest.approx(-0.10)
    assert selected["skew_25d"] == pytest.approx(0.20)
    assert selected["front_term_urgency"] == pytest.approx(0.10)
    assert selected["combined_relative_spread"] == pytest.approx(0.4 / 2.2)
    assert selected["iv_minus_realised_20d"] == pytest.approx(0.15)
    assert selected["near_spot_oi_concentration"] == pytest.approx(0.75)
    assert selected["call_put_oi_imbalance"] == pytest.approx(0.0)
    assert selected["skew_missing"] == 0
    assert selected["back_expiry_missing"] == 0
    assert selected["front_call_contract_id"] == "front-call-100.0"
    assert selected["front_put_contract_id"] == "front-put-100.0"


def test_all_eight_options_dimensions_and_development_median_imputation_are_frozen() -> None:
    values = np.arange(4.0)
    development = pd.DataFrame(
        {
            "symbol": "AAL",
            "session": pd.date_range("2024-01-02", periods=4),
            **{feature: values for feature in DAILY_OPTIONS_RAW_FEATURES},
            "skew_missing": 0,
            "back_expiry_missing": 0,
            "oi_concentration_missing": 0,
            "call_put_oi_imbalance_missing": 0,
        }
    )
    parameters = fit_options_dimension_parameters(development)

    transformed = apply_options_dimensions(development, parameters)
    last = transformed.iloc[-1]

    assert last["options_implied_tension"] == pytest.approx(1.0)
    assert last["options_premium_richness"] == pytest.approx(1.0)
    assert last["options_downside_asymmetry"] == pytest.approx(0.0)
    assert last["options_front_urgency"] == pytest.approx(1.0)
    assert last["options_liquidity_stress"] == pytest.approx(1.0)
    assert last["options_positioning_concentration"] == pytest.approx(1.0)
    assert last["options_directional_positioning"] == pytest.approx(1.0)
    assert last["options_surface_disagreement"] == pytest.approx(1.0)

    missing = development.iloc[[0]].assign(
        session=pd.Timestamp("2025-01-02"),
        skew_25d=np.nan,
        front_term_urgency=np.nan,
        near_spot_oi_concentration=np.nan,
        call_put_oi_imbalance=np.nan,
    )
    imputed = apply_options_dimensions(missing, parameters)
    assert np.isfinite(imputed[list(DAILY_OPTIONS_DIMENSIONS)].to_numpy(float)).all()


def test_all_six_cross_market_mismatch_features_use_development_frozen_z_scores() -> None:
    columns = (
        "daily_compression",
        "options_implied_tension",
        "daily_volatility_acceleration",
        "options_front_urgency",
        "prefix_family_entropy",
        "options_premium_richness",
        "transition_probability",
        "signed_pressure",
        "options_directional_positioning",
    )
    development = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-03"],
            **{column: [0.0, 2.0] for column in columns},
        }
    )
    standardization = fit_mismatch_standardization(development)
    row = development.iloc[[1]].assign(BROAD_CONFLICT=1)

    result = add_mismatch_features(row, standardization).iloc[0]

    assert tuple(MISMATCH_FEATURES) == (
        "mismatch_compression_vs_iv",
        "mismatch_volatility_vs_urgency",
        "mismatch_route_vs_premium",
        "mismatch_transition_vs_urgency",
        "mismatch_direction_agreement",
        "mismatch_complacent_conflict",
    )
    assert result["mismatch_compression_vs_iv"] == pytest.approx(0.0)
    assert result["mismatch_volatility_vs_urgency"] == pytest.approx(0.0)
    assert result["mismatch_route_vs_premium"] == pytest.approx(0.0)
    assert result["mismatch_transition_vs_urgency"] == pytest.approx(0.0)
    assert result["mismatch_direction_agreement"] == pytest.approx(1.0)
    assert result["mismatch_complacent_conflict"] == pytest.approx(-1.0)


def test_iv_horizon_outcomes_cover_15m_same_next_and_third_close_without_option_pnl() -> None:
    outcomes = iv_horizon_outcomes(
        entry_price=100.0,
        atm_iv=0.50,
        close_15m=101.0,
        same_session_close=102.0,
        next_session_close=103.0,
        third_session_close=104.0,
        remaining_regular_session_minutes=300,
    )

    expected_15m = abs(np.log(1.01))
    expected_iv_15m = 0.50 * np.sqrt((15.0 / 390.0) / 252.0) * np.sqrt(2.0 / np.pi)
    assert outcomes["absolute_log_return_15m"] == pytest.approx(expected_15m)
    assert outcomes["iv_expected_absolute_15m"] == pytest.approx(expected_iv_15m)
    assert outcomes["movement_exceeds_prior_close_iv_15m"] == int(expected_15m > expected_iv_15m)
    assert outcomes["iv_absolute_residual_to_close"] == pytest.approx(
        abs(np.log(1.02)) - 0.50 * np.sqrt((300.0 / 390.0) / 252.0) * np.sqrt(2.0 / np.pi)
    )
    assert outcomes["iv_absolute_residual_next_close"] == pytest.approx(
        abs(np.log(1.03)) - 0.50 * np.sqrt((1.0 + 300.0 / 390.0) / 252.0) * np.sqrt(2.0 / np.pi)
    )
    assert outcomes["iv_absolute_residual_third_close"] == pytest.approx(
        abs(np.log(1.04)) - 0.50 * np.sqrt((3.0 + 300.0 / 390.0) / 252.0) * np.sqrt(2.0 / np.pi)
    )
    assert "option_pnl" not in outcomes


def test_full_regular_session_daily_aggregation_excludes_incomplete_days() -> None:
    runner = load_runner()
    bars = pd.DataFrame(
        {
            "symbol": ["AAL"] * 4,
            "session": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "bar_ordinal": [0, 1, 0, 1],
            "open": [100.0, 101.0, 102.0, np.nan],
            "high": [102.0, 103.0, 104.0, np.nan],
            "low": [99.0, 100.0, 101.0, np.nan],
            "close": [101.0, 102.0, 103.0, np.nan],
            "volume": [10.0, 20.0, 30.0, np.nan],
        }
    )

    daily = runner.aggregate_daily_bars(bars)

    assert daily["session"].tolist() == ["2025-01-02"]
    assert daily.iloc[0]["open"] == pytest.approx(100.0)
    assert daily.iloc[0]["high"] == pytest.approx(103.0)
    assert daily.iloc[0]["low"] == pytest.approx(99.0)
    assert daily.iloc[0]["close"] == pytest.approx(102.0)
    assert daily.iloc[0]["activity"] == pytest.approx(30.0)


def test_daily_activity_is_missing_when_any_intraday_activity_observation_is_missing() -> None:
    runner = load_runner()
    bars = pd.DataFrame(
        {
            "symbol": ["AAL", "AAL"],
            "session": ["2025-01-02", "2025-01-02"],
            "bar_ordinal": [0, 1],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10.0, np.nan],
        }
    )

    daily = runner.aggregate_daily_bars(bars)

    assert len(daily) == 1
    assert pd.isna(daily.iloc[0]["activity"])


def test_persistence_horizons_use_exact_exchange_calendar_successors() -> None:
    runner = load_runner()
    rows: list[dict[str, object]] = []
    for session, close in (
        ("2025-01-03", 103.0),
        ("2025-01-06", 104.0),
        ("2025-01-07", 105.0),
        ("2025-01-08", 106.0),
    ):
        ordinals = (5, 6, 8, 77) if session == "2025-01-03" else (77,)
        for ordinal in ordinals:
            rows.append(
                {
                    "symbol": "AAL",
                    "session": session,
                    "bar_ordinal": ordinal,
                    "open": 100.0 if ordinal == 6 else close,
                    "close": 101.0 if ordinal == 8 else close,
                }
            )
    panel = pd.DataFrame(
        {
            "row_id": ["AAL|2025-01-03|6"],
            "symbol": ["AAL"],
            "session": ["2025-01-03"],
            "checkpoint_bar_ordinal_zero_based": [5],
            "atm_iv": [0.50],
        }
    )
    daily_raw = pd.DataFrame(
        {
            "symbol": ["AAL"] * 4,
            "session": ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"],
            "inferred_corporate_action_boundary": [0, 0, 0, 0],
        }
    )

    result = runner.attach_movement_horizons(panel, pd.DataFrame(rows), daily_raw).iloc[0]

    assert result["next_close_session"] == "2025-01-06"
    assert result["third_close_session"] == "2025-01-08"
    assert result["remaining_regular_session_minutes"] == 360
    assert result["absolute_log_return_next_close"] == pytest.approx(abs(np.log(104.0 / 100.0)))
    assert result["absolute_log_return_third_close"] == pytest.approx(abs(np.log(106.0 / 100.0)))


def test_bundle_permutation_preserves_complete_bundle_and_nonbundle_columns() -> None:
    frame = pd.DataFrame(
        {
            "period": "assessment",
            "session": "2025-01-02",
            "checkpoint": 6,
            "symbol": ["A", "B", "C"],
            "bundle_a": [1, 2, 3],
            "bundle_b": [10, 20, 30],
            "outcome": [0, 1, 0],
        }
    )

    first = permute_bundle_within_slates(frame, columns=("bundle_a", "bundle_b"), seed=20260723)
    second = permute_bundle_within_slates(frame, columns=("bundle_a", "bundle_b"), seed=20260723)

    pd.testing.assert_frame_equal(first, second)
    assert first["outcome"].tolist() == frame["outcome"].tolist()
    assert sorted(zip(first["bundle_a"], first["bundle_b"], strict=True)) == [
        (1, 10),
        (2, 20),
        (3, 30),
    ]


def test_overall_decision_uses_only_the_preregistered_categories() -> None:
    assert (
        choose_daily_context_decision(
            blocker=None,
            test_a_daily_stock_supported=True,
            test_a_daily_options_supported=True,
            test_b_daily_stock_supported=True,
            test_b_intraday_route_supported=True,
            mismatch_supported=True,
            descriptive=False,
        )
        == "daily_stock_and_options_context_supported_bidirectionally"
    )
    assert (
        choose_daily_context_decision(
            blocker="blocked_insufficient_daily_options_coverage",
            test_a_daily_stock_supported=False,
            test_a_daily_options_supported=False,
            test_b_daily_stock_supported=False,
            test_b_intraday_route_supported=False,
            mismatch_supported=False,
            descriptive=False,
        )
        == "blocked_insufficient_daily_options_coverage"
    )


def test_protected_market_and_option_observations_are_rejected() -> None:
    safe = pd.DataFrame({"observation_date": ["2025-08-22"]})
    reject_protected_observations(safe, date_columns=("observation_date",))

    protected = pd.DataFrame({"observation_date": ["2025-08-23"]})
    with pytest.raises(ValueError, match="protected observation"):
        reject_protected_observations(
            protected,
            date_columns=("observation_date",),
        )


def test_clean_completion_target_excludes_next_bar_and_one_transition_rows() -> None:
    rows = pd.DataFrame(
        {
            "row_id": ["eligible", "next-bar", "one-transition"],
            "symbol": ["AAL", "AAL", "AAL"],
            "session": ["2025-01-02"] * 3,
            "period": ["assessment"] * 3,
            "checkpoint": [6, 8, 10],
            "registered_completion_next_1_bar": [0, 1, 0],
            "any_prefix_one_transition_from_completion": [0, 0, 1],
            "first_completion_lead": [2, 1, 3],
            "completion_in_bars_2_or_3": [1, 0, 1],
            "advance_eligible": [1, 0, 0],
            "route_resolution_state": ["BROAD_CONFLICT"] * 3,
            "row_weight": [1.0, np.nan, np.nan],
            **{feature: 0.0 for feature in (*DENSE_H0_FEATURES, *ROUTE_FEATURES)},
        }
    )

    reconstructed, audit = reconstruct_clean_structural_panel(rows)

    assert audit["passed"] is True
    assert reconstructed["row_id"].tolist() == ["eligible"]
    assert reconstructed["registered_completion_clean_bars_2_or_3"].tolist() == [1]
    assert reconstructed["row_weight"].tolist() == [1.0]


def test_candidate_normalised_weights_equalise_stock_sessions() -> None:
    frame = pd.DataFrame(
        {
            "period": ["assessment"] * 5,
            "session": ["2025-01-02"] * 5,
            "symbol": ["A", "A", "A", "B", "B"],
            "advance_eligible": [1] * 5,
        }
    )

    weighted = candidate_normalized_weights(frame)

    assert weighted.loc[weighted["symbol"].eq("A"), "row_weight"].tolist() == pytest.approx(
        [1.0 / 6.0] * 3
    )
    assert weighted.loc[weighted["symbol"].eq("B"), "row_weight"].tolist() == pytest.approx(
        [1.0 / 4.0] * 2
    )
    assert weighted.groupby("symbol")["row_weight"].sum().to_dict() == pytest.approx(
        {"A": 0.5, "B": 0.5}
    )


def test_s0_s1_s2_and_o0_o1_o2_surfaces_are_strictly_frozen() -> None:
    runner = load_runner()

    assert set(runner.S1_FEATURES) - set(runner.S0_FEATURES) == set(runner.STOCK_CONTEXT_FEATURES)
    assert set(runner.S2_FEATURES) - set(runner.S1_FEATURES) == set(
        (*runner.OPTIONS_CONTEXT_FEATURES, *MISMATCH_FEATURES)
    )
    assert set(runner.O1_FEATURES) - set(runner.O0_FEATURES) == set(runner.STOCK_CONTEXT_FEATURES)
    assert set(runner.O2_FEATURES) - set(runner.O1_FEATURES) == set(
        (*runner.H0_NON_CLOCK_FEATURES, *ROUTE_FEATURES, *MISMATCH_FEATURES)
    )
    assert set(runner.CHECKPOINT_FEATURES).issubset(runner.S0_FEATURES)
    assert set(runner.CHECKPOINT_FEATURES).issubset(runner.O0_FEATURES)
    assert not set(runner.STOCK_CONTEXT_FEATURES).intersection(runner.S0_FEATURES)
    assert not set(runner.OPTIONS_CONTEXT_FEATURES).intersection(runner.S1_FEATURES)


def test_session_bootstrap_is_exactly_ten_fixed_whole_session_draws() -> None:
    sessions = pd.Series(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"])

    first = fixed_session_bootstrap_multiplicities(sessions, draws=10, seed=20260723)
    second = fixed_session_bootstrap_multiplicities(sessions, draws=10, seed=20260723)

    assert len(first) == 10
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in first)
    with pytest.raises(ValueError, match="exactly 10"):
        fixed_session_bootstrap_multiplicities(sessions, draws=11, seed=20260723)


def test_options_and_route_null_permutations_move_intact_bundles_only() -> None:
    frame = pd.DataFrame(
        {
            "period": ["assessment"] * 4,
            "session": ["2025-01-02"] * 4,
            "checkpoint": [6] * 4,
            "symbol": ["A", "B", "C", "D"],
            "options_a": [1, 2, 3, 4],
            "options_b": [10, 20, 30, 40],
            "route_a": [5, 6, 7, 8],
            "route_state": ["A", "B", "C", "D"],
            "outcome": [0, 1, 0, 1],
        }
    )

    options_null = permute_bundle_within_slates(
        frame,
        columns=("options_a", "options_b"),
        seed=20260723,
    )
    route_null = permute_bundle_within_slates(
        frame,
        columns=("route_a", "route_state"),
        seed=20260726,
    )

    assert sorted(zip(options_null["options_a"], options_null["options_b"], strict=True)) == [
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 40),
    ]
    assert sorted(zip(route_null["route_a"], route_null["route_state"], strict=True)) == [
        (5, "A"),
        (6, "B"),
        (7, "C"),
        (8, "D"),
    ]
    assert options_null["route_a"].tolist() == frame["route_a"].tolist()
    assert route_null["options_a"].tolist() == frame["options_a"].tolist()
    assert options_null["outcome"].tolist() == frame["outcome"].tolist()
    assert route_null["outcome"].tolist() == frame["outcome"].tolist()
