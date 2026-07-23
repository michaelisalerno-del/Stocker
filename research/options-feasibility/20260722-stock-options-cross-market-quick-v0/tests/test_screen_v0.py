from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stocker_research.stock_options_cross_market_quick_v0 import (
    BASE_OPTIONS_FEATURES,
    CROSS_MARKET_FEATURES,
    DENSE_H0_FEATURES,
    OPTIONS_MODEL_FEATURES,
    ROUTE_FEATURES,
    TEST_A_S0_NUMERIC,
    TEST_A_S1_NUMERIC,
    TEST_A_S2_NUMERIC,
    TEST_B_O0_NUMERIC,
    TEST_B_O1_NUMERIC,
    TEST_B_O2_NUMERIC,
    add_cross_market_disagreement,
    add_test_a_target,
    add_test_b_target,
    apply_stock_relative_options,
    assert_protected_dates,
    choose_overall_decision,
    extract_exact_history_records,
    fit_cross_market_model,
    fit_cross_market_standardization,
    fit_stock_relative_options,
    fixed_session_bootstrap_multiplicities,
    manual_model_prediction,
    permute_options_bundle,
    permute_route_bundle_and_state,
    previous_trading_session,
    route_state_movement_metrics,
    select_primary_atm_pair,
    trailing_realised_volatility_20d,
    validate_exact_previous_session_join,
)
from stocker_research.stock_options_cross_market_quick_v0 import (
    test_a_disagreement_increment_passes as a_disagreement_increment_passes,
)
from stocker_research.stock_options_cross_market_quick_v0 import (
    test_a_options_increment_passes as a_options_increment_passes,
)
from stocker_research.stock_options_cross_market_quick_v0 import (
    test_b_route_increment_passes as b_route_increment_passes,
)
from stocker_research.stock_options_cross_market_quick_v0 import (
    test_b_stock_increment_passes as b_stock_increment_passes,
)


def _option_row(
    *,
    option_type: str,
    expiration: str,
    dte: int,
    strike: float,
    contract: str,
    iv: float = 0.4,
    bid: float = 1.0,
    ask: float = 1.2,
    open_interest: int = 100,
    delta: float | None = None,
) -> dict[str, object]:
    return {
        "option_type": option_type,
        "expiration_date": expiration,
        "trade_date": date(2025, 1, 3),
        "dte": dte,
        "strike": strike,
        "contract_id": contract,
        "bid": bid,
        "ask": ask,
        "midpoint": (bid + ask) / 2.0,
        "implied_volatility": iv,
        "open_interest": open_interest,
        "delta": delta,
        "gamma": 0.01,
        "volume": 10,
    }


def test_previous_session_join_accepts_friday_and_rejects_same_day() -> None:
    signal = date(2025, 1, 6)
    required = previous_trading_session(signal)

    assert required == date(2025, 1, 3)
    validate_exact_previous_session_join(
        signal_date=signal,
        required_options_date=required,
        actual_options_date=required,
    )
    with pytest.raises(ValueError, match="exact previous trading session"):
        validate_exact_previous_session_join(
            signal_date=signal,
            required_options_date=required,
            actual_options_date=signal,
        )


def test_frozen_pair_selection_uses_nearest_expiry_and_never_quality_falls_back() -> None:
    rows = [
        _option_row(
            option_type=side,
            expiration="2025-01-17",
            dte=14,
            strike=strike,
            contract=f"{side}-{strike}",
            open_interest=5 if strike == 100.0 else 500,
        )
        for strike in (100.0, 105.0)
        for side in ("call", "put")
    ]
    rows.extend(
        [
            _option_row(
                option_type=side,
                expiration="2025-01-24",
                dte=21,
                strike=100.0,
                contract=f"later-{side}",
            )
            for side in ("call", "put")
        ]
    )

    selected = select_primary_atm_pair(pd.DataFrame(rows), previous_close=100.0)

    assert not selected.available
    assert selected.expiration_date == date(2025, 1, 17)
    assert selected.strike == 100.0
    assert selected.reason == "selected_pair_open_interest_below_10"


def test_stock_relative_scaling_is_fit_only_on_2024_development() -> None:
    development = pd.DataFrame(
        {
            "symbol": ["AAL", "AAL", "AAL"],
            "session": ["2024-01-02", "2024-02-01", "2024-03-01"],
            "atm_iv": [0.2, 0.3, 0.4],
            "straddle_mid_pct": [0.01, 0.02, 0.03],
            "skew_25d": [-0.02, 0.0, 0.02],
            "term_structure": [-0.01, 0.0, 0.01],
        }
    )
    parameters = fit_stock_relative_options(development)
    assessment = development.iloc[[0]].assign(
        session="2025-01-02",
        atm_iv=0.9,
        straddle_mid_pct=0.09,
        skew_25d=0.09,
        term_structure=0.09,
    )

    transformed = apply_stock_relative_options(assessment, parameters)

    assert transformed.iloc[0]["atm_iv_stock_percentile"] == 1.0
    assert transformed.iloc[0]["atm_iv_stock_robust_z"] > 3.0
    with pytest.raises(ValueError, match="2024 only"):
        fit_stock_relative_options(assessment)


def test_trailing_realised_volatility_uses_last_twenty_close_returns() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=24)
    returns = np.asarray([0.01, -0.005, 0.002, -0.003] * 6, dtype=float)
    closes = 100.0 * np.exp(np.cumsum(returns))
    bars = pd.DataFrame(
        {
            "symbol": "AAL",
            "session": sessions,
            "close": closes,
        }
    )

    result = trailing_realised_volatility_20d(bars)
    observed = float(result.iloc[-1]["realised_volatility_20d"])
    expected = float(np.std(returns[-20:], ddof=1) * math.sqrt(252.0))

    assert observed == pytest.approx(expected)
    assert int(result.iloc[-1]["valid_trailing_return_sessions"]) == 20


def test_five_disagreement_features_follow_the_fixed_formulas() -> None:
    development = pd.DataFrame(
        {
            "session": ["2024-01-02", "2024-01-03"],
            "tension": [1.0, 3.0],
            "prefix_family_entropy": [2.0, 4.0],
            "signed_pressure": [-1.0, 1.0],
            "call_put_iv_gap": [-0.2, 0.2],
            "transition_probability": [0.25, 0.75],
        }
    )
    parameters = fit_cross_market_standardization(development)
    frame = development.iloc[[1]].assign(
        route_resolution_state="BROAD_CONFLICT",
        BROAD_CONFLICT=1,
        atm_iv_stock_robust_z=0.5,
        straddle_move_stock_robust_z=-0.25,
        term_structure_stock_robust_z=0.75,
    )

    result = add_cross_market_disagreement(frame, parameters).iloc[0]

    assert result["complacent_conflict"] == pytest.approx(-0.5)
    assert result["structural_tension_gap"] == pytest.approx(0.5)
    assert result["route_vs_priced_move"] == pytest.approx(1.25)
    assert result["directional_agreement"] == pytest.approx(1.0)
    assert result["transition_vs_term_urgency"] == pytest.approx(0.25)
    assert tuple(column for column in CROSS_MARKET_FEATURES if column in result.index) == (
        CROSS_MARKET_FEATURES
    )


def test_test_a_target_excludes_next_bar_and_one_transition_rows() -> None:
    frame = pd.DataFrame(
        {
            "first_completion_lead": [2, 3, 1, 2, 0],
            "registered_completion_next_1_bar": [0, 0, 1, 0, 0],
            "any_prefix_one_transition_from_completion": [0, 0, 0, 1, 0],
        }
    )

    result = add_test_a_target(frame)

    assert result["clean_advance_eligible"].tolist() == [1, 1, 0, 0, 1]
    assert result["registered_completion_clean_bars_2_or_3"].tolist()[:2] == [1.0, 1.0]
    assert math.isnan(result.iloc[2]["registered_completion_clean_bars_2_or_3"])
    assert math.isnan(result.iloc[3]["registered_completion_clean_bars_2_or_3"])
    assert result.iloc[4]["registered_completion_clean_bars_2_or_3"] == 0.0


def test_test_b_target_is_absolute_movement_above_expected_absolute_iv() -> None:
    frame = pd.DataFrame(
        {
            "absolute_log_return_15m": [0.01, 0.0001],
            "atm_iv": [0.4, 0.4],
        }
    )

    result = add_test_b_target(frame)
    expected = 0.4 * math.sqrt(15 / (252 * 390)) * math.sqrt(2 / math.pi)

    assert result["iv_expected_absolute_15m"].tolist() == pytest.approx([expected, expected])
    assert result["movement_exceeds_prior_close_iv"].tolist() == [1, 0]
    assert result["iv_absolute_residual_15m"].tolist() == pytest.approx(
        [0.01 - expected, 0.0001 - expected]
    )


def test_model_feature_surfaces_are_nested_without_search() -> None:
    assert (*DENSE_H0_FEATURES, *ROUTE_FEATURES) == TEST_A_S0_NUMERIC
    assert (*TEST_A_S0_NUMERIC, *OPTIONS_MODEL_FEATURES) == TEST_A_S1_NUMERIC
    assert (*TEST_A_S1_NUMERIC, *CROSS_MARKET_FEATURES) == TEST_A_S2_NUMERIC
    assert TEST_B_O0_NUMERIC == OPTIONS_MODEL_FEATURES
    assert (*TEST_B_O0_NUMERIC, *DENSE_H0_FEATURES) == TEST_B_O1_NUMERIC
    assert (
        *TEST_B_O1_NUMERIC,
        *ROUTE_FEATURES,
        *CROSS_MARKET_FEATURES,
    ) == TEST_B_O2_NUMERIC
    assert len(BASE_OPTIONS_FEATURES) == 13
    assert len(CROSS_MARKET_FEATURES) == 5
    assert len(ROUTE_FEATURES) == 15


def test_route_state_statistics_include_tail_contribution_and_binding_contrast() -> None:
    frame = pd.DataFrame(
        [
            {
                "route_resolution_state": state,
                "absolute_log_return_15m": movement,
                "iv_expected_absolute_15m": 0.01,
                "iv_absolute_residual_15m": movement - 0.01,
                "movement_exceeds_prior_close_iv": int(movement > 0.01),
                "row_weight": 1.0,
                "session": f"2025-01-0{index + 2}",
                "symbol": symbol,
                "atm_iv": 0.4,
            }
            for index, (state, movement, symbol) in enumerate(
                [
                    ("BROAD_CONFLICT", 0.03, "AAL"),
                    ("BROAD_CONFLICT", 0.02, "MSTR"),
                    ("LOW_ROUTE_SUPPORT", 0.005, "WULF"),
                    ("NARROWING", 0.015, "SOFI"),
                    ("DOMINANT_ROUTE", 0.012, "RIOT"),
                ]
            )
        ]
    )

    table, contrast = route_state_movement_metrics(frame)

    assert table["route_state"].tolist() == [
        "BROAD_CONFLICT",
        "LOW_ROUTE_SUPPORT",
        "NARROWING",
        "OTHER",
    ]
    broad = table.loc[table["route_state"].eq("BROAD_CONFLICT")].iloc[0]
    assert broad["top_5pct_positive_residual_contribution"] == pytest.approx(2 / 3)
    assert contrast["mean_iv_residual_difference"] == pytest.approx(0.02)
    assert contrast["exceed_iv_rate_difference"] == pytest.approx(1.0)


def test_session_bootstrap_has_exactly_ten_draws_and_whole_session_multiplicity() -> None:
    sessions = pd.Series(["a", "a", "b", "b", "c"])

    draws = fixed_session_bootstrap_multiplicities(sessions)

    assert len(draws) == 10
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in draws)
    with pytest.raises(ValueError, match="exactly 10"):
        fixed_session_bootstrap_multiplicities(sessions, draws=9)


def _null_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, symbol in enumerate(("AAL", "MSTR", "WULF")):
        row: dict[str, object] = {
            "period": "development",
            "session": "2024-01-02",
            "checkpoint": 6,
            "symbol": symbol,
            "outcome": index % 2,
            "row_weight": 1.0 / 3.0,
            "route_resolution_state": ("BROAD_CONFLICT" if index == 0 else "LOW_ROUTE_SUPPORT"),
        }
        row.update(
            {feature: float(index * 100 + offset) for offset, feature in enumerate(ROUTE_FEATURES)}
        )
        row.update(
            {
                feature: float(index * 1000 + offset)
                for offset, feature in enumerate(OPTIONS_MODEL_FEATURES)
            }
        )
        row.update(
            {
                "tension": float(index + 1),
                "signed_pressure": float(index - 1),
                "transition_probability": float((index + 1) / 4),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_options_null_permutes_the_complete_bundle_and_preserves_everything_else() -> None:
    frame = _null_frame()
    standardization = fit_cross_market_standardization(frame)
    original_bundles = {
        tuple(row)
        for row in frame.loc[:, list(OPTIONS_MODEL_FEATURES)].itertuples(index=False, name=None)
    }

    result = permute_options_bundle(
        frame,
        seed=7,
        standardization=standardization,
    )

    observed_bundles = {
        tuple(row)
        for row in result.loc[:, list(OPTIONS_MODEL_FEATURES)].itertuples(index=False, name=None)
    }
    assert observed_bundles == original_bundles
    assert result["outcome"].tolist() == frame["outcome"].tolist()
    assert result["row_weight"].tolist() == frame["row_weight"].tolist()
    assert result.loc[:, list(ROUTE_FEATURES)].equals(frame.loc[:, list(ROUTE_FEATURES)])
    assert result["directional_agreement"].tolist() == pytest.approx(
        (result["standardised_signed_pressure"] * result["standardised_call_put_iv_gap"]).tolist()
    )


def test_route_null_permutes_route_features_and_state_as_one_bundle() -> None:
    frame = _null_frame()
    standardization = fit_cross_market_standardization(frame)
    columns = [*ROUTE_FEATURES, "route_resolution_state"]
    original_bundles = {
        tuple(row) for row in frame.loc[:, columns].itertuples(index=False, name=None)
    }

    result = permute_route_bundle_and_state(
        frame,
        seed=11,
        standardization=standardization,
    )

    observed_bundles = {
        tuple(row) for row in result.loc[:, columns].itertuples(index=False, name=None)
    }
    assert observed_bundles == original_bundles
    assert result["outcome"].tolist() == frame["outcome"].tolist()
    assert result.loc[:, list(OPTIONS_MODEL_FEATURES)].equals(
        frame.loc[:, list(OPTIONS_MODEL_FEATURES)]
    )
    assert result["route_vs_priced_move"].tolist() == pytest.approx(
        (
            result["standardised_prefix_family_entropy"] - result["straddle_move_stock_robust_z"]
        ).tolist()
    )


def test_protected_date_rejection_is_fail_closed() -> None:
    assert_protected_dates(
        pd.DataFrame({"session": ["2025-08-22"]}),
        columns=("session",),
    )
    with pytest.raises(ValueError, match="protected date"):
        assert_protected_dates(
            pd.DataFrame({"session": ["2025-08-23"]}),
            columns=("session",),
        )


def test_cached_history_decodes_only_exact_pre_boundary_records() -> None:
    raw = json.dumps(
        {
            "data": [
                {
                    "id": "TEST-2025-08-20",
                    "attributes": {"tradetime": "2025-08-20", "value": 1},
                },
                {
                    "id": "TEST-2025-08-21",
                    "attributes": {"tradetime": "2025-08-21", "value": 2},
                },
                {
                    "id": "TEST-2025-08-23",
                    "attributes": {"tradetime": "2025-08-23", "value": 3},
                },
            ]
        }
    )

    extraction = extract_exact_history_records(
        raw,
        required_date=date(2025, 8, 21),
    )

    assert len(extraction.records) == 1
    assert extraction.records[0]["attributes"]["value"] == 2
    assert extraction.cached_records_scanned == 3
    assert extraction.nonmatching_records_skipped == 1
    assert extraction.protected_records_skipped_before_materialisation == 1
    with pytest.raises(ValueError, match="protected boundary"):
        extract_exact_history_records(raw, required_date=date(2025, 8, 23))


def test_model_coefficients_reconstruct_probabilities_exactly() -> None:
    development = pd.DataFrame(
        {
            "period": "development",
            "session": pd.date_range("2024-01-02", periods=12, freq="7D"),
            "symbol": ["AAL", "MSTR"] * 6,
            "checkpoint": [6, 8, 10] * 4,
            "route_resolution_state": ["BROAD_CONFLICT", "LOW_ROUTE_SUPPORT"] * 6,
            "x": np.linspace(-2.0, 2.0, 12),
            "target": [0, 1] * 6,
            "row_weight": np.ones(12),
        }
    )
    model = fit_cross_market_model(
        development,
        model_id="worked",
        numeric_features=("x",),
        category_control_names=("stock", "checkpoint"),
        target_column="target",
        kind="logistic",
    )

    predicted = model.predict(development)
    reconstructed = manual_model_prediction(development, model.as_dict())

    assert float(np.max(np.abs(predicted - reconstructed))) <= 1e-15


def test_decision_logic_keeps_directions_separate_and_applies_exact_gates() -> None:
    assert (
        choose_overall_decision(
            blocker=None,
            test_a_supported=True,
            test_b_supported=False,
            disagreement_descriptive=False,
        )
        == "options_improve_stock_method_only"
    )
    assert (
        choose_overall_decision(
            blocker=None,
            test_a_supported=False,
            test_b_supported=True,
            disagreement_descriptive=False,
        )
        == "stock_structure_improves_options_forecast_only"
    )
    assert (
        choose_overall_decision(
            blocker="blocked_insufficient_cached_options_coverage",
            test_a_supported=True,
            test_b_supported=True,
            disagreement_descriptive=True,
        )
        == "blocked_insufficient_cached_options_coverage"
    )
    common = {
        "log_loss_improvement": 0.01,
        "brier_improvement": 0.01,
        "auc_improvement": 0.0,
        "average_precision_improvement": 0.01,
        "bootstrap_80_log_loss_lower": 0.0,
        "bootstrap_80_brier_lower": 0.0,
        "positive_months": 4,
        "real_log_loss_or_brier_exceeds_all_nulls": True,
        "real_proper_score_exceeds_all_nulls": True,
    }
    assert a_options_increment_passes(common)
    assert a_disagreement_increment_passes(common)
    assert b_stock_increment_passes(common)
    assert b_route_increment_passes(common)
