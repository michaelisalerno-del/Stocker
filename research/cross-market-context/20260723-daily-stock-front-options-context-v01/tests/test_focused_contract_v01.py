from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from stocker_research.broad_conflict_advance_hazard_v02 import (
    candidate_normalized_weights,
)
from stocker_research.daily_stock_front_options_context_v01 import (
    FRONT_MISMATCH_FEATURES,
    iv_excess_15m,
    iv_excess_15m_frame,
    prepare_front_options_raw,
)
from stocker_research.daily_stock_options_context_v0 import (
    permute_bundle_within_slates,
    previous_us_trading_session,
    reject_protected_observations,
    select_daily_options_surface,
    validate_daily_context_chronology,
)
from stocker_research.front_options_soft_regimes_v01 import (
    FRONT_OPTIONS_DIMENSIONS,
    FRONT_OPTIONS_MISSING_INDICATORS,
    FRONT_OPTIONS_RAW_FEATURES,
    apply_front_options_dimensions,
    apply_front_options_regime,
    fit_front_options_dimension_parameters,
    fit_front_options_regime,
)
from stocker_research.stock_options_cross_market_quick_v0 import (
    add_test_a_target,
    fixed_session_bootstrap_multiplicities,
)


def _predecessor_front_row(
    *,
    session: str = "2025-01-06",
    observation: str = "2025-01-03",
) -> dict[str, object]:
    return {
        "symbol": "AAL",
        "session": session,
        "period": "assessment",
        "required_options_date": "2025-01-03",
        "pair_available": True,
        "options_observation_date": observation,
        "previous_close_underlying_price": 100.0,
        "front_expiration_date": "2025-01-17",
        "front_strike": 100.0,
        "front_call_contract_id": "CALL",
        "front_put_contract_id": "PUT",
        "skew_put_contract_id": "PUT95",
        "skew_call_contract_id": "CALL105",
        "previous_close_chain_request_ids": "request-1",
        "atm_iv": 0.45,
        "straddle_mid_pct": 0.022,
        "call_put_iv_gap": -0.10,
        "skew_25d": 0.20,
        "combined_relative_spread": 0.18,
        "iv_minus_realised_20d": 0.15,
        "near_spot_oi_concentration": 0.75,
        "call_put_oi_imbalance": 0.0,
        "skew_missing": 0,
        "oi_concentration_missing": 0,
        "call_put_oi_imbalance_missing": 0,
    }


def test_exact_d_minus_one_front_join_and_same_day_rejection() -> None:
    valid = prepare_front_options_raw(pd.DataFrame([_predecessor_front_row()]))

    assert valid.iloc[0]["options_observation_date"] == "2025-01-03"
    assert set(FRONT_OPTIONS_MISSING_INDICATORS).issubset(valid.columns)
    with pytest.raises(ValueError, match="same-day or future"):
        prepare_front_options_raw(
            pd.DataFrame(
                [
                    _predecessor_front_row(
                        session="2025-01-06",
                        observation="2025-01-06",
                    )
                ]
            )
        )


def test_us_calendar_chronology_uses_exact_previous_trading_session() -> None:
    signal = date(2025, 1, 6)
    previous = previous_us_trading_session(signal)

    assert previous == date(2025, 1, 3)
    validate_daily_context_chronology(
        signal_date=signal,
        stock_information_date=previous,
        options_observation_date=previous,
    )
    with pytest.raises(ValueError, match="exact previous US trading session"):
        validate_daily_context_chronology(
            signal_date=signal,
            stock_information_date=previous,
            options_observation_date=date(2025, 1, 2),
        )


def _option_row(
    option_type: str,
    strike: float,
    contract: str,
    *,
    iv: float,
    delta: float,
    open_interest: float = 100.0,
) -> dict[str, object]:
    return {
        "underlying_symbol": "AAL",
        "trade_date": date(2025, 1, 3),
        "expiration_date": "2025-01-17",
        "dte": 14,
        "option_type": option_type,
        "strike": strike,
        "contract_id": contract,
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


def test_front_pair_atm_iv_straddle_skew_and_open_interest_are_frozen() -> None:
    rows: list[dict[str, object]] = []
    for strike in (95.0, 100.0, 105.0):
        rows.extend(
            [
                _option_row(
                    "call",
                    strike,
                    f"call-{strike}",
                    iv=0.35 if strike == 105.0 else 0.40,
                    delta=0.25 if strike == 105.0 else 0.50,
                ),
                _option_row(
                    "put",
                    strike,
                    f"put-{strike}",
                    iv=0.55 if strike == 95.0 else 0.50,
                    delta=-0.25 if strike == 95.0 else -0.50,
                ),
            ]
        )
    rows.extend(
        [
            _option_row("call", 80.0, "far-call", iv=0.8, delta=0.99),
            _option_row("put", 80.0, "far-put", iv=0.8, delta=-0.01),
        ]
    )

    selected = select_daily_options_surface(
        pd.DataFrame(rows),
        previous_close=100.0,
        realised_volatility_20d=0.30,
    )

    assert selected["pair_available"] is True
    assert selected["front_call_contract_id"] == "call-100.0"
    assert selected["front_put_contract_id"] == "put-100.0"
    assert selected["atm_iv"] == pytest.approx(0.45)
    assert selected["straddle_mid_pct"] == pytest.approx(0.022)
    assert selected["call_put_iv_gap"] == pytest.approx(-0.10)
    assert selected["skew_25d"] == pytest.approx(0.20)
    assert selected["combined_relative_spread"] == pytest.approx(0.4 / 2.2)
    assert selected["near_spot_oi_concentration"] == pytest.approx(0.75)
    assert selected["call_put_oi_imbalance"] == pytest.approx(0.0)
    assert selected["back_expiry_missing"] == 1


def test_front_dimension_imputation_is_fitted_on_2024_only() -> None:
    values = np.arange(4.0)
    development = pd.DataFrame(
        {
            "symbol": "AAL",
            "session": pd.date_range("2024-01-02", periods=4),
            **{feature: values for feature in FRONT_OPTIONS_RAW_FEATURES},
            **{indicator: 0 for indicator in FRONT_OPTIONS_MISSING_INDICATORS},
        }
    )
    parameters = fit_front_options_dimension_parameters(development)
    assessment = development.iloc[[0]].assign(
        session=pd.Timestamp("2025-01-02"),
        skew_25d=np.nan,
        near_spot_oi_concentration=np.nan,
        call_put_oi_imbalance=np.nan,
    )

    transformed = apply_front_options_dimensions(assessment, parameters)

    assert np.isfinite(transformed.loc[:, list(FRONT_OPTIONS_DIMENSIONS)].to_numpy(float)).all()
    assert parameters.fitted_period == "development_2024_only"


def test_front_regime_ids_are_canonical_lexicographic_states() -> None:
    rows: list[dict[str, object]] = []
    for component, center in enumerate((-6.0, -2.0, 2.0, 6.0)):
        for repeat in range(12):
            rows.append(
                {
                    "symbol": f"S{repeat % 8}",
                    "session": f"2024-{component + 1:02d}-{repeat + 1:02d}",
                    **{dimension: center + repeat * 0.01 for dimension in FRONT_OPTIONS_DIMENSIONS},
                    **{indicator: 0 for indicator in FRONT_OPTIONS_MISSING_INDICATORS},
                }
            )
    development = pd.DataFrame(rows)

    fitted = fit_front_options_regime(development)
    assigned = apply_front_options_regime(development, fitted)

    keys = [
        tuple(
            centroid[dimension]
            for dimension in (
                "front_options_implied_tension",
                "front_options_premium_richness",
                "front_options_downside_asymmetry",
                "front_options_liquidity_stress",
                "front_options_positioning_concentration",
            )
        )
        for centroid in fitted.canonical_centroids
    ]
    assert keys == sorted(keys)
    probabilities = [f"front_options_regime_p_{regime}" for regime in range(4)]
    assert np.allclose(assigned[probabilities].sum(axis=1), 1.0)


def test_clean_completion_target_excludes_next_bar_and_imminent_rows() -> None:
    frame = pd.DataFrame(
        {
            "first_completion_lead": [2, 3, 1, 2, 0],
            "registered_completion_next_1_bar": [0, 0, 1, 0, 0],
            "any_prefix_one_transition_from_completion": [0, 0, 0, 1, 0],
        }
    )

    result = add_test_a_target(frame)

    assert result["clean_advance_eligible"].tolist() == [1, 1, 0, 0, 1]
    assert result["registered_completion_clean_bars_2_or_3"].iloc[:2].tolist() == [
        1.0,
        1.0,
    ]
    assert math.isnan(result.iloc[2]["registered_completion_clean_bars_2_or_3"])
    assert math.isnan(result.iloc[3]["registered_completion_clean_bars_2_or_3"])


def test_fifteen_minute_iv_excess_target_uses_expected_absolute_movement() -> None:
    result = iv_excess_15m(
        entry_price=100.0,
        close_15m=101.0,
        atm_iv=0.40,
    )
    expected = 0.40 * math.sqrt(15.0 / (252.0 * 390.0)) * math.sqrt(2.0 / math.pi)

    assert result["absolute_log_return_15m"] == pytest.approx(abs(math.log(1.01)))
    assert result["iv_expected_absolute_15m"] == pytest.approx(expected)
    assert result["movement_exceeds_prior_close_iv_15m"] == 1
    assert "option_pnl" not in result


def test_fifteen_minute_iv_excess_vectorization_accepts_an_empty_branch() -> None:
    result = iv_excess_15m_frame(entry_price=[], close_15m=[], atm_iv=[])

    assert result.empty
    assert "movement_exceeds_prior_close_iv_15m" in result


def test_candidate_weights_normalise_each_session_across_stocks() -> None:
    frame = pd.DataFrame(
        {
            "period": ["assessment"] * 5,
            "session": ["2025-01-02"] * 5,
            "symbol": ["A", "A", "B", "B", "B"],
            "advance_eligible": [1] * 5,
        }
    )

    weighted = candidate_normalized_weights(frame)

    assert weighted.loc[weighted["symbol"].eq("A"), "row_weight"].tolist() == [
        0.25,
        0.25,
    ]
    assert weighted.loc[weighted["symbol"].eq("B"), "row_weight"].tolist() == pytest.approx(
        [1.0 / 6.0] * 3
    )
    assert weighted["row_weight"].sum() == pytest.approx(1.0)


def test_session_bootstrap_has_ten_whole_session_draws() -> None:
    sessions = pd.Series(["a", "a", "b", "b", "c"])

    draws = fixed_session_bootstrap_multiplicities(sessions, draws=10, seed=20260723)

    assert len(draws) == 10
    assert all(draw[0] == draw[1] and draw[2] == draw[3] for draw in draws)
    with pytest.raises(ValueError, match="exactly 10"):
        fixed_session_bootstrap_multiplicities(sessions, draws=9)


def _permutation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["development"] * 3,
            "session": ["2024-01-02"] * 3,
            "checkpoint": [6] * 3,
            "symbol": ["A", "B", "C"],
            "outcome": [0, 1, 0],
            "front_1": [1.0, 2.0, 3.0],
            "front_2": [10.0, 20.0, 30.0],
            "stock_1": [100.0, 200.0, 300.0],
            "route_resolution_state": [
                "BROAD_CONFLICT",
                "LOW_ROUTE_SUPPORT",
                "NARROWING",
            ],
        }
    )


def test_front_options_bundle_permutation_keeps_the_bundle_intact() -> None:
    frame = _permutation_frame()

    permuted = permute_bundle_within_slates(
        frame,
        columns=("front_1", "front_2"),
        seed=20260723,
    )

    original = set(frame[["front_1", "front_2"]].itertuples(index=False, name=None))
    observed = set(permuted[["front_1", "front_2"]].itertuples(index=False, name=None))
    assert observed == original
    assert permuted["outcome"].equals(frame["outcome"])
    assert permuted["stock_1"].equals(frame["stock_1"])


def test_stock_structure_bundle_permutation_preserves_front_options() -> None:
    frame = _permutation_frame()

    permuted = permute_bundle_within_slates(
        frame,
        columns=("stock_1", "route_resolution_state"),
        seed=20260726,
    )

    original = set(frame[["stock_1", "route_resolution_state"]].itertuples(index=False, name=None))
    observed = set(
        permuted[["stock_1", "route_resolution_state"]].itertuples(index=False, name=None)
    )
    assert observed == original
    assert permuted["front_1"].equals(frame["front_1"])
    assert permuted["front_2"].equals(frame["front_2"])


def test_protected_option_observation_is_rejected_but_expiration_is_not() -> None:
    with pytest.raises(ValueError, match="protected"):
        reject_protected_observations(
            pd.DataFrame({"trade_date": ["2025-08-23"]}),
            date_columns=("trade_date",),
        )
    reject_protected_observations(
        pd.DataFrame(
            {
                "trade_date": ["2025-08-22"],
                "expiration_date": ["2025-10-17"],
            }
        ),
        date_columns=("trade_date",),
    )


def test_front_only_authoritative_surface_has_exactly_five_mismatches() -> None:
    assert tuple(FRONT_MISMATCH_FEATURES) == (
        "mismatch_compression_vs_front_iv",
        "mismatch_daily_volatility_vs_front_iv",
        "mismatch_route_vs_front_premium",
        "mismatch_direction_agreement",
        "mismatch_complacent_broad_conflict",
    )
