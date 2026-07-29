from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from stocker_research.broad_conflict_options_iv_screen_v0 import (
    ChronologyError,
    calculate_optional_option_features,
    calculate_primary_option_features,
    compute_underlying_movement_outcomes,
    iv_movement_approximations,
    previous_trading_session,
    select_primary_atm_pair,
    split_boundary_is_ambiguous,
    validate_exact_previous_session_join,
    verify_structural_reconstruction,
)


def option_row(
    option_type: str,
    *,
    expiry: str = "2025-01-17",
    dte: int = 15,
    strike: float = 100.0,
    bid: float = 4.0,
    ask: float = 4.4,
    iv: float = 0.4,
    oi: int = 100,
    contract: str | None = None,
    delta: float | None = None,
) -> dict[str, object]:
    side = "C" if option_type == "call" else "P"
    return {
        "underlying_symbol": "AAPL",
        "trade_date": date(2025, 1, 2),
        "expiration_date": date.fromisoformat(expiry),
        "dte": dte,
        "strike": strike,
        "option_type": option_type,
        "contract_id": contract or f"AAPL-{expiry}-{strike:g}-{side}",
        "bid": bid,
        "ask": ask,
        "midpoint": (bid + ask) / 2.0,
        "implied_volatility": iv,
        "open_interest": oi,
        "volume": 10,
        "delta": delta,
    }


def test_previous_trading_session_respects_weekends_and_us_holidays() -> None:
    assert previous_trading_session(date(2024, 1, 16)) == date(2024, 1, 12)
    assert previous_trading_session(date(2024, 7, 5)) == date(2024, 7, 3)


@pytest.mark.parametrize(
    "actual",
    [date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 2)],
)
def test_same_day_future_and_older_chain_joins_are_rejected(actual: date) -> None:
    with pytest.raises(ChronologyError, match="exact previous trading session"):
        validate_exact_previous_session_join(
            signal_date=date(2025, 1, 6),
            required_options_date=date(2025, 1, 3),
            actual_options_date=actual,
        )


def test_split_between_prior_close_and_signal_is_ambiguous() -> None:
    assert split_boundary_is_ambiguous(
        options_date=date(2024, 8, 7),
        signal_date=date(2024, 8, 8),
        split_dates={date(2024, 8, 8)},
    )
    assert not split_boundary_is_ambiguous(
        options_date=date(2024, 8, 8),
        signal_date=date(2024, 8, 9),
        split_dates={date(2024, 8, 8)},
    )


def test_nearest_eligible_expiry_does_not_fall_back_when_pair_quality_fails() -> None:
    chain = pd.DataFrame(
        [
            option_row("call", expiry="2025-01-10", dte=8, oi=5),
            option_row("put", expiry="2025-01-10", dte=8, oi=5),
            option_row("call", expiry="2025-01-17", dte=15, oi=500),
            option_row("put", expiry="2025-01-17", dte=15, oi=500),
        ]
    )

    selection = select_primary_atm_pair(chain, previous_close=100.0)

    assert selection.expiration_date == date(2025, 1, 10)
    assert selection.available is False
    assert selection.reason == "selected_pair_open_interest_below_10"


def test_common_strike_ties_follow_frozen_open_interest_then_spread_order() -> None:
    lower = 50.0
    upper = 200.0
    rows = [
        option_row("call", strike=lower, oi=20, bid=4.0, ask=5.0, contract="lower-call"),
        option_row("put", strike=lower, oi=20, bid=4.0, ask=5.0, contract="lower-put"),
        option_row("call", strike=upper, oi=40, bid=4.0, ask=4.5, contract="upper-call"),
        option_row("put", strike=upper, oi=40, bid=4.0, ask=4.5, contract="upper-put"),
    ]

    selection = select_primary_atm_pair(pd.DataFrame(rows), previous_close=100.0)

    assert selection.available is True
    assert selection.strike == pytest.approx(upper)
    assert selection.call_contract_id == "upper-call"
    assert selection.put_contract_id == "upper-put"

    spread_rows = [
        option_row("call", strike=lower, oi=40, bid=4.0, ask=5.0),
        option_row("put", strike=lower, oi=40, bid=4.0, ask=5.0),
        option_row("call", strike=upper, oi=40, bid=4.0, ask=4.4),
        option_row("put", strike=upper, oi=40, bid=4.0, ask=4.4),
    ]
    spread_selection = select_primary_atm_pair(pd.DataFrame(spread_rows), previous_close=100.0)
    assert spread_selection.strike == upper

    iv_rows = [
        option_row("call", strike=lower, oi=40, iv=0.30),
        option_row("put", strike=lower, oi=40, iv=0.50),
        option_row("call", strike=upper, oi=40, iv=0.40),
        option_row("put", strike=upper, oi=40, iv=0.41),
    ]
    iv_selection = select_primary_atm_pair(pd.DataFrame(iv_rows), previous_close=100.0)
    assert iv_selection.strike == upper

    strike_rows = [
        option_row("call", strike=lower, oi=40),
        option_row("put", strike=lower, oi=40),
        option_row("call", strike=upper, oi=40),
        option_row("put", strike=upper, oi=40),
    ]
    strike_selection = select_primary_atm_pair(pd.DataFrame(strike_rows), previous_close=100.0)
    assert strike_selection.strike == lower

    contract_rows = [
        option_row("call", contract="b-call"),
        option_row("call", contract="a-call"),
        option_row("put", contract="b-put"),
        option_row("put", contract="a-put"),
    ]
    contract_selection = select_primary_atm_pair(pd.DataFrame(contract_rows), previous_close=100.0)
    assert contract_selection.call_contract_id == "a-call"
    assert contract_selection.put_contract_id == "a-put"


def test_pair_quality_rules_apply_to_each_contract_separately() -> None:
    chain = pd.DataFrame(
        [
            option_row("call", bid=0.0, ask=0.0),
            option_row("put", bid=1.0, ask=3.1),
        ]
    )

    selection = select_primary_atm_pair(chain, previous_close=100.0)

    assert selection.available is False
    assert selection.reason == "selected_pair_call_midpoint_not_positive"


def test_pair_quality_rejects_implausible_selected_greeks() -> None:
    call = option_row("call")
    put = option_row("put")
    call["delta"] = 1.2

    selection = select_primary_atm_pair(pd.DataFrame([call, put]), previous_close=100.0)

    assert selection.available is False
    assert selection.reason == "selected_pair_call_delta_implausible"


def test_atm_pair_features_and_iv_time_scaling_match_worked_values() -> None:
    chain = pd.DataFrame(
        [
            option_row("call", bid=4.0, ask=4.4, iv=0.42, oi=120, delta=0.52),
            option_row("put", bid=3.8, ask=4.2, iv=0.38, oi=80, delta=-0.48),
        ]
    )
    selection = select_primary_atm_pair(chain, previous_close=100.0)
    features = calculate_primary_option_features(selection, previous_close=100.0)
    approximation = iv_movement_approximations(features["atm_iv"])

    assert features["atm_iv"] == pytest.approx(0.4)
    assert features["call_put_iv_gap"] == pytest.approx(0.04)
    assert features["straddle_mid"] == pytest.approx(8.2)
    assert features["straddle_mid_pct"] == pytest.approx(0.082)
    assert features["combined_open_interest"] == 200
    assert features["log1p_combined_open_interest"] == pytest.approx(math.log1p(200))
    assert features["scaled_straddle_move_15m"] == pytest.approx(0.082 * math.sqrt(15 / (15 * 390)))
    assert approximation["iv_sigma_15m"] == pytest.approx(0.004941662111074008)
    assert approximation["iv_expected_absolute_15m"] == pytest.approx(0.003942875903130446)


def test_optional_skew_and_back_term_structure_do_not_control_pair_coverage() -> None:
    chain = pd.DataFrame(
        [
            option_row("call", iv=0.40, delta=0.52),
            option_row("put", iv=0.42, delta=-0.48),
            option_row("call", strike=110, iv=0.35, delta=0.27, contract="25c"),
            option_row("put", strike=90, iv=0.50, delta=-0.24, contract="25p"),
            option_row(
                "call",
                expiry="2025-03-03",
                dte=60,
                strike=100,
                iv=0.45,
                contract="back-c",
            ),
            option_row(
                "put",
                expiry="2025-03-03",
                dte=60,
                strike=100,
                iv=0.47,
                contract="back-p",
            ),
        ]
    )
    front = select_primary_atm_pair(chain, previous_close=100.0)

    optional = calculate_optional_option_features(
        chain, front_selection=front, previous_close=100.0
    )

    assert optional["skew_25d"] == pytest.approx(0.15)
    assert optional["skew_25d_missing"] == 0
    assert optional["term_structure"] == pytest.approx(0.05)
    assert optional["term_structure_missing"] == 0

    missing = calculate_optional_option_features(
        chain.loc[chain["dte"].eq(15) & chain["strike"].eq(100)],
        front_selection=front,
        previous_close=100.0,
    )
    assert math.isnan(missing["skew_25d"])
    assert missing["skew_25d_missing"] == 1
    assert math.isnan(missing["term_structure"])
    assert missing["term_structure_missing"] == 1


def test_fifteen_minute_underlying_outcome_uses_only_three_future_bars() -> None:
    structural = pd.DataFrame(
        [
            {
                "row_id": "row-1",
                "symbol": "AAPL",
                "session": "2025-01-06",
                "checkpoint_bar_ordinal_zero_based": 0,
            }
        ]
    )
    bars = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "session": "2025-01-06",
                "bar_ordinal": 0,
                "open": 99,
                "high": 101,
                "low": 98,
                "close": 100,
            },
            {
                "symbol": "AAPL",
                "session": "2025-01-06",
                "bar_ordinal": 1,
                "open": 100,
                "high": 103,
                "low": 99,
                "close": 102,
            },
            {
                "symbol": "AAPL",
                "session": "2025-01-06",
                "bar_ordinal": 2,
                "open": 102,
                "high": 104,
                "low": 100,
                "close": 101,
            },
            {
                "symbol": "AAPL",
                "session": "2025-01-06",
                "bar_ordinal": 3,
                "open": 101,
                "high": 106,
                "low": 100.5,
                "close": 105,
            },
            {
                "symbol": "AAPL",
                "session": "2025-01-06",
                "bar_ordinal": 4,
                "open": 105,
                "high": 150,
                "low": 50,
                "close": 120,
            },
        ]
    )
    bars["bar_start_timestamp"] = pd.date_range(
        "2025-01-06T14:30:00Z", periods=len(bars), freq="5min"
    )
    bars["bar_complete_timestamp"] = bars["bar_start_timestamp"] + pd.Timedelta(minutes=5)

    outcome = compute_underlying_movement_outcomes(structural, bars).iloc[0]

    assert outcome["entry_price"] == 100
    assert outcome["entry_bar_start_timestamp"] == bars.iloc[1]["bar_start_timestamp"]
    assert (
        outcome["primary_horizon_last_bar_complete_timestamp"]
        == bars.iloc[3]["bar_complete_timestamp"]
    )
    assert outcome["absolute_log_return_15m"] == pytest.approx(abs(math.log(1.05)))
    assert outcome["realised_range_15m"] == pytest.approx(math.log(106 / 99))
    assert outcome["maximum_absolute_excursion_15m"] == pytest.approx(math.log(1.06))
    expected_rv = math.log(102 / 100) ** 2 + math.log(101 / 102) ** 2 + math.log(105 / 101) ** 2
    assert outcome["realised_variance_15m"] == pytest.approx(expected_rv)


def test_structural_reconstruction_requires_exact_identity_state_and_features() -> None:
    reference = pd.DataFrame([{"row_id": "x", "route_resolution_state": "OTHER", "feature": 0.25}])
    identical = reference.copy()
    changed = reference.assign(feature=0.2500001)

    passed = verify_structural_reconstruction(reference, identical, feature_columns=["feature"])
    failed = verify_structural_reconstruction(reference, changed, feature_columns=["feature"])

    assert passed["row_identity_mismatches"] == 0
    assert passed["route_state_mismatches"] == 0
    assert passed["maximum_difference"] == 0.0
    assert failed["maximum_difference"] > 1e-12
