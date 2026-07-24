from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from stocker_prospective.market_data import MarketDataType
from stocker_prospective.shadow import (
    OptionExecutableQuote,
    ShadowStructureType,
    value_shadow_structures,
)

NOW = datetime(2026, 7, 24, 14, 35, tzinfo=UTC)


def quote(
    contract_id: int,
    right: str,
    strike: float,
    *,
    bid: float | None,
    ask: float | None,
    at: datetime = NOW,
    data_type: MarketDataType = MarketDataType.LIVE,
    stale: bool = False,
) -> OptionExecutableQuote:
    return OptionExecutableQuote(
        contract_id=contract_id,
        right=right,
        strike=strike,
        bid=bid,
        ask=ask,
        bid_size=10.0,
        ask_size=12.0,
        provider_timestamp=at,
        receive_timestamp=at,
        market_data_type=data_type,
        multiplier=100,
        stale=stale,
    )


def valid_surfaces() -> tuple[dict[tuple[str, float], OptionExecutableQuote], ...]:
    entry = {
        ("C", 95.0): quote(1, "C", 95.0, bid=7.8, ask=8.0),
        ("C", 100.0): quote(2, "C", 100.0, bid=4.8, ask=5.0),
        ("C", 105.0): quote(3, "C", 105.0, bid=2.0, ask=2.2),
        ("P", 95.0): quote(4, "P", 95.0, bid=1.8, ask=2.0),
        ("P", 100.0): quote(5, "P", 100.0, bid=4.0, ask=4.2),
        ("P", 105.0): quote(6, "P", 105.0, bid=7.1, ask=7.3),
    }
    exit_at = NOW + timedelta(minutes=5)
    exit_quotes = {
        ("C", 95.0): quote(1, "C", 95.0, bid=8.0, ask=8.2, at=exit_at),
        ("C", 100.0): quote(2, "C", 100.0, bid=6.0, ask=6.2, at=exit_at),
        ("C", 105.0): quote(3, "C", 105.0, bid=2.5, ask=2.7, at=exit_at),
        ("P", 95.0): quote(4, "P", 95.0, bid=1.0, ask=1.2, at=exit_at),
        ("P", 100.0): quote(5, "P", 100.0, bid=3.0, ask=3.2, at=exit_at),
        ("P", 105.0): quote(6, "P", 105.0, bid=6.1, ask=6.3, at=exit_at),
    }
    return entry, exit_quotes


def test_all_five_structures_use_executable_sides_and_multiplier() -> None:
    entry, exit_quotes = valid_surfaces()

    results = value_shadow_structures(
        entry_quotes=entry,
        exit_quotes=exit_quotes,
        atm_strike=100.0,
        lower_strike=95.0,
        upper_strike=105.0,
        target_timestamp=NOW + timedelta(minutes=5),
        maximum_capture_lag=timedelta(seconds=15),
        estimated_fee_per_contract=0.65,
    )
    by_type = {item.structure_type: item for item in results}

    call = by_type[ShadowStructureType.LONG_ATM_CALL]
    assert call.entry_debit == 5.0
    assert call.exit_credit == 6.0
    assert call.gross_pnl == 100.0
    assert call.gross_return_on_debit == 0.2
    assert call.estimated_fees == 1.3

    put = by_type[ShadowStructureType.LONG_ATM_PUT]
    assert put.entry_debit == 4.2
    assert put.exit_credit == 3.0

    straddle = by_type[ShadowStructureType.LONG_ATM_STRADDLE]
    assert straddle.entry_debit == 9.2
    assert straddle.exit_credit == 9.0
    assert len(straddle.legs) == 2

    call_spread = by_type[ShadowStructureType.CALL_DEBIT_SPREAD]
    assert call_spread.entry_debit == 3.0  # long ask - short bid
    assert call_spread.exit_credit == 3.3  # long bid - short ask
    assert [leg.side for leg in call_spread.legs] == ["long", "short"]

    put_spread = by_type[ShadowStructureType.PUT_DEBIT_SPREAD]
    assert put_spread.entry_debit == pytest.approx(2.4)  # long ask - short bid
    assert put_spread.exit_credit == pytest.approx(1.8)  # long bid - short ask
    assert all(item.rejection_reason is None for item in results)


def test_missing_leg_invalid_debit_stale_delayed_crossed_and_lag_are_rejected() -> None:
    entry, exit_quotes = valid_surfaces()
    entry.pop(("P", 95.0))
    entry[("C", 105.0)] = quote(3, "C", 105.0, bid=5.1, ask=5.2)
    exit_quotes[("C", 100.0)] = quote(
        2,
        "C",
        100.0,
        bid=6.0,
        ask=5.9,
        at=NOW + timedelta(minutes=5),
    )
    exit_quotes[("P", 100.0)] = quote(
        5,
        "P",
        100.0,
        bid=3.0,
        ask=3.2,
        at=NOW + timedelta(minutes=5),
        data_type=MarketDataType.DELAYED,
    )

    results = value_shadow_structures(
        entry_quotes=entry,
        exit_quotes=exit_quotes,
        atm_strike=100.0,
        lower_strike=95.0,
        upper_strike=105.0,
        target_timestamp=NOW + timedelta(minutes=5),
        maximum_capture_lag=timedelta(seconds=15),
    )
    reasons = {item.structure_type: item.rejection_reason for item in results}

    assert reasons[ShadowStructureType.LONG_ATM_CALL] == "crossed_market"
    assert reasons[ShadowStructureType.LONG_ATM_PUT] == "blocked_non_live_market_data"
    assert reasons[ShadowStructureType.LONG_ATM_STRADDLE] == "crossed_market"
    assert reasons[ShadowStructureType.CALL_DEBIT_SPREAD] == "invalid_or_nonpositive_debit"
    assert reasons[ShadowStructureType.PUT_DEBIT_SPREAD] == "missing_leg"

    entry2, exit2 = valid_surfaces()
    exit2[("C", 100.0)] = quote(
        2,
        "C",
        100.0,
        bid=6.0,
        ask=6.2,
        at=NOW + timedelta(minutes=5, seconds=16),
        stale=True,
    )
    delayed = value_shadow_structures(
        entry_quotes=entry2,
        exit_quotes=exit2,
        atm_strike=100.0,
        lower_strike=95.0,
        upper_strike=105.0,
        target_timestamp=NOW + timedelta(minutes=5),
        maximum_capture_lag=timedelta(seconds=15),
    )
    assert delayed[0].rejection_reason == "stale_quote"
    assert delayed[0].entry_debit is None
    assert delayed[0].exit_credit is None


def test_missing_values_are_not_coerced_to_zero_or_paper_fills() -> None:
    entry, exit_quotes = valid_surfaces()
    entry[("C", 100.0)] = quote(2, "C", 100.0, bid=4.8, ask=None)

    results = value_shadow_structures(
        entry_quotes=entry,
        exit_quotes=exit_quotes,
        atm_strike=100.0,
        lower_strike=95.0,
        upper_strike=105.0,
        target_timestamp=NOW + timedelta(minutes=5),
        maximum_capture_lag=timedelta(seconds=15),
    )

    assert results[0].rejection_reason == "missing_quote"
    assert results[0].entry_debit is None
    assert all("paper" not in field for field in type(results[0]).model_fields)
