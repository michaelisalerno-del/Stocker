"""Regression fixtures from the first non-transmitting futures server cutover."""

import asyncio
from datetime import date, timedelta
from types import SimpleNamespace as NS

import pytest
from ib_async import Contract, ContractDetails
from ib_async.objects import BarData

from stocker_execution.config import RULE_VERSION
from stocker_execution.contracts import (
    SIGNAL_PRODUCTS,
    completed_trade_dates,
    signal_future_request,
    verified_signal_futures,
)
from stocker_execution.runtime import Runtime
from test_futures import AT, setup


@pytest.mark.parametrize("market", SIGNAL_PRODUCTS)
def test_standard_signal_identity_rejects_other_products(market):
    request = signal_future_request(market)
    good = ContractDetails(contract=Contract(**{**request.dict(), "conId": 12}), priceMagnifier=1)
    bad = ContractDetails(
        contract=Contract(**{**request.dict(), "conId": 13, "multiplier": "1"}),
        priceMagnifier=1,
    )
    assert verified_signal_futures(market, [bad, good, good]) == [good]
    with pytest.raises(ValueError, match="SIGNAL_PRODUCT_IDENTITY_UNVERIFIED"):
        verified_signal_futures(market, [bad])


def test_ib_bitcoin_symbol_and_silver_micro_ambiguity():
    btc = signal_future_request("BTC")
    assert (btc.symbol, btc.tradingClass, btc.multiplier) == ("BRR", "BTC", "5")
    si = signal_future_request("SI")
    standard = ContractDetails(
        contract=Contract(**{**si.dict(), "conId": 535526329, "localSymbol": "SIZ6"}),
        priceMagnifier=1,
    )
    micro = ContractDetails(
        contract=Contract(
            **{
                **si.dict(),
                "conId": 906389986,
                "localSymbol": "SILV6",
                "multiplier": "1000",
                "tradingClass": "SIL",
            }
        ),
        priceMagnifier=1,
    )
    assert verified_signal_futures("SI", [micro, standard]) == [standard]


def test_missing_reference_volume_keeps_core_monitoring_but_blocks_strategy(tmp_path, monkeypatch):
    async def scenario():
        store, broker, ib = setup(tmp_path, monkeypatch)
        runtime = Runtime(broker.config, store, broker)
        monkeypatch.setattr("stocker_execution.runtime.now", lambda: AT)
        state = runtime.markets["NQ"]
        contract = Contract(
            **{
                **signal_future_request("NQ").dict(),
                "conId": 563947726,
                "lastTradeDateOrContractMonth": "20261218",
                "localSymbol": "NQZ6",
            }
        )
        detail = ContractDetails(
            contract=contract,
            priceMagnifier=1,
            timeZoneId="UTC",
            tradingHours="20260928:0000-20260928:2359",
        )

        async def details(request):
            assert request.tradingClass == "NQ" and request.multiplier == "20"
            return [detail]

        async def schedule(*args, **kwargs):
            return NS(
                timeZone="UTC",
                sessions=[
                    NS(
                        refDate=f"202609{day}",
                        startDateTime=f"202609{day}-00:00:00",
                        endDateTime=f"202609{day}-23:00:00",
                    )
                    for day in (23, 24, 25)
                ],
            )

        async def history(*args, **kwargs):
            # Current rollover is known; historical reference candidate volume is unavailable.
            return [NS(date=date(2026, 9, 25), volume=1000)]

        acquired = []
        bars = [
            BarData(
                date=AT - timedelta(minutes=31 - i),
                open=100 + i,
                high=102 + i,
                low=99 + i,
                close=101 + i,
                volume=10,
                average=101 + i,
            )
            for i in range(32)
        ]

        async def acquire(c, feed, *args, **kwargs):
            acquired.append((c.conId, feed))
            return NS(value=bars if feed == "BARS" else NS(marketDataType=1))

        ib.reqContractDetailsAsync = details
        ib.reqHistoricalScheduleAsync = schedule
        runtime.history = history
        broker.data.acquire = acquire
        # Old unconstrained-product references must not be reused.
        store.set_meta(f"futures_reference:{RULE_VERSION}:NQ:2026-09-28", [{}] * 5)
        await runtime.monitor(state)
        assert acquired == [(563947726, "BARS"), (563947726, "QUOTE")]
        assert state.live_since == AT and state.references == []
        assert (
            state.problem == "REFERENCE_HISTORY_BLOCKED:ROLLOVER_PRIOR_SESSION_VOLUME_UNAVAILABLE"
        )
        assert state.last_update == AT and len(state.bars) == 31
        await runtime.decisions()
        assert not ib.trades
        assert not runtime.overview()["markets"][4]["entry_enabled"]
        store.db.close()

    asyncio.run(scenario())


def test_weekend_crypto_trade_date_requires_all_maintenance_fragments_complete():
    # Actual IB schedule shape observed Sunday 2026-09-27, Central daylight time.
    schedule = NS(
        timeZone="US/Central",
        sessions=[
            NS(refDate="20260925", endDateTime="20260925-16:00:00"),
            NS(refDate="20260928", endDateTime="20260926-02:00:00"),
            NS(refDate="20260928", endDateTime="20260928-16:00:00"),
        ],
    )
    assert completed_trade_dates(schedule, AT - timedelta(days=1)) == [date(2026, 9, 25)]
    assert completed_trade_dates(schedule, AT + timedelta(hours=9)) == [
        date(2026, 9, 25),
        date(2026, 9, 28),
    ]
