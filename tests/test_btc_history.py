"""BTC rollover warm-up regression: request only each frozen reference window."""

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace as NS

from ib_async import Contract, ContractDetails
from ib_async.objects import BarData

from stocker_execution.contracts import signal_future_request
from stocker_execution.rules import NY, reference_summary
from stocker_execution.runtime import Runtime
from test_futures import setup

AT = datetime(2026, 9, 27, 16, 30, tzinfo=UTC)
DAYS = [date(2026, 9, n) for n in (17, 18, 21, 22, 23, 24, 25)]
REFERENCES = DAYS[-5:]


def minute_bars(day, count=540):
    start = datetime.combine(day, time(8), NY).astimezone(UTC)
    return [
        BarData(
            date=start + timedelta(minutes=i),
            open=100 + i * 0.01,
            high=102 + i * 0.01,
            low=99 + i * 0.01,
            close=101 + i * 0.01,
            volume=5 + i % 11,
            average=100.5 + i * 0.01,
        )
        for i in range(count)
    ]


def test_expired_btc_reference_windows_complete_without_large_history_request(
    tmp_path, monkeypatch
):
    async def scenario():
        store, broker, ib = setup(tmp_path, monkeypatch, armed=False)
        runtime = Runtime(broker.config, store, broker)
        monkeypatch.setattr("stocker_execution.runtime.now", lambda: AT)
        a = ContractDetails(
            contract=Contract(
                **{
                    **signal_future_request("BTC").dict(),
                    "conId": 772435574,
                    "lastTradeDateOrContractMonth": "20260925",
                    "localSymbol": "BTCU6",
                    "includeExpired": False,
                }
            ),
            priceMagnifier=1,
        )
        b = ContractDetails(
            contract=Contract(
                **{
                    **signal_future_request("BTC").dict(),
                    "conId": 876880607,
                    "lastTradeDateOrContractMonth": "20261030",
                    "localSymbol": "BTCV6",
                    "includeExpired": False,
                }
            ),
            priceMagnifier=1,
            timeZoneId="UTC",
            tradingHours="20260927:0000-20260927:2359",
        )

        async def details(request):
            return [a, b]

        async def schedule(*args, **kwargs):
            return NS(
                timeZone="UTC",
                sessions=[
                    NS(
                        refDate=d.strftime("%Y%m%d"),
                        startDateTime=d.strftime("%Y%m%d-00:00:00"),
                        endDateTime=d.strftime("%Y%m%d-23:00:00"),
                    )
                    for d in DAYS
                ],
            )

        requests = []

        async def history(c, end, duration, size, what, **kwargs):
            assert what == "TRADES" and kwargs["useRTH"] is False
            if size == "1 day":
                return [
                    NS(date=d, volume=1000 if c.conId == a.contract.conId else 100) for d in DAYS
                ]
            requests.append((c.conId, end, duration))
            if duration == "10 D":
                # Captured broker failure, made immediate and deterministic.
                raise TimeoutError()
            assert duration == "32400 S"
            assert end.astimezone(NY).time() == time(17)
            return minute_bars(end.astimezone(NY).date())

        async def acquire(c, feed, *args, **kwargs):
            return NS(
                value=minute_bars(AT.astimezone(NY).date(), 40)
                if feed == "BARS"
                else NS(marketDataType=1)
            )

        ib.reqContractDetailsAsync = details
        ib.reqHistoricalScheduleAsync = schedule
        ib.reqHistoricalDataAsync = history
        broker.data.acquire = acquire
        state = runtime.markets["BTC"]
        await runtime.monitor(state)
        assert state.reference_problem == ""
        assert len(state.references) == 5
        assert state.references == [
            reference_summary(runtime.convert(minute_bars(day), AT)) for day in REFERENCES
        ]
        assert [cid for cid, _, _ in requests] == [
            a.contract.conId,
            a.contract.conId,
            a.contract.conId,
            a.contract.conId,
            b.contract.conId,
        ]
        assert [end.astimezone(NY).date() for _, end, _ in requests] == REFERENCES
        assert all(duration == "32400 S" for _, _, duration in requests)
        # Completed reference cache prevents re-downloading on same-day recovery.
        await runtime.monitor(state)
        assert len(requests) == 5 and len(state.references) == 5
        assert not ib.trades
        await broker.data.close()
        store.db.close()

    asyncio.run(scenario())
