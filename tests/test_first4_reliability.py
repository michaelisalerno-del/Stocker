"""Offline regressions: the real ib_async wrapper, never a broker socket."""

import asyncio
from datetime import date, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from ib_async import Contract, ContractDetails, Order, OrderStatus, Trade
from ib_async.wrapper import RequestError

from stocker_execution.first4_runtime import Runtime
from stocker_execution.first4_store import Store
from test_first4 import CLOSE, OPEN, armed_config, broker, candidate, prepare_exit


def scanner_runtime(tmp_path, mode):
    runtime = Runtime(armed_config(), Store(tmp_path / "scan.sqlite"))
    ib = runtime.broker.ib
    ib.client.getReqId = Mock(return_value=7)
    ib.client.cancelScannerSubscription = Mock()
    runtime.execute = Mock()

    def response(req_id, *args):
        def deliver():
            if mode in {"rows", "partial"}:
                ib.wrapper.scannerData(
                    req_id,
                    0,
                    ContractDetails(contract=Contract(symbol="A", conId=1)),
                    "",
                    "",
                    "",
                    "",
                )
            if mode in {"error", "partial"}:
                ib.wrapper.error(req_id, 162, "scanner failed", "")
            elif mode != "hang":
                ib.wrapper.scannerDataEnd(req_id)

        asyncio.get_running_loop().call_soon(deliver)

    ib.client.reqScannerSubscription = response

    async def qualified(*args):
        return candidate("A", 1, None)

    runtime.candidate = qualified
    return runtime


@pytest.mark.parametrize("mode", ["empty", "rows", "error", "partial", "hang", "cancel"])
def test_scanner_completion_and_cleanup(tmp_path, mode):
    async def run():
        runtime = scanner_runtime(tmp_path, "hang" if mode == "cancel" else mode)

        async def scan():
            await runtime.scan("2025-07-21", OPEN, CLOSE, OPEN, OPEN)

        if mode == "cancel":
            task = asyncio.create_task(scan())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == "hang":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(scan(), 0.01)
        elif mode in {"error", "partial"}:
            with pytest.raises(Exception, match="SCANNER"):
                await scan()
        else:
            await scan()
            assert runtime.store.rows("sessions")[0]["last_clock"] == OPEN.isoformat()
            assert len(runtime.store.rows("events")) == int(mode == "rows")
        if mode not in {"empty", "rows"}:
            assert runtime.store.rows("events") == []
            assert runtime.store.rows("sessions")[0]["blocked"]
            runtime.broker.ib.client.reqScannerSubscription = lambda req_id, *a: (
                asyncio.get_running_loop().call_soon(
                    runtime.broker.ib.wrapper.scannerDataEnd, req_id
                )
            )
            with pytest.raises(ValueError):
                await scan()
        ib = runtime.broker.ib
        assert not ib.wrapper.reqId2Subscriber
        assert not ib.wrapper._futures
        assert not ib.wrapper._results
        assert ib.client.cancelScannerSubscription.called
        assert ib.RaiseRequestErrors is False

    asyncio.run(run())


def test_upstream_loss_revokes_locally_connected_readiness(tmp_path):
    b = broker(tmp_path)
    b.error(-1, 1100, "upstream lost")
    assert b.ib.isConnected()
    assert not b.reconciled
    with pytest.raises(ValueError):
        b.guard()


def test_stale_reconciliation_cannot_restore_readiness(tmp_path):
    b = broker(tmp_path)

    async def positions():
        b.disconnected()
        return []

    b.ib.reqPositionsAsync.side_effect = positions
    with pytest.raises(ValueError):
        asyncio.run(b.reconcile())
    assert not b.reconciled


def test_zero_exposure_with_working_exit_is_not_closed(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b, (0, 0))
    b.store.reserve_order(event, "EXIT", 2, payload)
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: CLOSE)
    asyncio.run(b.close_due())
    assert b.store.rows("events")[0]["outcome"] not in {"CLOSED", "ENTRY_UNFILLED"}
    assert not b.ib.placeOrder.called


@pytest.mark.parametrize("restore", [1101, 1102])
def test_restore_requires_reconciliation_and_keeps_gap(tmp_path, monkeypatch, restore):
    b = broker(tmp_path)
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN)
    b.store.observe("2025-07-21", OPEN, CLOSE, [])
    b.opening_verified_session = "2025-07-21"
    b.error(-1, 1100, "lost")
    b.error(-1, restore, "restored")
    assert not b.reconciled
    asyncio.run(b.reconcile())
    assert b.reconciled and not b.entries_armed()
    assert b.opening_verified_session is None
    assert b.store.rows("sessions")[0]["blocked"]
    for code in [2104, 2106, 2108, 2158]:
        b.error(-1, code, "information")
        assert b.reconciled


def test_repeated_closed_management_does_not_rewrite(tmp_path, monkeypatch):
    b = broker(tmp_path)
    prepare_exit(b, (0, 0))
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: CLOSE)
    asyncio.run(b.close_due())
    writes = b.store.db.total_changes
    asyncio.run(b.close_due())
    assert b.store.db.total_changes == writes


@pytest.mark.parametrize(
    "mode", ["invalid", "empty", "all_invalid", "ambiguous", "timeout", "cancel"]
)
def test_contract_candidates_and_request_cleanup(tmp_path, mode):
    async def run():
        runtime = Runtime(armed_config(), Store(tmp_path / "contracts.sqlite"))
        b, ib = runtime.broker, runtime.broker.ib
        ib.client.getReqId = Mock(side_effect=range(1, 100))
        b.chain = AsyncMock(
            return_value=[
                NS(
                    exchange="SMART",
                    tradingClass="ABC",
                    multiplier="100",
                    expirations={"20250722", "20250723"},
                    strikes=[97, 99, 101, 103],
                )
            ]
        )

        def request(req_id, contract):
            def deliver():
                if contract.lastTradeDateOrContractMonth == "20250723" or mode == "all_invalid":
                    if mode in {"timeout", "cancel"}:
                        return
                    if mode != "empty":
                        ib.wrapper.error(
                            req_id,
                            200,
                            "ambiguous contract"
                            if mode == "ambiguous"
                            else "No security definition has been found for the request",
                            "",
                        )
                        return
                else:
                    contract.conId = 101 if contract.right == "P" else 102
                    contract.localSymbol = (
                        f"ABC   {contract.lastTradeDateOrContractMonth[2:]}"
                        f"{contract.right}{round(contract.strike * 1000):08d}"
                    )
                    ib.wrapper.contractDetails(
                        req_id,
                        ContractDetails(
                            contract=contract,
                            underConId=1,
                            realExpirationDate=contract.lastTradeDateOrContractMonth,
                            lastTradeTime="16:00:00",
                            timeZoneId="US/Eastern",
                            orderTypes="LMT,GTD",
                        ),
                    )
                ib.wrapper.contractDetailsEnd(req_id)

            asyncio.get_running_loop().call_soon(deliver)

        ib.client.reqContractDetails = request
        task = asyncio.create_task(b.contracts(Contract(symbol="ABC", conId=1), 100, OPEN))
        if mode == "cancel":
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif mode == "timeout":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 0.01)
        elif mode in {"all_invalid", "ambiguous"}:
            with pytest.raises((ValueError, RequestError)):
                await task
        else:
            p, c, combo = await task
            assert p.contract.lastTradeDateOrContractMonth == "20250722"
            assert c.contract.lastTradeDateOrContractMonth == "20250722"
            assert [leg.conId for leg in combo.comboLegs] == [101, 102]
        assert not ib.wrapper._futures and not ib.wrapper._results
        assert not ib.wrapper._reqId2Contract
        assert ib.RaiseRequestErrors is False

    asyncio.run(run())


@pytest.mark.parametrize(
    "status", ["Submitted", "Cancelled", "ApiCancelled", "Inactive", "RESERVED", "Filled"]
)
def test_existing_exit_never_retried_or_assumed_closed(tmp_path, monkeypatch, status):
    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    ref = b.store.reserve_order(
        event, "EXIT", 2, {**payload, "exit_con_ids": [101, 102], "exit_quantity": 1}
    )
    b.store.db.execute("UPDATE first4_orders SET status=? WHERE reference=?", (status, ref))
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    if status == "Submitted":
        b.ib.openTrades.return_value = [
            Trade(order=Order(orderRef=ref), orderStatus=OrderStatus(status=status))
        ]
    asyncio.run(b.close_due())
    asyncio.run(b.close_due())
    assert not b.ib.placeOrder.called
    assert b.store.rows("events")[0]["outcome"].startswith("EXIT_")
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: CLOSE)
    asyncio.run(b.close_due())
    assert b.store.rows("events")[0]["outcome"] == "EXIT_OVERDUE"
    assert len(b.store.unresolved()) == 1


def test_late_execution_reopens_verified_zero_allocation(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b, (0, 0))
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: CLOSE)
    asyncio.run(b.close_due())
    assert not b.store.unresolved()
    fill = NS(
        execution=NS(
            acctNumber=b.config.expected_account,
            orderRef="F4:2025-07-21:1:ENTRY",
            execId="late",
            shares=1,
            price=1,
            side="BOT",
            time=OPEN,
        ),
        contract=Contract.create(**payload["put"]),
    )
    b.fill(None, fill)
    b.fill(None, fill)
    assert len(b.store.unresolved()) == 1 and len(b.store.rows("fills")) == 1
    asyncio.run(b.close_due())
    assert not b.reconciled and not b.ib.placeOrder.called


def test_calendar_requested_coverage_holiday_early_close_and_year_boundary(tmp_path):
    from stocker_data.calendars import get_market_calendar

    runtime = Runtime(armed_config(), Store(tmp_path / "calendar.sqlite"))
    real = get_market_calendar("NYSE")
    calendar = NS(schedule=Mock(wraps=real.schedule))

    async def run():
        for today in [date(2025, 7, 4), date(2025, 7, 5), date(2025, 7, 5)]:
            await runtime.refresh_calendar(calendar, today)
        assert calendar.schedule.call_count == 1
        assert not any(r[0] == "2025-07-04" for r in runtime.schedule)
        early = next(r for r in runtime.schedule if r[0] == "2025-07-03")
        assert early[2] - early[1] == timedelta(hours=3, minutes=30)
        await runtime.refresh_calendar(calendar, date(2025, 7, 6))
        await runtime.refresh_calendar(calendar, date(2025, 7, 6))
        assert calendar.schedule.call_count == 2
        await runtime.refresh_calendar(calendar, date(2025, 12, 31))
        await runtime.refresh_calendar(calendar, date(2026, 1, 1))
        assert calendar.schedule.call_count == 3
        assert not any(r[0] == "2026-01-01" for r in runtime.schedule)
        await runtime.refresh_calendar(calendar, date(2026, 1, 2))
        assert any(r[0] == "2026-01-02" for r in runtime.schedule)

    asyncio.run(run())


def test_partial_exit_restart_and_final_reconciled_zero(tmp_path, monkeypatch):
    from stocker_execution.first4_broker import PaperBroker
    from test_first4 import fake_ib

    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    asyncio.run(b.close_due())
    ref = "F4:2025-07-21:1:EXIT"

    def execution(con_id):
        return NS(
            execution=NS(
                acctNumber=b.config.expected_account,
                orderRef=ref,
                execId=f"sell-{con_id}",
                shares=1,
                price=1,
                side="SLD",
                time=OPEN,
            ),
            contract=Contract(secType="OPT", conId=con_id, multiplier="100"),
        )

    b.fill(None, execution(101))
    b.fill(None, execution(101))
    b.ib.positions.return_value = [NS(contract=Contract(conId=102), position=1)]
    b.store.db.close()
    b = PaperBroker(armed_config(), Store(tmp_path / "s.sqlite"), fake_ib())
    trade = Trade(
        order=Order(
            account=b.config.expected_account, clientId=81, orderId=200, orderRef=ref, permId=200
        ),
        orderStatus=OrderStatus(status="Submitted", permId=200),
    )
    b.ib.reqAllOpenOrdersAsync.return_value = [trade]
    b.ib.openTrades.return_value = [trade]
    b.ib.reqPositionsAsync.return_value = [
        NS(account=b.config.expected_account, contract=Contract(conId=102), position=1, avgCost=100)
    ]
    b.ib.positions.return_value = b.ib.reqPositionsAsync.return_value
    asyncio.run(b.reconcile())
    asyncio.run(b.close_due())
    assert b.store.rows("events")[0]["outcome"] == "EXIT_WORKING"
    assert not b.ib.placeOrder.called
    b.fill(None, execution(102))
    trade.orderStatus.status = "Filled"
    b.order_status(trade)
    b.ib.reqAllOpenOrdersAsync.return_value = []
    b.ib.openTrades.return_value = []
    b.ib.reqPositionsAsync.return_value = []
    b.ib.positions.return_value = []
    asyncio.run(b.close_due())
    assert b.store.rows("events")[0]["outcome"] == "CLOSED"
    assert not b.store.unresolved() and not b.ib.placeOrder.called
    # Duplicate acknowledgement/fill cannot reopen or produce another sell.
    writes = b.store.db.total_changes
    b.order_status(trade)
    b.fill(None, execution(102))
    asyncio.run(b.close_due())
    assert b.store.db.total_changes == writes
    trade.orderStatus.status = "Submitted"
    b.order_status(trade)
    assert not b.reconciled and b.store.unresolved()


@pytest.mark.parametrize("change", [{"bid": 0}, {"age": 6}])
def test_exit_quote_policy_stays_fail_closed(tmp_path, change):
    from datetime import UTC, datetime

    b = broker(tmp_path)

    class Updates:
        def __iadd__(self, callback):
            asyncio.get_running_loop().call_soon(callback, ticker)
            return self

        def __isub__(self, callback):
            return self

    stamp = datetime.now(UTC) - timedelta(seconds=change.get("age", 0))
    ticker = NS(
        bid=change.get("bid", 1),
        ask=1.1,
        bidSize=1,
        askSize=1,
        marketDataType=1,
        ticks=[NS(tickType=i, time=stamp) for i in [1, 2]],
        updateEvent=Updates(),
    )
    b.ib.reqMktData = Mock(return_value=ticker)
    b.ib.cancelMktData = Mock()
    with pytest.raises(ValueError, match="QUOTES"):
        asyncio.run(
            b.quotes(
                [Contract(conId=1), Contract(conId=2)],
                datetime.now(UTC) + timedelta(seconds=0.03),
                side="SELL",
            )
        )
    assert b.ib.cancelMktData.call_count == 2 and not b.ib.placeOrder.called


def test_dashboard_history_bounds_and_offset(tmp_path):
    import httpx

    from stocker_dashboard.app import create_dashboard_app

    runtime = Runtime(armed_config(), Store(tmp_path / "dashboard.sqlite"))
    with runtime.store.db:
        runtime.store.db.executemany(
            "INSERT INTO first4_meta VALUES (?,?)", [(str(i), "{}") for i in range(1000)]
        )
        runtime.store.db.executemany(
            "INSERT INTO first4_orders VALUES (?,?,?,'EXIT',0,0,'Cancelled','{}')",
            [(str(i), str(i), "A") for i in range(1000)],
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            data = (await client.get("/api/overview")).json()
            assert len(data["orders"]) == 100 and len(data["errors"]) == 150
            first = (await client.get("/api/orders?limit=100")).json()
            second = (await client.get("/api/orders?limit=100&offset=100")).json()
            assert first[0]["reference"] == "999" and second[0]["reference"] == "899"
            assert (await client.get("/api/orders?limit=10000")).status_code == 422

    asyncio.run(run())


def test_exit_rechecks_working_orders_after_quote_wait(tmp_path, monkeypatch):
    b = broker(tmp_path)
    prepare_exit(b)
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    quotes = b.quotes.return_value

    async def changed(*args, **kwargs):
        b.ib.openTrades.return_value = [
            Trade(
                contract=Contract(secType="OPT", conId=101),
                order=Order(action="SELL", totalQuantity=1),
                orderStatus=OrderStatus(status="Submitted"),
            )
        ]
        return quotes

    b.quotes = changed
    asyncio.run(b.close_due())
    assert not b.ib.placeOrder.called
    assert "CONFLICTING_WORKING_ORDER" in b.problem
    assert b.store.rows("events")[0]["outcome"] != "CLOSED"


def test_candidate_completion_speed_cannot_reorder_first4(tmp_path):
    async def run():
        runtime = Runtime(armed_config(), Store(tmp_path / "ordering.sqlite"))
        runtime.scanner_rows = AsyncMock(
            return_value=[
                NS(rank=i - 1, contractDetails=NS(contract=Contract(conId=i, symbol=str(i))))
                for i in range(1, 6)
            ]
        )

        async def delayed(row, *args):
            await asyncio.sleep((5 - row.rank) * 0.001)
            return candidate(str(row.rank + 1), row.rank + 1)

        runtime.candidate = delayed
        runtime.execute = AsyncMock()
        await runtime.scan("2025-07-21", OPEN, CLOSE, OPEN, OPEN)
        await asyncio.gather(*runtime.tasks)
        events = runtime.store.rows("events")
        assert [(e["symbol"], e["slot"]) for e in events[:4]] == [(str(i), i) for i in range(1, 5)]
        assert events[4]["decision"] == "DAILY_CAP"

    asyncio.run(run())


def test_repeated_restoration_with_unresolved_position_is_not_entry_ready(tmp_path):
    b = broker(tmp_path)
    for code in [1100, 1100, 1101, 1101, 1102]:
        b.error(-1, code, "connectivity notification")
        assert not b.reconciled
    b.ib.reqPositionsAsync.return_value = [
        NS(account=b.config.expected_account, contract=Contract(conId=999), position=1)
    ]
    asyncio.run(b.reconcile())
    assert b.entry_blocker == "UNOWNED_BROKER_POSITIONS:[999]"
    assert not b.ib.placeOrder.called
