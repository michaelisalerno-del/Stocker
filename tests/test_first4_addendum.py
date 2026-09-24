"""PR #10 / competing-session regressions. All IB objects are isolated fakes."""

import asyncio
from datetime import timedelta
from unittest.mock import Mock

import pytest
from ib_async import Contract, Order, OrderStatus, Trade

from stocker_execution.first4_config import PAPER_ACCOUNT
from test_first4 import OPEN, broker


def open_order(reference="foreign", **changes):
    fields = dict(account=PAPER_ACCOUNT, clientId=181, orderId=91, permId=901, orderRef=reference)
    fields.update(changes)
    return Trade(
        contract=Contract(conId=901),
        order=Order(**fields),
        orderStatus=OrderStatus(status="Submitted", permId=fields["permId"]),
    )


def test_foreign_open_order_blocks_flat_account_across_manager_passes(tmp_path):
    b = broker(tmp_path)
    b.ib.cancelOrder = Mock()
    b.ib.reqAllOpenOrdersAsync.return_value = [open_order()]
    asyncio.run(b.reconcile())
    assert b.reconciled
    assert "UNOWNED_BROKER_OPEN_ORDERS" in b.entry_blocker
    for _ in range(2):
        asyncio.run(b.cancel_due_entries())
        asyncio.run(b.close_due())
        assert "UNOWNED_BROKER_OPEN_ORDERS" in b.entry_blocker
    b.ib.cancelOrder.assert_not_called()
    b.ib.placeOrder.assert_not_called()
    b.ib.reqAllOpenOrdersAsync.return_value = []
    asyncio.run(b.reconcile())
    assert b.reconciled and not b.entry_blocker


@pytest.mark.parametrize("changes", [{}, {"clientId": 181}, {"orderId": 92}, {"permId": 902}])
def test_open_order_requires_durable_identifiers(tmp_path, changes):
    b = broker(tmp_path)
    ref = b.store.reserve_order(
        dict(session="2025-07-21", symbol="A", slot=1),
        "ENTRY",
        91,
        {"entry_deadline_at": (OPEN + timedelta(days=9999)).isoformat()},
    )
    with b.store.db:
        b.store.db.execute("UPDATE first4_orders SET perm_id=901,status='Submitted'")
    fields = dict(clientId=81)
    fields.update(changes)
    b.ib.reqAllOpenOrdersAsync.return_value = [open_order(ref, **fields)]
    if changes:
        with pytest.raises(ValueError):
            asyncio.run(b.reconcile())
        assert "UNOWNED_BROKER_OPEN_ORDERS" in b.entry_blocker
        assert b.store.order(ref)["perm_id"] == 901
    else:
        asyncio.run(b.reconcile())
        assert b.reconciled and not b.entry_blocker


def test_10197_is_active_and_survives_deadline_processing(tmp_path, monkeypatch):
    b = broker(tmp_path)
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN - timedelta(hours=1))
    b.error(17, 10197, "No market data during competing session", Contract(conId=71, symbol="A"))
    assert b.ib.isConnected() and b.reconciled
    assert not b.entries_armed()
    asyncio.run(b.cancel_due_entries())
    asyncio.run(b.reconcile())
    assert not b.entries_armed()
    assert b.market_data_block["request_id"] == 17
    assert b.market_data_block["contract"]["con_id"] == 71
    b.ib.placeOrder.assert_not_called()


@pytest.mark.parametrize(
    "changes", [{"account": "OTHER"}, {"clientId": 181}, {"orderId": 92}, {"permId": 902}]
)
def test_mismatched_callbacks_never_adopt_or_cancel_order(tmp_path, monkeypatch, changes):
    from types import SimpleNamespace as NS

    b = broker(tmp_path)
    ref = b.store.reserve_order(
        dict(session="2025-07-21", symbol="A", slot=1),
        "ENTRY",
        91,
        {"entry_deadline_at": OPEN.isoformat(), "exit_at": OPEN.isoformat()},
    )
    with b.store.db:
        b.store.db.execute("UPDATE first4_orders SET perm_id=901,status='Submitted'")
    fields = dict(clientId=81)
    fields.update(changes)
    trade = open_order(ref, **fields)
    b.order_status(trade)
    assert not b.reconciled
    assert b.store.order(ref)["perm_id"] == 901
    b.reconciled = True
    b.ib.openTrades.return_value = [trade]
    b.ib.cancelOrder = Mock()
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN)
    with pytest.raises(ValueError, match="identity mismatch"):
        asyncio.run(b.cancel_due_entries())
    with pytest.raises(ValueError, match="identity mismatch"):
        asyncio.run(b.close_one(ref))
    e = NS(
        acctNumber=trade.order.account,
        clientId=trade.order.clientId,
        orderId=trade.order.orderId,
        permId=trade.order.permId,
        orderRef=ref,
        execId="bad",
        shares=1,
        price=1,
        side="BOT",
        time=OPEN,
    )
    b.fill(None, NS(execution=e, contract=Contract(secType="OPT", conId=1, multiplier="100")))
    assert not b.store.rows("fills")
    b.ib.cancelOrder.assert_not_called()
    b.ib.placeOrder.assert_not_called()


def test_foreign_orders_leave_verified_owned_exit_operating(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    from test_first4 import CLOSE, prepare_exit

    b = broker(tmp_path)
    prepare_exit(b)
    b.ib.reqPositionsAsync.return_value = [
        NS(account=PAPER_ACCOUNT, contract=p.contract, position=p.position, avgCost=100)
        for p in b.ib.positions()
    ]
    b.ib.reqAllOpenOrdersAsync.return_value = [open_order()]
    b.ib.openTrades.return_value = [open_order()]
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    asyncio.run(b.reconcile())
    asyncio.run(b.cancel_due_entries())
    assert "UNOWNED_BROKER_OPEN_ORDERS" in b.entry_blocker
    asyncio.run(b.close_due())
    assert b.ib.placeOrder.call_count == 1
    assert b.ib.placeOrder.call_args.args[1].action == "SELL"
    assert "UNOWNED_BROKER_OPEN_ORDERS" in b.entry_blocker


def test_failed_snapshot_cannot_clear_foreign_blocker(tmp_path):
    b = broker(tmp_path)
    b.ib.reqAllOpenOrdersAsync.return_value = [open_order()]
    asyncio.run(b.reconcile())
    b.ib.reqAllOpenOrdersAsync.return_value = []

    async def interrupted():
        b.disconnected()
        return []

    b.ib.reqPositionsAsync.side_effect = interrupted
    with pytest.raises(ValueError):
        asyncio.run(b.reconcile())
    assert "UNOWNED_BROKER_OPEN_ORDERS" in b.entry_blocker


def data_probe(b, monkeypatch, mode="ok"):
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN)
    monkeypatch.setattr("stocker_execution.first4_readiness.now", lambda: OPEN)
    b.ib.reqMarketDataType = Mock()
    b.ib.qualifyContractsAsync = AsyncMock(return_value=[Contract(conId=42, symbol="F")])
    b.ib.reqTickersAsync = AsyncMock(
        return_value=[
            NS(
                marketPrice=lambda: 13,
                marketDataType=3 if mode == "delayed_stock" else 1,
                time=OPEN,
            )
        ]
    )
    b.chain = AsyncMock(
        return_value=[
            NS(
                exchange="SMART",
                tradingClass="F",
                multiplier="100",
                expirations={"20250723"},
                strikes={13},
            )
        ]
    )

    def details(c):
        c.conId = 101 if c.right == "P" else 102
        c.localSymbol = f"F     250723{c.right}00013000"
        return [
            NS(
                contract=c,
                underConId=42,
                minSize=1,
                sizeIncrement=1,
                minTick=0.01,
                realExpirationDate="20250723",
                lastTradeTime="16:00:00",
                timeZoneId="US/Eastern",
                orderTypes="LMT,GTD",
            )
        ]

    b.ib.reqContractDetailsAsync = AsyncMock(side_effect=details)

    def ticker(c, *args):
        class Updates:
            def __iadd__(self, callback):
                if mode == "new_error":
                    b.error(55, 10197, "competing again", c)
                asyncio.get_running_loop().call_soon(callback, t)
                return self

            def __isub__(self, callback):
                return self

        stamp = OPEN - timedelta(seconds=10) if mode == "stale" else OPEN
        t = NS(
            minTick=0.01,
            marketDataType=3 if mode == "delayed_options" else 1,
            bid=0.1,
            ask=0.2,
            bidSize=1,
            askSize=1,
            updateEvent=Updates(),
            ticks=[NS(tickType=1, time=stamp), NS(tickType=2, time=stamp)],
        )
        return t

    b.ib.reqMktData = Mock(side_effect=ticker)
    b.ib.cancelMktData = Mock()


@pytest.mark.parametrize("mode", ["ok", "delayed_stock", "delayed_options", "stale", "new_error"])
def test_10197_only_complete_fresh_verification_clears_active_block(tmp_path, monkeypatch, mode):
    from stocker_execution.first4_broker import PaperBroker
    from stocker_execution.first4_readiness import option_access
    from test_first4 import armed_config, fake_ib

    b = broker(tmp_path)
    b.opening_verified_session = "2025-07-21"
    b.error(17, 10197, "competing")
    data_probe(b, monkeypatch, mode)

    async def run():
        return await asyncio.wait_for(option_access(b, OPEN + timedelta(seconds=1)), 0.1)

    if mode == "ok":
        asyncio.run(run())
        assert not b.market_data_block
        assert b.store.get_meta("market_data_recovery")["checks"]["fresh_realtime_option_quotes"]
    else:
        with pytest.raises((ValueError, TimeoutError)):
            asyncio.run(run())
        assert b.market_data_block
    assert b.opening_verified_session is None
    assert b.store.get_meta("market_data_10197")["code"] == 10197
    restarted = PaperBroker(armed_config(), b.store, fake_ib())
    assert bool(restarted.market_data_block) == bool(b.market_data_block)
    b.ib.placeOrder.assert_not_called()
    assert b.ib.reqMarketDataType.call_args.args == (1,)
    assert b.ib.reqMktData.call_count == b.ib.cancelMktData.call_count


def test_10197_recovery_keeps_scanner_gap_and_dated_gate(tmp_path, monkeypatch):
    from stocker_execution.first4_readiness import option_access
    from test_first4 import CLOSE, opening_runtime

    runtime = opening_runtime(tmp_path, monkeypatch)
    b = runtime.broker
    runtime.store.observe("2025-07-21", OPEN, CLOSE, [])
    b.error(1, 10197, "competing")
    b.error(2, 10197, "competing again")
    diagnostic = dict(b.market_data_block)
    data_probe(b, monkeypatch)
    asyncio.run(option_access(b, OPEN + timedelta(seconds=1)))
    assert not b.market_data_block and not b.entries_armed()
    assert b.store.rows("sessions")[0]["blocked"]
    assert b.store.get_meta("market_data_10197") == diagnostic
    assert not b.store.rows("events")


def test_10197_keeps_owned_exit_and_overdue_alert(tmp_path, monkeypatch):
    from test_first4 import CLOSE, prepare_exit

    b = broker(tmp_path)
    prepare_exit(b)
    b.error(1, 10197, "competing")
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    asyncio.run(b.close_due())
    assert b.ib.placeOrder.call_count == 1  # Fresh exit quotes (fixture) still permit owned SELL.
    assert b.market_data_block
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: CLOSE)
    asyncio.run(b.close_due())
    assert b.store.rows("events")[0]["outcome"] == "EXIT_OVERDUE"
    assert b.ib.placeOrder.call_count == 1


def test_identical_position_callbacks_do_not_rewrite_snapshot(tmp_path):
    from types import SimpleNamespace as NS

    from test_first4 import prepare_exit

    b = broker(tmp_path)
    prepare_exit(b)
    p = NS(account=PAPER_ACCOUNT, contract=Contract(conId=101), position=1, avgCost=100)
    b.position(p)
    writes = b.store.db.total_changes
    b.position(p)
    assert b.store.db.total_changes == writes


@pytest.mark.parametrize("when", ["before", "contracts", "quotes"])
def test_10197_prevents_entry_before_or_during_preparation(tmp_path, monkeypatch, when):
    from datetime import datetime
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock

    from test_first4 import CLOSE, candidate

    b = broker(tmp_path)
    event = b.store.observe("2025-07-21", OPEN + timedelta(minutes=15), CLOSE, [candidate("A", 1)])[
        0
    ]
    baseline = datetime.fromisoformat(event["entry_at"])
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: baseline)
    legs = [
        NS(
            contract=Contract(secType="OPT", conId=i, multiplier="100", strike=10),
            sizeIncrement=1,
            minSize=1,
            realExpirationDate="20250723",
            lastTradeTime="16:00:00",
            timeZoneId="US/Eastern",
        )
        for i in (101, 102)
    ]

    async def contracts(*args):
        if when == "contracts":
            b.error(2, 10197, "competing")
        return *legs, Contract(secType="BAG")

    async def quotes(*args):
        if when == "quotes":
            b.error(3, 10197, "competing")
        return [
            dict(bid=0.1, ask=0.2, bid_at=baseline.isoformat(), ask_at=baseline.isoformat())
        ] * 2

    b.contracts = AsyncMock(side_effect=contracts)
    b.quotes = AsyncMock(side_effect=quotes)
    b.combo_tick = AsyncMock(return_value=0.01)
    if when == "before":
        b.error(1, 10197, "competing")
    with pytest.raises(ValueError, match="MARKET_DATA"):
        asyncio.run(b.enter(event, Contract(conId=1), 10))
    b.ib.placeOrder.assert_not_called()
    assert not b.store.rows("orders")
    assert b.store.rows("events")[0]["slot"] == 1


def test_10197_rejects_scanner_results_and_cleans_request(tmp_path):
    from test_first4 import CLOSE
    from test_first4_reliability import scanner_runtime

    async def run():
        runtime = scanner_runtime(tmp_path, "empty")
        ib = runtime.broker.ib

        def request(req_id, *args):
            def deliver():
                ib.wrapper.error(req_id, 10197, "competing", "")
                ib.wrapper.scannerDataEnd(req_id)

            asyncio.get_running_loop().call_soon(deliver)

        ib.client.reqScannerSubscription = request
        with pytest.raises(ValueError):
            await runtime.scan("2025-07-21", OPEN, CLOSE, OPEN, OPEN)
        assert runtime.broker.market_data_block
        assert not runtime.store.rows("events") and not runtime.store.rows("orders")
        assert not ib.wrapper.reqId2Subscriber
        ib.client.cancelScannerSubscription.assert_called_once()

    asyncio.run(run())


def test_completed_order_cannot_adopt_unknown_permanent_id(tmp_path):
    b = broker(tmp_path)
    ref = b.store.reserve_order(dict(session="2025-07-21", symbol="A", slot=1), "ENTRY", 1, {})
    b.ib.reqCompletedOrdersAsync.return_value = [
        Trade(
            order=Order(account=PAPER_ACCOUNT, orderRef=ref, permId=99),
            orderStatus=OrderStatus(status="Cancelled"),
        )
    ]
    with pytest.raises(ValueError, match="identity mismatch"):
        asyncio.run(b.reconcile())
    assert not b.reconciled
    assert b.store.order(ref)["status"] == "RESERVED"


def test_dashboard_history_page_is_bounded_and_does_not_read_other_histories(tmp_path):
    import httpx

    from stocker_dashboard.app import create_dashboard_app
    from stocker_execution.first4_runtime import Runtime
    from test_first4_capacity import historical_fixture

    b = broker(tmp_path)
    historical_fixture(b, 10000, 4)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    queries = []
    b.store.db.set_trace_callback(queries.append)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as c:
            response = await c.get("/api/overview?view=orders&offset=100")
            assert response.status_code == 200
            data = response.json()
            assert len(data["orders"]) == 100
            assert not data["candidates"] and not data["fills"] and not data["errors"]
            assert not any("FROM first4_events" in q or "FROM first4_fills" in q for q in queries)
            assert (await c.get("/api/overview?offset=-1")).status_code == 422
            assert (await c.get("/api/overview?view=bogus")).status_code == 422

    asyncio.run(run())


def test_old_chain_response_cannot_restore_assumptions_after_10197(tmp_path):
    from unittest.mock import AsyncMock

    b = broker(tmp_path)

    async def chain(*args):
        b.error(18, 10197, "competing")
        return ["stale"]

    b.ib.reqSecDefOptParamsAsync = AsyncMock(side_effect=chain)
    with pytest.raises(ValueError, match="MARKET_DATA_CHANGED"):
        asyncio.run(b.chain(Contract(conId=1, symbol="A")))
    assert not b.chains and b.market_data_block
