"""Offline regressions for FIRST4 supervision, bounded work and recovery."""

import asyncio
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from ib_async import Contract, Order, OrderStatus, RequestError, Trade

from stocker_dashboard.app import create_dashboard_app, session_pnl
from stocker_data.calendars import get_market_calendar
from stocker_execution.first4_broker import PaperBroker
from stocker_execution.first4_requests import First4IB
from stocker_execution.first4_runtime import Runtime
from stocker_execution.first4_store import Store
from test_first4 import CLOSE, OPEN, armed_config, broker, candidate, fake_ib, prepare_exit


def test_configured_arming_cannot_bypass_session_block(tmp_path, monkeypatch):
    b = broker(tmp_path)
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN)
    b.store.block("2025-07-21", "SCANNER_MINUTE_MISSED")
    assert not b.entries_armed()


def test_unowned_pending_order_blocks_entries_only(tmp_path):
    b = broker(tmp_path)
    trade = Trade(
        order=Order(account=b.config.expected_account, clientId=99, orderRef="unrelated"),
        orderStatus=OrderStatus(status="Submitted"),
    )
    b.ib.reqAllOpenOrdersAsync.return_value = [trade]
    asyncio.run(b.reconcile())
    assert "UNOWNED" in b.entry_blocker
    assert b.reconciled
    b.guard()  # Existing owned exit obligations still have an account-safe connection.
    assert not b.ib.placeOrder.called


def test_upstream_loss_is_not_local_connection_readiness(tmp_path):
    b = broker(tmp_path)
    b.error(-1, 1100, "Upstream lost")
    assert b.ib.isConnected()
    with pytest.raises(ValueError, match="UPSTREAM"):
        b.guard()
    assert not b.entries_armed()


def test_terminal_outcome_does_not_repeat_writes(tmp_path):
    b = broker(tmp_path)
    event = b.store.observe("2025-07-21", OPEN, CLOSE, [candidate("A", 1)])[0]
    b.store.outcome(event, "CLOSED")
    changes = b.store.db.total_changes
    b.store.outcome(event, "CLOSED")
    assert b.store.db.total_changes == changes


def test_dashboard_defaults_to_relevant_session(tmp_path):
    b = broker(tmp_path)
    prepare_exit(b)
    b.store.observe("2025-07-22", OPEN + timedelta(days=1), CLOSE, [candidate("B", 1)])
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    runtime.session = "2025-07-22"

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            data = (await client.get("/api/overview")).json()
            assert {r["session"] for r in data["candidates"]} == {"2025-07-22"}
            old = (await client.get("/api/overview?session=2025-07-21")).json()
            assert {r["session"] for r in old["candidates"]} == {"2025-07-21"}

    asyncio.run(check())


def test_health_survives_database_failure(tmp_path):
    b = broker(tmp_path)
    runtime = Runtime(b.config, b.store)
    b.store.db.close()
    status = runtime.status()
    assert status["ledger_available"] is False
    assert not status["armed"]


def test_critical_manager_termination_is_supervised(tmp_path):
    b = broker(tmp_path)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    runtime.maintain_broker = AsyncMock(side_effect=RuntimeError("manager died"))
    runtime.arm_at_open = AsyncMock()

    async def scan_forever():
        await asyncio.Event().wait()

    runtime.scan_sessions = scan_forever

    async def check():
        with pytest.raises(RuntimeError, match="manager died"):
            await asyncio.wait_for(runtime.run(), 0.2)
        assert runtime.status()["manager_health"] == "FAILED"
        assert not b.entries_armed()

    asyncio.run(check())


@pytest.mark.parametrize("role", ["manager", "worker"])
def test_critical_failure_and_reporting_failure_are_visible(tmp_path, caplog, role):
    b = broker(tmp_path)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    cleaned = []

    async def survivor():
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)

    async def fail():
        await asyncio.sleep(0)
        b.store.db.close()
        raise RuntimeError("critical fault")

    runtime.maintain_broker = fail if role == "manager" else survivor
    runtime.scan_sessions = fail if role == "worker" else survivor
    runtime.arm_at_open = AsyncMock()

    async def check():
        with pytest.raises(RuntimeError, match="critical fault"):
            await runtime.run()
        assert runtime.status()[role + "_health"] == "FAILED"
        assert not runtime.status()["armed"]
        assert cleaned and all(t.done() for t in runtime.critical_tasks)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            health = await client.get("/api/health")
            assert health.status_code == 503 and not health.json()["ledger_available"]
            assert (await client.get("/api/overview")).status_code == 503

    asyncio.run(check())
    assert "critical fault" in caplog.text and "could not persist" in caplog.text


def test_background_execution_exception_is_retrieved(tmp_path):
    b = broker(tmp_path)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b

    async def check():
        task = asyncio.create_task(AsyncMock(side_effect=sqlite3.OperationalError("disk full"))())
        runtime.tasks.add(task)
        task.add_done_callback(runtime.execution_done)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not runtime.tasks
        assert "EXECUTION_TASK_FAILED" in b.fatal_error
        assert not b.entries_armed()

    asyncio.run(check())


def test_reconnect_continuity_depends_on_due_observations(tmp_path, monkeypatch):
    b = broker(tmp_path)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    clock = OPEN - timedelta(minutes=20)
    monkeypatch.setattr("stocker_execution.first4_runtime.now", lambda: clock)
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: clock)
    b.ib.scanner = AsyncMock(return_value=[])
    calendar = get_market_calendar("NYSE")

    async def check():
        nonlocal clock
        await runtime.scan_step(calendar)
        assert runtime.session == "2025-07-21" and not runtime.problem
        b.opening_verified_session = runtime.session
        for _ in range(2):
            b.disconnected()
            await runtime.scan_step(calendar)
            await b.reconcile()
            await runtime.scan_step(calendar)
        assert not runtime.problem and b.opening_verified_session is None
        clock = OPEN + timedelta(minutes=1)
        await runtime.scan_step(calendar)
        assert b.store.rows("sessions")[0]["last_clock"] == clock.isoformat()
        b.disconnected()
        clock = OPEN + timedelta(minutes=2, seconds=3)
        await runtime.scan_step(calendar)
        assert runtime.problem == "SCANNER_MINUTE_MISSED"
        for _ in range(2):
            await b.reconcile()
            await runtime.scan_step(calendar)
        assert not b.entries_armed() and b.ib.scanner.await_count == 1
        clock = OPEN + timedelta(days=1) - timedelta(minutes=20)
        await runtime.scan_step(calendar)
        assert runtime.session == "2025-07-22" and not runtime.problem
        assert b.opening_verified_session is None

    asyncio.run(check())


@pytest.mark.parametrize(
    "stamp,session_open,session_close",
    [
        ("2025-07-19T12:00:00+00:00", None, None),  # Saturday
        ("2025-07-04T12:00:00+00:00", None, None),  # holiday
        ("2025-11-28T12:00:00+00:00", 14, 18),  # shortened session
        ("2025-03-07T12:00:00+00:00", 14, 21),  # before DST
        ("2025-03-10T12:00:00+00:00", 13, 20),  # after DST
    ],
)
def test_calendar_refresh_is_once_per_local_date(
    tmp_path, monkeypatch, stamp, session_open, session_close
):
    b = broker(tmp_path)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    clock = datetime.fromisoformat(stamp)
    monkeypatch.setattr("stocker_execution.first4_runtime.now", lambda: clock)
    calendar = get_market_calendar("NYSE")
    calendar.schedule = Mock(wraps=calendar.schedule)

    async def check():
        for _ in range(20):
            await runtime.scan_step(calendar)

    asyncio.run(check())
    assert calendar.schedule.call_count == 1
    session = next((r for r in runtime.schedule if r[0] == clock.date().isoformat()), None)
    if session_open is None:
        assert session is None and runtime.problem == "EXCHANGE_CLOSED"
    else:
        assert session[1].hour == session_open and session[2].hour == session_close
        assert runtime.next_clock == session[1] + timedelta(minutes=1)


def test_upstream_restoration_and_notices_do_not_clear_independent_blocks(tmp_path):
    b = broker(tmp_path)
    b.ib.disconnect = Mock(side_effect=b.disconnected)
    b.error(-1, 2103, "Market data farm disconnected:usfarm")
    b.error(-1, 2105, "Historical data farm disconnected:ushmds")
    b.error(-1, 2104, "Market data farm connected:usfarm")
    assert b.data_problem  # History farm is still down.
    b.error(-1, 2106, "Historical data farm connected:ushmds")
    assert not b.data_problem
    for notice in (2107, 2108, 2158):
        b.error(-1, notice, "Informational")
    assert b.reconciled
    b.opening_verified_session = "2025-07-21"
    b.error(-1, 1100, "Lost")
    b.error(-1, 1101, "Restored; data lost")
    assert b.ib.disconnect.call_count == 1
    assert not b.reconciled and b.opening_verified_session is None
    assert b.data_generation > 0
    b.store.set_meta("paused", True)
    asyncio.run(b.reconcile())
    assert not b.entries_armed()


def test_pinned_library_request_semantics_and_cleanup():
    async def check():
        ib = First4IB()
        ib.isConnected = Mock(return_value=True)  # Fake transport; no socket exists.
        ib.client.getReqId = Mock(side_effect=range(10, 30))
        ib.client.cancelHistoricalData = Mock()
        ib.client.reqHistoricalData = Mock(
            side_effect=lambda request_id, *args: ib.wrapper.historicalDataEnd(request_id, "", "")
        )
        assert await ib.history(Contract(conId=1), OPEN, "60 S") == []
        ib.client.reqHistoricalData.side_effect = lambda request_id, *args: ib.wrapper.error(
            request_id, 162, "historical service unavailable", ""
        )
        with pytest.raises(RequestError):
            await ib.history(Contract(conId=1), OPEN, "60 S")
        ib.client.reqHistoricalData.side_effect = None
        task = asyncio.create_task(ib.history(Contract(conId=1), OPEN, "60 S"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ib.client.cancelHistoricalData.call_count == 3
        assert not ib.wrapper._futures and not ib.wrapper._results
        ib.client.reqScannerSubscription = Mock()
        ib.client.cancelScannerSubscription = Mock()

        # Local loss before cancellation must not turn CancelledError into a
        # second "not connected" error or leave request/subscriber maps behind.
        for operation in (ib.history(Contract(conId=1), OPEN, "60 S"), ib.scanner(NS(), [])):
            task = asyncio.create_task(operation)
            await asyncio.sleep(0)
            ib.isConnected.return_value = False
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not ib.wrapper._futures and not ib.wrapper._results
            assert not ib.wrapper.reqId2Subscriber
            ib.isConnected.return_value = True
        assert not ib.wrapper._reqId2Contract
        ib.client.reqScannerSubscription = Mock()
        ib.client.cancelScannerSubscription = Mock()
        task = asyncio.create_task(ib.scanner(NS(), []))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ib.wrapper.reqId2Subscriber and not ib.wrapper._futures
        assert not ib.wrapper._results
        assert ib.client.cancelScannerSubscription.call_count == 1
        ib.client.reqContractDetails = Mock()
        task = asyncio.create_task(ib.reqContractDetailsAsync(Contract(conId=1)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ib.wrapper._futures and not ib.wrapper._results

    asyncio.run(check())


def test_scanner_burst_rank_partial_failure_and_cancellation(tmp_path, monkeypatch):
    b = broker(tmp_path)
    b.config = b.config.model_copy(update={"armed": False})
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    rows = [
        NS(rank=i, contractDetails=NS(contract=Contract(conId=i + 1, symbol=f"S{i:02}")))
        for i in range(25)
    ]
    b.ib.scanner = AsyncMock(return_value=list(reversed(rows)))
    active = peak = completed = 0
    clock = OPEN + timedelta(minutes=15)
    previous = CLOSE - timedelta(days=3)

    async def history(contract, end, duration):
        nonlocal active, peak, completed
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep((26 - contract.conId) * 0.0001)
            if contract.symbol == "S00":
                raise RequestError(1, 162, "service failed")
            if duration == "60 S":
                return [NS(date=previous - timedelta(minutes=1), close=9)]
            return [
                NS(date=OPEN + timedelta(minutes=i), open=10, high=11, low=10, close=10)
                for i in range(15)
            ]
        finally:
            active -= 1
            completed += 1

    b.ib.history = history

    async def check():
        await runtime.scan("2025-07-21", OPEN, CLOSE, previous, clock)
        await asyncio.gather(*runtime.tasks)
        events = b.store.rows("events")
        assert [e["symbol"] for e in events if e["slot"]] == ["S01", "S02", "S03", "S04"]
        assert "FAILED_OR_TIMED_OUT" in events[0]["detail"]
        assert len(events) == 25 and peak <= 4 and active == 0
        assert runtime.history_limit._value == 4
        assert runtime.scan_metrics["requests"] == 50
        await runtime.scan("2025-07-21", OPEN, CLOSE, previous, clock + timedelta(minutes=1))
        assert completed <= 50  # No later history/readmission for any first appearance.
        assert runtime.scan_metrics["requests"] == 0
        b.ib.history = AsyncMock(side_effect=lambda *args: None)

        async def slow(*args):
            await asyncio.Event().wait()

        b.ib.history = slow
        result = await runtime.candidate(
            rows[0], OPEN, clock, previous, asyncio.get_running_loop().time() + 0.01
        )
        assert result["detail"]["request_status"] == "FAILED_OR_TIMED_OUT"
        assert runtime.history_limit._value == 4
        task = asyncio.create_task(runtime.candidate(rows[0], OPEN, clock, previous))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.history_limit._value == 4

    asyncio.run(check())


def report_fill(b, reference, leg, exec_id, side="BOT", quantity=1, price=1):
    order = b.store.db.execute(
        "SELECT order_id,perm_id FROM first4_orders WHERE reference=?", (reference,)
    ).fetchone()
    b.fill(
        None,
        NS(
            contract=leg,
            execution=NS(
                acctNumber=b.config.expected_account,
                orderRef=reference,
                execId=exec_id,
                clientId=b.config.client_id,
                orderId=order["order_id"],
                permId=order["perm_id"] or 11,
                shares=quantity,
                price=price,
                side=side,
                time=OPEN,
            ),
        ),
    )


def test_completed_pnl_survives_open_allocation_and_late_fees(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    payload.update(deadline_reconciled=True, exit_quantity=1, exit_con_ids=[101, 102])
    ref = b.store.reserve_order(event, "EXIT", 2, payload)
    for i in (101, 102):
        leg = Contract(secType="OPT", conId=i, multiplier="100")
        report_fill(b, ref, leg, "sell" + str(i), "SLD", price=1.2)
    b.store.db.execute("UPDATE first4_orders SET payload=?", (json.dumps(payload),))
    b.store.db.execute("UPDATE first4_orders SET status='Filled' WHERE role='EXIT'")
    b.ib.positions.return_value = []
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: CLOSE)
    asyncio.run(b.close_due())
    assert not b.store.active_entries()
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    result = session_pnl(runtime, event["session"])
    assert result["completed_gross_usd"] == pytest.approx(40)
    assert result["realised"] is None
    assert result["pending_fee_executions"] == 4
    for exec_id in ("101", "102", "sell101", "sell102"):
        report = NS(currency="USD", commission=0.65, execId=exec_id)
        b.commission(None, None, report)
        changes = b.store.db.total_changes
        b.commission(None, None, report)
        assert changes == b.store.db.total_changes
    second = b.store.observe(
        event["session"], OPEN + timedelta(minutes=16), CLOSE, [candidate("B", 1)]
    )[0]
    other = b.store.reserve_order(second, "ENTRY", 3, payload)
    leg = Contract(secType="OPT", conId=201, multiplier="100")
    report_fill(b, other, leg, "other.01")
    result = session_pnl(runtime, event["session"])
    assert result["realised"] == pytest.approx(37.4)
    assert not result["fees_complete"] and result["open_owned_legs"][0]["quantity"] == 1
    other_exit = b.store.reserve_order(second, "EXIT", 4, payload, "201")
    report_fill(b, other_exit, leg, "partial.01", "SLD", 0.5, 1.5)
    result = session_pnl(runtime, event["session"])
    assert result["partial_close_gross_usd"] == 25
    assert result["realised"] == pytest.approx(37.4)
    reopened = Store(tmp_path / "s.sqlite")
    runtime.store = reopened
    assert session_pnl(runtime, event["session"]) == result


def test_late_fill_reopens_obligation_and_corrections_do_not_double_count(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b, (0, 0))
    reference = b.store.rows("orders")[0]["reference"]
    payload["deadline_reconciled"] = True
    b.store.db.execute("UPDATE first4_orders SET payload=?", (json.dumps(payload),))
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: CLOSE)
    asyncio.run(b.close_due())
    assert not b.store.active_entries()
    leg = Contract(secType="OPT", conId=101, multiplier="100")
    report_fill(b, reference, leg, "leg.01")
    assert b.store.active_entries() and b.owned_quantities()[101] == 1
    report_fill(b, reference, leg, "leg.02", quantity=0.5, price=1.1)
    report_fill(b, reference, leg, "leg.01")  # Older reconciliation cannot undo correction.
    assert b.owned_quantities()[101] == 0.5
    assert len(b.store.rows("fills")) == 2  # Preserve source reports.
    assert len(b.store.order_fills(reference)) == 1
    b.ib.positions.return_value = [NS(contract=leg, position=0.5)]
    asyncio.run(b.close_due())
    assert not b.ib.placeOrder.called
    assert b.store.event(event["session"], "A")["outcome"] == "EXIT_OVERDUE"
    assert "MISSED_SESSION_CLOSE" in b.operator_exceptions[reference]
    changes = b.store.db.total_changes
    asyncio.run(b.close_due())
    assert b.store.db.total_changes == changes


@pytest.mark.parametrize("state", ["Cancelled", "Inactive", "RESERVED", "Submitted"])
def test_incomplete_exit_never_resubmits_or_finishes(tmp_path, monkeypatch, state):
    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    payload.update(exit_quantity=1, exit_con_ids=[101, 102])
    ref = b.store.reserve_order(event, "EXIT", 2, payload)
    b.store.db.execute("UPDATE first4_orders SET status=? WHERE reference=?", (state, ref))
    report_fill(
        b, ref, Contract(secType="OPT", conId=101, multiplier="100"), "partial", "SLD", quantity=0.5
    )
    b.ib.positions.return_value[0].position = 0.5
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: CLOSE - timedelta(seconds=20)
    )
    asyncio.run(b.close_due())
    changes = b.store.db.total_changes
    asyncio.run(b.close_due())
    assert b.store.db.total_changes == changes
    assert not b.ib.placeOrder.called and b.store.active_entries()
    assert b.operator_exceptions


def qualification_fixture(b, mode):
    underlying = Contract(secType="STK", conId=1, symbol="ABC")
    b.chain = AsyncMock(
        return_value=[
            NS(
                exchange="SMART",
                tradingClass="ABC",
                multiplier="100",
                expirations={"20250723" if mode.startswith("exact_") else "20250722", "20250724"},
                strikes=[97, 103],
            )
        ]
    )

    def qualify(c):
        if c.lastTradeDateOrContractMonth == "20250724":
            if mode in {"empty", "exact_empty"}:
                return []
            if mode in {"unavailable", "exact_unavailable"}:
                raise RequestError(1, 200, "No security definition")
            if mode == "timeout":
                raise TimeoutError("metadata request timed out")
            if mode == "ambiguous":
                return [NS(), NS()]
            if mode == "ambiguous_error":
                raise RequestError(1, 200, "The contract description is ambiguous")
        if mode == "none":
            return []
        c.conId = 101 if c.right == "P" else 102
        c.localSymbol = (
            f"ABC   {c.lastTradeDateOrContractMonth[2:]}{c.right}{int(c.strike * 1000):08d}"
        )
        return [
            NS(
                contract=c,
                underConId=1,
                realExpirationDate=c.lastTradeDateOrContractMonth,
                lastTradeTime="16:00:00",
                timeZoneId="US/Eastern",
                orderTypes="LMT,GTD",
            )
        ]

    b.ib.reqContractDetailsAsync = AsyncMock(side_effect=qualify)
    return underlying


@pytest.mark.parametrize(
    "mode",
    [
        "empty",
        "unavailable",
        "exact_empty",
        "exact_unavailable",
        "valid",
        "timeout",
        "ambiguous",
        "ambiguous_error",
        "none",
    ],
)
def test_bounded_expiry_alternatives(tmp_path, mode):
    b = broker(tmp_path)
    underlying = qualification_fixture(b, mode)
    if mode in {"timeout", "ambiguous", "ambiguous_error", "none", "empty", "unavailable"}:
        with pytest.raises((ValueError, TimeoutError, RequestError)):
            asyncio.run(b.contracts(underlying, 100, OPEN))
    else:
        baseline = CLOSE if mode.startswith("exact_") else OPEN
        p, c, _ = asyncio.run(b.contracts(underlying, 100, baseline))
        expected = "20250723" if mode.startswith("exact_") else "20250722"
        assert p.realExpirationDate == c.realExpirationDate == expected
    assert b.ib.reqContractDetailsAsync.await_count <= 3
    assert {
        call.args[0].lastTradeDateOrContractMonth
        for call in b.ib.reqContractDetailsAsync.await_args_list
    } <= {"20250722", "20250723", "20250724"}


def test_session_block_during_entry_preparation_cannot_submit(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event = b.store.observe("2025-07-21", OPEN + timedelta(minutes=15), CLOSE, [candidate("A", 1)])[
        0
    ]
    monkeypatch.setattr(
        "stocker_execution.first4_broker.now", lambda: datetime.fromisoformat(event["entry_at"])
    )
    # Direct socket-boundary test after asynchronous preparation has completed.
    b.store.block(event["session"], "SCANNER_MINUTE_MISSED")
    with pytest.raises(ValueError, match="SCANNER_MINUTE_MISSED"):
        b.submit(event, Contract(secType="BAG"), Order(), {}, "ENTRY")
    assert not b.ib.placeOrder.called and not b.store.rows("orders")


def test_fault_injection_scanner_management_dashboard_and_persistence(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    payload["deadline_reconciled"] = True
    b.store.db.execute("UPDATE first4_orders SET payload=?", (json.dumps(payload),))
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    runtime.session, runtime.problem = event["session"], ""
    clock = CLOSE - timedelta(seconds=20)
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: clock)
    monkeypatch.setattr("stocker_execution.first4_runtime.now", lambda: clock)
    b.ib.scanner = AsyncMock(return_value=[])
    positions = [
        NS(account=b.config.expected_account, contract=Contract(conId=i), position=1, avgCost=100)
        for i in (101, 102)
    ]
    b.ib.reqPositionsAsync.return_value = positions
    open_trades = []
    b.ib.openTrades.return_value = open_trades
    b.ib.reqAllOpenOrdersAsync.return_value = open_trades

    def submit(contract, order):
        order.clientId = 81
        trade = Trade(
            contract=contract,
            order=order,
            orderStatus=OrderStatus(status="Submitted", permId=order.orderId),
        )
        open_trades.append(trade)
        return trade

    b.ib.placeOrder.side_effect = submit
    scanned = asyncio.Event()

    async def scanning():
        await runtime.scan(
            event["session"], OPEN, CLOSE, CLOSE - timedelta(days=3), CLOSE - timedelta(minutes=1)
        )
        scanned.set()
        await asyncio.Event().wait()

    runtime.scan_sessions = scanning

    async def check():
        worker = asyncio.create_task(runtime.run())
        await scanned.wait()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            for _ in range(3):
                assert (await client.get("/api/overview")).status_code == 200
                await asyncio.sleep(0.01)
            assert b.ib.placeOrder.call_count == 1
            b.error(-1, 1100, "recoverable upstream interruption")
            assert not b.entries_armed() and not runtime.status()["armed"]
            b.error(-1, 1102, "restored with data maintained")
            await b.reconcile()
            await b.close_due()
            assert b.ib.placeOrder.call_count == 1 and b.store.active_entries()
            assert b.store.rows("sessions")[0]["blocked"] is None
            runtime.store.set_meta = Mock(side_effect=sqlite3.OperationalError("disk full"))
            b.report_error("injected", {"error": "persistence failure"})
            with pytest.raises(sqlite3.OperationalError):
                await asyncio.wait_for(worker, 1)
            response = await client.get("/api/health")
            assert response.status_code == 503
            assert response.json()["manager_health"] == "FAILED"
            assert b.ib.placeOrder.call_count == 1 and b.store.active_entries()
            assert all(t.done() for t in runtime.critical_tasks)

    asyncio.run(check())


@pytest.mark.parametrize("web_failure", [False, True])
def test_cli_surfaces_worker_failure_and_web_failure_keeps_worker(
    tmp_path, monkeypatch, web_failure
):
    import uvicorn
    from typer.testing import CliRunner

    from stocker_core.cli import app

    web_failed = asyncio.Event()
    managed = []

    async def work(self):
        if web_failure:
            await web_failed.wait()
            await asyncio.sleep(0)
            assert self.web_health == "FAILED"
            managed.append("management continued")
        raise RuntimeError("worker failure reaches service supervisor")

    class Server:
        should_exit = False
        started = True

        def __init__(self, config):
            pass

        async def serve(self):
            if web_failure:
                web_failed.set()
                raise OSError("listener failed")
            await asyncio.Event().wait()

    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(Runtime, "run", work)
    monkeypatch.setattr(Runtime, "stop", AsyncMock())
    config = tmp_path / "config.yaml"
    config.write_text("armed: false\n")
    result = CliRunner().invoke(
        app, ["first4-run", "--config", str(config), "--database", str(tmp_path / "state.sqlite")]
    )
    assert result.exit_code != 0
    assert "worker failure reaches service supervisor" in str(result.exception)
    assert bool(managed) == web_failure


def test_pause_write_failure_is_not_reported_as_success(tmp_path):
    b = broker(tmp_path)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    runtime.store.set_meta = Mock(side_effect=sqlite3.OperationalError("disk full"))

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            response = await client.post("/api/first4/pause")
            assert response.status_code == 503
            assert runtime.pause and not runtime.broker.entries_armed()

    asyncio.run(check())


def test_opening_check_cannot_reauthorize_after_interruption(tmp_path, monkeypatch):
    from test_first4 import opening_runtime

    runtime = opening_runtime(tmp_path, monkeypatch)

    async def interrupted(*args):
        runtime.broker.disconnected()
        await runtime.broker.reconcile()
        return {
            "checks": {
                name: True
                for name in (
                    "qualified_usd_standard_multiplier",
                    "fresh_realtime_option_quotes",
                    "combo_price_increment",
                )
            },
            "blockers": [],
        }

    probe = AsyncMock(side_effect=interrupted)
    monkeypatch.setattr("stocker_execution.first4_runtime.option_access", probe)

    async def check():
        await runtime.check_opening("2025-07-21", OPEN)
        await runtime.arm_at_open()

    asyncio.run(check())
    report = runtime.store.get_meta("opening_check:2025-07-21")
    assert report["status"] == "FAILED" and report["error"] == "OPENING_CHECK_INTERRUPTED"
    assert runtime.broker.opening_verified_session is None and probe.await_count == 1


def test_history_farm_failure_does_not_prevent_fresh_exit_quotes(tmp_path):
    b = broker(tmp_path)
    b.error(-1, 2105, "Historical data farm disconnected:ushmds")
    assert not b.entries_armed()

    class Updates:
        def __iadd__(self, callback):
            asyncio.get_running_loop().call_soon(callback, ticker)
            return self

        def __isub__(self, callback):
            return self

    stamp = datetime.now(UTC)
    ticker = NS(
        bid=1,
        ask=1.1,
        bidSize=1,
        askSize=1,
        marketDataType=1,
        ticks=[NS(tickType=i, time=stamp) for i in (1, 2)],
        updateEvent=Updates(),
    )
    b.ib.reqMktData = Mock(return_value=ticker)
    b.ib.cancelMktData = Mock()

    async def check():
        quotes = await b.quotes([Contract(conId=101)], stamp + timedelta(seconds=1), side="SELL")
        assert quotes[0]["bid"] == 1

    asyncio.run(check())
    assert b.ib.cancelMktData.call_count == 1


@pytest.mark.parametrize("interrupt", [False, True])
def test_pre_change_schema_migration_preserves_ledger(tmp_path, interrupt):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE first4_orders (
            reference TEXT PRIMARY KEY, session TEXT, symbol TEXT, role TEXT,
            order_id INTEGER, perm_id INTEGER, status TEXT, payload TEXT);
        CREATE TABLE first4_fills (
            exec_id TEXT PRIMARY KEY, reference TEXT, con_id INTEGER, quantity REAL,
            price REAL, side TEXT, multiplier REAL, time TEXT, commission REAL);
        INSERT INTO first4_orders VALUES ('F4:2025-07-21:1:ENTRY','2025-07-21','A','ENTRY',
            1,10,'Cancelled','{}');
        INSERT INTO first4_fills VALUES ('old.01','F4:2025-07-21:1:ENTRY',101,1,1,'BOT',100,
            '2025-07-21T13:46:00+00:00',NULL);
        INSERT INTO first4_fills VALUES ('old.02','F4:2025-07-21:1:ENTRY',101,0.5,1,'BOT',100,
            '2025-07-21T13:46:00+00:00',NULL);
    """)
    if interrupt:
        db.executescript("""
            CREATE TRIGGER fail_migration BEFORE UPDATE ON first4_fills BEGIN
                SELECT RAISE(ABORT,'injected migration failure');
            END;
        """)
    db.close()
    if interrupt:
        with pytest.raises(sqlite3.IntegrityError, match="injected migration failure"):
            Store(path)
        with sqlite3.connect(path) as recovery:
            assert "superseded" not in {
                r[1] for r in recovery.execute("PRAGMA table_info(first4_fills)")
            }
            recovery.execute("DROP TRIGGER fail_migration")
    store = Store(path)
    assert len(store.active_entries()) == 1
    assert store.active_entries()[0]["order_id"] == 1
    assert len(store.rows("fills")) == 2
    b = PaperBroker(armed_config(), store, fake_ib())
    assert b.owned_quantities() == {101: 0.5}
    assert Store(path).rows("orders") == store.rows("orders")


def seed_closed_history(store, count):
    """Synthetic settled allocations; no broker state or market data."""
    with store.db:
        for i in range(count):
            day = (date(2020, 1, 1) + timedelta(days=i)).isoformat()
            at = datetime.combine(date.fromisoformat(day), datetime.min.time(), tzinfo=UTC)
            event = store.observe(day, at, at + timedelta(hours=7), [candidate("OLD", 1)])[0]
            payload = {
                "entry_deadline_at": at.isoformat(),
                "deadline_reconciled": True,
                "exit_at": at.isoformat(),
                "put": {"secType": "OPT", "conId": 901},
                "call": {"secType": "OPT", "conId": 902},
                "quantity": 1,
                "exit_quantity": 1,
                "exit_con_ids": [901, 902],
            }
            entry = store.reserve_order(event, "ENTRY", i * 2, payload)
            exit_ref = store.reserve_order(event, "EXIT", i * 2 + 1, payload)
            for reference, side, price in ((entry, "BOT", 1), (exit_ref, "SLD", 1.2)):
                for con_id in (901, 902):
                    store.db.execute(
                        "INSERT INTO first4_fills "
                        "(exec_id,reference,con_id,quantity,price,side,multiplier,time,commission) "
                        "VALUES (?,?,?,1,?,?,100,?,0.65)",
                        (reference + str(con_id), reference, con_id, price, side, at.isoformat()),
                    )
            store.db.execute("UPDATE first4_orders SET status='Filled' WHERE session=?", (day,))
            store.db.execute("UPDATE first4_orders SET obligation_done=1 WHERE session=?", (day,))
            store.outcome(event, "CLOSED")


def test_management_and_dashboard_cost_independent_of_closed_history(tmp_path, monkeypatch):
    b = broker(tmp_path)
    event, payload = prepare_exit(b)
    payload["deadline_reconciled"] = True
    b.store.db.execute("UPDATE first4_orders SET payload=?", (json.dumps(payload),))
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN + timedelta(hours=1))
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    runtime.session = event["session"]
    counts = []
    sizes = []
    quantity_steps = []

    async def measure():
        steps = 0

        def progress():
            nonlocal steps
            steps += 1
            return 0

        b.store.db.set_progress_handler(progress, 1)
        assert b.owned_quantities() == {101: 1, 102: 1}
        b.store.db.set_progress_handler(None, 0)
        quantity_steps.append(steps)
        statements = []
        b.store.db.set_trace_callback(statements.append)
        before = b.store.db.total_changes
        await b.cancel_due_entries()
        await b.close_due()
        counts.append((len(statements), b.store.db.total_changes - before))
        b.store.db.set_trace_callback(None)
        assert not any(sql == "SELECT * FROM first4_orders" for sql in statements)
        assert not any(sql == "SELECT * FROM first4_fills" for sql in statements)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(create_dashboard_app(runtime)),
            base_url="http://127.0.0.1",
        ) as client:
            response = await client.get("/api/overview")
            assert response.status_code == 200
            sizes.append(len(response.content))
            assert len(response.json()["orders"]) == 1
            assert len(response.json()["fills"]) == 2

    asyncio.run(measure())
    seed_closed_history(b.store, 1000)
    asyncio.run(measure())
    assert counts[0] == counts[1] == (3, 0)
    assert sizes[0] == sizes[1]
    assert quantity_steps[1] <= quantity_steps[0] + 20
    assert len(b.store.active_entries()) == 1
    print(
        {"management_queries_writes": counts, "overview_bytes": sizes, "closed_allocations": 1000}
    )
