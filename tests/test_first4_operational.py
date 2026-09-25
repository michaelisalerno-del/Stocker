"""Controlled clocks and wire callbacks; no broker connection or wall-clock sleeps."""

import asyncio
import json
import socket
import sqlite3
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from ib_async import Contract, OrderStatus, RequestError, Trade

from stocker_execution import first4_broker as broker_module
from stocker_execution import first4_runtime as runtime_module
from stocker_execution.first4_requests import First4IB
from stocker_execution.first4_store import Store
from test_first4 import CLOSE, OPEN, armed_config, candidate, opening_runtime
from test_first4_opening_retry import evidence


@pytest.fixture(autouse=True)
def no_wall_clock_waits(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Operational regressions must remain offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    sleep, wait = asyncio.sleep, asyncio.wait

    async def yield_only(seconds=0):
        await sleep(0)

    async def poll(futures, *, timeout=None, return_when=asyncio.ALL_COMPLETED):
        await sleep(0)
        return await wait(futures, timeout=0, return_when=return_when)

    monkeypatch.setattr(asyncio, "sleep", yield_only)
    monkeypatch.setattr(asyncio, "wait", poll)


def wired(tmp_path, monkeypatch):
    ib = First4IB()
    ib.isConnected = Mock(return_value=True)
    ib.client.getReqId = Mock(side_effect=range(42, 100))
    ib.client.reqMktData = Mock()
    ib.client.cancelMktData = Mock()
    ib.placeOrder = Mock()
    b = broker_module.PaperBroker(armed_config(), Store(tmp_path / "wire.sqlite"), ib)
    b.guard = Mock()
    clock = [OPEN]
    monkeypatch.setattr(broker_module, "now", lambda: clock[0])
    return b, ib, clock


def packet(ib, clock, ticks=(), data_type=1, req=42):
    ib.wrapper.tcpDataArrived()
    ib.wrapper.lastTime = clock
    ib.wrapper.marketDataType(req, data_type)
    for kind, price in ticks:
        ib.wrapper.priceSizeTick(req, kind, price, 10)
    ib.wrapper.tcpDataProcessed()


def clean(ib):
    assert not ib.wrapper._futures and not ib.wrapper._results
    assert not ib.wrapper._reqId2Contract and not ib.wrapper.reqId2Ticker
    assert not ib.wrapper.ticker2ReqId["mktData"]
    assert all(len(t.updateEvent) == 0 for t in ib.wrapper.tickers.values())
    ib.placeOrder.assert_not_called()


def test_old_snapshot_ages_early_valid_price_before_completion(tmp_path, monkeypatch):
    async def check():
        b, ib, clock = wired(tmp_path, monkeypatch)
        task = asyncio.create_task(ib.reqTickersAsync(Contract(conId=1)))
        await asyncio.sleep(0)
        packet(ib, OPEN, [(4, 13)])
        assert not task.done()
        clock[0] += timedelta(seconds=11)
        ib.wrapper.tickSnapshotEnd(42)
        ticker = (await task)[0]
        assert ticker.marketPrice() == 13
        assert (clock[0] - ticker.time).total_seconds() > b.config.number("quote_max_age_seconds")
        clean(ib)

    asyncio.run(check())


def test_fresh_diagnostic_stream_finishes_without_snapshot_end(tmp_path, monkeypatch):
    async def check():
        b, ib, clock = wired(tmp_path, monkeypatch)
        task = asyncio.create_task(
            b.stock_reference(Contract(conId=1), OPEN + timedelta(seconds=60), {})
        )
        await asyncio.sleep(0)
        packet(ib, OPEN, [(4, 13)])
        result = await task
        assert result == 13 and clock[0] == OPEN
        assert ib.client.reqMktData.call_count == 1
        assert ib.client.reqMktData.call_args.args[3] is False
        ib.client.cancelMktData.assert_called_once_with(42)
        clean(ib)

    asyncio.run(check())


@pytest.mark.parametrize("method", ["quotes", "combo_tick"])
def test_market_data_request_error_reaches_wait_and_cleans(tmp_path, monkeypatch, method):
    async def check():
        b, ib, clock = wired(tmp_path, monkeypatch)
        contract = Contract(conId=1)
        task = asyncio.create_task(
            getattr(b, method)(
                [contract] if method == "quotes" else contract, OPEN + timedelta(seconds=60)
            )
        )
        await asyncio.sleep(0)
        ib.wrapper.error(42, 354, "Synthetic missing subscription", "")
        # Advance the deadline: old implementation masks this as an unavailable quote.
        clock[0] += timedelta(seconds=60)
        with pytest.raises(RequestError) as error:
            await task
        assert error.value.reqId == 42
        clean(ib)

    asyncio.run(check())


def test_fifth_opening_attempt_can_recover_with_original_deadline(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    clock = [OPEN]
    monkeypatch.setattr(runtime_module, "now", lambda: clock[0])
    deadlines = []

    async def pause(seconds):
        assert seconds == 5
        clock[0] += timedelta(seconds=seconds)

    monkeypatch.setattr(runtime_module, "opening_pause", pause, raising=False)

    async def probe(*args):
        deadlines.append(args[1])
        if len(deadlines) < 5:
            raise ValueError("PROBE_STOCK_QUOTE_UNAVAILABLE")
        return evidence()

    monkeypatch.setattr(runtime_module, "option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    assert len(deadlines) == 5
    assert runtime.broker.opening_verified_session == "2025-07-21"
    report = runtime.store.get_meta("opening_check:2025-07-21")
    assert len(report["attempts"]) == 5
    assert [r["outcome"] for r in report["attempts"]] == ["FAILED"] * 4 + ["PASSED"]
    assert "last_attempt_error" not in report and "max_attempts" not in report
    assert deadlines == [OPEN + timedelta(seconds=60 + 5 * i) for i in range(5)]


@pytest.mark.parametrize(
    "case",
    [
        "stale",
        "future",
        "delayed",
        "frozen",
        "missing",
        "nan",
        "infinite",
        "zero",
        "negative",
        "cached",
        "unrelated",
        "size_only",
        "no_type",
    ],
)
def test_stock_reference_cannot_pass_bad_or_unrelated_data(tmp_path, monkeypatch, case):
    async def check():
        b, ib, clock = wired(tmp_path, monkeypatch)
        contract = Contract(conId=1)
        cached = ib.wrapper.startTicker(99, contract, "Last")
        cached.last, cached.marketDataType, cached.time = 13, 1, OPEN
        audit = {}
        task = asyncio.create_task(b.stock_reference(contract, OPEN + timedelta(seconds=60), audit))
        await asyncio.sleep(0)
        prices = {"nan": float("nan"), "infinite": float("inf"), "zero": 0, "negative": -1}
        stamp = OPEN + timedelta(seconds=-6 if case == "stale" else 1 if case == "future" else 0)
        kinds = [] if case in {"missing", "cached", "size_only"} else [(4, prices.get(case, 13))]
        if case == "no_type":
            ib.wrapper.tcpDataArrived()
            ib.wrapper.lastTime = stamp
            ib.wrapper.priceSizeTick(42, 4, 13, 10)
            ib.wrapper.tcpDataProcessed()
        else:
            packet(
                ib,
                stamp,
                kinds,
                3 if case == "delayed" else 2 if case == "frozen" else 1,
                req=99 if case == "unrelated" else 42,
            )
        if case in {"stale", "size_only"}:
            # Fresh size/general ticker time must not renew an old price.
            ib.wrapper.tcpDataArrived()
            ib.wrapper.lastTime = OPEN
            ib.wrapper.tickSize(42, 5, 100)
            ib.wrapper.tcpDataProcessed()
        for _ in range(8):
            await asyncio.sleep(0)
        assert not task.done()
        assert audit["stock_quote"]["failed_predicate"]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ib.wrapper.reqId2Ticker[99] is cached
        ib.wrapper.endTicker(cached, "Last")
        ib.wrapper.reqId2Ticker.pop(99)
        clean(ib)
        ib.client.cancelMktData.assert_called_once_with(42)

    asyncio.run(check())


def test_price_is_revalidated_when_task_consumes_update(tmp_path, monkeypatch):
    async def check():
        b, ib, clock = wired(tmp_path, monkeypatch)
        audit = {}
        task = asyncio.create_task(
            b.stock_reference(Contract(conId=1), OPEN + timedelta(seconds=60), audit)
        )
        await asyncio.sleep(0)
        packet(ib, OPEN, [(4, 13)])
        # Event is set but the consumer has not run yet.
        clock[0] += timedelta(seconds=6)
        for _ in range(8):
            await asyncio.sleep(0)
        assert not task.done()
        assert audit["stock_quote"]["ages_seconds"] == [6]
        packet(ib, clock[0], [(4, 14)])
        assert await task == 14
        assert audit["stock_quote"]["reference_price"] == 14
        assert audit["stock_quote"]["ages_seconds"] == [0]
        clean(ib)

    asyncio.run(check())


@pytest.mark.parametrize(
    "change", ["pause", "continuity", "permission", "fatal", "account", "ownership", "cancel"]
)
def test_safety_interrupts_a_pending_opening_attempt(tmp_path, monkeypatch, change):
    runtime = opening_runtime(tmp_path, monkeypatch)

    async def check():
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def blocked(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        monkeypatch.setattr(runtime_module, "option_access", blocked)
        task = asyncio.create_task(runtime.check_opening("2025-07-21", OPEN))
        await entered.wait()
        if change == "pause":
            runtime.pause = True
        elif change == "continuity":
            runtime.broker.data_generation += 1
        elif change == "permission":
            runtime.config = runtime.config.model_copy(update={"arm_after_quote_check_on": None})
        elif change == "fatal":
            runtime.broker.fatal_error = "FATAL_PERSISTENCE"
        elif change == "account":
            runtime.broker.ib.managedAccounts.return_value = ["wrong"]
        elif change == "ownership":
            runtime.broker.entry_blocker = "UNOWNED_PENDING_ORDERS"
        else:
            task.cancel()
        if change == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert cleaned.is_set()
        assert runtime.broker.opening_verified_session is None
        assert runtime.store.get_meta("opening_check:2025-07-21")["attempt"] == 1
        runtime.broker.ib.placeOrder.assert_not_called()

    asyncio.run(check())


def test_timeout_cannot_hide_fatal_error_raised_during_cleanup(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)

    async def probe(*args):
        try:
            monkeypatch.setattr(runtime_module, "now", lambda: OPEN + timedelta(seconds=60))
            await asyncio.Event().wait()
        finally:
            raise ValueError("OWNED_ORDER_IDENTITY_MISMATCH")

    monkeypatch.setattr(runtime_module, "option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    report = runtime.store.get_meta("opening_check:2025-07-21")
    assert report["attempt"] == 1 and report["error"] == "OWNED_ORDER_IDENTITY_MISMATCH"
    assert not runtime.broker.opening_verified_session


@pytest.mark.parametrize("case", ["valid", "contract", "quote", "ambiguous", "slow_metadata"])
@pytest.mark.parametrize("observe", [False, True])
def test_complete_allocated_candidate_to_paper_submission_boundary(
    tmp_path, monkeypatch, case, observe
):
    async def check():
        b, ib, clock = wired(tmp_path, monkeypatch)
        b.guard = broker_module.PaperBroker.guard.__get__(b)
        b.reconciled, b.entry_blocker = True, ""
        ib.managedAccounts = Mock(return_value=[b.config.expected_account])
        ib.client.clientId = b.config.client_id
        ib.client.reqTickByTickData = Mock()
        ib.client.cancelTickByTickData = Mock()
        runtime = runtime_module.Runtime(b.config, b.store)
        runtime.broker = b
        event = b.store.observe(
            "2025-07-21", OPEN + timedelta(minutes=15), CLOSE, [candidate("A", 1)]
        )[0]
        baseline = broker_module.datetime.fromisoformat(event["entry_at"])
        clock[0] = baseline - timedelta(seconds=1)
        monkeypatch.setattr(runtime_module, "now", lambda: clock[0])
        underlying = Contract(conId=1, symbol="A", secType="STK")

        async def chain(*args):
            from test_first4_anchor import deliver

            clock[0] = baseline
            if observe:
                ib.flow_wire.retain_last(underlying, lambda *args, **kwargs: None)
            deliver(ib, baseline, baseline, "10")
            await asyncio.sleep(0)
            return [
                NS(
                    exchange="SMART",
                    tradingClass="A",
                    multiplier="100",
                    expirations={"20250723"},
                    strikes={9.8, 10.2},
                )
            ]

        ib.reqSecDefOptParamsAsync = AsyncMock(side_effect=chain)

        def details(c):
            c.conId = 101 if c.right == "P" else 102
            c.localSymbol = f"{'A':<6}250723{c.right}{round(c.strike * 1000):08d}"
            if case == "contract":
                c.currency = "EUR"
            return [
                NS(
                    contract=c,
                    underConId=1,
                    realExpirationDate="20250723",
                    lastTradeTime="16:00:00",
                    timeZoneId="US/Eastern",
                    orderTypes="LMT,GTD",
                    sizeIncrement=1,
                    minSize=1,
                )
            ]

        ib.reqContractDetailsAsync = AsyncMock(side_effect=details)

        def market(req, c, *args):
            if c.secType == "BAG":
                if case == "slow_metadata":

                    async def later_metadata():
                        for _ in range(8):
                            await asyncio.sleep(0)
                        clock[0] += timedelta(seconds=6)
                        for option_req, ticker in list(ib.wrapper.reqId2Ticker.items()):
                            if ticker.contract.secType == "OPT":
                                packet(ib, clock[0], [(1, 0.2), (2, 0.3)], req=option_req)
                        ib.wrapper.tickReqParams(req, 0.01, "", 0)

                    asyncio.create_task(later_metadata())
                else:
                    ib.wrapper.tickReqParams(req, 0.01, "", 0)
            else:
                packet(ib, clock[0], [(1, 0.2), (2, 0.3)], 3 if case == "quote" else 1, req=req)
                if case == "quote":
                    clock[0] = baseline + timedelta(
                        seconds=b.config.number("entry_deadline_seconds")
                    )

        ib.client.reqMktData.side_effect = market

        def submit(c, order):
            assert b.store.rows("orders")[0]["status"] == "RESERVED"
            assert c.secType == "BAG" and order.account == b.config.expected_account
            assert order.totalQuantity == 1 and order.tif == "GTD"
            if case == "ambiguous":
                raise ConnectionError("Synthetic ambiguous socket write")
            return Trade(
                contract=c, order=order, orderStatus=OrderStatus(status="Submitted", permId=22)
            )

        ib.placeOrder.side_effect = submit
        await runtime.execute(event, underlying)
        if observe:
            assert ib.flow_last_retained(42)
            ib.flow_wire.release_last(underlying.conId)
        row = b.store.event(event["session"], event["symbol"])
        detail = row["detail"]
        if isinstance(detail, str):
            detail = json.loads(detail)
        assert ib.reqSecDefOptParamsAsync.await_count == 1
        assert not ib.wrapper._futures and not ib.wrapper.reqId2Ticker
        assert not ib.wrapper._reqId2Contract and not ib.wrapper.ticker2ReqId["mktData"]
        assert not ib.wrapper.ticker2ReqId["Last"]
        if case in {"valid", "slow_metadata"}:
            assert row["outcome"] == "ORDER_SUBMITTED"
            assert ib.placeOrder.call_count == 1
            assert len(b.store.rows("fills")) == 0
            assert ib.client.reqMktData.call_count == 3
        else:
            assert row["outcome"] == "EXECUTION_FAILED"
            assert detail["anchor_received"] and not detail["fill_observed"]
            assert (
                detail["stage"]
                == {
                    "contract": "CONTRACT_QUALIFICATION",
                    "quote": "OPTION_QUOTES",
                    "ambiguous": "ORDER_SUBMISSION",
                }[case]
            )
            if case == "ambiguous":
                assert detail["submission_unresolved"] and not detail["submitted_order_observed"]
                with pytest.raises(sqlite3.IntegrityError):
                    await b.enter(event, underlying, 10)
                assert ib.placeOrder.call_count == 1
                assert len(b.store.rows("orders")) == 1
            else:
                ib.placeOrder.assert_not_called()
                assert not detail["order_reserved"]

    asyncio.run(check())


@pytest.mark.parametrize("stage", ["RECONCILIATION", "OPTION_ACCESS"])
def test_reconciliation_timeout_is_terminal_and_attempt_budget_is_clipped(
    tmp_path, monkeypatch, stage
):
    runtime = opening_runtime(tmp_path, monkeypatch)
    clock = OPEN + timedelta(minutes=13, seconds=30)
    monkeypatch.setattr(runtime_module, "now", lambda: clock)
    if stage == "RECONCILIATION":
        runtime.broker.reconcile = AsyncMock(side_effect=TimeoutError("reconcile timeout"))
        probe = AsyncMock()
    else:

        async def check_deadline(broker, deadline):
            assert deadline == OPEN + timedelta(minutes=14)
            return evidence()

        probe = AsyncMock(side_effect=check_deadline)
    monkeypatch.setattr(runtime_module, "option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    report = runtime.store.get_meta("opening_check:2025-07-21")
    assert report["attempt"] == 1
    assert report["status"] == ("FAILED" if stage == "RECONCILIATION" else "ARMED")
    if stage == "RECONCILIATION":
        probe.assert_not_awaited()


@pytest.mark.parametrize("historical", ["ARMED", "CHECKING", "FAILED"])
def test_historical_opening_check_is_never_replayed(tmp_path, monkeypatch, historical):
    runtime = opening_runtime(tmp_path, monkeypatch)
    runtime.store.set_meta("opening_check:2025-07-21", {"status": historical})
    probe = AsyncMock()
    monkeypatch.setattr(runtime_module, "option_access", probe)
    asyncio.run(runtime.arm_at_open())
    probe.assert_not_awaited()
    assert not runtime.status()["armed"] and not runtime.status()["opening_check_active"]


@pytest.mark.parametrize("method", ["stock_reference", "quotes", "combo_tick"])
def test_cancelling_each_market_data_wait_releases_request_and_callback(
    tmp_path, monkeypatch, method
):
    async def check():
        b, ib, clock = wired(tmp_path, monkeypatch)
        contract = Contract(conId=1)
        args = ([contract] if method == "quotes" else contract, OPEN + timedelta(seconds=60))
        work = (
            getattr(b, method)(*args, {})
            if method == "stock_reference"
            else getattr(b, method)(*args)
        )
        task = asyncio.create_task(work)
        await asyncio.sleep(0)
        ticker = ib.wrapper.reqId2Ticker[42]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(ticker.updateEvent) == 0
        ib.client.cancelMktData.assert_called_once_with(42)
        clean(ib)

    asyncio.run(check())
