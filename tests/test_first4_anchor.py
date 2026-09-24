"""Offline wire-tick tests for FIRST4's frozen baseline minute."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ib_async import IB, Contract

from stocker_execution import first4_runtime as module
from stocker_execution.first4_requests import First4IB

BASELINE = datetime(2026, 9, 24, 14, 1, tzinfo=UTC)


def deliver(ib, stamp, received, price="1.25"):
    ib.wrapper.tcpDataArrived()
    ib.wrapper.lastTime = received
    ib.client.decoder.interpret(
        ["99", "42", "1", str(int(stamp.timestamp())), price, "100", "0", "TEST", ""]
    )
    ib.wrapper.tcpDataProcessed()


def test_wire_trade_time_preserved_without_changing_receipt_time():
    ib = First4IB()
    ticker = ib.wrapper.startTicker(42, Contract(conId=123), "Last")
    received = BASELINE + timedelta(milliseconds=200)
    trade_time = BASELINE - timedelta(seconds=1)
    deliver(ib, trade_time, received)
    assert ticker.tickByTicks[0].time == trade_time
    assert ticker.time == received and ib.wrapper.lastTime == received
    assert ticker.last == 1.25


def test_timestamp_adapter_is_instance_local_and_survives_wrapper_reset():
    adapted, ordinary = First4IB(), IB()
    adapted.wrapper.reset()
    for ib in (adapted, ordinary):
        ib.wrapper.startTicker(42, Contract(conId=123), "Last")
        deliver(ib, BASELINE, BASELINE + timedelta(seconds=2))
    assert adapted.wrapper.reqId2Ticker[42].tickByTicks[0].time == BASELINE
    assert ordinary.wrapper.reqId2Ticker[42].tickByTicks[0].time == BASELINE + timedelta(seconds=2)


def execution(monkeypatch):
    ib = First4IB()
    ib.client.getReqId = Mock(return_value=42)
    ib.client.reqTickByTickData = Mock()
    ib.client.cancelTickByTickData = Mock()
    runtime = module.Runtime.__new__(module.Runtime)
    runtime.pause = False
    runtime.config = SimpleNamespace(missing=lambda: [], number=lambda key: 180)
    runtime.store = SimpleNamespace(outcome=Mock())
    runtime.broker = SimpleNamespace(
        ib=ib,
        entries_armed=lambda: True,
        guard=lambda: None,
        data_generation=0,
        market_data_block="",
        chain=AsyncMock(return_value=[]),
        enter=AsyncMock(),
    )
    clock = [BASELINE - timedelta(seconds=1)]
    monkeypatch.setattr(module, "now", lambda: clock[0])
    event = {"symbol": "TEST", "entry_at": BASELINE.isoformat()}
    underlying = Contract(conId=123, symbol="TEST", secType="STK")
    return runtime, ib, clock, event, underlying


@pytest.mark.parametrize("valid", [False, True])
def test_execution_rejects_late_previous_minute_tick_and_uses_first_valid(monkeypatch, valid):
    async def check():
        runtime, ib, clock, event, underlying = execution(monkeypatch)

        async def chain(*args):
            received = BASELINE + timedelta(seconds=2)
            deliver(ib, BASELINE - timedelta(seconds=1), received, "9.99")
            if valid:
                deliver(ib, BASELINE + timedelta(seconds=1), received, "1.25")
                deliver(ib, BASELINE + timedelta(seconds=2), received, "1.50")
            clock[0] = BASELINE + timedelta(minutes=1)
            return []

        runtime.broker.chain.side_effect = chain
        await runtime.execute(event, underlying)
        if valid:
            runtime.broker.enter.assert_awaited_once_with(event, underlying, 1.25)
            detail = runtime.store.outcome.call_args_list[0].args[2]
            assert detail["broker_trade_time"] == (BASELINE + timedelta(seconds=1)).isoformat()
        else:
            runtime.broker.enter.assert_not_awaited()
            detail = runtime.store.outcome.call_args.args[2]
            assert detail["error"] == "BASELINE_TRADE_NOT_RECEIVED"
            assert detail["stage"] == "BASELINE_ANCHOR"
            assert detail["ticks_received"] == 1
        ib.client.cancelTickByTickData.assert_called_once_with(42)
        ticker = ib.ticker(underlying)
        assert len(ticker.updateEvent) == 0
        assert not ib.wrapper.ticker2ReqId["Last"]

    asyncio.run(check())


@pytest.mark.parametrize("stage", ["OPTION_CHAIN", "BASELINE_ANCHOR", "ENTRY_PREPARATION"])
def test_execution_timeout_reports_stage_and_cleans_subscription(monkeypatch, stage):
    async def check():
        runtime, ib, clock, event, underlying = execution(monkeypatch)

        async def chain(*args):
            if stage == "OPTION_CHAIN":
                raise TimeoutError()
            if stage == "ENTRY_PREPARATION":
                deliver(ib, BASELINE, BASELINE)
            clock[0] = BASELINE + timedelta(minutes=1)
            return []

        runtime.broker.chain.side_effect = chain
        runtime.broker.enter.side_effect = TimeoutError()
        await runtime.execute(event, underlying)
        detail = runtime.store.outcome.call_args.args[2]
        assert detail["stage"] == stage
        assert detail["error"] == (
            "BASELINE_TRADE_NOT_RECEIVED" if stage == "BASELINE_ANCHOR" else stage + "_TIMEOUT"
        )
        assert detail["exception"] == "TimeoutError()"
        if stage != "ENTRY_PREPARATION":
            runtime.broker.enter.assert_not_awaited()
        ib.client.cancelTickByTickData.assert_called_once_with(42)

    asyncio.run(check())


def test_anchor_wait_cancellation_propagates_and_releases_callback(monkeypatch):
    async def check():
        runtime, ib, clock, event, underlying = execution(monkeypatch)
        task = asyncio.create_task(runtime.execute(event, underlying))
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        runtime.broker.enter.assert_not_awaited()
        runtime.store.outcome.assert_not_called()
        ib.client.cancelTickByTickData.assert_called_once_with(42)
        assert len(ib.ticker(underlying).updateEvent) == 0

    asyncio.run(check())
