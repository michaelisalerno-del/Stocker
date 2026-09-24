"""Bounded read-only retries cannot replay or restore dated entry authority."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from ib_async import Contract, RequestError

from stocker_execution import first4_runtime as module
from stocker_execution.first4_requests import First4IB
from test_first4 import OPEN, opening_runtime


def evidence():
    return {
        "checks": {
            "qualified_usd_standard_multiplier": True,
            "fresh_realtime_option_quotes": True,
            "combo_price_increment": True,
        },
        "blockers": [],
    }


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError(),
        ValueError("PROBE_STOCK_QUOTE_UNAVAILABLE"),
        ValueError("OPTION_QUOTES_INVALID_STALE_OR_UNAVAILABLE"),
        ValueError("COMBO_PRICE_INCREMENT_UNAVAILABLE"),
    ],
)
def test_transient_check_retries_and_arms_only_after_success(tmp_path, monkeypatch, failure):
    runtime = opening_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "OPENING_RETRY_SECONDS", 0)
    calls = 0

    async def probe(*args):
        nonlocal calls
        calls += 1
        assert runtime.store.get_meta("opening_check:2025-07-21")["status"] == "CHECKING"
        assert not runtime.broker.opening_verified_session
        if calls == 1:
            raise failure
        return evidence()

    monkeypatch.setattr(module, "option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    report = runtime.store.get_meta("opening_check:2025-07-21")
    assert report["status"] == "ARMED" and report["attempt"] == calls == 2
    assert report["last_attempt_error"] and "next_attempt_at" not in report
    runtime.broker.ib.placeOrder.assert_not_called()


def test_retry_limit_is_terminal_and_not_replayed(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "OPENING_RETRY_SECONDS", 0)
    probe = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(module, "option_access", probe)

    async def check():
        await runtime.check_opening("2025-07-21", OPEN)
        await runtime.arm_at_open()

    asyncio.run(check())
    assert probe.await_count == 3
    assert runtime.store.get_meta("opening_check:2025-07-21")["status"] == "FAILED"
    assert not runtime.broker.opening_verified_session


def test_final_attempt_preserves_original_quote_window(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "OPENING_RETRY_SECONDS", 0)
    deadlines = []

    async def probe(broker, deadline):
        deadlines.append(deadline)
        if len(deadlines) < 3:
            raise TimeoutError()
        return evidence()

    monkeypatch.setattr(module, "option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    assert deadlines == [OPEN + timedelta(seconds=60)] * 2 + [OPEN + timedelta(minutes=14)]
    assert runtime.broker.opening_verified_session == "2025-07-21"


@pytest.mark.parametrize(
    "change", ["pause", "session", "account", "permission", "disconnect", "10197"]
)
def test_retry_cannot_cross_a_safety_block(tmp_path, monkeypatch, change):
    runtime = opening_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "OPENING_RETRY_SECONDS", 0)

    async def fail(*args):
        if change == "pause":
            runtime.store.set_meta("paused", True)
        elif change == "session":
            runtime.store.block("2025-07-21", "SCANNER_MINUTE_MISSED")
        elif change == "account":
            runtime.broker.ib.managedAccounts.return_value = ["wrong"]
        elif change == "permission":
            runtime.config.arm_after_quote_check_on = None
        elif change == "disconnect":
            runtime.broker.disconnected()
            await runtime.broker.reconcile()
        else:
            runtime.broker.error(1, 10197, "Competing session")
        raise TimeoutError()

    probe = AsyncMock(side_effect=fail)
    monkeypatch.setattr(module, "option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    assert probe.await_count == 1 and not runtime.broker.opening_verified_session
    assert runtime.store.get_meta("opening_check:2025-07-21")["status"] == "FAILED"


def test_retry_never_extends_opening_window(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)

    async def late(*args):
        monkeypatch.setattr(module, "now", lambda: OPEN + timedelta(minutes=14, seconds=-2))
        raise TimeoutError()

    probe = AsyncMock(side_effect=late)
    monkeypatch.setattr(module, "option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    report = runtime.store.get_meta("opening_check:2025-07-21")
    assert probe.await_count == 1 and report["error"] == "OPENING_VERIFICATION_WINDOW_EXPIRED"


def test_timed_out_attempt_is_drained_before_retry(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "OPENING_RETRY_SECONDS", 0)
    monkeypatch.setattr(module, "OPENING_ATTEMPT_SECONDS", 0.01)
    monkeypatch.setattr(module, "now", lambda: OPEN + timedelta(minutes=14, milliseconds=-10))
    active, attempts, cleaned = 0, 0, 0

    async def slow(*args):
        nonlocal active, attempts, cleaned
        assert active == 0
        active += 1
        attempts += 1
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
            cleaned += 1

    monkeypatch.setattr(module, "option_access", slow)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    assert attempts == cleaned == 3 and active == 0


def test_cancel_during_retry_backoff_stays_failed(tmp_path, monkeypatch):
    runtime = opening_runtime(tmp_path, monkeypatch)
    probe = AsyncMock(side_effect=TimeoutError())
    monkeypatch.setattr(module, "option_access", probe)

    async def check():
        task = asyncio.create_task(runtime.check_opening("2025-07-21", OPEN))
        for _ in range(20):
            await asyncio.sleep(0)
            if runtime.store.get_meta("opening_check:2025-07-21", {}).get("next_attempt_at"):
                break
        assert runtime.store.get_meta("opening_check:2025-07-21")["status"] == "CHECKING"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await runtime.arm_at_open()

    asyncio.run(check())
    assert probe.await_count == 1
    assert runtime.store.get_meta("opening_check:2025-07-21")["status"] == "FAILED"


@pytest.mark.parametrize(
    "failure",
    [ValueError("PROBE_CONTRACT_AMBIGUOUS"), RequestError(1, 354, "No data subscription")],
)
def test_nontransient_failure_does_not_retry(tmp_path, monkeypatch, failure):
    runtime = opening_runtime(tmp_path, monkeypatch)
    probe = AsyncMock(side_effect=failure)
    monkeypatch.setattr(module, "option_access", probe)
    asyncio.run(runtime.check_opening("2025-07-21", OPEN))
    assert probe.await_count == 1 and not runtime.broker.opening_verified_session


def test_snapshot_request_cleanup_for_success_error_and_cancel():
    async def check():
        ib = First4IB()
        ib.isConnected = Mock(return_value=True)
        ib.client.getReqId = Mock(side_effect=range(10, 20))
        ib.client.cancelMktData = Mock()
        ib.client.reqMktData = Mock(side_effect=lambda req, *args: ib.wrapper.tickSnapshotEnd(req))
        assert len(await ib.reqTickersAsync(Contract(conId=42))) == 1
        ib.client.reqMktData.side_effect = lambda req, *args: ib.wrapper.error(
            req, 354, "No subscription", ""
        )
        with pytest.raises(RequestError):
            await ib.reqTickersAsync(Contract(conId=42))
        ib.client.reqMktData.side_effect = None
        task = asyncio.create_task(ib.reqTickersAsync(Contract(conId=42)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not ib.wrapper._futures and not ib.wrapper._results
        assert not ib.wrapper._reqId2Contract and not ib.wrapper.reqId2Ticker
        assert not ib.wrapper.ticker2ReqId["snapshot"]
        assert ib.client.cancelMktData.call_count == 2

    asyncio.run(check())
