"""Offline wire/callback fixtures. No broker, entitlements or transmission required."""

import asyncio
import gzip
import json
import time
import tracemalloc
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest
from ib_async import Contract
from ib_async.objects import BarData

from stocker_execution.config import MARKETS, L2Config, MarketDataConfig
from stocker_execution.depth import Book, DepthObserver, coverage
from stocker_execution.pacing import CORE, EXPOSURE, OPTIONAL, URGENT, drain, priority
from stocker_execution.requests import BrokerConnection
from stocker_execution.rules import opportunity
from stocker_execution.store import Store
from stocker_execution.subscriptions import Subscriptions
from test_futures import AT, plan, setup


def contract(cid=1, kind="FUT", market="GC"):
    return Contract(conId=cid, secType=kind, symbol=market, exchange="COMEX", currency="USD")


def fixture(**changes):
    limits = MarketDataConfig(cancel_drain_seconds=0.1, **changes)
    ib = BrokerConnection(limits)
    ib.isConnected = lambda: True
    ib.client.getReqId = lambda: next(ids)
    ids = iter(range(1, 10000))
    requests, active = [], set()

    def start(kind, rid, *args):
        requests.append((kind, rid))
        active.add(rid)

    def stop(kind, rid, *args):
        requests.append((kind, rid))
        active.discard(rid)

    def bars(rid, c, end, duration, size, what, rth, dates, keep, options):
        assert size == "1 min" and what == "TRADES" and not rth and dates == 2 and keep
        start("bars", rid)
        ib.wrapper._results[rid].append(BarData(date=AT, close=100, volume=73))
        ib.wrapper.historicalDataEnd(rid, "", "")

    ib.client.reqMktData = lambda rid, *args: start("quote", rid)
    ib.client.cancelMktData = lambda rid: stop("cancel_quote", rid)
    ib.client.reqMktDepth = lambda rid, *args: start("depth", rid)
    ib.client.cancelMktDepth = lambda rid, smart: stop("cancel_depth", rid)
    ib.client.reqHistoricalData = bars
    ib.client.cancelHistoricalData = lambda rid: stop("cancel_bars", rid)
    ib.reqMktDepthExchangesAsync = routes
    return ib, Subscriptions(ib, limits), requests, active


async def routes():
    return [NS(exchange="COMEX", secType="FUT", serviceDataType="Deep")]


def row(generation=1, side=0, position=0, operation=0, price=101, size=2):
    return dict(
        kind="ROW",
        generation=generation,
        side=side,
        position=position,
        operation=operation,
        price=price,
        size=size,
    )


def test_counts_six_markets_four_exposures_fifteen_options_and_three_books():
    async def scenario():
        ib, data, wire, active = fixture()
        for i, market in enumerate(MARKETS, 1):
            c = contract(i, market=market)
            await data.acquire(c, "QUOTE", f"core:{market}", "core_quote")
            await data.acquire(c, "BARS", f"core:{market}", "core_bars", {"duration": "2 D"})
        await data.acquire(contract(90, "CASH"), "QUOTE", "fx", "fx", level=EXPOSURE)
        assert data.count() == 13
        for i in range(4):
            await data.acquire(
                contract(100 + i, "FOP"), "QUOTE", f"exposure:{i}", "exposure", level=EXPOSURE
            )
            await data.acquire(
                contract(200 + i), "QUOTE", f"old-future:{i}", "exposure", level=EXPOSURE
            )
        for i in range(3):
            await data.acquire(
                contract(i + 1), "DEPTH", f"depth:{i}", "depth", {"levels": 5}, OPTIONAL
            )
        deadline = asyncio.get_running_loop().time() + 10
        from contextlib import AsyncExitStack

        async with AsyncExitStack() as stack:
            for batch in range(3):
                await stack.enter_async_context(
                    data.option_batch(
                        [contract(300 + batch * 5 + i, "FOP") for i in range(5)],
                        f"batch:{batch}",
                        deadline,
                    )
                )
            assert data.count() == 39 and data.temporary_count() == 15
            with pytest.raises(ValueError, match="OPTION_QUOTE_POOL_EXHAUSTED"):
                await data.acquire(contract(400, "FOP"), "QUOTE", "extra", "selection")
            with pytest.raises(ValueError, match="L2_NOT_CAPTURED_CAPACITY"):
                await data.acquire(contract(5), "DEPTH", "fourth", "depth", {"levels": 5}, OPTIONAL)
            assert len(active) == 39
            handoff = await data.acquire(
                contract(500, "FOP"), "QUOTE", "admission", "exposure", level=EXPOSURE
            )
            assert data.count() == 40  # One serial selected-quote handoff before atomic admission.
            await data.release(handoff, "admission")
        assert data.count() == 24 and data.temporary_count() == 0
        await data.close()
        assert not data.items and not active and not ib.wrapper.reqId2Ticker
        assert not ib.wrapper.reqId2Subscriber

    asyncio.run(scenario())


def test_reference_counting_cancel_debt_and_disconnected_generations():
    async def scenario():
        ib, data, wire, active = fixture()
        c = contract()
        one, two = await asyncio.gather(
            data.acquire(c, "QUOTE", "core", "core_quote"),
            data.acquire(c, "QUOTE", "position", "exposure"),
        )
        assert one is two and len(wire) == 1
        await data.release(one, "core")
        assert data.count() == 1 and len(wire) == 1
        task = asyncio.create_task(data.release(two, "position"))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert data.count() == 1 and two.state == "CANCELLING"
        with pytest.raises(ValueError, match="CANCELLATION_PENDING"):
            await data.acquire(c, "QUOTE", "new", "core_quote")
        await asyncio.sleep(0.12)
        assert data.count() == 0
        new = await data.acquire(c, "QUOTE", "core", "core_quote")
        assert new.generation > one.generation
        ib.disconnectedEvent.emit()
        assert not data.items
        data.quote_update(new, new.value)
        assert new.state == "DISCONNECTED"
        await data.close()

    asyncio.run(scenario())


def test_reduced_external_allowance_and_broker_capacity_backoff():
    async def scenario():
        ib, data, wire, active = fixture(known_external_lines=80)
        assert data.effective_cap == 20
        for i in range(12):
            await data.acquire(contract(i + 1), "QUOTE", f"core:{i}", "core_quote")
        with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
            await data.acquire(contract(80, "FOP"), "QUOTE", "candidate", "selection")
        exposure = await data.acquire(
            contract(90, "FOP"), "QUOTE", "position", "exposure", level=EXPOSURE
        )
        data.error(-1, 101, "Other client consumed remaining lines")
        await asyncio.sleep(0)
        assert exposure in data.items.values()
        assert data.effective_cap == 12
        with pytest.raises(ValueError, match="BACKOFF"):
            await data.acquire(contract(81), "QUOTE", "retry", "core_quote")
        assert data.snapshot()["external_usage"] == 80
        await data.close()

    asyncio.run(scenario())


def test_one_pacer_reserves_urgent_capacity_and_bounds_optional_queue():
    async def scenario():
        ib = BrokerConnection()
        sent = []
        ib.client.conn.sendMsg = lambda message: sent.append(message[4:].decode())
        with priority(CORE) as core:
            for i in range(30):
                ib.client.sendMsg(f"core:{i}")
            await drain(core)
        with priority(OPTIONAL, asyncio.get_running_loop().time() + 0.03) as optional:
            for i in range(64):
                ib.client.sendMsg(f"optional:{i}")
            with pytest.raises(ValueError, match="API_QUEUE_FULL"):
                for _ in range(20):
                    ib.client.sendMsg("overflow")
            with priority(URGENT) as urgent:
                for i in range(10):
                    ib.client.sendMsg(f"cancel:{i}")
                await drain(urgent)
            assert len(sent) == 40 and sent[-1] == "cancel:9"
            with pytest.raises(TimeoutError, match="PACING_DEADLINE"):
                await drain(optional)
        assert not any(s.startswith("optional") for s in sent)
        ib.client.reset()
        assert not ib.client.pending

    asyncio.run(scenario())


def test_bar_requests_deduplicated_cached_separate_and_volume_unchanged():
    async def scenario():
        ib, data, wire, active = fixture()
        count = 0

        async def history():
            nonlocal count
            count += 1
            await asyncio.sleep(0.01)
            return [BarData(date=AT, volume=53)]

        result = await asyncio.gather(*(data.historical((1, "1 min"), history) for _ in range(10)))
        assert count == 1 and all(r[0].volume == 53 for r in result)
        assert (await data.historical((1, "1 min"), history))[0].volume == 53 and count == 1
        sub = await data.acquire(contract(), "BARS", "monitor", "core_bars", {"duration": "2 D"})
        assert sub.value[0].volume == 73 and data.count() == 1
        await data.close()

    asyncio.run(scenario())


def test_raw_depth_insert_update_delete_reset_and_stale_book():
    async def scenario():
        ib, data, wire, active = fixture()
        book = Book("GC", contract(), 1, L2Config(levels=2))

        def receive(event):
            book.receive(event, AT)

        sub = await data.acquire(
            contract(), "DEPTH", "depth:GC", "depth", {"levels": 2}, OPTIONAL, sink=receive
        )
        assert sub.generation == 1
        ib.wrapper.updateMktDepth(sub.request_id, 0, 0, 0, 102, 5)
        ib.wrapper.updateMktDepthL2(sub.request_id, 0, "COMEX", 0, 0, 101, 2, False)
        ib.wrapper.updateMktDepth(sub.request_id, 0, 0, 1, 99, 3)
        assert not book.valid
        ib.wrapper.updateMktDepth(sub.request_id, 1, 0, 1, 98, 4)
        assert book.valid and [r["price"] for r in book.sides[0]] == [101, 102]
        ib.wrapper.updateMktDepth(sub.request_id, 0, 1, 1, 100, 7)
        assert book.sides[1][0]["price"] == 100
        ib.wrapper.updateMktDepth(sub.request_id, 0, 2, 0, 0, 0)
        assert [r["price"] for r in book.sides[0]] == [102] and not book.valid
        ib.wrapper.error(sub.request_id, 317, "Market depth reset")
        assert book.sides == [[], []] and book.resets == 1
        ib.wrapper.updateMktDepth(sub.request_id, 4, 1, 0, 110, 2)
        assert book.gaps == 1
        assert book.view(AT + timedelta(seconds=31))["asks"] == []
        await data.release(sub, "depth:GC")
        before = book.sequence
        ib.wrapper.updateMktDepth(sub.request_id, 0, 0, 0, 101, 4)
        assert book.sequence == before
        await data.close()

    asyncio.run(scenario())


def test_three_slot_rotation_signal_preemption_and_coverage(tmp_path, monkeypatch):
    async def scenario():
        clock = [AT - timedelta(seconds=180)]
        monkeypatch.setattr("stocker_execution.depth.stamp", lambda: clock[0])
        ib, data, wire, active = fixture()
        store = Store(tmp_path / "capture.sqlite")
        observer = DepthObserver(data, store, L2Config(enabled=True), tmp_path / "l2")
        await observer.start()
        candidates = {
            m: NS(contract=contract(i, market=m), validExchanges="COMEX")
            for i, m in enumerate(MARKETS, 1)
        }
        next_times = dict.fromkeys(MARKETS, AT)
        await observer.tick(candidates, next_times, True)
        assert list(observer.feeds) == ["BTC", "CL", "GC"]
        old_sub = observer.feeds["GC"]
        clock[0] += timedelta(seconds=30)
        await observer.tick(candidates, next_times, True)
        assert list(observer.feeds) == ["BTC", "CL", "GC"]  # minimum dwell
        clock[0] = AT
        event = opportunity("SI", 6, AT)
        store.observe(event, "SKIP_BUDGET_TOO_SMALL", {})
        observer.signal(event)
        await observer.tick(candidates, next_times, True)
        assert "SI" in observer.feeds and len(observer.feeds) == 3
        sub = observer.feeds["SI"]
        for second in (1, 11, 21, 31, 41):
            clock[0] = AT + timedelta(seconds=second)
            ib.wrapper.updateMktDepth(sub.request_id, 0, 0 if second == 1 else 1, 0, 100, 3)
        assert observer.view("SI")["pre_seconds"] == 0
        assert observer.view("SI")["post_seconds"] == 40
        assert observer.view("SI")["complete_book_pre_seconds"] == 0
        clock[0] = AT + timedelta(seconds=121)
        await observer.tick(candidates, next_times, True)
        await observer.queue.join()
        summary = store.depth_summary(event["id"])
        assert summary["pre_seconds"] == 0 and summary["post_seconds"] == 40
        assert summary["allocated"] and summary["reason"] == "MISSING_PRE_TRIGGER_CONTEXT"
        artifact = json.loads(gzip.decompress((tmp_path / "l2" / summary["artifact"]).read_bytes()))
        assert not artifact["pre_trigger_events"] and artifact["post_trigger_events"]
        assert store.history(None, None, None, 0)[0]["reason"] == "SKIP_BUDGET_TOO_SMALL"
        # Reassignment and old callback cannot write into the new generation.
        before = observer.books.get("GC")
        ib.wrapper.updateMktDepth(old_sub.request_id, 0, 0, 0, 888, 9)
        assert observer.books.get("GC") is before
        count = maximum = 0
        for kind, _ in wire:
            count += (kind == "depth") - (kind == "cancel_depth")
            maximum = max(maximum, count)
        assert maximum == 3
        await observer.close()
        await data.close()

    asyncio.run(scenario())


def test_partial_pre_context_never_backfilled():
    book = Book("GC", contract(), 1, L2Config())
    for i in range(43):
        book.receive(row(operation=0 if i == 0 else 1), AT - timedelta(seconds=42 - i))
    c = coverage(list(book.events), AT, AT)
    assert c["pre_seconds"] == 42 and c["post_seconds"] == 0
    book.receive(dict(kind="RESET", generation=1), AT + timedelta(seconds=1))
    book.receive(row(), AT + timedelta(seconds=90))
    assert coverage(list(book.events), AT, AT + timedelta(seconds=120))["post_seconds"] == 0


def test_recording_memory_disk_and_denominator_bounds(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr("stocker_execution.depth.stamp", lambda: AT)
        ib, data, _, _ = fixture()
        store = Store(tmp_path / "bounded.sqlite")
        observer = DepthObserver(
            data,
            store,
            L2Config(enabled=True, memory_bytes=262144, disk_bytes=65536),
            tmp_path / "bounded",
        )
        await observer.start()
        observer.books["GC"] = Book("GC", contract(), 1, observer.config)
        event = opportunity("GC", 1, AT)
        store.observe(event, "SKIP_CAPACITY_FULL", {})
        observer.signal(event)
        tracemalloc.start()
        sql = []
        store.db.set_trace_callback(sql.append)
        start = time.perf_counter()
        for i in range(100000):
            observer.receive("GC", 1, row(operation=0 if i == 0 else 1))
        assert sql == []  # Depth callbacks never perform synchronous ledger writes.
        store.db.set_trace_callback(None)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert observer.paused == "RECORDING_MEMORY_LIMIT"
        assert observer.memory_used() <= observer.config.memory_bytes
        assert peak < 1048576
        await observer.tick({}, {}, False)
        await observer.queue.join()
        assert store.depth_summary(event["id"])["reason"] == "RECORDING_MEMORY_LIMIT"
        observer.disk_used = 65535
        with pytest.raises(ValueError, match="DISK_LIMIT"):
            observer.write(dict(kind="POLICY", action="test"))
        print(
            json.dumps(
                {
                    "fixture_updates": 100000,
                    "elapsed_seconds": round(elapsed, 3),
                    "traced_peak_bytes": peak,
                    "accounted_peak_bytes": observer.high_water,
                    "shed_reason": observer.paused,
                }
            )
        )
        await observer.close()
        await data.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["disabled", "enabled", "failed"])
def test_l2_cannot_change_frozen_order_path(tmp_path, monkeypatch, mode):
    store, broker, ib = setup(tmp_path, monkeypatch)
    observer = DepthObserver(
        broker.data, store, L2Config(enabled=mode != "disabled"), tmp_path / "observations"
    )
    if mode == "failed":
        observer.paused = "IBKR_354:NO_DEPTH_PERMISSION"
    event = opportunity("GC", 1, AT)
    store.observe(event, "", {})
    observer.signal(event)
    # No depth await or input exists on the broker's entry path.
    assert asyncio.run(broker.enter(event, plan())) == ""
    order = ib.trades[0].order
    assert (order.action, order.totalQuantity, order.lmtPrice, order.account) == (
        "BUY",
        1,
        0.1,
        "DUP655399",
    )
    assert store.capacity() == {"reserved_open_trades": 1, "allocation_pennies": 1000}
    assert any("exposure" in sub.consumers.values() for sub in broker.data.items.values())
    assert store.depth_summary(event["id"])["requested"] == (mode != "disabled")


def test_unconfirmed_submission_preserves_quote_ownership(tmp_path, monkeypatch):
    store, broker, ib = setup(tmp_path, monkeypatch)
    ib.raise_after_send = True
    event = opportunity("GC", 1, AT)
    store.observe(event, "", {})
    with pytest.raises(TimeoutError):
        asyncio.run(broker.enter(event, plan()))
    assert broker.data.count() == 1 and store.capacity()["reserved_open_trades"] == 1


def test_reconnect_during_cancel_does_not_remove_reused_request_id():
    async def scenario():
        ib, data, wire, _ = fixture()
        old = await data.acquire(contract(), "QUOTE", "old", "core_quote")
        release = asyncio.create_task(data.release(old, "old"))
        await asyncio.sleep(0.01)
        data.disconnected()
        ib.client.getReqId = lambda: old.request_id
        new = await data.acquire(contract(), "QUOTE", "new", "exposure", level=EXPOSURE)
        await release  # Dependency cancellation must not cancel its manager caller.
        await asyncio.sleep(0.12)
        assert data.by_request[new.request_id] is new
        assert ib.wrapper.reqId2Ticker[new.request_id] is new.value
        data._remove(old)  # An already dispatched old cleanup is harmless too.
        assert data.by_request[new.request_id] is new
        await data.close()

    asyncio.run(scenario())


def test_existing_exposure_never_waits_for_optional_teardown_at_full_cap():
    async def scenario():
        ib, data, wire, _ = fixture(known_external_lines=80)
        owned = await data.acquire(
            contract(100, "FOP"), "QUOTE", "position", "exposure", level=EXPOSURE
        )
        # Exactly full after a reduced externally observed allowance.
        optional = await data.acquire(
            contract(2), "DEPTH", "depth", "depth", {"levels": 5}, OPTIONAL
        )
        data.effective_cap = data.count()
        before = list(wire)
        same = await data.acquire(contract(100, "FOP"), "QUOTE", "exit", "exposure", level=EXPOSURE)
        assert same is owned and wire == before and optional in data.items.values()
        await data.close()

    asyncio.run(scenario())


def test_missing_history_response_and_disconnect_release_slots():
    async def scenario():
        ib, data, wire, _ = fixture()
        entered = asyncio.Event()

        async def absent():
            entered.set()
            await asyncio.Future()

        pending = asyncio.create_task(data.historical((1, "schedule"), absent))
        await entered.wait()
        data.disconnected()
        with pytest.raises(ValueError, match="HISTORICAL_DATA_DISCONNECTED"):
            await pending
        assert data.history_active == 0

        async def received():
            return [1]

        assert await data.historical((2, "schedule"), received) == [1]
        # Real installed wrapper cleanup on a request-local deadline, with no callback.
        ib.client.reqHistoricalData = lambda *args: None
        with priority(CORE, asyncio.get_running_loop().time() + 0.01), pytest.raises(TimeoutError):
            await ib.reqHistoricalScheduleAsync(contract(), 14)
        assert not ib.wrapper._futures and not ib.wrapper._results
        assert wire[-1][0] == "cancel_bars"
        await data.close()

    asyncio.run(scenario())


def test_optional_shed_invalidates_depth_without_affecting_core(tmp_path, monkeypatch):
    async def scenario():
        monkeypatch.setattr("stocker_execution.depth.stamp", lambda: AT)
        ib, data, _, _ = fixture()
        observer = DepthObserver(
            data, Store(tmp_path / "optional.sqlite"), L2Config(enabled=True), tmp_path / "l2"
        )
        await observer.start()
        candidates = {"GC": NS(contract=contract(), validExchanges="COMEX")}
        await observer.tick(candidates, {"GC": AT}, True)
        old = observer.feeds["GC"]
        await data.shed_optional()
        assert observer.books["GC"].gaps > 0
        await observer.tick(candidates, {"GC": AT}, True)
        assert observer.feeds["GC"].generation > old.generation
        data.disconnected()
        await observer.tick({}, {}, False)
        assert not observer.books and not observer.feeds
        assert observer.view("GC").get("asks", []) == []
        await observer.tick({}, {}, True, {"GC"})
        assert observer.view("GC")["status"] == "MARKET_CLOSED"
        assert observer.view("BTC")["status"] == "UNAVAILABLE"
        await observer.close()
        await data.close()

    asyncio.run(scenario())


def test_depth_capacity_error_does_not_block_option_entry_data():
    async def scenario():
        ib, data, _, _ = fixture()
        sub = await data.acquire(contract(), "DEPTH", "depth", "depth", {"levels": 5}, OPTIONAL)
        data.error(sub.request_id, 309, "Maximum market depth requests")
        quote = await data.acquire(contract(100, "FOP"), "QUOTE", "entry", "selection")
        assert quote.error == "" and data.capacity_cooldown == 0
        with pytest.raises(ValueError):
            await data.acquire(contract(), "DEPTH", "retry", "depth", {"levels": 5}, OPTIONAL)
        await data.close()

    asyncio.run(scenario())


def test_disconnect_during_bar_start_does_not_cancel_monitor():
    async def scenario():
        ib, data, _, _ = fixture()
        entered = asyncio.Event()
        ib.client.reqHistoricalData = lambda *args: entered.set()
        monitor = asyncio.create_task(
            data.acquire(contract(), "BARS", "core", "core_bars", {"duration": "2 D"})
        )
        await entered.wait()
        data.disconnected()
        with pytest.raises(ValueError, match="SUBSCRIPTION_DISCONNECTED"):
            await monitor
        assert not data.items and not ib.wrapper._futures

    asyncio.run(scenario())


@pytest.mark.parametrize("wire_id", ["", "0", "-1", "1.5", "１２"])
def test_option_chain_invalid_underlying_identity_fails_request(wire_id):
    async def scenario():
        ib = BrokerConnection()
        pending = ib.wrapper.startReq(123)
        ib.client.decoder.interpret(
            ["75", "123", "CME", wire_id, "BTC", "5", "1", "20261030", "1", "85000"]
        )
        ib.client.decoder.interpret(["76", "123"])
        with pytest.raises(ValueError, match="OPTION_CHAIN_UNDERLYING_ID_INVALID"):
            await pending
        # Late callbacks cannot revive a failed request or leak an abandoned chain.
        ib.client.decoder.interpret(
            ["75", "123", "CME", "876880607", "BTC", "5", "1", "20261030", "1", "85000"]
        )
        assert not ib.wrapper._futures and not ib.wrapper._results

    asyncio.run(scenario())


def test_option_chain_normalizes_id_without_changing_identity_or_listing():
    async def scenario():
        ib = BrokerConnection()
        pending = ib.wrapper.startReq(123)
        for cid in ["876880607", "876880608"]:
            ib.client.decoder.interpret(
                ["75", "123", "CME", cid, "BTC", "5", "1", "20261030", "1", "85000"]
            )
        ib.client.decoder.interpret(["76", "123"])
        chains = await pending
        assert [c.underlyingConId for c in chains] == [876880607, 876880608]
        assert all(type(c.underlyingConId) is int for c in chains)
        assert len([c for c in chains if c.underlyingConId == 876880607]) == 1
        assert chains[0].expirations == ["20261030"] and chains[0].strikes == [85000.0]
        assert chains[0].tradingClass == "BTC" and chains[0].multiplier == "5"

    asyncio.run(scenario())
