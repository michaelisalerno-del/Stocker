"""Bounded offline four-stock callback burst alongside the real FIRST4 entry coroutine.

No IBKR connection. Temporary recorder/ledger. Does not arm or submit real orders.
Run: uv run --no-sync python scripts/first4_order_flow_benchmark.py
"""

import asyncio
import itertools
import json
import platform
import statistics
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from ib_async import Contract, TickAttribBidAsk, TickAttribLast

from stocker_execution.first4_config import First4Config, OrderFlowConfig
from stocker_execution.first4_flow_observer import FlowObserver
from stocker_execution.first4_flow_store import FlowWriter, replay
from stocker_execution.first4_requests import First4IB
from stocker_execution.first4_runtime import Runtime
from stocker_execution.first4_store import Store


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p99": ordered[int(len(ordered) * 0.99)],
        "max": max(ordered),
    }


async def trial(root: Path, enabled: bool, packets: int = 1000) -> dict:
    baseline = datetime.now(UTC).replace(second=0, microsecond=0) + timedelta(minutes=1)
    clock = [baseline - timedelta(seconds=1)]
    config = First4Config(
        order_flow=OrderFlowConfig(
            enabled=enabled,
            available_tbt=12,
            raw_path=root / "flow",
            min_free_bytes=1_000_000,
        )
    )
    store = Store(root / "ledger.sqlite3")
    events = store.observe(
        baseline.date().isoformat(),
        baseline - timedelta(minutes=1),
        baseline + timedelta(hours=1),
        [
            dict(symbol=f"S{i}", con_id=i, rank=i, price=11, change_pct=6, prior15=5)
            for i in range(1, 5)
        ],
    )
    runtime = Runtime(config, store)
    runtime.config = NS(missing=lambda: [], number=lambda key: 180)
    runtime.broker.ib.disconnect()  # Never connected; this only disposes constructor state.
    ib = First4IB()
    ids = itertools.count(42)
    ib.client.getReqId = Mock(side_effect=lambda: next(ids))
    ib.client.reqTickByTickData = Mock()
    ib.client.cancelTickByTickData = Mock()
    ib.isConnected = Mock(return_value=True)
    runtime.broker = NS(
        ib=ib,
        entries_armed=lambda: True,
        guard=lambda: None,
        data_generation=0,
        market_data_block={},
        upstream_available=True,
        data_problem="",
        standard_chain=AsyncMock(return_value=[]),
        enter=AsyncMock(),
    )
    runtime.session = baseline.date().isoformat()
    runtime.order_flow = FlowObserver(
        NS(config=config, broker=runtime.broker, store=store, session=runtime.session, running=True)
    )
    flow = runtime.order_flow
    contracts = [Contract(conId=i, symbol=f"S{i}", secType="STK") for i in range(1, 5)]
    loop_samples, callbacks = [], []
    monitor_running = True

    async def concurrent_manager():
        previous = time.perf_counter()
        while monitor_running:
            await asyncio.sleep(0)
            current = time.perf_counter()
            loop_samples.append((current - previous) * 1000)
            previous = current

    with (
        patch("stocker_execution.first4_runtime.now", lambda: clock[0]),
        patch("stocker_execution.first4_flow_observer.utc_now", lambda: baseline),
    ):
        entries = [
            asyncio.create_task(runtime.execute(e, c))
            for e, c in zip(events, contracts, strict=True)
        ]
        await asyncio.sleep(0)
        if enabled:
            flow.writer = FlowWriter(config.order_flow)
            flow.writer.start()
            await asyncio.to_thread(flow.writer.ready.wait, 3)
            for e, c in zip(events, contracts, strict=True):
                flow.allocate(e, c)
                await flow.start_capture(flow.captures[c.conId])
                ib.flow_wire.requested[c.conId] -= 16
                flow.captures[c.conId].quote_request = ib.flow_wire.quote_request(
                    c, config.order_flow.feed_mode, flow.captures[c.conId].emit
                )
        monitor = asyncio.create_task(concurrent_manager())
        started = time.perf_counter()
        clock[0] = baseline
        last_ids = [ib.flow_wire.lasts[c.conId].request_id for c in contracts]
        for packet in range(packets):
            ib.wrapper.tcpDataArrived()
            for c, req in zip(contracts, last_ids, strict=True):
                if enabled:
                    qreq = flow.captures[c.conId].quote_request
                    before = time.perf_counter_ns()
                    ib.wrapper.tickByTickBidAsk(
                        qreq, int(baseline.timestamp()), 10, 11, 100, 100, TickAttribBidAsk()
                    )
                    callbacks.append((time.perf_counter_ns() - before) / 1000)
                before = time.perf_counter_ns()
                # After entry cleanup the disabled fixture keeps the same synthetic
                # input decoder workload via a trade ticker with no entry subscriber.
                if req not in ib.wrapper.reqId2Ticker:
                    ib.wrapper.startTicker(req, c, "Last")
                ib.wrapper.tickByTickAllLast(
                    req,
                    1,
                    int(baseline.timestamp()),
                    11 if packet % 2 else 10,
                    100,
                    TickAttribLast(),
                    "FIXTURE",
                    "",
                )
                callbacks.append((time.perf_counter_ns() - before) / 1000)
            ib.wrapper.tcpDataProcessed()
            if packet % 10 == 0:
                await asyncio.sleep(0)
        elapsed = time.perf_counter() - started
        await asyncio.gather(*entries)
        monitor_running = False
        await monitor
        if enabled:
            for capture in flow.captures.values():
                flow.finish(capture, "BENCHMARK_COMPLETE")
            await asyncio.to_thread(flow.writer.close)
        result = {
            "enabled": enabled,
            "stock_streams": 4,
            "trade_prints": 4 * packets,
            "quote_events": 4 * packets if enabled else 0,
            "burst_seconds": elapsed,
            "callback_microseconds": distribution(callbacks),
            "execution_loop_yield_ms": distribution(loop_samples),
            "concurrent_entry_preparations_completed": runtime.broker.enter.await_count,
            "entry_anchor_prices": [call.args[2] for call in runtime.broker.enter.call_args_list],
            "queue_high_water": flow.writer.high_water if enabled else 0,
            "dropped_events": flow.writer.dropped if enabled else 0,
            "queue_latency_max_ms": flow.writer.max_queue_latency_ms if enabled else 0,
            "writer_error": flow.writer.error if enabled else "",
        }
        if enabled:
            result["raw_replay_matches"] = all(
                all(
                    c["saved_totals_match"]
                    for c in replay(root / "flow", runtime.session, i)["captures"]
                )
                for i in range(1, 5)
            )
        store.db.close()
        return result


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="first4-flow-benchmark-") as directory:
        root = Path(directory)
        reports = []
        for iteration in range(3):
            for enabled in (False, True):
                path = root / f"{iteration}-{enabled}"
                path.mkdir()
                reports.append(await trial(path, enabled))
        print(
            json.dumps(
                {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "workload": "1000 packets x four stocks; cooperative manager yield probe",
                    "scope": "OFFLINE_FAKE_BROKER_NOT_PRODUCTION_LATENCY",
                    "trials": reports,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
