"""Deterministic broker latency and large-ledger probes. No sockets are used."""

import asyncio
import json
import subprocess
import time
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from ib_async import Contract, ContractDetails

from stocker_execution.first4_runtime import Runtime
from test_first4 import CLOSE, OPEN, broker


class VirtualLoop(asyncio.SelectorEventLoop):
    clock = 0.0

    def time(self):
        return self.clock

    def _run_once(self):
        if not self._ready and self._scheduled:
            self.clock = max(self.clock, self._scheduled[0]._when)
        super()._run_once()


def test_history_15_second_timeout_cleans_request(tmp_path):
    from stocker_execution.first4_store import Store
    from test_first4 import armed_config

    async def run():
        runtime = Runtime(armed_config(), Store(tmp_path / "timeout.sqlite"))
        ib = runtime.broker.ib
        ib.client.getReqId = Mock(return_value=10)
        ib.client.reqHistoricalData = Mock()
        ib.client.cancelHistoricalData = Mock()
        with pytest.raises(TimeoutError):
            await runtime.history(Contract(conId=1), OPEN, "960 S")
        assert asyncio.get_running_loop().time() == 15
        ib.client.cancelHistoricalData.assert_called_once_with(10)
        assert not ib.wrapper._futures and not ib.wrapper._results
        assert not ib.wrapper._reqId2Contract

    with asyncio.Runner(loop_factory=VirtualLoop) as runner:
        runner.run(run())


@pytest.mark.parametrize("latency,success", [(1, True), (4, False)])
def test_25_symbol_burst(tmp_path, latency, success):
    from stocker_execution.first4_store import Store
    from test_first4 import armed_config

    async def run():
        runtime = Runtime(armed_config(), Store(tmp_path / "burst.sqlite"))
        ib = runtime.broker.ib
        loop = asyncio.get_running_loop()
        ib.client.getReqId = Mock(side_effect=range(1, 1000))
        pending, starts, completed, cancelled = {}, [], [], []
        peak = 0

        def request(req_id, contract, *args):
            nonlocal peak
            starts.append((req_id, loop.time(), contract.conId, args[1]))

            def finish():
                pending.pop(req_id)
                completed.append(req_id)
                ib.wrapper.historicalDataEnd(req_id, "", "")

            pending[req_id] = loop.call_later(latency, finish)
            peak = max(peak, len(pending))

        def cancel(req_id):
            cancelled.append(req_id)
            if req_id in pending:
                pending.pop(req_id).cancel()

        ib.client.reqHistoricalData = request
        ib.client.cancelHistoricalData = cancel
        runtime.scanner_rows = AsyncMock(
            return_value=[
                NS(
                    rank=i,
                    contractDetails=ContractDetails(contract=Contract(symbol=f"S{i}", conId=i + 1)),
                )
                for i in range(25)
            ]
        )
        try:
            async with asyncio.timeout(45):
                await runtime.scan("2025-07-21", OPEN, CLOSE, OPEN - timedelta(days=3), OPEN)
        except TimeoutError:
            assert not success
        else:
            assert success
        await asyncio.sleep(0)
        assert not pending and not ib.wrapper._futures and not ib.wrapper._results
        assert not ib.wrapper._reqId2Contract
        assert peak == 4
        if success:
            assert len(starts) == len(completed) == 50
            assert len({(s[2], s[3]) for s in starts}) == 50
            assert loop.time() == 13
            assert len(runtime.store.rows("events")) == 25
        else:
            assert loop.time() == 45 and len(completed) == 44
            assert len(starts) == 48 and len(cancelled) == 4
            assert not runtime.store.rows("events")
            assert runtime.store.rows("sessions")[0]["blocked"]
        print(
            json.dumps(
                dict(
                    latency=latency,
                    started=len(starts),
                    completed=len(completed),
                    cancelled=len(cancelled),
                    peak=peak,
                    seconds=loop.time(),
                    last_queue_wait=starts[-1][1],
                )
            )
        )

    with asyncio.Runner(loop_factory=VirtualLoop) as runner:
        runner.run(run())


def historical_fixture(b, count, active):
    """10,000 closed two-leg allocations, with persisted verification markers."""
    entries, fills, events = [], [], []
    for i in range(count + active):
        day = (OPEN - timedelta(days=count - i + 1)).date().isoformat()
        ref = f"F4:{day}:1:"
        held = i >= count
        payload = dict(
            put=Contract(secType="OPT", conId=i * 2 + 1, multiplier="100").dict(),
            call=Contract(secType="OPT", conId=i * 2 + 2, multiplier="100").dict(),
            quantity=1,
            exit_con_ids=[i * 2 + 1, i * 2 + 2],
            exit_quantity=1,
            exit_at=(CLOSE if held else OPEN).isoformat(),
            entry_deadline_at=OPEN.isoformat(),
            deadline_reconciled=True,
            management_resolved=not held,
        )
        events.append(
            (
                day,
                "A",
                i,
                OPEN.isoformat(),
                1,
                "SELECTED",
                5,
                1,
                OPEN.isoformat(),
                CLOSE.isoformat(),
                CLOSE.isoformat(),
                "HELD" if held else "CLOSED",
                "{}",
            )
        )
        for role, side in [("ENTRY", "BOT")] + ([] if held else [("EXIT", "SLD")]):
            entries.append((ref + role, day, "A", role, i, i, "Filled", json.dumps(payload)))
            for con_id in payload["exit_con_ids"]:
                fills.append(
                    (
                        ref + role + str(con_id),
                        ref + role,
                        con_id,
                        1,
                        1,
                        side,
                        100,
                        OPEN.isoformat(),
                        0.65,
                    )
                )
    with b.store.db:
        b.store.db.executemany(
            "INSERT INTO first4_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", events
        )
        b.store.db.executemany("INSERT INTO first4_orders VALUES (?,?,?,?,?,?,?,?)", entries)
        b.store.db.executemany("INSERT INTO first4_fills VALUES (?,?,?,?,?,?,?,?,?)", fills)


@pytest.mark.parametrize("active", [0, 1, 4])
def test_hot_loop_10000_closed_allocations(tmp_path, monkeypatch, active):
    b = broker(tmp_path)
    historical_fixture(b, 10000, active)
    monkeypatch.setattr("stocker_execution.first4_broker.now", lambda: OPEN)
    statements = []
    steps = 0

    def progress():
        nonlocal steps
        steps += 1
        return 0

    b.store.db.set_trace_callback(statements.append)
    b.store.db.set_progress_handler(progress, 1)
    b.store.rows = Mock(side_effect=AssertionError("Unbounded ledger read in hot loop"))
    writes = b.store.db.total_changes
    started = time.perf_counter()

    async def cycle():
        await b.cancel_due_entries()
        await b.close_due()
        return b.owned_quantities()

    quantities = asyncio.run(cycle())
    elapsed = time.perf_counter() - started
    b.store.db.set_progress_handler(None, 0)
    assert b.store.db.total_changes == writes
    assert len(quantities) == active * 2
    assert len(statements) <= 4 + active
    assert steps < 1500
    print(
        json.dumps(
            dict(
                active=active,
                history=10000,
                queries=len(statements),
                sqlite_steps=steps,
                writes=0,
                seconds=elapsed,
            )
        )
    )


def measure_original_hot_loop_baseline(tmp_path):
    """Measure original actual reads and dispatch, not extrapolated full-loop times."""
    source = subprocess.check_output(
        [
            "rtk",
            "proxy",
            "git",
            "show",
            "0764f95680951c069c02f3c752466105a19f3d70:packages/stocker_execution/src/stocker_execution/first4_broker.py",
        ],
        text=True,
    )
    namespace = {}
    exec(compile(source, "baseline_first4_broker.py", "exec"), namespace)
    namespace["now"] = lambda: OPEN
    store_source = subprocess.check_output(
        [
            "rtk",
            "proxy",
            "git",
            "show",
            "0764f95680951c069c02f3c752466105a19f3d70:packages/stocker_execution/src/stocker_execution/first4_store.py",
        ],
        text=True,
    )
    store_namespace = {}
    exec(compile(store_source, "baseline_first4_store.py", "exec"), store_namespace)
    for active in [0, 1, 4]:
        path = tmp_path / str(active)
        path.mkdir()
        b = broker(path)
        b.store.db.close()
        b.store = store_namespace["Store"](path / "baseline.sqlite")
        historical_fixture(b, 10000, active)
        old = namespace["PaperBroker"](b.config, b.store, b.ib)
        old.reconciled = True
        counts = {"rows": 0, "queries": 0}
        original_rows = b.store.rows

        def counted(table, original_rows=original_rows, counts=counts):
            result = original_rows(table)
            counts["rows"] += len(result)
            counts["queries"] += 1
            return result

        b.store.rows = counted
        started = time.perf_counter()
        old.owned_quantities()
        old.deadline_blocker()
        selected = []

        async def dispatched(reference, selected=selected):
            selected.append(reference)

        old.close_one = dispatched
        asyncio.run(old.close_due())
        elapsed = time.perf_counter() - started
        assert len(selected) == 10000 + active
        print(
            json.dumps(
                dict(
                    baseline=True,
                    active=active,
                    history=10000,
                    dispatched=len(selected),
                    **counts,
                    seconds=elapsed,
                )
            )
        )
        # One real settled allocation, including its repeated no-op outcome write.
        old.close_one = namespace["PaperBroker"].close_one.__get__(old)
        writes = b.store.db.total_changes
        counts.update(rows=0, queries=0)
        started = time.perf_counter()
        asyncio.run(old.close_one(selected[0]))
        print(
            json.dumps(
                dict(
                    baseline_one_close=True,
                    active=active,
                    **counts,
                    writes=b.store.db.total_changes - writes,
                    seconds=time.perf_counter() - started,
                )
            )
        )
        assert b.store.db.total_changes == writes + 1


def measure_dashboard_before_after(tmp_path):
    """Optional local comparison with the inspected commit; not a CI timing test."""
    import statistics
    from pathlib import Path

    import httpx

    from stocker_dashboard.app import create_dashboard_app
    from stocker_execution.first4_runtime import Runtime

    source = subprocess.check_output(
        [
            "rtk",
            "proxy",
            "git",
            "show",
            "0c55648d70d97804a741f0aa006632131a2e7b84:packages/stocker_dashboard/src/stocker_dashboard/app.py",
        ],
        text=True,
    )
    namespace = {
        "__file__": str(
            Path(__file__).parents[1] / "packages/stocker_dashboard/src/stocker_dashboard/app.py"
        )
    }
    exec(compile(source, "baseline_dashboard.py", "exec"), namespace)
    b = broker(tmp_path)
    historical_fixture(b, 10000, 4)
    runtime = Runtime(b.config, b.store)
    runtime.broker = b
    runtime.session = "2025-07-21"
    for name, factory, paths in [
        (
            "before",
            namespace["create_dashboard_app"],
            ["/api/overview", "/api/orders?limit=100&offset=100"],
        ),
        ("after", create_dashboard_app, ["/api/overview?view=orders&offset=100"]),
    ]:
        app = factory(runtime)
        queries = []
        b.store.db.set_trace_callback(queries.append)
        samples = []
        writes = b.store.db.total_changes

        async def run(app=app, queries=queries, paths=paths, samples=samples):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
            ) as client:
                for _ in range(5):
                    queries.clear()
                    size = 0
                    start = time.perf_counter()
                    for path in paths:
                        response = await client.get(path)
                        assert response.status_code == 200
                        size += len(response.content)
                    samples.append(time.perf_counter() - start)
                return size

        size = asyncio.run(run())
        assert b.store.db.total_changes == writes
        print(
            json.dumps(
                dict(
                    dashboard=name,
                    history=10000,
                    active=4,
                    requests=len(paths),
                    queries=len(queries),
                    payload_bytes=size,
                    writes=0,
                    median_seconds=statistics.median(samples),
                    samples=5,
                )
            )
        )


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        import sys

        if "--dashboard" in sys.argv:
            measure_dashboard_before_after(Path(directory))
        else:
            measure_original_hot_loop_baseline(Path(directory))
