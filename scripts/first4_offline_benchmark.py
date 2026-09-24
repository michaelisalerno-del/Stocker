"""Compare deterministic offline work with a local baseline commit; no broker I/O.

Run: uv run --no-sync python scripts/first4_offline_benchmark.py
Requires the dev/server test environment and the named commit locally.
"""

import asyncio
import json
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from ib_async import Contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import stocker_launcher  # noqa: E402

stocker_launcher.configure_numeric_runtime()
stocker_launcher._ensure_monorepo_src_paths()

from stocker_dashboard.app import create_dashboard_app  # noqa: E402
from stocker_data.calendars import get_market_calendar  # noqa: E402
from stocker_execution import first4_broker, first4_runtime  # noqa: E402
from stocker_execution.first4_runtime import Runtime  # noqa: E402
from test_first4 import CLOSE, OPEN, broker, prepare_exit  # noqa: E402
from test_first4_repairs import seed_closed_history  # noqa: E402

BASELINE = "505d2429cf109a733f63d296a9943b974b222dcc"


def original(package: str, filename: str) -> ModuleType:
    name = "baseline_" + filename
    module = ModuleType(name)
    module.__file__ = str(ROOT / f"packages/{package}/src/{package}/{filename}.py")
    sys.modules[name] = module
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:packages/{package}/src/{package}/{filename}.py"],
        cwd=ROOT,
        text=True,
    )
    exec(compile(source, name, "exec"), module.__dict__)
    return module


async def measure(b, runtime, app_factory):
    statements = []
    b.store.db.set_trace_callback(statements.append)
    before = b.store.db.total_changes
    await b.cancel_due_entries()
    await b.close_due()
    changes = b.store.db.total_changes - before
    b.store.db.set_trace_callback(None)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app_factory(runtime)), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/api/overview")
        response.raise_for_status()
        result = response.json()
    return {
        "management_sql_statements": len(statements),
        "management_writes": changes,
        "overview_bytes": len(response.content),
        "overview_candidates": len(result["candidates"]),
        "overview_orders": len(result["orders"]),
        "overview_fills": len(result["fills"]),
    }


async def calendar_counts(old_module, before, after):
    clock = datetime(2025, 7, 19, 12, tzinfo=UTC)
    old_module.now = first4_runtime.now = lambda: clock
    calendar = get_market_calendar("NYSE")
    schedule = calendar.schedule
    calls = 0
    iterations = 0

    def refresh(**kwargs):
        nonlocal calls
        calls += 1
        return schedule(**kwargs)

    async def sleep(delay):
        nonlocal iterations
        iterations += 1
        if iterations == 20:
            before.running = False

    calendar.schedule = refresh
    old_module.get_market_calendar = lambda _: calendar
    old_module.asyncio = SimpleNamespace(
        to_thread=asyncio.to_thread, sleep=sleep, CancelledError=asyncio.CancelledError
    )
    await before.scan_sessions()
    old_count, calls = calls, 0
    for _ in range(20):
        await after.scan_step(calendar)
    return {"iterations": 20, "before": old_count, "after": calls}


async def scanner_batch(runtime, legacy, fail):
    rows = [
        SimpleNamespace(
            rank=i,
            contractDetails=SimpleNamespace(contract=Contract(conId=i + 1, symbol=f"S{i:02}")),
        )
        for i in range(25)
    ]
    active = peak = requests = 0
    clock, previous = OPEN + timedelta(minutes=15), CLOSE - timedelta(days=3)

    async def history(contract, end, duration):
        nonlocal active, peak, requests
        requests += 1
        active += 1
        peak = max(peak, active)
        try:
            if fail and contract.conId == 1 and duration == "60 S":
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.002)
            if fail and contract.conId == 1 and duration != "60 S":
                raise ValueError("controlled history failure")
            if duration == "60 S":
                return [SimpleNamespace(date=previous - timedelta(minutes=1), close=9)]
            return [
                SimpleNamespace(
                    date=OPEN + timedelta(minutes=i), open=10, high=11, low=10, close=10
                )
                for i in range(15)
            ]
        finally:
            active -= 1

    async def old_history(contract, **kwargs):
        return await history(contract, kwargs["endDateTime"], kwargs["durationStr"])

    runtime.broker.config = runtime.config = runtime.config.model_copy(update={"armed": False})
    runtime.broker.ib.history = history
    runtime.broker.ib.reqHistoricalDataAsync = old_history
    runtime.broker.ib.scanner = AsyncMock(return_value=rows)
    runtime.broker.ib.reqScannerDataAsync = AsyncMock(return_value=rows)
    started = asyncio.get_running_loop().time()
    await runtime.scan("2025-07-21", OPEN, CLOSE, previous, clock)
    metrics = {
        "requests": requests,
        "peak_concurrency": peak,
        "work_left_at_return": active,
        "batch_seconds": round(asyncio.get_running_loop().time() - started, 4),
    }
    if not legacy:
        metrics.update(runtime.scan_metrics)
    await asyncio.gather(*runtime.tasks)
    await asyncio.sleep(0.11)  # Drain the original implementation's orphaned sibling.
    return metrics


def main():
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline benchmark forbids every network connection")

    socket.socket.connect = forbidden
    old_store = original("stocker_execution", "first4_store")
    old_broker = original("stocker_execution", "first4_broker")
    old_runtime = original("stocker_execution", "first4_runtime")
    old_app = original("stocker_dashboard", "app")
    clock = OPEN + timedelta(hours=1)
    first4_broker.now = lambda: clock
    old_broker.now = lambda: clock
    with tempfile.TemporaryDirectory(prefix="first4-offline-benchmark-") as directory:
        root = Path(directory)
        b = broker(root)
        event, payload = prepare_exit(b)
        payload["deadline_reconciled"] = True
        b.store.db.execute("UPDATE first4_orders SET payload=?", (json.dumps(payload),))
        seed_closed_history(b.store, 1000)
        b.store.db.commit()
        baseline_store = old_store.Store(root / "s.sqlite")
        baseline_broker = old_broker.PaperBroker(b.config, baseline_store, b.ib)
        baseline_broker.reconciled, baseline_broker.entry_blocker = True, ""
        baseline_runtime = old_runtime.Runtime(b.config, baseline_store)
        baseline_runtime.broker, baseline_runtime.session = baseline_broker, event["session"]
        before = asyncio.run(
            measure(baseline_broker, baseline_runtime, old_app.create_dashboard_app)
        )
        baseline_store.db.close()
        runtime = Runtime(b.config, b.store)
        runtime.broker, runtime.session = b, event["session"]
        after = asyncio.run(measure(b, runtime, create_dashboard_app))
        calendar = asyncio.run(calendar_counts(old_runtime, baseline_runtime, runtime))
        # Restore the real asyncio module after the bounded calendar experiment.
        old_runtime.asyncio = asyncio
        scanner = {}
        for fail in (False, True):
            pair = {}
            for legacy in (True, False):
                local = root / f"scan-{legacy}-{fail}"
                local.mkdir()
                scan_broker = broker(local)
                cls = old_runtime.Runtime if legacy else Runtime
                scan_runtime = cls(scan_broker.config, scan_broker.store)
                scan_runtime.broker = scan_broker
                pair["before" if legacy else "after"] = asyncio.run(
                    scanner_batch(scan_runtime, legacy, fail)
                )
                scan_broker.store.db.close()
            scanner["partial_failure" if fail else "success"] = pair
        b.store.db.close()
    print(
        json.dumps(
            {
                "baseline_commit": BASELINE,
                "closed_allocations": 1000,
                "active_allocations": 1,
                "before": before,
                "after": after,
                "calendar_refreshes": calendar,
                "scanner": scanner,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
