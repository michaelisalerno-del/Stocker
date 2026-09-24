"""Offline synthetic dashboard measurement. Never starts Runtime or broker I/O."""

import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import httpx

from stocker_dashboard.app import create_dashboard_app
from stocker_execution.first4_config import First4Config
from stocker_execution.first4_runtime import Runtime
from stocker_execution.first4_store import Store


async def measure(history):
    with tempfile.TemporaryDirectory() as directory:
        store = Store(Path(directory) / "bench.sqlite")
        for day in range(history + 1):
            session = (
                "2026-09-24"
                if day == 0
                else f"2025-{1 + (day - 1) // 28:02}-{1 + (day - 1) % 28:02}"
            )
            store.db.execute("INSERT INTO first4_sessions VALUES (?,NULL,NULL)", (session,))
            for i in range(500):
                selected = i < 4
                store.db.execute(
                    "INSERT INTO first4_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        session,
                        f"S{i:04}",
                        i,
                        session + "T14:00:00+00:00",
                        i % 25 + 1,
                        "SELECTED" if selected else "NOT_Q5",
                        5 if selected else 4.1,
                        i + 1 if selected else None,
                        session + "T14:01:00+00:00",
                        None,
                        session + "T20:00:00+00:00",
                        "ORDER_SUBMITTED" if selected else "REJECTED",
                        json.dumps({"source": "SYNTHETIC", "diagnostic": "x" * 2048}),
                    ),
                )
                if selected:
                    ref = f"F4:{session}:{i + 1}:ENTRY"
                    payload = {
                        "quantity": 1,
                        "put": {"conId": i * 2 + 1},
                        "call": {"conId": i * 2 + 2},
                        "quotes": [{"ask": 1, "bid": 0.9}],
                        "diagnostic": "x" * 2048,
                    }
                    store.db.execute(
                        "INSERT INTO first4_orders VALUES (?,?,?,?,?,NULL,?,?,?)",
                        (
                            ref,
                            session,
                            f"S{i:04}",
                            "ENTRY",
                            i,
                            "Filled",
                            json.dumps(payload),
                            0 if day == 0 else 1,
                        ),
                    )
                    for leg in (1, 2):
                        store.db.execute(
                            "INSERT INTO first4_fills VALUES (?,?,?,?,?,?,?,?,?,0)",
                            (
                                ref + str(leg),
                                ref,
                                i * 2 + leg,
                                1,
                                1,
                                "BOT",
                                100,
                                session + "T14:01:01+00:00",
                                0.65,
                            ),
                        )
        store.db.execute("UPDATE first4_orders SET obligation_done=1 WHERE session<>'2026-09-24'")
        store.db.commit()
        runtime = Runtime(First4Config(), store)
        runtime.session = "2026-09-24"
        app = create_dashboard_app(runtime)
        result = {}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            paths = [
                "/api/overview",
                "/api/overview?view=candidates",
                "/api/overview?view=orders",
                "/api/system",
            ]
            if "--after" in sys.argv:
                paths = ["/api/overview", "/api/opportunities", "/api/execution", "/api/system"]
            for url in paths:
                latency = []
                for _ in range(31):
                    start = time.perf_counter()
                    response = await client.get(url)
                    assert response.status_code == 200, response.text
                    latency.append((time.perf_counter() - start) * 1000)
                statements = []
                steps = 0

                def progress():
                    nonlocal steps
                    steps += 1
                    return 0

                store.db.set_trace_callback(statements.append)
                store.db.set_progress_handler(progress, 1)
                original_loads = json.loads
                parses = 0

                def counted_loads(*args, _loads=original_loads, **kwargs):
                    nonlocal parses
                    parses += 1
                    return _loads(*args, **kwargs)

                json.loads = counted_loads
                try:
                    response = await client.get(url)
                finally:
                    json.loads = original_loads
                store.db.set_trace_callback(None)
                store.db.set_progress_handler(None, 0)
                result[url] = {
                    "bytes": len(response.content),
                    "median_ms": round(statistics.median(latency), 3),
                    "p95_ms": round(sorted(latency)[29], 3),
                    "sql_statements": len(statements),
                    "sqlite_vm_steps": steps,
                    "python_json_parses": parses,
                }
        store.db.close()
        return result


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "small_500_events": asyncio.run(measure(0)),
                "grown_50500_events": asyncio.run(measure(100)),
            },
            indent=2,
        )
    )
