"""Offline API workload for five concurrent dashboard clients; no Saxo requests."""

import argparse
import asyncio
import json
import resource
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from stocker_dashboard.app import create_dashboard_app
from stocker_execution.config import MARKETS, FuturesConfig
from stocker_execution.contracts import key
from stocker_execution.rules import Bar
from stocker_execution.runtime import Runtime
from stocker_execution.store import Store


async def benchmark(legacy: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="slrno-dashboard-fixture-") as folder:
        runtime = Runtime(FuturesConfig(), Store(Path(folder) / "ledger.sqlite"))
        at = datetime.now(UTC)
        for i, state in enumerate(runtime.markets.values()):
            state.identity = {
                "provider": "SAXO",
                "environment": "SAXO_SIM",
                "asset_type": "ContractFutures",
                "market": state.market,
                "uic": 100 + i,
                "symbol": state.market + " fixture",
                "expiry": "2027-12-01",
                "tick_size": 0.01,
            }
            state.problem = "REFERENCE_SESSION_AUDIT_OR_SAXO_COVERAGE_UNVERIFIED"
            state.bars = [
                Bar(
                    at - timedelta(minutes=180 - j),
                    70 + j * 0.01,
                    71 + j * 0.01,
                    69 + j * 0.01,
                    70 + j * 0.01,
                    10,
                )
                for j in range(180)
            ]
            payload = {
                "Quote": {
                    "Bid": 70,
                    "Ask": 70.01,
                    "BidSize": 2,
                    "AskSize": 3,
                    "DelayedByMinutes": 0,
                    "PriceTypeBid": "Tradable",
                    "PriceTypeAsk": "Tradable",
                },
                "MarketDepth": {
                    "Bid": [70 - j * 0.01 for j in range(10)],
                    "Ask": [70.01 + j * 0.01 for j in range(10)],
                    "BidSize": [2] * 10,
                    "AskSize": [3] * 10,
                    "NoOfBids": 10,
                    "NoOfOffers": 10,
                },
            }
            state.price.snapshot(payload, "fixture", at.timestamp())
            runtime.recorder.register(key(state.identity), state.identity)
            runtime.recorder.ingest(key(state.identity), "SNAPSHOT", payload, at.timestamp())
        # Closed ledger economics exercise the display cache; no broker activity.
        for i in range(400):
            identity = f"fixture-{i}"
            event = {
                "id": identity,
                "market": MARKETS[i % len(MARKETS)],
                "rule_version": "fixture",
                "signal_at": (at - timedelta(hours=i)).isoformat(),
                "exit_at": at.isoformat(),
            }
            runtime.store.observe(event, "", {})
            plan = {
                "quantity": 1,
                "cash_pennies": 800,
                "currency": "USD",
                "multiplier": 100,
                "price_unit_factor": 1,
            }
            runtime.store.reserve(identity, plan)
            with runtime.store.db:
                for n, (role, side, price) in enumerate(
                    (("ENTRY", "BOT", 0.1), ("EXIT", "SLD", 0.2))
                ):
                    reference = f"{identity}-{role}"
                    runtime.store.db.execute(
                        "INSERT INTO orders(reference,event_id,role,order_id,"
                        "status,deadline,payload) "
                        "VALUES(?,?,?,?,'Filled',?,'{}')",
                        (reference, identity, role, i * 2 + n, at.isoformat()),
                    )
                    runtime.store.db.execute(
                        "INSERT INTO fills(exec_id,reference,con_id,quantity,price,"
                        "side,at,commission,"
                        "commission_currency,fx) VALUES(?,?,1,1,?,?,?,.1,'GBP',.8)",
                        (reference, reference, price, side, at.isoformat()),
                    )
                runtime.store.db.execute(
                    "UPDATE reservations SET active=0,state='CLOSED' WHERE id=?", (identity,)
                )
        app = create_dashboard_app(runtime)
        routes = {
            "overview": ["/api/overview"],
            "markets": ["/api/market/CL"],
            "opportunities": ["/api/history?sort=asc"],
            "execution": ["/api/execution"],
            "system": ["/api/system"],
        }
        if legacy:
            routes = {
                "overview": ["/api/overview"],
                "markets": ["/api/overview"],
                "opportunities": ["/api/overview", "/api/history"],
                "execution": ["/api/overview", "/api/history"],
                "system": ["/api/overview", "/api/system", "/api/recordings"],
            }
        measurements = {name: [] for name in routes}
        sizes = {name: [] for name in routes}
        delays = []
        stop = False

        async def heartbeat():
            while not stop:
                before = time.perf_counter()
                await asyncio.sleep(0.001)
                delays.append(max(0, time.perf_counter() - before - 0.001))

        async def consumer(name, paths):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
            ) as client:
                for _ in range(40):
                    before = time.perf_counter()
                    size = 0
                    for route in paths:
                        response = await client.get(route)
                        response.raise_for_status()
                        size += len(response.content)
                    measurements[name].append((time.perf_counter() - before) * 1000)
                    sizes[name].append(size)
                    await asyncio.sleep(0)

        heartbeat_task = asyncio.create_task(heartbeat())
        start = time.perf_counter()
        cpu = time.process_time()
        await asyncio.gather(*(consumer(name, paths) for name, paths in routes.items()))
        elapsed = time.perf_counter() - start
        cpu = time.process_time() - cpu
        stop = True
        await heartbeat_task
        result = {
            "evidence": "OFFLINE_GENERATED_FIVE_API_CLIENTS",
            "legacy": legacy,
            "clients": 5,
            "refreshes_per_client": 40,
            "closed_trades": 400,
            "wall_seconds": elapsed,
            "cpu_seconds": cpu,
            "event_loop_max_delay_ms": max(delays) * 1000,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "broker_requests": runtime.data.client.calls,
            "pages": {
                name: {
                    "p50_ms": statistics.median(values),
                    "p95_ms": sorted(values)[int(0.95 * (len(values) - 1))],
                    "response_bytes": max(sizes[name]),
                }
                for name, values in measurements.items()
            },
        }
        await runtime.stop()
        runtime.store.db.close()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(benchmark(args.legacy))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
