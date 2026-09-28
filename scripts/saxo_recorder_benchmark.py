"""Accelerated, generated five-market workload. No provider connection or orders.

Measures the actual recorder implementation, including compression/fsync worker.
Virtual market time is not evidence of a live Saxo cadence or entitlement.
"""

import argparse
import asyncio
import json
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

from stocker_execution.config import MARKETS, RecorderConfig
from stocker_execution.recorder import Recorder


async def benchmark() -> dict:
    with tempfile.TemporaryDirectory(prefix="slrno-recorder-fixture-") as directory:
        recorder = Recorder(
            RecorderConfig(
                persistent_capture=True,
                recording_permission_evidence="GENERATED OFFLINE FIXTURE ONLY",
            ),
            Path(directory),
        )
        await recorder.start()
        for n, market in enumerate(MARKETS):
            recorder.register(
                market,
                {
                    "provider": "SAXO",
                    "environment": "SAXO_SIM",
                    "asset_type": "ContractFutures",
                    "uic": n + 100,
                    "market": market,
                    "fixture": True,
                },
            )
            recorder.ingest(market, "SNAPSHOT", {"Quote": {"Bid": 70, "Ask": 71}}, 0)
        timings, delays = [], []
        stopped = False

        async def heartbeat():
            while not stopped:
                before = time.perf_counter()
                await asyncio.sleep(0.001)
                delays.append(max(0, time.perf_counter() - before - 0.001))

        task = asyncio.create_task(heartbeat())
        started = time.perf_counter()
        for second in range(1, 7201):
            if second in (900, 1800):
                recorder.trigger(
                    "CL",
                    {
                        "id": f"fixture-CL-{second}",
                        "skip_reason": "PAPER_DISARMED",
                        "rule_version": "FIXTURE",
                    },
                    second,
                )
            if second == 1200:
                recorder.trigger(
                    "GC",
                    {"id": "fixture-GC", "skip_reason": "MONITOR_ONLY", "rule_version": "FIXTURE"},
                    second,
                )
            for n, market in enumerate(MARKETS):
                price = 70 + n + (second % 100) * 0.01
                payload = {
                    "Quote": {"Bid": price, "Ask": price + 0.01, "DelayedByMinutes": 0},
                    "PriceInfoDetails": {"BidSize": 2, "AskSize": 3},
                    "MarketDepth": {
                        "Bid": [price - i * 0.01 for i in range(10)],
                        "Ask": [price + (i + 1) * 0.01 for i in range(10)],
                        "BidSize": [second % 23 + i for i in range(10)],
                        "AskSize": [second % 31 + i for i in range(10)],
                        "NoOfBids": 10,
                        "NoOfOffers": 10,
                    },
                }
                before = time.perf_counter()
                recorder.ingest(
                    market,
                    "UPDATE",
                    payload,
                    second,
                    message_id=str(second * 17 + n),
                    provider_message={"Timestamp": second, "Data": payload},
                )
                timings.append(time.perf_counter() - before)
            recorder.tick(second)
            await asyncio.sleep(0)
            if second % 60 == 0:
                await recorder.queue.join()
        await recorder.close()
        stopped = True
        await task
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        return {
            "evidence": "OFFLINE_GENERATED_SIMULATION",
            "markets": list(MARKETS),
            "virtual_seconds": 7200,
            "requested_delivered_cadence_ms": 1000,
            "messages": 36005,
            "events": 3,
            "shared_segments": len(list(Path(directory).glob("*.gz"))),
            "wall_seconds": round(time.perf_counter() - started, 3),
            "process_peak_rss_bytes": rss_bytes,
            "ingest_p99_ms": round(statistics.quantiles(timings, n=100)[98] * 1000, 3),
            "ingest_max_ms": round(max(timings) * 1000, 3),
            "event_loop_max_delay_ms": round(max(delays) * 1000, 3),
            "buffer_coverage_seconds": {m: recorder.windows[m].coverage(7200) for m in MARKETS},
            **recorder.status(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(benchmark())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
