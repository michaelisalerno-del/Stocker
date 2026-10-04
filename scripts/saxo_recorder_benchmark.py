"""Accelerated, generated five-market workload. No provider connection or orders.

Measures the actual recorder implementation, including compression/fsync worker.
Virtual market time is not evidence of a live Saxo cadence or entitlement.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

from stocker_execution import option_context
from stocker_execution.book_flow import VERSION
from stocker_execution.config import MARKETS, RecorderConfig
from stocker_execution.recorder import Recorder
from stocker_execution.saxo_stream import PriceState


async def benchmark(
    option_count: int = 0, virtual_seconds: int = 7200, burst_events: bool = False
) -> dict:
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
                    "tick_size": 0.01,
                    "fixture": True,
                },
            )
            recorder.ingest(market, "SNAPSHOT", {"Quote": {"Bid": 70, "Ask": 71}}, 0)
        options = {}
        for n in range(option_count):
            instrument = f"option-{n}"
            recorder.register(
                instrument,
                {
                    "provider": "SAXO",
                    "environment": "SAXO_SIM",
                    "asset_type": "FuturesOption",
                    "uic": n + 1000,
                    "underlying_uic": 100 + n % 5,
                    "market": MARKETS[n % 5],
                    "right": "Call",
                    "strike": 70,
                },
            )
            recorder.metadata_version(
                instrument,
                {
                    "fixture": True,
                    "ContractSize": 1000,
                    "PriceToContractFactor": 1000,
                    "LotSize": 1,
                },
                0,
            )
            p = PriceState()
            p.snapshot({"Quote": {"Bid": 0.01, "Ask": 0.02}}, instrument, 0)
            options[instrument] = p
            recorder.ingest(
                instrument, "SNAPSHOT", p.value, 0, observation_context=p.observation_context()
            )
        timings, delays = [], []
        stopped = False

        async def heartbeat():
            while not stopped:
                before = time.perf_counter()
                await asyncio.sleep(0.001)
                delays.append(max(0, time.perf_counter() - before - 0.001))

        task = asyncio.create_task(heartbeat())
        started = time.perf_counter()
        cpu_started = time.process_time()
        event_count = 0
        peak_queue = 0
        for second in range(1, virtual_seconds + 1):
            if burst_events and second == 900:
                for market in MARKETS:
                    recorder.trigger(
                        market,
                        {"id": f"burst-{market}", "skip_reason": "OFFLINE_FIXTURE"},
                        second,
                        [k for k in options if recorder.windows[k].identity["market"] == market],
                    )
                    event_count += 1
                    peak_queue = max(peak_queue, recorder.queued_bytes)
            if second in (900, 1800):
                recorder.trigger(
                    "CL",
                    {
                        "id": f"fixture-CL-{second}",
                        "skip_reason": "PAPER_DISARMED",
                        "rule_version": "FIXTURE",
                    },
                    second,
                    [k for k in options if recorder.windows[k].identity["market"] == "CL"],
                )
                event_count += 1
            if second == 1200:
                recorder.trigger(
                    "GC",
                    {"id": "fixture-GC", "skip_reason": "MONITOR_ONLY", "rule_version": "FIXTURE"},
                    second,
                    [k for k in options if recorder.windows[k].identity["market"] == "GC"],
                )
                event_count += 1
            for n, market in enumerate(MARKETS):
                price = 70 + n + (second % 100) * 0.01
                payload = {
                    "Quote": {
                        "Bid": price,
                        "Ask": price + 0.01,
                        "BidSize": 2,
                        "AskSize": 3,
                        "DelayedByMinutes": 0,
                        "PriceTypeBid": "Tradable",
                        "PriceTypeAsk": "Tradable",
                    },
                    "PriceInfoDetails": {
                        "LastTraded": price,
                        "LastTradedSize": 1,
                        "Volume": second,
                    },
                    "MarketDepth": {
                        "Bid": [price - i * 0.01 for i in range(10)],
                        "Ask": [price + (i + 1) * 0.01 for i in range(10)],
                        "BidSize": [second % 23 + i for i in range(10)],
                        "AskSize": [second % 31 + i for i in range(10)],
                        "NoOfBids": 10,
                        "NoOfOffers": 10,
                        "UsingOrders": True,
                        "BidOrders": [2] * 10,
                        "AskOrders": [1] * 10,
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
                    observation_context={
                        "valid_until": second + 30,
                        "subscription_id": market,
                        "granted_refresh_ms": 1000,
                        "last_receipt": second,
                        "last_contact": second,
                        "last_field_change": second,
                        "last_depth_change": second,
                        "last_trade_observation_change": second,
                        "observed_receipt_ms": {
                            "samples": 60,
                            "mean": 1000,
                            "minimum": 1000,
                            "maximum": 1000,
                        },
                        "problem": "",
                    },
                )
                timings.append(time.perf_counter() - before)
            for instrument, p in options.items():
                payload = {
                    "Quote": {
                        "Bid": 0.01 + (second % 10) * 0.001,
                        "Ask": 0.03,
                        "BidSize": 2,
                        "AskSize": 3,
                        "PriceTypeBid": "Tradable",
                        "PriceTypeAsk": "Tradable",
                        "DelayedByMinutes": 0,
                    },
                    "Greeks": {
                        "Delta": 0.1,
                        "Gamma": 0.01,
                        "Theta": -0.01,
                        "Vega": 0.1,
                        "MidVol": 0.2,
                    },
                    "PriceInfoDetails": {"LastTraded": 0.02, "LastTradedSize": 1, "Volume": second},
                }
                if second % 60 == 0:
                    payload["InstrumentPriceDetails"] = {"OpenInterest": 1000}
                p.update(payload, str(second), second)
                before = time.perf_counter()
                recorder.ingest(
                    instrument,
                    "UPDATE",
                    payload,
                    second,
                    message_id=str(second),
                    observation_context=p.observation_context(),
                    provider_message={"Data": payload},
                )
                timings.append(time.perf_counter() - before)
                if second % 2 == 0:
                    side = {
                        "Uic": recorder.windows[instrument].identity["uic"],
                        "Greeks": {
                            "Delta": 0.1,
                            "MidVolatility": 0.2,
                            "BidVolatility": 0.19,
                            "AskVolatility": 0.21,
                        },
                    }
                    recorder.ingest(
                        instrument,
                        "CHAIN_CONTEXT",
                        side,
                        second,
                        context_source="OPTIONS_CHAIN",
                        observation_context={
                            "executable": False,
                            "analytics": option_context.fields(
                                option_context.chain_update(side), second, "OPTIONS_CHAIN"
                            ),
                        },
                    )
            recorder.tick(second)
            peak_queue = max(peak_queue, recorder.queued_bytes)
            await asyncio.sleep(0)
            if second % 60 == 0:
                await recorder.queue.join()
        await recorder.close()
        stopped = True
        await task
        elapsed = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        evidence = []
        duplicates = 0
        capture_states = []
        for path in sorted(Path(directory).glob("*.jsonl.gz")):
            seen = set()
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    if "local_sequence" in row:
                        identity = row["identity"]
                        identity_key = (identity["uic"], row["local_sequence"])
                        duplicates += identity_key in seen
                        seen.add(identity_key)
                        evidence.append(json.dumps(row, sort_keys=True))
        for path in Path(directory).glob("*.manifest.json"):
            capture_states.append(json.loads(path.read_text())["state"])
        evidence_hash = hashlib.sha256("\n".join(sorted(evidence)).encode()).hexdigest()
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024
        return {
            "evidence": "OFFLINE_GENERATED_SIMULATION",
            "feature_version": VERSION,
            "markets": list(MARKETS),
            "option_count": option_count,
            "virtual_seconds": virtual_seconds,
            "requested_delivered_cadence_ms": 1000,
            "messages": 5 * (virtual_seconds + 1)
            + (virtual_seconds + 1 + virtual_seconds // 2) * option_count,
            "option_chain_cadence_ms": 2000,
            "events": event_count,
            "simultaneous_five_market_triggers": burst_events,
            "queue_peak_bytes": peak_queue,
            "shared_segments": len(list(Path(directory).glob("*.gz"))),
            "wall_seconds": round(elapsed, 3),
            "cpu_seconds": round(cpu_seconds, 3),
            "cpu_percent_of_one_core": round(cpu_seconds / elapsed * 100, 2),
            "archived_message_rows": len(evidence),
            "duplicate_message_rows_within_segment": duplicates,
            "archived_message_sha256": evidence_hash,
            "capture_states": sorted(capture_states),
            "process_peak_rss_bytes": rss_bytes,
            "ingest_p99_ms": round(statistics.quantiles(timings, n=100)[98] * 1000, 3),
            "ingest_max_ms": round(max(timings) * 1000, 3),
            "event_loop_max_delay_ms": round(max(delays) * 1000, 3),
            "buffer_coverage_seconds": {
                m: recorder.windows[m].coverage(virtual_seconds) for m in MARKETS
            },
            "option_coverage_seconds": {
                k: recorder.windows[k].coverage(virtual_seconds) for k in options
            },
            **recorder.status(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--option-count", type=int, choices=range(17), default=0)
    parser.add_argument("--virtual-seconds", type=int, choices=(1200, 7200), default=7200)
    parser.add_argument("--burst-events", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(benchmark(args.option_count, args.virtual_seconds, args.burst_events))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
