"""Authenticated read-only capability snapshot. Never arms or submits test orders."""

import argparse
import asyncio
import fcntl
import json
from datetime import UTC, datetime
from pathlib import Path

from stocker_execution.config import load
from stocker_execution.runtime import Runtime
from stocker_execution.store import Store


async def preflight(config: Path, database: Path, seconds: int) -> dict:
    runtime = Runtime(load(config), Store(database))
    runtime.owner = (runtime.root / runtime.config.data_environment / "data.owner.lock").open("a")
    fcntl.flock(runtime.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    await runtime.recorder.start()
    # Deliberately omit the execution manager and trigger loop, even if this
    # ledger contains obligations. Their owning runtime must keep managing them.
    task = asyncio.create_task(runtime.data.run())
    runtime.tasks.add(task)
    try:
        await asyncio.sleep(min(seconds, 120))
        if task.done():
            task.result()
        try:
            result = await runtime.broker.preflight()
        except ValueError:
            result = {"non_transmitting": True, "problem": "PREFLIGHT_UNAVAILABLE"}
        return {
            "at": datetime.now(UTC).isoformat(),
            "orders_sent": 0,
            "paper_armed": False,
            "preflight": result,
            "system": runtime.status(),
            "markets": {m: runtime.data.capability_view(s) for m, s in runtime.markets.items()},
        }
    finally:
        await runtime.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        runtime.store.db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=60, choices=range(5, 121))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The shared environment ownership lock rejects running this beside the service.
    args.output.write_text(
        json.dumps(
            asyncio.run(preflight(args.config, args.database, args.seconds)), indent=2, default=str
        )
        + "\n"
    )
