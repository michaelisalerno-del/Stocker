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
    loaded = load(config)
    # Take the service's environment ownership lock before opening the ledger, so a
    # refused run cannot migrate, audit or otherwise write the live ledger.
    lock = database.resolve().parent / loaded.data_environment / "data.owner.lock"
    lock.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    owner = lock.open("a")
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner.close()
        raise SystemExit("DATA_OWNER_LOCK_HELD: stop the SLRNO service first") from None
    try:
        runtime = Runtime(loaded, Store(database))
    except BaseException:
        owner.close()
        raise
    runtime.owner = owner
    if runtime.root != lock.parent.parent:
        await runtime.stop()
        raise SystemExit("DATA_OWNER_LOCK_PATH_MISMATCH")
    await runtime.recorder.start()
    # Deliberately omit the execution manager and trigger loop, even if this
    # ledger contains obligations. Their owning runtime must keep managing them.
    task = asyncio.create_task(runtime.data.run())
    runtime.tasks.add(task)
    try:
        await asyncio.sleep(seconds)
        if task.done():
            task.result()
        try:
            result = await runtime.broker.preflight()
        except ValueError as exc:  # coded reasons only (ACCOUNT_SELECTION_REQUIRED, ...)
            result = {"non_transmitting": True, "problem": str(exc) or "PREFLIGHT_UNAVAILABLE"}
        return {
            "at": datetime.now(UTC).isoformat(),
            "orders_sent": 0,
            "paper_armed": False,
            "preflight": result,
            "system": runtime.status(),
            "markets": {m: runtime.data.capability_view(s) for m, s in runtime.markets.items()},
        }
    finally:
        await runtime.stop()  # cancels the data task it owns
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
