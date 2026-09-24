"""Non-transmitting PAPER/OPRA check. No strategy admissions or trading authority."""

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import stocker_launcher

stocker_launcher.configure_numeric_runtime()
stocker_launcher._ensure_monorepo_src_paths()

from stocker_execution.first4_broker import PaperBroker  # noqa: E402
from stocker_execution.first4_config import PAPER_ACCOUNT, load  # noqa: E402
from stocker_execution.first4_readiness import option_access  # noqa: E402
from stocker_execution.first4_store import Store  # noqa: E402


async def check(config_path: Path) -> dict:
    config = load(config_path)
    broker = PaperBroker(config, Store(Path(":memory:")))
    ib = broker.ib
    report = {
        "at": datetime.now(UTC).isoformat(),
        "purpose": "READ_ONLY_DATA_CHECK_NOT_FIRST4_SIGNAL",
        "transmitted_orders": 0,
        "settings": config.model_dump(mode="json"),
        "checks": {},
        "blockers": [],
    }

    def forbid(*args, **kwargs):
        raise AssertionError("This check cannot transmit orders, including what-if orders")

    ib.placeOrder = forbid
    ib.client.placeOrder = forbid
    try:
        await ib.connectAsync(
            config.host, config.port, clientId=181, readonly=True, account=PAPER_ACCOUNT, timeout=10
        )
        assert ib.managedAccounts() == [PAPER_ACCOUNT], "PAPER_IDENTITY_MISMATCH"
        await broker.reconcile()
        report["checks"]["paper_identity_and_reconciliation"] = (
            broker.reconciled and not broker.entry_blocker
        )
        report["checks"]["zero_open_orders"] = not await ib.reqAllOpenOrdersAsync()
        result = await option_access(broker, datetime.now(UTC) + timedelta(seconds=45))
        report["checks"].update(result.pop("checks"))
        report.update(result)
    except Exception as exc:
        report["blockers"].append(str(exc) or type(exc).__name__)
    finally:
        report["broker_messages"] = broker.store.rows("meta")
        ib.disconnect()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(check(args.config)), indent=2, default=str))
