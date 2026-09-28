"""Non-transmitting broker snapshot. Does not arm, cancel, adopt or place orders."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ib_async import IB, ExecutionFilter

from stocker_execution.config import PAPER_ACCOUNT, load


class ReadOnlyIB(IB):
    def placeOrder(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Preflight cannot transmit orders")

    def cancelOrder(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Preflight cannot cancel orders")

    def reqGlobalCancel(self) -> None:
        raise RuntimeError("Preflight cannot cancel orders")


async def preflight(path: Path) -> dict[str, Any]:
    config = load(path)
    ib = ReadOnlyIB()
    try:
        await ib.connectAsync(
            config.host, config.port, clientId=89, readonly=True, timeout=10, raiseSyncErrors=True
        )
        if ib.managedAccounts() != [PAPER_ACCOUNT]:
            raise ValueError("ALLOWLISTED_PAPER_ACCOUNT_NOT_VERIFIED")
        async with asyncio.timeout(20):
            orders = await ib.reqAllOpenOrdersAsync()
            executions = await ib.reqExecutionsAsync(ExecutionFilter(acctCode=PAPER_ACCOUNT))
            positions = await ib.reqPositionsAsync()
        return {
            "at": datetime.now(UTC).isoformat(),
            "account": PAPER_ACCOUNT,
            "orders_sent": 0,
            "configured_armed": config.armed,
            "open_orders": [
                {
                    "reference": t.order.orderRef,
                    "contract": t.contract.dict(),
                    "status": t.orderStatus.status,
                }
                for t in orders
                if t.order.account == PAPER_ACCOUNT
            ],
            "positions": [
                {"contract": p.contract.dict(), "quantity": p.position}
                for p in positions
                if p.account == PAPER_ACCOUNT and p.position
            ],
            "returned_executions": len(executions),
            "mapping_authority": {m: p.approval for m, p in config.mappings.items()},
            "market_data": {
                **config.market_data.model_dump(),
                "effective_app_budget": config.market_data.line_budget,
                "effective_outbound_cap": config.market_data.request_budget,
                "external_usage_observed_by_this_preflight": None,
                "entitlements_verified_by_this_preflight": False,
            },
            "l2_enabled": config.l2.enabled,
            "note": "Account snapshot only; quotes, products and actual fills remain unverified",
        }
    finally:
        ib.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(preflight(args.config)), indent=2, default=str))
