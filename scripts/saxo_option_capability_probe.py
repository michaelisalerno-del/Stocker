"""Read-only probe of what Saxo offers on listed futures options (Level 2 research note, Stage 0).

For each option UIC given: whether an info-price GET with the MarketDepth field group returns option depth (and how
many levels), and the instrument details that decide which execution schemes are possible: supported order types and
their durations, multi-leg participation, DMA and tradability flags, tick size. Also the option root's own flags from
its option space. GET requests only, through SaxoClient's allow-list; reuses the service's current access token without
refreshing it; prints field names, flags and prices only - never account keys or tokens.

Run as the service user, for example:
  runuser -u stocker -- .venv/bin/python scripts/saxo_option_capability_probe.py \\
      --config /etc/stocker/v1/saxo.live.paper.yaml --state /var/lib/stocker/v1 --uics 60567931,61204548
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from stocker_execution.config import load
from stocker_execution.saxo_auth import OAuth, SaxoError, private_read
from stocker_execution.saxo_client import SaxoClient

FLAG_WORDS = ("Dma", "MultiLeg", "OrderType", "Duration", "Tradable", "TickSize", "Exercise", "Lot", "Amount", "Status")


class CurrentToken(OAuth):
    """Read the service's latest access token; never refresh or rotate it."""

    async def access_token(self) -> str:
        tokens = private_read(self.token_file)
        if tokens.get("environment") != self.environment:
            raise ValueError("TOKEN_ENVIRONMENT_MISMATCH")
        if float(tokens.get("expires_at", 0)) - time.time() < 60:
            raise ValueError("ACCESS_TOKEN_NEAR_EXPIRY_RETRY_SHORTLY")
        return str(tokens["access_token"])


def flags(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if any(w in k for w in FLAG_WORDS)}


async def probe(client: SaxoClient, uic: int, account: str) -> dict[str, Any]:
    out: dict[str, Any] = {"uic": uic}
    details = await client.request(
        "GET",
        f"/ref/v1/instruments/details/{uic}/FuturesOption",
        params={"FieldGroups": "OrderSetting,SupportedOrderTypeSettings,TradingSessions"},
    )
    out["symbol"] = details.get("Symbol")
    out["detail_keys"] = sorted(details)
    out["detail_flags"] = flags(details)
    out["supported_order_type_settings"] = details.get("SupportedOrderTypeSettings")
    out["order_distances"] = details.get("OrderDistances")
    roots = details.get("RelatedOptionRootsEnhanced") or []
    out["related_roots"] = roots
    root = details.get("OptionRootId") or next((r.get("OptionRootId") for r in roots if r.get("AssetType") == "FuturesOption"), None)
    try:
        price = await client.request(
            "GET",
            "/trade/v1/infoprices",
            params={"Uic": uic, "AssetType": "FuturesOption", "AccountKey": account,
                    "FieldGroups": "Quote,MarketDepth"},
        )
        depth = price.get("MarketDepth")
        out["market_depth"] = (
            None if depth is None else
            {"keys": sorted(depth), "bid_levels": len(depth.get("Bid") or []), "ask_levels": len(depth.get("Ask") or []),
             "bid": depth.get("Bid"), "ask": depth.get("Ask"), "bid_size": depth.get("BidSize"),
             "ask_size": depth.get("AskSize"), "using_orders": depth.get("UsingOrders"),
             "level2": depth.get("Level2PriceFeed")}
        )
        quote = price.get("Quote") or {}
        out["quote"] = {k: quote.get(k) for k in ("Bid", "Ask", "BidSize", "AskSize", "DelayedByMinutes", "PriceTypeBid")}
    except SaxoError as exc:
        out["market_depth_problem"] = str(exc)
    if root:
        space = await client.request("GET", f"/ref/v1/instruments/contractoptionspaces/{root}")
        out["root"] = {"id": root, "keys": sorted(k for k in space if k != "OptionSpace"),
                       "flags": flags({k: v for k, v in space.items() if k != "OptionSpace"})}
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="Directory holding the ledger")
    parser.add_argument("--uics", required=True, help="Comma-separated FuturesOption UICs")
    args = parser.parse_args()
    config = load(args.config)
    client = SaxoClient(CurrentToken(config.data_environment, config.saxo, args.state))
    report = []
    try:
        for uic in (int(u) for u in args.uics.split(",")):
            try:
                report.append(await probe(client, uic, client.oauth.account_key))
            except (SaxoError, ValueError, KeyError, TypeError) as exc:
                report.append({"uic": uic, "problem": str(exc) if isinstance(exc, (SaxoError, ValueError)) else type(exc).__name__})
            await asyncio.sleep(1.1)
    finally:
        await client.close()
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    asyncio.run(main())
