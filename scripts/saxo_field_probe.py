"""Read-only Saxo field probe: checks live field shapes the dashboard and gates rely on.

Safe beside the running service: it reuses the service's current access token without ever
refreshing it (the service owns the rotating refresh token), every request goes through
SaxoClient's allow-list, no order endpoint is called, and the output contains field names,
enum values and prices only - never account keys, tokens or credentials. The one options-chain
subscription it opens (for a snapshot) is deleted immediately.

Run as the service user, for example:
  runuser -u stocker -- .venv/bin/python scripts/saxo_field_probe.py \\
      --config /etc/stocker/v1/saxo.sim.yaml --state /var/lib/stocker/v1
"""

import argparse
import asyncio
import json
import math
import re
import secrets
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocker_execution.config import MARKETS, load
from stocker_execution.saxo_auth import OAuth, private_read
from stocker_execution.saxo_client import SaxoClient, SaxoError

MONTHS = "FGHJKMNQUVXZ"
ALTERNATIVE_KEYWORDS = {
    "NQ": ("Nasdaq", "Nasdaq 100", "E-mini Nasdaq"),
    "CL": ("Crude",),
    "GC": ("Gold",),
    "NG": ("Natural Gas",),
    "SI": ("Silver",),
}


class CurrentToken(OAuth):
    """Read the service's latest access token; never refresh or rotate it."""

    async def access_token(self) -> str:
        tokens = private_read(self.token_file)
        if tokens.get("environment") != self.environment:
            raise ValueError("TOKEN_ENVIRONMENT_MISMATCH")
        if float(tokens.get("expires_at", 0)) - time.time() < 60:
            raise ValueError("ACCESS_TOKEN_NEAR_EXPIRY_RETRY_SHORTLY")
        return str(tokens["access_token"])


def contract_order(market: str, symbol: str, today: datetime) -> tuple[int, int] | None:
    match = re.match(market + r"([FGHJKMNQUVXZ])(\d{1,2})(?::|$)", symbol)
    if not match:
        return None
    digits, month = match[2], MONTHS.index(match[1]) + 1
    year = today.year - today.year % 10 ** len(digits) + int(digits)
    if year < today.year:
        year += 10 ** len(digits)
    return (year, month) if (year, month) >= (today.year, today.month) else None


def shape(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {k: value.get(k) for k in names if k in value}


async def probe(client: SaxoClient, market: str, account: str, chain: bool) -> dict[str, Any]:
    today = datetime.now(UTC)
    out: dict[str, Any] = {"market": market}
    listed = await client.request(
        "GET",
        "/ref/v1/instruments",
        params={"Keywords": market, "AssetTypes": "ContractFutures", "$top": 100},
    )
    ranked = sorted(
        (order, row)
        for row in listed.get("Data", [])
        if (order := contract_order(market, str(row.get("Symbol", "")), today))
    )
    out["listed_contracts"] = len(ranked)
    if not ranked:
        # Report what Saxo does list under broader names so a UIC can be chosen by hand.
        for words in ALTERNATIVE_KEYWORDS.get(market, ()):
            found = await client.request(
                "GET",
                "/ref/v1/instruments",
                params={"Keywords": words, "AssetTypes": "ContractFutures", "$top": 20},
            )
            out.setdefault("alternative_search", {})[words] = [
                shape(r, ("Identifier", "Symbol", "Description", "ExchangeId"))
                for r in found.get("Data", [])[:10]
            ]
    future = None
    for _, row in ranked[:3]:
        details = await client.request(
            "GET",
            f"/ref/v1/instruments/details/{row['Identifier']}/ContractFutures",
            params={"AccountKey": account, "FieldGroups": "TradingSessions"},
        )
        if str(details.get("ExpiryDate", "")) > today.isoformat()[:10]:
            future = details
            break
    if future is None:
        out["problem"] = "NO_UNEXPIRED_CONTRACT_FOUND"
        return out
    uic = future["Uic"]
    sessions = (future.get("TradingSessions") or {}).get("Sessions", [])
    out["future"] = {
        **shape(
            future,
            (
                "Uic",
                "Symbol",
                "ExpiryDate",
                "ContractSize",
                "PriceToContractFactor",
                "TickSize",
                "CurrencyCode",
                "IsTradable",
            ),
        ),
        "session_states": sorted({str(s.get("State")) for s in sessions}),
        "current_session": next(
            (
                s.get("State")
                for s in sessions
                if s.get("StartTime", "") <= today.isoformat() < s.get("EndTime", "")
            ),
            None,
        ),
    }
    bars = await client.request(
        "GET",
        "/chart/v3/charts",
        params={
            "Uic": uic,
            "AssetType": "ContractFutures",
            "Horizon": 1,
            "Count": 60,
            "FieldGroups": "Data,ChartInfo",
        },
    )
    rows = bars.get("Data", [])
    closes = [r["Close"] for r in rows if isinstance(r.get("Close"), (int, float))]
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False) if a > 0 and b > 0]
    out["chart_v3_minute"] = {
        "samples": len(rows),
        "sample_fields": sorted({k for r in rows for k in r}),
        "chart_info": bars.get("ChartInfo"),
        "has_data_version": "DataVersion" in bars,
        "all_on_minute_boundaries": all(str(r.get("Time", ""))[17:19] in {"00", ""} for r in rows),
        "first": rows[0].get("Time") if rows else None,
        "last": rows[-1].get("Time") if rows else None,
        # Annualised from one-minute log returns, the same basis the frozen model uses.
        "realised_vol_annualised": math.sqrt(statistics.fmean(r * r for r in returns) * 525600)
        if len(returns) > 10
        else None,
    }
    daily = await client.request(
        "GET",
        "/chart/v3/charts",
        params={
            "Uic": uic,
            "AssetType": "ContractFutures",
            "Horizon": 1440,
            "Count": 5,
            "FieldGroups": "Data",
        },
    )
    out["chart_v3_daily"] = {
        "samples": len(daily.get("Data", [])),
        "last": (daily.get("Data") or [{}])[-1],
    }
    price = await client.request(
        "GET",
        "/trade/v1/infoprices",
        params={
            "Uic": uic,
            "AssetType": "ContractFutures",
            "AccountKey": account,
            "FieldGroups": "Quote,PriceInfo,PriceInfoDetails,InstrumentPriceDetails,MarketDepth",
        },
    )
    depth = price.get("MarketDepth") or {}
    out["future_infoprice"] = {
        "quote": shape(
            price.get("Quote"),
            (
                "Bid",
                "Ask",
                "BidSize",
                "AskSize",
                "Mid",
                "PriceTypeBid",
                "PriceTypeAsk",
                "MarketState",
                "DelayedByMinutes",
                "ErrorCode",
                "PriceSource",
            ),
        ),
        "price_info": price.get("PriceInfo"),
        "price_info_details": price.get("PriceInfoDetails"),
        "instrument_price_details": shape(
            price.get("InstrumentPriceDetails"), ("IsMarketOpen", "OpenInterest", "NoticeDate")
        ),
        "depth_levels": {"bid": len(depth.get("Bid") or []), "ask": len(depth.get("Ask") or [])},
    }
    mid = (price.get("Quote") or {}).get("Mid")
    roots = [
        r["OptionRootId"]
        for r in future.get("RelatedOptionRootsEnhanced", [])
        if r.get("AssetType") == "FuturesOption"
    ]
    out["option_roots"] = roots
    if not roots or not isinstance(mid, (int, float)):
        out["option_problem"] = "NO_OPTION_ROOT_OR_UNDERLYING_MID"
        return out
    space = await client.request(
        "GET",
        f"/ref/v1/instruments/contractoptionspaces/{roots[0]}",
        params={"OptionSpaceSegment": "UnderlyingUic", "UnderlyingUic": uic},
    )
    expiries = [e for e in space.get("OptionSpace", []) if e.get("SpecificOptions")]
    if not expiries:
        out["option_problem"] = "NO_LISTED_EXPIRY_FOR_UNDERLYING"
        return out
    expiry = expiries[0]
    options = [o for o in expiry["SpecificOptions"] if o.get("UnderlyingUic") == uic]
    if not options:
        out["option_problem"] = "NO_OPTIONS_ON_SELECTED_UNDERLYING"
        return out
    atm = min(options, key=lambda o: (abs(float(o["StrikePrice"]) - mid), o["PutCall"]))
    out["option_expiry"] = shape(expiry, ("Expiry", "LastTradeDate", "DisplayDaysToExpiry"))
    out["options"] = []
    for right in ("Put", "Call"):
        option = next(
            (
                o
                for o in options
                if o["PutCall"] == right and o["StrikePrice"] == atm["StrikePrice"]
            ),
            None,
        )
        if option is None:
            continue
        quote = await client.request(
            "GET",
            "/trade/v1/infoprices",
            params={
                "Uic": option["Uic"],
                "AssetType": "FuturesOption",
                "AccountKey": account,
                "FieldGroups": "Quote,Greeks,PriceInfoDetails,InstrumentPriceDetails",
            },
        )
        out["options"].append(
            {
                "right": right,
                "strike": option["StrikePrice"],
                "uic": option["Uic"],
                "quote": shape(
                    quote.get("Quote"),
                    (
                        "Bid",
                        "Ask",
                        "BidSize",
                        "AskSize",
                        "PriceTypeBid",
                        "PriceTypeAsk",
                        "MarketState",
                        "DelayedByMinutes",
                        "ErrorCode",
                    ),
                ),
                "greeks": quote.get("Greeks"),
                "open_interest": (quote.get("InstrumentPriceDetails") or {}).get("OpenInterest"),
            }
        )
    if chain:

        async def window(start: int) -> dict[str, Any]:
            context, reference = "SLRNOPROBE" + secrets.token_hex(6), "P" + secrets.token_hex(6)
            path = "/trade/v1/optionschain/subscriptions"
            arguments = {
                "Identifier": roots[0],
                "AssetType": "FuturesOption",
                "AccountKey": account,
                "MaxStrikesPerExpiry": 5,
                "Expiries": [{"Index": 0, "StrikeStartIndex": start}],
            }
            try:
                result = await client.request(
                    "POST",
                    path,
                    body={
                        "ContextId": context,
                        "ReferenceId": reference,
                        "RefreshRate": 2000,
                        "Format": "application/json",
                        "Arguments": arguments,
                    },
                )
            finally:
                await client.request("DELETE", f"{path}/{context}/{reference}")
            return dict(result.get("Snapshot") or {})

        # Binary-search the five-strike window onto the underlying price.
        snapshot = await window(0)
        first = (snapshot.get("Expiries") or [{}])[0]
        low, high = 0, max(0, int(first.get("StrikeCount") or 0) - 5)
        target = first.get("MidStrikePrice") or mid
        for _ in range(7):
            if low >= high:
                break
            middle = (low + high) // 2
            attempt = await window(middle)
            strikes = (attempt.get("Expiries") or [{}])[0].get("Strikes") or []
            if not strikes:
                break
            snapshot = attempt
            if strikes[-1].get("Strike", 0) < target:
                low = middle + 1
            elif strikes[0].get("Strike", 0) > target:
                high = middle - 1
            else:
                break
        first = (snapshot.get("Expiries") or [{}])[0]
        out["chain_snapshot"] = {
            "top_fields": sorted(snapshot),
            "implied_volatility_data": snapshot.get("ImpliedVolatilityData"),
            "expiry": shape(
                first,
                (
                    "Expiry",
                    "DisplayDate",
                    "LastTradeDate",
                    "MidStrikePrice",
                    "StrikeCount",
                    "UnderlyingUic",
                ),
            ),
            "strikes": [
                {
                    "strike": s.get("Strike"),
                    "mid_volatility_pct": s.get("MidVolatilityPct"),
                    "put": shape(
                        s.get("Put"),
                        (
                            "Bid",
                            "Ask",
                            "OpenInterest",
                            "Volume",
                            "Greeks",
                            "PriceTypeBid",
                            "PriceTypeAsk",
                        ),
                    ),
                    "call": shape(
                        s.get("Call"),
                        ("Bid", "Ask", "OpenInterest", "Greeks", "PriceTypeBid", "PriceTypeAsk"),
                    ),
                }
                for s in first.get("Strikes") or []
            ],
        }
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="Directory holding the ledger")
    parser.add_argument("--markets", default=",".join(MARKETS))
    parser.add_argument("--no-chain", action="store_true")
    args = parser.parse_args()
    config = load(args.config)
    client = SaxoClient(CurrentToken(config.data_environment, config.saxo, args.state))
    report: dict[str, Any] = {
        "at": datetime.now(UTC).isoformat(),
        "environment": config.data_environment,
        "markets": [],
    }
    try:
        for market in args.markets.split(","):
            try:
                report["markets"].append(
                    await probe(client, market, client.oauth.account_key, not args.no_chain)
                )
            except (SaxoError, ValueError, KeyError, TypeError) as exc:
                code = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                report["markets"].append({"market": market, "problem": code})
    finally:
        await client.close()
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    asyncio.run(main())
