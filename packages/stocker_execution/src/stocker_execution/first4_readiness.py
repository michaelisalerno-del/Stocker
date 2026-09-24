"""One non-transmitting option-data check, shared by the CLI and opening gate."""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Any

from ib_async import ComboLeg, Contract, Option, Stock

from stocker_execution.first4_broker import PaperBroker, expiration_at, now
from stocker_execution.first4_config import PAPER_ACCOUNT


async def option_access(broker: PaperBroker, deadline: datetime) -> dict[str, Any]:
    """Check actual data; Ford ATM legs are diagnostic contracts, never candidates."""
    generation = broker.market_data_generation
    connection_generation = broker.connection_generation
    ib = broker.ib
    if ib.managedAccounts() != [PAPER_ACCOUNT]:
        raise ValueError("PAPER_IDENTITY_MISMATCH")
    broker.config.require_settings()
    ib.reqMarketDataType(1)
    # Qualification is bounded separately; only quote subscriptions wait for opening.
    async with asyncio.timeout(30):
        underlying = (await ib.qualifyContractsAsync(Stock("F", "SMART", "USD")))[0]
        ticker = (await ib.reqTickersAsync(underlying))[0]
        reference = ticker.marketPrice()
        if (
            ticker.marketDataType != 1
            or ticker.time is None
            or not 0
            <= (now() - ticker.time).total_seconds()
            <= broker.config.number("quote_max_age_seconds")
            or not math.isfinite(reference)
            or reference <= 0
        ):
            raise ValueError("PROBE_STOCK_QUOTE_UNAVAILABLE")
        chain = next(
            c
            for c in await broker.chain(underlying)
            if c.exchange == "SMART" and c.tradingClass == "F" and c.multiplier == "100"
        )
        target_day = (now() + timedelta(days=2)).date()
        expiry = min(
            (e for e in chain.expirations if e > now().strftime("%Y%m%d")),
            key=lambda e: (abs((datetime.strptime(e, "%Y%m%d").date() - target_day).days), -int(e)),
        )
        strike = min(chain.strikes, key=lambda k: (abs(k - reference), k))
        legs, metadata = [], []
        for right in ("P", "C"):
            details = await ib.reqContractDetailsAsync(
                Option(
                    "F",
                    expiry,
                    strike,
                    right,
                    "SMART",
                    multiplier="100",
                    currency="USD",
                    tradingClass="F",
                )
            )
            if len(details) != 1 or details[0].underConId != underlying.conId:
                raise ValueError("PROBE_CONTRACT_AMBIGUOUS")
            d = details[0]
            c = d.contract
            if (
                c.multiplier != "100"
                or c.currency != "USD"
                or c.secType != "OPT"
                or c.conId <= 0
                or c.tradingClass != "F"
                or c.localSymbol != f"{'F':<6}{expiry[2:]}{right}{round(strike * 1000):08d}"
                or not 0 < d.minSize <= 1
                or not 0 < d.sizeIncrement <= 1
                or not math.isclose(1 / d.sizeIncrement, round(1 / d.sizeIncrement))
                or not {"LMT", "GTD"}.issubset(d.orderTypes.split(","))
            ):
                raise ValueError("PROBE_CONTRACT_OR_QUANTITY_UNVERIFIED")
            legs.append(c)
            metadata.append(
                {
                    "contract": c.dict(),
                    "expiration_at": expiration_at(d).isoformat(),
                    "min_tick": d.minTick,
                    "min_size": d.minSize,
                    "size_increment": d.sizeIncrement,
                    "order_types": d.orderTypes,
                }
            )
    combo = Contract(
        secType="BAG",
        symbol="F",
        currency="USD",
        exchange="SMART",
        comboLegs=[ComboLeg(conId=c.conId, ratio=1, action="BUY", exchange="SMART") for c in legs],
    )
    # Resolve metadata before checking freshness, so a slow BAG response cannot
    # turn previously fresh leg quotes into evidence for arming minutes later.
    tick = await broker.combo_tick(combo, min(deadline, now() + timedelta(seconds=8)))
    quotes = await broker.quotes(legs, deadline)
    if connection_generation != broker.connection_generation:
        raise ValueError("CONNECTION_CHANGED_DURING_DATA_VERIFICATION")
    report = {
        "at": now().isoformat(),
        "purpose": "READ_ONLY_DATA_CHECK_NOT_FIRST4_SIGNAL",
        "transmitted_orders": 0,
        "contracts": metadata,
        "checks": {
            "qualified_usd_standard_multiplier": True,
            "fresh_realtime_option_quotes": True,
            "combo_price_increment": True,
        },
        "fresh_realtime_option_quotes": quotes,
        "combo_price_increment": tick,
        "blockers": [],
    }

    broker.confirm_market_data(generation, report)
    return report
