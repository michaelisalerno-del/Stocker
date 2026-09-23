"""Non-transmitting PAPER/OPRA check. No strategy admissions or trading authority."""

import argparse
import asyncio
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import stocker_launcher

stocker_launcher.configure_numeric_runtime()
stocker_launcher._ensure_monorepo_src_paths()

from ib_async import ComboLeg, Contract, Option, Stock  # noqa: E402

from stocker_execution.first4_broker import PaperBroker, expiration_at, listed_expiry  # noqa: E402
from stocker_execution.first4_config import PAPER_ACCOUNT, load  # noqa: E402
from stocker_execution.first4_store import Store  # noqa: E402


async def check(config_path: Path) -> dict:
    config = load(config_path)
    broker = PaperBroker(config, Store(Path(":memory:")))
    ib = broker.ib
    report = {
        "at": datetime.now(UTC).isoformat(),
        "purpose": "READ_ONLY_DATA_CHECK_NOT_FIRST4_SIGNAL",
        "transmitted_orders": 0,
        "settings": config.model_dump(),
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
        ib.reqMarketDataType(1)
        underlying = (await ib.qualifyContractsAsync(Stock("F", "SMART", "USD")))[0]
        ticker = (await ib.reqTickersAsync(underlying))[0]
        reference = ticker.marketPrice()
        assert math.isfinite(reference) and reference > 0, "PROBE_STOCK_QUOTE_UNAVAILABLE"
        chain = next(
            c
            for c in await broker.chain(underlying)
            if c.exchange == "SMART"
            and c.tradingClass == underlying.symbol
            and c.multiplier == "100"
        )
        expiry = listed_expiry(chain.expirations, datetime.now(UTC))
        # ATM contracts test OPRA access only, not the strategy's .98/1.02 mapping.
        strike = min(chain.strikes, key=lambda k: abs(k - reference))
        legs = []
        metadata = []
        for right in ("P", "C"):
            details = await ib.reqContractDetailsAsync(
                Option(
                    underlying.symbol,
                    expiry,
                    strike,
                    right,
                    "SMART",
                    multiplier="100",
                    currency="USD",
                    tradingClass=underlying.symbol,
                )
            )
            assert len(details) == 1 and details[0].underConId == underlying.conId
            d = details[0]
            assert d.contract.multiplier == "100" and d.contract.currency == "USD"
            legs.append(d.contract)
            metadata.append(
                {
                    "contract": d.contract.dict(),
                    "expiration_at": expiration_at(d).isoformat(),
                    "min_tick": d.minTick,
                    "min_size": d.minSize,
                    "size_increment": d.sizeIncrement,
                    "order_types": d.orderTypes,
                }
            )
        report["contracts"] = metadata
        report["checks"]["qualified_usd_standard_multiplier"] = True
        combo = Contract(
            secType="BAG",
            symbol=underlying.symbol,
            currency="USD",
            exchange="SMART",
            comboLegs=[
                ComboLeg(conId=leg.conId, ratio=1, action="BUY", exchange="SMART") for leg in legs
            ],
        )
        deadline = datetime.now(UTC) + timedelta(seconds=8)
        values = await asyncio.gather(
            broker.quotes(legs, deadline),
            broker.combo_tick(combo, deadline),
            return_exceptions=True,
        )
        for name, value in zip(
            ("fresh_realtime_option_quotes", "combo_price_increment"), values, strict=True
        ):
            report["checks"][name] = not isinstance(value, BaseException)
            if isinstance(value, BaseException):
                report["blockers"].append(f"{name}: {value}")
            else:
                report[name] = value
        report["broker_messages"] = broker.store.rows("meta")
    except Exception as exc:
        report["blockers"].append(str(exc) or type(exc).__name__)
    finally:
        ib.disconnect()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(check(args.config)), indent=2, default=str))
