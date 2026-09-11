import asyncio
import sqlite3
from datetime import UTC, datetime

import pytest

from stocker_core.markets import ActivityScanner, CapBucket, MarketId, get_market
from stocker_execution.activity_shortlist import ActivityShortlistService, ActivityShortlistStore
from test_session_hard_universe import MarketBroker
from test_stage10_extension_ibkr_scanner import ScannerClient, connection


@pytest.mark.parametrize("market_id", [MarketId.US_ALL, MarketId.UK_LSE, MarketId.AUSTRALIA_ASX])
def test_broker_filters_corporations_before_the_fifty_row_scanner_limit(market_id):
    market = get_market(market_id)

    class LiquidityClient(ScannerClient):
        def __init__(self):
            super().__init__()
            self.classification_requests = []

        async def reqContractDetailsAsync(self, contract):
            from types import SimpleNamespace

            self.classification_requests.append(contract.conId)
            if contract.conId == 100:
                raise TimeoutError("Classification unavailable")
            return [
                SimpleNamespace(
                    contract=contract,
                    stockType="ETC" if contract.conId == 99 else "COMMON",
                )
            ]

        async def reqScannerParametersAsync(self):
            return (
                "<ScanParameterResponse>"
                f"<Location><locationCode>{market.scanner_location}</locationCode></Location>"
                "<ScanType><scanCode>MOST_ACTIVE_AVG_USD</scanCode></ScanType>"
                "</ScanParameterResponse>"
            )

        async def reqScannerDataAsync(self, *args):
            rows = await super().reqScannerDataAsync(*args)
            # Observed UK bug: CORP returns ETCs, and scanner rows omit stockType.
            from copy import deepcopy

            fund = deepcopy(rows[0])
            fund.contractDetails.stockType = ""
            fund.contractDetails.contract.symbol = "FUND"
            fund.contractDetails.contract.conId = 99
            fund.rank = 1
            unknown = deepcopy(fund)
            unknown.contractDetails.contract.symbol = "UNKNOWN"
            unknown.contractDetails.contract.conId = 100
            unknown.rank = 2
            return [*rows, fund, unknown]

    client = LiquidityClient()
    broker = connection(client)
    rows = asyncio.run(
        broker.activity_scan(
            market=market,
            cap_bucket=CapBucket.ALL,
            component=ActivityScanner.MOST_ACTIVE_AVG_USD,
            max_results=50,
            stock_type_filter="CORP",
        )
    )
    subscription = client.scanner_requests[0]
    assert subscription.stockTypeFilter == "CORP"
    assert subscription.scanCode == "MOST_ACTIVE_AVG_USD"
    assert subscription.locationCode == market.scanner_location
    assert subscription.instrument == market.scanner_instrument
    assert subscription.numberOfRows == 50
    assert [r.symbol for r in rows] == ["MSFT"]
    assert "ETC" in rows[0].warning
    assert "UNKNOWN" in rows[0].warning
    asyncio.run(
        broker.activity_scan(
            market=market,
            cap_bucket=CapBucket.ALL,
            component=ActivityScanner.MOST_ACTIVE_AVG_USD,
            max_results=50,
            stock_type_filter="CORP",
        )
    )
    assert client.classification_requests.count(272093) == 1
    assert client.classification_requests.count(99) == 1


def test_old_snapshot_remains_readable_before_and_after_schema_upgrade(tmp_path):
    path = tmp_path / "old.sqlite"
    now = datetime(2026, 9, 8, 8, tzinfo=UTC)
    broker = MarketBroker(get_market(MarketId.UK_LSE))
    store = ActivityShortlistStore(path)
    original = asyncio.run(
        ActivityShortlistService(store).get_or_create(
            broker,
            market=broker.market,
            cap_bucket=CapBucket.ALL,
            session=now.date(),
            screen_at=now,
            now=now,
        )
    )
    with sqlite3.connect(path) as database:
        database.execute(
            "ALTER TABLE activity_shortlist_candidates DROP COLUMN most_active_avg_usd_rank"
        )
    assert (
        ActivityShortlistStore.latest_read_only(
            path,
            market_id=MarketId.UK_LSE.value,
            cap_bucket=CapBucket.ALL,
        )
        == original
    )
    upgraded = ActivityShortlistStore(path)
    assert upgraded.get(MarketId.UK_LSE.value, CapBucket.ALL, now.date()) == original
    assert (
        ActivityShortlistStore(path).get(
            MarketId.UK_LSE.value,
            CapBucket.ALL,
            now.date(),
        )
        == original
    )
