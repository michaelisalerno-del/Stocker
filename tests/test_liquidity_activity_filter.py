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
        async def reqScannerParametersAsync(self):
            return (
                "<ScanParameterResponse>"
                f"<Location><locationCode>{market.scanner_location}</locationCode></Location>"
                "<ScanType><scanCode>MOST_ACTIVE_AVG_USD</scanCode></ScanType>"
                "</ScanParameterResponse>"
            )

        async def reqScannerDataAsync(self, *args):
            rows = await super().reqScannerDataAsync(*args)
            # Defensively exclude a contradictory fund classification even when the
            # server was asked for corporations. Real scan rows often omit this field.
            from copy import deepcopy
            fund = deepcopy(rows[0])
            fund.contractDetails.stockType = "ETF"
            fund.contractDetails.contract.symbol = "FUND"
            fund.contractDetails.contract.conId = 99
            fund.rank = 1
            return [*rows, fund]

    client = LiquidityClient()
    rows = asyncio.run(connection(client).activity_scan(
        market=market, cap_bucket=CapBucket.ALL,
        component=ActivityScanner.MOST_ACTIVE_AVG_USD, max_results=50,
        stock_type_filter="CORP",
    ))
    subscription = client.scanner_requests[0]
    assert subscription.stockTypeFilter == "CORP"
    assert subscription.scanCode == "MOST_ACTIVE_AVG_USD"
    assert subscription.locationCode == market.scanner_location
    assert subscription.instrument == market.scanner_instrument
    assert subscription.numberOfRows == 50
    assert [r.symbol for r in rows] == ["MSFT"]


def test_old_snapshot_remains_readable_before_and_after_schema_upgrade(tmp_path):
    path = tmp_path / "old.sqlite"
    now = datetime(2026, 9, 8, 8, tzinfo=UTC)
    broker = MarketBroker(get_market(MarketId.UK_LSE))
    store = ActivityShortlistStore(path)
    original = asyncio.run(ActivityShortlistService(store).get_or_create(
        broker, market=broker.market, cap_bucket=CapBucket.ALL, session=now.date(),
        screen_at=now, now=now,
    ))
    with sqlite3.connect(path) as database:
        database.execute(
            "ALTER TABLE activity_shortlist_candidates DROP COLUMN most_active_avg_usd_rank"
        )
    assert ActivityShortlistStore.latest_read_only(
        path, market_id=MarketId.UK_LSE.value, cap_bucket=CapBucket.ALL,
    ) == original
    upgraded = ActivityShortlistStore(path)
    assert upgraded.get(MarketId.UK_LSE.value, CapBucket.ALL, now.date()) == original
    assert ActivityShortlistStore(path).get(
        MarketId.UK_LSE.value, CapBucket.ALL, now.date(),
    ) == original
